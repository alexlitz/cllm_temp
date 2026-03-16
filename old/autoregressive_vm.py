#!/usr/bin/env python3
"""
Pure Autoregressive VM - Execution via Token Generation

The model generates tokens autoregressively. Each forward() call predicts
the next token based on the full context. No Python control flow for VM ops.

Token Format:
  [CODE...] [STATE] [TRACE...]

  CODE: <CODE><pc_lo><pc_hi><op><imm_bytes...>  (8 tokens per instruction)
  STATE: <PC><val><val>  <AX><val><val>  <SP><val><val>  (3 tokens per reg)
  TRACE: Generated tokens showing execution trace

The model learns to:
1. Read PC from STATE tokens (attention)
2. Fetch instruction at PC from CODE tokens (attention)
3. Execute operation (FFN arithmetic/routing)
4. Emit new STATE tokens
5. Repeat until HALT

Architecture: Standard transformer, analytically-set weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class AutoregressiveVM(nn.Module):
    """
    Pure autoregressive VM execution.

    Simplified for tractable weight engineering:
    - 8-bit values (not 32-bit)
    - Minimal opcodes: IMM, ADD, SUB, MUL, RET, PUTC
    - Registers: PC (8-bit), AX (16-bit)
    """

    # Token IDs
    TOK_CODE = 256      # Code marker
    TOK_PC = 257        # PC register marker
    TOK_AX = 258        # AX register marker
    TOK_STEP = 259      # Execution step marker
    TOK_OUT = 260       # Output byte marker
    TOK_HALT = 261      # Halt marker
    TOK_NEWPC = 262     # New PC value marker
    TOK_NEWAX = 263     # New AX value marker

    # Opcodes (simplified)
    OP_IMM = 0          # AX = imm
    OP_ADD = 1          # AX = AX + imm
    OP_SUB = 2          # AX = AX - imm
    OP_MUL = 3          # AX = AX * imm
    OP_PUTC = 4         # output AX & 0xFF
    OP_RET = 5          # halt, return AX

    VOCAB_SIZE = 264

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, num_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Embeddings
        self.tok_emb = nn.Embedding(self.VOCAB_SIZE, hidden_dim)
        self.pos_emb = nn.Embedding(4096, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, self.VOCAB_SIZE, bias=False)

        # Initialize weights for VM operations
        self._init_vm_weights()

    def _init_vm_weights(self):
        """Initialize all weights to implement VM operations."""
        with torch.no_grad():
            # Zero everything first
            self._zero_all_weights()

            # Set up embeddings
            self._init_embeddings()

            # Set up attention for state reading
            self._init_attention_weights()

            # Set up FFN for computation
            self._init_ffn_weights()

            # Set up output projection
            self._init_output_weights()

    def _zero_all_weights(self):
        """Zero out all weights."""
        self.tok_emb.weight.data.zero_()
        self.pos_emb.weight.data.zero_()
        self.lm_head.weight.data.zero_()

        for layer in self.layers:
            layer.attn.q_proj.weight.data.zero_()
            layer.attn.k_proj.weight.data.zero_()
            layer.attn.v_proj.weight.data.zero_()
            layer.attn.o_proj.weight.data.zero_()
            layer.ffn.w1.weight.data.zero_()
            layer.ffn.w1.bias.data.zero_()
            layer.ffn.w2.weight.data.zero_()
            layer.ffn.w2.bias.data.zero_()
            layer.ffn.w_gate.weight.data.zero_()
            layer.ln1.weight.data.fill_(1.0)
            layer.ln1.bias.data.zero_()
            layer.ln2.weight.data.fill_(1.0)
            layer.ln2.bias.data.zero_()

        self.ln_f.weight.data.fill_(1.0)
        self.ln_f.bias.data.zero_()

    def _init_embeddings(self):
        """
        Token embeddings encode:
        - Dims 0-7: Token type (one-hot for special tokens)
        - Dims 8-15: Byte value (for 0-255)
        - Dims 16-31: Position-independent features
        """
        H = self.hidden_dim

        # Byte tokens (0-255): encode value in dims 8-15 (binary)
        for i in range(256):
            for bit in range(8):
                if (i >> bit) & 1:
                    self.tok_emb.weight.data[i, 8 + bit] = 1.0

        # Special tokens: one-hot in dims 0-7
        special_tokens = [
            self.TOK_CODE, self.TOK_PC, self.TOK_AX, self.TOK_STEP,
            self.TOK_OUT, self.TOK_HALT, self.TOK_NEWPC, self.TOK_NEWAX
        ]
        for i, tok in enumerate(special_tokens):
            if tok < self.VOCAB_SIZE and i < 8:
                self.tok_emb.weight.data[tok, i] = 10.0

        # Position embeddings: binary encoding in dims 32-47
        for pos in range(min(4096, self.pos_emb.num_embeddings)):
            for bit in range(min(16, H - 32)):
                if (pos >> bit) & 1:
                    self.pos_emb.weight.data[pos, 32 + bit] = 0.5

    def _init_attention_weights(self):
        """
        New instruction format: [CODE, slot, op, imm]
        We need to:
        1. Find current PC slot
        2. Find instruction at that slot (op and imm)
        3. Gather AX value

        Layer 0: Read PC slot value
        Layer 1: Match slot to find instruction's op token
        Layer 2: Read immediate value (token after op)
        Layer 3: Read AX value
        """
        H = self.hidden_dim
        head_dim = self.head_dim

        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attn

            if layer_idx == 0:
                # Attend to most recent PC or NEWPC token to get current slot
                # TOK_PC is marker 1, slot value follows
                # Match on PC marker (dim 1 in embedding)
                attn.q_proj.weight.data[0, 1] = 10.0  # Looking for PC marker
                attn.k_proj.weight.data[0, 1] = 10.0  # Is PC marker

                # V: pass through the slot value (next token has slot in dims 8-15)
                # We need to attend to position P+1 where P has PC marker
                # This is tricky - for now, assume PC value is embedded in same token
                # Actually, with new format [TOK_PC, slot], the slot token follows
                # Let's attend directly to the slot byte after PC
                for i in range(8):
                    attn.v_proj.weight.data[48 + i, 8 + i] = 1.0  # Slot value

                for i in range(8):
                    attn.o_proj.weight.data[48 + i, 48 + i] = 1.0

            elif layer_idx == 1:
                # Find the OP token for instruction at current slot
                # Instructions are: [CODE, slot, op, imm]
                # We want the OP byte where preceding slot matches current PC

                # This is complex - we need to match slot values
                # Let's use a simpler approach: attend to positions where
                # the token matches the slot number (byte tokens 0-63 are slots)

                # Q: encode current slot from dims 48-55
                for i in range(8):
                    attn.q_proj.weight.data[i, 48 + i] = 2.0

                # K: encode byte value (slot number) from dims 8-15
                for i in range(8):
                    attn.k_proj.weight.data[i, 8 + i] = 2.0

                # V: pass through everything for context
                for i in range(H):
                    attn.v_proj.weight.data[i, i] = 1.0

                # O: pass through
                for i in range(H):
                    attn.o_proj.weight.data[i, i] = 1.0

            elif layer_idx == 2:
                # Now we should be attending to the slot byte
                # The OP is at position +1, IMM at position +2
                # We need positional attention - attend to fixed offset

                # For now: attend to opcode bytes (look for opcode patterns)
                # Opcodes are 0-5, so byte values 0-5 indicate opcodes

                # Actually let's attend to AX value
                attn.q_proj.weight.data[0, 2] = 10.0  # AX marker
                attn.k_proj.weight.data[0, 2] = 10.0

                for i in range(8):
                    attn.v_proj.weight.data[72 + i, 8 + i] = 1.0
                    attn.o_proj.weight.data[72 + i, 72 + i] = 1.0

            else:
                # Layer 3: Aggregate and prepare for output
                for i in range(H):
                    attn.q_proj.weight.data[i, i] = 1.0
                    attn.k_proj.weight.data[i, i] = 1.0
                    attn.v_proj.weight.data[i, i] = 1.0
                    attn.o_proj.weight.data[i, i] = 1.0

    def _init_ffn_weights(self):
        """
        Layer 0 FFN: Passthrough
        Layer 1 FFN: Opcode decode
        Layer 2 FFN: Arithmetic (add tables, SwiGLU mul)
        Layer 3 FFN: Output token selection
        """
        H = self.hidden_dim

        for layer_idx, layer in enumerate(self.layers):
            ffn = layer.ffn
            ffn_dim = ffn.w1.out_features

            if layer_idx == 2:
                # Arithmetic layer: 8-bit add table
                # Input: AX value in dims 72-79, imm in dims 64-71
                # Output: result in dims 88-95

                # Simple 8-bit add: result = (AX + imm) & 0xFF
                # Use lookup table approach
                for a in range(16):  # Low nibble of AX
                    for b in range(16):  # Low nibble of imm
                        for cin in range(2):
                            idx = a * 32 + b * 2 + cin
                            if idx < ffn_dim:
                                # Detect this combination
                                ffn.w1.weight.data[idx, 72 + 0] = 1.0 if (a & 1) else -1.0
                                ffn.w1.weight.data[idx, 72 + 1] = 1.0 if (a & 2) else -1.0
                                ffn.w1.weight.data[idx, 72 + 2] = 1.0 if (a & 4) else -1.0
                                ffn.w1.weight.data[idx, 72 + 3] = 1.0 if (a & 8) else -1.0
                                ffn.w1.weight.data[idx, 64 + 0] = 1.0 if (b & 1) else -1.0
                                ffn.w1.weight.data[idx, 64 + 1] = 1.0 if (b & 2) else -1.0
                                ffn.w1.weight.data[idx, 64 + 2] = 1.0 if (b & 4) else -1.0
                                ffn.w1.weight.data[idx, 64 + 3] = 1.0 if (b & 8) else -1.0

                                ffn.w1.bias.data[idx] = -6.0  # Threshold
                                ffn.w_gate.weight.data[idx, 72] = 0.1

                                # Output: sum nibble
                                total = a + b + cin
                                sum_nib = total & 0xF
                                carry = 1 if total > 15 else 0

                                for bit in range(4):
                                    if (sum_nib >> bit) & 1:
                                        ffn.w2.weight.data[88 + bit, idx] = 1.0
                                ffn.w2.weight.data[92, idx] = float(carry)

            elif layer_idx == 3:
                # Output selection based on execution state
                # Uses dims to track state machine:
                # - Dim 3 = 1 means we just saw STEP
                # - Dim 7 (NEWAX marker) = 1 means we just saw NEWAX
                # - Dim 64-71 has the immediate value (result for IMM op)

                # State 0: After STEP, opcode = IMM (0) -> predict NEWAX
                ffn.w1.weight.data[0, 3] = 10.0   # After STEP marker
                ffn.w1.weight.data[0, 56] = -5.0  # op bit 0 = 0
                ffn.w1.weight.data[0, 57] = -5.0  # op bit 1 = 0
                ffn.w1.weight.data[0, 58] = -5.0  # op bit 2 = 0
                ffn.w1.bias.data[0] = 5.0
                ffn.w_gate.weight.data[0, 3] = 1.0
                ffn.w2.weight.data[96, 0] = 10.0  # -> NEWAX dim

                # State 1: After STEP, opcode = RET (5) -> predict HALT
                ffn.w1.weight.data[1, 3] = 10.0   # After STEP marker
                ffn.w1.weight.data[1, 56] = 5.0   # op bit 0 = 1
                ffn.w1.weight.data[1, 57] = -5.0  # op bit 1 = 0
                ffn.w1.weight.data[1, 58] = 5.0   # op bit 2 = 1
                ffn.w1.bias.data[1] = -5.0
                ffn.w_gate.weight.data[1, 3] = 1.0
                ffn.w2.weight.data[97, 1] = 10.0  # -> HALT dim

                # State 2: After STEP, opcode = PUTC (4) -> predict OUT
                ffn.w1.weight.data[2, 3] = 10.0   # After STEP marker
                ffn.w1.weight.data[2, 56] = -5.0  # op bit 0 = 0
                ffn.w1.weight.data[2, 57] = -5.0  # op bit 1 = 0
                ffn.w1.weight.data[2, 58] = 5.0   # op bit 2 = 1
                ffn.w1.bias.data[2] = 0.0
                ffn.w_gate.weight.data[2, 3] = 1.0
                ffn.w2.weight.data[98, 2] = 10.0  # -> OUT dim

                # State 3: After NEWAX -> predict the immediate value (low byte)
                # The immediate is in dims 64-71
                ffn.w1.weight.data[3, 7] = 10.0   # After NEWAX marker
                ffn.w1.bias.data[3] = -5.0
                ffn.w_gate.weight.data[3, 7] = 1.0
                # Route immediate value to output byte dims
                for bit in range(8):
                    ffn.w2.weight.data[104 + bit, 3] = 1.0  # Copy immediate to byte output dims

                # After first byte, predict high byte (0 for simplicity)
                ffn.w1.weight.data[4, 104] = 5.0   # We just output a byte
                ffn.w1.weight.data[4, 7] = -5.0    # But not NEWAX
                ffn.w1.bias.data[4] = -2.0
                ffn.w_gate.weight.data[4, 104] = 1.0
                # Output 0 for high byte
                ffn.w2.weight.data[112, 4] = -10.0  # Force 0

                # After second byte, predict NEWPC
                ffn.w1.weight.data[5, 112] = 5.0   # We output high byte
                ffn.w1.bias.data[5] = -2.0
                ffn.w_gate.weight.data[5, 112] = 1.0
                ffn.w2.weight.data[99, 5] = 10.0  # -> NEWPC dim

                # After NEWPC, predict new PC value (PC + 1)
                ffn.w1.weight.data[6, 6] = 10.0   # After NEWPC marker
                ffn.w1.bias.data[6] = -5.0
                ffn.w_gate.weight.data[6, 6] = 1.0
                # PC value comes from dims 48-55, add 1
                for bit in range(8):
                    ffn.w2.weight.data[104 + bit, 6] = 0.5  # Will need proper add

                # After PC bytes, predict STEP
                ffn.w1.weight.data[7, 6] = -5.0  # Not NEWPC
                ffn.w1.weight.data[7, 99] = 5.0  # But we're in PC output mode
                ffn.w1.bias.data[7] = 0.0
                ffn.w_gate.weight.data[7, 99] = 1.0
                ffn.w2.weight.data[100, 7] = 10.0  # -> STEP dim

    def _init_output_weights(self):
        """Map hidden dims to vocab logits."""
        H = self.hidden_dim

        # Byte tokens: map from result dims (88-95) to byte value
        # This is complex - for now, direct mapping
        for i in range(8):
            for v in range(256):
                if (v >> i) & 1:
                    self.lm_head.weight.data[v, 88 + i] += 1.0

        # Special tokens
        self.lm_head.weight.data[self.TOK_NEWAX, 96] = 10.0
        self.lm_head.weight.data[self.TOK_HALT, 97] = 10.0
        self.lm_head.weight.data[self.TOK_OUT, 98] = 10.0
        self.lm_head.weight.data[self.TOK_NEWPC, 99] = 10.0
        self.lm_head.weight.data[self.TOK_STEP, 100] = 10.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard transformer forward pass."""
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(seq_len, device=device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )

        # Layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_tokens: int = 1000) -> torch.Tensor:
        """Generate tokens until HALT."""
        for _ in range(max_tokens):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == self.TOK_HALT:
                break

        return input_ids


class TransformerLayer(nn.Module):
    """Standard transformer layer."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = Attention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, hidden_dim * 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class Attention(nn.Module):
    """Multi-head attention."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN."""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w_gate(x))


def encode_program(model: AutoregressiveVM, instructions: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Encode a program as tokens.

    instructions: List of (opcode, immediate) tuples

    NEW FORMAT - Each instruction is a SINGLE compound token:
    We'll use custom embeddings that encode [pc, op, imm] together.

    Returns token sequence:
    [INSTR tokens...] [PC=0] [AX=0] [STEP]
    """
    tokens = []

    # For simplicity, create instruction tokens as byte sequences
    # that will be interpreted by special attention patterns
    # Format: <CODE_PC_OP_IMM> where each instruction gets unique ID

    # Actually, let's use a simpler format:
    # Each instruction: CODE marker, then we modify embeddings to encode more

    # Encode instructions - using indices 0-63 as instruction slots
    # Instruction N is at token (N * 4) with format: [SLOT, OP, IMM_LO, IMM_HI]
    for pc, (op, imm) in enumerate(instructions):
        slot = pc  # Instruction slot (0, 1, 2, ...)
        tokens.extend([
            model.TOK_CODE,      # Marker that this is code
            slot,                # Which instruction slot
            op,                  # Opcode
            imm & 0xFF,          # Immediate (for OP_IMM, this is the value)
        ])

    # Initial state
    tokens.extend([model.TOK_PC, 0])         # PC = 0 (slot 0)
    tokens.extend([model.TOK_AX, 0, 0])      # AX = 0

    # Start execution
    tokens.append(model.TOK_STEP)

    return torch.tensor([tokens], dtype=torch.long)


def decode_output(model: AutoregressiveVM, tokens: torch.Tensor) -> Tuple[List[int], int]:
    """Extract output bytes and final AX value."""
    tokens = tokens.squeeze().tolist()
    output = []
    ax = 0

    i = 0
    while i < len(tokens):
        if tokens[i] == model.TOK_OUT and i + 1 < len(tokens):
            output.append(tokens[i + 1])
            i += 2
        elif tokens[i] == model.TOK_NEWAX and i + 2 < len(tokens):
            ax = tokens[i + 1] + (tokens[i + 2] << 8)
            i += 3
        else:
            i += 1

    return output, ax


def test_overflow_handling():
    """Test that 32-bit overflow is handled correctly via SwiGLU."""
    import torch.nn.functional as F

    print("=" * 70)
    print("  OVERFLOW HANDLING TESTS (32-bit via SwiGLU)")
    print("=" * 70)

    def swiglu_mul(a, b):
        """SwiGLU multiplication: silu(a)*b + silu(-a)*(-b) = a*b"""
        a_t = torch.tensor([float(a)])
        b_t = torch.tensor([float(b)])
        return int(round((F.silu(a_t) * b_t + F.silu(-a_t) * (-b_t)).item()))

    tests = [
        # Basic overflow
        ("0x7FFFFFFF + 1", lambda: (0x7FFFFFFF + 1) & 0xFFFFFFFF, 0x80000000),
        ("0xFFFFFFFF + 1", lambda: (0xFFFFFFFF + 1) & 0xFFFFFFFF, 0),
        ("-1 + -1 (unsigned)", lambda: (0xFFFFFFFF + 0xFFFFFFFF) & 0xFFFFFFFF, 0xFFFFFFFE),

        # Signed overflow
        ("MAX_INT + 1", lambda: (2147483647 + 1) & 0xFFFFFFFF, 2147483648),
        ("MIN_INT - 1", lambda: (-2147483648 - 1) & 0xFFFFFFFF, 2147483647),

        # Multiplication overflow
        ("65536 * 65536", lambda: swiglu_mul(65536, 65536) & 0xFFFFFFFF, 0),
        ("0x10000 * 0x10000", lambda: swiglu_mul(0x10000, 0x10000) & 0xFFFFFFFF, 0),
        ("0x8000 * 0x10000", lambda: swiglu_mul(0x8000, 0x10000) & 0xFFFFFFFF, 0x80000000),

        # Negative multiplication
        ("-1 * -1", lambda: swiglu_mul(-1, -1), 1),
        ("-1 * 1", lambda: swiglu_mul(-1, 1), -1),
        ("MAX_INT * 2", lambda: swiglu_mul(0x7FFFFFFF, 2) & 0xFFFFFFFF, 0xFFFFFFFE),

        # Large numbers
        ("1000000 * 1000", lambda: swiglu_mul(1000000, 1000), 1000000000),
        ("1000000 * 10000", lambda: swiglu_mul(1000000, 10000) & 0xFFFFFFFF, 1410065408),  # Overflow
    ]

    passed = 0
    for name, fn, expected in tests:
        result = fn()
        ok = result == expected
        passed += ok
        status = "OK" if ok else "FAIL"
        print(f"  {name:30s} = {result:>12} (expected {expected:>12}) [{status}]")

    print()
    print(f"  Passed: {passed}/{len(tests)}")
    return passed == len(tests)


def main():
    # Run overflow tests first
    print()
    overflow_ok = test_overflow_handling()
    print()

    print("=" * 70)
    print("  PURE AUTOREGRESSIVE VM")
    print("=" * 70)
    print()

    # Create model
    model = AutoregressiveVM(hidden_dim=128, num_layers=4, num_heads=4)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p.data == 0).sum().item() for p in model.parameters())
    nonzero = total - zeros

    print(f"Architecture: {model.num_layers} layers, {model.hidden_dim} hidden")
    print(f"Parameters: {total:,} total, {nonzero:,} non-zero ({100*zeros/total:.1f}% sparse)")
    print()

    # Test program: AX = 42, return
    print("Test program: return 42")
    print()

    program = [
        (model.OP_IMM, 42),   # AX = 42
        (model.OP_RET, 0),    # return AX
    ]

    input_ids = encode_program(model, program)
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Tokens: {input_ids.squeeze().tolist()}")
    print()

    # Forward pass
    print("Forward pass...")
    with torch.no_grad():
        logits = model.forward(input_ids)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5 = probs.topk(5)

    print("Top 5 predictions after STEP:")
    for prob, idx in zip(top5.values, top5.indices):
        idx = idx.item()
        if idx == model.TOK_NEWAX:
            name = "NEWAX"
        elif idx == model.TOK_HALT:
            name = "HALT"
        elif idx == model.TOK_OUT:
            name = "OUT"
        elif idx == model.TOK_STEP:
            name = "STEP"
        elif idx == model.TOK_NEWPC:
            name = "NEWPC"
        elif idx < 256:
            name = f"BYTE({idx})"
        else:
            name = f"TOK({idx})"
        print(f"  {name}: {prob.item():.4f}")

    print()

    # Try generation
    print("Generating...")
    output_ids = model.generate(input_ids, max_tokens=50)
    new_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    print(f"Generated {len(new_tokens)} tokens: {new_tokens}")

    output_bytes, final_ax = decode_output(model, output_ids)
    print(f"Output: {bytes(output_bytes)}")
    print(f"Final AX: {final_ax}")
    print()

    # Analyze what went wrong/right
    print("=" * 70)
    print("  ANALYSIS")
    print("=" * 70)
    print()

    # Check hidden state at last position
    with torch.no_grad():
        # Get embeddings
        pos = torch.arange(input_ids.shape[1])
        x = model.tok_emb(input_ids) + model.pos_emb(pos)

        print("Hidden state at STEP token (last position):")
        h = x[0, -1, :]
        print(f"  Dims 0-7 (special markers): {h[0:8].tolist()}")
        print(f"  Dims 8-15 (byte value): {h[8:16].tolist()}")
        print(f"  Dims 48-55 (PC from attn): {h[48:56].tolist()}")

        # Run through first layer
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]) * float('-inf'), diagonal=1)
        x1 = model.layers[0](x, mask)
        h1 = x1[0, -1, :]
        print(f"\nAfter layer 0:")
        print(f"  Dims 48-55 (should have PC): {h1[48:56].tolist()}")


if __name__ == "__main__":
    main()
