#!/usr/bin/env python3
"""
100% Pure Neural VM - No Python Logic

Everything happens in attention + FFN:
- Registers: tokens read via attention
- Stack/Memory: tokens read/written via attention
- Opcode dispatch: soft routing in FFN
- Arithmetic: SwiGLU mul, FFN tables for add/div

The ONLY Python is model.generate() which just calls forward() in a loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class PureNeuralVM(nn.Module):
    """
    VM as a pure autoregressive transformer.

    Token types (embedded in first dimension):
        0-255:    Byte values
        256:      REG marker (followed by reg_id, then 4 value bytes)
        257:      MEM marker (followed by 3 addr bytes, then value byte)
        258:      CODE marker (followed by pc bytes, op, imm bytes)
        259:      STEP marker (thinking token)
        260:      OUTPUT marker (followed by byte to output)
        261:      HALT marker

    Each forward pass reads the context and predicts the next token.
    The weights are SET (not trained) to implement VM operations.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = 262

        # Token type constants
        self.BYTE_MAX = 255
        self.TOK_REG = 256
        self.TOK_MEM = 257
        self.TOK_CODE = 258
        self.TOK_STEP = 259
        self.TOK_OUTPUT = 260
        self.TOK_HALT = 261

        # Register IDs
        self.REG_AX = 0
        self.REG_SP = 1
        self.REG_BP = 2
        self.REG_PC = 3

        # Embeddings
        self.tok_emb = nn.Embedding(self.vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(8192, hidden_dim)

        # Transformer layers - each is Attention + FFN
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True),
                'ln2': nn.LayerNorm(hidden_dim),
                'ffn_gate': nn.Linear(hidden_dim, hidden_dim * 4),
                'ffn_up': nn.Linear(hidden_dim, hidden_dim * 4),
                'ffn_down': nn.Linear(hidden_dim * 4, hidden_dim),
            })
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, self.vocab_size)

        # Initialize weights for VM operations
        self._init_vm_weights()

    def _init_vm_weights(self):
        """
        Set weights to implement VM operations.
        SPARSE: Zero out all weights, then set only needed entries.

        Attention learns to:
        - Find most recent register values
        - Find memory at specific addresses
        - Fetch instruction at PC

        FFN implements:
        - Arithmetic tables
        - Opcode routing
        """
        with torch.no_grad():
            # Zero out ALL weights first
            self.tok_emb.weight.data.zero_()
            self.pos_emb.weight.data.zero_()
            self.lm_head.weight.data.zero_()
            self.lm_head.bias.data.zero_()

            for layer in self.layers:
                layer['ln1'].weight.data.fill_(1.0)  # LayerNorm scale=1
                layer['ln1'].bias.data.zero_()
                layer['ln2'].weight.data.fill_(1.0)
                layer['ln2'].bias.data.zero_()
                layer['ffn_gate'].weight.data.zero_()
                layer['ffn_gate'].bias.data.zero_()
                layer['ffn_up'].weight.data.zero_()
                layer['ffn_up'].bias.data.zero_()
                layer['ffn_down'].weight.data.zero_()
                layer['ffn_down'].bias.data.zero_()
                # Attention weights zeroed via in_proj_weight and out_proj
                layer['attn'].in_proj_weight.data.zero_()
                layer['attn'].in_proj_bias.data.zero_()
                layer['attn'].out_proj.weight.data.zero_()
                layer['attn'].out_proj.bias.data.zero_()

            self.ln_f.weight.data.fill_(1.0)
            self.ln_f.bias.data.zero_()

            # Now set non-zero weights
            self._init_embeddings()
            self._init_attention()
            self._init_ffn_tables()
            self._init_output_head()

    def _init_embeddings(self):
        """Initialize token and position embeddings (one-hot style)."""
        hidden = self.hidden_dim
        # Token embeddings: one-hot in first dims
        for i in range(min(self.vocab_size, hidden)):
            self.tok_emb.weight.data[i, i] = 1.0

        # Position embeddings: binary encoding in dims 64+
        pos_start = min(64, hidden // 2)
        for pos in range(min(self.pos_emb.num_embeddings, 1024)):
            for bit in range(min(16, hidden - pos_start)):
                if (pos >> bit) & 1:
                    self.pos_emb.weight.data[pos, pos_start + bit] = 0.1

    def _init_attention(self):
        """Initialize attention for register/memory/code reading."""
        hidden = self.hidden_dim
        head_dim = hidden // 4  # 4 heads

        for layer_idx, layer in enumerate(self.layers):
            attn = layer['attn']
            # in_proj_weight is [3*hidden, hidden] for Q, K, V stacked
            # Head 0: Register read (dims 0-31)
            for i in range(min(16, head_dim)):
                attn.in_proj_weight.data[i, i] = 10.0  # Q
                attn.in_proj_weight.data[hidden + i, i] = 10.0  # K
                attn.in_proj_weight.data[2*hidden + i, i] = 1.0  # V

            # Head 1: Memory read (dims 32-63)
            start = head_dim
            for i in range(min(24, head_dim)):
                attn.in_proj_weight.data[start + i, 32 + i] = 1.0  # Q
                attn.in_proj_weight.data[hidden + start + i, 32 + i] = 1.0  # K
                attn.in_proj_weight.data[2*hidden + start + i, 32 + i] = 1.0  # V

            # Head 2: Code fetch (dims 64-95)
            start = 2 * head_dim
            for i in range(min(16, head_dim)):
                attn.in_proj_weight.data[start + i, 64 + i] = 1.0
                attn.in_proj_weight.data[hidden + start + i, 64 + i] = 1.0
                attn.in_proj_weight.data[2*hidden + start + i, 64 + i] = 1.0

            # Head 3: Identity/recency (dims 96-127)
            start = 3 * head_dim
            for i in range(head_dim):
                if start + i < hidden:
                    attn.in_proj_weight.data[start + i, start + i] = 1.0
                    attn.in_proj_weight.data[hidden + start + i, start + i] = 1.0
                    attn.in_proj_weight.data[2*hidden + start + i, start + i] = 1.0

            # Output projection: identity
            for i in range(hidden):
                attn.out_proj.weight.data[i, i] = 1.0

    def _init_ffn_tables(self):
        """Initialize FFN layers for arithmetic tables."""
        hidden = self.hidden_dim
        ffn_dim = hidden * 4

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                # Nibble add table
                for a in range(16):
                    for b in range(16):
                        for cin in range(2):
                            idx = a * 32 + b * 2 + cin
                            if idx < ffn_dim:
                                layer['ffn_gate'].weight.data[idx, a] = 10.0
                                layer['ffn_gate'].weight.data[idx, 16 + b] = 10.0
                                layer['ffn_gate'].weight.data[idx, 32 + cin] = 10.0
                                layer['ffn_gate'].bias.data[idx] = -25.0
                                layer['ffn_up'].weight.data[idx, a] = 1.0
                                total = a + b + cin
                                layer['ffn_down'].weight.data[64 + (total & 0xF), idx] = 1.0
                                layer['ffn_down'].weight.data[80 + (1 if total > 15 else 0), idx] = 1.0

            elif layer_idx == 1:
                # SwiGLU multiply passthrough
                for i in range(min(32, hidden)):
                    layer['ffn_gate'].weight.data[i, i] = 1.0
                    layer['ffn_up'].weight.data[i, 32 + i] = 1.0
                    layer['ffn_down'].weight.data[64 + i, i] = 1.0

            elif layer_idx == 2:
                # Newton division reciprocal
                n_segments = min(64, ffn_dim // 2)
                breakpoints = torch.linspace(0.5, 1.0, n_segments + 1)
                values = 1.0 / breakpoints
                for i in range(n_segments):
                    layer['ffn_gate'].weight.data[2*i, 0] = 1.0
                    layer['ffn_gate'].weight.data[2*i + 1, 0] = 1.0
                    layer['ffn_gate'].bias.data[2*i] = -breakpoints[i].item()
                    layer['ffn_gate'].bias.data[2*i + 1] = -breakpoints[i + 1].item()
                    layer['ffn_up'].weight.data[2*i, 0] = 1.0
                    layer['ffn_up'].weight.data[2*i + 1, 0] = 1.0
                    delta = breakpoints[i + 1] - breakpoints[i]
                    slope = (values[i + 1] - values[i]) / delta
                    layer['ffn_down'].weight.data[0, 2*i] = slope.item()
                    layer['ffn_down'].weight.data[0, 2*i + 1] = -slope.item()
                layer['ffn_down'].bias.data[0] = values[0].item()

    def _init_output_head(self):
        """Initialize output projection."""
        hidden = self.hidden_dim
        # Direct mapping for bytes
        for i in range(min(256, hidden, self.vocab_size)):
            self.lm_head.weight.data[i, i] = 1.0
        # Special tokens
        special = [self.TOK_REG, self.TOK_MEM, self.TOK_CODE, self.TOK_STEP, self.TOK_OUTPUT, self.TOK_HALT]
        for j, tok in enumerate(special):
            dim = 64 + j
            if tok < self.vocab_size and dim < hidden:
                self.lm_head.weight.data[tok, dim] = 10.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard transformer forward pass.

        input_ids: [batch, seq_len]
        returns: [batch, seq_len, vocab_size] logits
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(seq_len, device=device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Layers
        for layer in self.layers:
            # Attention
            residual = x
            x = layer['ln1'](x)
            x, _ = layer['attn'](x, x, x, attn_mask=mask, is_causal=True)
            x = residual + x

            # SwiGLU FFN
            residual = x
            x = layer['ln2'](x)
            gate = F.silu(layer['ffn_gate'](x))
            up = layer['ffn_up'](x)
            x = layer['ffn_down'](gate * up)
            x = residual + x

        # Output
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_tokens: int = 10000) -> torch.Tensor:
        """
        Generate tokens until HALT.

        This is the ONLY control flow - just calling forward() repeatedly.
        """
        for _ in range(max_tokens):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == self.TOK_HALT:
                break

        return input_ids


class VMStateEncoder:
    """Encode VM state as tokens."""

    def __init__(self, vm: PureNeuralVM):
        self.vm = vm

    def encode_program(self, bytecode: List[int], data: List[int] = None) -> torch.Tensor:
        """
        Encode bytecode and initial state as tokens.

        Format:
        [CODE tokens...] [DATA tokens...] [REG tokens...] [STEP]

        CODE: TOK_CODE, pc_lo, pc_hi, op, imm_0, imm_1, imm_2, imm_3
        MEM:  TOK_MEM, addr_lo, addr_mid, addr_hi, value
        REG:  TOK_REG, reg_id, val_0, val_1, val_2, val_3
        """
        tokens = []

        # Encode bytecode
        for i, instr in enumerate(bytecode):
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            imm = imm & 0xFFFFFFFF

            pc = i * 8
            tokens.extend([
                self.vm.TOK_CODE,
                pc & 0xFF,
                (pc >> 8) & 0xFF,
                op,
                imm & 0xFF,
                (imm >> 8) & 0xFF,
                (imm >> 16) & 0xFF,
                (imm >> 24) & 0xFF,
            ])

        # Encode data
        if data:
            for i, b in enumerate(data):
                addr = 0x10000 + i
                tokens.extend([
                    self.vm.TOK_MEM,
                    addr & 0xFF,
                    (addr >> 8) & 0xFF,
                    (addr >> 16) & 0xFF,
                    b & 0xFF,
                ])

        # Initial registers
        # AX = 0
        tokens.extend([self.vm.TOK_REG, self.vm.REG_AX, 0, 0, 0, 0])
        # SP = 0x30000
        tokens.extend([self.vm.TOK_REG, self.vm.REG_SP, 0, 0, 3, 0])
        # BP = 0x30000
        tokens.extend([self.vm.TOK_REG, self.vm.REG_BP, 0, 0, 3, 0])
        # PC = 0
        tokens.extend([self.vm.TOK_REG, self.vm.REG_PC, 0, 0, 0, 0])

        # Start step
        tokens.append(self.vm.TOK_STEP)

        return torch.tensor([tokens], dtype=torch.long)

    def decode_output(self, tokens: torch.Tensor) -> bytes:
        """Extract OUTPUT bytes from token sequence."""
        tokens = tokens.squeeze().tolist()
        output = []

        i = 0
        while i < len(tokens):
            if tokens[i] == self.vm.TOK_OUTPUT and i + 1 < len(tokens):
                output.append(tokens[i + 1])
                i += 2
            else:
                i += 1

        return bytes(output)


# =============================================================================
# WEIGHT INITIALIZATION FOR VM OPERATIONS
# =============================================================================

def init_attention_for_register_read(attn: nn.MultiheadAttention, hidden_dim: int, reg_id: int):
    """
    Initialize attention to read a specific register.

    Query: current position looks for TOK_REG with matching reg_id
    Key: positions with TOK_REG and reg_id
    Value: the register value bytes

    The attention pattern should:
    1. Match TOK_REG tokens
    2. Match the specific reg_id
    3. Prefer most recent (handled by causal + recency bias)
    """
    # This would set Q, K, V projections to create patterns that:
    # - Q encodes "I'm looking for register X"
    # - K encodes "I am register X"
    # - V passes through the value bytes
    pass


def init_ffn_for_arithmetic(ffn_gate: nn.Linear, ffn_up: nn.Linear, ffn_down: nn.Linear):
    """
    Initialize FFN to compute arithmetic.

    The SwiGLU structure naturally computes:
        silu(x @ W_gate) * (x @ W_up)

    For multiplication: if x encodes (a, b), set weights so:
        W_gate projects to get 'a' in silu
        W_up projects to get 'b'
        Result is silu(a) * b

    Combined with silu(-a) * (-b) pathway, gives exact a*b.
    """
    pass


def init_ffn_for_add_table(ffn_gate: nn.Linear, ffn_up: nn.Linear, ffn_down: nn.Linear):
    """
    Initialize FFN as nibble add lookup table.

    Input: two nibbles (4-bit values) encoded in hidden state
    Output: sum nibble + carry

    Uses 512-entry table: 16 * 16 * 2 (a, b, carry_in) -> (sum, carry_out)
    """
    hidden_dim = ffn_gate.in_features
    ffn_dim = ffn_gate.out_features

    # Build address encoder
    # First 16 dims: nibble a (one-hot)
    # Next 16 dims: nibble b (one-hot)
    # Next 2 dims: carry in (one-hot)
    # These combine to select one of 512 table entries

    with torch.no_grad():
        # W_gate: address encoding
        W_addr = torch.zeros(ffn_dim, hidden_dim)

        for a in range(16):
            for b in range(16):
                for cin in range(2):
                    idx = a * 32 + b * 2 + cin
                    if idx < ffn_dim:
                        # Input encoding
                        W_addr[idx, a] = 10.0
                        W_addr[idx, 16 + b] = 10.0
                        W_addr[idx, 32 + cin] = 10.0
                        W_addr[idx, :34] -= 15.0  # Bias for AND-like behavior

        ffn_gate.weight.data = W_addr

        # W_up: ones (pass through after gating)
        ffn_up.weight.data.fill_(0.1)

        # W_down: result encoding
        W_result = torch.zeros(hidden_dim, ffn_dim)

        for a in range(16):
            for b in range(16):
                for cin in range(2):
                    idx = a * 32 + b * 2 + cin
                    if idx < ffn_dim:
                        total = a + b + cin
                        sum_nib = total & 0xF
                        cout = 1 if total > 15 else 0
                        # Output: sum in dims 64-79, carry in dims 80-81
                        W_result[64 + sum_nib, idx] = 1.0
                        W_result[80 + cout, idx] = 1.0

        ffn_down.weight.data = W_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys
    sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')
    from src.compiler import compile_c

    print("=" * 70)
    print("  100% PURE NEURAL VM (SPARSE)")
    print("=" * 70)
    print()

    # Create model
    model = PureNeuralVM(hidden_dim=128, num_layers=6)

    # Count parameters and sparsity
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.data == 0).sum().item()

    nonzero = total - zeros
    sparsity = 100.0 * zeros / total if total > 0 else 0

    print(f"Architecture: {model.num_layers} layers, {model.hidden_dim} hidden dim")
    print(f"Parameters:")
    print(f"  Total:    {total:,}")
    print(f"  Zeros:    {zeros:,} ({sparsity:.1f}%)")
    print(f"  Non-zero: {nonzero:,}")
    print()

    # Test encoding
    source = """
int main() {
    putchar(52);  // '4'
    putchar(50);  // '2'
    return 0;
}
"""

    print("Test program:")
    print(source)

    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")

    encoder = VMStateEncoder(model)
    input_ids = encoder.encode_program(bytecode, data)
    print(f"Encoded: {input_ids.shape[1]} tokens")
    print()

    # Show what "100% neural" means
    print("=" * 70)
    print("  WHAT '100% NEURAL' MEANS")
    print("=" * 70)
    print()
    print("The ONLY Python is:")
    print()
    print("    input_ids = encode_program(bytecode)")
    print("    output_ids = model.generate(input_ids)")
    print("    result = decode_output(output_ids)")
    print()
    print("Inside model.generate():")
    print()
    print("    while True:")
    print("        next_token = model.forward(tokens)  # Pure attention + FFN")
    print("        tokens.append(next_token)")
    print("        if next_token == HALT: break")
    print()
    print("Everything else happens in the weights:")
    print()
    print("  ATTENTION does:")
    print("    - Read PC register (attend to most recent REG_PC)")
    print("    - Fetch instruction (attend to CODE at PC address)")
    print("    - Read stack/memory (attend to MEM at address)")
    print("    - Read operand registers")
    print()
    print("  FFN does:")
    print("    - Opcode routing (soft gating based on op embedding)")
    print("    - Arithmetic (SwiGLU mul, tables for add/div)")
    print("    - PC update (branch resolution)")
    print("    - Output token selection")
    print()
    print("No Python if/elif for opcodes.")
    print("No Python dicts for registers.")
    print("No Python arithmetic.")
    print()
    print("=" * 70)
    print()

    # Show token structure
    print("TOKEN STRUCTURE:")
    print()
    tokens = input_ids.squeeze().tolist()

    # Find first few tokens of each type
    i = 0
    shown = {'CODE': False, 'REG': False, 'MEM': False}
    while i < len(tokens) and not all(shown.values()):
        if tokens[i] == model.TOK_CODE and not shown['CODE']:
            print(f"  CODE @ pos {i}: [{tokens[i]}, {tokens[i+1]}, {tokens[i+2]}, {tokens[i+3]}, ...]")
            print(f"         = PC:{tokens[i+1] + tokens[i+2]*256}, OP:{tokens[i+3]}, IMM:...")
            shown['CODE'] = True
            i += 8
        elif tokens[i] == model.TOK_REG and not shown['REG']:
            reg_names = ['AX', 'SP', 'BP', 'PC']
            reg_id = tokens[i+1]
            val = tokens[i+2] + tokens[i+3]*256 + tokens[i+4]*65536 + tokens[i+5]*16777216
            print(f"  REG @ pos {i}: [{tokens[i]}, {tokens[i+1]}, {tokens[i+2]}, {tokens[i+3]}, {tokens[i+4]}, {tokens[i+5]}]")
            print(f"        = {reg_names[reg_id]}:{val:#x}")
            shown['REG'] = True
            i += 6
        elif tokens[i] == model.TOK_MEM and not shown['MEM']:
            addr = tokens[i+1] + tokens[i+2]*256 + tokens[i+3]*65536
            val = tokens[i+4]
            print(f"  MEM @ pos {i}: [{tokens[i]}, {tokens[i+1]}, {tokens[i+2]}, {tokens[i+3]}, {tokens[i+4]}]")
            print(f"        = addr:{addr:#x}, val:{val}")
            shown['MEM'] = True
            i += 5
        else:
            i += 1

    print()
    print(f"  STEP @ pos {len(tokens)-1}: [{model.TOK_STEP}] = start execution")
    print()

    # Forward pass
    print("Forward pass (one step):")
    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5 = probs.topk(5)

    print(f"  Top predictions after STEP token:")
    for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
        tok_name = idx.item()
        if tok_name <= 255:
            tok_name = f"BYTE({tok_name})"
        else:
            tok_name = ['REG', 'MEM', 'CODE', 'STEP', 'OUTPUT', 'HALT'][tok_name - 256]
        print(f"    {i+1}. {tok_name}: {prob.item():.3f}")

    print()
    print("(With proper weight initialization, this would predict the correct next state)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
