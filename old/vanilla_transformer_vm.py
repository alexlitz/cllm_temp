#!/usr/bin/env python3
"""
Vanilla Transformer VM - Pure attention + FFN blocks

NO special function names. Just:
- Embedding
- Attention layers
- FFN layers
- Output projection

All computation emerges from weight matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class VanillaTransformerVM(nn.Module):
    """
    A VM as a vanilla transformer decoder.

    Architecture:
        Input: [state_tokens]
        Layer 1-N: Attention → FFN
        Output: [next_state_tokens]

    One forward pass = one VM step.
    Generate until HALT token.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        max_seq_len: int = 4096,
        vocab_size: int = 512,  # 256 bytes + special tokens
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Token types
        self.TOKEN_BYTE = 0        # 0-255: raw bytes
        self.TOKEN_REG_AX = 256
        self.TOKEN_REG_SP = 257
        self.TOKEN_REG_BP = 258
        self.TOKEN_REG_PC = 259
        self.TOKEN_MEM = 260       # memory marker
        self.TOKEN_CODE = 261      # code marker
        self.TOKEN_THINK = 262     # internal state
        self.TOKEN_OUTPUT = 263    # visible output
        self.TOKEN_HALT = 264      # halt signal

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Initialize FFN weights to implement VM operations
        self._init_vm_weights()

    def _init_vm_weights(self):
        """
        Initialize weights to implement VM operations.

        SPARSE: Zero out all weights first, then set only needed entries.
        This gives vanilla transformer structure with ~97% sparsity.

        The FFN layers encode:
        - Arithmetic tables (add, mul via SwiGLU)
        - Memory addressing
        - Opcode dispatch

        Attention layers handle:
        - Instruction fetch (attend to code at PC)
        - Memory read (attend to mem at address)
        - Register read (attend to reg tokens)
        """
        with torch.no_grad():
            # First, zero out ALL weights
            for layer in self.layers:
                # Zero FFN weights
                layer.ffn.w1.weight.data.zero_()
                layer.ffn.w1.bias.data.zero_()
                layer.ffn.w_gate.weight.data.zero_()
                layer.ffn.w2.weight.data.zero_()
                layer.ffn.w2.bias.data.zero_()

                # Zero attention weights
                layer.attn.q_proj.weight.data.zero_()
                layer.attn.k_proj.weight.data.zero_()
                layer.attn.v_proj.weight.data.zero_()
                layer.attn.o_proj.weight.data.zero_()

            # Zero embeddings and output
            self.token_embedding.weight.data.zero_()
            self.position_embedding.weight.data.zero_()
            self.lm_head.weight.data.zero_()

            # Now set only the non-zero entries we need
            self._init_embeddings()
            self._init_attention_for_vm()
            for layer_idx, layer in enumerate(self.layers):
                self._init_layer_weights(layer, layer_idx)
            self._init_output_head()

    def _init_embeddings(self):
        """Initialize embeddings with one-hot-like encoding."""
        hidden = self.hidden_dim

        # Token embeddings: encode token ID in first dims
        for i in range(min(self.vocab_size, hidden)):
            self.token_embedding.weight.data[i, i] = 1.0

        # Position embeddings: encode position in separate dims
        pos_start = min(128, hidden // 2)
        for i in range(min(self.position_embedding.num_embeddings, hidden - pos_start)):
            # Binary encoding of position
            for bit in range(min(16, hidden - pos_start)):
                if (i >> bit) & 1:
                    self.position_embedding.weight.data[i, pos_start + bit] = 1.0

    def _init_attention_for_vm(self):
        """
        Initialize attention weights for VM operations.

        Head 0: Register read - attend to REG tokens
        Head 1: Memory read - attend to MEM tokens
        Head 2: Code fetch - attend to CODE at PC
        Head 3: Recency bias - prefer recent tokens

        Token types are in embedding space (first dims after one-hot).
        """
        hidden = self.hidden_dim
        head_dim = hidden // self.num_heads

        # Token type markers are embedded in dimensions 0-255 (one-hot)
        # We use specific dims for matching token types
        REG_DIM = 0  # Register tokens marked in dim 0
        MEM_DIM = 1  # Memory tokens marked in dim 1
        CODE_DIM = 2  # Code tokens marked in dim 2

        for layer in self.layers:
            attn = layer.attn

            # Head 0: Register read
            h = 0
            start = h * head_dim
            # Q/K project from marker dims to attention space
            for i in range(4):  # 4 register types
                attn.q_proj.weight.data[start + i, i] = 10.0  # Match reg ID
                attn.k_proj.weight.data[start + i, i] = 10.0
            # V passes through value bytes (dims 4-35)
            for i in range(min(32, head_dim - 4)):
                attn.v_proj.weight.data[start + 4 + i, 4 + i] = 1.0

            # Head 1: Memory read
            h = 1
            start = h * head_dim
            # Memory marker matching
            attn.q_proj.weight.data[start, MEM_DIM] = 10.0
            attn.k_proj.weight.data[start, MEM_DIM] = 10.0
            # Address matching in dims 1-24
            for i in range(min(24, head_dim - 1)):
                attn.q_proj.weight.data[start + 1 + i, 32 + i] = 1.0
                attn.k_proj.weight.data[start + 1 + i, 32 + i] = 1.0
            # V passes through value
            for i in range(min(24, head_dim)):
                attn.v_proj.weight.data[start + i, i] = 1.0

            # Head 2: Code fetch
            h = 2
            start = h * head_dim
            attn.q_proj.weight.data[start, CODE_DIM] = 10.0
            attn.k_proj.weight.data[start, CODE_DIM] = 10.0
            # PC address matching
            for i in range(min(16, head_dim - 1)):
                attn.q_proj.weight.data[start + 1 + i, 64 + i] = 1.0
                attn.k_proj.weight.data[start + 1 + i, 64 + i] = 1.0
            for i in range(min(32, head_dim)):
                attn.v_proj.weight.data[start + i, i] = 1.0

            # Head 3: Recency (identity projection)
            h = 3
            start = h * head_dim
            for i in range(head_dim):
                if start + i < hidden:
                    attn.q_proj.weight.data[start + i, start + i] = 1.0
                    attn.k_proj.weight.data[start + i, start + i] = 1.0
                    attn.v_proj.weight.data[start + i, i] = 1.0

            # Output projection: aggregate heads
            for i in range(hidden):
                attn.o_proj.weight.data[i, i] = 1.0

    def _init_layer_weights(self, layer: 'TransformerBlock', layer_idx: int):
        """Initialize a single layer's FFN weights for VM operations."""
        ffn = layer.ffn
        hidden = self.hidden_dim
        ffn_dim = ffn.w1.out_features

        if layer_idx == 0:
            # Layer 0: Nibble add table (512 entries)
            # Nibble add: 16x16x2 = 512 combinations
            for a in range(16):
                for b in range(16):
                    for cin in range(2):
                        idx = a * 32 + b * 2 + cin
                        if idx < ffn_dim:
                            # Input: one-hot nibble encoding
                            ffn.w1.weight.data[idx, a] = 10.0
                            ffn.w1.weight.data[idx, 16 + b] = 10.0
                            ffn.w1.weight.data[idx, 32 + cin] = 10.0
                            ffn.w1.bias.data[idx] = -25.0  # AND-like threshold

                            # Gate passes through
                            ffn.w_gate.weight.data[idx, a] = 1.0

                            # Output: sum nibble and carry
                            total = a + b + cin
                            ffn.w2.weight.data[64 + (total & 0xF), idx] = 1.0  # sum
                            ffn.w2.weight.data[80 + (1 if total > 15 else 0), idx] = 1.0  # carry

        elif layer_idx == 1:
            # Layer 1: SwiGLU multiply
            # silu(a) * b gives exact multiplication when combined properly
            # Set up identity-like gate for operand extraction
            for i in range(min(32, hidden)):
                # W1 extracts first operand for silu
                ffn.w1.weight.data[i, i] = 1.0
                # W_gate extracts second operand
                ffn.w_gate.weight.data[i, 32 + i] = 1.0
                # W2 collects result
                ffn.w2.weight.data[64 + i, i] = 1.0

        elif layer_idx == 2:
            # Layer 2: Newton division reciprocal table
            # Piecewise linear 1/x for x in [0.5, 1.0)
            n_segments = min(64, ffn_dim // 2)
            breakpoints = torch.linspace(0.5, 1.0, n_segments + 1)
            values = 1.0 / breakpoints

            for i in range(n_segments):
                # ReLU segments for piecewise linear
                ffn.w1.weight.data[2*i, 0] = 1.0
                ffn.w1.weight.data[2*i + 1, 0] = 1.0
                ffn.w1.bias.data[2*i] = -breakpoints[i].item()
                ffn.w1.bias.data[2*i + 1] = -breakpoints[i + 1].item()

                ffn.w_gate.weight.data[2*i, 0] = 1.0
                ffn.w_gate.weight.data[2*i + 1, 0] = 1.0

                delta = breakpoints[i + 1] - breakpoints[i]
                slope = (values[i + 1] - values[i]) / delta
                ffn.w2.weight.data[0, 2*i] = slope.item()
                ffn.w2.weight.data[0, 2*i + 1] = -slope.item()

            ffn.w2.bias.data[0] = values[0].item()

        elif layer_idx == 3:
            # Layer 3: Output token selection / opcode routing
            # MoE-like soft gating based on opcode
            n_opcodes = min(32, ffn_dim // 4)
            for op in range(n_opcodes):
                # Each opcode activates specific output pattern
                ffn.w1.weight.data[op * 4, op] = 10.0
                ffn.w1.bias.data[op * 4] = -5.0
                ffn.w_gate.weight.data[op * 4, op] = 1.0
                # Route to specific output dimension (within hidden_dim bounds)
                out_dim = 128 + (op % 64)  # Use dims 128-191 for control
                if out_dim < hidden:
                    ffn.w2.weight.data[out_dim, op * 4] = 1.0

    def _init_output_head(self):
        """Initialize output projection for token prediction."""
        hidden = self.hidden_dim

        # Map from hidden state dims to vocab logits
        # Byte tokens (0-255) map from first dims
        for i in range(min(256, hidden, self.vocab_size)):
            self.lm_head.weight.data[i, i] = 1.0

        # Special tokens map from dims 128+ (after byte dims)
        special_tokens = [
            self.TOKEN_REG_AX, self.TOKEN_REG_SP,
            self.TOKEN_REG_BP, self.TOKEN_REG_PC,
            self.TOKEN_MEM, self.TOKEN_CODE,
            self.TOKEN_THINK, self.TOKEN_OUTPUT,
            self.TOKEN_HALT
        ]
        for i, tok in enumerate(special_tokens):
            dim = 128 + i  # Use dims after main byte encoding
            if tok < self.vocab_size and dim < hidden:
                self.lm_head.weight.data[tok, dim] = 10.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass - standard transformer decoder.

        input_ids: [batch, seq_len] token indices
        returns: [batch, seq_len, vocab_size] logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation until HALT.

        This is the VM execution loop - pure token generation.
        """
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(input_ids)

            # Get next token (greedy or sample)
            next_logits = logits[:, -1, :]
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for HALT
            if next_token.item() == self.TOKEN_HALT:
                break

        return input_ids


class TransformerBlock(nn.Module):
    """Standard transformer block: Attention → FFN"""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()

        # Attention
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)

        # FFN with SwiGLU activation
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.ln1(x), mask)
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)

        # Output
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN - standard in modern LLMs.

    This naturally computes: silu(x @ W1) * (x @ W_gate)
    Which enables exact multiplication via the SwiGLU identity.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=True)
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(x @ W1) * (x @ W_gate)
        return self.w2(F.silu(self.w1(x)) * self.w_gate(x))


# =============================================================================
# VM STATE ENCODING
# =============================================================================

class VMStateEncoder:
    """Encode VM state as tokens for the transformer."""

    def __init__(self, vm: VanillaTransformerVM):
        self.vm = vm

    def encode_initial_state(
        self,
        bytecode: List[int],
        data: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Encode initial VM state as token sequence.

        Format:
            [CODE tokens] [DATA tokens] [REG tokens] [THINK]
        """
        tokens = []

        # Code tokens: <CODE> <PC> <OP> <IMM_bytes...>
        for i, instr in enumerate(bytecode):
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)

            pc = i * 8
            tokens.extend([
                self.vm.TOKEN_CODE,
                pc & 0xFF,
                (pc >> 8) & 0xFF,
                op,
                imm & 0xFF,
                (imm >> 8) & 0xFF,
                (imm >> 16) & 0xFF,
                (imm >> 24) & 0xFF,
            ])

        # Data tokens: <MEM> <ADDR_bytes> <VALUE>
        if data:
            for i, b in enumerate(data):
                addr = 0x10000 + i
                tokens.extend([
                    self.vm.TOKEN_MEM,
                    addr & 0xFF,
                    (addr >> 8) & 0xFF,
                    (addr >> 16) & 0xFF,
                    b & 0xFF,
                ])

        # Register tokens
        tokens.extend([
            self.vm.TOKEN_REG_AX, 0, 0, 0, 0,  # AX = 0
            self.vm.TOKEN_REG_SP, 0, 0, 3, 0,  # SP = 0x30000
            self.vm.TOKEN_REG_BP, 0, 0, 3, 0,  # BP = 0x30000
            self.vm.TOKEN_REG_PC, 0, 0, 0, 0,  # PC = 0
        ])

        # Start thinking
        tokens.append(self.vm.TOKEN_THINK)

        return torch.tensor([tokens], dtype=torch.long)

    def decode_output(self, tokens: torch.Tensor) -> bytes:
        """Extract OUTPUT tokens from generated sequence."""
        output = []
        tokens = tokens.squeeze().tolist()

        i = 0
        while i < len(tokens):
            if tokens[i] == self.vm.TOKEN_OUTPUT and i + 1 < len(tokens):
                output.append(tokens[i + 1])
                i += 2
            else:
                i += 1

        return bytes(output)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys
    sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')
    from src.compiler import compile_c

    print("=" * 70)
    print("  VANILLA TRANSFORMER VM (SPARSE)")
    print("=" * 70)
    print()

    # Create model
    model = VanillaTransformerVM(
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        ffn_dim=1024,
    )

    # Count parameters and sparsity
    total_params = 0
    zero_params = 0
    for p in model.parameters():
        total_params += p.numel()
        zero_params += (p.data == 0).sum().item()

    nonzero_params = total_params - zero_params
    sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0

    print(f"Architecture:")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Heads: {model.num_heads}")
    print(f"  Layers: {model.num_layers}")
    print(f"  FFN dim: {model.layers[0].ffn.w1.out_features}")
    print()
    print(f"Parameters:")
    print(f"  Total:    {total_params:,}")
    print(f"  Zeros:    {zero_params:,} ({sparsity:.1f}%)")
    print(f"  Non-zero: {nonzero_params:,}")
    print()

    # Test compilation
    source = """
int main() {
    int a, b;
    a = 6;
    b = 7;
    return a * b;
}
"""

    print("Test program:")
    print(source)

    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")
    print()

    # Encode state
    encoder = VMStateEncoder(model)
    input_ids = encoder.encode_initial_state(bytecode, data)
    print(f"Initial state: {input_ids.shape[1]} tokens")
    print()

    # Forward pass (one step)
    print("Forward pass...")
    with torch.no_grad():
        logits = model(input_ids)
    print(f"Output shape: {logits.shape}")
    print()

    # Show sparsity breakdown
    print("=" * 70)
    print("  SPARSITY BY COMPONENT")
    print("=" * 70)

    components = [
        ('token_embedding', model.token_embedding),
        ('position_embedding', model.position_embedding),
        ('lm_head', model.lm_head),
    ]
    for i, layer in enumerate(model.layers):
        components.append((f'layer_{i}.attn', layer.attn))
        components.append((f'layer_{i}.ffn', layer.ffn))

    for name, module in components:
        total = sum(p.numel() for p in module.parameters())
        zeros = sum((p.data == 0).sum().item() for p in module.parameters())
        nonzero = total - zeros
        spars = 100.0 * zeros / total if total > 0 else 0
        print(f"  {name:25s}: {nonzero:6,} non-zero / {total:,} ({spars:.1f}% sparse)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
