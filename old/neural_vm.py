#!/usr/bin/env python3
"""
Neural VM - Everything in transformer forward pass.

Attention layers: Find CODE/REG/MEM tokens
FFN layers: Compute ALU operations
Output head: Produce tokens

No separate NeuralALU calls. Everything flows through:
    next_token = model(context).argmax()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class Vocab:
    PAD, BOS, EOS = 0, 1, 2
    CODE, REG_PC, REG_AX, REG_SP, REG_BP, MEM = 3, 4, 5, 6, 7, 8
    STEP_END = 9
    BYTE_BASE = 16
    VOCAB_SIZE = 272

    @staticmethod
    def byte_tok(val: int) -> int:
        return Vocab.BYTE_BASE + (val & 0xFF)

    @staticmethod
    def tok_byte(tok: int) -> int:
        return tok - Vocab.BYTE_BASE


class NeuralVMTransformer(nn.Module):
    """
    Transformer with FFN weights baked for VM operations.

    Layer 0: Attention finds markers, FFN extracts position info
    Layer 1: Attention gathers operands, FFN computes ALU
    Layer 2: Attention refines, FFN formats output
    Output head: Produces token
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Token embedding
        self.tok_emb = nn.Embedding(Vocab.VOCAB_SIZE, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            self._make_layer() for _ in range(num_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, Vocab.VOCAB_SIZE, bias=False)

        self._bake_all_weights()

    def _make_layer(self):
        return nn.ModuleDict({
            'ln1': nn.LayerNorm(self.dim),
            'q_proj': nn.Linear(self.dim, self.dim, bias=False),
            'k_proj': nn.Linear(self.dim, self.dim, bias=False),
            'v_proj': nn.Linear(self.dim, self.dim, bias=False),
            'o_proj': nn.Linear(self.dim, self.dim, bias=False),
            'ln2': nn.LayerNorm(self.dim),
            'ffn_up': nn.Linear(self.dim, self.dim * 4, bias=False),
            'ffn_gate': nn.Linear(self.dim, self.dim * 4, bias=False),
            'ffn_down': nn.Linear(self.dim * 4, self.dim, bias=False),
        })

    def _bake_all_weights(self):
        """Bake all weights for VM operation."""
        with torch.no_grad():
            self._bake_embeddings()
            self._bake_layer0()
            self._bake_layer1()
            self._bake_layer2()
            self._bake_output()

    def _bake_embeddings(self):
        """Bake token embeddings."""
        self.tok_emb.weight.zero_()

        # Markers in first 16 dims
        markers = [Vocab.BOS, Vocab.EOS, Vocab.CODE, Vocab.REG_PC,
                   Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP, Vocab.MEM, Vocab.STEP_END]
        for i, m in enumerate(markers):
            self.tok_emb.weight[m, i] = 10.0

        # Byte values: one-hot in dims 16-271
        # Plus nibble decomposition in dims 272-303
        for b in range(256):
            tok = Vocab.BYTE_BASE + b
            self.tok_emb.weight[tok, 16 + b] = 1.0
            # Low nibble (dims 272-287)
            if 272 + (b & 0xF) < self.dim:
                self.tok_emb.weight[tok, 272 + (b & 0xF)] = 1.0
            # High nibble (dims 288-303)
            if 288 + ((b >> 4) & 0xF) < self.dim:
                self.tok_emb.weight[tok, 288 + ((b >> 4) & 0xF)] = 1.0

    def _bake_layer0(self):
        """Layer 0: Find position markers."""
        layer = self.layers[0]
        for key in layer:
            if hasattr(layer[key], 'weight'):
                layer[key].weight.zero_()

        # Attention: Q at last position looks for markers
        # K responds to marker tokens
        for i in range(9):  # 9 marker types
            layer['q_proj'].weight[i, i] = 5.0
            layer['k_proj'].weight[i, i] = 5.0

        # V passes through embeddings
        for i in range(min(self.dim, 320)):
            layer['v_proj'].weight[i, i] = 1.0
            layer['o_proj'].weight[i, i] = 1.0

        # FFN: identity for now
        for i in range(min(self.dim, self.dim * 4)):
            layer['ffn_up'].weight[i, i] = 1.0
            layer['ffn_gate'].weight[i, i] = 1.0
        for i in range(min(self.dim, self.dim * 4)):
            layer['ffn_down'].weight[i, i] = 1.0

    def _bake_layer1(self):
        """Layer 1: Gather operands, compute ALU."""
        layer = self.layers[1]
        for key in layer:
            if hasattr(layer[key], 'weight'):
                layer[key].weight.zero_()

        # Similar attention structure
        for i in range(min(self.dim, 320)):
            layer['q_proj'].weight[i, i] = 1.0
            layer['k_proj'].weight[i, i] = 1.0
            layer['v_proj'].weight[i, i] = 1.0
            layer['o_proj'].weight[i, i] = 1.0

        # FFN implements nibble ALU operations
        # Build nibble add table in FFN weights
        self._bake_nibble_add_in_ffn(layer)

    def _bake_nibble_add_in_ffn(self, layer):
        """Bake nibble addition lookup table into FFN."""
        # Input: two nibbles in dims 272-287 and 288-303
        # Output: result nibble + carry

        hidden = self.dim * 4
        # Address computation: combine both nibbles to unique address
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                if addr < hidden:
                    # Detect nibble a in low position
                    if 272 + a < self.dim:
                        layer['ffn_up'].weight[addr, 272 + a] = 5.0
                    # Detect nibble b in high position
                    if 288 + b < self.dim:
                        layer['ffn_up'].weight[addr, 288 + b] = 5.0

                    # Gate activates for match
                    if 272 + a < self.dim:
                        layer['ffn_gate'].weight[addr, 272 + a] = 5.0
                    if 288 + b < self.dim:
                        layer['ffn_gate'].weight[addr, 288 + b] = 5.0

                    # Output: result nibble
                    result = (a + b) & 0xF
                    if 272 + result < self.dim:
                        layer['ffn_down'].weight[272 + result, addr] = 1.0

    def _bake_layer2(self):
        """Layer 2: Final processing."""
        layer = self.layers[2]
        for key in layer:
            if hasattr(layer[key], 'weight'):
                layer[key].weight.zero_()

        # Pass through
        for i in range(min(self.dim, 320)):
            layer['q_proj'].weight[i, i] = 1.0
            layer['k_proj'].weight[i, i] = 1.0
            layer['v_proj'].weight[i, i] = 1.0
            layer['o_proj'].weight[i, i] = 1.0
            layer['ffn_up'].weight[i, i] = 1.0
            layer['ffn_gate'].weight[i, i] = 1.0
            layer['ffn_down'].weight[i, i] = 1.0

    def _bake_output(self):
        """Bake output head."""
        self.lm_head.weight.zero_()

        # Markers
        markers = [Vocab.BOS, Vocab.EOS, Vocab.CODE, Vocab.REG_PC,
                   Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP, Vocab.MEM, Vocab.STEP_END]
        for i, m in enumerate(markers):
            self.lm_head.weight[m, i] = 10.0

        # Bytes: from one-hot dims
        for b in range(256):
            self.lm_head.weight[Vocab.BYTE_BASE + b, 16 + b] = 10.0

    def _attention(self, layer, x, mask):
        """Apply one attention layer."""
        B, L, D = x.shape
        h = layer['ln1'](x)

        q = layer['q_proj'](h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = layer['k_proj'](h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = layer['v_proj'](h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return x + layer['o_proj'](out)

    def _ffn(self, layer, x):
        """Apply FFN with SwiGLU."""
        h = layer['ln2'](x)
        return x + layer['ffn_down'](F.silu(layer['ffn_up'](h)) * layer['ffn_gate'](h))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full forward pass through baked transformer."""
        B, L = tokens.shape
        x = self.tok_emb(tokens)

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), 1) * -1e9

        # Transform through layers
        for layer in self.layers:
            x = self._attention(layer, x, mask)
            x = self._ffn(layer, x)

        # Output logits for last position
        return self.lm_head(self.ln_f(x[:, -1]))


class NeuralVM:
    """
    VM where everything flows through transformer forward pass.

    generate_next_token() = model(context).argmax()
    """

    def __init__(self):
        self.model = NeuralVMTransformer()
        self.context: List[int] = []
        self.gen_count = 0
        self.halted = False

    def generate_next_token(self) -> int:
        """Generate via transformer forward."""
        tokens = torch.tensor([self.context], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(tokens)
        next_tok = logits.argmax(dim=-1).item()

        if next_tok == Vocab.EOS:
            self.halted = True

        self.context.append(next_tok)
        self.gen_count += 1
        return next_tok

    def load_code(self, bytecode: List[int]):
        """Load bytecode."""
        self.context = [Vocab.BOS]
        self.gen_count = 0
        self.halted = False

        for idx, instr in enumerate(bytecode):
            op, imm = instr & 0xFF, instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            pc = idx * 8
            self.context.extend([
                Vocab.CODE,
                Vocab.byte_tok((pc >> 8) & 0xFF),
                Vocab.byte_tok(pc & 0xFF),
                Vocab.byte_tok(op),
                *[Vocab.byte_tok((imm >> (i * 8)) & 0xFF) for i in range(4)]
            ])

    def run(self, max_tokens: int = 10000) -> List[int]:
        """Generate tokens until EOS or max."""
        generated = []
        while self.gen_count < max_tokens and not self.halted:
            tok = self.generate_next_token()
            generated.append(tok)
            if tok == Vocab.EOS:
                break
        return generated


def test_forward():
    print("=" * 60)
    print("  NEURAL VM")
    print("  Everything through transformer forward pass")
    print("=" * 60)
    print()

    model = NeuralVMTransformer()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Test forward pass
    context = [Vocab.BOS, Vocab.CODE, Vocab.byte_tok(0), Vocab.byte_tok(0),
               Vocab.byte_tok(1), Vocab.byte_tok(42), Vocab.byte_tok(0),
               Vocab.byte_tok(0), Vocab.byte_tok(0)]

    tokens = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens)

    pred = logits.argmax(dim=-1).item()
    print(f"After CODE + IMM(42), predicted token: {pred}")

    # Test embedding roundtrip
    correct = 0
    for b in range(256):
        tok = Vocab.byte_tok(b)
        emb = model.tok_emb.weight[tok]
        out = model.lm_head(model.ln_f(emb.unsqueeze(0).unsqueeze(0)))
        if out.argmax(dim=-1).item() == tok:
            correct += 1

    print(f"Byte roundtrip: {correct}/256 correct")
    print()

    print("Architecture:")
    print("  Embedding: tokens -> vectors")
    print("  Layer 0: Attention finds markers")
    print("  Layer 1: FFN has nibble add tables")
    print("  Layer 2: Final processing")
    print("  Output: vectors -> tokens")
    print()
    print("All in forward():")
    print("  next_tok = model(context).argmax()")


if __name__ == "__main__":
    test_forward()
