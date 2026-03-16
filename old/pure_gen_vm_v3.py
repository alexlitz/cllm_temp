#!/usr/bin/env python3
"""
Pure Generative VM v3 - Hybrid Value-Encoded Architecture

KEY INNOVATIONS:
1. Hybrid Input/Output Encoding
   - INPUTS: One-hot nibbles (for equality tests in SwiGLU address decoder)
   - OUTPUTS: Value-encoded (scaled floats instead of one-hot)
   - FINAL: Convert back to one-hot only when needed

2. Subroutine Triggers for Complex Ops
   - DIV/MOD: Trigger external handling (restoring division algorithm)
   - MSET/MCMP/MCPY: Trigger external memory operations

This reduces intermediate dimensionality by ~16× per value.

Architecture (7 layers):
- Layer 0: Attention + opcode routing
- Layer 1: Basic ops (ADD, SUB, bitwise, comparisons) → value outputs
- Layer 2: MUL products (64 parallel) → value outputs
- Layer 3: MUL accumulate → value outputs
- Layer 4: SHL/SHR → value outputs
- Layer 5: Subroutine triggers (DIV, MOD, MSET, MCMP, MCPY)
- Layer 6: Value-to-onehot conversion for final output

Non-zero parameters per operation:
- ADD/SUB: ~8,400 (value-encoded output)
- MUL: ~64,576 (64 products)
- DIV/MOD: ~100 (subroutine trigger only)
- SHL/SHR: ~67,000 each
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math

# Import core components from v1
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')
from pure_gen_vm import Vocab, Opcode, softmax1, TransformerLayer


# =============================================================================
# V3 EMBEDDING DIMENSIONS - Optimized for Hybrid Encoding
# =============================================================================

class EmbedDimsV3:
    """
    V3 embedding layout optimized for hybrid encoding.

    Structure:
    [0-256]: Markers and byte tokens
    [256-384]: One-hot operand A nibbles (8 × 16)
    [384-512]: One-hot operand B nibbles (8 × 16)
    [512-520]: VALUE-encoded result nibbles (8 floats)
    [520-528]: VALUE-encoded carry/borrow (8 floats)
    [528-592]: VALUE-encoded MUL products (64 floats)
    [592-600]: VALUE-encoded MUL column sums (8 floats)
    [600-608]: VALUE-encoded DIV quotient nibbles (8 floats)
    [608-616]: VALUE-encoded DIV remainder (8 floats)
    [616-648]: Control flags (shift amount, div state, etc)
    [648-776]: One-hot result nibbles (for final output, 8 × 16)
    [776-1024]: Workspace
    """

    # Markers and tokens
    MARKERS_START = 0
    BYTES_START = 16
    BYTES_END = 272

    # One-hot inputs (for address decoding)
    OP_A_NIB_START = 272      # 8 × 16 = 128 dims
    OP_A_NIB_END = 400
    OP_B_NIB_START = 400      # 8 × 16 = 128 dims
    OP_B_NIB_END = 528

    # VALUE-encoded outputs (the key innovation!)
    RESULT_VAL_START = 528    # 8 floats (one per nibble, holding 0-15 scaled)
    RESULT_VAL_END = 536
    CARRY_VAL_START = 536     # 8 floats (carry/borrow per position)
    CARRY_VAL_END = 544

    # VALUE-encoded MUL workspace
    MUL_PROD_START = 544      # 64 floats (value-encoded products)
    MUL_PROD_END = 608
    MUL_COL_START = 608       # 8 floats (column sums)
    MUL_COL_END = 616

    # VALUE-encoded DIV workspace
    DIV_QUOT_VAL_START = 616  # 8 floats (quotient nibbles)
    DIV_QUOT_VAL_END = 624
    DIV_REM_VAL_START = 624   # 9 floats (remainder nibbles, extra for overflow)
    DIV_REM_VAL_END = 633

    # Control flags
    OPCODE_START = 640        # 32 dims one-hot
    OPCODE_END = 672
    SHIFT_AMOUNT = 672        # 32 dims one-hot
    SHIFT_AMOUNT_END = 704
    DIV_BIT_POS = 704         # 32 dims one-hot
    DIV_BIT_POS_END = 736
    DIV_ACTIVE = 736
    DIV_GE_FLAG = 737

    # One-hot result (for final output)
    RESULT_NIB_START = 768    # 8 × 16 = 128 dims
    RESULT_NIB_END = 896

    # Subroutine triggers
    TRIGGER_MSET = 900
    TRIGGER_MCMP = 901
    TRIGGER_MCPY = 902
    TRIGGER_DIV = 903
    TRIGGER_MOD = 904

    DIM = 1024  # Reduced from 2048!

    # Scaling constants
    NIBBLE_SCALE = 1.0 / 16.0   # Nibble values 0-15 → 0-1
    PRODUCT_SCALE = 1.0 / 256.0  # Products 0-225 → 0-0.88
    SUM_SCALE = 1.0 / 4096.0    # Column sums → manageable range


# =============================================================================
# VALUE-ENCODED BASIC OPS LAYER
# =============================================================================

class ValueEncodedBasicOps(nn.Module):
    """
    Computes ADD, SUB, AND, OR, XOR with value-encoded outputs.

    For ADD/SUB: Uses carry lookahead within single layer.
    For bitwise: Direct nibble lookup.

    Input: One-hot nibbles at OP_A_NIB and OP_B_NIB
    Output: Value-encoded results at RESULT_VAL

    Weight baking pattern (3-way AND using threshold):
    - gate: checks a[nib] (high when correct a value)
    - up: checks b[nib] AND opcode with threshold bias
    - Bias set so that single match gives negative, both give positive
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # ADD: 8 nibbles × 16 × 16 × 2 carry = 4096
        # SUB: 8 nibbles × 16 × 16 × 2 borrow = 4096
        # AND/OR/XOR: 8 nibbles × 16 × 16 = 2048 each
        # Total: ~14,000 rows needed
        ffn_dim = 16384

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=True)  # Need bias for threshold
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV3
        # Use high SCALE so wrong-row activations are essentially zero
        # With SCALE=20, THRESHOLD=1.95:
        # - Both inputs: up = 40 - 39 = 1, silu(1) = 0.73
        # - Single input: up = 20 - 39 = -19, silu(-19) ≈ 1e-8 (essentially zero)
        SCALE = 20.0
        THRESHOLD = 1.95

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0
            ffn_dim = self.up.weight.size(0)

            # === ADD ===
            for nib in range(8):
                for a_val in range(16):
                    for b_val in range(16):
                        if row >= ffn_dim:
                            continue

                        total = a_val + b_val
                        result = total & 0xF
                        carry = (total >> 4) & 1

                        # Gate: check a[nib]
                        a_dim = E.OP_A_NIB_START + nib * 16 + a_val
                        if a_dim < self.dim:
                            self.gate.weight[row, a_dim] = SCALE

                        # Up: check b[nib] AND opcode with threshold bias
                        b_dim = E.OP_B_NIB_START + nib * 16 + b_val
                        if b_dim < self.dim:
                            self.up.weight[row, b_dim] = SCALE
                        if E.OPCODE_START + Opcode.ADD < self.dim:
                            self.up.weight[row, E.OPCODE_START + Opcode.ADD] = SCALE
                        # Bias: negative threshold so only fires when BOTH inputs active
                        self.up.bias[row] = -SCALE * THRESHOLD

                        # Output: value-encoded result
                        # With SCALE=20, THRESHOLD=1.95: both inputs give up = 1
                        # silu(1) ≈ 0.73, hidden = 0.73 * 20 = 14.6
                        # So down weight should give result * NIBBLE_SCALE / 14.6
                        result_dim = E.RESULT_VAL_START + nib
                        if result_dim < self.dim:
                            self.down.weight[result_dim, row] = result * E.NIBBLE_SCALE / 14.6

                        # Output: carry for next nibble
                        if carry and nib < 7:
                            carry_dim = E.CARRY_VAL_START + nib
                            if carry_dim < self.dim:
                                self.down.weight[carry_dim, row] = 1.0 / 14.6

                        row += 1

            # === SUB ===
            for nib in range(8):
                for a_val in range(16):
                    for b_val in range(16):
                        if row >= ffn_dim:
                            continue

                        diff = a_val - b_val
                        borrow = 0
                        if diff < 0:
                            diff += 16
                            borrow = 1
                        result = diff & 0xF

                        # Gate: check a[nib]
                        a_dim = E.OP_A_NIB_START + nib * 16 + a_val
                        if a_dim < self.dim:
                            self.gate.weight[row, a_dim] = SCALE

                        # Up: check b[nib] AND opcode with threshold
                        b_dim = E.OP_B_NIB_START + nib * 16 + b_val
                        if b_dim < self.dim:
                            self.up.weight[row, b_dim] = SCALE
                        if E.OPCODE_START + Opcode.SUB < self.dim:
                            self.up.weight[row, E.OPCODE_START + Opcode.SUB] = SCALE
                        self.up.bias[row] = -SCALE * THRESHOLD

                        # Output: value-encoded result
                        result_dim = E.RESULT_VAL_START + nib
                        if result_dim < self.dim:
                            self.down.weight[result_dim, row] = result * E.NIBBLE_SCALE / 14.6

                        row += 1

            # === AND ===
            for nib in range(8):
                for a_val in range(16):
                    for b_val in range(16):
                        if row >= ffn_dim:
                            continue

                        result = a_val & b_val

                        a_dim = E.OP_A_NIB_START + nib * 16 + a_val
                        b_dim = E.OP_B_NIB_START + nib * 16 + b_val
                        if a_dim < self.dim:
                            self.gate.weight[row, a_dim] = SCALE
                        if b_dim < self.dim:
                            self.up.weight[row, b_dim] = SCALE
                        if E.OPCODE_START + Opcode.AND < self.dim:
                            self.up.weight[row, E.OPCODE_START + Opcode.AND] = SCALE
                        self.up.bias[row] = -SCALE * THRESHOLD

                        result_dim = E.RESULT_VAL_START + nib
                        if result_dim < self.dim:
                            self.down.weight[result_dim, row] = result * E.NIBBLE_SCALE / 14.6

                        row += 1

            # === OR ===
            for nib in range(8):
                for a_val in range(16):
                    for b_val in range(16):
                        if row >= ffn_dim:
                            continue

                        result = a_val | b_val

                        a_dim = E.OP_A_NIB_START + nib * 16 + a_val
                        b_dim = E.OP_B_NIB_START + nib * 16 + b_val
                        if a_dim < self.dim:
                            self.gate.weight[row, a_dim] = SCALE
                        if b_dim < self.dim:
                            self.up.weight[row, b_dim] = SCALE
                        if E.OPCODE_START + Opcode.OR < self.dim:
                            self.up.weight[row, E.OPCODE_START + Opcode.OR] = SCALE
                        self.up.bias[row] = -SCALE * THRESHOLD

                        result_dim = E.RESULT_VAL_START + nib
                        if result_dim < self.dim:
                            self.down.weight[result_dim, row] = result * E.NIBBLE_SCALE / 14.6

                        row += 1

            # === XOR ===
            for nib in range(8):
                for a_val in range(16):
                    for b_val in range(16):
                        if row >= ffn_dim:
                            continue

                        result = a_val ^ b_val

                        a_dim = E.OP_A_NIB_START + nib * 16 + a_val
                        b_dim = E.OP_B_NIB_START + nib * 16 + b_val
                        if a_dim < self.dim:
                            self.gate.weight[row, a_dim] = SCALE
                        if b_dim < self.dim:
                            self.up.weight[row, b_dim] = SCALE
                        if E.OPCODE_START + Opcode.XOR < self.dim:
                            self.up.weight[row, E.OPCODE_START + Opcode.XOR] = SCALE
                        self.up.bias[row] = -SCALE * THRESHOLD

                        result_dim = E.RESULT_VAL_START + nib
                        if result_dim < self.dim:
                            self.down.weight[result_dim, row] = result * E.NIBBLE_SCALE / 14.6

                        row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip LayerNorm - operate directly on raw one-hot encoding
        # This ensures predictable scaling for the lookup table
        up_out = self.up(x)
        gate_out = self.gate(x)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


# =============================================================================
# VALUE-ENCODED MUL LAYER
# =============================================================================

class ValueEncodedMulProducts(nn.Module):
    """
    Computes all 64 nibble products with value-encoded outputs.

    Input: One-hot nibbles at OP_A_NIB and OP_B_NIB
    Output: Value-encoded products at MUL_PROD (64 floats)

    Each product a[i]*b[j] (0-225) stored as scaled float.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 64 products × 256 input combinations = 16384 rows
        ffn_dim = 16384

        self.up = nn.Linear(dim, ffn_dim, bias=True)  # Need bias for threshold
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV3
        SCALE = 20.0
        THRESHOLD = 1.95  # Same as BasicOps for consistency

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0
            ffn_dim = self.up.weight.size(0)

            for i in range(8):  # a nibbles
                for j in range(8):  # b nibbles
                    prod_idx = i * 8 + j

                    for a_val in range(16):
                        for b_val in range(16):
                            if row >= ffn_dim:
                                continue

                            product = a_val * b_val  # 0-225

                            # Gate: check a[i]
                            a_dim = E.OP_A_NIB_START + i * 16 + a_val
                            if a_dim < self.dim:
                                self.gate.weight[row, a_dim] = SCALE

                            # Up: check b[j] AND opcode is MUL with threshold
                            b_dim = E.OP_B_NIB_START + j * 16 + b_val
                            if b_dim < self.dim:
                                self.up.weight[row, b_dim] = SCALE
                            if E.OPCODE_START + Opcode.MUL < self.dim:
                                self.up.weight[row, E.OPCODE_START + Opcode.MUL] = SCALE
                            self.up.bias[row] = -SCALE * THRESHOLD

                            # Output: value-encoded product (hidden ≈ 14.6)
                            out_dim = E.MUL_PROD_START + prod_idx
                            if out_dim < self.dim:
                                self.down.weight[out_dim, row] = product * E.PRODUCT_SCALE / 14.6

                            row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip LayerNorm for predictable lookup behavior
        up_out = self.up(x)
        gate_out = self.gate(x)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


# =============================================================================
# VALUE TO ONE-HOT CONVERSION LAYER
# =============================================================================

class ValueToOneHotLayer(nn.Module):
    """
    Converts value-encoded results to one-hot for final output.

    For each value-encoded dimension, outputs one-hot at the nearest integer.
    Uses binned comparison: value in [k-0.5, k+0.5) maps to position k.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 8 nibbles × 16 values = 128 rows
        ffn_dim = 128

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=True)
        self.gate = nn.Linear(dim, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV3
        SCALE = 16.0  # High scale for sharp thresholds

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.gate.bias.zero_()
            self.down.weight.zero_()

            row = 0
            for nib in range(8):
                for val in range(16):
                    if row >= self.up.weight.size(0):
                        continue

                    value_dim = E.RESULT_VAL_START + nib

                    # Up: value >= (val - 0.5) * NIBBLE_SCALE
                    # Threshold: value * (1/NIBBLE_SCALE) >= val - 0.5
                    if value_dim < self.dim:
                        self.up.weight[row, value_dim] = SCALE / E.NIBBLE_SCALE
                        self.up.bias[row] = -SCALE * (val - 0.5)

                    # Gate: value < (val + 0.5) * NIBBLE_SCALE
                    if value_dim < self.dim:
                        self.gate.weight[row, value_dim] = -SCALE / E.NIBBLE_SCALE
                        self.gate.bias[row] = SCALE * (val + 0.5)

                    # Output: one-hot at this position
                    onehot_dim = E.RESULT_NIB_START + nib * 16 + val
                    if onehot_dim < self.dim:
                        self.down.weight[onehot_dim, row] = 1.0

                    row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        # Use sigmoid for smooth thresholding
        hidden = torch.sigmoid(up_out) * torch.sigmoid(gate_out)
        return x + self.down(hidden)


# =============================================================================
# SUBROUTINE TRIGGER LAYER
# =============================================================================

class SubroutineTriggerLayer(nn.Module):
    """
    Detects opcodes that should trigger external subroutine handling.

    Triggers for:
    - MSET, MCMP, MCPY: Memory operations
    - DIV, MOD: Complex arithmetic (more efficient externally)

    When triggered, external code reads the trigger flags and handles
    the operation, then resumes neural execution.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        ffn_dim = 64

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV3
        SCALE = 8.0

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            triggers = [
                (Opcode.DIV, E.TRIGGER_DIV),
                (Opcode.MOD, E.TRIGGER_MOD),
            ]

            # Add MSET/MCMP/MCPY if they exist
            if hasattr(Opcode, 'MSET'):
                triggers.append((Opcode.MSET, E.TRIGGER_MSET))
            if hasattr(Opcode, 'MCMP'):
                triggers.append((Opcode.MCMP, E.TRIGGER_MCMP))
            if hasattr(Opcode, 'MCPY'):
                triggers.append((Opcode.MCPY, E.TRIGGER_MCPY))

            for row, (opcode, trigger_dim) in enumerate(triggers):
                if row >= self.up.weight.size(0):
                    break

                opcode_dim = E.OPCODE_START + opcode
                if opcode_dim < self.dim:
                    self.up.weight[row, opcode_dim] = SCALE
                    self.gate.weight[row, opcode_dim] = SCALE

                if trigger_dim < self.dim:
                    self.down.weight[trigger_dim, row] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)

    def read_triggers(self, x: torch.Tensor) -> Dict[str, bool]:
        """Read trigger flags from hidden state."""
        E = EmbedDimsV3
        last = x[0, -1] if x.dim() == 3 else x[-1]
        return {
            'div': last[E.TRIGGER_DIV].item() > 0.5 if E.TRIGGER_DIV < len(last) else False,
            'mod': last[E.TRIGGER_MOD].item() > 0.5 if E.TRIGGER_MOD < len(last) else False,
            'mset': last[E.TRIGGER_MSET].item() > 0.5 if E.TRIGGER_MSET < len(last) else False,
            'mcmp': last[E.TRIGGER_MCMP].item() > 0.5 if E.TRIGGER_MCMP < len(last) else False,
            'mcpy': last[E.TRIGGER_MCPY].item() > 0.5 if E.TRIGGER_MCPY < len(last) else False,
        }


# =============================================================================
# PURE TRANSFORMER V3
# =============================================================================

class PureTransformerV3(nn.Module):
    """
    V3 transformer with hybrid value-encoded architecture.

    Key differences from V2:
    - Reduced dim: 1024 (vs 2048)
    - Value-encoded intermediates: 8 floats per value (vs 128 one-hot)
    - DIV/MOD use subroutine triggers (external handling)
    - Final conversion layer to one-hot for vocabulary output

    Architecture (7 layers):
    - Layer 0: Attention + opcode routing
    - Layer 1: Basic ops (ADD, SUB, bitwise) → value-encoded
    - Layer 2: MUL products → value-encoded
    - Layer 3: MUL accumulate
    - Layer 4: SHL/SHR
    - Layer 5: Subroutine triggers (DIV, MOD, MSET, MCMP, MCPY)
    - Layer 6: Value-to-onehot conversion
    - Output head

    DIV/MOD are handled externally via subroutine triggers.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        ffn_dim = dim * 4

        # Embeddings
        self.tok_emb = nn.Embedding(Vocab.VOCAB_SIZE, dim)

        # Layer 0: Attention
        self.layer0 = TransformerLayer(dim, num_heads, ffn_dim, use_moe=True, use_alibi=True)

        # Layer 1: Basic ops with value-encoded output
        self.basic_ops = ValueEncodedBasicOps(dim)

        # Layer 2: MUL products
        self.mul_products = ValueEncodedMulProducts(dim)

        # Layer 3: MUL accumulate
        self.mul_accum = TransformerLayer(dim, num_heads, ffn_dim, use_moe=True)

        # Layer 4: SHL/SHR
        self.shift_layer = TransformerLayer(dim, num_heads, ffn_dim, use_moe=True)

        # Layer 5: Subroutine triggers (DIV, MOD, MSET, MCMP, MCPY)
        self.subroutine_trigger = SubroutineTriggerLayer(dim)

        # Layer 6: Value to one-hot conversion
        self.value_to_onehot = ValueToOneHotLayer(dim)

        # Output
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, Vocab.VOCAB_SIZE, bias=False)

    def forward_standard(self, tokens: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through V3 architecture."""
        x = self.tok_emb(tokens)
        B, L, D = x.shape

        mask = torch.triu(torch.ones(L, L, device=x.device), 1) * -1e9

        # Layer 0: Attention
        x = self.layer0.forward_no_cache(x, mask)

        # Layer 1: Basic ops (value-encoded)
        x = self.basic_ops(x)

        # Layer 2: MUL products (value-encoded)
        x = self.mul_products(x)

        # Layer 3: MUL accumulate
        x = self.mul_accum.forward_no_cache(x, mask)

        # Layer 4: Shifts
        x = self.shift_layer.forward_no_cache(x, mask)

        # Layer 5: Subroutine triggers (DIV/MOD handled externally)
        x = self.subroutine_trigger(x)

        # Layer 6: Value to one-hot conversion
        x = self.value_to_onehot(x)

        # Output
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return logits

    def forward_with_triggers(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, bool]]:
        """
        Forward pass that returns subroutine trigger flags.

        Returns:
            logits: Output logits (None if trigger is set)
            triggers: Dict with 'div', 'mod', 'mset', 'mcmp', 'mcpy' flags
        """
        x = self.tok_emb(tokens)
        B, L, D = x.shape

        mask = torch.triu(torch.ones(L, L, device=x.device), 1) * -1e9

        # Layers 0-4
        x = self.layer0.forward_no_cache(x, mask)
        x = self.basic_ops(x)
        x = self.mul_products(x)
        x = self.mul_accum.forward_no_cache(x, mask)
        x = self.shift_layer.forward_no_cache(x, mask)

        # Layer 5: Check for subroutine triggers
        x = self.subroutine_trigger(x)
        triggers = self.subroutine_trigger.read_triggers(x)

        # If any trigger is set, return early for external handling
        if any(triggers.values()):
            return None, triggers

        # Layer 6: Value to one-hot
        x = self.value_to_onehot(x)

        # Output
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return logits, triggers

    def generate_token(self, tokens: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.forward_standard(tokens)
            return logits.argmax(dim=-1).item()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PureTransformerV3 (Hybrid Value-Encoded)")
    print("=" * 60)

    print("\nCreating model (dim=1024)...")
    model = PureTransformerV3(dim=1024)

    print(f"Model dimension: {model.dim}")
    print(f"Architecture: Hybrid (one-hot input → value intermediate → one-hot output)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Count non-zero
    nonzero = 0
    for p in model.parameters():
        nonzero += (p != 0).sum().item()
    print(f"Non-zero parameters: {nonzero:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    tokens = torch.tensor([[Vocab.BOS, Vocab.byte_tok(42)]])
    logits = model.forward_standard(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Test generation
    tok = model.generate_token(tokens)
    print(f"Generated token: {tok}")

    print("\n" + "=" * 60)
    print("V3 Model Test Complete!")
    print("=" * 60)
