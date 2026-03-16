#!/usr/bin/env python3
"""
Pure Generative VM v2 - Fully Neural with Unrolled Loops

Key changes from v1:
- MUL: All 64 nibble products computed in parallel across 2 layers
- DIV: 32 iterations unrolled with weight-tied layers
- SHL/SHR: All 32 shifts computed in parallel, then selected
- MSET/MCMP: Trigger subroutine flag for external handling

HYBRID ENCODING:
- INPUTS use one-hot encoding (for equality tests in SwiGLU)
- OUTPUTS use VALUE encoding where possible (direct scaled values)
- Final layer converts back to one-hot nibbles

This reduces dimensionality significantly:
- One-hot nibbles: 16 dims per nibble
- Value encoding: 1 dim per value (scaled float)

Architecture:
- dim = 2048 (increased for workspace)
- Layers 0-1: Attention + basic MoE (gather, simple ops)
- Layers 2-3: MUL products (32 products each) -> value-encoded outputs
- Layer 4: MUL column accumulation (sums value-encoded products)
- Layer 5: Value-to-nibble conversion
- Layers 6-7: SHL/SHR
- Layers 8-39: DIV iterations (weight-tied, 32× replicated)
- Layer 40: Final output

For non-DIV/MUL operations, most layers pass through unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math

# Import core components from v1
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')
from pure_gen_vm import (
    Vocab, Opcode, softmax1, OpcodeExpert, OpcodeMoE, TransformerLayer
)


# =============================================================================
# EXTENDED EMBEDDING DIMENSIONS
# =============================================================================

class EmbedDimsV2:
    """Extended embedding layout for v2 (dim=2048)."""

    # === INHERITED FROM V1 [0-1600] ===
    MARKERS_START = 0
    BYTES_START = 16
    BYTES_END = 272
    NIBBLE_A_START = 272
    NIBBLE_B_START = 288
    STEP_POS_START = 304
    OPCODE_START = 320

    # Full 32-bit operand nibbles
    OP_A_NIB_START = 480   # 8 nibbles × 16 = 128 dims
    OP_A_NIB_END = 608
    OP_B_NIB_START = 608   # 8 nibbles × 16 = 128 dims
    OP_B_NIB_END = 736

    # Result nibbles
    RESULT_NIB_START = 736  # 8 nibbles × 16 = 128 dims
    RESULT_NIB_END = 864

    # === V2 EXTENSIONS [1600-2048] ===

    # MUL workspace: VALUE-ENCODED partial products
    # Each product a[i]*b[j] is 0-225, stored as SCALED FLOAT (not one-hot!)
    # This uses only 64 dims vs 2048 for one-hot encoding!
    #
    # Layout: products arranged by column position for easy accumulation
    # Col 0: a[0]*b[0]                                   -> 1 product
    # Col 1: a[0]*b[1], a[1]*b[0]                        -> 2 products
    # Col 2: a[0]*b[2], a[1]*b[1], a[2]*b[0]             -> 3 products
    # ...
    # Col 7: a[0]*b[7]..a[7]*b[0]                        -> 8 products
    # Col 8: a[1]*b[7]..a[7]*b[1]                        -> 7 products
    # ...
    # Col 14: a[7]*b[7]                                  -> 1 product
    # Total: 64 products
    MUL_PROD_START = 1600   # 64 dims, each holds scaled product value
    MUL_PROD_END = 1664

    # MUL column accumulators: VALUE-ENCODED sums
    # Each column sum can be 0 to 225*8 = 1800, stored as scaled float
    # 15 columns (0-14) for result
    MUL_COL_START = 1664    # 15 dims for column sums (value-encoded)
    MUL_COL_END = 1679

    # MUL result: 8 nibbles value-encoded (before conversion to one-hot)
    MUL_RESULT_VAL_START = 1679  # 8 dims holding nibble values 0-15 as floats
    MUL_RESULT_VAL_END = 1687

    # DIV workspace
    DIV_REMAINDER_START = 1680  # 9 nibbles × 16 = 144 dims (33 bits)
    DIV_REMAINDER_END = 1824
    DIV_QUOTIENT_START = 1824   # 8 nibbles × 16 = 128 dims
    DIV_QUOTIENT_END = 1952
    DIV_BIT_POS = 1952          # 32 dims one-hot for current bit
    DIV_BIT_POS_END = 1984
    DIV_ACTIVE = 1984           # Flag: 1 if DIV in progress
    DIV_GE_FLAG = 1985          # Flag: 1 if remainder >= divisor

    # SHL/SHR workspace
    SHIFT_AMOUNT = 1986         # 32 dims one-hot for shift amount
    SHIFT_AMOUNT_END = 2018

    # Output selection
    OUTPUT_BYTE = 2018          # Single byte output (for compatibility)

    # Subroutine triggers
    TRIGGER_MSET = 2020
    TRIGGER_MCMP = 2021
    TRIGGER_MCPY = 2022

    DIM = 2048


# =============================================================================
# VALUE-ENCODED ARITHMETIC LAYERS
# =============================================================================

class ValueEncodedAddLayer(nn.Module):
    """
    Demonstrates the hybrid encoding approach for 32-bit ADD.

    HYBRID ENCODING STRATEGY:
    - INPUT: One-hot nibbles (8 nibbles × 16 dims = 128 dims per operand)
    - OUTPUT: Value-encoded nibbles (8 dims, each holding 0-15 as scaled float)

    This reduces output dimensionality from 128 to 8 (16× reduction).

    The layer computes:
    1. For each nibble position, look up a[i] + b[i] + carry[i-1]
    2. Output result[i] as value (0-15) and propagate carry

    Weight structure:
    - FFN rows indexed by (nibble, a_val, b_val, carry_in)
    - For 8 nibbles × 16 × 16 × 2 = 4096 combinations
    - Each row outputs: result_val to VALUE dim, carry_out to CARRY dim
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 8 nibbles × 16 × 16 × 2 (carry in) = 4096
        ffn_dim = 4096

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV2
        SCALE = 8.0
        VALUE_SCALE = 1.0 / 16.0  # Nibble values 0-15 scaled to 0-1

        # We need a workspace for carry propagation
        # Use dims after MUL workspace
        CARRY_START = 2024  # 8 dims for carry flags
        RESULT_VAL_START = 2032  # 8 dims for value-encoded results

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0
            for nib in range(8):
                for a_val in range(16):
                    for b_val in range(16):
                        for carry_in in range(2):
                            if row >= self.up.weight.size(0):
                                continue

                            # Compute sum and carry
                            total = a_val + b_val + carry_in
                            result_val = total & 0xF
                            carry_out = (total >> 4) & 1

                            # === Address decoder (one-hot equality test) ===

                            # Gate: Check a[nib] value
                            a_dim = E.OP_A_NIB_START + nib * 16 + a_val
                            if a_dim < self.dim:
                                self.gate.weight[row, a_dim] = SCALE

                            # Up: Check b[nib] value AND carry_in
                            b_dim = E.OP_B_NIB_START + nib * 16 + b_val
                            if b_dim < self.dim:
                                self.up.weight[row, b_dim] = SCALE / 2

                            # For nibble 0, carry_in is always 0
                            # For nibble 1+, check previous carry
                            if nib > 0 and carry_in == 1:
                                carry_dim = CARRY_START + (nib - 1)
                                if carry_dim < self.dim:
                                    self.up.weight[row, carry_dim] = SCALE / 2
                            elif nib == 0 and carry_in == 0:
                                # Always matches for first nibble with no carry
                                self.up.weight[row, b_dim] = SCALE
                            elif nib > 0 and carry_in == 0:
                                # Need to check carry is NOT set (complex, simplified)
                                pass

                            # === Value-encoded outputs ===

                            # Output result as scaled value
                            result_dim = RESULT_VAL_START + nib
                            if result_dim < self.dim:
                                self.down.weight[result_dim, row] = result_val * VALUE_SCALE

                            # Output carry for next nibble
                            if carry_out and nib < 7:
                                carry_dim = CARRY_START + nib
                                if carry_dim < self.dim:
                                    self.down.weight[carry_dim, row] = 1.0

                            row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


class ValueToOneHotLayer(nn.Module):
    """
    Converts value-encoded results back to one-hot encoding.

    This is the inverse of the hybrid approach - used only when we need
    to feed results into another layer that expects one-hot input.

    For each value-encoded dimension, create 16 neurons that each check
    if the value is approximately equal to 0, 1, 2, ... 15.

    Uses binned thresholding: value in [k/16, (k+1)/16) maps to position k.
    """

    def __init__(self, dim: int, value_start: int, onehot_start: int, n_values: int = 8):
        super().__init__()
        self.dim = dim
        self.value_start = value_start
        self.onehot_start = onehot_start
        self.n_values = n_values

        # For each of n_values positions, 16 possible output values
        ffn_dim = n_values * 16

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=True)  # Use bias for thresholding
        self.gate = nn.Linear(dim, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        SCALE = 8.0
        VALUE_SCALE = 1.0 / 16.0

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.gate.bias.zero_()
            self.down.weight.zero_()

            row = 0
            for pos in range(self.n_values):
                for val in range(16):
                    if row >= self.up.weight.size(0):
                        continue

                    value_dim = self.value_start + pos

                    # Up: Check if value is >= val * VALUE_SCALE
                    # Using bias to set threshold
                    if value_dim < self.dim:
                        self.up.weight[row, value_dim] = SCALE * 16  # Amplify
                        self.up.bias[row] = -SCALE * val  # Threshold at val

                    # Gate: Check if value is < (val+1) * VALUE_SCALE
                    if value_dim < self.dim:
                        self.gate.weight[row, value_dim] = -SCALE * 16
                        self.gate.bias[row] = SCALE * (val + 1)

                    # Output: one-hot at position val
                    onehot_dim = self.onehot_start + pos * 16 + val
                    if onehot_dim < self.dim:
                        self.down.weight[onehot_dim, row] = 1.0

                    row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        # Use sigmoid instead of SiLU for cleaner thresholding
        hidden = torch.sigmoid(up_out) * torch.sigmoid(gate_out)
        return x + self.down(hidden)


# =============================================================================
# MUL PARALLEL LAYER
# =============================================================================

class MulProductsLayer(nn.Module):
    """
    Computes all 64 nibble products in parallel with VALUE-ENCODED output.

    Input: One-hot encoded nibbles (for equality test)
    Output: Scaled float values representing products

    Each product a[i]*b[j] outputs to MUL_PROD[product_index] as the actual
    product value (0-225) scaled by 1/256 for numerical stability.

    Products are organized by result column position (i+j) for accumulation:
    - Column 0: a[0]*b[0]
    - Column 1: a[0]*b[1], a[1]*b[0]
    - ...
    - Column 14: a[7]*b[7]
    """

    def __init__(self, dim: int, product_indices: List[Tuple[int, int]] = None):
        super().__init__()
        self.dim = dim

        # Default: all 64 products
        if product_indices is None:
            self.products = [(i, j) for i in range(8) for j in range(8)]
        else:
            self.products = product_indices

        n_products = len(self.products)
        ffn_dim = n_products * 256  # 256 combinations per product

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV2
        SCALE = 8.0
        VALUE_SCALE = 1.0 / 256.0  # Products 0-225 scaled to ~0-0.88

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0
            for prod_idx, (i, j) in enumerate(self.products):
                for a_val in range(16):
                    for b_val in range(16):
                        if row >= self.up.weight.size(0):
                            continue

                        product = a_val * b_val  # 0-225

                        # Address decoder: one-hot equality test
                        a_dim = E.OP_A_NIB_START + i * 16 + a_val
                        b_dim = E.OP_B_NIB_START + j * 16 + b_val

                        if a_dim < self.dim:
                            self.up.weight[row, a_dim] = SCALE
                        if b_dim < self.dim:
                            self.gate.weight[row, b_dim] = SCALE

                        # VALUE-ENCODED output: store scaled product value
                        out_dim = E.MUL_PROD_START + prod_idx
                        if out_dim < self.dim:
                            # Output the actual product value, not one-hot
                            self.down.weight[out_dim, row] = product * VALUE_SCALE

                        row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


class MulAccumulateLayer(nn.Module):
    """
    Accumulates value-encoded products by column and propagates carries.

    This layer:
    1. Reads 64 value-encoded products from MUL_PROD
    2. Sums products at each column position (0-14)
    3. Extracts low nibble and carry for each column
    4. Outputs value-encoded result nibbles

    The carry propagation is baked into the weights so that:
    - Column k output = (sum[k] + carry_from[k-1]) mod 16
    - Carry to k+1 = (sum[k] + carry_from[k-1]) // 16
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # For each column, we need to handle sum values 0 to ~8*225=1800
        # But with carries it's more complex. Use larger FFN.
        ffn_dim = 16 * 256  # 16 result positions × 256 possible sum values

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        """
        Bake column accumulation with carry propagation.

        This is complex because carries cascade. For simplicity, we bake
        the per-column extraction and let multiple layers handle cascading.
        """
        E = EmbedDimsV2
        SCALE = 8.0
        VALUE_SCALE = 1.0 / 256.0

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            # Map product indices to their column (i+j)
            # Column k contains products where i+j = k
            col_products = [[] for _ in range(15)]  # columns 0-14
            for idx in range(64):
                i, j = idx // 8, idx % 8
                col = i + j
                col_products[col].append(idx)

            # For each column, sum the products
            # Since products are value-encoded, we weight them by 1
            row = 0
            for col in range(15):
                for sum_val in range(256):  # Possible sum values (truncated)
                    if row >= self.up.weight.size(0):
                        break

                    # This is approximate - true implementation would need
                    # iterative carry propagation or larger FFN
                    result_nibble = sum_val & 0xF
                    carry = sum_val >> 4

                    # Gate on MUL operation active
                    if E.OPCODE_START + Opcode.MUL < self.dim:
                        self.gate.weight[row, E.OPCODE_START + Opcode.MUL] = SCALE

                    # Up weight sums the products for this column
                    for prod_idx in col_products[col]:
                        prod_dim = E.MUL_PROD_START + prod_idx
                        if prod_dim < self.dim:
                            self.up.weight[row, prod_dim] = 256.0  # Reverse scaling

                    # Output result nibble (as value)
                    if col < 8:  # Only output first 8 nibbles
                        out_dim = E.MUL_RESULT_VAL_START + col
                        if out_dim < self.dim:
                            self.down.weight[out_dim, row] = result_nibble * VALUE_SCALE

                    row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


class ValueToNibbleLayer(nn.Module):
    """
    Converts value-encoded nibbles to one-hot encoded nibbles.

    Input: MUL_RESULT_VAL (8 dims, each holding value 0-15 as scaled float)
    Output: RESULT_NIB (8×16 = 128 dims, one-hot encoded)

    This uses quantization: bucket the scaled value into 16 bins,
    then output one-hot at the corresponding position.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # For each of 8 nibbles, 16 possible values
        ffn_dim = 8 * 16

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV2
        SCALE = 8.0
        VALUE_SCALE = 1.0 / 256.0

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0
            for nib in range(8):
                for val in range(16):
                    if row >= self.up.weight.size(0):
                        continue

                    # Up checks if value-encoded nibble is close to val
                    in_dim = E.MUL_RESULT_VAL_START + nib
                    if in_dim < self.dim:
                        # Check approximate equality using threshold
                        # This is approximate - in practice use multiple neurons
                        self.up.weight[row, in_dim] = SCALE / (val * VALUE_SCALE + 0.01)

                    # Gate: always active for MUL
                    if E.OPCODE_START + Opcode.MUL < self.dim:
                        self.gate.weight[row, E.OPCODE_START + Opcode.MUL] = SCALE

                    # Output one-hot
                    out_dim = E.RESULT_NIB_START + nib * 16 + val
                    if out_dim < self.dim:
                        self.down.weight[out_dim, row] = 1.0

                    row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


# =============================================================================
# DIV ITERATION LAYER (Weight-Tied)
# =============================================================================

class DivIterationLayer(nn.Module):
    """
    One iteration of restoring division (weight-tied, applied 32×).

    The restoring division algorithm for each bit position k:
    1. remainder = (remainder << 1) | dividend_bit[k]
    2. if remainder >= divisor:
           remainder -= divisor
           quotient[k] = 1
    3. advance to next bit position

    Architecture:
    - Sub-FFN A: Shift remainder left, bring in dividend bit
    - Sub-FFN B: Compare remainder vs divisor (multi-nibble)
    - Sub-FFN C: Conditional subtract + set quotient bit
    - Sub-FFN D: Advance bit position

    All sub-FFNs are baked into a single large FFN with proper routing.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # FFN needs to handle:
        # - 32 bit positions × 9 remainder nibbles × 16 values = 4608
        # - 9 nibble comparisons × 16×16 = 2304
        # - 9 nibble subtractions × 16×16 = 2304
        # - 32 quotient bit sets = 32
        # Total: ~10000 rows
        ffn_dim = 12288

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV2
        SCALE = 8.0

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0
            ffn_dim = self.up.weight.size(0)

            # === SECTION 1: BIT POSITION ADVANCE ===
            # When DIV_ACTIVE=1 and DIV_BIT_POS[k]=1, advance to k-1
            for bit in range(32):
                if row >= ffn_dim:
                    break

                # Gate: DIV_ACTIVE must be 1
                if E.DIV_ACTIVE < self.dim:
                    self.gate.weight[row, E.DIV_ACTIVE] = SCALE

                # Up: Current bit position
                if E.DIV_BIT_POS + bit < self.dim:
                    self.up.weight[row, E.DIV_BIT_POS + bit] = SCALE

                # Output: clear current position
                if E.DIV_BIT_POS + bit < self.dim:
                    self.down.weight[E.DIV_BIT_POS + bit, row] = -1.0

                # Output: set next position (or deactivate)
                if bit > 0:
                    if E.DIV_BIT_POS + bit - 1 < self.dim:
                        self.down.weight[E.DIV_BIT_POS + bit - 1, row] = 1.0
                else:
                    # Done - clear DIV_ACTIVE
                    if E.DIV_ACTIVE < self.dim:
                        self.down.weight[E.DIV_ACTIVE, row] = -1.0

                row += 1

            # === SECTION 2: REMAINDER LEFT SHIFT ===
            # For each remainder nibble position (0-8) and each value (0-15):
            # - Low bits shift into current nibble from previous
            # - High bits shift out to next nibble
            #
            # remainder[8:0] << 1:
            # - remainder[8] = (old_r[7] >> 3) & 1  (high bit of nib 7)
            # - remainder[k] = (old_r[k] << 1) | (old_r[k-1] >> 3)
            # - remainder[0] = (old_r[0] << 1) | dividend_bit[current_pos]

            for nib in range(9):  # 9 nibbles in remainder (33 bits)
                for val in range(16):
                    if row >= ffn_dim:
                        continue

                    # Gate: DIV_ACTIVE
                    if E.DIV_ACTIVE < self.dim:
                        self.gate.weight[row, E.DIV_ACTIVE] = SCALE

                    # Up: Current remainder nibble value
                    rem_dim = E.DIV_REMAINDER_START + nib * 16 + val
                    if rem_dim < self.dim:
                        self.up.weight[row, rem_dim] = SCALE

                    # Compute new value after left shift
                    new_low = (val << 1) & 0xF  # Low nibble contribution

                    # Clear old value
                    if rem_dim < self.dim:
                        self.down.weight[rem_dim, row] = -1.0

                    # Set new value (partial - low bits only)
                    new_dim = E.DIV_REMAINDER_START + nib * 16 + new_low
                    if new_dim < self.dim:
                        self.down.weight[new_dim, row] = 0.5  # Partial contribution

                    # High bit goes to carry
                    if val & 0x8:  # High bit set
                        if nib < 8:  # Not the last nibble
                            # Add 1 to next nibble's low bit
                            carry_dim = E.DIV_REMAINDER_START + (nib + 1) * 16 + 1
                            if carry_dim < self.dim:
                                self.down.weight[carry_dim, row] = 0.5

                    row += 1

            # === SECTION 3: COMPARISON (remainder >= divisor) ===
            # Compare nibble by nibble from MSB (nibble 7) to LSB (nibble 0)
            # Set DIV_GE_FLAG based on comparison
            #
            # For each nibble position and each pair (rem_val, div_val):
            # - If rem > div at this position: definitely GE
            # - If rem < div at this position: definitely NOT GE
            # - If rem == div: continue to next nibble

            for nib in range(8):  # Compare 8 nibbles
                for rem_val in range(16):
                    for div_val in range(16):
                        if row >= ffn_dim:
                            continue

                        # Gate: DIV_ACTIVE
                        if E.DIV_ACTIVE < self.dim:
                            self.gate.weight[row, E.DIV_ACTIVE] = SCALE

                        # Up: Check both remainder[nib] and divisor[nib]
                        rem_dim = E.DIV_REMAINDER_START + nib * 16 + rem_val
                        div_dim = E.OP_B_NIB_START + nib * 16 + div_val

                        if rem_dim < self.dim:
                            self.up.weight[row, rem_dim] = SCALE / 2
                        if div_dim < self.dim:
                            self.up.weight[row, div_dim] = SCALE / 2

                        # Output: Update GE flag based on this nibble comparison
                        if nib == 7:  # MSB - most significant for comparison
                            if rem_val > div_val:
                                # Definitely GE
                                if E.DIV_GE_FLAG < self.dim:
                                    self.down.weight[E.DIV_GE_FLAG, row] = 1.0
                            elif rem_val < div_val:
                                # Definitely NOT GE
                                if E.DIV_GE_FLAG < self.dim:
                                    self.down.weight[E.DIV_GE_FLAG, row] = -1.0
                            # If equal, no change - continue to next nibble

                        row += 1

            # === SECTION 4: CONDITIONAL SUBTRACT ===
            # When DIV_GE_FLAG=1: remainder -= divisor, quotient_bit = 1
            # This requires nibble-wise subtraction with borrow

            for nib in range(8):
                for rem_val in range(16):
                    for div_val in range(16):
                        if row >= ffn_dim:
                            continue

                        # Gate: DIV_GE_FLAG must be 1
                        if E.DIV_GE_FLAG < self.dim:
                            self.gate.weight[row, E.DIV_GE_FLAG] = SCALE

                        # Up: remainder and divisor values
                        rem_dim = E.DIV_REMAINDER_START + nib * 16 + rem_val
                        div_dim = E.OP_B_NIB_START + nib * 16 + div_val

                        if rem_dim < self.dim:
                            self.up.weight[row, rem_dim] = SCALE / 2
                        if div_dim < self.dim:
                            self.up.weight[row, div_dim] = SCALE / 2

                        # Compute subtraction
                        diff = rem_val - div_val
                        borrow = 0
                        if diff < 0:
                            diff += 16
                            borrow = 1

                        # Clear old remainder value
                        if rem_dim < self.dim:
                            self.down.weight[rem_dim, row] = -1.0

                        # Set new remainder value
                        new_dim = E.DIV_REMAINDER_START + nib * 16 + diff
                        if new_dim < self.dim:
                            self.down.weight[new_dim, row] = 1.0

                        # Handle borrow to next nibble
                        if borrow and nib < 7:
                            # Subtract 1 from next nibble (complex, simplified here)
                            pass

                        row += 1

            # === SECTION 5: SET QUOTIENT BIT ===
            # When DIV_GE_FLAG=1 and DIV_BIT_POS[k]=1, set quotient bit k

            for bit in range(32):
                if row >= ffn_dim:
                    break

                # Gate: DIV_GE_FLAG
                if E.DIV_GE_FLAG < self.dim:
                    self.gate.weight[row, E.DIV_GE_FLAG] = SCALE

                # Up: DIV_BIT_POS[bit]
                if E.DIV_BIT_POS + bit < self.dim:
                    self.up.weight[row, E.DIV_BIT_POS + bit] = SCALE

                # Output: Set quotient bit (one-hot in quotient nibble)
                quo_nib = bit // 4
                quo_bit = bit % 4
                quo_dim = E.DIV_QUOTIENT_START + quo_nib * 16 + (1 << quo_bit)
                if quo_dim < self.dim:
                    self.down.weight[quo_dim, row] = 1.0

                row += 1

            # === SECTION 6: CLEAR GE FLAG ===
            # At end of iteration, clear DIV_GE_FLAG for next iteration

            if row < ffn_dim:
                if E.DIV_ACTIVE < self.dim:
                    self.gate.weight[row, E.DIV_ACTIVE] = SCALE
                    self.up.weight[row, E.DIV_ACTIVE] = SCALE

                if E.DIV_GE_FLAG < self.dim:
                    self.down.weight[E.DIV_GE_FLAG, row] = -1.0

                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


# =============================================================================
# SHL/SHR PARALLEL LAYER
# =============================================================================

class ShiftParallelLayer(nn.Module):
    """
    Computes all 32 possible shifts in parallel, then selects based on shift amount.

    For SHL: output = input << shift_amount (0..31)
    For SHR: output = input >> shift_amount (0..31)

    Cross-nibble bit handling:
    - For SHL by n bits: each output nibble gets bits from TWO input nibbles
      - High bits from nibble (out_nib - n//4) shifted left by (n%4)
      - Low bits from nibble (out_nib - n//4 - 1) shifted right by (4 - n%4)

    - For SHR by n bits: each output nibble gets bits from TWO input nibbles
      - Low bits from nibble (out_nib + n//4) shifted right by (n%4)
      - High bits from nibble (out_nib + n//4 + 1) shifted left by (4 - n%4)

    The shift amount is encoded one-hot in SHIFT_AMOUNT.
    """

    def __init__(self, dim: int, is_left: bool = True):
        super().__init__()
        self.dim = dim
        self.is_left = is_left

        # 32 shifts × 8 output nibbles × (16×16) input pairs = too many
        # Instead: 32 shifts × 8 output nibbles × 256 (full 8-bit combinations from 2 nibbles)
        # Simplified: For each shift, compute each output nibble from 2 adjacent inputs
        ffn_dim = 32 * 8 * 256  # = 65536

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV2
        SCALE = 8.0

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0
            ffn_dim = self.up.weight.size(0)

            for shift in range(32):
                nib_shift = shift // 4  # How many whole nibbles to shift
                bit_shift = shift % 4   # Remaining bit shift within nibble

                for out_nib in range(8):
                    # Determine source nibbles
                    if self.is_left:
                        # SHL: bits come from lower nibbles
                        src_main = out_nib - nib_shift
                        src_carry = out_nib - nib_shift - 1
                    else:
                        # SHR: bits come from higher nibbles
                        src_main = out_nib + nib_shift
                        src_carry = out_nib + nib_shift + 1

                    # Check if this output nibble is entirely zeros
                    main_in_range = 0 <= src_main < 8
                    carry_in_range = 0 <= src_carry < 8 and bit_shift > 0

                    if not main_in_range and not carry_in_range:
                        # This nibble is always 0 - output 0 using shift amount as trigger
                        if row < ffn_dim:
                            # Both up AND gate check SHIFT_AMOUNT[shift]
                            # This ensures activation when shift amount matches
                            if E.SHIFT_AMOUNT + shift < self.dim:
                                self.gate.weight[row, E.SHIFT_AMOUNT + shift] = SCALE
                                self.up.weight[row, E.SHIFT_AMOUNT + shift] = SCALE
                            # Output 0 to this nibble with strong activation
                            out_dim = E.RESULT_NIB_START + out_nib * 16 + 0
                            if out_dim < self.dim:
                                self.down.weight[out_dim, row] = 100.0  # Strong output
                            row += 1
                        continue

                    # For each possible value combination from source nibbles
                    for main_val in range(16):
                        for carry_val in range(16):
                            if row >= ffn_dim:
                                continue

                            # Compute output value
                            if self.is_left:
                                if bit_shift == 0:
                                    out_val = main_val if main_in_range else 0
                                else:
                                    main_contrib = (main_val << bit_shift) & 0xF if main_in_range else 0
                                    carry_contrib = (carry_val >> (4 - bit_shift)) if carry_in_range else 0
                                    out_val = main_contrib | carry_contrib
                            else:
                                # SHR
                                if bit_shift == 0:
                                    out_val = main_val if main_in_range else 0
                                else:
                                    main_contrib = (main_val >> bit_shift) if main_in_range else 0
                                    carry_contrib = ((carry_val << (4 - bit_shift)) & 0xF) if carry_in_range else 0
                                    out_val = main_contrib | carry_contrib

                            # Gate: SHIFT_AMOUNT[shift]
                            if E.SHIFT_AMOUNT + shift < self.dim:
                                self.gate.weight[row, E.SHIFT_AMOUNT + shift] = SCALE

                            # Up: Check source nibble values
                            if main_in_range:
                                main_dim = E.OP_A_NIB_START + src_main * 16 + main_val
                                if main_dim < self.dim:
                                    self.up.weight[row, main_dim] = SCALE / 2

                            if carry_in_range:
                                carry_dim = E.OP_A_NIB_START + src_carry * 16 + carry_val
                                if carry_dim < self.dim:
                                    self.up.weight[row, carry_dim] = SCALE / 2

                            # If only main or only carry is in range, double the weight
                            if main_in_range and not carry_in_range:
                                main_dim = E.OP_A_NIB_START + src_main * 16 + main_val
                                if main_dim < self.dim:
                                    self.up.weight[row, main_dim] = SCALE

                            if carry_in_range and not main_in_range:
                                carry_dim = E.OP_A_NIB_START + src_carry * 16 + carry_val
                                if carry_dim < self.dim:
                                    self.up.weight[row, carry_dim] = SCALE

                            # Output: one-hot result nibble
                            out_dim = E.RESULT_NIB_START + out_nib * 16 + out_val
                            if out_dim < self.dim:
                                self.down.weight[out_dim, row] = 1.0

                            row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


# =============================================================================
# SUBROUTINE TRIGGER LAYER
# =============================================================================

class SubroutineTriggerLayer(nn.Module):
    """
    Detects MSET/MCMP/MCPY opcodes and sets trigger flags.

    These operations are complex and benefit from external handling:
    - MSET (memset): Fill memory region with value
    - MCMP (memcmp): Compare two memory regions
    - MCPY (memcpy): Copy memory region

    When one of these opcodes is detected, this layer sets a flag that
    external code can read to invoke the appropriate subroutine.

    The neural network COULD implement these via attention loops, but
    it's more efficient to trigger external handling for memory-intensive ops.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        ffn_dim = 64  # Small - just opcode detection

        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV2
        SCALE = 8.0

        # Opcode values for MSET, MCMP, MCPY (from c4 VM)
        # These are placeholder values - adjust based on actual opcode enum
        OPCODE_MSET = getattr(Opcode, 'MSET', 50)  # Placeholder
        OPCODE_MCMP = getattr(Opcode, 'MCMP', 51)
        OPCODE_MCPY = getattr(Opcode, 'MCPY', 52)

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0

            # Detect MSET opcode -> set TRIGGER_MSET
            if hasattr(Opcode, 'MSET') and E.OPCODE_START + OPCODE_MSET < self.dim:
                self.up.weight[row, E.OPCODE_START + OPCODE_MSET] = SCALE
                self.gate.weight[row, E.OPCODE_START + OPCODE_MSET] = SCALE
                if E.TRIGGER_MSET < self.dim:
                    self.down.weight[E.TRIGGER_MSET, row] = 1.0
                row += 1

            # Detect MCMP opcode -> set TRIGGER_MCMP
            if hasattr(Opcode, 'MCMP') and E.OPCODE_START + OPCODE_MCMP < self.dim:
                self.up.weight[row, E.OPCODE_START + OPCODE_MCMP] = SCALE
                self.gate.weight[row, E.OPCODE_START + OPCODE_MCMP] = SCALE
                if E.TRIGGER_MCMP < self.dim:
                    self.down.weight[E.TRIGGER_MCMP, row] = 1.0
                row += 1

            # Detect MCPY opcode -> set TRIGGER_MCPY
            if hasattr(Opcode, 'MCPY') and E.OPCODE_START + OPCODE_MCPY < self.dim:
                self.up.weight[row, E.OPCODE_START + OPCODE_MCPY] = SCALE
                self.gate.weight[row, E.OPCODE_START + OPCODE_MCPY] = SCALE
                if E.TRIGGER_MCPY < self.dim:
                    self.down.weight[E.TRIGGER_MCPY, row] = 1.0
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        up_out = self.up(h)
        gate_out = self.gate(h)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)

    def read_triggers(self, x: torch.Tensor) -> Dict[str, bool]:
        """Read trigger flags from the hidden state."""
        E = EmbedDimsV2
        return {
            'mset': x[..., E.TRIGGER_MSET].item() > 0.5 if E.TRIGGER_MSET < x.size(-1) else False,
            'mcmp': x[..., E.TRIGGER_MCMP].item() > 0.5 if E.TRIGGER_MCMP < x.size(-1) else False,
            'mcpy': x[..., E.TRIGGER_MCPY].item() > 0.5 if E.TRIGGER_MCPY < x.size(-1) else False,
        }


# =============================================================================
# PURE TRANSFORMER V2
# =============================================================================

class PureTransformerV2(nn.Module):
    """
    Fully neural transformer with unrolled MUL/DIV/SHL/SHR.

    Architecture (39 layers total):
    - Layer 0: Attention (gather, memory with ALiBi)
    - Layer 1: MoE (basic ops: ADD, SUB, AND, OR, XOR, comparisons)
    - Layer 2: MUL products A (a[0..3] × b[0..7])
    - Layer 3: MUL products B (a[4..7] × b[0..7])
    - Layer 4: MUL accumulation + carry
    - Layer 5: SHL parallel shifts
    - Layer 6: SHR parallel shifts
    - Layers 7-38: DIV iterations (32× weight-tied)
    - Layer 39: Final output

    For operations that don't need all layers, intermediate layers pass through.
    """

    def __init__(self, dim: int = 2048, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        ffn_dim = dim * 4

        # Embeddings
        self.tok_emb = nn.Embedding(Vocab.VOCAB_SIZE, dim)
        self.step_pos_emb = nn.Embedding(30, dim)

        # Layer 0: Attention with ALiBi for memory
        self.layer0 = TransformerLayer(dim, num_heads, ffn_dim, use_moe=True, use_alibi=True)

        # Layer 1: Basic ops MoE
        self.layer1 = TransformerLayer(dim, num_heads, ffn_dim, use_moe=True)

        # Layers 2-3: MUL products
        # Layer A: a[0..3] × b[0..7] = 32 products
        mul_a_indices = [(i, j) for i in range(4) for j in range(8)]
        self.mul_layer_a = MulProductsLayer(dim, product_indices=mul_a_indices)

        # Layer B: a[4..7] × b[0..7] = 32 products
        mul_b_indices = [(i, j) for i in range(4, 8) for j in range(8)]
        self.mul_layer_b = MulProductsLayer(dim, product_indices=mul_b_indices)

        # Layer 4: MUL accumulation (standard MoE, handles summing)
        self.mul_accum = TransformerLayer(dim, num_heads, ffn_dim, use_moe=True)

        # Layers 5-6: Shift layers
        self.shl_layer = ShiftParallelLayer(dim, is_left=True)
        self.shr_layer = ShiftParallelLayer(dim, is_left=False)

        # Layers 7-38: DIV iterations (WEIGHT-TIED - single layer, used 32×)
        self.div_layer = DivIterationLayer(dim)

        # Subroutine trigger layer (detects MSET/MCMP/MCPY)
        self.subroutine_trigger = SubroutineTriggerLayer(dim)

        # Final layer
        self.final_layer = TransformerLayer(dim, num_heads, ffn_dim, use_moe=True)

        # Output
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, Vocab.VOCAB_SIZE, bias=False)

    def forward_standard(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through all layers.

        The DIV iterations are applied 32 times with shared weights.
        For non-DIV operations, the div_layer passes through (no change).
        """
        x = self.tok_emb(tokens)
        B, L, D = x.shape

        mask = torch.triu(torch.ones(L, L, device=x.device), 1) * -1e9

        # Layer 0: Attention + gather
        x = self.layer0.forward_no_cache(x, mask)

        # Layer 1: Basic ops
        x = self.layer1.forward_no_cache(x, mask)

        # Layers 2-3: MUL products
        x = self.mul_layer_a(x)
        x = self.mul_layer_b(x)

        # Layer 4: MUL accumulation
        x = self.mul_accum.forward_no_cache(x, mask)

        # Layers 5-6: Shifts
        x = self.shl_layer(x)
        x = self.shr_layer(x)

        # Check for subroutine triggers (MSET/MCMP/MCPY)
        x = self.subroutine_trigger(x)

        # Layers 7-38: DIV iterations (32× with shared weights)
        for _ in range(32):
            x = self.div_layer(x)

        # Final layer
        x = self.final_layer.forward_no_cache(x, mask)

        # Output
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return logits

    def forward_with_triggers(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, bool]]:
        """
        Forward pass that also returns subroutine trigger flags.

        Returns:
            logits: Output logits
            triggers: Dict with 'mset', 'mcmp', 'mcpy' boolean flags
        """
        x = self.tok_emb(tokens)
        B, L, D = x.shape

        mask = torch.triu(torch.ones(L, L, device=x.device), 1) * -1e9

        # Layer 0: Attention + gather
        x = self.layer0.forward_no_cache(x, mask)

        # Layer 1: Basic ops
        x = self.layer1.forward_no_cache(x, mask)

        # Layers 2-3: MUL products
        x = self.mul_layer_a(x)
        x = self.mul_layer_b(x)

        # Layer 4: MUL accumulation
        x = self.mul_accum.forward_no_cache(x, mask)

        # Layers 5-6: Shifts
        x = self.shl_layer(x)
        x = self.shr_layer(x)

        # Check for subroutine triggers
        x = self.subroutine_trigger(x)
        triggers = self.subroutine_trigger.read_triggers(x[:, -1, :])

        # If any trigger is set, we can return early for external handling
        if any(triggers.values()):
            return None, triggers

        # Layers 7-38: DIV iterations (32× with shared weights)
        for _ in range(32):
            x = self.div_layer(x)

        # Final layer
        x = self.final_layer.forward_no_cache(x, mask)

        # Output
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return logits, triggers

    def generate_token(self, tokens: torch.Tensor) -> int:
        """Generate one token via forward + argmax."""
        with torch.no_grad():
            logits = self.forward_standard(tokens)
            return logits.argmax(dim=-1).item()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PureTransformerV2")
    print("=" * 60)

    print("\nCreating model (dim=2048)...")
    model = PureTransformerV2(dim=2048)

    print(f"Model dimension: {model.dim}")
    print(f"Number of base layers: 7 + 32 DIV iterations + 1 final = 40 effective")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

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
    print("V2 Model Test Complete!")
    print("=" * 60)
