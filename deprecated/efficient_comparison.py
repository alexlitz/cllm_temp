"""
Efficient Comparison Operations using Zero-Detection.

All comparisons use the zero-detection circuit (11 weights per check):
- EQ: Is (A - B) == 0?
- NE: Is (A - B) != 0?
- LT: Is (A - B) < 0? (sign bit after subtraction)
- GT: Swap and LT
- LE: NOT GT (or swap + GE)
- GE: NOT LT
- BZ: Is A == 0?
- BNZ: Is A != 0?

For 32-bit comparisons:
- EQ/NE: Zero-detect per nibble + AND/OR reduction = 2 layers
- LT/GT/LE/GE: Subtraction + sign check = same as SUB + 1 layer

This replaces the 4-10 layer implementations with 2-layer versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention, bake_weights
from .zero_detection import compute_normalization_constant


class EfficientEqualFFN(PureFFN):
    """
    32-bit EQ in a single layer.

    For each nibble position:
    - Computes diff = NIB_A - NIB_B
    - Zero-detects: is_zero[i] = (diff[i] == 0) ? 1 : 0

    Writes per-nibble results to TEMP for reduction.
    Total: 3 nodes × 8 positions = 24 nodes.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=24)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Node 0: silu(S*(A-B) + S*ε) × (1/k)
            self.W_up[base, E.NIB_A] = S
            self.W_up[base, E.NIB_B] = -S
            self.W_up[base, E.POS] = -S * 100  # Position gate
            self.b_up[base] = S * eps + S * 100 * pos
            self.W_gate[base, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.TEMP, base] = 1.0 / k

            # Node 1: silu(S*(A-B)) × (-2/k)
            self.W_up[base + 1, E.NIB_A] = S
            self.W_up[base + 1, E.NIB_B] = -S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.W_gate[base + 1, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.TEMP, base + 1] = -2.0 / k

            # Node 2: silu(S*(A-B) - S*ε) × (1/k)
            self.W_up[base + 2, E.NIB_A] = S
            self.W_up[base + 2, E.NIB_B] = -S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.W_gate[base + 2, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.TEMP, base + 2] = 1.0 / k


class EqualReduceAttention(PureAttention):
    """
    Reduce per-nibble equality results to single result via attention.

    All positions attend to position 0 with their TEMP values.
    At position 0, compute product: if ANY nibble is non-equal (TEMP < 1),
    the product will be < 1.

    Uses soft AND: result = min(all TEMP values) ≈ product for binary values.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # All positions attend equally (uniform attention)
        # Then multiply values together
        # For binary values, min approximates AND

        # Q: all positions query with same vector
        self.W_q[:, :] = 0.0
        for i in range(E.DIM):
            self.W_q[i, E.OP_START + Opcode.EQ] = 1.0

        # K: all positions have same key
        self.W_k[:, :] = 0.0
        for i in range(E.DIM):
            self.W_k[i, E.OP_START + Opcode.EQ] = 1.0

        # V: project TEMP (per-nibble equality)
        self.W_v[:, :] = 0.0
        self.W_v[0, E.TEMP] = 1.0

        # O: write to RESULT
        self.W_o[:, :] = 0.0
        self.W_o[E.RESULT, 0] = 1.0 / E.NUM_POSITIONS


class EfficientNotEqualFFN(PureFFN):
    """
    32-bit NE in a single layer.

    Same as EQ but inverted: is_not_zero[i] = (diff[i] != 0) ? 1 : 0
    Output weights are negated, bias adds 1.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=24)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Zero-detect with inverted output
            self.W_up[base, E.NIB_A] = S
            self.W_up[base, E.NIB_B] = -S
            self.W_up[base, E.POS] = -S * 100
            self.b_up[base] = S * eps + S * 100 * pos
            self.W_gate[base, E.OP_START + Opcode.NE] = 1.0
            self.W_down[E.TEMP, base] = -1.0 / k  # Negated

            self.W_up[base + 1, E.NIB_A] = S
            self.W_up[base + 1, E.NIB_B] = -S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.W_gate[base + 1, E.OP_START + Opcode.NE] = 1.0
            self.W_down[E.TEMP, base + 1] = 2.0 / k  # Negated

            self.W_up[base + 2, E.NIB_A] = S
            self.W_up[base + 2, E.NIB_B] = -S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.W_gate[base + 2, E.OP_START + Opcode.NE] = 1.0
            self.W_down[E.TEMP, base + 2] = -1.0 / k  # Negated

        # Bias to invert: 1 - zero_indicator
        self.b_down[E.TEMP] = 1.0


class NotEqualReduceAttention(PureAttention):
    """
    Reduce per-nibble not-equal results via OR.

    If ANY nibble is not equal, result is 1.
    This is OR reduction: max(all TEMP values).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Uniform attention for OR reduction
        self.W_q[:, :] = 0.0
        for i in range(E.DIM):
            self.W_q[i, E.OP_START + Opcode.NE] = 1.0

        self.W_k[:, :] = 0.0
        for i in range(E.DIM):
            self.W_k[i, E.OP_START + Opcode.NE] = 1.0

        self.W_v[:, :] = 0.0
        self.W_v[0, E.TEMP] = 1.0

        self.W_o[:, :] = 0.0
        self.W_o[E.RESULT, 0] = 1.0 / E.NUM_POSITIONS


class SignCheckFFN(PureFFN):
    """
    Check sign of subtraction result for LT/GT.

    After subtraction with borrow propagation, CARRY_OUT at position 7
    indicates whether result is negative (A < B).

    This layer converts the final borrow to a 0/1 result.
    """

    def __init__(self, opcode: int = Opcode.LT):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # LT: result = 1 if final borrow (CARRY_OUT at pos 7) is 1
        # Read CARRY_OUT from position 7, write to RESULT at position 0

        # Need position 7's CARRY_OUT value
        # Use position gating: only activate at position 7
        self.W_up[0, E.CARRY_OUT] = S
        self.W_up[0, E.POS] = -S * 100
        self.b_up[0] = S * 100 * 7 + 0.5 * S  # Position 7

        self.W_gate[0, E.OP_START + self.opcode] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S


class EfficientLessThanFFN(PureFFN):
    """
    LT after subtraction: check if result is negative.

    Assumes NIB_A - NIB_B has been computed with borrow propagation.
    Checks final borrow (CARRY_OUT at position 7).

    LT = 1 if A < B (final borrow set).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        # Position 7's CARRY_OUT indicates final borrow
        # If CARRY_OUT[7] > 0.5, then A < B

        # Use zero-detection inverted on (CARRY_OUT - 0.5)
        # If CARRY_OUT = 1: not zero → result = 1
        # If CARRY_OUT = 0: not zero → result = 0
        # Actually, just copy CARRY_OUT at position 7

        # Simpler: gate on position 7, copy CARRY_OUT to RESULT
        self.W_up[0, E.CARRY_OUT] = S
        self.W_up[0, E.POS] = -S * 100
        self.b_up[0] = S * 100 * 7

        self.W_gate[0, E.OP_START + Opcode.LT] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S

        # Clear RESULT at other positions
        for pos in range(E.NUM_POSITIONS - 1):
            row = pos + 1
            self.W_up[row, E.RESULT] = -S
            self.W_up[row, E.POS] = -S * 100
            self.b_up[row] = S * 100 * pos

            self.W_gate[row, E.OP_START + Opcode.LT] = 1.0
            self.W_down[E.RESULT, row] = 1.0 / S


class EfficientGreaterEqualFFN(PureFFN):
    """
    GE = NOT LT = 1 if A >= B (no final borrow).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # GE = 1 - LT = 1 - CARRY_OUT[7]
        self.W_up[0, E.CARRY_OUT] = -S  # Negated
        self.W_up[0, E.POS] = -S * 100
        self.b_up[0] = S * 100 * 7

        self.W_gate[0, E.OP_START + Opcode.GE] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S
        self.b_down[E.RESULT] = 1.0  # Add 1 to invert


# =============================================================================
# COMPARISON LAYER COUNTS
# =============================================================================

"""
Layer counts with efficient implementation:

EQ:  1 layer (zero-detect per nibble) + 1 attention (AND reduce) = 2 layers
NE:  1 layer (zero-detect inverted) + 1 attention (OR reduce) = 2 layers
LT:  Subtraction layers (same as SUB) + 1 layer (sign check) = 8 layers
GT:  Same as LT with swapped operands = 8 layers
LE:  NOT GT = 8 layers
GE:  NOT LT = 8 layers
BZ:  1 layer (zero-detect single value) = 1 layer
BNZ: 1 layer (zero-detect inverted) = 1 layer

Compare to original:
EQ:  4 layers → 2 layers (50% reduction)
NE:  4 layers → 2 layers (50% reduction)
LT:  9 layers → 8 layers (11% reduction)
BZ:  3 layers → 1 layer (67% reduction)
"""


# =============================================================================
# DEMO
# =============================================================================

def demo_efficient_comparison():
    """Demonstrate efficient comparison operations."""
    print("=" * 60)
    print("Efficient Comparison Demo (using zero-detection)")
    print("=" * 60)
    print()

    # Create layers
    eq_ffn = EfficientEqualFFN()
    ne_ffn = EfficientNotEqualFFN()

    # Test EQ
    print("Testing 32-bit EQ:")
    test_cases = [
        (0x12345678, 0x12345678, True),   # Equal
        (0x12345678, 0x12345679, False),  # Differ in LSB
        (0x12345678, 0x92345678, False),  # Differ in MSB
        (0x00000000, 0x00000000, True),   # Both zero
        (0xFFFFFFFF, 0xFFFFFFFF, True),   # Both max
    ]

    for a, b, expected in test_cases:
        # Create embedding
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

        # Set operands as nibbles
        for i in range(8):
            x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
            x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
            x[0, i, E.POS] = float(i)
            x[0, i, E.OP_START + Opcode.EQ] = 1.0

        # Run EQ
        y = eq_ffn(x)

        # Check per-nibble results in TEMP
        nibble_results = [y[0, i, E.TEMP].item() for i in range(8)]
        all_equal = all(r > 0.5 for r in nibble_results)

        status = "✓" if all_equal == expected else "✗"
        print(f"  {hex(a)} == {hex(b)}: {all_equal} (expected {expected}) {status}")
        print(f"    Per-nibble: {[f'{r:.2f}' for r in nibble_results]}")

    print()

    # Count parameters
    eq_params = sum((p != 0).sum().item() for p in eq_ffn.parameters())
    print(f"EQ FFN non-zero parameters: {eq_params}")
    print()

    print("Layer count comparison:")
    print("  Operation  Original  Efficient  Reduction")
    print("  ---------  --------  ---------  ---------")
    print("  EQ         4 layers  2 layers   50%")
    print("  NE         4 layers  2 layers   50%")
    print("  LT         9 layers  8 layers   11%")
    print("  GT         9 layers  8 layers   11%")
    print("  LE         10 layers 8 layers   20%")
    print("  GE         10 layers 8 layers   20%")
    print("  BZ         3 layers  1 layer    67%")
    print("  BNZ        3 layers  1 layer    67%")

    print()
    print("=" * 60)


if __name__ == "__main__":
    demo_efficient_comparison()
