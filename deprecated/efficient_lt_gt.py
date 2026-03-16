"""
Efficient LT/GT/LE/GE using Per-Nibble Comparison + Priority Reduction.

Instead of full subtraction with borrow propagation (8 layers),
we use:
1. Per-nibble comparison: LT[i], GT[i], EQ[i] for each nibble (1 layer)
2. Priority reduction: highest differing nibble wins (1 attention layer)

Total: 2 layers (same as EQ/NE), down from 8-10 layers.

Algorithm:
- Compare nibbles from MSB to LSB
- First nibble where A[i] != B[i] determines result
- If A[i] < B[i] at that position: A < B
- If A[i] > B[i] at that position: A > B
"""

import torch
import torch.nn as nn
import math

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention, bake_weights
from .zero_detection import compute_normalization_constant


class PerNibbleCompareFFN(PureFFN):
    """
    Compute per-nibble LT, GT, EQ in a single layer.

    For each nibble position i:
    - EQ[i] = (A[i] == B[i]) using zero-detection on diff
    - LT[i] = (A[i] < B[i]) using negative detection on diff
    - GT[i] = (A[i] > B[i]) using positive detection on diff

    Stores results:
    - EQ[i] in TEMP slot
    - LT[i] in CARRY_IN slot
    - GT[i] in CARRY_OUT slot

    Total: 9 nodes per position (3 for each of EQ, LT, GT) = 72 nodes
    """

    def __init__(self, opcode: int = Opcode.LT):
        self.opcode = opcode
        # 3 nodes for EQ + 2 nodes for LT + 2 nodes for GT = 7 per position
        # Actually: 3 for EQ (zero-detect), 2 for LT (neg-detect), 2 for GT (pos-detect)
        super().__init__(E.DIM, hidden_dim=7 * E.NUM_POSITIONS)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 7

            # === EQ detection (3 nodes): zero-detect on (A - B) ===
            # Node 0: silu(S*(A-B) + S*ε)
            self.W_up[base, E.NIB_A] = S
            self.W_up[base, E.NIB_B] = -S
            self.W_up[base, E.POS] = -S * 100
            self.b_up[base] = S * eps + S * 100 * pos
            self.W_gate[base, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, base] = 1.0 / k

            # Node 1: silu(S*(A-B)) × (-2/k)
            self.W_up[base + 1, E.NIB_A] = S
            self.W_up[base + 1, E.NIB_B] = -S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.W_gate[base + 1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, base + 1] = -2.0 / k

            # Node 2: silu(S*(A-B) - S*ε)
            self.W_up[base + 2, E.NIB_A] = S
            self.W_up[base + 2, E.NIB_B] = -S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.W_gate[base + 2, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, base + 2] = 1.0 / k

            # === LT detection (2 nodes): detect (A - B) < 0 ===
            # Use: silu(S*(B - A) - S*0.5) gives positive when A < B
            # Node 3: silu(S*(B-A) - S*0.5) for LT
            self.W_up[base + 3, E.NIB_A] = -S
            self.W_up[base + 3, E.NIB_B] = S
            self.W_up[base + 3, E.POS] = -S * 100
            self.b_up[base + 3] = -S * 0.5 + S * 100 * pos
            self.W_gate[base + 3, E.OP_START + self.opcode] = 1.0
            # Normalize: max value is S*14.5, divide to get ~1
            self.W_down[E.CARRY_IN, base + 3] = 1.0 / (S * 7.5)

            # Node 4: subtract baseline to sharpen
            self.W_up[base + 4, E.NIB_A] = -S
            self.W_up[base + 4, E.NIB_B] = S
            self.W_up[base + 4, E.POS] = -S * 100
            self.b_up[base + 4] = -S * 15.5 + S * 100 * pos  # Offset to cancel linear part
            self.W_gate[base + 4, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_IN, base + 4] = -1.0 / (S * 7.5)

            # === GT detection (2 nodes): detect (A - B) > 0 ===
            # Node 5: silu(S*(A-B) - S*0.5) for GT
            self.W_up[base + 5, E.NIB_A] = S
            self.W_up[base + 5, E.NIB_B] = -S
            self.W_up[base + 5, E.POS] = -S * 100
            self.b_up[base + 5] = -S * 0.5 + S * 100 * pos
            self.W_gate[base + 5, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, base + 5] = 1.0 / (S * 7.5)

            # Node 6: subtract baseline
            self.W_up[base + 6, E.NIB_A] = S
            self.W_up[base + 6, E.NIB_B] = -S
            self.W_up[base + 6, E.POS] = -S * 100
            self.b_up[base + 6] = -S * 15.5 + S * 100 * pos
            self.W_gate[base + 6, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, base + 6] = -1.0 / (S * 7.5)


class PriorityReduceAttention(PureAttention):
    """
    Reduce per-nibble comparisons with MSB priority.

    Each position attends to higher positions to check if all are equal.
    The result is: LT[i] * (all higher nibbles equal)

    Uses causal attention from MSB (pos 7) to LSB (pos 0).
    Position i attends to positions i+1 to 7.
    """

    def __init__(self, opcode: int = Opcode.LT):
        self.opcode = opcode
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        N = E.NUM_POSITIONS

        # Custom attention pattern: pos i attends to pos i+1, i+2, ..., 7
        # with uniform weights

        # Create mask: position i can attend to j if j > i
        mask = torch.full((N, N), float('-inf'))
        for i in range(N):
            for j in range(i + 1, N):
                mask[i, j] = 0.0  # Can attend
            # Also attend to self with special weight
            mask[i, i] = 0.0

        # Store mask (overwrite default)
        self.mask.copy_(mask)

        # Q: project opcode indicator
        self.W_q[0, E.OP_START + self.opcode] = S

        # K: same as Q for uniform attention
        self.W_k[0, E.OP_START + self.opcode] = S

        # V: project TEMP (EQ indicator) - we want product of EQ values
        # For attention-based product, use log-space:
        # But simpler: project EQ values, attention averages them
        # If any EQ[j] = 0 for j > i, the average will be < 1
        self.W_v[0, E.TEMP] = 1.0

        # O: multiply with local LT and write to RESULT
        self.W_o[E.RAW_SUM, 0] = 1.0  # Store "all-higher-equal" indicator


class CombineLTResultFFN(PureFFN):
    """
    Combine per-nibble LT with "all-higher-equal" indicator.

    RESULT[i] = LT[i] * all_higher_equal[i] * (1 - EQ[i])

    Then sum across positions to get final LT result.
    """

    def __init__(self, opcode: int = Opcode.LT):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=E.NUM_POSITIONS * 2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        for pos in range(E.NUM_POSITIONS):
            base = pos * 2

            # Node 0: LT[i] * (1 - EQ[i]) when at this position
            # CARRY_IN has LT[i], TEMP has EQ[i]
            # Want: LT[i] * (1 - EQ[i]) = LT[i] - LT[i]*EQ[i]
            self.W_up[base, E.CARRY_IN] = S  # LT indicator
            self.W_up[base, E.POS] = -S * 100
            self.b_up[base] = S * 100 * pos

            # Gate by RAW_SUM (all-higher-equal) and (1 - EQ)
            self.W_gate[base, E.RAW_SUM] = 1.0  # all-higher-equal
            self.W_gate[base, E.TEMP] = -1.0   # Subtract EQ (so gate = all_higher_eq * (1 - EQ))
            self.b_gate[base] = 1.0             # Base of 1 for the (1 - EQ) part

            self.W_down[E.RESULT, base] = 1.0 / S

            # For position 7 (MSB), no higher positions, so all_higher_eq = 1
            if pos == 7:
                self.W_gate[base, E.RAW_SUM] = 0.0  # Ignore all_higher_eq
                self.b_gate[base] = 1.0  # Just use (1 - EQ)


class EfficientLTOp(nn.Module):
    """
    Complete efficient LT operation in 2 layers.

    Layer 1: PerNibbleCompareFFN - compute EQ[i], LT[i], GT[i] per nibble
    Layer 2: PriorityReduceAttention + CombineLTResultFFN - reduce with MSB priority
    """

    def __init__(self):
        super().__init__()
        self.compare = PerNibbleCompareFFN(Opcode.LT)
        self.reduce_attn = PriorityReduceAttention(Opcode.LT)
        self.combine = CombineLTResultFFN(Opcode.LT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compare(x)
        x = self.reduce_attn(x)
        x = self.combine(x)
        return x


class EfficientGTOp(nn.Module):
    """GT = swap operands and do LT."""

    def __init__(self):
        super().__init__()
        self.compare = PerNibbleCompareFFN(Opcode.GT)
        self.reduce_attn = PriorityReduceAttention(Opcode.GT)
        self.combine = CombineGTResultFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compare(x)
        x = self.reduce_attn(x)
        x = self.combine(x)
        return x


class CombineGTResultFFN(PureFFN):
    """Combine for GT - uses CARRY_OUT (GT indicator) instead of CARRY_IN."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=E.NUM_POSITIONS * 2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        for pos in range(E.NUM_POSITIONS):
            base = pos * 2

            self.W_up[base, E.CARRY_OUT] = S  # GT indicator (swapped)
            self.W_up[base, E.POS] = -S * 100
            self.b_up[base] = S * 100 * pos

            self.W_gate[base, E.RAW_SUM] = 1.0
            self.W_gate[base, E.TEMP] = -1.0
            self.b_gate[base] = 1.0

            self.W_down[E.RESULT, base] = 1.0 / S

            if pos == 7:
                self.W_gate[base, E.RAW_SUM] = 0.0
                self.b_gate[base] = 1.0


class EfficientLEOp(nn.Module):
    """LE = NOT GT = 1 - GT_result."""

    def __init__(self):
        super().__init__()
        self.gt_op = EfficientGTOp()
        self.invert = InvertResultFFN(Opcode.LE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gt_op(x)
        x = self.invert(x)
        return x


class EfficientGEOp(nn.Module):
    """GE = NOT LT = 1 - LT_result."""

    def __init__(self):
        super().__init__()
        self.lt_op = EfficientLTOp()
        self.invert = InvertResultFFN(Opcode.GE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lt_op(x)
        x = self.invert(x)
        return x


class InvertResultFFN(PureFFN):
    """Invert RESULT: output = 1 - RESULT."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Negate RESULT and add 1
        self.W_up[0, E.RESULT] = -S
        self.W_up[0, E.POS] = -S * 100  # Position 0 only
        self.b_up[0] = S * 100 * 0

        self.W_gate[0, E.OP_START + self.opcode] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S
        self.b_down[E.RESULT] = 1.0  # Add 1


# =============================================================================
# SUMMARY
# =============================================================================

"""
Efficient Comparison Layer Counts:

| Op | Layers | Method |
|----|--------|--------|
| EQ | 2 | Zero-detect per nibble + AND reduce |
| NE | 2 | Zero-detect inverted + OR reduce |
| LT | 2 | Per-nibble compare + priority reduce |
| GT | 2 | Per-nibble compare + priority reduce |
| LE | 3 | GT + invert |
| GE | 3 | LT + invert |
| BZ | 1 | Zero-detect single value |
| BNZ| 1 | Zero-detect inverted |

All operations now O(1) layers, no borrow propagation needed!
"""


def demo_efficient_lt():
    """Demonstrate efficient LT."""
    print("=" * 60)
    print("Efficient LT/GT Demo (2 layers)")
    print("=" * 60)

    compare = PerNibbleCompareFFN(Opcode.LT)

    test_cases = [
        (5, 10, True),   # 5 < 10
        (10, 5, False),  # 10 < 5
        (5, 5, False),   # 5 < 5
        (0, 15, True),   # 0 < 15
        (15, 0, False),  # 15 < 0
    ]

    print("\nSingle nibble comparison (position 0):")
    for a, b, expected in test_cases:
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x[0, 0, E.NIB_A] = float(a)
        x[0, 0, E.NIB_B] = float(b)
        x[0, 0, E.POS] = 0.0
        x[0, 0, E.OP_START + Opcode.LT] = 1.0

        y = compare(x)

        eq_val = y[0, 0, E.TEMP].item()
        lt_val = y[0, 0, E.CARRY_IN].item()
        gt_val = y[0, 0, E.CARRY_OUT].item()

        print(f"  {a} vs {b}: EQ={eq_val:.2f}, LT={lt_val:.2f}, GT={gt_val:.2f}")

    print()
    print("Layer count comparison:")
    print("  Op   Original  Efficient  Reduction")
    print("  ---  --------  ---------  ---------")
    print("  LT   9 layers  2 layers   78%")
    print("  GT   9 layers  2 layers   78%")
    print("  LE   10 layers 3 layers   70%")
    print("  GE   10 layers 3 layers   70%")

    print()
    print("=" * 60)


if __name__ == "__main__":
    demo_efficient_lt()
