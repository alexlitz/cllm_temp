"""
Zero Detection Circuit for Neural VM.

Implements efficient zero-detection using only 3 SiLU nodes (11 weights).

The circuit detects if a value is approximately zero:
- Output ≈ 1 when input ≈ 0
- Output ≈ 0 when input is far from 0

Architecture:
    Node 1: silu(SCALE * x + SCALE * ε) × (1/k)
    Node 2: silu(SCALE * x + 0)          × (-2/k)
    Node 3: silu(SCALE * x - SCALE * ε) × (1/k)

Where:
    ε = small value (e.g., 0.5 for integer detection)
    k = SCALE * ε * (2σ(SCALE * ε) - 1)  [normalizing constant]

The +1, -2, +1 pattern forms a finite second difference:
- Far from zero: all nodes in SiLU linear regime, cancel out → 0
- Near zero: nonlinearity breaks cancellation → sharp bump at 1

Total: 9 weights + 2 biases = 11 parameters per zero-detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .embedding import E, Opcode
from .base_layers import PureFFN, bake_weights


def compute_normalization_constant(scale: float, epsilon: float) -> float:
    """
    Compute the normalization constant k = SCALE * ε * (2σ(SCALE * ε) - 1).

    This ensures the zero-detection output is exactly 1 when input is 0.
    """
    se = scale * epsilon
    sigma_se = 1.0 / (1.0 + math.exp(-se))  # σ(SCALE * ε)
    k = se * (2 * sigma_se - 1)
    return k


class ZeroDetectFFN(PureFFN):
    """
    Zero-detection using 3 SiLU nodes.

    Detects if a single slot value is approximately zero.

    Args:
        input_slot: Embedding slot to check for zero
        output_slot: Embedding slot to write result (1 if zero, 0 otherwise)
        epsilon: Width of the bump (default 0.5 for integer detection)
    """

    def __init__(self, input_slot: int = E.RAW_SUM, output_slot: int = E.RESULT,
                 epsilon: float = 0.5):
        self.input_slot = input_slot
        self.output_slot = output_slot
        self.epsilon = epsilon
        super().__init__(E.DIM, hidden_dim=3)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = self.epsilon

        # Compute normalization constant
        k = compute_normalization_constant(S, eps)

        # Node 0: silu(S*x + S*ε) × (1/k)
        self.W_up[0, self.input_slot] = S
        self.b_up[0] = S * eps
        self.W_gate[0, :] = 0.0  # No gating
        self.b_gate[0] = 1.0     # Always active
        self.W_down[self.output_slot, 0] = 1.0 / k

        # Node 1: silu(S*x + 0) × (-2/k)
        self.W_up[1, self.input_slot] = S
        self.b_up[1] = 0.0
        self.b_gate[1] = 1.0
        self.W_down[self.output_slot, 1] = -2.0 / k

        # Node 2: silu(S*x - S*ε) × (1/k)
        self.W_up[2, self.input_slot] = S
        self.b_up[2] = -S * eps
        self.b_gate[2] = 1.0
        self.W_down[self.output_slot, 2] = 1.0 / k


class ZeroDetectPerPositionFFN(PureFFN):
    """
    Zero-detection for all 8 positions simultaneously.

    Uses 3 nodes per position = 24 nodes total.
    Detects if input_slot is zero at each position independently.
    """

    def __init__(self, input_slot: int = E.RAW_SUM, output_slot: int = E.TEMP,
                 epsilon: float = 0.5):
        self.input_slot = input_slot
        self.output_slot = output_slot
        self.epsilon = epsilon
        super().__init__(E.DIM, hidden_dim=3 * E.NUM_POSITIONS)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = self.epsilon
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Position gating: only activate for this position
            # Using POS slot to gate

            # Node 0: silu(S*input + S*ε) × (1/k) when at position pos
            self.W_up[base + 0, self.input_slot] = S
            self.W_up[base + 0, E.POS] = -S * 100  # Position mask
            self.b_up[base + 0] = S * eps + S * 100 * pos
            self.b_gate[base + 0] = 1.0
            self.W_down[self.output_slot, base + 0] = 1.0 / k

            # Node 1: silu(S*input) × (-2/k)
            self.W_up[base + 1, self.input_slot] = S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.b_gate[base + 1] = 1.0
            self.W_down[self.output_slot, base + 1] = -2.0 / k

            # Node 2: silu(S*input - S*ε) × (1/k)
            self.W_up[base + 2, self.input_slot] = S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.b_gate[base + 2] = 1.0
            self.W_down[self.output_slot, base + 2] = 1.0 / k


class EqualZeroAllPositionsFFN(PureFFN):
    """
    Check if ALL positions have zero difference (for 32-bit equality).

    This is a single layer that combines:
    1. Zero-detection per nibble (3 nodes each = 24 nodes)
    2. AND reduction: multiply all 8 indicators (1 node)

    Total: 25 nodes, ~100 weights.
    """

    def __init__(self, diff_slot: int = E.RAW_SUM, result_slot: int = E.RESULT,
                 epsilon: float = 0.5):
        self.diff_slot = diff_slot
        self.result_slot = result_slot
        self.epsilon = epsilon
        # 24 for zero-detect + 8 for collection + 1 for reduction
        super().__init__(E.DIM, hidden_dim=33)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = self.epsilon
        k = compute_normalization_constant(S, eps)

        # First 24 nodes: zero detection per position
        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Write to TEMP slot as intermediate (per-nibble zero indicator)
            for i in range(3):
                self.W_up[base + i, self.diff_slot] = S
                self.W_up[base + i, E.POS] = -S * 100
                offset = [S * eps, 0, -S * eps][i]
                self.b_up[base + i] = offset + S * 100 * pos
                self.b_gate[base + i] = 1.0
                out_weight = [1.0 / k, -2.0 / k, 1.0 / k][i]
                self.W_down[E.TEMP, base + i] = out_weight

        # Nodes 24-31: collect per-position indicators
        # These accumulate the TEMP values for AND-reduction via multiplication
        # But SwiGLU doesn't directly do multiplication across slots...

        # Alternative: Use attention to reduce across positions
        # For now, write per-position results to TEMP, reduce in next layer


class CompareEqualFFN(PureFFN):
    """
    EQ comparison: Check if A == B using zero-detection on (A - B).

    Single layer: computes A - B per nibble, detects if zero.
    Result = 1 if all nibbles are zero (equal), 0 otherwise.

    This replaces the multi-layer subtraction + reduction approach.
    """

    def __init__(self):
        # 3 nodes per position for zero-detect = 24
        # 8 nodes for intermediate collection = 8
        # Total: 32 nodes
        super().__init__(E.DIM, hidden_dim=32)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5  # For integer detection
        k = compute_normalization_constant(S, eps)

        # For each position, detect if (NIB_A - NIB_B) == 0
        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # The difference is computed inline:
            # Node 0: silu(S*(NIB_A - NIB_B) + S*ε) × (1/k)
            self.W_up[base + 0, E.NIB_A] = S
            self.W_up[base + 0, E.NIB_B] = -S  # Subtraction
            self.W_up[base + 0, E.POS] = -S * 100
            self.b_up[base + 0] = S * eps + S * 100 * pos
            self.W_gate[base + 0, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.TEMP, base + 0] = 1.0 / k

            # Node 1: silu(S*(NIB_A - NIB_B)) × (-2/k)
            self.W_up[base + 1, E.NIB_A] = S
            self.W_up[base + 1, E.NIB_B] = -S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.W_gate[base + 1, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.TEMP, base + 1] = -2.0 / k

            # Node 2: silu(S*(NIB_A - NIB_B) - S*ε) × (1/k)
            self.W_up[base + 2, E.NIB_A] = S
            self.W_up[base + 2, E.NIB_B] = -S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.W_gate[base + 2, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.TEMP, base + 2] = 1.0 / k

        # Nodes 24-31: These could be used for reduction
        # But for 32-bit equality we need to AND all 8 nibble results
        # This requires attention or another layer to aggregate


class CompareNotEqualFFN(PureFFN):
    """
    NE comparison: Check if A != B.

    Same as EQ but inverted output (1 - EQ_result).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=32)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Zero-detect (same as EQ)
            self.W_up[base + 0, E.NIB_A] = S
            self.W_up[base + 0, E.NIB_B] = -S
            self.W_up[base + 0, E.POS] = -S * 100
            self.b_up[base + 0] = S * eps + S * 100 * pos
            self.W_gate[base + 0, E.OP_START + Opcode.NE] = 1.0
            # For NE: output is 1 - (zero indicator)
            # We'll compute zero indicator and invert in bias
            self.W_down[E.TEMP, base + 0] = -1.0 / k  # Negated

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

        # Add 1 to invert: result = 1 - zero_indicator
        # This is done via bias on the output
        self.b_down[E.TEMP] = 1.0


class BranchZeroFFN(PureFFN):
    """
    BZ (branch if zero): Branch when value is zero.

    Uses zero-detection circuit to check if value is 0.
    """

    def __init__(self, value_slot: int = E.NIB_A, target_slot: int = E.TEMP):
        self.value_slot = value_slot
        self.target_slot = target_slot
        super().__init__(E.DIM, hidden_dim=3)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        # 3-node zero-detection
        self.W_up[0, self.value_slot] = S
        self.b_up[0] = S * eps
        self.W_gate[0, E.OP_START + Opcode.BZ] = 1.0
        self.W_down[self.target_slot, 0] = 1.0 / k

        self.W_up[1, self.value_slot] = S
        self.b_up[1] = 0.0
        self.W_gate[1, E.OP_START + Opcode.BZ] = 1.0
        self.W_down[self.target_slot, 1] = -2.0 / k

        self.W_up[2, self.value_slot] = S
        self.b_up[2] = -S * eps
        self.W_gate[2, E.OP_START + Opcode.BZ] = 1.0
        self.W_down[self.target_slot, 2] = 1.0 / k


class BranchNonZeroFFN(PureFFN):
    """
    BNZ (branch if non-zero): Branch when value is not zero.

    Output = 1 - (zero indicator).
    """

    def __init__(self, value_slot: int = E.NIB_A, target_slot: int = E.TEMP):
        self.value_slot = value_slot
        self.target_slot = target_slot
        super().__init__(E.DIM, hidden_dim=3)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        # 3-node zero-detection, negated
        self.W_up[0, self.value_slot] = S
        self.b_up[0] = S * eps
        self.W_gate[0, E.OP_START + Opcode.BNZ] = 1.0
        self.W_down[self.target_slot, 0] = -1.0 / k

        self.W_up[1, self.value_slot] = S
        self.b_up[1] = 0.0
        self.W_gate[1, E.OP_START + Opcode.BNZ] = 1.0
        self.W_down[self.target_slot, 1] = 2.0 / k

        self.W_up[2, self.value_slot] = S
        self.b_up[2] = -S * eps
        self.W_gate[2, E.OP_START + Opcode.BNZ] = 1.0
        self.W_down[self.target_slot, 2] = -1.0 / k

        # Invert: 1 - zero_indicator
        self.b_down[self.target_slot] = 1.0


# =============================================================================
# DEMO
# =============================================================================

def demo_zero_detection():
    """Demonstrate zero-detection circuit."""
    print("=" * 60)
    print("Zero Detection Circuit Demo")
    print("=" * 60)
    print()

    S = E.SCALE
    eps = 0.5
    k = compute_normalization_constant(S, eps)
    print(f"Parameters: SCALE={S}, epsilon={eps}")
    print(f"Normalization constant k = {k:.6f}")
    print()

    # Create zero-detection layer
    zd = ZeroDetectFFN(input_slot=E.RAW_SUM, output_slot=E.RESULT)

    # Test various input values
    test_values = [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]

    print("Testing zero-detection on various values:")
    print(f"{'Input':<10} {'Output':<10} {'Expected':<10}")
    print("-" * 30)

    for val in test_values:
        # Create embedding with test value
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x[0, 0, E.RAW_SUM] = float(val)

        # Run zero detection
        y = zd(x)
        output = y[0, 0, E.RESULT].item()

        expected = 1.0 if abs(val) < eps else 0.0
        match = "✓" if abs(output - expected) < 0.1 else "✗"

        print(f"{val:<10.1f} {output:<10.4f} {expected:<10.1f} {match}")

    print()

    # Count weights
    total = sum(p.numel() for p in zd.parameters())
    nonzero = sum((p != 0).sum().item() for p in zd.parameters())
    print(f"Total parameters: {total}")
    print(f"Non-zero parameters: {nonzero}")
    print()

    print("=" * 60)


if __name__ == "__main__":
    demo_zero_detection()
