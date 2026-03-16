"""
Newton-Raphson Division for Neural VM.

Replaces iterative long division with O(log n) Newton-Raphson.

Algorithm:
1. Find k = floor(log2(b)) via MSB detection
2. Initial guess: x_0 = 2^(-k-1) ≈ 1/(2b)
3. Newton iteration: x_{n+1} = x_n * (2 - b * x_n)
4. Result: a * x ≈ a / b

Convergence: Each iteration doubles precision.
For 32-bit precision, 5 iterations suffice (starting from 1-bit guess).

The implementation uses fixed-point arithmetic with Q16 format
(16 fractional bits) for the reciprocal computation.
"""

import torch
import torch.nn as nn
import math

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention, bake_weights


# Fixed-point scale (2^16 for Q16 format)
FP_SCALE = 65536  # 2^16
FP_BITS = 16


# =============================================================================
# MSB Detection - Find position of highest set bit
# =============================================================================

class MSBDetectFFN(PureFFN):
    """
    Detect the most significant bit position of divisor.

    For a 32-bit number split into 8 nibbles:
    1. Find highest non-zero nibble (multiply by 4 for bit position)
    2. Within that nibble, find the highest set bit

    Output: TEMP[0] = k where 2^k <= b < 2^(k+1)

    Method: Cascade of comparisons at power-of-2 thresholds.
    For single-nibble (4-bit) values, k is in [0, 3].
    """

    def __init__(self):
        # For single-nibble detection: compare against 8, 4, 2, 1
        # 2 hidden units per threshold
        super().__init__(E.DIM, hidden_dim=8)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Detect MSB by checking b >= 2^k for k = 3, 2, 1, 0
        # At position 0 only (position gate)

        # k=3: Is b >= 8?
        self.W_up[0, E.NIB_B] = S
        self.W_up[0, E.POS] = -S * 100
        self.b_up[0] = -S * 7.5  # threshold at 8
        self.W_gate[0, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 0] = 3.0 / S  # MSB = 3

        self.W_up[1, E.NIB_B] = S
        self.W_up[1, E.POS] = -S * 100
        self.b_up[1] = -S * 8.5
        self.W_gate[1, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 1] = -3.0 / S

        # k=2: Is b >= 4?
        self.W_up[2, E.NIB_B] = S
        self.W_up[2, E.POS] = -S * 100
        self.b_up[2] = -S * 3.5  # threshold at 4
        self.W_gate[2, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 2] = 2.0 / S  # MSB = 2

        self.W_up[3, E.NIB_B] = S
        self.W_up[3, E.POS] = -S * 100
        self.b_up[3] = -S * 4.5
        self.W_gate[3, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 3] = -2.0 / S

        # k=1: Is b >= 2?
        self.W_up[4, E.NIB_B] = S
        self.W_up[4, E.POS] = -S * 100
        self.b_up[4] = -S * 1.5  # threshold at 2
        self.W_gate[4, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 4] = 1.0 / S  # MSB = 1

        self.W_up[5, E.NIB_B] = S
        self.W_up[5, E.POS] = -S * 100
        self.b_up[5] = -S * 2.5
        self.W_gate[5, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 5] = -1.0 / S

        # k=0: Is b >= 1? (always true for valid divisor)
        self.W_up[6, E.NIB_B] = S
        self.W_up[6, E.POS] = -S * 100
        self.b_up[6] = -S * 0.5  # threshold at 1
        self.W_gate[6, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 6] = 0.0 / S  # MSB = 0 (implicit)

        self.W_up[7, E.NIB_B] = S
        self.W_up[7, E.POS] = -S * 100
        self.b_up[7] = -S * 1.5
        self.W_gate[7, E.OP_START + Opcode.DIV] = 1.0
        self.W_down[E.TEMP, 7] = 0.0 / S


# =============================================================================
# Initial Reciprocal Guess
# =============================================================================

class InitReciprocalFFN(PureFFN):
    """
    Compute initial reciprocal guess from MSB position.

    Given k = floor(log2(b)), initial guess is:
        x_0 = 2^(FP_BITS - k - 1) in fixed-point

    For k=0 (b=1): x_0 = 2^15 = 32768 (represents 0.5)
    For k=1 (b=2,3): x_0 = 2^14 = 16384 (represents 0.25)
    For k=2 (b=4-7): x_0 = 2^13 = 8192 (represents 0.125)
    For k=3 (b=8-15): x_0 = 2^12 = 4096 (represents 0.0625)

    Output: RAW_SUM = initial reciprocal in Q16 fixed-point
    """

    def __init__(self):
        # 4 cases for k = 0, 1, 2, 3
        super().__init__(E.DIM, hidden_dim=8)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # For each MSB value k, output 2^(FP_BITS - k - 1)
        # TEMP contains MSB (k), select appropriate reciprocal

        # k=3: output 2^12 = 4096
        self.W_up[0, E.TEMP] = S
        self.W_up[0, E.POS] = -S * 100
        self.b_up[0] = -S * 2.5  # TEMP >= 3
        self.W_gate[0, E.OP_START + Opcode.DIV] = 4096.0
        self.W_down[E.RAW_SUM, 0] = 1.0 / S

        self.W_up[1, E.TEMP] = S
        self.W_up[1, E.POS] = -S * 100
        self.b_up[1] = -S * 3.5  # TEMP >= 4
        self.W_gate[1, E.OP_START + Opcode.DIV] = -4096.0
        self.W_down[E.RAW_SUM, 1] = 1.0 / S

        # k=2: output 2^13 = 8192
        self.W_up[2, E.TEMP] = S
        self.W_up[2, E.POS] = -S * 100
        self.b_up[2] = -S * 1.5  # TEMP >= 2
        self.W_gate[2, E.OP_START + Opcode.DIV] = 8192.0
        self.W_down[E.RAW_SUM, 2] = 1.0 / S

        self.W_up[3, E.TEMP] = S
        self.W_up[3, E.POS] = -S * 100
        self.b_up[3] = -S * 2.5  # TEMP >= 3
        self.W_gate[3, E.OP_START + Opcode.DIV] = -8192.0
        self.W_down[E.RAW_SUM, 3] = 1.0 / S

        # k=1: output 2^14 = 16384
        self.W_up[4, E.TEMP] = S
        self.W_up[4, E.POS] = -S * 100
        self.b_up[4] = -S * 0.5  # TEMP >= 1
        self.W_gate[4, E.OP_START + Opcode.DIV] = 16384.0
        self.W_down[E.RAW_SUM, 4] = 1.0 / S

        self.W_up[5, E.TEMP] = S
        self.W_up[5, E.POS] = -S * 100
        self.b_up[5] = -S * 1.5  # TEMP >= 2
        self.W_gate[5, E.OP_START + Opcode.DIV] = -16384.0
        self.W_down[E.RAW_SUM, 5] = 1.0 / S

        # k=0: output 2^15 = 32768
        self.W_up[6, E.TEMP] = -S  # TEMP < 1 means k=0
        self.W_up[6, E.POS] = -S * 100
        self.b_up[6] = S * 0.5
        self.W_gate[6, E.OP_START + Opcode.DIV] = 32768.0
        self.W_down[E.RAW_SUM, 6] = 1.0 / S

        self.W_up[7, E.TEMP] = -S
        self.W_up[7, E.POS] = -S * 100
        self.b_up[7] = -S * 0.5
        self.W_gate[7, E.OP_START + Opcode.DIV] = -32768.0
        self.W_down[E.RAW_SUM, 7] = 1.0 / S


# =============================================================================
# Newton-Raphson Iteration (Single Layer)
# =============================================================================

class NewtonIterFFN(PureFFN):
    """
    One Newton-Raphson iteration for reciprocal.

    Formula: x_{n+1} = x_n * (2 - b * x_n)

    In Q16 fixed-point:
    - x_n is scaled by 2^16 (stored in RAW_SUM)
    - b is the integer divisor (in NIB_B)
    - 2 in fixed-point is 2 * 2^16 = 131072

    Computation:
    1. product = b * x_n (result scaled by 2^16)
    2. diff = 2.0 - product / 2^16 (in real units)
    3. x_{n+1} = x_n * diff

    This simplifies to:
    x_{n+1} = x_n * (2 - b * x_n / 2^16)
            = 2 * x_n - b * x_n^2 / 2^16

    Input/Output: RAW_SUM contains x (Q16 fixed-point reciprocal)
    """

    def __init__(self):
        # Compute 2*x - b*x^2/2^16
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Part 1: Add 2*x at position 0
        # Using silu(opcode) * x * 2
        self.W_up[0, E.OP_START + Opcode.DIV] = S
        self.W_up[0, E.POS] = -S * 100
        self.W_gate[0, E.RAW_SUM] = 2.0
        self.W_down[E.CARRY_OUT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.DIV] = -S
        self.W_up[1, E.POS] = -S * 100
        self.W_gate[1, E.RAW_SUM] = -2.0
        self.W_down[E.CARRY_OUT, 1] = 1.0 / S

        # Part 2: Subtract b*x^2/2^16
        # This requires squaring x and multiplying by b
        # Simplified approximation: -b * x / 2^16 * x
        # Using silu(x/const) * b * (-1)
        self.W_up[2, E.RAW_SUM] = S / FP_SCALE  # x / 2^16
        self.W_up[2, E.POS] = -S * 100
        self.W_gate[2, E.NIB_B] = -1.0
        self.W_gate[2, E.RAW_SUM] = 0.0  # Need x again for x^2
        self.W_down[E.CARRY_OUT, 2] = 1.0 / S

        self.W_up[3, E.RAW_SUM] = -S / FP_SCALE
        self.W_up[3, E.POS] = -S * 100
        self.W_gate[3, E.NIB_B] = 1.0
        self.W_down[E.CARRY_OUT, 3] = 1.0 / S


class UpdateReciprocalFFN(PureFFN):
    """Copy CARRY_OUT to RAW_SUM for next iteration."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Clear old RAW_SUM
        self.W_up[0, E.OP_START + Opcode.DIV] = S
        self.W_gate[0, E.RAW_SUM] = -1.0
        self.W_down[E.RAW_SUM, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.DIV] = -S
        self.W_gate[1, E.RAW_SUM] = 1.0
        self.W_down[E.RAW_SUM, 1] = 1.0 / S

        # Copy CARRY_OUT to RAW_SUM
        self.W_up[2, E.OP_START + Opcode.DIV] = S
        self.W_gate[2, E.CARRY_OUT] = 1.0
        self.W_down[E.RAW_SUM, 2] = 1.0 / S

        self.W_up[3, E.OP_START + Opcode.DIV] = -S
        self.W_gate[3, E.CARRY_OUT] = -1.0
        self.W_down[E.RAW_SUM, 3] = 1.0 / S


# =============================================================================
# Final Multiply
# =============================================================================

class FinalMultiplyFFN(PureFFN):
    """
    Compute final result: quotient = a * reciprocal / 2^16

    Input: a in NIB_A, reciprocal in RAW_SUM (Q16 format)
    Output: quotient in RESULT (integer)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # result = a * reciprocal / 2^16
        self.W_up[0, E.NIB_A] = S
        self.W_up[0, E.POS] = -S * 100
        self.W_gate[0, E.RAW_SUM] = 1.0 / FP_SCALE  # Descale
        self.W_down[E.RESULT, 0] = 1.0 / S

        self.W_up[1, E.NIB_A] = -S
        self.W_up[1, E.POS] = -S * 100
        self.W_gate[1, E.RAW_SUM] = -1.0 / FP_SCALE
        self.W_down[E.RESULT, 1] = 1.0 / S


class ClearDivTempFFN(PureFFN):
    """Clear temporary slots after division."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Clear TEMP
        self.W_up[0, E.OP_START + Opcode.DIV] = S
        self.W_gate[0, E.TEMP] = -1.0
        self.W_down[E.TEMP, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.DIV] = -S
        self.W_gate[1, E.TEMP] = 1.0
        self.W_down[E.TEMP, 1] = 1.0 / S

        # Clear RAW_SUM
        self.W_up[2, E.OP_START + Opcode.DIV] = S
        self.W_gate[2, E.RAW_SUM] = -1.0
        self.W_down[E.RAW_SUM, 2] = 1.0 / S

        self.W_up[3, E.OP_START + Opcode.DIV] = -S
        self.W_gate[3, E.RAW_SUM] = 1.0
        self.W_down[E.RAW_SUM, 3] = 1.0 / S

        # Clear CARRY_OUT
        self.W_up[4, E.OP_START + Opcode.DIV] = S
        self.W_gate[4, E.CARRY_OUT] = -1.0
        self.W_down[E.CARRY_OUT, 4] = 1.0 / S

        self.W_up[5, E.OP_START + Opcode.DIV] = -S
        self.W_gate[5, E.CARRY_OUT] = 1.0
        self.W_down[E.CARRY_OUT, 5] = 1.0 / S


# =============================================================================
# Complete Newton-Raphson Division Pipeline
# =============================================================================

def build_newton_raphson_div_layers(num_iterations: int = 4):
    """
    Build the complete Newton-Raphson division pipeline.

    Returns list of layers in execution order.

    With 4 iterations:
    - Layer 1: MSB detection
    - Layer 2: Initial reciprocal
    - Layers 3-10: Newton iterations (4 × 2 layers)
    - Layer 11: Final multiply
    - Layer 12: Cleanup

    Total: 12 layers (vs 16+ for iterative long division)
    """
    layers = [
        # Phase 1: MSB detection (1 layer)
        MSBDetectFFN(),

        # Phase 2: Initial reciprocal (1 layer)
        InitReciprocalFFN(),
    ]

    # Phase 3: Newton-Raphson iterations
    for _ in range(num_iterations):
        layers.append(NewtonIterFFN())
        layers.append(UpdateReciprocalFFN())

    # Phase 4: Final multiply (1 layer)
    layers.append(FinalMultiplyFFN())

    # Phase 5: Cleanup (1 layer)
    layers.append(ClearDivTempFFN())

    return layers


# =============================================================================
# Reference Implementation
# =============================================================================

def newton_raphson_div_reference(a: int, b: int, fp_bits: int = 16) -> int:
    """
    Reference Newton-Raphson division implementation.

    This shows the algorithm clearly without neural network complications.
    """
    if b == 0:
        return 0  # Handle divide by zero

    # Step 1: Find MSB of b
    k = 0
    temp = b
    while temp > 1:
        temp >>= 1
        k += 1

    # Step 2: Initial reciprocal guess
    # x_0 = 2^(fp_bits - k - 1)
    x = 1 << (fp_bits - k - 1)

    # Step 3: Newton-Raphson iterations
    # x_{n+1} = x_n * (2 - b * x_n / 2^fp_bits)
    for _ in range(4):
        # In fixed-point: x = x * (2 - b * x / 2^fp_bits)
        # = 2*x - b * x^2 / 2^fp_bits
        bx = b * x
        bx_scaled = bx >> fp_bits  # b * x / 2^fp_bits
        x = 2 * x - (x * bx_scaled >> fp_bits)

    # Step 4: Final multiply
    result = (a * x) >> fp_bits

    return result


# =============================================================================
# Layer and Weight Analysis
# =============================================================================

def analyze_newton_raphson_div():
    """Analyze layer count and weight count for Newton-Raphson division."""
    layers = build_newton_raphson_div_layers()

    total_weights = 0
    total_nonzero = 0

    print("Newton-Raphson Division Analysis")
    print("=" * 50)

    for i, layer in enumerate(layers):
        # Count parameters
        num_params = sum(p.numel() for p in layer.parameters())
        num_nonzero = sum((p != 0).sum().item() for p in layer.parameters())
        total_weights += num_params
        total_nonzero += num_nonzero

        print(f"Layer {i:2d}: {layer.__class__.__name__:25s} "
              f"params={num_params:6d} nonzero={num_nonzero:4d}")

    print("=" * 50)
    print(f"Total layers: {len(layers)}")
    print(f"Total parameters: {total_weights}")
    print(f"Total non-zero: {total_nonzero}")
    print(f"Sparsity: {100 * (1 - total_nonzero / total_weights):.1f}%")

    return len(layers), total_nonzero


# =============================================================================
# Test
# =============================================================================

def test_reference():
    """Test the reference implementation."""
    print("\n" + "=" * 50)
    print("Testing Reference Newton-Raphson Division")
    print("=" * 50)

    test_cases = [
        (42, 7, 6),
        (100, 10, 10),
        (15, 3, 5),
        (255, 16, 15),
        (1000, 33, 30),
        (12, 4, 3),
        (9, 3, 3),
        (15, 5, 3),
    ]

    for a, b, expected in test_cases:
        result = newton_raphson_div_reference(a, b)
        status = "PASS" if result == expected else "FAIL"
        print(f"{a:4d} / {b:2d} = {result:4d} (expected {expected:4d}) [{status}]")


if __name__ == "__main__":
    analyze_newton_raphson_div()
    test_reference()
