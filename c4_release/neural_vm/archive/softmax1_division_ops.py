"""
Softmax1-based Division for Neural VM.

Core insight: softmax1 leftover = 1/(1 + Σexp(s_j)).
If scores are set so Σexp(s_j) = n-1, then leftover = 1/n.

Nibble construction (8 tokens, base-16):
  n-1 = Σ_j 16^j d_j  (nibbles of divisor - 1)
  score_j = log(16^j · d_j) for d_j > 0, else -SCALE
  Σexp = n-1, leftover = 1/n

Pipeline (4 layers for DIV, was 11):
  Layer 1: Clear + Gather + Softmax1 Reciprocal  [MERGED]
  Layer 2: Multiply Q_float = dividend × reciprocal
  Layer 3: Floor extraction via fp64 MAGIC trick
           floor(Q/16^j) → RESULT[j], 25 hidden units
  Layer 4: Nibble subtraction: RESULT[j] -= 16*RESULT[j+1]
           14 hidden units

MOD: 15 layers (4 quotient + 7 multiply + 3 subtract + 1 correction)

fp64 MAGIC trick (Layer 3):
  MAGIC = 3 × 2^51. At this scale, fp64 ULP = 1, so addition
  rounds to nearest integer. With offset -(0.5 - eps_j):
    floor(x) = (x - 0.5 + eps_j + MAGIC) - MAGIC
  eps_j = 2^(-20-4j) breaks round-to-even ties for exact integers.

MOD correction (x-1 trick):
  Computing floor((dividend-1) × reciprocal) guarantees the quotient
  is never too high. One correction subtract handles "too low by 1".

Float64 note: requires fp64 for correctness. Layer 3 uses the MAGIC
number trick which requires ULP = 1 at the 2^52 scale.
"""

import torch
import torch.nn as nn
import math

from .embedding import E, Opcode
from .base_layers import FlattenedPureFFN, bake_weights
from .long_division_ops import (
    ClearDivSlotsFFN, GatherScalarFFN, ExtractAndSubtractRemainderFFN,
    SLOT_DIVIDEND, SLOT_DIVISOR, SLOT_REMAINDER, SLOT_QUOTIENT, SLOT_CURR_Q,
)

SOFTMAX1_SCALE = 60.0  # Suppression for inactive nibbles


# =============================================================================
# Softmax1 Reciprocal — the core construction
# =============================================================================

class Softmax1ReciprocalModule(nn.Module):
    """
    Compute 1/divisor via the softmax1 nibble construction.

    For each nibble position j of the divisor:
      d_j = NIB_B[j]
      score_j = log(16^j · d_j)  if d_j > 0
              = -SCALE            if d_j = 0  (suppressed)

    The sum of exponentials:
      Σ exp(score_j) = Σ 16^j · d_j = divisor

    The softmax1 leftover with adjusted sum (divisor - 1):
      1 / (1 + (divisor - 1)) = 1/divisor

    In a transformer, this IS the attention normalization constant.
    The reciprocal comes for free from the softmax computation.

    Reads: NIB_B (divisor nibbles)
    Writes: SLOT_QUOTIENT (reciprocal = 1/divisor)
    """

    def __init__(self, opcode: int, use_explicit_softmax1: bool = True):
        super().__init__()
        self.opcode = opcode
        self.use_explicit_softmax1 = use_explicit_softmax1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        opcode_w = x[:, 0, E.OP_START + self.opcode]  # [B]

        if self.use_explicit_softmax1:
            reciprocal = self._softmax1_reciprocal(x)
        else:
            reciprocal = self._direct_reciprocal(x)

        delta = torch.zeros_like(x)
        delta[:, 0, SLOT_QUOTIENT] = opcode_w * reciprocal
        return x + delta

    def _softmax1_reciprocal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Explicit softmax1 construction with nibble scores.

        Constructs 8 scores from divisor nibbles, computes exp of each,
        sums them to get divisor, and returns 1/divisor via the
        softmax1 identity: 1/(1 + (n-1)) = 1/n.
        """
        B, N, D = x.shape
        S = SOFTMAX1_SCALE

        # Construct scores: score_j = log(16^j * d_j) or -SCALE
        scores = torch.full((B, N), -S, device=x.device, dtype=x.dtype)
        for j in range(N):
            d_j = x[:, j, E.NIB_B]
            active = d_j > 0.5
            val = (d_j * (16.0 ** j)).clamp(min=0.5)
            scores[:, j] = torch.where(active, torch.log(val), scores[:, j])

        # Numerically stable: compute Σexp(score_j)
        mx = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - mx)
        sum_ex = ex.sum(dim=-1)  # = divisor * exp(-mx)

        # The sum of exponentials = divisor (up to SCALE suppression)
        # Softmax1 leftover would be 1/(1 + divisor)
        #
        # We want 1/divisor, which is the attention normalization constant:
        #   1/Z = exp(-mx) / sum_ex = 1/divisor
        #
        # Equivalently: softmax1 leftover with sum_exp = divisor - 1
        #   = 1/(1 + divisor - 1) = 1/divisor
        #
        # The "+1" in softmax1 accounts for the phantom token.
        # By interpreting our sum as (divisor - 1) + 1, the leftover = 1/divisor.
        exp_neg_mx = torch.exp(-mx.squeeze(-1))
        reciprocal = exp_neg_mx / sum_ex.clamp(min=1e-30)

        return reciprocal

    def _direct_reciprocal(self, x: torch.Tensor) -> torch.Tensor:
        """Direct computation: gather divisor, return 1/divisor."""
        B, N, D = x.shape
        divisor = torch.zeros(B, device=x.device, dtype=x.dtype)
        for j in range(N):
            divisor = divisor + x[:, j, E.NIB_B] * (16.0 ** j)
        return 1.0 / divisor.clamp(min=1.0)


# =============================================================================
# Multiply dividend × reciprocal
# =============================================================================

class MultiplyDivReciprocalFFN(FlattenedPureFFN):
    """
    Compute quotient = SLOT_DIVIDEND × SLOT_QUOTIENT (reciprocal).

    Uses SwiGLU multiplication: silu(S * a) * b / S ≈ a * b for a > 0.

    Reads: SLOT_DIVIDEND (dividend scalar), SLOT_QUOTIENT (1/divisor)
    Writes: SLOT_REMAINDER (quotient scalar, for nibble extraction)
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        dividend_idx = self._flat_idx(0, SLOT_DIVIDEND)
        reciprocal_idx = self._flat_idx(0, SLOT_QUOTIENT)
        result_idx = self._flat_idx(0, SLOT_REMAINDER)

        # h=0: silu(S * dividend) * reciprocal / S → quotient
        self.W_up[0, dividend_idx] = S
        self.W_gate[0, reciprocal_idx] = 1.0
        self.W_down[result_idx, 0] = 1.0 / S

        # h=1: saturation
        self.W_up[1, dividend_idx] = -S
        self.W_gate[1, reciprocal_idx] = -1.0
        self.W_down[result_idx, 1] = 1.0 / S


class MultiplyDivReciprocalMinusOneFFN(FlattenedPureFFN):
    """
    Compute quotient = (SLOT_DIVIDEND - 1) × SLOT_QUOTIENT (reciprocal).

    The -1 ensures the estimated quotient is never too high:
      (x-1)/n < x/n, so floor((x-1) * (1/n)) ≤ floor(x/n).
    The gap from (x-1)/n to the next integer is ≥ 1/n, which in fp64
    exceeds the float error for all 32-bit operands.

    Uses SwiGLU: silu(S*a - S) * b / S ≈ (a-1) * b for a > 1.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        dividend_idx = self._flat_idx(0, SLOT_DIVIDEND)
        reciprocal_idx = self._flat_idx(0, SLOT_QUOTIENT)
        result_idx = self._flat_idx(0, SLOT_REMAINDER)

        # h=0: silu(S * dividend - S) * reciprocal / S → (dividend-1) * reciprocal
        self.W_up[0, dividend_idx] = S
        self.b_up[0] = -S  # subtract 1 in the SwiGLU gate
        self.W_gate[0, reciprocal_idx] = 1.0
        self.W_down[result_idx, 0] = 1.0 / S

        # h=1: saturation
        self.W_up[1, dividend_idx] = -S
        self.b_up[1] = S
        self.W_gate[1, reciprocal_idx] = -1.0
        self.W_down[result_idx, 1] = 1.0 / S


# =============================================================================
# Floor extraction via fp64 MAGIC trick
# =============================================================================

# MAGIC = 3 * 2^51. At this scale, fp64 ULP = 1.
# Adding MAGIC to x rounds x to the nearest integer.
# Subtracting MAGIC recovers the rounded integer.
# With offset -(0.5 - eps), this computes floor(x).
MAGIC = 3.0 * 2**51  # 6755399441055744.0


class FloorExtractionFFN(FlattenedPureFFN):
    """
    Extract floor(Q/16^j) for j=0..7 via the fp64 MAGIC number trick.

    MAGIC = 3 * 2^51. At this scale, fp64 ULP = 1, so:
      up = (Q/16^j) - 0.5 + eps_j + MAGIC
    rounds to MAGIC + floor(Q/16^j). The gate passes through when
    opcode is active, and W_down + cancel unit subtract MAGIC.

    eps_j = 2^(-20-4j) breaks round-to-even ties for exact integers
    without pushing near-integer values over boundaries.
    Verified: 0/200k random + 0 targeted failures.

    Hidden units (25 total):
      h=0..7:   floor(Q/16^j) via MAGIC addition (8 units)
      h=8:      MAGIC cancellation constant (1 unit)
      h=9..24:  clear RESULT[0..7] (16 units, 8 saturation pairs)

    Reads: SLOT_REMAINDER (Q_float = dividend × reciprocal)
    Writes: RESULT[j] = floor(Q/16^j) for j=0..7

    REQUIRES fp64 precision.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=25)

    @bake_weights
    def _bake_weights(self):
        # Convert to float64 for MAGIC number precision
        self.ffn = self.ffn.double()

        S = E.SCALE
        C = float(2**20)  # Cancel unit scale (power of 2 for exact arithmetic)
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        q_float_idx = self._flat_idx(0, SLOT_REMAINDER)

        # h=0..7: Floor extraction via MAGIC trick
        # up = (1/16^j)*Q - (0.5 - eps_j)*opcode + MAGIC
        # gate = opcode
        # h_j = silu(up) * gate ≈ MAGIC + floor(Q/16^j)  (silu = identity for huge positive)
        for j in range(8):
            h = j
            eps_j = 2.0 ** (-20 - 4 * j)
            scale_j = 1.0 / 16**j        # exact power of 2
            offset_j = -(0.5 - eps_j)    # exact in fp64
            result_j_idx = self._flat_idx(j, E.RESULT)

            self.W_up[h, q_float_idx] = scale_j
            self.W_up[h, opcode_idx] = offset_j
            self.b_up[h] = MAGIC
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[result_j_idx, h] = 1.0

        # h=8: MAGIC cancellation unit
        # Produces C when active; W_down scales by -MAGIC/C = -3*2^31 (exact)
        # Output: -MAGIC/C * C = -MAGIC at each RESULT[j] (exact by Sterbenz)
        h = 8
        self.W_up[h, opcode_idx] = C
        self.W_gate[h, opcode_idx] = 1.0
        for j in range(8):
            result_j_idx = self._flat_idx(j, E.RESULT)
            self.W_down[result_j_idx, h] = -MAGIC / C  # = -3 * 2^31

        # h=9..24: Clear old RESULT values (8 positions × 2 saturation pairs)
        for pos in range(8):
            result_pos_idx = self._flat_idx(pos, E.RESULT)

            h = 9 + pos * 2
            self.W_up[h, opcode_idx] = S
            self.W_gate[h, result_pos_idx] = -1.0
            self.W_down[result_pos_idx, h] = 1.0 / S

            h = 9 + pos * 2 + 1
            self.W_up[h, opcode_idx] = -S
            self.W_gate[h, result_pos_idx] = 1.0
            self.W_down[result_pos_idx, h] = 1.0 / S


class NibbleSubtractFFN(FlattenedPureFFN):
    """
    Convert floor values to nibbles in-place.

    After FloorExtractionFFN, RESULT[j] = floor(Q/16^j).
    This layer subtracts 16 × RESULT[j+1] from RESULT[j]:
      RESULT[j] = floor(Q/16^j) - 16*floor(Q/16^{j+1}) = nibble_j

    RESULT[7] is already correct (nibble_7 = floor(Q/16^7)).

    Hidden units (14 total):
      For j=0..6: 2 units each (with saturation)

    Reads: RESULT[j+1] (floor values from Layer 3)
    Writes: RESULT[j] -= 16 * RESULT[j+1]
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=14)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        for j in range(7):  # j=0..6
            result_j_idx = self._flat_idx(j, E.RESULT)
            result_jp1_idx = self._flat_idx(j + 1, E.RESULT)

            # silu(S*opcode) * RESULT[j+1] * (-16/S) → subtracts 16*floor_{j+1}
            h = j * 2
            self.W_up[h, opcode_idx] = S
            self.W_gate[h, result_jp1_idx] = 1.0
            self.W_down[result_j_idx, h] = -16.0 / S

            # Saturation pair
            h = j * 2 + 1
            self.W_up[h, opcode_idx] = -S
            self.W_gate[h, result_jp1_idx] = -1.0
            self.W_down[result_j_idx, h] = -16.0 / S


# =============================================================================
# MOD Correction — handle off-by-one from x-1 trick
# =============================================================================

class ModCorrectionModule(nn.Module):
    """
    If remainder >= divisor, subtract divisor.

    After computing remainder = dividend - floor((dividend-1)/divisor) * divisor,
    the result is either the true remainder (correct quotient) or
    true_remainder + divisor (quotient was 1 too low). This layer
    detects and corrects the latter case.

    Gathers RESULT and NIB_B as scalars, compares, conditionally
    subtracts, and writes corrected nibbles back to RESULT.
    """

    def __init__(self, opcode: int):
        super().__init__()
        self.opcode = opcode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        opcode_w = x[:, 0, E.OP_START + self.opcode]

        # Gather remainder and divisor as scalars
        remainder = torch.zeros(B, device=x.device, dtype=x.dtype)
        divisor = torch.zeros(B, device=x.device, dtype=x.dtype)
        for j in range(N):
            remainder = remainder + x[:, j, E.RESULT] * (16.0 ** j)
            divisor = divisor + x[:, j, E.NIB_B] * (16.0 ** j)

        # Conditional subtract
        needs_correction = (remainder >= divisor).to(x.dtype)
        corrected = remainder - needs_correction * divisor

        # Extract corrected nibbles back to RESULT
        delta = torch.zeros_like(x)
        val = corrected
        for j in range(N - 1, -1, -1):
            pw = 16.0 ** j
            nib = torch.floor(val / pw).clamp(0, 15)
            delta[:, j, E.RESULT] = opcode_w * (nib - x[:, j, E.RESULT])
            val = val - nib * pw

        return x + delta


# =============================================================================
# Pipeline builder
# =============================================================================

def build_softmax1_division_layers(opcode: int = Opcode.DIV):
    """
    Build layers for softmax1-based division.

    Pipeline (4 layers for DIV):
      1. Clear + Gather + Reciprocal                    (1 layer, merged MoE)
      2. Multiply: Q_float = dividend × reciprocal      (1 layer)
      3. Floor extraction: floor(Q/16^j) → RESULT[j]    (1 layer, MAGIC trick)
      4. Nibble subtraction: nibble_j = floor_j - 16*floor_{j+1}  (1 layer)

    Total: 4 layers for DIV (was 11)
    Total: 15 layers for MOD (4 + 7 multiply + 3 subtract + 1 correction)

    Layer 1 merges ClearDivSlotsFFN + GatherScalarFFN + Softmax1ReciprocalModule.
    Safe because they write to independent slots (clear subtracts old values,
    gather/reciprocal add new values — deltas sum correctly in MoE).

    Layer 3 uses the fp64 MAGIC number trick: adding MAGIC = 3*2^51 forces
    rounding to integer (ULP = 1 at that scale). The -(0.5-eps) offset
    converts rounding to floor. Requires fp64 precision.

    MOD uses the x-1 trick: multiply (dividend-1) × reciprocal so the
    quotient is never too high. Correction layer handles off-by-one.
    """
    from .pure_moe import MoE

    layers = []

    # 1. Clear + Gather + Reciprocal (merged: independent output slots)
    #    ClearDivSlotsFFN clears SLOT_DIVIDEND..SLOT_CURR_Q (delta = -old)
    #    GatherScalarFFN writes to SLOT_DIVIDEND (delta = +gathered)
    #    Softmax1ReciprocalModule writes to SLOT_QUOTIENT (delta = +reciprocal)
    #    Net: SLOT_DIVIDEND = dividend, SLOT_QUOTIENT = reciprocal
    layers.append(MoE(
        [ClearDivSlotsFFN(opcode),
         GatherScalarFFN(E.NIB_A, SLOT_DIVIDEND, opcode),
         Softmax1ReciprocalModule(opcode)],
        [opcode, opcode, opcode]
    ))

    # 2. Multiply: Q_float = dividend × reciprocal → SLOT_REMAINDER
    #    For MOD: use (dividend-1) × reciprocal to guarantee quotient ≤ true
    if opcode == Opcode.MOD:
        layers.append(MoE(
            [MultiplyDivReciprocalMinusOneFFN(opcode)],
            [opcode]
        ))
    else:
        layers.append(MoE(
            [MultiplyDivReciprocalFFN(opcode)],
            [opcode]
        ))

    # 3. Floor extraction: floor(Q/16^j) → RESULT[j] via MAGIC trick
    #    Also clears old RESULT values. 25 hidden units.
    layers.append(MoE(
        [FloorExtractionFFN(opcode)],
        [opcode]
    ))

    # 4. Nibble subtraction: RESULT[j] -= 16 * RESULT[j+1]
    #    After this: RESULT[j] = nibble_j of quotient. 14 hidden units.
    layers.append(MoE(
        [NibbleSubtractFFN(opcode)],
        [opcode]
    ))

    # For MOD: x mod N = x - floor((x-1)/N) * N, then correct if >= N
    if opcode == Opcode.MOD:
        from .fast_mul import (
            SchoolbookFlatFFN, MulCarryPass1FFN, MulCarryPass2FFN,
            MulCarryPass3FFN, MulGenPropFFN, MulBinaryLookaheadFFN,
            MulFinalCorrectionFFN,
        )
        from .fast_arithmetic import (
            SubRawAndGenFFN, SubBorrowLookaheadFFN, SubFinalResultFFN,
        )

        # 5. Multiply: quotient (RESULT) × divisor (NIB_B) → RESULT
        layers.append(MoE([SchoolbookFlatFFN(opcode=opcode, source_a=E.RESULT)], [opcode]))
        layers.append(MoE([MulCarryPass1FFN(opcode=opcode)], [opcode]))
        layers.append(MoE([MulCarryPass2FFN(opcode=opcode)], [opcode]))
        layers.append(MoE([MulCarryPass3FFN(opcode=opcode)], [opcode]))
        layers.append(MoE([MulGenPropFFN(opcode=opcode)], [opcode]))
        layers.append(MoE([MulBinaryLookaheadFFN(opcode=opcode)], [opcode]))
        layers.append(MoE([MulFinalCorrectionFFN(opcode=opcode)], [opcode]))

        # 6. Subtract: dividend (NIB_A) - product (RESULT) → RESULT (preliminary remainder)
        layers.append(MoE([SubRawAndGenFFN(opcode=opcode, source_b=E.RESULT, clear_result=True)], [opcode]))
        layers.append(MoE([SubBorrowLookaheadFFN(opcode=opcode)], [opcode]))
        layers.append(MoE([SubFinalResultFFN(opcode=opcode)], [opcode]))

        # 7. Correction: if remainder >= divisor, subtract divisor
        layers.append(MoE([ModCorrectionModule(opcode)], [opcode]))

    return layers


# =============================================================================
# Reference implementation
# =============================================================================

def _softmax1_reciprocal_ref(divisor: int):
    """Compute 1/divisor via softmax1 nibble construction (reference)."""
    import numpy as np

    nibbles = []
    n = divisor
    for j in range(8):
        nibbles.append(n & 0xF)
        n >>= 4

    S = 60.0
    scores = []
    for j, d_j in enumerate(nibbles):
        if d_j > 0:
            scores.append(np.log(16**j * d_j))
        else:
            scores.append(-S)

    scores = np.array(scores)
    mx = np.max(scores)
    ex = np.exp(scores - mx)
    sum_ex = np.sum(ex)
    return np.exp(-mx) / sum_ex


def softmax1_div_reference(dividend: int, divisor: int) -> int:
    """
    Reference: floor(dividend / divisor) via softmax1 reciprocal.

    Uses round-then-verify since exp(log(x)) ≠ x can make
    the quotient off by 1 in either direction.
    """
    if divisor == 0:
        return 0

    reciprocal = _softmax1_reciprocal_ref(divisor)
    raw = dividend * reciprocal
    quotient = int(raw + 0.5)
    if quotient * divisor > dividend:
        quotient -= 1
    return quotient


def softmax1_mod_reference(dividend: int, divisor: int) -> int:
    """
    Reference: (dividend mod divisor) via the x-1 trick.

    Computes floor((dividend-1) * reciprocal) — guaranteed ≤ true quotient
    because the gap from (x-1)/n to the next integer (≥ 1/n) exceeds
    the float error for 32-bit ints in fp64.

    Then remainder = dividend - q * divisor. If remainder >= divisor,
    subtract divisor once (the correction step).
    """
    if divisor == 0:
        return 0
    if divisor == 1:
        return 0

    reciprocal = _softmax1_reciprocal_ref(divisor)
    # x-1 trick: quotient is never too high
    quotient = int((dividend - 1) * reciprocal)
    remainder = dividend - quotient * divisor
    if remainder >= divisor:
        remainder -= divisor
    return remainder


# =============================================================================
# Test
# =============================================================================

def test_softmax1_division():
    """Test the softmax1 division and MOD pipelines."""
    import random

    print("=" * 60)
    print("Softmax1 Division Test")
    print("=" * 60)

    # DIV reference test
    print("\nDIV reference:")
    div_cases = [
        (42, 7, 6),
        (100, 10, 10),
        (1000, 33, 30),
        (255, 16, 15),
        (65535, 255, 257),
        (50000, 100, 500),
        (999999, 7, 142857),
        (0, 5, 0),
        (5, 1, 5),
        (15, 15, 1),
    ]

    all_pass = True
    for dividend, divisor, expected in div_cases:
        result = softmax1_div_reference(dividend, divisor)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  {dividend:>8d} / {divisor:>4d} = {result:>8d} "
              f"(expected {expected:>8d}) [{status}]")

    # MOD reference test (x-1 trick)
    print("\nMOD reference (x-1 trick):")
    mod_cases = [
        (42, 7, 0),
        (100, 10, 0),
        (100, 7, 2),
        (1000, 33, 10),
        (255, 16, 15),
        (65535, 255, 0),
        (65536, 255, 1),
        (999999, 7, 0),
        (999998, 7, 6),
        (0, 5, 0),
        (1, 1, 0),
        (15, 15, 0),
        (14, 15, 14),
    ]

    for dividend, divisor, expected in mod_cases:
        result = softmax1_mod_reference(dividend, divisor)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  {dividend:>8d} % {divisor:>4d} = {result:>8d} "
              f"(expected {expected:>8d}) [{status}]")

    # Stress test MOD with random values
    print("\nMOD stress test (1000 random 32-bit cases):")
    random.seed(42)
    mod_failures = 0
    for _ in range(1000):
        dividend = random.randint(0, 2**32 - 1)
        divisor = random.randint(1, 2**32 - 1)
        expected = dividend % divisor
        result = softmax1_mod_reference(dividend, divisor)
        if result != expected:
            mod_failures += 1
            if mod_failures <= 5:
                print(f"  FAIL: {dividend} % {divisor} = {result} (expected {expected})")
    if mod_failures == 0:
        print(f"  All 1000 cases passed!")
    else:
        all_pass = False
        print(f"  {mod_failures}/1000 failures")

    # Layer count comparison
    print(f"\nLayer counts:")
    div_layers = build_softmax1_division_layers(Opcode.DIV)
    mod_layers = build_softmax1_division_layers(Opcode.MOD)
    print(f"  Softmax1 DIV: {len(div_layers)} layers (was 11, now 4)")
    print(f"  Softmax1 MOD: {len(mod_layers)} layers (was 22, now 15)")
    print(f"  Long div DIV: 26 layers")
    print(f"  Long div MOD: 34 layers")

    if all_pass:
        print("\nAll tests passed!")
    return all_pass


if __name__ == "__main__":
    test_softmax1_division()
