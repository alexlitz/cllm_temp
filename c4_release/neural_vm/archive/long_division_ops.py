"""
32-bit DIV/MOD via nibble-wise long division.

Base-16 long division: O(8) iterations for 8 nibbles.

Algorithm for each nibble i from 7 down to 0:
  1. remainder = remainder * 16 + dividend_nibble[i]
  2. q = floor(remainder / divisor)  # 0-15 using step functions
  3. remainder = remainder - q * divisor
  4. quotient_nibble[i] = q

The key insight: q = sum_{k=1}^{15} step(remainder - k*divisor)
This counts how many times divisor fits into remainder (0 to 15).

All classes extend FlattenedPureFFN and only override _bake_weights().
No forward() overrides - uses base class forward.
"""

import torch

from .embedding import E, Opcode
from .base_layers import FlattenedPureFFN, bake_weights


# Slot assignments for long division
SLOT_DIVIDEND = E.TEMP       # Gathered dividend (32-bit scalar)
SLOT_DIVISOR = E.TEMP + 1    # Gathered divisor
SLOT_REMAINDER = E.TEMP + 2  # Running remainder
SLOT_QUOTIENT = E.TEMP + 3   # Running quotient
SLOT_CURR_Q = E.TEMP + 4     # Current quotient nibble (0-15)


class ClearDivSlotsFFN(FlattenedPureFFN):
    """Clear all division temp slots."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        slots_to_clear = [SLOT_DIVIDEND, SLOT_DIVISOR, SLOT_REMAINDER, SLOT_QUOTIENT, SLOT_CURR_Q]
        super().__init__(hidden_dim=len(slots_to_clear) * 2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        slots = [SLOT_DIVIDEND, SLOT_DIVISOR, SLOT_REMAINDER, SLOT_QUOTIENT, SLOT_CURR_Q]

        for i, slot in enumerate(slots):
            h = i * 2
            slot_idx = self._flat_idx(0, slot)
            # Clear: subtract current value
            self.W_up[h, opcode_idx] = S
            self.W_gate[h, slot_idx] = -1.0
            self.W_down[slot_idx, h] = 1.0 / S
            # Saturation
            h = i * 2 + 1
            self.W_up[h, opcode_idx] = -S
            self.W_gate[h, slot_idx] = 1.0
            self.W_down[slot_idx, h] = 1.0 / S


class GatherScalarFFN(FlattenedPureFFN):
    """Gather 8 nibbles into a 32-bit scalar value."""

    def __init__(self, source_slot: int, dest_slot: int, opcode: int):
        self.source_slot = source_slot
        self.dest_slot = dest_slot
        self.opcode = opcode
        super().__init__(hidden_dim=E.NUM_POSITIONS + 1)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        dst_idx = self._flat_idx(0, self.dest_slot)

        for i in range(E.NUM_POSITIONS):
            h = i
            src_idx = self._flat_idx(i, self.source_slot)
            self.W_up[h, opcode_idx] = S
            self.W_gate[h, src_idx] = float(16 ** i)
            self.W_down[dst_idx, h] = 1.0 / S

        # Saturation
        h = E.NUM_POSITIONS
        self.W_up[h, opcode_idx] = -S
        for i in range(E.NUM_POSITIONS):
            src_idx = self._flat_idx(i, self.source_slot)
            self.W_gate[h, src_idx] = -float(16 ** i)
        self.W_down[dst_idx, h] = 1.0 / S


class ShiftRemainderAddNibbleFFN(FlattenedPureFFN):
    """
    For nibble position i: remainder = remainder * 16 + dividend_nibble[7-i]

    We process from MSB to LSB, so nibble_idx goes 7, 6, 5, ..., 0.
    """

    def __init__(self, nibble_idx: int, opcode: int):
        self.nibble_idx = nibble_idx  # Which nibble to bring down (7 to 0)
        self.opcode = opcode
        super().__init__(hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        dividend_nib_idx = self._flat_idx(self.nibble_idx, E.NIB_A)

        # h=0: remainder += remainder * 15 (so total becomes remainder * 16)
        h = 0
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, remainder_idx] = 15.0
        self.W_down[remainder_idx, h] = 1.0 / S

        # h=1: saturation
        h = 1
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, remainder_idx] = -15.0
        self.W_down[remainder_idx, h] = 1.0 / S

        # h=2: add dividend nibble
        h = 2
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, dividend_nib_idx] = 1.0
        self.W_down[remainder_idx, h] = 1.0 / S

        # h=3: saturation
        h = 3
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, dividend_nib_idx] = -1.0
        self.W_down[remainder_idx, h] = 1.0 / S


class ClearCurrQFFN(FlattenedPureFFN):
    """Clear SLOT_CURR_Q before computing new quotient nibble."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        curr_q_idx = self._flat_idx(0, SLOT_CURR_Q)

        h = 0
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, curr_q_idx] = -1.0
        self.W_down[curr_q_idx, h] = 1.0 / S

        h = 1
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, curr_q_idx] = 1.0
        self.W_down[curr_q_idx, h] = 1.0 / S


class ShiftRemainderAndClearQFFN(FlattenedPureFFN):
    """Merged: shift remainder + add nibble + clear SLOT_CURR_Q.

    Combines ShiftRemainderAddNibbleFFN and ClearCurrQFFN into one layer.
    These write to independent slots (SLOT_REMAINDER vs SLOT_CURR_Q).
    """

    def __init__(self, nibble_idx: int, opcode: int):
        self.nibble_idx = nibble_idx
        self.opcode = opcode
        super().__init__(hidden_dim=6)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        dividend_nib_idx = self._flat_idx(self.nibble_idx, E.NIB_A)
        curr_q_idx = self._flat_idx(0, SLOT_CURR_Q)

        # h=0,1: remainder += remainder * 15 (so total becomes remainder * 16)
        h = 0
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, remainder_idx] = 15.0
        self.W_down[remainder_idx, h] = 1.0 / S

        h = 1
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, remainder_idx] = -15.0
        self.W_down[remainder_idx, h] = 1.0 / S

        # h=2,3: add dividend nibble
        h = 2
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, dividend_nib_idx] = 1.0
        self.W_down[remainder_idx, h] = 1.0 / S

        h = 3
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, dividend_nib_idx] = -1.0
        self.W_down[remainder_idx, h] = 1.0 / S

        # h=4,5: clear SLOT_CURR_Q
        h = 4
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, curr_q_idx] = -1.0
        self.W_down[curr_q_idx, h] = 1.0 / S

        h = 5
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, curr_q_idx] = 1.0
        self.W_down[curr_q_idx, h] = 1.0 / S


class ComputeQuotientNibbleFFN(FlattenedPureFFN):
    """
    Compute q = floor(remainder / divisor) for q in range 0-15.

    Uses step functions: q = sum_{k=1}^{15} step(remainder - k*divisor)
    Each step adds 1 when remainder >= k*divisor.

    Uses DIV_Q_SCALE (lower) to avoid float32 precision issues in ONNX.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=30)  # 15 comparisons * 2

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_Q_SCALE  # Use lower scale for ONNX precision
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        divisor_idx = self._flat_idx(0, SLOT_DIVISOR)
        curr_q_idx = self._flat_idx(0, SLOT_CURR_Q)

        for k in range(1, 16):
            h = (k - 1) * 2

            # Positive step: silu(S*(remainder - k*divisor + 1)) * 1/S
            self.W_up[h, remainder_idx] = S
            self.W_up[h, divisor_idx] = -S * k
            self.b_up[h] = S * 1.0  # Upper threshold
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[curr_q_idx, h] = 1.0 / S

            # Saturation
            h = (k - 1) * 2 + 1
            self.W_up[h, remainder_idx] = S
            self.W_up[h, divisor_idx] = -S * k
            self.b_up[h] = 0.0
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[curr_q_idx, h] = -1.0 / S


class SubtractQTimesDivisorFFN(FlattenedPureFFN):
    """
    Subtract q * divisor from remainder.

    Since q is in range 0-15, we use 15 step functions:
    remainder -= sum_{k=1}^{15} step(q >= k) * divisor
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=30)  # 15 steps * 2

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        divisor_idx = self._flat_idx(0, SLOT_DIVISOR)
        curr_q_idx = self._flat_idx(0, SLOT_CURR_Q)

        for k in range(1, 16):
            h = (k - 1) * 2

            # Positive step: silu(S*(q - k + 1)) * (-divisor) / S
            self.W_up[h, curr_q_idx] = S
            self.b_up[h] = S * (1.0 - k)
            self.W_gate[h, divisor_idx] = 1.0
            self.W_down[remainder_idx, h] = -1.0 / S

            # Saturation
            h = (k - 1) * 2 + 1
            self.W_up[h, curr_q_idx] = S
            self.b_up[h] = S * (0.0 - k)
            self.W_gate[h, divisor_idx] = 1.0
            self.W_down[remainder_idx, h] = 1.0 / S


class WriteQuotientNibbleFFN(FlattenedPureFFN):
    """Write curr_q directly to RESULT[nibble_idx]."""

    def __init__(self, nibble_idx: int, opcode: int):
        self.nibble_idx = nibble_idx
        self.opcode = opcode
        super().__init__(hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        curr_q_idx = self._flat_idx(0, SLOT_CURR_Q)
        result_idx = self._flat_idx(self.nibble_idx, E.RESULT)

        h = 0
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, curr_q_idx] = 1.0
        self.W_down[result_idx, h] = 1.0 / S

        h = 1
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, curr_q_idx] = -1.0
        self.W_down[result_idx, h] = 1.0 / S


class SubtractAndWriteQFFN(FlattenedPureFFN):
    """Merged: subtract q*divisor from remainder + write q to RESULT[nibble_idx].

    Combines SubtractQTimesDivisorFFN and WriteQuotientNibbleFFN into one layer.
    These write to independent slots (SLOT_REMAINDER vs RESULT[nibble_idx]).
    """

    def __init__(self, nibble_idx: int, opcode: int):
        self.nibble_idx = nibble_idx
        self.opcode = opcode
        super().__init__(hidden_dim=32)  # 15*2 for subtract + 2 for write

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        divisor_idx = self._flat_idx(0, SLOT_DIVISOR)
        curr_q_idx = self._flat_idx(0, SLOT_CURR_Q)
        result_idx = self._flat_idx(self.nibble_idx, E.RESULT)

        # h=0..29: subtract q*divisor from remainder (from SubtractQTimesDivisorFFN)
        for k in range(1, 16):
            h = (k - 1) * 2

            # Positive step: silu(S*(q - k + 1)) * (-divisor) / S
            self.W_up[h, curr_q_idx] = S
            self.b_up[h] = S * (1.0 - k)
            self.W_gate[h, divisor_idx] = 1.0
            self.W_down[remainder_idx, h] = -1.0 / S

            # Saturation
            h = (k - 1) * 2 + 1
            self.W_up[h, curr_q_idx] = S
            self.b_up[h] = S * (0.0 - k)
            self.W_gate[h, divisor_idx] = 1.0
            self.W_down[remainder_idx, h] = 1.0 / S

        # h=30,31: write curr_q to RESULT[nibble_idx]
        h = 30
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, curr_q_idx] = 1.0
        self.W_down[result_idx, h] = 1.0 / S

        h = 31
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, curr_q_idx] = -1.0
        self.W_down[result_idx, h] = 1.0 / S


class ClearResultsFFN(FlattenedPureFFN):
    """Clear all RESULT slots."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(hidden_dim=E.NUM_POSITIONS * 2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        h = 0
        for pos in range(E.NUM_POSITIONS):
            result_idx = self._flat_idx(pos, E.RESULT)
            self.W_up[h, opcode_idx] = S
            self.W_gate[h, result_idx] = -1.0
            self.W_down[result_idx, h] = 1.0 / S
            h += 1
            self.W_up[h, opcode_idx] = -S
            self.W_gate[h, result_idx] = 1.0
            self.W_down[result_idx, h] = 1.0 / S
            h += 1


class ExtractRemainderNibbleFFN(FlattenedPureFFN):
    """
    Extract one nibble from the remainder scalar.

    For nibble at position pos:
      nibble = floor(remainder / 16^pos) mod 16

    Uses step functions to count how many 16^pos fit into remainder.
    """

    def __init__(self, nibble_pos: int, opcode: int):
        self.nibble_pos = nibble_pos
        self.opcode = opcode
        self.divisor = 16 ** nibble_pos
        super().__init__(hidden_dim=15 * 2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        result_idx = self._flat_idx(self.nibble_pos, E.RESULT)

        for k in range(1, 16):
            h = (k - 1) * 2
            threshold = k * self.divisor

            # Step: add 1 when remainder >= k * divisor
            self.W_up[h, remainder_idx] = S
            self.b_up[h] = S * (1.0 - threshold)
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[result_idx, h] = 1.0 / S

            # Saturation
            h = (k - 1) * 2 + 1
            self.W_up[h, remainder_idx] = S
            self.b_up[h] = S * (0.0 - threshold)
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[result_idx, h] = -1.0 / S


class SubtractHigherNibblesFFN(FlattenedPureFFN):
    """
    After extracting higher nibbles, subtract their contribution from remainder
    so lower nibbles can be extracted correctly.

    remainder -= nibble_value * 16^pos
    """

    def __init__(self, nibble_pos: int, opcode: int):
        self.nibble_pos = nibble_pos
        self.opcode = opcode
        self.multiplier = 16 ** nibble_pos
        super().__init__(hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        result_idx = self._flat_idx(self.nibble_pos, E.RESULT)
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        # Subtract result[nibble_pos] * 16^nibble_pos from remainder
        h = 0
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, result_idx] = float(self.multiplier)
        self.W_down[remainder_idx, h] = -1.0 / S

        h = 1
        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, result_idx] = -float(self.multiplier)
        self.W_down[remainder_idx, h] = -1.0 / S


class ExtractAndSubtractRemainderFFN(FlattenedPureFFN):
    """Merged: extract nibble + subtract contribution + clear old RESULT.

    Combines ExtractRemainderNibbleFFN and SubtractHigherNibblesFFN into one
    layer per nibble position. Also clears the old RESULT[pos] value (quotient
    nibble left over from the division loop).

    Cascade: process from pos 7 (MSB) down to 0 (LSB). After extracting
    nibble at pos and subtracting nibble * 16^pos from SLOT_REMAINDER, the
    updated remainder is < 16^pos, so the next position's 15 step functions
    always suffice (floor(remainder / 16^(pos-1)) <= 15).

    Each step function writes to BOTH RESULT[pos] (+1) and SLOT_REMAINDER
    (-16^pos) through W_down, computing extract and subtract simultaneously
    from one set of hidden units.
    """

    def __init__(self, nibble_pos: int, opcode: int):
        self.nibble_pos = nibble_pos
        self.opcode = opcode
        # 2 for clear RESULT[pos] + 30 for 15 step function pairs
        super().__init__(hidden_dim=32)

    @bake_weights
    def _bake_weights(self):
        S = E.DIV_SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        remainder_idx = self._flat_idx(0, SLOT_REMAINDER)
        result_idx = self._flat_idx(self.nibble_pos, E.RESULT)
        divisor = 16 ** self.nibble_pos

        # h=0,1: Clear old RESULT[pos] (quotient nibble from division)
        self.W_up[0, opcode_idx] = S
        self.W_gate[0, result_idx] = -1.0
        self.W_down[result_idx, 0] = 1.0 / S

        self.W_up[1, opcode_idx] = -S
        self.W_gate[1, result_idx] = 1.0
        self.W_down[result_idx, 1] = 1.0 / S

        # h=2..31: Step functions for nibble extraction + remainder subtract
        for k in range(1, 16):
            h = 2 + (k - 1) * 2
            threshold = k * divisor

            # Step fires when remainder >= k * 16^pos
            self.W_up[h, remainder_idx] = S
            self.b_up[h] = S * (1.0 - threshold)
            self.W_gate[h, opcode_idx] = 1.0
            # Write +1 to RESULT[pos] (extract nibble)
            self.W_down[result_idx, h] = 1.0 / S
            # Write -16^pos to SLOT_REMAINDER (subtract contribution)
            self.W_down[remainder_idx, h] = -float(divisor) / S

            # Saturation
            h = 2 + (k - 1) * 2 + 1
            self.W_up[h, remainder_idx] = S
            self.b_up[h] = S * (0.0 - threshold)
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[result_idx, h] = -1.0 / S
            self.W_down[remainder_idx, h] = float(divisor) / S


def build_long_division_layers(opcode: int = Opcode.DIV):
    """
    Build layers for nibble-wise long division. O(8) iterations.

    For DIV: writes quotient nibbles directly to RESULT. 26 layers.
    For MOD: extracts remainder nibbles from SLOT_REMAINDER. 34 layers.

    Division core (26 layers = 2 setup + 8*3):
      1. ShiftRemainderAndClearQFFN (shift remainder + add nibble + clear curr_q)
      2. ComputeQuotientNibbleFFN (unchanged)
      3. SubtractAndWriteQFFN (subtract q*divisor + write quotient nibble)

    MOD remainder extraction (+8 layers):
      ExtractAndSubtractRemainderFFN for each nibble pos 7..0 (cascade).
      Each layer extracts nibble, clears old quotient from RESULT, and
      subtracts contribution from SLOT_REMAINDER in a single pass.
    """
    from .pure_moe import MoE

    layers = []

    # Clear temp slots
    layers.append(MoE([ClearDivSlotsFFN(opcode)], [opcode]))

    # Gather divisor (we'll read dividend nibbles directly)
    layers.append(MoE([GatherScalarFFN(E.NIB_B, SLOT_DIVISOR, opcode)], [opcode]))

    # 8 iterations: process nibbles from MSB (7) to LSB (0)
    for nibble_idx in range(7, -1, -1):
        # 1. Shift remainder, add nibble, clear curr_q (merged steps 1+2)
        layers.append(MoE([ShiftRemainderAndClearQFFN(nibble_idx, opcode)], [opcode]))

        # 2. Compute q = floor(remainder / divisor)
        layers.append(MoE([ComputeQuotientNibbleFFN(opcode)], [opcode]))

        # 3. Subtract q*divisor from remainder + write q to RESULT (merged steps 4+5)
        layers.append(MoE([SubtractAndWriteQFFN(nibble_idx, opcode)], [opcode]))

    # For MOD: extract remainder nibbles directly from SLOT_REMAINDER.
    # After division, SLOT_REMAINDER holds the scalar remainder.
    # Cascade extraction: process nibbles 7 (MSB) down to 0 (LSB).
    # Each layer extracts one nibble, clears old quotient from RESULT[pos],
    # and subtracts the nibble's contribution from SLOT_REMAINDER.
    # 8 layers total (one per nibble position).
    if opcode == Opcode.MOD:
        for pos in range(7, -1, -1):
            layers.append(MoE([ExtractAndSubtractRemainderFFN(pos, opcode)], [opcode]))

    return layers


if __name__ == "__main__":
    print("Testing nibble-wise long division algorithm...")

    test_cases = [
        (42, 6, 7),
        (100, 10, 10),
        (1000, 33, 30),
        (10000, 100, 100),
        (65535, 255, 257),
        (50000, 100, 500),
    ]

    for dividend, divisor, expected in test_cases:
        # Manual nibble-wise long division
        remainder = 0
        quotient = 0
        for i in range(7, -1, -1):
            nib = (dividend >> (i * 4)) & 0xF
            remainder = remainder * 16 + nib
            q = remainder // divisor if divisor > 0 else 0
            q = min(q, 15)  # Cap at nibble max
            remainder -= q * divisor
            quotient = quotient * 16 + q

        status = "PASS" if quotient == expected else "FAIL"
        print(f"{dividend} / {divisor} = {quotient} (expected {expected}) [{status}]")
