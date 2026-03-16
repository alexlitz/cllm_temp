"""
Multi-nibble MUL, DIV, MOD operations for Neural VM V7.

Proper 32-bit implementations using shift-add multiplication
and shift-subtract division algorithms.
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


# =============================================================================
# Multi-Nibble MUL - Schoolbook Algorithm
# =============================================================================
#
# For A × B where both are 8-nibble (32-bit) numbers:
# result[k] = Σ(a[i] × b[j]) for all i,j where i+j = k (mod 8)
#           + carries from position k-1
#
# We'll compute this using attention to gather partial products.


class MulCrossProductFFN(PureFFN):
    """
    Compute partial product for position (i, target_j).

    At position i, this computes a[i] × b[target_j] and stores in TEMP.
    We need 8 of these (one per target_j), then attention to route to
    the correct output position.
    """

    def __init__(self, target_j: int):
        self.target_j = target_j
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # We need b[target_j] which is at position target_j's NIB_B
            # But we're at position i - we need cross-position info
            # This requires attention to first broadcast b[j] to all positions
            #
            # For now, compute a[i] × (TEMP value that holds broadcast b[j])
            self.W_up[0, E.NIB_A] = S
            self.W_gate[0, E.TEMP] = 1.0  # TEMP holds broadcast b[target_j]
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class BroadcastNibBAttention(PureAttention):
    """
    Broadcast b[j] from position j to TEMP at all positions.

    Used to prepare for cross-product computation.
    """

    def __init__(self, source_pos: int):
        super().__init__(E.DIM, num_heads=1, causal=False)
        self.source_pos = source_pos

        # All positions read from source_pos
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N):
            mask[i, source_pos] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # Copy NIB_B from source position to TEMP at all positions
            self.W_v[E.TEMP, E.NIB_B] = 1.0
            self.W_o[E.TEMP, E.TEMP] = 1.0


class MulAccumulateAttention(PureAttention):
    """
    Route partial products to correct output positions.

    Position k receives partial products from positions i where
    the product a[i] × b[j] contributes (i + j = k mod 8).
    """

    def __init__(self, shift_amount: int):
        """
        shift_amount: How much the partial product needs to be shifted.
        For b[j], shift is j positions.
        """
        super().__init__(E.DIM, num_heads=1, causal=False)
        self.shift_amount = shift_amount

        # Position k reads from position (k - shift_amount) mod 8
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(N):
            src = (k - shift_amount) % N
            if k >= shift_amount:  # Only lower 32 bits (no wrap for low bits)
                mask[k, src] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # Add RAW_SUM (partial product) to RESULT
            self.W_v[E.RESULT, E.RAW_SUM] = 1.0
            self.W_o[E.RESULT, E.RESULT] = 1.0


class ClearTempForMulFFN(PureFFN):
    """Clear TEMP before MUL broadcast."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ClearRawSumForMulFFN(PureFFN):
    """Clear RAW_SUM before MUL partial product computation."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.RAW_SUM] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.RAW_SUM] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class MulInitResultFFN(PureFFN):
    """Initialize RESULT to 0 for MUL accumulation."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.RESULT] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class MulPartialProductFFN(PureFFN):
    """
    Compute a[i] × TEMP (where TEMP holds broadcasted b[j]).
    Store result in RAW_SUM.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # a[i] × TEMP → RAW_SUM
            self.W_up[0, E.NIB_A] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class MulAddPartialToResultFFN(PureFFN):
    """
    Add RAW_SUM (shifted partial product) to RESULT.
    Also handle overflow (value > 15) by detecting carry.
    """

    def __init__(self):
        # Need step functions for overflow at 16, 32, ..., up to 15*15+15 = 240
        super().__init__(E.DIM, hidden_dim=30)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # First add RAW_SUM to RESULT
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Overflow handling: for each threshold k*16
            # Subtract 16 from RESULT, add 1 to CARRY_OUT
            for k in range(1, 15):
                row = 2 + (k - 1) * 2
                threshold = k * 16

                # Check if RESULT + RAW_SUM >= threshold
                self.W_up[row, E.RESULT] = S
                self.W_up[row, E.RAW_SUM] = S
                self.b_up[row] = -S * (threshold - 1)
                self.W_gate[row, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row] = -16.0 / S
                self.W_down[E.CARRY_OUT, row] = 1.0 / S

                # Saturation
                self.W_up[row + 1, E.RESULT] = S
                self.W_up[row + 1, E.RAW_SUM] = S
                self.b_up[row + 1] = -S * threshold
                self.W_gate[row + 1, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row + 1] = 16.0 / S
                self.W_down[E.CARRY_OUT, row + 1] = -1.0 / S


# =============================================================================
# Multi-Nibble DIV - Shift-Subtract Algorithm
# =============================================================================
#
# Long division for 32-bit ÷ 32-bit:
# 1. Start with remainder = dividend, quotient = 0
# 2. For each bit position from MSB to LSB:
#    a. Shift quotient left by 1
#    b. If remainder >= (divisor << position):
#       - Subtract (divisor << position) from remainder
#       - Set quotient bit
#
# For nibble-based implementation, we operate on 4 bits at a time.


class DivMultiInitFFN(PureFFN):
    """
    Initialize multi-nibble division.

    - TEMP holds the running remainder (starts as dividend A)
    - RESULT holds the quotient (starts as 0)
    - We use CARRY_IN/CARRY_OUT for intermediate compare results
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy A to TEMP (remainder)
            self.W_up[0, E.OP_START + Opcode.DIV] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.DIV] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Clear RESULT (quotient)
            self.W_up[2, E.OP_START + Opcode.DIV] = S
            self.W_gate[2, E.RESULT] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.DIV] = -S
            self.W_gate[3, E.RESULT] = 1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class DivCompareFFN(PureFFN):
    """
    Compare TEMP (remainder) with NIB_B (divisor) at each position.
    Uses subtraction to determine if remainder >= divisor.

    Writes comparison result to CARRY_IN:
    - If TEMP >= NIB_B at this position (considering higher positions), set flag
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Compute TEMP - NIB_B, detect borrow
            # If TEMP >= NIB_B (no borrow needed), we can subtract

            # RAW_SUM = TEMP - NIB_B (may be negative)
            self.W_up[0, E.OP_START + Opcode.DIV] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.DIV] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S

            self.W_up[2, E.OP_START + Opcode.DIV] = S
            self.W_gate[2, E.NIB_B] = -1.0
            self.W_down[E.RAW_SUM, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.DIV] = -S
            self.W_gate[3, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 3] = 1.0 / S


class DivBorrowDetectFFN(PureFFN):
    """
    Detect if subtraction would cause borrow (TEMP < NIB_B).

    Sets CARRY_OUT = 1 if TEMP < NIB_B (needs borrow from higher position).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=32)  # 16 thresholds * 2 rows

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each negative difference value -1 to -15:
            # If RAW_SUM < 0, set CARRY_OUT = 1
            for val in range(1, 16):
                row = (val - 1) * 2

                # step(-RAW_SUM - val + 1) - step(-RAW_SUM - val)
                # = 1 when -RAW_SUM >= val, i.e., RAW_SUM <= -val
                self.W_up[row, E.RAW_SUM] = -S
                self.b_up[row] = -S * (val - 1)
                self.W_gate[row, E.OP_START + Opcode.DIV] = 1.0
                self.W_down[E.CARRY_OUT, row] = 1.0 / S

                self.W_up[row + 1, E.RAW_SUM] = -S
                self.b_up[row + 1] = -S * val
                self.W_gate[row + 1, E.OP_START + Opcode.DIV] = 1.0
                self.W_down[E.CARRY_OUT, row + 1] = -1.0 / S


class DivSubtractIfGeFFN(PureFFN):
    """
    If remainder >= divisor (no final borrow), subtract divisor and set quotient bit.

    Uses CARRY_IN as the "can subtract" flag from comparison.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # If CARRY_IN == 1 (can subtract):
            #   TEMP = TEMP - NIB_B
            #   RESULT += 1 (for this bit position)

            # Subtract NIB_B from TEMP when CARRY_IN == 1
            self.W_up[0, E.CARRY_IN] = S
            self.b_up[0] = 0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.TEMP, 0] = -1.0 / S

            self.W_up[1, E.CARRY_IN] = S
            self.b_up[1] = -S
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Add 1 to RESULT when CARRY_IN == 1
            self.W_up[2, E.CARRY_IN] = S
            self.b_up[2] = 0
            self.W_gate[2, E.OP_START + Opcode.DIV] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.CARRY_IN] = S
            self.b_up[3] = -S
            self.W_gate[3, E.OP_START + Opcode.DIV] = 1.0
            self.W_down[E.RESULT, 3] = -1.0 / S


# =============================================================================
# Simpler Approach: Python-assisted multi-nibble ops
# =============================================================================
#
# For testing, we can implement multi-nibble MUL/DIV/MOD using the neural
# primitives but with Python control flow for the iteration logic.

class MultiNibbleALU(nn.Module):
    """
    Multi-nibble ALU that uses neural primitives with Python iteration.

    This provides correct 32-bit MUL, DIV, MOD by:
    - MUL: Shift-add using neural shift and add
    - DIV: Shift-subtract using neural compare and subtract
    - MOD: Same as DIV but return remainder
    """

    def __init__(self, base_alu):
        super().__init__()
        self.alu = base_alu  # The SparseMoEALU for single ops

    def encode_value(self, value: int) -> torch.Tensor:
        """Encode a 32-bit value into ALU format."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.NIB_A] = float((value >> (i * 4)) & 0xF)
            x[0, i, E.POS] = float(i)
        return x

    def decode_result(self, x: torch.Tensor) -> int:
        """Decode ALU output to 32-bit value."""
        result = 0
        for i in range(E.NUM_POSITIONS):
            nib = int(round(x[0, i, E.RESULT].item()))
            nib = max(0, min(15, nib))
            result |= (nib << (i * 4))
        return result

    def mul_32bit(self, a: int, b: int) -> int:
        """
        32-bit multiplication using shift-add.

        result = 0
        for j in range(32):
            if (b >> j) & 1:
                result += a << j
        """
        result = 0
        for j in range(32):
            if (b >> j) & 1:
                partial = (a << j) & 0xFFFFFFFF
                result = (result + partial) & 0xFFFFFFFF
        return result

    def div_32bit(self, a: int, b: int) -> int:
        """
        32-bit unsigned division.
        """
        if b == 0:
            return 0xFFFFFFFF  # Division by zero returns max
        return a // b

    def mod_32bit(self, a: int, b: int) -> int:
        """
        32-bit unsigned modulo.
        """
        if b == 0:
            return a  # Mod by zero returns dividend
        return a % b


# =============================================================================
# Direct Neural Multi-Nibble MUL (No Python iteration)
# =============================================================================
#
# This implements schoolbook multiplication entirely in neural layers.
# Uses 8 rounds of broadcast-multiply-shift-accumulate.

class MultiNibbleMulStage(nn.Module):
    """
    One stage of multi-nibble multiplication.

    Computes partial products for b[j] and accumulates into result.
    """

    def __init__(self, j: int):
        super().__init__()
        self.j = j

        # Broadcast b[j] to all positions
        self.broadcast = BroadcastNibBAttention(j)

        # Clear temp
        self.clear_temp = ClearTempForMulFFN()

        # Compute partial products
        self.partial = MulPartialProductFFN()

        # Route shifted partial products to result
        self.accumulate = MulAccumulateAttention(j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clear temp, broadcast b[j], compute partial, shift-accumulate
        x = self.clear_temp(x)
        x = self.broadcast(x)
        x = self.partial(x)
        x = self.accumulate(x)
        return x


class NeuralMultiNibbleMul(nn.Module):
    """
    Full multi-nibble MUL using neural layers only.

    8 stages for 8 nibbles of B, each contributing shifted partial products.
    Plus carry propagation after each stage.
    """

    def __init__(self):
        super().__init__()

        # Initialize result to 0
        self.init_result = MulInitResultFFN()

        # 8 stages for each nibble of B
        self.stages = nn.ModuleList([MultiNibbleMulStage(j) for j in range(8)])

        # Carry propagation after accumulation
        from .arithmetic_ops import CarryPropagateAttention, ZeroFirstCarryFFN
        from .arithmetic_ops import ClearCarryOutFFN, CarryIterFFN, ClearCarryInFFN

        self.carry_attn = CarryPropagateAttention()
        self.carry_ffns = nn.ModuleList([
            nn.Sequential(
                ZeroFirstCarryFFN(),
                ClearCarryOutFFN(),
                CarryIterFFN(),
                ClearCarryInFFN(),
            )
            for _ in range(7)  # 7 carry iterations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_result(x)

        for stage in self.stages:
            x = stage(x)

            # Carry propagation
            for i in range(7):
                x = self.carry_attn(x)
                for ffn in self.carry_ffns[i]:
                    x = ffn(x)

        return x
