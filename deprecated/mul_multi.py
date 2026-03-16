"""
Multi-nibble MUL using schoolbook algorithm.

For A × B where A and B are 8-nibble (32-bit) numbers:
  result[k] = Σ(a[i] × b[j]) for all i,j where i+j = k
            + carries from position k-1

Algorithm:
1. Initialize accumulator to 0
2. For j = 0 to 7:
   a. Broadcast b[j] to all positions (via attention)
   b. Compute partial products: pp[i] = a[i] × b[j]
   c. Add pp to accumulator at positions [j, j+1, ..., 7]
   d. Propagate carries
3. Result is in accumulator (lower 32 bits)
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


class MulBroadcastBjAttention(PureAttention):
    """
    Broadcast b[j] from position j to TEMP at all positions.

    This is parameterized by source position j.
    All positions read NIB_B from position j.
    """

    def __init__(self, source_j: int):
        super().__init__(E.DIM, num_heads=1, causal=False)
        self.source_j = source_j

        # All positions read from source_j
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N):
            mask[i, source_j] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # Copy NIB_B from source position to TEMP at all positions
            self.W_v[E.TEMP, E.NIB_B] = 1.0
            self.W_o[E.TEMP, E.TEMP] = 1.0


class MulClearTempFFN(PureFFN):
    """Clear TEMP before broadcast."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.b_up[0] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class MulPartialProductFFN(PureFFN):
    """
    Compute partial product: RAW_SUM = a[i] × TEMP (where TEMP = b[j]).

    Uses: silu(S*a) × TEMP / S ≈ a × TEMP when a > 0
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # RAW_SUM = NIB_A × TEMP
            self.W_up[0, E.NIB_A] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class MulClearRawSumFFN(PureFFN):
    """Clear RAW_SUM before computing partial product."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.b_up[0] = S
            self.W_gate[0, E.RAW_SUM] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.RAW_SUM] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class MulShiftAddAttention(PureAttention):
    """
    Shift partial products by j positions and add to accumulator.

    Position k reads RAW_SUM from position k-j (if k >= j).
    Positions 0 to j-1 read nothing (get 0).
    """

    def __init__(self, shift_j: int):
        super().__init__(E.DIM, num_heads=1, causal=False)
        self.shift_j = shift_j

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(shift_j, N):
            src = k - shift_j
            mask[k, src] = 0.0
        # Positions 0 to shift_j-1 have no valid source
        # Use position 0 as dummy (will be masked by FFN)
        for k in range(shift_j):
            mask[k, k] = 0.0  # Read from self (will be zeroed)
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # Read RAW_SUM, add to CARRY_IN (temp accumulator)
            self.W_v[E.CARRY_IN, E.RAW_SUM] = 1.0
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class MulZeroLowPositionsFFN(PureFFN):
    """
    Zero out positions below shift_j in CARRY_IN.

    These positions shouldn't receive partial products from this iteration.
    """

    def __init__(self, shift_j: int):
        self.shift_j = shift_j
        super().__init__(E.DIM, hidden_dim=2 * shift_j if shift_j > 0 else 2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            if self.shift_j == 0:
                # No positions to zero
                return

            # For each position k < shift_j, zero out CARRY_IN
            for k in range(self.shift_j):
                row = k * 2
                # Active when POS == k
                # step(POS - k + 0.5) - step(POS - k - 0.5) = 1 when POS = k
                self.W_up[row, E.POS] = S
                self.b_up[row] = -S * (k - 0.5)
                self.W_gate[row, E.CARRY_IN] = -1.0
                self.W_down[E.CARRY_IN, row] = 1.0 / S

                self.W_up[row + 1, E.POS] = S
                self.b_up[row + 1] = -S * (k + 0.5)
                self.W_gate[row + 1, E.CARRY_IN] = 1.0
                self.W_down[E.CARRY_IN, row + 1] = 1.0 / S


class MulAccumulateFFN(PureFFN):
    """
    Add CARRY_IN (shifted partial products) to RESULT (accumulator).
    Handle overflow by detecting and propagating carries.
    """

    def __init__(self):
        # Need to handle sums up to 15*15 + 15*15 + ... = potentially large
        # For safety, handle overflow at 16, 32, ..., 240
        super().__init__(E.DIM, hidden_dim=32)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Add CARRY_IN to RESULT
            self.b_up[0] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Overflow handling: for each threshold k*16
            for k in range(1, 16):
                row = 2 + (k - 1) * 2
                threshold = k * 16

                # When RESULT + CARRY_IN >= threshold, subtract 16 and set carry
                self.W_up[row, E.RESULT] = S
                self.W_up[row, E.CARRY_IN] = S
                self.b_up[row] = -S * (threshold - 1)
                self.W_gate[row, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row] = -16.0 / S
                self.W_down[E.CARRY_OUT, row] = 1.0 / S

                self.W_up[row + 1, E.RESULT] = S
                self.W_up[row + 1, E.CARRY_IN] = S
                self.b_up[row + 1] = -S * threshold
                self.W_gate[row + 1, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row + 1] = 16.0 / S
                self.W_down[E.CARRY_OUT, row + 1] = -1.0 / S


class MulClearCarryInFFN(PureFFN):
    """Clear CARRY_IN before next iteration."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.b_up[0] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


class MultiNibbleMul(nn.Module):
    """
    Complete multi-nibble MUL using schoolbook algorithm.

    Computes A × B for 32-bit operands, returns lower 32 bits.
    """

    def __init__(self):
        super().__init__()

        # Layers for each iteration j
        self.clear_temp = MulClearTempFFN()
        self.clear_raw_sum = MulClearRawSumFFN()
        self.clear_carry_in = MulClearCarryInFFN()

        # Broadcast b[j] for each j
        self.broadcasts = nn.ModuleList([
            MulBroadcastBjAttention(j) for j in range(8)
        ])

        # Partial product computation
        self.partial_product = MulPartialProductFFN()

        # Shift-add for each shift amount
        self.shift_adds = nn.ModuleList([
            MulShiftAddAttention(j) for j in range(8)
        ])

        # Zero low positions for each shift
        self.zero_lows = nn.ModuleList([
            MulZeroLowPositionsFFN(j) for j in range(8)
        ])

        # Accumulate with overflow
        self.accumulate = MulAccumulateFFN()

        # Carry propagation
        from .arithmetic_ops import CarryPropagateAttention
        self.carry_attn = CarryPropagateAttention()

        # Initialize result to 0
        self.init_result = MulInitResultFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize RESULT to 0
        x = self.init_result(x)

        # For each nibble j of B
        for j in range(8):
            # Clear temporaries
            x = self.clear_temp(x)
            x = self.clear_raw_sum(x)
            x = self.clear_carry_in(x)

            # Broadcast b[j] to all positions
            x = self.broadcasts[j](x)

            # Compute partial products: a[i] × b[j]
            x = self.partial_product(x)

            # Shift and add to accumulator
            x = self.shift_adds[j](x)
            x = self.zero_lows[j](x)
            x = self.accumulate(x)

            # Carry propagation (7 iterations)
            for _ in range(7):
                x = self.carry_attn(x)

        return x


class MulInitResultFFN(PureFFN):
    """Initialize RESULT to 0 for MUL."""

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
