"""
Arithmetic operations for Neural VM V7.

ADD, SUB with carry/borrow propagation.
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


# =============================================================================
# ADD Operations
# =============================================================================

class AddRawSumFFN(PureFFN):
    """Computes raw_sum = nib_a + nib_b for each position."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.NIB_B] = S
            self.W_gate[0, E.OP_START + Opcode.ADD] = S
            self.W_down[E.RAW_SUM, 0] = 1.0 / (S * S)

            self.W_up[1, E.NIB_A] = -S
            self.W_up[1, E.NIB_B] = -S
            self.W_gate[1, E.OP_START + Opcode.ADD] = -S
            self.W_down[E.RAW_SUM, 1] = 1.0 / (S * S)


class InitResultFFN(PureFFN):
    """Initialize RESULT = RAW_SUM mod 16."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy raw_sum to result
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Subtract 16 when raw_sum >= 16
            self.W_up[2, E.RAW_SUM] = S
            self.b_up[2] = -S * 15.0
            self.W_gate[2, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 2] = -16.0 / S

            self.W_up[3, E.RAW_SUM] = S
            self.b_up[3] = -S * 16.0
            self.W_gate[3, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 3] = 16.0 / S


class CarryDetectFFN(PureFFN):
    """Detects if raw_sum >= 16, sets carry_out to ~1."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.RAW_SUM] = S
            self.b_up[0] = -S * 15.0
            self.W_gate[0, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = S
            self.b_up[1] = -S * 16.0
            self.W_gate[1, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S


class CarryPropagateAttention(PureAttention):
    """
    Each position i gets carry_in from position i-1's carry_out.
    Uses a "previous position only" mask.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(1, N):
            mask[i, i-1] = 0.0
        mask[0, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.CARRY_IN, E.CARRY_OUT] = 1.0
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class ZeroFirstCarryFFN(PureFFN):
    """Zeros out carry_in for position 0."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


class ClearCarryOutFFN(PureFFN):
    """Clears carry_out before detecting new carries."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class CarryIterFFN(PureFFN):
    """One iteration of carry propagation.

    Integrated clearing: clears old CARRY_OUT and CARRY_IN so carries
    don't accumulate across iterations. In SwiGLU with residual:
    old + (-old + new) = new.

    Loop order is: Propagate → ZeroFirst → Iter, so CARRY_IN is valid
    when this layer runs (pos 0 already zeroed by ZeroFirstCarryFFN).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Units 0-1: Add carry_in to result (cancel pair)
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Units 2-3: Detect overflow (result + carry_in >= 16)
            # Subtract 16 from RESULT, add 1 to CARRY_OUT
            self.W_up[2, E.RESULT] = S
            self.W_up[2, E.CARRY_IN] = S
            self.b_up[2] = -S * 15.0
            self.W_gate[2, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 2] = -16.0 / S
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            self.W_up[3, E.RESULT] = S
            self.W_up[3, E.CARRY_IN] = S
            self.b_up[3] = -S * 16.0
            self.W_gate[3, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 3] = 16.0 / S
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S

            # Units 4-5: Clear old CARRY_OUT (cancel pair: output = -old_carry_out)
            self.W_up[4, E.OP_START + Opcode.ADD] = S
            self.W_gate[4, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 4] = 1.0 / S

            self.W_up[5, E.OP_START + Opcode.ADD] = -S
            self.W_gate[5, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 5] = 1.0 / S

            # Units 6-7: Clear old CARRY_IN (cancel pair: output = -old_carry_in)
            self.W_up[6, E.OP_START + Opcode.ADD] = S
            self.W_gate[6, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 6] = 1.0 / S

            self.W_up[7, E.OP_START + Opcode.ADD] = -S
            self.W_gate[7, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 7] = 1.0 / S


class ClearCarryInFFN(PureFFN):
    """Clears carry_in before next propagation round."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


# =============================================================================
# SUB Operations
# =============================================================================

class SubRawDiffFFN(PureFFN):
    """Computes raw_diff = nib_a - nib_b for each position."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_gate[0, E.NIB_B] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class SubInitResultFFN(PureFFN):
    """Initialize RESULT = RAW_DIFF mod 16."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy raw_diff to result
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Add 16 when raw_diff < 0
            self.W_up[2, E.RAW_SUM] = -S
            self.b_up[2] = -S * 0.0
            self.W_gate[2, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 2] = 16.0 / S

            self.W_up[3, E.RAW_SUM] = -S
            self.b_up[3] = -S * 1.0
            self.W_gate[3, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 3] = -16.0 / S


class BorrowDetectFFN(PureFFN):
    """Detects if raw_diff < 0, sets carry_out (borrow) to ~1."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = -S * 0.0
            self.W_gate[0, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S


class ZeroFirstBorrowFFN(PureFFN):
    """Zeros out carry_in (borrow) for position 0 in SUB."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


class ClearBorrowOutFFN(PureFFN):
    """Clears carry_out (borrow) before detecting new borrows."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class BorrowIterFFN(PureFFN):
    """One iteration of borrow propagation.

    Integrated clearing: clears old CARRY_OUT and CARRY_IN so borrows
    don't accumulate across iterations.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Units 0-1: Subtract borrow_in from result (cancel pair)
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Units 2-3: Detect underflow (result - borrow_in < 0)
            # Add 16 to RESULT, set CARRY_OUT = 1
            self.W_up[2, E.RESULT] = -S
            self.W_up[2, E.CARRY_IN] = S
            self.b_up[2] = 0.0
            self.W_gate[2, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 2] = 16.0 / S
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            self.W_up[3, E.RESULT] = -S
            self.W_up[3, E.CARRY_IN] = S
            self.b_up[3] = -S * 1.0
            self.W_gate[3, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 3] = -16.0 / S
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S

            # Units 4-5: Clear old CARRY_OUT (cancel pair)
            self.W_up[4, E.OP_START + Opcode.SUB] = S
            self.W_gate[4, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 4] = 1.0 / S

            self.W_up[5, E.OP_START + Opcode.SUB] = -S
            self.W_gate[5, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 5] = 1.0 / S

            # Units 6-7: Clear old CARRY_IN (cancel pair)
            self.W_up[6, E.OP_START + Opcode.SUB] = S
            self.W_gate[6, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 6] = 1.0 / S

            self.W_up[7, E.OP_START + Opcode.SUB] = -S
            self.W_gate[7, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 7] = 1.0 / S


class ClearBorrowInFFN(PureFFN):
    """Clears carry_in (borrow) before next propagation round."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S
