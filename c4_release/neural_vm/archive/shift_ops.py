"""
Shift operations for Neural VM V7.

SHL, SHR - variable bit shift using binary decomposition.

Shift by b bits = conditional shifts by 16, 8, 4, 2, 1 bits based on bits of b.
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


class ClearTempBeforeShiftFFN(PureFFN):
    """
    Unconditionally clear TEMP before shift operations.
    Prevents garbage from affecting shift via attention.
    """

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


class ShiftLeftCopyFFN(PureFFN):
    """Copy NIB_A to TEMP, gated on SHL opcode."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHL] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHL] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ShiftLeftAttention(PureAttention):
    """
    Shift TEMP values left by one position (higher nibble positions).
    Position i reads TEMP from position i-1.
    Position 0 copies from itself (will be zeroed by subsequent FFN).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        # Position i reads from position i-1 (shift up)
        for i in range(1, N):
            mask[i, i-1] = 0.0
        # Position 0 copies from itself (will be zeroed separately)
        mask[0, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.CARRY_IN, E.TEMP] = 1.0
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class ClearCarryInFFNShift(PureFFN):
    """Clear CARRY_IN before shift attention to prevent accumulation."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear CARRY_IN unconditionally
            self.b_up[0] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


class ZeroShiftPos0FFN(PureFFN):
    """Zero out CARRY_IN at position 0 after shift attention (for SHL)."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # CARRY_IN[pos 0] = 0
            # At position 0: subtract CARRY_IN from itself
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5  # Active when POS < 0.5 (i.e., POS == 0)
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / (S * 0.5)


class ZeroShiftPos7FFN(PureFFN):
    """Zero out CARRY_IN at position 7 after shift attention (for SHR)."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # CARRY_IN[pos 7] = 0
            # At position 7: subtract CARRY_IN from itself
            self.W_up[0, E.POS] = S
            self.b_up[0] = -S * 6.5  # Active when POS > 6.5 (i.e., POS == 7)
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / (S * 0.5)


class CopyCarryToTempFFN(PureFFN):
    """Copy CARRY_IN to TEMP for next shift iteration."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear TEMP first
            self.b_up[0] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Copy CARRY_IN to TEMP
            self.b_up[2] = S
            self.W_gate[2, E.CARRY_IN] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.b_up[3] = -S
            self.W_gate[3, E.CARRY_IN] = -1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class ShiftLeftResultFFN(PureFFN):
    """Copy CARRY_IN (shifted values) to RESULT, gated on SHL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHL] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHL] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Zero CARRY_IN at position 0
            self.W_up[2, E.POS] = -S
            self.b_up[2] = S * 0.5
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / (S * 0.5)


class ShiftLeftClearFFN(PureFFN):
    """Clear TEMP and CARRY_IN after SHL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHL] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHL] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            self.W_up[2, E.OP_START + Opcode.SHL] = S
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.SHL] = -S
            self.W_gate[3, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 3] = 1.0 / S


class ShiftRightCopyFFN(PureFFN):
    """Copy NIB_A to TEMP, gated on SHR opcode."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHR] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHR] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ShiftRightAttention(PureAttention):
    """
    Shift TEMP values right by one position.
    Position i reads TEMP from position i+1.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N-1):
            mask[i, i+1] = 0.0
        mask[N-1, N-1] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.CARRY_IN, E.TEMP] = 1.0
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class ShiftRightResultFFN(PureFFN):
    """Copy CARRY_IN (shifted values) to RESULT, gated on SHR."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHR] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHR] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Zero out position 7 using integer-aligned thresholds
            self.W_up[2, E.POS] = S
            self.b_up[2] = -S * 6.0
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.POS] = S
            self.b_up[3] = -S * 7.0
            self.W_gate[3, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class ShiftRightClearFFN(PureFFN):
    """Clear TEMP and CARRY_IN after SHR."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHR] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHR] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            self.W_up[2, E.OP_START + Opcode.SHR] = S
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.SHR] = -S
            self.W_gate[3, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 3] = 1.0 / S
