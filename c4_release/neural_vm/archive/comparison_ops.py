"""
Comparison operations for Neural VM V7.

EQ, NE, LT, GT, LE, GE with multi-nibble borrow propagation.
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention
from .arithmetic_ops import CarryPropagateAttention


# =============================================================================
# EQ/NE Operations
# =============================================================================

class CompareDiffFFN(PureFFN):
    """Compute per-nibble difference: diff = nib_a - nib_b."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_gate[0, E.NIB_B] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class ClearRawSumFFN(PureFFN):
    """Clear RAW_SUM slot before attention-based reduction."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.RAW_SUM] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.RAW_SUM] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class CompareEqNibbleFFN(PureFFN):
    """Detect if diff == 0 for each nibble using integer-aligned thresholds."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: step(-diff + 1)
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = S * 1.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            # Row 1: Saturation
            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = 0.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # Row 2: step(-diff)
            self.W_up[2, E.RAW_SUM] = -S
            self.b_up[2] = 0.0
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 2] = -1.0 / S

            # Row 3: Saturation
            self.W_up[3, E.RAW_SUM] = -S
            self.b_up[3] = -S * 1.0
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class CompareNeNibbleFFN(PureFFN):
    """Detect if diff != 0 for each nibble."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For diff >= 1
            self.W_up[0, E.RAW_SUM] = S
            self.b_up[0] = 0.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # For diff <= -1
            self.W_up[2, E.RAW_SUM] = -S
            self.b_up[2] = 0.0
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, E.RAW_SUM] = -S
            self.b_up[3] = -S * 1.0
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 3] = -1.0 / S


class CompareLtNibbleFFN(PureFFN):
    """Detect if diff < 0 (a < b) for each nibble."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = -S * 0.5
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.5
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class CompareGtNibbleFFN(PureFFN):
    """Detect if diff > 0 (a > b) for each nibble."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.RAW_SUM] = S
            self.b_up[0] = -S * 0.5
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = S
            self.b_up[1] = -S * 1.5
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class CompareReduceEqAttention(PureAttention):
    """Sum per-nibble EQ results using attention."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0
        for i in range(1, N):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_v[E.RAW_SUM, E.TEMP] = 8.0
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class CompareReduceEqFFN(PureFFN):
    """Threshold the sum of per-nibble EQ results."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear RESULT at all positions
            self.W_up[0, E.OP_START + Opcode.EQ] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Threshold with position gating
            self.W_up[1, E.RAW_SUM] = S
            self.W_up[1, E.POS] = -S * 100.0
            self.b_up[1] = -S * 7.0
            self.W_gate[1, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Saturation
            self.W_up[2, E.RAW_SUM] = S
            self.W_up[2, E.POS] = -S * 100.0
            self.b_up[2] = -S * 8.0
            self.W_gate[2, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.RESULT, 2] = -1.0 / S


class CompareReduceNeAttention(PureAttention):
    """Sum per-nibble NE results using attention."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0
        for i in range(1, N):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_v[E.RAW_SUM, E.TEMP] = 8.0
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class CompareReduceNeFFN(PureFFN):
    """Threshold the sum of per-nibble NE results."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear RESULT
            self.W_up[0, E.OP_START + Opcode.NE] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Threshold with position gating
            self.W_up[1, E.RAW_SUM] = S
            self.W_up[1, E.POS] = -S * 100.0
            self.b_up[1] = 0.0
            self.W_gate[1, E.OP_START + Opcode.NE] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Saturation
            self.W_up[2, E.RAW_SUM] = S
            self.W_up[2, E.POS] = -S * 100.0
            self.b_up[2] = -S * 1.0
            self.W_gate[2, E.OP_START + Opcode.NE] = 1.0
            self.W_down[E.RESULT, 2] = -1.0 / S


class CompareCopyResultFFN(PureFFN):
    """Copy TEMP to RESULT for LT/GT/LE/GE."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


# =============================================================================
# Multi-nibble comparison with borrow propagation
# =============================================================================

class CmpRawDiffFFN(PureFFN):
    """Compute raw diff = a - b for comparison."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_gate[0, E.NIB_B] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class CmpRawDiffSwapFFN(PureFFN):
    """Compute raw diff = b - a for GT comparison (swap operands)."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_gate[0, E.NIB_A] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_gate[1, E.NIB_A] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class CmpBorrowDetectFFN(PureFFN):
    """Detect if raw_diff < 0 for comparison."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = 0.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S


class CmpZeroFirstBorrowFFN(PureFFN):
    """Zero out borrow_in for position 0."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


class CmpClearBorrowOutFFN(PureFFN):
    """Clear CARRY_OUT before detecting new borrows."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class CmpBorrowIterFFN(PureFFN):
    """One iteration of borrow propagation for comparison.

    Integrated clearing: clears old CARRY_OUT and CARRY_IN so borrows
    don't accumulate across iterations.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Units 0-1: Detect if (RAW_SUM - borrow_in) < 0
            self.W_up[0, E.CARRY_IN] = S
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = 0.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.CARRY_IN] = S
            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S

            # Units 2-3: Clear old CARRY_OUT (cancel pair)
            self.W_up[2, E.OP_START + self.opcode] = S
            self.W_gate[2, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + self.opcode] = -S
            self.W_gate[3, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 3] = 1.0 / S

            # Units 4-5: Clear old CARRY_IN (cancel pair)
            self.W_up[4, E.OP_START + self.opcode] = S
            self.W_gate[4, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 4] = 1.0 / S

            self.W_up[5, E.OP_START + self.opcode] = -S
            self.W_gate[5, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 5] = 1.0 / S


class CmpClearBorrowInFFN(PureFFN):
    """Clear CARRY_IN after using it."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


class CmpClearTempFFN(PureFFN):
    """Clear TEMP at all positions."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class CmpExtractMSBBorrowFFN(PureFFN):
    """Extract the final borrow at MSB (position 7) to TEMP."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Only activate at position 7
            self.W_up[0, E.POS] = S
            self.b_up[0] = -S * 6.0
            self.W_gate[0, E.CARRY_OUT] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.POS] = S
            self.b_up[1] = -S * 7.0
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class CmpClearResultFFN(PureFFN):
    """Clear RESULT at all positions."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.RESULT] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class CmpBroadcastResultAttention(PureAttention):
    """Copy TEMP from position 7 to RESULT at position 0."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        mask[0, N-1] = 0.0
        for i in range(1, N):
            mask[i, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.TEMP, E.TEMP] = 1.0
            self.W_o[E.RESULT, E.TEMP] = 1.0


class CmpInvertResultFFN(PureFFN):
    """Invert RESULT for LE (from GT) and GE (from LT)."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Add 1 at pos=0 only
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / (S * 0.5)

            self.W_up[1, E.POS] = -S
            self.b_up[1] = -S * 0.5
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / (S * 0.5)

            # Subtract 2*RESULT at pos=0
            self.W_up[2, E.POS] = -S
            self.b_up[2] = S * 0.5
            self.W_gate[2, E.RESULT] = -2.0
            self.W_down[E.RESULT, 2] = 1.0 / (S * 0.5)

            self.W_up[3, E.POS] = -S
            self.b_up[3] = -S * 0.5
            self.W_gate[3, E.RESULT] = 2.0
            self.W_down[E.RESULT, 3] = 1.0 / (S * 0.5)
