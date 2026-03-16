"""
Variable bit shift operations for Neural VM V7.

SHL, SHR with arbitrary shift amounts (0-31 bits).

Uses binary decomposition for nibble-aligned shifts (multiples of 4 bits):
- Bit 2 of shift amount: shift by 4 bits (1 nibble)
- Bit 3 of shift amount: shift by 8 bits (2 nibbles)
- Bit 4 of shift amount: shift by 16 bits (4 nibbles)

Each bit of the shift amount controls a conditional shift stage.
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


# =============================================================================
# Sub-nibble shift helpers (1-bit and 2-bit shifts)
# =============================================================================

class ExtractShiftBit0FFN(PureFFN):
    """
    Extract bit 0 from NIB_B (shift amount).
    Bit 0 = 1 when NIB_B is odd.
    Result stored in CARRY_OUT for immediate broadcast.

    Uses paired step functions for saturation:
    step(x >= k) = pair of (rise at k-1, fall at k) that saturates to 1.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        # 15 thresholds * 2 rows each = 30 hidden units
        super().__init__(E.DIM, hidden_dim=30)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # bit0 = step(>=1) - step(>=2) + step(>=3) - step(>=4) + ...
            # Each step needs 2 rows (rise at k-1, saturate at k)
            row = 0
            for k in range(1, 16):
                sign = 1 if (k % 2 == 1) else -1

                # Rise at k-1
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * (k - 1)
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = sign / S
                row += 1

                # Saturate at k
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * k
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = -sign / S
                row += 1


class ExtractShiftBit1FFN(PureFFN):
    """
    Extract bit 1 from NIB_B (shift amount).
    Bit 1 = 1 when NIB_B in {2,3,6,7,10,11,14,15}.
    Result stored in CARRY_OUT for immediate broadcast.

    Uses paired step functions for saturation.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        # 7 thresholds * 2 rows each = 14 hidden units
        super().__init__(E.DIM, hidden_dim=14)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(x>=2) - step(x>=4) + step(x>=6) - step(x>=8) + ...
            thresholds = [(2, 1), (4, -1), (6, 1), (8, -1), (10, 1), (12, -1), (14, 1)]
            row = 0
            for thresh, sign in thresholds:
                # Rise at thresh-1
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * (thresh - 1)
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = sign / S
                row += 1

                # Saturate at thresh
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * thresh
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = -sign / S
                row += 1


class SubNibbleShiftLeft1FFN(PureFFN):
    """
    Shift TEMP left by 1 bit (multiply by 2, overflow to CARRY_OUT).

    For each nibble:
    - new_value = (nibble * 2) mod 16
    - carry_out = 1 if nibble >= 8 else 0

    Uses IO_EXIT_CODE slot for carry flag (not TEMP+1 which overlaps OP_START!).
    The carry flag output is GATED by IO_INPUT_READY (shift bit 0) to prevent
    accumulation when this shift stage isn't active.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        CARRY_FLAG = E.IO_EXIT_CODE  # Use this slot for carry flag
        SHIFT_BIT = E.IO_INPUT_READY  # Bit 0 is stored here
        with torch.no_grad():
            # Part 1: Double TEMP -> CARRY_OUT
            self.W_up[0, E.TEMP] = S
            self.b_gate[0] = 2.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.TEMP] = -S
            self.b_gate[1] = -2.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S

            # Part 2: Detect overflow (TEMP >= 8), subtract 16
            self.W_up[2, E.TEMP] = S
            self.b_up[2] = -S * 7.0
            self.b_gate[2] = 1.0
            self.W_down[E.CARRY_OUT, 2] = -16.0 / S

            self.W_up[3, E.TEMP] = S
            self.b_up[3] = -S * 8.0
            self.b_gate[3] = 1.0
            self.W_down[E.CARRY_OUT, 3] = 16.0 / S

            # Part 3: Set carry flag (1 if TEMP >= 8) in IO_EXIT_CODE slot
            # GATED by shift bit to prevent accumulation
            self.W_up[4, E.TEMP] = S
            self.b_up[4] = -S * 7.0
            self.W_gate[4, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 4] = 1.0 / S

            self.W_up[5, E.TEMP] = S
            self.b_up[5] = -S * 8.0
            self.W_gate[5, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 5] = -1.0 / S


class SubNibbleCarryLeftAttention(PureAttention):
    """
    Route carry from position k-1 to position k for sub-nibble left shift.
    Position k reads IO_EXIT_CODE (carry flag) from position k-1.
    Position 0 reads from position 7 (which should have carry=0).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(1, N):
            mask[i, i - 1] = 0.0
        # Position 0 reads from position N-1 (which has carry=0 for most cases)
        mask[0, N - 1] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        CARRY_FLAG = E.IO_EXIT_CODE
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            # Read IO_EXIT_CODE (carry flag) from source, add to CARRY_OUT
            self.W_v[E.CARRY_OUT, CARRY_FLAG] = 1.0
            self.W_o[E.CARRY_OUT, E.CARRY_OUT] = 1.0


class ZeroPosition0CarryFFN(PureFFN):
    """Zero CARRY_OUT at position 0 after sub-nibble carry propagation."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Active when POS < 0.5 (i.e., position 0)
            # CARRY_OUT -= CARRY_OUT when at position 0
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 2.0 / S

            self.W_up[1, E.POS] = -S
            self.b_up[1] = -S * 0.5
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 2.0 / S


class CopyCarryOutToTempFFN(PureFFN):
    """Copy CARRY_OUT back to TEMP after sub-nibble shift."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear TEMP
            self.b_up[0] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Copy CARRY_OUT to TEMP
            self.b_up[2] = S
            self.W_gate[2, E.CARRY_OUT] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.b_up[3] = -S
            self.W_gate[3, E.CARRY_OUT] = -1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class SubNibbleShiftLeft2FFN(PureFFN):
    """
    Shift TEMP left by 2 bits (multiply by 4, overflow to CARRY_OUT).

    For each nibble:
    - new_value = (nibble * 4) mod 16
    - carry_out = nibble // 4 (0, 1, 2, or 3)

    Uses IO_EXIT_CODE slot for carry flag.
    The carry flag output is GATED by IO_NEED_INPUT (shift bit 1) to prevent
    accumulation when this shift stage isn't active.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=16)  # Increased for full TEMP >= 12 support

    def _bake_weights(self):
        S = E.SCALE
        CARRY_FLAG = E.IO_EXIT_CODE
        SHIFT_BIT = E.IO_NEED_INPUT  # Bit 1 is stored here
        with torch.no_grad():
            # Part 1: Multiply TEMP by 4 -> CARRY_OUT
            self.W_up[0, E.TEMP] = S
            self.b_gate[0] = 4.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.TEMP] = -S
            self.b_gate[1] = -4.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S

            # Part 2: Detect overflow levels and subtract 16
            # TEMP >= 4: subtract 16
            self.W_up[2, E.TEMP] = S
            self.b_up[2] = -S * 3.0
            self.b_gate[2] = 1.0
            self.W_down[E.CARRY_OUT, 2] = -16.0 / S

            self.W_up[3, E.TEMP] = S
            self.b_up[3] = -S * 4.0
            self.b_gate[3] = 1.0
            self.W_down[E.CARRY_OUT, 3] = 16.0 / S

            # TEMP >= 8: subtract another 16
            self.W_up[4, E.TEMP] = S
            self.b_up[4] = -S * 7.0
            self.b_gate[4] = 1.0
            self.W_down[E.CARRY_OUT, 4] = -16.0 / S

            self.W_up[5, E.TEMP] = S
            self.b_up[5] = -S * 8.0
            self.b_gate[5] = 1.0
            self.W_down[E.CARRY_OUT, 5] = 16.0 / S

            # TEMP >= 12: subtract another 16
            self.W_up[6, E.TEMP] = S
            self.b_up[6] = -S * 11.0
            self.b_gate[6] = 1.0
            self.W_down[E.CARRY_OUT, 6] = -16.0 / S

            self.W_up[7, E.TEMP] = S
            self.b_up[7] = -S * 12.0
            self.b_gate[7] = 1.0
            self.W_down[E.CARRY_OUT, 7] = 16.0 / S

            # Part 3: Compute carry (TEMP // 4) = step(>=4) + step(>=8) + step(>=12)
            # All carry flag outputs GATED by shift bit to prevent accumulation
            # Step for >= 4
            self.W_up[8, E.TEMP] = S
            self.b_up[8] = -S * 3.0
            self.W_gate[8, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 8] = 1.0 / S

            self.W_up[9, E.TEMP] = S
            self.b_up[9] = -S * 4.0
            self.W_gate[9, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 9] = -1.0 / S

            # Step for >= 8
            self.W_up[10, E.TEMP] = S
            self.b_up[10] = -S * 7.0
            self.W_gate[10, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 10] = 1.0 / S

            self.W_up[11, E.TEMP] = S
            self.b_up[11] = -S * 8.0
            self.W_gate[11, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 11] = -1.0 / S

            # Step for >= 12
            self.W_up[12, E.TEMP] = S
            self.b_up[12] = -S * 11.0
            self.W_gate[12, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 12] = 1.0 / S

            self.W_up[13, E.TEMP] = S
            self.b_up[13] = -S * 12.0
            self.W_gate[13, SHIFT_BIT] = 1.0  # Gate by shift bit!
            self.W_down[CARRY_FLAG, 13] = -1.0 / S


# =============================================================================
# Sub-nibble shift RIGHT (1-bit and 2-bit)
# =============================================================================

class SubNibbleShiftRight1FFN(PureFFN):
    """
    Shift TEMP right by 1 bit: compute floor(TEMP / 2) -> CARRY_OUT.

    For each nibble:
    - new_value = floor(nibble / 2) + 8 * (borrow from position i+1)

    The borrow (TEMP mod 2) is computed in a SEPARATE layer using:
    borrow = TEMP - 2 * floor(TEMP/2)

    This is more efficient than enumerating all odd values.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=14)  # Just floor(x/2): 7 thresholds * 2

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Compute floor(TEMP / 2) -> CARRY_OUT
            # floor(x/2) = sum of step(x >= 2k) for k = 1..7
            # = step(>=2) + step(>=4) + step(>=6) + ... + step(>=14)
            row = 0
            for k in range(1, 8):  # thresholds at 2, 4, 6, 8, 10, 12, 14
                threshold = k * 2
                # Rise at threshold - 1
                self.W_up[row, E.TEMP] = S
                self.b_up[row] = -S * (threshold - 1)
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = 1.0 / S
                row += 1
                # Saturate at threshold
                self.W_up[row, E.TEMP] = S
                self.b_up[row] = -S * threshold
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = -1.0 / S
                row += 1


class SubNibbleMod2FFN(PureFFN):
    """
    Compute TEMP mod 2 using floor-based formula: mod = TEMP - 2*CARRY_OUT.

    CARRY_OUT already contains floor(TEMP/2) from previous layer.
    Result goes to IO_EXIT_CODE (borrow flag), gated by shift bit.

    This is nearly linear: silu(S) * (TEMP - 2*CARRY_OUT) / S ≈ TEMP - 2*CARRY_OUT
    """

    def __init__(self, shift_bit_slot: int):
        self.shift_bit_slot = shift_bit_slot
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        BORROW_FLAG = E.IO_EXIT_CODE
        with torch.no_grad():
            # Compute: BORROW_FLAG = TEMP - 2*CARRY_OUT, gated by shift bit
            # Use: silu(S * shift_bit) * (TEMP - 2*CARRY_OUT) / S

            # Positive term: +shift_bit * TEMP
            self.W_up[0, self.shift_bit_slot] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[BORROW_FLAG, 0] = 1.0 / S

            self.W_up[1, self.shift_bit_slot] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[BORROW_FLAG, 1] = 1.0 / S

            # Negative term: -shift_bit * 2*CARRY_OUT
            self.W_up[2, self.shift_bit_slot] = S
            self.W_gate[2, E.CARRY_OUT] = -2.0
            self.W_down[BORROW_FLAG, 2] = 1.0 / S

            self.W_up[3, self.shift_bit_slot] = -S
            self.W_gate[3, E.CARRY_OUT] = 2.0
            self.W_down[BORROW_FLAG, 3] = 1.0 / S


class SubNibbleShiftRight2FFN(PureFFN):
    """
    Shift TEMP right by 2 bits: compute floor(TEMP / 4) -> CARRY_OUT.

    For each nibble:
    - new_value = floor(nibble / 4) + 4 * (borrow from position i+1)

    The borrow (TEMP mod 4) is computed in a SEPARATE layer using:
    borrow = TEMP - 4 * floor(TEMP/4)

    This is more efficient than enumerating all values.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=6)  # Just floor(x/4): 3 thresholds * 2

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Compute floor(TEMP / 4) -> CARRY_OUT
            # floor(x/4) = step(>=4) + step(>=8) + step(>=12)
            row = 0
            for k in range(1, 4):  # thresholds at 4, 8, 12
                threshold = k * 4
                # Rise at threshold - 1
                self.W_up[row, E.TEMP] = S
                self.b_up[row] = -S * (threshold - 1)
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = 1.0 / S
                row += 1
                # Saturate at threshold
                self.W_up[row, E.TEMP] = S
                self.b_up[row] = -S * threshold
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = -1.0 / S
                row += 1


class SubNibbleMod4FFN(PureFFN):
    """
    Compute TEMP mod 4 using floor-based formula: mod = TEMP - 4*CARRY_OUT.

    CARRY_OUT already contains floor(TEMP/4) from previous layer.
    Result goes to IO_EXIT_CODE (borrow flag), gated by shift bit.
    """

    def __init__(self, shift_bit_slot: int):
        self.shift_bit_slot = shift_bit_slot
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        BORROW_FLAG = E.IO_EXIT_CODE
        with torch.no_grad():
            # Compute: BORROW_FLAG = TEMP - 4*CARRY_OUT, gated by shift bit

            # Positive term: +shift_bit * TEMP
            self.W_up[0, self.shift_bit_slot] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[BORROW_FLAG, 0] = 1.0 / S

            self.W_up[1, self.shift_bit_slot] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[BORROW_FLAG, 1] = 1.0 / S

            # Negative term: -shift_bit * 4*CARRY_OUT
            self.W_up[2, self.shift_bit_slot] = S
            self.W_gate[2, E.CARRY_OUT] = -4.0
            self.W_down[BORROW_FLAG, 2] = 1.0 / S

            self.W_up[3, self.shift_bit_slot] = -S
            self.W_gate[3, E.CARRY_OUT] = 4.0
            self.W_down[BORROW_FLAG, 3] = 1.0 / S


class UndoPosition7BorrowFFN(PureFFN):
    """
    Undo the incorrect borrow addition at position 7 AFTER attention.

    Position 7 reads from itself during attention, adding multiplier * IO_EXIT_CODE
    to CARRY_OUT. But position 7 should have no incoming borrow (no position 8).

    This layer subtracts that incorrect addition at position 7 only.
    Uses paired step: step(POS >= 7) - step(POS >= 8) = 1 only at position 7.
    Since POS max is 7, step(POS >= 8) = 0 always.
    """

    def __init__(self, multiplier: int):
        self.multiplier = multiplier
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        BORROW_FLAG = E.IO_EXIT_CODE
        with torch.no_grad():
            # Paired step function for clean 0/1 at position 7 only
            # Row 0: rise at POS = 6 (active when POS > 6)
            self.W_up[0, E.POS] = S
            self.b_up[0] = -S * 6.0  # silu(S*(POS-6)), active when POS > 6
            self.W_gate[0, BORROW_FLAG] = -float(self.multiplier)
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            # Row 1: saturate at POS = 7 (cancel excess when POS > 7, but 7 is max)
            self.W_up[1, E.POS] = S
            self.b_up[1] = -S * 7.0  # silu(S*(POS-7)), active when POS > 7
            self.W_gate[1, BORROW_FLAG] = float(self.multiplier)
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class SubNibbleCarryRightAttention(PureAttention):
    """
    Route borrow from position k+1 to position k for sub-nibble right shift.
    Position k reads IO_EXIT_CODE (borrow flag) from position k+1.
    Position 7 reads from position 0 (which has borrow=0 after ZeroPosition7BorrowFFN).

    The borrow value is multiplied by 8 (for 1-bit shift) or 4 (for 2-bit shift)
    and added to CARRY_OUT.
    """

    def __init__(self, multiplier: int = 8):
        self.multiplier = multiplier
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N - 1):
            mask[i, i + 1] = 0.0
        # Position 7 reads from itself - IO_EXIT_CODE cleared by ZeroPosition7BorrowFFN
        mask[N - 1, N - 1] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        BORROW_FLAG = E.IO_EXIT_CODE
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            # Read IO_EXIT_CODE (borrow flag) from source, multiply and add to CARRY_OUT
            self.W_v[E.CARRY_OUT, BORROW_FLAG] = float(self.multiplier)
            self.W_o[E.CARRY_OUT, E.CARRY_OUT] = 1.0


# =============================================================================
# Helper: Extract shift amount bits using step functions
# =============================================================================

class ExtractShiftBit2FFN(PureFFN):
    """
    Extract bit 2 from NIB_B (shift amount) - computes at ALL positions.

    Bit 2 = 1 when NIB_B in {4,5,6,7,12,13,14,15}.
    Uses step functions: bit2 = step(x-4) - step(x-8) + step(x-12)

    Result is stored in CARRY_OUT. Only position 0 matters (where shift amount is).
    Broadcast attention will copy position 0's value to all positions.

    Note: SoftMoE wrapper handles opcode gating.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Step function pair: step(x >= 4)
            # Row 0-1: +1 when NIB_B >= 4
            self.W_up[0, E.NIB_B] = S
            self.b_up[0] = -S * 3.0  # Active when NIB_B > 3 (i.e., >= 4)
            self.b_gate[0] = 1.0  # Constant gate
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.NIB_B] = S
            self.b_up[1] = -S * 4.0  # Saturate when NIB_B > 4
            self.b_gate[1] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S

            # Row 2-3: -1 when NIB_B >= 8
            self.W_up[2, E.NIB_B] = S
            self.b_up[2] = -S * 7.0
            self.b_gate[2] = 1.0
            self.W_down[E.CARRY_OUT, 2] = -1.0 / S

            self.W_up[3, E.NIB_B] = S
            self.b_up[3] = -S * 8.0
            self.b_gate[3] = 1.0
            self.W_down[E.CARRY_OUT, 3] = 1.0 / S

            # Row 4-5: +1 when NIB_B >= 12
            self.W_up[4, E.NIB_B] = S
            self.b_up[4] = -S * 11.0
            self.b_gate[4] = 1.0
            self.W_down[E.CARRY_OUT, 4] = 1.0 / S

            self.W_up[5, E.NIB_B] = S
            self.b_up[5] = -S * 12.0
            self.b_gate[5] = 1.0
            self.W_down[E.CARRY_OUT, 5] = -1.0 / S


class ExtractShiftBit3FFN(PureFFN):
    """
    Extract bit 3 from NIB_B (shift amount) - computes at ALL positions.

    Bit 3 = 1 when NIB_B in {8,9,10,11,12,13,14,15}.
    Uses step function: bit3 = step(x-8)

    Result is stored in CARRY_IN. Only position 0 matters.

    Note: SoftMoE wrapper handles opcode gating.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Step function pair: step(x >= 8)
            self.W_up[0, E.NIB_B] = S
            self.b_up[0] = -S * 7.0  # Active when NIB_B > 7 (i.e., >= 8)
            self.b_gate[0] = 1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.NIB_B] = S
            self.b_up[1] = -S * 8.0
            self.b_gate[1] = 1.0
            self.W_down[E.CARRY_IN, 1] = -1.0 / S


class ExtractShiftBit4FFN(PureFFN):
    """
    Extract bit 4 from shift amount (16-bit shift component).

    Bit 4 comes from NIB_B at position 1 (the low bit of that nibble).
    Bit 4 = 1 when NIB_B[1] is odd (has bit 0 set).

    Computes at ALL positions (we'll broadcast from position 1).
    Result is stored in CARRY_OUT (will be broadcast to IO_CHAR).
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        # Need 16 rows for step functions at 0, 1, ..., 15
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Detect odd values in NIB_B:
            # bit0 = step(x >= 1) - step(x >= 2) + step(x >= 3) - ...
            # Using integer-boundary step functions
            #
            # At position 1, NIB_B contains bits 4-7 of shift amount.
            # We want bit 4, which is bit 0 of NIB_B[1].

            for k in range(1, 16):
                row = k - 1
                sign = 1 if (k % 2 == 1) else -1

                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * (k - 1)  # Active when NIB_B > k-1
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = sign / S


class BroadcastShiftBit2Attention(PureAttention):
    """
    Broadcast CARRY_OUT (shift bit 2) from position 0 to all positions.
    Stores result in TEMP+1 slot (will use a different slot).

    Actually, we'll use a different approach: store all shift bits at all positions.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        # All positions read from position 0
        for k in range(N):
            mask[k, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            # Read CARRY_OUT from position 0, write to RAW_SUM at all positions
            self.W_v[E.RAW_SUM, E.CARRY_OUT] = 1.0
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class BroadcastShiftBit3Attention(PureAttention):
    """
    Broadcast CARRY_IN (shift bit 3) from position 0 to all positions.
    Stores in CARRY_IN at all positions (overwrites but that's OK).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(N):
            mask[k, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.CARRY_IN, E.CARRY_IN] = 1.0
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class BroadcastShiftBit4Attention(PureAttention):
    """
    Broadcast CARRY_OUT (shift bit 4) from position 1 to all positions.
    Stores in IO_CHAR slot at all positions.

    Position 1's NIB_B contains bits 4-7 of shift amount.
    The low bit (extracted by ExtractShiftBit4FFN) is bit 4.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        # All positions read from position 1 (not 0!)
        for k in range(N):
            mask[k, 1] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            # Read CARRY_OUT from position 1, write to IO_CHAR at all positions
            self.W_v[E.IO_CHAR, E.CARRY_OUT] = 1.0
            self.W_o[E.IO_CHAR, E.IO_CHAR] = 1.0


class BroadcastShiftBit0Attention(PureAttention):
    """Broadcast CARRY_OUT (shift bit 0) from position 0 to IO_INPUT_READY."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(N):
            mask[k, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.IO_INPUT_READY, E.CARRY_OUT] = 1.0
            self.W_o[E.IO_INPUT_READY, E.IO_INPUT_READY] = 1.0


class BroadcastShiftBit1Attention(PureAttention):
    """Broadcast CARRY_OUT (shift bit 1) from position 0 to IO_NEED_INPUT."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(N):
            mask[k, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.IO_NEED_INPUT, E.CARRY_OUT] = 1.0
            self.W_o[E.IO_NEED_INPUT, E.IO_NEED_INPUT] = 1.0


# =============================================================================
# Unified bit extraction and broadcast (replaces per-bit extract+broadcast)
# =============================================================================

class ExtractAllShiftBitsFFN(PureFFN):
    """
    Extract all 4 shift bits from NIB_B in a single layer.

    Computes at ALL positions (only pos 0 and pos 1 values matter).
    Each bit is extracted to a separate temp slot to avoid conflicts:
    - bit 0 (odd/even) → CARRY_OUT (4)
    - bit 1 (mod 4)    → RESULT (5)
    - bit 2 (mod 8)    → SHIFT_EXTRACT_A (155)
    - bit 3 (>= 8)     → SHIFT_EXTRACT_B (156)

    Note: bit 4 of shift amount = bit 0 of NIB_B at position 1.
    Since bit 0 extraction is identical regardless of position,
    the broadcast for bit 4 simply reads CARRY_OUT from pos 1.
    """

    def __init__(self):
        # bit0: 15 thresholds * 2 = 30 hidden units
        # bit1: 7 thresholds * 2 = 14 hidden units
        # bit2: 3 pairs * 2 = 6 hidden units
        # bit3: 1 pair * 2 = 2 hidden units
        # Total: 52
        super().__init__(E.DIM, hidden_dim=52)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            row = 0

            # --- Bit 0: step(>=1) - step(>=2) + step(>=3) - ... → CARRY_OUT ---
            for k in range(1, 16):
                sign = 1 if (k % 2 == 1) else -1
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * (k - 1)
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = sign / S
                row += 1
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * k
                self.b_gate[row] = 1.0
                self.W_down[E.CARRY_OUT, row] = -sign / S
                row += 1

            # --- Bit 1: step(>=2) - step(>=4) + step(>=6) - ... → RESULT ---
            thresholds_b1 = [(2, 1), (4, -1), (6, 1), (8, -1),
                             (10, 1), (12, -1), (14, 1)]
            for thresh, sign in thresholds_b1:
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * (thresh - 1)
                self.b_gate[row] = 1.0
                self.W_down[E.RESULT, row] = sign / S
                row += 1
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * thresh
                self.b_gate[row] = 1.0
                self.W_down[E.RESULT, row] = -sign / S
                row += 1

            # --- Bit 2: step(>=4) - step(>=8) + step(>=12) → SHIFT_EXTRACT_A ---
            thresholds_b2 = [(4, 1), (8, -1), (12, 1)]
            for thresh, sign in thresholds_b2:
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * (thresh - 1)
                self.b_gate[row] = 1.0
                self.W_down[E.SHIFT_EXTRACT_A, row] = sign / S
                row += 1
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * thresh
                self.b_gate[row] = 1.0
                self.W_down[E.SHIFT_EXTRACT_A, row] = -sign / S
                row += 1

            # --- Bit 3: step(>=8) → SHIFT_EXTRACT_B ---
            self.W_up[row, E.NIB_B] = S
            self.b_up[row] = -S * 7.0
            self.b_gate[row] = 1.0
            self.W_down[E.SHIFT_EXTRACT_B, row] = 1.0 / S
            row += 1
            self.W_up[row, E.NIB_B] = S
            self.b_up[row] = -S * 8.0
            self.b_gate[row] = 1.0
            self.W_down[E.SHIFT_EXTRACT_B, row] = -1.0 / S


class BroadcastSlotAttention(PureAttention):
    """
    Broadcast a value from a specific slot at a source position to a
    destination slot at all positions.

    Generic broadcast: reads src_slot from src_pos, writes to dest_slot.
    """

    def __init__(self, src_pos: int, src_slot: int, dest_slot: int):
        self.src_pos = src_pos
        self.src_slot = src_slot
        self.dest_slot = dest_slot
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(N):
            mask[k, src_pos] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[self.dest_slot, self.src_slot] = 1.0
            self.W_o[self.dest_slot, self.dest_slot] = 1.0


class ShiftCopyToTempAndCleanFFN(PureFFN):
    """
    Copy NIB_A to TEMP and clean up extraction temp slots.

    Combines ShiftCopyToTempFFN + clearing RESULT and extraction temps.
    CARRY_OUT is NOT cleared here (ClearCarryOutFFN handles it before shifts).
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        # 2 for clear TEMP + 2 for copy NIB_A + 2 for clear RESULT
        # + 2 for clear SHIFT_EXTRACT_A + 2 for clear SHIFT_EXTRACT_B = 10
        super().__init__(E.DIM, hidden_dim=10)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            row = 0

            # Clear TEMP
            self.b_up[row] = S
            self.W_gate[row, E.TEMP] = -1.0
            self.W_down[E.TEMP, row] = 1.0 / S
            row += 1
            self.b_up[row] = -S
            self.W_gate[row, E.TEMP] = 1.0
            self.W_down[E.TEMP, row] = 1.0 / S
            row += 1

            # Copy NIB_A to TEMP (gated by opcode)
            self.W_up[row, E.OP_START + self.opcode] = S
            self.W_gate[row, E.NIB_A] = 1.0
            self.W_down[E.TEMP, row] = 1.0 / S
            row += 1
            self.W_up[row, E.OP_START + self.opcode] = -S
            self.W_gate[row, E.NIB_A] = -1.0
            self.W_down[E.TEMP, row] = 1.0 / S
            row += 1

            # Clear RESULT (has bit1 extraction garbage)
            self.b_up[row] = S
            self.W_gate[row, E.RESULT] = -1.0
            self.W_down[E.RESULT, row] = 1.0 / S
            row += 1
            self.b_up[row] = -S
            self.W_gate[row, E.RESULT] = 1.0
            self.W_down[E.RESULT, row] = 1.0 / S
            row += 1

            # Clear SHIFT_EXTRACT_A
            self.b_up[row] = S
            self.W_gate[row, E.SHIFT_EXTRACT_A] = -1.0
            self.W_down[E.SHIFT_EXTRACT_A, row] = 1.0 / S
            row += 1
            self.b_up[row] = -S
            self.W_gate[row, E.SHIFT_EXTRACT_A] = 1.0
            self.W_down[E.SHIFT_EXTRACT_A, row] = 1.0 / S
            row += 1

            # Clear SHIFT_EXTRACT_B
            self.b_up[row] = S
            self.W_gate[row, E.SHIFT_EXTRACT_B] = -1.0
            self.W_down[E.SHIFT_EXTRACT_B, row] = 1.0 / S
            row += 1
            self.b_up[row] = -S
            self.W_gate[row, E.SHIFT_EXTRACT_B] = 1.0
            self.W_down[E.SHIFT_EXTRACT_B, row] = 1.0 / S


# =============================================================================
# Simpler approach: Direct routing based on shift amount
# =============================================================================

class ShiftCopyToTempFFN(PureFFN):
    """Copy NIB_A to TEMP for shift operations."""

    def __init__(self, opcode: int):
        self.opcode = opcode
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

            # Copy NIB_A to TEMP, gated by opcode
            self.W_up[2, E.OP_START + self.opcode] = S
            self.W_gate[2, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, E.OP_START + self.opcode] = -S
            self.W_gate[3, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class ConditionalShiftLeftAttention(PureAttention):
    """
    Conditionally shift TEMP left by N nibbles.

    Position i reads from position i - N.
    Positions 0..N-1 read from themselves (will be zeroed).

    The shift is applied to CARRY_OUT, then blended with TEMP based on condition.
    """

    def __init__(self, nibble_shift: int):
        self.nibble_shift = nibble_shift
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(nibble_shift, N):
            mask[i, i - nibble_shift] = 0.0
        # Invalid positions read from themselves
        for i in range(nibble_shift):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            # Read TEMP from source position, write to CARRY_OUT
            self.W_v[E.CARRY_OUT, E.TEMP] = 1.0
            self.W_o[E.CARRY_OUT, E.CARRY_OUT] = 1.0


class ConditionalShiftRightAttention(PureAttention):
    """
    Conditionally shift TEMP right by N nibbles.

    Position i reads from position i + N.
    Positions N..7 read from themselves (will be zeroed).
    """

    def __init__(self, nibble_shift: int):
        self.nibble_shift = nibble_shift
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N - nibble_shift):
            mask[i, i + nibble_shift] = 0.0
        # Invalid positions read from themselves
        for i in range(N - nibble_shift, N):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.CARRY_OUT, E.TEMP] = 1.0
            self.W_o[E.CARRY_OUT, E.CARRY_OUT] = 1.0


class ZeroInvalidAfterShiftLeftFFN(PureFFN):
    """
    Zero CARRY_OUT at positions 0..nibble_shift-1 after a left shift.

    Uses step(POS < nibble_shift) via SwiGLU to zero CARRY_OUT at invalid positions.
    Opcode gating handled by SoftMoEFFN wrapper.
    """

    def __init__(self, nibble_shift: int):
        self._nibble_shift = nibble_shift
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Unit 0: silu(S*(nibble_shift - POS)) * (-CARRY_OUT) / S
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * self._nibble_shift
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            # Unit 1: silu(S*(nibble_shift-1 - POS)) * (CARRY_OUT) / S
            self.W_up[1, E.POS] = -S
            self.b_up[1] = S * (self._nibble_shift - 1)
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class ZeroInvalidAfterShiftRightFFN(PureFFN):
    """
    Zero CARRY_OUT at positions (8-nibble_shift)..7 after a right shift.

    Uses step(POS >= threshold) via SwiGLU to zero CARRY_OUT at invalid positions.
    Opcode gating handled by SoftMoEFFN wrapper.
    """

    def __init__(self, nibble_shift: int):
        self._nibble_shift = nibble_shift
        self._threshold = E.NUM_POSITIONS - nibble_shift
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        threshold = self._threshold
        with torch.no_grad():
            # step(POS >= threshold) ≈ [silu(S*(POS-threshold+1)) - silu(S*(POS-threshold))]/S
            # Unit 0: silu(S*(POS - threshold + 1)) * (-CARRY_OUT) / S
            self.W_up[0, E.POS] = S
            self.b_up[0] = -S * (threshold - 1)
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            # Unit 1: silu(S*(POS - threshold)) * (CARRY_OUT) / S
            self.W_up[1, E.POS] = S
            self.b_up[1] = -S * threshold
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class BlendShiftResultFFN(PureFFN):
    """
    Blend shifted result (CARRY_OUT) with original (TEMP) based on shift bit.

    If shift_bit == 1: TEMP = CARRY_OUT (use shifted value)
    If shift_bit == 0: TEMP = TEMP (keep original)

    shift_bit is stored in RAW_SUM (broadcast from position 0).

    Implementation:
    - Clear TEMP
    - Add CARRY_OUT * shift_bit (shifted value when bit=1)
    - Add TEMP_backup * (1 - shift_bit) (original when bit=0)

    Actually, simpler:
    - TEMP_new = TEMP + shift_bit * (CARRY_OUT - TEMP)
    - When shift_bit=1: TEMP_new = CARRY_OUT
    - When shift_bit=0: TEMP_new = TEMP
    """

    def __init__(self, opcode: int, shift_bit_slot: int):
        self.opcode = opcode
        self.shift_bit_slot = shift_bit_slot
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # TEMP += shift_bit * CARRY_OUT
            self.W_up[0, self.shift_bit_slot] = S
            self.W_gate[0, E.CARRY_OUT] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, self.shift_bit_slot] = -S
            self.W_gate[1, E.CARRY_OUT] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # TEMP -= shift_bit * TEMP
            # This is trickier because we're reading and writing TEMP
            # We need to do this in a separate layer


class ApplyConditionalShiftFFN(PureFFN):
    """
    Apply conditional shift: replace TEMP with CARRY_OUT when shift bit is set.

    If shift_bit_slot > 0.5: TEMP = CARRY_OUT
    Else: TEMP unchanged

    Implementation using identity:
    - TEMP_new = TEMP + bit * (CARRY_OUT - TEMP)
    - When bit=1: TEMP_new = TEMP + (CARRY_OUT - TEMP) = CARRY_OUT
    - When bit=0: TEMP_new = TEMP
    """

    def __init__(self, opcode: int, shift_bit_slot: int):
        self.opcode = opcode
        self.shift_bit_slot = shift_bit_slot
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Compute: TEMP += bit * CARRY_OUT - bit * TEMP
            # = TEMP + bit * (CARRY_OUT - TEMP)

            # Row 0-1: +bit * CARRY_OUT (identity using silu)
            # For bit=1: silu(S*1) * CARRY_OUT / S ≈ CARRY_OUT
            self.W_up[0, self.shift_bit_slot] = S
            self.W_gate[0, E.CARRY_OUT] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, self.shift_bit_slot] = -S
            self.W_gate[1, E.CARRY_OUT] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Row 2-3: -bit * TEMP
            self.W_up[2, self.shift_bit_slot] = S
            self.W_gate[2, E.TEMP] = -1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, self.shift_bit_slot] = -S
            self.W_gate[3, E.TEMP] = 1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class ClearCarryOutFFN(PureFFN):
    """Clear CARRY_OUT before shift attention to prevent accumulation."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Unconditionally clear CARRY_OUT
            self.b_up[0] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class ClearIOExitCodeFFN(PureFFN):
    """Clear IO_EXIT_CODE before sub-nibble shift to prevent carry accumulation."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Unconditionally clear IO_EXIT_CODE
            self.b_up[0] = S
            self.W_gate[0, E.IO_EXIT_CODE] = -1.0
            self.W_down[E.IO_EXIT_CODE, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, E.IO_EXIT_CODE] = 1.0
            self.W_down[E.IO_EXIT_CODE, 1] = 1.0 / S


class CopyTempToResultFFN(PureFFN):
    """Copy final TEMP value to RESULT."""

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


class ClearAfterShiftFFN(PureFFN):
    """Clear all shift-related slots after shift operation."""

    SLOTS = [
        E.TEMP, E.CARRY_OUT, E.CARRY_IN, E.RAW_SUM,
        E.IO_CHAR, E.IO_INPUT_READY, E.IO_NEED_INPUT,
        E.IO_EXIT_CODE,
    ]

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=len(self.SLOTS) * 2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            for i, slot in enumerate(self.SLOTS):
                row = i * 2
                self.W_up[row, E.OP_START + self.opcode] = S
                self.W_gate[row, slot] = -1.0
                self.W_down[slot, row] = 1.0 / S

                self.W_up[row + 1, E.OP_START + self.opcode] = -S
                self.W_gate[row + 1, slot] = 1.0
                self.W_down[slot, row + 1] = 1.0 / S


# =============================================================================
# Build variable shift layers
# =============================================================================

def build_variable_shift_left_layers():
    """
    Build layers for variable shift left.

    Uses unified bit extraction (all 5 bits in 3 layers) then binary
    decomposition for nibble-aligned and sub-nibble shifts.

    Extraction slots:
      bit 0 → CARRY_OUT(4), broadcast → IO_INPUT_READY(89)
      bit 1 → RESULT(5), broadcast → IO_NEED_INPUT(90)
      bit 2 → SHIFT_EXTRACT_A(155), broadcast → RAW_SUM(2)
      bit 3 → SHIFT_EXTRACT_B(156), broadcast → CARRY_IN(3)
      bit 4 = bit 0 of NIB_B at pos 1, broadcast CARRY_OUT from pos 1 → IO_CHAR(80)

    Returns list of layers.
    """
    from .pure_moe import MoE

    layers = []

    # 1. Extract all shift bits in one layer
    layers.append(MoE([ExtractAllShiftBitsFFN()], [Opcode.SHL]))

    # 2. Broadcast all 5 bits simultaneously (5 attention experts, same opcode)
    broadcast_experts = [
        BroadcastSlotAttention(0, E.CARRY_OUT, E.IO_INPUT_READY),        # bit 0
        BroadcastSlotAttention(0, E.RESULT, E.IO_NEED_INPUT),            # bit 1
        BroadcastSlotAttention(0, E.SHIFT_EXTRACT_A, E.RAW_SUM),         # bit 2
        BroadcastSlotAttention(0, E.SHIFT_EXTRACT_B, E.CARRY_IN),        # bit 3
        BroadcastSlotAttention(1, E.CARRY_OUT, E.IO_CHAR),               # bit 4
    ]
    layers.append(MoE(broadcast_experts, [Opcode.SHL] * 5))

    # 3. Copy NIB_A → TEMP + clean up extraction temp slots
    layers.append(MoE([ShiftCopyToTempAndCleanFFN(Opcode.SHL)], [Opcode.SHL]))

    # 4. Conditional 4-nibble shift (16 bits) - if bit 4 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHL]))
    layers.append(MoE([ConditionalShiftLeftAttention(4)], [Opcode.SHL]))
    layers.append(MoE([ZeroInvalidAfterShiftLeftFFN(4)], [Opcode.SHL]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHL, E.IO_CHAR)], [Opcode.SHL]))

    # 5. Conditional 2-nibble shift (8 bits) - if bit 3 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHL]))
    layers.append(MoE([ConditionalShiftLeftAttention(2)], [Opcode.SHL]))
    layers.append(MoE([ZeroInvalidAfterShiftLeftFFN(2)], [Opcode.SHL]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHL, E.CARRY_IN)], [Opcode.SHL]))

    # 6. Conditional 1-nibble shift (4 bits) - if bit 2 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHL]))
    layers.append(MoE([ConditionalShiftLeftAttention(1)], [Opcode.SHL]))
    layers.append(MoE([ZeroInvalidAfterShiftLeftFFN(1)], [Opcode.SHL]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHL, E.RAW_SUM)], [Opcode.SHL]))

    # 7. Conditional 2-bit shift - if bit 1 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHL]))
    layers.append(MoE([ClearIOExitCodeFFN()], [Opcode.SHL]))
    layers.append(MoE([SubNibbleShiftLeft2FFN(Opcode.SHL)], [Opcode.SHL]))
    layers.append(MoE([SubNibbleCarryLeftAttention()], [Opcode.SHL]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHL, E.IO_NEED_INPUT)], [Opcode.SHL]))

    # 8. Conditional 1-bit shift - if bit 0 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHL]))
    layers.append(MoE([ClearIOExitCodeFFN()], [Opcode.SHL]))
    layers.append(MoE([SubNibbleShiftLeft1FFN(Opcode.SHL)], [Opcode.SHL]))
    layers.append(MoE([SubNibbleCarryLeftAttention()], [Opcode.SHL]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHL, E.IO_INPUT_READY)], [Opcode.SHL]))

    # 9. Copy TEMP to RESULT
    layers.append(MoE([CopyTempToResultFFN(Opcode.SHL)], [Opcode.SHL]))

    # 10. Cleanup
    layers.append(MoE([ClearAfterShiftFFN(Opcode.SHL)], [Opcode.SHL]))

    return layers


def build_variable_shift_right_layers():
    """
    Build layers for variable shift right.

    Uses same unified bit extraction as SHL, then binary decomposition
    for nibble-aligned and sub-nibble shifts (right direction).
    """
    from .pure_moe import MoE

    layers = []

    # 1. Extract all shift bits in one layer
    layers.append(MoE([ExtractAllShiftBitsFFN()], [Opcode.SHR]))

    # 2. Broadcast all 5 bits simultaneously
    broadcast_experts = [
        BroadcastSlotAttention(0, E.CARRY_OUT, E.IO_INPUT_READY),        # bit 0
        BroadcastSlotAttention(0, E.RESULT, E.IO_NEED_INPUT),            # bit 1
        BroadcastSlotAttention(0, E.SHIFT_EXTRACT_A, E.RAW_SUM),         # bit 2
        BroadcastSlotAttention(0, E.SHIFT_EXTRACT_B, E.CARRY_IN),        # bit 3
        BroadcastSlotAttention(1, E.CARRY_OUT, E.IO_CHAR),               # bit 4
    ]
    layers.append(MoE(broadcast_experts, [Opcode.SHR] * 5))

    # 3. Copy NIB_A → TEMP + clean up extraction temp slots
    layers.append(MoE([ShiftCopyToTempAndCleanFFN(Opcode.SHR)], [Opcode.SHR]))

    # 4. Conditional 4-nibble shift (16 bits) - if bit 4 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHR]))
    layers.append(MoE([ConditionalShiftRightAttention(4)], [Opcode.SHR]))
    layers.append(MoE([ZeroInvalidAfterShiftRightFFN(4)], [Opcode.SHR]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHR, E.IO_CHAR)], [Opcode.SHR]))

    # 5. Conditional 2-nibble shift (8 bits) - if bit 3 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHR]))
    layers.append(MoE([ConditionalShiftRightAttention(2)], [Opcode.SHR]))
    layers.append(MoE([ZeroInvalidAfterShiftRightFFN(2)], [Opcode.SHR]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHR, E.CARRY_IN)], [Opcode.SHR]))

    # 6. Conditional 1-nibble shift (4 bits) - if bit 2 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHR]))
    layers.append(MoE([ConditionalShiftRightAttention(1)], [Opcode.SHR]))
    layers.append(MoE([ZeroInvalidAfterShiftRightFFN(1)], [Opcode.SHR]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHR, E.RAW_SUM)], [Opcode.SHR]))

    # 7. Conditional 2-bit shift - if bit 1 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHR]))
    layers.append(MoE([ClearIOExitCodeFFN()], [Opcode.SHR]))
    layers.append(MoE([SubNibbleShiftRight2FFN(Opcode.SHR)], [Opcode.SHR]))
    layers.append(MoE([SubNibbleMod4FFN(E.IO_NEED_INPUT)], [Opcode.SHR]))
    layers.append(MoE([SubNibbleCarryRightAttention(multiplier=4)], [Opcode.SHR]))
    layers.append(MoE([UndoPosition7BorrowFFN(4)], [Opcode.SHR]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHR, E.IO_NEED_INPUT)], [Opcode.SHR]))

    # 8. Conditional 1-bit shift - if bit 0 is set
    layers.append(MoE([ClearCarryOutFFN()], [Opcode.SHR]))
    layers.append(MoE([ClearIOExitCodeFFN()], [Opcode.SHR]))
    layers.append(MoE([SubNibbleShiftRight1FFN(Opcode.SHR)], [Opcode.SHR]))
    layers.append(MoE([SubNibbleMod2FFN(E.IO_INPUT_READY)], [Opcode.SHR]))
    layers.append(MoE([SubNibbleCarryRightAttention(multiplier=8)], [Opcode.SHR]))
    layers.append(MoE([UndoPosition7BorrowFFN(8)], [Opcode.SHR]))
    layers.append(MoE([ApplyConditionalShiftFFN(Opcode.SHR, E.IO_INPUT_READY)], [Opcode.SHR]))

    # 9. Copy TEMP to RESULT
    layers.append(MoE([CopyTempToResultFFN(Opcode.SHR)], [Opcode.SHR]))

    # 10. Cleanup
    layers.append(MoE([ClearAfterShiftFFN(Opcode.SHR)], [Opcode.SHR]))

    return layers


# =============================================================================
# Test function
# =============================================================================

def test_variable_shifts():
    """Test variable shift operations."""
    import torch.nn as nn

    shl_layers = build_variable_shift_left_layers()
    shr_layers = build_variable_shift_right_layers()

    shl_model = nn.Sequential(*shl_layers)
    shr_model = nn.Sequential(*shr_layers)

    def make_input(a: int, b: int, opcode: int) -> torch.Tensor:
        N = E.NUM_POSITIONS
        a_nibbles = [(a >> (4*i)) & 0xF for i in range(N)]
        b_nibbles = [(b >> (4*i)) & 0xF for i in range(N)]

        x = torch.zeros(1, N, E.DIM)
        x[:, :, 0] = 1.0  # Bias
        x[:, :, E.POS] = torch.arange(N).float()
        for i in range(N):
            x[:, i, E.NIB_A] = a_nibbles[i]
            x[:, i, E.NIB_B] = b_nibbles[i]
        x[:, :, E.OP_START + opcode] = 1.0
        return x

    def decode_result(x: torch.Tensor) -> int:
        N = E.NUM_POSITIONS
        result = 0
        for i in range(N):
            nib = int(round(x[0, i, E.RESULT].item()))
            nib = max(0, min(15, nib))
            result |= nib << (4 * i)
        return result

    print("\n=== Testing Variable SHL ===")
    shl_tests = [
        (1, 4, 0x10),           # 1 << 4 = 16
        (0xFF, 8, 0xFF00),      # 0xFF << 8
        (0x12345678, 4, 0x23456780),  # Large value << 4
    ]

    shl_passed = 0
    for a, b, expected in shl_tests:
        x = make_input(a, b, Opcode.SHL)
        x = shl_model(x)
        result = decode_result(x) & 0xFFFFFFFF
        if result == expected:
            print(f"  0x{a:08X} << {b} = 0x{result:08X} ✓")
            shl_passed += 1
        else:
            print(f"  0x{a:08X} << {b} = 0x{result:08X} (expected 0x{expected:08X}) ✗")

    print(f"\nSHL: {shl_passed}/{len(shl_tests)} passed")

    print("\n=== Testing Variable SHR ===")
    shr_tests = [
        (16, 4, 1),             # 16 >> 4 = 1
        (0xFF00, 8, 0xFF),      # 0xFF00 >> 8 = 0xFF
        (0x12345678, 4, 0x01234567),  # Large value >> 4
    ]

    shr_passed = 0
    for a, b, expected in shr_tests:
        x = make_input(a, b, Opcode.SHR)
        x = shr_model(x)
        result = decode_result(x)
        if result == expected:
            print(f"  0x{a:08X} >> {b} = 0x{result:08X} ✓")
            shr_passed += 1
        else:
            print(f"  0x{a:08X} >> {b} = 0x{result:08X} (expected 0x{expected:08X}) ✗")

    print(f"\nSHR: {shr_passed}/{len(shr_tests)} passed")

    return shl_passed == len(shl_tests) and shr_passed == len(shr_tests)


if __name__ == "__main__":
    test_variable_shifts()
