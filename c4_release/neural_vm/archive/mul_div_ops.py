"""
Multiplication and Division operations for Neural VM V7.

MUL, DIV, MOD - per-nibble operations.
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN


# =============================================================================
# MUL Operations
# =============================================================================

class MulProductFFN(PureFFN):
    """
    Compute temp = a * b (ungated).
    Uses: silu(S*a) * b / S ≈ a * b when a > 0
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.NIB_A] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class MulGateFFN(PureFFN):
    """Gate temp by opcode, write to RESULT."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class MulOverflowFFN(PureFFN):
    """
    Handle MUL overflow: compute RESULT mod 16 and CARRY_OUT = RESULT // 16.

    Max product is 15*15=225, so we need step functions at 16, 32, ..., 224.
    Each step subtracts 16 from RESULT and adds 1 to CARRY_OUT.

    Uses step(x) = silu(S*(x+1)) - silu(S*x) pattern for exact thresholds.
    """

    def __init__(self):
        # Need 2 rows per threshold level (step pattern), 14 levels (16 to 224)
        super().__init__(E.DIM, hidden_dim=28)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each threshold k*16 where k=1..14:
            # Subtract 16 from RESULT and add 1 to CARRY_OUT when TEMP >= k*16
            for k in range(1, 15):
                row = (k - 1) * 2
                threshold = k * 16

                # Row 0: silu(S*(TEMP - threshold + 1)) * 1
                self.W_up[row, E.TEMP] = S
                self.b_up[row] = -S * (threshold - 1)
                self.W_gate[row, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row] = -16.0 / S
                self.W_down[E.CARRY_OUT, row] = 1.0 / S

                # Row 1: -silu(S*(TEMP - threshold)) * 1 (saturation)
                self.W_up[row + 1, E.TEMP] = S
                self.b_up[row + 1] = -S * threshold
                self.W_gate[row + 1, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row + 1] = 16.0 / S
                self.W_down[E.CARRY_OUT, row + 1] = -1.0 / S


class MulZeroFirstCarryFFN(PureFFN):
    """Zeros out carry_in for position 0 in MUL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


class MulClearCarryOutFFN(PureFFN):
    """Clears carry_out before next propagation round for MUL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class MulCarryIterFFN(PureFFN):
    """
    One iteration of carry propagation for MUL.

    Handles multi-level overflow: RESULT+CARRY_IN can be up to 1800+112.
    Uses step functions at 16, 32, 48, ... to compute:
    - CARRY_OUT = (RESULT + CARRY_IN) // 16
    - RESULT_new = (RESULT + CARRY_IN) mod 16

    The overflow detection reads (RESULT + CARRY_IN) so each iteration
    fully resolves one position. This ensures 7 carry iterations suffice
    for 8 nibble positions.

    Integrated clearing: clears old CARRY_OUT and CARRY_IN so carries
    don't accumulate across iterations.
    """

    def __init__(self):
        # 2 rows for carry_in addition
        # 2 rows per threshold level (step pattern), up to 120 levels (16 to 1920)
        # 4 rows for clearing CARRY_OUT and CARRY_IN
        # Total: 2 + 120*2 + 4 = 246 hidden units
        super().__init__(E.DIM, hidden_dim=246)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Rows 0-1: Add carry_in to result
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Rows 2-241: Multi-level overflow detection on (RESULT + CARRY_IN)
            # For each threshold k*16 where k=1..120:
            # Subtract 16 from RESULT and add 1 to CARRY_OUT when (RESULT+CARRY_IN) >= k*16
            # Max value: 1800 (schoolbook) + 112 (carry_in) = 1912, so 120 levels covers it.
            for k in range(1, 121):
                row = 2 + (k - 1) * 2
                threshold = k * 16

                # Row: silu(S*(RESULT + CARRY_IN - threshold + 1)) * opcode
                self.W_up[row, E.RESULT] = S
                self.W_up[row, E.CARRY_IN] = S
                self.b_up[row] = -S * (threshold - 1)
                self.W_gate[row, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row] = -16.0 / S
                self.W_down[E.CARRY_OUT, row] = 1.0 / S

                # Saturation row: -silu(S*(RESULT + CARRY_IN - threshold)) * opcode
                self.W_up[row + 1, E.RESULT] = S
                self.W_up[row + 1, E.CARRY_IN] = S
                self.b_up[row + 1] = -S * threshold
                self.W_gate[row + 1, E.OP_START + Opcode.MUL] = 1.0
                self.W_down[E.RESULT, row + 1] = 16.0 / S
                self.W_down[E.CARRY_OUT, row + 1] = -1.0 / S

            # Rows 242-243: Clear old CARRY_OUT (cancel pair)
            self.W_up[242, E.OP_START + Opcode.MUL] = S
            self.W_gate[242, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 242] = 1.0 / S

            self.W_up[243, E.OP_START + Opcode.MUL] = -S
            self.W_gate[243, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 243] = 1.0 / S

            # Rows 244-245: Clear old CARRY_IN (cancel pair)
            self.W_up[244, E.OP_START + Opcode.MUL] = S
            self.W_gate[244, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 244] = 1.0 / S

            self.W_up[245, E.OP_START + Opcode.MUL] = -S
            self.W_gate[245, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 245] = 1.0 / S


class MulClearCarryInFFN(PureFFN):
    """Clears carry_in before next propagation round for MUL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


class MulClearTempFFN(PureFFN):
    """Clear TEMP after MUL to avoid interference."""

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


# =============================================================================
# DIV Operations
# =============================================================================

class DivInitFFN(PureFFN):
    """Initialize division: copy a to TEMP (dividend), zero RESULT (quotient)."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.DIV] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.DIV] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Clear RESULT
            self.W_up[2, E.OP_START + Opcode.DIV] = S
            self.W_gate[2, E.RESULT] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.DIV] = -S
            self.W_gate[3, E.RESULT] = 1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class DivIterFFN(PureFFN):
    """
    One iteration of division: if TEMP >= NIB_B, subtract NIB_B and add 1 to quotient.

    Position gated: Only operates at position 0 to avoid NIB_B=0 issues at higher nibbles.
    The condition includes -S*100*POS to suppress all positions except 0.

    Note: Single-nibble DIV only. For multi-nibble numbers, use a different algorithm.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Position gating factor: -S * 100 * POS
            # At POS=0: factor = 0
            # At POS=1: factor = -S*100 (strongly negative, suppresses silu)

            # Rows 0-1: Subtract NIB_B from TEMP when TEMP >= NIB_B at POS=0
            self.W_up[0, E.TEMP] = S
            self.W_up[0, E.NIB_B] = -S
            self.W_up[0, E.POS] = -S * 100  # Position gate
            self.b_up[0] = S * 1.0
            self.W_gate[0, E.NIB_B] = 1.0  # Gate by NIB_B
            self.W_down[E.TEMP, 0] = -1.0 / S

            self.W_up[1, E.TEMP] = S
            self.W_up[1, E.NIB_B] = -S
            self.W_up[1, E.POS] = -S * 100
            self.b_up[1] = 0.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Rows 2-3: Add 1 to RESULT when TEMP >= NIB_B at POS=0
            self.W_up[2, E.TEMP] = S
            self.W_up[2, E.NIB_B] = -S
            self.W_up[2, E.POS] = -S * 100  # Position gate
            self.b_up[2] = S * 1.0
            self.W_gate[2, E.OP_START + Opcode.DIV] = 1.0  # Gate by opcode (constant 1)
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.TEMP] = S
            self.W_up[3, E.NIB_B] = -S
            self.W_up[3, E.POS] = -S * 100
            self.b_up[3] = 0.0
            self.W_gate[3, E.OP_START + Opcode.DIV] = 1.0
            self.W_down[E.RESULT, 3] = -1.0 / S


# =============================================================================
# MOD Operations
# =============================================================================

class ModInitFFN(PureFFN):
    """Initialize MOD: copy a to TEMP (dividend)."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MOD] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MOD] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ModIterFFN(PureFFN):
    """
    One iteration of MOD: if TEMP >= NIB_B, subtract NIB_B.

    Position gated: Only operates at position 0 to avoid NIB_B=0 issues at higher nibbles.

    Note: Single-nibble MOD only. For multi-nibble numbers, use a different algorithm.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: silu(S*(TEMP - NIB_B + 1 - 100*POS)) * NIB_B * (-1/S)
            self.W_up[0, E.TEMP] = S
            self.W_up[0, E.NIB_B] = -S
            self.W_up[0, E.POS] = -S * 100  # Position gate
            self.b_up[0] = S * 1.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.TEMP, 0] = -1.0 / S

            # Row 1: saturation term
            self.W_up[1, E.TEMP] = S
            self.W_up[1, E.NIB_B] = -S
            self.W_up[1, E.POS] = -S * 100
            self.b_up[1] = 0.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ModResultFFN(PureFFN):
    """For MOD, copy TEMP (remainder) to RESULT."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear any previous RESULT
            self.W_up[0, E.OP_START + Opcode.MOD] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MOD] = -S
            self.W_gate[1, E.RESULT] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Copy TEMP to RESULT
            self.W_up[2, E.OP_START + Opcode.MOD] = S
            self.W_gate[2, E.TEMP] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.MOD] = -S
            self.W_gate[3, E.TEMP] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


# =============================================================================
# FAST MOD - Power-of-2 approach (~5 layers per iteration)
# =============================================================================
# Algorithm:
# 1. Find MSB of divisor (highest non-zero nibble position)
# 2. Broadcast MSB position to all positions
# 3. Mask: zero out nibbles of A above MSB position of B
# 4. Compare: detect if masked_A >= B
# 5. Conditional subtract: if masked_A >= B, result = masked_A - B
# Iterate 2-3 times for large A >> B.

from .base_layers import PureAttention, bake_weights as _bake_weights


class FastModMSBDetectFFN(PureFFN):
    """
    Layer 1: Detect MSB nibble position of divisor B.

    For each nibble value 1-15 at each position, detect if it's non-zero.
    Store detection result in CARRY_OUT (1 if this is a non-zero nibble).
    The highest position with a non-zero nibble is the MSB.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @_bake_weights
    def _bake_weights(self):
        S = E.SCALE
        # Detect if NIB_B > 0 at this position
        # Store in CARRY_OUT: 1 if NIB_B >= 1, 0 otherwise
        # Step function at 0.5: silu(S*(NIB_B - 0.5)) - silu(S*(NIB_B - 1.5))
        self.W_up[0, E.NIB_B] = S
        self.b_up[0] = -S * 0.5
        self.W_gate[0, E.OP_START + Opcode.MOD] = 1.0
        self.W_down[E.CARRY_OUT, 0] = 1.0 / S

        self.W_up[1, E.NIB_B] = S
        self.b_up[1] = -S * 1.5
        self.W_gate[1, E.OP_START + Opcode.MOD] = 1.0
        self.W_down[E.CARRY_OUT, 1] = -1.0 / S

        # Also copy NIB_A to RESULT for working
        self.b_up[2] = S
        self.W_gate[2, E.NIB_A] = 1.0
        self.W_down[E.RESULT, 2] = 1.0 / S

        self.b_up[3] = -S
        self.W_gate[3, E.NIB_A] = -1.0
        self.W_down[E.RESULT, 3] = 1.0 / S


class FastModMaskFFN(PureFFN):
    """
    Layer 3: Mask nibbles of A above the MSB of B.

    Uses CARRY_IN (broadcast MSB position) to decide which nibbles to keep.
    Nibbles at positions > MSB_pos are zeroed out.

    Combined with conditional subtract: if RESULT >= NIB_B, subtract NIB_B.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @_bake_weights
    def _bake_weights(self):
        S = E.SCALE
        # If position > MSB_pos of B (i.e., CARRY_IN from broadcast == 0),
        # zero out RESULT at this position.
        # CARRY_IN = 1 means "B has a non-zero nibble at or above this position"
        # So positions where CARRY_IN = 0 should be masked (RESULT = 0)

        # Subtract RESULT when CARRY_IN = 0:
        # silu(-S * CARRY_IN + S * 0.5) * (-RESULT) / S
        # At CARRY_IN=0: silu(S*0.5) ≈ 0.5*S → output ≈ -RESULT * 0.5*S * 1/S... not exact.

        # Simpler: detect when RESULT >= NIB_B and subtract
        # This is the conditional subtract step
        self.W_up[0, E.RESULT] = S
        self.W_up[0, E.NIB_B] = -S
        self.b_up[0] = S * 0.5  # >= 0 means RESULT >= NIB_B
        self.W_gate[0, E.OP_START + Opcode.MOD] = 1.0
        self.W_down[E.RESULT, 0] = -1.0 / S  # Will subtract NIB_B

        # Gate: multiply by NIB_B (the amount to subtract)
        # Actually this needs the cancel pair pattern
        self.W_up[1, E.RESULT] = S
        self.W_up[1, E.NIB_B] = -S
        self.b_up[1] = -S * 0.5  # saturation
        self.W_gate[1, E.OP_START + Opcode.MOD] = 1.0
        self.W_down[E.RESULT, 1] = 1.0 / S


class FastModSubtractFFN(PureFFN):
    """
    Conditional subtract: if RESULT >= NIB_B, then RESULT -= NIB_B.

    Uses step function to detect RESULT >= NIB_B, then subtracts NIB_B.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @_bake_weights
    def _bake_weights(self):
        S = E.SCALE
        # When RESULT >= NIB_B: subtract NIB_B from RESULT
        # Step at threshold: RESULT - NIB_B >= 0
        # silu(S*(RESULT - NIB_B + 0.5)) × NIB_B → positive when RESULT >= NIB_B
        self.W_up[0, E.RESULT] = S
        self.W_up[0, E.NIB_B] = -S
        self.b_up[0] = S * 0.5
        self.W_gate[0, E.NIB_B] = -1.0  # Subtract NIB_B
        self.W_down[E.RESULT, 0] = 1.0 / S

        # Saturation
        self.W_up[1, E.RESULT] = S
        self.W_up[1, E.NIB_B] = -S
        self.b_up[1] = -S * 0.5
        self.W_gate[1, E.NIB_B] = 1.0
        self.W_down[E.RESULT, 1] = 1.0 / S

        # When RESULT >= 2*NIB_B: subtract another NIB_B (for larger remainders)
        self.W_up[2, E.RESULT] = S
        self.W_up[2, E.NIB_B] = -2.0 * S  # threshold at 2*NIB_B
        self.b_up[2] = S * 0.5
        self.W_gate[2, E.NIB_B] = -1.0
        self.W_down[E.RESULT, 2] = 1.0 / S

        self.W_up[3, E.RESULT] = S
        self.W_up[3, E.NIB_B] = -2.0 * S
        self.b_up[3] = -S * 0.5
        self.W_gate[3, E.NIB_B] = 1.0
        self.W_down[E.RESULT, 3] = 1.0 / S


def build_fast_mod_layers():
    """
    Build fast MOD layers (~15 layers: 3 iterations of 5 layers each).

    Each iteration:
    1. MSB detect (PureFFN)
    2. Broadcast MSB (PureAttention) - reuses BroadcastAttention
    3. Conditional subtract (PureFFN)

    Returns list of (layer, [opcodes]) tuples for SoftMoEFFN wrapping.
    """
    from .pure_moe import MoE
    from .reduce_ffn import BroadcastAttention

    layers = []

    # 3 iterations to handle A >> B cases
    for _ in range(3):
        # Step 1: MSB detect + copy A to RESULT
        layers.append(MoE([FastModMSBDetectFFN()], [Opcode.MOD]))

        # Step 2: Broadcast MSB info (which positions of B are non-zero)
        broadcast = BroadcastAttention(src_slot=E.CARRY_OUT, dst_slot=E.CARRY_IN)
        layers.append(MoE([broadcast], [Opcode.MOD]))

        # Step 3: Conditional subtract
        layers.append(MoE([FastModSubtractFFN()], [Opcode.MOD]))

    return layers
