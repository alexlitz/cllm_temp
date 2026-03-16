"""
Fast multiplication using FlattenedPureFFN for parallel schoolbook + carry.

MUL: 42 → 7 layers

Layer 1: SchoolbookFlatFFN - all 36 partial products in one flat layer
Layer 2: MulCarryPass1FFN - mod 16 + carry extraction (up to 1800)
Layer 3: MulCarryPass2FFN - add carry from pass 1, new mod/carry (up to 127)
Layer 4: MulCarryPass3FFN - add carry from pass 2, mod/carry (up to 22), binary
Layer 5: MulGenPropFFN - compute G/P for binary carry, add carry, mod16
Layer 6: MulBinaryLookaheadFFN - carry-lookahead on G/P, clear G/P
Layer 7: MulFinalCorrectionFFN - add lookahead carry, mod16, clear
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN, FlattenedPureFFN


class SchoolbookFlatFFN(FlattenedPureFFN):
    """Layer 1: All 36 partial products in one FlattenedPureFFN layer.

    Schoolbook: result[k] = sum_{i+j=k, 0<=i,j<8} a[i] * b[j]
    Total valid products: 1+2+3+4+5+6+7+8 = 36
    Each product uses 2 hidden units (cancel pair).
    Also 2 units per position to clear RESULT first (16 units).

    Total: 36*2 + 8*2 = 88 hidden units.

    Args:
        opcode: Opcode to gate on (default MUL).
        source_a: Slot to read first operand from (default NIB_A).
                  For MOD: E.RESULT (reads quotient from division phase).
    """

    def __init__(self, opcode=Opcode.MUL, source_a=E.NIB_A):
        self.opcode = opcode
        self.source_a = source_a
        super().__init__(hidden_dim=88)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0

            # First: clear RESULT at all 8 positions (cancel pair per position)
            # Gate from opcode at position 0 so clearing works at all positions
            for pos in range(8):
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = S
                self.W_gate.data[h, fi(pos, E.RESULT)] = -1.0
                self.W_down.data[fi(pos, E.RESULT), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = -S
                self.W_gate.data[h, fi(pos, E.RESULT)] = 1.0
                self.W_down.data[fi(pos, E.RESULT), h] = 1.0 / S
                h += 1

            # Partial products: for each output position k, sum a[i]*b[k-i]
            for k in range(8):
                for i in range(k + 1):
                    j = k - i
                    if i < 8 and j < 8:
                        # Product a[i] * b[j] → RESULT[k]
                        self.W_up.data[h, fi(i, self.source_a)] = S
                        self.W_gate.data[h, fi(j, E.NIB_B)] = 1.0
                        self.W_down.data[fi(k, E.RESULT), h] = 1.0 / S
                        h += 1

                        self.W_up.data[h, fi(i, self.source_a)] = -S
                        self.W_gate.data[h, fi(j, E.NIB_B)] = -1.0
                        self.W_down.data[fi(k, E.RESULT), h] = 1.0 / S
                        h += 1

            assert h <= 88, f"Used {h} hidden units, expected <= 88"


class MulCarryPass1FFN(FlattenedPureFFN):
    """Layer 2: Extract carry from RESULT and apply mod 16.

    After schoolbook, RESULT[i] can be up to ~1800.
    Compute floor(RESULT[i] / 16) → CARRY_OUT[i], and RESULT[i] mod 16.
    Max steps: floor(1800/16) = 112. So 112 step pairs per position.

    Total: 112 * 2 * 8 = 1792 hidden units.
    """

    def __init__(self, opcode=Opcode.MUL):
        self.opcode = opcode
        super().__init__(hidden_dim=1792)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0
            op_gate = fi(0, E.OP_START + self.opcode)

            for pos in range(8):
                result_idx = fi(pos, E.RESULT)
                carry_idx = fi(pos, E.CARRY_OUT)

                for k in range(1, 113):
                    threshold = k * 16
                    # Rise
                    self.W_up.data[h, result_idx] = S
                    self.b_up.data[h] = -S * (threshold - 1)
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = -16.0 / S
                    self.W_down.data[carry_idx, h] = 1.0 / S
                    h += 1

                    # Saturation
                    self.W_up.data[h, result_idx] = S
                    self.b_up.data[h] = -S * threshold
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = 16.0 / S
                    self.W_down.data[carry_idx, h] = -1.0 / S
                    h += 1

            assert h == 1792


class MulCarryPass2FFN(FlattenedPureFFN):
    """Layer 3: Add carry from pass 1, compute new mod/carry.

    RESULT[i] is now 0..15 from pass 1.
    CARRY_OUT[i-1] is the carry from position i-1 (0..112).
    New value: RESULT[i] + CARRY_OUT[i-1], range 0..127.

    Step functions must read BOTH RESULT[i] AND CARRY_OUT[i-1] from input
    to correctly threshold on the combined value.

    floor(127/16) = 7, so 7 step pairs per position.
    Also add carry (cancel pair) and clear old CARRY_OUT.

    Total: ~200 hidden units.
    """

    def __init__(self, opcode=Opcode.MUL):
        self.opcode = opcode
        super().__init__(hidden_dim=200)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0
            op_gate = fi(0, E.OP_START + self.opcode)

            # Step 1: Add CARRY_OUT[i-1] to RESULT[i] for i=1..7
            # And clear CARRY_OUT at all positions
            for pos in range(8):
                if pos > 0:
                    # Add CARRY_OUT[pos-1] to RESULT[pos] (cancel pair)
                    self.W_up.data[h, fi(pos-1, E.CARRY_OUT)] = S
                    self.b_up.data[h] = 0.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[fi(pos, E.RESULT), h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, fi(pos-1, E.CARRY_OUT)] = -S
                    self.b_up.data[h] = 0.0
                    self.W_gate.data[h, op_gate] = -1.0
                    self.W_down.data[fi(pos, E.RESULT), h] = 1.0 / S
                    h += 1

                # Clear CARRY_OUT[pos] (cancel pair)
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = -1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = -S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = 1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

            # Step 2: Mod 16 on (RESULT + carry_from_left)
            # Step functions read RESULT[pos] AND CARRY_OUT[pos-1] from INPUT.
            for pos in range(8):
                result_idx = fi(pos, E.RESULT)
                carry_idx = fi(pos, E.CARRY_OUT)

                for k in range(1, 8):
                    threshold = k * 16
                    # Rise: threshold on RESULT[pos] + CARRY_OUT[pos-1]
                    self.W_up.data[h, result_idx] = S
                    if pos > 0:
                        self.W_up.data[h, fi(pos-1, E.CARRY_OUT)] = S
                    self.b_up.data[h] = -S * (threshold - 1)
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = -16.0 / S
                    self.W_down.data[carry_idx, h] = 1.0 / S
                    h += 1

                    # Saturation
                    self.W_up.data[h, result_idx] = S
                    if pos > 0:
                        self.W_up.data[h, fi(pos-1, E.CARRY_OUT)] = S
                    self.b_up.data[h] = -S * threshold
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = 16.0 / S
                    self.W_down.data[carry_idx, h] = -1.0 / S
                    h += 1

            assert h <= 200, f"Used {h} hidden units, expected <= 200"


class MulCarryPass3FFN(FlattenedPureFFN):
    """Layer 4: Add carry from pass 2, mod 16, write binary carry.

    RESULT[i] is 0..15 from pass 2.
    CARRY_OUT[i-1] from pass 2 is 0..7.
    Combined: RESULT[i] + CARRY_OUT[i-1] ∈ [0, 22].
    floor(22/16) = 1 → binary carry.

    Step functions read RESULT[i] + CARRY_OUT[i-1] directly from input.
    Writes binary carry to CARRY_IN[i].
    Clears CARRY_OUT.
    """

    def __init__(self, opcode=Opcode.MUL):
        self.opcode = opcode
        super().__init__(hidden_dim=100)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0
            op_gate = fi(0, E.OP_START + self.opcode)

            for pos in range(8):
                result_idx = fi(pos, E.RESULT)

                if pos > 0:
                    carry_from = fi(pos - 1, E.CARRY_OUT)

                    # Add CARRY_OUT[pos-1] to RESULT[pos] (cancel pair)
                    self.W_up.data[h, carry_from] = S
                    self.b_up.data[h] = 0.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, carry_from] = -S
                    self.b_up.data[h] = 0.0
                    self.W_gate.data[h, op_gate] = -1.0
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

                    # Detect overflow: step(RESULT[pos] + CARRY_OUT[pos-1] >= 16)
                    # Subtract 16 from RESULT and set CARRY_IN[pos] = 1
                    self.W_up.data[h, result_idx] = S
                    self.W_up.data[h, carry_from] = S
                    self.b_up.data[h] = -S * 15.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = -16.0 / S
                    self.W_down.data[fi(pos, E.CARRY_IN), h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, result_idx] = S
                    self.W_up.data[h, carry_from] = S
                    self.b_up.data[h] = -S * 16.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = 16.0 / S
                    self.W_down.data[fi(pos, E.CARRY_IN), h] = -1.0 / S
                    h += 1

            # Clear CARRY_OUT at all positions
            for pos in range(8):
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = -1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = -S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = 1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

            assert h <= 100, f"Used {h} hidden units, expected <= 100"


class MulGenPropFFN(FlattenedPureFFN):
    """Layer 5: Compute G/P for binary carry chain, add carry, mod16.

    Input: RESULT[i] ∈ [0,15], CARRY_IN[i] ∈ {0,1} (from pass 3).
    CARRY_IN[i] is the carry FROM position i to position i+1.
    So carry into position i is CARRY_IN[i-1].

    For carry-lookahead:
    G[i] = step(RESULT[i] + CARRY_IN[i-1] >= 16) → CARRY_OUT[i]
    P[i] = step(RESULT[i] + CARRY_IN[i-1] == 15) → TEMP[i]

    Also: add CARRY_IN[i-1] to RESULT[i], mod16.
    Clear CARRY_IN.
    """

    def __init__(self, opcode=Opcode.MUL):
        self.opcode = opcode
        super().__init__(hidden_dim=100)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0
            op_gate = fi(0, E.OP_START + self.opcode)

            for pos in range(8):
                result_idx = fi(pos, E.RESULT)

                if pos > 0:
                    carry_from = fi(pos - 1, E.CARRY_IN)

                    # Add CARRY_IN[pos-1] to RESULT[pos] (cancel pair)
                    self.W_up.data[h, carry_from] = S
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, carry_from] = -S
                    self.W_gate.data[h, op_gate] = -1.0
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

                    # G[pos] = step(RESULT[pos] + CARRY_IN[pos-1] >= 16)
                    # → -16 to RESULT (mod16), +1 to CARRY_OUT (G)
                    self.W_up.data[h, result_idx] = S
                    self.W_up.data[h, carry_from] = S
                    self.b_up.data[h] = -S * 15.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = -16.0 / S
                    self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, result_idx] = S
                    self.W_up.data[h, carry_from] = S
                    self.b_up.data[h] = -S * 16.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[result_idx, h] = 16.0 / S
                    self.W_down.data[fi(pos, E.CARRY_OUT), h] = -1.0 / S
                    h += 1

                    # P[pos] = step(RESULT[pos] + CARRY_IN[pos-1] == 15)
                    # = step(sum >= 15) - step(sum >= 16)
                    # step(>=15) → +TEMP
                    self.W_up.data[h, result_idx] = S
                    self.W_up.data[h, carry_from] = S
                    self.b_up.data[h] = -S * 14.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, result_idx] = S
                    self.W_up.data[h, carry_from] = S
                    self.b_up.data[h] = -S * 15.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[fi(pos, E.TEMP), h] = -1.0 / S
                    h += 1

                    # Subtract step(>=16) from TEMP for P = step(>=15) - step(>=16)
                    # Reuse the G step pair: add -1/S to TEMP from rise, +1/S from sat
                    # But those units already have W_down entries. We can add more:
                    self.W_down.data[fi(pos, E.TEMP), h-4] = -1.0 / S  # G rise → -TEMP
                    self.W_down.data[fi(pos, E.TEMP), h-3] = 1.0 / S   # G sat → +TEMP

                else:
                    # pos=0: no incoming carry. G[0]=0 always. P[0]=step(RESULT[0]==15)
                    # step(RESULT[0] >= 15): rise at 14, sat at 15
                    self.W_up.data[h, result_idx] = S
                    self.b_up.data[h] = -S * 14.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, result_idx] = S
                    self.b_up.data[h] = -S * 15.0
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[fi(pos, E.TEMP), h] = -1.0 / S
                    h += 1

            # Clear CARRY_IN at all positions
            for pos in range(8):
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = S
                self.W_gate.data[h, fi(pos, E.CARRY_IN)] = -1.0
                self.W_down.data[fi(pos, E.CARRY_IN), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = -S
                self.W_gate.data[h, fi(pos, E.CARRY_IN)] = 1.0
                self.W_down.data[fi(pos, E.CARRY_IN), h] = 1.0 / S
                h += 1

            assert h <= 100, f"Used {h} hidden units, expected <= 100"


class MulBinaryLookaheadFFN(FlattenedPureFFN):
    """Layer 6: Carry-lookahead on binary G/P, clear G/P.

    G[i] in CARRY_OUT[i], P[i] in TEMP[i].
    Standard prefix carry: 28 AND-gate units with W_down = 2.0/S.
    Writes carries to CARRY_IN[i].
    Clears CARRY_OUT (G) and TEMP (P).
    """

    def __init__(self, opcode=Opcode.MUL):
        self.opcode = opcode
        super().__init__(hidden_dim=96)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0
            op_gate = fi(0, E.OP_START + self.opcode)

            # Carry-lookahead: C[i] for i=1..7
            for i in range(1, 8):
                # G[i-1] directly
                self.W_up.data[h, fi(i-1, E.CARRY_OUT)] = S
                self.b_up.data[h] = -S * 0.5
                self.W_gate.data[h, op_gate] = 1.0
                self.W_down.data[fi(i, E.CARRY_IN), h] = 2.0 / S
                h += 1

                # P[i-1]*G[i-2], P[i-1]*P[i-2]*G[i-3], ...
                for j in range(i-2, -1, -1):
                    n_vars = (i - 1 - j) + 1
                    for k in range(j+1, i):
                        self.W_up.data[h, fi(k, E.TEMP)] = S  # P[k]
                    self.W_up.data[h, fi(j, E.CARRY_OUT)] = S  # G[j]
                    self.b_up.data[h] = -S * (n_vars - 0.5)
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[fi(i, E.CARRY_IN), h] = 2.0 / S
                    h += 1

            # Clear G (CARRY_OUT) and P (TEMP) at all positions
            for pos in range(8):
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = -1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = -S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = 1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = S
                self.W_gate.data[h, fi(pos, E.TEMP)] = -1.0
                self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                h += 1
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = -S
                self.W_gate.data[h, fi(pos, E.TEMP)] = 1.0
                self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                h += 1

            assert h <= 96, f"Used {h} hidden units, expected <= 96"


class MulFinalCorrectionFFN(FlattenedPureFFN):
    """Layer 7: Add carry from lookahead, mod16, clear CARRY_IN.

    CARRY_IN[i] = C[i] from lookahead (binary).
    RESULT[i] ∈ [0,15] from GenProp.
    Sum = RESULT[i] + C[i] ∈ [0,16].
    If 16: subtract 16 (no new carry — lookahead handled cascading).
    Clear CARRY_IN.
    """

    def __init__(self, opcode=Opcode.MUL):
        self.opcode = opcode
        super().__init__(hidden_dim=64)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0
            op_gate = fi(0, E.OP_START + self.opcode)

            for pos in range(8):
                result_idx = fi(pos, E.RESULT)
                carry_idx = fi(pos, E.CARRY_IN)

                # Add CARRY_IN to RESULT (cancel pair)
                self.W_up.data[h, carry_idx] = S
                self.W_gate.data[h, op_gate] = 1.0
                self.W_down.data[result_idx, h] = 1.0 / S
                h += 1

                self.W_up.data[h, carry_idx] = -S
                self.W_gate.data[h, op_gate] = -1.0
                self.W_down.data[result_idx, h] = 1.0 / S
                h += 1

                # step(RESULT + CARRY_IN >= 16) → subtract 16
                self.W_up.data[h, result_idx] = S
                self.W_up.data[h, carry_idx] = S
                self.b_up.data[h] = -S * 15.0
                self.W_gate.data[h, op_gate] = 1.0
                self.W_down.data[result_idx, h] = -16.0 / S
                h += 1

                self.W_up.data[h, result_idx] = S
                self.W_up.data[h, carry_idx] = S
                self.b_up.data[h] = -S * 16.0
                self.W_gate.data[h, op_gate] = 1.0
                self.W_down.data[result_idx, h] = 16.0 / S
                h += 1

            # Clear CARRY_IN at all positions
            for pos in range(8):
                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = S
                self.W_gate.data[h, fi(pos, E.CARRY_IN)] = -1.0
                self.W_down.data[fi(pos, E.CARRY_IN), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + self.opcode)] = -S
                self.W_gate.data[h, fi(pos, E.CARRY_IN)] = 1.0
                self.W_down.data[fi(pos, E.CARRY_IN), h] = 1.0 / S
                h += 1

            assert h <= 64, f"Used {h} hidden units, expected <= 64"
