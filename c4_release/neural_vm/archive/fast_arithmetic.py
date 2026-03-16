"""
Fast arithmetic operations using carry-lookahead via FlattenedPureFFN.

Replaces iterative carry propagation (7 rounds × 3 layers = 21 layers)
with parallel prefix carry computation in a single FlattenedPureFFN layer.

ADD: 24 → 3 layers
SUB: 24 → 3 layers
CMP (LT/GT/LE/GE): 27 → 3 layers

Key insight: FlattenedPureFFN flattens [B, 8, 160] → [B, 1, 1280], enabling
cross-position operations in a SINGLE layer. Carry-lookahead computes all
carries in parallel via prefix sums using AND-gate SwiGLU units.

For binary carry at position i:
  G[i] = generate (carry produced locally)
  P[i] = propagate (carry passes through)
  C[i] = G[i-1] OR (P[i-1] AND G[i-2]) OR (P[i-1] AND P[i-2] AND G[i-3]) OR ...

Since G[j]=1 implies P[j]=0, at most one term is nonzero → OR = SUM.
Each AND term uses one SwiGLU unit: up = S*(sum_of_vars - n + 0.5).
silu(S*0.5) ≈ 0.5*S, so W_down = 2.0/S to get output of 1.0.
Total AND terms for 8 positions: 1+2+3+4+5+6+7 = 28 hidden units.
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN, FlattenedPureFFN


# =============================================================================
# ADD Pipeline: 3 layers
# =============================================================================

class AddRawAndGenFFN(PureFFN):
    """Layer 1 of ADD: Compute RAW_SUM, G (generate), P (propagate).

    RAW_SUM = NIB_A + NIB_B (cancel pair, 2 units)
    G = step(NIB_A + NIB_B >= 16) → CARRY_OUT (step pair, 2 units)
    P = step(NIB_A + NIB_B == 15) → TEMP (2 units)

    Total: 6 hidden units.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Units 0-1: RAW_SUM = A + B (cancel pair)
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.NIB_B] = S
            self.W_gate[0, E.OP_START + Opcode.ADD] = S
            self.W_down[E.RAW_SUM, 0] = 1.0 / (S * S)

            self.W_up[1, E.NIB_A] = -S
            self.W_up[1, E.NIB_B] = -S
            self.W_gate[1, E.OP_START + Opcode.ADD] = -S
            self.W_down[E.RAW_SUM, 1] = 1.0 / (S * S)

            # Units 2-3: G = step(A+B >= 16) → CARRY_OUT
            # step(>=16): silu(S*(sum-15))/S - silu(S*(sum-16))/S
            self.W_up[2, E.NIB_A] = S
            self.W_up[2, E.NIB_B] = S
            self.b_up[2] = -S * 15.0
            self.W_gate[2, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            self.W_up[3, E.NIB_A] = S
            self.W_up[3, E.NIB_B] = S
            self.b_up[3] = -S * 16.0
            self.W_gate[3, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S

            # Units 4-5: P = step(A+B == 15) → TEMP
            # P = step(>=15) - step(>=16).
            # step(>=15) → +TEMP, step(>=16) → -TEMP
            # step(>=15): silu(S*(sum-14))/S - silu(S*(sum-15))/S
            self.W_up[4, E.NIB_A] = S
            self.W_up[4, E.NIB_B] = S
            self.b_up[4] = -S * 14.0
            self.W_gate[4, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.TEMP, 4] = 1.0 / S

            self.W_up[5, E.NIB_A] = S
            self.W_up[5, E.NIB_B] = S
            self.b_up[5] = -S * 15.0
            self.W_gate[5, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.TEMP, 5] = -1.0 / S

            # Also subtract step(>=16) from TEMP so TEMP = step(>=15) - step(>=16) = P
            self.W_down[E.TEMP, 2] = -1.0 / S  # subtract G from TEMP
            self.W_down[E.TEMP, 3] = 1.0 / S   # add back saturation


class AddCarryLookaheadFFN(FlattenedPureFFN):
    """Layer 2 of ADD: Parallel carry-lookahead using prefix computation.

    Reads G[0..7] from CARRY_OUT and P[0..7] from TEMP at each position.
    Computes carry into position i:
      C[i] = G[i-1] OR (P[i-1] AND G[i-2]) OR ... OR (P[i-1]...P[0] AND 0)

    Since carries are binary and G[j]=1 → P[j]=0, OR = SUM.
    Each AND term: up = S*(sum_of_vars - n + 0.5), gate = opcode.
    W_down = 2.0/S because silu(S*0.5) ≈ 0.5*S.

    28 AND-gate units for prefix carry chain.
    ~32 units to clear G (CARRY_OUT) and P (TEMP) at all 8 positions.

    Total: ~60 hidden units.
    """

    def __init__(self):
        super().__init__(hidden_dim=60)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0  # hidden unit counter

            for i in range(1, 8):
                # Term 0: C[i] includes G[i-1] directly (1-var AND)
                self.W_up.data[h, fi(i-1, E.CARRY_OUT)] = S
                self.b_up.data[h] = -S * 0.5  # threshold at 0.5
                self.W_gate.data[h, fi(0, E.OP_START + Opcode.ADD)] = 1.0
                self.W_down.data[fi(i, E.CARRY_IN), h] = 2.0 / S
                h += 1

                # Terms with P's: P[i-1]*G[i-2], P[i-1]*P[i-2]*G[i-3], ...
                for j in range(i-2, -1, -1):
                    n_vars = (i - 1 - j) + 1  # number of variables
                    for k in range(j+1, i):
                        self.W_up.data[h, fi(k, E.TEMP)] = S  # P[k]
                    self.W_up.data[h, fi(j, E.CARRY_OUT)] = S  # G[j]
                    self.b_up.data[h] = -S * (n_vars - 0.5)
                    self.W_gate.data[h, fi(0, E.OP_START + Opcode.ADD)] = 1.0
                    self.W_down.data[fi(i, E.CARRY_IN), h] = 2.0 / S
                    h += 1

            # Now clear G (CARRY_OUT) and P (TEMP) at all 8 positions
            for pos in range(8):
                # Clear CARRY_OUT[pos] (cancel pair)
                self.W_up.data[h, fi(pos, E.OP_START + Opcode.ADD)] = S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = -1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + Opcode.ADD)] = -S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = 1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

                # Clear TEMP[pos] (cancel pair)
                self.W_up.data[h, fi(pos, E.OP_START + Opcode.ADD)] = S
                self.W_gate.data[h, fi(pos, E.TEMP)] = -1.0
                self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                h += 1

                self.W_up.data[h, fi(pos, E.OP_START + Opcode.ADD)] = -S
                self.W_gate.data[h, fi(pos, E.TEMP)] = 1.0
                self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                h += 1

            assert h <= 60, f"Used {h} hidden units, expected <= 60"


class AddFinalResultFFN(PureFFN):
    """Layer 3 of ADD: Compute RESULT = (RAW_SUM + CARRY_IN) mod 16.

    Copy RAW_SUM + CARRY_IN to RESULT, subtract 16 when sum >= 16.
    Clear RAW_SUM and CARRY_IN.

    Total: 10 hidden units.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=10)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Units 0-1: Copy RAW_SUM to RESULT (cancel pair)
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Units 2-3: Add CARRY_IN to RESULT (cancel pair)
            self.W_up[2, E.OP_START + Opcode.ADD] = S
            self.W_gate[2, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.ADD] = -S
            self.W_gate[3, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S

            # Units 4-5: Subtract 16 when (RAW_SUM + CARRY_IN) >= 16
            # step pair at threshold 16
            self.W_up[4, E.RAW_SUM] = S
            self.W_up[4, E.CARRY_IN] = S
            self.b_up[4] = -S * 15.0
            self.W_gate[4, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 4] = -16.0 / S

            self.W_up[5, E.RAW_SUM] = S
            self.W_up[5, E.CARRY_IN] = S
            self.b_up[5] = -S * 16.0
            self.W_gate[5, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 5] = 16.0 / S

            # Units 6-7: Clear RAW_SUM (cancel pair)
            self.W_up[6, E.OP_START + Opcode.ADD] = S
            self.W_gate[6, E.RAW_SUM] = -1.0
            self.W_down[E.RAW_SUM, 6] = 1.0 / S

            self.W_up[7, E.OP_START + Opcode.ADD] = -S
            self.W_gate[7, E.RAW_SUM] = 1.0
            self.W_down[E.RAW_SUM, 7] = 1.0 / S

            # Units 8-9: Clear CARRY_IN (cancel pair)
            self.W_up[8, E.OP_START + Opcode.ADD] = S
            self.W_gate[8, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 8] = 1.0 / S

            self.W_up[9, E.OP_START + Opcode.ADD] = -S
            self.W_gate[9, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 9] = 1.0 / S


# =============================================================================
# SUB Pipeline: 3 layers
# =============================================================================

class SubRawAndGenFFN(PureFFN):
    """Layer 1 of SUB: Compute RAW_SUM (=A-B), G (borrow generate), P (propagate).

    RAW_SUM = NIB_A - source_b (cancel pair, 2 units)
    G = step(source_b > A) = step(source_b - A >= 1) → CARRY_OUT (step pair, 2 units)
    P = step(A == source_b) = step(A-source_b >= 0) - step(A-source_b >= 1) → TEMP (3 units, merged)

    Args:
        opcode: Opcode to gate on (default SUB).
        source_b: Slot to read second operand from (default NIB_B).
                  For MOD: E.RESULT (reads product from multiply phase).
        clear_result: If True, also clear RESULT (adds 2 hidden units).

    Total: 7 hidden units (9 with clear_result).
    """

    def __init__(self, opcode=Opcode.SUB, source_b=E.NIB_B, clear_result=False):
        self.opcode = opcode
        self.source_b = source_b
        self.clear_result = clear_result
        hdim = 9 if clear_result else 7
        super().__init__(E.DIM, hidden_dim=hdim)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Units 0-1: RAW_SUM = A - B (cancel pair)
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_gate[0, self.source_b] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_gate[1, self.source_b] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S

            # Units 2-3: G = step(B > A) = step(B-A >= 1)
            # silu(S*(B-A)) - silu(S*(B-A-1))
            self.W_up[2, self.source_b] = S
            self.W_up[2, E.NIB_A] = -S
            self.b_up[2] = 0.0  # rise: S*(B-A-1+1) = S*(B-A)
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            self.W_up[3, self.source_b] = S
            self.W_up[3, E.NIB_A] = -S
            self.b_up[3] = -S * 1.0  # sat: S*(B-A-1)
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S

            # Units 4-6: P = step(A == B) = step(d >= 0) - step(d >= 1)
            # where d = A - B.
            # 3-unit merged approach:
            #   silu(S*(d+1))/S - 2*silu(S*d)/S + silu(S*(d-1))/S

            # Unit 4 (rise of step(>=0)): silu(S*(d+1)), W_down = +1/S
            self.W_up[4, E.NIB_A] = S
            self.W_up[4, self.source_b] = -S
            self.b_up[4] = S * 1.0
            self.W_gate[4, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 4] = 1.0 / S

            # Unit 5 (merged sat+rise): silu(S*d), W_down = -2/S
            self.W_up[5, E.NIB_A] = S
            self.W_up[5, self.source_b] = -S
            self.b_up[5] = 0.0
            self.W_gate[5, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 5] = -2.0 / S

            # Unit 6 (sat of step(>=1)): silu(S*(d-1)), W_down = +1/S
            self.W_up[6, E.NIB_A] = S
            self.W_up[6, self.source_b] = -S
            self.b_up[6] = -S * 1.0
            self.W_gate[6, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 6] = 1.0 / S

            # Optional: clear RESULT (for MOD, where RESULT has the product)
            if self.clear_result:
                self.W_up[7, E.OP_START + self.opcode] = S
                self.W_gate[7, E.RESULT] = -1.0
                self.W_down[E.RESULT, 7] = 1.0 / S

                self.W_up[8, E.OP_START + self.opcode] = -S
                self.W_gate[8, E.RESULT] = 1.0
                self.W_down[E.RESULT, 8] = 1.0 / S


class SubBorrowLookaheadFFN(FlattenedPureFFN):
    """Layer 2 of SUB: Parallel borrow-lookahead.

    Identical prefix formula to ADD carry-lookahead:
      C[i] = G[i-1] OR (P[i-1] AND G[i-2]) OR ...

    W_down = 2.0/S for AND-gate units (silu(S*0.5) ≈ 0.5*S correction).
    28 AND-gate units + 32 clearing units = 60 hidden units.
    """

    def __init__(self, opcode=Opcode.SUB):
        self.opcode = opcode
        super().__init__(hidden_dim=60)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0

            for i in range(1, 8):
                # Term: G[i-1]
                self.W_up.data[h, fi(i-1, E.CARRY_OUT)] = S
                self.b_up.data[h] = -S * 0.5
                self.W_gate.data[h, fi(0, E.OP_START + self.opcode)] = 1.0
                self.W_down.data[fi(i, E.CARRY_IN), h] = 2.0 / S
                h += 1

                for j in range(i-2, -1, -1):
                    n_vars = (i - 1 - j) + 1
                    for k in range(j+1, i):
                        self.W_up.data[h, fi(k, E.TEMP)] = S
                    self.W_up.data[h, fi(j, E.CARRY_OUT)] = S
                    self.b_up.data[h] = -S * (n_vars - 0.5)
                    self.W_gate.data[h, fi(0, E.OP_START + self.opcode)] = 1.0
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

            assert h <= 60, f"Used {h} hidden units, expected <= 60"


class SubFinalResultFFN(PureFFN):
    """Layer 3 of SUB: Compute RESULT = (RAW_SUM - BORROW_IN + 16) mod 16.

    = (A - B - borrow + 16) mod 16.
    RAW_SUM = A - B (range -15..15).
    With borrow: RAW_SUM - CARRY_IN (range -16..15).
    Add 16 when negative, result is always 0..15.

    Total: 10 hidden units.
    """

    def __init__(self, opcode=Opcode.SUB):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=10)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Units 0-1: Copy RAW_SUM to RESULT (cancel pair)
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Units 2-3: Subtract CARRY_IN from RESULT (cancel pair)
            self.W_up[2, E.OP_START + self.opcode] = S
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + self.opcode] = -S
            self.W_gate[3, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 3] = 1.0 / S

            # Units 4-5: Add 16 when (RAW_SUM - CARRY_IN) < 0
            # step(CARRY_IN - RAW_SUM >= 1)
            self.W_up[4, E.CARRY_IN] = S
            self.W_up[4, E.RAW_SUM] = -S
            self.b_up[4] = 0.0
            self.W_gate[4, E.OP_START + self.opcode] = 1.0
            self.W_down[E.RESULT, 4] = 16.0 / S

            self.W_up[5, E.CARRY_IN] = S
            self.W_up[5, E.RAW_SUM] = -S
            self.b_up[5] = -S * 1.0
            self.W_gate[5, E.OP_START + self.opcode] = 1.0
            self.W_down[E.RESULT, 5] = -16.0 / S

            # Units 6-7: Clear RAW_SUM
            self.W_up[6, E.OP_START + self.opcode] = S
            self.W_gate[6, E.RAW_SUM] = -1.0
            self.W_down[E.RAW_SUM, 6] = 1.0 / S

            self.W_up[7, E.OP_START + self.opcode] = -S
            self.W_gate[7, E.RAW_SUM] = 1.0
            self.W_down[E.RAW_SUM, 7] = 1.0 / S

            # Units 8-9: Clear CARRY_IN
            self.W_up[8, E.OP_START + self.opcode] = S
            self.W_gate[8, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 8] = 1.0 / S

            self.W_up[9, E.OP_START + self.opcode] = -S
            self.W_gate[9, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 9] = 1.0 / S


# =============================================================================
# CMP Pipeline (LT/GT/LE/GE): 3 layers
# =============================================================================

class CmpRawDiffAndGenFFN(PureFFN):
    """Layer 1 of CMP: Compute RAW_SUM, G (borrow gen), P (propagate).

    For a given opcode (LT or GE: A-B, GT or LE: B-A):
    RAW_SUM = first - second
    G = step(second > first) (borrow generated)
    P = step(first == second) (borrow propagates)

    Total: 7 hidden units.
    """

    def __init__(self, opcode: int, swap: bool = False):
        self.opcode = opcode
        self.swap = swap  # True for GT, LE (compute B-A instead of A-B)
        super().__init__(E.DIM, hidden_dim=7)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            first = E.NIB_B if self.swap else E.NIB_A
            second = E.NIB_A if self.swap else E.NIB_B

            # Units 0-1: RAW_SUM = first - second (cancel pair)
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, first] = 1.0
            self.W_gate[0, second] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, first] = -1.0
            self.W_gate[1, second] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S

            # Units 2-3: G = step(second > first) = step(second - first >= 1)
            self.W_up[2, second] = S
            self.W_up[2, first] = -S
            self.b_up[2] = 0.0  # rise: S*(second-first)
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            self.W_up[3, second] = S
            self.W_up[3, first] = -S
            self.b_up[3] = -S * 1.0  # sat: S*(second-first-1)
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S

            # Units 4-6: P = step(first == second)
            # = step(d >= 0) - step(d >= 1) where d = first - second
            # 3-unit merged: silu(S*(d+1))/S - 2*silu(S*d)/S + silu(S*(d-1))/S
            self.W_up[4, first] = S
            self.W_up[4, second] = -S
            self.b_up[4] = S * 1.0
            self.W_gate[4, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 4] = 1.0 / S

            self.W_up[5, first] = S
            self.W_up[5, second] = -S
            self.b_up[5] = 0.0
            self.W_gate[5, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 5] = -2.0 / S

            self.W_up[6, first] = S
            self.W_up[6, second] = -S
            self.b_up[6] = -S * 1.0
            self.W_gate[6, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 6] = 1.0 / S


class CmpBorrowLookaheadFFN(FlattenedPureFFN):
    """Layer 2 of CMP: Borrow-lookahead + broadcast final borrow to RESULT.

    Same 28-unit prefix as SUB borrow-lookahead with W_down = 2.0/S.
    Additionally computes C[8] = final borrow after position 7.
    Writes RESULT = C[8] at position 0.

    For LT/GT: RESULT = final_borrow directly (A < B iff borrow out of MSB).
    For LE/GE: inverted in layer 3.

    28 carry units + 8 final-borrow units + 32 clearing units = ~68.
    """

    def __init__(self, opcodes: list):
        """opcodes: list of opcodes this expert handles (for gating)."""
        self.opcodes = opcodes
        super().__init__(hidden_dim=80)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0
            op_gate = fi(0, E.OP_START + self.opcodes[0])

            # Carry-lookahead: C[i] for i=1..7 (same as ADD/SUB)
            for i in range(1, 8):
                self.W_up.data[h, fi(i-1, E.CARRY_OUT)] = S
                self.b_up.data[h] = -S * 0.5
                self.W_gate.data[h, op_gate] = 1.0
                self.W_down.data[fi(i, E.CARRY_IN), h] = 2.0 / S
                h += 1

                for j in range(i-2, -1, -1):
                    n_vars = (i - 1 - j) + 1
                    for k in range(j+1, i):
                        self.W_up.data[h, fi(k, E.TEMP)] = S
                    self.W_up.data[h, fi(j, E.CARRY_OUT)] = S
                    self.b_up.data[h] = -S * (n_vars - 0.5)
                    self.W_gate.data[h, op_gate] = 1.0
                    self.W_down.data[fi(i, E.CARRY_IN), h] = 2.0 / S
                    h += 1

            # Final borrow C[8]: full prefix expansion.
            # G[7] directly (1-var term)
            self.W_up.data[h, fi(7, E.CARRY_OUT)] = S
            self.b_up.data[h] = -S * 0.5
            self.W_gate.data[h, op_gate] = 1.0
            self.W_down.data[fi(0, E.RESULT), h] = 2.0 / S
            h += 1

            # P[7]*G[6], P[7]*P[6]*G[5], ..., P[7]*...*P[1]*G[0]
            for j in range(6, -1, -1):
                n_vars = (7 - j) + 1
                for k in range(j+1, 8):
                    self.W_up.data[h, fi(k, E.TEMP)] = S  # P[k]
                self.W_up.data[h, fi(j, E.CARRY_OUT)] = S  # G[j]
                self.b_up.data[h] = -S * (n_vars - 0.5)
                self.W_gate.data[h, op_gate] = 1.0
                self.W_down.data[fi(0, E.RESULT), h] = 2.0 / S
                h += 1

            # Clear G (CARRY_OUT), P (TEMP), and CARRY_IN at all positions
            for pos in range(8):
                # Clear CARRY_OUT
                self.W_up.data[h, fi(pos, E.OP_START + self.opcodes[0])] = S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = -1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1
                self.W_up.data[h, fi(pos, E.OP_START + self.opcodes[0])] = -S
                self.W_gate.data[h, fi(pos, E.CARRY_OUT)] = 1.0
                self.W_down.data[fi(pos, E.CARRY_OUT), h] = 1.0 / S
                h += 1

                # Clear TEMP
                self.W_up.data[h, fi(pos, E.OP_START + self.opcodes[0])] = S
                self.W_gate.data[h, fi(pos, E.TEMP)] = -1.0
                self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                h += 1
                self.W_up.data[h, fi(pos, E.OP_START + self.opcodes[0])] = -S
                self.W_gate.data[h, fi(pos, E.TEMP)] = 1.0
                self.W_down.data[fi(pos, E.TEMP), h] = 1.0 / S
                h += 1

            assert h <= 80, f"Used {h} hidden units, expected <= 80"


class CmpClearRawSumFFN(PureFFN):
    """Clear RAW_SUM and CARRY_IN after CMP. Also clear RESULT at non-zero positions."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear RAW_SUM
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.RAW_SUM] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.RAW_SUM] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S

            # Clear CARRY_IN
            self.W_up[2, E.OP_START + self.opcode] = S
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 2] = 1.0 / S

            self.W_up[3, E.OP_START + self.opcode] = -S
            self.W_gate[3, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 3] = 1.0 / S

            # Clear RESULT at positions > 0 (result should only be at pos 0)
            # step(POS >= 1) * (-RESULT) → RESULT
            # Using integer thresholds for step(POS >= 1):
            # rise: silu(S*POS), sat: silu(S*(POS-1))
            # Unit 4 (rise): gate = -RESULT, W_down = +1/S
            self.W_up[4, E.POS] = S
            self.b_up[4] = 0.0  # rise: S*(POS - 0) = S*POS
            self.W_gate[4, E.RESULT] = -1.0
            self.W_down[E.RESULT, 4] = 1.0 / S

            # Unit 5 (sat): gate = -RESULT, W_down = -1/S
            self.W_up[5, E.POS] = S
            self.b_up[5] = -S * 1.0  # sat: S*(POS - 1)
            self.W_gate[5, E.RESULT] = -1.0
            self.W_down[E.RESULT, 5] = -1.0 / S
