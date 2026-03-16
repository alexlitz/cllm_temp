"""
Fast shift operations using FlattenedPureFFN — 2 layers.

SHL/SHR: 31 → 2 layers

Layer 1 (PureFFN): Precompute all 4 sub-nibble shifts of each input nibble.
Layer 2 (FlattenedPureFFN): Select and combine based on shift amount indicator.

For SHL by k = 4q + r:
  result[j] = s_r(a[j-q]) + c_r(a[j-q-1])
where:
  s_r(a) = (a * 2^r) mod 16     (sub-nibble shifted value)
  c_r(a) = floor(a * 2^r / 16)  (carry to next position)

Critical property: max(s_r) + max(c_r) = 15, so no overflow.

The shift amount is encoded in NIB_B at positions 0-1.
Indicator for step(val == k) uses 3-unit integer threshold pattern:
  silu(S*(val-k+1))/S - 2*silu(S*(val-k))/S + silu(S*(val-k-1))/S
This gives exactly 1 when val=k, 0 otherwise.
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN, FlattenedPureFFN


# =============================================================================
# SHL Pipeline: 2 layers
# =============================================================================

class ShiftPrecomputeFFN(PureFFN):
    """Layer 1 of SHL: Precompute sub-nibble shifts for each nibble.

    For each position i and sub-nibble shift r in {0,1,2,3}:
      r=0: s0=a[i] (already in NIB_A), c0=0
      r=1: s1=(2*a) mod 16, c1=step(a>=8) = floor(2*a/16)
      r=2: s2=(4*a) mod 16, c2=floor(4*a/16)
      r=3: s3=(8*a) mod 16, c3=floor(8*a/16)

    Storage layout:
      r=0: NIB_A (unchanged), carry=0 (implicit)
      r=1: s1 → RAW_SUM, c1 → TEMP
      r=2: s2 → CARRY_IN, c2 → SHIFT_EXTRACT_A
      r=3: s3 → CARRY_OUT, c3 → SHIFT_EXTRACT_B
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=50)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            h = 0

            # r=1: s1 = 2*a mod 16 → RAW_SUM, c1 = floor(2*a/16) → TEMP
            # s1 = 2*a (cancel pair)
            self.W_up[h, E.NIB_A] = S
            self.W_gate[h, E.OP_START + Opcode.SHL] = 2.0
            self.W_down[E.RAW_SUM, h] = 1.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = -S
            self.W_gate[h, E.OP_START + Opcode.SHL] = -2.0
            self.W_down[E.RAW_SUM, h] = 1.0 / S
            h += 1

            # c1 = step(a >= 8): floor(2a/16) = step(a >= 8)
            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 7.0
            self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
            self.W_down[E.TEMP, h] = 1.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 8.0
            self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
            self.W_down[E.TEMP, h] = -1.0 / S
            h += 1

            # Subtract 16 when 2*a >= 16 (i.e. a >= 8) from RAW_SUM
            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 7.0
            self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
            self.W_down[E.RAW_SUM, h] = -16.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 8.0
            self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
            self.W_down[E.RAW_SUM, h] = 16.0 / S
            h += 1

            # r=2: s2 = 4*a mod 16 → CARRY_IN, c2 = floor(4*a/16) → SHIFT_EXTRACT_A
            # s2 = 4*a (cancel pair)
            self.W_up[h, E.NIB_A] = S
            self.W_gate[h, E.OP_START + Opcode.SHL] = 4.0
            self.W_down[E.CARRY_IN, h] = 1.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = -S
            self.W_gate[h, E.OP_START + Opcode.SHL] = -4.0
            self.W_down[E.CARRY_IN, h] = 1.0 / S
            h += 1

            # c2 = floor(4*a/16): steps at a=4,8,12 (values 0,1,2,3)
            for thresh in [4, 8, 12]:
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (thresh - 1)
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.SHIFT_EXTRACT_A, h] = 1.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * thresh
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.SHIFT_EXTRACT_A, h] = -1.0 / S
                h += 1

            # Subtract 16*floor from CARRY_IN to get mod 16
            for thresh in [4, 8, 12]:
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (thresh - 1)
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.CARRY_IN, h] = -16.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * thresh
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.CARRY_IN, h] = 16.0 / S
                h += 1

            # r=3: s3 = 8*a mod 16 → CARRY_OUT, c3 = floor(8*a/16) → SHIFT_EXTRACT_B
            # s3 = 8*a (cancel pair)
            self.W_up[h, E.NIB_A] = S
            self.W_gate[h, E.OP_START + Opcode.SHL] = 8.0
            self.W_down[E.CARRY_OUT, h] = 1.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = -S
            self.W_gate[h, E.OP_START + Opcode.SHL] = -8.0
            self.W_down[E.CARRY_OUT, h] = 1.0 / S
            h += 1

            # c3 = floor(8*a/16): steps at a=2,4,6,8,10,12,14
            for thresh in [2, 4, 6, 8, 10, 12, 14]:
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (thresh - 1)
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.SHIFT_EXTRACT_B, h] = 1.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * thresh
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.SHIFT_EXTRACT_B, h] = -1.0 / S
                h += 1

            # Subtract 16*floor from CARRY_OUT to get mod 16
            for thresh in [2, 4, 6, 8, 10, 12, 14]:
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (thresh - 1)
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.CARRY_OUT, h] = -16.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * thresh
                self.W_gate[h, E.OP_START + Opcode.SHL] = 1.0
                self.W_down[E.CARRY_OUT, h] = 16.0 / S
                h += 1

            assert h <= 50, f"Used {h} hidden units, expected <= 50"


class ShiftSelectFFN(FlattenedPureFFN):
    """Layer 2 of SHL: Select and combine based on shift amount.

    For each output position j (0..7) and shift amount k (0..31):
      q = k // 4, r = k % 4
      If j >= q: RESULT[j] += (s_r[j-q] + c_r[j-q-1]) * indicator(val == k)

    Indicator uses 3-unit integer threshold pattern:
      silu(S*(val-k+1)) * gate * (1/S)   → +1 when val=k
      silu(S*(val-k))   * gate * (-2/S)  → -2+2=0
      silu(S*(val-k-1)) * gate * (1/S)   → +1-1=0

    144 valid pairs × 3 units = 432 + clearing ≈ 530.
    """

    def __init__(self):
        super().__init__(hidden_dim=550)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0

            s_slot = {0: E.NIB_A, 1: E.RAW_SUM, 2: E.CARRY_IN, 3: E.CARRY_OUT}
            c_slot = {0: None, 1: E.TEMP, 2: E.SHIFT_EXTRACT_A, 3: E.SHIFT_EXTRACT_B}

            val_0 = fi(0, E.NIB_B)
            val_1 = fi(1, E.NIB_B)

            for k in range(32):
                q = k // 4
                r = k % 4

                for j in range(8):
                    src = j - q
                    if src < 0 or src >= 8:
                        continue

                    src_carry = j - q - 1
                    result_idx = fi(j, E.RESULT)

                    # Build gate: s_r[src] + c_r[src_carry]
                    def set_gate(unit):
                        self.W_gate.data[unit, fi(src, s_slot[r])] = 1.0
                        if c_slot[r] is not None and src_carry >= 0:
                            self.W_gate.data[unit, fi(src_carry, c_slot[r])] = 1.0

                    # 3-unit indicator: step(val == k)
                    # Unit A: silu(S*(val-k+1)), W_down = +1/S
                    self.W_up.data[h, val_0] = S
                    self.W_up.data[h, val_1] = S * 16.0
                    self.b_up.data[h] = -S * (k - 1)
                    set_gate(h)
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

                    # Unit B: silu(S*(val-k)), W_down = -2/S
                    self.W_up.data[h, val_0] = S
                    self.W_up.data[h, val_1] = S * 16.0
                    self.b_up.data[h] = -S * k
                    set_gate(h)
                    self.W_down.data[result_idx, h] = -2.0 / S
                    h += 1

                    # Unit C: silu(S*(val-k-1)), W_down = +1/S
                    self.W_up.data[h, val_0] = S
                    self.W_up.data[h, val_1] = S * 16.0
                    self.b_up.data[h] = -S * (k + 1)
                    set_gate(h)
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

            # Clear precomputed slots at all positions
            clear_slots = [E.RAW_SUM, E.TEMP, E.CARRY_IN, E.CARRY_OUT,
                           E.SHIFT_EXTRACT_A, E.SHIFT_EXTRACT_B]
            for pos in range(8):
                for slot in clear_slots:
                    self.W_up.data[h, fi(pos, E.OP_START + Opcode.SHL)] = S
                    self.W_gate.data[h, fi(pos, slot)] = -1.0
                    self.W_down.data[fi(pos, slot), h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, fi(pos, E.OP_START + Opcode.SHL)] = -S
                    self.W_gate.data[h, fi(pos, slot)] = 1.0
                    self.W_down.data[fi(pos, slot), h] = 1.0 / S
                    h += 1

            assert h <= 550, f"Used {h} hidden units, expected <= 550"


# =============================================================================
# SHR Pipeline: 2 layers
# =============================================================================

class ShiftRPrecomputeFFN(PureFFN):
    """Layer 1 of SHR: Precompute sub-nibble right-shifts.

    For SHR by r bits (sub-nibble):
      s_r(a) = floor(a / 2^r)
      c_r(a) = (a mod 2^r) * 2^(4-r)

    Storage layout (same slots as SHL):
      r=0: s0=a (NIB_A), c0=0
      r=1: s1=floor(a/2) → RAW_SUM, c1=(a mod 2)*8 → TEMP
      r=2: s2=floor(a/4) → CARRY_IN, c2=(a mod 4)*4 → SHIFT_EXTRACT_A
      r=3: s3=floor(a/8) → CARRY_OUT, c3=(a mod 8)*2 → SHIFT_EXTRACT_B
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=50)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            h = 0

            # r=1: s1 = floor(a/2) → RAW_SUM
            for thresh in [2, 4, 6, 8, 10, 12, 14]:
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (thresh - 1)
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.RAW_SUM, h] = 1.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * thresh
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.RAW_SUM, h] = -1.0 / S
                h += 1

            # c1 = (a mod 2) * 8 → TEMP
            # bit0(a) alternates: 0,1,0,1,...  multiply by 8
            for k in range(1, 16):
                sign = 1 if (k % 2 == 1) else -1
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (k - 1)
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.TEMP, h] = sign * 8.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * k
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.TEMP, h] = -sign * 8.0 / S
                h += 1

            assert h <= 50, f"Used {h} hidden units (r=1), expected <= 50"


class ShiftRPrecompute2FFN(PureFFN):
    """Precompute r=2 and r=3 sub-nibble right-shifts.

    r=2: s2=floor(a/4) → CARRY_IN, c2=(a mod 4)*4 → SHIFT_EXTRACT_A
    r=3: s3=floor(a/8) → CARRY_OUT, c3=(a mod 8)*2 → SHIFT_EXTRACT_B
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=46)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            h = 0

            # r=2: s2 = floor(a/4) → CARRY_IN
            for thresh in [4, 8, 12]:
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (thresh - 1)
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.CARRY_IN, h] = 1.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * thresh
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.CARRY_IN, h] = -1.0 / S
                h += 1

            # c2 = (a mod 4) * 4 = 4*a - 16*floor(a/4) → SHIFT_EXTRACT_A
            # 4*a (cancel pair)
            self.W_up[h, E.NIB_A] = S
            self.W_gate[h, E.OP_START + Opcode.SHR] = 4.0
            self.W_down[E.SHIFT_EXTRACT_A, h] = 1.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = -S
            self.W_gate[h, E.OP_START + Opcode.SHR] = -4.0
            self.W_down[E.SHIFT_EXTRACT_A, h] = 1.0 / S
            h += 1

            # Subtract 16*floor(a/4) from SHIFT_EXTRACT_A
            for thresh in [4, 8, 12]:
                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * (thresh - 1)
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.SHIFT_EXTRACT_A, h] = -16.0 / S
                h += 1

                self.W_up[h, E.NIB_A] = S
                self.b_up[h] = -S * thresh
                self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
                self.W_down[E.SHIFT_EXTRACT_A, h] = 16.0 / S
                h += 1

            # r=3: s3 = floor(a/8) → CARRY_OUT
            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 7.0
            self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
            self.W_down[E.CARRY_OUT, h] = 1.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 8.0
            self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
            self.W_down[E.CARRY_OUT, h] = -1.0 / S
            h += 1

            # c3 = (a mod 8) * 2 = 2*a - 16*floor(a/8) → SHIFT_EXTRACT_B
            self.W_up[h, E.NIB_A] = S
            self.W_gate[h, E.OP_START + Opcode.SHR] = 2.0
            self.W_down[E.SHIFT_EXTRACT_B, h] = 1.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = -S
            self.W_gate[h, E.OP_START + Opcode.SHR] = -2.0
            self.W_down[E.SHIFT_EXTRACT_B, h] = 1.0 / S
            h += 1

            # Subtract 16*floor(a/8) from SHIFT_EXTRACT_B
            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 7.0
            self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
            self.W_down[E.SHIFT_EXTRACT_B, h] = -16.0 / S
            h += 1

            self.W_up[h, E.NIB_A] = S
            self.b_up[h] = -S * 8.0
            self.W_gate[h, E.OP_START + Opcode.SHR] = 1.0
            self.W_down[E.SHIFT_EXTRACT_B, h] = 16.0 / S
            h += 1

            assert h <= 46, f"Used {h} hidden units, expected <= 46"


class ShiftRSelectFFN(FlattenedPureFFN):
    """Layer 2 of SHR: Select and combine based on shift amount.

    For SHR by k = 4q + r:
      result[j] = s_r(a[j+q]) + c_r(a[j+q+1])

    SHR sources from HIGHER positions (j+q), unlike SHL (j-q).
    c_r comes from a[j+q+1] (the position above, whose low bits shift down).

    Same 3-unit indicator as ShiftSelectFFN.
    """

    def __init__(self):
        super().__init__(hidden_dim=550)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            h = 0

            s_slot = {0: E.NIB_A, 1: E.RAW_SUM, 2: E.CARRY_IN, 3: E.CARRY_OUT}
            c_slot = {0: None, 1: E.TEMP, 2: E.SHIFT_EXTRACT_A, 3: E.SHIFT_EXTRACT_B}

            val_0 = fi(0, E.NIB_B)
            val_1 = fi(1, E.NIB_B)

            for k in range(32):
                q = k // 4
                r = k % 4

                for j in range(8):
                    src = j + q  # source position for s_r (SHR: higher positions)
                    if src >= 8:
                        continue

                    src_carry = j + q + 1  # source for c_r (one above)
                    result_idx = fi(j, E.RESULT)

                    def set_gate(unit):
                        self.W_gate.data[unit, fi(src, s_slot[r])] = 1.0
                        if c_slot[r] is not None and src_carry < 8:
                            self.W_gate.data[unit, fi(src_carry, c_slot[r])] = 1.0

                    # 3-unit indicator: step(val == k)
                    # Unit A: silu(S*(val-k+1)), W_down = +1/S
                    self.W_up.data[h, val_0] = S
                    self.W_up.data[h, val_1] = S * 16.0
                    self.b_up.data[h] = -S * (k - 1)
                    set_gate(h)
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

                    # Unit B: silu(S*(val-k)), W_down = -2/S
                    self.W_up.data[h, val_0] = S
                    self.W_up.data[h, val_1] = S * 16.0
                    self.b_up.data[h] = -S * k
                    set_gate(h)
                    self.W_down.data[result_idx, h] = -2.0 / S
                    h += 1

                    # Unit C: silu(S*(val-k-1)), W_down = +1/S
                    self.W_up.data[h, val_0] = S
                    self.W_up.data[h, val_1] = S * 16.0
                    self.b_up.data[h] = -S * (k + 1)
                    set_gate(h)
                    self.W_down.data[result_idx, h] = 1.0 / S
                    h += 1

            # Clear precomputed slots at all positions
            clear_slots = [E.RAW_SUM, E.TEMP, E.CARRY_IN, E.CARRY_OUT,
                           E.SHIFT_EXTRACT_A, E.SHIFT_EXTRACT_B]
            for pos in range(8):
                for slot in clear_slots:
                    self.W_up.data[h, fi(pos, E.OP_START + Opcode.SHR)] = S
                    self.W_gate.data[h, fi(pos, slot)] = -1.0
                    self.W_down.data[fi(pos, slot), h] = 1.0 / S
                    h += 1

                    self.W_up.data[h, fi(pos, E.OP_START + Opcode.SHR)] = -S
                    self.W_gate.data[h, fi(pos, slot)] = 1.0
                    self.W_down.data[fi(pos, slot), h] = 1.0 / S
                    h += 1

            assert h <= 550, f"Used {h} hidden units, expected <= 550"
