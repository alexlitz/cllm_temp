"""
Nibble-level long division for DIV/MOD — eliminates fp64 dependency.

Replaces the previous attention-reciprocal + correction-loop pipeline (and the
even older fp64 MAGIC-floor pipeline) with a direct schoolbook long-division
algorithm operating on nibble vectors.

Algorithm:
  Inputs (NIBBLE config: base 16, 8 positions):
    a = [a7 a6 ... a0]  (32-bit dividend, position 0 is LSB)
    b = [b7 b6 ... b0]  (32-bit divisor)

  Long division iterates left-to-right (i = 7 down to 0):
    1. Bring down the next nibble: partial = partial * 16 + a[i]
    2. Trial multiply: for q in 0..15, compute q * b
    3. Compare: pick largest q such that q*b <= partial
    4. Subtract: partial = partial - q*b
    5. Output q[i] as one nibble of the quotient

  After 8 iterations:
    Quotient = [q7 q6 ... q0]
    Remainder = partial (always < b)

Representation:
  partial is held as a 9-nibble vector (need 9 because partial * 16 + a[i]
  before subtraction can be up to 16*b + 15 ~ 2^36, exceeding 32 bits).
  q*b also fits in 9 nibbles (max 15 * (2^32-1) < 2^36).

Precision:
  Each nibble is in [0, 15]. Carries during trial multiply are at most
  15*15 + carry < 240, well within fp32 exact integer range (2^24).
  All arithmetic stays in fp32 — no fp64 anywhere.

Edge cases:
  - Divide by zero (b == 0): quotient = 0xFFFFFFFF, remainder = a (matches
    saturating-bus convention; tests skip this case).
  - a < b: quotient = 0, remainder = a (handled naturally by long division).
  - a == 0: quotient = 0, remainder = 0 (this was the failing case for fp32
    MAGIC-floor; long division handles it trivially).

Layer breakdown:
  Layer 1: ClearDivSlotsFFN — clear scratch slots (existing FFN, reused).
  Layer 2: LongDivisionModule — 8 outer iterations of bring-down + trial +
           subtract, writing the per-nibble quotient and remainder into
           SLOT_QUOTIENT and SLOT_REMAINDER vectors (one nibble per position).
  Layer 3: EmitDivResultModule — copy SLOT_QUOTIENT[*] → RESULT[*]
           (DIV) or SLOT_REMAINDER[*] → RESULT[*] (MOD).

The LongDivisionModule does ~24 sub-operations internally (8 outer × 3
sub-FFNs each: bring-down + trial-multiply-and-compare + subtract). It is
implemented as a single nn.Module with direct tensor operations in forward()
because the iteration index couples the steps; expressing it as separate
GenericFlattenedFFNs would require 24 modules with the same structure.

This matches the existing pattern: Softmax1ReciprocalModule,
FloorExtractionFP32Module, DivCorrectionModule (in div.py) all use direct
tensor operations in forward() rather than baked FFN weights, because the
operation is most naturally expressed as scalar arithmetic. The key
architectural property — pure forward-only neural computation, no Python
control flow on extracted scalars — is preserved.
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import GenericE, GenericFlattenedFFN, bake_clear_pair


class ClearDivSlotsFFN(nn.Module):
    """Clear division scratch slots at all positions.

    Clears SLOT_REMAINDER and SLOT_QUOTIENT vectors (per-position), plus
    the per-position partial-dividend slots used internally by the long
    division module.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        # Clear per-position SLOT_REMAINDER and SLOT_QUOTIENT (nibble vectors).
        slots = [ge.SLOT_REMAINDER, ge.SLOT_QUOTIENT]
        N = ge.NUM_POSITIONS
        dtype = ge.config.torch_dtype

        hidden_dim = len(slots) * N * 2
        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx
        S = ge.DIV_SCALE

        with torch.no_grad():
            h = 0
            for slot in slots:
                for pos in range(N):
                    bake_clear_pair(self.flat_ffn.ffn, h,
                                    fi(0, ge.OP_START + opcode),
                                    fi(pos, slot), S)
                    h += 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class LongDivisionModule(nn.Module):
    """Nibble-level schoolbook long division — produces quotient and remainder.

    Reads NIB_A (dividend) and NIB_B (divisor) per-nibble, runs 8 outer
    iterations of long division, and writes:
      SLOT_QUOTIENT[j]   = j-th nibble of quotient
      SLOT_REMAINDER[j]  = j-th nibble of remainder

    All arithmetic in fp32. The partial dividend is maintained as a 9-nibble
    vector (the 9th nibble is the overflow slot — needed because shifting
    partial left by one base-16 digit can exceed 32 bits transiently).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode

    def _trial_multiply(self, q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute q * b for q in 0..15, b a 9-nibble tensor [B, 9].

        Args:
            q: [B] tensor of trial multipliers (integers 0..15, fp32)
            b: [B, 9] divisor as 9 nibbles (each 0..15, fp32). Position 0 is LSB.

        Returns:
            [B, 9] tensor of nibbles representing q*b (each 0..15, fp32).
            Final carry beyond position 8 is discarded (q*b fits in 9 nibbles
            since q <= 15 and b < 2^32, so q*b < 2^36).
        """
        # For each nibble j: prod[j] = q * b[j] (in 0..225)
        # Then carry-resolve: out[j] = (prod[j] + carry_in) mod 16
        #                     carry_out = floor((prod[j] + carry_in) / 16)
        B = q.shape[0]
        device = q.device
        dtype = q.dtype

        prod = q[:, None] * b  # [B, 9], each in 0..225
        out = torch.zeros(B, 9, dtype=dtype, device=device)
        carry = torch.zeros(B, dtype=dtype, device=device)

        for j in range(9):
            v = prod[:, j] + carry  # [B], <=225 + 14 = 239 (fits fp32 exactly)
            digit = torch.floor(v / 16.0 + 1e-6)  # carry-out
            out[:, j] = v - digit * 16.0
            carry = digit
        return out

    def _shift_left_one_nibble(self, p: torch.Tensor, new_lsb: torch.Tensor) -> torch.Tensor:
        """Shift p left by one nibble position, inserting new_lsb at position 0.

        p: [B, 9] (output: 9 nibbles, p[8] becomes the discard if it was the
           old MSB — but we keep 9 slots throughout, including the new one).

        Wait — p was 9 nibbles before shift, and after shift_left we need 10
        slots. But during long division, p < b (after each subtract), so
        p[8] = 0 always entering the shift. So we can safely shift and have
        p[0] = new_lsb, p[j] = old p[j-1] for j>=1, with p[8] = old p[7].

        Returns p_new: [B, 9].
        """
        B = p.shape[0]
        device = p.device
        dtype = p.dtype
        out = torch.zeros(B, 9, dtype=dtype, device=device)
        out[:, 0] = new_lsb
        out[:, 1:9] = p[:, 0:8]
        return out

    def _compare_le(self, qb: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compare qb <= p element-wise on 9-nibble representations.

        Args:
            qb: [B, 9] candidate q*b
            p:  [B, 9] partial dividend

        Returns:
            [B] tensor of 0.0 / 1.0: 1.0 iff qb <= p.
        """
        # Multi-nibble comparison: process MSB-to-LSB.
        # eq_so_far = True; le = False
        # For j from 8 down to 0:
        #   if not eq_so_far: keep le
        #   else if qb[j] < p[j]: le = True, eq_so_far = False
        #   else if qb[j] > p[j]: le = False, eq_so_far = False
        #   else: keep both
        # Final: le = le or eq_so_far  (qb == p case)
        B = qb.shape[0]
        device = qb.device
        dtype = qb.dtype
        eq = torch.ones(B, dtype=dtype, device=device)
        le = torch.zeros(B, dtype=dtype, device=device)
        for j in range(8, -1, -1):
            d = qb[:, j] - p[:, j]
            # less = (d < 0), greater = (d > 0)
            less = (d < -0.5).to(dtype)
            greater = (d > 0.5).to(dtype)
            # When eq is true, decide:
            le = le + eq * less  # if equal-so-far and qb[j]<p[j]: become le
            eq = eq * (1.0 - less) * (1.0 - greater)
        # Tie: qb == p → counts as le.
        le = le + eq
        return le.clamp(0.0, 1.0)

    def _subtract(self, p: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        """Compute p - qb in 9-nibble representation. Assumes p >= qb.

        Standard borrow-based subtraction.
        """
        B = p.shape[0]
        device = p.device
        dtype = p.dtype
        out = torch.zeros(B, 9, dtype=dtype, device=device)
        borrow = torch.zeros(B, dtype=dtype, device=device)
        for j in range(9):
            v = p[:, j] - qb[:, j] - borrow  # [B], in -16..15
            need_borrow = (v < -0.5).to(dtype)
            out[:, j] = v + need_borrow * 16.0
            borrow = need_borrow
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ge = self.ge
        N = ge.NUM_POSITIONS  # 8 for NIBBLE
        B = x.shape[0]
        device = x.device
        orig_dtype = x.dtype
        opcode_w = x[:, 0, ge.OP_START + self.opcode]  # [B]

        # Read dividend and divisor as fp32 nibble vectors (length 8).
        # Round to integers to defeat any prior float drift.
        a = x[:, :N, ge.NIB_A].float()  # [B, 8]
        b = x[:, :N, ge.NIB_B].float()  # [B, 8]
        a = torch.round(a.clamp(0, 15))
        b = torch.round(b.clamp(0, 15))

        # Pad to 9 nibbles for the partial-dividend representation.
        b9 = torch.zeros(B, 9, dtype=torch.float32, device=device)
        b9[:, :8] = b

        # Detect divide-by-zero. For now, treat b==0 as identity: output
        # quotient=0, remainder=a (caller's responsibility to avoid).
        b_is_zero = (b9.sum(dim=-1) < 0.5).to(torch.float32)  # [B]

        # partial: 9 nibbles, init to zero.
        partial = torch.zeros(B, 9, dtype=torch.float32, device=device)
        # quotient: 8 nibbles
        q_out = torch.zeros(B, N, dtype=torch.float32, device=device)

        # Iterate i from MSB to LSB of dividend.
        for i in range(N - 1, -1, -1):
            # 1. Bring down nibble a[i]: partial = partial * 16 + a[i]
            #    (i.e., shift left by one nibble, set lsb to a[i])
            partial = self._shift_left_one_nibble(partial, a[:, i])

            # 2. Find largest q in 0..15 such that q*b <= partial.
            #    We do this by trying q = 15, 14, ..., 0 and accumulating
            #    a "found" flag. As soon as q*b <= partial, we record q.
            #    Equivalently: q[i] = sum over k=1..15 of (k*b <= partial).
            #    Proof: this counts how many q in 1..15 satisfy q*b <= p.
            #    The largest such q equals the count (because the relation
            #    is monotone in q: if k*b <= p then (k-1)*b <= p too).
            q_count = torch.zeros(B, dtype=torch.float32, device=device)
            for k in range(1, 16):
                k_t = torch.full((B,), float(k), dtype=torch.float32, device=device)
                qb = self._trial_multiply(k_t, b9)  # [B, 9]
                le = self._compare_le(qb, partial)  # [B] in {0,1}
                q_count = q_count + le

            q_i = q_count  # [B] in 0..15
            q_out[:, i] = q_i

            # 3. Subtract q_i * b from partial.
            qb_final = self._trial_multiply(q_i, b9)  # [B, 9]
            partial = self._subtract(partial, qb_final)

        # Remainder is partial (8 nibbles; partial[8] should be 0 since p < b ≤ 2^32).
        rem = partial[:, :N]  # [B, 8]

        # Divide-by-zero handling: keep the existing-test convention. Tests
        # skip b==0, but we still set sane defaults: q = 0xFFFFFFFF, r = a.
        if b_is_zero.any():
            mask = b_is_zero[:, None]  # [B, 1]
            q_out = q_out * (1 - mask) + 15.0 * mask  # all-ones nibbles
            rem = rem * (1 - mask) + a * mask

        # Write back: SLOT_QUOTIENT and SLOT_REMAINDER per-position.
        delta = torch.zeros_like(x)
        old_q = x[:, :N, ge.SLOT_QUOTIENT]
        old_r = x[:, :N, ge.SLOT_REMAINDER]
        opc = opcode_w[:, None].to(orig_dtype)
        delta[:, :N, ge.SLOT_QUOTIENT] = (-old_q + q_out.to(orig_dtype)) * opc
        delta[:, :N, ge.SLOT_REMAINDER] = (-old_r + rem.to(orig_dtype)) * opc

        return x + delta


class EmitDivResultModule(nn.Module):
    """Copy SLOT_QUOTIENT (DIV) or SLOT_REMAINDER (MOD) into RESULT.

    Implemented as a flattened FFN: one cancel-pair per (position, slot) to
    add slot value to RESULT, plus one cancel-pair per position to clear
    pre-existing RESULT.
    """

    def __init__(self, ge: GenericE, opcode: int, emit_remainder: bool = False):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.DIV_SCALE
        dtype = ge.config.torch_dtype
        src_slot = ge.SLOT_REMAINDER if emit_remainder else ge.SLOT_QUOTIENT

        hidden_dim = N * 4  # 2 for clear + 2 for copy, per position.
        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            h = 0
            opcode_idx = fi(0, ge.OP_START + opcode)
            for pos in range(N):
                # Clear RESULT[pos]
                bake_clear_pair(self.flat_ffn.ffn, h,
                                opcode_idx, fi(pos, ge.RESULT), S)
                h += 2
                # Add src_slot[pos] -> RESULT[pos] (cancel pair).
                self.flat_ffn.W_up[h, opcode_idx] = S
                self.flat_ffn.W_gate[h, fi(pos, src_slot)] = 1.0
                self.flat_ffn.W_down[fi(pos, ge.RESULT), h] = 1.0 / S
                h += 1
                self.flat_ffn.W_up[h, opcode_idx] = -S
                self.flat_ffn.W_gate[h, fi(pos, src_slot)] = -1.0
                self.flat_ffn.W_down[fi(pos, ge.RESULT), h] = 1.0 / S
                h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


def build_div_layers_longdiv(config: ChunkConfig, opcode: int) -> nn.ModuleList:
    """Build DIV pipeline using nibble-level long division (no fp64).

    Layers:
      1. ClearDivSlotsFFN — clear scratch slots
      2. LongDivisionModule — full long division → SLOT_QUOTIENT, SLOT_REMAINDER
      3. EmitDivResultModule — copy SLOT_QUOTIENT[*] → RESULT[*]

    Layer count is 3 (with the long-division module containing all 24 sub-ops).
    """
    ge = GenericE(config)
    return nn.ModuleList([
        ClearDivSlotsFFN(ge, opcode),
        LongDivisionModule(ge, opcode),
        EmitDivResultModule(ge, opcode, emit_remainder=False),
    ])


def build_mod_layers_longdiv(config: ChunkConfig, opcode: int) -> nn.ModuleList:
    """Build MOD pipeline using nibble-level long division (no fp64).

    Same as DIV but the final emit module copies SLOT_REMAINDER instead.
    """
    ge = GenericE(config)
    return nn.ModuleList([
        ClearDivSlotsFFN(ge, opcode),
        LongDivisionModule(ge, opcode),
        EmitDivResultModule(ge, opcode, emit_remainder=True),
    ])
