"""native_fixed_mac_discrete.py — a GENUINELY-VANILLA + FULL-PRECISION (byte-exact)
fixed-point MAC through the DISCRETE-TOKEN round-trip on a stock Qwen2ForCausalLM.

The gap this closes
===================
``native_fp32_baked.py`` (``C4_FP32_ALU``) bakes ``FMUL`` = silu-gadget and
``FADD`` = residual-add, and an fp32 MAC runs through ``model.forward`` — but it
rides CONTINUOUS fp32 SCALARS in residual dims (one dim per value) and reads the
answer straight off the ACC dim.  That is **value-faithful** (~1e-7), NOT the
project's real vanilla-ness (the DISCRETE-TOKEN round-trip: value in as tokens ->
embed -> compute -> **re-quantise to discrete tokens via lm_head argmax** each step,
exactly like ``qwen_vanilla_vm``'s register emission).

This module is the byte-exact, discrete-token realisation of a MAC:

    a, b (fixed-point) enter as 8 DISCRETE NIBBLE TOKENS each
      -> embed_tokens (the REAL embedding — nibble token n embeds value n on CUR_NIB)
      -> vanilla Qwen2 layers (SwiGLU FFN + attention) compute the fixed-point
         product P = (A*B) >> Q  and the running sum  ACC += P, with the SAME
         byte-exact nibble gadgets qwen_vanilla_vm / nibble_alu32 use
      -> lm_head ARGMAX RE-QUANT: each result nibble is emitted as a discrete
         nibble token via  argmax_n(2*n*x - n^2) == round(x)  (the vanilla
         requantiser), so the answer survives  embed -> compute -> lm_head argmax
      -> the emitted nibble tokens are decoded back to the fixed-point integer.

The answer is NEVER read continuously off a residual dim; it round-trips through
the real ``lm_head`` argmax as discrete tokens, and is byte-exact vs numpy
fixed-point.

Why fixed-point, not IEEE-fp32 (the representation verdict)
===========================================================
* **Genuine byte-exact IEEE-fp32 multiply is NOT achievable** through a silu gadget
  (or through ANY discrete requant that isn't full IEEE-754 round-to-nearest-even
  bit surgery).  ``a*b`` in fp32 ROUNDS the 48-bit mantissa product to 24 bits with
  round-half-to-even; ``native_fp32_baked``'s ``signed_silu_mul`` reproduces that to
  the fp32 epsilon FLOOR (~1e-7 relative), not bit-for-bit, and re-quantising a
  *continuous* fp32 result to discrete tokens would require extracting sign/exp/
  mantissa bits (IEEE bit surgery), not a silu op.  So "full-precision byte-exact"
  == **32-bit fixed-point**, the representation the emulated ``c4vm.onnx`` and the
  integer VM actually carry (nibbles are integers; they re-quant cleanly through
  ``argmax_n(2*n*x - n^2)`` because n in 0..15 is an integer).
* Fixed-point Q(I).(F) with I+F = 32 bits carried as 8 nibbles is byte-exact for
  every operand in range; MUL = ``(A*B) >> F`` (full 64-bit product, arithmetic
  shift), ADD = ``A + B`` (two's-complement, low 32 bits).  Both are integer nibble
  arithmetic -> re-quant EXACTLY.

Verdict, stated plainly: **byte-exact IEEE-fp32 is NOT achievable via the vanilla
silu round-trip; byte-exact 32-bit FIXED-POINT is, and it is what this delivers.**

Cost (the honest tax)
=====================
The MAC emits its result as W=8 discrete nibble tokens through the lm_head argmax
(the discrete round-trip), so a MAC costs the emission of the result frame — near
the nibble rate, NOT the continuous 4/MAC.  ``measure_native_fixed_mac`` reports the
forwards/MAC on the REAL model, head-to-head vs 4 (continuous) and 76-101 (integer
VM).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from . import blogspec_vocab as V
from . import nibble_alu32 as A
from .nibble_vm import _empty_spec
from .qwen_full_vm import (
    NORM_K,
    QWEN2_5_ARCH,
    QwenArch,
    _bake_ffn,
    _qwen_config,
    _rope_lane_pair,
    rmsnorm_identity_gamma,
)


# =========================================================================== #
# gate                                                                        #
# =========================================================================== #
def fixed_mac_discrete_enabled() -> bool:
    """``C4_FIXED_MAC_DISCRETE`` gate (default OFF).  This is a distinct precision
    MODE (byte-exact fixed-point via the discrete-token round-trip); when off,
    nothing here is on any build path, so the byte-exact integer VM golden and the
    ``C4_FP32_ALU`` fp32 mode are both unaffected.  A caller that wants the discrete
    fixed-point MAC passes ``force=True`` to the builders."""
    return os.environ.get("C4_FIXED_MAC_DISCRETE", "0") == "1"


# =========================================================================== #
# Q(I).(F) fixed-point reference                                                #
# =========================================================================== #
FRAC_BITS = 16            # Q16.16: 16 integer bits, 16 fraction bits (32-bit word)
WORD_BITS = 32
W_NIB = WORD_BITS // 4    # 8 nibbles carry the 32-bit fixed-point word
_MASK32 = (1 << WORD_BITS) - 1


def to_fixed(x: float, frac: int = FRAC_BITS) -> int:
    """Round a real ``x`` to the Q(32-frac).(frac) fixed-point 32-bit word (two's
    complement).  Byte-exact carrier: the value the nibble tokens encode."""
    return int(round(x * (1 << frac))) & _MASK32


def from_fixed(v: int, frac: int = FRAC_BITS) -> float:
    """Inverse of :func:`to_fixed`: the real value a fixed-point word denotes."""
    v &= _MASK32
    if v & (1 << (WORD_BITS - 1)):
        v -= (1 << WORD_BITS)
    return v / (1 << frac)


def fixed_mul(A_: int, B_: int, frac: int = FRAC_BITS) -> int:
    """Byte-exact fixed-point multiply, TRUNCATE-TOWARD-ZERO: ``sign(a*b) * (|a|*|b|
    >> frac)`` (low 32 bits).  This is the C / ISA integer-multiply convention (C4's
    MUL truncates toward zero, like ``nibble_muldivmod.divmod32``), and it is what the
    magnitude nibble schoolbook computes exactly — the gadget multiplies |a|,|b| and
    re-applies the sign, so gadget and reference agree byte-for-byte.

    (The alternative, an arithmetic-shift FLOOR ``(a*b) >> frac``, rounds toward -inf
    and differs by 1 LSB from truncate for a negative product with a nonzero shifted
    remainder; we use the truncate convention throughout so the byte-exact claim is
    unambiguous.)"""
    a = A_ - (1 << WORD_BITS) if A_ & (1 << (WORD_BITS - 1)) else A_
    b = B_ - (1 << WORD_BITS) if B_ & (1 << (WORD_BITS - 1)) else B_
    mag = (abs(a) * abs(b)) >> frac
    signed = -mag if (a < 0) != (b < 0) else mag
    return signed & _MASK32


def fixed_add(A_: int, B_: int) -> int:
    """Byte-exact fixed-point add ``A + B`` (two's complement, low 32 bits)."""
    return (A_ + B_) & _MASK32


def fixed_mac(a: float, b: float, acc0: float = 0.0, frac: int = FRAC_BITS) -> int:
    """Reference fixed-point MAC ``acc0 + a*b`` as a 32-bit fixed-point word."""
    return fixed_add(to_fixed(acc0, frac),
                     fixed_mul(to_fixed(a, frac), to_fixed(b, frac), frac))


def nibbles_of_word(v: int, w: int = W_NIB) -> List[int]:
    return [(v >> (4 * j)) & 0xF for j in range(w)]


def word_of_nibbles(nibs) -> int:
    v = 0
    for j, n in enumerate(nibs):
        v |= (int(n) & 0xF) << (4 * j)
    return v & _MASK32


# =========================================================================== #
# residual layout: INTEGER-nibble bands (0..15) so every value re-quants cleanly #
# =========================================================================== #
class FixedMacLayout:
    """Named INTEGER-nibble residual bands for the discrete-token fixed-point MAC.

      ONE            constant-1 lane (every live row; silu-gate / bias lane)
      CUR_NIB        the embedded token's nibble value (0..15) — REAL embedding
      A_NIB   [8]    operand a, deposited from its 8 nibble tokens
      B_NIB   [8]    operand b
      ACC_NIB [8]    incoming accumulator (the MAC's acc0)
      P_COL   [16]   product byte/carry-save scratch
      P_NIB   [16]   normalised product nibbles (low 16 nibbles of A*B) + byte scratch
      OUT_NIB [8]    the RESULT nibbles (ACC + (A*B>>frac)) — read by lm_head requant
      EMIT_VAL       the single nibble routed to the lm_head requant at an emit slot
      IS_A/IS_B/IS_ACC  role flags on the operand nibble tokens (broadcast keys)
      SLOT_ADDR[6]   per-token within-frame slot address (structural template)
      SLOT_OH [FL]   slot one-hot (emit-select + marker lm-head bias)
    """

    def __init__(self, frame_len: int, head_dim: int = 64):
        self.frame_len = frame_len
        self.head_dim = head_dim
        self._off = 0
        self._names: Dict[str, Tuple[int, int]] = {}
        self.ONE = self._scalar("ONE")
        self.CUR_NIB = self._scalar("CUR_NIB")
        # RD_SLOT: each operand nibble token deposits its value into a DISTINCT dim
        # (reg*W+j), so the read-CAM (one head per operand register) reconstructs the
        # whole band as a plain SUM over that register's W flagged nibble tokens.
        self.RD_SLOT = self._band("RD_SLOT", 3 * W_NIB)
        self.REG_OF_NIB = self._band("REG_OF_NIB", 3)   # content key per operand token
        self.A_NIB = self._band("A_NIB", W_NIB)
        self.B_NIB = self._band("B_NIB", W_NIB)
        self.ACC_NIB = self._band("ACC_NIB", W_NIB)
        # signed-multiply handling: MAG_A/MAG_B are the magnitudes |A|,|B| the nibble
        # schoolbook multiplies; SGN is the product sign (sA xor sB); the shifted
        # magnitude product is conditionally negated back to two's complement.
        self.MAG_A = self._band("MAG_A", W_NIB)
        self.MAG_B = self._band("MAG_B", W_NIB)
        self.SGN_A = self._scalar("SGN_A")      # 1 iff A < 0 (A_NIB[7] high bit set)
        self.SGN_B = self._scalar("SGN_B")
        self.SGN_P = self._scalar("SGN_P")      # product sign = SGN_A xor SGN_B
        # PP: the 64 raw nibble partial products a_i*b_j (i,j in 0..7); formed in one
        # block, then split+accumulated into the 16 product columns in the next (a
        # unit reads the BLOCK INPUT, so products must live in the residual first).
        self.PP = self._band("PP", W_NIB * W_NIB)
        self.P_COL = self._band("P_COL", 2 * W_NIB)
        self.P_NIB = self._band("P_NIB", 2 * W_NIB)
        self.SHIFTED = self._band("SHIFTED", W_NIB)   # |a|*|b| >> frac (magnitude, 8 nib)
        self.OUT_NIB = self._band("OUT_NIB", W_NIB)
        # MIRROR: the OUT_NIB band broadcast from the RES marker onto each RES-nibble
        # emit position (a SEPARATE landing band so the emit positions' own compute
        # cannot corrupt it; the emit-select reads MIRROR, not the local OUT_NIB).
        self.MIRROR = self._band("MIRROR", W_NIB)
        self.EMIT_VAL = self._scalar("EMIT_VAL")
        self.IS_STEP = self._scalar("IS_STEP")          # 1 on the RES-marker compute row
        self.IS_STEP_END_SRC = self._scalar("IS_STEP_END_SRC")  # OUT-broadcast source (marker)
        self.IS_RES_NIB = self._scalar("IS_RES_NIB")    # 1 on a RES-nibble emit position
        self.SLOT_BITS = 6
        self.SLOT_ADDR = self._band("SLOT_ADDR", self.SLOT_BITS)
        self.SLOT_OH = self._band("SLOT_OH", frame_len)
        self.D_used = self._off

    def _band(self, name, size):
        off = self._off
        self._names[name] = (off, size)
        self._off += size
        return off

    def _scalar(self, name):
        return self._band(name, 1)


# =========================================================================== #
# The DISCRETE FRAME the model emits per MAC step (all DISCRETE nibble tokens).  #
#                                                                               #
#   [A_MARK]   a0 a1 .. a7   (8 nibbles of operand a, little-endian)  INPUT      #
#   [B_MARK]   b0 b1 .. b7                                            INPUT      #
#   [ACC_MARK] c0 c1 .. c7   (8 nibbles of the incoming accumulator)  INPUT      #
#   [RES_MARK] r0 r1 .. r7   (8 RESULT nibbles the model EMITS via lm_head)  OUT #
#   [STEP_END]                                                                  #
# =========================================================================== #
A_MARK = V.REG_AX
B_MARK = V.REG_BP
ACC_MARK = V.REG_SP
RES_MARK = V.REG_PC
STEP_END = V.STEP_END
_ROLE_MARKS = (("A", A_MARK), ("B", B_MARK), ("ACC", ACC_MARK))


def _frame_plan() -> List[Tuple[str, object]]:
    plan: List[Tuple[str, object]] = []
    for role, mk in _ROLE_MARKS:
        plan.append(("marker", (role, mk)))
        for j in range(W_NIB):
            plan.append(("in_nib", (role, j)))
    plan.append(("marker", ("RES", RES_MARK)))
    for j in range(W_NIB):
        plan.append(("out_nib", ("RES", j)))
    plan.append(("end", ("END", STEP_END)))
    return plan


FRAME_PLAN = _frame_plan()
FRAME_LEN = len(FRAME_PLAN)          # 4*(1+8) + 1 = 37
RES_MARK_SLOT = 3 * (1 + W_NIB)      # slot index of the RES marker (= 27)
RES_NIB0_SLOT = RES_MARK_SLOT + 1    # slot index of RES nibble 0 (= 28)


def _nibble_token(n: int) -> int:
    return n & 0xF


# =========================================================================== #
# FFN COMPUTE blocks (vanilla SwiGLU) — byte-exact nibble MUL + shift + ADD.     #
# =========================================================================== #
# nibble-schoolbook: 16 nibble columns cover the full 64-bit product A*B.  Every
# column keeps a raw sum < 256 (kmax=15 everywhere) — the SAME fp-discipline
# nibble_alu32 uses (byte schoolbook would push a column to ~260100 where RELU_S*x
# overflows fp32 integer exactness).
_N_NIB = 2 * W_NIB       # 16 product nibble columns


# --------------------------------------------------------------------------- #
# signed multiply: |A|,|B| magnitude + product sign (the divmod32 sign trick).   #
# The nibble schoolbook multiplies UNSIGNED magnitudes; the two's-complement       #
# arithmetic-shift result of a SIGNED product is  sign(a*b) * (|a|*|b| >> frac),   #
# which the magnitude path + a conditional final negate reproduce byte-exactly.    #
# --------------------------------------------------------------------------- #
def compile_sign_detect(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """SGN_A = [A_NIB[7] >= 8] (high bit of the top nibble), SGN_B similarly."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, 4 + 4)
    u = 0
    u = A._clear(spec, u, L.SGN_A)
    u = A._step_ge(spec, u, {L.A_NIB + W_NIB - 1: 1.0}, 0.0, 8, L.SGN_A, 1.0)
    u = A._clear(spec, u, L.SGN_B)
    u = A._step_ge(spec, u, {L.B_NIB + W_NIB - 1: 1.0}, 0.0, 8, L.SGN_B, 1.0)
    return A._truncate(spec, u, dim)


def compile_sign_xor(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """SGN_P = SGN_A xor SGN_B = [SGN_A + SGN_B == 1] = [>=1] - [>=2].  A SEPARATE
    layer (it reads SGN_A/SGN_B computed by the prior block)."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, 6)
    u = 0
    u = A._clear(spec, u, L.SGN_P)
    u = A._step_ge(spec, u, {L.SGN_A: 1.0, L.SGN_B: 1.0}, 0.0, 1, L.SGN_P, 1.0)
    u = A._step_ge(spec, u, {L.SGN_A: 1.0, L.SGN_B: 1.0}, 0.0, 2, L.SGN_P, -1.0)
    return A._truncate(spec, u, dim)


def _compile_magnitude(L: FixedMacLayout, dim: int, src: int, sgn: int, mag: int,
                       n: int) -> List[Dict[str, torch.Tensor]]:
    """Conditionally two's-complement negate ``src`` (n nibbles) into ``mag`` when
    ``sgn``=1, else copy: ``mag = src`` if sgn==0 else ``(~src)+1``.  Returns a LIST of
    FFN specs (needs a carry ripple for the +1, each round a separate block).

    Block A: mag[j] = src[j] + sgn*(15 - 2*src[j])  = src[j] if sgn==0 else 15-src[j],
             plus +1 into mag[0] gated on sgn (the two's-complement +1).
    Blocks B..: base-16 carry ripple to settle mag (the +1 can carry)."""
    specs = []
    A._ONE = L.ONE
    # block A: conditional invert + the +1.
    spec = _empty_spec(dim, n * 4 + 4)
    u = 0
    for j in range(n):
        u = A._clear(spec, u, mag + j)
        u = A._ident(spec, u, {src + j: 1.0}, 0.0, mag + j, 1.0)                # + src
        # + sgn*(15 - 2*src[j]) via the AND-gate: guard on sgn, value = 15 - 2*src.
        u = A._guard(spec, u, [(sgn, 1.0, 0.0)], {L.ONE: 15.0, src + j: -2.0}, 0.0, mag + j, 1.0)
    u = A._guard(spec, u, [(sgn, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, mag + 0, 1.0)    # +1 (two's compl)
    specs.append(A._truncate(spec, u, dim))
    # carry ripple (the +1 and the 15-nibbles keep columns < ~31, kmax=15 safe).
    for _ in range(n):
        spec = _empty_spec(dim, n * (2 + 15 * 2 + 15 * 2 + 2) + 2)
        u = A._nibble_carry_round(spec, 0, mag, mag, n)
        specs.append(A._truncate(spec, u, dim))
    return specs


def compile_products(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Form the 64 raw nibble partial products ``PP[i*8+j] = A_NIB[i]*B_NIB[j]``
    (0..225) via the 6-weight silu-gated multiply.  A SEPARATE block splits+accumulates
    them (a unit reads the BLOCK INPUT, so the products must live in the residual
    first)."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, W_NIB * W_NIB * 3 + 2)   # per pair: 1 clear + 2 mul units
    u = 0
    for i in range(W_NIB):
        for j in range(W_NIB):
            idx = i * W_NIB + j
            u = A._clear(spec, u, L.PP + idx)
            u = A._mul_gate(spec, u, L.MAG_A + i, L.MAG_B + j, L.PP + idx, 1.0)   # |a|_i*|b|_j
    return A._truncate(spec, u, dim)


def compile_split_accumulate(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Split each partial product ``p = PP[i*8+j]`` (0..225, now in the residual) into
    ``(p mod 16) + 16*(p>>4)``: low nibble -> product column ``i+j``, high nibble ->
    column ``i+j+1`` (one shared floor(p/16) staircase, kmax=15).  Every column
    receives <= 8 low + 8 high nibbles = < 256, so the downstream carry uses kmax=15."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, _N_NIB + W_NIB * W_NIB * (1 + 15 * 2) + 2)
    u = 0
    for c in range(_N_NIB):
        u = A._clear(spec, u, L.P_COL + c)
    for i in range(W_NIB):
        for j in range(W_NIB):
            c = i + j
            if c >= _N_NIB:
                continue
            pp = L.PP + i * W_NIB + j
            u = A._ident(spec, u, {pp: 1.0}, 0.0, L.P_COL + c, 1.0)          # + p into col c
            if c + 1 < _N_NIB:
                u = A._floor_div_pow2(spec, u, {pp: 1.0}, 0.0, 16, 15,
                                      L.P_COL + c, -16.0, L.P_COL + c + 1, 1.0)
            else:
                u = A._floor_div_pow(spec, u, {pp: 1.0}, 0.0, 16, 15, L.P_COL + c, -16.0)
    return A._truncate(spec, u, dim)


def compile_product_carry_copy(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Copy the 16 product columns P_COL -> P_NIB (the ripple works in place on
    P_NIB, a SEPARATE band, so the columns stay available for debugging)."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, _N_NIB * 2 + 2)
    u = 0
    for c in range(_N_NIB):
        u = A._clear(spec, u, L.P_NIB + c)
        u = A._ident(spec, u, {L.P_COL + c: 1.0}, 0.0, L.P_NIB + c, 1.0)
    return A._truncate(spec, u, dim)


def compile_carry_round(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """ONE base-16 carry-normalise round on the 16 P_NIB columns (each < 256), kmax=15
    — a SEPARATE FFN layer (a carry round must read the PRIOR round's settled columns,
    so each round is its own block).  Reuses the library ``_nibble_carry_round``."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, _N_NIB * (2 + 15 * 2 + 15 * 2 + 2) + 2)
    u = A._nibble_carry_round(spec, 0, L.P_NIB, L.P_NIB, _N_NIB)
    return A._truncate(spec, u, dim)


def compile_shift(L: FixedMacLayout, dim: int, frac: int) -> Dict[str, torch.Tensor]:
    """Fixed-point shift of the MAGNITUDE product ``|a|*|b| >> frac``: drop the low
    ``frac/4`` settled product nibbles -> SHIFTED[j] = P_NIB[j+shift] (8 nibbles)."""
    A._ONE = L.ONE
    shift_nibs = frac // 4
    spec = _empty_spec(dim, W_NIB * 2 + 2)
    u = 0
    for j in range(W_NIB):
        u = A._clear(spec, u, L.SHIFTED + j)
        if j + shift_nibs < _N_NIB:
            u = A._ident(spec, u, {L.P_NIB + j + shift_nibs: 1.0}, 0.0, L.SHIFTED + j, 1.0)
    return A._truncate(spec, u, dim)


def compile_signed_rawsum(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Apply the product SIGN and add the accumulator, RAW (per-nibble, carry settled
    downstream): the signed product in two's complement is ``SHIFTED`` if SGN_P==0
    else ``(~SHIFTED)+1``; then ``OUT_NIB = ACC_NIB + signed_product``.  Realised as
    ``OUT[j] = ACC[j] + SHIFTED[j] + SGN_P*(15 - 2*SHIFTED[j])`` (+ SGN_P into OUT[0]
    for the two's-complement +1).  Two's-complement add of two 32-bit words is a plain
    nibble add mod 2^32 — no sign bookkeeping beyond this."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, W_NIB * 4 + 4)
    u = 0
    for j in range(W_NIB):
        u = A._clear(spec, u, L.OUT_NIB + j)
        u = A._ident(spec, u, {L.ACC_NIB + j: 1.0, L.SHIFTED + j: 1.0}, 0.0, L.OUT_NIB + j, 1.0)
        u = A._guard(spec, u, [(L.SGN_P, 1.0, 0.0)], {L.ONE: 15.0, L.SHIFTED + j: -2.0}, 0.0,
                     L.OUT_NIB + j, 1.0)
    u = A._guard(spec, u, [(L.SGN_P, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, L.OUT_NIB + 0, 1.0)
    return A._truncate(spec, u, dim)


def compile_add_carry_round(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """ONE base-16 carry-normalise round on the 8 OUT_NIB columns (each < 31 initially),
    kmax=2.  A SEPARATE FFN layer (each round reads the prior settled columns)."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, W_NIB * (2 + 15 * 2 + 15 * 2 + 2) + 2)
    u = A._nibble_carry_round(spec, 0, L.OUT_NIB, L.OUT_NIB, W_NIB)
    return A._truncate(spec, u, dim)


# =========================================================================== #
# ingest + emit / requant scaffolding                                           #
# =========================================================================== #
def _slot_bit_windows(L: FixedMacLayout, slot: int):
    wins = []
    for b in range(L.SLOT_BITS):
        lane = L.SLOT_ADDR + b
        wins.append((lane, 1.0, 0.0) if (slot >> b) & 1 else (lane, -1.0, 1.0))
    return wins


def compile_ingest_operands(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Each INPUT operand nibble token deposits CUR_NIB into a DISTINCT RD_SLOT dim
    (reg*W+j) gated on its within-frame slot address, and lights REG_OF_NIB[reg] (the
    read-CAM content key).  The read-CAM (next layer) sums each register's W flagged
    tokens into its A_NIB/B_NIB/ACC_NIB band."""
    A._ONE = L.ONE
    spec = _empty_spec(dim, 3 * W_NIB * 3 + 4)
    u = 0
    for ri in range(3):
        for j in range(W_NIB):
            slot = 1 + ri * (1 + W_NIB) + j
            wins = _slot_bit_windows(L, slot)
            u = A._guard(spec, u, wins, {L.CUR_NIB: 1.0}, 0.0, L.RD_SLOT + ri * W_NIB + j, 1.0)
            u = A._guard(spec, u, wins, {L.ONE: 1.0}, 0.0, L.REG_OF_NIB + ri, 1.0)
    return A._truncate(spec, u, dim)


def _bake_read_cam(attn, L: FixedMacLayout, arch: QwenArch, G: float = 12.0):
    """One head per operand register (A/B/ACC): at the RES-marker compute row
    (IS_STEP=1), head r attends UNIFORMLY to the tokens whose REG_OF_NIB[r]=1 (its W
    nibble tokens), each of which deposited its nibble at RD_SLOT[r*W+j]; the value
    copy sums them (softmax weight ~1/W) into the register's nibble band, scaled by W.
    The SAME positional read-CAM ``qwen_vanilla_vm._bake_read_cam`` uses."""
    hd = arch.head_dim
    slow_lo, _ = _rope_lane_pair(hd, slow=True)
    q_w = attn.q_proj.weight; k_w = attn.k_proj.weight
    v_w = attn.v_proj.weight; o_w = attn.o_proj.weight
    reg_bases = [L.A_NIB, L.B_NIB, L.ACC_NIB]
    for r in range(3):
        base = r * hd
        c_lane = slow_lo - r
        q_w[base + c_lane, L.IS_STEP] = G           # query: the compute row
        k_w[c_lane, L.REG_OF_NIB + r] = G           # key: this register's nibble tokens
    assert hd >= 3 * W_NIB, (3 * W_NIB, hd)
    for i in range(3 * W_NIB):                       # shared V: copy RD_SLOT once
        v_w[i, L.RD_SLOT + i] = 1.0
    for r in range(3):
        base = r * hd
        for j in range(W_NIB):
            o_w[reg_bases[r] + j, base + r * W_NIB + j] = float(W_NIB)


def compile_slot_onehot(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    A._ONE = L.ONE
    spec = _empty_spec(dim, L.frame_len + 2)
    u = 0
    for s in range(L.frame_len):
        u = A._guard(spec, u, _slot_bit_windows(L, s), {L.ONE: 1.0}, 0.0, L.SLOT_OH + s, 1.0)
    return A._truncate(spec, u, dim)


def compile_emit_select(L: FixedMacLayout, dim: int) -> Dict[str, torch.Tensor]:
    """At slot ``s`` the hidden predicts slot ``s+1``.  If slot ``s+1`` emits RES
    nibble j, route MIRROR[j] -> EMIT_VAL (gated SLOT_OH[s]); the lm_head then argmaxes
    ``2*n*EMIT_VAL - n^2`` == round(MIRROR[j]).  MIRROR carries the settled OUT_NIB
    broadcast from the RES-marker compute row onto every position, so the emit at the
    RES-marker AND at each RES-nibble position reads a CLEAN result nibble (never the
    local compute, which only settled on the marker row)."""
    A._ONE = L.ONE
    FL = L.frame_len
    spec = _empty_spec(dim, FL + 4)
    u = 0
    u = A._clear(spec, u, L.EMIT_VAL)
    for s in range(FL):
        kind, payload = FRAME_PLAN[(s + 1) % FL]
        if kind == "out_nib":
            _role, j = payload
            u = A._guard(spec, u, [(L.SLOT_OH + s, 1.0, 0.0)], {L.MIRROR + j: 1.0}, 0.0,
                         L.EMIT_VAL, 1.0)
    return A._truncate(spec, u, dim)


# =========================================================================== #
# attention: broadcast a settled band forward (recency) — vanilla softmax head.  #
# =========================================================================== #
def _bake_broadcast_head(attn, L: FixedMacLayout, arch: QwenArch, src_off: int,
                         dst_off: int, n: int, flag_dim: int, head: int = 0,
                         G: float = 14.0):
    """One attention head: copy an n-dim band from the token whose ``flag_dim``=1
    onto later positions (content match on a slow RoPE lane, softmax over the flagged
    tokens).  Vanilla Qwen2 attention (RoPE + softmax)."""
    hd = arch.head_dim
    slow_lo, _ = _rope_lane_pair(hd, slow=True)
    base = head * hd
    attn.q_proj.weight[base + slow_lo, L.ONE] = G
    attn.k_proj.weight[slow_lo, flag_dim] = G
    assert n <= hd, (n, hd)
    for i in range(n):
        attn.v_proj.weight[i, src_off + i] = 1.0
        attn.o_proj.weight[dst_off + i, base + i] = 1.0


# =========================================================================== #
# lm_head: requant EMIT_VAL -> nibble token argmax; markers via SLOT_OH bias.    #
# =========================================================================== #
def _bake_lm_head(model, L: FixedMacLayout):
    lm = model.lm_head.weight
    lm.zero_()
    if model.lm_head.bias is not None:
        model.lm_head.bias.zero_()
    for n in range(16):                       # nibble requant: argmax_n(2n*EMIT - n^2)
        lm[n, L.EMIT_VAL] = 2.0 * n
        lm[n, L.ONE] = -(n * n)
    BIG = 5000.0
    FL = L.frame_len
    for s in range(FL):
        kind, payload = FRAME_PLAN[(s + 1) % FL]
        if kind in ("marker", "end"):
            _role, tok = payload
            lm[tok, L.SLOT_OH + s] += BIG


def _address_bits(v: int, n: int):
    return [float((v >> b) & 1) for b in range(n)]


# =========================================================================== #
# build                                                                         #
# =========================================================================== #
@dataclass
class FixedMacModel:
    model: object
    L: FixedMacLayout
    hidden_size: int
    n_layers: int
    frac: int
    frame_len: int


def build(frac: int = FRAC_BITS, arch: QwenArch = QWEN2_5_ARCH, K: float = NORM_K,
          force: bool = False) -> FixedMacModel:
    """Build a stock ``Qwen2ForCausalLM`` whose layers compute the byte-exact
    fixed-point MAC and whose lm_head re-quantises the 8 result nibbles as discrete
    nibble tokens (the discrete-token round-trip)."""
    if not force and not fixed_mac_discrete_enabled():
        raise RuntimeError(
            "the discrete fixed-point MAC is a distinct precision MODE gated by "
            "C4_FIXED_MAC_DISCRETE (default OFF; integer golden + fp32 mode "
            "unaffected). Pass force=True for an explicit build.")
    from transformers.models.qwen2 import Qwen2ForCausalLM

    L = FixedMacLayout(FRAME_LEN, head_dim=arch.head_dim)
    dim = L.D_used + 1
    comp = L.D_used

    # LAYER pipeline (a Qwen layer is attn THEN mlp; attn sees only PRIOR FFN bands):
    #   0  ingest FFN                deposit each input nibble into RD_SLOT + REG_OF_NIB
    #   1  read-CAM (attn) + slot-oh gather A/B/ACC bands onto the compute row (RES mk)
    #   2  sign-detect FFN           SGN_A/SGN_B/SGN_P
    #   3.. magnitude(A) + carry     |A| into MAG_A (conditional two's-compl negate)
    #   ..  magnitude(B) + carry     |B| into MAG_B
    #   ..  products FFN             nibble schoolbook |a|_i*|b|_j -> 64 partials
    #   ..  split-accumulate FFN     -> 16 product columns (< 256)
    #   ..  carry-copy + R carry rounds  base-16 ripple normalise (kmax=15)
    #   ..  shift FFN                |a|*|b| >> frac -> SHIFTED (magnitude)
    #   ..  signed-rawsum FFN        ACC + sign(P)*SHIFTED (two's complement) -> OUT_NIB
    #   ..  N_ADD_CARRY add rounds   settle the add ripple
    #   ..  emit-broadcast (attn) + emit-select FFN  -> EMIT_VAL (lm_head requant)
    N_CARRY = _N_NIB          # 16 rounds fully settle a 16-column base-16 ripple
    ffn = [
        ("ingest", compile_ingest_operands(L, dim)),
        ("read-cam+slot-oh", compile_slot_onehot(L, dim)),   # attn = read-CAM
        ("sign-detect", compile_sign_detect(L, dim)),
        ("sign-xor", compile_sign_xor(L, dim)),
    ]
    for nm, mspecs in (("magA", _compile_magnitude(L, dim, L.A_NIB, L.SGN_A, L.MAG_A, W_NIB)),
                       ("magB", _compile_magnitude(L, dim, L.B_NIB, L.SGN_B, L.MAG_B, W_NIB))):
        ffn += [(f"{nm}-{i}", s) for i, s in enumerate(mspecs)]
    ffn += [
        ("products", compile_products(L, dim)),
        ("split-accumulate", compile_split_accumulate(L, dim)),
        ("carry-copy", compile_product_carry_copy(L, dim)),
    ]
    ffn += [(f"carry-round-{r}", compile_carry_round(L, dim)) for r in range(N_CARRY)]
    ffn += [("shift", compile_shift(L, dim, frac))]
    ffn += [("signed-rawsum", compile_signed_rawsum(L, dim))]
    N_ADD_CARRY = W_NIB       # 8 rounds settle the 8-column add ripple
    ffn += [(f"add-carry-{r}", compile_add_carry_round(L, dim)) for r in range(N_ADD_CARRY)]
    ffn += [("emit-select", compile_emit_select(L, dim))]    # attn = emit-broadcast
    read_cam_layer, emit_layer = 1, len(ffn) - 1
    n_layers = len(ffn)

    intermediate = max(int(s["W_up"].shape[0]) for _, s in ffn)
    intermediate = max(intermediate, arch.num_attention_heads * arch.head_dim, 8)
    hidden_size = arch.hidden_for(L.D_used + 1)

    cfg = _qwen_config(hidden_size, intermediate, n_layers, V.VOCAB, arch)
    model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
    qm = model.model

    embed = torch.zeros(V.VOCAB, hidden_size)
    embed[:, L.ONE] = 1.0
    for n in range(16):                       # nibble token n embeds value n
        embed[n, L.CUR_NIB] = float(n)
    embed[:, comp] = K
    embed[V.BOS, :] = 0.0
    embed[V.BOS, comp] = K

    with torch.no_grad():
        gamma = rmsnorm_identity_gamma(hidden_size, K)
        qm.norm.weight.copy_(gamma)
        qm.embed_tokens.weight.copy_(embed)
        for layer in qm.layers:
            layer.input_layernorm.weight.copy_(gamma)
            layer.post_attention_layernorm.weight.copy_(gamma)
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.o_proj):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                lin.weight.zero_()
        for i, (_, spec) in enumerate(ffn):
            _bake_ffn(qm.layers[i].mlp, spec, L, comp)
        # read-CAM: gather A/B/ACC nibble tokens onto the RES-marker compute row.
        _bake_read_cam(qm.layers[read_cam_layer].self_attn, L, arch)
        # emit-broadcast: carry the settled OUT_NIB from the RES-marker row (the OUT
        # broadcast SOURCE, the unique IS_STEP_END_SRC=1 token) into MIRROR on every
        # position.  Sharp content match (BOS is the softmax sink for the query rows
        # before the marker, but only the RES-marker + RES-nibble rows read MIRROR).
        _bake_broadcast_head(qm.layers[emit_layer].self_attn, L, arch,
                             src_off=L.OUT_NIB, dst_off=L.MIRROR, n=W_NIB,
                             flag_dim=L.IS_STEP_END_SRC, head=0, G=22.0)
        _bake_lm_head(model, L)

    return FixedMacModel(model=model, L=L, hidden_size=hidden_size, n_layers=n_layers,
                         frac=frac, frame_len=FRAME_LEN)


# =========================================================================== #
# The STRUCTURAL FRAME TEMPLATE (program-INDEPENDENT; carries ZERO computed state).#
#                                                                               #
# Each stream position gets a fixed per-slot structural tag (its SLOT_ADDR bits +#
# IS_STEP / IS_STEP_END_SRC flags).  This is the SAME for every MAC (a fixed     #
# 37-slot skeleton, like a chat template).  The COMPUTED result nibbles are       #
# emitted by the model's lm_head argmax and NEVER written here.                    #
# =========================================================================== #
def _slot_flags(L: FixedMacLayout, s: int) -> Dict[int, float]:
    """Structural flags for frame-slot ``s`` (-1 == BOS/no role).  The RES marker
    slot carries IS_STEP (read-CAM compute query) + IS_STEP_END_SRC (OUT broadcast
    source)."""
    flags: Dict[int, float] = {}
    if s < 0:
        return flags
    for b, bit in enumerate(_address_bits(s, L.SLOT_BITS)):
        if bit:
            flags[L.SLOT_ADDR + b] = 1.0
    if s == RES_MARK_SLOT:                       # the RES marker = the compute row
        flags[L.IS_STEP] = 1.0
        flags[L.IS_STEP_END_SRC] = 1.0
    return flags


def _frame_ids_for(a_word: int, b_word: int, acc_word: int) -> List[int]:
    """The INPUT token stream of one MAC frame (little-endian nibble tokens): the
    A/B/ACC operand nibbles are SEEDED (they are the input); the RES marker + RES
    nibbles + STEP_END are placeholders the model OVERWRITES via its lm_head argmax."""
    ids: List[int] = []
    for (role, mk), word in zip(_ROLE_MARKS, (a_word, b_word, acc_word)):
        ids.append(mk)
        for j in range(W_NIB):
            ids.append(_nibble_token((word >> (4 * j)) & 0xF))
    ids.append(RES_MARK)
    ids += [0] * W_NIB                          # placeholders (model emits RES nibbles)
    ids.append(STEP_END)
    assert len(ids) == FRAME_LEN
    return ids


def run_mac(fm: FixedMacModel, a: float, b: float, acc0: float = 0.0,
            use_kv_cache: bool = True):
    """Run ONE fixed-point MAC ``acc0 + a*b`` through the DISCRETE-TOKEN round-trip on
    the REAL ``model.forward`` and return ``(result_word, info)``.

    tokens-in: A/B/ACC operand nibbles are seeded discrete tokens; the model embeds
    them, computes the fixed-point product+accumulate through vanilla SwiGLU/attn
    layers, and RE-QUANTISES the 8 result nibbles via the real ``lm_head`` argmax
    (``argmax_n(2n*EMIT - n^2)``).  The emitted nibble tokens are decoded back to the
    fixed-point word — byte-exact vs :func:`fixed_mac`.  No continuous residual read
    of the answer; it survives embed -> compute -> lm_head argmax -> decode.
    """
    import torch as _t
    L = fm.L
    model = fm.model
    a_word = to_fixed(a, fm.frac)
    b_word = to_fixed(b, fm.frac)
    acc_word = to_fixed(acc0, fm.frac)
    ids = _frame_ids_for(a_word, b_word, acc_word)
    FL = fm.frame_len

    # prefix = [BOS] + the input operand region up to AND INCLUDING the RES marker
    # (slot RES_MARK_SLOT at stream position 1+RES_MARK_SLOT).  The 8 RES nibbles are
    # DECODED autoregressively: the hidden at the RES marker predicts RES nibble 0, and
    # each emitted nibble predicts the next.
    full_ids = [V.BOS] + ids
    prefix_ids = full_ids[:1 + RES_MARK_SLOT + 1]   # BOS + operands + RES marker
    n_forwards = 0

    def _row_overlay(pos: int):
        s = -1 if pos == 0 else (pos - 1) % FL
        ov = _t.zeros(1, 1, fm.hidden_size)
        for d, val in _slot_flags(L, s).items():
            ov[0, 0, d] = val
        return ov

    def _overlay(n_positions):
        ov = _t.zeros(1, n_positions, fm.hidden_size)
        for p in range(n_positions):
            s = -1 if p == 0 else (p - 1) % FL
            for d, val in _slot_flags(L, s).items():
                ov[0, p, d] = val
        return ov

    emitted: List[int] = []
    if use_kv_cache:
        from transformers import DynamicCache
        cache = DynamicCache()
        ctx = _t.tensor([prefix_ids])
        emb = model.model.embed_tokens(ctx) + _overlay(len(prefix_ids))
        pos = _t.arange(len(prefix_ids)).unsqueeze(0)
        with _t.no_grad():
            out = model(inputs_embeds=emb, past_key_values=cache, position_ids=pos,
                        use_cache=True)
        n_forwards += 1
        emitted.append(int(out.logits[0, -1].argmax()))     # RES nibble 0
        for k in range(1, W_NIB):
            # the token just emitted sits at stream position 1+RES_NIB0_SLOT+(k-1); feed
            # it (with that position's slot overlay) to predict RES nibble k.
            feed_pos = 1 + RES_NIB0_SLOT + (k - 1)
            row = model.model.embed_tokens(_t.tensor([[emitted[-1]]])) + _row_overlay(feed_pos)
            with _t.no_grad():
                out = model(inputs_embeds=row, past_key_values=cache,
                            position_ids=_t.tensor([[feed_pos]]), use_cache=True)
            n_forwards += 1
            emitted.append(int(out.logits[0, -1].argmax()))
    else:
        # FULL-RECOMPUTE: one forward per RES nibble over the growing window.
        cur = list(prefix_ids)
        for _k in range(W_NIB):
            emb = model.model.embed_tokens(_t.tensor([cur])) + _overlay(len(cur))
            with _t.no_grad():
                out = model(inputs_embeds=emb, use_cache=False)
            n_forwards += 1
            nxt = int(out.logits[0, -1].argmax())
            emitted.append(nxt)
            cur.append(nxt)

    result = word_of_nibbles(emitted)
    ref = fixed_mac(a, b, acc0, fm.frac)
    info = {
        "result_word": result, "ref_word": ref, "exact": result == ref,
        "result_value": from_fixed(result, fm.frac), "ref_value": from_fixed(ref, fm.frac),
        "emitted_nibbles": emitted, "forwards": n_forwards,
        "tokens_per_mac": W_NIB,                # discrete result tokens emitted / MAC
        "used_inputs_embeds_for_computed_value": False,  # answer is EMITTED, not written
        "reencoded_state": False,
    }
    return result, info


def run_dot(fm: FixedMacModel, a: List[float], b: List[float],
            use_kv_cache: bool = True):
    """A length-K fixed-point DOT ``sum_k a[k]*b[k]`` as a CHAIN of MAC steps through
    the discrete round-trip: each step's ACCUMULATOR is the PRIOR step's EMITTED result
    word (fed back in as discrete nibble tokens — the answer round-trips through
    ``lm_head`` argmax AND re-enters through ``embed_tokens``).  Byte-exact vs the
    sequential fixed-point reference.

    Returns ``(result_word, info)`` with per-step forwards and the running trace."""
    acc = 0
    ref = 0
    forwards = 0
    trace = []
    exact = True
    for k in range(len(a)):
        acc_val = from_fixed(acc, fm.frac)
        res, info = run_mac(fm, a[k], b[k], acc_val, use_kv_cache=use_kv_cache)
        acc = res                              # feed the EMITTED word as the next acc
        forwards += info["forwards"]
        ref = fixed_add(ref, fixed_mul(to_fixed(a[k], fm.frac), to_fixed(b[k], fm.frac), fm.frac))
        exact = exact and (acc == ref)
        trace.append(from_fixed(acc, fm.frac))
    return acc, {
        "result_word": acc, "ref_word": ref, "exact": exact,
        "result_value": from_fixed(acc, fm.frac), "ref_value": from_fixed(ref, fm.frac),
        "forwards": forwards, "K": len(a), "trace": trace,
        "forwards_per_mac": forwards / max(1, len(a)),
        "tokens_per_mac": W_NIB,
    }
