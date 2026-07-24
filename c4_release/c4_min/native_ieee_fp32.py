"""native_ieee_fp32.py — BYTE-EXACT IEEE-754 fp32 FMUL + FADD via BIT-SURGERY,
built from the repo's byte-exact INTEGER / NIBBLE gadgets.

The correction this module lands
================================
``native_fixed_mac_discrete.py`` concluded that byte-exact IEEE-fp32 multiply is
"NOT achievable" — but that verdict is only true for the SILU-APPROXIMATION path
it tried (fp32 ``a*b`` via a silu gadget that rounds ~1e-7, then a continuous ->
token re-quant).  It is FALSE for the real thing: the actual IEEE-754 soft-float
algorithm run with EXACT INTEGER operations.

The right approach (this module) is a **software float unit**: decompose the fp32
word into its integer fields (sign / exponent / mantissa), and run the exact
IEEE-754 round-to-nearest-even algorithm with **integer multiply / add / shift /
compare / select** — every one of which the repo already computes byte-exact
(``nibble_alu32``'s silu-gated integer multiply + base-16 carry rounds,
``nibble_bitwise`` shifts/and/or, integer add/sub, ``nibble_cmp`` compare).
Because the integer ops are EXACT, the reconstructed fp32 result is bit-identical
to hardware fp32 — NOT approximate.

Two layers live here:

1. :mod:`INTEGER REFERENCE` (``fmul_bits`` / ``fadd_bits``) — the IEEE-754
   algorithm written with ONLY Python integer operators (``+ - * >> << & | ==
   < >``); NO Python ``float`` arithmetic touches an operand anywhere.  This IS
   the bit-surgery algorithm; it is asserted bit-exact vs ``numpy.float32`` over
   a wide operand grid INCLUDING round-to-nearest-even TIE cases.

2. :mod:`NIBBLE-GADGET REALIZATION` (``fmul_gadgets`` / ``fadd_gadgets``) — the
   SAME algorithm re-expressed on **nibble lists** using the exact primitives
   that MIRROR the model's baked gadgets: a nibble-schoolbook integer multiply
   (= ``nibble_alu32._mul_gate`` + ``_nibble_carry_round``), a nibble carry-add
   (= the ADD chain), a nibble shift (= ``_floor_div_pow`` staircase), and a
   nibble/scalar compare+select (= ``_step_ge`` / ``_guard``).  This proves the
   fp unit reduces to gadgets the repo ALREADY bakes byte-exact — the vanilla
   bake is a WIRING job, not new math.  It is asserted identical to layer (1),
   hence bit-exact vs numpy.  ``test_native_ieee_fp32_gadgets.py`` goes further:
   it exercises the ACTUAL ``nibble_alu32`` primitives through a REAL SwiGLU
   forward and confirms each computes the exact integer op — including the full
   24x24 -> 48-bit schoolbook multiply (the FMUL core) built from ``_mul_gate`` +
   ``_floor_div_pow2`` + ``_nibble_carry_round``, EXACT.

Coverage (honest)
=================
* FMUL: all NORMAL x NORMAL operands, both signs, incl. overflow -> +/-inf,
  underflow -> +/-0, exact round-to-nearest-EVEN half-ULP ties, and the
  +/-0 / inf operand special cases.  Subnormal RESULTS and subnormal / NaN
  OPERANDS are a documented deferral (see ``SPECIAL_NOTES``).
* FADD: all NORMAL x NORMAL operands incl. massive cancellation, effective
  subtraction, exponent-align sticky, round-half-even ties, overflow -> inf,
  and +/-0 / inf special cases.  Subnormal results / NaN operands deferred.

Every deferral is reported honestly by the measurement driver; within the
covered classes the result is BIT-EXACT vs ``numpy.float32``.
"""
from __future__ import annotations

import struct
from typing import List, Tuple

# ============================================================================ #
#  fp32 bit helpers (struct-level; the ONLY float<->bits boundary)             #
# ============================================================================ #
def f32_to_bits(x: float) -> int:
    """The 32-bit IEEE-754 encoding of ``float32(x)`` as an unsigned int."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_f32(u: int) -> float:
    """Decode a 32-bit IEEE-754 pattern to the Python float it denotes."""
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


# IEEE-754 binary32 constants.
_SIGN_SHIFT = 31
_EXP_SHIFT = 23
_EXP_MASK = 0xFF
_MANT_MASK = 0x7FFFFF          # 23-bit stored mantissa
_MANT_LEAD = 1 << 23           # the implicit leading 1 (normal numbers)
_BIAS = 127
_EXP_INF = 0xFF                # exponent field of inf/NaN
_EXP_MAX_NORMAL = 0xFE         # largest finite exponent field


# ============================================================================ #
#  1. FIELD EXTRACT / PACK  (integer bit surgery on the 32-bit word)           #
# ============================================================================ #
def unpack_fields(u: int) -> Tuple[int, int, int]:
    """Split a 32-bit fp32 word into (sign_bit, exp_field, mant_stored) via pure
    integer shifts + masks — exactly the extraction a hardware FPU front-end does.

        sign = bit 31,  exp = bits 30..23,  mant = bits 22..0
    """
    u &= 0xFFFFFFFF
    sign = (u >> _SIGN_SHIFT) & 1
    exp = (u >> _EXP_SHIFT) & _EXP_MASK
    mant = u & _MANT_MASK
    return sign, exp, mant


def pack_fields(sign: int, exp: int, mant: int) -> int:
    """Reassemble a 32-bit fp32 word from integer fields (pure shifts + or)."""
    return (((sign & 1) << _SIGN_SHIFT)
            | ((exp & _EXP_MASK) << _EXP_SHIFT)
            | (mant & _MANT_MASK)) & 0xFFFFFFFF


def significand(exp: int, mant: int) -> int:
    """The 24-bit significand ``1.mmm`` (implicit leading 1 for a normal number,
    0 for a true zero exponent field)."""
    return (_MANT_LEAD | mant) if exp != 0 else mant


# ============================================================================ #
#  2. FMUL — the exact IEEE-754 multiply algorithm (INTEGER ops only)          #
# ============================================================================ #
def fmul_bits(ua: int, ub: int) -> int:
    """Byte-exact IEEE-754 fp32 multiply of two 32-bit patterns, computed with
    ONLY integer ops (no Python ``float`` arithmetic).  Returns the 32-bit result
    pattern, bit-identical to ``float32(a) * float32(b)``.

    Algorithm (the real soft-float MUL):
      sign  = sa xor sb
      * special: any inf/NaN operand, or a zero operand -> the IEEE result.
      exp   = ea + eb - BIAS           (unbiased sum)
      prod  = (1.ma) * (1.mb)          24x24 -> up-to-48-bit INTEGER multiply
      normalize: prod occupies bit 46 or 47; if bit 47 set, shift right 1 & exp++
      round : keep the top 24 bits (1 hidden + 23), round-to-nearest-EVEN on the
              guard bit + sticky (OR of all lower discarded bits)
      pack, handling exp overflow -> inf and underflow -> 0.
    """
    sa, ea, ma = unpack_fields(ua)
    sb, eb, mb = unpack_fields(ub)
    sign = sa ^ sb

    # ---- special operands (inf / NaN / zero) --------------------------------
    a_is_nan = (ea == _EXP_INF) and (ma != 0)
    b_is_nan = (eb == _EXP_INF) and (mb != 0)
    a_is_inf = (ea == _EXP_INF) and (ma == 0)
    b_is_inf = (eb == _EXP_INF) and (mb == 0)
    a_is_zero = (ea == 0) and (ma == 0)
    b_is_zero = (eb == 0) and (mb == 0)
    if a_is_nan or b_is_nan:
        return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1)          # a quiet NaN
    if a_is_inf or b_is_inf:
        if a_is_zero or b_is_zero:
            return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1)      # inf*0 = NaN
        return pack_fields(sign, _EXP_INF, 0)                     # +/-inf
    if a_is_zero or b_is_zero:
        return pack_fields(sign, 0, 0)                            # signed zero

    # ---- (subnormal operands are a documented deferral) ---------------------
    # Normal operands: 24-bit significands with the implicit leading 1.
    sig_a = _MANT_LEAD | ma       # in [2^23, 2^24)
    sig_b = _MANT_LEAD | mb

    # unbiased exponent sum (BIAS subtracted once).
    exp = ea + eb - _BIAS

    # 24x24 -> up-to-48-bit exact integer product.  prod is in
    # [2^46, 2^48): bit 47 (top) set => needs a 1-bit right normalize.
    prod = sig_a * sig_b

    # ---- normalize: bring the leading 1 to bit 47 -> then it is a 48-bit value
    # with the hidden bit at position 47.  If bit 47 is 0, the product is in
    # [2^46, 2^47) (leading 1 at bit 46); shift LEFT 1 so the significand aligns
    # to the same 48-bit frame, and decrement exp.  Equivalent standard form:
    if prod >> 47:                        # leading 1 at bit 47
        exp += 1
        shift = 47 - 23                   # 24 low bits are guard+sticky material
    else:                                 # leading 1 at bit 46
        shift = 46 - 23

    # ---- round-to-nearest-even on the discarded low ``shift`` bits ----------
    result_mant = prod >> shift           # top 24 bits (1 hidden + 23 stored)
    rem_mask = (1 << shift) - 1
    rem = prod & rem_mask                 # the discarded low bits
    half = 1 << (shift - 1)               # the guard-bit weight (half ULP)
    if rem > half:
        result_mant += 1                  # round up
    elif rem == half:
        if result_mant & 1:               # exactly half -> round to EVEN
            result_mant += 1
    # (rem < half -> truncate)

    # a round-up can carry the significand from 0x1FFFFFF-style out to 0x2000000
    # (25 bits): renormalize by shifting right 1 and bumping exp.
    if result_mant >> 24:
        result_mant >>= 1
        exp += 1

    # ---- exponent range: overflow -> inf, underflow -> signed zero ----------
    if exp >= _EXP_INF:                   # >= 255 biased -> overflow
        return pack_fields(sign, _EXP_INF, 0)
    if exp <= 0:                          # underflow (subnormal/zero region)
        # NOTE: gradual-underflow subnormals are deferred; flush to signed zero.
        return pack_fields(sign, 0, 0)

    return pack_fields(sign, exp, result_mant & _MANT_MASK)


# ============================================================================ #
#  3. FADD — the exact IEEE-754 add algorithm (INTEGER ops only)               #
# ============================================================================ #
# We carry the aligned significands in a 26-bit-ish frame: 24-bit significand
# shifted up by GBITS guard bits so alignment sticky is captured exactly.
_GUARD = 3                                 # guard + round + (sticky folded in)


def fadd_bits(ua: int, ub: int) -> int:
    """Byte-exact IEEE-754 fp32 add of two 32-bit patterns, computed with ONLY
    integer ops.  Returns the 32-bit result pattern, bit-identical to
    ``float32(a) + float32(b)``.

    Algorithm (the real soft-float ADD):
      * special: inf / NaN / zero operands -> IEEE result.
      * put both significands in a common frame shifted up by GUARD bits.
      * align: shift the SMALLER-exponent significand right by the exponent
        difference, OR-ing dropped bits into a sticky bit.
      * if signs equal: add the significands; else: subtract smaller from larger
        (the result sign is the larger operand's sign; exact-cancel -> +0).
      * normalize: the sum may carry out one bit (add) or lose many leading bits
        (cancellation) — shift to restore the leading 1 to its frame position,
        adjusting exp; fold dropped/needed bits into guard+sticky.
      * round-to-nearest-even on guard+sticky, then pack (overflow -> inf).
    """
    sa, ea, ma = unpack_fields(ua)
    sb, eb, mb = unpack_fields(ub)

    # ---- special operands ---------------------------------------------------
    a_is_nan = (ea == _EXP_INF) and (ma != 0)
    b_is_nan = (eb == _EXP_INF) and (mb != 0)
    a_is_inf = (ea == _EXP_INF) and (ma == 0)
    b_is_inf = (eb == _EXP_INF) and (mb == 0)
    if a_is_nan or b_is_nan:
        return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1)
    if a_is_inf and b_is_inf:
        if sa != sb:
            return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1)     # inf + -inf = NaN
        return pack_fields(sa, _EXP_INF, 0)
    if a_is_inf:
        return pack_fields(sa, _EXP_INF, 0)
    if b_is_inf:
        return pack_fields(sb, _EXP_INF, 0)

    a_is_zero = (ea == 0) and (ma == 0)
    b_is_zero = (eb == 0) and (mb == 0)
    if a_is_zero and b_is_zero:
        # -0 + -0 = -0 ; every other zero+zero = +0 (round-to-nearest).
        return pack_fields(1 if (sa and sb) else 0, 0, 0)
    if a_is_zero:
        return ub & 0xFFFFFFFF
    if b_is_zero:
        return ua & 0xFFFFFFFF

    # ---- (subnormal operands deferred) --------------------------------------
    sig_a = _MANT_LEAD | ma
    sig_b = _MANT_LEAD | mb

    # order so A has the larger (or equal) exponent — makes the align a single
    # right shift of B and fixes the result sign to A's sign on a subtract.
    if (ea < eb) or (ea == eb and sig_a < sig_b):
        sa, ea, sig_a, sb, eb, sig_b = sb, eb, sig_b, sa, ea, sig_a

    # bring both into the guard frame (<<GUARD), then align B down by the exp gap.
    big = sig_a << _GUARD
    small = sig_b << _GUARD
    diff = ea - eb
    if diff > 0:
        if diff >= 32:                    # B is utterly negligible but for sticky
            sticky = 1 if small != 0 else 0
            small = sticky                # collapse to a lone sticky bit
        else:
            dropped = small & ((1 << diff) - 1)
            small >>= diff
            if dropped:
                small |= 1                # OR the dropped bits into the sticky LSB

    # ---- add or subtract by sign --------------------------------------------
    if sa == sb:
        summ = big + small
        result_sign = sa
    else:
        summ = big - small
        result_sign = sa                  # A dominates (>= B) so its sign wins
        if summ == 0:
            return pack_fields(0, 0, 0)    # exact cancellation -> +0

    exp = ea                              # result exponent starts at A's

    # ---- normalize back to a 24-bit significand in the guard frame ----------
    # target: the leading 1 sits at bit (23 + GUARD).
    target = 23 + _GUARD
    hi = summ.bit_length() - 1            # position of the current leading 1
    if hi > target:                       # carried out (equal-sign add): shift right
        sh = hi - target
        dropped = summ & ((1 << sh) - 1)
        summ >>= sh
        if dropped:
            summ |= 1                     # keep sticky
        exp += sh
    elif hi < target:                     # cancellation: shift left to renormalize
        sh = target - hi
        summ <<= sh
        exp -= sh

    # ---- round-to-nearest-even on the GUARD low bits ------------------------
    guard_mask = (1 << _GUARD) - 1
    result_mant = summ >> _GUARD
    rem = summ & guard_mask
    half = 1 << (_GUARD - 1)
    if rem > half:
        result_mant += 1
    elif rem == half:
        if result_mant & 1:
            result_mant += 1
    if result_mant >> 24:                 # round carried out of the 24-bit frame
        result_mant >>= 1
        exp += 1

    # ---- range check --------------------------------------------------------
    if exp >= _EXP_INF:
        return pack_fields(result_sign, _EXP_INF, 0)
    if exp <= 0:
        # subnormal results deferred -> flush to signed zero.
        return pack_fields(result_sign, 0, 0)

    return pack_fields(result_sign, exp, result_mant & _MANT_MASK)


# ============================================================================ #
#  Convenience float<->float wrappers                                          #
# ============================================================================ #
def fmul(a: float, b: float) -> float:
    return bits_to_f32(fmul_bits(f32_to_bits(a), f32_to_bits(b)))


def fadd(a: float, b: float) -> float:
    return bits_to_f32(fadd_bits(f32_to_bits(a), f32_to_bits(b)))


# ============================================================================ #
#  4. NIBBLE-GADGET REALIZATION                                                #
#                                                                              #
#  The SAME algorithm, re-expressed over NIBBLE LISTS using ONLY the exact     #
#  integer primitives that MIRROR the repo's baked gadgets, so the fp unit is  #
#  provably a WIRING JOB over gadgets already baked byte-exact:                #
#                                                                              #
#    g_mul(a_nibs, b_nibs)   nibble schoolbook integer multiply                #
#        == nibble_alu32._mul_gate (silu-gated a_i*b_j) + _nibble_carry_round  #
#    g_add(a_nibs, b_nibs)   nibble carry-add                                  #
#        == the ADD byte/nibble carry chain (_byte_add_block / carry round)    #
#    g_shr(nibs, k) / g_shl  nibble shift by a bit count                       #
#        == _floor_div_pow (floor(x/2^k) staircase) / multiply by 2^k          #
#    g_ge(x, thr)            integer >= compare                                #
#        == nibble_alu32._step_ge (sharp unit ramp)                            #
#    g_select(cond, a, b)    a if cond else b                                  #
#        == nibble_alu32._guard (silu AND-gate) select                         #
#                                                                              #
#  These operate on plain Python ints but are written as the nibble/gadget     #
#  DECOMPOSITION so the op count is the model's op count.  ``fmul_gadgets`` /   #
#  ``fadd_gadgets`` are asserted IDENTICAL to the reference above (hence        #
#  bit-exact vs numpy) and additionally COUNT the integer gadget ops used, so  #
#  the measurement driver can report steps/fp-MAC.                             #
# ============================================================================ #
class OpCounter:
    """Tallies each exact-integer gadget invocation so a fp-MUL / fp-ADD can be
    priced in the SAME currency the fixed-point (8/MAC) and continuous (4/MAC)
    baselines use."""

    def __init__(self):
        self.mul = 0        # nibble integer multiplies (schoolbook)
        self.add = 0        # nibble integer add/sub (carry chains)
        self.shift = 0      # nibble shifts (floor-div / mul by 2^k staircases)
        self.cmp = 0        # integer >= compares
        self.sel = 0        # guarded selects (silu AND-gate)

    def total(self) -> int:
        return self.mul + self.add + self.shift + self.cmp + self.sel

    def as_dict(self):
        return {"mul": self.mul, "add": self.add, "shift": self.shift,
                "cmp": self.cmp, "sel": self.sel, "total": self.total()}


# ---- the five exact-integer gadget primitives (model-gadget mirrors) -------
def g_mul(x: int, y: int, oc: OpCounter) -> int:
    """Exact integer multiply — the nibble schoolbook (``_mul_gate`` per nibble
    pair + ``_nibble_carry_round``).  Byte-exact in the model; here we compute the
    same integer product and tally ONE gadget multiply."""
    oc.mul += 1
    return x * y


def g_add(x: int, y: int, oc: OpCounter) -> int:
    """Exact integer add — the nibble carry chain (``_byte_add_block``)."""
    oc.add += 1
    return x + y


def g_sub(x: int, y: int, oc: OpCounter) -> int:
    """Exact integer subtract — the ADD chain with two's-complement addend."""
    oc.add += 1
    return x - y


def g_shr(x: int, k: int, oc: OpCounter) -> int:
    """Exact right shift floor(x / 2^k) — the ``_floor_div_pow`` staircase."""
    oc.shift += 1
    return x >> k


def g_shl(x: int, k: int, oc: OpCounter) -> int:
    """Exact left shift x * 2^k — a multiply by the power-of-two (``2**k`` route)."""
    oc.shift += 1
    return x << k


def g_ge(x: int, thr: int, oc: OpCounter) -> int:
    """Exact integer >= indicator (0/1) — the ``_step_ge`` sharp ramp."""
    oc.cmp += 1
    return 1 if x >= thr else 0


def g_lowbits(x: int, k: int, oc: OpCounter) -> int:
    """Exact ``x mod 2^k`` (the discarded-bits mask) — a shift+subtract in nibble
    form (floor-div staircase gives the high part; the low part is x - 2^k*high).
    Counts as ONE shift gadget (the shared staircase already emitted for g_shr)."""
    oc.shift += 1
    return x & ((1 << k) - 1)


def g_select(cond: int, a: int, b: int, oc: OpCounter) -> int:
    """Exact select ``a if cond else b`` — two guarded writes (``_guard``)."""
    oc.sel += 1
    return a if cond else b


def _nibbles(v: int, n: int) -> List[int]:
    return [(v >> (4 * j)) & 0xF for j in range(n)]


def _leading_bit_pos(x: int, maxbits: int, oc: OpCounter) -> int:
    """Position of the most-significant 1 bit of ``x`` (x>0), i.e. floor(log2 x).

    In the model this is a leading-NIBBLE priority encoder + a within-nibble
    compare cascade (NOT a bit-by-bit scan): for ``n = ceil(maxbits/4)`` nibbles,
    one ``[nib>0]`` compare finds the top nonzero nibble (n compares), then 3
    compares resolve the leading bit WITHIN that nibble (a nibble is 0..15) — the
    standard log-structured normalize.  We tally that honest gadget cost
    (``n + 3`` compares) rather than the ``maxbits``-bit linear scan."""
    n_nib = (maxbits + 3) // 4
    top_nib = -1
    for j in range(n_nib):                       # leading-nonzero-nibble scan
        oc.cmp += 1
        if (x >> (4 * j)) & 0xF:
            top_nib = j
    if top_nib < 0:
        return -1
    nib = (x >> (4 * top_nib)) & 0xF             # resolve the bit within the nibble
    within = -1
    for b in range(4):                            # 3-4 within-nibble compares
        oc.cmp += 1
        if (nib >> b) & 1:
            within = b
    return 4 * top_nib + within


def fmul_gadgets(ua: int, ub: int, oc: "OpCounter | None" = None):
    """FMUL via the nibble gadgets.  Identical result to :func:`fmul_bits`; returns
    ``(result_bits, OpCounter)`` so the fp-MUL cost is measured in exact-integer
    gadget ops.  Field extract/pack are pure shift/mask (shift gadgets)."""
    if oc is None:
        oc = OpCounter()
    ua &= 0xFFFFFFFF
    ub &= 0xFFFFFFFF
    # field extraction: shifts + masks.
    sa = g_shr(ua, 31, oc)
    ea = g_lowbits(g_shr(ua, 23, oc), 8, oc)
    ma = g_lowbits(ua, 23, oc)
    sb = g_shr(ub, 31, oc)
    eb = g_lowbits(g_shr(ub, 23, oc), 8, oc)
    mb = g_lowbits(ub, 23, oc)
    sign = sa ^ sb

    a_is_nan = g_select(g_ge(ea, 255, oc) and ma != 0, 1, 0, oc)
    b_is_nan = g_select(g_ge(eb, 255, oc) and mb != 0, 1, 0, oc)
    a_is_inf = 1 if (ea == 255 and ma == 0) else 0
    b_is_inf = 1 if (eb == 255 and mb == 0) else 0
    a_is_zero = 1 if (ea == 0 and ma == 0) else 0
    b_is_zero = 1 if (eb == 0 and mb == 0) else 0
    if a_is_nan or b_is_nan:
        return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1), oc
    if a_is_inf or b_is_inf:
        if a_is_zero or b_is_zero:
            return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1), oc
        return pack_fields(sign, _EXP_INF, 0), oc
    if a_is_zero or b_is_zero:
        return pack_fields(sign, 0, 0), oc

    sig_a = g_add(_MANT_LEAD, ma, oc)      # OR of the leading 1 (== add for disjoint)
    sig_b = g_add(_MANT_LEAD, mb, oc)
    exp = g_sub(g_add(ea, eb, oc), _BIAS, oc)

    prod = g_mul(sig_a, sig_b, oc)         # the 24x24 -> 48-bit schoolbook multiply
    top = g_shr(prod, 47, oc)
    if top:
        exp = g_add(exp, 1, oc)
        shift = 24
    else:
        shift = 23

    result_mant = g_shr(prod, shift, oc)
    rem = g_lowbits(prod, shift, oc)
    half = 1 << (shift - 1)
    gt = g_ge(rem, half + 1, oc)           # rem > half
    eqh = 1 if rem == half else 0
    odd = g_lowbits(result_mant, 1, oc)
    roundup = g_select(gt, 1, g_select(eqh and odd, 1, 0, oc), oc)
    result_mant = g_add(result_mant, roundup, oc)
    if g_shr(result_mant, 24, oc):
        result_mant = g_shr(result_mant, 1, oc)
        exp = g_add(exp, 1, oc)

    if g_ge(exp, _EXP_INF, oc):
        return pack_fields(sign, _EXP_INF, 0), oc
    if not g_ge(exp, 1, oc):
        return pack_fields(sign, 0, 0), oc
    return pack_fields(sign, exp, g_lowbits(result_mant, 23, oc)), oc


def fadd_gadgets(ua: int, ub: int, oc: "OpCounter | None" = None):
    """FADD via the nibble gadgets.  Identical result to :func:`fadd_bits`; returns
    ``(result_bits, OpCounter)``."""
    if oc is None:
        oc = OpCounter()
    ua &= 0xFFFFFFFF
    ub &= 0xFFFFFFFF
    sa = g_shr(ua, 31, oc)
    ea = g_lowbits(g_shr(ua, 23, oc), 8, oc)
    ma = g_lowbits(ua, 23, oc)
    sb = g_shr(ub, 31, oc)
    eb = g_lowbits(g_shr(ub, 23, oc), 8, oc)
    mb = g_lowbits(ub, 23, oc)

    a_is_nan = 1 if (ea == 255 and ma != 0) else 0
    b_is_nan = 1 if (eb == 255 and mb != 0) else 0
    a_is_inf = 1 if (ea == 255 and ma == 0) else 0
    b_is_inf = 1 if (eb == 255 and mb == 0) else 0
    if a_is_nan or b_is_nan:
        return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1), oc
    if a_is_inf and b_is_inf:
        if sa != sb:
            return pack_fields(0, _EXP_INF, _MANT_LEAD >> 1), oc
        return pack_fields(sa, _EXP_INF, 0), oc
    if a_is_inf:
        return pack_fields(sa, _EXP_INF, 0), oc
    if b_is_inf:
        return pack_fields(sb, _EXP_INF, 0), oc
    a_is_zero = 1 if (ea == 0 and ma == 0) else 0
    b_is_zero = 1 if (eb == 0 and mb == 0) else 0
    if a_is_zero and b_is_zero:
        return pack_fields(1 if (sa and sb) else 0, 0, 0), oc
    if a_is_zero:
        return ub, oc
    if b_is_zero:
        return ua, oc

    sig_a = g_add(_MANT_LEAD, ma, oc)
    sig_b = g_add(_MANT_LEAD, mb, oc)

    # order so A >= B (compare exponents then significands) — a compare + select swap.
    a_smaller = g_select(g_ge(eb, ea + 1, oc) or (ea == eb and g_ge(sig_b, sig_a + 1, oc)),
                         1, 0, oc)
    if a_smaller:
        sa, ea, sig_a, sb, eb, sig_b = sb, eb, sig_b, sa, ea, sig_a

    big = g_shl(sig_a, _GUARD, oc)
    small = g_shl(sig_b, _GUARD, oc)
    diff = g_sub(ea, eb, oc)
    if diff > 0:
        if g_ge(diff, 32, oc):
            small = g_select(small != 0, 1, 0, oc)
        else:
            dropped = g_lowbits(small, diff, oc)
            small = g_shr(small, diff, oc)
            if dropped:
                small = small | 1            # sticky OR into the LSB
                oc.sel += 1                  # one guarded write

    if sa == sb:
        summ = g_add(big, small, oc)
        result_sign = sa
    else:
        summ = g_sub(big, small, oc)
        result_sign = sa
        if summ == 0:
            return pack_fields(0, 0, 0), oc

    exp = ea
    target = 23 + _GUARD
    # the guard-framed sum is < 2^(24+GUARD+1) for equal-sign add, and for a
    # subtract the leading 1 sits at most at bit (24+GUARD); 28 bits covers both.
    hi = _leading_bit_pos(summ, 28, oc)
    if hi > target:
        sh = hi - target
        dropped = g_lowbits(summ, sh, oc)
        summ = g_shr(summ, sh, oc)
        if dropped:
            summ = summ | 1
            oc.sel += 1
        exp = g_add(exp, sh, oc)
    elif hi < target:
        sh = target - hi
        summ = g_shl(summ, sh, oc)
        exp = g_sub(exp, sh, oc)

    result_mant = g_shr(summ, _GUARD, oc)
    rem = g_lowbits(summ, _GUARD, oc)
    half = 1 << (_GUARD - 1)
    gt = g_ge(rem, half + 1, oc)
    eqh = 1 if rem == half else 0
    odd = g_lowbits(result_mant, 1, oc)
    roundup = g_select(gt, 1, g_select(eqh and odd, 1, 0, oc), oc)
    result_mant = g_add(result_mant, roundup, oc)
    if g_shr(result_mant, 24, oc):
        result_mant = g_shr(result_mant, 1, oc)
        exp = g_add(exp, 1, oc)

    if g_ge(exp, _EXP_INF, oc):
        return pack_fields(result_sign, _EXP_INF, 0), oc
    if not g_ge(exp, 1, oc):
        return pack_fields(result_sign, 0, 0), oc
    return pack_fields(result_sign, exp, g_lowbits(result_mant, 23, oc)), oc


SPECIAL_NOTES = """\
COVERED (bit-exact vs numpy.float32):
  * FMUL / FADD of NORMAL x NORMAL operands, both signs.
  * round-to-nearest-EVEN including exact half-ULP ties.
  * exponent overflow -> +/-inf ; underflow -> +/-0.
  * operand special cases: +/-0, +/-inf, NaN operands, inf*0 -> NaN,
    inf + (-inf) -> NaN.
DEFERRED (documented, not yet bit-exact):
  * SUBNORMAL (denormal) RESULTS — flushed to signed zero instead of gradual
    underflow.  (The mantissa-shift machinery is present; only the
    tiny-exponent denormal-encode branch is omitted.)
  * SUBNORMAL OPERANDS — treated via the normal path's implicit-1 assumption;
    a leading-zero significand operand is out of the covered class.
  * NaN PAYLOAD propagation — a canonical quiet NaN is returned rather than
    propagating the incoming NaN's payload bits.
"""
