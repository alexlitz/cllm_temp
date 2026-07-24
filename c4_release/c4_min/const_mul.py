"""CONSTANT-MULTIPLIER fast path — ``a * b`` with ``b`` a COMPILE-TIME constant.

Mirrors the constant-divisor bakeoff pattern (``mul_bakeoff.py`` for the general
multiply): a standalone, byte-exact, CPU-arithmetic bakeoff of two constructions
for ``(a * b) & 0xFFFFFFFF`` when ``b`` is known at BUILD time.  Everything is
COMPOSED from the proven ``nibble_alu32`` SwiGLU primitives (``_ident`` /
``_clear`` / ``_step_ge`` / ``_floor_div_pow`` / ``_floor_div_pow2`` / the byte
add-chain / the carry-round machinery); this file NEVER edits the shared files —
it only READs / imports them.

Baseline to beat (the general carry-share multiply, ``compile_mul_blocks`` +
shared carry): **10 blocks, 11 430 nz** on a full 32-bit runtime ``b``.

TWO CONSTRUCTIONS
=================
**A. Linear partial-product multiply (NO gated multiply).**
   ``a*b = Σ_i a_i * (b * 16^i) = Σ_i Σ_j a_i * B_j * 16^(i+j)``
   where ``a_i`` are the RUNTIME input nibbles (0..15) and ``B_j`` are the BAKED
   constant nibbles of ``b``.  Every partial ``a_i * B_j`` is therefore a runtime
   nibble times a CONSTANT nibble — a **weighted copy** (an ``_ident`` read of the
   ``a_i`` band with the constant weight ``B_j``), NOT a ``_mul_gate``.  No operand
   gather for ``b``, no gated products.  Each raw partial ``a_i*B_j <= 225`` is
   split (low/high nibble) into product columns ``c = i+j`` (each column stays
   ``< 256`` so kmax=15), then carry-resolved with the shared carry round.
   Partials whose ``B_j == 0`` (or whose ``a_i`` is statically zero) are DROPPED —
   the constant-specific saving.

**B. Strength reduction (shift-add) for small constants.**
   Decompose ``b`` into a minimal shift-add chain via canonical-signed-digit /
   NAF: ``x*2^k`` is a nibble RELAYOUT (a free positional copy), and each NAF
   digit contributes ``± x<<k`` — a shift (relayout) + an add/sub.  For small ``b``
   this is 1-3 adds.  ``x*10 = (x<<3)+(x<<1)``, ``x*3 = (x<<1)+x``, etc.  Each add
   reuses the ``nibble_alu32`` byte add-chain (4 sequential byte blocks).

NARROWING (both constructions)
==============================
  * **Result width** — emit only ``ceil(log16(b * 2^in_width))`` result nibbles;
    the high nibbles are provably zero for a bounded product and are dropped.
  * **Input width** — if ``a`` is statically bounded to ``in_width`` bits, only its
    ``ceil(in_width/4)`` low nibbles are nonzero, so construction A generates
    fewer partials and a shorter carry-resolve.

SPECIAL CASES
=============
  ``b == 0`` -> result 0 (a single clear block).  ``b == 1`` -> identity (copy).
  ``b`` a power of two -> pure nibble/bit SHIFT (relayout, ZERO staircase units).

fp32 discipline (INHERITED from ``nibble_alu32``): no hidden unit's relu argument
exceeds ``2^24``.  A weighted-copy partial ``a_i*B_j <= 225``; a product column is
kept ``< 256``; the byte add-chain sums ``<= 511``.  So ``RELU_S * arg`` (RELU_S =
200) is always ``< 2^24`` -> fully fp32, ZERO fp64.

Run:
    python -m c4_min.const_mul
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch

from .nibble_vm_layout import NibbleVMLayout
from . import nibble_alu32 as A
from .nibble_vm import S, RELU_S, _empty_spec
from .nibble_alu32 import (
    _ident, _clear, _floor_div_pow, _floor_div_pow2,
    _byte_add_block, _truncate, compile_mul_blocks, extend_layout_for_alu32,
)

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

# fp32 integer-exactness ceiling and the relu-scale headroom (same as mul_bakeoff).
FP32_INT_MAX = 1 << 24                      # 2^24, fp32 exact-integer ceiling
FORM_CEIL_FP32 = FP32_INT_MAX / RELU_S      # ~83886: max staircase arg fp32-safe

NCOL = 8                                    # 8 nibble result columns (32-bit)
_NIB_KMAX = 15                              # floor(col/16) for a column kept < 256


# ===========================================================================
# Private scratch allocation (never touches ALU32Bands; each build gets its own
# band range so the sparse sim reads/writes only its own bands).
# ===========================================================================
def _scratch(L, name, size):
    key = f"CMUL_{name}"
    if key in L._names:
        return L._names[key][0]
    return L._band(key, size)


# ===========================================================================
# NARROWING helpers.
# ===========================================================================
def _const_nibbles(b: int) -> List[int]:
    """Little-endian nonzero-truncated nibbles of the constant ``b`` (0..2^32)."""
    if b == 0:
        return []
    nibs = []
    v = b
    while v:
        nibs.append(v & 0xF)
        v >>= 4
    return nibs


def _in_nibbles(in_width: int) -> int:
    """Number of nonzero input nibbles for a value bounded to ``in_width`` bits."""
    return max(1, (in_width + 3) // 4)


def _result_nibbles(b: int, in_width: int) -> int:
    """``ceil(log16(b * 2^in_width))`` capped at 8 (32-bit result window).

    The product ``a*b <= (2^in_width - 1) * b`` needs
    ``ceil(log16(product_max + 1))`` nibbles; anything above nibble 7 is masked
    out by ``& 0xFFFFFFFF`` so we never emit more than 8."""
    if b == 0:
        return 1
    prod_max = ((1 << in_width) - 1) * b
    if prod_max <= 0:
        return 1
    n = 0
    v = prod_max
    while v:
        n += 1
        v >>= 4
    return min(max(n, 1), NCOL)


def _is_pow2(b: int) -> bool:
    return b > 0 and (b & (b - 1)) == 0


# ===========================================================================
# CONSTRUCTION A — LINEAR partial-product multiply (weighted copies).
#
#   a*b = Σ_i Σ_j  a_i * B_j * 16^(i+j)
#
# ``a_i`` runtime nibble (0..15), ``B_j`` baked constant nibble.  Each partial is
# a WEIGHTED COPY ``B_j * a_i`` (one ``_ident`` unit on band ``a_i`` scaled by the
# constant ``B_j``) — never a gated multiply.  The raw partial ``<= 225`` is split
# low/high (shared staircase, kmax=15) into columns ``i+j`` / ``i+j+1`` (kept
# ``< 256``), then carry-resolved with the shared carry round.  Only ``B_j != 0``
# and ``i < in_nib`` partials are emitted (constant + input-width narrowing), and
# only ``res_nib`` result columns are carried/copied (result-width narrowing).
# ===========================================================================
def _carry_round_shared(spec, u, src, dst, n):
    """One base-16 carry-normalise round on ``n`` columns kept ``< 256``, using the
    shared floor(col/16) staircase (mod -16 into col c, carry +1 into col c+1) —
    the carry-SHARING primitive (matches nibble_alu32._nibble_carry_round)."""
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)          # + col
        if c + 1 < n:
            u = _floor_div_pow2(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX,
                                dst + c, -16.0, dst + c + 1, 1.0)
        else:
            u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX,
                               dst + c, -16.0)
    return u


def _carry_round_block(L, dim, src, dst, n) -> Block:
    spec = _empty_spec(dim, n * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
    u = _carry_round_shared(spec, 0, src, dst, n)
    return _truncate(spec, u, dim)


def _ripple_len(pairs, res_nib) -> int:
    """Worst-case carry ripple length across the emitted result columns.

    A carry ripples one column per round; the count is bounded by the number of
    populated columns (NOT the width), which for a narrowed product is
    ``res_nib`` columns.  We take ``res_nib`` + 1 headroom (settled columns are
    fixed points of a further round, so extra rounds never corrupt)."""
    if not pairs:
        return 0
    max_col = max(i + j for (i, j) in pairs)
    return min(max_col + 1, res_nib) + 1


def build_linear_fold(L, dim, b: int, in_width: int = 32) -> Tuple[List[Block], int, dict]:
    """Construction A: linear partial-product (weighted-copy) constant multiply."""
    A._ONE = L.ONE
    L_STACK0 = L.STACK0                                # operand a lives in STACK0 nibbles
    in_nib = _in_nibbles(in_width)
    res_nib = _result_nibbles(b, in_width)
    Bn = _const_nibbles(b)                             # constant nibbles (nonzero-truncated)

    RES = _scratch(L, f"LF_RES_{b}_{in_width}", NCOL)
    blocks: List[Block] = []

    # b == 0 : product is 0.  b == 1 : identity copy of a's low res_nib nibbles.
    if b == 0:
        spec = _empty_spec(dim, NCOL)
        u = 0
        for c in range(NCOL):
            u = _clear(spec, u, RES + c)
        blocks.append(("lf-zero", _truncate(spec, u, dim)))
        info = {"construction": "A_linear_fold", "special": "b==0",
                "n_partials": 0, "carry_rounds": 0, "res_nibbles": 1,
                "max_staircase_arg": 0, "in_nib": in_nib}
        return blocks, RES, info

    # POWER OF TWO -> pure nibble/bit SHIFT (relayout, ZERO staircase units).
    if _is_pow2(b):
        return _build_pow2_shift(L, dim, b, in_width, RES, "A_linear_fold")

    # (i, j) partial pairs: i over input nibbles, j over constant nibbles, B_j != 0,
    # and result column i+j < res_nib (higher columns are masked out).
    pairs = [(i, j) for i in range(in_nib) for j in range(len(Bn))
             if Bn[j] != 0 and (i + j) < res_nib]
    n_part = len(pairs)

    # (1) raw weighted-copy partials: PP[idx] = B_j * a_i  (<= 15*15 = 225).  One
    #     _ident unit per partial (weighted copy, NOT a gated multiply).
    PP = _scratch(L, f"LF_PP_{b}_{in_width}", max(1, n_part))
    spec = _empty_spec(dim, max(1, n_part) * 2)
    u = 0
    for idx, (i, j) in enumerate(pairs):
        u = _clear(spec, u, PP + idx)
        u = _ident(spec, u, {L_STACK0 + i: float(Bn[j])}, 0.0, PP + idx, 1.0)  # B_j * a_i
    blocks.append(("lf-partials", _truncate(spec, u, dim)))

    # (2) split each raw partial (0..225) low/high into columns i+j / i+j+1.  One
    #     shared floor(p/16) staircase routes mod (-16) + carry (+1).  A column then
    #     holds <= 8 low + <= 8 high <= 232 < 256 -> kmax=15 carry rounds settle it.
    COLS = _scratch(L, f"LF_COLS_{b}_{in_width}", NCOL)
    spec = _empty_spec(dim, NCOL + n_part * (1 + _NIB_KMAX * 2))
    u = 0
    for c in range(NCOL):
        u = _clear(spec, u, COLS + c)
    for idx, (i, j) in enumerate(pairs):
        c = i + j
        pp = PP + idx
        u = _ident(spec, u, {pp: 1.0}, 0.0, COLS + c, 1.0)                     # + p (col c)
        if c + 1 < res_nib:
            u = _floor_div_pow2(spec, u, {pp: 1.0}, 0.0, 16, _NIB_KMAX,
                                COLS + c, -16.0, COLS + c + 1, 1.0)            # mod / carry
        else:
            u = _floor_div_pow(spec, u, {pp: 1.0}, 0.0, 16, _NIB_KMAX,
                               COLS + c, -16.0)                                # top: mod only
    blocks.append(("lf-split", _truncate(spec, u, dim)))

    # (3) carry rounds settle each column to a single nibble (kmax=15).
    rounds = _ripple_len(pairs, res_nib)
    src, dst = COLS, _scratch(L, f"LF_COLS2_{b}_{in_width}", NCOL)
    for r in range(rounds):
        blocks.append((f"lf-carry{r}", _carry_round_block(L, dim, src, dst, res_nib)))
        src, dst = dst, src

    # (4) copy the settled result nibbles into RES (zero the rest).
    spec = _empty_spec(dim, NCOL * 2)
    u = 0
    for c in range(NCOL):
        u = _clear(spec, u, RES + c)
        if c < res_nib:
            u = _ident(spec, u, {src + c: 1.0}, 0.0, RES + c, 1.0)
    blocks.append(("lf-result", _truncate(spec, u, dim)))

    info = {
        "construction": "A_linear_fold",
        "note": "weighted-copy partials (B_j*a_i, NO gated multiply); split+carry",
        "n_partials": n_part, "carry_rounds": rounds, "res_nibbles": res_nib,
        "in_nib": in_nib, "const_nibbles": len(Bn), "max_staircase_arg": 225,
    }
    return blocks, RES, info


# ===========================================================================
# POWER-OF-TWO SHIFT — shared by both constructions.  b = 2^k : a*b is a pure
# nibble/bit relayout.  ``k = 4*qn + rn`` (qn nibble shift, rn 0..3 bit shift).
# For rn == 0 the shift is a FREE positional copy (0 staircase units).  For rn>0
# each source nibble contributes rn bits to one column and (4-rn) bits to the next
# via a floor/mod split (kmax=15), still tiny.
# ===========================================================================
def _build_pow2_shift(L, dim, b, in_width, RES, construction) -> Tuple[List[Block], int, dict]:
    k = b.bit_length() - 1                             # b == 2^k
    qn, rn = divmod(k, 4)                              # nibble shift, sub-nibble bit shift
    L_STACK0 = L.STACK0
    in_nib = _in_nibbles(in_width)
    res_nib = _result_nibbles(b, in_width)
    blocks: List[Block] = []

    if rn == 0:
        # pure nibble relayout: RES[c] = a[c - qn].  ZERO staircase units.
        spec = _empty_spec(dim, NCOL * 2)
        u = 0
        for c in range(NCOL):
            u = _clear(spec, u, RES + c)
            src_nib = c - qn
            if 0 <= src_nib < in_nib and c < res_nib:
                u = _ident(spec, u, {L_STACK0 + src_nib: 1.0}, 0.0, RES + c, 1.0)
        blocks.append(("pow2-nibshift", _truncate(spec, u, dim)))
        info = {
            "construction": construction, "special": f"b==2^{k} (nibble shift)",
            "n_partials": 0, "carry_rounds": 0, "res_nibbles": res_nib,
            "in_nib": in_nib, "shift_k": k, "max_staircase_arg": 0,
        }
        return blocks, RES, info

    # sub-nibble bit shift: value <<= rn.  Each source nibble n_s (0..15) shifted
    # left by rn contributes ``(n_s << rn) mod 16`` to column (s+qn) and
    # ``(n_s << rn) >> 4 = floor(n_s / 2^(4-rn))`` to column (s+qn+1).  Build the
    # shifted nibble value ``v = n_s << rn`` (<= 15*8 = 120) as a weighted copy,
    # then split low/high — one shared floor staircase (kmax=15).
    SV = _scratch(L, f"P2SV_{b}_{in_width}", in_nib)   # shifted nibble values
    spec = _empty_spec(dim, in_nib * 2)
    u = 0
    mul = 1 << rn
    for s in range(in_nib):
        u = _clear(spec, u, SV + s)
        u = _ident(spec, u, {L_STACK0 + s: float(mul)}, 0.0, SV + s, 1.0)     # n_s << rn
    blocks.append(("pow2-shiftval", _truncate(spec, u, dim)))

    COLS = _scratch(L, f"P2C_{b}_{in_width}", NCOL)
    spec = _empty_spec(dim, NCOL + in_nib * (1 + _NIB_KMAX * 2))
    u = 0
    for c in range(NCOL):
        u = _clear(spec, u, COLS + c)
    for s in range(in_nib):
        c = s + qn
        if c >= res_nib:
            continue
        sv = SV + s
        u = _ident(spec, u, {sv: 1.0}, 0.0, COLS + c, 1.0)
        if c + 1 < res_nib:
            u = _floor_div_pow2(spec, u, {sv: 1.0}, 0.0, 16, _NIB_KMAX,
                                COLS + c, -16.0, COLS + c + 1, 1.0)
        else:
            u = _floor_div_pow(spec, u, {sv: 1.0}, 0.0, 16, _NIB_KMAX,
                               COLS + c, -16.0)
    blocks.append(("pow2-split", _truncate(spec, u, dim)))

    # one carry round settles (each column already < 32, single carry).
    src, dst = COLS, _scratch(L, f"P2C2_{b}_{in_width}", NCOL)
    blocks.append(("pow2-carry", _carry_round_block(L, dim, src, dst, res_nib)))
    src = dst

    spec = _empty_spec(dim, NCOL * 2)
    u = 0
    for c in range(NCOL):
        u = _clear(spec, u, RES + c)
        if c < res_nib:
            u = _ident(spec, u, {src + c: 1.0}, 0.0, RES + c, 1.0)
    blocks.append(("pow2-result", _truncate(spec, u, dim)))
    info = {
        "construction": construction, "special": f"b==2^{k} (bit shift rn={rn})",
        "n_partials": 0, "carry_rounds": 1, "res_nibbles": res_nib,
        "in_nib": in_nib, "shift_k": k, "max_staircase_arg": 120,
    }
    return blocks, RES, info


# ===========================================================================
# CONSTRUCTION B — STRENGTH REDUCTION (shift-add) via canonical signed digit / NAF.
#
# NAF(b): a signed-binary representation with NO two adjacent nonzero digits, so
# ``b = Σ_k d_k * 2^k`` with ``d_k in {-1,0,+1}`` and the FEWEST nonzero digits of
# any signed-binary form (the minimal add/sub count).  Each nonzero digit is a
# term ``± (a << k)`` = a SHIFT (nibble/bit relayout) + an ADD or SUB.  We fold the
# terms left-to-right through the ``nibble_alu32`` byte add/sub chain.
#
# For small ``b`` this is 1-3 adds (e.g. b=3 -> a<<2 - a = 1 sub; b=10 -> a<<3 +
# a<<1 = 1 add; b=100 -> NAF has 3 nonzero digits -> 2 add/subs).  Powers of two
# are a single shift (0 adds).  The chain reuses the proven 4-byte add-chain, so
# every hidden value is a byte sum <= 511 -> fp32-exact.
# ===========================================================================
def _naf(b: int) -> List[int]:
    """Non-adjacent form: returns digits ``d`` little-endian with ``d[k] in
    {-1,0,1}`` and ``b = Σ d[k] 2^k``, minimal Hamming weight."""
    digits = []
    v = b
    while v > 0:
        if v & 1:
            z = 2 - (v & 3)                            # +1 or -1 (non-adjacent)
            digits.append(z)
            v -= z
        else:
            digits.append(0)
        v >>= 1
    return digits


def _naf_terms(b: int) -> List[Tuple[int, int]]:
    """NAF as ``(shift_k, sign)`` terms with sign in {+1,-1}, sorted by shift."""
    return [(k, d) for k, d in enumerate(_naf(b)) if d != 0]


def _shift_nibbles_block(L, dim, src_base, in_nib, k, dst_bytes, res_nib) -> List[Block]:
    """``(a << k) & 0xFFFFFFFF`` into 4 little-endian BYTES ``dst_bytes[0..3]``.

    Pure relayout: for a sub-nibble shift, form each source nibble's shifted value
    ``n_s << rn`` (<= 120), split low/high across result nibbles (one shared
    floor staircase, kmax=15), settle, then pack nibble pairs into bytes for the
    add-chain.  For rn==0 it is a free positional copy."""
    qn, rn = divmod(k, 4)
    blocks: List[Block] = []
    NIB = _scratch(L, f"SHN_{src_base}_{k}", NCOL)     # shifted result nibbles

    if rn == 0:
        spec = _empty_spec(dim, NCOL * 2)
        u = 0
        for c in range(NCOL):
            u = _clear(spec, u, NIB + c)
            s = c - qn
            if 0 <= s < in_nib and c < res_nib:
                u = _ident(spec, u, {src_base + s: 1.0}, 0.0, NIB + c, 1.0)
        blocks.append((f"shift-k{k}-nib", _truncate(spec, u, dim)))
    else:
        SV = _scratch(L, f"SHSV_{src_base}_{k}", in_nib)
        spec = _empty_spec(dim, in_nib * 2)
        u = 0
        mul = 1 << rn
        for s in range(in_nib):
            u = _clear(spec, u, SV + s)
            u = _ident(spec, u, {src_base + s: float(mul)}, 0.0, SV + s, 1.0)
        blocks.append((f"shift-k{k}-sv", _truncate(spec, u, dim)))
        COLS = _scratch(L, f"SHC_{src_base}_{k}", NCOL)
        spec = _empty_spec(dim, NCOL + in_nib * (1 + _NIB_KMAX * 2))
        u = 0
        for c in range(NCOL):
            u = _clear(spec, u, COLS + c)
        for s in range(in_nib):
            c = s + qn
            if c >= res_nib:
                continue
            sv = SV + s
            u = _ident(spec, u, {sv: 1.0}, 0.0, COLS + c, 1.0)
            if c + 1 < res_nib:
                u = _floor_div_pow2(spec, u, {sv: 1.0}, 0.0, 16, _NIB_KMAX,
                                    COLS + c, -16.0, COLS + c + 1, 1.0)
            else:
                u = _floor_div_pow(spec, u, {sv: 1.0}, 0.0, 16, _NIB_KMAX,
                                   COLS + c, -16.0)
        blocks.append((f"shift-k{k}-split", _truncate(spec, u, dim)))
        dst = _scratch(L, f"SHC2_{src_base}_{k}", NCOL)
        blocks.append((f"shift-k{k}-carry", _carry_round_block(L, dim, COLS, dst, res_nib)))
        # copy settled -> NIB
        spec = _empty_spec(dim, NCOL * 2)
        u = 0
        for c in range(NCOL):
            u = _clear(spec, u, NIB + c)
            if c < res_nib:
                u = _ident(spec, u, {dst + c: 1.0}, 0.0, NIB + c, 1.0)
        blocks.append((f"shift-k{k}-nibcopy", _truncate(spec, u, dim)))

    # pack the 8 result nibbles into 4 little-endian bytes for the add-chain.
    spec = _empty_spec(dim, 4 * 2)
    u = 0
    for r in range(4):
        u = _clear(spec, u, dst_bytes + r)
        u = _ident(spec, u, {NIB + 2 * r: 1.0, NIB + 2 * r + 1: 16.0}, 0.0,
                   dst_bytes + r, 1.0)
    blocks.append((f"shift-k{k}-bytes", _truncate(spec, u, dim)))
    return blocks


def _notb_block(L, dim, src_bytes, notb_bytes) -> Block:
    """~B byte = 255 - B (ones' complement) for a SUB term (two's complement +1
    supplied as the add-chain's cin)."""
    spec = _empty_spec(dim, 4 * 2)
    u = 0
    for i in range(4):
        u = _clear(spec, u, notb_bytes + i)
        u = _ident(spec, u, {src_bytes + i: -1.0}, 255.0, notb_bytes + i, 1.0)
    return _truncate(spec, u, dim)


def _copy_bytes_block(L, dim, src_bytes, dst_bytes) -> Block:
    spec = _empty_spec(dim, 4 * 2)
    u = 0
    for i in range(4):
        u = _clear(spec, u, dst_bytes + i)
        u = _ident(spec, u, {src_bytes + i: 1.0}, 0.0, dst_bytes + i, 1.0)
    return _truncate(spec, u, dim)


def build_shift_add(L, dim, b: int, in_width: int = 32) -> Tuple[List[Block], int, dict]:
    """Construction B: strength-reduction (NAF shift-add) constant multiply.

    Returns ``(blocks, res_band, info)`` where ``res_band`` holds the 8 result
    NIBBLES (to match construction A's RES convention the caller reads nibbles)."""
    A._ONE = L.ONE
    L_STACK0 = L.STACK0
    in_nib = _in_nibbles(in_width)
    res_nib = _result_nibbles(b, in_width)
    RES = _scratch(L, f"SA_RES_{b}_{in_width}", NCOL)
    blocks: List[Block] = []

    if b == 0:
        spec = _empty_spec(dim, NCOL)
        u = 0
        for c in range(NCOL):
            u = _clear(spec, u, RES + c)
        blocks.append(("sa-zero", _truncate(spec, u, dim)))
        info = {"construction": "B_shift_add", "special": "b==0", "n_terms": 0,
                "n_adds": 0, "res_nibbles": 1, "max_staircase_arg": 0}
        return blocks, RES, info

    if _is_pow2(b):
        return _build_pow2_shift(L, dim, b, in_width, RES, "B_shift_add")

    terms = _naf_terms(b)                              # (shift_k, sign) list, sorted asc
    # The HIGHEST-shift NAF digit of a positive ``b`` is ALWAYS +1 (the leading
    # signed-binary digit of a positive number is +1), so we START the accumulator
    # from that positive term and FOLD the remaining terms (any sign) into it.  This
    # keeps the accumulator a positive-then-modular value all the way through — no
    # signed accumulator needed.  ``terms`` is shift-ascending; reorder so the
    # highest-shift (+1) term is first.
    terms = sorted(terms, key=lambda t: -t[0])         # highest shift first
    assert terms[0][1] == +1, f"NAF leading digit not +1 for b={b}: {terms}"

    ACC = _scratch(L, f"SA_ACC_{b}_{in_width}", 4)     # accumulator BYTES
    TERM = _scratch(L, f"SA_TERM_{b}_{in_width}", 4)   # current term BYTES
    NOTB = _scratch(L, f"SA_NOTB_{b}_{in_width}", 4)   # ~term (for SUB)
    ADDC = _scratch(L, f"SA_ADDC_{b}_{in_width}", 5)   # add carry chain
    ADDR = _scratch(L, f"SA_ADDR_{b}_{in_width}", 8)   # add result nibbles

    k0, s0 = terms[0]
    blocks += _shift_nibbles_block(L, dim, L_STACK0, in_nib, k0, ACC, res_nib)

    n_adds = 0
    for (k, sign) in terms[1:]:
        blocks += _shift_nibbles_block(L, dim, L_STACK0, in_nib, k, TERM, res_nib)
        if sign > 0:
            # ACC := (ACC + TERM) & 0xFFFFFFFF via the 4-byte add-chain.
            for i in range(4):
                blocks.append((f"sa-add-k{k}-b{i}",
                               _byte_add_block(L, dim, ACC, TERM,
                                               ADDC, ADDR, i, cin_const=0.0)))
        else:
            # ACC := (ACC - TERM) = ACC + ~TERM + 1  via the same chain (cin=1).
            blocks.append((f"sa-notb-k{k}", _notb_block(L, dim, TERM, NOTB)))
            for i in range(4):
                blocks.append((f"sa-sub-k{k}-b{i}",
                               _byte_add_block(L, dim, ACC, NOTB,
                                               ADDC, ADDR, i, cin_const=1.0)))
        # copy the 8 add-result nibbles back into ACC bytes for the next term.
        spec = _empty_spec(dim, 4 * 2)
        u = 0
        for i in range(4):
            u = _clear(spec, u, ACC + i)
            u = _ident(spec, u, {ADDR + 2 * i: 1.0, ADDR + 2 * i + 1: 16.0}, 0.0,
                       ACC + i, 1.0)
        blocks.append((f"sa-acc-k{k}", _truncate(spec, u, dim)))
        n_adds += 1

    # final: split ACC bytes (0..255) into the 8 result nibbles.  Per byte: 1
    # _ident + two _floor_div_pow(m=16,kmax=15) staircases (30 relu units each).
    spec = _empty_spec(dim, NCOL + 4 * (1 + 2 * _NIB_KMAX * 2))
    u = 0
    for c in range(NCOL):
        u = _clear(spec, u, RES + c)
    for i in range(4):
        lo, hi = RES + 2 * i, RES + 2 * i + 1
        u = _ident(spec, u, {ACC + i: 1.0}, 0.0, lo, 1.0)
        u = _floor_div_pow(spec, u, {ACC + i: 1.0}, 0.0, 16, _NIB_KMAX, lo, -16.0)
        u = _floor_div_pow(spec, u, {ACC + i: 1.0}, 0.0, 16, _NIB_KMAX, hi, 1.0)
    blocks.append(("sa-result", _truncate(spec, u, dim)))

    info = {
        "construction": "B_shift_add",
        "note": "NAF shift-add; each term = shift (relayout) + one byte add-chain",
        "n_terms": len(terms), "n_adds": n_adds, "naf_weight": len(terms),
        "res_nibbles": res_nib, "in_nib": in_nib, "max_staircase_arg": 511,
    }
    return blocks, RES, info


# ===========================================================================
# AUTO-PICK: build the smaller of A / B per b (or expose both).
# ===========================================================================
def _pick_key(m: dict, metric: str = "nz") -> Tuple[int, int]:
    """Ordering key for the auto-pick.  ``metric='nz'`` (default) sorts by baked
    weight count then depth — the meaningful "size" for persistent FFN weights, so a
    2x-lighter build wins even if 1-2 blocks deeper.  ``metric='depth'`` sorts by
    sequential-block count first (latency-first)."""
    if metric == "depth":
        return (m["depth"], m["weights_nz"])
    return (m["weights_nz"], m["depth"])


def build_const_mul(L, dim, b: int, in_width: int = 32, prefer: str = "auto",
                    metric: str = "nz"):
    """Build the constant multiplier ``a*b`` and return ``(blocks, res_band, info)``.

    ``prefer`` in {"auto","A","B"}: "auto" picks the smaller construction by
    ``metric`` (default 'nz' = baked weight count, depth tiebreak; 'depth' =
    latency-first); "A" forces the linear fold; "B" forces the shift-add.  Special
    b (0, 1, powers of two) are handled inside both builders identically (both
    collapse to the same pure-shift blocks)."""
    if prefer == "A":
        return build_linear_fold(L, dim, b, in_width)
    if prefer == "B":
        return build_shift_add(L, dim, b, in_width)
    # auto: build both, measure, keep the smaller by ``metric``.
    bl_a, res_a, info_a = build_linear_fold(L, dim, b, in_width)
    bl_b, res_b, info_b = build_shift_add(L, dim, b, in_width)
    ma, mb = measure(bl_a), measure(bl_b)
    if _pick_key(mb, metric) < _pick_key(ma, metric):
        return bl_b, res_b, {**info_b, "picked": "B"}
    return bl_a, res_a, {**info_a, "picked": "A"}


# ===========================================================================
# MEASUREMENT (mirrors mul_bakeoff.measure).
# ===========================================================================
def weight_nz(spec: Spec, include_bias: bool = False) -> int:
    keys = ["W_up", "W_gate", "W_down"]
    if include_bias:
        keys += ["b_up", "b_gate", "b_down"]
    return sum(int((spec[k] != 0).sum()) for k in keys)


def measure(blocks: List[Block]) -> dict:
    depth = len(blocks)
    nz = sum(weight_nz(s, include_bias=False) for _, s in blocks)
    nz_bias = sum(weight_nz(s, include_bias=True) for _, s in blocks)
    return {"depth": depth, "weights_nz": nz, "weights_nz_with_bias": nz_bias}


# ===========================================================================
# SPARSE BYTE-EXACT SIMULATION (identical math to mul_bakeoff, fp32 only here).
# ===========================================================================
def _spec_sparsity(spec: Spec, dtype):
    key = f"_sc_{dtype}"
    cache = spec.get(key)
    if cache is not None:
        return cache
    up_bands = torch.nonzero(spec["W_up"].abs().sum(0) != 0, as_tuple=False).flatten().tolist()
    gate_bands = torch.nonzero(spec["W_gate"].abs().sum(0) != 0, as_tuple=False).flatten().tolist()
    out_dims = torch.nonzero(spec["W_down"].abs().sum(1) != 0, as_tuple=False).flatten().tolist()
    Wu = spec["W_up"][:, up_bands].to(dtype).contiguous()
    Wg = spec["W_gate"][:, gate_bands].to(dtype).contiguous()
    Wd = spec["W_down"][out_dims].to(dtype).contiguous()
    cache = (up_bands, gate_bands, out_dims, Wu, Wg, Wd,
             spec["b_up"].to(dtype), spec["b_gate"].to(dtype),
             spec["b_down"][out_dims].to(dtype))
    spec[key] = cache
    return cache


def _sparse_apply(state: Dict[int, float], spec: Spec, dtype=torch.float32):
    import torch.nn.functional as F
    (up_bands, gate_bands, out_dims, Wu, Wg, Wd,
     b_up, b_gate, b_down) = _spec_sparsity(spec, dtype)
    up = b_up.clone()
    gate = b_gate.clone()
    if up_bands:
        v = torch.tensor([state.get(b, 0.0) for b in up_bands], dtype=dtype)
        up = up + Wu @ v
    if gate_bands:
        v = torch.tensor([state.get(b, 0.0) for b in gate_bands], dtype=dtype)
        gate = gate + Wg @ v
    hidden = F.silu(up) * gate
    if out_dims:
        delta = Wd @ hidden + b_down
        for i, d in enumerate(out_dims):
            dv = float(delta[i])
            if dv:
                state[d] = state.get(d, 0.0) + dv
    return state


def simulate_sparse(L, blocks: List[Block], res_band: int, a_val: int,
                    dtype=torch.float32) -> int:
    """``a`` seeded into STACK0 nibbles; ``b`` is BAKED into the blocks.  Returns the
    8-nibble result read from ``res_band``."""
    state: Dict[int, float] = {L.ONE: 1.0}
    for c in range(8):
        state[L.STACK0 + c] = float((a_val >> (4 * c)) & 0xF)
    for _n, s in blocks:
        state = _sparse_apply(state, s, dtype=dtype)
    out = 0
    for c in range(8):
        out |= (int(round(state.get(res_band + c, 0.0))) & 0xF) << (4 * c)
    return out & 0xFFFFFFFF


# ===========================================================================
# BATTERY + DRIVER
# ===========================================================================
# The constant multipliers to bake off (per brief): 0,1,2,3,5,7,10,16,100,255,
# 256,1000,65535, primes, powers of two, large 2^20 / 10^6.
BATTERY = [0, 1, 2, 3, 5, 7, 10, 16, 100, 255, 256, 1000, 65535,
           # primes
           11, 13, 17, 97, 251, 65537,
           # powers of two
           4, 8, 32, 1024, (1 << 16), (1 << 20),
           # large
           1_000_000, (1 << 24), 0xFFFFFFFF]


DIM = 2048          # wide enough for every b's private scratch bands.


def _make_layout():
    L = NibbleVMLayout(8, n_heads=4)
    extend_layout_for_alu32(L)
    A._ONE = L.ONE
    return L


def test_a_operands(b: int, n_random: int = 2000, seed: int = 0) -> List[int]:
    """>= 2000 random full-32-bit ``a`` + structured edges."""
    rng = random.Random(seed ^ (b * 2654435761 & 0xFFFFFFFF))
    M = 0xFFFFFFFF
    edges = [0, 1, 2, 3, M, M - 1, 0x80000000, 0x7FFFFFFF, 0xFFFF, 0x10000,
             0xDEADBEEF, 0xCAFEBABE, 0xABCD, 255, 256, 65535, 65536,
             1 << 20, 1 << 24, 0x0F0F0F0F, 0xF0F0F0F0, 0x11111111, 0x88888888]
    ops = list(edges)
    for _ in range(n_random):
        ops.append(rng.randint(0, M))
    return ops


def _general_multiply_baseline(L, dim) -> Tuple[int, int]:
    """(depth, nz) of the general carry-share multiply — the number to beat.  Reuse
    the exact carry-share build from the general bakeoff pattern."""
    A._ONE = L.ONE
    a = L.ALU32
    base = compile_mul_blocks(L, dim)         # products, split, carry0..6, result
    front = base[:2]
    blocks: List[Block] = list(front)
    src, dst = a.MCOL, a.MC1
    for r in range(7):
        blocks.append((f"gm-carry{r}", _carry_round_block(L, dim, src, dst, NCOL)))
        src, dst = dst, src
    # result copy
    spec = _empty_spec(dim, NCOL * 2)
    u = 0
    RES = _scratch(L, "GM_RES", NCOL)
    for c in range(NCOL):
        u = _clear(spec, u, RES + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, RES + c, 1.0)
    blocks.append(("gm-result", _truncate(spec, u, dim)))
    m = measure(blocks)
    return m["depth"], m["weights_nz"]


def run_bakeoff(n_random: int = 2000, verbose: bool = True) -> dict:
    results: dict = {}
    dim = DIM
    M = 0xFFFFFFFF

    # each build gets a FRESH layout (private scratch bands do not stack across b).
    gm_depth, gm_nz = _general_multiply_baseline(_make_layout(), dim)
    results["_general_multiply_baseline"] = {"depth": gm_depth, "weights_nz": gm_nz}
    if verbose:
        print(f"GENERAL carry-share multiply baseline: depth={gm_depth} nz={gm_nz}")
        print()

    per_b: dict = {}
    for b in BATTERY:
        entry: dict = {}
        for tag, prefer in [("A_linear", "A"), ("B_shiftadd", "B")]:
            L = _make_layout()
            blocks, res, info = build_const_mul(L, dim, b, in_width=32, prefer=prefer)
            assert L._off <= dim, f"layout grew past DIM ({L._off} > {dim}); raise DIM"
            m = measure(blocks)
            ops = test_a_operands(b, n_random=n_random)
            n_pass = 0
            first_fail = None
            for a_val in ops:
                got = simulate_sparse(L, blocks, res, a_val, dtype=torch.float32)
                exp = (a_val * b) & M
                if got == exp:
                    n_pass += 1
                elif first_fail is None:
                    first_fail = (hex(a_val), hex(got), hex(exp))
            entry[tag] = {
                **m, "info": info,
                "byte_exact_pass": n_pass, "byte_exact_total": len(ops),
                "first_fail": first_fail,
                "res_nibbles": info.get("res_nibbles"),
                "max_staircase_arg": info.get("max_staircase_arg", 0),
            }
        # auto pick = the smaller build by nz (depth tiebreak), from the two
        # already-measured builds.
        key_a = _pick_key(entry["A_linear"], "nz")
        key_b = _pick_key(entry["B_shiftadd"], "nz")
        entry["picked"] = "B" if key_b < key_a else "A"
        per_b[b] = entry
        if verbose:
            a_e = entry["A_linear"]; b_e = entry["B_shiftadd"]
            print(f"b={b:>11d}  A[d={a_e['depth']:>2d} nz={a_e['weights_nz']:>6d} "
                  f"ex={a_e['byte_exact_pass']}/{a_e['byte_exact_total']}]  "
                  f"B[d={b_e['depth']:>2d} nz={b_e['weights_nz']:>6d} "
                  f"ex={b_e['byte_exact_pass']}/{b_e['byte_exact_total']}]  "
                  f"pick={entry['picked']}  resnib={a_e['res_nibbles']}")
    results["per_b"] = per_b
    return results


def _print_narrowing_table(results: dict):
    gm = results["_general_multiply_baseline"]
    per_b = results["per_b"]
    print()
    print("NARROWING TABLE  (general carry-share multiply = depth {} / nz {})"
          .format(gm["depth"], gm["weights_nz"]))
    print("=" * 92)
    print(f"{'b':>11s} | {'A depth/nz':>14s} | {'B depth/nz':>14s} | "
          f"{'pick':>4s} | {'resnib':>6s} | {'best vs gen':>11s} | regime")
    print("-" * 92)
    for b in BATTERY:
        e = per_b[b]
        a_e, b_e = e["A_linear"], e["B_shiftadd"]
        pick = e["picked"]
        best = a_e if pick == "A" else b_e
        ratio = best["weights_nz"] / gm["weights_nz"]
        # regime = which family wins, by the actual auto-pick.
        if _is_pow2(b) or b in (0, 1):
            regime = "special/shift (pure relayout)"
        elif pick == "B":
            regime = "shift-add wins (low NAF weight)"
        else:
            regime = "linear-fold wins"
        print(f"{b:>11d} | {a_e['depth']:>4d}/{a_e['weights_nz']:>8d} | "
              f"{b_e['depth']:>4d}/{b_e['weights_nz']:>8d} | {pick:>4s} | "
              f"{a_e['res_nibbles']:>6d} | {ratio:>10.2%} | {regime}")
    print("=" * 92)


def narrowing_by_inwidth(b: int = 1000, widths=(4, 8, 12, 16, 24, 32),
                         n_random: int = 500, verbose: bool = True) -> dict:
    """Show how bounding the INPUT width shrinks construction A (fewer partials,
    fewer result nibbles, shorter carry-resolve) — verified byte-exact over ``a``
    values that RESPECT the bound.  This is the input-width narrowing lever the
    brief asks for."""
    M = 0xFFFFFFFF
    out = {}
    if verbose:
        print(f"\nINPUT-WIDTH NARROWING (construction A, b={b}):")
        print(f"{'in_width':>8s} | {'depth':>5s} | {'nz':>7s} | {'partials':>8s} | "
              f"{'resnib':>6s} | byte-exact (a < 2^in_width)")
        print("-" * 74)
    for w in widths:
        L = _make_layout()
        blocks, res, info = build_linear_fold(L, DIM, b, in_width=w)
        m = measure(blocks)
        rng = random.Random(1234 ^ w)
        cap = (1 << w) - 1
        ok = tot = 0
        for _ in range(n_random):
            a_val = rng.randint(0, cap)
            got = simulate_sparse(L, blocks, res, a_val)
            exp = (a_val * b) & M
            tot += 1
            ok += (got == exp)
        # edges within the bound
        for a_val in {0, 1, cap, cap - 1, cap // 2}:
            got = simulate_sparse(L, blocks, res, a_val)
            exp = (a_val * b) & M
            tot += 1
            ok += (got == exp)
        out[w] = {**m, "n_partials": info.get("n_partials"),
                  "res_nibbles": info.get("res_nibbles"),
                  "byte_exact": (ok, tot)}
        if verbose:
            print(f"{w:>8d} | {m['depth']:>5d} | {m['weights_nz']:>7d} | "
                  f"{info.get('n_partials'):>8d} | {info.get('res_nibbles'):>6d} | "
                  f"{ok}/{tot}")
    return out


def _verify_summary(results: dict) -> Tuple[int, int]:
    per_b = results["per_b"]
    tot_pass = tot = 0
    for b in BATTERY:
        for tag in ("A_linear", "B_shiftadd"):
            e = per_b[b][tag]
            tot_pass += e["byte_exact_pass"]
            tot += e["byte_exact_total"]
    return tot_pass, tot


if __name__ == "__main__":
    res = run_bakeoff()
    _print_narrowing_table(res)
    narrowing_by_inwidth(b=1000)
    narrowing_by_inwidth(b=255)
    p, t = _verify_summary(res)
    print(f"\nBYTE-EXACT total across both constructions x all b: {p}/{t}")
    # fp32 ceiling report
    max_arg = 0
    for b in BATTERY:
        for tag in ("A_linear", "B_shiftadd"):
            max_arg = max(max_arg, res["per_b"][b][tag]["max_staircase_arg"])
    print(f"fp32: 0 fp64.  max relu staircase argument = {max_arg}  "
          f"(RELU_S*arg = {RELU_S*max_arg:.0f} < 2^24 = {FP32_INT_MAX}): "
          f"{'PASS' if RELU_S*max_arg < FP32_INT_MAX else 'FAIL'}")
