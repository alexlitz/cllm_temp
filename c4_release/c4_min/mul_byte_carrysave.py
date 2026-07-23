"""BYTE MULTIPLIER, CARRY-SAVE ACCUMULATION, TUNABLE RELU_S — head-to-head.

Builds and MEASURES a 32-bit ``(a*b) & 0xFFFFFFFF`` multiplier as persistent
SwiGLU FFN blocks, COMPOSED from the proven ``nibble_alu32`` primitives
(``_mul_gate``, ``_empty_spec``, ``_ident``, ``_clear``, ``_truncate``, ``S``).
This file NEVER edits those; it also does not touch ``mul_bakeoff.py`` /
``mul_bakeoff_11bit.py`` (owned elsewhere).  It re-derives ONE thing on its own:
a **tunable-RELU_S** tripwire/peel, so the RELU_S knob is an explicit per-gadget
parameter with an fp32-headroom assertion — the whole reason for the design.

THE SUBSTRATE FACT (why RELU_S is a knob, not the fixed 200)
============================================================
The ``relu`` under a transformer is realised as ``silu(RELU_S*z)/RELU_S``, and a
tripwire ``[x >= t]`` is a difference of two such units,
``silu(RELU_S*(x-t+1))/RELU_S - silu(RELU_S*(x-t))/RELU_S``.  Two constraints pin
RELU_S from opposite sides:

  * SHARPNESS (lower bound).  The unit step needs ``silu(RELU_S*1) ~= RELU_S`` so
    the 0->1 transition is crisp on integers; empirically RELU_S >= ~20 gives a
    bit-exact step (RELU_S=20/32/64 all yield step(x=t)=1.000000, step(x=t-1)=0;
    RELU_S=8 already leaks ~3.3e-4, RELU_S=4 leaks ~1.8e-2).

  * fp32 EXACTNESS (upper bound).  The SiLU argument AND the baked bias
    ``RELU_S*t`` are fp32 numbers; fp32 holds integers exactly only below 2^24.
    So EVERY gadget is exact iff ``RELU_S * max(|x|, |t|) < 2^24``.  The max exact
    argument is ``2^24 / RELU_S`` (RELU_S=200 -> 83886; RELU_S=64 -> 262144;
    RELU_S=32 -> 524288).

Every gadget here is built with an explicit ``relu_s=`` and ASSERTs
``relu_s * max_arg < 2^24`` (``_assert_headroom``), reporting the tightest ratio.

THE DESIGN — byte multiplier, carry-save (the point of this file)
================================================================
Larger (byte) partial products, accumulate WITHOUT per-product splitting,
decompose the columns ONCE:

  1. **4 byte-chunks** per 32-bit operand.  **10 byte-products** ``a_i*b_j``
     (i+j<4), each an 8x8 = 16-bit product ``<= 65025``, via ``_mul_gate`` (its
     internal ``silu(S*a)*b ~ S*a*b <= 3.9M < 2^24`` for byte operands -> exact).
  2. **Accumulate into 4 byte-position columns** (column c = weight 256^c), NO
     per-product split.  Column c gathers the products with ``i+j == c``, so a
     column holds at most 4 products -> ``<= 4*65025 = 260100``.  This is the
     carry-SAVE step: partial products are summed raw into their weighted column,
     no intermediate normalisation.
  3. **Decompose each 260100-column into nibbles ONCE**, MSB-first, at a LOW
     RELU_S (<= 64, so ``RELU_S*260100 < 2^24``).  Crucially the peel uses a
     TIGHT per-power kmax so no baked threshold ``k*2^p`` ever exceeds the column
     value (a naive kmax=15 at the top nibble would bake ``RELU_S*15*65536``, a
     bias fp32 cannot represent — the subtle trap).  Then the 4 columns are
     combined at their byte offsets (column c -> nibbles 2c..2c+3) with carries
     into the 8 result nibbles.

MSB-first peel = for the largest remaining nibble-power ``2^p``,
``n = floor(x/2^p) in 0..kmax``, ``x -= n*2^p``, repeat down; each digit
``n = sum_{k=1..kmax} [x >= k*2^p]`` with the tunable-RELU_S unit step.  kmax is
15 for every power except the top, where it is ``floor(colmax/2^p)`` (tight).

fp32 verdict (measured, not hoped): RELU_S=64 uses 99.22% of the 2^24 ceiling
(0.78% headroom — real but FRAGILE); RELU_S=32 uses 49.61% (comfortable ~2x
headroom) and is STILL a bit-exact step.  We default to RELU_S=32 and report both.

HEAD-TO-HEAD (same lean harness):
  * nibble baseline    — wraps the production ``compile_mul_blocks`` (36 nibble
    products, 8 columns < 256, 7 carry rounds; every arg <= 225 -> deep fp32).
  * nibble Dadda CSA   — 3:2 nibble compressors (each == one base-16 carry round)
    on Dadda's log-depth schedule + a final carry-propagate.
  * byte carry-save    — THIS design.

MEASUREMENT is lean (seconds, no dense DIM forward): DEPTH = #blocks, WEIGHTS =
sum of nonzeros, TIGHTEST ``RELU_S*arg`` as a fraction of 2^24 (fp32 headroom),
and BYTE-EXACT via a sparse per-band arithmetic sim (``_sparse_apply``) over
~200 random + structured (a,b) pairs.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .nibble_vm_layout import NibbleVMLayout
from . import nibble_alu32 as A
from .nibble_vm import S, _empty_spec
from .nibble_alu32 import (
    _mul_gate, _ident, _clear, _truncate,
    compile_mul_blocks, extend_layout_for_alu32,
)

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

FP32_INT_MAX = 1 << 24                       # 2^24: fp32 exact-integer ceiling
NCOL = 8                                     # 8 nibble result columns (32-bit)

# RELU_S choices for the wide byte-column peel.  Column max = 4*65025 = 260100.
#   64 * 260100 = 16_646_400  < 2^24 (headroom 0.78%)  -> tight but exact
#   32 * 260100 =  8_323_200  < 2^24 (headroom 50.4%)  -> ~2x headroom, still sharp
RELU_S_TIGHT = 64.0
RELU_S_SAFE = 32.0
COLUMN_MAX = 4 * 255 * 255                    # 260100

# every gadget records (label, relu_s, max_arg) so the harness can report the
# single tightest RELU_S*arg across the whole gadget.
_HEADROOM_LOG: List[Tuple[str, float, int]] = []


def _reset_headroom():
    _HEADROOM_LOG.clear()


def _assert_headroom(label: str, relu_s: float, max_arg: int):
    """ASSERT the tunable-RELU_S fp32 discipline for one gadget and log it.

    The SiLU argument and the baked bias ``RELU_S*t`` are fp32 numbers; both must
    be integer-exact, i.e. ``RELU_S * max(|x|,|t|) < 2^24``.  ``max_arg`` is the
    largest magnitude (value OR threshold) the gadget feeds; we assert and record
    it so the bakeoff can surface the tightest ratio."""
    prod = relu_s * max_arg
    assert prod < FP32_INT_MAX, (
        f"{label}: RELU_S({relu_s})*max_arg({max_arg})={prod:.0f} "
        f">= 2^24={FP32_INT_MAX} -> fp32-LOSSY; lower RELU_S")
    _HEADROOM_LOG.append((label, relu_s, max_arg))


def tightest_headroom() -> dict:
    """The single tightest ``RELU_S*arg`` over all logged gadgets: the fp32
    fraction used (of 2^24) and the residual headroom."""
    if not _HEADROOM_LOG:
        return {"tightest_ratio": 0.0, "headroom": 1.0, "label": None,
                "relu_s": None, "max_arg": 0, "product": 0}
    label, relu_s, max_arg = max(_HEADROOM_LOG, key=lambda t: t[1] * t[2])
    prod = relu_s * max_arg
    return {"tightest_ratio": prod / FP32_INT_MAX, "headroom": 1.0 - prod / FP32_INT_MAX,
            "label": label, "relu_s": relu_s, "max_arg": max_arg, "product": prod}


# ===========================================================================
# Private scratch allocation (never touches ALU32Bands; own band range so the
# sparse sim reads/writes only this design's bands).
# ===========================================================================
def _scratch(L, name, size):
    key = f"BCS_{name}"
    if key in L._names:
        return L._names[key][0]
    return L._band(key, size)


# ===========================================================================
# TUNABLE-RELU_S tripwire / peel primitives.  These re-derive the relu-via-silu
# unit step with RELU_S as an explicit argument (the library ``_step_ge`` hard-
# codes RELU_S=200 and a 0.25 ramp).  ``[x >= t] = relu(x-t+1) - relu(x-t)`` for
# integer x,t: two silu-relu units whose difference is a bit-exact 0/1 step.
# ===========================================================================
def _relu_rs(spec, u, terms, const, dst, scale, relu_s):
    """dst += scale * relu(terms.x + const) at the given RELU_S.  up = RELU_S*form
    (silu-relu), gate = ONE, W_down carries form*scale (÷RELU_S undoes the SiLU
    gain)."""
    for band, coeff in terms.items():
        spec["W_up"][u, band] += relu_s * coeff
    spec["b_up"][u] += relu_s * const
    spec["W_gate"][u, A._ONE] = 1.0
    spec["W_down"][dst, u] += scale / relu_s
    return u + 1


def _step_ge_rs(spec, u, src_band, t, dst, scale, relu_s):
    """dst += scale * [src >= t], integer t, at RELU_S.
    = relu(src - t + 1) - relu(src - t).  The baked biases are RELU_S*(t-1) and
    RELU_S*t, so the CALLER must have asserted RELU_S*t < 2^24."""
    u = _relu_rs(spec, u, {src_band: 1.0}, float(-t + 1), dst, scale, relu_s)
    u = _relu_rs(spec, u, {src_band: 1.0}, float(-t), dst, -scale, relu_s)
    return u


_PEEL_W = 0.25          # sharp-ramp half-width for the MSB-first digit tripwire


def _digit_msb(spec, u, src_band, p, kmax, dst_digit, relu_s):
    """SET ``dst_digit = floor(src / 2^p) = sum_{k=1..kmax}[src >= k*2^p]`` (0..kmax)
    via kmax SHARP half-integer ramps (threshold ``k*2^p - 0.5``, width 0.25).

    The ramp (NOT the integer unit step) is load-bearing: the peel residue that
    feeds ``src`` is NOT a clean integer (it carries ~0.03 fp from the wide byte
    products), and an integer unit-step ``silu(RS(x-t+1))-silu(RS(x-t))`` is exact
    only for INTEGER x — on a noisy 1791.01 it under/over-counts.  The half-integer
    ramp flips at ``k*2^p - 0.5``, safely between achievable near-integer forms.
    kmax MUST be tight: the largest baked threshold ``kmax*2^p`` must stay within
    the value's range so ``RELU_S*threshold`` does not blow the fp32 budget."""
    m = 1 << p
    w = _PEEL_W
    u = _clear(spec, u, dst_digit)
    for k in range(1, kmax + 1):
        c = k * m - 0.5
        u = _relu_rs(spec, u, {src_band: 1.0}, -(c - w), dst_digit, 1.0 / w, relu_s)
        u = _relu_rs(spec, u, {src_band: 1.0}, -c, dst_digit, -1.0 / w, relu_s)
    return u


_SNAP_W = 0.25          # sharp-ramp half-width for the nibble-snap (rounds residue)
# The snap acts ONLY on nibble values in [~-0.5, 15.5] (thresholds <= 14.5), so it
# uses a HIGH RELU_S regardless of the peel's low column-RELU_S: sharpness here is
# free (RELU_S*14.5 = 2900 << 2^24, oceans of headroom) and a bit-PERFECT snap is
# what makes the subsequent ``digit * 2^p`` residue subtract lossless.  At the peel
# RELU_S=32 the snap would leave ~3e-4 residue that ``x 65536`` re-amplifies to ~23;
# at RELU_S=200 the snap is exact, so the low-RELU_S peel stays correct.
_SNAP_RELU_S = 200.0


def _snap_nibble(spec, u, band):
    """SET band := round(band) for a nibble value in [~-0.5, ~15.5] at the HIGH snap
    RELU_S, clearing the fp residue the wide byte products / column peel leave.
    round(v) = sum_{k=1..15} sharp_ramp[v >= k-0.5]: a unit ramp centred at each
    half-integer (safely between achievable nibble values), giving a clean integer
    0..15.  All thresholds <= 14.5 -> RELU_S*14.5 tiny -> deep fp32 headroom, so it
    is NOT the tightest gadget (the wide column peel is)."""
    _assert_headroom("snap", _SNAP_RELU_S, 15)         # max threshold 14.5 -> ~15
    u = _clear(spec, u, band)
    w = _SNAP_W
    for k in range(1, 16):
        c = k - 0.5
        u = _relu_rs(spec, u, {band: 1.0}, -(c - w), band, 1.0 / w, _SNAP_RELU_S)
        u = _relu_rs(spec, u, {band: 1.0}, -c, band, -1.0 / w, _SNAP_RELU_S)
    return u


def _column_peel_blocks(L, dim, col_band, colmax, out_nibs, scratch_r, tag,
                        relu_s) -> List[Block]:
    """Decompose ONE accumulated column ``x = col_band`` (0..colmax) into nibbles
    at ``out_nibs+0.. (LSB-first)``, MSB-first, at the tunable RELU_S — ONCE.

    Peel powers are nibble-aligned (p = 4*(n_nib-1) .. 0).  kmax per power is
    TIGHT: at the top power it is ``floor(colmax/2^p)`` (so the largest baked
    threshold stays <= colmax), and 15 for every lower power (a shrunk residue is
    < 16*2^p there).  Each peel step is its OWN block: within a SwiGLU block every
    unit reads the block INPUT, so the residue a step subtracts is only visible to
    the NEXT block — the peel is inherently sequential.

    ASSERTS ``relu_s * colmax < 2^24`` (the binding argument is the column value
    itself once kmax is tight, since no threshold exceeds colmax)."""
    n_nib = max(1, (colmax.bit_length() + 3) // 4)
    _assert_headroom(f"{tag}-colpeel", relu_s, colmax)
    blocks: List[Block] = []
    # block 0: residue <- column value.
    s = _empty_spec(dim, 2)
    u = _clear(s, 0, scratch_r)
    u = _ident(s, u, {col_band: 1.0}, 0.0, scratch_r, 1.0)
    blocks.append((f"{tag}-init", _truncate(s, u, dim)))
    for idx in range(n_nib - 1, -1, -1):
        p = 4 * idx
        m = 1 << p
        top = (idx == n_nib - 1)
        kmax = min(15, colmax // m) if top else 15
        kmax = max(1, kmax)
        # the largest baked threshold at this power; must stay <= colmax (tight).
        _assert_headroom(f"{tag}-p{p}", relu_s, kmax * m)
        # block A: digit = floor(residue / 2^p) into the nibble slot (sharp ramp).
        s = _empty_spec(dim, 1 + kmax * 2)
        u = _digit_msb(s, 0, scratch_r, p, kmax, out_nibs + idx, relu_s)
        blocks.append((f"{tag}-n{idx}", _truncate(s, u, dim)))
        # block B: SNAP the digit to a clean integer 0..kmax (own block: units read
        # the block input, so the digit must already be materialised).  This is the
        # crux of the fp discipline: the raw ramp digit carries a ~3e-4 residue, and
        # the NEXT block multiplies the digit by 2^p (up to 65536) for the residue
        # subtract — 3e-4 x 65536 = 22.9 of spurious residue that flips a low nibble.
        # Snapping to an EXACT integer first makes the x2^p multiply lossless.
        s = _empty_spec(dim, 1 + 15 * 2)
        u = _snap_nibble(s, 0, out_nibs + idx)
        blocks.append((f"{tag}-s{idx}", _truncate(s, u, dim)))
        # block C: residue -= digit * 2^p (a LINEAR _ident reading the CLEAN integer
        # digit, so the x2^p coefficient multiplies an exact integer -> no residue
        # amplification).  Only when a lower power remains (top nibble has none).
        if idx > 0:
            s = _empty_spec(dim, 1)
            u = _ident(s, 0, {out_nibs + idx: -float(m)}, 0.0, scratch_r, 1.0)
            blocks.append((f"{tag}-r{idx}", _truncate(s, u, dim)))
    return blocks


# ===========================================================================
# CARRY-NORMALISE ROUND (shared by the byte-column combine and the nibble
# contenders).  ONE base-16 round on ``n`` columns kept < 256:
#     dst[c] = src[c] mod 16 + floor(src[c-1]/16)
# realised with a shared floor(col/16) staircase.  Columns < 256 -> kmax=15, the
# modulus is 16 -> RELU_S*threshold = RELU_S*15*16 = RELU_S*240 (tiny), oceans of
# fp32 headroom at any RELU_S.
#
# CRUCIAL ROBUSTNESS DETAIL: the floor staircase uses a SHARP HALF-INTEGER RAMP
# (threshold k*16 - 0.5, width 0.25), NOT the integer unit step.  A carry chain
# does not deliver clean integers — a lane can arrive at ~15.96 (a carry of ~0.96
# from a slightly-imprecise floor below), and the integer unit-step
# ``silu(RS(x-t+1))-silu(RS(x-t))`` (which is only exact for INTEGER x) then floors
# 15.96 to 15 and STARVES the carry, so residue compounds up the ripple and flips
# a result nibble.  The half-integer ramp flips at 15.5, safely between the
# achievable near-integer values, so a 15.96 correctly carries.  This is exactly
# why the library ``_floor_div_pow`` uses ``thr-0.5`` (see nibble_alu32._step_ge).
# ===========================================================================
_CARRY_RELU_S = 200.0
_CARRY_RAMP_W = 0.25


def _floor16_shared(spec, u, src, dst_mod, dst_carry, relu_s):
    """From ONE floor(src/16) staircase (kmax=15, SHARP half-integer ramps), route
    -16*floor into ``dst_mod`` (=> src mod 16) and +floor into ``dst_carry`` (the
    carry up).  ``dst_carry`` may be None (top column).  Each floor step is
    ``[src >= k*16]`` as a unit ramp centred at ``k*16 - 0.5`` (width 0.25), immune
    to the ~0.1 fp residue a carry chain carries."""
    w = _CARRY_RAMP_W
    for k in range(1, 16):
        c = k * 16 - 0.5
        # [src >= k*16] ~= (relu(src-(c-w)) - relu(src-c))/w, routed to mod (-16*)
        # and carry (+1*) from the SAME two relu units.
        u = _relu_rs(spec, u, {src: 1.0}, -(c - w), dst_mod, -16.0 / w, relu_s)
        u = _relu_rs(spec, u, {src: 1.0}, -c, dst_mod, 16.0 / w, relu_s)
        if dst_carry is not None:
            u = _relu_rs(spec, u, {src: 1.0}, -(c - w), dst_carry, 1.0 / w, relu_s)
            u = _relu_rs(spec, u, {src: 1.0}, -c, dst_carry, -1.0 / w, relu_s)
    return u


def _carry_round_block(L, dim, src, dst, n, tag) -> Block:
    """One shared-staircase base-16 carry round on ``n`` columns (< 256)."""
    _assert_headroom(f"{tag}-carry", _CARRY_RELU_S, 15 * 16)
    spec = _empty_spec(dim, n * (2 + 15 * 2 + 15 * 2))
    u = 0
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)          # + col
        carry = (dst + c + 1) if c + 1 < n else None
        u = _floor16_shared(spec, u, src + c, dst + c, carry, _CARRY_RELU_S)
    return _truncate(spec, u, dim)


def _result_copy_block(L, dim, src, res) -> Block:
    copy = _empty_spec(dim, NCOL * 2)
    u = 0
    for c in range(NCOL):
        u = _clear(copy, u, res + c)
        u = _ident(copy, u, {src + c: 1.0}, 0.0, res + c, 1.0)
    return _truncate(copy, u, dim)


# ===========================================================================
# THE BYTE CARRY-SAVE MULTIPLIER.
# ===========================================================================
def build_byte_carrysave(L, dim, relu_s: float = RELU_S_SAFE) -> Tuple[List[Block], int, dict]:
    """4 byte chunks -> 10 byte products -> accumulate into 4 byte-columns (NO
    per-product split) -> peel each column ONCE at the tunable RELU_S -> combine
    at byte offsets with carries -> 8 result nibbles.

    ``relu_s`` (default 32) is the wide-column peel scale; the byte-column max is
    260100 so relu_s must be <= 64.  The carry rounds use RELU_S=200 (arg<=240,
    oceans of headroom) — only the column peel needs the low RELU_S."""
    A._ONE = L.ONE
    L_STACK0, L_AX = L.STACK0, L.AX
    assert relu_s * COLUMN_MAX < FP32_INT_MAX, (
        f"relu_s={relu_s} too large for byte columns (max {COLUMN_MAX}); need "
        f"relu_s < {FP32_INT_MAX / COLUMN_MAX:.1f}")

    RES = _scratch(L, "RES", NCOL)
    CHUNK_A = _scratch(L, "CA", 4)
    CHUNK_B = _scratch(L, "CB", 4)
    PROD = _scratch(L, "PROD", 10)
    # 4 byte-position columns (col c = product weight 256^c), each <= 260100.
    COLS = _scratch(L, "COLS", 4)
    # each column peels into up to 5 nibbles (260100 < 2^18 -> 5 nibbles).
    CNIBS = _scratch(L, "CNIBS", 4 * 5)
    RSCR = _scratch(L, "RSCR", 4)              # per-column peel residue scratch
    # the byte-offset combine columns: column c contributes to result nibbles
    # 2c,2c+1,2c+2,2c+3.  We gather into 9 raw nibble columns (<256) then carry.
    MCOL = _scratch(L, "MCOL", NCOL + 1)

    pairs = [(i, j) for i in range(4) for j in range(4) if i + j < 4]  # 10 products

    blocks: List[Block] = []
    # (1) nibble lanes -> byte chunks: chunk c = nib[2c] + 16*nib[2c+1].
    spec = _empty_spec(dim, 4 * 4)
    u = 0
    for c in range(4):
        u = _clear(spec, u, CHUNK_A + c)
        u = _ident(spec, u, {L_STACK0 + 2 * c: 1.0, L_STACK0 + 2 * c + 1: 16.0},
                   0.0, CHUNK_A + c, 1.0)
        u = _clear(spec, u, CHUNK_B + c)
        u = _ident(spec, u, {L_AX + 2 * c: 1.0, L_AX + 2 * c + 1: 16.0},
                   0.0, CHUNK_B + c, 1.0)
    blocks.append(("bcs-chunks", _truncate(spec, u, dim)))

    # (2) 10 byte products a_i*b_j (8-bit multiplicand -> _mul_gate exact; internal
    #     silu(S*a)*b ~ S*a*b <= 3.9M < 2^24).
    _assert_headroom("bcs-mulgate", S, 255 * 255)   # gate value <=255, up=S*a<=15300
    spec = _empty_spec(dim, 10 * 3)
    u = 0
    for idx, (i, j) in enumerate(pairs):
        u = _clear(spec, u, PROD + idx)
        u = _mul_gate(spec, u, CHUNK_A + i, CHUNK_B + j, PROD + idx, 1.0)
    blocks.append(("bcs-products", _truncate(spec, u, dim)))

    # (3) ACCUMULATE into 4 byte columns (col c = sum of products with i+j==c).
    #     NO per-product split — the carry-SAVE step.  Column c <= 4*65025.
    spec = _empty_spec(dim, 4 + 10)
    u = 0
    for c in range(4):
        u = _clear(spec, u, COLS + c)
    for idx, (i, j) in enumerate(pairs):
        u = _ident(spec, u, {PROD + idx: 1.0}, 0.0, COLS + (i + j), 1.0)
    blocks.append(("bcs-accum", _truncate(spec, u, dim)))

    # (4) DECOMPOSE each column to nibbles ONCE, MSB-first, at the tunable RELU_S.
    #     Each peel step snaps its digit to a clean integer inline (see
    #     _column_peel_blocks), so the CNIBS nibbles are already exact integers.
    for c in range(4):
        blocks += _column_peel_blocks(L, dim, COLS + c, COLUMN_MAX,
                                      CNIBS + 5 * c, RSCR + c, f"bcs-col{c}", relu_s)

    # (5) COMBINE the 4 column-nibble sets at their byte offsets (column c weight
    #     256^c -> starts at result nibble 2c) into 9 raw nibble columns (<256),
    #     then carry-normalise.  Column c has up to 5 nibbles at 2c..2c+4.
    spec = _empty_spec(dim, (NCOL + 1) + 4 * 5)
    u = 0
    for k in range(NCOL + 1):
        u = _clear(spec, u, MCOL + k)
    for c in range(4):
        base = 2 * c
        for k in range(5):
            slot = base + k
            if slot <= NCOL:                          # keep low 32 bits + 1 carry lane
                u = _ident(spec, u, {CNIBS + 5 * c + k: 1.0}, 0.0, MCOL + slot, 1.0)
    blocks.append(("bcs-combine", _truncate(spec, u, dim)))

    # (6) carry rounds settle MCOL (raw nibble sums < ~ a few*15 per lane) to single
    #     nibbles.  A carry ripples one lane per round; NCOL+1 lanes, <=NCOL rounds.
    src, dst = MCOL, _scratch(L, "MCOL2", NCOL + 1)
    for r in range(NCOL):
        blocks.append((f"bcs-carry{r}", _carry_round_block(L, dim, src, dst, NCOL + 1, "bcs")))
        src, dst = dst, src
    blocks.append(("bcs-result", _result_copy_block(L, dim, src, RES)))

    info = {
        "note": "4 byte chunks -> 10 byte products -> 4 byte-columns (NO split) -> "
                "peel each 260100-column ONCE at tunable RELU_S -> byte-offset combine",
        "n_products": 10, "n_columns": 4, "column_max": COLUMN_MAX,
        "relu_s_peel": relu_s, "max_staircase_arg": COLUMN_MAX,
        "peel_headroom_ratio": (relu_s * COLUMN_MAX) / FP32_INT_MAX,
    }
    return blocks, RES, info


# ===========================================================================
# CONTENDER 1 — nibble baseline (wrap the production compile_mul_blocks).
# ===========================================================================
def build_nibble_baseline(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    blocks = list(compile_mul_blocks(L, dim))
    # the baseline's staircases are the library's (RELU_S=200), all args <= 232.
    _assert_headroom("nibble-baseline", 200.0, 232)
    info = {
        "note": "production nibble schoolbook (compile_mul_blocks): 36 nibble "
                "products, 8 columns < 256, 7 carry rounds; every arg <= 232",
        "n_products": 36, "carry_rounds": 7, "max_staircase_arg": 232,
        "relu_s_peel": 200.0,
    }
    return blocks, L.ALU32.MUL_RES, info


# ===========================================================================
# CONTENDER 2 — nibble Dadda carry-save.  After the production products+split
# front-end the 8 nibble columns have heights 1..8; 3:2 nibble compressors (each
# == one base-16 carry round) reduce them on Dadda's log schedule, then a final
# carry-propagate.  All args <= 232 -> deep fp32 at any RELU_S.
# ===========================================================================
def _dadda_targets(max_h: int) -> List[int]:
    seq = [2]
    while seq[-1] < max_h:
        seq.append(seq[-1] * 3 // 2)
    return [d for d in seq if d < max_h][::-1]


def build_nibble_dadda(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    a = L.ALU32
    base = list(compile_mul_blocks(L, dim))
    front = base[:2]                            # products + split -> a.MCOL
    RES = _scratch(L, "DA_RES", NCOL)
    _assert_headroom("dadda", 200.0, 232)
    # column heights after split (low nibble of a_i*b_j into col i+j, high into i+j+1).
    heights = [0] * NCOL
    for i in range(NCOL):
        for j in range(NCOL):
            if i + j < NCOL:
                heights[i + j] += 1
                if i + j + 1 < NCOL:
                    heights[i + j + 1] += 1
    max_h = max(heights)
    targets = _dadda_targets(max_h)
    n_stages = len(targets) + 1                  # reduction stages + final CPA
    blocks: List[Block] = list(front)
    src, dst = a.MCOL, a.MC1
    for st in range(n_stages):
        blocks.append((f"dadda-stage{st}", _carry_round_block(L, dim, src, dst, NCOL, "dadda")))
        src, dst = dst, src
    blocks.append(("dadda-result", _result_copy_block(L, dim, src, RES)))
    info = {
        "note": "3:2 nibble compressors (base-16 rounds) on Dadda's log schedule "
                "+ final CPA; every arg <= 232",
        "n_products": 36, "col_heights": heights, "max_height": max_h,
        "dadda_targets": targets, "reduction_stages": len(targets),
        "total_stages_incl_cpa": n_stages, "max_staircase_arg": 232,
        "relu_s_peel": 200.0,
    }
    return blocks, RES, info


# ===========================================================================
# MEASUREMENT — depth / weights, and a SPARSE byte-exact sim (no dense forward).
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


def _sparse_apply(state: Dict[int, float], spec: Spec, dtype=torch.float32):
    """Apply ONE SwiGLU block's ``down(silu(up)*gate)`` over just the units that
    read a currently-nonzero band — the EXACT dense math, no DIM x DIM matmul."""
    W_up = spec["W_up"]; b_up = spec["b_up"]
    W_gate = spec["W_gate"]; b_gate = spec["b_gate"]
    W_down = spec["W_down"]; b_down = spec["b_down"]
    up = b_up.to(dtype).clone()
    gate = b_gate.to(dtype).clone()
    for band, val in state.items():
        cu = W_up[:, band]
        if torch.count_nonzero(cu):
            up += cu.to(dtype) * val
        cg = W_gate[:, band]
        if torch.count_nonzero(cg):
            gate += cg.to(dtype) * val
    hidden = F.silu(up) * gate
    delta = W_down.to(dtype) @ hidden + b_down.to(dtype)
    for d in torch.nonzero(delta, as_tuple=False).flatten().tolist():
        state[d] = state.get(d, 0.0) + float(delta[d])
    return state


def simulate_sparse(L, blocks: List[Block], res_band: int, a_val: int, b_val: int,
                    dtype=torch.float32) -> int:
    state: Dict[int, float] = {L.ONE: 1.0}
    for c in range(8):
        state[L.STACK0 + c] = float((a_val >> (4 * c)) & 0xF)
        state[L.AX + c] = float((b_val >> (4 * c)) & 0xF)
    for _n, s in blocks:
        state = _sparse_apply(state, s, dtype=dtype)
    out = 0
    for c in range(8):
        out |= (int(round(state.get(res_band + c, 0.0))) & 0xF) << (4 * c)
    return out & 0xFFFFFFFF


def test_operands(n_random: int = 180, seed: int = 0) -> List[Tuple[int, int]]:
    """~200 (a,b): structured edges (0, 0xFFFFFFFF, 0xDEADBEEF, powers of two,
    single-byte, full-width) + random."""
    rng = random.Random(seed)
    M = 0xFFFFFFFF
    edges = [0, 1, 2, 3, M, M - 1, 0xDEADBEEF, 0xCAFEBABE, 0x80000000,
             0x7FFFFFFF, 0xFFFF, 0x10000, 0xABCD, 0x1234, 65535, 65536,
             255, 256, 0x00FF00FF, 0xFF00FF00]
    pows = [1 << k for k in range(0, 32, 3)]
    singles = [0x000000AB, 0x0000CD00, 0x00EF0000, 0x12000000]  # single-byte lanes
    pairs: List[Tuple[int, int]] = []
    for a in edges + pows + singles:
        for b in [0, 1, M, 0xDEADBEEF, 3, 0x10000, 255]:
            pairs.append((a & M, b & M))
    for _ in range(n_random):
        pairs.append((rng.randint(0, M), rng.randint(0, M)))
    return pairs


DIM = 4096              # wide enough for every variant's private scratch bands.


def _make_layout():
    L = NibbleVMLayout(8, n_heads=4)
    extend_layout_for_alu32(L)
    A._ONE = L.ONE
    return L


# ===========================================================================
# BAKEOFF DRIVER
# ===========================================================================
def run_bakeoff(n_random: int = 180, verbose: bool = True,
                relu_s_variants=(RELU_S_SAFE, RELU_S_TIGHT)) -> dict:
    results: dict = {}
    L = _make_layout()
    dim = DIM
    M = 0xFFFFFFFF
    ops = test_operands(n_random=n_random)

    builders: List[Tuple[str, callable, dict]] = [
        ("nibble_baseline", build_nibble_baseline, {}),
        ("nibble_dadda", build_nibble_dadda, {}),
    ]
    for rs in relu_s_variants:
        builders.append((f"byte_carrysave_rs{int(rs)}",
                         lambda L, dim, rs=rs: build_byte_carrysave(L, dim, relu_s=rs), {}))

    for name, fn, _ in builders:
        _reset_headroom()
        blocks, res, info = fn(L, dim)
        assert L.D <= dim, f"layout grew past DIM ({L.D} > {dim}); raise DIM"
        head = tightest_headroom()
        m = measure(blocks)
        n_pass = 0
        first_fail = None
        for a, b in ops:
            got = simulate_sparse(L, blocks, res, a, b, dtype=torch.float32)
            exp = (a * b) & M
            if got == exp:
                n_pass += 1
            elif first_fail is None:
                first_fail = (hex(a), hex(b), hex(got), hex(exp))
        results[name] = {
            **m,
            "byte_exact_pass": n_pass, "byte_exact_total": len(ops),
            "first_fail": first_fail,
            "tightest_relu_s_arg": head["product"],
            "tightest_ratio_of_2p24": head["tightest_ratio"],
            "fp32_headroom": head["headroom"],
            "tightest_gadget": head["label"],
            "tightest_relu_s": head["relu_s"], "tightest_max_arg": head["max_arg"],
            **{k: v for k, v in info.items()},
        }
        if verbose:
            print(f"{name:22s} depth={m['depth']:4d} nz={m['weights_nz']:8d} "
                  f"exact={n_pass}/{len(ops)} tightest RELU_S*arg={head['product']:.0f} "
                  f"({100*head['tightest_ratio']:.2f}% of 2^24, headroom "
                  f"{100*head['headroom']:.2f}%) fail={first_fail}")
    return results


def _print_table(results: dict):
    order = ["nibble_baseline", "nibble_dadda"] + \
            [k for k in results if k.startswith("byte_carrysave")]
    print()
    print("VARIANT                 DEPTH   WEIGHTS   TIGHTEST RELU_S*arg   fp32 HEADROOM   BYTE-EXACT")
    print("-" * 92)
    for name in order:
        r = results[name]
        print(f"{name:22s} {r['depth']:5d}  {r['weights_nz']:8d}   "
              f"{r['tightest_relu_s_arg']:12.0f} ({100*r['tightest_ratio_of_2p24']:5.2f}%)   "
              f"{100*r['fp32_headroom']:7.2f}%       "
              f"{r['byte_exact_pass']:4d}/{r['byte_exact_total']:<4d}")


if __name__ == "__main__":
    res = run_bakeoff()
    _print_table(res)
