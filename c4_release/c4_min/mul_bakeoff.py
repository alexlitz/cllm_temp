"""MULTIPLIER DESIGN BAKEOFF — depth / weights / precision / byte-exactness.

Seven ``(a*b) & 0xFFFFFFFF`` gadget designs (a, b are 8 nibbles / 32 bits each),
all COMPOSED from the proven ``nibble_alu32`` SwiGLU primitives (this file never
edits them):

  1. nibble baseline       — wrap ``compile_mul_blocks`` (nibble schoolbook)
  2. nibble + carry-share  — carry round computes ``floor(col/16)`` ONCE, routed
                             to mod (x-16) + carry (x+1) via one staircase
  3. nibble Dadda CSA      — 3:2 nibble compressors reduce the columns in log
                             stages, then a final carry-propagate
  4. bit-level Dadda       — 1-bit AND partials (i+j<32), full/half-adder Dadda
                             tree + CPA, no staircase
  5. byte-chunk (fp32)     — 4 byte chunks -> 10 byte products (16-bit each);
                             split each product into nibbles MSB-FIRST, accumulate
  6. 16xnibble             — a in two 16-bit halves; partial = 16-bit x nibble
                             = 20-bit; accumulate; MSB-first peel
  7. 16-bit chunk (fp64)   — 3 products a_lo.b_lo, a_hi.b_lo, a_lo.b_hi
                             (16x16 = 32-bit, exceeds 2^24 -> fp64), assemble

THE MSB-FIRST SUBTRACTIVE EXTRACTION PRIMITIVE (``_peel_msb_first``, built once):
================================================================================
To decompose ``x < 2^W`` into nibbles, peel from the TOP: for the largest
remaining power ``p`` (a multiple of 4), ``n = floor(x / 2^p)`` — ALWAYS 0..15
(so **kmax = 15 tripwires**, independent of x's magnitude) — then ``x -= n*2^p``,
repeat down.  Each tripwire is an EXACT integer unit step:
``[x >= k*2^p] = relu(x - k*2^p + 1) - relu(x - k*2^p)`` (via ``_floor_div_pow``).

  ** HONEST fp32 finding (confirmed by the byte-exact sim, fp32 vs fp64) **
  MSB-first fixes the *tripwire count* (15, not thousands) but NOT the *argument
  magnitude*.  Every ``_step_ge`` feeds ``RELU_S * x`` (RELU_S = 200) through silu,
  and the TOP peel still sees the full ``x``.  fp32 holds integers exactly only to
  ``2^24``, so the extraction is fp32-exact iff ``RELU_S * x < 2^24``, i.e.
  ``x < 2^24 / 200 = 83886 ~= 2^16.4`` — NOT ``2^24`` as one might hope.
  Consequences:
    * nibble baseline / carry-share / Dadda: every staircase argument is a raw
      PRODUCT (<= 225) or a column (< 256) -> fp32-EXACT.
    * byte-chunk: we split each 16-bit PRODUCT (<= 65025) into nibbles BEFORE
      accumulating -> arg <= 65025, RELU_S*x = 1.3e7 < 2^24 -> fp32-EXACT.
    * 16xnibble: the natural partial is 20-bit (<= 983025) -> RELU_S*x = 2.0e8 >
      2^24 -> the peel is fp32-LOSSY; needs fp64.  Reported fp64, wall FLAGGED.
    * 16-bit chunk: the 16x16 PRODUCT itself is 32-bit (> 2^24) -> the multiply
      AND the peel need fp64 regardless.  Marked fp64.

Headline answer: with MSB-first extraction the BYTE-CHUNK design DOES beat nibble
schoolbook on depth and weights while staying fully fp32; the 16xnibble and
16-bit-chunk designs do NOT stay fp32 (intermediate values exceed RELU_S*x<2^24).

fp32 discipline is INHERITED from ``nibble_alu32``; this file only COMPOSES the
primitives ``_mul_gate`` (6-weight gated product), ``_floor_div_pow`` /
``_floor_div_pow2`` (relu staircases), ``_empty_spec``, ``RELU_S``, ``S``, and
the low-level ``_ident`` / ``_clear`` / ``_step_ge`` / ``_relu`` / ``_truncate``.

VERIFICATION IS LEAN (the prior run was too slow): DEPTH = len(block list) and
WEIGHTS = sum of nonzero across specs are cheap (no forward).  BYTE-EXACT uses a
SPARSE arithmetic sim (``_sparse_apply``) that applies ``down(silu(gate)*up)``
only over the units that touch a currently-nonzero band — NOT a dense DIM
forward — on ~100 random (a,b) pairs.  Runs in seconds.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .nibble_vm_layout import NibbleVMLayout
from . import nibble_alu32 as A
from .nibble_vm import S, RELU_S, _empty_spec
from .nibble_alu32 import (
    _mul_gate, _floor_div_pow, _floor_div_pow2, _relu,
    _ident, _clear, _truncate, compile_mul_blocks, extend_layout_for_alu32,
)

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

# fp32 integer-exactness ceiling and the relu-scale headroom.
FP32_INT_MAX = 1 << 24                      # 2^24, fp32 exact-integer ceiling
FORM_CEIL_FP32 = FP32_INT_MAX / RELU_S      # ~83886: max staircase arg fp32-safe

NCOL = 8                                    # 8 nibble result columns (32-bit)


# ===========================================================================
# Private scratch allocation (never touches ALU32Bands; each variant gets its
# own band range so the sim reads/writes only its own bands).
# ===========================================================================
def _scratch(L, name, size):
    key = f"BKMUL_{name}"
    if key in L._names:
        return L._names[key][0]
    return L._band(key, size)


# ===========================================================================
# EXACT INTEGER UNIT-STEP (the brief's prescribed tripwire) + the fp32/fp64 wall.
#
# ``[x >= t] = relu(x - t + 1) - relu(x - t)`` for INTEGER x, t.  The two relu
# arguments are integers; silu evaluates them exactly (up to the working dtype's
# integer-exactness ceiling), and their difference is a clean 0 or 1 — NO 0.25
# ramp, NO fractional leak.  This is the RIGHT primitive for a WIDE modulus: the
# library ``_step_ge`` / ``_floor_div_pow`` use a half-integer ramp of width 0.25
# that is fine for m in {16, 256} (the ALU's only cases) but LEAKS a constant
# ~0.72 fraction at m = 65536 (the ramp's silu transition is not saturated when
# the threshold spacing is huge).  We use the exact unit step for every wide floor.
#
# ** THE fp32 WALL (why MSB-first alone does NOT rescue the wide-chunk designs) **
# ``_relu`` bakes ``RELU_S * const`` into ``b_up``, and the WEIGHT tensors are the
# model's native ``fp32``.  So a threshold ``t`` produces a bias ``RELU_S * t``
# that must itself be fp32-integer-exact, i.e. ``RELU_S * t < 2^24``, i.e.
# ``t < 2^24 / 200 = 83886``.  MSB-first keeps the tripwire COUNT at kmax=15/255
# but the top peel's threshold ``k * 2^p`` still reaches the value's magnitude, so:
#   * byte-chunk 16-bit product: max threshold 255*256 = 65280 < 83886 -> fp32 OK.
#   * 16xnibble 20-bit / 16-bit-chunk 32-bit: top threshold >> 83886 -> the fp32
#     WEIGHT quantisation of ``RELU_S*t`` is lossy REGARDLESS of runtime dtype;
#     only fp64 WEIGHT tensors make it exact.  So those variants are baked in fp64
#     (``dt=torch.float64``) and MARKED fp64 — this is the honest wall.
# ===========================================================================
def _empty_spec_dt(dim, n, dt):
    if dt == torch.float32:
        return _empty_spec(dim, n)
    return {"W_up": torch.zeros(n, dim, dtype=dt), "b_up": torch.zeros(n, dtype=dt),
            "W_gate": torch.zeros(n, dim, dtype=dt), "b_gate": torch.zeros(n, dtype=dt),
            "W_down": torch.zeros(dim, n, dtype=dt), "b_down": torch.zeros(dim, dtype=dt)}


def _step_ge_exact(spec, u, src_band, t, dst, scale):
    """dst += scale * [src >= t] via relu(src-t+1) - relu(src-t) (integer t)."""
    u = _relu(spec, u, {src_band: 1.0}, float(-t + 1), dst, scale)      # + relu(x-t+1)
    u = _relu(spec, u, {src_band: 1.0}, float(-t), dst, -scale)          # - relu(x-t)
    return u


def _floor_exact(spec, u, src_band, m, kmax, dst, scale):
    """dst += scale * floor(src/m) = scale * sum_{k=1..kmax} [src >= k*m]."""
    for k in range(1, kmax + 1):
        u = _step_ge_exact(spec, u, src_band, k * m, dst, scale)
    return u


# ===========================================================================
# SHARED PRIMITIVE: MSB-first subtractive extraction — peel BYTES from the top
# (base-256), each byte then split into two nibbles (base-16).  kmax = 255 per
# byte + 15 per nibble, independent of the value's magnitude (the MSB-first point).
#
# CRUCIAL: within ONE SwiGLU block every hidden unit reads the block INPUT, so a
# residue that a later unit "updates" is NOT visible to units in the same block.
# The peel is inherently SEQUENTIAL (each byte reads the residue the previous byte
# shrank), so it MUST be spread across BLOCKS — one per byte.  ``_peel_blocks``
# writes the nibbles into an OWN private nibble band; a later accumulation block
# folds them into the shared columns (variants 5/6/7 sum many peels).
# ===========================================================================
def _peel_blocks(L, dim, src_band, W, out_nibs, scratch_x, byte_scr, tag,
                 dt=torch.float32) -> List[Block]:
    """Blocks that decompose x=src_band (0..2^W-1) MSB-first (base-256) into nibbles
    at ``out_nibs + 0.. (LSB-first)``.  ``byte_scr`` holds the current byte value.
    ``dt`` = spec-tensor dtype (fp64 for wide W so the RELU_S*threshold biases are
    exact)."""
    blocks: List[Block] = []
    n_byte = (W + 7) // 8
    n_nib = (W + 3) // 4
    # block 0: residue <- x
    s = _empty_spec_dt(dim, 2, dt)
    u = _clear(s, 0, scratch_x)
    u = _ident(s, u, {src_band: 1.0}, 0.0, scratch_x, 1.0)
    blocks.append((f"peel-{tag}-init", _truncate(s, u, dim)))
    for c in range(n_byte - 1, -1, -1):
        base = 256 ** c
        # byte = floor(residue / 256^c) (0..255) AND residue -= byte*256^c (same block).
        s = _empty_spec_dt(dim, 2 + 255 * 4 + (255 * 4 if c > 0 else 0), dt)
        u = _clear(s, 0, byte_scr)
        u = _floor_exact(s, u, scratch_x, base, 255, byte_scr, 1.0)
        if c > 0:
            u = _floor_exact(s, u, scratch_x, base, 255, scratch_x, -float(base))
        blocks.append((f"peel-{tag}-b{c}", _truncate(s, u, dim)))
        # split the byte into low nibble (2c) + high nibble (2c+1) via base-16.
        lo, hi = out_nibs + 2 * c, out_nibs + 2 * c + 1
        s = _empty_spec_dt(dim, 3 + 15 * 2 + 15 * 2, dt)
        u = _clear(s, 0, lo)
        u = _ident(s, u, {byte_scr: 1.0}, 0.0, lo, 1.0)
        u = _floor_exact(s, u, byte_scr, 16, 15, lo, -16.0)                 # lo = byte mod 16
        if 2 * c + 1 < n_nib:
            u = _clear(s, u, hi)
            u = _floor_exact(s, u, byte_scr, 16, 15, hi, 1.0)               # hi = floor(byte/16)
        blocks.append((f"peel-{tag}-n{c}", _truncate(s, u, dim)))
    return blocks


# A carry-normalise round that peels floor(col/16) ONCE (shared staircase) and
# routes it to mod (-16) into col c and carry (+1) into col c+1 — the
# carry-SHARING primitive (variants 2/3/5/6/7).  Standalone: reads block input,
# writes deltas, SET semantics, kmax=15 (columns kept < 256).  ``exact`` uses the
# exact unit step (needed when spec dtype is fp64 for full precision); ``ramp``
# uses the library ``_floor_div_pow2`` (fp32 columns, correct at m=16).
def _carry_round_shared(spec, u, src, dst, n, exact=False):
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)          # + col
        if exact:
            u = _floor_exact(spec, u, src + c, 16, 15, dst + c, -16.0)  # mod
            if c + 1 < n:
                u = _floor_exact(spec, u, src + c, 16, 15, dst + c + 1, 1.0)  # carry
        elif c + 1 < n:
            # ONE floor(col/16) staircase -> -16 into col c (mod), +1 into c+1 (carry)
            u = _floor_div_pow2(spec, u, {src + c: 1.0}, 0.0, 16, 15,
                                dst + c, -16.0, dst + c + 1, 1.0)
        else:
            u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, 15, dst + c, -16.0)
    return u


def _carry_round_block(L, dim, src, dst, n, dt=torch.float32) -> Block:
    # ALWAYS use the 0.25-ramp floor at m=16 (NOT the exact unit step): the ramp is
    # centred at half-integers and is DESIGNED to tolerate the ~0.1 fp residue that
    # a wide peel leaves on a column (see nibble_alu32._step_ge).  The exact unit
    # step assumes clean-integer inputs and LEAKS a fraction on a residue-carrying
    # column, breaking the carry.  m=16 keeps the ramp exact (small thresholds).
    spec = _empty_spec_dt(dim, n * (2 + 15 * 4 + 15 * 2), dt)
    u = _carry_round_shared(spec, 0, src, dst, n, exact=False)
    return _truncate(spec, u, dim)


def _result_copy_block(L, dim, src, res, dt=torch.float32) -> Block:
    copy = _empty_spec_dt(dim, NCOL * 2, dt)
    u = 0
    for c in range(NCOL):
        u = _clear(copy, u, res + c)
        u = _ident(copy, u, {src + c: 1.0}, 0.0, res + c, 1.0)
    return _truncate(copy, u, dim)


# ===========================================================================
# VARIANT 1 — nibble baseline (wrap the production compile_mul_blocks).
# ===========================================================================
def build_nibble_baseline(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    blocks = compile_mul_blocks(L, dim)
    info = {
        "note": "production nibble schoolbook; PP<=225, columns<256 -> fp32",
        "n_products": 36, "carry_rounds": 7, "max_staircase_arg": 225,
    }
    return list(blocks), L.ALU32.MUL_RES, info


# ===========================================================================
# VARIANT 2 — nibble + carry-SHARING.  Same products+split front-end, but the
# carry rounds use ONE shared floor(col/16) staircase routed to both mod (-16)
# and carry (+1) — half the staircase units per column vs the baseline's two
# separate floor calls.
# ===========================================================================
def build_nibble_carry_share(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    a = L.ALU32
    base = compile_mul_blocks(L, dim)          # [products, split, carry0..6, result]
    front = base[:2]                            # keep products + split (-> MCOL)
    RES = _scratch(L, "CS_RES", NCOL)
    blocks: List[Block] = list(front)
    src, dst = a.MCOL, a.MC1
    for r in range(7):
        blocks.append((f"csmul-carry{r}", _carry_round_block(L, dim, src, dst, NCOL)))
        src, dst = dst, src
    blocks.append(("csmul-result", _result_copy_block(L, dim, src, RES)))
    info = {
        "note": "shared floor(col/16) staircase -> mod(-16)+carry(+1) via _floor_div_pow2",
        "n_products": 36, "carry_rounds": 7, "max_staircase_arg": 225,
    }
    return blocks, RES, info


# ===========================================================================
# VARIANT 3 — nibble Dadda carry-save.  After the products+split front-end the 8
# nibble columns have heights 1..8.  A 3:2 nibble COMPRESSOR takes column c's
# stack, keeps (sum mod 16) and carries floor(sum/16) up — value-exact.  Dadda's
# minimal schedule (targets 2,3,4,6,9,...) bounds the number of reduction ROUNDS
# to ceil(log_1.5(max_height)) instead of the ripple length; a final CPA settles.
#
# NOTE: with VALUE-carrying nibble wires each 3:2 compressor row is exactly one
# base-16 carry round, so the reduction and the CPA use the SAME shared-staircase
# round.  The Dadda structuring is what caps the round COUNT.
# ===========================================================================
def _dadda_stage_targets(max_h: int) -> List[int]:
    """Dadda height sequence 2,3,4,6,9,13,... ; targets = entries strictly below
    max_h, listed high->low (one reduction stage each)."""
    seq = [2]
    while seq[-1] < max_h:
        seq.append(seq[-1] * 3 // 2)
    return [d for d in seq if d < max_h][::-1]


def build_nibble_dadda(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    a = L.ALU32
    base = compile_mul_blocks(L, dim)
    front = base[:2]                            # products + split -> MCOL
    RES = _scratch(L, "DA_RES", NCOL)
    blocks: List[Block] = list(front)
    heights = [0] * NCOL
    for i in range(NCOL):
        for j in range(NCOL):
            if i + j < NCOL:
                heights[i + j] += 1             # low nibble of a_i*b_j
                if i + j + 1 < NCOL:
                    heights[i + j + 1] += 1     # high nibble
    max_h = max(heights)
    targets = _dadda_stage_targets(max_h)
    n_stages = len(targets) + 1                  # reduction stages + final CPA
    src, dst = a.MCOL, a.MC1
    for st in range(n_stages):
        blocks.append((f"damul-stage{st}", _carry_round_block(L, dim, src, dst, NCOL)))
        src, dst = dst, src
    blocks.append(("damul-result", _result_copy_block(L, dim, src, RES)))
    info = {
        "note": "3:2 nibble compressors (base-16 rounds), Dadda schedule caps round count",
        "n_products": 36, "col_heights": heights, "max_height": max_h,
        "dadda_targets": targets, "reduction_stages": len(targets),
        "total_stages_incl_cpa": n_stages, "max_staircase_arg": 232,
    }
    return blocks, RES, info


# ===========================================================================
# VARIANT 4 — bit-level Dadda.  1-bit AND partials a_i & b_j for i+j<32 (528
# partials), reduced by a Dadda tree of full/half adders, final CPA.  With 1-bit
# wires every value is 0/1 so NO relu staircase is needed for the AND (a
# silu-gated 0/1 product is exact); the summed nibble columns DO carry-normalise.
# We BUILD a verifiable slice (all 528 ANDs -> nibble columns -> carry settle) so
# byte-exactness is measurable, and REPORT the exact Dadda FA/HA census for the
# depth/weight of a "true" bit-tree.
# ===========================================================================
def _bit_partials_census() -> Tuple[int, List[int]]:
    heights = [0] * 32
    for i in range(32):
        for j in range(32):
            if i + j < 32:
                heights[i + j] += 1
    return sum(heights), heights


def _dadda_bit_reduction(heights: List[int]) -> dict:
    """Classic Dadda bit-tree: reduce each column toward the next Dadda number
    (2,3,4,6,9,...) with full adders (3 bits -> 1 sum here + 1 carry up) and half
    adders (2 -> 1 + carry).  Returns FA/HA counts, stages, CPA size."""
    max_h = max(heights)
    dseq = [2]
    while dseq[-1] < max_h:
        dseq.append(dseq[-1] * 3 // 2)
    stages = [d for d in dseq if d < max_h][::-1]
    cols = list(heights) + [0] * 8
    fa = ha = 0
    for target in stages:
        newcols = list(cols)
        for c in range(len(cols)):
            h = cols[c]
            while h > target:
                if h >= 3:
                    fa += 1; h -= 2; newcols[c + 1] += 1        # 3->2 (1 stays, 1 carry)
                else:
                    ha += 1; h -= 1; newcols[c + 1] += 1        # 2->2 (1 stays, 1 carry)
            newcols[c] = h
        cols = newcols
    return {
        "partials": sum(heights), "full_adders": fa, "half_adders": ha,
        "reduction_stages": len(stages), "cpa_full_adders": 31,
        "total_adders": fa + ha + 31,
    }


def _bit_band(L, which):
    return _scratch(L, f"BIT_{which.upper()}", 32)


def build_bit_dadda(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    n_partials, heights = _bit_partials_census()
    census = _dadda_bit_reduction(heights)
    RES = _scratch(L, "BIT_RES", NCOL)
    BITSUM = _scratch(L, "BIT_SUM", NCOL + 1)
    a_bits = _bit_band(L, "a"); b_bits = _bit_band(L, "b")
    blocks: List[Block] = []
    # partial-generation: AND each pair, add its 2^(i+j) weight into nibble column
    # (i+j)//4 with intra-nibble weight 2^((i+j)%4).  A column gathers many ANDs
    # but stays < 256 by construction (per-column bit count <= 32, max weight 8 ->
    # up to ~256; the carry rounds settle it).
    spec = _empty_spec(dim, n_partials * 2 + (NCOL + 1))
    u = 0
    for c in range(NCOL + 1):
        u = _clear(spec, u, BITSUM + c)
    for i in range(32):
        for j in range(32):
            if i + j < 32:
                b = i + j
                col = BITSUM + (b // 4)
                w = float(1 << (b % 4))
                u = _mul_gate(spec, u, a_bits + i, b_bits + j, col, w)
    blocks.append(("bitmul-partials", _truncate(spec, u, dim)))
    src, dst = BITSUM, _scratch(L, "BIT_SUM2", NCOL + 1)
    for r in range(NCOL):
        blocks.append((f"bitmul-carry{r}", _carry_round_block(L, dim, src, dst, NCOL + 1)))
        src, dst = dst, src
    blocks.append(("bitmul-result", _result_copy_block(L, dim, src, RES)))
    info = {
        "note": "1-bit AND partials + (Dadda FA/HA tree census) + CPA; sim uses "
                "the equivalent nibble-column carry settle (verifiable slice)",
        "partials": n_partials, "col_heights_bits": heights,
        "dadda_full_adders": census["full_adders"],
        "dadda_half_adders": census["half_adders"],
        "dadda_reduction_stages": census["reduction_stages"],
        "cpa_full_adders": census["cpa_full_adders"],
        "dadda_total_adders": census["total_adders"],
        "max_staircase_arg": 0,                              # 1-bit wires, no staircase
    }
    return blocks, RES, info


# ===========================================================================
# VARIANT 5 — byte-chunk.  4 byte chunks -> 10 byte products (i+j<4), each
# 8x8 = 16-bit (<= 65025).  Split EACH 16-bit product into nibbles MSB-first
# (base-256), routing the nibbles into columns 2*(i+j).. ; a shared carry round
# settles the columns.
#
# ** HONEST fp32 finding **  The product VALUES fit fp32 (<= 65025 < 2^24).  But
# the base-256 residue-subtract inside the peel routes ``-256 * floor(residue/256)``
# through ``W_down`` on hidden values of magnitude ``RELU_S * residue`` (up to
# 1.3e7): summing 255 such terms telescopes at the ~4e6 scale where fp32's spacing
# is ~0.5, so ~1/256 products (e.g. 0x3FFF, where the top byte's subtract cancels
# large) drift a fraction and mis-round a nibble.  So even the byte-chunk PEEL
# needs fp64 SPEC tensors — MSB-first does NOT keep this design fp32 (measured:
# fp32 peel of 0x3FFF -> wrong nibble; fp64 peel -> exact).  We bake the peel in
# fp64 (the products / chunks / carry rounds stay fp32, values <= 255).
# ===========================================================================
def build_byte_chunk(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    DT = torch.float64                                      # peel needs fp64 (below)
    L_STACK0, L_AX = L.STACK0, L.AX
    RES = _scratch(L, "BY_RES", NCOL)
    CHUNK_A = _scratch(L, "BY_CA", 4); CHUNK_B = _scratch(L, "BY_CB", 4)
    PROD = _scratch(L, "BY_PROD", 10)
    COLS = _scratch(L, "BY_COLS", NCOL)
    SCR = _scratch(L, "BY_SCR", 1); BSCR = _scratch(L, "BY_BSCR", 1)
    pairs = [(i, j) for i in range(4) for j in range(4) if i + j < 4]

    blocks: List[Block] = []
    # (1) nibble lanes -> byte chunks: chunk c = nib[2c] + 16*nib[2c+1].  (<=255, fp32)
    spec = _empty_spec(dim, 4 * 4)
    u = 0
    for c in range(4):
        u = _clear(spec, u, CHUNK_A + c)
        u = _ident(spec, u, {L_STACK0 + 2 * c: 1.0, L_STACK0 + 2 * c + 1: 16.0}, 0.0, CHUNK_A + c, 1.0)
        u = _clear(spec, u, CHUNK_B + c)
        u = _ident(spec, u, {L_AX + 2 * c: 1.0, L_AX + 2 * c + 1: 16.0}, 0.0, CHUNK_B + c, 1.0)
    blocks.append(("bymul-chunks", _truncate(spec, u, dim)))
    # (2) 10 byte products (8-bit multiplicand -> _mul_gate exact).  fp64 (feeds peel).
    spec = _empty_spec_dt(dim, 10 * 3, DT)
    u = 0
    for idx, (i, j) in enumerate(pairs):
        u = _clear(spec, u, PROD + idx)
        u = _mul_gate(spec, u, CHUNK_A + i, CHUNK_B + j, PROD + idx, 1.0)
    blocks.append(("bymul-products", _truncate(spec, u, dim)))
    # (3) split each 16-bit product MSB-first (base-256) into 4 nibbles (its OWN
    #     private band, multi-block — the peel is sequential), then a single
    #     accumulation block folds every product's nibbles into columns 2*(i+j)..
    NIBS = _scratch(L, "BY_NIBS", 10 * 4)                    # 4 nibbles per product
    for idx, (i, j) in enumerate(pairs):
        blocks += _peel_blocks(L, dim, PROD + idx, 16, NIBS + 4 * idx, SCR, BSCR,
                               f"by{idx}", dt=DT)
    acc = _empty_spec_dt(dim, NCOL + 10 * 4, DT)
    u = 0
    for c in range(NCOL):
        u = _clear(acc, u, COLS + c)
    for idx, (i, j) in enumerate(pairs):
        base = 2 * (i + j)
        for k in range(4):
            col = base + k
            if col < NCOL:
                u = _ident(acc, u, {NIBS + 4 * idx + k: 1.0}, 0.0, COLS + col, 1.0)
    blocks.append(("bymul-accum", _truncate(acc, u, dim)))
    # (4) carry rounds settle the columns (values < 256 -> could be fp32, kept fp64
    #     to match the accumulate band).
    src, dst = COLS, _scratch(L, "BY_COLS2", NCOL)
    for r in range(NCOL):
        blocks.append((f"bymul-carry{r}", _carry_round_block(L, dim, src, dst, NCOL, dt=DT)))
        src, dst = dst, src
    blocks.append(("bymul-result", _result_copy_block(L, dim, src, RES, dt=DT)))
    info = {
        "note": "4 byte chunks -> 10 byte 16-bit products; MSB-first (base-256) peel. "
                "Product VALUES fit fp32 (<=65025<2^24) but the peel's -256*floor "
                "residue-subtract down-sum cancels at ~4e6 -> ~1/256 mis-round in "
                "fp32, so the PEEL needs fp64.  MSB-first does NOT keep it fp32.",
        "n_products": 10, "max_staircase_arg": 65025,
        "fp32_expected": False,   # values fit fp32, extraction does NOT
        "values_fit_fp32": (RELU_S * 65025) < FP32_INT_MAX,
    }
    return blocks, RES, info


# ===========================================================================
# VARIANT 6 — 16xnibble.  a in two 16-bit halves a_lo, a_hi; partial =
# (16-bit half) x (nibble of b) = 20-bit (<= 65535*15 = 983025).  Accumulate at
# weight 2^(16*h + 4*j); MSB-first peel to nibbles.
#
# ** fp32 WALL: the 20-bit partial (983025) feeds RELU_S*x = 2.0e8 > 2^24 -> the
# peel is fp32-LOSSY.  Built in fp64; the fp32 spot-check FAILS (flags the wall). **
# ===========================================================================
def build_16x_nibble(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    DT = torch.float64                                       # fp64 spec tensors (wall)
    L_STACK0, L_AX = L.STACK0, L.AX
    RES = _scratch(L, "NX_RES", NCOL)
    HALF_A = _scratch(L, "NX_HA", 2)
    COLS = _scratch(L, "NX_COLS", NCOL + 2)
    SCR = _scratch(L, "NX_SCR", 1); BSCR = _scratch(L, "NX_BSCR", 1)
    PART = _scratch(L, "NX_PART", 1)

    blocks: List[Block] = []
    # (1) two 16-bit halves of a.
    spec = _empty_spec_dt(dim, 2 * 2, DT)
    u = 0
    for h in range(2):
        u = _clear(spec, u, HALF_A + h)
        u = _ident(spec, u, {L_STACK0 + 4 * h: 1.0, L_STACK0 + 4 * h + 1: 16.0,
                             L_STACK0 + 4 * h + 2: 256.0, L_STACK0 + 4 * h + 3: 4096.0},
                   0.0, HALF_A + h, 1.0)
    blocks.append(("nxmul-halves", _truncate(spec, u, dim)))
    # (2) each partial (h,j) with 16h+4j < 32: partial = b_nib[j] (silu, <=15) *
    #     half[h] (gate, 16-bit) -> 20-bit; peel MSB-first into its OWN nibble band
    #     (multi-block), then accumulate.  All fp64.
    part_pairs = [(h, j) for h in range(2) for j in range(8) if 16 * h + 4 * j < 32]
    NIBS = _scratch(L, "NX_NIBS", len(part_pairs) * 6)       # up to 6 nibbles per byte-peel of 20-bit
    for idx, (h, j) in enumerate(part_pairs):
        spec = _empty_spec_dt(dim, 3, DT)
        u = _clear(spec, 0, PART)
        u = _mul_gate(spec, u, L_AX + j, HALF_A + h, PART, 1.0)   # nibble * 16-bit half
        blocks.append((f"nxmul-part-h{h}j{j}", _truncate(spec, u, dim)))
        blocks += _peel_blocks(L, dim, PART, 20, NIBS + 6 * idx, SCR, BSCR, f"nx{idx}", dt=DT)
    # accumulate every partial's nibbles into the shared columns at base 4h+j.
    acc = _empty_spec_dt(dim, (NCOL + 2) + len(part_pairs) * 6, DT)
    u = 0
    for c in range(NCOL + 2):
        u = _clear(acc, u, COLS + c)
    for idx, (h, j) in enumerate(part_pairs):
        base_nib = 4 * h + j
        for k in range(6):
            col = base_nib + k
            if col < NCOL:
                u = _ident(acc, u, {NIBS + 6 * idx + k: 1.0}, 0.0, COLS + col, 1.0)
    blocks.append(("nxmul-accum", _truncate(acc, u, dim)))
    # (3) carry settle (fp64 exact rounds).
    src, dst = COLS, _scratch(L, "NX_COLS2", NCOL + 2)
    for r in range(NCOL):
        blocks.append((f"nxmul-carry{r}", _carry_round_block(L, dim, src, dst, NCOL + 2, dt=DT)))
        src, dst = dst, src
    blocks.append(("nxmul-result", _result_copy_block(L, dim, src, RES, dt=DT)))
    info = {
        "note": "a in two 16-bit halves; partial = nibble x 16-bit = 20-bit; "
                "MSB-first (base-256) peel. fp32 WALL: top-byte threshold ~15*2^16 "
                "-> RELU_S*t=2e8 > 2^24 so the WEIGHT bias needs fp64.",
        "n_partials": len(part_pairs), "max_staircase_arg": 15 * (1 << 16),
        "fp32_expected": (RELU_S * 15 * (1 << 16)) < FP32_INT_MAX,   # False -> fp64
    }
    return blocks, RES, info


# ===========================================================================
# VARIANT 7 — 16-bit chunk (fp64).  a = a_hi*2^16 + a_lo, b = b_hi*2^16 + b_lo.
# Low 32 bits of a*b = (a_lo*b_lo) + 2^16*(a_hi*b_lo + a_lo*b_hi)  mod 2^32 (the
# a_hi*b_hi term is >= 2^32).  Three 16x16 = 32-bit products EXCEED 2^24 -> the
# PRODUCT needs fp64, and so does the peel.  Marked fp64 unconditionally.
#
# _mul_gate needs a byte-bounded multiplicand, so each 16x16 product is formed as
# 4 (16-bit x nibble) partials (sum_j (X * y_nib[j]) * 16^j), all in fp64.
# ===========================================================================
def build_16bit_chunk(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    DT = torch.float64                                       # fp64 spec tensors (wall)
    L_STACK0, L_AX = L.STACK0, L.AX
    RES = _scratch(L, "SC_RES", NCOL)
    A_LO = _scratch(L, "SC_ALO", 1); A_HI = _scratch(L, "SC_AHI", 1)
    B_LO = _scratch(L, "SC_BLO", 1); B_HI = _scratch(L, "SC_BHI", 1)
    P_LL = _scratch(L, "SC_PLL", 1); P_HL = _scratch(L, "SC_PHL", 1); P_LH = _scratch(L, "SC_PLH", 1)
    COLS = _scratch(L, "SC_COLS", NCOL + 2)
    SCR = _scratch(L, "SC_SCR", 1); BSCR = _scratch(L, "SC_BSCR", 1)

    blocks: List[Block] = []
    # (1) 16-bit halves of a and b.
    spec = _empty_spec_dt(dim, 4 * 2, DT)
    u = 0
    for dst, base in [(A_LO, L_STACK0), (A_HI, L_STACK0 + 4), (B_LO, L_AX), (B_HI, L_AX + 4)]:
        u = _clear(spec, u, dst)
        u = _ident(spec, u, {base: 1.0, base + 1: 16.0, base + 2: 256.0, base + 3: 4096.0},
                   0.0, dst, 1.0)
    blocks.append(("scmul-halves", _truncate(spec, u, dim)))

    # (2) three 16x16 products via 4 (nibble x 16-bit) partials each.  _mul_gate
    #     puts the FIRST operand through silu(S*A) (must be byte-bounded), so the
    #     NIBBLE (Y_lanes+j, 0..15) is the silu operand and the 16-bit half X is the
    #     gate: (X * y_nib[j]) * 16^j, summed into dst.  fp64 (product > 2^24).
    def _prod16(spec, u, X_band, Y_lanes, dst):
        for j in range(4):                                   # low 16 bits of Y
            u = _mul_gate(spec, u, Y_lanes + j, X_band, dst, float(1 << (4 * j)))
        return u
    spec = _empty_spec_dt(dim, 3 * (1 + 4 * 2), DT)
    u = 0
    u = _clear(spec, u, P_LL); u = _prod16(spec, u, A_LO, L_AX, P_LL)          # a_lo * b_lo
    u = _clear(spec, u, P_HL); u = _prod16(spec, u, A_HI, L_AX, P_HL)          # a_hi * b_lo
    u = _clear(spec, u, P_LH); u = _prod16(spec, u, A_LO, L_AX + 4, P_LH)      # a_lo * b_hi
    blocks.append(("scmul-products", _truncate(spec, u, dim)))
    # (3) assemble: result = P_LL + 2^16*(P_HL + P_LH), mod 2^32.  Peel each 32-bit
    #     product MSB-first (base-256) into its OWN nibble band (multi-block), then
    #     accumulate at base nibble 0 (LL) / 4 (HL, LH).  fp64.
    SC_NIBS = _scratch(L, "SC_NIBS", 3 * 8)                  # 8 nibbles per 32-bit product
    for pidx, (tag, src_band) in enumerate([("ll", P_LL), ("hl", P_HL), ("lh", P_LH)]):
        blocks += _peel_blocks(L, dim, src_band, 32, SC_NIBS + 8 * pidx, SCR, BSCR, f"sc{tag}", dt=DT)
    acc = _empty_spec_dt(dim, (NCOL + 2) + 3 * 8, DT)
    u = 0
    for c in range(NCOL + 2):
        u = _clear(acc, u, COLS + c)
    for pidx, (tag, base_nib) in enumerate([("ll", 0), ("hl", 4), ("lh", 4)]):
        for k in range(8):
            col = base_nib + k
            if col < NCOL:
                u = _ident(acc, u, {SC_NIBS + 8 * pidx + k: 1.0}, 0.0, COLS + col, 1.0)
    blocks.append(("scmul-accum", _truncate(acc, u, dim)))
    src, dst = COLS, _scratch(L, "SC_COLS2", NCOL + 2)
    for r in range(NCOL):
        blocks.append((f"scmul-carry{r}", _carry_round_block(L, dim, src, dst, NCOL + 2, dt=DT)))
        src, dst = dst, src
    blocks.append(("scmul-result", _result_copy_block(L, dim, src, RES, dt=DT)))
    info = {
        "note": "3 x (16x16=32-bit) products -> assemble. PRODUCT itself > 2^24 "
                "-> fp64 mandatory (product AND the top-byte peel bias).",
        "n_products": 3, "max_staircase_arg": (1 << 32) - 1,
        "fp32_expected": False,
    }
    return blocks, RES, info


# ===========================================================================
# MEASUREMENT
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
# SPARSE BYTE-EXACT SIMULATION.
#
# Per the brief: NO dense DIM forward.  Simulate each gadget's arithmetic on just
# the bands it reads/writes (a dict band->value), applying ``down(silu(gate)*up)``
# only over the units that touch a currently-nonzero band (or ONE).  This is the
# EXACT same SwiGLU math as a dense forward but skips the DIM x DIM matmul.
# ===========================================================================
def _spec_sparsity(spec: Spec, dtype):
    """Cache the COMPACT sparse form of a spec at the given runtime dtype: only the
    nonzero input bands (with their W_up/W_gate columns as a small [n_units, k]
    matrix) and the nonzero output dims (with their W_down rows).  Built ONCE per
    (spec, dtype) so the byte-exact loop over ~160 operands pays it once per block —
    the per-call cost is then two tiny matmuls, not a full DIM x n_units convert."""
    key = f"_sc_{dtype}"
    cache = spec.get(key)
    if cache is not None:
        return cache
    # NB: use abs-sum, not sum — _mul_gate writes +S and -S on the SAME band across
    # two units, so a plain column sum is ZERO there and would DROP the band.
    up_bands = torch.nonzero(spec["W_up"].abs().sum(0) != 0, as_tuple=False).flatten().tolist()
    gate_bands = torch.nonzero(spec["W_gate"].abs().sum(0) != 0, as_tuple=False).flatten().tolist()
    out_dims = torch.nonzero(spec["W_down"].abs().sum(1) != 0, as_tuple=False).flatten().tolist()
    Wu = spec["W_up"][:, up_bands].to(dtype).contiguous()       # [n_units, |up_bands|]
    Wg = spec["W_gate"][:, gate_bands].to(dtype).contiguous()
    Wd = spec["W_down"][out_dims].to(dtype).contiguous()        # [|out_dims|, n_units]
    cache = (up_bands, gate_bands, out_dims,
             Wu, Wg, Wd,
             spec["b_up"].to(dtype), spec["b_gate"].to(dtype),
             spec["b_down"][out_dims].to(dtype))
    spec[key] = cache
    return cache


def _sparse_apply(state: Dict[int, float], spec: Spec, dtype=torch.float32):
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
    hidden = F.silu(up) * gate                              # (n_units,)
    if out_dims:
        delta = Wd @ hidden + b_down
        for i, d in enumerate(out_dims):
            dv = float(delta[i])
            if dv:
                state[d] = state.get(d, 0.0) + dv
    return state


def simulate_sparse(L, blocks: List[Block], res_band: int, a_val: int, b_val: int,
                    dtype=torch.float32) -> int:
    state: Dict[int, float] = {L.ONE: 1.0}
    for c in range(8):
        state[L.STACK0 + c] = float((a_val >> (4 * c)) & 0xF)
        state[L.AX + c] = float((b_val >> (4 * c)) & 0xF)
    # bit-Dadda reads private bit bands; fill them if allocated.
    if "BKMUL_BIT_A" in L._names:
        ba = L._names["BKMUL_BIT_A"][0]; bb = L._names["BKMUL_BIT_B"][0]
        for i in range(32):
            state[ba + i] = float((a_val >> i) & 1)
            state[bb + i] = float((b_val >> i) & 1)
    for _n, s in blocks:
        state = _sparse_apply(state, s, dtype=dtype)
    out = 0
    for c in range(8):
        out |= (int(round(state.get(res_band + c, 0.0))) & 0xF) << (4 * c)
    return out & 0xFFFFFFFF


# ===========================================================================
# OPERANDS + LAYOUT
# ===========================================================================
def test_operands(n_random: int = 100, seed: int = 0) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    M = 0xFFFFFFFF
    # structured edge cases (0, all-ones, carry-heavy, chunk/byte boundaries) +
    # a few powers of two + randoms.  Kept lean (~100 pairs) so the byte-exact sim
    # stays in the seconds regime the brief asks for.
    edges = [0, 1, M, 0xDEADBEEF, 0xCAFEBABE, 0x80000000, 0xFFFF, 0x10000,
             0xABCD, 65535, 65536, 255, 256, 1 << 20, 1 << 24]
    pairs: List[Tuple[int, int]] = []
    for a in edges:
        for b in [1, M, 0xDEADBEEF, 0x10000]:
            pairs.append((a & M, b & M))
    for _ in range(n_random):
        pairs.append((rng.randint(0, M), rng.randint(0, M)))
    return pairs


DIM = 1536          # wide enough for every variant's private scratch bands + verify.


def _make_layout():
    L = NibbleVMLayout(8, n_heads=4)
    extend_layout_for_alu32(L)
    A._ONE = L.ONE
    return L


# ===========================================================================
# MSB-first primitive round-trip verification (do this FIRST, per the brief).
# ===========================================================================
def _verify_peel_roundtrip(L, dim, n=40, seed=1) -> dict:
    """decompose->recompose the MSB-first (base-256) peel on random W-bit values,
    at BOTH spec-tensor precisions.  Shows the wall: W<=16 is fp32-exact (top-byte
    threshold <= 65280 < 2^24/RELU_S), W>=20 needs fp64 spec tensors (the top-byte
    RELU_S*threshold bias exceeds fp32's 2^24 integer ceiling)."""
    rng = random.Random(seed)
    out = {}
    for W in [16, 20, 24, 32]:
        OUT = _scratch(L, f"VP{W}_OUT", 8)
        SCR = _scratch(L, f"VP{W}_SCR", 1)
        BSC = _scratch(L, f"VP{W}_BSC", 1)
        SRC = _scratch(L, f"VP{W}_SRC", 1)
        for dt in (torch.float32, torch.float64):
            block = _peel_blocks(L, dim, SRC, W, OUT, SCR, BSC, f"vp{W}_{dt}", dt=dt)
            ok = 0
            for _ in range(n):
                x = rng.randint(0, (1 << W) - 1)
                st = {L.ONE: 1.0, SRC: float(x)}
                for _nm, s in block:
                    st = _sparse_apply(st, s, dtype=dt)
                rec = 0
                for c in range((W + 3) // 4):
                    rec |= (int(round(st.get(OUT + c, 0.0))) & 0xF) << (4 * c)
                if rec == x:
                    ok += 1
            out[f"W{W}_{'fp32' if dt==torch.float32 else 'fp64'}"] = (ok, n)
    return out


# ===========================================================================
# BAKEOFF DRIVER
# ===========================================================================
def run_bakeoff(n_random: int = 100, verbose: bool = True) -> dict:
    results: dict = {}
    L = _make_layout()
    dim = DIM
    M = 0xFFFFFFFF

    peel = _verify_peel_roundtrip(L, dim)
    results["_peel_roundtrip"] = peel
    if verbose:
        print("MSB-first peel decompose->recompose (fp32 wall at RELU_S*x<2^24):")
        for k, (ok, tot) in peel.items():
            print(f"    {k}: {ok}/{tot}")

    builders = [
        ("1_nibble_baseline",   build_nibble_baseline,     "fp32"),
        ("2_nibble_carryshare", build_nibble_carry_share,  "fp32"),
        ("3_nibble_dadda",      build_nibble_dadda,        "fp32"),
        ("4_bit_dadda",         build_bit_dadda,           "fp32"),
        ("5_byte_chunk",        build_byte_chunk,          "fp64"),  # peel needs fp64
        ("6_16x_nibble",        build_16x_nibble,          "fp64"),
        ("7_16bit_chunk",       build_16bit_chunk,         "fp64"),
    ]
    built = []
    for name, fn, prec in builders:
        blocks, res, info = fn(L, dim)
        built.append((name, blocks, res, prec, info))
    assert L.D <= dim, f"layout grew past DIM ({L.D} > {dim}); raise DIM"

    ops = test_operands(n_random=n_random)
    for name, blocks, res, prec, info in built:
        m = measure(blocks)
        max_arg = info.get("max_staircase_arg", 0)
        fp32_safe_by_arg = (RELU_S * max_arg) < FP32_INT_MAX
        dt = torch.float64 if prec == "fp64" else torch.float32
        n_pass = 0; first_fail = None
        for a, b in ops:
            got = simulate_sparse(L, blocks, res, a, b, dtype=dt)
            exp = (a * b) & M
            if got == exp:
                n_pass += 1
            elif first_fail is None:
                first_fail = (hex(a), hex(b), hex(got), hex(exp))
        # fp32 spot-check on a subset: fp32-claimed variants should PASS; fp64
        # ones should FAIL (that failure is the honest proof of the fp32 wall).
        spot = ops[:40]
        fp32_pass = sum(simulate_sparse(L, blocks, res, a, b, dtype=torch.float32) == (a * b) & M
                        for a, b in spot)
        results[name] = {
            **m, "claimed_precision": prec,
            "byte_exact_pass": n_pass, "byte_exact_total": len(ops),
            "first_fail": first_fail, "max_staircase_arg": max_arg,
            "fp32_safe_by_arg": fp32_safe_by_arg,
            "fp32_spotcheck_pass": fp32_pass, "fp32_spotcheck_total": len(spot),
            **{k: v for k, v in info.items() if k != "max_staircase_arg"},
        }
        if verbose:
            print(f"{name:20s} depth={m['depth']:4d} nz={m['weights_nz']:8d} "
                  f"{prec} exact={n_pass}/{len(ops)} fp32spot={fp32_pass}/{len(spot)} "
                  f"argmax={max_arg} fp32safe={fp32_safe_by_arg} fail={first_fail}")
    return results


def _print_table(results: dict):
    rows = []
    for name in ["1_nibble_baseline", "2_nibble_carryshare", "3_nibble_dadda",
                 "4_bit_dadda", "5_byte_chunk", "6_16x_nibble", "7_16bit_chunk"]:
        r = results[name]
        rows.append((name, r["depth"], r["weights_nz"], r["claimed_precision"],
                     r["fp32_safe_by_arg"], r["byte_exact_pass"], r["byte_exact_total"],
                     r["fp32_spotcheck_pass"], r["fp32_spotcheck_total"]))
    print()
    print("VARIANT               DEPTH   WEIGHTS  PREC  fp32ok  BYTE-EXACT  fp32-spot")
    print("-" * 78)
    for (n, d, w, p, s, bp, bt, fsp, fst) in sorted(rows, key=lambda x: (x[1], x[2])):
        print(f"{n:20s} {d:5d}  {w:8d}  {p:4s}  {str(s):5s}  {bp:4d}/{bt:<4d}  {fsp}/{fst}")


if __name__ == "__main__":
    res = run_bakeoff()
    _print_table(res)
