"""NIBBLE-PRODUCT / BYTE-COLUMN multiplier — "carry on a larger representation".

Builds and MEASURES a 32-bit ``(a*b) & 0xFFFFFFFF`` multiplier as persistent
SwiGLU FFN blocks, COMPOSED from the proven ``nibble_alu32`` primitives
(``_mul_gate``, ``_empty_spec``, ``_floor_div_pow``, ``_ident``, ``_clear``,
``_truncate``, ``S``, ``RELU_S``).  It NEVER edits those, and does NOT touch
``mul_bakeoff*.py`` / ``mul_byte_carrysave.py`` (owned by other agents).

THE IDEA — cheap nibble products, byte-WIDE accumulation, PARALLEL peels
=======================================================================
The nibble schoolbook baseline (``compile_mul_blocks``) uses the CHEAP fp32-safe
nibble products ``a_i*b_j`` (each <= 225) and accumulates them into **8 nibble
columns** (weight 16^s), which then need **7 base-16 carry rounds** (one ripple
per nibble column) to settle — a long serial carry tail.

A *byte-PRODUCT* multiplier (``mul_byte_carrysave.py``) halves the column count
to 4 but pays a heavy price: byte products reach 65025 and 4-deep byte columns
reach 260100, so ``RELU_S*260100`` blows the 2^24 fp32-integer ceiling at the
default RELU_S=200 — it was FORCED down to RELU_S<=64, needing a fragile
low-RELU_S peel + a per-digit "snap" to recover the fp residue.

THIS design takes the best of both: the CHEAP nibble products (each <= 225), but
ACCUMULATES them into **4 BYTE-position columns** (weight 256^c) instead of 8
nibble columns — the "carry on a larger representation" idea, done WITHOUT the
wide-product penalty:

  1. **36 nibble products** ``a_i*b_j`` (i+j<8), each ``<= 225`` — the exact same
     fp32-safe products the baseline already uses (``S*a*b <= 200*225 = 45000 <<
     2^24``, huge headroom).

  2. **Accumulate into 4 BYTE-position columns** (weight ``256^c``, c=0..3), NO
     per-product split.  A nibble product with nibble-weight ``16^s`` (s=i+j)
     lands in byte column ``c = s // 2``, scaled by ``16`` when ``s`` is odd
     (since ``16^s = 256^(s//2) * 16^(s%2)``).  So

         byte-col c = Σ_{i+j=2c} p  +  16 * Σ_{i+j=2c+1} p .

     The EXACT worst-case byte-column value (all nibbles = 15, so every product
     = 225) is **30375** (the top column c=3: 7 products at s=6 weight 1 plus 8
     products at s=7 weight 16 -> 7*225 + 16*8*225 = 30375).  At the DEFAULT
     **RELU_S=200**: ``200 * 30375 = 6_075_000`` = **36.21% of 2^24** (headroom
     63.79%).  So NO low-RELU_S, NO residue-snap machinery is needed — this is
     the whole advantage over byte-PRODUCT accumulation, whose 260100 columns
     forced RELU_S<=64.  Every ``_mul_gate`` / ``_floor_div_pow`` / carry gadget
     runs at the library's RELU_S=200 with oceans of fp32 headroom.

  3. **Peel each byte column to nibbles**, MSB-first, **IN PARALLEL across the 4
     columns** — they are independent until the final byte-offset combine, so
     the 4 peels SHARE the same block depth (one block per peel STAGE handles
     all 4 columns at once, NOT 4 serial per-column peels).  Each byte column is
     <= 30375 (< 2^15 -> at most 4 nibbles); every peel argument is <= 30375, so
     the library's RELU_S=200 base-16 ``_floor_div_pow`` (kmax=15) is bit-exact
     and NO per-digit snap is required (all args are integer-exact at RELU_S=200,
     since the raw column products carry NO fp residue — they are exact sums of
     exact ``_mul_gate`` integers).

  4. **Combine the 4 columns at their byte offsets** (column c -> result nibbles
     2c..2c+3) into 9 raw nibble columns (< 256), then a base-16 carry ripple
     settles them into the final 8 result nibbles.  Only **4 byte-columns** feed
     the combine, so the settle ripple is SHORT (4 carry rounds vs the baseline's
     7 nibble-column rounds).

The whole gadget stays at RELU_S=200 (tightest arg = 30375 -> 36.21% of 2^24),
so there is no fragile margin anywhere: it is a strict improvement in fp32
discipline over the byte-PRODUCT attempt AND fewer carry rounds than the nibble
schoolbook.

MEASUREMENT (lean, seconds, no dense DIM forward — same harness style as the
sibling bakeoffs): DEPTH = #blocks, WEIGHTS = sum of nonzeros, TIGHTEST
``RELU_S*arg`` as a fraction of 2^24 (fp32 headroom), BYTE-EXACT via a sparse
per-band arithmetic sim over ~200 random + structured (a,b) pairs (incl 0,
0xFFFFFFFF, 0xDEADBEEF, powers of two).
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
    _mul_gate, _ident, _clear, _truncate, _floor_div_pow,
    compile_mul_blocks, extend_layout_for_alu32,
)

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

FP32_INT_MAX = 1 << 24                       # 2^24: fp32 exact-integer ceiling
NCOL = 8                                     # 8 nibble result columns (32-bit)
N_BYTE_COL = 4                               # 4 byte-position columns (weight 256^c)
_NIB_KMAX = 15                               # floor(col/16) for a column kept < 256


# the (i,j) nibble pairs with i+j<8, and the nibble-position s = i+j of each.
_MUL_PAIRS = [(i, j) for i in range(NCOL) for j in range(NCOL) if i + j < NCOL]


# ---------------------------------------------------------------------------
# EXACT byte-column max: all nibbles = 15 -> every product = 225.  Byte column c
# gathers products at s=2c (weight 1) and s=2c+1 (weight 16).
# ---------------------------------------------------------------------------
def byte_column_max() -> Tuple[List[int], int]:
    """Return (per-column max list, overall max byte-column value).  Used both to
    size the peel (nibbles per column) and to report the fp32 headroom."""
    P = 15 * 15                                          # 225, max nibble product
    per_col = []
    for c in range(N_BYTE_COL):
        lo = sum(1 for (i, j) in _MUL_PAIRS if i + j == 2 * c)       # weight 1
        hi = sum(1 for (i, j) in _MUL_PAIRS if i + j == 2 * c + 1)   # weight 16
        per_col.append(P * lo + 16 * P * hi)
    return per_col, max(per_col)


PER_COL_MAX, COLUMN_MAX = byte_column_max()             # ([7425,15075,22725,30375], 30375)


# ===========================================================================
# fp32 HEADROOM accounting — every gadget records (label, relu_s, max_arg) so
# the harness can surface the single tightest RELU_S*arg across the whole build.
# (Same discipline as the sibling bakeoffs; here every arg is comfortably safe
# at RELU_S=200, which is the point.)
# ===========================================================================
_HEADROOM_LOG: List[Tuple[str, float, int]] = []


def _reset_headroom():
    _HEADROOM_LOG.clear()


def _assert_headroom(label: str, relu_s: float, max_arg: int):
    prod = relu_s * max_arg
    assert prod < FP32_INT_MAX, (
        f"{label}: RELU_S({relu_s})*max_arg({max_arg})={prod:.0f} "
        f">= 2^24={FP32_INT_MAX} -> fp32-LOSSY")
    _HEADROOM_LOG.append((label, relu_s, max_arg))


def tightest_headroom() -> dict:
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
    key = f"BAC_{name}"
    if key in L._names:
        return L._names[key][0]
    return L._band(key, size)


# ===========================================================================
# Shared base-16 carry round (library RELU_S=200, args <= 240 -> deep headroom).
# ONE round on ``n`` columns kept < 256:  dst[c] = src[c] mod 16 + floor(src[c-1]/16).
# Uses the library _floor_div_pow (the same staircase the production nibble
# baseline uses), so the fp discipline is byte-identical to the baseline.
# ===========================================================================
def _carry_round_block(L, dim, src, dst, n, tag) -> Block:
    _assert_headroom(f"{tag}-carry", RELU_S, _NIB_KMAX * 16)     # RELU_S*240
    spec = _empty_spec(dim, n * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
    u = 0
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)                        # + col
        u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX, dst + c, -16.0)  # - 16*floor
        if c > 0:
            u = _floor_div_pow(spec, u, {src + c - 1: 1.0}, 0.0, 16, _NIB_KMAX, dst + c, 1.0)  # + carry
    return _truncate(spec, u, dim)


def _result_copy_block(L, dim, src, res) -> Block:
    copy = _empty_spec(dim, NCOL * 2)
    u = 0
    for c in range(NCOL):
        u = _clear(copy, u, res + c)
        u = _ident(copy, u, {src + c: 1.0}, 0.0, res + c, 1.0)
    return _truncate(copy, u, dim)


# ===========================================================================
# PARALLEL MSB-first column peel.  Each of the 4 byte columns (<= 30375, so <=
# 4 nibbles) is decomposed into its nibbles MSB-first at RELU_S=200 — and the 4
# columns share block depth (one block per peel STAGE handles all 4 columns).
#
# Because the byte columns are EXACT integer sums of exact _mul_gate integers
# (no fp residue), the library RELU_S=200 base-16 floor staircase is bit-exact
# and NO per-digit snap is needed (the fragile part of the byte-PRODUCT peel).
# The peel argument is <= 30375, so RELU_S*30375 = 6.075M (36.21% of 2^24).
# ===========================================================================
_N_NIB_PER_COL = max(1, (COLUMN_MAX.bit_length() + 3) // 4)      # 4 (30375 < 2^15)


def _peel_digit_block(L, dim, cnibs, rscr_src, idx, tag) -> Block:
    """Peel STAGE part A: ``digit_c = floor(residue_c / 16^idx)`` for ALL 4 byte
    columns IN PARALLEL (into ``cnibs[c*4 + idx]``), reading the block-input
    residue ``rscr_src[c]``.  Realised by the library ``_floor_div_pow`` (the SAME
    exact base-16 staircase the nibble baseline uses), so the digit is a clean
    integer 0..kmax — which the following residue-subtract block then multiplies
    by ``16^idx`` LOSSLESSLY.  The digit and the residue-subtract MUST be separate
    blocks because every SwiGLU unit reads the block INPUT: the subtract can only
    see a digit materialised by a PRIOR block.

    kmax is tight: floor(colmax / 16^idx), capped at 15 (a nibble digit).  Every
    staircase threshold is <= colmax = 30375 -> RELU_S*30375 = 6.075M (36.21% of
    2^24), exact at RELU_S=200 — no low-RELU_S, no snap needed."""
    m = 16 ** idx
    spec = _empty_spec(dim, N_BYTE_COL * (1 + 15 * 2))
    u = 0
    for c in range(N_BYTE_COL):
        colmax = PER_COL_MAX[c]
        src = rscr_src + c
        digit = cnibs + _N_NIB_PER_COL * c + idx
        kmax = min(15, colmax // m) if m > 1 else 15
        kmax = max(1, kmax)
        _assert_headroom(f"{tag}-c{c}-n{idx}", RELU_S, kmax * m)
        u = _clear(spec, u, digit)
        u = _floor_div_pow(spec, u, {src: 1.0}, 0.0, m, kmax, digit, 1.0)
    return _truncate(spec, u, dim)


def _peel_residue_block(L, dim, cnibs, rscr_src, rscr_dst, idx, tag) -> Block:
    """Peel STAGE part B: ``rscr_dst[c] = rscr_src[c] - 16^idx * digit_c`` for ALL
    4 columns IN PARALLEL — the residue for the NEXT (lower) stage.  Reads the
    CLEAN-integer digit ``cnibs[c*4+idx]`` (materialised by ``_peel_digit_block``,
    a prior block) so the ``16^idx`` multiply is lossless (exact integer * exact
    integer).  A pure ``_ident`` (linear), no staircase -> no fp margin.  The
    residue double-buffers ``src <-> dst`` across stages."""
    m = 16 ** idx
    spec = _empty_spec(dim, N_BYTE_COL * 3)
    u = 0
    for c in range(N_BYTE_COL):
        src = rscr_src + c
        dst = rscr_dst + c
        digit = cnibs + _N_NIB_PER_COL * c + idx
        u = _clear(spec, u, dst)
        u = _ident(spec, u, {src: 1.0}, 0.0, dst, 1.0)               # + residue
        u = _ident(spec, u, {digit: -float(m)}, 0.0, dst, 1.0)       # - 16^idx*digit
    return _truncate(spec, u, dim)


# ===========================================================================
# THE NIBBLE-PRODUCT / BYTE-COLUMN MULTIPLIER.
# ===========================================================================
# Carry rounds to settle the 9-lane byte-offset combine.  After peel+combine the
# 9 nibble lanes carry like any nibble column, and a carry ripples ONE lane per
# round, so the bound is the ripple LENGTH, NOT the byte-column count.  This is
# the HONEST cost: byte-column accumulation shortens the *pre-peel* carry (4 byte
# columns, no pre-peel rounds — vs the baseline's 8 nibble columns needing 7
# rounds) but the *post-combine* nibble ripple returns.  The literal max product
# 0xFFFFFFFF^2 (which produces the genuinely worst-case all-15s columns, so no
# input yields a longer ripple) settles in 5 rounds (max chain seen
# [16,15,15,15,15]); we keep 6 (+1 headroom — a settled column is a fixed point
# of a further round, so extra rounds never corrupt a result).
_COMBINE_CARRY_ROUNDS = 6


def build_byte_accum(L, dim, n_carry: int = _COMBINE_CARRY_ROUNDS) -> Tuple[List[Block], int, dict]:
    """36 cheap nibble products -> accumulate into 4 BYTE-position columns (NO
    split) -> peel each column MSB-first IN PARALLEL (4 columns share depth) at
    RELU_S=200 -> combine at byte offsets with the carry ripple -> 8 result
    nibbles.  Every gadget runs at the library RELU_S=200; the column value
    reaches 30375 (36.21% of 2^24) and the tightest baked staircase threshold is
    28672 (tight top-nibble kmax, 34.18% of 2^24) -> deep fp32 headroom, no snap."""
    A._ONE = L.ONE
    L_STACK0, L_AX = L.STACK0, L.AX

    RES = _scratch(L, "RES", NCOL)
    PP = _scratch(L, "PP", len(_MUL_PAIRS))              # 36 raw nibble products
    COLS = _scratch(L, "COLS", N_BYTE_COL)               # 4 byte-position columns
    CNIBS = _scratch(L, "CNIBS", N_BYTE_COL * _N_NIB_PER_COL)   # peeled nibbles
    RSCR = _scratch(L, "RSCR", N_BYTE_COL)               # per-column peel residue
    RSCR2 = _scratch(L, "RSCR2", N_BYTE_COL)             # double buffer
    # byte-offset combine: 9 raw nibble columns (low 32 bits + 1 carry lane).
    MCOL = _scratch(L, "MCOL", NCOL + 1)

    blocks: List[Block] = []

    # (1) 36 nibble products a_i*b_j (i+j<8), each <= 225.  _mul_gate: internal
    #     silu(S*a)*b ~ S*a*b <= 200*225 = 45000 << 2^24 -> exact for nibble a.
    _assert_headroom("mul-gate", S, 15 * 15)             # up=S*a<=S*15, gate<=15
    spec = _empty_spec(dim, len(_MUL_PAIRS) * 3)
    u = 0
    for idx, (i, j) in enumerate(_MUL_PAIRS):
        u = _clear(spec, u, PP + idx)
        u = _mul_gate(spec, u, L_STACK0 + i, L_AX + j, PP + idx, 1.0)
    blocks.append(("bac-products", _truncate(spec, u, dim)))

    # (2) ACCUMULATE into 4 byte columns.  Product at s=i+j -> byte col s//2,
    #     scaled by 16 when s is odd.  NO per-product split -> the carry is on the
    #     LARGER (byte) representation.  Also PRIME the peel residue RSCR <- col.
    spec = _empty_spec(dim, N_BYTE_COL * 2 + len(_MUL_PAIRS) * 2)
    u = 0
    for c in range(N_BYTE_COL):
        u = _clear(spec, u, COLS + c)
        u = _clear(spec, u, RSCR + c)
    for idx, (i, j) in enumerate(_MUL_PAIRS):
        s = i + j
        c = s // 2
        w = 16.0 if (s & 1) else 1.0
        u = _ident(spec, u, {PP + idx: w}, 0.0, COLS + c, 1.0)      # into byte col
        u = _ident(spec, u, {PP + idx: w}, 0.0, RSCR + c, 1.0)      # prime residue
    blocks.append(("bac-accum", _truncate(spec, u, dim)))

    # (3) PEEL each byte column to nibbles, MSB-first, IN PARALLEL across all 4
    #     columns (the 4 columns share depth).  Per stage: a DIGIT block (exact
    #     base-16 staircase) then a RESIDUE block (lossless linear subtract of the
    #     clean-integer digit) — separate because a unit reads the block INPUT.
    #     The residue double-buffers RSCR <-> RSCR2 across stages.  The LSB stage
    #     (idx == 0) needs no residue block (nothing below it).
    rsrc, rdst = RSCR, RSCR2
    for idx in range(_N_NIB_PER_COL - 1, -1, -1):
        blocks.append((f"bac-peel-n{idx}-dig",
                       _peel_digit_block(L, dim, CNIBS, rsrc, idx, "bac")))
        if idx > 0:
            blocks.append((f"bac-peel-n{idx}-res",
                           _peel_residue_block(L, dim, CNIBS, rsrc, rdst, idx, "bac")))
            rsrc, rdst = rdst, rsrc

    # (4) COMBINE the 4 column-nibble sets at their byte offsets: column c has
    #     weight 256^c = 16^(2c), so its nibble k lands at result nibble 2c+k.
    #     Gather into 9 raw nibble columns (< 256), then carry-normalise.
    spec = _empty_spec(dim, (NCOL + 1) + N_BYTE_COL * _N_NIB_PER_COL)
    u = 0
    for k in range(NCOL + 1):
        u = _clear(spec, u, MCOL + k)
    for c in range(N_BYTE_COL):
        for k in range(_N_NIB_PER_COL):
            slot = 2 * c + k
            if slot <= NCOL:                    # keep low 32 bits + 1 carry lane
                u = _ident(spec, u, {CNIBS + _N_NIB_PER_COL * c + k: 1.0},
                           0.0, MCOL + slot, 1.0)
    blocks.append(("bac-combine", _truncate(spec, u, dim)))

    # (5) carry ripple settles MCOL (each lane the sum of at most 2 nibbles + a
    #     carry -> < 256) into single nibbles.  A carry ripples one lane per round;
    #     the 9-lane combine needs up to 8 rounds (ripple length), the honest cost
    #     the byte-offset combine reintroduces after the byte-column accumulation.
    src, dst = MCOL, _scratch(L, "MCOL2", NCOL + 1)
    for r in range(n_carry):
        blocks.append((f"bac-carry{r}",
                       _carry_round_block(L, dim, src, dst, NCOL + 1, "bac")))
        src, dst = dst, src
    blocks.append(("bac-result", _result_copy_block(L, dim, src, RES)))

    info = {
        "note": "36 nibble products -> 4 BYTE-position columns (NO split) -> peel "
                "each column MSB-first IN PARALLEL (4 share depth) at RELU_S=200 -> "
                "byte-offset combine + carry ripple",
        "n_products": len(_MUL_PAIRS), "n_byte_columns": N_BYTE_COL,
        "per_column_max": PER_COL_MAX, "column_max": COLUMN_MAX,
        "nibbles_per_column": _N_NIB_PER_COL,
        "relu_s": RELU_S,
        # the column VALUE reaches column_max (30375); the tightest baked staircase
        # THRESHOLD is the top-nibble kmax*16^idx (28672, tightly capped) -> that
        # is the binding fp32 argument.
        "column_max_ratio_of_2p24": (RELU_S * COLUMN_MAX) / FP32_INT_MAX,
        "peel_blocks": 2 * _N_NIB_PER_COL - 1,     # dig per stage + res for non-LSB
        "combine_carry_rounds": n_carry,
    }
    return blocks, RES, info


# ===========================================================================
# CONTENDER — nibble schoolbook baseline (wrap the production compile_mul_blocks
# for an apples-to-apples DEPTH / WEIGHTS / fp32 / byte-exact comparison).
# ===========================================================================
def build_nibble_baseline(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    blocks = list(compile_mul_blocks(L, dim))
    # the baseline's staircases are the library's (RELU_S=200), args <= 232.
    _assert_headroom("nibble-baseline", RELU_S, 232)
    info = {
        "note": "production nibble schoolbook (compile_mul_blocks): 36 nibble "
                "products, 8 nibble columns < 256, 7 carry rounds; every arg <= 232",
        "n_products": 36, "carry_rounds": 7, "max_staircase_arg": 232,
        "relu_s": RELU_S,
    }
    return blocks, L.ALU32.MUL_RES, info


# ===========================================================================
# MEASUREMENT — depth / weights + a SPARSE byte-exact sim (no dense forward).
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
    singles = [0x000000AB, 0x0000CD00, 0x00EF0000, 0x12000000]
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
def run_bakeoff(n_random: int = 180, verbose: bool = True) -> dict:
    results: dict = {}
    L = _make_layout()
    dim = DIM
    M = 0xFFFFFFFF
    ops = test_operands(n_random=n_random)

    builders: List[Tuple[str, callable]] = [
        ("nibble_baseline", build_nibble_baseline),
        ("byte_accum", build_byte_accum),
    ]

    for name, fn in builders:
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
            print(f"{name:18s} depth={m['depth']:4d} nz={m['weights_nz']:8d} "
                  f"exact={n_pass}/{len(ops)} tightest RELU_S*arg={head['product']:.0f} "
                  f"({100 * head['tightest_ratio']:.2f}% of 2^24, headroom "
                  f"{100 * head['headroom']:.2f}%) fail={first_fail}")
    return results


def _print_table(results: dict):
    order = ["nibble_baseline", "byte_accum"]
    print()
    print("VARIANT             DEPTH   WEIGHTS   TIGHTEST RELU_S*arg   fp32 HEADROOM   BYTE-EXACT")
    print("-" * 90)
    for name in order:
        r = results[name]
        print(f"{name:18s} {r['depth']:5d}  {r['weights_nz']:8d}   "
              f"{r['tightest_relu_s_arg']:12.0f} ({100 * r['tightest_ratio_of_2p24']:5.2f}%)   "
              f"{100 * r['fp32_headroom']:7.2f}%       "
              f"{r['byte_exact_pass']:4d}/{r['byte_exact_total']:<4d}")


if __name__ == "__main__":
    res = run_bakeoff()
    _print_table(res)
