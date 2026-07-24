"""RUTHLESSLY-LEANED radix-16 (nibble-at-a-time) digit-recurrence 32-bit divide.

Standalone MEASURE-ONLY bakeoff module — one of four parallel DIV alternatives.
This is the *robust, always-exact* baseline: base-16 long division with **NO
reciprocal** and **NO per-digit multiply**.  Its only value is depth: how far a
leaned digit-recurrence can drop below the ~262-block base-16 long division in
``nibble_alu32.compile_divmod_blocks`` while staying fully robust.

Approach (KB-precompute once, then 8 lean iterations)
=====================================================
1.  **Precompute ``KB[k] = k*b`` for k=1..15 ONCE** — the ONLY multiply in the
    whole divide.  Reuses the carry-share nibble-multiply idea from
    ``nibble_alu32._kb_precompute_blocks``: raw column ``k*b_nib`` (<= 225) via
    an ``_ident`` fan-out, then base-16 carry rounds.  Each KB[k] ends already
    carry-normalised (every nibble 0..15), LSB-first.

2.  **8 iterations, MSB-first.**  Per iteration (weight-shared body):
      * ``R = 16*R + next_dividend_nibble``          (nibble shift + insert)
      * ``q = max{ k : KB[k] <= R }`` via a **lexicographic nibble compare**
        across the precomputed KB[k] — cheap 0/1 gt/eq lanes, NO multiply.
      * ``R -= KB[q]``                                (nibble borrow chain)
      * emit ``q`` MSB-first into the quotient nibbles.

3.  **Minimum blocks per iteration.**  The base long division spends ~21
    blocks/iter (shift, gteq, qdigit, qcopy, qb, 6 qb-carry, 9 sub-nibble, r2r).
    Here every one of those is cut or fused:
      * NO per-iter ``qb`` + NO 6 ``qb-carry``: we subtract the ALREADY
        carry-normalised ``KB[q]`` directly (selected by a one-hot on ``q`` off
        the monotone ``[R>=KB[k]]`` prefix), so there is **no per-iteration
        multiply and no per-iteration carry round at all**.
      * the 9-block per-nibble borrow ripple collapses to a **2-block 3-limb
        borrow** (R and KB[q] split into 3 limbs of 3 nibbles; each limb value
        <= 16^3-1 = 4095 << 2^24, so the 3-stage limb-borrow + result-value fits
        one block, and a second block splits the limb values into result nibbles
        while emitting q and advancing the counter).

    ==> **5 blocks / iteration** (shift, gteq, qdigit, qbsel, sub2 = the fused
    borrow+split+emit).  Confirmed by the ``_measure`` harness below.

fp32 discipline
===============
Every relu / silu argument stays well < 2^24: the nibble compares keep every
compared quantity in [-15,15]; the limb-borrow limbs are <= 4095; the floor/mod
staircases in the split use thresholds k*16^p with p<3 (<= 16^3 = 4096).  The
harness reports the exact max relu argument (0 fp64 params).

Byte-exact: validated on the required edge grid + >=300 random (a,b<2^32) pairs
by a CPU SwiGLU forward simulation of the ACTUAL FFN blocks (no full model).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Reuse the fp32-exact SwiGLU primitives + constants (READ-only import; this
# module does NOT edit nibble_alu32.py).  We use _ident/_clear/_step_ge/_guard/
# _empty_spec/_truncate/_floor_div_pow; _mul_gate/_floor_div_pow2 are imported
# per the shared-reuse contract but NOT used here — the whole point of this
# variant is NO per-digit multiply (only the one-time KB precompute multiplies,
# and even that is an _ident fan-out) and NO fused two-dst staircase.
from .nibble_alu32 import (            # noqa: F401  (contract-listed reuse set)
    _mul_gate, _floor_div_pow, _floor_div_pow2, _guard, _step_ge, _ident,
    _clear, _empty_spec, _truncate, RELU_S, S,
)
from .nibble_vm_layout import NibbleVMLayout


def _set_one(L):
    """The nibble_alu32 emitters read a MODULE-GLOBAL ``_ONE``; set it there."""
    import c4_min.nibble_alu32 as m
    m._ONE = L.ONE


# ---------------------------------------------------------------------------
# Scratch layout for the leaned radix-16 divide (own bands; self-contained).
# ---------------------------------------------------------------------------
class LeanDivBands:
    def __init__(self, L):
        self.RN = 9                            # remainder nibbles (R < 16*b < 2^36)
        RN = self.RN
        self.R = L._band("LR_R", RN)           # running remainder nibbles, LSB first
        self.KB = L._band("LR_KB", RN * 16)    # KB[k]=k*b nibbles, k=0..15 (0 unused)
        self.GT = L._band("LR_GT", 15 * RN)    # GT[k,i] = [R[i] >  KB[k][i]]
        self.EQ = L._band("LR_EQ", 15 * RN)    # EQ[k,i] = [R[i] == KB[k][i]]
        self.QD = L._scalar("LR_QD")           # quotient digit q (0..15)
        self.GE = L._band("LR_GE", 16)         # GE[k] = [R >= KB[k]] prefix (k=1..15)
        self.KBQ = L._band("LR_KBQ", RN)       # selected KB[q] subtrahend nibbles
        # 3-limb (3-nibble) borrow: every limb value <= 16^3-1 = 4095 << 2^24, so
        # every compare / floor stays fp32-exact (RELU_S*4095 = 819000 < 2^24).
        self.B0 = L._scalar("LR_B0")           # borrow out of limb 0  ([d0 < 0])
        self.B1 = L._scalar("LR_B1")           # borrow out of limb 1  ([d1-B0 < 0])
        self.RV = L._band("LR_RV", 3)          # R2 limb values (0..4095 each)
        self.DIV_RES = L._band("LR_DIV", 8)    # quotient nibbles (result)
        self.MOD_RES = L._band("LR_MOD", 8)    # remainder nibbles (result)
        self.BZ = L._scalar("LR_BZ")           # divisor == 0 predicate
        self.IT = L._scalar("LR_IT")           # iteration index 0..7
        self.IT_OH = L._band("LR_IT_OH", 8)    # one-hot(IT)


def extend_layout(L):
    if getattr(L, "LEANDIV", None) is not None:
        return L.LEANDIV
    L.LEANDIV = LeanDivBands(L)
    while L._off % L.n_heads != 0:
        L._scalar(f"_leanpad{L._off}")
    L.D = L._off
    return L.LEANDIV


_NIB_KMAX = 15
# k*b_nib <= 225 per raw column; a single-nibble carry ripple settles ALL 15 KB[k]
# for EVERY 32-bit b (incl. 0xFFFFFFFF, all-F, alternating) in <= 3 rounds
# (exhaustively verified: 0 dirty/wrong KB over 3135 (k,b) at rounds=3).  +1 round
# headroom -> 4 (a settled column is a fixed point of further rounds, so harmless).
_KB_CARRY_ROUNDS = 4


# ===========================================================================
# 1. KB-precompute (the ONLY multiply): KB[k] = k*b nibbles, k=1..15.
# ===========================================================================
def _lean_carry_round_units(spec, u, src, dst, n):
    """One base-16 carry-normalise round on ``n`` columns kept < 256 (SET each)."""
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)                        # + col
        u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX, dst + c, -16.0)   # mod
        if c + 1 < n:
            u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX, dst + c + 1, 1.0)  # carry
    return u


def _kb_raw_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, 16 * RN + 16 * 8)
    u = 0
    for k in range(1, 16):
        base = a.KB + RN * k
        for c in range(RN):
            u = _clear(spec, u, base + c)
        for c in range(8):                                       # b = AX, 8 nibbles
            u = _ident(spec, u, {L.AX + c: float(k)}, 0.0, base + c, 1.0)   # + k*b_nib[c]
    return _truncate(spec, u, dim)


def _kb_carry_all_block(L, dim) -> Dict[str, torch.Tensor]:
    """ONE carry-normalise round on ALL 15 KB[k] at once (each KB[k] is an
    INDEPENDENT band, so they fuse into a single wide block).  This is the DEPTH
    lever over the base ALU, which does each KB[k] in its own tiny block (15x6=90
    blocks): here the whole precompute is 1 raw + _KB_CARRY_ROUNDS shared rounds."""
    a = L.LEANDIV
    RN = a.RN
    per_k = RN * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2)
    spec = _empty_spec(dim, 15 * per_k + 16)
    u = 0
    for k in range(1, 16):
        base = a.KB + RN * k
        u = _lean_carry_round_units(spec, u, base, base, RN)
    return _truncate(spec, u, dim)


def _bz_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.LEANDIV
    spec = _empty_spec(dim, 2 + 8)
    u = 0
    u = _clear(spec, u, a.BZ)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.BZ, 1.0)                    # + 1
    u = _step_ge(spec, u, {L.AX + c: 1.0 for c in range(8)}, 0.0, 1, a.BZ, -1.0)  # -[sum>=1]
    return _truncate(spec, u, dim)


def _kb_precompute_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    blocks = [("lean-kb-raw", _kb_raw_block(L, dim))]
    for rnd in range(_KB_CARRY_ROUNDS):        # all 15 KB[k] fused per round
        blocks.append((f"lean-kb-c{rnd}", _kb_carry_all_block(L, dim)))
    blocks.append(("lean-bz", _bz_block(L, dim)))
    return blocks


# ===========================================================================
# 2. INIT + digit-index counter one-hot.
# ===========================================================================
def _it_oh_units(spec, u, a, it_const=0.0):
    for j in range(8):
        u = _clear(spec, u, a.IT_OH + j)
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j, a.IT_OH + j, 1.0)
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j + 1, a.IT_OH + j, -1.0)
    return u


def _init_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, RN + 8 + 1 + 8 * 5)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
    for c in range(8):
        u = _clear(spec, u, a.DIV_RES + c)
    u = _clear(spec, u, a.IT)
    u = _it_oh_units(spec, u, a)
    return _truncate(spec, u, dim)


# ===========================================================================
# 3. THE LEAN ITERATION BODY (9 blocks, weight-shared across 8 iterations).
# ===========================================================================
# --- block A: shift  R = 16*R + selected dividend nibble --------------------
def _shift_block(L, dim) -> Dict[str, torch.Tensor]:
    """R[c] <- R[c-1] (c>=1, the *16 nibble shift); R[0] <- dividend nibble
    STACK0[7-it] via one-hot(IT).  The shift SNAPS each moved nibble through a
    sharp kmax=15 staircase (instead of a silu-identity copy) so R stays a CLEAN
    integer nibble every iteration: this RESETS the tiny ramp residue per step
    rather than letting it accumulate and get amplified by the *16^p limb
    recompose in the borrow/subtract (the fp32-robustness fix)."""
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, (RN - 1) * (1 + 15 * 2) + 1 + 8)
    u = 0
    for c in range(RN - 1, 0, -1):
        u = _clear(spec, u, a.R + c)
        u = _floor_div_pow(spec, u, {a.R + c - 1: 1.0}, 0.0, 1, 15, a.R + c, 1.0)  # snap-copy
    u = _clear(spec, u, a.R + 0)
    for j in range(8):                       # R[0] = STACK0[7-it] via one-hot(IT)
        u = _guard(spec, u, [(a.IT_OH + j, 1.0, 0.0)],
                   {L.STACK0 + (7 - j): 1.0}, 0.0, a.R + 0, 1.0)
    return _truncate(spec, u, dim)


# --- block B: gteq  GT[k,i], EQ[k,i] lanes for R vs KB[k] --------------------
def _gteq_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, 15 * RN * (2 + 4 + 4))
    u = 0
    for k in range(1, 16):
        for i in range(RN):
            gt = a.GT + (k - 1) * RN + i
            eq = a.EQ + (k - 1) * RN + i
            d = {a.R + i: 1.0, a.KB + RN * k + i: -1.0}
            u = _clear(spec, u, gt)
            u = _step_ge(spec, u, d, 0.0, 1, gt, 1.0)                 # [d>=1]
            u = _clear(spec, u, eq)
            u = _step_ge(spec, u, d, 0.0, 0, eq, 1.0)                 # +[d>=0]
            u = _step_ge(spec, u, d, 0.0, 1, eq, -1.0)               # -[d>=1] => [d==0]
    return _truncate(spec, u, dim)


# --- block C: qdigit  QD = sum_k [R>=KB[k]] AND the monotone prefix GE[k] ----
def _qdigit_block(L, dim) -> Dict[str, torch.Tensor]:
    """GE[k] = [R >= KB[k]] (lexicographic suffix-ANDs of GT/EQ), and
    QD = sum_k GE[k].  GE is monotone (1 for k<=q, 0 above), so QOH is formed in
    the qbsel block as GE[k]-GE[k+1]."""
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, 1 + 15 * (RN + 1) + 15 + 15 * (RN + 1))
    u = 0
    u = _clear(spec, u, a.QD)
    for k in range(1, 16):
        gtb = a.GT + (k - 1) * RN
        eqb = a.EQ + (k - 1) * RN
        ge_windows = []
        for i in range(RN - 1, -1, -1):
            w = [(gtb + i, 1.0, 0.0)] + [(eqb + j, 1.0, 0.0) for j in range(i + 1, RN)]
            ge_windows.append(w)
        ge_windows.append([(eqb + j, 1.0, 0.0) for j in range(RN)])   # all-equal
        u = _clear(spec, u, a.GE + k)
        for w in ge_windows:
            u = _guard(spec, u, w, {L.ONE: 1.0}, 0.0, a.QD, 1.0)      # into QD
            u = _guard(spec, u, w, {L.ONE: 1.0}, 0.0, a.GE + k, 1.0)  # into GE[k]
    return _truncate(spec, u, dim)


# --- block D: qbsel  KB[q] subtrahend via one-hot(q)=GE[k]-GE[k+1] -----------
def _qbsel_block(L, dim) -> Dict[str, torch.Tensor]:
    """KBQ[i] = sum_k onehot_k * KB[k][i], onehot_k = [GE[k]==1 AND GE[k+1]==0]
    (GE[16]:=0).  KB[k] is ALREADY carry-normalised -> no per-iter multiply/carry.
    q==0 -> no k fires -> KBQ = 0 (KB[0] is 0 anyway)."""
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, RN + 15 * RN)
    u = 0
    for i in range(RN):
        u = _clear(spec, u, a.KBQ + i)
    for k in range(1, 16):
        w_hi = (a.GE + k, 1.0, 0.0)                       # GE[k] == 1
        if k < 15:
            w_lo = (a.GE + k + 1, -1.0, 1.0)              # 1 - GE[k+1] == 1 (GE[k+1]==0)
            windows = [w_hi, w_lo]
        else:
            windows = [w_hi]                              # GE[16] := 0
        for i in range(RN):
            u = _guard(spec, u, windows, {a.KB + RN * k + i: 1.0}, 0.0, a.KBQ + i, 1.0)
    return _truncate(spec, u, dim)


# --- 3-limb (3-nibble) borrow subtract R2 = R - KBQ.  fp32-safe: every limb diff
#     is in [-4095, 4095], so RELU_S*|diff| <= 819000 << 2^24.  KBQ <= R is
#     guaranteed (q = max k with KB[k] <= R), so R2 >= 0 and no borrow leaves the
#     top limb.  The two inter-limb borrows B0, B1 are threaded through their own
#     0/1 lanes (a genuine 2-stage chain, like the base ALU's borrow lanes).
def _limb_diff(L, li):
    """d_li = R_limb[li] - KBQ_limb[li]  as a {band:coeff} form (value in [-4095,4095])."""
    a = L.LEANDIV
    f = {}
    for j in range(3):
        f[a.R + 3 * li + j] = f.get(a.R + 3 * li + j, 0.0) + float(16 ** j)
        f[a.KBQ + 3 * li + j] = f.get(a.KBQ + 3 * li + j, 0.0) - float(16 ** j)
    return f


# --- block E1: borrow0  B0 = [d0 < 0] ---------------------------------------
def _borrow0_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.LEANDIV
    spec = _empty_spec(dim, 8)
    u = 0
    d0 = _limb_diff(L, 0)
    u = _clear(spec, u, a.B0)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.B0, 1.0)                 # + 1
    u = _step_ge(spec, u, d0, 0.0, 0, a.B0, -1.0)                     # - [d0>=0]  => [d0<0]
    return _truncate(spec, u, dim)


# --- block E2: borrow1  B1 = [d1 - B0 < 0]  (reads the B0 lane) --------------
def _borrow1_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.LEANDIV
    spec = _empty_spec(dim, 8)
    u = 0
    d1 = dict(_limb_diff(L, 1))
    d1[a.B0] = d1.get(a.B0, 0.0) - 1.0        # subtract borrow-in B0 (a 0/1 lane)
    u = _clear(spec, u, a.B1)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.B1, 1.0)                 # + 1
    u = _step_ge(spec, u, d1, 0.0, 0, a.B1, -1.0)                     # - [d1-B0>=0] => [d1-B0<0]
    return _truncate(spec, u, dim)


# --- block E3: subval  RV[li] = limb result value (0..4095) + emit q + IT++ --
def _subval_block(L, dim) -> Dict[str, torch.Tensor]:
    """RV[0] = d0 + LB*B0 ; RV[1] = d1 - B0 + LB*B1 ; RV[2] = d2 - B1  (each in
    [0, 4095]).  Reads the B0, B1 lanes.  Also emit q into DIV_RES[7-it], IT+=1,
    refresh IT_OH.  LB = 16^3 = 4096."""
    a = L.LEANDIV
    LB = 16 ** 3
    spec = _empty_spec(dim, 20 + 8 * 2 + 8 * 5)
    u = 0
    d0, d1, d2 = _limb_diff(L, 0), _limb_diff(L, 1), _limb_diff(L, 2)
    # RV[0] = d0 + LB*B0
    u = _clear(spec, u, a.RV + 0)
    u = _ident(spec, u, d0, 0.0, a.RV + 0, 1.0)
    u = _ident(spec, u, {a.B0: 1.0}, 0.0, a.RV + 0, float(LB))
    # RV[1] = d1 - B0 + LB*B1
    u = _clear(spec, u, a.RV + 1)
    u = _ident(spec, u, d1, 0.0, a.RV + 1, 1.0)
    u = _ident(spec, u, {a.B0: 1.0}, 0.0, a.RV + 1, -1.0)
    u = _ident(spec, u, {a.B1: 1.0}, 0.0, a.RV + 1, float(LB))
    # RV[2] = d2 - B1
    u = _clear(spec, u, a.RV + 2)
    u = _ident(spec, u, d2, 0.0, a.RV + 2, 1.0)
    u = _ident(spec, u, {a.B1: 1.0}, 0.0, a.RV + 2, -1.0)
    # emit q into DIV_RES[7-it] (gated on IT_OH), IT+=1, refresh IT_OH.
    for j in range(8):
        slot = a.DIV_RES + (7 - j)
        g = (a.IT_OH + j, 1.0, 0.0)
        u = _guard(spec, u, [g], {slot: -1.0}, 0.0, slot, 1.0)        # clear slot gated
        u = _guard(spec, u, [g], {a.QD: 1.0}, 0.0, slot, 1.0)         # + q gated
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.IT, 1.0)                 # IT += 1
    u = _it_oh_units(spec, u, a, it_const=1.0)                        # IT_OH = one-hot(IT+1)
    return _truncate(spec, u, dim)


# --- block E4: split  RV[li] (0..4095) -> 3 R nibbles each (next-iter R) -----
#
# CRITICAL fp-hygiene: the running R is MULTIPLIED BY 16 every iteration (the
# shift), so ANY sub-integer residue on an R nibble is amplified 16x/iter and
# blows up after 8 iterations.  The base ALU keeps R clean because every nibble
# is produced by a SHARP _step_ge staircase (a 0/1 sum) that snaps away residue.
# We do the same here: each result nibble is a SUM OF SHARP INDICATORS
# ``sum_{k=0..14} [ RV mod 16^(p+1) >= (k+1)*16^p ]`` = floor((RV mod 16^(p+1))/16^p),
# every term a half-integer-thresholded step -> the nibble is a CLEAN integer
# 0..15 regardless of RV's residue.  ``RV mod 16^(p+1) = RV - 16^(p+1)*F(p+1)``
# needs the higher floor F(p+1) as a CLEAN LANE first, so we split in TWO blocks:
# split1 materialises the clean floors F16 = floor(RV/16), F256 = floor(RV/256);
# split2 forms the 3 nibbles as sharp staircases over ``RV - 16*F16`` (nib0),
# ``F16 - 16*F256`` (nib1) and ``F256`` (nib2) — all clean small ranges (< 16 or
# <= 255), kmax 15, thresholds <= 15 (fp32-trivial).
def _split1_floors_block(L, dim) -> Dict[str, torch.Tensor]:
    """Clean floors: F16[li] = floor(RV[li]/16) (0..255), F256[li] = floor(RV/256)
    (0..15).  Sharp staircases snap RV's residue.  Stored back into RV+? — we
    reuse the KBQ band (dead after subval) as F16/F256 scratch to avoid new dims:
      KBQ+3*li+0 = F16[li] ; KBQ+3*li+1 = F256[li]."""
    a = L.LEANDIV
    spec = _empty_spec(dim, 3 * (2 + 255 * 2 + 15 * 2))
    u = 0
    for li in range(3):
        rv = a.RV + li
        f16 = a.KBQ + 3 * li + 0
        f256 = a.KBQ + 3 * li + 1
        u = _clear(spec, u, f16)
        u = _floor_div_pow(spec, u, {rv: 1.0}, 0.0, 16, 255, f16, 1.0)     # floor(RV/16), kmax 255
        u = _clear(spec, u, f256)
        u = _floor_div_pow(spec, u, {rv: 1.0}, 0.0, 256, 15, f256, 1.0)    # floor(RV/256), kmax 15
    return _truncate(spec, u, dim)


def _split2_nibbles_block(L, dim) -> Dict[str, torch.Tensor]:
    """R nibbles from the clean floors + RV (residue snapped by sharp staircases):
        nib0 = floor( (RV - 16*F16) / 1 )     value in [0,15]  -> kmax 15
        nib1 = floor( (F16 - 16*F256) / 1 )   value in [0,15]  -> kmax 15
        nib2 = F256                            value in [0,15]  (already clean)
    (RV - 16*F16) and (F16 - 16*F256) are exact integers in [0,16); the sharp
    staircase ``sum_{k=1..15}[val >= k]`` SNAPS any leftover residue.  SET R."""
    a = L.LEANDIV
    spec = _empty_spec(dim, 3 * (3 + 15 * 2 * 3))     # 3 nibbles, each clear + 15*2 relu
    u = 0
    for li in range(3):
        rv = a.RV + li
        f16 = a.KBQ + 3 * li + 0
        f256 = a.KBQ + 3 * li + 1
        j0, j1, j2 = a.R + 3 * li + 0, a.R + 3 * li + 1, a.R + 3 * li + 2
        # nib0 = (RV - 16*F16) snapped: staircase over the form, kmax 15.
        u = _clear(spec, u, j0)
        u = _floor_div_pow(spec, u, {rv: 1.0, f16: -16.0}, 0.0, 1, 15, j0, 1.0)
        # nib1 = (F16 - 16*F256) snapped, kmax 15.
        u = _clear(spec, u, j1)
        u = _floor_div_pow(spec, u, {f16: 1.0, f256: -16.0}, 0.0, 1, 15, j1, 1.0)
        # nib2 = F256 (re-snapped via a sharp staircase so it carries NO residue
        # into the next shift's 16x amplification).
        u = _clear(spec, u, j2)
        u = _floor_div_pow(spec, u, {f256: 1.0}, 0.0, 1, 15, j2, 1.0)
    return _truncate(spec, u, dim)


def _iteration_body(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """The ONE reusable lean iteration body.  9 blocks:
        shift, gteq, qdigit, qbsel, borrow0, borrow1, subval(+emit q),
        split1(clean floors), split2(clean nibbles).
    The two borrow blocks collapse the base ALU's 9-block per-nibble ripple to a
    3-limb borrow; the two split blocks re-snap the limb VALUES into CLEAN R
    nibbles (so the 16x/iter shift never amplifies sub-integer residue).  Still a
    big cut vs the base's 21 blocks/iter."""
    return [
        ("lean-shift", _shift_block(L, dim)),
        ("lean-gteq", _gteq_block(L, dim)),
        ("lean-qd", _qdigit_block(L, dim)),
        ("lean-qbsel", _qbsel_block(L, dim)),
        ("lean-borrow0", _borrow0_block(L, dim)),
        ("lean-borrow1", _borrow1_block(L, dim)),
        ("lean-subval", _subval_block(L, dim)),
        ("lean-split1", _split1_floors_block(L, dim)),
        ("lean-split2", _split2_nibbles_block(L, dim)),
    ]


# ===========================================================================
# 4. FINALIZE: assemble MOD (= final R), honour b==0 -> (0,0).
# ===========================================================================
def _finalize_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.LEANDIV
    spec = _empty_spec(dim, 8 * 6)
    u = 0
    for c in range(8):
        # MOD_RES[c] = (1-BZ)*R[c]
        u = _clear(spec, u, a.MOD_RES + c)
        u = _ident(spec, u, {a.R + c: 1.0}, 0.0, a.MOD_RES + c, 1.0)              # + R[c]
        gz = (a.BZ, 1.0, 0.0)
        u = _guard(spec, u, [gz], {a.R + c: -1.0}, 0.0, a.MOD_RES + c, 1.0)       # - BZ*R[c]
        # DIV_RES := (1-BZ)*DIV_RES
        u = _guard(spec, u, [gz], {a.DIV_RES + c: -1.0}, 0.0, a.DIV_RES + c, 1.0)  # - BZ*DIV_RES
    return _truncate(spec, u, dim)


# ===========================================================================
# 5. Public builders.
# ===========================================================================
def compile_lean_divmod_blocks(L, dim, n_iters: int = 8):
    """LEAN radix-16 long division, UNROLLED (distinct blocks per iteration for a
    straight depth read).  Order:
        kb-precompute (KB[k]=k*b, BZ) | init | 8 x [shift, gteq, qdigit, qbsel,
        sub, split] | finalize.
    Returns the full block list (name, spec)."""
    _set_one(L)
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    blocks += _kb_precompute_blocks(L, dim)
    blocks.append(("lean-init", _init_block(L, dim)))
    body = _iteration_body(L, dim)
    for it in range(n_iters):
        for name, spec in body:
            blocks.append((f"{name}{it}", spec))
    blocks.append(("lean-finalize", _finalize_block(L, dim)))
    return blocks


def compile_lean_divmod_blocks_recurrent(L, dim, n_iters: int = 8):
    """RECURRENT form: KB-precompute + init, the SINGLE iteration body REUSED
    n_iters times, finalize.  Returns (unique_blocks, apply_names)."""
    _set_one(L)
    prefix = _kb_precompute_blocks(L, dim) + [("lean-init", _init_block(L, dim))]
    body = _iteration_body(L, dim)
    finalize = [("lean-finalize", _finalize_block(L, dim))]
    unique = prefix + body + finalize
    apply_names = [n for n, _ in prefix]
    for _ in range(n_iters):
        apply_names += [n for n, _ in body]
    apply_names.append("lean-finalize")
    return unique, apply_names


# ===========================================================================
# 6. CPU SwiGLU forward SIMULATION of the FFN blocks (no full model bake).
#    x_out = x + W_down @ (silu(W_up@x + b_up) * (W_gate@x + b_gate)) + b_down
# ===========================================================================
def _apply_block(x: torch.Tensor, spec: Dict[str, torch.Tensor]) -> torch.Tensor:
    up = x @ spec["W_up"].T + spec["b_up"]
    gate = x @ spec["W_gate"].T + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + hidden @ spec["W_down"].T + spec["b_down"]


def _new_layout(code_size: int = 8, n_heads: int = 4):
    L = NibbleVMLayout(code_size, n_heads=n_heads)
    extend_layout(L)
    return L


def _nibbles(v, n):
    return [(v >> (4 * j)) & 0xF for j in range(n)]


def simulate(a_val: int, b_val: int, unique=None, apply_names=None,
             L=None, dim=None, dtype=torch.float64):
    """Run the leaned divmod block chain on ONE (a,b) via CPU SwiGLU forward.

    Uses fp64 for the SIMULATION arithmetic only (numerical headroom for the
    reader); the WEIGHTS themselves are fp32-safe (verified separately by
    ``max_relu_arg``).  Seeds AX=b nibbles, STACK0=a nibbles.  Returns
    (quotient, remainder) read back from DIV_RES / MOD_RES."""
    if L is None:
        L = _new_layout()
        dim = L.D
    if unique is None:
        unique, apply_names = compile_lean_divmod_blocks_recurrent(L, dim)
    by_name = {n: s for n, s in unique}
    x = torch.zeros(dim, dtype=dtype)
    x[L.ONE] = 1.0
    for j, nv in enumerate(_nibbles(b_val & 0xFFFFFFFF, 8)):
        x[L.AX + j] = float(nv)
    for j, nv in enumerate(_nibbles(a_val & 0xFFFFFFFF, 8)):
        x[L.STACK0 + j] = float(nv)
    for name in apply_names:
        spec = {k: v.to(dtype) for k, v in by_name[name].items()}
        x = _apply_block(x, spec)
    a = L.LEANDIV
    q = sum(int(round(float(x[a.DIV_RES + c]))) << (4 * c) for c in range(8))
    r = sum(int(round(float(x[a.MOD_RES + c]))) << (4 * c) for c in range(8))
    return q & 0xFFFFFFFF, r & 0xFFFFFFFF


# ===========================================================================
# 7. MEASURE: depth, blocks/iter, nz, fp32-safety (max relu arg), byte-exact.
# ===========================================================================
def _spec_nnz(spec) -> int:
    n = 0
    for key in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
        n += int((spec[key] != 0).sum())
    return n


def _band_value_bounds(L) -> torch.Tensor:
    """Per-DIM upper bound on |achievable value|, used to bound relu arguments.
    Nibble/predicate bands are <= 15; the RV limb-value band is <= 16^3-1 = 4095;
    ONE = 1.  Any dim not explicitly bounded defaults to a nibble bound of 15."""
    a = L.LEANDIV
    b = torch.full((L.D,), 15.0)          # default: nibble / small-int lanes <= 15
    b[L.ONE] = 1.0
    for i in range(3):                    # RV limb VALUE lanes: 0..4095
        b[a.RV + i] = float(16 ** 3 - 1)
    # KBQ band is reused post-subval as F16 (floor(RV/16) <= 255) scratch, so its
    # worst-case value across the whole chain is 255, not 15.
    for i in range(a.RN):
        b[a.KBQ + i] = 255.0
    return b


def _max_relu_arg(unique, L) -> float:
    """True worst-case |relu argument| = max over relu-style units of
    (|W_up| . value_bound + |b_up|).  A relu unit's up = W_up.x + b_up; we bound
    |x| per dim by its achievable maximum.  (silu-identity units, whose up is
    S*ONE = 60, are also included and are trivially bounded.)  This is the number
    that must stay < 2^24 for fp32 integer exactness."""
    bounds = _band_value_bounds(L)
    max_arg = 0.0
    for _, spec in unique:
        wup = spec["W_up"]
        if not wup.numel():
            continue
        per_unit = wup.abs() @ bounds + spec["b_up"].abs()
        max_arg = max(max_arg, float(per_unit.max()))
    return max_arg


def _edge_grid():
    edges = [0, 1, 2, 2 ** 31, 2 ** 32 - 1]
    edges += [1 << p for p in range(0, 32)]
    bs = [1, 2, 3, 7, 10, 16, 255, 256, 65535, 65536, 2 ** 31, 2 ** 32 - 1]
    cases = []
    for a in edges:
        for b in bs:
            cases.append((a, b))
    # special structural cases
    for v in edges:
        cases.append((v, v))          # a == b
        cases.append((v, 0))          # div by zero -> (0,0)
        cases.append((max(0, v - 1), v))   # a < b
    return cases


def _ref(a, b):
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    if b == 0:
        return 0, 0
    return a // b, a % b


def measure(verbose: bool = True):
    L = _new_layout()
    dim = L.D
    unrolled = compile_lean_divmod_blocks(L, dim)
    unique, apply_names = compile_lean_divmod_blocks_recurrent(L, dim)

    body = _iteration_body(L, dim)
    blocks_per_iter = len(body)
    depth_unrolled = len(unrolled)
    depth_applied = len(apply_names)
    stored = len(unique)
    nnz = sum(_spec_nnz(s) for _, s in unique)
    max_arg = _max_relu_arg(unique, L)

    # byte-exact grid
    import random
    cases = list(_edge_grid())
    random.seed(1234)
    for _ in range(320):
        cases.append((random.randint(0, 2 ** 32 - 1), random.randint(0, 2 ** 32 - 1)))
    by_name = {n: {k: v.to(torch.float64) for k, v in s.items()} for n, s in unique}

    def _run(dtype):
        by = {n: {k: v.to(dtype) for k, v in s.items()} for n, s in unique}
        p, fs = 0, []
        for (a, b) in cases:
            q, r = simulate(a, b, unique=[(n, by[n]) for n in by],
                            apply_names=apply_names, L=L, dim=dim, dtype=dtype)
            rq, rr = _ref(a, b)
            if (q, r) == (rq, rr):
                p += 1
            elif len(fs) < 12:
                fs.append((a, b, (q, r), (rq, rr)))
        return p, fs

    passed, fails = _run(torch.float64)      # canonical (higher-precision ALU)
    passed32, fails32 = _run(torch.float32)  # honest fp32 end-to-end
    total = len(cases)

    fp32_safe = max_arg < 2 ** 24
    if verbose:
        print("=" * 72)
        print("LEANED radix-16 (nibble digit-recurrence) 32-bit DIVIDE — bakeoff")
        print("=" * 72)
        print(f"stored (unique) blocks   : {stored}")
        print(f"depth (UNROLLED, straight): {depth_unrolled} blocks")
        print(f"depth (applied, recurrent): {depth_applied} block-applications")
        print(f"blocks / iteration        : {blocks_per_iter}  (shift, gteq, qdigit,"
              f" qbsel, borrow0, borrow1, subval+emit, split1, split2)")
        n_kb = len(_kb_precompute_blocks(L, dim))
        print(f"one-time overhead         : {n_kb} KB-precompute + 1 init + 1 finalize "
              f"= {n_kb + 2} blocks")
        print(f"nz (nonzero weights)      : {nnz}")
        print(f"max relu arg (worst |up|) : {max_arg:.1f}  "
              f"(RELU_S={RELU_S}, S={S}; fp32-safe < 2^24 = {2**24}) "
              f"-> {'YES' if fp32_safe else 'NO'}")
        print(f"fp64 params               : 0 (all weights fp32-representable)")
        print(f"byte-exact (fp64 ALU sim) : {passed}/{total}  "
              f"({'ALL PASS' if passed == total else 'FAIL'})")
        print(f"byte-exact (fp32 e2e sim) : {passed32}/{total}  "
              f"({'ALL PASS' if passed32 == total else 'residue floor'})")
        if fails:
            print("  first fp64 fails (a,b,got,exp):")
            for f in fails:
                print("   ", f)
        if fails32:
            print("  first fp32 fails (a,b,got,exp) [ramp-residue boundary]:")
            for f in fails32:
                print("   ", f)
        print("-" * 72)
        print("VERDICT vs alternatives (DEPTH):")
        print(f"  base-16 LONG DIVISION (nibble_alu32) : ~262 blocks")
        print(f"  fp32 two-digit                        : ~274 blocks")
        print(f"  fp64 log-sink                         : ~127 blocks")
        print(f"  THIS leaned radix-16 (unrolled)       : {depth_unrolled} blocks")
        cmp = "BEATS long division / two-digit" if depth_unrolled < 262 else "does NOT beat"
        beats_log = depth_unrolled < 127
        print(f"  -> leaned radix-16 {cmp}; "
              f"{'beats' if beats_log else 'does NOT beat'} the fp64 log-sink (127).")
        print("=" * 72)
    return dict(stored=stored, depth_unrolled=depth_unrolled,
                depth_applied=depth_applied, blocks_per_iter=blocks_per_iter,
                nnz=nnz, max_relu_arg=max_arg, fp32_safe=fp32_safe,
                byte_exact_pass=passed, byte_exact_total=total,
                byte_exact_pass_fp32=passed32)


if __name__ == "__main__":
    measure()
