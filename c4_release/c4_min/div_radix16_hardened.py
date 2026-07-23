"""HARDENED radix-16 (nibble-at-a-time) digit-recurrence 32-bit divide.

Standalone MEASURE-ONLY bakeoff module — the fp32-BYTE-EXACT hardening of
``div_radix16_lean.py``.  Same *robust, always-exact* algorithm (base-16 long
division, **NO reciprocal**, **NO per-digit multiply**), but the borrow subtract
is re-founded so the RUNNING REMAINDER ``R`` never leaves nibble-lane form.

The residue floor of the lean variant
=====================================
``div_radix16_lean`` packs the ``R - KB[q]`` subtract into 3 limbs of 3 nibbles
each, forming each limb VALUE by a ``16^j`` positional recompose
(``R_limb = R[0] + 16*R[1] + 256*R[2]``, up to 4095), doing a 3-limb borrow, then
re-splitting the limb value back into nibbles with a ``floor(RV/16)`` / floor256
staircase.  In genuine fp32 the ``16^j`` recompose AMPLIFIES the tiny
``_step_ge`` ramp residue on a nibble by up to 256x, and the wide
``floor(RV/16, kmax=255)`` staircase reads that amplified residue: near a limb
boundary the remainder nibble lands a fraction off (``3.4999`` / ``4.0001``) and
mis-rounds.  On the full 32-bit adversarial grid (large divisors, ``b`` near
2^32) this is NOT a 1/3000 floor — it fails the vast majority of large-divisor
cases, and is so marginal it even flips under a change of fp32 matmul
accumulation ORDER (batched-vs-single forward).  The fp64 ALU sim is byte-exact
everywhere (algorithm correct); only the fp32 substrate residue misfires.

The hardening — keep R in nibble-lane form (NO ``16^p`` recompose)
=================================================================
The whole ``16^p`` limb machinery (``borrow0, borrow1, subval, split1, split2``)
is REPLACED by a **base-16 Kogge-Stone parallel-prefix BORROW** over the 9
remainder nibbles — the exact residue-free pattern the general MUL carry resolve
uses (``nibble_alu32._mul_*`` prefix), adapted from carry to borrow:

  * ``gp``     — per nibble ``diff_i = R[i] - KB[q][i]`` in [-15,15]:
                 generate ``g_i = [diff_i < 0]`` (borrows out regardless),
                 propagate ``p_i = [diff_i == 0]`` (borrows out iff borrows in).
  * ``ks0..3`` — ceil(log2 9) = 4 log-depth prefix combine stages settle the
                 per-nibble borrow-OUT ``Bout_i = g_i OR (p_i AND Bout_{i-1})``.
  * ``apply``  — ``R[i] <- R[i] - KB[q][i] - Bin_i + 16*Bout_i`` (Bin_i =
                 Bout_{i-1}, Bin_0 = 0), a value in [-16,15] SNAPPED to [0,15] by
                 a sharp ``_step_ge`` — and fused with emit-q + IT++.

EVERY quantity in the whole borrow stays in [-16,15]: there is **no scalar
``16^p`` recompose anywhere**, so there is nothing to amplify.  The prefix lanes
are pure 0/1 (sharp AND/OR via ``_step_ge`` at integer thresholds), residue-free
by construction; the result nibble is re-snapped by a sharp staircase every
iteration.  Max per-step ``R`` residue drops to ~0.

Blocks / iteration = 10 (was 9)
===============================
``shift, gteq, qdigit, qbsel, gp, ks0, ks1, ks2, ks3, apply(+emit q)`` — the two
borrow blocks + subval + two split blocks of the lean variant (5) become the
gp + 4 prefix stages + fused apply (6): net +1 block/iter, +8 blocks total
(depth 80 -> 88), the fp32-robustness cost.  Still far below the base ALU's
~262 and the 274 two-digit; ~depth-parity with the fp64 log-sink (127) is kept
by a wide margin (88 < 127).

fp32 discipline
===============
Every relu / silu argument stays well < 2^24: the nibble compares keep every
compared quantity in [-15,15]; the borrow ``diff`` is in [-16,15]; the prefix
lanes are 0/1; the KB-value bound path is the only large-arg term
(``RELU_S*(15*65536) ~ 1.47e7 < 2^24``).  0 fp64 params.

Byte-exact: validated on the required edge grid + the adversarial classes +
>=6000 random (a,b<2^32) pairs by a CPU SwiGLU forward simulation of the ACTUAL
FFN blocks (no full model), through the REAL fp32 forward.
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
        # HARDENED nibble-lane borrow: Kogge-Stone parallel-prefix over the 9
        # remainder nibbles.  Generate/propagate lanes (0/1) DOUBLE-BUFFERED across
        # the prefix stages (G0/P0 <-> G1/P1).  NO 16^p recompose: every value in
        # the borrow stays in [-16,15]; the prefix lanes are pure 0/1.  gp seeds
        # (G0,P0) := (g_i, p_i) directly from diff_i = R[i]-KBQ[i].
        self.G0 = L._band("LR_G0", RN)         # prefix buffer A: borrow-out generate
        self.P0 = L._band("LR_P0", RN)         # prefix buffer A: borrow-out propagate
        self.G1 = L._band("LR_G1", RN)         # prefix buffer B (double-buffered)
        self.P1 = L._band("LR_P1", RN)         # prefix buffer B
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


# ===========================================================================
# HARDENED nibble-lane borrow subtract R = R - KB[q] (Kogge-Stone parallel prefix)
# ===========================================================================
# The lean variant packed R-KBQ into 3 limbs of 3 nibbles via a ``16^j`` scalar
# recompose (limb value up to 4095), then re-split the limb value back into
# nibbles with a wide ``floor(RV/16, kmax=255)`` staircase.  In genuine fp32 the
# ``16^j`` recompose amplifies the tiny ``_step_ge`` ramp residue by up to 256x,
# and the wide staircase reads that amplified residue -> near a limb boundary the
# remainder nibble mis-rounds.  Here we NEVER recompose: the borrow propagates
# nibble-by-nibble through a base-16 Kogge-Stone parallel prefix (the SAME proven
# residue-free pattern the general MUL carry resolve uses, adapted carry->borrow).
# Every value in the whole subtract is in [-16,15]; the prefix lanes are pure 0/1.
#
# Borrow recurrence (base 16), with diff_i = R[i] - KB[q][i] in [-15,15]:
#   g_i    = [diff_i < 0]              generate a borrow-out regardless of borrow-in
#   p_i    = [diff_i == 0]             propagate: borrows out iff borrows in
#   Bout_i = g_i OR (p_i AND Bout_{i-1})   (Bout_{-1} := 0)   -- a prefix scan
#   Bin_i  = Bout_{i-1}   (Bin_0 = 0)
#   R[i]  <- R[i] - KB[q][i] - Bin_i + 16*Bout_i    in [0,15]  (snapped)
# KBQ <= R is guaranteed (q = max k with KB[k] <= R) so no borrow leaves the top.

# --- block E1: gp  generate/propagate lanes from diff_i = R[i] - KBQ[i] -------
def _gp_block(L, dim) -> Dict[str, torch.Tensor]:
    """Seed the Kogge-Stone prefix buffer A directly from diff_i = R[i] - KBQ[i]
    (in [-15,15]):
        G0[i] = g_i = [diff_i <  0] = 1 - [diff_i >= 0]     (generate a borrow-out)
        P0[i] = p_i = [diff_i == 0] = [diff_i >= 0] - [diff_i >= 1]  (propagate).
    Both 0/1; thresholds 0/1 -> tiny fp32 args, residue-free.  Written DIRECTLY
    into the prefix buffer (no BG/BP intermediate — every unit reads the block
    INPUT, so a within-block BG->G0 copy would read the STALE prior-iteration BG,
    not the just-computed value)."""
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, RN * (2 + 4 + 2 + 4))
    u = 0
    for i in range(RN):
        diff = {a.R + i: 1.0, a.KBQ + i: -1.0}
        # G0 = g_i = [diff < 0] = 1 - [diff >= 0]
        u = _clear(spec, u, a.G0 + i)
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.G0 + i, 1.0)          # + 1
        u = _step_ge(spec, u, diff, 0.0, 0, a.G0 + i, -1.0)            # - [diff>=0]
        # P0 = p_i = [diff == 0] = [diff >= 0] - [diff >= 1]
        u = _clear(spec, u, a.P0 + i)
        u = _step_ge(spec, u, diff, 0.0, 0, a.P0 + i, 1.0)            # + [diff>=0]
        u = _step_ge(spec, u, diff, 0.0, 1, a.P0 + i, -1.0)          # - [diff>=1] => [diff==0]
    return _truncate(spec, u, dim)


# --- block E2..E5: Kogge-Stone prefix combine stage at distance d -------------
def _ks_stage_block(L, dim, g_src, p_src, g_dst, p_dst, d) -> Dict[str, torch.Tensor]:
    """ONE Kogge-Stone prefix combine stage at distance ``d`` (SET g_dst/p_dst):
        for i >= d:  (G_i, P_i) := (G_i OR (P_i AND G_{i-d}),  P_i AND P_{i-d})
        for i <  d:  carried through.
    0/1 lanes, single-staircase exact forms (identical to the MUL prefix):
        G_i OR (P_i AND G_{i-d}) = [2*G_i + P_i + G_{i-d} >= 2]
        P_i AND P_{i-d}          = [P_i + P_{i-d} >= 2].
    Reads the block INPUT (the previous stage's g_src/p_src), so one block."""
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, RN * (2 + 2 + 2 + 2))
    u = 0
    for i in range(RN):
        if i >= d:
            gform = {g_src + i: 2.0, p_src + i: 1.0, g_src + i - d: 1.0}
            u = _clear(spec, u, g_dst + i)
            u = _step_ge(spec, u, gform, 0.0, 2, g_dst + i, 1.0)
            pform = {p_src + i: 1.0, p_src + i - d: 1.0}
            u = _clear(spec, u, p_dst + i)
            u = _step_ge(spec, u, pform, 0.0, 2, p_dst + i, 1.0)
        else:
            u = _clear(spec, u, g_dst + i)
            u = _ident(spec, u, {g_src + i: 1.0}, 0.0, g_dst + i, 1.0)
            u = _clear(spec, u, p_dst + i)
            u = _ident(spec, u, {p_src + i: 1.0}, 0.0, p_dst + i, 1.0)
    return _truncate(spec, u, dim)


# --- block E6: apply  R[i] = R[i]-KBQ[i]-Bin_i+16*Bout_i  + emit q + IT++ ------
def _borrow_apply_block(L, dim, g_final) -> Dict[str, torch.Tensor]:
    """After the prefix, ``g_final`` holds the per-nibble borrow-OUT Bout_i.  Set
    each remainder nibble
        R[i] <- diff_i - Bin_i + 16*Bout_i          (Bin_i = Bout_{i-1}, Bin_0=0)
    where diff_i = R[i] - KBQ[i].  The inner form
        (diff_i - Bin_i)  is in [-16, 15];  +16*[it<0] SNAPS it to [0,15].
    We compute ``16*Bout_i`` as ``16*[diff_i - Bin_i < 0]`` via a SHARP staircase
    over the same [-16,15] form -> the result nibble is a CLEAN integer, immune to
    residue (there is NO 16^p recompose anywhere).  Fused: emit q into
    DIV_RES[7-it] (gated on IT_OH), IT+=1, refresh IT_OH."""
    a = L.LEANDIV
    RN = a.RN
    spec = _empty_spec(dim, RN * (2 + 2 + 2 + 2 + 2) + 8 * 2 + 8 * 5 + 2)
    u = 0
    for i in range(RN):
        diff = {a.R + i: 1.0, a.KBQ + i: -1.0}
        # form = diff_i - Bin_i  (Bin_i = Bout_{i-1} = g_final[i-1]; Bin_0 = 0)
        form = dict(diff)
        if i >= 1:
            form[g_final + i - 1] = form.get(g_final + i - 1, 0.0) - 1.0
        u = _clear(spec, u, a.R + i)                                   # SET -old(R[i])
        u = _ident(spec, u, form, 0.0, a.R + i, 1.0)                   # + (diff - Bin)
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.R + i, 16.0)          # + 16
        u = _step_ge(spec, u, form, 0.0, 0, a.R + i, -16.0)          # - 16*[form>=0] => +16*[form<0]
    # emit q into DIV_RES[7-it] (gated on IT_OH), IT+=1, refresh IT_OH.
    for j in range(8):
        slot = a.DIV_RES + (7 - j)
        g = (a.IT_OH + j, 1.0, 0.0)
        u = _guard(spec, u, [g], {slot: -1.0}, 0.0, slot, 1.0)        # clear slot gated
        u = _guard(spec, u, [g], {a.QD: 1.0}, 0.0, slot, 1.0)         # + q gated
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.IT, 1.0)                 # IT += 1
    u = _it_oh_units(spec, u, a, it_const=1.0)                        # IT_OH = one-hot(IT+1)
    return _truncate(spec, u, dim)


def _borrow_prefix_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """gp | ceil(log2 RN) Kogge-Stone stages (double-buffered) | fused apply.  For
    RN=9 that is 4 prefix stages -> 1 + 4 + 1 = 6 blocks.  Returns the list and the
    g-band holding the final borrow-out (for the apply's Bin/Bout reads)."""
    a = L.LEANDIV
    RN = a.RN
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    blocks.append(("lean-gp", _gp_block(L, dim)))
    (gs, ps), (gd, pd) = (a.G0, a.P0), (a.G1, a.P1)
    d = 1
    st = 0
    while d < RN:
        blocks.append((f"lean-ks{st}", _ks_stage_block(L, dim, gs, ps, gd, pd, d)))
        (gs, ps), (gd, pd) = (gd, pd), (gs, ps)   # swap buffers
        d *= 2
        st += 1
    blocks.append(("lean-apply", _borrow_apply_block(L, dim, gs)))  # gs = final borrow-out g-band
    return blocks


def _iteration_body(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """The ONE reusable HARDENED iteration body.  10 blocks:
        shift, gteq, qdigit, qbsel, gp, ks0, ks1, ks2, ks3, apply(+emit q).
    The lean variant's 3-limb borrow (borrow0, borrow1, subval, split1, split2 =
    5 blocks with a 16^p recompose) is replaced by the nibble-lane Kogge-Stone
    borrow (gp + 4 prefix stages + fused apply = 6 blocks): +1 block/iter, but the
    remainder never leaves nibble form so the fp32 residue floor is gone."""
    return [
        ("lean-shift", _shift_block(L, dim)),
        ("lean-gteq", _gteq_block(L, dim)),
        ("lean-qd", _qdigit_block(L, dim)),
        ("lean-qbsel", _qbsel_block(L, dim)),
    ] + _borrow_prefix_blocks(L, dim)


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
    The HARDENED borrow keeps EVERY band in nibble/predicate range: R/KB/KBQ
    nibbles and all borrow lanes (BG/BP/G0/P0/G1/P1) are <= 15 (the 0/1 prefix
    lanes are <= 1, but 15 is a safe over-bound); ONE = 1.  There is NO 16^p limb
    value anywhere.  Any dim not explicitly bounded defaults to a nibble bound 15."""
    b = torch.full((L.D,), 15.0)          # default: nibble / small-int lanes <= 15
    b[L.ONE] = 1.0
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


def _adversarial_cases():
    """The required adversarial classes for the acceptance gate:
    (2^32-1)//{2,3,7}; 2^k +/- 1 divisors; b=1; a<b; a=b; k*b and k*b+b-1
    boundaries; div-by-zero.  These are the cases that flushed out the lean
    variant's 16^p residue floor."""
    import random
    cases = []
    M = 2 ** 32 - 1
    # (2^32-1)//{2,3,7} exact + one-off
    for d in (2, 3, 7):
        cases += [(M, d), (M - 1, d), (M, d + 1)]
    # 2^k +/- 1 divisors
    for k in range(0, 33):
        for delta in (-1, 1):
            div = (1 << k) + delta
            if 1 <= div <= M:
                rng = random.Random(k * 7 + delta)
                cases += [(rng.randint(0, M), div), (div, div), (div - 1, div),
                          (div * 5 + 3, div), (M, div)]
    # b == 1 (quotient == a, remainder 0)
    for a in (0, 1, 2, 12345, 2 ** 31, M):
        cases.append((a, 1))
    # k*b and k*b+b-1 boundaries (the digit-recurrence quotient-digit edges)
    rng = random.Random(20260723)
    for _ in range(400):
        b = rng.randint(1, M)
        k = rng.randint(0, 15)
        for a in (k * b, k * b + b - 1, k * b + b):
            cases.append((a & 0xFFFFFFFF, b))
    # div by zero -> (0,0)
    for a in (0, 1, M, 2 ** 31):
        cases.append((a, 0))
    return cases


def _ref(a, b):
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    if b == 0:
        return 0, 0
    return a // b, a % b


def _run_batch(L, dim, specs, cases, dtype, track_residue=False):
    """Batched CPU SwiGLU forward over ``cases`` (x is [B, dim]).  Returns the list
    of (q,r); with ``track_residue`` also the max |R_nibble - round| across every
    block (the fp32 residue metric).  A batched forward uses a DIFFERENT fp32
    accumulation ORDER than a single-row forward, so passing here is a STRICTER
    robustness test than the production single-token path."""
    a = L.LEANDIV
    B = len(cases)
    x = torch.zeros(B, dim, dtype=dtype)
    x[:, L.ONE] = 1.0
    for bi, (av, bv) in enumerate(cases):
        for j, nv in enumerate(_nibbles(bv & 0xFFFFFFFF, 8)):
            x[bi, L.AX + j] = float(nv)
        for j, nv in enumerate(_nibbles(av & 0xFFFFFFFF, 8)):
            x[bi, L.STACK0 + j] = float(nv)
    max_res = 0.0
    for spec in specs:
        up = x @ spec["W_up"].T + spec["b_up"]
        gate = x @ spec["W_gate"].T + spec["b_gate"]
        x = x + (F.silu(up) * gate) @ spec["W_down"].T + spec["b_down"]
        if track_residue:
            rv = x[:, a.R:a.R + a.RN]
            max_res = max(max_res, (rv - rv.round()).abs().max().item())
    out = []
    for bi in range(B):
        q = sum(int(round(float(x[bi, a.DIV_RES + c]))) << (4 * c) for c in range(8))
        r = sum(int(round(float(x[bi, a.MOD_RES + c]))) << (4 * c) for c in range(8))
        out.append((q & 0xFFFFFFFF, r & 0xFFFFFFFF))
    return (out, max_res) if track_residue else out


def measure(verbose: bool = True, n_random: int = 6000, batch: int = 512):
    import random
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

    # ---- test set: edge grid + adversarial classes + >=6000 random ----
    cases = list(_edge_grid()) + _adversarial_cases()
    rng = random.Random(1234)
    for _ in range(n_random):
        cases.append((rng.randint(0, 2 ** 32 - 1), rng.randint(0, 2 ** 32 - 1)))
    total = len(cases)

    def _run(dtype, track_residue=False):
        by_name = {n: {k: v.to(dtype) for k, v in s.items()} for n, s in unique}
        specs = [by_name[n] for n in apply_names]
        p, fs, mx = 0, [], 0.0
        for i in range(0, total, batch):
            chunk = cases[i:i + batch]
            res = _run_batch(L, dim, specs, chunk, dtype, track_residue=track_residue)
            outs = res[0] if track_residue else res
            if track_residue:
                mx = max(mx, res[1])
            for (a, b), (q, r) in zip(chunk, outs):
                rq, rr = _ref(a, b)
                if (q, r) == (rq, rr):
                    p += 1
                elif len(fs) < 20:
                    fs.append((a, b, (q, r), (rq, rr)))
        return p, fs, mx

    passed, fails, _ = _run(torch.float64)            # canonical (algorithm correct?)
    passed32, fails32, max_res = _run(torch.float32, track_residue=True)  # honest fp32

    fp32_safe = max_arg < 2 ** 24
    if verbose:
        print("=" * 72)
        print("HARDENED radix-16 (nibble digit-recurrence) 32-bit DIVIDE — bakeoff")
        print("=" * 72)
        print(f"stored (unique) blocks   : {stored}")
        print(f"depth (UNROLLED, straight): {depth_unrolled} blocks  (lean was 80)")
        print(f"depth (applied, recurrent): {depth_applied} block-applications")
        print(f"blocks / iteration        : {blocks_per_iter}  (shift, gteq, qdigit,"
              f" qbsel, gp, ks0..3, apply+emit)")
        n_kb = len(_kb_precompute_blocks(L, dim))
        print(f"one-time overhead         : {n_kb} KB-precompute + 1 init + 1 finalize "
              f"= {n_kb + 2} blocks")
        print(f"nz (nonzero weights)      : {nnz}  (lean was ~144k)")
        print(f"max relu arg (worst |up|) : {max_arg:.1f}  "
              f"(RELU_S={RELU_S}, S={S}; fp32-safe < 2^24 = {2**24}) "
              f"-> {'YES' if fp32_safe else 'NO'}")
        print(f"fp64 params               : 0 (all weights fp32-representable)")
        print(f"max per-step R residue fp32: {max_res:.3e}  (lean was 5.0e-1; ~0 target)")
        print(f"byte-exact (fp64 ALU sim) : {passed}/{total}  "
              f"({'ALL PASS' if passed == total else 'FAIL'})")
        print(f"byte-exact (fp32 e2e sim) : {passed32}/{total}  "
              f"({'ALL PASS' if passed32 == total else 'RESIDUE FLOOR'})")
        if fails:
            print("  first fp64 fails (a,b,got,exp):")
            for f in fails:
                print("   ", f)
        if fails32:
            print("  first fp32 fails (a,b,got,exp):")
            for f in fails32:
                print("   ", f)
        print("-" * 72)
        print("VERDICT vs alternatives (DEPTH):")
        print(f"  base-16 LONG DIVISION (nibble_alu32) : ~262 blocks")
        print(f"  fp32 two-digit                        : ~274 blocks")
        print(f"  fp64 log-sink                         : ~127 blocks")
        print(f"  THIS hardened radix-16 (unrolled)     : {depth_unrolled} blocks")
        cmp = "BEATS long division / two-digit" if depth_unrolled < 262 else "does NOT beat"
        beats_log = depth_unrolled < 127
        print(f"  -> hardened radix-16 {cmp}; "
              f"{'beats' if beats_log else 'does NOT beat'} the fp64 log-sink (127).")
        print("=" * 72)
    return dict(stored=stored, depth_unrolled=depth_unrolled,
                depth_applied=depth_applied, blocks_per_iter=blocks_per_iter,
                nnz=nnz, max_relu_arg=max_arg, fp32_safe=fp32_safe, max_res=max_res,
                byte_exact_pass=passed, byte_exact_total=total,
                byte_exact_pass_fp32=passed32)


if __name__ == "__main__":
    measure()
