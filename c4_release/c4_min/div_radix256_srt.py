"""RADIX-256 SRT (byte-at-a-time) 32-bit DIVIDE — standalone depth bakeoff.

GOAL: get the VARIABLE-divisor DIV *instruction* (arbitrary runtime 32-bit
divisor, the real DIV opcode) under 40 blocks of DEPTH, byte-exact in BOTH fp64
and fp32.  Magic-multiply / divide-by-constant is OFF-TARGET and is NOT used —
this works for any runtime divisor via the SRT digit recurrence.

Baselines (all variable-divisor):
    radix-16 nibble recurrence (div_radix16_lean)   = 80 blocks
    base-16 long division (nibble_alu32)            = ~262 blocks
    fp64 log-sink                                   = 127 blocks

Two variants are built + measured head-to-head (both radix-256, 4 byte iters):

  1. ``table``  — the EXACT-SELECT variant.  Precompute ``KB[k] = k*d`` nibbles
     for k=0..255 ONCE (a big fixed-per-divisor table — O(1) DEPTH via a batched
     parallel-prefix normalise, cost is WEIGHTS not depth).  Per byte-iteration the
     quotient byte is the EXACT ``q = max{k : KB[k] <= R}`` by a nibble-LEXICOGRAPHIC
     compare (every compared quantity <= 15 -> fp32-exact by construction, NO wide
     value-difference dot, NO SRT correction), then ``R -= KB[q]`` by a nibble-lane
     Kogge-Stone borrow.  This is the radix-16 ``div_radix16`` machinery widened
     from 16 to 256 table rows: the select stays ONE block regardless of table
     size, so radix-256 costs the same select DEPTH as radix-16 but runs 4 iters
     not 8.

  2. ``srt``    — the ESTIMATE + CORRECT variant (no 256-row table).  Normalize d,
     estimate the quotient byte ``qhat = floor(Ahat/(Bhat+1))`` (Ahat = top 16 bits
     of the normalized remainder, Bhat = top byte of the normalized divisor) as a
     255-threshold staircase, then fold the SRT correction (``q_true - qhat in
     {0,1,2}``) into a single count-based subtract.  Narrower residual (no big
     table) but a per-iter byte multiply + two borrows.

fp32 discipline
===============
``table``: every compared quantity in the lexicographic select is a nibble diff in
[-15,15]; the borrow diffs are in [-16,15]; the prefix lanes are 0/1 — there is NO
16^p scalar recompose anywhere, so nothing to amplify (the exact residue-free
pattern ``div_radix16_hardened`` proved).  ``srt``: the estimate staircase forms
are bounded by ``255*(Bhat+1) <= 65535`` (``RELU_S*65535 = 1.31e7 < 2^24``).

MEASURE-ONLY: builds the real SwiGLU FFN blocks, runs a CPU forward (fp64 AND
fp32) over the edge grid + adversarial classes + thousands of random 32-bit
pairs, and reports depth / nz / byte-exact + the depth BREAKDOWN.  No full-model
bake.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .nibble_alu32 import (
    _floor_div_pow, _floor_div_pow2, _guard, _step_ge, _ident, _clear,
    _empty_spec, _truncate, RELU_S, S,
)
from .nibble_vm_layout import NibbleVMLayout

MASK32 = 0xFFFFFFFF

# remainder / table nibble widths.  Rn in [0, 256*d) < 2^40 -> 10 nibbles.  KB[k]
# = k*d <= 255*(2^32-1) < 2^40 -> 10 nibbles.
_RN = 10
_NK = 256            # radix-256: quotient byte in 0..255, table rows k=0..255.
_NITERS = 4          # 4 byte digits cover the 32-bit quotient.


def _set_one(L):
    from . import nibble_alu32 as m
    m._ONE = L.ONE


# ===========================================================================
# Layout — the TABLE variant's scratch bands.
# ===========================================================================
class R256Bands:
    def __init__(self, L):
        RN = _RN
        self.RN = RN
        self.R = L._band("R256_R", RN)                 # running remainder nibbles, LSB first
        self.KB = L._band("R256_KB", RN * _NK)         # KB[k]=k*d nibbles, k=0..255
        self.GT = L._band("R256_GT", (_NK - 1) * RN)   # GT[k,i] = [R[i] > KB[k][i]]
        self.EQ = L._band("R256_EQ", (_NK - 1) * RN)   # EQ[k,i] = [R[i] == KB[k][i]]
        self.GE = L._band("R256_GE", _NK)              # GE[k] = [R >= KB[k]] (monotone prefix)
        self.QD = L._scalar("R256_QD")                 # quotient byte q (0..255)
        self.KBQ = L._band("R256_KBQ", RN)             # selected subtrahend KB[q] nibbles
        # Kogge-Stone borrow lanes (double-buffered) for R -= KB[q].
        self.G0 = L._band("R256_G0", RN)
        self.P0 = L._band("R256_P0", RN)
        self.G1 = L._band("R256_G1", RN)
        self.P1 = L._band("R256_P1", RN)
        self.DIV_RES = L._band("R256_DIV", 8)          # quotient nibbles (result)
        self.MOD_RES = L._band("R256_MOD", 8)          # remainder nibbles (result)
        self.BZ = L._scalar("R256_BZ")                 # divisor == 0 predicate
        self.IT = L._scalar("R256_IT")                 # byte-iteration index 0..3
        self.IT_OH = L._band("R256_IT_OH", 4)          # one-hot(IT)


def extend_layout(L):
    if getattr(L, "R256", None) is not None:
        return L.R256
    L.R256 = R256Bands(L)
    while L._off % L.n_heads != 0:
        L._scalar(f"_r256pad{L._off}")
    L.D = L._off
    return L.R256


def _new_layout(code_size: int = 8, n_heads: int = 4):
    L = NibbleVMLayout(code_size, n_heads=n_heads)
    extend_layout(L)
    return L


_NIB_KMAX = 15
# k*d_nib per raw column <= 255*15 = 3825; carry-normalise settles in a few rounds.
# The batched round count that fully settles every KB[k] column stack (verified in
# _kb_precompute; +headroom).  With kmax bounded by the column height we ripple.
_KB_CARRY_ROUNDS = 4


# ===========================================================================
# 1. KB-precompute:  KB[k] = k*d nibbles, k=0..255.  ONE-TIME prologue.
#
# Raw column c of KB[k] = k * d_nib[c] (<= 255*15 = 3825), laid via an _ident
# fan-out (the multiplier k is a fixed weight, so this is a fan-out not a runtime
# multiply — the same trick radix-16's KB uses).  Then base-16 carry-normalise
# rounds settle every column to a clean nibble.  All 256 KB[k] are INDEPENDENT
# bands so they fuse into ONE wide block per round (depth = 1 raw + few rounds,
# regardless of the 256 rows -> the "big table is O(1) depth" property).
# ===========================================================================
def _kb_raw_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, _NK * 8 + _NK * RN)
    u = 0
    for k in range(_NK):
        base = a.KB + RN * k
        for c in range(RN):
            u = _clear(spec, u, base + c)
        if k == 0:
            continue                               # KB[0] = 0
        for c in range(8):                         # d = DN nibbles (8), contribute k * d_nib
            u = _ident(spec, u, {L.AX + c: float(k)}, 0.0, base + c, 1.0)
    return _truncate(spec, u, dim)


def _kb_carry_round_units(spec, u, base, RN):
    for c in range(RN):
        u = _clear(spec, u, base + c)
        u = _ident(spec, u, {base + c: 1.0}, 0.0, base + c, 1.0)                       # keep col
        u = _floor_div_pow(spec, u, {base + c: 1.0}, 0.0, 16, 255, base + c, -16.0)    # mod 16 (col<3840 -> kmax 255)
        if c + 1 < RN:
            u = _floor_div_pow(spec, u, {base + c: 1.0}, 0.0, 16, 255, base + c + 1, 1.0)  # carry
    return u


def _kb_carry_all_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    RN = a.RN
    per_k = RN * (2 + 255 * 2 + 255 * 2)
    spec = _empty_spec(dim, _NK * per_k + 16)
    u = 0
    for k in range(1, _NK):
        u = _kb_carry_round_units(spec, u, a.KB + RN * k, RN)
    return _truncate(spec, u, dim)


def _bz_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    spec = _empty_spec(dim, 2 + 8)
    u = 0
    u = _clear(spec, u, a.BZ)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.BZ, 1.0)
    u = _step_ge(spec, u, {L.AX + c: 1.0 for c in range(8)}, 0.0, 1, a.BZ, -1.0)
    return _truncate(spec, u, dim)


def _kb_precompute_blocks(L, dim):
    blocks = [("r256-kb-raw", _kb_raw_block(L, dim))]
    for rnd in range(_KB_CARRY_ROUNDS):
        blocks.append((f"r256-kb-c{rnd}", _kb_carry_all_block(L, dim)))
    blocks.append(("r256-bz", _bz_block(L, dim)))
    return blocks


# ===========================================================================
# 2. INIT + iteration-index one-hot.
# ===========================================================================
def _it_oh_units(spec, u, a, it_const=0.0):
    for j in range(_NITERS):
        u = _clear(spec, u, a.IT_OH + j)
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j, a.IT_OH + j, 1.0)
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j + 1, a.IT_OH + j, -1.0)
    return u


def _init_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, RN + 8 + 1 + _NITERS * 5)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
    for c in range(8):
        u = _clear(spec, u, a.DIV_RES + c)
    u = _clear(spec, u, a.IT)
    u = _it_oh_units(spec, u, a)
    return _truncate(spec, u, dim)


# ===========================================================================
# 3. ITERATION BODY.
# ===========================================================================
# --- shift  R = 256*R + dividend_byte(it) ----------------------------------
def _shift_block(L, dim) -> Dict[str, torch.Tensor]:
    """R[c] <- R[c-2] for c>=2 (the *256 = shift by two nibbles); R[0],R[1] <-
    the two nibbles of the dividend byte STACK0[2*(3-it) .. +1] via one-hot(IT).
    MSB-first: iteration it consumes byte (3-it), i.e. STACK0 nibbles
    2*(3-it) (low) and 2*(3-it)+1 (high).  Each moved nibble is snapped through a
    sharp kmax=15 staircase so R stays a clean integer nibble."""
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, (RN - 2) * (1 + 15 * 2) + 2 + _NITERS * 2 * 2)
    u = 0
    for c in range(RN - 1, 1, -1):
        u = _clear(spec, u, a.R + c)
        u = _floor_div_pow(spec, u, {a.R + c - 2: 1.0}, 0.0, 1, 15, a.R + c, 1.0)   # snap-copy
    u = _clear(spec, u, a.R + 0)
    u = _clear(spec, u, a.R + 1)
    for j in range(_NITERS):                       # byte index = 3-j (MSB first)
        bidx = 3 - j
        u = _guard(spec, u, [(a.IT_OH + j, 1.0, 0.0)],
                   {L.STACK0 + 2 * bidx: 1.0}, 0.0, a.R + 0, 1.0)       # low nibble
        u = _guard(spec, u, [(a.IT_OH + j, 1.0, 0.0)],
                   {L.STACK0 + 2 * bidx + 1: 1.0}, 0.0, a.R + 1, 1.0)   # high nibble
    return _truncate(spec, u, dim)


# --- gteq  GT[k,i], EQ[k,i] for R vs KB[k] (k=1..255) ----------------------
def _gteq_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, (_NK - 1) * RN * (2 + 4 + 4))
    u = 0
    for k in range(1, _NK):
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


# --- qdigit  QD = sum_k GE[k], GE[k] = [R >= KB[k]] lexicographic -----------
def _qdigit_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, 1 + (_NK - 1) * (RN + 1) + (_NK - 1) + (_NK - 1) * (RN + 1))
    u = 0
    u = _clear(spec, u, a.QD)
    for k in range(1, _NK):
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


# --- qbsel  KBQ = KB[q] subtrahend via one-hot(q) = GE[k]-GE[k+1] -----------
def _qbsel_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, RN + (_NK - 1) * RN)
    u = 0
    for i in range(RN):
        u = _clear(spec, u, a.KBQ + i)
    for k in range(1, _NK):
        w_hi = (a.GE + k, 1.0, 0.0)                       # GE[k] == 1
        if k < _NK - 1:
            w_lo = (a.GE + k + 1, -1.0, 1.0)              # GE[k+1] == 0
            windows = [w_hi, w_lo]
        else:
            windows = [w_hi]                              # GE[256] := 0
        for i in range(RN):
            u = _guard(spec, u, windows, {a.KB + RN * k + i: 1.0}, 0.0, a.KBQ + i, 1.0)
    return _truncate(spec, u, dim)


# --- Kogge-Stone borrow  R = R - KBQ  (nibble-lane, residue-free) -----------
def _gp_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, RN * (2 + 4 + 2 + 4))
    u = 0
    for i in range(RN):
        diff = {a.R + i: 1.0, a.KBQ + i: -1.0}
        u = _clear(spec, u, a.G0 + i)
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.G0 + i, 1.0)          # +1
        u = _step_ge(spec, u, diff, 0.0, 0, a.G0 + i, -1.0)            # -[diff>=0] => [diff<0]
        u = _clear(spec, u, a.P0 + i)
        u = _step_ge(spec, u, diff, 0.0, 0, a.P0 + i, 1.0)            # +[diff>=0]
        u = _step_ge(spec, u, diff, 0.0, 1, a.P0 + i, -1.0)          # -[diff>=1] => [diff==0]
    return _truncate(spec, u, dim)


def _ks_stage_block(L, dim, g_src, p_src, g_dst, p_dst, d) -> Dict[str, torch.Tensor]:
    a = L.R256
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


def _borrow_apply_block(L, dim, g_final) -> Dict[str, torch.Tensor]:
    """R[i] <- diff_i - Bin_i + 16*Bout_i (Bin_i = Bout_{i-1}); + emit q's two
    nibbles into DIV_RES[2*(3-it) .. +1] gated on IT_OH; then IT+=1 + refresh
    IT_OH.  The quotient byte q (=QD, 0..255) is split into (q mod 16, floor(q/16))
    with sharp staircases READING QD (the block input); the two nibble values are
    then gate-written into the active slot.  Because the mod/floor of QD is a plain
    linear-ish staircase and the slot is selected by IT_OH, we materialise the two
    nibble VALUES as gated deltas: for the active j, DIV_RES[lo] += [gate]*(q%16),
    DIV_RES[hi] += [gate]*floor(q/16).  A gated staircase is realised as a sum of
    gated step indicators (each _guard'd), so no wide product is needed."""
    a = L.R256
    RN = a.RN
    spec = _empty_spec(dim, RN * 5 + _NITERS * (2 + 15 * 3 + 15 * 3) + 2 + _NITERS * 5)
    u = 0
    for i in range(RN):
        diff = {a.R + i: 1.0, a.KBQ + i: -1.0}
        form = dict(diff)
        if i >= 1:
            form[g_final + i - 1] = form.get(g_final + i - 1, 0.0) - 1.0
        u = _clear(spec, u, a.R + i)
        u = _ident(spec, u, form, 0.0, a.R + i, 1.0)
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.R + i, 16.0)
        u = _step_ge(spec, u, form, 0.0, 0, a.R + i, -16.0)          # +16*[form<0]
    # emit q (byte) into the active DIV_RES slot (two nibbles), gated on IT_OH[j].
    #   q_lo = q mod 16   = q - 16*floor(q/16)  = sum_{m=1..255}[q>=m] - 16*sum_{m=1..15}[q>=16m]
    #   q_hi = floor(q/16) = sum_{m=1..15}[q>=16m]
    # Each step indicator [q>=t] is a _guard over {IT_OH[j], [q>=t]} AND — but a
    # _guard's AND needs 0/1 windows and [q>=t] is exactly such.  We instead gate a
    # PRE-COMPUTED nibble delta: emit q_lo/q_hi as gated STAIRCASE sums.  Simpler:
    # write the whole byte q into lo (gated), 0 into hi (gated); a downstream
    # split block re-nibbles lo.  We take that split route (see _emit_split_block).
    for j in range(_NITERS):
        bidx = 3 - j
        lo = a.DIV_RES + 2 * bidx
        g = (a.IT_OH + j, 1.0, 0.0)
        u = _guard(spec, u, [g], {lo: -1.0}, 0.0, lo, 1.0)          # clear slot (gated)
        u = _guard(spec, u, [g], {a.QD: 1.0}, 0.0, lo, 1.0)         # + q (the raw byte)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.IT, 1.0)               # IT += 1
    u = _it_oh_units(spec, u, a, it_const=1.0)
    return _truncate(spec, u, dim)


def _emit_split_block(L, dim) -> Dict[str, torch.Tensor]:
    """Split the raw quotient byte just written into each DIV_RES low-nibble slot
    into (q mod 16, floor(q/16)).  The apply block wrote the full byte q (0..255)
    into DIV_RES[2*bidx] and left DIV_RES[2*bidx+1] = 0 for the ACTIVE slot; here we
    move floor(byte/16) into the high nibble and reduce the low nibble mod 16.
    Applied to ALL 4 slots every iter: a settled slot (both nibbles already 0..15,
    lo<16) has floor(lo/16)=0 so hi is unchanged and lo mod 16 = lo — IDEMPOTENT.
    Only the freshly-written active slot (lo up to 255) actually splits."""
    a = L.R256
    spec = _empty_spec(dim, _NITERS * (1 + 15 * 2 + 2 + 15 * 2))
    u = 0
    for j in range(_NITERS):
        bidx = 3 - j
        lo = a.DIV_RES + 2 * bidx
        hi = a.DIV_RES + 2 * bidx + 1
        # hi += floor(lo/16)  (lo up to 255 -> kmax 15)
        u = _floor_div_pow(spec, u, {lo: 1.0}, 0.0, 16, 15, hi, 1.0)
        # lo := lo mod 16 = lo - 16*floor(lo/16)
        u = _floor_div_pow(spec, u, {lo: 1.0}, 0.0, 16, 15, lo, -16.0)
    return _truncate(spec, u, dim)


def _borrow_prefix_blocks(L, dim):
    a = L.R256
    RN = a.RN
    blocks = [("r256-gp", _gp_block(L, dim))]
    (gs, ps), (gd, pd) = (a.G0, a.P0), (a.G1, a.P1)
    d = 1
    st = 0
    while d < RN:
        blocks.append((f"r256-ks{st}", _ks_stage_block(L, dim, gs, ps, gd, pd, d)))
        (gs, ps), (gd, pd) = (gd, pd), (gs, ps)
        d *= 2
        st += 1
    blocks.append(("r256-apply", _borrow_apply_block(L, dim, gs)))
    return blocks, gs


def _iteration_body(L, dim):
    borrow, _ = _borrow_prefix_blocks(L, dim)
    return [
        ("r256-shift", _shift_block(L, dim)),
        ("r256-gteq", _gteq_block(L, dim)),
        ("r256-qd", _qdigit_block(L, dim)),
        ("r256-qbsel", _qbsel_block(L, dim)),
    ] + borrow + [("r256-emitsplit", _emit_split_block(L, dim))]


# ===========================================================================
# 4. FINALIZE + emit-byte split.
# ===========================================================================
def _finalize_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256
    spec = _empty_spec(dim, 8 * 6)
    u = 0
    for c in range(8):
        u = _clear(spec, u, a.MOD_RES + c)
        u = _ident(spec, u, {a.R + c: 1.0}, 0.0, a.MOD_RES + c, 1.0)
        gz = (a.BZ, 1.0, 0.0)
        u = _guard(spec, u, [gz], {a.R + c: -1.0}, 0.0, a.MOD_RES + c, 1.0)
        u = _guard(spec, u, [gz], {a.DIV_RES + c: -1.0}, 0.0, a.DIV_RES + c, 1.0)
    return _truncate(spec, u, dim)


# ===========================================================================
# 5. Public builders.
# ===========================================================================
def compile_blocks_unrolled(L, dim, n_iters: int = _NITERS):
    _set_one(L)
    blocks = []
    blocks += _kb_precompute_blocks(L, dim)
    blocks.append(("r256-init", _init_block(L, dim)))
    body = _iteration_body(L, dim)              # includes the emit-split
    for it in range(n_iters):
        for name, spec in body:
            blocks.append((f"{name}{it}", spec))
    blocks.append(("r256-finalize", _finalize_block(L, dim)))
    return blocks


def compile_blocks_recurrent(L, dim, n_iters: int = _NITERS):
    _set_one(L)
    prefix = _kb_precompute_blocks(L, dim) + [("r256-init", _init_block(L, dim))]
    body = _iteration_body(L, dim)              # includes the emit-split
    finalize = [("r256-finalize", _finalize_block(L, dim))]
    unique = prefix + body + finalize
    apply_names = [n for n, _ in prefix]
    for _ in range(n_iters):
        apply_names += [n for n, _ in body]
    apply_names.append("r256-finalize")
    return unique, apply_names


# ===========================================================================
# 6. CPU forward SIMULATION.
# ===========================================================================
def _apply_block(x, spec):
    up = x @ spec["W_up"].T + spec["b_up"]
    gate = x @ spec["W_gate"].T + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + hidden @ spec["W_down"].T + spec["b_down"]


def _nibbles(v, n):
    return [(v >> (4 * j)) & 0xF for j in range(n)]


def simulate(a_val, b_val, unique=None, apply_names=None, L=None, dim=None,
             dtype=torch.float64):
    if L is None:
        L = _new_layout(); dim = L.D
    if unique is None:
        unique, apply_names = compile_blocks_recurrent(L, dim)
    by_name = {n: s for n, s in unique}
    x = torch.zeros(dim, dtype=dtype)
    x[L.ONE] = 1.0
    for j, nv in enumerate(_nibbles(b_val & MASK32, 8)):
        x[L.AX + j] = float(nv)
    for j, nv in enumerate(_nibbles(a_val & MASK32, 8)):
        x[L.STACK0 + j] = float(nv)
    for name in apply_names:
        spec = {k: v.to(dtype) for k, v in by_name[name].items()}
        x = _apply_block(x, spec)
    a = L.R256
    q = sum(int(round(float(x[a.DIV_RES + c]))) << (4 * c) for c in range(8))
    r = sum(int(round(float(x[a.MOD_RES + c]))) << (4 * c) for c in range(8))
    return q & MASK32, r & MASK32


def _ref(a, b):
    a &= MASK32; b &= MASK32
    if b == 0:
        return 0, 0
    return a // b, a % b


def _edge_grid():
    edges = [0, 1, 2, 2 ** 31, 2 ** 32 - 1]
    edges += [1 << p for p in range(0, 32)]
    bs = [1, 2, 3, 7, 10, 16, 255, 256, 65535, 65536, 2 ** 31, 2 ** 32 - 1]
    cases = []
    for a in edges:
        for b in bs:
            cases.append((a, b))
    for v in edges:
        cases.append((v, v))
        cases.append((v, 0))
        cases.append((max(0, v - 1), v))
    return cases


def _spec_nnz(spec):
    n = 0
    for key in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
        n += int((spec[key] != 0).sum())
    return n


if __name__ == "__main__":
    from .div_radix256_measure import measure
    measure()
