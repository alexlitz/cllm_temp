"""BATCHED CARRY-LOOKAHEAD (KOGGE-STONE) KB-PRECOMPUTE — shrink the c4_min general
DIVIDE's KB-precompute depth by replacing the ``15 x _KB_CARRY_ROUNDS = 90`` SERIAL
ripple carry rounds with ONE batched base-16 Kogge-Stone parallel-prefix resolve that
normalises all 15 ``KB[k]=k*b`` at once.  Standalone byte-exact bakeoff, COMPOSED from
the proven ``nibble_alu32`` SwiGLU primitives (this file never edits them), and PORTING
the Kogge-Stone ``round1/G-P/KS-stage/apply`` logic from ``mul_lookahead.py``.

THE BASELINE (what we're beating)
=================================
``nibble_alu32._kb_precompute_blocks`` builds ``KB[k] = k*b`` for ``k=1..15`` from the
AX (=B) nibble bands.  Each ``KB[k]`` is a per-nibble multiply ``b_nib[c]*k <= 15*15 =
225`` laid into ``RN=9`` raw columns (``< 256``), then carry-normalised **in its own
tiny block** with ``_KB_CARRY_ROUNDS = 6`` ripple rounds:

  raw (1) | [ KB[k] : 6 ripple rounds ] x 15 = 90 | BZ (1)   = 92 blocks

The 15 ``KB[k]`` are INDEPENDENT numbers (nothing carries BETWEEN them), so their 90
carry-resolves are 15 SEPARATE 9-column ripples that need not run serially — they can
**batch into ONE parallel-prefix pass**.  The ripple DOMINATES the precompute depth
(90 of 92 blocks); the raw + BZ are only 2.

THE LEVER — batched base-16 carry-lookahead (Kogge-Stone), 15 bands at once
===========================================================================
Each ``KB[k]``'s 9 post-raw columns are a base-16 carry-save number (``< 256``): the
resolve is a carry-PROPAGATE add over the columns.  A parallel-prefix (Kogge-Stone)
resolve computes every column's carry in ``ceil(log2 9) = 4`` prefix stages instead of
6 serial ripples.  The SAME clean-binary-CLA trick as ``mul_lookahead``:

  ROUND-1 (one carry round) reduces each column from ``< 256`` to
  ``t_c = (col_c mod 16) + floor(col_{c-1}/16)`` in ``[0, 30]`` (digit 0..15 plus a
  single-nibble carry 0..15 from below).  Given an INCOMING BINARY carry
  ``b_c in {0,1}``, the column emits final digit ``(t_c+b_c) mod 16`` and a BINARY
  carry-out ``[t_c+b_c >= 16] in {0,1}`` (because ``t_c+b_c <= 31``).  So the carries
  are pure binary and obey the textbook CLA recurrence per band:

      generate   G_c = [t_c >= 16]      propagate  P_c = [t_c == 15]
      b_{c+1} = G_c OR (P_c AND b_c),    b_0 = 0.

  Kogge-Stone combines the pairs ``(G,P)`` with the associative operator
  ``(g_hi,p_hi) o (g_lo,p_lo) = (g_hi OR (p_hi AND g_lo), p_hi AND p_lo)`` in
  ``ceil(log2 9) = 4`` stages.  After the stages ``Gband[c]`` is the carry INTO column
  ``c+1``, so the carry into column ``c`` is ``b_c = Gband[c-1]`` (``b_0 = 0``).  The
  APPLY block writes the result nibble ``digit_c = t_c + Gband[c-1] - 16*Gband[c]``
  directly into ``KB[k]`` — no separate result-copy block.

THE BATCHING — the whole point vs mul_lookahead
===============================================
``mul_lookahead`` runs the prefix over ONE 8-column value.  Here the 15 ``KB[k]`` are
INDEPENDENT, so we lay their ``t/G/P`` lanes side-by-side (``15 x RN`` per lane band)
and every prefix block loops over all ``15 x RN`` positions in ONE block — a carry
NEVER crosses a KB[k] boundary (the prefix distance ``d`` stays inside each band's
9-column window, i.e. the combine at column ``c`` in band ``k`` only reaches
``c-d >= 0`` in the SAME band).  So one Kogge-Stone stage resolves ALL 15 bands
simultaneously.

DEPTH
=====
  raw (1) | round-1 (1) | G/P (1) | KS x 4 | fused apply/result (1) | BZ (1)
  = 9 blocks (vs 92), 4 prefix stages (vs 15 x 6 = 90 ripple rounds).

fp32 DISCIPLINE
===============
Every relu argument is trivially fp32-exact: raw columns ``<= 225 < 256``, round-1
outputs ``t_c <= 30``, and the entire prefix operates on {0,1} generate/propagate lanes
(weighted forms ``<= 4``).  The tightest ``RELU_S*arg`` is the round-1 floor staircase
(``RELU_S*(15*16) = 48000 << 2^24``); the harness reports the measured max.  0 fp64.

VERIFY (byte-exact, real SwiGLU forward, sparse sim — seconds, no dense DIM matmul):
  all 15 ``KB[k] == k*b`` (over the low 8 nibbles = ``(k*b) & 0xFFFFFFFF``) over the
  shared edge set + >= 1000 random ``b < 2^32``, INCLUDING ``0xFFFFFFFF`` (max carry
  propagation with ``k=15`` — the whole point of the carry path).

The div KNOCK-ON: this batched KB-precompute is what the recurrent / radix-16 divide
runs ONCE per divide before its 8 digit iterations; shaving 90 ripple blocks to 4
prefix stages cuts the divide's fixed prologue depth directly.
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
    _ident, _clear, _step_ge, _truncate, _floor_div_pow, _floor_div_pow2,
    _nibble_carry_round, _kb_precompute_blocks, extend_layout_for_alu32,
    _NIB_KMAX, _KB_CARRY_ROUNDS,
)

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

FP32_INT_MAX = 1 << 24                       # 2^24: fp32 exact-integer ceiling
NK = 15                                      # KB[k] for k = 1..15 (15 independent values)

# every gadget records (label, max_arg) so the harness reports the single tightest
# RELU_S*arg across the whole gadget (fp32 headroom).  RELU_S is the library 200.
_ARG_LOG: List[Tuple[str, int]] = []


def _reset_arglog():
    _ARG_LOG.clear()


def _log_arg(label: str, max_arg: int):
    """Record + assert the fp32 discipline for one gadget: RELU_S*max_arg < 2^24."""
    prod = RELU_S * max_arg
    assert prod < FP32_INT_MAX, (
        f"{label}: RELU_S({RELU_S})*max_arg({max_arg})={prod:.0f} >= 2^24 -> fp32-LOSSY")
    _ARG_LOG.append((label, max_arg))


def tightest_arg() -> dict:
    if not _ARG_LOG:
        return {"max_arg": 0, "product": 0.0, "ratio": 0.0, "label": None}
    label, max_arg = max(_ARG_LOG, key=lambda t: t[1])
    prod = RELU_S * max_arg
    return {"max_arg": max_arg, "product": prod, "ratio": prod / FP32_INT_MAX,
            "label": label}


# ===========================================================================
# Private scratch bands for the batched prefix — 15 KB[k] side-by-side, RN
# columns each.  Own band range so the sparse sim reads/writes only this
# design's bands and never collides with production KB scratch.
# ===========================================================================
def _scratch(L, name, size):
    key = f"DKB_{name}"
    if key in L._names:
        return L._names[key][0]
    return L._band(key, size)


def _n_prefix_stages(rn: int) -> int:
    n = 0
    d = 1
    while d < rn:
        n += 1
        d *= 2
    return n                                  # ceil(log2 rn)


# ===========================================================================
# THE BATCHED KOGGE-STONE KB RESOLVE.  Every block loops over all 15 KB[k]
# bands (RN columns each); the prefix distance stays INSIDE each band, so one
# block resolves all 15 independent numbers at the same stage simultaneously.
# ===========================================================================
def _kb_raw_block(L, dim, RN) -> Block:
    """Materialise the 15 raw KB[k] column stacks ``KB_raw[k][c] = k * b_nib[c]``
    (<= 225, only c<8 for b) into the production ``a.KB`` band, exactly like the
    baseline's raw block.  (SET each column.)"""
    a = L.ALU32
    _log_arg("kb-raw", 0)                       # pure linear identity terms, no relu
    spec = _empty_spec(dim, NK * RN + NK * 8)
    u = 0
    for k in range(1, 16):
        base = a.KB + RN * k
        for c in range(RN):
            u = _clear(spec, u, base + c)
        for c in range(8):                     # b has 8 nibbles
            u = _ident(spec, u, {L.AX + c: float(k)}, 0.0, base + c, 1.0)
    return _truncate(spec, u, dim)


def _round1_block(L, dim, RN, kb_base, t_band) -> Block:
    """ONE base-16 carry round over ALL 15 KB[k] at once, reducing each band's raw
    columns (``< 256``) to ``t_c = (col_c mod 16) + floor(col_{c-1}/16)`` in ``[0,30]``.
    Runs ``_nibble_carry_round`` independently per band (a carry stays inside a band's
    RN window).  After this round the carries are pure binary -> clean CLA prefix."""
    a = L.ALU32
    _log_arg("round1-floor", 15 * 16)          # floor(col/16) staircase thresholds
    # per band per column: clear + col + floor2 (2*15 relu) [+top: floor1]
    spec = _empty_spec(dim, NK * RN * (2 + _NIB_KMAX * 2 + 2))
    u = 0
    for k in range(1, 16):
        src = a.KB + RN * k
        dst = t_band + RN * (k - 1)
        u = _nibble_carry_round(spec, u, src, dst, RN)   # independent per band
    return _truncate(spec, u, dim)


def _gp_block(L, dim, RN, t_band, g_band, p_band) -> Block:
    """Generate / propagate lanes for ALL 15 bands from ``t_c`` in ``[0,30]`` (SET each):
        G_c = [t_c >= 16]        P_c = [t_c == 15] = [t_c >= 15] - [t_c >= 16].
    Both 0/1.  Thresholds 15/16 -> RELU_S*16 tiny, deep fp32 headroom."""
    _log_arg("gp-thr", 16)
    spec = _empty_spec(dim, NK * RN * (2 + 2 + 2 + 2))
    u = 0
    for kk in range(NK):
        for c in range(RN):
            tc = {t_band + kk * RN + c: 1.0}
            g = g_band + kk * RN + c
            p = p_band + kk * RN + c
            u = _clear(spec, u, g)
            u = _step_ge(spec, u, tc, 0.0, 16, g, 1.0)              # G = [t>=16]
            u = _clear(spec, u, p)
            u = _step_ge(spec, u, tc, 0.0, 15, p, 1.0)              # +[t>=15]
            u = _step_ge(spec, u, tc, 0.0, 16, p, -1.0)            # -[t>=16] => [t==15]
    return _truncate(spec, u, dim)


def _ks_stage_block(L, dim, RN, g_src, p_src, g_dst, p_dst, d) -> Block:
    """ONE Kogge-Stone prefix combine stage at distance ``d`` over ALL 15 bands (SET
    g_dst/p_dst).  Per band, for a column ``c`` (index INSIDE the band's RN window):
        c >= d:  (G_c,P_c) := (G_c OR (P_c AND G_{c-d}),  P_c AND P_{c-d})
        c <  d:  (G_c,P_c) carried through.
    The combine at ``c`` only reaches ``c-d`` in the SAME band (never crosses a KB[k]
    boundary), so the 15 bands resolve independently in one block.  For 0/1 lanes:
        G_dst = [2*G_c + P_c + G_{c-d} >= 2]   (single-level exact OR-of-AND)
        P_dst = [P_c + P_{c-d} >= 2]           (AND).
    Reads the block INPUT (previous stage's g_src/p_src) -> whole stage is one block."""
    _log_arg("ks-thr", 4)                      # weighted form <= 2*1+1+1 = 4
    spec = _empty_spec(dim, NK * RN * (2 + 2 + 2 + 2))
    u = 0
    for kk in range(NK):
        gs = g_src + kk * RN
        ps = p_src + kk * RN
        gd = g_dst + kk * RN
        pd = p_dst + kk * RN
        for c in range(RN):
            if c >= d:
                gform = {gs + c: 2.0, ps + c: 1.0, gs + c - d: 1.0}
                u = _clear(spec, u, gd + c)
                u = _step_ge(spec, u, gform, 0.0, 2, gd + c, 1.0)
                pform = {ps + c: 1.0, ps + c - d: 1.0}
                u = _clear(spec, u, pd + c)
                u = _step_ge(spec, u, pform, 0.0, 2, pd + c, 1.0)
            else:
                u = _clear(spec, u, gd + c)
                u = _ident(spec, u, {gs + c: 1.0}, 0.0, gd + c, 1.0)
                u = _clear(spec, u, pd + c)
                u = _ident(spec, u, {ps + c: 1.0}, 0.0, pd + c, 1.0)
    return _truncate(spec, u, dim)


def _apply_block(L, dim, RN, t_band, g_final, kb_base) -> Block:
    """Final digits into ``a.KB`` for ALL 15 bands (SET), fused carry-apply (no copy):
        b_c = G_final[c-1]  (carry INTO column c; b_0 = 0)
        digit_c = t_c + G_final[c-1] - 16*G_final[c].
    Each band's top carry-out ``G_final[RN-1]`` overflows past the kept nibbles and is
    dropped (KB[k] & 0xFFFFFFFF-worth of nibbles is what the divide reads).  All read
    the block INPUT (t_c + the FINAL prefix G lane) -> one block."""
    a = L.ALU32
    _log_arg("apply", 30)                      # t_c + b_c <= 31; coeff-16 form small
    spec = _empty_spec(dim, NK * RN * 4)
    u = 0
    for kk in range(NK):
        tb = t_band + kk * RN
        gf = g_final + kk * RN
        res = a.KB + RN * (kk + 1)             # KB[k], k = kk+1
        for c in range(RN):
            u = _clear(spec, u, res + c)
            u = _ident(spec, u, {tb + c: 1.0}, 0.0, res + c, 1.0)          # + t_c
            if c >= 1:
                u = _ident(spec, u, {gf + c - 1: 1.0}, 0.0, res + c, 1.0)  # + b_c
            u = _ident(spec, u, {gf + c: 1.0}, 0.0, res + c, -16.0)        # - 16*b_{c+1}
    return _truncate(spec, u, dim)


def _bz_block(L, dim) -> Block:
    """BZ = [b == 0] = 1 - [sum of b nibbles >= 1].  (Same as the baseline BZ block —
    the divisor-zero predicate the divide's finalize reads.)"""
    a = L.ALU32
    _log_arg("bz", 8)                          # sum of 8 nibbles <= 8*15, thr=1 tiny
    spec = _empty_spec(dim, 2 + 8 * 4)
    u = 0
    u = _clear(spec, u, a.BZ)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.BZ, 1.0)                     # + 1
    bsum = {L.AX + c: 1.0 for c in range(8)}
    u = _step_ge(spec, u, bsum, 0.0, 1, a.BZ, -1.0)                       # - [sum>=1]
    return _truncate(spec, u, dim)


def build_batched_prefix(L, dim) -> Tuple[List[Block], int, dict]:
    """Batched Kogge-Stone KB-precompute: raw | round-1 | G/P | 4 KS prefix stages |
    fused apply/result | BZ.  All 15 KB[k] resolve in the SAME blocks."""
    A._ONE = L.ONE
    a = L.ALU32
    RN = a.RN
    n_stages = _n_prefix_stages(RN)            # ceil(log2 9) = 4

    T = _scratch(L, "T", NK * RN)              # t_c in [0,30] (post round-1), 15 bands
    G0 = _scratch(L, "G0", NK * RN); P0 = _scratch(L, "P0", NK * RN)
    G1 = _scratch(L, "G1", NK * RN); P1 = _scratch(L, "P1", NK * RN)

    blocks: List[Block] = []
    blocks.append(("dkb-raw", _kb_raw_block(L, dim, RN)))
    blocks.append(("dkb-round1", _round1_block(L, dim, RN, a.KB, T)))
    blocks.append(("dkb-gp", _gp_block(L, dim, RN, T, G0, P0)))
    (gs, ps), (gd, pd) = (G0, P0), (G1, P1)
    d = 1
    st = 0
    while d < RN:
        blocks.append((f"dkb-ks{st}-d{d}", _ks_stage_block(L, dim, RN, gs, ps, gd, pd, d)))
        (gs, ps), (gd, pd) = (gd, pd), (gs, ps)   # swap buffers
        d *= 2
        st += 1
    blocks.append(("dkb-apply", _apply_block(L, dim, RN, T, gs, a.KB)))
    blocks.append(("dkb-bz", _bz_block(L, dim)))

    info = {
        "note": "batched Kogge-Stone base-16 KB-precompute: raw | round-1 | G/P | "
                f"{n_stages}x prefix combine (all 15 KB[k] at once) | fused apply/result | BZ",
        "prefix_stages": n_stages,
        "ripple_rounds_replaced": NK * _KB_CARRY_ROUNDS,
        "n_kb_values": NK, "columns_RN": RN,
        "max_staircase_arg": 15 * 16,
    }
    return blocks, a.KB, info


# ===========================================================================
# BASELINE contender — wrap the production _kb_precompute_blocks (15 x 6 ripple).
# ===========================================================================
def build_baseline(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    a = L.ALU32
    blocks = list(_kb_precompute_blocks(L, dim))
    _log_arg("baseline", 15 * 16)              # kb raw <= 225; round floor thr <= 240
    info = {
        "note": "production nibble KB-precompute (_kb_precompute_blocks): raw | "
                f"{NK} x {_KB_CARRY_ROUNDS} ripple carry rounds (per-KB[k]) | BZ",
        "ripple_rounds": NK * _KB_CARRY_ROUNDS,
        "n_kb_values": NK, "columns_RN": a.RN,
        "max_staircase_arg": 15 * 16,
    }
    return blocks, a.KB, info


# ===========================================================================
# MEASUREMENT + SPARSE byte-exact sim (no dense DIM forward).
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
    """Apply ONE SwiGLU block's ``down(silu(up)*gate)`` over just the units that read a
    currently-nonzero band — the EXACT dense math, no DIM x DIM matmul."""
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


def simulate_sparse(L, blocks: List[Block], kb_band: int, b_val: int,
                    dtype=torch.float32) -> List[int]:
    """Run the KB-precompute blocks for a divisor ``b`` and read back all 15 KB[k]
    (each the low 8 nibbles = ``(k*b) & 0xFFFFFFFF``).  b is loaded into the AX
    nibble band (=B); STACK0 is irrelevant to the precompute."""
    a = L.ALU32
    RN = a.RN
    state: Dict[int, float] = {L.ONE: 1.0}
    for c in range(8):
        state[L.AX + c] = float((b_val >> (4 * c)) & 0xF)
    for _n, s in blocks:
        state = _sparse_apply(state, s, dtype=dtype)
    out: List[int] = []
    for k in range(1, 16):
        base = kb_band + RN * k
        v = 0
        for c in range(8):                     # low 8 nibbles = 32-bit result
            v |= (int(round(state.get(base + c, 0.0))) & 0xF) << (4 * c)
        out.append(v & 0xFFFFFFFF)
    return out


def test_divisors(n_random: int = 1000, seed: int = 0) -> Tuple[List[int], int]:
    """Edge set (0, 1, 2^32-1, powers of two, single-byte / byte-lane patterns) +
    >= n_random random b < 2^32.  0xFFFFFFFF is ALWAYS included (k=15 -> max carry
    propagation, the whole point of the carry path)."""
    rng = random.Random(seed)
    M = 0xFFFFFFFF
    edges = [0, 1, 2, 3, 1 << 31, M, M - 1, 0x80000000, 0x7FFFFFFF,
             0xFFFF, 0x10000, 0xDEADBEEF, 0xCAFEBABE, 0xABCD, 0x1234,
             65535, 65536, 255, 256, 0x00FF00FF, 0xFF00FF00, 0x11111111,
             0x0FFFFFFF, 0x1FFFFFFF, 0x08888888]
    pows = [1 << k for k in range(0, 32)]              # every power of two 2^0..2^31
    singles = [0x000000AB, 0x0000CD00, 0x00EF0000, 0x12000000]
    vals: List[int] = []
    for b in edges + pows + singles:
        vals.append(b & M)
    n_struct = len(vals)
    for _ in range(n_random):
        vals.append(rng.randint(0, M))
    return vals, n_struct


DIM = 4096              # wide enough for the private prefix scratch bands.


def _make_layout():
    L = NibbleVMLayout(8, n_heads=4)
    # mul_lookahead=False so the layout's OWN dim is the pre-lookahead one; this
    # bakeoff builds its resolve on private DKB_* scratch bands regardless.
    extend_layout_for_alu32(L, mul_lookahead=False)
    A._ONE = L.ONE
    return L


# ===========================================================================
# DIV KNOCK-ON — the KB-precompute is the divide's fixed prologue.
# ===========================================================================
def div_knockon() -> dict:
    """The base-16 long division (``compile_divmod_blocks`` /
    ``compile_divmod_blocks_recurrent``) runs the KB-precompute ONCE per divide before
    its 8 digit iterations (``KB[k]=k*b`` is the per-k threshold table the quotient
    digit compares against).  The baseline precompute is
    ``15 x _KB_CARRY_ROUNDS = 90`` ripple carry blocks (+ raw + BZ = 92); the batched
    Kogge-Stone resolve is ``ceil(log2 RN)=4`` prefix stages + round-1 + G/P + apply =
    7 resolve blocks (+ raw + BZ = 9).  So the divide's fixed KB prologue drops from 92
    to 9 blocks — a straight depth saving of ``92 - 9 = 83`` blocks off EVERY divide,
    independent of the 8-iteration inner loop (whose per-iteration QB normalise is a
    separate, already-cheap 9-column ripple)."""
    RN = 9
    kb_ripple_rounds = NK * _KB_CARRY_ROUNDS          # 90
    ks_stages = _n_prefix_stages(RN)                   # 4
    kb_ks_resolve = 1 + 1 + ks_stages + 1              # round1 + gp + stages + apply = 7
    baseline_precompute = kb_ripple_rounds + 2         # + raw + BZ = 92
    batched_precompute = kb_ks_resolve + 2             # + raw + BZ = 9
    return {
        "kb_columns_RN": RN,
        "kb_values": NK,
        "kb_ripple_rounds_baseline": kb_ripple_rounds,
        "kb_ks_prefix_stages": ks_stages,
        "kb_ks_resolve_blocks": kb_ks_resolve,
        "baseline_precompute_blocks": baseline_precompute,
        "batched_precompute_blocks": batched_precompute,
        "precompute_blocks_saved": baseline_precompute - batched_precompute,
        "verdict": ("The KB-precompute is the divide's fixed prologue (runs ONCE, "
                    "before the 8 digit iterations).  Batching the 15 independent "
                    f"KB[k] carry-resolves into ONE {ks_stages}-stage Kogge-Stone pass "
                    f"cuts the prologue {baseline_precompute} -> {batched_precompute} "
                    f"blocks (saves {baseline_precompute - batched_precompute}), off "
                    "every divide.  This is the real batched-prefix win the QB "
                    "per-iteration normalise (9 cols, already 6 ripple rounds) is not."),
    }


# ===========================================================================
# BAKEOFF DRIVER
# ===========================================================================
def run_bakeoff(n_random: int = 1000, verbose: bool = True) -> dict:
    results: dict = {}
    L = _make_layout()
    dim = DIM
    M = 0xFFFFFFFF
    divisors, n_struct = test_divisors(n_random=n_random)

    builders: List[Tuple[str, callable]] = [
        ("baseline_ripple", build_baseline),
        ("batched_prefix", build_batched_prefix),
    ]

    for name, fn in builders:
        _reset_arglog()
        blocks, kb_band, info = fn(L, dim)
        assert L.D <= dim, f"layout grew past DIM ({L.D} > {dim}); raise DIM"
        head = tightest_arg()
        m = measure(blocks)
        # byte-exact: for each divisor b, all 15 KB[k] must equal (k*b)&M.
        n_pass = 0
        n_total = 0
        ffff_ok = None
        first_fail = None
        for b in divisors:
            got = simulate_sparse(L, blocks, kb_band, b, dtype=torch.float32)
            all_ok = True
            for k in range(1, 16):
                exp = (k * b) & M
                n_total += 1
                if got[k - 1] == exp:
                    n_pass += 1
                else:
                    all_ok = False
                    if first_fail is None:
                        first_fail = (hex(b), k, hex(got[k - 1]), hex(exp))
            if b == M:
                ffff_ok = all_ok
        results[name] = {
            **m,
            "byte_exact_pass": n_pass, "byte_exact_total": n_total,
            "n_divisors": len(divisors), "n_structured": n_struct,
            "n_random": len(divisors) - n_struct,
            "ffffffff_all_k_ok": ffff_ok,
            "first_fail": first_fail,
            "tightest_relu_s_arg": head["product"],
            "tightest_ratio_of_2p24": head["ratio"],
            "tightest_max_arg": head["max_arg"],
            "tightest_gadget": head["label"],
            **{k: v for k, v in info.items()},
        }
        if verbose:
            print(f"{name:20s} depth={m['depth']:3d} nz={m['weights_nz']:7d} "
                  f"exact={n_pass}/{n_total} 0xFFFFFFFF*k={ffff_ok} "
                  f"maxarg*RELU_S={head['product']:.0f} ({100*head['ratio']:.4f}% of 2^24) "
                  f"stages={info.get('prefix_stages', info.get('ripple_rounds', '-'))} "
                  f"fail={first_fail}")
    results["_div_knockon"] = div_knockon()
    return results


def _print_table(results: dict):
    order = ["baseline_ripple", "batched_prefix"]
    print()
    print("VARIANT              DEPTH   NZ       PREFIX/ROUNDS   TIGHT RELU_S*arg (%2^24)  0xFFFFFFFF*k   BYTE-EXACT")
    print("-" * 106)
    for name in order:
        r = results[name]
        stages = r.get("prefix_stages", r.get("ripple_rounds", "-"))
        print(f"{name:20s} {r['depth']:5d}  {r['weights_nz']:7d}   {str(stages):>12s}   "
              f"{r['tightest_relu_s_arg']:9.0f} ({100*r['tightest_ratio_of_2p24']:6.4f}%)     "
              f"{str(r['ffffffff_all_k_ok']):>5s}       "
              f"{r['byte_exact_pass']:6d}/{r['byte_exact_total']:<6d}")
    b = results["baseline_ripple"]; g = results["batched_prefix"]
    print()
    print(f"KB PRECOMPUTE DEPTH: {b['depth']} (ripple, {NK}x{_KB_CARRY_ROUNDS}) -> "
          f"{g['depth']} (batched prefix, {g['prefix_stages']} stages) "
          f"= -{b['depth'] - g['depth']} blocks; "
          f"nz {b['weights_nz']} -> {g['weights_nz']} "
          f"({100.0 * g['weights_nz'] / max(1, b['weights_nz']):.1f}% of baseline)")
    dk = results["_div_knockon"]
    print()
    print("DIV KNOCK-ON:")
    print(f"  KB-precompute prologue (once per divide): "
          f"{dk['baseline_precompute_blocks']} -> {dk['batched_precompute_blocks']} blocks "
          f"(saves {dk['precompute_blocks_saved']}, {dk['kb_ks_prefix_stages']} prefix "
          f"stages over {dk['kb_columns_RN']} columns x {dk['kb_values']} KB[k]).")
    print(f"  {dk['verdict']}")


if __name__ == "__main__":
    res = run_bakeoff()
    _print_table(res)
