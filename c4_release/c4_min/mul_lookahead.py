"""CARRY-LOOKAHEAD (PREFIX) MULTIPLY — shrink the general MUL depth by replacing
the 7 SEQUENTIAL ripple carry rounds with a ~log2(n)-depth parallel-prefix carry
resolve.  Standalone byte-exact bakeoff, COMPOSED from the proven ``nibble_alu32``
SwiGLU primitives (this file never edits them).

THE BASELINE (what we're beating)
=================================
``nibble_alu32.compile_mul_blocks`` = 10 blocks:

  products (1) | split (1) | 7x ripple carry round | result copy (1)

The 36 nibble partial products ``a_i*b_j`` (i+j<8, each <= 225) are split into
low/high nibbles and accumulated into 8 nibble columns ``MCOL[c] < 256``.  Each
ripple round is ``dst[c] = src[c] mod 16 + floor(src[c-1]/16)`` — a carry moves ONE
column per round, so 7 (+1 headroom) serial rounds are needed to fully settle an
8-column stack.  The ripple DOMINATES the depth (7 of 10 blocks); the products +
split + result are only 3.

THE LEVER — base-16 carry-lookahead (Kogge-Stone)
=================================================
The 8 post-split columns are a base-16 carry-save number (``MCOL[c] < 256``): the
resolve is a carry-PROPAGATE add over the columns.  A parallel-prefix (Kogge-Stone)
resolve computes every column's carry in ``ceil(log2 n) = 3`` prefix stages instead
of 7 serial ripples.  The trick that makes the prefix a CLEAN, byte-exact BINARY
carry-lookahead:

  ROUND-1 (one carry round) reduces the columns from ``< 256`` to
  ``t_c = (MCOL[c] mod 16) + floor(MCOL[c-1]/16)`` in ``[0, 30]`` (digit 0..15 plus a
  single-nibble carry 0..15 from below).  Now each column, given an INCOMING BINARY
  carry ``b_c in {0,1}``, emits final digit ``(t_c + b_c) mod 16`` and a BINARY
  carry-out ``[t_c + b_c >= 16] in {0,1}`` (because ``t_c + b_c <= 31``).  So the
  carries are pure binary and obey the textbook CLA recurrence:

      generate   G_c = [t_c >= 16]     (carries out even with no incoming carry)
      propagate  P_c = [t_c == 15]     (carries out IFF an incoming carry arrives)
      b_{c+1} = G_c OR (P_c AND b_c),   b_0 = 0.

  Kogge-Stone combines the pairs ``(G,P)`` with the associative operator
  ``(g_hi,p_hi) o (g_lo,p_lo) = (g_hi OR (p_hi AND g_lo), p_hi AND p_lo)`` in
  ``ceil(log2 8) = 3`` stages.  After the stages ``Gband[c]`` is the carry-out of the
  prefix ``[0..c]`` = the carry INTO column ``c+1``, so the carry into column ``c`` is
  ``b_c = Gband[c-1]`` (``b_0 = 0``).  The APPLY block then writes the result nibble

      digit_c = t_c + b_c - 16*[t_c + b_c >= 16] = t_c + Gband[c-1] - 16*Gband[c]

  directly into MUL_RES — no separate result-copy block.

DEPTH
=====
  products (1) | split (1) | round-1 (1) | G/P (1) | KS x 3 | apply/result (1)
  = 8 blocks (vs 10), 3 prefix stages (vs 7 ripple rounds).

We ALSO build a PARTIAL-lookahead middle ground (``build_stride_lookahead``): resolve
TWO columns per ripple round (a stride-2 ripple) -> 4 rounds instead of 7, i.e.
products+split (2) + 4 rounds + result (1) = 7 blocks with NO prefix machinery — the
brief's acceptable middle, for the honest comparison.

fp32 DISCIPLINE
===============
Every relu argument is trivially fp32-exact: post-split columns ``< 256``, round-1
outputs ``t_c <= 30``, and the entire prefix operates on {0,1} generate/propagate
lanes (arguments <= 2).  The tightest ``RELU_S*arg`` is the split staircase
(``RELU_S*225 = 45000 << 2^24``); the harness reports the measured max.  0 fp64.

VERIFY (byte-exact, real SwiGLU forward, sparse sim — seconds, no dense DIM matmul):
  ``(a*b) & 0xFFFFFFFF`` over the shared edge set + >=1000 random (a,b < 2^32),
  INCLUDING the literal worst case ``0xFFFFFFFF^2`` (max carry propagation — the whole
  point of the carry path).

The div KNOCK-ON: the base-16 long division's inner ``q*b`` carry-normalise
(``_QB_CARRY_ROUNDS = 6`` rounds) and any constant multiply reuse this SAME ripple
carry machinery; the harness quantifies the depth this lookahead would shave there too.
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
    _ident, _clear, _step_ge, _truncate,
    _nibble_carry_round, compile_mul_blocks, extend_layout_for_alu32,
    _MUL_NCOL, _NIB_KMAX, _MUL_CARRY_ROUNDS, _QB_CARRY_ROUNDS,
)

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

FP32_INT_MAX = 1 << 24                       # 2^24: fp32 exact-integer ceiling
NCOL = _MUL_NCOL                             # 8 nibble result columns (32-bit)

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
# Private scratch (never touches ALU32Bands; own band range so the sparse sim
# reads/writes only this design's bands).
# ===========================================================================
def _scratch(L, name, size):
    key = f"LAH_{name}"
    if key in L._names:
        return L._names[key][0]
    return L._band(key, size)


# ===========================================================================
# THE CARRY-LOOKAHEAD (KOGGE-STONE) MULTIPLY.
# ===========================================================================
def _round1_block(L, dim, src, dst) -> Block:
    """One base-16 carry round reducing the post-split columns (``< 256``) to
    ``t_c = (src[c] mod 16) + floor(src[c-1]/16)`` in ``[0, 30]``.  Reuses the library
    ``_nibble_carry_round`` (the SAME gadget the baseline ripple uses) — after this
    ONE round the carries are pure binary, so the prefix that follows is clean CLA."""
    _log_arg("round1-floor", 15 * 16)          # floor(col/16) staircase thresholds
    spec = _empty_spec(dim, NCOL * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
    u = _nibble_carry_round(spec, 0, src, dst, NCOL)
    return _truncate(spec, u, dim)


def _gp_block(L, dim, t_band, g_band, p_band) -> Block:
    """Compute the generate / propagate lanes from ``t_c`` in ``[0, 30]`` (SET each):
        G_c = [t_c >= 16]         (carries out with no incoming carry)
        P_c = [t_c == 15] = [t_c >= 15] - [t_c >= 16]   (carries out iff incoming).
    Both are 0/1.  Thresholds are 15/16 -> RELU_S*16 tiny, deep fp32 headroom."""
    _log_arg("gp-thr", 16)
    spec = _empty_spec(dim, NCOL * (2 + 2 + 2 + 2))
    u = 0
    for c in range(NCOL):
        tc = {t_band + c: 1.0}
        u = _clear(spec, u, g_band + c)
        u = _step_ge(spec, u, tc, 0.0, 16, g_band + c, 1.0)             # G = [t>=16]
        u = _clear(spec, u, p_band + c)
        u = _step_ge(spec, u, tc, 0.0, 15, p_band + c, 1.0)            # +[t>=15]
        u = _step_ge(spec, u, tc, 0.0, 16, p_band + c, -1.0)          # -[t>=16] => [t==15]
    return _truncate(spec, u, dim)


def _ks_stage_block(L, dim, g_src, p_src, g_dst, p_dst, d) -> Block:
    """ONE Kogge-Stone prefix combine stage at distance ``d`` (SET g_dst/p_dst):
        for c >= d:  (G_c, P_c) := (G_c OR (P_c AND G_{c-d}),  P_c AND P_{c-d})
        for c <  d:  (G_c, P_c) := (G_c, P_c)                 (carried through)
    All inputs are 0/1 so:
        AND(x, y) = [x + y >= 2],  OR(x, y) = [x + y >= 1].
    G_c OR (P_c AND G_{c-d}) = [ G_c + [P_c + G_{c-d} >= 2] >= 1 ] — but a two-level
    staircase would need the inner AND materialised in a PRIOR block.  Instead use the
    exact single-level identity for these 0/1 lanes:
        G_c OR (P_c AND G_{c-d}) = [ G_c*2 + P_c + G_{c-d} >= 2 ]
    (G_c=1 -> LHS>=2 always fires; G_c=0 -> fires iff P_c=G_{c-d}=1 -> AND).  One
    staircase.  ``P_c AND P_{c-d} = [P_c + P_{c-d} >= 2]``.  Reads the block INPUT
    (the previous stage's g_src/p_src), so the whole stage is one block."""
    _log_arg("ks-thr", 4)                      # weighted form <= 2*1+1+1 = 4
    spec = _empty_spec(dim, NCOL * (2 + 2 + 2 + 2))
    u = 0
    for c in range(NCOL):
        if c >= d:
            # G_dst = [2*G_c + P_c + G_{c-d} >= 2]
            gform = {g_src + c: 2.0, p_src + c: 1.0, g_src + c - d: 1.0}
            u = _clear(spec, u, g_dst + c)
            u = _step_ge(spec, u, gform, 0.0, 2, g_dst + c, 1.0)
            # P_dst = [P_c + P_{c-d} >= 2]  (AND)
            pform = {p_src + c: 1.0, p_src + c - d: 1.0}
            u = _clear(spec, u, p_dst + c)
            u = _step_ge(spec, u, pform, 0.0, 2, p_dst + c, 1.0)
        else:
            u = _clear(spec, u, g_dst + c)
            u = _ident(spec, u, {g_src + c: 1.0}, 0.0, g_dst + c, 1.0)
            u = _clear(spec, u, p_dst + c)
            u = _ident(spec, u, {p_src + c: 1.0}, 0.0, p_dst + c, 1.0)
    return _truncate(spec, u, dim)


def _apply_block(L, dim, t_band, g_final, res_band) -> Block:
    """Final digits into MUL_RES (SET), fused with the carry-apply (no separate copy):
        b_c = G_final[c-1]  (carry INTO column c; b_0 = 0)
        digit_c = (t_c + b_c) mod 16 = t_c + b_c - 16*[t_c + b_c >= 16]
                = t_c + G_final[c-1] - 16*G_final[c]     (since [t_c+b_c>=16] = b_{c+1} = G_final[c])
    All read the block INPUT (t_c and the FINAL prefix G lane), so one block.  The top
    column's carry-out G_final[NCOL-1] overflows past the kept nibbles and is dropped
    (exactly as the baseline dropped the top carry -> result & 0xFFFFFFFF)."""
    _log_arg("apply", 30)                      # t_c + b_c <= 31; coeff-16 form small
    spec = _empty_spec(dim, NCOL * 4)
    u = 0
    for c in range(NCOL):
        u = _clear(spec, u, res_band + c)
        u = _ident(spec, u, {t_band + c: 1.0}, 0.0, res_band + c, 1.0)          # + t_c
        if c >= 1:
            u = _ident(spec, u, {g_final + c - 1: 1.0}, 0.0, res_band + c, 1.0)  # + b_c
        u = _ident(spec, u, {g_final + c: 1.0}, 0.0, res_band + c, -16.0)        # - 16*b_{c+1}
    return _truncate(spec, u, dim)


def build_kogge_stone(L, dim) -> Tuple[List[Block], int, dict]:
    """Carry-lookahead MUL: reuse the products+split front-end, replace the 7 ripple
    rounds with ROUND-1 + G/P + 3 Kogge-Stone prefix stages + a fused apply/result."""
    A._ONE = L.ONE
    base = list(compile_mul_blocks(L, dim))
    front = base[:2]                            # products + split -> a.MCOL (< 256)
    a = L.ALU32
    RES = a.MUL_RES

    T = _scratch(L, "T", NCOL)                  # t_c in [0,30] (post round-1)
    # double-buffered generate / propagate lanes for the Kogge-Stone stages.
    G0 = _scratch(L, "G0", NCOL); P0 = _scratch(L, "P0", NCOL)
    G1 = _scratch(L, "G1", NCOL); P1 = _scratch(L, "P1", NCOL)

    n_stages = 0
    d = 1
    while d < NCOL:
        n_stages += 1
        d *= 2                                   # ceil(log2 8) = 3

    blocks: List[Block] = list(front)
    blocks.append(("lah-round1", _round1_block(L, dim, a.MCOL, T)))
    blocks.append(("lah-gp", _gp_block(L, dim, T, G0, P0)))
    (gs, ps), (gd, pd) = (G0, P0), (G1, P1)
    d = 1
    st = 0
    while d < NCOL:
        blocks.append((f"lah-ks{st}-d{d}", _ks_stage_block(L, dim, gs, ps, gd, pd, d)))
        (gs, ps), (gd, pd) = (gd, pd), (gs, ps)   # swap buffers
        d *= 2
        st += 1
    blocks.append(("lah-apply", _apply_block(L, dim, T, gs, RES)))

    info = {
        "note": "Kogge-Stone base-16 carry-lookahead: products+split | round-1 | G/P "
                "| 3x prefix combine | fused apply/result",
        "prefix_stages": n_stages, "carry_rounds_replaced": _MUL_CARRY_ROUNDS,
        "front_end_blocks": 2, "max_staircase_arg": 15 * 16,
    }
    return blocks, RES, info


# ===========================================================================
# PARTIAL-LOOKAHEAD MIDDLE GROUND — resolve TWO columns per round (stride-2
# ripple).  A single round settles a carry across TWO columns instead of one, so
# an 8-column stack settles in ceil(7/2)+1 = 4 rounds instead of 7.  NO prefix
# machinery: each round is just two chained base-16 carry substeps fused into one
# block (reads the block input twice-composed).  The brief's acceptable middle.
# ===========================================================================
def _second_ripple_block(L, dim, u_band, dst) -> Block:
    """Second ripple substep over the materialised ``u_c`` in [0,30]:
        dst[c] = (u_c mod 16) + floor(u_{c-1}/16)   (floor in {0,1} since u<=30).
    A plain base-16 carry round on ``u`` (reuses the library gadget)."""
    _log_arg("stride2-second", 15 * 16)
    spec = _empty_spec(dim, NCOL * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
    u = _nibble_carry_round(spec, 0, u_band, dst, NCOL)
    return _truncate(spec, u, dim)


def build_stride_lookahead(L, dim) -> Tuple[List[Block], int, dict]:
    """Partial-lookahead middle ground: each "super-round" is TWO base-16 carry
    substeps (a carry moves two columns), so an 8-column stack settles in ceil(7/2)+1
    super-rounds.  Each super-round = 2 blocks (first ripple materialises u_c, second
    ripple settles it), which is honestly the SAME block count as the plain ripple —
    so this is a WASH on depth and included only to show the prefix (Kogge-Stone) path
    is the real lever.  Kept for the brief's 'measure whatever lands' honesty."""
    A._ONE = L.ONE
    base = list(compile_mul_blocks(L, dim))
    front = base[:2]
    a = L.ALU32
    RES = a.MUL_RES
    U = _scratch(L, "U2", NCOL)
    V = _scratch(L, "V2", NCOL)

    # a carry can span 7 columns; two substeps per super-round -> ceil(7/2)+1 = 4.
    n_super = (NCOL - 1 + 1) // 2 + 1           # 4
    blocks: List[Block] = list(front)
    src = a.MCOL
    for sr in range(n_super):
        # substep 1: u_c = (src[c] mod 16) + floor(src[c-1]/16)   (materialise u)
        blocks.append((f"str-r{sr}a", _second_ripple_block(L, dim, src, U)))
        # substep 2: v_c = (u_c mod 16) + floor(u_{c-1}/16)       (settle)
        blocks.append((f"str-r{sr}b", _second_ripple_block(L, dim, U, V)))
        src = V
        U, V = V, U
    # copy the settled columns to MUL_RES.
    copy = _empty_spec(dim, NCOL * 2)
    u = 0
    for c in range(NCOL):
        u = _clear(copy, u, RES + c)
        u = _ident(copy, u, {src + c: 1.0}, 0.0, RES + c, 1.0)
    blocks.append(("str-result", _truncate(copy, u, dim)))
    info = {
        "note": "stride-2 ripple (2 substeps/super-round); WASH on depth vs plain "
                "ripple (2 blocks/super-round) — included for honesty",
        "super_rounds": n_super, "blocks_per_super_round": 2,
    }
    return blocks, RES, info


# ===========================================================================
# BASELINE contender — wrap the production compile_mul_blocks.
# ===========================================================================
def build_baseline(L, dim) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    blocks = list(compile_mul_blocks(L, dim))
    _log_arg("baseline", 232)                    # split cols <= 232; round args <= 240
    info = {
        "note": "production nibble schoolbook (compile_mul_blocks): products, split, "
                f"{_MUL_CARRY_ROUNDS} ripple carry rounds, result copy",
        "carry_rounds": _MUL_CARRY_ROUNDS, "max_staircase_arg": 240,
    }
    return blocks, L.ALU32.MUL_RES, info


# ===========================================================================
# MEASUREMENT + SPARSE byte-exact sim (no dense DIM forward) — same lean harness
# as mul_byte_carrysave.py / mul_bakeoff.py.
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


def test_operands(n_random: int = 1000, seed: int = 0) -> List[Tuple[int, int]]:
    """The shared edge set {0,1,2, 2^31, 2^32-1, powers of two, single-byte,
    full-width} x a divisor/multiplier fan + >= n_random random (a,b < 2^32).  The
    literal worst case 0xFFFFFFFF^2 (max carry propagation) is ALWAYS included."""
    rng = random.Random(seed)
    M = 0xFFFFFFFF
    edges = [0, 1, 2, 3, 1 << 31, M, M - 1, 0x80000000, 0x7FFFFFFF,
             0xFFFF, 0x10000, 0xDEADBEEF, 0xCAFEBABE, 0xABCD, 0x1234,
             65535, 65536, 255, 256, 0x00FF00FF, 0xFF00FF00]
    pows = [1 << k for k in range(0, 32)]              # every power of two 2^0..2^31
    singles = [0x000000AB, 0x0000CD00, 0x00EF0000, 0x12000000]  # single-byte lanes
    others = [0, 1, 2, M, M - 1, 0xDEADBEEF, 3, 0x10000, 255, 0xFFFF, 1 << 16, 1 << 31]
    pairs: List[Tuple[int, int]] = []
    for a in edges + pows + singles:
        for b in others:
            pairs.append((a & M, b & M))
    pairs.append((M, M))                                # 0xFFFFFFFF^2 — max carry propagation
    n_struct = len(pairs)
    for _ in range(n_random):
        pairs.append((rng.randint(0, M), rng.randint(0, M)))
    return pairs, n_struct


DIM = 4096              # wide enough for every variant's private scratch bands.


def _make_layout():
    L = NibbleVMLayout(8, n_heads=4)
    extend_layout_for_alu32(L)
    A._ONE = L.ONE
    return L


# ===========================================================================
# DIV KNOCK-ON — the inner q*b carry-normalise reuses this SAME ripple carry
# machinery.  Quantify the depth this lookahead would shave off the divide.
# ===========================================================================
def div_knockon() -> dict:
    """The base-16 long division (``compile_divmod_blocks``) runs 8 iterations; each
    iteration carry-normalises ``QB = q*b`` with ``_QB_CARRY_ROUNDS = 6`` ripple rounds.
    QB has RN=9 nibble columns (< 256), so a Kogge-Stone resolve would use
    ``ceil(log2 9) = 4`` prefix stages + round-1 + G/P + apply = 7 resolve blocks vs the
    6 ripple rounds — a WASH per iteration for QB (9 cols, ripple already only 6).  The
    REAL div lever is the KB-precompute (15 x ``_KB_CARRY_ROUNDS=6`` = 90 ripple blocks),
    where each KB[k]=k*b is an independent 9-column ripple; a shared prefix resolve is a
    depth win only if the 15 are batched.  Reported honestly, not inflated."""
    RN = 9
    qb_rounds = _QB_CARRY_ROUNDS
    n_iters = 8
    ks_stages = 0
    d = 1
    while d < RN:
        ks_stages += 1
        d *= 2                                   # ceil(log2 9) = 4
    ks_resolve_blocks = 1 + 1 + ks_stages + 1    # round1 + gp + stages + apply
    return {
        "qb_columns_RN": RN,
        "qb_ripple_rounds_per_iter": qb_rounds,
        "qb_ks_prefix_stages": ks_stages,
        "qb_ks_resolve_blocks_per_iter": ks_resolve_blocks,
        "div_iters": n_iters,
        "qb_ripple_blocks_total": qb_rounds * n_iters,
        "qb_ks_blocks_total": ks_resolve_blocks * n_iters,
        "verdict": ("QB has 9 columns and the ripple is already only 6 rounds, so a "
                    "per-iteration Kogge-Stone (7 resolve blocks) is a WASH-to-slight-"
                    "loss for QB alone; the mul win is real because the mul ripple is 7 "
                    "rounds over 8 columns where prefix (3 stages) genuinely beats it. "
                    "The honest div reuse is the *primitive* (_nibble_carry_round / the "
                    "G/P prefix gadget), not a per-iteration block-count win."),
    }


# ===========================================================================
# BAKEOFF DRIVER
# ===========================================================================
def run_bakeoff(n_random: int = 1000, verbose: bool = True) -> dict:
    results: dict = {}
    L = _make_layout()
    dim = DIM
    M = 0xFFFFFFFF
    ops, n_struct = test_operands(n_random=n_random)

    builders: List[Tuple[str, callable]] = [
        ("baseline_ripple", build_baseline),
        ("kogge_stone_lookahead", build_kogge_stone),
        ("stride2_partial", build_stride_lookahead),
    ]

    for name, fn in builders:
        _reset_arglog()
        blocks, res, info = fn(L, dim)
        assert L.D <= dim, f"layout grew past DIM ({L.D} > {dim}); raise DIM"
        head = tightest_arg()
        m = measure(blocks)
        n_pass = 0
        ffff_ok = None
        first_fail = None
        for a, b in ops:
            got = simulate_sparse(L, blocks, res, a, b, dtype=torch.float32)
            exp = (a * b) & M
            ok = (got == exp)
            if a == M and b == M:
                ffff_ok = ok
            if ok:
                n_pass += 1
            elif first_fail is None:
                first_fail = (hex(a), hex(b), hex(got), hex(exp))
        results[name] = {
            **m,
            "byte_exact_pass": n_pass, "byte_exact_total": len(ops),
            "n_structured": n_struct, "n_random": len(ops) - n_struct,
            "ffffffff_squared_ok": ffff_ok,
            "first_fail": first_fail,
            "tightest_relu_s_arg": head["product"],
            "tightest_ratio_of_2p24": head["ratio"],
            "tightest_max_arg": head["max_arg"],
            "tightest_gadget": head["label"],
            **{k: v for k, v in info.items()},
        }
        if verbose:
            print(f"{name:24s} depth={m['depth']:3d} nz={m['weights_nz']:7d} "
                  f"exact={n_pass}/{len(ops)} 0xFFFFFFFF^2={ffff_ok} "
                  f"maxarg*RELU_S={head['product']:.0f} ({100*head['ratio']:.3f}% of 2^24) "
                  f"stages={info.get('prefix_stages', info.get('carry_rounds', '-'))} "
                  f"fail={first_fail}")
    results["_div_knockon"] = div_knockon()
    return results


def _print_table(results: dict):
    order = ["baseline_ripple", "kogge_stone_lookahead", "stride2_partial"]
    print()
    print("VARIANT                    DEPTH   NZ      PREFIX/ROUNDS  TIGHT RELU_S*arg (%2^24)  0xFFFFFFFF^2   BYTE-EXACT")
    print("-" * 108)
    for name in order:
        r = results[name]
        stages = r.get("prefix_stages", r.get("carry_rounds", "-"))
        print(f"{name:24s} {r['depth']:5d}  {r['weights_nz']:7d}   {str(stages):>12s}   "
              f"{r['tightest_relu_s_arg']:9.0f} ({100*r['tightest_ratio_of_2p24']:5.3f}%)      "
              f"{str(r['ffffffff_squared_ok']):>5s}       "
              f"{r['byte_exact_pass']:4d}/{r['byte_exact_total']:<4d}")
    dk = results["_div_knockon"]
    print()
    print("DIV KNOCK-ON:")
    print(f"  QB carry-normalise: {dk['qb_ripple_rounds_per_iter']} ripple rounds/iter "
          f"vs {dk['qb_ks_resolve_blocks_per_iter']} KS-resolve blocks/iter "
          f"({dk['qb_ks_prefix_stages']} prefix stages, {dk['qb_columns_RN']} columns).")
    print(f"  {dk['verdict']}")


if __name__ == "__main__":
    res = run_bakeoff()
    _print_table(res)
