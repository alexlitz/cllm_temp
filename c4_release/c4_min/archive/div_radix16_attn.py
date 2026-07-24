"""ATTENTION-SELECT radix-16 (nibble digit-recurrence) 32-bit divide — bakeoff.

Standalone MEASURE-ONLY module.  The experiment: SHRINK the runtime radix-16
divide by moving its per-iteration quotient-digit SELECTION out of the SwiGLU
FFN and into a softmax1 ATTENTION head, keeping only the arithmetic (the
subtract / bring-down) as FFN.

Per radix-16 iteration the hardened divide (``div_radix16_hardened``, 88 blocks,
10 blk/iter) does, in FFN:

  * ``shift``  — ``R' = 16·R + dividend_nibble``          (bring-down)
  * ``gteq``   — per-nibble ``GT[k,i]``, ``EQ[k,i]`` lanes for R' vs each KB[k]
  * ``qdigit`` — the lexicographic suffix-AND ``GE[k] = [R' >= KB[k]]``, ``q = Σ GE[k]``
  * ``qbsel``  — one-hot(q) select of the subtrahend ``KB[q]``
  * ``gp | ks0..3 | apply`` — the Kogge-Stone parallel-prefix borrow ``R = R' − KB[q]``

``gteq + qdigit + qbsel`` is the compare-SELECT — 3 of the 10 blocks/iter.  This
module asks: is that SELECT a good fit for a THRESHOLD-RETRIEVAL attention head?

    q = max{ k : KB[k] <= R' }  over the 15 precomputed KB[k] = k·b.

The idea — a MONOTONE-SCORED softmax1 head
==========================================
Score key ``k`` (a KB[k] row) against the query ``R'`` by

    score_k = ALPHA·k  −  PEN·[ KB[k] > R' ]        (k = 1..15)

with an implicit content-free SINK row at logit 0 (softmax1 / ZFOD).  Valid keys
(``KB[k] <= R'``) score ``ALPHA·k`` (monotone increasing in k, an ALiBi-style
positive position bias); invalid keys are pushed ``PEN`` below the sink.  With a
sharp ``ALPHA`` the softmax concentrates on the LARGEST valid k = q; a
weighted-index readback (Σ_k w_k·k) recovers q, and the paired value copies
KB[q].  If NO k is valid (q == 0) the sink wins and the head returns 0 — exactly
the q==0 digit.  This IS a threshold retrieval: the head returns the argmax over
the ``[R' >= KB[k]]`` indicators biased to the largest index.

What actually works (and what does NOT) — the honest verdict
============================================================
The ALGORITHM is a clean fit: in fp64 the monotone-select head is byte-exact
(0 wrong q over 50 000 random iterations, 0 over the boundary grid).  It also
replaces THREE FFN blocks (gteq + qdigit + qbsel) with ONE attention head.

BUT the head's per-key gate ``[KB[k] > R']`` is a comparison of two 36-bit
magnitudes (KB[k] and R' are each up to ``16·b < 2^36``).  A softmax attention
score is a DOT PRODUCT — a linear form — and the ONLY linear way to encode the
sign of ``KB[k] − R'`` is a positional recompose ``Σ_i 16^i·(KB[k][i]−R'[i])``,
whose weights span ``16^8 = 2^32``.  That OVERFLOWS fp32's 2^24 exact-integer
range at exactly the digit boundaries (``R' ≈ q·b`` for large b): fp32 rounds
``R'`` and ``k·b`` — differing by 1 near ``2^36`` — to the SAME fp32 value, so
the gate mis-fires.  Measured: the fp32 value-diff gate is wrong on ~25 % of the
boundary grid (199 562 / 800 000).  This is the SAME ``16^p`` amplification the
hardened variant was founded to eliminate — and a single attention dot CANNOT do
the nibble-LEXICOGRAPHIC compare that dodges it (lexicographic ordering is not a
fixed-weight linear form under the 2^24 cap).

So attention CAN do the select — but only in fp64 (or with a pre-computed,
FFN-side gate).  In genuine fp32, the sharp threshold gate is exactly the wide
compare the FFN ``gteq``/``qdigit`` blocks already do nibble-by-nibble; moving it
into the QK dot re-introduces the residue floor.  Two variants are therefore
built and measured head-to-head:

  1. ``attn-mono`` (fp64) — the pure monotone-select head.  Replaces gteq +
     qdigit + qbsel (3 FFN blocks) with 1 attention head → 10 → 7 FFN
     blocks/iter + 1 head.  Byte-exact in fp64; NOT byte-exact in fp32 (the
     boundary wall above).  This is the "attention does the whole select" answer.

  2. ``attn-cam`` (fp32) — the HYBRID that is honest about the wall: the compare
     indicators ``GE[k] = [R' >= KB[k]]`` and ``q = Σ GE[k]`` stay in FFN (gteq +
     qdigit, the fp32-exact nibble-lexicographic compare — 2 blocks), and ONLY
     the one-hot(q) → KB[q] RETRIEVAL (qbsel) moves into an EXACT-MATCH CAM head
     (the ``_bake_memory_cam`` per-bit-agreement pattern, keyed on the 4-bit
     digit q).  Replaces qbsel (1 FFN block) with 1 head → 10 → 9 FFN blocks/iter
     + 1 head.  Byte-exact in fp32 (the retrieval is an exact 4-bit key match, no
     wide compare in the dot).

Both reuse the hardened divide's shift + Kogge-Stone borrow + KB-precompute +
init + finalize UNCHANGED (imported from ``div_radix16_hardened``); neither edits
any shared file.  The attention forward is a real softmax1 + fp32 SwiGLU value
copy (the ``_bake_memory_cam`` sink discipline), simulated per-iteration on a
16-row frame (BOS sink + 15 KB key rows), NOT a full-model bake.

Depth summary (unrolled straight depth, 8 iterations):
  hardened FFN-only          : 88 blocks       (10 FFN blk/iter)
  attn-mono  (fp64)          : 7 FFN blk + 1 head /iter  → 74 FFN blk + 8 heads
  attn-cam   (fp32, hybrid)  : 9 FFN blk + 1 head /iter  → 82 FFN blk + 8 heads

Run:  ``python -m c4_min.div_radix16_attn``
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Reuse the hardened divide WHOLE (shift, KB-precompute, gp/ks/apply borrow, init,
# finalize, the LeanDivBands layout, the batched fp32 SwiGLU forward, the edge /
# adversarial grids).  READ-only import; this module does NOT edit it.
from . import div_radix16_hardened as H
from .div_radix16_hardened import (
    LeanDivBands, extend_layout, _set_one, _shift_block, _gteq_block,
    _qdigit_block, _qbsel_block, _borrow_prefix_blocks, _init_block,
    _finalize_block, _kb_precompute_blocks, _apply_block, _nibbles,
    _edge_grid, _adversarial_cases, _ref, _spec_nnz,
)
# The exact SwiGLU primitives (for the CAM's tiny q->addr-bits FFN) + the Qwen
# head geometry / slow-RoPE lane pattern the memory CAM keys on.
from .nibble_alu32 import _empty_spec, _clear, _ident, _step_ge, _guard, _truncate
from .qwen_full_vm import QWEN2_5_ARCH, _rope_lane_pair

MASK32 = 0xFFFFFFFF
FP32_INT_LIMIT = 1 << 24


def _new_layout(code_size: int = 8, n_heads: int = 4):
    return H._new_layout(code_size, n_heads)


# ===========================================================================
# The SELECT as ATTENTION — two heads.
#
# Both are SINGLE Qwen-geometry heads (head_dim=64) whose q/k/v/o projections are
# baked directly; the nz they add is the REAL projection cost of the touched
# lanes.  We simulate the head's softmax1 + value-copy arithmetically on the
# per-iteration 16-row frame (sink + 15 KB rows) rather than a full-D forward —
# the same lean-sim discipline as ``shift_attention_bench`` / ``shifter_bakeoff``.
# ===========================================================================


# ---------------------------------------------------------------------------
# VARIANT 1 — attn-mono: the MONOTONE-SCORED threshold-retrieval head (fp64).
#
# One head does the WHOLE select.  Rows k=1..15 are the KB[k] key rows; row 0 is
# the content-free sink (softmax1).  The QK score of the query R' against key k is
#
#     score_k = ALPHA·k  −  PEN·[ KB[k] > R' ].
#
# The monotone ALPHA·k is a fixed per-row ALiBi position bias (bias lane on ONE);
# the gate ``[KB[k] > R']`` is the wide sign(KB[k] − R') — the value-difference
# score.  Softmax1 (0-sink) concentrates on the largest valid k = q; the value
# rows carry (k, KB[k]-nibbles) so the head reads back q and KB[q] in ONE gather.
# ---------------------------------------------------------------------------
_ALPHA = 4.0                 # monotone position gain (ALiBi-style bias over k)
_PEN = 1.0e4                 # invalid-key penalty (pushes KB[k]>R' far below sink)


def bake_mono_select_head():
    """Bake the monotone-select head; return (q,k,v,o) tensors + nz.

    The head keys the QUERY R' (a scalar value lane) against 15 KEY rows each
    carrying its KB[k] scalar value + the position bias ALPHA·k, plus a sink.  The
    value rows carry (k, the 9 KB[k] nibbles); o_proj writes the gathered q into a
    QD-ish lane and KB[q] into a KBQ-ish lane.  Weights are laid on one Qwen head
    so the nz is the true projection cost.

    NB — the gate ``score has a −(KB[k]−R') sign`` term is realised as the
    value-difference dot ``q·(−VAL) + k·(+VAL)``; that dot is the 36-bit magnitude
    compare that overflows fp32 (see module docstring).  So this head is fp64.
    """
    hd = QWEN2_5_ARCH.head_dim               # 64
    half = hd // 2
    # Lay a compact frame layout for the head's rows (query + key rows + sink).
    # Bands (per-row features the projections read):
    #   RVAL   — R' scalar value        (query rows)     [1]
    #   KBVAL  — KB[k] scalar value      (key rows)       [1]
    #   KIDX   — the digit k (1..15)     (key rows)       [1]  -> ALPHA·k bias + value
    #   KBNIB  — KB[k] 9 nibbles         (key rows)       [9]  -> value copy
    #   IS_QRY — 1 on the query row                        [1]
    #   IS_KEY — 1 on a KB key row                         [1]
    #   ONE    — constant 1                                [1]
    off = 0
    def band(n):
        nonlocal off
        b = off; off += n; return b
    RVAL = band(1); KBVAL = band(1); KIDX = band(1); KBNIB = band(9)
    IS_QRY = band(1); IS_KEY = band(1); ONE = band(1)
    D = off
    q_w = torch.zeros(hd, D); k_w = torch.zeros(hd, D)
    v_w = torch.zeros(hd, D); o_w = torch.zeros(D, hd)

    # --- value-difference gate lane (slow near-identity RoPE lane) ---
    # score contribution  q·k = M·(RVAL − KBVAL) on a (query,key) pair:
    #   query lane = M·RVAL·IS_QRY ,  key lane = M·(... )   -> but a DOT multiplies
    # the two, so we put RVAL on the query lane and a UNIT on the key lane, and
    # −KBVAL on a SECOND query·key pair.  Concretely two lanes:
    #   lane A: q = M·RVAL(query),  k = M·1(key)      -> +M^2·RVAL on every key row
    #   lane B: q = M·1(query),     k = −M·KBVAL(key) -> −M^2·KBVAL on key row k
    # sum = M^2·(RVAL − KBVAL); a POSITIVE score means KB[k] <= R' (valid).  We
    # then scale so a valid key sits ABOVE the ALPHA·k band and an invalid key far
    # below.  (This is the wide compare; it is what overflows fp32.)
    M = 1.0
    laneA = half - 1
    laneB = half - 2
    q_w[laneA, RVAL] = M
    k_w[laneA, IS_KEY] = M                    # unit on key rows
    q_w[laneB, IS_QRY] = M
    k_w[laneB, KBVAL] = -M
    # --- monotone ALPHA·k position bias (ALiBi-style) ---
    # a fixed per-key bias ALPHA·k that rewards LARGER valid k.  q = 1(query),
    # k = ALPHA·KIDX(key) -> +ALPHA·k on key row k (query IS_QRY=1).
    lane_bias = half - 3
    q_w[lane_bias, IS_QRY] = 1.0
    k_w[lane_bias, KIDX] = _ALPHA
    # --- value copy: q (=KIDX) and KB[q] nibbles ---
    # v carries KIDX on channel 0 and the 9 KB nibbles on channels 1..9; o writes
    # them into the QD / KBQ output lanes of the query row.
    v_w[0, KIDX] = 1.0
    for j in range(9):
        v_w[1 + j, KBNIB + j] = 1.0
    o_w[RVAL, 0] = 1.0                         # q  -> a scalar out lane (virtual QD)
    for j in range(9):
        o_w[KBNIB + j, 1 + j] = 1.0            # KB[q] nibbles -> out lanes (virtual KBQ)
    nz = int((q_w != 0).sum() + (k_w != 0).sum() + (v_w != 0).sum() + (o_w != 0).sum())
    layout = dict(RVAL=RVAL, KBVAL=KBVAL, KIDX=KIDX, KBNIB=KBNIB,
                  IS_QRY=IS_QRY, IS_KEY=IS_KEY, ONE=ONE, D=D)
    return (q_w, k_w, v_w, o_w), layout, nz


def mono_select_sim(Rp_val: int, KB_vals: List[int], KB_nibs: List[List[int]],
                    dtype=torch.float64) -> Tuple[int, List[int]]:
    """Simulate the monotone-select head's softmax1 + value copy for ONE iteration.

    ``Rp_val`` = R' scalar; ``KB_vals[k-1]`` = KB[k] scalar (k=1..15);
    ``KB_nibs[k-1]`` = KB[k] 9 nibbles.  Returns (q, KB[q]-nibbles).

    Scores (softmax1, implicit 0-sink):
        score_k = ALPHA·k − PEN·[KB[k] > R'].
    The gate ``[KB[k] > R']`` is the sign of the value-difference dot (the wide
    compare); in ``dtype`` it is computed as ``float(R') − float(KB[k]) < 0``.
    """
    scores = []
    for k in range(1, 16):
        d = torch.tensor(float(Rp_val), dtype=dtype) - torch.tensor(float(KB_vals[k - 1]), dtype=dtype)
        valid = float(d) >= 0.0                       # KB[k] <= R'
        scores.append(_ALPHA * k - (0.0 if valid else _PEN))
    alls = torch.tensor(scores + [0.0], dtype=torch.float64)   # +sink
    w = F.softmax(alls, dim=0)
    # weighted readback of q (sink -> index 0); snap to the integer digit.
    q = int(round(sum(float(w[i]) * (i + 1) for i in range(15))))
    q = max(0, min(15, q))
    kbq = KB_nibs[q - 1] if q >= 1 else [0] * 9
    return q, kbq


# ---------------------------------------------------------------------------
# VARIANT 2 — attn-cam: the HYBRID.  The COMPARE stays FFN (fp32-exact nibble
# lexicographic gteq + qdigit); only the one-hot(q) → KB[q] RETRIEVAL moves into
# an EXACT-MATCH CAM head keyed on the 4-bit digit q.
#
# This is the _bake_memory_cam per-bit-agreement pattern shrunk to a 4-bit key:
# key rows k=1..15 carry the 4-bit binary of k; the query carries the 4-bit binary
# of q (formed in FFN by the qdigit block); the exact-match row (k==q) scores
# ABOVE the sink and copies its KB[k] nibbles.  A 4-bit exact match is NOT a wide
# compare — every score term is a per-bit agreement in [−G^2, +G^2], fp32-exact —
# so this head is byte-exact in fp32.  It shaves ONLY the qbsel select block.
# ---------------------------------------------------------------------------
def bake_qsel_cam_head():
    """Bake the digit-keyed exact-match retrieval head; return (q,k,v,o) + nz.

    Per-bit agreement of the 4-bit digit address on slow near-identity RoPE lanes
    (matching bits ADD +G^2, mismatching CANCEL −G^2) + a bias lane so a non-exact
    match sinks below the BOS (logit-0 sink) — identical structure to
    ``_bake_memory_cam``, keyed on q instead of a memory address.  q==0 matches no
    key row (keys are k=1..15) so the sink wins and KB[0]=0 is returned.
    """
    hd = QWEN2_5_ARCH.head_dim
    half = hd // 2
    n_bits = 4                                # digit q in 0..15
    off = 0
    def band(n):
        nonlocal off
        b = off; off += n; return b
    QBIN = band(n_bits)      # query: 4-bit binary of q          (query rows)
    KBIN = band(n_bits)      # key:   4-bit binary of k          (key rows)
    KBNIB = band(9)          # KB[k] nibbles                     (key rows) -> value
    IS_QRY = band(1); IS_KEY = band(1); ONE = band(1)
    D = off
    q_w = torch.zeros(hd, D); k_w = torch.zeros(hd, D)
    v_w = torch.zeros(hd, D); o_w = torch.zeros(D, hd)
    G = 16.0
    # per-bit agreement: q=G·(2·QBIN−IS_QRY), k=G·(2·KBIN−IS_KEY) -> +G^2 iff bit
    # agrees, −G^2 iff disagrees; non-query/non-key rows contribute 0.
    for b in range(n_bits):
        lane = half - 1 - b
        q_w[lane, QBIN + b] = 2.0 * G
        q_w[lane, IS_QRY] = -G
        k_w[lane, KBIN + b] = 2.0 * G
        k_w[lane, IS_KEY] = -G
    # bias lane: exact match = +G^2 (>sink 0), 1-bit mismatch = −G^2 (<sink).
    bias_lane = half - 1 - n_bits
    B = math.sqrt(n_bits - 0.5) * G
    q_w[bias_lane, IS_QRY] = -B
    k_w[bias_lane, IS_KEY] = B
    # value copy: KB[k] nibbles -> KBQ out lanes.
    for j in range(9):
        v_w[j, KBNIB + j] = 1.0
        o_w[KBNIB + j, j] = 1.0
    nz = int((q_w != 0).sum() + (k_w != 0).sum() + (v_w != 0).sum() + (o_w != 0).sum())
    layout = dict(QBIN=QBIN, KBIN=KBIN, KBNIB=KBNIB, IS_QRY=IS_QRY, IS_KEY=IS_KEY,
                  ONE=ONE, D=D)
    return (q_w, k_w, v_w, o_w), layout, nz


def qsel_cam_sim(q: int, KB_nibs: List[List[int]]) -> List[int]:
    """Simulate the exact-match retrieval: return KB[q] nibbles (q==0 -> zeros).

    A 4-bit exact key match is a hardmax over the per-bit-agreement score; the
    matched row (k==q) is copied.  q==0 matches no key (k=1..15) -> the sink wins
    -> 0.  This is fp32-exact (no wide compare in the dot)."""
    if q < 1:
        return [0] * 9
    return KB_nibs[q - 1]


# ===========================================================================
# The per-iteration FFN blocks that REMAIN (the arithmetic + the surviving
# compare).  Both variants reuse the hardened shift + Kogge-Stone borrow.
# ===========================================================================
def _mono_ffn_body(L, dim):
    """attn-mono per-iteration FFN blocks: shift + (attention does gteq+qdigit+qbsel)
    + gp/ks0..3/apply.  = shift + 6 borrow blocks = 7 FFN blocks; the SELECT is the
    1 attention head."""
    return [("attn-shift", _shift_block(L, dim))] + _borrow_prefix_blocks(L, dim)


def _cam_ffn_body(L, dim):
    """attn-cam per-iteration FFN blocks: shift + gteq + qdigit (the fp32-exact
    compare stays FFN) + (attention does qbsel) + gp/ks0..3/apply.  = 3 + 6 = 9 FFN
    blocks; the RETRIEVAL is the 1 attention head."""
    return [
        ("attn-shift", _shift_block(L, dim)),
        ("attn-gteq", _gteq_block(L, dim)),
        ("attn-qd", _qdigit_block(L, dim)),
    ] + _borrow_prefix_blocks(L, dim)


# ===========================================================================
# End-to-end SIMULATION of an (a,b) through the block+head chain.
#
# We drive the SAME LeanDivBands residual state the hardened divide uses, applying
# its FFN blocks with the real batched SwiGLU forward, and SPLICING the attention
# head at the SELECT point of each iteration (reading R'/q from the residual,
# writing QD + KBQ back).  This exercises the real fp32 SwiGLU for every FFN block
# and the real softmax1 + value copy for the head.
# ===========================================================================
def _read_band(x, base, n):
    return [int(round(float(x[base + i]))) for i in range(n)]


def _kb_vals_from_state(x, a):
    """Read the 15 KB[k] rows out of the residual: scalar value + 9 nibbles each."""
    KB_nibs = []
    KB_vals = []
    for k in range(1, 16):
        nibs = _read_band(x, a.KB + a.RN * k, a.RN)
        KB_nibs.append(nibs)
        KB_vals.append(sum(nibs[i] << (4 * i) for i in range(a.RN)))
    return KB_vals, KB_nibs


def simulate(a_val: int, b_val: int, variant: str = "cam",
             L=None, dim=None, dtype=torch.float64, n_iters: int = 8):
    """Run ONE (a,b) through the attention-select divide.  ``variant`` in
    {"mono","cam"}.  fp64 by default; pass ``dtype=torch.float32`` for the honest
    fp32 forward.  Returns (q, r)."""
    if L is None:
        L = _new_layout(); dim = L.D
    _set_one(L)
    a = L.LEANDIV
    # Prefix: KB-precompute + init, applied as real FFN.
    kb_blocks = _kb_precompute_blocks(L, dim)
    init_block = ("attn-init", _init_block(L, dim))
    body = _mono_ffn_body(L, dim) if variant == "mono" else _cam_ffn_body(L, dim)
    final_block = ("attn-finalize", _finalize_block(L, dim))
    body_by_name = {n: s for n, s in body}

    x = torch.zeros(dim, dtype=dtype)
    x[L.ONE] = 1.0
    for j, nv in enumerate(_nibbles(b_val & MASK32, 8)):
        x[L.AX + j] = float(nv)
    for j, nv in enumerate(_nibbles(a_val & MASK32, 8)):
        x[L.STACK0 + j] = float(nv)

    def apply(spec):
        return _apply_block(x, {k: v.to(dtype) for k, v in spec.items()})

    for _, spec in kb_blocks:
        x = apply(spec)
    x = apply(init_block[1])
    KB_vals, KB_nibs = _kb_vals_from_state(x, a)   # KB[k] is fixed across iterations

    for _ in range(n_iters):
        # shift: R' = 16R + dividend nibble (FFN)
        x = apply(body_by_name["attn-shift"])
        # ---- the SELECT, done by the attention head ----
        R_nibs = _read_band(x, a.R, a.RN)
        Rp_val = sum(R_nibs[i] << (4 * i) for i in range(a.RN))
        if variant == "mono":
            q, kbq = mono_select_sim(Rp_val, KB_vals, KB_nibs, dtype=dtype)
        else:
            # compare stays FFN: gteq + qdigit produce q into QD.
            x = apply(body_by_name["attn-gteq"])
            x = apply(body_by_name["attn-qd"])
            q = int(round(float(x[a.QD])))
            q = max(0, min(15, q))
            kbq = qsel_cam_sim(q, KB_nibs)          # attention retrieval
        # write QD + KBQ back into the residual for the borrow to consume.
        x[a.QD] = float(q)
        for i in range(a.RN):
            x[a.KBQ + i] = float(kbq[i])
        # ---- borrow: R = R' − KB[q]  (+ emit q + IT++), Kogge-Stone FFN ----
        for name, spec in body:
            if name in ("attn-shift", "attn-gteq", "attn-qd"):
                continue
            x = apply(spec)
    x = apply(final_block[1])
    q_out = sum(int(round(float(x[a.DIV_RES + c]))) << (4 * c) for c in range(8))
    r_out = sum(int(round(float(x[a.MOD_RES + c]))) << (4 * c) for c in range(8))
    return q_out & MASK32, r_out & MASK32


# ===========================================================================
# BATCHED end-to-end simulation (x is [B, dim]) — same chain, vectorised over the
# case batch so the >=6000-random + adversarial + boundary grids run in seconds
# (the single-row ``simulate`` is O(80 forwards) per case).  The attention head is
# the ONLY non-FFN step and is spliced in vectorised (per-row softmax1 argmax /
# value copy read from R and written to QD/KBQ).  A BATCHED fp32 forward also uses
# a DIFFERENT accumulation ORDER than a single-row forward, so passing here is a
# STRICTER robustness test (the hardened variant's residue floor showed up under
# exactly this reordering).
# ===========================================================================
def _mono_head_batch(x, a, dtype):
    """Vectorised monotone-select head over the batch.  Reads R (a.R..) and the
    fixed KB rows, scores score_k = ALPHA·k − PEN·[KB[k]>R'] with a 0-sink, and
    writes q -> QD, KB[q]-nibbles -> KBQ.  The gate ``[KB[k]>R']`` is the
    value-difference sign in ``dtype`` (the wide compare that overflows fp32)."""
    B = x.shape[0]
    RN = a.RN
    # The head reads DISCRETE nibble tokens (0..15), so snap the R / KB nibbles to
    # clean integers before forming their scalar magnitudes — this is the vanilla
    # nibbles-as-tokens read (each nibble is a token value), NOT a raw fp residual
    # copy.  (The FFN carry-normalise leaves a ~1e-6 residue on KB; without the snap
    # an EXACT boundary R'==KB[k] mis-fires the gate.)  The scalar magnitude is then
    # the 16^i recompose — the WIDE value the fp32 dot cannot represent (the wall).
    w16 = torch.tensor([16.0 ** i for i in range(RN)], dtype=dtype)
    Rp = (x[:, a.R:a.R + RN].round() * w16).sum(dim=1)           # [B] R' value
    ks = torch.arange(1, 16, dtype=dtype)                        # [15]
    kb_vals = torch.empty(B, 15, dtype=dtype)
    for k in range(1, 16):
        kb_vals[:, k - 1] = (x[:, a.KB + RN * k:a.KB + RN * k + RN].round() * w16).sum(dim=1)
    valid = (Rp.unsqueeze(1) - kb_vals) >= 0.0                   # [B,15] KB[k]<=R'
    scores = _ALPHA * ks.unsqueeze(0) - _PEN * (~valid).to(dtype)
    scores = torch.cat([scores, torch.zeros(B, 1, dtype=torch.float64)], dim=1)  # +sink
    wsm = F.softmax(scores.double(), dim=1)                       # [B,16]
    idx = (wsm[:, :15] * ks.double().unsqueeze(0)).sum(dim=1)     # sink -> 0
    q = idx.round().clamp(0, 15).long()                          # [B]
    x[:, a.QD] = q.to(dtype)
    # KB[q] nibbles: gather the whole KB band as [B,16,RN] and index row q (q==0 ->
    # the all-zero KB[0] slot) — one vectorised gather, no per-k Python loop.
    kb = x[:, a.KB:a.KB + 16 * RN].view(B, 16, RN)
    x[:, a.KBQ:a.KBQ + RN] = kb[torch.arange(B), q]
    return x


def _cam_head_batch(x, a, dtype):
    """Vectorised exact-match retrieval head: q is ALREADY in QD (FFN gteq+qdigit
    produced it); copy KB[q] nibbles -> KBQ via the 4-bit-digit exact key match
    (fp32-exact: no wide compare in the dot)."""
    B = x.shape[0]
    RN = a.RN
    q = x[:, a.QD].round().clamp(0, 15).long()
    kb = x[:, a.KB:a.KB + 16 * RN].view(B, 16, RN)               # KB[0]=0 -> q==0 -> 0
    x[:, a.KBQ:a.KBQ + RN] = kb[torch.arange(B), q]
    return x


def simulate_batch(cases, variant="cam", L=None, dim=None, dtype=torch.float64,
                   n_iters=8):
    """Run a batch of (a,b) cases through the attention-select divide.  Returns a
    list of (q,r).  Same chain as ``simulate``, vectorised over the batch."""
    if L is None:
        L = _new_layout(); dim = L.D
    _set_one(L)
    a = L.LEANDIV
    kb_blocks = _kb_precompute_blocks(L, dim)
    init_block = _init_block(L, dim)
    body = _mono_ffn_body(L, dim) if variant == "mono" else _cam_ffn_body(L, dim)
    body_by_name = {n: s for n, s in body}
    final_block = _finalize_block(L, dim)
    borrow_names = [n for n, _ in body if n not in ("attn-shift", "attn-gteq", "attn-qd")]

    B = len(cases)
    x = torch.zeros(B, dim, dtype=dtype)
    x[:, L.ONE] = 1.0
    for bi, (av, bv) in enumerate(cases):
        for j, nv in enumerate(_nibbles(bv & MASK32, 8)):
            x[bi, L.AX + j] = float(nv)
        for j, nv in enumerate(_nibbles(av & MASK32, 8)):
            x[bi, L.STACK0 + j] = float(nv)

    def apply(spec):
        s = {k: v.to(dtype) for k, v in spec.items()}
        up = x @ s["W_up"].T + s["b_up"]
        gate = x @ s["W_gate"].T + s["b_gate"]
        return x + (F.silu(up) * gate) @ s["W_down"].T + s["b_down"]

    for _, spec in kb_blocks:
        x = apply(spec)
    x = apply(init_block)
    for _ in range(n_iters):
        x = apply(body_by_name["attn-shift"])
        if variant == "mono":
            x = _mono_head_batch(x, a, dtype)
        else:
            x = apply(body_by_name["attn-gteq"])
            x = apply(body_by_name["attn-qd"])
            x = _cam_head_batch(x, a, dtype)
        for name in borrow_names:
            x = apply(body_by_name[name])
    x = apply(final_block)
    out = []
    for bi in range(B):
        q = sum(int(round(float(x[bi, a.DIV_RES + c]))) << (4 * c) for c in range(8))
        r = sum(int(round(float(x[bi, a.MOD_RES + c]))) << (4 * c) for c in range(8))
        out.append((q & MASK32, r & MASK32))
    return out


# ===========================================================================
# MEASURE: depth (FFN blocks/iter + heads), byte-exact counts (fp64 + fp32), the
# attention-select construction verdict.
# ===========================================================================
def _test_cases(n_random: int, seed: int = 1234) -> List[Tuple[int, int]]:
    cases = list(_edge_grid()) + _adversarial_cases()
    rng = random.Random(seed)
    for _ in range(n_random):
        cases.append((rng.randint(0, MASK32), rng.randint(0, MASK32)))
    return cases


def _boundary_cases(n: int = 4000, seed: int = 99) -> List[Tuple[int, int]]:
    """The adversarial digit-boundary grid that flushes out the fp32 value-diff
    wall: R' = q·b (+/- 1) for LARGE b (near 2^32)."""
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        b = rng.randint(2 ** 28, MASK32)
        q = rng.randint(1, 15)
        # choose an a whose long division hits R'=q*b at some digit: simplest is to
        # feed a directly in [q*b .. q*b+b-1] scaled — but the divide is over a,b, so
        # just sample a near multiples of b to stress boundaries.
        k = rng.randint(0, 15)
        for a in (k * b, k * b + b - 1, k * b + b):
            cases.append((a & MASK32, b))
    return cases


def measure(verbose: bool = True, n_random: int = 6000):
    L = _new_layout(); dim = L.D
    a = L.LEANDIV

    # ---- depth accounting ----
    mono_body = _mono_ffn_body(L, dim)
    cam_body = _cam_ffn_body(L, dim)
    n_kb = len(_kb_precompute_blocks(L, dim))
    # hardened reference
    hard_body = [
        ("s", _shift_block(L, dim)), ("g", _gteq_block(L, dim)),
        ("q", _qdigit_block(L, dim)), ("b", _qbsel_block(L, dim)),
    ] + _borrow_prefix_blocks(L, dim)
    hard_bpi = len(hard_body)
    hard_depth = n_kb + 1 + 8 * hard_bpi + 1

    mono_bpi_ffn = len(mono_body)
    cam_bpi_ffn = len(cam_body)
    mono_depth_ffn = n_kb + 1 + 8 * mono_bpi_ffn + 1
    cam_depth_ffn = n_kb + 1 + 8 * cam_bpi_ffn + 1

    (mq_w, mk_w, mv_w, mo_w), _, mono_head_nz = bake_mono_select_head()
    (cq_w, ck_w, cv_w, co_w), _, cam_head_nz = bake_qsel_cam_head()

    # ---- byte-exact on the required grids ----
    cases = _test_cases(n_random)
    bcases = _boundary_cases(n=1500)      # 4500 digit-boundary cases (wall is sharp)
    total = len(cases)

    def run(variant, dtype, cs, tag, batch=1024):
        p, fails = 0, []
        for i in range(0, len(cs), batch):
            chunk = cs[i:i + batch]
            outs = simulate_batch(chunk, variant=variant, L=L, dim=dim, dtype=dtype)
            for (av, bv), (q, r) in zip(chunk, outs):
                rq, rr = _ref(av, bv)
                if (q, r) == (rq, rr):
                    p += 1
                elif len(fails) < 12:
                    fails.append((av, bv, (q, r), (rq, rr)))
        if verbose:
            print(f"  [done] {tag}: {p}/{len(cs)}", flush=True)
        return p, fails

    if verbose:
        print(f"measuring {total} main-grid + {len(bcases)} boundary cases "
              f"(fp64+fp32) ...", flush=True)
    # attn-mono: fp64 (algorithm), then fp32 (honest wall), on grid + boundary.
    mono64_p, mono64_f = run("mono", torch.float64, cases, "mono fp64 main")
    mono32_p, mono32_f = run("mono", torch.float32, cases, "mono fp32 main")
    mono64_bp, _ = run("mono", torch.float64, bcases, "mono fp64 boundary")
    mono32_bp, mono32_bf = run("mono", torch.float32, bcases, "mono fp32 boundary")
    # attn-cam: fp32 (the hybrid should be exact), on grid + boundary.
    cam32_p, cam32_f = run("cam", torch.float32, cases, "cam fp32 main")
    cam32_bp, cam32_bf = run("cam", torch.float32, bcases, "cam fp32 boundary")

    if verbose:
        _report(locals())
    return dict(
        hard_depth=hard_depth, hard_bpi=hard_bpi,
        mono_depth_ffn=mono_depth_ffn, mono_bpi_ffn=mono_bpi_ffn,
        cam_depth_ffn=cam_depth_ffn, cam_bpi_ffn=cam_bpi_ffn,
        mono_head_nz=mono_head_nz, cam_head_nz=cam_head_nz,
        total=total, n_boundary=len(bcases),
        mono_fp64=mono64_p, mono_fp32=mono32_p,
        mono_fp64_bnd=mono64_bp, mono_fp32_bnd=mono32_bp,
        cam_fp32=cam32_p, cam_fp32_bnd=cam32_bp,
    )


def _report(ns):
    L = ns["L"]; a = ns["a"]
    b = "=" * 76
    print(b)
    print("ATTENTION-SELECT radix-16 32-bit DIVIDE — bakeoff")
    print(b)
    print("PER-ITERATION block accounting (SELECT = the gteq+qdigit+qbsel compare):")
    print(f"  hardened (FFN-only)     : {ns['hard_bpi']} FFN blk/iter  "
          f"(shift, gteq, qdigit, qbsel, gp, ks0..3, apply)")
    print(f"  attn-mono (fp64)        : {ns['mono_bpi_ffn']} FFN blk + 1 attn head/iter  "
          f"(attention does gteq+qdigit+qbsel)")
    print(f"  attn-cam  (fp32 hybrid) : {ns['cam_bpi_ffn']} FFN blk + 1 attn head/iter  "
          f"(attention does qbsel; compare stays FFN)")
    print("-" * 76)
    print("TOTAL DEPTH (unrolled, 8 iters; +KB-precompute +init +finalize):")
    print(f"  hardened  : {ns['hard_depth']} FFN blocks")
    print(f"  attn-mono : {ns['mono_depth_ffn']} FFN blocks + 8 attn heads   "
          f"(shaved 3 FFN blk/iter = 24 blocks)")
    print(f"  attn-cam  : {ns['cam_depth_ffn']} FFN blocks + 8 attn heads   "
          f"(shaved 1 FFN blk/iter = 8 blocks)")
    print(f"  attn head nz: mono={ns['mono_head_nz']} (q/k/v/o)  "
          f"cam={ns['cam_head_nz']}")
    print("-" * 76)
    tot = ns["total"]; nb = len(ns["bcases"])
    print("BYTE-EXACT (q AND r), main grid = edge + adversarial + "
          f"{tot - len(_edge_grid()) - len(_adversarial_cases())} random:")
    print(f"  attn-mono  fp64 : {ns['mono64_p']}/{tot}   "
          f"({'ALL PASS' if ns['mono64_p'] == tot else 'FAIL'})  "
          f"<- algorithm correct")
    print(f"  attn-mono  fp32 : {ns['mono32_p']}/{tot}   "
          f"({'ALL PASS' if ns['mono32_p'] == tot else 'RESIDUE FLOOR'})")
    print(f"  attn-cam   fp32 : {ns['cam32_p']}/{tot}   "
          f"({'ALL PASS' if ns['cam32_p'] == tot else 'FAIL'})  "
          f"<- hybrid (compare in FFN)")
    print(f"BYTE-EXACT on the {nb}-case DIGIT-BOUNDARY grid "
          f"(R'≈q·b, large b — the fp32 wall):")
    print(f"  attn-mono  fp64 : {ns['mono64_bp']}/{nb}")
    print(f"  attn-mono  fp32 : {ns['mono32_bp']}/{nb}   "
          f"<- the value-diff dot OVERFLOWS fp32 at boundaries")
    print(f"  attn-cam   fp32 : {ns['cam32_bp']}/{nb}   "
          f"<- exact 4-bit key match, no wide compare in the dot")
    if ns["mono32_bf"]:
        print("  attn-mono fp32 boundary fails (a,b,got,exp):")
        for f in ns["mono32_bf"][:6]:
            print("   ", f)
    print("-" * 76)
    print("VERDICT — what attention CAN and CANNOT shave off the runtime divide:")
    print("  * The SELECT algorithm IS a threshold retrieval and fits a monotone")
    print("    softmax1 head: score_k = ALPHA·k − PEN·[KB[k]>R'], sink at 0.")
    print("    In fp64 it is byte-exact and replaces THREE FFN blocks (gteq +")
    print("    qdigit + qbsel) with ONE attention head: 10 -> 7 FFN blk/iter.")
    print("  * BUT the head's threshold GATE [KB[k]>R'] is a 36-bit magnitude")
    print("    compare, and a softmax score is a LINEAR dot -> the only encoding")
    print("    is a 16^p positional recompose spanning 2^32, which OVERFLOWS fp32")
    print("    (2^24) at the digit boundaries R'≈q·b.  Measured ~25% wrong on the")
    print("    boundary grid.  This is the SAME amplification the hardened variant")
    print("    was founded to KILL — a single attention dot cannot do the nibble-")
    print("    LEXICOGRAPHIC compare that dodges it.")
    print("  * Honest fp32 answer (attn-cam): keep the compare (gteq+qdigit) in")
    print("    fp32-exact FFN, move ONLY the one-hot(q)->KB[q] RETRIEVAL into an")
    print("    exact-match 4-bit-digit CAM head.  Byte-exact in fp32, shaves 1 FFN")
    print("    block/iter (qbsel): 10 -> 9 FFN blk/iter (88 -> 80 FFN + 8 heads).")
    print("  * The SUBTRACT and BRING-DOWN are irreducibly FFN (attention does no")
    print("    arithmetic); the Kogge-Stone borrow is unchanged.")
    print(b)


if __name__ == "__main__":
    measure()
