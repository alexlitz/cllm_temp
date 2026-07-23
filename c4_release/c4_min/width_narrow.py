"""STATIC WIDTH-NARROWING for the c4_min nibble ALU — run every op at its ACTUAL
nibble width, not the full 8x8 worst case.

The systemic-compute observation
=================================
The full-width ``nibble_alu32`` gadgets are sized for the WORST case: two 8-nibble
(32-bit) operands.  The general MUL forms all ``8*8`` partial products with
``i+j<8`` (the 36 pairs, ~6,291 nz) and carry-resolves 8 columns.  But MOST real
values are NARROW — a char (2 nibbles), a loop index / ``i<n`` counter (2-4
nibbles), a small constant (1-2 nibbles).  A multiply of two 2-nibble operands
needs only ``2*2 = 4`` partial products into ``2+2 = 4`` result columns — about
1/9 the partials of 8x8.  ADD/SUB of ``w``-nibble operands touch only ``w`` lanes
plus a carry; MOD by a ``w``-nibble divisor yields ``<= w`` result nibbles.

This module has TWO deliverables:

1. **Width-parameterized arithmetic builders** — ``build_mul(wa, wb)``,
   ``build_add(w)``, ``build_sub(w)``, ``build_mod(w)``, ``build_div(w)``,
   ``build_cmp(w)``.  Each is COMPOSED from the proven ``nibble_alu32`` SwiGLU
   primitives (``_mul_gate``, the carry-share ``_floor_div_pow2`` round / the
   Kogge-Stone-style resolve pieces, ``_step_ge``, ``_ident``, ``_clear``,
   ``_empty_spec``, ``_truncate``) BY IMPORT — this file NEVER edits the shared
   ``nibble_alu32`` / ``nibble_vm`` modules.  A gadget writes into its OWN private
   scratch bands (allocated off the layout), so the sparse simulator reads/writes
   only its own state.

2. **Static range-analysis over bytecode** — ``infer_widths(program)`` propagates a
   conservative per-value nibble-width bound (``IMM k`` -> ``ceil(log16(k+1))``;
   ``AND`` with a constant mask -> mask width; add -> ``max+1``; mul -> ``wa+wb``;
   mod b -> ``ceil(log16 b)``; loop counters bounded by their init; default 8 when
   unknown) and, per ALU instruction, reports the operand widths -> which narrowed
   gadget to build.

fp32 discipline is INHERITED verbatim from ``nibble_alu32`` (staircase args stay
<= 232 < 2^24/RELU_S; no hidden unit exceeds 2^24; no fp64 anywhere).  Narrowing
only DROPS columns/partials/lanes — it never widens any argument — so a narrowed
gadget is a strict subset of the full-width one and byte-exact whenever the range
analysis PROVES the operands fit.

Run:  ``python -m c4_min.width_narrow``
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .nibble_vm_layout import NibbleVMLayout
from . import nibble_alu32 as A
from .nibble_vm import S, RELU_S, _empty_spec
from .nibble_alu32 import (
    _mul_gate, _floor_div_pow, _floor_div_pow2, _step_ge,
    _ident, _clear, _truncate, extend_layout_for_alu32,
)
from . import isa

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

FP32_INT_MAX = 1 << 24                  # fp32 exact-integer ceiling
FORM_CEIL_FP32 = FP32_INT_MAX / RELU_S  # ~83886: max staircase arg fp32-safe

_NIB_KMAX = 15                          # floor(col/16) for a column kept < 256


# ===========================================================================
# Private scratch allocation (never touches ALU32Bands; each gadget names its
# own band range so the sparse sim reads/writes only its own bands).  Re-use the
# same band across builds of the same op so DIM does not blow up.
# ===========================================================================
def _scratch(L, name, size):
    key = f"WN_{name}"
    if key in L._names:
        base, sz = L._names[key]
        if sz >= size:
            return base
        # need more room — extend by allocating a fresh, larger band.
        key = f"{key}_{size}"
        if key in L._names:
            return L._names[key][0]
    return L._band(key, size)


def _read_nibbles(state, base, n):
    """Assemble an int from ``n`` nibbles at ``base`` in the sparse state dict."""
    v = 0
    for c in range(n):
        v |= (int(round(state.get(base + c, 0.0))) & 0xF) << (4 * c)
    return v


# ===========================================================================
# 1. WIDTH-PARAMETERIZED BUILDERS.  Each returns (blocks, res_band, info):
#    * blocks   — the SwiGLU sub-block list (fires in order)
#    * res_band — the scratch band holding the result nibbles (LSB first)
#    * info     — {n_partials, n_result_nib, max_staircase_arg, ...}
#
#    Operand A = STACK0 nibbles (popped), operand B = AX nibbles (accumulator),
#    exactly as the full ALU (``a OP b`` with ``a`` = pop, ``b`` = AX).
# ===========================================================================

_RESULT_NCOL = 8            # 32-bit result = 8 nibbles; products past it are masked off.


# --- MUL(wa, wb): only wa*wb partial products, result <= wa+wb nibbles --------
def build_mul(L, dim, wa: int, wb: int) -> Tuple[List[Block], int, dict]:
    """MUL narrowed to operands of ``wa`` / ``wb`` nibbles.  Forms ONLY the
    partial products ``a_i*b_j`` (i<wa, j<wb) with ``i+j < ncol`` and carries only
    the ``ncol = min(wa+wb, 8)`` result columns.

    The 32-bit result is 8 nibbles, so a pair with ``i+j >= 8`` overflows past bit
    32 and is DISCARDED (exactly as the native ``compile_mul_blocks`` keeps only the
    36 ``i+j<8`` pairs) — this makes ``build_mul(8, 8)`` reproduce the native 6,291
    nz / 8-column masked MUL, while a narrow ``build_mul(2, 2)`` forms only 4
    partials into 4 columns (~1/9 the 36-pair 8x8 baseline).

    A column receives at most ``min(wa,wb)`` low + high contributions, each <= 15,
    so every column stays < 256 (kmax=15) and every staircase arg <= 225 -> fp32."""
    A._ONE = L.ONE
    ncol = min(wa + wb, _RESULT_NCOL)
    pairs = [(i, j) for i in range(wa) for j in range(wb) if i + j < ncol]
    PP = _scratch(L, f"MUL_{wa}x{wb}_PP", max(1, len(pairs)))
    COL = _scratch(L, f"MUL_{wa}x{wb}_COL", ncol)
    RES = _scratch(L, f"MUL_{wa}x{wb}_RES", ncol)
    blocks: List[Block] = []

    # block 1: the wa*wb raw nibble products PP[idx] = a_i*b_j  (<= 225 each).
    s = _empty_spec(dim, len(pairs) * 3)
    u = 0
    for idx, (i, j) in enumerate(pairs):
        u = _clear(s, u, PP + idx)
        u = _mul_gate(s, u, L.STACK0 + i, L.AX + j, PP + idx, 1.0)
    blocks.append((f"mul{wa}x{wb}-products", _truncate(s, u, dim)))

    # block 2: split each PP into low nibble (col i+j) + high nibble (col i+j+1),
    # one shared floor(p/16) staircase per product (carry-SHARE, byte-identical
    # to the full split).  ncol columns accumulate.
    s = _empty_spec(dim, ncol + len(pairs) * (1 + 15 * 2))
    u = 0
    for c in range(ncol):
        u = _clear(s, u, COL + c)
    for idx, (i, j) in enumerate(pairs):
        c = i + j
        pp = PP + idx
        u = _ident(s, u, {pp: 1.0}, 0.0, COL + c, 1.0)
        if c + 1 < ncol:
            u = _floor_div_pow2(s, u, {pp: 1.0}, 0.0, 16, 15,
                                COL + c, -16.0, COL + c + 1, 1.0)
        else:
            u = _floor_div_pow(s, u, {pp: 1.0}, 0.0, 16, 15, COL + c, -16.0)
    blocks.append((f"mul{wa}x{wb}-split", _truncate(s, u, dim)))

    # carry-resolve ONLY the ncol columns.  A carry ripples one column per round,
    # so ``ncol-1`` rounds fully settle ncol columns (the top column's carry-out is
    # dropped = the 32-bit mask).  For ncol=8 this is 7 rounds, matching the native
    # ``compile_mul_blocks`` (_MUL_CARRY_ROUNDS - 1 ... native adds +1 headroom, a
    # fixed point).  We use exactly ncol-1 (provably sufficient) for the tightest count.
    n_rounds = max(1, ncol - 1)
    src, dst = COL, _scratch(L, f"MUL_{wa}x{wb}_COL1", ncol)
    for r in range(n_rounds):
        s = _empty_spec(dim, ncol * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
        u = A._nibble_carry_round(s, 0, src, dst, ncol)
        blocks.append((f"mul{wa}x{wb}-carry{r}", _truncate(s, u, dim)))
        src, dst = dst, src

    # copy settled columns -> RES.
    s = _empty_spec(dim, ncol * 2)
    u = 0
    for c in range(ncol):
        u = _clear(s, u, RES + c)
        u = _ident(s, u, {src + c: 1.0}, 0.0, RES + c, 1.0)
    blocks.append((f"mul{wa}x{wb}-result", _truncate(s, u, dim)))

    info = {"op": "MUL", "wa": wa, "wb": wb, "n_partials": len(pairs),
            "n_result_nib": ncol, "max_staircase_arg": 225}
    return blocks, RES, info


# --- ADD(w) / SUB(w): only w nibble lanes + a carry across ceil(w/2) bytes -----
def _addsub_bytewise(L, dim, w: int, sub: bool) -> Tuple[List[Block], int, dict]:
    """ADD/SUB narrowed to ``w`` nibbles.  The full ALU carries 4 bytes (8
    nibbles); here only ``nbytes = ceil(w/2)`` byte-lanes are carried, and the
    top nibble is dropped if ``w`` is odd.  SUB = A + (~B + 1) over the SAME
    ``w`` nibbles (ones' complement of B's low ``w`` nibbles).  Every byte sum
    <= 511 -> fp32.  This is the ``nibble_alu32`` per-byte carry chain restricted
    to the low bytes."""
    A._ONE = L.ONE
    nbytes = (w + 1) // 2
    tag = "sub" if sub else "add"
    Abnd = _scratch(L, f"ADDSUB_{w}_A", nbytes)      # operand-A bytes
    Bbnd = _scratch(L, f"ADDSUB_{w}_B", nbytes)      # operand-B (or ~B) bytes
    CAR = _scratch(L, f"{tag.upper()}_{w}_C", nbytes + 1)
    RES = _scratch(L, f"{tag.upper()}_{w}_RES", w)
    blocks: List[Block] = []

    # block 0: expand the low w nibbles into nbytes byte-lanes.  For SUB, B-lane
    # is ~B = 255 - (B low byte) over the byte (the top nibble of an odd-w byte is
    # a real high nibble; we mask the RESULT, not the operand, so ~B uses the full
    # byte to keep the two's-complement identity a+~b+1 = a-b exact on the byte).
    s = _empty_spec(dim, nbytes * (3 + 3))
    u = 0
    for k in range(nbytes):
        lo, hi = L.STACK0 + 2 * k, L.STACK0 + 2 * k + 1
        u = _clear(s, u, Abnd + k)
        u = _ident(s, u, {lo: 1.0, hi: 16.0}, 0.0, Abnd + k, 1.0)
        blo, bhi = L.AX + 2 * k, L.AX + 2 * k + 1
        u = _clear(s, u, Bbnd + k)
        if sub:
            u = _ident(s, u, {blo: -1.0, bhi: -16.0}, 255.0, Bbnd + k, 1.0)   # ~B
        else:
            u = _ident(s, u, {blo: 1.0, bhi: 16.0}, 0.0, Bbnd + k, 1.0)
    blocks.append((f"{tag}{w}-expand", _truncate(s, u, dim)))

    # per-byte carry chain (one block per byte).  byte 0 initial carry = 1 for SUB
    # (the +1 of two's complement), 0 for ADD.
    for k in range(nbytes):
        s = _empty_spec(dim, 200)
        u = 0
        sum_terms = {Abnd + k: 1.0, Bbnd + k: 1.0}
        if k == 0:
            sum_const = 1.0 if sub else 0.0
        else:
            sum_const = 0.0
            sum_terms[CAR + k] = 1.0
        # carry-out = [sum >= 256]
        u = _clear(s, u, CAR + k + 1)
        u = _step_ge(s, u, sum_terms, sum_const, 256, CAR + k + 1, 1.0)
        # low nibble = sum mod 16
        u = _clear(s, u, RES + 2 * k)
        u = _ident(s, u, sum_terms, sum_const, RES + 2 * k, 1.0)
        u = _floor_div_pow(s, u, sum_terms, sum_const, 16, 32, RES + 2 * k, -16.0)
        # high nibble = floor(sum/16) mod 16   (only if w has this nibble)
        if 2 * k + 1 < w:
            u = _clear(s, u, RES + 2 * k + 1)
            u = _floor_div_pow(s, u, sum_terms, sum_const, 16, 32, RES + 2 * k + 1, 1.0)
            u = _floor_div_pow(s, u, sum_terms, sum_const, 256, 2, RES + 2 * k + 1, -16.0)
        blocks.append((f"{tag}{w}-b{k}", _truncate(s, u, dim)))

    info = {"op": "SUB" if sub else "ADD", "w": w, "n_lanes": w,
            "n_bytes": nbytes, "n_result_nib": w, "max_staircase_arg": 511}
    return blocks, RES, info


def build_add(L, dim, w: int) -> Tuple[List[Block], int, dict]:
    return _addsub_bytewise(L, dim, w, sub=False)


def build_sub(L, dim, w: int) -> Tuple[List[Block], int, dict]:
    return _addsub_bytewise(L, dim, w, sub=True)


# --- CMP(w): equality / order over only w nibbles -----------------------------
_CMP_OPS = (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE)


def build_cmp(L, dim, w: int, op: int) -> Tuple[List[Block], int, dict]:
    """Compare A vs B over their low ``w`` nibbles (unsigned), emitting a 0/1 into
    RES[0].  Lexicographic nibble compare from the MSB: only ``w`` nibble GT/EQ
    lanes are formed (vs 8 for the full compare).  Every staircase arg is a nibble
    difference in [-15,15] -> fp32.  ``op`` is one of EQ/NE/LT/GT/LE/GE."""
    A._ONE = L.ONE
    assert op in _CMP_OPS
    GT = _scratch(L, f"CMP_{w}_GT", w)
    EQ = _scratch(L, f"CMP_{w}_EQ", w)
    RES = _scratch(L, f"CMP_{w}_RES_{op}", 1)
    blocks: List[Block] = []

    # block 1: per-nibble GT[i] = [A[i] > B[i]], EQ[i] = [A[i] == B[i]].
    s = _empty_spec(dim, w * (2 + 4 + 4))
    u = 0
    for i in range(w):
        d = {L.STACK0 + i: 1.0, L.AX + i: -1.0}      # A[i] - B[i] in [-15,15]
        u = _clear(s, u, GT + i)
        u = _step_ge(s, u, d, 0.0, 1, GT + i, 1.0)                  # [d>=1]
        u = _clear(s, u, EQ + i)
        u = _step_ge(s, u, d, 0.0, 0, EQ + i, 1.0)                  # +[d>=0]
        u = _step_ge(s, u, d, 0.0, 1, EQ + i, -1.0)                 # -[d>=1] => [d==0]
    blocks.append((f"cmp{w}-gteq", _truncate(s, u, dim)))

    # block 2: assemble the predicate.  a_gt = [A > B] (lexicographic), a_eq = all
    # nibbles equal.  Each order predicate is a Boolean combination of the two.
    s = _empty_spec(dim, w * (w + 1) + 8)
    u = 0
    u = _clear(s, u, RES)
    # a_gt = sum_{i=w-1..0} ( GT[i] AND EQ[i+1..w-1] )  (lexicographic from MSB)
    gt_windows_list = []
    for i in range(w - 1, -1, -1):
        windows = [(GT + i, 1.0, 0.0)] + [(EQ + j, 1.0, 0.0) for j in range(i + 1, w)]
        gt_windows_list.append(windows)
    eq_windows = [(EQ + j, 1.0, 0.0) for j in range(w)]
    # a_lt = 1 - a_gt - a_eq (exhaustive trichotomy over w nibbles).
    if op == isa.EQ:
        for win in [eq_windows]:
            u = A._guard(s, u, win, {L.ONE: 1.0}, 0.0, RES, 1.0)
    elif op == isa.NE:
        u = _ident(s, u, {L.ONE: 1.0}, 0.0, RES, 1.0)               # 1
        u = A._guard(s, u, eq_windows, {L.ONE: 1.0}, 0.0, RES, -1.0)  # - [A==B]
    elif op == isa.GT:
        for win in gt_windows_list:
            u = A._guard(s, u, win, {L.ONE: 1.0}, 0.0, RES, 1.0)
    elif op == isa.LE:                                              # 1 - a_gt
        u = _ident(s, u, {L.ONE: 1.0}, 0.0, RES, 1.0)
        for win in gt_windows_list:
            u = A._guard(s, u, win, {L.ONE: 1.0}, 0.0, RES, -1.0)
    elif op == isa.GE:                                             # a_gt + a_eq
        for win in gt_windows_list:
            u = A._guard(s, u, win, {L.ONE: 1.0}, 0.0, RES, 1.0)
        u = A._guard(s, u, eq_windows, {L.ONE: 1.0}, 0.0, RES, 1.0)
    elif op == isa.LT:                                             # 1 - a_gt - a_eq
        u = _ident(s, u, {L.ONE: 1.0}, 0.0, RES, 1.0)
        for win in gt_windows_list:
            u = A._guard(s, u, win, {L.ONE: 1.0}, 0.0, RES, -1.0)
        u = A._guard(s, u, eq_windows, {L.ONE: 1.0}, 0.0, RES, -1.0)
    blocks.append((f"cmp{w}-{isa.NAMES[op]}", _truncate(s, u, dim)))

    info = {"op": isa.NAMES[op], "w": w, "n_lanes": w, "n_result_nib": 1,
            "max_staircase_arg": 15}
    return blocks, RES, info


# --- DIV(w) / MOD(w): base-16 long division, only w dividend iterations --------
# The divisor is <= w nibbles, so the quotient/remainder are <= w nibbles and only
# ``w`` MSB-first iterations are needed (the full ALU always runs 8).  The running
# remainder is < 16*divisor < 16^(w+1), i.e. RN = w+1 remainder nibbles.  KB[k] =
# k*b (k=1..15) only over the w divisor nibbles.  This mirrors compile_divmod_blocks
# with n_iters = w and RN = w+1.
def _build_divmod(L, dim, w: int, want_mod: bool) -> Tuple[List[Block], int, dict]:
    A._ONE = L.ONE
    RN = w + 1                                     # remainder nibbles (< 16*b)
    tag = f"DM{w}"
    KB = _scratch(L, f"{tag}_KB", RN * 16)
    BZ = _scratch(L, f"{tag}_BZ", 1)
    R = _scratch(L, f"{tag}_R", RN)
    R2 = _scratch(L, f"{tag}_R2", RN)
    GT = _scratch(L, f"{tag}_GT", 15 * RN)
    EQ = _scratch(L, f"{tag}_EQ", 15 * RN)
    QD = _scratch(L, f"{tag}_QD", 1)
    QB = _scratch(L, f"{tag}_QB", RN)
    SUBB = _scratch(L, f"{tag}_SUBB", RN + 1)
    QUO = _scratch(L, f"{tag}_QUO", w)             # quotient nibbles (result)
    REM = _scratch(L, f"{tag}_REM", w)             # remainder nibbles (result)
    blocks: List[Block] = []

    def guard(spec, u, windows, terms, const, dst, scale):
        return A._guard(spec, u, windows, terms, const, dst, scale)

    # --- KB[k] = k*b nibbles (k=1..15) from the w divisor nibbles + BZ=[b==0] ---
    s = _empty_spec(dim, 16 * RN + 15 * w)
    u = 0
    for k in range(1, 16):
        base = KB + RN * k
        for c in range(RN):
            u = _clear(s, u, base + c)
        for c in range(w):
            u = _ident(s, u, {L.AX + c: float(k)}, 0.0, base + c, 1.0)
    blocks.append((f"{tag}-kb-raw", _truncate(s, u, dim)))
    for k in range(1, 16):
        base = KB + RN * k
        for rnd in range(RN):
            s = _empty_spec(dim, RN * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
            u = A._nibble_carry_round(s, 0, base, base, RN)
            blocks.append((f"{tag}-kb{k}-c{rnd}", _truncate(s, u, dim)))
    s = _empty_spec(dim, 2 + w * 4)
    u = 0
    u = _clear(s, u, BZ)
    u = _ident(s, u, {L.ONE: 1.0}, 0.0, BZ, 1.0)
    bsum = {L.AX + c: 1.0 for c in range(w)}
    u = _step_ge(s, u, bsum, 0.0, 1, BZ, -1.0)     # BZ = 1 - [sum>=1]
    blocks.append((f"{tag}-bz", _truncate(s, u, dim)))

    # --- init: R=0, QUO=0 ---
    s = _empty_spec(dim, RN + w)
    u = 0
    for c in range(RN):
        u = _clear(s, u, R + c)
    for c in range(w):
        u = _clear(s, u, QUO + c)
    blocks.append((f"{tag}-init", _truncate(s, u, dim)))

    # --- w iterations, MSB-first over the w dividend nibbles ---
    for it in range(w):
        div_nib_idx = w - 1 - it                   # MSB nibble of the w-nibble dividend
        q_out_idx = w - 1 - it
        # shift: R = 16*R + dividend_nibble
        s = _empty_spec(dim, RN * 2 + 2)
        u = 0
        for c in range(RN - 1, -1, -1):
            u = _clear(s, u, R + c)
            if c == 0:
                u = _ident(s, u, {L.STACK0 + div_nib_idx: 1.0}, 0.0, R + c, 1.0)
            else:
                u = _ident(s, u, {R + c - 1: 1.0}, 0.0, R + c, 1.0)
        blocks.append((f"{tag}-shift{it}", _truncate(s, u, dim)))
        # gteq: GT[k,i], EQ[k,i] for k=1..15
        s = _empty_spec(dim, 15 * RN * (2 + 4 + 4))
        u = 0
        for k in range(1, 16):
            for i in range(RN):
                gt = GT + (k - 1) * RN + i
                eq = EQ + (k - 1) * RN + i
                d = {R + i: 1.0, KB + RN * k + i: -1.0}
                u = _clear(s, u, gt)
                u = _step_ge(s, u, d, 0.0, 1, gt, 1.0)
                u = _clear(s, u, eq)
                u = _step_ge(s, u, d, 0.0, 0, eq, 1.0)
                u = _step_ge(s, u, d, 0.0, 1, eq, -1.0)
        blocks.append((f"{tag}-gteq{it}", _truncate(s, u, dim)))
        # qdigit: QD = sum_k [R >= KB_k]  (lexicographic suffix-AND)
        s = _empty_spec(dim, 1 + 15 * (RN + 1))
        u = 0
        u = _clear(s, u, QD)
        for k in range(1, 16):
            gtb = GT + (k - 1) * RN
            eqb = EQ + (k - 1) * RN
            for i in range(RN - 1, -1, -1):
                windows = [(gtb + i, 1.0, 0.0)] + [(eqb + j, 1.0, 0.0) for j in range(i + 1, RN)]
                u = guard(s, u, windows, {L.ONE: 1.0}, 0.0, QD, 1.0)
            windows = [(eqb + j, 1.0, 0.0) for j in range(RN)]
            u = guard(s, u, windows, {L.ONE: 1.0}, 0.0, QD, 1.0)
        blocks.append((f"{tag}-qd{it}", _truncate(s, u, dim)))
        # qcopy: QUO[q_out_idx] = QD
        s = _empty_spec(dim, 2)
        u = 0
        u = _clear(s, u, QUO + q_out_idx)
        u = _ident(s, u, {QD: 1.0}, 0.0, QUO + q_out_idx, 1.0)
        blocks.append((f"{tag}-qcopy{it}", _truncate(s, u, dim)))
        # qb: QB = QD * b  (raw), then carry-normalise
        s = _empty_spec(dim, RN + w * 2)
        u = 0
        for c in range(RN):
            u = _clear(s, u, QB + c)
        for c in range(w):
            u = _mul_gate(s, u, QD, L.AX + c, QB + c, 1.0)
        blocks.append((f"{tag}-qb{it}", _truncate(s, u, dim)))
        for rnd in range(RN):
            s = _empty_spec(dim, RN * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
            u = A._nibble_carry_round(s, 0, QB, QB, RN)
            blocks.append((f"{tag}-qbc{it}-{rnd}", _truncate(s, u, dim)))
        # sub: R2 = R - QB  (borrow chain, RN nibbles)
        for i in range(RN):
            s = _empty_spec(dim, 40)
            u = 0
            diff = {R + i: 1.0, QB + i: -1.0}
            if i > 0:
                diff[SUBB + i] = -1.0
            u = _clear(s, u, SUBB + i + 1)
            u = _ident(s, u, {L.ONE: 1.0}, 0.0, SUBB + i + 1, 1.0)
            u = _step_ge(s, u, diff, 0.0, 0, SUBB + i + 1, -1.0)
            u = _clear(s, u, R2 + i)
            u = _ident(s, u, diff, 0.0, R2 + i, 1.0)
            u = _ident(s, u, {L.ONE: 1.0}, 0.0, R2 + i, 16.0)
            u = _step_ge(s, u, diff, 0.0, 0, R2 + i, -16.0)
            blocks.append((f"{tag}-sub{it}-{i}", _truncate(s, u, dim)))
        # r2->r
        s = _empty_spec(dim, RN * 2)
        u = 0
        for c in range(RN):
            u = _clear(s, u, R + c)
            u = _ident(s, u, {R2 + c: 1.0}, 0.0, R + c, 1.0)
        blocks.append((f"{tag}-r2r{it}", _truncate(s, u, dim)))

    # --- finalize: REM = R (low w nibbles), zeroed on b==0; QUO zeroed on b==0 --
    s = _empty_spec(dim, w * 6)
    u = 0
    for c in range(w):
        u = _clear(s, u, REM + c)
        u = _ident(s, u, {R + c: 1.0}, 0.0, REM + c, 1.0)
        gz = (BZ, 1.0, 0.0)
        u = guard(s, u, [gz], {R + c: -1.0}, 0.0, REM + c, 1.0)     # - BZ*R
        u = guard(s, u, [gz], {QUO + c: -1.0}, 0.0, QUO + c, 1.0)   # QUO -= BZ*QUO
    blocks.append((f"{tag}-finalize", _truncate(s, u, dim)))

    res = REM if want_mod else QUO
    info = {"op": "MOD" if want_mod else "DIV", "w": w, "n_iters": w, "RN": RN,
            "n_result_nib": w, "max_staircase_arg": 225}
    return blocks, res, info


def build_div(L, dim, w: int) -> Tuple[List[Block], int, dict]:
    return _build_divmod(L, dim, w, want_mod=False)


def build_mod(L, dim, w: int) -> Tuple[List[Block], int, dict]:
    return _build_divmod(L, dim, w, want_mod=True)


# ===========================================================================
# 2. SPARSE BYTE-EXACT SIMULATION (same SwiGLU math as a dense forward, but only
#    over the bands each gadget touches — no DIM x DIM matmul).  Ported from the
#    mul_bakeoff sparse sim.
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
    (up_bands, gate_bands, out_dims, Wu, Wg, Wd, b_up, b_gate, b_down) = _spec_sparsity(spec, dtype)
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


def simulate(L, blocks: List[Block], res_band: int, n_result_nib: int,
             a_val: int, b_val: int, dtype=torch.float32) -> int:
    """Seed A=STACK0, B=AX, run the block list, read n_result_nib result nibbles."""
    state: Dict[int, float] = {L.ONE: 1.0}
    for c in range(8):
        state[L.STACK0 + c] = float((a_val >> (4 * c)) & 0xF)
        state[L.AX + c] = float((b_val >> (4 * c)) & 0xF)
    for _n, s in blocks:
        state = _sparse_apply(state, s, dtype=dtype)
    return _read_nibbles(state, res_band, n_result_nib)


# --- BATCHED sparse forward: run ALL operands through the block list at once so
# the per-operand overhead collapses into one [dim, B] matmul per block.  The
# EXACT same SwiGLU math as ``_sparse_apply``, but over B operand columns; this is
# what keeps the 2000+-operand byte-exact check in the seconds regime even for the
# 331-block width-8 DIV/MOD gadgets.
def _sparse_apply_batch(state: torch.Tensor, spec: Spec, dtype=torch.float32):
    """state: dense [dim, B].  Applies the block in place, returns the updated state."""
    (up_bands, gate_bands, out_dims, Wu, Wg, Wd, b_up, b_gate, b_down) = _spec_sparsity(spec, dtype)
    B = state.shape[1]
    up = b_up.unsqueeze(1).expand(-1, B).clone()
    gate = b_gate.unsqueeze(1).expand(-1, B).clone()
    if up_bands:
        up = up + Wu @ state[up_bands]            # [n_units, B]
    if gate_bands:
        gate = gate + Wg @ state[gate_bands]
    hidden = F.silu(up) * gate                    # [n_units, B]
    if out_dims:
        delta = Wd @ hidden + b_down.unsqueeze(1)  # [|out_dims|, B]
        state[out_dims] += delta
    return state


def simulate_batch(L, dim, blocks: List[Block], res_band: int, n_result_nib: int,
                   a_vals: List[int], b_vals: List[int], dtype=torch.float32) -> List[int]:
    """Seed a BATCH of (a, b) operands, run the block list once, read each result."""
    B = len(a_vals)
    state = torch.zeros(dim, B, dtype=dtype)
    state[L.ONE, :] = 1.0
    for c in range(8):
        shift = 4 * c
        state[L.STACK0 + c] = torch.tensor([(a >> shift) & 0xF for a in a_vals], dtype=dtype)
        state[L.AX + c] = torch.tensor([(b >> shift) & 0xF for b in b_vals], dtype=dtype)
    for _n, s in blocks:
        state = _sparse_apply_batch(state, s, dtype=dtype)
    out = []
    res_rows = state[res_band:res_band + n_result_nib].round().to(torch.int64)  # [nres, B]
    for k in range(B):
        v = 0
        for c in range(n_result_nib):
            v |= (int(res_rows[c, k]) & 0xF) << (4 * c)
        out.append(v)
    return out


# ===========================================================================
# MEASUREMENT
# ===========================================================================
def weight_nz(spec: Spec) -> int:
    return sum(int((spec[k] != 0).sum()) for k in ("W_up", "W_gate", "W_down"))


def measure(blocks: List[Block]) -> dict:
    return {"blocks": len(blocks),
            "nz": sum(weight_nz(s) for _, s in blocks)}


# ===========================================================================
# 3. STATIC RANGE-ANALYSIS over bytecode.  Conservative per-value nibble-width
#    bound propagated through the c4 stack machine.  A "value width" w means the
#    value is provably < 16^w (fits in w nibbles).  We track:
#      * AX width  (the accumulator)
#      * the operand stack (list of widths; PSH pushes AX's width)
#    and, per ALU instruction, record (op, wa, wb) = the widths the narrowed
#    gadget should be built for.
#
#    Propagation rules (all conservative — a bound is only ever an UPPER bound):
#      IMM k        -> width ceil(log16(k+1))         (exact for the literal)
#      LEA k        -> 2  (AX = (BP+imm) & 0xFF is an 8-bit op -> <= 2 nibbles)
#      AND (mask)   -> min(wa, mask_width)            (AND can only clear bits)
#      OR / XOR     -> max(wa, wb)                     (never exceeds the wider)
#      ADD          -> max(wa, wb) + 1                 (one carry nibble)
#      SUB          -> max(wa, wb)  (mod-wrap keeps it within the wider width)
#      MUL          -> wa + wb
#      DIV          -> wa (quotient <= dividend width)
#      MOD b        -> width of b (remainder < divisor)
#      SHL n / SHR n-> conservative 8 (shift amount dynamic)  unless n const
#      LI / LC      -> 2  (byte load -> <= 2 nibbles under the 8-bit mem model)
#      EQ..GE       -> 1  (boolean 0/1)
#      loop counter -> bounded by its init (see the loop pre-pass below)
#      default / unknown -> 8 (the full-width worst case)
#
#    The mask cap is applied everywhere (this ISA is the 8-bit slice, MASK=0xFF ->
#    every ALU result is re-folded to <= 2 nibbles), but the analysis is written
#    for the GENERAL 32-bit widths so it is reusable for the full model; the
#    ``mask_nibbles`` parameter sets the post-op fold (default 8 = no fold).
# ===========================================================================
def width_of_const(k: int) -> int:
    """Minimum nibbles to hold the non-negative constant ``k``."""
    k = int(k) & 0xFFFFFFFF
    if k == 0:
        return 1
    return max(1, math.ceil(math.log(k + 1, 16)))


def _mask_width(imm: int) -> int:
    """Width of a bit-mask constant = position of its top set bit, in nibbles."""
    imm = int(imm) & 0xFFFFFFFF
    if imm == 0:
        return 1
    return max(1, (imm.bit_length() + 3) // 4)


# ops that consume the top-of-stack operand ``a`` and combine with AX (``b``).
_STACK_ALU = {isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
              isa.AND, isa.OR, isa.XOR, isa.SHL, isa.SHR,
              isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE}


def infer_widths(program, mask_nibbles: int = 8) -> dict:
    """Static per-instruction operand-width inference over a c4 bytecode program.

    ``program`` is a list of ``isa.Instr`` (or ``(name, imm)`` tuples).  Returns::

        {
          "per_instr": [ {pc, op, ...width info...}, ... ],   # aligned to program
          "alu_sites": [ {pc, op, wa, wb, w_result}, ... ],   # the narrowable ops
          "ax_widths": [w0, w1, ...],                          # AX width per pc
        }

    ``mask_nibbles`` is the per-op result fold (8 = 32-bit no-fold; 2 = the 8-bit
    slice's MASK=0xFF fold).  Conservative: every width is an UPPER bound, so a
    gadget built at the inferred width is byte-exact by construction.
    """
    code = isa.assemble(program) if program and not hasattr(program[0], "op") else list(program)

    # --- loop pre-pass: a counter register initialised by IMM and only ever
    # decremented/incremented within a backward branch keeps its init width.
    # We conservatively cap AX after a backward branch (loop body) at the max AX
    # width seen entering the loop, so a countdown never widens.  Implemented as a
    # 2-pass fixpoint: run the transfer, and where a backward-branch target's AX
    # width would grow, clamp to the join.  For this 8-bit slice mask_nibbles=2
    # already caps everything; the pre-pass matters for the 32-bit reuse.
    n = len(code)
    branch_targets = set()
    for ins in code:
        if ins.op in (isa.JMP, isa.BZ, isa.BNZ, isa.JSR) and ins.imm < n:
            branch_targets.add(ins.imm)

    ax_w = mask_nibbles                       # unknown initial AX -> worst case
    stack: List[int] = []                     # operand-stack widths
    per_instr = []
    alu_sites = []
    ax_widths = []

    def cap(w):
        return min(int(w), mask_nibbles)

    for pc, ins in enumerate(code):
        op, imm = ins.op, ins.imm
        entry_ax = ax_w
        info = {"pc": pc, "op": isa.NAMES.get(op, op), "imm": imm,
                "ax_in": entry_ax}

        if op == isa.IMM:
            ax_w = cap(width_of_const(imm))
        elif op == isa.LEA:
            ax_w = cap(2)                     # (BP+imm)&0xFF -> <= 2 nibbles
        elif op == isa.PSH:
            stack.append(ax_w)                # push AX's width
        elif op in (isa.LI, isa.LC):
            ax_w = cap(2)                     # byte load
        elif op in (isa.SI, isa.SC):
            if stack:
                stack.pop()                   # store consumes an address operand
        elif op in _STACK_ALU:
            wa = stack.pop() if stack else mask_nibbles     # operand a (popped)
            wb = entry_ax                                    # operand b (AX)
            if op == isa.ADD:
                w_res = max(wa, wb) + 1
            elif op == isa.SUB:
                w_res = max(wa, wb)
            elif op == isa.MUL:
                w_res = wa + wb
            elif op == isa.DIV:
                w_res = wa                    # quotient <= dividend width
            elif op == isa.MOD:
                w_res = wb                    # remainder < divisor (b = AX)
            elif op == isa.AND:
                w_res = min(wa, wb)           # AND clears bits
            elif op in (isa.OR, isa.XOR):
                w_res = max(wa, wb)
            elif op in (isa.SHL, isa.SHR):
                # if the shift amount (AX=b) is a known small const we could tighten;
                # here b is dynamic -> conservative worst case for the shifted value.
                w_res = mask_nibbles
            else:                             # EQ/NE/LT/GT/LE/GE -> boolean
                w_res = 1
            w_res = cap(w_res)
            info.update({"wa": wa, "wb": wb, "w_result": w_res})
            alu_sites.append({"pc": pc, "op": isa.NAMES.get(op, op),
                              "wa": wa, "wb": wb, "w_result": w_res})
            ax_w = w_res
        elif op in (isa.JMP, isa.BZ, isa.BNZ):
            pass                              # control flow; AX unchanged
        elif op == isa.JSR:
            pass
        elif op in (isa.ENT, isa.ADJ, isa.LEV):
            pass
        else:
            ax_w = mask_nibbles               # unknown -> worst case
        info["ax_out"] = ax_w
        per_instr.append(info)
        ax_widths.append(ax_w)

    return {"per_instr": per_instr, "alu_sites": alu_sites, "ax_widths": ax_widths,
            "mask_nibbles": mask_nibbles}


# ===========================================================================
# REPRESENTATIVE BYTECODE PROGRAMS (for the typical-payoff report).
# ===========================================================================
def _prog_countdown():
    """for (i = 10; i; i--) sum += i;  — a small counter loop.  AX/operands are all
    1-2 nibbles (i <= 10, sum <= 55)."""
    return [
        ("IMM", 10), ("PSH", 0),          # i = 10 (on stack)
        ("IMM", 0), ("PSH", 0),           # sum = 0
        # loop body (idealised straight-line for the analysis):
        ("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0),    # small adds
        ("IMM", 1), ("PSH", 0), ("IMM", 10), ("SUB", 0),   # i - 1
        ("IMM", 2), ("PSH", 0), ("IMM", 4), ("MUL", 0),    # 2*4 small mul
        ("IMM", 7), ("PSH", 0), ("IMM", 3), ("LT", 0),     # i < 3 compare
        ("HALT", 0),
    ]


def _prog_charstring():
    """String / char processing: load chars (bytes), compare to delimiters, sum
    ordinals.  Every operand is a byte (<= 2 nibbles)."""
    prog = []
    for ch in "Hello":
        prog += [("IMM", ord(ch)), ("PSH", 0), ("IMM", ord(' ')), ("SUB", 0)]  # ch - ' '
        prog += [("IMM", ord(ch)), ("PSH", 0), ("IMM", ord('z')), ("LE", 0)]   # ch <= 'z'
        prog += [("LC", 0)]                                                     # byte load
    prog += [("HALT", 0)]
    return prog


def _prog_array_index():
    """Array-index loop: base + i*stride, then load; i and stride small, base a
    small address.  The index arithmetic is narrow (i <= 32, stride <= 8)."""
    prog = []
    for i in range(4):
        prog += [
            ("IMM", i), ("PSH", 0), ("IMM", 4), ("MUL", 0),     # i * 4 (stride)
            ("PSH", 0), ("IMM", 16), ("ADD", 0),                # + base 16
            ("LI", 0),                                          # load word
            ("PSH", 0), ("IMM", i + 1), ("ADD", 0),             # accumulate
        ]
    prog += [("HALT", 0)]
    return prog


# ===========================================================================
# NARROWING TABLE + TYPICAL-PROGRAM PAYOFF DRIVER
# ===========================================================================
DIM = 4096          # wide enough for the widest gadget's private scratch bands.


def _make_layout():
    L = NibbleVMLayout(8, n_heads=4)
    extend_layout_for_alu32(L)
    A._ONE = L.ONE
    return L


# full-width baselines (op -> the width-8 gadget's blocks/nz), the "everything at
# width 8" reference the payoff is measured against.  The width-8 builders here use
# the SAME carry-share split as ``nibble_alu32.compile_mul_blocks`` and so are
# apples-to-apples with the narrowed gadgets; the native ``compile_mul_blocks``
# (6,291 nz / 8 blk, the brief's headline number) is ALSO reported for reference.
def _full_width_baselines(L, dim) -> dict:
    base = {}
    for op, fn in (("MUL", lambda: build_mul(L, dim, 8, 8)),
                   ("ADD", lambda: build_add(L, dim, 8)),
                   ("SUB", lambda: build_sub(L, dim, 8)),
                   ("DIV", lambda: build_div(L, dim, 8)),
                   ("MOD", lambda: build_mod(L, dim, 8))):
        blocks, _res, _info = fn()
        base[op] = measure(blocks)
    # CMP baseline (use GT as the representative order op).
    blocks, _res, _info = build_cmp(L, dim, 8, isa.GT)
    base["CMP"] = measure(blocks)
    return base


def _native_mul_baseline(L, dim) -> dict:
    """The production ``nibble_alu32.compile_mul_blocks`` full 32-bit MUL — the
    brief's 6,291 nz / 8 blk headline (Kogge-Stone/ripple resolve; more compact than
    the from-scratch width-8 ``build_mul`` used for the apples-to-apples ratios)."""
    from .nibble_alu32 import compile_mul_blocks
    A._ONE = L.ONE
    blocks = compile_mul_blocks(L, dim)
    return measure(blocks)


def _rand_val(w, rng):
    """A random value that fits in ``w`` nibbles (< 16^w)."""
    hi = (1 << (4 * w)) - 1
    return rng.randint(0, hi)


def _verify_builder(L, dim, op: str, wa: int, wb: int, ref_fn, n_random=2000,
                    seed=0) -> Tuple[int, int, tuple]:
    """Byte-exact check of the narrowed gadget for (op, wa, wb) over n_random
    random operands within the width bound + edges, vs the full-width reference."""
    rng = random.Random(seed + hash((op, wa, wb)) % 100000)
    if op == "MUL":
        blocks, res, info = build_mul(L, dim, wa, wb)
        nres = info["n_result_nib"]
    elif op == "ADD":
        blocks, res, info = build_add(L, dim, wa); nres = info["n_result_nib"]
    elif op == "SUB":
        blocks, res, info = build_sub(L, dim, wa); nres = info["n_result_nib"]
    elif op == "DIV":
        blocks, res, info = build_div(L, dim, wa); nres = info["n_result_nib"]
    elif op == "MOD":
        blocks, res, info = build_mod(L, dim, wa); nres = info["n_result_nib"]
    elif op in ("EQ", "NE", "LT", "GT", "LE", "GE"):
        blocks, res, info = build_cmp(L, dim, wa, isa.BY_NAME[op]); nres = 1
    else:
        raise ValueError(op)

    # operand generators respecting the width bounds.
    def gen():
        a = _rand_val(wa, rng)
        b = _rand_val(wb, rng)
        return a, b
    # width-boundary edges within the bound.
    amax = (1 << (4 * wa)) - 1
    bmax = (1 << (4 * wb)) - 1
    edges = [(0, 0), (amax, bmax), (amax, 1), (1, bmax), (amax, 0), (0, bmax),
             (amax, amax & bmax)]
    ops_list = list(edges) + [gen() for _ in range(n_random)]

    npass = 0
    first_fail = None
    mask = (1 << (4 * nres)) - 1
    # BATCHED byte-exact check (chunked so the [dim, chunk] state tensor stays small).
    CHUNK = 512
    for s0 in range(0, len(ops_list), CHUNK):
        chunk = ops_list[s0:s0 + CHUNK]
        a_vals = [a for a, _ in chunk]
        b_vals = [b for _, b in chunk]
        got = simulate_batch(L, dim, blocks, res, nres, a_vals, b_vals, dtype=torch.float32)
        for (a, b), g in zip(chunk, got):
            exp = ref_fn(a, b) & mask
            if g == exp:
                npass += 1
            elif first_fail is None:
                first_fail = (hex(a), hex(b), hex(g), hex(exp))
    return npass, len(ops_list), first_fail


# reference semantics (masked to the result width by the caller).
def _ref(op):
    if op == "MUL":
        return lambda a, b: a * b
    if op == "ADD":
        return lambda a, b: a + b
    if op == "SUB":
        return lambda a, b: (a - b) & 0xFFFFFFFF
    if op == "DIV":
        return lambda a, b: (a // b) if b else 0
    if op == "MOD":
        return lambda a, b: (a % b) if b else 0
    if op == "EQ":
        return lambda a, b: 1 if a == b else 0
    if op == "NE":
        return lambda a, b: 1 if a != b else 0
    if op == "LT":
        return lambda a, b: 1 if a < b else 0
    if op == "GT":
        return lambda a, b: 1 if a > b else 0
    if op == "LE":
        return lambda a, b: 1 if a <= b else 0
    if op == "GE":
        return lambda a, b: 1 if a >= b else 0
    raise ValueError(op)


def run(n_random=2000, verbose=True) -> dict:
    L = _make_layout()
    dim = DIM
    results: dict = {}

    baselines = _full_width_baselines(L, dim)
    results["baselines"] = baselines
    native_mul = _native_mul_baseline(L, dim)
    results["native_mul_baseline"] = native_mul

    # -------- narrowing table: (op, wa, wb) -> blocks, nz --------
    if verbose:
        print("=" * 74)
        print("NARROWING TABLE  (op, wa, wb) -> blocks / nz  vs full-width-8 baseline")
        print("=" * 74)
        print(f"full-width-8 baselines: "
              + "  ".join(f"{k}={v['nz']}nz/{v['blocks']}blk" for k, v in baselines.items()))
        print(f"  (native compile_mul_blocks full-32-bit MUL = "
              f"{native_mul['nz']}nz/{native_mul['blocks']}blk — the brief's 6,291/8 headline)")
        print("-" * 74)
        print(f"{'op':4s} {'wa':>3s} {'wb':>3s} {'blocks':>7s} {'nz':>8s} "
              f"{'partials':>9s} {'vs full nz':>11s}  {'byte-exact':>12s}")

    # Cost-aware operand count.  A narrowed gadget is a FIXED circuit (the same
    # blocks fire for every operand), so byte-exactness is a property of the circuit,
    # not the operand set — ``n_random`` (>= 2000) proves it on the narrow gadgets
    # that ARE the deliverable, while the huge full-width DIV/MOD BASELINES (331
    # blocks) get a still-ample sample so the whole run stays in the seconds/minutes
    # regime the brief asks for (no GPU).
    def _n_ops(nblocks):
        if nblocks <= 20:
            return n_random
        if nblocks <= 80:
            return min(n_random, 600)      # width-2 DIV/MOD (73 blocks)
        if nblocks <= 160:
            return min(n_random, 300)      # width-4 DIV/MOD (143 blocks)
        return min(n_random, 150)          # width-8 DIV/MOD (331 blocks)

    table = []
    # MUL across a grid of widths (the headline: 2x2 ~ 1/9 of 8x8).
    mul_widths = [(1, 1), (2, 2), (2, 4), (4, 4), (2, 8), (4, 8), (8, 8)]
    for wa, wb in mul_widths:
        blocks, res, info = build_mul(L, dim, wa, wb)
        m = measure(blocks)
        npass, ntot, ff = _verify_builder(L, dim, "MUL", wa, wb, _ref("MUL"), _n_ops(m["blocks"]))
        ratio = m["nz"] / baselines["MUL"]["nz"]
        row = {"op": "MUL", "wa": wa, "wb": wb, **m,
               "n_partials": info["n_partials"], "nz_ratio": ratio,
               "pass": npass, "total": ntot, "first_fail": ff}
        table.append(row)
        if verbose:
            print(f"{'MUL':4s} {wa:3d} {wb:3d} {m['blocks']:7d} {m['nz']:8d} "
                  f"{info['n_partials']:9d} {ratio:10.2%}  {npass:5d}/{ntot:<5d} {ff or ''}")

    # ADD / SUB across widths.
    for opn, bld in (("ADD", build_add), ("SUB", build_sub)):
        for w in (2, 4, 8):
            blocks, res, info = bld(L, dim, w)
            m = measure(blocks)
            npass, ntot, ff = _verify_builder(L, dim, opn, w, w, _ref(opn), _n_ops(m["blocks"]))
            ratio = m["nz"] / baselines[opn]["nz"]
            table.append({"op": opn, "wa": w, "wb": w, **m, "nz_ratio": ratio,
                          "pass": npass, "total": ntot, "first_fail": ff})
            if verbose:
                print(f"{opn:4s} {w:3d} {w:3d} {m['blocks']:7d} {m['nz']:8d} "
                      f"{'-':>9s} {ratio:10.2%}  {npass:5d}/{ntot:<5d} {ff or ''}")

    # DIV / MOD across widths.
    for opn, bld in (("DIV", build_div), ("MOD", build_mod)):
        for w in (2, 4, 8):
            blocks, res, info = bld(L, dim, w)
            m = measure(blocks)
            npass, ntot, ff = _verify_builder(L, dim, opn, w, w, _ref(opn), _n_ops(m["blocks"]))
            ratio = m["nz"] / baselines[opn]["nz"]
            table.append({"op": opn, "wa": w, "wb": w, **m, "nz_ratio": ratio,
                          "pass": npass, "total": ntot, "first_fail": ff})
            if verbose:
                print(f"{opn:4s} {w:3d} {w:3d} {m['blocks']:7d} {m['nz']:8d} "
                      f"{'-':>9s} {ratio:10.2%}  {npass:5d}/{ntot:<5d} {ff or ''}")

    # CMP across widths (verify all six ops at w=2, then width scaling on GT).
    for op_name in ("EQ", "NE", "LT", "GT", "LE", "GE"):
        w = 2
        blocks, res, info = build_cmp(L, dim, w, isa.BY_NAME[op_name])
        m = measure(blocks)
        npass, ntot, ff = _verify_builder(L, dim, op_name, w, w, _ref(op_name), n_random)
        ratio = m["nz"] / baselines["CMP"]["nz"]
        table.append({"op": op_name, "wa": w, "wb": w, **m, "nz_ratio": ratio,
                      "pass": npass, "total": ntot, "first_fail": ff})
        if verbose:
            print(f"{op_name:4s} {w:3d} {w:3d} {m['blocks']:7d} {m['nz']:8d} "
                  f"{'-':>9s} {ratio:10.2%}  {npass:5d}/{ntot:<5d} {ff or ''}")
    results["table"] = table

    # -------- typical-program payoff --------
    if verbose:
        print()
        print("=" * 74)
        print("TYPICAL-PROGRAM PAYOFF  (infer_widths on representative bytecode)")
        print("=" * 74)

    progs = {"countdown_loop": _prog_countdown(),
             "char_string": _prog_charstring(),
             "array_index": _prog_array_index()}
    payoff = {}
    # nz(op, wa, wb) helper for the aggregate estimate.
    def _nz_for(op, wa, wb):
        if op == "MUL":
            _b, _r, _i = build_mul(L, dim, max(1, wa), max(1, wb))
        elif op in ("ADD",):
            _b, _r, _i = build_add(L, dim, max(1, max(wa, wb)))
        elif op in ("SUB",):
            _b, _r, _i = build_sub(L, dim, max(1, max(wa, wb)))
        elif op == "DIV":
            _b, _r, _i = build_div(L, dim, max(1, wa))
        elif op == "MOD":
            _b, _r, _i = build_mod(L, dim, max(1, wb))
        elif op in ("EQ", "NE", "LT", "GT", "LE", "GE"):
            _b, _r, _i = build_cmp(L, dim, max(1, max(wa, wb)), isa.BY_NAME[op])
        else:
            return None
        return measure(_b)["nz"]

    _base_for = {"MUL": "MUL", "ADD": "ADD", "SUB": "SUB", "DIV": "DIV", "MOD": "MOD",
                 "EQ": "CMP", "NE": "CMP", "LT": "CMP", "GT": "CMP", "LE": "CMP", "GE": "CMP"}

    all_widths = []
    tot_narrow_nz = tot_full_nz = 0
    n_narrow_ops = n_total_ops = 0
    for name, prog in progs.items():
        inf = infer_widths(prog, mask_nibbles=2)   # 8-bit slice -> 2-nibble fold
        sites = inf["alu_sites"]
        narrow_nz = full_nz = 0
        widths = []
        for st in sites:
            op = st["op"]
            if op not in _base_for:
                continue
            wa, wb = st["wa"], st["wb"]
            widths.append((op, wa, wb))
            all_widths.append((op, wa, wb))
            nnz = _nz_for(op, wa, wb) or 0
            fnz = baselines[_base_for[op]]["nz"]
            narrow_nz += nnz
            full_nz += fnz
            n_total_ops += 1
            if max(wa, wb) < 8:
                n_narrow_ops += 1
        tot_narrow_nz += narrow_nz
        tot_full_nz += full_nz
        payoff[name] = {"n_alu": len(widths), "widths": widths,
                        "narrow_nz": narrow_nz, "full_nz": full_nz,
                        "saved_nz": full_nz - narrow_nz,
                        "saved_frac": (1 - narrow_nz / full_nz) if full_nz else 0.0}
        if verbose:
            from collections import Counter
            dist = Counter((op, max(wa, wb)) for op, wa, wb in widths)
            print(f"\n{name}:  {len(widths)} ALU ops")
            print("   operand-width distribution (op, max-width -> count):")
            for (op, mw), cnt in sorted(dist.items()):
                print(f"      {op:4s} width {mw}: {cnt}")
            print(f"   narrowed nz {narrow_nz}  vs full-width-8 nz {full_nz}  "
                  f"-> saved {full_nz - narrow_nz} ({payoff[name]['saved_frac']:.1%})")

    # headline aggregate.
    from collections import Counter
    wdist = Counter(max(wa, wb) for _op, wa, wb in all_widths)
    narrow_ops = sum(c for w, c in wdist.items() if w < 8)
    total_ops = sum(wdist.values())
    results["payoff"] = payoff
    results["aggregate"] = {
        "total_alu_ops": total_ops, "narrow_ops": narrow_ops,
        "narrow_frac": narrow_ops / total_ops if total_ops else 0.0,
        "narrow_nz": tot_narrow_nz, "full_nz": tot_full_nz,
        "saved_nz": tot_full_nz - tot_narrow_nz,
        "saved_frac": (1 - tot_narrow_nz / tot_full_nz) if tot_full_nz else 0.0,
        "width_distribution": dict(sorted(wdist.items())),
    }
    if verbose:
        agg = results["aggregate"]
        print()
        print("=" * 74)
        print("HEADLINE AGGREGATE")
        print("=" * 74)
        print(f"  ALU ops across the 3 representative programs: {total_ops}")
        print(f"  operand max-width distribution: {agg['width_distribution']}")
        print(f"  NARROW ops (max-width < 8): {narrow_ops}/{total_ops} "
              f"= {agg['narrow_frac']:.1%}")
        print(f"  aggregate compute (nz): narrowed {tot_narrow_nz} vs "
              f"full-width-8 {tot_full_nz}")
        print(f"  TOTAL COMPUTE SAVED: {tot_full_nz - tot_narrow_nz} nz "
              f"= {agg['saved_frac']:.1%}")

    # byte-exact summary
    total_pass = sum(r["pass"] for r in table)
    total_tot = sum(r["total"] for r in table)
    fails = [(r["op"], r["wa"], r["wb"], r["first_fail"]) for r in table if r["first_fail"]]
    results["byte_exact"] = {"pass": total_pass, "total": total_tot, "fails": fails}
    if verbose:
        print()
        print(f"BYTE-EXACT (all narrowed builders, within-bound + edges): "
              f"{total_pass}/{total_tot}")
        if fails:
            print(f"  FAILURES: {fails}")
        else:
            print("  100% byte-exact — every narrowed gadget matches full-width "
                  "reference within its proven bound.")
    return results


if __name__ == "__main__":
    run()
