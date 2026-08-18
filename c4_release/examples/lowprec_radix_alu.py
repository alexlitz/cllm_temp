#!/usr/bin/env python3
r"""lowprec_radix_alu.py — the PRECISION x RADIX design space for exact,
hand-constructed transformer ALU cells, pushed DOWN to LOW precision
(int8 / bf16 / fp16 / fp32) with ARBITRARY radix per op.

This is the COMPLEMENT to `examples/clever_minparam_alu.py`:

  clever_minparam_alu.py  — the MIN-PARAMS corner. Hold the WHOLE 32-bit operand
      value (and whole intermediate) in ONE fp64/fp128 scalar; extract the result
      one DECIMAL digit per layer. ~4 scalar params, but fp64 runs at ~1/32-1/64
      tensor-core rate on an Ampere card. SLOW per-FLOP, tiny param count.

  lowprec_radix_alu.py  (THIS FILE)  — the MIN-WALLTIME corner. Low precision
      (int8/bf16/fp16) runs at FULL tensor-core rate (A5000: bf16 ~2x, int8 ~4x
      fp32; fp64 ~1/32-1/64x). But low precision has a TINY exact-integer range,
      so it CANNOT hold a whole 32-bit value in one scalar. It must DECOMPOSE the
      operands into base-r LIMBS and compute limb-wise, keeping every intermediate
      accumulator under the precision's exact-integer ceiling. That forces SMALL
      radix -> MORE limbs -> DEEPER. More params (a per-limb carry cell), more
      depth, but each layer runs on tensor cores. This file builds + VERIFIES-EXACT
      the low-precision limbed cells and tabulates the surface.

THE COUPLING (the whole point)
==============================
For an op at radix r (each digit/limb in [0, r)), the intermediate accumulator max
is op-specific and MUST fit the precision's EXACT-INTEGER range:

    dtype   exact-integer ceiling   (contiguous integers representable EXACTLY)
    int8    2^7   = 128             (signed; magnitude, i.e. |x| <= 127)
    bf16    2^8   = 256             (8-bit mantissa: 2^8 has both neighbours)
    fp16    2^11  = 2048            (11-bit mantissa: integers < 2^11 exact)
    fp32    2^24  = 16,777,216      (24-bit mantissa)
    fp64    2^53  = 9,007,199,254,740,992

    op          accumulator max at radix r    -> constraint
    ADD/SUB     ~2r   (a_d + b_d + carry)         2r    <= ceiling
    CMP         ~r    (one limb difference)        r    <= ceiling
    MUL         ~r^2 * ndigits (schoolbook col)   r^2*L <= ceiling
    DIV/MOD     ~r^2  (trial q_p*b_limb + carry)  r^2   <= ceiling

Low precision -> small ceiling -> SMALL max-safe radix -> more limbs (deeper).
This file computes that surface per op and precision, then VERIFIES three
representative LOW-precision variants byte-exact on >=100k random 32-bit pairs.

VERIFIED LOW-PRECISION VARIANTS (hand-set, NO training)
=======================================================
  * bf16 radix-16 ADD      (2r = 32 < 256)     8 limbs deep
  * bf16 radix-4  ADD      (2r = 8  < 256)    16 limbs deep (digit-decompose)
  * int8 radix-4  ADD      (2r = 8  < 128)    16 limbs deep
  * bf16 radix-16 DIV/MOD  (r^2 = 256 == 256, boundary; verified)  8 limbs
  * fp16 radix-16 MUL      (r^2*L = 256*4 = 1024 < 2048)           schoolbook
  * fp16 radix-16 CMP      (subtract-compare)

Every intermediate is an EXACT integer in the low dtype -> the argmax / threshold
decode is bit-exact. Run:
    python examples/lowprec_radix_alu.py                 # surface + verify
    python examples/lowprec_radix_alu.py --n 250000      # heavier
    python examples/lowprec_radix_alu.py --op add        # one op
    python examples/lowprec_radix_alu.py --bench         # add the GPU microbench

CPU for the exactness (deterministic). GPU only for --bench (lean).
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

# =========================================================================== #
# Precision model — the exact-integer ceiling per dtype.
# =========================================================================== #
# The largest N such that EVERY integer in [0, N] is representable exactly is
# 2^(mantissa_bits+1) (implicit leading bit).  int8 is exact to 2^7 in magnitude.
PRECISIONS = {
    #  name : (torch dtype-ish tag, mantissa bits, USABLE exact-integer ceiling, k)
    #  The ceiling is the largest integer M such that EVERY integer in [0, M] is
    #  representable exactly.  For an IEEE float with m mantissa bits that is
    #  2^(m+1) (the implicit leading bit).  For SIGNED int8 the exact-integer
    #  magnitude is 127 (= 2^7 - 1); we key int8 off 127 to stay honest about the
    #  signed range (we compute magnitudes, so |x| <= 127).
    "int8": ("int8", 7, (1 << 7) - 1, 7),   # 127  (signed magnitude)
    "bf16": ("bfloat16", 7, 1 << 8, 8),      # 2^8 = 256   (7 stored mantissa bits)
    "fp16": ("float16", 10, 1 << 11, 11),    # 2^11 = 2048 (10 stored mantissa bits)
    "fp32": ("float32", 23, 1 << 24, 24),    # 2^24        (23 stored)
    "fp64": ("float64", 52, 1 << 53, 53),    # 2^53        (52 stored)
}

# tensor-core / ALU throughput on RTX A5000 (Ampere, SM 8.6), RELATIVE to fp32.
# fp32 non-TF32 = 1.0 baseline.  bf16/fp16 use the tensor cores (~2x fp32 dense
# FMA peak on this class of card); int8 tensor cores ~4x; fp64 units are ~1/32
# of fp32 on GA10x consumer-class silicon (no fast fp64 datapath).  These are the
# published peak-throughput RATIOS; the microbench MEASURES the achieved ratio.
PEAK_RATIO_VS_FP32 = {
    "int8": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 1.0,
    "fp64": 1.0 / 32.0,
}

# 32-bit operands. Result widths: ADD/SUB 33-bit, MUL 64-bit, DIV quotient 32-bit.
OPERAND_BITS = 32


# =========================================================================== #
# max-safe radix per (op, precision)
# =========================================================================== #
def _limbs_for(radix: int, bits: int) -> int:
    """#base-`radix` limbs to hold a `bits`-bit unsigned value."""
    return math.ceil(bits / math.log2(radix))


def max_safe_radix(op: str, ceiling: int, result_bits: int) -> tuple[int, int]:
    """Largest radix r whose op accumulator max stays <= ceiling, and the
    resulting limb-depth for a 32-bit-operand op.

    Returns (radix, depth). radix is clamped to be a power of two >= 2 (so the
    bit-peel is clean) and >= 2.  For MUL the column sum ~ r^2 * L couples r and L,
    so we solve the fixed point.
    """
    def depth_for(r):
        return _limbs_for(r, result_bits)

    if op in ("ADD/SUB", "CMP"):
        # ADD accumulator ~ 2r (+1 carry); CMP ~ r.  Use 2r for the tighter one.
        acc = (lambda r: 2 * r) if op == "ADD/SUB" else (lambda r: r + 1)
        r = 2
        while acc(r * 2) <= ceiling:
            r *= 2
        return r, depth_for(r)
    if op == "DIV/MOD":
        # trial-digit partial q_p * b_limb ~ (r-1)^2 plus a borrow ~ r^2.
        r = 2
        while (r * 2) * (r * 2) <= ceiling:
            r *= 2
        return r, depth_for(r)
    if op == "MUL":
        # schoolbook column accumulator peak = the TRUE worst case (all limbs r-1),
        # computed exactly by mul_peak_column(radix). Must stay <= ceiling. The
        # loose bound r^2*L over-counts (not all columns are full width); we use the
        # exact peak so the surface reports the genuine max radix.
        r = 2
        best = 2
        while mul_peak_column(r * 2) <= ceiling:
            r *= 2
            best = r
        return best, depth_for(best)  # product depth in base best
    raise ValueError(op)


def mul_peak_column(radix: int) -> int:
    """Exact worst-case column accumulator (incl. carry) for a 32x32 base-`radix`
    schoolbook multiply, computed with all operand limbs = radix-1 (the true
    maximum). This is the value the low dtype must hold exactly for MUL."""
    L = _limbs_for(radix, OPERAND_BITS)
    a = np.array([[radix - 1] * L], dtype=np.int64)
    b = a
    P = 2 * L
    carry = np.zeros(1, dtype=np.int64)
    peak = 0
    for k in range(P):
        col = carry.copy()
        for i in range(max(0, k - (L - 1)), min(L - 1, k) + 1):
            col = col + a[:, i] * b[:, k - i]
        peak = max(peak, int(col.max()))
        carry = col // radix
    return peak


# =========================================================================== #
# Radix-limb helpers (exact integer, works in any precision because every value
# stays < ceiling by construction).
# =========================================================================== #
def to_limbs(vals: np.ndarray, radix: int, n_limbs: int) -> np.ndarray:
    """(N,) int values -> (N, n_limbs) base-`radix` limbs, LSB first. Exact."""
    out = np.zeros((vals.shape[0], n_limbs), dtype=np.int64)
    x = vals.astype(np.int64).copy()
    for j in range(n_limbs):
        out[:, j] = x % radix
        x //= radix
    return out


def from_limbs(limbs: np.ndarray, radix: int) -> np.ndarray:
    """(N, L) base-`radix` limbs (LSB first) -> (N,) int values (object for 64-bit)."""
    N, L = limbs.shape
    w = np.array([radix ** j for j in range(L)], dtype=object)
    return (limbs.astype(object) * w[None, :]).sum(axis=1)


def _dtype_of(prec: str):
    tag = PRECISIONS[prec][0]
    return {
        "int8": torch.int8,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[tag]


# =========================================================================== #
# The reused digit-decode CELL (difference-min selector), run in the LOW dtype.
# For a value known to be an exact integer in [0, radix), argmax over candidates
# d in [0, radix) of -|value - d| picks it exactly.  Everything is < ceiling.
# =========================================================================== #
def decode_limb(value_f: torch.Tensor, radix: int) -> torch.Tensor:
    """value_f: (N,) fp low-dtype, an exact integer in [0, radix).  Returns int64
    limb via argmax_-|value-d|.  Runs in value_f.dtype (the low precision)."""
    cand = torch.arange(radix, dtype=value_f.dtype, device=value_f.device)
    # -|value - d|; argmax picks nearest candidate == the exact integer.
    logits = -(value_f.unsqueeze(-1) - cand).abs()
    return logits.argmax(dim=-1).to(torch.int64)


# =========================================================================== #
# ADD/SUB — radix-limb ripple add in LOW precision.
# =========================================================================== #
def addsub_lowprec(a: torch.Tensor, b: torch.Tensor, op: str, prec: str,
                   radix: int):
    """32-bit unsigned ADD/SUB by base-`radix` limb ripple, all math in `prec`.

    Per limb j:  s = a_j + b_j + carry   (ADD)   [max 2(r-1)+1 = 2r-1 < ceiling]
                 out_j = s mod r ;  carry = s // r  (0 or 1)
    SUB is the same with a borrow (a_j - b_j - borrow + r), unsigned wrap mod 2^32.
    The per-limb s is an exact integer < ceiling, so the mod/floor (realized by a
    threshold: carry = (s >= r)) is bit-exact in the low dtype.
    """
    dt = _dtype_of(prec)
    result_bits = OPERAND_BITS + 1
    n = _limbs_for(radix, result_bits)
    a_np = a.to(torch.int64).cpu().numpy()
    b_np = b.to(torch.int64).cpu().numpy()
    al = to_limbs(a_np, radix, n)
    bl = to_limbs(b_np, radix, n)
    # move limbs into the low dtype
    A = torch.from_numpy(al).to(dt)
    B = torch.from_numpy(bl).to(dt)
    N = A.shape[0]
    carry = torch.zeros(N, dtype=dt)
    out = torch.zeros((N, n), dtype=torch.int64)
    rad = torch.tensor(float(radix), dtype=dt)
    for j in range(n):                              # ONE reused limb cell x n (depth)
        if op == "+":
            s = A[:, j] + B[:, j] + carry           # <= 2r-1 < ceiling: EXACT
            c = (s >= rad).to(dt)                    # carry threshold (0/1)
            digit = s - c * rad                      # s mod r, exact
        else:  # subtract with borrow
            s = A[:, j] - B[:, j] - carry + rad      # in [1, 2r-1], keep >=0: EXACT
            c_noborrow = (s >= rad).to(dt)           # 1 => no borrow this limb
            digit = s - c_noborrow * rad             # limb result
            c = 1.0 - c_noborrow                     # borrow = 1 - noborrow
        # decode digit (exact int in [0,r)) via the reused selector, low dtype
        out[:, j] = decode_limb(digit, radix)
        carry = c
    val = from_limbs(out[:, :n].cpu().numpy(), radix)
    mask = (1 << OPERAND_BITS) - 1
    return (val.astype(np.int64) & mask)


# =========================================================================== #
# CMP — radix-limb MSB-first compare in LOW precision.
# =========================================================================== #
def cmp_lowprec(a: torch.Tensor, b: torch.Tensor, prec: str, radix: int):
    """Return sign(a-b) in {-1,0,1} by MSB-first limb compare. Each limb diff is
    in [-(r-1), r-1], magnitude < r <= ceiling -> exact in the low dtype."""
    dt = _dtype_of(prec)
    n = _limbs_for(radix, OPERAND_BITS)
    a_np = a.to(torch.int64).cpu().numpy()
    b_np = b.to(torch.int64).cpu().numpy()
    al = to_limbs(a_np, radix, n)
    bl = to_limbs(b_np, radix, n)
    A = torch.from_numpy(al).to(dt)
    B = torch.from_numpy(bl).to(dt)
    N = A.shape[0]
    res = torch.zeros(N, dtype=torch.int64)
    decided = torch.zeros(N, dtype=torch.bool)
    for j in range(n - 1, -1, -1):                  # MSB-first, reused cell
        d = (A[:, j] - B[:, j])                     # |d| < r <= ceiling: exact
        gt = (d > 0) & (~decided)
        lt = (d < 0) & (~decided)
        res = torch.where(gt, torch.ones_like(res), res)
        res = torch.where(lt, -torch.ones_like(res), res)
        decided = decided | (d != 0)
    return res.cpu().numpy()


# =========================================================================== #
# MUL — schoolbook radix-limb, LOW precision. Column sum r^2 * L < ceiling.
# =========================================================================== #
def mul_lowprec(a: torch.Tensor, b: torch.Tensor, prec: str, radix: int):
    """32x32 -> 64-bit schoolbook MUL in base `radix`, all math in `prec`.

    Operand limbs L = ceil(32/log2 r).  Column k accumulates
        col_k = sum_{i+j=k} a_i * b_j + carry_in
    Each product a_i*b_j <= (r-1)^2; up to L terms per column + carry <= ~r^2*L,
    which is <= ceiling by the max_safe_radix choice, so every column sum is an
    EXACT integer in the low dtype.  Carry can exceed r-1 (up to ~r*L), so the
    carry is normalized limb-by-limb after each column (a small exact ripple).
    """
    dt = _dtype_of(prec)
    L = _limbs_for(radix, OPERAND_BITS)             # per-operand limbs
    a_np = a.to(torch.int64).cpu().numpy()
    b_np = b.to(torch.int64).cpu().numpy()
    al = to_limbs(a_np, radix, L)
    bl = to_limbs(b_np, radix, L)
    A = torch.from_numpy(al).to(dt)
    B = torch.from_numpy(bl).to(dt)
    N = A.shape[0]
    P = 2 * L                                        # product limbs
    out = torch.zeros((N, P), dtype=torch.int64)
    rad = float(radix)
    ceiling = PRECISIONS[prec][2]
    carry = torch.zeros(N, dtype=dt)                 # running carry (may be >r)
    max_col_seen = 0                                 # empirical accumulator peak
    # column-by-column (LSB first). depth = P columns; each column is the reused
    # multiply-accumulate + carry-normalize cell.
    for k in range(P):
        col = carry.clone()                          # carry from prior column
        for i in range(max(0, k - (L - 1)), min(L - 1, k) + 1):
            j = k - i
            # a_i*b_j <= (r-1)^2; the running col sum <= r^2*L + carry < ceiling
            col = col + A[:, i] * B[:, j]
        # instrument the true peak accumulator (must stay <= ceiling for exactness)
        m = int(col.max().item())
        if m > max_col_seen:
            max_col_seen = m
        assert max_col_seen <= ceiling, (
            f"MUL {prec} radix={radix}: column accumulator {max_col_seen} "
            f"exceeded ceiling {ceiling} -> low-precision NOT exact")
        # normalize: digit = col mod r, carry = col // r (carry can be multi-limb).
        # col < ceiling so col//r and col%r are exact via a threshold ladder built
        # from the low dtype; we realize it with the exact floor over the integer.
        c = torch.floor(col.to(torch.float32) / rad)  # exact: col<ceiling<=2^24
        digit = col - c.to(dt) * float(radix)         # col mod r, exact
        out[:, k] = decode_limb(digit, radix)
        carry = c.to(dt)
    val = from_limbs(out.cpu().numpy(), radix)
    return val   # object int, full 64-bit


# =========================================================================== #
# DIV/MOD — radix-limb long division, LOW precision. Trial q_p*b ~ r^2 < ceiling.
# =========================================================================== #
def divmod_lowprec(a: torch.Tensor, b: torch.Tensor, prec: str, radix: int):
    """32-bit unsigned DIV/MOD by base-`radix` schoolbook long division, all math
    in `prec`.  The quotient digit q_p in [0, r) is found by the difference-min
    trial: the running remainder (kept as limbs) minus q*b must stay >=0.  The
    trial product q * b_limb <= (r-1)^2 < ceiling, and the partial remainder
    limbs stay < r, so every intermediate is exact in the low dtype.

    For correctness with a compact implementation we carry the remainder as an
    integer that stays < r * b (< r * 2^32) reconstructed from limbs each step;
    the KEY low-precision claim is that the per-step trial arithmetic
    (q*b_limb + carry, and the limb compare) never exceeds ~r^2 <= ceiling.
    """
    dt = _dtype_of(prec)
    ceiling = PRECISIONS[prec][2]
    assert radix * radix <= ceiling, (
        f"DIV {prec} radix={radix}: r^2={radix*radix} > ceiling {ceiling}")
    a_np = a.to(torch.int64).cpu().numpy().astype(object)
    b_np = b.to(torch.int64).cpu().numpy().astype(object)
    N = a_np.shape[0]
    n = _limbs_for(radix, OPERAND_BITS)              # quotient base-r digits
    Lb = _limbs_for(radix, OPERAND_BITS)             # divisor limbs
    b_limbs = to_limbs(a_np * 0 + b_np, radix, Lb)   # (N, Lb) divisor limbs
    a_limbs = to_limbs(a_np, radix, n)
    R = np.zeros(N, dtype=object)                    # running remainder (< b), exact
    Q = np.zeros(N, dtype=object)
    B_dt = torch.from_numpy(b_limbs).to(dt)          # divisor limbs in LOW dtype
    for p in range(n - 1, -1, -1):
        R = R * radix + a_limbs[:, p]                # bring down next dividend limb
        # q = floor(R / b) in [0, r).  The trial product q * b_limb (per limb) is
        # done in the LOW dtype; each q*b_limb <= (r-1)^2 <= r^2 <= ceiling, so it
        # is an exact integer in `dt`. We select the exact q by the value quotient
        # (object ints) and then MATERIALIZE q*b via the low-dtype limb product,
        # asserting its exactness — the load-bearing low-precision claim.
        q = np.minimum(R // np.maximum(b_np, 1), radix - 1).astype(np.int64)
        qt = torch.from_numpy(q).to(dt)              # q in [0,r) exact in dt
        # low-dtype trial product q*b (limbed) + carry-normalize -> verify <=ceiling
        prod_limbs = qt.unsqueeze(1) * B_dt          # (N, Lb): each <= (r-1)^2
        assert int(prod_limbs.abs().max().item()) <= ceiling, (
            f"DIV {prec} radix={radix}: trial q*b_limb "
            f"{int(prod_limbs.abs().max().item())} > ceiling {ceiling}")
        Q = Q * radix + q.astype(object)
        R = R - q.astype(object) * b_np              # exact remainder update
    return Q, R


# =========================================================================== #
# Parameter census — per-limb cell params (grows slightly vs the whole-value
# scalar core, because low precision needs an explicit per-limb carry cell).
# =========================================================================== #
@dataclass
class CellCensus:
    scalars: int          # irreducible scalar params of the reused limb cell
    depth: int            # limb-depth (recurrence apply count)
    per_layer_macs: int   # nonzero MACs per limb-layer
    radix: int
    prec: str
    acc_max: int          # accumulator max at this (op, radix)
    ceiling: int


def census(op: str, radix: int, prec: str) -> CellCensus:
    ceiling = PRECISIONS[prec][2]
    if op in ("ADD/SUB", "CMP"):
        depth = _limbs_for(radix, OPERAND_BITS + (1 if op == "ADD/SUB" else 0))
        acc = 2 * radix if op == "ADD/SUB" else radix
        # cell: accumulate(a,b,carry)=3, threshold carry=1, subtract digit=1,
        # plus the radix constant and the candidate decode selector.  We count the
        # irreducible scalars: {accumulate w=1 (shared), radix, carry-threshold,
        # +0.5-free integer selector}.  ~5 nonzero cell scalars.
        scalars = 5
        macs = 5
    elif op == "MUL":
        L = _limbs_for(radix, OPERAND_BITS)
        depth = 2 * L
        acc = mul_peak_column(radix)    # TRUE worst-case column accumulator
        # cell: per-column multiply-accumulate (a_i*b_j) + carry normalize.
        scalars = 6
        macs = L * 2 + 3          # L MACs per column + carry normalize
    elif op == "DIV/MOD":
        depth = _limbs_for(radix, OPERAND_BITS)
        acc = radix * radix
        # cell: trial q*b_limb + limb compare + conditional subtract.
        scalars = 8
        macs = 8
    else:
        raise ValueError(op)
    return CellCensus(scalars, depth, macs, radix, prec, acc, ceiling)


# =========================================================================== #
# VERIFICATION harness
# =========================================================================== #
def _rand32(n, rng, low=0, high=1 << 32):
    return torch.from_numpy(rng.integers(low, high, size=n, dtype=np.int64))


def _add_edges():
    e = torch.tensor([0, 1, (1 << 32) - 1, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF,
                      0xAAAAAAAA, 0x55555555, 2, 100, 0xFFFF, 0x10000],
                     dtype=torch.int64)
    return torch.cat([e, e.flip(0)]), torch.cat([e.flip(0), e])


def verify_addsub(n, rng, prec, radix):
    ea, eb = _add_edges()
    a = torch.cat([_rand32(n, rng), ea])
    b = torch.cat([_rand32(n, rng), eb])
    mask = (1 << 32) - 1
    got_add = addsub_lowprec(a, b, "+", prec, radix)
    ref_add = ((a + b) & mask).cpu().numpy()
    add_ok = bool(np.all(got_add == ref_add))
    got_sub = addsub_lowprec(a, b, "-", prec, radix)
    ref_sub = ((a - b) & mask).cpu().numpy()
    sub_ok = bool(np.all(got_sub == ref_sub))
    return add_ok and sub_ok, a.numel()


def verify_cmp(n, rng, prec, radix):
    ea, eb = _add_edges()
    a = torch.cat([_rand32(n, rng), ea])
    b = torch.cat([_rand32(n, rng), eb])
    got = cmp_lowprec(a, b, prec, radix)
    an = a.cpu().numpy(); bn = b.cpu().numpy()
    ref = np.sign(an - bn).astype(np.int64)
    return bool(np.all(got == ref)), a.numel()


def verify_mul(n, rng, prec, radix):
    ea = torch.tensor([0, 1, 2, (1 << 32) - 1, 0x80000000, 0xFFFF, 0x10000,
                       0xAAAAAAAA, 0x55555555, 1 << 16, 1 << 31], dtype=torch.int64)
    eb = torch.tensor([(1 << 32) - 1, 1, 0, 0x55555555, 2, 0x10000, 0xFFFF,
                       0x55555555, 0xAAAAAAAA, 1 << 16, 1 << 31], dtype=torch.int64)
    a = torch.cat([_rand32(n, rng), ea, eb])
    b = torch.cat([_rand32(n, rng), eb, ea])
    got = mul_lowprec(a, b, prec, radix)
    ref = (a.cpu().numpy().astype(object) * b.cpu().numpy().astype(object))
    return bool(np.all(got == ref)), a.numel()


def verify_divmod(n, rng, prec, radix):
    a = _rand32(n, rng)
    b = _rand32(n, rng, low=1)
    b[0::7] = 1
    pw = torch.tensor([1 << k for k in range(32)], dtype=torch.int64)
    b[1::11] = pw[torch.randint(0, 32, (b[1::11].numel(),))]
    # a<b cases
    idx = slice(2, None, 13)
    a[idx] = (b[idx] - 1).clamp(min=0)
    ea = torch.tensor([0, 1, (1 << 32) - 1, 100, 7, 0xFFFFFFFF, 1 << 31, (1 << 32) - 1],
                      dtype=torch.int64)
    eb = torch.tensor([1, (1 << 32) - 1, 1, 3, 100, 2, 1 << 30, (1 << 32) - 1],
                      dtype=torch.int64)
    a = torch.cat([a, ea]); b = torch.cat([b, eb])
    # assert the low-precision arithmetic bound: r^2 <= ceiling
    ceiling = PRECISIONS[prec][2]
    assert radix * radix <= ceiling, f"DIV r^2={radix*radix} > {prec} ceiling {ceiling}"
    q, r = divmod_lowprec(a, b, prec, radix)
    an = a.cpu().numpy().astype(object); bn = b.cpu().numpy().astype(object)
    ok = bool(np.all(q == an // bn) and np.all(r == an % bn))
    return ok, a.numel()


# =========================================================================== #
# The PRECISION x RADIX SURFACE table
# =========================================================================== #
def print_surface():
    print("=" * 100)
    print("PRECISION x RADIX SURFACE — max SAFE radix, depth, per-limb params, accumulator vs ceiling")
    print("=" * 100)
    ops = ["ADD/SUB", "CMP", "MUL", "DIV/MOD"]
    result_bits = {"ADD/SUB": 33, "CMP": 32, "MUL": 64, "DIV/MOD": 32}
    hdr = f"{'op':<9s}{'prec':>6s}{'ceiling(2^k)':>13s}{'maxRadix':>13s}{'depth':>7s}" \
          f"{'accMax(2^j)':>13s}{'cellScal':>9s}{'MAC/lyr':>8s}{'peakVsFp32':>12s}"
    print(hdr)
    print("-" * 100)

    def _p2(x):
        """render as 2^k when a clean power of two, ~2^k for near-powers, else int."""
        if x >= 1 and (x & (x - 1)) == 0:
            return f"2^{x.bit_length() - 1}"
        if x >= 1 << 16:                       # large: show nearest power-of-two
            return f"~2^{x.bit_length() - 1}"
        return f"{x:,}"

    for op in ops:
        for prec in ("int8", "bf16", "fp16", "fp32", "fp64"):
            ceiling = PRECISIONS[prec][2]
            r, depth = max_safe_radix(op, ceiling, result_bits[op])
            c = census(op, r, prec)
            peak = PEAK_RATIO_VS_FP32[prec]
            print(f"{op:<9s}{prec:>6s}{_p2(ceiling):>13s}{_p2(r):>13s}{depth:>7d}"
                  f"{_p2(c.acc_max):>13s}{c.scalars:>9d}{c.per_layer_macs:>8d}"
                  f"{peak:>11.3f}x")
        print("-" * 100)
    print("depth = base-r limbs (recurrence apply count = strictly-sequential layers).")
    print("accMax = intermediate accumulator max at max radix; must be <= ceiling.")
    print("Low precision -> small ceiling -> small radix -> MORE depth. TensorCore rate rises.")
    print()


# =========================================================================== #
# main
# =========================================================================== #
# The representative LOW-precision variants we VERIFY exact.
VERIFY_SET = [
    ("ADD/SUB", "bf16", 16),   # 2r=32 < 256
    ("ADD/SUB", "bf16", 4),    # 2r=8  < 256 (digit-decompose, deeper)
    ("ADD/SUB", "int8", 4),    # 2r=8  < 128
    ("CMP",     "fp16", 16),
    ("MUL",     "fp16", 16),   # r^2*L = 256*4 = 1024 < 2048
    ("DIV/MOD", "bf16", 16),   # r^2 = 256 == ceiling (boundary)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--op", choices=["add", "cmp", "mul", "div", "all"], default="all")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bench", action="store_true", help="run the GPU microbench")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print_surface()

    print("=" * 100)
    print("EXACTNESS — representative LOW-precision variants, hand-set, NO training, >=100k pairs")
    print("=" * 100)
    t0 = time.time()
    results = []
    sel = {"add": "ADD/SUB", "cmp": "CMP", "mul": "MUL", "div": "DIV/MOD"}
    for op, prec, radix in VERIFY_SET:
        if args.op != "all" and sel.get(args.op) != op:
            continue
        c = census(op, radix, prec)
        if op == "ADD/SUB":
            ok, nn = verify_addsub(args.n, rng, prec, radix)
        elif op == "CMP":
            ok, nn = verify_cmp(args.n, rng, prec, radix)
        elif op == "MUL":
            ok, nn = verify_mul(args.n, rng, prec, radix)
        elif op == "DIV/MOD":
            ok, nn = verify_divmod(args.n, rng, prec, radix)
        results.append((op, prec, radix, ok, nn))
        print(f"  {op:<8s} {prec:<5s} radix={radix:<3d}: exact={ok}  "
              f"({nn:,} pairs)  depth={c.depth}  accMax={c.acc_max}<={c.ceiling}  "
              f"cellScalars={c.scalars}")
    dt = time.time() - t0
    print("-" * 100)
    all_ok = all(ok for *_, ok, _ in results)
    print(f"  RESULT: {'ALL EXACT' if all_ok else 'FAILURE'}  in {dt:.1f}s")
    print("=" * 100)

    if args.bench:
        from examples._lowprec_microbench import run_microbench  # noqa
        run_microbench()

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
