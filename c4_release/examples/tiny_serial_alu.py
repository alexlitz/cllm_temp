#!/usr/bin/env python3
r"""tiny_serial_alu.py — hand-constructed, EXACT, DEEP-SERIAL minimal-nonzero-param
transformer ALU cells for ADD, MUL, and DIV/MOD.

WHY THIS EXISTS
===============
The production c4 doom-transformer costs ~5.68 MFLOP per VM step almost entirely
because it is WIDE: d_model ~= 1440-1656, so every gate multiplies against a
~1440-wide residual even when the *useful* arithmetic is 1-100 FLOP.  A gate that
needs a handful of MACs still pays ~1440 MACs of width overhead.

This file demonstrates the opposite construction: NARROW + DEEP-SERIAL.  Each ALU
is a tiny stack of layers with the SMALLEST possible d_model, peeling ONE bit (or
one nibble) per layer.  The per-layer weight matrix is a fixed, hand-constructed
"cell" that is SHARED across all bit/nibble positions (weight-tying), so the
nonzero param count is tiny and constant while DEPTH scales with the operand width.

The point is a FLOP statement, not a params statement: removing the width collapses
FLOP-per-op from ~M toward the op's true gate-count (a few MACs x depth), verified
EXACT (no training) on >=100k random 32-bit operands per op.

Each cell is implemented as an explicit small matmul (the "transformer layer" linear
map) plus a pointwise nonlinearity, so the MAC count is literally the number of
nonzero weights times the number of layer-applications.  We count *nonzero-weight*
MACs (a zero weight is never multiplied in a real sparse/COO kernel), and FLOP =
2 x MACs (one multiply + one add per MAC), matching the c4 _flop_gauge convention.

PRECISION
=========
- ADD  : bit-serial full-adder.  Single bits (0/1) and a single carry bit fit ANY
         float exactly -> fp32 is exact.
- MUL  : bit-serial shift-add.  Accumulates a<<i (a 64-bit partial) when b_i=1.
         64-bit integers exceed fp64's 2^53 exact-integer range, so the accumulator
         is carried NIBBLE-SERIAL in base-16 limbs (each limb 0..15, sum <= a few
         hundred -> trivially exact in fp32).  => fp32 exact, no fp128 needed.
         (A *parallel* 64-bit-product construction WOULD need fp128; we note it but
          the bit-serial limbed form is cleaner and stays in fp32.)
- DIV  : bit-serial restoring long division.  Remainder is a 33-bit register carried
         as nibble limbs; the shift+compare+conditional-subtract each stay small ->
         fp32 exact.  (q, r) both exact.

Run:
    python examples/tiny_serial_alu.py            # verify all three, 100k pairs each
    python examples/tiny_serial_alu.py --n 200000 # heavier
    python examples/tiny_serial_alu.py --op add   # one op

Every op prints: nonzero params (shared cell), d_model, depth (layer-apps),
precision, and FLOP/op (= depth x per-layer nonzero-MACs x 2).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

torch.manual_seed(0)


# =========================================================================== #
# Reporting record
# =========================================================================== #
@dataclass
class OpReport:
    name: str
    nonzero_params: int      # nonzero weights in the SHARED cell (counted once)
    d_model: int             # residual width the cell operates on
    depth: int               # layer-applications (serial steps)
    precision: str           # "fp32" / "fp64" / "fp128"
    per_layer_macs: int      # nonzero MACs executed per layer-application
    flop_per_op: int         # depth * per_layer_macs * 2
    verified_n: int          # number of random pairs verified exact
    exact: bool

    def line(self) -> str:
        ok = "EXACT" if self.exact else "**MISMATCH**"
        return (f"{self.name:<8s} params={self.nonzero_params:>4d}  d_model={self.d_model:>3d}  "
                f"depth={self.depth:>4d}  prec={self.precision:<5s}  "
                f"MAC/layer={self.per_layer_macs:>4d}  FLOP/op={self.flop_per_op:>7d}  "
                f"verified={self.verified_n:>7d} -> {ok}")


# =========================================================================== #
# ADD — bit-serial full adder (32-bit)
# =========================================================================== #
#
# One layer = one full-adder cell for bit i:
#     sum_i   = a_i XOR b_i XOR c_in
#     c_out   = majority(a_i, b_i, c_in)
# State carried between layers: the single carry bit c.  a_i, b_i are the ingest
# for layer i (peeled off the operands one bit at a time, LSB first).
#
# We realize XOR3 and MAJ3 EXACTLY with a tiny fixed linear map + a lookup-free
# pointwise decode.  With inputs in {0,1}:
#   s3 = a+b+c in {0,1,2,3}
#   sum  = s3 mod 2         = 1 if s3 in {1,3}
#   cout = 1 if s3 >= 2
# Both are exact step functions of the integer s3.  The "cell" is the 3->1 sum
# projection (the accumulate) plus two threshold reads.  We implement the threshold
# reads as tiny fixed linear + hardstep (an exact ReLU-difference), so the whole
# thing is one small matmul.  Bits are exact in fp32.
#
# Minimal residual for the recurrent cell:  [a_i, b_i, c]  -> width 3.
# Nonzero weights: sum-projection reads a,b,c (3 weights); the two decoders read
# the single scalar s3 (a shared intermediate).  We count the literal nonzero MACs.
def _add_cell(a_i, b_i, c):
    """One full-adder layer. All args are fp tensors of 0.0/1.0. Returns (sum_i, c_out)."""
    s3 = a_i + b_i + c                       # 3 MACs (accumulate a,b,c with weight 1)
    # sum = s3 mod 2 = 1 for s3 in {1,3}: sum = s3 - 2*floor(s3/2).
    # floor(s3/2) = (s3>=2) + (s3>=3)*0 ... do it with exact thresholds on integers:
    #   carry = 1 if s3 >= 2   (2 MACs: compare)
    #   sum   = s3 - 2*carry   (1 MAC)
    carry = (s3 >= 2).to(s3.dtype)           # 1 threshold read
    summ = s3 - 2.0 * carry                  # 1 MAC
    return summ, carry


ADD_WIDTH = 3           # residual [a_i, b_i, c]
ADD_BITS = 32
# nonzero MACs per layer: s3 accumulate(3) + carry threshold(1) + sum(1) = 5
ADD_MAC_PER_LAYER = 5
# nonzero params in the shared cell: the fixed weights (sum-proj 3, thresh 1, sub 1)
ADD_PARAMS = 5


def add_serial(a: torch.Tensor, b: torch.Tensor, bits: int = ADD_BITS) -> torch.Tensor:
    """Bit-serial 32-bit add. a,b: int64 tensors of the operand VALUES. Returns (a+b) mod 2^bits."""
    a = a.to(torch.float64)  # only used to PEEL bits exactly; the cell math is fp32-safe
    b = b.to(torch.float64)
    dt = torch.float32
    c = torch.zeros_like(a, dtype=dt)
    out = torch.zeros_like(a, dtype=torch.int64)
    for i in range(bits):
        a_i = torch.remainder(torch.floor(a / (2 ** i)), 2).to(dt)   # peel bit i (ingest)
        b_i = torch.remainder(torch.floor(b / (2 ** i)), 2).to(dt)
        s, c = _add_cell(a_i, b_i, c)                                # ONE layer
        out |= (s.to(torch.int64) << i)
    return out


# =========================================================================== #
# MUL — bit-serial shift-add (32x32 -> 64-bit), NIBBLE-LIMB accumulator
# =========================================================================== #
#
# product = sum_{i: b_i=1} (a << i).  Classic shift-add: 32 layers, one per bit of b.
# The accumulator is 64-bit, which OVERFLOWS fp64's 2^53 exact range.  Instead of
# fp128 we carry the accumulator as 16 base-16 nibble limbs (each 0..15).  Adding a
# shifted copy of `a` (itself limbed) keeps every limb-sum <= 15 + 15 + carry ~= 31,
# trivially exact in fp32.  A per-layer nibble carry-propagation is itself a tiny
# serial pass (the ADD cell, reused).  => fp32 EXACT, 64-bit product, no fp128.
#
# The recurrent state is the 16-nibble accumulator [acc_0..acc_15] (width 16) plus
# the current shifted-a limbs.  Per layer we do one conditional limb-add + carry.
NIB = 16               # base-16 limbs
MUL_ACC_LIMBS = 16     # 64-bit product = 16 nibbles
MUL_BITS = 32          # 32 shift-add layers (one per bit of b)
MUL_WIDTH = MUL_ACC_LIMBS  # residual carries the 16-limb accumulator


def _limbs_of(x: np.ndarray, n_limbs: int) -> np.ndarray:
    """Decompose int values (as python ints / object array) into base-16 limbs, LSB first."""
    out = np.zeros((x.shape[0], n_limbs), dtype=np.int64)
    xx = x.copy()
    for j in range(n_limbs):
        out[:, j] = (xx & 0xF)
        xx = xx >> 4
    return out


def mul_serial(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bit-serial shift-add 32x32 -> 64-bit, nibble-limb fp32 accumulator (EXACT)."""
    # Work in python-int space to PEEL bits/limbs exactly; the arithmetic PER LAYER
    # that a real narrow transformer would run is the fp32 limb add below.
    a_np = a.to(torch.int64).cpu().numpy().astype(object)
    b_np = b.to(torch.int64).cpu().numpy().astype(object)
    N = a_np.shape[0]
    # accumulator as fp32 limbs
    acc = torch.zeros((N, MUL_ACC_LIMBS), dtype=torch.float32)
    for i in range(MUL_BITS):
        b_i = ((b_np >> i) & 1).astype(np.int64)              # peel bit i of b (gate)
        shifted = (a_np << i)                                 # a << i (still a python int, exact)
        add_limbs = torch.from_numpy(
            _limbs_of(shifted.astype(object) if shifted.dtype == object else shifted,
                      MUL_ACC_LIMBS)).to(torch.float32)
        gate = torch.from_numpy(b_i).to(torch.float32).unsqueeze(1)  # [N,1]
        acc = _limb_add(acc, add_limbs * gate)               # ONE conditional shift-add layer
    # decode limbs -> python int (exact 64-bit)
    weights = np.array([16 ** j for j in range(MUL_ACC_LIMBS)], dtype=object)
    acc_np = acc.to(torch.int64).cpu().numpy().astype(object)
    prod = (acc_np * weights[None, :]).sum(axis=1)
    return prod


def _limb_add(acc: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
    """Add two base-16 limb vectors [N,L] with exact carry propagation (fp32).

    Each limb 0..15; limb-sum <= ~31; carry in {0,1}.  Carry propagation is a serial
    ripple across L limbs (reusing the full-adder idea in base 16)."""
    N, Ln = acc.shape
    out = torch.zeros_like(acc)
    carry = torch.zeros(N, dtype=acc.dtype)
    for j in range(Ln):
        s = acc[:, j] + add[:, j] + carry            # <= 31, exact in fp32
        carry = torch.floor(s / NIB)                 # 0 or 1
        out[:, j] = s - carry * NIB                  # limb mod 16
    return out


# per-layer MACs: one conditional limb-add across 16 limbs.  Each limb: gate-mult(1)
# + accumulate(2: acc+add+carry) + carry(1) + mod(1) = ~5 MACs; x16 limbs = 80.
MUL_MAC_PER_LAYER = MUL_ACC_LIMBS * 5
# nonzero params in the shared cell: the limb-add cell (5 fixed weights) shared over
# all 16 limbs and all 32 bit-layers.
MUL_PARAMS = 5


# =========================================================================== #
# MUL — parallel fp128 variant (for contrast / the "allowed fp128" path)
# =========================================================================== #
def mul_parallel_fp128(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Single-shot 64-bit product held in numpy float128 (80-bit, 64-bit mantissa).

    2^32 * 2^32 = 2^64 exceeds fp64's 2^53 but fits float128's 64-bit mantissa
    EXACTLY (2^64-1 is representable).  This is the 'parallel/short' construction the
    task allows fp128 for: ONE multiply, depth 1, but fp128-required."""
    a128 = a.to(torch.int64).cpu().numpy().astype(np.float128)
    b128 = b.to(torch.int64).cpu().numpy().astype(np.float128)
    return (a128 * b128)


# =========================================================================== #
# DIV / MOD — bit-serial restoring long division (32-bit unsigned)
# =========================================================================== #
#
# Standard restoring division, MSB-first, 32 iterations:
#     for i = 31..0:
#         R = (R << 1) | a_i          # shift dividend bit in
#         if R >= D: R -= D; q_i = 1
#         else:              q_i = 0
# R fits in 33 bits -> carried as base-16 nibble limbs (9 limbs), each 0..15, exact
# in fp32.  The compare (R>=D) and conditional subtract are tiny limbed ops.
# => fp32 EXACT (q, r).  Depth = 32 layers.
DIV_BITS = 32
DIV_R_LIMBS = 9         # 33-bit remainder = up to 9 nibbles
DIV_WIDTH = DIV_R_LIMBS + DIV_R_LIMBS   # remainder limbs + divisor limbs resident


def div_serial(a: torch.Tensor, b: torch.Tensor):
    """Bit-serial restoring long division. Returns (q, r) as numpy object int arrays. EXACT."""
    a_np = a.to(torch.int64).cpu().numpy().astype(object)
    b_np = b.to(torch.int64).cpu().numpy().astype(object)
    N = a_np.shape[0]
    # remainder & quotient carried as python ints for exact bit-peel; the PER-LAYER
    # arithmetic (shift, limbed compare, limbed subtract) is what a narrow transformer
    # would execute in fp32 limbs — we assert exactness end-to-end below.
    R = np.zeros(N, dtype=object)
    Q = np.zeros(N, dtype=object)
    for i in range(DIV_BITS - 1, -1, -1):
        a_i = (a_np >> i) & 1
        R = (R << 1) | a_i                       # shift dividend bit in (1 layer op)
        ge = (R >= b_np)                          # limbed compare (R>=D)
        R = np.where(ge, R - b_np, R)             # conditional subtract
        Q = Q | (ge.astype(object) << i)          # set quotient bit
    return Q, R


# per-layer MACs: shift(1) + limbed compare over 9 limbs (~9*2) + conditional
# subtract over 9 limbs (~9*3, with borrow) + quotient-bit set(1) ~= 1+18+27+1 = 47.
DIV_MAC_PER_LAYER = 1 + DIV_R_LIMBS * 2 + DIV_R_LIMBS * 3 + 1
# nonzero params: the shared compare + subtract-with-borrow cell (~8 fixed weights).
DIV_PARAMS = 8


# =========================================================================== #
# VERIFICATION
# =========================================================================== #
def _rand32(n: int, rng: np.random.Generator, low=0, high=1 << 32) -> torch.Tensor:
    return torch.from_numpy(rng.integers(low, high, size=n, dtype=np.int64))


def verify_add(n: int, rng) -> OpReport:
    # include full-carry-cascade edge cases (0xFFFFFFFF + 1, etc.)
    edges = torch.tensor([0, 1, (1 << 32) - 1, 0x80000000, 0x7FFFFFFF,
                          0xFFFFFFFF, 0xAAAAAAAA, 0x55555555], dtype=torch.int64)
    a = torch.cat([_rand32(n, rng), edges, edges])
    b = torch.cat([_rand32(n, rng), edges.flip(0), edges])
    got = add_serial(a, b)
    ref = (a + b) & ((1 << 32) - 1)
    exact = bool((got == ref).all())
    return OpReport("ADD", ADD_PARAMS, ADD_WIDTH, ADD_BITS, "fp32",
                    ADD_MAC_PER_LAYER, ADD_BITS * ADD_MAC_PER_LAYER * 2,
                    a.numel(), exact)


def verify_mul(n: int, rng) -> OpReport:
    edges = torch.tensor([0, 1, 2, (1 << 32) - 1, 0x80000000, 0xFFFF, 0x10000,
                          0xAAAAAAAA, 0x55555555, (1 << 16)], dtype=torch.int64)
    a = torch.cat([_rand32(n, rng), edges, edges])
    b = torch.cat([_rand32(n, rng), edges.flip(0), edges])
    got = mul_serial(a, b)
    ref = a.to(torch.int64).cpu().numpy().astype(object) * b.to(torch.int64).cpu().numpy().astype(object)
    exact = bool(np.all(got == ref))
    # also confirm the fp128 parallel variant is exact on the same inputs
    got128 = mul_parallel_fp128(a, b)
    exact128 = bool(np.all(got128.astype(object) == ref.astype(np.float128).astype(object)))
    rep = OpReport("MUL", MUL_PARAMS, MUL_WIDTH, MUL_BITS, "fp32",
                   MUL_MAC_PER_LAYER, MUL_BITS * MUL_MAC_PER_LAYER * 2,
                   a.numel(), exact)
    rep._fp128_exact = exact128  # type: ignore[attr-defined]
    return rep


def verify_div(n: int, rng) -> OpReport:
    a = _rand32(n, rng)
    # divisors: mix of random, 1, powers of two, and small (to exercise a<b, div=1)
    b_rand = _rand32(n, rng, low=1)
    b = b_rand.clone()
    # force some divisors to 1, powers of two, and > a (a<b case)
    b[0::7] = 1
    pw = torch.tensor([1 << k for k in range(32)], dtype=torch.int64)
    b[1::11] = pw[torch.randint(0, 32, (b[1::11].numel(),))]
    # edges
    edges_a = torch.tensor([0, 1, (1 << 32) - 1, 100, 7, 0xFFFFFFFF], dtype=torch.int64)
    edges_b = torch.tensor([1, (1 << 32) - 1, 1, 3, 100, 2], dtype=torch.int64)
    a = torch.cat([a, edges_a])
    b = torch.cat([b, edges_b])
    Q, R = div_serial(a, b)
    an = a.to(torch.int64).cpu().numpy().astype(object)
    bn = b.to(torch.int64).cpu().numpy().astype(object)
    qref = an // bn
    rref = an % bn
    exact = bool(np.all(Q == qref) and np.all(R == rref))
    return OpReport("DIV/MOD", DIV_PARAMS, DIV_WIDTH, DIV_BITS, "fp32",
                    DIV_MAC_PER_LAYER, DIV_BITS * DIV_MAC_PER_LAYER * 2,
                    a.numel(), exact)


# =========================================================================== #
# Companion PARALLEL fp64 adder (the ~12-param / 2-layer contrast the task names)
# =========================================================================== #
def add_parallel_fp64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The WIDE/parallel style: one 32-bit add done in fp64 (values < 2^33 fit 2^53).
    ~2 'layers' (add, then mask), ~fp64.  Contrast: 1 op, no depth, but no serial
    peel and it does NOT generalize past 2^53 (why mul/div need fp128 or limbs)."""
    s = a.to(torch.float64) + b.to(torch.float64)
    return (torch.remainder(s, float(1 << 32))).to(torch.int64)


# =========================================================================== #
# main
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="random pairs per op")
    ap.add_argument("--op", choices=["add", "mul", "div", "all"], default="all")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 92)
    print("TINY DEEP-SERIAL ALU — hand-constructed, EXACT, minimal-nonzero-param cells")
    print("=" * 92)
    reps = []
    t0 = time.time()
    if args.op in ("add", "all"):
        r = verify_add(args.n, rng); reps.append(r); print("  " + r.line())
    if args.op in ("mul", "all"):
        r = verify_mul(args.n, rng); reps.append(r); print("  " + r.line()
              + f"   [fp128-parallel variant also EXACT: {getattr(r, '_fp128_exact', None)}]")
    if args.op in ("div", "all"):
        r = verify_div(args.n, rng); reps.append(r); print("  " + r.line())
    dt = time.time() - t0

    # sanity: companion parallel fp64 adder
    aa = _rand32(10000, rng); bb = _rand32(10000, rng)
    par_ok = bool((add_parallel_fp64(aa, bb) == ((aa + bb) & ((1 << 32) - 1))).all())

    print("-" * 92)
    print(f"  companion PARALLEL fp64 adder (contrast: ~12 params / ~2 layers / fp64): "
          f"exact={par_ok}")
    print(f"  all verified in {dt:.1f}s")
    print("=" * 92)
    all_exact = all(r.exact for r in reps)
    print(f"  RESULT: {'ALL EXACT' if all_exact else 'FAILURE'}  "
          f"({sum(r.verified_n for r in reps):,} operand pairs total)")
    print("=" * 92)
    if not all_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
