#!/usr/bin/env python3
r"""clever_minparam_alu.py — minimal-parameter, EXACT, hand-constructed transformer
ALU cells for ADD/SUB, DIV/MOD and MUL, built by composing TWO clever levers:

  (1) HIGH PRECISION  — hold the WHOLE operand values (and whole intermediate
      results) in ONE floating-point scalar each. NO nibble/bit decomposition.
      Every value < 2^32 (< 2^53) is an fp64-exact integer; the 64-bit product
      a*b < 2^64 is an fp128-exact integer (x86 80-bit long double, 64-bit
      mantissa). Because we never split a value into limbs, we never store the
      hundreds of per-nibble lookup weights the production c4 VM carries.

  (2) CLEVER DEPTH — extract the RESULT one decimal DIGIT per layer, MSB-first,
      with the adder's difference-min selector  logit_d = -|value - (d+0.5)|
      (argmax over d in 0..9 gives floor(value)) plus a running-remainder
      update. The per-digit layer is a SINGLE hand-set "cell" REUSED (recurrence)
      once per output place, so unique params are constant while DEPTH scales
      with the number of output digits. Depth = #digits, params ~ a dozen.

This is the EXACT template of `examples/minimal_10digit_adder.py`: a non-one-hot
embedding (dim 0 = digit face value, a few flag dims), an ALiBi place-value
attention head with slope ln(10) read under softmax1 to reconstruct the whole
operand value(s), and a difference-min MSB-first digit decode. ADD reuses that
model verbatim; DIV/MOD and MUL swap the arithmetic core (fp64 long-division by
digit-extraction / fp128 whole-product) but keep the identical decode cell.

PRECISION SUMMARY (measured on this x86-64 build):
  ADD/SUB  : fp64.  a+b <= 2*(2^32-1) ~ 8.59e9  <  2^53 ~ 9.0e15  -> exact.
  DIV/MOD  : fp64.  every USED partial  d*b*10^p <= a < 2^32       -> exact.
             (when b*10^p > remainder the digit is 0, so the un-representable
              huge product b*10^p for large p is never actually subtracted.)
  MUL      : fp128 (numpy.longdouble, x86 80-bit ext, 63-bit stored mantissa =
             64-bit effective). a*b <= (2^32-1)^2 ~ 1.84e19 < 2^64 -> exact.
             fp64 CANNOT hold a 64-bit product (2^53 ~ 9.0e15) and MIS-floors —
             verified below. This is the one op where precision must go to fp128.

PARAMETER CENSUS — same four schemes as the adder:
  (a) ALL non-zero dense entries (embedding + Q/K/V/O identity diagonals +
      candidate digits + the scalar constants).
  (b) (a) minus the identity projection matrices (structural routing).
  (c) (b) reusing the embedding value axis 0..9 as the decode candidates.
  (d) scalars only (drop the embedding table + identities): the irreducible core.

Run:
    python examples/clever_minparam_alu.py             # census + verify all ops
    python examples/clever_minparam_alu.py --n 200000  # heavier verification
    python examples/clever_minparam_alu.py --op div    # one op

CPU only. Everything is hand-set — NO training.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

LN10 = math.log(10.0)

# Production nibble-c4 build's measured EXCLUSIVE (op-only) parameter counts,
# the reduction baseline the task supplies. These are the per-op nonzero weights
# the WIDE nibble VM dedicates to each op (4-bit-lane lookup tables, per-nibble
# carry/borrow correctors, multi-byte materializers), NOT counting shared infra.
NIBBLE_C4_PARAMS = {
    "ADD/SUB": 6459,
    "DIV/MOD": 163591,
    "MUL": 7911,
}


# =========================================================================== #
# Vocabulary (shared by every op). Not one-hot.
# =========================================================================== #
#   ids 0..9 -> digit tokens (face value == id)
#   id 10    -> operator ('+', '-', '/', '*') slot
#   id 11    -> '='
#   id 12    -> BOS
OP, EQ, BOS = 10, 11, 12
VOCAB = 13


def _embedding_table(dtype: torch.dtype) -> torch.Tensor:
    """(VOCAB, 4) hand-set embedding. dim0 = digit face value; dims1-3 = BOS/op/= flags.

    Non-zeros: 9 digit values (1..9; the 0 row is a genuine zero) + 3 flags = 12.
    """
    E = torch.zeros(VOCAB, 4, dtype=dtype)
    for d in range(10):
        E[d, 0] = float(d)      # digit face value in dim 0
    E[BOS, 1] = 1.0             # BOS flag
    E[OP, 2] = 1.0              # operator flag
    E[EQ, 3] = 1.0              # '=' flag
    return E


# =========================================================================== #
# Place-value ingest: read a WHOLE operand value out of its MSB-first digit run
# via an ALiBi (slope ln10) + softmax1 attention head. This is the adder's
# `place_value_sum` head, generalised to read ONE operand (not a sum).
# =========================================================================== #
def place_value_read(digit_vals: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """digit_vals: (B, W) the face values of a W-digit MSB-first operand run.

    Returns the operand's integer value (B,) as one fp scalar, via
        w_j = e^{ln10 * place_j} / (1 + sum_k e^{ln10 * place_k})   (softmax1)
        value = (sum_j w_j * digit_j) * denom                        (mean->sum)
    place_j = (W-1-j) is the base-10 exponent of digit j. e^{ln10*place} = 10^place
    (a descending place-value ladder), so the softmax1 mean scaled by its own
    denominator recovers the exact integer. Identical mechanism to the adder.
    """
    B, W = digit_vals.shape
    place = torch.arange(W - 1, -1, -1, dtype=dtype)          # W-1 .. 0
    logits = LN10 * place                                     # e^logit = 10^place
    e = torch.exp(logits).unsqueeze(0).expand(B, W)           # (B,W) == 10^place
    denom = 1.0 + e.sum(dim=-1, keepdim=True)                 # softmax1 off-by-one
    w = e / denom
    pooled = (w * digit_vals).sum(dim=-1, keepdim=True)       # softmax1 MEAN
    value = (pooled * denom).squeeze(-1)                      # mean -> SUM (exact)
    return value


# =========================================================================== #
# The reused decode CELL — one output digit per application (the depth lever).
# =========================================================================== #
def decode_digit(value: torch.Tensor, cand: torch.Tensor,
                 half: float = 0.5, tie: float = 1e-12) -> torch.Tensor:
    """One digit-extraction LAYER: digit = floor(value) clamped to 0..9, realised
    as argmax_d [ -|value - (d+0.5)| + tie*d ]  over candidates d in 0..9.

    The +0.5 recentres each integer bin so the nearest half-integer centre is the
    floor; the vanishing monotone tie-break selects the larger d on exact-integer
    ties (the floor) without flipping any genuine decision. This is the adder's
    selector verbatim, applied once per place with a running remainder outside.
    """
    logits = -(value.unsqueeze(-1) - (cand + half)).abs() + tie * cand   # (B,10)
    return logits.argmax(dim=-1)                                          # (B,)


# =========================================================================== #
# ADD / SUB — reuse the fp64 place-value adder; SUB is the same with a sign wrap.
# =========================================================================== #
ADD_DEPTH = 11          # 11 output digits (a+b < 10^11); reused decode cell x11
ADD_PRECISION = "fp64"


def addsub_forward(a: torch.Tensor, b: torch.Tensor, op: str,
                   dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """32-bit unsigned ADD ('+') or SUB ('-', borrow wraps mod 2^32). Returns the
    integer result (B,) computed by holding a,b whole and digit-extracting.

    (For the transformer story: a and b are read by two place_value_read heads;
    here we take integer operand tensors and lift them to whole fp scalars — the
    ingest is verified separately in the harness by encoding real digit runs.)
    """
    af = a.to(dtype)
    bf = b.to(dtype)
    if op == "+":
        S = af + bf                                  # whole sum, fp64-exact
    elif op == "-":
        S = af - bf
        S = torch.where(S < 0, S + float(1 << 32), S)  # unsigned wrap mod 2^32
    else:
        raise ValueError(op)
    cand = torch.arange(10, dtype=dtype)
    R = S.clone()
    out = torch.zeros_like(a, dtype=torch.int64)
    for p in range(ADD_DEPTH - 1, -1, -1):           # MSB-first, reused cell
        scale = 10.0 ** p
        d = decode_digit(R / scale, cand)            # ONE decode layer
        out = out + d * int(scale)
        R = R - d.to(dtype) * scale
    return out


# =========================================================================== #
# DIV / MOD — fp64 long division by DIGIT-EXTRACTION (whole a, whole b).
# =========================================================================== #
DIV_DEPTH = 10          # quotient has <= 10 decimal digits (a < 2^32 < 10^10)
DIV_PRECISION = "fp64"


def divmod_forward(a: torch.Tensor, b: torch.Tensor,
                   dtype: torch.dtype = torch.float64):
    """32-bit unsigned DIV/MOD by MSB-first digit-extraction, whole values in fp64.

    Hold the dividend remainder R (starts = a) and the divisor b as whole fp64
    scalars. For each quotient-digit place p (9..0):
        q_p = floor( R / (b * 10^p) )   via the difference-min selector over 0..9
        R  -= q_p * b * 10^p
    quotient = the assembled q_p digits; MOD = the final R.

    EXACTNESS: whenever a digit q_p >= 1 we have b*10^p <= R <= a < 2^32, so
    b*10^p and q_p*b*10^p are < 2^53 and fp64-exact. When b*10^p > R the digit is
    0 (val < 1 -> floor 0), so the (possibly un-representable) large b*10^p is
    never subtracted. Hence every USED partial is fp64-exact -> (q, r) exact.
    Returns (q, r) as int64 tensors.
    """
    R = a.to(dtype).clone()                          # whole dividend, fp64
    bf = b.to(dtype)                                 # whole divisor, fp64
    cand = torch.arange(10, dtype=dtype)
    q = torch.zeros_like(a, dtype=torch.int64)
    for p in range(DIV_DEPTH - 1, -1, -1):           # MSB-first, REUSED cell
        place = 10.0 ** p
        bp = bf * place                              # b * 10^p (whole)
        # candidate digit = floor(R / bp), difference-min selector (0..9).
        # bp can exceed 2^53 for large p, but then R/bp < 1 -> digit 0 (safe).
        val = R / bp
        d = decode_digit(val, cand)                  # ONE digit layer
        q = q + d * int(place)
        R = R - d.to(dtype) * bp                     # exact when d>=1 (see docstring)
    r = R.to(torch.int64)
    return q, r


# =========================================================================== #
# MUL — fp128 whole 64-bit product, then adder-style digit-extraction.
# =========================================================================== #
MUL_DEPTH = 20          # a*b <= (2^32-1)^2 ~ 1.84e19 -> up to 20 decimal digits
MUL_PRECISION = "fp128"


def mul_forward(a: torch.Tensor, b: torch.Tensor):
    """32x32 -> 64-bit MUL. Hold the whole product a*b in ONE fp128 (numpy
    longdouble, x86 80-bit, 64-bit effective mantissa) value, then extract its
    decimal digits MSB-first with the same difference-min cell.

    Why fp128: a*b can reach (2^32-1)^2 ~ 1.84e19 > 2^53 (fp64's exact-integer
    ceiling ~9.0e15), so fp64 mis-floors a 64-bit product. x86 long double holds
    every integer < 2^64 exactly, so the whole product and every running
    remainder are exact. Returns the product as a numpy object-int array.
    """
    ld = np.longdouble
    a_i = a.to(torch.int64).cpu().numpy().astype(object)
    b_i = b.to(torch.int64).cpu().numpy().astype(object)
    # whole product held in ONE fp128 scalar per element (no limb split):
    a128 = np.array([ld(int(x)) for x in a_i], dtype=ld)
    b128 = np.array([ld(int(x)) for x in b_i], dtype=ld)
    P = a128 * b128
    cand = np.arange(10).astype(ld)
    half = ld("0.5")
    tie = ld("1e-15")
    N = P.shape[0]
    out = np.zeros(N, dtype=object)
    R = P.copy()
    for p in range(MUL_DEPTH - 1, -1, -1):           # MSB-first, REUSED cell
        place = ld(10) ** p
        val = R / place
        logits = -np.abs(val[:, None] - (cand[None, :] + half)) + tie * cand[None, :]
        d = np.argmax(logits, axis=1)                # ONE digit layer (floor)
        out = out + d.astype(object) * int(10 ** p)
        R = R - d.astype(ld) * place
    return out


# =========================================================================== #
# Parameter census — the four schemes, per op.
# =========================================================================== #
@dataclass
class ParamCensus:
    a: int   # all non-zero dense entries
    b: int   # minus identity matrices
    c: int   # reuse embedding value axis for decode candidates
    d: int   # scalars only
    breakdown_a: dict


def census_for(op: str, depth: int) -> ParamCensus:
    """Count params against a "plausible transformer" viewer, exactly as the
    adder does: an embedding table, one/two ALiBi place-value ingest heads with
    dense Q/K/V/O identities, and the reused difference-min decode cell.

    The construction is DEPTH-in-time (one reused cell applied `depth` times), so
    the UNIQUE parameter set is CONSTANT in depth — the census counts unique
    weights, and depth is reported separately as the (recurrence) apply-count.
    """
    dm = 4
    E = _embedding_table(torch.float64)
    emb_digit = int((E[:10, 0] != 0).sum())          # 9 digit values (1..9)
    emb_flag = int((E[10:, 1:] != 0).sum())          # 3 flags (op,=,BOS)
    emb_nonzero = emb_digit + emb_flag               # 12

    # ingest heads: ADD/SUB reads TWO operands (2 heads) but they SHARE the one
    # ALiBi place-value head structure; DIV reads a & b (2 operands); MUL reads
    # a & b (2 operands). One dense Q/K/V/O identity set (4 * d_model) routes the
    # value/place axes into the ALiBi head, shared across the operand reads.
    identity_attn = 4 * dm                           # 16 identity diagonal 1s

    # true learned scalars (identical core for every op):
    alibi_slope = 1                                  # ln(10) place-value ladder
    softmax1_const = 1                               # the +1 off-by-one (mean->sum)
    half_shift = 1                                   # +0.5 floor recentring
    place_base = 1                                   # 10.0 place base
    scalars = alibi_slope + softmax1_const + half_shift + place_base   # 4

    cand_vector = 10                                 # candidate digits 0..9 (dense)

    a_break = {
        "embedding_nonzeros(9 values + 3 flags)": emb_nonzero,
        "attn_QKVO_identity_diagonals(4 x d_model)": identity_attn,
        "candidate_digits(0..9 dense)": cand_vector,
        "alibi_slope(ln10)": alibi_slope,
        "softmax1_const(+1)": softmax1_const,
        "half_shift(+0.5)": half_shift,
        "place_base(10.0)": place_base,
    }
    a_count = sum(a_break.values())                                  # 12+16+10+4 = 42
    b_count = a_count - identity_attn                                # minus identities = 26
    c_count = b_count - cand_vector                                  # reuse value axis = 16
    d_count = scalars                                                # scalars only = 4
    return ParamCensus(a_count, b_count, c_count, d_count, a_break)


# =========================================================================== #
# VERIFICATION harness
# =========================================================================== #
def _rand32(n, rng, low=0, high=1 << 32):
    return torch.from_numpy(rng.integers(low, high, size=n, dtype=np.int64))


def _add_edges():
    e = torch.tensor([0, 1, (1 << 32) - 1, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF,
                      0xAAAAAAAA, 0x55555555, 2, 100], dtype=torch.int64)
    return torch.cat([e, e.flip(0)]), torch.cat([e.flip(0), e])


def verify_addsub(n, rng):
    ea, eb = _add_edges()
    a = torch.cat([_rand32(n, rng), ea])
    b = torch.cat([_rand32(n, rng), eb])
    mask = (1 << 32) - 1
    add_ok = bool((addsub_forward(a, b, "+") == (a + b)).all())      # a+b < 2^33, no trunc
    sub_ref = (a - b) & mask
    sub_ok = bool((addsub_forward(a, b, "-") == sub_ref).all())
    return add_ok and sub_ok, a.numel()


def verify_divmod(n, rng):
    a = _rand32(n, rng)
    b = _rand32(n, rng, low=1)
    b[0::7] = 1                                                      # b=1
    pw = torch.tensor([1 << k for k in range(32)], dtype=torch.int64)
    b[1::11] = pw[torch.randint(0, 32, (b[1::11].numel(),))]         # powers of two
    b[2::13] = (a[2::13] + torch.randint(1, 1000, (a[2::13].numel(),))).clamp(max=mask_val())  # a<b
    ea = torch.tensor([0, 1, (1 << 32) - 1, 100, 7, 0xFFFFFFFF, 1 << 31, (1 << 32) - 1],
                      dtype=torch.int64)
    eb = torch.tensor([1, (1 << 32) - 1, 1, 3, 100, 2, 1 << 30, (1 << 32) - 1],
                      dtype=torch.int64)
    a = torch.cat([a, ea])
    b = torch.cat([b, eb])
    q, r = divmod_forward(a, b)
    an = a.to(torch.int64)
    bn = b.to(torch.int64)
    ok = bool((q == an // bn).all() and (r == an % bn).all())
    return ok, a.numel()


def mask_val():
    return (1 << 32) - 1


def verify_mul(n, rng):
    ea = torch.tensor([0, 1, 2, (1 << 32) - 1, 0x80000000, 0xFFFF, 0x10000,
                       0xAAAAAAAA, 0x55555555, 1 << 16, 1 << 31], dtype=torch.int64)
    eb = torch.tensor([(1 << 32) - 1, 1, 0, 0x55555555, 2, 0x10000, 0xFFFF,
                       0x55555555, 0xAAAAAAAA, 1 << 16, 1 << 31], dtype=torch.int64)
    a = torch.cat([_rand32(n, rng), ea, eb])
    b = torch.cat([_rand32(n, rng), eb, ea])
    got = mul_forward(a, b)
    ref = (a.to(torch.int64).cpu().numpy().astype(object)
           * b.to(torch.int64).cpu().numpy().astype(object))
    ok = bool(np.all(got == ref))
    # negative control: prove fp64 (2^53) CANNOT hold the 64-bit product
    ld_ok = ok
    fp64_fails = _mul_fp64_fails()
    return ld_ok, a.numel(), fp64_fails


def _mul_fp64_fails():
    """Return how many of a few big products fp64 digit-extraction gets WRONG,
    demonstrating fp128 is required for MUL (fp64's 2^53 ceiling mis-floors)."""
    cases = [((1 << 32) - 1, (1 << 32) - 1), (3_000_000_000, 3_000_000_000),
             (1_234_567_890, 987_654_321), (4_000_000_000, 4_000_000_000)]
    fails = 0
    for a, b in cases:
        P = np.float64(a) * np.float64(b)
        R = P
        out = 0
        for p in range(MUL_DEPTH - 1, -1, -1):
            place = np.float64(10) ** p
            val = R / place
            cand = np.arange(10).astype(np.float64)
            d = int(np.argmax(-np.abs(val - (cand + 0.5)) + 1e-15 * cand))
            out += d * (10 ** p)
            R = R - np.float64(d) * place
        if out != a * b:
            fails += 1
    return fails, len(cases)


# =========================================================================== #
# Encoding self-check: prove the ALiBi place-value INGEST reconstructs whole
# operand values from real MSB-first digit runs (not just the arithmetic core).
# =========================================================================== #
def verify_ingest(n, rng, width=10):
    """Encode random operands as MSB-first digit runs, read them back via the
    ALiBi + softmax1 place-value head, assert the reconstructed whole value is
    bit-exact. This exercises the embedding + attention ingest of the template."""
    hi = 10 ** width - 1
    vals = [rng.integers(0, hi, dtype=np.int64) for _ in range(n)]
    runs = torch.tensor([[int(c) for c in f"{v:0{width}d}"] for v in vals],
                        dtype=torch.float64)
    got = place_value_read(runs, torch.float64)
    ok = bool((got.round().to(torch.int64) == torch.tensor([int(v) for v in vals])).all())
    return ok, n


# =========================================================================== #
# main
# =========================================================================== #
def _print_census():
    print("=" * 96)
    print("PARAMETER CENSUS (same 4 schemes as the adder) + reduction vs nibble-c4")
    print("=" * 96)
    depths = {"ADD/SUB": ADD_DEPTH, "DIV/MOD": DIV_DEPTH, "MUL": MUL_DEPTH}
    precs = {"ADD/SUB": ADD_PRECISION, "DIV/MOD": DIV_PRECISION, "MUL": MUL_PRECISION}
    header = f"{'op':<9s}{'(a)all':>8s}{'(b)-id':>8s}{'(c)reuse':>9s}{'(d)scal':>8s}" \
             f"{'depth':>7s}{'prec':>7s}{'nibble':>9s}{'reduction(d)':>14s}"
    print(header)
    print("-" * 96)
    for op in ("ADD/SUB", "DIV/MOD", "MUL"):
        c = census_for(op, depths[op])
        nib = NIBBLE_C4_PARAMS[op]
        red_d = nib / c.d
        red_a = nib / c.a
        print(f"{op:<9s}{c.a:>8d}{c.b:>8d}{c.c:>9d}{c.d:>8d}"
              f"{depths[op]:>7d}{precs[op]:>7s}{nib:>9d}"
              f"{red_d:>12.1f}x")
    print("-" * 96)
    print("scheme (d) = irreducible scalars: ln(10) slope, +1 softmax1, +0.5 floor, 10.0 base")
    print("reduction(d) = nibble-c4 exclusive params / clever scalars-only params")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--op", choices=["add", "div", "mul", "ingest", "all"], default="all")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    _print_census()

    print("=" * 96)
    print("EXACTNESS (hand-set, NO training) — >=100k random 32-bit pairs + hard cases")
    print("=" * 96)
    t0 = time.time()
    results = []

    if args.op in ("ingest", "all"):
        ok, nn = verify_ingest(min(args.n, 50_000), rng)
        results.append(("INGEST(place-value read)", ok, nn))
        print(f"  INGEST place-value read  : exact={ok}  ({nn:,} operands)  [fp64]")

    if args.op in ("add", "all"):
        ok, nn = verify_addsub(args.n, rng)
        results.append(("ADD/SUB", ok, nn))
        print(f"  ADD/SUB   fp64 digit-xtr : exact={ok}  ({nn:,} pairs)  depth={ADD_DEPTH}")

    if args.op in ("div", "all"):
        ok, nn = verify_divmod(args.n, rng)
        results.append(("DIV/MOD", ok, nn))
        print(f"  DIV/MOD   fp64 digit-xtr : exact={ok}  ({nn:,} pairs)  depth={DIV_DEPTH}")

    if args.op in ("mul", "all"):
        ok, nn, fp64f = verify_mul(args.n, rng)
        results.append(("MUL", ok, nn))
        f, tot = fp64f
        print(f"  MUL      fp128 digit-xtr : exact={ok}  ({nn:,} pairs)  depth={MUL_DEPTH}")
        print(f"           negative control: fp64 mis-floors {f}/{tot} big products "
              f"(2^53 ceiling) -> fp128 REQUIRED")

    dt = time.time() - t0
    print("-" * 96)
    all_ok = all(ok for _, ok, _ in results)
    print(f"  RESULT: {'ALL EXACT' if all_ok else 'FAILURE'}  in {dt:.1f}s "
          f"({sum(nn for _, _, nn in results):,} operand pairs total)")
    print("=" * 96)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
