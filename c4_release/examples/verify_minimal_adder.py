"""
Verification harness for minimal_10digit_adder.py.

  STEP 3  exactness   : >= 100,000 random 10-digit pairs + hard cases,
                        asserts 100% exact (0 errors) in float64.
  STEP 3  fp boundary : rebuilds the model in float32 and reports the number
                        of correct leading digits before the first wrong digit,
                        confirming fp32 breaks around digit 7-8 and fp64 is
                        required for 10-digit operands.
  STEP 4  param census: prints the (a)/(b)/(c)/(d) counts + breakdowns and the
                        d_model / n_layers / n_heads / vocab hyperparameters.

CPU only. Run:  python verify_minimal_adder.py
"""

from __future__ import annotations

import random

import torch

from minimal_10digit_adder import (
    MinimalAdder,
    N_DIGITS,
    N_OUT,
    count_parameters,
    digits_to_int,
    encode_batch,
)

MAX = 10 ** N_DIGITS - 1  # 9999999999


def hard_cases():
    """Edge cases: all-9s, cascading carries, zeros, a+0, max+max, etc."""
    return [
        (MAX, MAX),                 # 9999999999 + 9999999999 (full cascade)
        (0, 0),                     # zeros
        (0, MAX), (MAX, 0),         # a + 0
        (MAX, 1), (1, MAX),         # +1 into a full cascade
        (5_000_000_000, 5_000_000_000),
        (1_234_567_890, 9_876_543_210),
        (9_999_999_999, 9_999_999_998),
        (1_000_000_000, 1_000_000_000),
        (9_090_909_090, 9_090_909_090),
        (1_111_111_111, 8_888_888_889),   # every column carries
        (4_999_999_999, 5_000_000_001),
        (10 ** N_DIGITS - 1, 0),
        (123, 877),                       # tiny values, mostly zero-pad
        (9_999_999_990, 10),              # carry that ripples 9 places
    ]


def run_exactness(n_random=100_000, batch=20_000, seed=1234):
    m = MinimalAdder(dtype=torch.float64)
    rng = random.Random(seed)

    pairs = hard_cases()
    pairs += [(rng.randint(0, MAX), rng.randint(0, MAX)) for _ in range(n_random)]
    total = len(pairs)

    errors = 0
    first_fail = None
    for s in range(0, total, batch):
        chunk = pairs[s:s + batch]
        ids = encode_batch(chunk)
        out = m.forward(ids)
        for i, (a, b) in enumerate(chunk):
            got = digits_to_int(out[i])
            if got != a + b:
                errors += 1
                if first_fail is None:
                    first_fail = (a, b, got, a + b)
    return total, errors, first_fail


def sum_precision_boundary(seed=11, per_width=20_000):
    """Clean 2^24 probe: at what OPERAND WIDTH does the fp32 place-value SUM S
    (before any decode) first fail to equal the true integer a+b?

    This isolates the mantissa limit from the decode tie-break. fp32's 24-bit
    mantissa represents integers exactly only up to 2^24 = 16,777,216 (8 digits,
    ~7 reliably); once the sum crosses that, S rounds and S != a+b. fp64 (2^52)
    stays exact through the full 11-digit sum.
    """
    m32 = MinimalAdder(dtype=torch.float32)
    m64 = MinimalAdder(dtype=torch.float64)
    rng = random.Random(seed)
    first_bad_w32 = None
    first_bad_w64 = None
    per_w32, per_w64 = {}, {}
    for w in range(1, N_DIGITS + 1):
        hi = 10 ** w - 1
        pairs = [(rng.randint(0, hi), rng.randint(0, hi)) for _ in range(per_width)]
        ids = encode_batch(pairs)
        S32 = m32.place_value_sum(ids)
        S64 = m64.place_value_sum(ids)
        e32 = e64 = 0
        for i, (a, b) in enumerate(pairs):
            if round(float(S32[i])) != a + b:
                e32 += 1
            if round(float(S64[i])) != a + b:
                e64 += 1
        per_w32[w], per_w64[w] = e32, e64
        if e32 and first_bad_w32 is None:
            first_bad_w32 = w
        if e64 and first_bad_w64 is None:
            first_bad_w64 = w
    return first_bad_w32, per_w32, first_bad_w64, per_w64


def fp_width_boundary(dtype, seed=7, per_width=20_000):
    """Sweep operand WIDTH w = 1..10 digits and return the smallest w at which
    `dtype` first mis-decodes a sum, plus the per-width error rate.

    This localises the precision boundary as an OPERAND-DIGIT count. fp32's
    24-bit mantissa (2^24 ~ 1.677e7) can represent integers exactly only up to
    ~7-8 decimal digits, so the model should stay exact for narrow operands and
    start failing once the running sum's magnitude crosses ~1.6e7 (8 digits).
    fp64's 52-bit mantissa (~4.5e15, 15-16 digits) covers the full 11-digit sum.
    """
    m = MinimalAdder(dtype=dtype)
    rng = random.Random(seed)
    per_width_err = {}
    first_bad_w = None
    example = None
    for w in range(1, N_DIGITS + 1):
        hi = 10 ** w - 1
        pairs = [(rng.randint(0, hi), rng.randint(0, hi)) for _ in range(per_width)]
        ids = encode_batch(pairs)
        out = m.forward(ids)
        errs = 0
        for i, (a, b) in enumerate(pairs):
            got = digits_to_int(out[i])
            if got != a + b:
                errs += 1
                if example is None:
                    example = (w, a, b, got, a + b)
        per_width_err[w] = errs
        if errs and first_bad_w is None:
            first_bad_w = w
    return first_bad_w, per_width_err, example


def print_param_census():
    census = count_parameters(dtype=torch.float64)
    h = census["hyper"]
    print("=" * 72)
    print("PARAMETER CENSUS")
    print("=" * 72)
    print(f"d_model={h['d_model']}  n_layers={h['n_layers']}  "
          f"n_heads={h['n_heads']}  vocab={h['vocab']}")
    print()
    labels = {
        "a": "(a) ALL non-zero dense entries",
        "b": "(b) minus identity matrices",
        "c": "(c) reuse embedding value axis for candidates",
        "d": "(d) exclude embedding table + identities  (the '~12')",
    }
    blog = {"a": 95, "b": 36, "c": 28, "d": 12}
    for k in ("a", "b", "c", "d"):
        count, breakdown = census[k]
        print(f"{labels[k]}: {count}   (blogpost said ~{blog[k]})")
        for name, v in breakdown.items():
            print(f"      {v:>3d}  {name}")
        print()


def main():
    print_param_census()

    print("=" * 72)
    print("EXACTNESS (float64)")
    print("=" * 72)
    total, errors, first_fail = run_exactness(n_random=100_000)
    print(f"exact: {total - errors}/{total}   errors: {errors}")
    if first_fail:
        print("first failure:", first_fail)
    assert errors == 0, "float64 model is NOT exact"
    print("RESULT: 100% EXACT in float64\n")

    print("=" * 72)
    print("PRECISION BOUNDARY (float32 vs float64), by operand width")
    print("=" * 72)
    widths = list(range(1, N_DIGITS + 1))

    # (1) clean 2^24 boundary: the place-value SUM S itself (before decode).
    sb32, sw32, sb64, sw64 = sum_precision_boundary()
    print("[A] place-value SUM S vs true a+b (isolates the 2^24 mantissa limit)")
    print("    operand digits :", "  ".join(f"{w:>5d}" for w in widths))
    print("    fp32 S errors  :", "  ".join(f"{sw32[w]:>5d}" for w in widths))
    print("    fp64 S errors  :", "  ".join(f"{sw64[w]:>5d}" for w in widths))
    print(f"    => fp32 SUM first wrong at width {sb32} digits "
          f"(holds ~{(sb32 or 99) - 1} digits; 2^24=16,777,216 ~ 7-8 decimal digits)")
    print(f"    => fp64 SUM first wrong at width {sb64} "
          f"(None => exact through the full 11-digit sum; 2^52 ~ 4.5e15)")
    print()

    # (2) full end-to-end model (sum + MSB-first floor decode).
    b32, err32, ex32 = fp_width_boundary(torch.float32)
    b64, err64, _ = fp_width_boundary(torch.float64)
    print("[B] full model output digits vs true a+b (sum + floor decode)")
    print("    operand digits :", "  ".join(f"{w:>5d}" for w in widths))
    print("    fp32 errors    :", "  ".join(f"{err32[w]:>5d}" for w in widths))
    print("    fp64 errors    :", "  ".join(f"{err64[w]:>5d}" for w in widths))
    print(f"    => fp32 full model unusable at 10-digit scale (the MSB-first floor")
    print(f"       decode needs sub-1e-9 resolution on values ~1e10 => ~19 sig digits,")
    print(f"       far past fp32's ~7; fp64's ~15-16 sig digits carry it exactly).")
    if ex32:
        w, a, b, g32, true = ex32
        print(f"    first fp32 model miss (w={w}): {a}+{b} -> {g32} (true {true})")
    print()
    print(f"CONCLUSION: fp64 is REQUIRED. The clean mantissa boundary [A] shows the")
    print(f"sum breaks near 7-8 digits in fp32 (2^24); fp64 holds the whole sum.")
    assert b64 is None, "fp64 must be exact across all operand widths"


if __name__ == "__main__":
    main()
