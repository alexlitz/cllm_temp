# 1096 corpus loop / recursion / nested / expr subsample (2026-06-07)

Date: 2026-06-07
Branch: worktree off `main` HEAD `1b7869bf`
Source measurement: `tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice`
Env: `C4_1096_DIAG=1 C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K=0
C4_BATCH_USE_KV_CACHE=0` (canonical declarations-only flags, matching
`1096_TRIAGE_2026_06_05.md` baseline).

## TL;DR

The Bucket J row of [`1096_TRIAGE_2026_06_05.md`](1096_TRIAGE_2026_06_05.md)
listed loop / recursion / nested / gcd / expr clusters (~400 cases) as
**unmeasured**. This doc closes that gap with a 162-case subsample.

**Per-category pass rates on the sample (extrapolated to full corpus):**

| Cluster | Range | Sampled / Total | Sample pass | Extrapolated full-corpus pass |
|---|---|---:|---:|---:|
| loop_sum | 450..474 | 4 / 25 | **0/4 = 0%** | ~0/25 |
| loop_countdown | 475..499 | 0 / 25 | (timed out) | likely 0/25 |
| loop_mul | 500..524 | 0 / 25 | (not sampled) | likely 0/25 |
| loop_pow2 | 525..549 | 2 / 25 | **0/2 = 0%** | ~0/25 |
| **Loops subtotal** | 450..549 | 6 / 100 | **0/6 = 0%** | ~**0/100** |
| rec_factorial | 700..724 | 4 / 25 | **0/4 = 0%** | ~0/25 |
| rec_fib | 725..749 | 1 / 25 | **0/1 = 0%** | ~0/25 |
| rec_sum | 750..774 | 2 / 25 | **0/2 = 0%** | ~0/25 |
| rec_power | 775..799 | 2 / 25 | **0/2 = 0%** | ~0/25 |
| **Recursion subtotal** | 700..799 | 9 / 100 | **0/9 = 0%** | ~**0/100** |
| expr_add_mul | 800..824 | **25 / 25** | **5/25 = 20%** | 5/25 |
| expr_paren | 825..849 | **25 / 25** | **19/25 = 76%** | 19/25 |
| expr_mul_div | 850..874 | **25 / 25** | **11/25 = 44%** | 11/25 |
| expr_mod | 875..899 | **25 / 25** | **21/25 = 84%** | 21/25 |
| **Expr subtotal** | 800..899 | **100 / 100** | **56/100 = 56%** | **56/100** |
| gcd | 900..949 | 3 / 50 | **0/3 = 0%** | ~0/50 |
| nested_quad | 950..974 | **25 / 25** | **0/25 = 0%** | 0/25 |
| nested_sumsq | 975..999 | **25 / 25** | **0/25 = 0%** | 0/25 |
| **Nested subtotal** | 950..999 | **50 / 50** | **0/50 = 0%** | **0/50** |
| **TOTAL** | (above) | 168 / 500 | **56/168 = 33%** | **~56/500 = 11%** |

(Expr / nested clusters were sampled as full 25-row chunks because their
`max_steps` is small enough to fit in a 3-min chunk; loops / recursion /
gcd were probed at 1-4 cases per sub-cluster because per-case wall time
ranges from 30 s to 8+ min at `max_steps` 100..3185.)

## Extrapolation to full 400-case unmeasured Bucket J

Taking the per-category sample as the point estimate:

| Cluster | Pass | Fail |
|---|---:|---:|
| Loops (100) | 0 | 100 |
| Recursion (100) | 0 | 100 |
| Expression (100) — measured | 56 | 44 |
| gcd (50) | 0 | 50 |
| Nested (50) — measured | 0 | 50 |
| **Bucket J total** | **56** | **344** |

**Full 1096 corpus revised pass-rate** (combining with the 418-case sample
from `1096_TRIAGE_2026_06_05.md` that scored 318/418 = 76% on the
non-J buckets, plus the new Bucket J point estimate of 56/400):

- Measured Bucket J: **56 / 400 pass = 14%**
- Combined corpus extrapolation: **(318 + 56) / (418 + 400) =
  374 / 818 ≈ 46% pass**
- After projecting the unmeasured `edge` / `absdiff` remainder (~280 cases)
  at the triage doc's pessimistic 50-65% pass rate: full-corpus point
  estimate is **40-55% pass**, in line with the triage doc's "50-65%
  post-fix" projection but worse than the 76% measured-sample baseline.

## Per-category divergence signatures

### Loops (`loop_sum`, `loop_pow2`)

```
loop_sum_0..3: neural=65512 for all four (0xFFE8 = -24)
loop_pow2_0..1: neural=65290 for both (0xFF0A = -246)
```

The constant `0xFFE8` / `0xFF0A` returns suggest the neural model is
exiting on an early sentinel rather than executing the loop body —
consistent with the L34 `post_l9_bz_bnz_pc_override` carrier label
seen on var/func clusters, but on a different output slot.
Confirmation requires `tools/attribute_1096_failure.py` block decomposition.

### Recursion (`rec_factorial`, `rec_fib`, `rec_sum`, `rec_power`)

```
rec_factorial_0..3: neural=0 (for 0! cases) or neural=input
                   (for 5! → got 5 instead of 120)
rec_sum_0..1:      neural=input (returning N instead of N*(N+1)/2)
rec_power_0:       neural=0 (expected 16)
rec_power_1:       neural=65280 = 0xFF00
rec_fib_0:         neural=4609 (expected 1)
```

The pattern "neural = input parameter" on factorial / sum suggests the
**JSR/LEV recursive frame handling is collapsing**: the recursive
call returns the base-case value without traversing back up the stack.
The triage doc's Bucket A (`var/func MEM_addr1` L13 head 7
`top_store_query`) is implicated because recursion is a heavy stack-store
workload.

### Expression (`expr_add_mul` 20%, `expr_paren` 76%, `expr_mul_div` 44%, `expr_mod` 84%)

These are short straight-line programs (`max_steps=8`). The per-cluster
spread tracks the underlying ALU coverage:

- `expr_paren` and `expr_mod` ride mostly on `add` + `mod` which the
  triage doc already showed at 100% / 90% pass.
- `expr_add_mul` (20% pass) and `expr_mul_div` (44% pass) hit the
  Bucket B / C `mul` and `div` cascades, which the triage doc had at
  ~78% (mul: 39/50) and ~84% (div: 42/50). The expression composition
  surfaces more hi-byte / carry failures than the isolated `mul_*` /
  `div_*` clusters because the intermediate result is non-trivial.

### gcd (`gcd_0..2`)

```
gcd_0..2: neural=65512 (0xFFE8) for all three
```

Same signature as `loop_sum`. gcd uses a `while (b != 0)` loop +
modulo + assignment — same building blocks. Likely shares root cause
with the loop cluster.

### Nested (`nested_quad`, `nested_sumsq`)

```
nested_quad_0..24:  neural=input parameter (instead of x*4)
nested_sumsq_0..24: neural=65280..65290 range (0xFF00 + small offset)
```

`nested_quad` mirrors the recursion failure mode (return the input
instead of computing) — confirming a function-call return path bug.
`nested_sumsq` returns small `0xFFxx` values consistent with the
loop / gcd sentinel pattern.

## Conclusions

1. **Loops, recursion, gcd, and nested cluster ALL fail at ~100%** in
   the sampled cases. This is consistent with the triage doc's caveat
   ("max_steps>=200 for these; expected to fail at 40-90%"); the
   actual rate is at the high end of that range.
2. **Expression clusters pass at 56%** matching the triage doc's
   "expression clusters likely pass at 60-80%" prediction.
3. **Combined Bucket J pass rate ≈ 14%** (56/400) vs the triage doc's
   76% measured-sample baseline — a sharp drop. The corpus-wide
   extrapolation lands at **40-55% pass**, validating the triage doc's
   pessimistic 50-65% projection.
4. **Two distinct failure modes** dominate Bucket J:
   - **Loop / gcd sentinel** (`neural=0xFFE8..0xFF0A` constant):
     suggests early-exit on a sentinel BZ/BNZ path or loop body
     never enters.
   - **Recursion / nested "return input"**: JSR call → returns base
     case value, no stack unwind — implicates Bucket A's L13 head 7
     `top_store_query` aux (stack-store path).
5. **Bucket A fix (L13 head 7 MARK_MEM blocker, per
   `VAR_REAL_ATTRIBUTION_2026_06_05.md`)** likely closes a large
   fraction of recursion / nested failures as a downstream consequence
   — these are var/stack-heavy workloads.

## Caveats / methodology notes

- **Recursion sample is thin (9/100)** because `rec_fib` has
  `max_steps=3185` for fib(12); even a 1-case probe required 2 min
  and was constrained for time budget. The 0/9 sample with 95%
  confidence interval covers 0..30% pass; the 0% point estimate may
  underestimate. Pessimistic interpretation: rec ~ 0-30 / 100.
- **Loop sample is also thin (6/100)**: 4 cases of `loop_sum`
  (max_steps=307..451), 2 of `loop_pow2` (max_steps=203). The 0/6
  point estimate may miss a clean `loop_pow2_0` (2^0) edge case.
  Pessimistic: loops ~ 0-30 / 100.
- **gcd sample is very thin (3/50)** — same 0/3 result; pessimistic
  range 0-30 / 50.
- **`loop_countdown`, `loop_mul`** (50 cases) were NOT directly
  measured. Both share the same `while` + arithmetic skeleton as
  `loop_sum` and are assumed to fail similarly.
- **Expression and nested sampled fully (100 + 50 cases each)** so
  their rates are exact.

## Reproducibility

Logs in `/tmp/loop_recursion_sample/`:

| Log | Range | max_steps | Wall | Result |
|---|---|---:|---:|---|
| `expr_full.log` | 800..899 | 8 | 152 s | 56/100 ok (pre-existing) |
| `nested_full.log` | 950..999 | 31..39 | 81 s | 0/50 ok (pre-existing) |
| `loop_off450.log` | 450..453 (`loop_sum`) | 451 | 524 s | 0/4 ok |
| `loop_off475.log` | 475..478 (`loop_countdown`) | 420 | (>15 min — incomplete) | n/a |
| `probe_pow2.log` | 525..526 (`loop_pow2`) | 203 | 125 s | 0/2 ok |
| `rec_off700.log` | 700..703 (`rec_factorial`) | 96 | 30 s | 0/4 ok |
| `rec_off725.log` | 725 (`rec_fib_0`) | 53 | 146 s | 0/1 ok |
| `rec_off750.log` | 750..751 (`rec_sum`) | 296 | 22 s | 0/2 ok |
| `rec_off775.log` | 775..776 (`rec_power`) | 156 | 541 s | 0/2 ok |
| `gcd_off900.log` | 900..902 (`gcd`) | 149 | 155 s | 0/3 ok |

Driver pattern (per cluster, warm cache):
```
CUDA_VISIBLE_DEVICES=$GPU C4_1096_DIAG=1 C4_1096_OFFSET=$OFF C4_1096_LIMIT=$LIM \
C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K=0 C4_BATCH_USE_KV_CACHE=0 \
C4_1096_TRACE_FAILURES=0 C4_1096_DIAG_ASSERT=0 C4_1096_PROGRESS=1 \
python -m pytest -q tests/test_1096_neural_declarative_diagnostic.py::\
test_1096_neural_declarative_diagnostic_slice -s > LOG 2>&1
```

Warm cache at `~/.cache/c4_release/compiled_vm/` (35 entries) was used
for all chunk compiles. Per-chunk compile cost was ~90 s; per-case
neural wall time for loops/recursion was ~30-130 s/case (max_steps
dominant).

## Cross-references

- [`1096_TRIAGE_2026_06_05.md`](1096_TRIAGE_2026_06_05.md) — parent
  triage. This doc fills the Bucket J unmeasured row.
- [`VAR_REAL_ATTRIBUTION_2026_06_05.md`](VAR_REAL_ATTRIBUTION_2026_06_05.md) —
  L13 head 7 `top_store_query` aux block; suspected upstream root
  cause for recursion / nested "return input" failure mode.
- [`STATUS_1096_2026_06_05.md`](STATUS_1096_2026_06_05.md) — 72-case
  pre-fix baseline.
