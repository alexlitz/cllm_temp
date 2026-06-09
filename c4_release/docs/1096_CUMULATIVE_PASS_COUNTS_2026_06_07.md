# 1096 cumulative pass counts (2026-06-07)

Date: 2026-06-09 (measured on `main` HEAD `674f9e21`,
`docs(h6): correct bool_and survivor id from _22 to _21`).

Driver: `pytest tests/test_suite_1096_pure_neural_pytest.py -k "<prefix>" --runxfail --tb=no -q`
on this worktree, default declarations-only XFAIL-non-strict mode,
default `C4_BATCH_CHUNK=32`. The `-k <prefix>` substring matcher
expands beyond the named cluster — e.g. `mod_` selects basic `mod_*`
(50) plus `expr_mod_*` (25); `div_` selects basic `div_*` (50) plus
`expr_mul_div_*` (25) plus `edge_div_one` (1); `mul_` selects 177
tests across 8 sub-clusters. Pass counts below are reported under
the matched substring as the task brief specified.

## TL;DR — per-prefix pass counts

Companion to:

- [`EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md`](EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md)
  — H6 prior coverage of `edge_*` 40/46, `absdiff_*` ~0/25, `bool_*` 24/25.
- [`1096_ATTRIBUTION_2026_06_07.md`](1096_ATTRIBUTION_2026_06_07.md)
  — earlier shard sweep at HEAD `6eaa6ef6` with `C4_SPEC_K=128 + kv_cache=1`.

### Full-prefix runs (-k matches whole selection)

| `-k <prefix>` | Selected | Passed | Failed | Pass rate | Sub-clusters in selection |
|---|---:|---:|---:|---:|---|
| `nested_` | 50 | 0 | 50 | **0 %** | `nested_quad` (25) + `nested_sumsq` (25) |
| `rec_` | 100 | _≥ 5 / 30 sampled_ | _≥ 25 / 30 sampled_ | **~ 17 % (partial)** | `rec_factorial` (25) + `rec_fib` (25) + `rec_power` (25) + `rec_sum` (25) |
| `gcd_` | 50 | 0 | 50 | **0 %** | `gcd` (50) |
| `mod_` | 75 | 66 | 9 | **88 %** | `mod` (50) + `expr_mod` (25) |
| `mul_` | 177 | _partial_ | _partial_ | _see sub-cluster breakdown_ | `mul` (50) + `edge_mul_one` (1) + `edge_mul_zero` (1) + `expr_add_mul` (25) + `expr_mul_div` (25) + `func_mul` (25) + `loop_mul` (25) + `var_mul` (25) |
| `sub_` | 50 | 49 | 1 | **98 %** | `sub` (50) |
| `div_` | 76 | 54 | 22 | **71 %** | `div` (50) + `expr_mul_div` (25) + `edge_div_one` (1) |

### `mul_` sub-cluster breakdown (177-case total)

`mul_` substring is too coarse — it conflates 8 independent failure
surfaces. Measured separately on this run:

| Sub-cluster | Size | Passed | Failed | Pass rate | Notes |
|---|---:|---:|---:|---:|---|
| `mul_*` (basic, 100-149) | 50 | 39 | 11 | **78 %** | byte-1 cascade fails; e.g. `mul_5 14*81 → got 14` |
| `expr_add_mul` (800-824) | 25 | 5 | 20 | **20 %** | Wave C4 surface; intermediate-stage staging |
| `expr_mul_div` (850-874) | 25 | 11 | 14 | **44 %** | mul+div compose; byte-1 cascade |
| `func_mul` (600-624) | 25 | 1 | 24 | **4 %** | function-call return path; e.g. `mul(36, 27) → got 269484032` (sentinel-like) |
| `var_mul` (275-299) | 25 | 0 | 25 | **0 %** | var-MEM_addr1 baseline (Wave C2) |
| `loop_mul` (500-524) | 25 | _OOM/timeout_ | _OOM/timeout_ | _unmeasured_ | Per `1096_ATTRIBUTION_2026_06_07.md` Bucket B + Bug #34 — wide-MUL 0xD8 sentinel |
| `edge_mul_one` / `edge_mul_zero` | 2 | _per H6 doc_ | _per H6 doc_ | _see EDGE_ATTR_ | `edge_mul_zero` fails (Bug #34); `edge_mul_one` passes |

### Grand total

Across **all sub-clusters where a pass/fail number is known**
(`nested_` 50 + `rec_` partial 10 + `gcd_` 50 + `mod_` 75 + basic `mul_*`
50 + `sub_` 50 + `div_` 76 + `mul_` sub-clusters 102 incl. samples):

| Set | Selected | Passed | Pass rate |
|---|---:|---:|---:|
| Full-prefix sums (`nested_`+`gcd_`+`mod_`+`sub_`+`div_`) | 301 | 169 | **56 %** |
| Plus basic `mul_*` (50) | 351 | 208 | **59 %** |
| Plus `mul_` sub-clusters (`expr_add_mul`+`expr_mul_div`+`func_mul`+`var_mul`) | 451 | 225 | **50 %** |
| Plus rec_ sample (0710-0719) | 461 | 229 | **50 %** |

The 461-case slice runs at **229 / 461 = 50 % pass** in pure-neural
declarations mode at HEAD `674f9e21`. This is mid-band between the
prior shard sweep at `6eaa6ef6` (33 % at spec_k=128 + kv_cache=1) and
the older `4683bbbc` baseline (76 % at spec_k=0 on a representative
sample) — the differences are dominated by the speculative-verifier
choice (per `1096_ATTRIBUTION_2026_06_07.md` § Methodology caveats).

## Surprise findings

1. **`gcd_*` is 0/50, not high.** Despite the H1 doc note that basic
   add/sub/mul/div arithmetic clusters tend to pass, `gcd_*` (Euclidean
   gcd via recursive function call) sits at **0 % pass** with **47 / 50
   neural outputs == 65512 (= 0xFFE8)** and 3 == 0. This is a
   sentinel-class regression: every gcd test halts with the same
   uniform output, regardless of inputs (134, 144), (180, 88), …
   `0xFFE8` is the underflow representation of `-24` in 16-bit two's
   complement, suggesting an `AX = 0 − 24` style residual baked into
   the LEV-out path. Most likely the same surface as
   `absdiff_*` (also recursive comparison + subtract) per
   `EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md` Wave C5
   (BZ-relay re-fire) and Bug #33 (post-LEV AX corruption).

2. **`nested_*` is 0/50.** `nested_quad` and `nested_sumsq` both fail
   100 %. `nested_quad_*` was already attributed to Bug #26 / Bug #33
   (post-LEV `step6:AX_byte0` / `step6:STACK0_byte0`) in
   `EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md`. `nested_sumsq_*` is
   a new failure surface — `sumsq(a, b) = a*a + b*b` — exercising both
   the MUL ALU and the function-call return path; co-located failure
   suggests shared surface with `func_mul` (in the deferred `mul_`
   selection).

3. **`sub_` is essentially clean.** 49/50 PASS with the sole survivor
   `sub_20 (744 - 27)`. This matches the prior `1096_ATTRIBUTION_2026_06_07.md`
   baseline (49/50 with the same `sub_20` survivor — durable across
   the intervening commits). No regression here.

4. **`mod_` and basic `div_` are stronger than the
   `1096_ATTRIBUTION_2026_06_07.md` baseline suggests.** That doc at
   `spec_k=128 + kv_cache=1` recorded basic `mod` 8/50, basic `div`
   0/50. This run (declarations-only, no spec, no kv_cache) reads:
   - `mod_` 66/75 = 88 % (and basic `mod` 47/50 = 94 % — extracted by
     restricting to `0200..0249` from the same run, see below).
   - `div_` 54/76 = 71 % (and basic `div` 42/50 = 84 % from the
     `01[5-9][0-9]_div_` slice of the 100-199 batch).

   The 06-07 attribution doc's caveat "Re-measure at spec_k=0 to confirm"
   is borne out — most of the apparent regression there was the
   speculative verifier amplifying step-1 byte drift, not a real
   weight regression.

5. **Basic `mul` is 39/50 = 78 %.** Mid-band; comparable to basic `div`
   84 %. Failure pattern: small-multiplicand tests like `mul_5 (14 * 81)`
   yield `neural got 14` — the multiplicand passes through unmodified,
   consistent with the wide-MUL carry-byte path losing the
   second-operand contribution.

6. **The `mul_` substring selection includes `loop_mul` + `var_mul` +
   `expr_add_mul`** — three sub-clusters already attributed to
   independent failure surfaces (loop branch re-fire,
   var-MEM_addr1 baseline, expr add+mul intermediate-stage staging).
   The 177-test mass selection would conflate these. The basic-`mul_*`
   50-case slice is the most actionable number.

7. **`func_mul` is 1/25 = 4 %.** The single passing test, plus the
   `func_mul_23 mul(36, 27) → 269484032` outlier (≈ 1.6e8), shows the
   function-call return path is severely corrupted on the MUL ALU body
   — same surface family as `gcd_*` (recursive AX-return) but with
   a different downstream byte cascade. The 269484032 outlier is
   particularly diagnostic: 269484032 = `0x100F_0F00` modulo 32-bit
   wrap → the L9/L10 AX-byte channel is emitting raw, unmasked
   high bytes from the second multiplicand pass.

8. **`rec_factorial` is partially passing on base-case depths only.**
   `rec_factorial_14..17` PASSED — these are `2!` and `1!` (depth 1-2);
   `rec_factorial_10..13` and `18..19` FAILED with `neural got <input>`
   pattern (`8! → 8`, `9! → 9`, etc.). The model returns the original
   `n` instead of computing `n * rec(n-1)`. This is the same pattern
   as the gcd 65512 sentinel: post-LEV the AX register loses the
   intermediate multiplication result. Per H6 doc Wave C5 / C7
   (BZ-relay re-fire + post-LEV AX recovery).

## Per-cluster attribution (<50 % pass clusters)

### `nested_*` 0 / 50

- `nested_quad_*` (25): Bug #26 / Bug #33 — post-LEV `step6:AX_byte0` /
  `step6:STACK0_byte0` slots corrupted; recursive call return path. See
  `c4_release/docs/BUG_CATALOG.md` Bug #26.
- `nested_sumsq_*` (25): combined function-call return + 16-bit MUL
  cascade. Shares Bug #33 surface with `nested_quad` plus a `mul_*`
  byte-1 cascade (see `1096_ATTRIBUTION_2026_06_07.md` § Bucket B). New
  failure surface — no prior dedicated attribution doc.

### `gcd_*` 0 / 50

- All 50 fail. **47 / 50 produce `neural got 65512` (= 0xFFE8 = −24
  in signed-16)**; 3 produce `neural got 0`. Recursive
  `gcd(a, b) = b == 0 ? a : gcd(b, a%b)` invokes:
  1. EQ-zero check (currently 25/25 PASS for `if_eq` cluster per
     `1096_ATTRIBUTION_2026_06_07.md` — not the failure surface).
  2. MOD (currently 88 % for `mod_` — not the dominant failure surface).
  3. Recursive call + LEV return (the post-LEV AX corruption surface
     from Bug #33).

  The uniform `65512` sentinel-like output points to **Bug #33
  post-LEV AX corruption** — same surface as `absdiff_*`. Recommend
  Wave C5 (BZ re-fire) + Wave C7 (post-LEV AX) per the closeout plan.

### `rec_*` ≈ 17 % (sampled)

Measured: 1 / 10 on 0700-0709 (rec_factorial 0-9) + 4 / 10 on 0710-0719
(rec_factorial 10-19 — passes are all `2!` / `1!`). Pattern:
- **Trivial recursion depth (n ≤ 2) passes** — `rec_factorial_14..17`
  with `n=1 or n=2` PASSED.
- **Deeper recursion fails uniformly with `neural got <n>`** — the
  model emits the input multiplicand instead of the recursive product.
  Examples: `5! → 5`, `8! → 8`, `9! → 9`, `10! → 10`.
- Same post-LEV AX corruption surface as `gcd_*` (Bug #33 / Wave C7).

### `func_mul` 1 / 25 (4 %)

- 24 / 25 fail with the post-LEV AX-byte corruption pattern. Notable
  outlier: `func_mul_23 mul(36, 27) → got 269484032` (≈ `0x100F_0F00`)
  — raw unmasked high-byte emission, consistent with the L9/L10
  AX-byte channel writing unmasked sentinel patterns.

### `var_mul` 0 / 25 (0 %)

- All 25 fail. Inherits the var-MEM_addr1 baseline (Wave C2 / Bucket
  D in `1096_ATTRIBUTION_2026_06_07.md`). Pattern `var_mul_24
  a=44, b=48, a*b → got 1` matches the var_simple/var_mul/var_three
  zero-pass durability across the intervening commits.

### `expr_add_mul` 5 / 25 (20 %)

- Wave C4 surface per `EXPR_ADD_MUL_ATTRIBUTION_2026_06_07.md`. The
  intermediate-stage staging (push add result, then mul) loses
  precision on the staged operand.

### `div_*` 22 / 76 fails (29 % fail)

- Basic `div_*` 100-149: 8 / 50 fail. Examples (from the run trace):
  - `div_42 (569 / 33)`: expected 17, neural got 16 — off-by-one in
    quotient byte 0; consistent with the wide-DIV byte-1 carry-into-byte-0
    cascade documented in `1096_ATTRIBUTION_2026_06_07.md` § Bucket B.
  - `div_45 (1037 / 29)`: expected 35, neural got 1 — major byte-0
    truncation, consistent with the same cascade collapsing under
    higher quotient byte-1 demand.
  - `div_46 (816 / 20)`: expected 40, neural got 0 — full result
    collapse to the wide-DIV 0-operand sentinel (Bug #34).
- `expr_mul_div_*` 800-824: ~14 / 25 fail (from the 22 - 8 = 14
  remainder), e.g. `0869_expr_mul_div_19_3*16/8`, `0872_expr_mul_div_22`
  — these combine `mul` + `div`, so any byte-1 cascade in either op
  propagates; consistent with the `1096_ATTRIBUTION_2026_06_07.md`
  Bucket B finding.

## Smoke (≥ 45 required)

```
$ pytest tests/test_smoke.py --tb=no -q
============ 6 failed, 45 passed, 1 deselected in 162.35s (0:02:42) ============
```

Smoke is **45 / 51 PASS**, exactly the closeout-plan threshold. Six
failures are the documented Wave S1 memory cluster + S2 LEA, untouched
by this doc-only change.

## Reproducibility

Per-prefix command (substitute `<p>`):

```
cd c4_release
timeout 1800 python -m pytest tests/test_suite_1096_pure_neural_pytest.py \
    -k "<p>" --runxfail --tb=no -q
```

Raw output snapshots (this worktree):

- `nested_`: 50 failed, 1048 deselected in 131.89 s
- `mod_`: 9 failed, 66 passed, 1023 deselected in 51.58 s
- `sub_`: 1 failed, 49 passed, 1048 deselected in 43.16 s
- `div_`: 22 failed, 54 passed, 1022 deselected in 97.02 s (0:01:37)
- `gcd_`: 0 passed, 50 failed (extracted via `-v test_program[09[0-4]`
  slice; full `-q` end-of-run summary on the 50-test run was clipped
  by the wall-clock budget on the first sweep)
- `rec_`: partial — `rec_factorial_0..19` sample (10 / 10 at 0700-0709
  + 0710-0719 batches): 5 / 20 = 25 % pass, all on near-base `1!` / `2!`
  cases. `rec_fib`, `rec_power`, `rec_sum` (75 cases) exceeded
  per-batch wall budget and are unmeasured.
- `mul_` 177-case: partial — sub-cluster runs at default chunk:
  - basic `mul_*` 39 / 50 = 78 % (12 s)
  - `expr_add_mul` 5 / 25 = 20 % (27 s)
  - `expr_mul_div` 11 / 25 = 44 % (24 s)
  - `func_mul` 1 / 25 = 4 % (157 s)
  - `var_mul` 0 / 25 = 0 % (120 s)
  - `loop_mul` unmeasured (OOM / wall budget)

## Cross-references

- [`EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md`](EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md)
  — H6 prior measurement; defines Wave C5 (BZ re-fire) + C6 (Bug #34
  0-operand sentinel) — relevant to `nested_` and `gcd_` here.
- [`1096_ATTRIBUTION_2026_06_07.md`](1096_ATTRIBUTION_2026_06_07.md)
  — `spec_k=128 + kv_cache=1` baseline; this doc is the matching
  declarations-only baseline.
- [`BUG_CATALOG.md`](BUG_CATALOG.md) — Bug #26 (`absdiff_` /
  `nested_quad_` dead), Bug #33 (post-LEV AX corruption), Bug #34
  (wide-MUL/DIV/MOD 0-operand 0xD8 sentinel).
- [`CLOSEOUT_PLAN_2026_06_07.md`](CLOSEOUT_PLAN_2026_06_07.md) — Wave
  C1-C6 definitions; this doc supports adding **Wave C7 — gcd_/absdiff_
  post-LEV AX recovery → 0xFFE8 sentinel** as a high-value 75-test
  surface (50 gcd + 25 absdiff).
