# tail_sp_byte1_ff MARK_AX blocker fix — REVERTED (2026-06-04)

Date: 2026-06-04
Worktree: `/tmp/c4-tail-sp-byte1-blocker/c4_release/`
Branch: `tail-sp-byte1-blocker` (off `d110afbb` on `speedup-cache-and-buckets`)
Author: agent (Claude Opus 4.7)
Brief: `EQ_CALL90_DIFF_2026_06_04.md` §"Concrete fix surface" step 3 —
raise `tail_sp_byte1_ff_from_initial_stack_exact`'s MARK_AX blocker from
clamped −1e6 to unclamped −1e9 by bumping `max_abs_weight` to
`1_000_000_000`.

## TL;DR

**The fix did not recover EQ(17,17) or EQ(42,42).** Both still return
exit 0 (draft_divergence bail) after applying the recommended
MARK_AX=−1e9 + max_abs_weight=1e9 patch. Verifier reports identical
strength/scope violation counts on `tail_bit32_result_correction`
(43,681 / 1,446 — pre-existing baseline). Smoke shows 28 passed (vs
expected 34/52 floor); CMP family (`test_eq_true`, `test_eq_false`,
`test_ne_true`, `test_gt_true`, `test_ge_true`) all still FAIL. Patch
reverted; no production code changed. This adds another entry to the
0/N tally of single-rule fix attempts (per
`feedback_single_rule_fixes_are_zero_sum.md`).

## Patch attempted

Single edit at
`c4_release/neural_vm/unified_compiler/ops/l10_ops.py:5297-5331`:

```diff
     *exact_output_byte_rules(
         name="tail_sp_byte1_ff_from_initial_stack_exact",
         ...
-        ("MARK_AX", -1000000.0),
+        ("MARK_AX", -1_000_000_000.0),
         ...
         threshold=40.0,
         active_value=50.0,
+        max_abs_weight=1_000_000_000.0,
     ),
```

The default `max_abs_weight=1_000_000.0` inside
`expected_byte_guarantee_rules` (`band_guarantees.py:460`) clamps any
condition weight to ±1e6 magnitude. The sibling
`tail_sp_marker_byte0_f8_from_initial_stack_exact` at line 5252
explicitly passes `max_abs_weight=1_000_000_000.0` so its MARK_AX=−1e9
blocker survives lowering; the byte1_ff rule did not, so its MARK_AX
gate landed at −1e6. The patch was a literal mirror of the SP variant.

## Probe results

```
EQ(5,5)   = exit 1 (expected 1)  ← PASS, FETCH_HI is 0 for operand < 16
EQ(17,17) = exit 0 (expected 1)  ← FAIL, unchanged from baseline
EQ(42,42) = exit 0 (expected 1)  ← FAIL, unchanged from baseline
```

The single ALU+CMP step that produces a 0xFF byte-1 carry on TRUE
comparisons is still corrupted, despite the (in theory) much stronger
MARK_AX guard.

## Verifier check (single run, ONE constraint honoured)

`CUDA_VISIBLE_DEVICES="" python verify_op.py tail_bit32_result_correction`
on the patched tree:

| metric              | count |
|---------------------|------:|
| strength_violation  | 43,681 |
| scope_violation     | 1,446 |
| no_dominates_at     | 0     |

Identical to the pre-patch baseline (also 43,681 / 1,446 / 0,
spot-checked from `speedup-cache-and-buckets` HEAD). The patch does
not introduce new strength/scope regressions inside the op. The
existing 43,681 baseline is dominated by `tail_stack0_pop_marker_zero`
shortfalls vs `tail_sp_pop_byte3_zero` — neither rule the patch
touched. (`my=0.0 / comp=550000.0 / shortfall=550001.0` on OUTPUT_HI+0
is representative; this is a pre-existing competition pattern unrelated
to the MARK_AX guard chain.)

## Smoke results (per-test 60s timeout, single run)

```
28 passed, 18 failed, 30 xfailed, 10 xpassed, 6 errors (22m 27s)
```

Compared to the brief's stated baseline of 29/52 → 34/52 expected:

* 28 passed is essentially the brief's 29/52 baseline (the −1 may be
  variance from the per-test 60s timeout terminating one borderline
  passer; the 6 ERRORs are all `TestSmokeMemory::test_si_li_*` /
  `test_sc_lc_roundtrip` timeouts, which historically run > 60s
  individually and would otherwise pass — those are timeout artifacts,
  not regressions from this patch).
* CMP family unchanged: `test_eq_true`, `test_eq_false`, `test_ne_true`,
  `test_gt_true`, `test_ge_true` all still FAIL. None of the 5 expected
  recoveries materialised.
* No new regressions on previously-passing tests (other than the
  timeout-induced ERRORs in `TestSmokeMemory`).

The headline expected delta (+5 CMP recoveries → 34/52) did not happen.

## Why the brief's hypothesis didn't pan out

The brief's quantitative trigger:

> AX-byte-1 row with `IS_BYTE=5, H1+2 ≈ 20` (from FETCH_HI leak) clears
> the rule's threshold of 25.04 → spuriously fires

contains two arithmetic errors that the patch agent (this session) only
caught after applying the patch:

1. **The threshold is 40.0, not 25.04.** Line 5329 explicitly sets
   `threshold=40.0`. The 25.04 figure appears to be from a different
   rule (possibly the byte0_f8 sibling's `threshold=20.04` + min_margin
   adjustment, or an earlier draft of the byte1_ff rule).

2. **`IS_BYTE=5, H1+2 ≈ 20` gates the activation to 5×IS_BYTE_val +
   20×H1+2_val (not weight×weight).** The actual evidence row is the
   AX-byte-1 emission row where `IS_BYTE` is approximately 1.0 and
   `H1+2` is approximately 1.0 in clean state. The activation totals to
   ~5 + ~20 = ~25, well under the 40 threshold. FETCH_HI leakage would
   have to push H1+2 from ~1 to ~2 to clear 40, which the L6 seed at
   norm 1.99 spread across `H1+{0..4}` may or may not do at the
   AX-byte-1 row specifically.

The MARK_AX=−1e9 blocker only matters if the rule was firing AT ALL on
AX rows. Given EQ(17) still fails identically, the rule is not the
amplifier — or it is being overpowered by a different downstream path
that the L10 dim-clamp doesn't reach.

## What the verifier output actually says

The 43,681 strength violations on `tail_bit32_result_correction` are
pre-existing and dominated by:

* `tail_stack0_pop_marker_zero` (~16 OUTPUT_* dims × many lanes,
  shortfall vs `tail_sp_pop_byte3_zero` at 550k magnitude)
* Other STACK0-marker rules with `my=0.0` against existing competitors
  with strength ≥ 2k

None of these mention `tail_sp_byte1_ff_*` or its byte0_f8 sibling.
The MARK_AX guard chain is verifier-clean both before and after the
patch — meaning the verifier was never going to flag this particular
fix surface even if the hypothesis were correct, because the verifier
doesn't reason about residual-stream contamination from upstream
(L6/L7/L13) writers; it only reasons about same-op rule-vs-rule
competition.

The brief's claim "`verify_rule_strength` can sanity-check the fix"
turned out to be a NEGATIVE check (no regression) only, not a positive
predictor of fix efficacy. The verifier said "patch is safe" and the
empirics said "patch is inert."

## Where the amplifier actually lives (unchanged from the brief)

The EQ(17)/EQ(42) → exit 0 chain still appears to be:

* block 6 L6 routing FFN: 1.99-norm FETCH_HI seed (intentional mixing)
* block 27 `layer15_memory_lookup` attn: 40× amplification (1.99 → 80)
* block 28 `layer16_lev_routing` FFN: 100× amplification (80 → 7805)
* block 34 `tail_bit32_result_correction` FFN: 150,000× amplification
  (7805 → 1.2e9)

The `tail_sp_byte1_ff_*` rule is not the dominant byte-1 writer at AX
rows under FETCH_HI contamination; some other rule inside the 2059-unit
block-34 FFN is. Candidates not yet examined:

* `tail_sp_pop_byte3_zero` — flagged by the verifier as the top
  competitor against many sibling rules; high-strength STACK0 writer.
* `tail_stack0_*` family — broad set of rules with `mark == STACK0`
  scope that may overlap into AX-marker rows under contamination.
* Other `tail_sp_*` or `tail_ax_*` byte-emit rules whose names didn't
  surface in the verifier top-20.

A productive next step would be to instrument
`tail_bit32_result_correction`'s lowered FFN with a per-rule
activation hook on the failing EQ(17) call-90 residual and identify
which of the 2059 hidden units fire at the AX-byte-1 row. That probe
was out of scope for this fix attempt (constraint: ONE
compile/smoke/verifier each).

## Single-rule attempts tally

Per `feedback_single_rule_fixes_are_zero_sum.md`, prior single-rule fix
attempts net zero (0/5 historical). This session adds **0/6** —
verifier-friendly + verifier-blind + brief-vouched, and still inert.

Future fix briefs on this surface should require an empirical
positive-evidence trigger (e.g. a hook proving the targeted rule
actually fires at the corruption row) before patching, not just an
inferred trigger from upstream contamination magnitudes.

## What stays unchanged

* Production code: zero diff. `git status` clean on
  `tail-sp-byte1-blocker` after revert.
* The EQ(17)/EQ(42) failure remains open. Brief
  `EQ_CALL90_DIFF_2026_06_04.md` measurements (block-by-block diff,
  FETCH_HI-uniform seed signature) still stand; they correctly localise
  the amplifier to block 34 but the specific in-block rule attribution
  was wrong.
* Verifier output identical (43,681 strength / 1,446 scope), confirming
  the patch was structurally safe — just empirically inert.

## Files

* New: `c4_release/docs/TAIL_SP_BYTE1_BLOCKER_FAILED_2026_06_04.md`
  (this file).
* Touched then reverted:
  `c4_release/neural_vm/unified_compiler/ops/l10_ops.py` (3 lines:
  comment block + `MARK_AX` weight + `max_abs_weight=`).
* Verifier run: `/tmp/verify_out.txt` (off-tree, captured stdout of
  `verify_op.py tail_bit32_result_correction`).
* Smoke run: 22m 27s, single invocation,
  `pytest tests/test_smoke.py tests/test_smoke_pure_neural.py
  --timeout=60`. Captured to background task output.

## Confidence

* **High** that the literal patch landed correctly. Lowered weight at
  the touched conditions reflected in re-grep of the file before revert
  (`MARK_AX = -1_000_000_000.0`, `max_abs_weight=1_000_000_000.0` both
  present at lines 5328 and 5343).
* **High** that EQ(17)/EQ(42) did not recover. The probe is
  reproducible: 5/17/42 → 1/0/0 every run, identical to baseline.
* **High** that the verifier did not flag the patch (counts identical
  to pre-patch).
* **Medium** that the brief's amplifier identification (block 34) is
  still correct; the patch attempt does not refute it. What is refuted
  is the specific rule (`tail_sp_byte1_ff_from_initial_stack_exact`)
  being THE responsible byte-1 writer at AX rows.
* **Low** that any single-rule patch within `tail_bit32_result_correction`
  will recover CMP without first instrumenting per-rule activations on
  the actual EQ(17) corruption residual. Per the brief's own caveat
  citing `feedback_single_rule_fixes_are_zero_sum.md`, this attempt
  empirically confirms that warning.
