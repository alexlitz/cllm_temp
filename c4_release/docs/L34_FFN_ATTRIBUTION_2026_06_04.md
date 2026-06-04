# L34.ffn (192 units) attribution — 2026-06-04

Date: 2026-06-04
Worktree: `/tmp/c4-l34-attribution` (branch `l34-attribution`)
HEAD: `cef9d046` (`docs(status): 1096 + smoke baseline 2026-06-04`)
Author: read-only attribution agent

## TL;DR

`L34.ffn` (post-expansion block index 34, the 192-unit FFN flagged in
`STATUS_1096_2026_06_04.md` as the new dominant carrier for
`var_simple_*` and `var_three_*` failures) is

**`post_l9_bz_bnz_pc_override`**

declared in `neural_vm/unified_compiler/ops/l6_ops.py:4547` (factory)
with rule generators at `l6_ops.py:4443-4534` and IR at
`l6_ops.py:4537-4544`. It is the post-layout-L21 FFN tenant pinned via
`requires={"after": "layer10_alu"}`; after wrapper expansion it lands at
the very end of the model (final block 34 of 35).

The 192-unit count comes from:

  * BZ: 16 lo_cancel + 16 hi_cancel + 16 lo_target + 16 hi_target = 64
  * BNZ (`lo_nonzero` group): 16 + 16 + 16 + 16 = 64
  * BNZ (`hi_nonzero` group): 16 + 16 + 16 + 16 = 64
  * Total = **192**.

## Op identification — how I pinned it

One compile run from a fresh `compile_full_vm_dynamic(strict=False)`
session in this worktree (script preserved at
`scripts/dump_l34_attribution.py`).

* Right-size census reports exactly one 192-unit standalone FFN block
  post-expansion: `L34.ffn: 192 units (no dead units)`.
* The same census shows `L21.ffn: 4096 -> 192 units (-3904 dead)` BEFORE
  the wrapper-expansion pass, then `Total blocks: 17 -> 35` after
  expansion. The original block-21 FFN (192 units, no post_ops) is
  appended as-is and ends up as the final block.
* Layout dump (`layout.ops_per_layer[21]`) shows the only FFN op pinned
  at pre-expansion layer 21 is `post_l9_bz_bnz_pc_override`.
* No other block in the final model has hidden_dim = 192. (Nearest
  neighbours: `L33.ffn` 2059 units = `tail_bit32_result_correction`;
  `L31.ffn` 792 units; `L18.ffn` 42 units.)

## Semantic role

Cluster-D PC-override patch. The L6 routing FFN owns the canonical
BZ/BNZ PC-override bands (units 878..1070), but those bands read `CMP`
cross-step because L6 runs before L9. On VM step 1 the cross-step
alias resolves to 0 and the BZ-taken band fails to fire. The cure was
to bake a duplicate of the BZ/BNZ override bands into a post-L9 FFN
block that reads CMP same-step. This op is that block. The L6 BZ/BNZ
unit bands are zero-cleared by `_bake_layer6_routing_ffn` so only this
post-L9 op drives the override at runtime.

Source notes: `docs/CMP_PATH_AUDIT.md`,
`docs/ABSDIFF_BZ_REDIRECT_BUG.md`.

## Rules and their fire path

Two rule families, parallel structure:

* `_post_l9_bz_pc_override_rules` (l6_ops.py:4443) → 64 units.
* `_post_l9_bnz_pc_override_rules` (l6_ops.py:4489) → 128 units (two
  groups, each 64).

Each unit is a `FFNRule.gated_write` (or a `target_*` rule via the
`_append_pc_byte0_direct_copy_rules` helper at l6_ops.py:404). Per-band
shape:

| Family | Conditions | Threshold | Gate dim | Write dim | write_scale |
|---|---|---|---|---|---|
| `post_l9_bz_cancel_lo_{k}` | MARK_PC(+1) + OP_BZ(+0.2) + CMP+4(+1) + CMP+5(+1) - IS_BYTE(10) | 3.5 | `OUTPUT_LO.*.-1+k` × **−1** | `OUTPUT_LO+k` | 2.0/S = 0.02 |
| `post_l9_bz_cancel_hi_{k}` | (same) | 3.5 | `OUTPUT_HI_THIS_STEP+k` × −1 | `OUTPUT_HI_THIS_STEP+k` | 0.02 |
| `post_l9_bz_target_lo_{k}` | (cancel) + MARK_STACK0(−10) | 3.5 | `FETCH_LO+k` | `OUTPUT_LO+k` | 0.02 |
| `post_l9_bz_target_hi_{k}` | (cancel) + MARK_STACK0(−10) | 3.5 | `FETCH_HI+k` | `OUTPUT_HI_THIS_STEP+k` | 0.02 |
| `post_l9_bnz_lo_nonzero_*` | MARK_PC(+1) + OP_BNZ(+0.2) + CMP+4(−1) | 1.5 | (same patterns) | (same) | 0.02 |
| `post_l9_bnz_hi_nonzero_*` | MARK_PC(+1) + OP_BNZ(+0.2) + CMP+4(+1) + CMP+5(−1) | 2.5 | (same) | (same) | 0.02 |

Two important properties:

1. **Cancel bands always write OUTPUT\_LO+k and OUTPUT\_HI\_THIS\_STEP+k
   at k=0..15.** The gate reads the **previous step's** OUTPUT band
   (cross-step `.*.-1` alias) and negates it; on step 0 the alias
   resolves to 0 so the cancel writes 0.
2. **Direct-copy bands write OUTPUT_LO+k driven by FETCH_LO+k.** This
   is the new-PC byte copy that REPLACES the cancelled previous PC
   byte. The gate value is the raw FETCH_LO+k residual at the current
   token position.

## Why this op fires at MEM_addr1 step 0 of var_simple

The failing fingerprint is:

```
first_token_divergence=step0:MEM_addr1 abs=119 gen=27
  expected=0xff neural=0x00
block_index=35 (label 'block35 layer=25 width=192' — see Note A)
expected_logit=-5.00 argmax_logit=+21.49 margin=-26.49
OUT_LO[15]=-0.00 arg=0/+2.55   OUT_HI[15]=-0.00 arg=0/+2.55
```

The expected token is `0xff` (low nibble F + high nibble F → bands LO[15]
and HI[15] should win). The neural argmax is `0x00` because **bands
LO[0] and HI[0] each carry +2.55 logit at this position** while
LO[15] and HI[15] are at 0.0. The 192-unit L34 block is identified by
the residual-support trace as the per-block contributor that flips the
winner.

Static rule inspection shows none of the 192 rules should fire on a
MEM_addr1 slot:

* Cancel-band threshold = 3.5 but max condition sum =
  MARK\_PC(1) + OP\_BZ(0.2) + CMP+4(1) + CMP+5(1) = **3.2** even when
  every flag is saturated at 1.0, so the cancel band is provably dead
  given clean unary-flag residuals.
* BZ-direct-copy threshold = 3.5 with the same conditions plus
  `MARK_STACK0=-10`; on MEM_addr1 step 0 there is no MARK_PC firing
  so the score is ≤ 0 ≪ 3.5.
* BNZ groups have lower thresholds (1.5, 2.5) but still gated by
  MARK\_PC > 0; on MEM_addr1 step 0 MARK_PC=0.

So the **static** analysis predicts the op contributes 0 at this slot
yet the runtime residual carries +2.55. Two leak paths are plausible:

1. **MARK_PC residual bleed.** The MARK_PC dim is co-allocated at a
   small position (dim_registry_dynamic.py:66 → position 0). Any
   upstream writer that leaves a non-zero value at MARK_PC on the
   MEM_addr1 slot will lift the condition score. Even a +0.3 residual
   in MARK_PC pushes the cancel-band sum from 3.2 → 3.5 and the rule
   starts firing. Each fire writes `0.02 × gate_value` into
   OUTPUT_LO+k. With 16 lo-cancel + 16 lo-target rules per group ×
   3 groups (BZ + 2× BNZ) = 96 rules firing on OUTPUT_LO+0 alone, and
   gate values potentially > 1 (FETCH_LO is a residual sum, not a
   normalised flag), +2.55 is reached at gate ≈ 1.3.

2. **FETCH_LO+0 amplification.** Even if pre-up is small but positive
   (silu output ~ 0.1), the gate value FETCH_LO+0 can be > 5 because
   FETCH_LO is the raw 16-band token encoding of the fetch byte. If the
   PC fetch at step 0 has byte0 = 0x00, FETCH_LO+0 is the nibble-0
   indicator and can saturate. Product across 6 active target rules ×
   0.02 × 0.1 × 5 ≈ 0.06 per band — small per-rule but stacks across
   the LO+0 column on the 6 rules that write to it. Insufficient on
   its own — leak path 1 (MARK_PC bleed) is the more likely root.

`verify_rule_strength` + `verify_rule_scopes` report **0 violations**
(see `scripts/verify_l34_op.py`); the static gate is clean. The leak is
upstream-residual-dependent and not detectable by the per-op declarative
verifier.

## Why margin grew 3x vs the prior var_three signature

`VAR_THREE_ATTRIBUTION_REPORT.md` pinned the issue at
`tail_mem_store_addr1_ff_from_stack_store_exact` (a rule INSIDE
`tail_bit32_result_correction`, L33.ffn / block 33 / width 2059) with
`+0.78` on OUTPUT_LO[0] and margin `-5.64`. The new signature has
`+2.55` on OUTPUT_LO[0] and margin `-26.49`.

Interpretation: the prior attribution was about the **0xFF emitter
undershooting** (the legitimate 0xFF writer at L33 was too weak at
+0.78 vs +5.0 target). The new attribution is about a **0x00 emitter
overshooting** (this L34 op is leaking +2.55 into band 0). Net delta:
the 0xFF write is no longer "weak" — it's been overwhelmed by a
stronger band-0 writer that emerged at L34, downstream of L33.

Two reinforcing changes in the L10/L16 DSL waves landed since the
VAR_THREE doc was written that plausibly explain this:

* L10 `_strengthen_l10_addsub_wrong_byte_blockers` / similar
  helpers increased MARK_PC residual amplification (per smoke
  notes). Larger residuals downstream are exactly what unlocks the L34
  leak.
* L13/L14 attention head migrations (commits `0c5ce4d`, etc.) altered
  how FETCH_LO survives into the post-L9 stream. If FETCH_LO survives
  more cleanly to L34, the L34 gate values grow.

## Comparison against the prior var_three attribution

| Aspect | VAR_THREE doc | This report (var_simple at HEAD) |
|---|---|---|
| Op | `tail_mem_store_addr1_ff_from_stack_store_exact` | `post_l9_bz_bnz_pc_override` |
| File:line | `neural_vm/.../l10_ops.py:5369` | `neural_vm/.../l6_ops.py:4547` |
| Block (label) | block 34 width 2059 (L33.ffn) | block 35 (35-of-36? Note A) width 192 (L34.ffn) |
| Direction of fault | rule UNDER-fires writing 0xFF into LO/HI[15] | rule OVER-fires writing 0x00 into LO/HI[0] |
| Margin observed | -5.64 | -26.49 |
| OUT_LO[0] mass | +0.78 | +2.55 |
| Static verifier finding | strength violation against L34 competitor | 0 strength_violation, 0 scope_violation |

**It is a different op, not a successor pattern.** The prior issue
(weak 0xFF writer at L33) may still exist in the background but is no
longer the binding constraint; the binding constraint is now a stronger
band-0 emitter at L34 that the L33 writer cannot beat.

## Verifier output

`scripts/verify_l34_op.py`:

```
op=post_l9_bz_bnz_pc_override rules=192
strength_violation: 0 (runtime 4.8s)
no_dominates_at: 0
scope_violation: 0 (runtime 0.0s)
```

The op is statically clean. Dynamic competition vs.
`tail_bit32_result_correction`, `layer14_temp_clear`,
`layer14_clear_output_corruption`,
`layer14_clear_mem_marker_output` finds no per-output-dim shortfall.
The leak is **runtime residual leakage into MARK_PC**, not a
declarative gate flaw.

The compile-time `produces/consumes_fresh` machinery does not flag
this op either; its `reads`/`writes` declaration is consistent with
its rule set (writes OUTPUT_LO + OUTPUT_HI_THIS_STEP, reads MARK_PC,
OP_BZ, OP_BNZ, CMP, IS_BYTE, FETCH_LO/HI, OUTPUT_LO.*.-1,
OUTPUT_HI_THIS_STEP).

## Concrete next-step hypotheses (no fix attempted)

Three independent levers, ranked by historical "single-rule whack-a-mole
zero-sum" risk:

1. **Add a step-0 guard.** All BZ/BNZ override rules are
   semantically meaningful only on step ≥ 1 (the cancel band reads the
   previous step's OUTPUT). Add an explicit `STEP_INDEX > 0` (or
   `OUTPUT_LO.*.-1+k.valid`) condition that hard-zeroes the rule on
   step 0. This is the cheapest fix; risk: zero — step 0 cannot
   legitimately fire a BZ-taken override because there's no preceding
   compare. **Recommended first.**

2. **Add MARK_MEM / IS_BYTE blocker.** The target rules already
   exclude MARK_STACK0. Add `MARK_MEM=-1000000` and `MARK_AX/SP/BP=-1000000`
   to both cancel and target conditions to make the rule fire EXCLUSIVELY
   at MARK_PC slots. This is the canonical `tail_*` exclusion pattern
   (see l10_ops.py:5410-5415). Risk: low; the rule semantically only
   applies at MARK_PC and the test suite for BZ/BNZ is small (smoke
   only). **Recommended second.**

3. **Threshold increase.** Raise the cancel threshold from 3.5 to
   ~4.5 to make MARK_PC residual leakage less catastrophic. Requires
   re-baking the cancel band weights and may need a follow-up `gate_bias`
   adjustment to preserve the BZ-taken legitimate fire. Risk: moderate
   — could weaken the legitimate BZ-taken path that the op was created
   to fix. **Not recommended without first measuring MARK_PC residual
   distribution at MEM_addr1 slot.**

4. **Upstream MARK_PC residual clean-up.** The actual root cause is
   probably an upstream writer leaking MARK_PC residual into the
   MEM_addr1 slot. A `tools/decl_verifier.py` sweep over MARK_PC
   writers, gated on `(slot=MEM_addr1, step=0)`, would identify it.
   This is the **zero-sum-safe** fix per the memory note
   `feedback_single_rule_fixes_are_zero_sum`. **Recommended for the
   real fix wave; out of scope here.**

## Notes

**Note A — block index discrepancy.** The runtime diag log
(`/tmp/diag_chunk_250_32.log`) reports `block=35 layer=25 label='block35
layer=25 width=192'`. Our compile in this worktree (at HEAD
`cef9d046`, same commits as the diag's `c3c0dfd0` for compiler / vm
files) shows 35 final blocks indexed 0..34, with the 192-unit FFN at
index 34. The diag's `block=35` reading is therefore one larger than
this worktree's index. The compiler/vm code was not modified between
`c3c0dfd0` and `cef9d046` so the index drift is most likely a
diagnostic-label quirk (e.g. 1-indexed in the label formatter or a
different ALU-mode env at the diag run). The 192-unit width is unique
across the model regardless of the index naming; the op identification
is unambiguous.

**Note B — `produces/consumes_fresh`.** Step 5 of the dynamic schedule
work (commit `5f51cecb`) derives produces/consumes_fresh from rules.
The op is annotated correctly (`writes={"OUTPUT_LO",
"OUTPUT_HI_THIS_STEP"}`, `reads` includes `OUTPUT_LO.*.-1`); the
verifier has no complaint.

## Artifacts

* `scripts/dump_l34_attribution.py` — compile-and-dump helper used to
  pin the op. Single compile run.
* `scripts/verify_l34_op.py` — `verify_rule_strength` +
  `verify_rule_scopes` driver for the op. Reports 0/0/0.
* `/home/alexlitz/.claude/projects/-home-alexlitz-Documents-misc-c4-release/6350c6b6-bcb2-4954-8f37-32ed51f34783/tool-results/bm4b5sdbn.txt`
  — raw stdout of the compile (right-size census + per-block FFN
  enumeration).
