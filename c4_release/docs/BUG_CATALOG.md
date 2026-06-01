# Bug catalog (consolidated)

Canonical, single-page enumeration of the 32 bugs tracked across the
2026-05-29..2026-05-31 multi-batch campaign on
`speedup-cache-and-buckets`. Bugs 1-26 come from the original C9-v1
summary (commit `bcb0fd1`, Section 3); bugs 27-32 come from the C9-v2
update (commit `de26008`, Section 5) which appended after the
B7-6/B7-7 merges and the B7-9 lowering revalidation on
`integration/batch-merge-full-v4`.

**Status legend.** O = open; U = under investigation; F = fixed and
landed; D = diagnosed with PLAN / REPORT but no fix landed; R = known
regression. (The original catalog only used F / D / U; the broader
legend was added for bugs 27-32 and is back-applied where the prior
note explicitly distinguished a fix-vs-doc state.)

---

## Bugs 1-26 (original catalog, C9-v1 summary @ `bcb0fd1`)

### Bug #1: `tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority` overpowers ENT-main
- **Source**: original catalog (C9-v1, `bcb0fd1`); discovered during B2-A / B3-eta.
- **Symptom**: 5e9-strength L10 tail rule out-votes the ENT-main path on `tail_mem_store_addr0_e0_*` rows.
- **Affected**: ENT step set.
- **Root cause**: strength-escalating addr0 rule with no upstream gate.
- **Status**: F.
- **Fix branch / commit**: landed in the B5/B6 sweep; see Section 4 of the C9-v1 summary (cross-refs below).
- **Cross-ref**: bugs #2, #3, #7, #13.
- **File**: `l10_ops.py:3889`.

### Bug #2: `tail_mem_store_addr0_00_from_global_exact` false-fires at step 5 (OUTPUT proxy ambiguity)
- **Source**: B2-A diagnosis.
- **Symptom**: global-store ids miscompile at step 5 because the OUTPUT proxy is ambiguous.
- **Affected**: global-store ids.
- **Root cause**: OUTPUT_LO / OUTPUT_HI proxy reads have no structural gate.
- **Status**: F (via `1ef127c`, plus B5-D partial reroute through L13 ADDR_B0).
- **Cross-ref**: bug #12 (ADDR_B0 staleness), #27 (soft ADDR_B0 evidence regression).
- **File**: `l10_ops.py:3696`.

### Bug #3: `tail_sp_marker_byte0_f8_from_initial_stack_exact` fires on residual MARK_SP > 1.0
- **Source**: B3-gamma + B5-J.
- **Symptom**: initial-SP ids miscompile because the SP marker rule fires on residual MARK_SP.
- **Affected**: initial-SP ids.
- **Root cause**: no `SP_BYTE0_IS_F8` upstream gate; rule uses OUTPUT proxy.
- **Status**: D (B7-6 plan); upstream slot 95 allocated in `35e0d6b`; consumer rewrite B7-6 (`cdd0d0a`) — see catalog v2 for landed status.
- **Cross-ref**: bugs #7, #13, #14.
- **File**: `l10_ops.py:3465`.

### Bug #4: `MEM_ADDR_SRC=1` injection wrongly fired on PSH/POP (not SI/SC)
- **Source**: B-cycle review of L7 head 7.
- **Symptom**: PSH/POP store/load ids miscompile because MEM_ADDR_SRC mis-injects.
- **Affected**: PSH/POP store/load ids.
- **Root cause**: MEM head 7 lacked SI/SC opcode predicate; injected on every store/load.
- **Status**: F (`0316600`).
- **File**: `l7_ops.py` MEM head 7.

### Bug #5: Current-step MEM marker missing in batched runner (causes false STACK0 attention)
- **Source**: Recovery B2-H.
- **Symptom**: STACK0 attention misfires only on the batched runner path.
- **Affected**: batched-only ids.
- **Root cause**: batched runner didn't emit the current-step MEM marker that the single-step runner did.
- **Status**: F (`e2f1e8e`, Recovery B2-H).
- **File**: batched runner.

### Bug #6: L11 -> L12 wide-MUL amplitude mismatch
- **Source**: MUL pipeline diagnosis.
- **Symptom**: wide-MUL outputs at wrong amplitude.
- **Affected**: MUL ids.
- **Root cause**: amplitude scaling mismatch between L11 producer and L12 consumer.
- **Status**: F (`cc55474`).
- **Cross-ref**: bug #20 (MUL declaration alignment regression).
- **File**: `alu_ops.py` MUL pipeline.

### Bug #7: L10 SP-marker rule circular OUTPUT self-amp (B5-J's `HAS_SE -1e9` was a hack)
- **Source**: B5-J investigation (`d0065e8`).
- **Symptom**: SP marker rule self-amplifies on OUTPUT; B5-J tried HAS_SE -1e9 hack and saw -5 regression.
- **Affected**: initial-SP ids (still pending).
- **Root cause**: SP_BYTE0_IS_F8 not available as an upstream signal; rule resorts to a circular OUTPUT proxy.
- **Status**: D (B7-6 plan); B7-6 (`cdd0d0a`) rewrote the family using slots 95/96 — see v2 status.
- **Cross-ref**: bugs #3, #13.
- **File**: `l10_ops.py:3465`.

### Bug #8: L4 `sp_to_addr_key` mis-flagged as enabled when STACK0 mem-attention disabled
- **Source**: B3-θ audit.
- **Symptom**: per-op declaration claims `sp_to_addr_key` enabled when its consumer attention path is gated off.
- **Affected**: none directly; unblocks diagnosis.
- **Root cause**: L8 STACK0 mem-attention `enable=False`; L4 flag wasn't updated.
- **Status**: F-as-absent (`f461e57`, flagged disabled).
- **File**: L4 op (`l4_ops.py`).

### Bug #9: spec_k cache corruption on draft VM divergence
- **Source**: B1 spec-decoding regression suite.
- **Symptom**: spec_k cache corrupted when draft and target VMs diverge.
- **Affected**: spec-decoding regression suite.
- **Root cause**: spec engine kept stale draft KV after divergence.
- **Status**: F (`e7c4d63`, fallback + draft MEM marker).
- **File**: spec engine.

### Bug #10: KV eviction max_tokens corruption on bounded batches
- **Source**: B5-G revalidation.
- **Symptom**: bounded-batch runs corrupt the eviction max_tokens accounting.
- **Affected**: KV-eviction suite.
- **Root cause**: runner double-counted across batch boundaries.
- **Status**: F (`4d069f7` validated by B5-G).
- **File**: runner.

### Bug #11: L17 hidden FFNs (`l10_post_ops_combined` + `tail_bit32_result_correction`) mis-attributed to L17
- **Source**: B6-L inventory.
- **Symptom**: layout names "L17 / block27 / layer27" but the FFN factory actually comes from `l10_ops.py`.
- **Affected**: none directly; investigation primer for future block-NN renaming.
- **Root cause**: `vm_step.py:_expand_wrapper_blocks` indirection.
- **Status**: D (B6-L inventory).
- **Cross-ref**: bug #21.
- **File**: layout `vm_step.py:2627`.

### Bug #12: ADDR_B0 staging stale at MARK_MEM rows (residual leak from MARK_STACK0)
- **Source**: L8 SP-gather audit.
- **Symptom**: ADDR_B0 stays at the MARK_STACK0 value through MARK_MEM rows.
- **Affected**: tail-addr family.
- **Root cause**: L8 SP-gather fired at MARK_STACK0 instead of MARK_SP.
- **Status**: F partial (`a9a45b6` fires SP gather at MARK_SP); full fix via B7-3/B7-5 lifecycle bits.
- **Cross-ref**: bugs #15, #16.
- **File**: `l8_ops.py` SP gather.

### Bug #13: L10 OUTPUT_LO+8 / OUTPUT_HI+15 proxy for SP byte 0 = 0xF8 (no structural signal)
- **Source**: B7 architectural review.
- **Symptom**: SP marker family lacks any structural signal for "SP byte 0 is 0xF8".
- **Affected**: SP marker family.
- **Root cause**: no `SP_BYTE0_IS_F8` upstream dim.
- **Status**: D — slot 95 allocated at `e00709f` (later confirmed via B7-2 `35e0d6b`); consumer rewrite B7-6.
- **Cross-ref**: bugs #3, #7.
- **File**: `l10_ops.py:3465`.

### Bug #14: No `IN_STEP_FRESH` lifecycle bit -> tail rules can't distinguish fresh vs. residual
- **Source**: B6-G structural-signal audit.
- **Symptom**: tail rules cannot distinguish a fresh-this-step signal from residual marker activations.
- **Affected**: tail family.
- **Root cause**: missing L1 attn head emitting "fresh-this-step" gating bit.
- **Status**: F (`2da184b` slot 96 allocated, B7-1); consumer pending (B7-6/B7-7).
- **File**: L1 attn (head 5 extension).

### Bug #15: No `ADDR_B0_VALID` lifecycle bit -> L10 family can't gate on freshness
- **Source**: B6-G structural-signal audit.
- **Symptom**: L10 tail-addr family can't gate on whether L13 gather actually fired this step.
- **Affected**: tail-addr family.
- **Root cause**: L13 gather did not emit a validity bit; only emitted the value.
- **Status**: F (`e00709f` slot 97 allocated, B7-4); consumer pending.
- **File**: L13 gather.

### Bug #16: No `SP_GATHERED_THIS_STEP` lifecycle bit
- **Source**: B6-G structural-signal audit.
- **Symptom**: SP-marker family can't tell whether SP was gathered this step or carried from a prior step.
- **Affected**: SP-marker family.
- **Root cause**: L8 gather emitted no validity bit.
- **Status**: F (`3fe4ffa` slot 98 allocated, B7-5); consumer pending.
- **File**: L8 gather (at MARK_SP).

### Bug #17: L0 H5/H6/H7 attention heads dead (write, no reader, 21 dims wasted)
- **Source**: B6-K BD dim usage map (`8d3ee75`).
- **Symptom**: 21 dims allocated to L0 attention heads with no downstream consumer.
- **Affected**: reclaim 21 dims (4 reclaimed by B7-1..5; 17 remain).
- **Root cause**: heads have writers but no readers in the current op graph.
- **Status**: D (B6-K usage map).
- **Cross-ref**: bug #5 in v2 recommendations (slots 99-115 reclamation).
- **File**: `l0_ops.py:141`.

### Bug #18: L16 `e0`/`f0` STACK0 marker fails to preserve post-PSH stack top across IMM
- **Source**: B6-D.
- **Symptom**: post-PSH IMM rows fail because STACK0 marker doesn't survive the IMM boundary.
- **Affected**: post-PSH IMM ids.
- **Root cause**: L16 marker rule was not IMM-aware.
- **Status**: F (`6f8920b`).
- **File**: `l16_ops.py`.

### Bug #19: L10 OP_IMM AX byte 1 not zeroed
- **Source**: rule audit.
- **Symptom**: AX byte 1 carries residual value on OP_IMM rows.
- **Affected**: IMM AX-byte-1 ids.
- **Root cause**: missing zero rule on `tail_ax_imm_byte1_hi_zero`.
- **Status**: F (`cb110ce`).
- **File**: `l10_ops.py` (`tail_ax_imm_byte1_hi_zero`).

### Bug #20: L11/L12 MUL declaration alignment regression
- **Source**: MUL regression suite.
- **Symptom**: MUL pipeline regressed after an unrelated change.
- **Affected**: MUL regression suite.
- **Root cause**: L11/L12 declarations drifted out of alignment.
- **Status**: F (`b29d3ed`).
- **Cross-ref**: bug #6.
- **File**: `alu_ops.py` MUL.

### Bug #21: Block27/layer27 = L17 — B4-B misattribution
- **Source**: B4-B / B6-L follow-up.
- **Symptom**: investigations into block27 / layer27 were chasing the wrong layer.
- **Affected**: investigation primer; no row impact.
- **Root cause**: `_expand_wrapper_blocks` renaming.
- **Status**: D (B6-L inventory).
- **Cross-ref**: bug #11.
- **File**: `l10_ops.py:1289`, `l10_ops.py:4529`.

### Bug #22: L7 head-5 ENT-SP fetch-gate limitation
- **Source**: L7-L9 structural audit (`846d8fe` documented).
- **Symptom**: ENT-SP ids cannot route through L7 head 5 because of a fetch-gate limitation.
- **Affected**: ENT-SP ids; unresolved.
- **Root cause**: head 5 gate excludes the ENT-SP pattern.
- **Status**: D (documented at `846d8fe`); U (no fix landed).
- **File**: `l7_ops.py` head 5.

### Bug #23: B3-gamma MARK_SP/OUTPUT inversion (rejected; can't be solved at L10)
- **Source**: B3-gamma (`5ef58b0` rejection).
- **Symptom**: attempting to invert MARK_SP and OUTPUT inside L10 produces no working signal path.
- **Affected**: initial-SP ids (structural fix via B7-6).
- **Root cause**: not L10-solvable; needs upstream `SP_BYTE0_IS_F8`.
- **Status**: D (rejection rationale at `5ef58b0`); structural fix landed in B7-6.
- **Cross-ref**: bugs #3, #7, #13.
- **File**: `l10_ops.py:3465-3502`.

### Bug #24: ONNX runtime path: cummax + dynamo guard broken
- **Source**: ONNX export audit (`7d4a5e3`).
- **Symptom**: ONNX export pipeline fails on cummax and a dynamo guard.
- **Affected**: export pipeline.
- **Root cause**: unsupported ops in ONNX path.
- **Status**: U.
- **File**: exporter.

### Bug #25: C runtime path: build broken, no baseline
- **Source**: C runtime audit (`48c13a6`).
- **Symptom**: C runtime build fails.
- **Affected**: deployment validation outside PyTorch reference.
- **Root cause**: build infrastructure.
- **Status**: U.
- **File**: C runtime.

### Bug #26: `absdiff` (0/25) and `nested_quad` (0/16) categories entirely dead
- **Source**: B3 investigation (`a8924be`).
- **Symptom**: 41 ids in two test categories produce zero passes.
- **Affected**: 41 ids; root cause unresolved.
- **Root cause**: suspected missing compiler path; bug #27 (SP_byte0 step 2) likely partial cause per B7-9 audit.
- **Status**: D (surface documented); U (root cause unidentified).
- **Cross-ref**: bug #27, #29.
- **File**: `tests/` (`absdiff_*`, `nested_quad_*`).

---

## Bugs 27-32 (C9-v2 update, current `CAMPAIGN_SUMMARY.md` @ `de26008`)

### Bug #27: L10 soft ADDR_B0/B1/B2 evidence (B5-D + B6-B) misfires on SP byte 0 step 2
- **Source**: B7-9 audit (`6e8ab77`); D1 retest shard concentration (`audit/post-merge-retest-a03f600`).
- **Symptom**: SP_byte0 first-fatal count doubled from 461 to 893 corpus-wide; concentrates +432 across `func_*`/`rec_*`/`nested_*`/`absdiff_*`.
- **Affected**: ~150-250 ids in shards 137 / 274 / 822 of the 1096 corpus.
- **Root cause**: soft ADDR_B0/B1/B2 evidence reads added by B5-D (`66d9e12`) and B6-B (`d4b2a90`) fire at partial strength on rows where they should abstain.
- **Status**: R (regression with revert candidate D2 and targeted candidate D3 scoped; neither landed at `a03f600`).
- **Fix branch / commit (if any)**: `fix/revert-l10-soft-addr-b0-evidence` (D2, scoped); `investigation/sp-byte0-regression-source` (D3, scoped).
- **Cross-ref**: bugs #2, #12, #29 (subsumed), #26 (partial), #28/#30/#31 (sibling new buckets).

### Bug #28: New `step1:STACK0_byte2` cluster on `var_three_*` rows (25 cases)
- **Source**: B7-9 audit, slice 274-547.
- **Symptom**: STACK0_byte2 is the first fatal slot on 25 `var_three_*` rows; this bucket did not exist pre-batch.
- **Affected**: `var_three_*` family.
- **Root cause**: unidentified; not subsumed by bug #27.
- **Status**: U.
- **Cross-ref**: bugs #27 / #30 / #31 (sibling new buckets).

### Bug #29: New `step2:SP_byte0` cluster on `func_*`/`rec_*`/`nested_*`/`absdiff_*` (~275 cases)
- **Source**: B7-9 audit, slices 548-821 + 822-1095.
- **Symptom**: SP_byte0 is the first fatal lane at step 2 on ~275 function-call-family rows.
- **Affected**: function-call families (`func_*`, `rec_*`, `nested_*`, `absdiff_*`).
- **Root cause**: likely the same SP-lane lowering bug as #27; #29 is the rowset view of #27's first-fatal histogram.
- **Status**: U (likely subsumed by #27).
- **Cross-ref**: bug #27 (probable parent), #26 (dead categories partially explained).

### Bug #30: New `PC_byte0` cluster (42 cases) in slice 548-821
- **Source**: B7-9 audit.
- **Symptom**: PC_byte0 is the first fatal lane on 42 rows in slice 548-821; new bucket.
- **Affected**: unidentified family within slice 548-821.
- **Root cause**: unknown; independent of soft ADDR_B0 hypothesis per the v2 summary.
- **Status**: U.

### Bug #31: New `PC_byte1` cluster (46 cases) in slice 822-1095
- **Source**: B7-9 audit.
- **Symptom**: PC_byte1 is the first fatal lane on 46 rows in slice 822-1095; new bucket.
- **Affected**: unidentified family within slice 822-1095.
- **Root cause**: unknown; independent of soft ADDR_B0 hypothesis.
- **Status**: U.

### Bug #32: 1069 / 1096 rows produce >= 1 lowering fatal; only 27 rows are info-only / clean
- **Source**: B7-9 audit.
- **Symptom**: the 238 / 1096 declarative pass count reflects "first-fatal does not propagate to final OUTPUT" rather than "model lowers cleanly". Of 1096 rows, only 27 are clean (19 info + 8 errored); the remaining 1069 have at least one fatal.
- **Affected**: corpus-wide observation, not a localized bug.
- **Root cause**: by construction — pass rate is a downstream-tolerance metric, not a lowering-cleanliness metric.
- **Status**: D (documented as a metric caveat).

---

## Open vs fixed summary

- **Fixed (F)**: 1, 2, 4, 5, 6, 9, 10, 14, 15, 16, 18, 19, 20 — 13 bugs.
- **Partial / fix-as-absent**: 8, 12 — 2 bugs.
- **Documented / diagnosed, no fix landed (D)**: 3, 7, 11, 13, 17, 21, 22, 23, 26 (also U), 32 — 10 bugs.
- **Under investigation (U)**: 22 (also D), 24, 25, 26 (also D), 28, 29, 30, 31 — 8 bugs.
- **Regression with revert candidate (R)**: 27 — 1 bug.

Bugs 3 / 7 / 13 / 23 share an upstream-signal lineage and are functionally
addressed by the B7-1..5 + B7-6/B7-7 architectural cleanup landed in
`a03f600`; the C9-v1 summary status for them is "D" because the
*consumer-side* rewrite was still pending at `bcb0fd1`.

---

## Top 5 by impact (row count where available)

1. **Bug #27** — L10 soft ADDR_B0 misfire on SP_byte0 step 2: ~150-250 ids; dominant residual at `a03f600`.
2. **Bug #29** — `step2:SP_byte0` cluster on `func_*`/`rec_*`/`nested_*`/`absdiff_*`: ~275 first-fatal cases (likely subsumed by #27).
3. **Bug #32** — corpus-wide lowering health: 1069 / 1096 rows carry >=1 fatal (metric caveat).
4. **Bug #26** — `absdiff` (0/25) + `nested_quad` (0/16) dead categories: 41 ids; partially explained by #27.
5. **Bug #31** — new `PC_byte1` cluster in slice 822-1095: 46 cases.

(Bug #30 — `PC_byte0` 42 cases — and bug #28 — `STACK0_byte2` 25 cases —
are the next two by raw count.)

The original C9-v1 catalog rolls per-batch contributions up to a
~150-200 ids unblocking estimate for the L10 tail family (bugs #1-3,
#7, #12-16); the B7-6/B7-7 merges shipped a +9 delta on the D1
retest, with the bulk of the rest gated on bug #27 D2/D3.

---

## What's still untracked

Several failure modes are described in nearby docs but **never
assigned a bug number** in either the C9-v1 or C9-v2 catalog, so they
do not appear above and should not be confused with "fixed" / "open"
bug numbers in the table. These include: the wide-byte ALU
MUL/DIV/SHL/SHR long tail (the C9-v1 "rough category breakdown"
estimates ~120-150 ids here; only bugs #6 and #20 enumerate
sub-issues); stack / JSR / LEV protocol edges (~80-100 ids, no bug
numbers); the conv-I/O / PRTF pure-neural path (~150-200 ids, post-V9
design `feddfc1` landed but the rowset was out of scope for this
campaign); long-tail single-id regressions (~100-150 ids, no
enumeration). The user-memory note
`project_l10_psh_addr_ent_bug.md` documents an open L10 PSH addr0_e0
missing `OP_ENT` guard at `l10_ops.py:3888-3927` that blocks
`func_identity_*` and likely many others; it is not numbered here
because no campaign batch assigned it a slot. Similarly,
`project_l16_bp_frame_byte1_ff_dual.md` (L16 dual victim/aggressor
finding for `if_var`) and `project_l3_sp_byte0_dormant.md` (L3
SP_byte0 rewrite function dormant — actual logic at
`vm_step.py:4604+`) describe substrate-level bugs that have not been
folded into the numbered catalog. Any future cluster-fix wave should
either renumber these as #33+ or treat them as separate substrate
bugs.

---

## Sources consulted

- `c4_release/docs/CAMPAIGN_SUMMARY.md` at HEAD (`de26008`, C9-v2) — Section 5 supplied bugs #27-#32 verbatim.
- `c4_release/docs/CAMPAIGN_SUMMARY.md` at `bcb0fd1` (C9-v1) — Section 3 supplied bugs #1-#26 verbatim; Sections 4-7 supplied per-bug context (root cause, fix branch, affected ids).
- Git commit messages on `speedup-cache-and-buckets`: `bcb0fd1` (C9-v1 cover note) and `de26008` (C9-v2 cover note).
- B7-9 lowering revalidation: `git show 6e8ab77:.agent-logs/lowering-revalidation-v4/SUMMARY.md` (cited by C9-v2 for bugs #27-#32; first-fatal histograms).
- D1 post-merge retest: `git show origin/audit/post-merge-retest-a03f600:.agent-logs/post-merge-retest-a03f600/_run.log` (cited by C9-v2 for the 238 / 1096 pass count and shard breakdown).
- B6-G L7-L9 audit, B6-K BD dim usage map, B6-L L17 post-op inventory, B4-H L10 refactor PLAN — referenced as cross-refs in both C9-v1 and C9-v2; not re-read for this catalog (already summarized in the parent summaries).
- User-memory notes (for the "untracked" section, not the numbered catalog): `project_l10_psh_addr_ent_bug.md`, `project_l16_bp_frame_byte1_ff_dual.md`, `project_l3_sp_byte0_dormant.md`, `project_1096_sentinel_baseline.md`, `feedback_single_rule_fixes_are_zero_sum.md`.

This catalog covers **all 32 numbered bugs** referenced by the two
campaign summaries; no bug numbers 1-32 are missing. The C9-v1
summary explicitly states "26-row bug catalog" in its commit message
(`bcb0fd1`), so #1-#26 is the canonical original set and #27-#32 are
the post-merge appendix — there is no hidden lower range (e.g., bugs
1-7 absent from any reachable doc); the C9-v1 enumeration starts at
#1 and is complete through #26.
