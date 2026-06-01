# Bug catalog (consolidated)

Canonical, single-page enumeration of the 38 bugs tracked across the
2026-05-29..2026-06-01 multi-batch campaign on
`speedup-cache-and-buckets`. Bugs 1-26 come from the original C9-v1
summary (commit `bcb0fd1`, Section 3); bugs 27-32 come from the C9-v2
update (commit `de26008`, Section 5) which appended after the
B7-6/B7-7 merges and the B7-9 lowering revalidation on
`integration/batch-merge-full-v4`; bugs 33-38 come from the round-4
parallel-debug wave (2026-06-01) documented in
`CAMPAIGN_SUMMARY.md` §10 and the gitignored triage docs
`.agent-logs/{wide_alu,stack_jsr_lev,conv_io_prtf,attribution,no_claims_survey}_2026_06_01.md`.

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
- **Status**: F (`cc55474`) **with regression as of 2026-06-01** — `.agent-logs/wide_alu_triage_2026_06_01.md` §6 documents 3 `MUL_direct::multi_byte_mul_wrong` rows (`mul_20`, `mul_29`, `mul_36`) where the constant non-trivial offset suggests a regression of the original fix or a co-occurring new mechanism. Sibling new bug #35 covers the high-byte-loss shape.
- **Cross-ref**: bug #20 (MUL declaration alignment regression), bug #35 (high-byte loss).
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
- **Status**: F (`b29d3ed`) **with regression as of 2026-06-01** — `.agent-logs/wide_alu_triage_2026_06_01.md` §6 documents ~46 rows where `a*b ≤ 255` is also failing (`MUL_*::single_byte_mul_wrong`); single-byte MUL should not need the wide-MUL pipeline at all. This is a single-byte-side regression of #20 and is now carried separately by bug #34 (uniform-216 sentinel) which captures the rowset.
- **Cross-ref**: bug #6, bug #34 (uniform-216 sentinel sub-set).
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
- **Root cause**: suspected missing compiler path; bug #27 (SP_byte0 step 2) likely partial cause per B7-9 audit. **Re-attributed 2026-06-01**: per `.agent-logs/stack_jsr_lev_triage_2026_06_01.md`, all 25 `absdiff_*` rows and all 25 `nested_quad_*` rows now first-fatal on the post-LEV `step6:AX_byte0` / `step6:STACK0_byte0` slots that motivate bug #33 (post-LEV AX corruption). Fixing #33 is expected to unblock the bulk of these.
- **Status**: D (surface documented); partially attributed to #33 (status: U on the root cause itself; bug #29 remains a probable contributing factor on the pre-call SP_byte side).
- **Cross-ref**: bug #27, #29, **#33** (probable parent for the post-LEV first-fatal).
- **File**: `tests/` (`absdiff_*`, `nested_quad_*`).

---

## Bugs 27-32 (C9-v2 update, current `CAMPAIGN_SUMMARY.md` @ `de26008`)

### Bug #27: L10 soft ADDR_B0/B1/B2 evidence (B5-D + B6-B) misfires on SP byte 0 step 2
- **Source**: B7-9 audit (`6e8ab77`); D1 retest shard concentration (`audit/post-merge-retest-a03f600`).
- **Symptom**: SP_byte0 first-fatal count doubled from 461 to 893 corpus-wide; concentrates +432 across `func_*`/`rec_*`/`nested_*`/`absdiff_*`.
- **Affected**: ~150-250 ids in shards 137 / 274 / 822 of the 1096 corpus.
- **Root cause**: soft ADDR_B0/B1/B2 evidence reads added by B5-D (`66d9e12`) and B6-B (`d4b2a90`) fire at partial strength on rows where they should abstain.
- **Status (2026-06-01)**: R, **partially attacked**. The D2 revert landed at `361357a` (merged at `65f80f1`) was a 2-line cleanup with **near-zero impact**; B7-7 had already absorbed most of B5-D + B6-B during the original consumer rewrite. The "expected +50 to +75 ids" forecast (C9-v2 §4) is revised down to ~0. The residual SP_byte0/1 cluster is open; the attention-side root is now better attributed by new bug **#37** (L8 SP-gather attention strength violations). The targeted candidate D3 (`investigation/sp-byte0-regression-source`) was not landed and is superseded by #37.
- **Fix branch / commit (if any)**: D2 revert landed `361357a` → `65f80f1` (near-zero impact); D3 scoped but superseded.
- **Cross-ref**: bugs #2, #12, #29 (superseded), #26 (partial), #28/#30/#31 (sibling new buckets), **#37** (attention-side root).

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
- **Status (2026-06-01)**: **superseded** for the function-call family. Per `.agent-logs/attribution_2026_06_01.md`, `func_*` / `nested_*` / `rec_*` first-fatal slots are now `step6:AX_byte0` / `step6:STACK0_byte0` (post-LEV, attributed to #33) rather than `step2:SP_byte0`. The pre-call SP-side shape persists on `add_*` / `sub_*` (94 rows on `step3:SP_byte1`) but is now better described as the L16 LEV / L15 nibble_copy candidate set surfaced by the per-row attribution tool, not by the soft-ADDR-B0 hypothesis.
- **Cross-ref**: bug #27 (probable parent), #26 (dead categories partially explained), **#33** (post-LEV AX corruption — primary attribution as of 2026-06-01).

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
- **Status**: D (documented as a metric caveat). Re-measured on 2026-06-01 sweep: 864 / 1096 divergent at the final-OUTPUT layer, 232 ok, 0 error.

---

## Bugs 33-38 (round-4 parallel-debug wave, 2026-06-01)

Sources for this batch: `CAMPAIGN_SUMMARY.md` §10 +
`.agent-logs/{wide_alu,stack_jsr_lev,conv_io_prtf,attribution,no_claims_survey}_2026_06_01.md`.
Commits during the wave: `887164b..e18885a` on
`speedup-cache-and-buckets` (see §10.1 for the full list).

### Bug #33: Post-LEV AX / STACK0 byte0 corruption on function-return rows
- **Source**: `.agent-logs/stack_jsr_lev_triage_2026_06_01.md` (375 row triage); `.agent-logs/attribution_2026_06_01.md` (per-row first-fatal attribution, 237/237 covered on shard 0 + 548).
- **Symptom**: 185 rows in the stack / JSR / LEV cohort (not subsumed by #27 / #28 / #29 / #30 / #31) first-fatal at `step6:AX_byte0` (91 rows) or `step6:STACK0_byte0` (50 rows) immediately after the LEV-return boundary. The co-occurring pattern (`func_identity_*`, `func_square_*`, `rec_factorial_*`, `rec_fib_*`, plus a sub-cluster on `step3:STACK0_byte0` n=15 = pre-JSR arg-push) indicates a single missing LEV-return-recovery routing rule, not 14 independent edges.
- **Affected**: 185 rows directly (cluster); 25 / 25 `absdiff_*`, 25 / 25 `nested_quad_*`, and ~97 `MUL_*::neural_None` rows from the wide-ALU triage are downstream of the same L10 PSH addr0_e0 OP_ENT-guard miss documented in memory note `project_l10_psh_addr_ent_bug.md` (`l10_ops.py:3888-3927`).
- **Root cause**: per stack_jsr_lev triage §Recommendation — the LEV op must copy `STACK0_byte0 → AX_byte0` *and* preserve `STACK0_byte0` on the dropped frame; it currently overwrites `AX_byte0` with the wrong source. Cluster suspects: `layer16_lev_routing` (`l16_lev_set_output_lo0_byte0`, `l16_lev_clear_output_lo10_byte0`, `l16_lev_*_ax_*`), `layer6_ax_load` (callee return-value materialization fallback), `tail_bit32_result_correction` (post-LEV AX byte0 patch), and the L10 PSH addr0_e0 OP_ENT guard for the entry-side blocker.
- **Status**: U (cluster identified, fix not landed). Per `feedback_single_rule_fixes_are_zero_sum.md`, must be attacked as one coordinated edit on the L16 + L6 family rather than 14 separate rule patches.
- **Expected impact**: ~141 rows directly on the top two slots; ~140-165 net after secondary `step3:STACK0_byte0` (15) and `step0:AX_byte0` (4) churn settles.
- **Cross-ref**: bug #26 (absdiff / nested_quad now attributed here), bug #29 (function-call family re-attributed here), wide-ALU triage `MUL_*::neural_None` (97 rows; same OP_ENT-guard root).
- **File**: `c4_release/neural_vm/unified_compiler/ops/l16_ops.py` (LEV routing); `c4_release/neural_vm/unified_compiler/ops/l6_ops.py` (AX load); `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:3888-3927` (PSH addr0_e0 OP_ENT guard).

### Bug #34: Wide-MUL single-byte regression to 0xD8 = 216 sentinel
- **Source**: `.agent-logs/wide_alu_triage_2026_06_01.md` §3 (failure-shape grouping), §7.4 (uniform-sentinel diagnosis).
- **Symptom**: every `loop_pow2_*` row (`SHL_via_repeated_mul2`, 25 ids, ids 0525-0549) yields `neural=216` regardless of exponent (2^0, 2^1, 2^2, ..., 2^16 all → 216). Same sentinel value appears in `MUL_via_loop_add::single_byte_mul_wrong` (25 ids, ids 0500-0524) — every `loop_mul_*` row yields `neural=216` regardless of operands. Total uniform-sentinel footprint: ~46 rows where `a*b ≤ 255` should succeed but produces a fixed 216 = 0xD8.
- **Affected**: ~46 ids across `MUL_via_loop_add` + `SHL_via_repeated_mul2` + sub-fractions of `MUL_direct::single_byte_mul_wrong` (2 rows) and `MUL_via_var::single_byte_mul_wrong` (4 rows).
- **Root cause**: the loop body (`result = result * 2` or `result = result + a`) writes a constant 216 regardless of the operand. Triage §7.4 hypothesis: stale-`result` read in L7 GPR fetch interacting with the loop-body MUL/ADD — `result` is read fresh-on-write but the writer ignores the fresh value and emits the sentinel. This is **distinct** from bug #20 (which was a declaration alignment regression) — single-byte products that should never need the wide pipeline are still failing.
- **Status**: U. Single-rule-fix candidate via the verifier (op-local; one writer, sentinel pattern is the signature of one mis-fired rule).
- **Expected impact**: ~46 ids if the sentinel-writer is patched cleanly. Folded into the ~143-row "surgical via verifier" estimate in `CAMPAIGN_SUMMARY.md` §10.6.
- **Cross-ref**: bug #20 (sibling), bug #6 (sibling).
- **File**: `c4_release/neural_vm/unified_compiler/ops/l11_ops.py::make_layer11_mul_partial_op` and/or `c4_release/neural_vm/unified_compiler/ops/l12_ops.py::make_layer12_mul_combine_op` (sentinel-emission locus pending verifier attribution).

### Bug #35: Wide-MUL high-byte loss (variable-fed operands)
- **Source**: `.agent-logs/wide_alu_triage_2026_06_01.md` §3, §7.2.
- **Symptom**: variable-fed MUL drops the entire high byte and writes 0. Sub-cluster counts: `MUL_via_var::high_byte_lost` (19 / 25, all `neural=0`), `MUL_recursive::high_byte_lost` (11 rows, e.g. `rec_factorial_7: 6! exp=720 neural=1`), `MUL_square::high_byte_lost` (3 rows), `SHL_via_repeated_mul2::high_byte_lost` (6 rows). Distinct from #6 (wrong-but-nonzero) and #34 (uniform sentinel).
- **Affected**: ~39 ids directly; mechanism plausibly extends to other multi-byte wide-MUL rows downstream of `MARK_AX` / `AX_CARRY_LO` staleness when operands flow through L7 GPR-fetch (vs. immediates).
- **Root cause**: per triage §7.2 — `MARK_AX` / `AX_CARRY_LO` staleness invariant fails when operands flow through L7 GPR-fetch rather than from immediates. Distinct mechanism from `MUL_direct::multi_byte_mul_wrong` (bug #6 residual, which produces wrong-but-nonzero hi).
- **Status**: U. Requires attention-verifier-aided diagnosis (multi-byte MUL failure shape varies, suggesting multiple mechanisms — see triage §8).
- **Expected impact**: ~39-52 ids if attention-side root identified and patched.
- **Cross-ref**: bug #6 (sibling — distinct shape), bug #34 (sibling — sentinel sub-shape).
- **File**: `c4_release/neural_vm/unified_compiler/ops/l12_ops.py` (combine) + `make_l10_post_ops_combined` (carry propagation post-op).

### Bug #36: Long-division SLOT_REMAINDER → OUTPUT_LO/HI projection failure
- **Source**: `.agent-logs/wide_alu_triage_2026_06_01.md` §5 (compiler op-surface), §7.1 (expr_mod value-dependent pattern). Pre-existing FIXME at `alu/ops/mod.py:14-53` landed by merge `2b7b34a` (`investigation/expr-mod-divergences`).
- **Symptom**: DIV / MOD with non-power-of-2 divisor/modulus produces wrong results; value-dependent failure pattern. Sub-cluster counts: `DIV_direct::nonpow2_divisor_wrong` (32 rows), `MOD_direct::nonpow2_modulus_wrong` (26 rows), `MOD_iterative::nonpow2_modulus_wrong` (49 rows on `gcd_*`, neural=65280=0xFF00 or 0), `MOD_in_expression::nonpow2_modulus_wrong` (2-5 rows). Total: ~109 in-sweep rows. (`MOD_in_expression` only 5 / 25 swept; expected ceiling closer to 130 once the sweep is filled in.)
- **Affected**: ~109 rows directly (DIV + MOD nonpow2). `MOD_iterative::pow2_modulus_wrong` (gcd_28: gcd(102, 32) → 0; expected 2) shows that even pow2 mod fails inside loops, partially refuting the "pow2 always works" hypothesis from the FIXME.
- **Root cause**: long-division compute appears correct for trivial cases (`mod_2: 154%8 → 2` is correct); the failure surface is the multi-nibble `SLOT_REMAINDER → OUTPUT_LO/HI` projection through L11-L17. `MOD_iterative` `gcd` neural=65280 (= 0xFF00) is the classic high-byte-only artifact, suggesting MOD inside the loop fires correctly once but state corrupts across iterations.
- **Status**: D (FIXME landed at `2b7b34a`; cluster-level fix not landed). Per `feedback_single_rule_fixes_are_zero_sum.md`, single-rule fixes will be zero-sum here — this needs FFNRule IR migration of the projection chain (see `CAMPAIGN_SUMMARY.md` §10.6 effort estimate: 3-5 days).
- **Expected impact**: ~109-130 ids if the SLOT_REMAINDER → OUTPUT_LO/HI projection is migrated to an FFNRule IR pass.
- **Cross-ref**: investigation merge `2b7b34a`.
- **File**: `c4_release/neural_vm/unified_compiler/ops/alu/ops/mod.py:14-53` (FIXME); `c4_release/neural_vm/unified_compiler/ops/alu/ops/divmod_longdiv.py::FlattenedDivMod`; L11-L17 OUTPUT_LO/HI projection chain.

### Bug #37: L8 SP-gather attention strength violation on ADDR_B[012]_LO/HI
- **Source**: `attention_verifier` V1 (commit `34497f1`); V1 sweep against `verify_rule_strength` ported to attention heads.
- **Symptom**: 128 strength_violations on the L8 SP-gather head reading `ADDR_B0_LO`, `ADDR_B0_HI`, `ADDR_B1_LO`, `ADDR_B1_HI`, `ADDR_B2_LO`, `ADDR_B2_HI`. V1-ops verifier had previously seen a *declarative scope* violation but could not localize it to a head; V1-attn now does.
- **Affected**: not individually-attributable yet (V1 attention verifier emits violation counts but not per-row mapping); structurally the **attention-side root of bug #27** (the soft ADDR_B0/B1/B2 evidence misfire surfaced on SP_byte0 step 2).
- **Root cause**: L8 SP-gather head is over-strength on ADDR_B[012]_LO/HI reads, causing the soft evidence to win contests it should lose. The on-ops side reads were the **symptom** that motivated bug #27; the head-side strength bug is the underlying source.
- **Status**: U. V1 attention verifier identifies the head + read pair; V2 (head-level sign-aware strength algebra, per the S-2-followup pattern at `bddf10f`) is needed to attribute the 128 violations to specific rules.
- **Expected impact**: bound the residual SP_byte0 cluster (#27 / #29 family) once V2 lands; not yet quantified.
- **Cross-ref**: bug #27 (declarative-side symptom), bug #29 (rowset view of #27).
- **File**: `c4_release/neural_vm/unified_compiler/ops/l8_ops.py` SP gather (head identified by V1 attention verifier).

### Bug #38: L15 head 8 / 13 cross-modality bleed on OUTPUT_LO/HI
- **Source**: `attention_verifier` V1 (commit `34497f1`).
- **Symptom**: 96 strength_violations on OUTPUT_LO/HI traced to head 8 leaking into head 13's output band. Head 8 and head 13 are distinct modalities at L15 (per the layout's nibble-copy attention head assignment) — head 8 should write into a non-OUTPUT slot but its strength on OUTPUT_LO/HI is high enough to dominate head 13's intended writes.
- **Affected**: not yet individually-attributable. Plausibly contributes to `step1:STACK0_byte2` (#28, 25 rows on `var_three_*`) and parts of the `step3:SP_byte1` / `step6:STACK0_byte0` clusters from `attribution_2026_06_01.md` since both attribution slots' candidate sets include `layer15_nibble_copy`.
- **Root cause**: head-strength imbalance at L15. Distinct from #37 (which is L8 SP-gather); this is the second attention-side root surfaced by V1-attn.
- **Status**: U.
- **Expected impact**: not yet quantified; needs V2 attention verifier for per-rule attribution.
- **Cross-ref**: bug #28 (probable downstream symptom).
- **File**: `c4_release/neural_vm/unified_compiler/ops/l15_ops.py` (nibble_copy heads 8 and 13).

---

## Out-of-scope category: conv-I/O / PRTF (not numbered)

The conv-I/O / PRTF rowset has been formally re-classified as
**OUT-OF-SCOPE for the 1096 metric** per
`.agent-logs/conv_io_prtf_triage_2026_06_01.md`. Evidence:

- `tests/test_suite_1000.generate_test_programs()` (the source of all
  1096 rows) returns **0 PRTF / putchar / getchar / syscall** programs —
  pattern scan over the full generator source yields no matches for any
  of `printf, prtf, putchar, getchar, open, read, clos, malc, free, mset,
  mcmp, scan, gets, write, fopen, fclos, %d, %s`.
- The category breakdown in the generator covers `func` (150), `var` (100),
  `if` (100), `loop` (100), `rec` (100), `expr` (100), `add/sub/mul/div/mod`
  (50 each), `gcd` (50), `nested` (50), `edge` (46), `absdiff` (25),
  `bool` (25) — none of which emit I/O bytecode.
- The earlier BUG_CATALOG estimate of "~150-200 ids" for conv-I/O was
  off; that surface is exercised by `test_conversational_io_*.py` and the
  V9 PRTF plan, which are **separate test surfaces** not driven by the
  1096 sweep.

Future references to conv-I/O coverage should target those separate
surfaces; campaign work targeting "1096 pass rate" should not allocate
effort to this category.

---

## Open vs fixed summary

Updated 2026-06-01 to reflect round-4 status changes.

- **Fixed (F)**: 1, 2, 4, 5, 9, 10, 14, 15, 16, 18, 19 — 11 bugs.
- **Fixed-with-regression (F-with-regression as of 2026-06-01)**: 6, 20 — 2 bugs (new shapes surfaced; carried by #34 / #35).
- **Partial / fix-as-absent**: 8, 12 — 2 bugs.
- **Documented / diagnosed, no fix landed (D)**: 3, 7, 11, 13, 17, 21, 22, 23, 26 (also U), 32, 36 — 11 bugs.
- **Under investigation (U)**: 22 (also D), 24, 25, 26 (also D), 28, 30, 31, 33, 34, 35, 37, 38 — 12 bugs.
- **Regression / partially attacked (R)**: 27 — 1 bug (D2 revert landed `361357a → 65f80f1` with near-zero impact; #37 surfaces the attention-side root).
- **Superseded by re-attribution**: 29 — function-call family now attributed to #33 / #37.

Bugs 3 / 7 / 13 / 23 share an upstream-signal lineage and are functionally
addressed by the B7-1..5 + B7-6/B7-7 architectural cleanup landed in
`a03f600`; the C9-v1 summary status for them is "D" because the
*consumer-side* rewrite was still pending at `bcb0fd1`.

Bug #27 (D2 outcome) and bug #29 (re-attribution) are the principal
revisions from the round-4 wave; bugs 33-38 are the new additions.

---

## Top 5 by impact (row count where available, updated 2026-06-01)

1. **Bug #33** — Post-LEV AX corruption: 185 unaddressed rows + ~97 `MUL_*::neural_None` rows downstream of the same L10 PSH addr0_e0 OP_ENT-guard miss; expected ~141 rows directly + ~50 collateral on bugs #26 (absdiff / nested_quad). Largest single-cluster target at HEAD.
2. **Bug #36** — Long-division SLOT_REMAINDER projection: ~109 in-sweep rows (`MOD_iterative` 49 + `DIV_direct` 32 + `MOD_direct` 26 + `MOD_in_expression` 2-5); expected ceiling ~130 once the sweep is filled in.
3. **Bug #35** — Wide-MUL high-byte loss: ~39 rows (`MUL_via_var` 19 + `MUL_recursive` 11 + `MUL_square` 3 + `SHL_via_repeated_mul2` 6); attention-verifier-aided diagnosis required.
4. **Bug #34** — Wide-MUL single-byte regression to 0xD8 = 216 sentinel: ~46 rows (uniform sentinel across `loop_pow2_*` + `loop_mul_*` + sub-fractions); single-rule-fix candidate.
5. **Bug #32** — Corpus-wide lowering health: 864 / 1096 rows divergent on the 06-01 sweep (metric caveat).

(Older top-impact estimates from C9-v2 are downgraded by 2026-06-01:
bug #27 D2 outcome was near-zero impact, not the +50-75 forecast; bug
#29's function-call rowset is re-attributed to #33; bug #26's 41 ids
are partially attributed to #33's post-LEV cluster. Bugs #37 / #38 are
attention-side roots without yet-quantified row impact.)

The round-4 wave shipped no direct headline pass-rate delta (the D2
revert at `361357a → 65f80f1` was a 2-line cleanup with ~0 impact, and
no fresh full sweep has been run at `e18885a`); the `887164b`
SP-marker merge documents +12 / 137 on shard 548-684 as the round's
biggest measured win, which is plausibly the post-`794155e` pass-rate
lift.

---

## What's still untracked

Updated 2026-06-01: the round-4 wave **promoted three of the
previously-untracked categories into numbered bugs**:

- Wide-byte ALU long tail (was "~120-150 ids, no bug numbers"): now
  measured at **525 in-corpus rows** (4x prior estimate; see
  `.agent-logs/wide_alu_triage_2026_06_01.md`). The category is
  decomposed into bugs **#34** (single-byte 216 sentinel, ~46 rows),
  **#35** (high-byte loss, ~39 rows), and **#36** (long-division
  projection, ~109 rows). Bugs **#6** and **#20** carry the residuals
  of the pre-existing pipeline fixes.
- Stack / JSR / LEV protocol edges (was "~80-100 ids"): now measured at
  **375 in-corpus rows** (4x prior estimate; see
  `.agent-logs/stack_jsr_lev_triage_2026_06_01.md`). The 185
  unaddressed rows after subtracting bugs #27 / #29 / #30 / #31 / #28
  are captured by **bug #33** (post-LEV AX corruption). Bug #33 also
  subsumes the L10 PSH addr0_e0 missing `OP_ENT` guard described in
  memory note `project_l10_psh_addr_ent_bug.md` — the OP_ENT-guard fix
  is one sub-component of the #33 cluster fix.
- Conv-I/O / PRTF pure-neural path (was "~150-200 ids"): now formally
  classified as **OUT-OF-SCOPE for the 1096 metric** per
  `.agent-logs/conv_io_prtf_triage_2026_06_01.md` (0 rows in the 1096
  corpus). See the dedicated "Out-of-scope category" section above.

Still untracked:
- Long-tail single-id regressions (~100-150 ids, no enumeration).
- `project_l16_bp_frame_byte1_ff_dual.md` (L16 dual victim/aggressor
  finding for `if_var`) — separately documented via investigation
  merge `4644edd` (FIXME), but not yet captured as a numbered bug.
- `project_l3_sp_byte0_dormant.md` (L3 SP_byte0 rewrite function
  dormant — actual logic at `vm_step.py:4604+`) — substrate-level
  observation, not yet folded into the numbered catalog.

Also: the no-claims survey at `.agent-logs/no_claims_survey_2026_06_01.md`
finds 73 / 115 ops with empty `claims` and 25 `should_backfill`
candidates (e.g. `layer3_ffn`, `layer4_ffn`, `opcode_decode_ffn`,
`layer6_routing_ffn`, `layer8_alu`, `layer9_alu`, `layer10_alu`,
`layer11_mul_partial`). These are not bugs per se but are a
correctness-debt list for the verifier coverage — the L14 + model_ops
backfill chains landed in round 4 closed 10 of the 25.

---

## Sources consulted

- `c4_release/docs/CAMPAIGN_SUMMARY.md` at HEAD (now §10, round-4 update) — Section 5 supplied bugs #27-#32 verbatim; Section 10 supplied the round-4 context for bugs #33-#38 and the status updates on #6 / #20 / #26 / #27 / #29.
- `c4_release/docs/CAMPAIGN_SUMMARY.md` at `bcb0fd1` (C9-v1) — Section 3 supplied bugs #1-#26 verbatim; Sections 4-7 supplied per-bug context (root cause, fix branch, affected ids).
- Git commit messages on `speedup-cache-and-buckets`: `bcb0fd1` (C9-v1 cover note), `de26008` (C9-v2 cover note), `887164b..e18885a` (round-4 wave).
- B7-9 lowering revalidation: `git show 6e8ab77:.agent-logs/lowering-revalidation-v4/SUMMARY.md` (cited by C9-v2 for bugs #27-#32; first-fatal histograms).
- D1 post-merge retest: `git show origin/audit/post-merge-retest-a03f600:.agent-logs/post-merge-retest-a03f600/_run.log` (cited by C9-v2 for the 238 / 1096 pass count and shard breakdown).
- Round-4 diagnostic triage docs (gitignored under `.agent-logs/`):
  `wide_alu_triage_2026_06_01.md` (bugs #34, #35, #36; status updates on #6, #20),
  `stack_jsr_lev_triage_2026_06_01.md` (bug #33; cluster sizing for the function-call families),
  `conv_io_prtf_triage_2026_06_01.md` (formal out-of-scope classification; 0 in-corpus rows),
  `attribution_2026_06_01.md` (per-row first-fatal attribution on shard 0 + 548; supports #27 re-attribution and #33 candidate sets),
  `no_claims_survey_2026_06_01.md` (verifier coverage debt list).
- Round-4 sweep baseline: `.agent-logs/sweep-2026-06-01/shard_{0,137,274,411,548,685,822,959}.log` (232 ok / 864 diverge / 0 error).
- Round-4 tooling commits: `f7c4e0f` (`attribute_failures.py`), `34497f1` (`attention_verifier` V1 — surfaces #37 / #38), `7e27ad0` (per-op L1/L10 harness), `bcd9d04` (slot-99-115 allocation proposal), `e23155b` (L16 verifier honesty), `775bd79` (L10 strength/scope cleanup), `f869367` (B8-A `ADDR_B1_VALID` + `ADDR_B2_VALID` slot allocation).
- Round-4 fix merge with measurable impact: `887164b` (L10 SP-marker CMP+2 / OP_ENT blocker fix; +12 / 137 on shard 548-684).
- L14 claims backfill chain: `caaa8f6 → d43ea7d → 1aac36f → cfe5d52 → a173f32 → 298ad08 → e18885a`.
- model_ops claims backfill: `760b010 → 701c0d8 → 91ea7d3 → 6be03a2`.
- D2 revert (near-zero impact): `361357a → 65f80f1`.
- Symbolic-vs-lowered probe (`.agent-logs/symbolic-bounds-checks.log`): refutes 1e16+ delta hypothesis; lowered weights within ~300 max-abs of symbolic.
- B6-G L7-L9 audit, B6-K BD dim usage map, B6-L L17 post-op inventory, B4-H L10 refactor PLAN — referenced as cross-refs in both C9-v1 and C9-v2; not re-read for this catalog (already summarized in the parent summaries).
- User-memory notes (for the "untracked" section, not the numbered catalog): `project_l10_psh_addr_ent_bug.md` (now folded into #33 as a sub-component), `project_l16_bp_frame_byte1_ff_dual.md`, `project_l3_sp_byte0_dormant.md`, `project_1096_sentinel_baseline.md`, `feedback_single_rule_fixes_are_zero_sum.md`.

This catalog covers **all 38 numbered bugs** referenced by the three
campaign summaries; no bug numbers 1-38 are missing. The C9-v1
summary explicitly states "26-row bug catalog" in its commit message
(`bcb0fd1`), so #1-#26 is the canonical original set, #27-#32 are
the post-B7-9 appendix, and #33-#38 are the round-4 (2026-06-01)
parallel-debug-wave additions — there is no hidden lower range
(e.g., bugs 1-7 absent from any reachable doc); the C9-v1
enumeration starts at #1 and is complete through #26.
