# c4_release STATUS

Auto-refreshed by `tools/refresh_status_doc.py`. The curated narrative
between the markers below is hand-maintained; the recent-commits and
in-flight scan below the markers are regenerated.

A new session should be able to read this single file and skip the 30+
other docs in `c4_release/docs/`.

<!-- BEGIN_CURATED -->

## 1. What works

- **Structural compiler**: V1 imperative cells fully migrated
  (`imperative_trivial = 0`, `imperative_heavy = 0`), V1 no_ir = 0.
- **Compile path**: `compile_full_vm_dynamic(alu_mode='efficient',
  alu_mode='lookup', n_heads=8, ffn_hidden=4096, max_seq_len=8192)`
  works after the L10 d_model fix (`24cca5ee`, sibling time bomb fix
  `bda17245`).
- **Mixtral RMSNorm**: end-to-end through padded + non-padded HF export
  paths, regression-pinned (`24967832`, `e0b01183`).
- **Byte-identity gates**: declarative bakes (`null_terminator_detection`,
  `_l10_comparison_combine_rules`, etc.) are byte-identical to the
  imperative versions they replaced.
- **Cache hygiene**: LRU evictor for `compiled_vm/` disk cache landed
  (`cb52e54e`); safetensors warm-disk hit <0.5s (`f43463a2`).
- **V2 dynamic heads / GQA**: done.
- **V3 dim_ref corpus migration**: mostly done; CARRY+0 collapsed in
  L9 (`0d0012a3`).
- **Phase 7 / 8 / 9**: closed per phase-status table in
  `HANDOFF_2026_06_03.md`.

## 2. What's broken

- **Smoke pass rate**: 12 pass / 39 fail. 6 of the failures
  (`TestSmokeComparison::test_*_true`) trace to a missing OP_<NAME>
  decode in the L5 opcode-decode FFN — `OP_EQ`, `OP_NE`, `OP_LT`, etc.
  are zero at every block position. The default-leak / MARK_PC frame
  was wrong; see `SMOKE_COMPARISON_OP_DECODE_MISSING.md`.
- **1096 lookup-mode baseline**: ~11% pass (real declarations-only
  baseline). 5 distinct failure clusters identified.
- **L17.post_ops[0] block=27**: 58 of 384 1096 failures cluster here;
  margin uniformly -16.00, OUT_HI[0] = +9.4e6 runaway.
- **edge_pow2 sign-extension**: `2^3 = -24` — 3-bit value read as 8-bit
  signed.
- **if_eq branch collapse**: always takes one branch.
- **absdiff halt-horizon miss**: runs past the expected halt step.
- **L5 FFN slot collision**: `block.5.ffn.W_up` has 129 suppressor
  units reading OPCODE_BYTE_LO+1 / OPCODE_BYTE_HI+1 with magnitude
  ~1e3, but no writer to `OP_*` dims (187..217) exists outside the
  embedding heads.
- **Efficient-mode fix gap**: declarative IR fixes to L8/L10/L11/L13
  are inert for smoke because efficient mode replaces those FFNs
  wholesale.

## 3. In flight (10-step incremental plan)

Per `IR_INCREMENTAL_IMPROVEMENTS.md`:

| Step | Description | Status |
|--:|---|---|
| 1 | Unblock smoke (find OP_* zeroer between L7-L14) | in flight |
| 2 | Strengthen one rule at a time (dominates_at) | not started |
| 3 | Bare-literal lint | not started |
| 4 | SSA value distinction (step=0 vs step=-1) | scaffolding (`ef6ef561`) |
| 5 | Drop produces/consumes_fresh as separate fields | in flight (waves 1-7 landed) |
| 6 | Auto-attribute 1096 failures | not started |
| 7 | Strict-mode compile gate | not started |
| 8 | Smoke pass-count CI metric | not started |
| 9 | Worktree + cache hygiene | partial (LRU landed; worktree cleanup pending) |
| 10 | STATUS.md + auto-refresh | **this commit** |

## 4. Priority next (top 3)

1. **Find the OP_* zeroer between L7 and L14** (Step 1). Smoke 12/39 and
   the 89% 1096 mass should move together once this lands. The
   blocker-1 cluster agent is targeting the cross-step rename in
   `l5_ops.py:527` (`OPCODE_BYTE_LO.*.-1`).
2. **Run the cross-step SSA rename audit corpus-wide** as a parallel
   root-cause sweep — Tier A revert ~30 records already documented
   (`979094ed`).
3. **Wire NormSpec / PositionalEncodingSpec / activation-spec
   dataclasses into `compile_full_vm_dynamic`** once runtime baseline
   stabilizes. Scaffolding landed (`9068b6b4`, `a563f925`); the four
   imperative kwargs can then retire.

<!-- END_CURATED -->

<!-- BEGIN_AUTOGEN -->

## 5. Recently landed (auto-generated)

Last 20 commits as of 2026-06-03 06:34Z:

- `ca8a5558` docs: smoke comparison root cause is missing OP_<NAME> decode, not cmp combine
- `203ec828` docs: IR incremental improvements plan (10 small steps, 10-13 sessions)
- `f38c9d81` produces/consumes_fresh: wave 7 — user_input + audited_empty marker
- `f43463a2` cache: switch model weights to safetensors (warm disk hit <0.5s)
- `9871d67f` docs: smoke memory trace — 5 failures share AX-write break, not memory bug
- `09bbe144` produces/consumes_fresh: wave 4 — l0-l5 token pipeline ops
- `d325d63f` produces/consumes_fresh: wave 5 — alu_ops
- `a2a1a308` produces/consumes_fresh: wave 2 — 5 more IR-bearing ops across 5 modules
- `1d3902f3` docs: var_* cluster root cause — L15 nibble_copy 0x0a leak into STACK0
- `e4b7b888` produces/consumes_fresh: wave 3 — control_flow + flag_gated ops
- `2ed33168` docs: efficient mode replaces L8/L10/L11/L13 FFNs — fix gap
- `30d0f1f8` produces/consumes_fresh: wave 1 — 4 IR-bearing l14 ops
- `153a25e0` fix(l11_l12): single-fire FlattenedALUMul to prevent block 25/27 doubling
- `5eb12dc3` fix: edge_pow2 + if_eq cluster root-cause patches
- `80e99093` docs: CMP write/read chain audit (Tier B BZ-override follow-up)
- `cb52e54e` cache: LRU evictor for compiled_vm/ disk cache
- `c7e020f7` docs: if_eq root cause — _cmp_default OP_EQ unconditional OUTPUT_LO+0 leak
- `9baf9783` docs: edge_pow2 root cause — L16 PSH-MEM rule OP_IMM gate leak
- `bda17245` fix(l10): same d_model derivation at line 6622 sibling time bomb
- `af92e434` docs: absdiff cluster root cause — BZ taken-path PC redirect bug

## 6. Active investigation docs (auto-scanned)

- `c4_release/docs/ADD_ROOT_CAUSE_FOUND.md` — ADD Operation Root Cause Analysis - SOLVED
- `c4_release/docs/BUG_ROOT_CAUSE_JSR_LEV.md` — Root Cause Analysis: JSR/LEV Infinite Loop
- `c4_release/docs/CROSS_STEP_SSA_ANTIPATTERN_AUDIT.md` — Cross-step SSA Rename Anti-pattern Audit
- `c4_release/docs/EFFICIENT_MODE_FIX_GAP.md` — Smoke fixes don't land: efficient mode replaces ALU FFNs
- `c4_release/docs/HANDOFF_2026_06_02.md` — Handoff Doc — 2026-06-02 Session
- `c4_release/docs/HANDOFF_2026_06_03.md` — Handoff Doc — 2026-06-03 Session
- `c4_release/docs/IR_INCREMENTAL_IMPROVEMENTS.md` — IR incremental improvements
- `c4_release/docs/ROOT_CAUSE_UPDATED.md` — ADD Root Cause - Updated Investigation
- `c4_release/docs/SMOKE_COMPARISON_OP_DECODE_MISSING.md` — Smoke comparison failures — root cause is upstream of ComparisonCombine
- `c4_release/docs/SMOKE_MEMORY_TRACE_20260603.md` — Smoke `TestSmokeMemory` trace — 2026-06-03 worktree session
- `c4_release/docs/VAR_CLUSTER_NIBBLE_COPY_LEAK.md` — var_three / var_update / var_mul cluster: L15 nibble_copy 0x0a leak

---

_Auto-refreshed by `tools/refresh_status_doc.py` at 2026-06-03 06:34Z. Edit curated sections between BEGIN_CURATED / END_CURATED; recent commits + in-flight scan are regenerated._

<!-- END_AUTOGEN -->
