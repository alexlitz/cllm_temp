# Docs archive manifest — 2026-07-23

CPU-only docs housekeeping. Archived (via `git mv` into `docs/archive/`, history
preserved — reversible) a **conservative** set of clearly-stale, closed,
superseded artifacts from the OLD 1096-corpus **neural-VM** debugging campaign
(the `neural_vm/vm_step.py` era). None of the current c4_min / BLOG_SPEC / Qwen-0.5B
/ nibble / self-emulation / radix-16 system was touched.

## Selection rule (why so few, given ~250 candidates)

The bias was hard toward KEEP. A file was archived **only if ALL** of:

1. It is a dated (2026-06 or earlier) or clearly-closed per-bug / per-fix /
   root-cause / attribution / investigation / diagnosis / triage / session note
   from the superseded neural-VM campaign.
2. It has **zero** references from any `.py`, `.sh`, `.txt`, `.json`, or from
   `CLAUDE.md`. (A huge fraction of the 2026-06 campaign notes ARE still cited by
   `neural_vm/**/*.py` ops files and `tests/test_*` as design rationale — those
   were all KEPT, per the task's KEEP rule.)
3. It has **zero** references from any doc that remains in place (a "kept doc").
   The only inbound links to the archived files now come from other `docs/archive/`
   files (archive→archive links, which are fine).

**Total moved: 25.**  **Total top-level docs kept in place: 321** (346 − 25).

Convergence note: the "no kept-doc reference" rule cascaded — pulling one
campaign note back into KEEP (because a kept doc linked it) turned it into a
protector for others, so several tightly-cross-linked JSR / SMOKE-memory / SESSION
chains were kept wholesale rather than partially archived. See "KEPT (borderline)".

---

## Archived files, grouped by theme

### Neural arithmetic (ADD / LEA) root-cause & fix-attempt logs (2026-04 era)
- `ADD_INVESTIGATION.md` — 2026-04 "why does neural ADD return operand 10 not 42" investigation; long-closed, superseded by the declarative ALU.
- `ADD_ROOT_CAUSE_FOUND.md` — 2026-04-07 "ADD root cause SOLVED" note for the handler-disabled ADD bug; closed campaign artifact.
- `LEA_FIX_ATTEMPT_SUMMARY.md` — session summary of an old LEA first-step fix attempt; superseded, no code/test refs.

### CMP / opcode-decode investigations
- `CMP_POLARITY_INVESTIGATION_2026_06_03.md` — 2026-06-03 `tail_cmp_lt_false_00` polarity investigation (fix loses other tests); closed, no commit.
- `CMP_DECODE_REWIRE_FINDINGS.md` — 2026-06-05 "CMP-decode-rewire premise refuted, no model change"; findings-only dead end.
- `OPCODE_ONE_HOT_FINDINGS_2026_06_05.md` — 2026-06-05 opcode one-hot enforcement findings from an old speedup worktree; closed.
- `TOKEN6_IMM_INVESTIGATION.md` — old "Token 6 / AX byte-0 IMM predicts 0" investigation; superseded by per-op IMM decode.

### Post-ENT / framing-desync & CMP-decode-wall notes (2026-06)
- `POST_ENT_DESYNC_REAL_PRODUCER_2026_06_14.md` — 2026-06-14 post-ENT MEM-desync producer attribution; closed campaign note, no code refs.
- `POST_ENT_DESYNC_BYTE3_SHORT_STEP_2026_06_16.md` — 2026-06-16 "34-token short step, byte-3 dropped" follow-up; closed.
- `POST_ENT_SE_VALUE_SUPPRESSOR_NEGATIVE_2026_06_14.md` — 2026-06-14 DEFAULT-OFF `C4_POST_ENT_SE_SUPPRESS` experiment that "works but does not advance programs"; dead-end flag note.
- `WALL4_SE_CMP_DECODE_ROW_FRONTIER_2026_06_11.md` — 2026-06-11 "SE CMP decode row is the architectural wall" (no behavioural change landed); superseded frontier note.
- `WALL4_SE_CMP_DECODE_ROW_FRONTIER_2026_06_11_SESSION2.md` — session-2 continuation of the same wall note; superseded.

### func / var per-cluster attribution & zero-sum fix logs
- `FUNC_LEA_BUG2_BLK42_ZEROSUM_2026_06_25.md` — 2026-06-25 "func re-read LEA bug #2 machinery landed, ZERO-SUM, DEFAULT-OFF"; closed zero-sum experiment.

### Smoke-test triage / bake-collision (2026-06)
- `SMOKE_TRIAGE_POST_L5.md` — 2026-06-03 post-L5-fix smoke triage snapshot (25/51 pass); stale status.
- `SMOKE_ROOT_CAUSE_BAKE_COLLISION.md` — old "block-5 FFN bake collision" root cause; superseded by the declarative compiler.

### Conversational-I/O triage (2026-06)
- `CONVO_IO_TRIAGE_2026_06_07.md` — 2026-06-07 conversational-I/O category triage snapshot; stale status, no code refs.

### Phase-N multibyte / IMM / LEV diagnoses (2026-05 era, pure-neural test suite)
- `PHASE_1_MULTI_IMM_DIAGNOSIS.md` — 2026-05 Phase-1 multi-IMM AX-carry diagnosis against `test_pure_neural_pc.py`; closed.
- `PHASE_3_MULTIBYTE_DIAGNOSIS.md` — 2026-05 Phase-3 multibyte AX diagnosis; closed.
- `PHASE_5_LEV_INSTRUMENTATION.md` — 2026-05-11 Phase-5 JSR/ENT/LEV instrumentation root-cause findings; closed.

### Lookup-mode migration regression
- `LOOKUP_MODE_BUG_DIAGNOSIS.md` — 2026-05-10 `migrate-lookup-mode-hybrid-alu` smoke-regression diagnosis; migration-era dead end.

### Cluster-D revert finding
- `CLUSTER_D_ATTEMPT_3_FINDINGS.md` — 2026-06-03 "Cluster D attempt 3 (revert)" finding; closed dead end.

### Old session / handoff / summary snapshots
- `SESSION_2026-04-07_NEURAL_WEIGHTS_INVESTIGATION.md` — 2026-04-07 neural-weights investigation session log; superseded.
- `SESSION_PROGRESS_FINAL.md` — 2026-06-03 cumulative-measurement session snapshot at a specific HEAD; stale.
- `HANDOFF_2026_06_02_PROMPT.md` — a 2026-06-02 agent handoff *prompt* (not a status doc); one-shot, obsolete.
- `INVESTIGATION_SUMMARY.md` — 2026-04-07 "neural arithmetic weights investigation complete summary"; superseded, only archive→archive inbound links.

---

## KEPT (borderline) — considered for archive but deliberately kept

All of these are stale-looking neural-VM campaign notes I *would* have archived on
content alone, but kept because a KEPT doc, a `test_*`, or `neural_vm/**` code still
references them (the task's explicit KEEP rule). Flagging for your double-check:

- `ACTUAL_FIX_AX_CARRY.md` — referenced by `tests/test_layer6_head_allocation.py` + `tests/test_arithmetic_no_handlers.py`.
- `FIX_AX_CARRY_ISSUE.md` — referenced by kept docs `TESTING_THE_FIX.md`, `ACTUAL_FIX_AX_CARRY.md`.
- `FIX_ATTEMPT_SUMMARY.md` — referenced by kept docs `ROOT_CAUSE_UPDATED.md`, `INVESTIGATION_FINAL_SUMMARY.md`.
- `ROOT_CAUSE_UPDATED.md` — referenced by `STATUS.md` (a heavily-cited kept status doc).
- `INVESTIGATION_FINAL_SUMMARY.md` — referenced by kept `FIX_AX_CARRY_ISSUE.md`.
- `JSR_NEURAL_BUG.md` / `BUG_ROOT_CAUSE_JSR_LEV.md` — referenced by kept `JSR_FIX_SUCCESS.md` (which is cited by `model_ops.py`) and `STATUS.md`.
- `SESSION_2026-03-31_CONTINUED.md` / `_LEV_INVESTIGATION.md` / `_FINAL_STATUS.md` — mutually cross-linked with kept `JSR_FIX_SUCCESS.md`, `OPCODE_TEST_STATUS.md`, `SESSION_2026-04-07_OPCODE_VERIFICATION.md`.
- `SESSION_2026-04-07_OPCODE_VERIFICATION.md` — referenced by kept `HANDLER_DEPENDENCY_DISCOVERY.md`.
- `HANDOFF_2026_06_03.md` — referenced by `STATUS.md`.
- `SMOKE_FAILURES_2026_06_05.md` + `SMOKE_MEMORY_FIX_ATTEMPT_20260605.md` / `_V2` / `_V3` / `_V4` — chain cited by `neural_vm/unified_compiler/ops/l13_ops.py` and kept `MEMORY_PHASE2/4_BLOCKER_2026_06_05.md`.
- `REMOVAL_1_REAL_SURFACE_2026_06_06.md` — referenced by kept `DSL_ATTENTION_GATE_FINDING_2026_06_07.md`.
- `VAR_THREE_ATTRIBUTION_REPORT.md` / `VAR_CLUSTER_NIBBLE_COPY_LEAK.md` — referenced by kept `MULTICLUSTER_ATTRIBUTION_2026_06_04.md`, `L34_FFN_ATTRIBUTION_2026_06_04.md`, `STATUS.md`.
- `PHASE_4_BZ_BNZ_INSTRUMENTATION.md` / `PHASE_4_BZ_BNZ_DIAGNOSIS.md` — referenced by kept `L3_PC_CARRY_FORWARD_REBALANCE.md`.
- **~95 other 2026-06 campaign notes** (e.g. `VAR_REAL_ATTRIBUTION_*`, `DEAD_UNIT_AUDIT_*`, `SLOT_REGISTRY_AUDIT_*`, `AX_BYTE1_DUMP_*`, `STACK0_BYTE0_DUMP_*`, `WAVE_B_CLUSTER_*`, all `QWEN_*_PROTOTYPE_2026_06_07`, etc.) were kept because `neural_vm/**/*.py` ops files or `tests/test_*` cite them directly as the design source for still-present (if legacy) weights/heads. These are a much larger latent archive target but are NOT safe under the "referenced by code/test → keep" rule until the underlying `neural_vm/` code is itself removed.
