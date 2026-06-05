# Removal 3 deep dive — `ebb3f09a` 32-bit cascade extension

Date: 2026-06-05
Worktree: `.claude/worktrees/agent-a81c9d9483836152b`
Parent plan: `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md`
Sister: `REMOVAL_1_IMM_OVERRIDE_2026_06_05.md`

## TL;DR

Removal 3 is **not separable from Removal 2** (collapsed-step synth).
The 32-bit cascade extension at `ebb3f09a` did not introduce a new
runner-side computation — it widened the existing `_CMP_OPS` recovery
to also fire for `_NEURAL_32BIT_OPS = {ADD, SUB, OR, XOR, AND}`. There
is no 32-bit-specific cascade bug below L9 to fix: the multi-byte
emit is autoregressive (4 separate AX byte token positions), not a
single-step OUTPUT_HI[byte>0] lane.

**No model fix landed. No code change made.** Override remains in
place. Smoke baseline preserved.

## Override surface (what `ebb3f09a` did)

`c4_release/neural_vm/batched_pure_neural.py:2177-2192`. Renamed
`_CMP_OPS` to `_NON_COLLAPSED_RECOVERY_OPS`; added ADD/SUB/OR/XOR/AND.
Same shape as the CMP recovery — when `exec_op in
_NON_COLLAPSED_RECOVERY_OPS`, call `_compute_alu_legacy(exec_op,
last_pushed_value, prev_ax)` and overwrite the step's REG_AX bytes.

Commit body itself flagged: SUB/OR/XOR recovered (+3 smoke), but
**ADD and ADD_carry_cascade still fail** — result remains
`ax_after_imm` (= the second IMM byte). The override fires for ADD
mechanically but the step's REG_AX slot is never written to.

## Why ADD specifically still fails

Trace for `IMM 200, PSH, IMM 100, ADD, EXIT` (test_add_16bit):

1. Step IMM 200: `_dispatch_pure_neural` runs, IMM AX override sets
   `s.last_ax = 200`.
2. Step PSH: `s.last_pushed_value = prev_ax = 200`.
3. Step IMM 100: IMM AX override sets `s.last_ax = 100`. Model's
   emitted `REG_PC` after this step controls what happens next.
   - **Case A — model emits PC pointing to ADD (idx 3)**: dispatch
     returns; model is asked to emit the ADD step.
   - **Case B — model collapses + emits PC = EXIT (idx 4)**: collapsed-
     step branch at line 2147-2156 fires with `skipped_op = ADD ∈
     _BINARY_POP_OPS`, calls `_compute_alu_legacy(ADD, 200, 100) = 300`,
     overrides AX, halts at next_op == EXIT. **Test would pass.**
4. In Case A: the model emits an ADD step but apparently **never emits
   a REG_AX marker token in that step**. `_extract_register` returns
   None, `neural_ax` stays None, `s.last_ax` stays at 100. The
   `_NON_COLLAPSED_RECOVERY_OPS` branch fires (`exec_op = ADD`,
   last_pushed_value = 200, prev_ax = 100) and computes 300 — but
   `_override_register_in_last_step` silently no-ops when no REG_AX
   token is present in the scan-back window. `s.last_ax = 300` in
   memory but the context's last REG_AX is still 100. On EXIT, the
   bail/decode path reads context → returns 100.

So the bug is **the model never emits REG_AX during the ADD step**,
not a 32-bit cascade wiring issue. This is the same broken-inner-step
that Removal 2 targets (collapsed-step / missing register block under
STEP_END).

## Why the brief's L9 / wide_add_rules hypothesis doesn't apply

The brief suggested checking `wide_add_rules` / `wide_sub_rules` for
multi-byte carry propagation. Two reasons this isn't the bug:

1. **All `wide_*_rules` callers use `width_bytes=1`**
   (`c4_release/neural_vm/unified_compiler/ops/alu_ops.py:774, 1530,
   1540, 1553, 1563`). The multi-byte cascade is not baked through
   the declarative DSL path at all.
2. The actual multi-byte ADD/SUB ALU lives in the imperative
   `AddSub5StageBlock` (`c4_release/neural_vm/efficient_alu_addsub_
   split.py`) via `build_add_layers(NIBBLE, ...)` / `build_sub_layers
   (NIBBLE, ...)` (`c4_release/neural_vm/alu/ops/add.py`,
   `sub.py`). NIBBLE = 4 bits × 8 positions = full 32-bit with
   prefix carry-lookahead. The cascade IS handled correctly across
   all 8 nibble positions — the GE-format computation is byte-
   identical to a reference 32-bit add at the ALU layer.

The bug is downstream: **the autoregressive AX byte-emit chain** for
the ADD step. AX is emitted as 4 separate token positions (one per
byte) per step. If the model's ADD step doesn't emit those AX byte
tokens at all, the runner has nothing to read.

## What OUTPUT_LO / OUTPUT_HI actually mean

Critical clarification from `c4_release/neural_vm/dim_registry.py:606-
609`: `OUTPUT_LO` and `OUTPUT_HI` are **16 dims each** — one-hot lo /
hi nibble of **one byte**, not byte-1..byte-3 lanes. The SMOKE_FAILURES
doc's hypothesis about "OUTPUT_HI byte-1..byte-3 lanes dropped or
inverted" mistakes the residual layout: the 32-bit AX value is emitted
byte-by-byte across 4 autoregressive token positions, not all four
bytes in a single residual position. There is no "OUTPUT_HI[byte 1]"
to be missing.

The neural model decodes the 4 AX bytes serially using OUTPUT_LO/HI
at the AX_BYTE0/1/2/3 marker positions. The 32-bit cascade test
failures (#10-14 in SMOKE_FAILURES_2026_06_05.md) are about whether
the ALU's 32-bit result is correctly **propagated** to the AX_CARRY_LO/
HI bands at each of those 4 marker positions.

## Coordination check with `agent-a0f6dd6c` (test_add_16bit agent)

That worktree exists and contains `FIXES_COMPLETED.md` / `TODO.md`
dated 2026-06-05 19:10Z but no new commits past `1bdb721d` (BNZ-branch
fix unrelated to ADD). No new docs about test_add_16bit specifically.
No overlap to coordinate.

## Recommendation

**Tie Removal 3 to Removal 2.** Per the parent plan: "Removal 3
follows automatically once 2 lands." Confirmed independently here.
The collapsed-step model fix (whatever fixes ADD's missing register-
block under STEP_END) will simultaneously restore correct AX-byte
emit for all 5 `_NEURAL_32BIT_OPS` tests:

- `test_add_16bit`, `test_add_carry_cascade` — ADD non-emit
- `test_sub_16bit` — SUB borrow cascade
- `test_or_16bit`, `test_xor_16bit` — bitwise high-nibble

Do not attempt a wide_add_rules width_bytes=2/3/4 expansion: the L9
ALU isn't the failing layer; the autoregressive byte-emit chain at
L10-L16 is.

## Files touched

None. Override at `c4_release/neural_vm/batched_pure_neural.py:2177-
2192` remains in place.

## Files added

- `c4_release/docs/REMOVAL_3_DEEP_DIVE_2026_06_05.md` (this file).

## Cross-references

- `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md` — parent plan,
  Removal 2 + Removal 3 sections.
- `REMOVAL_1_IMM_OVERRIDE_2026_06_05.md` — sister deep-dive
  (same parent plan, same "documented + no code change" pattern).
- `SMOKE_FAILURES_2026_06_05.md` — cluster definition (cascade-32bit,
  tests #10-14). Note that the doc's "OUTPUT_HI byte-1..byte-3"
  framing is incorrect (see "What OUTPUT_LO / OUTPUT_HI actually
  mean" above).
- `ebb3f09a` commit body — original observation that ADD recovery
  fired for SUB/OR/XOR/AND but not ADD itself; flagged "model
  emits an unrecognized step boundary for ADD" as the open question.
- `c4_release/neural_vm/efficient_alu_addsub_split.py` —
  AddSub5StageBlock; 5-stage GE-format pipeline that is the actual
  multi-byte ADD/SUB compute, not the declarative wide_add_rules
  path.
- `c4_release/neural_vm/alu/ops/add.py` — build_add_layers; NIBBLE
  config = 8 positions × 4 bits = full 32-bit prefix carry-lookahead.

## Confidence

- **High** that the wide_add_rules / wide_sub_rules path is NOT the
  failing layer (all callers `width_bytes=1`; multi-byte ADD goes
  through AddSub5StageBlock instead).
- **High** that the OUTPUT_LO/HI dims are single-byte not multi-byte
  (dim_registry source).
- **High** that ADD's specific failure mode is "no REG_AX token in
  the ADD step's emit" (commit `ebb3f09a` body + override-fires-but-
  no-op logic in `_override_register_in_last_step`).
- **Medium-High** that the bug is the same root cause as Removal 2
  (collapsed-step / broken inner step under one STEP_END for binary
  ops). Resolving Removal 2 should resolve Removal 3.
- **Low** that any 1-compile fix can land independently for Removal 3
  this session. The override is the right pragmatic stay-in-place
  decision until Removal 2's model fix is identified.
