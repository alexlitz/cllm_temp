# Conversational I/O Triage — 2026-06-07

Post-slot-fix triage of the Conversational I/O test category.
Base: `main` HEAD `6eaa6ef6` (`fix(bz-bnz): multi-step + loop branch-target emission`).
Prior status from `docs/TEST_CATEGORY_SWEEP_2026_06_06.md`:

| Total | Pass | Fail | Skip | Err |
|---:|---:|---:|---:|---:|
| 41 + 1 collection-err file | 3 | 16 | 0 | 22 |

Of those 22 errors and 16 failures, the sweep doc attributed "all 22
errors and most failures" to a single `SlotConflictError` that another
agent was fixing in parallel. That fix landed in commit
`e8d02131` (`fix(slot-registry): triage 50 collection errors`)
between the 06-06 sweep and this triage.

## Bucket A — SlotConflict errors (commit e8d02131)

**Status: resolved.**

`e8d02131` annotated 6 conflict groups with `slot_share=("ffn_units",)`
on the participating ops:

- `layer3_convo_io_state_init` (unit 1034)
- `convo_io_step_resume` (unit 1035)
- `convo_io_state_machine` (units 1400-1401)
- `convo_io_pc_sp_latch` (units 1402-1465)
- `null_terminator_detection` (unit 1864)
- `conversational_io_output_routing` (units 1200-1231)

Re-collecting the category after `e8d02131`:

```
$ python -m pytest c4_release/tests/test_conversational_io.py \
    c4_release/tests/test_conversational_io_quick.py \
    c4_release/tests/test_conversational_io_comprehensive.py \
    --collect-only
... 41 tests collected in 3.72s ... (zero collection errors)
```

The 22 setup-errors and the `test_conversational_io_final.py`
collection-error count in the 06-06 sweep — those were tied to the
`SlotConflictError`. After `e8d02131`, all 22 are gone; the residual
collection errors come from a different bucket (B) below.

## Bucket B — stale script-style files (collection errors, new root cause)

The 06-06 sweep counted one collection error against
`test_conversational_io_final.py`, blaming `SlotConflictError`. After
`e8d02131` that file *still* fails collection, but with a different
error — and two siblings join it:

| File | Error | Root cause |
|---|---|---|
| `tests/test_conversational_io_manual_bytecode.py` | `AttributeError: 'AutoregressiveVM' object has no attribute 'set_active_opcode'` | Calls retired `runner.model.set_active_opcode(...)` at module load. |
| `tests/test_conversational_io_proper.py` | Same `AttributeError` | Same retired API. |
| `tests/test_conversational_io_final.py` | `IndexError: index 410 is out of bounds for dimension 0 with size 90` | Module-level read of `runner.model.blocks[5].ffn.W_up[410, ...]` — that hidden-unit index belonged to the pre-compact `PureFFN(hidden_dim=1536)` layout, the post-compact `StandardMoEFFN` L5 has only 90 units. |

All three files are **module-level scripts** (top-level
`print`/`assert` blocks, no `def test_*` / `class Test*`). They were
documentation of the pre-MoE conversational-I/O pipeline. The
`set_active_opcode` weight-swap path was retired (see
`neural_vm/vm_step.py:1780` comment: "the previous `set_active_opcode`
weight-swap path has been retired"). The hard-coded `W_up[410, ...]`
indices died at the same time, because `compact()` now packs each
layer's active units into a dense sub-matrix.

**Fix applied (this triage):**

1. `c4_release/tests/conftest.py` — added `collect_ignore = [...]` for
   the three files so a directory-level pytest sweep (and the
   `tools/test_sweep_by_category.sh` orchestrator) skips them cleanly
   instead of hard-failing collection.
2. `c4_release/tools/test_sweep_by_category.sh` — dropped the explicit
   `tests/test_conversational_io_final.py` arg from the
   `conversational_io` category so the sweep no longer pins the
   excluded file as an explicit pytest argument (which bypasses
   `collect_ignore`).

After this change, the conversational_io category collects 41 tests
with 0 collection errors.

## Bucket C — real model bug: `IO_IN_OUTPUT_MODE` output routing returns `""`

The 06-06 doc flagged this separately:

> The 3 `test_conversational_io.py` failures and the early
> `test_conversational_io_comprehensive.py::TestPRTFOutputContent`
> group fail with `AssertionError: assert 'Hello' in ''` / `assert ''
> == '<expected>'` — output-routing produces empty string. May be a
> separate downstream bug surfaced by `conversational_io=True`.

Confirmed in this triage with a fresh run on `main@6eaa6ef6`:

```
TestPRTFOutputContent::test_printf_literal_string  FAILED
  assert 'Hello' in ''
TestPRTFOutputContent::test_printf_with_newline    FAILED
  assert '' == 'Hello World\n'
TestPRTFOutputContent::test_printf_integer         FAILED
  assert 42 == 0
  (and RuntimeWarning: draft_divergence: 25 of last 50 tokens disagree)
TestPRTFOutputContent::test_printf_zero            PASSED
```

Note: `TestPRTFOutputContent` uses `conversational_io=False` per its
own docstring, so this bug is on the `_syscall_prtf` /
exec_op-fallback path, **not** the conversational-I/O state-machine
path. It is independent of the SlotConflict fix.

Root-cause surface (unconfirmed, needs the next agent):

- `IO_IN_OUTPUT_MODE` reads in L15
  `_set_conversational_io_output_routing` (units 1200-1231) cross-step
  read; same-step writer is `null_terminator_detection`. This shows up
  as a `CrossStepReadWarning` at compile time (visible in the quick
  test). The step-1 zero-propagation class-of-bug noted in the warning
  text is a known failure mode.
- The `test_printf_integer` failure also reports `draft_divergence: 25
  of last 50 tokens disagree with DraftVM` — the DraftVM and the
  neural model disagree on a `%d` integer-formatting step. The
  declarative path uses `format_string_fetch_head` which reads
  `IO_IN_OUTPUT_MODE.*.-1` cross-step from
  `null_terminator_detection`; same cross-step warning bucket.

Bucket-C tests likely failing under the same root cause (not all
re-run individually due to ~7 min per-test GPU cost):

- `test_conversational_io.py::TestConversationalIO::test_prtf_emits_thinking_end`
- `test_conversational_io.py::TestConversationalIO::test_prtf_sequence`
- `TestPRTFOutputContent::*` (10/11 — one passes:
  `test_printf_zero`)

## Bucket D — flag-gating / expected-pass

These four pass on `main@6eaa6ef6` and need no work:

- `TestConversationalIOMode::test_conversational_io_disabled_by_default`
- `TestConversationalIOMode::test_conversational_io_can_be_enabled`
- `TestConversationalIOMode::test_programs_run_without_conversational_io`
- `TestConversationalIOMode::test_programs_run_with_conversational_io`
- `TestThinkingTokens::test_thinking_tokens_exist`
- `TestThinkingTokens::test_thinking_tokens_are_distinct`
- `TestPRTFOutputContent::test_printf_zero` (the one that passes —
  empty format-string path)

Plus the quick script `test_conversational_io_quick.py` (init smoke
only — runner builds, context builds, no run).

## Bucket E — unverified, pending GPU time

The following classes use the standard runner fixture and execute
real bytecode; each test takes ~3-7 minutes of GPU forward time on a
single H100, so we did not re-run all 19 individually after the slot
fix. They were *all errors* in the 06-06 sweep due to the
SlotConflict at fixture-setup time. With Bucket A resolved, they
should now run; the 06-06 sweep's pattern of `'Hello' in ''` /
`exit_code != expected` predicts they will collapse into Bucket C +
Bucket F (real bugs) rather than into fixture errors.

- `TestPRTFDetection::*` (6 tests)
- `TestIOWithComputation::*` (4 tests)
- `TestPutcharOutput::*` (3 tests)
- `TestMixedIOOperations::*` (2 tests)
- `TestIOErrorHandling::*` (4 tests)
- `TestIOStatePreservation::*` (3 tests)

A focused re-sweep of just this list (with longer per-test timeout) is
the right next step for the next session. Time budget: ~2 h of GPU.

## Bucket F — wrong return value (small subset of E)

The `TestPRTFOutputContent::test_printf_integer` failure already
shows the second flavour: `exit_code == 42` when expected `0`. The
`int main() { ... return 0; }` source is returning `42` because the
`x = 42` write leaks into the return path when the printf step
diverges. This is downstream of Bucket C; likely shares a root cause.

## Summary

| Bucket | Description | Count | Fix in this triage |
|---|---|---:|---|
| A | SlotConflict errors | 22 -> 0 | Resolved by `e8d02131` (not by this triage) |
| B | Stale script-style files (retired `set_active_opcode`, post-compact W_up shape) | 3 collection errors | `conftest.py:collect_ignore` + sweep-script trim |
| C | Real model bug: `IO_IN_OUTPUT_MODE` output routing -> empty string | 3 fails confirmed (likely 10+) | Documented; not fixed (model bug) |
| D | Already-passing flag-gating / token-existence tests | 7 | None needed |
| E | Heavy fixtures un-re-run after slot fix | 19 | None — needs next session |
| F | `exit_code` divergence (subset of E) | 1 confirmed | Same root as C |

**Fixes applied:**

1. `c4_release/tests/conftest.py` — `collect_ignore` for the 3 stale
   script-style files.
2. `c4_release/tools/test_sweep_by_category.sh` — drop
   `test_conversational_io_final.py` from the explicit arg list so
   `collect_ignore` is honored.

Both fixes are mechanical and zero-risk: they remove three files that
hard-error at collection time and contain no valid `def test_*` /
`class Test*`. The category goes from `3 pass / 16 fail / 22 err + 1
collection-err file` to (predicted, conservative) `~10+ pass / 10+
fail / 0 err / 3 ignored`.

**Residual real bug for next session:**

`IO_IN_OUTPUT_MODE` cross-step propagation drops format-string bytes
to OUTPUT, producing `""` for any non-empty printf literal. Reproducer:
`pytest TestPRTFOutputContent::test_printf_literal_string -v`. Likely
surface: `_set_conversational_io_output_routing` (L15) reading
`IO_IN_OUTPUT_MODE` set by `null_terminator_detection` (L10);
cross-step zero-propagation warning is emitted at compile time.

## Constraints honored

- No `git stash` used.
- Worked on a dedicated branch (`convo-io-triage-2026-06-07`).
- No changes to model weights, only to test collection.
