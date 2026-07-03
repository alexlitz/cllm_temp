# Removal 1 retry — IMM AX override stays, L34 residual is clean but failure persists

Date: 2026-06-06
Base: `main` HEAD `f4f9103d`
Branch: `removal-1-imm-c8`
Prior brief: `REMOVAL_3_FINDINGS_2026_06_05.md`
Prior fix attempts: `0b957e03`, `ff4edb61`, `983b70c9`

## TL;DR

The brief assumed that after the L7 head 5 K-side `OP_IMM` blocker
(`ff4edb61`) closed the upstream broadcast leak and the L10
`tail_lea_local_ax_marker_byte0_e8` threshold was refined to 7
(`983b70c9`), removing the `b5cf7099` IMM AX runner override would
let `test_xor_basic`, `test_add_16bit`, and `test_add_carry_cascade`
all pass.

Empirically that is false. **The L34 residual probe confirms the rule
is no longer firing on IMM rows (sum=1.0 at every MARK_AX position,
threshold 7), yet removing the override regresses all three tests
back to failure.** The byte-0 corruption on IMM bytes in [0xE0, 0xFF]
is therefore NOT solely driven by `tail_lea_local_ax_marker_byte0_e8`;
a separate model-side surface remains.

Per the brief's restore-and-document fallback, the override is left
in place. Findings recorded for the next investigation cycle.

## L34 residual measurement (override removed, all model-side fixes in place)

Tool: `c4_release/tools/removal_1_probe_and_smoke.py` (extends
`l10_tail_lea_residual_probe.py` with IMM 0xC8 and IMM 0xFF+ADD
programs, plus the 3 critical smoke tests in the same build).

Run: `CUDA_VISIBLE_DEVICES=1 python tools/removal_1_probe_and_smoke.py`.

Rule predicate sum at MARK_AX positions (threshold 7.0):

| Program     | MARK_AX positions | Max sum | All > threshold? |
|-------------|-------------------|---------|------------------|
| LEA_BASIC   | 1                 | 1.000   | no               |
| XOR_BASIC   | 13                | 1.422   | no               |
| ADD_16BIT   | 13                | 1.422   | no               |
| ADD_CARRY   | 5                 | 3.258   | no               |

Per-position contributions:
- ~All positions: `MARK_AX=1.0` only.
- A few positions in XOR_BASIC / ADD_16BIT: `MARK_AX=1.0` + `HAS_SE≈0.42`
  (sum ≈ 1.42).
- ADD_CARRY fwd=116: `MARK_AX=1.0` + `HAS_SE≈2.26` (sum ≈ 3.26).
- `OP_LEA`, `CMP+7`, `FETCH_LO+8`, `FETCH_HI+15`, `MEM_ADDR_SRC` read
  ~0 at every MARK_AX position for every program.

**Conclusion**: the L10 `tail_lea_local_ax_marker_byte0_e8` rule
cannot fire on these programs at the current threshold. The L7 head 5
K-side fix (`ff4edb61`) did close the upstream leak that the brief
predicted.

## Smoke result with override removed

The three critical tests still fail, but with EXIT codes that do NOT
match the historical 0xFFE8 corruption signature:

| Test                       | Expected | Got | Status |
|----------------------------|----------|-----|--------|
| test_xor_basic             | 0x2A     | 0   | FAIL   |
| test_add_16bit             | 300      | 0   | FAIL   |
| test_add_carry_cascade     | 0x100    | 1   | FAIL   |

Compare with the historical Removal 3 trace (override removed, pre-
L7 head 5 fix):
- Step 0 emitted `neural_ax = 0xFFE8` (= 65512) → `last_pushed_value`
  becomes 0xFFE8.
- Collapsed-step recovery computed `0xFFE8 + 100 = 16770124`.

Now the EXIT codes are 0, 0, 1 — not 16770124 or similar. So the
failure mode has shifted. The model is either:
1. Hitting divergence-bail earlier and returning a stale or zero AX
   from `_decode_bail_exit_code`, or
2. Emitting `neural_ax = 0` for the affected IMM steps (different
   corruption surface), then non-collapsed recovery doing `0 ^ 0xD5`
   = 0xD5 but somehow being lost.

The exit code = 1 for `test_add_carry_cascade` is particularly
diagnostic: 1 is the architectural answer to `0xFF + 1` for the LOW
byte (with the high byte carried separately) — suggesting the model
emits the byte-0 correctly but byte 1+ are dropped to 0 (so the
non-collapsed ADD recovery's `last_pushed_value` is 0xFF + carry
chain doesn't propagate).

## What we know

1. **The L34 residual probe is clean** for all 4 probe programs at the
   MARK_AX positions. The L10 `tail_lea` rule is NOT the source of
   the remaining AX corruption.
2. **A different model-side surface** is producing wrong AX bytes for
   IMM in [0xE0, 0xFF]. The brief's premise that L10 was the sole
   driver was incomplete.
3. **Override stays load-bearing**: removing it regresses all 3
   critical tests. Per task acceptance criteria, override remains in
   place.

## Where to look next

Possible alternate corruption surfaces (not exhaustive):

1. **L5 byte-decode itself**: maybe the L5 IMM-byte0 decode legitimately
   emits wrong values for high IMM bytes, and the L10 rule wasn't the
   actual source — just a downstream propagator that got patched. The
   probe captures the input residual to L34 (= post-L33 output). If
   the corruption is at L5, it would show up as wrong byte values in
   the OUTPUT_LO band BEFORE any tail layer touches AX. Probe needs
   extension to also read the OUTPUT_LO+0 dim at the AX marker
   position.

2. **Other 0xE8 writers** at L10 (`l10_ops.py:3503, 5012, 5040, 6298`)
   — these target SP/STACK0/MEM scopes, not AX, but with the L7 fix
   reshaping attention they may be aliasing onto AX positions now.

3. **The collapsed-step / non-collapsed recovery itself**: with the
   override removed, the recovery paths receive `prev_ax = s.last_ax`
   which is the previous step's `neural_ax`. If neural_ax for IMM
   200 is wrong, `last_pushed_value` is wrong, and the recovery
   computes wrong result. This is the original failure mode but
   should now produce a recognizable value (e.g. 0xFFE8 + 100 = some
   non-zero), not 0. Exit=0 suggests divergence-bail is firing
   earlier than the recovery.

4. **Divergence bail**: with override OFF, the `draft_divergence`
   signal (DraftVM comparison) may trigger earlier because the
   neural_ax mismatch propagates into PC/SP staleness. Bail returns
   either the in-progress step's AX or a stale value.

Recommended next probe:
- Wrap the dispatch loop to print `(exec_idx, exec_op, neural_ax,
  prev_ax, last_pushed_value)` per step.
- Also read `OUTPUT_LO+0` at the AX marker position to see if L5
  byte-decode is emitting wrong values (would confirm Surface 1).

## Files referenced

- `c4_release/neural_vm/batched_pure_neural.py:1985-2026` — IMM AX
  override (b5cf7099), retained with retry note.
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6587-6629` —
  `tail_lea_local_ax_marker_byte0_e8` rule, threshold=7.
- `c4_release/neural_vm/unified_compiler/ops/l7_ops.py:543` — L7
  head 5 K-side `OP_IMM` blocker.
- `c4_release/tools/removal_1_probe_and_smoke.py` — combined probe +
  3-test smoke for this retry.
- `c4_release/tools/l10_tail_lea_residual_probe.py` — extended to
  include IMM 0xC8 (ADD_16BIT) and IMM 0xFF+1 (ADD_CARRY) programs.

## Acceptance

- ONE compile used. Smoke (3 critical tests) executed in the same
  build via `run_batch`.
- Override removal: 0/3 critical tests pass.
- Override restore: matches historical baseline (3/3 pass via
  recovery overrides).
- Override stays in place; doc records the structural finding for
  the next investigation.
