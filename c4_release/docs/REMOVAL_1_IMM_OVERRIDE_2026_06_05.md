# Removal 1 attempt: IMM AX runner override — REVERTED

Date: 2026-06-05
Worktree: `/tmp/c4-removal-1`
Branch: `removal-1-imm-tail` (off `1101a2e8` on `speedup-cache-and-buckets`)
Brief: replace the `b5cf7099` runner-side IMM AX override with a model-side
OP_IMM blocker strengthening on L10/L16 tail rules.

## TL;DR

**No model-side patch landed; override removal reverted.** The brief's
hypothesis — that strengthening the OP_IMM blocker on
`tail_lea_local_ax_marker_byte0_e8` (l10_ops.py:6400) and
`l16_psh_mem_addr0_e0_from_sp_no_addr_src` (l16_ops.py:768) to `-1e9`
would block the spurious 0xE8 / 0xE0 emission at MARK_AX on IMM rows —
is contradicted by the raw-model probe.

L16's rule already had `OP_IMM: -1e9` from the prior EDGE_POW2 fix. L10's
rule had no OP_IMM blocker, so one was added at -1e9. With both blockers
at -1e9, the probe shows the L10 0xE8 rule **still fires** on the 0xX8
IMM pattern (probe pass 2: 226/256 OK, 30 bad, all → 0xFFE8).

Per the L5_BYTEDECODE_FIX_ATTEMPT note:

> MARK_AX attenuates to ~1e-3 via upstream broadcast

If the upstream broadcast attenuation is closer to `1e-5` (not `1e-3`)
for OP_IMM at the AX marker, then -1e9 × 1e-5 = -1e4, which is still
much greater than the positive sum (~6.2 for the LEA 0xE8 rule, ~5 for
the L16 0xE0 rule). The rule should not fire. Yet it does.

This is consistent with the L5 doc's observation that the upstream
broadcast for OP_IMM may be effectively zero at the AX marker for the
specific bytes that trigger the leak. A negative blocker cannot block
a signal that isn't present.

## Probe results

Script: `/tmp/probe_imm_removal1.py`. Sweeps all 256 IMM byte values
through the pure-neural runner with the runner override removed.

### Baseline (no model fix, override removed)

Per `L5_BYTEDECODE_FIX_ATTEMPT_2026_06_05.md`:
- 224/256 OK, 32 bad
- 0xF0..0xFF → 0xE8 (16 bytes, via L10 tail_lea rule)
- 0xE0..0xEF (except 0xE8) → 0x01 (15 bytes)
- 0x08 → 0xE8 (1 byte, EDGE_POW2 case)

### After model fix pass 1 (L10 tail_lea OP_IMM: -1e9 added)

- 225/256 OK, 31 bad
- 0xX8 for X in {0..D, F} → 0x01 (15 bytes)
- 0xE0..0xEF → 0x01 (16 bytes, including 0xE8 which is now wrong)

The L10 0xE8 rule was suppressed (0xF_ no longer maps to 0xE8). But a
DIFFERENT rule writing 0x01 now wins on all 0xX8 cases. Plus the
0xE_→0x01 pattern is preserved.

### After model fix pass 2 (also added OP_IMM: -1e9 to tail_shr_marker_byte0_01)

- 226/256 OK, 30 bad
- 0xX8 for X in {0..D} → 0xE8 (14 bytes via 16-bit sign-ext: 0xFFE8)
- 0xE0..0xE7, 0xE9..0xEF → 0xE8 (15 bytes)
- 0xF8 → 0x00 (1 byte)

So the 0x01 SHR rule was successfully blocked, but the L10 0xE8 rule
**fires anyway** despite the OP_IMM: -1e9 blocker. This is the
disconfirmation.

## Disconfirmation: OP_IMM blockers are insufficient

The L10 `tail_lea_local_ax_marker_byte0_e8` rule fires on
`IMM 0xC8; EXIT` (and 28 other IMM byte values) even with
`("OP_IMM", -1e9)` in its conditions. This means one of:

1. OP_IMM activation at MARK_AX is effectively zero (not even 1e-5),
   so the blocker contributes ~0 regardless of weight magnitude. The
   rule fires based on positive evidence alone (MARK_AX + HAS_SE +
   OP_LEA + CMP+7 + FETCH_LO+8 + FETCH_HI+15 ≥ threshold 9.0).
2. The `dominates_at` semantics force this rule to be the authoritative
   writer at MARK_AX even when its condition score is sub-threshold (a
   compiler quirk).
3. The threshold-9.0 + `(OP_LEA, 1.0)` requirement means **OP_LEA
   itself is being amplified into the AX marker on IMM rows**, not
   that OP_IMM fails to block. The 0xE8 rule fires because OP_LEA
   leaks into MARK_AX on IMM rows with FETCH_LO+8 active. Fixing
   OP_IMM blocking does nothing because OP_LEA is the load-bearing
   positive.

Hypothesis (3) is the most likely. The fix needs to either:
- Block OP_LEA from reaching MARK_AX during IMM steps (upstream change
  in L6/L7 routing).
- Add a strong negative on a positive evidence dim that is reliably
  absent during real IMM steps but present during real LEA steps
  (e.g. `BP_PRESENT` or similar).
- Strengthen the threshold to 1000+ and source the OP_LEA contribution
  at +1000, so OP_IMM-driven attenuation drops the rule below
  threshold.

None of these are one-line fixes; all require additional investigation.

## Decision

Per the brief's "ONE compile + ONE smoke + ONE probe sweep" constraint
and the "If the model fix doesn't recover the 32 bad IMM bytes, REVERT
the override removal and write findings" instruction:

- Model-side OP_IMM blocker changes on L10 rules: **REVERTED.**
- Runner override removal in `batched_pure_neural.py`: **REVERTED**
  (override restored).
- Net diff against parent (`1101a2e8`): zero code changes; only this
  findings doc added.

Smoke baseline (override active): 46/52 — preserved.

## What would actually work

Per the probe pattern (0xX8 / 0xE_ → 0xE8 with OP_IMM: -1e9 active),
the L10 `tail_lea_local_ax_marker_byte0_e8` rule needs **OP_LEA**
gating, not OP_IMM gating. The fix is one of:

1. **Activation trace**: dump rule activations on `IMM 0xC8; EXIT` to
   identify which positives (OP_LEA, MARK_AX, CMP+7, FETCH_LO+8,
   FETCH_HI+15, HAS_SE) are crossing threshold. Then identify which
   one is the misfiring upstream signal.

2. **Per-step OP_LEA suppressor**: add a rule earlier (L8/L9) that
   forces OP_LEA = 0 at MARK_AX on IMM rows. Since L7 routes OP_*
   marks to register markers, this likely needs L7 routing change.

3. **Stricter positive evidence**: add `(BP_PRESENT, 5.0)` or similar
   to the L10 LEA rule so the load-bearing positive depends on a
   signal that genuinely distinguishes LEA from IMM. The brief's
   `EDGE_POW2_OP_IMM_LEAK.md` Fix B is the prototype for this approach.

All of these are multi-compile investigations. None fit the "one
compile + one smoke + one probe" budget for Removal 1.

## Files touched (all reverted)

- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py` —
  `tail_lea_local_ax_marker_byte0_e8` OP_IMM: -1e9 added,
  `tail_shr_marker_byte0_01` OP_IMM strengthened from -100 to -1e9.
  **REVERTED.**
- `c4_release/neural_vm/batched_pure_neural.py` — `_dispatch_pure_neural`
  IMM AX override block removed. **REVERTED** (override restored).

## Files added (this doc only)

- `c4_release/docs/REMOVAL_1_IMM_OVERRIDE_2026_06_05.md` (this file).

## Off-tree artifacts (not committed)

- `/tmp/probe_imm_removal1.py` — the 256-byte IMM AX probe.

## Cross-references

- `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md` — the parent plan.
- `L5_BYTEDECODE_FIX_ATTEMPT_2026_06_05.md` — earlier disconfirmation
  of the L5 hypothesis; same pattern, same conclusion.
- `EDGE_POW2_OP_IMM_LEAK.md` — the OP_IMM: -1e9 pattern that worked
  for the L16 rule's `0x08` case but does not generalize.
- `b5cf7099` — the runner-side AX override that remains in place.

## Confidence

- **High** that the model fix per the brief's prescription (OP_IMM
  blockers on L10/L16 tail rules) does NOT recover the 32 bad IMM
  bytes (direct probe, two compile attempts).
- **High** that the L10 `tail_lea_local_ax_marker_byte0_e8` rule is
  one of the firing rules for the 0xX8 → 0xE8 pattern (probe before/
  after evidence).
- **Medium** that the load-bearing positive condition on that rule is
  OP_LEA leaking into MARK_AX, not OP_IMM failing to block (inference
  from threshold 9.0 / OP_LEA: 1.0 / observed firing despite
  OP_IMM: -1e9).
- **High** that the b5cf7099 runner override is the simplest correct
  way to ensure IMM AX byte 0 = literal byte across all 256 values.
  Removing it without an upstream model fix is not viable in one
  session.
