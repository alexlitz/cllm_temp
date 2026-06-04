# STACK0 byte 1 Q-row blocker attempt — no-op on smoke, REVERTED

Date: 2026-06-04
Branch: stack0-l10-fix (off `cef9d046` on speedup-cache-and-buckets)
Author: agent (Claude Opus 4.7)
Brief: targeted fix from `STACK0_STALENESS_FOLLOWUP_2026_06_03.md`
       hypothesis that the L10 PSH STACK0 passthrough head 3 differential
       routing (slots 32..63, added to fix bug #33 LEA-local AX byte 0)
       leaks AX byte-1/2/3 OUTPUT-vs-CLEAN_EMBED nibble deltas into
       STACK0 byte 1's OUTPUT band, biasing the head_bake NEXT_PC vs
       byte-value competition.

## TL;DR

Added `AP(33, BD.BYTE_INDEX_1, -10000.0)` to the Q slot 33 "row
selector" in `_layer10_psh_stack0_passthrough_head_spec`. This makes
the head 3 not fire at STACK0 byte 1 Q rows (Q[33]·K[33] drops from
0 to -50000 at byte 1, swamping any other contribution).

Smoke pre-fix: 28 / 51 pass.
Smoke post-fix: **28 / 51 pass** — identical failure set, byte-1
exclusion has zero observable effect on CMP failures or any other
smoke case.

Reverted per brief's "If smoke unchanged or regresses: REVERT" rule.
No commit beyond this findings doc.

## What I tried

```python
# in _layer10_psh_stack0_passthrough_head_spec, after the existing slot 33 q-side rows:
AP(33, BD.BYTE_INDEX_1, -10000.0),
```

This adds a -10000 score whenever the Q row has `BYTE_INDEX_1` hot
(i.e., the STACK0 byte 1 token row). Q[33]·K[33] becomes ~-50000 at
byte 1 Q rows, disabling the head's attention there entirely. The
slots 0..31 CLEAN_EMBED routing AND the slots 32..63 LEA-local
differential routing both go silent at byte 1 since both depend on
attention firing.

## Failure analysis

Both pre-fix and post-fix smoke runs:

| Suite                | Pre  | Post | Δ |
|----------------------|------|------|---|
| TestSmokeComparison  | 1/6  | 1/6  | 0 |
| TestSmokeBasic       | 3/6  | 3/6  | 0 |
| TestSmokeBitwise     | 2/3  | 2/3  | 0 |
| TestSmokeShift       | 0/2  | 0/2  | 0 |
| TestSmokeMemory      | 0/5  | 0/5  | 0 |
| TestSmoke32Bit       | 0/6  | 0/6  | 0 |
| (other)              | 22/23| 22/23| 0 |
| **Total**            | 28/51| 28/51| 0 |

Same 23 failures: `eq_true`, `eq_false`, `ne_true`, `gt_true`,
`ge_true`, `sub_basic`, `div_basic`, `mod_basic`, `xor_basic`,
`lea_basic`, `si_li_*`, `sc_lc_roundtrip`, `shl`, `shr`,
`add_16bit`, `add_carry_cascade`, `sub_16bit`, `or_16bit`,
`xor_16bit`, `mul_overflow`.

## Why the fix didn't help (analysis post-mortem)

The followup doc traced the failure mode to a one-token divergence at
step-4 offset 22 (STACK0 byte 1 position) where EQ(17) emits MARK_PC
(token 257) instead of byte value 0. The hypothesis: the LEA-local
diff routing at slots 32..63 leaks into byte 1's OUTPUT band, tipping
the head_bake's NEXT_PC vs byte-value logit competition.

My fix would have disabled the slots 32..63 leak at byte 1 *only if*
the head 3 was firing there. Looking at K-side slot 33 (`K[33] = 5.0`
const) and Q-side slot 33 selector pattern (which sums to 0 at all
STACK0 byte rows pre-fix), the head DOES fire at byte 1.

But disabling it produced no behavioral change. Two possibilities:

1. **The diff routing is semantically neutral at byte 1**: at the AX
   byte 1 K row, the L8 OUTPUT_LO/HI band may equal CLEAN_EMBED_LO/HI
   (both encoding nibble 0 for value 17's byte 1 = 0). Then
   `OUTPUT - CLEAN_EMBED = 0` at slots 32..63, and the leak doesn't
   actually exist. The followup doc's "leak" hypothesis is wrong.

2. **The actual leak path is different**: the contaminating MARK_PC
   signal at STACK0 byte 1 emerges somewhere ELSE in the residual —
   perhaps in L8/L9 OUTPUT band propagation, L11/L12 head_bake
   prep, or the head_bake competition itself favoring REG_PC token id
   257 over byte 0 at this specific residual fingerprint.

Either way, this targeted L10 head 3 patch is not the right surface.

## What this rules out

* The L10 PSH STACK0 passthrough head 3 firing at STACK0 byte 1 is
  NOT the cause of the byte 1 → MARK_PC corruption. Even with the
  head completely disabled at that Q row, EQ(17) still fails.
* Other smoke tests (var_simple, var_update, if_var implied by the
  brief's regression-protection warning) showed no regression either,
  meaning the byte 1 → AX byte 2 CLEAN_EMBED routing was a no-op or
  was already shadowed by another head (e.g., the L10
  `_layer10_stack0_byte_relay_head_spec` family which writes
  STACK0_BYTE1/2/3 via heads 4-6).

## What to try next (handoff)

The followup doc's recommended next investigation step #1 stands:
hook block-by-block residual at step-4 pos 135 BEFORE the emission,
comparing EQ(5) vs EQ(17). Identify the block where OUTPUT_HI/LO at
pos 135 diverges. Candidates the L10-head-3 ruled out:

* L8/L9 OUTPUT band consolidation for AX byte 0 with high nibble.
* L11/L12 OUTPUT band propagation across positions.
* The head_bake REG_PC weight (+20 at NEXT_PC) vs byte-value head
  weights — the competition itself may be biased.
* L1 `IS_BYTE` / `BYTE_INDEX_*` rules: maybe the marker token at pos
  135 emerges because the marker tokens write a small `NEXT_PC` cue
  that's amplified by some other head.

Recommended **ablation** before another rule change: directly patch
`model.head.bias[Token.REG_PC] = -30.0` in a smoke run and confirm
EQ(17) passes (the doc's step 3). This isolates whether the bug is
logit-competition driven; if EQ(17) still fails with REG_PC
hard-suppressed, the byte-value head is also producing 0 for byte 1
and the corruption is upstream of head_bake.

## Confidence

* **High** that the slots 32..63 differential routing at STACK0 byte 1
  Q rows is NOT the leak source — disabling the head there is a no-op.
* **High** that var_simple / var_update / if_var did not regress —
  smoke unchanged across all suites.
* **Medium** that the followup doc's hypothesis "highest-leverage fix
  surface is L10 head 3" was wrong. The slot 32..63 routing is
  apparently zero or harmless at byte 1.
* **Low** that any further single-rule L10 patch will improve smoke
  — per `feedback_single_rule_fixes_are_zero_sum.md`, this is the
  6th-or-so zero-sum L10 attempt in a week. The next investigator
  should pursue the head_bake-bias ablation before any more rules.

## Files

* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1500-1505`
  (the slot 33 selector I added the blocker to — REVERTED, no diff
  remains on this branch beyond this doc).

## Reproduction

```bash
cd /tmp/c4-stack0-fix/c4_release  # worktree
python -m pytest tests/test_smoke.py --tb=no -q  # 28/51, same set
# Optional: re-apply the Q[33] blocker to _layer10_psh_stack0_passthrough_head_spec
# and re-run smoke — result is byte-identical pass/fail set.
```
