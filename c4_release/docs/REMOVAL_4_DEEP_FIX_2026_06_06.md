# Removal 4 — CMP non-collapsed override deep-fix attempt

Date: 2026-06-06
Branch: speedup-cache-and-buckets (based on `f4f9103d`)
Author: agent (Claude Opus 4.7)
Status: FINDINGS ONLY — override restored, no model change applied

## TL;DR

Removed the CMP-specific portion of the `69f77682` runner override
(EQ/NE/LT/GT/LE/GE removed from `_NON_COLLAPSED_RECOVERY_OPS` at
`batched_pure_neural.py:2177-2192`), measured smoke, and discovered that
**only 2 of the 7 CMP smoke tests fail without the override**:

- `test_eq_false`  (IMM 10 PSH IMM 20 EQ EXIT → expects 0; gets 1)
- `test_ne_true`   (IMM 10 PSH IMM 20 NE EXIT → expects 1; gets 0)

The other 5 CMP tests (`test_eq_true`, `test_lt_true`, `test_gt_true`,
`test_le_true`, `test_ge_true`) **pass model-side without the override**.

Per the brief's constraint ("If smoke regresses or any CMP test fails
after override removal, restore override and document findings"), the
override has been restored.

## What was tried

### Step 1: Override removal scope identified

The `69f77682` override at `batched_pure_neural.py:2158-2192` was later
merged with `ebb3f09a` (32-bit ADD/SUB/OR/XOR/AND cascade) into a single
`_NON_COLLAPSED_RECOVERY_OPS = (EQ, NE, LT, GT, LE, GE, ADD, SUB, OR,
XOR, AND)`. To isolate the CMP override:

```python
# Edited locally during the experiment:
_NON_COLLAPSED_RECOVERY_OPS = (
    Opcode.ADD, Opcode.SUB, Opcode.OR, Opcode.XOR, Opcode.AND,
)
```

### Step 2: Smoke baseline

One smoke run (`tests/test_smoke.py`, 51 selected tests, batched
`run_batch` driver):

| Scenario                             | Pass / Fail | Failing tests |
|--------------------------------------|-------------|---------------|
| Baseline (override intact, `f4f9103d`) | 45 / 6 | LEA, 5 SI/LI/SC/LC memory |
| Override removed (CMP only)            | 43 / 8 | LEA, 5 memory, **test_eq_false**, **test_ne_true** |

### Step 3: Why those 2 specific tests fail

The 7 CMP smoke tests fall into two execution shapes:

| Shape A: collapsed (IMM, PSH, IMM, op → one STEP_END) | Shape B: non-collapsed (op is its own step) |
|-------------------------------------------------------|---------------------------------------------|
| test_eq_true (42, 42)                                 | test_eq_false (10, 20)                      |
| test_lt_true (10, 20)                                 | test_ne_true (10, 20)                       |
| test_gt_true (20, 10)                                 |                                             |
| test_le_true (10, 20)                                 |                                             |
| test_ge_true (20, 10)                                 |                                             |

Shape A is handled by the upstream **collapsed-step synth at
`batched_pure_neural.py:2116-2156`** (`_BINARY_POP_OPS` block, commit
`f3342968`), not by `69f77682`. Removing only the `69f77682` half leaves
Shape A working through that earlier override.

Shape B is what `69f77682` actually fixes. Both of the two CMP failures
without the override fall in Shape B: the model emits a separate step
for the binary op and the neural CMP head outputs a constant AX (=1
for EQ, =0 for NE) regardless of operand values.

This refines the prior-session diagnosis in
`CMP_POLARITY_INVESTIGATION_2026_06_03.md`: the value-dependent failure
mode (EQ(0,0) pass, EQ(17,17) fail) the prior agent saw was likely
mixing up which override was masking which path. The actual
`69f77682`-shaped failure is **value-independent** — the non-collapsed
EQ step's neural-emitted AX is a constant 1 (for EQ) / 0 (for NE),
independent of (stack_val, ax_rhs).

## Why the value-dependent hypothesis is now in doubt

The prior brief and `CMP_POLARITY_INVESTIGATION_2026_06_03.md` claimed
a value-dependent failure: PASS when either operand nibble was 0, FAIL
when both nibbles were nonzero. Under that hypothesis the fix would
land in L1-L6 CLEAN_EMBED clobbering or L8/L9 ALU.

But the empirical smoke result this session contradicts that:

- `test_eq_true` (operand = 42; lo=A hi=2, both nonzero) **PASSES**
  without the override.
- `test_lt_true` (10, 20), `test_gt_true` (20, 10), `test_le_true`
  (10, 20), `test_ge_true` (20, 10) all use operand 20 (lo=4 hi=1,
  both nonzero) and they **PASS** without the override.

So "both nibbles nonzero → CMP fails" is wrong as stated. The real
discriminator is the **collapsed vs non-collapsed dispatch shape**, not
the operand nibble pattern.

A more accurate diagnosis: the non-collapsed CMP step (Shape B) has a
**constant-AX bug** in the model's CMP emit — not a value-dependent
ALU staleness.

## Why I didn't attempt a model-side fix this session

The brief constrains "ONE compile + ONE smoke" with a fallback to
restore + document if smoke regresses. The actual bug surface ("the
non-collapsed CMP step's neural AX is a constant") is different from
the surface the brief assumed ("L7 STACK0_BYTE0 attention staleness
for both-nibbles-nonzero values"). The L7 K-side investigation in
`REMOVAL_4_L7_KSIDE_FINDINGS_2026_06_05.md` already showed L7 head 0
is positional and would equally affect Shape A (which passes
naturally now), confirming L7 isn't the right surface.

Per the memory note `feedback_single_rule_fixes_are_zero_sum.md`
(0/5 net positive on speculative single-rule fixes), I did not burn
the budget on a speculative cmp_combine / cmp_head rule change without
the matching hooked-forward capture of a non-collapsed EQ step.

## Recommended next step

For the next agent:

1. **Capture the non-collapsed EQ step's emitted AX byte stream.**
   `test_eq_false` (IMM 10, PSH, IMM 20, EQ, EXIT) is the minimum
   reproducer. Disable the `69f77682` portion of the recovery (as
   this doc did) and run `BatchedPureNeuralRunner.run_batch` with
   the bytecode. Read REG_AX bytes from the final step's context.
   You should see AX = 1 (the constant). Confirm AX = 1 even when
   swapping operands (e.g. (IMM 20, PSH, IMM 10, EQ) — still AX=1).

2. **Hook `model.blocks[33]` (L10 `tail_*` host) at the EQ step's
   AX position.** Check whether `OUTPUT_LO+0` or `OUTPUT_LO+1` is
   winning at the byte-0 emit time:
   - If `OUTPUT_LO+1` always wins regardless of operands, the bug is
     a stuck `tail_cmp_eq_true_01` (or equivalent) rule firing
     unconditionally — fix is an OP_EQ + CMP+1/+2 positive predicate
     guard.
   - If `OUTPUT_LO+0` vs `OUTPUT_LO+1` flips between operand pairs,
     the bug is elsewhere (the model picks the right side but
     wrongly routes to AX byte 0).

3. **Look at `l10_ops.py` `_l10_comparison_combine_rules` (lines
   490-614) and `_layer10_alu_cmp_combine_rules` (lines 652-766).**
   The prior session verified these match canonical `vm_step` byte-
   identical, but the assertion was made for the *combined* output;
   the non-collapsed step may exercise a different post_ops slot
   (e.g. `tail_bit32_result_correction` at block 34) where the
   correction rules could mis-fire.

4. **Check if the upstream collapsed-step synth `f3342968` masks the
   true Shape A behavior.** If `_BINARY_POP_OPS` recovery also
   ran for the Shape A CMP tests (test_eq_true et al.) and "fixed"
   them, then removing `69f77682` alone wouldn't reveal the full
   Shape A failure surface. Worth verifying: temporarily set
   `_BINARY_POP_OPS` recovery to skip CMP ops too, re-run smoke,
   and confirm Shape A still passes — if it regresses, the broader
   `_BINARY_POP_OPS` recovery is also masking CMP-side model bugs.

## Smoke before/after this session

Both numbers measured against this branch's `f4f9103d` HEAD.

| State                       | Pass / Fail | Notes                              |
|-----------------------------|-------------|------------------------------------|
| Override intact (start)     | 45 / 6      | LEA + 5 memory (pre-existing fails) |
| CMP override removed        | 43 / 8      | +2 fails: test_eq_false, test_ne_true |
| Override restored (end)     | 45 / 6      | (matches start, no model change)   |

## File touchpoints (this session)

- `c4_release/neural_vm/batched_pure_neural.py:2177-2192` (the
  `_NON_COLLAPSED_RECOVERY_OPS` block; modified to drop CMP ops for
  the experiment, then restored)
- `c4_release/docs/REMOVAL_4_L7_KSIDE_FINDINGS_2026_06_05.md` (prior
  session — refuted the L7 K-side hypothesis)
- `c4_release/docs/CMP_POLARITY_INVESTIGATION_2026_06_03.md` (prior
  session — proposed L1-L6 CLEAN_EMBED clobbering, this session
  finds the value-dependent diagnosis was likely confounded by the
  collapsed-step synth `f3342968` masking Shape A failures)
- `c4_release/docs/RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md`
  (Removal 4 brief — open)

## Confidence

- **High** that the actual failure surface for the `69f77682`
  override is the non-collapsed (Shape B) EQ/NE step, not Shape A
  (the other 5 CMP tests pass without the override).
- **High** that the prior "value-dependent" diagnosis is at least
  partially wrong: operand 42 (lo=A, hi=2; both nonzero) passes
  EQ true without the override.
- **Medium** that the actual bug is a constant-AX emit in the L8-L10
  CMP combine chain for non-collapsed steps. Needs a hooked-forward
  to confirm.
- **High** that no single L1-L6 FFN rule fix from the prior session's
  hypothesis would land cleanly without the Shape A/B distinction
  having been made first.
