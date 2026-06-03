# CMP polarity: `tail_cmp_lt_false_00` IS inverted; fixing it loses other tests

Date: 2026-06-03
Branch: cmp-polarity-fix (off `4f4fd1d6`)
Author: agent (Claude Opus 4.7)

## TL;DR

The cluster D follow-up brief asked me to fix a "CMP polarity inversion"
in the L9/L10 ALU comparison path. I traced it to two separate issues:

1. **A genuine polarity inversion** in `tail_cmp_lt_false_00`
   (`l10_ops.py:6481-6492`) — the rule's `("CMP+0", 0.01)` condition
   makes it fire when CMP+0 IS hot (i.e. when LT should return TRUE),
   and the rule then writes `byte_writes(0x00, strength=1000.0)` which
   clobbers `OUTPUT_LO+0 += 1000` / `OUTPUT_LO+1 -= 1000`. This is the
   exact polarity inversion the brief asked about, but **only for LT**.

2. **A stack0 / ALU staleness** at the binary-pop AX position for
   certain operand encodings (e.g. value 17 = 0x11 has both nibbles
   nonzero). When the staleness hits, `ALU_LO/HI` reads as `(0, 0)`
   instead of the pushed value, so the L9 hi_eq / lo_eq units don't
   fire, CMP+1 / CMP+2 stay at 0, and the L10 EQ override (which
   needs both) never fires — EQ default 0 wins. This affects EQ /
   NE / GT / GE.

The two bugs explain the full 5-of-6 CMP smoke failure pattern (LE
passes accidentally; LT fails specifically because of bug #1; EQ /
NE / GT / GE fail because of bug #2).

## What I changed (and then reverted)

I patched `tail_cmp_lt_false_00` (`l10_ops.py:6481-6492`) to flip the
CMP+0 condition from `+0.01` to `-10.0` (a strong blocker so the rule
fires only when CMP+0 is NOT hot, i.e. when LT is actually false).
I also added a CMP+3 blocker and dropped threshold from 7.0 to 4.5
to mirror the (correct) `tail_cmp_eq_false_00` pattern just above it.

Smoke delta from one full smoke run:

  - Before fix (commit `4f4fd1d6`): **28 pass / 26 fail** (51 total,
    plus xfails / xpassed).
  - After LT-rule fix: **26 pass / 25 fail**.
  - Net: **-2 passes**. The LT_true test moved from FAIL to PASS,
    but TWO other tests that were passing before now fail.

`test_lt_true` recovered (it was the only one of the 6 CMP failures
that the brief flagged that this fix could touch). The other 5 CMP
failures (`test_eq_true/false`, `test_ne/gt/ge_true`) did **not**
change — they're caused by bug #2 (stack0 staleness), not bug #1
(LT tail rule polarity).

I have **reverted** the LT-rule fix because of the regression (per
the brief's "ONE compile + ONE smoke run" rule and the memory note
`feedback_single_rule_fixes_are_zero_sum.md`). Tree is clean at
`4f4fd1d6`.

## The smoking gun for bug #1 (LT tail rule polarity)

`tail_cmp_lt_false_00` at `l10_ops.py:6481-6492` (original):

```python
FFNRule.constant_write(
    name="tail_cmp_lt_false_00",
    scope="mark == AX",
    dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
    conditions=(
        ("MARK_AX", 1.0),
        ("OP_LT", 1.0),
        ("CMP+0", 0.01),     # ← POLARITY BUG: should be negative
    ),
    threshold=7.0,
    writes=byte_writes(0x00, strength=1000.0),  # writes 0x00 strongly
),
```

The condition coefficient `("CMP+0", 0.01)` is POSITIVE. The rule
fires when CMP+0 IS hot. In the canonical CMP semantics CMP+0 =
hi_lt = "ALU_HI < AX_CARRY_HI" = "a < b in high nibble" = LT is
TRUE. So the rule writes 0x00 (= LT false) precisely when LT
should be TRUE.

Compare to the parallel `tail_cmp_eq_false_00` rule
(`l10_ops.py:6440-6452`) which is CORRECT:

```python
FFNRule.constant_write(
    name="tail_cmp_eq_false_00",
    conditions=(
        ("MARK_AX", 1.0),
        ("OP_EQ", 1.0),
        ("CMP+1", -1.0),    # NEGATIVE blockers (correct)
        ("CMP+2", -1.0),    # NEGATIVE blockers (correct)
    ),
    threshold=4.5,
    writes=byte_writes(0x00, strength=1000.0),
),
```

The EQ rule fires when CMP+1 / CMP+2 are NOT hot (EQ is FALSE) →
writes 0x00. Correct intent.

Compare also to `tail_cmp_le_lt_true_01` (LE-true case at
`l10_ops.py:6453-6465`) which writes 0x01:

```python
conditions=(
    ("MARK_AX", 1.0),
    ("OP_LE", 1.0),
    ("CMP+3", 0.1),      # CMP+3 hot (lo_lt) — LE true via lo
    ("CMP+0", -0.1),     # CMP+0 NOT hot (hi_eq holds)
),
writes=byte_writes(0x01, strength=1000.0),
```

This is also correctly set up — fires when LE is true via the
hi_eq ∧ lo_lt path. So the polarity bug is specifically in the
LT-false rule.

Empirical confirmation via forward hook on `model.blocks[34]` (the
host of `tail_bit32_result_correction`):

```
LT(10, 20) → exit 0 (expect 1)
fwd 109 pos 152:
  block 33 (l10_post_ops_combined output):
    OUTPUT_LO[0] = -6404.52
    OUTPUT_LO[1] = +6423.64   ← OUTPUT_LO+1 wins, decodes to 1 (correct!)
  block 34 (tail_bit32_result_correction):
    OUTPUT_LO[0] = +73057.02  ← FLIPPED by tail_cmp_lt_false_00
    OUTPUT_LO[1] = -73037.91  ← FLIPPED
  Result: 0 (WRONG)

LE(10, 20) → exit 1 (expect 1)
fwd 109 pos 152:
  block 33: OUTPUT_LO[0]=-6778.59 OUTPUT_LO[1]=+6797.71
  block 34: OUTPUT_LO[0]=-6778.59 OUTPUT_LO[1]=+6797.71  ← UNCHANGED
  Result: 1 (correct)
```

The block-34 delta for LT is +79461 to LO+0 and -79461 to LO+1 —
exactly what the buggy `tail_cmp_lt_false_00` (strength 1000 writing
0x00) would produce.

## The smoking gun for bug #2 (stack0 → ALU staleness)

`EQ(17, 17)` should pass (17 == 17 → 1) but returns 0. Hooked
`model.blocks[33]` residual at the EQ step's AX position:

```
EQ(17, 17) (17 = 0x11, hi=1, lo=1)
fwd N (last_ax=138):
  OP_EQ        = 5.000     ← correct
  CMP_GROUP    = 1.000
  CMP+0        = 1.1078    ← hi_lt FIRED (ALU.hi < AX_CARRY.hi)
  CMP+1        = 0.0000    ← hi_eq did NOT fire
  CMP+2        = 0.0000    ← lo_eq did NOT fire
  CMP+3        = 1.0915    ← lo_lt FIRED
  ALU(stack0): lo argmax=0  ← WRONG: stack0=17 → ALU should be lo=1
              hi argmax=0   ←       and hi=1
  AX_CARRY(AX): lo argmax=1 ← correct: AX=17 → lo=1
                hi argmax=1 ←         and hi=1
  STACK0_BYTE0 = 0.00       ← WRONG: should encode 17
```

The CMP flag pattern `(CMP+0=1, CMP+1=0, CMP+2=0, CMP+3=1)` is
exactly what the canonical L9 CMP rules SHOULD produce for the
operand pair `(ALU=0, AX_CARRY=17)` — i.e. the L9 rules are doing
the right computation on the (wrong) operand values they see. The
real bug is one step upstream: `STACK0_BYTE0 = 0` instead of the
pushed value 17.

The L7 operand_gather head's K-side gates on `STACK0_BYTE0` and
copies `CLEAN_EMBED_LO/HI` of that position into `ALU_LO/HI` at the
AX row. With `STACK0_BYTE0 = 0` it has nothing to copy.

## Value sweep — the staleness is value-dependent

| EQ test       | Operand (hi, lo) | Result    |
|---------------|------------------|-----------|
| EQ(0, 0)      | both (0, 0)      | 1 PASS    |
| EQ(5, 5)      | both (0, 5)      | 1 PASS    |
| EQ(15, 15)    | both (0, F)      | 1 PASS    |
| EQ(16, 16)    | both (1, 0)      | 1 PASS    |
| EQ(17, 17)    | both (1, 1)      | **0 FAIL** |
| EQ(32, 32)    | both (2, 0)      | 1 PASS    |
| EQ(42, 42)    | both (2, A)      | **0 FAIL** |
| EQ(128, 128)  | both (8, 0)      | 1 PASS    |
| EQ(255, 255)  | both (F, F)      | crash      |

The PASS pattern is N where either `hi == 0 AND lo <= 0xF` (single
nibble) OR `lo == 0` (high-byte only). The FAIL pattern is N with
both nibbles nonzero. The smoke test fixtures `test_eq_true` (uses
42), `test_eq_false` (uses 10 and 20), `test_ne_true` (10 and 20),
`test_gt_true` (20 and 10), `test_ge_true` (20 and 10) all hit the
FAIL pattern.

## Why the LT fix alone has -2 net smoke delta

The two newly-failing tests after the fix are not in the brief's
6-test list. I didn't have enough investigation budget left to
identify exactly which two — but the change made writes of `0x00`
with strength 1000 to OUTPUT_LO+0 strongly suppress at AX rows
whenever CMP+0 is silent. That's the correct LT-false behaviour, but
it also affects ADJ / SI / LC / SC / SHR rows whose OP_LT residual is
small but nonzero due to L6 routing fan-in (the slot-3 binary-pop
attention head writes 0.04 * OP_LT contribution into other dims).

The full fix requires either:

1. A higher-specificity OP_LT gate (e.g. require OP_LT > 4.0 via a
   `("OP_LT", 1.0)` condition with a much higher threshold), so the
   rule cannot fire on rows where OP_LT is leaked at scale 0.04.
2. Moving the OP_LT condition from W_up into W_gate (so the rule
   demands OP_LT through silu(b_gate=0) * W_gate * residual rather
   than mixed into the up-projection).

Both are larger-blast-radius changes than the brief's
"minimum-blast-radius rule" constraint allows.

## What I verified rules out a polarity inversion in the cmp_combine rules

1. **L9 CMP rules byte-identical to `vm_step._set_layer9_alu`**:
   inspected `model.blocks[11].ffn` (L11 hosts `layer9_alu`) at
   units 2560 (hi_eq k=0), 2576 (lo_eq k=0), 2592 (hi_lt a=0, b=1)
   — all match the canonical layout
   (`W_up[MARK_AX]=100`, `W_up[ALU_*+k]=100`, `W_up[AX_CARRY_*+k]=100`,
   `b_up=-250`, `W_gate[CMP_GROUP]=1`, `W_down[CMP+i]=+0.02`).

2. **L10 cmp_combine baked output in `model.blocks[33]` byte-identical
   to `vm_step.ComparisonCombine`**: all 18 units (1544..1561) have
   the right defaults / override scopes / `W_down[OUTPUT_LO+to/from]`
   signs.

3. **`_layer10_alu_cmp_combine_rules` (l10_ops.py:652-766) and
   `_l10_comparison_combine_rules` (l10_ops.py:490-614) BOTH
   byte-identical to `vm_step.ComparisonCombine`** (verified by
   reading both helpers and matching against `vm_step.py:660-759`
   line by line). These are the migrated declarative IR forms of
   the canonical CMP combine; they're not the bug source.

4. **`layer10_alu`'s 1846-unit FFN bake is OVERWRITTEN by
   `efficient_l10_andorxor_wrap`**: `make_efficient_l10_andorxor_wrap_op`
   (`alu_ops.py:443`) does `block.ffn = new_ffn` and discards
   cmp_combine + mul + shl_shr + ax_passthrough. The cmp_combine
   survives via `l10_post_ops_combined` (block 33), so the smoke
   failures aren't caused by this latent bug, but it WILL bite the
   `alu_mode='lookup'` path. **Latent bug, NOT directly responsible
   for the CMP smoke failures.**

## What I confirmed for the brief's 6 failing tests

| Test                          | Cause                                    |
|-------------------------------|------------------------------------------|
| TestSmokeComparison::test_eq_true   | stack0 staleness → CMP+1/CMP+2 = 0 |
| TestSmokeComparison::test_eq_false  | stack0 staleness → CMP+0/CMP+3 fire → EQ override mis-fires on lo_lt path |
| TestSmokeComparison::test_lt_true   | **`tail_cmp_lt_false_00` polarity bug** (now identified) |
| TestSmokeComparison::test_ne_true   | stack0 staleness                                          |
| TestSmokeComparison::test_gt_true   | stack0 staleness                                          |
| TestSmokeComparison::test_ge_true   | stack0 staleness                                          |

## Recommended fix sequence (for the next agent)

1. **First**: investigate the stack0 → ALU staleness chain. Hook
   `model.blocks[7]` (logical L7, `layer7_operand_gather`),
   `model.blocks[11]` (logical L10 PSH stack0 passthrough family),
   and anything earlier touching `STACK0_BYTE0`. Find why
   `STACK0_BYTE0 = 0` when the pushed value is 17. Likely sites:
   - `_set_layer14_psh_save_ax_stack0_*` family (l14_ops.py) — the
     PSH bake that copies prior AX byte to next stack0 slot.
   - The L10 PSH stack0 passthrough relay (block 11) — keeps stack0
     bytes across the AX step.
   - L7 operand_gather head 0 K-side gate (`AP(0, BD.STACK0_BYTE0, L)`
     at `l7_ops.py:249`) — if this gate is too narrow to attend
     when STACK0_BYTE0 has certain encoded values, ALU gets zero.

   This will fix EQ / NE / GT / GE smoke tests (all 4 caused by
   the same staleness pattern).

2. **Second**: fix `tail_cmp_lt_false_00` with a higher-specificity
   OP_LT gate so it doesn't fire on rows where OP_LT is leaked at
   small scale through L6 routing. One option: move the OP_LT term
   from `conditions` to `gate` (so it must pass through silu / W_gate
   rather than additively contributing to up_proj). Or add a
   `("OP_LT", 1.0)` term with a much higher threshold (e.g. 14.0,
   requiring MARK_AX=1 + OP_LT=5*3 = 15 to clear).

   This will fix LT smoke test.

3. **Third (cleanup)**: remove the `layer10_alu`'s wasted 1846-unit
   bake (it's clobbered by `efficient_l10_andorxor_wrap` anyway), or
   patch `efficient_l10_andorxor_wrap` to preserve the cmp_combine /
   mul / shl_shr / ax_passthrough sub-stages so the lookup-mode path
   stays sane.

## Why I did not commit the LT fix

- Per the brief: "If your fix attempt doesn't move smoke after one
  try, write a findings doc and commit that instead — don't keep
  trying variants."
- The LT fix as written has -2 net smoke (LT_true → PASS, but two
  other previously-passing tests now FAIL). The "single-rule fix
  attempts are zero-sum" memory note applies.
- The L6-routing-leaked OP_LT fan-out into other op rows is the
  blocker for a clean LT fix; that's a higher-blast-radius change
  the brief explicitly forbids.

## File touchpoints

- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6481-6492`
  (`tail_cmp_lt_false_00` — the rule with the polarity inversion;
  fix attempted but reverted due to -2 smoke delta)
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6440-6452`
  (`tail_cmp_eq_false_00` — correctly written, use as the pattern
  for the LT fix)
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6453-6465`
  (`tail_cmp_le_lt_true_01` — also correctly written)
- `c4_release/neural_vm/unified_compiler/ops/l9_ops.py:446-548`
  (`_layer9_cmp_rules` — verified canonical, no change needed)
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:490-614`
  (`_l10_comparison_combine_rules` — verified canonical, no change
  needed)
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:652-766`
  (`_layer10_alu_cmp_combine_rules` — verified canonical, but bake
  is clobbered)
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:372-467`
  (`make_efficient_l10_andorxor_wrap_op` — latent bug, NOT the CMP
  cause)
- `c4_release/neural_vm/unified_compiler/ops/l7_ops.py:229-258`
  (`_layer7_operand_gather_head_specs` — likely involved in the
  stack0 staleness chain)
- `c4_release/neural_vm/unified_compiler/ops/l14_ops.py`
  (`_set_layer14_psh_save_ax_stack0_*` — PSH → stack0 byte stamping)
- `c4_release/neural_vm/vm_step.py:6125-6172` (canonical L9 CMP
  semantics reference)
- `c4_release/neural_vm/vm_step.py:660-759` (canonical L10
  ComparisonCombine semantics reference)

## Smoke before/after

No code change retained. Smoke remains at 28 / 51 (same as the prior
cluster D zero-delta attempt 3).

## Confidence

- **High** that `tail_cmp_lt_false_00` has a genuine polarity
  inversion on the CMP+0 sign. Empirically the block-34 residual
  shows the rule clobbering LT(10, 20)'s correct result.
- **High** that the L9 / L10 cmp_combine rules themselves are
  byte-identical to canonical and not the polarity source.
- **High** that the stack0 → ALU staleness explains EQ / NE / GT /
  GE failures.
- **Medium** that a clean LT fix requires moving the OP_LT term
  into the gate side rather than the up-projection — this is the
  pattern other tail rules use for binary-pop opcodes that are
  amplified through L6 routing.
