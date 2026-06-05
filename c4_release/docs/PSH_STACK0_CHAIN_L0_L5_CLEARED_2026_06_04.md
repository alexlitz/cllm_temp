# PSH→STACK0 chain — L0..L5 cleared via DSLInterpreter; bug is L6+

Date: 2026-06-04
Worktree: `/tmp/c4-psh-stack0-chain/c4_release/`
Branch: `psh-stack0-chain` (off `e3acaad2` on `speedup-cache-and-buckets`)
Author: agent (Claude Opus 4.7)
Brief: PSH→stack0 byte-1 data-flow chain fix attempt — recover the 5
CMP-cluster smoke tests (`test_eq_true`, `test_ne_true`, `test_lt_false`,
`test_gt_true`, `test_ge_true`).

## TL;DR

**No code patch made.** The brief's hypothesis that the bug originates
in L0..L5 (blocks 0..5) is contradicted by two independent measurements:

1. The block-by-block prediction-row residual diff in
   [`EQ_BLOCK_DIFF_2026_06_04.md`](EQ_BLOCK_DIFF_2026_06_04.md) shows
   blocks 0..5 produce **byte-identical** prediction-row residuals
   between EQ(5,5) and EQ(17,17). First divergence is block 6.
2. A symbolic `DSLInterpreter` trace over the L0..L5 ops (this doc)
   confirms: **no STACK0_BYTE0 / ALU_LO/HI / OUTPUT_LO/HI / AX_CARRY_LO/HI
   state dim differs between byte=5 and byte=17 inputs** through the
   full L0..L5 op chain. The only dims that differ are the
   CLEAN_EMBED_LO/HI and FETCH_LO/HI input columns themselves — which
   *should* differ (those carry the IMM operand byte).

Per the zero-sum memory note
(`feedback_single_rule_fixes_are_zero_sum.md`, 0/5 historical) and the
brief's "stop after 1 attempt" rule, the correct action is to
**document the L0..L5 clearance** and decline a blind L6+ patch.

The brief's three sub-hypotheses for the bug surface (L1 STACK0_BYTE0
rule, L4 PC side-effect on STACK0, L5 LO/HI asymmetry) are all
**directly disconfirmed** by the DSL trace:

* L1 STACK0_BYTE0 rule fires *symmetrically* (gate is byte-value-
  independent: `L1H4[BP] AND IS_BYTE AND NOT H1[BP]`). Verified.
* L4 PC management does not write STACK0_BYTE0 (writes FETCH_LO/HI +
  TEMP only). Verified.
* L5 byte decode writes FETCH_LO/HI / OPCODE_BYTE_LO/HI via symmetric
  `_band_output_writes` (16 LO + 16 HI per head; same weight
  structure). Verified.

## Baseline measurements

Smoke baseline at `e3acaad2` HEAD: **28 passed / 23 failed** / 51
collected. CMP cluster:

| Test                  | Status |
|-----------------------|--------|
| `test_eq_true` (42,42)| FAIL   |
| `test_eq_false`(10,20)| FAIL   |
| `test_ne_true` (10,20)| FAIL   |
| `test_lt_true` (10,20)| PASS   |
| `test_lt_false` ?     | PASS   |
| `test_gt_true` (20,10)| FAIL   |
| `test_ge_true` (20,10)| FAIL   |

(`test_lt_true` passes per the L9 OP_LT gate fix at `61a7e12c`. The
brief's "5 CMP failures" list cited `test_lt_false`, but `test_lt_false`
is actually passing at this HEAD — the failing CMP set is
`test_eq_true`, `test_eq_false`, `test_ne_true`, `test_gt_true`,
`test_ge_true`. Updated for the record.)

## DSLInterpreter trace results

Script: `/tmp/dsl_psh_full.py` (off-tree, not committed).

The interpreter walked all L0..L5 ops (from
`compile_full_vm_dynamic(strict=False).ops_per_layer[0..6]`) on two
initial state dicts:

* `state_byte_5`: `CLEAN_EMBED_LO+5=1`, `CLEAN_EMBED_HI+0=1`,
  `FETCH_LO+5=1`, `FETCH_HI+0=1` (plus markers).
* `state_byte_17`: `CLEAN_EMBED_LO+1=1`, `CLEAN_EMBED_HI+1=1`,
  `FETCH_LO+1=1`, `FETCH_HI+1=1` (plus markers).

Both share `MARK_AX=1`, `OP_PSH=1`, `CONST=1`.

### Step-1: who writes STACK0_BYTE0 in L0..L5?

```
=== STACK0_BYTE0 writers in L0..L5 ===
  layer1_ffn  (the only writer)
```

### Step-2: state-dim diff post L0..L5

```
8 dims differ between byte=5 and byte=17 inputs:
  CLEAN_EMBED_HI+0: 1.000 -> 0.000 (delta=-1.000)
  CLEAN_EMBED_HI+1: 0.000 -> 1.000 (delta=+1.000)
  CLEAN_EMBED_LO+1: 0.000 -> 1.000 (delta=+1.000)
  CLEAN_EMBED_LO+5: 1.000 -> 0.000 (delta=-1.000)
  FETCH_HI+0: 1.000 -> 0.000 (delta=-1.000)
  FETCH_HI+1: 0.000 -> 1.000 (delta=+1.000)
  FETCH_LO+1: 0.000 -> 1.000 (delta=+1.000)
  FETCH_LO+5: 1.000 -> 0.000 (delta=-1.000)

--- STACK0/ALU/OUTPUT/AX_CARRY dims that differ ---
  (none)
```

Every STACK0/ALU/OUTPUT/AX_CARRY band stays byte-identical between
the two operands through L0..L5. The brief's L0..L5 hypothesis is
empirically false.

### Cross-check vs. EQ_BLOCK_DIFF measurement

| blk | norm(pred17 - pred5) |
|----:|---------------------:|
|  0  | 0.0000               |
|  1  | 0.0000               |
|  2  | 0.0000               |
|  3  | 0.0000               |
|  4  | 0.0000               |
|  5  | 0.0000               |
| **6**  | **339.4113**     | ← first divergence

Matches the symbolic trace: blocks 0..5 produce byte-identical
prediction-row residuals; block 6 is where mixing introduces the
delta.

## Why I'm not patching anything

1. **The brief's mental model is wrong.** "Blocks 0-5 are where the
   bug actually originates" is contradicted by both block-residual
   measurement (`EQ_BLOCK_DIFF_2026_06_04.md`) and symbolic trace
   (this doc). The legitimate IMM-byte input delta lives at the
   input-token columns (positions 2, 18) and propagates through L0..L5
   without contaminating STACK0/ALU/OUTPUT bands at the prediction
   row. The bug is the L6 attention/routing-FFN mixing of those
   columns into the prediction row.

2. **L6 routing FFN is verifier-blind.** Per
   [`L6_EQ_VERIFIER_BLIND_2026_06_04.md`](L6_EQ_VERIFIER_BLIND_2026_06_04.md),
   1315 of 1315 rules in `layer6_routing_ffn` carry no `scope` /
   `dominates_at` annotations. `verify_rule_strength` /
   `verify_rule_scopes` return zero issues — they cannot localize.

3. **Two prior attempts on this exact bug were no-ops or regressions.**
   * `STACK0_BYTE1_BLOCKER_ATTEMPT_2026_06_04.md` (L10 head 3 Q-row
     blocker): smoke 28/51 → 28/51, REVERTED.
   * `STACK0_ABLATION_2026_06_04.md` (`head.bias[REG_PC] -= 30`):
     made EQ(5,5) regress AND did not fix EQ(17,17). REVERTED.

4. **Zero-sum memory.** Per
   `feedback_single_rule_fixes_are_zero_sum.md`, 0/5 single-rule fix
   agents net positive. A 7th attempt at the same surface family is
   expected-value-negative.

## What this rules out (positive contribution)

The brief listed three L0..L5 sub-hypotheses. All disconfirmed:

| Hypothesis                                          | Status        |
|-----------------------------------------------------|---------------|
| L1 STACK0_BYTE0 rule not firing for the right marker | Disconfirmed: fires symmetrically; gate is `L1H4[BP]·IS_BYTE·NOT H1[BP]`, no byte-value dependency |
| L4 PC management has side-effect on STACK0          | Disconfirmed: L4 writes only FETCH_LO/HI and TEMP, never STACK0_BYTE0 |
| L5 byte decode has LO/HI nibble asymmetry for IMM   | Disconfirmed: symmetric `_band_output_writes(FETCH_LO, 32)` / `_band_output_writes(FETCH_HI, 48)` in every L5 fetch head spec |

This is a useful negative result: the next investigator can skip these
three rabbit-holes and focus on L6 attention / L6 routing FFN
asymmetry.

## Where the bug actually lives (handoff)

Per the existing measurements:

* **L6 attention (`layer6_attn` heads 0..5) + relay heads 6/7**.
  Inject the operand-column residual into the prediction row at
  block 6. Norm jump 0 → 339 at the prediction row in one block.
* **L6 routing FFN (`layer6_routing_ffn`, 1315 rules, ~1486 active
  hidden units)**. Per-opcode AX_CARRY / FETCH → OUTPUT routing band.
  Verifier-blind without rule annotation.
* **Block 28 (`ALUShiftComposite`)** — second injection, norm jump
  773 → 1290. Held as second suspect per block-diff doc.

What would actually move the needle (in priority order):

1. **Annotate L6 routing FFN rules with `scope` / `dominates_at`.**
   Even the 16 `_layer6_imm_fetch_to_output_hi_k` rules alone (the
   ones writing OUTPUT_HI_THIS_STEP+k at the AX-marker / IMM-step row
   gated on FETCH_HI+k) would let `verify_rule_strength` compare their
   max contribution against the OUTPUT_LO / OUTPUT_HI_THIS_STEP writer
   set authored elsewhere and surface the first
   "they-think-they-dominate-but-don't" pair. Per the
   `L6_EQ_VERIFIER_BLIND` doc, this is the budget-equivalent of one
   blind-patch loop.

2. **Block-diff at call ~90 (the first ILLEGITIMATE emit)** instead of
   call 7 (the legitimate IMM byte emission). The L6 injection at call
   7 is *intentional* (it's how the IMM byte enters the AX register);
   the divergence at call 90 is where some downstream consumer fails
   to scrub the L6 high-nibble path correctly. The current block-diff
   captures the injection point; a call-90 block-diff would catch the
   non-scrub point.

3. **EQ(42) vs EQ(17) block-diff.** Both fail (per
   `L6_EQ_VERIFIER_BLIND_2026_06_04.md`'s new EQ(42,42)=FAIL
   observation). Diffing two failing cases with different FETCH_HI
   nibbles (1 vs 2) tells you whether the bug is "fires per
   popcount(FETCH_HI)" or "fires once any FETCH_HI bit is set".

## Files

* New: `c4_release/docs/PSH_STACK0_CHAIN_L0_L5_CLEARED_2026_06_04.md`
  (this file).
* Off-tree (not committed): `/tmp/dsl_psh_full.py` (the DSLInterpreter
  trace), `/tmp/dsl_psh_trace.py` (an earlier scoped trace).
* Used (read-only): `c4_release/neural_vm/unified_compiler/dsl_interpreter.py`,
  `c4_release/neural_vm/unified_compiler/ops/l1_ops.py`,
  `c4_release/neural_vm/unified_compiler/ops/l2_ops.py`,
  `c4_release/neural_vm/unified_compiler/ops/l3_ops.py`,
  `c4_release/neural_vm/unified_compiler/ops/l4_ops.py`,
  `c4_release/neural_vm/unified_compiler/ops/l5_ops.py`,
  `c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py`.

## Reproduction

```bash
cd /tmp/c4-psh-stack0-chain/c4_release
python -W ignore /tmp/dsl_psh_full.py | tail -50
# Confirms: "STACK0/ALU/OUTPUT/AX_CARRY dims that differ — (none)"
python -m pytest tests/test_smoke.py --tb=no -q | tail -5
# 28 passed / 23 failed / 51 collected (pristine baseline; no smoke
# regression, no smoke recovery since no patch made)
```

## Confidence

* **High** that L0..L5 ops produce byte-identical STACK0/ALU/OUTPUT
  /AX_CARRY state for byte=5 vs byte=17 inputs. Direct symbolic
  measurement via `DSLInterpreter.run(ops_per_layer[0..6])`. Matches
  the prior `EQ_BLOCK_DIFF` block-residual measurement.
* **High** that the three L0..L5 sub-hypotheses in the brief are all
  disconfirmed. Each disconfirmation is a single-line trace of the
  op's `writes` / rule `conditions`.
* **High** that a blind patch in L0..L5 would not recover any CMP
  test (those layers don't carry the bug).
* **Medium-high** that the bug lives in L6 attention/routing-FFN.
  Block-diff measurement plus the unique-norm-jump pattern at block 6
  point there. The L6 routing FFN is verifier-blind, so further
  decl-verifier signal requires rule annotation first.
* **High** that following the brief's "stop after 1 attempt" rule on
  the *fix* side is the correct call. The contribution here is the
  negative-result clarification of where the bug is *not*.
