# L6 EQ(17,17) fix attempt — verifier-blind, no patch made — 2026-06-04

Date: 2026-06-04
Worktree: `/tmp/c4-l6-eq-fix/c4_release/`
Branch: `l6-eq-fix` (off `0863d7cb` on `speedup-cache-and-buckets`)
Author: agent (Claude Opus 4.7)
Brief: §"Recommended next step" of
[`EQ_BLOCK_DIFF_2026_06_04.md`](EQ_BLOCK_DIFF_2026_06_04.md) — attempt
the L6 EQ(17,17) byte-1 corruption fix.

## TL;DR

**No code patch made.** The verifier (`verify_rule_strength` /
`verify_rule_scopes`) cannot localize the L6 EQ regression because
**0 of 1315** rules in `layer6_routing_ffn` carry the `scope` /
`dominates_at` annotations the verifier checks against. Per the brief
("Do NOT patch any L6 rule weight directly without first running the
verifier") and the memory note
[`feedback_single_rule_fixes_are_zero_sum.md`](../../.claude/projects/-home-alexlitz-Documents-misc-c4-release/memory/feedback_single_rule_fixes_are_zero_sum.md)
(0/5 historical success on blind single-rule L6 patches), the correct
move is to NOT patch and document the blocker.

Additional finding: **EQ(42,42) also fails** (the brief assumed only
EQ(17,17) failed). The failure mode is therefore "any IMM with
FETCH_HI != 0", not value-17-specific — strengthening the hypothesis
that the bug surface is along the FETCH_HI / OUTPUT_HI relay band,
but **not** narrowing which of the 144 FETCH_HI-gated L6 rules is at
fault.

## Baseline measurements (single compile each)

### Pure-neural smoke (`tests/test_smoke.py`, 51 collected)

| Suite                  | Result                                 |
|------------------------|----------------------------------------|
| pre-attempt baseline   | **28 passed / 23 failed** / 51 total   |

(The brief stated "≥ 29 / 52" as the floor. Current HEAD is 28/51,
i.e. one shy of the cited floor before any change. Reported pristine.)

### EQ probe (off-tree script `/tmp/eq_probe.py`)

| Program        | Result | Status |
|----------------|-------:|--------|
| EQ(5,5)        | 1      | PASS   |
| EQ(17,17)      | 0      | FAIL   |
| EQ(42,42)      | 0      | FAIL   |

The EQ(42,42) failure is a new observation. 42 = `0x2A` has FETCH_HI
nibble = 2 (bit 1 set) and FETCH_LO nibble = 10 (bit 3 set). 17 = `0x11`
has FETCH_HI=1 and FETCH_LO=1. 5 = `0x05` has FETCH_HI=0. The shared
property of failing cases is `FETCH_HI != 0` — i.e. any IMM whose
high nibble is non-zero. This generalises the prior
EQ_BLOCK_DIFF_2026_06_04.md observation beyond the specific 17-vs-5
operand pair.

## Verifier run (the brief's procedure step 2)

The brief's "decl_verifier.py verify_rule_strength --op
layer6_routing_ffn" is a library call, not a CLI. Invoked via
`verify_op.py`-style shim:

```python
from neural_vm.unified_compiler.decl_verifier import (
    verify_rule_strength, verify_rule_scopes,
    _collect_ffn_rules_from_op, collect_all_authored_ops,
)
from neural_vm.unified_compiler.ops.l6_ops import make_layer6_routing_ffn_op
op = make_layer6_routing_ffn_op()
rules = _collect_ffn_rules_from_op(op)
competition = [o for o in collect_all_authored_ops() if o.name != op.name]
s_issues = verify_rule_strength(op, registry, backbone_bounds=bb,
                                ops_for_competition=competition)
sc_issues = verify_rule_scopes(op, registry, require_scope=False)
```

Result:

```
op=layer6_routing_ffn rules=1315
strength runtime: 3.2s
strength_violation: 0
no_dominates_at: 0
scope_violation: 0
```

Drilling into why everything reports clean:

```
total=1315  with scope=0  with dominates_at=0
```

**Every single rule in `layer6_routing_ffn` has `scope=None` and
`dominates_at=None`.** Both verifier checks silently skip annotation-
less rules (`require_scope=False` and `require_dominates=False` are
their defaults). The verifier therefore cannot answer "which rule is
the bug" for this op; it can only attest that nothing it knows about
fails. The brief's own caveat anticipated this ("Per
feedback_single_rule_fixes_are_zero_sum.md, manual rule patches are
0/5 historically. The candidate fix surface is..." — i.e. the audit
must precede any patch).

Sibling L6 ops verified for comparison (also no issues found):

| Op                                          | Rules | With scope/dominates_at |
|---------------------------------------------|------:|------------------------:|
| `layer6_attn`                               |     0 |                       — |
| `layer6_relay_heads`                        |     0 |                       — |
| `layer6_routing_ffn`                        |  1315 |                   0 / 0 |
| `layer6_ent_after_jsr_sp_byte0_fixup`       |     7 |                   7 / 7 |

Only the small `ent_after_jsr_sp_byte0_fixup` op (7 rules, irrelevant
to EQ — it targets ENT-after-JSR SP fixup) is verifier-instrumented.
The big surface (`layer6_routing_ffn`, ~1486 active hidden units per
the prior block-diff doc) is unannotated.

## Structural scan of FETCH_HI-gated L6 routing rules

Implemented a manual scan as a partial substitute for the verifier
sweep. Of the 1315 rules in `layer6_routing_ffn`:

| Category                                                     | Count |
|--------------------------------------------------------------|------:|
| Rules gated on `FETCH_HI+k` (any k)                          |   144 |
|   ... writing `OUTPUT_HI_THIS_STEP`                          |    80 |
|   ... writing `OUTPUT_LO`                                    |    64 |
|   ... writing `AX_CARRY_HI`                                  |    16 |
| Of the 144, also have `MARK_AX > 0` condition (AX marker)    |    16 |
|   ... all writing `OUTPUT_HI_THIS_STEP+k` (none write LO)    |    16 |

The 16 AX-marker / FETCH_HI / OUTPUT_HI writers are the
`_layer6_imm_fetch_to_output_hi_k` rules (k=0..15). Their conditions:

```
OP_IMM*1 + OP_EXIT*-20 + OP_JMP*-20 + MARK_AX*1 + MARK_PC*-8 + IS_BYTE*-10
threshold = 4.0
gate = FETCH_HI+k
write = (OUTPUT_HI_THIS_STEP+k, 0.02)
```

These rules are **architecturally correct** as written: they're the
intended path by which the IMM byte's high nibble lands in the AX
register's high byte. The prior block diff's call-7 "first
divergence" at block 6 IS this write firing, and the brief itself
acknowledges call 7 is a *legitimate* divergence (EQ(5) emits 5,
EQ(17) emits 17 — both correct). The corruption is at call ~90, not
call 7.

**No rule in the FETCH_HI / AX-marker / OUTPUT_HI band has an
obvious "LO works but HI is missing a blocker" asymmetry.** The
threshold and condition geometry is symmetric between the
`_layer6_imm_fetch_to_output_lo_*` and `_layer6_imm_fetch_to_output_hi_*`
families (same conditions, same threshold, same gate-by-FETCH band
structure, just LO vs HI gate dim).

If the bug is in this band, it is more subtle than a missing blocker —
e.g. an interaction with a downstream consumer's normalisation
assumption, a cross-step staleness in the OUTPUT_HI_PREV_STEP read
used by the next step, or a competing writer outside L6 that the
brief's localisation didn't account for (the brief flags block 28
`ALUShiftComposite` as a second-injection candidate).

## Why no patch was attempted

The brief's procedure step 5 was "Make ONE targeted fix attempt: …".
Each suboption:

* **Add a missing MARK_PC blocker** — the 16 AX-marker /
  `_layer6_imm_fetch_to_output_hi_k` rules already have `MARK_PC*-8`
  in their conditions; MARK_PC=1 alone takes the condition sum from
  `2.0` to `-6.0`, well below the `4.0` threshold. Strengthening this
  blocker further has no effect because the rule already cannot fire
  at MARK_PC rows.
* **Rebalance the threshold so FETCH_HI is not enough alone** —
  FETCH_HI is the gate, not a condition. It scales the write but does
  not contribute to firing. The conditions sum (≈2.0 at AX-marker for
  IMM, before negative markers reduce it) is *already* below the
  threshold (4.0). Either there is additional residual amplification
  I am not modeling (likely — MARK_AX may not be 1.0 in practice; the
  L5 opcode-decode writes OP_IMM with weight `10.0/S * S = 10.0`
  cf. `_opcode_decode_main_rules`), or the bake's actual firing
  condition is different from the literal symbolic score I derived
  by hand. Either way, raising the threshold blindly risks killing
  the rule entirely → EQ(5,5) regression.
* **Scope to MARK_PC=1 only** — would invert the intended semantic
  (this rule fires at MARK_AX, not MARK_PC). Trivial regression.

None of the three suboptions has a defensible hypothesis without
either (a) measuring the actual residual values feeding L6 at the
suspect position class, or (b) the verifier annotations the rule
already lacks. Per the memory note's 0/5 history, attempting one
anyway is expected-value-negative.

## What would unblock the next attempt

1. **Annotate the L6 routing FFN rules with `scope` and
   `dominates_at`.** Once even the 16 `_layer6_imm_fetch_to_output_*`
   rules carry the predicate-DSL declarations the verifier expects,
   `verify_rule_strength` can compare their max contribution against
   the OUTPUT_LO / OUTPUT_HI_THIS_STEP writer set authored elsewhere
   (L8 multibyte_routing, L10 stack0_byte_relay, L14 cleanups) and
   surface the first "they-think-they-dominate-but-don't" pair. This
   is the same path that S-6 unblocked for L10's
   `tail_bit32_result_correction` (4040 lines of L10 ops are how the
   `collect_all_authored_ops` registry got populated). 1486 active L6
   units * ~10 min/agent for hand-annotation is roughly the same
   budget as a single failed blind-fix loop.

2. **Run the block-diff at call ~90** (where the actual corruption
   shows up — not call 7 where the legitimate IMM divergence enters
   the residual). The current block-diff doc localises *call 7*'s
   first-divergence, which is intended behavior. Re-running the same
   per-block diff at call 90 — the first ILLEGITIMATE emit — would
   point at the layer that *fails to scrub* the L6-injected delta
   when the cached prefix replays.

3. **Compare against EQ(42,42).** This run also fails (new
   observation). Diffing per-block residual norms between
   EQ(42,42) and EQ(17,17) — both failing, with different FETCH_HI
   nibbles — would distinguish "bug fires per-nibble-bit" (delta will
   scale with `popcount(FETCH_HI)`) from "bug fires once any
   FETCH_HI bit is set" (delta will be flat across the two). The
   prior diff used EQ(5)-vs-EQ(17), which can't distinguish these.

## Confidence

* **High** that `verify_rule_strength` has no information to offer
  for `layer6_routing_ffn` in its current annotation state. Direct
  measurement: 0/1315 rules carry `scope` / `dominates_at`.
* **High** that EQ(42,42) also fails (direct measurement on this
  worktree's pristine HEAD).
* **High** that the brief's three patch suboptions cannot be applied
  with positive expected value given (1) no verifier signal, (2) no
  obvious LO/HI asymmetry in the FETCH_HI-gated routing rules, (3)
  0/5 historical blind-patch success rate, (4) borderline baseline
  smoke (28/51 vs the cited 29/52 floor).
* **Medium** that re-running the block-diff at call ~90 would
  identify a different (downstream) layer as the actual scrub
  failure. The brief's own block-diff at call 7 is by design
  capturing the L6 *injection*, not the downstream *non-scrub*; the
  bug is plausibly in a downstream consumer that depends on a
  specific FETCH_HI=0 baseline.

## Files

* New: `c4_release/docs/L6_EQ_VERIFIER_BLIND_2026_06_04.md` (this
  file).
* Used (read-only): `c4_release/neural_vm/unified_compiler/decl_verifier.py`,
  `c4_release/neural_vm/unified_compiler/ops/l6_ops.py`,
  `c4_release/neural_vm/unified_compiler/ir.py`,
  `c4_release/neural_vm/unified_compiler/primitives.py`,
  `c4_release/neural_vm/unified_compiler/building_blocks_dsl.py`,
  `c4_release/neural_vm/unified_compiler/ops/l5_ops.py`,
  `c4_release/verify_op.py`.
* Off-tree script: `/tmp/eq_probe.py` (EQ(5/17/42) probe — not
  committed).
* No production code modified.
