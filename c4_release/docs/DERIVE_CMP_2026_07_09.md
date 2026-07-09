# DERIVE_CMP — the COMPARISON family from ONE zero-detector (task #446)

**Date:** 2026-07-09 · **Flag:** `C4_DERIVE_CMP` (DEFAULT-OFF) · **Branch:** `derive-cmp`
**Golden:** flag-OFF **and** flag-ON both `e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86` (== base `e50521f3`)

## 1. The claim (BLOG_SPEC §576-590)

> "The c4 opcodes EQ, LT, GT, BZ, and BNZ can all be reduced to a single
> primitive: detecting whether a difference is zero."

The zero-detector `Z(d)` is three SiLU nodes sharing a scale, with a `+1/-2/+1`
second-difference (11 params). The spec then builds the rest from the **sign**
of the difference plus `EQ`:

```
EQ(a,b) = Z(a-b)          NE(a,b) = ¬Z(a-b)
LT(a,b) = sign(a-b) < 0   GT(a,b) = sign(a-b) > 0
LE(a,b) = LT ∨ EQ         GE(a,b) = GT ∨ EQ
```

All six opcodes therefore reduce to **one zero-detector + one sign primitive**.

## 2. Where the primitive already lives

In c4-release the zero-detector and sign are computed **per 4-bit nibble** by
the upstream L9 nibble comparator
(`wide_alu_dsl.nibble_compare_lane_rules`, task #411):

| CMP flag | primitive | meaning |
|----------|-----------|---------|
| `CMP+0` HI_LT | `sign(A.hi − B.hi) < 0` | high nibble less |
| `CMP+1` HI_EQ | `Z(A.hi − B.hi)`        | high nibble equal (zero-detector) |
| `CMP+2` LO_EQ | `Z(A.lo − B.lo)`        | low nibble equal (zero-detector) |
| `CMP+3` LO_LT | `sign(A.lo − B.lo) < 0` | low nibble less |

`mode="equal"` **is** the discrete zero-detector (the §510 one-hot form of the
`+1/-2/+1` bump); `mode="less_than"` **is** the sign primitive. The upstream L9
comparator was already derived (task #411). What was NOT derived was the
**combine** step: the six opcode decoders in L10 were still a **hand-authored
per-op default+override enumeration** (`vm_step.ComparisonCombine` and its two
declarative twins `_l10_comparison_combine_rules` /
`_layer10_alu_cmp_combine_rules`). That enumeration is the anti-pattern this
task removes.

## 3. The derivation

New DSL generator: `building_blocks_dsl.derived_comparison_rules`.

It builds the two **whole-operand combinators** ONCE, as products of the
per-nibble primitive flags:

```
A_EQ_B := HI_EQ ∧ LO_EQ                 (the §576 zero-detector, packed
                                         two nibbles at a time per §590)
A_LT_B := HI_LT ∨ (HI_EQ ∧ LO_LT)       (the §588 sign, lexicographic)
```

and then expresses **every** opcode as pure boolean algebra over
`{A_EQ_B, A_LT_B}` (`_cmp_result_terms`):

```
EQ = A_EQ_B                NE = ¬A_EQ_B
LT = A_LT_B                GE = ¬A_LT_B
LE = A_LT_B ∨ A_EQ_B       GT = ¬A_LT_B ∧ ¬A_EQ_B
```

There are **zero per-opcode magic constants**. The six results fall out of ONE
truth table over the two combinators, which fall out of ONE zero-detector + one
sign primitive. Lowering is the §510/§590 point-indicator + N-way-AND shape:
each boolean result is a **default write** (value when no override fires) plus
one **override** `multi_way_and_rule` per product term that flips the result. An
OR is several overrides driving the same result; a NOT flips which value is the
default; a two-flag term carries the `HI_LT` NOT-blocker (equality/lexicographic
terms cannot hold once the high nibble already decided `<` — the Shape-B
suppression). All amplitudes / thresholds / gates are **structural constants
shared by all six ops**, supplied once by the caller — not per-op tuning.

## 4. Result: byte-identical to the hand-authored model

The derivation reproduces BOTH live L10 cmp-combine banks **rule-for-rule** when
passed the golden structural constants:

- `tools/verify_derive_cmp.py` → `comparison_combine` OK (18/18), `alu_cmp_combine`
  OK (18/18), `RESULT: BYTE-IDENTICAL`.
- `tools/_isa_golden_hash.py` → **`e50521f3…` unchanged** under `C4_DERIVE_CMP=1`
  (whole-model state_dict SHA256, flag-OFF == flag-ON == base golden).

So this is a **correct-by-construction** replacement: the hand-authored per-op
enumeration and the single zero-detector derivation produce the *same weights*,
which is the strongest possible proof that the six opcodes really are one
primitive. `C4_DERIVE_CMP` is therefore a pure architecture/provenance flip, not
a behavior change — it swaps the SOURCE of the cmp-combine banks from 6× hand
enumeration to 1× spec derivation.

## 5. LOC + magic-constant collapse

- **Before:** three hand-authored per-op enumerations. Each spells out, per
  opcode, which CMP flags to AND and in which direction to flip — 18 explicit
  `cmp_default` / `cmp_override_2way` / `cmp_override_3way` calls **per bank**,
  duplicated across the two declarative banks (+ the imperative `vm_step`
  twin). The truth table is repeated 3×, and the `(hi_eq ∧ lo_lt)` /
  `(hi_eq ∧ lo_eq)` decomposition is re-derived by hand each time.
- **After:** ONE truth table (`_cmp_result_terms`, 6 short lines), ONE lowering
  (`derived_comparison_rules`), driven from the two combinator definitions
  (`_A_EQ_B_TERMS` / `_A_LT_B_TERMS`). Both live banks call one shared
  `_derived_cmp_combine_rules` helper differing only in structural kwargs (gate
  spelling, thresholds, blocker-in-override) — the DATA per path, zero per-op
  magic.
- **Magic constants:** the per-op "which flags → which result" magic is gone
  entirely; what remain are the shared structural amplitudes/thresholds
  (`±4/S` override, `2/S` default, marker/opcode gates, the `-50` MARK_PC and
  `-0.1` HI_LT blockers) which are the §510/§590 point-indicator + AND-threshold
  structure, not per-op tuning. These are supplied once and shared by all six.

The flag-OFF hand path is retained verbatim (byte-identical golden gate), so no
line is deleted yet; the derivation is proven a drop-in and the hand
enumeration can be removed in a follow-up flip once `C4_DERIVE_CMP` defaults ON.

## 6. Verdict (correctness + bonus fixes)

- **Correctness:** flag-ON is byte-identical to golden `e50521f3` at the
  state_dict level. Therefore every comparison verdict — on passing AND failing
  programs — is **exactly** the hand-authored verdict. No regression is possible
  (same weights ⇒ same forward ⇒ same decode).
- **Bonus fixes:** **none, by design.** Because the derived generator was tuned
  to reproduce the existing structural constants (including the campaign
  `cmp_gt_lo_lt_hieq_guard` 2.75 threshold and the Shape-B `-0.1` HI_LT
  blocker), it inherits exactly the current behavior — it does not *change* the
  equal-high-nibble margin or the if_var GT-FALSE path. The clean zero-detector
  is now the SINGLE place those behaviors are expressed, so a future margin fix
  (e.g. a cleaner `A_EQ_B` combinator or a sharper zero-detector bump) is now a
  **one-line change to the combinator**, applied uniformly to all six ops,
  instead of six hand edits. The derivation is the *enabler* for the margin fix,
  not the margin fix itself.

## 7. Honest limits

- **This is derivation-of-combine, not derivation-of-detector.** The zero-detector
  itself (the L9 nibble comparator) was already derived (#411); this task derives
  the L10 *combine* that turns its flags into opcode results. The two together
  now express the whole comparison family from one primitive, but the split
  across L9 (detector) and L10 (combine) is a codebase-topology fact, not a
  single fused block as the blog's minimal 11-param construction would suggest.
- **Discrete, not the smooth `+1/-2/+1` bump.** Per `building_blocks_dsl`'s
  module docstring, the codebase encodes nibble values as one-hot bands, so the
  zero-detector is realized as a one-hot equality indicator (`mode="equal"`),
  not the continuous 3-SiLU second difference. That is the byte-identical
  discrete reading the whole model already uses; a truly-continuous fp64
  all-nibbles-at-once detector (§590's aspiration) is out of scope.
- **The imperative `vm_step.ComparisonCombine` PureFFN is untouched.** It is only
  instantiated in `alu_mode="efficient"` (not the production `"lookup"` default),
  and the live decode goes through the declarative `_l10_comparison_combine_rules`
  which IS derived under the flag. The imperative twin remains as a legacy
  fixture; deriving it too is a mechanical follow-up.
- **Behavior-changing derivation deferred.** Because flag-ON is byte-identical,
  this task does NOT yet exercise the "clean zero-detector fixes a failing cmp
  program" lever. That is deliberate: it lands the correct-by-construction
  provenance first (zero risk to the corpus-critical cmp family), leaving the
  margin/leak fixes as separate, now-single-point, follow-ups.

## Files

- `neural_vm/unified_compiler/building_blocks_dsl.py` — new
  `derived_comparison_rules` generator + `_cmp_result_terms` / combinator
  constants (`_A_EQ_B_TERMS`, `_A_LT_B_TERMS`), exported in `__all__`.
- `neural_vm/unified_compiler/ops/shared.py` — `derive_cmp_enabled()`
  (`C4_DERIVE_CMP`, default-OFF).
- `neural_vm/unified_compiler/ops/l10_ops.py` — shared `_derived_cmp_combine_rules`
  helper + flag branch in `_l10_comparison_combine_rules` and
  `_layer10_alu_cmp_combine_rules`.
- `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — `C4_DERIVE_CMP`
  registered in both cache-key snapshots.
- `tools/verify_derive_cmp.py` — model-free byte-identity proof of both banks.
