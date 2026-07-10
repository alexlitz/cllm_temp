# LAND DERIVATIONS — flip byte-identical spec-derivations DEFAULT-ON + delete subsumed enumeration

**Date:** 2026-07-09 · **Branch:** `land-derivations` (base `59a9de19`) ·
**Golden gate:** `e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86`
(== `e50521f3`) · flags-OFF `f725c06e`

## Mission

Realize the concrete `<5k-CODE` win from the BYTE-IDENTICAL spec-derivation
families. Each derivation produces weights byte-identical to the hand-authored
path (verified flag-ON == flag-OFF == `e50521f3`). The code collapse only
materializes when the derivation REPLACES the hand-authored enumeration. Done
safely, ONE family at a time, with the golden hash as the authoritative gate
for every step (merge → flip DEFAULT-ON → verify golden → delete subsumed
enumeration → re-verify golden + smoke → commit).

`derive-memory` (`C4_DERIVE_MEMORY`) is a pure consolidation umbrella flag with
no authoring-collapse to land, so it is out of scope for the deletion pass (see
§4).

## Result summary

| family  | merged | flipped DEFAULT-ON | golden held (flip) | golden held (after delete) | smoke | source-LOC deleted (net) |
|---------|--------|--------------------|--------------------|----------------------------|-------|--------------------------|
| bitwise | ✅ `0e45a17d` | ✅ `C4_DERIVE_BITWISE` 0→1 | ✅ `e50521f3` | ✅ `e50521f3` | 51/51 | wide_alu_dsl.py **−3 net** (−23/+20) |
| shift   | ✅ `af99fe3e` | ✅ `C4_DERIVE_SHIFT` 0→1  | ✅ `e50521f3` | ✅ `e50521f3` | 51/51 | l13_ops.py **−5 net** (−22/+17) |
| cmp     | ✅ `30125ef2` | ✅ `C4_DERIVE_CMP` 0→1    | ✅ `e50521f3` | ✅ `e50521f3` | 51/51 | l10_ops.py **−270 net** (−320/+50) |

Smoke was confirmed `51 passed, 1 deselected` (pytest exit 0) on the
fully-collapsed tree with ALL THREE families flipped-ON and their enumerations
deleted, and again on the bitwise + shift intermediate states. (The runs are
slow — ~40 min on the shared CPU under contention — but the golden state_dict is
byte-identical to the 51/51 baseline at every checkpoint, so an identical smoke
verdict is guaranteed by construction regardless.)

**Total authoring-LOC collapsed: −278 net source lines** (−365 deletions /
+87 insertions across the three enumeration files), the biggest single win being
the comparison family's per-op default+override enumeration (two banks).

Every step was gated on `tools/_isa_golden_hash.py --disk_cache=False ==
e50521f3`; the hash held at EVERY checkpoint (flip and post-delete for all three
families), so the collapse carries ZERO pass-rate risk.

---

## 1. Bitwise (OR / XOR / AND)

**Flag:** `C4_DERIVE_BITWISE` 0→1 (both cache-key snapshots +
`ops.shared.derive_bitwise_enabled` + the local `wide_alu_dsl` mirror).
**Golden held** at flip AND after deletion.

**Subsumed enumeration deleted** (`wide_alu_dsl.py`):
* `_BITWISE_OP_FN` — the dict of 3 enumerated Python bit operators
  (`operator.and_ / or_ / xor`).
* the `else: op_fn = _BITWISE_OP_FN[op]` branch in `bitwise_rules`.
* the now-unused `import operator`.

`bitwise_rules` now unconditionally uses `_bitwise_result_from_spec_formula`
(the BLOG_SPEC §568 one-formula `c_a·a + c_b·b + c_ab·a·b`, coefficient triples
`(1,1,-1)/(0,0,1)/(1,1,-2)`). Verified exhaustively: derived == operator on all
256 nibble pairs, 0 mismatches. The rule COUNT is width-locked (256/op ×
replication) so it does not shrink — the collapse is the qualitative one (3
per-op bit operators → 1 spec formula, zero magic constants). Net −3 source
lines. `C4_DERIVE_BITWISE=0` retained as a registered no-op kill-switch.

## 2. Shift (SHL / SHR)

**Flag:** `C4_DERIVE_SHIFT` 0→1 (both cache-key snapshots +
`ops.shared.derive_shift_enabled`). **Golden held** at flip AND after deletion.

**Subsumed enumeration deleted** (`ops/l13_ops.py`):
* `_layer13_shl_rules`: the `lambda v, s: (v << s) & 0xFF` hand-authored bit-op.
* `_layer13_shr_rules`: the `lambda v, s: (v >> s) & 0xFF` hand-authored bit-op.
* the `shift_fn = derived if flag else lambda` selection in both.

Both now pass `shl_result` / `shr_result` unconditionally (the BLOG_SPEC §599
"powers-of-two + mod-by-floor" recipe in `shift_semantics_dsl.py`). Verified
exhaustively: derived == bit-op on all 2048 `(v,s)` pairs each, 0 mismatches.
The 4096-unit lookup structure is unchanged. Net −5 source lines.
`C4_DERIVE_SHIFT=0` retained as a registered no-op kill-switch.

## 3. Comparison (EQ / NE / LT / GT / LE / GE) — the big collapse

**Flag:** `C4_DERIVE_CMP` 0→1 (both cache-key snapshots +
`ops.shared.derive_cmp_enabled`). **Golden held** at flip AND after deletion;
FFN unit count unchanged (42149 → 42149).

**Subsumed enumeration deleted** (`ops/l10_ops.py`, BOTH live banks):
* `_l10_comparison_combine_rules` (ComparisonCombine decode-row path): the
  nested `cmp_default` / `cmp_override_2way` / `cmp_override_3way` helpers +
  the 18-call per-op default+override `else` branch.
* `_layer10_alu_cmp_combine_rules` (L10-main ALU cmp lane): the nested
  `_cmp_default` / `_cmp_override_2way` / `_cmp_override_3way` helpers + the
  18-entry hand-enumerated `return (...)` tuple.

Both banks now call `_derived_cmp_combine_rules(...)` unconditionally, which
generates all 18 units per bank from the single BLOG_SPEC §576-590
zero-detector/sign derivation (`derived_comparison_rules`): two combinators
`A_EQ_B := HI_EQ ∧ LO_EQ` and `A_LT_B := HI_LT ∨ (HI_EQ ∧ LO_LT)` built once,
then every opcode is pure boolean algebra over them — ZERO per-op magic. The
per-path DATA (gate spelling, thresholds, whether overrides carry the MARK_PC
blocker, the campaign `_gt_gtge_3way_thresh` guard) is passed via kwargs and is
byte-identical to the corresponding hand path. Net **−270 source lines** — the
largest authoring collapse of the three families.

`tools/verify_derive_cmp.py` proves both banks 18/18 rule-for-rule
byte-identical to the derivation. The whole-model golden hash held `e50521f3`
under `C4_DERIVE_CMP=1` BEFORE any deletion (byte-identity corpus-wide), and
again AFTER deleting both enumerations — so the smoke verdict is identical to
the 51/51 golden baseline by construction (byte-identical model ⇒ identical
forward ⇒ identical decode on every program). `C4_DERIVE_CMP=0` retained as a
registered no-op kill-switch.

The imperative `vm_step.ComparisonCombine` PureFFN twin is **untouched** — it
is only instantiated in `alu_mode="efficient"` (a legacy fixture, NOT the
production `"lookup"` default); the live decode goes through the two declarative
banks, which are now the derivation. Deriving that legacy twin too is a
mechanical follow-up, out of scope here.

## 4. Memory (`C4_DERIVE_MEMORY`) — no authoring-collapse to land

`derive-memory` is a pure consolidation umbrella flag for the derived
binary-address-CAM memory-fix path (already collapsed). There is no hand-authored
per-op enumeration for the derivation to REPLACE, so there is nothing to delete
for the `<5k` LOC win. It is therefore excluded from the deletion pass; the
mission's three authoring-collapse families are bitwise, shift, and cmp.

## 5. Safety / provenance

* **Golden gate at every step.** `tools/_isa_golden_hash.py` (CPU,
  `disk_cache=False`) returned `e50521f3` at: the `land-derivations` baseline;
  each family's flip DEFAULT-ON; and each family's post-deletion state. The
  hash never moved.
* **Byte-identity is the strongest equivalence.** A byte-identical `state_dict`
  ⇒ the model is the same tensor-for-tensor ⇒ its verdict on EVERY program
  (corpus AND smoke) is identical. No pass-rate risk.
* **Kill-switches retained.** Each `C4_DERIVE_*=0` is now a registered no-op
  kill-switch (the enumeration it would restore is deleted). Kept only for
  cache-key isolation (so an off/on build never shares a serialised entry) and
  as a provenance marker; setting it changes nothing.
* **Reproduce.**
  ```
  export C4_VM_CACHE_DIR=/tmp/ld_$$    # isolated cache
  CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py            # e50521f3
  CUDA_VISIBLE_DEVICES="" python tools/verify_derive_cmp.py           # both banks 18/18 BYTE-IDENTICAL
  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_smoke.py        # 51 passed, 1 deselected
  ```

## 6. Bottom line

Three byte-identical spec-derivation families (bitwise, shift, cmp) are now the
DEFAULT-ON golden path, and their subsumed hand-authored enumerations have been
deleted: **−278 net source lines collapsed**, dominated by the comparison
family's per-op default+override enumeration (−270). The golden hash `e50521f3`
held at every checkpoint — this is concrete `<5k-CODE` progress with zero
pass-rate risk.
