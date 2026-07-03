# struct_/array_/global_/static_ cluster attribution (2026-06-07)

## TL;DR

**Neither the `struct_/array_` nor the `global_/static_` cluster exists
in the 1096 pure-neural corpus.** Both pytest filters select zero
tests. Attribution analysis cannot proceed against a non-existent
parametrisation. This doc captures the negative result so the next
agent does not re-run the same dead probe.

## Pattern collection

| Filter | Selected | Deselected | Result |
|---|--:|--:|---|
| `-k "struct_ or array_"` | 0 | 1098 | empty |
| `-k "global_ or static_"` | 0 | 1098 | empty |

Both runs terminate in ~0.22s with `1098 deselected in 0.22s`. No
`--runxfail` failures, no errors, no skips.

## Actual 1096 category inventory

`pytest --collect-only` on `tests/test_suite_1096_pure_neural_pytest.py`
yields these category prefixes (count of parametrised cases per
prefix):

| Prefix | Count |
|---|--:|
| `absdiff_` | 25 |
| `add_` | 50 |
| `bool_` | 25 |
| `div_` | 50 |
| `edge_` | 46 |
| `expr_` | 100 |
| `func_` | 150 |
| `gcd_` | 50 |
| `if_` | 100 |
| `loop_` | 100 |
| `mod_` | 50 |
| `mul_` | 50 |
| `nested_` | 50 |
| `rec_` | 100 |
| `sub_` | 50 |
| `var_` | 100 |
| (uncategorised / numeric-only) | 20 |
| **Total** | **1098** |

Case-insensitive `grep -i "struct|array|global|static"` against the
full collected-test list returns **0 matches**. The closeout plan
(`docs/CLOSEOUT_PLAN_2026_06_07.md`) confirms the four currently
attributed clusters are `func_add_*`, `expr_add_mul_*`, `var_*`, and
Shape-B `EQ_FALSE`/`NE_TRUE` — not struct/array/global/static.

## Where did the cluster names come from?

The brief almost certainly inherited terminology from a C-language
program-classification taxonomy (struct access, array access, global
variables, static storage) that was never instantiated as 1096 test
parametrisations. The 1096 corpus is generated from
`add/sub/mul/div/mod`-style arithmetic + control-flow templates; it
does not exercise the C type system at the lexical-category level the
brief assumes.

## Cluster 1 — struct_/array_

**Pass count:** N/A (0 tests selected).

**Representative failure:** none — filter is empty.

**Root surface:** undefined. No evidence in either the test corpus or
the attribution doc index that struct- or array-shaped programs are
gated by a distinct code surface separate from `var_*` (variable
read/write through stack/memory dim families).

**If interpreted as a proxy for stack-indexed loads/stores:** the
nearest analog in the 1096 corpus is `var_*` (100 tests, 0 passing per
closeout plan Wave C2). Root surface there is L3 `SP_byte2`
OUTPUT_LO[1]-vs-[0] divergence at block=3 gen=13, with the
`_rewrite_initial_sp_marker_to_f8` op dormant
(`var_failure_mode_shifted` memory note). The `_set_layerN_*` SI/LI
roundtrip surface (Wave S1 in closeout) is the structural-memory
analog at the smoke level.

## Cluster 2 — global_/static_

**Pass count:** N/A (0 tests selected).

**Representative failure:** none — filter is empty.

**Root surface:** undefined. C globals and statics are stored in the
data segment and addressed through LEA + LI/SI; the 1096 corpus does
not have a distinct category for these. If a future agent adds the
parametrisation, the predicted root surface is the same Wave S1 memory
cluster (L10 PSH `STACK0_BYTE_VAL_h` propagation) plus L8 SP gather
audit (`docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md`).

## Dependencies on in-flight work

- **A3 (in-flight):** no dependency. A3 (per recent attribution-doc
  index) operates on `var_*` / JSR-step-0 `MEM_value1`. Even if
  struct/array were present, they'd share the L10 PSH surface and
  inherit A3's outcome — but with zero selected tests this is moot.
- **Wave 2 F1-F3:** no dependency. F1-F3 (closeout C1 `func_add`, C2
  `var`, C3 EQ Shape-B) target categories that exist. None of them
  would change the struct/array/global/static selection count from 0.

## Smoke

51-test smoke run: **45 passed, 6 failed** (memory + LEA per closeout
Wave S1/S2). Threshold ≥45 met. Failures are the same expected six:
`test_lea_basic`, `test_si_li_roundtrip`, `test_sc_lc_roundtrip`,
`test_si_li_multiple_stores`, `test_si_li_overwrite`,
`test_si_li_16bit_value`. No new regressions introduced by this
doc-only commit.

## Recommendation

Replace the brief's struct/array/global/static cluster names with the
actual closeout-plan attribution waves (C1 `func_add`, C2 `var`, C3 EQ
Shape-B, C4 `expr_add_mul`) before dispatching the next attribution
agent. If C-type-system coverage is desired, it must be added as a
1096 parametrisation in a separate task.
