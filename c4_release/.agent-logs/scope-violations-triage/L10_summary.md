# L10 scope_violations triage

Op: `make_tail_bit32_result_correction_op` — total scope_violations: **255**

## Category totals

| Category | Count |
|---|---:|
| A — scope-annotation error | 0 |
| B — genuine E5 bug | 0 |
| C — imprecise semantics | 0 |
| D — unsatisfiable effective | 255 |

## Rule families per category

### Category D
- `tail_stack0_store_loaded_byte_XX`: 255

## Top examples per category

### Category D

- **rule**: `tail_stack0_store_loaded_byte_10`
  - scope: `mark == STACK0`
  - effective markers: pos=['AX', 'MEM'] neg=['AX', 'BP', 'MEM', 'PC', 'SP']
  - name markers: ['STACK0']
  - effective_satisfiable: False
  - why: effective predicate is unsatisfiable per S-3
  - verifier reason: disjunct {NOT is_byte AND NOT mark == AX AND NOT mark == BP AND NOT mark == MEM AND NOT mark == PC AND NOT mark == SP AND has_se AND is_byte AND mark == AX AND mark == MEM AND opcode_in_step in {PSH, SC, SI}} of p does not entail any disjun
- **rule**: `tail_stack0_store_loaded_byte_20`
  - scope: `mark == STACK0`
  - effective markers: pos=['AX', 'MEM'] neg=['AX', 'BP', 'MEM', 'PC', 'SP']
  - name markers: ['STACK0']
  - effective_satisfiable: False
  - why: effective predicate is unsatisfiable per S-3
  - verifier reason: disjunct {NOT is_byte AND NOT mark == AX AND NOT mark == BP AND NOT mark == MEM AND NOT mark == PC AND NOT mark == SP AND has_se AND is_byte AND mark == AX AND mark == MEM AND opcode_in_step in {PSH, SC, SI}} of p does not entail any disjun
- **rule**: `tail_stack0_store_loaded_byte_30`
  - scope: `mark == STACK0`
  - effective markers: pos=['AX', 'MEM'] neg=['AX', 'BP', 'MEM', 'PC', 'SP']
  - name markers: ['STACK0']
  - effective_satisfiable: False
  - why: effective predicate is unsatisfiable per S-3
  - verifier reason: disjunct {NOT is_byte AND NOT mark == AX AND NOT mark == BP AND NOT mark == MEM AND NOT mark == PC AND NOT mark == SP AND has_se AND is_byte AND mark == AX AND mark == MEM AND opcode_in_step in {PSH, SC, SI}} of p does not entail any disjun
- **rule**: `tail_stack0_store_loaded_byte_40`
  - scope: `mark == STACK0`
  - effective markers: pos=['AX', 'MEM'] neg=['AX', 'BP', 'MEM', 'PC', 'SP']
  - name markers: ['STACK0']
  - effective_satisfiable: False
  - why: effective predicate is unsatisfiable per S-3
  - verifier reason: disjunct {NOT is_byte AND NOT mark == AX AND NOT mark == BP AND NOT mark == MEM AND NOT mark == PC AND NOT mark == SP AND has_se AND is_byte AND mark == AX AND mark == MEM AND opcode_in_step in {PSH, SC, SI}} of p does not entail any disjun
- **rule**: `tail_stack0_store_loaded_byte_50`
  - scope: `mark == STACK0`
  - effective markers: pos=['AX', 'MEM'] neg=['AX', 'BP', 'MEM', 'PC', 'SP']
  - name markers: ['STACK0']
  - effective_satisfiable: False
  - why: effective predicate is unsatisfiable per S-3
  - verifier reason: disjunct {NOT is_byte AND NOT mark == AX AND NOT mark == BP AND NOT mark == MEM AND NOT mark == PC AND NOT mark == SP AND has_se AND is_byte AND mark == AX AND mark == MEM AND opcode_in_step in {PSH, SC, SI}} of p does not entail any disjun

## Recommended next step per category

- **A — scope-annotation error**: bulk-fix by scanning each rule's effective marker and replacing the declared `scope=` with the matching marker. Safe automated rewrite once the name/effective agreement is confirmed.
- **B — genuine E5 bug**: surgical per-rule review. The rule's conditions admit positions outside intent; tighten the conditions (add a marker term, narrow byte_index, etc.).
- **C — imprecise semantics**: tighten the registry semantics (`build_default_registry`) for the dims with placeholder `is_byte OR NOT is_byte` semantics so F-5 doesn't drop to vacuous tautologies.
- **D — unsatisfiable effective**: usually means F-5 is ignoring a gated_write `gate=` argument that supplies the true firing marker. Two complementary fixes: (1) teach F-5 to AND in the gate's semantics so the effective predicate is no longer contradictory; (2) revisit which conditions list `MARK_MEM, -1e6` as a hard blocker when the rule actually fires at MEM positions (the blocker is double-counted with the scope/gate). Either fix collapses the entire D bucket without per-rule edits.

**Biggest bucket:** D_unsatisfiable (255/255). See the corresponding recommendation above.
