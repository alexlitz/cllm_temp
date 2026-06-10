# STEP_END migration template -- Wave B recipe

Canonical 6-step recipe for migrating one compute rule (or one
attention head's anchor row) from `MARK_AX` gating to `MARK_SE_ONLY`
gating. Pairs with
[`STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md`](STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md)
(architectural design) and uses
`c4_release/neural_vm/unified_compiler/step_end_migration.py`
(helper module).

This template is intentionally short. Each Wave B agent should be
able to read it once and produce a single commit that migrates one
rule from the Wave B table in the architecture doc.

## 0. Preconditions

Before starting any migration, confirm:

1. **Wave A `step_end_operand_relay` head has landed.** Verify by
   running `c4_release/tools/probe_step_end_completeness.py`. The
   probe must report `max|OP_<NAME>|` and `max|AX_CARRY_LO|` at
   `MARK_SE` >= 0.9 for the opcode you're migrating. If the probe
   still shows "**dispatch flag DEAD at SE**", stop -- Wave A is
   not yet wired in for this slot.
2. **The byte-identity gate is green on the source rule today.**
   Run `compare_symbolic_to_lowered_ffn` on the source op's
   `compiler_ir` and confirm no `lowering` or
   `weight_output_mismatch` issues. If the source op is already
   drifting, fix that first -- don't migrate a broken rule.
3. **You have read the architecture doc.** Particularly section 4's
   Wave C list ("rules that STAY at MARK_AX") -- if your candidate
   appears there, it's not eligible.

## 1. Identify the rule

Pick one row from the Wave B table in
`STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md` section 4. The
canonical order is row #1 (`_layer10_alu_cmp_combine_rules`
overrides) -> #2 -> ... but you may migrate any row whose Wave A
relay precondition is met.

Open the source file (e.g. `c4_release/neural_vm/unified_compiler/ops/l10_ops.py`)
and locate the rule factory. Each rule factory in the table is a
function that returns a tuple of `FFNRule`s. The function body is
where you'll edit.

## 2. Verify the relay broadcasts the needed input dims

For every condition in the rule, write down whether its dim is:

* a **marker** (`MARK_AX`, `MARK_PC`, ...) -- handled by
  `migrate_rule_to_step_end` automatically;
* an **operand slot** that lives at MARK_AX today (`OP_<NAME>`,
  `AX_CARRY_LO`, `AX_CARRY_HI`, `ALU_LO`, `ALU_HI`, `CMP`,
  `STACK0_BYTE*`) -- must be in Wave A's broadcast list;
* a **global** (`CONST`, `HAS_SE`, `IS_BYTE`) -- already visible at
  every row, no work needed.

If any operand slot is NOT in the Wave A broadcast list, stop. File
a follow-up to extend Wave A first; don't try to migrate without
the relay.

## 3. Use `migrate_rule_to_step_end()`

The helper is a one-call rewrite. Inside the rule factory, swap:

```python
def _layer10_alu_bitwise_or_rules(S: float) -> tuple[FFNRule, ...]:
    ...
    return _layer10_alu_bitwise_rules(S, op_name="OR", op_fn=operator.or_)
```

for:

```python
from c4_release.neural_vm.unified_compiler.step_end_migration import (
    migrate_rule_to_step_end,
)

def _layer10_alu_bitwise_or_rules(S: float) -> tuple[FFNRule, ...]:
    source = _layer10_alu_bitwise_rules(
        S, op_name="OR", op_fn=operator.or_,
    )
    return tuple(
        migrate_rule_to_step_end(
            rule,
            relayed_dims=("ALU_LO", "ALU_HI", "AX_CARRY_LO",
                          "AX_CARRY_HI", "opcode_flag_OR"),
        )
        for rule in source
    )
```

The helper:

* rewrites each `MARK_AX` condition (and `MARK_AX` gate-terms) to
  `MARK_SE_ONLY` at the same weight and offset;
* updates the rule's `scope` annotation via word-boundary
  substitution (so `MARK_AX_CARRY_HI` stays intact);
* appends `_step_end` to the rule's `name` for verifier
  attribution;
* runs `assert_migration_safe` first and refuses to migrate a rule
  that writes byte-emission slots (`OUTPUT_LO` / `OUTPUT_HI`) or
  row-local marker dims.

If `assert_migration_safe` raises `MigrationSafetyError`, the rule
is mis-classified. Either reclassify as Wave C (stays at MARK_AX) or
split the rule into a STEP_END part (the compute) and a per-byte
part (the byte write).

## 4. Run the parity test

The parallel byte-identity framework (built by the
`test_step_end_migration_parity.py` agent) exposes:

```bash
python -m pytest c4_release/tests/test_step_end_migration_parity.py \
    -k <rule_name_prefix> -x
```

The parity test confirms that, under the Wave A relay, the migrated
rule fires at the same effective `(query_pos, byte_value)` tuples
as the source rule did at MARK_AX. The test compares symbolic
forward of both versions on a fixed seed corpus.

If the parity test fails:

* check the Wave A probe again (it may have regressed);
* check that you included every operand condition's dim in
  `relayed_dims`;
* check `decl_verifier.verify_rule_scopes` -- the new scope may
  have stale predicate references.

## 5. Smoke check

Run the narrow opcode smoke that exercises this rule, plus the
1096 corpus sentinel:

```bash
python -m pytest c4_release/tests/test_smoke_<opcode>.py -x
python -m pytest c4_release/tests/test_1096_neural_declarative_diagnostic.py -x
```

The sentinel baseline depends on env flags (see memory note
`project_1096_sentinel_baseline.md`); the migration must hold the
baseline at the same env-flag setting, not against a historical
absolute number.

## 6. Commit

Single commit, conventional message:

```
refactor(l10): migrate bitwise_or rules from MARK_AX to STEP_END

Wave B row #3. Wave A relay broadcasts ALU_LO/HI, AX_CARRY_LO/HI,
opcode_flag_OR into MARK_SE_ONLY; rule semantics unchanged. Parity
test green; 1096 sentinel unchanged at +ALU_DECL_ONLY=1.
```

No piggyback fixes. One row per commit -- if a parity test fails
mid-row, fix that row before opening another.

## Example walkthrough -- `_layer10_alu_bitwise_or_rules`

This is the simplest Wave B row (#3 in the table). It's a 512-unit
3-way AND across `(MARK_AX, ALU_LO[a], AX_CARRY_LO[b])` gated on
`opcode_flag_OR`. No CMP, no MARK_PC blocker, no byte writes -- the
result lands in `OUTPUT_LO` and `OUTPUT_HI_THIS_STEP`.

**Wait** -- `OUTPUT_LO` and `OUTPUT_HI_THIS_STEP` ARE byte-emission
slots. `assert_migration_safe` will reject the raw rule. This is
the expected case for any "compute that ends in a byte write" path:
the migration is NOT a flat MARK_AX -> MARK_SE swap.

The right shape for the migration is:

1. Add a new intermediate slot (`OR_RESULT_LO` / `OR_RESULT_HI`)
   to `dim_registry_dynamic.py` and the appropriate allocator.
2. Migrate the compute (3-way AND) to STEP_END gating on
   `MARK_SE_ONLY`, writing the intermediate slot.
3. Add a one-row-per-byte relay (or per-byte FFN write) that lifts
   `OR_RESULT_*` into `OUTPUT_*` at the byte rows where they're
   read.

This pattern is general: any compute whose target is a byte-emit
slot must split into (compute @ SE) + (relay/copy @ byte rows).
The `allow_step_end_writes_to=` arg on `migrate_rule_to_step_end`
exists for the very narrow case where a rule already writes into a
"byte-shaped" slot that turns out to be safe at SE -- not for the
common case above.

For a fully-trivial migration (no byte-emit writes), see Wave B
row #1 (`_layer10_alu_cmp_combine_rules` overrides) once it's
unblocked -- its write target is the `CMP` slot, which IS in the
Wave A broadcast list, so the migration is a single
`migrate_rule_to_step_end` call.

## Helper module quick reference

`c4_release/neural_vm/unified_compiler/step_end_migration.py` exposes:

| Symbol                                  | Purpose                                                                  |
|-----------------------------------------|--------------------------------------------------------------------------|
| `migrate_rule_to_step_end(rule, ...)`   | Copy an `FFNRule` with `MARK_AX` conditions rewritten to `MARK_SE_ONLY`. |
| `migrate_attention_head_to_step_end(...)` | Same idea for a `DeclarativeAttentionHeadSpec`'s Q and K writes.       |
| `assert_migration_safe(rule, ...)`      | Conservative check: rejects byte-emission and row-local marker writes.   |
| `MigrationSafetyError`                  | Raised by `assert_migration_safe`. Subclass of `ValueError`.             |
| `MARK_AX_NAME` / `MARK_SE_NAME`         | The string constants used in conditions / scopes.                        |

Tests: `c4_release/tests/test_step_end_migration.py` (13 unit
tests covering the three helpers and the safety predicates).
