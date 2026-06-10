# Wave B — Cluster 2: L8 ALU stages + LEV + sentinel

Migration plan for the 14 L8 position-role violations enumerated by
`tools/lint_position_role.py` (commit 199479f4). All entries are
``COMPUTE`` rules that currently gate on ``MARK_AX`` (or ``MARK_BP`` for
LEV byte rules) and must move to ``MARK_SE_ONLY`` once Wave A's
operand relay lands.

Refer to ``docs/STEP_END_MIGRATION_TEMPLATE.md`` for the per-row 6-step
recipe.

## Rules in cluster

| # | Source file (line) | Rule-factory function | Lint name skeleton |
|---|---|---|---|
| 1 | ``l8_ops.py:387`` | ``_layer8_alu_add_lo_rules`` | ``l8_alu_add_lo_a*_b*`` |
| 2 | ``l8_ops.py:425`` | ``_layer8_alu_lea_lo_rules`` | ``l8_alu_lea_lo_a*_b*`` |
| 3 | ``l8_ops.py:459`` | ``_layer8_alu_sub_lo_rules`` | ``l8_alu_sub_lo_a*_b*`` |
| 4 | ``l8_ops.py:503`` | ``_layer8_alu_add_carry_rules`` | ``l8_alu_add_carry_a*_b*`` |
| 5 | ``l8_ops.py:542`` | ``_layer8_alu_lea_carry_rules`` | ``l8_alu_lea_carry_a*_b*`` |
| 6 | ``l8_ops.py:577`` | ``_layer8_alu_adj_lo_rules`` | ``l8_alu_adj_lo_a*_b*`` |
| 7 | ``l8_ops.py:616`` | ``_layer8_alu_adj_carry_rules`` | ``l8_alu_adj_carry_a*_b*`` |
| 8 | ``l8_ops.py:653`` | ``_layer8_alu_sub_borrow_rules`` | ``l8_alu_sub_borrow_a*_b*`` |
| 9 | ``l8_ops.py:688`` | ``_layer8_alu_ent_lo_rules`` | ``l8_alu_ent_lo_sp*_imm*`` |
| 10 | ``l8_ops.py:728`` | ``_layer8_alu_ent_borrow_rules`` | ``l8_alu_ent_borrow_sp*_imm*`` |
| 11 | ``l8_ops.py:755`` | ``_layer8_alu_cmp_group_rules`` | ``l8_alu_cmp_group`` |
| 12 | ``l8_ops.py:803`` | ``_layer8_alu_cmp_clear_rules`` | ``l8_alu_cmp_clear_k*`` |
| 13 | ``l8_ops.py:927`` | ``_layer8_alu_lev_b1_rules`` (gates on ``MARK_BP``) | ``l8_alu_lev_b1`` |
| 14 | ``l8_ops.py:952`` | ``_layer8_alu_lev_b2_rules`` (gates on ``MARK_BP``) | ``l8_alu_lev_b2`` |

## `migrate_rule_to_step_end` calls

```python
from neural_vm.unified_compiler.step_end_migration import (
    migrate_rule_to_step_end,
)

_L8_ALU_LO_RELAYED = (
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "OP_ADD", "OP_SUB", "OP_LEA", "OP_ADJ", "OP_ENT",
)
_L8_ALU_CMP_RELAYED = (
    "CMP",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)
_L8_LEV_RELAYED = (
    "ALU_LO", "ALU_HI",
    "BP_FRAME_BYTE0", "BP_FRAME_BYTE1", "BP_FRAME_BYTE2",
    "OP_LEV",
)

# Rows 1-10 (ALU lo-byte + carry stages)
migrate_rule_to_step_end(rule, relayed_dims=_L8_ALU_LO_RELAYED)

# Rows 11-12 (CMP group decode + clear)
migrate_rule_to_step_end(rule, relayed_dims=_L8_ALU_CMP_RELAYED)

# Rows 13-14 (LEV b1/b2 - SPECIAL: source gates on MARK_BP, not MARK_AX)
# These need a custom migration: rewrite MARK_BP -> MARK_SE_ONLY,
# NOT MARK_AX. Use migrate_rule_to_step_end(..., from_marker="MARK_BP")
# (the helper currently only swaps MARK_AX; LEV rows need a one-line
# helper extension OR a manual condition swap. The dispatcher logs
# these as "needs from_marker= extension".)
migrate_rule_to_step_end(rule, relayed_dims=_L8_LEV_RELAYED)
```

## Parity tests to check

These factories are NOT yet in
``test_step_end_migration_parity.py:RULE_FACTORIES``. Add wrappers
for each new factory before flipping:

```python
def _factory_l8_alu_add_lo():
    from neural_vm.unified_compiler.ops.l8_ops import _layer8_alu_add_lo_rules
    return _layer8_alu_add_lo_rules(100.0)
# ... and one per row 1-14
```

Smoke gate:

```bash
python -m pytest c4_release/tests/test_smoke_add.py -x
python -m pytest c4_release/tests/test_smoke_sub.py -x
python -m pytest c4_release/tests/test_smoke_cmp.py -x
python -m pytest c4_release/tests/test_smoke_lev.py -x
python -m pytest c4_release/tests/test_smoke_ent.py -x
```

## Xfails that flip on migration

None of these rules currently have isolation tests in
``test_step_end_compute_isolated.py``. Add them as the dispatcher
flips each row (mirror the L9/L10 patterns at the bottom of the file).

## Baseline decrement

After this cluster lands:

```python
"c4_release/neural_vm/unified_compiler/ops/l8_ops.py": 0,  # was 14
```
