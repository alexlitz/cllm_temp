# Wave B — Cluster 1: L10 compute family

Migration plan for the 15 L10 position-role violations enumerated by
`tools/lint_position_role.py` (commit 199479f4). All entries here are
``COMPUTE`` rules that currently gate on ``MARK_AX`` and must move to
``MARK_SE_ONLY`` once Wave A's ``step_end_operand_relay`` head lands.

Refer to ``docs/STEP_END_MIGRATION_TEMPLATE.md`` for the per-row 6-step
recipe. This cluster collects 15 rows; the dispatcher
(``tools/wave_b_dispatcher.py``) chases them in the order below.

## Rules in cluster

| # | Source file (line) | Rule-factory function | Lint name skeleton |
|---|---|---|---|
| 1 | ``l10_ops.py:544`` | ``_l10_comparison_combine_rules`` | ``l10_cmp_default_*_*`` |
| 2 | ``l10_ops.py:565`` | ``_l10_comparison_combine_rules`` | ``l10_cmp_override2_*_*`` |
| 3 | ``l10_ops.py:606`` | ``_l10_comparison_combine_rules`` | ``l10_cmp_override3_*_*`` |
| 4 | ``l10_ops.py:715`` | ``_layer10_alu_cmp_combine_rules`` | ``l10_cmp_*_default`` |
| 5 | ``l10_ops.py:740`` | ``_layer10_alu_cmp_combine_rules`` | ``l10_cmp_*_override2_*`` |
| 6 | ``l10_ops.py:773`` | ``_layer10_alu_cmp_combine_rules`` | ``l10_cmp_*_override3_*`` |
| 7 | ``l10_ops.py:850`` | ``_layer10_alu_bitwise_rules`` (called by ``_or``/``_xor``/``_and``) | ``l10_bitwise_*_*_a*_b*`` |
| 8 | ``l10_ops.py:917`` | ``_layer10_alu_shl_shr_zero_rules`` | ``l10_*_shift_ge16_zero`` |
| 9 | ``l10_ops.py:1049`` | ``_layer10_alu_mul_lo_rules`` | ``l10_mul_lo_a*_b*`` |
| 10 | ``l10_ops.py:7137`` | ``_tail_bit32_result_correction_rules`` (NE branch) | ``tail_cmp_ne_true_01`` |
| 11 | ``l10_ops.py:7149`` | ``_tail_bit32_result_correction_rules`` (EQ branch) | ``tail_cmp_eq_false_00`` |
| 12 | ``l10_ops.py:7162`` | ``_tail_bit32_result_correction_rules`` (LE LT branch) | ``tail_cmp_le_lt_true_01`` |
| 13 | ``l10_ops.py:7175`` | ``_tail_bit32_result_correction_rules`` (LE EQ prefix) | ``tail_cmp_le_eq_prefix_false_00`` |
| 14 | ``l10_ops.py:7203`` | ``_tail_bit32_result_correction_rules`` (LT branch) | ``tail_cmp_lt_false_00`` |
| 15 | ``l10_ops.py:7217`` | ``_tail_bit32_result_correction_rules`` (GT branch) | ``tail_cmp_gt_false_00`` |

## `migrate_rule_to_step_end` calls

The dispatcher wraps each rule factory's return tuple with
``migrate_rule_to_step_end`` and passes the operand dims that Wave A
relays. The canonical relayed-dim list for L10:

```python
from neural_vm.unified_compiler.step_end_migration import (
    migrate_rule_to_step_end,
)

_L10_CMP_RELAYED = (
    "CMP",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)
_L10_BITWISE_RELAYED = (
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "OP_OR", "OP_XOR", "OP_AND",
)
_L10_SHIFT_RELAYED = ("ALU_LO", "ALU_HI", "OP_SHL", "OP_SHR")
_L10_MUL_LO_RELAYED = ("ALU_LO", "ALU_HI", "AX_CARRY_LO", "OP_MUL")
_L10_TAIL_CMP_RELAYED = (
    "CMP", "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)
```

Per-rule invocation (one example each; the dispatcher iterates):

```python
# Row 1-3 (_l10_comparison_combine_rules)
migrate_rule_to_step_end(rule, relayed_dims=_L10_CMP_RELAYED)

# Row 4-6 (_layer10_alu_cmp_combine_rules)
migrate_rule_to_step_end(rule, relayed_dims=_L10_CMP_RELAYED)

# Row 7 (_layer10_alu_bitwise_or/xor/and_rules)
migrate_rule_to_step_end(rule, relayed_dims=_L10_BITWISE_RELAYED,
                         allow_step_end_writes_to=("OUTPUT_LO",
                                                    "OUTPUT_HI_THIS_STEP"))
# NOTE: bitwise rules write OUTPUT_LO+0..15 directly; the safety check
# rejects them as-is. Either (a) split into intermediate slot then
# per-byte relay (see STEP_END_MIGRATION_TEMPLATE.md section 6 example),
# or (b) pass allow_step_end_writes_to=. Default plan: split.

# Row 8 (_layer10_alu_shl_shr_zero_rules)
migrate_rule_to_step_end(rule, relayed_dims=_L10_SHIFT_RELAYED,
                         allow_step_end_writes_to=("OUTPUT_LO",
                                                    "OUTPUT_HI"))

# Row 9 (_layer10_alu_mul_lo_rules)
migrate_rule_to_step_end(rule, relayed_dims=_L10_MUL_LO_RELAYED,
                         allow_step_end_writes_to=("OUTPUT_LO",))

# Row 10-15 (_tail_bit32_result_correction_rules tail_cmp_* branch)
migrate_rule_to_step_end(rule, relayed_dims=_L10_TAIL_CMP_RELAYED)
```

## Parity tests to check

```bash
python -m pytest c4_release/tests/test_step_end_migration_parity.py \
    -k "cmp_combine or cmp_default or bitwise_or or bitwise_xor or bitwise_and" -x
```

The five existing factories cover rows 1-7; rows 8-15 need factories
added at the end of ``RULE_FACTORIES`` in the parity test before they
can be checked symbolically. Smoke gate:

```bash
python -m pytest c4_release/tests/test_l10_tail_correction.py -x
python -m pytest c4_release/tests/test_smoke_cmp.py -x
python -m pytest c4_release/tests/test_smoke_bitwise.py -x
```

## Xfails that flip on migration

In ``c4_release/tests/test_step_end_compute_isolated.py`` (commit
d20960e2):

| Row # | xfail that becomes XPASS |
|---|---|
| 1 | ``test_l10_comparison_combine_eq_default_at_step_end`` |
| 1 | ``test_l10_comparison_combine_ge_default_at_step_end`` |
| 4 | ``test_l10_alu_cmp_combine_eq_override_at_step_end`` |
| 5 | ``test_l10_alu_cmp_combine_ne_override_at_step_end`` |
| 6 | ``test_l10_alu_cmp_combine_lt_override_at_step_end`` |
| 7 (OR) | ``test_l10_alu_bitwise_or_at_step_end`` |
| 7 (XOR) | ``test_l10_alu_bitwise_xor_at_step_end`` |
| 7 (AND) | ``test_l10_alu_bitwise_and_at_step_end`` |
| 8 (SHL) | ``test_l10_alu_shl_zero_at_step_end`` |
| 8 (SHR) | ``test_l10_alu_shr_zero_at_step_end`` |

Rows 9-15 do not have isolation tests yet; add them as the dispatcher
flips each row.

## Baseline decrement

After this cluster lands, in ``_BASELINE`` of
``tools/lint_position_role.py``:

```python
"c4_release/neural_vm/unified_compiler/ops/l10_ops.py": 0,  # was 15
```
