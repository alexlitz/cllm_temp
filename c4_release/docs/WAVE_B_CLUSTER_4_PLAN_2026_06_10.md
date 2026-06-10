# Wave B — Cluster 4: L11 / L12 MUL partial + combine

Migration plan for the 2 L11/L12 position-role violations enumerated by
`tools/lint_position_role.py` (commit 199479f4). Both entries are
``COMPUTE`` rules that currently gate on ``MARK_AX`` and must move to
``MARK_SE_ONLY``.

Refer to ``docs/STEP_END_MIGRATION_TEMPLATE.md`` for the per-row 6-step
recipe.

## Rules in cluster

| # | Source file (line) | Rule-factory function | Lint name skeleton |
|---|---|---|---|
| 1 | ``l11_ops.py:145`` | ``_layer11_mul_partial_rules_for_a_lo`` (called by ``_layer11_mul_partial_rules``) | ``l11_mul_partial_a*_b*_h*`` |
| 2 | ``l12_ops.py:96``  | ``_layer12_mul_combine_rules`` | ``l12_mul_combine_p*_ah*_bl*`` |

The audit also lists L13 (``_layer13_shifts_substage_rules``,
``_shl_rules``, ``_shr_rules``, ``_shifts_rules``) as Wave B
candidates, but their current rule-names don't include any of the
``_cmp_`` / ``_alu_`` / ``_mul_`` / etc. substrings the lint pattern
matches, so they're under-counted. Track separately and add to a
follow-up cluster once the lint pattern is widened.

## `migrate_rule_to_step_end` calls

```python
from neural_vm.unified_compiler.step_end_migration import (
    migrate_rule_to_step_end,
)

_L11_MUL_PARTIAL_RELAYED = (
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "MUL_PARTIAL_LO", "MUL_PARTIAL_HI",
    "OP_MUL",
)
_L12_MUL_COMBINE_RELAYED = (
    "MUL_PARTIAL_LO", "MUL_PARTIAL_HI",
    "MUL_ACCUM_LO", "MUL_ACCUM_HI",
    "OP_MUL",
)

# Row 1 (_layer11_mul_partial_rules)
migrate_rule_to_step_end(rule, relayed_dims=_L11_MUL_PARTIAL_RELAYED,
                         allow_step_end_writes_to=("MUL_PARTIAL_LO",
                                                    "MUL_PARTIAL_HI"))

# Row 2 (_layer12_mul_combine_rules)
migrate_rule_to_step_end(rule, relayed_dims=_L12_MUL_COMBINE_RELAYED,
                         allow_step_end_writes_to=("MUL_ACCUM_LO",
                                                    "MUL_ACCUM_HI"))
```

## Parity tests to check

Not yet in ``RULE_FACTORIES``. Add wrappers:

```python
def _factory_l11_mul_partial():
    from neural_vm.unified_compiler.ops.l11_ops import _layer11_mul_partial_rules
    return _layer11_mul_partial_rules(100.0)
def _factory_l12_mul_combine():
    from neural_vm.unified_compiler.ops.l12_ops import _layer12_mul_combine_rules
    return _layer12_mul_combine_rules(100.0)
```

Smoke gate:

```bash
python -m pytest c4_release/tests/test_smoke_mul.py -x
```

## Xfails that flip on migration

No isolation tests exist for L11/L12 in
``test_step_end_compute_isolated.py`` yet. Add them after migration
(mirror the L10 ``_layer10_alu_cmp_combine_rules`` test stanza).

## Baseline decrement

After this cluster lands:

```python
"c4_release/neural_vm/unified_compiler/ops/l11_ops.py": 0,  # was 1
"c4_release/neural_vm/unified_compiler/ops/l12_ops.py": 0,  # was 1
```
