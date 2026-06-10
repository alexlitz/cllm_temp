# Wave B — Cluster 5: scattered L3 + L6 cleanup + reconciliation

Migration plan for the 6 scattered position-role violations in L3
(4) and L6 (2). Unlike clusters 1-4, these are NOT all
``MARK_AX``-gated COMPUTE rules: the L3 entries are linter
*false-positive*-shaped TOKEN_EMIT rules that gate on ``HAS_SE``
without an obvious token-position marker, and the L6 entries are
``MARK_AX``-gated COMPUTE rules that fall outside the canonical ALU
families covered in cluster 2.

Refer to ``docs/STEP_END_MIGRATION_TEMPLATE.md`` for the L6 rows.
L3 rows require a different fix: either widen the scope to include
the actual ``BYTE_INDEX_1`` marker (already present in conditions but
not visible to the lint because it's referenced via ``_BYTE_INDEX[1]``
indirection), OR add a lint exemption for the ``_first_step_*`` family.

## Rules in cluster

| # | Source file (line) | Rule-factory function | Lint name skeleton | Verdict |
|---|---|---|---|---|
| 1 | ``l3_ops.py:258``  | ``_layer3_ffn_rules`` | ``layer3_ffn.sp_byte_1_first_step_lo`` | TOKEN_EMIT (lint FP: hides ``BYTE_INDEX_1`` behind ``_BYTE_INDEX[1]`` indirection) |
| 2 | ``l3_ops.py:265``  | ``_layer3_ffn_rules`` | ``layer3_ffn.sp_byte_1_first_step_hi`` | TOKEN_EMIT (lint FP) |
| 3 | ``l3_ops.py:303``  | ``_layer3_ffn_rules`` | ``layer3_ffn.bp_byte_1_first_step_lo`` | TOKEN_EMIT (lint FP) |
| 4 | ``l3_ops.py:310``  | ``_layer3_ffn_rules`` | ``layer3_ffn.bp_byte_1_first_step_hi`` | TOKEN_EMIT (lint FP) |
| 5 | ``l6_ops.py:1753`` | ``_layer6_alu_lo_clear_*`` (inside ``_layer6_tail_cleanup_rules`` or sibling) | ``l6_alu_lo_clear_*`` | COMPUTE — migrate normally |
| 6 | ``l6_ops.py:1760`` | ``_layer6_alu_hi_clear_*`` | ``l6_alu_hi_clear_*`` | COMPUTE — migrate normally |

The Wave B audit also lists scattered candidates in L2
(``_layer2_lookback_detection_head``), L4 (``_nibble_rotation_chain``,
``_pc_plus1_ax``, ``_pc_plus_offset_byte``, ``_sp_to_addr_key``), and
L14 (``_alu_high_byte_relay``, ``_addr_key_neural_decode_load_query``).
None of these are flagged by the current lint, but they should be
visited as the lint pattern is widened. They're documented here as
"deferred" so the dispatcher can pick them up in a follow-on cluster.

## `migrate_rule_to_step_end` calls

### L3 rows (1-4)

L3 rules don't migrate via ``migrate_rule_to_step_end``. The fix is
one of:

```python
# Option A: surface BYTE_INDEX_1 to the lint (rewrite the indexed
# reference to a literal string in the rule construction):
conditions=((f"H1+{_SP_I}", 1.0), ("BYTE_INDEX_1", 1.0),
            ("HAS_SE", -1.0)),  # was _BYTE_INDEX[1]

# Option B: add an exemption tag to the rule name (the lint reads
# name-prefix substrings, so a leading "tokenemit_" prefix opts out
# of the COMPUTE-pattern check). The dispatcher prefers Option A
# because it preserves grep-ability.
```

### L6 rows (5-6)

```python
from neural_vm.unified_compiler.step_end_migration import (
    migrate_rule_to_step_end,
)

_L6_ALU_CLEAR_RELAYED = (
    "ALU_LO", "ALU_HI",
    "OP_ADD", "OP_SUB", "OP_LEA", "OP_ADJ", "OP_ENT",
)

# Rows 5-6 (_layer6_alu_lo/hi_clear_*)
migrate_rule_to_step_end(rule, relayed_dims=_L6_ALU_CLEAR_RELAYED)
```

## Parity tests to check

L3 false-positives: confirm the byte-identity gate
(``compare_symbolic_to_lowered_ffn``) on ``make_layer3_ffn_op`` stays
green after either fix.

L6: add factories to ``RULE_FACTORIES``:

```python
def _factory_l6_alu_lo_clear():
    # Synthesize from the surrounding context in l6_ops.py:1753
    ...
```

Smoke gate:

```bash
python -m pytest c4_release/tests/test_smoke_ent.py -x
python -m pytest c4_release/tests/test_smoke_lev.py -x
python -m pytest c4_release/tests/test_1096_neural_declarative_diagnostic.py -x
```

## Xfails that flip on migration

None directly — no isolation tests cover L3 / L6 STEP_END behavior
yet. Add after migration if needed.

## Baseline decrement

After this cluster lands:

```python
"c4_release/neural_vm/unified_compiler/ops/l3_ops.py": 0,  # was 4
"c4_release/neural_vm/unified_compiler/ops/l6_ops.py": 0,  # was 2
```
