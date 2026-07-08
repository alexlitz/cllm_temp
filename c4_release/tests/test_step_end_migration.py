"""Unit tests for ``verification.step_end_migration`` helpers.

These exercise the scaffolding the Wave B migrations rely on without
touching any model weights or actually moving a production rule. See
``docs/STEP_END_MIGRATION_TEMPLATE.md`` for the end-to-end recipe and
``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md`` for the
architectural motivation.
"""

from __future__ import annotations

import pytest

from c4_release.neural_vm.unified_compiler.ir import FFNRule
from c4_release.neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
)
from c4_release.neural_vm.verification.step_end_migration import (
    MARK_AX_NAME,
    MARK_SE_NAME,
    MigrationSafetyError,
    assert_migration_safe,
    migrate_attention_head_to_step_end,
    migrate_rule_to_step_end,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _l10_bitwise_or_like_rule() -> FFNRule:
    """A miniature stand-in for one unit of
    ``_layer10_alu_bitwise_or_rules``: 3-way AND across ``MARK_AX``,
    ``ALU_LO[5]``, ``AX_CARRY_LO[3]``, gated on ``opcode_flag_OR``,
    writes ``TEMP+0`` (a non-byte slot picked so the safety check
    passes).
    """

    return FFNRule.gated_write(
        name="probe_bitwise_or_a5_b3",
        conditions=(
            ("MARK_AX", 40.0),
            ("ALU_LO+5", 30.0),
            ("AX_CARRY_LO+3", 30.0),
        ),
        threshold=80.0,
        gate="opcode_flag_OR",
        gate_weight=1.0,
        writes=(("TEMP+0", 2.0),),
        scope="MARK_AX and opcode_flag_OR",
    )


# ---------------------------------------------------------------------------
# migrate_rule_to_step_end
# ---------------------------------------------------------------------------


def test_migrate_rule_rewrites_mark_ax_condition_to_mark_se_only():
    rule = _l10_bitwise_or_like_rule()
    migrated = migrate_rule_to_step_end(rule)

    names = [c.dim.name for c in migrated.conditions]
    assert MARK_AX_NAME not in names
    assert MARK_SE_NAME in names
    # Operand conditions pass through unchanged.
    assert ("ALU_LO", 30.0) in [
        (c.dim.name, c.weight) for c in migrated.conditions
    ]
    assert ("AX_CARRY_LO", 30.0) in [
        (c.dim.name, c.weight) for c in migrated.conditions
    ]
    # Threshold, gate, writes are unchanged.
    assert migrated.threshold == 80.0
    assert migrated.gate is not None and migrated.gate.name == "opcode_flag_OR"
    assert [(w.dim.name, w.weight) for w in migrated.writes] == [
        ("TEMP", 2.0),
    ]


def test_migrate_rule_preserves_condition_weight_and_offset():
    rule = FFNRule.constant_write(
        name="probe_offsetted",
        conditions=(("MARK_AX+3", 7.5), ("CONST", 1.0)),
        threshold=8.0,
        writes=(("TEMP+1", 1.0),),
    )
    migrated = migrate_rule_to_step_end(rule)

    se_term = next(
        c for c in migrated.conditions if c.dim.name == MARK_SE_NAME
    )
    assert se_term.dim.offset == 3
    assert se_term.weight == 7.5


def test_migrate_rule_updates_scope_string_word_boundaries_only():
    rule = FFNRule.constant_write(
        name="probe_scope",
        conditions=(("MARK_AX", 1.0),),
        threshold=0.5,
        writes=(("TEMP+0", 1.0),),
        # ``MARK_AX_CARRY_HI`` is a fake dim name used here to confirm
        # we don't replace inside compound tokens.
        scope="MARK_AX and not MARK_PC and MARK_AX_CARRY_HI",
    )
    migrated = migrate_rule_to_step_end(rule)

    assert migrated.scope is not None
    assert "MARK_SE_ONLY" in migrated.scope
    # The compound token must NOT have been rewritten.
    assert "MARK_AX_CARRY_HI" in migrated.scope
    # The standalone MARK_AX should be gone.
    assert " MARK_AX " not in f" {migrated.scope} "


def test_migrate_rule_appends_step_end_suffix_to_name():
    rule = _l10_bitwise_or_like_rule()
    migrated = migrate_rule_to_step_end(rule)
    assert migrated.name == "probe_bitwise_or_a5_b3_step_end"
    # Idempotent on a second pass.
    migrated2 = migrate_rule_to_step_end(migrated, safety_check=False)
    assert migrated2.name == "probe_bitwise_or_a5_b3_step_end"


def test_migrate_rule_propagates_to_gate_terms():
    rule = FFNRule.gated_write(
        name="probe_gate_terms",
        conditions=(("MARK_AX", 1.0),),
        threshold=0.5,
        gate_terms=(("MARK_AX", 2.0), ("CONST", 1.0)),
        writes=(("TEMP+0", 1.0),),
    )
    migrated = migrate_rule_to_step_end(rule)
    gate_names = [c.dim.name for c in migrated.gate_terms]
    assert MARK_AX_NAME not in gate_names
    assert MARK_SE_NAME in gate_names


# ---------------------------------------------------------------------------
# assert_migration_safe
# ---------------------------------------------------------------------------


def test_assert_migration_safe_rejects_rule_without_mark_ax_condition():
    rule = FFNRule.constant_write(
        name="probe_no_mark_ax",
        conditions=(("CONST", 1.0),),
        threshold=0.5,
        writes=(("TEMP+0", 1.0),),
    )
    with pytest.raises(MigrationSafetyError, match="no MARK_AX condition"):
        assert_migration_safe(rule)


def test_assert_migration_safe_rejects_byte_emission_writes():
    rule = FFNRule.constant_write(
        name="probe_byte_write",
        conditions=(("MARK_AX", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 1.0),),
    )
    with pytest.raises(MigrationSafetyError, match="byte-emission slot"):
        assert_migration_safe(rule)


def test_assert_migration_safe_rejects_marker_writes():
    rule = FFNRule.constant_write(
        name="probe_marker_write",
        conditions=(("MARK_AX", 1.0),),
        threshold=0.5,
        writes=(("MARK_PC+0", 1.0),),
    )
    with pytest.raises(MigrationSafetyError, match="row-local marker dim"):
        assert_migration_safe(rule)


def test_assert_migration_safe_passes_for_compute_rule():
    rule = _l10_bitwise_or_like_rule()
    # Should not raise.
    assert_migration_safe(rule)


def test_migrate_rule_safety_check_off_skips_validation():
    rule = FFNRule.constant_write(
        name="probe_no_mark_ax",
        conditions=(("CONST", 1.0),),
        threshold=0.5,
        writes=(("TEMP+0", 1.0),),
    )
    # safety_check=False: helper returns a (semantically vacuous)
    # copy without raising. Useful for unit tests + idempotent
    # double-migration.
    migrated = migrate_rule_to_step_end(rule, safety_check=False)
    assert migrated.name == "probe_no_mark_ax_step_end"


def test_assert_migration_safe_allow_list_unblocks_specific_write():
    rule = FFNRule.constant_write(
        name="probe_allow",
        conditions=(("MARK_AX", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 1.0),),
    )
    # Caller asserts this OUTPUT_LO write is actually safe at SE.
    # Should not raise.
    assert_migration_safe(
        rule, allow_step_end_writes_to=("OUTPUT_LO",)
    )


# ---------------------------------------------------------------------------
# migrate_attention_head_to_step_end
# ---------------------------------------------------------------------------


def test_migrate_attention_head_rewrites_q_and_k_writes():
    # MARK_AX = dim 100, MARK_SE_ONLY = dim 200 (fake indices for test).
    spec = DeclarativeAttentionHeadSpec(
        head_idx=3,
        q=(AP(0, 100, 1.0), AP(1, 7, 2.0)),
        k=(AP(0, 100, 10.0), AP(1, 8, -3.0)),
        v=(AP(0, 9, 1.0),),
        o=(AO(11, 0, 1.0),),
        alibi_slope=0.5,
    )
    migrated = migrate_attention_head_to_step_end(
        spec, mark_ax_dim_idx=100, mark_se_dim_idx=200
    )
    q_dims = [w.dim for w in migrated.q]
    k_dims = [w.dim for w in migrated.k]
    assert 100 not in q_dims and 200 in q_dims
    assert 100 not in k_dims and 200 in k_dims
    # V and O untouched.
    assert migrated.v == spec.v
    assert migrated.o == spec.o
    # head_idx / alibi_slope carried over.
    assert migrated.head_idx == 3
    assert migrated.alibi_slope == 0.5


def test_migrate_attention_head_no_op_when_dim_absent():
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, 7, 1.0),),
        k=(AP(0, 8, 1.0),),
    )
    migrated = migrate_attention_head_to_step_end(
        spec, mark_ax_dim_idx=999, mark_se_dim_idx=1000
    )
    assert migrated.q == spec.q
    assert migrated.k == spec.k
