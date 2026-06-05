"""Tests for the typed dim IR layer (``ir_types.py``)."""

from __future__ import annotations

import pytest

from c4_release.neural_vm.unified_compiler.ir_types import (
    DimSchema,
    DimSpec,
    DimType,
    TypeIssue,
    Value,
    check_rule_types,
    check_rules,
    global_schema,
    register_dim,
    schema_for,
)
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    multi_way_and_rule,
    one_hot_indicator_rule,
    step_function_rule,
)
from c4_release.neural_vm.unified_compiler.ir import FFNRule


# ===========================================================================
# DimType + DimSpec basics
# ===========================================================================


def test_dim_spec_cell_validates_single_cell_offset():
    spec = DimSpec("MARK_AX", DimType.MARKER, 1)
    # offset 0 ok
    v = spec.cell(0)
    assert v.name == "MARK_AX"
    assert v.offset == 0
    # offset 5 rejected
    with pytest.raises(ValueError, match="single-cell"):
        spec.cell(5)


def test_dim_spec_cell_validates_band_offset_range():
    spec = DimSpec("ALU_LO", DimType.BAND, width=16)
    spec.cell(0)
    spec.cell(15)
    with pytest.raises(ValueError, match="out of width"):
        spec.cell(16)
    with pytest.raises(ValueError, match="out of width"):
        spec.cell(-1)


def test_dim_spec_is_band_helper():
    assert DimSpec("ALU_LO", DimType.BAND, 16).is_band
    assert DimSpec("OUTPUT_LO", DimType.OUTPUT, 16).is_band
    assert not DimSpec("MARK_AX", DimType.MARKER).is_band
    assert not DimSpec("CONST", DimType.SCALAR).is_band


# ===========================================================================
# Schema registration + inference
# ===========================================================================


def test_global_schema_has_canonical_markers():
    schema = global_schema()
    for name in ("MARK_PC", "MARK_AX", "MARK_SP"):
        spec = schema.get(name)
        assert spec is not None
        assert spec.type == DimType.MARKER


def test_global_schema_has_canonical_opcode_flags():
    for name in ("OP_ADD", "OP_LEV", "OP_IMM"):
        assert schema_for(name).type == DimType.OPCODE_FLAG


def test_global_schema_has_canonical_output_bands():
    out_lo = schema_for("OUTPUT_LO")
    assert out_lo.type == DimType.OUTPUT
    assert out_lo.width == 16


def test_global_schema_has_canonical_carries():
    assert schema_for("CARRY").type == DimType.CARRY
    assert schema_for("BORROW").type == DimType.CARRY


def test_inference_falls_back_for_unregistered_names():
    # An ALU_LO-like band that's not registered should still infer BAND
    spec = schema_for("FOO_LO")
    assert spec.type == DimType.BAND
    assert spec.width == 16


def test_inference_recognizes_mark_prefix():
    assert schema_for("MARK_WEIRDO").type == DimType.MARKER


def test_inference_falls_back_to_unknown_for_random_names():
    assert schema_for("ZZZQQQ").type == DimType.UNKNOWN


# ===========================================================================
# Value type
# ===========================================================================


def test_value_of_typed_lookup():
    v = Value.of("ALU_LO", 5)
    assert v.name == "ALU_LO"
    assert v.offset == 5
    assert v.type == DimType.BAND


def test_value_parse_simple():
    v = Value.parse("MARK_AX")
    assert v.name == "MARK_AX"
    assert v.offset == 0
    assert v.version is None
    assert v.type == DimType.MARKER


def test_value_parse_with_offset():
    v = Value.parse("ALU_LO+5")
    assert v.name == "ALU_LO"
    assert v.offset == 5
    assert v.type == DimType.BAND


def test_value_parse_cross_step_alias():
    v = Value.parse("OUTPUT_LO.*.-1")
    assert v.name == "OUTPUT_LO"
    assert v.offset == 0
    assert v.version == -1
    assert v.is_cross_step
    assert v.type == DimType.OUTPUT


def test_value_to_key_roundtrips():
    for key in ("MARK_AX", "ALU_LO+5", "OUTPUT_LO.*.-1"):
        assert Value.parse(key).to_key() == key


# ===========================================================================
# Type checking against real building-block rules
# ===========================================================================


def test_step_function_rule_clean_under_type_check():
    """A step_function_rule on a known marker should produce zero issues."""

    rule = step_function_rule(
        input_dim="MARK_AX", threshold=0.5, write_dim="OUTPUT_LO+5",
        write_value=2.0,
    )
    issues = check_rule_types(rule)
    assert issues == []


def test_one_hot_indicator_rule_clean_under_type_check():
    rule = one_hot_indicator_rule(
        band="ALU_LO", value=7, write_dim="OUTPUT_LO+0", write_value=2.0,
    )
    issues = check_rule_types(rule)
    assert issues == []


def test_multi_way_and_rule_clean_under_type_check():
    rule = multi_way_and_rule(
        conditions=(("MARK_AX", 40.0), ("ALU_LO+5", 30.0), ("ALU_HI+3", 30.0)),
        threshold=80.0,
        writes=(("OUTPUT_LO+5", 0.02),),
        gate="OP_AND",
    )
    issues = check_rule_types(rule)
    assert issues == []


def test_type_check_catches_offset_on_non_band():
    """A condition like ``MARK_AX+5`` should flag because markers are
    single-cell."""

    rule = FFNRule.constant_write(
        conditions=(("MARK_AX+5", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO+5", 0.02),),
    )
    issues = check_rule_types(rule)
    assert any(issue.kind == "single_cell_offset" for issue in issues), (
        f"expected single_cell_offset issue, got {issues!r}"
    )


def test_type_check_catches_offset_out_of_range():
    """ALU_LO+20 is past the 16-cell band width."""

    rule = FFNRule.constant_write(
        conditions=(("ALU_LO+20", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 0.02),),
    )
    issues = check_rule_types(rule)
    assert any(
        issue.kind == "offset_out_of_range" for issue in issues
    ), f"expected offset_out_of_range issue, got {issues!r}"


def test_type_check_ignores_unknown_dims():
    """Unregistered dims shouldn't trigger spurious failures."""

    rule = FFNRule.constant_write(
        conditions=(("ZZZ_UNKNOWN", 1.0),),
        threshold=0.5,
        writes=(("YYY_UNKNOWN+5", 0.02),),
    )
    issues = check_rule_types(rule)
    assert issues == [], (
        f"unknown dims should be silently accepted, got {issues!r}"
    )


def test_check_rules_flattens_issues_across_rule_list():
    bad_rules = [
        FFNRule.constant_write(
            conditions=(("MARK_AX+5", 1.0),),
            threshold=0.5,
            writes=(("OUTPUT_LO+0", 0.02),),
        ),
        FFNRule.constant_write(
            conditions=(("ALU_LO+20", 1.0),),
            threshold=0.5,
            writes=(("OUTPUT_LO+0", 0.02),),
        ),
    ]
    issues = check_rules(bad_rules)
    assert len(issues) == 2
    kinds = {issue.kind for issue in issues}
    assert "single_cell_offset" in kinds
    assert "offset_out_of_range" in kinds


# ===========================================================================
# Schema instance can run independently
# ===========================================================================


def test_dim_schema_local_instance():
    """Tests can build their own DimSchema without polluting the global."""

    schema = DimSchema()
    schema.register(DimSpec("FOO", DimType.MARKER, 1))
    assert schema.get("FOO").type == DimType.MARKER
    assert schema.get("BAR") is None
    assert "FOO" in schema
    assert len(schema) == 1


def test_dim_schema_rejects_conflicting_register():
    schema = DimSchema()
    schema.register(DimSpec("FOO", DimType.MARKER, 1))
    # Same spec is fine (idempotent)
    schema.register(DimSpec("FOO", DimType.MARKER, 1))
    # Conflicting spec rejected
    with pytest.raises(ValueError, match="conflicting"):
        schema.register(DimSpec("FOO", DimType.CARRY, 1))


# ===========================================================================
# Operand views — typed reads/writes + producer/consumer queries
# ===========================================================================


class _FakeOp:
    """Minimal Operation stand-in for testing the typed-operand layer
    without pulling in the real LayerCompiler dependency."""

    def __init__(self, name, reads, writes):
        self.name = name
        self.reads = set(reads)
        self.writes = set(writes)


def test_typed_reads_returns_value_list_with_types():
    from c4_release.neural_vm.unified_compiler.ir_types import typed_reads

    op = _FakeOp("foo", reads={"MARK_AX", "ALU_LO+5"}, writes={"OUTPUT_LO+0"})
    reads = typed_reads(op)
    assert len(reads) == 2
    names = {r.name for r in reads}
    assert names == {"MARK_AX", "ALU_LO"}
    by_name = {r.name: r for r in reads}
    assert by_name["MARK_AX"].type == DimType.MARKER
    assert by_name["ALU_LO"].type == DimType.BAND
    assert by_name["ALU_LO"].offset == 5


def test_typed_writes_returns_value_list():
    from c4_release.neural_vm.unified_compiler.ir_types import typed_writes

    op = _FakeOp("foo", reads=set(), writes={"OUTPUT_LO+0", "OUTPUT_HI+15"})
    writes = typed_writes(op)
    assert len(writes) == 2
    for w in writes:
        assert w.type == DimType.OUTPUT


def test_operand_use_def_returns_both_lists():
    from c4_release.neural_vm.unified_compiler.ir_types import operand_use_def

    op = _FakeOp("foo", reads={"MARK_AX"}, writes={"OUTPUT_LO+0"})
    reads, writes = operand_use_def(op)
    assert len(reads) == 1
    assert len(writes) == 1
    assert reads[0].name == "MARK_AX"
    assert writes[0].name == "OUTPUT_LO"


def test_typed_reads_handles_cross_step_alias():
    from c4_release.neural_vm.unified_compiler.ir_types import typed_reads

    op = _FakeOp("foo", reads={"OUTPUT_LO.*.-1"}, writes=set())
    reads = typed_reads(op)
    assert len(reads) == 1
    assert reads[0].name == "OUTPUT_LO"
    assert reads[0].version == -1
    assert reads[0].is_cross_step


def test_find_producers_locates_writers():
    from c4_release.neural_vm.unified_compiler.ir_types import find_producers

    op1 = _FakeOp("alpha", reads=set(), writes={"OUTPUT_LO"})
    op2 = _FakeOp("beta", reads={"MARK_AX"}, writes={"CARRY"})
    op3 = _FakeOp("gamma", reads=set(), writes={"OUTPUT_LO+5"})

    producers = find_producers([op1, op2, op3], "OUTPUT_LO")
    names = {op.name for op in producers}
    assert names == {"alpha", "gamma"}
    # op2 doesn't write OUTPUT_LO
    assert "beta" not in names


def test_find_consumers_locates_readers():
    from c4_release.neural_vm.unified_compiler.ir_types import find_consumers

    op1 = _FakeOp("alpha", reads={"MARK_AX"}, writes=set())
    op2 = _FakeOp("beta", reads={"ALU_LO+5", "MARK_AX+0"}, writes=set())
    op3 = _FakeOp("gamma", reads={"CMP+3"}, writes=set())

    consumers = find_consumers([op1, op2, op3], "MARK_AX")
    names = {op.name for op in consumers}
    assert "alpha" in names
    assert "beta" in names
    assert "gamma" not in names
