"""Phase 9 SSA dim-name parser tests.

Smoke tests for :mod:`neural_vm.unified_compiler.ssa_dim`. Covers the
parser, the round-trip ``canonical`` accessor, and the cheap
:func:`is_ssa_form` / :func:`base_of` helpers used on the hot
add_op path.
"""

import pytest

from neural_vm.unified_compiler.ssa_dim import (
    SSA_ANY_WRITER,
    SsaDimName,
    base_of,
    is_ssa_form,
    make_ssa_name,
    parse_ssa_name,
)


def test_unversioned_name_parses_as_current_step():
    parsed = parse_ssa_name("OUTPUT_LO")
    assert parsed.base_dim == "OUTPUT_LO"
    assert parsed.writer_op is None
    assert parsed.step_offset == 0
    assert not parsed.is_cross_step
    assert not parsed.is_any_writer


def test_dotted_name_parses_writer_and_offset():
    parsed = parse_ssa_name("OUTPUT_LO.layer16_lev_routing.-1")
    assert parsed.base_dim == "OUTPUT_LO"
    assert parsed.writer_op == "layer16_lev_routing"
    assert parsed.step_offset == -1
    assert parsed.is_cross_step
    assert not parsed.is_any_writer


def test_wildcard_writer():
    parsed = parse_ssa_name("OUTPUT_LO.*.-1")
    assert parsed.writer_op == "*"
    assert parsed.is_any_writer
    assert parsed.is_cross_step


def test_canonical_round_trip_unversioned():
    name = "OUTPUT_LO"
    assert parse_ssa_name(name).canonical == name


def test_canonical_round_trip_dotted():
    name = "OUTPUT_LO.layer9_alu.-1"
    assert parse_ssa_name(name).canonical == name


def test_canonical_round_trip_wildcard():
    name = "OUTPUT_HI.*.-1"
    assert parse_ssa_name(name).canonical == name


def test_invalid_segment_count_raises():
    with pytest.raises(ValueError, match="exactly 3"):
        parse_ssa_name("OUTPUT_LO.layer9_alu")
    with pytest.raises(ValueError, match="exactly 3"):
        parse_ssa_name("a.b.c.d")


def test_empty_base_raises():
    with pytest.raises(ValueError, match="empty base dim"):
        parse_ssa_name(".layer9_alu.-1")


def test_empty_writer_raises():
    with pytest.raises(ValueError, match="empty writer op"):
        parse_ssa_name("OUTPUT_LO..-1")


def test_non_integer_offset_raises():
    with pytest.raises(ValueError, match="not an int"):
        parse_ssa_name("OUTPUT_LO.layer9_alu.last")


def test_is_ssa_form():
    assert not is_ssa_form("OUTPUT_LO")
    assert is_ssa_form("OUTPUT_LO.layer9_alu.-1")
    assert is_ssa_form("OUTPUT_LO.*.-1")


def test_base_of_strips_suffix():
    assert base_of("OUTPUT_LO") == "OUTPUT_LO"
    assert base_of("OUTPUT_LO.layer9_alu.-1") == "OUTPUT_LO"
    assert base_of("ADDR_KEY.*.-2") == "ADDR_KEY"


def test_make_ssa_name_round_trip():
    name = make_ssa_name("OUTPUT_HI", "layer16_lev_routing", -1)
    assert name == "OUTPUT_HI.layer16_lev_routing.-1"
    parsed = parse_ssa_name(name)
    assert parsed == SsaDimName("OUTPUT_HI", "layer16_lev_routing", -1)


def test_make_ssa_name_wildcard():
    assert make_ssa_name("OUTPUT_LO", SSA_ANY_WRITER, -1) == "OUTPUT_LO.*.-1"


def test_make_ssa_name_rejects_dot_in_components():
    with pytest.raises(ValueError, match="base_dim"):
        make_ssa_name("OUTPUT.LO", "writer", -1)
    with pytest.raises(ValueError, match="writer_op"):
        make_ssa_name("OUTPUT_LO", "layer.9", -1)


def test_step_offset_can_be_positive():
    # Forward-step reads are syntactically valid; semantics is
    # "value the writer WILL produce". Currently unused, but the parser
    # is uniform.
    parsed = parse_ssa_name("OUTPUT_LO.layer16_lev_routing.1")
    assert parsed.step_offset == 1
    assert parsed.is_cross_step
