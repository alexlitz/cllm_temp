"""Parser tests for the predicate DSL in neural_vm.unified_compiler.predicates."""

from __future__ import annotations

import pytest

from neural_vm.unified_compiler.predicates import (
    And,
    BoolAtom,
    ByteIndexEq,
    ByteIndexIn,
    ByteValueEq,
    ByteValueHiNibbleEq,
    ByteValueIn,
    ByteValueLoNibbleEq,
    MarkEq,
    MarkIn,
    Not,
    OpcodeAtAxEq,
    OpcodeAtAxIn,
    OpcodeInStepIn,
    Or,
    OutputHiNibbleEq,
    OutputLoNibbleEq,
    SpByte0Eq,
    StepIndexEq,
    StepIndexInRange,
    StepIndexInSet,
    parse,
)


# ---------------------------------------------------------------------------
# Per-family parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mark == SP", MarkEq("SP")),
        ("mark == BP", MarkEq("BP")),
        ("mark == MEM", MarkEq("MEM")),
        ("mark == NONE", MarkEq("NONE")),
        ("mark == STACK0", MarkEq("STACK0")),
        ("mark in {SP, BP}", MarkIn(frozenset({"SP", "BP"}))),
        ("mark in {SE}", MarkIn(frozenset({"SE"}))),
    ],
)
def test_parse_marker_atoms(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "name",
    [
        "is_byte",
        "has_se",
        "step_is_fresh",
        "in_step_fresh",
        "addr_b0_valid",
        "addr_b1_valid",
        "addr_b2_valid",
        "sp_gathered_this_step",
    ],
)
def test_parse_boolean_atoms(name):
    assert parse(name) == BoolAtom(name)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("step_index == 0", StepIndexEq(0)),
        ("step_index == 42", StepIndexEq(42)),
        ("step_index in [0, 3)", StepIndexInRange(0, 3)),
        ("step_index in [5, 10)", StepIndexInRange(5, 10)),
        ("step_index in {0, 2, 4}", StepIndexInSet(frozenset({0, 2, 4}))),
        ("step_index in {7}", StepIndexInSet(frozenset({7}))),
    ],
)
def test_parse_step_index_atoms(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("opcode_at_AX == JSR", OpcodeAtAxEq("JSR")),
        ("opcode_at_AX == CMP", OpcodeAtAxEq("CMP")),
        (
            "opcode_at_AX in {JSR, RTS}",
            OpcodeAtAxIn(frozenset({"JSR", "RTS"})),
        ),
        (
            "opcode_in_step in {LDA, STA}",
            OpcodeInStepIn(frozenset({"LDA", "STA"})),
        ),
    ],
)
def test_parse_opcode_atoms(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("byte_index == 0", ByteIndexEq(0)),
        ("byte_index == 4", ByteIndexEq(4)),
        ("byte_index in {0, 1, 2}", ByteIndexIn(frozenset({0, 1, 2}))),
    ],
)
def test_parse_byte_index_atoms(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("byte_value == 0xF8", ByteValueEq(0xF8)),
        ("byte_value == 0x00", ByteValueEq(0x00)),
        ("byte_value in {0x01, 0x02}", ByteValueIn(frozenset({0x01, 0x02}))),
        ("byte_value.lo_nibble == 0x8", ByteValueLoNibbleEq(0x8)),
        ("byte_value.hi_nibble == 0xF", ByteValueHiNibbleEq(0xF)),
        ("sp_byte0 == 0xE0", SpByte0Eq(0xE0)),
    ],
)
def test_parse_byte_value_atoms(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("output_lo_nibble == 0x3", OutputLoNibbleEq(0x3)),
        ("output_hi_nibble == 0xA", OutputHiNibbleEq(0xA)),
    ],
)
def test_parse_output_atoms(text, expected):
    assert parse(text) == expected


# ---------------------------------------------------------------------------
# Connectives, paren grouping, precedence
# ---------------------------------------------------------------------------


def test_paren_grouping_respected():
    p = parse("(mark == SP OR mark == BP) AND step_is_fresh")
    assert isinstance(p, And)
    assert len(p.children) == 2
    assert isinstance(p.children[0], Or)
    assert p.children[1] == BoolAtom("step_is_fresh")


def test_precedence_or_lower_than_and():
    # A OR B AND C  ==>  A OR (B AND C)
    p = parse("mark == SP OR step_is_fresh AND has_se")
    assert isinstance(p, Or)
    assert p.children[0] == MarkEq("SP")
    assert isinstance(p.children[1], And)
    assert p.children[1].children == (BoolAtom("step_is_fresh"), BoolAtom("has_se"))


def test_precedence_not_higher_than_and():
    # NOT A AND B  ==>  (NOT A) AND B
    p = parse("NOT mark == SP AND step_is_fresh")
    assert isinstance(p, And)
    assert isinstance(p.children[0], Not)
    assert p.children[0].child == MarkEq("SP")
    assert p.children[1] == BoolAtom("step_is_fresh")


def test_nary_and():
    p = parse("mark == SP AND step_is_fresh AND has_se")
    assert isinstance(p, And)
    assert len(p.children) == 3


def test_nary_or():
    p = parse("mark == SP OR mark == BP OR mark == AX")
    assert isinstance(p, Or)
    assert len(p.children) == 3


def test_double_negation():
    p = parse("NOT NOT mark == SP")
    assert isinstance(p, Not)
    assert isinstance(p.child, Not)
    assert p.child.child == MarkEq("SP")


# ---------------------------------------------------------------------------
# Whitespace tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "mark==SP",
        "mark == SP",
        "  mark   ==   SP  ",
        "mark\t==\tSP",
        "(mark==SP)",
        "( mark == SP )",
    ],
)
def test_whitespace_tolerance(text):
    assert parse(text) == MarkEq("SP")


# ---------------------------------------------------------------------------
# Bad inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,needle",
    [
        ("frobnicate", "unknown atom"),
        ("mark == FOO", "unknown mark role"),
        ("(mark == SP", "unmatched"),
        ("", "empty"),
        ("   ", "empty"),
        ("byte_value == 0xZZ", "hex"),
        ("byte_value == 0x", "invalid hex literal"),
        ("step_index ==", "unexpected end"),
        ("mark == SP OR", "unexpected end"),
        ("mark in {SP", "unexpected end"),
        ("step_index in [0, 3]", "expected ')'"),
    ],
)
def test_bad_input_raises_value_error(text, needle):
    with pytest.raises(ValueError) as exc:
        parse(text)
    assert needle.lower() in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Pretty-printing round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "mark == SP",
        "mark in {BP, SP}",
        "step_is_fresh",
        "NOT mark == SP",
        "step_index == 0",
        "step_index in [0, 3)",
        "step_index in {0, 1, 2}",
        "opcode_at_AX == JSR",
        "opcode_at_AX in {JSR, RTS}",
        "opcode_in_step in {LDA, STA}",
        "byte_index == 0",
        "byte_index in {0, 1, 2}",
        "byte_value == 0xF8",
        "byte_value in {0x01, 0x02}",
        "byte_value.lo_nibble == 0x8",
        "byte_value.hi_nibble == 0xF",
        "sp_byte0 == 0xE0",
        "output_lo_nibble == 0x3",
        "output_hi_nibble == 0xA",
        "(mark == SP AND step_is_fresh)",
        "(mark == SP OR mark == BP)",
        "(NOT mark == SP AND step_is_fresh)",
        "(mark == SP OR (step_is_fresh AND has_se))",
    ],
)
def test_pretty_print_roundtrip(text):
    ast1 = parse(text)
    text2 = str(ast1)
    ast2 = parse(text2)
    assert ast1 == ast2, f"round-trip failed: {text!r} -> {text2!r}"
