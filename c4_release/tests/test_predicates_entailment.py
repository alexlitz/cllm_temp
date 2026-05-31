"""Entailment tests for the predicate DSL."""

from __future__ import annotations

import pytest

from neural_vm.unified_compiler.predicates import (
    entails,
    explain_failure,
    parse,
)


# ---------------------------------------------------------------------------
# Reflexivity (p ⊨ p) across representative predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "mark == SP",
        "mark in {SP, BP}",
        "step_index == 0",
        "step_index in [0, 3)",
        "step_index in {0, 1, 2}",
        "opcode_at_AX == JSR",
        "opcode_at_AX in {JSR, RTS}",
        "opcode_in_step in {LDA, STA}",
        "byte_index == 4",
        "byte_index in {0, 1, 2}",
        "byte_value == 0xF8",
        "byte_value in {0x01, 0x02}",
        "byte_value.lo_nibble == 0x8",
        "byte_value.hi_nibble == 0xF",
        "sp_byte0 == 0xE0",
        "output_lo_nibble == 0x3",
        "is_byte",
        "NOT mark == SP",
        "NOT step_is_fresh",
        "mark == SP AND step_is_fresh",
        "mark == SP OR mark == BP",
        "(mark == SP AND step_is_fresh) OR mark == BP",
    ],
)
def test_reflexive(text):
    p = parse(text)
    assert entails(p, p), f"p ⊨ p failed for {text!r}"
    assert explain_failure(p, p) is None


# ---------------------------------------------------------------------------
# Marker family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_text,q_text,expected",
    [
        ("mark == SP", "mark in {SP, BP}", True),
        ("mark in {SP, BP}", "mark == SP", False),
        ("mark in {SP}", "mark == SP", True),
        ("mark in {SP, BP}", "mark in {SP, BP, AX}", True),
        ("mark in {SP, BP, AX}", "mark in {SP, BP}", False),
        ("mark == SP", "mark == BP", False),
        ("mark == SP", "NOT mark == BP", True),
        ("mark == SP", "NOT mark == SP", False),
    ],
)
def test_marker_entailment(p_text, q_text, expected):
    p, q = parse(p_text), parse(q_text)
    assert entails(p, q) is expected, f"entails({p_text!r}, {q_text!r}) != {expected}"


# ---------------------------------------------------------------------------
# Step-index family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_text,q_text,expected",
    [
        ("step_index == 0", "step_index in [0, 3)", True),
        ("step_index in [0, 3)", "step_index == 0", False),
        ("step_index == 2", "step_index in [0, 3)", True),
        ("step_index == 3", "step_index in [0, 3)", False),
        ("step_index == 0", "step_index in {0, 1, 2}", True),
        ("step_index in {0, 1}", "step_index in {0, 1, 2}", True),
        ("step_index in {0, 1, 2}", "step_index in {0, 1}", False),
        ("step_index in [0, 2)", "step_index in {0, 1}", True),
        ("step_index in [0, 2)", "step_index in [0, 5)", True),
        ("step_index in [0, 5)", "step_index in [0, 2)", False),
    ],
)
def test_step_index_entailment(p_text, q_text, expected):
    p, q = parse(p_text), parse(q_text)
    assert entails(p, q) is expected


# ---------------------------------------------------------------------------
# Byte-value to nibble entailment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_text,q_text,expected",
    [
        ("byte_value == 0xF8", "byte_value.lo_nibble == 0x8", True),
        ("byte_value == 0xF8", "byte_value.hi_nibble == 0xF", True),
        ("byte_value == 0xF8", "byte_value.lo_nibble == 0x7", False),
        ("byte_value.lo_nibble == 0x8", "byte_value == 0xF8", False),
        ("byte_value in {0x18, 0x28}", "byte_value.lo_nibble == 0x8", True),
        ("byte_value in {0x18, 0x29}", "byte_value.lo_nibble == 0x8", False),
        ("byte_value in {0xF1, 0xF2}", "byte_value.hi_nibble == 0xF", True),
    ],
)
def test_byte_value_nibble_entailment(p_text, q_text, expected):
    p, q = parse(p_text), parse(q_text)
    assert entails(p, q) is expected


# ---------------------------------------------------------------------------
# Opcode family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_text,q_text,expected",
    [
        ("opcode_at_AX == JSR", "opcode_at_AX in {JSR, RTS}", True),
        ("opcode_at_AX in {JSR, RTS}", "opcode_at_AX == JSR", False),
        ("opcode_at_AX == JSR", "opcode_at_AX == CMP", False),
        ("opcode_at_AX == JSR", "NOT opcode_at_AX == CMP", True),
        ("opcode_in_step in {LDA}", "opcode_in_step in {LDA, STA}", True),
    ],
)
def test_opcode_entailment(p_text, q_text, expected):
    p, q = parse(p_text), parse(q_text)
    assert entails(p, q) is expected


# ---------------------------------------------------------------------------
# AND / OR / NOT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_text,q_text,expected",
    [
        # AND tightens: more conjuncts on left subsumes fewer on right.
        ("mark == SP AND step_is_fresh", "mark == SP", True),
        ("mark == SP AND step_is_fresh", "step_is_fresh", True),
        # Reverse: dropping a conjunct on the left weakens it (not allowed).
        ("mark == SP", "mark == SP AND step_is_fresh", False),
        # OR widens: a more-specific p fits inside a wider q-disjunction.
        ("mark == SP", "mark == SP OR mark == BP", True),
        ("mark == SP OR mark == BP", "mark == SP", False),
        # Both sides have OR.
        ("mark == SP OR mark == BP", "mark in {SP, BP, AX}", True),
        ("mark in {SP, BP, AX}", "mark == SP OR mark == BP", False),
    ],
)
def test_and_or_entailment(p_text, q_text, expected):
    p, q = parse(p_text), parse(q_text)
    assert entails(p, q) is expected


# ---------------------------------------------------------------------------
# NOT / De Morgan
# ---------------------------------------------------------------------------


def test_not_roundtrip():
    p = parse("NOT mark == SP")
    q = parse("NOT mark == SP")
    assert entails(p, q)
    assert entails(q, p)


def test_de_morgan_or_to_not_member():
    # NOT (mark == SP OR mark == BP)
    #   ==> NOT mark == SP AND NOT mark == BP
    # So it entails NOT mark == SP.
    p = parse("NOT (mark == SP OR mark == BP)")
    q = parse("NOT mark == SP")
    assert entails(p, q)


def test_de_morgan_and():
    # NOT (mark == SP AND step_is_fresh)
    #   ==> NOT mark == SP OR NOT step_is_fresh
    # The disjunct "NOT mark == SP" doesn't entail "NOT step_is_fresh", so
    # the whole left side doesn't either.
    p = parse("NOT (mark == SP AND step_is_fresh)")
    q = parse("NOT step_is_fresh")
    assert not entails(p, q)


def test_double_negation_cancels():
    p = parse("NOT NOT mark == SP")
    q = parse("mark == SP")
    assert entails(p, q)
    assert entails(q, p)


# ---------------------------------------------------------------------------
# E5 case — the headline catch
# ---------------------------------------------------------------------------


def test_e5_producer_does_not_entail_more_specific_consumer():
    """Producer 'opcode_at_AX == JSR' should NOT entail the consumer's
    stronger assumption 'opcode_at_AX == JSR AND step_index == 0'.
    """
    producer = parse("opcode_at_AX == JSR")
    consumer = parse("opcode_at_AX == JSR AND step_index == 0")
    assert not entails(producer, consumer)

    msg = explain_failure(producer, consumer)
    assert msg is not None
    assert "step_index" in msg


def test_e5_reverse_direction_holds():
    """Conversely, the stronger conjunction entails the weaker conjunct."""
    producer = parse("opcode_at_AX == JSR AND step_index == 0")
    consumer = parse("opcode_at_AX == JSR")
    assert entails(producer, consumer)
    assert explain_failure(producer, consumer) is None


# ---------------------------------------------------------------------------
# explain_failure shape
# ---------------------------------------------------------------------------


def test_explain_failure_returns_none_on_success():
    assert explain_failure(parse("mark == SP"), parse("mark in {SP, BP}")) is None


def test_explain_failure_string_mentions_atom():
    msg = explain_failure(parse("mark == SP"), parse("mark == BP"))
    assert msg is not None
    assert "BP" in msg


# ---------------------------------------------------------------------------
# Cross-family unrelated atoms
# ---------------------------------------------------------------------------


def test_cross_family_unrelated():
    # mark == SP says nothing about step_index.
    assert not entails(parse("mark == SP"), parse("step_index == 0"))
    assert not entails(parse("step_index == 0"), parse("mark == SP"))


# ---------------------------------------------------------------------------
# Mixed AND / OR realistic predicates
# ---------------------------------------------------------------------------


def test_distributive_and_over_or():
    # (A AND (B OR C)) is equivalent to ((A AND B) OR (A AND C)).
    p = parse("mark == SP AND (step_is_fresh OR has_se)")
    q = parse("(mark == SP AND step_is_fresh) OR (mark == SP AND has_se)")
    assert entails(p, q)
    assert entails(q, p)


def test_range_entails_set():
    p = parse("step_index in [0, 3)")
    q = parse("step_index in {0, 1, 2, 3, 4}")
    assert entails(p, q)


def test_set_entails_range_when_in_bounds():
    p = parse("step_index in {0, 1, 2}")
    q = parse("step_index in [0, 5)")
    assert entails(p, q)
