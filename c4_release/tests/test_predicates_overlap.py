"""S-3: predicate satisfiability + overlap tests."""
import pytest
from neural_vm.unified_compiler.predicates import parse, satisfiable, overlaps


# === satisfiable ===

def test_atom_satisfiable():
    assert satisfiable(parse("mark == SP"))


def test_conjunction_satisfiable():
    assert satisfiable(parse("mark == SP AND step_index == 0"))


def test_self_contradiction_unsatisfiable():
    assert not satisfiable(parse("mark == SP AND mark == AX"))


def test_disjoint_set_unsatisfiable():
    assert not satisfiable(parse("mark == SP AND mark in {AX, PC}"))


def test_step_index_range_contradicts_eq():
    assert not satisfiable(parse("step_index == 0 AND step_index in [1, 5)"))


def test_step_index_set_disjoint():
    assert not satisfiable(parse("step_index in {0, 1} AND step_index in {2, 3}"))


def test_byte_value_contradicts():
    assert not satisfiable(parse("byte_value == 0xF8 AND byte_value == 0x00"))


def test_byte_value_with_compatible_nibble_ok():
    # 0xF8 has lo nibble 0x8 — these are compatible
    assert satisfiable(parse("byte_value == 0xF8 AND byte_value.lo_nibble == 0x8"))


def test_byte_value_with_incompatible_nibble_contradicts():
    # 0xF8's lo nibble is 0x8, not 0x0
    assert not satisfiable(parse("byte_value == 0xF8 AND byte_value.lo_nibble == 0x0"))


def test_or_satisfiable_if_any_disjunct():
    # second disjunct is unsatisfiable but first is fine
    assert satisfiable(parse("mark == SP OR (mark == SP AND mark == AX)"))


def test_or_unsatisfiable_iff_all_disjuncts():
    assert not satisfiable(parse("(mark == SP AND mark == AX) OR (step_index == 0 AND step_index == 1)"))


# === overlaps ===

def test_overlap_reflexive():
    assert overlaps(parse("mark == SP"), parse("mark == SP"))


def test_overlap_specialization():
    """A more-specific predicate overlaps with a less-specific one."""
    assert overlaps(
        parse("mark == SP AND step_index == 0"),
        parse("mark == SP"),
    )


def test_overlap_disjoint_marker():
    assert not overlaps(parse("mark == SP"), parse("mark == AX"))


def test_overlap_independent_dimensions():
    """Cross-family predicates always overlap (mark and step_index are independent)."""
    assert overlaps(parse("mark == SP"), parse("step_index == 0"))


def test_overlap_set_intersection():
    """mark in {SP, BP} overlaps with mark in {BP, AX} (BP is common)."""
    assert overlaps(parse("mark in {SP, BP}"), parse("mark in {BP, AX}"))


def test_overlap_set_disjoint():
    assert not overlaps(parse("mark in {SP, BP}"), parse("mark in {AX, PC}"))


def test_overlap_e5_case():
    """E5: rule's effective predicate (any JSR) overlaps with intended scope (step-0 JSR)."""
    effective = parse("opcode_at_AX == JSR")
    scope = parse("opcode_at_AX == JSR AND step_index == 0")
    assert overlaps(effective, scope)


def test_overlap_byte_value_compatible():
    """0xF8 overlaps with byte_value.lo_nibble == 0x8."""
    assert overlaps(parse("byte_value == 0xF8"), parse("byte_value.lo_nibble == 0x8"))


def test_overlap_byte_value_incompatible():
    """0xF8's lo nibble is 0x8, NOT 0x0, so these don't overlap."""
    assert not overlaps(parse("byte_value == 0xF8"), parse("byte_value.lo_nibble == 0x0"))


# === Cross-family: marker rows vs byte rows are mutually exclusive ===


def test_mark_eq_contradicts_is_byte():
    """Non-NONE marker rows are not byte rows (residual-tagging invariant)."""
    for role in ("SP", "AX", "PC", "BP", "MEM", "STACK0", "SE"):
        assert not satisfiable(parse(f"mark == {role} AND is_byte")), role


def test_mark_in_set_contradicts_is_byte():
    """A set of non-NONE marker roles is disjoint from is_byte."""
    assert not satisfiable(parse("mark in {AX, MEM} AND is_byte"))


def test_mark_none_does_not_contradict_is_byte():
    """``mark == NONE`` is the byte-row marker; satisfiable with is_byte."""
    assert satisfiable(parse("mark == NONE AND is_byte"))


def test_mark_in_set_with_none_does_not_contradict_is_byte():
    """A set including NONE remains satisfiable with is_byte."""
    assert satisfiable(parse("mark in {NONE, AX} AND is_byte"))


def test_opcode_byte_lo_byte_index_0_disjoint_from_mark_mem():
    """The textbook OPCODE_BYTE_LO read at byte-row 0 must NOT overlap
    with ADDR_B0_LO's ``mark == MEM`` semantics (cross-family disjoint)."""
    eff = parse("is_byte AND byte_index == 0")
    addr_b0_lo_sem = parse("mark == MEM")
    assert not overlaps(eff, addr_b0_lo_sem)
