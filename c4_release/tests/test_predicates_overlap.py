"""S-3: predicate satisfiability + overlap tests."""
import pytest
from neural_vm.verification.predicates import (
    is_tautology,
    overlaps,
    parse,
    satisfiable,
    strictly_refines,
)


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


# === is_tautology (Improvement A — dim-alias verifier TEMP umbrella) ===


def test_temp_umbrella_semantics_is_tautology():
    """``is_byte OR NOT is_byte`` is the production umbrella semantics
    declared on TEMP, TEMP_PREV_STEP, OUTPUT_BYTE_*, STACK0_BYTE*.
    The verifier MUST detect it as a tautology so those slots can be
    skipped as alias parents."""
    assert is_tautology(parse("is_byte OR NOT is_byte"))


def test_marker_atom_is_not_tautology():
    """A non-trivial constraint like ``mark == AX`` is NOT a tautology
    — there exist positions where it fails. The verifier must keep
    flagging it as a real alias parent."""
    assert not is_tautology(parse("mark == AX"))


def test_disjunction_of_non_complementary_atoms_is_not_tautology():
    """``mark == AX OR mark == MEM`` excludes other marker rows and
    byte rows; not every position satisfies it."""
    assert not is_tautology(parse("mark == AX OR mark == MEM"))


def test_three_way_tautology_with_complementary_pair():
    """A predicate is a tautology when at least one pair of disjuncts
    is the complement of another — the verifier's solver must spot
    this even when an extra disjunct is mixed in."""
    assert is_tautology(parse("mark == AX OR is_byte OR NOT is_byte"))


# === strictly_refines (Improvement B — sub-bank discriminator) ===


def test_op_lea_strictly_refines_opcode_base_same_extent():
    """The textbook same-extent parent/child: OP_LEA's semantics adds
    an ``opcode_at_AX == LEA`` atom on top of OPCODE_BASE's
    ``mark == AX`` disjunct. The verifier must classify this as a
    sub-bank refinement (suppress) — both occupy slot 262 size 1."""
    op_lea = parse("mark == AX AND opcode_at_AX == LEA")
    opcode_base = parse("mark == AX OR (is_byte AND byte_index == 0)")
    assert strictly_refines(op_lea, opcode_base)
    # The reverse direction must NOT hold — parent does not refine child.
    assert not strictly_refines(opcode_base, op_lea)


def test_addr_b0_lo_does_not_refine_opcode_byte_lo_textbook_alias():
    """ADDR_B0_LO ``mark == MEM`` equals one disjunct of OPCODE_BYTE_LO
    ``mark == MEM OR (is_byte AND byte_index == 0)`` VERBATIM — no
    added atoms, so it is NOT a refinement. The verifier must keep
    this textbook alias flagged."""
    addr_b0_lo = parse("mark == MEM")
    opcode_byte_lo = parse("mark == MEM OR (is_byte AND byte_index == 0)")
    assert not strictly_refines(addr_b0_lo, opcode_byte_lo)
    assert not strictly_refines(opcode_byte_lo, addr_b0_lo)


def test_op_lev_strictly_refines_opcode_flags_different_extent():
    """The original strict-containment case still works: OP_LEV (1 wide
    at byte 270) sits inside OPCODE_FLAGS (34 wide at 262..295) and its
    semantics adds ``opcode_at_AX == LEV``."""
    op_lev = parse("mark == AX AND opcode_at_AX == LEV")
    opcode_flags = parse("mark == AX OR (is_byte AND byte_index == 0)")
    assert strictly_refines(op_lev, opcode_flags)


def test_identical_semantics_is_not_a_refinement():
    """Two slots that share the SAME semantics (e.g. OP_OR and
    OPCODE_FLAGS both declared with the umbrella
    ``mark == AX OR ...``) are NOT in a parent/child refinement
    relation — they're declared the same way and must remain visible
    to the alias verifier."""
    a = parse("mark == AX OR (is_byte AND byte_index == 0)")
    b = parse("mark == AX OR (is_byte AND byte_index == 0)")
    assert not strictly_refines(a, b)
    assert not strictly_refines(b, a)
