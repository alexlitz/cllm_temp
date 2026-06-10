"""Tests for :mod:`neural_vm.unified_compiler.dim_alias_verifier`.

The verifier catches a class of bug invisible to the existing scope
checker: an FFNRule reading an aliased dim (one whose byte range is
shared with another semantically distinct dim) at a position where the
alias could carry a different value.

The textbook example: ``OPCODE_BYTE_LO`` (slot 12) shares its byte
range with ``ADDR_B0_LO``. At ``mark == MEM`` positions the slot
carries the address byte; at ``is_byte AND byte_index == 0`` positions
it can carry the opcode byte. A rule reading ``OPCODE_BYTE_LO+k`` whose
effective predicate admits ``mark == MEM`` will silently read the
address byte instead.
"""

from __future__ import annotations

import pytest

from neural_vm.dim_registry import DimRegistry, build_default_registry
from neural_vm.unified_compiler.ir import FFNOp, FFNRule
from neural_vm.unified_compiler.dim_alias_verifier import (
    AliasGroup,
    AliasViolation,
    enumerate_dim_aliases,
    format_violations,
    verify_dim_aliases,
    verify_dim_aliases_for_ops,
)


class _FakeOp:
    """Minimal stand-in for an Operation; the verifier only reads
    ``.name`` and ``.compiler_ir``."""

    def __init__(self, rules, name="fake_op"):
        self.name = name
        self.compiler_ir = FFNOp(rules=list(rules))


# ---------------------------------------------------------------------------
# 1) enumerate_dim_aliases on a small synthetic registry
# ---------------------------------------------------------------------------


def _make_minimal_registry() -> DimRegistry:
    """A small registry that mirrors the production OPCODE_BYTE_LO/
    ADDR_B0_LO conflict so the verifier can be tested in isolation."""
    reg = DimRegistry(d_model=64)
    # MARK_AX / MARK_MEM atomic markers (used by FFNRule conditions).
    reg.alloc("MARK_MEM", 0, 1, "MEM marker", semantics="mark == MEM")
    reg.alloc("MARK_AX", 1, 1, "AX marker", semantics="mark == AX")
    # Aliased pair: ADDR_B0_LO (mark == MEM) and OPCODE_BYTE_LO
    # (mark == MEM OR (is_byte AND byte_index == 0)) share slot [12..28).
    reg.alloc(
        "ADDR_B0_LO", 12, 16, "address byte 0 lo nibble",
        semantics="mark == MEM",
    )
    # NOTE: DimRegistry.alloc rejects exact duplicate slot names but
    # tolerates overlap on different names if `check_overlaps` isn't
    # called. We need to bypass alloc()'s start-bound check on
    # collisions — simply instantiate a DimSlot directly and inject.
    from neural_vm.dim_registry import DimSlot
    reg.slots["OPCODE_BYTE_LO"] = DimSlot(
        name="OPCODE_BYTE_LO",
        start=12,
        size=16,
        desc="opcode byte lo nibble",
        semantics="mark == MEM OR (is_byte AND byte_index == 0)",
    )
    # A simple output dim and harmless gating dim.
    reg.alloc("OUT", 32, 1, "out", semantics="mark == AX")
    return reg


def test_enumerate_finds_opcode_byte_alias_pair():
    reg = _make_minimal_registry()
    groups = enumerate_dim_aliases(reg)
    matching = [
        g for g in groups
        if set(g.members) == {"ADDR_B0_LO", "OPCODE_BYTE_LO"}
    ]
    assert len(matching) == 1, f"expected 1 group, got {groups!r}"
    g = matching[0]
    assert g.start == 12
    assert g.end == 28
    assert g.size == 16


def test_enumerate_on_production_registry_includes_opcode_byte_lo():
    """Regression: the production registry MUST present
    OPCODE_BYTE_LO/ADDR_B0_LO as one alias group at slot 12."""
    reg = build_default_registry()
    groups = enumerate_dim_aliases(reg)
    matching = [
        g for g in groups
        if "OPCODE_BYTE_LO" in g.members and "ADDR_B0_LO" in g.members
    ]
    assert matching, (
        "production registry must expose OPCODE_BYTE_LO/ADDR_B0_LO alias"
    )
    assert matching[0].start == 12
    assert matching[0].end == 28


# ---------------------------------------------------------------------------
# 2) verify_dim_aliases — the core check
# ---------------------------------------------------------------------------


def test_rule_reading_opcode_byte_at_mem_marker_is_flagged():
    """The textbook bug: a rule reads OPCODE_BYTE_LO+k but its effective
    predicate admits ``mark == MEM`` positions where the slot carries
    the address byte. The verifier MUST flag this."""
    reg = _make_minimal_registry()
    bad_rule = FFNRule.constant_write(
        conditions=(
            ("MARK_MEM", 1.0),    # fires at MEM marker rows
            ("OPCODE_BYTE_LO+5", 1.0),  # reads aliased dim
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="bad_read",
    )
    op = _FakeOp([bad_rule], name="op_with_bad_read")
    violations = verify_dim_aliases(op, reg)
    assert any(
        v.read_dim == "OPCODE_BYTE_LO"
        and v.conflicting_alias == "ADDR_B0_LO"
        for v in violations
    ), f"expected OPCODE_BYTE_LO/ADDR_B0_LO violation, got {violations!r}"


def test_rule_reading_opcode_byte_with_mem_excluded_is_safe():
    """A rule that explicitly blocks ``mark == MEM`` via a hard
    NOT-blocker on MARK_MEM has an effective predicate disjoint from
    ADDR_B0_LO's semantics. No violation should be reported."""
    reg = _make_minimal_registry()
    safe_rule = FFNRule.constant_write(
        conditions=(
            ("MARK_AX", 1.0),
            ("MARK_MEM", -1e9),  # hard blocker → NOT (mark == MEM)
            ("OPCODE_BYTE_LO+5", 1.0),
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="safe_read",
    )
    op = _FakeOp([safe_rule], name="op_with_safe_read")
    violations = verify_dim_aliases(op, reg)
    flagged = [
        v for v in violations
        if v.read_dim == "OPCODE_BYTE_LO"
        and v.conflicting_alias == "ADDR_B0_LO"
    ]
    assert not flagged, (
        f"safe rule should not be flagged; got {flagged!r}"
    )


def test_rule_without_aliased_reads_yields_no_violations():
    reg = _make_minimal_registry()
    rule = FFNRule.constant_write(
        conditions=(("MARK_AX", 1.0),),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="no_alias_read",
    )
    op = _FakeOp([rule], name="op_clean")
    assert verify_dim_aliases(op, reg) == []


def test_colocated_alias_is_not_flagged():
    """Parent/child siblings whose semantics ENTAIL each other describe
    the SAME physical signal — not an aliasing bug. The verifier's
    ``skip_colocated`` (default True) filters those out."""
    reg = DimRegistry(d_model=32)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    # Parent + child at the same slot — child's semantics entails parent's.
    reg.alloc(
        "OPCODE_FLAGS", 4, 16,
        "opcode one-hot bank",
        semantics="mark == AX",
    )
    from neural_vm.dim_registry import DimSlot
    reg.slots["OP_LEV"] = DimSlot(
        name="OP_LEV",
        start=4,
        size=1,
        desc="LEV flag (cell of OPCODE_FLAGS)",
        semantics="mark == AX AND opcode_at_AX == LEV",
    )
    reg.alloc("OUT", 24, 1, "out", semantics="mark == AX")
    rule = FFNRule.constant_write(
        conditions=(("MARK_AX", 1.0), ("OP_LEV+0", 1.0)),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="lev_dispatch",
    )
    op = _FakeOp([rule], name="lev_op")
    # With co-location filtering on (default), no violation.
    assert verify_dim_aliases(op, reg) == []
    # With co-location filtering OFF, the parent/child overlap surfaces.
    violations = verify_dim_aliases(
        op, reg, skip_colocated_subbank=False,
    )
    assert any(
        v.read_dim == "OP_LEV" and v.conflicting_alias == "OPCODE_FLAGS"
        for v in violations
    )


def test_format_violations_handles_empty_and_populated():
    assert format_violations([]) == "no dim-alias violations"
    v = AliasViolation(
        op_name="op_a",
        rule_name="r1",
        read_dim="OPCODE_BYTE_LO",
        read_offset=5,
        conflicting_alias="ADDR_B0_LO",
        conflicting_semantics="mark == MEM",
        effective_predicate="mark == MEM",
    )
    text = format_violations([v])
    assert "op_a" in text
    assert "OPCODE_BYTE_LO+5" in text
    assert "ADDR_B0_LO" in text


def test_verify_for_ops_dedupes_across_ops():
    """The same FFNRule appearing in multiple ops yields one violation
    per (rule_name, read_dim, conflicting_alias) tuple, not many."""
    reg = _make_minimal_registry()
    bad_rule = FFNRule.constant_write(
        conditions=(
            ("MARK_MEM", 1.0),
            ("OPCODE_BYTE_LO+5", 1.0),
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="dup_rule",
    )
    op_a = _FakeOp([bad_rule], name="op_a")
    op_b = _FakeOp([bad_rule], name="op_b")
    violations = verify_dim_aliases_for_ops([op_a, op_b], reg)
    # Both ops carry the same rule, so two violation entries are
    # expected (one per op_name) but no duplicate within an op.
    assert len(violations) == 2
    # Confirm de-dup by op+rule+dim+alias triple held: no triple repeats.
    keys = {
        (v.op_name, v.rule_name, v.read_dim, v.read_offset,
         v.conflicting_alias)
        for v in violations
    }
    assert len(keys) == len(violations)


def test_production_registry_does_not_crash():
    """Smoke: enumerate_dim_aliases on the production registry runs
    without error and finds non-empty alias groups."""
    reg = build_default_registry()
    groups = enumerate_dim_aliases(reg)
    assert groups, "production registry must have alias groups"
    # Every reported group must have >=2 members.
    for g in groups:
        assert isinstance(g, AliasGroup)
        assert len(g.members) >= 2


def test_verify_collects_violations_on_production_ops():
    """Smoke: running the verifier over the curated authored ops returns
    a deterministic, JSON-serialisable list of issues."""
    from neural_vm.unified_compiler.decl_verifier import (
        collect_all_authored_ops,
    )
    reg = build_default_registry()
    ops = collect_all_authored_ops()
    violations = verify_dim_aliases_for_ops(ops, reg)
    # Make sure every violation round-trips through as_dict (no exotic
    # values that would break a JSON encoder downstream).
    for v in violations:
        d = v.as_dict()
        assert d["kind"] == "dim_alias_read_without_disambiguation"
        assert "read_dim" in d
        assert "conflicting_alias" in d


# ---------------------------------------------------------------------------
# 3) Improvement A (tautology) + Improvement B (same-extent refinement)
# ---------------------------------------------------------------------------


def test_tautological_sibling_suppresses_violations():
    """A rule reading a dim whose alias parent carries a tautological
    semantics (``is_byte OR NOT is_byte`` — the TEMP umbrella pattern)
    must NOT be flagged: the parent declares the slot ambient and
    overlaps every effective predicate by construction."""
    reg = DimRegistry(d_model=64)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    # Real dim with a meaningful semantics:
    reg.alloc("AX_FULL_LO", 8, 16, "AX low half",
              semantics="mark == AX")
    # Tautological umbrella aliasing the same range (TEMP-style):
    from neural_vm.dim_registry import DimSlot
    reg.slots["TEMP_UMBRELLA"] = DimSlot(
        name="TEMP_UMBRELLA",
        start=8, size=16,
        desc="scratch (ambient)",
        semantics="is_byte OR NOT is_byte",
    )
    reg.alloc("OUT", 32, 1, "out", semantics="mark == AX")
    rule = FFNRule.constant_write(
        conditions=(("MARK_AX", 1.0), ("AX_FULL_LO+5", 1.0)),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="reads_ax_full_against_temp_umbrella",
    )
    op = _FakeOp([rule], name="op_temp_umbrella")
    # With tautology skip on (default), no violation.
    assert verify_dim_aliases(op, reg) == []
    # With it OFF, the umbrella sibling surfaces as an alias.
    violations = verify_dim_aliases(
        op, reg, skip_tautological_siblings=False,
    )
    assert any(
        v.conflicting_alias == "TEMP_UMBRELLA" for v in violations
    ), f"expected TEMP_UMBRELLA flagged with skip off, got {violations!r}"


def test_same_extent_subbank_refinement_is_suppressed():
    """OPCODE_BASE/OP_LEA both occupy slot 262 size 1; OP_LEA's
    semantics strictly refines OPCODE_BASE's via ``opcode_at_AX == LEA``.
    Improvement B recognises this as a same-extent parent/child
    sub-bank and suppresses the false positive."""
    reg = DimRegistry(d_model=64)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    # Parent + same-extent child:
    reg.alloc(
        "OPCODE_BASE_LIKE", 4, 1, "opcode base banded write",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    from neural_vm.dim_registry import DimSlot
    reg.slots["OP_LEA_LIKE"] = DimSlot(
        name="OP_LEA_LIKE",
        start=4, size=1,
        desc="LEA one-hot (same slot as OPCODE_BASE_LIKE)",
        semantics="mark == AX AND opcode_at_AX == LEA",
    )
    reg.alloc("OUT", 16, 1, "out", semantics="mark == AX")
    rule = FFNRule.constant_write(
        conditions=(("MARK_AX", 1.0), ("OP_LEA_LIKE+0", 1.0)),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="lea_dispatch",
    )
    op = _FakeOp([rule], name="lea_subbank")
    # With colocated-subbank skip on (default), no violation.
    assert verify_dim_aliases(op, reg) == []


# ---------------------------------------------------------------------------
# 4) Improvement C (2026-06-10): opcode_in_step disjointness
# ---------------------------------------------------------------------------


def _make_opcode_aliasing_registry() -> DimRegistry:
    """Registry with two slots aliased at the same byte range whose
    semantics are identical (both ``mark == AX``) but whose opcode
    owners are disjoint — AX_CARRY_LO style vs POST_PRTF_SP_LO style.

    Also includes ``OP_ADD`` and ``OP_PRTF`` opcode-flag slots so rules
    can carry positive OP_<X> references for the disjointness derivation.
    """
    reg = DimRegistry(d_model=128)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    # OP_ADD-like flag.
    reg.alloc(
        "OP_ADD", 4, 1, "ADD opcode flag",
        semantics="mark == AX AND opcode_at_AX == ADD",
    )
    # OP_PRTF-like flag (PRTF is shorthand for the print tool-call;
    # using "PUTCHAR" as an in-registry stand-in to avoid colliding with
    # the production registry's missing PRTF opcode atom).
    reg.alloc(
        "OP_PUTCHAR", 5, 1, "PUTCHAR opcode flag",
        semantics="mark == AX AND opcode_at_AX == PUTCHAR",
    )
    # Aliased pair at slot 32 size 16 — identical semantics, distinct
    # opcode owners (the test wires AX_CARRY_LO_LIKE to opcode-derived
    # rules and asserts POST_PRTF_SP_LO_LIKE is in the static owner table
    # by name).
    reg.alloc(
        "AX_CARRY_LO_LIKE", 32, 16, "AX carry lo nibble",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    from neural_vm.dim_registry import DimSlot
    reg.slots["POST_PRTF_SP_LO"] = DimSlot(
        name="POST_PRTF_SP_LO",
        start=32, size=16,
        desc="post-PRTF SP lo (aliases AX_CARRY_LO_LIKE)",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    reg.alloc("OUT", 64, 1, "out", semantics="mark == AX")
    return reg


def test_opcode_in_step_disjoint_suppresses_violation():
    """A rule reading AX_CARRY_LO_LIKE under an OP_ADD positive
    condition has opcode set {ADD}. The sibling POST_PRTF_SP_LO carries
    a static owner set {PRTF} from ``_SLOT_OPCODE_OWNERS``. Disjoint —
    skip the violation."""
    reg = _make_opcode_aliasing_registry()
    rule = FFNRule.constant_write(
        conditions=(
            ("MARK_AX", 1.0),
            ("OP_ADD+0", 1.0),
            ("AX_CARRY_LO_LIKE+5", 1.0),
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="add_reads_ax_carry",
    )
    op = _FakeOp([rule], name="op_add_under_post_prtf_alias")
    # Default: opcode_in_step disjointness on -> no violation.
    assert verify_dim_aliases(op, reg) == []
    # Disabled: violation surfaces (the semantics-only check sees
    # overlap because both slots admit mark == AX).
    violations = verify_dim_aliases(
        op, reg, skip_opcode_in_step_disjoint=False,
    )
    assert any(
        v.read_dim == "AX_CARRY_LO_LIKE"
        and v.conflicting_alias == "POST_PRTF_SP_LO"
        for v in violations
    ), f"with disjointness off, expected violation, got {violations!r}"


def test_opcode_in_step_overlapping_does_not_suppress():
    """A rule reading the same slot under an OP_PUTCHAR positive
    condition has opcode set {PUTCHAR}. The sibling POST_PRTF_SP_LO's
    static owner set {PRTF} is disjoint from {PUTCHAR} — but a rule
    that mentions BOTH OP_ADD and OP_PUTCHAR widens the set. This test
    exercises the negative: a rule with opcode set {PRTF}-overlapping
    must NOT be suppressed.

    Since the test registry uses PUTCHAR as the in-registry stand-in
    (the predicate DSL accepts PUTCHAR as a valid opcode name), we
    extend ``_SLOT_OPCODE_OWNERS`` for the duration of this test to
    include PUTCHAR. Restoring afterwards keeps the table immutable in
    user code.
    """
    from neural_vm.unified_compiler import dim_alias_verifier as _dav
    reg = _make_opcode_aliasing_registry()
    # Override the static table: POST_PRTF_SP_LO is owned by {PUTCHAR}
    # in this test world, matching the rule's opcode set -> the
    # disjointness check should NOT suppress.
    saved = _dav._SLOT_OPCODE_OWNERS.get("POST_PRTF_SP_LO")
    _dav._SLOT_OPCODE_OWNERS["POST_PRTF_SP_LO"] = frozenset({"PUTCHAR"})
    try:
        rule = FFNRule.constant_write(
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_PUTCHAR+0", 1.0),
                ("AX_CARRY_LO_LIKE+5", 1.0),
            ),
            threshold=0.5,
            writes=(("OUT", 1.0),),
            name="putchar_reads_ax_carry",
        )
        op = _FakeOp([rule], name="op_putchar_overlap")
        violations = verify_dim_aliases(op, reg)
        assert any(
            v.read_dim == "AX_CARRY_LO_LIKE"
            and v.conflicting_alias == "POST_PRTF_SP_LO"
            for v in violations
        ), (
            "Rule and sibling share opcode {PUTCHAR}; disjointness must "
            f"NOT suppress; got {violations!r}"
        )
    finally:
        if saved is None:
            _dav._SLOT_OPCODE_OWNERS.pop("POST_PRTF_SP_LO", None)
        else:
            _dav._SLOT_OPCODE_OWNERS["POST_PRTF_SP_LO"] = saved


def test_opcode_in_step_unknown_rule_set_keeps_violation():
    """When the rule carries NO positive OP_<X> reference, its opcode
    set is ``None`` (unknown) — the disjointness check must NOT
    suppress (conservative)."""
    reg = _make_opcode_aliasing_registry()
    rule = FFNRule.constant_write(
        conditions=(
            ("MARK_AX", 1.0),
            ("AX_CARRY_LO_LIKE+5", 1.0),
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="reads_ax_carry_without_opcode",
    )
    op = _FakeOp([rule], name="op_no_opcode_context")
    violations = verify_dim_aliases(op, reg)
    assert any(
        v.read_dim == "AX_CARRY_LO_LIKE"
        and v.conflicting_alias == "POST_PRTF_SP_LO"
        for v in violations
    ), (
        "Rule with unknown opcode context must NOT be suppressed; "
        f"got {violations!r}"
    )


def test_same_extent_alias_without_refinement_stays_flagged():
    """The textbook OPCODE_BYTE_LO/ADDR_B0_LO pair has same extent AND
    child semantics is a subset of parent's — but the child equals one
    parent disjunct VERBATIM (no added atoms). Improvement B must NOT
    suppress this; it remains the textbook real-bug case."""
    reg = _make_minimal_registry()
    # The minimal registry already creates OPCODE_BYTE_LO/ADDR_B0_LO
    # at slot 12 size 16. Reproduce the textbook bad read.
    bad_rule = FFNRule.constant_write(
        conditions=(
            ("MARK_MEM", 1.0),
            ("OPCODE_BYTE_LO+5", 1.0),
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="bad_read_post_improvement_b",
    )
    op = _FakeOp([bad_rule], name="op_textbook_alias")
    violations = verify_dim_aliases(op, reg)
    assert any(
        v.read_dim == "OPCODE_BYTE_LO"
        and v.conflicting_alias == "ADDR_B0_LO"
        for v in violations
    ), (
        "Improvement B must not suppress textbook OPCODE_BYTE_LO/"
        "ADDR_B0_LO alias (no semantic refinement, just verbatim "
        f"disjunct equality); got {violations!r}"
    )
