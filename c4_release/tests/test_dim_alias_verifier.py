"""Tests for :mod:`neural_vm.verification.dim_alias_verifier`.

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
from neural_vm.verification.dim_alias_verifier import (
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
    # Improvement G (role-contained-in-sibling) ALSO recognises OP_LEV as
    # the narrower role inside OPCODE_FLAGS and suppresses independently,
    # so disable it here to isolate the colocated-subbank path under test.
    violations = verify_dim_aliases(
        op, reg,
        skip_colocated_subbank=False,
        skip_role_contained_in_sibling=False,
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
    from neural_vm.verification.decl_verifier import (
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
    from neural_vm.verification import dim_alias_verifier as _dav
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


# ---------------------------------------------------------------------------
# 5) Improvement D/E (2026-06-10): expanded opcode_in_step extraction
# ---------------------------------------------------------------------------


def _make_extended_opcode_aliasing_registry() -> DimRegistry:
    """Registry with an OP_OR-like flag whose semantics is the DSL-
    keyword fallback (no opcode atom) AND an AX_CARRY_LO_LIKE/
    POST_PRTF_SP_LO alias pair, plus a CMP_GROUP_LIKE non-OP_ flag with
    explicit opcode-atom semantics for Phase 2 coverage."""
    reg = DimRegistry(d_model=128)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    reg.alloc(
        "MARK_SE_ONLY", 1, 1, "STEP_END marker",
        semantics="mark == SE",
    )
    # OP_OR_LIKE has the DSL-keyword fallback semantics (position-only,
    # no opcode atom). The literal fallback in _rule_opcode_in_step_set
    # must recover the "OR" role from the OP_<X> suffix.
    reg.alloc(
        "OP_OR", 4, 1, "OR opcode flag (DSL-keyword fallback)",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    # CMP_GROUP_LIKE has semantics with an explicit opcode_at_AX atom set.
    # Phase 2 path should pick it up via _opcodes_from_semantics fallback
    # inside _slot_opcode_in_step_set.
    reg.alloc(
        "CMP_GROUP_LIKE", 5, 1, "Comparison-group flag",
        semantics="mark == AX AND opcode_at_AX in {EQ, NE, LT}",
    )
    # Aliased pair at slot 32 size 16, identical semantics, distinct
    # opcode owners (AX_CARRY_LO_LIKE: not in static table; the
    # production registry's AX_CARRY_LO IS, but the test uses a
    # synthetic name to isolate the literal-fallback path).
    reg.alloc(
        "AX_CARRY_LO_LIKE", 32, 16, "AX carry lo nibble (test stub)",
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


def test_op_or_literal_fallback_pins_opcode_for_dsl_keyword():
    """Improvement D: a rule with ``gate=OP_OR`` (a DSL-keyword opcode
    whose semantics is position-only) must recover ``{OR}`` from the
    dim-name literal so disjointness with ``POST_PRTF_SP_LO`` (owners
    ``{PRTF}``) fires."""
    reg = _make_extended_opcode_aliasing_registry()
    rule = FFNRule.gated_write(
        conditions=(
            ("MARK_SE_ONLY", 1.0),
            ("AX_CARRY_LO_LIKE+5", 1.0),
        ),
        threshold=0.5,
        gate="OP_OR+0",
        writes=(("OUT", 1.0),),
        name="l10_bitwise_or_under_dsl_keyword",
    )
    op = _FakeOp([rule], name="op_l10_bitwise_or_like")
    # Default: opcode_in_step disjointness ON -> {OR} disjoint from
    # POST_PRTF_SP_LO {PRTF} -> no violation.
    assert verify_dim_aliases(op, reg) == [], (
        "OP_OR literal fallback should pin rule opcode to {OR}; "
        "disjoint from {PRTF} -> suppression must fire."
    )


def test_cmp_group_like_phase2_intersection_pins_opcodes():
    """Improvement E: a rule with NO positive OP_<X> reference but a
    gate on a CMP_GROUP-like slot (semantics encodes opcode_at_AX in
    {EQ, NE, LT}) must recover ``{EQ, NE, LT}`` via Phase 2 intersection,
    and the disjointness with POST_PRTF_SP_LO's ``{PRTF}`` owner set must
    suppress."""
    reg = _make_extended_opcode_aliasing_registry()
    rule = FFNRule.gated_write(
        conditions=(
            ("MARK_SE_ONLY", 1.0),
            ("AX_CARRY_LO_LIKE+3", 1.0),
        ),
        threshold=0.5,
        gate="CMP_GROUP_LIKE+0",
        writes=(("OUT", 1.0),),
        name="l9_cmp_under_group_flag",
    )
    op = _FakeOp([rule], name="op_l9_cmp_under_group_like")
    assert verify_dim_aliases(op, reg) == [], (
        "Phase 2 intersection should pin rule opcode to a subset of "
        "{EQ, NE, LT}; disjoint from {PRTF} -> suppression must fire."
    )


def test_phase2_intersection_collapses_to_empty_returns_none():
    """When Phase 2 intersection collapses to the empty set (mutually
    exclusive positive references), the rule's opcode set must be
    treated as UNKNOWN rather than vacuously disjoint. The conservative
    choice: keep the violation."""
    from neural_vm.verification import dim_alias_verifier as _dav
    reg = _make_extended_opcode_aliasing_registry()
    # Add a second non-OP_ slot whose owners disjoint from CMP_GROUP_LIKE
    # owners. Borrow the static table for a deterministic test.
    saved = _dav._SLOT_OPCODE_OWNERS.get("MARK_AX")
    _dav._SLOT_OPCODE_OWNERS["MARK_AX"] = frozenset({"PRTF"})
    try:
        rule = FFNRule.gated_write(
            conditions=(
                # MARK_AX -> {PRTF} (override above)
                ("MARK_AX", 1.0),
                ("AX_CARRY_LO_LIKE+3", 1.0),
            ),
            threshold=0.5,
            # CMP_GROUP_LIKE -> {EQ, NE, LT}; intersection with {PRTF} = empty
            gate="CMP_GROUP_LIKE+0",
            writes=(("OUT", 1.0),),
            name="phase2_empty_intersection",
        )
        op = _FakeOp([rule], name="op_phase2_empty_intersection")
        # Phase 2 collapses to empty -> opcode set unknown -> violation stands.
        violations = verify_dim_aliases(op, reg)
        assert any(
            v.read_dim == "AX_CARRY_LO_LIKE"
            and v.conflicting_alias == "POST_PRTF_SP_LO"
            for v in violations
        ), (
            "Phase 2 empty intersection must NOT vacuously suppress; "
            f"got {violations!r}"
        )
    finally:
        if saved is None:
            _dav._SLOT_OPCODE_OWNERS.pop("MARK_AX", None)
        else:
            _dav._SLOT_OPCODE_OWNERS["MARK_AX"] = saved


def test_op_x_phase_takes_precedence_over_phase2():
    """When a rule has BOTH an OP_<X> positive reference AND non-OP_
    positive references (e.g. L10 ALU rules with positive AX_CARRY_*
    conditions), the rule's opcode set must come from Phase 1 (the
    OP_<X> union), NOT Phase 2's intersection of the broader non-OP
    refs. This preserves the historical narrowing — the rule's true
    opcode set is the gate, not the broader ALU+CMP fingerprint of
    AX_CARRY_*."""
    reg = _make_extended_opcode_aliasing_registry()
    rule = FFNRule.gated_write(
        conditions=(
            ("MARK_SE_ONLY", 1.0),
            # AX_CARRY_LO_LIKE has no static-table entry; Phase 2 would
            # ignore it. Adding the OP_OR gate (literal fallback {OR})
            # pins Phase 1's union to {OR}.
            ("AX_CARRY_LO_LIKE+5", 1.0),
        ),
        threshold=0.5,
        gate="OP_OR+0",
        writes=(("OUT", 1.0),),
        name="phase1_dominates_phase2",
    )
    op = _FakeOp([rule], name="op_phase1_dominates")
    assert verify_dim_aliases(op, reg) == [], (
        "Phase 1 OP_<X> pinning must dominate; {OR} disjoint from "
        "{PRTF} -> suppression must fire."
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


# ---------------------------------------------------------------------------
# 5) Improvement D (2026-06-10): phase_in_step disjointness
# ---------------------------------------------------------------------------


def _make_phase_aliasing_registry() -> DimRegistry:
    """Registry with a FETCH-phase slot and an EXEC-phase slot at the
    same byte range (FETCH_HI / DIV_STAGING analog). Both slots'
    semantics overlap (each admits ``is_byte AND byte_index == 0``);
    only the static ``_SLOT_PHASE_OWNERS`` table disambiguates them.
    """
    reg = DimRegistry(d_model=128)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    reg.alloc(
        "IS_BYTE", 1, 1, "byte-row marker",
        semantics="is_byte",
    )
    reg.alloc(
        "BYTE_INDEX_0", 2, 1, "byte_index == 0",
        semantics="byte_index == 0",
    )
    # FETCH_HI-like slot at byte 64 — semantics says PC or byte_index∈{0..3}.
    reg.alloc(
        "FETCH_HI", 64, 16, "FETCH_HI-like (FETCH-phase)",
        semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
    )
    # DIV_STAGING-like slot at the same byte range — semantics admits
    # mark == AX OR is_byte AND byte_index == 0; opcode owner is {DIV,MOD}
    # but DIV/MOD ⊂ all-opcodes (FETCH is opcode-universal), so the
    # opcode_in_step check CAN'T fire here.
    from neural_vm.dim_registry import DimSlot
    reg.slots["DIV_STAGING"] = DimSlot(
        name="DIV_STAGING",
        start=64, size=16,
        desc="DIV_STAGING-like (EXEC-phase)",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    reg.alloc("OUT", 96, 1, "out", semantics="mark == AX")
    return reg


def test_phase_in_step_fetch_vs_exec_suppresses_violation():
    """A rule that gates on ``FETCH_HI`` (FETCH-phase) reading FETCH_HI
    has its conflict-alias DIV_STAGING (EXEC-phase) silently suppressed.
    Opcode_in_step disjointness CAN'T do this because FETCH is
    opcode-universal — only phase_in_step disjointness sees it."""
    reg = _make_phase_aliasing_registry()
    rule = FFNRule.constant_write(
        conditions=(
            ("IS_BYTE", 1.0),
            ("BYTE_INDEX_0", 1.0),
            ("FETCH_HI+5", 1.0),
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="fetch_phase_reads_fetch_hi",
    )
    op = _FakeOp([rule], name="op_fetch_phase")
    # Default: phase disjointness on -> no violation.
    assert verify_dim_aliases(op, reg) == []
    # Disabled: the violation surfaces (semantics-only overlap holds).
    violations = verify_dim_aliases(
        op, reg, skip_phase_in_step_disjoint=False,
    )
    assert any(
        v.read_dim == "FETCH_HI"
        and v.conflicting_alias == "DIV_STAGING"
        for v in violations
    ), (
        "with phase disjointness off, expected FETCH_HI<->DIV_STAGING "
        f"violation, got {violations!r}"
    )


def test_phase_in_step_unknown_rule_set_keeps_violation():
    """When a rule carries NO positive reference to a phase-scoped slot,
    its phase set is ``None`` (unknown) — the disjointness check must
    NOT suppress (conservative).

    Note (Improvement G, 2026-06-10 scattered-sweep): NEGATIVE-weight
    references are blockers, not value reads — they are excluded from
    both the alias-check read set AND the phase-set derivation. To
    exercise "unknown phase" while still reading an aliased slot, we
    use an aliased slot that is NOT in ``_SLOT_PHASE_OWNERS`` (so its
    positive read does not pin a phase) but still shares a byte range
    with FETCH_HI / DIV_STAGING.
    """
    reg = _make_phase_aliasing_registry()
    # Add a non-phase-owned aliased slot at the same byte range
    # (64..80) — the rule's positive read of this slot doesn't pin a
    # phase, so the rule's phase set is None and disjointness must NOT
    # suppress the aliasing of this slot with DIV_STAGING / FETCH_HI.
    from neural_vm.dim_registry import DimSlot
    reg.slots["NON_PHASE_ALIAS"] = DimSlot(
        name="NON_PHASE_ALIAS",
        start=64, size=16,
        desc="Aliased slot not tracked by _SLOT_PHASE_OWNERS",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    rule_truly_unknown = FFNRule.constant_write(
        conditions=(
            ("IS_BYTE", 1.0),
            ("BYTE_INDEX_0", 1.0),
            ("NON_PHASE_ALIAS+5", 1.0),  # positive read, no phase pin
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="reads_non_phase_alias_unknown_phase_ctx",
    )
    op2 = _FakeOp([rule_truly_unknown], name="op_truly_no_phase_ctx")
    violations2 = verify_dim_aliases(op2, reg)
    # The rule reads NON_PHASE_ALIAS (aliased to FETCH_HI / DIV_STAGING).
    # Phase set is None (no positive phase-scoped reference). Phase
    # disjointness MUST NOT suppress.
    assert any(
        v.read_dim == "NON_PHASE_ALIAS"
        and v.conflicting_alias in {"FETCH_HI", "DIV_STAGING"}
        for v in violations2
    ), (
        "Rule with unknown phase context must NOT be suppressed; "
        f"got {violations2!r}"
    )


def test_phase_in_step_overlapping_phases_do_not_suppress():
    """A rule reading FETCH_HI whose aliased sibling is also FETCH-phase
    (e.g. IMM_STAGING) must NOT be suppressed — the phase sets overlap."""
    reg = _make_phase_aliasing_registry()
    # Add IMM_STAGING-like slot overlapping FETCH_HI; phase set
    # {FETCH, DECODE} matches FETCH_HI's set.
    from neural_vm.dim_registry import DimSlot
    reg.slots["IMM_STAGING"] = DimSlot(
        name="IMM_STAGING",
        start=64, size=16,
        desc="IMM_STAGING-like (FETCH-phase, intersects)",
        semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
    )
    rule = FFNRule.constant_write(
        conditions=(
            ("IS_BYTE", 1.0),
            ("BYTE_INDEX_0", 1.0),
            ("FETCH_HI+5", 1.0),
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="fetch_reads_fetch_hi_imm_overlap",
    )
    op = _FakeOp([rule], name="op_fetch_phase_overlap_imm")
    # These stubs share IDENTICAL semantics AND identical phase owners —
    # exactly the same-signal pattern Improvement M (equivalent-same-
    # signal) collapses. Disable M here to isolate the phase-disjointness
    # path under test (the point of this case is that phase disjointness
    # alone CANNOT fire for a same-phase pair).
    violations = verify_dim_aliases(
        op, reg, skip_equivalent_same_signal=False,
    )
    # FETCH_HI's owners {FETCH,DECODE} intersect IMM_STAGING's
    # {FETCH,DECODE}; phase disjointness CAN'T fire — violation must stand.
    assert any(
        v.read_dim == "FETCH_HI"
        and v.conflicting_alias == "IMM_STAGING"
        for v in violations
    ), (
        "Same-phase pair (both FETCH/DECODE) must NOT be suppressed; "
        f"got {violations!r}"
    )


# ---------------------------------------------------------------------------
# Scattered-sweep (2026-06-10) Improvements F, G, I tests
# ---------------------------------------------------------------------------


def test_partial_overlap_read_outside_overlap_byte_not_flagged():
    """Improvement F: FETCH_HI (bytes 436..452) and IMM_STAGING
    (bytes 448..464) share bytes 448..451 only. A rule reading
    FETCH_HI+0 lands at physical byte 436 — outside IMM_STAGING's
    range — no alias risk."""
    reg = DimRegistry(d_model=512)
    reg.alloc("MARK_PC", 0, 1, "PC marker", semantics="mark == PC")
    reg.alloc(
        "FETCH_HI", 436, 16, "fetch hi bus",
        semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
    )
    from neural_vm.dim_registry import DimSlot
    reg.slots["IMM_STAGING"] = DimSlot(
        name="IMM_STAGING", start=448, size=16,
        desc="immediate staging (partial overlap with FETCH_HI)",
        semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
    )
    reg.alloc("OUT", 200, 1, "out", semantics="mark == AX")
    # Read at offset 0 -> byte 436 -> NOT inside IMM_STAGING [448, 464).
    rule = FFNRule.constant_write(
        conditions=(("MARK_PC", 1.0), ("FETCH_HI+0", 1.0)),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="fetch_hi_low_offset",
    )
    op = _FakeOp([rule], name="op_fetch_hi_low")
    # No violation — read byte is outside the sibling's range.
    assert verify_dim_aliases(op, reg) == []


def test_negative_weight_read_is_skipped():
    """Improvement G: A negative-weight reference is a blocker, not a
    value read — it can only suppress firing, never cause incorrect
    activation on the aliased value. Skip it."""
    reg = _make_minimal_registry()
    # Reads OPCODE_BYTE_LO+5 NEGATIVELY (blocker).
    rule = FFNRule.constant_write(
        conditions=(
            ("MARK_MEM", 1.0),
            ("OPCODE_BYTE_LO+5", -100.0),  # blocker, not a value read
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="blocker_use_of_alias",
    )
    op = _FakeOp([rule], name="blocker_op")
    # No violation: the negative-weight read is a guard, not a value read.
    assert verify_dim_aliases(op, reg) == []


def test_displaced_ambient_slot_suppresses_violation():
    """Improvement I: A rule pinned to {PRTF} reading a slot whose
    sibling is ambient AX_FULL_HI-style (always alive) but in the
    ``_SLOT_DISPLACED_BY`` table with displacer ⊇ rule's opcode set
    is suppressed. At PRTF rows the byte carries the displacer's
    content, not the sibling's."""
    from neural_vm.verification import dim_alias_verifier as _dav
    reg = DimRegistry(d_model=128)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    # A slot pinned to PRTF (via static owner table entry below).
    reg.alloc(
        "LAST_WAS_THINKING_START_LIKE", 32, 1,
        "PRTF-state-machine flag",
        semantics="NOT is_byte",
    )
    # A wider ambient sibling that overlaps; semantics is opcode-
    # agnostic but the displaced-by table marks it as overlaid at PRTF.
    from neural_vm.dim_registry import DimSlot
    reg.slots["AX_FULL_HI_LIKE"] = DimSlot(
        name="AX_FULL_HI_LIKE", start=24, size=16,
        desc="ambient AX register hi nibble (PRTF-displaced)",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    reg.alloc("OUT", 64, 1, "out", semantics="mark == AX")

    saved_owner = _dav._SLOT_OPCODE_OWNERS.get(
        "LAST_WAS_THINKING_START_LIKE"
    )
    saved_displ = _dav._SLOT_DISPLACED_BY.get("AX_FULL_HI_LIKE")
    _dav._SLOT_OPCODE_OWNERS["LAST_WAS_THINKING_START_LIKE"] = \
        frozenset({"PRTF"})
    _dav._SLOT_DISPLACED_BY["AX_FULL_HI_LIKE"] = frozenset({"PRTF"})
    try:
        rule = FFNRule.constant_write(
            conditions=(
                ("MARK_AX", 1.0),
                ("LAST_WAS_THINKING_START_LIKE+0", 1.0),
            ),
            threshold=0.5,
            writes=(("OUT", 1.0),),
            name="prtf_reads_last_was_thinking_start",
        )
        op = _FakeOp([rule], name="op_displaced_ambient")
        # Default: displaced-by suppresses the violation.
        assert verify_dim_aliases(op, reg) == []
    finally:
        if saved_owner is None:
            _dav._SLOT_OPCODE_OWNERS.pop(
                "LAST_WAS_THINKING_START_LIKE", None,
            )
        else:
            _dav._SLOT_OPCODE_OWNERS[
                "LAST_WAS_THINKING_START_LIKE"
            ] = saved_owner
        if saved_displ is None:
            _dav._SLOT_DISPLACED_BY.pop("AX_FULL_HI_LIKE", None)
        else:
            _dav._SLOT_DISPLACED_BY["AX_FULL_HI_LIKE"] = saved_displ


# ---------------------------------------------------------------------------
# Raw-corpus closers (2026-06-11): Improvements G, K, L, M
# ---------------------------------------------------------------------------


def test_role_contained_in_sibling_suppresses_reverse_read():
    """Improvement G: a rule reading the NARROWER alias (ADDR_B0_LO,
    ``mark == MEM``) at a position the narrower alias owns is safe even
    though the WIDER sibling (OPCODE_BYTE_LO) co-claims a superset of
    positions. The read returns the read_dim's own value."""
    reg = _make_minimal_registry()
    rule = FFNRule.constant_write(
        conditions=(
            ("MARK_MEM", 1.0),
            ("ADDR_B0_LO+5", 1.0),  # reads the NARROWER alias
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="l9_bp_plus8_shift_like",
    )
    op = _FakeOp([rule], name="op_reverse_read")
    # Default: Improvement G suppresses (read_dim is the narrower role).
    assert verify_dim_aliases(op, reg) == []
    # Disabled: the overlap surfaces.
    violations = verify_dim_aliases(
        op, reg, skip_role_contained_in_sibling=False,
    )
    assert any(
        v.read_dim == "ADDR_B0_LO"
        and v.conflicting_alias == "OPCODE_BYTE_LO"
        for v in violations
    ), f"with G off, expected ADDR_B0_LO<->OPCODE_BYTE_LO, got {violations!r}"


def test_role_containment_does_not_suppress_forward_textbook_bug():
    """Improvement G must NOT suppress the textbook FORWARD bug: reading
    the WIDER alias (OPCODE_BYTE_LO) at ``mark == MEM`` where the slot
    actually carries the address byte. There read_dim is the wider slot
    so ``read_dim_sem`` does NOT entail ``sibling_sem`` and G stands
    down."""
    reg = _make_minimal_registry()
    bad = FFNRule.constant_write(
        conditions=(
            ("MARK_MEM", 1.0),
            ("OPCODE_BYTE_LO+5", 1.0),  # reads the WIDER alias
        ),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="forward_textbook_bug",
    )
    op = _FakeOp([bad], name="op_forward_bug")
    violations = verify_dim_aliases(op, reg)
    assert any(
        v.read_dim == "OPCODE_BYTE_LO"
        and v.conflicting_alias == "ADDR_B0_LO"
        for v in violations
    ), (
        "Improvement G must not suppress the textbook forward bug; "
        f"got {violations!r}"
    )


def test_eff_disjoint_from_read_dim_helper():
    """Improvement K (helper unit test): ``_eff_disjoint_from_read_dim``
    returns True iff the rule's eff fires ONLY at positions the read_dim
    does NOT claim (``eff |= NOT read_dim_sem``) — a deliberate alias-
    traversal of the sibling's content. Mirrors the L14 addr_key decode
    reading ADDR_B1_LO (``mark == MEM``) at NON-MEM byte rows
    (``eff == (is_byte OR NOT is_byte) AND NOT mark == MEM``).

    Tested at the helper level because ``effective_predicate``'s
    composition of positive-read semantics + hard blockers is hard to
    reproduce faithfully on a synthetic registry; the END-TO-END coverage
    for K is the raw-corpus ratchet
    (``test_raw_factory_corpus_has_zero_dim_alias_violations``), where the
    real L14 ``ADDR_B1_LO <-> OPCODE_BYTE_HI`` family is suppressed by K.
    """
    from neural_vm.verification.dim_alias_verifier import (
        _eff_disjoint_from_read_dim,
    )
    from neural_vm.verification.predicates import parse

    # L14-shaped eff: fires only at NON-MEM rows.
    eff = parse("(is_byte OR NOT is_byte) AND NOT mark == MEM")
    # ADDR_B1_LO's own zone is ``mark == MEM`` — disjoint from eff.
    assert _eff_disjoint_from_read_dim(eff, "mark == MEM") is True
    # The textbook FORWARD bug eff (``mark == MEM``) is INSIDE the
    # read_dim's zone — NOT disjoint, so K must stand down.
    eff_forward = parse("mark == MEM")
    assert _eff_disjoint_from_read_dim(eff_forward, "mark == MEM") is False
    # An eff that partly overlaps the zone is also not disjoint.
    eff_partial = parse("mark == MEM OR mark == AX")
    assert _eff_disjoint_from_read_dim(eff_partial, "mark == MEM") is False
    # No read semantics -> conservative False.
    assert _eff_disjoint_from_read_dim(eff, None) is False


def test_opcode_flag_subbank_cell_is_suppressed():
    """Improvement L: an opcode-flag CELL whose registry semantics is the
    DSL-keyword fallback (identical to the parent bank, e.g. OP_OR /
    OP_AND inside OPCODE_FLAGS) is a one-hot lane of the bank. Byte
    containment + logically-equivalent semantics => same physical signal;
    suppress."""
    reg = DimRegistry(d_model=64)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    # Parent bank.
    reg.alloc(
        "OPCODE_FLAGS_LIKE", 4, 16, "one-hot opcode bank",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    # Cell with the DSL-keyword fallback (identical sem to the bank).
    from neural_vm.dim_registry import DimSlot
    reg.slots["OP_OR_LIKE"] = DimSlot(
        name="OP_OR_LIKE", start=10, size=1,
        desc="OPCODE_FLAGS_LIKE[6] = OR active flag (DSL-keyword fallback)",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    reg.alloc("OUT", 24, 1, "out", semantics="mark == AX")
    rule = FFNRule.constant_write(
        conditions=(("MARK_AX", 1.0), ("OP_OR_LIKE+0", 1.0)),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="l10_bitwise_or_like",
    )
    op = _FakeOp([rule], name="op_opcode_cell")
    # Default: colocated-subbank (containment + equiv) suppresses.
    assert verify_dim_aliases(op, reg) == []
    # Disabled: the cell/bank overlap surfaces. Improvement M would ALSO
    # collapse this (equivalent sem, equal owners), so disable it too to
    # isolate the colocated-subbank containment path under test.
    violations = verify_dim_aliases(
        op, reg,
        skip_colocated_subbank=False,
        skip_equivalent_same_signal=False,
    )
    assert any(
        v.read_dim == "OP_OR_LIKE"
        and v.conflicting_alias == "OPCODE_FLAGS_LIKE"
        for v in violations
    ), f"with subbank skip off, expected cell/bank overlap, got {violations!r}"


def test_equivalent_same_signal_alias_is_suppressed():
    """Improvement M: two overlapping slots with logically-equivalent
    semantics and IDENTICAL owner sets are the same physical signal under
    two names (FETCH_HI <-> IMM_STAGING fetched-immediate bus). Suppress.
    """
    reg = DimRegistry(d_model=128)
    reg.alloc("MARK_PC", 0, 1, "PC marker", semantics="mark == PC")
    reg.alloc(
        "ALIAS_A", 64, 16, "signal A",
        semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
    )
    from neural_vm.dim_registry import DimSlot
    # Equal-extent equivalent-semantics alias (the IO_FORMAT_POS<->MEM_EXEC
    # equal-extent case the byte-containment branch deliberately skips).
    reg.slots["ALIAS_B"] = DimSlot(
        name="ALIAS_B", start=64, size=16,
        desc="signal B (deprecated alias of ALIAS_A)",
        semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
    )
    reg.alloc("OUT", 96, 1, "out", semantics="mark == AX")
    rule = FFNRule.constant_write(
        conditions=(("MARK_PC", 1.0), ("ALIAS_A+5", 1.0)),
        threshold=0.5,
        writes=(("OUT", 1.0),),
        name="reads_alias_a",
    )
    op = _FakeOp([rule], name="op_same_signal")
    # Default: Improvement M suppresses (equivalent sem, equal owners).
    assert verify_dim_aliases(op, reg) == []
    # Disabled: the overlap surfaces.
    violations = verify_dim_aliases(
        op, reg, skip_equivalent_same_signal=False,
    )
    assert any(
        v.read_dim == "ALIAS_A" and v.conflicting_alias == "ALIAS_B"
        for v in violations
    ), f"with M off, expected ALIAS_A<->ALIAS_B, got {violations!r}"


def test_equivalent_same_signal_does_not_collapse_distinct_owners():
    """Improvement M must NOT collapse two equivalent-semantics slots when
    their owner sets DIFFER (one pinned to {PRTF}, the other agnostic) —
    that asymmetry marks them as distinct time-shared signals
    (AX_CARRY_LO <-> POST_PRTF_SP_LO style)."""
    from neural_vm.verification import dim_alias_verifier as _dav
    reg = DimRegistry(d_model=128)
    reg.alloc("MARK_AX", 0, 1, "AX marker", semantics="mark == AX")
    reg.alloc(
        "CARRY_LIKE", 32, 16, "ALU carry (agnostic owners)",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    from neural_vm.dim_registry import DimSlot
    reg.slots["POST_PRTF_LIKE"] = DimSlot(
        name="POST_PRTF_LIKE", start=32, size=16,
        desc="post-PRTF save (PRTF-owned, equivalent sem to CARRY_LIKE)",
        semantics="mark == AX OR (is_byte AND byte_index == 0)",
    )
    reg.alloc("OUT", 64, 1, "out", semantics="mark == AX")
    saved = _dav._SLOT_OPCODE_OWNERS.get("POST_PRTF_LIKE")
    _dav._SLOT_OPCODE_OWNERS["POST_PRTF_LIKE"] = frozenset({"PRTF"})
    try:
        rule = FFNRule.constant_write(
            conditions=(("MARK_AX", 1.0), ("CARRY_LIKE+5", 1.0)),
            threshold=0.5,
            writes=(("OUT", 1.0),),
            name="reads_carry_like",
        )
        op = _FakeOp([rule], name="op_distinct_owner_alias")
        violations = verify_dim_aliases(op, reg)
        assert any(
            v.read_dim == "CARRY_LIKE"
            and v.conflicting_alias == "POST_PRTF_LIKE"
            for v in violations
        ), (
            "Improvement M must NOT collapse equivalent-sem slots with "
            f"differing owner sets; got {violations!r}"
        )
    finally:
        if saved is None:
            _dav._SLOT_OPCODE_OWNERS.pop("POST_PRTF_LIKE", None)
        else:
            _dav._SLOT_OPCODE_OWNERS["POST_PRTF_LIKE"] = saved
