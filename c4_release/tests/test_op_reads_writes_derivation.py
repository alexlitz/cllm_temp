"""Tests for ``Operation.reads`` / ``Operation.writes`` derivation infra.

Covers ``c4_release.neural_vm.verification.op_introspect``: the
``derive_op_reads_writes_from_rules`` walker, the
``assert_declared_matches_derived`` contract check, and the
``Operation.derive_reads_writes`` method.

The architectural goal (per ``docs/UNDECLARED_DIM_AUDIT_2026_06_09.md``):
eliminate hand-annotation drift by *deriving* the contract sets from the
declarative ``compiler_ir`` rule contents. These tests pin the derivation's
semantics so future migrations of the 162-op corpus can land
``derive_reads_writes_flag=True`` per-op without surprises.
"""

from __future__ import annotations

import warnings

import pytest

from c4_release.neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
)
from c4_release.neural_vm.verification.op_introspect import (
    DerivationMismatch,
    DerivedReadsWrites,
    assert_declared_matches_derived,
    derive_op_reads_writes_from_rules,
    derive_operation,
    normalize_declared_set,
)
from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    FFNRule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_bake(module, dim_positions, S):
    return None


def _build_synthetic_ffn_ir() -> CompilerIR:
    """Tiny CompilerIR with one FFNRule for unit testing the walker.

    Rule reads ``H0`` (condition) and ``H1`` (gate), writes ``OUTPUT_LO``
    and ``OUTPUT_HI``. The threshold/weights are irrelevant — derivation
    is purely structural.
    """
    rule = FFNRule.constant_write(
        conditions=(("H0", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO", 1.0), ("OUTPUT_HI", 1.0)),
        name="synthetic_rule",
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)
    return ir


def _build_synthetic_ffn_ir_with_gate() -> CompilerIR:
    """Same as above but with a gate, gate_terms, and an offset-suffix write."""
    rule = FFNRule.gated_write(
        conditions=(("MARK_PC", 1.0),),
        threshold=0.5,
        gate="OP_ADD",
        gate_terms=(("IS_BYTE", 1.0),),
        writes=(("ALU_LO+1", 1.0),),
        name="gated_synthetic",
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)
    return ir


# ---------------------------------------------------------------------------
# Pure unit tests (no layout build — fast)
# ---------------------------------------------------------------------------


def test_normalize_declared_set_strips_offset_suffix():
    """``+N`` and ``.*.N`` suffixes get stripped to the base dim name."""
    raw = {"OUTPUT_LO", "OUTPUT_LO+1", "OUTPUT_HI.*.-1", "H0+5"}
    assert normalize_declared_set(raw) == {"OUTPUT_LO", "OUTPUT_HI", "H0"}


def test_normalize_declared_set_handles_empty_and_none():
    """Empty / None inputs return an empty set without raising."""
    assert normalize_declared_set(set()) == set()
    assert normalize_declared_set(None) == set()


def test_normalize_declared_set_ignores_non_strings():
    """Non-string entries (defensive) are silently dropped."""
    assert normalize_declared_set({"H0", 42, None}) == {"H0"}


def test_derive_reads_writes_from_ffn_rule_conditions_and_writes():
    """Derivation surfaces conditions as reads and writes as writes."""
    op = Operation(
        name="synthetic",
        reads=set(),
        writes=set(),
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
    )

    derived = derive_op_reads_writes_from_rules(op)

    assert derived.ir_available is True
    assert derived.reads == {"H0"}
    assert derived.writes == {"OUTPUT_LO", "OUTPUT_HI"}


def test_derive_reads_writes_picks_up_gate_and_gate_terms():
    """Gate and gate_terms dims land in reads alongside conditions."""
    op = Operation(
        name="synthetic_gated",
        reads=set(),
        writes=set(),
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir_with_gate(),
    )

    derived = derive_op_reads_writes_from_rules(op)

    # conditions -> MARK_PC, gate -> OP_ADD, gate_terms -> IS_BYTE.
    assert derived.reads == {"MARK_PC", "OP_ADD", "IS_BYTE"}
    # ``ALU_LO+1`` strips to ``ALU_LO`` on the IR side because WriteTerm
    # carries the base name + offset separately.
    assert derived.writes == {"ALU_LO"}


def test_derive_reads_writes_skips_zero_weight_writes():
    """Explicit zero-weight writes do not contribute to writes (audit parity)."""
    rule = FFNRule.constant_write(
        conditions=(("H0", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO", 0.0), ("OUTPUT_HI", 1.0)),
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    op = Operation(
        name="zero_write",
        reads=set(),
        writes=set(),
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=ir,
    )

    derived = derive_op_reads_writes_from_rules(op)
    assert derived.writes == {"OUTPUT_HI"}


def test_derive_reads_writes_empty_ir_returns_empty_sets():
    """Op with no IR (legacy imperative bake) yields ``ir_available=False``."""
    op = Operation(
        name="no_ir",
        reads={"H0"},
        writes={"OUTPUT_LO"},
        kind="ffn",
        bake_fn=_noop_bake,
    )

    derived = derive_op_reads_writes_from_rules(op)

    assert derived.ir_available is False
    assert derived.reads == set()
    assert derived.writes == set()


def test_assert_declared_matches_derived_clean_when_declared_superset():
    """When declared >= derived, the mismatch is_clean and has no findings."""
    op = Operation(
        name="superset",
        # Over-declared: contains derived ({H0} / {OUTPUT_LO, OUTPUT_HI})
        # plus extras. Under default (non-strict) contract that's OK.
        reads={"H0", "H1"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "TEMP"},
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
    )

    m = assert_declared_matches_derived(op)
    assert m.undeclared_reads == ()
    assert m.undeclared_writes == ()
    # Over-declarations only reported under strict.
    assert m.over_declared_reads == ()
    assert m.over_declared_writes == ()


def test_assert_declared_matches_derived_strict_flags_over_declarations():
    """Strict mode reports over-declared names too."""
    op = Operation(
        name="strict_over",
        reads={"H0", "H1"},  # H1 not in derivation
        writes={"OUTPUT_LO", "OUTPUT_HI", "TEMP"},  # TEMP not in derivation
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
    )

    m = assert_declared_matches_derived(op, strict=True)
    assert m.undeclared_reads == ()
    assert m.undeclared_writes == ()
    assert m.over_declared_reads == ("H1",)
    assert m.over_declared_writes == ("TEMP",)
    assert m.is_clean is False


def test_assert_declared_matches_derived_surfaces_undeclared():
    """Under-declaration: derivation finds dims the contract doesn't list."""
    op = Operation(
        name="missing",
        reads=set(),  # missing H0 entirely
        writes={"OUTPUT_LO"},  # missing OUTPUT_HI
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
    )

    m = assert_declared_matches_derived(op)
    assert m.undeclared_reads == ("H0",)
    assert m.undeclared_writes == ("OUTPUT_HI",)
    assert m.has_undeclared is True
    assert m.is_clean is False


def test_assert_declared_matches_derived_strips_declared_suffixes():
    """Declared names with +N / .*.N suffixes are normalized before compare."""
    op = Operation(
        name="suffixed",
        reads={"H0+3"},          # +3 offset
        writes={"OUTPUT_LO.*.-1", "OUTPUT_HI+2"},
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
    )

    m = assert_declared_matches_derived(op)
    assert m.is_clean


def test_derive_operation_returns_copy_with_rewritten_sets():
    """``derive_operation`` returns a new Operation with derived reads/writes."""
    op = Operation(
        name="rewriteme",
        reads={"WRONG"},
        writes={"ALSO_WRONG"},
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
        phase=42,
        ffn_units_used=1,
    )

    rewritten = derive_operation(op)

    assert isinstance(rewritten, Operation)
    assert rewritten.name == "rewriteme"
    # Reads/writes replaced by derivation.
    assert rewritten.reads == {"H0"}
    assert rewritten.writes == {"OUTPUT_LO", "OUTPUT_HI"}
    # Other fields preserved.
    assert rewritten.phase == 42
    assert rewritten.ffn_units_used == 1
    assert rewritten.kind == "ffn"
    # Original is untouched.
    assert op.reads == {"WRONG"}
    assert op.writes == {"ALSO_WRONG"}


def test_derive_operation_rejects_non_operation():
    """Calling ``derive_operation`` on a non-Operation raises ``TypeError``."""
    with pytest.raises(TypeError):
        derive_operation({"name": "not-an-op"})


def test_operation_derive_reads_writes_method_delegates_to_introspect():
    """``Operation.derive_reads_writes()`` is the public method entry point."""
    op = Operation(
        name="method_test",
        reads={"WRONG"},
        writes={"WRONG"},
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
    )

    rewritten = op.derive_reads_writes()

    assert rewritten.reads == {"H0"}
    assert rewritten.writes == {"OUTPUT_LO", "OUTPUT_HI"}


def test_operation_derive_reads_writes_flag_default_false():
    """The migration opt-in flag defaults to False on every Operation."""
    op = Operation(
        name="default_flag",
        reads=set(),
        writes=set(),
        kind="ffn",
        bake_fn=_noop_bake,
    )
    assert op.derive_reads_writes_flag is False


def test_assert_declared_matches_derived_alias_canon_collapses_aliases():
    """When two names alias the same slot, ``alias_canon`` prevents false positives."""
    op = Operation(
        name="aliased",
        # Declares OUTPUT_HI but IR writes OUTPUT_HI_THIS_STEP (synthetic
        # alias for this test).
        reads={"H0"},
        writes={"OUTPUT_HI"},
        kind="ffn",
        bake_fn=_noop_bake,
        compiler_ir=_build_synthetic_ffn_ir(),
    )
    # Force OUTPUT_LO -> OUTPUT_HI canonicalization to test the mechanism.
    alias_canon = {"OUTPUT_LO": "OUTPUT_HI"}

    m = assert_declared_matches_derived(op, alias_canon=alias_canon)
    # Both OUTPUT_LO and OUTPUT_HI canonicalize to OUTPUT_HI; the declared
    # set only carries OUTPUT_HI; with the canon, undeclared_writes is empty.
    assert m.undeclared_writes == ()


# ---------------------------------------------------------------------------
# Layout-driven integration test (slower — builds the full layout once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _full_layout():
    """Real compiled layout for end-to-end derivation parity checks."""
    from c4_release.neural_vm.verification.decl_verifier import (
        _build_layout_only,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_layout_only(
            alu_mode="efficient",
            enable_conversational_io=False,
            enable_tool_calling=False,
            n_heads=8,
        )


def _find_op(layout, name: str):
    from c4_release.neural_vm.verification.dim_flow import (
        _walk_ops_with_layers,
    )

    for _, op in _walk_ops_with_layers(layout):
        if getattr(op, "name", None) == name:
            return op
    pytest.skip(f"op {name!r} not found in production layout")
    return None  # pragma: no cover


def test_layer2_mem_byte_flags_derivation_matches_audit(_full_layout):
    """``layer2_mem_byte_flags`` derivation lines up with the
    2026-06-09 audit findings exactly.

    Per the audit doc: undeclared_reads = {H2, H3, L1H4, L2H0};
    undeclared_writes = {} (already fixed in commit d756d9d8).
    """
    layout = _full_layout
    op = _find_op(layout, "layer2_mem_byte_flags")

    m = assert_declared_matches_derived(
        op,
        dim_positions=layout.dim_positions,
        dim_sizes=getattr(layout, "dim_sizes", {}),
    )

    assert set(m.undeclared_reads) == {"H2", "H3", "L1H4", "L2H0"}
    # Writes were brought in line by commit d756d9d8 — no drift remains.
    assert m.undeclared_writes == ()


def test_layer1_ffn_derivation_matches_audit(_full_layout):
    """``layer1_ffn`` undeclared reads = {L1H0, L1H1, L1H2, L1H4}."""
    layout = _full_layout
    op = _find_op(layout, "layer1_ffn")

    m = assert_declared_matches_derived(
        op,
        dim_positions=layout.dim_positions,
        dim_sizes=getattr(layout, "dim_sizes", {}),
    )

    assert set(m.undeclared_reads) == {"L1H0", "L1H1", "L1H2", "L1H4"}
    assert m.undeclared_writes == ()


def test_phase_a_ffn_derivation_matches_audit(_full_layout):
    """``phase_a_ffn`` undeclared reads = {H0..H4}; writes fully declared."""
    layout = _full_layout
    op = _find_op(layout, "phase_a_ffn")

    m = assert_declared_matches_derived(
        op,
        dim_positions=layout.dim_positions,
        dim_sizes=getattr(layout, "dim_sizes", {}),
    )

    assert set(m.undeclared_reads) == {"H0", "H1", "H2", "H3", "H4"}
    assert m.undeclared_writes == ()


def test_derive_operation_on_production_op_byte_identical_except_sets(
    _full_layout,
):
    """``derive_operation`` only mutates reads/writes; every other field
    is preserved on a real production op.
    """
    layout = _full_layout
    op = _find_op(layout, "layer2_mem_byte_flags")

    rewritten = derive_operation(
        op,
        dim_positions=layout.dim_positions,
        dim_sizes=getattr(layout, "dim_sizes", {}),
    )

    # Fields that should stay the same:
    assert rewritten.name == op.name
    assert rewritten.kind == op.kind
    assert rewritten.bake_fn is op.bake_fn
    assert rewritten.compiler_ir is op.compiler_ir
    assert rewritten.claims == op.claims
    assert rewritten.smoke_tests == op.smoke_tests
    assert rewritten.spec_section == op.spec_section

    # Reads/writes now strictly equal the derivation (no offset suffixes,
    # no leftover hand-declared dims).
    derived = derive_op_reads_writes_from_rules(
        op,
        dim_positions=layout.dim_positions,
        dim_sizes=getattr(layout, "dim_sizes", {}),
    )
    assert rewritten.reads == set(derived.reads)
    assert rewritten.writes == set(derived.writes)
