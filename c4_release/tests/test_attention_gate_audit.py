"""Tests for the symbolic attention-gate effectivity check.

The audit (``c4_release/neural_vm/unified_compiler/dsl_interpreter.py``)
classifies each Q-side condition gate in a
:class:`DeclarativeAttentionHeadSpec` as one of:

* ``safe`` — K-side at the same slot writes a non-CONST discriminator;
* ``no_op`` — K-side at the slot writes only CONST(s) (the canonical
  "silent leak" case from
  ``c4_release/docs/Q_SIDE_GATE_AUDIT_2026_06_07.md``);
* ``q_only`` — K-side has no write at the slot at all (literal dead
  weight).

The synthetic-spec tests pin down each classification with a minimal
fixture. The production test exercises the compile-time hook end-to-end
and cross-checks the runtime no-op count against the 70 instances
listed in the audit doc.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import List

import pytest

from c4_release.neural_vm.unified_compiler.dsl_interpreter import (
    GATE_AUDIT_SKIP_ENV,
    GATE_AUDIT_STRICT_ENV,
    GateAuditEntry,
    GateAuditError,
    audit_attention_gates,
    audit_compiler_attention_gates,
    build_dim_name_map,
    format_gate_audit_report,
    run_attention_gate_audit,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
)
from c4_release.neural_vm.unified_compiler.ir import CompilerIR
from c4_release.neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
)
from c4_release.neural_vm.vm_step import _SetDim as BD


# ---------------------------------------------------------------------------
# Synthetic-spec classification fixtures
# ---------------------------------------------------------------------------


def _safe_spec() -> DeclarativeAttentionHeadSpec:
    """Q has MARK_PC at slot 0; K has L1H1 (real discriminator) at slot 0."""
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, BD.MARK_PC, 15.0),),
        k=(AP(0, BD.L1H1, 15.0),),
        v=(),
        o=(),
    )


def _noop_spec() -> DeclarativeAttentionHeadSpec:
    """Q has MARK_PC at slot 33; K has only CONST at slot 33.

    This is the canonical anti-pattern from the audit doc — the GATE=33
    pattern in ``_carry_forward_head_spec``.
    """
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(33, BD.MARK_PC, 15.0), AP(33, BD.CONST, -7.5)),
        k=(AP(33, BD.CONST, 15.0),),
        v=(),
        o=(),
    )


def _q_only_spec() -> DeclarativeAttentionHeadSpec:
    """Q has MARK_PC at slot 7; K has no write at slot 7 at all."""
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(7, BD.MARK_PC, 15.0),),
        k=(),
        v=(),
        o=(),
    )


def _is_byte_gate_noop_spec() -> DeclarativeAttentionHeadSpec:
    """Verify ``IS_*`` prefixed dims are also recognised as gate conditions.

    Mirrors the L10 PSH STACK0 passthrough no-op (slot 33, Q=IS_BYTE,
    K=CONST only) from the audit doc.
    """
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(33, BD.IS_BYTE, 10000.0),),
        k=(AP(33, BD.CONST, 5.0),),
        v=(),
        o=(),
    )


def _op_gate_safe_spec() -> DeclarativeAttentionHeadSpec:
    """Q has OP_LEV at slot 33; K has a real BYTE_INDEX_0 discriminator."""
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(33, BD.OP_LEV, -10000.0),),
        k=(AP(33, BD.BYTE_INDEX_0, 1.0),),
        v=(),
        o=(),
    )


class TestSyntheticClassification:
    def test_known_safe_gate_classified_safe(self):
        entries = audit_attention_gates(_safe_spec())
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "safe"
        assert e.slot == 0
        assert e.q_dim_name == "MARK_PC"
        assert "L1H1" in e.k_dim_names
        assert e.head_idx == 0

    def test_known_noop_gate_classified_no_op(self):
        entries = audit_attention_gates(_noop_spec())
        # Q has two writes at slot 33; only MARK_PC is a condition dim.
        # The CONST write at the same slot is not a condition gate.
        gate_entries = [e for e in entries if e.q_dim_name == "MARK_PC"]
        assert len(gate_entries) == 1
        e = gate_entries[0]
        assert e.kind == "no_op"
        assert e.slot == 33
        assert e.k_dim_names == ("CONST",)

    def test_q_only_gate_classified_q_only(self):
        entries = audit_attention_gates(_q_only_spec())
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "q_only"
        assert e.slot == 7
        assert e.q_dim_name == "MARK_PC"
        assert e.k_dim_names == ()

    def test_is_prefix_gate_recognised(self):
        entries = audit_attention_gates(_is_byte_gate_noop_spec())
        assert len(entries) == 1
        assert entries[0].q_dim_name == "IS_BYTE"
        assert entries[0].kind == "no_op"

    def test_op_prefix_gate_with_real_k_discriminator_is_safe(self):
        entries = audit_attention_gates(_op_gate_safe_spec())
        assert len(entries) == 1
        assert entries[0].q_dim_name == "OP_LEV"
        assert entries[0].kind == "safe"

    def test_non_condition_q_dim_skipped(self):
        """A Q-side write to a non-condition dim (e.g. CONST itself) is
        not an attention gate and must not appear in the audit."""
        spec = DeclarativeAttentionHeadSpec(
            head_idx=0,
            q=(AP(0, BD.CONST, -10.0),),
            k=(AP(0, BD.CONST, 10.0),),
        )
        assert audit_attention_gates(spec) == []

    def test_passes_op_name_through(self):
        entries = audit_attention_gates(
            _noop_spec(), op_name="layer3_carry_forward_attn",
        )
        assert all(e.op_name == "layer3_carry_forward_attn" for e in entries)


# ---------------------------------------------------------------------------
# Dim-name map helpers
# ---------------------------------------------------------------------------


class TestDimNameMap:
    def test_falls_back_to_setdim_when_none(self):
        m = build_dim_name_map(None)
        # CONST is at position 8 in _SetDim.
        assert "CONST" in m[BD.CONST]
        # MARK_PC is at position 0.
        assert "MARK_PC" in m[BD.MARK_PC]

    def test_custom_positions_used(self):
        m = build_dim_name_map({"MARK_PC": 100, "CONST": 101})
        assert m.get(100) == ("MARK_PC",)
        assert m.get(101) == ("CONST",)

    def test_aliased_dims_collected(self):
        m = build_dim_name_map(None)
        # H5+0 and SP_BYTE0_IS_F8 alias to position 95.
        names_at_95 = set(m.get(95, ()))
        assert "SP_BYTE0_IS_F8" in names_at_95


# ---------------------------------------------------------------------------
# Compile-time hook
# ---------------------------------------------------------------------------


def _noop_bake(module, dims, S):
    return None


def _make_attn_op(name: str, spec: DeclarativeAttentionHeadSpec) -> Operation:
    """Wrap a single spec in a minimal Operation with a precomputed IR."""

    ir = CompilerIR()
    ir.layer(0).attention.add_head(spec)
    return Operation(
        name=name,
        reads=set(),
        writes=set(),
        kind="attn",
        bake_fn=_noop_bake,
        compiler_ir=ir,
    )


class TestCompilerIntegration:
    def test_audit_finds_synthetic_no_op_in_compiler(self):
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _noop_spec()))
        c.add_op(_make_attn_op("safe_op", _safe_spec()))
        entries = audit_compiler_attention_gates(c)
        kinds = {e.op_name: e.kind for e in entries}
        assert kinds.get("noop_op") == "no_op"
        assert kinds.get("safe_op") == "safe"

    def test_run_attention_gate_audit_warns_on_no_op(self):
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _noop_spec()))
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            run_attention_gate_audit(c)
        msgs = [str(w.message) for w in wlist if "ATTENTION GATE AUDIT" in str(w.message)]
        assert msgs, "audit should warn when no-op gates are present"
        assert "noop_op" in msgs[0]

    def test_run_attention_gate_audit_silent_when_only_safe(self):
        c = LayerCompiler()
        c.add_op(_make_attn_op("safe_op", _safe_spec()))
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            run_attention_gate_audit(c)
        msgs = [str(w.message) for w in wlist if "ATTENTION GATE AUDIT" in str(w.message)]
        assert not msgs

    def test_skip_env_disables_check(self, monkeypatch):
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _noop_spec()))
        monkeypatch.setenv(GATE_AUDIT_SKIP_ENV, "1")
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            entries = run_attention_gate_audit(c)
        assert entries == []
        msgs = [str(w.message) for w in wlist if "ATTENTION GATE AUDIT" in str(w.message)]
        assert not msgs

    def test_strict_env_raises(self, monkeypatch):
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _noop_spec()))
        monkeypatch.setenv(GATE_AUDIT_STRICT_ENV, "1")
        with pytest.raises(GateAuditError) as exc_info:
            run_attention_gate_audit(c)
        # Flagged entries are attached for programmatic inspection.
        assert exc_info.value.flagged
        assert exc_info.value.flagged[0].kind == "no_op"


class TestReporting:
    def test_format_groups_by_op(self):
        entries = [
            GateAuditEntry(
                slot=33, q_dim_name="MARK_PC", kind="no_op",
                k_dim_names=("CONST",), head_idx=0, op_name="op_a",
            ),
            GateAuditEntry(
                slot=33, q_dim_name="IS_BYTE", kind="no_op",
                k_dim_names=("CONST",), head_idx=1, op_name="op_b",
            ),
            GateAuditEntry(
                slot=0, q_dim_name="MARK_PC", kind="safe",
                k_dim_names=("L1H1",), head_idx=0, op_name="op_a",
            ),
        ]
        report = format_gate_audit_report(entries)
        assert "3 Q-side condition gate(s)" in report
        assert "2 no-op" in report
        assert "1 safe" in report
        assert "op_a" in report
        assert "op_b" in report


# ---------------------------------------------------------------------------
# Production cross-check
# ---------------------------------------------------------------------------

# The static AST scan in ``tools/q_side_gate_audit.py`` flags 70 no-op
# instances across ``unified_compiler/ops/*_ops.py`` (see
# ``docs/Q_SIDE_GATE_AUDIT_2026_06_07.md``). The dynamic compile-time
# audit walks every ``DeclarativeAttentionHeadSpec`` at runtime, which
# may exceed the static count because a single helper-defined spec
# (e.g. ``_carry_forward_head_spec``) can be instantiated multiple
# times — each instance becomes its own runtime gate.
PRODUCTION_NO_OP_COUNT_FLOOR = 70


def _compile_and_capture_audit() -> List[GateAuditEntry]:
    """Run a full ``compile_full_vm_dynamic`` and return the gate audit."""

    from c4_release.neural_vm.unified_compiler import layer_compiler as lc_module
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    captured: List[List[GateAuditEntry]] = []
    original = lc_module.LayerCompiler.compile

    def _wrapped(self):
        result = original(self)
        captured.append(
            list(getattr(self, "_last_attention_gate_audit", []) or [])
        )
        return result

    lc_module.LayerCompiler.compile = _wrapped
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile_full_vm_dynamic(disk_cache=False)
    finally:
        lc_module.LayerCompiler.compile = original

    assert captured, "compile_full_vm_dynamic did not call LayerCompiler.compile"
    return captured[-1]


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    os.environ.get(GATE_AUDIT_SKIP_ENV) == "1",
    reason=f"{GATE_AUDIT_SKIP_ENV}=1 in environment",
)
def test_production_no_op_gate_count_meets_doc_floor():
    """Production compile flags at least the 70 no-op gates from the audit doc.

    ``Q_SIDE_GATE_AUDIT_2026_06_07.md`` enumerates 70 instances via
    static AST scan. The dynamic audit is a strict superset: every
    static-scan hit produces at least one runtime instance, plus any
    helper spec instantiated more than once contributes additional
    instances. The floor is a regression guard: if the count drops below
    70 a real fix landed (good — bump the floor) OR the audit lost
    coverage (bad — investigate).
    """
    entries = _compile_and_capture_audit()
    flagged = [e for e in entries if e.kind != "safe"]
    assert len(flagged) >= PRODUCTION_NO_OP_COUNT_FLOOR, (
        f"Flagged gate count {len(flagged)} fell below the doc floor "
        f"{PRODUCTION_NO_OP_COUNT_FLOOR}. Either a fix shrunk the set "
        "(update the floor) or the audit regressed (investigate "
        "tools/q_side_gate_audit.py vs dsl_interpreter audit)."
    )


# Op names the audit doc explicitly calls out as containing no-op /
# q-only gates. The production audit must flag every one of these — if
# a name drops out, either the op was repaired (good) or the audit
# lost coverage (bad).
DOC_FLAGGED_OPS_FLOOR: frozenset = frozenset({
    # GATE=33 anti-pattern in L3 carry-forward + relay heads.
    "layer3_carry_forward_attn",
    # L4 PC relay (declarative GATE=33 anti-pattern).
    "layer4_pc_relay",
    # L5 fetch heads — slot 32 dead weight + slot 33 K=CONST blockers.
    "layer5_fetch",
    # L7 operand gather — OP_LEA / OP_ADJ / OP_ENT blockers with K=CONST.
    "layer7_operand_gather",
    # L8 sp_gather — slot 33 MARK_STACK0 / MARK_SP with K=CONST.
    "layer8_sp_gather_bake",
    # L9 LEV addr / BP->PC relay (GATE=33 anti-pattern).
    "layer9_lev_addr_relay",
    "layer9_lev_bp_to_pc_relay",
    # L10 stack0 persistence + byte passthrough chain heads.
    "layer10_stack0_byte_relay_bake",
    # L14 MEM-generation slot 33/38 K=CONST blockers.
    "layer14_mem_generation",
    # L15 ALU high-byte relay slot 33/34 K=CONST blockers.
    "layer15_alu_high_byte_relay",
    # model_ops.py function-call routing path (slot 33/34).
    "function_call_weights",
})


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    os.environ.get(GATE_AUDIT_SKIP_ENV) == "1",
    reason=f"{GATE_AUDIT_SKIP_ENV}=1 in environment",
)
def test_production_doc_listed_ops_all_flagged():
    """Every op the doc names is flagged in the production audit."""
    entries = _compile_and_capture_audit()
    flagged_ops = {
        e.op_name for e in entries if e.kind != "safe" and e.op_name
    }
    missing = DOC_FLAGGED_OPS_FLOOR - flagged_ops
    assert not missing, (
        f"Doc-listed ops missing from production audit: {sorted(missing)}. "
        "Either those ops were repaired (remove from "
        "DOC_FLAGGED_OPS_FLOOR) or the audit lost coverage."
    )
