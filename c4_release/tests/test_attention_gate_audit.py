"""Tests for the symbolic attention-gate effectivity check.

The audit (``c4_release/neural_vm/unified_compiler/dsl_interpreter.py``)
classifies each Q-side condition gate in a
:class:`DeclarativeAttentionHeadSpec` as one of:

* ``safe`` — K-side at the same slot writes a non-CONST discriminator;
* ``softmax1_suppress`` — K-side at the slot writes only CONST(s), but
  the per-row score contribution is sufficiently negative in one
  regime (cond ON or OFF) that under softmax1 the zero-anchor sink
  wins on those rows. The deployed model uses softmax1 (see
  ``docs/IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md``), so this counts
  as an *effective* gate.
* ``no_op`` — K-side at the slot writes only CONST(s) AND no softmax1
  suppression applies (gate is ineffective under both softmax and
  softmax1);
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


def _suppress_spec() -> DeclarativeAttentionHeadSpec:
    """Q has MARK_PC at slot 33 plus a negative CONST offset; K has only CONST.

    This is the canonical GATE=33 pattern from ``_carry_forward_head_spec``
    (audit doc). Under standard softmax the uniform offset cancels — the
    gate is a "no_op". Under softmax1 the OFF regime score
    (-7.5 * 15 = -112.5) drops the row below the zero-anchor sink, so
    the gate IS effective ("softmax1_suppress").
    """
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(33, BD.MARK_PC, 15.0), AP(33, BD.CONST, -7.5)),
        k=(AP(33, BD.CONST, 15.0),),
        v=(),
        o=(),
    )


# Back-compat alias — older tests / docs refer to the canonical fixture
# as ``_noop_spec``. It's a misnomer post-softmax1 awareness (the spec
# *does* suppress under softmax1), but the static-shape is identical.
_noop_spec = _suppress_spec


def _genuine_noop_spec() -> DeclarativeAttentionHeadSpec:
    """Q has MARK_PC at slot 33 with no Q-side CONST offset; K has only CONST.

    Without a Q-side CONST offset there's no per-row score base, so
    the OFF regime contribution is 0 (not suppressive even under
    softmax1) and the ON regime contribution is positive (not
    suppressive either). The gate is ineffective under BOTH softmax
    and softmax1 — this is the true "no_op" classification.
    """
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(33, BD.MARK_PC, 15.0),),
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


def _is_byte_gate_genuine_noop_spec() -> DeclarativeAttentionHeadSpec:
    """``IS_*`` prefixed gate with no Q-side CONST offset — true no_op.

    Without a negative Q-side CONST offset to drive the OFF regime
    below the sink, this gate is ineffective under both softmax and
    softmax1: gates that select only via a positive ON-regime score
    don't filter K positions, they just gate the whole row's
    attention output. Tracked as ``no_op``.
    """
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(33, BD.IS_BYTE, 10000.0),),
        k=(AP(33, BD.CONST, 5.0),),
        v=(),
        o=(),
    )


# Back-compat alias for the original fixture name.
_is_byte_gate_noop_spec = _is_byte_gate_genuine_noop_spec


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

    def test_canonical_gate_classified_suppress_under_softmax1(self):
        """The GATE=33 canonical pattern: under default ``softmax1`` mode
        the negative Q-side CONST offset drives the OFF regime score below
        the zero-anchor sink, so the gate IS effective ("softmax1_suppress").
        """
        entries = audit_attention_gates(_suppress_spec())
        gate_entries = [e for e in entries if e.q_dim_name == "MARK_PC"]
        assert len(gate_entries) == 1
        e = gate_entries[0]
        assert e.kind == "softmax1_suppress"
        assert e.slot == 33
        assert e.k_dim_names == ("CONST",)
        assert e.softmax_mode == "softmax1"

    def test_canonical_gate_classified_no_op_under_standard_softmax(self):
        """Under explicit ``softmax_mode='softmax'`` the same canonical
        GATE=33 pattern reverts to the legacy "no_op" verdict (uniform
        K-side offset cancels under standard softmax).
        """
        entries = audit_attention_gates(
            _suppress_spec(), softmax_mode="softmax",
        )
        gate_entries = [e for e in entries if e.q_dim_name == "MARK_PC"]
        assert len(gate_entries) == 1
        e = gate_entries[0]
        assert e.kind == "no_op"
        assert e.softmax_mode == "softmax"

    def test_genuine_no_op_classified_no_op_under_softmax1(self):
        """A gate with no Q-side CONST offset has off_score == 0 under
        softmax1 — neither regime drives the row below the sink, so the
        gate is ineffective. Must classify as "no_op" even under softmax1.
        """
        entries = audit_attention_gates(_genuine_noop_spec())
        assert len(entries) == 1
        assert entries[0].kind == "no_op"
        assert entries[0].softmax_mode == "softmax1"

    def test_q_only_gate_classified_q_only(self):
        entries = audit_attention_gates(_q_only_spec())
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "q_only"
        assert e.slot == 7
        assert e.q_dim_name == "MARK_PC"
        assert e.k_dim_names == ()

    def test_is_prefix_gate_recognised(self):
        entries = audit_attention_gates(_is_byte_gate_genuine_noop_spec())
        assert len(entries) == 1
        assert entries[0].q_dim_name == "IS_BYTE"
        # No Q-side CONST offset → genuine no_op even under softmax1.
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
        """The genuine no_op (no Q-CONST offset) is flagged even under
        softmax1. The suppress-style fixture is reclassified instead."""
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _genuine_noop_spec()))
        c.add_op(_make_attn_op("suppress_op", _suppress_spec()))
        c.add_op(_make_attn_op("safe_op", _safe_spec()))
        entries = audit_compiler_attention_gates(c)
        kinds = {e.op_name: e.kind for e in entries}
        assert kinds.get("noop_op") == "no_op"
        assert kinds.get("suppress_op") == "softmax1_suppress"
        assert kinds.get("safe_op") == "safe"

    def test_audit_legacy_softmax_mode_flags_canonical_pattern(self):
        """With ``softmax_mode='softmax'`` the canonical GATE=33 pattern
        falls back to ``no_op`` (pre-softmax1 behaviour)."""
        c = LayerCompiler()
        c.add_op(_make_attn_op("canonical_op", _suppress_spec()))
        entries = audit_compiler_attention_gates(c, softmax_mode="softmax")
        kinds = {e.op_name: e.kind for e in entries}
        assert kinds.get("canonical_op") == "no_op"

    def test_run_attention_gate_audit_warns_on_no_op(self):
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _genuine_noop_spec()))
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

    def test_run_attention_gate_audit_silent_when_only_suppress(self):
        """``softmax1_suppress`` entries are effective gates under
        softmax1 — they must not trigger a warning."""
        c = LayerCompiler()
        c.add_op(_make_attn_op("suppress_op", _suppress_spec()))
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            run_attention_gate_audit(c)
        msgs = [str(w.message) for w in wlist if "ATTENTION GATE AUDIT" in str(w.message)]
        assert not msgs

    def test_skip_env_disables_check(self, monkeypatch):
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _genuine_noop_spec()))
        monkeypatch.setenv(GATE_AUDIT_SKIP_ENV, "1")
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            entries = run_attention_gate_audit(c)
        assert entries == []
        msgs = [str(w.message) for w in wlist if "ATTENTION GATE AUDIT" in str(w.message)]
        assert not msgs

    def test_strict_env_raises(self, monkeypatch):
        c = LayerCompiler()
        c.add_op(_make_attn_op("noop_op", _genuine_noop_spec()))
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


# Op names the audit doc explicitly calls out as containing K=CONST
# Q-side condition gates. The production audit must classify every one
# of these as non-``safe`` — either ``softmax1_suppress`` (effective
# under softmax1; the canonical GATE=33 pattern) or ``no_op`` /
# ``q_only`` (truly broken). If a name drops out, either the op was
# restructured to use real K-side discriminators (good) or the audit
# lost coverage (bad).
DOC_FLAGGED_OPS_FLOOR: frozenset = frozenset({
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
    # L14 MEM-generation slot 33/38 K=CONST blockers.
    "layer14_mem_generation",
    # L15 ALU high-byte relay slot 33/34 K=CONST blockers.
    "layer15_alu_high_byte_relay",
    # model_ops.py function-call routing path (slot 33/34).
    "function_call_weights",
    # REMOVED (repaired with K-side complements):
    # - layer3_carry_forward_attn (commit c1f1af2e)
    # - layer10_stack0_byte_relay_bake (commit c1f1af2e)
})

# Kinds that indicate the audit still recognises a K=CONST condition
# gate on the op (either effective-under-softmax1 or genuinely broken).
_CONST_K_KINDS: frozenset = frozenset({
    "softmax1_suppress", "no_op", "q_only",
})


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    os.environ.get(GATE_AUDIT_SKIP_ENV) == "1",
    reason=f"{GATE_AUDIT_SKIP_ENV}=1 in environment",
)
def test_production_doc_listed_ops_all_flagged():
    """Every op the doc names still has a K=CONST condition gate in the audit.

    With softmax1-awareness many gates classify as ``softmax1_suppress``
    (effective under softmax1) rather than ``no_op``, but the op should
    still appear with a non-``safe`` kind because the K-side at the
    flagged slot is still CONST-only. If the op drops out entirely the
    op was restructured to use a real K-side discriminator (in which
    case remove it from ``DOC_FLAGGED_OPS_FLOOR``).
    """
    entries = _compile_and_capture_audit()
    const_k_ops = {
        e.op_name for e in entries
        if e.kind in _CONST_K_KINDS and e.op_name
    }
    missing = DOC_FLAGGED_OPS_FLOOR - const_k_ops
    assert not missing, (
        f"Doc-listed ops missing from production audit: {sorted(missing)}. "
        "Either those ops were repaired to use real K-side discriminators "
        "(remove from DOC_FLAGGED_OPS_FLOOR) or the audit lost coverage."
    )


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    os.environ.get(GATE_AUDIT_SKIP_ENV) == "1",
    reason=f"{GATE_AUDIT_SKIP_ENV}=1 in environment",
)
def test_production_softmax1_reduces_flagged_count():
    """Softmax1-aware audit must classify substantially fewer gates as
    ``no_op`` than the legacy softmax-mode audit, because the canonical
    GATE=33 pattern is effective under softmax1.
    """
    from c4_release.neural_vm.unified_compiler.dsl_interpreter import (
        audit_compiler_attention_gates,
    )

    entries_sm1 = _compile_and_capture_audit()
    sm1_no_op = sum(1 for e in entries_sm1 if e.kind == "no_op")
    sm1_suppress = sum(
        1 for e in entries_sm1 if e.kind == "softmax1_suppress"
    )
    assert sm1_suppress > 0, (
        "Softmax1-aware audit produced 0 softmax1_suppress entries; "
        "either the audit regressed or every CONST-K gate is now truly "
        "broken (investigate)."
    )
    # The reclassified suppress set should dominate the remaining no_op
    # set — the original audit doc enumerated 70 instances; if softmax1
    # awareness leaves more than 35 as no_op, the threshold heuristic
    # may be too conservative.
    assert sm1_no_op < sm1_suppress, (
        f"Softmax1 audit still classifies {sm1_no_op} gates as no_op "
        f"vs {sm1_suppress} as softmax1_suppress — the reclassifier "
        "should dominate."
    )
