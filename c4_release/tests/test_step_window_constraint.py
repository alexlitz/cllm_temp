"""Tests for the step-window-constraint IR field + verifier.

Background: the autoregressive VM emits ``Token.STEP_TOKENS = 35``
tokens per VM step. Compute is supposed to fire at STEP_END within a
single step's window. Attention heads that mix tokens across windows
without an ALiBi recency slope or an explicit K-side step-boundary
suppressor are vulnerable to cross-step dilution -- the L8 head 4
OP_IMM relay bug at commit b23f818c is the canonical example.

This test module covers:

1. The :class:`StepWindowConstraint` enum + the ``step_window`` field
   on :class:`DeclarativeAttentionHeadSpec`.
2. The :func:`verify_step_window_constraint` /
   :func:`verify_step_window_constraints` verifier functions.
3. Three POC heads annotated in ops/l1_ops.py / ops/l7_ops.py /
   ops/l8_ops.py demonstrating each enum value.
4. The pre-b23f818c L8 head 4 regression scenario, showing the
   verifier catches a CURRENT_STEP_ONLY head with no slope and no
   K-side suppressor as a violation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.decl_verifier import (  # noqa: E402
    StepWindowConstraintIssue,
    StepWindowConstraintReport,
    verify_step_window_constraint,
    verify_step_window_constraints,
)
from neural_vm.unified_compiler.ir import (  # noqa: E402
    STEP_BOUNDARY_K_SUPPRESSOR_DIMS,
    STEP_WINDOW_MIN_ALIBI_SLOPE,
    StepWindowConstraint,
)
from neural_vm.unified_compiler.ops.l7_ops import (  # noqa: E402
    _layer7_memory_head_specs,
)
from neural_vm.unified_compiler.ops.l8_ops import (  # noqa: E402
    _L8_HEAD_LAYOUT_BY_NAME,
    _layer8_op_imm_relay_head_spec,
)
from neural_vm.unified_compiler.primitives import (  # noqa: E402
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
)
from neural_vm.vm_step import _SetDim  # noqa: E402


# ---------------------------------------------------------------------------
# Core enum / field surface
# ---------------------------------------------------------------------------


def test_step_window_enum_has_three_values():
    """The three documented values are present and stable."""
    values = {v.name for v in StepWindowConstraint}
    assert values == {"CURRENT_STEP_ONLY", "PREV_STEP_OK", "ANY_STEP"}


def test_declarative_attention_head_spec_default_step_window_is_current_step_only():
    """Backward-compat default: heads that omit ``step_window`` are
    CURRENT_STEP_ONLY (the safer choice for compute-intent heads)."""
    spec = DeclarativeAttentionHeadSpec(head_idx=0)
    assert spec.step_window is StepWindowConstraint.CURRENT_STEP_ONLY


def test_declarative_attention_head_spec_carries_explicit_step_window():
    """Specs can override the default per head."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=2,
        step_window=StepWindowConstraint.ANY_STEP,
    )
    assert spec.step_window is StepWindowConstraint.ANY_STEP


# ---------------------------------------------------------------------------
# Verifier: CURRENT_STEP_ONLY discipline
# ---------------------------------------------------------------------------


def test_current_step_only_without_slope_or_suppressor_is_violation():
    """The canonical pre-b23f818c L8 head 4 scenario: a relay head
    declared CURRENT_STEP_ONLY but with neither ALiBi decay nor a
    K-side step-boundary suppressor leaks attention to prior-step
    marker tokens. The verifier must flag this."""
    BD = _SetDim
    spec = DeclarativeAttentionHeadSpec(
        head_idx=4,
        # Q/K mimic the L8 op_imm_relay head 4 shape (AX-marker target).
        q=(AP(0, BD.IS_BYTE, 20.0), AP(0, BD.MARK_AX, 20.0)),
        k=(AP(0, BD.MARK_AX, 20.0),),
        v=(AP(0, BD.OP_IMM, 1.0),),
        o=(AO(BD.OP_IMM, 0, 1.0),),
        # alibi_slope omitted -- this is the pre-b23f818c regression.
        step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
    )
    issue = verify_step_window_constraint(spec)
    assert issue.kind == "violation"
    assert issue.head_idx == 4
    assert issue.declared == "CURRENT_STEP_ONLY"
    assert issue.alibi_slope is None
    assert "ALiBi slope" in issue.reason
    assert "step-boundary suppressor" in issue.reason


def test_current_step_only_with_alibi_slope_passes():
    """An ALiBi slope >= STEP_WINDOW_MIN_ALIBI_SLOPE confines softmax
    mass to the current 35-token window."""
    BD = _SetDim
    spec = DeclarativeAttentionHeadSpec(
        head_idx=4,
        q=(AP(0, BD.MARK_AX, 20.0),),
        k=(AP(0, BD.MARK_AX, 20.0),),
        v=(AP(0, BD.OP_IMM, 1.0),),
        o=(AO(BD.OP_IMM, 0, 1.0),),
        alibi_slope=STEP_WINDOW_MIN_ALIBI_SLOPE,
        step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
    )
    issue = verify_step_window_constraint(spec)
    assert issue.kind == "ok"
    assert issue.alibi_slope == STEP_WINDOW_MIN_ALIBI_SLOPE


def test_current_step_only_with_step_boundary_suppressor_passes():
    """A negative K read of MARK_SE_ONLY pushes prior-step tokens out
    of the softmax mass -- substitutes for the ALiBi slope."""
    BD = _SetDim
    dim_positions = {
        name: int(getattr(BD, name))
        for name in vars(BD)
        if name.isupper() and isinstance(getattr(BD, name), int)
    }
    # Use a suppressor dim that the verifier recognises by name.
    suppressor_dim_name = STEP_BOUNDARY_K_SUPPRESSOR_DIMS[0]
    assert suppressor_dim_name == "MARK_SE_ONLY"
    spec = DeclarativeAttentionHeadSpec(
        head_idx=4,
        q=(AP(0, BD.MARK_AX, 20.0),),
        k=(
            AP(0, BD.MARK_AX, 20.0),
            AP(0, dim_positions[suppressor_dim_name], -50.0),
        ),
        v=(AP(0, BD.OP_IMM, 1.0),),
        o=(AO(BD.OP_IMM, 0, 1.0),),
        step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
    )
    issue = verify_step_window_constraint(spec, dim_positions=dim_positions)
    assert issue.kind == "ok", issue.reason
    assert "step-boundary suppressor" in issue.reason


# ---------------------------------------------------------------------------
# Verifier: ANY_STEP discipline
# ---------------------------------------------------------------------------


def test_any_step_with_memory_marker_passes():
    """Memory-lookup heads read MARK_MEM (or similar) and carry
    unit-weight V/O -- they are correctly tagged ANY_STEP."""
    BD = _SetDim
    dim_positions = {
        name: int(getattr(BD, name))
        for name in vars(BD)
        if name.isupper() and isinstance(getattr(BD, name), int)
    }
    spec = DeclarativeAttentionHeadSpec(
        head_idx=7,
        q=(AP(0, BD.MARK_MEM, 15.0),),
        k=(AP(0, BD.MARK_MEM, 15.0),),
        v=(AP(1, BD.MEM_STORE, 1.0),),
        o=(AO(BD.MEM_STORE, 1, 1.0),),
        step_window=StepWindowConstraint.ANY_STEP,
    )
    issue = verify_step_window_constraint(spec, dim_positions=dim_positions)
    assert issue.kind == "ok"


def test_any_step_compute_intent_warns():
    """A compute-intent head (no memory-marker K, large relay
    weights) tagged ANY_STEP is suspicious -- compute heads almost
    always want CURRENT_STEP_ONLY."""
    BD = _SetDim
    dim_positions = {
        name: int(getattr(BD, name))
        for name in vars(BD)
        if name.isupper() and isinstance(getattr(BD, name), int)
    }
    spec = DeclarativeAttentionHeadSpec(
        head_idx=5,
        q=(AP(0, BD.MARK_AX, 20.0),),
        k=(AP(0, BD.MARK_AX, 20.0),),
        # Big relay weight; no MEM-marker K read -> looks compute-intent.
        v=(AP(0, BD.OP_IMM, 20.0),),
        o=(AO(BD.OP_IMM, 0, 20.0),),
        step_window=StepWindowConstraint.ANY_STEP,
    )
    issue = verify_step_window_constraint(spec, dim_positions=dim_positions)
    assert issue.kind == "any_step_compute_warning"
    assert "compute-intent" in issue.reason


# ---------------------------------------------------------------------------
# Verifier: PREV_STEP_OK is recorded as informational
# ---------------------------------------------------------------------------


def test_prev_step_ok_is_recorded_as_informational():
    """PREV_STEP_OK has no structural constraint; the verifier
    returns an informational issue for audit context."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=1,
        step_window=StepWindowConstraint.PREV_STEP_OK,
    )
    issue = verify_step_window_constraint(spec)
    assert issue.kind == "prev_step_ok_info"
    assert issue.declared == "PREV_STEP_OK"


# ---------------------------------------------------------------------------
# Batch verifier + report formatting
# ---------------------------------------------------------------------------


def test_verify_step_window_constraints_aggregates_ok_and_failures():
    """The batch wrapper retains only non-ok findings but counts all."""
    BD = _SetDim
    specs = [
        # OK: CURRENT_STEP_ONLY with slope.
        DeclarativeAttentionHeadSpec(
            head_idx=0,
            alibi_slope=STEP_WINDOW_MIN_ALIBI_SLOPE,
            q=(AP(0, BD.MARK_AX, 10.0),),
            k=(AP(0, BD.MARK_AX, 10.0),),
            v=(AP(0, BD.OP_IMM, 1.0),),
            o=(AO(BD.OP_IMM, 0, 1.0),),
            step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
        ),
        # Violation: CURRENT_STEP_ONLY without slope or suppressor.
        DeclarativeAttentionHeadSpec(
            head_idx=1,
            q=(AP(0, BD.MARK_AX, 10.0),),
            k=(AP(0, BD.MARK_AX, 10.0),),
            v=(AP(0, BD.OP_IMM, 1.0),),
            o=(AO(BD.OP_IMM, 0, 1.0),),
            step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
        ),
    ]
    report = verify_step_window_constraints(specs)
    assert isinstance(report, StepWindowConstraintReport)
    assert report.n_heads_checked == 2
    assert report.has_violations()
    assert len(report.violations()) == 1
    assert report.violations()[0].head_idx == 1
    # ``format()`` should mention the violation count and head index.
    text = report.format()
    assert "Violations:    1" in text
    assert "head=1" in text


# ---------------------------------------------------------------------------
# POC annotations on real ops
# ---------------------------------------------------------------------------


def test_l8_op_imm_relay_head_4_is_annotated_current_step_only():
    """The L8 OP_IMM relay head was bug-fixed in b23f818c by adding
    ``alibi_slope=0.5``. We now annotate the spec as
    CURRENT_STEP_ONLY so the verifier flags any future regression
    that removes the slope at decl-time."""
    BD = _SetDim
    spec = _layer8_op_imm_relay_head_spec(BD)
    assert spec.step_window is StepWindowConstraint.CURRENT_STEP_ONLY
    assert spec.alibi_slope == 0.5
    issue = verify_step_window_constraint(spec)
    assert issue.kind == "ok", issue.reason


def test_l7_memory_heads_head_7_is_annotated_any_step():
    """L7 head 7 broadcasts MEM-marker flags (MEM_STORE,
    MEM_ADDR_SRC, OP_JSR, OP_ENT) across the step. Memory
    persistence spans steps by design -- this is ANY_STEP."""
    BD = _SetDim
    specs = _layer7_memory_head_specs(BD)
    head_7_idx = next(
        i for i, s in enumerate(specs)
        if s.head_idx
        and any(getattr(BD, "MARK_MEM", None) == w.dim for w in s.k)
    )
    spec = specs[head_7_idx]
    assert spec.step_window is StepWindowConstraint.ANY_STEP


def test_pre_b23f818c_op_imm_relay_spec_would_be_flagged():
    """Synthesize the *pre-b23f818c* L8 head 4 spec (no
    ``alibi_slope``) and confirm the verifier flags it. This is the
    "violation enumerated" demonstration: had the framework existed
    before commit b23f818c, the verifier would have surfaced the
    IMM dilution bug at compile time.

    The original head's K side reads MARK_AX positively and IS_BYTE
    negatively. IS_BYTE is *not* a step-boundary marker (step bound
    markers are MARK_SE_ONLY / MARK_CS / MARK_SE), so resolving dim
    IDs via ``dim_positions`` correctly classifies the negative
    IS_BYTE read as **not** a suppressor -- and the verifier flags
    the absent slope as a violation. Without ``dim_positions`` the
    verifier falls back to "any negative K weight counts" and would
    pass the pre-fix spec; this test pins the name-aware path.
    """
    BD = _SetDim
    dim_positions = {
        name: int(getattr(BD, name))
        for name in vars(BD)
        if name.isupper() and isinstance(getattr(BD, name), int)
    }
    AX_I = 1
    L8_relay = 20.0
    pre_fix_spec = DeclarativeAttentionHeadSpec(
        head_idx=_L8_HEAD_LAYOUT_BY_NAME["layer8_op_imm_relay.head_4"],
        q=(
            AP(0, BD.IS_BYTE, L8_relay),
            AP(0, BD.H1 + AX_I, L8_relay),
            AP(0, BD.CONST, -L8_relay * 1.5),
        ),
        k=(
            AP(0, BD.MARK_AX, L8_relay),
            AP(0, BD.IS_BYTE, -L8_relay * 10),
        ),
        v=(AP(0, BD.OP_IMM, 1.0),),
        o=(AO(BD.OP_IMM, 0, 1.0),),
        # alibi_slope NOT set -- this is what the pre-b23f818c bake
        # had. step_window remains the default CURRENT_STEP_ONLY.
    )
    issue = verify_step_window_constraint(
        pre_fix_spec, dim_positions=dim_positions
    )
    assert issue.kind == "violation", issue.reason
    assert issue.alibi_slope is None
    # The post-fix spec passes (carries alibi_slope=0.5).
    post_fix_spec = _layer8_op_imm_relay_head_spec(BD)
    assert (
        verify_step_window_constraint(
            post_fix_spec, dim_positions=dim_positions
        ).kind
        == "ok"
    )
