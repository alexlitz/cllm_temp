"""Byte-identity parity for L10 null-terminator detection migration.

Phase 11.A migrated ``_set_null_terminator_detection`` (one unit at
L10 FFN index 1864) from inline ``vm_step`` helper into a declarative
``FFNRule.gated_write`` exposed via ``compiler_ir`` on
``make_null_terminator_detection_op``. This test pins the byte-identity
contract: the per-cell W_up / b_up / W_gate / b_gate / W_down weights
produced by the new declarative lowering must be tensor-equal to the
ones the legacy imperative helper writes.

When ``enable_conversational_io=False`` (or ``alu_mode != 'lookup'``)
both code paths are no-ops and the op exposes an empty ``CompilerIR``.
"""

import torch

from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.ops.flag_gated_ops import (
    _NULL_TERMINATOR_DETECTION_START_UNIT,
    _lower_null_terminator_detection_ir,
    _null_terminator_detection_ir,
    _null_terminator_detection_rules,
    make_null_terminator_detection_op,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives
from c4_release.neural_vm.setup_helpers_l10 import (
    _set_null_terminator_detection,
)
from c4_release.neural_vm.vm_step import _SetDim


_L10_HIDDEN_DIM = 2048  # margin above the 1864-unit footprint


class _StubFFN:
    def __init__(self, *, d_model: int = 736, hidden_dim: int = _L10_HIDDEN_DIM):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn_units(
    actual: _StubFFN,
    expected: _StubFFN,
    start: int,
    end: int,
):
    assert torch.equal(actual.W_up[start:end], expected.W_up[start:end]), (
        f"W_up mismatch in units {start}..{end}"
    )
    assert torch.equal(actual.b_up[start:end], expected.b_up[start:end]), (
        f"b_up mismatch in units {start}..{end}"
    )
    assert torch.equal(actual.W_gate[start:end], expected.W_gate[start:end]), (
        f"W_gate mismatch in units {start}..{end}"
    )
    assert torch.equal(actual.b_gate[start:end], expected.b_gate[start:end]), (
        f"b_gate mismatch in units {start}..{end}"
    )
    assert torch.equal(
        actual.W_down[:, start:end], expected.W_down[:, start:end]
    ), f"W_down mismatch in units {start}..{end}"


def test_null_terminator_detection_rules_match_legacy_unit():
    """Single unit at 1864: byte-identity vs ``_set_null_terminator_detection``.

    The new declarative ``FFNRule.gated_write`` lowered via
    :func:`_lower_null_terminator_detection_ir` must produce identical
    W_up / b_up / W_gate / b_gate / W_down rows at unit 1864 as the
    legacy imperative helper.
    """

    actual = _StubFFN()
    expected = _StubFFN()

    _lower_null_terminator_detection_ir(actual, 100.0, _SetDim)
    _set_null_terminator_detection(expected, 100.0, _SetDim)

    start = _NULL_TERMINATOR_DETECTION_START_UNIT
    end = start + 1
    _assert_same_ffn_units(actual, expected, start, end)


def test_null_terminator_detection_compare_symbolic_to_lowered_ffn():
    """``compare_symbolic_to_lowered_ffn`` reports no structural drift."""

    rules = _null_terminator_detection_rules(100.0)
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    names = Primitives.ffn_rule_dim_names(ir.layer(0).ffn.rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    report = compare_symbolic_to_lowered_ffn(
        ir, dim_positions, S=100.0, atol=1e-4, rtol=1e-4,
    )
    structural = [
        issue for issue in report.issues
        if issue.kind in ("declaration_semantics", "lowering")
    ]
    assert not structural, (
        "compare_symbolic_to_lowered_ffn structural drift:\n"
        + "\n".join(f"  [{i.kind}] {i.message}" for i in structural)
    )


def test_null_terminator_detection_compiler_ir_exposed_when_enabled():
    """When convo-IO + lookup mode are both active, ``compiler_ir`` carries
    the matching ``FFNRule`` so the declarative verifier sees what the
    bake lowers. Otherwise it is empty (matches the no-op bake)."""

    enabled_op = make_null_terminator_detection_op(
        enable_conversational_io=True, alu_mode="lookup",
    )
    assert len(enabled_op.compiler_ir.layer(0).ffn.rules) == 1
    rule = enabled_op.compiler_ir.layer(0).ffn.rules[0]
    assert rule.name == "null_terminator_detection"

    # No-op states: compiler_ir is empty.
    for kwargs in (
        dict(enable_conversational_io=False, alu_mode="lookup"),
        dict(enable_conversational_io=True, alu_mode="efficient"),
        dict(enable_conversational_io=False, alu_mode="efficient"),
    ):
        op = make_null_terminator_detection_op(**kwargs)
        assert len(op.compiler_ir.layer(0).ffn.rules) == 0, kwargs


def test_null_terminator_detection_no_op_when_disabled():
    """When the bake is a no-op (flag off or efficient mode) calling it
    on a clean FFN must leave every weight at zero."""

    op = make_null_terminator_detection_op(
        enable_conversational_io=False, alu_mode="lookup",
    )

    class _Block:
        def __init__(self, ffn):
            self.ffn = ffn

    ffn = _StubFFN()
    block = _Block(ffn)
    dim_positions = Primitives.dim_positions_from_bd(
        _SetDim,
        ("OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI", "IO_IN_OUTPUT_MODE",
         "IO_OUTPUT_COMPLETE", "NEXT_THINKING_START"),
    )
    op.bake_fn(block, dim_positions, 100.0)

    assert torch.equal(ffn.W_up, torch.zeros_like(ffn.W_up))
    assert torch.equal(ffn.b_up, torch.zeros_like(ffn.b_up))
    assert torch.equal(ffn.W_gate, torch.zeros_like(ffn.W_gate))
    assert torch.equal(ffn.b_gate, torch.zeros_like(ffn.b_gate))
    assert torch.equal(ffn.W_down, torch.zeros_like(ffn.W_down))


def test_null_terminator_detection_ir_helper_returns_compiler_ir():
    """``_null_terminator_detection_ir`` returns a CompilerIR carrying the
    rule on layer 0 (independent of production start_unit=1864)."""

    ir = _null_terminator_detection_ir()
    assert isinstance(ir, CompilerIR)
    assert len(ir.layer(0).ffn.rules) == 1
