"""Parity tests for the L13 FFN bake migrated in Phase 6 Wave 4B.

Covers:
- ``layer13_shifts`` -- migrated from ``setup_helpers._set_layer13_shifts``
  (a single 4096-unit lookup table covering OP_SHL and OP_SHR for shift
  amounts 0..7) to a declarative
  :class:`c4_release.neural_vm.unified_compiler.ir.FFNRule` list lowered
  via :func:`Primitives.lower_ffn_rules`.

Sub-stage parity:
- SHL rules (units 0..2047) match the SHL portion of the legacy helper.
- SHR rules (units 2048..4095) match the SHR portion of the legacy helper.
- Combined IR lowers byte-identically across the whole 4096-unit table.
- ``compare_symbolic_to_lowered_ffn`` confirms declaration + lowering
  contract are consistent for the full IR.
"""

import torch

from c4_release.neural_vm.setup_helpers import _set_layer13_shifts
from c4_release.neural_vm.vm_step import _SetDim
from c4_release.neural_vm.unified_compiler.ir import (
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.ops.l13_ops import (
    _bake_layer13_shifts,
    _L13_SHIFTS_UNIT_LAYOUT,
    _layer13_shifts_ir,
    _layer13_shifts_rules,
    _layer13_shl_rules,
    _layer13_shr_rules,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 4096):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        a = getattr(actual, name)
        e = getattr(expected, name)
        diff = (a - e).abs()
        max_diff = float(diff.max().item()) if diff.numel() else 0.0
        assert torch.equal(a, e), (
            f"{name} differs (max_diff={max_diff:.3e}); "
            f"declarative bake is not byte-identical to legacy bake"
        )


def _assert_same_ffn_units(
    actual: _StubFFN, expected: _StubFFN, start: int, end: int
):
    assert torch.equal(actual.W_up[start:end], expected.W_up[start:end])
    assert torch.equal(actual.b_up[start:end], expected.b_up[start:end])
    assert torch.equal(actual.W_gate[start:end], expected.W_gate[start:end])
    assert torch.equal(actual.b_gate[start:end], expected.b_gate[start:end])
    assert torch.equal(actual.W_down[:, start:end], expected.W_down[:, start:end])


def test_layer13_shifts_rule_count_is_4096():
    """SHL + SHR = 4096 rules; sub-stage offsets match the layout table."""

    shl = _layer13_shl_rules(100.0)
    shr = _layer13_shr_rules(100.0)
    assert len(shl) == 2048
    assert len(shr) == 2048
    combined = _layer13_shifts_rules(100.0)
    assert len(combined) == 4096
    # First half is SHL, second half is SHR (matches pinned offsets).
    assert combined[:2048] == shl
    assert combined[2048:] == shr

    # Layout table totals must equal the rule count (byte-identity guard).
    layout_total = sum(n for _, _, n in _L13_SHIFTS_UNIT_LAYOUT)
    assert layout_total == 4096


def test_layer13_shl_substage_matches_legacy_units():
    """SHL rules (units 0..2047) byte-identical to the SHL portion of
    ``_set_layer13_shifts``.
    """
    actual = _StubFFN(hidden_dim=4096)
    expected = _StubFFN(hidden_dim=4096)

    rules = _layer13_shl_rules(100.0)
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(
        actual, rules, dim_positions, start_unit=0, S=100.0
    )
    _set_layer13_shifts(expected, 100.0, _SetDim)

    assert end == 2048
    _assert_same_ffn_units(actual, expected, 0, 2048)


def test_layer13_shr_substage_matches_legacy_units():
    """SHR rules (units 2048..4095) byte-identical to the SHR portion of
    ``_set_layer13_shifts``.
    """
    actual = _StubFFN(hidden_dim=4096)
    expected = _StubFFN(hidden_dim=4096)

    rules = _layer13_shr_rules(100.0)
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(
        actual, rules, dim_positions, start_unit=2048, S=100.0
    )
    _set_layer13_shifts(expected, 100.0, _SetDim)

    assert end == 4096
    _assert_same_ffn_units(actual, expected, 2048, 4096)


def test_layer13_shifts_declarative_matches_legacy_helper():
    """Full 4096-rule lowering byte-identical to ``_set_layer13_shifts``."""

    actual = _StubFFN(hidden_dim=4096)
    expected = _StubFFN(hidden_dim=4096)

    next_free = _bake_layer13_shifts(actual, 100.0, _SetDim)
    assert next_free == 4096, (
        f"_bake_layer13_shifts wrote {next_free} units, expected 4096"
    )
    _set_layer13_shifts(expected, 100.0, _SetDim)

    _assert_same_ffn(actual, expected)


def test_layer13_shifts_ir_passes_declaration_and_lowering_checks():
    """``compare_symbolic_to_lowered_ffn`` over the full 4096-rule IR has
    no declaration-semantics or lowering-contract failures.

    The per-cell weight check (W_up / b_up / W_gate / b_gate / W_down) and
    declaration resolution are byte-identical guarantees for the rule
    list. ``weight_output_mismatch`` failures from synthetic-state fan-in
    (the lookup table fires only on the exact 5-way condition match) are
    accepted: the byte-identity test above already pins forward semantics
    against the legacy helper, so structural correctness is the only new
    check needed here.
    """
    ir = _layer13_shifts_ir(100.0)
    names = Primitives.ffn_rule_dim_names(ir.layer(0).ffn.rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    report = compare_symbolic_to_lowered_ffn(
        ir,
        dim_positions,
        S=100.0,
        atol=1e-4,
        rtol=1e-4,
    )

    structural_failures = [
        issue for issue in report.issues
        if issue.kind in ("declaration_semantics", "lowering")
    ]
    assert not structural_failures, (
        "structural compare_symbolic_to_lowered_ffn failures: "
        + "\n".join(f"  [{i.kind}] {i.message}" for i in structural_failures)
    )
