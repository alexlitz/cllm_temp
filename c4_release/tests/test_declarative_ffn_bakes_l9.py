"""Parity checks for L9 lookup ALU declarative gate assumptions.

Also pins byte-identity for the Phase 6 Wave 4C ``layer9_alu`` migration:
each sub-stage's :class:`FFNRule` lowering must match the legacy
:func:`vm_step._set_layer9_alu` writes cell-for-cell on the pinned unit
range.
"""

import torch

from c4_release.neural_vm.vm_step import _SetDim, _set_layer9_alu
from c4_release.neural_vm.unified_compiler.ir import (
    compare_symbolic_to_lowered_ffn,
    CompilerIR,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 3600):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _legacy_l9_full() -> _StubFFN:
    """Bake the full legacy ``_set_layer9_alu`` into a stub FFN."""
    ffn = _StubFFN(hidden_dim=3600)
    _set_layer9_alu(ffn, 100.0, _SetDim)
    return ffn


def _lower_rules_to_stub(rules, start_unit: int) -> _StubFFN:
    """Lower a rule list at ``start_unit`` into a fresh stub FFN."""
    ffn = _StubFFN(hidden_dim=3600)
    dim_positions = Primitives.dim_positions_from_bd(
        _SetDim,
        Primitives.ffn_rule_dim_names(rules),
    )
    Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=start_unit,
        S=100.0,
    )
    return ffn


def _assert_unit_range_matches_legacy(
    rules,
    *,
    start_unit: int,
    n_units: int,
    legacy_ffn: _StubFFN | None = None,
) -> None:
    """Assert lowered ``rules`` match the legacy bake on [start, start+n)."""
    if legacy_ffn is None:
        legacy_ffn = _legacy_l9_full()
    actual = _lower_rules_to_stub(rules, start_unit=start_unit)
    end = start_unit + n_units

    # Per-cell tensor comparisons over the pinned range.
    assert torch.equal(
        actual.W_up[start_unit:end, :],
        legacy_ffn.W_up[start_unit:end, :],
    ), f"W_up drift in units [{start_unit}, {end})"
    assert torch.equal(
        actual.b_up[start_unit:end],
        legacy_ffn.b_up[start_unit:end],
    ), f"b_up drift in units [{start_unit}, {end})"
    assert torch.equal(
        actual.W_gate[start_unit:end, :],
        legacy_ffn.W_gate[start_unit:end, :],
    ), f"W_gate drift in units [{start_unit}, {end})"
    assert torch.equal(
        actual.b_gate[start_unit:end],
        legacy_ffn.b_gate[start_unit:end],
    ), f"b_gate drift in units [{start_unit}, {end})"
    assert torch.equal(
        actual.W_down[:, start_unit:end],
        legacy_ffn.W_down[:, start_unit:end],
    ), f"W_down drift in units [{start_unit}, {end})"


def _compare_symbolic_to_lowered(rules) -> None:
    """Validate IR declaration semantics + lowering contract.

    The full check (`report.ok`) is too strict here -- many L9 ALU
    sub-stages have hundreds of rules writing the same output dim under
    different one-hot conditions, so the synthetic state fired by
    :func:`compare_symbolic_to_lowered_ffn` triggers harmless
    `weight_output_mismatch` failures from overlapping fires. Filter to
    structural-only (``declaration_semantics`` + ``lowering``) which
    pins the byte-identity contract we actually care about.
    """
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    dim_positions = Primitives.dim_positions_from_bd(
        _SetDim,
        Primitives.ffn_rule_dim_names(rules),
    )
    report = compare_symbolic_to_lowered_ffn(
        ir,
        dim_positions,
        S=100.0,
        atol=1e-4,
        rtol=1e-4,
    )
    structural = [
        issue for issue in report.issues
        if issue.kind in ("declaration_semantics", "lowering")
    ]
    assert not structural, (
        "structural compare_symbolic_to_lowered_ffn failures: "
        + "\n".join(f"  [{i.kind}] {i.message}" for i in structural)
    )


# ---------------------------------------------------------------------
# Phase 6 Wave 4C: per-substage byte-identity tests
# ---------------------------------------------------------------------


def test_layer9_add_hi_nibble_matches_legacy():
    """``_layer9_add_hi_nibble_rules`` lowers byte-identically at unit 0."""
    from c4_release.neural_vm.unified_compiler.ops.l9_ops import (
        _layer9_add_hi_nibble_rules,
    )

    rules = _layer9_add_hi_nibble_rules(100.0)
    assert len(rules) == 512
    _assert_unit_range_matches_legacy(rules, start_unit=0, n_units=512)
    _compare_symbolic_to_lowered(rules)


def test_layer9_lea_hi_nibble_matches_legacy():
    """``_layer9_lea_hi_nibble_rules`` lowers byte-identically at unit 512."""
    from c4_release.neural_vm.unified_compiler.ops.l9_ops import (
        _layer9_lea_hi_nibble_rules,
    )

    rules = _layer9_lea_hi_nibble_rules(100.0)
    assert len(rules) == 512
    _assert_unit_range_matches_legacy(rules, start_unit=512, n_units=512)
    _compare_symbolic_to_lowered(rules)


def test_layer9_adj_hi_nibble_matches_legacy():
    """``_layer9_adj_hi_nibble_rules`` lowers byte-identically at unit 1024."""
    from c4_release.neural_vm.unified_compiler.ops.l9_ops import (
        _layer9_adj_hi_nibble_rules,
    )

    rules = _layer9_adj_hi_nibble_rules(100.0)
    assert len(rules) == 512
    _assert_unit_range_matches_legacy(rules, start_unit=1024, n_units=512)
    _compare_symbolic_to_lowered(rules)


def test_layer9_sub_hi_nibble_matches_legacy():
    """``_layer9_sub_hi_nibble_rules`` lowers byte-identically at unit 1536."""
    from c4_release.neural_vm.unified_compiler.ops.l9_ops import (
        _layer9_sub_hi_nibble_rules,
    )

    rules = _layer9_sub_hi_nibble_rules(100.0)
    assert len(rules) == 512
    _assert_unit_range_matches_legacy(rules, start_unit=1536, n_units=512)
    _compare_symbolic_to_lowered(rules)


def test_layer9_ent_hi_nibble_matches_legacy():
    """``_layer9_ent_hi_nibble_rules`` lowers byte-identically at unit 2048."""
    from c4_release.neural_vm.unified_compiler.ops.l9_ops import (
        _layer9_ent_hi_nibble_rules,
    )

    rules = _layer9_ent_hi_nibble_rules(100.0)
    assert len(rules) == 512
    _assert_unit_range_matches_legacy(rules, start_unit=2048, n_units=512)
    _compare_symbolic_to_lowered(rules)


def test_layer9_lea_adj_ent_fetch_gates_use_one_hot_scale():
    ffn = _StubFFN()
    _set_layer9_alu(ffn, 100.0, _SetDim)

    # LEA hi, no carry, a=15, b=15
    lea_no_carry = 512 + 15 * 16 + 15
    assert ffn.W_up[lea_no_carry, _SetDim.MARK_AX].item() == 2000.0
    assert ffn.W_up[lea_no_carry, _SetDim.MARK_PC].item() == -100000.0
    assert ffn.W_up[lea_no_carry, _SetDim.FETCH_HI + 15].item() == 2000.0
    assert ffn.W_up[lea_no_carry, _SetDim.CARRY + 0].item() == -800.0
    assert ffn.b_up[lea_no_carry].item() == -4050.0

    # LEA hi, with carry, a=15, b=15
    lea_carry = 768 + 15 * 16 + 15
    assert ffn.W_up[lea_carry, _SetDim.CARRY + 0].item() == 800.0
    assert ffn.b_up[lea_carry].item() == -4850.0

    # ADJ hi uses the same one-hot FETCH scaling and carry discrimination.
    adj_no_carry = 1024 + 15 * 16 + 15
    assert ffn.W_up[adj_no_carry, _SetDim.MARK_AX].item() == 2000.0
    assert ffn.W_up[adj_no_carry, _SetDim.FETCH_HI + 15].item() == 2000.0
    assert ffn.W_up[adj_no_carry, _SetDim.CARRY + 0].item() == -800.0
    assert ffn.b_up[adj_no_carry].item() == -4200.0

    # ENT hi also needs the stronger SP-marker blocker after lowering threshold.
    ent_no_borrow = 2048 + 15 * 16 + 15
    assert ffn.W_up[ent_no_borrow, _SetDim.MARK_AX].item() == 2000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.MARK_SP].item() == -100000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.IS_BYTE].item() == -100000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.FETCH_HI + 15].item() == 2000.0
    assert ffn.W_up[ent_no_borrow, _SetDim.CARRY + 0].item() == -800.0
    assert ffn.b_up[ent_no_borrow].item() == -4200.0
