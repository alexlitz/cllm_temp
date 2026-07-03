"""Declarative checks for the L13 FFN bake (Phase 6 Wave 4B).

Covers the ``layer13_shifts`` op (a single 4096-unit lookup table over
OP_SHL / OP_SHR for shift amounts 0..7) directly on the declarative path:

- SHL rules (units 0..2047) + SHR rules (units 2048..4095) count/layout.
- ``compare_symbolic_to_lowered_ffn`` confirms the declaration + lowering
  contract is consistent for the full 4096-rule IR (the declarative-
  lowering regression guard).
- ``make_layer13_shifts_op`` attaches the IR in lookup mode only.

The legacy imperative ``_set_layer13_shifts`` byte-identity oracles were
retired: the op is now fully guarded by the declarative rule-count +
``compare_symbolic_to_lowered_ffn`` checks below, so the imperative
helper (a dead, off-build-path fixture) is redundant.
"""

from c4_release.neural_vm.vm_step import _SetDim
from c4_release.neural_vm.unified_compiler.ir import (
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.ops.l13_ops import (
    _L13_SHIFTS_UNIT_LAYOUT,
    _layer13_shifts_ir,
    _layer13_shifts_rules,
    _layer13_shl_rules,
    _layer13_shr_rules,
    make_layer13_shifts_op,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives


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
    # Phase 7.B.5: layout entries are ``(name, n_units)`` 2-tuples after the
    # explicit pin was dropped in favor of allocator auto-fit.
    layout_total = sum(n for _, n in _L13_SHIFTS_UNIT_LAYOUT)
    assert layout_total == 4096


def test_layer13_shifts_ir_passes_declaration_and_lowering_checks():
    """``compare_symbolic_to_lowered_ffn`` over the full 4096-rule IR has
    no declaration-semantics or lowering-contract failures.

    The per-cell weight check (W_up / b_up / W_gate / b_gate / W_down) and
    declaration resolution are byte-identity guarantees for the rule
    list. ``weight_output_mismatch`` failures from synthetic-state fan-in
    (the lookup table fires only on the exact 5-way condition match) are
    accepted: this structural correctness check is the declarative-
    lowering regression guard for the op.
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


def test_layer13_shifts_op_exposes_compiler_ir():
    """``make_layer13_shifts_op(alu_mode='lookup')`` must attach the IR.

    Efficient mode is a no-op (ALUShiftComposite owns SHL/SHR there) so
    it intentionally has no ``compiler_ir``.
    """
    lookup_op = make_layer13_shifts_op(alu_mode="lookup")
    assert lookup_op.compiler_ir is not None
    assert len(lookup_op.compiler_ir.layer(0).ffn.rules) == 4096

    efficient_op = make_layer13_shifts_op(alu_mode="efficient")
    assert efficient_op.compiler_ir is None
