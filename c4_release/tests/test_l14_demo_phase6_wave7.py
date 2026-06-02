"""Regression coverage for the Phase 6 Wave 7 declare-only demo op.

The demo (``layer14_demo_phase6_wave7``) demonstrates the
``docs/HOW_TO_ADD_A_CORRECTIVE_OP.md`` end-to-end flow:

* declare a single :class:`FFNRule` in ``_layer14_demo_phase6_wave7_rules``,
* let the L14 cleanup chain's :class:`FFNUnitAllocator` first-fit the unit
  index (``pin=None`` in :data:`_L14_CLEANUP_CHAIN_LAYOUT`),
* lower via :meth:`CompilerIR.lower_ffn`,
* clear :func:`compare_symbolic_to_lowered_ffn` byte-identity gate.

These tests pin those four steps so a future regression bisect sees a
focused failure rather than a corpus-level drift.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ir import (  # noqa: E402
    CompilerIR,
    compare_symbolic_to_lowered_ffn,
)
from neural_vm.unified_compiler.ops.l14_ops import (  # noqa: E402
    _L14_CLEANUP_CHAIN_LAYOUT,
    _l14_chain_alloc,
    _layer14_demo_phase6_wave7_ir,
    _layer14_demo_phase6_wave7_rules,
    make_layer14_demo_phase6_wave7_op,
)
from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops  # noqa: E402
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.unified_compiler.ops.shared import _as_setdim_proxy  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402


class _StubFFN:
    """Minimal FFN with just the four weight buffers ``lower_ffn`` writes."""

    def __init__(self, *, d_model: int = 512, hidden_dim: int = 2048):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _apply_stub_ffn(ffn: _StubFFN, x: torch.Tensor) -> torch.Tensor:
    up = torch.nn.functional.silu(x @ ffn.W_up.t() + ffn.b_up)
    gate = x @ ffn.W_gate.t() + ffn.b_gate
    return x + (up * gate) @ ffn.W_down.t()


def test_demo_rules_are_a_single_never_fires_rule():
    """One :class:`FFNRule`, scope ``__never_fires__``, write weight 0."""

    rules = _layer14_demo_phase6_wave7_rules(100.0)
    assert len(rules) == 1
    (rule,) = rules
    assert rule.name == "l14_demo_phase6_wave7_decl_only_noop"
    assert rule.scope == "__never_fires__"
    # Single CONST=-100 condition guarantees pre-SiLU is -100*S at every
    # row, so the unit is byte-identically silent on every position.
    assert len(rule.conditions) == 1
    cond = rule.conditions[0]
    assert cond.dim.name == "CONST" and cond.dim.offset == 0
    assert cond.weight == -100.0
    # The single TEMP+0 write has weight 0, so even if the unit somehow
    # fired the residual cell would still be unchanged.
    assert len(rule.writes) == 1
    write = rule.writes[0]
    assert write.dim.name == "TEMP" and write.dim.offset == 0
    assert write.weight == 0.0


def test_demo_ir_clears_byte_identity_gate():
    """``compare_symbolic_to_lowered_ffn`` reports OK with no issues."""

    ir = _layer14_demo_phase6_wave7_ir()
    dim_positions = {name: getattr(_SetDim, name) for name in ("CONST", "TEMP")}
    report = compare_symbolic_to_lowered_ffn(ir, dim_positions, S=100.0)
    assert report.ok, report.format()
    assert report.issues == []


def test_demo_auto_fit_lands_after_alu_nocarry_chain_tail():
    """``pin=None`` first-fit picks the slot after the prior chain claims.

    The L14 cleanup chain ends at unit 1874 before this demo
    (``layer14_alu_nocarry_ax_bytes_zero`` occupies 1870..1873). The demo
    is a single-unit rule, so the allocator picks unit 1874 -- proving the
    auto-fit story without the author writing the offset.
    """

    assert _L14_CLEANUP_CHAIN_LAYOUT["layer14_demo_phase6_wave7"] == (None, 1)
    start = _l14_chain_alloc("layer14_demo_phase6_wave7")
    assert start == 1874


def test_demo_bake_path_writes_zero_residual_delta():
    """Lowered FFN forward leaves the residual stream untouched.

    Mirrors the bake function in :func:`make_layer14_demo_phase6_wave7_op`:
    allocate via the chain, lower via :meth:`CompilerIR.lower_ffn`, and run
    a representative residual row through the SwiGLU forward. The rule's
    pre-SiLU input is ``-100 * S`` at every position, so the unit's
    contribution to the post-down delta is numerically zero.
    """

    ffn = _StubFFN()
    start_unit = _l14_chain_alloc("layer14_demo_phase6_wave7")
    ir = _layer14_demo_phase6_wave7_ir(100.0)
    rules = ir.layer(0).ffn.rules
    dim_map = Primitives.dim_positions_from_bd(
        _as_setdim_proxy({name: getattr(_SetDim, name) for name in ("CONST", "TEMP")}),
        Primitives.ffn_rule_dim_names(rules),
    )
    next_unit = ir.lower_ffn(ffn, dim_map, start_unit=start_unit, S=100.0)
    assert next_unit == start_unit + 1

    # The unit's W_up[CONST] should be -10000 (S * conditions.CONST.weight).
    # Verify the lowering contract before running the forward.
    assert ffn.W_up[start_unit, _SetDim.CONST].item() == -10_000.0
    # The W_down write weight on TEMP+0 must be 0 -- the rule's writes
    # specify weight=0 so no residual cell receives a contribution even if
    # SiLU somehow returned a nonzero value.
    assert ffn.W_down[_SetDim.TEMP, start_unit].item() == 0.0

    # Forward through a representative residual row. CONST is always 1.0;
    # IS_BYTE and a few opcode flags exercise the lowered weights through
    # a realistic input. The delta must be 0 at every dim.
    x = torch.zeros(1, 1, 512)
    x[..., _SetDim.CONST] = 1.0
    x[..., _SetDim.IS_BYTE] = 1.0
    x[..., _SetDim.OP_EQ] = 1.0
    x[..., _SetDim.MARK_AX] = 1.0

    y = _apply_stub_ffn(ffn, x)
    delta = (y - x).abs()
    assert float(delta.max().item()) == 0.0


def test_demo_op_is_registered_in_all_core_ops():
    """``all_core_ops()`` includes the demo op exactly once.

    Catches accidental deregistration or duplicate registration during
    future L14 chain edits.
    """

    ops = all_core_ops()
    demo_ops = [op for op in ops if op.name == "layer14_demo_phase6_wave7"]
    assert len(demo_ops) == 1
    op = demo_ops[0]
    assert op.kind == "block"
    assert op.layer_idx == 14
    assert op.migrated is True
    assert op.declarative_authority == "spec_generated"
    # Carries the chain-tail ``ffn_units_used`` so the per-block FFN
    # sizing pass sees the correct cumulative width.
    assert op.ffn_units_used == 1875


def test_demo_op_factory_returns_consistent_operation():
    """``make_layer14_demo_phase6_wave7_op()`` is deterministic.

    Two calls return identical fields. The factory is a pure constructor;
    nothing about the demo depends on environment state.
    """

    a = make_layer14_demo_phase6_wave7_op()
    b = make_layer14_demo_phase6_wave7_op()
    assert a.name == b.name == "layer14_demo_phase6_wave7"
    assert a.reads == b.reads == {"CONST"}
    assert a.writes == b.writes == {"TEMP"}
    assert a.phase == b.phase == 14.95
    # IR rules round-trip identically (frozen dataclasses + tuples).
    assert (
        tuple(r.name for r in a.compiler_ir.layer(0).ffn.rules)
        == tuple(r.name for r in b.compiler_ir.layer(0).ffn.rules)
        == ("l14_demo_phase6_wave7_decl_only_noop",)
    )
