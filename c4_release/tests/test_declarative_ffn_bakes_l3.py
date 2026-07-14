"""Parity tests for the L3 FFN bake migrated in Phase 6 Wave 3D.

Covers:
- ``layer3_ffn`` -- migrated from ``vm_step._set_layer3_ffn`` +
  ``_suppress_stack0_marker_carry_projection`` +
  ``_add_pc_byte1_output_rules`` to a 136-rule
  :class:`c4_release.neural_vm.unified_compiler.ir.FFNRule` list lowered via
  :func:`Primitives.lower_ffn_rules`.
- ``layer3_convo_io_state_init`` and ``convo_io_step_resume`` -- already
  declarative pre-migration; the parity test here pins the rule shape so
  the byte-identity guard catches future drift.
"""

import torch

from c4_release.neural_vm.vm_step import _SetDim
from tests.oracles.vm_step_layer_bakes import _set_layer3_ffn
from c4_release.neural_vm.unified_compiler.ops.l3_ops import (
    _convo_io_state_init_ir,
    _register_default_ffn_ir,
    _register_default_ffn_rules,
    _pc_byte1_output_rules,
    _lower_register_default_ffn_ir,
    _lower_pc_byte1_output_rules_ir,
    _suppress_stack0_marker_carry_projection,
)
from c4_release.neural_vm.unified_compiler.ops.flag_gated_ops import (
    _convo_io_step_resume_ir,
)


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 200):
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


def _legacy_set_layer3_ffn_full(ffn, S, BD):
    """Run the full legacy bake stack: ``_set_layer3_ffn`` +
    suppressor + ``_add_pc_byte1_output_rules`` (inlined here to
    avoid coupling to the now-migrated production helper).
    """
    _set_layer3_ffn(ffn, S, BD)
    _suppress_stack0_marker_carry_projection(ffn, S, BD)

    # Inlined pre-migration _add_pc_byte1_output_rules body
    # (so this test pins the exact legacy weights, not whatever the
    # migrated helper now produces).
    PC_I = 0
    unit = 134
    common = (
        (BD.H1 + PC_I, 1.0),
        (BD.BYTE_INDEX_0, 1.0),
        (BD.IS_BYTE, 1.0),
        (BD.HAS_SE, 1.0),
    )
    extra_wrap = common + (
        (BD.CLEAN_EMBED_LO + 2, 1.0),
        (BD.CLEAN_EMBED_HI + 0, 1.0),
    )
    for dim, weight in extra_wrap:
        ffn.W_up[unit, dim] = S * weight
    ffn.b_up[unit] = -S * 5.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -500.0 / S
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 500.0 / S
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 500.0 / S
    unit += 1

    for dim, weight in common:
        ffn.W_up[unit, dim] = S * weight
    ffn.W_up[unit, BD.TEMP + 1] = S
    ffn.W_up[unit, BD.TEMP + 16] = S
    for hi in range(5):
        ffn.W_up[unit, BD.CLEAN_EMBED_HI + hi] = S
    ffn.b_up[unit] = -S * 6.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -500.0 / S
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 500.0 / S
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 500.0 / S


def test_layer3_ffn_rule_count_is_136():
    """Pin the rule count so accidental adds/removes are caught."""
    main = _register_default_ffn_rules(100.0)
    trailing = _pc_byte1_output_rules(100.0)
    assert len(main) == 134
    assert len(trailing) == 2
    ir = _register_default_ffn_ir()
    assert len(ir.layer(0).ffn.rules) == 136


def test_layer3_ffn_declarative_matches_legacy_helper():
    """Byte-identity: 136-rule declarative lowering produces exactly the
    same ``W_up`` / ``b_up`` / ``W_gate`` / ``b_gate`` / ``W_down``
    tensors as ``_set_layer3_ffn`` + suppressor + legacy
    ``_add_pc_byte1_output_rules``.
    """
    actual = _StubFFN(hidden_dim=200)
    expected = _StubFFN(hidden_dim=200)

    next_free = _lower_register_default_ffn_ir(actual, 100.0, _SetDim)
    assert next_free == 134, (
        f"main rule lowering wrote {next_free} units, expected 134"
    )
    _lower_pc_byte1_output_rules_ir(
        actual, 100.0, _SetDim, start_unit=next_free,
    )

    _legacy_set_layer3_ffn_full(expected, 100.0, _SetDim)

    _assert_same_ffn(actual, expected)


def test_layer3_ffn_stack0_carry_projection_writes_are_suppressed():
    """The 32 STACK0 carry-projection rules (units 50..81) must emit
    *zero* ``W_down`` writes -- that is the declarative replacement for
    the legacy
    ``_suppress_stack0_marker_carry_projection`` post-pass.
    """
    ffn = _StubFFN(hidden_dim=200)
    _lower_register_default_ffn_ir(ffn, 100.0, _SetDim)

    for unit in range(50, 82):
        # W_up / W_gate / b_up survive (the rule still fires).
        assert ffn.b_up[unit].item() != 0.0, (
            f"unit {unit} has zero b_up; rule did not fire"
        )
        # W_down column must be entirely zero.
        col = ffn.W_down[:, unit]
        assert torch.equal(col, torch.zeros_like(col)), (
            f"unit {unit} W_down column is non-zero; STACK0 carry "
            f"projection writes are NOT being suppressed"
        )


def test_layer3_convo_io_state_init_ir_pins_single_rule():
    ir = _convo_io_state_init_ir()
    rules = ir.layer(0).ffn.rules
    assert len(rules) == 1
    rule = rules[0]
    assert rule.name == "convo_io_enter_output_mode"
    # Pin the threshold/writes shape so future edits are noticed.
    assert rule.threshold == 0.5
    assert len(rule.writes) == 1
    assert rule.writes[0].dim.name == "IO_IN_OUTPUT_MODE"


def test_convo_io_step_resume_ir_pins_single_rule():
    ir = _convo_io_step_resume_ir()
    rules = ir.layer(0).ffn.rules
    assert len(rules) == 1
    rule = rules[0]
    assert rule.name == "convo_io_step_resume"
    assert rule.threshold == 0.5
    # Three coupled writes: enter step (NEXT_PC=1), exit output mode
    # (IO_IN_OUTPUT_MODE=-1), clear IO_STATE.
    write_dims = sorted(w.dim.name for w in rule.writes)
    assert write_dims == ["IO_IN_OUTPUT_MODE", "IO_STATE", "NEXT_PC"]
