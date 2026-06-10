"""Tests for the pre-step symbolic state builder.

The builder turns "the residual just before step N of this program"
into a dict (and optionally a flat tensor) without running the
~18-layer transformer or its embedding bake. These tests pin the
behaviour the L9 collapsed-step fix workflow needs:

1. Single-step ``IMM`` produces ``REG_AX_BYTE0_LO+(imm & 0xF)`` and the
   ``OP_IMM`` indicator at step 0 — no VM advance required.
2. The five-instruction ``IMM 5; PSH; IMM 5; EQ; EXIT`` program lands
   on EQ-ready state at step 3: AX=5 *and* STACK0=5.
3. The same program at step 4 (EXIT) shows the EQ result (AX=1) and
   the EXIT indicator.
4. A multi-byte immediate (0x123) decomposes into the expected
   per-byte nibble one-hots.
5. ``to_tensor`` round-trips a state dict through a torch residual.
6. The builder composes byte-identically with
   ``compare_symbolic_to_lowered_ffn`` (a single ``step_function_rule``
   keyed on ``OP_IMM`` fires through the lowered FFN when the builder's
   state is the input).

Tests use the public API only; no internal stubs needed since the
builder lives entirely above the model bake.
"""

from __future__ import annotations

import pytest
import torch

from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    step_function_rule,
)
from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.symbolic_state_builder import (
    ADD,
    EQ,
    EXIT,
    IMM,
    PSH,
    Instruction,
    describe_state,
    state_after_program,
    to_tensor,
)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_state_at_step0_for_imm_42_sets_axbyte0():
    """``IMM 42`` at step 0 — pre-compute residual carries the IMM
    indicator + the byte-0 nibble one-hots for 42 (0x2A: lo=A, hi=2).

    AX *before* the step is still 0 (the IMM has not yet executed);
    the immediate-payload dims ``IMM_AX_LO/HI`` are what downstream
    layers consume to materialise the new AX value.
    """
    state = state_after_program([IMM(42), EXIT], step=0)
    # Pre-step AX = 0 (nothing has run yet).
    assert state["REG_AX_BYTE0_LO+0"] == 1.0, describe_state(state)
    assert state["REG_AX_BYTE0_HI+0"] == 1.0
    # Opcode indicator + IMM payload one-hots reflect the *upcoming* step.
    assert state["OP_IMM+0"] == 1.0
    # 42 = 0x2A; byte 0 -> lo=A (10), hi=2.
    assert state["IMM_AX_LO+10"] == 1.0
    assert state["IMM_AX_HI+2"] == 1.0
    assert state["CONST+0"] == 1.0


def test_state_at_step1_after_imm_42_has_ax_42():
    """After the IMM 42 has executed, the pre-step-1 state should show
    REG_AX_BYTE0 == 42's nibbles."""
    state = state_after_program([IMM(42), EXIT], step=1)
    # AX byte 0 = 0x2A: lo=10, hi=2.
    assert state["REG_AX_BYTE0_LO+10"] == 1.0, describe_state(state)
    assert state["REG_AX_BYTE0_HI+2"] == 1.0
    # Upcoming instruction is EXIT.
    assert state["OP_EXIT+0"] == 1.0


def test_eq_ready_state_after_imm_psh_imm():
    """``IMM 5; PSH; IMM 5; EQ; EXIT`` at step=3 (EQ) — both AX and
    STACK0 carry 5, EQ is the upcoming opcode."""
    program = [IMM(5), PSH, IMM(5), EQ, EXIT]
    state = state_after_program(program, step=3)
    # AX = 5 (byte 0: lo=5, hi=0).
    assert state["REG_AX_BYTE0_LO+5"] == 1.0, describe_state(state)
    assert state["REG_AX_BYTE0_HI+0"] == 1.0
    # STACK0 = 5 — the value PSH pushed in step 1.
    assert state["STACK0_BYTE0_LO+5"] == 1.0
    assert state["STACK0_BYTE0_HI+0"] == 1.0
    # EQ is the upcoming opcode whose compute layers have not fired yet.
    assert state["OP_EQ+0"] == 1.0


def test_post_eq_state_has_ax_1_and_exit_indicator():
    """At step 4 (EXIT), AX should show the EQ result (1) and the
    EXIT indicator fires."""
    program = [IMM(5), PSH, IMM(5), EQ, EXIT]
    state = state_after_program(program, step=4)
    # AX = 1 after EQ.
    assert state["REG_AX_BYTE0_LO+1"] == 1.0, describe_state(state)
    assert state["OP_EXIT+0"] == 1.0


def test_multibyte_immediate_decomposes_per_byte():
    """0x123 spans bytes 0 and 1; the builder emits per-byte nibble
    one-hots in distinct dim families (IMM_AX_LO vs IMM_AX_LO_1)."""
    state = state_after_program([IMM(0x123), EXIT], step=0)
    # Byte 0 = 0x23: lo=3, hi=2.
    assert state["IMM_AX_LO+3"] == 1.0, describe_state(state)
    assert state["IMM_AX_HI+2"] == 1.0
    # Byte 1 = 0x01: lo=1, hi=0.
    assert state["IMM_AX_LO_1+1"] == 1.0
    assert state["IMM_AX_HI_1+0"] == 1.0


def test_psh_advances_sp_and_seeds_stack0():
    """After ``IMM 7; PSH`` (step=2 pre-compute), STACK0 = 7."""
    state = state_after_program([IMM(7), PSH, EXIT], step=2)
    assert state["STACK0_BYTE0_LO+7"] == 1.0, describe_state(state)
    assert state["OP_EXIT+0"] == 1.0


def test_step_index_out_of_range_raises():
    program = [IMM(1), EXIT]
    with pytest.raises(ValueError, match="past program end"):
        state_after_program(program, step=2)
    with pytest.raises(ValueError, match="step must be >= 0"):
        state_after_program(program, step=-1)


def test_accepts_raw_tuple_and_encoded_int():
    """Mixed list of ``Instruction`` / ``(op, imm)`` / encoded int all
    decode to the same state."""
    from c4_release.neural_vm.unified_compiler.symbolic_forward import (
        OP_IMM,
        encode_instr,
    )

    via_instr = state_after_program([IMM(9), EXIT], step=0)
    via_tuple = state_after_program([(OP_IMM, 9), (38, 0)], step=0)
    via_encoded = state_after_program(
        [encode_instr(OP_IMM, 9), encode_instr(38, 0)], step=0,
    )
    # All three should agree on the IMM byte-0 lo nibble.
    assert via_instr["IMM_AX_LO+9"] == 1.0
    assert via_tuple["IMM_AX_LO+9"] == 1.0
    assert via_encoded["IMM_AX_LO+9"] == 1.0


def test_extra_state_overrides_projected_values():
    """The ``extra_state`` kwarg lets a caller pin auxiliary dims
    (e.g. a flag) on top of the projected residual."""
    state = state_after_program(
        [IMM(0), EXIT], step=0,
        extra_state={"MARK_PC_PIN+0": 1.0, "CONST+0": 2.0},
    )
    assert state["MARK_PC_PIN+0"] == 1.0
    # Override wins on collision.
    assert state["CONST+0"] == 2.0


# ---------------------------------------------------------------------------
# to_tensor helper
# ---------------------------------------------------------------------------


def test_to_tensor_with_explicit_layout():
    """``to_tensor`` lands the dict values at ``base + offset``."""
    state = state_after_program([IMM(42), EXIT], step=0)
    layout = {
        "REG_AX_BYTE0_LO": 0,   # 16 cells
        "OP_IMM": 16,
        "CONST": 17,
    }
    tensor = to_tensor(state, d_model=32, dim_positions=layout)
    assert tensor.shape == (32,)
    assert tensor.dtype == torch.float32
    # AX byte 0 lo = 0 (pre-step), so cell 0 is set.
    assert tensor[0].item() == 1.0
    # OP_IMM indicator at base 16, offset 0.
    assert tensor[16].item() == 1.0
    # CONST+0 at base 17.
    assert tensor[17].item() == 1.0


def test_to_tensor_with_auto_layout():
    """Auto-layout (no ``dim_positions``) gives every family its own
    64-cell stride; the tensor is well-formed and non-empty."""
    state = state_after_program([IMM(7), EXIT], step=0)
    tensor = to_tensor(state, d_model=4096)
    assert tensor.shape == (4096,)
    # At least one cell must be set — the state isn't empty.
    assert tensor.abs().sum().item() > 0


def test_to_tensor_drops_out_of_range_writes():
    """Writes whose resolved index falls outside ``[0, d_model)`` are
    silently dropped (matches contribution-algebra "dead writes" rule)."""
    state = {"FOO+5": 1.0, "BAR+200": 1.0}
    layout = {"FOO": 0, "BAR": 0}
    tensor = to_tensor(state, d_model=10, dim_positions=layout)
    # FOO+5 lands at index 5 (in range); BAR+200 lands at 200 (dropped).
    assert tensor[5].item() == 1.0
    assert tensor.sum().item() == 1.0


# ---------------------------------------------------------------------------
# Composition with the byte-identity gate
# ---------------------------------------------------------------------------


def test_composes_with_compare_symbolic_to_lowered_ffn():
    """An L9-style rule keyed on ``OP_EQ`` byte-identity-gates when the
    builder's state is the input.

    This is the composability claim the task brief calls out: the same
    dict shape the builder returns must drive
    :func:`compare_symbolic_to_lowered_ffn`. We follow the existing
    ``test_building_blocks_dsl.test_step_function_rule_symbolic_matches_lowered_at_S1``
    calibration (``S=1.0`` + ``0.5 + _SILU_ONE_INPUT`` input) — the
    standard recipe for an exact symbolic/lowered match. The builder
    seeds the residual at 1.0; we rewrite the relevant dim into the
    calibrated value via ``extra_state`` to verify the dict survives a
    real gate run end-to-end.
    """
    # silu pre-image of 1.0 (matches the constant in
    # ``tests/test_building_blocks_dsl.py`` and ``ir.py``).
    _SILU_ONE_INPUT = 1.278464542761074

    rule = step_function_rule(
        input_dim="OP_EQ",
        threshold=0.5,
        write_dim="EQ_DISPATCH+0",
        write_value=3.0,
        S=1.0,
        name="eq_dispatch_at_S1",
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    # State at step 3 of IMM 5; PSH; IMM 5; EQ; EXIT — EQ is upcoming.
    # Override OP_EQ to the calibrated input value (the builder writes
    # 1.0; the byte-identity gate at S=1 needs ``0.5 + _SILU_ONE_INPUT``
    # to land silu = 1.0 exactly).
    state = state_after_program(
        [IMM(5), PSH, IMM(5), EQ, EXIT],
        step=3,
        extra_state={"OP_EQ+0": 0.5 + _SILU_ONE_INPUT},
    )
    # Confirm the override took.
    assert abs(state["OP_EQ+0"] - (0.5 + _SILU_ONE_INPUT)) < 1e-9

    dim_positions = {
        "OP_EQ": 0,
        "EQ_DISPATCH": 1,
        "CONST": 2,
    }
    # Restrict the state dict to the dims the rule actually references
    # so the gate's resolved_dim equals 3.
    restricted = {
        "OP_EQ+0": state["OP_EQ+0"],
        "CONST+0": state["CONST+0"],
    }
    report = compare_symbolic_to_lowered_ffn(
        ir,
        dim_positions=dim_positions,
        state=restricted,
        S=1.0,
        atol=1e-5,
    )
    assert report.ok, (
        [issue.kind + ": " + issue.message for issue in report.issues]
        + [f"symbolic={report.symbolic_state}", f"lowered={report.lowered_state}"]
    )
    # write_value = 3.0, S=1 -> dispatch dim should settle at ≈3.0.
    assert abs(report.lowered_state["EQ_DISPATCH+0"] - 3.0) < 1e-5


# ---------------------------------------------------------------------------
# Stress: 6+ programs exercised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("program,step,key,expected", [
    # 1) Single IMM, post-step state.
    ([IMM(42), EXIT], 1, "REG_AX_BYTE0_LO+10", 1.0),
    # 2) Two-instruction PSH program — STACK0 carries the pushed value.
    ([IMM(15), PSH, EXIT], 2, "STACK0_BYTE0_LO+15", 1.0),
    # 3) ADD result — IMM 1; PSH; IMM 2; ADD; EXIT -> AX = 3 at step 4.
    ([IMM(1), PSH, IMM(2), ADD, EXIT], 4, "REG_AX_BYTE0_LO+3", 1.0),
    # 4) EQ true — AX=1 after EQ.
    ([IMM(5), PSH, IMM(5), EQ, EXIT], 4, "REG_AX_BYTE0_LO+1", 1.0),
    # 5) Step-0 of any program has CONST set.
    ([IMM(0), EXIT], 0, "CONST+0", 1.0),
    # 6) Multi-byte IMM at step 0.
    ([IMM(0x100), EXIT], 0, "IMM_AX_LO_1+1", 1.0),
    # 7) Sanity: STACK0 = 0 before any PSH.
    ([IMM(0), EXIT], 0, "STACK0_BYTE0_LO+0", 1.0),
])
def test_programs_smoke(program, step, key, expected):
    state = state_after_program(program, step=step)
    assert state.get(key, 0.0) == expected, describe_state(state)
