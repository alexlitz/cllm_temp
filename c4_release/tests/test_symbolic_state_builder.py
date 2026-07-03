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
from c4_release.neural_vm.verification.symbolic_state_builder import (
    ADD,
    EQ,
    EXIT,
    IMM,
    PSH,
    Instruction,
    describe_state,
    state_after_program,
    state_at_step_end,
    to_tensor,
    to_tensor_at_step_end,
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
    from c4_release.neural_vm.verification.symbolic_forward import (
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


# ---------------------------------------------------------------------------
# STEP_END residual builder — sibling of ``state_after_program`` that
# models the L0/L1 relay's broadcast target.
# ---------------------------------------------------------------------------
#
# The brief (2026-06-10): "extend the symbolic state builder to model the
# STEP_END position's residual — not just the MARK_AX position."
# STEP_END at step N has the operand state that the L0/L1 relay has
# deposited there: OP_<NAME>, REG_*/STACK0 byte one-hots, AX_CARRY_LO/HI,
# plus the *previous* step's ALU_LO/HI and CMP staging (which the relay
# carries forward into the new step's STEP_END window until L9's clear
# rules fire).
#
# Acceptance criteria from the brief:
#   1. At step 3 of ``IMM 5; PSH; IMM 5; EQ; EXIT``, STEP_END state
#      has OP_EQ active, AX_CARRY_LO[5]=1, STACK0_BYTE0_LO[5]=1.
#   2. At step 4 (post-EQ), STEP_END state has CMP+2=1 (lo_eq fired
#      during the EQ at step 3).
#   3. Composes byte-identically with ``compare_symbolic_to_lowered_ffn``.


def test_step_end_eq_ready_at_step3_carries_operands():
    """Brief acceptance #1: at the STEP_END of step 3 (EQ) of
    ``IMM 5; PSH; IMM 5; EQ; EXIT`` the in-flight operand state is
    visible — both AX_CARRY_LO+5 (operand B = AX) and STACK0_BYTE0_LO+5
    (operand A = top of stack), with OP_EQ as the step's opcode."""
    program = [IMM(5), PSH, IMM(5), EQ, EXIT]
    state = state_at_step_end(program, step=3)
    assert state["OP_EQ+0"] == 1.0, describe_state(state)
    assert state["AX_CARRY_LO+5"] == 1.0
    assert state["AX_CARRY_HI+0"] == 1.0
    assert state["STACK0_BYTE0_LO+5"] == 1.0
    assert state["STACK0_BYTE0_HI+0"] == 1.0
    # STEP_END marker is what scopes a migrated rule to this row.
    assert state["MARK_SE+0"] == 1.0
    assert state["CONST+0"] == 1.0


def test_step_end_post_eq_has_cmp_lo_eq_relayed():
    """Brief acceptance #2: at the STEP_END of step 4 (EXIT) the prior
    step's EQ partial flags are still in the relay window. With operands
    5 and 5, both hi_eq (CMP+1) and lo_eq (CMP+2) fire; hi_lt (CMP+0)
    and lo_lt (CMP+3) stay cold."""
    program = [IMM(5), PSH, IMM(5), EQ, EXIT]
    state = state_at_step_end(program, step=4)
    assert state["CMP+2"] == 1.0, describe_state(state)
    assert state["CMP+1"] == 1.0
    assert state["CMP+0"] == 0.0
    assert state["CMP+3"] == 0.0
    # Step 4's opcode is EXIT.
    assert state["OP_EXIT+0"] == 1.0
    # AX at the start of step 4 = the EQ result.
    assert state["REG_AX_BYTE0_LO+1"] == 1.0
    # ALU staging from step 3's EQ: popped operand A = 5.
    assert state["ALU_LO+5"] == 1.0
    assert state["ALU_HI+0"] == 1.0


def test_step_end_step0_has_no_prior_cmp_or_alu_flags():
    """Step 0 has no previous step, so no CMP / ALU staging is emitted."""
    state = state_at_step_end([IMM(5), PSH, EXIT], step=0)
    # OP_IMM is the current step's opcode.
    assert state["OP_IMM+0"] == 1.0
    # No CMP flags emitted (no prior CMP step).
    assert "CMP+0" not in state or state["CMP+0"] == 0.0
    assert "CMP+1" not in state or state["CMP+1"] == 0.0
    assert "CMP+2" not in state or state["CMP+2"] == 0.0
    assert "CMP+3" not in state or state["CMP+3"] == 0.0
    # No ALU staging either.
    assert "ALU_LO+0" not in state or sum(
        state.get(f"ALU_LO+{k}", 0.0) for k in range(16)
    ) == 0.0
    # MARK_SE still fires (this is a STEP_END residual).
    assert state["MARK_SE+0"] == 1.0


def test_step_end_after_imm_no_cmp_no_alu_relay():
    """After an IMM (non-ALU op), step 2's STEP_END should not carry
    CMP / ALU staging — IMM doesn't pop a stack operand and the L9
    clear rules wipe ALU_LO/HI on non-ALU ops."""
    program = [IMM(0), IMM(7), EXIT]
    state = state_at_step_end(program, step=2)
    # Prev step was IMM 7. AX at start of step 2 = 7.
    assert state["REG_AX_BYTE0_LO+7"] == 1.0
    assert state["AX_CARRY_LO+7"] == 1.0
    # CMP flags absent (or zero) — IMM did not run the cmp cascade.
    for off in range(4):
        assert state.get(f"CMP+{off}", 0.0) == 0.0
    # ALU staging absent.
    for off in range(16):
        assert state.get(f"ALU_LO+{off}", 0.0) == 0.0
        assert state.get(f"ALU_HI+{off}", 0.0) == 0.0


def test_step_end_after_lt_relays_partial_flags():
    """LT at step 3 of ``IMM 3; PSH; IMM 7; LT; EXIT`` (top=3, ax=7):
    hi_eq fires (0==0), lo_lt fires (3<7), hi_lt doesn't (0<0 false),
    lo_eq doesn't (3==7 false). At step 4's STEP_END the relay carries
    these partials forward."""
    from c4_release.neural_vm.verification.symbolic_state_builder import (
        LT,
    )
    program = [IMM(3), PSH, IMM(7), LT, EXIT]
    state = state_at_step_end(program, step=4)
    assert state["CMP+1"] == 1.0, describe_state(state)   # hi_eq
    assert state["CMP+3"] == 1.0                          # lo_lt
    assert state["CMP+0"] == 0.0                          # hi_lt
    assert state["CMP+2"] == 0.0                          # lo_eq
    # ALU_LO+3 staged (operand A = popped top = 3).
    assert state["ALU_LO+3"] == 1.0


def test_step_end_extra_state_overrides_projection():
    """The ``extra_state`` kwarg lets a caller pin auxiliary dims on top
    of the projected STEP_END residual (same contract as
    state_after_program)."""
    state = state_at_step_end(
        [IMM(0), EXIT], step=0,
        extra_state={"NEXT_PC+0": 1.0, "CONST+0": 2.0},
    )
    assert state["NEXT_PC+0"] == 1.0
    assert state["CONST+0"] == 2.0
    assert state["MARK_SE+0"] == 1.0


def test_step_end_rejects_out_of_range_step():
    program = [IMM(1), EXIT]
    with pytest.raises(ValueError, match="past program end"):
        state_at_step_end(program, step=2)
    with pytest.raises(ValueError, match="step must be >= 0"):
        state_at_step_end(program, step=-1)


def test_to_tensor_at_step_end_lays_out_residual():
    """``to_tensor_at_step_end`` is a thin wrapper over :func:`to_tensor`;
    it must honour the explicit ``dim_positions`` layout the same way."""
    state = state_at_step_end(
        [IMM(5), PSH, IMM(5), EQ, EXIT], step=3,
    )
    layout = {
        "OP_EQ": 0,
        "AX_CARRY_LO": 1,        # 16 cells
        "STACK0_BYTE0_LO": 17,   # 16 cells
        "MARK_SE": 33,
        "CONST": 34,
    }
    tensor = to_tensor_at_step_end(state, d_model=64, dim_positions=layout)
    assert tensor.shape == (64,)
    assert tensor.dtype == torch.float32
    # OP_EQ indicator at base 0.
    assert tensor[0].item() == 1.0
    # AX_CARRY_LO+5 -> base 1 + 5 = 6.
    assert tensor[1 + 5].item() == 1.0
    # STACK0_BYTE0_LO+5 -> base 17 + 5 = 22.
    assert tensor[17 + 5].item() == 1.0
    # MARK_SE and CONST.
    assert tensor[33].item() == 1.0
    assert tensor[34].item() == 1.0


def test_step_end_composes_with_compare_symbolic_to_lowered_ffn():
    """Brief acceptance #3: a STEP_END-scoped rule keyed on ``MARK_SE``
    + ``OP_EQ`` byte-identity-gates when ``state_at_step_end`` provides
    the input residual. This exercises the same composability claim as
    ``test_composes_with_compare_symbolic_to_lowered_ffn`` above, but
    on the STEP_END builder.
    """
    _SILU_ONE_INPUT = 1.278464542761074

    # Single rule: fires at MARK_SE on OP_EQ steps, writes a dispatch dim.
    rule = step_function_rule(
        input_dim="OP_EQ",
        threshold=0.5,
        write_dim="STEP_END_EQ_DISPATCH+0",
        write_value=2.0,
        S=1.0,
        name="step_end_eq_dispatch_at_S1",
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    state = state_at_step_end(
        [IMM(5), PSH, IMM(5), EQ, EXIT],
        step=3,
        extra_state={"OP_EQ+0": 0.5 + _SILU_ONE_INPUT},
    )
    assert state["MARK_SE+0"] == 1.0
    assert abs(state["OP_EQ+0"] - (0.5 + _SILU_ONE_INPUT)) < 1e-9

    dim_positions = {
        "OP_EQ": 0,
        "STEP_END_EQ_DISPATCH": 1,
        "CONST": 2,
    }
    # Restrict to the rule's reads so resolved_dim equals 3.
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
        + [f"symbolic={report.symbolic_state}",
           f"lowered={report.lowered_state}"]
    )
    assert abs(report.lowered_state["STEP_END_EQ_DISPATCH+0"] - 2.0) < 1e-5


@pytest.mark.parametrize("program,step,key,expected", [
    # 1) Step-0 STEP_END: opcode is current step's opcode, regardless of
    #    register state.
    ([IMM(0), EXIT], 0, "OP_IMM+0", 1.0),
    # 2) STACK0 byte 0 at PSH-ed value.
    ([IMM(0xAB), PSH, EXIT], 2, "STACK0_BYTE0_LO+11", 1.0),  # 0xAB lo=B(11)
    # 3) AX_CARRY_HI for AX=0x80 — hi nibble 8.
    ([IMM(0x80), EXIT], 1, "AX_CARRY_HI+8", 1.0),
    # 4) Post-ADD ALU staging — operand A = 1.
    ([IMM(1), PSH, IMM(2), ADD, EXIT], 4, "ALU_LO+1", 1.0),
    # 5) Post-EQ CMP+1 (hi_eq) fires.
    ([IMM(5), PSH, IMM(5), EQ, EXIT], 4, "CMP+1", 1.0),
    # 6) MARK_SE fires unconditionally — it's the row identity dim.
    ([IMM(0), EXIT], 0, "MARK_SE+0", 1.0),
    # 7) CONST always fires.
    ([IMM(0), EXIT], 1, "CONST+0", 1.0),
])
def test_state_at_step_end_programs_smoke(program, step, key, expected):
    state = state_at_step_end(program, step=step)
    assert state.get(key, 0.0) == expected, describe_state(state)
