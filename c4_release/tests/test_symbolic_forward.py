"""Tests for the symbolic forward executor.

Unit tests verify:

1. Encoding/decoding round-trips and the default embedding mapper
   sets the right indicator dims.
2. A minimal 2-block layout with a single FFN rule produces the
   expected residual snapshot after one step.
3. ``get_dim_trace`` returns one entry per (step, block) pair where
   the dim has been written.
4. ``diff_against_expected`` returns ``None`` on a match and the first
   divergence on a mismatch.

These do not depend on ``compile_full_vm_dynamic`` — they build a
lightweight ``Operation`` + ``ModelLayout`` shim so the test runs
sub-second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

import pytest

from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    step_function_rule,
)
from c4_release.neural_vm.unified_compiler.ir import CompilerIR
from c4_release.neural_vm.unified_compiler.symbolic_forward import (
    OP_EXIT,
    OP_IMM,
    OP_PSH,
    BlockSnapshot,
    SymbolicForwardRunner,
    decode_instr,
    default_embedding_for_instruction,
    encode_instr,
)


# ---------------------------------------------------------------------------
# Op + compiler shims (no LayerCompiler.compile() needed)
# ---------------------------------------------------------------------------


@dataclass
class _StubOp:
    name: str
    kind: str = "ffn"
    compiler_ir: Optional[Any] = None
    layer_idx: Optional[int] = None
    target_op_name: Optional[str] = None
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)


@dataclass
class _StubCompiler:
    ops_per_layer: List[List[_StubOp]]
    block_ops: List[_StubOp] = field(default_factory=list)
    model_ops: List[_StubOp] = field(default_factory=list)
    dim_positions: Optional[dict] = None


def _ir_with_rule(rule) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)
    return ir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip():
    word = encode_instr(OP_IMM, 0x200)
    op, imm = decode_instr(word)
    assert op == OP_IMM
    assert imm == 0x200

    # PSH and EXIT carry no immediate.
    assert decode_instr(encode_instr(OP_PSH)) == (OP_PSH, 0)
    assert decode_instr(encode_instr(OP_EXIT)) == (OP_EXIT, 0)


def test_default_embedding_sets_op_and_imm_indicators():
    """``IMM 0x200`` -> ``OP_IMM`` indicator + IMM_AX one-hot bytes."""
    state = default_embedding_for_instruction(OP_IMM, 0x200, pc=0)
    assert state["OP_IMM+0"] == 1.0
    assert state["CONST+0"] == 1.0
    # 0x200 byte 0 = 0x00; byte 1 = 0x02; byte 2 = 0x00.
    # Byte 0: lo=0 hi=0; byte 1: lo=2 hi=0; byte 2: lo=0 hi=0.
    assert state["IMM_AX_LO+0"] == 1.0
    assert state["IMM_AX_HI+0"] == 1.0
    assert state["IMM_AX_LO_1+2"] == 1.0
    assert state["IMM_AX_HI_1+0"] == 1.0

    # PSH at pc=1 -> OP_PSH indicator, all IMM bytes are zero.
    state_psh = default_embedding_for_instruction(OP_PSH, 0, pc=1)
    assert state_psh["OP_PSH+0"] == 1.0
    assert state_psh["IMM_AX_LO+0"] == 1.0
    assert state_psh["PC+0"] == 1.0


def test_single_block_runner_records_snapshot():
    """One FFN rule fires when its input dim is set in the embedding."""
    # Rule: when OP_IMM is set, write 1.0 to STACK0_BYTE_VAL_1_LO+2 (the
    # byte-1 low-nibble cell that 0x200 lights up).
    rule = step_function_rule(
        input_dim="OP_IMM",
        threshold=0.5,
        write_dim="STACK0_BYTE_VAL_1_LO+2",
        write_value=1.0,
    )
    op = _StubOp(name="demo_op", kind="ffn", compiler_ir=_ir_with_rule(rule))
    compiler = _StubCompiler(ops_per_layer=[[op]])

    program = [encode_instr(OP_IMM, 0x200)]
    runner = SymbolicForwardRunner(compiler, program)
    snaps = runner.step()

    assert len(snaps) == 1
    snap = snaps[0]
    assert isinstance(snap, BlockSnapshot)
    assert snap.step_idx == 0
    assert snap.block_idx == 0
    # Rule fired -> 1.0 / S(=100) = 0.01 contribution.
    assert pytest.approx(
        runner.get_residual(0, 0, "STACK0_BYTE_VAL_1_LO+2"), abs=1e-9,
    ) == 0.01


def test_get_dim_trace_records_per_block_per_step():
    """Multi-block + multi-step program produces one trace entry per
    (block, step) snapshot that holds the dim."""
    rule = step_function_rule(
        input_dim="OP_IMM",
        threshold=0.5,
        write_dim="STACK0_BYTE_VAL_1_LO+2",
        write_value=1.0,
    )
    block_a = _StubOp(name="op_block0", compiler_ir=_ir_with_rule(rule))
    block_b = _StubOp(name="op_block1", compiler_ir=_ir_with_rule(rule))
    compiler = _StubCompiler(ops_per_layer=[[block_a], [block_b]])

    program = [
        encode_instr(OP_IMM, 0x200),
        encode_instr(OP_PSH),
        encode_instr(OP_EXIT),
    ]
    runner = SymbolicForwardRunner(compiler, program)
    runner.run_all()

    trace = runner.get_dim_trace("STACK0_BYTE_VAL_1_LO+2")
    # Rule fires only on the IMM step (rule conditions on OP_IMM).
    # 2 blocks * 1 firing step = 2 entries. Each subsequent block
    # accumulates on top of the previous (state carries within a step).
    imm_entries = [e for e in trace if e.step_idx == 0]
    assert len(imm_entries) == 2
    # First block: 0.01; second block: 0.02 (accumulated).
    assert pytest.approx(imm_entries[0].value, abs=1e-9) == 0.01
    assert pytest.approx(imm_entries[1].value, abs=1e-9) == 0.02
    # PSH/EXIT steps did not fire the rule -> the dim is not present.
    assert all(e.step_idx == 0 for e in trace), (
        f"trace should be IMM-only; got {trace}"
    )


def test_diff_against_expected_finds_first_divergence():
    """Mismatched expected value -> first divergence triple; match -> None."""
    rule = step_function_rule(
        input_dim="OP_IMM",
        threshold=0.5,
        write_dim="STACK0_BYTE_VAL_1_LO+2",
        write_value=1.0,
    )
    op = _StubOp(name="demo_op", compiler_ir=_ir_with_rule(rule))
    compiler = _StubCompiler(ops_per_layer=[[op]])

    runner = SymbolicForwardRunner(
        compiler, [encode_instr(OP_IMM, 0x200)],
    )
    runner.step()

    # Matching expected trace -> None.
    matching = {(0, 0, 0, "STACK0_BYTE_VAL_1_LO+2"): 0.01}
    assert runner.diff_against_expected(matching) is None

    # Mismatched expected trace -> first divergence reported.
    mismatched = {(0, 0, 0, "STACK0_BYTE_VAL_1_LO+2"): 0.99}
    div = runner.diff_against_expected(mismatched)
    assert div is not None
    key, actual, expected = div
    assert key == (0, 0, 0, "STACK0_BYTE_VAL_1_LO+2")
    assert pytest.approx(actual, abs=1e-9) == 0.01
    assert expected == 0.99


def test_step_past_end_raises():
    """Stepping past the program end raises IndexError."""
    op = _StubOp(name="noop", compiler_ir=CompilerIR())
    compiler = _StubCompiler(ops_per_layer=[[op]])
    runner = SymbolicForwardRunner(
        compiler, [encode_instr(OP_EXIT)],
    )
    runner.step()
    with pytest.raises(IndexError):
        runner.step()
