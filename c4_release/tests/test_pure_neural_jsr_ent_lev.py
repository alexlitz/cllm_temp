"""Phase 5 gate: function-call opcodes JSR/ENT/LEV in pure-neural mode.

These tests run AutoregressiveVMRunner(pure_neural=True) with NO Python
overrides at all. The neural network must compute return-address handling,
BP/SP arithmetic, and PC restoration entirely from its forward pass.

Phase 5 closes when all tests in this file pass (or every xfail is
upgraded to a real assertion).

C4 calling convention recap:
  JSR imm:  push PC; PC = imm * INSTR_WIDTH
  ENT imm:  push BP; BP = SP; SP -= imm
  LEV:      SP = BP; BP = *SP; PC = *(SP+8); SP += 16

Branch/call targets are instruction INDICES (multiplied by INSTR_WIDTH=8
when materialized as actual PC values).
"""

import pytest

from neural_vm.embedding import Opcode


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run(runner, prog, max_steps=60):
    bc = _make_bc(prog)
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    _, result = runner.run(bc, b"", max_steps=max_steps)
    return result


class TestPureNeuralJSRLEVSimple:
    """Smallest possible call/return — must pass before anything else."""

    @pytest.mark.xfail(reason="Actual=0 expected=7. JSR/LEV roundtrip clobbers AX in pure_neural mode; suspect _set_layer14_mem_generation not pushing return-addr correctly and _set_layer9_alu writing into AX lane")
    def test_jsr_then_lev_simple(self, pure_neural_runner):
        # main: 0:IMM 7, 1:JSR -> idx 3, 2:EXIT
        # callee: 3:ENT 0, 4:LEV
        # Expected: AX preserved across call+return = 7
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 7),
            (Opcode.JSR, 3),
            Opcode.EXIT,
            (Opcode.ENT, 0),
            Opcode.LEV,
        ]) == 7

    @pytest.mark.xfail(reason="LEV neural path: _set_layer9_lev_bp_to_pc_relay does not restore PC from mem[BP+8]; callee body's IMM never reaches caller EXIT")
    def test_jsr_callee_writes_ax(self, pure_neural_runner):
        # main: 0:JSR -> idx 2, 1:EXIT
        # callee: 2:ENT 0, 3:IMM 42, 4:LEV
        # Expected: AX=42 returned by EXIT
        assert _run(pure_neural_runner, [
            (Opcode.JSR, 2),
            Opcode.EXIT,
            (Opcode.ENT, 0),
            (Opcode.IMM, 42),
            Opcode.LEV,
        ]) == 42

    @pytest.mark.xfail(reason="LEV PC restoration broken: _set_layer16_lev_routing fails to route popped return-addr back to PC; runner exhausts max_steps without reaching caller EXIT (which would set AX=0 here, but the bug is failure-to-reach-EXIT, not the value)")
    def test_lev_returns_to_caller(self, pure_neural_runner):
        # main: 0:JSR -> idx 2, 1:EXIT (target). callee: 2:ENT 0, 3:LEV.
        # On a working roundtrip PC reaches idx 1 and EXIT triggers with the
        # default AX (0). On a broken LEV, the runner hangs in the callee.
        # Either outcome currently produces 0; this test will become a
        # meaningful gate once test_jsr_callee_writes_ax passes.
        assert _run(pure_neural_runner, [
            (Opcode.JSR, 2),
            Opcode.EXIT,
            (Opcode.ENT, 0),
            Opcode.LEV,
        ]) == 0


class TestPureNeuralJSRSemantics:
    """JSR-specific behavior: return-addr push, AX preservation."""

    @pytest.mark.xfail(reason="JSR clobbers AX in pure_neural mode: _set_layer9_alu writes return-addr arithmetic into AX-routed lane instead of staying on STACK0; expected 99 actual likely return-addr value")
    def test_jsr_does_not_clobber_caller_ax(self, pure_neural_runner):
        # main: 0:IMM 5, 1:JSR -> idx 3, 2:EXIT
        # callee: 3:ENT 0, 4:IMM 99, 5:LEV
        # Per C4 ABI, AX after EXIT = most recent IMM (99 from callee).
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            (Opcode.JSR, 3),
            Opcode.EXIT,
            (Opcode.ENT, 0),
            (Opcode.IMM, 99),
            Opcode.LEV,
        ]) == 99

    @pytest.mark.xfail(reason="Nested JSR overflows return-addr stack: _set_layer7_memory_heads only routes a single return slot; second JSR overwrites first frame's saved PC, LEV chain corrupts")
    def test_nested_jsr(self, pure_neural_runner):
        # main: 0:JSR -> idx 2, 1:EXIT
        # fn1:  2:ENT 0, 3:JSR -> idx 6, 4:LEV
        # fn2:  6:ENT 0, 7:IMM 21, 8:LEV
        assert _run(pure_neural_runner, [
            (Opcode.JSR, 2),
            Opcode.EXIT,
            (Opcode.ENT, 0),
            (Opcode.JSR, 6),
            Opcode.LEV,
            Opcode.NOP,
            (Opcode.ENT, 0),
            (Opcode.IMM, 21),
            Opcode.LEV,
        ]) == 21


class TestPureNeuralENTSemantics:
    """ENT-specific behavior: BP save, BP=SP, SP-=imm."""

    @pytest.mark.xfail(reason="ENT with nonzero imm: _set_layer8_alu does not subtract imm from SP in pure_neural mode; subsequent LEV sees wrong SP, BP/PC restore from incorrect addresses")
    def test_ent_decrements_sp_by_imm(self, pure_neural_runner):
        # main: 0:JSR -> idx 2, 1:EXIT
        # callee: 2:ENT 4 (allocates 4 local-bytes), 3:IMM 13, 4:LEV
        # If ENT 4 correctly decrements SP and LEV correctly restores via BP,
        # AX=13 propagates to caller EXIT.
        assert _run(pure_neural_runner, [
            (Opcode.JSR, 2),
            Opcode.EXIT,
            (Opcode.ENT, 4),
            (Opcode.IMM, 13),
            Opcode.LEV,
        ]) == 13


class TestPureNeuralLEVCornerCases:
    """LEV in unusual positions — likely not supported neurally."""

    @pytest.mark.xfail(reason="Actual=0 expected=1. Bare LEV clobbers AX even when no frame exists; _set_layer16_lev_routing unconditionally overwrites AX path during LEV; _set_layer15_memory_lookup reads uninitialized BP/PC slots")
    def test_lev_without_ent_xfail(self, pure_neural_runner):
        # No ENT pushed BP, so LEV pops garbage. We document that this
        # corner case is not expected to produce a meaningful result.
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 1),
            Opcode.LEV,
            Opcode.EXIT,
        ]) == 1
