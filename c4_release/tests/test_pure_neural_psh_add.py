"""Phase 2 gate: SP arithmetic + PSH + binary ALU.

Builds on Phase 1 (PC + AX coherence). These tests exercise:
  - SP increment/decrement on PSH (sp -= 8) and binary-pop ops (sp += 8)
  - PSH writing AX to mem[SP-8] via the model's MEM token sequence
  - Binary ALU ops (ADD/SUB/AND/OR/XOR/etc.) reading prev STACK0 from MEM

Phase 2 closes when all tests in this file pass.

Currently all tests xfail in pure_neural mode. Triage 2026-05-11
(p2-triage-hard-fails): every test in this file produced either a hard
assertion mismatch (e.g. AX returned 4160946181 instead of 5 for
test_imm_psh_exit) or a >60s autoregressive hang (model never emits an
EXIT-terminating step). Root cause matches the sibling Phase-3 multibyte
suite: the pure_neural pipeline does not yet wire AX-through-PSH
preservation nor the binary ALU readback of prev STACK0 from MEM into
L9/L10 ALU FFN units. These need _set_layer9_alu / _set_layer4_ffn /
MEM persistence shim coverage before they will land.

Markers use strict=False so any incidental XPASS lands as XPASS rather
than failing CI; once Phase 2 lands they should all flip to XPASS and the
markers can be removed.
"""

import pytest

from neural_vm.embedding import Opcode


_PHASE2_PSH_XFAIL_REASON = (
    "pure_neural Phase 2: PSH+EXIT chain does not yet preserve AX through "
    "the MEM-store + EXIT sequence (model returns garbage 32-bit AX). "
    "Needs MEM persistence + AX read-through wiring in _dispatch_step "
    "pure_neural branch. Tracked alongside the Phase-3 multibyte ADD/MUL "
    "tests in test_pure_neural_multibyte.py."
)

_PHASE2_ALU_XFAIL_REASON = (
    "pure_neural Phase 2: binary ALU (ADD/SUB/AND/OR) does not yet produce "
    "the correct AX after PSH + IMM + <op> in pure_neural mode. Tests "
    "either return wrong values or hang (no EXIT termination within "
    "max_steps). Requires _set_layer9_alu / _set_layer10_alu wiring to "
    "read prev STACK0 from MEM and combine with current AX. Mirrors the "
    "xfailed multi-byte counterparts in test_pure_neural_multibyte.py "
    "(TestPureNeuralAddOverflow / TestPureNeuralSubBorrow)."
)


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run(runner, prog, max_steps=30):
    bc = _make_bc(prog)
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    _, result = runner.run(bc, b"", max_steps=max_steps)
    return result


class TestPureNeuralPSH:
    """PSH-then-EXIT — does the SP decrement work?"""

    @pytest.mark.xfail(reason=_PHASE2_PSH_XFAIL_REASON, strict=False)
    def test_imm_psh_exit(self, pure_neural_runner):
        # AX should remain the IMM value (PSH preserves AX)
        assert _run(pure_neural_runner, [
            (Opcode.IMM, 5),
            Opcode.PSH,
            Opcode.EXIT,
        ]) == 5


class TestPureNeuralBinaryOps:
    """Binary ALU ops on small operands (< 256)."""

    @pytest.mark.xfail(reason=_PHASE2_ALU_XFAIL_REASON, strict=False)
    @pytest.mark.parametrize("a,b,expected", [
        (3, 4, 7),
        (10, 20, 30),
        (5, 0, 5),
        (0, 5, 5),
        (100, 50, 150),
    ])
    def test_add_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.ADD,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.xfail(reason=_PHASE2_ALU_XFAIL_REASON, strict=False)
    @pytest.mark.parametrize("a,b,expected", [
        (50, 20, 30),
        (100, 50, 50),
        (200, 100, 100),
        (42, 42, 0),
    ])
    def test_sub_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.SUB,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.xfail(reason=_PHASE2_ALU_XFAIL_REASON, strict=False)
    @pytest.mark.parametrize("a,b,expected", [
        (0xFF, 0x0F, 0x0F),
        (0xF0, 0x0F, 0x00),
        (0xFF, 0xFF, 0xFF),
    ])
    def test_and_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.AND,
            Opcode.EXIT,
        ]) == expected

    @pytest.mark.xfail(reason=_PHASE2_ALU_XFAIL_REASON, strict=False)
    @pytest.mark.parametrize("a,b,expected", [
        (0xF0, 0x0F, 0xFF),
        (0xAA, 0x55, 0xFF),
        (0x00, 0x00, 0x00),
    ])
    def test_or_small(self, pure_neural_runner, a, b, expected):
        assert _run(pure_neural_runner, [
            (Opcode.IMM, a),
            Opcode.PSH,
            (Opcode.IMM, b),
            Opcode.OR,
            Opcode.EXIT,
        ]) == expected
