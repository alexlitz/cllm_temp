"""MUL/DIV/MOD as bytecode SUBROUTINES on the lean shallow model.

CPU tests validate the subroutine bytecode against a byte-exact memory-exposing
8-bit reference interpreter (``_interp_mem``, mirroring ``isa.interpret``).  A
GPU-guarded test proves the 8-bit subroutine runs BYTE-EXACT through the actual
lean neural forward on the ``SUBSET_BITWISE`` (~15-layer, NO deep muldiv) model.
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min import lean_subroutine_muldiv as M


# ---------------------------------------------------------------------------
# A byte-exact memory-exposing 8-bit reference (mirrors isa.interpret + memory).
# ---------------------------------------------------------------------------
def _interp_mem(code, max_steps: int = 4_000_000):
    ax = 0
    sp = 256
    pc = 0
    mem = [0] * 256
    stack = [0] * 257

    def push(v):
        nonlocal sp
        sp -= 1
        stack[sp] = v & 0xFF

    def pop():
        nonlocal sp
        v = stack[sp]
        sp += 1
        return v & 0xFF

    steps = 0
    while pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.PSH:
            push(ax)
        elif op == isa.ADD:
            ax = (pop() + ax) & 0xFF
        elif op == isa.SUB:
            ax = (pop() - ax) & 0xFF
        elif op == isa.AND:
            ax = pop() & ax
        elif op == isa.OR:
            ax = pop() | ax
        elif op == isa.XOR:
            ax = pop() ^ ax
        elif op == isa.SHL:
            ax = (pop() << ax) & 0xFF
        elif op == isa.SHR:
            ax = (pop() >> ax) & 0xFF
        elif op == isa.LI:
            ax = mem[ax] & 0xFF
        elif op == isa.SI:
            mem[pop()] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
    return ax, mem, steps


def _read_mb(mem, cells):
    return sum((mem[c] & 0xFF) << (8 * i) for i, c in enumerate(cells))


# ---------------------------------------------------------------------------
# 8-bit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [(3, 4), (12, 11), (200, 3), (255, 255),
                                 (0, 7), (7, 0), (13, 10), (1, 255)])
def test_mul8_byte_exact(a, b):
    ax, _, _ = _interp_mem(M.program_mul8(a, b))
    assert ax == (a * b) & 0xFF


@pytest.mark.parametrize("a,b", [(200, 17), (255, 16), (100, 7), (0, 5),
                                 (5, 0), (255, 255), (13, 13)])
def test_div8_byte_exact(a, b):
    ax, _, _ = _interp_mem(M.program_div8(a, b))
    assert ax == ((a // b) if b else 0) & 0xFF


@pytest.mark.parametrize("a,b", [(200, 17), (255, 16), (100, 7), (0, 5),
                                 (5, 0), (255, 255), (13, 13)])
def test_mod8_byte_exact(a, b):
    ax, _, _ = _interp_mem(M.program_mod8(a, b))
    assert ax == ((a % b) if b else 0) & 0xFF


# ---------------------------------------------------------------------------
# 16- and 32-bit (result read from memory result cells)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nbytes,mask", [(2, 0xFFFF), (4, 0xFFFFFFFF)])
@pytest.mark.parametrize("a,b", [(3, 4), (300, 7), (1000, 1000), (65535, 1),
                                 (123456 % (1 << 32), 789)])
def test_mul_mb_byte_exact(nbytes, mask, a, b):
    a &= mask
    b &= mask
    _, mem, _ = _interp_mem(M.program_mul_mb(a, b, nbytes))
    assert _read_mb(mem, M.M_BYTES[:nbytes]) == (a * b) & mask


@pytest.mark.parametrize("nbytes,mask", [(2, 0xFFFF), (4, 0xFFFFFFFF)])
@pytest.mark.parametrize("a,b", [(300, 7), (1000, 1000), (65535, 1),
                                 (0, 999), (50000 % (1 << 32), 3)])
def test_div_mb_byte_exact(nbytes, mask, a, b):
    a &= mask
    b &= mask
    _, mem, _ = _interp_mem(M.program_div_mb(a, b, nbytes))
    assert _read_mb(mem, M.Q_BYTES[:nbytes]) == ((a // b) if b else 0)
    _, mem, _ = _interp_mem(M.program_mod_mb(a, b, nbytes))
    assert _read_mb(mem, M.M_BYTES[:nbytes]) == ((a % b) if b else 0)


def test_random_sweep_16bit():
    import random
    random.seed(7)
    for _ in range(50):
        a = random.randint(0, 65535)
        b = random.randint(0, 65535)
        _, mem, _ = _interp_mem(M.program_mul_mb(a, b, 2))
        assert _read_mb(mem, M.M_BYTES[:2]) == (a * b) & 0xFFFF
        _, mem, _ = _interp_mem(M.program_div_mb(a, b, 2))
        assert _read_mb(mem, M.Q_BYTES[:2]) == ((a // b) if b else 0)
        _, mem, _ = _interp_mem(M.program_mod_mb(a, b, 2))
        assert _read_mb(mem, M.M_BYTES[:2]) == ((a % b) if b else 0)


def test_step_count_scales_with_precision():
    """Lower precision -> far fewer VM steps (the whole point of the variants)."""
    s8 = len(isa.interpret(M.program_mul8(0xFF, 0xFF), max_steps=4_000_000))
    s16 = len(isa.interpret(M.program_mul_mb(0xFFFF, 0xFFFF, 2), max_steps=4_000_000))
    s32 = len(isa.interpret(M.program_mul_mb(0xFFFFFFFF, 0xFFFFFFFF, 4), max_steps=4_000_000))
    assert s8 < s16 < s32


# ---------------------------------------------------------------------------
# GPU: byte-exact through the ACTUAL lean neural forward (SUBSET_BITWISE).
# ---------------------------------------------------------------------------
def _cuda1_or_skip():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs cuda:1")
    return "cuda:1"


@pytest.fixture(scope="module")
def lean_bitwise():
    dev = _cuda1_or_skip()
    os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_agent")
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    vm = Q.build(code_size=64, subset=Q.SUBSET_BITWISE)
    return LF.LeanQwenVM.from_full_vm(vm, device=dev)


@pytest.mark.parametrize("a,b", [(12, 11), (200, 3), (13, 10)])
def test_mul8_through_neural_forward(lean_bitwise, a, b):
    from c4_min import qwen_lean_forward as LF
    r = LF.run_program_lean(lean_bitwise, M.program_mul8(a, b), max_steps=5000)
    assert r["ax_trace"][-1] == (a * b) & 0xFF


@pytest.mark.parametrize("a,b", [(200, 17), (100, 7)])
def test_div8_through_neural_forward(lean_bitwise, a, b):
    from c4_min import qwen_lean_forward as LF
    r = LF.run_program_lean(lean_bitwise, M.program_div8(a, b), max_steps=5000)
    assert r["ax_trace"][-1] == (a // b) & 0xFF
