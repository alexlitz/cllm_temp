"""Test MUL behavior in pure_neural mode (no overflow needed)."""
import sys
sys.path.insert(0, '/tmp/p3-multibyte/c4_release')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
runner._func_call_handlers = {}
runner._syscall_handlers = {}


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def run_prog(prog, max_steps=30):
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    bc = _make_bc(prog)
    _, result = runner.run(bc, b"", max_steps=max_steps)
    return result


# MUL byte 0 only (no overflow)
for (a, b) in [(2, 3), (5, 5), (10, 10), (15, 17), (30, 10), (100, 100), (255, 255)]:
    r = run_prog([
        (Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.MUL, Opcode.EXIT,
    ])
    expect = a * b
    ok = "OK" if r == expect else f"WRONG (expect {expect})"
    print(f"{a} * {b} = {r} {ok}")
