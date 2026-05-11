"""Test ADD with no overflow: 100 + 50 = 150"""
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


# Test 1: 100 + 50 = 150 (no overflow)
r = run_prog([
    (Opcode.IMM, 100),
    Opcode.PSH,
    (Opcode.IMM, 50),
    Opcode.ADD,
    Opcode.EXIT,
])
print(f"100 + 50 = {r} (expect 150, hex 0x{r:08X})")

# Test 2: 200 + 100 = 300
r = run_prog([
    (Opcode.IMM, 200),
    Opcode.PSH,
    (Opcode.IMM, 100),
    Opcode.ADD,
    Opcode.EXIT,
])
print(f"200 + 100 = {r} (expect 300, hex 0x{r:08X})")

# Test 3: 200 + 56 = 256 (clean byte1 carry)
r = run_prog([
    (Opcode.IMM, 200),
    Opcode.PSH,
    (Opcode.IMM, 56),
    Opcode.ADD,
    Opcode.EXIT,
])
print(f"200 + 56 = {r} (expect 256, hex 0x{r:08X})")

# Test 4: 255 + 1 = 256
r = run_prog([
    (Opcode.IMM, 255),
    Opcode.PSH,
    (Opcode.IMM, 1),
    Opcode.ADD,
    Opcode.EXIT,
])
print(f"255 + 1 = {r} (expect 256, hex 0x{r:08X})")

# Test 5: 10 + 32 (A2 bug)
r = run_prog([
    (Opcode.IMM, 10),
    Opcode.PSH,
    (Opcode.IMM, 32),
    Opcode.ADD,
    Opcode.EXIT,
])
print(f"10 + 32 = {r} (expect 42, hex 0x{r:08X})")
