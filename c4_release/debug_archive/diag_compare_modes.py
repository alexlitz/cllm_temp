"""Compare trust_neural_alu vs pure_neural on same programs."""
import sys
sys.path.insert(0, '/tmp/p3-multibyte/c4_release')

from neural_vm.run_vm import AutoregressiveVMRunner
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


print("=== trust_neural_alu mode ===")
runner = AutoregressiveVMRunner(trust_neural_alu=True)
runner._func_call_handlers = {}
runner._syscall_handlers = {}
for (a, b) in [(5, 3), (200, 100), (10, 32)]:
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    bc = _make_bc([
        (Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.ADD, Opcode.EXIT,
    ])
    _, r = runner.run(bc, b"", max_steps=30)
    print(f"{a}+{b} = {r} (expect {a+b}) {'OK' if r == a+b else 'WRONG'}")
del runner
import gc; gc.collect()

print("\n=== pure_neural mode ===")
runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
runner._func_call_handlers = {}
runner._syscall_handlers = {}
for (a, b) in [(5, 3), (200, 100), (10, 32)]:
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    bc = _make_bc([
        (Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.ADD, Opcode.EXIT,
    ])
    _, r = runner.run(bc, b"", max_steps=30)
    print(f"{a}+{b} = {r} (expect {a+b}) {'OK' if r == a+b else 'WRONG'}")
