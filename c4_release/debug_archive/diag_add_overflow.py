"""Diagnostic: trace ADD overflow 200+100=300 in pure_neural mode.

Run the same program as test_add_overflow_300:
  IMM 200; PSH; IMM 100; ADD; EXIT
Expect result == 300 = 0x12C  → byte0=44, byte1=1.
Currently returns 124 = 0x7C.

Print AX bytes (low/high nibble per byte) after each step.
"""
import sys
sys.path.insert(0, '/tmp/p3-multibyte/c4_release')
sys.path.insert(0, '/tmp/p3-multibyte')

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
runner._func_call_handlers = {}
runner._syscall_handlers = {}
runner._memory = {}
runner._mem_history = {}
runner._mem_access_order = []

prog = [
    (Opcode.IMM, 200),
    Opcode.PSH,
    (Opcode.IMM, 100),
    Opcode.ADD,
    Opcode.EXIT,
]


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


bc = _make_bc(prog)
print(f"Bytecode: {bc}")
print(f"  IMM=1, PSH=13, ADD=25, EXIT=38")

_, result = runner.run(bc, b"", max_steps=30)
print(f"\nResult: {result}")
print(f"Expected: 300 = 0x12C (byte0=44=0x2C, byte1=1, byte2=0, byte3=0)")
print(f"Actual hex: 0x{result & 0xFFFFFFFF:08X}")
print(f"  byte0={result & 0xFF} ({(result & 0xFF):#x})")
print(f"  byte1={(result >> 8) & 0xFF}")
print(f"  byte2={(result >> 16) & 0xFF}")
print(f"  byte3={(result >> 24) & 0xFF}")
