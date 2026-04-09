"""Trace execution of manual bytecode."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

bytecode = [
    1 | (0x10000 << 8),  # IMM 0x10000
    15 | (0 << 8),       # PSH
    33 | (0 << 8),       # PRTF
    34 | (0 << 8),       # EXIT
]

data = b"Hello from Neural VM!\x00"

print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    print(f"  {i}: op={op:3d} imm={imm}")

runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track opcodes being set
opcodes_set = []
original_set = runner.model.set_active_opcode
def track_opcode(op):
    opcodes_set.append(op)
    print(f"  Step {len(opcodes_set)-1}: set_active_opcode({op})")
    original_set(op)
runner.model.set_active_opcode = track_opcode

print("\nRunning (max 5 steps)...")
output, exit_code = runner.run(bytecode, data, [], max_steps=5)

print(f"\nResults:")
print(f"  Steps executed: {len(opcodes_set)}")
print(f"  Opcodes set: {opcodes_set}")
print(f"  Exit code: {exit_code}")

print(f"\nOpcode names:")
for i, op in enumerate(opcodes_set):
    op_names = {1: "IMM", 15: "PSH", 33: "PRTF", 34: "EXIT"}
    print(f"  Step {i}: {op_names.get(op, f'op_{op}')}")

if 33 in opcodes_set:
    print(f"\n✅ PRTF was executed at step {opcodes_set.index(33)}")
else:
    print(f"\n❌ PRTF was never executed")
    print(f"   Program got stuck before reaching instruction 2 (PRTF)")
