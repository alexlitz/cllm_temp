"""Simple inspection of PRTF execution."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

c_code = '''
int main() {
    printf("Hi");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

print("\nInstructions (first 10):")
for i in range(min(10, len(code))):
    instr = code[i]
    op = instr & 0xFF
    print(f"  {i}: opcode={op:3d} (0x{op:02x})")

print("\nCreating runner...")
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Intercept generate_next to track opcode changes
opcodes_seen = []
original_set_opcode = runner.model.set_active_opcode

def track_opcode(opcode_value):
    opcodes_seen.append(opcode_value)
    original_set_opcode(opcode_value)

runner.model.set_active_opcode = track_opcode

print("\nRunning for 7 steps...")
try:
    output, exit_code = runner.run(code, data, [], max_steps=7)
    print(f"Exit code: {exit_code}")
    print(f"\nOpcodes seen: {len(opcodes_seen)}")
    for i, op in enumerate(opcodes_seen[:10]):
        if op is not None:
            print(f"  Step {i}: {op:3d} (0x{op:02x}) {'<-- PRTF!' if op == 33 else ''}")
        else:
            print(f"  Step {i}: None")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
