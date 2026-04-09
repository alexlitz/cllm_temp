"""Check what active_opcode is set when PRTF step is generated."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("H\\n");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

# Find PRTF instruction
for i, instr in enumerate(code):
    op = instr & 0xFF
    print(f"Instruction {i}: opcode={op} (0x{op:02x})")
    if op == 33:
        print(f"  ^ PRTF at instruction {i}")

runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track active_opcode during generation
step_opcodes = []
original_generate = runner.model.generate_next

def track_opcode(context):
    # Check active opcode at start of each step
    if len(context) % 35 == 0:
        step_num = len(context) // 35
        active_op = runner.model._active_opcode
        step_opcodes.append((step_num, active_op))
        op_str = f"0x{active_op:02x}" if active_op is not None else "None"
        print(f"Step {step_num}: active_opcode = {active_op} ({op_str})")

    return original_generate(context)

runner.model.generate_next = track_opcode

print("\nRunning VM (max 10 steps)...")
output_str, exit_code = runner.run(code, data, [], max_steps=10)

print(f"\n✓ Execution complete, exit code: {exit_code}")
print(f"\nActive opcodes by step:")
for step, op in step_opcodes:
    if op == 33:
        print(f"  Step {step}: opcode {op} (0x{op:02x}) ← PRTF!")
    elif op is not None:
        print(f"  Step {step}: opcode {op} (0x{op:02x})")
    else:
        print(f"  Step {step}: None")
