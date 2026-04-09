"""Test PRTF with debug output to trace opcode setting."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("Hi");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)
print(f"Instructions: {len(code)}")

# Print all opcodes
for i, instr in enumerate(code):
    op = instr & 0xFF
    print(f"  {i}: opcode {op} (0x{op:02x})" + (" <-- PRTF" if op == 33 else ""))

runner = AutoregressiveVMRunner(conversational_io=True)

# Patch set_active_opcode to print when called
original_set = runner.model.set_active_opcode

def debug_set_active_opcode(opcode_value):
    if opcode_value == 33:
        print(f"\n>>> set_active_opcode(33) CALLED! <<<\n")
    original_set(opcode_value)

runner.model.set_active_opcode = debug_set_active_opcode

print("\nRunning (max 10 steps)...")
try:
    output, exit_code = runner.run(code, data, [], max_steps=10)
    print(f"\n✓ Execution complete")
    print(f"Exit code: {exit_code}")
    print(f"Output: {repr(output)}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
