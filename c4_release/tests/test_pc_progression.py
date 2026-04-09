"""Test PC progression with conversational_io."""

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

# Test 1: Base VM (no conversational_io)
print("\n" + "="*60)
print("TEST 1: Base VM (conversational_io=False)")
print("="*60)
runner1 = AutoregressiveVMRunner(conversational_io=False)
if torch.cuda.is_available():
    runner1.model = runner1.model.cuda()

print("Running 10 steps...")
try:
    # Track PC values
    pc_history = []

    # Monkey-patch to track PC
    original_extract = runner1._extract_register
    def track_pc(context, marker):
        val = original_extract(context, marker)
        if marker == Token.REG_PC and val is not None:
            pc_history.append(val)
        return val
    runner1._extract_register = track_pc

    output_str, exit_code = runner1.run(code, data, [], max_steps=10)
    print(f"✓ Exit code: {exit_code}")
    print(f"PC history: {[f'0x{pc:04x}' for pc in pc_history[:10]]}")
    print(f"Reached instruction indices: {[pc//6 for pc in pc_history[:10]]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: With conversational_io
print("\n" + "="*60)
print("TEST 2: With conversational_io=True")
print("="*60)
runner2 = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner2.model = runner2.model.cuda()

print("Running 10 steps...")
try:
    pc_history2 = []

    original_extract2 = runner2._extract_register
    def track_pc2(context, marker):
        val = original_extract2(context, marker)
        if marker == Token.REG_PC and val is not None:
            pc_history2.append(val)
        return val
    runner2._extract_register = track_pc2

    output_str, exit_code = runner2.run(code, data, [], max_steps=10)
    print(f"✓ Exit code: {exit_code}")
    print(f"PC history: {[f'0x{pc:04x}' for pc in pc_history2[:10]]}")
    print(f"Reached instruction indices: {[pc//6 for pc in pc_history2[:10]]}")

    # Check if PC is stuck
    if len(set(pc_history2[:5])) == 1:
        print(f"⚠️ PC STUCK at {pc_history2[0]:04x} (instruction {pc_history2[0]//6})")
    else:
        print(f"✓ PC is progressing normally")

except Exception as e:
    print(f"✗ Error: {e}")
