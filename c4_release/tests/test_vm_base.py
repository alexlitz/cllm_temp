"""Test base VM without conversational_io."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

c_code = '''
int main() {
    return 42;
}
'''

print("Compiling...")
code, data = compile_c(c_code)
print(f"✓ Compiled")

print("\nCreating runner WITHOUT conversational_io...")
runner = AutoregressiveVMRunner(conversational_io=False)

if torch.cuda.is_available():
    print("Moving to CUDA...")
    runner.model = runner.model.cuda()

print("✓ Runner created")

print("\nRunning VM (max 10 steps)...")
try:
    output_str, exit_code = runner.run(code, data, [], max_steps=10)
    print(f"\n✓ Execution complete")
    print(f"Exit code: {exit_code}")
    print(f"Output: {repr(output_str)}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
