"""Test simple printf with literal string."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

# Test program: printf with single character
c_code = '''
int main() {
    printf("H\\n");
    return 0;
}
'''

print("Compiling test program...")
code, data = compile_c(c_code)
print(f"✓ Compiled: {len(code)} instructions, {len(data)} data bytes")
print(f"Data section: {bytes(data).hex()}")

print("\nCreating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)
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
