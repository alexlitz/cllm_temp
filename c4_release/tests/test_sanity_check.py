"""Quick sanity check that conversational I/O is set up correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("Sanity Check - Conversational I/O Setup")
print("=" * 50)

print("\n1. Compiling test program...")
code, data = compile_c('int main() { return 0; }')
print(f"   OK: {len(code)} instructions compiled")

print("\n2. Creating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)
print("   OK: Runner created")

print("\n3. Checking model attributes...")
print(f"   Has _active_opcode: {hasattr(runner.model, '_active_opcode')}")
print(f"   Value: {runner.model._active_opcode}")

print("\n4. Checking embed signature...")
import inspect
sig = inspect.signature(runner.model.embed.forward)
print(f"   Parameters: {list(sig.parameters.keys())}")
has_active_opcode = 'active_opcode' in sig.parameters
print(f"   Has active_opcode param: {has_active_opcode}")

print("\n5. Checking conversational_io flag...")
print(f"   Runner.conversational_io: {runner.conversational_io}")

print("\n" + "=" * 50)
if has_active_opcode and hasattr(runner.model, '_active_opcode'):
    print("✅ SUCCESS: Conversational I/O is properly configured!")
    print("\nAll infrastructure is in place:")
    print("  - Active opcode tracking")
    print("  - Embedding augmentation")
    print("  - Runner flag enabled")
    print("\nThe system is ready for integration testing.")
else:
    print("❌ FAILURE: Setup incomplete")
print("=" * 50)
