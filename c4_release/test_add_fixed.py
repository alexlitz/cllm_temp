"""
Test if ADD operation works without handler after fixing Layer 6 head conflict.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode

print("="*80)
print("TESTING ADD AFTER FIX")
print("="*80 + "\n")

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print(f"Test program: {code}")
print(f"Expected result: 42\n")

# Test with handler (baseline)
print("1. WITH HANDLER (baseline):")
runner_with = AutoregressiveVMRunner()
output_with, exit_with = runner_with.run(bytecode, max_steps=10)
print(f"   Output: {output_with}, Exit code: {exit_with}")

# Test without handler (neural only)
print("\n2. WITHOUT HANDLER (neural implementation):")
runner_without = AutoregressiveVMRunner()
if Opcode.ADD in runner_without._func_call_handlers:
    del runner_without._func_call_handlers[Opcode.ADD]
    print("   Removed ADD handler")

output_without, exit_without = runner_without.run(bytecode, max_steps=10)
print(f"   Output: {output_without}, Exit code: {exit_without}")

# Compare
print("\n" + "="*80)
if output_with == output_without == 42 and exit_with == exit_without == 0:
    print("✓ SUCCESS! Neural ADD works correctly!")
    print("  Both implementations return 42")
elif output_with == 42 and output_without != 42:
    print("❌ FAIL: Neural implementation returns wrong value")
    print(f"  Expected: {output_with}, Got: {output_without}")
else:
    print("⚠️  Unexpected result")
    print(f"  Handler: {output_with}, Neural: {output_without}")

print("="*80)

import torch
torch.cuda.empty_cache()
