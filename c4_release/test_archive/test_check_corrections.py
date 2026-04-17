#!/usr/bin/env python3
"""Check if shadow state corrections are being applied."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

# Compile simple program
code = "int main() { return 42; }"
bytecode, data = compile_c(code)

# Patch runner to see if corrections are applied
class DebugRunner(AutoregressiveVMRunner):
    def _override_register_in_last_step(self, context, marker_token, value):
        print(f"  CORRECTION: Setting {marker_token} to 0x{value:08x} ({value})")
        return super()._override_register_in_last_step(context, marker_token, value)

# Create runner
print("Creating runner with pure_attention_memory=False (default)")
runner = DebugRunner(pure_attention_memory=False)
set_vm_weights(runner.model)
runner.model.compact(block_size=32)
runner.model.compact_moe()

print(f"pure_attention_memory: {runner.pure_attention_memory}\n")

print("Running...\n")
output, exit_code = runner.run(bytecode, data, max_steps=10)

print(f"\nOutput: '{output}'")
print(f"Exit code: {exit_code}")
print(f"Expected: 42")
