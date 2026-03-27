#!/usr/bin/env python3
"""Debug the runner to see what tokens are generated."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

# Compile simple program
code = "int main() { return 42; }"
bytecode, data = compile_c(code)

print(f"Bytecode: {bytecode}")
print(f"Instructions:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    print(f"  {i}: op={op}, imm={imm}")

# Create runner
runner = AutoregressiveVMRunner()
set_vm_weights(runner.model)
runner.model.compact(block_size=32)
runner.model.compact_moe()

# Run with debugging
print("\nRunning...")
output, exit_code = runner.run(bytecode, data, max_steps=100)

print(f"\nOutput: '{output}'")
print(f"Exit code: {exit_code}")
print(f"Expected exit code: 42")

# Check internal state
print(f"\n_last_ax: {runner._last_ax}")
print(f"_last_pc: {runner._last_pc}")
print(f"_last_sp: {hex(runner._last_sp) if runner._last_sp else None}")
