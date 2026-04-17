#!/usr/bin/env python3
"""Very simple test with just 1 step."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

print("Setup...")
runner = AutoregressiveVMRunner()
set_vm_weights(runner.model)

source = "int main() { return 0; }"
bytecode, data = compile_c(source)

print(f"Running with max_steps=1...")
import time
start = time.time()
output, exit_code = runner.run(bytecode, data or b"", argv=[], max_steps=1)
elapsed = time.time() - start

print(f"Completed in {elapsed:.1f}s")
print(f"  Output: {output!r}")
print(f"  Exit code: {exit_code}")
