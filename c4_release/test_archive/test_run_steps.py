#!/usr/bin/env python3
"""Test run with different step counts."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

code = 'int main() { return 42; }'
bytecode, data = compile_c(code)
runner = AutoregressiveVMRunner()

for max_steps in [10, 20, 50, 100]:
    print(f"\nTrying max_steps={max_steps}...", flush=True)
    result = runner.run(bytecode, data, max_steps=max_steps)
    print(f"  Result: {result}", flush=True)
    
    # Re-create runner for next iteration
    if max_steps < 100:
        runner = AutoregressiveVMRunner()
