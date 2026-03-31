#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

runner = AutoregressiveVMRunner()
runner.model.cuda()

print("Running with max_steps=100...")
output, exit_code = runner.run(bytecode, max_steps=100)
print(f"Result: exit_code={exit_code}")
