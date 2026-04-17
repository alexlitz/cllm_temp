#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print(f"Bytecode length: {len(bytecode)}")
for i, instr in enumerate(bytecode[:10]):
    print(f"  {i}: 0x{instr:08x}")

runner = AutoregressiveVMRunner()
runner.model.cuda()

print("\nRunning with max_steps=20...")
try:
    output, exit_code = runner.run(bytecode, max_steps=20)
    print(f"Result: exit_code={exit_code}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
