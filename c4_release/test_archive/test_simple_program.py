#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print(f"Compiled: {len(bytecode)} instructions")

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

print("Running...")
result = runner.run(bytecode, max_steps=100)

print(f"Result: {result}")
if result == ('', 42):
    print("✅✅✅ SUCCESS!")
else:
    print(f"❌ FAIL: Expected ('', 42)")
