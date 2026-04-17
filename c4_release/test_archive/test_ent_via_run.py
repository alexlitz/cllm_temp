#!/usr/bin/env python3
"""Test ENT handler via runner.run() method."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.constants import INSTR_WIDTH

print(f'Testing ENT with INSTR_WIDTH={INSTR_WIDTH}')

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print('\nBytecode:')
for i in range(min(6, len(bytecode))):
    op = bytecode[i] & 0xFF
    pc = i * INSTR_WIDTH
    op_names = {1: 'IMM', 3: 'JSR', 6: 'ENT', 8: 'LEV', 38: 'EXIT'}
    print(f'  idx={i} PC={pc:3d}: {op_names.get(op, f"op{op}")}')

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()
    print('\nUsing CUDA')

print('\nRunning program...')
result = runner.run(bytecode, max_steps=100)
print(f'\nResult: {result}')
print(f'Expected: 42')
print(f'Success: {result == 42}')
