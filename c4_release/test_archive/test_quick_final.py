#!/usr/bin/env python3
"""Quick test of fixes."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH

print(f'PC_OFFSET={PC_OFFSET}, INSTR_WIDTH={INSTR_WIDTH}')

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print(f'Bytecode: {len(bytecode)} instructions')
print(f'First 3: JSR {bytecode[0] >> 8}, op {bytecode[1] & 0xFF}, op {bytecode[2] & 0xFF}')

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

print('Running with max_steps=50...')
try:
    result = runner.run(bytecode, max_steps=50)
    print(f'Result: {result}')
    if result == 42:
        print('✓ SUCCESS!')
    else:
        print(f'✗ Expected 42, got {result}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
