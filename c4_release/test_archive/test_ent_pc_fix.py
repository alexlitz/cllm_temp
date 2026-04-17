#!/usr/bin/env python3
"""Test that ENT handler now advances PC correctly."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.constants import INSTR_WIDTH

print(f'Testing ENT PC advancement with INSTR_WIDTH={INSTR_WIDTH}')

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

# Show bytecode
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

runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])

print('\nExecution trace:')
for step_num in range(5):
    # Generate tokens
    for i in range(40):
        next_token = runner.model.generate_next(context)
        context.append(next_token)
        if next_token in (Token.STEP_END, Token.HALT):
            break

    # Extract PC
    pc = runner._extract_register(context, Token.REG_PC)
    exec_pc = runner._exec_pc()

    # Get executed opcode
    exec_idx = exec_pc // INSTR_WIDTH
    if 0 <= exec_idx < len(bytecode):
        exec_op = bytecode[exec_idx] & 0xFF
        op_names = {1: 'IMM', 3: 'JSR', 6: 'ENT', 8: 'LEV', 38: 'EXIT'}
        op_name = op_names.get(exec_op, f'op{exec_op}')
        print(f'  Step {step_num}: exec PC={exec_pc:3d} ({op_name:4s}) → output PC={pc:3d}')

    # Check ENT specifically
    if exec_op == 6:  # ENT
        expected_pc = exec_pc + INSTR_WIDTH
        if pc == expected_pc:
            print(f'    ✓ ENT advanced PC correctly: {exec_pc} + {INSTR_WIDTH} = {pc}')
        else:
            print(f'    ✗ ENT did not advance PC: expected {expected_pc}, got {pc}')

    # Update runner state
    runner._last_pc = pc

    # Check for halt
    if context[-1] == Token.HALT:
        print('  HALTED')
        break

print('\nTest complete')
