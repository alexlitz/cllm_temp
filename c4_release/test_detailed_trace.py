#!/usr/bin/env python3
"""Detailed execution trace."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH

print(f'PC_OFFSET={PC_OFFSET}, INSTR_WIDTH={INSTR_WIDTH}')

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

# Show bytecode
print(f'\nBytecode:')
op_names = {0: 'LEA', 1: 'IMM', 3: 'JSR', 6: 'ENT', 8: 'LEV', 38: 'EXIT'}
for i in range(len(bytecode)):
    op = bytecode[i] & 0xFF
    imm = (bytecode[i] >> 8) & 0xFFFFFFFF
    pc = i * INSTR_WIDTH + PC_OFFSET
    print(f'  idx={i} PC={pc:3d}: {op_names.get(op, f"op{op}"):6s} {imm}')

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

# Manual step-by-step execution
runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])
output = []

print(f'\nExecution trace:')
for step_num in range(10):
    # Generate tokens
    for i in range(40):
        next_token = runner.model.generate_next(context)
        context.append(next_token)
        if next_token in (Token.STEP_END, Token.HALT):
            break

    # Extract state
    pc = runner._extract_register(context, Token.REG_PC)
    ax = runner._extract_register(context, Token.REG_AX)

    # Get executed instruction
    exec_pc = runner._exec_pc()
    exec_idx = exec_pc // INSTR_WIDTH

    if 0 <= exec_idx < len(bytecode):
        exec_op = bytecode[exec_idx] & 0xFF
        exec_imm = (bytecode[exec_idx] >> 8) & 0xFFFFFFFF
        print(f'  Step {step_num}: exec idx={exec_idx} PC={exec_pc} ({op_names.get(exec_op, f"op{exec_op}")}) → out PC={pc} AX={ax}')

    # Update runner state
    runner._last_pc = pc

    # Check for halt
    if context[-1] == Token.HALT:
        print(f'\n  HALTED at step {step_num}')
        break

print(f'\nFinal AX: {ax}')
print(f'Expected: 42')
