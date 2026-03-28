#!/usr/bin/env python3
"""Trace ENT handler."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch

from src.compiler import compile_c, Op
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.constants import INSTR_WIDTH

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])

# Step 0: JSR
for i in range(40):
    next_token = runner.model.generate_next(context)
    context.append(next_token)
    if next_token in (Token.STEP_END, Token.HALT):
        break

print('Step 0: JSR')
pc0 = runner._extract_register(context, Token.REG_PC)
print(f'  Output PC: {pc0}')
runner._last_pc = pc0

# Step 1: ENT
print(f'\nStep 1: ENT at PC={runner._exec_pc()}')
exec_idx = runner._exec_pc() // INSTR_WIDTH
exec_op = bytecode[exec_idx] & 0xFF
print(f'  exec_op: {exec_op} (ENT=6)')

# Check Opcode enum
print(f'  Op.ENT = {Op.ENT}')
print(f'  Op.LEA = {Op.LEA}')
print(f'  Op.JSR = {Op.JSR}')
print(f'  Op.LEV = {Op.LEV}')

# Generate step
for i in range(40):
    next_token = runner.model.generate_next(context)
    context.append(next_token)
    if next_token in (Token.STEP_END, Token.HALT):
        break

pc1 = runner._extract_register(context, Token.REG_PC)
print(f'  Output PC: {pc1}')
print(f'  Expected PC: 24 (16+8)')
print(f'  PC advanced: {pc1 != pc0}')
