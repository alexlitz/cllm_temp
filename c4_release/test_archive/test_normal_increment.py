#!/usr/bin/env python3
"""Test normal PC increment to understand the baseline."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

# Simple program: NOP; EXIT
bytecode = [Opcode.NOP, Opcode.EXIT]

runner = AutoregressiveVMRunner()
model = runner.model

context = runner._build_context(bytecode, b"", [], "")
print(f"Initial context: {context}")

# Generate first step
for i in range(35):
    next_token = model.generate_next(context)
    context.append(next_token)

# Check PC value
pc_idx = len(context) - 35
pc_bytes = context[pc_idx+1:pc_idx+5]
pc_value = sum([b << (i*8) for i, b in enumerate(pc_bytes)])

print(f"After executing NOP:")
print(f"  PC value: {pc_value}")
print(f"  Expected: 10 (next instruction at index 1)")
print(f"  PC = instruction_index * INSTR_WIDTH + PC_OFFSET = 1 * 8 + 2 = 10")

if pc_value == 10:
    print("\n✓ Normal PC increment works correctly")
else:
    print(f"\n✗ Unexpected PC: {pc_value}")
