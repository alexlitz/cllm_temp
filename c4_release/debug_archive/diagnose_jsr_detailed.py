#!/usr/bin/env python3
"""Detailed JSR diagnostic - check values after key layers."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

bytecode = [Opcode.JSR | (25 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

# Build initial context and generate first step
context = runner._build_context(bytecode, b"", [], "")
print(f"Initial context: {context}")
print(f"Tokens: CODE_START={context[0]}, opcode={context[1]}, imm={context[2]}")

# Generate first 35 tokens (one full step)
for i in range(35):
    next_token = model.generate_next(context)
    context.append(next_token)
    if i == 0:
        print(f"First generated token: {next_token}")

print(f"\nAfter first step, context length: {len(context)}")
print(f"Last 35 tokens (the generated step): {context[-35:]}")

# Now check if PC was overridden
# PC marker should be at position len(context) - 35
pc_idx = len(context) - 35
print(f"\nPC marker at position {pc_idx}: token={context[pc_idx]}")

# Extract PC value from the step
pc_bytes = context[pc_idx+1:pc_idx+5]
pc_value = sum([b << (i*8) for i, b in enumerate(pc_bytes)])
print(f"PC bytes: {pc_bytes}")
print(f"PC value: {pc_value}")
print(f"Expected PC: 25 (if JSR worked) or 5 (PC+5 if JSR failed)")

if pc_value == 25:
    print("\n✓ JSR WORKED! PC was set to jump target 25")
elif pc_value == 5:
    print("\n✗ JSR FAILED. PC advanced normally to PC+5")
else:
    print(f"\n? Unexpected PC value: {pc_value}")
