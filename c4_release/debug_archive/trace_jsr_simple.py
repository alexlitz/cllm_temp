#!/usr/bin/env python3
"""Simple JSR trace - just show context."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

bytecode = [Opcode.JSR | (25 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

context = runner._build_context(bytecode, b"", [], "")
print(f"Initial context: {context}")
print(f"Context length: {len(context)}")

device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

print("\nCODE section with OP_JSR:")
for i in range(min(10, len(context))):
    token = context[i]
    op_jsr = x[0, i, BD.OP_JSR].item()
    if op_jsr > 0.5:
        print(f"  Position {i}: token={token}, OP_JSR={op_jsr:.1f}")

print("\nNow testing actual JSR execution...")
result = runner.run(bytecode, b"", [], "")
print(f"Exit code: {result}")
print(f"Expected: 42")
print(f"JSR works: {result == 42}")
