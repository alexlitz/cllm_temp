#!/usr/bin/env python3
"""Check ADDR_KEY values for JSR instruction bytes."""

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
device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

print("JSR instruction ADDR_KEY values:")
print("Expected: CODE_START at 0, opcode at 1, imm bytes at 2-5")
print()

for i in range(min(10, len(context))):
    token = context[i]
    # Decode ADDR_KEY
    addr_lo = sum([k for k in range(16) if x[0, i, BD.ADDR_KEY + k].item() > 0.5])
    addr_hi = sum([k for k in range(16) if x[0, i, BD.ADDR_KEY + 16 + k].item() > 0.5])
    addr_top = sum([k for k in range(16) if x[0, i, BD.ADDR_KEY + 32 + k].item() > 0.5])
    addr = addr_lo + addr_hi * 16 + addr_top * 256

    op_jsr = x[0, i, BD.OP_JSR].item()

    if addr > 0:
        print(f"Position {i}: token={token:3d}, ADDR_KEY={addr:3d}, OP_JSR={op_jsr:.1f}")

print()
print("For L5 head 3 to fetch immediate:")
print("  Should find ADDR_KEY=3 (PC_OFFSET + 1)")
print("  Immediate byte should be 25 (0x19)")
