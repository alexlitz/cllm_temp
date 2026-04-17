#\!/usr/bin/env python3
"""Trace where OP_IMM comes from in successful IMM execution."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

# IMM program that works
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
model = runner.model

# Generate first step
context = runner._build_context(bytecode, b"", [], "")
for i in range(35):
    context.append(model.generate_next(context))

# Trace through layers
device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

pc_idx = 12
ax_idx = 17

print("Tracing OP_IMM through layers:")
print(f"After embedding:")
print(f"  PC marker: {x[0, pc_idx, BD.OP_IMM].item():.3f}")
print(f"  AX marker: {x[0, ax_idx, BD.OP_IMM].item():.3f}")

for i in range(7):
    x = model.blocks[i](x)
    if i == 4:  # After L5
        print(f"After L{i+1} (L5 head 7 relay):")
        print(f"  PC marker: {x[0, pc_idx, BD.OP_IMM].item():.3f}")
        print(f"  AX marker: {x[0, ax_idx, BD.OP_IMM].item():.3f}")
    if i == 5:  # After L6
        print(f"After L{i+1} (L6 head 5 relay):")
        print(f"  PC marker: {x[0, pc_idx, BD.OP_IMM].item():.3f}")
        print(f"  AX marker: {x[0, ax_idx, BD.OP_IMM].item():.3f}")

print("\nIf AX marker has OP_IMM ≈ 5-6 after L6, that's how IMM works\!")
