#!/usr/bin/env python3
"""Check if MARK_PC is set at PC marker."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

bytecode = [Opcode.JSR | (25 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

context = runner._build_context(bytecode, b"", [], "")
# Generate first step
for i in range(35):
    context.append(model.generate_next(context))

device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

pc_idx = len(context) - 35  # PC marker position

print(f"PC marker at position {pc_idx}:")
print(f"  Token: {context[pc_idx]} (257 = PC_MARKER)")
print(f"  MARK_PC: {x[0, pc_idx, BD.MARK_PC].item():.3f}")
print(f"  HAS_SE: {x[0, pc_idx, BD.HAS_SE].item():.3f}")

print(f"\nFor L5 head 7 to fire:")
print(f"  Needs MARK_PC ≈ 1.0 AND HAS_SE ≈ 0.0")
print(f"  Q score = L * MARK_PC - L * HAS_SE")
print(f"  Q score = L * {x[0, pc_idx, BD.MARK_PC].item():.1f} - L * {x[0, pc_idx, BD.HAS_SE].item():.1f}")
print(f"  Q score = L * {x[0, pc_idx, BD.MARK_PC].item() - x[0, pc_idx, BD.HAS_SE].item():.1f}")

if x[0, pc_idx, BD.MARK_PC].item() < 0.5:
    print(f"\n❌ MARK_PC is 0! L5 head 7 won't fire!")
    print(f"  PC marker token (257) doesn't have MARK_PC set in embedding.")
