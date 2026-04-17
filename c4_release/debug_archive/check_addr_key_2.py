#\!/usr/bin/env python3
"""Check how many bytes have ADDR_KEY=2."""

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

print("Checking which bytes have ADDR_KEY=2:")
for i in range(len(context)):
    addr_lo = sum([k for k in range(16) if x[0, i, BD.ADDR_KEY + k].item() > 0.5])
    addr_hi = sum([k for k in range(16) if x[0, i, BD.ADDR_KEY + 16 + k].item() > 0.5])
    addr = addr_lo + addr_hi * 16
    if addr == 2:
        print(f"  Position {i}: token={context[i]}, OP_JSR={x[0, i, BD.OP_JSR].item():.1f}")

print("\nIf multiple positions have ADDR_KEY=2, softmax will split weight\!")
