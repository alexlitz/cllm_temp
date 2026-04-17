#\!/usr/bin/env python3
"""Check if FETCH has the jump target for JSR."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

# JSR 25
bytecode = [Opcode.JSR | (25 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

# Generate first step
context = runner._build_context(bytecode, b"", [], "")
for i in range(35):
    context.append(model.generate_next(context))

# Check values
device = next(model.parameters()).device
x = torch.tensor([context], dtype=torch.long, device=device)
x = model.embed(x)

pc_idx = 12

# Run through L6
for i in range(6):
    x = model.blocks[i](x)

print("After L6 at PC marker:")
print(f"  OP_JSR: {x[0, pc_idx, BD.OP_JSR].item():.3f}")
print(f"  TEMP[0] (IS_JSR): {x[0, pc_idx, BD.TEMP + 0].item():.3f}")
print(f"  FETCH_LO[0-3] (jump target bytes):")
for k in range(4):
    val_lo = sum(x[0, pc_idx, BD.FETCH_LO + b].item() * (1 << (4*b)) for b in range(k*2, k*2+2))
    print(f"    Byte {k}: {val_lo:.1f}")

# Decode byte 0 from nibbles
nibbles = [x[0, pc_idx, BD.FETCH_LO + k].item() for k in range(16)]
byte0 = sum(nibbles[k] * (2**k) for k in range(8) if nibbles[k] > 0.5)
print(f"  FETCH byte 0 decoded: {int(byte0)}")

print(f"\n  OUTPUT_LO (PC byte 0):")
nibbles_out = [x[0, pc_idx, BD.OUTPUT_LO + k].item() for k in range(16)]
byte0_out = sum(nibbles_out[k] * (2**k) for k in range(8) if nibbles_out[k] > 0.5)
print(f"    Byte 0: {int(byte0_out)}")

print("\nEXPECTED:")
print("  - TEMP[0] ≈ 5.0 ✓ (threshold 4.0)")
print("  - FETCH byte 0 = 25 (jump target)")
print("  - OUTPUT byte 0 = 25 (if JSR override works)")
print("\nIf OUTPUT ≠ 25, L6 FFN JSR override is not firing despite TEMP[0] > threshold.")
