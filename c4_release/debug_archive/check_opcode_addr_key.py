#\!/usr/bin/env python3
"""Check if JSR opcode byte has correct ADDR_KEY."""

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

# Check JSR opcode byte (position 1)
opcode_idx = 1
print(f"JSR opcode byte at context position {opcode_idx}:")
print(f"  Token value: {context[opcode_idx]}")
print(f"  OP_JSR: {x[0, opcode_idx, BD.OP_JSR].item():.3f}")
print(f"  ADDR_KEY nibbles:")
for k in range(16):
    lo_val = x[0, opcode_idx, BD.ADDR_KEY + k].item()
    hi_val = x[0, opcode_idx, BD.ADDR_KEY + 16 + k].item()
    top_val = x[0, opcode_idx, BD.ADDR_KEY + 32 + k].item()
    if lo_val > 0.5 or hi_val > 0.5 or top_val > 0.5:
        print(f"    Nibble {k}: LO={lo_val:.1f}, HI={hi_val:.1f}, TOP={top_val:.1f}")

# Decode ADDR_KEY
addr_lo = sum([k for k in range(16) if x[0, opcode_idx, BD.ADDR_KEY + k].item() > 0.5])
addr_hi = sum([k for k in range(16) if x[0, opcode_idx, BD.ADDR_KEY + 16 + k].item() > 0.5])
addr_top = sum([k for k in range(16) if x[0, opcode_idx, BD.ADDR_KEY + 32 + k].item() > 0.5])
print(f"\n  ADDR_KEY decoded: lo={addr_lo}, hi={addr_hi}, top={addr_top}")
print(f"  ADDR_KEY value: {addr_lo + addr_hi * 16 + addr_top * 256}")

print(f"\n  Expected ADDR_KEY = 2 (PC_OFFSET)")
print(f"  Expected OP_JSR = 1.0")
