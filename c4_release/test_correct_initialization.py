#!/usr/bin/env python3
"""Test with correct initialization (no manual STEP_END)."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Use runner's context builder (no manual STEP_END)
context = runner._build_context(bytecode, b'', [])
print(f"Initial context: {len(context)} tokens (no STEP_END)")

# Generate step 0
print("\nGenerating step 0...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"  Step 0 complete at token {i}")
        break

# Check PC value at last position (before STEP_END)
se_pos = len(context) - 1
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(16):
        x = model.blocks[i](x, kv_cache=None)

    # Check HAS_SE and PC at STEP_END marker
    print(f"\nAt STEP_END position {se_pos}:")
    print(f"  HAS_SE: {x[0, se_pos, BD.HAS_SE].item():.3f}")

    # Check PC bytes in OUTPUT
    pc_bytes = []
    for i in range(4):
        vals = x[0, se_pos, BD.OUTPUT_LO + i*16 : BD.OUTPUT_LO + (i+1)*16]
        byte_val = vals.argmax(-1).item()
        pc_bytes.append(byte_val)
    pc = sum(b << (i*8) for i, b in enumerate(pc_bytes))
    print(f"  PC: 0x{pc:08x} (expected 0x0000000a)")

    if pc == 0x0a:
        print("✓ PC UPDATE WORKS!")
    else:
        print(f"✗ PC UPDATE BROKEN")

# Generate step 1 to REG_AX marker
print("\nGenerating step 1 to REG_AX...")
for i in range(10):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_AX:
        print(f"  REG_AX at position {len(context)-1}")
        break

# Predict first AX byte
marker_pos = len(context) - 1
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')
with torch.no_grad():
    logits = model.forward(token_ids)
    prediction = logits[0, marker_pos, :].argmax(-1).item()

print(f"\nPredicted AX byte 0: 0x{prediction:02x}")
if prediction == 0x2a:
    print("✓ SUCCESS!")
else:
    print(f"✗ FAILED! Expected 0x2a")
