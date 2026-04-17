#\!/usr/bin/env python3
"""Test with fresh model build - no caching."""
import sys
import torch

# Force reimport of modules
if 'neural_vm' in sys.modules:
    del sys.modules['neural_vm']
if 'neural_vm.vm_step' in sys.modules:
    del sys.modules['neural_vm.vm_step']

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode

BD = _SetDim

print("Building fresh model...")
model = AutoregressiveVM()
print("Setting weights...")
set_vm_weights(model)
print("Moving to CUDA...")
model = model.cuda()
model.eval()

# Build context manually
from neural_vm.embedding import Token
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = [Token.CODE_START]
for instr in bytecode:
    opcode_byte = instr & 0xFF
    imm_bytes = [(instr >> (8 + i*8)) & 0xFF for i in range(4)]
    context.append(opcode_byte)
    context.extend(imm_bytes)
    context.extend([0, 0, 0])  # 3 padding bytes
context.append(Token.CODE_END)
context.append(Token.DATA_START)
context.append(Token.DATA_END)
context.append(Token.STEP_END)

print(f"Initial context: {len(context)} tokens")

# Generate step 0
print("Generating step 0...")
for i in range(100):
    token_ids = torch.tensor([context], dtype=torch.long, device='cuda')
    with torch.no_grad():
        logits = model.forward(token_ids)
        tok = logits[0, -1, :].argmax(-1).item()
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"  Step 0 complete at token {i}")
        break

# Generate step 1 to REG_AX marker
print("Generating step 1 to REG_AX...")
for i in range(10):
    token_ids = torch.tensor([context], dtype=torch.long, device='cuda')
    with torch.no_grad():
        logits = model.forward(token_ids)
        tok = logits[0, -1, :].argmax(-1).item()
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

print(f"\nPredicted: 0x{prediction:02x}")
if prediction == 0x2a:
    print("✓ SUCCESS\!")
    sys.exit(0)
else:
    print(f"✗ FAILED\! Expected 0x2a")
    top5 = torch.topk(logits[0, marker_pos, :256], 5)
    print("Top 5:")
    for val, idx in zip(top5.values, top5.indices):
        print(f"  0x{idx.item():02x}: {val.item():.1f}")
    sys.exit(1)
