#!/usr/bin/env python3
"""Test IMM prediction at REG_AX marker (like test_fresh_build.py)."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

context = runner._build_context(bytecode, b'', [])
print(f"Program: IMM 42, EXIT")
print(f"Initial context: {len(context)} tokens\n")

# Add initial STEP_END (like test_fresh_build.py did)
context.append(Token.STEP_END)
print("Added manual STEP_END to context\n")

# Generate step 0 (should be IMM execution)
print("Generating step 0...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"  Step 0 complete at token {i}\n")
        break

# Generate step 1 up to REG_AX marker
print("Generating step 1 up to REG_AX...")
for i in range(10):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == 260:  # REG_AX
        print(f"  REG_AX at position {len(context)-1}\n")
        break

# Predict first AX byte at REG_AX marker
marker_pos = len(context) - 1
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    logits = model.forward(token_ids)
    prediction = logits[0, marker_pos, :256].argmax(-1).item()

print(f"Predicted AX byte 0: 0x{prediction:02x}")

if prediction == 0x2a:
    print("✓ SUCCESS! Model predicts 0x2a at REG_AX marker")
else:
    print(f"✗ FAILED! Expected 0x2a, got 0x{prediction:02x}")

    # Show top 5
    top5 = torch.topk(logits[0, marker_pos, :256], 5)
    print("\nTop 5 predictions:")
    for val, idx in zip(top5.values, top5.indices):
        print(f"  0x{idx.item():02x}: {val.item():.1f}")

# Now generate the actual byte and see what we get
print("\nGenerating actual AX bytes...")
for i in range(4):
    tok = model.generate_next(context)
    context.append(tok)
    print(f"  Byte {i}: 0x{tok:02x}")

ax_bytes = [context[marker_pos + 1 + i] for i in range(4)]
ax_value = sum(b << (i*8) for i, b in enumerate(ax_bytes))
print(f"\nGenerated AX: 0x{ax_value:08x}")

if ax_value == 0x2a:
    print("✓ Generated AX is correct!")
else:
    print(f"✗ Generated AX is wrong (expected 0x2a)")
