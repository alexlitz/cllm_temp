#!/usr/bin/env python3
"""Test if transformer understands JSR (Jump to Subroutine) behavior."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.speculative import DraftVM
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH

print("=" * 70)
print("TESTING JSR INSTRUCTION UNDERSTANDING")
print("=" * 70)

print(f"\nConfiguration:")
print(f"  PC_OFFSET = {PC_OFFSET}")
print(f"  INSTR_WIDTH = {INSTR_WIDTH}")

# Create a minimal JSR test
# JSR 16 means: push PC, jump to address 16
bytecode = [
    0x1003,  # JSR 16 (opcode 3, immediate 16)
]

print(f"\nBytecode:")
print(f"  [0] 0x{bytecode[0]:08x} = JSR 16")

# Build context
context = [Token.CODE_START]
for instr in bytecode:
    op = instr & 0xFF
    imm = instr >> 8
    context.append(op)
    for i in range(4):
        context.append((imm >> (i * 8)) & 0xFF)
context.append(Token.CODE_END)
context.append(Token.DATA_START)
context.append(Token.DATA_END)

print(f"\nContext: {context}")

# Test DraftVM
print("\n" + "=" * 70)
print("DRAFTVM")
print("=" * 70)

vm = DraftVM(bytecode)
print(f"Before: idx={vm.idx}, pc={vm.pc}")
vm.step()
print(f"After:  idx={vm.idx}, pc={vm.pc}")

expected_pc = 16  # Should jump to address 16
if vm.pc == expected_pc:
    print(f"✓ DraftVM correctly jumps to PC={expected_pc}")
else:
    print(f"✗ DraftVM has PC={vm.pc}, expected {expected_pc}")

# Test Transformer
print("\n" + "=" * 70)
print("TRANSFORMER")
print("=" * 70)

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

token_ids = torch.tensor([context], dtype=torch.long)
logits = model.forward(token_ids)

# Get top 5 predictions for first token (should be REG_PC)
print("\nTop 5 predictions for first output token:")
probs = torch.softmax(logits[0, -1, :], dim=-1)
top5 = torch.topk(probs, 5)
for i, (prob, tok) in enumerate(zip(top5.values, top5.indices)):
    marker = "✓" if tok.item() == Token.REG_PC else ""
    print(f"  {i+1}. Token {tok.item():3d} : {prob.item():.4f} {marker}")

# Now predict the PC value byte by byte
print("\nPredicting PC value:")
pred_tokens = []
for i in range(5):
    full_ctx = context + pred_tokens
    token_ids = torch.tensor([full_ctx], dtype=torch.long)
    logits = model.forward(token_ids)
    pred = logits[0, -1, :].argmax(-1).item()
    pred_tokens.append(pred)

    if i == 0:
        print(f"  Token {i}: {pred:3d} (REG_PC marker)")
    else:
        print(f"  Token {i}: {pred:3d} (PC byte {i-1})")

predicted_pc = pred_tokens[1] | (pred_tokens[2] << 8) | (pred_tokens[3] << 16) | (pred_tokens[4] << 24)
print(f"\nPredicted PC: {predicted_pc}")
print(f"Expected PC:  {expected_pc}")

if predicted_pc == expected_pc:
    print(f"✓ Transformer correctly predicts PC={expected_pc}")
elif predicted_pc == 8:
    print(f"✗ Transformer predicts PC=8 (next instruction, not jump target)")
    print(f"  This means the model learned to ignore JSR and just advance PC")
else:
    print(f"✗ Transformer predicts PC={predicted_pc} (unexpected value)")

# Check what the transformer sees for instruction encoding
print("\n" + "=" * 70)
print("INSTRUCTION ENCODING CHECK")
print("=" * 70)

print("\nContext encoding of JSR 16:")
print(f"  Context[1] = {context[1]} (opcode byte)")
print(f"  Context[2:6] = {context[2:6]} (immediate bytes)")

print(f"\nDoes opcode 3 appear in context? {3 in context}")
print(f"Does immediate 16 appear as first byte? {context[2] == 16}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if predicted_pc == 8:
    print("""
✗ CRITICAL BUG FOUND:

The transformer is NOT executing the JSR instruction correctly.

Expected behavior:
  JSR 16 → PC = 16 (jump to target address)

Actual behavior:
  JSR 16 → PC = 8 (next sequential instruction)

This means:
1. The transformer has learned to advance PC by INSTR_WIDTH (8 bytes)
2. But it IGNORES the JSR jump behavior
3. This is a fundamental execution bug

Possible causes:
1. Model weights may not have JSR behavior encoded
2. PC_OFFSET configuration mismatch between training and inference
3. Bytecode encoding issue in training data
4. Weight loading error
    """)
elif predicted_pc == expected_pc:
    print("\n✓ Transformer correctly executes JSR!")
else:
    print(f"\n✗ Transformer has unexpected PC={predicted_pc}")
