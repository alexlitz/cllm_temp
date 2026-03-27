#!/usr/bin/env python3
"""Fresh test of token 21 with identity carry fix."""
import sys
import importlib

# Force module reload
if 'neural_vm.vm_step' in sys.modules:
    del sys.modules['neural_vm.vm_step']
if 'neural_vm.embedding' in sys.modules:
    del sys.modules['neural_vm.embedding']
if 'neural_vm.speculative' in sys.modules:
    del sys.modules['neural_vm.speculative']

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

print("Creating fresh model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build sequence up to token 20
current_context = context[:]
for i in range(21):
    current_context.append(draft_tokens[i])

ctx_tensor = torch.tensor([current_context], dtype=torch.long)

# Capture OUTPUT_HI after Layer 6
l6_out = None
def hook_l6(module, input, output):
    global l6_out
    l6_out = output.detach().clone()

model.blocks[6].register_forward_hook(hook_l6)

# Forward pass
print("Running forward pass...")
with torch.no_grad():
    logits = model.forward(ctx_tensor)

predicted = torch.argmax(logits[0, -1]).item()

print()
print("=" * 70)
print(f"Token 21 prediction: {predicted} (expected: 0)")
print()

if l6_out is not None:
    pos = 29
    output_hi = l6_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    argmax_hi = torch.argmax(output_hi).item()
    max_hi = torch.max(output_hi).item()

    print(f"OUTPUT_HI at position {pos} after Layer 6:")
    print(f"  argmax: {argmax_hi}")
    print(f"  max value: {max_hi:.2f}")
    print(f"  OUTPUT_HI[2]: {output_hi[2]:.2f}")
    print()

    if argmax_hi == 0 and output_hi[2] < 0.1:
        print("✅ FIXED! Layer 6 identity carry no longer activating at byte position")
    else:
        print("❌ STILL BROKEN: Layer 6 still writing to OUTPUT_HI at byte position")

print("=" * 70)

if predicted == 0:
    print("✅ Token 21 PASSES!")
else:
    print(f"❌ Token 21 FAILS (predicted {predicted} instead of 0)")
