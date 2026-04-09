#!/usr/bin/env python3
"""Debug PC byte prediction at step 0 - first step has no HAS_SE."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)
vm.step()
draft_step0 = vm.draft_tokens()

print("Step 0 (IMM 1):")
print(f"  Full step tokens: {draft_step0}")
print(f"  PC marker + bytes: {draft_step0[0:5]}")

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        op, imm = instr & 0xFF, instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])
    return context

context = build_context(bytecode)
print(f"\nContext length: {len(context)}")
print(f"Context ends with: {context[-5:]}")

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

# Test PC byte 0 prediction (at PC marker position)
full_context = context + draft_step0[:1]  # Just PC marker
token_ids = torch.tensor([full_context], dtype=torch.long)

activations = {}
def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

with torch.no_grad():
    logits = model(token_ids)

predicted = logits[0, -1, :].argmax().item()
expected = draft_step0[1]
print(f"\nPrediction for PC byte 0 (at PC marker):")
print(f"  Expected: {expected}")
print(f"  Predicted: {predicted}")
print(f"  Match: {predicted == expected}")

# Analyze at PC marker position
h = activations['L3_ffn'][0, -1, :]
print(f"\n--- At PC marker (L3 FFN) ---")
print(f"MARK_PC = {h[BD.MARK_PC].item():.3f}")
print(f"HAS_SE = {h[BD.HAS_SE].item():.3f}")
print(f"OUTPUT_LO: argmax={h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()}, max={h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item():.3f}")

# Check L3 PC default output
print(f"\n--- L3 PC default ---")
for lo in range(16):
    val = h[BD.OUTPUT_LO + lo].item()
    if abs(val) > 0.1:
        print(f"OUTPUT_LO[{lo}] = {val:.3f}")

h15 = activations['L15_ffn'][0, -1, :]
print(f"\n--- Final OUTPUT (L15 FFN) ---")
print(f"OUTPUT_LO: argmax={h15[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()}, max={h15[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item():.3f}")
