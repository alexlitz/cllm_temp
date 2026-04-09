#!/usr/bin/env python3
"""Debug PC byte 1 prediction at step 0."""
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
print(f"  PC bytes: {draft_step0[1:5]} (expected: [10, 0, 0, 0])")

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

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

# Test PC byte 1 prediction (at PC byte 0 position)
full_context = context + draft_step0[:2]  # PC marker + PC byte 0
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
expected = draft_step0[2]
print(f"\nPrediction for PC byte 1 (at PC byte 0 pos):")
print(f"  Expected: {expected}")
print(f"  Predicted: {predicted}")
print(f"  Match: {predicted == expected}")

# Analyze at PC byte 0 position
h = activations['L15_ffn'][0, -1, :]
print(f"\n--- At PC byte 0 position ---")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")
print(f"HAS_SE = {h[BD.HAS_SE].item():.3f}")
print(f"H1[PC] = {h[BD.H1 + 0].item():.3f}")

# Check EMBED
embed_lo = h[BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi = h[BD.EMBED_HI:BD.EMBED_HI+16]
print(f"\nEMBED_LO: argmax={embed_lo.argmax().item()}, max={embed_lo.max().item():.3f}")
print(f"EMBED_HI: argmax={embed_hi.argmax().item()}, max={embed_hi.max().item():.3f}")

# Check OUTPUT
out_lo = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
out_hi = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"\nOUTPUT_LO: argmax={out_lo.argmax().item()}, max={out_lo.max().item():.3f}")
print(f"OUTPUT_HI: argmax={out_hi.argmax().item()}, max={out_hi.max().item():.3f}")

# The issue: at first step PC bytes, nibble copy is suppressed at PC positions.
# But at first step, we need to output the default PC bytes (0 for bytes 1-3).
# Check if there's a mechanism to output 0 at PC byte positions on first step.
print(f"\n--- Trace OUTPUT through layers ---")
for layer in [3, 6, 10, 15]:
    h_l = activations[f'L{layer}_ffn'][0, -1, :]
    lo_max = h_l[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    lo_val = h_l[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()
    print(f"L{layer:2d}: OUTPUT_LO[{lo_max}] = {lo_val:.3f}")

print("\n--- Analysis ---")
print("PC bytes 1-3 are 0, so OUTPUT_LO[0] should be ~1.0")
print("Current implementation suppresses nibble copy at PC positions.")
print("For first step, PC bytes 1-3 need a default output of 0.")
print("This requires either:")
print("  1. Enable nibble copy at PC positions on first step (HAS_SE=0)")
print("  2. Add a first-step-only mechanism to output 0 at PC bytes 1-3")
