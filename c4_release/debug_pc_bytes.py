#!/usr/bin/env python3
"""Debug PC byte prediction at step 0."""
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
print(f"  PC bytes 0-3: {draft_step0[1:5]}")
print(f"  AX bytes 0-3: {draft_step0[6:10]}")

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

# Test PC byte 1 prediction (at token 1 position, predicting token 2)
full_context = context + draft_step0[:2]
token_ids = torch.tensor([full_context], dtype=torch.long)

activations = {}
def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook_fn(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

with torch.no_grad():
    logits = model(token_ids)

predicted = logits[0, -1, :].argmax().item()
expected = draft_step0[2]
print(f"\nPrediction for PC byte 1:")
print(f"  Expected: {expected}")
print(f"  Predicted: {predicted}")
print(f"  Match: {predicted == expected}")

# Analyze at PC byte 0 position
h = activations['L15_ffn'][0, -1, :]
print(f"\n--- At PC byte 0 position ---")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")
print(f"HAS_SE = {h[BD.HAS_SE].item():.3f}")
print(f"H1[PC] = {h[BD.H1 + 0].item():.3f}")

# Check OUTPUT
out_lo = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
out_hi = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"\nOUTPUT_LO: argmax={out_lo.argmax().item()}, max={out_lo.max().item():.3f}")
print(f"OUTPUT_HI: argmax={out_hi.argmax().item()}, max={out_hi.max().item():.3f}")

# Show significant values
sig_out_lo = [(i, f'{v:.3f}') for i, v in enumerate(out_lo.tolist()) if abs(v) > 0.1]
print(f"OUTPUT_LO significant: {sig_out_lo}")

# Trace OUTPUT through layers
print(f"\n--- OUTPUT_LO trace through layers ---")
for layer in range(16):
    h = activations[f'L{layer}_ffn'][0, -1, :]
    lo_max = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    lo_val = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()
    print(f"L{layer:2d}: OUTPUT lo={lo_max} ({lo_val:.3f})")
