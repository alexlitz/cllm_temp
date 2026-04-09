#!/usr/bin/env python3
"""Debug BP bytes at step 1."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)
vm.step(); draft_step0 = vm.draft_tokens()
vm.step(); draft_step1 = vm.draft_tokens()

print("Step 0 (IMM 1):")
print(f"  BP bytes: {draft_step0[16:20]}")
print("Step 1 (PSH):")
print(f"  BP bytes: {draft_step1[16:20]}")

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

# Test BP byte 1 prediction at step 1 (at BP byte 0 position)
full_context = context + draft_step0 + draft_step1[:17]  # Up to BP byte 0
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
expected = draft_step1[17]
print(f"\nPrediction for BP byte 1 (step 1):")
print(f"  Expected: {expected}")
print(f"  Predicted: {predicted}")
print(f"  Match: {predicted == expected}")

# Analyze
h = activations['L15_ffn'][0, -1, :]
print(f"\n--- At BP byte 0 position (step 1) ---")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")
print(f"HAS_SE = {h[BD.HAS_SE].item():.3f}")
print(f"H1[BP] = {h[BD.H1 + 3].item():.3f}")

# Check OUTPUT
out_lo = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
out_hi = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"\nOUTPUT_LO: argmax={out_lo.argmax().item()}, max={out_lo.max().item():.3f}")
print(f"OUTPUT_HI: argmax={out_hi.argmax().item()}, max={out_hi.max().item():.3f}")

print("\n--- Analysis ---")
print("BP bytes on subsequent steps should passthrough from previous step.")
print("BP is not modified by PSH, so bytes 1-3 should stay [0, 1, 0].")
