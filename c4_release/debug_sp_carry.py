#!/usr/bin/env python3
"""Debug SP carry-forward at step 2 (IMM after PSH)."""
import os
import sys
sys.path.insert(0, os.getcwd())

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# Test program: IMM 0, PSH, IMM 0, MUL, EXIT
bytecode = [Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

# Run DraftVM for all steps
vm = DraftVM(bytecode)
vm.step()  # Step 0: IMM 0
draft_step0 = vm.draft_tokens()
vm.step()  # Step 1: PSH
draft_step1 = vm.draft_tokens()
vm.step()  # Step 2: IMM 0
draft_step2 = vm.draft_tokens()

# Build context
def build_context(bytecode):
    context = []
    context.append(Token.CODE_START)
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.append(Token.DATA_END)
    return context

context = build_context(bytecode)

# Full context up to step 2 token 11
full_context = context + draft_step0 + draft_step1 + draft_step2[:12]

print(f"Bytecode prefix length: {len(context)}")
print(f"Context after step 0: {len(context) + len(draft_step0)}")
print(f"Context after step 1: {len(context) + len(draft_step0) + len(draft_step1)}")
print(f"Prediction context length: {len(full_context)}")

# Step 2 positions
step2_start = len(context) + len(draft_step0) + len(draft_step1)
print(f"\nStep 2 starts at token {step2_start}")
print(f"SP marker (step 2) at token {step2_start + 10}")
print(f"SP byte 0 (step 2) at token {step2_start + 11}")
print(f"SP byte 1 (step 2) at token {step2_start + 12}")

# Step 1 positions (previous step for carry-forward)
step1_start = len(context) + len(draft_step0)
print(f"\nStep 1 (previous) starts at token {step1_start}")
print(f"SP byte 1 (step 1) at token {step1_start + 12}")
print(f"SP byte 1 (step 1) value: {draft_step1[12]}")

# Check expected vs DraftVM
print(f"\nDraftVM step 2 SP byte 1: {draft_step2[12]} (should be 255)")

# Create model
print("\n--- Creating model ---")
model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()

BD = _SetDim

token_ids = torch.tensor([full_context], dtype=torch.long)
print(f"Input sequence length: {token_ids.shape[1]}")

# Get activations
activations = {}
def hook_fn(name):
    def fn(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach().clone()
        else:
            activations[name] = output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook_fn(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

# Run forward pass
print("\n--- Running forward pass ---")
with torch.no_grad():
    logits = model(token_ids)

# Check prediction (logits[-1] predicts next token after context)
predicted = logits[0, -1, :].argmax().item()
expected = draft_step2[12]
print(f"\nPrediction for SP byte 1 (step 2):")
print(f"  Expected: {expected} (0x{expected:02X})")
print(f"  Predicted: {predicted} (0x{predicted:02X})")
print(f"  Match: {predicted == expected}")

# Analyze at step 2 SP byte 0 position (last token in context, position -1)
# This is the position from which we predict SP byte 1
final_h = activations['L15_ffn'][0, -1, :]

print(f"\n--- At step 2 SP byte 0 position (last token) ---")
print(f"PSH_AT_SP = {final_h[BD.PSH_AT_SP].item():.3f}")
print(f"H1[SP] (dim {BD.H1 + 2}) = {final_h[BD.H1 + 2].item():.3f}")
print(f"BYTE_INDEX_0 = {final_h[BD.BYTE_INDEX_0].item():.3f}")
print(f"BYTE_INDEX_1 = {final_h[BD.BYTE_INDEX_1].item():.3f}")
print(f"HAS_SE = {final_h[BD.HAS_SE].item():.3f}")
print(f"MARK_SP = {final_h[BD.MARK_SP].item():.3f}")
print(f"IS_BYTE = {final_h[BD.IS_BYTE].item():.3f}")

# EMBED should have carry-forward value from step 1's SP byte 1
print(f"\nEMBED values (should have step 1 SP byte 1 = 255 = 0xFF):")
print(f"EMBED_LO[15] = {final_h[BD.EMBED_LO + 15].item():.3f}")
print(f"EMBED_HI[15] = {final_h[BD.EMBED_HI + 15].item():.3f}")
print(f"EMBED_LO max: {final_h[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}, val={final_h[BD.EMBED_LO:BD.EMBED_LO+16].max().item():.3f}")
print(f"EMBED_HI max: {final_h[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}, val={final_h[BD.EMBED_HI:BD.EMBED_HI+16].max().item():.3f}")

# OUTPUT values
print(f"\nOUTPUT values (should be 255 = 0xFF at nibbles 15,15):")
print(f"OUTPUT_LO[15] = {final_h[BD.OUTPUT_LO + 15].item():.3f}")
print(f"OUTPUT_HI[15] = {final_h[BD.OUTPUT_HI + 15].item():.3f}")
print(f"OUTPUT_LO max: {final_h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()}, val={final_h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item():.3f}")
print(f"OUTPUT_HI max: {final_h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()}, val={final_h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].max().item():.3f}")

# Check OP_IMM at step 2 AX marker
ax_pos = step2_start + 5 - len(context)
print(f"\n--- OP_IMM at step 2 AX marker (position {ax_pos} in step context) ---")
for layer in [4, 5, 6]:
    ffn_h = activations[f'L{layer}_ffn'][0, step2_start + 5, :]
    print(f"L{layer}: OP_IMM={ffn_h[BD.OP_IMM].item():.3f}, OP_PSH={ffn_h[BD.OP_PSH].item():.3f}")

# Check carry-forward attention at step 2 SP byte 0
print(f"\n--- EMBED at step 2 SP byte 0 through layers ---")
for layer in range(16):
    ffn_h = activations[f'L{layer}_ffn'][0, -1, :]
    embed_lo_max = ffn_h[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
    embed_hi_max = ffn_h[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
    print(f"L{layer:2d}: EMBED_LO max={embed_lo_max}, EMBED_HI max={embed_hi_max}")
