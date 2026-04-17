#!/usr/bin/env python3
"""Debug PSH SP byte prediction issue."""
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

# Run DraftVM to get context + draft tokens for step 2
vm = DraftVM(bytecode)
vm.step()  # Step 0: IMM 0
draft_step1 = vm.draft_tokens()
vm.step()  # Step 1: PSH
draft_step2 = vm.draft_tokens()

# Build context (same as UltraBatchRunner._build_context)
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

# Append step 1 draft tokens
context = context + draft_step1

print(f"Context length after step 1: {len(context)}")
print(f"Step 2 draft tokens (first 15): {draft_step2[:15]}")

# Token positions in step 2:
step2_start = len(context)
print(f"\nStep 2 starts at token {step2_start}")
print(f"SP marker at token {step2_start + 10}")
print(f"SP byte 1 at token {step2_start + 12}")

# Expected values for step 2:
# After PSH: SP = 0x10000 - 4 = 0xFFFC
# SP bytes: [0xFC, 0xFF, 0x00, 0x01]
print(f"\nExpected SP value after PSH: 0xFFFC")
print(f"SP byte 0: 0xFC (252)")
print(f"SP byte 1: 0xFF (255)")
print(f"SP byte 2: 0x00 (0)")
print(f"SP byte 3: 0x01 (1)")

# Create model
print("\n--- Creating model ---")
model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()

BD = _SetDim

# We want to predict token 12 of step 2 (SP byte 1)
# Context for prediction: step 1 full + step 2 tokens 0-11
pred_context = context + draft_step2[:12]

token_ids = torch.tensor([pred_context], dtype=torch.long)
print(f"Input sequence length: {token_ids.shape[1]}")

# Get activations at each layer
activations = {}
def hook_fn(name):
    def fn(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach().clone()
        else:
            activations[name] = output.detach().clone()
    return fn

# Register hooks
for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook_fn(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

# Run forward pass
print("\n--- Running forward pass ---")
with torch.no_grad():
    logits = model(token_ids)

# Check the prediction
predicted = logits[0, -1, :].argmax().item()
expected = draft_step2[12]
print(f"\nPrediction for SP byte 1:")
print(f"  Expected: {expected} (0x{expected:02X})")
print(f"  Predicted: {predicted} (0x{predicted:02X})")
print(f"  Match: {predicted == expected}")

# Analyze key dimensions at SP byte 1 position (last token in context)
final_h = activations['L15_ffn'][0, -1, :]

print(f"\n--- At SP byte 1 position (last token) after L15 FFN ---")
print(f"PSH_AT_SP = {final_h[BD.PSH_AT_SP].item():.3f}")
print(f"H1[SP] (dim {BD.H1 + 2}) = {final_h[BD.H1 + 2].item():.3f}")
print(f"BYTE_INDEX_0 = {final_h[BD.BYTE_INDEX_0].item():.3f}")
print(f"BYTE_INDEX_1 = {final_h[BD.BYTE_INDEX_1].item():.3f}")
print(f"MARK_SP = {final_h[BD.MARK_SP].item():.3f}")
print(f"IS_BYTE = {final_h[BD.IS_BYTE].item():.3f}")

# Check OUTPUT values
print(f"\nOUTPUT values:")
print(f"OUTPUT_LO (first 8): {[f'{final_h[BD.OUTPUT_LO + i].item():.2f}' for i in range(8)]}")
print(f"OUTPUT_HI (first 4): {[f'{final_h[BD.OUTPUT_HI + i].item():.2f}' for i in range(4)]}")
print(f"OUTPUT_LO[15] = {final_h[BD.OUTPUT_LO + 15].item():.3f}")
print(f"OUTPUT_HI[15] = {final_h[BD.OUTPUT_HI + 15].item():.3f}")

# Trace PSH_AT_SP through layers
print(f"\n--- PSH_AT_SP at SP byte 1 position through layers ---")
for layer in range(16):
    attn_h = activations[f'L{layer}_attn'][0, -1, :]
    ffn_h = activations[f'L{layer}_ffn'][0, -1, :]
    print(f"L{layer:2d}: after_attn={attn_h[BD.PSH_AT_SP].item():7.3f}, after_ffn={ffn_h[BD.PSH_AT_SP].item():7.3f}")

# Check OP_PSH at AX marker position
# AX marker is token 5 of step 2, which is at pred_context[step2_start + 5]
ax_marker_offset = len(pred_context) - (12 - 5)  # Position of AX marker in pred_context
print(f"\n--- OP_PSH at AX marker (absolute position {step2_start + 5}) ---")
for layer in [4, 5, 6]:
    ffn_h = activations[f'L{layer}_ffn'][0, step2_start + 5 - len(context), :]
    if step2_start + 5 < len(pred_context):
        print(f"L{layer}: OP_PSH={ffn_h[BD.OP_PSH].item():.3f}")

# Check SP marker position
sp_marker_offset = step2_start + 10 - len(context)
print(f"\n--- PSH_AT_SP at SP marker (absolute position {step2_start + 10}) ---")
if sp_marker_offset < len(pred_context) - len(context):
    for layer in [5, 6, 7]:
        ffn_h = activations[f'L{layer}_ffn'][0, sp_marker_offset, :]
        print(f"L{layer}: PSH_AT_SP={ffn_h[BD.PSH_AT_SP].item():.3f}")
