#!/usr/bin/env python3
"""Debug byte 2 OUTPUT trace - what's writing 28?"""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)

for i in range(4):
    vm.step()
    if i == 2:
        draft_step2 = vm.draft_tokens()
    if i == 3:
        draft_step3 = vm.draft_tokens()

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
S = 100.0

vm2 = DraftVM(bytecode)
all_tokens = context[:]
for i in range(3):
    vm2.step()
    all_tokens.extend(vm2.draft_tokens())

vm2.step()
step3_tokens = vm2.draft_tokens()

# Register hooks
activations = {}
def hook(name):
    def fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook(f'L{i}_ffn'))

# Test at byte 2 position (context ends with SP byte 1)
# step3_tokens[:13] = PC marker + 4 bytes + AX marker + 4 bytes + SP marker + 2 bytes
test_context = all_tokens + step3_tokens[:13]
token_ids = torch.tensor([test_context], dtype=torch.long)

print(f"=== At byte 2 position (predicting SP byte 2) ===")
print(f"Context ends with token: {test_context[-1]} (SP byte 1)")
print(f"Expected prediction: {step3_tokens[13]} (should be 1)")
print(f"Prev step SP bytes: {draft_step2[11:15]}")
print(f"This step SP bytes: {draft_step3[11:15]}")

with torch.no_grad():
    logits = model(token_ids)

predicted = logits[0, -1, :].argmax().item()
print(f"Predicted: {predicted}")

# Trace OUTPUT_LO/HI through layers
print(f"\n=== OUTPUT_LO/HI trace through layers ===")
for layer in [0, 3, 5, 6, 7, 8, 9, 10, 11, 15]:
    h = activations[f'L{layer}_ffn'][0, -1, :]
    out_lo_vals = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
    out_hi_vals = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
    lo_idx = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    hi_idx = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
    lo_max = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()
    hi_max = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16].max().item()
    value = lo_idx + hi_idx * 16
    print(f"L{layer:2d}: OUTPUT_LO[{lo_idx}]={lo_max:.2f}, OUTPUT_HI[{hi_idx}]={hi_max:.2f} → value={value}")

# Check key flags at byte 2 position
print(f"\n=== Key flags at byte 2 position (after L10) ===")
h = activations['L10_ffn'][0, -1, :]
print(f"H1[SP]: {h[BD.H1 + 2].item():.3f}")
print(f"IS_BYTE: {h[BD.IS_BYTE].item():.3f}")
print(f"CMP[3] (binary pop): {h[BD.CMP + 3].item():.3f}")
print(f"SP_POP_CARRY_0: {h[BD.SP_POP_CARRY_0].item():.3f}")
print(f"BYTE_INDEX_0: {h[BD.BYTE_INDEX_0].item():.3f}")
print(f"BYTE_INDEX_1: {h[BD.BYTE_INDEX_1].item():.3f}")
print(f"BYTE_INDEX_2: {h[BD.BYTE_INDEX_2].item():.3f}")
print(f"BYTE_INDEX_3: {h[BD.BYTE_INDEX_3].item():.3f}")
print(f"HAS_SE: {h[BD.HAS_SE].item():.3f}")
print(f"PSH_AT_SP: {h[BD.PSH_AT_SP].item():.3f}")

# Check EMBED at this position
h_embed = activations['L6_attn'][0, -1, :]
embed_lo = h_embed[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
embed_hi = h_embed[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
print(f"\nEMBED: lo={embed_lo}, hi={embed_hi} → value={embed_lo + embed_hi*16}")

# Check what the passthrough should give
print(f"\n=== L10 Passthrough Analysis ===")
# Passthrough should copy EMBED → OUTPUT
h10_attn = activations['L10_attn'][0, -1, :]
h10_ffn = activations['L10_ffn'][0, -1, :]

print("After L10 attn (before FFN):")
out_lo_idx = h10_attn[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
out_hi_idx = h10_attn[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
print(f"  OUTPUT: lo_idx={out_lo_idx}, hi_idx={out_hi_idx} → value={out_lo_idx + out_hi_idx*16}")

print("\nAfter L10 FFN:")
out_lo_idx = h10_ffn[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
out_hi_idx = h10_ffn[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
print(f"  OUTPUT: lo_idx={out_lo_idx}, hi_idx={out_hi_idx} → value={out_lo_idx + out_hi_idx*16}")

# Also check L9 (before passthrough)
h9 = activations['L9_ffn'][0, -1, :]
out_lo_idx = h9[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
out_hi_idx = h9[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
print(f"\nBefore L10 (after L9): OUTPUT_LO[{out_lo_idx}], OUTPUT_HI[{out_hi_idx}] → value={out_lo_idx + out_hi_idx*16}")
