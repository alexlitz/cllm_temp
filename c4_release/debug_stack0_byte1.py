#!/usr/bin/env python3
"""Debug STACK0 byte 1 prediction for IMM after PSH."""
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
vm.step(); draft_step2 = vm.draft_tokens()

print(f"Step 1 (PSH): STACK0 = {draft_step1[21:25]}")
print(f"Step 2 (IMM): STACK0 = {draft_step2[21:25]}")
print(f"Expected STACK0 byte 1 at step 2: {draft_step2[22]}")

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
# Position: predicting STACK0 byte 1 (token 22 within step 2)
full_context = context + draft_step0 + draft_step1 + draft_step2[:22]

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

activations = {}
def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook_fn(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook_fn(f'L{i}_ffn'))

token_ids = torch.tensor([full_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)

predicted = logits[0, -1, :].argmax().item()
expected = draft_step2[22]
print(f"\nPrediction: Expected {expected}, Predicted {predicted}, Match: {predicted == expected}")

# Check at last position (STACK0 byte 0)
h = activations['L15_ffn'][0, -1, :]
print(f"\n--- At STACK0 byte 0 position (predicting byte 1) ---")
print(f"MARK_STACK0 = {h[BD.MARK_STACK0].item():.3f}")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")
print(f"PSH_AT_SP = {h[BD.PSH_AT_SP].item():.3f}")
print(f"CMP[3] = {h[BD.CMP + 3].item():.3f}")

# Check EMBED and OUTPUT
embed_lo = h[BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi = h[BD.EMBED_HI:BD.EMBED_HI+16]
out_lo = h[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
out_hi = h[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"\nEMBED_LO: argmax={embed_lo.argmax().item()}, max={embed_lo.max().item():.3f}")
print(f"EMBED_HI: argmax={embed_hi.argmax().item()}, max={embed_hi.max().item():.3f}")
print(f"OUTPUT_LO: argmax={out_lo.argmax().item()}, max={out_lo.max().item():.3f}")
print(f"OUTPUT_HI: argmax={out_hi.argmax().item()}, max={out_hi.max().item():.3f}")

# Show significant values
sig_out_lo = [(i, f'{v:.3f}') for i, v in enumerate(out_lo.tolist()) if abs(v) > 0.1]
print(f"OUTPUT_LO significant: {sig_out_lo}")

# Trace OUTPUT_LO through layers
print(f"\n--- OUTPUT_LO trace through layers ---")
for layer in range(16):
    h_attn = activations[f'L{layer}_attn'][0, -1, :]
    h_ffn = activations[f'L{layer}_ffn'][0, -1, :]

    lo_attn_max = h_attn[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    lo_attn_val = h_attn[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()
    lo_ffn_max = h_ffn[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    lo_ffn_val = h_ffn[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()

    print(f"L{layer:2d}: attn lo={lo_attn_max} ({lo_attn_val:.3f}), ffn lo={lo_ffn_max} ({lo_ffn_val:.3f})")
