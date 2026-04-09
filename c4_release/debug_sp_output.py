#\!/usr/bin/env python3
"""Trace OUTPUT at SP marker through layers."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)
vm.step(); draft_step0 = vm.draft_tokens()
vm.step(); draft_step1 = vm.draft_tokens()
vm.step(); draft_step2 = vm.draft_tokens()

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
full_context = context + draft_step0 + draft_step1 + draft_step2[:11]

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

print("--- OUTPUT at SP marker (last token) through layers ---")
print("Expected: lo=8, hi=15 for SP byte 0 = 248 = 0xF8")
print()
for layer in range(16):
    h = activations[f'L{layer}_ffn'][0, -1, :]
    out_lo = [h[BD.OUTPUT_LO + i].item() for i in range(16)]
    out_hi = [h[BD.OUTPUT_HI + i].item() for i in range(16)]
    lo_max_idx, lo_max_val = max(enumerate(out_lo), key=lambda x: x[1])
    hi_max_idx, hi_max_val = max(enumerate(out_hi), key=lambda x: x[1])
    
    # Also show EMBED at marker
    embed_lo = [h[BD.EMBED_LO + i].item() for i in range(16)]
    embed_hi = [h[BD.EMBED_HI + i].item() for i in range(16)]
    emb_lo_max_idx = max(enumerate(embed_lo), key=lambda x: x[1])[0]
    emb_hi_max_idx = max(enumerate(embed_hi), key=lambda x: x[1])[0]
    
    print(f"L{layer:2d}: OUTPUT_LO max={lo_max_idx:2d} ({lo_max_val:6.3f}), OUTPUT_HI max={hi_max_idx:2d} ({hi_max_val:6.3f}) | EMBED_LO max={emb_lo_max_idx}, EMBED_HI max={emb_hi_max_idx}")
    
    # Show all significant OUTPUT_LO values
    if layer >= 5:
        sig_lo = [(i, v) for i, v in enumerate(out_lo) if abs(v) > 0.5]
        if sig_lo:
            print(f"       OUTPUT_LO significant: {sig_lo}")
