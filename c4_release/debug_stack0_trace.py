#\!/usr/bin/env python3
"""Trace OUTPUT at STACK0 marker through all layers."""
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
full_context = context + draft_step0 + draft_step1 + draft_step2[:21]

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

print("--- OUTPUT_LO at STACK0 marker through all layers ---")
print("Expected: OUTPUT_LO[1] should be high (value 1)")
print()
for layer in range(16):
    h_attn = activations[f'L{layer}_attn'][0, -1, :]
    h_ffn = activations[f'L{layer}_ffn'][0, -1, :]
    
    lo_attn = [h_attn[BD.OUTPUT_LO + i].item() for i in range(16)]
    lo_ffn = [h_ffn[BD.OUTPUT_LO + i].item() for i in range(16)]
    
    lo_attn_max = max(enumerate(lo_attn), key=lambda x: x[1])
    lo_ffn_max = max(enumerate(lo_ffn), key=lambda x: x[1])
    
    # Show significant values
    sig_ffn = [(i, f'{v:.3f}') for i, v in enumerate(lo_ffn) if abs(v) > 0.1]
    
    print(f"L{layer:2d}: attn max={lo_attn_max[0]} ({lo_attn_max[1]:.3f}), ffn max={lo_ffn_max[0]} ({lo_ffn_max[1]:.3f})")
    if layer >= 10 and sig_ffn:
        print(f"     significant: {sig_ffn}")
