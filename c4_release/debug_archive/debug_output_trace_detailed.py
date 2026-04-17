#!/usr/bin/env python3
"""Trace OUTPUT_LO[1] through all layers at byte 1 position."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

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

vm = DraftVM(bytecode)
all_tokens = context[:]
for i in range(3):
    vm.step()
    all_tokens.extend(vm.draft_tokens())
vm.step()
step3_tokens = vm.draft_tokens()

activations = {}
def hook(name):
    def fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook(f'L{i}_ffn'))

# At byte 1 position (predicting byte 2)
test_context = all_tokens + step3_tokens[:13]
token_ids = torch.tensor([test_context], dtype=torch.long)

with torch.no_grad():
    logits = model(token_ids)

print("=== OUTPUT_LO[0,1] and OUTPUT_HI[0,1] trace through all layers ===")
print("(At SP byte 1 position, predicting byte 2)")
print()

for layer in range(16):
    h_attn = activations[f'L{layer}_attn'][0, -1, :]
    h_ffn = activations[f'L{layer}_ffn'][0, -1, :]
    
    lo0_attn = h_attn[BD.OUTPUT_LO + 0].item()
    lo1_attn = h_attn[BD.OUTPUT_LO + 1].item()
    hi0_attn = h_attn[BD.OUTPUT_HI + 0].item()
    hi1_attn = h_attn[BD.OUTPUT_HI + 1].item()
    
    lo0_ffn = h_ffn[BD.OUTPUT_LO + 0].item()
    lo1_ffn = h_ffn[BD.OUTPUT_LO + 1].item()
    hi0_ffn = h_ffn[BD.OUTPUT_HI + 0].item()
    hi1_ffn = h_ffn[BD.OUTPUT_HI + 1].item()
    
    # Check for significant change
    lo1_delta_attn = lo1_attn - (activations[f'L{layer-1}_ffn'][0, -1, BD.OUTPUT_LO + 1].item() if layer > 0 else 0)
    lo1_delta_ffn = lo1_ffn - lo1_attn
    
    flag = ""
    if abs(lo1_delta_attn) > 0.1:
        flag += " <-- attn writes LO[1]!"
    if abs(lo1_delta_ffn) > 0.1:
        flag += " <-- ffn writes LO[1]!"
    
    print(f"L{layer:2d} attn: LO[0]={lo0_attn:6.3f}, LO[1]={lo1_attn:6.3f}, HI[0]={hi0_attn:6.3f}, HI[1]={hi1_attn:6.3f}{flag}")
    print(f"L{layer:2d} ffn:  LO[0]={lo0_ffn:6.3f}, LO[1]={lo1_ffn:6.3f}, HI[0]={hi0_ffn:6.3f}, HI[1]={hi1_ffn:6.3f}")
    print()
