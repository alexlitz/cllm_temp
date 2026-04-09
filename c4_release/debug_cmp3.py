#\!/usr/bin/env python3
"""Check CMP[3] at SP byte positions during step 3 (MUL)."""
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
vm.step(); draft_step3 = vm.draft_tokens()

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
# Context up to step 3 token 11 (predicting SP byte 1)
full_context = context + draft_step0 + draft_step1 + draft_step2 + draft_step3[:12]

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

print("--- Step 3 SP byte 0 position (last token, predicting byte 1) ---")
print(f"Expected SP byte 1: {draft_step3[12]} (MUL pops, SP=0x10000, byte 1 = 0)")
print()

h = activations['L10_attn'][0, -1, :]
print(f"CMP[0] = {h[BD.CMP + 0].item():.3f}")
print(f"CMP[1] = {h[BD.CMP + 1].item():.3f}")
print(f"CMP[2] = {h[BD.CMP + 2].item():.3f}")
print(f"CMP[3] = {h[BD.CMP + 3].item():.3f}")
print(f"PSH_AT_SP = {h[BD.PSH_AT_SP].item():.3f}")
print(f"H1[SP] = {h[BD.H1 + 2].item():.3f}")
print(f"HAS_SE = {h[BD.HAS_SE].item():.3f}")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")

# Check OP_MUL at AX marker
step3_ax_pos = len(context) + len(draft_step0) + len(draft_step1) + len(draft_step2) + 5
h_ax = activations['L6_ffn'][0, step3_ax_pos, :]
print(f"\nAt step 3 AX marker:")
print(f"OP_MUL = {h_ax[BD.OP_MUL].item():.3f}")

# Check CMP[3] relay at SP marker
step3_sp_marker = len(context) + len(draft_step0) + len(draft_step1) + len(draft_step2) + 10
h_sp = activations['L7_ffn'][0, step3_sp_marker, :]
print(f"\nAt step 3 SP marker (after L7):")
print(f"CMP[3] = {h_sp[BD.CMP + 3].item():.3f}")
