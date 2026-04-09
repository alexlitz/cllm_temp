#!/usr/bin/env python3
"""Debug L10 head 3 STACK0 passthrough activation."""
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

# Get intermediate values before L10 attention
pre_l10 = {}
def pre_hook(name):
    def fn(module, input):
        pre_l10[name] = input[0].detach().clone()
    return fn

model.blocks[10].attn.register_forward_pre_hook(pre_hook('L10_attn_input'))

token_ids = torch.tensor([full_context], dtype=torch.long)
with torch.no_grad():
    logits = model(token_ids)

# Analyze at last position (STACK0 byte 0)
h = pre_l10['L10_attn_input'][0, -1, :]
print(f"\n--- Pre-L10 attention at STACK0 byte 0 position ---")
print(f"IS_BYTE = {h[BD.IS_BYTE].item():.3f}")
print(f"HAS_SE = {h[BD.HAS_SE].item():.3f}")
print(f"L1H4[3] (BP area) = {h[BD.L1H4 + 3].item():.3f}")
print(f"H1[3] (BP) = {h[BD.H1 + 3].item():.3f}")
print(f"BYTE_INDEX_0 = {h[BD.BYTE_INDEX_0].item():.3f}")
print(f"BYTE_INDEX_3 = {h[BD.BYTE_INDEX_3].item():.3f}")
print(f"PSH_AT_SP = {h[BD.PSH_AT_SP].item():.3f}")
print(f"CMP[3] = {h[BD.CMP + 3].item():.3f}")

# Calculate expected Q gate value for head 3
# Q_gate = -30000 + 10000*L1H4[BP] - 10000*H1[BP] + 10000*HAS_SE - 10000*PSH_AT_SP - 10000*CMP[3]
l1h4_bp = h[BD.L1H4 + 3].item()
h1_bp = h[BD.H1 + 3].item()
has_se = h[BD.HAS_SE].item()
psh_at_sp = h[BD.PSH_AT_SP].item()
cmp3 = h[BD.CMP + 3].item()
q_gate = -30000 + 10000*l1h4_bp - 10000*h1_bp + 10000*has_se - 10000*psh_at_sp - 10000*cmp3
print(f"\nExpected Q_gate = {q_gate:.1f}")
print(f"  -30000 + 10000*{l1h4_bp:.3f} - 10000*{h1_bp:.3f} + 10000*{has_se:.3f} - 10000*{psh_at_sp:.3f} - 10000*{cmp3:.3f}")

# Check previous step's STACK0 byte 1 position
prev_step1_stack0_byte1_pos = len(context) + len(draft_step0) + 22  # step 1's STACK0 byte 1
print(f"\nPrev step STACK0 byte 1 position: {prev_step1_stack0_byte1_pos}")
print(f"Current position: {len(full_context)-1}")
if prev_step1_stack0_byte1_pos < len(full_context):
    h_k = pre_l10['L10_attn_input'][0, prev_step1_stack0_byte1_pos, :]
    print(f"\n--- K at prev step STACK0 byte 1 ---")
    print(f"IS_BYTE = {h_k[BD.IS_BYTE].item():.3f}")
    print(f"L1H4[3] = {h_k[BD.L1H4 + 3].item():.3f}")
    print(f"H1[3] = {h_k[BD.H1 + 3].item():.3f}")
    print(f"BYTE_INDEX_1 = {h_k[BD.BYTE_INDEX_1].item():.3f}")
    print(f"CLEAN_EMBED_LO[0] = {h_k[BD.CLEAN_EMBED_LO + 0].item():.3f}")
