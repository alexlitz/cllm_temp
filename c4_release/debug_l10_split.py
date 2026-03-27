#!/usr/bin/env python3
"""Check if corruption is from L10 attention or FFN."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

BD = _SetDim

# Simple test: IMM 42, EXIT
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Build context
context = [Token.CODE_START]
for instr in bytecode:
    opcode = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFFFF
    context.append(opcode)
    for i in range(4):
        context.append((imm >> (i * 8)) & 0xFF)
    context.extend([0, 0, 0])
context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])

# Generate draft tokens
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

# Forward pass
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

full_context = context + draft_tokens
device = model.embed.weight.device
x = torch.tensor([full_context], dtype=torch.long, device=device)

with torch.no_grad():
    h = model.embed(x)
    model._inject_code_addr_keys(x, h)
    model._inject_mem_store(x, h)

    ctx_len = len(context)
    pc_b0_pos = ctx_len + 1  # Position 21

    # Run through L0-L9
    for i in range(10):
        h = model.blocks[i](h)

    print(f'PC byte 0 OUTPUT before L10 (after L9):')
    lo_vals = [h[0, pc_b0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [h[0, pc_b0_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f'  LO: {[f"{v:.2f}" for v in lo_vals[:12]]}')
    print(f'  HI: {[f"{v:.2f}" for v in hi_vals[:4]]}')

    # Run L10 attention only
    h_after_attn = model.blocks[10].attn(h)  # L10 attention

    print(f'\nPC byte 0 OUTPUT after L10 attention:')
    lo_vals = [h_after_attn[0, pc_b0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [h_after_attn[0, pc_b0_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f'  LO: {[f"{v:.2f}" for v in lo_vals[:12]]}')
    print(f'  HI: {[f"{v:.2f}" for v in hi_vals[:4]]}')
    if abs(lo_vals[10]) > 1.5:
        print(f'  → Corruption from L10 ATTENTION!')

    # Run L10 FFN
    h_after_ffn = model.blocks[10].ffn(h_after_attn)  # L10 FFN

    print(f'\nPC byte 0 OUTPUT after L10 FFN:')
    lo_vals = [h_after_ffn[0, pc_b0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [h_after_ffn[0, pc_b0_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f'  LO: {[f"{v:.2f}" for v in lo_vals[:12]]}')
    print(f'  HI: {[f"{v:.2f}" for v in hi_vals[:4]]}')
    if abs(lo_vals[10]) > 1.5:
        print(f'  → Corruption from L10 FFN!')
