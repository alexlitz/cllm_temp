#\!/usr/bin/env python3
"""Decompose L15 Head 0 scores for val byte positions."""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

def build_context(bytecode, data=b''):
    context = [Token.CODE_START]
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
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

def main():
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    context = build_context(BYTECODE)
    draft = DraftVM(BYTECODE)
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    step_start = len(context) - 35
    target_token = 21
    context_for_pred = context[:step_start + target_token]
    query_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)
    with torch.no_grad():
        for i in range(15):
            x = model.blocks[i](x)

    attn = model.blocks[15].attn
    B, S, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    with torch.no_grad():
        Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
        K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)

    h = 0
    scale = attn.scale  # 1/sqrt(HD) = 1/8 = 0.125
    q = Q[0, h, query_pos, :]

    print(f"Scale = {scale:.4f} (1/sqrt(HD))")
    print(f"\n=== Q·K decomposition for val byte positions ===")
    
    for pos in [109, 112]:
        k = K[0, h, pos, :]
        total = (q * k).sum().item() * scale
        
        print(f"\nPosition {pos} (token {context_for_pred[pos]}):")
        print(f"  Total score: {total:.2f}")
        
        # Decompose by dim groups
        # Dim 0: bias
        d0 = q[0] * k[0] * scale
        print(f"  Dim 0 (bias): Q={q[0]:.2f}, K={k[0]:.2f}, contrib={d0:.2f}")
        
        # Dim 1: store anchor
        d1 = q[1] * k[1] * scale
        print(f"  Dim 1 (store): Q={q[1]:.2f}, K={k[1]:.2f}, contrib={d1:.2f}")
        
        # Dim 2: ZFOD offset
        d2 = q[2] * k[2] * scale
        print(f"  Dim 2 (ZFOD): Q={q[2]:.2f}, K={k[2]:.2f}, contrib={d2:.2f}")
        
        # Dim 3: byte select
        d3 = q[3] * k[3] * scale
        print(f"  Dim 3 (byte): Q={q[3]:.2f}, K={k[3]:.2f}, contrib={d3:.2f}")
        
        # Dims 4-27: address (24 bits)
        d_addr = (q[4:28] * k[4:28]).sum().item() * scale
        print(f"  Dims 4-27 (addr): contrib={d_addr:.2f}")
        
        # Dim 28: position gate
        d28 = q[28] * k[28] * scale
        print(f"  Dim 28 (pos gate): Q={q[28]:.2f}, K={k[28]:.2f}, contrib={d28:.2f}")
        
        # Remaining dims
        d_rest = (q[29:] * k[29:]).sum().item() * scale
        print(f"  Dims 29+: contrib={d_rest:.2f}")

if __name__ == "__main__":
    main()
