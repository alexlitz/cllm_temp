#!/usr/bin/env python3
"""Debug NOP - trace L4 attention vs FFN separately."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

def build_context(bytecode, data=b""):
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

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

bytecode = [Opcode.NOP]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
draft_tokens = draft.draft_tokens()

# Build context up to AX marker
current_ctx = context + draft_tokens[:6]  # Through AX marker

token_ids = torch.tensor([current_ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    # Run L0-L3
    for i in range(4):
        x = model.blocks[i](x)

    print(f"After L3: AX_FULL_HI[8]={x[0, -1, BD.AX_FULL_HI + 8].item():.4f}")

    # Run L4 attention only
    block4 = model.blocks[4]
    x_attn = block4.attn(x)
    x_after_attn = x + x_attn  # residual

    print(f"After L4 attn: AX_FULL_HI[8]={x_after_attn[0, -1, BD.AX_FULL_HI + 8].item():.4f}")

    # Run L4 FFN
    x_ffn = block4.ffn(x_after_attn)
    x_after_ffn = x_after_attn + x_ffn  # residual

    print(f"After L4 FFN: AX_FULL_HI[8]={x_after_ffn[0, -1, BD.AX_FULL_HI + 8].item():.4f}")

    # Check which dim 491 value changes
    print(f"\nDim 491 (AX_FULL_HI[8]):")
    print(f"  After L3:     {x[0, -1, 491].item():.4f}")
    print(f"  After L4 attn: {x_after_attn[0, -1, 491].item():.4f}")
    print(f"  After L4 FFN:  {x_after_ffn[0, -1, 491].item():.4f}")

    # What about dim 496 (AX_FULL_HI[13])?
    print(f"\nDim 496 (AX_FULL_HI[13]):")
    print(f"  After L3:     {x[0, -1, 496].item():.4f}")
    print(f"  After L4 attn: {x_after_attn[0, -1, 496].item():.4f}")
    print(f"  After L4 FFN:  {x_after_ffn[0, -1, 496].item():.4f}")
