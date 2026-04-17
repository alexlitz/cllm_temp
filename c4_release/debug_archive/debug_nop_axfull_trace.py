#!/usr/bin/env python3
"""Debug NOP - trace AX_FULL source layer by layer."""
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

print(f"AX_FULL_LO dims: {BD.AX_FULL_LO}-{BD.AX_FULL_LO+15} (467-482)")
print(f"AX_FULL_HI dims: {BD.AX_FULL_HI}-{BD.AX_FULL_HI+15} (483-498)")
print(f"FORMAT_PTR_LO dims: {BD.FORMAT_PTR_LO if hasattr(BD, 'FORMAT_PTR_LO') else 'N/A'}")
print(f"FORMAT_PTR_HI dims: {BD.FORMAT_PTR_HI if hasattr(BD, 'FORMAT_PTR_HI') else 'N/A'}")

with torch.no_grad():
    x = model.embed(token_ids)

    # Check after embedding
    h8_embed = x[0, -1, BD.AX_FULL_HI + 8].item()
    h13_embed = x[0, -1, BD.AX_FULL_HI + 13].item()
    print(f"\nAfter embed: AX_FULL_HI[8]={h8_embed:.4f}, [13]={h13_embed:.4f}")

    # Check after each layer
    for i in range(6):
        x = model.blocks[i](x)
        h8 = x[0, -1, BD.AX_FULL_HI + 8].item()
        h13 = x[0, -1, BD.AX_FULL_HI + 13].item()
        if abs(h8) > 0.01 or abs(h13) > 0.01:
            print(f"After L{i}: AX_FULL_HI[8]={h8:.4f}, [13]={h13:.4f}")

    # Also check what HAS_SE is at AX marker
    print(f"\nHAS_SE at AX marker: {x[0, -1, BD.HAS_SE].item()}")
