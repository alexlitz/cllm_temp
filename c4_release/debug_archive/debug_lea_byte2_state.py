#!/usr/bin/env python3
"""Check available dimensions at AX byte 2 position for LEA."""
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

# LEA 8: AX = BP + 8 = 0x10000 + 8 = 0x10008
# Byte 2 should be 1 (from BP)
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
step1_tokens = draft.draft_tokens()

print(f"LEA 8: AX = {draft.ax} (0x{draft.ax:08X})")
print(f"  Expected AX byte 2: {(draft.ax >> 16) & 0xFF}")
print(f"  BP byte 2: {(draft.bp >> 16) & 0xFF}")

# Context up to AX byte 2 position
# PC marker (1) + PC bytes (4) + AX marker (1) + AX byte 0 (1) + AX byte 1 (1) = 8 tokens
ctx = context + step1_tokens[:8]
ax_byte2_pos = len(ctx) - 1

print(f"\nAX byte 2 position: {ax_byte2_pos}")
print(f"Context length: {len(ctx)}")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)

    print(f"\nAfter embedding at AX byte 2 (pos {ax_byte2_pos}):")
    print(f"  Token value: {ctx[ax_byte2_pos]}")
    print(f"  EMBED_LO: {[x[0, ax_byte2_pos, BD.EMBED_LO + k].item() for k in range(16) if x[0, ax_byte2_pos, BD.EMBED_LO + k].abs().item() > 0.1]}")
    print(f"  OP_LEA: {x[0, ax_byte2_pos, BD.OP_LEA].item():.4f}")
    print(f"  MARK_AX: {x[0, ax_byte2_pos, BD.MARK_AX].item():.4f}")
    print(f"  NEXT_AX: {x[0, ax_byte2_pos, BD.NEXT_AX].item():.4f}")
    print(f"  IS_BYTE: {x[0, ax_byte2_pos, BD.IS_BYTE].item():.4f}")

    # Check if any marker for "AX byte N"
    print(f"\n  Checking for byte position markers:")
    print(f"  AX_BYTE_IDX: {[x[0, ax_byte2_pos, BD.AX_BYTE_IDX + k].item() for k in range(4)]}" if hasattr(BD, 'AX_BYTE_IDX') else "  AX_BYTE_IDX: not defined")

    # Run through layers
    for i in range(len(model.blocks)):
        x = model.blocks[i](x)

    print(f"\nAfter all layers at AX byte 2 (pos {ax_byte2_pos}):")
    output_lo = [x[0, ax_byte2_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, ax_byte2_pos, BD.OUTPUT_HI + k].item() for k in range(16)]

    lo_pred = max(range(16), key=lambda k: output_lo[k])
    hi_pred = max(range(16), key=lambda k: output_hi[k])
    byte_pred = lo_pred + (hi_pred << 4)

    print(f"  OUTPUT_LO: {[(k, output_lo[k]) for k in range(16) if output_lo[k] > 0.5]}")
    print(f"  OUTPUT_HI: {[(k, output_hi[k]) for k in range(16) if output_hi[k] > 0.5]}")
    print(f"  Predicted: {byte_pred}")
    print(f"  Expected: 1")
