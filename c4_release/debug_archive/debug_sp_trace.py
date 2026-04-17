#!/usr/bin/env python3
"""Trace where OUTPUT_LO[15] comes from at SP byte 2 position."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

print("Initializing model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# LEA 8
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)

# Position 21 is SP byte 2 (draft index 12)
pos_21 = ctx_len + 12  # SP byte 2 position

print(f"Tracing OUTPUT_LO[15] at SP byte 2 position ({pos_21}):")
print()

# Forward pass with layer-by-layer inspection
with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    output_lo_15 = x[0, pos_21, BD.OUTPUT_LO + 15].item()
    print(f"After embedding: OUTPUT_LO[15] = {output_lo_15:.2f}")

    for i, block in enumerate(model.blocks):
        x = block(x)
        output_lo_15 = x[0, pos_21, BD.OUTPUT_LO + 15].item()
        if abs(output_lo_15) > 0.1:
            print(f"After L{i}: OUTPUT_LO[15] = {output_lo_15:.2f} ***")
        else:
            print(f"After L{i}: OUTPUT_LO[15] = {output_lo_15:.2f}")

# Now let's check what features at position 21 might trigger the FFN
print()
print("=" * 60)
print("Checking features at position 21 after L8 (when OUTPUT_LO[15] appears):")

with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb
    for i, block in enumerate(model.blocks):
        x = block(x)
        if i == 8:  # After L8
            print(f"\n  MARK_SP = {x[0, pos_21, BD.MARK_SP].item():.2f}")
            print(f"  H1[SP] = {x[0, pos_21, BD.H1 + 2].item():.2f}")
            print(f"  IS_BYTE = {x[0, pos_21, BD.IS_BYTE].item():.2f}")
            byte_idx = [x[0, pos_21, BD.BYTE_INDEX_0 + k].item() for k in range(4)]
            print(f"  BYTE_INDEX = {[f'{v:.2f}' for v in byte_idx]}")
            print(f"  HAS_SE = {x[0, pos_21, BD.HAS_SE].item():.2f}")

            # Check CMP dimensions
            cmp = [x[0, pos_21, BD.CMP + k].item() for k in range(8)]
            print(f"  CMP = {[f'{v:.2f}' for v in cmp]}")

            # Check TEMP dimensions
            temp = [x[0, pos_21, BD.TEMP + k].item() for k in range(8)]
            print(f"  TEMP = {[f'{v:.2f}' for v in temp]}")

            # Check PSH related
            print(f"  PSH_AT_SP = {x[0, pos_21, BD.PSH_AT_SP].item():.2f}")

            # Check OP flags
            print(f"  OP_LI = {x[0, pos_21, BD.OP_LI].item():.2f}")
            print(f"  OP_LC = {x[0, pos_21, BD.OP_LC].item():.2f}")
            print(f"  OP_LEA = {x[0, pos_21, BD.OP_LEA].item():.2f}")
            print(f"  OP_LI_RELAY = {x[0, pos_21, BD.OP_LI_RELAY].item():.2f}")
            print(f"  OP_LC_RELAY = {x[0, pos_21, BD.OP_LC_RELAY].item():.2f}")

            break
