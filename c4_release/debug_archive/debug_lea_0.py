#!/usr/bin/env python3
"""Debug LEA 0 specifically to find why it outputs 192."""

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

# LEA 0 vs LEA 8
for imm, label in [(0, "LEA 0"), (8, "LEA 8")]:
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    print(f"\n{'='*60}")
    print(f"{label}: imm={imm}")
    print(f"Context: {context}")
    print(f"Draft tokens: {draft_tokens[:10]}...")

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
    ctx_len = len(context)

    # Position 6 is AX byte 0 (predicts AX byte 1)
    pos_5 = ctx_len + 5  # AX marker position
    pos_6 = ctx_len + 6  # AX byte 0 position

    with torch.no_grad():
        emb = model.embed(ctx_tensor)
        x = emb

        for i, block in enumerate(model.blocks):
            x = block(x)

            if i == 15:  # After L15
                print(f"\n  After L15 at AX byte 0 (pos {pos_6}):")
                output_lo = [x[0, pos_6, BD.OUTPUT_LO + k].item() for k in range(16)]
                output_hi = [x[0, pos_6, BD.OUTPUT_HI + k].item() for k in range(16)]
                print(f"    OUTPUT_LO: {[f'{v:.1f}' for v in output_lo]}")
                print(f"    OUTPUT_HI: {[f'{v:.1f}' for v in output_hi]}")

                max_lo = max(output_lo)
                max_hi = max(output_hi)
                max_lo_idx = output_lo.index(max_lo)
                max_hi_idx = output_hi.index(max_hi)
                pred = max_lo_idx + 16 * max_hi_idx
                expected = draft_tokens[6]  # AX byte 1
                print(f"    Max lo: index {max_lo_idx} = {max_lo:.1f}")
                print(f"    Max hi: index {max_hi_idx} = {max_hi:.1f}")
                print(f"    Predicted: {pred}, Expected: {expected}")

                # Also check key features
                print(f"\n  Features at pos {pos_6}:")
                print(f"    MARK_AX: {x[0, pos_6, BD.MARK_AX].item():.2f}")
                print(f"    H1[AX]: {x[0, pos_6, BD.H1 + 1].item():.2f}")
                print(f"    IS_BYTE: {x[0, pos_6, BD.IS_BYTE].item():.2f}")
                print(f"    BYTE_INDEX_0: {x[0, pos_6, BD.BYTE_INDEX_0].item():.2f}")

                # Check what's writing OUTPUT_HI[12]
                fetch_hi = [x[0, pos_6, BD.FETCH_HI + k].item() for k in range(16)]
                print(f"    FETCH_HI: {[f'{v:.1f}' for v in fetch_hi]}")
                break
