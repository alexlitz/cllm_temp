#!/usr/bin/env python3
"""Debug OUTPUT_HI[15] at BP marker."""

import torch
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

    # Run 3 steps
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    # Step 4
    draft.step()
    step4_tokens = draft.draft_tokens()

    # Build context up to BP marker
    context_for_pred = context + step4_tokens[:16]
    bp_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)

    print(f"=== Tracing OUTPUT_HI[15] at BP marker (pos {bp_marker_pos}) ===")

    with torch.no_grad():
        for layer_idx in range(16):
            x_before = x.clone()
            x = model.blocks[layer_idx](x)
            delta = (x - x_before)[0, bp_marker_pos, BD.OUTPUT_HI + 15].item()
            if abs(delta) > 0.01:
                print(f"L{layer_idx}: OUTPUT_HI[15] delta = {delta:.4f}")

    print(f"\nFinal OUTPUT_HI[15]: {x[0, bp_marker_pos, BD.OUTPUT_HI + 15].item():.4f}")

    # Check what's writing to OUTPUT_HI[15]
    # L3 copies EMBED_HI to OUTPUT_HI for carry-forward
    # Let me check EMBED_HI at the BP byte 0 positions being attended

    print(f"\n=== EMBED_HI at BP byte 0 positions ===")
    x = model.embed(token_ids)
    with torch.no_grad():
        for i in range(3):  # Run through L0-L2
            x = model.blocks[i](x)

    # BP byte 0 positions (calculated with correct prefix)
    prefix_len = 44
    for step in range(1, 4):
        bp_byte0_pos = prefix_len + (step - 1) * 35 + 16
        if bp_byte0_pos < len(context_for_pred):
            tok = context_for_pred[bp_byte0_pos]
            embed_hi_15 = x[0, bp_byte0_pos, BD.EMBED_HI + 15].item()
            embed_hi_0 = x[0, bp_byte0_pos, BD.EMBED_HI + 0].item()
            print(f"  Step {step} BP byte 0 (pos {bp_byte0_pos}, tok={tok}): EMBED_HI[0]={embed_hi_0:.4f}, EMBED_HI[15]={embed_hi_15:.4f}")

if __name__ == "__main__":
    main()
