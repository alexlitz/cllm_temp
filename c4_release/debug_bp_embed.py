#!/usr/bin/env python3
"""Debug EMBED_LO at BP marker in step 4."""

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

    # Run 3 steps to get to step 4
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    # Step 4
    draft.step()
    step4_tokens = draft.draft_tokens()

    # Build context up to BP marker
    target_token = 16  # BP byte 0
    context_for_pred = context + step4_tokens[:target_token]
    bp_marker_pos = len(context_for_pred) - 1

    print(f"=== Debugging EMBED at BP marker (pos {bp_marker_pos}) ===")
    print(f"BP marker token: {context_for_pred[bp_marker_pos]} (expected 260)")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    # Check raw embedding
    x = model.embed(token_ids)
    print(f"\n=== Initial embedding at BP marker ===")
    print(f"EMBED_LO[0:4]: {x[0, bp_marker_pos, BD.EMBED_LO:BD.EMBED_LO+4].tolist()}")
    print(f"EMBED_HI[0:4]: {x[0, bp_marker_pos, BD.EMBED_HI:BD.EMBED_HI+4].tolist()}")
    print(f"MARK_BP: {x[0, bp_marker_pos, BD.MARK_BP].item():.4f}")
    print(f"IS_BYTE: {x[0, bp_marker_pos, BD.IS_BYTE].item():.4f}")

    # Check through each layer
    with torch.no_grad():
        for i in range(6):
            x_before = x.clone()
            x = model.blocks[i](x)

            # Check EMBED_LO changes
            embed_delta = (x - x_before)[0, bp_marker_pos, BD.EMBED_LO:BD.EMBED_LO+4]
            if embed_delta.abs().max() > 0.01:
                print(f"\nL{i}: EMBED_LO delta = {embed_delta.tolist()}")

    print(f"\n=== EMBED_LO at BP marker after L5 (input to L6 FFN) ===")
    print(f"EMBED_LO[0:8]: {x[0, bp_marker_pos, BD.EMBED_LO:BD.EMBED_LO+8].tolist()}")

    # Check what the BP marker should have
    # BP = 0 in draft
    print(f"\n=== Expected BP ===")
    print(f"Draft BP: {draft.bp:#x}")

    # Check if BP_CARRY_LO is being used instead
    print(f"\n=== BP_CARRY_LO at BP marker ===")
    try:
        bp_carry_lo = x[0, bp_marker_pos, BD.BP_CARRY_LO:BD.BP_CARRY_LO+8].tolist()
        print(f"BP_CARRY_LO[0:8]: {bp_carry_lo}")
    except:
        print("BP_CARRY_LO not defined")

    # Check CLEAN_EMBED_LO
    print(f"\n=== CLEAN_EMBED_LO at BP marker ===")
    clean_embed_lo = x[0, bp_marker_pos, BD.CLEAN_EMBED_LO:BD.CLEAN_EMBED_LO+8].tolist()
    print(f"CLEAN_EMBED_LO[0:8]: {clean_embed_lo}")

if __name__ == "__main__":
    main()
