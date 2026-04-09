#!/usr/bin/env python3
"""Debug MARK_BP at various positions."""

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
    target_token = 16
    context_for_pred = context + step4_tokens[:target_token]
    bp_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    # Check raw embedding (before any layers)
    x = model.embed(token_ids)

    print(f"=== Token 260 (BP marker) vs Token 261 (MEM marker) ===")
    print(f"Token.REG_BP = {Token.REG_BP}")
    print(f"Token.MEM = {Token.MEM}")

    print(f"\n=== MARK_BP in initial embedding ===")
    # Find positions with MARK_BP set
    mark_bp_vals = x[0, :, BD.MARK_BP].tolist()

    # Show all positions with MARK_BP > 0.5
    print("Positions with MARK_BP > 0.5:")
    for pos, val in enumerate(mark_bp_vals):
        if val > 0.5:
            tok = context_for_pred[pos]
            print(f"  Pos {pos}: tok={tok}, MARK_BP={val:.4f}")

    # Check MEM marker positions specifically
    print(f"\n=== MEM marker positions (offset 25 from step start) ===")
    for step, step_start in enumerate([34, 69, 104, 139], 1):
        mem_marker_pos = step_start + 25
        if mem_marker_pos >= len(context_for_pred):
            continue
        tok = context_for_pred[mem_marker_pos]
        mark_bp = x[0, mem_marker_pos, BD.MARK_BP].item()
        mark_mem = x[0, mem_marker_pos, BD.MARK_MEM].item()
        print(f"  Step {step} MEM marker (pos {mem_marker_pos}): tok={tok}, MARK_BP={mark_bp:.4f}, MARK_MEM={mark_mem:.4f}")

    # Check actual BP marker positions (offset 15 from step start)
    print(f"\n=== BP marker positions (offset 15 from step start) ===")
    for step, step_start in enumerate([34, 69, 104, 139], 1):
        bp_marker_pos_in_step = step_start + 15
        if bp_marker_pos_in_step >= len(context_for_pred):
            continue
        tok = context_for_pred[bp_marker_pos_in_step]
        mark_bp = x[0, bp_marker_pos_in_step, BD.MARK_BP].item()
        print(f"  Step {step} BP marker (pos {bp_marker_pos_in_step}): tok={tok}, MARK_BP={mark_bp:.4f}")

    # Check if MEM marker token (261) has MARK_BP set in embedding
    print(f"\n=== Embedding for MEM marker token (261) ===")
    # Create a single-token context with just the MEM marker
    test_context = [Token.MEM]
    test_x = model.embed(torch.tensor([test_context], dtype=torch.long))
    print(f"Token 261 MARK_BP: {test_x[0, 0, BD.MARK_BP].item():.4f}")
    print(f"Token 261 MARK_MEM: {test_x[0, 0, BD.MARK_MEM].item():.4f}")

    # What if the MEM marker has the wrong token value in the draft?
    print(f"\n=== Checking step 3 token sequence ===")
    step3_start = 104
    for offset in range(35):
        pos = step3_start + offset
        if pos >= len(context_for_pred):
            break
        tok = context_for_pred[pos]
        if tok >= 256:  # Only show marker tokens
            print(f"  Offset {offset} (pos {pos}): tok={tok}")

if __name__ == "__main__":
    main()
