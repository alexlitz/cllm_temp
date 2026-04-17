#!/usr/bin/env python3
"""Debug L3 head 3 K vectors at attended positions."""

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

    print(f"=== L3 Head 3 K vectors at various positions ===")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)

    # Run through L0-L2 (L3 head 3 reads after L2)
    with torch.no_grad():
        for i in range(3):
            x = model.blocks[i](x)

    BP_I = 3
    L = 15.0
    HD = 64

    attn = model.blocks[3].attn

    # Print K weights
    base = 3 * HD  # Head 3
    print(f"\nL3 Head 3 K weight config:")
    print(f"  K[0] weights: L1H1[{BP_I}] = {attn.W_k[base, BD.L1H1 + BP_I].item():.1f}, L1H0[{BP_I}] = {attn.W_k[base, BD.L1H0 + BP_I].item():.1f}")
    print(f"  K[33] weights: CONST = {attn.W_k[base + 33, BD.CONST].item():.1f}")

    # Check L1H1[BP_I] and L1H0[BP_I] at various positions
    print(f"\n=== L1H1/L1H0 values at key positions ===")
    positions_to_check = [60, 95, 130, 50, 85, 120]  # MEM addr byte 0 and BP byte 0
    for pos in positions_to_check:
        if pos >= len(context_for_pred):
            continue
        l1h1_bp = x[0, pos, BD.L1H1 + BP_I].item()
        l1h0_bp = x[0, pos, BD.L1H0 + BP_I].item()
        tok = context_for_pred[pos]
        print(f"  Pos {pos} (tok={tok}): L1H1[BP]={l1h1_bp:.4f}, L1H0[BP]={l1h0_bp:.4f}")

    # Also check what the step structure looks like
    print(f"\n=== Step 3 structure (starts at pos 104) ===")
    step3_start = 104
    for offset, name in enumerate(['PC', 'PC_b0', 'PC_b1', 'PC_b2', 'PC_b3',
                                    'AX', 'AX_b0', 'AX_b1', 'AX_b2', 'AX_b3',
                                    'SP', 'SP_b0', 'SP_b1', 'SP_b2', 'SP_b3',
                                    'BP', 'BP_b0', 'BP_b1', 'BP_b2', 'BP_b3',
                                    'STACK0', 'STACK0_b0', 'STACK0_b1', 'STACK0_b2', 'STACK0_b3',
                                    'MEM', 'MEM_a0', 'MEM_a1', 'MEM_a2', 'MEM_a3',
                                    'MEM_v0', 'MEM_v1', 'MEM_v2', 'MEM_v3', 'SE']):
        pos = step3_start + offset
        if pos >= len(context_for_pred):
            continue
        tok = context_for_pred[pos]
        l1h1_bp = x[0, pos, BD.L1H1 + BP_I].item()
        l1h0_bp = x[0, pos, BD.L1H0 + BP_I].item()
        if l1h1_bp > 0.5:
            print(f"  Pos {pos} ({name}): tok={tok}, L1H1[BP]={l1h1_bp:.2f}, L1H0[BP]={l1h0_bp:.2f}")

    # Check BP byte 0 positions specifically
    print(f"\n=== BP byte 0 positions (should match) ===")
    for step, step_start in enumerate([34, 69, 104, 139], 1):
        bp_byte0_pos = step_start + 16  # BP marker at 15, byte 0 at 16
        if bp_byte0_pos >= len(context_for_pred):
            continue
        tok = context_for_pred[bp_byte0_pos]
        l1h1_bp = x[0, bp_byte0_pos, BD.L1H1 + BP_I].item()
        l1h0_bp = x[0, bp_byte0_pos, BD.L1H0 + BP_I].item()
        print(f"  Step {step} BP byte 0 (pos {bp_byte0_pos}): tok={tok}, L1H1[BP]={l1h1_bp:.4f}, L1H0[BP]={l1h0_bp:.4f}")

if __name__ == "__main__":
    main()
