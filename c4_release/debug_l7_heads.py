#!/usr/bin/env python3
"""Debug L7 attention heads - trace Head 0 and Head 1 separately."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.IMM | (6 << 8), Opcode.PSH, Opcode.IMM | (7 << 8), Opcode.MUL, Opcode.EXIT]

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
        tokens = draft.draft_tokens()
        context.extend(tokens)

    draft.step()
    step4_tokens = draft.draft_tokens()
    context_for_pred = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    with torch.no_grad():
        x = model.embed(token_ids)
        # Run through L1-L6
        for i in range(7):
            x = model.blocks[i](x)

    print("At AX marker (pos 154):")
    print(f"  MARK_AX = {x[0, ax_marker_pos, BD.MARK_AX].item():.2f}")
    print(f"  OP_LEA = {x[0, ax_marker_pos, BD.OP_LEA].item():.2f}")
    print(f"  OP_ADJ = {x[0, ax_marker_pos, BD.OP_ADJ].item():.2f}")
    print(f"  OP_MUL = {x[0, ax_marker_pos, BD.OP_MUL].item():.2f}")

    print("\n" + "=" * 60)
    print("MARK_BP positions and their OUTPUT_LO values:")
    for pos in range(token_ids.shape[1]):
        mark_bp = x[0, pos, BD.MARK_BP].item()
        if mark_bp > 0.5:
            output_lo = []
            for k in range(16):
                v = x[0, pos, BD.OUTPUT_LO + k].item()
                if abs(v) > 0.3:
                    output_lo.append((k, v))
            print(f"  pos {pos}: MARK_BP={mark_bp:.2f}, OUTPUT_LO={output_lo}")

    print("\n" + "=" * 60)
    print("MARK_SP positions and their OUTPUT_LO values:")
    for pos in range(token_ids.shape[1]):
        mark_sp = x[0, pos, BD.MARK_SP].item()
        if mark_sp > 0.5:
            output_lo = []
            for k in range(16):
                v = x[0, pos, BD.OUTPUT_LO + k].item()
                if abs(v) > 0.3:
                    output_lo.append((k, v))
            print(f"  pos {pos}: MARK_SP={mark_sp:.2f}, OUTPUT_LO={output_lo}")

    # Check L7 Head 0 and Head 1 separately
    print("\n" + "=" * 60)
    print("L7 Attention analysis (blocks[7]):")

    attn7 = model.blocks[7].attn
    HD = 64

    # Head 0: Q at MARK_AX, K at STACK0_BYTE0
    print("\nHead 0 (STACK0→ALU):")
    q0 = x[0, ax_marker_pos, BD.MARK_AX].item() * 15.0
    op_lea = x[0, ax_marker_pos, BD.OP_LEA].item() * (-15.0)
    q0_total = q0 + op_lea
    print(f"  Q[0] at AX marker: MARK_AX*15 + OP_LEA*(-15) = {q0} + {op_lea} = {q0_total:.2f}")

    # Head 1: Q at OP_LEA/OP_ADJ, K at MARK_BP/MARK_SP
    print("\nHead 1 (LEA/ADJ):")
    q1_lea = x[0, ax_marker_pos, BD.OP_LEA].item() * 15.0
    q1_adj = x[0, ax_marker_pos, BD.OP_ADJ].item() * 15.0
    q1_total = q1_lea + q1_adj
    print(f"  Q[0] at AX marker: OP_LEA*15 + OP_ADJ*15 = {q1_lea} + {q1_adj} = {q1_total:.2f}")

    # If Q[0] ≈ 0, the softmax1 scores will all be negative (from ALiBi), yielding small attention
    print(f"\n  With Q≈0, softmax1 scores = -slope*d, giving near-zero attention")
    print(f"  But if any spurious activation exists, it could leak through")

    # Check what's being copied from BP/SP positions
    print("\n" + "=" * 60)
    print("Checking previous step's BP marker (d=25 from AX marker):")
    prev_bp_pos = ax_marker_pos - 25
    if prev_bp_pos >= 0:
        print(f"  pos {prev_bp_pos}:")
        print(f"    MARK_BP = {x[0, prev_bp_pos, BD.MARK_BP].item():.2f}")
        output_vals = []
        for k in range(16):
            v = x[0, prev_bp_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.1:
                output_vals.append((k, v))
        print(f"    OUTPUT_LO significant: {output_vals}")

if __name__ == "__main__":
    main()
