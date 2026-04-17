#!/usr/bin/env python3
"""Debug ALU_LO values layer by layer."""
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

    # Run 3 steps to build context
    for step in range(3):
        draft.step()
        tokens = draft.draft_tokens()
        context.extend(tokens)

    # Step 4 - MUL execution
    draft.step()
    step4_tokens = draft.draft_tokens()

    # Build context up to AX marker
    context_for_pred = context + step4_tokens[:6]  # PC section + AX marker
    ax_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    print(f"Tracing ALU_LO at AX marker (pos {ax_marker_pos}):")
    print("=" * 60)

    with torch.no_grad():
        x = model.embed(token_ids)

        for i, block in enumerate(model.blocks):
            x_prev = x.clone()
            x = block(x)

            # Check ALU_LO[6] and ALU_LO[7] after each layer
            alu_lo_6 = x[0, ax_marker_pos, BD.ALU_LO + 6].item()
            alu_lo_7 = x[0, ax_marker_pos, BD.ALU_LO + 7].item()
            alu_lo_0 = x[0, ax_marker_pos, BD.ALU_LO + 0].item()

            delta_6 = alu_lo_6 - x_prev[0, ax_marker_pos, BD.ALU_LO + 6].item()
            delta_7 = alu_lo_7 - x_prev[0, ax_marker_pos, BD.ALU_LO + 7].item()
            delta_0 = alu_lo_0 - x_prev[0, ax_marker_pos, BD.ALU_LO + 0].item()

            # Always print L7-L8 or if significant change
            if i in [6, 7] or abs(delta_6) > 0.01 or abs(delta_7) > 0.01 or abs(delta_0) > 0.01:
                print(f"L{i+1}: ALU_LO[6]={alu_lo_6:.2f} (Δ={delta_6:+.2f}), " +
                      f"ALU_LO[7]={alu_lo_7:.2f} (Δ={delta_7:+.2f}), " +
                      f"ALU_LO[0]={alu_lo_0:.2f} (Δ={delta_0:+.2f})")

    print("\n" + "=" * 60)
    print("Final ALU_LO values:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        if abs(val) > 0.1:
            print(f"  ALU_LO[{k}] = {val:.2f}")

if __name__ == "__main__":
    main()
