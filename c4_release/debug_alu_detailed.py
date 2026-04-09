#!/usr/bin/env python3
"""Debug ALU_LO values - trace attention vs FFN in L7-L8."""
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
            if i in [6, 7]:  # L7, L8 (0-indexed)
                # Trace attention (attention has internal residual)
                x_pre_attn = x.clone()
                x_post_attn = block.attn(x)

                alu6_attn = x_post_attn[0, ax_marker_pos, BD.ALU_LO + 6].item()
                alu7_attn = x_post_attn[0, ax_marker_pos, BD.ALU_LO + 7].item()
                alu0_attn = x_post_attn[0, ax_marker_pos, BD.ALU_LO + 0].item()

                delta6_attn = alu6_attn - x_pre_attn[0, ax_marker_pos, BD.ALU_LO + 6].item()
                delta7_attn = alu7_attn - x_pre_attn[0, ax_marker_pos, BD.ALU_LO + 7].item()
                delta0_attn = alu0_attn - x_pre_attn[0, ax_marker_pos, BD.ALU_LO + 0].item()

                print(f"L{i+1} Attn: ALU_LO[6]={alu6_attn:.2f} (Δ={delta6_attn:+.2f}), " +
                      f"ALU_LO[7]={alu7_attn:.2f} (Δ={delta7_attn:+.2f}), " +
                      f"ALU_LO[0]={alu0_attn:.2f} (Δ={delta0_attn:+.2f})")

                # Trace FFN (FFN has internal residual)
                x = block.ffn(x_post_attn)

                alu6_ffn = x[0, ax_marker_pos, BD.ALU_LO + 6].item()
                alu7_ffn = x[0, ax_marker_pos, BD.ALU_LO + 7].item()
                alu0_ffn = x[0, ax_marker_pos, BD.ALU_LO + 0].item()

                delta6_ffn = alu6_ffn - alu6_attn
                delta7_ffn = alu7_ffn - alu7_attn
                delta0_ffn = alu0_ffn - alu0_attn

                print(f"L{i+1} FFN:  ALU_LO[6]={alu6_ffn:.2f} (Δ={delta6_ffn:+.2f}), " +
                      f"ALU_LO[7]={alu7_ffn:.2f} (Δ={delta7_ffn:+.2f}), " +
                      f"ALU_LO[0]={alu0_ffn:.2f} (Δ={delta0_ffn:+.2f})")

                # Apply post_ops
                for op in block.post_ops:
                    x = op(x)
            else:
                x = block(x)

    print("\n" + "=" * 60)
    print("Final ALU_LO values at AX marker:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        if abs(val) > 0.1:
            print(f"  ALU_LO[{k}] = {val:.2f}")

    # Also check at STACK0 positions where ALU should NOT be written
    print("\nChecking STACK0 positions for ALU contamination:")
    for off in range(5):
        stack0_pos = ax_marker_pos + 15 + off  # STACK0 is 15 tokens after AX marker
        if stack0_pos < token_ids.shape[1]:
            for k in [6, 7]:
                val = x[0, stack0_pos, BD.ALU_LO + k].item()
                if abs(val) > 0.1:
                    print(f"  STACK0 pos {off}: ALU_LO[{k}] = {val:.2f}")

if __name__ == "__main__":
    main()
