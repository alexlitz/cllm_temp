#!/usr/bin/env python3
"""Trace OUTPUT_LO[1] through all layers sequentially for ADD 10+32."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (32 << 8), Opcode.ADD, Opcode.EXIT]

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
        context.extend(draft.draft_tokens())

    draft.step()
    step4_tokens = draft.draft_tokens()
    context_for_ax = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_ax) - 1

    token_ids = torch.tensor([context_for_ax], dtype=torch.long)

    print("=" * 80)
    print(f"Sequential trace of OUTPUT_LO[1] at AX marker (pos {ax_marker_pos})")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        print(f"After embed: OUTPUT_LO[1] = {x[0, ax_marker_pos, BD.OUTPUT_LO + 1].item():.4f}")

        for i in range(16):
            # Attention
            attn_out = model.blocks[i].attn(x)
            x_after_attn = x + attn_out
            delta_attn = x_after_attn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item() - x[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()

            # FFN
            ffn_out = model.blocks[i].ffn(x_after_attn)
            x_after_ffn = x_after_attn + ffn_out
            delta_ffn = x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item() - x_after_attn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()

            val = x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()

            if abs(delta_attn) > 0.1 or abs(delta_ffn) > 0.1:
                print(f"blocks[{i:2d}]: attn Δ={delta_attn:+8.4f}, FFN Δ={delta_ffn:+8.4f}, total={val:.4f}")

            x = x_after_ffn

        print(f"\nFinal OUTPUT_LO[1] = {x[0, ax_marker_pos, BD.OUTPUT_LO + 1].item():.4f}")

        # Also show all OUTPUT_LO values
        print(f"\nAll OUTPUT_LO values at AX marker:")
        for k in range(16):
            v = x[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.5:
                print(f"  OUTPUT_LO[{k:2d}] = {v:.4f}")

if __name__ == "__main__":
    main()
