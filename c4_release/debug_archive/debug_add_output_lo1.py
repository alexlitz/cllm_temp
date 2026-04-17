#!/usr/bin/env python3
"""Trace OUTPUT_LO[1] = 15.0 source layer by layer for ADD 10+32."""
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
    print(f"Tracing OUTPUT_LO[1] layer by layer at AX marker (pos {ax_marker_pos})")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        print(f"After embed: OUTPUT_LO[1] = {x[0, ax_marker_pos, BD.OUTPUT_LO + 1].item():.4f}")

        for i in range(16):
            x_before = x.clone()
            x = model.blocks[i](x)
            delta = x[0, ax_marker_pos, BD.OUTPUT_LO + 1].item() - x_before[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()
            val = x[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()
            if abs(delta) > 0.01:
                print(f"After blocks[{i:2d}]: OUTPUT_LO[1] = {val:.4f} (Δ={delta:+.4f})")

        # Break down blocks with large delta
        print("\n" + "=" * 80)
        print("Breaking down layers with significant OUTPUT_LO[1] changes")
        print("=" * 80)

        for layer_idx in [6, 8, 9, 10]:  # common ALU layers
            x = model.embed(token_ids)
            for i in range(layer_idx):
                x = model.blocks[i](x)

            x_before_attn = x.clone()
            attn_out = model.blocks[layer_idx].attn(x)
            x_after_attn = x + attn_out
            val_after_attn = x_after_attn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()
            delta_attn = val_after_attn - x_before_attn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()

            ffn_out = model.blocks[layer_idx].ffn(x_after_attn)
            x_after_ffn = x_after_attn + ffn_out
            val_after_ffn = x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item()
            delta_ffn = val_after_ffn - val_after_attn

            if abs(delta_attn) > 0.01 or abs(delta_ffn) > 0.01:
                print(f"\nblocks[{layer_idx}]:")
                if abs(delta_attn) > 0.01:
                    print(f"  attn: OUTPUT_LO[1] = {val_after_attn:.4f} (Δ={delta_attn:+.4f})")
                if abs(delta_ffn) > 0.01:
                    print(f"  FFN:  OUTPUT_LO[1] = {val_after_ffn:.4f} (Δ={delta_ffn:+.4f})")

if __name__ == "__main__":
    main()
