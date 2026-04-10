#!/usr/bin/env python3
"""Directly compute blocks[6].ffn output for ADD 10+32."""
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
    print(f"Direct FFN computation at AX marker for ADD")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        for i in range(6):
            x = model.blocks[i](x)

        attn_out = model.blocks[6].attn(x)
        x_after_attn = x + attn_out

        ffn6 = model.blocks[6].ffn

        # Compute FFN manually
        x_ax = x_after_attn[0, ax_marker_pos, :]

        # Up projection + SiLU
        up_proj = ffn6.W_up @ x_ax + ffn6.b_up
        up_silu = up_proj * torch.sigmoid(up_proj)

        # Gate projection + Sigmoid
        gate_proj = ffn6.W_gate @ x_ax + ffn6.b_gate
        gate_sig = torch.sigmoid(gate_proj)

        # Combined activation
        hidden = up_silu * gate_sig

        # Down projection
        ffn_out = ffn6.W_down @ hidden

        print(f"\nFFN output at OUTPUT_LO dims:")
        for k in range(16):
            v = ffn_out[BD.OUTPUT_LO + k].item()
            if abs(v) > 0.1:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

        # Also check model's actual FFN output
        actual_ffn_out = ffn6(x_after_attn)
        print(f"\nActual FFN output (via model) at OUTPUT_LO:")
        for k in range(16):
            v = actual_ffn_out[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.1:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

        # What's the OUTPUT_LO[1] value before and after blocks[6]?
        print(f"\nOUTPUT_LO[1] before blocks[6] attn: {x[0, ax_marker_pos, BD.OUTPUT_LO + 1].item():.4f}")
        print(f"OUTPUT_LO[1] after blocks[6] attn: {x_after_attn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item():.4f}")
        x_after_ffn = x_after_attn + actual_ffn_out
        print(f"OUTPUT_LO[1] after blocks[6] FFN: {x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + 1].item():.4f}")

if __name__ == "__main__":
    main()
