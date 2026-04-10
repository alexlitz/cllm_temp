#!/usr/bin/env python3
"""Debug blocks[10] (L11) for ADD 10+32 - trace attn vs FFN vs post_ops."""
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
    print(f"Debug blocks[10] (L11) for ADD 10+32 at AX marker (pos {ax_marker_pos})")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        for i in range(10):
            x = model.blocks[i](x)

        print(f"\nBefore blocks[10]:")
        for k in range(16):
            v = x[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

        # Run attention only
        x_before = x.clone()
        x_after_attn = model.blocks[10].attn(x)
        print(f"\nAfter blocks[10].attn:")
        for k in range(16):
            v = x_after_attn[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            delta = v - x_before[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(delta) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f} (Δ={delta:+.4f})")

        # Run FFN
        x_after_ffn = model.blocks[10].ffn(x_after_attn)
        print(f"\nAfter blocks[10].ffn:")
        for k in range(16):
            v = x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            delta = v - x_after_attn[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(delta) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f} (Δ={delta:+.4f})")

        # Run post_ops
        x_after_ops = x_after_ffn.clone()
        for i, op in enumerate(model.blocks[10].post_ops):
            x_after_ops = op(x_after_ops)
            print(f"\nAfter blocks[10].post_ops[{i}] ({type(op).__name__}):")
            for k in range(16):
                v = x_after_ops[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
                delta = v - x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
                if abs(delta) > 0.01:
                    print(f"  OUTPUT_LO[{k}]: {v:.4f} (Δ={delta:+.4f})")

        print(f"\nFinal after blocks[10]:")
        for k in range(16):
            v = x_after_ops[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.5:
                print(f"  OUTPUT_LO[{k}] = {v:.4f}")

if __name__ == "__main__":
    main()
