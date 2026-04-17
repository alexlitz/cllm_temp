#!/usr/bin/env python3
"""Correctly trace OUTPUT_LO through all layers for ADD 10+32."""
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
    print(f"Correct trace of OUTPUT_LO at AX marker (pos {ax_marker_pos}) for ADD 10+32")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        print(f"After embed: OUTPUT_LO = {[x[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16) if abs(x[0, ax_marker_pos, BD.OUTPUT_LO + k].item()) > 0.01]}")

        for i in range(16):
            x_before = x.clone()
            # blocks[i] includes both attn and ffn with residual connections
            x = model.blocks[i](x)

            delta_vals = []
            for k in range(16):
                delta = x[0, ax_marker_pos, BD.OUTPUT_LO + k].item() - x_before[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
                if abs(delta) > 0.01:
                    delta_vals.append((k, delta))

            if delta_vals:
                print(f"blocks[{i:2d}]: OUTPUT_LO deltas = {delta_vals}")

        print(f"\nFinal OUTPUT_LO at AX marker:")
        for k in range(16):
            v = x[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.5:
                print(f"  OUTPUT_LO[{k:2d}] = {v:.4f}")

        # Get predictions
        logits = model(token_ids)
        print(f"\nTop 5 predictions for AX byte 0:")
        top5 = torch.topk(logits[0, -1, :], 5)
        for val, idx in zip(top5.values, top5.indices):
            print(f"  token {idx.item()}: logit {val.item():.2f}")

        print(f"\nExpected: token 42 (10 + 32)")

if __name__ == "__main__":
    main()
