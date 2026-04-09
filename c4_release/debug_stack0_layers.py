#\!/usr/bin/env python3
"""Debug which layer adds OUTPUT_LO[0] = 1.0 for STACK0 byte 0."""

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
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    step_start = len(context) - 35
    target_token = 21
    context_for_pred = context[:step_start + target_token]
    query_pos = len(context_for_pred) - 1

    print(f"Predicting STACK0 byte 0 from position {query_pos} (STACK0 marker)")
    print(f"Expected: 1, needs OUTPUT_LO[1]=1.0, OUTPUT_LO[0]=0")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)

    print(f"\n=== Per-layer OUTPUT_LO changes ===")
    for layer_idx in range(16):
        x_before = x.clone()
        with torch.no_grad():
            x = model.blocks[layer_idx](x)

        out_lo_0_before = x_before[0, query_pos, BD.OUTPUT_LO].item()
        out_lo_0_after = x[0, query_pos, BD.OUTPUT_LO].item()
        out_lo_1_before = x_before[0, query_pos, BD.OUTPUT_LO+1].item()
        out_lo_1_after = x[0, query_pos, BD.OUTPUT_LO+1].item()

        delta_0 = out_lo_0_after - out_lo_0_before
        delta_1 = out_lo_1_after - out_lo_1_before

        if abs(delta_0) > 0.001 or abs(delta_1) > 0.001:
            print(f"Layer {layer_idx:2d}: OUTPUT_LO[0]: {out_lo_0_before:.4f} -> {out_lo_0_after:.4f} (delta={delta_0:+.4f})")
            print(f"         OUTPUT_LO[1]: {out_lo_1_before:.4f} -> {out_lo_1_after:.4f} (delta={delta_1:+.4f})")

    print(f"\nFinal OUTPUT_LO[0:4]: {x[0, query_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+4].tolist()}")

if __name__ == "__main__":
    main()
