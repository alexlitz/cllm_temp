#\!/usr/bin/env python3
"""Check address encoding at MEM value byte positions."""

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

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)
    with torch.no_grad():
        for i in range(15):
            x = model.blocks[i](x)

    print(f"=== ADDR_B0_LO dims at key positions ===")
    print(f"Expected: SP address = 0xfff8, so ADDR_B0 = 0xf8")
    print(f"ADDR_B0_LO (nibble 8) should be one-hot at index 8")
    
    for pos in [109, 112, 134]:
        name = {109: "v0", 112: "v3", 134: "STACK0 marker"}[pos]
        addr_b0_lo = x[0, pos, BD.ADDR_B0_LO:BD.ADDR_B0_LO+16].tolist()
        addr_b0_hi = x[0, pos, BD.ADDR_B0_HI:BD.ADDR_B0_HI+16].tolist()
        addr_b1_lo = x[0, pos, BD.ADDR_B1_LO:BD.ADDR_B1_LO+16].tolist()
        addr_b1_hi = x[0, pos, BD.ADDR_B1_HI:BD.ADDR_B1_HI+16].tolist()
        
        print(f"\nPosition {pos} ({name}):")
        print(f"  ADDR_B0_LO max idx: {addr_b0_lo.index(max(addr_b0_lo))} (val={max(addr_b0_lo):.2f})")
        print(f"  ADDR_B0_HI max idx: {addr_b0_hi.index(max(addr_b0_hi))} (val={max(addr_b0_hi):.2f})")
        print(f"  ADDR_B1_LO max idx: {addr_b1_lo.index(max(addr_b1_lo))} (val={max(addr_b1_lo):.2f})")
        print(f"  ADDR_B1_HI max idx: {addr_b1_hi.index(max(addr_b1_hi))} (val={max(addr_b1_hi):.2f})")
        
        # For a more detailed view, print the full vectors
        print(f"  ADDR_B0_LO: {[f'{v:.1f}' for v in addr_b0_lo]}")

if __name__ == "__main__":
    main()
