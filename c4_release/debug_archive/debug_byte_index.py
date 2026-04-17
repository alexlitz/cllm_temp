#\!/usr/bin/env python3
"""Debug BYTE_INDEX at PC byte positions."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

bytecode = [Opcode.JMP | (16 << 8)]

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4): tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(3): tokens.append(0)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
tokens = draft.draft_tokens()

# Full context with step 1 tokens
ctx = context + tokens

print(f"Checking BYTE_INDEX at PC byte positions (first step)")
print(f"PC marker at pos {len(context)} (token 257)")

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    
    # Check PC marker and bytes (positions len(context) to len(context)+5)
    pc_marker_pos = len(context)
    for offset in range(5):
        pos = pc_marker_pos + offset
        tok = ctx[pos]
        bi0 = x[0, pos, BD.BYTE_INDEX_0].item()
        bi1 = x[0, pos, BD.BYTE_INDEX_1].item()
        bi2 = x[0, pos, BD.BYTE_INDEX_2].item()
        bi3 = x[0, pos, BD.BYTE_INDEX_3].item()
        is_byte = x[0, pos, BD.IS_BYTE].item()
        mark_pc = x[0, pos, BD.MARK_PC].item()
        print(f"pos {pos} (tok={tok:3d}): IS_BYTE={is_byte:.2f}, MARK_PC={mark_pc:.2f}, " +
              f"BI0={bi0:.2f}, BI1={bi1:.2f}, BI2={bi2:.2f}, BI3={bi3:.2f}")
