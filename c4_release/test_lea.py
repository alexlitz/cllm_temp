"""Test LEA opcode."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

for opcode_name, bytecode in [('LEA 8', [Opcode.LEA | (8 << 8)]), ('LEA 16', [Opcode.LEA | (16 << 8)])]:
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    print(f'{opcode_name}:')
    for i in range(min(10, len(draft_tokens))):
        ctx = context + draft_tokens[:i]
        with torch.no_grad():
            logits = model.forward(torch.tensor([ctx], dtype=torch.long))
            pred = torch.argmax(logits[0, -1, :]).item()

        match = 'OK' if pred == draft_tokens[i] else 'FAIL'
        print(f'  Token {i}: exp={draft_tokens[i]:3d}, pred={pred:3d} {match}')
        if not (pred == draft_tokens[i]):
            break
    print()
