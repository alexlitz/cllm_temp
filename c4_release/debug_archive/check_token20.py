"""Check token 20 (STACK0) to understand the marker context."""
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

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Context and draft tokens:")
print(f"Context: {context}")
print(f"Token.STACK0 = {Token.STACK0}")
print()

print("Draft tokens:")
for i, tok in enumerate(draft_tokens[:25]):
    token_name = ""
    if i == 0:
        token_name = "REG_PC"
    elif i == 1:
        token_name = "PC_b0"
    elif i == 20:
        token_name = "STACK0"
    elif i == 21:
        token_name = "ST_b0"

    marker = ""
    if tok == Token.STACK0:
        marker = " ← STACK0 marker"

    print(f"Token {i:2d} ({token_name:10s}): {tok:3d}{marker}")
