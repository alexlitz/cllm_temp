"""Debug LEA correction parsing."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
torch.set_num_threads(1)
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH

bytecode = [Opcode.LEA | (8 << 8), Opcode.EXIT]

def build_context(bc):
    tokens = [Token.CODE_START]
    for instr in bc:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
expected_tokens = draft.draft_tokens()

print("Context tokens:")
for i, tok in enumerate(context):
    if tok == Token.CODE_START:
        print(f"  [{i}] CODE_START")
    elif tok == Token.CODE_END:
        print(f"  [{i}] CODE_END")
    elif tok == Token.DATA_START:
        print(f"  [{i}] DATA_START")
    elif tok == Token.DATA_END:
        print(f"  [{i}] DATA_END")
    else:
        print(f"  [{i}] {tok}")

print(f"\nDraft tokens (first 10):")
for i, tok in enumerate(expected_tokens[:10]):
    print(f"  [{i}] {tok}")

print(f"\nPC from draft tokens:")
pc_bytes = expected_tokens[1:5]
pc = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)
print(f"  PC = {pc} (bytes: {pc_bytes})")
print(f"  PC_OFFSET = {PC_OFFSET}")
print(f"  INSTR_WIDTH = {INSTR_WIDTH}")

if pc >= PC_OFFSET:
    idx = (pc - PC_OFFSET) // INSTR_WIDTH
    print(f"  Instruction index = {idx}")

print(f"\nBP from draft tokens:")
bp_bytes = expected_tokens[16:20]
bp = bp_bytes[0] | (bp_bytes[1] << 8) | (bp_bytes[2] << 16) | (bp_bytes[3] << 24)
print(f"  BP = {bp} (0x{bp:08x})")

print(f"\nFind CODE_START:")
code_start_idx = None
for i in range(len(context) - 1, -1, -1):
    if context[i] == Token.CODE_START:
        code_start_idx = i
        print(f"  Found at index {i}")
        break

if code_start_idx is not None and pc >= PC_OFFSET:
    idx = (pc - PC_OFFSET) // INSTR_WIDTH
    instr_offset = code_start_idx + 1 + idx * 5
    print(f"\nInstruction parsing:")
    print(f"  Instruction offset in context = {instr_offset}")
    print(f"  Context length = {len(context)}")

    if instr_offset + 5 <= len(context):
        opcode = context[instr_offset]
        imm_bytes = context[instr_offset + 1:instr_offset + 5]
        immediate = (imm_bytes[0] |
                     (imm_bytes[1] << 8) |
                     (imm_bytes[2] << 16) |
                     (imm_bytes[3] << 24))

        print(f"  Opcode = {opcode}")
        print(f"  Immediate bytes = {imm_bytes}")
        print(f"  Immediate value = {immediate}")

        if opcode == 0:
            correct_ax = (bp + immediate) & 0xFFFFFFFF
            correct_ax_byte0 = correct_ax & 0xFF
            print(f"\n  LEA detected!")
            print(f"  BP + immediate = {bp} + {immediate} = {correct_ax} (0x{correct_ax:08x})")
            print(f"  Correct AX byte 0 = {correct_ax_byte0}")
    else:
        print(f"  ERROR: Instruction offset {instr_offset} + 5 > context length {len(context)}")
