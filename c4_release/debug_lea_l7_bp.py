"""Check if Layer 7 head 1 copies BP to ALU for LEA."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
torch.set_num_threads(1)
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

BD = _SetDim

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

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

input_tokens = context + expected_tokens[0:6]
ax_marker_pos = len(input_tokens) - 1
bp_marker_pos = ax_marker_pos + 10  # BP marker is 10 positions after AX

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 6
    for i in range(7):
        x = model.blocks[i](x)

    print("Before Layer 7, AX marker:")
    print(f"OP_LEA: {x[0, ax_marker_pos, BD.OP_LEA].item():.4f}")

    print("\nBefore Layer 7, BP marker:")
    print("OUTPUT_LO (BP byte 0 low nibble):")
    for k in range(16):
        val = x[0, bp_marker_pos, BD.OUTPUT_LO + k].item()
        if abs(val) > 0.01:
            print(f"  OUTPUT_LO[{k}] = {val:.4f}")
    print("OUTPUT_HI (BP byte 0 high nibble):")
    for k in range(16):
        val = x[0, bp_marker_pos, BD.OUTPUT_HI + k].item()
        if abs(val) > 0.01:
            print(f"  OUTPUT_HI[{k}] = {val:.4f}")

    # Run Layer 7
    x = model.blocks[7](x)

    print("\nAfter Layer 7, AX marker:")
    print("ALU_LO (should be BP low nibble = 0):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        if abs(val) > 0.01:
            print(f"  ALU_LO[{k}] = {val:.4f}")
    print("ALU_HI (should be BP high nibble = 1 for BP=0x00010000):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_HI + k].item()
        if abs(val) > 0.01:
            print(f"  ALU_HI[{k}] = {val:.4f}")

    print("\nExpected for BP=0x00010000: ALU_LO[0]=1.0, ALU_HI[1]=1.0")
