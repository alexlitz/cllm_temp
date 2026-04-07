"""Check FETCH at Layer 8 input."""
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

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 7 (before Layer 8)
    for i in range(8):
        x = model.blocks[i](x)

    print("At Layer 8 input, AX marker:")
    print("\nFETCH_LO:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k}] = {val:.4f}")

    print("\nFETCH_HI:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.FETCH_HI + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_HI[{k}] = {val:.4f}")

    print("\nALU_LO (BP low nibble):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_LO + k].item()
        if abs(val) > 0.01:
            print(f"  ALU_LO[{k}] = {val:.4f}")

    print("\nALU_HI (BP high nibble):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.ALU_HI + k].item()
        if abs(val) > 0.01:
            print(f"  ALU_HI[{k}] = {val:.4f}")

    print("\nExpected: FETCH_LO[8]=1.0, FETCH_HI[0]=1.0 (immediate=8)")
    print("Expected: ALU_LO[0]=1.0, ALU_HI[1]=1.0 (BP=0x00010000=65536)")
