"""Debug all FETCH values."""
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

# Teacher forcing up to AX marker
input_tokens = context + expected_tokens[0:6]

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))
    pos = len(input_tokens) - 1

    # Run through Layer 5
    for i in range(6):
        x = model.blocks[i](x)

    print("FETCH values at AX marker position:")
    print()
    print("FETCH_LO:")
    for k in range(16):
        val = x[0, pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k:2d}] = {val:.4f}")
    print()
    print("FETCH_HI:")
    for k in range(16):
        val = x[0, pos, BD.FETCH_HI + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_HI[{k:2d}] = {val:.4f}")
    print()
    print(f"Expected: immediate = 8 (0x08)")
    print(f"  Low nibble = 8, high nibble = 0")
