"""Check PC value in EMBED at AX marker."""
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

ax_marker_pos = len(input_tokens) - 1

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 3 (before L4 FFN which computes PC+1)
    for i in range(4):
        x = model.blocks[i](x)

    print("Before Layer 4 FFN, AX marker EMBED (should be PC=2):")
    print("EMBED_LO:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.EMBED_LO + k].item()
        if abs(val) > 0.01:
            print(f"  EMBED_LO[{k}] = {val:.4f}")
    print()
    print("EMBED_HI:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.EMBED_HI + k].item()
        if abs(val) > 0.01:
            print(f"  EMBED_HI[{k}] = {val:.4f}")
    print()
    print("Expected for PC=2:")
    print("  EMBED_LO[2] = 1.0 (lo nibble)")
    print("  EMBED_HI[0] = 1.0 (hi nibble)")
