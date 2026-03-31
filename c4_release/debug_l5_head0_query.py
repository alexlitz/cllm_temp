"""Debug what address Layer 5 head 0 queries for."""
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

    # Run through Layer 4
    for i in range(5):
        x = model.blocks[i](x)

    print("After Layer 4, AX marker TEMP (PC+1 query address):")
    print("TEMP_LO (address low nibble):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.TEMP + k].item()
        if abs(val) > 0.01:
            print(f"  TEMP[{k}] = {val:.4f} (address nibble {k})")
    print()
    print("TEMP_HI (address high nibble):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.TEMP + 16 + k].item()
        if abs(val) > 0.01:
            print(f"  TEMP[{16+k}] = {val:.4f} (address nibble {k})")
    print()
    print("Expected for first step:")
    print("  TEMP should encode address 1 (PC=0, PC+1=1)")
    print("  But bytecode immediate is at address 3 (PC_OFFSET+1)")
    print("  This mismatch causes head 0 to fetch wrong byte!")
