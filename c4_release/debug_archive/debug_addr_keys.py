"""Check ADDR_KEY values in the context."""
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

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

with torch.no_grad():
    x = model.embed(torch.tensor([context], dtype=torch.long))

    print("ADDR_KEY at context positions:")
    for pos in range(min(6, len(context))):
        addr_key_lo = []
        addr_key_hi = []
        for k in range(16):
            val = x[0, pos, BD.ADDR_KEY + k].item()
            if abs(val) > 0.01:
                addr_key_lo.append(k)
        for k in range(16):
            val = x[0, pos, BD.ADDR_KEY + 16 + k].item()
            if abs(val) > 0.01:
                addr_key_hi.append(k)
        print(f"  Pos {pos} (token {context[pos]:3d}): LO={addr_key_lo}, HI={addr_key_hi}")
