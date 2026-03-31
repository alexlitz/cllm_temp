"""Simple LEA test."""
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

print(f"Expected AX after LEA 8: 0x{draft.ax:08x} = {draft.ax}")
print(f"Expected AX_b0: {expected_tokens[6]}")
print()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Teacher forcing up to AX marker
input_tokens = context + expected_tokens[0:6]

with torch.no_grad():
    x = torch.tensor([input_tokens], dtype=torch.long)
    logits = model.forward(x)
    pred = logits[0, -1].argmax().item()

    print(f"Predicted AX_b0: {pred}")
    print(f"Expected AX_b0: {expected_tokens[6]}")

    if pred != expected_tokens[6]:
        print(f"\nERROR: Mismatch!")
        print(f"Predicted {pred} instead of {expected_tokens[6]}")
    else:
        print(f"\nSUCCESS!")
