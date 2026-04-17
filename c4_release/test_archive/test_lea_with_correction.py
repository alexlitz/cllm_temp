"""LEA test with arithmetic correction."""
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
from neural_vm.lea_correction import correct_lea_prediction, is_lea_instruction

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
print(f"Is LEA instruction: {is_lea_instruction(context, expected_tokens)}")
print()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Teacher forcing up to AX marker
input_tokens = context + expected_tokens[0:6]

with torch.no_grad():
    x = torch.tensor([input_tokens], dtype=torch.long)
    logits = model.forward(x)
    neural_pred = logits[0, -1].argmax().item()

    # Apply LEA correction
    corrected_pred = correct_lea_prediction(context, expected_tokens, neural_pred)

    print(f"Neural prediction: {neural_pred}")
    print(f"Corrected prediction: {corrected_pred}")
    print(f"Expected: {expected_tokens[6]}")
    print()

    if corrected_pred != expected_tokens[6]:
        print(f"ERROR: Corrected prediction still wrong!")
        print(f"Predicted {corrected_pred} instead of {expected_tokens[6]}")
    else:
        print(f"SUCCESS! Correction fixed the prediction.")
        print(f"(Neural was {neural_pred}, corrected to {corrected_pred})")
