#\!/usr/bin/env python3
import torch, sys
sys.path.insert(0, ".")
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.JMP | (16 << 8)]

def build_context(bytecode, data=b""):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context = build_context(BYTECODE)
draft = DraftVM(BYTECODE)
draft.step()
step1_tokens = draft.draft_tokens()

print(f"Step 1 expected: {step1_tokens[:10]}")
print(f"Draft PC: {draft.pc}")

with torch.no_grad():
    for i in range(5):
        full = context + step1_tokens[:i]
        ids = torch.tensor([full], dtype=torch.long)
        logits = model(ids)
        pred = logits[0, -1, :].argmax().item()
        exp = step1_tokens[i]
        m = "OK" if pred == exp else f"FAIL({pred})"
        print(f"S1T{i}: exp={exp} pred={pred} {m}")
