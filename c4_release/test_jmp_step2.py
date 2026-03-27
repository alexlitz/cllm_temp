"""Test if JMP works on step 2 (after NOP)."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Test: NOP, JMP 16 - does JMP work on step 2?
bytecode = [Opcode.NOP, Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)

# Step 1: NOP
draft_vm.step()
draft1 = draft_vm.draft_tokens()
print(f"Step 1 (NOP): PC should be 8")
print(f"  Draft tokens: PC={draft1[1]} (expected 8)")

# Step 2: JMP 16
context_step1 = context + draft1
draft_vm.step()
draft2 = draft_vm.draft_tokens()
print(f"\nStep 2 (JMP 16): PC should be 16")
print(f"  Draft tokens: PC={draft2[1]} (expected 16)")

# Test transformer prediction for step 2
context_step2 = context_step1 + draft2
ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len = len(context_step1)

with torch.no_grad():
    logits = model.forward(ctx_tensor)
    predicted_pc = logits[0, ctx_len, :].argmax().item()
    print(f"  Transformer prediction: PC={predicted_pc}")
    print(f"  Match: {'✓' if predicted_pc == 16 else '✗'}")
