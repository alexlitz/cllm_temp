"""Debug JMP 12 behavior."""
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

# JMP 12
bytecode = [Opcode.JMP | (12 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

print("DraftVM behavior with JMP 12:")
print("="*70)
print(f"Initial state: PC={draft_vm.pc}, idx={draft_vm.idx}")

draft_vm.step()
print(f"After JMP 12:  PC={draft_vm.pc}, idx={draft_vm.idx}")
print(f"  PC bytes: {draft_vm.draft_tokens()[1:5]}")
print()
print("Note: JMP 12 sets PC=12 directly (not PC=10)")
print("      idx = (12-2)//8 = 1")
print()

# Neural VM
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

step1_tokens = draft_vm.draft_tokens()
context_step1 = context + step1_tokens
ctx_tensor = torch.tensor([context_step1], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
pc_byte0_pred = logits[0, ctx_len, :].argmax().item()

print("Neural VM prediction:")
print("="*70)
print(f"DraftVM outputs: PC byte 0 = {step1_tokens[1]} (PC=12)")
print(f"Neural VM predicts: PC byte 0 = {pc_byte0_pred}")
print()

if pc_byte0_pred == 10:
    print("Neural VM predicts PC=10 (old PC + 8)")
    print("This is the one-step delay: step 1 outputs old PC+8, step 2 would output JMP target")
elif pc_byte0_pred == 12:
    print("Neural VM predicts PC=12 (matches DraftVM, no delay)")
else:
    print(f"Neural VM predicts PC={pc_byte0_pred} (unexpected)")

print()
print("Expected behavior:")
print("  Step 1: Neural outputs PC=10 (old PC=2 + 8), DraftVM outputs PC=12 (JMP target)")
print("  Step 2: Neural would output PC=12 (JMP relay fires)")
