"""Detailed comparison of DraftVM vs Neural VM predictions."""
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

# Test IMM 5
bytecode = [Opcode.IMM | (5 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

print("DraftVM state before step:")
print(f"  PC: {draft_vm.pc}")
print(f"  AX: {draft_vm.ax}")
print()

draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("DraftVM state after IMM 5:")
print(f"  PC: {draft_vm.pc}")
print(f"  AX: {draft_vm.ax}")
print()

print("DraftVM draft tokens (35 tokens):")
print(f"  PC marker: {draft_tokens[0]}")
print(f"  PC bytes:  {draft_tokens[1:5]}")
print(f"  AX marker: {draft_tokens[5]}")
print(f"  AX bytes:  {draft_tokens[6:10]}")
print(f"  SP marker: {draft_tokens[10]}")
print(f"  SP bytes:  {draft_tokens[11:15]}")
print()

# Load model and predict
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
print("Neural VM predictions:")
print(f"  Context length: {ctx_len}")
print()

# Predict each token
for i in range(10):  # Just first 10 tokens
    pos = ctx_len + i
    expected = draft_tokens[i]
    predicted = logits[0, pos - 1, :].argmax(-1).item()

    marker_names = {257: "REG_PC", 258: "REG_AX", 259: "REG_SP",
                    260: "REG_BP", 261: "STACK0", 262: "MEM"}

    if i == 0:
        token_desc = "PC marker"
    elif 1 <= i <= 4:
        token_desc = f"PC byte {i-1}"
    elif i == 5:
        token_desc = "AX marker"
    elif 6 <= i <= 9:
        token_desc = f"AX byte {i-6}"

    match = "✓" if expected == predicted else "✗"
    print(f"  [{i:2d}] {token_desc:12s}: exp={expected:3d}, pred={predicted:3d} {match}")
