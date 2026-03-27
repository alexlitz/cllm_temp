"""Clarify autoregressive prediction positions."""
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

# Test JMP 12
bytecode = [Opcode.JMP | (12 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Autoregressive Prediction Positions")
print("="*70)
print(f"Context length: {len(context)}")
print(f"Draft tokens (first 10): {draft_tokens[:10]}")
print()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Logits shape: {logits.shape}")
print(f"  Batch: {logits.shape[0]}")
print(f"  Sequence: {logits.shape[1]}")
print(f"  Vocab: {logits.shape[2]}")
print()

# In autoregressive models:
# logits[0, i, :] predicts the token at position i+1
# So to predict token at position ctx_len, we use logits[0, ctx_len-1, :]

ctx_len = len(context)
print(f"To predict tokens after context (positions {ctx_len}+):")
print(f"  Position {ctx_len} (should be REG_PC={Token.REG_PC}):")
print(f"    Predicted from logits[0, {ctx_len-1}, :]")
pred = logits[0, ctx_len-1, :].argmax().item()
print(f"    Prediction: {pred}, Expected: {draft_tokens[0]}, Match: {pred == draft_tokens[0]}")

print(f"\n  Position {ctx_len+1} (should be PC byte 0 = {draft_tokens[1]}):")
print(f"    Predicted from logits[0, {ctx_len}, :]")
pred = logits[0, ctx_len, :].argmax().item()
print(f"    Prediction: {pred}, Expected: {draft_tokens[1]}, Match: {pred == draft_tokens[1]}")

print(f"\n  Position {ctx_len+2} (should be PC byte 1 = {draft_tokens[2]}):")
print(f"    Predicted from logits[0, {ctx_len+1}, :]")
pred = logits[0, ctx_len+1, :].argmax().item()
print(f"    Prediction: {pred}, Expected: {draft_tokens[2]}, Match: {pred == draft_tokens[2]}")
