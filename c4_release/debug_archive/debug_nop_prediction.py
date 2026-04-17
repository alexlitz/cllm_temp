"""Debug NOP prediction to see what's happening."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    """Build context tokens for bytecode."""
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

# Setup
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

bytecode = [Opcode.NOP]  # NOP = 39 (0x27)
context = build_context(bytecode)

print("=== CONTEXT ===")
print(f"Context tokens: {context}")
print(f"Context length: {len(context)}")

# Get DraftVM output
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("\n=== DRAFT TOKENS (EXPECTED) ===")
print(f"Draft tokens: {draft_tokens[:10]}...")  # First 10
print(f"Token 0 (REG_PC): {draft_tokens[0]} (expected 257)")
print(f"Token 1 (PC_b0): {draft_tokens[1]}")
print(f"Token 2 (PC_b1): {draft_tokens[2]}")

# Get transformer predictions
ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
print(f"\n=== TRANSFORMER PREDICTIONS ===")
print(f"Context length: {ctx_len}")
print(f"Predicting from position: {ctx_len - 1}")

for i in range(5):
    expected = draft_tokens[i]
    predicted = logits[0, ctx_len - 1 + i, :].argmax().item()
    top5 = torch.topk(logits[0, ctx_len - 1 + i, :], 5)
    print(f"Token {i}: expected={expected:3d}, predicted={predicted:3d}, top5={top5.indices.tolist()}")

# Check the last position of context to see what it's predicting
print(f"\n=== LAST CONTEXT POSITION ===")
print(f"Last context token: {context[-1]} (should be DATA_END)")
last_pred = logits[0, ctx_len - 1, :].argmax().item()
print(f"Prediction at ctx_len-1: {last_pred}")
print(f"Expected (first draft token): {draft_tokens[0]}")
