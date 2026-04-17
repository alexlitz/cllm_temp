"""Check full model prediction for step 2."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
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

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1

print("=== STEP 1 ===")
print(f"Draft tokens[0-2]: {draft1[0:3]} (should be [257, 8, 0])")

draft_vm.step()
draft2 = draft_vm.draft_tokens()
context_step2 = context_step1 + draft2

print("\n=== STEP 2 ===")
print(f"Draft tokens[0-2]: {draft2[0:3]} (should be [257, 16, 0])")

# Full forward pass
ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len = len(context_step1)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

    # Check predictions for step 2 tokens
    print("\n=== TRANSFORMER PREDICTIONS FOR STEP 2 ===")
    for i in range(5):
        expected = draft2[i]
        predicted = logits[0, ctx_len - 1 + i, :].argmax().item()
        match = "✓" if expected == predicted else "✗"
        print(f"Token {i}: expected={expected:3d}, predicted={predicted:3d} {match}")

    # Detailed check for PC byte 0
    token_1_logits = logits[0, ctx_len, :]
    top5 = torch.topk(token_1_logits, 5)
    print(f"\nToken 1 (PC byte 0) top 5 predictions:")
    for val, idx in zip(top5.values, top5.indices):
        print(f"  {idx.item():3d}: {val.item():.3f}")
