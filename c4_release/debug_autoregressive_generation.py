"""Test if the model can generate tokens autoregressively."""
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
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.NOP]
context = build_context(bytecode)

# Get expected tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
expected_tokens = draft_vm.draft_tokens()

print("=== EXPECTED TOKENS ===")
print(f"Expected: {expected_tokens[:10]}")

# Try autoregressive generation (1 token at a time)
print("\n=== AUTOREGRESSIVE GENERATION ===")
generated = []
current_context = context[:]

with torch.no_grad():
    for i in range(10):  # Generate first 10 tokens
        ctx_tensor = torch.tensor([current_context], dtype=torch.long)
        logits = model.forward(ctx_tensor)
        next_token = logits[0, -1, :].argmax().item()
        generated.append(next_token)
        current_context.append(next_token)

        exp = expected_tokens[i] if i < len(expected_tokens) else -1
        match = "✓" if next_token == exp else "✗"
        print(f"Token {i}: predicted={next_token:3d}, expected={exp:3d} {match}")

print(f"\nGenerated: {generated}")
print(f"Expected:  {expected_tokens[:10]}")
matches = sum(1 for i, (g, e) in enumerate(zip(generated, expected_tokens[:10])) if g == e)
print(f"Matches: {matches}/10")
