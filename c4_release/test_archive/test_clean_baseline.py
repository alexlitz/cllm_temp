"""Test clean baseline Neural VM - no session modifications."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode

# Build context manually
def build_simple_context(bytecode):
    """Build minimal context for testing."""
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

# Test program: JMP 12, EXIT
bytecode = [Opcode.JMP | (12 << 8), Opcode.EXIT]
context = build_simple_context(bytecode)

print("Testing clean baseline Neural VM")
print("="*60)
print(f"Program: JMP 12, EXIT")
print(f"Context length: {len(context)}")

# Load model
print("\nLoading model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()
print("Model loaded")

# Generate one step (35 tokens)
print("\nGenerating first VM step...")
with torch.no_grad():
    generated = []
    current_context = context

    for i in range(35):
        ctx_tensor = torch.tensor([current_context], dtype=torch.long)
        logits = model.forward(ctx_tensor)
        next_token = logits[0, -1, :].argmax(-1).item()
        generated.append(next_token)
        current_context = current_context + [next_token]

        if next_token == Token.STEP_END:
            print(f"  STEP_END at token {i}")
            break

print(f"\nGenerated {len(generated)} tokens")
print(f"First 10 tokens: {generated[:10]}")

# Check if it matches expected pattern:
# Token 0: REG_PC or REG_AX marker
# Tokens 1-4: 4 bytes
# Token 5: REG_AX or REG_PC marker
# ...
print(f"\nToken analysis:")
print(f"  Token 0 (marker): {generated[0]} (REG_PC={Token.REG_PC}, REG_AX={Token.REG_AX})")
if len(generated) > 1:
    print(f"  Token 1 (byte 0): {generated[1]}")
print("\nBaseline test complete")
