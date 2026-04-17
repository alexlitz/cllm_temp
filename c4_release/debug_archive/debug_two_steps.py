"""Check if JMP takes effect in step 2."""
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

# Test JMP 12, then NOP
bytecode = [Opcode.JMP | (12 << 8), Opcode.NOP]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

print("Testing JMP 12 with TWO steps")
print("="*70)

# Run two steps
draft_vm.step()
step1_tokens = draft_vm.draft_tokens()
print(f"Step 1 (after JMP):")
print(f"  DraftVM PC: {draft_vm.pc}")
print(f"  PC bytes: {step1_tokens[1:5]}")

draft_vm.step()
step2_tokens = draft_vm.draft_tokens()
print(f"\nStep 2 (after NOP):")
print(f"  DraftVM PC: {draft_vm.pc}")
print(f"  PC bytes: {step2_tokens[1:5]}")

# Now test neural prediction
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Generate predictions for step 1 and step 2
context_with_drafts = context + step1_tokens + step2_tokens
ctx_tensor = torch.tensor([context_with_drafts], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)

print("\n" + "="*70)
print("Neural VM Predictions:")
print("-"*70)

print("\nStep 1 PC bytes (positions {}{}):".format(ctx_len+1, "-"+str(ctx_len+4)))
for i in range(4):
    pos = ctx_len + 1 + i  # PC byte positions
    expected = step1_tokens[1 + i]
    predicted = logits[0, pos - 1, :].argmax().item()
    match = "✓" if expected == predicted else "✗"
    print(f"  PC byte {i}: expected {expected:3d}, predicted {predicted:3d} {match}")

step2_start = ctx_len + 35  # Step 2 starts 35 tokens after step 1
print(f"\nStep 2 PC bytes (positions {step2_start+1}-{step2_start+4}):")
for i in range(4):
    pos = step2_start + 1 + i  # PC byte positions in step 2
    expected = step2_tokens[1 + i]
    predicted = logits[0, pos - 1, :].argmax().item()
    match = "✓" if expected == predicted else "✗"
    print(f"  PC byte {i}: expected {expected:3d}, predicted {predicted:3d} {match}")

print("\n" + "="*70)
print("Analysis:")
print("-"*70)
print("If JMP has one-step delay:")
print("  Step 1 should output PC=0 or PC=5 (default/increment)")
print("  Step 2 should output PC=12 (JMP target)")
print()
print("DraftVM behavior:")
print(f"  Step 1: PC={step1_tokens[1]} (byte 0)")
print(f"  Step 2: PC={step2_tokens[1]} (byte 0)")
