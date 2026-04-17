"""Test neural VM with proper 2-step execution."""
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

# Test: IMM 5, then NOP
bytecode = [Opcode.IMM | (5 << 8), Opcode.NOP]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

# Get step 1 and step 2 from DraftVM
draft_vm.step()  # Execute IMM 5
step1_tokens = draft_vm.draft_tokens()
draft_vm.step()  # Execute NOP
step2_tokens = draft_vm.draft_tokens()

print("DraftVM Execution:")
print("="*70)
print(f"After IMM 5 (step 1):")
print(f"  PC: {step1_tokens[1]} (should be 7 = 0*5+2 + 5)")
print(f"  AX: {step1_tokens[6]} (should be 5)")
print()
print(f"After NOP (step 2):")
print(f"  PC: {step2_tokens[1]} (should be 12 = 1*5+2 + 5)")
print(f"  AX: {step2_tokens[6]} (should still be 5)")
print()

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Test if neural model can predict step 2 using step 1
context_with_steps = context + step1_tokens + step2_tokens
ctx_tensor = torch.tensor([context_with_steps], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
step1_start = ctx_len
step2_start = ctx_len + 35

print("Neural VM Predictions:")
print("="*70)

# Check if step 2 PC is predicted correctly
# Neural model at end of step 1 should predict step 2 PC
step2_pc_marker = step2_start  # Position of REG_PC marker
step2_pc_byte0 = step2_start + 1  # Position of PC byte 0

# Logits at position i-1 predict token at position i
pred_pc_marker = logits[0, step2_pc_marker - 1, :].argmax().item()
pred_pc_byte0 = logits[0, step2_pc_byte0 - 1, :].argmax().item()

print(f"Step 2 PC marker prediction:")
print(f"  Expected: {step2_tokens[0]} (REG_PC=257)")
print(f"  Got: {pred_pc_marker}")
print(f"  Match: {'✓' if pred_pc_marker == step2_tokens[0] else '✗'}")
print()
print(f"Step 2 PC byte 0 prediction:")
print(f"  Expected: {step2_tokens[1]} (should be 12)")
print(f"  Got: {pred_pc_byte0}")
print(f"  Match: {'✓' if pred_pc_byte0 == step2_tokens[1] else '✗'}")

# The theory is that the neural VM JMP delay means:
# - Step 1 outputs PC from first-step default + any IMM effects
# - Step 2 outputs PC that includes JMP effects from step 1
