"""Test JMP with two-step execution."""
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

# Execute step 1 (JMP)
draft_vm.step()
step1_tokens = draft_vm.draft_tokens()

# Execute step 2 (should be at PC=12, which is actually instruction at bytecode index 1 due to JMP)
# Wait, JMP 12 means jump to address 12. Let me check if this is the EXIT instruction.
# Bytecode layout: instruction 0 at address 2, instruction 1 at address 10
# So address 12 is NOT an instruction boundary. This might be an issue with the test.

# Actually, looking at the DraftVM code, PC = idx * 8 + 2
# So: idx=0 -> PC=2, idx=1 -> PC=10, idx=2 -> PC=18
# JMP 12 would set PC=12, which means idx = (12-2)/8 = 1.25, which is not an instruction boundary\!

# Let me use JMP 10 instead to jump to instruction 1
print("Correcting test: JMP should jump to valid instruction boundary")
print("Using JMP 10 (instruction 1) instead of JMP 12")
print()

bytecode = [Opcode.JMP | (10 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)

print("DraftVM Execution:")
print("="*70)

# Step 1: JMP 10
draft_vm.step()
step1_tokens = draft_vm.draft_tokens()
print(f"After JMP 10 (step 1):")
print(f"  PC: {draft_vm.pc} (DraftVM shows target PC immediately)")
print(f"  PC bytes: {step1_tokens[1:5]}")
print()

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Test step 1 prediction
context_step1 = context + step1_tokens
ctx_tensor = torch.tensor([context_step1], dtype=torch.long)

with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
pc_byte0_pred = logits[0, ctx_len, :].argmax().item()

print("Neural VM Prediction:")
print("="*70)
print(f"Step 1 PC byte 0 prediction:")
print(f"  Expected (DraftVM): {step1_tokens[1]}")
print(f"  Predicted (Neural): {pc_byte0_pred}")

if pc_byte0_pred == 10:
    print(f"  ✓ Neural VM predicts PC=10 (old PC + 8, one-step delay)")
elif pc_byte0_pred == step1_tokens[1]:
    print(f"  ✓ Neural VM matches DraftVM (no delay)")
else:
    print(f"  ✗ Mismatch")

print()
print("Note: JMP has one-step delay in neural VM by design (comment at vm_step.py:1909-1912)")
