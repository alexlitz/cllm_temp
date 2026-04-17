"""Test if JMP executes in one step with runner overrides."""
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

# Test: JMP to address 16
bytecode = [Opcode.JMP | (16 << 8)]
draft_vm = DraftVM(bytecode)

context = build_context(bytecode)

print("Testing JMP 16:")
print("="*70)
print(f"Initial DraftVM PC: {draft_vm.pc}")

# Execute one step with DraftVM
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()
draft_pc = draft_tokens[1] | (draft_tokens[2] << 8) | (draft_tokens[3] << 16) | (draft_tokens[4] << 24)

print(f"DraftVM after step 1: PC={draft_pc}")
print(f"  DraftVM predicts PC bytes: {draft_tokens[1:5]}")
print()

# Neural VM prediction
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)

neural_pc_bytes = [logits[0, ctx_len + i - 1, :].argmax().item() for i in range(1, 5)]
neural_pc = neural_pc_bytes[0] | (neural_pc_bytes[1] << 8) | (neural_pc_bytes[2] << 16) | (neural_pc_bytes[3] << 24)

print(f"Neural VM predicts: PC={neural_pc}")
print(f"  Neural VM predicts PC bytes: {neural_pc_bytes}")
print()

print("Analysis:")
if neural_pc == draft_pc:
    print("✓ MATCH! Neural VM and DraftVM agree")
    print("  One-step JMP is working correctly")
else:
    print(f"✗ MISMATCH!")
    print(f"  Neural VM: PC={neural_pc}")
    print(f"  DraftVM: PC={draft_pc}")
    if neural_pc == 8:
        print("  → Neural VM shows one-step delay (PC=8 = old PC + 8)")
        print("  → Runner override should fix this to PC=16")
    else:
        print(f"  → Unexpected value")
