"""Debug PC byte encoding difference between DraftVM and transformer."""
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

# Setup
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.NOP]
context = build_context(bytecode)

# Get DraftVM output
draft_vm = DraftVM(bytecode)
print("=== DRAFTVM STATE BEFORE STEP ===")
print(f"PC: {draft_vm.pc}")
print(f"AX: {draft_vm.ax}")

draft_vm.step()
print("\n=== DRAFTVM STATE AFTER NOP ===")
print(f"PC: {draft_vm.pc}")
print(f"AX: {draft_vm.ax}")

draft_tokens = draft_vm.draft_tokens()
print("\n=== DRAFTVM OUTPUT TOKENS ===")
print(f"Token 0 (REG_PC): {draft_tokens[0]}")
print(f"Token 1 (PC_b0): {draft_tokens[1]}")
print(f"Token 2 (PC_b1): {draft_tokens[2]}")
print(f"Token 3 (PC_b2): {draft_tokens[3]}")
print(f"Token 4 (PC_b3): {draft_tokens[4]}")
print(f"PC as integer from bytes: {draft_tokens[1] | (draft_tokens[2] << 8) | (draft_tokens[3] << 16) | (draft_tokens[4] << 24)}")

# Get transformer prediction
current_context = context[:]
with torch.no_grad():
    for i in range(5):  # Generate first 5 tokens (REG_PC + 4 bytes)
        ctx_tensor = torch.tensor([current_context], dtype=torch.long)
        logits = model.forward(ctx_tensor)
        next_token = logits[0, -1, :].argmax().item()
        current_context.append(next_token)
        if i == 0:
            print(f"\n=== TRANSFORMER PREDICTION ===")
            print(f"Token {i} (REG_PC): {next_token}")
        else:
            print(f"Token {i} (PC_b{i-1}): {next_token}")

trans_pc = current_context[-4] | (current_context[-3] << 8) | (current_context[-2] << 16) | (current_context[-1] << 24)
print(f"PC as integer from transformer bytes: {trans_pc}")
