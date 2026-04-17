"""Test autoregressive generation token by token."""
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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

# Get DraftVM expected output
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"Context: {context}")
print(f"Expected draft tokens: {draft_tokens[:10]}")
print()

# Generate autoregressively
current_context = context[:]
predictions = []

with torch.no_grad():
    for i in range(10):  # Generate 10 tokens (first VM step)
        ctx_tensor = torch.tensor([current_context], dtype=torch.long)
        logits = model.forward(ctx_tensor)
        predicted = logits[0, -1, :].argmax().item()
        predictions.append(predicted)
        current_context.append(predicted)

        expected = draft_tokens[i]
        match = "✓" if predicted == expected else "✗"
        token_names = ['REG_PC', 'PC_b0', 'PC_b1', 'PC_b2', 'PC_b3',
                       'REG_AX', 'AX_b0', 'AX_b1', 'AX_b2', 'AX_b3']
        name = token_names[i] if i < len(token_names) else f'Token{i}'
        print(f"{i:2d}  {name:8s} expected={expected:3d}, predicted={predicted:3d}  {match}")

print(f"\nPredictions:  {predictions}")
print(f"Expected:     {draft_tokens[:10]}")
