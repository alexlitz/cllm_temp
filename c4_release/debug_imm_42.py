"""Debug IMM 42 predictions."""
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

# Get DraftVM output
draft_vm = DraftVM(bytecode)
print(f"DraftVM before: PC={draft_vm.pc}, AX={draft_vm.ax}")
draft_vm.step()
print(f"DraftVM after:  PC={draft_vm.pc}, AX={draft_vm.ax}")
draft_tokens = draft_vm.draft_tokens()

# Use teacher forcing
ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)

print("\n=== COMPARISON ===")
print("Pos Token      Expected Predicted Match")
token_names = ['REG_PC', 'PC_b0', 'PC_b1', 'PC_b2', 'PC_b3',
               'REG_AX', 'AX_b0', 'AX_b1', 'AX_b2', 'AX_b3']

for i in range(10):
    expected = draft_tokens[i]
    predicted = logits[0, ctx_len - 1 + i, :].argmax().item()
    match = "✓" if expected == predicted else "✗"
    name = token_names[i] if i < len(token_names) else f'Token{i}'
    print(f"{i:2d}  {name:8s} {expected:3d}      {predicted:3d}       {match}")

print(f"\nDraftVM: PC=10 (0x0A) = bytes [10, 0, 0, 0]")
print(f"         AX=42 (0x2A) = bytes [42, 0, 0, 0]")
