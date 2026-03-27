"""Quick verification of selective OUTPUT gating fix."""
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

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Testing critical tokens with selective OUTPUT gating:\n")

# Test tokens: 1 (PC_b0), 2 (PC_b1), 13 (SP_b2), 21 (ST_b0)
critical_tokens = {
    1: ("PC_b0", 16),
    2: ("PC_b1", 0),
    13: ("SP_b2", 1),
    21: ("ST_b0", 0)
}

current_context = context[:]
all_pass = True

for token_idx in range(22):
    ctx_tensor = torch.tensor([current_context], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    predicted = torch.argmax(logits[0, -1]).item()
    draft = draft_tokens[token_idx]

    if token_idx in critical_tokens:
        name, expected = critical_tokens[token_idx]
        match = "✓" if predicted == expected else "✗"
        print(f"Token {token_idx:2d} ({name:6s}): draft={draft:3d}, predicted={predicted:3d}, expected={expected:3d} {match}")
        if predicted != expected:
            all_pass = False

    current_context.append(draft_tokens[token_idx])

print()
if all_pass:
    print("✅ ALL CRITICAL TOKENS PASS - Selective OUTPUT gating works!")
else:
    print("❌ SOME TOKENS FAILED - Need further investigation")
