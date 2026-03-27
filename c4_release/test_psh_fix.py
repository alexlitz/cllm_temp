"""Test PSH fix with CMP[0] gating."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

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

print("Testing PSH Fix with CMP[0] Gating")
print("=" * 70)
print()

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

# Test JMP (should still work)
print("Test 1: JMP 16 (should still work with PSH fix)")
print("-" * 70)

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context[:]
jmp_pass = True

for i in [1, 2, 13, 21]:  # Critical tokens
    for j in range(i):
        current_context.append(draft_tokens[j])

    ctx_tensor = torch.tensor([current_context], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    predicted = torch.argmax(logits[0, -1]).item()
    draft = draft_tokens[i]

    match = predicted == draft
    if not match:
        jmp_pass = False

    marker = "✓" if match else "✗"
    names = {1: "PC_b0", 2: "PC_b1", 13: "SP_b2", 21: "ST_b0"}
    print(f"  Token {i:2d} ({names[i]:6s}): draft={draft:3d}, predicted={predicted:3d} {marker}")

if jmp_pass:
    print("  ✅ JMP still works!")
else:
    print("  ❌ JMP broken by PSH fix")

print()
print()

# TODO: Test actual PSH operation once we understand the expected behavior
print("PSH test requires understanding expected PSH behavior")
print("Current status: PSH units re-enabled with CMP[0] gating")
