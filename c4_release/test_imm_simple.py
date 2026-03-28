"""Simple test of IMM 42 to check if it really fails."""
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

print("Loading model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

print("Testing IMM 42")
print("=" * 70)

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"Testing first 10 tokens...")
print()

current_context = context[:]
for i in range(10):
    ctx_tensor = torch.tensor([current_context], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    predicted = torch.argmax(logits[0, -1]).item()
    draft = draft_tokens[i]

    match = "✓" if predicted == draft else "✗"
    print(f"Token {i:2d}: draft={draft:3d}, predicted={predicted:3d} {match}")

    if predicted != draft:
        print(f"\n❌ FIRST FAILURE at token {i}")
        break

    # Append draft token for next prediction
    current_context.append(draft)
else:
    print("\n✅ All 10 tokens match!")
