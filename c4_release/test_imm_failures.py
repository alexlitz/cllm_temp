"""Quick test of IMM 42 and IMM 255 to see what's failing."""
import sys
if 'neural_vm.vm_step' in sys.modules:
    del sys.modules['neural_vm.vm_step']
if 'neural_vm.speculative' in sys.modules:
    del sys.modules['neural_vm.speculative']

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

print("Testing IMM 42 and IMM 255")
print("=" * 70)
print()

for value in [42, 255]:
    print(f"IMM {value}:")
    print("-" * 70)

    bytecode = [Opcode.IMM | (value << 8)]
    context = build_context(bytecode)

    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    # Test first few tokens
    current_context = context[:]
    all_match = True
    first_mismatch = None

    for i in range(min(10, len(draft_tokens))):
        ctx_tensor = torch.tensor([current_context], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(ctx_tensor)

        predicted = torch.argmax(logits[0, -1]).item()
        draft = draft_tokens[i]

        match = predicted == draft
        if not match and first_mismatch is None:
            first_mismatch = i
            all_match = False

        marker = "✓" if match else "✗"

        # Identify token name
        if i == 0:
            name = "REG_AX"
        elif i <= 4:
            name = f"AX_b{i-1}"
        elif i == 5:
            name = "REG_PC"
        elif i <= 9:
            name = f"PC_b{i-6}"
        else:
            name = f"Token{i}"

        if not match or i < 10:
            print(f"  Token {i:2d} ({name:8s}): draft={draft:3d}, predicted={predicted:3d} {marker}")

        # Append draft token to build context incrementally
        current_context.append(draft)

    print()
    if all_match:
        print(f"  ✅ IMM {value} ALL TOKENS MATCH")
    else:
        print(f"  ❌ IMM {value} FAILS at token {first_mismatch}")
    print()
    print()
