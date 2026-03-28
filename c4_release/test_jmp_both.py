"""Test both JMP 16 and JMP 8."""
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

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

print("JMP Test: Compare JMP 16 vs JMP 8")
print("=" * 70)
print()

for jmp_val in [16, 8]:
    print(f"JMP {jmp_val}:")

    bytecode = [Opcode.JMP | (jmp_val << 8)]
    context = build_context(bytecode)

    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    # Test first 10 tokens
    current_context = context[:]
    all_match = True
    first_mismatch = None

    for i in range(min(10, len(draft_tokens))):
        ctx_tensor = torch.tensor([current_context], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(ctx_tensor)

        predicted = torch.argmax(logits[0, -1]).item()
        draft = draft_tokens[i]

        if predicted != draft:
            if first_mismatch is None:
                first_mismatch = i
            all_match = False
            print(f"  Token {i}: draft={draft:3d}, predicted={predicted:3d} ✗")
        elif i < 5:  # Show first 5 tokens
            print(f"  Token {i}: draft={draft:3d}, predicted={predicted:3d} ✓")

        # Append draft for next prediction
        current_context.append(draft)

    if all_match:
        print(f"  ✅ All 10 tokens match!")
    else:
        print(f"  ❌ First mismatch at token {first_mismatch}")

    print()
