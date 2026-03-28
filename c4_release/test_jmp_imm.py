"""Quick test for JMP and IMM fixes."""
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

tests = [
    ('NOP', [Opcode.NOP]),
    ('IMM 0', [Opcode.IMM | (0 << 8)]),
    ('IMM 42', [Opcode.IMM | (42 << 8)]),
    ('IMM 255', [Opcode.IMM | (255 << 8)]),
    ('JMP 8', [Opcode.JMP | (8 << 8)]),
    ('JMP 16', [Opcode.JMP | (16 << 8)]),
]

print("Current Test Status:")
print("=" * 50)

for test_name, bytecode in tests:
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    all_match = True
    first_fail = -1
    for i in range(min(10, len(draft_tokens))):
        ctx = context + draft_tokens[:i]
        with torch.no_grad():
            logits = model.forward(torch.tensor([ctx], dtype=torch.long))
            pred = torch.argmax(logits[0, -1, :]).item()
        if pred != draft_tokens[i]:
            all_match = False
            first_fail = i
            break

    status_str = "PASS" if all_match else f"FAIL at token {first_fail}"
    symbol = "✓" if all_match else "✗"
    print(f"{symbol} {test_name:12s}: {status_str}")

print("=" * 50)
