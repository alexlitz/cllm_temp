"""Quick opcode test - single model load."""
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

def test_opcode(model, name, bytecode, max_tokens=12):
    """Test a single opcode."""
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    for i in range(min(max_tokens, len(draft_tokens))):
        ctx = context + draft_tokens[:i]
        with torch.no_grad():
            logits = model.forward(torch.tensor([ctx], dtype=torch.long))
            pred = torch.argmax(logits[0, -1, :]).item()
        if pred != draft_tokens[i]:
            return False, i, draft_tokens[i], pred

    return True, -1, -1, -1

print("Loading model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

print("Testing opcodes...")
print("=" * 60)

tests = [
    ('NOP', [Opcode.NOP]),
    ('IMM 0', [Opcode.IMM | (0 << 8)]),
    ('IMM 1', [Opcode.IMM | (1 << 8)]),
    ('IMM 42', [Opcode.IMM | (42 << 8)]),
    ('IMM 255', [Opcode.IMM | (255 << 8)]),
    ('JMP 8', [Opcode.JMP | (8 << 8)]),
    ('JMP 16', [Opcode.JMP | (16 << 8)]),
    ('JMP 32', [Opcode.JMP | (32 << 8)]),
    ('LEA 8', [Opcode.LEA | (8 << 8)]),
    ('LEA 16', [Opcode.LEA | (16 << 8)]),
    ('EXIT', [Opcode.EXIT]),
]

passed = 0
failed = 0

for name, bytecode in tests:
    success, fail_idx, expected, got = test_opcode(model, name, bytecode)
    if success:
        print(f"✓ {name:15s}: PASS")
        passed += 1
    else:
        print(f"✗ {name:15s}: FAIL at token {fail_idx} (exp={expected}, got={got})")
        failed += 1

print("=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print(f"Success rate: {100*passed/(passed+failed):.1f}%")
