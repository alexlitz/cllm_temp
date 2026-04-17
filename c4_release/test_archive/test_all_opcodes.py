"""Test all basic opcodes after fixes."""
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
    # Basic operations
    ('NOP', [Opcode.NOP]),

    # Immediate loads
    ('IMM 0', [Opcode.IMM | (0 << 8)]),
    ('IMM 1', [Opcode.IMM | (1 << 8)]),
    ('IMM 42', [Opcode.IMM | (42 << 8)]),
    ('IMM 255', [Opcode.IMM | (255 << 8)]),

    # Jumps
    ('JMP 8', [Opcode.JMP | (8 << 8)]),
    ('JMP 16', [Opcode.JMP | (16 << 8)]),
    ('JMP 32', [Opcode.JMP | (32 << 8)]),

    # Load effective address
    ('LEA 8', [Opcode.LEA | (8 << 8)]),
    ('LEA 16', [Opcode.LEA | (16 << 8)]),

    # Arithmetic (need setup)
    ('IMM+ADD', [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.ADD]),

    # Exit
    ('EXIT', [Opcode.EXIT]),
]

print("Comprehensive Opcode Test Suite")
print("=" * 60)
print()

passed = 0
failed = 0
errors = 0

for test_name, bytecode in tests:
    try:
        context = build_context(bytecode)
        draft_vm = DraftVM(bytecode)

        # Run native VM
        try:
            draft_vm.step()
            draft_tokens = draft_vm.draft_tokens()
        except Exception as e:
            print(f"⚠ {test_name:15s}: ERROR (Native VM: {e})")
            errors += 1
            continue

        # Test neural VM predictions
        all_match = True
        first_fail = -1
        first_fail_desc = ""

        for i in range(min(15, len(draft_tokens))):
            ctx = context + draft_tokens[:i]
            with torch.no_grad():
                logits = model.forward(torch.tensor([ctx], dtype=torch.long))
                pred = torch.argmax(logits[0, -1, :]).item()

            if pred != draft_tokens[i]:
                all_match = False
                first_fail = i
                first_fail_desc = f"exp={draft_tokens[i]}, got={pred}"
                break

        if all_match:
            print(f"✓ {test_name:15s}: PASS")
            passed += 1
        else:
            print(f"✗ {test_name:15s}: FAIL at token {first_fail} ({first_fail_desc})")
            failed += 1

    except Exception as e:
        print(f"⚠ {test_name:15s}: ERROR ({str(e)[:40]})")
        errors += 1

print()
print("=" * 60)
print(f"Results: {passed} passed, {failed} failed, {errors} errors")
print(f"Success rate: {passed}/{passed+failed+errors} ({100*passed/(passed+failed+errors):.1f}%)")
