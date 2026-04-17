"""Quick test of all 9 strict neural prediction test cases."""
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

def test_case(name, bytecode):
    """Test a single case and return pass/fail."""
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.compact(block_size=32)
    model.compact_moe()
    model.eval()

    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    current_context = context[:]
    mismatches = 0

    with torch.no_grad():
        for i in range(len(draft_tokens)):
            ctx_tensor = torch.tensor([current_context], dtype=torch.long)
            logits = model.forward(ctx_tensor)
            predicted = logits[0, -1, :].argmax().item()

            if predicted != draft_tokens[i]:
                mismatches += 1

            current_context.append(draft_tokens[i])

    status = "✓ PASS" if mismatches == 0 else f"✗ FAIL ({mismatches}/35 mismatches)"
    print(f"{name:15s}: {status}")
    return mismatches == 0

print("Running strict neural prediction tests...\n")

tests = [
    ("NOP", [Opcode.NOP]),
    ("IMM 0", [Opcode.IMM | (0 << 8)]),
    ("IMM 255", [Opcode.IMM | (255 << 8)]),
    ("IMM 42", [Opcode.IMM | (42 << 8)]),
    ("JMP 16", [Opcode.JMP | (16 << 8)]),
    ("JMP 8", [Opcode.JMP | (8 << 8)]),
    ("EXIT", [Opcode.EXIT]),
    ("ADD", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.ADD]),
    ("LEA 8", [Opcode.LEA | (8 << 8)]),
]

results = []
for name, bytecode in tests:
    passed = test_case(name, bytecode)
    results.append((name, passed))

print(f"\n{'='*50}")
print(f"Results: {sum(1 for _, p in results if p)}/{len(results)} tests passing")
print(f"{'='*50}")

if all(p for _, p in results):
    print("\n🎉 ALL TESTS PASSING!")
else:
    print("\nFailing tests:")
    for name, passed in results:
        if not passed:
            print(f"  - {name}")
