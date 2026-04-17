"""Test basic opcodes with 8-byte addressing."""
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

def test_opcode(name, bytecode, check_ax=False, expected_ax=None):
    """Test a single opcode and compare DraftVM vs Neural VM."""
    draft_vm = DraftVM(bytecode)
    context = build_context(bytecode)

    # Execute one step
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    # Neural VM forward pass
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    context_with_step = context + draft_tokens
    ctx_tensor = torch.tensor([context_with_step], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    # Check predictions (logits at position i predict token at position i+1)
    ctx_len = len(context)
    matches = 0
    mismatches = []

    for i in range(35):
        predicted = logits[0, ctx_len - 1 + i, :].argmax().item()
        expected = draft_tokens[i]
        if predicted == expected:
            matches += 1
        else:
            mismatches.append((i, expected, predicted))

    match_rate = matches / 35 * 100

    print(f"\nTest: {name}")
    print("-" * 70)
    print(f"  Match rate: {match_rate:.1f}% ({matches}/35 tokens)")

    if check_ax and expected_ax is not None:
        actual_ax = draft_vm.ax
        print(f"  AX value: {actual_ax} (expected: {expected_ax})")
        if actual_ax != expected_ax:
            print(f"    ERROR: AX mismatch!")

    if mismatches:
        print(f"  Mismatches:")
        for pos, exp, pred in mismatches[:5]:  # Show first 5 mismatches
            if pos == 0:
                print(f"    PC marker : expected {exp:8}, got {pred:8}")
            elif 1 <= pos <= 4:
                print(f"    PC byte {pos-1:2}: expected {exp:8}, got {pred:8}")
            elif pos == 5:
                print(f"    AX marker : expected {exp:8}, got {pred:8}")
            elif 6 <= pos <= 9:
                print(f"    AX byte {pos-6:2}: expected {exp:8}, got {pred:8}")
            else:
                print(f"    pos {pos:2}     : expected {exp:8}, got {pred:8}")

    return match_rate == 100.0

# Test cases
print("Testing basic opcodes with 8-byte addressing:")
print("=" * 70)

# IMM: AX = immediate value
test_opcode("IMM 5", [Opcode.IMM | (5 << 8)], check_ax=True, expected_ax=5)

# PSH: push AX to stack
test_opcode("PSH", [Opcode.IMM | (42 << 8), Opcode.PSH])

# NOP: no operation
test_opcode("NOP", [Opcode.NOP])

# JMP: should show one-step delay
test_opcode("JMP 12", [Opcode.JMP | (12 << 8), Opcode.EXIT])

print("\n" + "=" * 70)
print("Note: JMP showing mismatch is expected (one-step delay by design)")
print("      Runner overrides will fix this in actual execution")
