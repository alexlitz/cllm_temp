"""Comprehensive opcode testing with 8-byte addressing."""
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

def test_opcode(name, bytecode, num_steps=1, show_details=False):
    """Test opcode execution and compare DraftVM vs Neural VM."""
    draft_vm = DraftVM(bytecode)
    context = build_context(bytecode)

    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    all_match = True

    for step in range(num_steps):
        draft_vm.step()
        draft_tokens = draft_vm.draft_tokens()

        context_with_step = context + draft_tokens
        ctx_tensor = torch.tensor([context_with_step], dtype=torch.long)

        with torch.no_grad():
            logits = model.forward(ctx_tensor)

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

        if step == 0:  # Only print first step
            status = "✓" if match_rate == 100.0 else "⚠"
            print(f"{status} {name:20} {match_rate:5.1f}% ({matches}/35)", end="")

            if mismatches and show_details:
                print()
                for pos, exp, pred in mismatches[:3]:
                    if 1 <= pos <= 4:
                        print(f"    PC byte {pos-1}: expected {exp:3}, got {pred:3}")
                    elif 6 <= pos <= 9:
                        print(f"    AX byte {pos-6}: expected {exp:3}, got {pred:3}")
            elif mismatches:
                print(f"  (mismatches at positions: {[m[0] for m in mismatches[:3]]})")
            else:
                print()

        if match_rate != 100.0:
            all_match = False

        # Update context for next step
        context = context_with_step

    return all_match

print("=" * 70)
print("Comprehensive Opcode Test - 8-Byte Addressing")
print("=" * 70)

print("\n--- Basic Opcodes ---")
test_opcode("IMM 5", [Opcode.IMM | (5 << 8)])
test_opcode("IMM 255", [Opcode.IMM | (255 << 8)])
test_opcode("NOP", [Opcode.NOP])
test_opcode("EXIT", [Opcode.EXIT])

print("\n--- Stack Operations ---")
test_opcode("PSH", [Opcode.IMM | (42 << 8), Opcode.PSH])
test_opcode("PSH + PSH", [
    Opcode.IMM | (10 << 8),
    Opcode.PSH,
    Opcode.IMM | (20 << 8),
    Opcode.PSH
])

print("\n--- Binary Operations ---")
test_opcode("ADD", [
    Opcode.IMM | (5 << 8),
    Opcode.PSH,
    Opcode.IMM | (3 << 8),
    Opcode.ADD
])
test_opcode("SUB", [
    Opcode.IMM | (10 << 8),
    Opcode.PSH,
    Opcode.IMM | (3 << 8),
    Opcode.SUB
])
test_opcode("MUL", [
    Opcode.IMM | (6 << 8),
    Opcode.PSH,
    Opcode.IMM | (7 << 8),
    Opcode.MUL
])

print("\n--- Jumps & Branches (one-step delay expected) ---")
test_opcode("JMP 12", [Opcode.JMP | (12 << 8), Opcode.EXIT], show_details=True)
test_opcode("BZ (taken)", [
    Opcode.IMM | (0 << 8),
    Opcode.BZ | (12 << 8),
    Opcode.EXIT
], show_details=True)
test_opcode("BZ (not taken)", [
    Opcode.IMM | (5 << 8),
    Opcode.BZ | (12 << 8),
    Opcode.NOP
])
test_opcode("BNZ (taken)", [
    Opcode.IMM | (5 << 8),
    Opcode.BNZ | (12 << 8),
    Opcode.EXIT
], show_details=True)
test_opcode("BNZ (not taken)", [
    Opcode.IMM | (0 << 8),
    Opcode.BNZ | (12 << 8),
    Opcode.NOP
])

print("\n--- Comparisons ---")
test_opcode("EQ (true)", [
    Opcode.IMM | (5 << 8),
    Opcode.PSH,
    Opcode.IMM | (5 << 8),
    Opcode.EQ
])
test_opcode("EQ (false)", [
    Opcode.IMM | (5 << 8),
    Opcode.PSH,
    Opcode.IMM | (3 << 8),
    Opcode.EQ
])
test_opcode("LT (true)", [
    Opcode.IMM | (10 << 8),
    Opcode.PSH,
    Opcode.IMM | (3 << 8),
    Opcode.LT
])

print("\n--- Bitwise Operations ---")
test_opcode("OR", [
    Opcode.IMM | (0x0F << 8),
    Opcode.PSH,
    Opcode.IMM | (0xF0 << 8),
    Opcode.OR
])
test_opcode("AND", [
    Opcode.IMM | (0xFF << 8),
    Opcode.PSH,
    Opcode.IMM | (0x0F << 8),
    Opcode.AND
])
test_opcode("XOR", [
    Opcode.IMM | (0xAA << 8),
    Opcode.PSH,
    Opcode.IMM | (0x55 << 8),
    Opcode.XOR
])

print("\n" + "=" * 70)
print("Note: JMP/BZ/BNZ showing PC byte 0 mismatch is expected")
print("      (one-step delay by design, runner overrides fix this)")
print("=" * 70)
