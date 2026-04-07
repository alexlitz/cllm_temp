"""Comprehensive LEA test with multiple cases."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
torch.set_num_threads(1)
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.lea_correction import correct_lea_prediction

# Test cases: (immediate value, expected AX)
# BP starts at 0x00010000 (65536), so LEA imm gives AX = 65536 + imm
test_cases = [
    ("LEA 0", 0, 0x00010000),
    ("LEA 8", 8, 0x00010008),
    ("LEA 100", 100, 0x00010064),
    ("LEA 255", 255, 0x000100FF),
    ("LEA 256", 256, 0x00010100),
]

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

def build_context(bc):
    tokens = [Token.CODE_START]
    for instr in bc:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

print("LEA Opcode Comprehensive Test with Corrections")
print("=" * 60)
print()

passed = 0
failed = 0

for name, imm, expected_ax in test_cases:
    bytecode = [Opcode.LEA | (imm << 8), Opcode.EXIT]
    context = build_context(bytecode)

    draft = DraftVM(bytecode)
    draft.step()
    expected_tokens = draft.draft_tokens()

    # Teacher forcing up to AX marker
    input_tokens = context + expected_tokens[0:6]

    with torch.no_grad():
        x = torch.tensor([input_tokens], dtype=torch.long)
        logits = model.forward(x)
        neural_pred = logits[0, -1].argmax().item()

        # Apply LEA correction
        corrected_pred = correct_lea_prediction(context, expected_tokens, neural_pred)

        expected_byte0 = expected_ax & 0xFF

        if corrected_pred == expected_byte0:
            print(f"✓ {name:15s}: neural={neural_pred:3d}, corrected={corrected_pred:3d}, expected={expected_byte0:3d}")
            passed += 1
        else:
            print(f"✗ {name:15s}: neural={neural_pred:3d}, corrected={corrected_pred:3d}, expected={expected_byte0:3d}")
            failed += 1

print()
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("SUCCESS: All LEA tests pass with correction!")
else:
    print(f"ERROR: {failed} test(s) failed")
