"""Quick test for LEA fix."""
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

def test_lea(model, immediate):
    """Test LEA with given immediate value."""
    bytecode = [Opcode.LEA | (immediate << 8), Opcode.EXIT]
    context = build_context(bytecode)
    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    # Expected: AX = immediate + 0x00010000 (initial BP)
    expected_ax = immediate + 0x00010000
    ax_bytes = [
        expected_ax & 0xFF,
        (expected_ax >> 8) & 0xFF,
        (expected_ax >> 16) & 0xFF,
        (expected_ax >> 24) & 0xFF
    ]

    # Run model and compare
    current_context = list(context)
    for i, expected in enumerate(draft_tokens):
        ctx_tensor = torch.tensor([current_context], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(ctx_tensor)
        pred = torch.argmax(logits[0, -1, :]).item()
        current_context.append(pred)

        # Check first 4 tokens (AX bytes)
        if i < 4:
            if pred != ax_bytes[i]:
                return False, f"byte {i}: got {pred}, expected {ax_bytes[i]}"

    return True, "PASS"

# Load model once
print("Loading model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

# Test LEA 8 and LEA 16
print("Testing LEA fix...")
print()

for imm in [8, 16]:
    passed, msg = test_lea(model, imm)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"LEA {imm}: {status}")
    if not passed:
        print(f"  {msg}")

print()
print("Done!")
