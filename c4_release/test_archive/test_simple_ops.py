#!/usr/bin/env python3
"""Simple opcode test - single step operations."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

def build_context(bytecode, data=b""):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

# Initialize model once
print("Loading model...", flush=True)
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()
print("Model loaded.\n")

def test_single_step(name, bytecode, expected_ax):
    """Test a single-step operation."""
    context = build_context(bytecode)
    draft = DraftVM(bytecode)
    draft.step()

    # Predict step 1 tokens autoregressively
    predicted_tokens = []
    current_ctx = context[:]

    with torch.no_grad():
        for i in range(39):
            token_ids = torch.tensor([current_ctx], dtype=torch.long)
            logits = model(token_ids)
            next_logits = logits[0, -1, :]
            predicted = next_logits.argmax().item()
            predicted_tokens.append(predicted)
            current_ctx.append(predicted)

    # Extract AX bytes (positions 6-9)
    ax_bytes = predicted_tokens[6:10]
    predicted_ax = ax_bytes[0] | (ax_bytes[1] << 8) | (ax_bytes[2] << 16) | (ax_bytes[3] << 24)

    match = predicted_ax == expected_ax
    if match:
        print(f"{name}: PASS (0x{predicted_ax:08X})")
    else:
        print(f"{name}: FAIL")
        print(f"  Expected:  0x{expected_ax:08X}")
        print(f"  Predicted: 0x{predicted_ax:08X}")
        print(f"  Draft:     0x{draft.ax:08X}")
    return match

passed = 0
failed = 0

# First-step operations
tests = [
    ("IMM 0", [Opcode.IMM | (0 << 8)], 0),
    ("IMM 42", [Opcode.IMM | (42 << 8)], 42),
    ("IMM 255", [Opcode.IMM | (255 << 8)], 255),
    ("LEA 0", [Opcode.LEA | (0 << 8)], 0x10000),
    ("LEA 8", [Opcode.LEA | (8 << 8)], 0x10008),
    ("LEA 255", [Opcode.LEA | (255 << 8)], 0x100FF),
]

for name, bytecode, expected in tests:
    if test_single_step(name, bytecode, expected):
        passed += 1
    else:
        failed += 1

print(f"\nResults: {passed}/{passed+failed} passed")
if failed > 0:
    sys.exit(1)
