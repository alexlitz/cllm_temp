#!/usr/bin/env python3
"""Test LEA with large immediates."""
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
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

def test_lea(imm):
    """Test LEA imm where result = BP + imm = 0x10000 + imm."""
    bytecode = [Opcode.NOP, Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft = DraftVM(bytecode)

    # Step 1: NOP
    draft.step()
    step1_tokens = draft.draft_tokens()

    # Step 2: LEA
    draft.step()
    step2_tokens = draft.draft_tokens()

    # Run neural VM for step 2 (after step 1 tokens added)
    full_context = context + step1_tokens

    # Predict each token of step 2 autoregressively
    predicted_tokens = []
    current_ctx = full_context[:]

    with torch.no_grad():
        for i in range(39):  # 39 tokens per step
            token_ids = torch.tensor([current_ctx], dtype=torch.long)
            logits = model(token_ids)
            next_logits = logits[0, -1, :]
            predicted = next_logits.argmax().item()
            predicted_tokens.append(predicted)
            current_ctx.append(predicted)

    # Extract AX bytes from tokens (positions 6-9 after markers)
    # Step tokens: PC_marker, 4 PC bytes, AX_marker, 4 AX bytes, ...
    ax_bytes = predicted_tokens[6:10]
    predicted_ax = ax_bytes[0] | (ax_bytes[1] << 8) | (ax_bytes[2] << 16) | (ax_bytes[3] << 24)

    expected_ax = 0x10000 + imm
    draft_ax = draft.ax

    match = predicted_ax == expected_ax

    if match:
        print(f"LEA {imm}: PASS (predicted 0x{predicted_ax:08X}, expected 0x{expected_ax:08X})")
    else:
        print(f"LEA {imm}: FAIL")
        print(f"  Expected:  0x{expected_ax:08X}")
        print(f"  Predicted: 0x{predicted_ax:08X}")
        print(f"  Draft:     0x{draft_ax:08X}")
        print(f"  AX bytes:  {ax_bytes} vs expected {[expected_ax & 0xFF, (expected_ax >> 8) & 0xFF, (expected_ax >> 16) & 0xFF, (expected_ax >> 24) & 0xFF]}")

    return match

# Test cases
tests = [
    0,      # AX = 0x10000
    1,      # AX = 0x10001
    8,      # AX = 0x10008
    16,     # AX = 0x10010
    255,    # AX = 0x100FF
    256,    # AX = 0x10100 (byte 1 = 1)
    511,    # AX = 0x101FF (byte 1 = 1)
    512,    # AX = 0x10200 (byte 1 = 2)
    4096,   # AX = 0x11000 (byte 2 = 1)
    65535,  # AX = 0x1FFFF (byte 1 = 255, byte 2 = 1)
]

print("Testing LEA with large immediates:\n")
passed = 0
failed = 0
for imm in tests:
    if test_lea(imm):
        passed += 1
    else:
        failed += 1

print(f"\nResults: {passed}/{passed+failed} passed")
if failed > 0:
    sys.exit(1)
