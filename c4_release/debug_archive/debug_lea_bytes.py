#!/usr/bin/env python3
"""Debug LEA byte 2-3 predictions."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
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

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

def test_lea_all_bytes(imm):
    """Test all 4 AX bytes for LEA imm."""
    bytecode = [Opcode.LEA | (imm << 8)]
    context = build_context(bytecode)
    draft = DraftVM(bytecode)
    draft.step()
    step1_tokens = draft.draft_tokens()

    expected_ax = draft.ax
    print(f"\nLEA {imm}: AX = BP + {imm} = {draft.bp} + {imm} = {expected_ax} (0x{expected_ax:08X})")

    results = []
    # Test each AX byte (tokens 6-9 after PC marker)
    for byte_idx in range(4):
        # Context includes: PC marker + 4 PC bytes + AX marker + byte_idx AX bytes
        token_count = 6 + byte_idx  # 1 PC marker + 4 PC bytes + 1 AX marker + byte_idx
        ctx = context + step1_tokens[:token_count]
        ax_byte_pos = len(ctx) - 1

        token_ids = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            x = model.embed(token_ids)
            for block in model.blocks:
                x = block(x)

            output_lo = [x[0, ax_byte_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
            output_hi = [x[0, ax_byte_pos, BD.OUTPUT_HI + k].item() for k in range(16)]

            lo_pred = max(range(16), key=lambda k: output_lo[k])
            hi_pred = max(range(16), key=lambda k: output_hi[k])
            byte_pred = lo_pred + (hi_pred << 4)

            expected_byte = (expected_ax >> (byte_idx * 8)) & 0xFF
            status = "PASS" if byte_pred == expected_byte else "FAIL"
            results.append((byte_idx, byte_pred, expected_byte, status))
            print(f"  Byte {byte_idx}: predicted {byte_pred}, expected {expected_byte} - {status}")

    return results

# Test cases that involve byte 2-3
print("Testing LEA with various immediates:")
test_lea_all_bytes(0)      # AX = BP = 0x10000, byte 2 = 1
test_lea_all_bytes(8)      # AX = 0x10008
test_lea_all_bytes(256)    # AX = 0x10100, byte 1 = 1
test_lea_all_bytes(65535)  # AX = 0x1FFFF, bytes 0-1 = 0xFF
test_lea_all_bytes(65536)  # AX = 0x20000, byte 2 = 2
