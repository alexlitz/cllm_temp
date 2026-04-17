#!/usr/bin/env python3
"""Debug SP byte 2 and 3 mismatches for LEA."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
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

print("Initializing model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# LEA 8
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)

# Position 11 predicts position 12 (SP byte 1 → SP byte 2)
# Position 12 predicts position 13 (SP byte 2 → SP byte 3)
pos_11 = ctx_len + 11  # SP byte 1 position
pos_12 = ctx_len + 12  # SP byte 2 position

print(f"\nContext length: {ctx_len}")
print(f"SP marker at draft index 10, seq pos {ctx_len + 10}")
print(f"SP byte 1 at draft index 11, seq pos {pos_11} (predicts SP byte 2)")
print(f"SP byte 2 at draft index 12, seq pos {pos_12} (predicts SP byte 3)")
print()

# Forward pass with layer-by-layer inspection
with torch.no_grad():
    emb = model.embed(ctx_tensor)
    x = emb

    for i, block in enumerate(model.blocks):
        x = block(x)

        if i == 14:  # After L14 (before L15 FFN)
            print("=" * 60)
            print(f"After L14 (before L15 FFN):")
            for pos, name in [(pos_11, "SP byte 1"), (pos_12, "SP byte 2")]:
                print(f"\n  At {name} (pos {pos}):")

                # Check key dimensions
                mark_sp = x[0, pos, BD.MARK_SP].item()
                h1_sp = x[0, pos, BD.H1 + 2].item()  # SP is register 2
                is_byte = x[0, pos, BD.IS_BYTE].item()
                byte_idx_0 = x[0, pos, BD.BYTE_INDEX_0].item()
                byte_idx_1 = x[0, pos, BD.BYTE_INDEX_1].item()
                byte_idx_2 = x[0, pos, BD.BYTE_INDEX_2].item()
                byte_idx_3 = x[0, pos, BD.BYTE_INDEX_3].item()
                has_se = x[0, pos, BD.HAS_SE].item()

                # Check any OUTPUT_LO/HI values
                output_lo = [x[0, pos, BD.OUTPUT_LO + k].item() for k in range(16)]
                output_hi = [x[0, pos, BD.OUTPUT_HI + k].item() for k in range(16)]

                print(f"    MARK_SP={mark_sp:.2f}, H1[SP]={h1_sp:.2f}")
                print(f"    IS_BYTE={is_byte:.2f}")
                print(f"    BYTE_INDEX: [{byte_idx_0:.2f}, {byte_idx_1:.2f}, {byte_idx_2:.2f}, {byte_idx_3:.2f}]")
                print(f"    HAS_SE={has_se:.2f}")
                print(f"    OUTPUT_LO (before L15): {[f'{v:.1f}' for v in output_lo]}")
                print(f"    OUTPUT_HI (before L15): {[f'{v:.1f}' for v in output_hi]}")

                # Check CMP dimensions
                cmp = [x[0, pos, BD.CMP + k].item() for k in range(8)]
                print(f"    CMP: {[f'{v:.2f}' for v in cmp]}")

                # Check what might be contributing to OUTPUT
                psh_at_sp = x[0, pos, BD.PSH_AT_SP].item()
                print(f"    PSH_AT_SP={psh_at_sp:.2f}")

        if i == 15:  # After L15
            print("\n" + "=" * 60)
            print(f"After L15 FFN:")
            for pos, name, expected in [(pos_11, "SP byte 1", 0), (pos_12, "SP byte 2", 1)]:
                print(f"\n  At {name} (pos {pos}), expecting byte {expected}:")

                output_lo = [x[0, pos, BD.OUTPUT_LO + k].item() for k in range(16)]
                output_hi = [x[0, pos, BD.OUTPUT_HI + k].item() for k in range(16)]

                print(f"    OUTPUT_LO: {[f'{v:.1f}' for v in output_lo]}")
                print(f"    OUTPUT_HI: {[f'{v:.1f}' for v in output_hi]}")

                max_lo = max(output_lo)
                max_hi = max(output_hi)
                max_lo_idx = output_lo.index(max_lo)
                max_hi_idx = output_hi.index(max_hi)
                pred_token = max_lo_idx + 16 * max_hi_idx
                print(f"    Predicted: {pred_token} (expected: {expected})")

                # Calculate what expected values would need
                exp_lo = expected % 16
                exp_hi = expected // 16
                print(f"    Expected nibbles: lo={exp_lo}, hi={exp_hi}")
                print(f"    Actual max: lo[{max_lo_idx}]={max_lo:.1f}, hi[{max_hi_idx}]={max_hi:.1f}")
