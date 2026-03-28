"""Debug LEA 8 - check L9 CARRY[1] write."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

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

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Check at AX marker
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - L9 CARRY[1] Write Debug")
print("=" * 70)
print()

# Capture L8 FFN output (= L9 FFN input) and L9 FFN output
l8_ffn_out = None
l9_ffn_out = None

def l8_ffn_hook(module, input, output):
    global l8_ffn_out
    l8_ffn_out = output.detach().clone()

def l9_ffn_hook(module, input, output):
    global l9_ffn_out
    l9_ffn_out = output.detach().clone()

model.blocks[8].ffn.register_forward_hook(l8_ffn_hook)
model.blocks[9].ffn.register_forward_hook(l9_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker):")
print()

# Check L8 output (should have CARRY[0] for low-nibble carry)
if l8_ffn_out is not None:
    x_l8 = l8_ffn_out[0, pos, :]
    carry_0_l8 = x_l8[BD.CARRY + 0].item()
    carry_1_l8 = x_l8[BD.CARRY + 1].item()

    print(f"After L8 FFN:")
    print(f"  CARRY[0] (lo-nibble carry): {carry_0_l8:.3f}")
    print(f"  CARRY[1] (byte carry): {carry_1_l8:.3f}")

    alu_lo = x_l8[BD.ALU_LO:BD.ALU_LO+16]
    alu_hi = x_l8[BD.ALU_HI:BD.ALU_HI+16]
    ax_carry_lo = x_l8[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = x_l8[BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

    print(f"  ALU: {torch.argmax(alu_lo).item()} + 16*{torch.argmax(alu_hi).item()}")
    print(f"  AX_CARRY: {torch.argmax(ax_carry_lo).item()} + 16*{torch.argmax(ax_carry_hi).item()}")

    op_add = x_l8[BD.OP_ADD].item()
    op_lea = x_l8[BD.OP_LEA].item()
    print(f"  OP_ADD: {op_add:.3f}")
    print(f"  OP_LEA: {op_lea:.3f}")
    print()

# Check L9 output (should write CARRY[1] for byte carry)
if l9_ffn_out is not None:
    x_l9 = l9_ffn_out[0, pos, :]
    carry_0_l9 = x_l9[BD.CARRY + 0].item()
    carry_1_l9 = x_l9[BD.CARRY + 1].item()

    print(f"After L9 FFN:")
    print(f"  CARRY[0]: {carry_0_l9:.3f}")
    print(f"  CARRY[1] (byte carry): {carry_1_l9:.3f}")
    print()

    # Calculate delta
    if l8_ffn_out is not None:
        delta_carry_1 = carry_1_l9 - carry_1_l8
        print(f"L9 FFN wrote {delta_carry_1:.1f} to CARRY[1]")
        print()

        # Expected value for LEA 8 + 0x00010000
        # Lo nibble: 8 + 0 = 8 (no carry)
        # Hi nibble: 0 + 0 + 0 = 0 (no carry)
        # So byte carry should be 0
        print("Expected:")
        print("  LEA 8: AX = 8 + 0x00010000 = 0x00010008")
        print("  Lo nibble: 8 + 0 = 8 (no carry)")
        print("  Hi nibble: 0 + 0 = 0 (no carry)")
        print("  → Byte carry (CARRY[1]) should be 0")
        print()

        if abs(delta_carry_1) > 1.0:
            print(f"  ❌ CARRY[1] = {delta_carry_1:.1f} is way too high!")
            print("  → This causes the carry override unit to activate incorrectly")
            print()
            print("Possible causes:")
            print("  1. Multiple carry detection units activating (should only be one)")
            print("  2. Gate value (OP_ADD + OP_LEA) too high")
            print("  3. Model compaction corrupted weights")
        else:
            print(f"  ✓ CARRY[1] = {delta_carry_1:.3f} is reasonable")
