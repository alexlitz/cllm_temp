"""Debug LEA 8 - check why L8 CARRY[0] is so high."""
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

print("LEA 8 - L8 CARRY[0] Debug")
print("=" * 70)
print()

# Capture L8 input and output
l8_ffn_in = None
l8_ffn_out = None

def l8_ffn_hook(module, input, output):
    global l8_ffn_in, l8_ffn_out
    if isinstance(input, tuple):
        l8_ffn_in = input[0].detach().clone()
    else:
        l8_ffn_in = input.detach().clone()
    l8_ffn_out = output.detach().clone()

model.blocks[8].ffn.register_forward_hook(l8_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker):")
print()

if l8_ffn_in is not None:
    x_in = l8_ffn_in[0, pos, :]
    carry_0_in = x_in[BD.CARRY + 0].item()
    carry_1_in = x_in[BD.CARRY + 1].item()

    print("Before L8 FFN:")
    print(f"  CARRY[0]: {carry_0_in:.3f}")
    print(f"  CARRY[1]: {carry_1_in:.3f}")

    alu_lo = x_in[BD.ALU_LO:BD.ALU_LO+16]
    alu_hi = x_in[BD.ALU_HI:BD.ALU_HI+16]
    ax_carry_lo = x_in[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = x_in[BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

    print(f"  ALU: {torch.argmax(alu_lo).item()} + 16*{torch.argmax(alu_hi).item()}")
    print(f"  AX_CARRY: {torch.argmax(ax_carry_lo).item()} + 16*{torch.argmax(ax_carry_hi).item()}")

    op_add = x_in[BD.OP_ADD].item()
    op_lea = x_in[BD.OP_LEA].item()
    mark_ax = x_in[BD.MARK_AX].item()
    print(f"  OP_ADD: {op_add:.3f}")
    print(f"  OP_LEA: {op_lea:.3f}")
    print(f"  MARK_AX: {mark_ax:.3f}")
    print()

if l8_ffn_out is not None:
    x_out = l8_ffn_out[0, pos, :]
    carry_0_out = x_out[BD.CARRY + 0].item()
    carry_1_out = x_out[BD.CARRY + 1].item()

    print("After L8 FFN:")
    print(f"  CARRY[0]: {carry_0_out:.3f}")
    print(f"  CARRY[1]: {carry_1_out:.3f}")
    print()

    if l8_ffn_in is not None:
        delta_carry_0 = carry_0_out - carry_0_in
        print(f"L8 FFN wrote {delta_carry_0:.1f} to CARRY[0]")
        print()

        # Check carry detection logic
        # L8 ADD lo-nibble carry: activates when ALU_LO + AX_CARRY_LO >= 16
        alu_lo_val = torch.argmax(x_in[BD.ALU_LO:BD.ALU_LO+16]).item()
        ax_carry_lo_val = torch.argmax(x_in[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]).item()

        print(f"Carry detection check:")
        print(f"  ALU_LO = {alu_lo_val}")
        print(f"  AX_CARRY_LO = {ax_carry_lo_val}")
        print(f"  Sum = {alu_lo_val + ax_carry_lo_val}")

        if alu_lo_val + ax_carry_lo_val >= 16:
            print(f"  → Carry SHOULD activate (sum >= 16)")
            print(f"  → Expected CARRY[0] ≈ 1.0")
        else:
            print(f"  → Carry should NOT activate (sum < 16)")
            print(f"  → Expected CARRY[0] ≈ 0.0")

        print()

        if abs(delta_carry_0) > 10:
            print(f"  ❌ CARRY[0] delta = {delta_carry_0:.1f} is way too high!")
            print("  → Expected ~1 for carry, ~0 for no carry")
            print()
            print("Root cause investigation:")
            print("  1. Check gate value (OP_ADD + OP_LEA)")
            gate_val = op_add + op_lea
            print(f"     Gate = OP_ADD + OP_LEA = {op_add:.3f} + {op_lea:.3f} = {gate_val:.3f}")
            if abs(gate_val) > 10:
                print(f"     ❌ Gate = {gate_val:.3f} is abnormally high!")
            else:
                print(f"     ✓ Gate = {gate_val:.3f} is reasonable")
            print()

            print("  2. Check if multiple units are activating")
            print("     Expected: 1 unit if carry, 0 units if no carry")
            print(f"     Observed: CARRY[0] = {delta_carry_0:.1f}")
            print(f"     Number of units ≈ {delta_carry_0 / gate_val:.0f} (if each contributes ~gate)")
