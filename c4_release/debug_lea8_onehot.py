"""Debug LEA 8 - check if ALU_LO/AX_CARRY_LO are properly one-hot."""
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

print("LEA 8 - One-Hot Check")
print("=" * 70)
print()

# Capture L8 FFN input
l8_ffn_in = None

def l8_ffn_hook(module, input, output):
    global l8_ffn_in
    if isinstance(input, tuple):
        l8_ffn_in = input[0].detach().clone()
    else:
        l8_ffn_in = input.detach().clone()

model.blocks[8].ffn.register_forward_hook(l8_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l8_ffn_in is not None:
    x = l8_ffn_in[0, pos, :]

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Check ALU_LO
    alu_lo = x[BD.ALU_LO:BD.ALU_LO+16]
    print("ALU_LO (should be one-hot for value 0):")
    print(f"  Values: {alu_lo.tolist()}")
    alu_lo_max = torch.argmax(alu_lo).item()
    alu_lo_vals = alu_lo.tolist()
    print(f"  Max index: {alu_lo_max} (value={alu_lo_vals[alu_lo_max]:.3f})")

    # Check for residual values
    alu_lo_sorted = sorted([(i, v) for i, v in enumerate(alu_lo_vals)], key=lambda x: -x[1])
    print(f"  Top 5 dimensions:")
    for i, (idx, val) in enumerate(alu_lo_sorted[:5]):
        print(f"    [{idx}]: {val:.3f}")

    # Count dimensions > 0.1
    high_dims = [i for i, v in enumerate(alu_lo_vals) if abs(v) > 0.1]
    print(f"  Dimensions with |value| > 0.1: {len(high_dims)} (expected: 1)")
    print()

    # Check AX_CARRY_LO
    ax_carry_lo = x[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    print("AX_CARRY_LO (should be one-hot for value 8):")
    print(f"  Values: {ax_carry_lo.tolist()}")
    ax_carry_lo_max = torch.argmax(ax_carry_lo).item()
    ax_carry_lo_vals = ax_carry_lo.tolist()
    print(f"  Max index: {ax_carry_lo_max} (value={ax_carry_lo_vals[ax_carry_lo_max]:.3f})")

    # Check for residual values
    ax_carry_lo_sorted = sorted([(i, v) for i, v in enumerate(ax_carry_lo_vals)], key=lambda x: -x[1])
    print(f"  Top 5 dimensions:")
    for i, (idx, val) in enumerate(ax_carry_lo_sorted[:5]):
        print(f"    [{idx}]: {val:.3f}")

    # Count dimensions > 0.1
    high_dims = [i for i, v in enumerate(ax_carry_lo_vals) if abs(v) > 0.1]
    print(f"  Dimensions with |value| > 0.1: {len(high_dims)} (expected: 1)")
    print()

    # Analysis
    print("Analysis:")
    print("  If multiple dimensions have high values, the carry detection logic")
    print("  will activate for multiple (a, b) pairs, causing CARRY[0] to be too high.")
    print()

    alu_spread = len([v for v in alu_lo_vals if abs(v) > 0.1])
    ax_carry_spread = len([v for v in ax_carry_lo_vals if abs(v) > 0.1])

    if alu_spread > 1 or ax_carry_spread > 1:
        print(f"  ❌ NOT properly one-hot!")
        print(f"     ALU_LO has {alu_spread} dimensions > 0.1")
        print(f"     AX_CARRY_LO has {ax_carry_spread} dimensions > 0.1")
        print(f"     → {alu_spread * ax_carry_spread} carry units could activate")
        print(f"     → Observed ~7 units, so this matches!")
    else:
        print(f"  ✓ Properly one-hot")
        print(f"     → Issue must be elsewhere (compaction corruption?)")
