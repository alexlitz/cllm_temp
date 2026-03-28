"""Debug LEA 8 - check why carry override activates at marker."""
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

print("LEA 8 - Carry Override Activation Debug")
print("=" * 70)
print()

# Capture L10 FFN input
l10_ffn_input = None

def l10_ffn_hook(module, input, output):
    global l10_ffn_input
    if isinstance(input, tuple):
        l10_ffn_input = input[0].detach().clone()
    else:
        l10_ffn_input = input.detach().clone()

model.blocks[10].ffn.register_forward_hook(l10_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l10_ffn_input is not None:
    x = l10_ffn_input[0, pos, :]

    print(f"Position {pos} (REG_AX marker):")
    print()

    # Check carry override activation conditions
    print("Carry override unit activation conditions:")
    carry_1 = x[BD.CARRY + 1].item()
    is_byte = x[BD.IS_BYTE].item()
    byte_index_0 = x[BD.BYTE_INDEX_0].item()
    h1_ax = x[BD.H1 + 1].item()  # H1[AX_IDX=1]

    print(f"  CARRY[1] (ADD carry): {carry_1:.3f}")
    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  BYTE_INDEX_0: {byte_index_0:.3f}")
    print(f"  H1[AX] (gate): {h1_ax:.3f}")
    print()

    # Compute expected activation
    S = 100.0
    up_expected = S * carry_1 + S * is_byte + S * byte_index_0 - S * 2.5
    gate_expected = h1_ax
    hidden_expected = up_expected * gate_expected if up_expected > 0 else 0  # Simplified silu

    print(f"Expected activation:")
    print(f"  up = {up_expected:.1f}")
    print(f"  gate = {gate_expected:.3f}")
    print(f"  hidden ≈ {hidden_expected:.1f} (simplified)")
    print()

    # Check if it should activate
    print("Analysis:")
    if is_byte < 0.5:
        print("  ✓ IS_BYTE=0: This is a MARKER position, not a byte position")
        print("  → Carry override should NOT activate (requires IS_BYTE=1)")
    else:
        print("  ✗ IS_BYTE=1: This is a byte position")

    if carry_1 > 0.5:
        print(f"  ⚠ CARRY[1]={carry_1:.3f}: ADD carry flag is SET")
        print("  → This means a carry was detected from byte 0 addition")
    else:
        print(f"  ✓ CARRY[1]={carry_1:.3f}: No carry flag")

    if byte_index_0 > 0.5:
        print(f"  ⚠ BYTE_INDEX_0=1: Position is byte 0")
    else:
        print(f"  ✓ BYTE_INDEX_0=0: Not byte 0 position")

    print()

    # The real issue: check if dimensions have huge residual values
    print("Checking for dimension pollution:")
    if abs(carry_1) > 10 or abs(is_byte) > 10 or abs(byte_index_0) > 10:
        print("  ❌ FOUND IT! Dimension has abnormally large value!")
        print("  → This could cause massive 'up' activation even though it should be 0")
        if abs(carry_1) > 10:
            print(f"     CARRY[1] = {carry_1:.1f} (should be 0-1)")
        if abs(is_byte) > 10:
            print(f"     IS_BYTE = {is_byte:.1f} (should be 0-1)")
        if abs(byte_index_0) > 10:
            print(f"     BYTE_INDEX_0 = {byte_index_0:.1f} (should be 0-1)")
    else:
        print("  ✓ All dimensions are in normal range (0-1)")

    # Check other CARRY values
    print()
    print("Other CARRY dimensions:")
    for i in range(4):
        print(f"  CARRY[{i}]: {x[BD.CARRY + i].item():.3f}")
