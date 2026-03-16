#!/usr/bin/env python3
"""
Simple test for V3 ValueEncodedBasicOps layer.
Tests that ADD computes correctly with baked weights.
"""
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')

import torch
from pure_gen_vm_v3 import ValueEncodedBasicOps, EmbedDimsV3
from pure_gen_vm import Opcode

print("=" * 60)
print("Testing V3 ValueEncodedBasicOps")
print("=" * 60)

E = EmbedDimsV3
DIM = E.DIM

def create_test_input(a_val, b_val, opcode):
    """Create input tensor with one-hot encoding for a, b, and opcode."""
    x = torch.zeros(1, 1, DIM)

    # Set all 8 nibbles for operand a (a_val in nibble 0, rest 0)
    for nib in range(8):
        if nib == 0:
            x[0, 0, E.OP_A_NIB_START + nib * 16 + a_val] = 1.0
        else:
            x[0, 0, E.OP_A_NIB_START + nib * 16 + 0] = 1.0

    # Set all 8 nibbles for operand b (b_val in nibble 0, rest 0)
    for nib in range(8):
        if nib == 0:
            x[0, 0, E.OP_B_NIB_START + nib * 16 + b_val] = 1.0
        else:
            x[0, 0, E.OP_B_NIB_START + nib * 16 + 0] = 1.0

    # Set opcode one-hot
    x[0, 0, E.OPCODE_START + opcode] = 1.0

    return x

# Create layer
layer = ValueEncodedBasicOps(DIM)

# Test cases
test_cases = [
    (5, 3, Opcode.ADD, 8, "ADD"),   # 5 + 3 = 8
    (7, 8, Opcode.ADD, 15, "ADD"),  # 7 + 8 = 15
    (9, 9, Opcode.ADD, 2, "ADD"),   # 9 + 9 = 18 = 0x12 (nibble 0 = 2, carry = 1)
    (10, 3, Opcode.SUB, 7, "SUB"),  # 10 - 3 = 7
    (5, 10, Opcode.SUB, 11, "SUB"), # 5 - 10 = -5 = 16 - 5 = 11 (with borrow)
    (0xF, 0x5, Opcode.AND, 0x5, "AND"),
    (0xA, 0x5, Opcode.OR, 0xF, "OR"),
    (0xF, 0xA, Opcode.XOR, 0x5, "XOR"),
]

print(f"\nEmbedding positions:")
print(f"  OP_A_NIB_START: {E.OP_A_NIB_START}")
print(f"  OP_B_NIB_START: {E.OP_B_NIB_START}")
print(f"  OPCODE_START: {E.OPCODE_START}")
print(f"  RESULT_VAL_START: {E.RESULT_VAL_START}")
print(f"\nOpcode values: ADD={Opcode.ADD}, SUB={Opcode.SUB}, AND={Opcode.AND}, OR={Opcode.OR}, XOR={Opcode.XOR}")
print()

all_pass = True
for a, b, op, expected, op_name in test_cases:

    x = create_test_input(a, b, op)

    with torch.no_grad():
        out = layer(x)

    # Read value-encoded result for nibble 0
    result_val = out[0, 0, E.RESULT_VAL_START].item()

    # Convert back to integer: val = result * NIBBLE_SCALE, so result = val / NIBBLE_SCALE
    decoded = result_val / E.NIBBLE_SCALE
    decoded_int = round(decoded)

    passed = decoded_int == expected
    if not passed:
        all_pass = False

    carry_val = out[0, 0, E.CARRY_VAL_START].item() if op == Opcode.ADD else 0

    print(f"  {op_name}: {a} op {b} = {expected}")
    print(f"    Result val: {result_val:.6f}")
    print(f"    Decoded: {decoded:.2f} → {decoded_int}")
    print(f"    Carry: {carry_val:.4f}")
    print(f"    {'PASS' if passed else 'FAIL'}")
    print()

# Debug: trace through for 5+3
print("\n" + "=" * 60)
print("DEBUG: Tracing 5 + 3")
print("=" * 60)

x = create_test_input(5, 3, Opcode.ADD)

# Check input setup
a_dim = E.OP_A_NIB_START + 0 * 16 + 5
b_dim = E.OP_B_NIB_START + 0 * 16 + 3
op_dim = E.OPCODE_START + Opcode.ADD

print(f"Input x[{a_dim}] (a[0]=5): {x[0, 0, a_dim].item()}")
print(f"Input x[{b_dim}] (b[0]=3): {x[0, 0, b_dim].item()}")
print(f"Input x[{op_dim}] (opcode=ADD): {x[0, 0, op_dim].item()}")

# Compute intermediate values
with torch.no_grad():
    up_out = layer.up(x)
    gate_out = layer.gate(x)

# Row for (nib=0, a=5, b=3) is: 0 * 256 + 5 * 16 + 3 = 83
row = 83
print(f"\nRow {row} (nib=0, a=5, b=3):")
print(f"  up[{row}]: {up_out[0, 0, row].item():.4f}")
print(f"  gate[{row}]: {gate_out[0, 0, row].item():.4f}")
print(f"  silu(up): {torch.nn.functional.silu(up_out[0, 0, row:row+1]).item():.4f}")
print(f"  hidden = silu(up) * gate: {(torch.nn.functional.silu(up_out[0, 0, row:row+1]) * gate_out[0, 0, row]).item():.4f}")

# Check gate and up weights for this row
print(f"\nWeights for row {row}:")
print(f"  gate.weight[{row}, {a_dim}]: {layer.gate.weight[row, a_dim].item():.4f}")
print(f"  up.weight[{row}, {b_dim}]: {layer.up.weight[row, b_dim].item():.4f}")
print(f"  up.weight[{row}, {op_dim}]: {layer.up.weight[row, op_dim].item():.4f}")
print(f"  up.bias[{row}]: {layer.up.bias[row].item():.4f}")

result_dim = E.RESULT_VAL_START + 0
print(f"  down.weight[{result_dim}, {row}]: {layer.down.weight[result_dim, row].item():.6f}")

print("\n" + "=" * 60)
print(f"Overall: {'ALL TESTS PASSED!' if all_pass else 'SOME TESTS FAILED'}")
print("=" * 60)
