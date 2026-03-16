#!/usr/bin/env python3
"""
Full test for V3 ValueEncodedBasicOps layer.
Tests multi-nibble operations (32-bit values).
"""
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')

import torch
from pure_gen_vm_v3 import ValueEncodedBasicOps, EmbedDimsV3
from pure_gen_vm import Opcode

print("=" * 60)
print("Testing V3 ValueEncodedBasicOps - Multi-Nibble")
print("=" * 60)

E = EmbedDimsV3
DIM = E.DIM

def int_to_nibbles(v):
    """Convert integer to list of 8 nibbles (LSB first)."""
    return [(v >> (i * 4)) & 0xF for i in range(8)]

def nibbles_to_int(nibs):
    """Convert 8 nibbles to integer."""
    v = 0
    for i, n in enumerate(nibs):
        v |= (n & 0xF) << (i * 4)
    return v

def create_test_input(a, b, opcode):
    """Create input tensor with one-hot encoding for a, b, and opcode."""
    x = torch.zeros(1, 1, DIM)

    a_nibs = int_to_nibbles(a)
    b_nibs = int_to_nibbles(b)

    # Set one-hot for each nibble
    for nib in range(8):
        x[0, 0, E.OP_A_NIB_START + nib * 16 + a_nibs[nib]] = 1.0
        x[0, 0, E.OP_B_NIB_START + nib * 16 + b_nibs[nib]] = 1.0

    # Set opcode one-hot
    x[0, 0, E.OPCODE_START + opcode] = 1.0

    return x

def read_result(out):
    """Read value-encoded result and convert to integer."""
    result_nibs = []
    for nib in range(8):
        val = out[0, 0, E.RESULT_VAL_START + nib].item()
        # Convert from scaled (0-1) back to nibble (0-15)
        decoded = val / E.NIBBLE_SCALE
        result_nibs.append(max(0, min(15, round(decoded))))
    return nibbles_to_int(result_nibs)

# Create layer
layer = ValueEncodedBasicOps(DIM)

# Test cases with expected results
tests = [
    # ADD tests
    ("ADD", Opcode.ADD, 0x12, 0x34, 0x46),
    ("ADD", Opcode.ADD, 0xFF, 0x01, 0x100),  # Carry test
    ("ADD", Opcode.ADD, 0x1234, 0x5678, 0x68AC),
    ("ADD", Opcode.ADD, 0x12345678, 0x11111111, 0x23456789),

    # SUB tests
    ("SUB", Opcode.SUB, 0x100, 0x01, 0xFF),  # Borrow test
    ("SUB", Opcode.SUB, 0x5678, 0x1234, 0x4444),
    ("SUB", Opcode.SUB, 0x12345678, 0x11111111, 0x01234567),

    # AND tests
    ("AND", Opcode.AND, 0xFF, 0x55, 0x55),
    ("AND", Opcode.AND, 0xFFFF, 0x5555, 0x5555),
    ("AND", Opcode.AND, 0x12345678, 0xF0F0F0F0, 0x10305070),

    # OR tests
    ("OR", Opcode.OR, 0xAA, 0x55, 0xFF),
    ("OR", Opcode.OR, 0xAAAA, 0x5555, 0xFFFF),
    ("OR", Opcode.OR, 0x12340000, 0x00005678, 0x12345678),

    # XOR tests
    ("XOR", Opcode.XOR, 0xFF, 0xAA, 0x55),
    ("XOR", Opcode.XOR, 0xFFFF, 0xAAAA, 0x5555),
    ("XOR", Opcode.XOR, 0x12345678, 0x12345678, 0),  # Self XOR = 0
]

passed = 0
failed = 0
failures = []

for name, op, a, b, expected in tests:
    x = create_test_input(a, b, op)

    with torch.no_grad():
        out = layer(x)

    result = read_result(out)

    # For multi-nibble values, need to handle carry propagation
    # The current layer computes nibble-wise, carry is stored but not propagated
    # For now, compare nibble-by-nibble ignoring carry

    ok = result == expected

    if ok:
        passed += 1
        print(f"  PASS: {name}: {hex(a)} op {hex(b)} = {hex(expected)}")
    else:
        failed += 1
        failures.append((name, a, b, expected, result))
        print(f"  FAIL: {name}: {hex(a)} op {hex(b)}")
        print(f"    Expected: {hex(expected)}")
        print(f"    Got:      {hex(result)}")
        # Show nibble comparison
        exp_nibs = int_to_nibbles(expected)
        got_nibs = int_to_nibbles(result)
        for i in range(8):
            if exp_nibs[i] != got_nibs[i]:
                print(f"    Nibble {i}: expected {exp_nibs[i]}, got {got_nibs[i]}")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} passed")
if failures:
    print("\nNote: Carry propagation between nibbles requires attention layer")
    print("The basic ops layer computes each nibble independently.")
print("=" * 60)
