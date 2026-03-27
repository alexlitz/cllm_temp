#!/usr/bin/env python3
"""Debug fp32 DIV floor extraction failures."""

import torch
from neural_vm.alu.ops.div import build_div_layers, build_div_layers_fp32
from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.common import GenericE

def test_div_case(layers, a, b, ge, opcode, label=""):
    """Test a single DIV case and return (expected, actual, passed)."""
    N = ge.NUM_POSITIONS
    base = ge.BASE

    # Build input tensor
    x = torch.zeros(1, N, ge.DIM, dtype=torch.float32)

    # Set opcode active
    x[:, 0, ge.OP_START + opcode] = 1.0

    # Encode operands as nibbles
    for i in range(N):
        nib_a = (a >> (i * ge.config.chunk_bits)) & (base - 1)
        nib_b = (b >> (i * ge.config.chunk_bits)) & (base - 1)
        x[:, i, ge.NIB_A] = float(nib_a)
        x[:, i, ge.NIB_B] = float(nib_b)

    # Run pipeline
    y = x
    for layer in layers:
        y = layer(y)

    # Extract result
    result = 0
    for i in range(N):
        nib = int(round(y[0, i, ge.RESULT].item()))
        nib = max(0, min(base - 1, nib))
        result += nib * (base ** i)

    expected = a // b if b > 0 else 0
    passed = (result == expected)

    return expected, result, passed

def main():
    ge = GenericE(NIBBLE)
    opcode = 31

    # Build both versions
    layers_fp64 = build_div_layers(NIBBLE, opcode, fp32_floor=False)
    layers_fp32 = build_div_layers_fp32(NIBBLE, opcode)

    # Test same cases that were used in the test
    import random
    random.seed(42)

    failures_fp32 = []

    for _ in range(50):
        a = random.randint(1, 0xFFFFFFFF)
        b = random.randint(1, 0xFFFF)  # Divisor 16-bit

        exp64, res64, pass64 = test_div_case(layers_fp64, a, b, ge, opcode, "fp64")
        exp32, res32, pass32 = test_div_case(layers_fp32, a, b, ge, opcode, "fp32")

        if not pass32:
            failures_fp32.append({
                'a': a,
                'b': b,
                'expected': exp32,
                'got_fp32': res32,
                'got_fp64': res64,
                'quotient_float': a / b,
            })

    print(f"fp32 failures: {len(failures_fp32)}/50\n")

    for f in failures_fp32:
        print(f"a={f['a']}, b={f['b']}")
        print(f"  expected: {f['expected']}")
        print(f"  got_fp32: {f['got_fp32']}")
        print(f"  got_fp64: {f['got_fp64']}")
        print(f"  Q_float:  {f['quotient_float']:.6f}")
        print()

    # Detailed analysis of first failure
    if failures_fp32:
        f = failures_fp32[0]
        print("=== Detailed analysis of first failure ===")
        analyze_case(layers_fp32, f['a'], f['b'], ge, opcode)

def analyze_case(layers, a, b, ge, opcode):
    """Detailed analysis of a single case."""
    N = ge.NUM_POSITIONS
    base = ge.BASE

    print(f"a={a} (0x{a:08x}), b={b} (0x{b:04x})")
    print(f"expected quotient: {a // b}")
    print(f"float quotient: {a / b}")
    print()

    # Build input tensor
    x = torch.zeros(1, N, ge.DIM, dtype=torch.float32)
    x[:, 0, ge.OP_START + opcode] = 1.0

    for i in range(N):
        nib_a = (a >> (i * ge.config.chunk_bits)) & (base - 1)
        nib_b = (b >> (i * ge.config.chunk_bits)) & (base - 1)
        x[:, i, ge.NIB_A] = float(nib_a)
        x[:, i, ge.NIB_B] = float(nib_b)

    # Run through each layer
    y = x
    for i, layer in enumerate(layers):
        y = layer(y)
        print(f"After layer {i} ({type(layer).__name__}):")

        if hasattr(layer, 'reciprocal') or 'Merged' in type(layer).__name__:
            # After layer 1: check SLOT_DIVIDEND and SLOT_QUOTIENT (reciprocal)
            dividend = y[0, 0, ge.SLOT_DIVIDEND].item()
            reciprocal = y[0, 0, ge.SLOT_QUOTIENT].item()
            print(f"  SLOT_DIVIDEND: {dividend}")
            print(f"  SLOT_QUOTIENT (reciprocal): {reciprocal}")
            print(f"  1/b expected: {1/b}")

        if 'Multiply' in type(layer).__name__:
            # After layer 2: check Q_float
            q_float = y[0, 0, ge.SLOT_REMAINDER].item()
            print(f"  SLOT_REMAINDER (Q_float): {q_float}")
            print(f"  Expected Q_float: {a/b}")

        if 'Floor' in type(layer).__name__:
            # After floor: check RESULT at each position
            print("  Floor values (RESULT before ChunkSubtract):")
            for j in range(N):
                floor_j = y[0, j, ge.RESULT].item()
                expected_floor_j = int(a // b) // (base ** j)
                print(f"    RESULT[{j}]: {floor_j:.4f} (expected floor: {expected_floor_j})")

        if 'ChunkSubtract' in type(layer).__name__:
            # After chunk subtract: final nibble values
            print("  Final nibble values:")
            for j in range(N):
                nib = y[0, j, ge.RESULT].item()
                expected_nib = (a // b >> (j * ge.config.chunk_bits)) & (base - 1)
                print(f"    RESULT[{j}]: {nib:.4f} (expected: {expected_nib})")

        print()

if __name__ == "__main__":
    main()
