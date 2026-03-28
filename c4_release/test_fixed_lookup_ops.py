#!/usr/bin/env python3
"""
Test all fixed lookup-table operations with cancel-pair pattern.

Tests:
- Bitwise: OR, XOR, AND
- Arithmetic: MUL, DIV, MOD
- Shifts: SHL, SHR

Expected: All operations should return correct results with new cancel-pair pattern.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode


def test_operation(vm, embed, opcode, name, a, b, expected, layer):
    """Test a single operation.

    Args:
        vm: The VM instance
        embed: The embedding instance
        opcode: The opcode to test
        name: Human-readable name
        a: First operand (stack_top)
        b: Second operand (ax)
        expected: Expected result
        layer: Which layer this operation is in
    """
    # Encode input state
    input_emb = embed.encode_vm_state(
        pc=0, ax=b, sp=0, bp=0,
        opcode=opcode, stack_top=a, batch_size=1
    )

    # Run through specified layer
    with torch.no_grad():
        x = input_emb.unsqueeze(1)
        output = vm.blocks[layer](x)
        result = embed.decode_result_nibbles(output.squeeze(1))

    match = '✓' if result == expected else '✗'
    if result == expected:
        print(f'{match} {name}({a}, {b}) = {result}')
    else:
        print(f'{match} {name}({a}, {b}) = {result} (expected {expected})')

    return result == expected


def main():
    print("=" * 70)
    print("TESTING FIXED LOOKUP-TABLE OPERATIONS")
    print("=" * 70)
    print()

    # Auto-calculate required FFN size
    loader = CompiledWeightLoader()  # Auto-calculates FFN size
    ffn_size = loader.ffn_hidden

    print(f"Creating VM with d_model=1352, ffn_hidden={ffn_size} (auto-calculated)...")
    vm = AutoregressiveVM(d_model=1352, n_layers=16, ffn_hidden=ffn_size)
    vm.eval()
    print("✓ VM created")
    print()

    print("Loading weights with bit-level primitives...")
    loader.load_all_weights(vm, verbose=False)
    print("✓ Weights loaded")
    print()

    embed = NibbleVMEmbedding(d_model=1352)

    all_passed = True

    # Test bitwise operations (each in its own layer)
    print("-" * 70)
    print("BITWISE OPERATIONS")
    print("-" * 70)

    print("OR (Layer 6):")
    or_tests = [
        (Opcode.OR, "OR", 5, 3, 7, 6),
        (Opcode.OR, "OR", 12, 10, 14, 6),
        (Opcode.OR, "OR", 255, 15, 255, 6),
    ]
    for opcode, name, a, b, expected, layer in or_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print("\nXOR (Layer 7):")
    xor_tests = [
        (Opcode.XOR, "XOR", 5, 3, 6, 7),
        (Opcode.XOR, "XOR", 12, 10, 6, 7),
        (Opcode.XOR, "XOR", 255, 255, 0, 7),
    ]
    for opcode, name, a, b, expected, layer in xor_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print("\nAND (Layer 8):")
    and_tests = [
        (Opcode.AND, "AND", 5, 3, 1, 8),
        (Opcode.AND, "AND", 12, 10, 8, 8),
        (Opcode.AND, "AND", 255, 15, 15, 8),
    ]
    for opcode, name, a, b, expected, layer in and_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print()

    # Test arithmetic operations
    print("-" * 70)
    print("ARITHMETIC OPERATIONS")
    print("-" * 70)

    print("DIV (Layer 0):")
    div_tests = [
        (Opcode.DIV, "DIV", 15, 3, 5, 0),
        (Opcode.DIV, "DIV", 120, 10, 12, 0),
        (Opcode.DIV, "DIV", 100, 7, 14, 0),
    ]
    for opcode, name, a, b, expected, layer in div_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print("\nMOD (Layer 1):")
    mod_tests = [
        (Opcode.MOD, "MOD", 15, 3, 0, 1),
        (Opcode.MOD, "MOD", 17, 5, 2, 1),
        (Opcode.MOD, "MOD", 100, 7, 2, 1),
    ]
    for opcode, name, a, b, expected, layer in mod_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print("\nMUL (Layer 2):")
    mul_tests = [
        (Opcode.MUL, "MUL", 5, 3, 15, 2),
        (Opcode.MUL, "MUL", 12, 10, 120, 2),
        (Opcode.MUL, "MUL", 100, 100, 10000, 2),
    ]
    for opcode, name, a, b, expected, layer in mul_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print()

    # Test shifts
    print("-" * 70)
    print("SHIFT OPERATIONS")
    print("-" * 70)

    print("SHL (Layer 3):")
    shl_tests = [
        (Opcode.SHL, "SHL", 5, 1, 10, 3),
        (Opcode.SHL, "SHL", 5, 2, 20, 3),
        (Opcode.SHL, "SHL", 1, 8, 256, 3),
    ]
    for opcode, name, a, b, expected, layer in shl_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print("\nSHR (Layer 4):")
    shr_tests = [
        (Opcode.SHR, "SHR", 10, 1, 5, 4),
        (Opcode.SHR, "SHR", 20, 2, 5, 4),
        (Opcode.SHR, "SHR", 256, 8, 1, 4),
    ]
    for opcode, name, a, b, expected, layer in shr_tests:
        passed = test_operation(vm, embed, opcode, name, a, b, expected, layer)
        all_passed = all_passed and passed

    print()
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
