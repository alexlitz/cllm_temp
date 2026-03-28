#!/usr/bin/env python3
"""Test operations with shared layers."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader_v2 import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

# Create and load VM
loader = CompiledWeightLoader(use_layer_sharing=True, verbose=False)
vm = AutoregressiveVM(d_model=1352, n_layers=loader.n_layers, ffn_hidden=loader.ffn_hidden)
vm.eval()
loader.load_all_weights(vm)

embed = NibbleVMEmbedding(d_model=1352)

print("=" * 70)
print("TESTING OPERATIONS WITH SHARED LAYERS")
print("=" * 70)
print()

def test_op(opcode, name, a, b, expected):
    """Test a single operation."""
    input_emb = embed.encode_vm_state(
        pc=0, ax=b, sp=0, bp=0,
        opcode=opcode, stack_top=a, batch_size=1
    )

    with torch.no_grad():
        x = input_emb.unsqueeze(1)
        # Run through ALL layers (multi-layer operations need full pipeline)
        for layer in vm.blocks:
            x = layer(x)
        result = embed.decode_result_nibbles(x.squeeze(1))

    match = '✓' if result == expected else '✗'
    status = "PASS" if result == expected else f"FAIL (got {result})"
    print(f"{match} {name:10s}({a:5d}, {b:5d}) = {expected:8d}  [{status}]")
    return result == expected

# Test all operations
all_passed = True

print("Bitwise Operations:")
all_passed &= test_op(Opcode.OR, "OR", 5, 3, 7)
all_passed &= test_op(Opcode.OR, "OR", 12, 10, 14)
all_passed &= test_op(Opcode.XOR, "XOR", 5, 3, 6)
all_passed &= test_op(Opcode.XOR, "XOR", 12, 10, 6)
all_passed &= test_op(Opcode.AND, "AND", 5, 3, 1)
all_passed &= test_op(Opcode.AND, "AND", 12, 10, 8)

print("\nArithmetic Operations:")
all_passed &= test_op(Opcode.ADD, "ADD", 5, 3, 8)
all_passed &= test_op(Opcode.ADD, "ADD", 100, 200, 300)
all_passed &= test_op(Opcode.SUB, "SUB", 10, 3, 7)
all_passed &= test_op(Opcode.SUB, "SUB", 200, 50, 150)
all_passed &= test_op(Opcode.MUL, "MUL", 5, 3, 15)
all_passed &= test_op(Opcode.MUL, "MUL", 12, 10, 120)

print("\nShift Operations:")
all_passed &= test_op(Opcode.SHL, "SHL", 5, 1, 10)
all_passed &= test_op(Opcode.SHL, "SHL", 5, 2, 20)
all_passed &= test_op(Opcode.SHR, "SHR", 10, 1, 5)
all_passed &= test_op(Opcode.SHR, "SHR", 20, 2, 5)

print()
print("=" * 70)
if all_passed:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED")
print("=" * 70)
