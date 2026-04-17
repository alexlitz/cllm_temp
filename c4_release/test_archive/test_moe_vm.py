#!/usr/bin/env python3
"""Test MoE VM with opcode-based expert routing."""

import torch
from neural_vm.moe_vm import MoEAutoregressiveVM
from neural_vm.moe_weight_loader import MoEWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

print("=" * 70)
print("TESTING MoE VM WITH OPCODE-ROUTED EXPERTS")
print("=" * 70)
print()

# Create weight loader and expert configs
loader = MoEWeightLoader(verbose=True)
experts_per_layer = loader.create_expert_configs()

print()
print("Creating MoE VM...")
vm = MoEAutoregressiveVM(
    d_model=1352,
    n_layers=loader.n_layers,
    n_heads=8,
    experts_per_layer=experts_per_layer
)
vm.eval()

print(f"✓ VM created with {loader.n_layers} layers")
print()

# Test operations
embed = NibbleVMEmbedding(d_model=1352)

def test_op(opcode, name, a, b, expected):
    """Test a single operation."""
    input_emb = embed.encode_vm_state(
        pc=0, ax=b, sp=0, bp=0,
        opcode=opcode, stack_top=a, batch_size=1
    )

    with torch.no_grad():
        x = input_emb.unsqueeze(1)
        output = vm(x)
        result = embed.decode_result_nibbles(output.squeeze(1))

    match = '✓' if result == expected else '✗'
    status = "PASS" if result == expected else f"FAIL (got {result})"
    print(f"{match} {name:10s}({a:5d}, {b:5d}) = {expected:8d}  [{status}]")
    return result == expected

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
    print("✓ ALL TESTS PASSED - MoE routing works perfectly!")
else:
    print("✗ SOME TESTS FAILED")
print("=" * 70)
