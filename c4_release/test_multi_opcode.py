"""
Test Multiple Opcodes with Unit Offset Allocation

This test verifies that multiple opcodes can coexist in the same layer
without interfering with each other, now that we have unit offset allocation.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

def test_multi_opcode():
    """Test that multiple opcodes work correctly when loaded together."""

    print("="*70)
    print("MULTI-OPCODE TEST (with Unit Offset Allocation)")
    print("="*70)
    print()

    # Create VM
    print("Creating VM...")
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    # Load ALL opcode weights with unit offsets
    print("Loading all opcode weights...")
    loader = CompiledWeightLoader()
    stats = loader.load_all_weights(vm, verbose=True)

    print()
    print("="*70)
    print("RUNNING TEST CASES")
    print("="*70)
    print()

    # Create embedding
    embed = NibbleVMEmbedding(d_model=1280)

    # Get layer for ALU operations
    layer_idx = loader.layer_map.PRIMARY_ALU
    layer = vm.blocks[layer_idx]

    # Test cases: [(opcode, name, a, b, expected)]
    test_cases = [
        (Opcode.ADD, "ADD", 2, 3, 5),
        (Opcode.ADD, "ADD", 100, 200, 300),
        (Opcode.SUB, "SUB", 10, 3, 7),
        (Opcode.SUB, "SUB", 100, 25, 75),
        (Opcode.MUL, "MUL", 3, 4, 12),
        (Opcode.MUL, "MUL", 10, 10, 100),
        (Opcode.EQ, "EQ", 5, 5, 1),
        (Opcode.EQ, "EQ", 5, 6, 0),
        (Opcode.NE, "NE", 5, 6, 1),
        (Opcode.NE, "NE", 5, 5, 0),
        (Opcode.LT, "LT", 3, 5, 1),
        (Opcode.LT, "LT", 5, 3, 0),
        (Opcode.OR, "OR", 0b1100, 0b1010, 0b1110),
        (Opcode.XOR, "XOR", 0b1100, 0b1010, 0b0110),
        (Opcode.AND, "AND", 0b1100, 0b1010, 0b1000),
    ]

    passed = 0
    failed = 0

    for opcode, name, a, b, expected in test_cases:
        # Encode input
        input_embedding = embed.encode_vm_state(
            pc=0, ax=a, sp=4096, bp=4096,
            opcode=opcode, stack_top=b, batch_size=1,
        )

        with torch.no_grad():
            # FFN forward with residual
            up = layer.ffn.W_up @ input_embedding.T + layer.ffn.b_up.unsqueeze(1)
            gate = layer.ffn.W_gate @ input_embedding.T + layer.ffn.b_gate.unsqueeze(1)
            hidden = torch.nn.functional.silu(up) * gate
            delta = layer.ffn.W_down @ hidden + layer.ffn.b_down.unsqueeze(1)
            output = input_embedding.T + delta
            output_embedding = output.T

        # Decode result
        result = embed.decode_result_nibbles(output_embedding)

        # Check result
        if result == expected:
            print(f"  ✅ {name:5s} {a:6d} op {b:6d} = {result:6d} (expected {expected:6d})")
            passed += 1
        else:
            print(f"  ❌ {name:5s} {a:6d} op {b:6d} = {result:6d} (expected {expected:6d})")
            failed += 1

    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Passed: {passed}/{len(test_cases)}")
    print(f"  Failed: {failed}/{len(test_cases)}")
    print()

    if failed == 0:
        print("  🎉 ALL TESTS PASSED!")
        print("  Unit offset allocation is working correctly!")
    else:
        print("  ⚠️  Some tests failed - may need to adjust unit allocations")

    print("="*70)

    return failed == 0

if __name__ == "__main__":
    success = test_multi_opcode()
    exit(0 if success else 1)
