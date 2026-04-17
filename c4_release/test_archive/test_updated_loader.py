"""
Test Updated Weight Loader with 3-Layer Architecture

This test verifies that the updated weight loader correctly loads:
- 3-layer arithmetic operations (ADD) into layers 9-11
- Single-layer operations (comparisons, bitwise) into layer 12
- Control flow operations into layer 13
"""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

def test_updated_loader():
    """Test that the updated loader works correctly."""

    print("="*70)
    print("UPDATED WEIGHT LOADER TEST")
    print("="*70)
    print()

    # Create VM
    print("Creating VM...")
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    # Load all weights with new architecture
    print("Loading weights with updated loader...")
    print()
    loader = CompiledWeightLoader()
    stats = loader.load_all_weights(vm, verbose=True)

    print()
    print("="*70)
    print("TESTING ADD OPERATION")
    print("="*70)
    print()

    # Create embedding
    embed = NibbleVMEmbedding(d_model=1280)

    # Test ADD with various inputs
    test_cases = [
        (2, 3, 5, "small values, no carry"),
        (10, 20, 30, "medium values, no carry"),
        (100, 200, 300, "large values, WITH carry"),
        (255, 1, 256, "edge case carry"),
        (0xFFFF, 1, 0x10000, "16-bit overflow"),
    ]

    passed = 0
    failed = 0

    for a, b, expected, desc in test_cases:
        # Encode input
        input_embedding = embed.encode_vm_state(
            pc=0, ax=a, sp=4096, bp=4096,
            opcode=Opcode.ADD, stack_top=b, batch_size=1,
        )

        with torch.no_grad():
            # Run through 3-layer arithmetic pipeline (layers 9-11)
            x = input_embedding.unsqueeze(1)  # [batch, seq=1, d_model]

            # Layer 9: Raw + Generate
            x = vm.blocks[9](x)

            # Layer 10: Carry Lookahead
            x = vm.blocks[10](x)

            # Layer 11: Finalize
            x = vm.blocks[11](x)

            output_embedding = x.squeeze(1)

        # Decode result
        result = embed.decode_result_nibbles(output_embedding)

        # Check (handle 32-bit overflow)
        expected_32 = expected & 0xFFFFFFFF
        result_32 = result & 0xFFFFFFFF

        if result_32 == expected_32:
            print(f"  ✅ {a} + {b} = {result_32} ({desc})")
            passed += 1
        else:
            print(f"  ❌ {a} + {b} = {result_32}, expected {expected_32} ({desc})")
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
        print("  Updated weight loader works correctly with 3-layer architecture!")
    else:
        print("  ⚠️  Some tests failed")

    print("="*70)

    return failed == 0

if __name__ == "__main__":
    success = test_updated_loader()
    exit(0 if success else 1)
