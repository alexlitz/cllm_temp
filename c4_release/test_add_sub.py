"""
Test ADD and SUB with Updated Loader

Verifies that both ADD and SUB operations work correctly with the
3-layer architecture.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

def test_add_sub():
    """Test ADD and SUB operations."""

    print("="*70)
    print("ADD & SUB TEST")
    print("="*70)
    print()

    # Create VM
    print("Creating VM...")
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    # Load all weights
    print("Loading weights...")
    print()
    loader = CompiledWeightLoader()
    stats = loader.load_all_weights(vm, verbose=True)

    print()
    print("="*70)
    print("TESTING OPERATIONS")
    print("="*70)
    print()

    # Create embedding
    embed = NibbleVMEmbedding(d_model=1280)

    # Test cases: (opcode, name, a, b, expected, desc)
    test_cases = [
        # ADD tests
        (Opcode.ADD, "ADD", 2, 3, 5, "small values, no carry"),
        (Opcode.ADD, "ADD", 100, 200, 300, "large values, with carry"),
        (Opcode.ADD, "ADD", 255, 1, 256, "edge case carry"),
        (Opcode.ADD, "ADD", 0xFFFF, 1, 0x10000, "16-bit overflow"),

        # SUB tests
        (Opcode.SUB, "SUB", 5, 3, 2, "small values, no borrow"),
        (Opcode.SUB, "SUB", 10, 7, 3, "medium values, no borrow"),
        (Opcode.SUB, "SUB", 300, 200, 100, "large values, with borrow"),
        (Opcode.SUB, "SUB", 256, 1, 255, "edge case borrow"),
        (Opcode.SUB, "SUB", 100, 200, 0xFFFFFF9C, "underflow (wraps)"),

        # Edge cases
        (Opcode.ADD, "ADD", 0, 0, 0, "zero + zero"),
        (Opcode.SUB, "SUB", 0, 0, 0, "zero - zero"),
        (Opcode.ADD, "ADD", 123456, 654321, 777777, "random large values"),
        (Opcode.SUB, "SUB", 654321, 123456, 530865, "random large values"),
    ]

    passed = 0
    failed = 0

    for opcode, op_name, a, b, expected, desc in test_cases:
        # Encode input
        input_embedding = embed.encode_vm_state(
            pc=0, ax=a, sp=4096, bp=4096,
            opcode=opcode, stack_top=b, batch_size=1,
        )

        with torch.no_grad():
            # Run through 3-layer arithmetic pipeline (layers 9-11)
            x = input_embedding.unsqueeze(1)  # [batch, seq=1, d_model]

            # Layer 9: Raw + Generate
            x = vm.blocks[9](x)

            # Layer 10: Carry/Borrow Lookahead
            x = vm.blocks[10](x)

            # Layer 11: Finalize
            x = vm.blocks[11](x)

            output_embedding = x.squeeze(1)

        # Decode result
        result = embed.decode_result_nibbles(output_embedding)

        # Check (handle 32-bit wrap)
        expected_32 = expected & 0xFFFFFFFF
        result_32 = result & 0xFFFFFFFF

        if result_32 == expected_32:
            print(f"  ✅ {op_name:3s} {a:10d} {op_name[0].lower()} {b:10d} = {result_32:10d} ({desc})")
            passed += 1
        else:
            print(f"  ❌ {op_name:3s} {a:10d} {op_name[0].lower()} {b:10d} = {result_32:10d}, expected {expected_32:10d} ({desc})")
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
        print("  Both ADD and SUB work correctly with 3-layer architecture!")
    else:
        print("  ⚠️  Some tests failed")

    print("="*70)

    return failed == 0

if __name__ == "__main__":
    success = test_add_sub()
    exit(0 if success else 1)
