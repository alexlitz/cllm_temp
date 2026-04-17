"""
Test ADD with Extracted Weights in AutoregressiveVM format

This test verifies that we can extract weights from the real 3-layer ADD
implementation and use them in the AutoregressiveVM architecture.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.alu_weight_extractor import ALUWeightExtractor
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E

def test_extracted_add():
    """Test ADD with weights extracted from real implementation."""

    print("="*70)
    print("EXTRACTED ADD TEST (3-Layer in AutoregressiveVM)")
    print("="*70)
    print()

    # Create VM
    print("Creating VM...")
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    # Extract ADD weights
    print("Extracting ADD weights from real implementation...")
    extractor = ALUWeightExtractor()
    three_layer = extractor.extract_add_weights(opcode=Opcode.ADD)

    # Load weights into layers 9, 10, 11
    print(f"Loading into layers 9-11...")
    print(f"  Layer 9 (raw+gen): {three_layer.layer1['W_up'].shape}")
    print(f"  Layer 10 (carry):  {three_layer.layer2['W_up'].shape}")
    print(f"  Layer 11 (final):  {three_layer.layer3['W_up'].shape}")
    print()

    # Load layer 9 (raw + generate)
    vm.blocks[9].ffn.W_up.data = three_layer.layer1['W_up']
    vm.blocks[9].ffn.b_up.data = three_layer.layer1['b_up']
    vm.blocks[9].ffn.W_gate.data = three_layer.layer1['W_gate']
    vm.blocks[9].ffn.b_gate.data = three_layer.layer1['b_gate']
    vm.blocks[9].ffn.W_down.data = three_layer.layer1['W_down']
    vm.blocks[9].ffn.b_down.data = three_layer.layer1['b_down']

    # Load layer 10 (carry lookahead)
    vm.blocks[10].ffn.W_up.data = three_layer.layer2['W_up']
    vm.blocks[10].ffn.b_up.data = three_layer.layer2['b_up']
    vm.blocks[10].ffn.W_gate.data = three_layer.layer2['W_gate']
    vm.blocks[10].ffn.b_gate.data = three_layer.layer2['b_gate']
    vm.blocks[10].ffn.W_down.data = three_layer.layer2['W_down']
    vm.blocks[10].ffn.b_down.data = three_layer.layer2['b_down']

    # Load layer 11 (finalize)
    vm.blocks[11].ffn.W_up.data = three_layer.layer3['W_up']
    vm.blocks[11].ffn.b_up.data = three_layer.layer3['b_up']
    vm.blocks[11].ffn.W_gate.data = three_layer.layer3['W_gate']
    vm.blocks[11].ffn.b_gate.data = three_layer.layer3['b_gate']
    vm.blocks[11].ffn.W_down.data = three_layer.layer3['W_down']
    vm.blocks[11].ffn.b_down.data = three_layer.layer3['b_down']

    # Create embedding
    embed = NibbleVMEmbedding(d_model=1280)

    # Test cases
    test_cases = [
        (2, 3, 5, "small values, no carry"),
        (10, 20, 30, "medium values, no carry"),
        (100, 200, 300, "large values, WITH carry"),
        (255, 1, 256, "edge case carry"),
    ]

    passed = 0
    failed = 0

    for a, b, expected, desc in test_cases:
        print(f"Test: {a} + {b} = {expected} ({desc})")
        print("-" * 70)

        # Encode input
        input_embedding = embed.encode_vm_state(
            pc=0, ax=a, sp=4096, bp=4096,
            opcode=Opcode.ADD, stack_top=b, batch_size=1,
        )

        with torch.no_grad():
            # Run through 3 layers
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

        if result == expected:
            print(f"  ✅ PASS: {result}")
            passed += 1
        else:
            print(f"  ❌ FAIL: got {result}, expected {expected}")
            # Show nibble details
            print(f"  Nibble breakdown:")
            for pos in range(8):
                base_idx = pos * 160
                val = output_embedding[0, base_idx + E.RESULT].item()
                nibble = (result >> (pos * 4)) & 0xF
                exp_nibble = (expected >> (pos * 4)) & 0xF
                print(f"    Pos {pos}: {val:7.2f} → {nibble} (expected {exp_nibble})")
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
        print("  Extracted 3-layer ADD weights work correctly!")
    else:
        print("  ⚠️  Some tests failed")

    print("="*70)

    return failed == 0

if __name__ == "__main__":
    success = test_extracted_add()
    exit(0 if success else 1)
