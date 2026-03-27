"""
Test ADD with Multi-Layer Execution (FFN + Attention + FFN)

The nibble-based architecture requires multiple layers for carry propagation:
1. Layer N (FFN): Compute per-nibble sums and CARRY_OUT
2. Layer N+1 (Attention): Propagate CARRY_OUT[i] → CARRY_IN[i+1]
3. Layer N+2 (FFN): Finalize results using CARRY_IN

This test runs through the full transformer layers to enable carry propagation.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E

def test_multilayer_add():
    """Test ADD through multiple layers with carry propagation."""

    print("="*70)
    print("MULTI-LAYER ADD TEST (with Carry Propagation)")
    print("="*70)
    print()

    # Create VM
    print("Creating VM...")
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    # Load all weights
    print("Loading all opcode weights...")
    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    # Create embedding
    embed = NibbleVMEmbedding(d_model=1280)

    print()
    print("="*70)
    print("TEST 1: 2 + 3 = 5 (no carry)")
    print("="*70)

    # Test 1: 2 + 3 = 5
    input_embedding = embed.encode_vm_state(
        pc=0, ax=2, sp=4096, bp=4096,
        opcode=Opcode.ADD, stack_top=3, batch_size=1,
    )

    with torch.no_grad():
        # Run through multiple layers
        x = input_embedding.unsqueeze(1)  # [batch, seq=1, d_model]

        # Layer 9: Primary ALU (FFN)
        x = vm.blocks[9](x)

        # Layer 10: Attention (for carry propagation)
        x = vm.blocks[10](x)

        # Layer 11: Finalize (FFN)
        x = vm.blocks[11](x)

        output_embedding = x.squeeze(1)  # [batch, d_model]

    result = embed.decode_result_nibbles(output_embedding)

    print(f"Input: 2 + 3")
    print(f"Result: {result}")
    print(f"Expected: 5")
    print(f"{'✅ PASS' if result == 5 else '❌ FAIL'}")
    print()

    print("="*70)
    print("TEST 2: 100 + 200 = 300 (with carry)")
    print("="*70)

    # Test 2: 100 + 200 = 300
    input_embedding2 = embed.encode_vm_state(
        pc=0, ax=100, sp=4096, bp=4096,
        opcode=Opcode.ADD, stack_top=200, batch_size=1,
    )

    with torch.no_grad():
        # Run through multiple layers
        x = input_embedding2.unsqueeze(1)  # [batch, seq=1, d_model]

        # Layer 9: Primary ALU (FFN)
        x = vm.blocks[9](x)

        # Show CARRY_OUT after layer 9
        print("After Layer 9 (FFN - compute sums):")
        for pos in range(8):
            base_idx = pos * 160
            carry = x[0, 0, base_idx + E.CARRY_OUT].item()
            result_val = x[0, 0, base_idx + E.RESULT].item()
            print(f"  Pos {pos}: RESULT={result_val:7.2f}, CARRY_OUT={carry:7.2f}")
        print()

        # Layer 10: Attention (for carry propagation)
        x = vm.blocks[10](x)

        # Show CARRY_IN after attention
        print("After Layer 10 (Attention - propagate carries):")
        for pos in range(8):
            base_idx = pos * 160
            carry = x[0, 0, base_idx + E.CARRY_IN].item()
            result_val = x[0, 0, base_idx + E.RESULT].item()
            print(f"  Pos {pos}: RESULT={result_val:7.2f}, CARRY_IN={carry:7.2f}")
        print()

        # Layer 11: Finalize (FFN)
        x = vm.blocks[11](x)

        print("After Layer 11 (FFN - finalize):")
        for pos in range(8):
            base_idx = pos * 160
            result_val = x[0, 0, base_idx + E.RESULT].item()
            exp_nibble = (300 >> (pos * 4)) & 0xF
            print(f"  Pos {pos}: RESULT={result_val:7.2f} (expected {exp_nibble})")
        print()

        output_embedding2 = x.squeeze(1)  # [batch, d_model]

    result2 = embed.decode_result_nibbles(output_embedding2)

    print(f"Input: 100 + 200")
    print(f"Result: {result2}")
    print(f"Expected: 300")
    print(f"{'✅ PASS' if result2 == 300 else '❌ FAIL'}")
    print()

    print("="*70)

    return result == 5 and result2 == 300

if __name__ == "__main__":
    success = test_multilayer_add()
    exit(0 if success else 1)
