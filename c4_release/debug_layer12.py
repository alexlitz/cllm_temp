"""
Debug Layer 12 Output

Check what layer 12 actually outputs to understand how to decode results.
"""

import torch
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E


def debug_layer12_comparison():
    """Debug what layer 12 outputs for a comparison."""

    print("=" * 70)
    print("DEBUG LAYER 12 COMPARISON")
    print("=" * 70)
    print()

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    embed = NibbleVMEmbedding(d_model=1280)

    # Test EQ with 5 == 5 (should be 1)
    print("Test: 5 == 5 (should return 1)")
    print()

    input_embedding = embed.encode_vm_state(
        pc=0, ax=5, sp=0, bp=0,
        opcode=Opcode.EQ, stack_top=5, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)
        output = vm.blocks[12](x)
        output_flat = output.squeeze(1)

    # Check all possible output slots
    print("Checking output slots:")
    for pos in range(8):
        base_idx = pos * 160
        print(f"\nPosition {pos}:")
        print(f"  NIB_A: {output_flat[0, base_idx + E.NIB_A].item():.4f}")
        print(f"  NIB_B: {output_flat[0, base_idx + E.NIB_B].item():.4f}")
        print(f"  RAW_SUM: {output_flat[0, base_idx + E.RAW_SUM].item():.4f}")
        print(f"  CARRY_IN: {output_flat[0, base_idx + E.CARRY_IN].item():.4f}")
        print(f"  CARRY_OUT: {output_flat[0, base_idx + E.CARRY_OUT].item():.4f}")
        print(f"  RESULT: {output_flat[0, base_idx + E.RESULT].item():.4f}")
        print(f"  TEMP: {output_flat[0, base_idx + E.TEMP].item():.4f}")

    # Try decoding from different slots
    print("\nDecoding from different slots:")
    print(f"  RESULT slot: {embed.decode_result_nibbles(output_flat)}")
    print(f"  NIB_A slot: {embed.decode_pc_nibbles(output_flat, E.NIB_A)}")
    print(f"  RAW_SUM slot: {embed.decode_pc_nibbles(output_flat, E.RAW_SUM)}")

    print()
    print("=" * 70)


def debug_layer12_bitwise():
    """Debug what layer 12 outputs for bitwise."""

    print()
    print("=" * 70)
    print("DEBUG LAYER 12 BITWISE")
    print("=" * 70)
    print()

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    embed = NibbleVMEmbedding(d_model=1280)

    # Test OR with 5 | 3 (should be 7)
    print("Test: 5 | 3 (should return 7)")
    print()

    input_embedding = embed.encode_vm_state(
        pc=0, ax=3, sp=0, bp=0,
        opcode=Opcode.OR, stack_top=5, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)
        output = vm.blocks[12](x)
        output_flat = output.squeeze(1)

    # Check first position
    pos = 0
    base_idx = pos * 160
    print(f"Position {pos} (should be 7 = 0111 binary, nibble = 7):")
    print(f"  NIB_A: {output_flat[0, base_idx + E.NIB_A].item():.4f}")
    print(f"  NIB_B: {output_flat[0, base_idx + E.NIB_B].item():.4f}")
    print(f"  RESULT: {output_flat[0, base_idx + E.RESULT].item():.4f}")

    print()
    print(f"Decoded from RESULT: {embed.decode_result_nibbles(output_flat)}")
    print(f"Expected: 7")

    print()
    print("=" * 70)


if __name__ == "__main__":
    debug_layer12_comparison()
    debug_layer12_bitwise()
