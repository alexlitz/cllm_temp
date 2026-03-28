"""
Test Full Pipeline Forward Pass

Try passing through all layers and see what emerges.
"""

import torch
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E


def test_full_forward_pass():
    """Test a complete forward pass through all 16 layers."""

    print("=" * 70)
    print("FULL PIPELINE TEST")
    print("=" * 70)
    print()

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    embed = NibbleVMEmbedding(d_model=1280)

    # Simple test: ADD 5 + 3
    print("Test: 5 + 3 through full pipeline")
    print()

    input_embedding = embed.encode_vm_state(
        pc=0, ax=3, sp=0, bp=0,
        opcode=Opcode.ADD, stack_top=5, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)

        print("Passing through layers:")

        # Layers 9-11: Arithmetic
        print("  Layers 9-11 (arithmetic)...")
        x = vm.blocks[9](x)
        x = vm.blocks[10](x)
        x = vm.blocks[11](x)

        result_after_arith = embed.decode_result_nibbles(x.squeeze(1))
        print(f"    Result after arithmetic: {result_after_arith}")

        # Layer 12: Comparisons (should pass through for ADD)
        print("  Layer 12 (comparisons)...")
        x = vm.blocks[12](x)

        # Layer 13: Control flow (should compute next PC?)
        print("  Layer 13 (control flow)...")
        x = vm.blocks[13](x)

        # Layers 14-15: Memory
        print("  Layers 14-15 (memory)...")
        x = vm.blocks[14](x)
        x = vm.blocks[15](x)

        output = x.squeeze(1)

    # Check final output
    print()
    print("Final output:")
    print(f"  RESULT slot: {embed.decode_result_nibbles(output)}")
    print(f"  TEMP slot: {embed.decode_pc_nibbles(output, E.TEMP)}")
    print(f"  Expected result: 8")

    print()
    print("=" * 70)


if __name__ == "__main__":
    test_full_forward_pass()
