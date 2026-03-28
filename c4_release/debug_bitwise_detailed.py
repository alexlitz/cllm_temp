"""
Debug Bitwise Operations - Detailed Investigation

Check what layer 12 outputs for bitwise operations and where results are stored.
"""

import torch
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E


def inspect_layer12_output(opcode, opcode_name, a, b, expected):
    """Inspect layer 12 output for a specific bitwise operation."""

    print(f"\n{'=' * 70}")
    print(f"DEBUG: {opcode_name} ({a} {opcode_name} {b} = {expected})")
    print(f"{'=' * 70}")

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    embed = NibbleVMEmbedding(d_model=1280)

    # Encode the operation
    # For bitwise: stack_top op ax
    input_embedding = embed.encode_vm_state(
        pc=0, ax=b, sp=0, bp=0,
        opcode=opcode, stack_top=a, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)

        # Pass through layer 12
        output = vm.blocks[12](x)
        output_flat = output.squeeze(1)

    # Check all slots at position 0
    print("\nOutput slots at position 0:")
    base = 0
    print(f"  NIB_A:    {output_flat[0, base + E.NIB_A].item():.4f}")
    print(f"  NIB_B:    {output_flat[0, base + E.NIB_B].item():.4f}")
    print(f"  RAW_SUM:  {output_flat[0, base + E.RAW_SUM].item():.4f}")
    print(f"  CARRY_IN: {output_flat[0, base + E.CARRY_IN].item():.4f}")
    print(f"  CARRY_OUT:{output_flat[0, base + E.CARRY_OUT].item():.4f}")
    print(f"  RESULT:   {output_flat[0, base + E.RESULT].item():.4f}")
    print(f"  TEMP:     {output_flat[0, base + E.TEMP].item():.4f}")

    # Try decoding from different slots
    print("\nDecoding attempts:")

    # 1. Try nibble decoding from RESULT
    result_nibbles = embed.decode_result_nibbles(output_flat)
    print(f"  RESULT (nibbles): {result_nibbles} (0x{result_nibbles:x})")

    # 2. Try nibble decoding from TEMP
    temp_nibbles = embed.decode_pc_nibbles(output_flat, E.TEMP)
    print(f"  TEMP (nibbles):   {temp_nibbles} (0x{temp_nibbles:x})")

    # 3. Try threshold from TEMP (like comparisons)
    temp_val = output_flat[0, E.TEMP].item()
    temp_threshold = 1 if temp_val > 0.5 else 0
    print(f"  TEMP (threshold): {temp_threshold}")

    # 4. Try reading raw RESULT value
    result_val = output_flat[0, E.RESULT].item()
    print(f"  RESULT (raw):     {result_val:.4f}")

    # 5. Check if result is 32-bit encoded across all positions
    print("\nAll position RESULT values:")
    for pos in range(8):
        base_idx = pos * 160
        result_pos = output_flat[0, base_idx + E.RESULT].item()
        print(f"  Position {pos}: {result_pos:.4f}")

    # 6. Try interpreting RESULT as integer at position 0
    result_int = int(round(output_flat[0, E.RESULT].item()))
    print(f"\n  RESULT (rounded): {result_int}")

    print(f"\nExpected: {expected}")
    print(f"{'=' * 70}")


def main():
    """Test all bitwise operations."""

    print("=" * 70)
    print("BITWISE OPERATIONS DETAILED DEBUG")
    print("=" * 70)

    # Test OR
    inspect_layer12_output(Opcode.OR, "OR", 5, 3, 7)
    inspect_layer12_output(Opcode.OR, "OR", 12, 10, 14)

    # Test XOR
    inspect_layer12_output(Opcode.XOR, "XOR", 5, 3, 6)
    inspect_layer12_output(Opcode.XOR, "XOR", 12, 10, 6)

    # Test AND
    inspect_layer12_output(Opcode.AND, "AND", 5, 3, 1)
    inspect_layer12_output(Opcode.AND, "AND", 12, 10, 8)

    # Test SHL
    inspect_layer12_output(Opcode.SHL, "SHL", 5, 2, 20)
    inspect_layer12_output(Opcode.SHL, "SHL", 3, 3, 24)

    # Test SHR
    inspect_layer12_output(Opcode.SHR, "SHR", 20, 2, 5)
    inspect_layer12_output(Opcode.SHR, "SHR", 24, 3, 3)


if __name__ == "__main__":
    main()
