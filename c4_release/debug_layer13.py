"""
Debug Layer 13 (PC Updates)

Check what layer 13 outputs for control flow operations.
"""

import torch
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E


def debug_layer13_jmp():
    """Debug JMP instruction."""

    print("=" * 70)
    print("DEBUG LAYER 13 - JMP")
    print("=" * 70)
    print()

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    embed = NibbleVMEmbedding(d_model=1280)

    # Test: JMP to address 0x100 (256)
    # Current PC: 0x08, Immediate: 0x100
    # Expected next PC: 0x100

    pc = 0x08
    imm = 0x100

    print(f"Test: JMP from PC={pc:#x} to IMM={imm:#x}")
    print(f"Expected next PC: {imm:#x}")
    print()

    input_embedding = embed.encode_vm_state(
        pc=pc, ax=0, sp=0, bp=0,
        opcode=Opcode.JMP, imm=imm, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)

        # Pass through layer 13
        output = vm.blocks[13](x)
        output_flat = output.squeeze(1)

    # Check what's in various slots
    print("Output slots at position 0:")
    base = 0
    print(f"  NIB_A: {output_flat[0, base + E.NIB_A].item():.4f}")
    print(f"  RESULT: {output_flat[0, base + E.RESULT].item():.4f}")
    print(f"  TEMP: {output_flat[0, base + E.TEMP].item():.4f}")
    print()

    # Try decoding from different slots
    print("Attempting to decode PC from different slots:")
    print(f"  RESULT: {embed.decode_result_nibbles(output_flat):#x}")
    print(f"  TEMP: {embed.decode_pc_nibbles(output_flat, E.TEMP):#x}")
    print(f"  RAW_SUM: {embed.decode_pc_nibbles(output_flat, E.RAW_SUM):#x}")

    # Check if immediate is passed through
    print()
    print(f"  Expected: {imm:#x}")

    print()
    print("=" * 70)


def debug_layer13_bz():
    """Debug BZ instruction."""

    print()
    print("=" * 70)
    print("DEBUG LAYER 13 - BZ (Branch if Zero)")
    print("=" * 70)
    print()

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    embed = NibbleVMEmbedding(d_model=1280)

    # Test 1: AX=0, should branch
    pc = 0x10
    imm = 0x100
    ax = 0

    print(f"Test 1: BZ with AX={ax} (should branch)")
    print(f"  Current PC: {pc:#x}")
    print(f"  Immediate: {imm:#x}")
    print(f"  Expected: {imm:#x} (branch taken)")
    print()

    input_embedding = embed.encode_vm_state(
        pc=pc, ax=ax, sp=0, bp=0,
        opcode=Opcode.BZ, imm=imm, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)
        output = vm.blocks[13](x)
        output_flat = output.squeeze(1)

    result_pc = embed.decode_result_nibbles(output_flat)
    print(f"  Layer 13 output (RESULT): {result_pc:#x}")

    # Test 2: AX=5, should NOT branch
    ax = 5
    expected_pc = pc + 8

    print()
    print(f"Test 2: BZ with AX={ax} (should NOT branch)")
    print(f"  Current PC: {pc:#x}")
    print(f"  Expected: {expected_pc:#x} (PC + 8)")
    print()

    input_embedding = embed.encode_vm_state(
        pc=pc, ax=ax, sp=0, bp=0,
        opcode=Opcode.BZ, imm=imm, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)
        output = vm.blocks[13](x)
        output_flat = output.squeeze(1)

    result_pc = embed.decode_result_nibbles(output_flat)
    print(f"  Layer 13 output (RESULT): {result_pc:#x}")

    print()
    print("=" * 70)


def debug_layer13_default():
    """Debug default PC increment."""

    print()
    print("=" * 70)
    print("DEBUG LAYER 13 - Default (PC + 8)")
    print("=" * 70)
    print()

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    embed = NibbleVMEmbedding(d_model=1280)

    # Test with ADD (should just increment PC)
    pc = 0x20
    expected_pc = pc + 8

    print(f"Test: ADD instruction (default PC increment)")
    print(f"  Current PC: {pc:#x}")
    print(f"  Expected: {expected_pc:#x} (PC + 8)")
    print()

    input_embedding = embed.encode_vm_state(
        pc=pc, ax=5, sp=0, bp=0,
        opcode=Opcode.ADD, imm=0, stack_top=3, batch_size=1
    )

    with torch.no_grad():
        x = input_embedding.unsqueeze(1)
        # Layer 13 should pass through or compute PC+8
        output = vm.blocks[13](x)
        output_flat = output.squeeze(1)

    # Check output
    result_pc = embed.decode_result_nibbles(output_flat)
    temp_pc = embed.decode_pc_nibbles(output_flat, E.TEMP)

    print(f"  Layer 13 output (RESULT): {result_pc:#x}")
    print(f"  Layer 13 output (TEMP): {temp_pc:#x}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    debug_layer13_jmp()
    debug_layer13_bz()
    debug_layer13_default()
