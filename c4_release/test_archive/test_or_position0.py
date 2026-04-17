#!/usr/bin/env python3
"""Test OR on position 0 only."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E

# Create VM
loader = CompiledWeightLoader()
vm = AutoregressiveVM(d_model=1352, n_layers=16, ffn_hidden=loader.ffn_hidden)
vm.eval()
loader.load_all_weights(vm, verbose=False)

embed = NibbleVMEmbedding(d_model=1352)

# Test OR(5, 3) = 7
print("Testing OR(5, 3) = 7 at position 0")
print("=" * 70)

# Encode: ax=3 (in NIB_A), stack_top=5 (in NIB_B)
input_emb = embed.encode_vm_state(
    pc=0, ax=3, sp=0, bp=0,
    opcode=Opcode.OR, stack_top=5, batch_size=1
)

print("Input at position 0:")
print(f"  NIB_A (scalar): {input_emb[0, E.NIB_A].item()}")
print(f"  NIB_B (scalar): {input_emb[0, E.NIB_B].item()}")
print(f"  NIB_A bits: [{input_emb[0, E.NIB_A_BIT0].item()}, {input_emb[0, E.NIB_A_BIT1].item()}, {input_emb[0, E.NIB_A_BIT2].item()}, {input_emb[0, E.NIB_A_BIT3].item()}]")
print(f"  NIB_B bits: [{input_emb[0, E.NIB_B_BIT0].item()}, {input_emb[0, E.NIB_B_BIT1].item()}, {input_emb[0, E.NIB_B_BIT2].item()}, {input_emb[0, E.NIB_B_BIT3].item()}]")
print(f"  OPCODE: {input_emb[0, E.OPCODE].item()} (should be {Opcode.OR})")
print()

# Run through layer 6 (OR is now in layer 6)
with torch.no_grad():
    x = input_emb.unsqueeze(1)
    output = vm.blocks[6](x).squeeze()

    print("Output at position 0:")
    result_val = output[E.RESULT].item()
    print(f"  RESULT (slot {E.RESULT}): {result_val}")
    print()

    # Check if result is close to 7
    expected = 7.0
    error = abs(result_val - expected)
    print(f"Expected: {expected}, Got: {result_val}, Error: {error}")

    if error < 0.1:
        print("✓ PASS")
    else:
        print("✗ FAIL")

        # Show which bits contributed
        print()
        print("Debug: Check individual bit results")
        # Expected: 5 OR 3 = 0b0101 OR 0b0011 = 0b0111 = 7
        # bit0: 1 OR 1 = 1 (contributes 1)
        # bit1: 0 OR 1 = 1 (contributes 2)
        # bit2: 1 OR 0 = 1 (contributes 4)
        # bit3: 0 OR 0 = 0 (contributes 0)
        print("  Expected contributions: bit0=1, bit1=2, bit2=4, bit3=0")
        print(f"  Got result: {result_val} = 0b{int(result_val):04b}")
