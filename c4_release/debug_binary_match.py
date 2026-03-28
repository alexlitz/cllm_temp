#!/usr/bin/env python3
"""Debug binary bit matching."""

import torch
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import E, Opcode

# Test encoding
embed = NibbleVMEmbedding(d_model=1344)

# Encode a simple state
state_emb = embed.encode_vm_state(
    pc=0, ax=3, sp=0, bp=0,
    opcode=Opcode.OR, stack_top=5, batch_size=1
)

print("State embedding shape:", state_emb.shape)
print()

# Check binary encoding for position 0
pos = 0
base_idx = pos * 168

print("Position 0 (nibble 0):")
print(f"  NIB_A (scalar): {state_emb[0, base_idx + E.NIB_A].item()}")
print(f"  NIB_B (scalar): {state_emb[0, base_idx + E.NIB_B].item()}")
print()

print("  NIB_A binary bits:")
for i in range(4):
    bit_val = state_emb[0, base_idx + E.NIB_A_BIT0 + i].item()
    print(f"    bit{i}: {bit_val}")

print()
print("  NIB_B binary bits:")
for i in range(4):
    bit_val = state_emb[0, base_idx + E.NIB_B_BIT0 + i].item()
    print(f"    bit{i}: {bit_val}")

print()
print("Expected:")
print("  ax=3 → nibble=3 → 0b0011 → [1, 1, 0, 0]")
print("  stack_top=5 → nibble=5 → 0b0101 → [1, 0, 1, 0]")
