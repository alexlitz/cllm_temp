#!/usr/bin/env python3
"""Final check: What does the embedding actually contain for byte tokens?"""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

print("=" * 80)
print("FINAL EMBEDDING CHECK")
print("=" * 80)

# Create runner and check embedding
runner = AutoregressiveVMRunner()
model = runner.model

print("\n### Check CLEAN_EMBED in embedding matrix ###")
# The embedding object builds the weight matrix in .forward()
# Let's call it with a few byte tokens to check
import torch
device = next(model.parameters()).device
test_tokens = torch.tensor([[0, 1, 3]], device=device)  # batch=1, seq=3: byte 0, IMM, JSR
embeds = model.embed(test_tokens)  # shape: (1, 3, hidden_dim)
embeds = embeds[0]  # Remove batch dim: (3, hidden_dim)

print(f"Embedding shape: {embeds.shape}")

# Check byte 0 (what's at address 2 for both programs)
byte_0_embed = embeds[0, :]
print(f"\nByte 0 CLEAN_EMBED:")
for k in range(16):
    val_lo = byte_0_embed[BD.CLEAN_EMBED_LO + k].item()
    val_hi = byte_0_embed[BD.CLEAN_EMBED_HI + k].item()
    if val_lo > 0.5 or val_hi > 0.5:
        print(f"  CLEAN_EMBED_LO[{k}] = {val_lo:.3f}, CLEAN_EMBED_HI[{k}] = {val_hi:.3f}")

# Check byte 1 (IMM opcode)
byte_1_embed = embeds[1, :]
print(f"\nByte 1 (IMM opcode) CLEAN_EMBED:")
for k in range(16):
    val_lo = byte_1_embed[BD.CLEAN_EMBED_LO + k].item()
    val_hi = byte_1_embed[BD.CLEAN_EMBED_HI + k].item()
    if val_lo > 0.5 or val_hi > 0.5:
        print(f"  CLEAN_EMBED_LO[{k}] = {val_lo:.3f}, CLEAN_EMBED_HI[{k}] = {val_hi:.3f}")

# Check byte 3 (JSR opcode)
byte_3_embed = embeds[2, :]  # Index 2 in our test_tokens
print(f"\nByte 3 (JSR opcode) CLEAN_EMBED:")
for k in range(16):
    val_lo = byte_3_embed[BD.CLEAN_EMBED_LO + k].item()
    val_hi = byte_3_embed[BD.CLEAN_EMBED_HI + k].item()
    if val_lo > 0.5 or val_hi > 0.5:
        print(f"  CLEAN_EMBED_LO[{k}] = {val_lo:.3f}, CLEAN_EMBED_HI[{k}] = {val_hi:.3f}")

print("\n### Check OPCODE_FLAGS in embedding (if any) ###")
# Check if OP_IMM is set directly in embedding
print(f"\nByte 1 (IMM) OP_IMM flag: {byte_1_embed[BD.OP_IMM].item():.3f}")
print(f"Byte 3 (JSR) OP_JSR flag: {byte_3_embed[BD.OP_JSR].item():.3f}")
print(f"Byte 0 (padding) OP_LEA flag: {byte_0_embed[BD.OP_LEA].item():.3f}")

print("\n### THEORY ###")
print("If CLEAN_EMBED shows (0,0) for byte 0, then L5 head 2 fetching from")
print("address 2 would give OPCODE_BYTE=(0,0) for both IMM and JSR.")
print("\nBut maybe the embedding ALSO has OP_* flags pre-set?")
print("If byte 1 embedding has OP_IMM=1 directly, that would explain why IMM works!")
