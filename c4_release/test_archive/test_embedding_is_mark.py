#!/usr/bin/env python3
"""Test if embedding correctly sets IS_MARK at marker tokens."""

from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD, set_vm_weights
import torch

# Create model
model = AutoregressiveVM(n_layers=17)

# IMPORTANT: Must call set_vm_weights() to configure the embedding!
set_vm_weights(model)

print("=" * 80)
print("Embedding IS_MARK Configuration Test")
print("=" * 80)

# Check embedding weight matrix directly
embed_weight = model.embed.embed.weight

print("\nIS_MARK values in embedding weight matrix:")
print(f"{'Token':<15} {'Value':<10} {'dim':<5} {'Expected':<10} {'Status'}")
print("-" * 60)

marker_tokens = [
    ("REG_PC", Token.REG_PC, BD.MARK_PC),
    ("REG_AX", Token.REG_AX, BD.MARK_AX),
    ("REG_SP", Token.REG_SP, BD.MARK_SP),
    ("REG_BP", Token.REG_BP, BD.MARK_BP),
    ("MEM", Token.MEM, BD.MARK_MEM),
    ("CODE_START", Token.CODE_START, BD.MARK_CS),
    ("STACK0", Token.STACK0, BD.MARK_STACK0),
]

for name, tok, mark_dim in marker_tokens:
    is_mark_val = embed_weight[tok, BD.IS_MARK].item()
    mark_val = embed_weight[tok, mark_dim].item()
    status = "✓" if abs(is_mark_val - 1.0) < 0.01 else "✗"
    print(f"{name:<15} {tok:<10} {is_mark_val:5.2f} {mark_val:5.2f} {'1.0':<10} {status}")

# Also check some non-marker tokens
print("\nIS_MARK for non-marker tokens (should be 0):")
print(f"{'Token':<15} {'Value':<10} {'Expected':<10} {'Status'}")
print("-" * 50)

non_marker_tokens = [
    ("Byte 0x00", 0),
    ("Byte 0xFF", 0xFF),
    ("Byte 0x2A", 0x2A),
    ("STEP_END", Token.STEP_END),
]

for name, tok in non_marker_tokens:
    is_mark_val = embed_weight[tok, BD.IS_MARK].item()
    status = "✓" if abs(is_mark_val) < 0.01 else "✗"
    print(f"{name:<15} {tok:<10} {is_mark_val:5.2f} {'0.0':<10} {status}")

print("\n" + "=" * 80)

# Now test after forward pass through embedding
print("\nIS_MARK after embedding forward pass:")
print("=" * 80)

context = [
    Token.REG_SP,  # Position 0: SP marker
    0xF8,          # Position 1: SP byte 0
    0xF7,          # Position 2: SP byte 1
]

input_ids = torch.tensor([context])
x = model.embed(input_ids)

print(f"\n{'Pos':<5} {'Token':<6} {'IS_MARK':<10} {'MARK_SP':<10} {'Expected':<10} {'Status'}")
print("-" * 60)
for i, tok in enumerate(context):
    is_mark = x[0, i, BD.IS_MARK].item()
    mark_sp = x[0, i, BD.MARK_SP].item()
    expected = "1.0" if tok == Token.REG_SP else "0.0"
    expected_val = 1.0 if tok == Token.REG_SP else 0.0
    status = "✓" if abs(is_mark - expected_val) < 0.01 else "✗"
    print(f"{i:<5} {tok:<6} {is_mark:10.2f} {mark_sp:10.2f} {expected:<10} {status}")

print("\n" + "=" * 80)
print("Diagnosis:")
print("- If embedding weights are correct (1.0 at markers), problem is in layers")
print("- If embedding weights are wrong, problem is in initialization")
print("=" * 80)
