#!/usr/bin/env python3
"""Deep dive into threshold attention V and O matrices."""

from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD, set_vm_weights
import torch

# Create model
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)

print("=" * 80)
print("Threshold Attention V and O Matrix Configuration")
print("=" * 80)

# Get L0 attention layer
attn0 = model.blocks[0].attn
HD = attn0.head_dim

# Check head 1 (threshold=4.5, slope=10.0)
head_idx = 1
base = head_idx * HD

print(f"\nHead {head_idx} (HD={HD}, base={base}):")
print(f"Threshold=4.5, Slope=10.0")

# Check V matrix for all marker dimensions
print("\nV matrix configuration (copying marker flags):")
print(f"{'V_dim':<8} {'MARK_PC':<10} {'MARK_AX':<10} {'MARK_SP':<10} {'MARK_BP':<10} {'MARK_MEM':<10} {'MARK_SE':<10} {'MARK_CS':<10}")
print("-" * 90)

markers = [
    ("MARK_PC", BD.MARK_PC),
    ("MARK_AX", BD.MARK_AX),
    ("MARK_SP", BD.MARK_SP),
    ("MARK_BP", BD.MARK_BP),
    ("MARK_MEM", BD.MARK_MEM),
    ("MARK_SE", BD.MARK_SE),
    ("MARK_CS", BD.MARK_CS),
]

for i in range(7):
    v_dim = base + 1 + i
    weights = []
    for name, mark_dim in markers:
        w = attn0.W_v[v_dim, mark_dim].item()
        weights.append(f"{w:.2f}")
    print(f"V[{v_dim:<4}] " + " ".join(f"{w:>9}" for w in weights))

# Check O matrix routing
print("\nO matrix configuration (routing V output to threshold dims):")
threshold_dims = [BD.H0, BD.H1, BD.H2, BD.H3, BD.H4, BD.H5, BD.H6, BD.H7]
out_base = threshold_dims[head_idx]  # H1 for head 1

print(f"Output base: dim {out_base} (H{head_idx})")
print(f"\n{'O_out':<8} ", end="")
for i in range(7):
    print(f"{'V['+str(base+1+i)+']':<10}", end="")
print()
print("-" * 80)

for m in range(7):
    o_dim = out_base + m
    weights = []
    for i in range(7):
        v_dim = base + 1 + i
        w = attn0.W_o[o_dim, v_dim].item()
        weights.append(f"{w:.2f}")
    marker_name = markers[m][0]
    print(f"O[{o_dim:<4}] " + " ".join(f"{w:>9}" for w in weights) + f"  → {marker_name}")

# Now test with actual forward pass
print("\n" + "=" * 80)
print("Testing with actual context")
print("=" * 80)

context = [
    Token.REG_SP,  # Position 0: SP marker
    0xF8,          # Position 1: SP byte 0 (d=1)
    0xF7,          # Position 2: SP byte 1 (d=2)
]

input_ids = torch.tensor([context])
x = model.embed(input_ids)

print("\nInput embedding at position 0 (SP marker):")
print(f"{'Dim':<15} {'Value':<10}")
print("-" * 30)
for name, mark_dim in markers:
    val = x[0, 0, mark_dim].item()
    print(f"{name:<15} {val:10.2f}")

# Run through L0
x_out = model.blocks[0](x)

print("\nL0 output at position 1 (SP byte 0, d=1):")
print(f"{'Dim':<15} {'Value':<10} {'Expected':<10}")
print("-" * 40)
for i, (name, _) in enumerate(markers):
    dim = out_base + i  # H1 + marker_index
    val = x_out[0, 1, dim].item()
    # Should be 1.0 for all markers (threshold=4.5 > d=1)
    expected = "1.0"
    status = "✓" if abs(val - 1.0) < 0.1 else "✗"
    print(f"{name:<15} {val:10.2f} {expected:<10} {status}")

print("\n" + "=" * 80)
print("Diagnosis:")
print("- V should copy all 7 marker flags from input")
print("- O should route to threshold output dims (H1+0 through H1+6)")
print("- Output at d=1 should be ~1.0 for all markers (within threshold 4.5)")
print("=" * 80)
