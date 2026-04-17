"""Debug script to check H1[STACK0] values at different token positions."""

import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim as BD, Token, set_vm_weights

# Create model
model = AutoregressiveVM(n_layers=17)

# Initialize weights
print("Initializing VM weights...")
set_vm_weights(model)
print("Weights initialized.")

# Create a dummy context with the standard 35-token structure
# PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1)
context = [
    Token.REG_PC, 0, 0, 0, 0,          # PC marker + 4 bytes (pos 0-4)
    Token.REG_AX, 21, 0, 0, 0,         # AX marker + 4 bytes (pos 5-9), AX=21
    Token.REG_SP, 0, 0, 1, 0,          # SP marker + 4 bytes (pos 10-14)
    Token.REG_BP, 0, 0, 1, 0,          # BP marker + 4 bytes (pos 15-19)
    Token.STACK0, 0, 0, 0, 0,          # STACK0 marker + 4 bytes (pos 20-24)
    Token.MEM, 0, 0, 0, 0, 0, 0, 0, 0, # MEM section (pos 25-33)
    Token.STEP_END                     # END (pos 34)
]

print(f"Context length: {len(context)}")
print(f"Context: {context}")

# Check raw embedding weights for token 0 (byte value)
print(f"\nChecking raw embedding weights for token 0:")
token_0_embed = model.embed.embed.weight[0]  # [512]
print(f"IS_BYTE dimension ({BD.IS_BYTE}): {token_0_embed[BD.IS_BYTE].item():.3f}")
print(f"EMBED_LO[0] dimension ({BD.EMBED_LO}): {token_0_embed[BD.EMBED_LO].item():.3f}")

# Embed tokens and run through Layer 0 to get hop counts
x = model.embed(torch.tensor([context], dtype=torch.long))  # [1, 35, 512]
print(f"\nEmbedded shape: {x.shape}")

# Run through Layer 0 to compute hop counts
print("Running through Layer 0...")
x = model.blocks[0].attn(x)  # Layer 0 attention computes hop counts
x = model.blocks[0].ffn(x)   # Layer 0 FFN
print("Layer 0 complete.")

# Run through Layer 1 to compute BYTE_INDEX
print("Running through Layer 1...")
x = model.blocks[1].attn(x)  # Layer 1 attention

# Check values BEFORE Layer 1 FFN
print(f"\nBEFORE Layer 1 FFN at STACK0 byte 0:")
print(f"  IS_BYTE: {x[0, 21, BD.IS_BYTE].item():.3f}")
print(f"  L1H4[BP]: {x[0, 21, BD.L1H4 + 3].item():.3f}")
print(f"  H1[BP]: {x[0, 21, BD.H1 + 3].item():.3f}")
print(f"  BYTE_INDEX_0: {x[0, 21, BD.BYTE_INDEX_0].item():.6f}")

x = model.blocks[1].ffn(x)   # Layer 1 FFN computes BYTE_INDEX
print(f"\nAFTER Layer 1 FFN:")
print(f"  STACK0 byte 0 BYTE_INDEX_0: {x[0, 21, BD.BYTE_INDEX_0].item():.6f}")
print(f"  SP byte 0 (pos 11) BYTE_INDEX_0: {x[0, 11, BD.BYTE_INDEX_0].item():.6f}")
print(f"  AX byte 0 (pos 6) BYTE_INDEX_0: {x[0, 6, BD.BYTE_INDEX_0].item():.6f}")
print(f"  PC byte 0 (pos 1) BYTE_INDEX_0: {x[0, 1, BD.BYTE_INDEX_0].item():.6f}")
print("Layer 1 complete.")

# Check H1 dimensions at different positions
STACK0_I = 4  # Index for STACK0 in H1 array (as used in JSR code)
print(f"\nSTACK0_I = {STACK0_I}")
print(f"BD.H1 = {BD.H1}, size = 7 dimensions")
print(f"BD.H1 + STACK0_I = {BD.H1 + STACK0_I}")

# Print H1 values for each position
positions = {
    0: "PC marker",
    1: "PC byte 0",
    15: "BP marker",
    16: "BP byte 0",
    20: "STACK0 marker",
    21: "STACK0 byte 0",
    22: "STACK0 byte 1",
    23: "STACK0 byte 2",
    24: "STACK0 byte 3",
}

print("\nH1[STACK0_I] values at different positions:")
print("=" * 60)
for pos, label in positions.items():
    h1_stack0_value = x[0, pos, BD.H1 + STACK0_I].item()
    print(f"Position {pos:2d} ({label:15s}): H1[STACK0_I] = {h1_stack0_value:.3f}")

# Also check other H dimensions for STACK0 byte 0
print("\nAll H[STACK0_I] values at STACK0 byte 0 (position 21):")
print("=" * 60)
for i in range(8):
    h_dim = BD.H0 + i * 7  # H0, H1, H2, ... H7 are spaced 7 apart
    h_value = x[0, 21, h_dim + STACK0_I].item()
    print(f"H{i}[STACK0_I] = {h_value:.3f}")

# Check other marker flags at STACK0 byte 0
print("\nMarker and position flags at STACK0 byte 0 (position 21):")
print("=" * 60)
print(f"MARK_STACK0: {x[0, 21, BD.MARK_STACK0].item():.3f}")
print(f"IS_BYTE: {x[0, 21, BD.IS_BYTE].item():.3f}")
print(f"BYTE_INDEX_0: {x[0, 21, BD.BYTE_INDEX_0].item():.6f}")
print(f"BYTE_INDEX_1: {x[0, 21, BD.BYTE_INDEX_1].item():.6f}")
print(f"BYTE_INDEX_2: {x[0, 21, BD.BYTE_INDEX_2].item():.6f}")
print(f"BYTE_INDEX_3: {x[0, 21, BD.BYTE_INDEX_3].item():.6f}")

# Check L1H dimensions for BYTE_INDEX computation
BP_I = 3
print(f"\nL1H dimensions for BYTE_INDEX computation at STACK0 byte 0:")
print(f"L1H0[BP]: {x[0, 21, BD.L1H0 + BP_I].item():.3f}")
print(f"L1H1[BP]: {x[0, 21, BD.L1H2 + BP_I].item():.3f}")
print(f"L1H2[BP]: {x[0, 21, BD.L1H2 + BP_I].item():.3f}")
print(f"L1H4[BP]: {x[0, 21, BD.L1H4 + BP_I].item():.3f} (threshold 6.5)")
print(f"L2H0[BP]: {x[0, 21, BD.L2H0 + BP_I].item():.3f} (threshold 5.5)")
print(f"\nExpected for BYTE_INDEX_0 at dist=6: L1H4[BP]=1 (6 < 6.5) AND H1[BP]=0 (6 > 4.5)")
print(f"Actual H1[BP]: {x[0, 21, BD.H1 + BP_I].item():.3f}")

# Compare with AX byte 0 (position 6)
print("\nComparison with AX byte 0 (position 6):")
print("=" * 60)
AX_I = 1  # Index for AX in H1 array
h1_ax_value = x[0, 6, BD.H1 + AX_I].item()
h1_stack0_value = x[0, 21, BD.H1 + STACK0_I].item()
print(f"AX byte 0   H1[AX_I={AX_I}]: {h1_ax_value:.3f}")
print(f"STACK0 byte 0 H1[STACK0_I={STACK0_I}]: {h1_stack0_value:.3f}")

# Check BP-relative hop counts at STACK0
print("\nBP-relative hop counts at STACK0 byte 0 (position 21):")
print("Distance from BP marker (pos 15) to STACK0 byte 0 (pos 21) = 6")
print("=" * 60)
BP_I = 3  # Index for BP in H1 array
for i in range(8):
    h_dim = BD.H0 + i * 7  # H0, H1, H2, ... H7
    h_value = x[0, 21, h_dim + BP_I].item()
    # H thresholds: H0=3.5, H1=4.5, H2=5.5, H3=9.5, H4=10.5, H5=14.5, H6=15.5, H7=19.5
    thresholds = [3.5, 4.5, 5.5, 9.5, 10.5, 14.5, 15.5, 19.5]
    expected = 1 if 6 <= thresholds[i] else 0
    print(f"H{i}[BP_I={BP_I}] = {h_value:.3f}  (expected {expected} since dist=6 {'<=' if 6 <= thresholds[i] else '>'} {thresholds[i]})")
