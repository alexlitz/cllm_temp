"""Debug script to check hop counts at PC byte positions."""

import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim as BD, Token, set_vm_weights

# Create model
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)

# Create standard 35-token context
context = [
    Token.REG_PC, 0, 1, 2, 3,          # PC marker + 4 bytes (pos 0-4)
    Token.REG_AX, 10, 11, 12, 13,      # AX marker + 4 bytes (pos 5-9)
    Token.REG_SP, 248, 255, 0, 0,      # SP marker + 4 bytes (pos 10-14), SP=0x0000fff8
    Token.REG_BP, 0, 0, 1, 0,          # BP marker + 4 bytes (pos 15-19)
    Token.STACK0, 0, 0, 0, 0,          # STACK0 marker + 4 bytes (pos 20-24)
    Token.MEM, 0, 0, 0, 0, 0, 0, 0, 0, # MEM section (pos 25-33)
    Token.STEP_END                     # END (pos 34)
]

print(f'Context: {len(context)} tokens')
print(f'PC bytes at positions 1-4: {context[1:5]}')

# Define marker indices
PC_I = 0  # Index for PC in hop count arrays

# Embed and run through Layers 0 and 1
x = model.embed(torch.tensor([context], dtype=torch.long))

# Layer 0
x = model.blocks[0].attn(x)
x_after_l0_attn = x.clone()
x = model.blocks[0].ffn(x)
x_after_l0 = x.clone()

# Layer 1 attention (computes L1H0/1/2)
x = model.blocks[1].attn(x)
x_after_l1_attn = x.clone()

# Check L1H0/1/2 immediately after L1 attention (before FFN)
print('\nL1H0/1/2 immediately after L1 attention (before FFN):')
print('=' * 70)
for pos in [0, 1, 2, 3, 4]:
    dist = pos
    print(f'\nPosition {pos} ({"PC marker" if pos == 0 else f"PC byte {pos-1}"}, dist={dist}):')
    l1h0 = x_after_l1_attn[0, pos, BD.L1H0 + PC_I].item()
    l1h1 = x_after_l1_attn[0, pos, BD.L1H1 + PC_I].item()
    l1h2 = x_after_l1_attn[0, pos, BD.L1H2 + PC_I].item()
    print(f'  L1H0[PC] = {l1h0:.3f}  (threshold 0.5, expected {1 if dist <= 0.5 else 0})')
    print(f'  L1H1[PC] = {l1h1:.3f}  (threshold 1.5, expected {1 if dist <= 1.5 else 0})')
    print(f'  L1H2[PC] = {l1h2:.3f}  (threshold 2.5, expected {1 if dist <= 2.5 else 0})')

# Now run FFN
x = model.blocks[1].ffn(x)

# Check hop counts at PC byte positions (after FFN)
print('\nHop counts at PC byte positions (after L1 FFN):')
print('=' * 70)

hop_dims = [
    ('L1H0', BD.L1H0, 0.5),
    ('L1H1', BD.L1H1, 1.5),
    ('L1H2', BD.L1H2, 2.5),
    ('H0', BD.H0, 3.5),
    ('H1', BD.H1, 4.5),
]

for pos in [0, 1, 2, 3, 4]:
    dist = pos  # Distance from PC marker
    print(f'\nPosition {pos} ({"PC marker" if pos == 0 else f"PC byte {pos-1}"}, dist={dist}):')
    for name, dim_base, threshold in hop_dims:
        value = x[0, pos, dim_base + PC_I].item()
        expected = 1 if dist <= threshold else 0
        status = '✓' if abs(value - expected) < 0.1 else '✗'
        print(f'  {name}[PC] = {value:.3f}  (threshold {threshold}, expected {expected}) {status}')

# Check threshold differences
print('\n\nThreshold differences at PC byte positions:')
print('=' * 70)

for pos in [1, 2, 3, 4]:
    dist = pos
    print(f'\nPosition {pos} (PC byte {pos-1}, dist={dist}):')

    # Byte 1 matching: L1H2 - L1H1 fires at d∈(1.5, 2.5]
    l1h2_val = x[0, pos, BD.L1H2 + PC_I].item()
    l1h1_val = x[0, pos, BD.L1H1 + PC_I].item()
    diff_1 = l1h2_val - l1h1_val
    print(f'  Byte 1 match (L1H2 - L1H1): {l1h2_val:.3f} - {l1h1_val:.3f} = {diff_1:.3f}')
    if pos == 2:
        print(f'    ✓ Should fire at pos 2 (d=2 ∈ (1.5, 2.5])')

    # Byte 2 matching: H0 - L1H2 fires at d∈(2.5, 3.5]
    h0_val = x[0, pos, BD.H0 + PC_I].item()
    diff_2 = h0_val - l1h2_val
    print(f'  Byte 2 match (H0 - L1H2): {h0_val:.3f} - {l1h2_val:.3f} = {diff_2:.3f}')
    if pos == 3:
        print(f'    ✓ Should fire at pos 3 (d=3 ∈ (2.5, 3.5])')

    # Byte 3 matching: H1 - H0 fires at d∈(3.5, 4.5]
    h1_val = x[0, pos, BD.H1 + PC_I].item()
    diff_3 = h1_val - h0_val
    print(f'  Byte 3 match (H1 - H0): {h1_val:.3f} - {h0_val:.3f} = {diff_3:.3f}')
    if pos == 4:
        print(f'    ✓ Should fire at pos 4 (d=4 ∈ (3.5, 4.5])')

print('\n\n[ANALYSIS]')
print('For the hop-count fix to work, threshold differences must be:')
print('  - Positive at the target position')
print('  - Zero or negative at other positions')
print('If all hop counts are correct, the bug is elsewhere (attention calculation).')
