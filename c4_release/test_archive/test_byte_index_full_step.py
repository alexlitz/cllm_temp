#!/usr/bin/env python3
"""Test BYTE_INDEX values in a full 35-token step."""

from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD, set_vm_weights
import torch

# Create model
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)  # IMPORTANT: Configure weights!

# Create a full 35-token step
# PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1)
context = [
    # PC section
    Token.REG_PC, 0x00, 0x00, 0x00, 0x00,
    # AX section
    Token.REG_AX, 0x2A, 0x00, 0x00, 0x00,
    # SP section
    Token.REG_SP, 0xF8, 0xF7, 0x01, 0x00,
    # BP section
    Token.REG_BP, 0x00, 0x00, 0x01, 0x00,
    # STACK0 section
    Token.STACK0, 0x2A, 0x00, 0x00, 0x00,
    # MEM section
    Token.MEM, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    # STEP_END
    Token.STEP_END,
]

print("=" * 70)
print("BYTE_INDEX Test with Full 35-Token Step")
print("=" * 70)

input_ids = torch.tensor([context])

# Run through model layers
x = model.embed(input_ids)
x = model.blocks[0](x)  # L0: threshold attention
x = model.blocks[1](x)  # L1: BYTE_INDEX generation

# Check BYTE_INDEX at SP byte positions
SP_START = 10  # SP section starts at position 10
print("\nSP section BYTE_INDEX values:")
for i in range(5):
    pos = SP_START + i
    pos_name = ['SP marker', 'SP byte 0', 'SP byte 1', 'SP byte 2', 'SP byte 3'][i]

    byte_idx_0 = x[0, pos, BD.BYTE_INDEX_0].item()
    byte_idx_1 = x[0, pos, BD.BYTE_INDEX_1].item()
    byte_idx_2 = x[0, pos, BD.BYTE_INDEX_2].item()
    byte_idx_3 = x[0, pos, BD.BYTE_INDEX_3].item()

    # Highlight which should be high
    expected = ['none', '0', '1', '2', '3'][i]
    marker = ''
    if i == 1 and byte_idx_0 > 1.0:
        marker = ' ✓'
    elif i == 2 and byte_idx_1 > 1.0:
        marker = ' ✓'
    elif i == 3 and byte_idx_2 > 1.0:
        marker = ' ✓'
    elif i == 4 and byte_idx_3 > 1.0:
        marker = ' ✓'
    elif i > 0:
        marker = ' ✗ WRONG'

    print(f"  {pos_name:12s} (expect {expected}): "
          f"_0={byte_idx_0:6.2f}, _1={byte_idx_1:6.2f}, "
          f"_2={byte_idx_2:6.2f}, _3={byte_idx_3:6.2f}{marker}")

# Also check hop-count threshold values at SP positions
print("\nSP section hop-count threshold values:")
for i in range(5):
    pos = SP_START + i
    pos_name = ['SP marker', 'SP byte 0', 'SP byte 1', 'SP byte 2', 'SP byte 3'][i]

    l1h0 = x[0, pos, BD.L1H0 + 2].item()  # SP is marker index 2
    l1h1 = x[0, pos, BD.L1H1 + 2].item()
    l1h2 = x[0, pos, BD.L1H2 + 2].item()
    h0 = x[0, pos, BD.H0 + 2].item()
    h1 = x[0, pos, BD.H1 + 2].item()

    # Distance from SP marker
    d = i
    print(f"  {pos_name:12s} (d={d}): "
          f"L1H0={l1h0:5.2f}, L1H1={l1h1:5.2f}, L1H2={l1h2:5.2f}, "
          f"H0={h0:5.2f}, H1={h1:5.2f}")

print("\n" + "=" * 70)
