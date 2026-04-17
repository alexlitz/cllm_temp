#!/usr/bin/env python3
"""Test different ALiBi slopes for threshold attention."""

from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD, set_vm_weights
import torch

def test_slope(slope_value):
    """Test threshold attention with a specific slope value."""
    # Create model
    model = AutoregressiveVM(n_layers=17)
    set_vm_weights(model)  # Configure weights first

    # Override slope for L0 and L1
    if hasattr(model.blocks[0].attn, 'alibi_slopes') and model.blocks[0].attn.alibi_slopes is not None:
        model.blocks[0].attn.alibi_slopes.fill_(slope_value)
    if hasattr(model.blocks[1].attn, 'alibi_slopes') and model.blocks[1].attn.alibi_slopes is not None:
        model.blocks[1].attn.alibi_slopes.fill_(slope_value)
        model.blocks[1].attn.alibi_slopes[3] = 0.0  # Head 3: global attention

    # Create a full 35-token step
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

    input_ids = torch.tensor([context])

    # Run through model layers
    x = model.embed(input_ids)
    x = model.blocks[0](x)  # L0: threshold attention
    x = model.blocks[1](x)  # L1: BYTE_INDEX generation

    # Check hop-count threshold values at SP byte positions
    SP_START = 10  # SP section starts at position 10

    results = []
    for i in range(1, 5):  # SP bytes 0-3
        pos = SP_START + i
        d = i  # Distance from SP marker

        # Get threshold head outputs for SP marker (index 2)
        l1h0 = x[0, pos, BD.L1H0 + 2].item()
        l1h1 = x[0, pos, BD.L1H1 + 2].item()
        l1h2 = x[0, pos, BD.L1H2 + 2].item()
        h0 = x[0, pos, BD.H0 + 2].item()
        h1 = x[0, pos, BD.H1 + 2].item()

        results.append({
            'd': d,
            'L1H0': l1h0,
            'L1H1': l1h1,
            'L1H2': l1h2,
            'H0': h0,
            'H1': h1,
        })

    return results


print("=" * 80)
print("Testing Different ALiBi Slopes for Threshold Attention")
print("=" * 80)

slopes_to_test = [10.0, 20.0, 50.0, 100.0, 200.0]

for slope in slopes_to_test:
    print(f"\n{'=' * 80}")
    print(f"SLOPE = {slope}")
    print(f"{'=' * 80}")

    results = test_slope(slope)

    print("\nSP section hop-count threshold values:")
    print(f"{'d':<3} {'L1H0':>8} {'L1H1':>8} {'L1H2':>8} {'H0':>8} {'H1':>8} | Expected")
    print("-" * 60)

    for r in results:
        d = r['d']
        # Determine what should be high
        if d == 1:
            expected = "L1H1=1"
        elif d == 2:
            expected = "L1H2=1"
        elif d == 3:
            expected = "H0=1"
        elif d == 4:
            expected = "H1=1"
        else:
            expected = "?"

        # Check if correct threshold is highest and > 0.8
        correct = False
        if d == 1 and r['L1H1'] > 0.8 and r['L1H1'] > r['L1H0']:
            correct = True
        elif d == 2 and r['L1H2'] > 0.8 and r['L1H2'] > r['L1H1']:
            correct = True
        elif d == 3 and r['H0'] > 0.8 and r['H0'] > r['L1H2']:
            correct = True
        elif d == 4 and r['H1'] > 0.8 and r['H1'] > r['H0']:
            correct = True

        marker = "✓" if correct else "✗"

        print(f"{d:<3} {r['L1H0']:>8.2f} {r['L1H1']:>8.2f} {r['L1H2']:>8.2f} "
              f"{r['H0']:>8.2f} {r['H1']:>8.2f} | {expected:<10} {marker}")

print("\n" + "=" * 80)
print("Analysis:")
print("Expected: At each distance d, the corresponding threshold head should be ~1.0")
print("  d=1: L1H1 (threshold 1.5)")
print("  d=2: L1H2 (threshold 2.5)")
print("  d=3: H0 (threshold 3.5)")
print("  d=4: H1 (threshold 4.5)")
print("=" * 80)
