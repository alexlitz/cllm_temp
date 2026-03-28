#!/usr/bin/env python3
"""Debug the cancel-pair pattern logic."""

import torch

# Test the threshold logic for cancel-pair pattern
# Goal: Fire only when NIB_A=3 and NIB_B=5

S = 100.0  # Scale factor

# Current implementation (WRONG?)
# Positive: fires when NIB_A >= a+0.5 (3.5) AND NIB_B >= b+0.5 (5.5)
# Negative: fires when NIB_A >= a+1.5 (4.5) AND NIB_B >= b+1.5 (6.5)

print("Current implementation:")
print("  Positive thresholds: NIB_A >= 3.5, NIB_B >= 5.5")
print("  Negative thresholds: NIB_A >= 4.5, NIB_B >= 6.5")
print()

for nib_a in [2, 3, 4, 5]:
    for nib_b in [4, 5, 6, 7]:
        # Positive unit
        up_pos = S * nib_a - S * 3.5
        gate_pos = S * nib_b - S * 5.5

        # Negative unit
        up_neg = S * nib_a - S * 4.5
        gate_neg = S * nib_b - S * 6.5

        # SwiGLU: hidden = relu(up) * relu(gate)
        hidden_pos = max(0, up_pos) * max(0, gate_pos) / S  # Simplified
        hidden_neg = max(0, up_neg) * max(0, gate_neg) / S

        net = hidden_pos - hidden_neg
        fires = "YES" if net > 0.5 else "no"

        print(f"  ({nib_a}, {nib_b}): pos={hidden_pos:6.1f}, neg={hidden_neg:6.1f}, net={net:6.1f} → {fires}")

print()
print("=" * 70)
print()

# Corrected implementation
# For discrete values, we want to fire only when NIB_A=3 (not 3.5)
# Positive: fires when NIB_A >= a-0.5 (2.5) AND NIB_B >= b-0.5 (4.5)
# Negative: fires when NIB_A >= a+0.5 (3.5) OR NIB_B >= b+0.5 (5.5)

print("Corrected implementation:")
print("  Positive thresholds: NIB_A >= 2.5, NIB_B >= 4.5")
print("  Negative thresholds: NIB_A >= 3.5, NIB_B >= 5.5")
print()

for nib_a in [2, 3, 4, 5]:
    for nib_b in [4, 5, 6, 7]:
        # Positive unit
        up_pos = S * nib_a - S * 2.5
        gate_pos = S * nib_b - S * 4.5

        # Negative unit
        up_neg = S * nib_a - S * 3.5
        gate_neg = S * nib_b - S * 5.5

        # SwiGLU: hidden = relu(up) * relu(gate)
        hidden_pos = max(0, up_pos) * max(0, gate_pos) / S  # Simplified
        hidden_neg = max(0, up_neg) * max(0, gate_neg) / S

        net = hidden_pos - hidden_neg
        fires = "YES" if net > 0.5 else "no"

        print(f"  ({nib_a}, {nib_b}): pos={hidden_pos:6.1f}, neg={hidden_neg:6.1f}, net={net:6.1f} → {fires}")
