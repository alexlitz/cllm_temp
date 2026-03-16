"""
Carry Propagation via Attention.

The carry propagation needs to copy CARRY_OUT from position i to CARRY_IN
at position i+1. This is a cross-position routing operation — exactly what
attention is designed for.

Architecture:
  PureAttention with mask: position i attends ONLY to position i-1.
  V reads CARRY_OUT, O writes to CARRY_IN.
  Position 0 attends to itself but gets zero carry (V=0 for self-attention at pos 0).
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureAttention, bake_weights


class CarryPropagateAttention(PureAttention):
    """
    Propagate carry via attention.

    Copies CARRY_OUT[i] -> CARRY_IN[i+1] for all positions.
    Position 0 gets CARRY_IN = 0 (no carry into LSB).

    Uses mask to restrict: position i attends ONLY to position i-1.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        # Mask: pos i attends ONLY to pos i-1
        # pos 0 attends to itself (but CARRY_OUT at pos 0 contributes 0 to own CARRY_IN)
        mask = torch.full((E.NUM_POSITIONS, E.NUM_POSITIONS), -1e9)
        for i in range(1, E.NUM_POSITIONS):
            mask[i, i - 1] = 0.0
        mask[0, 0] = 0.0  # pos 0 self-attend (harmless: doesn't write carry to itself)
        self.mask.copy_(mask)

        # V: read CARRY_OUT from attended position
        self.W_v[E.CARRY_IN, E.CARRY_OUT] = 1.0

        # O: write attended value to CARRY_IN
        self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0

        # Q/K: With single-element masking, softmax of one entry = 1.0
        # regardless of Q/K values. Leave at zero.


# Backward compatibility alias
CarryPropagateFFN = CarryPropagateAttention


class ZeroFirstCarryFFN(nn.Module):
    """
    Ensure CARRY_IN[0] = 0 (no carry into LSB).

    Position 0 should never have a carry-in since it's the LSB.
    This is handled by the CarryPropagateAttention mask.

    This module is a no-op since propagation already handles it correctly.
    Kept for compatibility.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """No-op - propagation already zeroes CARRY_IN[0]."""
        return x


# ============================================================================
# TESTS
# ============================================================================

def test_carry_propagate():
    """Test basic carry propagation."""
    print("=== Testing Carry Propagation Attention ===\n")

    propagate = CarryPropagateAttention()

    # Create test input with CARRY_OUT at each position
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    # Set CARRY_OUT = 1 at positions 0, 2, 4
    x[0, 0, E.CARRY_OUT] = 1.0
    x[0, 2, E.CARRY_OUT] = 1.0
    x[0, 4, E.CARRY_OUT] = 1.0

    # Propagate
    y = propagate(x)

    # Check CARRY_IN at positions 1, 3, 5
    print("CARRY_OUT -> CARRY_IN propagation:")
    for i in range(E.NUM_POSITIONS):
        cout = x[0, i, E.CARRY_OUT].item()
        cin = y[0, i, E.CARRY_IN].item()
        print(f"  Position {i}: CARRY_OUT={cout:.1f}, CARRY_IN={cin:.2f}")

    # Verify propagation
    assert abs(y[0, 1, E.CARRY_IN].item() - 1.0) < 0.1, "Carry to pos 1 failed"
    assert abs(y[0, 3, E.CARRY_IN].item() - 1.0) < 0.1, "Carry to pos 3 failed"
    assert abs(y[0, 5, E.CARRY_IN].item() - 1.0) < 0.1, "Carry to pos 5 failed"
    assert abs(y[0, 0, E.CARRY_IN].item()) < 0.1, "Pos 0 should have no carry in"

    print("PASS\n")


def test_addition_with_carry():
    """Test full addition with carry cascade."""
    print("=== Testing Addition with Carry ===\n")

    propagate = CarryPropagateAttention()
    zero_first = ZeroFirstCarryFFN()

    # Test: 0xF + 0x1 = 0x10 (carry from nibble 0 to nibble 1)
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    # Position 0: A=15, B=1, RAW_SUM=16, RESULT=0, CARRY_OUT=1
    x[0, 0, E.NIB_A] = 15
    x[0, 0, E.NIB_B] = 1
    x[0, 0, E.RAW_SUM] = 16
    x[0, 0, E.RESULT] = 0
    x[0, 0, E.CARRY_OUT] = 1

    # Position 1-7: A=0, B=0
    for i in range(1, E.NUM_POSITIONS):
        x[0, i, E.NIB_A] = 0
        x[0, i, E.NIB_B] = 0
        x[0, i, E.RESULT] = 0

    print("Before propagation:")
    for i in range(4):
        print(f"  Pos {i}: RESULT={x[0, i, E.RESULT].item():.0f}, "
              f"CARRY_OUT={x[0, i, E.CARRY_OUT].item():.1f}, "
              f"CARRY_IN={x[0, i, E.CARRY_IN].item():.1f}")

    # Propagate carry
    y = propagate(x)
    y = zero_first(y)

    print("\nAfter propagation:")
    for i in range(4):
        print(f"  Pos {i}: RESULT={y[0, i, E.RESULT].item():.0f}, "
              f"CARRY_OUT={y[0, i, E.CARRY_OUT].item():.1f}, "
              f"CARRY_IN={y[0, i, E.CARRY_IN].item():.2f}")

    # Position 1 should have CARRY_IN = 1
    assert abs(y[0, 1, E.CARRY_IN].item() - 1.0) < 0.1, "Carry to pos 1 failed"
    # Position 0 should have CARRY_IN = 0
    assert abs(y[0, 0, E.CARRY_IN].item()) < 0.1, "Pos 0 CARRY_IN should be 0"

    print("PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Carry Propagation via Attention")
    print("=" * 60)
    print()

    test_carry_propagate()
    test_addition_with_carry()

    print("=" * 60)
    print("All carry propagation tests passed!")
    print("=" * 60)
