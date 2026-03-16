"""
Neural-native schoolbook multiplication layers for 32-bit MUL.

Schoolbook algorithm: result[k] = sum of a[i] * b[k-i] for all valid i,j pairs.

Uses attention to route b values to correct positions, then accumulates
partial products. The ZeroInvalidPositions layer masks invalid positions.
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention
from .pure_moe import MoE


class ZeroInvalidPositionsFFN(PureFFN):
    """
    Zero a slot at positions 0..offset-1 using SwiGLU step function.

    Uses step(POS < offset) = [silu(S*(offset-POS)) - silu(S*(offset-1-POS))]/S
    to produce -slot_value at positions < offset, canceling via residual.

    Unconditional — harmless for non-MUL ops since slot is already 0.
    """
    def __init__(self, offset: int, slot: int):
        self._offset = offset
        self._slot = slot
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Unit 0: silu(S*(offset - POS)) * (-slot_value) / S
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * self._offset
            self.W_gate[0, self._slot] = -1.0
            self.W_down[self._slot, 0] = 1.0 / S

            # Unit 1: silu(S*(offset-1 - POS)) * (slot_value) / S
            self.W_up[1, E.POS] = -S
            self.b_up[1] = S * (self._offset - 1)
            self.W_gate[1, self._slot] = 1.0
            self.W_down[self._slot, 1] = 1.0 / S


class ClearSlotForMulFFN(PureFFN):
    """Clear a specific slot unconditionally (opcode gating done by SoftMoE wrapper)."""
    def __init__(self, slot: int):
        self.slot = slot
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Unconditional clear - SoftMoE handles opcode gating
            self.b_up[0] = S
            self.W_gate[0, self.slot] = -1.0
            self.W_down[self.slot, 0] = 1.0 / S

            self.b_up[1] = -S
            self.W_gate[1, self.slot] = 1.0
            self.W_down[self.slot, 1] = 1.0 / S


class BroadcastAAttention(PureAttention):
    """Broadcast a[src_pos] to a destination slot at all positions."""
    def __init__(self, src_pos: int, dest_slot: int):
        self.src_pos = src_pos
        self.dest_slot = dest_slot
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(N):
            mask[k, src_pos] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[self.dest_slot, E.NIB_A] = 1.0
            self.W_o[self.dest_slot, self.dest_slot] = 1.0


class ShiftBAttention(PureAttention):
    """
    Shift b values: position k reads b[k-offset] to destination slot.
    Invalid positions (k < offset) read from self (will be zeroed later).
    Opcode gating done by SoftMoE wrapper.
    """
    def __init__(self, offset: int, dest_slot: int):
        self.offset = offset
        self.dest_slot = dest_slot
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for k in range(N):
            src = k - offset
            if 0 <= src < N:
                mask[k, src] = 0.0
            else:
                # Invalid: read from self (will be zeroed by ZeroInvalidPositions)
                mask[k, k] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[self.dest_slot, E.NIB_B] = 1.0
            self.W_o[self.dest_slot, self.dest_slot] = 1.0


class MulAccumFFN(PureFFN):
    """Multiply a_slot * b_slot and accumulate to RESULT."""
    def __init__(self, a_slot: int, b_slot: int):
        self.a_slot = a_slot
        self.b_slot = b_slot
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # product = silu(S * b) * a / S ≈ a * b for positive values
            self.W_up[0, self.b_slot] = S
            self.W_gate[0, self.a_slot] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Symmetric term for negative handling
            self.W_up[1, self.b_slot] = -S
            self.W_gate[1, self.a_slot] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class ClearResultForMulFFN(PureFFN):
    """Clear RESULT slot before multiplication, gated by MUL opcode."""
    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.RESULT] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


def build_schoolbook_mul_layers():
    """
    Build fused double-offset schoolbook multiplication layers.

    Processes pairs of offsets simultaneously using 4 temp slots:
    - Pair 1: a_slot1=TEMP(6), b_slot1=TEMP+1(7)
    - Pair 2: a_slot2=TEMP_A2(157), b_slot2=TEMP_B2(158)

    For each pair (2i, 2i+1), 4 layers:
    1. Clear 4 temp slots
    2. Broadcast+Shift for both offsets
    3. ZeroInvalid for both offsets
    4. Accumulate both products

    Returns list of MoE-wrapped layers (17 total, down from 32).
    All layers are gated by MUL opcode via MoE.
    """
    layers = []

    a_slot1 = E.TEMP        # 6
    b_slot1 = E.TEMP + 1    # 7
    a_slot2 = E.TEMP_A2     # 157
    b_slot2 = E.TEMP_B2     # 158

    # Clear result first (wrapped in MoE for opcode gating)
    layers.append(MoE([ClearResultForMulFFN()], [Opcode.MUL]))

    # Process 4 pairs: (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        offset1 = pair_idx * 2       # 0, 2, 4, 6
        offset2 = pair_idx * 2 + 1   # 1, 3, 5, 7

        # Step 1: Clear 4 temp slots
        layers.append(MoE(
            [ClearSlotForMulFFN(a_slot1), ClearSlotForMulFFN(b_slot1),
             ClearSlotForMulFFN(a_slot2), ClearSlotForMulFFN(b_slot2)],
            [Opcode.MUL, Opcode.MUL, Opcode.MUL, Opcode.MUL]
        ))

        # Step 2: Broadcast a + Shift b for both offsets
        layers.append(MoE(
            [BroadcastAAttention(offset1, a_slot1),
             ShiftBAttention(offset1, b_slot1),
             BroadcastAAttention(offset2, a_slot2),
             ShiftBAttention(offset2, b_slot2)],
            [Opcode.MUL, Opcode.MUL, Opcode.MUL, Opcode.MUL]
        ))

        # Step 3: Zero invalid positions (MoE-wrapped)
        zero_experts = []
        zero_opcodes = []
        if offset1 > 0:
            zero_experts.append(ZeroInvalidPositionsFFN(offset1, b_slot1))
            zero_opcodes.append(Opcode.MUL)
        # offset2 is always >= 1
        zero_experts.append(ZeroInvalidPositionsFFN(offset2, b_slot2))
        zero_opcodes.append(Opcode.MUL)
        layers.append(MoE(zero_experts, zero_opcodes))

        # Step 4: Accumulate both products
        layers.append(MoE(
            [MulAccumFFN(a_slot1, b_slot1), MulAccumFFN(a_slot2, b_slot2)],
            [Opcode.MUL, Opcode.MUL]
        ))

    return layers


def test_schoolbook_mul():
    """Test the schoolbook multiplication layers."""
    layers = build_schoolbook_mul_layers()
    model = nn.Sequential(*layers)

    def test_mul(a: int, b: int) -> int:
        N = E.NUM_POSITIONS
        a_nibbles = [(a >> (4*i)) & 0xF for i in range(N)]
        b_nibbles = [(b >> (4*i)) & 0xF for i in range(N)]

        x = torch.zeros(1, N, E.DIM)
        x[:, :, 0] = 1.0
        x[:, :, E.POS] = torch.arange(N).float()
        for i in range(N):
            x[:, i, E.NIB_A] = a_nibbles[i]
            x[:, i, E.NIB_B] = b_nibbles[i]
        # Set MUL opcode (one-hot at all positions)
        x[:, :, E.OP_START + Opcode.MUL] = 1.0

        x = model(x)

        # Convert raw result back
        result = sum(int(round(x[0, i, E.RESULT].item())) * (16 ** i) for i in range(N))
        return result & 0xFFFFFFFF

    tests = [
        (2, 3, 6),
        (5, 5, 25),
        (15, 15, 225),
        (255, 255, 65025),
        (1000, 1000, 1000000),
        (65535, 65535, 0xFFFE0001),
    ]

    passed = 0
    for a, b, expected in tests:
        result = test_mul(a, b)
        if result == expected:
            passed += 1
            print(f"  {a} * {b} = {result} ✓")
        else:
            print(f"  {a} * {b} = {result} (expected {expected}) ✗")

    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("Testing schoolbook multiplication layers...")
    test_schoolbook_mul()
