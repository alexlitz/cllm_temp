"""
32-bit DIV/MOD via flattened FFN approach.

The key insight: Standard per-position FFNs can't do cross-position division.
We need to:
1. Gather all nibbles into a single scalar representation
2. Perform division/modulo using step functions
3. Scatter results back to nibbles

For practical 32-bit division with arbitrary divisors, we use iterative
subtraction with flattened FFN that can access all positions.

Architecture:
  - GatherValueFFN: Sum weighted nibbles into a single accumulator slot
  - DivModIterFFN: Flattened FFN for iterative subtraction
  - ScatterResultFFN: Distribute result nibbles back to positions
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, bake_weights


class GatherValueFFN(nn.Module):
    """
    Gather 8 nibbles into a single 32-bit value stored in accumulator.

    Uses flattened FFN: [batch, 8, dim] -> [batch, 8*dim]

    value = sum(NIB[i] * 16^i) for i in 0..7

    Stores result in TEMP slot at position 0.
    """

    def __init__(self, source_slot: int, dest_slot: int = E.TEMP, opcode: int = Opcode.DIV):
        super().__init__()
        self.source_slot = source_slot
        self.dest_slot = dest_slot
        self.opcode = opcode
        self.num_positions = E.NUM_POSITIONS
        self.dim = E.DIM
        self.flat_dim = self.num_positions * self.dim

        # Hidden units: 1 per position for gathering + 2 for opcode gating
        hidden_dim = self.num_positions + 2

        self.register_buffer('W_up', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_up', torch.zeros(hidden_dim))
        self.register_buffer('W_gate', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_gate', torch.zeros(hidden_dim))
        self.register_buffer('W_down', torch.zeros(self.flat_dim, hidden_dim))
        self.register_buffer('b_down', torch.zeros(self.flat_dim))

        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # For each position, add NIB * 16^pos to dest_slot at position 0
        # Use silu(opcode_S) * NIB * 16^i / S pattern
        for i in range(self.num_positions):
            h = i
            src_idx = self._flat_idx(i, self.source_slot)
            opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
            dst_idx = self._flat_idx(0, self.dest_slot)

            # silu(S * opcode) * NIB * weight
            self.W_up[h, opcode_idx] = S
            self.W_gate[h, src_idx] = float(16 ** i)  # Weight by position
            self.W_down[dst_idx, h] = 1.0 / S

        # Saturation terms for negative opcode
        h = self.num_positions
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)
        dst_idx = self._flat_idx(0, self.dest_slot)

        self.W_up[h, opcode_idx] = -S
        total_gate = sum(16 ** i for i in range(self.num_positions))
        for i in range(self.num_positions):
            src_idx = self._flat_idx(i, self.source_slot)
            self.W_gate[h, src_idx] = -float(16 ** i)
        self.W_down[dst_idx, h] = 1.0 / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, N * D)

        up = torch.nn.functional.linear(x_flat, self.W_up, self.b_up)
        gate = torch.nn.functional.linear(x_flat, self.W_gate, self.b_gate)
        hidden = torch.nn.functional.silu(up) * gate
        delta = torch.nn.functional.linear(hidden, self.W_down, self.b_down)

        out_flat = x_flat + delta
        return out_flat.reshape(B, N, D)


class DivIterFlatFFN(nn.Module):
    """
    One iteration of 32-bit division using flattened FFN.

    Operates on gathered values in position 0:
    - TEMP = dividend (gathered)
    - TEMP+1 = divisor (gathered)
    - RESULT = quotient (accumulated)

    Each iteration: if TEMP >= TEMP+1, subtract and add 1 to RESULT.
    After all iterations, TEMP contains remainder.
    """

    def __init__(self, opcode: int = Opcode.DIV):
        super().__init__()
        self.opcode = opcode
        self.num_positions = E.NUM_POSITIONS
        self.dim = E.DIM
        self.flat_dim = self.num_positions * self.dim

        # 4 hidden units for the comparison-based subtraction
        hidden_dim = 4

        self.register_buffer('W_up', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_up', torch.zeros(hidden_dim))
        self.register_buffer('W_gate', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_gate', torch.zeros(hidden_dim))
        self.register_buffer('W_down', torch.zeros(self.flat_dim, hidden_dim))
        self.register_buffer('b_down', torch.zeros(self.flat_dim))

        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # All values are at position 0
        dividend_idx = self._flat_idx(0, E.TEMP)  # Current remainder
        divisor_idx = self._flat_idx(0, E.TEMP + 1)  # Divisor
        quotient_idx = self._flat_idx(0, E.RESULT)  # Quotient accumulator
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        # Rows 0-1: Subtract divisor from dividend when dividend >= divisor
        # Use wider step for sharp transition: threshold at +1 and 0 instead of +0.5/-0.5
        # This ensures clean step at integer boundaries

        # Row 0: silu(S*(dividend - divisor + 1)) * divisor * (-1/S)
        self.W_up[0, dividend_idx] = S
        self.W_up[0, divisor_idx] = -S
        self.b_up[0] = S * 1.0  # Upper threshold
        self.W_gate[0, divisor_idx] = 1.0  # Gate by divisor value for subtraction
        self.W_down[dividend_idx, 0] = -1.0 / S

        # Row 1: Saturation term - threshold at 0
        self.W_up[1, dividend_idx] = S
        self.W_up[1, divisor_idx] = -S
        self.b_up[1] = 0.0  # Lower threshold at exact equality
        self.W_gate[1, divisor_idx] = 1.0
        self.W_down[dividend_idx, 1] = 1.0 / S

        # Rows 2-3: Add 1 to quotient when dividend >= divisor
        # Row 2: silu(S*(dividend - divisor + 1)) * 1 * (1/S)
        self.W_up[2, dividend_idx] = S
        self.W_up[2, divisor_idx] = -S
        self.b_up[2] = S * 1.0
        self.W_gate[2, opcode_idx] = 1.0  # Gate by opcode (constant 1 when active)
        self.W_down[quotient_idx, 2] = 1.0 / S

        # Row 3: Saturation term - threshold at 0
        self.W_up[3, dividend_idx] = S
        self.W_up[3, divisor_idx] = -S
        self.b_up[3] = 0.0
        self.W_gate[3, opcode_idx] = 1.0
        self.W_down[quotient_idx, 3] = -1.0 / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, N * D)

        up = torch.nn.functional.linear(x_flat, self.W_up, self.b_up)
        gate = torch.nn.functional.linear(x_flat, self.W_gate, self.b_gate)
        hidden = torch.nn.functional.silu(up) * gate
        delta = torch.nn.functional.linear(hidden, self.W_down, self.b_down)

        out_flat = x_flat + delta
        return out_flat.reshape(B, N, D)


class CopyToTempFFN(nn.Module):
    """Copy source_slot[0] to temp_slot[0] for scatter source preservation."""

    def __init__(self, source_slot: int, temp_slot: int, opcode: int):
        super().__init__()
        self.source_slot = source_slot
        self.temp_slot = temp_slot
        self.opcode = opcode
        self.num_positions = E.NUM_POSITIONS
        self.dim = E.DIM
        self.flat_dim = self.num_positions * self.dim

        hidden_dim = 2

        self.register_buffer('W_up', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_up', torch.zeros(hidden_dim))
        self.register_buffer('W_gate', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_gate', torch.zeros(hidden_dim))
        self.register_buffer('W_down', torch.zeros(self.flat_dim, hidden_dim))
        self.register_buffer('b_down', torch.zeros(self.flat_dim))

        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        src_idx = self._flat_idx(0, self.source_slot)
        dst_idx = self._flat_idx(0, self.temp_slot)
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        # Copy source to temp
        self.W_up[0, opcode_idx] = S
        self.W_gate[0, src_idx] = 1.0
        self.W_down[dst_idx, 0] = 1.0 / S

        self.W_up[1, opcode_idx] = -S
        self.W_gate[1, src_idx] = -1.0
        self.W_down[dst_idx, 1] = 1.0 / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, N * D)
        up = torch.nn.functional.linear(x_flat, self.W_up, self.b_up)
        gate = torch.nn.functional.linear(x_flat, self.W_gate, self.b_gate)
        hidden = torch.nn.functional.silu(up) * gate
        delta = torch.nn.functional.linear(hidden, self.W_down, self.b_down)
        out_flat = x_flat + delta
        return out_flat.reshape(B, N, D)


class ClearAllResultsFFN(nn.Module):
    """Clear RESULT slot at all positions."""

    def __init__(self, opcode: int):
        super().__init__()
        self.opcode = opcode
        self.num_positions = E.NUM_POSITIONS
        self.dim = E.DIM
        self.flat_dim = self.num_positions * self.dim

        hidden_dim = self.num_positions * 2

        self.register_buffer('W_up', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_up', torch.zeros(hidden_dim))
        self.register_buffer('W_gate', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_gate', torch.zeros(hidden_dim))
        self.register_buffer('W_down', torch.zeros(self.flat_dim, hidden_dim))
        self.register_buffer('b_down', torch.zeros(self.flat_dim))

        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        h = 0
        for pos in range(self.num_positions):
            result_idx = self._flat_idx(pos, E.RESULT)

            self.W_up[h, opcode_idx] = S
            self.W_gate[h, result_idx] = -1.0
            self.W_down[result_idx, h] = 1.0 / S
            h += 1

            self.W_up[h, opcode_idx] = -S
            self.W_gate[h, result_idx] = 1.0
            self.W_down[result_idx, h] = 1.0 / S
            h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, N * D)
        up = torch.nn.functional.linear(x_flat, self.W_up, self.b_up)
        gate = torch.nn.functional.linear(x_flat, self.W_gate, self.b_gate)
        hidden = torch.nn.functional.silu(up) * gate
        delta = torch.nn.functional.linear(hidden, self.W_down, self.b_down)
        out_flat = x_flat + delta
        return out_flat.reshape(B, N, D)


class ScatterResultFFN(nn.Module):
    """
    Scatter a 32-bit value from CARRY_OUT[0] back to 8 nibble positions in RESULT.

    For position i: RESULT[i] = (value >> (4*i)) & 0xF

    This reads from CARRY_OUT[0] (preserved during clearing) and writes to RESULT.
    Uses step functions to extract each nibble with proper mod 16 wrapping.
    """

    def __init__(self, source_slot: int = E.CARRY_OUT, dest_slot: int = E.RESULT,
                 opcode: int = Opcode.DIV):
        super().__init__()
        self.source_slot = source_slot  # Read from CARRY_OUT (preserved)
        self.dest_slot = dest_slot
        self.opcode = opcode
        self.num_positions = E.NUM_POSITIONS
        self.dim = E.DIM
        self.flat_dim = self.num_positions * self.dim

        # For each position: 15 step functions (1-15) + 15 wrap functions (for mod 16)
        # 4 positions * (15 + 15) * 2 = 240 hidden units
        # But for simplicity, let's support values up to 255 (positions 0-1)
        # Position 0: count 1-15, subtract at 16,32,...
        # Position 1: count 1-15 at scale 16, wrap at 256,512,...
        hidden_dim = 200

        self.register_buffer('W_up', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_up', torch.zeros(hidden_dim))
        self.register_buffer('W_gate', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_gate', torch.zeros(hidden_dim))
        self.register_buffer('W_down', torch.zeros(self.flat_dim, hidden_dim))
        self.register_buffer('b_down', torch.zeros(self.flat_dim))

        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        src_idx = self._flat_idx(0, self.source_slot)
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        h = 0

        # Position 0: Extract value mod 16
        # Copy value directly, then subtract 16 for each multiple of 16
        pos = 0
        dst_idx = self._flat_idx(pos, self.dest_slot)

        # Copy value to position 0
        self.W_up[h, opcode_idx] = S
        self.W_gate[h, src_idx] = 1.0
        self.W_down[dst_idx, h] = 1.0 / S
        h += 1

        self.W_up[h, opcode_idx] = -S
        self.W_gate[h, src_idx] = -1.0
        self.W_down[dst_idx, h] = 1.0 / S
        h += 1

        # Wrap: subtract 16 for each multiple of 16 (up to 256)
        for mult in range(1, 17):  # Up to 256 for values up to ~256
            threshold = mult * 16
            self.W_up[h, src_idx] = S
            self.b_up[h] = -S * (threshold - 1)
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[dst_idx, h] = -16.0 / S
            h += 1

            self.W_up[h, src_idx] = S
            self.b_up[h] = -S * threshold
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[dst_idx, h] = 16.0 / S
            h += 1

        # Position 1: Extract (value // 16) mod 16
        # Add 1 for each multiple of 16 crossed, wrap at 256
        pos = 1
        dst_idx = self._flat_idx(pos, self.dest_slot)

        # Step up for each 16*k where k=1-15
        for k in range(1, 16):
            threshold = k * 16
            self.W_up[h, src_idx] = S
            self.b_up[h] = -S * (threshold - 1)
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[dst_idx, h] = 1.0 / S
            h += 1

            self.W_up[h, src_idx] = S
            self.b_up[h] = -S * threshold
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[dst_idx, h] = -1.0 / S
            h += 1

        # Wrap: subtract 16 for each multiple of 256
        for mult in range(1, 5):  # Up to 1024
            threshold = mult * 256
            self.W_up[h, src_idx] = S
            self.b_up[h] = -S * (threshold - 1)
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[dst_idx, h] = -16.0 / S
            h += 1

            self.W_up[h, src_idx] = S
            self.b_up[h] = -S * threshold
            self.W_gate[h, opcode_idx] = 1.0
            self.W_down[dst_idx, h] = 16.0 / S
            h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, N * D)

        up = torch.nn.functional.linear(x_flat, self.W_up, self.b_up)
        gate = torch.nn.functional.linear(x_flat, self.W_gate, self.b_gate)
        hidden = torch.nn.functional.silu(up) * gate
        delta = torch.nn.functional.linear(hidden, self.W_down, self.b_down)

        out_flat = x_flat + delta
        return out_flat.reshape(B, N, D)


class ClearDivSlotsFFN(nn.Module):
    """Clear temporary slots before DIV/MOD operation."""

    def __init__(self, opcode: int = Opcode.DIV):
        super().__init__()
        self.opcode = opcode
        self.num_positions = E.NUM_POSITIONS
        self.dim = E.DIM
        self.flat_dim = self.num_positions * self.dim

        # Clear TEMP, TEMP+1, RESULT at position 0
        hidden_dim = 6  # 2 per slot

        self.register_buffer('W_up', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_up', torch.zeros(hidden_dim))
        self.register_buffer('W_gate', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_gate', torch.zeros(hidden_dim))
        self.register_buffer('W_down', torch.zeros(self.flat_dim, hidden_dim))
        self.register_buffer('b_down', torch.zeros(self.flat_dim))

        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        opcode_idx = self._flat_idx(0, E.OP_START + self.opcode)

        slots_to_clear = [E.TEMP, E.TEMP + 1, E.RESULT]
        h = 0
        for slot in slots_to_clear:
            slot_idx = self._flat_idx(0, slot)

            self.W_up[h, opcode_idx] = S
            self.W_gate[h, slot_idx] = -1.0
            self.W_down[slot_idx, h] = 1.0 / S
            h += 1

            self.W_up[h, opcode_idx] = -S
            self.W_gate[h, slot_idx] = 1.0
            self.W_down[slot_idx, h] = 1.0 / S
            h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, N * D)

        up = torch.nn.functional.linear(x_flat, self.W_up, self.b_up)
        gate = torch.nn.functional.linear(x_flat, self.W_gate, self.b_gate)
        hidden = torch.nn.functional.silu(up) * gate
        delta = torch.nn.functional.linear(hidden, self.W_down, self.b_down)

        out_flat = x_flat + delta
        return out_flat.reshape(B, N, D)


def build_div32_layers(num_iters: int = 256, opcode: int = Opcode.DIV):
    """
    Build complete 32-bit division pipeline.

    Steps:
    1. Clear temporary slots (TEMP, TEMP+1, RESULT at position 0)
    2. Gather dividend (NIB_A) into TEMP[0]
    3. Gather divisor (NIB_B) into TEMP+1[0]
    4. Run iterative subtraction (num_iters times)
    5. Copy result to CARRY_OUT[0] for preservation
    6. Clear RESULT at all positions
    7. Scatter from CARRY_OUT[0] to RESULT[all positions]

    For dividend up to 2^32 and divisor >= 1, need up to 2^32 iterations.
    For practical use, limit to reasonable number.
    """
    from .pure_moe import MoE

    layers = []

    # 1. Clear slots
    layers.append(MoE([ClearDivSlotsFFN(opcode)], [opcode]))

    # 2. Gather dividend
    layers.append(MoE(
        [GatherValueFFN(E.NIB_A, E.TEMP, opcode)],
        [opcode]
    ))

    # 3. Gather divisor
    layers.append(MoE(
        [GatherValueFFN(E.NIB_B, E.TEMP + 1, opcode)],
        [opcode]
    ))

    # 4. Iterative division
    div_iter = MoE([DivIterFlatFFN(opcode)], [opcode])
    for _ in range(num_iters):
        layers.append(div_iter)

    # 5. Copy result to CARRY_OUT for preservation during clear
    if opcode == Opcode.MOD:
        # For MOD, copy TEMP (remainder)
        layers.append(MoE(
            [CopyToTempFFN(E.TEMP, E.CARRY_OUT, opcode)],
            [opcode]
        ))
    else:
        # For DIV, copy RESULT (quotient)
        layers.append(MoE(
            [CopyToTempFFN(E.RESULT, E.CARRY_OUT, opcode)],
            [opcode]
        ))

    # 6. Clear RESULT at all positions
    layers.append(MoE(
        [ClearAllResultsFFN(opcode)],
        [opcode]
    ))

    # 7. Scatter from CARRY_OUT to RESULT
    layers.append(MoE(
        [ScatterResultFFN(E.CARRY_OUT, E.RESULT, opcode)],
        [opcode]
    ))

    return layers


# =============================================================================
# TEST
# =============================================================================

def test_gather():
    """Test gather operation."""
    print("=== Testing Gather ===")

    gather = GatherValueFFN(E.NIB_A, E.TEMP, Opcode.DIV)

    # Test: 0x2A = 42 = 2*16 + 10
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    x[0, 0, E.NIB_A] = 10  # Nibble 0 = A = 10
    x[0, 1, E.NIB_A] = 2   # Nibble 1 = 2
    x[0, :, E.OP_START + Opcode.DIV] = 1.0  # Set opcode

    y = gather(x)

    gathered = y[0, 0, E.TEMP].item()
    expected = 10 + 2 * 16
    print(f"  Gathered value: {gathered:.1f} (expected {expected})")

    return abs(gathered - expected) < 1


def test_div_iter():
    """Test single division iteration."""
    print("\n=== Testing DivIter ===")

    div_iter = DivIterFlatFFN(Opcode.DIV)

    # Test: 10 / 3 -> first iter: 10-3=7, quotient=1
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    x[0, 0, E.TEMP] = 10  # Dividend
    x[0, 0, E.TEMP + 1] = 3  # Divisor
    x[0, 0, E.RESULT] = 0  # Initial quotient
    x[0, :, E.OP_START + Opcode.DIV] = 1.0

    y = div_iter(x)

    remainder = y[0, 0, E.TEMP].item()
    quotient = y[0, 0, E.RESULT].item()
    print(f"  After 1 iter: remainder={remainder:.1f}, quotient={quotient:.1f}")
    print(f"  Expected: remainder=7, quotient=1")

    # Run more iterations
    for i in range(2, 5):
        y = div_iter(y)
        remainder = y[0, 0, E.TEMP].item()
        quotient = y[0, 0, E.RESULT].item()
        print(f"  After {i} iters: remainder={remainder:.1f}, quotient={quotient:.1f}")

    return True


def test_full_div():
    """Test complete division pipeline."""
    print("\n=== Testing Full Division ===")

    layers = build_div32_layers(num_iters=64, opcode=Opcode.DIV)
    model = nn.Sequential(*layers)

    def test_div(a: int, b: int) -> int:
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

        # Set nibbles
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.NIB_A] = float((a >> (4*i)) & 0xF)
            x[0, i, E.NIB_B] = float((b >> (4*i)) & 0xF)
            x[0, i, E.OP_START + Opcode.DIV] = 1.0
            x[0, i, E.POS] = float(i)

        y = model(x)

        # Get result from position 0 (gathered value)
        result = int(round(y[0, 0, E.RESULT].item()))
        return result

    tests = [
        (6, 2, 3),
        (10, 3, 3),
        (42, 6, 7),
        (100, 10, 10),
        (15, 5, 3),
    ]

    passed = 0
    for a, b, expected in tests:
        result = test_div(a, b)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {a} / {b} = {result} (expected {expected}) [{status}]")
        if result == expected:
            passed += 1

    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    test_gather()
    test_div_iter()
    test_full_div()
