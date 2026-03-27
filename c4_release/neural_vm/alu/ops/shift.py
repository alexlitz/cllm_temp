"""
Chunk-generic SHL/SHR pipeline: 2 layers each.

Layer 1 (per-position): Precompute sub-chunk shifts for r=0..k-1.
Layer 2 (flattened): Select and route based on shift amount indicator.

For SHL by k = chunk_bits*q + r:
  result[j] = s_r(a[j-q]) + c_r(a[j-q-1])
  s_r(a) = (a * 2^r) mod base
  c_r(a) = floor(a * 2^r / base)

For SHR by k = chunk_bits*q + r:
  result[j] = s_r(a[j+q]) + c_r(a[j+q+1])
  s_r(a) = floor(a / 2^r)
  c_r(a) = (a mod 2^r) * 2^(chunk_bits-r)

Shift amount is encoded in NIB_B chunks (scalar value 0..31).

Supported configs: BIT, PAIR, NIBBLE, BYTE.
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import (
    GenericE, GenericPureFFN, GenericFlattenedFFN,
    bake_clear_pair,
)


def _shift_slots(ge, k):
    """Return (s_slots, c_slots) for sub-chunk shifts r=0..k-1.

    s_slots[r]: slot storing shifted value for sub-shift r
    c_slots[r]: slot storing carry for sub-shift r (None for r=0)
    """
    # r=0: NIB_A, no carry
    s_slots = [ge.NIB_A]
    c_slots = [None]

    # Fixed slots for r=1..3
    fixed_s = [ge.RAW_SUM, ge.CARRY_IN, ge.CARRY_OUT]
    fixed_c = [ge.TEMP, 80, 81]  # slot 80, 81 for extra carries

    for r in range(1, k):
        if r - 1 < len(fixed_s):
            s_slots.append(fixed_s[r - 1])
            c_slots.append(fixed_c[r - 1])
        else:
            # For r >= 4: use slots 82+
            idx = r - 1 - len(fixed_s)
            s_slots.append(82 + idx * 2)
            c_slots.append(82 + idx * 2 + 1)

    return s_slots, c_slots


class ShlPrecomputeFFN(nn.Module):
    """Layer 1 of SHL: Precompute sub-chunk left-shifts.

    For r=1..k-1:
      s_r = (2^r * a) mod base → s_slot[r]
      c_r = floor(2^r * a / base) → c_slot[r]

    Uses merged step pairs: each step writes to BOTH s and c slots.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        k = ge.config.chunk_bits
        S = ge.SCALE
        base = ge.BASE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        s_slots, c_slots = _shift_slots(ge, k)

        # Hidden units: for each r=1..k-1:
        #   2 (cancel pair for 2^r * a) + 2*(2^r - 1) (merged step pairs)
        hidden_dim = 0
        for r in range(1, k):
            hidden_dim += 2 + 2 * ((1 << r) - 1)

        self.ffn = GenericPureFFN(dim, hidden_dim=hidden_dim, dtype=dtype)

        with torch.no_grad():
            h = 0
            for r in range(1, k):
                multiplier = float(1 << r)  # 2^r
                s_slot = s_slots[r]
                c_slot = c_slots[r]
                step_size = base // (1 << r)  # base / 2^r = 2^(k-r)
                max_carry = (1 << r) - 1  # 2^r - 1

                # Cancel pair: 2^r * a → s_slot
                self.ffn.W_up[h, ge.NIB_A] = S
                self.ffn.W_gate[h, ge.OP_START + opcode] = multiplier
                self.ffn.W_down[s_slot, h] = 1.0 / S
                h += 1

                self.ffn.W_up[h, ge.NIB_A] = -S
                self.ffn.W_gate[h, ge.OP_START + opcode] = -multiplier
                self.ffn.W_down[s_slot, h] = 1.0 / S
                h += 1

                # Merged step pairs: step(a >= m*step_size) → -base to s_slot, +1 to c_slot
                for m in range(1, max_carry + 1):
                    threshold = m * step_size

                    # Rise
                    self.ffn.W_up[h, ge.NIB_A] = S
                    self.ffn.b_up[h] = -S * (threshold - 1.0)
                    self.ffn.W_gate[h, ge.OP_START + opcode] = 1.0
                    self.ffn.W_down[s_slot, h] = -float(base) / S
                    self.ffn.W_down[c_slot, h] = 1.0 / S
                    h += 1

                    # Saturation
                    self.ffn.W_up[h, ge.NIB_A] = S
                    self.ffn.b_up[h] = -S * float(threshold)
                    self.ffn.W_gate[h, ge.OP_START + opcode] = 1.0
                    self.ffn.W_down[s_slot, h] = float(base) / S
                    self.ffn.W_down[c_slot, h] = -1.0 / S
                    h += 1

            assert h == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class ShrPrecomputeFFN(nn.Module):
    """Layer 1 of SHR: Precompute sub-chunk right-shifts.

    For r=1..k-1:
      s_r = floor(a / 2^r) → s_slot[r]  (step pairs at multiples of 2^r)
      c_r = (a mod 2^r) * 2^(k-r) → c_slot[r]  (= 2^(k-r)*a - base*s_r)
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        k = ge.config.chunk_bits
        S = ge.SCALE
        base = ge.BASE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        s_slots, c_slots = _shift_slots(ge, k)

        # Hidden units: for each r=1..k-1:
        #   2 (cancel pair for 2^(k-r)*a → c_slot) + 2*(2^(k-r) - 1) (merged step pairs)
        hidden_dim = 0
        for r in range(1, k):
            num_steps = (1 << (k - r)) - 1  # floor values 0..num_steps
            hidden_dim += 2 + 2 * num_steps

        self.ffn = GenericPureFFN(dim, hidden_dim=hidden_dim, dtype=dtype)

        with torch.no_grad():
            h = 0
            for r in range(1, k):
                s_slot = s_slots[r]
                c_slot = c_slots[r]
                step_size = 1 << r  # 2^r
                c_multiplier = float(1 << (k - r))  # 2^(k-r)
                max_floor = (1 << (k - r)) - 1

                # Cancel pair: 2^(k-r) * a → c_slot
                self.ffn.W_up[h, ge.NIB_A] = S
                self.ffn.W_gate[h, ge.OP_START + opcode] = c_multiplier
                self.ffn.W_down[c_slot, h] = 1.0 / S
                h += 1

                self.ffn.W_up[h, ge.NIB_A] = -S
                self.ffn.W_gate[h, ge.OP_START + opcode] = -c_multiplier
                self.ffn.W_down[c_slot, h] = 1.0 / S
                h += 1

                # Merged step pairs: step(a >= m*2^r) → +1 to s_slot, -base to c_slot
                for m in range(1, max_floor + 1):
                    threshold = m * step_size

                    # Rise
                    self.ffn.W_up[h, ge.NIB_A] = S
                    self.ffn.b_up[h] = -S * (threshold - 1.0)
                    self.ffn.W_gate[h, ge.OP_START + opcode] = 1.0
                    self.ffn.W_down[s_slot, h] = 1.0 / S
                    self.ffn.W_down[c_slot, h] = -float(base) / S
                    h += 1

                    # Saturation
                    self.ffn.W_up[h, ge.NIB_A] = S
                    self.ffn.b_up[h] = -S * float(threshold)
                    self.ffn.W_gate[h, ge.OP_START + opcode] = 1.0
                    self.ffn.W_down[s_slot, h] = -1.0 / S
                    self.ffn.W_down[c_slot, h] = float(base) / S
                    h += 1

            assert h == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class ShiftSelectFFN(nn.Module):
    """Layer 2: Select and combine based on shift amount.

    For each shift amount k=0..31 and output position j:
      q = k // chunk_bits, r = k % chunk_bits
      SHL: source = j-q, carry_source = j-q-1
      SHR: source = j+q, carry_source = j+q+1

    Uses 3-unit integer indicator: step(val == k).
    Also clears precomputed slots.
    """

    def __init__(self, ge: GenericE, opcode: int, is_shr: bool = False):
        super().__init__()
        N = ge.NUM_POSITIONS
        k = ge.config.chunk_bits
        S = ge.SCALE
        base = ge.BASE
        dtype = ge.config.torch_dtype

        s_slots, c_slots = _shift_slots(ge, k)

        # Count valid (shift_amount, output_pos) pairs
        num_valid = 0
        for shift_amt in range(32):
            q = shift_amt // k
            for j in range(N):
                if is_shr:
                    src = j + q
                else:
                    src = j - q
                if 0 <= src < N:
                    num_valid += 1

        # 3 units per valid pair + clearing
        clear_slots_per_pos = 2 * (k - 1)  # s_slots[1..k-1] and c_slots[1..k-1]
        hidden_dim = num_valid * 3 + clear_slots_per_pos * N * 2

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        # Determine which NIB_B positions encode the shift amount
        # Need enough positions to cover 0..31
        shift_positions = []
        coverage = 1
        for i in range(N):
            shift_positions.append(i)
            coverage *= base
            if coverage > 31:
                break

        # Positions beyond shift_positions should be 0 for valid shift
        # We'll suppress output if any higher nibble is non-zero
        high_nibble_positions = list(range(len(shift_positions), N))

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down

            h = 0

            for shift_amt in range(32):
                q = shift_amt // k
                r = shift_amt % k

                for j in range(N):
                    if is_shr:
                        src = j + q
                        src_carry = j + q + 1
                    else:
                        src = j - q
                        src_carry = j - q - 1

                    if src < 0 or src >= N:
                        continue

                    result_idx = fi(j, ge.RESULT)

                    # Build gate: s_r[src] + c_r[src_carry]
                    def set_gate(unit):
                        W_gate.data[unit, fi(src, s_slots[r])] = 1.0
                        if c_slots[r] is not None and 0 <= src_carry < N:
                            W_gate.data[unit, fi(src_carry, c_slots[r])] = 1.0

                    # 3-unit indicator: step(val == shift_amt)
                    # W_up reads shift amount = sum(NIB_B[i] * base^i)
                    # Also subtract large penalty for any high nibble to suppress output
                    def set_up(unit, offset):
                        for i in shift_positions:
                            W_up.data[unit, fi(i, ge.NIB_B)] = S * float(base ** i)
                        b_up.data[unit] = -S * float(shift_amt + offset)
                        # Suppress when any high nibble is non-zero (shift >= 32)
                        # silu() will clamp to ~0 when input is very negative
                        for hi_pos in high_nibble_positions:
                            W_up.data[unit, fi(hi_pos, ge.NIB_B)] = -S * 100.0

                    # Unit A: silu(S*(val-k+1)), W_down = +1/S
                    set_up(h, -1)
                    set_gate(h)
                    W_down.data[result_idx, h] = 1.0 / S
                    h += 1

                    # Unit B: silu(S*(val-k)), W_down = -2/S
                    set_up(h, 0)
                    set_gate(h)
                    W_down.data[result_idx, h] = -2.0 / S
                    h += 1

                    # Unit C: silu(S*(val-k-1)), W_down = +1/S
                    set_up(h, 1)
                    set_gate(h)
                    W_down.data[result_idx, h] = 1.0 / S
                    h += 1

            # Clear precomputed slots at all positions
            for pos in range(N):
                for r in range(1, k):
                    bake_clear_pair(self.flat_ffn.ffn, h,
                                    fi(pos, ge.OP_START + opcode),
                                    fi(pos, s_slots[r]), S)
                    h += 2
                    bake_clear_pair(self.flat_ffn.ffn, h,
                                    fi(pos, ge.OP_START + opcode),
                                    fi(pos, c_slots[r]), S)
                    h += 2

            assert h <= hidden_dim, f"Used {h} hidden units, allocated {hidden_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


def build_shl_layers(config: ChunkConfig, opcode: int = 23) -> nn.ModuleList:
    ge = GenericE(config)
    layers = []
    if config.chunk_bits > 1:
        layers.append(ShlPrecomputeFFN(ge, opcode))
    layers.append(ShiftSelectFFN(ge, opcode, is_shr=False))
    return nn.ModuleList(layers)


def build_shr_layers(config: ChunkConfig, opcode: int = 24) -> nn.ModuleList:
    ge = GenericE(config)
    layers = []
    if config.chunk_bits > 1:
        layers.append(ShrPrecomputeFFN(ge, opcode))
    layers.append(ShiftSelectFFN(ge, opcode, is_shr=True))
    return nn.ModuleList(layers)
