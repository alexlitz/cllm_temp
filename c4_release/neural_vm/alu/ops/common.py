"""
Shared chunk-generic ALU building blocks.

GenericE: Embedding layout parameterized by ChunkConfig.
Shared FFN builders for carry-lookahead, slot clearing, scalar gather.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chunk_config import ChunkConfig


class GenericE:
    """Embedding layout parameterized by ChunkConfig.

    Same slot semantics as E (NIB_A=0, NIB_B=1, ..., TEMP=6) but with
    configurable NUM_POSITIONS and DIM.
    """

    # Per-position feature slots (unchanged)
    NIB_A = 0
    NIB_B = 1
    RAW_SUM = 2
    CARRY_IN = 3
    CARRY_OUT = 4
    RESULT = 5
    TEMP = 6

    # Opcode encoding
    OP_START = 7
    NUM_OPS = 72

    # Position
    POS = 79

    # Division temp slots
    SLOT_DIVIDEND = 6   # = TEMP
    SLOT_DIVISOR = 7    # = OP_START (reused — division clears/restores)
    SLOT_REMAINDER = 8
    SLOT_QUOTIENT = 9

    def __init__(self, config: ChunkConfig):
        self.config = config
        self.NUM_POSITIONS = config.num_positions
        self.DIM = 160  # Keep same DIM as original E
        self.SCALE = config.scale
        self.DIV_SCALE = config.div_scale
        self.BASE = config.base
        self.CHUNK_MAX = config.chunk_max


class GenericPureFFN(nn.Module):
    """SwiGLU FFN with configurable dtype. Same forward as PureFFN."""

    def __init__(self, dim: int, hidden_dim: int, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.W_up = nn.Parameter(torch.zeros(hidden_dim, dim, dtype=dtype))
        self.b_up = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))
        self.W_gate = nn.Parameter(torch.zeros(hidden_dim, dim, dtype=dtype))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))
        self.W_down = nn.Parameter(torch.zeros(dim, hidden_dim, dtype=dtype))
        self.b_down = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.linear(x, self.W_up, self.b_up)
        gate = F.linear(x, self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        return x + F.linear(hidden, self.W_down, self.b_down)


class GenericFlattenedFFN(nn.Module):
    """Flattened FFN parameterized by GenericE. Flattens [B, N, D] → [B, 1, N*D]."""

    def __init__(self, ge: GenericE, hidden_dim: int, dtype=torch.float32):
        super().__init__()
        self.ge = ge
        self.dim = ge.DIM
        self.num_positions = ge.NUM_POSITIONS
        self.flat_dim = ge.NUM_POSITIONS * ge.DIM
        self.hidden_dim = hidden_dim
        self.ffn = GenericPureFFN(dim=self.flat_dim, hidden_dim=hidden_dim, dtype=dtype)

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    # Convenience properties
    @property
    def W_up(self): return self.ffn.W_up
    @property
    def b_up(self): return self.ffn.b_up
    @property
    def W_gate(self): return self.ffn.W_gate
    @property
    def b_gate(self): return self.ffn.b_gate
    @property
    def W_down(self): return self.ffn.W_down
    @property
    def b_down(self): return self.ffn.b_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, 1, N * D)
        y_flat = self.ffn(x_flat)
        return y_flat.reshape(B, N, D)


def bake_cancel_pair(ffn, h, up_slot, gate_slot, out_slot, S):
    """Set weights for a cancel pair at hidden units h, h+1.

    Cancel pair: silu(+S*up)*gate/S + silu(-S*up)*(-gate)/S = up*gate/S² × 2
    Net effect: cleanly copies gate_value * sign(up_value) to out_slot.
    """
    with torch.no_grad():
        ffn.W_up.data[h, up_slot] = S
        ffn.W_gate.data[h, gate_slot] = 1.0
        ffn.W_down.data[out_slot, h] = 1.0 / S

        ffn.W_up.data[h + 1, up_slot] = -S
        ffn.W_gate.data[h + 1, gate_slot] = -1.0
        ffn.W_down.data[out_slot, h + 1] = 1.0 / S


def bake_step_pair(ffn, h, up_slots_and_weights, gate_slot, out_slot, threshold, S, out_weight=1.0):
    """Set weights for a step pair at hidden units h, h+1.

    step(sum >= threshold): rise at threshold-1, saturate at threshold.
    up_slots_and_weights: list of (slot, weight) for W_up.
    """
    with torch.no_grad():
        for slot, w in up_slots_and_weights:
            ffn.W_up.data[h, slot] = S * w
            ffn.W_up.data[h + 1, slot] = S * w
        ffn.b_up.data[h] = -S * (threshold - 1.0)
        ffn.b_up.data[h + 1] = -S * threshold
        ffn.W_gate.data[h, gate_slot] = 1.0
        ffn.W_gate.data[h + 1, gate_slot] = 1.0
        ffn.W_down.data[out_slot, h] = out_weight / S
        ffn.W_down.data[out_slot, h + 1] = -out_weight / S


def bake_clear_pair(ffn, h, op_slot, slot_to_clear, S):
    """Set weights for a cancel pair that clears a slot (sets it to 0)."""
    with torch.no_grad():
        ffn.W_up.data[h, op_slot] = S
        ffn.W_gate.data[h, slot_to_clear] = -1.0
        ffn.W_down.data[slot_to_clear, h] = 1.0 / S

        ffn.W_up.data[h + 1, op_slot] = -S
        ffn.W_gate.data[h + 1, slot_to_clear] = 1.0
        ffn.W_down.data[slot_to_clear, h + 1] = 1.0 / S
