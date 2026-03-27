"""
Shared chunk-generic ALU building blocks.

GenericE: Embedding layout parameterized by ChunkConfig.
Shared FFN builders for carry-lookahead, slot clearing, scalar gather.
MAGIC floor utilities for fp32 (chunk-level operations only, max value 2^24).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chunk_config import ChunkConfig


# =============================================================================
# MAGIC Floor Constants and Utilities
# =============================================================================

# fp32: at 1.5*2^23 scale, ULP = 1 (2^23 is at boundary where ULP=0.5 still applies)
# Valid for values in range [0, 2^24) = [0, 16777216)
# For chunk-based operations (bytes 0-255, carry sums up to ~260100), this is sufficient.
MAGIC32 = 1.5 * float(2**23)  # 12582912.0


def magic_floor32(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """Compute floor(x) using fp32 MAGIC trick.

    At scale 2^23, fp32's ULP = 1, so only integers are representable.
    Adding MAGIC forces rounding to nearest integer. Subtracting (0.5 - eps)
    converts round-to-nearest into floor.

    Args:
        x: Input tensor (should be fp32 or will be converted)
        eps: Small offset to avoid round-to-even boundary issues (default 0.001)

    Returns:
        floor(x) as tensor (same dtype as input)
    """
    orig_dtype = x.dtype
    x32 = x.float() if x.dtype != torch.float32 else x
    x_shifted = x32 - 0.5 + eps
    result = (x_shifted + MAGIC32) - MAGIC32
    return result.to(orig_dtype)


class MagicFloorFFN(nn.Module):
    """Pure FFN that computes floor(x) using MAGIC trick baked into weights.

    The MAGIC floor trick is implemented as an FFN:
        Layer 1: x_shifted = x - 0.5 + eps (baked into W_up=1, b_up=-0.499)
        Layer 2: temp = x_shifted + MAGIC (baked into W_gate=1, b_gate=MAGIC)
        Layer 3: result = temp - MAGIC (baked into W_down=1, b_down=-MAGIC)

    The fp32 rounding happens automatically during the additions.
    This uses identity activations (linear FFN) since we need exact arithmetic.

    Weights: 6 parameters (W_up, b_up, W_gate, b_gate, W_down, b_down)
    """

    def __init__(self, eps: float = 0.001):
        super().__init__()
        MAGIC = 1.5 * float(2**23)  # 12582912.0 - safely in ULP=1 range

        # Layer 1: shift by -0.5 + eps
        self.W_up = nn.Parameter(torch.ones(1, 1))
        self.b_up = nn.Parameter(torch.tensor([-0.5 + eps]))

        # Layer 2: add MAGIC (via gate bias)
        self.W_gate = nn.Parameter(torch.ones(1, 1))
        self.b_gate = nn.Parameter(torch.tensor([MAGIC]))

        # Layer 3: subtract MAGIC
        self.W_down = nn.Parameter(torch.ones(1, 1))
        self.b_down = nn.Parameter(torch.tensor([-MAGIC]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute floor(x) via FFN with MAGIC constants baked into weights."""
        orig_shape = x.shape
        x_flat = x.view(-1, 1).float()  # Must be fp32 for MAGIC trick

        # Apply MAGIC floor: (x - 0.5 + eps + MAGIC) - MAGIC
        shifted = F.linear(x_flat, self.W_up, self.b_up)  # x - 0.5 + eps
        with_magic = shifted + self.b_gate  # + MAGIC (fp32 rounding happens here)
        result = with_magic + self.b_down  # - MAGIC

        return result.view(orig_shape).to(x.dtype)


class StaircaseFloorFFN(nn.Module):
    """Pure FFN that computes floor(x) using staircase of sigmoid steps.

    For floor in range [0, max_val]:
        - M = max_val hidden units
        - Each unit k detects if x >= k via sigmoid
        - Sum of all units = floor(x)

    This is baked into FFN weights - no runtime operations needed.
    Uses sigmoid (not SwiGLU) for sharp step transitions.

    Weights: 3*M + 1 (W_up, b_up, W_down, b_down)
    """

    def __init__(self, max_val: int = 256, scale: float = 10000.0, eps: float = 0.002):
        super().__init__()
        self.max_val = max_val
        self.scale = scale
        M = max_val

        # W_up: all weights = scale (detects x - threshold)
        self.W_up = nn.Parameter(torch.full((M, 1), scale))

        # b_up: encodes thresholds 1, 2, ..., M
        thresholds = torch.arange(1, M + 1, dtype=torch.float32)
        self.b_up = nn.Parameter(scale * (-thresholds + eps))

        # W_down: sum all hidden units
        self.W_down = nn.Parameter(torch.ones(1, M))
        self.b_down = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute floor(x) via pure FFN."""
        orig_shape = x.shape
        x_flat = x.view(-1, 1)

        # hidden = sigmoid(W_up @ x + b_up) - each unit is a step at threshold k
        pre_act = F.linear(x_flat, self.W_up, self.b_up)
        hidden = torch.sigmoid(pre_act)

        # output = sum of all hidden units = floor(x)
        output = F.linear(hidden, self.W_down, self.b_down)
        return output.view(orig_shape)


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
