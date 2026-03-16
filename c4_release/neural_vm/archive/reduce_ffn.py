"""
Reduce/Broadcast operations via FlattenedPureFFN (autoregressive, no attention).

These operations move data across nibble positions using FlattenedPureFFN:
- Reduce: Sum values from all positions to position 0
- Broadcast: Copy value from position 0 to all positions
- Copy: Copy value from position X to position Y

All use FlattenedPureFFN with SwiGLU cancel pairs. Old PureAttention classes
kept for backward compatibility but are no longer used in the active pipeline.
"""

import torch
import torch.nn as nn
import sys
import os

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from neural_vm.embedding import E, Opcode
    from neural_vm.base_layers import PureAttention, FlattenedPureFFN, bake_weights
else:
    from .embedding import E, Opcode
    from .base_layers import PureAttention, FlattenedPureFFN, bake_weights


class ReduceSumAttention(PureAttention):
    """
    Sum values from all 8 positions into position 0 via attention.

    Position 0 attends uniformly to ALL positions.
    V reads src_slot, O writes sum to dst_slot.
    Other positions attend only to themselves (no-op via residual).
    """

    def __init__(self, src_slot: int, dst_slot: int, scale: float = 1.0,
                 broadcast: bool = False, opcode: int = None):
        self._src_slot = src_slot
        self._dst_slot = dst_slot
        self._scale = scale
        self._broadcast = broadcast
        self._opcode = opcode
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        # Mask: position 0 attends to ALL positions (uniform)
        # Other positions: attend only to self if not broadcasting
        mask = torch.full((E.NUM_POSITIONS, E.NUM_POSITIONS), -1e9)
        if self._broadcast:
            # All positions attend to all positions (uniform sum everywhere)
            mask[:, :] = 0.0
        else:
            # Only position 0 does the reduction
            mask[0, :] = 0.0  # pos 0 attends to all
            for i in range(1, E.NUM_POSITIONS):
                mask[i, i] = 0.0  # others self-attend (no-op)
        self.mask.copy_(mask)

        # V: read src_slot with scale
        # After softmax uniform over N positions: each weight = 1/N
        # So sum = N * (scale * value / N) = scale * value... wait.
        # Attention computes: sum_j(attn_j * V_j) where attn_j = 1/N
        # So output = (1/N) * sum_j(V_j)
        # We want: sum_j(value_j), so V must scale by N
        N = E.NUM_POSITIONS
        self.W_v[self._dst_slot, self._src_slot] = self._scale * N

        # O: identity for dst_slot
        self.W_o[self._dst_slot, self._dst_slot] = 1.0


# Keep backward-compatible name
ReduceSumFFN = ReduceSumAttention


class BroadcastAttention(PureAttention):
    """
    Copy value from position 0 to all positions via attention.

    All positions attend ONLY to position 0.
    V reads src_slot, O writes to dst_slot.
    """

    def __init__(self, src_slot: int, dst_slot: int):
        self._src_slot = src_slot
        self._dst_slot = dst_slot
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        # Mask: all positions attend only to position 0
        mask = torch.full((E.NUM_POSITIONS, E.NUM_POSITIONS), -1e9)
        mask[:, 0] = 0.0  # all attend to pos 0
        self.mask.copy_(mask)

        # V: read src_slot from pos 0
        self.W_v[self._dst_slot, self._src_slot] = 1.0

        # O: write to dst_slot
        self.W_o[self._dst_slot, self._dst_slot] = 1.0


BroadcastFFN = BroadcastAttention


class CopyPositionAttention(PureAttention):
    """
    Copy value from src_pos to dst_pos via attention.

    dst_pos attends ONLY to src_pos. Other positions attend to a neutral
    position where src_slot is 0 (to avoid V leaking src_slot values
    from self-attention into dst_slot at non-dst positions).
    """

    def __init__(self, src_pos: int, dst_pos: int, src_slot: int, dst_slot: int):
        self._src_pos = src_pos
        self._dst_pos = dst_pos
        self._src_slot = src_slot
        self._dst_slot = dst_slot
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        # Find a neutral position (not src, not dst) for non-participating positions.
        # Neutral pos has src_slot = 0 after prior clearing steps,
        # so V produces 0 and residual is unchanged.
        neutral = None
        for p in range(E.NUM_POSITIONS):
            if p != self._src_pos and p != self._dst_pos:
                neutral = p
                break
        assert neutral is not None, "Need at least 3 positions"

        # Mask: dst_pos → src_pos, all others → neutral (where src_slot = 0)
        mask = torch.full((E.NUM_POSITIONS, E.NUM_POSITIONS), -1e9)
        mask[self._dst_pos, self._src_pos] = 0.0
        for i in range(E.NUM_POSITIONS):
            if i != self._dst_pos:
                mask[i, neutral] = 0.0  # attend to neutral (V reads 0)
        self.mask.copy_(mask)

        # V: read src_slot from attended position
        self.W_v[self._dst_slot, self._src_slot] = 1.0

        # O: write to dst_slot
        self.W_o[self._dst_slot, self._dst_slot] = 1.0


CopyPositionFFN = CopyPositionAttention


# =============================================================================
# FLATTENED FFN REPLACEMENTS (no attention, fully autoregressive)
# =============================================================================

class ReduceSumFlatFFN(FlattenedPureFFN):
    """
    Sum values from all 8 positions into position 0 via FlattenedPureFFN.

    Replaces ReduceSumAttention. Matches exact attention behavior:
    - Position 0: dst[0] += scale * Σ src[j] (2 units, cancel pair)
    - Positions 1-7: dst[i] += 8 * scale * src[i] (14 units, 7 cancel pairs
      reproducing attention self-attend side effect)

    hidden_dim=16: 1 cancel pair for reduce + 7 cancel pairs for self-attend.
    """

    def __init__(self, src_slot: int, dst_slot: int, scale: float = 1.0):
        self._src_slot = src_slot
        self._dst_slot = dst_slot
        self._scale = scale
        super().__init__(hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            # Cancel pair 0: Position 0 reduce — dst[0] += scale * Σ_j src[j]
            self.b_up.data[0] = S
            for j in range(E.NUM_POSITIONS):
                self.W_gate.data[0, fi(j, self._src_slot)] = self._scale
            self.W_down.data[fi(0, self._dst_slot), 0] = 1.0 / S

            self.b_up.data[1] = -S
            for j in range(E.NUM_POSITIONS):
                self.W_gate.data[1, fi(j, self._src_slot)] = -self._scale
            self.W_down.data[fi(0, self._dst_slot), 1] = 1.0 / S

            # Cancel pairs 1-7: Positions 1-7 self-attend side effect
            # dst[i] += 8 * scale * src[i]
            for i in range(1, E.NUM_POSITIONS):
                u = i * 2  # unit index
                self.b_up.data[u] = S
                self.W_gate.data[u, fi(i, self._src_slot)] = 8.0 * self._scale
                self.W_down.data[fi(i, self._dst_slot), u] = 1.0 / S

                self.b_up.data[u + 1] = -S
                self.W_gate.data[u + 1, fi(i, self._src_slot)] = -8.0 * self._scale
                self.W_down.data[fi(i, self._dst_slot), u + 1] = 1.0 / S


class BroadcastFlatFFN(FlattenedPureFFN):
    """
    Copy value from position 0 to all positions via FlattenedPureFFN.

    Replaces BroadcastAttention. Cancel pair reads fi(0, src_slot),
    writes to fi(j, dst_slot) for all j=0..7.

    hidden_dim=2.
    """

    def __init__(self, src_slot: int, dst_slot: int):
        self._src_slot = src_slot
        self._dst_slot = dst_slot
        super().__init__(hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            # Cancel pair: broadcast src[0] to all positions' dst
            self.b_up.data[0] = S
            self.W_gate.data[0, fi(0, self._src_slot)] = 1.0
            for j in range(E.NUM_POSITIONS):
                self.W_down.data[fi(j, self._dst_slot), 0] = 1.0 / S

            self.b_up.data[1] = -S
            self.W_gate.data[1, fi(0, self._src_slot)] = -1.0
            for j in range(E.NUM_POSITIONS):
                self.W_down.data[fi(j, self._dst_slot), 1] = 1.0 / S


class CopyPositionFlatFFN(FlattenedPureFFN):
    """
    Copy value from src_pos to dst_pos via FlattenedPureFFN.

    Replaces CopyPositionAttention. Cancel pair reads fi(src_pos, src_slot),
    writes to fi(dst_pos, dst_slot). No side effects on other positions.

    hidden_dim=2.
    """

    def __init__(self, src_pos: int, dst_pos: int, src_slot: int, dst_slot: int):
        self._src_pos = src_pos
        self._dst_pos = dst_pos
        self._src_slot = src_slot
        self._dst_slot = dst_slot
        super().__init__(hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        fi = self._flat_idx
        with torch.no_grad():
            # Cancel pair: copy src[src_pos, src_slot] → dst[dst_pos, dst_slot]
            self.b_up.data[0] = S
            self.W_gate.data[0, fi(self._src_pos, self._src_slot)] = 1.0
            self.W_down.data[fi(self._dst_pos, self._dst_slot), 0] = 1.0 / S

            self.b_up.data[1] = -S
            self.W_gate.data[1, fi(self._src_pos, self._src_slot)] = -1.0
            self.W_down.data[fi(self._dst_pos, self._dst_slot), 1] = 1.0 / S


# =============================================================================
# SPECIFIC REDUCE/BROADCAST CLASSES (drop-in replacements)
# =============================================================================

class CompareReduceEqFFNNew(ReduceSumFlatFFN):
    """Sum per-nibble EQ results via FlatFFN."""
    def __init__(self):
        super().__init__(src_slot=E.TEMP, dst_slot=E.RAW_SUM, scale=1.0)


class CompareReduceNeFFNNew(ReduceSumFlatFFN):
    """Sum per-nibble NE results via FlatFFN."""
    def __init__(self):
        super().__init__(src_slot=E.TEMP, dst_slot=E.RAW_SUM, scale=1.0)


class CmpBroadcastResultFFN(CopyPositionFlatFFN):
    """Copy TEMP from position 7 to RESULT at position 0 via FlatFFN."""
    def __init__(self):
        super().__init__(src_pos=7, dst_pos=0, src_slot=E.TEMP, dst_slot=E.RESULT)


class BranchConditionFFN(BroadcastFlatFFN):
    """Broadcast branch condition from position 0 to all via FlatFFN."""
    def __init__(self):
        super().__init__(src_slot=E.TEMP, dst_slot=E.RAW_SUM)


class BzReduceFFN(ReduceSumFlatFFN):
    """Sum per-nibble zero flags for BZ via FlatFFN."""
    def __init__(self):
        super().__init__(src_slot=E.TEMP, dst_slot=E.RAW_SUM, scale=1.0)


class BnzReduceFFN(ReduceSumFlatFFN):
    """Sum per-nibble non-zero flags for BNZ via FlatFFN."""
    def __init__(self):
        super().__init__(src_slot=E.TEMP, dst_slot=E.RAW_SUM, scale=1.0)


class McmpReduceFFN(ReduceSumFlatFFN):
    """Sum per-nibble memcmp results via FlatFFN."""
    def __init__(self):
        super().__init__(src_slot=E.TEMP, dst_slot=E.RAW_SUM, scale=1.0)


# =============================================================================
# TEST
# =============================================================================

def test_reduce_ffn():
    """Test the reduce/broadcast attention layers."""
    print("=" * 60)
    print("REDUCE/BROADCAST ATTENTION TESTS")
    print("=" * 60)

    # Test ReduceSumAttention
    print("\n1. ReduceSumAttention (sum TEMP from all positions to RAW_SUM[0])")
    reduce = ReduceSumAttention(src_slot=E.TEMP, dst_slot=E.RAW_SUM, scale=1.0)

    x = torch.zeros(1, 8, E.DIM)
    # Set TEMP = position number at each position
    for i in range(8):
        x[0, i, E.TEMP] = float(i + 1)  # 1, 2, 3, 4, 5, 6, 7, 8

    y = reduce(x)
    expected_sum = sum(range(1, 9))  # 36
    actual_sum = y[0, 0, E.RAW_SUM].item()
    print(f"  Input TEMP: {[x[0, i, E.TEMP].item() for i in range(8)]}")
    print(f"  Output RAW_SUM[0]: {actual_sum:.1f} (expected {expected_sum})")
    print(f"  PASS: {abs(actual_sum - expected_sum) < 1.0}")

    # Test BroadcastAttention
    print("\n2. BroadcastAttention (copy TEMP[0] to RAW_SUM at all positions)")
    broadcast = BroadcastAttention(src_slot=E.TEMP, dst_slot=E.RAW_SUM)

    x = torch.zeros(1, 8, E.DIM)
    x[0, 0, E.TEMP] = 42.0

    y = broadcast(x)
    results = [y[0, i, E.RAW_SUM].item() for i in range(8)]
    print(f"  Input TEMP[0]: 42")
    print(f"  Output RAW_SUM: {[f'{r:.1f}' for r in results]}")
    all_42 = all(abs(r - 42.0) < 1.0 for r in results)
    print(f"  PASS: {all_42}")

    # Test CopyPositionAttention (TEMP[7] -> RESULT[0])
    print("\n3. CopyPositionAttention (copy TEMP[7] to RESULT[0])")
    copy = CopyPositionAttention(src_pos=7, dst_pos=0, src_slot=E.TEMP, dst_slot=E.RESULT)

    x = torch.zeros(1, 8, E.DIM)
    x[0, 7, E.TEMP] = 99.0

    y = copy(x)
    result = y[0, 0, E.RESULT].item()
    print(f"  Input TEMP[7]: 99")
    print(f"  Output RESULT[0]: {result:.1f}")
    print(f"  PASS: {abs(result - 99.0) < 1.0}")

    print("\n" + "=" * 60)
    print("All reduce/broadcast attention tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_reduce_ffn()
