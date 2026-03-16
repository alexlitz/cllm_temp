"""
Attention-Based Cross-Nibble Operations.

These are alternative implementations that use attention for cross-nibble
data movement instead of the default flattened FFN approach.

COMPARISON:
| Operation | Attention (this file) | FFN (default) |
|-----------|----------------------|---------------|
| EQ reduce | CompareReduceEqAttention | CompareReduceEqFFNNew |
| NE reduce | CompareReduceNeAttention | CompareReduceNeFFNNew |
| Cmp broadcast | CmpBroadcastResultAttention | CmpBroadcastResultFFN |
| Branch broadcast | BranchConditionAttention | BranchConditionFFN |
| BZ reduce | BzReduceAttention | BzReduceFFN |
| BNZ reduce | BnzReduceAttention | BnzReduceFFN |
| MCMP reduce | McmpReduceAttention | McmpReduceFFN |

The FFN approach is preferred because:
1. No Q/K computation needed (just linear projection)
2. More explicit routing via weight matrices
3. Consistent with carry propagation approach

The attention approach is valid for:
1. Architectures that have attention but limited FFN capacity
2. Research comparing attention vs FFN for routing
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.embedding import E, Opcode
from neural_vm.base_layers import PureAttention, bake_weights


class CompareReduceEqAttention(PureAttention):
    """
    Sum per-nibble EQ results using attention.

    Position 0 attends to all positions, gathering TEMP values.
    Other positions attend only to themselves (identity).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0  # Position 0 sees all
        for i in range(1, N):
            mask[i, i] = 0.0  # Others see only self
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # V projects TEMP to RAW_SUM with scale 8.0
            self.W_v[E.RAW_SUM, E.TEMP] = 8.0
            # O is identity for RAW_SUM
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class CompareReduceNeAttention(PureAttention):
    """
    Sum per-nibble NE results using attention.

    Same pattern as EQ - position 0 gathers from all positions.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0
        for i in range(1, N):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_v[E.RAW_SUM, E.TEMP] = 8.0
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class CmpBroadcastResultAttention(PureAttention):
    """
    Copy TEMP from position 7 to RESULT at position 0.

    Position 0 attends only to position 7.
    Other positions attend to position 0 (to get the result).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        mask[0, N-1] = 0.0  # Position 0 reads from position 7
        for i in range(1, N):
            mask[i, 0] = 0.0  # Others read from position 0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0
            self.W_v[E.TEMP, E.TEMP] = 1.0
            self.W_o[E.RESULT, E.TEMP] = 1.0


class BranchConditionAttention(PureAttention):
    """
    Broadcast branch condition from position 0 to all positions.

    All positions attend to position 0, copying TEMP to RAW_SUM.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N):
            mask[i, 0] = 0.0  # All positions read from position 0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_v[E.RAW_SUM, E.TEMP] = 1.0
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class BzReduceAttention(PureAttention):
    """
    Reduce per-nibble zero flags for BZ (branch if zero).

    All nibbles must be zero for the branch to be taken.
    Position 0 gathers from all positions.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0
        for i in range(1, N):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Uniform attention across all positions
        self.W_q[:, :] = 0.0
        self.W_k[:, :] = 0.0
        for i in range(E.DIM):
            self.W_q[i, E.OP_START + Opcode.BZ] = 1.0
            self.W_k[i, E.OP_START + Opcode.BZ] = 1.0

        # V projects TEMP to RAW_SUM
        self.W_v[E.RAW_SUM, E.TEMP] = 1.0
        self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class BnzReduceAttention(PureAttention):
    """
    Reduce per-nibble non-zero flags for BNZ (branch if not zero).

    Any non-zero nibble triggers the branch.
    Position 0 gathers from all positions.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0
        for i in range(1, N):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        self.W_q[:, :] = 0.0
        self.W_k[:, :] = 0.0
        for i in range(E.DIM):
            self.W_q[i, E.OP_START + Opcode.BNZ] = 1.0
            self.W_k[i, E.OP_START + Opcode.BNZ] = 1.0

        self.W_v[E.RAW_SUM, E.TEMP] = 1.0
        self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class McmpReduceAttention(PureAttention):
    """
    Reduce per-nibble memcmp results.

    Position 0 gathers difference flags from all positions.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0
        for i in range(1, N):
            mask[i, i] = 0.0
        self.register_buffer('mask', mask)

    @bake_weights
    def _bake_weights(self):
        self.W_v[E.RAW_SUM, E.TEMP] = 1.0
        self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


# =============================================================================
# TEST
# =============================================================================

def test_attention_ops():
    """Test the attention-based operations."""
    print("=" * 60)
    print("ATTENTION-BASED CROSS-NIBBLE OPS TEST")
    print("=" * 60)

    # Test CompareReduceEqAttention
    print("\n1. CompareReduceEqAttention (sum TEMP from all positions)")
    reduce_eq = CompareReduceEqAttention()

    x = torch.zeros(1, 8, E.DIM)
    for i in range(8):
        x[0, i, E.TEMP] = 1.0  # All nibbles match

    y = reduce_eq(x)
    expected = 8 * 8.0  # 64 (8 positions × scale 8.0)
    actual = y[0, 0, E.RAW_SUM].item()
    print(f"  Input TEMP: all 1.0")
    print(f"  Output RAW_SUM[0]: {actual} (expected {expected})")
    print(f"  PASS: {abs(actual - expected) < 1.0}")

    # Test BranchConditionAttention
    print("\n2. BranchConditionAttention (broadcast TEMP[0] to RAW_SUM all)")
    broadcast = BranchConditionAttention()

    x = torch.zeros(1, 8, E.DIM)
    x[0, 0, E.TEMP] = 42.0

    y = broadcast(x)
    all_42 = all(abs(y[0, i, E.RAW_SUM].item() - 42.0) < 0.1 for i in range(8))
    print(f"  Input TEMP[0]: 42")
    print(f"  Output RAW_SUM: {[round(y[0, i, E.RAW_SUM].item(), 1) for i in range(8)]}")
    print(f"  PASS: {all_42}")

    print("\n" + "=" * 60)
    print("Attention-based ops tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_attention_ops()
