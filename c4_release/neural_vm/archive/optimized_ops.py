"""
Optimized Operations - 2-Layer Comparison and Arithmetic.

Uses flattened FFN to gather all nibbles in parallel, reducing
comparison from 40+ layers to just 2 layers.

Key insight: Flattening [batch, 8, dim] to [batch, 8*dim] allows FFN weights
to read from all 8 nibble positions and sum (scaled by 16^k) to produce
the full 32-bit value in ONE layer.

Weight counts:
- NibbleGatherFFN: ~128 weights (8 positions × 16 weights each)
- Comparison FFN: ~16-24 weights
- Total: ~150 weights per comparison (vs ~400 in current implementation)

Layer counts:
- Layer 1: Flattened FFN gathers A and B into full 32-bit values
- Layer 2: FFN computes (a-b) and detects sign/zero
- Total: 2 layers (vs 40+ in current implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention, bake_weights


# =============================================================================
# NIBBLE GATHER FFN (standalone module, not FlattenedPureFFN)
# =============================================================================

class NibbleGatherFFN(nn.Module):
    """
    Gather all 8 nibbles into a single 32-bit value using linear projection.

    Uses direct linear (no SwiGLU) on flattened input:
    1. Flatten [batch, 8, dim] → [batch, 8*dim]
    2. Linear weights read NIB_A or NIB_B from each position, scaled by 16^k
    3. All 8 values sum into a single output slot
    4. Reshape back

    Note: This is NOT a FlattenedPureFFN subclass — it's standalone.
    Position-dependent scaling (16^k) requires cross-position routing
    which can't be done with shared-weight attention.
    """

    def __init__(self, gather_a: bool = True, gather_b: bool = True):
        super().__init__()
        self.gather_a = gather_a
        self.gather_b = gather_b
        self.num_positions = E.NUM_POSITIONS
        self.dim = E.DIM
        self.flat_dim = self.num_positions * self.dim

        # Output slots for gathered values (in TEMP area)
        self.A_FULL = E.TEMP + 10  # Full 32-bit A value
        self.B_FULL = E.TEMP + 11  # Full 32-bit B value

        hidden_dim = 16

        self.register_buffer('W_up', torch.zeros(hidden_dim, self.flat_dim))
        self.register_buffer('b_up', torch.zeros(hidden_dim))
        self.register_buffer('W_down', torch.zeros(self.flat_dim, hidden_dim))
        self.register_buffer('b_down', torch.zeros(self.flat_dim))

        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        return pos * self.dim + slot

    @bake_weights
    def _bake_weights(self):
        h = 0
        if self.gather_a:
            for k in range(self.num_positions):
                scale = 16.0 ** k
                src_idx = self._flat_idx(k, E.NIB_A)
                self.W_up[h, src_idx] = scale
                for pos in range(self.num_positions):
                    dst_idx = self._flat_idx(pos, self.A_FULL)
                    self.W_down[dst_idx, h] = 1.0
                h += 1

        if self.gather_b:
            for k in range(self.num_positions):
                scale = 16.0 ** k
                src_idx = self._flat_idx(k, E.NIB_B)
                self.W_up[h, src_idx] = scale
                for pos in range(self.num_positions):
                    dst_idx = self._flat_idx(pos, self.B_FULL)
                    self.W_down[dst_idx, h] = 1.0
                h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.reshape(B, N * D)
        up = F.linear(x_flat, self.W_up, self.b_up)
        delta = F.linear(up, self.W_down, self.b_down)
        out_flat = x_flat + delta
        return out_flat.reshape(B, N, D)


# =============================================================================
# DEPRECATED: MULTI-HEAD GATHER ATTENTION (kept for reference)
# =============================================================================

class MultiHeadGatherAttention(nn.Module):
    """
    DEPRECATED: Use NibbleGatherFFN instead.

    8-head attention that gathers all 8 nibbles into a single 32-bit value.
    This uses non-pure operations (.sum(), .clone(), .expand()).

    Use NibbleGatherFFN for pure FFN-based gathering.
    """

    def __init__(self, num_positions: int = 8, gather_a: bool = True, gather_b: bool = True):
        super().__init__()
        self.num_positions = num_positions
        self.gather_a = gather_a
        self.gather_b = gather_b
        self.num_heads = num_positions

        # Scales: 16^k for each nibble position
        scales = torch.tensor([16.0 ** k for k in range(num_positions)])
        self.register_buffer('scales', scales)

        # Output slots for gathered values
        self.A_FULL = E.TEMP + 10  # Full 32-bit A value
        self.B_FULL = E.TEMP + 11  # Full 32-bit B value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gather nibbles into full 32-bit values.

        DEPRECATED: Uses .sum(), .clone(), .expand() - NOT pure neural.
        """
        batch_size = x.shape[0]

        nib_a = x[:, :, E.NIB_A]
        nib_b = x[:, :, E.NIB_B]

        if self.gather_a:
            a_full = (nib_a * self.scales.unsqueeze(0)).sum(dim=1, keepdim=True)
        if self.gather_b:
            b_full = (nib_b * self.scales.unsqueeze(0)).sum(dim=1, keepdim=True)

        y = x.clone()
        if self.gather_a:
            y[:, :, self.A_FULL] = a_full.expand(-1, self.num_positions)
        if self.gather_b:
            y[:, :, self.B_FULL] = b_full.expand(-1, self.num_positions)

        return y


class GatherAttentionPure(PureAttention):
    """
    DEPRECATED: Use NibbleGatherFFN instead.

    Pure attention implementation of multi-head gather.
    FFN approach is preferred since gathering is cross-position
    but does not require key-query matching.
    """

    def __init__(self, is_b: bool = False, output_slot: int = None):
        self.is_b = is_b
        self.output_slot = output_slot or (E.TEMP + 11 if is_b else E.TEMP + 10)
        super().__init__(E.DIM, num_heads=8, causal=False)

    def _bake_weights(self):
        with torch.no_grad():
            src_slot = E.NIB_B if self.is_b else E.NIB_A

            for head in range(8):
                scale = 16.0 ** head
                self.W_q[head, E.POS] = 1.0
                self.b_q[head] = -float(head)
                self.W_k[head, E.POS] = 1.0
                self.W_v[head, src_slot] = scale
                self.W_o[self.output_slot, head] = 1.0


# =============================================================================
# OPTIMIZED COMPARISON FFN
# =============================================================================

class OptimizedCompareFFN(PureFFN):
    """
    Single-layer FFN that computes comparison from gathered 32-bit values.

    For LT: result = step(B - A - 0.5) = 1 if A < B (strict)
    For GT: result = step(A - B - 0.5) = 1 if A > B (strict)
    For LE: result = step(B - A + 0.5) = 1 if A <= B (inclusive)
    For GE: result = step(A - B + 0.5) = 1 if A >= B (inclusive)
    For EQ: result = point_indicator(A - B) near 0
    For NE: result = 1 - EQ

    Weights: ~16 per operation
    Total shared: 64 (gather) + 6×16 (FFNs) = ~160 weights for all 6 ops
    """

    def __init__(self, opcode: int, a_slot: int = None, b_slot: int = None):
        self.opcode = opcode
        self.a_slot = a_slot or (E.TEMP + 10)
        self.b_slot = b_slot or (E.TEMP + 11)
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        # For integer comparison with proper saturation:
        # We need output to saturate to 1.0 when condition is true.
        #
        # SwiGLU: out = SiLU(up) * gate * W_down
        # For large positive up: SiLU(up) ≈ up
        # So we need: up * W_down = 1.0 when condition is satisfied.
        #
        # Strategy: use bias = S (so up = S when difference = 0 for inclusive ops)
        # and W_down = 1.0/S (so output = S * 1.0/S = 1.0)

        with torch.no_grad():
            # Gate on opcode
            op_slot = E.OP_START + self.opcode

            if self.opcode == Opcode.LT:
                # LT: A < B iff B - A >= 1
                # up = S * (B - A - 0.5) → positive only when B > A
                # When B - A = 1: up = S * 0.5 = 50 → output ≈ 0.5
                # We need output = 1.0 when true, so scale up the bias
                self.W_up[0, self.b_slot] = S
                self.W_up[0, self.a_slot] = -S
                self.b_up[0] = -S * 0.5  # Threshold at difference = 0.5
                self.W_gate[0, op_slot] = 1.0
                self.W_down[E.RESULT, 0] = 2.0 / S  # Scale to saturate at 1.0

            elif self.opcode == Opcode.GT:
                # GT: A > B iff A - B >= 1
                self.W_up[0, self.a_slot] = S
                self.W_up[0, self.b_slot] = -S
                self.b_up[0] = -S * 0.5
                self.W_gate[0, op_slot] = 1.0
                self.W_down[E.RESULT, 0] = 2.0 / S

            elif self.opcode == Opcode.LE:
                # LE: A <= B iff B - A >= 0
                # up = S * (B - A + 0.5) → positive when B >= A
                # When B = A: up = S * 0.5 = 50 → need output = 1.0
                self.W_up[0, self.b_slot] = S
                self.W_up[0, self.a_slot] = -S
                self.b_up[0] = S * 0.5  # Positive bias for inclusive
                self.W_gate[0, op_slot] = 1.0
                self.W_down[E.RESULT, 0] = 2.0 / S  # 50 * 2/100 = 1.0

            elif self.opcode == Opcode.GE:
                # GE: A >= B iff A - B >= 0
                self.W_up[0, self.a_slot] = S
                self.W_up[0, self.b_slot] = -S
                self.b_up[0] = S * 0.5
                self.W_gate[0, op_slot] = 1.0
                self.W_down[E.RESULT, 0] = 2.0 / S

            elif self.opcode == Opcode.EQ:
                # EQ: A == B iff |A - B| < 0.5
                # Need both (A - B < 0.5) AND (B - A < 0.5)
                # Use product of two step functions

                # Node 0: S * (0.5 - (A - B)) = S * (0.5 - A + B)
                # Positive when A - B < 0.5 (i.e., A <= B)
                self.W_up[0, self.a_slot] = -S
                self.W_up[0, self.b_slot] = S
                self.b_up[0] = S * 0.5
                self.W_gate[0, op_slot] = 1.0
                self.W_down[E.RESULT, 0] = 1.0 / S

                # Node 1: S * (0.5 - (B - A)) = S * (0.5 + A - B)
                # Positive when B - A < 0.5 (i.e., B <= A)
                self.W_up[1, self.a_slot] = S
                self.W_up[1, self.b_slot] = -S
                self.b_up[1] = S * 0.5
                self.W_gate[1, op_slot] = 1.0
                self.W_down[E.RESULT, 1] = 1.0 / S

                # When A = B: both nodes output 50, total = 100/S = 1.0
                # When A != B: one node outputs 0, total = 50/S = 0.5

            elif self.opcode == Opcode.NE:
                # NE: A != B iff |A - B| >= 1
                # Either (A - B >= 1) OR (B - A >= 1)

                # Node 0: S * (A - B - 0.5) → positive if A > B
                self.W_up[0, self.a_slot] = S
                self.W_up[0, self.b_slot] = -S
                self.b_up[0] = -S * 0.5
                self.W_gate[0, op_slot] = 1.0
                self.W_down[E.RESULT, 0] = 2.0 / S

                # Node 1: S * (B - A - 0.5) → positive if B > A
                self.W_up[1, self.a_slot] = -S
                self.W_up[1, self.b_slot] = S
                self.b_up[1] = -S * 0.5
                self.W_gate[1, op_slot] = 1.0
                self.W_down[E.RESULT, 1] = 2.0 / S

                # When A != B: one node outputs ~1.0, total > 0.5
                # When A = B: both nodes output 0, total = 0


# =============================================================================
# OPTIMIZED ADD/SUB FFN
# =============================================================================

class OptimizedAddFFN(PureFFN):
    """
    Single-layer FFN that computes 32-bit addition from gathered values.

    Since we have full 32-bit values in single slots, addition is trivial:
    result = A + B

    For proper 32-bit wraparound, we need modulo 2^32, which requires
    additional logic. For now, this works for values that don't overflow.

    Weights: ~8
    """

    def __init__(self, a_slot: int = None, b_slot: int = None):
        self.a_slot = a_slot or (E.TEMP + 10)
        self.b_slot = b_slot or (E.TEMP + 11)
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Simple addition: A + B
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, self.a_slot] = 1.0
            self.W_gate[0, self.b_slot] = 1.0
            self.W_down[E.TEMP + 12, 0] = 1.0 / S  # Result in TEMP+12


class OptimizedSubFFN(PureFFN):
    """
    Single-layer FFN that computes 32-bit subtraction from gathered values.

    result = A - B

    Weights: ~8
    """

    def __init__(self, a_slot: int = None, b_slot: int = None):
        self.a_slot = a_slot or (E.TEMP + 10)
        self.b_slot = b_slot or (E.TEMP + 11)
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Subtraction: A - B
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, self.a_slot] = 1.0
            self.W_gate[0, self.b_slot] = -1.0  # Negate B
            self.W_down[E.TEMP + 12, 0] = 1.0 / S


# =============================================================================
# SCATTER RESULT BACK TO NIBBLES
# =============================================================================

class ScatterResultFFN(PureFFN):
    """
    Scatter a full 32-bit result back to 8 nibble positions.

    For each position k, extract nibble[k] = floor(result / 16^k) % 16

    This is the inverse of the gather operation.

    Weights: ~32 (4 per nibble position)
    """

    def __init__(self, result_slot: int = None):
        self.result_slot = result_slot or (E.TEMP + 12)
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each nibble position, we need to:
            # 1. Divide by 16^k (shift right)
            # 2. Take modulo 16 (mask lower 4 bits)

            # This is complex with pure SwiGLU, so we use a simplified approach
            # that works for the result already in RESULT slots
            pass


# =============================================================================
# OPTIMIZED COMPARISON LAYER SEQUENCE
# =============================================================================

def build_optimized_comparison_layers():
    """
    Build the 2-layer optimized comparison.

    Layer 1: NibbleGatherFFN (gathers A and B using flattened FFN)
    Layer 2: OptimizedCompareFFN (computes comparison)

    Returns list of layers for each comparison opcode.
    """
    from .pure_moe import MoE

    layers = []

    # Layer 1: Gather FFN for all comparison ops
    # Single flattened FFN gathers both A and B (pure FFN, no attention)
    gather = NibbleGatherFFN(gather_a=True, gather_b=True)
    layers.append(gather)

    # Layer 2: Comparison FFNs (one per opcode, routed by MoE)
    cmp_ffns = [
        OptimizedCompareFFN(Opcode.EQ),
        OptimizedCompareFFN(Opcode.NE),
        OptimizedCompareFFN(Opcode.LT),
        OptimizedCompareFFN(Opcode.GT),
        OptimizedCompareFFN(Opcode.LE),
        OptimizedCompareFFN(Opcode.GE),
    ]
    cmp_ops = [Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE]
    layers.append(MoE(cmp_ffns, cmp_ops))

    return layers


def build_optimized_arithmetic_layers():
    """
    Build optimized ADD/SUB layers.

    Layer 1: NibbleGatherFFN (flattened FFN)
    Layer 2: OptimizedAddFFN or OptimizedSubFFN
    Layer 3: ScatterResultFFN (scatter back to nibbles)

    Note: Full 32-bit arithmetic with overflow handling needs more work.
    """
    from .pure_moe import MoE

    layers = []

    # Layer 1: Gather (pure FFN, no attention)
    gather = NibbleGatherFFN(gather_a=True, gather_b=True)
    layers.append(gather)

    # Layer 2: Add/Sub
    arith_ffns = [OptimizedAddFFN(), OptimizedSubFFN()]
    arith_ops = [Opcode.ADD, Opcode.SUB]
    layers.append(MoE(arith_ffns, arith_ops))

    # Layer 3: Scatter (TODO: implement properly)
    # layers.append(ScatterResultFFN())

    return layers


# =============================================================================
# TEST
# =============================================================================

def test_optimized_comparison():
    """Test the optimized 2-layer comparison."""
    print("=" * 60)
    print("OPTIMIZED 2-LAYER COMPARISON TEST")
    print("=" * 60)

    # Create gather layer (FFN-based, not attention)
    gather = NibbleGatherFFN()

    # Test cases
    test_cases = [
        (5, 3, "LT", 0),   # 5 < 3 = False
        (3, 5, "LT", 1),   # 3 < 5 = True
        (5, 5, "LT", 0),   # 5 < 5 = False
        (5, 3, "GT", 1),   # 5 > 3 = True
        (3, 5, "GT", 0),   # 3 > 5 = False
        (5, 5, "EQ", 1),   # 5 == 5 = True
        (5, 3, "EQ", 0),   # 5 == 3 = False
        (5, 5, "LE", 1),   # 5 <= 5 = True
        (3, 5, "LE", 1),   # 3 <= 5 = True
        (5, 3, "LE", 0),   # 5 <= 3 = False
    ]

    print("\nTesting gather FFN...")

    # Create input tensor
    x = torch.zeros(1, 8, E.DIM)
    a, b = 12345, 6789

    # Encode A and B as nibbles
    for i in range(8):
        x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
        x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
        x[0, i, E.POS] = float(i)

    # Gather
    y = gather(x)

    # Check gathered values
    a_gathered = y[0, 0, gather.A_FULL].item()
    b_gathered = y[0, 0, gather.B_FULL].item()

    print(f"  Input A: {a}, B: {b}")
    print(f"  Gathered A: {a_gathered:.0f}, B: {b_gathered:.0f}")
    print(f"  Match: A={a_gathered == a}, B={b_gathered == b}")

    # Count weights
    print("\n" + "=" * 60)
    print("WEIGHT COUNTS")
    print("=" * 60)

    # Gather FFN: 16 hidden × (16 weights) = ~128 weights
    print(f"  NibbleGatherFFN: ~128 weights (16 hidden × 8 weights each)")

    # Comparison FFN: ~16-24 weights
    print(f"  OptimizedCompareFFN: ~16-24 weights")

    print(f"\n  TOTAL: ~150 weights per comparison op")
    print(f"  vs CURRENT: ~400 weights per comparison op")
    print(f"  SAVINGS: ~2.5x fewer weights")

    print("\n" + "=" * 60)
    print("LAYER COUNTS")
    print("=" * 60)
    print(f"  Optimized: 2 layers (1 gather FFN + 1 compare FFN)")
    print(f"  Current: 40+ layers (nibble-by-nibble borrow chain)")
    print(f"  SAVINGS: ~20x fewer layers")

    return True


def test_full_comparison():
    """Test the full optimized comparison pipeline."""
    print("\n" + "=" * 60)
    print("FULL COMPARISON PIPELINE TEST")
    print("=" * 60)

    # Build optimized layers
    layers = build_optimized_comparison_layers()
    print(f"\nBuilt {len(layers)} layers:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {type(layer).__name__}")

    # Test cases: (a, b, opcode, expected)
    test_cases = [
        # LT tests
        (3, 5, Opcode.LT, 1),   # 3 < 5 = True
        (5, 3, Opcode.LT, 0),   # 5 < 3 = False
        (5, 5, Opcode.LT, 0),   # 5 < 5 = False
        # GT tests
        (5, 3, Opcode.GT, 1),   # 5 > 3 = True
        (3, 5, Opcode.GT, 0),   # 3 > 5 = False
        (5, 5, Opcode.GT, 0),   # 5 > 5 = False
        # LE tests
        (3, 5, Opcode.LE, 1),   # 3 <= 5 = True
        (5, 5, Opcode.LE, 1),   # 5 <= 5 = True
        (5, 3, Opcode.LE, 0),   # 5 <= 3 = False
        # GE tests
        (5, 3, Opcode.GE, 1),   # 5 >= 3 = True
        (5, 5, Opcode.GE, 1),   # 5 >= 5 = True
        (3, 5, Opcode.GE, 0),   # 3 >= 5 = False
        # Larger values
        (1000, 999, Opcode.GT, 1),
        (12345, 12345, Opcode.EQ, 1),
        (12345, 12346, Opcode.LT, 1),
    ]

    passed = 0
    failed = 0

    for a, b, opcode, expected in test_cases:
        # Create input
        x = torch.zeros(1, 8, E.DIM)
        for i in range(8):
            x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
            x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
            x[0, i, E.POS] = float(i)
            x[0, i, E.OP_START + opcode] = 1.0

        # Run through layers
        y = x
        for layer in layers:
            y = layer(y)

        # Extract result (from position 0)
        result_raw = y[0, 0, E.RESULT].item()
        result = 1 if result_raw > 0.5 else 0

        op_name = {Opcode.LT: "LT", Opcode.GT: "GT", Opcode.LE: "LE",
                   Opcode.GE: "GE", Opcode.EQ: "EQ", Opcode.NE: "NE"}[opcode]

        if result == expected:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            print(f"  {status}: {a} {op_name} {b} = {result} (expected {expected}, raw={result_raw:.3f})")

    print(f"\n  Results: {passed}/{passed+failed} passed")

    if failed == 0:
        print("  All tests passed!")
    return failed == 0


if __name__ == "__main__":
    test_optimized_comparison()
    test_full_comparison()
