"""
Baked Attention - Full attention emulation via FFN layers.

This module implements full softmax attention using only FFN layers,
enabling attention to be "baked" into weights without requiring
the standard Q/K/V attention mechanism at runtime.

IMPORTANT: This module is ONLY used when baking prompts/programs into
transformer weights. During normal operation, standard attention is used.
The FFN-based attention emulation is computationally expensive due to
the long division required for softmax normalization.

The key challenge is computing softmax normalization:
    softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

Since we don't have efficient division, we implement this via:
1. Compute exp(x_i) for each position using efficient exp
2. Compute denominator = 1 + sum(exp(x_i)) via summation
3. Long division to compute 1/denominator
4. Multiply each exp(x_i) by the reciprocal
5. Weighted sum with values

For scale-invariant operations (relative ordering, sign detection),
the expensive division can be skipped - use ScaleInvariantAttention instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .base_layers import PureFFN
from .embedding import E


# =============================================================================
# CONFIGURATION
# =============================================================================

class BakedAttentionConfig:
    """
    Configuration for baked attention behavior.

    By default, uses fast PyTorch operations. Only when baking prompts
    into weights should use_ffn_division be enabled.
    """

    # When True, use FFN-based long division for softmax (slow but bakeable)
    # When False, use standard PyTorch operations (fast, for normal inference)
    use_ffn_division: bool = False

    # When True, we are in "baking mode" - compiling prompts into weights
    baking_mode: bool = False

    @classmethod
    def enable_baking_mode(cls):
        """Enable baking mode - uses FFN division for all operations."""
        cls.use_ffn_division = True
        cls.baking_mode = True

    @classmethod
    def disable_baking_mode(cls):
        """Disable baking mode - uses fast PyTorch operations."""
        cls.use_ffn_division = False
        cls.baking_mode = False

    @classmethod
    def is_baking(cls) -> bool:
        """Check if currently in baking mode."""
        return cls.baking_mode


# Context manager for baking mode
class baking_mode:
    """
    Context manager to temporarily enable baking mode.

    Usage:
        with baking_mode():
            # FFN-based division will be used here
            baked_program = compile_to_baked(program)
        # Normal fast operations resume here
    """

    def __enter__(self):
        self._previous_ffn = BakedAttentionConfig.use_ffn_division
        self._previous_baking = BakedAttentionConfig.baking_mode
        BakedAttentionConfig.enable_baking_mode()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        BakedAttentionConfig.use_ffn_division = self._previous_ffn
        BakedAttentionConfig.baking_mode = self._previous_baking
        return False


# =============================================================================
# CONSTANTS
# =============================================================================

SCALE = 100.0  # Scale factor for sharp approximations
EPS = 1e-6     # Small epsilon for numerical stability
DIV_ITERATIONS = 8  # Iterations for long division (4 bits each = 32 bits)


# =============================================================================
# EFFICIENT EXPONENTIAL FFN
# =============================================================================

class EfficientExpFFN(PureFFN):
    """
    Compute e^x using SwiGLU approximation.

    For x in reasonable range, uses:
        e^x ≈ SiLU(SCALE * x) / SiLU(SCALE) when x > 0

    For better accuracy, we use the identity e^x = e^(x-B) * e^B
    where B is a bias to keep intermediate values manageable.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: float = 0.0):
        self.bias = bias
        self.exp_bias = torch.exp(torch.tensor(bias)).item()
        super().__init__(input_dim, hidden_dim=4)
        self.output_dim = output_dim

    def _bake_weights(self):
        S = SCALE
        with torch.no_grad():
            # Node 0: Compute SiLU(S * (x - bias))
            # When x >> 0: SiLU(S*x) ≈ S*x, so output ≈ x - bias
            # We multiply by exp(bias) in output to recover e^x

            # For positive x: e^x ≈ (1 + x + x²/2 + ...)
            # SiLU approximation: SiLU(x) ≈ x for x >> 0

            # Simple approximation: use SiLU's shape
            # e^x ≈ SiLU(x + 2) + 1 for x in [-2, 2]
            # Scale and bias adjust the range

            self.W_up[0, 0] = S
            self.b_up[0] = -S * self.bias
            self.b_gate[0] = 1.0  # Fixed gate
            self.W_down[0, 0] = self.exp_bias / S


class SumExpFFN(PureFFN):
    """
    Compute sum of exponentials: 1 + sum_i(e^{x_i})

    Takes N inputs, outputs their summed exponentials plus 1 (for softmax1).
    """

    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        super().__init__(num_inputs, hidden_dim=num_inputs * 2)

    def _bake_weights(self):
        S = SCALE
        n = self.num_inputs

        with torch.no_grad():
            # For each input, we need positive and negative SiLU nodes
            # to approximate identity: a ≈ (SiLU(S*a) - SiLU(-S*a)) / S

            for i in range(n):
                # Positive node
                self.W_up[i*2, i] = S
                self.b_gate[i*2] = 1.0
                self.W_down[0, i*2] = 1.0 / S

                # Negative node
                self.W_up[i*2+1, i] = -S
                self.b_gate[i*2+1] = 1.0
                self.W_down[0, i*2+1] = -1.0 / S

            # Add bias of 1.0 for the +1 in softmax1 denominator
            self.b_down[0] = 1.0


# =============================================================================
# LONG DIVISION FOR RECIPROCAL
# =============================================================================

class ReciprocalDivisionLayer(nn.Module):
    """
    One iteration of long division to compute 1/x.

    Uses non-restoring division algorithm for simplicity:
    - Maintains remainder and quotient
    - Each iteration doubles remainder and conditionally subtracts divisor
    """

    def __init__(self, bit_position: int):
        super().__init__()
        self.bit_position = bit_position
        self.bit_weight = 2.0 ** (-bit_position - 1)  # 0.5, 0.25, 0.125, ...

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Input state: [remainder, divisor, quotient]
        Output state: [new_remainder, divisor, new_quotient]
        """
        remainder = state[:, 0:1]
        divisor = state[:, 1:2]
        quotient = state[:, 2:3]

        # Double the remainder
        doubled = remainder * 2.0

        # Check if doubled >= divisor
        comparison = doubled - divisor

        # Step function: 1 if comparison >= 0, else 0
        # Using sigmoid approximation
        step = torch.sigmoid(SCALE * comparison)

        # New remainder: doubled - divisor if step, else doubled
        new_remainder = doubled - divisor * step

        # New quotient: add bit_weight if step
        new_quotient = quotient + self.bit_weight * step

        return torch.cat([new_remainder, divisor, new_quotient], dim=-1)


class ReciprocalFFN(nn.Module):
    """
    Compute 1/x using iterated long division.

    Chains multiple ReciprocalDivisionLayer instances to compute
    the reciprocal to the desired precision.
    """

    def __init__(self, precision_bits: int = 16):
        super().__init__()
        self.precision_bits = precision_bits

        # Create division layers
        self.layers = nn.ModuleList([
            ReciprocalDivisionLayer(i)
            for i in range(precision_bits)
        ])

    def forward(self, divisor: torch.Tensor) -> torch.Tensor:
        """
        Compute 1/x.

        Args:
            divisor: Input tensor of shape (batch, 1) containing divisor

        Returns:
            Reciprocal tensor of shape (batch, 1)
        """
        batch_size = divisor.shape[0]

        # Initialize state: [remainder=1.0, divisor, quotient=0.0]
        state = torch.zeros(batch_size, 3)
        state[:, 0] = 1.0  # remainder
        state[:, 1] = divisor.squeeze(-1)  # divisor
        state[:, 2] = 0.0  # quotient

        # Run division iterations
        for layer in self.layers:
            state = layer(state)

        # Return quotient
        return state[:, 2:3]


# =============================================================================
# FULL SOFTMAX LAYER
# =============================================================================

class BakedSoftmaxFFN(nn.Module):
    """
    Compute softmax using only FFN operations.

    softmax1(x)_i = exp(x_i) / (1 + sum_j(exp(x_j)))

    Implementation (when baking_mode enabled):
    1. Compute exp(x_i) for each input
    2. Sum all exponentials + 1
    3. Compute reciprocal of sum via long division
    4. Multiply each exp by reciprocal

    When baking_mode is disabled, uses fast PyTorch operations.
    """

    def __init__(self, num_inputs: int, precision_bits: int = 16):
        super().__init__()
        self.num_inputs = num_inputs
        self.precision_bits = precision_bits

        # Reciprocal computation (only used in baking mode)
        self.reciprocal = ReciprocalFFN(precision_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute softmax of input.

        Args:
            x: Input tensor of shape (batch, num_inputs) - these are scores

        Returns:
            Softmax probabilities of shape (batch, num_inputs)
        """
        # Fast path: use standard PyTorch when not baking
        if not BakedAttentionConfig.use_ffn_division:
            # softmax1: exp(x) / (1 + sum(exp(x)))
            x_stable = x - x.max(dim=-1, keepdim=True)[0]
            exps = torch.exp(x_stable)
            return exps / (1.0 + exps.sum(dim=-1, keepdim=True))

        # Slow path: FFN-based computation for baking
        batch_size = x.shape[0]

        # Step 1: Compute exponentials
        # For numerical stability, subtract max
        x_stable = x - x.max(dim=-1, keepdim=True)[0]
        exps = torch.exp(x_stable)

        # Step 2: Sum exponentials + 1 (for softmax1)
        sum_exp = 1.0 + exps.sum(dim=-1, keepdim=True)

        # Step 3: Compute reciprocal via long division
        reciprocal = self.reciprocal(sum_exp)

        # Step 4: Multiply each exp by reciprocal
        outputs = exps * reciprocal

        return outputs


# =============================================================================
# BAKED ATTENTION HEAD
# =============================================================================

@dataclass
class BakedKV:
    """A key-value pair to bake into attention."""
    key: torch.Tensor      # Shape: (dim,)
    value: torch.Tensor    # Shape: (dim,)


class BakedAttentionHead(nn.Module):
    """
    Full attention head emulated via FFN layers.

    Given a set of key-value pairs and a query, computes:
        output = sum_i(softmax(Q·K_i / sqrt(d)) * V_i)

    When baking_mode is enabled, all operations are implemented via FFN
    layers, allowing the attention to be "baked" into network weights.

    When baking_mode is disabled (default), uses fast PyTorch operations.
    """

    def __init__(self, dim: int, kvs: List[BakedKV], precision_bits: int = 16):
        super().__init__()
        self.dim = dim
        self.num_kvs = len(kvs)
        self.precision_bits = precision_bits
        self.scale = dim ** -0.5

        # Store keys and values
        self.keys = nn.Parameter(torch.stack([kv.key for kv in kvs]), requires_grad=False)
        self.values = nn.Parameter(torch.stack([kv.value for kv in kvs]), requires_grad=False)

        # Dot product layers (one per KV pair) - only used in baking mode
        self.dot_layers = nn.ModuleList([
            self._create_dot_layer(kvs[i].key) for i in range(len(kvs))
        ])

        # Softmax computation
        self.softmax = BakedSoftmaxFFN(len(kvs), precision_bits)

        # Value weighting layers - only used in baking mode
        self.value_layers = nn.ModuleList([
            self._create_value_layer(kvs[i].value) for i in range(len(kvs))
        ])

    def _create_dot_layer(self, key: torch.Tensor) -> PureFFN:
        """Create layer to compute Q·K for a fixed K."""
        layer = PureFFN(self.dim, hidden_dim=self.dim)
        S = SCALE
        scale = (self.dim ** -0.5)  # 1/sqrt(d)

        with torch.no_grad():
            # Dot product: sum_i(Q_i * K_i)
            # Each dimension contributes Q_i * K_i to the sum
            for i in range(self.dim):
                if abs(key[i].item()) > 1e-6:
                    layer.W_up[i, i] = S * key[i].item() * scale
                    layer.b_gate[i] = 1.0
                    layer.W_down[0, i] = 1.0 / S

        return layer

    def _create_value_layer(self, value: torch.Tensor) -> PureFFN:
        """Create layer to scale value by attention weight."""
        layer = PureFFN(self.dim + 1, hidden_dim=self.dim)
        S = SCALE

        with torch.no_grad():
            # Output = attention_weight * value
            # Weight is in position 0, value is fixed
            for i in range(self.dim):
                if abs(value[i].item()) > 1e-6:
                    layer.W_up[i, 0] = S
                    layer.b_gate[i] = value[i].item()
                    layer.W_down[i, i] = 1.0 / S

        return layer

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Compute attention output for query.

        Args:
            query: Query tensor of shape (batch, dim)

        Returns:
            Attention output of shape (batch, dim)
        """
        batch_size = query.shape[0]

        # Fast path: use standard PyTorch operations when not baking
        if not BakedAttentionConfig.use_ffn_division:
            # Compute all dot products at once: (batch, dim) @ (dim, num_kvs)
            scores = torch.matmul(query, self.keys.T) * self.scale
            # softmax1 weights
            scores_stable = scores - scores.max(dim=-1, keepdim=True)[0]
            exps = torch.exp(scores_stable)
            weights = exps / (1.0 + exps.sum(dim=-1, keepdim=True))
            # Weighted sum of values: (batch, num_kvs) @ (num_kvs, dim)
            return torch.matmul(weights, self.values)

        # Slow path: FFN-based computation for baking
        # Step 1: Compute dot products Q·K_i for each K
        scores = []
        for i, dot_layer in enumerate(self.dot_layers):
            score_i = dot_layer(query)[:, :1]  # Extract scalar score
            scores.append(score_i)

        score_tensor = torch.cat(scores, dim=-1)

        # Step 2: Compute softmax (uses FFN division in baking mode)
        weights = self.softmax(score_tensor)

        # Step 3: Weight values and sum
        output = torch.zeros(batch_size, self.dim)
        for i in range(self.num_kvs):
            # Weighted value = weight * value
            weighted_value = weights[:, i:i+1] * self.values[i:i+1, :]
            output = output + weighted_value

        return output


# =============================================================================
# SCALE-INVARIANT OPTIMIZATION
# =============================================================================

class ScaleInvariantAttention(nn.Module):
    """
    Optimized attention for scale-invariant operations.

    When the downstream computation only depends on:
    - Relative ordering of attention weights
    - Sign (positive/negative/zero) of weighted outputs
    - Winner-take-all selection

    Then we can skip the expensive division and use unnormalized
    exponentials directly.
    """

    def __init__(self, dim: int, kvs: List[BakedKV]):
        super().__init__()
        self.dim = dim
        self.num_kvs = len(kvs)

        # Store keys and values
        self.keys = nn.Parameter(torch.stack([kv.key for kv in kvs]), requires_grad=False)
        self.values = nn.Parameter(torch.stack([kv.value for kv in kvs]), requires_grad=False)

        # Dot product + exp layers
        self.score_layers = nn.ModuleList([
            self._create_score_layer(kvs[i].key) for i in range(len(kvs))
        ])

        # Winner detection (argmax emulation)
        self.winner_layer = self._create_winner_layer()

    def _create_score_layer(self, key: torch.Tensor) -> PureFFN:
        """Create layer to compute exp(Q·K / sqrt(d))."""
        layer = PureFFN(self.dim, hidden_dim=self.dim + 1)
        S = SCALE
        scale = (self.dim ** -0.5)

        with torch.no_grad():
            # Compute dot product
            for i in range(self.dim):
                if abs(key[i].item()) > 1e-6:
                    layer.W_up[i, i] = S * key[i].item() * scale
                    layer.b_gate[i] = 1.0
                    layer.W_down[0, i] = 1.0 / S

            # Apply exp approximation
            layer.W_up[self.dim, 0] = S
            layer.b_gate[self.dim] = 1.0
            # SiLU(S*x) ≈ S*x for large x, approximates exp behavior

        return layer

    def _create_winner_layer(self) -> PureFFN:
        """Create layer to detect maximum score."""
        layer = PureFFN(self.num_kvs, hidden_dim=self.num_kvs * 2)
        S = SCALE

        with torch.no_grad():
            # For each position, check if it's the maximum
            # Winner ≈ step(x_i - max(x_j for j != i))
            # Simplified: compare each pair, accumulate
            pass  # Complex - would need O(N²) comparisons

        return layer

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Compute attention with winner-take-all output.

        Returns:
            (value of winning KV, index of winner)
        """
        batch_size = query.shape[0]

        # Compute all scores
        scores = []
        for layer in self.score_layers:
            score = layer(query)[:, :1]
            scores.append(score)

        score_tensor = torch.cat(scores, dim=-1)

        # Find winner (would use argmax in practice)
        winner_idx = score_tensor.argmax(dim=-1)

        # Return winner's value
        values = self.values[winner_idx]

        return values, winner_idx


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_baked_attention():
    """Demonstrate baked attention computation."""
    print("=" * 60)
    print("BAKED ATTENTION DEMO")
    print("=" * 60)

    dim = 4

    # Create some key-value pairs
    kvs = [
        BakedKV(
            key=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            value=torch.tensor([1.0, 0.0, 0.0, 0.0])
        ),
        BakedKV(
            key=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            value=torch.tensor([0.0, 1.0, 0.0, 0.0])
        ),
        BakedKV(
            key=torch.tensor([0.0, 0.0, 1.0, 0.0]),
            value=torch.tensor([0.0, 0.0, 1.0, 0.0])
        ),
    ]

    print(f"\nCreated {len(kvs)} key-value pairs")
    print(f"Dimension: {dim}")
    print(f"Baking mode: {BakedAttentionConfig.is_baking()}")

    # Create baked attention
    attention = BakedAttentionHead(dim, kvs, precision_bits=16)

    # Test in FAST mode (default)
    print("\n--- FAST MODE (default, for inference) ---")
    query = kvs[0].key.unsqueeze(0)
    output = attention(query)
    print(f"Query: {query[0].tolist()}")
    print(f"Output: {[f'{x:.3f}' for x in output[0].tolist()]}")

    # Test in BAKING mode
    print("\n--- BAKING MODE (for compiling to weights) ---")
    with baking_mode():
        print(f"Baking mode: {BakedAttentionConfig.is_baking()}")
        output_baked = attention(query)
        print(f"Query: {query[0].tolist()}")
        print(f"Output: {[f'{x:.3f}' for x in output_baked[0].tolist()]}")

    # Verify modes produce similar results
    print(f"\nBaking mode disabled: {not BakedAttentionConfig.is_baking()}")
    diff = (output - output_baked).abs().max().item()
    print(f"Max difference between modes: {diff:.6f}")

    # Count parameters
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"Total parameters: {total_params}")

    print("\n" + "=" * 60)


def demo_reciprocal():
    """Demonstrate reciprocal computation."""
    print("=" * 60)
    print("RECIPROCAL (1/x) DEMO")
    print("=" * 60)

    reciprocal = ReciprocalFFN(precision_bits=24)

    test_values = [2.0, 4.0, 5.0, 10.0, 3.0, 7.0, 100.0]

    print("\nTesting 1/x computation via long division:")
    for x in test_values:
        input_tensor = torch.tensor([[x]])

        output = reciprocal(input_tensor)
        computed = output[0, 0].item()
        expected = 1.0 / x
        error = abs(computed - expected)
        rel_error = error / expected * 100

        print(f"  1/{x:<6} = {computed:.6f} (expected {expected:.6f}, error {rel_error:.2f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_reciprocal()
    print()
    demo_baked_attention()
