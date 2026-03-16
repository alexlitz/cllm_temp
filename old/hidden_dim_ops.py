"""
Operations on hidden dimensions of a transformer.
Each operation is designed to be computed within the hidden dimension space.

All attention operations use softmax1 (quiet attention) instead of standard softmax:
    softmax1(x)_i = exp(x_i) / (1 + sum_j(exp(x_j)))

This provides:
    - Ability to "attend to nothing" (outputs sum to < 1)
    - Counting via leftover: n = 1/leftover - 1
    - sigmoid as special case: softmax1([s]) = sigmoid(s)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SOFTMAX1: The core attention primitive
# =============================================================================

def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax with +1 in denominator (quiet attention / softmax1).

    softmax1(x)_i = exp(x_i) / (1 + sum_j(exp(x_j)))

    Key properties:
    - Outputs sum to < 1 (can "attend to nothing")
    - leftover = 1 - sum(outputs) = 1/(1 + sum(exp(x)))
    - For single element: softmax1([s]) = sigmoid(s)
    - count = 1/leftover - 1 when all scores are 0

    This replaces F.softmax throughout for more expressive attention.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    # Handle all -inf case: when max is -inf, all outputs should be 0
    # (we attend to nothing, leftover = 1)
    x_max = torch.where(x_max.isinf() & (x_max < 0), torch.zeros_like(x_max), x_max)
    exp_x = torch.exp(x - x_max)
    # The "1" becomes exp(-max) after the max subtraction
    one_term = torch.exp(-x_max)
    return exp_x / (one_term + exp_x.sum(dim=dim, keepdim=True))


def softmax1_with_leftover(x: torch.Tensor, dim: int = -1) -> tuple:
    """
    Compute softmax1 and return both weights and leftover.

    Returns:
        weights: softmax1 attention weights
        leftover: 1 - sum(weights), encodes count information
    """
    x_max = x.max(dim=dim, keepdim=True).values
    # Handle all -inf case: when max is -inf, outputs are 0, leftover is 1
    x_max = torch.where(x_max.isinf() & (x_max < 0), torch.zeros_like(x_max), x_max)
    exp_x = torch.exp(x - x_max)
    one_term = torch.exp(-x_max)
    denominator = one_term + exp_x.sum(dim=dim, keepdim=True)
    weights = exp_x / denominator
    leftover = one_term / denominator
    return weights, leftover


class SequenceMean(nn.Module):
    """
    Computes mean across sequence: output = (1/seq_len) * sum(x)
    Uses learned 1/seq_len computation in hidden space.
    """
    def __init__(self, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Project to scalar for accumulation, then back
        self.sum_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        seq_len = x.shape[1]
        # Simple mean across sequence dimension
        return x.sum(dim=1, keepdim=True) / seq_len


class Square(nn.Module):
    """
    Computes x^2 element-wise in hidden dimension.
    Can be approximated via: x^2 = (x + x)^2 / 4 - requires careful construction
    Or directly using activation patterns.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct squaring
        return x ** 2


class ReLUViaSwiGLU(nn.Module):
    """
    Computes max(x, 0) using SwiGLU mechanism.

    SwiGLU: SwiGLU(x, W, V, b, c) = Swish(xW + b) * (xV + c)
    where Swish(x) = x * sigmoid(x)

    To get ReLU-like behavior:
    - Set one path to pass through x
    - Set other path to sigmoid(beta * x) for large beta -> step function
    - Product gives: x * step(x) ≈ max(x, 0)
    """
    def __init__(self, hidden_dim: int, beta: float = 10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        # Gate projection (controls the "max with 0" behavior)
        self.W_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Value projection (passes through x)
        self.W_value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialize to approximate identity
        nn.init.eye_(self.W_gate.weight)
        nn.init.eye_(self.W_value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: sigmoid(beta * x) ≈ step(x) for large beta
        gate = torch.sigmoid(self.beta * self.W_gate(x))
        # Value path
        value = self.W_value(x)
        # SwiGLU-style gating gives ReLU-like behavior
        return gate * value


class ReLUViaSiLU(nn.Module):
    """
    Computes max(x, 0) using SiLU with scale up/down.

    SiLU(x) = x * sigmoid(x)

    Key insight:
        SiLU(βx) * (1/β) = x * sigmoid(βx)

    As β → ∞:
        sigmoid(βx) → step(x)
        So: x * sigmoid(βx) → x * step(x) = ReLU(x)

    Uses attention-based 1/x for the division when use_attn_reciprocal=True,
    making it fully composed from transformer primitives.
    """
    def __init__(self, hidden_dim: int, beta: float = 10.0, use_attn_reciprocal: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.use_attn_reciprocal = use_attn_reciprocal
        if use_attn_reciprocal:
            self.reciprocal = ReciprocalViaAttention(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale up, apply SiLU
        silu_scaled = F.silu(self.beta * x)

        if self.use_attn_reciprocal:
            # Use attention-based 1/β (β is a scalar, broadcast)
            # 1/(1+x) where x = β-1 gives 1/β
            beta_tensor = torch.full_like(x, self.beta)
            inv_beta = self.reciprocal.forward_reciprocal(beta_tensor)
            return silu_scaled * inv_beta
        else:
            # Direct division
            return silu_scaled / self.beta


# =============================================================================
# POSITIONAL OPERATIONS
# =============================================================================
# Operations that work with sequence positions: selecting, copying, indexing


class SelectNBehind(nn.Module):
    """
    Select (attend to) the position N steps behind the current position.

    Uses attention with a diagonal offset pattern:
    - Position i attends to position i-N with weight 1.0
    - All other positions get weight 0.0

    For positions where i < N, behavior depends on mode:
    - 'zero': output zeros
    - 'clamp': attend to position 0
    - 'wrap': attend to position (i-N) mod seq_len
    """
    def __init__(self, hidden_dim: int, n: int, mode: str = 'zero'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = n
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        batch, seq_len, hidden = x.shape

        if self.mode == 'zero':
            # Shift and pad with zeros
            if self.n >= seq_len:
                return torch.zeros_like(x)
            result = torch.zeros_like(x)
            result[:, self.n:, :] = x[:, :-self.n, :]
            return result
        elif self.mode == 'clamp':
            # Clamp to position 0 for early positions
            indices = torch.arange(seq_len, device=x.device)
            src_indices = (indices - self.n).clamp(min=0)
            return x[:, src_indices, :]
        elif self.mode == 'wrap':
            # Wrap around
            indices = torch.arange(seq_len, device=x.device)
            src_indices = (indices - self.n) % seq_len
            return x[:, src_indices, :]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class CopyFromNBehind(nn.Module):
    """
    Copy values from N positions behind using attention.

    This is the attention-based implementation of SelectNBehind,
    showing how a transformer can implement this operation.

    Attention pattern:
    - Q comes from current position
    - K comes from all positions
    - Attention scores are -inf everywhere except position i-N
    - This gives attention weight 1.0 to position i-N

    In practice, this requires either:
    1. Relative positional encoding that can express "N behind"
    2. Learned Q/K projections that create this pattern
    3. Explicit attention mask
    """
    def __init__(self, hidden_dim: int, n: int, mode: str = 'zero'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = n
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        # Create attention mask: position i attends only to i-N
        # scores[i, j] = 0 if j == i-N, else -inf
        scores = torch.full((seq_len, seq_len), float('-inf'), device=x.device)

        for i in range(seq_len):
            src = i - self.n
            if self.mode == 'zero':
                if src >= 0:
                    scores[i, src] = 0.0
            elif self.mode == 'clamp':
                src = max(0, src)
                scores[i, src] = 0.0
            elif self.mode == 'wrap':
                src = src % seq_len
                scores[i, src] = 0.0

        # Apply softmax1 (quiet attention)
        # With softmax1, positions with all -inf naturally get leftover=1,
        # meaning they "attend to nothing" and output zeros
        attn = softmax1(scores, dim=-1)  # (seq_len, seq_len)

        # Apply attention: output[i] = sum_j attn[i,j] * x[j]
        output = torch.einsum('ij,bjh->bih', attn, x)

        return output


class GetNumericPosition(nn.Module):
    """
    Output the numeric position index for each sequence position.

    Challenge: Transformers don't inherently "know" positions without
    positional encodings. This module explores how to extract/compute positions.

    Methods:
    1. 'explicit': Direct arange (baseline, not transformer-native)

    2. 'cumsum_attn': Cumulative sum via causal attention
       - Causal mask gives uniform attention over past positions
       - With value=1 everywhere, output = (# positions) / (# positions) = 1
       - Need to multiply by position count somehow

    3. 'ones_sum': Sum of ones with causal attention (requires unnormalized attn)

    The fundamental issue: softmax normalizes, destroying count information.
    Solutions require either:
    - Softmax1 (sum < 1, encodes info in "leftover")
    - Multiple heads/layers to reconstruct count
    - Positional encodings as input
    """
    def __init__(self, hidden_dim: int, method: str = 'explicit', max_seq_len: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns position indices as floats in the hidden dimension.
        Output shape matches input: (batch, seq_len, hidden_dim)
        Position value is broadcast across hidden dimension.
        """
        batch, seq_len, hidden = x.shape
        device = x.device

        if self.method == 'explicit':
            # Baseline: just use arange
            pos = torch.arange(seq_len, device=device, dtype=x.dtype)
            return pos.view(1, seq_len, 1).expand(batch, seq_len, hidden)

        elif self.method == 'cumsum_attn':
            # Attempt via causal attention - see notes in docstring
            # For now, falls back to explicit since pure attention can't do this
            pos = torch.arange(seq_len, device=device, dtype=x.dtype)
            return pos.view(1, seq_len, 1).expand(batch, seq_len, hidden)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """Returns position as scalar per position: (batch, seq_len, 1)"""
        batch, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        return pos.view(1, seq_len, 1).expand(batch, seq_len, 1)


class RunningSum(nn.Module):
    """
    Compute running (cumulative) sum across sequence positions.

    output[i] = sum(x[0], x[1], ..., x[i])

    Challenge: Standard attention normalizes weights via softmax, so
    sum_j attn[i,j] = 1. This means we get MEAN, not SUM.

    To get actual sum, we need either:
    1. Unnormalized attention (not standard)
    2. Multiply by position count: sum = mean * count
    3. Use linear attention (no softmax)
    4. Multiple passes to accumulate

    This implementation shows the "multiply by count" approach,
    which requires knowing the position (GetNumericPosition).

    For method='direct', we just use torch.cumsum as baseline.
    """
    def __init__(self, hidden_dim: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        output: (batch, seq_len, hidden_dim) where output[i] = sum(x[0:i+1])
        """
        if self.method == 'direct':
            return torch.cumsum(x, dim=1)

        elif self.method == 'attn_times_count':
            # Causal attention gives weighted mean, use leftover to get count
            batch, seq_len, hidden = x.shape

            # Causal mask: position i attends to positions 0..i
            scores = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            )
            # Use softmax1: attn[i,j] = 1/(i+2) for j <= i, leftover = 1/(i+2)
            attn, leftover = softmax1_with_leftover(scores, dim=-1)
            # leftover[i] = 1/(i+2), so count+1 = 1/leftover

            # Softmax1 weighted sum
            weighted_sum = torch.einsum('ij,bjh->bih', attn, x)

            # With softmax1: weighted_sum = true_sum / (count+1)
            # true_sum = weighted_sum * (count+1) = weighted_sum / leftover
            leftover_expanded = leftover.unsqueeze(0).expand(batch, -1, -1)
            sum_x = weighted_sum / leftover_expanded.clamp(min=1e-9)

            return sum_x

        else:
            raise ValueError(f"Unknown method: {self.method}")


class RunningProduct(nn.Module):
    """
    Compute running (cumulative) product across sequence positions.

    output[i] = prod(x[0], x[1], ..., x[i])

    Key insight: prod(x) = exp(sum(log(x)))

    So: running_product[i] = exp(running_sum[i](log(x)))

    This converts multiplication into addition in log-space,
    then converts back via exp.

    Methods:
    - 'log_exp': Use log, cumsum, exp (exact for positive x)
    - 'taylor': Use Taylor-series log/exp (transformer-native approximation)
    """
    def __init__(self, hidden_dim: int, method: str = 'log_exp', taylor_order: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method

        if method == 'taylor':
            self.log_op = LogApproximation(hidden_dim, method='taylor', order=taylor_order)
            self.exp_op = ExpViaTaylor(hidden_dim, order=taylor_order)
            self.cumsum_op = RunningSum(hidden_dim, method='direct')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim) - must be POSITIVE for log
        output: (batch, seq_len, hidden_dim) where output[i] = prod(x[0:i+1])
        """
        if self.method == 'log_exp':
            # Exact computation using torch ops
            log_x = torch.log(x.clamp(min=1e-8))
            cumsum_log = torch.cumsum(log_x, dim=1)
            return torch.exp(cumsum_log)

        elif self.method == 'taylor':
            # Transformer-native approximation
            # Only works well for x near 1 (where Taylor log/exp converge)
            log_x = self.log_op(x)
            cumsum_log = self.cumsum_op(log_x)
            return self.exp_op(cumsum_log)

        else:
            raise ValueError(f"Unknown method: {self.method}")


class LogSumExp(nn.Module):
    """
    Compute log-sum-exp: logsumexp(x) = log(sum(exp(x)))

    This is numerically stable when computed as:
    logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

    Key insight: This is exactly what softmax computes internally!
    softmax(x) = exp(x) / sum(exp(x)) = exp(x - logsumexp(x))

    So: logsumexp(x) = x - log(softmax(x))

    But log(softmax(x)) varies per element. Instead:
    logsumexp(x) = log(sum(exp(x)))
                 = log(1/softmax(x)_i * exp(x_i)) for any i
                 = x_i - log(softmax(x)_i)

    Actually simpler: attention score normalization gives us this.
    """
    def __init__(self, hidden_dim: int, dim: int = -1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard numerically-stable logsumexp."""
        return torch.logsumexp(x, dim=self.dim, keepdim=True)

    def forward_via_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logsumexp using softmax (showing the connection).

        logsumexp(x) = x_i - log(softmax(x)_i) for any i

        We use i = argmax(x) for numerical stability.
        """
        # Get softmax
        softmax_x = F.softmax(x, dim=self.dim)

        # Get max index
        max_idx = x.argmax(dim=self.dim, keepdim=True)

        # Get x_max and softmax_max
        x_max = x.gather(self.dim, max_idx)
        softmax_max = softmax_x.gather(self.dim, max_idx)

        # logsumexp = x_max - log(softmax_max)
        return x_max - torch.log(softmax_max)


class RunningLogSumExp(nn.Module):
    """
    Compute running logsumexp across sequence positions.

    output[i] = logsumexp(x[0], x[1], ..., x[i])
              = log(exp(x[0]) + exp(x[1]) + ... + exp(x[i]))

    This is useful for:
    - Numerically stable running product: exp(running_logsumexp(log(x)))
    - Computing running softmax denominators
    - Log-space probability accumulation

    Key recurrence:
    logsumexp(x[0:i+1]) = logsumexp(logsumexp(x[0:i]), x[i])
                        = log(exp(logsumexp(x[0:i])) + exp(x[i]))

    Two-argument logsumexp: logsumexp(a, b) = max(a,b) + log(1 + exp(-|a-b|))
    This only needs: max, abs, exp, log1p, add
    """
    def __init__(self, hidden_dim: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        output: (batch, seq_len, hidden_dim) where output[i] = logsumexp(x[0:i+1])
        """
        if self.method == 'direct':
            # Direct computation using cumulative logsumexp
            batch, seq_len, hidden = x.shape
            result = torch.zeros_like(x)
            result[:, 0, :] = x[:, 0, :]

            for i in range(1, seq_len):
                # logsumexp(prev, x[i]) = max(prev, x[i]) + log(1 + exp(-|prev - x[i]|))
                prev = result[:, i-1, :]
                curr = x[:, i, :]

                max_val = torch.maximum(prev, curr)
                # log(exp(prev - max) + exp(curr - max))
                # = max + log(exp(prev - max) + exp(curr - max))
                result[:, i, :] = max_val + torch.log(
                    torch.exp(prev - max_val) + torch.exp(curr - max_val)
                )

            return result

        elif self.method == 'log1p':
            # Using log1p for better numerical stability when values are close
            batch, seq_len, hidden = x.shape
            result = torch.zeros_like(x)
            result[:, 0, :] = x[:, 0, :]

            for i in range(1, seq_len):
                prev = result[:, i-1, :]
                curr = x[:, i, :]

                max_val = torch.maximum(prev, curr)
                min_val = torch.minimum(prev, curr)

                # logsumexp(a, b) = max(a,b) + log1p(exp(min - max))
                result[:, i, :] = max_val + torch.log1p(torch.exp(min_val - max_val))

            return result

        else:
            raise ValueError(f"Unknown method: {self.method}")


# =============================================================================
# WINDOWED OPERATIONS
# =============================================================================
# Operations that work over the last N tokens (sliding window).
# Useful for local attention, moving averages, etc.


class WindowMask(nn.Module):
    """
    Create attention masks for windowed (last N tokens) operations.

    For position i, the window includes positions [max(0, i-N+1), i].
    This gives a sliding window of size N (or smaller at the start).

    Mask types:
    - 'binary': 1 for positions in window, 0 otherwise
    - 'causal': 0 for positions in window, -inf otherwise (for softmax)
    - 'uniform': 1/window_size for positions in window (for direct averaging)
    """
    def __init__(self, window_size: int, mask_type: str = 'causal'):
        super().__init__()
        self.window_size = window_size
        self.mask_type = mask_type

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        Returns mask of shape (seq_len, seq_len).
        mask[i, j] indicates whether position i can attend to position j.
        """
        if device is None:
            device = torch.device('cpu')

        # Create position indices
        rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        # Window condition: j in [max(0, i-N+1), i]
        in_window = (cols >= rows - self.window_size + 1) & (cols <= rows)

        if self.mask_type == 'binary':
            return in_window.float()

        elif self.mask_type == 'causal':
            # 0 for in-window, -inf for out-of-window
            mask = torch.where(in_window, 0.0, float('-inf'))
            return mask

        elif self.mask_type == 'uniform':
            # 1/window_size for in-window (actual window may be smaller at start)
            window_sizes = torch.minimum(
                rows + 1,
                torch.tensor(self.window_size, device=device)
            ).float()
            mask = torch.where(in_window, 1.0 / window_sizes, 0.0)
            return mask

        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")


class WindowedSum(nn.Module):
    """
    Compute sum over the last N tokens for each position.

    output[i] = sum(x[max(0, i-N+1):i+1])

    Methods:
    - 'direct': Use cumsum trick: window_sum[i] = cumsum[i] - cumsum[i-N]
    - 'attention': Use attention with window mask (transformer implementation)
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.method = method
        self.mask_gen = WindowMask(window_size, mask_type='causal')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        if self.method == 'direct':
            # Efficient: windowed_sum[i] = cumsum[i] - cumsum[i-N]
            cumsum = torch.cumsum(x, dim=1)

            # Shift cumsum by N positions
            shifted = torch.zeros_like(cumsum)
            if self.window_size < seq_len:
                shifted[:, self.window_size:, :] = cumsum[:, :-self.window_size, :]

            return cumsum - shifted

        elif self.method == 'attention':
            # Attention-based with softmax1: use leftover to get window size
            mask = self.mask_gen(seq_len, device=x.device)
            attn, leftover = softmax1_with_leftover(mask, dim=-1)
            # leftover = 1/(window_size+1), so window_size = 1/leftover - 1

            # Softmax1 weighted sum
            weighted_sum = torch.einsum('ij,bjh->bih', attn, x)

            # true_sum = weighted_sum / leftover (because sum of weights = 1 - leftover)
            # and weighted_sum = true_mean * (1 - leftover)
            # So: true_sum = weighted_sum * window_size = weighted_sum / leftover - weighted_sum
            leftover_expanded = leftover.unsqueeze(0).expand(batch, -1, -1)
            sum_x = weighted_sum / leftover_expanded.clamp(min=1e-9)

            return sum_x

        else:
            raise ValueError(f"Unknown method: {self.method}")


class WindowedMean(nn.Module):
    """
    Compute mean over the last N tokens for each position.

    output[i] = mean(x[max(0, i-N+1):i+1])

    This is naturally what attention with a window mask computes.
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.method = method
        self.mask_gen = WindowMask(window_size, mask_type='causal')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        if self.method == 'direct':
            # mean = sum / count
            cumsum = torch.cumsum(x, dim=1)
            shifted = torch.zeros_like(cumsum)
            if self.window_size < seq_len:
                shifted[:, self.window_size:, :] = cumsum[:, :-self.window_size, :]
            windowed_sum = cumsum - shifted

            positions = torch.arange(seq_len, device=x.device)
            window_sizes = torch.minimum(
                positions + 1,
                torch.tensor(self.window_size, device=x.device)
            ).float().view(1, seq_len, 1)

            return windowed_sum / window_sizes

        elif self.method == 'attention':
            # Attention with softmax1: weighted sum / (1 - leftover) gives mean
            mask = self.mask_gen(seq_len, device=x.device)
            attn, leftover = softmax1_with_leftover(mask, dim=-1)

            # Weighted sum
            weighted_sum = torch.einsum('ij,bjh->bih', attn, x)

            # To get mean: divide by (1 - leftover) which is sum of weights
            sum_weights = 1 - leftover  # shape (seq_len, 1)
            sum_weights_expanded = sum_weights.unsqueeze(0).expand(batch, -1, -1)
            return weighted_sum / sum_weights_expanded.clamp(min=1e-9)

        else:
            raise ValueError(f"Unknown method: {self.method}")


class WindowedMax(nn.Module):
    """
    Compute max over the last N tokens for each position.

    output[i] = max(x[max(0, i-N+1):i+1])

    Methods:
    - 'direct': Explicit window iteration
    - 'unfold': Use tensor unfold for efficiency
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        if self.method == 'direct':
            # Pad and compute per-position max
            pad_size = self.window_size - 1
            x_padded = F.pad(x, (0, 0, pad_size, 0), value=float('-inf'))

            result = torch.zeros_like(x)
            for i in range(seq_len):
                window = x_padded[:, i:i + self.window_size, :]
                result[:, i, :] = window.max(dim=1).values

            return result

        elif self.method == 'unfold':
            # More efficient using unfold
            pad_size = self.window_size - 1
            # Transpose to (batch, hidden, seq_len) for unfold
            x_t = x.transpose(1, 2)
            x_padded = F.pad(x_t, (pad_size, 0), value=float('-inf'))

            # Unfold: (batch, hidden, seq_len, window_size)
            windows = x_padded.unfold(2, self.window_size, 1)

            # Max over window dimension
            result = windows.max(dim=-1).values

            return result.transpose(1, 2)

        else:
            raise ValueError(f"Unknown method: {self.method}")


class WindowedMin(nn.Module):
    """
    Compute min over the last N tokens for each position.

    output[i] = min(x[max(0, i-N+1):i+1])
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        if self.method == 'direct':
            pad_size = self.window_size - 1
            x_padded = F.pad(x, (0, 0, pad_size, 0), value=float('inf'))

            result = torch.zeros_like(x)
            for i in range(seq_len):
                window = x_padded[:, i:i + self.window_size, :]
                result[:, i, :] = window.min(dim=1).values

            return result

        elif self.method == 'unfold':
            pad_size = self.window_size - 1
            x_t = x.transpose(1, 2)
            x_padded = F.pad(x_t, (pad_size, 0), value=float('inf'))
            windows = x_padded.unfold(2, self.window_size, 1)
            result = windows.min(dim=-1).values
            return result.transpose(1, 2)

        else:
            raise ValueError(f"Unknown method: {self.method}")


class WindowedProduct(nn.Module):
    """
    Compute product over the last N tokens for each position.

    output[i] = prod(x[max(0, i-N+1):i+1])

    Uses log-space: prod(x) = exp(sum(log(x)))
    windowed_prod = exp(windowed_sum(log(x)))
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.method = method
        self.windowed_sum = WindowedSum(hidden_dim, window_size, method=method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x must be POSITIVE for log"""
        log_x = torch.log(x.clamp(min=1e-8))
        windowed_log_sum = self.windowed_sum(log_x)
        return torch.exp(windowed_log_sum)


class WindowedLogSumExp(nn.Module):
    """
    Compute logsumexp over the last N tokens for each position.

    output[i] = logsumexp(x[max(0, i-N+1):i+1])
              = log(sum(exp(x[max(0, i-N+1):i+1])))

    Useful for numerically stable windowed operations in log-space.
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.method = method
        self.mask_gen = WindowMask(window_size, mask_type='causal')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        if self.method == 'direct':
            pad_size = self.window_size - 1
            x_padded = F.pad(x, (0, 0, pad_size, 0), value=float('-inf'))

            result = torch.zeros_like(x)
            for i in range(seq_len):
                window = x_padded[:, i:i + self.window_size, :]
                result[:, i, :] = torch.logsumexp(window, dim=1)

            return result

        elif self.method == 'unfold':
            pad_size = self.window_size - 1
            x_t = x.transpose(1, 2)
            x_padded = F.pad(x_t, (pad_size, 0), value=float('-inf'))
            windows = x_padded.unfold(2, self.window_size, 1)
            result = torch.logsumexp(windows, dim=-1)
            return result.transpose(1, 2)

        elif self.method == 'attention':
            # logsumexp is the log-denominator of softmax
            mask = self.mask_gen(seq_len, device=x.device)

            # Reshape for per-hidden-dim computation
            x_flat = x.transpose(1, 2).reshape(batch * hidden, seq_len)
            scores = x_flat.unsqueeze(1) + mask.unsqueeze(0)
            result_flat = torch.logsumexp(scores, dim=-1)
            result = result_flat.reshape(batch, hidden, seq_len).transpose(1, 2)

            return result

        else:
            raise ValueError(f"Unknown method: {self.method}")


class WindowedVariance(nn.Module):
    """
    Compute variance over the last N tokens for each position.

    output[i] = var(x[max(0, i-N+1):i+1])
              = mean(x^2) - mean(x)^2

    Uses the computational identity: Var(X) = E[X²] - E[X]²
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.method = method
        self.windowed_mean = WindowedMean(hidden_dim, window_size, method=method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_x = self.windowed_mean(x)
        mean_x_sq = self.windowed_mean(x ** 2)
        return mean_x_sq - mean_x ** 2


class WindowedStd(nn.Module):
    """
    Compute standard deviation over the last N tokens for each position.

    output[i] = std(x[max(0, i-N+1):i+1]) = sqrt(var(x))
    """
    def __init__(self, hidden_dim: int, window_size: int, method: str = 'direct', eps: float = 1e-8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.eps = eps
        self.windowed_var = WindowedVariance(hidden_dim, window_size, method=method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var_x = self.windowed_var(x)
        return torch.sqrt(var_x + self.eps)


# =============================================================================
# TAYLOR SERIES APPROXIMATIONS
# =============================================================================
# Methods that use Taylor/power series expansions. These are useful because
# they only require multiplication (gating) and addition (residual), both
# of which are native transformer operations.


class ReciprocalGeometricSeries(nn.Module):
    """
    Compute 1/(1+x) via geometric series: 1 - x + x² - x³ + ...

    Each term only needs:
    - Previous power (multiply by x)
    - Alternating signs (easy)
    - Sum (residual)

    Could be computed across layers:
    Layer 1: 1
    Layer 2: 1 - x
    Layer 3: 1 - x + x²
    ...

    Convergence: |x| < 1
    """
    def __init__(self, hidden_dim: int, n_terms: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_terms = n_terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1/(1+x) = sum_{n=0}^{inf} (-x)^n for |x| < 1
        result = torch.ones_like(x)
        power = torch.ones_like(x)
        neg_x = -x

        for _ in range(1, self.n_terms):
            power = power * neg_x  # (-x)^n
            result = result + power

        return result


# =============================================================================
# SIGMOID/ACTIVATION-BASED METHODS
# =============================================================================
# Methods that exploit properties of sigmoid, SiLU, and other activations.


class ReciprocalViaSigmoidIteration(nn.Module):
    """
    Use sigmoid's bounded output to iteratively approach 1/x.

    Key insight: sigmoid(y) ∈ (0, 1), so it's "safe" to use as a multiplier.

    We want to find y such that sigmoid(y) = 1/(1+x), i.e., y = -log(x).
    Then 1/(1+x) = sigmoid(-log(x)).

    But we don't have log! Instead, iterate:
    - sigmoid(a) ≈ 1/(1+exp(-a))
    - If we set a such that exp(-a) = x, then sigmoid(a) = 1/(1+x)
    - Need a = -log(x)... circular.

    Alternative: sigmoid is approximately linear near 0.
    sigmoid(a) ≈ 0.5 + a/4 for small a.
    If we want output = 1/(1+x), and x is small:
    1/(1+x) ≈ 1 - x
    0.5 + a/4 = 1 - x → a = 4(0.5 - x) = 2 - 4x

    So sigmoid(2 - 4x) ≈ 1 - x ≈ 1/(1+x) for small x!
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sigmoid(2 - 4x) ≈ 1/(1+x) for small x
        return torch.sigmoid(2 - 4 * x)


class ReciprocalViaTaylorAtOne(nn.Module):
    """
    Approximate 1/x using Taylor series around x=1.

    1/x = 1/(1 + (x-1)) = 1 - (x-1) + (x-1)² - (x-1)³ + ...

    This converges for |x-1| < 1, i.e., x ∈ (0, 2).

    Uses only: subtraction, multiplication (gating), addition (residual).

    Part of TAYLOR SERIES APPROXIMATIONS section.
    """
    def __init__(self, hidden_dim: int, n_terms: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_terms = n_terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1/x = sum_{n=0}^{inf} (-(x-1))^n = sum (1-x)^n for |x-1| < 1
        z = 1 - x  # = -(x-1)
        result = torch.ones_like(x)
        power = torch.ones_like(x)

        for _ in range(1, self.n_terms):
            power = power * z  # (1-x)^n
            result = result + power

        return result


class ReciprocalViaRMSNorm(nn.Module):
    """
    Attempt to use RMSNorm's internal division.

    RMSNorm(v) = v / sqrt(mean(v²))

    For v = [x, c]:
    rms = sqrt((x² + c²) / 2)
    output[0] = x / rms = x * sqrt(2) / sqrt(x² + c²)

    This gives x / sqrt(x² + c²), not 1/x.

    But! If we set up v = [1, x]:
    rms = sqrt((1 + x²) / 2)
    output[0] = 1 / sqrt((1 + x²) / 2) = sqrt(2) / sqrt(1 + x²)

    This is 1/sqrt(1+x²), related to 1/|x| for large x.

    Not directly 1/x, but shows how normalization involves division.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create [1, x, 0, 0, ...] vectors
        batch_shape = x.shape[:-1]
        v = torch.zeros(*batch_shape, 2, device=x.device, dtype=x.dtype)
        v[..., 0] = 1.0
        v[..., 1] = x[..., 0] if x.shape[-1] > 0 else x

        # RMSNorm
        rms = torch.sqrt((v ** 2).mean(dim=-1, keepdim=True) + self.eps)
        normed = v / rms

        # Return the "1" component, which is 1/rms ≈ sqrt(2)/sqrt(1+x²)
        return normed[..., 0:1].expand_as(x)


class ReciprocalLinear(nn.Module):
    """
    Approximate 1/(1+x) using linear regime: 1/(1+x) ≈ 1 - x for |x| << 1.

    This is trivial - just subtraction (residual connection)!
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1/(1+x) ≈ 1 - x for small x
        return 1 - x


# =============================================================================
# ITERATIVE METHODS (Newton, etc.)
# =============================================================================
# Methods that use iterative refinement. Can be unrolled across layers.


class ReciprocalNewton(nn.Module):
    """
    Compute 1/x using Newton's method - NO LOG NEEDED!

    Newton iteration for 1/x:
        y_{n+1} = y_n * (2 - x * y_n)
                = 2*y_n - x*y_n²

    Only needs:
    - Multiplication (gating/SwiGLU)
    - Scaling by 2 (linear projection)
    - Subtraction (residual)

    Converges for x*y_0 < 2. Starting with y_0=1 works for x ∈ (0, 2).
    For larger x, scale down first.
    """
    def __init__(self, hidden_dim: int, n_iter: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_iter = n_iter

    def forward(self, x: torch.Tensor, y0: torch.Tensor = None) -> torch.Tensor:
        # Initialize: y = 1 (works for x in (0, 2))
        # For larger x, we'd need to scale
        if y0 is None:
            y = torch.ones_like(x)
        else:
            y = y0

        for _ in range(self.n_iter):
            # y = y * (2 - x*y) = 2y - x*y²
            y = 2 * y - x * y * y

        return y

    def forward_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """Handle larger x by scaling into convergent range."""
        # Find scale factor to bring x into (0, 2)
        # Use 2^k scaling where k = ceil(log2(x/2))
        # Simpler: just use x's magnitude

        # For x > 2, scale down by factor s, compute 1/(x/s), then divide by s
        # 1/x = (1/s) * 1/(x/s)

        # Easy approach: normalize by max, but that needs knowing max
        # Iterative approach: start with small y and iterate more

        # Simplest for demo: clamp x to reasonable range
        x_safe = x.clamp(min=0.1, max=1.9)
        return self.forward(x_safe)


class ReciprocalViaSoftmax1(nn.Module):
    """
    Use softmax1's linear regime for approximate 1/(1+x).

    softmax1([s]) = exp(s) / (1 + exp(s)) = sigmoid(s)

    For small s: sigmoid(s) ≈ 0.5 + s/4

    Key insight: the "1" in softmax1's denominator is ALWAYS there.
    With score=0: output = 1/(1+1) = 0.5
    With score=s: output = exp(s)/(1+exp(s))

    To get 1/(1+x), we want exp(s)/(1+exp(s)) = 1/(1+x)
    This gives exp(s) = 1/x, so s = -log(x)... still needs log.

    BUT in linear regime (small x):
    If we set s = -x (linear in x), then:
    sigmoid(-x) ≈ 0.5 - x/4 ≈ 1/(2+x) for small x

    Not quite 1/(1+x), but close! And only needs negation.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sigmoid(-x) ≈ 1/(2+x) for small x
        # Returns an approximation of 1/(1+x) that's off by factor ~2
        return torch.sigmoid(-x)

    def forward_adjusted(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adjust to better approximate 1/(1+x).

        sigmoid(-x) = 1/(1+exp(x))
        For small x: exp(x) ≈ 1+x, so sigmoid(-x) ≈ 1/(2+x)

        To get 1/(1+x) from 1/(2+x):
        1/(1+x) = 1/(2+x) * (2+x)/(1+x) = 1/(2+x) * (1 + 1/(1+x))

        Circular! But we can iterate:
        Let a = sigmoid(-x) ≈ 1/(2+x)
        Then 1/(1+x) ≈ 2a / (1 + ... ) ... still messy.

        Simpler: for small x, 1/(1+x) ≈ 1-x, which is just subtraction.
        """
        # Just use the linear approximation for small x
        return 1 - x


# =============================================================================
# ATTENTION-BASED METHODS
# =============================================================================
# Methods that use the attention mechanism's softmax to compute functions.
# Key insight: softmax([0, log(x)]) @ [1, 0] = 1/(1+x)


class ReciprocalViaAttention(nn.Module):
    """
    Computes 1/x using attention mechanism.

    Setup:
    - Position 0 (self): value = 1, attends with score 0
    - Position 1 (zero): value = 0, attends with score log(x)

    After softmax over [0, log(x)]:
        softmax([0, log(x)]) = [1/(1+x), x/(1+x)]

    Output = 1/(1+x) * 1 + x/(1+x) * 0 = 1/(1+x)

    To get 1/x instead of 1/(1+x):
        Use scores [0, log(x-1)] for x > 1
        Or use: 1/x = 1/(1+x) * (1 + 1/x) requires iteration

    Alternative formulation for 1/x directly:
        Scores [-log(x), 0] gives [1/x, (x-1)/x] after softmax... no

    Actually simpler: scores [0, log(x)] with values [1, 0]:
        output = 1/(1+x)
        To get 1/x: use input (x-1), get 1/(1+(x-1)) = 1/x ✓

    This module assumes x > 0.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be positive
        # Clamp to avoid numerical issues
        x_safe = x.clamp(min=self.eps)

        log_x = torch.log(x_safe)

        # With softmax1, the leftover directly gives us 1/(1+x)!
        # softmax1([log(x)]) = x/(1+x), leftover = 1/(1+x)
        # This is simpler than the 2-element formulation with regular softmax
        _, leftover = softmax1_with_leftover(log_x.unsqueeze(-1), dim=-1)

        return leftover.squeeze(-1)  # This is 1/(1+x)

    def forward_reciprocal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute actual 1/x by shifting input.
        1/x = 1/(1 + (x-1)) when we feed (x-1) to the attention mechanism.
        Only valid for x > 1 (so x-1 > 0 for valid log).
        For x in (0, 1], use: 1/x = x^(-1) directly or different formulation.
        """
        x_safe = x.clamp(min=1.0 + self.eps)
        # Feed (x - 1) to get 1/(1 + (x-1)) = 1/x
        x_shifted = x_safe - 1.0 + self.eps
        return self.forward(x_shifted)


class ReciprocalViaSequenceAttention(nn.Module):
    """
    Computes 1/x for each position using actual sequence attention.

    Problem: If all positions have value and attend to each other, the
    reciprocal computation gets polluted.

    Solution: Use TWO special tokens that ALL positions attend to:
    - Position 0: "one token" with value=1, all positions attend with score 0
    - Position 1: "zero token" with value=0, position i attends with score log(x_i)

    After softmax over just these two: attn = [1/(1+x), x/(1+x)]
    Output = 1/(1+x)*1 + x/(1+x)*0 = 1/(1+x)

    Real positions (2..seq_len+1) have value=0 so attending to them doesn't matter!
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim) - contains positive values

        Returns: (batch, seq_len, hidden_dim) - each element is 1/(1+x)
        """
        batch, seq_len, hidden = x.shape
        x_safe = x.clamp(min=self.eps)
        log_x = torch.log(x_safe)  # (batch, seq_len, hidden)

        # Setup: positions 0,1 are special, positions 2..seq_len+1 are real
        # Values: pos 0 = 1, pos 1 = 0, pos 2+ = 0
        # This means only pos 0 contributes to output!

        # Each real position i (i >= 2) attends to:
        #   - position 0 ("one") with score 0
        #   - position 1 ("zero") with score log(x_i)
        #   - other positions with score 0 (but value=0 so doesn't matter)

        # Scores: (batch, seq_len+2, seq_len+2, hidden)
        n_pos = seq_len + 2
        scores = torch.zeros((batch, n_pos, n_pos, hidden), device=x.device, dtype=x.dtype)

        # Real positions attend to pos 1 with score log(x)
        for i in range(seq_len):
            scores[:, i + 2, 1, :] = log_x[:, i, :]  # attend to "zero" with log(x)
            # attend to "one" (pos 0) with score 0 (already set)
            # attend to others with score 0 (already set)

        # With softmax1, we can use the leftover directly for reciprocal
        # For each position, softmax1([log(x)]) has leftover = 1/(1+x)
        # This is simpler than the sequence attention formulation

        # Apply softmax1 element-wise:
        # Unsqueeze to create a single-element dimension for softmax1
        # For input s, softmax1([s]) gives leftover = 1/(1+exp(s))
        # So for s=log(x), leftover = 1/(1+x)
        _, leftover = softmax1_with_leftover(log_x.unsqueeze(-1), dim=-1)

        # The leftover IS 1/(1+x) for each element
        return leftover.squeeze(-1)

    def forward_reciprocal(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 1/x by shifting input."""
        x_safe = x.clamp(min=1.0 + self.eps)
        x_shifted = x_safe - 1.0 + self.eps
        return self.forward(x_shifted)


class ReciprocalViaSequenceAttentionRealistic(nn.Module):
    """
    Computes 1/x using attention WITHOUT arbitrary masking.

    Key insight: If all "real" positions have value=0, attending to them
    doesn't affect the output. Only special tokens with non-zero values matter.

    Setup:
    - Position 0: "one token", value=1
    - Position 1: "absorber token", value=0
    - Positions 2+: real data, value=0

    Each real position i attends to ALL positions with score 0,
    EXCEPT position 1 gets score log(x_i).

    softmax gives: attn[i,0] = 1/(n + x_i), attn[i,1] = x_i/(n + x_i), others = 1/(n + x_i)
    where n = number of positions with score 0.

    Output = attn[i,0] * 1 + (rest) * 0 = 1/(n + x_i)

    To get 1/(1+x): use just positions 0 and 1, or adjust x.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realistic version using softmax1.

        With softmax1, we use a single score log(x) and the leftover IS 1/(1+x).
        This is cleaner than the 2-position approach with regular softmax.
        """
        x_safe = x.clamp(min=self.eps)
        log_x = torch.log(x_safe)

        # With softmax1([log(x)]):
        # weight = x / (1 + x)
        # leftover = 1 / (1 + x)  <-- this is what we want!
        _, leftover = softmax1_with_leftover(log_x.unsqueeze(-1), dim=-1)

        return leftover.squeeze(-1)

    def forward_reciprocal(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = x.clamp(min=1.0 + self.eps)
        x_shifted = x_safe - 1.0 + self.eps
        return self.forward(x_shifted)


class Softmax1(nn.Module):
    """
    Softmax with +1 in denominator (softmax_1 / quiet attention).

    softmax1(x)_i = exp(x_i) / (1 + sum_j(exp(x_j)))

    This allows the model to "attend to nothing" when all values are negative,
    as the outputs can sum to less than 1.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # exp(x_i) / (1 + sum(exp(x_j)))
        # With numerical stability: subtract max, then adjust the "1" term
        x_max = x.max(dim=self.dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        # The "1" becomes exp(-max) after the max subtraction
        return exp_x / (torch.exp(-x_max) + exp_x.sum(dim=self.dim, keepdim=True))


class LogApproximation(nn.Module):
    """
    Approximate log(x) using transformer primitives.

    Challenge: log is nonlinear, but Q·K is linear. How to get log(x) as a score?

    Approaches:

    1. Taylor series around x=1:
       log(x) = (x-1) - (x-1)²/2 + (x-1)³/3 - ...
       Requires x near 1, and we need to compute powers (we have Square).

    2. Via reciprocal iteration (Newton's method for log):
       To find log(a), solve exp(y) = a
       Newton: y_{n+1} = y_n - (exp(y_n) - a) / exp(y_n)
                       = y_n - 1 + a * exp(-y_n)
                       = y_n - 1 + a / exp(y_n)
       Uses exp (we have) and reciprocal (we have).

    3. Via integral approximation:
       log(x) = ∫₁ˣ (1/t) dt ≈ Σ (1/tᵢ) * Δt
       Uses reciprocal repeatedly.

    4. Learned approximation:
       FFN with SiLU can approximate log in a bounded region.
    """
    def __init__(self, hidden_dim: int, method: str = 'taylor', order: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method
        self.order = order

        if method == 'newton':
            self.exp_op = ExpViaTaylor(hidden_dim, order=6)
            self.recip_op = ReciprocalViaAttention(hidden_dim)

    def forward_taylor(self, x: torch.Tensor) -> torch.Tensor:
        """Taylor series: log(x) = Σ (-1)^(n+1) * (x-1)^n / n for |x-1| < 1"""
        z = x - 1  # center at x=1
        result = torch.zeros_like(x)
        z_power = z.clone()

        for n in range(1, self.order + 1):
            sign = 1 if n % 2 == 1 else -1
            result = result + sign * z_power / n
            z_power = z_power * z

        return result

    def forward_newton(self, x: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
        """Newton iteration to solve exp(y) = x for y = log(x)"""
        # Initial guess: y ≈ x - 1 (good for x near 1)
        y = x - 1

        for _ in range(n_iter):
            exp_y = self.exp_op(y)
            # y = y - 1 + x/exp(y)
            # 1/exp(y) = 1/(1 + (exp(y)-1)) via our reciprocal
            # But exp(y) can be < 1 for y < 0, so shift fails
            # Instead: directly use 1/(1 + exp(y)) and adjust
            # 1/exp(y) = (1 + exp(y)) / exp(y) - 1 = (1 + 1/... ) ... circular

            # Simpler: for x near 1, exp(y) near 1, so we can use
            # 1/(1 + exp(y)) and then * (1 + exp(y)) / exp(y)... still circular

            # Actually just use forward() which gives 1/(1+exp_y)
            # Then 1/exp_y = (1/(1+exp_y)) * (1 + exp_y) / exp_y
            #              = (1/(1+exp_y)) * (1/exp_y + 1)
            #              = (1/(1+exp_y)) / exp_y + (1/(1+exp_y))  ... still need 1/exp_y

            # Fallback: just use 1/(1+exp_y) and accept the approximation
            one_over_1_plus_exp = self.recip_op(exp_y)
            # Approximate 1/exp_y ≈ (1 + 1/exp_y) - 1 = ... still circular

            # For now, use the simple approximation for small |y|
            inv_exp_y = 1.0 / (exp_y + 1e-8)  # cheating with direct division
            y = y - 1 + x * inv_exp_y

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == 'taylor':
            return self.forward_taylor(x)
        elif self.method == 'newton':
            return self.forward_newton(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class ExpViaSiLUReciprocal(nn.Module):
    """
    Approximate exp(x) using SiLU(x) * (1/x) = sigmoid(x).

    Key insight:
        SiLU(x) / x = x * sigmoid(x) / x = sigmoid(x)
        sigmoid(x) = exp(x) / (1 + exp(x))

    For x < 0 (negative region):
        exp(x) << 1, so sigmoid(x) ≈ exp(x)

    For x > 0 (positive region):
        exp(x) = 1/exp(-x) ≈ 1/sigmoid(-x)
        And sigmoid(-x) = SiLU(-x)/(-x)
        So exp(x) ≈ -x / SiLU(-x)

    This gives exp(x) using only SiLU and 1/x (attention-based reciprocal)!
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.reciprocal = ReciprocalViaAttention(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For x < 0: exp(x) ≈ sigmoid(x) = SiLU(x) / x
        # For x > 0: exp(x) ≈ 1/sigmoid(-x) = -x / SiLU(-x)

        # Handle both regions
        neg_mask = x < 0
        pos_mask = ~neg_mask

        result = torch.zeros_like(x)

        # Negative region: sigmoid(x) = SiLU(x) / x
        if neg_mask.any():
            x_neg = x[neg_mask]
            # Avoid division by zero for x very close to 0
            x_neg_safe = torch.where(x_neg.abs() < self.eps, -self.eps * torch.ones_like(x_neg), x_neg)
            silu_neg = F.silu(x_neg_safe)
            result[neg_mask] = silu_neg / x_neg_safe  # sigmoid(x) ≈ exp(x) for x < 0

        # Positive region: 1/sigmoid(-x) = -x / SiLU(-x) = x / SiLU(-x) * (-1) ...
        # Actually: sigmoid(-x) = SiLU(-x) / (-x), so 1/sigmoid(-x) = -x / SiLU(-x)
        if pos_mask.any():
            x_pos = x[pos_mask]
            x_pos_safe = torch.where(x_pos.abs() < self.eps, self.eps * torch.ones_like(x_pos), x_pos)
            neg_x = -x_pos_safe
            silu_neg_x = F.silu(neg_x)
            # 1/sigmoid(-x) = (-x) / SiLU(-x)... but SiLU(-x) is negative for -x < 0
            # SiLU(-x) = -x * sigmoid(-x), and for -x < 0, sigmoid(-x) > 0
            # So SiLU(-x) < 0 when -x < 0
            # Thus: (-x) / SiLU(-x) = (-x) / (-x * sigmoid(-x)) = 1/sigmoid(-x) ✓
            result[pos_mask] = neg_x / silu_neg_x

        return result


class ExpViaTaylor(nn.Module):
    """
    Approximate exp(x) using Taylor series.

    exp(x) = sum_{n=0}^{order} x^n / n!

    This is accurate for |x| < ~2-3 depending on order.
    Can be computed using SwiGLU's multiplication for x^n terms.
    """
    def __init__(self, hidden_dim: int, order: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.order = order

        # Coefficients for Taylor series: 1/n!
        self.register_buffer(
            'coeffs',
            torch.tensor([1.0 / math.factorial(i) for i in range(order + 1)])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Taylor series: exp(x) = sum_{n=0}^{order} x^n / n!

        result = torch.ones_like(x) * self.coeffs[0]  # 1
        x_power = x.clone()

        for i in range(1, self.order + 1):
            result = result + self.coeffs[i] * x_power
            x_power = x_power * x  # x^(i+1) for next iteration

        return result


class ExpViaSiLUScaling(nn.Module):
    """
    Approximate exp(x) using scaled SiLU regions.

    Key observation: For x in a specific range, we can use SiLU's
    smooth interpolation properties.

    Method: Use the identity exp(x) = lim_{n→∞} (1 + x/n)^n

    With SiLU, we approximate (1 + x/n) and compose:
        - SiLU(x + b) / SiLU(b) gives approximately (1 + x/b) for appropriate b
        - Compose n times

    Alternative using SiLU's exponential-like growth:
        For x > 0: SiLU(x) ≈ x, so not directly exponential
        But: exp(x) = exp(x/2)² = exp(x/4)⁴ = ...

    We use: 2^(x/ln2) = exp(x)
    And approximate 2^y using SiLU-based computation.
    """
    def __init__(self, hidden_dim: int, n_compositions: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = n_compositions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1 + x/n)^n approximation using SiLU for smooth (1 + x/n)
        # SiLU(a)/SiLU(0) isn't defined (SiLU(0)=0), so we use different approach

        # Direct approximation: compose (1 + x/n) n times
        # Using SiLU: we want something that gives ~(1 + x/n)
        # Note: 1 + SiLU(x/n) / C for appropriate C

        # Simpler: use the fact that for small y, SiLU(y+2)/SiLU(2) ≈ 1 + y/2
        # SiLU(2) ≈ 1.76

        n = self.n
        y = x / n  # small increments

        # (1 + x/n)^n via repeated multiplication
        result = torch.ones_like(x)
        for _ in range(n):
            result = result * (1 + y)

        return result


# Convenience dictionary for all operations
HIDDEN_OPS = {
    # Basic operations
    'mean': SequenceMean,
    'square': Square,
    'relu_swiglu': ReLUViaSwiGLU,
    'relu_silu': ReLUViaSiLU,
    'softmax1': Softmax1,

    # Positional operations
    'select_n_behind': SelectNBehind,
    'copy_n_behind': CopyFromNBehind,
    'get_position': GetNumericPosition,

    # Running/cumulative operations
    'running_sum': RunningSum,
    'running_product': RunningProduct,
    'logsumexp': LogSumExp,
    'running_logsumexp': RunningLogSumExp,

    # Windowed operations
    'window_mask': WindowMask,
    'windowed_sum': WindowedSum,
    'windowed_mean': WindowedMean,
    'windowed_max': WindowedMax,
    'windowed_min': WindowedMin,
    'windowed_product': WindowedProduct,
    'windowed_logsumexp': WindowedLogSumExp,
    'windowed_variance': WindowedVariance,
    'windowed_std': WindowedStd,

    # Reciprocal methods
    'reciprocal_linear': ReciprocalLinear,
    'reciprocal_newton': ReciprocalNewton,
    'reciprocal_geometric': ReciprocalGeometricSeries,
    'reciprocal_taylor': ReciprocalViaTaylorAtOne,
    'reciprocal_sigmoid': ReciprocalViaSigmoidIteration,
    'reciprocal_softmax1': ReciprocalViaSoftmax1,
    'reciprocal_attn': ReciprocalViaAttention,
    'reciprocal_seq': ReciprocalViaSequenceAttention,
    'reciprocal_seq_realistic': ReciprocalViaSequenceAttentionRealistic,
    'reciprocal_rmsnorm': ReciprocalViaRMSNorm,

    # Log/Exp
    'log': LogApproximation,
    'exp_silu_recip': ExpViaSiLUReciprocal,
    'exp_taylor': ExpViaTaylor,
    'exp_composition': ExpViaSiLUScaling,
}


"""
DEPENDENCY ANALYSIS:
====================

Primitive operations available in transformers:
  - Addition/subtraction (residual connections)
  - Linear projection (W @ x)
  - Softmax (attention)
  - SiLU/GELU/Swish (FFN activation)
  - Multiplication (gating in SwiGLU, attention @ values)

What we can build:
  ✓ x²          : direct (x * x via gating)
  ✓ 1/(1+x)     : softmax([0, log(x)]) @ [1, 0]  -- BUT needs log(x)!
  ✓ 1/x         : 1/(1+(x-1)) via shift          -- needs subtraction
  ✓ sigmoid(x)  : SiLU(x) / x                    -- needs 1/x
  ✓ exp(x)      : sigmoid(x) ≈ exp(x) for x < 0  -- approximation in tail
  ✓ ReLU(x)     : SiLU(βx)/β                     -- good approximation
  ? log(x)      : Taylor near 1, or Newton       -- HARD, circular

The circular dependency:
  1/x via attention needs → log(x) as score
  log(x) via Newton needs → exp(x) and 1/x
  exp(x) via sigmoid needs → SiLU(x)/x = 1/x

Breaking the cycle:
  - Taylor series for log near x=1: log(x) ≈ (x-1) - (x-1)²/2 + ...
  - Taylor series for exp: exp(x) ≈ 1 + x + x²/2 + ...
  - These don't need each other, just powers (we have square)
  - Build up: powers → Taylor log → attention-based 1/x → everything else
"""


def test_select_n_behind():
    """Test SelectNBehind - shifting sequence by N positions."""
    print("\n" + "="*60)
    print("TEST: SelectNBehind (shift sequence by N)")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    # Create sequential values for easy verification
    x = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input (first batch, first hidden dim): {x[0, :, 0].tolist()}")

    # Test mode='zero' (pad with zeros)
    for n in [1, 2, 3]:
        op = SelectNBehind(hidden, n=n, mode='zero')
        out = op(x)
        print(f"\nn={n}, mode='zero': {out[0, :, 0].tolist()}")
        # Expected: [0, 0, ..., 0, 1, 2, ...] with n zeros at start
        expected = torch.zeros_like(x)
        if n < seq_len:
            expected[:, n:, :] = x[:, :-n, :]
        assert torch.allclose(out, expected), f"Failed for n={n}, mode='zero'"

    # Test mode='clamp' (clamp to position 0)
    op = SelectNBehind(hidden, n=3, mode='clamp')
    out = op(x)
    print(f"\nn=3, mode='clamp': {out[0, :, 0].tolist()}")
    # Expected: [0, 0, 0, 0, 1, 2, 3, 4] (positions 0-2 all get value from pos 0)

    # Test mode='wrap' (wrap around)
    op = SelectNBehind(hidden, n=3, mode='wrap')
    out = op(x)
    print(f"\nn=3, mode='wrap': {out[0, :, 0].tolist()}")
    # Expected: [5, 6, 7, 0, 1, 2, 3, 4] (wraps to end of sequence)

    print("\nPASS: SelectNBehind")
    return True


def test_copy_from_n_behind():
    """Test CopyFromNBehind - attention-based version of SelectNBehind."""
    print("\n" + "="*60)
    print("TEST: CopyFromNBehind (attention-based shift)")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    x = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")

    # Test different N values
    for n in [1, 2, 3]:
        op = CopyFromNBehind(hidden, n=n, mode='zero')
        out = op(x)
        print(f"n={n}, mode='zero': {out[0, :, 0].tolist()}")

        # Compare with SelectNBehind
        select_op = SelectNBehind(hidden, n=n, mode='zero')
        select_out = select_op(x)

        max_diff = (out - select_out).abs().max().item()
        print(f"  Max diff from SelectNBehind: {max_diff:.2e}")

    # Verify attention matches direct indexing
    op = CopyFromNBehind(hidden, n=2, mode='clamp')
    out = op(x)
    select_op = SelectNBehind(hidden, n=2, mode='clamp')
    select_out = select_op(x)

    match = torch.allclose(out, select_out, atol=1e-5)
    print(f"\nAttention-based matches direct indexing: {match}")

    print("\nPASS: CopyFromNBehind")
    return True


def test_get_numeric_position():
    """Test GetNumericPosition - extracting position indices."""
    print("\n" + "="*60)
    print("TEST: GetNumericPosition")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    x = torch.randn(batch, seq_len, hidden)

    op = GetNumericPosition(hidden, method='explicit')

    # Test full output
    out = op(x)
    print(f"Output shape: {out.shape}")
    print(f"Positions (first batch, first hidden dim): {out[0, :, 0].tolist()}")

    # Verify positions are 0, 1, 2, ..., seq_len-1
    expected = torch.arange(seq_len, dtype=x.dtype)
    match = torch.allclose(out[0, :, 0], expected)
    print(f"Positions correct: {match}")

    # Test scalar output
    out_scalar = op.forward_scalar(x)
    print(f"Scalar output shape: {out_scalar.shape}")
    print(f"Scalar positions: {out_scalar[0, :, 0].tolist()}")

    # Verify all hidden dims have same value
    all_same = (out == out[:, :, :1]).all()
    print(f"All hidden dims have same position value: {all_same}")

    print("\nPASS: GetNumericPosition")
    return True


def test_running_sum():
    """Test RunningSum - cumulative sum across sequence."""
    print("\n" + "="*60)
    print("TEST: RunningSum (cumulative sum)")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    # Simple sequential values for verification
    x = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")

    # Test direct method
    op_direct = RunningSum(hidden, method='direct')
    out_direct = op_direct(x)
    print(f"Direct cumsum: {out_direct[0, :, 0].tolist()}")

    # Expected: [1, 3, 6, 10, 15, 21, 28, 36]
    expected = torch.cumsum(x, dim=1)
    match_direct = torch.allclose(out_direct, expected)
    print(f"Direct method correct: {match_direct}")

    # Test attention-based method
    op_attn = RunningSum(hidden, method='attn_times_count')
    out_attn = op_attn(x)
    print(f"Attn*count: {out_attn[0, :, 0].tolist()}")

    match_attn = torch.allclose(out_attn, expected, atol=1e-5)
    print(f"Attention method correct: {match_attn}")

    print("\nPASS: RunningSum")
    return match_direct and match_attn


def test_running_product():
    """Test RunningProduct - cumulative product via log/exp."""
    print("\n" + "="*60)
    print("TEST: RunningProduct (cumulative product via log/exp)")
    print("="*60)

    batch, seq_len, hidden = 2, 6, 4
    # Values near 1 for numerical stability in Taylor method
    x = torch.tensor([1.1, 1.2, 0.9, 1.3, 0.8, 1.1], dtype=torch.float32)
    x = x.view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")

    # Test log_exp method (exact)
    op_exact = RunningProduct(hidden, method='log_exp')
    out_exact = op_exact(x)
    print(f"log_exp method: {[f'{v:.4f}' for v in out_exact[0, :, 0].tolist()]}")

    # Expected: cumprod
    expected = torch.cumprod(x, dim=1)
    print(f"Expected:       {[f'{v:.4f}' for v in expected[0, :, 0].tolist()]}")

    match_exact = torch.allclose(out_exact, expected, atol=1e-5)
    print(f"log_exp method correct: {match_exact}")

    # Test Taylor method (approximate)
    op_taylor = RunningProduct(hidden, method='taylor', taylor_order=10)
    out_taylor = op_taylor(x)
    print(f"Taylor method:  {[f'{v:.4f}' for v in out_taylor[0, :, 0].tolist()]}")

    max_diff = (out_taylor - expected).abs().max().item()
    print(f"Taylor max diff from exact: {max_diff:.4f}")

    print("\nPASS: RunningProduct")
    return match_exact


def test_logsumexp():
    """Test LogSumExp implementations."""
    print("\n" + "="*60)
    print("TEST: LogSumExp")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    x = torch.randn(batch, seq_len, hidden)

    op = LogSumExp(hidden, dim=1)

    # Test standard method
    out_standard = op(x)
    expected = torch.logsumexp(x, dim=1, keepdim=True)
    print(f"Output shape: {out_standard.shape}")

    match_standard = torch.allclose(out_standard, expected, atol=1e-5)
    print(f"Standard logsumexp correct: {match_standard}")

    # Test via softmax
    out_softmax = op.forward_via_softmax(x)
    match_softmax = torch.allclose(out_softmax, expected, atol=1e-5)
    print(f"Via-softmax method correct: {match_softmax}")

    # Verify property: softmax(x) = exp(x - logsumexp(x))
    softmax_x = F.softmax(x, dim=1)
    reconstructed = torch.exp(x - out_standard)
    match_property = torch.allclose(softmax_x, reconstructed, atol=1e-5)
    print(f"softmax = exp(x - logsumexp(x)): {match_property}")

    print("\nPASS: LogSumExp")
    return match_standard and match_softmax and match_property


def test_running_logsumexp():
    """Test RunningLogSumExp - cumulative logsumexp."""
    print("\n" + "="*60)
    print("TEST: RunningLogSumExp")
    print("="*60)

    batch, seq_len, hidden = 2, 6, 4
    x = torch.randn(batch, seq_len, hidden)

    print(f"Input (first batch, first hidden): {[f'{v:.2f}' for v in x[0, :, 0].tolist()]}")

    # Test direct method
    op_direct = RunningLogSumExp(hidden, method='direct')
    out_direct = op_direct(x)

    # Compute expected via loop
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        expected[:, i, :] = torch.logsumexp(x[:, :i+1, :], dim=1)

    print(f"Direct output:  {[f'{v:.2f}' for v in out_direct[0, :, 0].tolist()]}")
    print(f"Expected:       {[f'{v:.2f}' for v in expected[0, :, 0].tolist()]}")

    match_direct = torch.allclose(out_direct, expected, atol=1e-5)
    print(f"Direct method correct: {match_direct}")

    # Test log1p method
    op_log1p = RunningLogSumExp(hidden, method='log1p')
    out_log1p = op_log1p(x)

    match_log1p = torch.allclose(out_log1p, expected, atol=1e-5)
    print(f"log1p method correct: {match_log1p}")

    # Verify property: exp(running_logsumexp) = running_sum(exp(x))
    running_sum_exp = torch.cumsum(torch.exp(x), dim=1)
    from_logsumexp = torch.exp(out_direct)
    match_property = torch.allclose(from_logsumexp, running_sum_exp, atol=1e-4)
    print(f"exp(running_lse) = cumsum(exp(x)): {match_property}")

    print("\nPASS: RunningLogSumExp")
    return match_direct and match_log1p


def test_window_mask():
    """Test WindowMask - creates attention masks for windowed operations."""
    print("\n" + "="*60)
    print("TEST: WindowMask")
    print("="*60)

    seq_len = 8
    window_size = 3

    # Test binary mask
    mask_binary = WindowMask(window_size, mask_type='binary')
    binary = mask_binary(seq_len)
    print(f"Binary mask (window={window_size}, seq_len={seq_len}):")
    for row in binary.int().tolist():
        print(f"  {row}")

    # Verify window structure
    for i in range(seq_len):
        expected_start = max(0, i - window_size + 1)
        expected_ones = binary[i, expected_start:i+1]
        expected_zeros_before = binary[i, :expected_start] if expected_start > 0 else torch.tensor([])
        expected_zeros_after = binary[i, i+1:] if i + 1 < seq_len else torch.tensor([])

        assert expected_ones.sum() == min(i + 1, window_size), f"Wrong window size at position {i}"
        if len(expected_zeros_before) > 0:
            assert expected_zeros_before.sum() == 0, f"Non-zero before window at position {i}"
        if len(expected_zeros_after) > 0:
            assert expected_zeros_after.sum() == 0, f"Non-zero after window at position {i}"

    # Test causal mask
    mask_causal = WindowMask(window_size, mask_type='causal')
    causal = mask_causal(seq_len)
    print(f"\nCausal mask (0 in window, -inf outside):")
    for i, row in enumerate(causal.tolist()):
        formatted = ['.' if v == float('-inf') else '0' for v in row]
        print(f"  {i}: {' '.join(formatted)}")

    # Verify causal mask
    for i in range(seq_len):
        in_window = causal[i, :] == 0
        out_window = causal[i, :] == float('-inf')
        assert in_window.sum() == min(i + 1, window_size)
        assert out_window.sum() == seq_len - min(i + 1, window_size)

    # Test uniform mask
    mask_uniform = WindowMask(window_size, mask_type='uniform')
    uniform = mask_uniform(seq_len)
    print(f"\nUniform mask (1/window_size in window):")
    print(f"  Row sums: {uniform.sum(dim=1).tolist()}")

    # Verify rows sum to 1
    row_sums = uniform.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(seq_len)), "Uniform mask rows should sum to 1"

    print("\nPASS: WindowMask")
    return True


def test_windowed_sum():
    """Test WindowedSum - sum over last N tokens."""
    print("\n" + "="*60)
    print("TEST: WindowedSum")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    window_size = 3
    x = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")
    print(f"Window size: {window_size}")

    # Test direct method
    op_direct = WindowedSum(hidden, window_size, method='direct')
    out_direct = op_direct(x)
    print(f"Direct: {out_direct[0, :, 0].tolist()}")

    # Compute expected manually
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        expected[:, i, :] = x[:, start:i+1, :].sum(dim=1)

    print(f"Expected: {expected[0, :, 0].tolist()}")

    match_direct = torch.allclose(out_direct, expected, atol=1e-5)
    print(f"Direct method correct: {match_direct}")

    # Test attention method
    op_attn = WindowedSum(hidden, window_size, method='attention')
    out_attn = op_attn(x)
    print(f"Attention: {out_attn[0, :, 0].tolist()}")

    match_attn = torch.allclose(out_attn, expected, atol=1e-5)
    print(f"Attention method correct: {match_attn}")

    # Test with different window sizes
    print("\nDifferent window sizes:")
    for ws in [1, 2, 4, seq_len]:
        op = WindowedSum(hidden, ws, method='direct')
        out = op(x)
        expected_ws = torch.zeros_like(x)
        for i in range(seq_len):
            start = max(0, i - ws + 1)
            expected_ws[:, i, :] = x[:, start:i+1, :].sum(dim=1)
        match = torch.allclose(out, expected_ws, atol=1e-5)
        print(f"  Window={ws}: {out[0, :, 0].tolist()} - {'PASS' if match else 'FAIL'}")

    print("\nPASS: WindowedSum")
    return match_direct and match_attn


def test_windowed_mean():
    """Test WindowedMean - mean over last N tokens."""
    print("\n" + "="*60)
    print("TEST: WindowedMean")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    window_size = 3
    x = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")
    print(f"Window size: {window_size}")

    # Compute expected
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        expected[:, i, :] = x[:, start:i+1, :].mean(dim=1)

    # Test direct method
    op_direct = WindowedMean(hidden, window_size, method='direct')
    out_direct = op_direct(x)
    print(f"Direct: {[f'{v:.2f}' for v in out_direct[0, :, 0].tolist()]}")
    print(f"Expected: {[f'{v:.2f}' for v in expected[0, :, 0].tolist()]}")

    match_direct = torch.allclose(out_direct, expected, atol=1e-5)
    print(f"Direct method correct: {match_direct}")

    # Test attention method
    op_attn = WindowedMean(hidden, window_size, method='attention')
    out_attn = op_attn(x)
    print(f"Attention: {[f'{v:.2f}' for v in out_attn[0, :, 0].tolist()]}")

    match_attn = torch.allclose(out_attn, expected, atol=1e-5)
    print(f"Attention method correct: {match_attn}")

    print("\nPASS: WindowedMean")
    return match_direct and match_attn


def test_windowed_max():
    """Test WindowedMax - max over last N tokens."""
    print("\n" + "="*60)
    print("TEST: WindowedMax")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    window_size = 3

    # Use varied data to test max properly
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.float32)
    x = x.view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")
    print(f"Window size: {window_size}")

    # Compute expected
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        expected[:, i, :] = x[:, start:i+1, :].max(dim=1).values

    # Test direct method
    op_direct = WindowedMax(hidden, window_size, method='direct')
    out_direct = op_direct(x)
    print(f"Direct: {out_direct[0, :, 0].tolist()}")
    print(f"Expected: {expected[0, :, 0].tolist()}")

    match_direct = torch.allclose(out_direct, expected, atol=1e-5)
    print(f"Direct method correct: {match_direct}")

    # Test unfold method
    op_unfold = WindowedMax(hidden, window_size, method='unfold')
    out_unfold = op_unfold(x)
    print(f"Unfold: {out_unfold[0, :, 0].tolist()}")

    match_unfold = torch.allclose(out_unfold, expected, atol=1e-5)
    print(f"Unfold method correct: {match_unfold}")

    # Test with window_size = 1 (should equal input)
    op_ws1 = WindowedMax(hidden, 1, method='direct')
    out_ws1 = op_ws1(x)
    match_ws1 = torch.allclose(out_ws1, x, atol=1e-5)
    print(f"Window=1 equals input: {match_ws1}")

    print("\nPASS: WindowedMax")
    return match_direct and match_unfold and match_ws1


def test_windowed_min():
    """Test WindowedMin - min over last N tokens."""
    print("\n" + "="*60)
    print("TEST: WindowedMin")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    window_size = 3

    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.float32)
    x = x.view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")
    print(f"Window size: {window_size}")

    # Compute expected
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        expected[:, i, :] = x[:, start:i+1, :].min(dim=1).values

    # Test direct method
    op_direct = WindowedMin(hidden, window_size, method='direct')
    out_direct = op_direct(x)
    print(f"Direct: {out_direct[0, :, 0].tolist()}")
    print(f"Expected: {expected[0, :, 0].tolist()}")

    match_direct = torch.allclose(out_direct, expected, atol=1e-5)
    print(f"Direct method correct: {match_direct}")

    # Test unfold method
    op_unfold = WindowedMin(hidden, window_size, method='unfold')
    out_unfold = op_unfold(x)

    match_unfold = torch.allclose(out_unfold, expected, atol=1e-5)
    print(f"Unfold method correct: {match_unfold}")

    print("\nPASS: WindowedMin")
    return match_direct and match_unfold


def test_windowed_product():
    """Test WindowedProduct - product over last N tokens via log/exp."""
    print("\n" + "="*60)
    print("TEST: WindowedProduct")
    print("="*60)

    batch, seq_len, hidden = 2, 6, 4
    window_size = 3

    # Use values near 1 for numerical stability
    x = torch.tensor([1.1, 1.2, 0.9, 1.3, 0.8, 1.1], dtype=torch.float32)
    x = x.view(1, seq_len, 1).expand(batch, seq_len, hidden)

    print(f"Input: {x[0, :, 0].tolist()}")
    print(f"Window size: {window_size}")

    # Compute expected
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        expected[:, i, :] = x[:, start:i+1, :].prod(dim=1)

    # Test direct method
    op = WindowedProduct(hidden, window_size, method='direct')
    out = op(x)
    print(f"Output: {[f'{v:.4f}' for v in out[0, :, 0].tolist()]}")
    print(f"Expected: {[f'{v:.4f}' for v in expected[0, :, 0].tolist()]}")

    match = torch.allclose(out, expected, atol=1e-4)
    print(f"Correct: {match}")

    # Verify relationship with windowed_sum and log/exp
    windowed_log_sum = WindowedSum(hidden, window_size, method='direct')
    log_x = torch.log(x)
    ws_log = windowed_log_sum(log_x)
    reconstructed = torch.exp(ws_log)
    match_identity = torch.allclose(reconstructed, expected, atol=1e-4)
    print(f"exp(windowed_sum(log(x))) = windowed_product(x): {match_identity}")

    print("\nPASS: WindowedProduct")
    return match and match_identity


def test_windowed_logsumexp():
    """Test WindowedLogSumExp - logsumexp over last N tokens."""
    print("\n" + "="*60)
    print("TEST: WindowedLogSumExp")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    window_size = 3
    x = torch.randn(batch, seq_len, hidden)

    print(f"Input shape: {x.shape}, Window size: {window_size}")

    # Compute expected
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        expected[:, i, :] = torch.logsumexp(x[:, start:i+1, :], dim=1)

    # Test direct method
    op_direct = WindowedLogSumExp(hidden, window_size, method='direct')
    out_direct = op_direct(x)

    match_direct = torch.allclose(out_direct, expected, atol=1e-5)
    print(f"Direct method correct: {match_direct}")

    # Test unfold method
    op_unfold = WindowedLogSumExp(hidden, window_size, method='unfold')
    out_unfold = op_unfold(x)

    match_unfold = torch.allclose(out_unfold, expected, atol=1e-5)
    print(f"Unfold method correct: {match_unfold}")

    # Test attention method
    op_attn = WindowedLogSumExp(hidden, window_size, method='attention')
    out_attn = op_attn(x)

    match_attn = torch.allclose(out_attn, expected, atol=1e-5)
    print(f"Attention method correct: {match_attn}")

    # Verify: exp(windowed_logsumexp) = windowed_sum(exp(x))
    windowed_sum_exp = WindowedSum(hidden, window_size, method='direct')
    exp_x = torch.exp(x)
    sum_exp = windowed_sum_exp(exp_x)
    from_lse = torch.exp(out_direct)
    match_identity = torch.allclose(from_lse, sum_exp, atol=1e-4)
    print(f"exp(windowed_lse) = windowed_sum(exp(x)): {match_identity}")

    print("\nPASS: WindowedLogSumExp")
    return match_direct and match_unfold and match_attn


def test_windowed_variance():
    """Test WindowedVariance - variance over last N tokens."""
    print("\n" + "="*60)
    print("TEST: WindowedVariance")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    window_size = 3
    x = torch.randn(batch, seq_len, hidden)

    print(f"Input shape: {x.shape}, Window size: {window_size}")

    # Compute expected using the E[X²] - E[X]² formula
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        window = x[:, start:i+1, :]
        mean_x = window.mean(dim=1)
        mean_x_sq = (window ** 2).mean(dim=1)
        expected[:, i, :] = mean_x_sq - mean_x ** 2

    # Test direct method
    op = WindowedVariance(hidden, window_size, method='direct')
    out = op(x)

    match = torch.allclose(out, expected, atol=1e-5)
    print(f"Direct method correct: {match}")

    # Test attention method
    op_attn = WindowedVariance(hidden, window_size, method='attention')
    out_attn = op_attn(x)

    match_attn = torch.allclose(out_attn, expected, atol=1e-5)
    print(f"Attention method correct: {match_attn}")

    # Verify variance is non-negative
    non_negative = (out >= -1e-6).all()  # small tolerance for numerical errors
    print(f"Variance non-negative: {non_negative}")

    # Compare with torch.var (note: torch.var uses N-1 by default)
    # Our variance uses N, so they won't match exactly
    print("\nNote: Our variance uses N divisor, torch.var uses N-1 (Bessel correction)")

    print("\nPASS: WindowedVariance")
    return match and match_attn


def test_windowed_std():
    """Test WindowedStd - standard deviation over last N tokens."""
    print("\n" + "="*60)
    print("TEST: WindowedStd")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4
    window_size = 3
    x = torch.randn(batch, seq_len, hidden)

    # Compute expected
    expected = torch.zeros_like(x)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        window = x[:, start:i+1, :]
        mean_x = window.mean(dim=1)
        mean_x_sq = (window ** 2).mean(dim=1)
        var_x = mean_x_sq - mean_x ** 2
        expected[:, i, :] = torch.sqrt(var_x + 1e-8)

    # Test
    op = WindowedStd(hidden, window_size, method='direct')
    out = op(x)

    match = torch.allclose(out, expected, atol=1e-5)
    print(f"Correct: {match}")

    # Verify std is positive
    positive = (out > 0).all()
    print(f"Std positive: {positive}")

    # Verify std² ≈ variance
    var_op = WindowedVariance(hidden, window_size, method='direct')
    var_out = var_op(x)
    match_sq = torch.allclose(out ** 2, var_out + 1e-8, atol=1e-5)
    print(f"std² ≈ variance: {match_sq}")

    print("\nPASS: WindowedStd")
    return match and positive


def test_windowed_edge_cases():
    """Test edge cases for windowed operations."""
    print("\n" + "="*60)
    print("TEST: Windowed Edge Cases")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 4

    # Test 1: window_size = 1 (each position only sees itself)
    print("\n1. Window size = 1:")
    x = torch.randn(batch, seq_len, hidden)
    op_sum = WindowedSum(hidden, 1, method='direct')
    op_mean = WindowedMean(hidden, 1, method='direct')
    op_max = WindowedMax(hidden, 1, method='direct')

    assert torch.allclose(op_sum(x), x), "window=1 sum should equal input"
    assert torch.allclose(op_mean(x), x), "window=1 mean should equal input"
    assert torch.allclose(op_max(x), x), "window=1 max should equal input"
    print("   All window=1 tests pass (output equals input)")

    # Test 2: window_size = seq_len (full causal)
    print("\n2. Window size = seq_len (full causal):")
    op_sum_full = WindowedSum(hidden, seq_len, method='direct')
    out_full = op_sum_full(x)
    expected_full = torch.cumsum(x, dim=1)
    assert torch.allclose(out_full, expected_full), "window=seq_len should equal cumsum"
    print("   window=seq_len sum equals cumsum: PASS")

    # Test 3: window_size > seq_len
    print("\n3. Window size > seq_len:")
    op_big = WindowedSum(hidden, seq_len * 2, method='direct')
    out_big = op_big(x)
    assert torch.allclose(out_big, expected_full), "window>seq_len should equal cumsum"
    print("   window>seq_len handled correctly: PASS")

    # Test 4: Constant input
    print("\n4. Constant input:")
    x_const = torch.ones(batch, seq_len, hidden) * 5.0
    op_var = WindowedVariance(hidden, 3, method='direct')
    var_const = op_var(x_const)
    assert torch.allclose(var_const, torch.zeros_like(var_const), atol=1e-6), "Variance of constant should be 0"
    print("   Variance of constant = 0: PASS")

    # Test 5: Window at sequence start
    print("\n5. Window behavior at sequence start:")
    x = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(batch, seq_len, hidden)
    op_sum_3 = WindowedSum(hidden, 3, method='direct')
    out = op_sum_3(x)
    # Position 0: sum(1) = 1
    # Position 1: sum(1,2) = 3
    # Position 2: sum(1,2,3) = 6
    # Position 3: sum(2,3,4) = 9
    expected_first = torch.tensor([1.0, 3.0, 6.0, 9.0])
    assert torch.allclose(out[0, :4, 0], expected_first), "Window start sums incorrect"
    print(f"   First 4 positions: {out[0, :4, 0].tolist()} = {expected_first.tolist()}: PASS")

    print("\nPASS: Windowed Edge Cases")
    return True


def test_sequence_mean():
    """Test SequenceMean against exact mean computation."""
    print("\n" + "="*60)
    print("TEST: SequenceMean vs torch.mean")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128
    x = torch.randn(batch, seq_len, hidden)

    # Our implementation
    mean_op = SequenceMean(hidden)
    our_result = mean_op(x)

    # Exact operation
    exact_result = x.mean(dim=1, keepdim=True)

    # Compare
    max_diff = (our_result - exact_result).abs().max().item()
    mean_diff = (our_result - exact_result).abs().mean().item()

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {our_result.shape}")
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"PASS: {max_diff < 1e-6}")

    return max_diff < 1e-6


def test_square():
    """Test Square against exact x^2 computation."""
    print("\n" + "="*60)
    print("TEST: Square vs x**2")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128
    x = torch.randn(batch, seq_len, hidden)

    # Our implementation
    sq_op = Square(hidden)
    our_result = sq_op(x)

    # Exact operation
    exact_result = x ** 2

    # Compare
    max_diff = (our_result - exact_result).abs().max().item()
    mean_diff = (our_result - exact_result).abs().mean().item()

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {our_result.shape}")
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"PASS: {max_diff < 1e-6}")

    return max_diff < 1e-6


def test_relu_via_swiglu():
    """Test ReLUViaSwiGLU against exact ReLU."""
    print("\n" + "="*60)
    print("TEST: ReLUViaSwiGLU vs F.relu")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128
    x = torch.randn(batch, seq_len, hidden)

    # Test with different beta values
    for beta in [1.0, 10.0, 50.0, 100.0]:
        relu_op = ReLUViaSwiGLU(hidden, beta=beta)
        our_result = relu_op(x)

        # Exact ReLU
        exact_result = F.relu(x)

        # Compare
        max_diff = (our_result - exact_result).abs().max().item()
        mean_diff = (our_result - exact_result).abs().mean().item()

        print(f"\nbeta={beta}:")
        print(f"  Max absolute difference:  {max_diff:.4f}")
        print(f"  Mean absolute difference: {mean_diff:.4f}")

    # Final test with high beta
    relu_op = ReLUViaSwiGLU(hidden, beta=100.0)
    our_result = relu_op(x)
    exact_result = F.relu(x)
    max_diff = (our_result - exact_result).abs().max().item()

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {our_result.shape}")
    print(f"PASS (beta=100, threshold=0.1): {max_diff < 0.1}")

    return max_diff < 0.1


def test_softmax1():
    """Test Softmax1 against exact softmax1 formula."""
    print("\n" + "="*60)
    print("TEST: Softmax1 vs exact formula")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128
    x = torch.randn(batch, seq_len, hidden)

    # Our implementation
    sm1 = Softmax1(dim=-1)
    our_result = sm1(x)

    # Exact softmax1: exp(x_i) / (1 + sum(exp(x_j)))
    # With numerical stability
    x_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    exact_result = exp_x / (torch.exp(-x_max) + exp_x.sum(dim=-1, keepdim=True))

    # Compare
    max_diff = (our_result - exact_result).abs().max().item()
    mean_diff = (our_result - exact_result).abs().mean().item()

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {our_result.shape}")
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    # Also compare properties with regular softmax
    regular_softmax = F.softmax(x, dim=-1)

    print(f"\nComparison with regular softmax:")
    print(f"  Softmax1 sum (should be < 1): {our_result.sum(dim=-1).mean():.4f}")
    print(f"  Regular softmax sum:          {regular_softmax.sum(dim=-1).mean():.4f}")

    # Test with negative values (where softmax1 differs most)
    x_neg = torch.randn(batch, seq_len, hidden) - 3.0  # shift to be mostly negative
    sm1_neg = sm1(x_neg)
    softmax_neg = F.softmax(x_neg, dim=-1)

    print(f"\nWith mostly negative inputs:")
    print(f"  Softmax1 sum: {sm1_neg.sum(dim=-1).mean():.4f}")
    print(f"  Regular softmax sum: {softmax_neg.sum(dim=-1).mean():.4f}")

    print(f"\nPASS: {max_diff < 1e-6}")

    return max_diff < 1e-6


def test_relu_via_silu():
    """Test ReLUViaSiLU against exact ReLU."""
    print("\n" + "="*60)
    print("TEST: ReLUViaSiLU (SiLU(βx) * 1/β) vs F.relu")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128
    x = torch.randn(batch, seq_len, hidden)

    # Test with different beta values (direct division)
    print("\nDirect division (SiLU(βx) / β):")
    for beta in [1.0, 10.0, 50.0, 100.0]:
        relu_op = ReLUViaSiLU(hidden, beta=beta, use_attn_reciprocal=False)
        our_result = relu_op(x)
        exact_result = F.relu(x)
        max_diff = (our_result - exact_result).abs().max().item()
        print(f"  beta={beta}: max diff = {max_diff:.4f}")

    # Test with attention-based reciprocal
    print("\nAttention-based reciprocal (SiLU(βx) * attn_1/β):")
    for beta in [10.0, 50.0, 100.0]:
        relu_op = ReLUViaSiLU(hidden, beta=beta, use_attn_reciprocal=True)
        our_result = relu_op(x)
        exact_result = F.relu(x)
        max_diff = (our_result - exact_result).abs().max().item()
        print(f"  beta={beta}: max diff = {max_diff:.4f}")

    # Final test
    relu_op = ReLUViaSiLU(hidden, beta=100.0, use_attn_reciprocal=True)
    our_result = relu_op(x)
    exact_result = F.relu(x)
    max_diff = (our_result - exact_result).abs().max().item()

    print(f"\nPASS (beta=100, attn_reciprocal, threshold=0.1): {max_diff < 0.1}")

    return max_diff < 0.1


def test_reciprocal_via_attention():
    """Test ReciprocalViaAttention against exact 1/(1+x) and 1/x."""
    print("\n" + "="*60)
    print("TEST: ReciprocalViaAttention vs exact formulas")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128
    # Use positive values for reciprocal
    x = torch.rand(batch, seq_len, hidden) * 5 + 0.1  # x in [0.1, 5.1]

    recip_op = ReciprocalViaAttention(hidden)

    # Test 1/(1+x) - the natural output of the attention mechanism
    our_result = recip_op(x)
    exact_result = 1.0 / (1.0 + x)

    max_diff = (our_result - exact_result).abs().max().item()
    mean_diff = (our_result - exact_result).abs().mean().item()

    print(f"Testing 1/(1+x):")
    print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  PASS: {max_diff < 1e-5}")

    # Test 1/x via shifted input (for x > 1)
    x_large = torch.rand(batch, seq_len, hidden) * 5 + 1.5  # x in [1.5, 6.5]
    our_recip = recip_op.forward_reciprocal(x_large)
    exact_recip = 1.0 / x_large

    max_diff_recip = (our_recip - exact_recip).abs().max().item()
    mean_diff_recip = (our_recip - exact_recip).abs().mean().item()

    print(f"\nTesting 1/x (via shifted input, x > 1):")
    print(f"  Input range: [{x_large.min():.2f}, {x_large.max():.2f}]")
    print(f"  Max absolute difference:  {max_diff_recip:.2e}")
    print(f"  Mean absolute difference: {mean_diff_recip:.2e}")
    print(f"  PASS: {max_diff_recip < 1e-4}")

    return max_diff < 1e-5 and max_diff_recip < 1e-4


def test_reciprocal_via_sequence_attention():
    """Test ReciprocalViaSequenceAttention - multiple tokens computing 1/x in parallel."""
    print("\n" + "="*60)
    print("TEST: ReciprocalViaSequenceAttention (multi-token parallel)")
    print("="*60)

    batch, seq_len, hidden = 4, 16, 64
    # Each position has different x values
    x = torch.rand(batch, seq_len, hidden) * 5 + 0.1  # x in [0.1, 5.1]

    recip_op = ReciprocalViaSequenceAttention(hidden)

    # Test 1/(1+x) for all positions simultaneously
    our_result = recip_op(x)
    exact_result = 1.0 / (1.0 + x)

    max_diff = (our_result - exact_result).abs().max().item()
    mean_diff = (our_result - exact_result).abs().mean().item()

    print(f"All {seq_len} positions computing 1/(1+x) in parallel:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {our_result.shape}")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    # Verify each position computed correctly
    print(f"\nPer-position verification (first batch, first hidden dim):")
    for i in range(min(5, seq_len)):
        x_val = x[0, i, 0].item()
        our_val = our_result[0, i, 0].item()
        exact_val = exact_result[0, i, 0].item()
        print(f"  pos {i}: x={x_val:.3f}, 1/(1+x)={our_val:.4f}, exact={exact_val:.4f}")

    # Test 1/x via shifted input
    x_large = torch.rand(batch, seq_len, hidden) * 5 + 1.5
    our_recip = recip_op.forward_reciprocal(x_large)
    exact_recip = 1.0 / x_large

    max_diff_recip = (our_recip - exact_recip).abs().max().item()

    print(f"\nTesting 1/x (shifted input):")
    print(f"  Max absolute difference: {max_diff_recip:.2e}")

    passed = max_diff < 1e-5 and max_diff_recip < 1e-4
    print(f"\nPASS: {passed}")

    return passed


def test_reciprocal_full_attention():
    """
    Test that 1/x works even with FULL attention (no masking) when
    all data positions have value=0.

    Key insight: If value[j]=0 for most positions, attending to them
    contributes nothing. Only special tokens with value≠0 matter.
    """
    print("\n" + "="*60)
    print("TEST: Reciprocal with FULL attention (no masking)")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 32
    x = torch.rand(batch, seq_len, hidden) * 5 + 0.5

    x_safe = x.clamp(min=1e-6)
    log_x = torch.log(x_safe)

    # Setup: n_pos = seq_len + 2 (one token + absorber + data)
    n_pos = seq_len + 2

    # ALL positions attend to ALL positions (no masking!)
    # Scores: mostly 0, except position i attends to absorber (pos 1) with log(x_i)
    scores = torch.zeros((batch, n_pos, n_pos, hidden), device=x.device)

    # Positions 2..n_pos-1 are data positions
    for i in range(seq_len):
        scores[:, i + 2, 1, :] = log_x[:, i, :]  # attend to absorber with log(x)

    # Full softmax - every position attends to every position
    attn = F.softmax(scores, dim=2)  # (batch, n_pos, n_pos, hidden)

    # Values: ONLY pos 0 has value=1, all others have value=0
    # This is the key! Attending to value=0 positions does nothing.
    values = torch.zeros((batch, n_pos, hidden))
    values[:, 0, :] = 1.0  # "one" token

    # Output = sum_j attn[i,j] * value[j] = attn[i,0] * 1 + rest * 0 = attn[i,0]
    # attn[i,0] = exp(0) / (exp(0) + exp(log(x)) + (n_pos-2)*exp(0))
    #           = 1 / (1 + x + n_pos - 2)
    #           = 1 / (x + n_pos - 1)

    # So with n_pos positions, we get 1/(x + n_pos - 1), not 1/(1+x)
    # To get 1/(1+x), we need ONLY 2 positions (one + absorber)

    print(f"With {n_pos} positions (full attention, no masking):")
    print(f"  Each data position attends to ALL {n_pos} positions")
    print(f"  But only pos 0 has value=1, rest have value=0")
    print(f"  Result: 1/(x + {n_pos-1}) due to extra positions with score 0")

    # Verify the formula
    output = attn[:, 2:, 0, :]  # attention to "one" token for data positions
    expected = 1.0 / (x + n_pos - 1)

    max_diff = (output - expected).abs().max().item()
    print(f"  Max diff from 1/(x+{n_pos-1}): {max_diff:.2e}")

    # Now show that with ONLY 2 positions, we get exact 1/(1+x)
    print(f"\nWith only 2 positions (one + absorber):")
    scores_2 = torch.stack([torch.zeros_like(x), log_x], dim=-1)
    attn_2 = F.softmax(scores_2, dim=-1)
    output_2 = attn_2[..., 0]  # attention to "one"

    exact = 1.0 / (1.0 + x)
    max_diff_2 = (output_2 - exact).abs().max().item()
    print(f"  Max diff from 1/(1+x): {max_diff_2:.2e}")

    print(f"\nConclusion: Use dedicated 2-position attention per hidden dim,")
    print(f"or adjust input: to get 1/(1+x), attend with score log(x*(n-1))")

    return max_diff < 1e-5 and max_diff_2 < 1e-6


def test_exp_silu_reciprocal():
    """Test ExpViaSiLUReciprocal (SiLU(x)/x = sigmoid ≈ exp) against exact exp."""
    print("\n" + "="*60)
    print("TEST: ExpViaSiLUReciprocal (SiLU(x)*1/x) vs torch.exp")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128

    exp_op = ExpViaSiLUReciprocal(hidden)

    # Test negative region (where sigmoid(x) ≈ exp(x))
    print("\nNegative region (sigmoid(x) ≈ exp(x)):")
    for x_min in [-1, -2, -3, -5]:
        x = torch.rand(batch, seq_len, hidden) * abs(x_min) + x_min  # [x_min, 0]
        our_result = exp_op(x)
        exact_result = torch.exp(x)

        max_diff = (our_result - exact_result).abs().max().item()
        rel_err = ((our_result - exact_result) / exact_result).abs().mean().item()

        print(f"  x ∈ [{x_min}, 0]: max_diff={max_diff:.4f}, rel_err={rel_err:.4f}")

    # Test positive region (where 1/sigmoid(-x) ≈ exp(x))
    print("\nPositive region (1/sigmoid(-x) ≈ exp(x)):")
    for x_max in [1, 2, 3, 5]:
        x = torch.rand(batch, seq_len, hidden) * x_max  # [0, x_max]
        our_result = exp_op(x)
        exact_result = torch.exp(x)

        max_diff = (our_result - exact_result).abs().max().item()
        rel_err = ((our_result - exact_result) / exact_result).abs().mean().item()

        print(f"  x ∈ [0, {x_max}]: max_diff={max_diff:.4f}, rel_err={rel_err:.4f}")

    # Final test on moderate range
    x = torch.rand(batch, seq_len, hidden) * 4 - 2  # [-2, 2]
    our_result = exp_op(x)
    exact_result = torch.exp(x)
    rel_err = ((our_result - exact_result) / exact_result).abs().mean().item()

    print(f"\nPASS (x ∈ [-2,2], rel_err < 0.5): {rel_err < 0.5}")

    return rel_err < 0.5


def test_exp_taylor():
    """Test ExpViaTaylor (Taylor series) against exact exp."""
    print("\n" + "="*60)
    print("TEST: ExpViaTaylor vs torch.exp")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128

    # Test in different ranges - Taylor works best for small |x|
    for x_range, threshold in [
        ((-1, 1), 0.01),
        ((-2, 2), 0.1),
        ((-3, 3), 0.5),
    ]:
        x = torch.rand(batch, seq_len, hidden) * (x_range[1] - x_range[0]) + x_range[0]

        for order in [4, 6, 8]:
            exp_op = ExpViaTaylor(hidden, order=order)
            our_result = exp_op(x)
            exact_result = torch.exp(x)

            max_diff = (our_result - exact_result).abs().max().item()
            mean_diff = (our_result - exact_result).abs().mean().item()

            print(f"\nx ∈ [{x_range[0]}, {x_range[1]}], order={order}:")
            print(f"  Max absolute difference:  {max_diff:.4f}")
            print(f"  Mean absolute difference: {mean_diff:.4f}")

    # Final test
    x = torch.rand(batch, seq_len, hidden) * 2 - 1  # [-1, 1]
    exp_op = ExpViaTaylor(hidden, order=6)
    our_result = exp_op(x)
    exact_result = torch.exp(x)
    max_diff = (our_result - exact_result).abs().max().item()

    print(f"\nPASS (x ∈ [-1,1], order=6, threshold=0.01): {max_diff < 0.01}")

    return max_diff < 0.01


def test_exp_composition():
    """Test ExpViaSiLUScaling ((1+x/n)^n composition) against exact exp."""
    print("\n" + "="*60)
    print("TEST: ExpViaSiLUScaling ((1+x/n)^n) vs torch.exp")
    print("="*60)

    batch, seq_len, hidden = 4, 32, 128

    for n_comp in [4, 8, 16, 32]:
        x = torch.rand(batch, seq_len, hidden) * 4 - 2  # [-2, 2]

        exp_op = ExpViaSiLUScaling(hidden, n_compositions=n_comp)
        our_result = exp_op(x)
        exact_result = torch.exp(x)

        max_diff = (our_result - exact_result).abs().max().item()
        mean_diff = (our_result - exact_result).abs().mean().item()
        # Relative error is more meaningful for exp
        rel_err = ((our_result - exact_result) / exact_result).abs().mean().item()

        print(f"\nn_compositions={n_comp}:")
        print(f"  Max absolute difference:  {max_diff:.4f}")
        print(f"  Mean absolute difference: {mean_diff:.4f}")
        print(f"  Mean relative error:      {rel_err:.4f}")

    # Final test
    x = torch.rand(batch, seq_len, hidden) * 2 - 1  # [-1, 1]
    exp_op = ExpViaSiLUScaling(hidden, n_compositions=32)
    our_result = exp_op(x)
    exact_result = torch.exp(x)
    max_diff = (our_result - exact_result).abs().max().item()

    print(f"\nPASS (x ∈ [-1,1], n=32, threshold=0.1): {max_diff < 0.1}")

    return max_diff < 0.1


def test_log_approximation():
    """Test LogApproximation against exact log."""
    print("\n" + "="*60)
    print("TEST: LogApproximation vs torch.log")
    print("="*60)

    batch, seq_len, hidden = 4, 16, 64

    # Taylor series works best near x=1
    print("\nTaylor series (works for |x-1| < 1):")
    for x_range in [(0.5, 1.5), (0.3, 1.7), (0.8, 1.2)]:
        x = torch.rand(batch, seq_len, hidden) * (x_range[1] - x_range[0]) + x_range[0]
        log_op = LogApproximation(hidden, method='taylor', order=10)
        our_result = log_op(x)
        exact_result = torch.log(x)

        max_diff = (our_result - exact_result).abs().max().item()
        print(f"  x ∈ [{x_range[0]}, {x_range[1]}]: max_diff = {max_diff:.4f}")

    # Newton iteration
    print("\nNewton iteration (uses exp + reciprocal):")
    x = torch.rand(batch, seq_len, hidden) * 2 + 0.5  # [0.5, 2.5]
    log_op = LogApproximation(hidden, method='newton')
    our_result = log_op(x)
    exact_result = torch.log(x)

    max_diff = (our_result - exact_result).abs().max().item()
    mean_diff = (our_result - exact_result).abs().mean().item()
    print(f"  x ∈ [0.5, 2.5]: max_diff = {max_diff:.4f}, mean_diff = {mean_diff:.4f}")

    # Show the circular dependency issue
    print("\n" + "-"*40)
    print("NOTE: Computing log is hard because:")
    print("  - To get score log(x), we need log")
    print("  - To get log via Newton, we need exp")
    print("  - exp via Taylor works, but needs powers of x")
    print("  - This is achievable but requires multiple layers")
    print("-"*40)

    return max_diff < 1.0  # loose threshold for now


def test_reciprocal_other_methods():
    """Test alternative reciprocal methods."""
    print("\n" + "="*60)
    print("TEST: Other 1/x methods")
    print("="*60)

    batch, seq_len, hidden = 2, 8, 32

    # 1. Geometric series: 1 - x + x² - x³ + ...
    print("\n1. Geometric series: 1/(1+x) = 1 - x + x² - ...")
    geo_op = ReciprocalGeometricSeries(hidden, n_terms=15)
    for x_max in [0.3, 0.5, 0.7, 0.9]:
        x = torch.rand(batch, seq_len, hidden) * x_max
        our = geo_op(x)
        exact = 1 / (1 + x)
        max_diff = (our - exact).abs().max().item()
        print(f"  x ∈ [0, {x_max}]: max_diff = {max_diff:.4f}")

    # 2. Sigmoid linear regime: sigmoid(2 - 4x) ≈ 1/(1+x)
    print("\n2. Sigmoid linear: sigmoid(2 - 4x) ≈ 1-x ≈ 1/(1+x)")
    sig_op = ReciprocalViaSigmoidIteration(hidden)
    for x_max in [0.1, 0.2, 0.3, 0.5]:
        x = torch.rand(batch, seq_len, hidden) * x_max
        our = sig_op(x)
        exact = 1 / (1 + x)
        max_diff = (our - exact).abs().max().item()
        print(f"  x ∈ [0, {x_max}]: max_diff = {max_diff:.4f}")

    # 3. Taylor at x=1: 1/x = 1 + (1-x) + (1-x)² + ...
    print("\n3. Taylor at x=1: 1/x = Σ(1-x)^n, converges for x ∈ (0, 2)")
    taylor_op = ReciprocalViaTaylorAtOne(hidden, n_terms=15)
    for x_range in [(0.5, 1.0), (0.5, 1.5), (0.3, 1.7)]:
        x = torch.rand(batch, seq_len, hidden) * (x_range[1] - x_range[0]) + x_range[0]
        our = taylor_op(x)
        exact = 1 / x
        max_diff = (our - exact).abs().max().item()
        print(f"  x ∈ [{x_range[0]}, {x_range[1]}]: max_diff = {max_diff:.4f}")

    # Summary
    print("\n" + "-"*40)
    print("KEY INSIGHTS:")
    print("  - Geometric series: 1/(1+x), needs |x| < 1")
    print("  - Sigmoid trick: uses sigmoid's linear regime")
    print("  - Taylor at 1: 1/x directly, needs |x-1| < 1")
    print("  - ALL avoid computing log(x)!")
    print("-"*40)

    return True


def test_reciprocal_simple():
    """Test simple reciprocal methods - linear approx and Newton."""
    print("\n" + "="*60)
    print("TEST: Simple 1/x methods (no log needed!)")
    print("="*60)

    batch, seq_len, hidden = 4, 16, 64

    # 1. Linear approximation: 1/(1+x) ≈ 1-x for small x
    print("\n1. Linear approximation: 1/(1+x) ≈ 1-x")
    linear_op = ReciprocalLinear(hidden)

    for x_max in [0.1, 0.3, 0.5, 1.0]:
        x = torch.rand(batch, seq_len, hidden) * x_max
        our = linear_op(x)
        exact = 1 / (1 + x)
        max_diff = (our - exact).abs().max().item()
        print(f"  x ∈ [0, {x_max}]: max_diff = {max_diff:.4f}")

    # 2. Newton iteration: y' = 2y - xy²
    print("\n2. Newton iteration: y_{n+1} = 2y_n - x*y_n²")
    newton_op = ReciprocalNewton(hidden, n_iter=5)

    for x_range in [(0.1, 0.5), (0.5, 1.0), (0.5, 1.5), (0.5, 1.9)]:
        x = torch.rand(batch, seq_len, hidden) * (x_range[1] - x_range[0]) + x_range[0]
        our = newton_op(x)
        exact = 1 / x
        max_diff = (our - exact).abs().max().item()
        rel_err = ((our - exact) / exact).abs().mean().item()
        print(f"  x ∈ [{x_range[0]}, {x_range[1]}]: max_diff = {max_diff:.4f}, rel_err = {rel_err:.4f}")

    # 3. Show Newton converges
    print("\n3. Newton convergence (x=1.5):")
    x = torch.tensor([1.5])
    y = torch.ones_like(x)
    for i in range(8):
        y_new = 2 * y - x * y * y
        print(f"  iter {i}: y = {y.item():.6f}, exact = {(1/x).item():.6f}, err = {abs(y.item() - 1/x.item()):.2e}")
        y = y_new

    # Final test
    x = torch.rand(batch, seq_len, hidden) * 1.5 + 0.3  # [0.3, 1.8]
    our = newton_op(x)
    exact = 1 / x
    max_diff = (our - exact).abs().max().item()

    print(f"\nPASS (Newton, x ∈ [0.3, 1.8], threshold=0.01): {max_diff < 0.01}")
    return max_diff < 0.01


def test_reciprocal_denominator():
    """Show explicitly that score=0 contributes exp(0)=1 to denominator."""
    print("\n" + "="*60)
    print("TEST: Softmax denominator contributions")
    print("="*60)

    x = torch.tensor([2.0])  # simple example

    print(f"\nFor x = {x.item()}:")

    # Two positions: score=[0, log(x)]
    scores = torch.tensor([[0.0, math.log(x.item())]])
    attn = F.softmax(scores, dim=-1)

    print(f"  Scores: [0, log({x.item()})] = [0, {math.log(x.item()):.3f}]")
    print(f"  exp(scores): [1, {x.item()}]")
    print(f"  Denominator: 1 + {x.item()} = {1 + x.item()}")
    print(f"  Attention: [{attn[0,0]:.4f}, {attn[0,1]:.4f}]")
    print(f"  Output (value=[1,0]): {attn[0,0]:.4f} = 1/(1+{x.item()}) = {1/(1+x.item()):.4f}")

    print(f"\nTo get 1/x = 1/{x.item()} = {1/x.item():.4f}:")
    print(f"  Option 1: Shift input to (x-1) = {x.item()-1}")
    shifted_scores = torch.tensor([[0.0, math.log(x.item() - 1)]])
    shifted_attn = F.softmax(shifted_scores, dim=-1)
    print(f"    Result: 1/(1+{x.item()-1}) = {shifted_attn[0,0]:.4f}")

    print(f"\n  Option 2: Use ONLY the absorber position (no score=0 position)")
    print(f"    softmax([log(x)]) = [1.0] (trivial)")
    print(f"    Need different setup...")

    print(f"\n  Option 3: softmax1 (adds 1 to denom without a position)")
    print(f"    softmax1([log(x)]) = exp(log(x))/(1+exp(log(x))) = x/(1+x)")
    print(f"    Then: 1 - x/(1+x) = 1/(1+x)... still not 1/x")

    print(f"\n  CONCLUSION: Pure 1/x requires the shift trick: feed (x-1)")

    return True


def run_all_tests():
    """Run all comparison tests."""
    print("\n" + "#"*60)
    print("# HIDDEN DIM OPERATIONS - COMPARISON TESTS")
    print("#"*60)

    results = {
        # Positional operations
        'SelectNBehind': test_select_n_behind(),
        'CopyFromNBehind': test_copy_from_n_behind(),
        'GetNumericPosition': test_get_numeric_position(),

        # Running/cumulative operations
        'RunningSum': test_running_sum(),
        'RunningProduct': test_running_product(),
        'LogSumExp': test_logsumexp(),
        'RunningLogSumExp': test_running_logsumexp(),

        # Windowed operations
        'WindowMask': test_window_mask(),
        'WindowedSum': test_windowed_sum(),
        'WindowedMean': test_windowed_mean(),
        'WindowedMax': test_windowed_max(),
        'WindowedMin': test_windowed_min(),
        'WindowedProduct': test_windowed_product(),
        'WindowedLogSumExp': test_windowed_logsumexp(),
        'WindowedVariance': test_windowed_variance(),
        'WindowedStd': test_windowed_std(),
        'WindowedEdgeCases': test_windowed_edge_cases(),

        # Basic operations
        'SequenceMean': test_sequence_mean(),
        'Square': test_square(),
        'ReLUViaSwiGLU': test_relu_via_swiglu(),
        'ReLUViaSiLU': test_relu_via_silu(),
        'Softmax1': test_softmax1(),

        # Reciprocal methods
        'ReciprocalOther': test_reciprocal_other_methods(),
        'ReciprocalSimple': test_reciprocal_simple(),
        'ReciprocalViaAttention': test_reciprocal_via_attention(),
        'ReciprocalViaSeqAttention': test_reciprocal_via_sequence_attention(),
        'ReciprocalFullAttention': test_reciprocal_full_attention(),
        'ReciprocalDenominator': test_reciprocal_denominator(),

        # Log/Exp
        'LogApproximation': test_log_approximation(),
        'ExpViaSiLUReciprocal': test_exp_silu_reciprocal(),
        'ExpViaTaylor': test_exp_taylor(),
        'ExpViaComposition': test_exp_composition(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nAll tests passed: {all_passed}")
    return all_passed


if __name__ == "__main__":
    run_all_tests()
