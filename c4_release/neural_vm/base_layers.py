"""
Base neural layers for Neural VM V7.

PureFFN: SwiGLU FFN with fixed forward, subclasses bake weights
PureAttention: Attention with fixed forward, subclasses bake weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from typing import Literal

from .embedding import E


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs for RoPE.

    Matches the convention used by ``neural_vm.vm_step.rotate_half`` and the
    standard Llama-family rotary implementation: pair adjacent feature lanes,
    swap them, and negate the second.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE rotation requires an even feature dimension")
    x_pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair.unbind(dim=-1)
    return torch.stack((-x1, x0), dim=-1).reshape_as(x)


def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0,
                          device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables with shape ``[max_seq_len, head_dim]``.

    Identical convention to ``neural_vm.vm_step.precompute_rope_cache`` so the
    two attention implementations agree at the bit level when the same
    ``(head_dim, max_seq_len, base)`` are supplied.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    half_idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (half_idx / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    angles = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return angles.cos(), angles.sin()


def sparse_linear(x, weight_sparse, bias=None):
    """F.linear drop-in for sparse COO weight matrices.

    Computes x @ weight.T + bias, where weight is sparse [out_D, D].
    Handles 2D [N, D] and 3D [B, S, D] inputs.
    """
    if x.dim() == 3:
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
    else:
        x_flat = x
    # W_sparse: [out_D, D], x_flat.T: [D, N] -> result: [out_D, N] -> [N, out_D]
    out = torch.sparse.mm(weight_sparse, x_flat.t()).t()
    if bias is not None:
        out = out + bias
    if x.dim() == 3:
        out = out.reshape(B, S, -1)
    return out


def bake_weights(method):
    """
    Decorator for _bake_weights methods.

    Wraps the method in torch.no_grad() context for efficient weight initialization.
    Also provides S (scale factor) as a convenience.

    Usage:
        @bake_weights
        def _bake_weights(self):
            S = E.SCALE
            self.W_up[0, E.NIB_A] = S
            ...
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with torch.no_grad():
            return method(self, *args, **kwargs)
    return wrapper


class PureFFN(nn.Module):
    """
    Pure SwiGLU FFN with FINAL forward pass.

    Subclasses ONLY override _bake_weights() to set weight values.
    Forward is: output = x + W_down @ (silu(W_up @ x + b_up) * (W_gate @ x + b_gate)) + b_down
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.W_up = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_up = nn.Parameter(torch.zeros(hidden_dim))
        self.W_gate = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.W_down = nn.Parameter(torch.zeros(dim, hidden_dim))
        self.b_down = nn.Parameter(torch.zeros(dim))

        self._bake_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - Standard SwiGLU with all biases. DO NOT OVERRIDE."""
        # Manually add biases to ensure correct broadcast behavior
        up = F.linear(x, self.W_up) + self.b_up
        gate = F.linear(x, self.W_gate) + self.b_gate
        hidden = F.silu(up) * gate
        return x + F.linear(hidden, self.W_down, self.b_down)

    def sparsify(self):
        """Convert weight matrices to COO sparse format."""
        self.W_up = nn.Parameter(self.W_up.data.to_sparse_coo().coalesce())
        self.W_gate = nn.Parameter(self.W_gate.data.to_sparse_coo().coalesce())
        self.W_down = nn.Parameter(self.W_down.data.to_sparse_coo().coalesce())

    def compact(self, block_size=1):
        """Prune to dense sub-matrices of only active hidden units.

        Identifies hidden units where W_up, W_gate, or W_down have non-zero
        weights, keeps only those rows/columns, and stores dense sub-matrices.

        Args:
            block_size: Align active units to blocks of this size. block_size=1
                gives minimal compaction. Larger values (e.g., 32, 64) include
                padding zeros but give better vectorization and memory alignment.

        Typically reduces 4096 hidden → ~500-2000 active, giving 2-8x speedup
        over sparse COO and better cache locality than sparse ops.
        """
        # Densify if currently sparse
        W_up = self.W_up.data.to_dense() if self.W_up.is_sparse else self.W_up.data
        W_gate = self.W_gate.data.to_dense() if self.W_gate.is_sparse else self.W_gate.data
        W_down = self.W_down.data.to_dense() if self.W_down.is_sparse else self.W_down.data
        H = W_up.shape[0]

        # Find active hidden units (any non-zero weight in up, gate, or down)
        active = (
            (W_up.abs().sum(dim=1) > 0)
            | (W_gate.abs().sum(dim=1) > 0)
            | (self.b_up.data.abs() > 0)
            | (self.b_gate.data.abs() > 0)
            | (W_down.abs().sum(dim=0) > 0)
        )

        if block_size > 1:
            # Expand active mask to full blocks
            active_blocks = set()
            for idx in active.nonzero(as_tuple=True)[0].tolist():
                active_blocks.add(idx // block_size)
            indices = []
            for blk in sorted(active_blocks):
                start = blk * block_size
                indices.extend(range(start, min(start + block_size, H)))
            active_idx = torch.tensor(indices, dtype=torch.long) if indices else torch.tensor([0], dtype=torch.long)
        else:
            active_idx = active.nonzero(as_tuple=True)[0]
            if len(active_idx) == 0:
                active_idx = torch.tensor([0], dtype=torch.long)

        n = len(active_idx)
        self._compact_size = n
        self.hidden_dim = n

        # Extract dense sub-matrices: [n, dim] and [dim, n]
        self.W_up = nn.Parameter(W_up[active_idx].contiguous())
        self.b_up = nn.Parameter(self.b_up.data[active_idx].contiguous())
        self.W_gate = nn.Parameter(W_gate[active_idx].contiguous())
        self.b_gate = nn.Parameter(self.b_gate.data[active_idx].contiguous())
        self.W_down = nn.Parameter(W_down[:, active_idx].contiguous())

    def _bake_weights(self):
        """Override to bake operation-specific weights."""
        pass


class PureAttention(nn.Module):
    """
    Pure Attention with FINAL forward pass.

    Subclasses ONLY override _bake_weights().
    Used for carry propagation between nibble positions.

    KV eviction (Phase 7.F.2)
    -------------------------
    ``eviction_state`` is an optional :class:`KVEvictionState` attached
    by the compiler when ``--kv-eviction-policy=static_liveness`` is
    selected. ``None`` (the default) preserves byte-identity with all
    historical baselines: the forward pass detects the absence of state
    and skips the eviction hook entirely. The hook itself is also a
    no-op when this module has no live KV cache (the standard
    ``PureAttention.forward`` below recomputes K/V from the residual on
    every call); the state is still consulted so that determinism gates
    in tests can observe the same decisions across spec-decode and
    main-decode paths.

    Position encoding (Phase 8.O.1)
    -------------------------------
    ``position_encoding`` selects how positional information enters the
    attention scores:

    * ``"alibi"`` (default) — preserves the historical behaviour exactly.
      No RoPE rotation is applied; positional bias comes solely from
      ``self.mask`` (which child classes may bake with ALiBi-style
      slopes or leave as zeros). This branch is byte-identical to every
      pre-8.O.1 baseline.
    * ``"rope"`` — applies standard rotary position embeddings to Q and
      K before computing scores. Uses the same convention as
      :func:`neural_vm.vm_step.precompute_rope_cache` /
      :func:`apply_rotary_emb`, which mirrors
      ``transformers.models.llama.modeling_llama``. Requires an even
      ``head_dim``. ``rope_base`` selects the frequency base
      (default 10000.0).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        causal: bool = True,
        position_encoding: Literal["alibi", "rope"] = "alibi",
        rope_base: float = 10000.0,
        rope_max_seq_len: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        if position_encoding not in {"alibi", "rope"}:
            raise ValueError(
                "position_encoding must be one of {'alibi', 'rope'}"
            )
        self.position_encoding = position_encoding
        self.rope_base = float(rope_base)

        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))

        # Default mask is zeros (no masking). Child classes override with custom masks.
        self.register_buffer('mask', torch.zeros(E.NUM_POSITIONS, E.NUM_POSITIONS))

        # RoPE cache. Only allocated when position_encoding == "rope" so
        # that the ALiBi default path remains byte-identical to the
        # pre-8.O.1 module (no extra buffers, no extra ops).
        if position_encoding == "rope":
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"RoPE requires an even head_dim, got {self.head_dim} "
                    f"(dim={dim}, num_heads={num_heads})"
                )
            max_seq_len = (
                int(rope_max_seq_len)
                if rope_max_seq_len is not None
                else int(E.NUM_POSITIONS)
            )
            cos, sin = precompute_rope_cache(
                self.head_dim, max_seq_len, base=self.rope_base
            )
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)
            self._rope_max_seq_len = max_seq_len
        else:
            self._rope_cos = None
            self._rope_sin = None
            self._rope_max_seq_len = 0

        # Phase 7.F.2: precomputed runtime eviction state. Compiler
        # attaches a ``KVEvictionState`` when the eviction policy is on;
        # remains ``None`` otherwise (the byte-identity baseline).
        self.eviction_state = None
        # Track step index for the optional step-boundary eviction hook.
        # The runtime caller bumps this by overwriting the attribute or
        # by passing ``step_idx`` to :meth:`run_eviction_hook`.
        self._eviction_step_idx = 0

        self._bake_weights()

    def _extend_rope_cache(self, new_max_seq_len: int) -> None:
        """Grow the RoPE cache to cover ``new_max_seq_len`` positions.

        No-op for the ``"alibi"`` path (so ALiBi callers never pay any
        cost). Also a no-op when the existing cache is already large
        enough, matching :meth:`AutoregressiveAttention._extend_rope_cache`.
        """
        if self._rope_cos is None:
            return
        if new_max_seq_len <= self._rope_max_seq_len:
            return
        cos_new, sin_new = precompute_rope_cache(
            self.head_dim,
            new_max_seq_len,
            base=self.rope_base,
            device=self._rope_cos.device,
        )
        self.register_buffer("_rope_cos", cos_new, persistent=False)
        self.register_buffer("_rope_sin", sin_new, persistent=False)
        self._rope_max_seq_len = int(new_max_seq_len)

    def run_eviction_hook(self, step_idx: int | None = None) -> int:
        """Apply the attached :class:`KVEvictionState` at step boundary.

        Safe to call regardless of whether ``eviction_state`` is set —
        when it's ``None`` or ``KVEvictionPolicy.OFF`` the call is a
        zero-cost no-op and returns ``0`` rows zeroed.

        Returns the number of cache rows zeroed (always 0 for the
        default cache-less ``PureAttention.forward``).
        """

        state = getattr(self, "eviction_state", None)
        if state is None:
            return 0
        if step_idx is None:
            step_idx = int(getattr(self, "_eviction_step_idx", 0))
        # Lazy import to avoid a hard dependency at module import time
        # (``kv_eviction`` may import IR types that haven't loaded yet).
        from .kv_eviction import apply_eviction

        return apply_eviction(self, state, step_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - Standard attention. DO NOT OVERRIDE."""
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        Q = F.linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)
        K = F.linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
        V = F.linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)

        # Phase 8.O.1: RoPE rotation on Q/K. ALiBi path skips this
        # branch entirely and remains byte-identical to the baseline.
        if self._rope_cos is not None:
            if S > self._rope_max_seq_len:
                self._extend_rope_cache(S)
            cos = self._rope_cos[:S].unsqueeze(0).unsqueeze(0)  # [1, 1, S, HD]
            sin = self._rope_sin[:S].unsqueeze(0).unsqueeze(0)  # [1, 1, S, HD]
            Q = (Q * cos) + (rotate_half(Q) * sin)
            K = (K * cos) + (rotate_half(K) * sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Always add mask (mask is zeros when no masking needed)
        scores = scores + self.mask[:S, :S]

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        result = x + F.linear(out, self.W_o)

        # Phase 7.F.2 step-boundary hook. The hook is a no-op when no
        # eviction state is attached, preserving byte-identity with the
        # baseline. When attached, the hook consults the precomputed
        # ``KVEvictionState`` (same decisions across spec/main decode).
        if getattr(self, "eviction_state", None) is not None:
            self.run_eviction_hook()

        return result

    def _bake_weights(self):
        """Override to bake attention weights."""
        pass


class FlattenedPureFFN(nn.Module):
    """
    Wrapper that flattens input, applies a PureFFN, and unflattens output.

    For cross-position operations (reading from one position, writing to another).
    Uses composition - the inner ffn is a standard PureFFN with no forward override.

    Subclasses override _bake_weights() to set weights via self.ffn.W_up, etc.
    Use _flat_idx(pos, slot) to compute indices into the flattened dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dim = E.DIM
        self.flat_dim = E.NUM_POSITIONS * E.DIM
        self.hidden_dim = hidden_dim
        # Use standard PureFFN on flattened dimension
        self.ffn = PureFFN(dim=self.flat_dim, hidden_dim=hidden_dim)
        # Call subclass bake_weights after ffn is created
        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        """Get flattened index for position and slot."""
        return pos * self.dim + slot

    # Convenience properties to access inner FFN weights
    @property
    def W_up(self):
        return self.ffn.W_up

    @property
    def b_up(self):
        return self.ffn.b_up

    @property
    def W_gate(self):
        return self.ffn.W_gate

    @property
    def b_gate(self):
        return self.ffn.b_gate

    @property
    def W_down(self):
        return self.ffn.W_down

    @property
    def b_down(self):
        return self.ffn.b_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten -> PureFFN -> Unflatten. FINAL - DO NOT OVERRIDE."""
        B, N, D = x.shape
        x_flat = x.reshape(B, 1, N * D)  # [B, 1, flat_dim] - single "position"
        y_flat = self.ffn(x_flat)
        return y_flat.reshape(B, N, D)

    def _bake_weights(self):
        """Override to bake operation-specific weights."""
        pass
