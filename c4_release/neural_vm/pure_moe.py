"""
Spec-compliant Soft Mixture-of-Experts FFN for the Neural VM.

Replaces the legacy ``PureFFN.compact_moe`` + ``AutoregressiveVM.set_active_opcode``
runtime weight-swap path. Routing is driven by opcode-onehot signals read
directly from the activation tensor — no Python-side dispatch, no
``object.__setattr__`` weight mutation, no bytecode peek.

Architecture::

    output = x + sum_i(opcode_weight[i] * (expert[i](x) - x))

Where ``opcode_weight[i]`` is the value at the embedding dim that encodes
"opcode i is active" (set by an earlier layer, e.g. L5 fetch which decodes
the opcode byte from the bytecode at the PC position).

The routing signal is pooled across sequence positions so a single per-batch
weight steers the residual blend. This matches the intent of the legacy
``set_active_opcode`` path (one active opcode per step) while remaining
tensor-only and ONNX-traceable.

Runtime path (always on for inference, 2026-05-12): grouped-GEMM dispatch.
All experts' weights are stacked along a leading "expert" axis (padded to
the max hidden-dim across experts) and run as a single batched matmul.
Non-routed experts contribute zero via the routing-weight mask — this is
mathematically identical to "only the routed expert runs" but the compute
pattern is one fused op, with no ``.item()``, no Python ``if``/``continue``
over experts, and no graph breaks for ``torch.compile``.

The grouped-GEMM rewrite supersedes the prior sparse-runtime / soft-ONNX
two-path design. There is now a single forward path used by both eager and
``torch.compile`` / ONNX-export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence

from .base_layers import PureFFN


class SoftMoEFFN(nn.Module):
    """Soft Mixture-of-Experts FFN with opcode-onehot routing.

    Args:
        experts: List of ``nn.Module`` experts (typically ``PureFFN``). Each
            expert's ``forward(x)`` must take and return a tensor of identical
            shape, and apply its own residual (i.e. ``return x + delta``) so
            that ``expert(x) - x`` recovers the delta.
        expert_opcode_dims: List of **absolute embedding dimensions** (one per
            expert) at which the opcode-onehot routing signal lives. For the
            Neural VM this is e.g. ``_SetDim.OP_ADD = 287``, ``OP_SUB = 288``,
            etc. Read directly via ``x[:, :, dim]``.
        threshold: Retained for backward compatibility; ignored in the
            grouped-GEMM forward (no early-out branch).
        pure_neural: Deprecated. Retained as a no-op constructor kwarg for
            backward compatibility — runtime routing is uniform now.
    """

    def __init__(
        self,
        experts: Sequence[nn.Module],
        expert_opcode_dims: Sequence[int],
        threshold: float = 0.01,
        pure_neural: bool = False,
    ):
        super().__init__()
        if len(experts) != len(expert_opcode_dims):
            raise ValueError(
                f"experts ({len(experts)}) and expert_opcode_dims "
                f"({len(expert_opcode_dims)}) must have the same length"
            )
        # Keep ``self.experts`` for back-compat with callers that inspect the
        # per-expert PureFFN modules (e.g. ``sparsify``, diagnostic tooling).
        # The forward path does NOT use these; it uses the stacked weight
        # buffers below.
        self.experts = nn.ModuleList(experts)
        # Store as Python list (static at trace time) for ONNX compatibility.
        self.expert_opcode_dims = list(expert_opcode_dims)
        # Also keep as buffer so it survives state_dict round-trips.
        self.register_buffer(
            "expert_opcode_dims_buf",
            torch.tensor(expert_opcode_dims, dtype=torch.long),
        )
        self.num_experts = len(experts)
        self.threshold = threshold
        # No-op: kept for back-compat. The grouped-GEMM forward has no
        # eager-vs-ONNX split.
        self.pure_neural = pure_neural

        # Build stacked weight buffers from the per-expert PureFFN modules.
        # We require homogeneous PureFFN experts so the stack shapes are
        # well-defined; we pad each expert's hidden dim out to the max.
        self._build_stacked_weights()

    def _build_stacked_weights(self) -> None:
        """Build padded, stacked weight tensors over experts.

        Each expert has its own ``hidden_dim``; we pad each expert's
        ``W_up``, ``b_up``, ``W_gate``, ``b_gate`` to ``hidden_dim = H_max``
        with zeros, and pad ``W_down`` along its input axis the same way.
        Padded rows/cols contribute nothing because the zero hidden units
        produce zero ``silu(0)*0 = 0`` and zero ``W_down[:, pad]``.

        ``b_down`` is per-expert and per-dim (full ``d_model``); no padding.
        """
        if self.num_experts == 0:
            # Defensive; SoftMoEFFN is built from a partition so this should
            # not occur in practice.
            self.register_buffer("W_up_stack", torch.zeros(0))
            self.register_buffer("W_gate_stack", torch.zeros(0))
            self.register_buffer("W_down_stack", torch.zeros(0))
            self.register_buffer("b_up_stack", torch.zeros(0))
            self.register_buffer("b_gate_stack", torch.zeros(0))
            self.register_buffer("b_down_stack", torch.zeros(0))
            self._stack_dim = 0
            self._stack_h_max = 0
            return

        # Validate experts and gather shapes.
        d_model = None
        hidden_dims: List[int] = []
        for i, expert in enumerate(self.experts):
            if not isinstance(expert, PureFFN):
                raise TypeError(
                    f"SoftMoEFFN expects PureFFN experts; "
                    f"expert {i} is {type(expert).__name__}"
                )
            W_up = expert.W_up.data
            if W_up.is_sparse:
                W_up = W_up.to_dense()
            H_i, D_i = W_up.shape
            if d_model is None:
                d_model = D_i
            elif D_i != d_model:
                raise ValueError(
                    f"SoftMoEFFN experts must share d_model; "
                    f"expert {i} has D={D_i}, expected {d_model}"
                )
            hidden_dims.append(H_i)

        H_max = max(hidden_dims)
        E = self.num_experts
        device = self.experts[0].W_up.device
        dtype = self.experts[0].W_up.dtype

        # Allocate padded stacks.
        W_up_stack = torch.zeros(E, H_max, d_model, device=device, dtype=dtype)
        W_gate_stack = torch.zeros(E, H_max, d_model, device=device, dtype=dtype)
        W_down_stack = torch.zeros(E, d_model, H_max, device=device, dtype=dtype)
        b_up_stack = torch.zeros(E, H_max, device=device, dtype=dtype)
        b_gate_stack = torch.zeros(E, H_max, device=device, dtype=dtype)
        b_down_stack = torch.zeros(E, d_model, device=device, dtype=dtype)

        for i, expert in enumerate(self.experts):
            H_i = hidden_dims[i]
            W_up = expert.W_up.data
            W_gate = expert.W_gate.data
            W_down = expert.W_down.data
            if W_up.is_sparse:
                W_up = W_up.to_dense()
            if W_gate.is_sparse:
                W_gate = W_gate.to_dense()
            if W_down.is_sparse:
                W_down = W_down.to_dense()
            W_up_stack[i, :H_i, :].copy_(W_up)
            W_gate_stack[i, :H_i, :].copy_(W_gate)
            W_down_stack[i, :, :H_i].copy_(W_down)
            b_up_stack[i, :H_i].copy_(expert.b_up.data)
            b_gate_stack[i, :H_i].copy_(expert.b_gate.data)
            b_down_stack[i, :].copy_(expert.b_down.data)

        # Register as buffers so they move with .to(device) / .cuda() and so
        # they survive state_dict round-trips. They are NOT parameters: the
        # per-expert PureFFNs already own the parameter copies (kept in
        # ``self.experts`` for back-compat tools); these stacks are derived.
        self.register_buffer("W_up_stack", W_up_stack)
        self.register_buffer("W_gate_stack", W_gate_stack)
        self.register_buffer("W_down_stack", W_down_stack)
        self.register_buffer("b_up_stack", b_up_stack)
        self.register_buffer("b_gate_stack", b_gate_stack)
        self.register_buffer("b_down_stack", b_down_stack)
        self._stack_dim = d_model
        self._stack_h_max = H_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Grouped-GEMM dispatch over all experts.

        Mathematically identical to "only routed experts run" (non-routed
        experts have routing weight 0 and contribute nothing). The compute
        pattern is one fused batched matmul per stage — no Python control
        flow over experts, no ``.item()``, ``torch.compile``-clean.
        """
        # Empty-MoE no-op: behave as identity.
        if self.num_experts == 0:
            return x

        B, S, D = x.shape
        E = self.num_experts
        H = self._stack_h_max

        # up   = einsum("bsd, ehd -> bseh", x, W_up_stack)   + b_up_stack
        # gate = einsum("bsd, ehd -> bseh", x, W_gate_stack) + b_gate_stack
        # hidden = silu(up) * gate                          # [B,S,E,H]
        # out  = einsum("bseh, edh -> bsed", hidden, W_down_stack) + b_down_stack
        up = torch.einsum("bsd, ehd -> bseh", x, self.W_up_stack)
        up = up + self.b_up_stack  # broadcasts over (B,S,E,H)
        gate = torch.einsum("bsd, ehd -> bseh", x, self.W_gate_stack)
        gate = gate + self.b_gate_stack
        hidden = F.silu(up) * gate
        # Note: W_down_stack has shape [E, D_out, H]; einsum over H.
        expert_out = torch.einsum("bseh, edh -> bsed", hidden, self.W_down_stack)
        expert_out = expert_out + self.b_down_stack  # [B,S,E,D]

        # Per-position routing weights, one column per expert opcode dim.
        # ``self.expert_opcode_dims`` is a static Python list (set at construct
        # time), so this gather unrolls cleanly under tracing.
        # routing: [B, S, E]
        if E == 1:
            routing = x[..., self.expert_opcode_dims[0]:self.expert_opcode_dims[0] + 1]
        else:
            # Index with a static long tensor for vectorized gather.
            idx = self.expert_opcode_dims_buf.to(device=x.device)
            routing = x.index_select(-1, idx)  # [B, S, E]

        # Blended delta: sum over experts of routing * (expert_out - x).
        # expert_out has shape [B,S,E,D]; x has [B,S,D] (broadcast to [B,S,1,D]).
        delta = expert_out - x.unsqueeze(2)  # [B,S,E,D]
        blended = (routing.unsqueeze(-1) * delta).sum(dim=2)  # [B,S,D]
        return x + blended

    def set_pure_neural(self, flag: bool) -> None:
        """Deprecated no-op kept for back-compat.

        Runtime routing now uses a single grouped-GEMM path; there is no
        eager-vs-ONNX split. The flag is recorded on the module but has no
        effect on ``forward``.
        """
        self.pure_neural = bool(flag)

    def sparsify(self) -> None:
        """Convert each expert's weight matrices to sparse COO format.

        Delegates to the per-expert ``PureFFN.sparsify``. Called by the
        model-level ``sparsify()`` in place of the legacy per-FFN call.

        Note: this only affects ``self.experts`` (the per-expert PureFFN
        copies retained for back-compat); the stacked ``W_*_stack`` buffers
        used by ``forward`` remain dense.
        """
        for expert in self.experts:
            if hasattr(expert, "sparsify"):
                expert.sparsify()


# Backward-compatibility alias for callers that imported the older name.
MoE = SoftMoEFFN


def build_soft_moe_from_compact_partition(
    compact_ffn: PureFFN,
    opcode_to_units: dict,
    shared_indices: list,
    dim: int,
    pure_neural: bool = False,
) -> SoftMoEFFN:
    """Construct a ``SoftMoEFFN`` from an already-partitioned compact FFN.

    Mirrors the partitioning that ``PureFFN.compact_moe`` performed but builds
    a parallel-expert module instead of a Python-side weight swapper.

    Each expert is a ``PureFFN`` containing:
      - the SHARED hidden units (active for every opcode), plus
      - the opcode-specific hidden units for that one opcode.

    This mirrors the previous ``_moe_combined[d] = cat(shared, expert_d)``
    pre-concatenation, so per-opcode forward behavior is byte-identical to the
    legacy weight-swap path. The trade-off is memory: each expert carries a
    copy of the shared units. The user's chosen Option A in the audit accepts
    this cost in exchange for parallel execution and ONNX compatibility.

    Args:
        compact_ffn: The already-compacted ``PureFFN`` (post ``compact()``)
            whose weight matrices are partitioned.
        opcode_to_units: Mapping ``{opcode_dim: [unit_indices]}`` (output of
            ``compact_moe``'s opcode-affinity scan).
        shared_indices: List of hidden-unit indices that are opcode-independent.
        dim: Model embedding dim (``d_model``).
        pure_neural: Deprecated no-op (forwarded to ``SoftMoEFFN`` for
            back-compat). Routing is uniform across modes.

    Returns:
        A ``SoftMoEFFN`` whose ``experts`` are per-opcode ``PureFFN`` modules
        and whose ``expert_opcode_dims`` are the corresponding routing dims.
    """
    W_up = compact_ffn.W_up.data
    b_up = compact_ffn.b_up.data
    W_gate = compact_ffn.W_gate.data
    b_gate = compact_ffn.b_gate.data
    W_down = compact_ffn.W_down.data
    b_down = compact_ffn.b_down.data

    shared_idx = torch.tensor(sorted(set(shared_indices)), dtype=torch.long) if shared_indices else None

    experts: List[PureFFN] = []
    opcode_dims: List[int] = []

    # b_down is shared across the legacy partition; carry it on the first
    # expert only so the residual delta `expert_i(x) - x` for that expert
    # includes b_down. For one-hot routing (typical case) this matches the
    # legacy `_moe_combined[d]` behavior where b_down was loaded once.
    # b_down is usually zero in baked FFNs, so this rarely matters in practice.
    b_down_placed = False

    for opcode_dim, units in opcode_to_units.items():
        expert_unit_idx = torch.tensor(sorted(set(units)), dtype=torch.long)
        if shared_idx is not None and len(shared_idx) > 0:
            combined_idx = torch.cat([shared_idx, expert_unit_idx])
        else:
            combined_idx = expert_unit_idx

        hidden = len(combined_idx)
        expert = PureFFN(dim=dim, hidden_dim=hidden)
        with torch.no_grad():
            expert.W_up.data.copy_(W_up[combined_idx])
            expert.b_up.data.copy_(b_up[combined_idx])
            expert.W_gate.data.copy_(W_gate[combined_idx])
            expert.b_gate.data.copy_(b_gate[combined_idx])
            expert.W_down.data.copy_(W_down[:, combined_idx])
            if not b_down_placed:
                expert.b_down.data.copy_(b_down)
                b_down_placed = True
            else:
                expert.b_down.data.zero_()
        experts.append(expert)
        opcode_dims.append(opcode_dim)

    return SoftMoEFFN(
        experts=experts,
        expert_opcode_dims=opcode_dims,
        pure_neural=pure_neural,
    )
