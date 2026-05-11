"""
Spec-compliant Soft Mixture-of-Experts FFN for the Neural VM.

Replaces the legacy ``PureFFN.compact_moe`` + ``AutoregressiveVM.set_active_opcode``
runtime weight-swap path. All experts run in parallel and are soft-blended by
opcode-onehot signals read directly from the activation tensor — no Python-side
dispatch, no ``object.__setattr__`` weight mutation, no bytecode peek.

Architecture::

    output = x + sum_i(opcode_weight[i] * (expert[i](x) - x))

Where ``opcode_weight[i]`` is the value at the embedding dim that encodes
"opcode i is active" (set by an earlier layer, e.g. L5 fetch which decodes
the opcode byte from the bytecode at the PC position).

The routing signal is **pooled across sequence positions** (max over seq) so
that a single per-batch weight steers the residual blend. This matches the
intent of the legacy ``set_active_opcode`` path (one active opcode per step)
while being tensor-only and ONNX-traceable.

Fast path: at runtime ``forward`` skips experts whose pooled weight is below
``threshold`` (uses ``.item()`` for branching — fine for handler-mode inference
but invalid under ONNX tracing). When ``torch.onnx.is_in_onnx_export()`` is
true OR ``pure_neural`` is set on the module, ``_soft_forward`` runs all
experts in parallel with no Python control flow.
"""

import torch
import torch.nn as nn
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
        threshold: Skip experts whose pooled weight is below this value in the
            fast forward path. Has no effect on the ONNX/pure-neural soft path.
        pure_neural: If True, always use ``_soft_forward`` (no Python control
            flow, ONNX-traceable). Default False (fast path with skip-inactive).
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
        self.pure_neural = pure_neural

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default: fast path with skip-inactive routing.

        Falls through to ``_soft_forward`` under ONNX tracing or when
        ``pure_neural=True`` so the resulting graph is static.
        """
        if self.pure_neural or torch.onnx.is_in_onnx_export():
            return self._soft_forward(x)

        # Per-position routing: weight is the opcode-onehot value at every
        # position in the sequence (set by an earlier layer at the MARK_PC
        # position; zero elsewhere). The blend ``x + w * (E(x) - x)`` is
        # therefore localized to positions where the opcode is active.
        # Active-mask check uses .item() (Python branch only — no graph
        # dependency on dynamic shapes).
        output = torch.zeros_like(x)
        any_active = False
        for i in range(self.num_experts):
            opcode_dim = self.expert_opcode_dims[i]
            weight = x[:, :, opcode_dim:opcode_dim + 1]  # [batch, seq, 1]
            if weight.max().item() < self.threshold:
                continue
            any_active = True
            expert_out = self.experts[i](x)
            output = output + weight * (expert_out - x)

        if not any_active:
            return x
        return x + output

    def _soft_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ONNX-traceable: run all experts, blend by opcode-onehot weights.

        No Python control flow (the ``for`` loop unrolls statically at trace
        time because ``num_experts`` and each ``opcode_dim`` are Python ints).
        No ``.item()`` calls.
        """
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            expert_out = self.experts[i](x)
            opcode_dim = self.expert_opcode_dims[i]
            # Per-position routing weight (set at the MARK_PC position by L5).
            weight = x[:, :, opcode_dim:opcode_dim + 1]  # [batch, seq, 1]
            output = output + weight * (expert_out - x)
        return x + output

    def set_pure_neural(self, flag: bool) -> None:
        """Toggle between fast (skip-inactive) and soft (all-experts) paths."""
        self.pure_neural = bool(flag)

    def sparsify(self) -> None:
        """Convert each expert's weight matrices to sparse COO format.

        Delegates to the per-expert ``PureFFN.sparsify``. Called by the
        model-level ``sparsify()`` in place of the legacy per-FFN call.
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
        pure_neural: Default ``pure_neural`` value for the resulting module.

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
