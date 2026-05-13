"""
Standard top-K Mixture-of-Experts FFN for the Neural VM.

Replaces the previous Soft-MoE-style parallel-blending implementation with
a Mixtral/DeepSeek/Qwen-style top-K MoE: only the K selected experts run
per token, gated by the router. For C4 we use ``top_k=1`` because exactly
one opcode is active per step (the OP_* one-hot dim selects the expert).

Architecture::

    shared_out = shared_ffn(x)                          # always runs
    router_logits = x[..., expert_opcode_dims]          # [B,S,E]
    topk_vals, topk_idx = router_logits.topk(K, -1)     # [B,S,K]
    gate = topk_vals                                    # raw one-hot gating
    expert_out = sum_k(gate_k * dispatch(expert[idx_k], x))
    return x + (shared_out - x) + (expert_out - x_zero_baseline)

Why "raw one-hot gating" (gate=topk_vals) instead of softmax-renorm:
    The OP_* dims are exact one-hot (0.0 or 1.0). When a position has an
    active opcode (e.g. MARK_PC), exactly one router logit is 1.0 and the
    rest are 0; top-1 selects it with gate=1.0 → the expert contributes
    fully (matching the dense path's opcode-onehot-gated activation).
    When NO opcode is active (every non-MARK_PC position), all logits are
    0.0; top-1 picks an arbitrary expert with gate=0.0 → that expert
    contributes nothing (matching the dense path, where the opcode-specific
    units self-gate to zero via their opcode-onehot input weight).

    A standard softmax over [0,0,...,0] yields uniform [1/E,...,1/E] which
    would *contaminate* every non-MARK_PC position. The raw-gate variant
    preserves byte-identity with the dense-FFN path; opcode-onehot routing
    *is* the gate.

Shared vs. routed units:
    Per the legacy ``compact_moe`` partition, hidden units fall into two
    classes:
      - opcode-INDEPENDENT (shared) — fire at every position
      - opcode-SPECIFIC (per-expert) — fire only when their opcode is active
    The shared units are pulled out into a small ``shared_ffn`` (a PureFFN)
    that runs unconditionally. Each expert holds only its opcode-specific
    units. This mirrors the DeepSeek shared-expert pattern and matches the
    dense FFN's contribution at every position. ``b_down`` is owned by the
    shared FFN (one application; experts carry zero ``b_down``).

Forward implementation:
    We compute all experts' outputs via a batched (grouped-GEMM) matmul,
    then multiply by a one-hot top-K mask to retain only the K selected
    experts' contributions. Mathematically this IS "only the routed experts
    run" because non-selected experts get multiplied by zero. The compute
    pattern is one fused batched op (no Python loops over experts, no
    ``.item()``), ``torch.compile`` / ONNX-trace clean.

Backward-compatibility:
    - Class name ``SoftMoEFFN`` is retained (this module supersedes the
      original Soft-MoE; the name is historical).
    - ``MoE`` is an alias for ``SoftMoEFFN``.
    - ``build_soft_moe_from_compact_partition(...)`` still constructs the
      MoE from a compacted PureFFN partition; the API is unchanged.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_layers import PureFFN


class SoftMoEFFN(nn.Module):
    """Standard top-K transformer MoE with opcode-onehot routing.

    Args:
        experts: List of ``nn.Module`` per-opcode experts (each holding ONLY
            that opcode's specific hidden units; shared units live in
            ``shared_ffn`` if provided). Each expert is a ``PureFFN`` so its
            forward is ``x + delta_expert``; we subtract ``x`` to recover
            the delta when blending.
        expert_opcode_dims: Per-expert absolute residual-stream dim that
            carries the OP_* one-hot routing signal (e.g. dim 287 = OP_ADD).
            Read directly as ``x[..., d]`` for the router.
        shared_ffn: Optional ``PureFFN`` containing the opcode-INDEPENDENT
            hidden units. Runs unconditionally on every position. When
            ``None``, no shared-expert path is applied (legacy behavior;
            useful for partitions with zero shared units).
        top_k: Number of experts selected per token. Defaults to 1 because
            C4 dispatches exactly one opcode per VM step.
        threshold: Retained for back-compat; ignored.
        pure_neural: Retained for back-compat; ignored. The forward path is
            uniform between eager and ``torch.compile`` / ONNX modes.
    """

    def __init__(
        self,
        experts: Sequence[nn.Module],
        expert_opcode_dims: Sequence[int],
        shared_ffn: PureFFN = None,
        top_k: int = 1,
        threshold: float = 0.01,
        pure_neural: bool = False,
    ):
        super().__init__()
        if len(experts) != len(expert_opcode_dims):
            raise ValueError(
                f"experts ({len(experts)}) and expert_opcode_dims "
                f"({len(expert_opcode_dims)}) must have the same length"
            )
        # Retain ``self.experts`` as an ``nn.ModuleList`` so external tools
        # (sparsify, the runtime-vanilla audit, diagnostic walkers) keep
        # finding the per-expert PureFFNs.
        self.experts = nn.ModuleList(experts)
        self.expert_opcode_dims = list(expert_opcode_dims)
        self.register_buffer(
            "expert_opcode_dims_buf",
            torch.tensor(expert_opcode_dims, dtype=torch.long),
        )
        self.num_experts = len(experts)
        self.top_k = int(top_k)
        if self.top_k < 1 or self.top_k > self.num_experts:
            # Top-K must be in [1, E]; clamp to a sane range. Most callers
            # use top_k=1 (C4's natural per-step dispatch).
            self.top_k = max(1, min(int(top_k), self.num_experts))
        self.threshold = threshold
        self.pure_neural = pure_neural

        # Optional always-on shared FFN (DeepSeek-style shared expert).
        # We hold ``None`` if absent so PyTorch doesn't register a dummy
        # sub-module; callers that want a shared path pass one explicitly
        # from ``build_soft_moe_from_compact_partition``.
        self.shared_ffn = shared_ffn  # nn.Module | None
        self._has_shared = shared_ffn is not None

        # Stacked weight buffers for the grouped-GEMM all-experts compute
        # (we mask down to top-K post-hoc). One forward = one batched matmul
        # over all experts; the top-K mask zeros non-selected experts.
        self._build_stacked_weights()

    def _build_stacked_weights(self) -> None:
        """Pack each expert's ``W_up`` / ``W_gate`` / ``W_down`` plus biases
        into a batched-expert tensor of shape ``[E, H_max, D]`` (or its
        transpose for ``W_down``). Experts with hidden_dim < H_max are
        zero-padded; padded rows contribute zero because ``silu(0)*0=0``
        and ``W_down[:, pad]=0``.
        """
        if self.num_experts == 0:
            self.register_buffer("W_up_stack", torch.zeros(0))
            self.register_buffer("W_gate_stack", torch.zeros(0))
            self.register_buffer("W_down_stack", torch.zeros(0))
            self.register_buffer("b_up_stack", torch.zeros(0))
            self.register_buffer("b_gate_stack", torch.zeros(0))
            self.register_buffer("b_down_stack", torch.zeros(0))
            self._stack_dim = 0
            self._stack_h_max = 0
            return

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

        H_max = max(hidden_dims) if hidden_dims else 0
        E = self.num_experts
        device = self.experts[0].W_up.device
        dtype = self.experts[0].W_up.dtype

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

        self.register_buffer("W_up_stack", W_up_stack)
        self.register_buffer("W_gate_stack", W_gate_stack)
        self.register_buffer("W_down_stack", W_down_stack)
        self.register_buffer("b_up_stack", b_up_stack)
        self.register_buffer("b_gate_stack", b_gate_stack)
        self.register_buffer("b_down_stack", b_down_stack)
        self._stack_dim = d_model
        self._stack_h_max = H_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Top-K MoE forward (sparse per-expert dispatch).

        For each token position, only the ``top_k`` experts selected by
        ``router_logits = x[..., expert_opcode_dims]`` contribute. Gates
        are the raw one-hot routing values (not softmax-renormalized): an
        active opcode yields gate=1.0 for the selected expert; a position
        where the OP_* signal is exactly 0 yields gate=0.0 (the expert
        runs on no tokens — the Python-side ``mask.any()`` early-out
        skips the gather entirely).

        Implementation: standard sparse Mixtral-style dispatch. The
        top-K routing tensor is computed once via ``torch.topk``; then
        for each expert we (a) build a boolean ``mask`` of token slots
        routed to it, (b) gather only those tokens, (c) run that expert
        on the sub-batch, (d) scatter the gated delta back.

        Memory: peak intermediate is ``[N_e, H_e]`` per expert (only the
        tokens routed to that expert) rather than ``[B*S, E, H_max]`` for
        a grouped-GEMM. This matters for the production model where some
        blocks have ~30 experts and ~2000-token sequences — the grouped
        form materialises multi-GB tensors per layer.

        Byte-identity note: This MoE does NOT byte-match the dense
        compacted FFN. The legacy ``_partition_compact_ffn_by_opcode``
        partition uses ``W_up[i, OP_dim] > 0.5`` (plus a CMP relay map)
        to label hidden units as opcode-specific, but those units still
        receive substantial contributions from other input dims at
        non-MARK_PC positions in the dense path. Routing those units to
        an opcode-gated expert drops their non-MARK_PC contribution.
        See ``c4_release/docs/MOE_ROUTING_AUDIT.md`` for the discussion.
        """
        # Always-on shared expert path runs first; it handles the
        # opcode-independent units and ``b_down`` at every position.
        if self._has_shared:
            out = self.shared_ffn(x)  # = x + delta_shared (PureFFN residual)
        else:
            out = x

        if self.num_experts == 0:
            return out

        B, S, D = x.shape
        E = self.num_experts
        K = self.top_k

        # Router logits: one column per expert opcode dim. We read from
        # the ORIGINAL ``x`` (not ``out``) so the router sees the
        # unmodified residual stream including the OP_* one-hot.
        idx = self.expert_opcode_dims_buf.to(device=x.device)
        router_logits = x.index_select(-1, idx)  # [B,S,E]

        # Top-K selection. ``topk_vals`` IS the gate (raw one-hot value):
        # no softmax renormalization — see class docstring.
        topk_vals, topk_indices = router_logits.topk(K, dim=-1)  # [B,S,K]

        # Flatten (B,S) into one token axis. Sparse dispatch operates on
        # the flat (N, D) view.
        N = B * S
        x_flat = x.reshape(N, D)
        delta_flat = torch.zeros_like(x_flat)
        topk_indices_flat = topk_indices.reshape(N, K)
        topk_vals_flat = topk_vals.reshape(N, K)

        # Detect tracing-mode once (avoids repeated check inside loop).
        is_tracing = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()

        # Per-expert dispatch loop. ``num_experts`` is a Python int so
        # this unrolls at trace time. For each expert e: gather tokens
        # routed to it, run the expert, scatter the gated delta back.
        for e in range(E):
            mask_per_k = (topk_indices_flat == e)  # [N, K] bool
            mask = mask_per_k.any(dim=-1)  # [N] bool
            # Sum gates across the K slots where this expert is chosen.
            # For top-1 with one-hot routing this is exactly the raw
            # one-hot at expert e's opcode dim (0.0 or 1.0).
            gate = (mask_per_k.to(topk_vals_flat.dtype) * topk_vals_flat).sum(
                dim=-1, keepdim=True
            )  # [N, 1]
            # Python-side early-out for empty routing. Skipping when no
            # token routes to this expert is a big win in eager mode
            # (avoids spurious 0-token expert calls). Under tracing we
            # always run the full body for graph-static behavior.
            if not is_tracing and not bool(mask.any()):
                continue
            x_e = x_flat[mask]  # [N_e, D]
            gate_e = gate[mask]  # [N_e, 1]
            # Expert is a PureFFN whose forward is ``x + delta_e``; we
            # subtract ``x_e`` to recover the delta only.
            expert_out = self.experts[e](x_e.unsqueeze(0)).squeeze(0) - x_e
            delta_flat[mask] = delta_flat[mask] + gate_e * expert_out

        delta = delta_flat.reshape(B, S, D)
        return out + delta

    def set_pure_neural(self, flag: bool) -> None:
        """Deprecated no-op kept for back-compat."""
        self.pure_neural = bool(flag)

    def sparsify(self) -> None:
        """Sparsify per-expert PureFFNs (back-compat hook for model-level sparsify)."""
        for expert in self.experts:
            if hasattr(expert, "sparsify"):
                expert.sparsify()
        if self._has_shared and hasattr(self.shared_ffn, "sparsify"):
            self.shared_ffn.sparsify()


# Backward-compatibility alias.
MoE = SoftMoEFFN
# Forward-looking name. The module's behavior is now standard top-K MoE
# (Mixtral/DeepSeek/Qwen), not Soft MoE; ``StandardMoEFFN`` is the
# preferred alias for new code.
StandardMoEFFN = SoftMoEFFN


def _tighten_partition_by_no_opcode_firing(
    compact_ffn: PureFFN,
    opcode_to_units: dict,
    shared_indices: list,
    opcode_dims_all: Sequence[int],
    relay_dims: Sequence[int] = (),
    epsilon: float = 1e-4,
    batch_size: int = 1024,
    seed: int = 0,
) -> tuple:
    """Re-classify opcode-specific units that still fire without their opcode.

    Path A from ``c4_release/docs/MOE_ROUTING_AUDIT.md`` §8.3: the legacy
    ``_partition_compact_ffn_by_opcode`` labels a hidden unit as opcode-X
    specific when ``W_up[i, OP_X] > 0.5`` (plus a CMP relay map). But the
    same unit can still pick up substantial activation from OTHER input
    dims at non-MARK_PC positions. Routing it to an opcode-gated expert
    (gate=0 when OP_X is inactive) drops that contribution, breaking
    byte-identity with the dense FFN.

    This pass empirically measures each candidate unit's silu*gate
    contribution on a "no-opcode" synthetic batch (random residual values,
    OP_* and relay dims forced to zero). Units whose max contribution
    exceeds ``epsilon`` are moved to the shared expert. Clean units (those
    that truly fire only when their opcode is active) stay in their expert.

    Returns:
        ``(tightened_opcode_to_units, tightened_shared_indices)``.
    """
    W_up = compact_ffn.W_up.data
    b_up = compact_ffn.b_up.data
    W_gate = compact_ffn.W_gate.data
    b_gate = compact_ffn.b_gate.data

    D = W_up.shape[1]
    device = W_up.device
    dtype = W_up.dtype

    # Gaussian residual-stream values with OP_* and relay dims forced to
    # zero — simulates the non-MARK_PC positions where the MoE routing
    # gate is 0 but other dims still carry signal.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(batch_size, D, generator=gen).to(device=device, dtype=dtype)
    excluded_idx = sorted(
        {int(d) for d in opcode_dims_all} | {int(d) for d in relay_dims}
    )
    excluded_idx = [d for d in excluded_idx if 0 <= d < D]
    if excluded_idx:
        x.index_fill_(
            -1,
            torch.tensor(excluded_idx, dtype=torch.long, device=device),
            0.0,
        )

    # We test the hidden activation (silu*gate), not the post-W_down delta,
    # so units with small W_down columns still get flagged when they'd
    # compose with downstream layers.
    with torch.no_grad():
        up = F.linear(x, W_up, b_up)
        gate = F.linear(x, W_gate, b_gate)
        unit_max = (F.silu(up) * gate).abs().amax(dim=0).tolist()

    dirty: set = set()
    tight_opcode_to_units: dict = {}
    for opcode_dim, units in opcode_to_units.items():
        kept = [int(i) for i in units if unit_max[i] <= epsilon]
        dirty.update(int(i) for i in units if unit_max[i] > epsilon)
        if kept:
            tight_opcode_to_units[opcode_dim] = kept

    # A relay-mapped unit may have been kept under one opcode and flagged
    # dirty under another; drop those leftovers now that the full dirty
    # set is known.
    for opcode_dim in list(tight_opcode_to_units.keys()):
        cleaned = [i for i in tight_opcode_to_units[opcode_dim] if i not in dirty]
        if cleaned:
            tight_opcode_to_units[opcode_dim] = cleaned
        else:
            del tight_opcode_to_units[opcode_dim]

    tight_shared = sorted(set(shared_indices).union(dirty))
    return tight_opcode_to_units, tight_shared


def build_soft_moe_from_compact_partition(
    compact_ffn: PureFFN,
    opcode_to_units: dict,
    shared_indices: list,
    dim: int,
    pure_neural: bool = False,
    top_k: int = 1,
    tighten: bool = True,
    tighten_epsilon: float = 1e-4,
    opcode_dims_all: Sequence[int] = None,
    relay_dims: Sequence[int] = (),
) -> SoftMoEFFN:
    """Construct a top-K ``SoftMoEFFN`` from an already-partitioned compact FFN.

    Mirrors the partition produced by ``_partition_compact_ffn_by_opcode``
    and ``PureFFN.compact_moe``. Each opcode-specific unit set becomes one
    expert (holding ONLY its specific units); the opcode-independent units
    form an always-on ``shared_ffn``. ``b_down`` is carried on the shared
    FFN so it applies once per position (matching the dense path).

    Args:
        compact_ffn: Already-compacted ``PureFFN`` (post ``compact()``).
        opcode_to_units: ``{opcode_dim: [unit_indices]}`` per-opcode unit
            assignment.
        shared_indices: Indices of opcode-independent hidden units.
        dim: Model embedding dim (``d_model``).
        pure_neural: Back-compat no-op forwarded to ``SoftMoEFFN``.
        top_k: Number of experts to route per token. Defaults to 1 because
            C4 dispatches exactly one opcode per step.
        tighten: When True (default), run the Path A "no-opcode firing"
            tightening pass that re-routes any opcode-specific unit which
            still produces non-trivial output on a synthetic batch with
            all OP_* / relay dims zeroed. Units that fail this test go to
            the shared expert so their dense-path contribution at
            non-MARK_PC positions is preserved (closes the byte-identity
            gap described in ``MOE_ROUTING_AUDIT.md`` §8). Set to False
            for the legacy (non-byte-identical) partition.
        tighten_epsilon: Threshold for "non-trivial" firing. Units whose
            ``max|silu(up) * gate|`` on the no-opcode batch exceeds this
            value are reclassified as shared.
        opcode_dims_all: Full set of OP_* dims to zero on the no-opcode
            synthetic batch. Defaults to the keys of ``opcode_to_units``.
            Callers with a wider opcode range (compiler-allocated layout)
            should pass it explicitly so unused OP_* dims are also zeroed.
        relay_dims: Additional dims (e.g. CMP relay columns) to zero on
            the synthetic batch. Without this, units that depend on a CMP
            relay dim would appear "dirty" simply because the relay dim
            is non-zero in the random batch.

    Returns:
        ``SoftMoEFFN`` with one expert per opcode, optional shared FFN,
        and the routing-dim list pre-populated.
    """
    if tighten:
        if opcode_dims_all is None:
            opcode_dims_all = list(opcode_to_units.keys())
        opcode_to_units, shared_indices = _tighten_partition_by_no_opcode_firing(
            compact_ffn,
            opcode_to_units,
            shared_indices,
            opcode_dims_all=opcode_dims_all,
            relay_dims=relay_dims,
            epsilon=tighten_epsilon,
        )

    W_up = compact_ffn.W_up.data
    b_up = compact_ffn.b_up.data
    W_gate = compact_ffn.W_gate.data
    b_gate = compact_ffn.b_gate.data
    W_down = compact_ffn.W_down.data
    b_down = compact_ffn.b_down.data

    # Always-on shared expert: opcode-independent hidden units plus the
    # full ``b_down``. If the partition has no shared units, we still
    # build a tiny "bias-only" shared FFN with hidden_dim=1 and all
    # weights zero so ``b_down`` is applied once at every position.
    shared_unit_idx = sorted(set(shared_indices))
    if shared_unit_idx:
        h_shared = len(shared_unit_idx)
        shared_ffn = PureFFN(dim=dim, hidden_dim=h_shared)
        idx_t = torch.tensor(shared_unit_idx, dtype=torch.long)
        with torch.no_grad():
            shared_ffn.W_up.data.copy_(W_up[idx_t])
            shared_ffn.b_up.data.copy_(b_up[idx_t])
            shared_ffn.W_gate.data.copy_(W_gate[idx_t])
            shared_ffn.b_gate.data.copy_(b_gate[idx_t])
            shared_ffn.W_down.data.copy_(W_down[:, idx_t])
            shared_ffn.b_down.data.copy_(b_down)
    else:
        # Inject a 1-unit "dead" hidden so the PureFFN forward still
        # applies ``b_down``. ``W_up`` / ``W_gate`` are zero → ``hidden=0``
        # → ``F.linear(hidden, W_down)=0`` → ``b_down`` is the only
        # contribution.
        shared_ffn = PureFFN(dim=dim, hidden_dim=1)
        with torch.no_grad():
            shared_ffn.W_up.data.zero_()
            shared_ffn.b_up.data.zero_()
            shared_ffn.W_gate.data.zero_()
            shared_ffn.b_gate.data.zero_()
            shared_ffn.W_down.data.zero_()
            shared_ffn.b_down.data.copy_(b_down)

    # Per-opcode experts: hold ONLY that opcode's specific hidden units.
    # No shared units copy (the shared path runs them) and ``b_down``
    # is zero on each expert (the shared path applies it).
    experts: List[PureFFN] = []
    opcode_dims: List[int] = []
    for opcode_dim, units in opcode_to_units.items():
        expert_unit_idx = torch.tensor(sorted(set(units)), dtype=torch.long)
        hidden = len(expert_unit_idx)
        if hidden == 0:
            continue
        expert = PureFFN(dim=dim, hidden_dim=hidden)
        with torch.no_grad():
            expert.W_up.data.copy_(W_up[expert_unit_idx])
            expert.b_up.data.copy_(b_up[expert_unit_idx])
            expert.W_gate.data.copy_(W_gate[expert_unit_idx])
            expert.b_gate.data.copy_(b_gate[expert_unit_idx])
            expert.W_down.data.copy_(W_down[:, expert_unit_idx])
            expert.b_down.data.zero_()  # b_down lives on shared_ffn
        experts.append(expert)
        opcode_dims.append(opcode_dim)

    return SoftMoEFFN(
        experts=experts,
        expert_opcode_dims=opcode_dims,
        shared_ffn=shared_ffn,
        top_k=top_k,
        pure_neural=pure_neural,
    )
