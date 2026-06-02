"""Mixtral state_dict adapter for the Neural VM.

This module shape-translates a compiled :class:`AutoregressiveVM` into a
:class:`transformers.MixtralForCausalLM` so the resulting HF model is
runnable (forward passes don't crash, weights line up). It is NOT a
semantic-equivalence claim — the VM's per-step ADDR_KEY / MEM_STORE
embedding augmentations, ALiBi slope buffers, and RoPE caches are
dropped, and Mixtral-only structure (router gates, multiple experts,
final RMSNorm) is synthesized.

VM → Mixtral key mapping
------------------------
- ``embed.embed.weight`` → ``model.embed_tokens.weight``
- ``blocks.{i}.attn.W_q`` → ``model.layers.{i}.self_attn.q_proj.weight``
- ``blocks.{i}.attn.W_k`` → ``model.layers.{i}.self_attn.k_proj.weight``
- ``blocks.{i}.attn.W_v`` → ``model.layers.{i}.self_attn.v_proj.weight``
- ``blocks.{i}.attn.W_o`` → ``model.layers.{i}.self_attn.o_proj.weight``
- ``blocks.{i}.ffn.W_gate`` → ``model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight``
  (the VM's dense FFN gate is replicated into every expert)
- ``blocks.{i}.ffn.W_up`` → ``model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight``
- ``blocks.{i}.ffn.W_down`` → ``model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight``
- ``blocks.{i}.attn_norm.weight`` → ``model.layers.{i}.input_layernorm.weight``
- ``blocks.{i}.ffn_norm.weight`` → ``model.layers.{i}.post_attention_layernorm.weight``
- ``head.weight`` → ``lm_head.weight``

Synthesized (zeros / ones) when the VM doesn't carry them:
- ``model.layers.{i}.block_sparse_moe.gate.weight`` — zeros (uniform routing)
- ``model.layers.{i}.input_layernorm.weight`` / ``post_attention_layernorm.weight``
  — ones when the VM block has no RMSNorm enabled. When the block has
  ``use_rms_norm=True`` the actual ``attn_norm.weight`` /
  ``ffn_norm.weight`` are copied (and padded with ones to ``hidden_size``).
- ``model.norm.weight`` — ones (Mixtral has a final RMSNorm, VM does not)

Dropped from the VM side:
- ``embed.*`` buffers other than ``embed.embed.weight`` (ADDR_KEY positional
  encoding, MEM_STORE markers, etc.)
- ``blocks.{i}.attn.alibi_slopes``, ``_rope_cos``, ``_rope_sin``,
  ``_softmax1_anchor``
- FFN biases (``b_up`` / ``b_gate`` / ``b_down``) — Mixtral MLPs are
  bias-free, and the VM ships zero biases by default
- ``head.bias`` — Mixtral lm_head is bias-free
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# Default VM shape that maps cleanly into a Mixtral-shaped HF model.
# These are the dynamic-allocator defaults the rest of the repo targets:
# ``DimRegistry(d_model=736)`` / 18 layers / 8 heads / head_dim 92.
DEFAULT_MIXTRAL_VM_SHAPE: Dict[str, int] = {
    "d_model": 736,
    "n_layers": 18,
    "num_heads": 8,
    "head_dim": 92,
}


# Canonical Mixtral-8x7B geometry. Used as the default ``target_shape``
# when callers pass ``pad_to_mixtral=True`` without their own envelope.
MIXTRAL_8X7B_SHAPE: Dict[str, int] = {
    "d_model": 4096,
    "n_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,  # GQA
    "head_dim": 128,
    "ffn_hidden": 14336,
    "vocab_size": 32000,
}


class MixtralShapeMismatchError(ValueError):
    """Raised when the VM's shape can't be expressed as a Mixtral layout."""


@dataclass(frozen=True)
class _VMShape:
    vocab_size: int
    d_model: int
    n_layers: int
    num_heads: int
    head_dim: int
    ffn_hidden: int
    use_rms_norm: bool
    max_position_embeddings: int
    rope_base: float
    rms_norm_eps: float
    ffn_hidden_per_layer: List[int] = field(default_factory=list)


def _read_vm_shape(model: Any) -> _VMShape:
    """Read the shape parameters off a VM model in a tolerant way.

    The VM doesn't centralize these as attributes everywhere, so we read
    from ``model`` then fall back to the first block's modules.
    """

    blocks = list(getattr(model, "blocks", []))
    if not blocks:
        raise MixtralShapeMismatchError(
            "VM model has no .blocks; expected AutoregressiveVM-style layout."
        )
    attn0 = getattr(blocks[0], "attn", None)
    ffn0 = getattr(blocks[0], "ffn", None)
    if attn0 is None or ffn0 is None:
        raise MixtralShapeMismatchError(
            "VM model's first block missing .attn or .ffn module."
        )

    d_model = int(getattr(model, "d_model", getattr(attn0, "dim", 0)))
    num_heads = int(getattr(attn0, "num_heads", 0))
    head_dim = int(getattr(attn0, "head_dim", 0))
    if d_model <= 0 or num_heads <= 0 or head_dim <= 0:
        raise MixtralShapeMismatchError(
            f"VM model has invalid shape: d_model={d_model}, "
            f"num_heads={num_heads}, head_dim={head_dim}."
        )
    if num_heads * head_dim != d_model:
        raise MixtralShapeMismatchError(
            f"VM attention has num_heads * head_dim = {num_heads * head_dim} "
            f"but d_model = {d_model}; Mixtral requires "
            f"num_attention_heads * head_dim == hidden_size."
        )

    ffn_hidden = int(getattr(ffn0, "hidden_dim", 0))
    if ffn_hidden <= 0:
        raise MixtralShapeMismatchError(
            "VM FFN hidden_dim is 0; nothing to export."
        )
    ffn_hidden_per_layer = [int(b.ffn.hidden_dim) for b in blocks]

    vocab_size = int(getattr(model, "vocab_size", 0))
    if vocab_size <= 0:
        # Try reading from embedding layer
        emb = getattr(model, "embed", None)
        inner = getattr(emb, "embed", None)
        if inner is not None and hasattr(inner, "num_embeddings"):
            vocab_size = int(inner.num_embeddings)
    if vocab_size <= 0:
        raise MixtralShapeMismatchError("VM model has no usable vocab_size.")

    use_rms_norm = bool(getattr(model, "use_rms_norm", False))
    max_position_embeddings = int(getattr(model, "max_seq_len", 1024))
    rope_base = float(getattr(model, "rope_base", 10000.0))
    rms_norm_eps = float(getattr(model, "rms_norm_eps", 1e-6))

    return _VMShape(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=len(blocks),
        num_heads=num_heads,
        head_dim=head_dim,
        ffn_hidden=ffn_hidden,
        use_rms_norm=use_rms_norm,
        max_position_embeddings=max_position_embeddings,
        rope_base=rope_base,
        rms_norm_eps=rms_norm_eps,
        ffn_hidden_per_layer=ffn_hidden_per_layer,
    )


def _check_mixtral_compatible(shape: _VMShape) -> None:
    """Verify the VM's shape is expressible as a single Mixtral config.

    Mixtral has one ``intermediate_size`` shared by every expert in every
    layer; per-layer FFN widths can't be expressed without padding/zero-fill.
    """

    distinct = set(shape.ffn_hidden_per_layer)
    if len(distinct) > 1:
        raise MixtralShapeMismatchError(
            f"VM has per-layer FFN widths {sorted(distinct)}; Mixtral's "
            "intermediate_size is shared across all layers/experts. Run "
            "the right-sizer to harmonize FFN widths before export, or "
            "pad VM FFNs to the maximum width prior to calling this adapter."
        )


def infer_mixtral_config_kwargs(
    model: Any,
    *,
    num_local_experts: int = 8,
    num_experts_per_tok: int = 2,
) -> Dict[str, Any]:
    """Return MixtralConfig kwargs inferred from a VM model.

    The VM uses MHA (every Q-head has its own K/V), so
    ``num_key_value_heads`` defaults to ``num_attention_heads``.
    """

    shape = _read_vm_shape(model)
    _check_mixtral_compatible(shape)
    return {
        "vocab_size": shape.vocab_size,
        "hidden_size": shape.d_model,
        "intermediate_size": shape.ffn_hidden,
        "num_hidden_layers": shape.n_layers,
        "num_attention_heads": shape.num_heads,
        "num_key_value_heads": shape.num_heads,
        "head_dim": shape.head_dim,
        "max_position_embeddings": shape.max_position_embeddings,
        "rms_norm_eps": shape.rms_norm_eps,
        "rope_theta": shape.rope_base,
        "tie_word_embeddings": False,
        "num_local_experts": num_local_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "sliding_window": None,
    }


def _get_param(module: Any, name: str) -> Optional[torch.Tensor]:
    val = getattr(module, name, None)
    if val is None:
        return None
    if isinstance(val, torch.nn.Parameter):
        return val.data
    if isinstance(val, torch.Tensor):
        return val
    return None


def _pad_2d(
    w: torch.Tensor, target_shape: Tuple[int, int], *, dtype=None
) -> torch.Tensor:
    """Place ``w`` in the top-left corner of a zero-filled target tensor."""
    out_dtype = dtype if dtype is not None else w.dtype
    out = torch.zeros(target_shape, dtype=out_dtype, device=w.device)
    r = min(int(w.shape[0]), int(target_shape[0]))
    c = min(int(w.shape[1]), int(target_shape[1]))
    out[:r, :c] = w[:r, :c].to(out_dtype)
    return out


def _pad_1d_norm(
    w: torch.Tensor, target_len: int, *, dtype=None
) -> torch.Tensor:
    """Place a 1-D RMSNorm weight vector at the top of a ones-filled target.

    Layer-norm weights default to ones (identity scale) so unused lanes
    beyond the VM's ``d_model`` keep identity behaviour after padding.
    """
    out_dtype = dtype if dtype is not None else w.dtype
    out = torch.ones(target_len, dtype=out_dtype, device=w.device)
    n = min(int(w.shape[0]), int(target_len))
    out[:n] = w[:n].to(out_dtype)
    return out


def export_to_mixtral_state_dict(
    model: Any,
    *,
    num_local_experts: int = 8,
    num_experts_per_tok: int = 2,
    pad_to_mixtral: bool = False,
    target_shape: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    """Translate a compiled VM model into a Mixtral state_dict.

    Returns a dict mapping Mixtral parameter names to tensors. The dense
    VM FFN is replicated into every expert; the router gate is zeroed
    (uniform routing across experts, which makes the resulting MoE
    behaviourally equivalent to a single dense FFN before training).

    Args:
        model: The VM model. Must have ``.blocks``, ``.embed.embed``, and
            ``.head`` attributes shaped like an :class:`AutoregressiveVM`.
        num_local_experts: How many MoE experts the target Mixtral has.
        num_experts_per_tok: Top-k router config the target Mixtral uses.
            Not used by the state-dict map directly but kept here so callers
            can pass the same value when building the config.
        pad_to_mixtral: When True, route through the pad/zero-fill path
            (``_export_padded_state_dict``) which tolerates per-layer FFN
            width variation, vocab mismatch, n_heads/head_dim/d_model
            mismatch, and synthesizes extra layers / extra KV heads when
            the target is larger than the VM. Most weights in the output
            are zero, but the resulting state_dict loads cleanly into
            ``MixtralForCausalLM`` and runs a forward pass. This is wiring
            correctness, not numerical equivalence.
        target_shape: Required when ``pad_to_mixtral=True``. A dict with
            keys ``d_model``, ``n_layers``, ``num_heads``, ``num_kv_heads``
            (optional, defaults to ``num_heads``), ``head_dim``,
            ``ffn_hidden``, ``vocab_size``. Defaults to
            ``MIXTRAL_8X7B_SHAPE`` when None.

    Raises:
        MixtralShapeMismatchError: When ``pad_to_mixtral=False`` and the VM
            isn't Mixtral-shaped (e.g. per-layer FFN widths differ, or
            num_heads * head_dim != d_model).
    """

    if pad_to_mixtral:
        return _export_padded_state_dict(
            model,
            target=target_shape if target_shape is not None else MIXTRAL_8X7B_SHAPE,
            num_local_experts=num_local_experts,
        )

    del num_experts_per_tok  # consumed by config, not state_dict
    shape = _read_vm_shape(model)
    _check_mixtral_compatible(shape)

    sd: Dict[str, torch.Tensor] = {}

    # --- embeddings ---
    embed = getattr(model, "embed", None)
    inner_embed = getattr(embed, "embed", None) if embed is not None else None
    embed_weight = _get_param(inner_embed, "weight")
    if embed_weight is None:
        raise MixtralShapeMismatchError(
            "VM model has no embed.embed.weight to map to model.embed_tokens.weight."
        )
    sd["model.embed_tokens.weight"] = embed_weight.detach().clone()

    # --- per-layer transformer stack ---
    blocks = list(model.blocks)
    for i, block in enumerate(blocks):
        prefix = f"model.layers.{i}"
        attn = block.attn
        ffn = block.ffn

        for src_name, dst in [
            ("W_q", "self_attn.q_proj.weight"),
            ("W_k", "self_attn.k_proj.weight"),
            ("W_v", "self_attn.v_proj.weight"),
            ("W_o", "self_attn.o_proj.weight"),
        ]:
            w = _get_param(attn, src_name)
            if w is None:
                raise MixtralShapeMismatchError(
                    f"block {i} attn missing {src_name}."
                )
            sd[f"{prefix}.{dst}"] = w.detach().clone()

        # MoE: replicate the dense VM FFN into every expert.
        # VM FFN → Mixtral expert weight name mapping:
        #   W_gate (hidden_dim, dim)   →  w1 (intermediate_size, hidden_size)
        #   W_up   (hidden_dim, dim)   →  w3 (intermediate_size, hidden_size)
        #   W_down (dim, hidden_dim)   →  w2 (hidden_size, intermediate_size)
        # Mixtral expert forward is ``w2(SiLU(w1(x)) * w3(x))`` which is
        # exactly the SwiGLU the VM's PureFFN computes.
        w_gate = _get_param(ffn, "W_gate")
        w_up = _get_param(ffn, "W_up")
        w_down = _get_param(ffn, "W_down")
        if w_gate is None or w_up is None or w_down is None:
            raise MixtralShapeMismatchError(
                f"block {i} ffn missing W_gate/W_up/W_down."
            )
        for e in range(num_local_experts):
            sd[f"{prefix}.block_sparse_moe.experts.{e}.w1.weight"] = w_gate.detach().clone()
            sd[f"{prefix}.block_sparse_moe.experts.{e}.w3.weight"] = w_up.detach().clone()
            sd[f"{prefix}.block_sparse_moe.experts.{e}.w2.weight"] = w_down.detach().clone()

        # Router gate: zeros means uniform routing across experts at init.
        sd[f"{prefix}.block_sparse_moe.gate.weight"] = torch.zeros(
            num_local_experts, shape.d_model, dtype=embed_weight.dtype
        )

        # Layer norms
        if getattr(block, "use_rms_norm", False):
            attn_norm_w = _get_param(getattr(block, "attn_norm", None), "weight")
            ffn_norm_w = _get_param(getattr(block, "ffn_norm", None), "weight")
            if attn_norm_w is None or ffn_norm_w is None:
                raise MixtralShapeMismatchError(
                    f"block {i} has use_rms_norm but is missing attn_norm/ffn_norm weights."
                )
            sd[f"{prefix}.input_layernorm.weight"] = attn_norm_w.detach().clone()
            sd[f"{prefix}.post_attention_layernorm.weight"] = ffn_norm_w.detach().clone()
        else:
            sd[f"{prefix}.input_layernorm.weight"] = torch.ones(
                shape.d_model, dtype=embed_weight.dtype
            )
            sd[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(
                shape.d_model, dtype=embed_weight.dtype
            )

    # --- final norm (synthesized) ---
    sd["model.norm.weight"] = torch.ones(shape.d_model, dtype=embed_weight.dtype)

    # --- lm_head ---
    head = getattr(model, "head", None)
    head_weight = _get_param(head, "weight")
    if head_weight is None:
        raise MixtralShapeMismatchError(
            "VM model has no head.weight to map to lm_head.weight."
        )
    sd["lm_head.weight"] = head_weight.detach().clone()

    return sd


def _export_padded_state_dict(
    model: Any,
    *,
    target: Dict[str, int],
    num_local_experts: int = 8,
) -> Dict[str, torch.Tensor]:
    """Pad-fill export: VM weights → Mixtral-shape state_dict (top-left corner).

    Per-layer FFN width variation is tolerated (each layer padded
    independently to ``target['ffn_hidden']``). When the VM has fewer
    layers than the target, extra layers are synthesized with zero
    attention + zero FFN + ones-norm (identity via residual). When the
    VM has more layers, layers beyond the target count are dropped. GQA
    targets (``num_kv_heads`` < ``num_heads``) are supported by sizing
    K/V projections to ``num_kv_heads * head_dim`` rather than
    ``num_heads * head_dim``.

    The resulting state_dict loads cleanly into ``MixtralForCausalLM``
    and runs a forward pass; most weights are zero, so logits won't be
    informative — this is wiring correctness, not numerical equivalence.
    """
    t_vocab = int(target["vocab_size"])
    t_dmodel = int(target["d_model"])
    t_nlayers = int(target["n_layers"])
    t_nheads = int(target["num_heads"])
    t_nkv = int(target.get("num_kv_heads", t_nheads))
    t_head_dim = int(target["head_dim"])
    t_ffn = int(target["ffn_hidden"])
    t_q_dim = t_nheads * t_head_dim
    t_kv_dim = t_nkv * t_head_dim

    sd: Dict[str, torch.Tensor] = {}

    inner_embed = getattr(getattr(model, "embed", None), "embed", None)
    embed_weight = _get_param(inner_embed, "weight")
    if embed_weight is None:
        raise MixtralShapeMismatchError(
            "VM model has no embed.embed.weight to map to model.embed_tokens.weight."
        )
    dtype = embed_weight.dtype

    sd["model.embed_tokens.weight"] = _pad_2d(
        embed_weight.detach(), (t_vocab, t_dmodel), dtype=dtype
    )

    blocks = list(model.blocks)
    n_vm_layers = len(blocks)

    for i in range(t_nlayers):
        prefix = f"model.layers.{i}"
        if i < n_vm_layers:
            block = blocks[i]
            attn = block.attn
            ffn = block.ffn

            w_q = _get_param(attn, "W_q")
            w_k = _get_param(attn, "W_k")
            w_v = _get_param(attn, "W_v")
            w_o = _get_param(attn, "W_o")
            sd[f"{prefix}.self_attn.q_proj.weight"] = (
                _pad_2d(w_q.detach(), (t_q_dim, t_dmodel), dtype=dtype)
                if w_q is not None
                else torch.zeros((t_q_dim, t_dmodel), dtype=dtype)
            )
            sd[f"{prefix}.self_attn.k_proj.weight"] = (
                _pad_2d(w_k.detach(), (t_kv_dim, t_dmodel), dtype=dtype)
                if w_k is not None
                else torch.zeros((t_kv_dim, t_dmodel), dtype=dtype)
            )
            sd[f"{prefix}.self_attn.v_proj.weight"] = (
                _pad_2d(w_v.detach(), (t_kv_dim, t_dmodel), dtype=dtype)
                if w_v is not None
                else torch.zeros((t_kv_dim, t_dmodel), dtype=dtype)
            )
            sd[f"{prefix}.self_attn.o_proj.weight"] = (
                _pad_2d(w_o.detach(), (t_dmodel, t_q_dim), dtype=dtype)
                if w_o is not None
                else torch.zeros((t_dmodel, t_q_dim), dtype=dtype)
            )

            w_gate = _get_param(ffn, "W_gate")
            w_up = _get_param(ffn, "W_up")
            w_down = _get_param(ffn, "W_down")
            # Some VM blocks (ALU post_op blocks etc.) use non-SwiGLU FFN
            # variants that don't expose W_gate/W_up/W_down. Fall back to
            # zero-init for those — they'll be no-op blocks in the Mixtral
            # target, which is fine for wiring correctness.
            if w_gate is not None and w_up is not None and w_down is not None:
                padded_w1 = _pad_2d(w_gate.detach(), (t_ffn, t_dmodel), dtype=dtype)
                padded_w3 = _pad_2d(w_up.detach(), (t_ffn, t_dmodel), dtype=dtype)
                padded_w2 = _pad_2d(w_down.detach(), (t_dmodel, t_ffn), dtype=dtype)
            else:
                padded_w1 = torch.zeros((t_ffn, t_dmodel), dtype=dtype)
                padded_w3 = torch.zeros((t_ffn, t_dmodel), dtype=dtype)
                padded_w2 = torch.zeros((t_dmodel, t_ffn), dtype=dtype)
        else:
            sd[f"{prefix}.self_attn.q_proj.weight"] = torch.zeros(
                (t_q_dim, t_dmodel), dtype=dtype
            )
            sd[f"{prefix}.self_attn.k_proj.weight"] = torch.zeros(
                (t_kv_dim, t_dmodel), dtype=dtype
            )
            sd[f"{prefix}.self_attn.v_proj.weight"] = torch.zeros(
                (t_kv_dim, t_dmodel), dtype=dtype
            )
            sd[f"{prefix}.self_attn.o_proj.weight"] = torch.zeros(
                (t_dmodel, t_q_dim), dtype=dtype
            )
            padded_w1 = torch.zeros((t_ffn, t_dmodel), dtype=dtype)
            padded_w3 = torch.zeros((t_ffn, t_dmodel), dtype=dtype)
            padded_w2 = torch.zeros((t_dmodel, t_ffn), dtype=dtype)

        # Layer norms: copy the VM's RMSNorm weights when ``use_rms_norm=True``,
        # otherwise synthesize ones (identity scale). Unused lanes beyond the
        # VM's ``d_model`` always pad with ones so the synthesized portion is
        # an identity scale on the zero-filled FFN/attn outputs above.
        attn_norm_w = None
        ffn_norm_w = None
        if i < n_vm_layers and getattr(blocks[i], "use_rms_norm", False):
            attn_norm_w = _get_param(
                getattr(blocks[i], "attn_norm", None), "weight"
            )
            ffn_norm_w = _get_param(
                getattr(blocks[i], "ffn_norm", None), "weight"
            )
        if attn_norm_w is not None:
            sd[f"{prefix}.input_layernorm.weight"] = _pad_1d_norm(
                attn_norm_w.detach(), t_dmodel, dtype=dtype
            )
        else:
            sd[f"{prefix}.input_layernorm.weight"] = torch.ones(
                t_dmodel, dtype=dtype
            )
        if ffn_norm_w is not None:
            sd[f"{prefix}.post_attention_layernorm.weight"] = _pad_1d_norm(
                ffn_norm_w.detach(), t_dmodel, dtype=dtype
            )
        else:
            sd[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(
                t_dmodel, dtype=dtype
            )

        for e in range(num_local_experts):
            sd[f"{prefix}.block_sparse_moe.experts.{e}.w1.weight"] = padded_w1.clone()
            sd[f"{prefix}.block_sparse_moe.experts.{e}.w3.weight"] = padded_w3.clone()
            sd[f"{prefix}.block_sparse_moe.experts.{e}.w2.weight"] = padded_w2.clone()

        sd[f"{prefix}.block_sparse_moe.gate.weight"] = torch.zeros(
            num_local_experts, t_dmodel, dtype=dtype
        )

    sd["model.norm.weight"] = torch.ones(t_dmodel, dtype=dtype)

    head_weight = _get_param(getattr(model, "head", None), "weight")
    if head_weight is None:
        raise MixtralShapeMismatchError(
            "VM model has no head.weight to map to lm_head.weight."
        )
    sd["lm_head.weight"] = _pad_2d(
        head_weight.detach(), (t_vocab, t_dmodel), dtype=dtype
    )

    return sd


def load_into_mixtral(model: Any, config: Optional[Any] = None) -> Any:
    """Build a :class:`MixtralForCausalLM` and fill it from the VM.

    Args:
        model: The VM model.
        config: An optional pre-built :class:`MixtralConfig`. When ``None``,
            one is inferred from the VM via :func:`infer_mixtral_config_kwargs`
            with default expert settings (8 experts, top-2).

    Returns:
        A populated :class:`MixtralForCausalLM` ready for ``.forward()``.
        ``load_state_dict(..., strict=True)`` is used so any shape /
        key mismatch surfaces immediately.

    Raises:
        RuntimeError: When ``transformers`` (or its Mixtral classes) are
            unavailable.
        MixtralShapeMismatchError: When the VM isn't Mixtral-shaped.
    """

    try:
        from transformers import MixtralConfig, MixtralForCausalLM
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "transformers (with Mixtral support) is required to call "
            "load_into_mixtral; install transformers and retry."
        ) from exc

    if config is None:
        config = MixtralConfig(**infer_mixtral_config_kwargs(model))

    num_local_experts = int(getattr(config, "num_local_experts", 8))
    num_experts_per_tok = int(getattr(config, "num_experts_per_tok", 2))

    sd = export_to_mixtral_state_dict(
        model,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    hf_model = MixtralForCausalLM(config)
    # Cast all tensors to the HF model's dtype so a fp32-default VM doesn't
    # collide with an fp16 / bf16 config.
    target_dtype = next(hf_model.parameters()).dtype
    sd = {k: v.to(dtype=target_dtype) for k, v in sd.items()}
    missing, unexpected = hf_model.load_state_dict(sd, strict=False)
    # Surface mismatches loudly — callers want to know if the adapter is
    # out of sync with the installed transformers version.
    if missing:
        raise RuntimeError(
            f"load_into_mixtral: {len(missing)} Mixtral parameters not "
            f"populated by the adapter (first few): {missing[:5]}"
        )
    if unexpected:
        raise RuntimeError(
            f"load_into_mixtral: {len(unexpected)} adapter-produced keys "
            f"not consumed by the HF model (first few): {unexpected[:5]}"
        )
    return hf_model
