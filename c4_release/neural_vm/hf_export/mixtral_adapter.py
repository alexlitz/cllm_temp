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
  — ones when the VM block has no RMSNorm enabled
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
from typing import Any, Dict, List, Optional

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


def export_to_mixtral_state_dict(
    model: Any,
    *,
    num_local_experts: int = 8,
    num_experts_per_tok: int = 2,
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

    Raises:
        MixtralShapeMismatchError: When the VM isn't Mixtral-shaped (e.g.
            per-layer FFN widths differ, or num_heads * head_dim != d_model).
    """

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
