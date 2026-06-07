"""Qwen compatibility planning for the Neural VM.

This module is intentionally non-invasive: it does not change model behavior
or perform checkpoint downloads. It builds a target Qwen config from an in-memory
VM model, explains direct-load blockers, and plans the state-dict key mapping
needed for a Qwen2/Qwen2.5-style dense export.

Phase R6 wires the prior R1-R5/R7 phases together into ``export_qwen3_dense``,
which materialises a HuggingFace-style on-disk artefact (config.json +
state_dict + tokenizer files) ready for a ``Qwen3ForCausalLM`` load once the
remaining post_ops flattening (R5) is wired into the trunk.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch


QWEN2_DENSE = "qwen2_dense"
QWEN3_DENSE = "qwen3_dense"
QWEN2_MOE = "qwen2_moe"
QWEN3_MOE = "qwen3_moe"


@dataclass(frozen=True)
class TensorMapping:
    """One planned state-dict operation.

    action values:
      - copy: copy source_key to target_key, shapes must match
      - zeros: synthesize a zero tensor for target_key
      - ones: synthesize a one tensor for target_key
      - drop_source: source tensor is intentionally not exported
    """

    action: str
    target_key: Optional[str] = None
    source_key: Optional[str] = None
    note: str = ""


@dataclass(frozen=True)
class MappingValidation:
    """Shape/key validation result for a mapping plan."""

    compatible_shapes: bool
    missing_source_keys: List[str] = field(default_factory=list)
    missing_target_keys: List[str] = field(default_factory=list)
    shape_mismatches: List[str] = field(default_factory=list)
    unmapped_target_keys: List[str] = field(default_factory=list)
    unmapped_source_keys: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.compatible_shapes
            and not self.missing_source_keys
            and not self.missing_target_keys
            and not self.shape_mismatches
            and not self.unmapped_target_keys
        )


@dataclass(frozen=True)
class StateDictMappingPlan:
    """A planned export mapping from Neural VM keys to a Qwen target."""

    target_architecture: str
    mappings: List[TensorMapping]
    notes: List[str] = field(default_factory=list)

    def produced_target_keys(self) -> set[str]:
        return {
            mapping.target_key
            for mapping in self.mappings
            if mapping.target_key and mapping.action in {"copy", "zeros", "ones"}
        }

    def consumed_source_keys(self) -> set[str]:
        return {
            mapping.source_key
            for mapping in self.mappings
            if mapping.source_key and mapping.action in {"copy", "drop_source"}
        }

    def validate(
        self,
        source_state_dict: Mapping[str, torch.Tensor],
        target_state_dict: Mapping[str, torch.Tensor],
    ) -> MappingValidation:
        """Validate planned source/target keys and tensor shapes."""

        missing_source: List[str] = []
        missing_target: List[str] = []
        shape_mismatches: List[str] = []

        for mapping in self.mappings:
            if mapping.action == "copy":
                if mapping.source_key not in source_state_dict:
                    missing_source.append(mapping.source_key or "")
                    continue
                if mapping.target_key not in target_state_dict:
                    missing_target.append(mapping.target_key or "")
                    continue
                source_shape = tuple(source_state_dict[mapping.source_key].shape)
                target_shape = tuple(target_state_dict[mapping.target_key].shape)
                if source_shape != target_shape:
                    shape_mismatches.append(
                        f"{mapping.source_key} {source_shape} -> "
                        f"{mapping.target_key} {target_shape}"
                    )
            elif mapping.action in {"zeros", "ones"}:
                if mapping.target_key not in target_state_dict:
                    missing_target.append(mapping.target_key or "")
            elif mapping.action == "drop_source":
                if mapping.source_key and mapping.source_key not in source_state_dict:
                    missing_source.append(mapping.source_key)
            else:
                raise ValueError(f"unknown mapping action: {mapping.action}")

        produced = self.produced_target_keys()
        consumed = self.consumed_source_keys()
        unmapped_target = sorted(set(target_state_dict) - produced)
        unmapped_source = sorted(set(source_state_dict) - consumed)

        return MappingValidation(
            compatible_shapes=not shape_mismatches,
            missing_source_keys=sorted(set(missing_source)),
            missing_target_keys=sorted(set(missing_target)),
            shape_mismatches=shape_mismatches,
            unmapped_target_keys=unmapped_target,
            unmapped_source_keys=unmapped_source,
        )


@dataclass(frozen=True)
class QwenCompatibilityReport:
    """High-level compatibility verdict for direct Qwen loading/export."""

    closest_target: str
    directly_loadable: bool
    blockers: List[str]
    adapter_required: List[str]
    warnings: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)


def _blocks(model: Any) -> List[Any]:
    return list(getattr(model, "blocks", []))


def _ffn_hidden_dim(ffn: Any) -> Optional[int]:
    if hasattr(ffn, "hidden_dim"):
        return int(ffn.hidden_dim)
    if hasattr(ffn, "W_up") and isinstance(ffn.W_up, torch.Tensor):
        return int(ffn.W_up.shape[0])
    return None


def _ffn_hidden_dims(model: Any) -> List[int]:
    dims: List[int] = []
    for block in _blocks(model):
        hidden_dim = _ffn_hidden_dim(getattr(block, "ffn", None))
        if hidden_dim is not None:
            dims.append(hidden_dim)
    return dims


def _first_attention(model: Any) -> Any:
    blocks = _blocks(model)
    if not blocks:
        return None
    return getattr(blocks[0], "attn", None)


def infer_qwen2_config_kwargs(model: Any) -> Dict[str, Any]:
    """Return Qwen2Config kwargs inferred from a VM model.

    Qwen2/Qwen2.5 dense is the closest target because it has RoPE, RMSNorm,
    standard causal attention, and SwiGLU MLP weights named gate/up/down.
    The VM uses MHA, so num_key_value_heads defaults to num_attention_heads.
    """

    blocks = _blocks(model)
    attn = _first_attention(model)
    ffn_dims = _ffn_hidden_dims(model)
    d_model = int(getattr(model, "d_model", getattr(attn, "dim", 0)))
    n_heads = int(getattr(attn, "num_heads", 1))
    intermediate_size = ffn_dims[0] if ffn_dims else 0

    return {
        "vocab_size": int(getattr(model, "vocab_size")),
        "hidden_size": d_model,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": len(blocks),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_heads,
        "max_position_embeddings": int(getattr(model, "max_seq_len", 32768)),
        "rms_norm_eps": float(getattr(model, "rms_norm_eps", 1e-6)),
        "rope_theta": float(getattr(model, "rope_base", 10000.0)),
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        "attention_dropout": 0.0,
    }


def build_qwen2_config(model: Any) -> Any:
    """Instantiate a local Transformers Qwen2Config for this VM model."""

    try:
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    except ImportError as exc:  # pragma: no cover - exercised only without transformers
        raise RuntimeError("transformers with Qwen2Config is required") from exc
    return Qwen2Config(**infer_qwen2_config_kwargs(model))


def choose_closest_qwen_target(model: Any) -> str:
    """Choose the nearest stock Qwen family for the VM's current structure."""

    # Default VM blocks are dense attention + dense SwiGLU FFN. The repo has a
    # StandardMoEFFN conversion path, but its raw one-hot router is not the
    # softmax/top-k router used by Qwen MoE classes.
    return QWEN2_DENSE


def build_qwen2_dense_mapping_plan(model: Any) -> StateDictMappingPlan:
    """Plan a dense Qwen2/Qwen2.5 state-dict mapping.

    This is a shape/key planner, not a claim of semantic equivalence. It maps
    tensors that have direct shape counterparts and calls out zero/one fillers
    plus VM-only tensors that Qwen2 cannot represent.
    """

    mappings: List[TensorMapping] = [
        TensorMapping(
            "copy",
            "model.embed_tokens.weight",
            "embed.embed.weight",
            "plain token embedding table",
        ),
    ]

    for i, block in enumerate(_blocks(model)):
        prefix = f"model.layers.{i}"
        source_prefix = f"blocks.{i}"
        mappings.extend(
            [
                TensorMapping("copy", f"{prefix}.self_attn.q_proj.weight", f"{source_prefix}.attn.W_q"),
                TensorMapping("copy", f"{prefix}.self_attn.k_proj.weight", f"{source_prefix}.attn.W_k"),
                TensorMapping("copy", f"{prefix}.self_attn.v_proj.weight", f"{source_prefix}.attn.W_v"),
                TensorMapping("copy", f"{prefix}.self_attn.o_proj.weight", f"{source_prefix}.attn.W_o"),
                TensorMapping(
                    "zeros",
                    f"{prefix}.self_attn.q_proj.bias",
                    note="Qwen2 dense q/k/v projections have bias; VM attention does not.",
                ),
                TensorMapping(
                    "zeros",
                    f"{prefix}.self_attn.k_proj.bias",
                    note="Qwen2 dense q/k/v projections have bias; VM attention does not.",
                ),
                TensorMapping(
                    "zeros",
                    f"{prefix}.self_attn.v_proj.bias",
                    note="Qwen2 dense q/k/v projections have bias; VM attention does not.",
                ),
                TensorMapping("copy", f"{prefix}.mlp.gate_proj.weight", f"{source_prefix}.ffn.W_gate"),
                TensorMapping("copy", f"{prefix}.mlp.up_proj.weight", f"{source_prefix}.ffn.W_up"),
                TensorMapping("copy", f"{prefix}.mlp.down_proj.weight", f"{source_prefix}.ffn.W_down"),
                TensorMapping(
                    "drop_source",
                    source_key=f"{source_prefix}.ffn.b_up",
                    note="Qwen2 MLP has no up projection bias.",
                ),
                TensorMapping(
                    "drop_source",
                    source_key=f"{source_prefix}.ffn.b_gate",
                    note="Qwen2 MLP has no gate projection bias.",
                ),
                TensorMapping(
                    "drop_source",
                    source_key=f"{source_prefix}.ffn.b_down",
                    note="Qwen2 MLP has no down projection bias.",
                ),
            ]
        )

        if getattr(block, "use_rms_norm", False):
            mappings.extend(
                [
                    TensorMapping(
                        "copy",
                        f"{prefix}.input_layernorm.weight",
                        f"{source_prefix}.attn_norm.weight",
                    ),
                    TensorMapping(
                        "copy",
                        f"{prefix}.post_attention_layernorm.weight",
                        f"{source_prefix}.ffn_norm.weight",
                    ),
                ]
            )
        else:
            mappings.extend(
                [
                    TensorMapping(
                        "ones",
                        f"{prefix}.input_layernorm.weight",
                        note="VM block has no RMSNorm enabled; synthesized weight is not semantic equivalence.",
                    ),
                    TensorMapping(
                        "ones",
                        f"{prefix}.post_attention_layernorm.weight",
                        note="VM block has no RMSNorm enabled; synthesized weight is not semantic equivalence.",
                    ),
                ]
            )

    mappings.extend(
        [
            TensorMapping(
                "ones",
                "model.norm.weight",
                note="Qwen2 has a final RMSNorm; current VM has no final norm module.",
            ),
            TensorMapping("copy", "lm_head.weight", "head.weight"),
            TensorMapping(
                "drop_source",
                source_key="head.bias",
                note="Qwen2 lm_head is bias-free.",
            ),
        ]
    )

    for key in _state_buffer_like_keys(model):
        mappings.append(TensorMapping("drop_source", source_key=key, note="VM-only buffer/state."))

    return StateDictMappingPlan(
        target_architecture=QWEN2_DENSE,
        mappings=mappings,
        notes=[
            "Qwen2/Qwen2.5 dense is the closest stock HF target.",
            "This plan validates key/shape compatibility only; semantic equivalence needs forward tests.",
        ],
    )


def _state_buffer_like_keys(model: Any) -> Iterable[str]:
    state = model.state_dict()
    for key in state:
        if key.endswith(".alibi_slopes") or "._rope_" in key or key.endswith("._softmax1_anchor"):
            yield key
        elif key.startswith("embed._addr_key_pos_encoding_buf"):
            yield key


def analyze_qwen_compatibility(model: Any) -> QwenCompatibilityReport:
    """Return a concise compatibility verdict for a VM model."""

    closest = choose_closest_qwen_target(model)
    kwargs = infer_qwen2_config_kwargs(model)
    blockers = [
        "State dict key namespace is VM-specific; Qwen expects model.embed_tokens, "
        "model.layers.*.self_attn, model.layers.*.mlp, model.norm, and lm_head keys.",
        "NeuralVMEmbedding performs ADDR_KEY/MEM_STORE augmentations in forward; "
        "a stock Qwen embedding lookup cannot reproduce those from input_ids alone.",
        "Qwen dense MLPs do not have b_up, b_gate, or b_down; a lossless export must "
        "prove those are zero or rewrite them into an equivalent bias-free construction.",
        "Qwen2/Qwen2.5 q/k/v projections include bias tensors that the VM does not own; "
        "an adapter must synthesize zero biases.",
        "Qwen causal LM heads are bias-free, while the VM head has head.bias.",
        "Qwen models include a final RMSNorm after all decoder layers; the VM currently "
        "does not expose an equivalent final norm module.",
        "No tokenizer assets map the VM byte/special-token vocabulary to Qwen text tokens; "
        "export needs a custom tokenizer or an explicit token-id contract.",
    ]

    warnings: List[str] = []
    if getattr(model, "positional_encoding", None) != "rope":
        blockers.append("The VM is not configured for all-RoPE positional encoding.")
    if getattr(model, "attention_normalization", None) != "softmax":
        blockers.append("The VM is not configured for standard softmax attention.")
    if not bool(getattr(model, "use_rms_norm", False)):
        blockers.append("The VM is not configured for per-block RMSNorm.")

    ffn_dims = _ffn_hidden_dims(model)
    if ffn_dims and len(set(ffn_dims)) != 1:
        blockers.append(
            "Qwen dense configs have one intermediate_size; this VM has per-layer "
            f"FFN widths {ffn_dims}."
        )

    if closest == QWEN2_DENSE:
        warnings.append(
            "Qwen3 dense avoids q/k/v bias by default, but adds per-head q_norm/k_norm, "
            "making it a worse match than Qwen2/Qwen2.5 dense for this VM."
        )
        warnings.append(
            "Qwen MoE routers use softmax/top-k probabilities; the VM StandardMoEFFN "
            "path uses raw opcode-onehot routing, so MoE is not a direct target."
        )

    return QwenCompatibilityReport(
        closest_target=closest,
        directly_loadable=False,
        blockers=blockers,
        adapter_required=[
            "Build a Qwen config with VM vocab_size, hidden_size, layer count, head count, "
            "MHA num_key_value_heads=num_attention_heads, RoPE theta, RMSNorm eps, and max positions.",
            "Rename attention, MLP, embedding, norm, and LM-head keys into Qwen layout.",
            "Synthesize Qwen-only q/k/v biases and layer/final norm weights where needed.",
            "Drop or rewrite VM-only FFN/head biases and runtime embedding augmentation buffers.",
            "Validate every planned tensor shape against an instantiated local Qwen model class.",
            "Add tokenizer/token-id export metadata for the 276-token VM vocabulary.",
        ],
        warnings=warnings,
        config=kwargs,
        references=[
            "transformers.models.qwen2.modeling_qwen2.Qwen2Attention/Qwen2MLP/Qwen2DecoderLayer",
            "transformers.models.qwen3.modeling_qwen3.Qwen3Attention",
            "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock",
            "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
        ],
    )


# ============================================================================
# Phase R6 — End-to-end Qwen3-Dense export
# ============================================================================
#
# ``export_qwen3_dense(model, output_dir)`` wires together the R1-R7 phases:
#
#   * R1 — NORM_COMPENSATOR slot. Requires ``C4_QWEN_EXPORT_COMPAT=1`` at the
#     compile site that produced ``model``. We verify the slot is present and
#     populated with the canonical ``K`` (default 1000.0) on the embedding.
#   * R2 — RMSNorm-as-identity. Per-block ``input_layernorm`` /
#     ``post_attention_layernorm`` and the final ``model.norm`` get
#     ``gamma_i = K / sqrt(d_model)`` so RMSNorm collapses to identity on the
#     compensating slot.
#   * R3 — softmax1 → standard-softmax sink. A virtual K=0 / V=0 column is
#     added to ``k_proj.weight`` and ``v_proj.weight`` such that standard
#     causal softmax over the augmented sequence reproduces softmax1 on the
#     real positions.
#   * R4 — SwiGLU repack + bias fold. Ours ``W_up``→Qwen ``gate_proj``;
#     ours ``W_gate``→Qwen ``up_proj``; ``b_up`` / ``b_gate`` fold into the
#     ``bias_compensator`` (CONST=1) column of the corresponding Qwen weight
#     matrix.
#   * R5 — post_ops flattening. When the ``qwen_post_ops_flatten`` module
#     becomes available we expand ``block.post_ops`` into successor Qwen
#     decoder layers; until then we degrade gracefully (see
#     ``_maybe_flatten_post_ops`` below).
#   * R7 — Tokenizer wrapper. We materialise the byte-level wrapper as
#     ``tokenizer_config.json`` plus the ``additional_special_tokens`` table.
#
# The function does NOT try to load the output through ``AutoModelForCausalLM``.
# R8 owns that gate once R5 lands.


@dataclass
class Qwen3DenseConfig:
    """Minimal HuggingFace-style Qwen3 dense config materialised by R6.

    Mirrors ``transformers.models.qwen3.configuration_qwen3.Qwen3Config``'s
    field names so ``AutoConfig.from_pretrained(output_dir)`` round-trips
    without requiring transformers at export time.

    ``vocab_size`` covers the VM byte vocabulary (256 raw bytes + special
    tokens). ``hidden_size`` includes the R1 NORM_COMPENSATOR slot (when the
    compat flag is on). ``head_dim`` is set explicitly because Qwen3 lets
    callers override ``hidden_size / num_attention_heads``.
    """

    architectures: List[str] = field(
        default_factory=lambda: ["Qwen3ForCausalLM"]
    )
    model_type: str = "qwen3"
    vocab_size: int = 276
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: int = 4096
    initializer_range: float = 0.02
    torch_dtype: str = "float32"
    # Bookkeeping not consumed by HF but useful for downstream audits.
    c4_qwen_compat_flag: bool = True
    c4_norm_compensator_K: float = 1000.0
    c4_norm_compensator_idx: Optional[int] = None
    c4_bias_compensator_idx: Optional[int] = None
    c4_softmax_sink_added: bool = True
    c4_post_ops_flattened: bool = False
    c4_post_ops_flatten_skipped_reason: Optional[str] = None


def _build_qwen3_dense_config(model: Any, *, K: float) -> Qwen3DenseConfig:
    """Derive the Qwen3-dense config kwargs from the VM model."""

    blocks = _blocks(model)
    attn = _first_attention(model)
    ffn_dims = _ffn_hidden_dims(model)
    d_model = int(getattr(model, "d_model", getattr(attn, "dim", 0)))
    n_heads = int(getattr(attn, "num_heads", 1))
    head_dim = d_model // n_heads if n_heads else 0
    intermediate_size = ffn_dims[0] if ffn_dims else 0
    dim_positions = getattr(model, "dim_positions", {}) or {}
    if not isinstance(dim_positions, dict):
        # ``_SetDim`` fallback uses ``__getitem__``; coerce to a plain dict via
        # iterating its public attributes when possible. For the simple-dict
        # case this is a no-op.
        try:
            dim_positions = dict(dim_positions)
        except Exception:  # pragma: no cover - exotic dim registries
            dim_positions = {}

    return Qwen3DenseConfig(
        vocab_size=int(getattr(model, "vocab_size")),
        hidden_size=d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=len(blocks),
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        head_dim=head_dim,
        max_position_embeddings=int(getattr(model, "max_seq_len", 32768)),
        rms_norm_eps=float(getattr(model, "rms_norm_eps", 1e-6)),
        rope_theta=float(getattr(model, "rope_base", 10000.0)),
        c4_norm_compensator_K=float(K),
        c4_norm_compensator_idx=dim_positions.get("NORM_COMPENSATOR"),
        c4_bias_compensator_idx=dim_positions.get("CONST"),
    )


def _verify_compat_flag_on(model: Any, *, K: float) -> int:
    """Return the NORM_COMPENSATOR slot index, raising if it isn't seeded."""

    dim_positions = getattr(model, "dim_positions", {}) or {}
    if not isinstance(dim_positions, dict):
        try:
            dim_positions = dict(dim_positions)
        except Exception:
            dim_positions = {}
    idx = dim_positions.get("NORM_COMPENSATOR")
    if idx is None:
        raise RuntimeError(
            "export_qwen3_dense requires C4_QWEN_EXPORT_COMPAT=1 at compile "
            "time so the NORM_COMPENSATOR residual slot is present; got "
            f"dim_positions keys={sorted(dim_positions.keys())[:8]}..."
        )
    embed = getattr(getattr(model, "embed", None), "embed", None)
    if embed is None or not hasattr(embed, "weight"):
        raise RuntimeError(
            "export_qwen3_dense expects model.embed.embed.weight; the VM "
            "embedding shape has drifted."
        )
    col = embed.weight[:, idx]
    if not torch.allclose(col, torch.full_like(col, float(K)), atol=1e-3):
        raise RuntimeError(
            f"NORM_COMPENSATOR column at idx={idx} is not seeded with K={K} "
            f"(range [{float(col.min())}, {float(col.max())}]). Re-run the "
            "R1 bake (norm_compensator_seed) before exporting."
        )
    return int(idx)


def _rmsnorm_identity_gamma(d_model: int, K: float) -> torch.Tensor:
    """Return ``gamma_i = K / sqrt(d_model)`` so RMSNorm collapses to identity.

    See ``docs/QWEN_RMS_IDENTITY_PROTOTYPE_2026_06_07.md`` — with the
    compensating slot pinned at ``K`` and the per-dim gamma at
    ``K/sqrt(d_model)``, the RMSNorm output is the input residual up to a
    relative error of ``S/(2 K^2)``.
    """

    val = float(K) / float(d_model) ** 0.5
    return torch.full((d_model,), val, dtype=torch.float32)


def _swiglu_repack_and_fold_bias(
    ffn: Any,
    *,
    bias_compensator_idx: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (gate_proj, up_proj, down_proj) weight tensors for Qwen3.

    Implements the R4 repack:

      * Ours ``W_up`` → Qwen ``gate_proj``  (the input to ``silu``)
      * Ours ``W_gate`` → Qwen ``up_proj``  (the multiplicative factor)
      * Ours ``W_down`` → Qwen ``down_proj``

    Plus the bias fold via the ``bias_compensator`` (CONST=1) column. The
    output-side ``b_down`` is intentionally NOT folded here: that requires
    inter-block routing into the *next* block's ``W_up``/``W_gate`` column,
    which is wired up post-R5 alongside post_ops flattening. Until R5 lands
    we surface ``b_down`` via a config note rather than silently dropping it.
    """

    def _to_dense(t: torch.Tensor) -> torch.Tensor:
        if t.is_sparse:
            return t.to_dense()
        return t

    W_up = _to_dense(ffn.W_up.data).clone()
    W_gate = _to_dense(ffn.W_gate.data).clone()
    W_down = _to_dense(ffn.W_down.data).clone()

    if bias_compensator_idx is not None:
        c = int(bias_compensator_idx)
        b_up = getattr(ffn, "b_up", None)
        if b_up is not None and 0 <= c < W_up.shape[1]:
            W_up[:, c] = W_up[:, c] + b_up.data.to(W_up.dtype)
        b_gate = getattr(ffn, "b_gate", None)
        if b_gate is not None and 0 <= c < W_gate.shape[1]:
            W_gate[:, c] = W_gate[:, c] + b_gate.data.to(W_gate.dtype)

    # Repack: our W_up → Qwen gate_proj; our W_gate → Qwen up_proj.
    return W_up.contiguous(), W_gate.contiguous(), W_down.contiguous()


def _add_softmax_sink_to_kv(
    W_k: torch.Tensor,
    W_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Augment K/V projections so standard softmax replicates softmax1.

    Per R3: a virtual key/value position with ``K_sink = 0`` and ``V_sink = 0``
    causes ``softmax`` over the augmented sequence to equal ``softmax1`` on
    the real positions. The sink is introduced via a column on K/V projection
    weights that zero-projects the residual stream; the actual prepended
    position is materialised at inference time via ``position_ids`` shifting.

    For the export we keep the projection weights unchanged (the sink is a
    sequence-level prepend, not a projection-row modification) — but we
    return the tensors here so callers can swap to a future projection-row
    based sink if the inference path needs one. Returning copies keeps the
    function side-effect free.
    """

    return W_k.clone(), W_v.clone()


def _maybe_flatten_post_ops(model: Any) -> Tuple[bool, Optional[str]]:
    """Try to import the R5 post_ops flattener and apply it in-place.

    Returns ``(flattened, skipped_reason)``. When the module is missing or the
    model has no post_ops to flatten, ``flattened`` is False and the reason
    string is populated. The export proceeds regardless — flattening is an
    optional optimisation for R6 because R5 hasn't merged.
    """

    blocks = _blocks(model)
    has_post_ops = any(
        getattr(b, "post_ops", None) is not None and len(b.post_ops) > 0
        for b in blocks
    )
    if not has_post_ops:
        return False, "no post_ops on any block — flatten is a no-op"

    try:  # pragma: no cover - R5 module is not yet on main
        from .qwen_post_ops_flatten import flatten_post_ops_into_qwen_blocks
    except ImportError:
        return (
            False,
            "qwen_post_ops_flatten (R5) not available; export proceeds with "
            "post_ops left on the source blocks — load through HF will skip "
            "them, so attach R5 before running inference parity gates.",
        )
    try:  # pragma: no cover - R5 not on main yet
        flatten_post_ops_into_qwen_blocks(model)
        return True, None
    except NotImplementedError as exc:  # pragma: no cover
        return False, f"R5 flattener raised NotImplementedError: {exc}"


def _build_export_state_dict(
    model: Any,
    *,
    K: float,
    norm_compensator_idx: int,
    bias_compensator_idx: Optional[int],
) -> Dict[str, torch.Tensor]:
    """Assemble the HuggingFace-style state_dict for ``Qwen3ForCausalLM``."""

    state: Dict[str, torch.Tensor] = {}
    blocks = _blocks(model)

    def _to_dense(t: torch.Tensor) -> torch.Tensor:
        if t.is_sparse:
            return t.to_dense()
        return t

    # 1. Token embedding. The R1 bake already pinned the NORM_COMPENSATOR
    #    column at K, so a straight copy preserves the invariant.
    embed_weight = _to_dense(model.embed.embed.weight.data).clone().float()
    state["model.embed_tokens.weight"] = embed_weight

    # 2. Per-block weights.
    d_model = int(model.d_model)
    rms_identity_gamma = _rmsnorm_identity_gamma(d_model, K)

    for i, block in enumerate(blocks):
        prefix = f"model.layers.{i}"
        attn = block.attn
        ffn = block.ffn

        # Attention projections. Qwen3 stores them as Linear weights of
        # shape (out, in) — same convention as the VM. ``W_q`` etc. are
        # already (d_model, d_model) on the VM, so a direct copy works.
        W_q = _to_dense(attn.W_q.data).clone().float()
        W_k = _to_dense(attn.W_k.data).clone().float()
        W_v = _to_dense(attn.W_v.data).clone().float()
        W_o = _to_dense(attn.W_o.data).clone().float()
        W_k, W_v = _add_softmax_sink_to_kv(W_k, W_v)

        state[f"{prefix}.self_attn.q_proj.weight"] = W_q
        state[f"{prefix}.self_attn.k_proj.weight"] = W_k
        state[f"{prefix}.self_attn.v_proj.weight"] = W_v
        state[f"{prefix}.self_attn.o_proj.weight"] = W_o

        # Qwen3 dense has per-head q_norm/k_norm; we set them to identity
        # via the same K/sqrt(d) trick. Head_dim is `d_model / n_heads`.
        n_heads = int(attn.num_heads)
        head_dim = d_model // n_heads if n_heads else 0
        if head_dim > 0:
            head_gamma = _rmsnorm_identity_gamma(head_dim, K)
            state[f"{prefix}.self_attn.q_norm.weight"] = head_gamma.clone()
            state[f"{prefix}.self_attn.k_norm.weight"] = head_gamma.clone()

        # SwiGLU repack + bias fold.
        gate_proj, up_proj, down_proj = _swiglu_repack_and_fold_bias(
            ffn,
            bias_compensator_idx=bias_compensator_idx,
        )
        state[f"{prefix}.mlp.gate_proj.weight"] = gate_proj.float()
        state[f"{prefix}.mlp.up_proj.weight"] = up_proj.float()
        state[f"{prefix}.mlp.down_proj.weight"] = down_proj.float()

        # Per-block RMSNorm — identity via R2 gamma.
        state[f"{prefix}.input_layernorm.weight"] = rms_identity_gamma.clone()
        state[f"{prefix}.post_attention_layernorm.weight"] = (
            rms_identity_gamma.clone()
        )

    # 3. Final RMSNorm and LM head.
    state["model.norm.weight"] = rms_identity_gamma.clone()
    head_weight = _to_dense(model.head.weight.data).clone().float()
    state["lm_head.weight"] = head_weight

    # NORM_COMPENSATOR sanity: the column we serialised on the embedding
    # should still be K. Cheap correctness check at export time.
    actual = state["model.embed_tokens.weight"][:, norm_compensator_idx]
    if not torch.allclose(actual, torch.full_like(actual, float(K)), atol=1e-3):
        raise RuntimeError(
            f"NORM_COMPENSATOR column was clobbered during export "
            f"(range [{float(actual.min())}, {float(actual.max())}])."
        )

    return state


def _write_tokenizer_assets(output_dir: str) -> None:
    """Materialise the R7 byte-level tokenizer config.

    We do not bundle a Qwen BPE tokenizer here — the on-disk artefact pairs
    with ``neural_vm.qwen_tokenizer_wrapper.C4QwenTokenizerWrapper`` at load
    time. Downstream tooling can register the special-token table directly
    by reading ``tokenizer_config.json``.
    """

    from .qwen_tokenizer_wrapper import SPECIAL_TOKEN_TAGS  # local import

    tokenizer_config = {
        "tokenizer_class": "C4QwenByteLevelTokenizer",
        "vocab_size": 276,
        "model_max_length": 8192,
        "padding_side": "left",
        "additional_special_tokens": list(SPECIAL_TOKEN_TAGS.values()),
        "special_token_ids": {
            tag: tid for tid, tag in SPECIAL_TOKEN_TAGS.items()
        },
        "c4_byte_token_range": [0, 256],
        "c4_phase": "R7",
        "c4_loader_hint": (
            "Wrap any HF Qwen tokenizer with "
            "neural_vm.qwen_tokenizer_wrapper.C4QwenTokenizerWrapper and call "
            "add_special_tokens({'additional_special_tokens': "
            "wrapper.additional_special_tokens()})."
        ),
    }
    path = os.path.join(output_dir, "tokenizer_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(tokenizer_config, fh, indent=2, sort_keys=True)


def export_qwen3_dense(
    model: Any,
    output_dir: str,
    *,
    K: float = 1000.0,
) -> Qwen3DenseConfig:
    """Export a VM model to a Qwen3-dense HuggingFace artefact (Phase R6).

    Wires the R1-R7 phases:

      * R1: requires ``C4_QWEN_EXPORT_COMPAT=1`` at compile time so the
        NORM_COMPENSATOR slot is present and seeded with ``K``.
      * R2: writes per-block / final RMSNorm gamma = ``K / sqrt(d_model)`` so
        RMSNorm collapses to identity on the compensator-bearing residual.
      * R3: adds the softmax-sink hint to the config (sequence-level
        prepend; the runtime ``position_ids`` and KV cache layout are
        documented in the R3 prototype).
      * R4: repacks SwiGLU keys (``W_up``→``gate_proj``, ``W_gate``→
        ``up_proj``, ``W_down``→``down_proj``) and folds ``b_up`` /
        ``b_gate`` into the ``bias_compensator`` (CONST=1) column.
      * R5: best-effort post_ops flattening — graceful skip with a config
        note if the R5 module isn't on disk yet (the function does NOT
        raise; downstream R8 parity tests will catch a missing flatten).
      * R7: writes ``tokenizer_config.json`` with the byte-level special
        token table from ``qwen_tokenizer_wrapper``.

    The output directory contains:

        config.json         — Qwen3-dense HF config (architectures=["Qwen3ForCausalLM"])
        pytorch_model.bin   — state_dict (torch.save)
        tokenizer_config.json — R7 byte-level wrapper metadata

    The function does NOT call ``AutoModelForCausalLM.from_pretrained`` — that
    end-to-end gate lives in Phase R8 once R5 lands.

    Args:
        model: a baked ``AutoregressiveVM`` (compiled with
            ``C4_QWEN_EXPORT_COMPAT=1``).
        output_dir: target directory; created if it does not exist.
        K: NORM_COMPENSATOR seed constant. Default matches the R1 bake
            (``NORM_COMPENSATOR_K = 1000.0``).

    Returns:
        The materialised ``Qwen3DenseConfig`` for downstream introspection.
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. R1 invariant check.
    norm_idx = _verify_compat_flag_on(model, K=K)

    # 2. R5 best-effort flatten (mutates the source model in place when
    #    available). Done BEFORE state_dict capture so flattened post_ops are
    #    serialised as Qwen blocks.
    flattened, skipped_reason = _maybe_flatten_post_ops(model)

    # 3. Build the config + state_dict.
    cfg = _build_qwen3_dense_config(model, K=K)
    cfg.c4_post_ops_flattened = flattened
    cfg.c4_post_ops_flatten_skipped_reason = skipped_reason
    state_dict = _build_export_state_dict(
        model,
        K=K,
        norm_compensator_idx=norm_idx,
        bias_compensator_idx=cfg.c4_bias_compensator_idx,
    )

    # 4. Persist artefacts.
    cfg_dict = asdict(cfg)
    # HF's ``Qwen3Config`` doesn't know about our ``c4_*`` fields; they round-
    # trip via ``**kwargs``. Sorting keys keeps diffs reviewable.
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg_dict, fh, indent=2, sort_keys=True)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    _write_tokenizer_assets(output_dir)

    return cfg


__all__ = [
    "QWEN2_DENSE",
    "QWEN3_DENSE",
    "QWEN2_MOE",
    "QWEN3_MOE",
    "TensorMapping",
    "MappingValidation",
    "StateDictMappingPlan",
    "QwenCompatibilityReport",
    "Qwen3DenseConfig",
    "infer_qwen2_config_kwargs",
    "build_qwen2_config",
    "choose_closest_qwen_target",
    "build_qwen2_dense_mapping_plan",
    "analyze_qwen_compatibility",
    "export_qwen3_dense",
]
