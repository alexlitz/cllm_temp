"""Qwen compatibility planning for the Neural VM.

This module is intentionally non-invasive: it does not change model behavior
or perform checkpoint downloads. It builds a target Qwen config from an in-memory
VM model, explains direct-load blockers, and plans the state-dict key mapping
needed for a Qwen2/Qwen2.5-style dense export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

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
