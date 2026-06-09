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
from torch import nn


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


# Names of the composite FFN classes the Qwen export's R4 repack cannot
# consume directly (no top-level W_up / W_gate / W_down). See
# ``docs/QWEN_R8_E2E_2026_06_07.md`` Blocker 1 for context.
_COMPOSITE_FFN_CLASS_NAMES = frozenset({
    "AddSub5StageBlock",
    "FlattenedDivMod",
    "FlattenedALUMul",
    "ALUShiftComposite",
})


def _is_composite_ffn(ffn: Any) -> bool:
    """Return True if ``ffn`` is one of the documented composite blocks.

    Composite blocks lift the BD residual into a GE workspace, run several
    sub-FFN stages there, then project back. They don't expose a single
    SwiGLU ``W_up`` / ``W_gate`` / ``W_down`` triple on the residual stream.
    Identification is by class name (per the R8 doc), with a fallback to
    "lacks top-level W_up but is a non-trivial nn.Module".
    """

    if ffn is None:
        return False
    if type(ffn).__name__ in _COMPOSITE_FFN_CLASS_NAMES:
        return True
    if hasattr(ffn, "W_up") and isinstance(getattr(ffn, "W_up"), torch.Tensor):
        return False
    # Heuristic: a module without W_up but with submodules is treated as
    # composite. PureFFN subclasses always have W_up as a Parameter so this
    # check only fires for genuinely structured blocks.
    if hasattr(ffn, "_modules") and len(ffn._modules) > 0:
        return True
    return False


def _ffn_hidden_dim(ffn: Any) -> Optional[int]:
    if hasattr(ffn, "hidden_dim"):
        return int(ffn.hidden_dim)
    if hasattr(ffn, "W_up") and isinstance(ffn.W_up, torch.Tensor):
        return int(ffn.W_up.shape[0])
    return None


def _ffn_hidden_dims(model: Any) -> List[int]:
    """Return per-block FFN hidden dims, skipping composite blocks.

    Composite blocks (see ``_is_composite_ffn``) have no single
    ``hidden_dim`` because the SwiGLU triple is synthesised by
    :func:`extract_composite_ffn_weights` at export time. The Qwen3 dense
    config's ``intermediate_size`` is derived from the PureFFN blocks; the
    composite blocks then materialise zero-valued (d_model, intermediate)
    tensors so the Qwen MLP contributes a zero delta on those layers.
    """

    dims: List[int] = []
    for block in _blocks(model):
        ffn = getattr(block, "ffn", None)
        if _is_composite_ffn(ffn):
            continue
        hidden_dim = _ffn_hidden_dim(ffn)
        if hidden_dim is not None:
            dims.append(hidden_dim)
    return dims


def _iter_inner_pure_ffns(module: nn.Module) -> Iterable[Any]:
    """Yield every nested module that exposes a (W_up, W_gate, W_down) triple.

    Walks ``module.modules()`` (which is depth-first over all descendants
    including ``module`` itself). The yielded modules are the PureFFN-shaped
    sub-FFNs inside the composite blocks (e.g. ``AddRawAndGenFFN.ffn``).
    """

    for sub in module.modules():
        w_up = getattr(sub, "W_up", None)
        w_gate = getattr(sub, "W_gate", None)
        w_down = getattr(sub, "W_down", None)
        if (
            isinstance(w_up, torch.Tensor)
            and isinstance(w_gate, torch.Tensor)
            and isinstance(w_down, torch.Tensor)
        ):
            yield sub


def extract_composite_ffn_weights(
    block: Any,
    *,
    d_model: int,
    intermediate_size: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract ``(W_up, W_gate, W_down)`` for a composite-FFN block.

    Per Blocker 1 in ``docs/QWEN_R8_E2E_2026_06_07.md``: composite FFNs
    (``AddSub5StageBlock``, ``FlattenedDivMod``, ``FlattenedALUMul``,
    ``ALUShiftComposite``) project the BD residual stream into a GE
    workspace, run several sub-FFN stages there, then project back. None
    of those sub-FFNs operate directly on the model's d_model residual,
    so a flat Qwen SwiGLU triple cannot be a byte-identity equivalent.

    Fold feasibility (algebraic statement). A single Qwen SwiGLU MLP is
    ``down_proj(silu(gate_proj(x)) * up_proj(x))`` — exactly ONE SiLU
    nonlinearity and zero data-dependent gating. The composite blocks
    chain 3+ SwiGLU sub-FFNs in series in the GE workspace plus a
    runtime ``opcode_mask`` multiplication at the GE→BD writeback
    (gating on ``MARK_AX`` and per-position opcode markers from the
    residual). Function-composition of three SiLUs cannot be re-expressed
    as a single SiLU regardless of the chosen W_up/W_gate/W_down, and
    the opcode_mask is a function of the input residual that no static
    weight can replicate. Conclusion: no non-zero (W_up, W_gate, W_down)
    triple can match the composite's TRIGGERED-input behaviour. The
    deferred multi-block flatten (one Qwen layer per composite stage)
    is the only path to byte identity. See
    ``tests/test_qwen_composite_ffn_export.py::
    test_addsub5stage_fold_into_single_swiglu_infeasible_when_triggered``
    for the pinned counterexample.

    The R8 plan's path forward (one Qwen successor block per sub-stage)
    is deferred to a follow-up phase. Until then this helper makes export
    proceed without raising ``AttributeError`` by:

      1. Walking the block recursively to collect every inner PureFFN-
         shaped sub-FFN (those with all three of W_up / W_gate / W_down).
      2. For sub-FFNs whose input dim matches ``d_model`` (i.e. they
         already act on the residual stream): concatenate their hidden
         units into a single (intermediate_size, d_model) gate/up pair
         and (d_model, intermediate_size) down. Padding / truncation
         keeps the Qwen ``intermediate_size`` uniform across layers.
      3. For sub-FFNs that operate in a smaller GE workspace (the
         common case for L10/L12/L23/L26/L28): return zero-valued
         tensors of the Qwen-expected shape. With ``W_gate = 0`` the
         SwiGLU output ``silu(gate_proj(x)) * up_proj(x)`` is zero and
         the Qwen decoder layer's residual stream passes through
         unchanged — the same "zero-init skip-pass" idiom used by
         ``_make_skip_pass_block`` for post_ops.

         The zero-init MLP is EXACT (not approximate) on inputs where
         ``MARK_AX < 0.5`` or the relevant opcode markers are off,
         because in that regime the composite's GE→BD writeback
         multiplies all its writes by ``opcode_mask * (mark_ax > 0.5)
         == 0``, so the composite degenerates to residual-identity.
         The tiny-VM forward-parity test in
         ``tests/test_qwen_r8_e2e.py`` operates entirely in that
         regime, so the zero-init choice does not degrade the R8 tiny
         gate. See
         ``tests/test_qwen_composite_ffn_export.py::
         test_addsub5stage_composite_matches_qwen_skip_pass_mlp``.

    Args:
        block: A ``TransformerBlock`` whose ``block.ffn`` is a composite.
        d_model: Residual stream width (matches ``model.d_model``).
        intermediate_size: Qwen3-dense ``intermediate_size`` chosen by
            ``_build_qwen3_dense_config`` (= the largest PureFFN
            ``hidden_dim`` in the model).
        dtype: Output dtype. Defaults to ``torch.float32`` to match the
            rest of the export.

    Returns:
        Tuple ``(W_up, W_gate, W_down)`` shaped to match a PureFFN's
        attributes: ``W_up`` and ``W_gate`` are
        ``(intermediate_size, d_model)``; ``W_down`` is
        ``(d_model, intermediate_size)``. Downstream
        :func:`_swiglu_repack_and_fold_bias` consumes them via the normal
        ``ffn.W_up`` / ``ffn.W_gate`` / ``ffn.W_down`` interface.
    """

    ffn = getattr(block, "ffn", None)
    if ffn is None:
        raise AttributeError(
            "composite FFN extraction expected block.ffn to be present"
        )

    # Collect every inner PureFFN-shaped sub-FFN whose input dim already
    # matches d_model — those are usable on the residual stream directly.
    direct_subffns: List[Any] = []
    indirect_subffns: List[Any] = []
    for sub in _iter_inner_pure_ffns(ffn):
        w_up = sub.W_up
        w_up_dense = w_up.to_dense() if w_up.is_sparse else w_up
        if w_up_dense.dim() == 2 and int(w_up_dense.shape[1]) == int(d_model):
            direct_subffns.append(sub)
        else:
            indirect_subffns.append(sub)

    # Allocate zero-init output buffers shaped for the Qwen export.
    W_up = torch.zeros((int(intermediate_size), int(d_model)), dtype=dtype)
    W_gate = torch.zeros((int(intermediate_size), int(d_model)), dtype=dtype)
    W_down = torch.zeros((int(d_model), int(intermediate_size)), dtype=dtype)

    if not direct_subffns:
        # Pure-composite case (the documented L10/L12/L23/L26/L28): no
        # sub-FFN acts on d_model directly. Zero weights → SwiGLU output
        # is zero → Qwen residual passes through unchanged.
        return W_up, W_gate, W_down

    # Direct sub-FFNs exist. Concatenate them along the hidden axis,
    # zero-padding (or truncating) to fit ``intermediate_size``.
    write_cursor = 0
    for sub in direct_subffns:
        if write_cursor >= int(intermediate_size):
            break
        sub_w_up = sub.W_up.to_dense() if sub.W_up.is_sparse else sub.W_up.data
        sub_w_gate = (
            sub.W_gate.to_dense() if sub.W_gate.is_sparse else sub.W_gate.data
        )
        sub_w_down = (
            sub.W_down.to_dense() if sub.W_down.is_sparse else sub.W_down.data
        )
        sub_hidden = int(sub_w_up.shape[0])
        end = min(write_cursor + sub_hidden, int(intermediate_size))
        copy_h = end - write_cursor
        W_up[write_cursor:end, :] = sub_w_up[:copy_h, :].to(dtype)
        W_gate[write_cursor:end, :] = sub_w_gate[:copy_h, :].to(dtype)
        # W_down's hidden axis is dim=1.
        W_down[:, write_cursor:end] = sub_w_down[:, :copy_h].to(dtype)
        write_cursor = end

    return W_up, W_gate, W_down


def _first_attention(model: Any) -> Any:
    blocks = _blocks(model)
    if not blocks:
        return None
    return getattr(blocks[0], "attn", None)


def _attn_head_counts(model: Any) -> List[int]:
    """Return per-block attention head counts.

    The VM has heterogeneous per-block attention widths — most blocks use
    the default 8 heads, but a few are wider (production model: L13 has
    12 heads, L31 has 14 heads). Qwen3-dense requires a uniform
    ``num_attention_heads`` across all layers, so the export config picks
    ``max(_attn_head_counts(model))`` and :func:`_build_export_state_dict`
    zero-pads short blocks' Q/K/V rows and W_o columns up to that max.

    Padding with zeros is semantically a no-op: the extra padded heads
    have Q=K=V=0 (so their attention output is 0 regardless of softmax)
    AND W_o has zero columns for them (so any non-zero attention output
    is multiplied by 0). The contribution of every padded head is
    therefore exactly zero on every input.
    """

    counts: List[int] = []
    for block in _blocks(model):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        n = getattr(attn, "num_heads", None)
        if n is None:
            continue
        counts.append(int(n))
    return counts


def _pad_attn_proj_rows(t: torch.Tensor, *, target_rows: int) -> torch.Tensor:
    """Zero-pad a (rows, d_model) attention projection up to ``target_rows``.

    Used for ``W_q`` / ``W_k`` / ``W_v`` whose first axis is
    ``n_heads * head_dim``. Padding with zero rows materialises extra
    "phantom" heads whose Q/K/V output is zero on every input.
    """

    current = int(t.shape[0])
    if current >= int(target_rows):
        return t
    pad_amount = int(target_rows) - current
    return torch.nn.functional.pad(
        t, [0, 0, 0, pad_amount], mode="constant", value=0.0
    )


def _pad_attn_o_cols(t: torch.Tensor, *, target_cols: int) -> torch.Tensor:
    """Zero-pad a (d_model, cols) ``W_o`` up to ``target_cols`` along axis=1.

    Mirror of :func:`_pad_attn_proj_rows` for the output projection. With
    zero columns for the phantom heads, even a non-zero attention output
    from those heads would be multiplied by zero before contributing to
    the residual stream.
    """

    current = int(t.shape[1])
    if current >= int(target_cols):
        return t
    pad_amount = int(target_cols) - current
    return torch.nn.functional.pad(
        t, [0, pad_amount, 0, 0], mode="constant", value=0.0
    )


# Sentinel magnitude for the per-head q_norm / k_norm identity bake.
# Must dominate typical ``RMS(q_head)`` so RMSNorm's denominator is set
# by the sentinel slot. Production VM Q/K magnitudes are O(1..10); 100
# leaves a >10x margin while keeping ``K_h^2 / sqrt(head_dim)`` (the
# constant sentinel contribution to attention scores) at a softmax-safe
# magnitude. See :func:`_bake_qk_norm_sentinel` for the derivation.
_QK_NORM_SENTINEL_K_H = 100.0


def _bake_qk_norm_sentinel(
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    *,
    n_heads: int,
    head_dim: int,
    norm_compensator_idx: int,
    K: float,
    K_h: float = _QK_NORM_SENTINEL_K_H,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bake a per-head sentinel into ``W_q`` / ``W_k`` so q_norm/k_norm act as identity.

    Qwen3 applies ``q_norm`` / ``k_norm`` per head over ``head_dim``: the
    output of q_proj / k_proj is reshaped to ``(..., n_heads, head_dim)``
    and each head slice is RMSNorm'd. The VM's ``W_q`` / ``W_k`` were
    not trained against q_norm, so the natural ``q_head`` magnitude
    varies arbitrarily — a constant ``q_norm.weight`` cannot recover
    identity (Blocker 3 in ``docs/QWEN_R8_E2E_2026_06_07.md``).

    This helper installs a per-head **sentinel pair** at the last two
    head-dim indices (``s-1`` and ``s = head_dim - 1``). The transform:

      1. Zero ``W_q[h*head_dim + s - 1, :]`` and ``W_q[h*head_dim + s, :]``
         for every head ``h in [0, n_heads)`` — the sentinel pair contributes
         nothing from the original input.
      2. Set ``W_q[h*head_dim + s, norm_compensator_idx] = K_h / K``. The
         residual NORM_COMPENSATOR slot carries ``K`` on every real token,
         so the sentinel row outputs ``q_head[s] = K_h`` independent of
         position.

    With ``K_h`` larger than the natural ``RMS(q_head)`` of the other
    ``head_dim - 2`` slots, the per-head RMS is dominated by the sentinel:
    ``mean(q_head^2) ≈ K_h^2 / head_dim``. Pairing this with
    ``q_norm.weight = (K_h / sqrt(head_dim)) * ones`` makes RMSNorm an
    identity on every head: ``out[i] = w[i] * q[i] / RMS(q) ≈ q[i]``.

    The same recipe is applied to ``W_k`` so ``k_head`` also carries
    ``K_h`` at index ``s``. The attention dot product gets a constant
    ``K_h^2`` contribution from the sentinel slot (softmax-invariant).

    RoPE caveat: the sentinel pair (``s-1``, ``s``) lives at the lowest
    RoPE frequency (``theta^(-(head_dim/2 - 1) / (head_dim/2))``), where
    rotation over the test window (~30 positions) is essentially zero,
    so ``q_head[s] ≈ K_h`` after RoPE as well. The variation in the
    constant sentinel contribution to attention is O(rotation^2), which
    is negligible compared to typical attention score magnitudes.

    Args:
        W_q: Q projection weight, shape ``(n_heads * head_dim, d_model)``.
        W_k: K projection weight, same shape.
        n_heads: Number of attention heads (uniform across exported layers).
        head_dim: Per-head dimension.
        norm_compensator_idx: Residual index of the NORM_COMPENSATOR slot.
        K: NORM_COMPENSATOR residual value (e.g. 1000.0).
        K_h: Sentinel magnitude — must dominate ``RMS(q_head_orig)``.

    Returns:
        ``(W_q, W_k)`` with the sentinel baked in (fresh tensors).
    """

    if n_heads <= 0 or head_dim < 2 or norm_compensator_idx < 0:
        return W_q, W_k
    expected_rows = n_heads * head_dim
    if W_q.shape[0] < expected_rows or W_k.shape[0] < expected_rows:
        return W_q, W_k
    if norm_compensator_idx >= W_q.shape[1]:
        return W_q, W_k

    W_q_out = W_q.clone()
    W_k_out = W_k.clone()
    scale = float(K_h) / float(K) if K != 0 else 0.0

    for h in range(int(n_heads)):
        last = h * head_dim + head_dim - 1
        prev = last - 1
        # Zero the RoPE-paired sentinel rows so the original input's
        # contribution to those head slots is gone.
        W_q_out[prev, :] = 0.0
        W_q_out[last, :] = 0.0
        W_k_out[prev, :] = 0.0
        W_k_out[last, :] = 0.0
        # Inject the sentinel at the last index only. After RoPE, the
        # pair rotates: q_rope[last] ≈ cos*K_h, q_rope[prev] ≈ sin*K_h.
        # At lowest RoPE frequency over small windows cos ≈ 1, sin ≈ 0,
        # so the sentinel is approximately position-invariant.
        W_q_out[last, norm_compensator_idx] = scale
        W_k_out[last, norm_compensator_idx] = scale

    return W_q_out, W_k_out


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
    # ID of the reserved sink-token in the exported vocab. The inference
    # caller must prepend this token at position 0 of every input_ids
    # sequence so the K=0/V=0 sink position is materialised and standard
    # softmax over (sink + real) reproduces softmax1 over real positions.
    # See Blocker 2 in QWEN_R8_E2E_2026_06_07.md and R3 prototype.
    c4_softmax_sink_token_id: Optional[int] = None
    c4_post_ops_flattened: bool = False
    c4_post_ops_flatten_skipped_reason: Optional[str] = None
    # R8 Blocker 5 — NeuralVMEmbedding augmentations metadata. The exported
    # ``model.embed_tokens`` is a plain ``nn.Embedding`` whose row for the
    # original VM vocabulary matches ``model.embed.embed.weight``. The
    # ADDR_KEY and MEM_STORE augmentations that ``NeuralVMEmbedding.forward``
    # adds on top of that lookup are NOT writable into the (vocab_size,
    # d_model) table — ADDR_KEY is a per-position one-hot gated by the
    # in-sequence CODE_END detector, and MEM_STORE depends on per-row state
    # populated by the KV-cache eviction logic. Both are exported as a
    # documented runtime wrapper protocol (see :class:`NeuralVMEmbeddingWrapper`
    # and :func:`apply_neural_vm_embedding_augmentations`). These config
    # fields tell a downstream loader where to write the deltas and which
    # token id marks CODE_END / MEM.
    c4_addr_key_idx: Optional[int] = None
    c4_addr_key_width: int = 48
    c4_mem_store_idx: Optional[int] = None
    c4_mem_addr_src_idx: Optional[int] = None
    c4_addr_key_pc_offset: int = 2
    c4_addr_key_instr_width: int = 8
    c4_addr_key_data_bytes: int = 5
    c4_code_end_token_id: int = 265
    c4_mem_token_id: int = 261


def _build_qwen3_dense_config(model: Any, *, K: float) -> Qwen3DenseConfig:
    """Derive the Qwen3-dense config kwargs from the VM model."""

    blocks = _blocks(model)
    attn = _first_attention(model)
    ffn_dims = _ffn_hidden_dims(model)
    d_model = int(getattr(model, "d_model", getattr(attn, "dim", 0)))
    # Qwen3 dense requires a uniform ``num_attention_heads`` across all
    # layers. The VM has heterogeneous per-block attention widths
    # (production model: most blocks use 8 heads but L13 has 12 and L31
    # has 14). The head_dim is uniform (= d_model / default_n_heads), so
    # the wider layers manifest as more (head_dim-wide) Q/K/V rows. The
    # config picks ``max`` and ``_build_export_state_dict`` zero-pads
    # short blocks' Q/K/V/O up to that max; padded heads have Q=K=V=0
    # AND W_o has zero columns for them, so their contribution is exactly
    # zero on every input.
    head_counts = _attn_head_counts(model)
    base_n_heads = int(getattr(attn, "num_heads", 1))
    n_heads = max(head_counts) if head_counts else base_n_heads
    # head_dim is taken from the base (first) layer because it is uniform
    # across the VM's attention layers; the per-layer total Q/K/V width
    # is ``num_heads * head_dim`` and only ``num_heads`` varies.
    head_dim = d_model // base_n_heads if base_n_heads else 0
    # Qwen3 dense requires a uniform ``intermediate_size`` across all layers.
    # The VM has heterogeneous PureFFN widths (production model: widths in
    # {1, 7, 8, 42, 64, 192, 512, 792, 1536, 1846, 1890, 2059, 3405, 4096}).
    # ``_build_export_state_dict`` zero-pads every block to ``max(ffn_dims)``,
    # so the config must match — otherwise ``torch.save`` records tensor
    # shapes that disagree with the config's ``intermediate_size`` and the
    # ZIP record layout drifts (observed: ``unexpected pos N vs M``).
    intermediate_size = max(ffn_dims) if ffn_dims else 0
    dim_positions = getattr(model, "dim_positions", {}) or {}
    if not isinstance(dim_positions, dict):
        # ``_SetDim`` fallback uses ``__getitem__``; coerce to a plain dict via
        # iterating its public attributes when possible. For the simple-dict
        # case this is a no-op.
        try:
            dim_positions = dict(dim_positions)
        except Exception:  # pragma: no cover - exotic dim registries
            dim_positions = {}

    # R3 sink: reserve one extra vocab slot (the original ``vocab_size``)
    # for the softmax-sink token. Its embedding row is zero, so K=V=0 for
    # the sink position and standard softmax over (sink + real) reproduces
    # softmax1 over the real positions. The exported ``vocab_size`` is the
    # VM's vocab_size + 1.
    base_vocab_size = int(getattr(model, "vocab_size"))
    sink_token_id = base_vocab_size  # zero-indexed, sits past the VM vocab
    exported_vocab_size = base_vocab_size + 1

    # R8 Blocker 5 — record the dim positions that NeuralVMEmbedding writes
    # into. ``ADDR_KEY`` is a 48-dim band (three one-hot nibbles); ``MEM_STORE``
    # and ``MEM_ADDR_SRC`` are 1-dim flags. The loader wrapper uses these to
    # reproduce the augmentations on top of the plain ``nn.Embedding`` lookup.
    addr_key_idx = dim_positions.get("ADDR_KEY")
    mem_store_idx = dim_positions.get("MEM_STORE")
    mem_addr_src_idx = dim_positions.get("MEM_ADDR_SRC")
    from .constants import INSTR_WIDTH, PC_OFFSET
    from .vm_step import Token

    return Qwen3DenseConfig(
        vocab_size=exported_vocab_size,
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
        c4_softmax_sink_token_id=sink_token_id,
        c4_addr_key_idx=int(addr_key_idx) if addr_key_idx is not None else None,
        c4_mem_store_idx=int(mem_store_idx) if mem_store_idx is not None else None,
        c4_mem_addr_src_idx=(
            int(mem_addr_src_idx) if mem_addr_src_idx is not None else None
        ),
        c4_addr_key_pc_offset=int(PC_OFFSET),
        c4_addr_key_instr_width=int(INSTR_WIDTH),
        c4_addr_key_data_bytes=5,
        c4_code_end_token_id=int(Token.CODE_END),
        c4_mem_token_id=int(Token.MEM),
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


def _pad_or_truncate_swiglu_proj(
    t: torch.Tensor,
    *,
    target_hidden: int,
    axis: int,
) -> torch.Tensor:
    """Zero-pad or truncate a SwiGLU projection along the hidden axis.

    Qwen3-dense requires a uniform ``intermediate_size`` across all
    layers. The VM has heterogeneous per-block FFN widths (production
    model: widths in ``{0, 1, 8, 42, 64, 192, 512, 792, 1536, 4096}``),
    so we pad short blocks with zeros along the hidden axis and truncate
    over-wide blocks. Padding is a no-op semantically: zero hidden units
    contribute zero to the SwiGLU sum.

    Args:
        t: The (already repacked) projection tensor.
        target_hidden: The Qwen3 ``intermediate_size``.
        axis: Hidden axis. For ``gate_proj`` / ``up_proj`` the shape is
            ``(hidden, d_model)`` so ``axis=0``; for ``down_proj`` it is
            ``(d_model, hidden)`` so ``axis=1``.

    Returns:
        A tensor with size ``target_hidden`` along ``axis``.
    """

    current = int(t.shape[axis])
    if current == int(target_hidden):
        return t
    if current > int(target_hidden):
        # Truncate. Almost never hits — intermediate_size is
        # max(hidden_dims) — but keep the branch for safety.
        index = torch.arange(int(target_hidden))
        return t.index_select(axis, index)
    # Pad with zeros along ``axis``. ``torch.nn.functional.pad`` pads
    # right-to-left starting from the last dim — translate the axis.
    pad_amount = int(target_hidden) - current
    pad_spec: List[int] = []
    rank = t.dim()
    for dim_idx in range(rank - 1, -1, -1):
        if dim_idx == axis:
            pad_spec.extend([0, pad_amount])
        else:
            pad_spec.extend([0, 0])
    return torch.nn.functional.pad(t, pad_spec, mode="constant", value=0.0)


class _CompositeFFNAdapter:
    """Minimal duck-typed adapter so :func:`_swiglu_repack_and_fold_bias`
    can consume the output of :func:`extract_composite_ffn_weights`.

    PureFFN-style modules expose ``.W_up.data`` / ``.W_gate.data`` /
    ``.W_down.data`` as Parameters. The adapter wraps plain tensors and
    presents them under the same attribute path so the repack helper
    needs no branching. ``b_up`` / ``b_gate`` are absent (zero-bias) so
    the bias-fold path is a no-op.
    """

    def __init__(
        self,
        *,
        W_up: torch.Tensor,
        W_gate: torch.Tensor,
        W_down: torch.Tensor,
    ) -> None:
        self.W_up = _CompositeParamView(W_up)
        self.W_gate = _CompositeParamView(W_gate)
        self.W_down = _CompositeParamView(W_down)
        # b_up / b_gate intentionally omitted — composite SwiGLU is
        # zero-init so a bias-fold would have nothing to absorb.


class _CompositeParamView:
    """Object exposing ``.data`` and ``.is_sparse`` like ``nn.Parameter``."""

    def __init__(self, t: torch.Tensor) -> None:
        self.data = t
        self.is_sparse = bool(t.is_sparse)


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
    """Return K/V projection weights unchanged.

    The softmax-sink is realised as a reserved sink-token id whose embedding
    row in ``model.embed_tokens.weight`` is zero. Since Qwen3 dense has
    ``attention_bias=False`` (no bias on ``k_proj`` / ``v_proj``), the linear
    projection of a zero residual is zero, so the sink position contributes
    ``K = V = 0`` to attention. Standard softmax over (sink + real
    positions) then reproduces softmax1 over the real positions (R3 math).

    The K/V projection weights themselves do NOT need patching: see
    :func:`prepend_softmax_sink` for the inference-time helper that
    materialises the sink position by prepending the reserved token to
    ``input_ids``.
    """

    return W_k.clone(), W_v.clone()


def prepend_softmax_sink(
    input_ids: torch.Tensor,
    sink_token_id: int,
) -> torch.Tensor:
    """Prepend the reserved sink-token to ``input_ids`` along the seq dim.

    R3 / Blocker 2 wiring: standard softmax over a sequence whose first
    K/V entries are zero is identical to softmax1 over the remaining
    (real) positions. The exported model reserves
    ``c4_softmax_sink_token_id`` (= original ``vocab_size``) with a
    zero embedding row, so prepending that id to ``input_ids`` injects
    the K=0 / V=0 sink position automatically.

    Args:
        input_ids: ``(B, T)`` long tensor of token ids.
        sink_token_id: ``cfg.c4_softmax_sink_token_id`` from the exported
            config.

    Returns:
        ``(B, T + 1)`` long tensor with ``sink_token_id`` at column 0.
    """

    if input_ids.dim() != 2:
        raise ValueError(
            f"prepend_softmax_sink expects (B, T) input_ids, got shape "
            f"{tuple(input_ids.shape)}"
        )
    sink_col = torch.full(
        (input_ids.shape[0], 1),
        int(sink_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([sink_col, input_ids], dim=1)


# ---------------------------------------------------------------------------
# Phase R8 / Blocker 5 — NeuralVMEmbedding augmentation export
# ---------------------------------------------------------------------------
#
# ``NeuralVMEmbedding.forward`` (see ``neural_embedding.py``) runs three
# components on top of a plain ``nn.Embedding`` lookup:
#
#   1. The base embedding table — exported losslessly as
#      ``model.embed_tokens.weight``.
#   2. ``_add_code_addr_keys`` — a position-only one-hot scatter into the
#      ``ADDR_KEY`` residual band (48 dims), gated by an in-sequence
#      ``CODE_END`` detector. Mathematically a deterministic function of
#      ``(position, token_ids == CODE_END)``; recoverable from input_ids
#      alone.
#   3. ``_inject_mem_store`` — sets a 1-dim ``MEM_STORE`` flag (and an
#      optional ``MEM_ADDR_SRC`` flag) at MEM-marker positions inside the
#      retained history region. Depends on runtime state set externally:
#      ``_mem_history_end``, ``_mem_store_positions``,
#      ``_mem_addr_src_positions``. NOT computable from input_ids alone.
#
# Because (2) depends on position and (3) depends on runtime state, the
# augmentations cannot be folded into a ``(vocab_size, d_model)`` lookup
# table. They are exported as a documented **runtime wrapper protocol**:
# the loader calls :func:`apply_neural_vm_embedding_augmentations` (or
# wraps the Qwen embed_tokens with :class:`NeuralVMEmbeddingWrapper`) and
# the resulting tensor matches ``NeuralVMEmbedding.forward(input_ids)``
# bit-for-bit when the dim_positions / state inputs match.
#
# The exported ``config.json`` records ``c4_addr_key_idx``,
# ``c4_mem_store_idx``, ``c4_mem_addr_src_idx``, the ADDR_KEY layout
# constants, and the ``CODE_END`` / ``MEM`` token ids so the loader can
# reconstruct the augmentation surface without importing ``neural_vm``.


def apply_neural_vm_embedding_augmentations(
    embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    addr_key_idx: Optional[int],
    addr_key_width: int = 48,
    mem_store_idx: Optional[int] = None,
    mem_addr_src_idx: Optional[int] = None,
    code_end_token_id: int = 265,
    mem_token_id: int = 261,
    pc_offset: int = 2,
    instr_width: int = 8,
    data_bytes: int = 5,
    mem_history_end: int = 0,
    mem_store_positions: Optional[List[List[int]]] = None,
    mem_addr_src_positions: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Apply ADDR_KEY / MEM_STORE augmentations on top of a Qwen embedding.

    Drop-in equivalent to ``NeuralVMEmbedding.forward(input_ids)`` when
    given the same ``input_ids``, ``embeddings = embed_tokens(input_ids)``,
    and the matching dim positions / token ids. Returns the augmented
    residual ``(B, T, d_model)``. The input tensor is NOT mutated; a clone
    is augmented in-place and returned (callers can pass a freshly looked-up
    embedding directly).

    Args:
        embeddings: ``(B, T, d_model)`` base embedding lookup output. Cloned
            before mutation so the caller's tensor is left intact.
        input_ids: ``(B, T)`` long token ids, used to gate ADDR_KEY (by
            ``CODE_END``) and MEM_STORE (by ``MEM``).
        addr_key_idx: starting column of the ADDR_KEY band. When ``None``
            the ADDR_KEY scatter is skipped — useful for tiny test VMs whose
            residual layout doesn't carry an ADDR_KEY slot.
        addr_key_width: ADDR_KEY band width (always 48 = 3 nibbles * 16).
        mem_store_idx: column of the MEM_STORE flag, or ``None`` to skip.
        mem_addr_src_idx: column of the MEM_ADDR_SRC flag, or ``None`` to
            skip.
        code_end_token_id: vocab id of the ``CODE_END`` marker.
        mem_token_id: vocab id of the ``MEM`` marker.
        pc_offset, instr_width, data_bytes: ADDR_KEY layout constants
            mirroring ``PC_OFFSET`` / ``INSTR_WIDTH`` / ``DATA_BYTES`` in
            ``neural_vm.constants`` and ``neural_embedding._build_addr_key_pos_encoding``.
        mem_history_end: end of the retained MEM history region. The
            wrapper writes MEM_STORE=1 / MEM_ADDR_SRC=1 at every position
            ``[0, mem_history_end)`` whose token equals ``MEM``.
        mem_store_positions: optional per-row list of tracked MEM marker
            positions. Mirrors ``NeuralVMEmbedding.set_mem_store_positions``.
        mem_addr_src_positions: optional per-row subset of
            ``mem_store_positions`` that should also receive MEM_ADDR_SRC=1.
            When ``None`` every tracked store row receives both flags.

    Returns:
        ``(B, T, d_model)`` augmented residual stream.
    """

    import torch.nn.functional as F

    if embeddings.dim() != 3:
        raise ValueError(
            f"embeddings must be (B, T, d_model); got {tuple(embeddings.shape)}"
        )
    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must be (B, T); got {tuple(input_ids.shape)}"
        )
    if embeddings.shape[:2] != input_ids.shape:
        raise ValueError(
            f"embeddings/input_ids shape mismatch: {tuple(embeddings.shape[:2])} "
            f"vs {tuple(input_ids.shape)}"
        )

    x = embeddings.clone()
    B, S = input_ids.shape
    device = x.device
    dtype = x.dtype

    # --- ADDR_KEY scatter -------------------------------------------------
    if addr_key_idx is not None and addr_key_width > 0:
        BYTES_PER_INSTR = int(instr_width)
        DATA_BYTES = int(data_bytes)
        pos = torch.arange(S, device=device, dtype=torch.long)
        seq_pos = pos - 1
        byte_offset = seq_pos % BYTES_PER_INSTR
        instr_idx = seq_pos // BYTES_PER_INSTR
        addr = instr_idx * int(instr_width) + int(pc_offset) + byte_offset

        pos_valid = (seq_pos >= 0) & (byte_offset < DATA_BYTES)

        # CODE_END detector: a position is "before the first CODE_END" iff
        # the cumulative count of CODE_END tokens up to and including that
        # position is zero. Per-batch.
        is_code_end = (input_ids == int(code_end_token_id)).to(torch.long)
        before_code_end = (is_code_end.cumsum(dim=1) == 0)
        mask = before_code_end & pos_valid.unsqueeze(0)
        mask_f = mask.to(dtype).unsqueeze(-1)

        # Three nibble one-hots. ``clamp`` mirrors the embed code's defensive
        # guard so F.one_hot never sees an out-of-range index on the i==0
        # row that the mask gates to zero anyway.
        lo = (addr % 16).clamp(min=0, max=15)
        hi = ((addr // 16) % 16).clamp(min=0, max=15)
        top = ((addr // 256) % 16).clamp(min=0, max=15)
        oh_lo = F.one_hot(lo, num_classes=16).to(dtype)
        oh_hi = F.one_hot(hi, num_classes=16).to(dtype)
        oh_top = F.one_hot(top, num_classes=16).to(dtype)
        enc = torch.cat([oh_lo, oh_hi, oh_top], dim=-1)  # [S, 48]

        gated = enc.unsqueeze(0) * mask_f
        ak = int(addr_key_idx)
        x[:, :, ak:ak + int(addr_key_width)].add_(gated)

    # --- MEM_STORE / MEM_ADDR_SRC scatter ---------------------------------
    if mem_store_idx is None and mem_addr_src_idx is None:
        return x

    rows: List[int] = []
    cols: List[int] = []
    addr_src_flags: List[int] = []

    if mem_store_positions is not None:
        for b in range(min(B, len(mem_store_positions))):
            row_positions = mem_store_positions[b]
            if not row_positions:
                continue
            if (
                mem_addr_src_positions is not None
                and b < len(mem_addr_src_positions)
            ):
                addr_src_row = frozenset(int(p) for p in mem_addr_src_positions[b])
                addr_src_row_is_subset = True
            else:
                addr_src_row = None
                addr_src_row_is_subset = False
            for pos_i in row_positions:
                pos_i = int(pos_i)
                if not (0 <= pos_i < S):
                    continue
                rows.append(b)
                cols.append(pos_i)
                if not addr_src_row_is_subset:
                    addr_src_flags.append(1)
                elif pos_i in addr_src_row:
                    addr_src_flags.append(1)
                else:
                    addr_src_flags.append(0)

    if mem_history_end > 0:
        limit = min(int(mem_history_end), S)
        for b in range(B):
            for i in range(0, limit):
                rows.append(b)
                cols.append(i)
                addr_src_flags.append(1)

    if not rows:
        return x

    row_idx = torch.tensor(rows, dtype=torch.long, device=device)
    col_idx = torch.tensor(cols, dtype=torch.long, device=device)
    addr_src_mask = torch.tensor(addr_src_flags, dtype=torch.bool, device=device)

    gathered = input_ids[row_idx, col_idx]
    is_mem = gathered == int(mem_token_id)
    if not bool(is_mem.any()):
        return x

    keep_row = row_idx[is_mem]
    keep_col = col_idx[is_mem]
    keep_addr_src = addr_src_mask[is_mem]

    ones = torch.ones(keep_row.shape[0], device=device, dtype=dtype)

    if mem_store_idx is not None:
        x.index_put_(
            (keep_row, keep_col, torch.full_like(keep_row, int(mem_store_idx))),
            ones,
        )
    if mem_addr_src_idx is not None and bool(keep_addr_src.any()):
        sub_row = keep_row[keep_addr_src]
        sub_col = keep_col[keep_addr_src]
        sub_ones = ones[: sub_row.shape[0]]
        x.index_put_(
            (sub_row, sub_col, torch.full_like(sub_row, int(mem_addr_src_idx))),
            sub_ones,
        )

    return x


class NeuralVMEmbeddingWrapper(nn.Module):
    """Module wrapper that re-applies NeuralVMEmbedding augmentations on Qwen.

    Wraps an ``nn.Embedding`` (typically the exported
    ``model.embed_tokens``) and adds the ADDR_KEY / MEM_STORE augmentations
    that ``NeuralVMEmbedding.forward`` runs in the native VM. The MEM_STORE
    state — ``mem_history_end``, ``mem_store_positions``,
    ``mem_addr_src_positions`` — is mutable via the same ``set_*`` setters
    the native embedding exposes so callers can drop this in front of a
    Qwen3 ``model.embed_tokens`` without changing their step-loop wiring.

    Construct directly with the relevant dim positions, or use
    :meth:`from_config` to build it from an exported
    :class:`Qwen3DenseConfig` (round-trips via ``config.json``).
    """

    def __init__(
        self,
        embed: nn.Embedding,
        *,
        addr_key_idx: Optional[int],
        addr_key_width: int = 48,
        mem_store_idx: Optional[int] = None,
        mem_addr_src_idx: Optional[int] = None,
        code_end_token_id: int = 265,
        mem_token_id: int = 261,
        pc_offset: int = 2,
        instr_width: int = 8,
        data_bytes: int = 5,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.addr_key_idx = addr_key_idx
        self.addr_key_width = int(addr_key_width)
        self.mem_store_idx = mem_store_idx
        self.mem_addr_src_idx = mem_addr_src_idx
        self.code_end_token_id = int(code_end_token_id)
        self.mem_token_id = int(mem_token_id)
        self.pc_offset = int(pc_offset)
        self.instr_width = int(instr_width)
        self.data_bytes = int(data_bytes)

        # Mirror NeuralVMEmbedding's mutable runtime state. Setters below
        # match the native API so callers can swap in this module without
        # touching their KV-cache eviction code path.
        self._mem_history_end: int = 0
        self._mem_store_positions: Optional[List[List[int]]] = None
        self._mem_addr_src_positions: Optional[List[List[int]]] = None

    @classmethod
    def from_config(
        cls,
        embed: nn.Embedding,
        config: Mapping[str, Any],
    ) -> "NeuralVMEmbeddingWrapper":
        """Construct from an exported ``config.json`` dict (or ``Qwen3DenseConfig``)."""

        if hasattr(config, "__dataclass_fields__"):
            cfg = asdict(config)
        else:
            cfg = dict(config)
        return cls(
            embed,
            addr_key_idx=cfg.get("c4_addr_key_idx"),
            addr_key_width=int(cfg.get("c4_addr_key_width", 48)),
            mem_store_idx=cfg.get("c4_mem_store_idx"),
            mem_addr_src_idx=cfg.get("c4_mem_addr_src_idx"),
            code_end_token_id=int(cfg.get("c4_code_end_token_id", 265)),
            mem_token_id=int(cfg.get("c4_mem_token_id", 261)),
            pc_offset=int(cfg.get("c4_addr_key_pc_offset", 2)),
            instr_width=int(cfg.get("c4_addr_key_instr_width", 8)),
            data_bytes=int(cfg.get("c4_addr_key_data_bytes", 5)),
        )

    def set_mem_history_end(self, end: int) -> None:
        self._mem_history_end = int(end)

    def set_mem_store_positions(self, positions: Optional[List[List[int]]]) -> None:
        self._mem_store_positions = positions

    def set_mem_addr_src_positions(
        self, positions: Optional[List[List[int]]]
    ) -> None:
        self._mem_addr_src_positions = positions

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.embed(input_ids)
        return apply_neural_vm_embedding_augmentations(
            base,
            input_ids,
            addr_key_idx=self.addr_key_idx,
            addr_key_width=self.addr_key_width,
            mem_store_idx=self.mem_store_idx,
            mem_addr_src_idx=self.mem_addr_src_idx,
            code_end_token_id=self.code_end_token_id,
            mem_token_id=self.mem_token_id,
            pc_offset=self.pc_offset,
            instr_width=self.instr_width,
            data_bytes=self.data_bytes,
            mem_history_end=self._mem_history_end,
            mem_store_positions=self._mem_store_positions,
            mem_addr_src_positions=self._mem_addr_src_positions,
        )


def install_neural_vm_embedding_wrapper(
    qmodel: Any,
    config: Any,
) -> "NeuralVMEmbeddingWrapper":
    """Swap a loaded HF Qwen3 model's ``embed_tokens`` for the augmentation wrapper.

    Plain ``AutoModelForCausalLM.from_pretrained`` reconstructs
    ``model.embed_tokens`` as an ``nn.Embedding`` — the ADDR_KEY scatter and
    MEM_STORE / MEM_ADDR_SRC writes that :class:`NeuralVMEmbedding` performs in
    the native VM are silently dropped on the Qwen3 forward path, which leaves
    L5/L7 content-addressed bytecode fetch (and the L15 retained-history lookup)
    without their key columns. This helper rebuilds the wrapper from the
    exported config (``c4_addr_key_idx`` / ``c4_mem_store_idx`` /
    ``c4_mem_addr_src_idx``) and swaps it in-place at
    ``qmodel.model.embed_tokens`` so the augmentations apply on the next
    forward.

    The original ``nn.Embedding`` is preserved as ``wrapper.embed``.
    ``tie_word_embeddings`` is False on every export (see
    :func:`_build_qwen3_dense_config`), so swapping the module does not affect
    ``lm_head``. The wrapper is returned so callers can drive the mutable MEM
    state (``set_mem_history_end`` / ``set_mem_store_positions`` /
    ``set_mem_addr_src_positions``) without re-walking the model tree. Calling
    this helper twice is idempotent — the second call returns the existing
    wrapper unchanged.

    Args:
        qmodel: a loaded ``Qwen3ForCausalLM`` (or any object with a
            ``model.embed_tokens`` attribute) freshly returned from
            ``AutoModelForCausalLM.from_pretrained``.
        config: an exported :class:`Qwen3DenseConfig`, the HF
            ``PretrainedConfig`` from ``qmodel.config`` (whose ``c4_*`` fields
            were round-tripped from ``config.json``), or a plain dict.

    Returns:
        The installed :class:`NeuralVMEmbeddingWrapper`.
    """

    inner = qmodel.model.embed_tokens
    if isinstance(inner, NeuralVMEmbeddingWrapper):
        return inner  # idempotent — re-install is a no-op.
    if not isinstance(inner, nn.Embedding):
        raise TypeError(
            "install_neural_vm_embedding_wrapper expects qmodel.model.embed_tokens "
            f"to be an nn.Embedding (or NeuralVMEmbeddingWrapper); got {type(inner)!r}."
        )

    if hasattr(config, "to_dict"):
        cfg_dict: Mapping[str, Any] = config.to_dict()
    elif hasattr(config, "__dataclass_fields__"):
        cfg_dict = asdict(config)
    elif isinstance(config, Mapping):
        cfg_dict = config
    elif hasattr(config, "__dict__"):
        cfg_dict = vars(config)
    else:
        cfg_dict = dict(config)

    wrapper = NeuralVMEmbeddingWrapper.from_config(inner, cfg_dict)
    qmodel.model.embed_tokens = wrapper
    return wrapper


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
    sink_token_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Assemble the HuggingFace-style state_dict for ``Qwen3ForCausalLM``.

    When ``sink_token_id`` is provided (R3 softmax-sink wiring), the
    exported embedding table and LM head are extended by one row whose
    embedding values are all zero. Loaders that prepend the sink token
    at inference time (see :func:`prepend_softmax_sink`) will then see
    K = V = 0 at the sink position because Qwen3 dense has
    ``attention_bias=False`` — zero residual times any weight + zero
    bias is still zero. The lm_head row for the sink id is also zero
    so the model can never *predict* the sink token.
    """

    state: Dict[str, torch.Tensor] = {}
    blocks = _blocks(model)

    def _to_dense(t: torch.Tensor) -> torch.Tensor:
        if t.is_sparse:
            return t.to_dense()
        return t

    def _append_zero_row(t: torch.Tensor) -> torch.Tensor:
        """Append a single zero row along dim=0."""

        zero_row = torch.zeros(
            (1, t.shape[1]), dtype=t.dtype, device=t.device
        )
        return torch.cat([t, zero_row], dim=0).contiguous()

    # 1. Token embedding. The R1 bake already pinned the NORM_COMPENSATOR
    #    column at K, so a straight copy preserves the invariant. R3 then
    #    appends one zero row as the sink-token embedding (k_proj/v_proj
    #    have no bias, so K=V=0 at that position).
    embed_weight = _to_dense(model.embed.embed.weight.data).clone().float()
    if sink_token_id is not None:
        expected_id = embed_weight.shape[0]
        if sink_token_id != expected_id:
            raise RuntimeError(
                f"sink_token_id={sink_token_id} must equal the original "
                f"vocab_size ({expected_id}); the sink row is appended "
                f"past the last VM token."
            )
        embed_weight = _append_zero_row(embed_weight)
    state["model.embed_tokens.weight"] = embed_weight

    # 2. Per-block weights.
    d_model = int(model.d_model)
    rms_identity_gamma = _rmsnorm_identity_gamma(d_model, K)
    # The Qwen3-dense config has a single ``intermediate_size``. The VM
    # actually has heterogeneous per-block hidden_dims (production model:
    # widths in {0, 1, 8, 42, 64, 192, 512, ..., 4096}). We pick the max
    # across PureFFN blocks and zero-pad / truncate every block to that
    # uniform size. Composite blocks (Blocker 1) synthesise zeros of the
    # same shape — see ``extract_composite_ffn_weights``.
    ffn_dims = _ffn_hidden_dims(model)
    intermediate_size = max(ffn_dims) if ffn_dims else 0
    # Qwen3 dense requires a uniform ``num_attention_heads`` across all
    # layers; the VM has heterogeneous per-block head counts. Pick the
    # max and zero-pad short blocks' Q/K/V rows + W_o columns up to
    # ``max_n_heads * head_dim``. See ``_attn_head_counts`` for the
    # algebraic argument that zero-padded heads contribute exactly zero.
    head_counts = _attn_head_counts(model)
    base_attn = _first_attention(model)
    base_n_heads = int(getattr(base_attn, "num_heads", 1)) if base_attn else 1
    max_n_heads = max(head_counts) if head_counts else base_n_heads
    head_dim = d_model // base_n_heads if base_n_heads else 0
    target_attn_rows = max_n_heads * head_dim

    for i, block in enumerate(blocks):
        prefix = f"model.layers.{i}"
        attn = block.attn
        ffn = block.ffn

        # Attention projections. Qwen3 stores them as Linear weights of
        # shape (out, in) — same convention as the VM. The VM's per-block
        # ``W_q`` etc. are ``(n_heads_block * head_dim, d_model)``, so
        # blocks whose ``n_heads_block < max_n_heads`` get zero-padded
        # along the row axis (and W_o gets zero-padded along the column
        # axis) to match Qwen3's uniform ``num_attention_heads`` config.
        W_q = _to_dense(attn.W_q.data).clone().float()
        W_k = _to_dense(attn.W_k.data).clone().float()
        W_v = _to_dense(attn.W_v.data).clone().float()
        W_o = _to_dense(attn.W_o.data).clone().float()
        W_k, W_v = _add_softmax_sink_to_kv(W_k, W_v)

        if target_attn_rows > 0:
            W_q = _pad_attn_proj_rows(W_q, target_rows=target_attn_rows)
            W_k = _pad_attn_proj_rows(W_k, target_rows=target_attn_rows)
            W_v = _pad_attn_proj_rows(W_v, target_rows=target_attn_rows)
            W_o = _pad_attn_o_cols(W_o, target_cols=target_attn_rows)

        # Bake the per-head q_norm/k_norm sentinel (Blocker 3). Each head
        # gets a constant ``K_h`` at its last head-dim slot so the
        # subsequent ``q_norm`` / ``k_norm`` RMSNorm collapses to identity
        # under a matching ``K_h/sqrt(head_dim)`` gamma. The bake applies
        # to every head up to ``max_n_heads`` (including phantom-padded
        # heads); padded heads still have zero W_o columns so their
        # residual contribution stays zero.
        if head_dim >= 2 and max_n_heads > 0:
            W_q, W_k = _bake_qk_norm_sentinel(
                W_q,
                W_k,
                n_heads=max_n_heads,
                head_dim=head_dim,
                norm_compensator_idx=norm_compensator_idx,
                K=K,
            )

        state[f"{prefix}.self_attn.q_proj.weight"] = W_q.contiguous()
        state[f"{prefix}.self_attn.k_proj.weight"] = W_k.contiguous()
        state[f"{prefix}.self_attn.v_proj.weight"] = W_v.contiguous()
        state[f"{prefix}.self_attn.o_proj.weight"] = W_o.contiguous()

        # Qwen3 dense has per-head q_norm/k_norm. With the sentinel bake
        # above, the per-head RMS is ``≈ K_h / sqrt(head_dim)``; pairing
        # that with ``gamma = K_h / sqrt(head_dim)`` makes RMSNorm an
        # identity. ``head_dim`` is uniform across the VM's attention
        # layers, so the gamma vector is the same regardless of the
        # per-block ``num_heads``.
        if head_dim > 0:
            head_gamma = _rmsnorm_identity_gamma(
                head_dim, _QK_NORM_SENTINEL_K_H
            )
            state[f"{prefix}.self_attn.q_norm.weight"] = head_gamma.clone()
            state[f"{prefix}.self_attn.k_norm.weight"] = head_gamma.clone()

        # SwiGLU repack + bias fold. For composite FFN blocks (Blocker 1)
        # synthesise a Qwen-shaped triple via ``extract_composite_ffn_weights``
        # so the repack helper sees a flat PureFFN-like interface. The
        # composite-derived weights are zero-init (skip-pass), which keeps
        # export structural and unblocks ``AutoModelForCausalLM.from_pretrained``.
        if _is_composite_ffn(ffn):
            W_up_c, W_gate_c, W_down_c = extract_composite_ffn_weights(
                block,
                d_model=d_model,
                intermediate_size=intermediate_size,
                dtype=torch.float32,
            )
            ffn_for_repack = _CompositeFFNAdapter(
                W_up=W_up_c, W_gate=W_gate_c, W_down=W_down_c
            )
        else:
            ffn_for_repack = ffn
        gate_proj, up_proj, down_proj = _swiglu_repack_and_fold_bias(
            ffn_for_repack,
            bias_compensator_idx=bias_compensator_idx,
        )
        # Pad PureFFN blocks whose hidden_dim < intermediate_size with
        # zeros so every layer's MLP has the same shape (Qwen3 requires a
        # uniform intermediate_size across the config). Padding is
        # mathematically a no-op: zero hidden units contribute zero to
        # the SwiGLU sum and zero to the down-projection output.
        gate_proj = _pad_or_truncate_swiglu_proj(
            gate_proj, target_hidden=intermediate_size, axis=0
        )
        up_proj = _pad_or_truncate_swiglu_proj(
            up_proj, target_hidden=intermediate_size, axis=0
        )
        down_proj = _pad_or_truncate_swiglu_proj(
            down_proj, target_hidden=intermediate_size, axis=1
        )
        state[f"{prefix}.mlp.gate_proj.weight"] = gate_proj.contiguous().float()
        state[f"{prefix}.mlp.up_proj.weight"] = up_proj.contiguous().float()
        state[f"{prefix}.mlp.down_proj.weight"] = down_proj.contiguous().float()

        # Per-block RMSNorm — identity via R2 gamma.
        state[f"{prefix}.input_layernorm.weight"] = rms_identity_gamma.clone()
        state[f"{prefix}.post_attention_layernorm.weight"] = (
            rms_identity_gamma.clone()
        )

    # 3. Final RMSNorm and LM head.
    state["model.norm.weight"] = rms_identity_gamma.clone()
    head_weight = _to_dense(model.head.weight.data).clone().float()
    if sink_token_id is not None:
        head_weight = _append_zero_row(head_weight)
    state["lm_head.weight"] = head_weight

    # NORM_COMPENSATOR sanity: the column we serialised on the embedding
    # should still be K on every real-token row. The appended sink row
    # is zero by construction; verify only the real-token rows here.
    real_rows = (
        embed_weight.shape[0] - 1
        if sink_token_id is not None
        else embed_weight.shape[0]
    )
    actual = state["model.embed_tokens.weight"][:real_rows, norm_compensator_idx]
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
        sink_token_id=cfg.c4_softmax_sink_token_id,
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


# ---------------------------------------------------------------------------
# Phase R8 / Blocker 4 — ALiBi → RoPE export-time transform
# ---------------------------------------------------------------------------
#
# The production VM compiles with ``positional_encoding="alibi"``. Qwen3 is
# RoPE-only. ALiBi and RoPE have fundamentally different functional forms:
#
#   * ALiBi adds an additive bias ``-slope_h * |i - j|`` to the QK^T scores
#     before softmax. The bias is per-head, per-distance, content-independent.
#   * RoPE rotates Q[i] and K[j] in their head_dim feature space by angles
#     that depend on absolute position. The resulting Q'[i] · K'[j] depends
#     on relative position via rotation but does NOT contribute an additive
#     constant term to the score.
#
# There is no weight transform on ``W_q`` / ``W_k`` that turns the additive
# ALiBi bias term ``-slope*|i-j|`` (a constant in the input embedding) into
# the multiplicative rotation that RoPE applies. The two formulations are
# only equivalent on **causal sequence position 0** (a sequence of length
# 1), where there is exactly one valid attention pair ``(0,0)``, ``|i-j|=0``
# makes the ALiBi term zero, and the RoPE angle at position 0 is also
# zero — so both reduce to ``scaled_dot_product`` of the un-rotated Q/K.
#
# ``alibi_to_rope_export`` is therefore a **best-effort** exporter:
#
#   * Walks every attention block and removes the ALiBi additive term by
#     zeroing ``alibi_slopes`` (so the additive bias is identically 0).
#   * Re-allocates a RoPE cos/sin cache and flips the per-attention
#     ``_positional_encoding`` flag to ``"rope"`` so subsequent forward
#     passes route through the RoPE branch.
#   * Re-bakes ``W_q`` / ``W_k`` only in the trivial sense (no change) —
#     there is no algebraic transform that compensates for the dropped
#     additive bias under RoPE rotation in the general case.
#
# Byte-identity guarantee: forward outputs match the original ALiBi VM
# **exactly on position 0** (sequence length 1, after the conversion).
# For longer sequences the helper documents the bound: the score
# perturbation at position ``i`` attending to ``j`` is
# ``slope_h * |i-j| + (cos(theta_i)cos(theta_j) + sin(theta_i)sin(theta_j) - 1)
#  · q · k`` — i.e. bounded by the largest ALiBi slope times the maximum
# relative position considered, plus the RoPE angular drift.
#
# The companion test ``tests/test_qwen_alibi_to_rope.py`` pins the
# position-0 byte-identity equivalence and asserts a monotonic divergence
# bound for positions 1..31 so future regressions land here.


def _build_alibi_bias(
    alibi_slopes: torch.Tensor,
    seq_len: int,
    causal: bool = True,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Return the additive ALiBi bias matrix at shape ``[H, S, S]``.

    Mirrors the ALiBi convention in
    :meth:`AutoregressiveAttention.forward`: ``bias[h, i, j] =
    -slope_h * |i - j|``. The causal mask is NOT applied here; it is the
    caller's responsibility (the test path bakes the unmasked bias for
    diagnostics and the runtime forward stacks the causal mask
    separately).

    Used as a reference oracle for the equivalence test and the
    diagnostic delta-bound the helper reports.
    """

    device = device if device is not None else alibi_slopes.device
    dtype = dtype if dtype is not None else alibi_slopes.dtype
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # [S, S]
    bias = -alibi_slopes.view(-1, 1, 1).to(dtype) * dist.unsqueeze(0)
    if causal:
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        bias = bias + causal_mask.unsqueeze(0)
    return bias


@dataclass(frozen=True)
class AlibiToRopeReport:
    """Diagnostic record returned by :func:`alibi_to_rope_export`.

    Pinning the converted block count + the (asserted) position range over
    which the transform is byte-identical lets downstream tests detect
    regressions in either direction (a future helper that *does* close the
    full equivalence gap would push ``byte_identity_max_pos`` past 0).
    """

    converted_blocks: int
    skipped_blocks: int
    skipped_reasons: Tuple[str, ...]
    byte_identity_max_pos: int  # inclusive upper bound: 0 in current impl
    max_alibi_slope: float
    rope_base: float


def _rebake_attention_for_rope(
    attn: Any,
    *,
    positional_encoding: str,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    rope_base: float,
    reference_distance: float = 0.0,
) -> Dict[str, Any]:
    """Pre-rotate ``W_q`` / ``W_k`` to partially compensate the ALiBi→RoPE
    semantic gap on address-bit-style content-addressed heads.

    Research-level partial fix. Per the module note above and
    BLOG_SPEC.md §410, the load-bearing semantics of attention scores in
    the production VM is the binary-address match on L15 head dim slots
    4..27 (24 bits, ±scale=10). Under ALiBi those Q·K contributions are
    position-independent (the bias term is additive and head-uniform).
    Under RoPE the same dims get a multiplicative rotation by
    ``θ_p = p * inv_freq``, and the score contribution at a pair
    ``(2k, 2k+1)`` becomes::

        Q'·K' = (Q[2k]K[2k] + Q[2k+1]K[2k+1]) * cos(θ_q − θ_k)
              + (Q[2k+1]K[2k] − Q[2k]K[2k+1]) * sin(θ_q − θ_k)

    There is **no general algebraic transform** that recovers
    ALiBi-equivalent scores under RoPE for arbitrary inputs (the
    additive bias and the rotational mixing live in different
    function-form classes — see also the module-level note above the
    :func:`alibi_to_rope_export` definition).

    What we CAN do, and what this helper does, is pre-rotate ``W_k`` by
    a per-head **reference offset** ``θ_ref = reference_distance *
    inv_freq``. The effect is to shift the *peak* of the post-RoPE
    Q·K(Δp) curve from ``Δp = 0`` to ``Δp = reference_distance``::

        Q'·K' peaks at  Δp = reference_distance   (cos argument is 0)

    For ``reference_distance == 0`` (the default) the rebake is a
    no-op and the byte-identity guarantee at causal position 0 is
    preserved. For positive ``reference_distance`` the helper biases
    the address-match peak to the expected lookback distance, trading
    position-0 exactness for typical-distance recovery on L15-style
    address-lookup heads.

    Args:
        attn: an attention module with ``W_q`` / ``W_k`` parameters,
            shape ``[dim, dim]`` viewed as ``[dim, H, HD]`` per head.
        positional_encoding: must be ``"rope"`` — the helper is a
            no-op for any other value (ALiBi blocks should be
            converted via :func:`alibi_to_rope_export` first, then
            optionally re-baked through this helper).
        num_heads: head count.
        head_dim: per-head feature dim. Must be even (RoPE constraint).
        max_seq_len: cache size; only used to bound
            ``reference_distance`` sanity.
        rope_base: RoPE base frequency (Qwen3 default 10000.0).
        reference_distance: typical lookback distance in tokens. ``0``
            preserves position-0 byte identity (no compensation).
            Positive values bias the score peak toward ``Δp =
            reference_distance``.

    Returns:
        A diagnostic dict with the per-head rotation angles applied and
        the dim pairs touched. Lets callers gate the rebake on
        head-specific knowledge (e.g. L15 heads 0-3 use the address-bit
        pattern; other heads do not).
    """

    if positional_encoding != "rope":
        return {
            "rebaked": False,
            "reason": f"positional_encoding={positional_encoding!r} != 'rope'",
        }
    if head_dim % 2 != 0:
        raise ValueError(
            f"_rebake_attention_for_rope: head_dim={head_dim} must be even"
        )
    if reference_distance == 0.0:
        # Default no-op: preserves position-0 byte identity.
        return {
            "rebaked": False,
            "reason": "reference_distance=0 (preserves position-0 identity)",
            "num_pairs": head_dim // 2,
        }
    if reference_distance < 0 or reference_distance > max_seq_len:
        raise ValueError(
            f"_rebake_attention_for_rope: reference_distance="
            f"{reference_distance} out of [0, {max_seq_len}]"
        )

    # Per-pair inverse frequencies (mirrors precompute_rope_cache).
    half_idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=attn.W_k.device)
    inv_freq = 1.0 / (rope_base ** (half_idx / head_dim))  # [HD/2]
    theta_ref = reference_distance * inv_freq  # [HD/2]
    # Build a per-head rotation block-diagonal: each (2k, 2k+1) pair is
    # rotated by ``θ_ref[k]``. The rotation matrix on a pair (x0, x1)
    # under the RoPE-matched convention (interleaved pairs) is::
    #
    #     [ cos  -sin ]
    #     [ sin   cos ]
    #
    # i.e. ``x0' = cos*x0 - sin*x1``, ``x1' = sin*x0 + cos*x1``.
    # ``W_k`` has shape ``[dim_in, dim_out]`` where ``dim_out`` = H * HD.
    # Reshape ``dim_out`` as ``[H, HD/2, 2]`` and apply the rotation to
    # the last two axes.
    cos = theta_ref.cos()  # [HD/2]
    sin = theta_ref.sin()  # [HD/2]
    with torch.no_grad():
        W_k = attn.W_k.data  # [dim_in, dim_out]
        dim_in, dim_out = W_k.shape
        if dim_out != num_heads * head_dim:
            return {
                "rebaked": False,
                "reason": (
                    f"unexpected W_k shape {tuple(W_k.shape)}; expected "
                    f"dim_out=num_heads*head_dim={num_heads*head_dim}"
                ),
            }
        Wk_view = W_k.view(dim_in, num_heads, head_dim // 2, 2)
        x0 = Wk_view[..., 0]  # [dim_in, H, HD/2]
        x1 = Wk_view[..., 1]
        cos_b = cos.view(1, 1, -1)  # broadcast across (dim_in, H)
        sin_b = sin.view(1, 1, -1)
        x0_new = cos_b * x0 - sin_b * x1
        x1_new = sin_b * x0 + cos_b * x1
        new_Wk = torch.stack((x0_new, x1_new), dim=-1).reshape_as(W_k)
        attn.W_k.data = new_Wk.contiguous()

    return {
        "rebaked": True,
        "reference_distance": float(reference_distance),
        "num_pairs": head_dim // 2,
        "max_rotation": float(theta_ref.max().item()),
        "min_rotation": float(theta_ref.min().item()),
    }


def alibi_to_rope_export(
    model: Any,
    *,
    rope_base: Optional[float] = None,
    rebake_reference_distance: float = 0.0,
) -> AlibiToRopeReport:
    """Convert every ALiBi attention block in ``model`` to RoPE in-place.

    Walks every ``block.attn`` (and any ``post_op`` whose attention is
    ALiBi-baked) and performs the export-time transform:

      1. Verifies the block is on the ALiBi branch
         (``_positional_encoding == "alibi"``). RoPE blocks pass through
         unchanged; hybrid layers are converted only on the ALiBi sub-range
         (the hybrid contract pins ``layer_idx < 3`` to ALiBi — those flip
         to ``"rope"``; layer >= 3 are already RoPE).
      2. Zeroes the ``alibi_slopes`` buffer so the additive bias term
         ``-slope*|i-j|`` collapses to zero in
         :meth:`AutoregressiveAttention.forward`.
      3. Allocates ``_rope_cos`` / ``_rope_sin`` buffers using
         :func:`precompute_rope_cache` so the RoPE branch (line ~458 of
         ``vm_step.py``) becomes live.
      4. Flips ``_positional_encoding`` to ``"rope"`` so any code that
         keys off the string sees the new state.

    The ``W_q`` / ``W_k`` weights are NOT re-baked. As documented above,
    there is no algebraic transform that turns the ALiBi additive
    constant into a RoPE rotation for arbitrary inputs. The helper is
    therefore "best-effort" and the equivalence is *exact* only for
    sequence length 1 (causal position 0, where ``|i-j|=0`` and the RoPE
    angle at position 0 is also 0).

    Args:
        model: a baked VM (typically an :class:`AutoregressiveVM`).
        rope_base: optional override for the RoPE base frequency. When
            ``None``, falls back to ``model.rope_base`` (default 10000.0).

    Returns:
        An :class:`AlibiToRopeReport` summarising the conversion.

    Raises:
        ValueError: if any attention block has an odd ``head_dim`` (RoPE
            requires an even feature dimension).
    """

    # Local imports keep qwen_compat importable without the heavyweight
    # vm_step module load at planning time.
    from .vm_step import precompute_rope_cache as _precompute_rope_cache

    if rope_base is None:
        rope_base = float(getattr(model, "rope_base", 10000.0))
    else:
        rope_base = float(rope_base)

    converted = 0
    skipped = 0
    reasons: List[str] = []
    max_slope = 0.0

    def _convert_attn(attn: Any, layer_idx: int) -> None:
        nonlocal converted, skipped, max_slope
        if attn is None or not hasattr(attn, "W_q"):
            skipped += 1
            reasons.append(f"layer {layer_idx}: attn has no W_q (skipped)")
            return
        pe = getattr(attn, "_positional_encoding", None)
        # Already on the RoPE branch — nothing to do.
        if pe == "rope":
            skipped += 1
            reasons.append(f"layer {layer_idx}: already RoPE (no-op)")
            return
        if pe not in {"alibi", "hybrid"}:
            skipped += 1
            reasons.append(
                f"layer {layer_idx}: unknown positional_encoding={pe!r} (skipped)"
            )
            return
        head_dim = int(getattr(attn, "head_dim"))
        if head_dim % 2 != 0:
            raise ValueError(
                f"alibi_to_rope_export: layer {layer_idx} has odd head_dim="
                f"{head_dim}; RoPE requires an even head_dim."
            )

        # Capture the slopes for the diagnostic bound BEFORE zeroing.
        slopes = getattr(attn, "alibi_slopes", None)
        if slopes is not None:
            max_slope = max(max_slope, float(slopes.abs().max().item()))
            with torch.no_grad():
                slopes.zero_()

        # Allocate / overwrite the RoPE cache.
        max_seq_len = int(getattr(attn, "max_seq_len", 1024))
        device = attn.W_q.device
        cos, sin = _precompute_rope_cache(
            head_dim, max_seq_len, base=rope_base, device=device
        )
        # The ALiBi branch assigns ``_rope_cos = None`` as a plain
        # attribute (NOT a registered buffer). ``register_buffer`` raises
        # ``KeyError`` when the name already exists, even when it's just
        # a ``None`` attribute. Clear both possible homes (instance dict
        # for the ``None`` case, ``_buffers`` for a previously registered
        # buffer) before re-registering.
        attn.__dict__.pop("_rope_cos", None)
        attn.__dict__.pop("_rope_sin", None)
        attn._buffers.pop("_rope_cos", None)
        attn._buffers.pop("_rope_sin", None)
        attn.register_buffer("_rope_cos", cos)
        attn.register_buffer("_rope_sin", sin)

        # Flip the positional-encoding flag last so a partially-converted
        # attn (e.g. an exception in cos/sin alloc) leaves the original
        # branch active.
        attn._positional_encoding = "rope"
        attn.rope_base = rope_base

        # Optional W_q/W_k rebake: research-level partial compensation.
        # Default ``rebake_reference_distance=0.0`` is a no-op and
        # preserves the position-0 byte-identity guarantee. Positive
        # values shift the post-RoPE Q·K peak toward the typical
        # lookback distance — only meaningful on L15-style content-
        # addressed heads. See ``_rebake_attention_for_rope``.
        if rebake_reference_distance != 0.0:
            _rebake_attention_for_rope(
                attn,
                positional_encoding="rope",
                num_heads=int(getattr(attn, "num_heads")),
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                reference_distance=float(rebake_reference_distance),
            )
        converted += 1

    for layer_idx, block in enumerate(_blocks(model)):
        _convert_attn(getattr(block, "attn", None), layer_idx)
        for post_op in getattr(block, "post_ops", None) or []:
            # Some post_ops carry their own attention (skip-pass blocks).
            inner_attn = getattr(post_op, "attn", None)
            if inner_attn is not None:
                _convert_attn(inner_attn, layer_idx)

    # Propagate the flag at the top-level config so subsequent introspection
    # (and ``analyze_qwen_compatibility``) reports the new state.
    if hasattr(model, "positional_encoding"):
        model.positional_encoding = "rope"

    return AlibiToRopeReport(
        converted_blocks=converted,
        skipped_blocks=skipped,
        skipped_reasons=tuple(reasons),
        # Byte-identity holds for the single-token case only — see the
        # module-level math note above.
        byte_identity_max_pos=0,
        max_alibi_slope=float(max_slope),
        rope_base=rope_base,
    )


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
    "prepend_softmax_sink",
    "extract_composite_ffn_weights",
    "flatten_post_ops_for_qwen_export",
    "count_post_ops",
    "expanded_qwen_layer_count",
    "summarize_post_ops_flattening",
    "PostOpsFlatteningReport",
    "alibi_to_rope_export",
    "AlibiToRopeReport",
    "_rebake_attention_for_rope",
    "apply_neural_vm_embedding_augmentations",
    "NeuralVMEmbeddingWrapper",
    "install_neural_vm_embedding_wrapper",
]


# ---------------------------------------------------------------------------
# Phase R5 — post_ops flattening for Qwen export
# ---------------------------------------------------------------------------
#
# Qwen2/Qwen3 decoder layers are a strict ``attn -> ffn`` pair plus residuals
# (with norms). They have no concept of ``post_ops``. Our
# :class:`neural_vm.vm_step.TransformerBlock` runs an arbitrary
# ``post_ops: nn.ModuleList`` after ``attn`` + ``ffn``, where each post-op is a
# residual FFN-shaped transformation (most are :class:`PureFFN` subclasses;
# some are composite ALU modules).
#
# The runtime VM expands these via ``_expand_wrapper_blocks`` (see
# ``vm_step.py``) into ``1 + N`` blocks with a zero-init "skip-pass" attention
# (W_q/W_k/W_v/W_o all zero so the post-attention residual stream is
# ``x + 0 = x``). The same structural transform is what Qwen export needs.
#
# This module provides a non-mutating equivalent suitable for the export plan:
# given a VM model, return a list of fresh :class:`TransformerBlock`
# instances such that running them in sequence reproduces the original model's
# block-by-block forward exactly. The base block reuses the original ``attn``
# and ``ffn`` modules (its own ``post_ops`` is empty); each skip-pass block
# wraps a single post-op as its FFN and a zero-init attention so the only
# delta on the residual stream comes from the post-op.


def count_post_ops(model: Any) -> List[int]:
    """Return ``len(block.post_ops)`` for every block in order.

    Blocks without a ``post_ops`` attribute count as 0. This is the data the
    Qwen export uses to compute the expanded layer count.
    """

    counts: List[int] = []
    for block in _blocks(model):
        post_ops = getattr(block, "post_ops", None)
        counts.append(int(len(post_ops)) if post_ops is not None else 0)
    return counts


def expanded_qwen_layer_count(model: Any) -> int:
    """Effective Qwen ``num_hidden_layers`` after post_ops flattening.

    Equals ``sum(1 + len(block.post_ops) for block in model.blocks)``: each
    base block contributes 1 Qwen layer, plus one skip-pass layer per post-op.
    """

    return sum(1 + n for n in count_post_ops(model))


def _make_skip_pass_attention(template_attn: Any) -> Any:
    """Build a zero-init attention whose forward delta is 0.

    Mirrors ``_expand_wrapper_blocks._make_passthrough_block`` in
    ``vm_step.py``: a fresh :class:`AutoregressiveAttention` with the same
    shape, positional encoding, and normalization as ``template_attn``. All
    weights default to zero in the constructor, so the attention output is the
    zero tensor and the residual ``x + attn(x) = x`` is preserved.

    The constructor zero-fills ``W_q``/``W_k``/``W_v``/``W_o`` — no further
    initialisation is required to guarantee the skip-pass invariant.
    """

    # Local import keeps qwen_compat importable without forcing the heavyweight
    # vm_step module load at planning time.
    from neural_vm.vm_step import AutoregressiveAttention

    attn = AutoregressiveAttention(
        dim=int(template_attn.dim),
        num_heads=int(template_attn.num_heads),
        max_seq_len=int(template_attn.max_seq_len),
        layer_idx=getattr(template_attn, "layer_idx", None),
        use_flash_attention=getattr(template_attn, "use_flash_attention", True),
        positional_encoding=getattr(template_attn, "_positional_encoding", None),
        attention_normalization=getattr(
            template_attn, "attention_normalization", None
        ),
        rope_base=getattr(template_attn, "rope_base", None),
    )
    # Defensive: confirm all four projections are zero so attn(x) ≡ 0.
    # The constructor already zeros them; this guards against future drift.
    with torch.no_grad():
        attn.W_q.zero_()
        attn.W_k.zero_()
        attn.W_v.zero_()
        attn.W_o.zero_()
    return attn


def _make_skip_pass_block(template_block: Any, post_op: nn.Module) -> Any:
    """Build a TransformerBlock with zero-attn + ``post_op`` as the FFN.

    The skip-pass block deliberately runs without RMSNorm even when the
    parent ``template_block`` uses RMSNorm. Two reasons:

      * The parent block runs its ``post_ops`` AFTER the rms-norm
        attention/ffn pair on the bare residual stream (see
        :class:`TransformerBlock.forward`). The post-op already receives the
        un-normalized stream and is expected to compute ``x + delta`` from
        that input. Wrapping it in a rms-norm sub-block would re-normalize
        the input and change semantics.
      * With rms-norm enabled the block forward becomes
        ``x = x + (attn_out - attn_in)``. For attn_out=0 (our skip-pass),
        that reduces to ``x - rmsnorm(x)`` — NOT a pass-through. Disabling
        rms-norm gives the clean ``x + attn(x) = x + 0 = x`` we need.

    Without rms-norm the block forward is ``ffn(attn(x))`` and the post-op
    (a residual ``x + delta`` FFN) reproduces the parent's post-op exactly.
    """

    from neural_vm.vm_step import TransformerBlock

    attn = _make_skip_pass_attention(template_block.attn)
    return TransformerBlock(
        attn=attn,
        ffn=post_op,
        use_rms_norm=False,
    )


def _make_base_block_without_post_ops(template_block: Any) -> Any:
    """Wrap ``template_block``'s attn + ffn in a fresh empty-post_ops block.

    The returned block shares the underlying ``attn`` and ``ffn`` parameter
    tensors with ``template_block`` (no clone), so forward output is
    bit-identical to ``template_block`` with ``post_ops`` removed. This
    matches what Qwen2 export consumes: one ``attn -> ffn`` pair per layer.
    """

    from neural_vm.vm_step import TransformerBlock

    new_block = TransformerBlock(
        attn=template_block.attn,
        ffn=template_block.ffn,
        use_rms_norm=bool(getattr(template_block, "use_rms_norm", False)),
        rms_norm_eps=float(
            getattr(getattr(template_block, "attn_norm", None), "eps", 1e-6)
            if getattr(template_block, "use_rms_norm", False)
            else 1e-6
        ),
    )
    if getattr(template_block, "use_rms_norm", False):
        # Reuse the same RMSNorm parameter tensors so weights are shared, not
        # copied. The export plan's "copy" mapping then reads the same source.
        new_block.attn_norm = template_block.attn_norm
        new_block.ffn_norm = template_block.ffn_norm
    return new_block


def flatten_post_ops_for_qwen_export(model: Any) -> List[Any]:
    """Return a flattened list of TransformerBlock instances for Qwen export.

    For each block in ``model.blocks`` with N post_ops, the returned list
    contains 1 + N entries:

      * One base block exposing the original ``attn`` + ``ffn`` with no
        post_ops. Parameters are shared with the source block (no clone), so
        the existing Qwen2 mapping plan keys (``blocks.<i>.attn.W_q`` etc.)
        continue to copy the same tensors.
      * N skip-pass blocks. Each has a freshly allocated zero-init
        :class:`AutoregressiveAttention` whose forward delta is the zero
        tensor (residual gives ``x + 0 = x``), and the corresponding post-op
        as its ``ffn``. Most post-ops are :class:`PureFFN` subclasses whose
        ``forward`` is ``x + delta``; the skip-pass block's overall forward
        is therefore ``post_op(x)`` after the attention pass-through.

    The length of the returned list equals
    :func:`expanded_qwen_layer_count`. The function does not mutate
    ``model``: the original blocks (and their ``post_ops`` ModuleLists) are
    left untouched.
    """

    flattened: List[Any] = []
    for block in _blocks(model):
        flattened.append(_make_base_block_without_post_ops(block))
        post_ops = getattr(block, "post_ops", None) or []
        for post_op in post_ops:
            flattened.append(_make_skip_pass_block(block, post_op))
    return flattened


@dataclass(frozen=True)
class PostOpsFlatteningReport:
    """Diagnostic summary of the R5 flattening pass."""

    original_block_count: int
    post_op_counts: Tuple[int, ...]
    expanded_block_count: int

    @property
    def total_post_ops(self) -> int:
        return sum(self.post_op_counts)


def summarize_post_ops_flattening(model: Any) -> PostOpsFlatteningReport:
    """Return a frozen diagnostic record without instantiating new blocks."""

    counts = count_post_ops(model)
    return PostOpsFlatteningReport(
        original_block_count=len(counts),
        post_op_counts=tuple(counts),
        expanded_block_count=sum(1 + n for n in counts),
    )
