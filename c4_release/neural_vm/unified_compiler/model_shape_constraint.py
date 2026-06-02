"""Declarative shape constraint for external-model bake targets.

This module provides ``ModelShapeConstraint``, an IR object that lets a
caller specify a target architecture's shape envelope (``d_model``,
``num_hidden_layers``, ``num_attention_heads``, ``num_key_value_heads``,
``head_dim``, ``intermediate_size``, ``vocab_size``, plus per-layer
overrides) and a ``validate_against_shape`` helper that diffs the
constraint against a compiled VM model.

Why a separate IR object (and not just kwargs to
``compile_full_vm_dynamic``)?

The dynamic-heads + GQA + per-head ``head_dim`` work that landed earlier
this session (commits ``754bf153`` and ``2c636e8c``, plus the allocator
+ primitives changes in ``attention_head_allocator.py`` /
``primitives.py``) made the per-layer Q/KV head count and head-dim
queryable on a compiled model. External models — Mixtral, Llama,
GPT-NeoX, custom — are typically described by an architecture config
(``config.json``) whose fields match the constraint fields below
verbatim. Lifting those fields into an IR dataclass means a caller can:

* Hand a single ``ModelShapeConstraint`` to ``compile_full_vm_dynamic``
  and have the compiler refuse to emit a model that doesn't match.
* Round-trip the constraint through serialization (a plain
  ``dataclasses.asdict``).
* Mix per-layer overrides into the constraint without growing the
  ``compile_full_vm_dynamic`` kwarg surface.

The constraint is purely descriptive — it does NOT drive op selection
or layer counts. It runs after the compile, compares against the
compiled model's actual shape attributes (``model.d_model``,
``model.blocks[i].attn.num_heads``, etc.), and raises on mismatch. That
gives callers a safety net without entangling the constraint with the
allocator / scheduler.

The C4 VM target's natural shape (``d_model=512``, 18 layers,
``num_heads=8``, MHA so ``num_key_value_heads=num_heads``,
``head_dim=64``) is reachable with ``target="custom"`` and matching
fields. The named ``"mixtral"`` / ``"llama"`` targets are convenience
labels — they don't currently change semantics, they just record the
intent of the caller in the constraint object itself so a downstream
audit tool can group reports by target family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_VALID_TARGETS = frozenset({"mixtral", "llama", "gpt_neox", "custom"})


@dataclass
class ModelShapeConstraint:
    """Declarative shape envelope for a compiled VM model.

    All scalar fields are optional; ``None`` means "don't check this
    dimension". Per-layer overrides live in ``per_layer_overrides``:
    a mapping from ``layer_idx`` to a dict of any of the same scalar
    field names (``num_attention_heads``, ``num_key_value_heads``,
    ``head_dim``, ``intermediate_size``). When a per-layer override is
    set, it takes precedence over the top-level constraint for that
    layer only.

    ``target`` is a free-form label naming the architecture family the
    caller intends to match. The validator does not currently switch
    semantics on ``target`` — it's recorded in mismatch messages so an
    operator can see which family was intended. Unknown values raise.
    """

    target: str
    d_model: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    vocab_size: Optional[int] = None
    per_layer_overrides: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target not in _VALID_TARGETS:
            raise ValueError(
                f"ModelShapeConstraint.target={self.target!r} is not one of "
                f"{sorted(_VALID_TARGETS)}"
            )
        # Validate per_layer_overrides shape early so callers see a clean
        # error at construction time, not deep inside ``validate_against_shape``.
        for layer_idx, overrides in self.per_layer_overrides.items():
            if not isinstance(layer_idx, int):
                raise TypeError(
                    f"per_layer_overrides keys must be int (got {type(layer_idx).__name__})"
                )
            if not isinstance(overrides, dict):
                raise TypeError(
                    f"per_layer_overrides[{layer_idx}] must be a dict "
                    f"(got {type(overrides).__name__})"
                )
            for key in overrides:
                if key not in _PER_LAYER_FIELDS:
                    raise ValueError(
                        f"per_layer_overrides[{layer_idx}] has unknown field "
                        f"{key!r}; expected one of {sorted(_PER_LAYER_FIELDS)}"
                    )


# Fields that may appear inside a ``per_layer_overrides`` entry. Only the
# attention/FFN-shape fields are per-layer; ``d_model`` / ``vocab_size``
# are global and cannot vary per-layer in this VM.
_PER_LAYER_FIELDS = frozenset({
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
})


def _get_attn(block: Any) -> Any:
    """Return the attention sub-module of a transformer block, or ``None``."""
    return getattr(block, "attn", None)


def _get_ffn(block: Any) -> Any:
    return getattr(block, "ffn", None)


def _layer_num_q_heads(block: Any) -> Optional[int]:
    attn = _get_attn(block)
    if attn is None:
        return None
    n = getattr(attn, "num_heads", None)
    return int(n) if n is not None else None


def _layer_num_kv_heads(block: Any) -> Optional[int]:
    """Best-effort read of per-layer KV head count.

    Phase 8.O.2 introduced GQA in the declarative IR via ``group_size`` on
    ``DeclarativeAttentionHeadSpec`` (see ``primitives.py``). The runtime
    attention module (``AutoregressiveAttention``) carries
    ``num_kv_heads`` when GQA is wired through, otherwise we fall back to
    ``num_heads`` (MHA).
    """
    attn = _get_attn(block)
    if attn is None:
        return None
    for attr in ("num_kv_heads", "num_key_value_heads"):
        v = getattr(attn, attr, None)
        if v is not None:
            return int(v)
    return _layer_num_q_heads(block)


def _layer_head_dim(block: Any) -> Optional[int]:
    attn = _get_attn(block)
    if attn is None:
        return None
    hd = getattr(attn, "head_dim", None)
    return int(hd) if hd is not None else None


def _layer_intermediate_size(block: Any) -> Optional[int]:
    ffn = _get_ffn(block)
    if ffn is None:
        return None
    # PureFFN exposes ``W_up`` whose first dim is the hidden width. Some
    # variants set ``hidden_dim`` directly.
    hd = getattr(ffn, "hidden_dim", None)
    if hd is not None:
        return int(hd)
    w_up = getattr(ffn, "W_up", None)
    if w_up is not None:
        try:
            return int(w_up.shape[0])
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _resolved_per_layer(
    constraint: ModelShapeConstraint, layer_idx: int, key: str
) -> Optional[int]:
    """Per-layer override wins; otherwise fall back to top-level field."""
    override = constraint.per_layer_overrides.get(layer_idx, {}).get(key)
    if override is not None:
        return int(override)
    top = getattr(constraint, key, None)
    return int(top) if top is not None else None


def validate_against_shape(
    model: Any, constraint: ModelShapeConstraint
) -> List[str]:
    """Return a list of shape mismatch strings.

    An empty list means the model matches the constraint. Each entry is
    a short human-readable description (``"d_model expected 4096, got 512"``)
    that names the offending field. The caller decides how to surface
    the mismatches; ``compile_full_vm_dynamic`` raises ``ValueError`` if
    the list is non-empty.

    The check is deliberately lenient: ``None`` fields are skipped, and
    fields the model does not expose are also skipped (they cannot
    mismatch a constraint we cannot read). This keeps the constraint
    usable as a partial spec — Mixtral-only callers can pin
    ``num_key_value_heads`` without having to also pin ``vocab_size``.
    """

    mismatches: List[str] = []

    # Top-level scalars.
    if constraint.d_model is not None:
        actual = getattr(model, "d_model", None)
        if actual is not None and int(actual) != constraint.d_model:
            mismatches.append(
                f"d_model expected {constraint.d_model}, got {int(actual)}"
            )

    if constraint.vocab_size is not None:
        actual = getattr(model, "vocab_size", None)
        if actual is not None and int(actual) != constraint.vocab_size:
            mismatches.append(
                f"vocab_size expected {constraint.vocab_size}, got {int(actual)}"
            )

    blocks = list(getattr(model, "blocks", []) or [])
    n_layers = len(blocks)
    if constraint.num_hidden_layers is not None:
        if n_layers != constraint.num_hidden_layers:
            mismatches.append(
                f"num_hidden_layers expected {constraint.num_hidden_layers}, "
                f"got {n_layers}"
            )

    # Per-layer fields (top-level applies uniformly unless overridden).
    for i, block in enumerate(blocks):
        expected_q = _resolved_per_layer(constraint, i, "num_attention_heads")
        if expected_q is not None:
            actual_q = _layer_num_q_heads(block)
            if actual_q is not None and actual_q != expected_q:
                mismatches.append(
                    f"layer {i} num_attention_heads expected {expected_q}, "
                    f"got {actual_q}"
                )

        expected_kv = _resolved_per_layer(constraint, i, "num_key_value_heads")
        if expected_kv is not None:
            actual_kv = _layer_num_kv_heads(block)
            if actual_kv is not None and actual_kv != expected_kv:
                mismatches.append(
                    f"layer {i} num_key_value_heads expected {expected_kv}, "
                    f"got {actual_kv}"
                )

        expected_hd = _resolved_per_layer(constraint, i, "head_dim")
        if expected_hd is not None:
            actual_hd = _layer_head_dim(block)
            if actual_hd is not None and actual_hd != expected_hd:
                mismatches.append(
                    f"layer {i} head_dim expected {expected_hd}, got {actual_hd}"
                )

        expected_is = _resolved_per_layer(constraint, i, "intermediate_size")
        if expected_is is not None:
            actual_is = _layer_intermediate_size(block)
            if actual_is is not None and actual_is != expected_is:
                mismatches.append(
                    f"layer {i} intermediate_size expected {expected_is}, "
                    f"got {actual_is}"
                )

    # Per-layer overrides referencing out-of-range layers are a constraint
    # bug, surface them explicitly.
    for layer_idx in constraint.per_layer_overrides:
        if layer_idx < 0 or layer_idx >= n_layers:
            mismatches.append(
                f"per_layer_overrides references layer {layer_idx}, but model "
                f"has {n_layers} layers"
            )

    return mismatches


class ModelShapeMismatchError(ValueError):
    """Raised when ``validate_against_shape`` reports mismatches.

    Carries the original mismatch list as ``.mismatches`` so callers can
    re-format it without re-running the validator.
    """

    def __init__(self, mismatches: List[str], target: str):
        self.mismatches = list(mismatches)
        self.target = target
        body = "; ".join(mismatches)
        super().__init__(
            f"ModelShapeConstraint(target={target!r}) mismatches: {body}"
        )
