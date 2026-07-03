"""Tests for ``ModelShapeConstraint`` and ``validate_against_shape``.

Covers:

1. A Mixtral-shape constraint passes against a synthetic Mixtral-shaped
   model (32 Q heads, 8 KV heads, ``head_dim=128``, ``d_model=4096``,
   ``intermediate_size=14336``, 32 layers).
2. A mismatched constraint raises ``ModelShapeMismatchError`` when wired
   through ``compile_full_vm_dynamic``.
3. ``target="custom"`` with explicit fields validates correctly.
4. ``per_layer_overrides`` are honored, including catching per-layer
   mismatches and out-of-range layer indices.

The constraint validator only reads attributes off the model
(``d_model``, ``vocab_size``, ``blocks[i].attn.num_heads``,
``blocks[i].attn.num_kv_heads``, ``blocks[i].attn.head_dim``,
``blocks[i].ffn.hidden_dim`` / ``W_up.shape[0]``), so most tests use
lightweight synthetic stand-ins. One end-to-end test runs
``compile_full_vm_dynamic`` with a deliberately wrong constraint to
prove the integration raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from c4_release.neural_vm.verification.model_shape_constraint import (
    ModelShapeConstraint,
    ModelShapeMismatchError,
    validate_against_shape,
)


# ---------------------------------------------------------------------------
# Synthetic model stand-ins
# ---------------------------------------------------------------------------


@dataclass
class _FakeAttn:
    num_heads: int
    head_dim: int
    num_kv_heads: Optional[int] = None


@dataclass
class _FakeFFN:
    hidden_dim: int


@dataclass
class _FakeBlock:
    attn: _FakeAttn
    ffn: _FakeFFN


@dataclass
class _FakeModel:
    d_model: int
    vocab_size: int
    blocks: List[_FakeBlock] = field(default_factory=list)


def _make_uniform_model(
    *,
    d_model: int,
    vocab_size: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> _FakeModel:
    blocks = [
        _FakeBlock(
            attn=_FakeAttn(num_heads=n_heads, head_dim=head_dim, num_kv_heads=n_kv_heads),
            ffn=_FakeFFN(hidden_dim=intermediate_size),
        )
        for _ in range(n_layers)
    ]
    return _FakeModel(d_model=d_model, vocab_size=vocab_size, blocks=blocks)


# ---------------------------------------------------------------------------
# (1) Mixtral-shape constraint passes against a Mixtral-shaped model
# ---------------------------------------------------------------------------


def test_mixtral_constraint_passes_on_synthetic_mixtral_model():
    """Mixtral-8x7B shape: d_model=4096, 32 layers, 32 Q / 8 KV heads,
    head_dim=128, intermediate_size=14336, vocab_size=32000.
    """
    model = _make_uniform_model(
        d_model=4096,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
    )
    constraint = ModelShapeConstraint(
        target="mixtral",
        d_model=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=32000,
    )
    mismatches = validate_against_shape(model, constraint)
    assert mismatches == [], f"Unexpected mismatches: {mismatches}"


def test_mixtral_constraint_catches_wrong_kv_head_count():
    """Mixtral expects 8 KV heads; a model with 32 (full MHA) must fail."""
    model = _make_uniform_model(
        d_model=4096,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,  # wrong: MHA, not GQA
        head_dim=128,
        intermediate_size=14336,
    )
    constraint = ModelShapeConstraint(
        target="mixtral",
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    mismatches = validate_against_shape(model, constraint)
    assert mismatches, "Expected num_key_value_heads mismatch"
    assert any("num_key_value_heads" in m for m in mismatches)


# ---------------------------------------------------------------------------
# (2) Mismatch raises through compile_full_vm_dynamic
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compile_full_vm_dynamic_raises_on_mismatch():
    """A deliberately wrong constraint (claims d_model=4096 when the VM
    bakes d_model<<4096) must raise ``ModelShapeMismatchError`` AFTER the
    bake completes, so the caller knows the bake itself was structurally
    fine — the mismatch is purely against the declared envelope.
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    bad_constraint = ModelShapeConstraint(
        target="custom",
        d_model=4096,  # VM is much smaller
    )
    with pytest.raises(ModelShapeMismatchError) as ei:
        # strict=False to bypass the unrelated phase_required_but_undeclared
        # admission check (``putchar_think_protocol`` baseline issue,
        # tracked in the dynamic scheduler migration plan) — the test
        # cares about the constraint mismatch, not the scheduler.
        compile_full_vm_dynamic(
            disk_cache=False,
            strict=False,
            model_shape_constraint=bad_constraint,
        )
    assert "d_model" in str(ei.value)
    assert ei.value.target == "custom"
    assert ei.value.mismatches, "Expected non-empty mismatch list"


@pytest.mark.slow
def test_compile_full_vm_dynamic_accepts_matching_constraint():
    """Construct a constraint that matches the actual baked shape.

    We first compile the VM without any constraint to discover the
    actual shape, then re-compile with a constraint built from the live
    shape — that must succeed (no mismatches → no raise).

    Note: per-layer heads are heterogeneous on the C4 VM (the dynamic
    head allocator emits different counts per layer), so we pin global
    fields (``d_model``, ``num_hidden_layers``, ``vocab_size``) and use
    ``per_layer_overrides`` for the per-layer shape of layer 0 — that
    exercises BOTH top-level AND override paths together.
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    model, _ = compile_full_vm_dynamic(disk_cache=False, strict=False)
    n_layers = len(model.blocks)
    d_model = int(model.d_model)
    vocab_size = int(getattr(model, "vocab_size"))
    attn0 = model.blocks[0].attn
    n_heads_0 = int(getattr(attn0, "num_heads"))
    head_dim_0 = int(getattr(attn0, "head_dim"))

    constraint = ModelShapeConstraint(
        target="custom",
        d_model=d_model,
        num_hidden_layers=n_layers,
        vocab_size=vocab_size,
        # No top-level head count — the VM is heterogeneous across
        # layers, so we pin layer 0 explicitly via per_layer_overrides.
        per_layer_overrides={
            0: {"num_attention_heads": n_heads_0, "head_dim": head_dim_0},
        },
    )
    # Should not raise.
    compile_full_vm_dynamic(
        disk_cache=False,
        strict=False,
        model_shape_constraint=constraint,
    )


# ---------------------------------------------------------------------------
# (3) target="custom" with explicit fields
# ---------------------------------------------------------------------------


def test_custom_target_validates_partial_constraint():
    """A ``target="custom"`` constraint pinning only a few fields must
    skip the unspecified ones (None means "don't check").
    """
    model = _make_uniform_model(
        d_model=512,
        vocab_size=256,
        n_layers=18,
        n_heads=8,
        n_kv_heads=8,
        head_dim=64,
        intermediate_size=2048,
    )
    constraint = ModelShapeConstraint(
        target="custom",
        d_model=512,
        num_attention_heads=8,
        # Intentionally not pinning num_hidden_layers / intermediate_size /
        # vocab_size — they should be ignored.
    )
    assert validate_against_shape(model, constraint) == []


def test_custom_target_full_field_set_matches():
    """A custom-target constraint with all fields set must match exactly."""
    model = _make_uniform_model(
        d_model=512,
        vocab_size=256,
        n_layers=18,
        n_heads=8,
        n_kv_heads=8,
        head_dim=64,
        intermediate_size=2048,
    )
    constraint = ModelShapeConstraint(
        target="custom",
        d_model=512,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2048,
        vocab_size=256,
    )
    assert validate_against_shape(model, constraint) == []


def test_unknown_target_raises_at_construction():
    """``target`` is restricted to the known family labels."""
    with pytest.raises(ValueError, match="target="):
        ModelShapeConstraint(target="gpt-j")


# ---------------------------------------------------------------------------
# (4) per_layer_overrides
# ---------------------------------------------------------------------------


def test_per_layer_overrides_match_when_declared():
    """A heterogeneous model (layer 0 has 16 heads, others have 8) must
    pass a constraint that uses ``per_layer_overrides`` to capture the
    deviation.
    """
    blocks = [
        _FakeBlock(
            attn=_FakeAttn(num_heads=16, head_dim=64, num_kv_heads=4),
            ffn=_FakeFFN(hidden_dim=2048),
        )
    ]
    blocks.extend(
        _FakeBlock(
            attn=_FakeAttn(num_heads=8, head_dim=64, num_kv_heads=8),
            ffn=_FakeFFN(hidden_dim=2048),
        )
        for _ in range(3)
    )
    model = _FakeModel(d_model=512, vocab_size=256, blocks=blocks)

    constraint = ModelShapeConstraint(
        target="custom",
        d_model=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2048,
        per_layer_overrides={
            0: {"num_attention_heads": 16, "num_key_value_heads": 4},
        },
    )
    mismatches = validate_against_shape(model, constraint)
    assert mismatches == [], f"Unexpected mismatches: {mismatches}"


def test_per_layer_overrides_catch_mismatch():
    """If the model's layer 0 doesn't match the override, the validator
    must surface a mismatch naming layer 0 specifically.
    """
    blocks = [
        _FakeBlock(
            attn=_FakeAttn(num_heads=8, head_dim=64, num_kv_heads=8),
            ffn=_FakeFFN(hidden_dim=2048),
        )
        for _ in range(4)
    ]
    model = _FakeModel(d_model=512, vocab_size=256, blocks=blocks)

    constraint = ModelShapeConstraint(
        target="custom",
        per_layer_overrides={
            0: {"num_attention_heads": 16},  # model has 8, not 16
        },
    )
    mismatches = validate_against_shape(model, constraint)
    assert any("layer 0" in m and "num_attention_heads" in m for m in mismatches), (
        f"Expected layer-0 num_attention_heads mismatch in {mismatches}"
    )


def test_per_layer_overrides_out_of_range_layer_index():
    """References to layers that don't exist on the model must surface."""
    blocks = [
        _FakeBlock(
            attn=_FakeAttn(num_heads=8, head_dim=64, num_kv_heads=8),
            ffn=_FakeFFN(hidden_dim=2048),
        )
        for _ in range(2)
    ]
    model = _FakeModel(d_model=512, vocab_size=256, blocks=blocks)

    constraint = ModelShapeConstraint(
        target="custom",
        per_layer_overrides={
            5: {"num_attention_heads": 8},  # model has 2 layers, not 6
        },
    )
    mismatches = validate_against_shape(model, constraint)
    assert any("layer 5" in m and "2 layers" in m for m in mismatches), (
        f"Expected out-of-range layer error in {mismatches}"
    )


def test_per_layer_overrides_reject_unknown_field_at_construction():
    """``per_layer_overrides`` only accepts the per-layer field names; a
    global-only field like ``d_model`` raises at constraint construction.
    """
    with pytest.raises(ValueError, match="unknown field"):
        ModelShapeConstraint(
            target="custom",
            per_layer_overrides={0: {"d_model": 512}},
        )


def test_per_layer_overrides_reject_non_int_layer_index():
    with pytest.raises(TypeError, match="int"):
        ModelShapeConstraint(
            target="custom",
            per_layer_overrides={"0": {"num_attention_heads": 8}},  # type: ignore[dict-item]
        )
