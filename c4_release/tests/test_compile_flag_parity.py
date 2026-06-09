"""Parity tests for the model-semantics umbrella compile flags.

See ``docs/MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md``.

Two byte-identity claims gate the umbrella design:

1. ``preset="native"`` is byte-identical to the bare-defaults compile.
   This proves that the umbrella signature is purely additive — every
   existing call site sees no diff. This test passes today.

2. ``preset="native"`` and ``preset="qwen"`` produce byte-identical
   ``OUTPUT_LO`` / ``OUTPUT_HI`` on a single-token program. This is the
   semantic-parity claim: the variant choice is invariant at the
   observable output level, modulo intentional model-quality
   differences. This test is sketched here and ``pytest.skip``-ed until
   the per-axis variant implementations land.
"""

from __future__ import annotations

import os

import pytest
import torch

from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    _MODEL_SEMANTICS_AXES,
    _MODEL_SEMANTICS_PRESETS,
    _resolve_semantics_flags,
    compile_full_vm_dynamic,
)


# ---------------------------------------------------------------------------
# Preset table sanity (no model compile required).
# ---------------------------------------------------------------------------


def test_axes_table_has_six_entries():
    """The umbrella commits to exactly six axes."""
    assert len(_MODEL_SEMANTICS_AXES) == 6
    axis_names = {axis for axis, _ in _MODEL_SEMANTICS_AXES}
    assert axis_names == {
        "positional_encoding",
        "softmax_variant",
        "normalization",
        "ffn_variant",
        "per_head_qk_norm",
        "ffn_routing",
    }


def test_presets_cover_every_axis():
    """Each preset must set a value for every axis (no gaps)."""
    axis_names = {axis for axis, _ in _MODEL_SEMANTICS_AXES}
    for name, table in _MODEL_SEMANTICS_PRESETS.items():
        assert set(table.keys()) == axis_names, name


def test_preset_native_uses_native_values():
    """``preset="native"`` picks the first value of each axis."""
    for axis, values in _MODEL_SEMANTICS_AXES:
        assert _MODEL_SEMANTICS_PRESETS["native"][axis] == values[0]


def test_preset_qwen_uses_qwen_values():
    """``preset="qwen"`` picks the second value of each axis."""
    for axis, values in _MODEL_SEMANTICS_AXES:
        assert _MODEL_SEMANTICS_PRESETS["qwen"][axis] == values[1]


# ---------------------------------------------------------------------------
# _resolve_semantics_flags: error surface + default behavior.
# ---------------------------------------------------------------------------


def test_resolve_no_flags_is_passthrough():
    """No umbrella flag supplied → legacy kwargs flow through unchanged.

    This is the backward-compat fast path. The bake pipeline downstream
    must see exactly the same (positional_encoding, attention_normalization,
    use_rms_norm, enable_moe_routing) tuple it would have seen before the
    umbrella existed.
    """
    out = _resolve_semantics_flags(
        preset=None,
        positional_encoding=None,
        softmax_variant=None,
        attention_normalization="softmax1",
        normalization=None,
        use_rms_norm=False,
        ffn_variant=None,
        per_head_qk_norm=None,
        ffn_routing=None,
        enable_moe_routing=False,
        arch=None,
    )
    assert out == (None, "softmax1", False, False)


def test_resolve_preset_native_pins_legacy_kwargs():
    """``preset="native"`` resolves to the legacy native values."""
    out = _resolve_semantics_flags(
        preset="native",
        positional_encoding=None,
        softmax_variant=None,
        attention_normalization="softmax1",
        normalization=None,
        use_rms_norm=False,
        ffn_variant=None,
        per_head_qk_norm=None,
        ffn_routing=None,
        enable_moe_routing=True,  # env-derived; preset must clobber
        arch=None,
    )
    pos_enc, attn_norm, use_rms, moe = out
    assert pos_enc == "alibi"
    assert attn_norm == "softmax1"
    assert use_rms is False
    assert moe is False  # preset="native" → ffn_routing="single"


def test_resolve_single_axis_ablation_leaves_others_alone():
    """A single per-axis kwarg ablates only that axis.

    The other axes keep their incoming legacy-kwarg values — the
    umbrella does not silently force unrelated axes to native.
    """
    out = _resolve_semantics_flags(
        preset=None,
        positional_encoding=None,
        softmax_variant="standard",
        attention_normalization="softmax1",
        normalization=None,
        use_rms_norm=True,  # caller-supplied, must survive
        ffn_variant=None,
        per_head_qk_norm=None,
        ffn_routing=None,
        enable_moe_routing=True,  # env-set, must survive
        arch=None,
    )
    pos_enc, attn_norm, use_rms, moe = out
    assert pos_enc is None  # unchanged
    assert attn_norm == "softmax"  # softmax_variant="standard" → "softmax"
    assert use_rms is True  # preserved
    assert moe is True  # preserved


def test_resolve_invalid_value_rejected():
    """An out-of-range per-axis value is a ``ValueError``."""
    with pytest.raises(ValueError, match="softmax_variant"):
        _resolve_semantics_flags(
            preset=None,
            positional_encoding=None,
            softmax_variant="not-a-real-value",
            attention_normalization=None,
            normalization=None,
            use_rms_norm=None,
            ffn_variant=None,
            per_head_qk_norm=None,
            ffn_routing=None,
            enable_moe_routing=None,
            arch=None,
        )


def test_resolve_invalid_preset_rejected():
    """An unknown preset is a ``ValueError``."""
    with pytest.raises(ValueError, match="preset"):
        _resolve_semantics_flags(
            preset="not-a-real-preset",
            positional_encoding=None,
            softmax_variant=None,
            attention_normalization=None,
            normalization=None,
            use_rms_norm=None,
            ffn_variant=None,
            per_head_qk_norm=None,
            ffn_routing=None,
            enable_moe_routing=None,
            arch=None,
        )


def test_resolve_swiglu_reserved_not_implemented():
    """``ffn_variant="swiglu"`` is reserved on the signature only."""
    with pytest.raises(NotImplementedError, match="swiglu"):
        _resolve_semantics_flags(
            preset=None,
            positional_encoding=None,
            softmax_variant=None,
            attention_normalization=None,
            normalization=None,
            use_rms_norm=None,
            ffn_variant="swiglu",
            per_head_qk_norm=None,
            ffn_routing=None,
            enable_moe_routing=None,
            arch=None,
        )


def test_resolve_per_head_qk_norm_qwen_reserved_not_implemented():
    """``per_head_qk_norm="qwen"`` is reserved on the signature only."""
    with pytest.raises(NotImplementedError, match="per_head_qk_norm"):
        _resolve_semantics_flags(
            preset=None,
            positional_encoding=None,
            softmax_variant=None,
            attention_normalization=None,
            normalization=None,
            use_rms_norm=None,
            ffn_variant=None,
            per_head_qk_norm="qwen",
            ffn_routing=None,
            enable_moe_routing=None,
            arch=None,
        )


def test_resolve_preset_with_arch_is_rejected():
    """Mixing ``preset=`` with ``arch=`` mirrors the existing
    ``arch=`` vs individual-kwarg gate."""

    class _FakeArch:
        """Stand-in for ``ModelArchitectureSpec`` -- the gate just
        checks ``arch is not None`` before any field access."""

    with pytest.raises(TypeError, match="model-semantics umbrella"):
        _resolve_semantics_flags(
            preset="native",
            positional_encoding=None,
            softmax_variant=None,
            attention_normalization=None,
            normalization=None,
            use_rms_norm=None,
            ffn_variant=None,
            per_head_qk_norm=None,
            ffn_routing=None,
            enable_moe_routing=None,
            arch=_FakeArch(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Byte-identity parity: preset="native" == bare default.
# ---------------------------------------------------------------------------


def _clear_semantics_env(monkeypatch):
    """Pin every env-driven default so the test compares like-for-like.

    The umbrella's backward-compat claim is "identical legacy kwargs in
    => identical bake out". Env-driven defaults (``C4_ENABLE_MOE_ROUTING``,
    ``NEURAL_VM_POS_ENCODING``, etc.) would otherwise drift the baseline
    on any host that has them set.
    """
    for var in (
        "C4_ENABLE_MOE_ROUTING",
        "C4_BATCH_ENABLE_MOE_ROUTING",
        "C4_DECLARATIONS_ONLY_BAKE",
        "C4_REQUIRE_DECLARATIVE_BAKE",
        "C4_QWEN_EXPORT_COMPAT",
        "NEURAL_VM_POS_ENCODING",
        "NEURAL_VM_ATTENTION_NORMALIZATION",
        "NEURAL_VM_USE_RMS_NORM",
    ):
        monkeypatch.delenv(var, raising=False)


def _walk_attention_weights(model):
    """Yield every attention block's weight + buffer tensors by name.

    The block iterator handles both ``model.blocks`` (production
    ``AutoregressiveVM``) and a flat sequence — keeps the test robust
    to the post-compile rebuild path.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:  # pragma: no cover - defensive
        return
    for i, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        for name in ("W_q", "W_k", "W_v", "W_o"):
            tensor = getattr(attn, name, None)
            if isinstance(tensor, torch.nn.Parameter):
                yield f"block[{i}].attn.{name}", tensor.data
            elif isinstance(tensor, torch.Tensor):
                yield f"block[{i}].attn.{name}", tensor


def test_preset_native_byte_identical_to_default(monkeypatch):
    """``compile_full_vm_dynamic(preset="native")`` byte-identical to
    ``compile_full_vm_dynamic()``.

    The load-bearing backward-compat guarantee: the umbrella signature
    is purely additive. The two compiles must produce identical
    attention weights, identical FFN weights, and identical
    ``layout.dim_positions``.
    """
    _clear_semantics_env(monkeypatch)

    m1, layout1 = compile_full_vm_dynamic(disk_cache=False)
    m2, layout2 = compile_full_vm_dynamic(preset="native", disk_cache=False)

    # Layout parity: every dim position matches.
    assert layout1.dim_positions == layout2.dim_positions
    assert layout1.dim_sizes == layout2.dim_sizes
    assert layout1.d_model == layout2.d_model

    # Attention-weight parity: every Q/K/V/O matrix in every block.
    weights1 = dict(_walk_attention_weights(m1))
    weights2 = dict(_walk_attention_weights(m2))
    assert weights1.keys() == weights2.keys(), (
        f"attention weight name sets differ: "
        f"only-in-default={weights1.keys() - weights2.keys()}, "
        f"only-in-preset={weights2.keys() - weights1.keys()}"
    )
    for name in weights1:
        assert torch.equal(weights1[name], weights2[name]), (
            f"attention weight {name} differs between default and "
            f"preset='native' compiles"
        )


def test_preset_native_byte_identical_full_state_dict(monkeypatch):
    """Full-state-dict variant of the above: every parameter and buffer
    on the compiled model must match byte-for-byte.

    Catches umbrella-induced drift in FFN weights, norm scales, mask
    buffers, RoPE caches, and any other parameter the per-block walker
    misses.
    """
    _clear_semantics_env(monkeypatch)

    m1, _ = compile_full_vm_dynamic(disk_cache=False)
    m2, _ = compile_full_vm_dynamic(preset="native", disk_cache=False)

    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys(), (
        f"state_dict key sets differ: "
        f"only-in-default={sd1.keys() - sd2.keys()}, "
        f"only-in-preset={sd2.keys() - sd1.keys()}"
    )
    mismatches = [
        name
        for name in sd1
        if not torch.equal(sd1[name], sd2[name])
    ]
    assert not mismatches, (
        f"{len(mismatches)} parameters/buffers differ between default "
        f"and preset='native': {mismatches[:5]!r}"
        + ("..." if len(mismatches) > 5 else "")
    )


# ---------------------------------------------------------------------------
# Deferred: semantic parity preset="native" == preset="qwen" at OUTPUT_LO/HI.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Semantic parity (preset='qwen' vs preset='native' equal at "
        "OUTPUT_LO/HI on a single-token program) is the deferred claim. "
        "Lands when the per-axis variant docs (RoPE, standard softmax, "
        "RMSNorm, SwiGLU, per-head qk-norm, composite routing) ship "
        "their lowerings. Tracked in "
        "docs/MODEL_SEMANTICS_COMPILE_FLAGS_2026_06_09.md §4.2."
    )
)
def test_preset_qwen_output_parity_single_token():
    """Sketch: feed a single-token program through both preset compiles,
    assert ``OUTPUT_LO`` / ``OUTPUT_HI`` byte-equal.

    The premise: the semantic dim bands are the same regardless of which
    attention / FFN / norm representation the compute layers use. So
    even though ``W_q`` / ``W_k`` / ``W_v`` / ``W_o`` storage shapes
    differ across presets, every observable VM output is invariant.
    """
    # Implementation sketch:
    #   m_native, layout_n = compile_full_vm_dynamic(
    #       preset="native", disk_cache=False
    #   )
    #   m_qwen,   layout_q = compile_full_vm_dynamic(
    #       preset="qwen", disk_cache=False
    #   )
    #   tok = single_token_program(...)
    #   out_n = run(m_native, tok)
    #   out_q = run(m_qwen, tok)
    #   for band in ("OUTPUT_LO", "OUTPUT_HI"):
    #       lo_n, hi_n = layout_n.dim_positions[band], ...
    #       assert (
    #           out_n[..., lo_n:hi_n].to(torch.int) ==
    #           out_q[..., lo_q:hi_q].to(torch.int)
    #       ).all()
    raise AssertionError("unreachable: skip mark above gates execution")
