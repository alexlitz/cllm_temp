"""End-to-end tests for the ``preset=`` umbrella on ``compile_full_vm_dynamic``.

Sister file to ``tests/test_compile_flag_parity.py``: that file covers the
unit-level invariants on the preset/axis tables and the
``_resolve_semantics_flags`` shape; this file gates the full
``compile_full_vm_dynamic`` call paths, which is what every downstream
caller actually invokes.

Three claims pinned here:

  1. A bare ``compile_full_vm_dynamic()`` and ``preset="native"`` produce
     byte-identical ``state_dict()``s. This is the backward-compat gate
     for the umbrella signature.
  2. ``preset="qwen"`` reaches a ``(model, layout)`` return without
     raising. The output is NOT byte-equal to a Qwen reference -- the
     unwired axes (SwiGLU, per-head Q/K norm) still fall back to the
     legacy bake. But the plumbing must not crash when callers opt into
     the open-weights configuration via the preset.
  3. The plumbing reaches the runtime model: at minimum
     ``positional_encoding``, ``attention_normalization`` and
     ``use_rms_norm`` on the compiled model reflect the preset choice.

The semantic-output parity claim (``preset='qwen'`` producing
identical ``OUTPUT_LO`` / ``OUTPUT_HI`` to ``preset='native'`` on a
single-token program) lives in ``test_compile_flag_parity.py`` and is
``pytest.skip``-ed there until the per-axis variant lowerings land.
"""

from __future__ import annotations

import pytest
import torch

from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compile_full_vm_dynamic,
)


# ---------------------------------------------------------------------------
# Env hygiene: pin every env-driven default so compiles compare like-for-like.
# ---------------------------------------------------------------------------


def _clear_semantics_env(monkeypatch):
    """Drop every env var that would otherwise drift the bake baseline.

    The umbrella's backward-compat claim is "identical kwargs in =>
    identical bake out". Env-driven defaults
    (``C4_ENABLE_MOE_ROUTING``, ``NEURAL_VM_POS_ENCODING``, etc.) would
    otherwise shift the baseline on any host that has them set.
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


def _state_dict_mismatches(sd1, sd2):
    """Return a list of ``(key, reason)`` tuples where ``sd1`` differs from ``sd2``.

    Catches missing keys, shape/dtype drift, and value drift. Returned
    list is empty iff the two state dicts are byte-identical.
    """
    diffs = []
    for k in set(sd1) ^ set(sd2):
        diffs.append((k, "missing"))
    for k in sd1.keys() & sd2.keys():
        t1, t2 = sd1[k], sd2[k]
        if t1.shape != t2.shape:
            diffs.append((k, f"shape {tuple(t1.shape)} != {tuple(t2.shape)}"))
        elif t1.dtype != t2.dtype:
            diffs.append((k, f"dtype {t1.dtype} != {t2.dtype}"))
        elif not torch.equal(t1, t2):
            diffs.append((k, "values"))
    return diffs


# ---------------------------------------------------------------------------
# Claim 1: default compile == preset="native" (byte-identical state_dict).
# ---------------------------------------------------------------------------


def test_default_compile_equals_preset_native_byte_identical(monkeypatch):
    """A no-flag compile and ``preset='native'`` must produce identical
    ``state_dict()``s.

    This is the byte-identity gate for the umbrella surface: if it ever
    fails, ``preset='native'`` is silently selecting different semantics
    than the historical default and every downstream test that runs
    against the default would diverge. Every parameter and buffer on the
    compiled model must match byte-for-byte.
    """
    _clear_semantics_env(monkeypatch)

    m_default, _ = compile_full_vm_dynamic(disk_cache=False)
    m_native, _ = compile_full_vm_dynamic(preset="native", disk_cache=False)

    sd_default = m_default.state_dict()
    sd_native = m_native.state_dict()
    diffs = _state_dict_mismatches(sd_default, sd_native)
    assert not diffs, (
        f"preset='native' diverges from default compile on "
        f"{len(diffs)} tensors: {diffs[:5]!r}"
        + ("..." if len(diffs) > 5 else "")
    )


# ---------------------------------------------------------------------------
# Claim 2: preset="qwen" runs without crash and reaches the runtime model.
# ---------------------------------------------------------------------------


def test_preset_qwen_compiles_without_crash(monkeypatch):
    """``preset='qwen'`` must reach a ``(model, layout)`` return without
    raising.

    The output is NOT byte-equal to a Qwen reference -- the unwired axes
    (SwiGLU, per-head Q/K norm) fall back to the legacy bake. But the
    plumbing must not crash for callers that opt into the open-weights
    configuration via the preset shortcut.
    """
    _clear_semantics_env(monkeypatch)

    model, layout = compile_full_vm_dynamic(preset="qwen", disk_cache=False)
    assert model is not None
    assert layout is not None


def test_preset_qwen_propagates_to_runtime_model(monkeypatch):
    """``preset='qwen'`` must wire its three legacy-aliased axes
    through to the compiled model.

    The three axes that the bake pipeline already consumes
    (``positional_encoding``, ``attention_normalization``, ``use_rms_norm``)
    must reflect the preset choice on the returned model. This is the
    spot-check that the preset isn't being silently dropped between
    ``_resolve_semantics_flags`` and the downstream ctors.
    """
    _clear_semantics_env(monkeypatch)

    model, _ = compile_full_vm_dynamic(preset="qwen", disk_cache=False)
    assert getattr(model, "positional_encoding", None) == "rope"
    assert getattr(model, "attention_normalization", None) == "softmax"
    assert bool(getattr(model, "use_rms_norm", False)) is True


# ---------------------------------------------------------------------------
# Claim 3: per-axis ablation on a preset is allowed (composition rule).
# ---------------------------------------------------------------------------


def test_preset_native_with_axis_override_compiles(monkeypatch):
    """``preset='native'`` + a single per-axis override compiles cleanly.

    The umbrella's documented composition rule (§3 of the design doc) is
    "preset + per-axis kwarg = per-axis kwarg overrides preset on that
    axis only". The other axes keep the preset's value. This is the
    one-axis ablation surface that callers use for A/B testing a single
    semantic at a time.
    """
    _clear_semantics_env(monkeypatch)

    # Override softmax to "standard" on top of preset="native"; this is
    # the lightest cross-axis combination that exercises the per-axis
    # override path through the bake pipeline.
    model, layout = compile_full_vm_dynamic(
        preset="native",
        softmax_variant="standard",
        disk_cache=False,
    )
    assert model is not None
    assert layout is not None
    assert getattr(model, "attention_normalization", None) == "softmax"
    # The other axes are unchanged from preset="native".
    assert getattr(model, "positional_encoding", None) == "alibi"
    assert bool(getattr(model, "use_rms_norm", False)) is False


# ---------------------------------------------------------------------------
# Surface-level error contracts.
# ---------------------------------------------------------------------------


def test_unknown_preset_raises_value_error(monkeypatch):
    """An unknown preset label is a programming error; reject it loudly
    with the accepted set listed in the message.
    """
    _clear_semantics_env(monkeypatch)
    with pytest.raises(ValueError, match="preset"):
        compile_full_vm_dynamic(
            preset="totally-not-a-preset", disk_cache=False
        )


def test_unknown_per_axis_value_raises_value_error(monkeypatch):
    """Invalid per-axis values are rejected loudly.

    The vocabulary is the umbrella's, not the legacy bake's, so the
    caller sees the documented axis name (``softmax_variant``) in the
    error rather than the legacy term (``attention_normalization``).
    """
    _clear_semantics_env(monkeypatch)
    with pytest.raises(ValueError, match="softmax_variant"):
        compile_full_vm_dynamic(
            softmax_variant="not-a-softmax", disk_cache=False
        )
    with pytest.raises(ValueError, match="normalization"):
        compile_full_vm_dynamic(
            normalization="not-a-norm", disk_cache=False
        )
    with pytest.raises(ValueError, match="ffn_variant"):
        compile_full_vm_dynamic(
            ffn_variant="not-an-ffn", disk_cache=False
        )
    with pytest.raises(ValueError, match="per_head_qk_norm"):
        compile_full_vm_dynamic(
            per_head_qk_norm="not-a-norm", disk_cache=False
        )
