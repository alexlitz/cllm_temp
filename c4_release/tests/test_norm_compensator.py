"""Qwen R1 Phase R1 — NORM_COMPENSATOR residual seed validation.

Two-mode test:

* **Flag ON** (``C4_QWEN_EXPORT_COMPAT=1``): compile + run ``IMM 42; EXIT``,
  capture the residual stream at the end of layers L0, L5, L10 and L17,
  and assert that ``residual[batch, pos, NORM_COMPENSATOR_idx]`` equals
  the seed constant ``K = 1000.0`` within ``1e-3``. The dim's K-preservation
  is the core acceptance criterion for R1 (see
  ``docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md`` §"Verification
  matrix"). The W_o / W_down rows for the slot are defensively zeroed at
  bake time so attention and FFN cannot clobber the constant.

* **Flag OFF**: the dim must not appear in ``layout.dim_positions`` and the
  compiled model's d_model + embedding weights must be byte-identical to
  a fresh ``compile_full_vm_dynamic()`` invoked without the flag, modulo
  the in-process / disk cache. The flag OFF mode is the smoke baseline,
  so any drift would surface as smoke regressions; this test catches
  shape / d_model drift directly without waiting for the full smoke pass.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

NORM_COMPENSATOR_K = 1000.0
K_TOLERANCE = 1.0e-3


def _compile_with_flag(flag_on: bool):
    """Compile the full VM with C4_QWEN_EXPORT_COMPAT set or unset.

    Uses ``disk_cache=False`` so each call recompiles deterministically
    from the current source tree; the in-process memo cache still applies
    within a session but its key includes the flag, so ON and OFF builds
    do not collide.
    """
    prior = os.environ.get("C4_QWEN_EXPORT_COMPAT")
    if flag_on:
        os.environ["C4_QWEN_EXPORT_COMPAT"] = "1"
    else:
        os.environ.pop("C4_QWEN_EXPORT_COMPAT", None)
    try:
        from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
            compile_full_vm_dynamic,
        )
        return compile_full_vm_dynamic(disk_cache=False)
    finally:
        if prior is None:
            os.environ.pop("C4_QWEN_EXPORT_COMPAT", None)
        else:
            os.environ["C4_QWEN_EXPORT_COMPAT"] = prior


@pytest.fixture(scope="module")
def compiled_on():
    """Compile once per module with the flag ON."""
    return _compile_with_flag(flag_on=True)


@pytest.fixture(scope="module")
def compiled_off():
    """Compile once per module with the flag OFF."""
    return _compile_with_flag(flag_on=False)


# ---------------------------------------------------------------------------
# Flag-ON tests
# ---------------------------------------------------------------------------


def test_norm_compensator_declared_when_flag_on(compiled_on):
    """The NORM_COMPENSATOR dim should be present in dim_positions with flag ON."""
    model, layout = compiled_on
    assert "NORM_COMPENSATOR" in layout.dim_positions, (
        "NORM_COMPENSATOR must be in dim_positions when C4_QWEN_EXPORT_COMPAT=1"
    )
    idx = layout.dim_positions["NORM_COMPENSATOR"]
    assert 0 <= idx < layout.d_model, (
        f"NORM_COMPENSATOR idx {idx} out of range [0, {layout.d_model})"
    )
    # Token embedding must be K=1000.0 for every token id at that slot.
    embed_weight = model.embed.embed.weight
    col = embed_weight[:, idx]
    assert torch.allclose(
        col,
        torch.full_like(col, NORM_COMPENSATOR_K),
        atol=K_TOLERANCE,
    ), (
        f"Token embedding column for NORM_COMPENSATOR must be K={NORM_COMPENSATOR_K}; "
        f"got min={col.min().item()} max={col.max().item()}"
    )


def test_norm_compensator_w_o_row_zero(compiled_on):
    """W_o[idx, :] must be zero on every block (defensive write-protection)."""
    model, layout = compiled_on
    idx = layout.dim_positions["NORM_COMPENSATOR"]
    for i, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "W_o"):
            continue
        w_o = attn.W_o.data
        if w_o.is_sparse:
            w_o = w_o.to_dense()
        if w_o.dim() != 2 or idx >= w_o.shape[0]:
            continue
        row_max = w_o[idx, :].abs().max().item()
        assert row_max == 0.0, (
            f"block[{i}].attn.W_o[{idx}, :] must be zero; got |max|={row_max}"
        )


def test_norm_compensator_w_down_row_zero(compiled_on):
    """W_down[idx, :] must be zero on every block FFN (defensive write-protection)."""
    model, layout = compiled_on
    idx = layout.dim_positions["NORM_COMPENSATOR"]

    def _walk_ffn(ffn):
        # Bare PureFFN exposes W_down directly; composites (Sequential) wrap
        # one or more sub-FFNs.
        if hasattr(ffn, "W_down"):
            yield ffn
            return
        try:
            children = list(ffn)
        except TypeError:
            return
        for child in children:
            yield from _walk_ffn(child)

    for i, block in enumerate(model.blocks):
        ffn = getattr(block, "ffn", None)
        if ffn is None:
            continue
        for j, sub in enumerate(_walk_ffn(ffn)):
            w_down = sub.W_down.data
            if w_down.is_sparse:
                w_down = w_down.to_dense()
            if w_down.dim() != 2 or idx >= w_down.shape[0]:
                continue
            row = w_down[idx, :]
            if row.numel() == 0:
                # ``right_size_ffns`` can leave a sub-FFN with hidden_dim=0
                # when nothing was ever programmed there; a zero-width row
                # is trivially "all zero".
                continue
            row_max = row.abs().max().item()
            assert row_max == 0.0, (
                f"block[{i}].ffn (sub {j}).W_down[{idx}, :] must be zero; "
                f"got |max|={row_max}"
            )


def test_k_preserved_through_sample_layers(compiled_on):
    """Compile + run ``IMM 42; EXIT`` and verify K survives at L0/L5/L10/L17."""
    model, layout = compiled_on
    idx = layout.dim_positions["NORM_COMPENSATOR"]

    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.embedding import Opcode

    runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    # Use the model we compiled (matching the flag-ON build).
    runner.model = model

    captures: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            captures[name] = output.detach().clone()
        return _hook

    sample_layers = [0, 5, 10, 17]
    handles = []
    for li in sample_layers:
        if li < len(model.blocks):
            handles.append(
                model.blocks[li].register_forward_hook(_make_hook(f"L{li}"))
            )

    try:
        bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
        try:
            runner.run(bytecode, b"", max_steps=10)
        except Exception:
            # Runner may exit before reaching all sample layers; the hooks
            # have still fired in the layers we care about.
            pass
    finally:
        for h in handles:
            h.remove()

    for li in sample_layers:
        if li >= len(model.blocks):
            continue
        name = f"L{li}"
        assert name in captures, f"forward hook for {name} did not fire"
        resid = captures[name]
        col = resid[..., idx]
        dev = (col - NORM_COMPENSATOR_K).abs().max().item()
        assert dev <= K_TOLERANCE, (
            f"K not preserved at {name}: max deviation {dev} > {K_TOLERANCE}; "
            f"actual range = [{col.min().item()}, {col.max().item()}]"
        )


# ---------------------------------------------------------------------------
# Flag-OFF tests
# ---------------------------------------------------------------------------


def test_norm_compensator_absent_when_flag_off(compiled_off):
    """With flag OFF, NORM_COMPENSATOR must not appear and d_model is unchanged."""
    model, layout = compiled_off
    assert "NORM_COMPENSATOR" not in layout.dim_positions, (
        "NORM_COMPENSATOR must NOT be in dim_positions when C4_QWEN_EXPORT_COMPAT is unset"
    )


def test_byte_identity_off_vs_repeat_off():
    """Compile twice with flag OFF and verify embedding + d_model are byte-identical.

    A direct byte-identity check against pre-R1 main isn't accessible from
    inside the test (no second worktree), but compiling twice in the same
    process under the OFF flag is the next-best invariant: any
    flag-dependent drift in shared code paths would surface as a delta
    between the two builds.
    """
    model_a, layout_a = _compile_with_flag(flag_on=False)
    # Force a separate compile by clearing the in-process memo cache for
    # the OFF snapshot so the second call walks the full bake.
    from neural_vm.unified_compiler import full_vm_compiler_dynamic as _fvm
    if hasattr(_fvm, "_INPROC_COMPILE_CACHE"):
        _fvm._INPROC_COMPILE_CACHE.clear()
    model_b, layout_b = _compile_with_flag(flag_on=False)

    assert layout_a.d_model == layout_b.d_model
    assert (
        layout_a.dim_positions.keys() == layout_b.dim_positions.keys()
    ), "OFF builds must declare identical dim sets"
    # Spot-check: token embedding tables byte-identical.
    assert torch.equal(model_a.embed.embed.weight, model_b.embed.embed.weight), (
        "OFF builds must produce byte-identical token embeddings"
    )
