"""Phase 10.B regression: wrapper-block expansion reduction.

``_expand_wrapper_blocks`` (``neural_vm/vm_step.py``) inflates the 18
native transformer layers into ~32 blocks by wrapping each ``post_op``
in a dedicated ``TransformerBlock`` with a zero-init passthrough
attention. Each wrapper attention costs ``4 * d_model**2`` params and
contributes nothing functionally — only the FFN slot of the wrapper
block (the post_op) actually runs.

This test pins:

  * default (no env flag): the legacy behavior remains. The model
    expands to >= 30 blocks and >= 180M parameters at the production
    d_model.
  * merged (``C4_DISABLE_WRAPPER_EXPANSION=1``): wrapper expansion is
    skipped; each merge-safe post_op is folded into the parent block's
    FFN as an ``nn.Sequential`` pipeline. The block count drops back to
    the native layer count and the parameter savings exceed the Phase
    10.B 35M floor.

The merged path's forward math is identical on the default
``use_rms_norm=False`` path: each ``PureFFN`` / ALU composite already
bakes the ``x + delta(x)`` residual into its own forward, so chaining
them via ``nn.Sequential`` produces the same residual chain that the
expanded path produces through the zero-init wrapper attention
(``passthrough_attn(x) == x`` because all the attention output
projection weights are zero).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _compile_with_flag(disable: bool):
    """Compile the full VM with the wrapper-expansion env flag toggled.

    Imports ``compile_full_vm_dynamic`` after setting the flag so that the
    module-level dispatch in ``make_expand_wrapper_blocks_op`` reads the
    intended value of ``C4_DISABLE_WRAPPER_EXPANSION``.
    """
    if disable:
        os.environ["C4_DISABLE_WRAPPER_EXPANSION"] = "1"
    else:
        os.environ.pop("C4_DISABLE_WRAPPER_EXPANSION", None)
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    # The flag is folded into the cache key via the kwargs_snapshot
    # (full_vm_compiler_dynamic.py), so disk_cache=True returns the right
    # branch on each call without cross-contamination.
    model, _layout = compile_full_vm_dynamic(strict=False)
    return model


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.fixture(scope="module")
def baseline_model():
    """Default-expansion model (legacy wrapper-block path)."""
    return _compile_with_flag(disable=False)


@pytest.fixture(scope="module")
def merged_model():
    """Phase 10.B merged model: post_ops folded into parent FFNs."""
    return _compile_with_flag(disable=True)


def test_baseline_block_count_at_least_30(baseline_model):
    """Legacy expansion produces >= 30 blocks (17 native + 13-14 wrappers)."""
    n_blocks = len(baseline_model.blocks)
    assert n_blocks >= 30, (
        f"Expected >=30 blocks under legacy wrapper expansion, got {n_blocks}"
    )


def test_merged_block_count_at_most_18(merged_model):
    """Merged path keeps block count == n_native_layers (18 in the current
    layout: L0..L17). The merge folds all wrapper post_ops into their parent
    block's FFN slot, so the only blocks left are the native transformer
    layers the compiler emits at op-collection time. The Phase 10.B brief
    cited "~17" as the native layer count; the production op set today
    emits 18, so we use 18 as the upper bound."""
    n_blocks = len(merged_model.blocks)
    assert n_blocks <= 18, (
        f"Expected <=18 blocks with wrapper expansion disabled, got {n_blocks}"
    )


def test_merged_saves_at_least_35M_params(baseline_model, merged_model):
    """Param savings >= Phase 10.B target (~35M, matching the baseline
    measurement: 14 wrappers * 4 * 800**2 = 35.84M dead attention params).
    Each wrapper attention is a fresh zero-init AutoregressiveAttention
    whose 4*d_model**2 weights are pure dead weight."""
    baseline_params = _param_count(baseline_model)
    merged_params = _param_count(merged_model)
    savings = baseline_params - merged_params
    assert savings >= 35_000_000, (
        f"Phase 10.B target: expected >=35M saved, got "
        f"baseline={baseline_params:,} merged={merged_params:,} "
        f"savings={savings:,}"
    )


def test_merged_forward_runs(merged_model):
    """Merged model's forward pass executes without crashing."""
    merged_model.eval()
    device = next(merged_model.parameters()).device
    x = torch.zeros(1, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        out = merged_model(x)
    assert out.dim() == 3
    assert out.shape[0] == 1 and out.shape[1] == 16


def test_merged_block_ffns_are_sequential_or_pure(merged_model):
    """After the merge, every block's FFN is either a bare PureFFN-class
    module or an ``nn.Sequential`` pipeline (post_op chain)."""
    for i, block in enumerate(merged_model.blocks):
        ffn = block.ffn
        is_sequential = isinstance(ffn, torch.nn.Sequential)
        # Any non-Sequential FFN must at least carry W_up / W_down so it
        # behaves like a PureFFN-equivalent at the leaf level.
        if not is_sequential:
            assert hasattr(ffn, "W_up") or hasattr(ffn, "weight"), (
                f"L{i} ffn={type(ffn).__name__} is neither Sequential nor "
                f"a PureFFN-equivalent"
            )
        # The merge path clears post_ops; the only post_ops remaining
        # should be empty ModuleLists.
        post_ops = getattr(block, "post_ops", None)
        assert post_ops is None or len(post_ops) == 0, (
            f"L{i} still has post_ops after merge: "
            f"{[type(o).__name__ for o in post_ops]}"
        )
