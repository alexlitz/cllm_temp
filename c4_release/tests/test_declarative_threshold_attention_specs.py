"""Tests for declarative threshold-attention bake specs."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import (  # noqa: E402
    AutoregressiveAttention,
    _SetDim,
    _set_threshold_attn,
)


def _new_attention(layer_idx: int = 0):
    d_model = 512
    num_heads = 8
    return AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=layer_idx, use_flash_attention=False
    )


def _assert_threshold_specs_match_legacy(
    *,
    layer_idx: int,
    thresholds,
    out_bases,
    heads=None,
    slope: float = 10.0,
):
    legacy = _new_attention(layer_idx)
    generated = _new_attention(layer_idx)
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_threshold_attn(
            legacy,
            thresholds,
            out_bases,
            slope,
            hd,
            heads=heads,
            BD=_SetDim,
        )
        Primitives.generate_threshold_attention_heads(
            generated,
            thresholds,
            out_bases,
            slope,
            hd,
            heads=heads,
            bd=_SetDim,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_l0_threshold_attention_specs_match_legacy_helper():
    _assert_threshold_specs_match_legacy(
        layer_idx=0,
        thresholds=[3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5],
        out_bases=[
            _SetDim.H0,
            _SetDim.H1,
            _SetDim.H2,
            _SetDim.H3,
            _SetDim.H4,
            _SetDim.H5,
            _SetDim.H6,
            _SetDim.H7,
        ],
    )


def test_l1_threshold_attention_specs_match_legacy_helper():
    _assert_threshold_specs_match_legacy(
        layer_idx=1,
        thresholds=[0.5, 1.5, 2.5, 6.5],
        out_bases=[_SetDim.L1H0, _SetDim.L1H1, _SetDim.L1H2, _SetDim.L1H4],
        heads=[0, 1, 2, 4],
    )


def test_l2_threshold_attention_specs_match_legacy_helper():
    _assert_threshold_specs_match_legacy(
        layer_idx=2,
        thresholds=[5.5],
        out_bases=[_SetDim.L2H0],
        heads=[0],
    )
