"""Tests for declarative L10 byte-passthrough attention specs."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _bake_layer10_byte_passthrough_head,
    _bake_layer10_sp_byte_passthrough_head,
)
from neural_vm.vm_step import (  # noqa: E402
    AutoregressiveAttention,
    _SetDim,
    _set_layer10_byte_passthrough,
    _set_layer10_sp_byte_passthrough,
)


def _new_attention():
    d_model = 512
    num_heads = 8
    return AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=10, use_flash_attention=False
    )


def _assert_attention_equal(legacy, generated):
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name
    assert torch.equal(legacy.alibi_slopes, generated.alibi_slopes)


def test_l10_ax_byte_passthrough_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_byte_passthrough(legacy, 100.0, _SetDim, hd)
        _bake_layer10_byte_passthrough_head(generated, _SetDim, 100.0, hd)

    _assert_attention_equal(legacy, generated)


def test_l10_sp_byte_passthrough_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_sp_byte_passthrough(legacy, 100.0, _SetDim, hd)
        _bake_layer10_sp_byte_passthrough_head(generated, _SetDim, 100.0, hd)

    _assert_attention_equal(legacy, generated)
