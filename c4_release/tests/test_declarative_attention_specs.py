"""Tests for declarative attention bake specs."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ops.l7_ops import _layer7_memory_head_specs  # noqa: E402
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import (  # noqa: E402
    AutoregressiveAttention,
    _SetDim,
    _set_layer7_memory_heads,
)


def test_layer7_memory_heads_declarative_byte_identical_to_legacy_helper():
    """The declarative L7 memory-head specs reproduce legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=7, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=7, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer7_memory_heads(legacy, 100.0, _SetDim, hd)
        # Keep this test aligned with make_layer7_memory_heads_op(): that op
        # preserves the post-helper softmax-sharpness fix for head 5 by doubling
        # its K row after the legacy bake.
        legacy.W_k.data[5 * hd] *= 2.0

        Primitives.generate_attention_heads(
            generated, _layer7_memory_head_specs(_SetDim), hd
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        legacy_tensor = getattr(legacy, name)
        generated_tensor = getattr(generated, name)
        assert torch.equal(legacy_tensor, generated_tensor), name
