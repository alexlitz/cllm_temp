"""Phase R5 — post_ops flattening for Qwen structural export.

Validates that
:func:`neural_vm.qwen_compat.flatten_post_ops_for_qwen_export` produces a list
of TransformerBlocks whose chained forward equals the source model's
block-by-block forward, and that the expanded layer count matches the
``sum(1 + len(post_ops))`` formula from the structural adapter plan §R5.
"""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.base_layers import PureFFN
from neural_vm.qwen_compat import (
    count_post_ops,
    expanded_qwen_layer_count,
    flatten_post_ops_for_qwen_export,
    summarize_post_ops_flattening,
)
from neural_vm.vm_step import AutoregressiveVM, TransformerBlock


D_MODEL = 32
N_HEADS = 4
FFN_HIDDEN = 64


def _tiny_vm(use_rms_norm: bool = False) -> AutoregressiveVM:
    """A small VM whose attention shape matches what R5 will receive in tests."""

    return AutoregressiveVM(
        vocab_size=276,
        d_model=D_MODEL,
        n_layers=2,
        n_heads=N_HEADS,
        ffn_hidden=FFN_HIDDEN,
        max_seq_len=128,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_rms_norm=use_rms_norm,
        use_flash_attention=False,
    )


def _random_pure_ffn(seed: int) -> PureFFN:
    """Build a PureFFN with small random weights so its forward delta is non-zero.

    PureFFN's forward is ``x + W_down(silu(W_up x + b_up) * (W_gate x + b_gate)) + b_down``
    — the residual is built in, matching how post_ops compose on
    :class:`TransformerBlock` (each post_op consumes and returns the residual
    stream).
    """

    gen = torch.Generator().manual_seed(seed)
    ffn = PureFFN(dim=D_MODEL, hidden_dim=FFN_HIDDEN)
    with torch.no_grad():
        ffn.W_up.normal_(generator=gen, std=0.02)
        ffn.W_gate.normal_(generator=gen, std=0.02)
        ffn.W_down.normal_(generator=gen, std=0.02)
        ffn.b_up.normal_(generator=gen, std=0.02)
        ffn.b_gate.normal_(generator=gen, std=0.02)
        ffn.b_down.normal_(generator=gen, std=0.02)
    return ffn


def _attach_random_post_ops(model: AutoregressiveVM, plan: dict[int, int]) -> None:
    """Append ``count`` randomly-initialised PureFFN post_ops to each indexed block."""

    seed = 1000
    for block_idx, count in plan.items():
        block = model.blocks[block_idx]
        for _ in range(count):
            block.post_ops.append(_random_pure_ffn(seed=seed))
            seed += 1


# ---------------------------------------------------------------------------
# Layer count
# ---------------------------------------------------------------------------


def test_expanded_layer_count_with_no_post_ops_equals_block_count():
    vm = _tiny_vm()
    assert count_post_ops(vm) == [0, 0]
    assert expanded_qwen_layer_count(vm) == 2

    report = summarize_post_ops_flattening(vm)
    assert report.original_block_count == 2
    assert report.post_op_counts == (0, 0)
    assert report.expanded_block_count == 2
    assert report.total_post_ops == 0


def test_expanded_layer_count_matches_post_op_sum_formula():
    vm = _tiny_vm()
    _attach_random_post_ops(vm, plan={0: 3, 1: 1})

    counts = count_post_ops(vm)
    assert counts == [3, 1]

    # The Qwen structural adapter plan §R5: expanded layer count is
    # sum(1 + len(block.post_ops) for block in model.blocks)
    assert expanded_qwen_layer_count(vm) == 1 + 3 + 1 + 1
    assert expanded_qwen_layer_count(vm) == sum(
        1 + len(b.post_ops) for b in vm.blocks
    )


def test_flatten_returns_one_block_per_layer_plus_post_op():
    vm = _tiny_vm()
    _attach_random_post_ops(vm, plan={0: 2, 1: 0})

    flattened = flatten_post_ops_for_qwen_export(vm)

    # 2 blocks: block 0 with 2 post_ops -> 3 blocks; block 1 with 0 -> 1 block
    assert len(flattened) == expanded_qwen_layer_count(vm) == 4
    for block in flattened:
        assert isinstance(block, TransformerBlock)
        # Flattened blocks must not themselves carry post_ops.
        assert len(block.post_ops) == 0


# ---------------------------------------------------------------------------
# Skip-pass structure
# ---------------------------------------------------------------------------


def test_skip_pass_blocks_have_zero_init_attention():
    """Each post-op block's attention must be a true pass-through."""

    vm = _tiny_vm()
    _attach_random_post_ops(vm, plan={0: 2, 1: 1})

    flattened = flatten_post_ops_for_qwen_export(vm)

    # Block layout: [base_0, skip_0a, skip_0b, base_1, skip_1a]
    skip_pass_indices = [1, 2, 4]
    for idx in skip_pass_indices:
        attn = flattened[idx].attn
        assert torch.equal(attn.W_q, torch.zeros_like(attn.W_q))
        assert torch.equal(attn.W_k, torch.zeros_like(attn.W_k))
        assert torch.equal(attn.W_v, torch.zeros_like(attn.W_v))
        assert torch.equal(attn.W_o, torch.zeros_like(attn.W_o))


def test_base_blocks_share_parameters_with_source():
    """Base block parameters must be identical objects to source — no clone.

    The Qwen2 mapping plan copies ``blocks.<i>.attn.W_q`` and friends. Keeping
    parameter identity guarantees the existing copy mappings hit the right
    tensors after R5 flattening.
    """

    vm = _tiny_vm()
    _attach_random_post_ops(vm, plan={0: 1, 1: 2})

    flattened = flatten_post_ops_for_qwen_export(vm)

    # Base blocks are at indices 0 and 2 (1 base + 1 skip + 1 base + 2 skip).
    assert flattened[0].attn.W_q.data_ptr() == vm.blocks[0].attn.W_q.data_ptr()
    assert flattened[0].ffn.W_up.data_ptr() == vm.blocks[0].ffn.W_up.data_ptr()
    assert flattened[2].attn.W_q.data_ptr() == vm.blocks[1].attn.W_q.data_ptr()
    assert flattened[2].ffn.W_up.data_ptr() == vm.blocks[1].ffn.W_up.data_ptr()


# ---------------------------------------------------------------------------
# Forward equivalence — the structural-adapter §R5 acceptance criterion.
# ---------------------------------------------------------------------------


def _forward_through_blocks(blocks, x: torch.Tensor) -> torch.Tensor:
    out = x
    for block in blocks:
        out = block(out)
    return out


def test_flattened_blocks_match_source_forward_on_single_token():
    """Phase R5 acceptance: single-token forward parity within fp32 tolerance.

    The plan calls for "final residual byte-identical". Our skip-pass
    attention is constructed to produce a true zero (zero W_o folds an
    arbitrary attention pattern back to the zero tensor before the residual
    add), but a tolerance margin of 1e-6 accommodates accumulating softmax
    + matmul noise from the non-zero base blocks.
    """

    torch.manual_seed(0)
    vm = _tiny_vm()
    _attach_random_post_ops(vm, plan={0: 2, 1: 1})

    flattened = flatten_post_ops_for_qwen_export(vm)

    x = torch.randn(1, 1, D_MODEL)

    expected = _forward_through_blocks(vm.blocks, x)
    actual = _forward_through_blocks(flattened, x)

    assert expected.shape == actual.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_flattened_blocks_match_source_forward_on_short_sequence():
    """Multi-token sequence also matches — the skip-pass attention is
    position-invariant (zero output for every position) so the equivalence
    is not coincidental on length 1."""

    torch.manual_seed(1)
    vm = _tiny_vm()
    _attach_random_post_ops(vm, plan={0: 1, 1: 2})

    flattened = flatten_post_ops_for_qwen_export(vm)

    x = torch.randn(1, 5, D_MODEL)

    expected = _forward_through_blocks(vm.blocks, x)
    actual = _forward_through_blocks(flattened, x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_flatten_is_non_mutating():
    """The flattening pass must not pop post_ops off the source blocks."""

    vm = _tiny_vm()
    _attach_random_post_ops(vm, plan={0: 2, 1: 1})

    pre_counts = count_post_ops(vm)
    _ = flatten_post_ops_for_qwen_export(vm)
    post_counts = count_post_ops(vm)

    assert pre_counts == post_counts == [2, 1]


# ---------------------------------------------------------------------------
# RMSNorm path
# ---------------------------------------------------------------------------


def test_flatten_with_rms_norm_keeps_base_norms_and_disables_skip_pass_norm():
    """When use_rms_norm=True (R2 prep), the base block must keep the original
    norm tensors. Skip-pass blocks must run *without* RMSNorm so the
    zero-attention residual stays a true pass-through (see the docstring on
    ``_make_skip_pass_block`` for the math).
    """

    torch.manual_seed(2)
    vm = _tiny_vm(use_rms_norm=True)

    # Give the norms non-trivial values so the rms-norm path is actually
    # exercised — not silently a no-op via gamma==1.
    with torch.no_grad():
        for block in vm.blocks:
            block.attn_norm.weight.normal_(std=0.05)
            block.ffn_norm.weight.normal_(std=0.05)

    _attach_random_post_ops(vm, plan={0: 1, 1: 0})

    flattened = flatten_post_ops_for_qwen_export(vm)
    assert len(flattened) == 3

    # Base blocks at indices 0 and 2 reuse the original RMSNorm modules
    # (object identity, so the Qwen mapping plan's
    # ``blocks.<i>.attn_norm.weight`` copy keys still resolve to the same
    # tensors).
    assert flattened[0].attn_norm is vm.blocks[0].attn_norm
    assert flattened[0].ffn_norm is vm.blocks[0].ffn_norm
    assert flattened[2].attn_norm is vm.blocks[1].attn_norm
    assert flattened[2].ffn_norm is vm.blocks[1].ffn_norm

    # Skip-pass block (index 1) has NO RMSNorm modules attached — by design.
    assert flattened[1].use_rms_norm is False
    assert not hasattr(flattened[1], "attn_norm")
    assert not hasattr(flattened[1], "ffn_norm")

    # Forward equivalence still holds with RMSNorm enabled on base blocks.
    x = torch.randn(1, 1, D_MODEL)
    expected = _forward_through_blocks(vm.blocks, x)
    actual = _forward_through_blocks(flattened, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
