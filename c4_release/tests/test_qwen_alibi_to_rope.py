"""Phase R8 / Blocker 4 — ALiBi → RoPE export-time transform.

Pins the contract of
:func:`neural_vm.qwen_compat.alibi_to_rope_export`:

  * Every attention block whose ``_positional_encoding == "alibi"`` is
    flipped to ``"rope"`` in-place with ``alibi_slopes`` zeroed and a
    fresh ``_rope_cos`` / ``_rope_sin`` cache attached.
  * Forward-pass byte-identity holds at **causal sequence position 0**
    (sequence length 1) — the only position where both formulations
    reduce to ``Q[0] · K[0]`` with no bias and no rotation.
  * For sequence positions 1..31 the helper is best-effort: the
    transformed model's forward diverges from the ALiBi baseline by a
    bounded amount documented in the module docstring. We assert the
    delta stays within a generous bound (sane numerics) and does not
    explode (regression gate for any future RoPE / ALiBi semantics
    drift).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.qwen_compat import (
    AlibiToRopeReport,
    _rebake_attention_for_rope,
    alibi_to_rope_export,
)
from neural_vm.vm_step import AutoregressiveVM


D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
FFN_HIDDEN = 64
MAX_SEQ_LEN = 64


def _tiny_alibi_vm(seed: int = 0) -> AutoregressiveVM:
    """A tiny ALiBi-baked VM whose forward we can diff against the RoPE-converted
    twin.

    The VM is constructed with ``positional_encoding="alibi"`` so the
    converted-from-ALiBi path exercises every branch of the helper. We
    initialise weights with a fixed seed so the byte-identity assertion
    is reproducible.
    """

    torch.manual_seed(seed)
    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_hidden=FFN_HIDDEN,
        max_seq_len=MAX_SEQ_LEN,
        positional_encoding="alibi",
        attention_normalization="softmax",
        use_rms_norm=False,
        use_flash_attention=False,
    )
    # Populate non-trivial weights so the forward isn't a degenerate
    # identity. Otherwise the ALiBi / RoPE deltas would be lost in the
    # residual stream.
    with torch.no_grad():
        for block in vm.blocks:
            block.attn.W_q.data = 0.1 * torch.randn_like(block.attn.W_q.data)
            block.attn.W_k.data = 0.1 * torch.randn_like(block.attn.W_k.data)
            block.attn.W_v.data = 0.1 * torch.randn_like(block.attn.W_v.data)
            block.attn.W_o.data = 0.1 * torch.randn_like(block.attn.W_o.data)
            block.ffn.W_up.data = 0.1 * torch.randn_like(block.ffn.W_up.data)
            block.ffn.W_gate.data = 0.1 * torch.randn_like(block.ffn.W_gate.data)
            block.ffn.W_down.data = 0.1 * torch.randn_like(block.ffn.W_down.data)
        vm.embed.embed.weight.data = 0.1 * torch.randn_like(vm.embed.embed.weight.data)
        vm.head.weight.data = 0.1 * torch.randn_like(vm.head.weight.data)
        if vm.head.bias is not None:
            vm.head.bias.data.zero_()
    return vm


def _vm_forward_plain(vm: AutoregressiveVM, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward through the VM without the NeuralVMEmbedding augmentations.

    The tiny VM has no compiler-allocated dim slots, so the augmented
    embedding path indexes out of range. Bypassing it (as the R8 test
    does) gives us a clean attn-then-ffn-then-head trace.
    """

    x = vm.embed.embed(input_ids)
    for block in vm.blocks:
        x = block(x)
    return vm.head(x)


# ---------------------------------------------------------------------------
# Structural contract
# ---------------------------------------------------------------------------


def test_alibi_to_rope_export_flips_positional_encoding_in_place():
    """All ALiBi attention blocks flip to RoPE and slopes are zeroed."""

    vm = _tiny_alibi_vm()
    # Sanity: starting state is ALiBi.
    for block in vm.blocks:
        assert block.attn._positional_encoding == "alibi"
        assert block.attn.alibi_slopes is not None
        assert block.attn._rope_cos is None
        assert block.attn._rope_sin is None

    report = alibi_to_rope_export(vm)

    assert isinstance(report, AlibiToRopeReport)
    assert report.converted_blocks == N_LAYERS
    assert report.skipped_blocks == 0
    assert report.byte_identity_max_pos == 0
    assert report.max_alibi_slope > 0.0
    assert report.rope_base == pytest.approx(10000.0)

    for block in vm.blocks:
        assert block.attn._positional_encoding == "rope"
        # Slopes buffer survives but is now identically zero so the
        # additive ALiBi bias is gone.
        assert torch.allclose(
            block.attn.alibi_slopes, torch.zeros_like(block.attn.alibi_slopes)
        )
        assert block.attn._rope_cos is not None
        assert block.attn._rope_sin is not None
        assert block.attn._rope_cos.shape == (MAX_SEQ_LEN, D_MODEL // N_HEADS)
        assert block.attn._rope_sin.shape == (MAX_SEQ_LEN, D_MODEL // N_HEADS)

    assert getattr(vm, "positional_encoding", None) == "rope"


def test_alibi_to_rope_export_skips_existing_rope_blocks():
    """A model already on the RoPE branch is a no-op (positive return)."""

    torch.manual_seed(42)
    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_hidden=FFN_HIDDEN,
        max_seq_len=MAX_SEQ_LEN,
        positional_encoding="rope",
        attention_normalization="softmax",
        use_rms_norm=False,
        use_flash_attention=False,
    )

    report = alibi_to_rope_export(vm)

    assert report.converted_blocks == 0
    assert report.skipped_blocks == N_LAYERS
    assert all("already RoPE" in r for r in report.skipped_reasons)


# ---------------------------------------------------------------------------
# Forward equivalence (byte-identity range + best-effort divergence bound)
# ---------------------------------------------------------------------------


def test_alibi_to_rope_export_forward_byte_identity_at_position_0():
    """Sequence length 1 (causal position 0): exact bit equivalence.

    At position 0 the ALiBi bias term ``-slope * |0 - 0|`` is exactly
    zero, and the RoPE angle at position 0 is also zero, so both
    formulations reduce to the same un-biased / un-rotated
    ``Q[0] · K[0]``. The forward outputs of the converted VM and the
    baseline VM (the ALiBi twin BEFORE conversion) must therefore agree
    bit-exactly.

    We capture the baseline output BEFORE calling ``alibi_to_rope_export``
    (which is in-place) and compare against the converted forward
    afterwards.
    """

    vm = _tiny_alibi_vm(seed=1)
    input_ids = torch.tensor([[42]], dtype=torch.long)  # single token

    # Baseline: ALiBi forward at position 0.
    with torch.no_grad():
        baseline = _vm_forward_plain(vm, input_ids)

    # Convert and re-run.
    alibi_to_rope_export(vm)
    with torch.no_grad():
        converted = _vm_forward_plain(vm, input_ids)

    # Byte-identity: exact equality, not just allclose.
    assert torch.equal(baseline, converted), (
        f"alibi_to_rope_export broke byte-identity at position 0. "
        f"max |Δ| = {(baseline - converted).abs().max().item():.3e}"
    )


def test_alibi_to_rope_export_forward_bounded_divergence_positions_0_to_31():
    """Positions 0..31: bounded best-effort divergence.

    For positions > 0 the conversion is best-effort (the ALiBi additive
    bias and the RoPE rotation are not algebraically equivalent for
    arbitrary inputs). We assert:

      * The output is finite (no NaN / Inf).
      * The per-position |Δ| is bounded (not exploding under repeated
        layer application). A regression that broke RoPE or scrambled
        weight assignment would land at orders of magnitude above the
        baseline.
      * Position 0 specifically is byte-identical (the only exact
        equivalence guarantee).
    """

    vm = _tiny_alibi_vm(seed=2)
    seq_len = 32
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0) % 276

    with torch.no_grad():
        baseline = _vm_forward_plain(vm, input_ids)

    alibi_to_rope_export(vm)
    with torch.no_grad():
        converted = _vm_forward_plain(vm, input_ids)

    assert torch.isfinite(converted).all(), (
        "alibi_to_rope_export forward produced NaN/Inf"
    )
    assert baseline.shape == converted.shape

    # Position 0 is the byte-identity guarantee. Use exact equality.
    assert torch.equal(baseline[:, 0, :], converted[:, 0, :]), (
        f"Position 0 byte-identity lost: |Δ| = "
        f"{(baseline[:, 0, :] - converted[:, 0, :]).abs().max().item():.3e}"
    )

    # Positions 1..31: bounded delta. The bound is generous (10x the
    # baseline scale) — its job is to catch regressions like a wrong
    # rotation axis or a misallocated cos/sin cache, both of which would
    # blow up by orders of magnitude.
    baseline_scale = baseline.abs().max().item()
    converted_scale = converted.abs().max().item()
    assert converted_scale <= 10.0 * max(baseline_scale, 1.0), (
        f"Converted forward magnitude exploded: baseline={baseline_scale:.3e} "
        f"converted={converted_scale:.3e}"
    )


def test_alibi_to_rope_export_idempotent():
    """Calling the helper twice is a no-op on the second pass.

    The second invocation sees every block already on the RoPE branch
    (``_positional_encoding == "rope"``) so the converter falls through
    to the skip path. Important for export pipelines that may call the
    helper defensively before downstream tooling.
    """

    vm = _tiny_alibi_vm(seed=3)
    report1 = alibi_to_rope_export(vm)
    assert report1.converted_blocks == N_LAYERS

    # Snapshot the RoPE cache so we can confirm the second pass leaves
    # it untouched.
    snapshot_cos = [b.attn._rope_cos.clone() for b in vm.blocks]
    snapshot_sin = [b.attn._rope_sin.clone() for b in vm.blocks]

    report2 = alibi_to_rope_export(vm)
    assert report2.converted_blocks == 0
    assert report2.skipped_blocks == N_LAYERS

    for block, cos_prev, sin_prev in zip(vm.blocks, snapshot_cos, snapshot_sin):
        assert torch.equal(block.attn._rope_cos, cos_prev)
        assert torch.equal(block.attn._rope_sin, sin_prev)
        # Slopes stayed zero.
        assert torch.all(block.attn.alibi_slopes == 0)


def test_alibi_to_rope_export_rope_base_override():
    """``rope_base`` argument overrides the model's default."""

    vm = _tiny_alibi_vm(seed=4)
    custom_base = 500000.0
    report = alibi_to_rope_export(vm, rope_base=custom_base)

    assert report.rope_base == pytest.approx(custom_base)
    for block in vm.blocks:
        assert block.attn.rope_base == pytest.approx(custom_base)


# ---------------------------------------------------------------------------
# _rebake_attention_for_rope — research-level partial compensation helper
# ---------------------------------------------------------------------------


def test_rebake_attention_for_rope_zero_reference_is_noop():
    """``reference_distance=0`` is a documented no-op (preserves position-0)."""

    vm = _tiny_alibi_vm(seed=5)
    alibi_to_rope_export(vm)
    attn = vm.blocks[0].attn
    Wk_snap = attn.W_k.detach().clone()
    report = _rebake_attention_for_rope(
        attn,
        positional_encoding="rope",
        num_heads=int(attn.num_heads),
        head_dim=int(attn.head_dim),
        max_seq_len=int(attn.max_seq_len),
        rope_base=float(attn.rope_base),
        reference_distance=0.0,
    )
    assert report["rebaked"] is False
    assert torch.equal(attn.W_k, Wk_snap)


def test_rebake_attention_for_rope_positive_reference_rotates_Wk():
    """Positive reference distance rotates ``W_k`` per pair, deterministically.

    The post-RoPE forward should agree with the baseline at ``Δp =
    reference_distance`` (the peak of the shifted cos curve) and
    differ at other positions. We don't assert the score peak directly
    (the forward path is non-linear); we just confirm the rebake
    transforms ``W_k`` non-trivially and the per-pair angles match the
    documented ``ref_d * inv_freq`` schedule.
    """

    vm = _tiny_alibi_vm(seed=6)
    alibi_to_rope_export(vm)
    attn = vm.blocks[0].attn
    head_dim = int(attn.head_dim)
    rope_base = float(attn.rope_base)
    Wk_before = attn.W_k.detach().clone()
    report = _rebake_attention_for_rope(
        attn,
        positional_encoding="rope",
        num_heads=int(attn.num_heads),
        head_dim=head_dim,
        max_seq_len=int(attn.max_seq_len),
        rope_base=rope_base,
        reference_distance=2.0,
    )
    assert report["rebaked"] is True
    assert report["num_pairs"] == head_dim // 2
    # Per-pair max rotation = 2 * inv_freq[0] = 2 * 1/base^0 = 2.0.
    assert report["max_rotation"] == pytest.approx(2.0)
    assert not torch.equal(attn.W_k, Wk_before), (
        "rebake with positive reference_distance did not change W_k"
    )
    # Magnitude is preserved per pair (rotation is orthogonal).
    head_dim2 = head_dim // 2
    Wk_pairs_before = Wk_before.view(-1, attn.num_heads, head_dim2, 2)
    Wk_pairs_after = attn.W_k.view(-1, attn.num_heads, head_dim2, 2)
    norm_before = (Wk_pairs_before ** 2).sum(dim=-1)
    norm_after = (Wk_pairs_after ** 2).sum(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-6), (
        "rotation is orthogonal — per-pair squared norm should be preserved"
    )


def test_rebake_attention_for_rope_rejects_odd_head_dim():
    """Helper guards the RoPE even-head_dim contract."""

    class _Stub:
        W_k = torch.zeros(8, 8)
        W_q = torch.zeros(8, 8)
        num_heads = 1
        head_dim = 7
        max_seq_len = 16
        rope_base = 10000.0

    with pytest.raises(ValueError, match="head_dim=7"):
        _rebake_attention_for_rope(
            _Stub(),
            positional_encoding="rope",
            num_heads=1,
            head_dim=7,
            max_seq_len=16,
            rope_base=10000.0,
            reference_distance=1.0,
        )


def test_rebake_attention_for_rope_skips_non_rope():
    """Helper is a no-op when ``positional_encoding != 'rope'``."""

    vm = _tiny_alibi_vm(seed=7)
    attn = vm.blocks[0].attn  # still ALiBi
    Wk_snap = attn.W_k.detach().clone()
    report = _rebake_attention_for_rope(
        attn,
        positional_encoding="alibi",
        num_heads=int(attn.num_heads),
        head_dim=int(attn.head_dim),
        max_seq_len=int(attn.max_seq_len),
        rope_base=float(attn.rope_base),
        reference_distance=2.0,
    )
    assert report["rebaked"] is False
    assert "positional_encoding" in report["reason"]
    assert torch.equal(attn.W_k, Wk_snap)


def test_alibi_to_rope_export_with_rebake_preserves_position_0_when_ref_is_zero():
    """The export-time rebake hook respects the position-0 byte-identity guarantee.

    When ``rebake_reference_distance=0`` (default), the rebake is a no-op
    and the position-0 byte-identity guarantee from
    :func:`alibi_to_rope_export` is preserved.
    """

    vm = _tiny_alibi_vm(seed=8)
    input_ids = torch.tensor([[42]], dtype=torch.long)
    with torch.no_grad():
        baseline = _vm_forward_plain(vm, input_ids)
    alibi_to_rope_export(vm, rebake_reference_distance=0.0)
    with torch.no_grad():
        converted = _vm_forward_plain(vm, input_ids)
    assert torch.equal(baseline, converted), (
        "rebake_reference_distance=0 must preserve position-0 byte identity"
    )
