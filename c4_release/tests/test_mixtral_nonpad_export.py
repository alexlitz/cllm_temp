"""Tests for the non-padded Mixtral export path (``pad_to_mixtral=False``).

The sibling ``test_mixtral_rmsnorm_e2e.py`` covers the padded path
(``pad_to_mixtral=True``) where VM RMSNorm weights flow through
``_export_padded_state_dict``. This file pins the *non-padded* branch in
:func:`export_to_mixtral_state_dict`:

* tensor shapes follow the VM's natural geometry (no zero-fill to a
  larger Mixtral target),
* when the VM block has ``use_rms_norm=True``, the produced
  ``input_layernorm.weight`` / ``post_attention_layernorm.weight`` carry
  the VM's actual RMSNorm weights (the bug the padded path had:
  commit 18c55589).

Small synthetic Mixtral-shaped VM (d_model=64, 2 layers, 4 heads,
head_dim=16, ffn_hidden=128) keeps the suite CPU-fast.
"""

from __future__ import annotations

import os
import sys

import torch

# Make the package importable when tests are run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from neural_vm.hf_export.mixtral_adapter import (  # noqa: E402
    export_to_mixtral_state_dict,
)


def _build_synthetic_vm(
    *,
    vocab_size: int = 32,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    ffn_hidden: int = 128,
    max_seq_len: int = 64,
    use_rms_norm: bool = False,
):
    """Build a small Mixtral-shaped VM, optionally with RMSNorm enabled."""

    from neural_vm.vm_step import AutoregressiveVM

    vm = AutoregressiveVM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
        use_rms_norm=use_rms_norm,
    )

    # Seed parameters with deterministic non-zero values so we can
    # distinguish "copied" from "synthesized as ones" for the layer norms.
    g = torch.Generator()
    g.manual_seed(0xC4DEBEEF)
    with torch.no_grad():
        for p in vm.parameters():
            p.data.normal_(generator=g)

    return vm


def test_nonpad_state_dict_uses_vm_natural_shapes():
    """``pad_to_mixtral=False`` produces tensors at the VM's geometry.

    Crucially this is NOT the canonical Mixtral-8x7B envelope
    (d_model=4096, ffn_hidden=14336, ...). The non-padded path must not
    pad up to that target — it should emit tensors sized to the VM
    itself.
    """

    vm = _build_synthetic_vm()
    num_experts = 2
    sd = export_to_mixtral_state_dict(
        vm, num_local_experts=num_experts, pad_to_mixtral=False
    )

    # Embedding / head / final norm sized to VM, not to Mixtral-8x7B
    # (vocab=32000, d_model=4096).
    assert sd["model.embed_tokens.weight"].shape == (32, 64)
    assert sd["lm_head.weight"].shape == (32, 64)
    assert sd["model.norm.weight"].shape == (64,)

    for i in range(2):
        prefix = f"model.layers.{i}"
        # Attention projections at VM-natural d_model (not padded to 4096).
        assert sd[f"{prefix}.self_attn.q_proj.weight"].shape == (64, 64)
        assert sd[f"{prefix}.self_attn.k_proj.weight"].shape == (64, 64)
        assert sd[f"{prefix}.self_attn.v_proj.weight"].shape == (64, 64)
        assert sd[f"{prefix}.self_attn.o_proj.weight"].shape == (64, 64)
        # FFN at VM-natural ffn_hidden=128 (not padded to 14336).
        for e in range(num_experts):
            assert sd[
                f"{prefix}.block_sparse_moe.experts.{e}.w1.weight"
            ].shape == (128, 64)
            assert sd[
                f"{prefix}.block_sparse_moe.experts.{e}.w2.weight"
            ].shape == (64, 128)
            assert sd[
                f"{prefix}.block_sparse_moe.experts.{e}.w3.weight"
            ].shape == (128, 64)
        # Layer norms at VM-natural d_model.
        assert sd[f"{prefix}.input_layernorm.weight"].shape == (64,)
        assert sd[f"{prefix}.post_attention_layernorm.weight"].shape == (64,)
        # Router gate at VM-natural d_model.
        assert sd[
            f"{prefix}.block_sparse_moe.gate.weight"
        ].shape == (num_experts, 64)

    # And the explicit non-padding contract: tensors must match VM
    # weight data byte-for-byte (the non-padded export is a clone, not a
    # zero-fill at the top-left corner).
    assert torch.equal(
        sd["model.embed_tokens.weight"], vm.embed.embed.weight.data
    )
    assert torch.equal(sd["lm_head.weight"], vm.head.weight.data)


def test_nonpad_rmsnorm_on_carries_vm_weights():
    """RMSNorm-on branch must copy VM ``attn_norm``/``ffn_norm`` weights.

    This is the same regression the padded path had (commit 18c55589):
    the adapter previously synthesized ones for the layer norms even
    when the VM block was RMSNorm-enabled. The non-padded branch sits
    at ``mixtral_adapter.py:363-371``.
    """

    vm = _build_synthetic_vm(use_rms_norm=True)

    # Sanity: the VM blocks actually have RMSNorm weights to copy.
    assert getattr(vm.blocks[0], "use_rms_norm", False) is True
    attn_norm_w = vm.blocks[0].attn_norm.weight.data
    ffn_norm_w = vm.blocks[0].ffn_norm.weight.data
    # Random-init values, definitely not the synthesized ones-vector.
    assert not torch.allclose(attn_norm_w, torch.ones_like(attn_norm_w))
    assert not torch.allclose(ffn_norm_w, torch.ones_like(ffn_norm_w))

    sd = export_to_mixtral_state_dict(
        vm, num_local_experts=2, pad_to_mixtral=False
    )

    for i in range(len(vm.blocks)):
        prefix = f"model.layers.{i}"
        # VM attn_norm → Mixtral input_layernorm.
        assert torch.equal(
            sd[f"{prefix}.input_layernorm.weight"],
            vm.blocks[i].attn_norm.weight.data,
        ), (
            f"Layer {i} input_layernorm.weight does not carry the VM's "
            "attn_norm.weight. This is the bug the padded path had — "
            "see commit 18c55589."
        )
        # VM ffn_norm → Mixtral post_attention_layernorm.
        assert torch.equal(
            sd[f"{prefix}.post_attention_layernorm.weight"],
            vm.blocks[i].ffn_norm.weight.data,
        ), (
            f"Layer {i} post_attention_layernorm.weight does not carry "
            "the VM's ffn_norm.weight."
        )
        # Explicit negative check: must not be synthesized ones.
        assert not torch.allclose(
            sd[f"{prefix}.input_layernorm.weight"], torch.ones(vm.d_model)
        )
        assert not torch.allclose(
            sd[f"{prefix}.post_attention_layernorm.weight"],
            torch.ones(vm.d_model),
        )


def test_nonpad_rmsnorm_off_synthesizes_ones():
    """RMSNorm-off branch keeps the historic ones-fill (identity scale).

    Complements the rmsnorm-on test: when the VM has no RMSNorm, the
    layer-norm slots should be synthesized as ones so the adapted
    Mixtral runs as identity-scale-norm.
    """

    vm = _build_synthetic_vm(use_rms_norm=False)
    assert getattr(vm.blocks[0], "use_rms_norm", False) is False

    sd = export_to_mixtral_state_dict(
        vm, num_local_experts=2, pad_to_mixtral=False
    )

    ones = torch.ones(vm.d_model, dtype=sd["model.embed_tokens.weight"].dtype)
    for i in range(len(vm.blocks)):
        prefix = f"model.layers.{i}"
        assert torch.allclose(sd[f"{prefix}.input_layernorm.weight"], ones)
        assert torch.allclose(
            sd[f"{prefix}.post_attention_layernorm.weight"], ones
        )
