"""Tests for the Mixtral HF export adapter.

These tests use a small synthetic Mixtral-shaped VM (d_model=64, 2 layers,
4 heads, head_dim=16, ffn_hidden=128) so they run on CPU in seconds. The
goal is to prove the adapter's key/shape contract is correct and the
resulting HF model is runnable end-to-end (forward pass on a dummy token
does not crash).

If ``transformers`` (or Mixtral within it) isn't installed in the test
environment, the loader/forward tests are skipped with a clear message;
the pure adapter-side tests still run because they don't touch the HF
classes.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pytest
import torch

# Make the package importable when tests are run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from neural_vm.hf_export.mixtral_adapter import (  # noqa: E402
    MixtralShapeMismatchError,
    export_to_mixtral_state_dict,
    infer_mixtral_config_kwargs,
    load_into_mixtral,
)


def _have_transformers_mixtral() -> bool:
    try:
        import transformers  # noqa: F401
        from transformers import MixtralConfig, MixtralForCausalLM  # noqa: F401
    except Exception:
        return False
    return True


_TRANSFORMERS_SKIP_REASON = (
    "transformers (with Mixtral support) is not available in this environment; "
    "install transformers>=4.36 to exercise the load_into_mixtral / forward paths."
)


def _build_synthetic_vm(
    vocab_size: int = 32,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    ffn_hidden: int = 128,
    max_seq_len: int = 64,
):
    """Build a small Mixtral-shaped VM.

    ``num_heads * head_dim == d_model`` (4 * 16 == 64) and the FFN width
    is constant across layers — both required by the Mixtral adapter.
    """

    from neural_vm.vm_step import AutoregressiveVM

    vm = AutoregressiveVM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
    )

    # Seed the parameters with deterministic non-zero values so the
    # adapter has something distinguishable to copy through. The VM
    # zero-inits weights pre-bake; that's fine for shape testing, but
    # giving non-zero values lets us check that copies (not just shapes)
    # land in the right slots.
    g = torch.Generator()
    g.manual_seed(0xC4DEBEEF)
    with torch.no_grad():
        for p in vm.parameters():
            p.data.normal_(generator=g)

    return vm


def test_infer_mixtral_config_kwargs_basic():
    vm = _build_synthetic_vm()
    kwargs = infer_mixtral_config_kwargs(vm)

    assert kwargs["vocab_size"] == 32
    assert kwargs["hidden_size"] == 64
    assert kwargs["intermediate_size"] == 128
    assert kwargs["num_hidden_layers"] == 2
    assert kwargs["num_attention_heads"] == 4
    # The VM is MHA — every Q head has its own K/V — so kv heads == q heads.
    assert kwargs["num_key_value_heads"] == 4
    assert kwargs["head_dim"] == 16
    assert kwargs["max_position_embeddings"] == 64
    # No sliding-window — VM is fully causal MHA.
    assert kwargs["sliding_window"] is None


def test_export_to_mixtral_state_dict_keys_and_shapes():
    vm = _build_synthetic_vm()
    num_experts = 2
    sd = export_to_mixtral_state_dict(vm, num_local_experts=num_experts)

    # Embeddings + lm_head + final norm
    assert sd["model.embed_tokens.weight"].shape == (32, 64)
    assert sd["lm_head.weight"].shape == (32, 64)
    assert sd["model.norm.weight"].shape == (64,)
    # Final norm is synthesized as ones (VM has no final norm).
    assert torch.allclose(sd["model.norm.weight"], torch.ones(64))

    # Per-layer keys exist with the right shapes.
    for i in range(2):
        prefix = f"model.layers.{i}"
        assert sd[f"{prefix}.self_attn.q_proj.weight"].shape == (64, 64)
        assert sd[f"{prefix}.self_attn.k_proj.weight"].shape == (64, 64)
        assert sd[f"{prefix}.self_attn.v_proj.weight"].shape == (64, 64)
        assert sd[f"{prefix}.self_attn.o_proj.weight"].shape == (64, 64)
        assert sd[f"{prefix}.block_sparse_moe.gate.weight"].shape == (num_experts, 64)
        # Gate is zero (uniform routing at init).
        assert torch.allclose(
            sd[f"{prefix}.block_sparse_moe.gate.weight"], torch.zeros(num_experts, 64)
        )
        # input/post-attention layer norms exist (synthesized ones when VM
        # has no RMSNorm).
        assert sd[f"{prefix}.input_layernorm.weight"].shape == (64,)
        assert sd[f"{prefix}.post_attention_layernorm.weight"].shape == (64,)
        # Experts: every expert sees w1/w2/w3 of the right shapes.
        for e in range(num_experts):
            assert sd[f"{prefix}.block_sparse_moe.experts.{e}.w1.weight"].shape == (128, 64)
            assert sd[f"{prefix}.block_sparse_moe.experts.{e}.w2.weight"].shape == (64, 128)
            assert sd[f"{prefix}.block_sparse_moe.experts.{e}.w3.weight"].shape == (128, 64)


def test_export_replicates_ffn_across_experts():
    """Every expert in the MoE should be a copy of the VM's dense FFN."""

    vm = _build_synthetic_vm()
    num_experts = 3
    sd = export_to_mixtral_state_dict(vm, num_local_experts=num_experts)

    for i in range(2):
        prefix = f"model.layers.{i}.block_sparse_moe.experts"
        # All w1 weights should be identical across experts (replicated VM gate).
        w1_0 = sd[f"{prefix}.0.w1.weight"]
        # And they should match the VM's W_gate exactly.
        vm_gate = vm.blocks[i].ffn.W_gate.data
        assert torch.equal(w1_0, vm_gate)
        for e in range(1, num_experts):
            assert torch.equal(sd[f"{prefix}.{e}.w1.weight"], w1_0)
            assert torch.equal(
                sd[f"{prefix}.{e}.w2.weight"], sd[f"{prefix}.0.w2.weight"]
            )
            assert torch.equal(
                sd[f"{prefix}.{e}.w3.weight"], sd[f"{prefix}.0.w3.weight"]
            )

        # VM W_up → expert w3, W_down → expert w2.
        assert torch.equal(sd[f"{prefix}.0.w3.weight"], vm.blocks[i].ffn.W_up.data)
        assert torch.equal(sd[f"{prefix}.0.w2.weight"], vm.blocks[i].ffn.W_down.data)


def test_attention_projections_copy_through():
    vm = _build_synthetic_vm()
    sd = export_to_mixtral_state_dict(vm, num_local_experts=2)

    for i in range(2):
        assert torch.equal(
            sd[f"model.layers.{i}.self_attn.q_proj.weight"],
            vm.blocks[i].attn.W_q.data,
        )
        assert torch.equal(
            sd[f"model.layers.{i}.self_attn.k_proj.weight"],
            vm.blocks[i].attn.W_k.data,
        )
        assert torch.equal(
            sd[f"model.layers.{i}.self_attn.v_proj.weight"],
            vm.blocks[i].attn.W_v.data,
        )
        assert torch.equal(
            sd[f"model.layers.{i}.self_attn.o_proj.weight"],
            vm.blocks[i].attn.W_o.data,
        )


def test_embedding_and_head_copy_through():
    vm = _build_synthetic_vm()
    sd = export_to_mixtral_state_dict(vm, num_local_experts=2)
    assert torch.equal(sd["model.embed_tokens.weight"], vm.embed.embed.weight.data)
    assert torch.equal(sd["lm_head.weight"], vm.head.weight.data)


def test_per_layer_ffn_width_mismatch_raises():
    """Mixtral has one intermediate_size; per-layer widths should fail loud."""

    vm = _build_synthetic_vm()
    # Surgically resize layer 1's FFN to a different width.
    import torch.nn as nn

    new_hidden = 96
    vm.blocks[1].ffn.W_up = nn.Parameter(torch.zeros(new_hidden, vm.d_model))
    vm.blocks[1].ffn.W_gate = nn.Parameter(torch.zeros(new_hidden, vm.d_model))
    vm.blocks[1].ffn.W_down = nn.Parameter(torch.zeros(vm.d_model, new_hidden))
    vm.blocks[1].ffn.b_up = nn.Parameter(torch.zeros(new_hidden))
    vm.blocks[1].ffn.b_gate = nn.Parameter(torch.zeros(new_hidden))
    vm.blocks[1].ffn.hidden_dim = new_hidden

    with pytest.raises(MixtralShapeMismatchError, match="intermediate_size"):
        export_to_mixtral_state_dict(vm, num_local_experts=2)


def test_head_dim_mismatch_raises():
    """When num_heads * head_dim != d_model, the export must fail loud."""

    vm = _build_synthetic_vm()
    # Force an invalid head_dim that doesn't tile d_model.
    vm.blocks[0].attn.head_dim = 17  # 4 * 17 = 68 != 64

    with pytest.raises(MixtralShapeMismatchError, match="num_heads"):
        infer_mixtral_config_kwargs(vm)


# ----------------------------------------------------------------------------
# HF-runtime tests (skipped if transformers/Mixtral is missing)
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_transformers_mixtral(), reason=_TRANSFORMERS_SKIP_REASON
)
def test_load_into_mixtral_builds_runnable_model():
    """The adapter should produce an HF model that loads strict=True-ish.

    ``load_into_mixtral`` is intentionally strict: any key it produces that
    Mixtral doesn't consume, or any Mixtral parameter it forgets to
    populate, raises RuntimeError. That guards against the adapter
    drifting from the installed transformers version.
    """

    from transformers import MixtralConfig, MixtralForCausalLM

    vm = _build_synthetic_vm()
    cfg_kwargs = infer_mixtral_config_kwargs(vm)
    cfg_kwargs["num_local_experts"] = 2
    cfg_kwargs["num_experts_per_tok"] = 1
    cfg = MixtralConfig(**cfg_kwargs)

    hf_model = load_into_mixtral(vm, config=cfg)
    assert isinstance(hf_model, MixtralForCausalLM)
    # Sanity: parameters are populated, not freshly random.
    embed_weight = hf_model.model.embed_tokens.weight.data
    assert torch.allclose(
        embed_weight.float(), vm.embed.embed.weight.data.float(), atol=1e-5
    )


@pytest.mark.skipif(
    not _have_transformers_mixtral(), reason=_TRANSFORMERS_SKIP_REASON
)
def test_forward_pass_does_not_crash():
    """One dummy token through the adapted model should not crash."""

    from transformers import MixtralConfig

    vm = _build_synthetic_vm()
    cfg_kwargs = infer_mixtral_config_kwargs(vm)
    cfg_kwargs["num_local_experts"] = 2
    cfg_kwargs["num_experts_per_tok"] = 1
    cfg = MixtralConfig(**cfg_kwargs)

    hf_model = load_into_mixtral(vm, config=cfg)
    hf_model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([[1]], dtype=torch.long)
        out = hf_model(input_ids=input_ids)

    assert out.logits.shape == (1, 1, 32)
    assert torch.isfinite(out.logits).all(), "Forward produced NaN/inf logits."


@pytest.mark.skipif(
    not _have_transformers_mixtral(), reason=_TRANSFORMERS_SKIP_REASON
)
def test_forward_pass_multi_token_does_not_crash():
    """Multi-token forward pass should also run cleanly."""

    from transformers import MixtralConfig

    vm = _build_synthetic_vm()
    cfg_kwargs = infer_mixtral_config_kwargs(vm)
    cfg_kwargs["num_local_experts"] = 2
    cfg_kwargs["num_experts_per_tok"] = 1
    cfg = MixtralConfig(**cfg_kwargs)

    hf_model = load_into_mixtral(vm, config=cfg)
    hf_model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        out = hf_model(input_ids=input_ids)

    assert out.logits.shape == (1, 5, 32)
    assert torch.isfinite(out.logits).all()
