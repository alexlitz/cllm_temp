"""End-to-end RMSNorm toggle test for the Mixtral export adapter.

Verifies the ``use_rms_norm`` toggle is functional all the way through
the Mixtral export pipeline:

1. Compile a VM (or build a synthetic one when the heavy compile path
   is unavailable) with ``use_rms_norm=True`` and with
   ``use_rms_norm=False``.
2. Randomize each RMSNorm weight on the ``True`` build to a non-identity
   distribution (mean=1, stdev>0) so the assertion has signal.
3. Export both builds via
   ``mixtral_adapter.export_to_mixtral_state_dict(model, pad_to_mixtral=True)``.
4. Assert the RMSNorm build's ``input_layernorm.weight`` /
   ``post_attention_layernorm.weight`` keys carry the VM's actual
   weights (not the synthesized ones the no-rmsnorm path produces).
5. Load both state_dicts into ``MixtralForCausalLM`` (when transformers
   is available) and run a forward pass on each. Assert the outputs
   differ — the toggle should produce a numerically observable change.

This pins the bug the padded-export adapter previously had: it was
unconditionally synthesizing ones for the layer-norm weights regardless
of ``use_rms_norm``, silently dropping the VM's RMSNorm scales.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import pytest
import torch
import torch.nn as nn


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _have_transformers_mixtral() -> bool:
    try:
        import transformers  # noqa: F401
        from transformers import MixtralConfig, MixtralForCausalLM  # noqa: F401
    except Exception:
        return False
    return True


_SKIP_REASON = (
    "transformers (with Mixtral support) is not available in this environment; "
    "install transformers>=4.36 to exercise the Mixtral load/forward path."
)


_SMALL_TARGET: Dict[str, int] = {
    "d_model": 128,
    "n_layers": 4,
    "num_heads": 4,
    "num_kv_heads": 2,  # GQA
    "head_dim": 32,
    "ffn_hidden": 256,
    "vocab_size": 64,
}


def _build_synthetic_vm(use_rms_norm: bool, *, seed: int = 0):
    """Build a small ``AutoregressiveVM`` directly.

    Skips the heavy ``compile_full_vm_dynamic`` bake; the adapter and
    the ``use_rms_norm`` toggle are exercised purely via the VM's
    structural layout. Q/K/V/O are randomized so the forward pass has
    enough signal that the RMSNorm scaling actually changes the output.
    """
    from c4_release.neural_vm.vm_step import AutoregressiveVM

    torch.manual_seed(seed)
    vm = AutoregressiveVM(
        vocab_size=32,
        d_model=64,
        n_layers=3,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=64,
        use_rms_norm=use_rms_norm,
        use_flash_attention=False,
    )

    # Randomize Q/K/V/O + FFN weights so the forward pass isn't a
    # near-zero identity — that's what makes the RMSNorm toggle
    # numerically observable downstream.
    with torch.no_grad():
        for blk in vm.blocks:
            for name in ("W_q", "W_k", "W_v", "W_o"):
                p = getattr(blk.attn, name)
                p.data.copy_(0.02 * torch.randn_like(p.data))
            for name in ("W_up", "W_gate", "W_down"):
                p = getattr(blk.ffn, name)
                p.data.copy_(0.02 * torch.randn_like(p.data))
        vm.embed.embed.weight.data.copy_(
            0.02 * torch.randn_like(vm.embed.embed.weight.data)
        )
        vm.head.weight.data.copy_(0.02 * torch.randn_like(vm.head.weight.data))

    return vm


def _randomize_rms_weights(vm, *, seed: int = 7) -> None:
    """Replace the default ones-init RMSNorm weights with mean=1, std>0 values.

    Without this the test can't tell whether the adapter copied the VM's
    norm weights or synthesized identity ones — both look the same.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for blk in vm.blocks:
            if not getattr(blk, "use_rms_norm", False):
                continue
            attn_w = blk.attn_norm.weight
            ffn_w = blk.ffn_norm.weight
            attn_w.data.copy_(
                torch.ones_like(attn_w.data) + 0.1 * torch.randn_like(attn_w.data)
            )
            ffn_w.data.copy_(
                torch.ones_like(ffn_w.data) + 0.1 * torch.randn_like(ffn_w.data)
            )


def test_adapter_pad_export_copies_vm_rmsnorm_weights():
    """Padded export must copy VM RMSNorm weights when ``use_rms_norm=True``.

    Compares the padded state_dict's layer-norm entries against the VM
    block's actual ``attn_norm.weight`` / ``ffn_norm.weight``. Bug
    surface: prior to the fix the padded path synthesized ones
    unconditionally, so this assertion failed for all layers.
    """
    from c4_release.neural_vm.hf_export.mixtral_adapter import (
        export_to_mixtral_state_dict,
    )

    vm = _build_synthetic_vm(use_rms_norm=True, seed=0)
    _randomize_rms_weights(vm, seed=7)

    # Sanity: at least one block must have non-identity RMSNorm weights,
    # otherwise the rest of the test trivially passes.
    sample_w = vm.blocks[0].attn_norm.weight.data
    assert float(sample_w.std()) > 1e-3, (
        f"randomization failed: std={float(sample_w.std())}"
    )
    assert abs(float(sample_w.mean()) - 1.0) < 0.1

    sd = export_to_mixtral_state_dict(
        vm,
        pad_to_mixtral=True,
        target_shape=_SMALL_TARGET,
        num_local_experts=2,
    )

    t_dmodel = _SMALL_TARGET["d_model"]
    vm_dmodel = vm.d_model
    for i, blk in enumerate(vm.blocks):
        if i >= _SMALL_TARGET["n_layers"]:
            break
        prefix = f"model.layers.{i}"
        in_norm = sd[f"{prefix}.input_layernorm.weight"]
        post_norm = sd[f"{prefix}.post_attention_layernorm.weight"]

        assert in_norm.shape == (t_dmodel,)
        assert post_norm.shape == (t_dmodel,)

        # First vm_dmodel lanes must equal the VM's stored weights.
        assert torch.allclose(
            in_norm[:vm_dmodel].float(),
            blk.attn_norm.weight.data.float(),
            atol=1e-6,
        ), (
            f"layer {i} input_layernorm.weight does not match VM attn_norm "
            f"weight (adapter is likely synthesizing ones)."
        )
        assert torch.allclose(
            post_norm[:vm_dmodel].float(),
            blk.ffn_norm.weight.data.float(),
            atol=1e-6,
        ), (
            f"layer {i} post_attention_layernorm.weight does not match VM "
            f"ffn_norm weight."
        )

        # Unused lanes pad with ones (identity scale on zero-padded acts).
        assert torch.allclose(
            in_norm[vm_dmodel:].float(),
            torch.ones(t_dmodel - vm_dmodel),
            atol=1e-6,
        )

        # The VM's weights are NOT all ones (that's the point of the
        # randomization above), so the copied region's std must be >0.
        assert float(in_norm[:vm_dmodel].std()) > 1e-3, (
            f"layer {i} input_layernorm.weight collapsed to identity — "
            f"adapter is dropping VM's RMSNorm scale."
        )


def test_adapter_pad_export_no_rmsnorm_synthesizes_ones():
    """When ``use_rms_norm=False`` the padded export must emit ones."""
    from c4_release.neural_vm.hf_export.mixtral_adapter import (
        export_to_mixtral_state_dict,
    )

    vm = _build_synthetic_vm(use_rms_norm=False, seed=0)
    assert not getattr(vm.blocks[0], "use_rms_norm", False)
    assert not hasattr(vm.blocks[0], "attn_norm")

    sd = export_to_mixtral_state_dict(
        vm,
        pad_to_mixtral=True,
        target_shape=_SMALL_TARGET,
        num_local_experts=2,
    )

    t_dmodel = _SMALL_TARGET["d_model"]
    for i in range(_SMALL_TARGET["n_layers"]):
        prefix = f"model.layers.{i}"
        in_norm = sd[f"{prefix}.input_layernorm.weight"]
        post_norm = sd[f"{prefix}.post_attention_layernorm.weight"]
        assert torch.allclose(
            in_norm, torch.ones(t_dmodel, dtype=in_norm.dtype)
        ), (
            f"layer {i} input_layernorm should be all-ones when "
            f"use_rms_norm=False, got mean={float(in_norm.mean())}"
        )
        assert torch.allclose(
            post_norm, torch.ones(t_dmodel, dtype=post_norm.dtype)
        )


def _build_mixtral_config_kwargs_for_target(
    model, target: Dict[str, int], num_local_experts: int, num_experts_per_tok: int
):
    return {
        "vocab_size": int(target["vocab_size"]),
        "hidden_size": int(target["d_model"]),
        "intermediate_size": int(target["ffn_hidden"]),
        "num_hidden_layers": int(target["n_layers"]),
        "num_attention_heads": int(target["num_heads"]),
        "num_key_value_heads": int(target.get("num_kv_heads", target["num_heads"])),
        "head_dim": int(target["head_dim"]),
        "max_position_embeddings": int(getattr(model, "max_seq_len", 1024)),
        "rms_norm_eps": float(getattr(model, "rms_norm_eps", 1e-6)),
        "rope_theta": float(getattr(model, "rope_base", 10000.0)),
        "tie_word_embeddings": False,
        "num_local_experts": num_local_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "sliding_window": None,
    }


@pytest.mark.skipif(not _have_transformers_mixtral(), reason=_SKIP_REASON)
def test_mixtral_rmsnorm_toggle_changes_forward_output():
    """Toggle ``use_rms_norm`` on/off and assert HF Mixtral logits differ.

    Builds the same synthetic VM twice (rmsnorm on, rmsnorm off) with
    identical attention/FFN/embed weights, exports each via
    ``pad_to_mixtral=True``, loads each into ``MixtralForCausalLM``, and
    asserts the two forward passes produce different logits. This is the
    end-to-end functional check that the RMSNorm toggle survives the
    compile -> export -> HF-load pipeline.
    """
    from transformers import MixtralConfig, MixtralForCausalLM

    from c4_release.neural_vm.hf_export.mixtral_adapter import (
        export_to_mixtral_state_dict,
    )

    # Build BOTH models from the same seed so attention/FFN/embed
    # weights match — only the RMSNorm toggle differs.
    vm_off = _build_synthetic_vm(use_rms_norm=False, seed=42)
    vm_on = _build_synthetic_vm(use_rms_norm=True, seed=42)
    _randomize_rms_weights(vm_on, seed=7)

    # Sanity: attention/FFN weights are identical (same seed).
    for blk_off, blk_on in zip(vm_off.blocks, vm_on.blocks):
        assert torch.allclose(blk_off.attn.W_q.data, blk_on.attn.W_q.data)
        assert torch.allclose(blk_off.ffn.W_up.data, blk_on.ffn.W_up.data)

    sd_off = export_to_mixtral_state_dict(
        vm_off, pad_to_mixtral=True, target_shape=_SMALL_TARGET,
        num_local_experts=2,
    )
    sd_on = export_to_mixtral_state_dict(
        vm_on, pad_to_mixtral=True, target_shape=_SMALL_TARGET,
        num_local_experts=2,
    )

    # The layer-norm keys must DIFFER between the two state_dicts —
    # this is what the adapter fix establishes.
    diff_found = False
    for i in range(_SMALL_TARGET["n_layers"]):
        k = f"model.layers.{i}.input_layernorm.weight"
        if not torch.allclose(sd_off[k], sd_on[k]):
            diff_found = True
            break
    assert diff_found, (
        "input_layernorm.weight is identical between rmsnorm=on/off "
        "exports — the adapter is not propagating the VM's RMSNorm scale."
    )

    # The non-norm weights must be identical (same source VM seed).
    same_key = "model.layers.0.self_attn.q_proj.weight"
    assert torch.allclose(sd_off[same_key], sd_on[same_key]), (
        "q_proj differs between rmsnorm=on/off exports — the test setup "
        "isn't isolating the RMSNorm toggle as a single-variable change."
    )

    # The norm weights on the on-path must NOT be all-ones.
    on_norm = sd_on["model.layers.0.input_layernorm.weight"]
    assert not torch.allclose(on_norm, torch.ones_like(on_norm)), (
        "rmsnorm=on export still emits all-ones input_layernorm.weight."
    )

    # Build HF models + load + forward on each.
    cfg_kwargs = _build_mixtral_config_kwargs_for_target(
        vm_off, _SMALL_TARGET, num_local_experts=2, num_experts_per_tok=1
    )
    config = MixtralConfig(**cfg_kwargs)

    def _load_and_forward(sd):
        hf = MixtralForCausalLM(config)
        target_dtype = next(hf.parameters()).dtype
        cast = {k: v.to(dtype=target_dtype) for k, v in sd.items()}
        missing, unexpected = hf.load_state_dict(cast, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        hf.eval()
        with torch.no_grad():
            out = hf(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
        return out.logits

    torch.manual_seed(0)
    logits_off = _load_and_forward(sd_off)
    torch.manual_seed(0)
    logits_on = _load_and_forward(sd_on)

    assert logits_off.shape == logits_on.shape == (1, 3, 64)
    assert torch.isfinite(logits_off).all()
    assert torch.isfinite(logits_on).all()

    # The whole point: RMSNorm changes Mixtral's behavior. If the toggle
    # is wired functionally end-to-end through export, the two HF
    # forward passes MUST disagree on at least one logit.
    #
    # Note: MixtralForCausalLM applies its own ``model.norm`` (final
    # RMSNorm) on both paths with weight=1.0, so the difference comes
    # purely from per-layer ``input_layernorm`` /
    # ``post_attention_layernorm`` weights.
    assert not torch.allclose(logits_off, logits_on, atol=1e-4), (
        "Mixtral logits identical between rmsnorm=on/off — the RMSNorm "
        "toggle is not actually changing behaviour end-to-end."
    )


def test_compiled_vm_rmsnorm_export_carries_actual_weights():
    """``compile_full_vm_dynamic(use_rms_norm=True)`` -> pad-export.

    The full compile path produces ~35 blocks (17 base + 14 post-op
    wrapper expansions + a few re-baked vanilla PureFFNs). Only the
    base blocks have ``use_rms_norm=True``; the wrapper passthrough
    blocks created by ``_expand_wrapper_blocks`` use the default
    ``use_rms_norm=False`` for their passthrough attn+ffn.

    We assert at least one block in the padded state_dict carries
    non-identity ``input_layernorm.weight`` and that those weights
    match the source VM block's ``attn_norm.weight``.

    Self-skips when ``compile_full_vm_dynamic(use_rms_norm=True)`` fails
    on an unrelated upstream invariant; the synthetic-VM tests above
    cover the adapter regardless.
    """
    from c4_release.neural_vm.hf_export.mixtral_adapter import (
        export_to_mixtral_state_dict,
    )
    try:
        from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
            compile_full_vm_dynamic,
        )
        model, _layout = compile_full_vm_dynamic(
            disk_cache=True,
            strict=False,
            declarations_only=True,
            use_rms_norm=True,
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(
            f"compile_full_vm_dynamic(use_rms_norm=True) unavailable: {exc!r}"
        )

    assert getattr(model, "use_rms_norm", False)
    n_rms_blocks = sum(
        1 for b in model.blocks if getattr(b, "use_rms_norm", False)
    )
    assert n_rms_blocks > 0

    # Force a non-identity scale on the RMSNorm weights so the export
    # assertion has a real signal.
    torch.manual_seed(11)
    with torch.no_grad():
        for blk in model.blocks:
            if not getattr(blk, "use_rms_norm", False):
                continue
            for norm_name in ("attn_norm", "ffn_norm"):
                w = getattr(blk, norm_name).weight
                w.data.copy_(
                    torch.ones_like(w.data) + 0.05 * torch.randn_like(w.data)
                )

    sd = export_to_mixtral_state_dict(model, pad_to_mixtral=True)

    # Find a block with use_rms_norm=True whose index is within the
    # Mixtral-8x7B target's 32 layers.
    target_n_layers = 32
    target_dmodel = 4096
    vm_dmodel = model.d_model
    matched = 0
    for i, blk in enumerate(model.blocks):
        if i >= target_n_layers:
            break
        if not getattr(blk, "use_rms_norm", False):
            continue
        k_in = f"model.layers.{i}.input_layernorm.weight"
        k_post = f"model.layers.{i}.post_attention_layernorm.weight"
        in_norm = sd[k_in]
        post_norm = sd[k_post]
        assert in_norm.shape == (target_dmodel,)
        assert torch.allclose(
            in_norm[:vm_dmodel].float(),
            blk.attn_norm.weight.data.float(),
            atol=1e-6,
        )
        assert torch.allclose(
            post_norm[:vm_dmodel].float(),
            blk.ffn_norm.weight.data.float(),
            atol=1e-6,
        )
        assert float(in_norm[:vm_dmodel].std()) > 1e-3
        matched += 1

    assert matched > 0, (
        "No use_rms_norm blocks within the Mixtral-8x7B 32-layer window "
        "had their weights copied through; the adapter dropped them."
    )
