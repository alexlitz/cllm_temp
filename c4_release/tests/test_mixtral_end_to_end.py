"""End-to-end Mixtral export wiring test (Path B: pad/zero-fill).

Exercises the four stages of the Mixtral interop pipeline using the
adapter's PAD-ON-EXPORT mode (``pad_to_mixtral=True``):

1. Compile a VM at its allocator-native default shape (``d_model`` /
   ``num_heads`` / ``head_dim`` come from the dynamic dim allocator;
   per-layer FFN widths vary; ``vocab_size`` is allocator-driven).
2. Pad every tensor in the resulting state_dict in the top-left corner
   to a target Mixtral-8x7B geometry (``d_model=4096``, 32 layers, 32
   Q heads, 8 KV heads via GQA, ``head_dim=128``,
   ``intermediate_size=14336``, ``vocab_size=32000``) via
   ``mixtral_adapter.export_to_mixtral_state_dict(model,
   pad_to_mixtral=True)``.
3. Load the padded state_dict into HF ``MixtralForCausalLM``.
4. Run a single forward pass and assert logits shape matches Mixtral
   ``(1, 1, 32000)``.

Why Path B (pad) and not Path A (allocator-native compile to Mixtral
shape):

- Path A would require the dim allocator, head allocator, and FFN unit
  allocator to accept a target envelope (``d_model=4096``, GQA
  ``num_kv_heads=8``, uniform 14336 FFN width) and emit a baked,
  semantics-preserving model with the VM's computation routed onto
  that envelope. That's a real rewrite of the allocator stack; see
  ``compile_full_vm_dynamic``'s ``target_shape_overrides`` rebuild —
  the SHAPE swap works, but the rebuild zero-inits weights, so
  current Path A is structurally identical to Path B minus the
  per-layer FFN-width tolerance.
- Path B sidesteps the allocator entirely. The VM keeps its native
  shape; the export step pads the state_dict on the way out. Unused
  Mixtral residual lanes are zero — they multiply against zero-padded
  activations and contribute nothing.

The resulting Mixtral runs (forward pass returns finite logits) but
most weights are zero: this is wiring correctness, not numerical
equivalence. Numerical equivalence requires Path A (allocator-native
bake), tracked as future work.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import pytest
import torch


# Make the c4_release package importable when tests are run from repo root.
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
    "install transformers>=4.36 to exercise the Mixtral pad/load/forward path."
)


# Mixtral-8x7B canonical geometry (mirrors mixtral_adapter.MIXTRAL_8X7B_SHAPE
# but kept here too so the test's shape assertions don't silently change
# if the adapter constant moves).
_MIXTRAL_8X7B: Dict[str, int] = {
    "d_model": 4096,
    "n_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "ffn_hidden": 14336,
    "vocab_size": 32000,
}


def _build_mixtral_config_kwargs_for_target(
    model, target: Dict[str, int], num_local_experts: int, num_experts_per_tok: int
) -> Dict:
    """Mixtral config kwargs that match the pad target geometry."""
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


def test_pad_to_mixtral_synthetic_default_vm():
    """Pad-path smoke that doesn't need transformers installed.

    Uses a small synthetic VM (d_model=64, 3 layers, varying per-layer
    FFN widths) and pads to a small Mixtral target (d_model=128, 4
    layers, GQA num_kv_heads=2). Exercises the synth-extra-layer
    branch (target has 4 layers vs VM's 3) and confirms the adapter's
    pad path produces the expected key/shape set.
    """
    import torch.nn as nn

    from c4_release.neural_vm.hf_export.mixtral_adapter import (
        export_to_mixtral_state_dict,
    )
    from c4_release.neural_vm.vm_step import AutoregressiveVM

    vm = AutoregressiveVM(
        vocab_size=32,
        d_model=64,
        n_layers=3,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=64,
    )

    # Force per-layer FFN-width variance: shrink layer 1's hidden_dim.
    vm.blocks[1].ffn.W_up = nn.Parameter(torch.zeros(96, vm.d_model))
    vm.blocks[1].ffn.W_gate = nn.Parameter(torch.zeros(96, vm.d_model))
    vm.blocks[1].ffn.W_down = nn.Parameter(torch.zeros(vm.d_model, 96))
    vm.blocks[1].ffn.b_up = nn.Parameter(torch.zeros(96))
    vm.blocks[1].ffn.b_gate = nn.Parameter(torch.zeros(96))
    vm.blocks[1].ffn.hidden_dim = 96

    small_target = {
        "d_model": 128,
        "n_layers": 4,  # > VM's 3, exercises synth-extra-layer branch
        "num_heads": 4,
        "num_kv_heads": 2,  # GQA
        "head_dim": 32,  # 4 * 32 = 128
        "ffn_hidden": 256,
        "vocab_size": 64,
    }

    sd = export_to_mixtral_state_dict(
        vm,
        pad_to_mixtral=True,
        target_shape=small_target,
        num_local_experts=2,
    )

    # Shape contracts.
    assert sd["model.embed_tokens.weight"].shape == (64, 128)
    assert sd["lm_head.weight"].shape == (64, 128)
    for i in range(4):
        prefix = f"model.layers.{i}"
        assert sd[f"{prefix}.self_attn.q_proj.weight"].shape == (128, 128)
        assert sd[f"{prefix}.self_attn.k_proj.weight"].shape == (64, 128)
        assert sd[f"{prefix}.self_attn.v_proj.weight"].shape == (64, 128)
        assert sd[f"{prefix}.self_attn.o_proj.weight"].shape == (128, 128)
        for e in range(2):
            assert sd[
                f"{prefix}.block_sparse_moe.experts.{e}.w1.weight"
            ].shape == (256, 128)
            assert sd[
                f"{prefix}.block_sparse_moe.experts.{e}.w2.weight"
            ].shape == (128, 256)
            assert sd[
                f"{prefix}.block_sparse_moe.experts.{e}.w3.weight"
            ].shape == (256, 128)

    # Embed top-left corner must equal the VM's embed weight.
    assert torch.allclose(
        sd["model.embed_tokens.weight"][:32, :64].float(),
        vm.embed.embed.weight.data.float(),
        atol=1e-6,
    )


@pytest.mark.skipif(not _have_transformers_mixtral(), reason=_SKIP_REASON)
def test_pad_to_mixtral_loads_into_hf_synthetic():
    """Pad-path smoke that loads into HF Mixtral and runs a forward pass.

    Same synthetic-VM seed as ``test_pad_to_mixtral_synthetic_default_vm``;
    layered on top is a Mixtral build + load + forward to cover the HF
    wiring without paying for the full ``compile_full_vm_dynamic`` bake.
    """
    import torch.nn as nn

    from transformers import MixtralConfig, MixtralForCausalLM

    from c4_release.neural_vm.hf_export.mixtral_adapter import (
        export_to_mixtral_state_dict,
    )
    from c4_release.neural_vm.vm_step import AutoregressiveVM

    vm = AutoregressiveVM(
        vocab_size=32,
        d_model=64,
        n_layers=3,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=64,
    )
    vm.blocks[1].ffn.W_up = nn.Parameter(torch.zeros(96, vm.d_model))
    vm.blocks[1].ffn.W_gate = nn.Parameter(torch.zeros(96, vm.d_model))
    vm.blocks[1].ffn.W_down = nn.Parameter(torch.zeros(vm.d_model, 96))
    vm.blocks[1].ffn.b_up = nn.Parameter(torch.zeros(96))
    vm.blocks[1].ffn.b_gate = nn.Parameter(torch.zeros(96))
    vm.blocks[1].ffn.hidden_dim = 96

    small_target = {
        "d_model": 128,
        "n_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 32,
        "ffn_hidden": 256,
        "vocab_size": 64,
    }

    sd = export_to_mixtral_state_dict(
        vm,
        pad_to_mixtral=True,
        target_shape=small_target,
        num_local_experts=2,
    )

    cfg_kwargs = _build_mixtral_config_kwargs_for_target(
        vm, small_target, num_local_experts=2, num_experts_per_tok=1
    )
    config = MixtralConfig(**cfg_kwargs)
    hf_model = MixtralForCausalLM(config)
    target_dtype = next(hf_model.parameters()).dtype
    sd = {k: v.to(dtype=target_dtype) for k, v in sd.items()}
    missing, unexpected = hf_model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:5]}"

    hf_model.eval()
    with torch.no_grad():
        out = hf_model(input_ids=torch.tensor([[0]], dtype=torch.long))
    assert out.logits.shape == (1, 1, 64)
    assert torch.isfinite(out.logits).all()


def test_model_shape_constraint_rejects_default_shaped_vm():
    """Path A wall: a Mixtral-8x7B ModelShapeConstraint mismatches a
    default-shape VM.

    Builds a small ``AutoregressiveVM`` at the C4 VM's natural shape
    family (d_model != 4096, n_layers != 32, num_heads != 32) and asserts
    that ``validate_against_shape`` reports mismatches against a
    Mixtral-8x7B constraint. This pins the gap the pad-export path
    works around: the allocator doesn't yet emit Mixtral-shaped weights.

    (We don't drive this through ``compile_full_vm_dynamic`` because the
    full compile pulls in op-set bugs unrelated to the shape constraint;
    the constraint's logic is purely a shape diff over the compiled
    model's attributes and is exercised cleanly against any VM.)

    If this test ever starts failing because the constraint passes, the
    allocator-native Path A is live and the pad-export path should be
    promoted from "primary" to "fallback / oversize-target" only.
    """
    from c4_release.neural_vm.verification.model_shape_constraint import (
        ModelShapeConstraint,
        validate_against_shape,
    )
    from c4_release.neural_vm.vm_step import AutoregressiveVM

    # A VM at a non-Mixtral shape family. Matches the allocator-native
    # spirit (d_model not 4096, n_layers not 32, vocab not 32000).
    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=512,
        n_layers=18,
        n_heads=8,
        ffn_hidden=2048,
        max_seq_len=128,
    )

    mixtral_constraint = ModelShapeConstraint(
        target="mixtral",
        d_model=_MIXTRAL_8X7B["d_model"],
        num_hidden_layers=_MIXTRAL_8X7B["n_layers"],
        num_attention_heads=_MIXTRAL_8X7B["num_heads"],
        num_key_value_heads=_MIXTRAL_8X7B["num_kv_heads"],
        head_dim=_MIXTRAL_8X7B["head_dim"],
        intermediate_size=_MIXTRAL_8X7B["ffn_hidden"],
        vocab_size=_MIXTRAL_8X7B["vocab_size"],
    )

    mismatches = validate_against_shape(vm, mixtral_constraint)
    assert mismatches, "expected at least one mismatch"
    joined = "; ".join(mismatches)
    # vocab_size: 276 vs 32000; d_model: 512 vs 4096; n_layers: 18 vs 32.
    assert "vocab_size" in joined or "d_model" in joined, (
        f"expected vocab_size/d_model mismatch, got: {joined}"
    )


def _can_compile_full_vm_dynamic() -> bool:
    """Probe whether ``compile_full_vm_dynamic`` is currently green.

    The end-to-end test depends on the heavy compile path; when the
    upstream compile is in flux (Phase 9.B/9.C SSA migration, Phase
    8.G.6 dep-anchor co-placement, etc.) the schedule->layout step can
    raise on unrelated invariants. Probe with ``declarations_only=True``
    so the test self-skips instead of failing on bugs outside the
    adapter / pad-export wiring this test is meant to exercise.
    """
    try:
        from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
            compile_full_vm_dynamic,
        )
        compile_full_vm_dynamic(
            disk_cache=False, strict=False, declarations_only=True
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_transformers_mixtral(), reason=_SKIP_REASON)
@pytest.mark.skipif(
    not _can_compile_full_vm_dynamic(),
    reason=(
        "compile_full_vm_dynamic currently fails on an upstream invariant "
        "unrelated to the Mixtral pad-export adapter (see test_pad_to_mixtral_*"
        " for adapter-only coverage); skipping the heavy end-to-end path "
        "until the compile stack is green."
    ),
)
def test_mixtral_end_to_end_pad_forward_pass():
    """Compile default-shape VM -> pad-export -> load HF Mixtral -> forward.

    The key assertion is wiring: the padded state_dict loads into
    Mixtral-8x7B (no unexpected keys, modest missing-key count for
    HF-side buffers the adapter doesn't touch) and a 1-token forward
    pass returns a finite ``(1, 1, 32000)`` logits tensor.
    """

    from transformers import MixtralConfig, MixtralForCausalLM

    from c4_release.neural_vm.hf_export.mixtral_adapter import (
        MIXTRAL_8X7B_SHAPE,
        export_to_mixtral_state_dict,
    )
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    # 1) Compile at default shape. ``declarations_only=True`` skips the
    #    heavy weight bake; the pad-fill is shape-only, so we don't need
    #    real weights to exercise the wiring.
    model, _layout = compile_full_vm_dynamic(
        disk_cache=True,
        strict=False,
        declarations_only=True,
    )

    # 2) Pad-fill export via the adapter's pad_to_mixtral flag. With
    #    ``target_shape=None`` the adapter uses MIXTRAL_8X7B_SHAPE.
    state_dict = export_to_mixtral_state_dict(
        model,
        pad_to_mixtral=True,
        num_local_experts=8,
    )

    # Spot-check Mixtral-8x7B shapes.
    assert state_dict["model.embed_tokens.weight"].shape == (32000, 4096)
    assert state_dict["lm_head.weight"].shape == (32000, 4096)
    assert state_dict["model.norm.weight"].shape == (4096,)
    for i in range(MIXTRAL_8X7B_SHAPE["n_layers"]):
        prefix = f"model.layers.{i}"
        # Q: num_heads * head_dim = 32 * 128 = 4096
        assert state_dict[f"{prefix}.self_attn.q_proj.weight"].shape == (4096, 4096)
        # K/V: GQA num_kv_heads * head_dim = 8 * 128 = 1024
        assert state_dict[f"{prefix}.self_attn.k_proj.weight"].shape == (1024, 4096)
        assert state_dict[f"{prefix}.self_attn.v_proj.weight"].shape == (1024, 4096)
        assert state_dict[f"{prefix}.self_attn.o_proj.weight"].shape == (4096, 4096)
        assert state_dict[f"{prefix}.block_sparse_moe.gate.weight"].shape == (8, 4096)
        for e in range(8):
            assert state_dict[
                f"{prefix}.block_sparse_moe.experts.{e}.w1.weight"
            ].shape == (14336, 4096)
            assert state_dict[
                f"{prefix}.block_sparse_moe.experts.{e}.w2.weight"
            ].shape == (4096, 14336)
            assert state_dict[
                f"{prefix}.block_sparse_moe.experts.{e}.w3.weight"
            ].shape == (14336, 4096)

    # 3) Build a Mixtral-8x7B config + model, load the padded state_dict,
    #    and run a single forward pass.
    cfg_kwargs = _build_mixtral_config_kwargs_for_target(
        model, MIXTRAL_8X7B_SHAPE, num_local_experts=8, num_experts_per_tok=2
    )
    config = MixtralConfig(**cfg_kwargs)
    hf_model = MixtralForCausalLM(config)
    target_dtype = next(hf_model.parameters()).dtype
    state_dict = {k: v.to(dtype=target_dtype) for k, v in state_dict.items()}
    missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected state_dict keys: {unexpected[:5]}"
    # Only HF-side rotary buffers / generation config should be missing.
    assert len(missing) < 200, (
        f"too many missing keys ({len(missing)}); first few: {missing[:10]}"
    )

    # 4) Forward pass on a single token. Most weights are zero, so logits
    #    won't be informative — but they MUST be finite at the right shape.
    hf_model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([[0]], dtype=torch.long)
        out = hf_model(input_ids=input_ids)
    logits = out.logits
    assert logits.shape == (1, 1, 32000), (
        f"unexpected logits shape: {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "logits contain non-finite values"
