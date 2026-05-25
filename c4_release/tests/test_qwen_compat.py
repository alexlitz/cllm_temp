"""Qwen compatibility scaffolding tests.

These tests instantiate tiny local models only. They do not download Qwen
checkpoints and they do not assert semantic equivalence.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("transformers")

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from neural_vm.qwen_compat import (
    QWEN2_DENSE,
    analyze_qwen_compatibility,
    build_qwen2_config,
    build_qwen2_dense_mapping_plan,
    infer_qwen2_config_kwargs,
)
from neural_vm.vm_step import AutoregressiveVM


def _tiny_vm(*, use_rms_norm=True):
    return AutoregressiveVM(
        vocab_size=276,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_hidden=64,
        max_seq_len=128,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_rms_norm=use_rms_norm,
        use_flash_attention=False,
    )


def test_qwen2_config_is_built_from_vm_dimensions():
    vm = _tiny_vm()

    kwargs = infer_qwen2_config_kwargs(vm)
    config = build_qwen2_config(vm)

    assert kwargs["vocab_size"] == 276
    assert kwargs["hidden_size"] == 32
    assert kwargs["intermediate_size"] == 64
    assert kwargs["num_hidden_layers"] == 2
    assert kwargs["num_attention_heads"] == 4
    assert kwargs["num_key_value_heads"] == 4
    assert config.model_type == "qwen2"
    assert config.rope_theta == vm.rope_base


def test_qwen2_dense_mapping_plan_matches_tiny_qwen2_shapes():
    vm = _tiny_vm(use_rms_norm=True)
    qwen = Qwen2ForCausalLM(build_qwen2_config(vm))

    plan = build_qwen2_dense_mapping_plan(vm)
    validation = plan.validate(vm.state_dict(), qwen.state_dict())

    assert plan.target_architecture == QWEN2_DENSE
    assert validation.ok
    assert validation.shape_mismatches == []
    assert validation.unmapped_target_keys == []
    assert validation.missing_source_keys == []
    assert validation.missing_target_keys == []

    mappings = {(m.action, m.target_key, m.source_key) for m in plan.mappings}
    assert (
        "copy",
        "model.layers.0.self_attn.q_proj.weight",
        "blocks.0.attn.W_q",
    ) in mappings
    assert (
        "copy",
        "model.layers.0.mlp.gate_proj.weight",
        "blocks.0.ffn.W_gate",
    ) in mappings
    assert ("zeros", "model.layers.0.self_attn.q_proj.bias", None) in mappings
    assert ("drop_source", None, "head.bias") in mappings


def test_compat_report_never_claims_direct_qwen_loadability():
    report = analyze_qwen_compatibility(_tiny_vm(use_rms_norm=False))

    assert report.closest_target == QWEN2_DENSE
    assert report.directly_loadable is False
    assert any("State dict key namespace" in blocker for blocker in report.blockers)
    assert any("per-block RMSNorm" in blocker for blocker in report.blockers)
    assert any("tokenizer" in item.lower() for item in report.adapter_required)
