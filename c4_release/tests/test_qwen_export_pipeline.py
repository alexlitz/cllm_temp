"""Phase R6 — end-to-end Qwen3-dense export pipeline tests.

These tests build a tiny ``AutoregressiveVM`` instance (NOT through
``compile_full_vm_dynamic`` — too slow for unit tests), manually seed the
NORM_COMPENSATOR slot the same way the R1 bake would, then call
``export_qwen3_dense`` and inspect the on-disk artefact.

The tests intentionally do NOT call ``AutoModelForCausalLM.from_pretrained``
on the output — that gate is owned by Phase R8 once Phase R5's post_ops
flattening lands and the per-block-count round-trip becomes meaningful.
The R6 acceptance criterion is "config.json + state_dict + tokenizer files
materialise correctly with the R1-R4 invariants preserved".
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.qwen_compat import (
    Qwen3DenseConfig,
    export_qwen3_dense,
)
from neural_vm.vm_step import AutoregressiveVM


NORM_COMPENSATOR_K = 1000.0


def _build_tiny_vm_with_compensator(
    *,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    ffn_hidden: int = 64,
    norm_compensator_idx: int = 0,
    bias_compensator_idx: int = 1,
):
    """Build a tiny VM with the NORM_COMPENSATOR slot pre-seeded.

    Mirrors the layout that ``compile_full_vm_dynamic`` would produce with
    ``C4_QWEN_EXPORT_COMPAT=1``: the dim is registered in
    ``model.dim_positions`` and the embedding column at that index is
    pinned to ``K=1000.0``. We also defensively zero the corresponding
    ``W_o`` and ``W_down`` rows so the R6 sanity check passes.
    """

    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=128,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_rms_norm=True,
        use_flash_attention=False,
    )

    # The base AutoregressiveVM uses ``_SetDim`` as the dim_positions
    # fallback. Replace it with an explicit dict carrying the compensator
    # entries so the R1 invariant check finds them.
    vm.dim_positions = {
        "NORM_COMPENSATOR": norm_compensator_idx,
        "CONST": bias_compensator_idx,
    }

    with torch.no_grad():
        embed_weight = vm.embed.embed.weight
        embed_weight[:, norm_compensator_idx] = NORM_COMPENSATOR_K
        for block in vm.blocks:
            # Defensive zero: matches the R1 bake. Without this, downstream
            # attention/FFN could write through the compensator slot and
            # break RMS identity in inference; the export gate checks the
            # *embedding* column only, but we mirror the bake for fidelity.
            block.attn.W_o.data[norm_compensator_idx, :] = 0.0
            block.ffn.W_down.data[norm_compensator_idx, :] = 0.0
            # Plant non-trivial biases so the R4 bias-fold has something to
            # absorb — otherwise the test would pass via empty input.
            block.ffn.b_up.data.fill_(0.5)
            block.ffn.b_gate.data.fill_(0.25)

    return vm


def test_export_qwen3_dense_writes_expected_files():
    vm = _build_tiny_vm_with_compensator()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        assert isinstance(cfg, Qwen3DenseConfig)
        # Three artefacts: config.json, pytorch_model.bin, tokenizer_config.json.
        files = set(os.listdir(tmp))
        assert "config.json" in files
        assert "pytorch_model.bin" in files
        assert "tokenizer_config.json" in files


def test_config_json_has_qwen3_architecture_fields():
    vm = _build_tiny_vm_with_compensator(d_model=32, n_layers=2, n_heads=4, ffn_hidden=64)
    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        with open(os.path.join(tmp, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)

    assert cfg["architectures"] == ["Qwen3ForCausalLM"]
    assert cfg["model_type"] == "qwen3"
    assert cfg["vocab_size"] == 276
    assert cfg["hidden_size"] == 32
    assert cfg["intermediate_size"] == 64
    assert cfg["num_hidden_layers"] == 2
    assert cfg["num_attention_heads"] == 4
    assert cfg["num_key_value_heads"] == 4
    assert cfg["head_dim"] == 8  # 32 / 4
    assert cfg["hidden_act"] == "silu"
    assert cfg["c4_qwen_compat_flag"] is True
    assert cfg["c4_norm_compensator_K"] == NORM_COMPENSATOR_K
    assert cfg["c4_norm_compensator_idx"] == 0
    assert cfg["c4_bias_compensator_idx"] == 1
    # R5 isn't on main → expect the graceful skip.
    assert cfg["c4_post_ops_flattened"] is False


def test_state_dict_has_expected_qwen3_keys():
    vm = _build_tiny_vm_with_compensator(d_model=32, n_layers=2, n_heads=4, ffn_hidden=64)
    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        sd = torch.load(
            os.path.join(tmp, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )

    expected = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for i in range(2):
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            expected.add(f"model.layers.{i}.{suffix}")
    missing = expected - set(sd.keys())
    assert not missing, f"missing keys in exported state_dict: {sorted(missing)}"


def test_norm_compensator_row_preserved_in_embedding():
    """R1 invariant: embedding column at NORM_COMPENSATOR idx must hold K."""
    vm = _build_tiny_vm_with_compensator(norm_compensator_idx=0)
    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        sd = torch.load(
            os.path.join(tmp, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )

    col = sd["model.embed_tokens.weight"][:, 0]
    assert torch.allclose(
        col, torch.full_like(col, NORM_COMPENSATOR_K), atol=1e-3
    ), (
        f"NORM_COMPENSATOR column drift: range "
        f"[{float(col.min())}, {float(col.max())}]"
    )


def test_rmsnorm_gamma_is_identity_compensating():
    """R2 invariant: every RMSNorm gamma = K / sqrt(d_model)."""
    vm = _build_tiny_vm_with_compensator(d_model=32, n_layers=2)
    expected = NORM_COMPENSATOR_K / math.sqrt(32)
    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        sd = torch.load(
            os.path.join(tmp, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )

    for key in (
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
    ):
        gamma = sd[key]
        assert gamma.shape == (32,), f"{key} shape {tuple(gamma.shape)} != (32,)"
        assert torch.allclose(
            gamma,
            torch.full_like(gamma, expected),
            atol=1e-5,
        ), f"{key} gamma not at identity-compensating value {expected}"


def test_swiglu_repack_and_bias_fold():
    """R4 invariant: gate_proj == VM W_up + folded b_up at CONST col."""
    bias_idx = 1
    vm = _build_tiny_vm_with_compensator(
        d_model=32, n_layers=1, ffn_hidden=64, bias_compensator_idx=bias_idx
    )
    block = vm.blocks[0]
    original_W_up = block.ffn.W_up.data.clone()
    original_W_gate = block.ffn.W_gate.data.clone()
    original_b_up = block.ffn.b_up.data.clone()
    original_b_gate = block.ffn.b_gate.data.clone()

    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        sd = torch.load(
            os.path.join(tmp, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )

    gate_proj = sd["model.layers.0.mlp.gate_proj.weight"]
    up_proj = sd["model.layers.0.mlp.up_proj.weight"]

    expected_gate_proj = original_W_up.clone().float()
    expected_gate_proj[:, bias_idx] = expected_gate_proj[:, bias_idx] + original_b_up.float()
    expected_up_proj = original_W_gate.clone().float()
    expected_up_proj[:, bias_idx] = expected_up_proj[:, bias_idx] + original_b_gate.float()

    assert torch.allclose(gate_proj, expected_gate_proj, atol=1e-6), (
        "gate_proj does not equal VM W_up with b_up folded at CONST column"
    )
    assert torch.allclose(up_proj, expected_up_proj, atol=1e-6), (
        "up_proj does not equal VM W_gate with b_gate folded at CONST column"
    )


def test_tokenizer_config_has_byte_level_metadata():
    vm = _build_tiny_vm_with_compensator()
    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        with open(os.path.join(tmp, "tokenizer_config.json"), encoding="utf-8") as fh:
            tok = json.load(fh)

    assert tok["tokenizer_class"] == "C4QwenByteLevelTokenizer"
    assert tok["vocab_size"] == 276
    assert tok["c4_byte_token_range"] == [0, 256]
    assert "<|c4:STEP_END|>" in tok["additional_special_tokens"]
    # Spot-check the round-trip ID table: STEP_END token id is well-known.
    from neural_vm.vm_step import Token

    assert tok["special_token_ids"]["<|c4:STEP_END|>"] == int(Token.STEP_END)


def test_export_raises_without_compensator_slot():
    """If the dim_positions don't include NORM_COMPENSATOR, raise — R1 gate."""
    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ffn_hidden=64,
        max_seq_len=128,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_rms_norm=True,
        use_flash_attention=False,
    )
    # Explicit empty dict — _SetDim fallback would also fail the check,
    # but we want a deterministic dict for the assertion message.
    vm.dim_positions = {"CONST": 0}
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="C4_QWEN_EXPORT_COMPAT"):
            export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)


def test_export_skipped_reason_when_no_post_ops():
    """The R5 graceful-skip path: no post_ops means flatten is a no-op."""
    vm = _build_tiny_vm_with_compensator()
    # Tiny VM has no post_ops by construction.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
    assert cfg.c4_post_ops_flattened is False
    assert cfg.c4_post_ops_flatten_skipped_reason is not None
    assert "no post_ops" in cfg.c4_post_ops_flatten_skipped_reason
