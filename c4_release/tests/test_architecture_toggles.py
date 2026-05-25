import itertools
import os
import sys
from unittest.mock import patch

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.base_layers import PureFFN
from neural_vm.config import VMConfig, get_config, reset_config, set_config
from neural_vm.unified_compiler import full_vm_compiler
from neural_vm.vm_step import (
    AutoregressiveAttention,
    AutoregressiveVM,
    RMSNorm,
    Token,
    TransformerBlock,
)


@pytest.fixture(autouse=True)
def reset_global_vm_config(monkeypatch):
    for name in (
        "NEURAL_VM_POS_ENCODING",
        "NEURAL_VM_ATTENTION_NORMALIZATION",
        "NEURAL_VM_USE_RMS_NORM",
        "NEURAL_VM_RMS_NORM_EPS",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    yield
    reset_config()


def test_default_architecture_toggles_are_legacy_compatible():
    config = get_config()

    assert config.positional_encoding == "alibi"
    assert config.attention_normalization == "softmax1"
    assert config.use_softmax1 is True
    assert config.use_rms_norm is False

    attn = AutoregressiveAttention(dim=32, num_heads=4)
    assert attn._positional_encoding == "alibi"
    assert attn.attention_normalization == "softmax1"
    assert attn.use_softmax1 is True
    assert attn.alibi_slopes is not None
    assert attn._rope_cos is None


def test_env_controls_attention_normalization_and_rmsnorm(monkeypatch):
    monkeypatch.setenv("NEURAL_VM_POS_ENCODING", "rope")
    monkeypatch.setenv("NEURAL_VM_ATTENTION_NORMALIZATION", "softmax")
    monkeypatch.setenv("NEURAL_VM_USE_RMS_NORM", "1")
    reset_config()

    config = get_config()

    assert config.positional_encoding == "rope"
    assert config.attention_normalization == "softmax"
    assert config.use_softmax1 is False
    assert config.use_rms_norm is True


def test_attention_constructor_overrides_global_config():
    set_config(VMConfig.open_model_like_mode())

    attn = AutoregressiveAttention(
        dim=32,
        num_heads=4,
        positional_encoding="alibi",
        attention_normalization="softmax1",
    )

    assert attn._positional_encoding == "alibi"
    assert attn.attention_normalization == "softmax1"
    assert attn.use_softmax1 is True
    assert attn.alibi_slopes is not None
    assert attn._rope_cos is None


def _sdpa_kv_length_for(attention_normalization):
    captured = {}

    def fake_sdpa(q, k, v, *args, **kwargs):
        captured["k_len"] = k.shape[2]
        return torch.zeros_like(q)

    attn = AutoregressiveAttention(
        dim=32,
        num_heads=4,
        max_seq_len=8,
        positional_encoding="alibi",
        attention_normalization=attention_normalization,
        use_flash_attention=True,
    )
    x = torch.randn(1, 4, 32)

    with patch(
        "torch.nn.functional.scaled_dot_product_attention",
        side_effect=fake_sdpa,
    ):
        out = attn(x)

    assert torch.isfinite(out).all()
    return captured["k_len"]


def test_standard_softmax_does_not_append_softmax1_sink():
    assert _sdpa_kv_length_for("softmax") == 4
    assert _sdpa_kv_length_for("softmax1") == 5


def test_rmsnorm_modules_exist_only_when_enabled():
    attn = AutoregressiveAttention(
        dim=32,
        num_heads=4,
        positional_encoding="alibi",
        attention_normalization="softmax1",
    )
    block = TransformerBlock(attn=attn, ffn=PureFFN(32, 8), use_rms_norm=False)
    assert not any(isinstance(module, RMSNorm) for module in block.modules())
    assert not hasattr(block, "attn_norm")
    assert not hasattr(block, "ffn_norm")

    attn = AutoregressiveAttention(
        dim=32,
        num_heads=4,
        positional_encoding="alibi",
        attention_normalization="softmax1",
    )
    block = TransformerBlock(attn=attn, ffn=PureFFN(32, 8), use_rms_norm=True)
    norms = [module for module in block.modules() if isinstance(module, RMSNorm)]
    assert len(norms) == 2
    assert hasattr(block, "attn_norm")
    assert hasattr(block, "ffn_norm")


@pytest.mark.parametrize(
    ("use_rope", "use_standard_softmax", "use_rms_norm"),
    itertools.product((False, True), repeat=3),
)
def test_all_architecture_toggle_combinations_run_forward(
    use_rope, use_standard_softmax, use_rms_norm
):
    model = AutoregressiveVM(
        n_layers=2,
        d_model=512,
        n_heads=8,
        ffn_hidden=16,
        max_seq_len=16,
        positional_encoding="rope" if use_rope else "alibi",
        attention_normalization="softmax" if use_standard_softmax else "softmax1",
        use_rms_norm=use_rms_norm,
        use_flash_attention=False,
    )
    token_ids = torch.tensor(
        [[Token.CODE_START, 1, 2, 3, Token.CODE_END, Token.STEP_END]],
        dtype=torch.long,
    )

    logits = model(token_ids)

    assert logits.shape == (1, 6, Token.VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_open_model_like_factory_sets_standard_architecture_toggles():
    config = VMConfig.open_model_like_mode()

    assert config.positional_encoding == "rope"
    assert config.attention_normalization == "softmax"
    assert config.use_softmax1 is False
    assert config.use_rms_norm is True


def test_compiler_cache_key_distinguishes_architecture_toggles(monkeypatch):
    monkeypatch.setattr(full_vm_compiler, "_hash_source_bytes", lambda: "source")
    base = {
        "positional_encoding": "alibi",
        "attention_normalization": "softmax1",
        "rope_base": 10000.0,
        "use_rms_norm": False,
        "rms_norm_eps": 1e-6,
    }

    base_key = full_vm_compiler._cache_key(base)
    assert full_vm_compiler._cache_key({**base, "positional_encoding": "rope"}) != base_key
    assert full_vm_compiler._cache_key({**base, "attention_normalization": "softmax"}) != base_key
    assert full_vm_compiler._cache_key({**base, "use_rms_norm": True}) != base_key
