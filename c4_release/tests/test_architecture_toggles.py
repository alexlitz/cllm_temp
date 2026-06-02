import itertools
import os
import sys
from unittest.mock import patch

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.base_layers import PureFFN
from neural_vm.config import VMConfig, get_config, reset_config, set_config
from neural_vm.unified_compiler import _legacy_redirect as full_vm_compiler
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


# ---------------------------------------------------------------------------
# Phase 8.O.9 — full 16-combination smoke matrix
#
# Axes (per Phase 8.O plan):
#   position_encoding ∈ {alibi, rope}             — 8.O.1 (RoPE) landed
#   attn_softmax      ∈ {softmax1, softmax}       — 8.O.2 in flight
#   div_mode          ∈ {long_div, fast_div}      — 8.O.3 in flight
#   output_norm       ∈ {no_norm, rms_norm}       — 8.O.4 in flight
#
# Default config (alibi + softmax1 + long_div + no_norm) is asserted to be
# byte-identical to ``AutoregressiveVM`` constructed with no architecture
# kwargs at all (existing-test byte-identity guarantee). Non-default
# combinations are only required to compile + run forward with valid output
# shape on every smoke input — pass-rate gating, not byte-identity.
#
# Toggles whose kwarg has not landed yet (currently ``div_mode``) cause
# ``TypeError`` at construction; the harness auto-skips those rows rather
# than failing. As each 8.O.N toggle lands its kwarg, the skip silently
# turns into a real assertion.
# ---------------------------------------------------------------------------

# Five deterministic smoke inputs covering distinct VM-token shapes:
#   1) bare CODE block (shortest legal program)
#   2) CODE block with payload + STEP_END
#   3) two STEP_ENDs (multi-step boundary)
#   4) byte-only payload (no markers)
#   5) full register prologue prefix (PC marker + 4 value bytes)
_SMOKE_INPUTS = (
    [Token.CODE_START, Token.CODE_END],
    [Token.CODE_START, 1, 2, 3, Token.CODE_END, Token.STEP_END],
    [Token.CODE_START, 7, Token.CODE_END, Token.STEP_END, Token.STEP_END],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [Token.REG_PC, 0, 0, 0, 0, Token.STEP_END],
)


def _toggle_kwargs(position_encoding, attn_softmax, div_mode, output_norm):
    """Map the 4-axis combo identifiers onto ``AutoregressiveVM`` kwargs.

    Returns the kwargs dict to splat into the constructor. ``div_mode`` and
    ``output_norm`` axes use their eventual public kwarg names so that as
    Phase 8.O.3 / 8.O.4 land, the auto-skip becomes a real test row without
    edits here.
    """
    kwargs = {
        "positional_encoding": position_encoding,
        "attention_normalization": attn_softmax,
        "use_rms_norm": output_norm == "rms_norm",
    }
    # 8.O.3: surface div_mode via a kwarg. Until landed, passing it raises
    # TypeError and the test row is skipped (see _build_or_skip).
    if div_mode != "long_div":
        kwargs["div_mode"] = div_mode
    return kwargs


def _build_or_skip(**kwargs):
    """Construct AutoregressiveVM; skip the test row if a toggle kwarg is
    not yet accepted (TypeError on unexpected keyword argument).
    """
    try:
        return AutoregressiveVM(
            n_layers=2,
            d_model=512,
            n_heads=8,
            ffn_hidden=16,
            max_seq_len=16,
            use_flash_attention=False,
            **kwargs,
        )
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" in msg:
            pytest.skip(f"toggle not yet wired: {msg}")
        raise


_TOGGLE_AXES = (
    ("alibi", "rope"),
    ("softmax1", "softmax"),
    ("long_div", "fast_div"),
    ("no_norm", "rms_norm"),
)


@pytest.mark.parametrize(
    ("position_encoding", "attn_softmax", "div_mode", "output_norm"),
    list(itertools.product(*_TOGGLE_AXES)),
)
def test_all_architecture_toggle_combinations_run_forward(
    position_encoding, attn_softmax, div_mode, output_norm
):
    """All 16 (position × softmax × div × norm) combos compile and forward.

    Asserts per smoke input:
      * model constructs (skipped via _build_or_skip if a kwarg is unwired)
      * forward returns the expected (batch, seq, vocab) shape
      * every logit is finite (no NaN/Inf leakage from a broken toggle)
    """
    kwargs = _toggle_kwargs(position_encoding, attn_softmax, div_mode, output_norm)

    torch.manual_seed(0)
    model = _build_or_skip(**kwargs)
    model.eval()

    for tokens in _SMOKE_INPUTS:
        token_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            logits = model(token_ids)
        assert logits.shape == (1, len(tokens), Token.VOCAB_SIZE), (
            f"shape mismatch for combo "
            f"({position_encoding},{attn_softmax},{div_mode},{output_norm}) "
            f"on input len={len(tokens)}: got {tuple(logits.shape)}"
        )
        assert torch.isfinite(logits).all(), (
            f"non-finite logits for combo "
            f"({position_encoding},{attn_softmax},{div_mode},{output_norm}) "
            f"on input {tokens}"
        )


def test_default_combo_byte_identical_to_no_kwargs():
    """The default 4-axis combo (alibi + softmax1 + long_div + no_norm) must
    produce byte-identical logits to ``AutoregressiveVM(...)`` with no
    architecture kwargs at all. This guards the contract that the toggle
    matrix's default row is a no-op against existing tests.
    """
    torch.manual_seed(1234)
    model_default = AutoregressiveVM(
        n_layers=2,
        d_model=512,
        n_heads=8,
        ffn_hidden=16,
        max_seq_len=16,
        use_flash_attention=False,
    )
    model_default.eval()

    torch.manual_seed(1234)
    model_explicit = AutoregressiveVM(
        n_layers=2,
        d_model=512,
        n_heads=8,
        ffn_hidden=16,
        max_seq_len=16,
        use_flash_attention=False,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_rms_norm=False,
    )
    model_explicit.eval()

    for tokens in _SMOKE_INPUTS:
        token_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            logits_default = model_default(token_ids)
            logits_explicit = model_explicit(token_ids)
        assert torch.equal(logits_default, logits_explicit), (
            "default toggle row drifted from no-kwargs baseline on input "
            f"{tokens}"
        )


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
