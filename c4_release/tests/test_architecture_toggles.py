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


# ---------------------------------------------------------------------------
# Toggle differentiation tests (per TESTING_CHECKLIST / Phase 8.O).
#
# For each architectural toggle we assert two things:
#   (a) the non-default value produces observably different behaviour than
#       the default (i.e. the toggle actually toggles something), and
#   (b) the current default value remains byte-identical to a model built
#       with no architecture kwarg (the no-regress guarantee).
# Byte-identity at the 4-axis default is also covered by
# ``test_default_combo_byte_identical_to_no_kwargs``.
# ---------------------------------------------------------------------------


def _randomize_attention_inplace(attn, seed: int = 0):
    """Fill Q/K/V/O with nontrivial weights so position encoding actually
    influences the attention scores.

    With the default zero-init weights every score collapses to zero
    regardless of RoPE/ALiBi, so any difference between modes would be
    invisible. A small random fill keeps the test sensitive while leaving
    the toggle logic itself untouched.
    """
    g = torch.Generator().manual_seed(seed)
    for p in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        p.data.normal_(generator=g)
        p.data.mul_(0.1)


def test_rope_vs_alibi_produces_different_attention_outputs():
    """RoPE and ALiBi must produce observably different attention outputs
    on the same input + same Q/K/V/O weights. This pins the position-
    encoding toggle to actual numerical behaviour, not just buffer presence.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 6, 32)

    attn_alibi = AutoregressiveAttention(
        dim=32, num_heads=4, max_seq_len=8,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_flash_attention=False,
    )
    _randomize_attention_inplace(attn_alibi, seed=1)

    attn_rope = AutoregressiveAttention(
        dim=32, num_heads=4, max_seq_len=8,
        positional_encoding="rope",
        attention_normalization="softmax1",
        use_flash_attention=False,
    )
    # Identical Q/K/V/O so the only remaining axis of variation is the
    # position-encoding branch (alibi_slopes-as-bias vs RoPE Q/K rotation).
    with torch.no_grad():
        attn_rope.W_q.copy_(attn_alibi.W_q)
        attn_rope.W_k.copy_(attn_alibi.W_k)
        attn_rope.W_v.copy_(attn_alibi.W_v)
        attn_rope.W_o.copy_(attn_alibi.W_o)

    with torch.no_grad():
        out_alibi = attn_alibi(x)
        out_rope = attn_rope(x)

    assert torch.isfinite(out_alibi).all()
    assert torch.isfinite(out_rope).all()
    assert not torch.allclose(out_alibi, out_rope, atol=1e-6), (
        "RoPE and ALiBi attention produced identical outputs — the "
        "position-encoding toggle is not influencing attention scores."
    )
    # Buffer presence matches the toggle (defensive — guards against a
    # regression that nominally accepts the kwarg but never populates
    # the RoPE cache).
    assert attn_alibi.alibi_slopes is not None and attn_alibi._rope_cos is None
    assert attn_rope.alibi_slopes is None and attn_rope._rope_cos is not None


def test_alibi_default_byte_identical_to_no_kwarg():
    """The current position-encoding default (``"alibi"``) must be
    byte-identical to constructing AutoregressiveAttention with no
    ``positional_encoding`` kwarg.
    """
    torch.manual_seed(0)
    attn_default = AutoregressiveAttention(
        dim=32, num_heads=4, max_seq_len=8,
        attention_normalization="softmax1",
        use_flash_attention=False,
    )
    _randomize_attention_inplace(attn_default, seed=4)

    torch.manual_seed(0)
    attn_explicit = AutoregressiveAttention(
        dim=32, num_heads=4, max_seq_len=8,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_flash_attention=False,
    )
    _randomize_attention_inplace(attn_explicit, seed=4)

    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        y_default = attn_default(x)
        y_explicit = attn_explicit(x)

    assert torch.equal(y_default, y_explicit), (
        "positional_encoding default drifted from no-kwarg baseline."
    )


def test_div_mode_log_softmax1_short_circuits_before_long_div_pipeline():
    """``div_mode='log_softmax1'`` is documented as a Phase 8.O.3 stub
    (BLOG_SPEC.md "Via Attention With Log Sink") and must raise at
    DIV/MOD execution time, while ``div_mode='long_div'`` runs through
    to the long-division pipeline lookup tables.

    This pins the div_mode toggle to a real behavioural difference at
    runtime (different code paths for the ALU lookup-table stage) rather
    than just config-flag plumbing. We fake an installed pipeline so the
    forward() reaches the cfg.div_mode gate (the install ops require the
    full compiler pipeline to bake the baked-table stages).
    """
    from neural_vm.efficient_alu_divmod_split import FlattenedDivMod

    class _BDProxy:
        OP_DIV = 0
        OP_MOD = 1
        MARK_AX = 2
        CONST = 3

    composite = FlattenedDivMod(S=100.0, BD=_BDProxy)

    captured = {}

    def fake_pipeline(x):
        captured["ran"] = True
        return x

    composite.__dict__["pipeline"] = fake_pipeline

    # x_bd[..., OP_DIV] > 0.1 trips the early-out gate and reaches the
    # div_mode branch.
    x_bd = torch.zeros(1, 4, 64)
    x_bd[..., _BDProxy.OP_DIV] = 1.0

    # log_softmax1: stub raises NotImplementedError before pipeline runs.
    set_config(VMConfig(
        positional_encoding="alibi",
        attention_normalization="softmax1",
        div_mode="log_softmax1",
    ))
    captured.clear()
    with pytest.raises(NotImplementedError, match="log_softmax1"):
        composite(x_bd)
    assert "ran" not in captured, (
        "log_softmax1 must short-circuit before the long-division pipeline."
    )

    # long_div: gate passes through to the (fake) pipeline — this is the
    # baked-lookup-table path that actually computes DIV/MOD.
    set_config(VMConfig(
        positional_encoding="alibi",
        attention_normalization="softmax1",
        div_mode="long_div",
    ))
    captured.clear()
    out = composite(x_bd)
    assert captured.get("ran") is True, (
        "long_div must execute the long-division pipeline (the lookup-"
        "table path) when DIV is active."
    )
    assert out.shape == x_bd.shape


def test_div_mode_log_softmax1_requires_softmax1_attention():
    """``div_mode='log_softmax1'`` depends on the softmax1 +1 sink
    denominator (1/(1+(n-1)) = 1/n). Pairing it with plain softmax must
    fail at config construction time — the documented contract in
    ``VMConfig.__post_init__``.
    """
    with pytest.raises(ValueError, match="softmax1"):
        VMConfig(
            positional_encoding="alibi",
            attention_normalization="softmax",
            div_mode="log_softmax1",
        )


def test_div_mode_default_long_div_byte_identical_to_no_kwarg():
    """The current div_mode default (``"long_div"``) must be byte-
    identical to a VMConfig constructed with no ``div_mode`` kwarg.
    """
    default = VMConfig(
        positional_encoding="alibi",
        attention_normalization="softmax1",
    )
    explicit = VMConfig(
        positional_encoding="alibi",
        attention_normalization="softmax1",
        div_mode="long_div",
    )
    assert default.div_mode == explicit.div_mode == "long_div"


def test_swiglu_ffn_hidden_ratio_configurable_for_mixtral():
    """The SwiGLU FFN hidden dim is an explicit constructor kwarg, so
    Mixtral-style ratios (intermediate = 3.5 * d_model) are reachable.

    Asserts:
      * ``PureFFN(dim=d, hidden_dim=int(3.5*d))`` constructs cleanly.
      * The baked weight shapes match the ratio (no silent rounding /
        clamping).
      * Forward runs and returns the expected shape.
      * An ``AutoregressiveVM`` with ``ffn_hidden=int(3.5*d_model)``
        also constructs and each block's FFN reports the requested
        width — i.e. the toggle propagates end-to-end.
      * A different ratio produces different weight shapes (the kwarg
        isn't a no-op constant).
    """
    d_model = 64

    # Mixtral expects intermediate = 3.5 * hidden_size.
    mixtral_hidden = int(d_model * 3.5)
    ffn = PureFFN(dim=d_model, hidden_dim=mixtral_hidden)
    assert ffn.hidden_dim == mixtral_hidden
    assert ffn.W_up.shape == (mixtral_hidden, d_model)
    assert ffn.W_gate.shape == (mixtral_hidden, d_model)
    assert ffn.W_down.shape == (d_model, mixtral_hidden)
    assert ffn.b_up.shape == (mixtral_hidden,)

    x = torch.randn(1, 3, d_model)
    with torch.no_grad():
        y = ffn(x)
    assert y.shape == x.shape

    # A non-Mixtral ratio must yield a different shape — proves the
    # kwarg isn't being clamped to some hard-coded width.
    vanilla_hidden = d_model * 4
    ffn_vanilla = PureFFN(dim=d_model, hidden_dim=vanilla_hidden)
    assert ffn_vanilla.hidden_dim == vanilla_hidden
    assert ffn_vanilla.W_up.shape != ffn.W_up.shape

    # End-to-end: AutoregressiveVM honours per-block ffn_hidden too.
    d_model_vm = 32
    mixtral_vm_hidden = int(d_model_vm * 3.5)
    torch.manual_seed(0)
    vm = AutoregressiveVM(
        n_layers=2,
        d_model=d_model_vm,
        n_heads=4,
        ffn_hidden=mixtral_vm_hidden,
        max_seq_len=16,
        use_flash_attention=False,
    )
    for block in vm.blocks:
        assert block.ffn.hidden_dim == mixtral_vm_hidden
        assert block.ffn.W_up.shape == (mixtral_vm_hidden, d_model_vm)


def test_swiglu_ffn_per_block_widths_dict_propagates():
    """``ffn_hidden`` accepts a dict[int, int] of per-block widths from
    the compiler's ``ModelLayout.ffn_widths``. This pins the per-block
    FFN-ratio toggle (a generalisation of the Mixtral 3.5x case).
    """
    d_model = 32
    per_block = {0: 16, 1: int(d_model * 3.5)}  # Mixtral on block 1

    torch.manual_seed(0)
    vm = AutoregressiveVM(
        n_layers=2,
        d_model=d_model,
        n_heads=4,
        ffn_hidden=per_block,
        max_seq_len=16,
        use_flash_attention=False,
    )
    assert vm.blocks[0].ffn.hidden_dim == per_block[0]
    assert vm.blocks[1].ffn.hidden_dim == per_block[1]
    assert vm.blocks[0].ffn.hidden_dim != vm.blocks[1].ffn.hidden_dim


def test_rmsnorm_toggle_changes_block_output():
    """The ``use_rms_norm`` toggle must produce observably different
    block outputs on the same input + identical attention/FFN weights.

    Existing test_rmsnorm_modules_exist_only_when_enabled covers the
    structural side (presence of attn_norm / ffn_norm submodules); this
    test covers the numerical side — that the toggle actually changes
    forward output. Without nontrivial Q/K/V/O the zero-init attn
    contribution is zero and the toggle has no visible effect, so we
    randomize Q/K/V/O first.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)

    def make_block(use_rms_norm):
        attn = AutoregressiveAttention(
            dim=32, num_heads=4, max_seq_len=8,
            positional_encoding="alibi",
            attention_normalization="softmax1",
            use_flash_attention=False,
        )
        _randomize_attention_inplace(attn, seed=2)
        ffn = PureFFN(32, 16)
        return TransformerBlock(attn=attn, ffn=ffn, use_rms_norm=use_rms_norm)

    block_off = make_block(use_rms_norm=False)
    block_on = make_block(use_rms_norm=True)

    assert not hasattr(block_off, "attn_norm")
    assert hasattr(block_on, "attn_norm")
    assert isinstance(block_on.attn_norm, RMSNorm)
    assert isinstance(block_on.ffn_norm, RMSNorm)

    with torch.no_grad():
        y_off = block_off(x)
        y_on = block_on(x)

    assert torch.isfinite(y_off).all()
    assert torch.isfinite(y_on).all()
    assert y_off.shape == y_on.shape == x.shape
    assert not torch.allclose(y_off, y_on, atol=1e-6), (
        "RMSNorm toggle did not change the block output — toggle is "
        "wired structurally but has no numerical effect."
    )


def test_rmsnorm_default_off_byte_identical_to_no_kwarg():
    """The current default (``use_rms_norm=False``) must be byte-
    identical to constructing a block with no ``use_rms_norm`` kwarg.
    Mixtral-style ``use_rms_norm=True`` flips the default; this test
    guards that the False path remains a no-op against existing tests.
    """
    torch.manual_seed(0)
    attn1 = AutoregressiveAttention(
        dim=32, num_heads=4, max_seq_len=8,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_flash_attention=False,
    )
    _randomize_attention_inplace(attn1, seed=3)
    block_default = TransformerBlock(attn=attn1, ffn=PureFFN(32, 16))

    torch.manual_seed(0)
    attn2 = AutoregressiveAttention(
        dim=32, num_heads=4, max_seq_len=8,
        positional_encoding="alibi",
        attention_normalization="softmax1",
        use_flash_attention=False,
    )
    _randomize_attention_inplace(attn2, seed=3)
    block_explicit = TransformerBlock(
        attn=attn2, ffn=PureFFN(32, 16), use_rms_norm=False
    )

    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        y_default = block_default(x)
        y_explicit = block_explicit(x)

    assert torch.equal(y_default, y_explicit), (
        "use_rms_norm=False is not byte-identical to no-kwarg default."
    )
