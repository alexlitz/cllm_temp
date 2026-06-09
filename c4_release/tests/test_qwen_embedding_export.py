"""R8 Blocker 5 — NeuralVMEmbedding ADDR_KEY + MEM_STORE export tests.

Pins the runtime-wrapper protocol that exports the ``NeuralVMEmbedding``
augmentations to a Qwen3 artefact. The ADDR_KEY scatter is a deterministic
function of (position, in-sequence CODE_END detector) and is therefore
not bakeable into the ``(vocab_size, d_model)`` lookup table; MEM_STORE
depends on per-row state set by the KV-cache eviction logic. Both are
exposed via :func:`apply_neural_vm_embedding_augmentations` and
:class:`NeuralVMEmbeddingWrapper` in ``neural_vm.qwen_compat``.

The tests below assert:

1. The wrapper output matches the native ``NeuralVMEmbedding.forward``
   bit-for-bit when the dim positions and runtime state match.
2. ``Qwen3DenseConfig`` round-trips the new ``c4_addr_key_*`` /
   ``c4_mem_store_idx`` / ``c4_mem_addr_src_idx`` / ``c4_code_end_token_id``
   / ``c4_mem_token_id`` fields through ``config.json``.
3. The wrapper can be reconstructed from a written config without
   importing the native VM.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.constants import INSTR_WIDTH, PC_OFFSET  # noqa: E402
from neural_vm.neural_embedding import NeuralVMEmbedding  # noqa: E402
from neural_vm.qwen_compat import (  # noqa: E402
    NeuralVMEmbeddingWrapper,
    Qwen3DenseConfig,
    apply_neural_vm_embedding_augmentations,
)
from neural_vm.vm_step import AutoregressiveVM, Token  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addr_key_dim_positions(
    *,
    addr_key_idx: int = 16,
    mem_store_idx: int = 64,
    mem_addr_src_idx: int = 65,
    extra: dict | None = None,
) -> dict:
    """Build a dim_positions dict where ADDR_KEY/MEM_STORE/MEM_ADDR_SRC fit."""

    out = {
        "ADDR_KEY": addr_key_idx,
        "MEM_STORE": mem_store_idx,
        "MEM_ADDR_SRC": mem_addr_src_idx,
    }
    if extra:
        out.update(extra)
    return out


def _seed_token_grid(
    *,
    seq_len: int,
    code_end_pos: int,
    mem_positions: list[int],
) -> torch.Tensor:
    """Build a ``(1, seq_len)`` input_ids tensor with a CODE_END and MEM markers."""

    ids = torch.arange(seq_len, dtype=torch.long) % 200  # arbitrary byte tokens
    if 0 <= code_end_pos < seq_len:
        ids[code_end_pos] = int(Token.CODE_END)
    for p in mem_positions:
        if 0 <= p < seq_len:
            ids[p] = int(Token.MEM)
    return ids.unsqueeze(0)


# ---------------------------------------------------------------------------
# Test 1: wrapper matches native NeuralVMEmbedding bit-for-bit on ADDR_KEY.
# ---------------------------------------------------------------------------


def test_addr_key_wrapper_matches_native_embedding():
    """Wrapper ADDR_KEY scatter reproduces ``NeuralVMEmbedding.forward``."""

    torch.manual_seed(0)
    vocab_size, d_model, seq_len = 276, 128, 48
    dim_positions = _addr_key_dim_positions()

    native = NeuralVMEmbedding(vocab_size, d_model, dim_positions=dim_positions,
                                max_seq_len=seq_len)
    # Wrap the SAME nn.Embedding so the base lookup is identical.
    wrapper = NeuralVMEmbeddingWrapper(
        native.embed,
        addr_key_idx=dim_positions["ADDR_KEY"],
        addr_key_width=48,
        mem_store_idx=dim_positions["MEM_STORE"],
        mem_addr_src_idx=dim_positions["MEM_ADDR_SRC"],
        code_end_token_id=int(Token.CODE_END),
        mem_token_id=int(Token.MEM),
        pc_offset=PC_OFFSET,
        instr_width=INSTR_WIDTH,
    )

    input_ids = _seed_token_grid(
        seq_len=seq_len, code_end_pos=40, mem_positions=[]
    )
    native_out = native.forward(input_ids)
    wrapper_out = wrapper.forward(input_ids)
    assert torch.equal(native_out, wrapper_out), (
        f"wrapper diverged from native NeuralVMEmbedding by max "
        f"{float((native_out - wrapper_out).abs().max())}"
    )

    # Sanity: the ADDR_KEY *delta* (augmented minus base) is nonzero for
    # code-byte positions and identically zero at/after CODE_END. The base
    # embedding lookup also lives on those columns so we cannot inspect the
    # absolute band — only the additive contribution from the augmentation.
    ak = dim_positions["ADDR_KEY"]
    base = native.embed(input_ids)
    delta = (wrapper_out - base)[0, :, ak:ak + 48]
    assert delta[1].abs().sum() > 0, "first code byte ADDR_KEY delta was empty"
    assert delta[40:].abs().sum() == 0, "ADDR_KEY leaked past CODE_END"
    # And the one-hot count at each valid code-byte position is exactly 3
    # (lo + hi + top nibble).
    assert torch.allclose(delta[1].sum(), torch.tensor(3.0))


# ---------------------------------------------------------------------------
# Test 2: MEM_STORE + MEM_ADDR_SRC injection matches the native code path.
# ---------------------------------------------------------------------------


def test_mem_store_wrapper_matches_native_with_positions():
    """MEM_STORE / MEM_ADDR_SRC writes track set_mem_store_positions."""

    torch.manual_seed(1)
    vocab_size, d_model, seq_len = 276, 96, 32
    dim_positions = _addr_key_dim_positions(
        addr_key_idx=8, mem_store_idx=72, mem_addr_src_idx=73
    )

    native = NeuralVMEmbedding(vocab_size, d_model, dim_positions=dim_positions,
                                max_seq_len=seq_len)
    wrapper = NeuralVMEmbeddingWrapper(
        native.embed,
        addr_key_idx=dim_positions["ADDR_KEY"],
        mem_store_idx=dim_positions["MEM_STORE"],
        mem_addr_src_idx=dim_positions["MEM_ADDR_SRC"],
    )

    mem_positions = [10, 20, 26]
    input_ids = _seed_token_grid(
        seq_len=seq_len, code_end_pos=8, mem_positions=mem_positions
    )

    # Track all MEM positions, but only 20 gets MEM_ADDR_SRC=1.
    native.set_mem_store_positions([list(mem_positions)])
    native.set_mem_addr_src_positions([[20]])
    wrapper.set_mem_store_positions([list(mem_positions)])
    wrapper.set_mem_addr_src_positions([[20]])

    native_out = native.forward(input_ids)
    wrapper_out = wrapper.forward(input_ids)
    assert torch.equal(native_out, wrapper_out), (
        f"MEM_STORE wrapper diverged: max |Δ| = "
        f"{float((native_out - wrapper_out).abs().max())}"
    )

    # Sanity asserts on the produced flags. MEM_STORE / MEM_ADDR_SRC are
    # ``index_put_`` (overwriting set), not additive, so the wrapper output
    # column equals exactly 1.0 at the target positions and the base
    # embedding value elsewhere.
    ms = dim_positions["MEM_STORE"]
    mas = dim_positions["MEM_ADDR_SRC"]
    for p in mem_positions:
        assert wrapper_out[0, p, ms] == 1.0, f"MEM_STORE not set at pos {p}"
    assert wrapper_out[0, 20, mas] == 1.0, "MEM_ADDR_SRC missing at pos 20"
    # Pos 10 / 26 keep the base embedding value (not set to 1.0).
    base = native.embed(input_ids)
    assert wrapper_out[0, 10, mas] == base[0, 10, mas]
    assert wrapper_out[0, 26, mas] == base[0, 26, mas]


# ---------------------------------------------------------------------------
# Test 3: mem_history_end fall-back matches the native behaviour.
# ---------------------------------------------------------------------------


def test_mem_history_end_writes_both_flags():
    """When positions list is None and history_end>0, both flags fire."""

    torch.manual_seed(2)
    vocab_size, d_model, seq_len = 276, 80, 16
    dim_positions = _addr_key_dim_positions(
        addr_key_idx=4, mem_store_idx=60, mem_addr_src_idx=61
    )

    native = NeuralVMEmbedding(vocab_size, d_model, dim_positions=dim_positions,
                                max_seq_len=seq_len)
    wrapper = NeuralVMEmbeddingWrapper(
        native.embed,
        addr_key_idx=dim_positions["ADDR_KEY"],
        mem_store_idx=dim_positions["MEM_STORE"],
        mem_addr_src_idx=dim_positions["MEM_ADDR_SRC"],
    )

    input_ids = _seed_token_grid(
        seq_len=seq_len, code_end_pos=5, mem_positions=[7, 11]
    )
    native.set_mem_history_end(12)
    wrapper.set_mem_history_end(12)

    native_out = native.forward(input_ids)
    wrapper_out = wrapper.forward(input_ids)
    assert torch.equal(native_out, wrapper_out)

    ms = dim_positions["MEM_STORE"]
    mas = dim_positions["MEM_ADDR_SRC"]
    # Both flags set at MEM positions within history; outside history (>=12)
    # not touched (although mem position 11 is < 12 so it gets both).
    for p in (7, 11):
        assert wrapper_out[0, p, ms] == 1.0
        assert wrapper_out[0, p, mas] == 1.0


# ---------------------------------------------------------------------------
# Test 4: optional dim positions (addr_key_idx=None) skip ADDR_KEY safely.
# ---------------------------------------------------------------------------


def test_apply_with_none_addr_key_idx_is_passthrough_on_addr_band():
    """When addr_key_idx is None, only base embedding is returned (no ADDR_KEY)."""

    torch.manual_seed(3)
    vocab_size, d_model, seq_len = 276, 32, 10
    embed = torch.nn.Embedding(vocab_size, d_model)
    input_ids = _seed_token_grid(
        seq_len=seq_len, code_end_pos=4, mem_positions=[6]
    )
    base = embed(input_ids)
    out = apply_neural_vm_embedding_augmentations(
        base,
        input_ids,
        addr_key_idx=None,
        mem_store_idx=None,
        mem_addr_src_idx=None,
    )
    assert torch.equal(base, out), (
        "apply with all augmentations disabled should be a no-op"
    )


# ---------------------------------------------------------------------------
# Test 5: Qwen3DenseConfig carries the new c4_addr_key_* / c4_mem_* fields.
# ---------------------------------------------------------------------------


def test_qwen3_dense_config_carries_embedding_metadata():
    """Qwen3DenseConfig advertises the augmentation indices via c4_* fields."""

    cfg = Qwen3DenseConfig(
        vocab_size=277,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        c4_norm_compensator_idx=0,
        c4_bias_compensator_idx=1,
        c4_addr_key_idx=16,
        c4_mem_store_idx=72,
        c4_mem_addr_src_idx=73,
        c4_code_end_token_id=int(Token.CODE_END),
        c4_mem_token_id=int(Token.MEM),
    )
    from dataclasses import asdict

    payload = asdict(cfg)
    assert payload["c4_addr_key_idx"] == 16
    assert payload["c4_addr_key_width"] == 48
    assert payload["c4_mem_store_idx"] == 72
    assert payload["c4_mem_addr_src_idx"] == 73
    assert payload["c4_code_end_token_id"] == int(Token.CODE_END)
    assert payload["c4_mem_token_id"] == int(Token.MEM)
    assert payload["c4_addr_key_pc_offset"] == PC_OFFSET
    assert payload["c4_addr_key_instr_width"] == INSTR_WIDTH

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for k in (
            "c4_addr_key_idx",
            "c4_mem_store_idx",
            "c4_mem_addr_src_idx",
            "c4_code_end_token_id",
            "c4_mem_token_id",
        ):
            assert loaded[k] == payload[k], f"{k} did not round-trip"


# ---------------------------------------------------------------------------
# Test 6: NeuralVMEmbeddingWrapper.from_config reconstructs from a dict.
# ---------------------------------------------------------------------------


def test_wrapper_from_config_reconstructs_runtime_state():
    """from_config builds an equivalent wrapper from a config dict."""

    torch.manual_seed(4)
    vocab_size, d_model, seq_len = 276, 64, 24
    dim_positions = _addr_key_dim_positions(
        addr_key_idx=12, mem_store_idx=56, mem_addr_src_idx=57
    )
    native = NeuralVMEmbedding(vocab_size, d_model, dim_positions=dim_positions,
                                max_seq_len=seq_len)

    cfg_dict = {
        "c4_addr_key_idx": dim_positions["ADDR_KEY"],
        "c4_addr_key_width": 48,
        "c4_mem_store_idx": dim_positions["MEM_STORE"],
        "c4_mem_addr_src_idx": dim_positions["MEM_ADDR_SRC"],
        "c4_code_end_token_id": int(Token.CODE_END),
        "c4_mem_token_id": int(Token.MEM),
        "c4_addr_key_pc_offset": PC_OFFSET,
        "c4_addr_key_instr_width": INSTR_WIDTH,
        "c4_addr_key_data_bytes": 5,
    }
    wrapper = NeuralVMEmbeddingWrapper.from_config(native.embed, cfg_dict)

    input_ids = _seed_token_grid(
        seq_len=seq_len, code_end_pos=18, mem_positions=[20]
    )
    native.set_mem_store_positions([[20]])
    wrapper.set_mem_store_positions([[20]])
    native.set_mem_addr_src_positions([[20]])
    wrapper.set_mem_addr_src_positions([[20]])

    assert torch.equal(native.forward(input_ids), wrapper.forward(input_ids))


# ---------------------------------------------------------------------------
# Test 7: export_qwen3_dense writes the new c4_* fields into config.json.
# ---------------------------------------------------------------------------


def test_export_qwen3_dense_records_embedding_indices_in_config_json():
    """The exported config.json round-trips ADDR_KEY/MEM_STORE indices."""

    from neural_vm.qwen_compat import export_qwen3_dense

    NORM_COMPENSATOR_K = 1000.0
    d_model = 32
    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=d_model,
        n_layers=1,
        n_heads=4,
        ffn_hidden=32,
        max_seq_len=64,
        positional_encoding="rope",
        attention_normalization="softmax",
        use_rms_norm=True,
        use_flash_attention=False,
    )
    # The tiny VM uses a small dim_positions layout for the R8 forward parity
    # tests; we override it here with realistic ADDR_KEY / MEM_STORE slots so
    # the export records them in the config.
    vm.dim_positions = {
        "NORM_COMPENSATOR": 0,
        "CONST": 1,
        "ADDR_KEY": 4,
        "MEM_STORE": 28,
        "MEM_ADDR_SRC": 29,
    }
    with torch.no_grad():
        vm.embed.embed.weight[:, 0] = NORM_COMPENSATOR_K
        # Defensive-zero W_o/W_down rows on the compensator slot so the
        # R1 invariant holds through the export.
        for block in vm.blocks:
            block.attn.W_o.data[0, :] = 0.0
            block.ffn.W_down.data[0, :] = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        with open(os.path.join(tmp, "config.json"), "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    assert loaded["c4_addr_key_idx"] == 4
    assert loaded["c4_addr_key_width"] == 48
    assert loaded["c4_mem_store_idx"] == 28
    assert loaded["c4_mem_addr_src_idx"] == 29
    assert loaded["c4_code_end_token_id"] == int(Token.CODE_END)
    assert loaded["c4_mem_token_id"] == int(Token.MEM)
    assert loaded["c4_addr_key_pc_offset"] == PC_OFFSET
    assert loaded["c4_addr_key_instr_width"] == INSTR_WIDTH


# ---------------------------------------------------------------------------
# Test 8: install_neural_vm_embedding_wrapper swaps the HF embed_tokens.
# ---------------------------------------------------------------------------


def test_install_neural_vm_embedding_wrapper_attaches_to_hf_qwen3():
    """The runtime helper rewires HF embed_tokens so ADDR_KEY actually applies.

    Pins the gap that R8 step 8 hit: ``AutoModelForCausalLM.from_pretrained``
    rebuilds ``model.embed_tokens`` as a plain ``nn.Embedding`` and the
    ADDR_KEY scatter / MEM_STORE writes silently drop. After
    :func:`install_neural_vm_embedding_wrapper` swaps the wrapper in, the
    qmodel forward path emits embeddings byte-identical to
    ``NeuralVMEmbedding.forward`` on the same inputs.
    """

    transformers = pytest.importorskip("transformers")
    from neural_vm.qwen_compat import (
        export_qwen3_dense,
        install_neural_vm_embedding_wrapper,
    )

    NORM_COMPENSATOR_K = 1000.0
    torch.manual_seed(7)
    d_model = 64
    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=d_model,
        n_layers=1,
        n_heads=4,
        ffn_hidden=32,
        max_seq_len=64,
        positional_encoding="rope",
        attention_normalization="softmax",
        use_rms_norm=True,
        use_flash_attention=False,
    )
    vm.dim_positions = {
        "NORM_COMPENSATOR": 0,
        "CONST": 1,
        "ADDR_KEY": 8,
        "MEM_STORE": 56,
        "MEM_ADDR_SRC": 57,
    }
    # NeuralVMEmbedding reads dim_positions through its own attribute, not
    # via the VM. Mirror the same layout so the native augmentation lands
    # on the matching slot.
    vm.embed._dim_positions = vm.dim_positions
    with torch.no_grad():
        vm.embed.embed.weight[:, 0] = NORM_COMPENSATOR_K
        for block in vm.blocks:
            block.attn.W_o.data[0, :] = 0.0
            block.ffn.W_down.data[0, :] = 0.0
    vm.eval()

    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        from transformers import AutoModelForCausalLM
        qmodel = AutoModelForCausalLM.from_pretrained(tmp)
    qmodel.eval()

    # Build a sequence carrying CODE_END and a couple of MEM markers so
    # both ADDR_KEY and MEM_STORE augmentations have something to do.
    input_ids = _seed_token_grid(
        seq_len=20, code_end_pos=12, mem_positions=[14, 16]
    )

    # Pre-swap: HF embed lookup misses the ADDR_KEY band by construction.
    with torch.no_grad():
        plain_qwen = qmodel.model.embed_tokens(input_ids)
        native = vm.embed(input_ids)
    ak = qmodel.config.c4_addr_key_idx
    assert ak is not None
    # ADDR_KEY band diverges on the plain HF embedding (the bug).
    assert not torch.equal(
        plain_qwen[:, :, ak: ak + 48], native[:, :, ak: ak + 48]
    )

    # Install the wrapper and re-run the forward path used by HF.
    wrapper = install_neural_vm_embedding_wrapper(qmodel, qmodel.config)
    assert qmodel.model.embed_tokens is wrapper
    with torch.no_grad():
        wrapped = qmodel.model.embed_tokens(input_ids)
    assert torch.equal(native, wrapped), (
        f"wrapper-installed HF embed_tokens diverged from native "
        f"NeuralVMEmbedding by max |Δ| = "
        f"{float((native - wrapped).abs().max())}"
    )

    # Idempotent: calling twice returns the same wrapper.
    again = install_neural_vm_embedding_wrapper(qmodel, qmodel.config)
    assert again is wrapper
