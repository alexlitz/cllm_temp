"""R8 Blocker 2 — softmax1 sink wired into the Qwen3 export.

Validates the R3 math at *export* time: the sink-token row appended to
``model.embed_tokens.weight`` causes K = V = 0 at the sink position when
the loader (or :func:`prepend_softmax_sink`) materialises the sink token
at column 0 of ``input_ids``. Standard Qwen3 softmax over the augmented
sequence then reproduces our model's softmax1 over the real positions.

The math (recapped from
``docs/QWEN_SOFTMAX_SINK_PROTOTYPE_2026_06_07.md``):

    softmax1(s)_i = exp(s_i) / (1 + Σ_j exp(s_j))
    softmax([0, s])_i = exp(s_i) / (exp(0) + Σ_j exp(s_j))   (for i ≥ 1)
                     = exp(s_i) / (1 + Σ_j exp(s_j))
                     = softmax1(s)_i.

With ``V_sink = 0`` the sink contributes nothing to attention(V_real).

This test exercises a tiny exported model end-to-end:

1. Build a tiny VM, export with the sink.
2. Reload through HF ``AutoModelForCausalLM``.
3. Probe one block's ``self_attn``: prepend the sink token id and verify
   the K/V rows at the sink position are byte-zero.
4. Compare standard softmax over (sink + real) to softmax1 over (real)
   on the same Q, K, V — assert allclose at fp32 noise floor.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

transformers = pytest.importorskip("transformers")

import torch

from neural_vm.qwen_compat import export_qwen3_dense, prepend_softmax_sink
from neural_vm.vm_step import AutoregressiveVM


NORM_COMPENSATOR_K = 1000.0


def _build_tiny_qwen_compatible_vm(
    *,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    ffn_hidden: int = 64,
) -> AutoregressiveVM:
    """Tiny VM that exports cleanly through R6.

    Mirrors the fixture in ``test_qwen_r8_e2e.py`` so the two tests
    pin compatible behaviour.
    """

    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=128,
        positional_encoding="rope",
        attention_normalization="softmax",
        use_rms_norm=True,
        use_flash_attention=False,
    )
    vm.dim_positions = {"NORM_COMPENSATOR": 0, "CONST": 1}
    with torch.no_grad():
        vm.embed.embed.weight[:, 0] = NORM_COMPENSATOR_K
        for block in vm.blocks:
            block.attn.W_o.data[0, :] = 0.0
            block.ffn.W_down.data[0, :] = 0.0
            block.ffn.b_up.data.fill_(0.1)
            block.ffn.b_gate.data.fill_(0.05)
    return vm


def _softmax1(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax1(s)_i = exp(s_i) / (1 + Σ_j exp(s_j)), numerically stable."""

    m = torch.maximum(scores.max(dim=dim, keepdim=True).values, torch.zeros_like(scores).max(dim=dim, keepdim=True).values)
    e = torch.exp(scores - m)
    e_zero = torch.exp(-m).squeeze(dim)
    denom = e.sum(dim=dim) + e_zero
    return e / denom.unsqueeze(dim)


# ---------------------------------------------------------------------------
# Export carries a sink-token id and the embedding row is zero
# ---------------------------------------------------------------------------


def test_export_records_sink_token_id_and_zero_embedding_row():
    """Config has c4_softmax_sink_token_id; embedding row is all zeros."""

    import json

    vm = _build_tiny_qwen_compatible_vm()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        with open(os.path.join(tmp, "config.json"), encoding="utf-8") as fh:
            cfg_json = json.load(fh)
        sd = torch.load(
            os.path.join(tmp, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )

    sink_id = cfg.c4_softmax_sink_token_id
    assert sink_id is not None
    assert sink_id == vm.vocab_size
    assert cfg_json["c4_softmax_sink_token_id"] == sink_id
    assert cfg_json["c4_softmax_sink_added"] is True
    assert cfg_json["vocab_size"] == vm.vocab_size + 1

    embed = sd["model.embed_tokens.weight"]
    assert embed.shape[0] == vm.vocab_size + 1
    sink_row = embed[sink_id]
    assert torch.all(sink_row == 0), (
        f"sink row must be all zeros so K = V = 0 (max abs="
        f"{float(sink_row.abs().max())})"
    )
    head = sd["lm_head.weight"]
    assert head.shape[0] == vm.vocab_size + 1
    assert torch.all(head[sink_id] == 0), (
        "sink row of lm_head must be zero so the model never predicts the "
        "sink token."
    )


# ---------------------------------------------------------------------------
# Helper: prepend_softmax_sink shape contract
# ---------------------------------------------------------------------------


def test_prepend_softmax_sink_adds_one_column():
    sink_id = 276
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = prepend_softmax_sink(input_ids, sink_id)
    assert out.shape == (2, 4)
    assert torch.all(out[:, 0] == sink_id)
    assert torch.all(out[:, 1:] == input_ids)


# ---------------------------------------------------------------------------
# Core acceptance: softmax-over-sink at one block reproduces softmax1
# ---------------------------------------------------------------------------


def test_block0_k_and_v_at_sink_position_are_zero():
    """Loaded Qwen3 layer 0: sink-token K and V projections are exactly 0."""

    from transformers import AutoModelForCausalLM

    vm = _build_tiny_qwen_compatible_vm()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        qmodel = AutoModelForCausalLM.from_pretrained(tmp)
    qmodel.eval()

    sink_id = cfg.c4_softmax_sink_token_id
    input_ids = torch.tensor([[sink_id, 17, 42, 200]], dtype=torch.long)

    with torch.no_grad():
        h = qmodel.model.embed_tokens(input_ids)

    # Block 0 sees the raw embedding through input_layernorm first; with
    # gamma = K/sqrt(d_model) on a compensator-bearing residual the
    # RMSNorm collapses to ~identity on real tokens. For the sink the
    # residual is the zero vector, so RMSNorm(0) = 0 and k_proj(0) = 0,
    # v_proj(0) = 0 (attention_bias=False).
    layer = qmodel.model.layers[0]
    with torch.no_grad():
        normed = layer.input_layernorm(h)
        k = layer.self_attn.k_proj(normed)
        v = layer.self_attn.v_proj(normed)

    # The sink position (column 0) must be exactly zero.
    assert torch.all(k[:, 0, :] == 0), (
        f"k_proj at sink position has non-zero values; max abs="
        f"{float(k[:, 0, :].abs().max())}"
    )
    assert torch.all(v[:, 0, :] == 0), (
        f"v_proj at sink position has non-zero values; max abs="
        f"{float(v[:, 0, :].abs().max())}"
    )


def test_softmax_over_sink_equals_softmax1_on_real_positions():
    """At one block: softmax(sink + real) == softmax1(real). Byte-identity gate.

    Uses the exported model's actual K/V projections to compute attention
    weights two ways:

      A. Prepend the sink to input_ids; run k_proj/v_proj; compute
         standard softmax over (sink + real).
      B. Drop the sink column from input_ids; run the same k_proj/v_proj;
         compute softmax1 over (real only).

    Assert weights agree to within fp32 noise (atol=1e-6, rtol=1e-5).

    This is the R3 prototype gate (max abs error 3e-8 in the
    ``weights_real_positions`` row of
    docs/QWEN_SOFTMAX_SINK_PROTOTYPE_2026_06_07.md) re-run against the
    exported model's weights.
    """

    from transformers import AutoModelForCausalLM

    vm = _build_tiny_qwen_compatible_vm()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        qmodel = AutoModelForCausalLM.from_pretrained(tmp)
    qmodel.eval()

    sink_id = cfg.c4_softmax_sink_token_id
    torch.manual_seed(0)
    real_ids = torch.randint(0, vm.vocab_size, (1, 8))
    input_ids_with_sink = prepend_softmax_sink(real_ids, sink_id)

    layer = qmodel.model.layers[0]
    attn = layer.self_attn

    with torch.no_grad():
        # A. With sink.
        h_aug = qmodel.model.embed_tokens(input_ids_with_sink)
        normed_aug = layer.input_layernorm(h_aug)
        n_heads = attn.config.num_attention_heads
        head_dim = attn.head_dim
        hidden_shape = (*input_ids_with_sink.shape, n_heads, head_dim)
        q_aug = attn.q_norm(attn.q_proj(normed_aug).view(hidden_shape)).transpose(1, 2)
        k_aug = attn.k_norm(attn.k_proj(normed_aug).view(hidden_shape)).transpose(1, 2)
        # Skip RoPE for this comparison — the sink contributes via the
        # zero K, not via positional rotation. The real-position Q and K
        # are perturbed by RoPE in both branches consistently if we apply
        # the same RoPE to both. The R3 math is RoPE-agnostic.
        scores_aug = torch.matmul(q_aug, k_aug.transpose(-1, -2)) / math.sqrt(head_dim)
        weights_aug = torch.softmax(scores_aug, dim=-1)

        # B. Without sink.
        h_real = qmodel.model.embed_tokens(real_ids)
        normed_real = layer.input_layernorm(h_real)
        hidden_shape_real = (*real_ids.shape, n_heads, head_dim)
        q_real = attn.q_norm(attn.q_proj(normed_real).view(hidden_shape_real)).transpose(1, 2)
        k_real = attn.k_norm(attn.k_proj(normed_real).view(hidden_shape_real)).transpose(1, 2)
        scores_real = torch.matmul(q_real, k_real.transpose(-1, -2)) / math.sqrt(head_dim)
        weights_sm1 = _softmax1(scores_real, dim=-1)

    # The (B) shape is (B, H, T_real, T_real); the (A) shape is
    # (B, H, T_aug, T_aug) where T_aug = T_real + 1. Compare
    # weights_aug[..., 1:, 1:] (queries at real positions, attending to
    # real keys) to weights_sm1.
    weights_aug_real = weights_aug[..., 1:, 1:]
    assert weights_aug_real.shape == weights_sm1.shape, (
        f"shape mismatch: aug_real {tuple(weights_aug_real.shape)} vs "
        f"sm1 {tuple(weights_sm1.shape)}"
    )

    diff = (weights_aug_real - weights_sm1).abs()
    max_abs = float(diff.max())
    assert torch.allclose(weights_aug_real, weights_sm1, atol=1e-6, rtol=1e-5), (
        f"softmax(sink + real) != softmax1(real) at fp32 noise floor; "
        f"max abs err = {max_abs:.3e}"
    )

    # Sink absorbs the residual mass: column 0 of weights_aug per query
    # should equal 1 - Σ_j weights_sm1(query, j_real).
    sink_mass = weights_aug[..., 1:, 0]
    expected_sink = 1.0 - weights_sm1.sum(dim=-1)
    assert torch.allclose(sink_mass, expected_sink, atol=1e-6, rtol=1e-5), (
        f"sink mass != 1 - Σ softmax1; max abs="
        f"{float((sink_mass - expected_sink).abs().max()):.3e}"
    )
