"""Focused diagnostics for batched neural KV eviction correctness.

These tests avoid DraftVM substitution. They exercise the neural forward
path directly and keep the runtime small by using tiny model/attention
slices where the expected parity condition is explicit.
"""

from __future__ import annotations

import torch

from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
from neural_vm.kv_cache import LayerKVCache, TransformerKVCache
from neural_vm.vm_step import AutoregressiveAttention, AutoregressiveVM, Token


_TINY_DIM_POSITIONS = {
    "ADDR_KEY": 0,
    "MEM_STORE": 48,
    "MEM_ADDR_SRC": 49,
}


def _make_tiny_vm() -> AutoregressiveVM:
    """Build a tiny causal neural model with non-trivial attention weights."""
    torch.manual_seed(1234)
    model = AutoregressiveVM(
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_hidden=8,
        max_seq_len=16,
        dim_positions=_TINY_DIM_POSITIONS,
        use_flash_attention=False,
        positional_encoding="alibi",
        attention_normalization="softmax1",
    )
    model.eval()

    with torch.no_grad():
        model.embed.embed.weight.normal_(mean=0.0, std=0.05)
        model.head.weight.normal_(mean=0.0, std=0.05)
        model.head.bias.zero_()
        for block in model.blocks:
            for weight in (
                block.attn.W_q,
                block.attn.W_k,
                block.attn.W_v,
                block.attn.W_o,
            ):
                weight.normal_(mean=0.0, std=0.04)
    return model


def test_batched_incremental_kv_small_window_matches_fresh_full_forward():
    """Incremental batched KV is identical to a fresh forward when no history
    has been dropped and the cache window is small but sufficient."""
    model = _make_tiny_vm()
    prefix = torch.tensor(
        [
            [Token.CODE_START, 1, 2, Token.CODE_END, Token.REG_PC],
            [Token.CODE_START, 5, 6, Token.CODE_END, Token.REG_PC],
        ],
        dtype=torch.long,
    )
    suffix = torch.tensor(
        [
            [7, 8, Token.STEP_END],
            [9, 10, Token.STEP_END],
        ],
        dtype=torch.long,
    )
    full = torch.cat([prefix, suffix], dim=1)

    kv_cache = LayerKVCache(
        num_layers=len(model.blocks),
        max_tokens=full.shape[1],
        num_heads=model.blocks[0].attn.num_heads,
        head_dim=model.blocks[0].attn.head_dim,
        device="cpu",
    )

    with torch.no_grad():
        fresh_logits = model.forward(full)
        model.forward(prefix, kv_cache=kv_cache)
        cached_logits = model.forward(
            full,
            kv_cache=kv_cache,
            cached_prefix_len=prefix.shape[1],
        )

    torch.testing.assert_close(
        cached_logits,
        fresh_logits[:, prefix.shape[1] :, :],
        rtol=1e-5,
        atol=1e-5,
    )
    for layer_cache in kv_cache.caches:
        assert layer_cache.cache_size == full.shape[1]
        assert layer_cache.stats.tokens_evicted == 0
        assert layer_cache.cached_pos_ids.tolist() == [
            list(range(full.shape[1])),
            list(range(full.shape[1])),
        ]


def _make_identity_attention(dim: int = 16, heads: int = 2) -> AutoregressiveAttention:
    attn = AutoregressiveAttention(
        dim=dim,
        num_heads=heads,
        max_seq_len=16,
        use_flash_attention=False,
        positional_encoding="alibi",
        attention_normalization="softmax1",
    )
    attn.eval()
    with torch.no_grad():
        eye = torch.eye(dim)
        attn.W_q.copy_(eye)
        attn.W_k.copy_(eye)
        attn.W_v.copy_(eye)
        attn.W_o.copy_(eye)
        attn.alibi_slopes.fill_(100.0)
    return attn


def test_batched_kv_eviction_matches_full_forward_when_evicted_prefix_is_inert():
    """A small cache window with actual eviction can still match a full neural
    forward when the dropped prefix cannot affect the attended values."""
    torch.manual_seed(5678)
    attn = _make_identity_attention()
    x = torch.zeros(2, 8, attn.dim)
    x[:, 4:, :] = torch.randn(2, 4, attn.dim)

    cache = TransformerKVCache(
        max_tokens=4,
        num_heads=attn.num_heads,
        head_dim=attn.head_dim,
        device="cpu",
    )

    with torch.no_grad():
        fresh = attn(x)
        attn(x[:, :4, :], kv_cache=cache)
        cached_tail = attn(x[:, 4:, :], kv_cache=cache, x_is_new_only=True)

    assert cache.cache_size == 4
    assert cache.stats.tokens_evicted == 8
    assert cache.cached_pos_ids.tolist() == [[4, 5, 6, 7], [4, 5, 6, 7]]
    torch.testing.assert_close(cached_tail, fresh[:, 4:, :], rtol=1e-5, atol=1e-5)


class _ForwardRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, token_ids, kv_cache=None, cached_prefix_len=0):
        self.calls.append(
            {
                "shape": tuple(token_ids.shape),
                "kv_cache": kv_cache,
                "cached_prefix_len": cached_prefix_len,
            }
        )
        vocab = Token.VOCAB_SIZE
        logits = torch.zeros(
            token_ids.shape[0],
            token_ids.shape[1],
            vocab,
            device=token_ids.device,
        )
        logits[..., Token.STEP_END] = 1.0
        return logits


def test_batched_runner_guard_fresh_forwards_instead_of_silent_mem_drop():
    """If a historical MEM token falls outside the bounded KV window, the
    batched runner must take the fresh path instead of evicting it silently."""
    runner = object.__new__(BatchedPureNeuralRunner)
    runner.model = _ForwardRecorder()
    runner._device = torch.device("cpu")
    runner.use_kv_cache = True
    runner.incremental_kv_safe = True
    runner.kv_cache_max_tokens = 5
    runner.kv_flush_interval = 0
    runner.kv_cache_verify = False
    runner.kv_cache_verify_interval = 1
    runner._kv_cache_obj = LayerKVCache(
        num_layers=1,
        max_tokens=runner.kv_cache_max_tokens,
        num_heads=1,
        head_dim=1,
        device="cpu",
    )
    runner._kv_active_idx = (0, 1)
    runner._kv_cached_rows = [[Token.CODE_START], [Token.CODE_START]]
    runner._kv_incremental_count = 3
    runner._kv_stats = {
        "calls": 0,
        "hits": 0,
        "fallbacks": 0,
        "mismatches": 0,
        "eviction_pressure": 0,
        "verifications": 0,
        "fresh_forwards": 0,
        "kv_forwards": 0,
        "verification_forwards": 0,
        "cache_rebuilds": 0,
        "reused_token_slots": 0,
        "spec_fresh_bypass": 0,
        "unsafe_model_fresh_bypass": 0,
        "bounded_evictions": 0,
        "bounded_positions_evicted": 0,
        "bounded_eviction_fallbacks": 0,
    }
    sequences = [
        [Token.CODE_START, Token.MEM, 1, 2, 3, 4, 5, Token.STEP_END],
        [Token.CODE_START, Token.MEM, 6, 7, 8, 9, 10, Token.STEP_END],
    ]
    assert sequences[0].index(Token.MEM) < len(sequences[0]) - runner.kv_cache_max_tokens

    preds, pred_start, real_lens = runner._forward_argmax_batch(
        sequences,
        active_idx=[0, 1],
        first_logit_pos=6,
        allow_kv=True,
    )

    assert real_lens == [8, 8]
    assert pred_start == 0
    assert preds == [[Token.STEP_END] * 8, [Token.STEP_END] * 8]
    assert runner._kv_cache_obj is None
    assert runner._kv_active_idx is None
    assert runner._kv_cached_rows == []
    assert runner._kv_incremental_count == 0
    assert runner._kv_stats["eviction_pressure"] == 1
    assert runner._kv_stats["fresh_forwards"] == 1
    assert runner._kv_stats["kv_forwards"] == 0
    assert runner._kv_stats["calls"] == 0
    assert runner.model.calls == [
        {"shape": (2, 8), "kv_cache": None, "cached_prefix_len": 0}
    ]
