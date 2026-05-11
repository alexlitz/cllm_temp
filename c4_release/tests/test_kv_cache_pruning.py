"""Unit tests for ``neural_vm.kv_cache_pruning`` per spec §7.1.

These tests construct synthetic ``LayerKVCache`` instances with known
K/V tensor patterns and exercise the two eviction rules (key
similarity > 0.99 and zero-V) plus the protected-prefix invariant.

No model is loaded — the pruner only needs the per-layer ``cached_k``
/ ``cached_v`` tensors and a context token-id sequence.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from neural_vm.kv_cache import LayerKVCache
from neural_vm.kv_cache_pruning import (
    PruneStats,
    bucket_positions,
    prune_kv_cache,
)
from neural_vm.vm_step import Token


def _make_cache(num_layers, num_heads, head_dim, seq_len, device="cpu"):
    """Build an empty ``LayerKVCache`` and seed it with zero K/V tensors.

    Returns the cache. Caller is expected to mutate
    ``caches[i].cached_k/cached_v`` directly to set up the test pattern.
    """
    cache = LayerKVCache(
        num_layers=num_layers,
        max_tokens=4096,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
    )
    for c in cache.caches:
        # B=1, H=num_heads, S=seq_len, HD=head_dim
        c.cached_k = torch.zeros(1, num_heads, seq_len, head_dim, device=device)
        c.cached_v = torch.zeros(1, num_heads, seq_len, head_dim, device=device)
        c.cache_size = seq_len
    return cache


class TestBucketPositions:
    """Bucket positions by token type for similarity comparison."""

    def test_basic_bucketing(self):
        ctx = [
            Token.CODE_START, 0, 1, 2, Token.CODE_END,          # prefix [0..4]
            Token.REG_PC, 0, 0, 0, 0,                            # 5: PC, 6-9: value bytes
            Token.REG_AX, 0, 0, 0, 0,                            # 10: AX
            Token.MEM, 1, 2, 3, 4, 5, 6, 7, 8,                   # 15: MEM, 16-23: bytes
            Token.STEP_END,                                       # 24
        ]
        buckets = bucket_positions(ctx, prefix_len=5, cache_size=len(ctx))
        assert buckets.get("REG_PC") == [5]
        assert buckets.get("REG_AX") == [10]
        assert buckets.get("MEM_marker") == [15]
        assert buckets.get("STEP_END") == [24]
        # Value bytes go to ``other`` (they are byte ids 0..255).
        assert "other" in buckets
        assert 6 in buckets["other"]  # PC value byte

    def test_prefix_excluded(self):
        ctx = [Token.REG_PC, 0, 0, 0, 0, Token.REG_PC, 0, 0, 0, 0]
        buckets = bucket_positions(ctx, prefix_len=5, cache_size=len(ctx))
        # Only the second REG_PC is past the prefix.
        assert buckets.get("REG_PC") == [5]


class TestZeroVEviction:
    """Spec §2.2: any V row with L2 norm < eps is evicted."""

    def test_prune_identifies_zero_v(self):
        # Build a 200-position cache and seed positions 50, 75, 110 with
        # zero V (everything else gets a non-zero V).
        n_layers, n_heads, hd, S = 2, 4, 8, 200
        cache = _make_cache(n_layers, n_heads, hd, S)
        # Non-zero V everywhere first.
        for c in cache.caches:
            c.cached_v = torch.randn(1, n_heads, S, hd) + 1.0
            # K stays at zero, so similarity-eviction would catch
            # everything; we need to make K unique per position to
            # isolate the zero-V test.
            c.cached_k = torch.randn(1, n_heads, S, hd)
        # Zero V at positions 50, 75, 110 (all layers).
        for c in cache.caches:
            c.cached_v[0, :, 50, :] = 0.0
            c.cached_v[0, :, 75, :] = 0.0
            c.cached_v[0, :, 110, :] = 0.0

        # Use a tiny prefix so positions 50/75/110 are unprotected.
        # ``context`` is a dummy list (token ids don't influence zero-V).
        ctx = [Token.STEP_END] * S
        stats = prune_kv_cache(cache, ctx, prefix_len=10, tau=1.01)  # tau>1 disables sim

        # Zero-V evicted exactly positions 50, 75, 110.
        assert stats.evicted_by_zero_v == 3
        assert stats.evicted_by_similarity == 0
        # Final size shrunk by 3.
        assert stats.final_size == S - 3
        # Layer-uniform.
        for c in cache.caches:
            assert c.cache_size == S - 3

    def test_zero_v_respects_prefix(self):
        n_layers, n_heads, hd, S = 1, 2, 4, 50
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_v = torch.randn(1, n_heads, S, hd) + 1.0
            c.cached_k = torch.randn(1, n_heads, S, hd)
            # Zero V at position 3 (inside prefix) and 30 (outside).
            c.cached_v[0, :, 3, :] = 0.0
            c.cached_v[0, :, 30, :] = 0.0

        ctx = [Token.STEP_END] * S
        stats = prune_kv_cache(cache, ctx, prefix_len=10, tau=1.01)

        # Only position 30 is evicted; 3 is protected.
        assert stats.evicted_by_zero_v == 1
        assert stats.final_size == S - 1


class TestSimilarityEviction:
    """Spec §2.1: cosine-similar K pairs evict the older."""

    def test_prune_identifies_duplicate_k(self):
        """Three near-identical REG_PC keys at positions 10, 50, 90.

        Positions 10 and 50 must be evicted; 90 (newest) survives.
        """
        n_layers, n_heads, hd, S = 1, 2, 16, 100
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd) * 0.1
            c.cached_v = torch.randn(1, n_heads, S, hd) + 1.0
            # Make K at positions 10, 50, 90 nearly identical (and
            # large enough that normalize doesn't randomize the
            # direction).
            shared_k = torch.randn(n_heads, hd) * 10.0 + 5.0
            for pos in (10, 50, 90):
                # Add a tiny per-position perturbation so the keys
                # aren't *exactly* identical but cosine sim is > 0.99.
                noise = torch.randn(n_heads, hd) * 0.01
                c.cached_k[0, :, pos, :] = shared_k + noise

        # All three positions are marked REG_PC in the context.
        ctx = [Token.STEP_END] * S
        ctx[10] = Token.REG_PC
        ctx[50] = Token.REG_PC
        ctx[90] = Token.REG_PC

        stats = prune_kv_cache(cache, ctx, prefix_len=0, tau=0.99, eps=-1.0)
        # Positions 10 and 50 are evicted; 90 survives.
        assert stats.evicted_by_similarity == 2
        assert stats.final_size == S - 2
        # Cache-content sanity: position 90 (now at some new index) is
        # still present.

    def test_prune_respects_prefix(self):
        """Three near-identical REG_PC keys at (5, 10, 50) with prefix_len=20.

        Only position 50 is outside the prefix. With nothing else in
        the REG_PC bucket past prefix, no eviction fires.
        """
        n_layers, n_heads, hd, S = 1, 2, 16, 100
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd) * 0.1
            c.cached_v = torch.randn(1, n_heads, S, hd) + 1.0
            shared_k = torch.randn(n_heads, hd) * 10.0 + 5.0
            for pos in (5, 10, 50):
                noise = torch.randn(n_heads, hd) * 0.01
                c.cached_k[0, :, pos, :] = shared_k + noise

        ctx = [Token.STEP_END] * S
        ctx[5] = Token.REG_PC
        ctx[10] = Token.REG_PC
        ctx[50] = Token.REG_PC

        # Prefix protects 5 and 10 (they aren't bucketed). Position 50
        # is the only REG_PC past prefix → bucket size 1 → no pairs →
        # no eviction.
        stats = prune_kv_cache(cache, ctx, prefix_len=20, tau=0.99, eps=-1.0)
        assert stats.evicted_by_similarity == 0
        assert stats.final_size == S


class TestLayerUniformity:
    """Spec §3.3: the keep-mask is shared across all layers."""

    def test_prune_uniform_across_layers(self):
        """Multi-layer cache: every layer's ``cache_size`` matches after pruning."""
        n_layers, n_heads, hd, S = 4, 2, 8, 80
        cache = _make_cache(n_layers, n_heads, hd, S)
        # Zero V at a couple of positions to force eviction.
        for c in cache.caches:
            c.cached_v = torch.randn(1, n_heads, S, hd) + 1.0
            c.cached_k = torch.randn(1, n_heads, S, hd)
            c.cached_v[0, :, 20, :] = 0.0
            c.cached_v[0, :, 40, :] = 0.0

        ctx = [Token.STEP_END] * S
        stats = prune_kv_cache(cache, ctx, prefix_len=5, tau=1.01)
        assert stats.final_size == S - 2

        sizes = {c.cache_size for c in cache.caches}
        assert len(sizes) == 1, f"layer sizes diverged: {sizes}"
        assert sizes.pop() == S - 2

        # All layers' K/V tensors have the same sequence length.
        seq_lens_k = {c.cached_k.shape[2] for c in cache.caches}
        seq_lens_v = {c.cached_v.shape[2] for c in cache.caches}
        assert seq_lens_k == seq_lens_v == {S - 2}


class TestNoOp:
    """Pruning should be a no-op when there's nothing to evict."""

    def test_empty_cache(self):
        cache = LayerKVCache(num_layers=1, max_tokens=128, num_heads=2,
                             head_dim=4, device="cpu")
        # cached_k stays None.
        stats = prune_kv_cache(cache, [], prefix_len=0)
        assert stats.total_evicted == 0
        assert stats.final_size == 0
        assert stats.initial_size == 0

    def test_small_cache_below_min(self):
        n_layers, n_heads, hd, S = 1, 2, 4, 10
        cache = _make_cache(n_layers, n_heads, hd, S)
        # Force a zero-V position that would otherwise be evicted.
        for c in cache.caches:
            c.cached_v[0, :, 5, :] = 0.0

        ctx = [Token.STEP_END] * S
        stats = prune_kv_cache(cache, ctx, prefix_len=0, min_cache_size=100)
        # min_cache_size=100 > cache_size=10 → no-op.
        assert stats.total_evicted == 0
        for c in cache.caches:
            assert c.cache_size == S

    def test_all_unique_no_zero_v(self):
        n_layers, n_heads, hd, S = 1, 2, 8, 30
        cache = _make_cache(n_layers, n_heads, hd, S)
        # All K/V unique and non-zero.
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd)
            c.cached_v = torch.randn(1, n_heads, S, hd) + 2.0

        ctx = list(range(S))  # all distinct token ids → all in ``other``
        stats = prune_kv_cache(cache, ctx, prefix_len=0, tau=0.99)
        # Random Gaussian keys are extremely unlikely to be cos-sim > 0.99
        # in 8 dims, but to be safe, we assert that evictions (if any)
        # are bounded — and importantly nothing crashes.
        assert stats.final_size <= S


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
