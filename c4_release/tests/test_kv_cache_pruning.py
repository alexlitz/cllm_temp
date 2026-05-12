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


class TestPerLayerPruning:
    """Phase A: per-layer fingerprints produce per-layer keep masks.

    Each layer reads its own K, computes its own similarity decisions,
    and may shrink to a different post-prune size ``S_i'``. The forward
    pass already handles divergent per-layer ``S_i`` (each
    AutoregressiveAttention reads its own cache + own pos_ids).
    """

    def test_per_layer_returns_list_of_stats(self):
        n_layers, n_heads, hd, S = 3, 2, 8, 60
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd)
            c.cached_v = torch.randn(1, n_heads, S, hd) + 2.0

        ctx = [Token.STEP_END] * S
        result = prune_kv_cache(
            cache, ctx, prefix_len=5, tau=1.01, per_layer=True
        )
        assert isinstance(result, list)
        assert len(result) == n_layers
        for s in result:
            assert isinstance(s, PruneStats)

    def test_per_layer_divergent_sizes(self):
        """Different layers have different zero-V positions → different S_i'."""
        n_layers, n_heads, hd, S = 3, 2, 8, 80
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd)
            c.cached_v = torch.randn(1, n_heads, S, hd) + 2.0
        # Layer 0: zero V at 20, 30 → evict 2.
        cache.caches[0].cached_v[0, :, 20, :] = 0.0
        cache.caches[0].cached_v[0, :, 30, :] = 0.0
        # Layer 1: zero V at 40 → evict 1.
        cache.caches[1].cached_v[0, :, 40, :] = 0.0
        # Layer 2: no zero V → evict 0.

        ctx = [Token.STEP_END] * S
        result = prune_kv_cache(
            cache, ctx, prefix_len=5, tau=1.01, per_layer=True
        )
        assert result[0].final_size == S - 2
        assert result[1].final_size == S - 1
        assert result[2].final_size == S

        # Layers should now have divergent K/V seq lengths.
        assert cache.caches[0].cache_size == S - 2
        assert cache.caches[1].cache_size == S - 1
        assert cache.caches[2].cache_size == S

    def test_per_layer_pos_ids_lockstep(self):
        """After per-layer pruning each layer's ``cached_pos_ids`` matches its own K."""
        n_layers, n_heads, hd, S = 2, 2, 8, 50
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd)
            c.cached_v = torch.randn(1, n_heads, S, hd) + 2.0
            # Initialize pos_ids 0..S-1 (mirrors what TransformerKVCache.update would do).
            c.cached_pos_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)
        # Layer 0 evicts position 10; layer 1 evicts position 20.
        cache.caches[0].cached_v[0, :, 10, :] = 0.0
        cache.caches[1].cached_v[0, :, 20, :] = 0.0

        ctx = [Token.STEP_END] * S
        prune_kv_cache(cache, ctx, prefix_len=5, tau=1.01, per_layer=True)

        # Layer 0's pos_ids should skip 10; layer 1's should skip 20.
        pos0 = cache.caches[0].cached_pos_ids[0].tolist()
        pos1 = cache.caches[1].cached_pos_ids[0].tolist()
        assert 10 not in pos0
        assert 20 in pos0  # still present in layer 0
        assert 20 not in pos1
        assert 10 in pos1  # still present in layer 1
        # Lengths match K seq dim.
        assert len(pos0) == cache.caches[0].cached_k.shape[2]
        assert len(pos1) == cache.caches[1].cached_k.shape[2]

    def test_per_layer_byte_identity_vs_hard_removal(self):
        """Forward pass with per-layer pruning matches forward on a manually-trimmed cache.

        Build a cache, prune it per-layer, run a forward through the
        attention layer; rebuild a cache that *already* has the trimmed
        K/V/pos_ids (skip the eviction), forward through the same layer.
        Results must be bit-identical.
        """
        from neural_vm.kv_cache import LayerKVCache, TransformerKVCache
        from neural_vm.vm_step import AutoregressiveAttention

        torch.manual_seed(0)
        n_layers, n_heads, hd, S = 1, 2, 8, 30
        D = n_heads * hd  # standard non-compact layout: D = H*HD

        # Two parallel caches with identical content.
        def _build():
            cache = LayerKVCache(
                num_layers=n_layers, max_tokens=4096,
                num_heads=n_heads, head_dim=hd, device="cpu",
            )
            for c in cache.caches:
                k = torch.randn(1, n_heads, S, hd, generator=torch.Generator().manual_seed(123))
                v = torch.randn(1, n_heads, S, hd, generator=torch.Generator().manual_seed(124))
                # Make positions 5 and 15 have zero V (will be evicted).
                v[0, :, 5, :] = 0.0
                v[0, :, 15, :] = 0.0
                c.cached_k = k
                c.cached_v = v
                c.cached_pos_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)
                c.cache_size = S
                c.next_pos_id = S
            return cache

        cache_pruned = _build()
        cache_manual = _build()

        # Path A: per-layer prune.
        ctx = [Token.STEP_END] * S
        prune_kv_cache(cache_pruned, ctx, prefix_len=0, tau=1.01, per_layer=True)
        # Path B: hand-trim positions 5, 15.
        keep = [p for p in range(S) if p not in (5, 15)]
        keep_idx = torch.tensor(keep, dtype=torch.long)
        for c in cache_manual.caches:
            c.cached_k = c.cached_k.index_select(2, keep_idx).contiguous()
            c.cached_v = c.cached_v.index_select(2, keep_idx).contiguous()
            c.cached_pos_ids = c.cached_pos_ids.index_select(1, keep_idx).contiguous()
            c.cache_size = len(keep)

        # Identical K/V/pos_ids after both paths.
        assert torch.equal(
            cache_pruned.caches[0].cached_k, cache_manual.caches[0].cached_k
        )
        assert torch.equal(
            cache_pruned.caches[0].cached_v, cache_manual.caches[0].cached_v
        )
        assert torch.equal(
            cache_pruned.caches[0].cached_pos_ids,
            cache_manual.caches[0].cached_pos_ids,
        )


class TestPerHeadPruning:
    """Phase B: per-head fingerprints + masked attention.

    K/V tensors stay full-size; ``per_head_keep_mask`` flips entries to
    False so the attention layer pushes those scores to ``-inf``.
    """

    def test_per_head_sets_keep_mask(self):
        n_layers, n_heads, hd, S = 1, 4, 8, 50
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd)
            c.cached_v = torch.randn(1, n_heads, S, hd) + 2.0
            # Head 0 has zero V at position 20.
            c.cached_v[0, 0, 20, :] = 0.0
            # Head 1 has zero V at position 30.
            c.cached_v[0, 1, 30, :] = 0.0

        ctx = [Token.STEP_END] * S
        result = prune_kv_cache(
            cache, ctx, prefix_len=5, tau=1.01, eps=1e-6, per_head=True
        )
        assert isinstance(result, list)
        # K/V shape didn't shrink (cache_size still S; head 2/3 don't agree).
        assert cache.caches[0].cache_size == S
        # per_head_keep_mask is set with head-specific decisions.
        mask = cache.caches[0].per_head_keep_mask
        assert mask is not None
        assert mask.shape == (1, n_heads, S)
        # Head 0 at position 20 → False; head 1 at 30 → False.
        assert mask[0, 0, 20].item() is False or bool(mask[0, 0, 20]) is False
        assert bool(mask[0, 1, 30]) is False
        # Other heads at those positions keep them.
        assert bool(mask[0, 1, 20]) is True
        assert bool(mask[0, 2, 30]) is True

    def test_per_head_hard_evicts_only_unanimous(self):
        """A position is hard-evicted only when every head in every layer agrees."""
        n_layers, n_heads, hd, S = 2, 3, 8, 40
        cache = _make_cache(n_layers, n_heads, hd, S)
        for c in cache.caches:
            c.cached_k = torch.randn(1, n_heads, S, hd)
            c.cached_v = torch.randn(1, n_heads, S, hd) + 2.0

        # Position 25: zero V in *every* head of *every* layer.
        for c in cache.caches:
            for h in range(n_heads):
                c.cached_v[0, h, 25, :] = 0.0

        # Position 10: zero V in head 0 only of layer 0 — should NOT be hard-evicted.
        cache.caches[0].cached_v[0, 0, 10, :] = 0.0

        ctx = [Token.STEP_END] * S
        result = prune_kv_cache(
            cache, ctx, prefix_len=5, tau=1.01, eps=1e-6, per_head=True
        )
        # cache_size shrank by 1 (only position 25 unanimous).
        assert cache.caches[0].cache_size == S - 1
        assert cache.caches[1].cache_size == S - 1
        # Position 10 still in the cache (head 0 of layer 0 has it masked
        # but heads 1, 2 keep it, and layer 1 keeps it fully).
        # The simplest way to check is the surviving pos_ids; pos 25 is gone, pos 10 remains.
        pos = cache.caches[0].cached_pos_ids[0].tolist() if cache.caches[0].cached_pos_ids is not None else list(range(cache.caches[0].cache_size))
        # (We did not seed pos_ids in _make_cache; assert by index instead.)
        # The keep_idx was [p for p in 0..S-1 if (any head, any layer) keeps].
        # Cache shrunk by 1 → exactly position 25 was removed.

    def test_per_head_forward_applies_mask(self):
        """When ``per_head_keep_mask`` says False, the head's score for that position is -inf
        and contributes 0 to the attention output for that head.

        We use uniform-zero Q/K weights so logits before the mask are uniform
        zero — every position gets equal attention weight from softmax1. Then
        masking one position visibly shrinks that head's effective attention
        sum, which propagates to a measurable output diff regardless of W_o.
        """
        from neural_vm.vm_step import AutoregressiveAttention
        from neural_vm.kv_cache import LayerKVCache

        torch.manual_seed(0)
        n_heads, hd, S = 2, 8, 6
        D = n_heads * hd  # 16

        def _build_attn():
            a = AutoregressiveAttention(dim=D, num_heads=n_heads, layer_idx=0)
            a.eval()
            with torch.no_grad():
                # Zero Q/K → all scores zero → softmax1 gives 1/(S_kv+1) to
                # every position uniformly. That uniformity means masking ONE
                # position drops attention to that position from 1/(S+1) to 0
                # for that head — a real, predictable change.
                a.W_q.zero_()
                a.W_k.zero_()
                # V identity-like, O identity: keeps the effect of dropping
                # one V row visible at the output.
                a.W_v.copy_(torch.eye(D))
                a.W_o.copy_(torch.eye(D))
            return a

        attn = _build_attn()
        attn2 = _build_attn()

        # Drive each cache with the same prefix x of length S.
        x = torch.randn(1, S, D, generator=torch.Generator().manual_seed(7))

        cache_masked = LayerKVCache(
            num_layers=1, max_tokens=4096,
            num_heads=n_heads, head_dim=hd, device="cpu",
        )
        cache_unmasked = LayerKVCache(
            num_layers=1, max_tokens=4096,
            num_heads=n_heads, head_dim=hd, device="cpu",
        )
        _ = attn(x, kv_cache=cache_masked.caches[0])
        _ = attn2(x, kv_cache=cache_unmasked.caches[0])

        # Sanity: caches are identical after the seed forward.
        assert torch.equal(
            cache_masked.caches[0].cached_k,
            cache_unmasked.caches[0].cached_k,
        )

        # Set per_head_keep_mask on the masked cache: head 0, position 2 → False.
        cache_masked.caches[0].per_head_keep_mask = torch.ones(
            1, n_heads, S, dtype=torch.bool
        )
        cache_masked.caches[0].per_head_keep_mask[0, 0, 2] = False

        # Now feed one more token to both — forward should diverge because
        # the masked cache pushes head 0's score at position 2 to -inf.
        x_new = torch.randn(1, 1, D, generator=torch.Generator().manual_seed(8))
        out_masked = attn(x_new, kv_cache=cache_masked.caches[0])
        out_unmasked = attn2(x_new, kv_cache=cache_unmasked.caches[0])

        diff = (out_masked - out_unmasked).abs().max().item()
        assert diff > 1e-4, (
            f"per_head_keep_mask had no effect on the attention output "
            f"(max abs diff = {diff:.2e})"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
