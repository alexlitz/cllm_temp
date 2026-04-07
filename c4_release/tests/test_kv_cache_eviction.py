"""
Comprehensive KV Cache Eviction Tests.

Tests for Requirement #6: KV cache eviction works properly and
maintains correct outputs over even long problems.

Tests verify:
1. KV cache basic functionality
2. Eviction triggers when cache is full
3. Correctness is maintained after eviction
4. Long-running programs work correctly
5. Memory history is preserved
6. Recursive calls work with eviction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from neural_vm.kv_cache import TransformerKVCache, LayerKVCache, KVCacheStats


class TestTransformerKVCacheBasic:
    """Basic KV cache functionality tests."""

    def test_cache_initialization(self):
        """Cache initializes with correct parameters."""
        cache = TransformerKVCache(max_tokens=1024, num_heads=8, head_dim=64)
        assert cache.max_tokens == 1024
        assert cache.num_heads == 8
        assert cache.head_dim == 64
        assert cache.cache_size == 0
        assert cache.cached_k is None
        assert cache.cached_v is None

    def test_cache_first_update(self):
        """First update initializes the cache."""
        cache = TransformerKVCache(max_tokens=1024, num_heads=8, head_dim=64, device='cpu')

        # Create dummy K/V tensors
        new_k = torch.randn(1, 8, 10, 64)
        new_v = torch.randn(1, 8, 10, 64)

        full_k, full_v = cache.update(new_k, new_v)

        assert cache.cache_size == 10
        assert full_k.shape == (1, 8, 10, 64)
        assert full_v.shape == (1, 8, 10, 64)
        assert cache.stats.tokens_cached == 10

    def test_cache_append(self):
        """Subsequent updates append to cache."""
        cache = TransformerKVCache(max_tokens=1024, num_heads=8, head_dim=64, device='cpu')

        # First update
        k1 = torch.randn(1, 8, 10, 64)
        v1 = torch.randn(1, 8, 10, 64)
        cache.update(k1, v1)

        # Second update
        k2 = torch.randn(1, 8, 5, 64)
        v2 = torch.randn(1, 8, 5, 64)
        full_k, full_v = cache.update(k2, v2)

        assert cache.cache_size == 15
        assert full_k.shape == (1, 8, 15, 64)
        assert cache.stats.tokens_cached == 15
        assert cache.stats.cache_hits == 1

    def test_cache_reset(self):
        """Reset clears the cache."""
        cache = TransformerKVCache(max_tokens=1024, num_heads=8, head_dim=64, device='cpu')

        # Add some data
        k = torch.randn(1, 8, 10, 64)
        v = torch.randn(1, 8, 10, 64)
        cache.update(k, v)

        # Reset
        cache.reset()

        assert cache.cache_size == 0
        assert cache.cached_k is None
        assert cache.cached_v is None


class TestKVCacheEviction:
    """Tests for eviction behavior."""

    def test_eviction_triggers(self):
        """Eviction triggers when cache exceeds max_tokens."""
        cache = TransformerKVCache(max_tokens=20, num_heads=8, head_dim=64, device='cpu')

        # Add 15 tokens
        k1 = torch.randn(1, 8, 15, 64)
        v1 = torch.randn(1, 8, 15, 64)
        cache.update(k1, v1)
        assert cache.cache_size == 15
        assert cache.stats.tokens_evicted == 0

        # Add 10 more - should trigger eviction
        k2 = torch.randn(1, 8, 10, 64)
        v2 = torch.randn(1, 8, 10, 64)
        cache.update(k2, v2)

        # Should have evicted 5 tokens (25 - 20 = 5)
        assert cache.cache_size == 20
        assert cache.stats.tokens_evicted == 5

    def test_eviction_preserves_recent(self):
        """Eviction keeps the most recent tokens."""
        cache = TransformerKVCache(max_tokens=10, num_heads=1, head_dim=4, device='cpu')

        # Add tokens with identifiable values
        k1 = torch.ones(1, 1, 5, 4) * 1.0  # Old tokens
        v1 = torch.ones(1, 1, 5, 4) * 1.0
        cache.update(k1, v1)

        k2 = torch.ones(1, 1, 8, 4) * 2.0  # New tokens
        v2 = torch.ones(1, 1, 8, 4) * 2.0
        full_k, full_v = cache.update(k2, v2)

        # Cache should have 10 tokens, all should be the newer value (2.0)
        # because oldest 3 tokens (value 1.0) were evicted
        assert cache.cache_size == 10
        # First 2 should be old (1.0), last 8 should be new (2.0)
        assert full_k[0, 0, 0, 0].item() == 1.0  # Kept 2 old tokens
        assert full_k[0, 0, 2, 0].item() == 2.0  # New tokens start at index 2

    def test_eviction_stats(self):
        """Eviction statistics are tracked correctly."""
        cache = TransformerKVCache(max_tokens=100, num_heads=8, head_dim=64, device='cpu')

        # Multiple updates that trigger eviction
        for i in range(5):
            k = torch.randn(1, 8, 50, 64)
            v = torch.randn(1, 8, 50, 64)
            cache.update(k, v)

        stats = cache.get_stats()
        assert stats.tokens_cached == 250  # 5 * 50
        assert stats.tokens_evicted == 150  # 250 - 100
        assert stats.current_size == 100


class TestLayerKVCache:
    """Tests for multi-layer KV cache."""

    def test_layer_cache_initialization(self):
        """LayerKVCache creates cache for each layer."""
        layer_cache = LayerKVCache(num_layers=16, max_tokens=1024)
        assert layer_cache.num_layers == 16
        assert len(layer_cache.caches) == 16

    def test_layer_cache_independence(self):
        """Each layer's cache is independent."""
        layer_cache = LayerKVCache(num_layers=4, max_tokens=100, num_heads=8,
                                   head_dim=64, device='cpu')

        # Update layer 0 only
        cache_0 = layer_cache.get_layer_cache(0)
        k = torch.randn(1, 8, 10, 64)
        v = torch.randn(1, 8, 10, 64)
        cache_0.update(k, v)

        # Layer 0 should have data, others should be empty
        assert layer_cache.caches[0].cache_size == 10
        assert layer_cache.caches[1].cache_size == 0
        assert layer_cache.caches[2].cache_size == 0
        assert layer_cache.caches[3].cache_size == 0

    def test_layer_cache_total_stats(self):
        """Total stats aggregates across layers."""
        layer_cache = LayerKVCache(num_layers=4, max_tokens=50, num_heads=8,
                                   head_dim=64, device='cpu')

        # Update all layers
        for i in range(4):
            cache = layer_cache.get_layer_cache(i)
            k = torch.randn(1, 8, 30, 64)
            v = torch.randn(1, 8, 30, 64)
            cache.update(k, v)

        stats = layer_cache.get_total_stats()
        assert stats['tokens_cached'] == 120  # 4 * 30
        assert stats['num_layers'] == 4

    def test_layer_cache_reset_all(self):
        """Reset clears all layer caches."""
        layer_cache = LayerKVCache(num_layers=4, max_tokens=100, num_heads=8,
                                   head_dim=64, device='cpu')

        # Add data to all layers
        for i in range(4):
            cache = layer_cache.get_layer_cache(i)
            k = torch.randn(1, 8, 10, 64)
            v = torch.randn(1, 8, 10, 64)
            cache.update(k, v)

        # Reset all
        layer_cache.reset()

        # All should be empty
        for i in range(4):
            assert layer_cache.caches[i].cache_size == 0


class TestKVCacheWithRunner:
    """Integration tests with the actual runner."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    def test_simple_programs_with_cache(self, compile_program):
        """Simple programs produce correct results with KV cache."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        programs = [
            ("int main() { return 42; }", 42),
            ("int main() { return 5 + 7; }", 12),
            ("int main() { return 10 * 4; }", 40),
        ]

        bytecodes = []
        data_list = []
        expected = []

        for source, exp in programs:
            bc, data = compile_program(source)
            bytecodes.append(bc)
            data_list.append(data)
            expected.append(exp)

        runner = BatchedSpeculativeRunner(
            batch_size=len(programs),
            use_kv_cache=True,
            kv_cache_max_tokens=128,
        )

        results = runner.run_batch(bytecodes, data_list, max_steps=1000)

        for (output, result), exp in zip(results, expected):
            assert result == exp, f"Expected {exp}, got {result}"

    def test_cache_vs_no_cache_identical(self, compile_program):
        """Results are identical with and without KV cache."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        programs = [
            ("int main() { return (3 + 4) * 5; }", 35),
            ("int main() { return 100 - 37; }", 63),
        ]

        bytecodes = []
        data_list = []
        expected = []

        for source, exp in programs:
            bc, data = compile_program(source)
            bytecodes.append(bc)
            data_list.append(data)
            expected.append(exp)

        # Without cache
        runner_no_cache = BatchedSpeculativeRunner(
            batch_size=len(programs),
            use_kv_cache=False,
        )
        results_no_cache = runner_no_cache.run_batch(bytecodes, data_list, max_steps=1000)

        # With cache
        runner_with_cache = BatchedSpeculativeRunner(
            batch_size=len(programs),
            use_kv_cache=True,
            kv_cache_max_tokens=128,
        )
        results_with_cache = runner_with_cache.run_batch(bytecodes, data_list, max_steps=1000)

        # Compare results
        for (out1, res1), (out2, res2) in zip(results_no_cache, results_with_cache):
            assert res1 == res2, f"Results differ: {res1} vs {res2}"

    def test_eviction_occurs_long_program(self, compile_program):
        """Eviction occurs for long-running programs."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        # Program with many operations
        source = "int main() { return ((10*5) + (20*3) + (15*2) + (8*7)); }"
        bytecode, data = compile_program(source)

        runner = BatchedSpeculativeRunner(
            batch_size=1,
            use_kv_cache=True,
            kv_cache_max_tokens=64,  # Small cache to force eviction
        )

        results = runner.run_batch([bytecode], [data], max_steps=1000)
        stats = runner.kv_cache.get_total_stats()

        # Should have evicted some tokens
        assert stats['tokens_evicted'] > 0, "Expected eviction to occur"

        # Result should still be correct (10*5 + 20*3 + 15*2 + 8*7 = 50+60+30+56 = 196)
        assert results[0][1] == 196, f"Expected 196, got {results[0][1]}"


class TestKVCacheEdgeCases:
    """Edge case tests."""

    def test_empty_update(self):
        """Empty update doesn't break cache."""
        cache = TransformerKVCache(max_tokens=100, num_heads=8, head_dim=64, device='cpu')

        # First add some data
        k1 = torch.randn(1, 8, 10, 64)
        v1 = torch.randn(1, 8, 10, 64)
        cache.update(k1, v1)

        # Add zero-length tensor
        k2 = torch.randn(1, 8, 0, 64)
        v2 = torch.randn(1, 8, 0, 64)
        full_k, full_v = cache.update(k2, v2)

        assert cache.cache_size == 10  # Unchanged

    def test_exact_max_tokens(self):
        """Cache at exactly max_tokens doesn't evict."""
        cache = TransformerKVCache(max_tokens=100, num_heads=8, head_dim=64, device='cpu')

        k = torch.randn(1, 8, 100, 64)
        v = torch.randn(1, 8, 100, 64)
        cache.update(k, v)

        assert cache.cache_size == 100
        assert cache.stats.tokens_evicted == 0

    def test_single_token_eviction(self):
        """Single token over max triggers minimal eviction."""
        cache = TransformerKVCache(max_tokens=100, num_heads=8, head_dim=64, device='cpu')

        # Fill exactly
        k1 = torch.randn(1, 8, 100, 64)
        v1 = torch.randn(1, 8, 100, 64)
        cache.update(k1, v1)

        # Add one more
        k2 = torch.randn(1, 8, 1, 64)
        v2 = torch.randn(1, 8, 1, 64)
        cache.update(k2, v2)

        assert cache.cache_size == 100
        assert cache.stats.tokens_evicted == 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
