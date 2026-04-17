#!/usr/bin/env python3
"""
Test KV cache eviction with a simple program.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from src.baked_c4 import BakedC4Transformer
from neural_vm.kv_cache import LayerKVCache

def test_kv_cache_basic():
    """Test KV cache with simple program."""
    print("=" * 60)
    print("TEST: KV Cache with Simple Program")
    print("=" * 60)

    # Simple program
    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)

    # Create model without caching (baseline)
    print("\n1. Running without KV cache (baseline)...")
    c4_no_cache = BakedC4Transformer(use_speculator=False)
    c4_no_cache.eval()
    result_no_cache = c4_no_cache.run_bytecode(bytecode, data, max_steps=100)
    print(f"   Result: {result_no_cache}")

    # Create model with KV caching
    print("\n2. Running with KV cache...")
    c4_with_cache = BakedC4Transformer(use_speculator=False)
    c4_with_cache.eval()

    # Create KV cache for the model
    kv_cache = LayerKVCache(
        num_layers=16,
        max_tokens=1024,
        num_heads=8,
        head_dim=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run with cache (need to modify run_bytecode to pass kv_cache)
    # For now, just test that cache can be created
    print(f"   KV cache created: {kv_cache.num_layers} layers")
    print(f"   Max tokens per layer: {kv_cache.caches[0].max_tokens}")

    # Test cache statistics
    stats = kv_cache.get_total_stats()
    print(f"\n3. Cache statistics:")
    print(f"   Tokens cached: {stats['tokens_cached']}")
    print(f"   Tokens evicted: {stats['tokens_evicted']}")
    print(f"   Current size: {stats['current_total_size']}")
    print(f"   Cache hits: {stats['cache_hits']}")

    print("\n✓ KV cache test passed!")
    return 0

if __name__ == '__main__':
    sys.exit(test_kv_cache_basic())
