"""
KV Cache for Transformer Attention with Eviction.

This implements incremental KV caching for attention layers to reduce
memory usage and computation. Instead of recomputing K/V for all tokens
every forward pass, we cache K/V and only compute for new tokens.

Eviction Strategy:
- Sliding window: Keep only the most recent N tokens
- Reduces memory from O(sequence_length^2) to O(window_size^2)
- Perfect for long-running VM execution with local attention patterns
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class KVCacheStats:
    """Statistics for KV cache monitoring."""
    tokens_cached: int = 0
    tokens_evicted: int = 0
    cache_hits: int = 0
    recomputations: int = 0
    current_size: int = 0
    max_size: int = 0


class TransformerKVCache(nn.Module):
    """
    KV Cache for transformer attention layers.

    Caches Key and Value tensors from previous tokens to avoid recomputation.
    Implements sliding window eviction when cache exceeds max_tokens.

    Args:
        max_tokens: Maximum number of tokens to cache (evict oldest when full)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to store cache on
    """

    def __init__(self, max_tokens: int = 2048, num_heads: int = 8,
                 head_dim: int = 64, device='cuda', batch_size: int = 1):
        super().__init__()
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.batch_size = batch_size

        # Cache storage: [batch, num_heads, seq_len, head_dim]
        # Start empty, grow as needed
        self.cached_k: Optional[torch.Tensor] = None
        self.cached_v: Optional[torch.Tensor] = None
        self.cache_size = 0  # Number of tokens currently cached

        # Statistics
        self.stats = KVCacheStats(max_size=max_tokens)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K/V tensors and return full K/V including cached.

        Args:
            new_k: New keys [batch, num_heads, new_seq_len, head_dim]
            new_v: New values [batch, num_heads, new_seq_len, head_dim]

        Returns:
            (full_k, full_v): Complete K/V tensors including cache
        """
        B, H, new_len, HD = new_k.shape

        if self.cached_k is None:
            # First update - initialize cache
            self.cached_k = new_k
            self.cached_v = new_v
            self.cache_size = new_len
            self.stats.tokens_cached += new_len * B
        else:
            # Append new tokens to cache
            self.cached_k = torch.cat([self.cached_k, new_k], dim=2)
            self.cached_v = torch.cat([self.cached_v, new_v], dim=2)
            self.cache_size += new_len
            self.stats.tokens_cached += new_len * B
            self.stats.cache_hits += 1

        # Evict oldest tokens if cache exceeds max_tokens
        if self.cache_size > self.max_tokens:
            tokens_to_evict = self.cache_size - self.max_tokens
            self.cached_k = self.cached_k[:, :, tokens_to_evict:, :]
            self.cached_v = self.cached_v[:, :, tokens_to_evict:, :]
            self.cache_size = self.max_tokens
            self.stats.tokens_evicted += tokens_to_evict * B

        self.stats.current_size = self.cache_size

        return self.cached_k, self.cached_v

    def reset(self):
        """Clear the cache."""
        self.cached_k = None
        self.cached_v = None
        self.cache_size = 0
        self.stats = KVCacheStats(max_size=self.max_tokens)

    def get_stats(self) -> KVCacheStats:
        """Get cache statistics."""
        return self.stats


class LayerKVCache:
    """
    Manages KV caches for all layers in a transformer.

    Each layer gets its own TransformerKVCache instance.
    """

    def __init__(self, num_layers: int, max_tokens: int = 2048,
                 num_heads: int = 8, head_dim: int = 64, device='cuda'):
        self.num_layers = num_layers
        self.caches = [
            TransformerKVCache(max_tokens, num_heads, head_dim, device)
            for _ in range(num_layers)
        ]

    def get_layer_cache(self, layer_idx: int) -> TransformerKVCache:
        """Get cache for specific layer."""
        return self.caches[layer_idx]

    def reset(self):
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    def get_total_stats(self) -> dict:
        """Get aggregated statistics across all layers."""
        total_cached = sum(c.stats.tokens_cached for c in self.caches)
        total_evicted = sum(c.stats.tokens_evicted for c in self.caches)
        total_size = sum(c.stats.current_size for c in self.caches)
        cache_hits = sum(c.stats.cache_hits for c in self.caches)

        return {
            'tokens_cached': total_cached,
            'tokens_evicted': total_evicted,
            'current_total_size': total_size,
            'cache_hits': cache_hits,
            'num_layers': self.num_layers,
        }
