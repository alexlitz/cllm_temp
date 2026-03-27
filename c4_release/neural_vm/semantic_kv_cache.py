"""
Semantic KV Cache for VM Execution with Smart Eviction.

Unlike FIFO eviction, this implements semantically-aware eviction:
- Overwritten memory: Evict old value when same address is written
- I/O entries: Evict all previous I/O on new I/O write
- Register overwrites: Evict old state when register is updated
- Zero writes (free): Don't cache zeros at all

This matches VM semantics: latest-write-wins for memory/registers,
and I/O is ephemeral.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Set
from dataclasses import dataclass, field


@dataclass
class SemanticKVCacheStats:
    """Statistics for semantic KV cache monitoring."""
    tokens_cached: int = 0
    tokens_evicted: int = 0
    memory_overwrites_evicted: int = 0
    io_evictions: int = 0
    register_overwrites_evicted: int = 0
    zero_writes_skipped: int = 0
    cache_hits: int = 0
    current_size: int = 0
    max_size: int = 0


class SemanticTransformerKVCache(nn.Module):
    """
    Semantic-aware KV Cache for VM execution.

    Eviction policy based on VM semantics:
    1. Memory overwrites: Keep only latest write to each address
    2. I/O writes: Evict all previous I/O on new I/O
    3. Register overwrites: Keep only latest state of each register
    4. Zero writes: Skip caching (zeros = freed memory)

    Args:
        max_tokens: Maximum tokens to cache before forced eviction
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
        self.cached_k: Optional[torch.Tensor] = None
        self.cached_v: Optional[torch.Tensor] = None
        self.cache_size = 0

        # Semantic tracking for eviction
        # Maps: address/register -> list of sequence positions
        self.memory_positions: Dict[int, list] = {}  # address -> [positions]
        self.register_positions: Dict[str, list] = {}  # register -> [positions]
        self.io_positions: list = []  # All I/O positions

        # Track which positions are still valid (not evicted)
        self.valid_positions: Set[int] = set()

        # Position counter (tracks absolute position in stream)
        self.position_counter = 0

        # Statistics
        self.stats = SemanticKVCacheStats(max_size=max_tokens)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor,
               token_metadata: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K/V tensors using semantic eviction.

        Args:
            new_k: New keys [batch, num_heads, new_seq_len, head_dim]
            new_v: New values [batch, num_heads, new_seq_len, head_dim]
            token_metadata: List of dicts with semantic info for each new token:
                {
                    'type': 'memory_write' | 'memory_read' | 'io' | 'register' | 'other',
                    'address': int (for memory),
                    'register': str (for registers),
                    'value': int (to detect zero writes),
                }

        Returns:
            (full_k, full_v): Complete K/V tensors with semantic eviction applied
        """
        B, H, new_len, HD = new_k.shape

        # Apply semantic eviction if metadata provided
        if token_metadata is not None:
            self._semantic_evict(token_metadata, new_len)

        # Add new tokens to cache
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

        # Update valid positions for new tokens
        for i in range(new_len):
            self.valid_positions.add(self.position_counter + i)
        self.position_counter += new_len

        # Fallback: If cache still exceeds max after semantic eviction, use FIFO
        if self.cache_size > self.max_tokens:
            tokens_to_evict = self.cache_size - self.max_tokens
            self.cached_k = self.cached_k[:, :, tokens_to_evict:, :]
            self.cached_v = self.cached_v[:, :, tokens_to_evict:, :]
            self.cache_size = self.max_tokens
            self.stats.tokens_evicted += tokens_to_evict * B

        self.stats.current_size = self.cache_size

        return self.cached_k, self.cached_v

    def _semantic_evict(self, token_metadata: list, new_len: int):
        """
        Apply semantic eviction based on token metadata.

        Eviction rules:
        1. Memory overwrite: Evict previous write to same address
        2. I/O write: Evict ALL previous I/O
        3. Register overwrite: Evict previous write to same register
        4. Zero write: Don't cache at all (skip)
        """
        positions_to_evict = set()

        for i, meta in enumerate(token_metadata[:new_len]):
            token_type = meta.get('type', 'other')
            value = meta.get('value', None)

            # Rule 4: Zero writes - skip caching entirely
            if value == 0 and token_type in ['memory_write', 'register']:
                self.stats.zero_writes_skipped += 1
                # Mark this position as invalid immediately
                positions_to_evict.add(self.position_counter + i)
                continue

            # Rule 1: Memory overwrites
            if token_type == 'memory_write':
                address = meta.get('address')
                if address is not None:
                    # Evict previous writes to this address
                    if address in self.memory_positions:
                        for old_pos in self.memory_positions[address]:
                            if old_pos in self.valid_positions:
                                positions_to_evict.add(old_pos)
                                self.stats.memory_overwrites_evicted += 1
                    # Track this new write
                    self.memory_positions[address] = [self.position_counter + i]

            # Rule 2: I/O writes
            elif token_type == 'io':
                # Evict ALL previous I/O
                for old_pos in self.io_positions:
                    if old_pos in self.valid_positions:
                        positions_to_evict.add(old_pos)
                        self.stats.io_evictions += 1
                # Track this I/O
                self.io_positions = [self.position_counter + i]

            # Rule 3: Register overwrites
            elif token_type == 'register':
                register = meta.get('register')
                if register is not None:
                    # Evict previous state of this register
                    if register in self.register_positions:
                        for old_pos in self.register_positions[register]:
                            if old_pos in self.valid_positions:
                                positions_to_evict.add(old_pos)
                                self.stats.register_overwrites_evicted += 1
                    # Track this new register state
                    self.register_positions[register] = [self.position_counter + i]

        # Apply evictions by removing positions from valid set
        self.valid_positions -= positions_to_evict

        # Compact cache if needed (remove invalidated positions)
        if positions_to_evict and self.cached_k is not None:
            self._compact_cache()

    def _compact_cache(self):
        """
        Compact cache by removing invalidated positions.

        This is expensive (requires tensor slicing), so only do it when
        we've accumulated significant dead entries.
        """
        if self.cache_size == 0:
            return

        # Build mask of valid positions in current cache
        cache_start_pos = self.position_counter - self.cache_size
        valid_mask = []
        for i in range(self.cache_size):
            pos = cache_start_pos + i
            valid_mask.append(pos in self.valid_positions)

        # If all positions are valid, nothing to compact
        if all(valid_mask):
            return

        # Create tensor mask and select valid positions
        mask_tensor = torch.tensor(valid_mask, device=self.device)

        # Select valid K/V entries
        # K/V shape: [B, H, S, HD]
        valid_k_list = []
        valid_v_list = []
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                valid_k_list.append(self.cached_k[:, :, i:i+1, :])
                valid_v_list.append(self.cached_v[:, :, i:i+1, :])

        if valid_k_list:
            self.cached_k = torch.cat(valid_k_list, dim=2)
            self.cached_v = torch.cat(valid_v_list, dim=2)
            self.cache_size = len(valid_k_list)
        else:
            self.cached_k = None
            self.cached_v = None
            self.cache_size = 0

    def reset(self):
        """Clear the cache and all semantic tracking."""
        self.cached_k = None
        self.cached_v = None
        self.cache_size = 0
        self.memory_positions.clear()
        self.register_positions.clear()
        self.io_positions.clear()
        self.valid_positions.clear()
        self.position_counter = 0
        self.stats = SemanticKVCacheStats(max_size=self.max_tokens)

    def get_stats(self) -> SemanticKVCacheStats:
        """Get cache statistics."""
        return self.stats


class SemanticLayerKVCache:
    """
    Manages semantic KV caches for all layers in a transformer.

    Each layer gets its own SemanticTransformerKVCache instance.
    """

    def __init__(self, num_layers: int, max_tokens: int = 2048,
                 num_heads: int = 8, head_dim: int = 64, device='cuda'):
        self.num_layers = num_layers
        self.caches = [
            SemanticTransformerKVCache(max_tokens, num_heads, head_dim, device)
            for _ in range(num_layers)
        ]

    def get_layer_cache(self, layer_idx: int) -> SemanticTransformerKVCache:
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
        memory_overwrites = sum(c.stats.memory_overwrites_evicted for c in self.caches)
        io_evictions = sum(c.stats.io_evictions for c in self.caches)
        register_overwrites = sum(c.stats.register_overwrites_evicted for c in self.caches)
        zero_writes_skipped = sum(c.stats.zero_writes_skipped for c in self.caches)
        total_size = sum(c.stats.current_size for c in self.caches)
        cache_hits = sum(c.stats.cache_hits for c in self.caches)

        return {
            'tokens_cached': total_cached,
            'tokens_evicted': total_evicted,
            'memory_overwrites_evicted': memory_overwrites,
            'io_evictions': io_evictions,
            'register_overwrites_evicted': register_overwrites,
            'zero_writes_skipped': zero_writes_skipped,
            'current_total_size': total_size,
            'cache_hits': cache_hits,
            'num_layers': self.num_layers,
        }
