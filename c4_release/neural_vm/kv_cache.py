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
from typing import List, Optional, Tuple, Union
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
        # Absolute position id per cached token, shape [batch, seq_len], int64.
        # Filled at append time with ``next_pos_id``; survives pruning
        # (``LayerKVCache.prune`` index-selects this in lock-step with K/V).
        # Per ``KV_CACHE_PRUNING_SPEC.md`` §9: surviving positions must retain
        # their original indices so ALiBi/RoPE distances stay correct after
        # middle-position eviction.
        self.cached_pos_ids: Optional[torch.Tensor] = None
        # Per-head keep mask, shape ``[batch, num_heads, seq_len]``, bool.
        # Used by Phase B per-head pruning: a head h may "soft-evict"
        # position s by setting ``per_head_keep_mask[:, h, s] = False``
        # without dropping it from the K/V tensors. ``AutoregressiveAttention``
        # reads this and masks the corresponding logits to ``-inf`` so the
        # head ignores that position. Hard global eviction (a row removed
        # from K/V) happens only when *all* heads in *all* layers agree to
        # drop it (the caller is responsible for that AND).
        # ``None`` means "all heads keep all positions" — i.e. legacy
        # behaviour, no per-head masking applied.
        self.per_head_keep_mask: Optional[torch.Tensor] = None
        self.cache_size = 0  # Number of tokens currently cached
        # Next absolute position id to assign on append. Monotonically
        # increases over the cache's lifetime; pruning shrinks ``cache_size``
        # but never rewinds ``next_pos_id`` (positions are stable identifiers).
        self.next_pos_id = 0

        # Statistics
        self.stats = KVCacheStats(max_size=max_tokens)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor,
               pos_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K/V tensors and return full K/V including cached.

        Args:
            new_k: New keys [batch, num_heads, new_seq_len, head_dim]
            new_v: New values [batch, num_heads, new_seq_len, head_dim]
            pos_ids: Optional explicit absolute position ids for the new
                tokens, shape ``[batch, new_seq_len]``, int64. When omitted,
                positions are assigned sequentially starting from
                ``self.next_pos_id`` — this matches the legacy 0..S-1
                semantics when no pruning has occurred.

        Returns:
            (full_k, full_v): Complete K/V tensors including cache
        """
        B, H, new_len, HD = new_k.shape

        # Default pos_ids: sequential from next_pos_id (preserves legacy
        # 0..S-1 numbering when pruning has not yet happened).
        if pos_ids is None:
            pos_ids = torch.arange(
                self.next_pos_id, self.next_pos_id + new_len,
                dtype=torch.long, device=new_k.device,
            ).unsqueeze(0).expand(B, -1).contiguous()

        if self.cached_k is None:
            # First update - initialize cache
            self.cached_k = new_k
            self.cached_v = new_v
            self.cached_pos_ids = pos_ids
            self.cache_size = new_len
            self.stats.tokens_cached += new_len * B
            # Per-head mask stays None on init — only allocated when the
            # pruner first writes per-head decisions.
        else:
            # Append new tokens to cache
            self.cached_k = torch.cat([self.cached_k, new_k], dim=2)
            self.cached_v = torch.cat([self.cached_v, new_v], dim=2)
            self.cached_pos_ids = torch.cat([self.cached_pos_ids, pos_ids], dim=1)
            # New rows are "kept by all heads" by default; only extend the
            # per-head mask if it has been allocated (Phase B pruning ran).
            if self.per_head_keep_mask is not None:
                ones = torch.ones(
                    self.per_head_keep_mask.shape[0],
                    self.per_head_keep_mask.shape[1],
                    new_len,
                    dtype=torch.bool,
                    device=self.per_head_keep_mask.device,
                )
                self.per_head_keep_mask = torch.cat(
                    [self.per_head_keep_mask, ones], dim=2
                )
            self.cache_size += new_len
            self.stats.tokens_cached += new_len * B
            self.stats.cache_hits += 1
        # Advance the monotonic position counter.
        # Note: positions are 1-D in time, so we bump by ``new_len`` only —
        # batch positions are duplicates across the batch dim.
        self.next_pos_id = int(pos_ids.max().item()) + 1 if pos_ids.numel() > 0 else self.next_pos_id

        # Evict oldest tokens if cache exceeds max_tokens
        if self.cache_size > self.max_tokens:
            tokens_to_evict = self.cache_size - self.max_tokens
            self.cached_k = self.cached_k[:, :, tokens_to_evict:, :]
            self.cached_v = self.cached_v[:, :, tokens_to_evict:, :]
            self.cached_pos_ids = self.cached_pos_ids[:, tokens_to_evict:]
            if self.per_head_keep_mask is not None:
                self.per_head_keep_mask = self.per_head_keep_mask[
                    :, :, tokens_to_evict:
                ]
            self.cache_size = self.max_tokens
            self.stats.tokens_evicted += tokens_to_evict * B

        self.stats.current_size = self.cache_size

        return self.cached_k, self.cached_v

    def reset(self):
        """Clear the cache."""
        self.cached_k = None
        self.cached_v = None
        self.cached_pos_ids = None
        self.per_head_keep_mask = None
        self.cache_size = 0
        self.next_pos_id = 0
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

    def prune(
        self,
        keep_mask: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Union[int, List[int]]:
        """Apply a keep-mask (or list of per-layer keep-masks) across layers.

        Two modes:

        - **Uniform (single mask)** — per ``KV_CACHE_PRUNING_SPEC.md`` §3.3:
          the keep-mask is shared across all layers so the per-position
          absolute id (and therefore the ALiBi/RoPE distance) stays in
          lock-step. Pass a single ``BoolTensor[S]`` or index ``LongTensor``
          and get back an ``int`` new cache size.

        - **Per-layer (list of masks)** — Phase A: each layer i gets its
          own ``BoolTensor[S_i]`` or index tensor and may shrink to a
          different post-prune length ``S_i'``. ``cached_pos_ids`` is
          index-selected per-layer so each layer's pos_ids stay aligned
          with its own K/V. Returns a ``List[int]`` of per-layer new sizes.

        Forward-pass compatibility: ``AutoregressiveAttention.forward``
        already reads from the per-layer cache (``kv_cache.caches[i]``) and
        uses that cache's own ``cached_pos_ids`` for ALiBi distances, so
        divergent per-layer ``S_i`` works out of the box.

        Args:
            keep_mask: Either
                - a single ``BoolTensor[S]`` / index ``LongTensor`` applied
                  to every layer (legacy uniform), OR
                - a list of length ``num_layers`` where entry ``i`` is the
                  mask/indices for layer ``i``.

        Returns:
            ``int`` (uniform mode) or ``List[int]`` (per-layer mode):
            new cache size per layer.
        """
        if not self.caches:
            return 0 if not isinstance(keep_mask, list) else []

        per_layer_mode = isinstance(keep_mask, (list, tuple))
        if per_layer_mode:
            if len(keep_mask) != len(self.caches):
                raise ValueError(
                    f"per-layer keep_mask list length ({len(keep_mask)}) "
                    f"!= num_layers ({len(self.caches)})"
                )

        # Use the first non-empty layer to derive a fallback device.
        ref = None
        for c in self.caches:
            if c.cached_k is not None:
                ref = c
                break
        if ref is None:
            return 0 if not per_layer_mode else [0] * len(self.caches)

        if not per_layer_mode:
            # Uniform mask: keep the legacy single-int return contract.
            cache_size = ref.cache_size
            device = ref.cached_k.device
            keep_idx = self._normalize_to_idx(keep_mask, cache_size, device)
            new_size = int(keep_idx.numel())
            evicted = cache_size - new_size
            for c in self.caches:
                if c.cached_k is None:
                    continue
                self._apply_keep_idx(c, keep_idx, evicted)
            return new_size

        # Per-layer mask: each layer index-selects independently. Per
        # ``KV_CACHE_PRUNING_SPEC.md`` §9 (extension): pos_ids travel with
        # the K/V they belong to, so per-layer divergent ``S_i`` keeps ALiBi
        # distance computation correct as long as each layer's
        # ``cached_pos_ids`` is kept in lockstep with its own K/V.
        new_sizes: List[int] = []
        for idx, c in enumerate(self.caches):
            if c.cached_k is None:
                new_sizes.append(0)
                continue
            layer_keep = keep_mask[idx]
            layer_size = c.cache_size
            layer_device = c.cached_k.device
            keep_idx = self._normalize_to_idx(layer_keep, layer_size, layer_device)
            new_size = int(keep_idx.numel())
            evicted = layer_size - new_size
            self._apply_keep_idx(c, keep_idx, evicted)
            new_sizes.append(new_size)
        return new_sizes

    @staticmethod
    def _normalize_to_idx(
        keep_mask: torch.Tensor, cache_size: int, device: torch.device
    ) -> torch.Tensor:
        """Convert a bool mask or LongTensor of indices to a LongTensor on ``device``."""
        if isinstance(keep_mask, torch.Tensor) and keep_mask.dtype == torch.bool:
            assert keep_mask.shape[-1] == cache_size, (
                f"keep_mask len ({keep_mask.shape[-1]}) != cache_size ({cache_size})"
            )
            return keep_mask.to(device=device).nonzero(as_tuple=False).flatten()
        return keep_mask.to(device=device, dtype=torch.long)

    @staticmethod
    def _apply_keep_idx(c, keep_idx: torch.Tensor, evicted: int) -> None:
        """Index-select K/V/pos_ids/per-head-mask on a single layer cache."""
        new_size = int(keep_idx.numel())
        c.cached_k = c.cached_k.index_select(2, keep_idx).contiguous()
        c.cached_v = c.cached_v.index_select(2, keep_idx).contiguous()
        if c.cached_pos_ids is not None:
            c.cached_pos_ids = c.cached_pos_ids.index_select(1, keep_idx).contiguous()
        if getattr(c, "per_head_keep_mask", None) is not None:
            c.per_head_keep_mask = c.per_head_keep_mask.index_select(
                2, keep_idx
            ).contiguous()
        c.cache_size = new_size
        c.stats.current_size = new_size
        if evicted > 0:
            c.stats.tokens_evicted += evicted

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
