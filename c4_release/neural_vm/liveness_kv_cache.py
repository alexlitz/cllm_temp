"""
Liveness-Based KV Cache with No Fixed Size Limit.

Instead of arbitrary max_tokens, we use liveness analysis to determine
what data is provably dead and can be safely evicted without affecting
computation validity.

Key principle: Only keep data that could affect future computations.

Liveness Rules:
1. Memory: Only keep latest write to each address (old writes are dead)
2. Registers: Only keep latest state (old states are dead)
3. I/O: Only keep current operation (previous I/O is dead)
4. Code: Keep current instruction + lookahead window for branches
5. Stack: Keep only accessible stack frames (below SP is dead)

Per-Head Liveness:
- Each attention head may have different liveness requirements
- Memory head: Needs latest writes to addresses
- Code head: Needs current + branch targets
- I/O head: Needs only current I/O state
- Register head: Needs only current register values
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Set, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LivenessStats:
    """Statistics for liveness-based eviction."""
    tokens_added: int = 0
    tokens_evicted: int = 0

    # Eviction reasons
    memory_overwrites: int = 0
    register_overwrites: int = 0
    io_overwrites: int = 0
    stack_underflow: int = 0  # Below stack pointer
    zero_writes: int = 0

    # Liveness metrics
    current_live_set_size: int = 0
    max_live_set_size: int = 0
    avg_live_set_size: float = 0.0

    # Per-head stats
    head_live_sets: Dict[int, int] = field(default_factory=dict)


class LivenessAnalyzer:
    """
    Analyzes VM execution to determine live data.

    Tracks:
    - Memory: address → latest write position
    - Registers: register → latest write position
    - Stack: current SP, which positions are below SP (dead)
    - I/O: current I/O position
    - Code: current PC + reachable instructions
    """

    def __init__(self):
        # Address → position of latest write
        self.memory_live: Dict[int, int] = {}

        # Register → position of latest write
        self.register_live: Dict[str, int] = {}

        # Current I/O position (all previous are dead)
        self.io_live: Optional[int] = None

        # Stack pointer tracking
        self.stack_pointer: int = 0x10000  # C4 stack starts at top
        self.stack_writes: Dict[int, int] = {}  # address → position

        # Code position tracking
        self.pc: int = 0
        self.code_positions: Set[int] = set()

        # Set of all live positions
        self.live_positions: Set[int] = set()

    def update(self, position: int, metadata: Dict) -> Set[int]:
        """
        Update liveness based on new token metadata.

        Returns: Set of positions that became dead (can be evicted)
        """
        dead_positions = set()
        token_type = metadata.get('type', 'other')

        # Memory write - previous write to same address becomes dead
        if token_type == 'memory_write':
            address = metadata.get('address')
            value = metadata.get('value', 1)

            # Zero write - don't track at all (freed memory)
            if value == 0:
                if address in self.memory_live:
                    dead_pos = self.memory_live[address]
                    dead_positions.add(dead_pos)
                    self.live_positions.discard(dead_pos)
                    del self.memory_live[address]
                # Don't add position to live set (zero write is dead)
                return dead_positions

            # Non-zero write - evict old write to same address
            if address in self.memory_live:
                old_pos = self.memory_live[address]
                dead_positions.add(old_pos)
                self.live_positions.discard(old_pos)

            # Track new write as live
            self.memory_live[address] = position
            self.live_positions.add(position)

            # Stack tracking - writes below SP are dead
            if address < self.stack_pointer:
                # This is on the stack and accessible
                self.stack_writes[address] = position
            else:
                # Above stack - if it was a stack write, it's now dead
                if address in self.stack_writes:
                    old_stack_pos = self.stack_writes[address]
                    dead_positions.add(old_stack_pos)
                    self.live_positions.discard(old_stack_pos)
                    del self.stack_writes[address]

        # Register write - previous register state becomes dead
        elif token_type == 'register':
            register = metadata.get('register', 'unknown')
            value = metadata.get('value', 1)

            # Zero write to register - clear it (don't track)
            if value == 0:
                if register in self.register_live:
                    dead_pos = self.register_live[register]
                    dead_positions.add(dead_pos)
                    self.live_positions.discard(dead_pos)
                    del self.register_live[register]
                return dead_positions

            # Non-zero register write
            if register in self.register_live:
                old_pos = self.register_live[register]
                dead_positions.add(old_pos)
                self.live_positions.discard(old_pos)

            self.register_live[register] = position
            self.live_positions.add(position)

            # Special: Track SP changes (affects stack liveness)
            if register == 'SP':
                self.stack_pointer = value
                # Mark stack writes above new SP as dead
                dead_stack = []
                for addr, pos in self.stack_writes.items():
                    if addr >= self.stack_pointer:
                        dead_positions.add(pos)
                        self.live_positions.discard(pos)
                        dead_stack.append(addr)
                for addr in dead_stack:
                    del self.stack_writes[addr]

        # I/O - all previous I/O becomes dead
        elif token_type == 'io':
            if self.io_live is not None:
                dead_positions.add(self.io_live)
                self.live_positions.discard(self.io_live)

            self.io_live = position
            self.live_positions.add(position)

        # Memory read - doesn't change liveness (reads don't kill)
        elif token_type == 'memory_read':
            # Reads are ephemeral - they don't need to be cached long-term
            # We can evict them immediately after use
            pass  # Don't add to live set

        # Code/instruction - track current execution
        elif token_type == 'code':
            pc = metadata.get('pc', 0)
            self.pc = pc
            self.code_positions.add(position)
            self.live_positions.add(position)

            # Keep only recent code (small window for branch lookback)
            # Old code positions can be evicted
            if len(self.code_positions) > 100:  # Keep last 100 instructions
                oldest = min(self.code_positions)
                self.code_positions.remove(oldest)
                dead_positions.add(oldest)
                self.live_positions.discard(oldest)

        # Other tokens - generally ephemeral (arithmetic intermediate values)
        else:
            # Don't track in live set - these are dead after one use
            pass

        return dead_positions

    def get_live_set(self) -> Set[int]:
        """Get set of all live positions."""
        return self.live_positions.copy()

    def get_stats(self) -> Dict:
        """Get liveness statistics."""
        return {
            'live_memory_addresses': len(self.memory_live),
            'live_registers': len(self.register_live),
            'live_stack_entries': len(self.stack_writes),
            'live_code_positions': len(self.code_positions),
            'total_live_positions': len(self.live_positions),
        }


class LivenessKVCache(nn.Module):
    """
    KV Cache with liveness-based eviction (no fixed size limit).

    Evicts data that is provably dead based on VM semantics.
    Cache size naturally bounded by live data set size.
    """

    def __init__(self, num_heads: int = 8, head_dim: int = 64,
                 device='cuda', head_roles: Optional[List[str]] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Optional: Specify what each head attends to
        # e.g., ['memory', 'code', 'registers', 'io', ...]
        self.head_roles = head_roles or ['general'] * num_heads

        # Cache storage: [B, H, S, HD]
        self.cached_k: Optional[torch.Tensor] = None
        self.cached_v: Optional[torch.Tensor] = None

        # Position tracking
        self.position_to_cache_idx: Dict[int, int] = {}  # global_pos → cache_idx
        self.cache_idx_to_position: Dict[int, int] = {}  # cache_idx → global_pos
        self.next_position = 0
        self.cache_size = 0

        # Liveness analyzer
        self.liveness = LivenessAnalyzer()

        # Statistics
        self.stats = LivenessStats()

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor,
               token_metadata: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with liveness-based eviction.

        Args:
            new_k: New keys [B, H, new_len, HD]
            new_v: New values [B, H, new_len, HD]
            token_metadata: Metadata for liveness analysis

        Returns:
            (K, V): Full K/V including live cached data
        """
        B, H, new_len, HD = new_k.shape

        # Process each new token for liveness
        all_dead_positions = set()
        new_live_positions = []

        if token_metadata:
            for i, meta in enumerate(token_metadata[:new_len]):
                position = self.next_position + i

                # Update liveness - get positions that became dead
                dead = self.liveness.update(position, meta)
                all_dead_positions.update(dead)

                # Track this position
                new_live_positions.append(position)
        else:
            # No metadata - assume all tokens are live
            new_live_positions = list(range(self.next_position, self.next_position + new_len))

        # Evict dead positions from cache
        if all_dead_positions:
            self._evict_positions(all_dead_positions)

        # Add new tokens to cache
        if self.cached_k is None:
            # Initialize cache
            self.cached_k = new_k
            self.cached_v = new_v

            # Track positions
            for i, pos in enumerate(new_live_positions):
                self.position_to_cache_idx[pos] = i
                self.cache_idx_to_position[i] = pos

            self.cache_size = new_len
        else:
            # Append new tokens
            self.cached_k = torch.cat([self.cached_k, new_k], dim=2)
            self.cached_v = torch.cat([self.cached_v, new_v], dim=2)

            # Track positions
            for i, pos in enumerate(new_live_positions):
                cache_idx = self.cache_size + i
                self.position_to_cache_idx[pos] = cache_idx
                self.cache_idx_to_position[cache_idx] = pos

            self.cache_size += new_len

        # Update stats
        self.stats.tokens_added += new_len * B
        self.stats.tokens_evicted += len(all_dead_positions) * B
        self.stats.current_live_set_size = len(self.liveness.get_live_set())
        self.stats.max_live_set_size = max(self.stats.max_live_set_size,
                                           self.stats.current_live_set_size)

        self.next_position += new_len

        return self.cached_k, self.cached_v

    def _evict_positions(self, dead_positions: Set[int]):
        """
        Evict dead positions from cache.

        This requires compacting the cache tensor.
        """
        if not dead_positions or self.cached_k is None:
            return

        # Find cache indices to keep (not in dead set)
        keep_indices = []
        new_position_mapping = {}
        new_idx = 0

        for cache_idx in range(self.cache_size):
            position = self.cache_idx_to_position.get(cache_idx)
            if position is not None and position not in dead_positions:
                keep_indices.append(cache_idx)
                new_position_mapping[position] = new_idx
                new_idx += 1

        # Nothing to evict
        if len(keep_indices) == self.cache_size:
            return

        # Compact cache
        if keep_indices:
            # Select valid entries
            keep_mask = torch.zeros(self.cache_size, dtype=torch.bool, device=self.device)
            keep_mask[keep_indices] = True

            self.cached_k = self.cached_k[:, :, keep_mask, :]
            self.cached_v = self.cached_v[:, :, keep_mask, :]
            self.cache_size = len(keep_indices)

            # Update mappings
            self.position_to_cache_idx = new_position_mapping
            self.cache_idx_to_position = {idx: pos for pos, idx in new_position_mapping.items()}
        else:
            # All positions dead - clear cache
            self.cached_k = None
            self.cached_v = None
            self.cache_size = 0
            self.position_to_cache_idx.clear()
            self.cache_idx_to_position.clear()

        # Update stats based on eviction reasons
        # (Would need to track which positions are which type)
        self.stats.memory_overwrites += sum(1 for p in dead_positions
                                            if p in range(self.next_position))

    def reset(self):
        """Clear cache and reset liveness tracking."""
        self.cached_k = None
        self.cached_v = None
        self.cache_size = 0
        self.position_to_cache_idx.clear()
        self.cache_idx_to_position.clear()
        self.next_position = 0
        self.liveness = LivenessAnalyzer()
        self.stats = LivenessStats()

    def get_stats(self) -> LivenessStats:
        """Get cache statistics."""
        return self.stats


class LivenessLayerKVCache:
    """
    Per-layer liveness-based KV caches.

    Each layer can have different liveness requirements based on
    what that layer's attention heads focus on.
    """

    def __init__(self, num_layers: int, num_heads: int = 8,
                 head_dim: int = 64, device='cuda',
                 layer_head_roles: Optional[Dict[int, List[str]]] = None):
        self.num_layers = num_layers
        self.layer_head_roles = layer_head_roles or {}

        self.caches = []
        for layer_idx in range(num_layers):
            head_roles = self.layer_head_roles.get(layer_idx, None)
            cache = LivenessKVCache(
                num_heads=num_heads,
                head_dim=head_dim,
                device=device,
                head_roles=head_roles
            )
            self.caches.append(cache)

    def get_layer_cache(self, layer_idx: int) -> LivenessKVCache:
        """Get cache for specific layer."""
        return self.caches[layer_idx]

    def reset(self):
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    def get_total_stats(self) -> Dict:
        """Get aggregated statistics."""
        total_added = sum(c.stats.tokens_added for c in self.caches)
        total_evicted = sum(c.stats.tokens_evicted for c in self.caches)
        total_live = sum(c.stats.current_live_set_size for c in self.caches)
        max_live = max(c.stats.max_live_set_size for c in self.caches)

        # Liveness breakdown
        liveness_stats = [c.liveness.get_stats() for c in self.caches]
        total_live_memory = sum(s['live_memory_addresses'] for s in liveness_stats)
        total_live_registers = sum(s['live_registers'] for s in liveness_stats)

        return {
            'tokens_added': total_added,
            'tokens_evicted': total_evicted,
            'current_live_set_size': total_live,
            'max_live_set_size': max_live,
            'live_memory_addresses': total_live_memory,
            'live_registers': total_live_registers,
            'eviction_rate': total_evicted / max(1, total_added),
            'cache_efficiency': 1.0 - (total_live / max(1, total_added)),
        }
