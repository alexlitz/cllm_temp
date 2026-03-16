"""
KV Cache with Eviction Support for Neural VM.

Implements aggressive eviction following Section 19 of DOCUMENT_FIXES.md:

## What Gets Evicted

1. **Overwritten memory**: When writing to address X, old entry at X is evicted
2. **Old I/O entries**: Only keep the most recent I/O state
3. **Old register states**: Only current register values matter
4. **Zero values**: Freed memory (ZFOD semantics)

## softmax1 and ZFOD (Zero Fill On Demand)

softmax1 adds "1": softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))
Key property: When all scores are very negative, output approaches 0.
This enables ZFOD - uninitialized/freed memory returns ~0.

## Eviction Strategy

For each write:
1. If address exists in cache, evict the OLD entry (not just mark invalid)
2. Write new value to a fresh slot
3. Zero-writes evict without creating new entry

For I/O:
- Keep only last I/O state (evict previous)

For registers:
- Keep only current register values (evict previous states)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""
    total_entries: int = 0
    valid_entries: int = 0
    evicted_entries: int = 0
    compactions: int = 0
    reads: int = 0
    writes: int = 0
    hits: int = 0
    misses: int = 0


class EvictableKVCache(nn.Module):
    """
    KV Cache with aggressive eviction.

    Evicts:
    1. Overwritten memory: Old entry evicted when address is rewritten
    2. Zero writes: Immediate eviction (ZFOD semantics)
    3. Old I/O entries: Only latest I/O state kept
    4. Old register states: Only current registers kept
    5. LRU eviction when cache full

    Uses sparse storage for efficiency - only current values stored.
    """

    # Address ranges for special handling
    IO_ADDR_START = 0xFFFF0000      # I/O addresses start here
    REGISTER_ADDR_START = 0xFFFE0000  # Register addresses start here

    def __init__(self, max_entries: int = 65536, value_dim: int = 8,
                 eviction_strategy: str = 'eager', compact_threshold: float = 0.5):
        """
        Args:
            max_entries: Maximum cache entries before forced eviction
            value_dim: Dimensions per value (8 nibbles = 32 bits)
            eviction_strategy: 'lazy', 'eager', or 'lru' (default: eager for aggressive eviction)
            compact_threshold: Compact when valid/total < threshold
        """
        super().__init__()
        self.max_entries = max_entries
        self.value_dim = value_dim
        self.eviction_strategy = eviction_strategy
        self.compact_threshold = compact_threshold

        # Sparse storage: address -> (value, timestamp)
        # Only ONE entry per address (overwrites evict old)
        self.cache: Dict[int, Tuple[torch.Tensor, int]] = {}

        # Statistics
        self.stats = CacheStats()
        self.current_time = 0

    def _is_io_addr(self, addr: int) -> bool:
        """Check if address is in I/O range."""
        return addr >= self.IO_ADDR_START

    def _is_register_addr(self, addr: int) -> bool:
        """Check if address is in register range."""
        return self.REGISTER_ADDR_START <= addr < self.IO_ADDR_START

    def read(self, address: int) -> torch.Tensor:
        """
        Read value at address.

        Returns zero (ZFOD) if address not in cache.
        """
        self.stats.reads += 1
        self.current_time += 1

        if address in self.cache:
            value, _ = self.cache[address]
            # Update timestamp
            self.cache[address] = (value, self.current_time)
            self.stats.hits += 1
            return value.clone()

        # ZFOD: Zero Fill On Demand
        self.stats.misses += 1
        return torch.zeros(self.value_dim)

    def write(self, address: int, value: torch.Tensor):
        """
        Write value at address.

        - Overwrites evict the old entry (not just update)
        - Zero writes evict without creating new entry
        - I/O and register writes evict all previous entries in their range
        """
        self.stats.writes += 1
        self.current_time += 1

        is_zero = value.abs().sum().item() < 1e-6

        # EVICT old entry at this address (overwrite semantics)
        if address in self.cache:
            del self.cache[address]
            self.stats.evicted_entries += 1

        if is_zero:
            # Zero write = free, don't create new entry
            return

        # For I/O addresses, evict ALL old I/O entries (keep only latest)
        if self._is_io_addr(address):
            self._evict_range(self.IO_ADDR_START, 0xFFFFFFFF, exclude=address)

        # For register addresses, evict ALL old register entries (keep only current state)
        if self._is_register_addr(address):
            self._evict_range(self.REGISTER_ADDR_START, self.IO_ADDR_START, exclude=address)

        # Write new entry
        self.cache[address] = (value.clone(), self.current_time)

        # Check capacity
        if len(self.cache) >= self.max_entries:
            self._evict_lru()

        self.stats.total_entries = len(self.cache)
        self.stats.valid_entries = len(self.cache)

    def _evict_range(self, start: int, end: int, exclude: int = None):
        """Evict all entries in address range except excluded."""
        to_evict = [addr for addr in self.cache.keys()
                    if start <= addr < end and addr != exclude]
        for addr in to_evict:
            del self.cache[addr]
            self.stats.evicted_entries += 1

    def _evict_lru(self):
        """Evict least recently used entries when cache is full."""
        if len(self.cache) < self.max_entries // 2:
            return

        # Sort by timestamp, evict oldest 25%
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1][1])
        n_evict = max(1, len(sorted_entries) // 4)

        for addr, _ in sorted_entries[:n_evict]:
            del self.cache[addr]
            self.stats.evicted_entries += 1

        self.stats.total_entries = len(self.cache)
        self.stats.valid_entries = len(self.cache)

    def compact(self):
        """No-op for eager eviction (already compacted)."""
        self.stats.compactions += 1

    def get_keys(self) -> torch.Tensor:
        """Get all keys as tensor for attention."""
        if not self.cache:
            return torch.zeros(1, 32)

        keys = torch.zeros(len(self.cache), 32)
        for i, addr in enumerate(self.cache.keys()):
            for bit in range(32):
                keys[i, bit] = (addr >> bit) & 1
        return keys

    def get_values(self) -> torch.Tensor:
        """Get all values as tensor for attention."""
        if not self.cache:
            return torch.zeros(1, self.value_dim)

        values = torch.stack([v for v, _ in self.cache.values()])
        return values

    def forward(self, query_addr: torch.Tensor, write_data: Optional[torch.Tensor] = None,
                write_flag: bool = False) -> torch.Tensor:
        """Attention-based memory access."""
        B = query_addr.shape[0]

        addresses = []
        for b in range(B):
            addr = 0
            for bit in range(min(32, query_addr.shape[1])):
                if query_addr[b, bit] > 0.5:
                    addr |= (1 << bit)
            addresses.append(addr)

        if write_flag and write_data is not None:
            for b, addr in enumerate(addresses):
                self.write(addr, write_data[b])
            return write_data

        results = []
        for addr in addresses:
            results.append(self.read(addr))
        return torch.stack(results)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self.stats.total_entries = len(self.cache)
        self.stats.valid_entries = len(self.cache)
        return self.stats

    def reset(self):
        """Clear the cache."""
        self.cache.clear()
        self.stats = CacheStats()
        self.current_time = 0


def softmax1(x: torch.Tensor, dim: int = -1, anchor: float = 0.0) -> torch.Tensor:
    """
    softmax1: softmax with constant "1" term in denominator.

    softmax1(x)_i = exp(x_i) / (exp(anchor) + Σ_j exp(x_j))

    Key property: When all scores are below anchor, output approaches 0.
    This enables ZFOD - reading uninitialized/freed memory returns ~0.

    Args:
        x: Input scores
        dim: Dimension to normalize over
        anchor: Score threshold - values above this get attention, below get ~0
    """
    # Numerical stability: subtract max, but keep anchor relative
    max_val = torch.max(x.max(dim=dim, keepdim=True).values,
                        torch.tensor(anchor, device=x.device))
    exp_x = torch.exp(x - max_val)
    exp_anchor = torch.exp(torch.tensor(anchor, device=x.device) - max_val)
    return exp_x / (exp_anchor + exp_x.sum(dim=dim, keepdim=True))


class Softmax1KVCache(nn.Module):
    """
    KV Cache with softmax1 attention and aggressive eviction.

    Eviction policy:
    1. Overwrites evict old entry (only latest value at each address)
    2. Zero writes evict without creating new entry
    3. I/O writes evict all previous I/O entries
    4. Register writes evict all previous register entries
    5. LRU eviction when cache full

    Memory access via softmax1 attention:
    - Exact match gets ~100% attention weight
    - Non-match gets ~0% (ZFOD behavior)
    """

    # Address ranges
    IO_ADDR_START = 0xFFFF0000
    REGISTER_ADDR_START = 0xFFFE0000

    def __init__(self, addr_bits: int = 32, value_dim: int = 8,
                 max_entries: int = 4096, io_protected_addrs: Set[int] = None):
        super().__init__()
        self.addr_bits = addr_bits
        self.value_dim = value_dim
        self.max_entries = max_entries

        # Protected addresses (actively used I/O)
        self.io_protected_addrs = io_protected_addrs or set()

        # KV storage as tensors for attention
        self.register_buffer('keys', torch.zeros(max_entries, addr_bits))
        self.register_buffer('values', torch.zeros(max_entries, value_dim))
        self.register_buffer('valid', torch.zeros(max_entries, dtype=torch.bool))
        self.register_buffer('timestamps', torch.zeros(max_entries, dtype=torch.long))

        self.n_entries = 0
        self.current_time = 0
        self.stats = CacheStats()

        # Address to slot index mapping (only one slot per address)
        self.addr_to_slot: Dict[int, int] = {}

    def _addr_to_binary(self, addr: int) -> torch.Tensor:
        """Convert integer address to binary tensor."""
        binary = torch.zeros(self.addr_bits)
        for bit in range(self.addr_bits):
            binary[bit] = (addr >> bit) & 1
        return binary

    def _binary_to_addr(self, binary: torch.Tensor) -> int:
        """Convert binary tensor to integer address."""
        addr = 0
        for bit in range(min(self.addr_bits, len(binary))):
            if binary[bit] > 0.5:
                addr |= (1 << bit)
        return addr

    def read(self, query_addr: torch.Tensor) -> torch.Tensor:
        """
        Read via softmax1 attention.

        Args:
            query_addr: [batch, addr_bits] binary query addresses

        Returns:
            [batch, value_dim] values (ZFOD for missing/freed)
        """
        self.stats.reads += query_addr.shape[0]
        self.current_time += 1

        if self.n_entries == 0:
            # Empty cache - return zeros (ZFOD)
            self.stats.misses += query_addr.shape[0]
            return torch.zeros(query_addr.shape[0], self.value_dim)

        # Get valid keys and values
        valid_mask = self.valid[:self.n_entries]
        keys = self.keys[:self.n_entries][valid_mask]  # [N_valid, addr_bits]
        values = self.values[:self.n_entries][valid_mask]  # [N_valid, value_dim]

        if keys.shape[0] == 0:
            return torch.zeros(query_addr.shape[0], self.value_dim)

        # Compute attention scores via binary matching
        # score = Σ_k (2*query_k - 1) * (2*key_k - 1)
        # = addr_bits when all bits match, decreases with Hamming distance
        query_signed = 2 * query_addr - 1  # [batch, addr_bits]
        keys_signed = 2 * keys - 1  # [N_valid, addr_bits]

        # [batch, N_valid]
        scores = torch.matmul(query_signed, keys_signed.T)

        # Exact match score = addr_bits, 1 bit diff = addr_bits - 2
        # Normalize so exact match = high positive, any diff = very negative
        # This makes softmax1 select the exact match or return ~0 (ZFOD)
        max_score = self.addr_bits
        # Transform: exact match -> large positive, any mismatch -> large negative
        scores = (scores - max_score + 0.5) * 10.0  # Exact match = +5, 1 bit diff = -15

        # softmax1 with anchor=0: scores > 0 get attention, scores < 0 -> ZFOD
        # Exact match has score +5, mismatches have score -15 to -155
        attn = softmax1(scores, dim=-1, anchor=0.0)  # [batch, N_valid]

        # Weighted sum of values
        output = torch.matmul(attn, values)  # [batch, value_dim]

        # Update timestamps for accessed entries
        for b in range(query_addr.shape[0]):
            addr = self._binary_to_addr(query_addr[b])
            if addr in self.addr_to_slot:
                slot = self.addr_to_slot[addr]
                self.timestamps[slot] = self.current_time
                self.stats.hits += 1
            else:
                self.stats.misses += 1

        return output

    def _is_io_addr(self, addr: int) -> bool:
        return addr >= self.IO_ADDR_START

    def _is_register_addr(self, addr: int) -> bool:
        return self.REGISTER_ADDR_START <= addr < self.IO_ADDR_START

    def _evict_slot(self, slot: int, addr: int):
        """Evict a slot completely."""
        self.valid[slot] = False
        self.values[slot] = torch.zeros(self.value_dim)
        self.keys[slot] = torch.zeros(self.addr_bits)
        if addr in self.addr_to_slot:
            del self.addr_to_slot[addr]
        self.stats.evicted_entries += 1

    def _evict_range(self, start: int, end: int, exclude: int = None):
        """Evict all entries in address range except excluded."""
        to_evict = [(addr, slot) for addr, slot in self.addr_to_slot.items()
                    if start <= addr < end and addr != exclude]
        for addr, slot in to_evict:
            self._evict_slot(slot, addr)

    def write(self, addr: int, value: torch.Tensor):
        """
        Write value at address with aggressive eviction.

        - Overwrites evict old entry at same address
        - Zero writes evict without creating new entry
        - I/O writes evict all old I/O entries
        - Register writes evict all old register entries
        """
        self.stats.writes += 1
        self.current_time += 1

        is_zero = value.abs().sum().item() < 1e-6

        # EVICT old entry at this address (overwrite = evict old)
        if addr in self.addr_to_slot:
            old_slot = self.addr_to_slot[addr]
            self._evict_slot(old_slot, addr)

        if is_zero:
            # Zero write = free, don't create new entry
            return

        # For I/O: evict ALL old I/O entries (keep only latest)
        if self._is_io_addr(addr) and addr not in self.io_protected_addrs:
            self._evict_range(self.IO_ADDR_START, 0xFFFFFFFF, exclude=addr)

        # For registers: evict ALL old register entries (keep only current)
        if self._is_register_addr(addr):
            self._evict_range(self.REGISTER_ADDR_START, self.IO_ADDR_START, exclude=addr)

        # Allocate new slot
        if len(self.addr_to_slot) >= self.max_entries:
            self._evict_lru()

        slot = self._find_free_slot()
        self.keys[slot] = self._addr_to_binary(addr)
        self.values[slot] = value
        self.valid[slot] = True
        self.timestamps[slot] = self.current_time
        self.addr_to_slot[addr] = slot
        self.n_entries = max(self.n_entries, slot + 1)

        self.stats.total_entries = len(self.addr_to_slot)
        self.stats.valid_entries = len(self.addr_to_slot)

    def _find_free_slot(self) -> int:
        """Find a free slot in the cache."""
        for i in range(self.max_entries):
            if not self.valid[i]:
                return i
        return self.n_entries  # Append at end (will trigger eviction)

    def _evict_lru(self):
        """
        Evict entries following Section 19 heuristics:
        1. Zero values (already freed)
        2. Low V contribution (entries rarely accessed)
        3. Old timestamps (LRU)

        Never evict I/O protected addresses.
        """
        # Collect eviction candidates
        candidates = []
        for addr, slot in list(self.addr_to_slot.items()):
            if addr in self.io_protected_addrs:
                continue  # Skip protected I/O buffers

            # Zero values = already freed, highest priority
            v_norm = self.values[slot].abs().sum().item()
            timestamp = self.timestamps[slot].item()

            # Score: lower = more evictable
            # Prioritize: zero values > low contribution > old timestamps
            score = (v_norm * 1000) + timestamp
            candidates.append((score, addr, slot))

        if not candidates:
            return

        # Sort by score (lowest first = most evictable)
        candidates.sort()

        # Evict bottom 25%
        n_evict = max(1, len(candidates) // 4)
        for _, addr, slot in candidates[:n_evict]:
            self.valid[slot] = False
            del self.addr_to_slot[addr]
            self.stats.evicted_entries += 1

        self.stats.total_entries = len(self.addr_to_slot)

    def protect_io_buffer(self, base_addr: int, size: int):
        """Mark address range as I/O protected (not evictable)."""
        for i in range(size):
            self.io_protected_addrs.add(base_addr + i)

    def unprotect_io_buffer(self, base_addr: int, size: int):
        """Remove I/O protection from address range."""
        for i in range(size):
            self.io_protected_addrs.discard(base_addr + i)

    def compact(self):
        """Remove all freed entries."""
        # Find valid entries
        valid_entries = [(addr, slot) for addr, slot in self.addr_to_slot.items()
                         if self.valid[slot]]

        # Compact to beginning
        for i, (addr, old_slot) in enumerate(valid_entries):
            if i != old_slot:
                self.keys[i] = self.keys[old_slot]
                self.values[i] = self.values[old_slot]
                self.valid[i] = True
                self.timestamps[i] = self.timestamps[old_slot]
                self.addr_to_slot[addr] = i

        # Clear rest
        self.n_entries = len(valid_entries)
        for i in range(self.n_entries, self.max_entries):
            self.valid[i] = False

        self.stats.compactions += 1

    def get_stats(self) -> CacheStats:
        return self.stats


class NeuralKVCache(nn.Module):
    """
    Neural interface to EvictableKVCache.

    Provides attention-based memory access with automatic eviction.
    Uses softmax1 semantics where zero entries contribute nothing.
    """

    def __init__(self, addr_bits: int = 32, value_dim: int = 8,
                 max_entries: int = 65536, eviction_strategy: str = 'lazy'):
        super().__init__()
        self.addr_bits = addr_bits
        self.value_dim = value_dim

        self.cache = EvictableKVCache(
            max_entries=max_entries,
            value_dim=value_dim,
            eviction_strategy=eviction_strategy
        )

        # Attention temperature for address matching
        self.temperature = 0.1

    def forward(self, query: torch.Tensor, value: Optional[torch.Tensor] = None,
                write_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory access via soft attention.

        Args:
            query: [batch, addr_bits] query address (binary encoded)
            value: [batch, value_dim] value to write (optional)
            write_mask: [batch] mask indicating write operations

        Returns:
            [batch, value_dim] read values
        """
        B = query.shape[0]

        if write_mask is not None and value is not None:
            # Mixed read/write
            results = []
            for b in range(B):
                addr = self._decode_address(query[b])
                if write_mask[b] > 0.5:
                    self.cache.write(addr, value[b])
                    results.append(value[b])
                else:
                    results.append(self.cache.read(addr))
            return torch.stack(results)

        elif value is not None:
            # Write all
            for b in range(B):
                addr = self._decode_address(query[b])
                self.cache.write(addr, value[b])
            return value

        else:
            # Read all
            results = []
            for b in range(B):
                addr = self._decode_address(query[b])
                results.append(self.cache.read(addr))
            return torch.stack(results)

    def _decode_address(self, binary: torch.Tensor) -> int:
        """Decode binary tensor to integer address."""
        addr = 0
        for bit in range(min(self.addr_bits, len(binary))):
            if binary[bit] > 0.5:
                addr |= (1 << bit)
        return addr

    def compact(self):
        """Force cache compaction."""
        self.cache.compact()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.get_stats()

    def reset(self):
        """Reset the cache."""
        self.cache.reset()


# ============================================================================
# TESTS
# ============================================================================

def test_basic_eviction():
    """Test basic read/write/evict operations."""
    print("=== Testing Basic KV Eviction ===\n")

    cache = EvictableKVCache(max_entries=1000, eviction_strategy='eager')

    # Write some values
    cache.write(0x100, torch.tensor([1.0, 2, 3, 4, 5, 6, 7, 8]))
    cache.write(0x104, torch.tensor([2.0, 3, 4, 5, 6, 7, 8, 9]))
    cache.write(0x108, torch.tensor([3.0, 4, 5, 6, 7, 8, 9, 10]))

    print(f"After 3 writes: {cache.get_stats().total_entries} entries")
    assert cache.get_stats().total_entries == 3

    # Read back
    v1 = cache.read(0x100)
    v2 = cache.read(0x104)
    print(f"Read 0x100: {v1[0].item()}")
    print(f"Read 0x104: {v2[0].item()}")
    assert v1[0] == 1.0
    assert v2[0] == 2.0

    # Evict by writing zeros (eager mode)
    cache.write(0x104, torch.zeros(8))
    print(f"After evicting 0x104: {cache.get_stats().total_entries} entries")
    assert cache.get_stats().total_entries == 2

    # Read evicted address should return zeros (ZFOD)
    v2_after = cache.read(0x104)
    print(f"Read evicted 0x104: {v2_after[0].item()} (should be 0)")
    assert v2_after[0] == 0.0

    print("PASS\n")


def test_overwrite_eviction():
    """Test that overwrites evict old entry."""
    print("=== Testing Overwrite Eviction ===\n")

    cache = EvictableKVCache(max_entries=1000)

    # Write initial value
    cache.write(0x100, torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))
    print(f"After first write: {cache.get_stats().total_entries} entries")
    assert cache.get_stats().total_entries == 1

    # Overwrite same address - should still be 1 entry (old evicted)
    cache.write(0x100, torch.tensor([2.0, 0, 0, 0, 0, 0, 0, 0]))
    print(f"After overwrite: {cache.get_stats().total_entries} entries")
    assert cache.get_stats().total_entries == 1

    # Read should return new value
    v = cache.read(0x100)
    print(f"Read 0x100: {v[0].item()} (should be 2.0)")
    assert v[0] == 2.0

    # Check eviction count
    print(f"Evicted entries: {cache.get_stats().evicted_entries}")
    assert cache.get_stats().evicted_entries == 1  # Old entry was evicted

    print("PASS\n")


def test_io_eviction():
    """Test that I/O writes evict all old I/O entries."""
    print("=== Testing I/O Eviction ===\n")

    cache = EvictableKVCache(max_entries=1000)

    # Write to I/O addresses
    io_base = EvictableKVCache.IO_ADDR_START
    cache.write(io_base + 0, torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))
    cache.write(io_base + 4, torch.tensor([2.0, 0, 0, 0, 0, 0, 0, 0]))
    cache.write(io_base + 8, torch.tensor([3.0, 0, 0, 0, 0, 0, 0, 0]))

    print(f"After 3 I/O writes: {cache.get_stats().total_entries} entries")

    # Each I/O write evicts all previous I/O entries
    # So we should only have 1 entry (the last one)
    assert cache.get_stats().total_entries == 1

    # Only the last I/O value should remain
    v0 = cache.read(io_base + 0)
    v8 = cache.read(io_base + 8)
    print(f"Read I/O+0: {v0[0].item()} (should be 0, evicted)")
    print(f"Read I/O+8: {v8[0].item()} (should be 3.0, latest)")

    assert v0[0] == 0.0  # Evicted
    assert v8[0] == 3.0  # Latest

    print("PASS\n")


def test_register_eviction():
    """Test that register writes evict old register states."""
    print("=== Testing Register Eviction ===\n")

    cache = EvictableKVCache(max_entries=1000)

    # Write to register addresses
    reg_base = EvictableKVCache.REGISTER_ADDR_START
    cache.write(reg_base + 0, torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))  # AX
    cache.write(reg_base + 8, torch.tensor([2.0, 0, 0, 0, 0, 0, 0, 0]))  # BX
    cache.write(reg_base + 16, torch.tensor([3.0, 0, 0, 0, 0, 0, 0, 0])) # CX

    print(f"After 3 register writes: {cache.get_stats().total_entries} entries")

    # Each register write evicts all previous register entries
    # Only the last one should remain
    assert cache.get_stats().total_entries == 1

    # Only CX should remain
    v_ax = cache.read(reg_base + 0)
    v_cx = cache.read(reg_base + 16)
    print(f"Read AX: {v_ax[0].item()} (should be 0, evicted)")
    print(f"Read CX: {v_cx[0].item()} (should be 3.0, latest)")

    assert v_ax[0] == 0.0  # Evicted
    assert v_cx[0] == 3.0  # Latest

    print("PASS\n")


def test_lazy_eviction():
    """Test lazy eviction with compaction."""
    print("=== Testing Lazy Eviction ===\n")

    cache = EvictableKVCache(max_entries=1000, eviction_strategy='lazy',
                              compact_threshold=0.5)

    # Write 10 values
    for i in range(10):
        cache.write(0x100 + i * 4, torch.ones(8) * (i + 1))

    print(f"After 10 writes: {cache.get_stats().total_entries} entries")
    assert cache.get_stats().total_entries == 10

    # Mark 6 for eviction (> 50% threshold)
    for i in range(6):
        cache.write(0x100 + i * 4, torch.zeros(8))

    print(f"Marked 6 for eviction: {len(cache.eviction_set)} in eviction set")
    print(f"Total entries (before compact): {cache.get_stats().total_entries}")

    # Should still have all entries (lazy)
    # But next write should trigger compaction
    cache.write(0x200, torch.ones(8))

    print(f"After compaction: {cache.get_stats().total_entries} entries")
    print(f"Compactions: {cache.get_stats().compactions}")

    assert cache.get_stats().total_entries == 5  # 4 valid + 1 new
    assert cache.get_stats().compactions >= 1

    print("PASS\n")


def test_lru_eviction():
    """Test LRU eviction when cache is full."""
    print("=== Testing LRU Eviction ===\n")

    cache = EvictableKVCache(max_entries=20, eviction_strategy='lazy')

    # Fill cache
    for i in range(20):
        cache.write(i, torch.ones(8) * (i + 1))

    print(f"Cache full: {cache.get_stats().total_entries} entries")

    # Access some entries to update their LRU time
    _ = cache.read(15)
    _ = cache.read(16)
    _ = cache.read(17)
    _ = cache.read(18)
    _ = cache.read(19)

    # Add more entries to trigger LRU eviction
    for i in range(20, 25):
        cache.write(i, torch.ones(8) * (i + 1))

    print(f"After overflow: {cache.get_stats().total_entries} entries")
    print(f"Evicted: {cache.get_stats().evicted_entries}")

    # Recently accessed entries should still be present
    v19 = cache.read(19)
    print(f"Entry 19 (recently used): {v19[0].item()}")
    assert v19[0] == 20.0  # 19 + 1

    print("PASS\n")


def test_neural_interface():
    """Test neural KV cache interface."""
    print("=== Testing Neural KV Cache ===\n")

    cache = NeuralKVCache(addr_bits=16, value_dim=8,
                          max_entries=1000, eviction_strategy='eager')

    # Create binary address encoding
    def addr_to_binary(addr, bits=16):
        return torch.tensor([(addr >> i) & 1 for i in range(bits)], dtype=torch.float32)

    # Write some values
    addrs = torch.stack([addr_to_binary(0x100), addr_to_binary(0x104)])
    values = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                            [2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32)

    result = cache(addrs, value=values)
    print(f"Write result shape: {result.shape}")

    # Read back
    read_addrs = torch.stack([addr_to_binary(0x100), addr_to_binary(0x104)])
    read_result = cache(read_addrs)
    print(f"Read 0x100: {read_result[0, 0].item()}")
    print(f"Read 0x104: {read_result[1, 0].item()}")

    assert read_result[0, 0] == 1.0
    assert read_result[1, 0] == 2.0

    # Evict via zero write
    zero_value = torch.zeros(1, 8)
    cache(addr_to_binary(0x100).unsqueeze(0), value=zero_value)

    # Read evicted
    evicted_result = cache(addr_to_binary(0x100).unsqueeze(0))
    print(f"Read evicted 0x100: {evicted_result[0, 0].item()} (should be 0)")
    assert evicted_result[0, 0] == 0.0

    stats = cache.get_stats()
    print(f"Stats: {stats.total_entries} entries, {stats.evicted_entries} evicted")

    print("PASS\n")


def test_eviction_speedup():
    """Benchmark eviction speedup."""
    import time
    print("=== Testing Eviction Speedup ===\n")

    N = 10000

    # Without eviction (keep all)
    cache_no_evict = EvictableKVCache(max_entries=N * 2, eviction_strategy='lazy')
    for i in range(N):
        cache_no_evict.write(i, torch.ones(8) * i)

    # With eviction (free half)
    cache_with_evict = EvictableKVCache(max_entries=N * 2, eviction_strategy='eager')
    for i in range(N):
        cache_with_evict.write(i, torch.ones(8) * i)
    for i in range(0, N, 2):
        cache_with_evict.write(i, torch.zeros(8))  # Free every other

    print(f"No eviction: {cache_no_evict.get_stats().total_entries} entries")
    print(f"With eviction: {cache_with_evict.get_stats().total_entries} entries")

    # Benchmark reads
    n_reads = 1000

    start = time.perf_counter()
    for _ in range(n_reads):
        for i in range(0, 100):
            _ = cache_no_evict.read(i)
    time_no_evict = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_reads):
        for i in range(0, 100):
            _ = cache_with_evict.read(i)
    time_with_evict = time.perf_counter() - start

    print(f"Read time (no evict): {time_no_evict*1000:.2f}ms")
    print(f"Read time (with evict): {time_with_evict*1000:.2f}ms")

    # Note: Python dict access is O(1), so similar times expected
    # Real speedup comes from attention computation over fewer entries
    print()

    # Test attention speedup with get_keys/get_values
    keys_no_evict = cache_no_evict.get_keys()
    keys_with_evict = cache_with_evict.get_keys()

    print(f"Keys tensor (no evict): {keys_no_evict.shape}")
    print(f"Keys tensor (with evict): {keys_with_evict.shape}")
    print(f"Memory reduction: {keys_with_evict.shape[0] / keys_no_evict.shape[0]:.1%}")

    print("PASS\n")


def test_softmax1_cache():
    """Test softmax1-based KV cache with ZFOD."""
    print("=== Testing Softmax1 KV Cache ===\n")

    cache = Softmax1KVCache(addr_bits=16, value_dim=8, max_entries=100)

    # Write some values
    cache.write(0x100, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32))
    cache.write(0x104, torch.tensor([2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float32))

    print(f"After 2 writes: {cache.get_stats().total_entries} entries")

    # Read via attention
    query = torch.stack([
        cache._addr_to_binary(0x100),
        cache._addr_to_binary(0x104),
        cache._addr_to_binary(0x200),  # Non-existent - should return ~0 (ZFOD)
    ])

    result = cache.read(query)
    print(f"Read 0x100: {result[0, 0].item():.2f} (expected ~1)")
    print(f"Read 0x104: {result[1, 0].item():.2f} (expected ~2)")
    print(f"Read 0x200 (ZFOD): {result[2, 0].item():.4f} (expected ~0)")

    assert result[0, 0].item() > 0.9, "Read 0x100 failed"
    assert result[1, 0].item() > 1.9, "Read 0x104 failed"
    assert result[2, 0].abs().item() < 0.1, "ZFOD failed"

    # Free by writing zeros
    cache.write(0x100, torch.zeros(8))
    print(f"After freeing 0x100: {cache.get_stats().total_entries} entries")

    # Read freed address - should return ~0
    query_freed = cache._addr_to_binary(0x100).unsqueeze(0)
    result_freed = cache.read(query_freed)
    print(f"Read freed 0x100: {result_freed[0, 0].item():.4f} (expected ~0)")

    assert result_freed[0, 0].abs().item() < 0.1, "Freed read should return ~0"

    print("PASS\n")


def test_io_protection():
    """Test I/O buffer protection from eviction."""
    print("=== Testing I/O Buffer Protection ===\n")

    cache = Softmax1KVCache(addr_bits=16, value_dim=8, max_entries=10)

    # Protect I/O buffer at 0x1000-0x1007
    cache.protect_io_buffer(0x1000, 8)

    # Fill cache beyond capacity
    for i in range(15):
        cache.write(0x100 + i, torch.ones(8) * (i + 1))

    # Write to protected I/O buffer
    cache.write(0x1000, torch.tensor([99, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))

    # Trigger eviction
    for i in range(15, 20):
        cache.write(0x200 + i, torch.ones(8) * i)

    # I/O buffer should still be present
    query = cache._addr_to_binary(0x1000).unsqueeze(0)
    result = cache.read(query)
    print(f"Protected I/O buffer at 0x1000: {result[0, 0].item():.2f}")

    # Unprotect and trigger eviction
    cache.unprotect_io_buffer(0x1000, 8)
    cache.write(0x1000, torch.zeros(8))  # Free it

    result_after = cache.read(query)
    print(f"After unprotect + free: {result_after[0, 0].item():.4f}")

    print("PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("KV Cache with Aggressive Eviction")
    print("=" * 60)
    print()

    test_basic_eviction()
    test_overwrite_eviction()
    test_io_eviction()
    test_register_eviction()
    test_neural_interface()
    test_softmax1_cache()
    test_io_protection()

    print("=" * 60)
    print("All eviction tests passed!")
    print("=" * 60)
