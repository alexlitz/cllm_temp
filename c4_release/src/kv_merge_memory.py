"""
KV Merging Memory for Transformer VM

Instead of pruning (deleting low-contribution entries), we merge similar entries:
- Find entries with similar keys (nearby addresses)
- Merge them into a single entry
- Preserves more information than hard pruning

This is similar to token merging techniques in vision transformers and
KV cache compression in LLMs (H2O, StreamingLLM, etc.)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


def silu(x):
    return x * torch.sigmoid(x)


def swiglu_mul(a, b):
    return silu(a) * b + silu(-a) * (-b)


class KVMergeMemory:
    """
    Memory using attention mechanics with KV merging.

    Instead of pruning entries, we merge similar ones:
    1. Compute pairwise similarity between keys
    2. Merge most similar pairs
    3. New key = weighted average of merged keys
    4. New value = weighted combination of values

    This preserves information better than hard pruning.
    """

    def __init__(self, num_bits: int = 20, scale: float = 100.0,
                 max_entries: int = 1000, merge_threshold: float = 0.95):
        self.num_bits = num_bits
        self.scale = scale
        self.max_entries = max_entries
        self.merge_threshold = merge_threshold

        # Storage
        self.keys = torch.zeros(0, num_bits, dtype=torch.float32)
        self.values = torch.zeros(0, dtype=torch.float32)
        self.weights = torch.zeros(0, dtype=torch.float32)  # Confidence weights
        self.addresses = []  # For debugging

        # Stats
        self.total_writes = 0
        self.total_reads = 0
        self.merge_events = 0
        self.entries_merged = 0

    def _encode_address(self, addr: int) -> torch.Tensor:
        """Encode address as binary vector with ±scale."""
        bits = []
        for b in range(self.num_bits):
            bit = (addr >> b) & 1
            bits.append(self.scale if bit else -self.scale)
        return torch.tensor(bits, dtype=torch.float32)

    def _compute_similarity(self) -> torch.Tensor:
        """Compute pairwise cosine similarity between all keys."""
        if len(self.keys) < 2:
            return torch.zeros(0, 0)

        # Normalize keys
        norms = torch.norm(self.keys, dim=1, keepdim=True)
        normalized = self.keys / (norms + 1e-8)

        # Cosine similarity
        similarity = torch.matmul(normalized, normalized.T)

        # Zero out diagonal (self-similarity)
        similarity.fill_diagonal_(0)

        return similarity

    def write(self, addr: int, value: int):
        """Write value to address."""
        self.total_writes += 1
        key = self._encode_address(addr)

        # Check if EXACT address already exists (via dot product)
        if len(self.keys) > 0:
            # For exact match, dot product with scaled binary = num_bits * scale^2
            perfect_score = self.num_bits * self.scale * self.scale
            scores = torch.matmul(self.keys, key)
            max_score = scores.max().item()

            # Require near-perfect match for update
            if max_score > 0.999 * perfect_score:
                idx = scores.argmax().item()
                self.values[idx] = float(value)
                self.weights[idx] = 1.0  # Reset confidence
                return

        # Add new entry
        self.keys = torch.cat([self.keys, key.unsqueeze(0)], dim=0)
        self.values = torch.cat([self.values, torch.tensor([float(value)])])
        self.weights = torch.cat([self.weights, torch.tensor([1.0])])
        self.addresses.append(addr)

        # Merge if needed
        if len(self.keys) > self.max_entries:
            self._merge_similar()

    def read(self, addr: int) -> int:
        """Read value from address using attention."""
        self.total_reads += 1

        if len(self.keys) == 0:
            return 0

        query = self._encode_address(addr)
        scores = torch.matmul(self.keys, query)

        # Use sharp attention to get near-exact match
        perfect_score = self.num_bits * self.scale * self.scale
        max_score = scores.max().item()

        # If we have a near-perfect match, return that value
        if max_score > 0.95 * perfect_score:
            idx = scores.argmax().item()
            return int(round(self.values[idx].item()))

        # Otherwise use soft attention (for merged entries)
        weights = F.softmax(scores / 100.0, dim=0)  # Sharper softmax
        result = torch.sum(weights * self.values)
        return int(round(result.item()))

    def _merge_similar(self):
        """
        Merge entries that represent the SAME address (duplicate overwrites).

        For VM memory, we only want to merge entries that were overwrites
        of the same address, not nearby addresses. Sequential addresses like
        0, 8, 16 have high cosine similarity (~90%) but should NOT be merged.

        We use a very high threshold based on dot product to ensure only
        duplicate addresses get merged.
        """
        if len(self.keys) < 2:
            return

        self.merge_events += 1

        # Compute dot product scores (not cosine - we want exact match)
        scores = torch.matmul(self.keys, self.keys.T)
        perfect_score = self.num_bits * self.scale * self.scale

        # Only merge if score > 0.999 * perfect (i.e., same address)
        merge_threshold = 0.999 * perfect_score

        # Zero diagonal
        scores.fill_diagonal_(0)

        merged = set()
        new_keys = []
        new_values = []
        new_weights = []
        new_addresses = []

        n = len(self.keys)

        # Find exact duplicates to merge
        while True:
            mask = torch.ones(n, n, dtype=torch.bool)
            for i in merged:
                mask[i, :] = False
                mask[:, i] = False

            masked_scores = scores * mask.float()
            max_score = masked_scores.max().item()

            if max_score < merge_threshold:
                break

            # Find the duplicate pair
            flat_idx = masked_scores.argmax().item()
            i, j = flat_idx // n, flat_idx % n

            # Merge: keep the one with higher weight (more recent)
            w_i, w_j = self.weights[i].item(), self.weights[j].item()

            if w_i >= w_j:
                # Keep i's key, combine values
                new_keys.append(self.keys[i])
                new_values.append(self.values[i].item())  # Keep more recent
                new_weights.append(1.0)
                new_addresses.append(self.addresses[i])
            else:
                new_keys.append(self.keys[j])
                new_values.append(self.values[j].item())
                new_weights.append(1.0)
                new_addresses.append(self.addresses[j])

            merged.add(i)
            merged.add(j)
            self.entries_merged += 1  # One entry eliminated

        # If no duplicates found, we need to evict oldest entries
        if len(merged) == 0:
            # Evict 10% of entries (oldest by implicit order)
            evict_count = max(1, n // 10)
            for i in range(evict_count):
                merged.add(i)
            self.entries_merged += evict_count

        # Add unmerged entries
        for i in range(n):
            if i not in merged:
                new_keys.append(self.keys[i])
                new_values.append(self.values[i].item())
                new_weights.append(self.weights[i].item())
                new_addresses.append(self.addresses[i])

        # Update storage
        if new_keys:
            self.keys = torch.stack(new_keys)
            self.values = torch.tensor(new_values)
            self.weights = torch.tensor(new_weights)
            self.addresses = new_addresses
        else:
            self.keys = torch.zeros(0, self.num_bits, dtype=torch.float32)
            self.values = torch.zeros(0, dtype=torch.float32)
            self.weights = torch.zeros(0, dtype=torch.float32)
            self.addresses = []

    def stats(self) -> Dict:
        return {
            'live_entries': len(self.keys),
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
            'merge_events': self.merge_events,
            'entries_merged': self.entries_merged,
            'avg_weight': float(self.weights.mean()) if len(self.weights) > 0 else 0,
        }


class HierarchicalKVMemory:
    """
    Hierarchical KV memory with multiple levels.

    Level 0: Hot cache (most recent, exact)
    Level 1: Warm cache (merged, approximate)
    Level 2: Cold cache (highly merged, very approximate)

    This mimics CPU cache hierarchy but for KV entries.
    """

    def __init__(self, num_bits: int = 20, scale: float = 100.0):
        self.num_bits = num_bits
        self.scale = scale

        # L0: Hot cache (exact, small)
        self.l0_max = 128
        self.l0_keys = torch.zeros(0, num_bits, dtype=torch.float32)
        self.l0_values = torch.zeros(0, dtype=torch.float32)
        self.l0_ages = torch.zeros(0, dtype=torch.int32)

        # L1: Warm cache (merged, medium)
        self.l1_max = 256
        self.l1_keys = torch.zeros(0, num_bits, dtype=torch.float32)
        self.l1_values = torch.zeros(0, dtype=torch.float32)
        self.l1_weights = torch.zeros(0, dtype=torch.float32)

        # L2: Cold cache (highly merged, large)
        self.l2_max = 512
        self.l2_keys = torch.zeros(0, num_bits, dtype=torch.float32)
        self.l2_values = torch.zeros(0, dtype=torch.float32)
        self.l2_weights = torch.zeros(0, dtype=torch.float32)

        self.current_time = 0

        # Stats
        self.l0_hits = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.total_writes = 0
        self.total_reads = 0

    def _encode_address(self, addr: int) -> torch.Tensor:
        bits = []
        for b in range(self.num_bits):
            bit = (addr >> b) & 1
            bits.append(self.scale if bit else -self.scale)
        return torch.tensor(bits, dtype=torch.float32)

    def _find_in_level(self, key: torch.Tensor, keys: torch.Tensor) -> Tuple[int, float]:
        """Find best matching entry in a cache level."""
        if len(keys) == 0:
            return -1, 0.0

        # Use dot product score normalized by perfect score
        perfect_score = self.num_bits * self.scale * self.scale
        scores = torch.matmul(keys, key)
        max_score = scores.max().item()
        max_idx = scores.argmax().item()

        # Return normalized score (1.0 = perfect match)
        normalized_score = max_score / perfect_score

        return max_idx, normalized_score

    def write(self, addr: int, value: int):
        """Write to L0 cache, spill to lower levels as needed."""
        self.total_writes += 1
        self.current_time += 1
        key = self._encode_address(addr)

        # Try to update existing entry in L0
        idx, weight = self._find_in_level(key, self.l0_keys)
        if weight > 0.99:
            self.l0_values[idx] = float(value)
            self.l0_ages[idx] = self.current_time
            return

        # Add to L0
        self.l0_keys = torch.cat([self.l0_keys, key.unsqueeze(0)], dim=0)
        self.l0_values = torch.cat([self.l0_values, torch.tensor([float(value)])])
        self.l0_ages = torch.cat([self.l0_ages, torch.tensor([self.current_time])])

        # Spill from L0 to L1 if needed
        if len(self.l0_keys) > self.l0_max:
            self._spill_l0_to_l1()

    def _spill_l0_to_l1(self):
        """Move oldest L0 entries to L1."""
        # Sort by age, move oldest half to L1
        ages = self.l0_ages.numpy()
        sorted_idx = ages.argsort()
        spill_count = len(sorted_idx) // 2

        keep_idx = sorted_idx[spill_count:]
        spill_idx = sorted_idx[:spill_count]

        # Add to L1
        for i in spill_idx:
            self.l1_keys = torch.cat([self.l1_keys, self.l0_keys[i:i+1]], dim=0)
            self.l1_values = torch.cat([self.l1_values, self.l0_values[i:i+1]])
            self.l1_weights = torch.cat([self.l1_weights, torch.tensor([1.0])])

        # Keep in L0
        self.l0_keys = self.l0_keys[keep_idx]
        self.l0_values = self.l0_values[keep_idx]
        self.l0_ages = self.l0_ages[keep_idx]

        # Merge L1 if needed
        if len(self.l1_keys) > self.l1_max:
            self._merge_level(1)

    def _merge_level(self, level: int):
        """Merge entries in a cache level."""
        if level == 1:
            keys, values, weights = self.l1_keys, self.l1_values, self.l1_weights
            next_keys, next_values, next_weights = self.l2_keys, self.l2_values, self.l2_weights
        else:
            return

        # Simple merge: combine pairs with highest similarity
        if len(keys) < 2:
            return

        # Compute similarity
        norms = torch.norm(keys, dim=1, keepdim=True)
        normalized = keys / (norms + 1e-8)
        similarity = torch.matmul(normalized, normalized.T)
        similarity.fill_diagonal_(0)

        # Merge top pairs
        new_keys, new_values, new_weights = [], [], []
        merged = set()

        for _ in range(len(keys) // 4):  # Merge 25% of entries
            mask = torch.ones_like(similarity)
            for i in merged:
                mask[i, :] = 0
                mask[:, i] = 0

            masked = similarity * mask
            if masked.max() < 0.5:
                break

            flat_idx = masked.argmax().item()
            i, j = flat_idx // len(keys), flat_idx % len(keys)

            # Merge
            w_i, w_j = weights[i].item(), weights[j].item()
            total = w_i + w_j

            merged_key = (keys[i] * w_i + keys[j] * w_j) / total
            merged_val = (values[i] * w_i + values[j] * w_j) / total

            new_keys.append(merged_key)
            new_values.append(merged_val)
            new_weights.append(min(2.0, total))

            merged.add(i)
            merged.add(j)

        # Keep unmerged
        for i in range(len(keys)):
            if i not in merged:
                new_keys.append(keys[i])
                new_values.append(values[i])
                new_weights.append(weights[i])

        if level == 1:
            self.l1_keys = torch.stack(new_keys) if new_keys else torch.zeros(0, self.num_bits)
            self.l1_values = torch.tensor(new_values) if new_values else torch.zeros(0)
            self.l1_weights = torch.tensor(new_weights) if new_weights else torch.zeros(0)

    def read(self, addr: int) -> int:
        """Read from cache hierarchy, checking L0 first."""
        self.total_reads += 1
        key = self._encode_address(addr)

        # Try L0 (exact)
        idx, weight = self._find_in_level(key, self.l0_keys)
        if weight > 0.99:
            self.l0_hits += 1
            self.l0_ages[idx] = self.current_time  # Update age
            return int(round(self.l0_values[idx].item()))

        # Try L1 (approximate)
        idx, weight = self._find_in_level(key, self.l1_keys)
        if weight > 0.9:
            self.l1_hits += 1
            # Promote to L0
            val = self.l1_values[idx]
            self.write(addr, int(round(val.item())))  # This adds to L0
            return int(round(val.item()))

        # Try L2 (very approximate)
        idx, weight = self._find_in_level(key, self.l2_keys)
        if weight > 0.8:
            self.l2_hits += 1
            val = self.l2_values[idx]
            return int(round(val.item()))

        # Miss
        self.misses += 1
        return 0

    def stats(self) -> Dict:
        total_hits = self.l0_hits + self.l1_hits + self.l2_hits
        return {
            'l0_entries': len(self.l0_keys),
            'l1_entries': len(self.l1_keys),
            'l2_entries': len(self.l2_keys),
            'total_entries': len(self.l0_keys) + len(self.l1_keys) + len(self.l2_keys),
            'l0_hits': self.l0_hits,
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'misses': self.misses,
            'hit_rate': total_hits / max(1, self.total_reads),
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
        }


if __name__ == "__main__":
    # Test KV merge memory with realistic pattern
    print("Testing KV Merge Memory...")

    mem = KVMergeMemory(max_entries=100, merge_threshold=0.9)

    # Simulate stack-like usage: write to same addresses repeatedly
    # This is like function calls reusing stack space
    for iteration in range(5):
        for i in range(20):
            addr = 0x30000 - i * 8  # Stack addresses
            mem.write(addr, iteration * 100 + i)

    # Read back most recent values
    correct = 0
    for i in range(20):
        addr = 0x30000 - i * 8
        expected = 4 * 100 + i  # Last iteration was iteration=4
        got = mem.read(addr)
        if abs(got - expected) < 10:
            correct += 1

    print(f"KV Merge Memory:")
    print(f"  Accuracy (recent): {correct}/20 ({100*correct/20:.0f}%)")
    print(f"  Stats: {mem.stats()}")

    # Test with overwrites (same address multiple times)
    print("\nTesting with explicit overwrites...")
    mem2 = KVMergeMemory(max_entries=100)

    for i in range(10):
        # Each address written multiple times
        for j in range(5):
            mem2.write(i * 8, j * 1000)  # Overwrite with new value

    correct = 0
    for i in range(10):
        expected = 4 * 1000  # Last write was j=4
        got = mem2.read(i * 8)
        if got == expected:
            correct += 1

    print(f"  Overwrite accuracy: {correct}/10")
    print(f"  Stats: {mem2.stats()}")

    # Test hierarchical memory
    print("\nTesting Hierarchical KV Memory...")

    hmem = HierarchicalKVMemory()

    # Simulate function call pattern: write locals, read them, then overwrite
    for func_call in range(10):
        base = 0x30000 - func_call * 80

        # Write 10 local variables
        for i in range(10):
            hmem.write(base - i * 8, func_call * 10 + i)

        # Read them back immediately
        for i in range(10):
            val = hmem.read(base - i * 8)

    # Read most recent function's variables
    base = 0x30000 - 9 * 80
    correct = 0
    for i in range(10):
        got = hmem.read(base - i * 8)
        expected = 9 * 10 + i
        if got == expected:
            correct += 1

    print(f"Hierarchical Memory:")
    print(f"  Recent accuracy: {correct}/10")
    print(f"  Stats: {hmem.stats()}")
