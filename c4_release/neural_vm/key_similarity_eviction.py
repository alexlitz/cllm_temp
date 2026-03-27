"""
Key Similarity-Based Eviction for Neural VM Context.

Implements the eviction mechanism described in the documentation:
- Computes pairwise cosine similarity between attention keys
- Evicts older entries when similarity > threshold (default 0.99)
- ALiBi recency bias already downweights older duplicates
- Runs automatically every ~120 tokens (~3 VM steps)

This naturally implements latest-write-wins for:
- Memory: Writing to same address produces similar keys
- Registers: Each register marker produces consistent key patterns
- I/O: I/O operations produce similar keys

Zero writes: Attending to zero vector value = not attending, effectively evicted.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Set
from dataclasses import dataclass


@dataclass
class EvictionStats:
    """Statistics for eviction monitoring."""
    total_evicted: int = 0
    evictions_by_type: dict = None
    last_eviction_step: int = 0
    total_steps: int = 0

    def __post_init__(self):
        if self.evictions_by_type is None:
            self.evictions_by_type = {}


class KeySimilarityEviction:
    """
    Key similarity-based eviction for transformer KV cache.

    Evicts entries when:
    1. Two keys have cosine similarity > threshold (default 0.99)
    2. The older entry is evicted (ALiBi already downweights it)

    Runs automatically every eviction_interval tokens (default 120).
    """

    def __init__(self,
                 model,
                 similarity_threshold: float = 0.99,
                 eviction_interval: int = 120,
                 min_context_size: int = 100):
        """
        Args:
            model: AutoregressiveVM with transformer blocks
            similarity_threshold: Cosine similarity threshold for eviction (0.99)
            eviction_interval: Tokens between eviction runs (120 ≈ 3 VM steps)
            min_context_size: Minimum context size before eviction starts
        """
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.eviction_interval = eviction_interval
        self.min_context_size = min_context_size

        self.stats = EvictionStats()
        self.tokens_since_last_eviction = 0

    def should_evict(self, context_len: int) -> bool:
        """Check if eviction should run."""
        if context_len < self.min_context_size:
            return False

        if self.tokens_since_last_eviction >= self.eviction_interval:
            return True

        return False

    def compute_keys(self,
                     token_ids: torch.Tensor,
                     layer_idx: int = 0) -> torch.Tensor:
        """
        Compute attention keys for all positions.

        Args:
            token_ids: [batch, seq_len] token IDs
            layer_idx: Which layer to use for key computation (default 0)

        Returns:
            [seq_len, d_model] keys
        """
        with torch.no_grad():
            # Get embeddings (same as first part of forward())
            x = self.model.embed(token_ids)  # [batch, seq_len, d_model]

            # Add code addr keys (position-dependent metadata)
            self.model._add_code_addr_keys(token_ids, x)

            # Inject MEM_STORE flags
            self.model._inject_mem_store(token_ids, x)

            # Get keys from specified attention layer
            # Keys are computed as x @ W_k
            block = self.model.blocks[layer_idx]
            attn = block.attn

            # Compute keys
            keys = F.linear(x, attn.W_k)  # [batch, seq_len, d_model]

            return keys[0]  # [seq_len, d_model]

    def compute_pairwise_similarity(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            keys: [seq_len, d_model] attention keys

        Returns:
            [seq_len, seq_len] similarity matrix
        """
        # Normalize keys
        keys_norm = F.normalize(keys, p=2, dim=-1)  # [seq_len, d_model]

        # Compute cosine similarity matrix
        similarity = torch.matmul(keys_norm, keys_norm.T)  # [seq_len, seq_len]

        return similarity

    def find_duplicate_pairs(self,
                            similarity: torch.Tensor,
                            protected_range: Tuple[int, int] = None) -> List[Tuple[int, int]]:
        """
        Find pairs of positions with high similarity.

        Args:
            similarity: [seq_len, seq_len] similarity matrix
            protected_range: (start, end) indices to protect from eviction

        Returns:
            List of (keep_idx, evict_idx) pairs where evict_idx is older
        """
        seq_len = similarity.shape[0]
        pairs = []

        # Protected range (e.g., bytecode prefix)
        protect_start, protect_end = protected_range or (0, 0)

        # Find pairs where similarity > threshold
        # Only consider upper triangle (i < j) to avoid duplicates
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Skip if either is protected
                if protect_start <= i < protect_end:
                    continue
                if protect_start <= j < protect_end:
                    continue

                if similarity[i, j] > self.similarity_threshold:
                    # j is newer (higher index), i is older
                    # Evict i (older), keep j (newer)
                    pairs.append((j, i))  # (keep, evict)

        return pairs

    def select_victims(self,
                      pairs: List[Tuple[int, int]]) -> Set[int]:
        """
        Select positions to evict from duplicate pairs.

        Args:
            pairs: List of (keep_idx, evict_idx) tuples

        Returns:
            Set of position indices to evict
        """
        victims = set()

        for keep_idx, evict_idx in pairs:
            # Always evict the older entry (lower index)
            victims.add(evict_idx)

        return victims

    def evict_context(self,
                     context: List[int],
                     token_ids: torch.Tensor,
                     protected_range: Tuple[int, int] = None,
                     layer_idx: int = 0) -> List[int]:
        """
        Evict duplicate entries from context based on key similarity.

        Args:
            context: List of token IDs
            token_ids: [batch, seq_len] tensor of token IDs
            protected_range: (start, end) indices to always keep
            layer_idx: Which layer to use for key computation

        Returns:
            Pruned context (list of token IDs)
        """
        if len(context) < self.min_context_size:
            return context

        # Compute keys
        keys = self.compute_keys(token_ids, layer_idx=layer_idx)  # [seq_len, d_model]

        # Compute similarity matrix
        similarity = self.compute_pairwise_similarity(keys)  # [seq_len, seq_len]

        # Find duplicate pairs
        pairs = self.find_duplicate_pairs(similarity, protected_range=protected_range)

        # Select victims
        victims = self.select_victims(pairs)

        if not victims:
            return context

        # Evict victims
        pruned = [context[i] for i in range(len(context)) if i not in victims]

        # Update stats
        self.stats.total_evicted += len(victims)
        self.stats.last_eviction_step = self.stats.total_steps
        self.tokens_since_last_eviction = 0

        return pruned

    def step(self,
            context: List[int],
            protected_range: Tuple[int, int] = None,
            layer_idx: int = 0) -> List[int]:
        """
        Run one eviction step.

        Args:
            context: Current context (list of token IDs)
            protected_range: (start, end) indices to protect
            layer_idx: Which layer to use for key computation

        Returns:
            Pruned context
        """
        self.stats.total_steps += 1
        self.tokens_since_last_eviction += 1

        # Check if eviction should run
        if not self.should_evict(len(context)):
            return context

        # Run eviction
        token_tensor = torch.tensor([context], dtype=torch.long)
        pruned = self.evict_context(
            context=context,
            token_ids=token_tensor,
            protected_range=protected_range,
            layer_idx=layer_idx
        )

        return pruned

    def get_stats(self) -> EvictionStats:
        """Get eviction statistics."""
        return self.stats

    def reset_stats(self):
        """Reset eviction statistics."""
        self.stats = EvictionStats()
        self.tokens_since_last_eviction = 0


def demo_key_similarity_eviction():
    """Demonstrate key similarity-based eviction."""
    print("=" * 60)
    print("Key Similarity-Based Eviction Demo")
    print("=" * 60)

    print("\nMechanism:")
    print("  1. Compute attention keys for all context positions")
    print("  2. Find pairs with cosine similarity > 0.99")
    print("  3. Evict older entry (ALiBi already downweights it)")
    print("  4. Run automatically every 120 tokens (~3 VM steps)")

    print("\nNatural Latest-Write-Wins:")
    print("  • Memory: Same address → similar keys → evict old")
    print("  • Registers: Same register → similar keys → evict old")
    print("  • I/O: I/O operations → similar keys → evict old")

    print("\nZero Writes:")
    print("  • Writing zero creates zero value embedding")
    print("  • Attending to zero vector = not attending")
    print("  • Effectively evicted without explicit removal")

    print("\nEviction Timing:")
    print("  • Runs every 120 tokens (~3 VM steps)")
    print("  • Keeps cache at 1-10K tokens")
    print("  • Logarithmic growth for arbitrarily long programs")

    print("\nExample:")
    print("  Step 1: Write PC=100 (creates key K1)")
    print("  Step 2: Write PC=105 (creates key K2)")
    print("  Step 3: Write PC=110 (creates key K3)")
    print("")
    print("  All three PC writes produce similar keys (same marker token)")
    print("  cosine(K1, K2) > 0.99, cosine(K2, K3) > 0.99")
    print("")
    print("  Eviction:")
    print("    - K1 (oldest) evicted → only K2, K3 kept")
    print("    - K2 (older) evicted → only K3 kept")
    print("    - Result: Only latest PC value in cache")

    print()


if __name__ == "__main__":
    demo_key_similarity_eviction()
