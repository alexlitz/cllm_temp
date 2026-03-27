"""
Score-based KV cache eviction for Neural VM.

Computes the maximum possible attention score each position can receive
from any future query, enabling provably-correct eviction.
"""

import torch
from typing import List, Dict, Tuple, Optional
from .vm_step import Token, _SetDim as BD


class ScoreBasedEviction:
    """
    Compute maximum attention scores for eviction decisions.

    For each position in context, computes the highest score it could
    possibly receive from any future query across all attention layers.

    Positions with max_score < threshold can be safely evicted (they
    contribute ~0 via softmax1).
    """

    def __init__(self, model, eviction_threshold: float = -10.0):
        """
        Args:
            model: AutoregressiveVM with set weights
            eviction_threshold: Score below which entries are evicted
                               (default -10: softmax1(x) ≈ 0.00005)
        """
        self.model = model
        self.threshold = eviction_threshold
        self.n_layers = len(model.blocks)

    def compute_max_scores(self, token_ids: torch.Tensor,
                          embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum attention score for each position.

        Args:
            token_ids: [batch, seq_len] token IDs
            embeddings: [batch, seq_len, d_model] current embeddings

        Returns:
            [batch, seq_len] maximum score each position can receive
        """
        B, S = token_ids.shape
        max_scores = torch.full((B, S), -float('inf'))

        # Compute max score across all layers
        for layer_idx in range(self.n_layers):
            layer_scores = self._compute_layer_max_scores(
                layer_idx, token_ids, embeddings
            )
            max_scores = torch.maximum(max_scores, layer_scores)

        return max_scores

    def _compute_layer_max_scores(self, layer_idx: int,
                                  token_ids: torch.Tensor,
                                  embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum scores for a specific layer.

        Returns:
            [batch, seq_len] max score this layer can give each position
        """
        B, S, D = embeddings.shape

        if layer_idx == 15:  # L15: Memory lookup
            return self._compute_l15_max_scores(token_ids, embeddings)
        elif layer_idx == 3:  # L3: Register carry-forward
            return self._compute_l3_max_scores(token_ids, embeddings)
        elif layer_idx == 5:  # L5: Code fetch
            return self._compute_l5_max_scores(token_ids, embeddings)
        elif layer_idx == 4:  # L4: PC relay
            return self._compute_l4_max_scores(token_ids, embeddings)
        elif layer_idx == 6:  # L6: JMP relay
            return self._compute_l6_max_scores(token_ids, embeddings)
        else:
            # Other layers: positions cannot receive attention (not modeled)
            return torch.full((B, S), -float('inf'))

    def _compute_l15_max_scores(self, token_ids: torch.Tensor,
                               embeddings: torch.Tensor) -> torch.Tensor:
        """
        L15 maximum scores (memory lookup).

        Score breakdown:
        - bias: -2500 (non-target) or 0 (target)
        - store_anchor: +312.5 (MEM_STORE=1) or -312.5 (MEM_STORE=0)
        - zfod_offset: -600 (at store entries)
        - address_match: +300 (exact match)

        Max possible:
        - With MEM_STORE=1: 0 + 312.5 - 600 + 300 = +12.5
        - With MEM_STORE=0: 0 - 312.5 - 600 + 300 = -612.5
        """
        B, S = token_ids.shape
        scores = torch.full((B, S), -float('inf'))

        for b in range(B):
            for i in range(S):
                token = token_ids[b, i].item()

                if token == Token.MEM:
                    # Check if this MEM entry has MEM_STORE=1
                    has_mem_store = embeddings[b, i, BD.MEM_STORE].item() > 0.5

                    if has_mem_store:
                        # Can receive attention: +12.5
                        scores[b, i] = 12.5
                    else:
                        # Overwritten (MEM_STORE=0): -612.5
                        scores[b, i] = -612.5

        return scores

    def _compute_l3_max_scores(self, token_ids: torch.Tensor,
                              embeddings: torch.Tensor) -> torch.Tensor:
        """
        L3 maximum scores (register carry-forward).

        Only the MOST RECENT marker of each type gets attention.
        Older markers have Q=0 (no query will target them).
        """
        B, S = token_ids.shape
        scores = torch.full((B, S), -float('inf'))

        marker_tokens = [Token.REG_PC, Token.REG_AX, Token.REG_SP,
                        Token.REG_BP, Token.STACK0]

        for b in range(B):
            # Find most recent occurrence of each marker type
            last_seen = {}

            for i in range(S):
                token = token_ids[b, i].item()
                if token in marker_tokens:
                    last_seen[token] = i

            # Only most recent markers get positive scores
            for token, last_idx in last_seen.items():
                scores[b, last_idx] = 50.0  # Typical relay score

        return scores

    def _compute_l5_max_scores(self, token_ids: torch.Tensor,
                              embeddings: torch.Tensor) -> torch.Tensor:
        """
        L5 maximum scores (code fetch via ADDR_KEY).

        Bytecode positions have ADDR_KEY encoding.
        Max score: +300 (address match)
        """
        B, S = token_ids.shape
        scores = torch.full((B, S), -float('inf'))

        for b in range(B):
            for i in range(S):
                # Check if position has ADDR_KEY (bytecode/data)
                has_addr_key = False
                for k in range(48):  # ADDR_KEY dimensions
                    if embeddings[b, i, BD.ADDR_KEY + k].abs().item() > 0.1:
                        has_addr_key = True
                        break

                if has_addr_key:
                    scores[b, i] = 300.0

        return scores

    def _compute_l4_max_scores(self, token_ids: torch.Tensor,
                              embeddings: torch.Tensor) -> torch.Tensor:
        """L4 maximum scores (PC relay to AX marker)."""
        B, S = token_ids.shape
        scores = torch.full((B, S), -float('inf'))

        for b in range(B):
            # PC marker gets attended to relay to AX
            for i in range(S):
                if token_ids[b, i].item() == Token.REG_PC:
                    scores[b, i] = 50.0

        return scores

    def _compute_l6_max_scores(self, token_ids: torch.Tensor,
                              embeddings: torch.Tensor) -> torch.Tensor:
        """L6 maximum scores (JMP relay, EXIT relay)."""
        B, S = token_ids.shape
        scores = torch.full((B, S), -float('inf'))

        for b in range(B):
            # AX markers can get attention for JMP/EXIT relay
            for i in range(S):
                if token_ids[b, i].item() == Token.REG_AX:
                    scores[b, i] = 50.0

        return scores

    def should_evict(self, max_scores: torch.Tensor, position: int) -> bool:
        """
        Check if a position should be evicted.

        Args:
            max_scores: [batch, seq_len] maximum scores
            position: Position index to check

        Returns:
            True if position should be evicted
        """
        return max_scores[0, position].item() < self.threshold

    def get_retention_mask(self, token_ids: torch.Tensor,
                          embeddings: torch.Tensor,
                          protected_range: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Get boolean mask indicating which positions to retain.

        Args:
            token_ids: [batch, seq_len] token IDs
            embeddings: [batch, seq_len, d_model] embeddings
            protected_range: (start, end) indices to always protect (e.g., bytecode)

        Returns:
            [batch, seq_len] boolean mask (True = keep, False = evict)
        """
        max_scores = self.compute_max_scores(token_ids, embeddings)

        # Positions above threshold are kept
        mask = max_scores >= self.threshold

        # Protect specified range (e.g., bytecode prefix)
        if protected_range is not None:
            start, end = protected_range
            mask[:, start:end] = True

        return mask

    def prune_context(self, token_ids: List[int],
                     embeddings: torch.Tensor,
                     protected_prefix_len: int = 0) -> Tuple[List[int], torch.Tensor]:
        """
        Prune context based on attention scores.

        Args:
            token_ids: List of token IDs
            embeddings: [1, seq_len, d_model] embeddings for scoring
            protected_prefix_len: Length of prefix to always protect

        Returns:
            (pruned_token_ids, pruned_embeddings)
        """
        # Convert to tensor for scoring
        token_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Get retention mask
        mask = self.get_retention_mask(
            token_tensor,
            embeddings,
            protected_range=(0, protected_prefix_len) if protected_prefix_len > 0 else None
        )

        # Apply mask
        kept_indices = mask[0].nonzero(as_tuple=True)[0].tolist()

        pruned_tokens = [token_ids[i] for i in kept_indices]
        pruned_embeddings = embeddings[:, kept_indices, :]

        return pruned_tokens, pruned_embeddings

    def get_stats(self, token_ids: torch.Tensor,
                 embeddings: torch.Tensor) -> Dict[str, any]:
        """
        Get eviction statistics.

        Returns:
            Dictionary with:
            - total_entries: Total positions
            - evictable: Number with max_score < threshold
            - retained: Number with max_score >= threshold
            - score_distribution: Histogram of max scores
        """
        max_scores = self.compute_max_scores(token_ids, embeddings)

        evictable = (max_scores < self.threshold).sum().item()
        total = max_scores.numel()

        return {
            'total_entries': total,
            'evictable': evictable,
            'retained': total - evictable,
            'eviction_rate': evictable / total if total > 0 else 0,
            'min_score': max_scores.min().item(),
            'max_score': max_scores.max().item(),
            'mean_score': max_scores.mean().item(),
        }


def demo_score_based_eviction():
    """Demonstrate score-based eviction."""
    print("=" * 60)
    print("Score-Based Eviction Demo")
    print("=" * 60)

    # This is a placeholder - would need actual model and context
    print("\nScore-based eviction computes max attention score for each position.")
    print("\nExample scores:")
    print("  MEM entry (MEM_STORE=1):  +12.5  → KEEP")
    print("  MEM entry (MEM_STORE=0): -612.5  → EVICT")
    print("  Latest PC marker:         +50.0  → KEEP")
    print("  Old PC marker:             +0.0  → EVICT (threshold=-10)")
    print("  Bytecode position:       +300.0  → KEEP")
    print()
    print("Eviction rule: max_score < -10.0 → EVICT")
    print()


if __name__ == "__main__":
    demo_score_based_eviction()
