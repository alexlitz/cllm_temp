"""
Weighted Scoring Eviction for Neural VM.

Reimplements score-based eviction as a clean weighted scoring algorithm,
similar to traditional cache algorithms (ARC, LIRS) but with weights
learned from transformer attention patterns.

Score Formula:
    score(position) = max over layers (
        Σ_features (weight[feature] * feature_value)
    )

    evict if score < threshold
"""

import torch
from typing import Dict, List, Tuple, Optional
from .vm_step import Token, _SetDim as BD


class WeightConfig:
    """Configuration for weighted scoring eviction."""

    def __init__(self, threshold: float = -10.0):
        """
        Args:
            threshold: Eviction threshold. Positions with score < threshold are evicted.
                      Default -10.0 means softmax1(score) ≈ 0.00005
        """
        self.threshold = threshold

        # Layer 3: Register carry-forward (recency-based like LRU)
        self.l3_weights = {
            'is_most_recent_marker': 50.0,      # Most recent PC/AX/SP/BP
            'is_old_marker': -float('inf'),     # Old markers
        }

        # Layer 5: Code fetch (importance-based)
        self.l5_weights = {
            'has_addr_key': 300.0,              # Bytecode/data positions
            'no_addr_key': -float('inf'),       # Non-code positions
        }

        # Layer 15: Memory lookup (validity + importance)
        self.l15_weights = {
            'base': 0.0,                        # Base Q·K score
            'mem_store_valid': 312.5,           # MEM_STORE=1 (valid)
            'mem_store_invalid': -312.5,        # MEM_STORE=0 (overwritten)
            'zfod_offset': -600.0,              # Zero-fill offset
            'addr_match': 300.0,                # Address match bonus
        }

        # Layer 4: PC relay
        self.l4_weights = {
            'is_pc_marker': 50.0,               # PC marker
            'not_pc_marker': -float('inf'),     # Other tokens
        }

        # Layer 6: JMP relay
        self.l6_weights = {
            'is_ax_marker': 50.0,               # AX marker
            'not_ax_marker': -float('inf'),     # Other tokens
        }


class WeightedEviction:
    """
    Weighted scoring eviction algorithm.

    Similar to traditional weighted cache eviction (ARC, LIRS) but:
    - Weights learned from neural network attention patterns
    - Different weight sets per layer
    - Takes maximum score across all layers
    - Semantically aware (understands validity, not just recency)
    """

    def __init__(self, model, config: Optional[WeightConfig] = None):
        """
        Args:
            model: AutoregressiveVM with set weights
            config: Weight configuration (or use defaults)
        """
        self.model = model
        self.config = config or WeightConfig()
        self.n_layers = len(model.blocks)

    def extract_features(self, token_ids: torch.Tensor,
                        embeddings: torch.Tensor,
                        position: int) -> Dict[str, float]:
        """
        Extract feature vector for weighted scoring.

        Args:
            token_ids: [batch, seq_len] token IDs
            embeddings: [batch, seq_len, d_model] embeddings
            position: Position index

        Returns:
            Dictionary of feature_name -> feature_value
        """
        batch_idx = 0  # Assume single batch
        token = token_ids[batch_idx, position].item()
        features = {}

        # === Recency Features (L3 - like LRU) ===
        marker_tokens = [Token.REG_PC, Token.REG_AX, Token.REG_SP,
                        Token.REG_BP, Token.STACK0]

        if token in marker_tokens:
            # Check if this is the most recent occurrence of this marker type
            is_most_recent = self._is_most_recent_marker(
                token_ids[batch_idx], position, token
            )
            features['is_most_recent_marker'] = 1.0 if is_most_recent else 0.0
            features['is_old_marker'] = 0.0 if is_most_recent else 1.0

        # === Validity Features (L15 - semantic understanding) ===
        if token == Token.MEM:
            # MEM_STORE flag indicates if memory is valid (not overwritten)
            mem_store = embeddings[batch_idx, position, BD.MEM_STORE].item()
            features['mem_store_valid'] = mem_store
            features['mem_store_invalid'] = 1.0 - mem_store
            features['zfod_offset'] = 1.0
            features['addr_match'] = 1.0  # Simplified: assume can match

        # === Importance Features (L5 - priority-based) ===
        has_addr_key = False
        for k in range(48):  # ADDR_KEY dimensions
            if embeddings[batch_idx, position, BD.ADDR_KEY + k].abs().item() > 0.1:
                has_addr_key = True
                break
        features['has_addr_key'] = 1.0 if has_addr_key else 0.0
        features['no_addr_key'] = 0.0 if has_addr_key else 1.0

        # === Layer-Specific Features ===
        features['is_pc_marker'] = 1.0 if token == Token.REG_PC else 0.0
        features['not_pc_marker'] = 0.0 if token == Token.REG_PC else 1.0

        features['is_ax_marker'] = 1.0 if token == Token.REG_AX else 0.0
        features['not_ax_marker'] = 0.0 if token == Token.REG_AX else 1.0

        # Base feature (always present)
        features['base'] = 1.0

        return features

    def _is_most_recent_marker(self, token_ids: torch.Tensor,
                               position: int, marker_token: int) -> bool:
        """Check if this is the most recent occurrence of a marker type."""
        # Search forward from this position
        seq_len = token_ids.shape[0]
        for i in range(position + 1, seq_len):
            if token_ids[i].item() == marker_token:
                return False  # Found a more recent one
        return True  # This is the most recent

    def compute_layer_score(self, layer_idx: int, features: Dict[str, float]) -> float:
        """
        Compute weighted score for a specific layer.

        Args:
            layer_idx: Layer index (0-15)
            features: Feature dictionary

        Returns:
            Weighted sum: Σ(weight[feature] * feature_value)
        """
        score = 0.0
        has_any_feature = False

        # Select weight set for this layer
        if layer_idx == 3:
            weights = self.config.l3_weights
        elif layer_idx == 4:
            weights = self.config.l4_weights
        elif layer_idx == 5:
            weights = self.config.l5_weights
        elif layer_idx == 6:
            weights = self.config.l6_weights
        elif layer_idx == 15:
            weights = self.config.l15_weights
        else:
            # Other layers: no contribution
            return -float('inf')

        # Compute weighted sum
        for feature_name, weight in weights.items():
            feature_value = features.get(feature_name, 0.0)

            # Skip if feature is 0 (avoid -inf * 0 = nan)
            if feature_value == 0.0:
                continue

            # If weight is -inf and feature is present, return -inf immediately
            if weight == -float('inf'):
                return -float('inf')

            score += weight * feature_value
            has_any_feature = True

        # If no features matched, this position can't receive attention from this layer
        if not has_any_feature:
            return -float('inf')

        return score

    def compute_score(self, token_ids: torch.Tensor,
                     embeddings: torch.Tensor,
                     position: int) -> float:
        """
        Compute overall score for a position.

        Score = max over layers (weighted sum of features)

        Args:
            token_ids: [batch, seq_len] token IDs
            embeddings: [batch, seq_len, d_model] embeddings
            position: Position index

        Returns:
            Maximum score across all layers
        """
        # Extract features
        features = self.extract_features(token_ids, embeddings, position)

        # Compute score for each layer
        layer_scores = []
        for layer_idx in range(self.n_layers):
            score = self.compute_layer_score(layer_idx, features)
            layer_scores.append(score)

        # Return maximum (any layer might need this position)
        return max(layer_scores)

    def should_evict(self, score: float) -> bool:
        """Check if a position should be evicted based on its score."""
        return score < self.config.threshold

    def get_retention_mask(self, token_ids: torch.Tensor,
                          embeddings: torch.Tensor,
                          protected_range: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Get boolean mask indicating which positions to retain.

        Args:
            token_ids: [batch, seq_len] token IDs
            embeddings: [batch, seq_len, d_model] embeddings
            protected_range: (start, end) indices to always protect

        Returns:
            [batch, seq_len] boolean mask (True = keep, False = evict)
        """
        batch_size, seq_len = token_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        for b in range(batch_size):
            for i in range(seq_len):
                score = self.compute_score(token_ids, embeddings, i)
                mask[b, i] = not self.should_evict(score)

        # Protect specified range (e.g., bytecode prefix)
        if protected_range is not None:
            start, end = protected_range
            mask[:, start:end] = True

        return mask

    def prune_context(self, token_ids: List[int],
                     embeddings: torch.Tensor,
                     protected_prefix_len: int = 0) -> Tuple[List[int], torch.Tensor]:
        """
        Prune context based on weighted scores.

        Args:
            token_ids: List of token IDs
            embeddings: [1, seq_len, d_model] embeddings
            protected_prefix_len: Length of prefix to always protect

        Returns:
            (pruned_token_ids, pruned_embeddings)
        """
        # Convert to tensor
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
            - evictable: Number with score < threshold
            - retained: Number with score >= threshold
            - score_distribution: Min/max/mean scores
            - layer_contributions: Which layers contributed max scores
        """
        batch_size, seq_len = token_ids.shape
        scores = []
        layer_contributions = {i: 0 for i in range(self.n_layers)}

        for i in range(seq_len):
            features = self.extract_features(token_ids, embeddings, i)

            # Get score from each layer
            layer_scores = []
            for layer_idx in range(self.n_layers):
                score = self.compute_layer_score(layer_idx, features)
                layer_scores.append(score)

            # Track which layer gave max score
            max_score = max(layer_scores)
            max_layer = layer_scores.index(max_score)
            layer_contributions[max_layer] += 1

            scores.append(max_score)

        scores_tensor = torch.tensor(scores)
        evictable = (scores_tensor < self.config.threshold).sum().item()

        return {
            'total_entries': seq_len,
            'evictable': evictable,
            'retained': seq_len - evictable,
            'eviction_rate': evictable / seq_len if seq_len > 0 else 0,
            'min_score': scores_tensor.min().item(),
            'max_score': scores_tensor.max().item(),
            'mean_score': scores_tensor.mean().item(),
            'layer_contributions': layer_contributions,
        }


def demo_weighted_eviction():
    """Demonstrate weighted scoring eviction."""
    print("=" * 60)
    print("Weighted Scoring Eviction Demo")
    print("=" * 60)

    print("\nWeight Configuration:")
    config = WeightConfig()

    print("\nLayer 3 (Register Carry - Recency):")
    print(f"  is_most_recent_marker: {config.l3_weights['is_most_recent_marker']}")
    print(f"  is_old_marker: {config.l3_weights['is_old_marker']}")

    print("\nLayer 5 (Code Fetch - Importance):")
    print(f"  has_addr_key: {config.l5_weights['has_addr_key']}")

    print("\nLayer 15 (Memory - Validity):")
    print(f"  mem_store_valid: {config.l15_weights['mem_store_valid']}")
    print(f"  mem_store_invalid: {config.l15_weights['mem_store_invalid']}")
    print(f"  zfod_offset: {config.l15_weights['zfod_offset']}")
    print(f"  addr_match: {config.l15_weights['addr_match']}")

    print("\nScore Examples:")
    print("  Valid MEM:       0 + 312.5 - 600 + 300 = +12.5  → KEEP")
    print("  Overwritten MEM: 0 - 312.5 - 600 + 300 = -612.5 → EVICT")
    print("  Most recent PC:  50.0                  → KEEP")
    print("  Old PC:          -inf                  → EVICT")
    print("  Bytecode:        300.0                 → KEEP")

    print(f"\nEviction threshold: {config.threshold}")
    print(f"  score < {config.threshold} → EVICT")
    print(f"  score >= {config.threshold} → KEEP")
    print()


if __name__ == "__main__":
    demo_weighted_eviction()
