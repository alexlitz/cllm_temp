"""
Standard Eviction Policy Interface for Neural VM Context.

Follows the standard cache eviction algorithm signature:
- on_insert(key, metadata)
- on_access(key)
- select_victims(budget)
- get_score(key)

This matches traditional algorithms like LRU, LFU, ARC, LIRS.
"""

import torch
from typing import Dict, List, Set, Optional, Any
from .vm_step import Token, _SetDim as BD


class EvictionPolicy:
    """
    Base class for eviction policies.

    Standard interface matching traditional cache eviction algorithms.
    """

    def on_insert(self, key: int, metadata: Dict[str, Any]):
        """
        Called when a new entry is inserted into the cache.

        Args:
            key: Position in context (integer)
            metadata: Entry metadata (token, embedding, etc.)
        """
        raise NotImplementedError

    def on_access(self, key: int):
        """
        Called when an entry is accessed (attended to).

        Args:
            key: Position in context
        """
        raise NotImplementedError

    def get_score(self, key: int) -> float:
        """
        Get eviction score for a key.

        Args:
            key: Position in context

        Returns:
            Eviction score (higher = more important to keep)
        """
        raise NotImplementedError

    def select_victims(self, budget: int) -> List[int]:
        """
        Select entries to evict.

        Args:
            budget: Number of entries to evict

        Returns:
            List of keys (positions) to evict
        """
        raise NotImplementedError


class LRUEviction(EvictionPolicy):
    """
    Least Recently Used eviction policy.

    Standard LRU implementation:
    - Track access time for each entry
    - Evict least recently accessed
    """

    def __init__(self):
        self.access_time = {}  # key -> last access time
        self.time = 0

    def on_insert(self, key: int, metadata: Dict[str, Any]):
        """Record insertion time."""
        self.time += 1
        self.access_time[key] = self.time

    def on_access(self, key: int):
        """Update access time."""
        self.time += 1
        self.access_time[key] = self.time

    def get_score(self, key: int) -> float:
        """Return access time (higher = more recent = keep)."""
        return self.access_time.get(key, 0)

    def select_victims(self, budget: int) -> List[int]:
        """Select least recently used entries."""
        # Sort by access time (ascending)
        sorted_keys = sorted(self.access_time.keys(),
                            key=lambda k: self.access_time[k])
        return sorted_keys[:budget]


class LFUEviction(EvictionPolicy):
    """
    Least Frequently Used eviction policy.

    Standard LFU implementation:
    - Track access frequency for each entry
    - Evict least frequently accessed
    """

    def __init__(self):
        self.frequency = {}  # key -> access count

    def on_insert(self, key: int, metadata: Dict[str, Any]):
        """Initialize frequency to 1."""
        self.frequency[key] = 1

    def on_access(self, key: int):
        """Increment frequency."""
        self.frequency[key] = self.frequency.get(key, 0) + 1

    def get_score(self, key: int) -> float:
        """Return frequency (higher = more important = keep)."""
        return self.frequency.get(key, 0)

    def select_victims(self, budget: int) -> List[int]:
        """Select least frequently used entries."""
        sorted_keys = sorted(self.frequency.keys(),
                            key=lambda k: self.frequency[k])
        return sorted_keys[:budget]


class WeightedScoringEviction(EvictionPolicy):
    """
    Weighted scoring eviction policy.

    Standard scoring-based eviction (like ARC, LIRS) but with:
    - Weights learned from transformer attention
    - Multiple feature types (recency, validity, importance)
    - Multi-layer scoring

    Signature matches traditional scoring policies:
    - on_insert(): Record entry metadata
    - on_access(): Update access patterns (optional for this policy)
    - get_score(): Compute weighted score
    - select_victims(): Select lowest-scoring entries
    """

    def __init__(self, weights: Optional[Dict[str, Dict[str, float]]] = None,
                 threshold: float = -10.0):
        """
        Args:
            weights: Per-layer feature weights
            threshold: Eviction threshold (score < threshold → evict)
        """
        self.entries = {}  # key -> metadata
        self.threshold = threshold

        # Default weights from transformer analysis
        self.weights = weights or {
            'l3_recency': {
                'is_most_recent': 50.0,
                'is_old': -float('inf'),
            },
            'l5_importance': {
                'has_addr_key': 300.0,
            },
            'l15_validity': {
                'mem_store_valid': 312.5,
                'mem_store_invalid': -312.5,
                'zfod_offset': -600.0,
                'addr_match': 300.0,
            },
        }

    def on_insert(self, key: int, metadata: Dict[str, Any]):
        """
        Record entry metadata.

        Metadata should contain:
        - token: Token ID
        - embedding: Embedding vector
        - position: Position in sequence
        """
        self.entries[key] = metadata

    def on_access(self, key: int):
        """
        Update on access (no-op for this policy).

        This policy computes scores from metadata, not access patterns.
        """
        pass

    def extract_features(self, key: int) -> Dict[str, float]:
        """
        Extract features for scoring.

        Args:
            key: Entry key (position)

        Returns:
            Feature dictionary
        """
        if key not in self.entries:
            return {}

        metadata = self.entries[key]
        token = metadata['token']
        embedding = metadata.get('embedding')

        features = {}

        # Recency features
        if token in [Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP]:
            # Check if this is the most recent marker of this type
            is_most_recent = self._is_most_recent_of_type(key, token)
            features['is_most_recent'] = 1.0 if is_most_recent else 0.0
            features['is_old'] = 0.0 if is_most_recent else 1.0

        # Validity features (if embedding available)
        if embedding is not None and token == Token.MEM:
            mem_store = embedding[BD.MEM_STORE].item()
            features['mem_store_valid'] = mem_store
            features['mem_store_invalid'] = 1.0 - mem_store
            features['zfod_offset'] = 1.0
            features['addr_match'] = 1.0

        # Importance features
        if embedding is not None:
            has_addr_key = any(embedding[BD.ADDR_KEY + k].abs().item() > 0.1
                              for k in range(48))
            features['has_addr_key'] = 1.0 if has_addr_key else 0.0

        return features

    def _is_most_recent_of_type(self, key: int, token: int) -> bool:
        """Check if this is the most recent marker of this token type."""
        # Check if any later key has the same token type
        for k in self.entries:
            if k > key and self.entries[k]['token'] == token:
                return False
        return True

    def get_score(self, key: int) -> float:
        """
        Compute weighted score for an entry.

        Score = Σ_layers max_features (weight * feature_value)

        Higher score = more important = keep
        Lower score = less important = evict

        Args:
            key: Entry key

        Returns:
            Eviction score
        """
        if key not in self.entries:
            return -float('inf')

        features = self.extract_features(key)

        # Compute score for each layer
        layer_scores = []

        for layer_name, layer_weights in self.weights.items():
            score = 0.0
            has_feature = False

            for feature_name, weight in layer_weights.items():
                feature_value = features.get(feature_name, 0.0)

                if feature_value == 0.0:
                    continue

                if weight == -float('inf'):
                    layer_scores.append(-float('inf'))
                    has_feature = True
                    break

                score += weight * feature_value
                has_feature = True

            if has_feature:
                layer_scores.append(score)

        # Return max score across layers
        return max(layer_scores) if layer_scores else -float('inf')

    def select_victims(self, budget: int) -> List[int]:
        """
        Select entries to evict based on scores.

        Standard victim selection:
        - Compute score for all entries
        - Sort by score (ascending)
        - Return lowest-scoring entries

        Args:
            budget: Number of entries to evict

        Returns:
            List of keys to evict
        """
        # Compute scores for all entries
        scores = {key: self.get_score(key) for key in self.entries}

        # Sort by score (ascending - lowest first)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])

        # Return lowest-scoring entries up to budget
        victims = []
        for key in sorted_keys:
            if scores[key] < self.threshold:
                victims.append(key)
                if len(victims) >= budget:
                    break

        return victims


class AdaptiveEviction(EvictionPolicy):
    """
    Adaptive eviction combining multiple policies.

    Similar to ARC (Adaptive Replacement Cache):
    - Combines recency (LRU) and frequency (LFU)
    - Adapts weights based on performance
    """

    def __init__(self):
        self.lru = LRUEviction()
        self.lfu = LFUEviction()
        self.lru_weight = 0.5  # Adaptive weight
        self.lfu_weight = 0.5

    def on_insert(self, key: int, metadata: Dict[str, Any]):
        """Update both policies."""
        self.lru.on_insert(key, metadata)
        self.lfu.on_insert(key, metadata)

    def on_access(self, key: int):
        """Update both policies."""
        self.lru.on_access(key)
        self.lfu.on_access(key)

    def get_score(self, key: int) -> float:
        """Combine scores from both policies."""
        lru_score = self.lru.get_score(key)
        lfu_score = self.lfu.get_score(key)
        return self.lru_weight * lru_score + self.lfu_weight * lfu_score

    def select_victims(self, budget: int) -> List[int]:
        """Select victims using combined score."""
        all_keys = set(self.lru.access_time.keys()) | set(self.lfu.frequency.keys())
        scores = {key: self.get_score(key) for key in all_keys}
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
        return sorted_keys[:budget]


# Aliases for common patterns
class TransformerEviction(WeightedScoringEviction):
    """
    Eviction policy optimized for transformer KV cache.

    Alias for WeightedScoringEviction with transformer-specific defaults.
    """
    pass


def create_policy(policy_type: str = "weighted", **kwargs) -> EvictionPolicy:
    """
    Factory function for creating eviction policies.

    Args:
        policy_type: Type of policy ("lru", "lfu", "weighted", "adaptive")
        **kwargs: Policy-specific parameters

    Returns:
        EvictionPolicy instance
    """
    policies = {
        "lru": LRUEviction,
        "lfu": LFUEviction,
        "weighted": WeightedScoringEviction,
        "transformer": TransformerEviction,
        "adaptive": AdaptiveEviction,
    }

    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}")

    return policies[policy_type](**kwargs)
