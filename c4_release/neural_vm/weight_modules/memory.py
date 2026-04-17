"""
Memory Operations Weight Module.

Sets memory read/write weights for layers 14-15.
"""

from typing import List
from .base import WeightModule


class MemoryWeights(WeightModule):
    """Weight module for memory operations (L14-L15)."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def layers(self) -> List[int]:
        return [14, 15]

    @property
    def dimensions(self) -> List[int]:
        return []

    def set_weights(self, model) -> None:
        """Set memory operation weights."""
        pass
