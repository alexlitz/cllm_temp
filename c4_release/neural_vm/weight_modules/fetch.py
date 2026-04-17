"""
Instruction Fetch Weight Module.

Sets instruction fetch weights for layer 5.
"""

from typing import List
from .base import WeightModule


class FetchWeights(WeightModule):
    """Weight module for instruction fetch (L5)."""

    @property
    def name(self) -> str:
        return "fetch"

    @property
    def layers(self) -> List[int]:
        return [5]

    @property
    def dimensions(self) -> List[int]:
        return []

    def set_weights(self, model) -> None:
        """Set fetch weights."""
        pass
