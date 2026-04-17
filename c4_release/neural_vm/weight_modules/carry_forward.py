"""
Carry Forward Weight Module.

Sets register carry-forward weights for layer 3.
"""

from typing import List
from .base import WeightModule


class CarryForwardWeights(WeightModule):
    """Weight module for register carry-forward (L3)."""

    @property
    def name(self) -> str:
        return "carry_forward"

    @property
    def layers(self) -> List[int]:
        return [3]

    @property
    def dimensions(self) -> List[int]:
        return []

    def set_weights(self, model) -> None:
        """Set carry-forward weights."""
        pass
