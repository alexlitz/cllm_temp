"""
ALU Weight Module.

Sets arithmetic/logic operation weights for layers 8-12.
"""

from typing import List
from .base import WeightModule


class ALUWeights(WeightModule):
    """Weight module for ALU operations (L8-L12)."""

    @property
    def name(self) -> str:
        return "alu"

    @property
    def layers(self) -> List[int]:
        return [8, 9, 10, 11, 12]

    @property
    def dimensions(self) -> List[int]:
        return []

    def set_weights(self, model) -> None:
        """Set ALU weights."""
        pass
