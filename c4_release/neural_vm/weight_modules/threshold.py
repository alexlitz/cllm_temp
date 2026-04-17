"""
Threshold Attention Weight Module.

Sets threshold attention weights for layers 0-2.
"""

from typing import List
from .base import WeightModule, WeightConfig


class ThresholdWeights(WeightModule):
    """Weight module for threshold attention (L0-L2)."""

    @property
    def name(self) -> str:
        return "threshold"

    @property
    def layers(self) -> List[int]:
        return [0, 1, 2]

    @property
    def dimensions(self) -> List[int]:
        return []  # Determined by config

    def set_weights(self, model) -> None:
        """Set threshold attention weights.

        Delegates to vm_step._set_threshold_attn for now.
        """
        # TODO: Implement modular threshold setting
        pass
