"""
Global configuration for Neural VM.

Provides a centralized config system for architectural choices like
positional encoding (ALiBi vs RoPE), attention mechanisms, etc.
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class VMConfig:
    """Configuration for Neural VM architecture.

    Controls positional encoding, attention mechanisms, and other
    architectural choices. Use factory methods for common configs:

    - VMConfig.alibi_mode() - 100% ALiBi (default, backwards compatible)
    - VMConfig.rope_mode() - 100% RoPE
    - VMConfig.hybrid_mode() - ALiBi for L0-L2, RoPE for rest
    """

    # Positional encoding strategy
    positional_encoding: Literal["alibi", "rope", "hybrid"] = "alibi"

    # RoPE configuration
    rope_base: float = 10000.0  # Standard RoPE base frequency

    # Attention configuration
    use_softmax1: bool = False  # Use softmax with temperature=1 instead of default

    @classmethod
    def alibi_mode(cls, **kwargs) -> "VMConfig":
        """Create config using 100% ALiBi positional encoding (default)."""
        return cls(positional_encoding="alibi", **kwargs)

    @classmethod
    def rope_mode(cls, **kwargs) -> "VMConfig":
        """Create config using 100% RoPE positional encoding."""
        return cls(positional_encoding="rope", **kwargs)

    @classmethod
    def hybrid_mode(cls, **kwargs) -> "VMConfig":
        """Create config using hybrid: ALiBi for L0-L2, RoPE for rest."""
        return cls(positional_encoding="hybrid", **kwargs)


# Global config instance
_global_config: VMConfig = None


def get_config() -> VMConfig:
    """Get the global VM configuration.

    Returns the global config, creating a default one if not set.
    Config can be customized via environment variable NEURAL_VM_POS_ENCODING.
    """
    global _global_config
    if _global_config is None:
        pos_encoding = os.environ.get("NEURAL_VM_POS_ENCODING", "alibi")
        if pos_encoding == "rope":
            _global_config = VMConfig.rope_mode()
        elif pos_encoding == "hybrid":
            _global_config = VMConfig.hybrid_mode()
        else:
            _global_config = VMConfig.alibi_mode()
    return _global_config


def set_config(config: VMConfig):
    """Set the global VM configuration.

    Args:
        config: VMConfig instance to use globally
    """
    global _global_config
    _global_config = config


def reset_config():
    """Reset to default config (useful for testing)."""
    global _global_config
    _global_config = None
