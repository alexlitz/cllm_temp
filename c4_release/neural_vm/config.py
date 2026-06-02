"""
Global configuration for Neural VM.

Provides a centralized config system for architectural choices like
positional encoding (ALiBi vs RoPE), attention mechanisms, etc.
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional


_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off"})


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUTHY_ENV_VALUES:
        return True
    if normalized in _FALSY_ENV_VALUES:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class VMConfig:
    """Configuration for Neural VM architecture.

    Controls positional encoding, attention mechanisms, and other
    architectural choices. Use factory methods for common configs:

    - VMConfig.alibi_mode() - 100% ALiBi (default, backwards compatible)
    - VMConfig.rope_mode() - 100% RoPE
    - VMConfig.hybrid_mode() - ALiBi for L0-L2, RoPE for rest
    - VMConfig.open_model_like_mode() - RoPE + standard softmax + RMSNorm
    """

    # Positional encoding strategy
    positional_encoding: Literal["alibi", "rope", "hybrid"] = "alibi"

    # RoPE configuration
    rope_base: float = 10000.0  # Standard RoPE base frequency

    # Attention configuration
    attention_normalization: Literal["softmax1", "softmax"] = "softmax1"
    use_softmax1: Optional[bool] = None  # Back-compat constructor alias.

    # Block normalization
    use_rms_norm: bool = False
    rms_norm_eps: float = 1e-6

    # Division implementation (Phase 8.O.3)
    # - "long_div" (default): base-16 long division via threshold counting
    #   per quotient digit (q = Σ_{k=1..15} step(rem - k * div)). This is
    #   the ALiBi-equivalent default: byte-identical to every pre-8.O.3
    #   baseline because the existing FlattenedDivMod pipeline IS the long
    #   division implementation. See BLOG_SPEC.md §"Long Division
    #   Implementation".
    # - "log_softmax1": attention-based 1/n via softmax1 sink + exp-scale
    #   keys at the first 8 tokens (BLOG_SPEC.md §"Via Attention With Log
    #   Sink"). REQUIRES ``attention_normalization == "softmax1"``: the
    #   construction relies on the +1 in the softmax1 denominator giving
    #   weight 1/(1+(n-1)) = 1/n on the sink.
    #   STATUS (Phase 8.O.3): documented as a stub. The current
    #   FlattenedDivMod pipeline is the long-division path; wiring the
    #   alternative attention construction into the L10 DIV staging path
    #   would require new attention heads at L10 with masked exp-scale
    #   key tables at the first 8 tokens (BLOG_SPEC notes this can't
    #   correctly divide in the first 8 tokens) plus query setup from
    #   the (n-1) nibble decomposition. That is invasive enough that
    #   the spec itself sets the default behavior to long division. The
    #   toggle is plumbed so log_softmax1 raises ``NotImplementedError``
    #   at DIV/MOD execution time with a pointer back to BLOG_SPEC.md.
    div_mode: Literal["long_div", "log_softmax1"] = "long_div"

    def __post_init__(self):
        if self.positional_encoding not in {"alibi", "rope", "hybrid"}:
            raise ValueError(
                "positional_encoding must be one of {'alibi', 'rope', 'hybrid'}"
            )

        if self.use_softmax1 is not None:
            self.attention_normalization = (
                "softmax1" if self.use_softmax1 else "softmax"
            )
        if self.attention_normalization not in {"softmax1", "softmax"}:
            raise ValueError(
                "attention_normalization must be one of {'softmax1', 'softmax'}"
            )

        # Keep the legacy attribute useful after construction.
        self.use_softmax1 = self.attention_normalization == "softmax1"

        # Phase 8.O.3 div_mode validation: log_softmax1 requires softmax1
        # attention because the 1/n construction relies on the +1 in the
        # softmax1 denominator giving weight 1/(1+(n-1)) = 1/n on the sink.
        if self.div_mode not in {"long_div", "log_softmax1"}:
            raise ValueError(
                "div_mode must be one of {'long_div', 'log_softmax1'}"
            )
        if (
            self.div_mode == "log_softmax1"
            and self.attention_normalization != "softmax1"
        ):
            raise ValueError(
                "div_mode='log_softmax1' requires attention_normalization="
                "'softmax1' (the 1/n attention construction relies on the "
                "softmax1 sink). Got attention_normalization="
                f"{self.attention_normalization!r}."
            )

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

    @classmethod
    def open_model_like_mode(cls, **kwargs) -> "VMConfig":
        """Create config with RoPE, standard softmax attention, and RMSNorm."""
        return cls(
            positional_encoding="rope",
            attention_normalization="softmax",
            use_rms_norm=True,
            **kwargs,
        )


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
        attention_normalization = os.environ.get(
            "NEURAL_VM_ATTENTION_NORMALIZATION", "softmax1"
        )
        use_rms_norm = _env_bool("NEURAL_VM_USE_RMS_NORM", False)
        rms_norm_eps = _env_float("NEURAL_VM_RMS_NORM_EPS", 1e-6)
        div_mode = os.environ.get("NEURAL_VM_DIV_MODE", "long_div")

        if pos_encoding == "rope":
            factory = VMConfig.rope_mode
        elif pos_encoding == "hybrid":
            factory = VMConfig.hybrid_mode
        else:
            factory = VMConfig.alibi_mode

        _global_config = factory(
            attention_normalization=attention_normalization,
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            div_mode=div_mode,
        )
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
