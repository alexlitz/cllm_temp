"""Hugging Face export adapters for the Neural VM.

Adapters that translate the compiled VM's state_dict into the shape /
naming conventions of a stock HF model class. This is shape translation,
not weight preservation — VM-only buffers (alibi slopes, RoPE caches,
ADDR_KEY positional encoding tables, MEM_STORE markers) are dropped, and
HF-only tensors (final RMSNorm weight, router gates, q/k/v biases on
families that have them) are synthesized.
"""

from .mixtral_adapter import (  # noqa: F401
    MIXTRAL_8X7B_SHAPE,
    MixtralShapeMismatchError,
    export_to_mixtral_state_dict,
    infer_mixtral_config_kwargs,
    load_into_mixtral,
)
