"""
Chunk-generic MOD pipeline: nibble-level long division (no fp64).

MOD shares the long-division core with DIV: 8 outer iterations produce both
quotient and remainder simultaneously. The MOD entry point emits the
remainder; the DIV entry point emits the quotient. See
``divmod_longdiv.py`` for algorithm details.

Pipeline:
  Layer 1: ClearDivSlotsFFN — clear scratch slots
  Layer 2: LongDivisionModule — long division → SLOT_REMAINDER, SLOT_QUOTIENT
  Layer 3: EmitDivResultModule — copy SLOT_REMAINDER[*] → RESULT[*]
"""

from torch import nn

from ..chunk_config import ChunkConfig


def build_mod_layers(config: ChunkConfig, opcode: int = 29) -> nn.ModuleList:
    """Build MOD pipeline using nibble-level long division (no fp64).

    All arithmetic is fp32. See ``divmod_longdiv.py``.
    """
    from .divmod_longdiv import build_mod_layers_longdiv
    return build_mod_layers_longdiv(config, opcode)
