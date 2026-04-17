"""
Embedding Weight Module.

Sets token embedding weights for the Neural VM.
"""

from typing import List
from .base import WeightModule, WeightConfig, get_dimension_registry


class EmbeddingWeights(WeightModule):
    """Weight module for token embeddings."""

    @property
    def name(self) -> str:
        return "embedding"

    @property
    def layers(self) -> List[int]:
        return []  # Embedding is not in transformer layers

    @property
    def dimensions(self) -> List[int]:
        return list(range(512))  # Uses all dimensions

    def set_weights(self, model) -> None:
        """Set embedding weights."""
        BD = get_dimension_registry()
        embed = model.embed.embed.weight
        embed.zero_()

        V = model.vocab_size

        # CONST dimension for all tokens
        for tok in range(V):
            embed[tok, BD.CONST] = 1.0

        # Register markers
        from neural_vm.vm_step import Token
        for tok, dim in [
            (Token.REG_PC, BD.MARK_PC),
            (Token.REG_AX, BD.MARK_AX),
            (Token.REG_SP, BD.MARK_SP),
            (Token.REG_BP, BD.MARK_BP),
            (Token.MEM, BD.MARK_MEM),
            (Token.CODE_START, BD.MARK_CS),
        ]:
            embed[tok, dim] = 1.0
            embed[tok, BD.IS_MARK] = 1.0

        # STACK0 marker (no IS_MARK)
        embed[Token.STACK0, BD.MARK_STACK0] = 1.0

        # Step-end markers
        for tok in [Token.STEP_END, Token.DATA_END, Token.HALT]:
            embed[tok, BD.MARK_SE] = 1.0
            embed[tok, BD.IS_MARK] = 1.0

        embed[Token.STEP_END, BD.MARK_SE_ONLY] = 1.0

        # Byte embeddings
        for b in range(256):
            embed[b, BD.IS_BYTE] = 1.0
            embed[b, BD.EMBED_LO + (b & 0xF)] = 1.0
            embed[b, BD.EMBED_HI + ((b >> 4) & 0xF)] = 1.0
            embed[b, BD.CLEAN_EMBED_LO + (b & 0xF)] = 1.0
            embed[b, BD.CLEAN_EMBED_HI + ((b >> 4) & 0xF)] = 1.0
