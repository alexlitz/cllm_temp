"""
Chunk-generic ALU configuration.

ChunkConfig parameterizes ALU operations by chunk size, enabling the same
algorithms to run at different precisions: 1-bit through 32-bit chunks.

Different chunk sizes map to different floating-point precision requirements:
  fp16: 1-bit, 2-bit (max chunk value 1 or 3, simple carry)
  fp32: 4-bit nibble, 8-bit byte (max value 15 or 255)
  fp64: 16-bit halfword, 32-bit whole-word (MAGIC trick for floor extraction)
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for chunk-generic ALU operations.

    Attributes:
        chunk_bits: Bits per chunk (1, 2, 4, 8, 16, 32).
        total_bits: Total register width (always 32 for C4 VM).
        precision: Float precision ("fp16", "fp32", "fp64").
    """
    chunk_bits: int
    total_bits: int = 32
    precision: str = "fp64"

    @property
    def base(self) -> int:
        """Radix = 2^chunk_bits."""
        return 1 << self.chunk_bits

    @property
    def num_positions(self) -> int:
        """Number of chunk positions = total_bits / chunk_bits."""
        return self.total_bits // self.chunk_bits

    @property
    def chunk_max(self) -> int:
        """Maximum value per chunk = base - 1."""
        return self.base - 1

    @property
    def torch_dtype(self):
        return {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[self.precision]

    @property
    def scale(self) -> float:
        """SwiGLU approximation scale. Must be large enough for sharp steps."""
        if self.precision == "fp16":
            return 10.0
        return 100.0

    @property
    def div_scale(self) -> float:
        """Scale for division quotient computation."""
        if self.precision == "fp16":
            return 10.0
        return 100.0

    @property
    def carry_lookahead_hidden(self) -> int:
        """Hidden units for carry-lookahead: N*(N-1)/2 AND-gates + 4*N clearing."""
        N = self.num_positions
        return N * (N - 1) // 2 + 4 * N

    def __post_init__(self):
        assert self.total_bits % self.chunk_bits == 0
        assert self.chunk_bits in (1, 2, 4, 8, 16, 32)
        assert self.precision in ("fp16", "fp32", "fp64")


# Pre-built configurations
BIT = ChunkConfig(chunk_bits=1, precision="fp16")       # 32 positions
PAIR = ChunkConfig(chunk_bits=2, precision="fp16")       # 16 positions
NIBBLE = ChunkConfig(chunk_bits=4, precision="fp32")     # 8 positions (current)
BYTE = ChunkConfig(chunk_bits=8, precision="fp32")       # 4 positions
HALFWORD = ChunkConfig(chunk_bits=16, precision="fp64")  # 2 positions
WORD = ChunkConfig(chunk_bits=32, precision="fp64")      # 1 position

ALL_CONFIGS = [BIT, PAIR, NIBBLE, BYTE, HALFWORD, WORD]
