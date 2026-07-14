"""Spec-faithful NIBBLE residual layout (BLOG_SPEC §Internal Representation).

    "We represent each [32-bit value] as 16 4-bit nibbles and operate on these.
     ... Registers are also all loaded in the same manner in different dims."

So every register (PC, AX, SP, BP) is a band of **16 dims**, one dim per nibble,
little-endian: dim ``REG+j`` holds the integer value (0..15) of the ``j``-th
4-bit nibble of the register. This is NOT a scalar-per-register — it is the
spec's nibble decomposition. An 8-bit value uses nibbles 0 and 1; the remaining
14 nibble dims are reserved (and stay 0), so the identical layout scales to the
full 32-bit ISA without changing shapes.

The foundation program is 8-bit, so the ALU gadgets here operate on nibbles 0
and 1; higher nibbles are carried but always 0 for this slice.

Bands
-----
Registers (16 nibble dims each): ``PC AX SP BP``.
Working stack top mirror (16 nibble dims): ``STACK0`` — the pushed value, so
    ``ADD``/``SUB`` can read "pop" without a separate memory attention (the
    foundation slice keeps the stack shallow; deeper stacks use the SP-indexed
    memory extension, out of scope here).
Scratch used by the emit path and the token-ingest FFN:
    ``CUR_NIB`` (16 dims)   — nibbles of the byte value the *current* token
                              carries (0 for marker tokens).
    ``CTX``     (NUM_CTX)   — one-hot "which register frame are we inside", set
                              by the marker tokens so a following byte token
                              knows which register+offset it fills.
    ``BYTE_OFS`` (1)        — how many byte-tokens seen since the last marker
                              (0..3); selects which byte (=2 nibbles) of the
                              register the current token fills.
    ``ONE`` (1)             — constant 1.0 lane (baked into every embedding row).
"""
from __future__ import annotations

NIB_PER_REG = 16          # 16 4-bit nibbles per 32-bit register (§569-571)

# Context frames a marker opens (which register the following bytes fill).
CTX_NONE, CTX_PC, CTX_AX, CTX_SP, CTX_BP, CTX_MEM = range(6)
NUM_CTX = 6


class NibbleLayout:
    """Named residual-band allocator; every register band is 16 nibble dims."""

    def __init__(self, n_heads: int = 4):
        self._off = 0
        self._names = {}

        # 32-bit registers, each 16 nibble dims (little-endian).
        self.PC = self._band("PC", NIB_PER_REG)
        self.AX = self._band("AX", NIB_PER_REG)
        self.SP = self._band("SP", NIB_PER_REG)
        self.BP = self._band("BP", NIB_PER_REG)
        # stack-top mirror (the value a PSH pushed), also 16 nibble dims.
        self.STACK0 = self._band("STACK0", NIB_PER_REG)

        # per-token ingest scratch
        self.CUR_NIB = self._band("CUR_NIB", NIB_PER_REG)  # nibbles of this token
        self.CTX = self._band("CTX", NUM_CTX)              # which reg frame we're in
        self.BYTE_OFS = self._scalar("BYTE_OFS")           # byte index since marker

        self.ONE = self._scalar("ONE")                     # constant lane

        while self._off % n_heads != 0:
            self._scalar(f"_pad{self._off}")
        self.D = self._off

    # -- band helpers --------------------------------------------------------
    def _scalar(self, name: str) -> int:
        return self._band(name, 1)

    def _band(self, name: str, size: int) -> int:
        base = self._off
        self._names[name] = (base, size)
        self._off += size
        return base

    # -- convenience accessors ----------------------------------------------
    def reg_nibble(self, reg_base: int, j: int) -> int:
        """Residual dim of nibble ``j`` (0..15) of the register at ``reg_base``."""
        return reg_base + j

    def ctx_dim(self, ctx: int) -> int:
        return self.CTX + ctx

    def band(self, name: str):
        return self._names[name]

    def __repr__(self) -> str:
        return f"NibbleLayout(D={self.D}, bands={list(self._names)})"
