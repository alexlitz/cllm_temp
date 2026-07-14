"""c4_min residual-band layout: a minimal named-dim allocator.

Scalar-per-register (no nibble one-hots) except OP_ONEHOT. See DESIGN.md (b).
"""
from __future__ import annotations

from .isa import NUM_OPS


class Layout:
    """Assigns residual-band offsets. ``self.D`` is the total residual width."""

    def __init__(self, n_heads: int = 4):
        self._off = 0
        self._names = {}

        # scalar register bands
        self.AX = self._scalar("AX")
        self.SP = self._scalar("SP")
        self.BP = self._scalar("BP")
        self.PC = self._scalar("PC")

        # stack-top mirror + alu scratch
        self.STACK0 = self._scalar("STACK0")
        self.ALU_A = self._scalar("ALU_A")
        self.ALU_B = self._scalar("ALU_B")
        self.ALU_LO = self._scalar("ALU_LO")

        # instruction fetch result
        self.IMM = self._scalar("IMM")
        self.OP = self._band("OP_ONEHOT", NUM_OPS)  # one-hot region [OP, OP+NUM_OPS)

        # emission / decode
        self.OUTPUT = self._scalar("OUTPUT")
        self.HALTED = self._scalar("HALTED")

        # control-flow scratch (PC-driven dispatch, see CONTROL_FLOW.md):
        #   AX_ZERO  == relu(1-AX): 1.0 iff AX==0 (branch predicate for BZ/BNZ)
        self.AX_ZERO = self._scalar("AX_ZERO")

        # constant bias lane (baked to 1.0 in the embedding, never written)
        self.ONE = self._scalar("ONE")

        # position/step index lane (baked by the embedding; read by fetch/carry)
        self.POS = self._scalar("POS")

        # pad D up to a multiple of n_heads
        while self._off % n_heads != 0:
            self._scalar(f"_pad{self._off}")
        self.D = self._off

    def _scalar(self, name: str) -> int:
        return self._band(name, 1)

    def _band(self, name: str, size: int) -> int:
        base = self._off
        self._names[name] = (base, size)
        self._off += size
        return base

    def op_dim(self, opcode: int) -> int:
        """Residual dim for one-hot entry of ``opcode``."""
        return self.OP + opcode

    def __repr__(self) -> str:
        return f"Layout(D={self.D}, bands={list(self._names)})"
