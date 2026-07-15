"""Residual layout for the UNIVERSAL NIBBLE VM (BLOG_SPEC-faithful base).

This extends the foundation nibble layout (``blogspec_layout.NibbleLayout``: 16
4-bit nibble dims per register, §Internal Representation) with the scratch bands
the *baked* VM step needs — the port of the proven scalar dispatch/universal
substrate onto the nibble representation.

The canonical persistent VM state is the **nibble bands** (``PC AX SP BP
STACK0``), exactly the spec's representation. Everything the dispatch machinery
needs is derived from those nibbles *within a step* and does not persist:

  scalar VALUE LANES (recomputed from the nibble bands each step)
    ``PC_VAL AX_VAL SP_VAL BP_VAL STK_VAL`` — the scalar image of each register,
    materialised by a nibble->scalar recompose FFN (``compile_nibble_to_scalar``)
    so the proven scalar dispatch/fetch/decode algebra (``control.py`` /
    ``universal.py``) runs unchanged. A scalar lane is ``Σ_j 16^j * nibble_j``.

  DATA-MEMORY code table (the program lives here as INPUT, not baked)
    ``CODE_OP[i] CODE_IMM[i]`` — one ``[op, imm]`` cell per slot (DESIGN.md
    WIDTH=2). ``load_program`` writes these into the initial state; the weights
    never see the program. This is what makes ONE model run ANY program.

  fetch / decode / dispatch scratch (0/1 or small ints, recomputed each step)
    ``PC_IS[i]``  — PC one-hot (exact triangular pulse of ``PC_VAL``).
    ``AX_ZERO``   — (AX == 0) predicate for BZ/BNZ.
    ``OP_VAL``    — fetched scalar opcode (= Σ_i PC_IS[i]*CODE_OP[i]).
    ``IMM``       — fetched scalar immediate.
    ``OP_IS[op]`` — decoded opcode one-hot (exact pulse of ``OP_VAL``).
    ``HALTED``    — latched by HALT; drives the run loop's stop + HALT token.

    Dispatch writes the next-state directly into the value lanes (SET semantics);
    the driver's frame round-trip (``nibble_vm._emit_and_reembed``) writes those
    scalars back into the canonical nibble bands. No separate NEXT_* lanes.

The residual is one position wide per VM step (the step-block is applied
recurrently by the vanilla token round-trip driver); attention is only used for
the register-ingest gather that reconstructs the nibble state from the emitted
30-token frame (the spec's "write registers each step, retrieve by attending").
"""
from __future__ import annotations

from typing import List

from . import isa
from .blogspec_layout import NIB_PER_REG, NUM_CTX

# Context frames (re-exported for the emit path; same ids as the foundation).
from .blogspec_layout import (CTX_NONE, CTX_PC, CTX_AX, CTX_SP, CTX_BP, CTX_MEM)


class NibbleVMLayout:
    """Named residual allocator for the universal nibble VM step-block.

    Registers are 16 nibble dims each (the spec representation). Scalar value
    lanes + dispatch scratch are added so the proven scalar dispatch runs on the
    nibble state. ``code_size`` slots hold the program as DATA (universal fetch).
    """

    def __init__(self, code_size: int, n_heads: int = 4):
        self._off = 0
        self._names = {}
        self.code_size = code_size

        # --- canonical nibble register bands (16 dims each, §569-571) ---------
        self.PC = self._band("PC", NIB_PER_REG)
        self.AX = self._band("AX", NIB_PER_REG)
        self.SP = self._band("SP", NIB_PER_REG)
        self.BP = self._band("BP", NIB_PER_REG)
        self.STACK0 = self._band("STACK0", NIB_PER_REG)

        # --- per-token ingest scratch (foundation) ----------------------------
        self.CUR_NIB = self._band("CUR_NIB", NIB_PER_REG)  # this token's nibbles
        self.CTX = self._band("CTX", NUM_CTX)              # marker context one-hot
        self.BYTE_OFS = self._scalar("BYTE_OFS")

        # --- scalar value lanes (recomposed from the nibble bands each step) ---
        self.PC_VAL = self._scalar("PC_VAL")
        self.AX_VAL = self._scalar("AX_VAL")
        self.SP_VAL = self._scalar("SP_VAL")
        self.BP_VAL = self._scalar("BP_VAL")
        self.STK_VAL = self._scalar("STK_VAL")
        # BP's LOW BYTE only (nibbles 0,1) — LEA is an 8-bit op ``AX=(BP+imm)&0xFF``
        # so it adds the frame-pointer's low byte, keeping AX within one mod-256.
        self.BP_LOW = self._scalar("BP_LOW")

        # --- data-memory code table (the PROGRAM, loaded as INPUT) ------------
        self.CODE_OP = [self._scalar(f"CODE_OP_{i}") for i in range(code_size)]
        self.CODE_IMM = [self._scalar(f"CODE_IMM_{i}") for i in range(code_size)]

        # --- fetch / decode scratch -------------------------------------------
        self.PC_IS = [self._scalar(f"PC_IS_{i}") for i in range(code_size)]
        self.AX_ZERO = self._scalar("AX_ZERO")
        self.OP_VAL = self._scalar("OP_VAL")
        self.IMM = self._scalar("IMM")
        self.OP_IS = self._band("OP_IS", isa.NUM_OPS)   # decoded opcode one-hot

        # The dispatch writes the next-state directly into the value lanes above
        # (SET semantics, per universal.py); the driver's frame round-trip
        # (nibble_vm._emit_and_reembed) writes those scalars back into the nibble
        # bands. No separate NEXT_* lanes are needed.
        self.HALTED = self._scalar("HALTED")
        self.ONE = self._scalar("ONE")                  # constant 1.0 lane

        while self._off % n_heads != 0:
            self._scalar(f"_pad{self._off}")
        self.D = self._off
        self.n_heads = n_heads

    # -- band helpers --------------------------------------------------------
    def _scalar(self, name: str) -> int:
        return self._band(name, 1)

    def _band(self, name: str, size: int) -> int:
        base = self._off
        self._names[name] = (base, size)
        self._off += size
        return base

    # -- convenience ---------------------------------------------------------
    def reg_nibble(self, reg_base: int, j: int) -> int:
        return reg_base + j

    def ctx_dim(self, ctx: int) -> int:
        return self.CTX + ctx

    def band(self, name: str):
        return self._names[name]

    # (nibble_base, value_lane) for each register the VM carries — used by the
    # recompose bridge and the driver's frame round-trip.
    def reg_pairs(self):
        return [
            (self.PC, self.PC_VAL),
            (self.AX, self.AX_VAL),
            (self.SP, self.SP_VAL),
            (self.BP, self.BP_VAL),
            (self.STACK0, self.STK_VAL),
        ]

    def __repr__(self) -> str:
        return (f"NibbleVMLayout(D={self.D}, code_size={self.code_size}, "
                f"bands={list(self._names)[:8]}...)")
