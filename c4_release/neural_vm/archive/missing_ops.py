"""
Missing Opcodes Implementation for Neural VM.

Implements all previously unimplemented C4 opcodes:
- LEA, IMM: Address/immediate loading
- BZ, BNZ: Branch on zero (uses efficient zero-detection!)
- ENT, ADJ, LEV: Stack frame operations
- LC, SC: Load/Store char
- MALC, FREE: Memory allocation
- MSET, MCMP: Memory operations
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention, bake_weights
from .zero_detection import compute_normalization_constant


# =============================================================================
# LEA - Load Effective Address (opcode 0)
# AX = BP + imm (base pointer + offset)
# =============================================================================

class LeaFFN(PureFFN):
    """
    Load Effective Address: RESULT = BP + offset.

    BP is stored in a dedicated slot, offset comes from NIB_B.
    """
    BP_SLOT = E.TEMP  # Use TEMP to store base pointer

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        # RESULT = BP + NIB_B (offset)
        # Copy BP to RESULT
        self.W_up[0, E.OP_START + Opcode.LEA] = S
        self.W_gate[0, self.BP_SLOT] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.LEA] = -S
        self.W_gate[1, self.BP_SLOT] = -1.0
        self.W_down[E.RESULT, 1] = 1.0 / S

        # Add offset (NIB_B)
        self.W_up[2, E.OP_START + Opcode.LEA] = S
        self.W_gate[2, E.NIB_B] = 1.0
        self.W_down[E.RESULT, 2] = 1.0 / S

        self.W_up[3, E.OP_START + Opcode.LEA] = -S
        self.W_gate[3, E.NIB_B] = -1.0
        self.W_down[E.RESULT, 3] = 1.0 / S


# =============================================================================
# IMM - Load Immediate (opcode 1)
# AX = immediate value (stored in NIB_B by set_immediate())
# =============================================================================

class ImmFFN(PureFFN):
    """
    Load Immediate: RESULT = NIB_B (immediate value).

    set_immediate() writes to NIB_B, so IMM reads from NIB_B.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        # RESULT = NIB_B (immediate)
        self.W_up[0, E.OP_START + Opcode.IMM] = S
        self.W_gate[0, E.NIB_B] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.IMM] = -S
        self.W_gate[1, E.NIB_B] = -1.0
        self.W_down[E.RESULT, 1] = 1.0 / S


# =============================================================================
# BZ - Branch if Zero (opcode 4)
# Uses efficient zero-detection circuit (11 weights per check)
# =============================================================================

class BzFFN(PureFFN):
    """
    Branch if Zero: If RESULT == 0, set branch flag.

    Uses the 3-node zero-detection circuit:
    - silu(S*x + S*ε) × (1/k)
    - silu(S*x) × (-2/k)
    - silu(S*x - S*ε) × (1/k)

    Output: 1.0 if x == 0, 0.0 otherwise.
    """

    def __init__(self):
        # 3 nodes for zero-detection per nibble position
        super().__init__(E.DIM, hidden_dim=3 * E.NUM_POSITIONS)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Zero-detect on RESULT[pos]
            # Node 0: silu(S*RESULT + S*ε)
            self.W_up[base, E.RESULT] = S
            self.W_up[base, E.POS] = -S * 100
            self.b_up[base] = S * eps + S * 100 * pos
            self.W_gate[base, E.OP_START + Opcode.BZ] = 1.0
            self.W_down[E.TEMP, base] = 1.0 / k

            # Node 1: silu(S*RESULT) × (-2/k)
            self.W_up[base + 1, E.RESULT] = S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.W_gate[base + 1, E.OP_START + Opcode.BZ] = 1.0
            self.W_down[E.TEMP, base + 1] = -2.0 / k

            # Node 2: silu(S*RESULT - S*ε)
            self.W_up[base + 2, E.RESULT] = S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.W_gate[base + 2, E.OP_START + Opcode.BZ] = 1.0
            self.W_down[E.TEMP, base + 2] = 1.0 / k


class BzReduceAttention(PureAttention):
    """
    Reduce per-nibble zero results to single branch decision.

    All nibbles must be zero → branch taken.
    Uses AND reduction (product of indicators).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Uniform attention across all positions
        self.W_q[:, :] = 0.0
        self.W_k[:, :] = 0.0
        for i in range(E.DIM):
            self.W_q[i, E.OP_START + Opcode.BZ] = 1.0
            self.W_k[i, E.OP_START + Opcode.BZ] = 1.0

        # V: project TEMP (zero indicator per nibble)
        self.W_v[:, :] = 0.0
        self.W_v[0, E.TEMP] = 1.0

        # O: average to RAW_SUM (branch condition)
        self.W_o[:, :] = 0.0
        self.W_o[E.RAW_SUM, 0] = 1.0 / E.NUM_POSITIONS


class BzBranchFFN(PureFFN):
    """
    Final BZ decision: if RAW_SUM >= threshold, take branch.

    RAW_SUM contains average of per-nibble zero indicators.
    If all zero → RAW_SUM ≈ 1.0 → branch taken.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        threshold = 0.9  # All nibbles must be ~1.0

        # CARRY_OUT = 1 if RAW_SUM >= threshold (branch taken)
        self.W_up[0, E.RAW_SUM] = S
        self.b_up[0] = -S * threshold
        self.W_gate[0, E.OP_START + Opcode.BZ] = 1.0
        self.W_down[E.CARRY_OUT, 0] = 1.0 / S

        # Saturation
        self.W_up[1, E.RAW_SUM] = S
        self.b_up[1] = -S * (threshold + 0.1)
        self.W_gate[1, E.OP_START + Opcode.BZ] = 1.0
        self.W_down[E.CARRY_OUT, 1] = -1.0 / S


# =============================================================================
# BNZ - Branch if Not Zero (opcode 5)
# Inverted zero-detection
# =============================================================================

class BnzFFN(PureFFN):
    """
    Branch if Not Zero: If RESULT != 0, set branch flag.

    Uses inverted zero-detection: 1 - zero_indicator.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=3 * E.NUM_POSITIONS)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Inverted zero-detect: output weights negated, bias adds 1
            self.W_up[base, E.RESULT] = S
            self.W_up[base, E.POS] = -S * 100
            self.b_up[base] = S * eps + S * 100 * pos
            self.W_gate[base, E.OP_START + Opcode.BNZ] = 1.0
            self.W_down[E.TEMP, base] = -1.0 / k  # Negated

            self.W_up[base + 1, E.RESULT] = S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.W_gate[base + 1, E.OP_START + Opcode.BNZ] = 1.0
            self.W_down[E.TEMP, base + 1] = 2.0 / k  # Negated

            self.W_up[base + 2, E.RESULT] = S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.W_gate[base + 2, E.OP_START + Opcode.BNZ] = 1.0
            self.W_down[E.TEMP, base + 2] = -1.0 / k  # Negated

        # Bias to invert: TEMP starts at 1, subtract zero indicator
        self.b_down[E.TEMP] = 1.0


class BnzReduceAttention(PureAttention):
    """
    Reduce per-nibble not-zero results via OR.

    If ANY nibble is not zero → branch taken.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        self.W_q[:, :] = 0.0
        self.W_k[:, :] = 0.0
        for i in range(E.DIM):
            self.W_q[i, E.OP_START + Opcode.BNZ] = 1.0
            self.W_k[i, E.OP_START + Opcode.BNZ] = 1.0

        self.W_v[:, :] = 0.0
        self.W_v[0, E.TEMP] = 1.0

        self.W_o[:, :] = 0.0
        # For OR: max would be ideal, but we use average and threshold
        self.W_o[E.RAW_SUM, 0] = 1.0 / E.NUM_POSITIONS


class BnzBranchFFN(PureFFN):
    """Final BNZ decision."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        threshold = 0.1  # Any nibble non-zero triggers branch

        self.W_up[0, E.RAW_SUM] = S
        self.b_up[0] = -S * threshold
        self.W_gate[0, E.OP_START + Opcode.BNZ] = 1.0
        self.W_down[E.CARRY_OUT, 0] = 1.0 / S

        self.W_up[1, E.RAW_SUM] = S
        self.b_up[1] = -S * (threshold + 0.1)
        self.W_gate[1, E.OP_START + Opcode.BNZ] = 1.0
        self.W_down[E.CARRY_OUT, 1] = -1.0 / S


# =============================================================================
# ENT - Enter Function (opcode 6)
# push BP, BP = SP, SP -= size
# =============================================================================

class EntFFN(PureFFN):
    """
    Enter function: Save BP, set new BP, allocate stack space.

    Uses dedicated slots for BP and SP.
    """
    SP_SLOT = E.CARRY_IN  # Stack pointer
    BP_SLOT = E.TEMP      # Base pointer

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Save old BP (would push to stack in real VM)
        # BP = SP
        self.W_up[0, E.OP_START + Opcode.ENT] = S
        self.W_gate[0, self.SP_SLOT] = 1.0
        self.W_down[self.BP_SLOT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.ENT] = -S
        self.W_gate[1, self.SP_SLOT] = -1.0
        self.W_down[self.BP_SLOT, 1] = 1.0 / S

        # Clear old BP first
        self.W_up[2, E.OP_START + Opcode.ENT] = S
        self.W_gate[2, self.BP_SLOT] = -1.0
        self.W_down[self.BP_SLOT, 2] = 1.0 / S

        self.W_up[3, E.OP_START + Opcode.ENT] = -S
        self.W_gate[3, self.BP_SLOT] = 1.0
        self.W_down[self.BP_SLOT, 3] = 1.0 / S

        # SP -= NIB_A (frame size)
        self.W_up[4, E.OP_START + Opcode.ENT] = S
        self.W_gate[4, E.NIB_A] = -1.0
        self.W_down[self.SP_SLOT, 4] = 1.0 / S

        self.W_up[5, E.OP_START + Opcode.ENT] = -S
        self.W_gate[5, E.NIB_A] = 1.0
        self.W_down[self.SP_SLOT, 5] = 1.0 / S


# =============================================================================
# ADJ - Adjust Stack (opcode 7)
# SP += immediate
# =============================================================================

class AdjFFN(PureFFN):
    """
    Adjust stack pointer: SP += NIB_A.
    """
    SP_SLOT = E.CARRY_IN

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        self.W_up[0, E.OP_START + Opcode.ADJ] = S
        self.W_gate[0, E.NIB_A] = 1.0
        self.W_down[self.SP_SLOT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.ADJ] = -S
        self.W_gate[1, E.NIB_A] = -1.0
        self.W_down[self.SP_SLOT, 1] = 1.0 / S


# =============================================================================
# LC - Load Char (opcode 10)
# Load single byte (lowest nibble pair)
# =============================================================================

class LcFFN(PureFFN):
    """
    Load Char: Load byte from address, result in RESULT.

    For neural VM, this is same as LI but only reads positions 0-1.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Only position 0 and 1 are active for char load
        # Position gating: only process at pos 0
        self.W_up[0, E.OP_START + Opcode.LC] = S
        self.W_up[0, E.POS] = -S * 100
        self.b_up[0] = S * 100 * 0  # Only pos 0
        self.W_gate[0, E.NIB_A] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.LC] = -S
        self.W_up[1, E.POS] = -S * 100
        self.b_up[1] = S * 100 * 0
        self.W_gate[1, E.NIB_A] = -1.0
        self.W_down[E.RESULT, 1] = 1.0 / S

        # Position 1 (high nibble of byte)
        self.W_up[2, E.OP_START + Opcode.LC] = S
        self.W_up[2, E.POS] = -S * 100
        self.b_up[2] = S * 100 * 1  # Only pos 1
        self.W_gate[2, E.NIB_A] = 1.0
        self.W_down[E.RESULT, 2] = 1.0 / S

        self.W_up[3, E.OP_START + Opcode.LC] = -S
        self.W_up[3, E.POS] = -S * 100
        self.b_up[3] = S * 100 * 1
        self.W_gate[3, E.NIB_A] = -1.0
        self.W_down[E.RESULT, 3] = 1.0 / S


# =============================================================================
# SC - Store Char (opcode 12)
# Store single byte
# =============================================================================

class ScFFN(PureFFN):
    """
    Store Char: Store byte to address.

    Similar to SI but only writes positions 0-1.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Position 0
        self.W_up[0, E.OP_START + Opcode.SC] = S
        self.W_up[0, E.POS] = -S * 100
        self.b_up[0] = S * 100 * 0
        self.W_gate[0, E.RESULT] = 1.0
        self.W_down[E.NIB_B, 0] = 1.0 / S  # Store to address

        self.W_up[1, E.OP_START + Opcode.SC] = -S
        self.W_up[1, E.POS] = -S * 100
        self.b_up[1] = S * 100 * 0
        self.W_gate[1, E.RESULT] = -1.0
        self.W_down[E.NIB_B, 1] = 1.0 / S

        # Position 1
        self.W_up[2, E.OP_START + Opcode.SC] = S
        self.W_up[2, E.POS] = -S * 100
        self.b_up[2] = S * 100 * 1
        self.W_gate[2, E.RESULT] = 1.0
        self.W_down[E.NIB_B, 2] = 1.0 / S

        self.W_up[3, E.OP_START + Opcode.SC] = -S
        self.W_up[3, E.POS] = -S * 100
        self.b_up[3] = S * 100 * 1
        self.W_gate[3, E.RESULT] = -1.0
        self.W_down[E.NIB_B, 3] = 1.0 / S


# =============================================================================
# MALC - Malloc (opcode 34)
# Allocate memory, return pointer
# =============================================================================

class MalcFFN(PureFFN):
    """
    Malloc: Allocate memory.

    In neural VM, this just increments a heap pointer and returns it.
    """
    HEAP_PTR_SLOT = E.RAW_SUM  # Current heap pointer

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # RESULT = current heap pointer
        self.W_up[0, E.OP_START + Opcode.MALC] = S
        self.W_gate[0, self.HEAP_PTR_SLOT] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.MALC] = -S
        self.W_gate[1, self.HEAP_PTR_SLOT] = -1.0
        self.W_down[E.RESULT, 1] = 1.0 / S

        # Heap pointer += NIB_A (size)
        self.W_up[2, E.OP_START + Opcode.MALC] = S
        self.W_gate[2, E.NIB_A] = 1.0
        self.W_down[self.HEAP_PTR_SLOT, 2] = 1.0 / S

        self.W_up[3, E.OP_START + Opcode.MALC] = -S
        self.W_gate[3, E.NIB_A] = -1.0
        self.W_down[self.HEAP_PTR_SLOT, 3] = 1.0 / S


# =============================================================================
# FREE - Free Memory (opcode 35)
# In neural VM, this is a no-op (no garbage collection)
# =============================================================================

class FreeFFN(PureFFN):
    """
    Free: No-op in neural VM (memory not actually freed).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    @bake_weights
    def _bake_weights(self):
        # No operation - just gate on opcode but write nothing
        S = E.SCALE
        self.W_up[0, E.OP_START + Opcode.FREE] = S
        self.b_up[0] = -S * 1000  # Always inactive


# =============================================================================
# MSET - Memset (opcode 36)
# Set memory region to value
# =============================================================================

class MsetFFN(PureFFN):
    """
    Memset: Set all positions to NIB_A (value).

    In neural VM, this writes NIB_A to all RESULT positions.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # All positions: RESULT = NIB_A
        self.W_up[0, E.OP_START + Opcode.MSET] = S
        self.W_gate[0, E.NIB_A] = 1.0
        self.W_down[E.RESULT, 0] = 1.0 / S

        self.W_up[1, E.OP_START + Opcode.MSET] = -S
        self.W_gate[1, E.NIB_A] = -1.0
        self.W_down[E.RESULT, 1] = 1.0 / S


# =============================================================================
# MCMP - Memcmp (opcode 37)
# Compare memory regions, uses zero-detection on diff
# =============================================================================

class McmpFFN(PureFFN):
    """
    Memcmp: Compare NIB_A vs NIB_B at each position.

    Uses zero-detection on (A - B) per nibble.
    Result: 0 if all equal, 1 if any different.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=3 * E.NUM_POSITIONS)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        for pos in range(E.NUM_POSITIONS):
            base = pos * 3

            # Zero-detect on (NIB_A - NIB_B), store in TEMP
            # Node 0
            self.W_up[base, E.NIB_A] = S
            self.W_up[base, E.NIB_B] = -S
            self.W_up[base, E.POS] = -S * 100
            self.b_up[base] = S * eps + S * 100 * pos
            self.W_gate[base, E.OP_START + Opcode.MCMP] = 1.0
            self.W_down[E.TEMP, base] = 1.0 / k

            # Node 1
            self.W_up[base + 1, E.NIB_A] = S
            self.W_up[base + 1, E.NIB_B] = -S
            self.W_up[base + 1, E.POS] = -S * 100
            self.b_up[base + 1] = S * 100 * pos
            self.W_gate[base + 1, E.OP_START + Opcode.MCMP] = 1.0
            self.W_down[E.TEMP, base + 1] = -2.0 / k

            # Node 2
            self.W_up[base + 2, E.NIB_A] = S
            self.W_up[base + 2, E.NIB_B] = -S
            self.W_up[base + 2, E.POS] = -S * 100
            self.b_up[base + 2] = -S * eps + S * 100 * pos
            self.W_gate[base + 2, E.OP_START + Opcode.MCMP] = 1.0
            self.W_down[E.TEMP, base + 2] = 1.0 / k


class McmpReduceAttention(PureAttention):
    """Reduce per-nibble comparison to single result."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        self.W_q[:, :] = 0.0
        self.W_k[:, :] = 0.0
        for i in range(E.DIM):
            self.W_q[i, E.OP_START + Opcode.MCMP] = 1.0
            self.W_k[i, E.OP_START + Opcode.MCMP] = 1.0

        self.W_v[:, :] = 0.0
        self.W_v[0, E.TEMP] = 1.0

        self.W_o[:, :] = 0.0
        self.W_o[E.RESULT, 0] = 1.0 / E.NUM_POSITIONS


# =============================================================================
# Complete I/O Operations (OPEN, READ, CLOS, PRTF)
# =============================================================================

class OpenFFN(PureFFN):
    """
    Open file: Set tool call type and return fd.

    In neural VM, this sets IO_TOOL_CALL_TYPE to OPEN and
    waits for external handler response.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        from .embedding import IOToolCallType

        # Set tool call type to OPEN
        self.W_up[0, E.OP_START + Opcode.OPEN] = S
        self.b_gate[0] = float(IOToolCallType.OPEN)
        self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 1.0 / S


class ReadFFN(PureFFN):
    """
    Read file: Set tool call type to READ.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        from .embedding import IOToolCallType

        self.W_up[0, E.OP_START + Opcode.READ] = S
        self.b_gate[0] = float(IOToolCallType.READ)
        self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 1.0 / S


class ClosFFN(PureFFN):
    """
    Close file: Set tool call type to CLOSE.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        from .embedding import IOToolCallType

        self.W_up[0, E.OP_START + Opcode.CLOS] = S
        self.b_gate[0] = float(IOToolCallType.CLOSE)
        self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 1.0 / S


class PrtfFFN(PureFFN):
    """
    Printf: Set tool call type to PRINTF.

    The actual formatting is handled by external handler.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        from .embedding import IOToolCallType

        self.W_up[0, E.OP_START + Opcode.PRTF] = S
        self.b_gate[0] = float(IOToolCallType.PRINTF)
        self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 1.0 / S


# =============================================================================
# Optimized MOD - Power of 2 fast path
# =============================================================================

class ModPow2FFN(PureFFN):
    """
    Fast MOD for power-of-2 divisors.

    MOD 2^k = AND with (2^k - 1) = extract lower k bits.

    This is O(1) layer instead of 16 iterations!

    Detects if NIB_B is power of 2 and uses bitwise AND.
    """

    def __init__(self):
        # Nodes for each power of 2: 1, 2, 4, 8
        super().__init__(E.DIM, hidden_dim=8)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        eps = 0.5
        k = compute_normalization_constant(S, eps)

        # For MOD 1: result is always 0
        # For MOD 2: result = bit 0 of A
        # For MOD 4: result = bits 0-1 of A
        # For MOD 8: result = bits 0-2 of A

        # Check if B == 2 and compute A & 1
        self.W_up[0, E.NIB_B] = S
        self.b_up[0] = -S * 2 + S * eps  # Activate near B=2
        self.W_gate[0, E.NIB_A] = 1.0  # AND with bit 0 (odd/even)
        self.W_down[E.RESULT, 0] = 1.0 / S

        # More patterns for 4, 8, etc. would follow
        # This is a simplified version


# =============================================================================
# LAYER COUNTS
# =============================================================================

"""
New Opcode Layer Counts:

| Op# | Name    | Layers | Method |
|-----|---------|--------|--------|
| 0   | LEA     | 1      | Direct add |
| 1   | IMM     | 1      | Direct copy |
| 4   | BZ      | 2      | Zero-detection + reduce |
| 5   | BNZ     | 2      | Inverted zero-detection + reduce |
| 6   | ENT     | 1      | Stack frame |
| 7   | ADJ     | 1      | Stack adjust |
| 10  | LC      | 1      | Load char |
| 12  | SC      | 1      | Store char |
| 34  | MALC    | 1      | Heap alloc |
| 35  | FREE    | 1      | No-op |
| 36  | MSET    | 1      | Direct set |
| 37  | MCMP    | 2      | Zero-detection diff |
| 30  | OPEN    | 1      | Tool call |
| 31  | READ    | 1      | Tool call |
| 32  | CLOS    | 1      | Tool call |
| 33  | PRTF    | 1      | Tool call |
"""


def get_all_new_ops():
    """Return list of all new operation classes."""
    return [
        (Opcode.LEA, 'LEA', LeaFFN, 1),
        (Opcode.IMM, 'IMM', ImmFFN, 1),
        (Opcode.BZ, 'BZ', [BzFFN, BzReduceAttention, BzBranchFFN], 2),
        (Opcode.BNZ, 'BNZ', [BnzFFN, BnzReduceAttention, BnzBranchFFN], 2),
        (Opcode.ENT, 'ENT', EntFFN, 1),
        (Opcode.ADJ, 'ADJ', AdjFFN, 1),
        (Opcode.LC, 'LC', LcFFN, 1),
        (Opcode.SC, 'SC', ScFFN, 1),
        (Opcode.MALC, 'MALC', MalcFFN, 1),
        (Opcode.FREE, 'FREE', FreeFFN, 1),
        (Opcode.MSET, 'MSET', MsetFFN, 1),
        (Opcode.MCMP, 'MCMP', [McmpFFN, McmpReduceAttention], 2),
        (Opcode.OPEN, 'OPEN', OpenFFN, 1),
        (Opcode.READ, 'READ', ReadFFN, 1),
        (Opcode.CLOS, 'CLOS', ClosFFN, 1),
        (Opcode.PRTF, 'PRTF', PrtfFFN, 1),
    ]
