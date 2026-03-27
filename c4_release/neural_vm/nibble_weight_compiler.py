"""
Nibble-Based Weight Compiler: Integration with AutoregressiveVM

Compiles computation graphs to FFN weights that operate on the nibble-based
embedding format used by the autoregressive VM. Unlike the abstract graph
weight compiler, this works with the specific E.DIM=160 × E.NUM_POSITIONS=8
structure and nibble slot semantics.

Architecture:
    Virtual Registers → Nibble Slots (E.NIB_A, E.NIB_B, E.RESULT, etc.)
    Computation Graph → FFN Weights for nibble operations
    OpType Operations → Per-nibble FFN units

Integration with AutoregressiveVM:
    - Generates weights for layers 9-12 (ALU)
    - Operates on nibble representation (8 positions × 160 dims)
    - Compatible with chunk-based ALU architecture
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass

from .embedding import E, Opcode
from .graph_weight_compiler import OpType, ComputationGraph, IRNode


# =============================================================================
# Nibble Slot Allocation
# =============================================================================

@dataclass
class NibbleRegisterMap:
    """Maps virtual registers to nibble embedding slots."""

    # Standard VM registers (32-bit values as 8 nibbles)
    AX_BASE = 0  # Position offset for AX nibbles (relative to position 0-7)
    BP_BASE = 0  # BP shares same nibble positions during computation
    SP_BASE = 0  # SP shares same nibble positions during computation

    # Computation slots (per-nibble features from E class)
    NIB_A = E.NIB_A        # Operand A nibble
    NIB_B = E.NIB_B        # Operand B nibble
    RAW_SUM = E.RAW_SUM    # Raw sum/diff before carry
    CARRY_IN = E.CARRY_IN  # Carry/borrow from lower nibble
    CARRY_OUT = E.CARRY_OUT # Carry/borrow to higher nibble
    RESULT = E.RESULT      # Result nibble
    TEMP = E.TEMP          # Temporary storage

    # Opcode encoding (one-hot, shared across positions)
    OP_START = E.OP_START  # Start of opcode one-hot (72 opcodes)

    # Dimensions
    DIM = E.DIM            # 160 dims per position
    NUM_POSITIONS = E.NUM_POSITIONS  # 8 nibble positions
    SCALE = E.SCALE        # 100.0 for SwiGLU identity

    def flat_index(self, position: int, slot: int) -> int:
        """Convert (position, slot) to flattened dimension index.

        Args:
            position: Nibble position (0-7)
            slot: Feature slot (0-159)

        Returns:
            Flattened index in range [0, 1280)
        """
        return position * self.DIM + slot

    def opcode_index(self, opcode: int) -> int:
        """Get dimension index for opcode one-hot.

        Opcodes are shared across all positions (not per-position).
        """
        return self.OP_START + opcode


# =============================================================================
# Nibble Weight Emitter
# =============================================================================

class NibbleWeightEmitter:
    """Emits FFN weights for nibble-based operations.

    Unlike the abstract WeightEmitter which works on arbitrary dimensions,
    this emits weights that operate on the specific nibble structure:
    - 8 nibble positions (4-bit chunks of 32-bit value)
    - 160 dimensions per position
    - Specific slots: NIB_A, NIB_B, RESULT, CARRY_IN, etc.
    - Opcode one-hot for operation selection
    """

    def __init__(self, opcode: int, num_positions: int = 8, unit_offset: int = 0):
        """Initialize nibble weight emitter.

        Args:
            opcode: C4 opcode for operation gating (from Opcode class)
            num_positions: Number of nibble positions (default 8 for 32-bit)
            unit_offset: Hidden unit offset for this opcode (for non-overlapping allocation)
        """
        self.opcode = opcode
        self.num_positions = num_positions
        self.unit_offset_base = unit_offset  # Base offset for this opcode
        self.reg_map = NibbleRegisterMap()

        # Flattened dimension: num_positions × DIM
        self.dim = num_positions * self.reg_map.DIM  # 8 × 160 = 1280
        self.hidden_dim = 4096  # Match AutoregressiveVM default

        # Weight matrices (same structure as PureFFN)
        self.W_up = torch.zeros(self.hidden_dim, self.dim)
        self.b_up = torch.zeros(self.hidden_dim)
        self.W_gate = torch.zeros(self.hidden_dim, self.dim)
        self.b_gate = torch.zeros(self.hidden_dim)
        self.W_down = torch.zeros(self.dim, self.hidden_dim)
        self.b_down = torch.zeros(self.dim)

        self.unit_offset = self.unit_offset_base  # Current hidden unit index (starts at base)

    def _fi(self, position: int, slot: int) -> int:
        """Flat index helper."""
        return self.reg_map.flat_index(position, slot)

    def _opcode_gate(self, unit: int):
        """Apply opcode gating to a hidden unit."""
        op_idx = self.reg_map.opcode_index(self.opcode)
        # For simplicity, we'll use position 0's opcode slot
        # (opcodes are replicated across positions in real VM)
        self.W_gate[unit, op_idx] = 1.0

    def emit_add_nibble(self, position: int):
        """Emit weights for ADD at a specific nibble position.

        Computes: RESULT[pos] = (NIB_A[pos] + NIB_B[pos] + CARRY_IN[pos]) mod 16
        Also generates CARRY_OUT[pos] for next position.

        This is simplified version - full version needs 3-layer pipeline
        with carry lookahead (see alu/ops/add.py).
        """
        S = self.reg_map.SCALE
        base = 16  # Nibble base

        # Input slots
        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        cin_slot = self._fi(position, self.reg_map.CARRY_IN)

        # Output slots
        result_slot = self._fi(position, self.reg_map.RESULT)
        cout_slot = self._fi(position, self.reg_map.CARRY_OUT)

        # Unit 0-1: RAW_SUM = A + B (cancel pair pattern)
        self.W_up[self.unit_offset, a_slot] = S
        self.W_up[self.unit_offset, b_slot] = S
        self._opcode_gate(self.unit_offset)
        self.W_down[result_slot, self.unit_offset] = 1.0 / S

        self.W_up[self.unit_offset + 1, a_slot] = -S
        self.W_up[self.unit_offset + 1, b_slot] = -S
        self._opcode_gate(self.unit_offset + 1)
        self.W_down[result_slot, self.unit_offset + 1] = 1.0 / S

        # Unit 2-3: Add CARRY_IN to RESULT
        self.W_up[self.unit_offset + 2, cin_slot] = S
        self._opcode_gate(self.unit_offset + 2)
        self.W_down[result_slot, self.unit_offset + 2] = 1.0 / S

        self.W_up[self.unit_offset + 3, cin_slot] = -S
        self._opcode_gate(self.unit_offset + 3)
        self.W_down[result_slot, self.unit_offset + 3] = 1.0 / S

        # Unit 4-5: Generate CARRY_OUT = step(A + B + CIN >= 16)
        self.W_up[self.unit_offset + 4, a_slot] = S
        self.W_up[self.unit_offset + 4, b_slot] = S
        self.W_up[self.unit_offset + 4, cin_slot] = S
        self.b_up[self.unit_offset + 4] = -S * (base - 1.0)
        self._opcode_gate(self.unit_offset + 4)
        self.W_down[cout_slot, self.unit_offset + 4] = 1.0 / S

        self.W_up[self.unit_offset + 5, a_slot] = S
        self.W_up[self.unit_offset + 5, b_slot] = S
        self.W_up[self.unit_offset + 5, cin_slot] = S
        self.b_up[self.unit_offset + 5] = -S * float(base)
        self._opcode_gate(self.unit_offset + 5)
        self.W_down[cout_slot, self.unit_offset + 5] = -1.0 / S

        # Unit 6-7: Modulo reduction (subtract 16 if overflow)
        self.W_up[self.unit_offset + 6, result_slot] = S
        self.b_up[self.unit_offset + 6] = -S * (base - 1.0)
        self._opcode_gate(self.unit_offset + 6)
        self.W_down[result_slot, self.unit_offset + 6] = -float(base) / S

        self.W_up[self.unit_offset + 7, result_slot] = S
        self.b_up[self.unit_offset + 7] = -S * float(base)
        self._opcode_gate(self.unit_offset + 7)
        self.W_down[result_slot, self.unit_offset + 7] = float(base) / S

        self.unit_offset += 8

    def emit_sub_nibble(self, position: int):
        """Emit weights for SUB at a specific nibble position.

        Computes: RESULT[pos] = (NIB_A[pos] - NIB_B[pos] - BORROW_IN[pos]) mod 16
        (Borrow is represented as negative carry)
        """
        S = self.reg_map.SCALE
        base = 16

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        borrow_slot = self._fi(position, self.reg_map.CARRY_IN)  # Reuse carry for borrow
        result_slot = self._fi(position, self.reg_map.RESULT)
        bout_slot = self._fi(position, self.reg_map.CARRY_OUT)

        # Unit 0-1: RESULT = A - B (cancel pair)
        self.W_up[self.unit_offset, a_slot] = S
        self.W_up[self.unit_offset, b_slot] = -S
        self._opcode_gate(self.unit_offset)
        self.W_down[result_slot, self.unit_offset] = 1.0 / S

        self.W_up[self.unit_offset + 1, a_slot] = -S
        self.W_up[self.unit_offset + 1, b_slot] = S
        self._opcode_gate(self.unit_offset + 1)
        self.W_down[result_slot, self.unit_offset + 1] = 1.0 / S

        # Unit 2-3: Subtract BORROW_IN from RESULT
        self.W_up[self.unit_offset + 2, borrow_slot] = -S
        self._opcode_gate(self.unit_offset + 2)
        self.W_down[result_slot, self.unit_offset + 2] = 1.0 / S

        self.W_up[self.unit_offset + 3, borrow_slot] = S
        self._opcode_gate(self.unit_offset + 3)
        self.W_down[result_slot, self.unit_offset + 3] = 1.0 / S

        # Unit 4-5: Generate BORROW_OUT = step(A - B - BIN < 0)
        self.W_up[self.unit_offset + 4, a_slot] = S
        self.W_up[self.unit_offset + 4, b_slot] = -S
        self.W_up[self.unit_offset + 4, borrow_slot] = -S
        self.b_up[self.unit_offset + 4] = S  # Threshold at 0
        self._opcode_gate(self.unit_offset + 4)
        self.W_down[bout_slot, self.unit_offset + 4] = -1.0 / S  # Negative for borrow

        self.W_up[self.unit_offset + 5, a_slot] = S
        self.W_up[self.unit_offset + 5, b_slot] = -S
        self.W_up[self.unit_offset + 5, borrow_slot] = -S
        self.b_up[self.unit_offset + 5] = 0.0
        self._opcode_gate(self.unit_offset + 5)
        self.W_down[bout_slot, self.unit_offset + 5] = 1.0 / S

        # Unit 6-7: Modulo wrap (add 16 if negative)
        self.W_up[self.unit_offset + 6, result_slot] = S
        self.b_up[self.unit_offset + 6] = S  # Threshold at 0
        self._opcode_gate(self.unit_offset + 6)
        self.W_down[result_slot, self.unit_offset + 6] = float(base) / S

        self.W_up[self.unit_offset + 7, result_slot] = S
        self.b_up[self.unit_offset + 7] = 0.0
        self._opcode_gate(self.unit_offset + 7)
        self.W_down[result_slot, self.unit_offset + 7] = -float(base) / S

        self.unit_offset += 8

    def emit_cmp_eq_nibble(self, position: int):
        """Emit weights for equality comparison at nibble position.

        Computes: TEMP[pos] = (NIB_A[pos] == NIB_B[pos]) ? 1 : 0

        This is per-nibble; full equality needs AND across all positions.
        """
        S = self.reg_map.SCALE

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        temp_slot = self._fi(position, self.reg_map.TEMP)

        # Unit 0-2: step(>=0) - 2*step(>=1) + step(>=2) = 1 when diff==0, else other
        # For diff = a - b:
        #   diff == 0: step(0>=0) - 2*step(0>=1) + step(0>=2) = 1 - 0 + 0 = 1
        #   diff != 0: different pattern

        # Unit 0: step(a - b >= 0)
        self.W_up[self.unit_offset, a_slot] = S
        self.W_up[self.unit_offset, b_slot] = -S
        self.b_up[self.unit_offset] = S
        self._opcode_gate(self.unit_offset)
        self.W_down[temp_slot, self.unit_offset] = 1.0 / S

        # Unit 1: -2*step(a - b >= 1)
        self.W_up[self.unit_offset + 1, a_slot] = S
        self.W_up[self.unit_offset + 1, b_slot] = -S
        self.b_up[self.unit_offset + 1] = 0.0
        self._opcode_gate(self.unit_offset + 1)
        self.W_down[temp_slot, self.unit_offset + 1] = -2.0 / S

        # Unit 2: step(a - b >= 2)
        self.W_up[self.unit_offset + 2, a_slot] = S
        self.W_up[self.unit_offset + 2, b_slot] = -S
        self.b_up[self.unit_offset + 2] = -S
        self._opcode_gate(self.unit_offset + 2)
        self.W_down[temp_slot, self.unit_offset + 2] = 1.0 / S

        self.unit_offset += 3

    def emit_cmp_ne_nibble(self, position: int):
        """Emit weights for not-equal comparison at nibble position.

        Computes: TEMP[pos] = (NIB_A[pos] != NIB_B[pos]) ? 1 : 0
        """
        S = self.reg_map.SCALE

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        temp_slot = self._fi(position, self.reg_map.TEMP)

        # Not-equal is opposite of equal: 1 - eq_result
        # Unit 0: Write constant 1
        self._opcode_gate(self.unit_offset)
        self.b_gate[self.unit_offset] = S
        self.W_down[temp_slot, self.unit_offset] = 1.0 / S

        # Units 1-3: Subtract equality check (same as emit_cmp_eq)
        self.W_up[self.unit_offset + 1, a_slot] = S
        self.W_up[self.unit_offset + 1, b_slot] = -S
        self.b_up[self.unit_offset + 1] = S
        self._opcode_gate(self.unit_offset + 1)
        self.W_down[temp_slot, self.unit_offset + 1] = -1.0 / S

        self.W_up[self.unit_offset + 2, a_slot] = S
        self.W_up[self.unit_offset + 2, b_slot] = -S
        self.b_up[self.unit_offset + 2] = 0.0
        self._opcode_gate(self.unit_offset + 2)
        self.W_down[temp_slot, self.unit_offset + 2] = 2.0 / S

        self.W_up[self.unit_offset + 3, a_slot] = S
        self.W_up[self.unit_offset + 3, b_slot] = -S
        self.b_up[self.unit_offset + 3] = -S
        self._opcode_gate(self.unit_offset + 3)
        self.W_down[temp_slot, self.unit_offset + 3] = -1.0 / S

        self.unit_offset += 4

    def emit_cmp_lt_nibble(self, position: int):
        """Emit weights for less-than comparison at nibble position.

        Computes: TEMP[pos] = (NIB_A[pos] < NIB_B[pos]) ? 1 : 0
        """
        S = self.reg_map.SCALE

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        temp_slot = self._fi(position, self.reg_map.TEMP)

        # a < b is equivalent to step(b - a - 1 >= 0)
        self.W_up[self.unit_offset, b_slot] = S
        self.W_up[self.unit_offset, a_slot] = -S
        self.b_up[self.unit_offset] = 0.0  # threshold at 1
        self._opcode_gate(self.unit_offset)
        self.W_down[temp_slot, self.unit_offset] = 1.0 / S

        self.unit_offset += 1

    def emit_cmp_gt_nibble(self, position: int):
        """Emit weights for greater-than comparison at nibble position.

        Computes: TEMP[pos] = (NIB_A[pos] > NIB_B[pos]) ? 1 : 0
        """
        S = self.reg_map.SCALE

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        temp_slot = self._fi(position, self.reg_map.TEMP)

        # a > b is equivalent to step(a - b - 1 >= 0)
        self.W_up[self.unit_offset, a_slot] = S
        self.W_up[self.unit_offset, b_slot] = -S
        self.b_up[self.unit_offset] = 0.0  # threshold at 1
        self._opcode_gate(self.unit_offset)
        self.W_down[temp_slot, self.unit_offset] = 1.0 / S

        self.unit_offset += 1

    def emit_cmp_le_nibble(self, position: int):
        """Emit weights for less-or-equal comparison at nibble position.

        Computes: TEMP[pos] = (NIB_A[pos] <= NIB_B[pos]) ? 1 : 0
        """
        S = self.reg_map.SCALE

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        temp_slot = self._fi(position, self.reg_map.TEMP)

        # a <= b is equivalent to NOT(a > b) = 1 - step(a - b - 1 >= 0)
        # Unit 0: Write constant 1
        self._opcode_gate(self.unit_offset)
        self.b_gate[self.unit_offset] = S
        self.W_down[temp_slot, self.unit_offset] = 1.0 / S

        # Unit 1: Subtract step(a - b - 1 >= 0)
        self.W_up[self.unit_offset + 1, a_slot] = S
        self.W_up[self.unit_offset + 1, b_slot] = -S
        self.b_up[self.unit_offset + 1] = 0.0
        self._opcode_gate(self.unit_offset + 1)
        self.W_down[temp_slot, self.unit_offset + 1] = -1.0 / S

        self.unit_offset += 2

    def emit_cmp_ge_nibble(self, position: int):
        """Emit weights for greater-or-equal comparison at nibble position.

        Computes: TEMP[pos] = (NIB_A[pos] >= NIB_B[pos]) ? 1 : 0
        """
        S = self.reg_map.SCALE

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        temp_slot = self._fi(position, self.reg_map.TEMP)

        # a >= b is equivalent to step(a - b >= 0)
        self.W_up[self.unit_offset, a_slot] = S
        self.W_up[self.unit_offset, b_slot] = -S
        self.b_up[self.unit_offset] = S
        self._opcode_gate(self.unit_offset)
        self.W_down[temp_slot, self.unit_offset] = 1.0 / S

        self.unit_offset += 1

    def emit_move_nibble(self, position: int):
        """Emit weights for MOVE at nibble position.

        Computes: RESULT[pos] = NIB_A[pos]
        """
        S = self.reg_map.SCALE

        a_slot = self._fi(position, self.reg_map.NIB_A)
        result_slot = self._fi(position, self.reg_map.RESULT)

        # Cancel pair: copy A to RESULT
        self.W_up[self.unit_offset, a_slot] = S
        self._opcode_gate(self.unit_offset)
        self.W_down[result_slot, self.unit_offset] = 1.0 / S

        self.W_up[self.unit_offset + 1, a_slot] = -S
        self._opcode_gate(self.unit_offset + 1)
        self.W_down[result_slot, self.unit_offset + 1] = 1.0 / S

        self.unit_offset += 2

    def emit_flag_set(self, flag_dim: int):
        """Emit weights to set a flag dimension to 1.0.

        Args:
            flag_dim: Dimension index of flag to set

        This is a simple constant write operation.
        """
        S = self.reg_map.SCALE

        # Unit 0-1: Write 1.0 to flag dimension (cancel pair pattern)
        self._opcode_gate(self.unit_offset)
        self.b_gate[self.unit_offset] = S  # Constant S
        self.W_down[flag_dim, self.unit_offset] = 1.0 / S  # Result = S/S = 1.0

        self._opcode_gate(self.unit_offset + 1)
        self.b_gate[self.unit_offset + 1] = -S
        self.W_down[flag_dim, self.unit_offset + 1] = 1.0 / S

        self.unit_offset += 2

    def emit_flag_clear(self, flag_dim: int):
        """Emit weights to clear a flag dimension to 0.0.

        Args:
            flag_dim: Dimension index of flag to clear

        Uses subtraction: flag = flag - flag = 0
        """
        S = self.reg_map.SCALE

        # Unit 0-1: Read current flag value and subtract it (result = 0)
        self.W_up[self.unit_offset, flag_dim] = S
        self._opcode_gate(self.unit_offset)
        self.W_down[flag_dim, self.unit_offset] = -1.0 / S

        self.W_up[self.unit_offset + 1, flag_dim] = -S
        self._opcode_gate(self.unit_offset + 1)
        self.W_down[flag_dim, self.unit_offset + 1] = -1.0 / S

        self.unit_offset += 2

    def emit_mem_read_request(self, src_nibbles: list, mem_addr_start: int, mem_read_flag: int):
        """Emit weights for memory read request.

        Copies source nibbles to MEM_ADDR mailbox and sets MEM_READ flag.

        Args:
            src_nibbles: List of (position, slot) tuples for source address nibbles
            mem_addr_start: Starting dimension of MEM_ADDR mailbox (E.MEM_ADDR_BASE)
            mem_read_flag: Dimension index of MEM_READ flag (E.MEM_READ)
        """
        S = self.reg_map.SCALE

        # Copy each nibble to memory address mailbox
        for i, (src_pos, src_slot) in enumerate(src_nibbles):
            src_idx = self._fi(src_pos, src_slot)
            dest_idx = mem_addr_start + i

            # Cancel pair: copy source nibble to MEM_ADDR[i]
            self.W_up[self.unit_offset, src_idx] = S
            self._opcode_gate(self.unit_offset)
            self.W_down[dest_idx, self.unit_offset] = 1.0 / S

            self.W_up[self.unit_offset + 1, src_idx] = -S
            self._opcode_gate(self.unit_offset + 1)
            self.W_down[dest_idx, self.unit_offset + 1] = 1.0 / S

            self.unit_offset += 2

        # Set MEM_READ flag = 1.0
        self.emit_flag_set(mem_read_flag)

    def emit_mem_write_request(self, addr_nibbles: list, data_nibbles: list,
                               mem_addr_start: int, mem_data_start: int, mem_write_flag: int):
        """Emit weights for memory write request.

        Copies address and data nibbles to mailbox and sets MEM_WRITE flag.

        Args:
            addr_nibbles: List of (position, slot) for address
            data_nibbles: List of (position, slot) for data
            mem_addr_start: MEM_ADDR mailbox start (E.MEM_ADDR_BASE)
            mem_data_start: MEM_DATA mailbox start (E.MEM_DATA_BASE)
            mem_write_flag: MEM_WRITE flag dimension (E.MEM_WRITE)
        """
        S = self.reg_map.SCALE

        # Copy address nibbles
        for i, (src_pos, src_slot) in enumerate(addr_nibbles):
            src_idx = self._fi(src_pos, src_slot)
            dest_idx = mem_addr_start + i

            self.W_up[self.unit_offset, src_idx] = S
            self._opcode_gate(self.unit_offset)
            self.W_down[dest_idx, self.unit_offset] = 1.0 / S

            self.W_up[self.unit_offset + 1, src_idx] = -S
            self._opcode_gate(self.unit_offset + 1)
            self.W_down[dest_idx, self.unit_offset + 1] = 1.0 / S

            self.unit_offset += 2

        # Copy data nibbles
        for i, (src_pos, src_slot) in enumerate(data_nibbles):
            src_idx = self._fi(src_pos, src_slot)
            dest_idx = mem_data_start + i

            self.W_up[self.unit_offset, src_idx] = S
            self._opcode_gate(self.unit_offset)
            self.W_down[dest_idx, self.unit_offset] = 1.0 / S

            self.W_up[self.unit_offset + 1, src_idx] = -S
            self._opcode_gate(self.unit_offset + 1)
            self.W_down[dest_idx, self.unit_offset + 1] = 1.0 / S

            self.unit_offset += 2

        # Set MEM_WRITE flag = 1.0
        self.emit_flag_set(mem_write_flag)

    def emit_pc_conditional(self, cond_slot: int, target_nibbles: list, pc_nibbles: list):
        """Emit weights for conditional PC update.

        Computes: PC = cond ? target : (PC + instruction_width)

        Args:
            cond_slot: Dimension index of condition flag (0 or 1)
            target_nibbles: List of (position, slot) for target address
            pc_nibbles: List of (position, slot) for current PC
        """
        # This is just a SELECT operation applied to PC nibbles
        # For each nibble position: PC[i] = cond ? target[i] : fallthrough[i]

        S = self.reg_map.SCALE

        for i in range(len(target_nibbles)):
            tgt_pos, tgt_slot = target_nibbles[i]
            pc_pos, pc_slot = pc_nibbles[i]

            tgt_idx = self._fi(tgt_pos, tgt_slot)
            pc_idx = self._fi(pc_pos, pc_slot)
            result_idx = pc_idx  # Write result back to PC

            # SELECT implementation: result = cond * target + (1 - cond) * fallthrough
            # Unit 0-1: cond * target
            self.W_up[self.unit_offset, cond_slot] = S
            self.W_gate[self.unit_offset, tgt_idx] = 1.0
            self.W_down[result_idx, self.unit_offset] = 1.0 / S

            self.W_up[self.unit_offset + 1, cond_slot] = -S
            self.W_gate[self.unit_offset + 1, tgt_idx] = -1.0
            self.W_down[result_idx, self.unit_offset + 1] = 1.0 / S

            # Unit 2-3: (1 - cond) * fallthrough
            self.b_up[self.unit_offset + 2] = S  # Constant 1
            self.W_up[self.unit_offset + 2, cond_slot] = -S  # Subtract cond
            self.W_gate[self.unit_offset + 2, pc_idx] = 1.0
            self.W_down[result_idx, self.unit_offset + 2] = 1.0 / S

            self.b_up[self.unit_offset + 3] = -S
            self.W_up[self.unit_offset + 3, cond_slot] = S
            self.W_gate[self.unit_offset + 3, pc_idx] = -1.0
            self.W_down[result_idx, self.unit_offset + 3] = 1.0 / S

            self.unit_offset += 4

    def emit_bitwise_op_nibble(self, position: int, op_type: str):
        """Emit weights for bitwise operation at nibble position.

        Uses lookup table: one hidden unit per (a, b) pair.
        For base=16 (nibbles), need 16×16 = 256 units per position.

        Args:
            position: Nibble position (0-7)
            op_type: "or", "xor", or "and"
        """
        S = self.reg_map.SCALE
        base = 16

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        result_slot = self._fi(position, self.reg_map.RESULT)

        # Lookup table: 256 units (16×16 combinations)
        for a in range(base):
            for b in range(base):
                # Compute result based on operation
                if op_type == "or":
                    result = a | b
                elif op_type == "xor":
                    result = a ^ b
                elif op_type == "and":
                    result = a & b
                else:
                    raise ValueError(f"Unknown bitwise op: {op_type}")

                # Unit detects (a, b) pair and outputs result
                # Pattern: step(A >= a) AND step(A < a+1) AND step(B >= b) AND step(B < b+1)
                # Simplified: Use exact match via threshold

                # W_up: detect exact values
                self.W_up[self.unit_offset, a_slot] = S
                self.W_up[self.unit_offset, b_slot] = S
                self.b_up[self.unit_offset] = -S * (a + b - 0.5)  # Threshold

                # W_gate: opcode gating
                self._opcode_gate(self.unit_offset)

                # W_down: output result
                self.W_down[result_slot, self.unit_offset] = result / S

                self.unit_offset += 1

    def emit_mul_nibble(self, position: int):
        """Emit weights for multiplication at nibble position.

        MUL needs carry propagation across nibbles, but for single nibble:
        result = (a * b) mod 16, carry_out = (a * b) // 16
        """
        S = self.reg_map.SCALE
        base = 16

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        result_slot = self._fi(position, self.reg_map.RESULT)
        carry_slot = self._fi(position, self.reg_map.CARRY_OUT)

        # Lookup table: 256 units
        for a in range(base):
            for b in range(base):
                product = a * b
                result = product % base
                carry = product // base

                # Detect (a, b) pair
                self.W_up[self.unit_offset, a_slot] = S
                self.W_up[self.unit_offset, b_slot] = S
                self.b_up[self.unit_offset] = -S * (a + b - 0.5)

                self._opcode_gate(self.unit_offset)

                # Output result and carry
                self.W_down[result_slot, self.unit_offset] = result / S
                self.W_down[carry_slot, self.unit_offset] = carry / S

                self.unit_offset += 1

    def emit_div_nibble(self, position: int):
        """Emit weights for division at nibble position.

        For single nibble: result = a // b, carry_out = a % b (remainder)
        Division by zero returns 0.
        """
        S = self.reg_map.SCALE
        base = 16

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        result_slot = self._fi(position, self.reg_map.RESULT)
        carry_slot = self._fi(position, self.reg_map.CARRY_OUT)

        # Lookup table: 256 units
        for a in range(base):
            for b in range(base):
                if b == 0:
                    quotient = 0
                    remainder = a
                else:
                    quotient = a // b
                    remainder = a % b

                # Detect (a, b) pair
                self.W_up[self.unit_offset, a_slot] = S
                self.W_up[self.unit_offset, b_slot] = S
                self.b_up[self.unit_offset] = -S * (a + b - 0.5)

                self._opcode_gate(self.unit_offset)

                # Output quotient and remainder
                self.W_down[result_slot, self.unit_offset] = quotient / S
                self.W_down[carry_slot, self.unit_offset] = remainder / S

                self.unit_offset += 1

    def emit_mod_nibble(self, position: int):
        """Emit weights for modulo at nibble position.

        For single nibble: result = a % b
        Modulo by zero returns 0.
        """
        S = self.reg_map.SCALE
        base = 16

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        result_slot = self._fi(position, self.reg_map.RESULT)

        # Lookup table: 256 units
        for a in range(base):
            for b in range(base):
                if b == 0:
                    result = 0
                else:
                    result = a % b

                # Detect (a, b) pair
                self.W_up[self.unit_offset, a_slot] = S
                self.W_up[self.unit_offset, b_slot] = S
                self.b_up[self.unit_offset] = -S * (a + b - 0.5)

                self._opcode_gate(self.unit_offset)

                # Output result
                self.W_down[result_slot, self.unit_offset] = result / S

                self.unit_offset += 1

    def emit_shl_nibble(self, position: int):
        """Emit weights for shift left at nibble position.

        For single nibble: result = (a << (b % 4)) % 16, carry = (a << (b % 4)) // 16
        Shift amount limited to 0-3 for nibble size.
        """
        S = self.reg_map.SCALE
        base = 16

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        result_slot = self._fi(position, self.reg_map.RESULT)
        carry_slot = self._fi(position, self.reg_map.CARRY_OUT)

        # Lookup table: 256 units
        for a in range(base):
            for b in range(base):
                shift_amt = b % 4  # Limit to 0-3
                shifted = a << shift_amt
                result = shifted % base
                carry = shifted // base

                # Detect (a, b) pair
                self.W_up[self.unit_offset, a_slot] = S
                self.W_up[self.unit_offset, b_slot] = S
                self.b_up[self.unit_offset] = -S * (a + b - 0.5)

                self._opcode_gate(self.unit_offset)

                # Output result and carry
                self.W_down[result_slot, self.unit_offset] = result / S
                self.W_down[carry_slot, self.unit_offset] = carry / S

                self.unit_offset += 1

    def emit_shr_nibble(self, position: int):
        """Emit weights for shift right at nibble position.

        For single nibble: result = a >> (b % 4)
        Shift amount limited to 0-3 for nibble size.
        """
        S = self.reg_map.SCALE
        base = 16

        a_slot = self._fi(position, self.reg_map.NIB_A)
        b_slot = self._fi(position, self.reg_map.NIB_B)
        result_slot = self._fi(position, self.reg_map.RESULT)

        # Lookup table: 256 units
        for a in range(base):
            for b in range(base):
                shift_amt = b % 4  # Limit to 0-3
                result = a >> shift_amt

                # Detect (a, b) pair
                self.W_up[self.unit_offset, a_slot] = S
                self.W_up[self.unit_offset, b_slot] = S
                self.b_up[self.unit_offset] = -S * (a + b - 0.5)

                self._opcode_gate(self.unit_offset)

                # Output result
                self.W_down[result_slot, self.unit_offset] = result / S

                self.unit_offset += 1

    def emit_io_putchar_request(self, char_nibbles: list, io_char_start: int, io_ready_flag: int):
        """Emit weights for PUTCHAR I/O request.

        Copies character nibbles to IO_CHAR mailbox and sets IO_OUTPUT_READY flag.

        Args:
            char_nibbles: List of (position, slot) for character nibbles
            io_char_start: IO_CHAR mailbox start (E.IO_CHAR)
            io_ready_flag: IO_OUTPUT_READY flag (E.IO_OUTPUT_READY)
        """
        S = self.reg_map.SCALE

        # Copy character nibbles to I/O mailbox
        for i, (src_pos, src_slot) in enumerate(char_nibbles):
            src_idx = self._fi(src_pos, src_slot)
            dest_idx = io_char_start + i

            # Cancel pair: copy nibble
            self.W_up[self.unit_offset, src_idx] = S
            self._opcode_gate(self.unit_offset)
            self.W_down[dest_idx, self.unit_offset] = 1.0 / S

            self.W_up[self.unit_offset + 1, src_idx] = -S
            self._opcode_gate(self.unit_offset + 1)
            self.W_down[dest_idx, self.unit_offset + 1] = 1.0 / S

            self.unit_offset += 2

        # Set IO_OUTPUT_READY flag
        self.emit_flag_set(io_ready_flag)

    def emit_io_getchar_request(self, io_need_input_flag: int):
        """Emit weights for GETCHAR I/O request.

        Simply sets IO_NEED_INPUT flag.

        Args:
            io_need_input_flag: IO_NEED_INPUT flag (E.IO_NEED_INPUT)
        """
        self.emit_flag_set(io_need_input_flag)

    def emit_stack_push_request(self, sp_nibbles: list, data_nibbles: list,
                                mem_addr_start: int, mem_data_start: int, mem_write_flag: int):
        """Emit weights for stack push operation.

        Computes: SP -= 8, *SP = data

        Args:
            sp_nibbles: List of (position, slot) for SP register
            data_nibbles: List of (position, slot) for data to push
            mem_addr_start: MEM_ADDR mailbox start
            mem_data_start: MEM_DATA mailbox start
            mem_write_flag: MEM_WRITE flag
        """
        S = self.reg_map.SCALE

        # First, decrement SP by 8
        # For nibble 0 (least significant): SP[0] -= 8
        sp_pos, sp_slot = sp_nibbles[0]
        sp_idx = self._fi(sp_pos, sp_slot)

        # Subtract 8 from SP[0]
        # Cancel pair: SP[0] = SP[0] - 8
        self.W_up[self.unit_offset, sp_idx] = S
        self._opcode_gate(self.unit_offset)
        self.b_gate[self.unit_offset] = -8.0  # Subtract 8
        self.W_down[sp_idx, self.unit_offset] = 1.0 / S

        self.W_up[self.unit_offset + 1, sp_idx] = -S
        self._opcode_gate(self.unit_offset + 1)
        self.b_gate[self.unit_offset + 1] = 8.0
        self.W_down[sp_idx, self.unit_offset + 1] = 1.0 / S

        self.unit_offset += 2

        # Copy updated SP to MEM_ADDR, data to MEM_DATA, set MEM_WRITE
        self.emit_mem_write_request(sp_nibbles, data_nibbles,
                                    mem_addr_start, mem_data_start, mem_write_flag)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return weight matrices for PureFFN integration."""
        return {
            'W_up': self.W_up,
            'b_up': self.b_up,
            'W_gate': self.W_gate,
            'b_gate': self.b_gate,
            'W_down': self.W_down,
            'b_down': self.b_down,
        }

    def get_active_units(self) -> int:
        """Return number of hidden units used."""
        return self.unit_offset


# =============================================================================
# Nibble Weight Compiler (High-Level Interface)
# =============================================================================

class NibbleWeightCompiler:
    """High-level interface for compiling operations to nibble-based FFN weights.

    Usage:
        compiler = NibbleWeightCompiler()

        # Compile ADD operation
        weights = compiler.compile_operation(OpType.ADD, Opcode.ADD)

        # Apply weights to AutoregressiveVM layer
        vm.blocks[9].ffn.W_up.data = weights['W_up']
        vm.blocks[9].ffn.b_up.data = weights['b_up']
        # ... etc
    """

    def __init__(self, num_positions: int = 8):
        """Initialize nibble weight compiler.

        Args:
            num_positions: Number of nibble positions (8 for 32-bit)
        """
        self.num_positions = num_positions
        self.reg_map = NibbleRegisterMap()

    def compile_operation(self, op_type: OpType, opcode: int, unit_offset: int = 0) -> Dict[str, torch.Tensor]:
        """Compile a single operation to nibble-based FFN weights.

        Args:
            op_type: Operation type from graph_weight_compiler.OpType
            opcode: C4 opcode for gating (from embedding.Opcode)
            unit_offset: Hidden unit offset for non-overlapping allocation

        Returns:
            Dictionary of weight matrices compatible with PureFFN
        """
        from .embedding import E

        emitter = NibbleWeightEmitter(opcode, self.num_positions, unit_offset=unit_offset)

        # Emit operation for all nibble positions
        for pos in range(self.num_positions):
            if op_type == OpType.ADD:
                emitter.emit_add_nibble(pos)
            elif op_type == OpType.SUB:
                emitter.emit_sub_nibble(pos)
            elif op_type == OpType.MUL:
                emitter.emit_mul_nibble(pos)
            elif op_type == OpType.DIV:
                emitter.emit_div_nibble(pos)
            elif op_type == OpType.MOD:
                emitter.emit_mod_nibble(pos)
            elif op_type == OpType.CMP_EQ:
                emitter.emit_cmp_eq_nibble(pos)
            elif op_type == OpType.CMP_NE:
                emitter.emit_cmp_ne_nibble(pos)
            elif op_type == OpType.CMP_LT:
                emitter.emit_cmp_lt_nibble(pos)
            elif op_type == OpType.CMP_GT:
                emitter.emit_cmp_gt_nibble(pos)
            elif op_type == OpType.CMP_LE:
                emitter.emit_cmp_le_nibble(pos)
            elif op_type == OpType.CMP_GE:
                emitter.emit_cmp_ge_nibble(pos)
            elif op_type == OpType.BIT_OR:
                emitter.emit_bitwise_op_nibble(pos, "or")
            elif op_type == OpType.BIT_XOR:
                emitter.emit_bitwise_op_nibble(pos, "xor")
            elif op_type == OpType.BIT_AND:
                emitter.emit_bitwise_op_nibble(pos, "and")
            elif op_type == OpType.SHL:
                emitter.emit_shl_nibble(pos)
            elif op_type == OpType.SHR:
                emitter.emit_shr_nibble(pos)
            elif op_type == OpType.MOVE:
                emitter.emit_move_nibble(pos)
            else:
                raise NotImplementedError(f"OpType {op_type} not yet implemented for nibble compilation")

        # Special case: flag operations (not per-nibble)
        if op_type == OpType.FLAG_SET:
            # Example: SET IO_PROGRAM_END flag
            emitter.emit_flag_set(E.IO_PROGRAM_END)
        elif op_type == OpType.FLAG_CLEAR:
            emitter.emit_flag_clear(E.IO_PROGRAM_END)

        return emitter.get_weights()

    def compile_graph(self, graph: ComputationGraph, opcode: int) -> Dict[str, torch.Tensor]:
        """Compile a computation graph to nibble-based FFN weights.

        This is more complex - requires register allocation, topological sort,
        and sequential emission. For now, we support single-operation graphs.

        Args:
            graph: Computation graph to compile
            opcode: C4 opcode for gating

        Returns:
            Dictionary of weight matrices
        """
        # Check if single-operation graph
        ops = [n for n in graph.nodes.values() if n.op != OpType.CONST]

        if len(ops) == 0:
            raise ValueError("Graph contains no operations")

        if len(ops) == 1:
            # Single operation - compile directly
            return self.compile_operation(ops[0].op, opcode)
        else:
            # Multi-operation graph - needs sequential layers
            raise NotImplementedError(
                "Multi-operation graphs require multi-layer compilation. "
                "Currently only single-operation graphs are supported."
            )

    def print_weight_summary(self, weights: Dict[str, torch.Tensor]):
        """Print summary of compiled weights."""
        total_params = 0
        nonzero_params = 0

        for name, w in weights.items():
            total = w.numel()
            nonzero = (w.abs() > 1e-9).sum().item()
            total_params += total
            nonzero_params += nonzero
            print(f"  {name:8s}: {w.shape} - {nonzero}/{total} non-zero ({100*nonzero/total:.2f}%)")

        sparsity = 100 * (1 - nonzero_params / total_params)
        print(f"\n  Total: {nonzero_params}/{total_params} non-zero parameters ({sparsity:.2f}% sparse)")
