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

    def __init__(self, opcode: int, num_positions: int = 8):
        """Initialize nibble weight emitter.

        Args:
            opcode: C4 opcode for operation gating (from Opcode class)
            num_positions: Number of nibble positions (default 8 for 32-bit)
        """
        self.opcode = opcode
        self.num_positions = num_positions
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

        self.unit_offset = 0  # Current hidden unit index

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

    def compile_operation(self, op_type: OpType, opcode: int) -> Dict[str, torch.Tensor]:
        """Compile a single operation to nibble-based FFN weights.

        Args:
            op_type: Operation type from graph_weight_compiler.OpType
            opcode: C4 opcode for gating (from embedding.Opcode)

        Returns:
            Dictionary of weight matrices compatible with PureFFN
        """
        emitter = NibbleWeightEmitter(opcode, self.num_positions)

        # Emit operation for all nibble positions
        for pos in range(self.num_positions):
            if op_type == OpType.ADD:
                emitter.emit_add_nibble(pos)
            elif op_type == OpType.SUB:
                emitter.emit_sub_nibble(pos)
            elif op_type == OpType.CMP_EQ:
                emitter.emit_cmp_eq_nibble(pos)
            elif op_type == OpType.MOVE:
                emitter.emit_move_nibble(pos)
            else:
                raise NotImplementedError(f"OpType {op_type} not yet implemented for nibble compilation")

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
