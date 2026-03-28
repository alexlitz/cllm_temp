"""
Bit-Level Primitives for Neural FFN

Implements fundamental binary logic operations (AND, OR, NOT, XOR) as FFN weight patterns.
These can be composed to build complex operations.

Each primitive operates on binary inputs {0, 1} and produces binary outputs.
Uses large scale (S=100) to approximate step functions via SwiGLU.
"""

import torch
from typing import Tuple


class BitPrimitives:
    """Emits FFN weights for bit-level logic operations."""

    def __init__(self, scale: float = 100.0):
        """Initialize bit primitives.

        Args:
            scale: Scaling factor for sharp activation (default 100.0)
        """
        self.S = scale

    def emit_and(self,
                 W_up: torch.Tensor,
                 b_up: torch.Tensor,
                 W_gate: torch.Tensor,
                 b_gate: torch.Tensor,
                 W_down: torch.Tensor,
                 unit_idx: int,
                 bit_a_slot: int,
                 bit_b_slot: int,
                 output_slot: int):
        """Emit weights for AND(a, b).

        Computes: output = a AND b
        Uses 1 hidden unit.

        Mechanism:
            up = S × bit_a
            gate = S × bit_b
            hidden = silu(up) × gate ≈ S² when both 1, 0 otherwise
            output = hidden / S²

        Args:
            W_up, b_up, W_gate, b_gate, W_down: Weight tensors to modify
            unit_idx: Hidden unit index
            bit_a_slot, bit_b_slot: Input bit slot indices
            output_slot: Output slot index
        """
        S = self.S

        # W_up: read bit_a
        W_up[unit_idx, bit_a_slot] = S
        b_up[unit_idx] = 0

        # W_gate: read bit_b
        W_gate[unit_idx, bit_b_slot] = S
        b_gate[unit_idx] = 0

        # W_down: scale by 1/S²
        W_down[output_slot, unit_idx] = 1.0 / (S * S)

    def emit_passthrough(self,
                        W_up: torch.Tensor,
                        b_up: torch.Tensor,
                        W_gate: torch.Tensor,
                        b_gate: torch.Tensor,
                        W_down: torch.Tensor,
                        unit_idx: int,
                        bit_slot: int,
                        output_slot: int,
                        weight: float = 1.0):
        """Emit weights for bit pass-through.

        Computes: output = weight × bit
        Uses 1 hidden unit.

        Mechanism:
            up = S × bit
            gate = 1 (constant)
            hidden = silu(S × bit) × 1 ≈ S when bit=1, 0 when bit=0
            output = hidden × weight / S

        Args:
            W_up, b_up, W_gate, b_gate, W_down: Weight tensors to modify
            unit_idx: Hidden unit index
            bit_slot: Input bit slot index
            output_slot: Output slot index
            weight: Output weight multiplier
        """
        S = self.S

        # W_up: read bit with scale S
        W_up[unit_idx, bit_slot] = S
        b_up[unit_idx] = 0

        # W_gate: constant 1
        # (leave at 0 weights, set bias to 1)
        b_gate[unit_idx] = 1.0

        # W_down: scale by weight/S
        W_down[output_slot, unit_idx] = weight / S

    def emit_or(self,
                W_up: torch.Tensor,
                b_up: torch.Tensor,
                W_gate: torch.Tensor,
                b_gate: torch.Tensor,
                W_down: torch.Tensor,
                unit_start_idx: int,
                bit_a_slot: int,
                bit_b_slot: int,
                output_slot: int) -> int:
        """Emit weights for OR(a, b).

        Computes: output = a OR b = a + b - a×b
        Uses 3 hidden units.

        Args:
            W_up, b_up, W_gate, b_gate, W_down: Weight tensors to modify
            unit_start_idx: Starting hidden unit index
            bit_a_slot, bit_b_slot: Input bit slot indices
            output_slot: Output slot index

        Returns:
            Number of units used (3)
        """
        # Unit 1: pass-through bit_a with weight +1
        self.emit_passthrough(W_up, b_up, W_gate, b_gate, W_down,
                            unit_start_idx, bit_a_slot, output_slot, weight=1.0)

        # Unit 2: pass-through bit_b with weight +1
        self.emit_passthrough(W_up, b_up, W_gate, b_gate, W_down,
                            unit_start_idx + 1, bit_b_slot, output_slot, weight=1.0)

        # Unit 3: AND(a, b) with weight -1
        self.emit_and(W_up, b_up, W_gate, b_gate, W_down,
                     unit_start_idx + 2, bit_a_slot, bit_b_slot, output_slot)
        # Adjust W_down for negative contribution
        W_down[output_slot, unit_start_idx + 2] *= -1.0

        return 3

    def emit_xor(self,
                 W_up: torch.Tensor,
                 b_up: torch.Tensor,
                 W_gate: torch.Tensor,
                 b_gate: torch.Tensor,
                 W_down: torch.Tensor,
                 unit_start_idx: int,
                 bit_a_slot: int,
                 bit_b_slot: int,
                 output_slot: int) -> int:
        """Emit weights for XOR(a, b).

        Computes: output = a XOR b = a + b - 2×a×b
        Uses 3 hidden units.

        Args:
            W_up, b_up, W_gate, b_gate, W_down: Weight tensors to modify
            unit_start_idx: Starting hidden unit index
            bit_a_slot, bit_b_slot: Input bit slot indices
            output_slot: Output slot index

        Returns:
            Number of units used (3)
        """
        # Unit 1: pass-through bit_a with weight +1
        self.emit_passthrough(W_up, b_up, W_gate, b_gate, W_down,
                            unit_start_idx, bit_a_slot, output_slot, weight=1.0)

        # Unit 2: pass-through bit_b with weight +1
        self.emit_passthrough(W_up, b_up, W_gate, b_gate, W_down,
                            unit_start_idx + 1, bit_b_slot, output_slot, weight=1.0)

        # Unit 3: AND(a, b) with weight -2
        self.emit_and(W_up, b_up, W_gate, b_gate, W_down,
                     unit_start_idx + 2, bit_a_slot, bit_b_slot, output_slot)
        # Adjust W_down for -2× contribution
        W_down[output_slot, unit_start_idx + 2] *= -2.0

        return 3

    def emit_not(self,
                 W_up: torch.Tensor,
                 b_up: torch.Tensor,
                 W_gate: torch.Tensor,
                 b_gate: torch.Tensor,
                 W_down: torch.Tensor,
                 unit_idx: int,
                 bit_slot: int,
                 output_slot: int):
        """Emit weights for NOT(a).

        Computes: output = NOT a = 1 - a
        Uses 1 hidden unit.

        Mechanism:
            up = S (constant)
            gate = S - S×a = S×(1 - a)
            hidden = silu(S) × S×(1 - a) ≈ S² × (1 - a)
            output = hidden / S²

        Args:
            W_up, b_up, W_gate, b_gate, W_down: Weight tensors to modify
            unit_idx: Hidden unit index
            bit_slot: Input bit slot index
            output_slot: Output slot index
        """
        S = self.S

        # W_up: constant S
        b_up[unit_idx] = S

        # W_gate: S - S×bit = S×(1 - bit)
        W_gate[unit_idx, bit_slot] = -S
        b_gate[unit_idx] = S

        # W_down: scale by 1/S²
        W_down[output_slot, unit_idx] = 1.0 / (S * S)
