"""
Neural PC Computation Layer

Layer 13 computes next PC based on current PC, opcode, immediate, and AX.

Logic:
- Most opcodes: next_PC = PC + 8
- JMP: next_PC = immediate
- BZ: next_PC = immediate if AX==0 else PC+8
- BNZ: next_PC = immediate if AX!=0 else PC+8
- JSR: next_PC = immediate
- (LEV handled in memory layer)

Implementation:
- Use SwiGLU step functions for conditionals
- W_gate: Activate based on opcode
- W_up: Compute conditions (AX==0, etc.)
- W_down: Write next PC to RESULT slots
"""

import torch
import torch.nn as nn
from typing import Optional
from neural_vm.embedding import E, Opcode


class NeuralPCLayer(nn.Module):
    """
    Computes next PC neurally using layer 13 weights.

    Input embedding has:
    - Current PC in TEMP slots (8 nibbles)
    - Immediate in AX_BASE slots (8 nibbles)
    - Opcode one-hot in OP_START slots
    - AX in NIB_A slots (for BZ/BNZ)

    Output embedding has:
    - Next PC in RESULT slots (8 nibbles)
    """

    def __init__(self, d_model: int = 1280):
        super().__init__()
        self.d_model = d_model
        self.dim_per_pos = 160
        self.num_positions = 8

    def forward(self, x: torch.Tensor, opcode: int, pc: int, imm: int, ax: int) -> torch.Tensor:
        """
        Compute next PC.

        For now, use Python logic with neural execution planned.
        This is a temporary implementation to get the framework working.

        Args:
            x: Input embedding [batch, seq, d_model]
            opcode: Current opcode
            pc: Current PC
            imm: Immediate value
            ax: AX register value

        Returns:
            Updated embedding with next PC in RESULT slots
        """
        # Python fallback for now - will be replaced with neural
        if opcode == Opcode.JMP:
            next_pc = imm
        elif opcode == Opcode.BZ:
            next_pc = imm if ax == 0 else pc + 8
        elif opcode == Opcode.BNZ:
            next_pc = imm if ax != 0 else pc + 8
        elif opcode == Opcode.JSR:
            next_pc = imm
        else:
            # Default: increment PC
            next_pc = pc + 8

        # Encode next PC into RESULT slots
        for pos in range(self.num_positions):
            base_idx = pos * self.dim_per_pos
            pc_nibble = (next_pc >> (pos * 4)) & 0xF
            x[:, :, base_idx + E.RESULT] = float(pc_nibble)

        return x


def create_neural_pc_computer(vm_layer: nn.Module) -> callable:
    """
    Create a function that uses a VM layer to compute next PC.

    Args:
        vm_layer: Layer 13 from AutoregressiveVM

    Returns:
        Function that computes next PC neurally
    """
    def compute_next_pc(embedding: torch.Tensor, opcode: int, pc: int, imm: int, ax: int) -> torch.Tensor:
        """
        Compute next PC through layer 13.

        Args:
            embedding: Current state [1, 1, 1280]
            opcode: Current opcode
            pc: Current PC
            imm: Immediate value
            ax: AX value

        Returns:
            Updated embedding with next PC
        """
        # For now, use Python logic (hybrid approach)
        # TODO: Full neural implementation using layer 13 weights

        # Compute next PC
        if opcode == Opcode.JMP:
            next_pc = imm
        elif opcode == Opcode.BZ:
            next_pc = imm if ax == 0 else pc + 8
        elif opcode == Opcode.BNZ:
            next_pc = imm if ax != 0 else pc + 8
        elif opcode == Opcode.JSR:
            next_pc = imm
        elif opcode == Opcode.LEV:
            # LEV next PC comes from stack (handled separately)
            next_pc = pc  # Placeholder
        else:
            next_pc = pc + 8

        # Encode into RESULT slots
        for pos in range(8):
            base_idx = pos * 160
            pc_nibble = (next_pc >> (pos * 4)) & 0xF
            embedding[:, :, base_idx + E.RESULT] = float(pc_nibble)

        return embedding

    return compute_next_pc
