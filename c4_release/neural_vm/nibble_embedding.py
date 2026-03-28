"""
Nibble-Based VM Embedding

Converts VM state to nibble-based embeddings for d_model=1280 architecture.

Key differences from token-based embedding:
- Direct nibble representation (8 positions × 160 dims = 1280)
- Opcode encoding replicated across positions
- Register values split into nibbles
- No sequential token format - just VM state

Usage:
    embed = NibbleVMEmbedding(d_model=1280)
    
    # Encode VM state
    state_embedding = embed.encode_vm_state(
        pc=0, ax=5, sp=4096, bp=4096,
        opcode=Opcode.ADD, imm=0
    )
"""

import torch
import torch.nn as nn
from typing import Optional

from .embedding import E, Opcode


class NibbleVMEmbedding(nn.Module):
    """
    Embedding layer for nibble-based VM.
    
    Instead of token sequences, this directly encodes VM state as:
    - 8 positions (nibbles 0-7 of 32-bit values)
    - 160 dimensions per position
    - Total: 1280 dimensions
    
    Each position contains:
    - Computation slots (NIB_A, NIB_B, RESULT, CARRY, etc.)
    - Opcode encoding (one-hot across 72 opcodes)
    - Immediate value encoding
    """
    
    def __init__(self, d_model: int = 1352):
        super().__init__()

        if d_model != 1352:
            raise ValueError(f"NibbleVMEmbedding requires d_model=1352, got {d_model}")

        self.d_model = d_model
        self.dim_per_pos = 169  # DIM from E class (was 168, now 169 with OPCODE scalar)
        self.num_positions = 8   # 8 nibbles for 32-bit
        
        # No learnable parameters - this is a deterministic encoding
        # (Unlike token embedding which learns from data)
        
    def encode_vm_state(
        self,
        pc: int,
        ax: int,
        sp: int,
        bp: int,
        opcode: int,
        imm: int = 0,
        stack_top: Optional[int] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Encode VM state as nibble-based embedding.

        Args:
            pc: Program counter
            ax: AX register value
            sp: Stack pointer
            bp: Base pointer
            opcode: Current opcode (0-71)
            imm: Immediate value
            stack_top: Top of stack value (optional)
            batch_size: Number of parallel VMs

        Returns:
            Tensor of shape [batch_size, d_model]
        """
        # Create embedding tensor
        embedding = torch.zeros(batch_size, self.d_model)

        # For each position (nibble 0-7)
        for pos in range(self.num_positions):
            base_idx = pos * self.dim_per_pos

            # Extract nibble from AX (this will be the operand)
            ax_nibble = (ax >> (pos * 4)) & 0xF

            # Encode AX nibble in NIB_A slot (scalar for ADD/SUB)
            embedding[:, base_idx + E.NIB_A] = float(ax_nibble)

            # Encode NIB_A as binary bits (for exact lookup operations)
            embedding[:, base_idx + E.NIB_A_BIT0] = float((ax_nibble >> 0) & 1)
            embedding[:, base_idx + E.NIB_A_BIT1] = float((ax_nibble >> 1) & 1)
            embedding[:, base_idx + E.NIB_A_BIT2] = float((ax_nibble >> 2) & 1)
            embedding[:, base_idx + E.NIB_A_BIT3] = float((ax_nibble >> 3) & 1)

            # If we have stack top, encode in NIB_B
            if stack_top is not None:
                stack_nibble = (stack_top >> (pos * 4)) & 0xF
                embedding[:, base_idx + E.NIB_B] = float(stack_nibble)

                # Encode NIB_B as binary bits
                embedding[:, base_idx + E.NIB_B_BIT0] = float((stack_nibble >> 0) & 1)
                embedding[:, base_idx + E.NIB_B_BIT1] = float((stack_nibble >> 1) & 1)
                embedding[:, base_idx + E.NIB_B_BIT2] = float((stack_nibble >> 2) & 1)
                embedding[:, base_idx + E.NIB_B_BIT3] = float((stack_nibble >> 3) & 1)

            # Encode current PC in TEMP slot (will be used by control flow)
            pc_nibble = (pc >> (pos * 4)) & 0xF
            embedding[:, base_idx + E.TEMP] = float(pc_nibble)

            # Encode immediate value in AX_BASE slots (reusing these for IMM)
            # AX_BASE is at dim 128, so we can use positions there
            imm_nibble = (imm >> (pos * 4)) & 0xF
            if base_idx + E.AX_BASE < self.d_model:
                embedding[:, base_idx + E.AX_BASE] = float(imm_nibble)

            # Encode opcode as scalar value (shared across positions, for continuous gating)
            embedding[:, base_idx + E.OPCODE] = float(opcode)

            # Encode opcode as one-hot (for clean operation gating in shared layers)
            if 0 <= opcode < E.NUM_OPS:
                embedding[:, base_idx + E.OP_START + opcode] = 1.0

            # Initialize carry to 0
            embedding[:, base_idx + E.CARRY_IN] = 0.0

        return embedding
    
    def encode_register_nibbles(
        self,
        value: int,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Encode a 32-bit register value as 8 nibbles.
        
        Args:
            value: 32-bit integer
            batch_size: Batch size
            
        Returns:
            Tensor of shape [batch_size, d_model] with nibbles in NIB_A slots
        """
        embedding = torch.zeros(batch_size, self.d_model)
        
        for pos in range(self.num_positions):
            base_idx = pos * self.dim_per_pos
            nibble = (value >> (pos * 4)) & 0xF
            embedding[:, base_idx + E.NIB_A] = float(nibble)
        
        return embedding
    
    def decode_result_nibbles(self, embedding: torch.Tensor) -> int:
        """
        Decode result from RESULT slots back to 32-bit integer.

        Args:
            embedding: Output tensor [batch_size, d_model]

        Returns:
            32-bit integer result
        """
        result = 0

        for pos in range(self.num_positions):
            base_idx = pos * self.dim_per_pos
            # Extract value from RESULT slot
            nibble_val = embedding[0, base_idx + E.RESULT].item()
            # Round to nearest integer (should already be close)
            nibble = int(round(nibble_val)) & 0xF
            # Combine into 32-bit value
            result |= (nibble << (pos * 4))

        return result

    def decode_pc_nibbles(self, embedding: torch.Tensor, slot: int = E.RESULT) -> int:
        """
        Decode PC from specified slots (RESULT by default for next PC).

        Args:
            embedding: Output tensor [batch_size, d_model]
            slot: Which slot to decode from (default: E.RESULT for next PC)

        Returns:
            32-bit PC value
        """
        pc = 0

        for pos in range(self.num_positions):
            base_idx = pos * self.dim_per_pos
            # Extract value from specified slot
            nibble_val = embedding[0, base_idx + slot].item()
            # Round to nearest integer
            nibble = int(round(nibble_val)) & 0xF
            # Combine into 32-bit value
            pc |= (nibble << (pos * 4))

        return pc

    def decode_comparison_result(self, embedding: torch.Tensor) -> int:
        """
        Decode comparison result (0 or 1) from TEMP slot at position 0.

        Comparisons write 1.0 or 0.0 to all TEMP slots. We just read position 0.

        Args:
            embedding: Output tensor [batch_size, d_model]

        Returns:
            0 or 1
        """
        # Read TEMP from position 0
        temp_val = embedding[0, E.TEMP].item()
        # Round to 0 or 1
        return 1 if temp_val > 0.5 else 0
    
    def forward(self, vm_state_dict: dict) -> torch.Tensor:
        """
        Forward pass: encode VM state dictionary.
        
        Args:
            vm_state_dict: Dictionary with keys:
                - 'pc', 'ax', 'sp', 'bp', 'opcode', 'imm', 'stack_top' (optional)
                - 'batch_size' (optional, default 1)
        
        Returns:
            Embedding tensor [batch_size, d_model]
        """
        return self.encode_vm_state(**vm_state_dict)


class NibbleSequenceEmbedding(nn.Module):
    """
    Embedding for sequences of VM states (for autoregressive execution).
    
    Each timestep encodes one VM state, building up a sequence of embeddings.
    """
    
    def __init__(self, d_model: int = 1280):
        super().__init__()
        self.nibble_embed = NibbleVMEmbedding(d_model)
        
    def encode_vm_sequence(
        self,
        states: list,  # List of VM state dicts
    ) -> torch.Tensor:
        """
        Encode a sequence of VM states.
        
        Args:
            states: List of VM state dictionaries
            
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size = states[0].get('batch_size', 1)
        seq_len = len(states)
        
        # Stack embeddings for each timestep
        embeddings = []
        for state in states:
            emb = self.nibble_embed.encode_vm_state(**state)
            embeddings.append(emb)
        
        # Shape: [batch_size, seq_len, d_model]
        return torch.stack(embeddings, dim=1)
    
    def forward(self, states: list) -> torch.Tensor:
        """Forward pass for sequence."""
        return self.encode_vm_sequence(states)
