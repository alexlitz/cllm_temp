"""
Integrated Neural VM - ALU + KV Cache Memory

Combines the MoE-based ALU with attention-based KV cache memory
for a complete virtual machine implementation.

Key components:
- SparseMoEALU: Handles arithmetic, logic, control flow
- KVMemoryCache: Handles LOAD/STORE via attention
- VMState: Tracks PC, registers, flags
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .embedding import E, Opcode
from .sparse_moe_alu import SparseMoEALU
from .kv_memory import (
    KVMemoryConfig, KVMemoryCache, VMStateEncoder,
    VMStepLayout, AddressEncoder
)


def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with +1 in denominator - graceful missing value handling."""
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return exp_x / (1.0 + exp_x.sum(dim=dim, keepdim=True))


@dataclass
class VMState:
    """Current state of the virtual machine."""
    pc: int = 0           # Program counter
    ax: int = 0           # Accumulator
    sp: int = 0xFFFF      # Stack pointer (starts at top)
    bp: int = 0           # Base pointer
    halted: bool = False
    step: int = 0


class MemoryLoadAttention(nn.Module):
    """
    Attention layer for LOAD operation.

    Query: Address we want to read
    Keys: All previously stored addresses
    Values: Corresponding stored values

    Uses high temperature scaling for sharp address matching.
    """

    def __init__(self, dim: int = 64, num_heads: int = 1, alibi_slope: float = 0.01):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # High scale for sharp matching
        self.scale = 10.0 / math.sqrt(self.head_dim)
        self.alibi_slope = alibi_slope

    def forward(
        self,
        query_addr: torch.Tensor,  # [batch, dim] - address to load
        key_addrs: torch.Tensor,   # [batch, cache_len, dim] - stored addresses
        values: torch.Tensor,       # [batch, cache_len, dim] - stored values
        query_pos: int,             # Current position
        key_pos: torch.Tensor,      # [batch, cache_len] - positions of stored values
    ) -> torch.Tensor:
        """
        Attention-based memory load.

        Returns loaded value, or zero vector if address not found.
        """
        batch = query_addr.shape[0]
        cache_len = key_addrs.shape[1]

        Q = query_addr.unsqueeze(1)  # [batch, 1, dim]
        K = key_addrs  # [batch, cache_len, dim]
        V = values  # [batch, cache_len, dim]

        # Attention scores - dot product for exact matching
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, 1, cache_len]

        # ALiBi bias: prefer recent writes (small effect)
        distances = query_pos - key_pos.unsqueeze(1)  # [batch, 1, cache_len]
        alibi_bias = -self.alibi_slope * distances.abs()
        scores = scores + alibi_bias

        # Standard softmax for sharp selection
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        output = torch.matmul(weights, V)  # [batch, 1, dim]
        return output.squeeze(1)


class IntegratedVM(nn.Module):
    """
    Integrated Neural VM with ALU and KV Cache Memory.

    Execution flow per instruction:
    1. Fetch instruction from program memory (external)
    2. Encode operands into ALU input format
    3. Run ALU for arithmetic/logic/control
    4. For LOAD: Run attention over KV cache
    5. For STORE: Append to KV cache
    6. Update VM state (PC, registers)
    """

    def __init__(
        self,
        mem_dim: int = 64,
        num_heads: int = 4,
        max_memory: int = 65536,
    ):
        super().__init__()

        # ALU for computation
        self.alu = SparseMoEALU()

        # Memory configuration
        self.mem_config = KVMemoryConfig(
            dim=mem_dim,
            num_heads=num_heads,
            memory_size=max_memory,
        )

        # Memory cache
        self.memory = KVMemoryCache(self.mem_config)

        # Address encoder for memory ops
        self.addr_encoder = AddressEncoder(self.mem_config)

        # Memory load attention
        self.load_attn = MemoryLoadAttention(mem_dim, num_heads)

        # State
        self.state = VMState()

    def reset(self):
        """Reset VM to initial state."""
        self.state = VMState()
        self.memory.reset_cache()

    def encode_operands(
        self,
        opcode: int,
        operand_a: int,
        operand_b: int,
    ) -> torch.Tensor:
        """
        Encode instruction operands for ALU input.

        Returns [1, 8, DIM] tensor in ALU format.
        """
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

        for i in range(E.NUM_POSITIONS):
            x[0, i, E.NIB_A] = float((operand_a >> (i * 4)) & 0xF)
            x[0, i, E.NIB_B] = float((operand_b >> (i * 4)) & 0xF)
            x[0, i, E.OP_START + opcode] = 1.0
            x[0, i, E.POS] = float(i)

        return x

    def decode_result(self, x: torch.Tensor) -> int:
        """Extract 32-bit result from ALU output."""
        result = 0
        for i in range(E.NUM_POSITIONS):
            nib = int(round(x[0, i, E.RESULT].item()))
            nib = max(0, min(15, nib))
            result |= (nib << (i * 4))
        return result

    def execute_alu(
        self,
        opcode: int,
        operand_a: int,
        operand_b: int,
    ) -> int:
        """Execute ALU operation and return result."""
        x = self.encode_operands(opcode, operand_a, operand_b)
        x = self.alu(x)
        return self.decode_result(x)

    def execute_load(self, address: int) -> int:
        """
        Execute LOAD instruction via KV cache attention.

        Returns value at address, or 0 if not found.
        """
        if self.memory.cache_len == 0:
            return 0

        # Encode address as nibbles
        addr_nibbles = torch.tensor([[
            (address >> 0) & 0xF,
            (address >> 4) & 0xF,
            (address >> 8) & 0xF,
            (address >> 12) & 0xF,
        ]])

        # Encode address for attention
        query = self.addr_encoder.encode_address(addr_nibbles)

        # Run attention over cache
        result = self.load_attn(
            query,
            self.memory.cache_keys,
            self.memory.cache_values,
            self.state.step,
            self.memory.cache_positions,
        )

        # Decode value from embedding (stored in first 32 dims as nibbles)
        value = 0
        for i in range(8):
            nib = int(round(result[0, i].item()))
            nib = max(0, min(15, nib))
            value |= (nib << (i * 4))
        return value

    def execute_store(self, address: int, value: int):
        """
        Execute STORE instruction by appending to KV cache.

        Newer writes at same address will shadow older ones via ALiBi.
        """
        # Encode address as nibbles
        addr_nibbles = torch.tensor([[
            (address >> 0) & 0xF,
            (address >> 4) & 0xF,
            (address >> 8) & 0xF,
            (address >> 12) & 0xF,
        ]])

        # Encode address as key
        key = self.addr_encoder.encode_address(addr_nibbles)

        # Encode value as nibbles in first 8 dimensions
        val = torch.zeros(1, self.mem_config.dim)
        for i in range(8):
            val[0, i] = float((value >> (i * 4)) & 0xF)

        # Position
        pos = torch.tensor([[self.state.step]])

        # Append to cache
        self.memory.append_to_cache(
            key.unsqueeze(1),
            val.unsqueeze(1),
            pos,
        )

    def step(
        self,
        opcode: int,
        operand_a: int = 0,
        operand_b: int = 0,
    ) -> Tuple[int, bool]:
        """
        Execute one instruction.

        Returns (result, halted).
        """
        if self.state.halted:
            return 0, True

        result = 0

        # Memory operations
        if opcode == Opcode.LOAD:
            result = self.execute_load(operand_a)

        elif opcode == Opcode.STORE:
            self.execute_store(operand_a, operand_b)
            result = operand_b

        elif opcode == Opcode.PUSH:
            self.state.sp -= 4  # Decrement stack pointer
            self.execute_store(self.state.sp, operand_a)
            result = operand_a

        elif opcode == Opcode.POP:
            result = self.execute_load(self.state.sp)
            self.state.sp += 4  # Increment stack pointer

        elif opcode == Opcode.HALT:
            self.state.halted = True
            result = 0

        # Control flow
        elif opcode == Opcode.JMP:
            self.state.pc = operand_b
            result = operand_b

        elif opcode in {Opcode.BEQ, Opcode.BNE, Opcode.BLT, Opcode.BGE}:
            result = self.execute_alu(opcode, operand_a, operand_b)
            if result != 0:
                self.state.pc = result
            else:
                self.state.pc += 1

        # ALU operations
        else:
            result = self.execute_alu(opcode, operand_a, operand_b)
            self.state.ax = result
            self.state.pc += 1

        self.state.step += 1
        return result, self.state.halted


def demo_integrated_vm():
    """Demonstrate the integrated VM."""
    print("=== Integrated Neural VM Demo ===\n")

    vm = IntegratedVM()

    # Test ALU operations
    print("1. ALU Operations:")
    result = vm.step(Opcode.ADD, 10, 20)
    print(f"   ADD 10, 20 = {result[0]}")

    result = vm.step(Opcode.MUL, 6, 7)
    print(f"   MUL 6, 7 = {result[0]}")

    # Test memory operations
    print("\n2. Memory Operations:")
    vm.step(Opcode.STORE, 0x100, 42)
    print(f"   STORE 0x100, 42")

    result = vm.step(Opcode.LOAD, 0x100)
    print(f"   LOAD 0x100 = {result[0]}")

    # Test stack operations
    print("\n3. Stack Operations:")
    vm.step(Opcode.PUSH, 100)
    print(f"   PUSH 100 (SP = {vm.state.sp})")

    vm.step(Opcode.PUSH, 200)
    print(f"   PUSH 200 (SP = {vm.state.sp})")

    result = vm.step(Opcode.POP)
    print(f"   POP = {result[0]} (SP = {vm.state.sp})")

    # Test control flow
    print("\n4. Control Flow:")
    vm.state.pc = 0
    result = vm.step(Opcode.BEQ, 0, 100)
    print(f"   BEQ 0, 100 -> PC = {vm.state.pc} (branch taken)")

    result = vm.step(Opcode.BEQ, 1, 200)
    print(f"   BEQ 1, 200 -> PC = {vm.state.pc} (branch not taken)")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_integrated_vm()
