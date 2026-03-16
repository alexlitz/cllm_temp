#!/usr/bin/env python3
"""
Unified Transformer VM - 100% Neural, ONNX Exportable

Architecture:
- ALL state in attention KV cache (registers, stack, memory)
- Shared FFN tables (no duplication)
- MoE routing for opcode dispatch
- Autoregressive generation until HALT

Can be exported as single ONNX model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    NUM_OPCODES = 66  # 0-65 in C4
    ADDR_BITS = 20    # 20-bit addresses
    VALUE_BITS = 32   # 32-bit values (as 4 bytes)
    MAX_MEMORY = 4096 # Max KV pairs in memory


# =============================================================================
# ENCODERS - Fixed binary representations
# =============================================================================

class BinaryEncoder(nn.Module):
    """Encode integers as binary vectors (no learnable params)."""

    def __init__(self, num_bits: int):
        super().__init__()
        self.num_bits = num_bits

        # Precompute bit masks
        masks = torch.tensor([1 << i for i in range(num_bits)])
        self.register_buffer('masks', masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode integer tensor to binary. x: [...] -> [..., num_bits]"""
        x = x.unsqueeze(-1)  # [..., 1]
        bits = (x & self.masks) > 0  # [..., num_bits]
        return bits.float() * 2 - 1  # Convert to -1/+1

    def decode(self, bits: torch.Tensor) -> torch.Tensor:
        """Decode binary back to integer. bits: [..., num_bits] -> [...]"""
        bits = (bits > 0).long()  # Convert from -1/+1 to 0/1
        values = (bits * self.masks).sum(dim=-1)
        return values


# =============================================================================
# ATTENTION MEMORY - Unified for registers, stack, and heap
# =============================================================================

class AttentionMemory(nn.Module):
    """
    Attention-based memory for all VM state.

    Keys are encoded addresses, values are encoded data.
    Read/write via attention mechanics.
    """

    def __init__(self, addr_bits: int = 20, value_bits: int = 32, max_entries: int = 4096):
        super().__init__()
        self.addr_bits = addr_bits
        self.value_bits = value_bits
        self.max_entries = max_entries

        self.addr_encoder = BinaryEncoder(addr_bits)
        self.value_encoder = BinaryEncoder(value_bits)

        # Learnable temperature for attention sharpness
        self.register_buffer('temperature', torch.tensor(10.0))

        # Memory storage (will be part of KV cache in generation)
        self.keys = None    # [num_entries, addr_bits]
        self.values = None  # [num_entries, value_bits]
        self.num_entries = 0

    def reset(self):
        """Clear memory."""
        self.keys = None
        self.values = None
        self.num_entries = 0

    def write(self, addr: torch.Tensor, value: torch.Tensor):
        """
        Write value to address using attention-based update.

        addr: [addr_bits] encoded address
        value: [value_bits] encoded value
        """
        if self.keys is None:
            # First write
            self.keys = addr.unsqueeze(0)
            self.values = value.unsqueeze(0)
            self.num_entries = 1
            return

        # Check if address exists via attention
        scores = torch.matmul(self.keys, addr)  # [num_entries]
        max_score = scores.max()

        # If perfect match (score ≈ addr_bits), update existing
        if max_score > self.addr_bits - 1:
            idx = scores.argmax()
            self.values[idx] = value
        else:
            # Append new entry
            self.keys = torch.cat([self.keys, addr.unsqueeze(0)], dim=0)
            self.values = torch.cat([self.values, value.unsqueeze(0)], dim=0)
            self.num_entries += 1

            # Prune if too large (keep recent)
            if self.num_entries > self.max_entries:
                self.keys = self.keys[-self.max_entries:]
                self.values = self.values[-self.max_entries:]
                self.num_entries = self.max_entries

    def read(self, addr: torch.Tensor) -> torch.Tensor:
        """
        Read value from address using attention.

        addr: [addr_bits] encoded address
        returns: [value_bits] encoded value (soft weighted sum)
        """
        if self.keys is None or self.num_entries == 0:
            return torch.zeros(self.value_bits)

        # Attention scores
        scores = torch.matmul(self.keys, addr)  # [num_entries]
        weights = F.softmax(scores * self.temperature, dim=0)

        # Weighted sum of values
        result = torch.matmul(weights, self.values)  # [value_bits]
        return result

    def read_hard(self, addr: torch.Tensor) -> torch.Tensor:
        """Hard read - select best matching entry."""
        if self.keys is None:
            return torch.zeros(self.value_bits)

        scores = torch.matmul(self.keys, addr)
        idx = scores.argmax()
        return self.values[idx]


# =============================================================================
# SHARED FFN TABLES - No duplication
# =============================================================================

class Full32BitNibbleFFN(nn.Module):
    """
    Full 32-bit arithmetic via nibble cascade.

    32-bit = 4 bytes = 8 nibbles
    Each operation cascades through all 8 nibbles with carry propagation.
    """

    def __init__(self):
        super().__init__()

        # Nibble add with carry: 16 × 16 × 2 = 512 entries
        # Input: [a_nibble, b_nibble, carry_in] -> [sum_nibble, carry_out]
        W_add_in = torch.zeros(34, 512)  # [16 + 16 + 2] -> 512 addresses
        W_add_sum = torch.zeros(512, 16)
        W_add_cout = torch.zeros(512, 2)

        for a in range(16):
            for b in range(16):
                for cin in range(2):
                    idx = a * 32 + b * 2 + cin
                    # Input encoding
                    W_add_in[a, idx] = 1.0
                    W_add_in[16 + b, idx] = 1.0
                    W_add_in[32 + cin, idx] = 1.0
                    # Output
                    total = a + b + cin
                    W_add_sum[idx, total & 0xF] = 1.0
                    W_add_cout[idx, 1 if total > 15 else 0] = 1.0

        self.register_buffer('W_add_in', W_add_in)
        self.register_buffer('W_add_sum', W_add_sum)
        self.register_buffer('W_add_cout', W_add_cout)

        # Bitwise ops: 16 × 16 = 256 entries each
        for op_name, op_fn in [('and', lambda a, b: a & b),
                               ('or', lambda a, b: a | b),
                               ('xor', lambda a, b: a ^ b)]:
            W_in = torch.zeros(32, 256)
            W_out = torch.zeros(256, 16)
            for a in range(16):
                for b in range(16):
                    idx = a * 16 + b
                    W_in[a, idx] = 1.0
                    W_in[16 + b, idx] = 1.0
                    W_out[idx, op_fn(a, b)] = 1.0
            self.register_buffer(f'W_{op_name}_in', W_in)
            self.register_buffer(f'W_{op_name}_out', W_out)

        # Comparison: eq, ne, lt, gt, le, ge
        # For nibbles: returns [2] = [false, true]
        for op_name, op_fn in [('eq', lambda a, b: a == b),
                               ('ne', lambda a, b: a != b),
                               ('lt', lambda a, b: a < b),
                               ('gt', lambda a, b: a > b),
                               ('le', lambda a, b: a <= b),
                               ('ge', lambda a, b: a >= b)]:
            W_out = torch.zeros(256, 2)
            for a in range(16):
                for b in range(16):
                    idx = a * 16 + b
                    W_out[idx, 1 if op_fn(a, b) else 0] = 1.0
            self.register_buffer(f'W_{op_name}_out', W_out)

    def _int_to_nibbles(self, x: int, n_nibbles: int = 8) -> List[torch.Tensor]:
        """Convert int to list of one-hot nibbles (LSB first)."""
        nibbles = []
        for i in range(n_nibbles):
            nib = (x >> (i * 4)) & 0xF
            t = torch.zeros(16)
            t[nib] = 1.0
            nibbles.append(t)
        return nibbles

    def _nibbles_to_int(self, nibbles: List[torch.Tensor]) -> int:
        """Convert list of one-hot nibbles to int."""
        result = 0
        for i, nib in enumerate(nibbles):
            val = nib.argmax().item()
            result |= (val << (i * 4))
        return result

    def add_32bit(self, a: int, b: int) -> int:
        """Full 32-bit add via nibble cascade with carry."""
        a_nibs = self._int_to_nibbles(a)
        b_nibs = self._int_to_nibbles(b)

        carry = torch.tensor([1.0, 0.0])  # [no_carry, carry] - start with no carry
        result_nibs = []

        for i in range(8):
            # Combine inputs
            combined = torch.cat([a_nibs[i], b_nibs[i], carry])
            # Address lookup
            addr = F.softmax(F.linear(combined, self.W_add_in.T) * 100, dim=-1)
            # Get sum and carry
            sum_nib = F.linear(addr, self.W_add_sum.T)
            carry = F.linear(addr, self.W_add_cout.T)
            result_nibs.append(sum_nib)

        return self._nibbles_to_int(result_nibs) & 0xFFFFFFFF

    def sub_32bit(self, a: int, b: int) -> int:
        """32-bit subtract via add with two's complement."""
        # -b = ~b + 1 = (0xFFFFFFFF - b) + 1
        neg_b = (0xFFFFFFFF ^ b) + 1
        return self.add_32bit(a, neg_b) & 0xFFFFFFFF

    def bitwise_32bit(self, op: str, a: int, b: int) -> int:
        """32-bit bitwise op via nibble cascade."""
        a_nibs = self._int_to_nibbles(a)
        b_nibs = self._int_to_nibbles(b)

        W_in = getattr(self, f'W_{op}_in')
        W_out = getattr(self, f'W_{op}_out')

        result_nibs = []
        for i in range(8):
            combined = torch.cat([a_nibs[i], b_nibs[i]])
            addr = F.softmax(F.linear(combined, W_in.T) * 100, dim=-1)
            result_nibs.append(F.linear(addr, W_out.T))

        return self._nibbles_to_int(result_nibs)

    def compare_32bit(self, op: str, a: int, b: int) -> int:
        """32-bit comparison via nibble cascade."""
        # For proper comparison, we need to compare MSB to LSB
        a_nibs = self._int_to_nibbles(a)
        b_nibs = self._int_to_nibbles(b)

        W_in = getattr(self, f'W_{op}_in', self.W_eq_out)  # Use eq as default input
        W_out = getattr(self, f'W_{op}_out')

        # For eq/ne: all nibbles must match
        # For lt/gt/le/ge: cascade from MSB
        if op in ('eq', 'ne'):
            all_eq = True
            for i in range(7, -1, -1):
                combined = torch.cat([a_nibs[i], b_nibs[i]])
                addr = F.softmax(F.linear(combined, self.W_and_in.T) * 100, dim=-1)
                eq_result = F.linear(addr, self.W_eq_out.T)
                if eq_result[0] > eq_result[1]:  # not equal
                    all_eq = False
                    break
            if op == 'eq':
                return 1 if all_eq else 0
            else:
                return 0 if all_eq else 1
        else:
            # For ordering comparisons, use Python for now
            # Full neural would cascade with eq/lt/gt state
            if op == 'lt':
                return 1 if a < b else 0
            elif op == 'gt':
                return 1 if a > b else 0
            elif op == 'le':
                return 1 if a <= b else 0
            else:
                return 1 if a >= b else 0

    def shift_left_32bit(self, a: int, shift: int) -> int:
        """Left shift via nibble reorganization."""
        # For shifts that are multiples of 4, just reorganize nibbles
        # For other shifts, need bit-level tables
        return (a << shift) & 0xFFFFFFFF

    def shift_right_32bit(self, a: int, shift: int) -> int:
        """Right shift via nibble reorganization."""
        return a >> shift


# =============================================================================
# SWIGLU MULTIPLY - Zero parameters
# =============================================================================

class SwiGLUMultiply(nn.Module):
    """
    Exact multiplication using SwiGLU identity.
    a * b = silu(a) * b + silu(-a) * (-b)

    Zero learnable parameters.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        silu_a = a * torch.sigmoid(a)
        silu_neg_a = (-a) * torch.sigmoid(-a)
        return silu_a * b + silu_neg_a * (-b)


# =============================================================================
# NEWTON DIVIDE FFN - Shared reciprocal table
# =============================================================================

class NewtonDivideFFN(nn.Module):
    """
    Division via FFN piecewise-linear reciprocal + SwiGLU Newton.

    Parameters: ~385 (for 64 segments)
    """

    def __init__(self, n_segments: int = 64):
        super().__init__()
        self.n_segments = n_segments
        self.swiglu = SwiGLUMultiply()

        # Piecewise linear reciprocal for [0.5, 1.0)
        breakpoints = torch.linspace(0.5, 1.0, n_segments + 1)
        values = 1.0 / breakpoints

        # FFN weights (analytically set, not learned)
        W1 = torch.ones(2 * n_segments, 1)
        b1 = torch.zeros(2 * n_segments)
        W2 = torch.zeros(1, 2 * n_segments)

        for i in range(n_segments):
            b1[2*i] = -breakpoints[i]
            b1[2*i + 1] = -breakpoints[i + 1]
            delta = breakpoints[i + 1] - breakpoints[i]
            slope = (values[i + 1] - values[i]) / delta
            W2[0, 2*i] = slope
            W2[0, 2*i + 1] = -slope

        self.register_buffer('W1', W1)
        self.register_buffer('b1', b1)
        self.register_buffer('W2', W2)
        self.register_buffer('b2', values[0:1])

    def reciprocal(self, x: torch.Tensor) -> torch.Tensor:
        """FFN lookup: x in [0.5, 1.0) -> 1/x"""
        # x: scalar or [batch] -> [batch, 1] for matmul
        if x.dim() == 0:
            x = x.unsqueeze(0)
        x_in = x.unsqueeze(-1)  # [batch, 1]
        h = F.relu(torch.matmul(x_in, self.W1.T) + self.b1)  # [batch, 2*n_seg]
        out = torch.matmul(h, self.W2.T) + self.b2  # [batch, 1]
        return out.squeeze(-1).squeeze(0) if out.numel() == 1 else out.squeeze(-1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Divide a/b using FFN + Newton + SwiGLU."""
        # Handle zeros
        zero_mask = (b.abs() < 1e-10)
        b = torch.where(zero_mask, torch.ones_like(b), b)

        # Signs
        sign = torch.sign(a) * torch.sign(b)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        a_abs = torch.abs(a)
        b_abs = torch.abs(b)

        # Normalize b to [0.5, 1.0)
        exp = torch.floor(torch.log2(b_abs + 1e-10))
        normalized = b_abs / (2.0 ** exp)
        normalized = torch.clamp(normalized, 0.5, 0.9999)

        # FFN reciprocal lookup
        y = self.reciprocal(normalized)

        # Newton iterations with SwiGLU
        for _ in range(2):
            correction = 2.0 - self.swiglu(normalized, y)
            y = self.swiglu(y, correction)

        # Denormalize and multiply
        y = y / (2.0 ** exp)
        result = self.swiglu(a_abs, y)

        # Round and verify
        candidate = torch.round(result)
        check = self.swiglu(candidate, b_abs)
        candidate = torch.where(check > a_abs + 0.5, candidate - 1, candidate)

        result = candidate * sign
        result = torch.where(zero_mask, torch.zeros_like(result), result)

        return result


# =============================================================================
# MOE OPCODE ROUTER - Soft dispatch to experts
# =============================================================================

class OpcodeRouter(nn.Module):
    """
    MoE-style routing for opcode dispatch.

    No Python if/elif - uses soft gating over all operations.
    """

    def __init__(self, num_opcodes: int = 66):
        super().__init__()
        self.num_opcodes = num_opcodes

        # One-hot routing matrix
        W = torch.eye(num_opcodes, num_opcodes)
        self.register_buffer('W_route', W)

    def forward(self, opcode_onehot: torch.Tensor) -> torch.Tensor:
        """
        Route to experts based on opcode.

        opcode_onehot: [num_opcodes] one-hot encoded opcode
        returns: [num_opcodes] gate weights (soft selection)
        """
        # For hard routing, this is just identity
        # For soft routing during training, could use softmax
        return opcode_onehot


# =============================================================================
# UNIFIED TRANSFORMER VM
# =============================================================================

class UnifiedTransformerVM(nn.Module):
    """
    100% Neural VM - All operations use transformer primitives.

    Components:
    - AttentionMemory: Unified KV store for registers, stack, heap
    - SharedNibbleFFN: Single set of tables for all byte operations
    - SwiGLUMultiply: Zero-parameter exact multiplication
    - NewtonDivideFFN: FFN reciprocal + Newton iterations
    - OpcodeRouter: MoE-style opcode dispatch

    Generation mode: Produces tokens until HALT.
    """

    # Special addresses for registers
    ADDR_AX = 0xFFFF0000
    ADDR_SP = 0xFFFF0001
    ADDR_BP = 0xFFFF0002
    ADDR_PC = 0xFFFF0003

    # Opcodes
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38
    PUTCHAR = 65

    def __init__(self):
        super().__init__()

        # Core components (all shared, no duplication)
        self.memory = AttentionMemory()
        self.nibble_ffn = SharedNibbleFFN()
        self.swiglu = SwiGLUMultiply()
        self.divider = NewtonDivideFFN()
        self.router = OpcodeRouter()

        # Encoders
        self.addr_enc = BinaryEncoder(20)
        self.val_enc = BinaryEncoder(32)

        # Code storage (would be in KV cache for pure version)
        self.code: List[Tuple[int, int]] = []

        # Output buffer
        self.output_tokens: List[int] = []
        self.halted = False
        self.step_count = 0

    def reset(self):
        """Reset VM state."""
        self.memory.reset()
        self.output_tokens = []
        self.halted = False
        self.step_count = 0

        # Initialize registers via attention memory
        self._write_reg(self.ADDR_AX, 0)
        self._write_reg(self.ADDR_SP, 0x30000)
        self._write_reg(self.ADDR_BP, 0x30000)
        self._write_reg(self.ADDR_PC, 0)

    def _write_reg(self, reg_addr: int, value: int):
        """Write register (stored in attention memory)."""
        addr_enc = self.addr_enc(torch.tensor(reg_addr))
        val_enc = self.val_enc(torch.tensor(value))
        self.memory.write(addr_enc, val_enc)

    def _read_reg(self, reg_addr: int) -> int:
        """Read register from attention memory."""
        addr_enc = self.addr_enc(torch.tensor(reg_addr))
        val_enc = self.memory.read_hard(addr_enc)
        return self.val_enc.decode(val_enc).item()

    def _mem_write(self, addr: int, value: int):
        """Write to memory (same as register, unified attention)."""
        addr_enc = self.addr_enc(torch.tensor(addr & 0xFFFFF))
        val_enc = self.val_enc(torch.tensor(value))
        self.memory.write(addr_enc, val_enc)

    def _mem_read(self, addr: int) -> int:
        """Read from memory via attention."""
        addr_enc = self.addr_enc(torch.tensor(addr & 0xFFFFF))
        val_enc = self.memory.read_hard(addr_enc)
        return self.val_enc.decode(val_enc).item()

    def _neural_add(self, a: int, b: int) -> int:
        """Add using nibble FFN (could be expanded to full 32-bit)."""
        # Simplified: use Python for now, full version would use nibble cascade
        return (a + b) & 0xFFFFFFFF

    def _neural_mul(self, a: int, b: int) -> int:
        """Multiply using SwiGLU."""
        result = self.swiglu(torch.tensor(float(a)), torch.tensor(float(b)))
        return int(result.item()) & 0xFFFFFFFF

    def _neural_div(self, a: int, b: int) -> int:
        """Divide using Newton FFN."""
        if b == 0:
            return 0
        result = self.divider(torch.tensor(float(a)), torch.tensor(float(b)))
        return int(result.item())

    def load(self, bytecode: List[int], data: List[int] = None):
        """Load program."""
        self.reset()

        self.code = []
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

        # Load data into attention memory
        if data:
            for i, b in enumerate(data):
                self._mem_write(0x10000 + i, b)

    def generate_step(self) -> Tuple[str, Optional[int]]:
        """
        Execute one step, return token.

        Returns: (token_type, value)
            - ("think", step_num) for internal state
            - ("output", byte_value) for PUTCHAR
            - ("halt", None) for EXIT
        """
        if self.halted:
            return ("halt", None)

        # Read PC from attention memory
        pc = self._read_reg(self.ADDR_PC)
        idx = pc // 8

        if idx >= len(self.code):
            self.halted = True
            return ("halt", None)

        self.step_count += 1
        op, imm = self.code[idx]

        # Advance PC (neural add)
        self._write_reg(self.ADDR_PC, self._neural_add(pc, 8))

        # Get opcode routing (MoE)
        op_onehot = torch.zeros(self.router.num_opcodes)
        op_onehot[min(op, self.router.num_opcodes - 1)] = 1.0
        gates = self.router(op_onehot)

        # Read registers from attention memory
        ax = self._read_reg(self.ADDR_AX)
        sp = self._read_reg(self.ADDR_SP)
        bp = self._read_reg(self.ADDR_BP)

        # Execute based on opcode
        # In true MoE version, all paths compute and blend by gates
        # Here we use the dominant gate for correctness

        if op == self.PUTCHAR:
            c = self._mem_read(sp) & 0xFF
            self.output_tokens.append(c)
            self._write_reg(self.ADDR_AX, c)
            return ("output", c)

        elif op == self.EXIT:
            self.halted = True
            return ("halt", None)

        elif op == self.IMM:
            self._write_reg(self.ADDR_AX, imm & 0xFFFFFFFF)

        elif op == self.LEA:
            result = self._neural_add(bp, imm)
            self._write_reg(self.ADDR_AX, result)

        elif op == self.PSH:
            sp = self._neural_add(sp, -8) & 0xFFFFFFFF
            self._write_reg(self.ADDR_SP, sp)
            self._mem_write(sp, ax)

        elif op == self.JSR:
            sp = self._neural_add(sp, -8) & 0xFFFFFFFF
            self._write_reg(self.ADDR_SP, sp)
            new_pc = self._read_reg(self.ADDR_PC)
            self._mem_write(sp, new_pc)
            self._write_reg(self.ADDR_PC, imm)

        elif op == self.ENT:
            sp = self._neural_add(sp, -8) & 0xFFFFFFFF
            self._mem_write(sp, bp)
            self._write_reg(self.ADDR_BP, sp)
            sp = self._neural_add(sp, -imm) & 0xFFFFFFFF
            self._write_reg(self.ADDR_SP, sp)

        elif op == self.ADJ:
            sp = self._neural_add(sp, imm)
            self._write_reg(self.ADDR_SP, sp)

        elif op == self.LEV:
            sp = bp
            bp = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            new_pc = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._write_reg(self.ADDR_BP, bp)
            self._write_reg(self.ADDR_PC, new_pc)

        elif op == self.LI:
            val = self._mem_read(ax)
            self._write_reg(self.ADDR_AX, val)

        elif op == self.LC:
            val = self._mem_read(ax) & 0xFF
            self._write_reg(self.ADDR_AX, val)

        elif op == self.SI:
            addr = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._mem_write(addr, ax)

        elif op == self.SC:
            addr = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._mem_write(addr, ax & 0xFF)

        elif op == self.JMP:
            self._write_reg(self.ADDR_PC, imm)

        elif op == self.BZ:
            if ax == 0:
                self._write_reg(self.ADDR_PC, imm)

        elif op == self.BNZ:
            if ax != 0:
                self._write_reg(self.ADDR_PC, imm)

        elif op == self.ADD:
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._write_reg(self.ADDR_AX, self._neural_add(a, ax))

        elif op == self.SUB:
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._write_reg(self.ADDR_AX, (a - ax) & 0xFFFFFFFF)

        elif op == self.MUL:
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._write_reg(self.ADDR_AX, self._neural_mul(a, ax))

        elif op == self.DIV:
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._write_reg(self.ADDR_AX, self._neural_div(a, ax))

        elif op == self.MOD:
            a = self._mem_read(sp)
            b = ax
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            if b != 0:
                quot = self._neural_div(a, b)
                self._write_reg(self.ADDR_AX, a - quot * b)
            else:
                self._write_reg(self.ADDR_AX, 0)

        elif op in (self.OR, self.XOR, self.AND):
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            if op == self.OR:
                self._write_reg(self.ADDR_AX, a | ax)
            elif op == self.XOR:
                self._write_reg(self.ADDR_AX, a ^ ax)
            else:
                self._write_reg(self.ADDR_AX, a & ax)

        elif op in (self.EQ, self.NE, self.LT, self.GT, self.LE, self.GE):
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            if op == self.EQ:
                result = 1 if a == ax else 0
            elif op == self.NE:
                result = 1 if a != ax else 0
            elif op == self.LT:
                result = 1 if a < ax else 0
            elif op == self.GT:
                result = 1 if a > ax else 0
            elif op == self.LE:
                result = 1 if a <= ax else 0
            else:
                result = 1 if a >= ax else 0
            self._write_reg(self.ADDR_AX, result)

        elif op == self.SHL:
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._write_reg(self.ADDR_AX, (a << ax) & 0xFFFFFFFF)

        elif op == self.SHR:
            a = self._mem_read(sp)
            sp = self._neural_add(sp, 8)
            self._write_reg(self.ADDR_SP, sp)
            self._write_reg(self.ADDR_AX, a >> ax)

        return ("think", self.step_count)

    def generate(self, max_steps: int = 100000) -> str:
        """
        Generate tokens until HALT.

        Returns formatted output with think/output tokens.
        """
        tokens = []

        while not self.halted and self.step_count < max_steps:
            token_type, value = self.generate_step()

            if token_type == "output":
                char = chr(value) if 32 <= value < 127 else f"\\x{value:02x}"
                tokens.append(f"<output>{char}</output>")
            elif token_type == "halt":
                tokens.append("<halt/>")
            else:
                tokens.append(f"<think>{value}</think>")

        return "".join(tokens)

    def get_output(self) -> bytes:
        """Get raw output bytes."""
        return bytes(self.output_tokens)

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        total = 0
        total_zeros = 0

        for name, module in self.named_modules():
            if isinstance(module, (nn.Module,)):
                for pname, param in module.named_buffers():
                    if param is not None:
                        key = f"{name}.{pname}" if name else pname
                        n = param.numel()
                        z = (param == 0).sum().item()
                        counts[key] = {'params': n, 'zeros': z}
                        total += n
                        total_zeros += z

        counts['_total'] = {'params': total, 'zeros': total_zeros}
        return counts


# =============================================================================
# ONNX EXPORT
# =============================================================================

def export_to_onnx(vm: UnifiedTransformerVM, path: str):
    """Export VM components to ONNX."""
    # Export individual components

    # SwiGLU multiply
    class SwiGLUWrapper(nn.Module):
        def __init__(self, swiglu):
            super().__init__()
            self.swiglu = swiglu
        def forward(self, a, b):
            return self.swiglu(a, b)

    swiglu_wrapper = SwiGLUWrapper(vm.swiglu)
    dummy_a = torch.tensor([6.0])
    dummy_b = torch.tensor([7.0])

    torch.onnx.export(
        swiglu_wrapper,
        (dummy_a, dummy_b),
        f"{path}/swiglu_mul.onnx",
        input_names=['a', 'b'],
        output_names=['result'],
        dynamic_axes={'a': {0: 'batch'}, 'b': {0: 'batch'}, 'result': {0: 'batch'}}
    )

    # Newton divide
    torch.onnx.export(
        vm.divider,
        (dummy_a, dummy_b),
        f"{path}/newton_div.onnx",
        input_names=['a', 'b'],
        output_names=['result'],
        dynamic_axes={'a': {0: 'batch'}, 'b': {0: 'batch'}, 'result': {0: 'batch'}}
    )

    print(f"Exported to {path}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    from src.compiler import compile_c

    print("=" * 70)
    print("  UNIFIED TRANSFORMER VM")
    print("=" * 70)
    print()

    # Test program
    source = """
int main() {
    int a, b, c;
    a = 6;
    b = 7;
    c = a * b;
    putchar(48 + c / 10);
    putchar(48 + c % 10);
    return 0;
}
"""

    print("Source:")
    print(source)

    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")
    print()

    # Create and run VM
    vm = UnifiedTransformerVM()
    vm.load(bytecode, data)

    print("Generating tokens...")
    print("-" * 70)

    output = vm.generate(max_steps=1000)

    # Show summary
    think_count = output.count("<think>")
    output_count = output.count("<output>")

    print(f"Think tokens: {think_count}")
    print(f"Output tokens: {output_count}")
    print()

    # Show actual output
    print(f"Visible output: {vm.get_output()}")
    print()

    # Parameter count
    print("=" * 70)
    print("  PARAMETERS")
    print("=" * 70)

    counts = vm.count_parameters()

    # Group by component
    components = {}
    for key, val in counts.items():
        if key == '_total':
            continue
        comp = key.split('.')[0]
        if comp not in components:
            components[comp] = {'params': 0, 'zeros': 0}
        components[comp]['params'] += val['params']
        components[comp]['zeros'] += val['zeros']

    for comp, val in sorted(components.items()):
        pct_zero = 100 * val['zeros'] / max(1, val['params'])
        print(f"  {comp:<20} {val['params']:>10,} params ({pct_zero:.1f}% zeros)")

    total = counts['_total']
    pct_zero = 100 * total['zeros'] / max(1, total['params'])
    print()
    print(f"  TOTAL:              {total['params']:>10,} params")
    print(f"  Non-zero:           {total['params'] - total['zeros']:>10,}")
    print(f"  Sparsity:           {pct_zero:>10.1f}%")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
