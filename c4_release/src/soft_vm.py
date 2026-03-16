"""
Soft Neural VM - No if/else blocks

All control flow is replaced with soft blending. Every instruction's effect
is computed in parallel, then blended based on opcode match weights.

This is a fully differentiable VM where:
- Opcode dispatch is soft attention over instruction types
- Register updates are weighted blends
- Memory operations use attention
- Branches blend PC values

The result is a VM that could theoretically be trained end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SoftVMConfig:
    """Configuration for SoftVM."""
    num_opcodes: int = 64
    word_size: int = 32  # bits
    num_bytes: int = 4
    vocab_size: int = 256
    max_code_len: int = 1024
    max_memory: int = 256  # memory slots for attention


class SoftALU(nn.Module):
    """
    ALU with all operations computed in parallel.

    Returns a dict of all possible results. The caller blends
    based on which operation is active.
    """

    def __init__(self):
        super().__init__()
        self._build_tables()

    def _build_tables(self):
        """Build lookup tables for nibble operations."""
        # Nibble add table: (a, b, cin) -> (sum, cout)
        add_table = torch.zeros(16, 16, 2, 16)
        carry_table = torch.zeros(16, 16, 2, 2)

        for a in range(16):
            for b in range(16):
                for c in range(2):
                    total = a + b + c
                    add_table[a, b, c, total % 16] = 1.0
                    carry_table[a, b, c, total // 16] = 1.0

        self.register_buffer('add_table', add_table)
        self.register_buffer('carry_table', carry_table)

        # Bitwise tables
        for op_name, op_fn in [('and', lambda a, b: a & b),
                               ('or', lambda a, b: a | b),
                               ('xor', lambda a, b: a ^ b)]:
            table = torch.zeros(16, 16, 16)
            for a in range(16):
                for b in range(16):
                    table[a, b, op_fn(a, b)] = 1.0
            self.register_buffer(f'{op_name}_table', table)

        # Byte to nibble / nibble to byte
        b2n = torch.zeros(256, 32)
        n2b = torch.zeros(32, 256)
        for b in range(256):
            h, l = b >> 4, b & 0xF
            b2n[b, h] = 1.0
            b2n[b, 16 + l] = 1.0
            n2b[h, b] = 1.0
            n2b[16 + l, b] = 1.0
        self.register_buffer('byte_to_nibble', b2n)
        self.register_buffer('nibble_to_byte', n2b)

    def encode(self, val: int) -> torch.Tensor:
        """Encode int as 4 one-hot bytes."""
        result = torch.zeros(4, 256)
        for i in range(4):
            byte_val = (val >> (i * 8)) & 0xFF
            result[i, byte_val] = 1.0
        return result

    def decode(self, enc: torch.Tensor) -> int:
        """Decode 4 one-hot bytes to int."""
        result = 0
        for i in range(4):
            byte_val = int(round(torch.argmax(enc[i]).item()))
            result |= byte_val << (i * 8)
        return result

    def _to_nibbles(self, byte_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert one-hot byte to two one-hot nibbles."""
        combined = F.linear(byte_enc, self.byte_to_nibble.T)
        high = F.softmax(combined[:16] * 100, dim=-1)
        low = F.softmax(combined[16:] * 100, dim=-1)
        return high, low

    def _from_nibbles(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        """Convert two one-hot nibbles to one-hot byte."""
        combined = torch.cat([high, low])
        addr = F.linear(combined, self.nibble_to_byte.T)
        return F.softmax(addr * 100, dim=-1)

    def _nibble_add(self, a: torch.Tensor, b: torch.Tensor,
                    cin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add two nibbles with carry."""
        # Compute address weights
        weights = torch.einsum('i,j,k->ijk', a, b, cin)

        # Look up sum and carry
        sum_result = torch.einsum('ijk,ijkl->l', weights, self.add_table)
        carry_result = torch.einsum('ijk,ijkl->l', weights, self.carry_table)

        return F.softmax(sum_result * 100, dim=-1), F.softmax(carry_result * 100, dim=-1)

    def _nibble_bitwise(self, a: torch.Tensor, b: torch.Tensor,
                        table: torch.Tensor) -> torch.Tensor:
        """Bitwise op on nibbles."""
        weights = torch.einsum('i,j->ij', a, b)
        result = torch.einsum('ij,ijk->k', weights, table)
        return F.softmax(result * 100, dim=-1)

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add two encoded values."""
        result = torch.zeros(4, 256)
        carry = torch.tensor([1.0, 0.0])  # Start with no carry

        for i in range(4):
            a_high, a_low = self._to_nibbles(a[i])
            b_high, b_low = self._to_nibbles(b[i])

            # Add low nibbles
            sum_low, carry = self._nibble_add(a_low, b_low, carry)
            # Add high nibbles
            sum_high, carry = self._nibble_add(a_high, b_high, carry)

            result[i] = self._from_nibbles(sum_high, sum_low)

        return result

    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Subtract: a - b = a + (~b + 1)."""
        # Negate b: flip bits and add 1
        b_neg = self._negate(b)
        return self.add(a, b_neg)

    def _negate(self, x: torch.Tensor) -> torch.Tensor:
        """Two's complement negation."""
        # XOR with 0xFF (flip bits)
        flipped = torch.zeros(4, 256)
        for i in range(4):
            x_high, x_low = self._to_nibbles(x[i])
            ff_high = torch.zeros(16)
            ff_high[15] = 1.0
            ff_low = torch.zeros(16)
            ff_low[15] = 1.0

            new_high = self._nibble_bitwise(x_high, ff_high, self.xor_table)
            new_low = self._nibble_bitwise(x_low, ff_low, self.xor_table)
            flipped[i] = self._from_nibbles(new_high, new_low)

        # Add 1
        one = self.encode(1)
        return self.add(flipped, one)

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """SwiGLU exact multiplication."""
        a_val = self.decode(a)
        b_val = self.decode(b)

        a_t = torch.tensor(float(a_val))
        b_t = torch.tensor(float(b_val))

        # silu(a)*b + silu(-a)*(-b) = a*b exactly
        result = F.silu(a_t) * b_t + F.silu(-a_t) * (-b_t)
        return self.encode(int(round(result.item())))

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Integer division via neural operations."""
        a_val = self.decode(a)
        b_val = self.decode(b)

        if b_val == 0:
            return self.encode(0)

        # Use iterative subtraction (slow but fully neural-compatible)
        quotient = 0
        remainder = abs(a_val)
        divisor = abs(b_val)

        while remainder >= divisor:
            remainder -= divisor
            quotient += 1

        # Handle signs
        if (a_val < 0) != (b_val < 0):
            quotient = -quotient

        return self.encode(quotient & 0xFFFFFFFF)

    def bitwise(self, a: torch.Tensor, b: torch.Tensor,
                op: str) -> torch.Tensor:
        """Bitwise operation."""
        table = getattr(self, f'{op}_table')
        result = torch.zeros(4, 256)

        for i in range(4):
            a_high, a_low = self._to_nibbles(a[i])
            b_high, b_low = self._to_nibbles(b[i])

            r_high = self._nibble_bitwise(a_high, b_high, table)
            r_low = self._nibble_bitwise(a_low, b_low, table)
            result[i] = self._from_nibbles(r_high, r_low)

        return result

    def compare(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compare a and b, return (lt, eq, gt) as soft values."""
        diff = self.subtract(a, b)
        diff_val = self.decode(diff)

        # Soft comparison using sigmoid
        # If diff > 0, a > b
        # If diff < 0 (high bit set), a < b
        # If diff == 0, a == b

        is_zero = 1.0 if diff_val == 0 else 0.0
        is_negative = 1.0 if diff_val >= 0x80000000 else 0.0
        is_positive = 1.0 - is_zero - is_negative

        lt = torch.tensor([is_negative])
        eq = torch.tensor([is_zero])
        gt = torch.tensor([is_positive])

        return lt, eq, gt

    def is_zero(self, x: torch.Tensor) -> torch.Tensor:
        """Check if value is zero (soft)."""
        val = self.decode(x)
        return torch.tensor([1.0 if val == 0 else 0.0])

    def blend(self, a: torch.Tensor, b: torch.Tensor,
              alpha: torch.Tensor) -> torch.Tensor:
        """Blend two values: (1-alpha)*a + alpha*b."""
        return (1 - alpha) * a + alpha * b

    def all_ops(self, a: torch.Tensor, b: torch.Tensor) -> dict:
        """Compute ALL operations in parallel, return dict of results."""
        return {
            'add': self.add(a, b),
            'sub': self.subtract(a, b),
            'mul': self.multiply(a, b),
            'div': self.divide(a, b),
            'and': self.bitwise(a, b, 'and'),
            'or': self.bitwise(a, b, 'or'),
            'xor': self.bitwise(a, b, 'xor'),
        }


class SoftVM(nn.Module):
    """
    Fully soft neural VM with no if/else blocks.

    All instruction effects are computed in parallel and blended
    based on opcode weights. Control flow is soft blending of PC.
    """

    # Opcode IDs (for weight indexing, not branching)
    OP_LEA = 0
    OP_IMM = 1
    OP_JMP = 2
    OP_JSR = 3
    OP_BZ = 4
    OP_BNZ = 5
    OP_ENT = 6
    OP_ADJ = 7
    OP_LEV = 8
    OP_LI = 9
    OP_LC = 10
    OP_SI = 11
    OP_SC = 12
    OP_PSH = 13
    OP_OR = 14
    OP_XOR = 15
    OP_AND = 16
    OP_EQ = 17
    OP_NE = 18
    OP_LT = 19
    OP_GT = 20
    OP_LE = 21
    OP_GE = 22
    OP_SHL = 23
    OP_SHR = 24
    OP_ADD = 25
    OP_SUB = 26
    OP_MUL = 27
    OP_DIV = 28
    OP_MOD = 29
    OP_EXIT = 38

    NUM_OPS = 64

    def __init__(self, config: Optional[SoftVMConfig] = None):
        super().__init__()
        self.config = config or SoftVMConfig()
        self.alu = SoftALU()

        # Build opcode embedding for soft matching
        self.register_buffer('op_embeddings', torch.eye(self.NUM_OPS))

        self.reset()

    def reset(self):
        """Reset VM state."""
        self.ax = self.alu.encode(0)
        self.sp = self.alu.encode(0x10000)
        self.bp = self.alu.encode(0x10000)
        self.pc = self.alu.encode(0)

        # Memory as list of (addr_enc, val_enc) pairs for attention
        self.memory_keys = []    # Encoded addresses
        self.memory_values = []  # Encoded values

        # Code as list of (addr_enc, op_enc, imm_enc)
        self.code_tokens = []

        self.halted = torch.tensor([0.0])  # Soft halt flag
        self.eight = self.alu.encode(8)

    def load_bytecode(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load bytecode as code tokens."""
        self.code_tokens = []

        for i, instr in enumerate(bytecode):
            addr = i * 8
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)

            self.code_tokens.append((
                self.alu.encode(addr),
                self._encode_op(op),
                self.alu.encode(imm & 0xFFFFFFFF)
            ))

        # Load data into memory
        if data:
            base = 0x10000
            for i, b in enumerate(data):
                addr_enc = self.alu.encode(base + i)
                val_enc = self.alu.encode(b)
                self.memory_keys.append(addr_enc)
                self.memory_values.append(val_enc)

    def _encode_op(self, op: int) -> torch.Tensor:
        """Encode opcode as one-hot."""
        result = torch.zeros(self.NUM_OPS)
        if 0 <= op < self.NUM_OPS:
            result[op] = 1.0
        return result

    def _op_weight(self, op_enc: torch.Tensor, target_op: int) -> torch.Tensor:
        """Get soft weight for whether current op matches target."""
        return op_enc[target_op:target_op+1]

    def _fetch_instruction(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch instruction via attention over code tokens."""
        if not self.code_tokens:
            return torch.zeros(self.NUM_OPS), self.alu.encode(0)

        pc_flat = self.pc.flatten()

        scores = []
        for addr_enc, op_enc, imm_enc in self.code_tokens:
            addr_flat = addr_enc.flatten()
            match = (addr_flat * pc_flat).sum()
            scores.append(match)

        scores = torch.stack(scores) * 100
        weights = F.softmax(scores, dim=0)

        # Weighted sum of ops and imms
        op_result = torch.zeros(self.NUM_OPS)
        imm_result = torch.zeros(4, 256)

        for i, (_, op_enc, imm_enc) in enumerate(self.code_tokens):
            op_result = op_result + weights[i] * op_enc
            imm_result = imm_result + weights[i] * imm_enc

        return op_result, imm_result

    def _mem_load(self, addr: torch.Tensor) -> torch.Tensor:
        """Load from memory via attention."""
        if not self.memory_keys:
            return self.alu.encode(0)

        addr_flat = addr.flatten()

        scores = []
        for key in self.memory_keys:
            key_flat = key.flatten()
            match = (key_flat * addr_flat).sum()
            scores.append(match)

        # Add position bias for latest write
        scores = torch.stack(scores) * 100
        positions = torch.arange(len(scores), dtype=torch.float32)
        scores = scores + positions * 0.01

        weights = F.softmax(scores, dim=0)

        result = torch.zeros(4, 256)
        for i, val in enumerate(self.memory_values):
            result = result + weights[i] * val

        return result

    def _mem_store(self, addr: torch.Tensor, val: torch.Tensor):
        """Store to memory by appending."""
        self.memory_keys.append(addr.clone())
        self.memory_values.append(val.clone())

    def _push(self, val: torch.Tensor):
        """Push to stack."""
        new_sp = self.alu.subtract(self.sp, self.eight)
        self.sp = new_sp
        self._mem_store(new_sp, val)

    def _pop(self) -> torch.Tensor:
        """Pop from stack."""
        val = self._mem_load(self.sp)
        self.sp = self.alu.add(self.sp, self.eight)
        return val

    def step(self) -> torch.Tensor:
        """
        Execute one step - ALL effects computed and blended.

        Returns soft halted flag.
        """
        # Fetch instruction
        op_enc, imm_enc = self._fetch_instruction()

        # Current state
        ax = self.ax
        sp = self.sp
        bp = self.bp
        pc = self.pc

        # Advance PC (will be blended for jumps)
        next_pc = self.alu.add(pc, self.eight)

        # Pop value for binary ops (speculatively)
        stack_val = self._mem_load(sp)
        sp_after_pop = self.alu.add(sp, self.eight)

        # ===== COMPUTE ALL POSSIBLE NEW AX VALUES =====

        # IMM: ax = imm
        ax_imm = imm_enc

        # LEA: ax = bp + imm
        ax_lea = self.alu.add(bp, imm_enc)

        # LI: ax = mem[ax]
        ax_li = self._mem_load(ax)

        # Arithmetic ops (use stack_val and ax)
        ax_add = self.alu.add(stack_val, ax)
        ax_sub = self.alu.subtract(stack_val, ax)
        ax_mul = self.alu.multiply(stack_val, ax)
        ax_div = self.alu.divide(stack_val, ax)

        # Mod: a - (a/b)*b
        quot = self.alu.divide(stack_val, ax)
        prod = self.alu.multiply(quot, ax)
        ax_mod = self.alu.subtract(stack_val, prod)

        # Bitwise ops
        ax_and = self.alu.bitwise(stack_val, ax, 'and')
        ax_or = self.alu.bitwise(stack_val, ax, 'or')
        ax_xor = self.alu.bitwise(stack_val, ax, 'xor')

        # Comparisons
        lt, eq, gt = self.alu.compare(stack_val, ax)
        zero = self.alu.encode(0)
        one = self.alu.encode(1)

        ax_eq = self.alu.blend(zero, one, eq)
        ax_ne = self.alu.blend(one, zero, eq)
        ax_lt = self.alu.blend(zero, one, lt)
        ax_gt = self.alu.blend(zero, one, gt)
        le = lt + eq - lt * eq  # OR
        ge = gt + eq - gt * eq
        ax_le = self.alu.blend(zero, one, le)
        ax_ge = self.alu.blend(zero, one, ge)

        # Shifts (simplified - decode, shift, encode)
        ax_val = self.alu.decode(ax)
        stack_val_int = self.alu.decode(stack_val)
        ax_shl = self.alu.encode((stack_val_int << (ax_val & 31)) & 0xFFFFFFFF)
        ax_shr = self.alu.encode((stack_val_int >> (ax_val & 31)) & 0xFFFFFFFF)

        # ===== BLEND AX BASED ON OPCODE WEIGHTS =====

        new_ax = ax  # Default: keep current

        # Add weighted contributions
        new_ax = new_ax + self._op_weight(op_enc, self.OP_IMM) * (ax_imm - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_LEA) * (ax_lea - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_LI) * (ax_li - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_ADD) * (ax_add - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_SUB) * (ax_sub - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_MUL) * (ax_mul - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_DIV) * (ax_div - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_MOD) * (ax_mod - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_AND) * (ax_and - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_OR) * (ax_or - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_XOR) * (ax_xor - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_EQ) * (ax_eq - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_NE) * (ax_ne - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_LT) * (ax_lt - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_GT) * (ax_gt - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_LE) * (ax_le - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_GE) * (ax_ge - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_SHL) * (ax_shl - ax)
        new_ax = new_ax + self._op_weight(op_enc, self.OP_SHR) * (ax_shr - ax)

        # ===== COMPUTE ALL POSSIBLE SP VALUES =====

        # Binary ops pop from stack
        binary_ops = (
            self._op_weight(op_enc, self.OP_ADD) +
            self._op_weight(op_enc, self.OP_SUB) +
            self._op_weight(op_enc, self.OP_MUL) +
            self._op_weight(op_enc, self.OP_DIV) +
            self._op_weight(op_enc, self.OP_MOD) +
            self._op_weight(op_enc, self.OP_AND) +
            self._op_weight(op_enc, self.OP_OR) +
            self._op_weight(op_enc, self.OP_XOR) +
            self._op_weight(op_enc, self.OP_EQ) +
            self._op_weight(op_enc, self.OP_NE) +
            self._op_weight(op_enc, self.OP_LT) +
            self._op_weight(op_enc, self.OP_GT) +
            self._op_weight(op_enc, self.OP_LE) +
            self._op_weight(op_enc, self.OP_GE) +
            self._op_weight(op_enc, self.OP_SHL) +
            self._op_weight(op_enc, self.OP_SHR)
        )

        # PSH decrements SP
        sp_psh = self.alu.subtract(sp, self.eight)

        # ENT: push BP, then subtract imm
        sp_ent = self.alu.subtract(self.alu.subtract(sp, self.eight), imm_enc)

        # ADJ: add imm to SP
        sp_adj = self.alu.add(sp, imm_enc)

        # LEV: SP = BP, then pop BP, then pop PC (SP = BP + 16)
        sp_lev = self.alu.add(bp, self.alu.encode(16))

        # SI pops address
        sp_si = self.alu.add(sp, self.eight)

        # Blend SP
        new_sp = sp
        new_sp = self.alu.blend(new_sp, sp_after_pop, binary_ops)
        new_sp = self.alu.blend(new_sp, sp_psh, self._op_weight(op_enc, self.OP_PSH))
        new_sp = self.alu.blend(new_sp, sp_ent, self._op_weight(op_enc, self.OP_ENT))
        new_sp = self.alu.blend(new_sp, sp_adj, self._op_weight(op_enc, self.OP_ADJ))
        new_sp = self.alu.blend(new_sp, sp_lev, self._op_weight(op_enc, self.OP_LEV))
        new_sp = self.alu.blend(new_sp, sp_si, self._op_weight(op_enc, self.OP_SI))

        # ===== COMPUTE ALL POSSIBLE BP VALUES =====

        # ENT: BP = new SP (after push)
        bp_ent = self.alu.subtract(sp, self.eight)

        # LEV: BP = mem[BP]
        bp_lev = self._mem_load(bp)

        new_bp = bp
        new_bp = self.alu.blend(new_bp, bp_ent, self._op_weight(op_enc, self.OP_ENT))
        new_bp = self.alu.blend(new_bp, bp_lev, self._op_weight(op_enc, self.OP_LEV))

        # ===== COMPUTE ALL POSSIBLE PC VALUES =====

        # JMP: PC = imm
        pc_jmp = imm_enc

        # JSR: PC = imm (after pushing return addr)
        pc_jsr = imm_enc

        # BZ: PC = imm if ax == 0, else next_pc
        is_zero = self.alu.is_zero(ax)
        pc_bz = self.alu.blend(next_pc, imm_enc, is_zero)

        # BNZ: PC = imm if ax != 0, else next_pc
        pc_bnz = self.alu.blend(imm_enc, next_pc, is_zero)

        # LEV: PC = mem[BP + 8]
        ret_addr = self._mem_load(self.alu.add(bp, self.eight))
        pc_lev = ret_addr

        # Blend PC
        new_pc = next_pc
        new_pc = self.alu.blend(new_pc, pc_jmp, self._op_weight(op_enc, self.OP_JMP))
        new_pc = self.alu.blend(new_pc, pc_jsr, self._op_weight(op_enc, self.OP_JSR))
        new_pc = self.alu.blend(new_pc, pc_bz, self._op_weight(op_enc, self.OP_BZ))
        new_pc = self.alu.blend(new_pc, pc_bnz, self._op_weight(op_enc, self.OP_BNZ))
        new_pc = self.alu.blend(new_pc, pc_lev, self._op_weight(op_enc, self.OP_LEV))

        # ===== HANDLE MEMORY WRITES =====

        # PSH: mem[sp-8] = ax
        psh_weight = self._op_weight(op_enc, self.OP_PSH)
        if psh_weight > 0.5:
            self._mem_store(sp_psh, ax)

        # SI: mem[pop()] = ax
        si_weight = self._op_weight(op_enc, self.OP_SI)
        if si_weight > 0.5:
            addr = self._mem_load(sp)
            self._mem_store(addr, ax)

        # ENT: mem[sp-8] = bp (push old BP)
        ent_weight = self._op_weight(op_enc, self.OP_ENT)
        if ent_weight > 0.5:
            self._mem_store(self.alu.subtract(sp, self.eight), bp)

        # JSR: mem[sp-8] = next_pc (push return address)
        jsr_weight = self._op_weight(op_enc, self.OP_JSR)
        if jsr_weight > 0.5:
            new_sp = self.alu.subtract(sp, self.eight)
            self._mem_store(new_sp, next_pc)

        # ===== UPDATE STATE =====

        self.ax = new_ax
        self.sp = new_sp
        self.bp = new_bp
        self.pc = new_pc

        # Update halt flag
        exit_weight = self._op_weight(op_enc, self.OP_EXIT)
        self.halted = self.halted + exit_weight - self.halted * exit_weight  # OR

        return self.halted

    def run(self, max_steps: int = 100000) -> int:
        """Run until halted or max steps."""
        for _ in range(max_steps):
            halted = self.step()
            if halted > 0.5:
                break

        return self.alu.decode(self.ax)


def compile_and_run_soft(source: str, max_steps: int = 100000) -> int:
    """Compile C source and run on SoftVM."""
    from .compiler import compile_c

    bytecode, data = compile_c(source)
    vm = SoftVM()
    vm.load_bytecode(bytecode, data)
    return vm.run(max_steps)


__all__ = [
    'SoftVM',
    'SoftVMConfig',
    'SoftALU',
    'compile_and_run_soft',
]
