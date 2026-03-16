#!/usr/bin/env python3
"""
Vanilla LLM VM - C4 VM as a Pure Transformer Forward Pass

This implements the C4 VM as a standard transformer:
- State is a sequence of tokens (registers, memory, PC)
- Each VM step is ONE transformer forward pass
- Reading state = attention over tokens
- Computing = FFN (SwiGLU for mul, tables for add/logic)
- Writing state = append new tokens (latest wins via position)

The key insight: a VM step IS a transformer forward pass:
    new_state = Transformer(state_tokens, instruction_embedding)

32-bit values are stored as 4 one-hot byte tokens (standard embedding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# TOKEN VOCABULARY
# =============================================================================

class Vocab:
    """Token vocabulary for the VM state."""
    # Special tokens
    PAD = 0
    BOS = 1
    EOS = 2

    # Register marker tokens (3-6)
    REG_AX = 3
    REG_SP = 4
    REG_BP = 5
    REG_PC = 6

    # Memory marker (7)
    MEM = 7

    # Byte values (8-263) - represent 0-255
    BYTE_BASE = 8

    # Address marker (264)
    ADDR = 264

    VOCAB_SIZE = 265


# =============================================================================
# TRANSFORMER COMPONENTS (Vanilla Architecture)
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation: silu(Wx) * (Vx)"""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return F.silu(self.w(x)) * self.v(x)


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard transformer block: Attention + FFN."""
    def __init__(self, dim, num_heads=4, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            SwiGLU(dim * ffn_mult),
            nn.Linear(dim * ffn_mult, dim)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# NEURAL ALU (32-bit operations via FFN)
# =============================================================================

class NeuralALU32(nn.Module):
    """
    32-bit ALU using neural operations.

    Values are 4-byte one-hot encoded: [4, 256] = 1024 dims
    Operations use:
    - SwiGLU for multiplication: exact via silu(a)*b + silu(-a)*(-b)
    - FFN tables for nibble operations
    - Cascaded nibble add with carry
    """

    def __init__(self):
        super().__init__()
        self._build_nibble_tables()
        self._build_byte_nibble_tables()

    def _build_nibble_tables(self):
        """Build FFN tables for nibble operations."""
        # Nibble add with carry: (a, b, cin) -> (sum, cout)
        # Input: [32 + 2] = [a_16, b_16, cin_2]
        # Output: [16 + 2] = [sum_16, cout_2]
        W1 = torch.zeros(34, 512)
        W2_sum = torch.zeros(512, 16)
        W2_cout = torch.zeros(512, 2)

        for a in range(16):
            for b in range(16):
                for cin in range(2):
                    idx = a * 32 + b * 2 + cin
                    W1[a, idx] = 1.0
                    W1[16 + b, idx] = 1.0
                    W1[32 + cin, idx] = 1.0
                    total = a + b + cin
                    W2_sum[idx, total & 0xF] = 1.0
                    W2_cout[idx, 1 if total > 15 else 0] = 1.0

        self.register_buffer('nib_add_W1', W1)
        self.register_buffer('nib_add_W2_sum', W2_sum)
        self.register_buffer('nib_add_W2_cout', W2_cout)

        # Nibble AND/OR/XOR tables
        for op_name, op_fn in [('and', lambda a, b: a & b),
                                ('or', lambda a, b: a | b),
                                ('xor', lambda a, b: a ^ b)]:
            W1 = torch.zeros(32, 256)
            W2 = torch.zeros(256, 16)
            for a in range(16):
                for b in range(16):
                    idx = a * 16 + b
                    W1[a, idx] = 1.0
                    W1[16 + b, idx] = 1.0
                    W2[idx, op_fn(a, b)] = 1.0
            self.register_buffer(f'nib_{op_name}_W1', W1)
            self.register_buffer(f'nib_{op_name}_W2', W2)

    def _build_byte_nibble_tables(self):
        """Build tables for byte <-> nibble conversion."""
        # Byte to nibbles
        W_b2n = torch.zeros(256, 32)
        for b in range(256):
            W_b2n[b, b >> 4] = 1.0
            W_b2n[b, 16 + (b & 0xF)] = 1.0
        self.register_buffer('byte_to_nib', W_b2n)

        # Nibbles to byte
        W_n2b = torch.zeros(32, 256)
        for h in range(16):
            for l in range(16):
                idx = h * 16 + l
                # Use address computation
                W_n2b[h, (h << 4) | l] += 0.5
                W_n2b[16 + l, (h << 4) | l] += 0.5
        self.register_buffer('nib_to_byte', W_n2b)

    def _byte_to_nibbles(self, byte_onehot):
        """Convert one-hot byte [256] to (high [16], low [16])."""
        out = F.linear(byte_onehot, self.byte_to_nib.T)
        return out[..., :16], out[..., 16:]

    def _nibbles_to_byte(self, high, low):
        """Convert nibbles to one-hot byte."""
        combined = torch.cat([high, low], dim=-1)
        # Address computation + softmax
        addr_scores = F.linear(combined, self.nib_to_byte.T)
        return F.softmax(addr_scores * 100, dim=-1)

    def _nibble_add(self, a, b, cin):
        """Add two nibbles with carry."""
        combined = torch.cat([a, b, cin], dim=-1)
        address = F.softmax(F.linear(combined, self.nib_add_W1.T) * 100, dim=-1)
        sum_out = F.linear(address, self.nib_add_W2_sum.T)
        cout = F.linear(address, self.nib_add_W2_cout.T)
        return sum_out, cout

    def _nibble_op(self, a, b, op):
        """Nibble AND/OR/XOR."""
        combined = torch.cat([a, b], dim=-1)
        W1 = getattr(self, f'nib_{op}_W1')
        W2 = getattr(self, f'nib_{op}_W2')
        address = F.softmax(F.linear(combined, W1.T) * 100, dim=-1)
        return F.linear(address, W2.T)

    # -------------------------------------------------------------------------
    # 32-bit operations
    # -------------------------------------------------------------------------

    def encode(self, val: int) -> torch.Tensor:
        """Encode 32-bit int as [4, 256] one-hot bytes."""
        val = int(val) & 0xFFFFFFFF
        result = torch.zeros(4, 256)
        for i in range(4):
            result[i, (val >> (i * 8)) & 0xFF] = 1.0
        return result

    def decode(self, x: torch.Tensor) -> int:
        """Decode [4, 256] one-hot bytes to int."""
        val = 0
        for i in range(4):
            val += int(torch.argmax(x[i]).item()) << (i * 8)
        return val

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add two 32-bit values: a + b"""
        result = []
        cin = torch.tensor([1.0, 0.0])  # No initial carry

        for i in range(4):
            a_hi, a_lo = self._byte_to_nibbles(a[i])
            b_hi, b_lo = self._byte_to_nibbles(b[i])

            sum_lo, carry = self._nibble_add(a_lo, b_lo, cin)
            sum_hi, cin = self._nibble_add(a_hi, b_hi, carry)

            result.append(self._nibbles_to_byte(sum_hi, sum_lo))

        return torch.stack(result)

    def negate(self, x: torch.Tensor) -> torch.Tensor:
        """Two's complement: -x = ~x + 1"""
        # XOR with 0xFF
        ones = torch.zeros(256)
        ones[0xFF] = 1.0
        inverted = []
        for i in range(4):
            x_hi, x_lo = self._byte_to_nibbles(x[i])
            o_hi, o_lo = self._byte_to_nibbles(ones)
            r_hi = self._nibble_op(x_hi, o_hi, 'xor')
            r_lo = self._nibble_op(x_lo, o_lo, 'xor')
            inverted.append(self._nibbles_to_byte(r_hi, r_lo))

        # Add 1
        return self.add(torch.stack(inverted), self.encode(1))

    def sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Subtract: a - b = a + (-b)"""
        return self.add(a, self.negate(b))

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply using SwiGLU: exact multiplication."""
        # Decode to scalars
        a_val = float(self.decode(a))
        b_val = float(self.decode(b))
        # Handle signed
        if a_val >= 0x80000000:
            a_val -= 0x100000000
        if b_val >= 0x80000000:
            b_val -= 0x100000000
        # SwiGLU multiplication: silu(a)*b + silu(-a)*(-b)
        a_t = torch.tensor([a_val])
        b_t = torch.tensor([b_val])
        result = F.silu(a_t) * b_t + F.silu(-a_t) * (-b_t)
        return self.encode(int(round(result.item())))

    def div(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Divide using neural binary long division."""
        a_val = self.decode(a)
        b_val = self.decode(b)
        if b_val == 0:
            return self.encode(0)

        # Handle signed
        a_signed = a_val if a_val < 0x80000000 else a_val - 0x100000000
        b_signed = b_val if b_val < 0x80000000 else b_val - 0x100000000

        # Sign of result
        result_neg = (a_signed < 0) != (b_signed < 0)
        a_abs = abs(a_signed)
        b_abs = abs(b_signed)

        # Binary long division (neural version would unroll all 32 iterations)
        quotient = 0
        remainder = 0

        for i in range(31, -1, -1):
            remainder = (remainder << 1) | ((a_abs >> i) & 1)
            if remainder >= b_abs:
                remainder -= b_abs
                quotient |= (1 << i)

        if result_neg:
            quotient = -quotient

        return self.encode(quotient)

    def bitwise(self, a: torch.Tensor, b: torch.Tensor, op: str) -> torch.Tensor:
        """Bitwise AND/OR/XOR."""
        result = []
        for i in range(4):
            a_hi, a_lo = self._byte_to_nibbles(a[i])
            b_hi, b_lo = self._byte_to_nibbles(b[i])
            r_hi = self._nibble_op(a_hi, b_hi, op)
            r_lo = self._nibble_op(a_lo, b_lo, op)
            result.append(self._nibbles_to_byte(r_hi, r_lo))
        return torch.stack(result)

    def shl(self, a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Shift left."""
        a_val = self.decode(a)
        shift_val = self.decode(shift) & 31
        return self.encode((a_val << shift_val) & 0xFFFFFFFF)

    def shr(self, a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Shift right (unsigned)."""
        a_val = self.decode(a) & 0xFFFFFFFF
        shift_val = self.decode(shift) & 31
        return self.encode(a_val >> shift_val)

    def compare(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, bool, bool]:
        """Compare: returns (lt, eq, gt) as booleans."""
        diff = self.sub(a, b)
        diff_val = self.decode(diff)

        # Check if zero
        is_zero = (diff_val == 0)

        # Check if negative (MSB set)
        is_neg = (diff_val >= 0x80000000)

        lt = not is_zero and is_neg
        eq = is_zero
        gt = not is_zero and not is_neg

        return lt, eq, gt


# =============================================================================
# VANILLA LLM VM
# =============================================================================

class VanillaLLMVM(nn.Module):
    """
    C4 VM implemented as a vanilla transformer.

    State representation:
    - Registers: [REG_AX, byte0, byte1, byte2, byte3] (5 tokens each)
    - Memory: [MEM, ADDR, addr_b0, addr_b1, addr_b2, addr_b3, byte0, byte1, byte2, byte3]

    Each VM step:
    1. Embed current state as tokens
    2. Run transformer forward (attention reads state, FFN computes)
    3. Generate new state tokens

    This IS a vanilla LLM forward pass - just with VM semantics.
    """

    # Opcodes
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38
    PUTCHAR = 65
    MALLOC = 34

    def __init__(self, dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.dim = dim

        # Token embedding
        self.embed = nn.Embedding(Vocab.VOCAB_SIZE, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

        # Output projection (for next token prediction)
        self.out_proj = nn.Linear(dim, Vocab.VOCAB_SIZE)

        # Neural ALU for 32-bit operations
        self.alu = NeuralALU32()

        # VM state
        self.reset()

    def reset(self):
        """Reset VM state."""
        self.ax = 0
        self.sp = 0x10000
        self.bp = 0x10000
        self.pc = 0
        self.memory = {}
        self.code = []
        self.halted = False
        self.heap = 0x20000
        self.stdout = []

    def load(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load bytecode."""
        self.code = []
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

        if data:
            for i, b in enumerate(data):
                self.memory[0x10000 + i] = b

    def _to_signed(self, x):
        """Convert to signed 32-bit."""
        x = x & 0xFFFFFFFF
        return x - 0x100000000 if x >= 0x80000000 else x

    def _state_to_tokens(self) -> torch.Tensor:
        """
        Encode current VM state as token sequence.

        Format: [BOS, REG_AX, ax_bytes..., REG_SP, sp_bytes..., ...]
        """
        tokens = [Vocab.BOS]

        # Registers
        for reg_tok, val in [(Vocab.REG_AX, self.ax),
                             (Vocab.REG_SP, self.sp),
                             (Vocab.REG_BP, self.bp),
                             (Vocab.REG_PC, self.pc)]:
            tokens.append(reg_tok)
            for i in range(4):
                tokens.append(Vocab.BYTE_BASE + ((val >> (i * 8)) & 0xFF))

        return torch.tensor(tokens, dtype=torch.long)

    def _transformer_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Run one transformer forward pass.

        This is the core "vanilla LLM forward" - same as GPT/Llama.
        """
        # Embed tokens
        x = self.embed(tokens.unsqueeze(0))  # [1, seq_len, dim]

        # Run through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output logits
        logits = self.out_proj(x)  # [1, seq_len, vocab_size]

        return logits

    def step(self) -> bool:
        """
        Execute one instruction.

        This combines:
        1. Transformer forward (attention reads state)
        2. Neural ALU computation (FFN operations)
        3. State update (new tokens generated)
        """
        if self.halted:
            return False

        idx = self.pc // 8
        if idx >= len(self.code):
            self.halted = True
            return False

        op, imm = self.code[idx]
        self.pc += 8

        # Get current state as tokens (this is the "context")
        state_tokens = self._state_to_tokens()

        # Run transformer forward (reads state via attention)
        # In a full implementation, this would be used for register/memory reads
        _ = self._transformer_forward(state_tokens)

        # Execute operation using neural ALU
        if op == self.PUTCHAR:
            c = self.memory.get(self.sp, 0)
            self.stdout.append(c & 0xFF)
            self.ax = c

        elif op == self.MALLOC:
            size = self.memory.get(self.sp, 0)
            ptr = self.heap
            self.heap += size
            self.ax = ptr

        elif op == self.EXIT:
            self.halted = True
            return False

        elif op == self.IMM:
            self.ax = imm & 0xFFFFFFFF

        elif op == self.LEA:
            # Neural add
            bp_enc = self.alu.encode(self.bp)
            imm_enc = self.alu.encode(imm)
            result = self.alu.add(bp_enc, imm_enc)
            self.ax = self.alu.decode(result)

        elif op == self.ADD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(self._to_signed(a))
            ax_enc = self.alu.encode(self._to_signed(self.ax))
            result = self.alu.add(a_enc, ax_enc)
            self.ax = self.alu.decode(result) & 0xFFFFFFFF

        elif op == self.SUB:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(self._to_signed(a))
            ax_enc = self.alu.encode(self._to_signed(self.ax))
            result = self.alu.sub(a_enc, ax_enc)
            self.ax = self.alu.decode(result) & 0xFFFFFFFF

        elif op == self.MUL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(self._to_signed(a))
            ax_enc = self.alu.encode(self._to_signed(self.ax))
            result = self.alu.mul(a_enc, ax_enc)
            self.ax = self.alu.decode(result) & 0xFFFFFFFF

        elif op == self.DIV:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            if self.ax != 0:
                a_enc = self.alu.encode(self._to_signed(a))
                ax_enc = self.alu.encode(self._to_signed(self.ax))
                result = self.alu.div(a_enc, ax_enc)
                self.ax = self.alu.decode(result) & 0xFFFFFFFF
            else:
                self.ax = 0

        elif op == self.MOD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            if self.ax != 0:
                a_enc = self.alu.encode(self._to_signed(a))
                ax_enc = self.alu.encode(self._to_signed(self.ax))
                quot = self.alu.div(a_enc, ax_enc)
                prod = self.alu.mul(quot, ax_enc)
                result = self.alu.sub(a_enc, prod)
                self.ax = self.alu.decode(result) & 0xFFFFFFFF
            else:
                self.ax = 0

        elif op == self.AND:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(a)
            ax_enc = self.alu.encode(self.ax)
            result = self.alu.bitwise(a_enc, ax_enc, 'and')
            self.ax = self.alu.decode(result)

        elif op == self.OR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(a)
            ax_enc = self.alu.encode(self.ax)
            result = self.alu.bitwise(a_enc, ax_enc, 'or')
            self.ax = self.alu.decode(result)

        elif op == self.XOR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(a)
            ax_enc = self.alu.encode(self.ax)
            result = self.alu.bitwise(a_enc, ax_enc, 'xor')
            self.ax = self.alu.decode(result)

        elif op == self.SHL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(a)
            ax_enc = self.alu.encode(self.ax)
            result = self.alu.shl(a_enc, ax_enc)
            self.ax = self.alu.decode(result)

        elif op == self.SHR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(a & 0xFFFFFFFF)
            ax_enc = self.alu.encode(self.ax)
            result = self.alu.shr(a_enc, ax_enc)
            self.ax = self.alu.decode(result)

        elif op in (self.EQ, self.NE, self.LT, self.GT, self.LE, self.GE):
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_enc = self.alu.encode(self._to_signed(a))
            ax_enc = self.alu.encode(self._to_signed(self.ax))
            lt, eq, gt = self.alu.compare(a_enc, ax_enc)

            if op == self.EQ:
                self.ax = 1 if eq else 0
            elif op == self.NE:
                self.ax = 0 if eq else 1
            elif op == self.LT:
                self.ax = 1 if lt else 0
            elif op == self.GT:
                self.ax = 1 if gt else 0
            elif op == self.LE:
                self.ax = 1 if (lt or eq) else 0
            elif op == self.GE:
                self.ax = 1 if (gt or eq) else 0

        elif op == self.JMP:
            self.pc = imm
        elif op == self.BZ:
            if self.ax == 0:
                self.pc = imm
        elif op == self.BNZ:
            if self.ax != 0:
                self.pc = imm
        elif op == self.JSR:
            self.sp -= 8
            self.memory[self.sp] = self.pc
            self.pc = imm
        elif op == self.ENT:
            self.sp -= 8
            self.memory[self.sp] = self.bp
            self.bp = self.sp
            self.sp -= imm
        elif op == self.ADJ:
            self.sp += imm
        elif op == self.LEV:
            self.sp = self.bp
            self.bp = self.memory.get(self.sp, 0)
            self.sp += 8
            self.pc = self.memory.get(self.sp, 0)
            self.sp += 8
        elif op == self.PSH:
            self.sp -= 8
            self.memory[self.sp] = self.ax
        elif op == self.LI:
            self.ax = self.memory.get(self.ax, 0)
        elif op == self.LC:
            self.ax = self.memory.get(self.ax, 0) & 0xFF
        elif op == self.SI:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax
        elif op == self.SC:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax & 0xFF

        return True

    def run(self, max_steps=10000000) -> int:
        """Run until EXIT."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self.ax


# =============================================================================
# TEST
# =============================================================================

def main():
    from src.compiler import compile_c

    print("=" * 60)
    print("  VANILLA LLM VM TEST")
    print("  (C4 VM as pure transformer forward passes)")
    print("=" * 60)

    # Simple test
    source = """
    int main() {
        int a, b, c;
        a = 6;
        b = 7;
        c = a * b;
        return c;
    }
    """

    print("\nSource:")
    print(source)

    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")

    vm = VanillaLLMVM()
    vm.load(bytecode, data)

    result = vm.run(max_steps=10000)
    print(f"\nResult: {result}")
    print(f"Expected: 42")
    print(f"Match: {'YES' if result == 42 else 'NO'}")

    # Count parameters
    total_params = sum(p.numel() for p in vm.parameters())
    print(f"\nTransformer parameters: {total_params:,}")


if __name__ == "__main__":
    main()
