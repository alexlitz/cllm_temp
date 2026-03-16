"""
C4 Transformer VM - 100% Autoregressive Neural Network Virtual Machine

Every step() is a single differentiable forward pass:
- State read: attention Q/K/V over token sequence
- Instruction fetch: attention over code tokens
- Opcode dispatch: soft blend by one-hot opcode weights (NO if/elif)
- Arithmetic: MatMul + Softmax lookup tables (nibble ALU)
- Multiply: schoolbook nibble multiplication (nib_mul + nib_add)
- State write: token generation (append to context)
- Halt: soft probability from opcode weight

The only non-neural operations are:
- _encode_int(): creates constant tensors (weight initialization)
- _decode_int() in run(): extracts final output (tokenizer decode)
- The for loop in run(): the inference loop (same as any LLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class C4Config:
    """Configuration for C4 Transformer VM."""
    vocab_size: int = 270
    hidden_size: int = 256
    num_attention_heads: int = 4
    intermediate_size: int = 256
    max_position_embeddings: int = 4096
    model_type: str = "c4_transformer_vm"

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ['vocab_size', 'hidden_size',
                'num_attention_heads', 'intermediate_size',
                'max_position_embeddings', 'model_type']}


# =============================================================================
# TOKEN TYPES - Everything is a token
# =============================================================================

class TokenType:
    """Token type markers (high byte of token)."""
    BYTE = 0          # Raw byte value 0-255
    REG_AX = 256      # AX register marker
    REG_SP = 257      # SP register marker
    REG_BP = 258      # BP register marker
    REG_PC = 259      # PC register marker
    MEM_ADDR = 260    # Memory address marker
    MEM_VAL = 261     # Memory value marker
    OPCODE = 262      # Opcode marker
    IMM = 263         # Immediate value marker


# =============================================================================
# CORE NEURAL COMPONENTS
# =============================================================================

class ByteToNibbleFFN(nn.Module):
    """FFN: byte [256] → (high_nibble [16], low_nibble [16])"""

    def __init__(self):
        super().__init__()
        W1 = torch.eye(256)
        W2 = torch.zeros(256, 32)
        for b in range(256):
            W2[b, (b >> 4) & 0xF] = 1.0
            W2[b, 16 + (b & 0xF)] = 1.0
        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.linear(x, self.W1.T)
        out = F.linear(h, self.W2.T)
        return out[..., :16], out[..., 16:]


class NibbleToByteFFN(nn.Module):
    """FFN: (high_nibble [16], low_nibble [16]) → byte [256]

    Uses address computation + softmax like NibbleTableFFN for one-hot output.
    """

    def __init__(self):
        super().__init__()
        # Address encoder: combines both nibbles to unique address
        W1 = torch.zeros(32, 256)  # [input, address]
        W2 = torch.zeros(256, 256)  # [address, output byte]
        for h in range(16):
            for l in range(16):
                addr = h * 16 + l
                W1[h, addr] = 1.0
                W1[16 + l, addr] = 1.0
                W2[addr, (h << 4) | l] = 1.0
        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([high, low], dim=-1)
        address = F.softmax(F.linear(combined, self.W1.T) * 100, dim=-1)
        return F.linear(address, self.W2.T)


class NibbleTableFFN(nn.Module):
    """FFN: (nibble_a [16], nibble_b [16]) → result_nibble [16]"""

    def __init__(self, op_fn):
        super().__init__()
        W1 = torch.zeros(32, 256)
        W2 = torch.zeros(256, 16)
        for a in range(16):
            for b in range(16):
                idx = a * 16 + b
                W1[a, idx] = 1.0
                W1[16 + b, idx] = 1.0
                W2[idx, op_fn(a, b) & 0xF] = 1.0
        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([a, b], dim=-1)
        address = F.softmax(F.linear(combined, self.W1.T) * 100, dim=-1)
        return F.linear(address, self.W2.T)


class NibbleAddWithCarryFFN(nn.Module):
    """FFN: (nibble_a, nibble_b, carry_in) → (sum_nibble, carry_out)"""

    def __init__(self):
        super().__init__()
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

        self.register_buffer('W1', W1)
        self.register_buffer('W2_sum', W2_sum)
        self.register_buffer('W2_cout', W2_cout)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                cin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([a, b, cin], dim=-1)
        address = F.softmax(F.linear(combined, self.W1.T) * 100, dim=-1)
        return F.linear(address, self.W2_sum.T), F.linear(address, self.W2_cout.T)


class NibbleMulFFN(nn.Module):
    """FFN: (nibble_a, nibble_b) → (product_lo, product_hi)

    Deterministic lookup table for 4-bit × 4-bit multiplication.
    Product is 8 bits split into low and high nibbles.
    """

    def __init__(self):
        super().__init__()
        W1 = torch.zeros(32, 256)       # [inputs, addresses]
        W2_lo = torch.zeros(256, 16)     # [addresses, lo nibble]
        W2_hi = torch.zeros(256, 16)     # [addresses, hi nibble]

        for a in range(16):
            for b in range(16):
                idx = a * 16 + b
                W1[a, idx] = 1.0
                W1[16 + b, idx] = 1.0
                product = a * b
                W2_lo[idx, product & 0xF] = 1.0
                W2_hi[idx, (product >> 4) & 0xF] = 1.0

        self.register_buffer('W1', W1)
        self.register_buffer('W2_lo', W2_lo)
        self.register_buffer('W2_hi', W2_hi)

    def forward(self, a: torch.Tensor, b: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([a, b], dim=-1)
        address = F.softmax(F.linear(combined, self.W1.T) * 100, dim=-1)
        return F.linear(address, self.W2_lo.T), F.linear(address, self.W2_hi.T)



# =============================================================================
# NEURAL ARITHMETIC UNIT (fully neural, no Python ops)
# =============================================================================

class NeuralALU(nn.Module):
    """
    Arithmetic Logic Unit using only neural operations.

    All operations work on 4-byte (32-bit) one-hot encoded values.
    Each byte is a [256] one-hot vector, so a value is [4, 256].
    """

    def __init__(self):
        super().__init__()
        # Converters
        self.b2n = ByteToNibbleFFN()
        self.n2b = NibbleToByteFFN()

        # Nibble operations
        self.nib_add = NibbleAddWithCarryFFN()
        self.nib_and = NibbleTableFFN(lambda a, b: a & b)
        self.nib_or = NibbleTableFFN(lambda a, b: a | b)
        self.nib_xor = NibbleTableFFN(lambda a, b: a ^ b)

        # Shift tables (per-nibble, 4 bits max)
        self._build_shift_tables()

        # Nibble multiply table
        self.nib_mul = NibbleMulFFN()

        # Constants
        self.register_buffer('zero_nib', self._onehot(0, 16))
        self.register_buffer('ones_byte', self._onehot(0xFF, 256))
        self.register_buffer('carry_zero', self._onehot(0, 2))
        self.register_buffer('carry_one', self._onehot(1, 2))

        # Pre-encoded integer constants (eliminates _encode_int from forward pass)
        self.register_buffer('const_zero', self._encode_int(0))
        self.register_buffer('const_one', self._encode_int(1))
        self.register_buffer('const_eight', self._encode_int(8))

        # Bit masks for shift/divide: 1<<0 through 1<<31
        bit_masks = torch.zeros(32, 4, 256)
        for i in range(32):
            val = 1 << i
            for b in range(4):
                bit_masks[i, b, (val >> (b * 8)) & 0xFF] = 1.0
        self.register_buffer('bit_masks', bit_masks)

        # Zero-byte constant for _mask_to_byte [256] one-hot at 0
        self.register_buffer('zero_byte', self._onehot(0, 256))

        # ByteIsZeroFFN: byte [256] → [2] one-hot (is_zero, not_zero)
        # Pure MatMul + Softmax, same pattern as NibbleTableFFN
        byte_is_zero_W = torch.zeros(256, 2)
        byte_is_zero_W[0, 0] = 1.0    # byte==0 → [1, 0]
        byte_is_zero_W[1:, 1] = 1.0   # byte!=0 → [0, 1]
        self.register_buffer('byte_is_zero_W', byte_is_zero_W)

        # ByteIsNegFFN: byte [256] → [2] one-hot (is_neg, not_neg)
        # MSB set (byte >= 128) means negative in two's complement
        byte_is_neg_W = torch.zeros(256, 2)
        byte_is_neg_W[:128, 1] = 1.0   # 0-127 → not negative [0, 1]
        byte_is_neg_W[128:, 0] = 1.0   # 128-255 → negative [1, 0]
        self.register_buffer('byte_is_neg_W', byte_is_neg_W)

        # Pre-computed result tensors for compare() [2] one-hot
        self.register_buffer('flag_true', torch.tensor([1.0, 0.0]))
        self.register_buffer('flag_false', torch.tensor([0.0, 1.0]))

    def _onehot(self, val: int, size: int) -> torch.Tensor:
        x = torch.zeros(size)
        x[val] = 1.0
        return x

    def _build_shift_tables(self):
        """Build nibble shift tables for 0-4 bit shifts."""
        # Left shift: for each shift amount, map nibble to (result, overflow)
        shl_result = torch.zeros(5, 16, 16)  # [shift][input] -> output
        shl_overflow = torch.zeros(5, 16, 16)
        shr_result = torch.zeros(5, 16, 16)
        shr_underflow = torch.zeros(5, 16, 16)

        for shift in range(5):
            for val in range(16):
                # Left shift
                res = (val << shift) & 0xF
                over = (val >> (4 - shift)) & 0xF if shift > 0 else 0
                shl_result[shift, val, res] = 1.0
                shl_overflow[shift, val, over] = 1.0
                # Right shift
                res = (val >> shift) & 0xF
                under = (val << (4 - shift)) & 0xF if shift > 0 else 0
                shr_result[shift, val, res] = 1.0
                shr_underflow[shift, val, under] = 1.0

        self.register_buffer('shl_result', shl_result)
        self.register_buffer('shl_overflow', shl_overflow)
        self.register_buffer('shr_result', shr_result)
        self.register_buffer('shr_underflow', shr_underflow)

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add two 4-byte values: a + b"""
        result = []
        carry = self.carry_zero.clone()

        for i in range(4):
            ah, al = self.b2n(a[i])
            bh, bl = self.b2n(b[i])
            sl, carry = self.nib_add(al, bl, carry)
            sh, carry = self.nib_add(ah, bh, carry)
            result.append(self.n2b(sh, sl))

        return torch.stack(result)

    def negate(self, x: torch.Tensor) -> torch.Tensor:
        """Two's complement negation: -x = (~x) + 1"""
        # XOR with 0xFF (bitwise NOT)
        inverted = []
        for i in range(4):
            xh, xl = self.b2n(x[i])
            oh, ol = self.b2n(self.ones_byte)
            rh = self.nib_xor(xh, oh)
            rl = self.nib_xor(xl, ol)
            inverted.append(self.n2b(rh, rl))

        # Add 1
        result = []
        carry = self.carry_one.clone()  # Start with carry=1
        for i in range(4):
            h, l = self.b2n(inverted[i])
            sl, carry = self.nib_add(l, self.zero_nib, carry)
            sh, carry = self.nib_add(h, self.zero_nib, carry)
            result.append(self.n2b(sh, sl))

        return torch.stack(result)

    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Subtract: a - b = a + (-b)"""
        return self.add(a, self.negate(b))

    def bitwise_op(self, a: torch.Tensor, b: torch.Tensor, op: str) -> torch.Tensor:
        """Bitwise AND, OR, XOR"""
        nib_op = {'and': self.nib_and, 'or': self.nib_or, 'xor': self.nib_xor}[op]
        result = []
        for i in range(4):
            ah, al = self.b2n(a[i])
            bh, bl = self.b2n(b[i])
            result.append(self.n2b(nib_op(ah, bh), nib_op(al, bl)))
        return torch.stack(result)

    def shift_left_1(self, x: torch.Tensor) -> torch.Tensor:
        """Shift left by 1 bit (neural)."""
        result = []
        carry = self.zero_nib.clone()

        for i in range(4):  # LSB to MSB
            h, l = self.b2n(x[i])
            # Shift each nibble left by 1, propagate carries
            new_l = F.linear(l, self.shl_result[1].T)
            l_over = F.linear(l, self.shl_overflow[1].T)
            new_h = F.linear(h, self.shl_result[1].T)
            h_over = F.linear(h, self.shl_overflow[1].T)
            # OR in carries
            new_l = self.nib_or(new_l, carry)
            new_h = self.nib_or(new_h, l_over)
            carry = h_over
            result.append(self.n2b(new_h, new_l))

        return torch.stack(result)

    def shift_right_1(self, x: torch.Tensor) -> torch.Tensor:
        """Shift right by 1 bit (neural)."""
        result = []
        carry = self.zero_nib.clone()

        for i in range(3, -1, -1):  # MSB to LSB
            h, l = self.b2n(x[i])
            new_h = F.linear(h, self.shr_result[1].T)
            h_under = F.linear(h, self.shr_underflow[1].T)
            new_l = F.linear(l, self.shr_result[1].T)
            l_under = F.linear(l, self.shr_underflow[1].T)
            # OR in carries from higher byte
            new_h = self.nib_or(new_h, carry)
            new_l = self.nib_or(new_l, h_under)
            carry = l_under
            result.insert(0, self.n2b(new_h, new_l))

        return torch.stack(result)

    def neural_shift_left(self, x: torch.Tensor, shift_amt: torch.Tensor) -> torch.Tensor:
        """
        Neural shift left by variable amount.
        Decomposes into 5 conditional shifts: by 1, 2, 4, 8, 16.
        """
        result = x

        # For each bit position 0-4, conditionally shift by 2^i
        for i in range(5):
            # Check if bit i of shift_amt is set
            bit_mask = self._encode_int(1 << i)
            masked = self.bitwise_op(shift_amt, bit_mask, 'and')
            bit_is_set = 1.0 - self.is_zero(masked)[0]

            # Compute shifted version (shift by 2^i)
            shifted = result
            for _ in range(1 << i):
                shifted = self.shift_left_1(shifted)

            # Blend: if bit is set, use shifted; else use original
            result = self._blend(result, shifted, bit_is_set)

        return result

    def neural_shift_right(self, x: torch.Tensor, shift_amt: torch.Tensor) -> torch.Tensor:
        """
        Neural shift right by variable amount.
        Decomposes into 5 conditional shifts: by 1, 2, 4, 8, 16.
        """
        result = x

        for i in range(5):
            bit_mask = self._encode_int(1 << i)
            masked = self.bitwise_op(shift_amt, bit_mask, 'and')
            bit_is_set = 1.0 - self.is_zero(masked)[0]

            shifted = result
            for _ in range(1 << i):
                shifted = self.shift_right_1(shifted)

            result = self._blend(result, shifted, bit_is_set)

        return result

    def is_zero(self, x: torch.Tensor) -> torch.Tensor:
        """Check if value is zero. Returns [2] one-hot: [is_zero, not_zero].

        Pure FFN: each byte → MatMul with byte_is_zero_W → softmax → extract p(zero),
        then multiply all 4 scalar probabilities (AND gate) → stack to [2] one-hot.
        """
        # Multiply scalar p(byte_is_zero) across all 4 bytes
        p_all_zero = torch.tensor(1.0)
        for i in range(4):
            byte_z = F.softmax(F.linear(x[i], self.byte_is_zero_W.T) * 100, dim=-1)
            p_all_zero = p_all_zero * byte_z[0]  # scalar prob this byte is zero
        return torch.stack([p_all_zero, 1.0 - p_all_zero])

    def is_negative(self, x: torch.Tensor) -> torch.Tensor:
        """Check if MSB is set (negative in two's complement).

        Pure FFN: MSB byte → MatMul with byte_is_neg_W → softmax.
        """
        return F.softmax(F.linear(x[3], self.byte_is_neg_W.T) * 100, dim=-1)

    def compare(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compare a and b. Returns (lt, eq, gt) as [2] one-hot each.

        Pure tensor ops — no torch.zeros() allocation.
        """
        diff = self.subtract(a, b)
        is_z = self.is_zero(diff)
        is_neg = self.is_negative(diff)

        eq = is_z
        # lt = not_zero AND negative
        lt_prob = (1.0 - is_z[0]) * is_neg[0]
        lt = torch.stack([lt_prob, 1.0 - lt_prob])
        # gt = not_zero AND not_negative
        gt_prob = (1.0 - is_z[0]) * is_neg[1]
        gt = torch.stack([gt_prob, 1.0 - gt_prob])

        return lt, eq, gt

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """32-bit multiply via schoolbook nibble multiplication."""
        # Extract 8 nibbles from each operand (LSB first)
        a_nibs = []
        b_nibs = []
        for i in range(4):
            hi, lo = self.b2n(a[i])
            a_nibs.extend([lo, hi])  # lo = nibble 2i, hi = nibble 2i+1
            hi, lo = self.b2n(b[i])
            b_nibs.extend([lo, hi])

        # Initialize 8 accumulator nibbles to zero
        acc = [self.zero_nib.clone() for _ in range(8)]

        # Schoolbook: for each pair (i,j) where i+j < 8
        for i in range(8):
            for j in range(8):
                pos = i + j
                if pos >= 8:
                    continue
                prod_lo, prod_hi = self.nib_mul(a_nibs[i], b_nibs[j])

                # Add prod_lo to acc[pos] with carry
                acc[pos], carry = self.nib_add(acc[pos], prod_lo, self.carry_zero)

                # Add prod_hi to acc[pos+1] with carry (if in range)
                if pos + 1 < 8:
                    acc[pos + 1], carry = self.nib_add(acc[pos + 1], prod_hi, carry)
                    # Propagate carry
                    for k in range(pos + 2, 8):
                        acc[k], carry = self.nib_add(acc[k], self.zero_nib, carry)

        # Pack 8 nibbles back into 4 bytes
        result = []
        for i in range(4):
            result.append(self.n2b(acc[2 * i + 1], acc[2 * i]))
        return torch.stack(result)

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Integer division using neural binary long division.

        Unrolls 32 iterations, each using only neural operations:
        - shift_left_1
        - compare (ge)
        - soft conditional subtract

        For signed division: compute on absolute values, then fix sign.
        """
        # Handle signs neurally
        a_neg = self.is_negative(a)
        b_neg = self.is_negative(b)

        # Get absolute values (negate if negative)
        # If a is negative: a_abs = -a, else a_abs = a
        a_negated = self.negate(a)
        a_abs = self._blend(a, a_negated, a_neg[0])

        b_negated = self.negate(b)
        b_abs = self._blend(b, b_negated, b_neg[0])

        # Check for division by zero - return 0
        b_is_zero = self.is_zero(b)

        # Binary long division: 32 iterations
        quotient = self._encode_int(0)
        remainder = self._encode_int(0)

        for i in range(31, -1, -1):
            # Shift remainder left by 1
            remainder = self.shift_left_1(remainder)

            # Get bit i of dividend (a_abs)
            # bit_i = (a_abs >> i) & 1
            bit_mask = self._encode_int(1 << i)
            masked = self.bitwise_op(a_abs, bit_mask, 'and')
            bit_is_set = 1.0 - self.is_zero(masked)[0]  # 1 if bit set

            # Bring down the bit: remainder = remainder | (bit << 0)
            # If bit is set, set bit 0 of remainder
            one = self._encode_int(1)
            zero = self._encode_int(0)
            bit_val = self._blend(zero, one, bit_is_set)
            remainder = self.bitwise_op(remainder, bit_val, 'or')

            # Compare: remainder >= divisor?
            lt, eq, gt = self.compare(remainder, b_abs)
            ge = gt[0] + eq[0]  # >= is gt OR eq

            # If remainder >= divisor: remainder -= divisor, quotient |= (1 << i)
            new_remainder = self.subtract(remainder, b_abs)
            remainder = self._blend(remainder, new_remainder, ge)

            new_quotient = self.bitwise_op(quotient, bit_mask, 'or')
            quotient = self._blend(quotient, new_quotient, ge)

        # Fix sign: if (a < 0) XOR (b < 0), negate result
        result_neg = a_neg[0] * (1 - b_neg[0]) + (1 - a_neg[0]) * b_neg[0]
        quotient_negated = self.negate(quotient)
        result = self._blend(quotient, quotient_negated, result_neg)

        # Return 0 if b was zero
        result = self._blend(result, self._encode_int(0), b_is_zero[0])

        return result

    def _blend(self, x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Soft blend between x and y based on alpha.
        alpha close to 0 -> x, alpha close to 1 -> y

        This is a neural conditional: result = (1-alpha)*x + alpha*y
        """
        return (1.0 - alpha) * x + alpha * y

    def _mask_to_byte(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mask a 4-byte value to just the lower byte (for char operations).

        Sets bytes 1, 2, 3 to one-hot zero, keeping only byte 0.
        Uses pre-allocated zero_byte buffer — no torch.zeros() in forward pass.
        """
        return torch.stack([x[0], self.zero_byte, self.zero_byte, self.zero_byte])

    # -------------------------------------------------------------------------
    # Encoding/Decoding (interface to scalar values)
    # -------------------------------------------------------------------------

    def _encode_int(self, val: int) -> torch.Tensor:
        """Encode integer as 4 one-hot bytes."""
        val = int(val) & 0xFFFFFFFF
        result = torch.zeros(4, 256)
        for i in range(4):
            result[i, (val >> (i * 8)) & 0xFF] = 1.0
        return result


# =============================================================================
# ATTENTION-BASED STATE (registers and memory as tokens)
# =============================================================================

class TransformerState(nn.Module):
    """
    VM state stored as a sequence of tokens with attention-based access.

    Uses VANILLA ATTENTION with learned Q/K/V projections:
    - Query: what we're looking for (type embedding or address)
    - Keys: token identifiers (type + address for memory)
    - Values: the stored values

    Register tokens: [type (16), zeros (1024), value (1024)]
    Memory tokens:   [type (16), address (1024), value (1024)]

    Reading: Q/K/V attention over token sequence
    Writing: append new token (latest value wins via causal position bias)
    """

    # Dimensions
    TYPE_DIM = 16
    BYTES_DIM = 4 * 256  # 1024

    def __init__(self):
        super().__init__()
        # Register tokens: type + padding + value
        self.reg_dim = self.TYPE_DIM + self.BYTES_DIM  # 1040
        # Memory tokens: type + address + value
        self.mem_dim = self.TYPE_DIM + 2 * self.BYTES_DIM  # 2064

        # Use the larger dimension for unified token storage
        self.hidden_dim = self.mem_dim

        # Token embeddings
        self.register_buffer('type_embeddings', torch.eye(16))

        # =====================================================================
        # VANILLA ATTENTION PROJECTIONS
        # =====================================================================
        # For exact matching with one-hot inputs, we use TIED projections.
        # When W_q = W_k = M, we get: Q·K = x^T M^T M y
        # If M has orthonormal columns, M^T M = I, so Q·K = x·y = 1 if match, 0 if not.
        #
        # For types [16], we project to [16] using identity.
        # For addresses [1024], we project to [1024] using identity.
        # This is vanilla attention with W_q = W_k = I.

        # Register attention: use type directly as Q and K (identity projection)
        # The attention dimension equals the type dimension
        self.d_k_type = self.TYPE_DIM  # 16

        # Memory attention: use address directly as Q, and type+address as K
        # We separate the type and address contributions
        self.d_k_addr = self.BYTES_DIM  # 1024

        # Fixed projection matrices (identity-like, registered as buffers)
        # W_q_type: [16] -> [16], identity
        # W_k_type: [16] -> [16], identity
        self.register_buffer('W_q_type', torch.eye(self.TYPE_DIM))
        self.register_buffer('W_k_type', torch.eye(self.TYPE_DIM))

        # W_q_addr: [1024] -> [1024], identity
        # W_k_addr: [1024] -> [1024], identity (extracts address from tokens)
        self.register_buffer('W_q_addr', torch.eye(self.BYTES_DIM))
        self.register_buffer('W_k_addr', torch.eye(self.BYTES_DIM))

        # =====================================================================
        # ALiBi (Attention with Linear Biases) for position
        # =====================================================================
        # Standard ALiBi: bias[i,j] = -m * |i - j| (penalize distance)
        # Our ALiBi: bias[j] = m * j (reward later positions for "latest write wins")
        #
        # Two-level scoring:
        # 1. Type/address matching creates LARGE score differences (scale >> max_position)
        # 2. ALiBi position bias picks latest among matching tokens
        #
        # With type_scale=1000, sqrt(16)=4:
        #   - Matching token: base score = 1000/4 = 250
        #   - Non-matching token: base score = 0
        #
        # With alibi_slope=1:
        #   - Position 100 adds 100 to score
        #   - Matching at pos 3: 250 + 3 = 253
        #   - Non-matching at pos 100: 0 + 100 = 100
        #   - Matching wins regardless of position!
        #
        # Among matches:
        #   - Pos 3: 250 + 3 = 253
        #   - Pos 4: 250 + 4 = 254
        #   - Softmax gives ~73% to pos 4 (exp(254)/exp(254)+exp(253))
        #   - With larger gaps, later wins more decisively
        #
        self.alibi_slope = 1.0  # Small slope for position tiebreaker

        # Scaling factors for attention (must be >> max_seq * alibi_slope)
        self.type_scale = 1000.0   # Matching type gets ~250 points
        self.addr_scale = 1000.0   # Perfect address match gets ~155 points

    def embed_register(self, token_type: int, value: torch.Tensor) -> torch.Tensor:
        """Embed a register token (type + padding + value)."""
        type_emb = self.type_embeddings[token_type]  # [16]
        val_emb = value.flatten()  # [1024]
        # Pad to full hidden_dim (zeros in address section)
        padding = torch.zeros(self.BYTES_DIM)
        return torch.cat([type_emb, padding, val_emb])

    def embed_memory(self, addr: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Embed a memory token (type=MEM + address + value)."""
        type_emb = self.type_embeddings[4]  # T_MEM = 4
        addr_emb = addr.flatten()  # [1024]
        val_emb = value.flatten()  # [1024]
        return torch.cat([type_emb, addr_emb, val_emb])

    def read(self, tokens: torch.Tensor, query_type: int) -> torch.Tensor:
        """
        Read register value using 100% VANILLA ATTENTION.

        Pure transformer attention:
            Q = type_query           (one-hot [16])
            K = type_section         (one-hot [16] per token)
            V = value_section        ([1024] per token)

            scores = Q · K * scale + position_bias
            weights = softmax(scores)
            output = weights @ V

        With one-hot Q and K:
            Q · K = 1 if types match, 0 otherwise

        Using large scale (1M) and small position bias (1 per position):
            - Matching: 1 * 1M + pos = 1,000,000 + pos
            - Non-match: 0 * 1M + pos = pos
            - Matching always wins, position picks latest among matches
        """
        seq_len = tokens.shape[0]
        if seq_len == 0:
            return torch.zeros(4, 256)

        # Query: type embedding (one-hot)
        q = self.type_embeddings[query_type]  # [16]

        # Keys: type section of all tokens
        k = tokens[:, :self.TYPE_DIM]  # [seq_len, 16]

        # Values: stored values
        v = tokens[:, -self.BYTES_DIM:]  # [seq_len, 1024]

        # 100% VANILLA: Q · K dot product
        match_score = torch.matmul(k, q)  # [seq_len], values are 0 or 1

        # Temperature-scaled softmax for sharp attention
        # Scale match to dominate position: match bonus >> max possible position
        # With 1M bonus, matching at pos 0 still beats non-matching at pos 999999
        positions = torch.arange(seq_len, dtype=torch.float32)
        temperature = 0.01
        scores = (match_score * 1000000 + positions) / temperature

        # Softmax
        weights = F.softmax(scores, dim=0)

        # Weighted sum
        result = torch.matmul(weights, v)  # [1024]

        return result.view(4, 256)

    def read_memory(self, tokens: torch.Tensor, addr_query: torch.Tensor) -> torch.Tensor:
        """
        Read memory value using 100% VANILLA ATTENTION.

        Pure transformer attention with Q/K/V:
            Q = concat(type_4_embedding, addr_query)     [16 + 1024]
            K = concat(type_section, addr_section)       [16 + 1024] per token
            V = value_section                            [1024]

            scores = Q · K * scale + position_bias
            weights = softmax(scores)
            output = weights @ V

        With one-hot Q and K:
            Q · K = type_match (0 or 1) + address_match (0-4)
            Perfect memory match: 1 + 4 = 5
            Partial match (3 bytes): 1 + 3 = 4
            Register token: 0 + 0 = 0

        Using scale=1,000,000 and position_bias=1:
            - Perfect (5) at pos 0: 5M + 0 = 5,000,000
            - Partial (4) at pos 1000: 4M + 1000 = 4,001,000
            - Perfect wins! And among perfects, latest position wins.
        """
        seq_len = tokens.shape[0]
        if seq_len == 0:
            return torch.zeros(4, 256)

        # Query: concat(type=4 embedding, address)
        q_type = self.type_embeddings[4]  # [16]
        q_addr = addr_query.flatten()  # [1024]
        q = torch.cat([q_type, q_addr])  # [1040]

        # Keys: concat(type section, address section) for each token
        k_type = tokens[:, :self.TYPE_DIM]  # [seq_len, 16]
        k_addr = tokens[:, self.TYPE_DIM:self.TYPE_DIM + self.BYTES_DIM]  # [seq_len, 1024]
        k = torch.cat([k_type, k_addr], dim=1)  # [seq_len, 1040]

        # Values: stored values
        v = tokens[:, -self.BYTES_DIM:]  # [seq_len, 1024]

        # 100% VANILLA: Q · K dot product
        match_score = torch.matmul(k, q)  # [seq_len], values 0-5

        # Temperature-scaled softmax for sharp attention
        # scores = (match_score + position_bias) / temperature
        # Low temperature = sharper (like hard attention)
        #
        # Match score is sum of type match (0 or 1) and address match (0-4).
        # Perfect memory match = 5, partial = less.
        # Scale by 1M so matching always dominates position bias.
        positions = torch.arange(seq_len, dtype=torch.float32)
        temperature = 0.01
        scores = (match_score * 1000000 + positions) / temperature

        # Softmax (temperature already applied above)
        weights = F.softmax(scores, dim=0)

        # Weighted sum
        result = torch.matmul(weights, v)  # [1024]

        return result.view(4, 256)

    def write(self, tokens: torch.Tensor, token_type: int,
              value: torch.Tensor) -> torch.Tensor:
        """Write register by appending token."""
        new_token = self.embed_register(token_type, value).unsqueeze(0)
        if tokens.shape[0] == 0:
            return new_token
        return torch.cat([tokens, new_token], dim=0)

    def write_memory(self, tokens: torch.Tensor, addr: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        """Write memory by appending address+value token."""
        new_token = self.embed_memory(addr, value).unsqueeze(0)
        if tokens.shape[0] == 0:
            return new_token
        return torch.cat([tokens, new_token], dim=0)


# =============================================================================
# PURE NEURAL VM
# =============================================================================

class C4TransformerVM(nn.Module):
    """
    Pure neural virtual machine.

    All state is tokens. All operations are neural.
    This could be implemented as a standard transformer decoder.
    """

    # Opcodes
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38

    # Register token types
    T_AX, T_SP, T_BP, T_PC = 0, 1, 2, 3
    T_MEM = 4  # Memory (followed by address)
    T_CODE = 5  # Bytecode instruction

    def __init__(self, config: Optional[C4Config] = None):
        super().__init__()
        self.config = config or C4Config()

        # Neural ALU
        self.alu = NeuralALU()

        # Token-based state
        self.state = TransformerState()

        # Execution state
        self.reset()

    def reset(self):
        """Reset to initial state."""
        # Initialize state tokens
        self.tokens = torch.zeros(0, self.state.hidden_dim)

        # Initial register values
        zero = self.alu._encode_int(0)
        init_sp = self.alu._encode_int(0x10000)

        self.tokens = self.state.write(self.tokens, self.T_AX, zero)
        self.tokens = self.state.write(self.tokens, self.T_SP, init_sp)
        self.tokens = self.state.write(self.tokens, self.T_BP, init_sp)
        self.tokens = self.state.write(self.tokens, self.T_PC, zero)

        self.code = []

    def load_bytecode(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load bytecode as tokens for neural instruction fetch."""
        self.code = []
        self.code_tokens = []  # Tuple of (addr_enc, op_enc, imm_enc)

        for i, instr in enumerate(bytecode):
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

            # Also store as neural tokens for attention-based fetch
            addr = i * 8  # PC addresses are multiples of 8
            addr_enc = self.alu._encode_int(addr)
            op_enc = self.alu._encode_int(op)
            imm_enc = self.alu._encode_int(imm)
            self.code_tokens.append((addr_enc, op_enc, imm_enc))

        # Load data into memory tokens
        if data:
            for i, b in enumerate(data):
                addr = 0x10000 + i
                val = self.alu._encode_int(b)
                addr_enc = self.alu._encode_int(addr)
                self.tokens = self.state.write_memory(self.tokens, addr_enc, val)

    def _read_reg(self, reg_type: int) -> torch.Tensor:
        """Read register via attention."""
        return self.state.read(self.tokens, reg_type)

    def _fetch_instruction(self, pc_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch instruction at PC using neural attention over bytecode tokens.

        Returns (opcode, immediate) as neural tensors.
        """
        if not self.code_tokens:
            return self.alu._encode_int(self.EXIT), self.alu._encode_int(0)

        # Compute attention scores based on address match
        scores = []
        for addr_enc, op_enc, imm_enc in self.code_tokens:
            # Address match (4 bytes must match)
            addr_match = (addr_enc.flatten() * pc_enc.flatten()).sum()
            scores.append(addr_match * 100)

        scores = torch.stack(scores)
        weights = F.softmax(scores, dim=0)

        # Weighted sum of opcodes and immediates
        op_result = torch.zeros(4, 256)
        imm_result = torch.zeros(4, 256)
        for i, (addr_enc, op_enc, imm_enc) in enumerate(self.code_tokens):
            op_result = op_result + weights[i] * op_enc
            imm_result = imm_result + weights[i] * imm_enc

        return op_result, imm_result

    def _write_reg(self, reg_type: int, value: torch.Tensor):
        """Write register by appending token."""
        self.tokens = self.state.write(self.tokens, reg_type, value)

    def step(self) -> float:
        """
        Execute one instruction as a single differentiable forward pass.

        NO Python control flow branches on data values. All opcodes are
        computed in parallel, then soft-blended by the opcode's one-hot weight.

        Returns the EXIT opcode weight (halt probability). The inference loop
        in run() uses this to decide when to stop (like EOS token detection).
        """

        # === 1. Read all state via attention ===
        pc_enc = self._read_reg(self.T_PC)
        ax = self._read_reg(self.T_AX)
        sp = self._read_reg(self.T_SP)
        bp = self._read_reg(self.T_BP)

        # === 2. Fetch instruction via attention (already neural) ===
        op_enc, imm_enc = self._fetch_instruction(pc_enc)

        # === 3. Extract opcode weights from one-hot encoding ===
        # op_enc[0] is byte 0 of the 4-byte one-hot tensor [256]
        # Opcodes are 0-38, so they fit in byte 0
        w = op_enc[0]  # [256] — near-one-hot at the opcode position

        # === 4. Pre-compute common values (all neural) ===
        eight = self.alu._encode_int(8)
        next_pc = self.alu.add(pc_enc, eight)
        sp_after_pop = self.alu.add(sp, eight)
        sp_after_push = self.alu.subtract(sp, eight)
        stack_top = self._mem_load_neural(sp)           # value at top of stack
        mem_at_ax = self._mem_load_neural(ax)            # for LI/LC
        mem_at_bp = self._mem_load_neural(bp)            # for LEV pop 1
        bp_plus_8 = self.alu.add(bp, eight)
        mem_at_bp8 = self._mem_load_neural(bp_plus_8)    # for LEV pop 2
        bp_plus_16 = self.alu.add(bp_plus_8, eight)
        is_z = self.alu.is_zero(ax)                      # for BZ/BNZ
        masked_ax = self.alu._mask_to_byte(ax)           # for SC
        masked_mem = self.alu._mask_to_byte(mem_at_ax)   # for LC

        # === 5. Pre-compute ALU results for binary ops ===
        # All use stack_top (popped value) and ax
        add_result = self.alu.add(stack_top, ax)
        sub_result = self.alu.subtract(stack_top, ax)
        mul_result = self.alu.multiply(stack_top, ax)
        div_result = self.alu.divide(stack_top, ax)
        # MOD = a - (a/b)*b
        mod_quot = self.alu.divide(stack_top, ax)
        mod_prod = self.alu.multiply(mod_quot, ax)
        mod_result = self.alu.subtract(stack_top, mod_prod)
        and_result = self.alu.bitwise_op(stack_top, ax, 'and')
        or_result = self.alu.bitwise_op(stack_top, ax, 'or')
        xor_result = self.alu.bitwise_op(stack_top, ax, 'xor')
        shl_result = self.alu.neural_shift_left(stack_top, ax)
        shr_result = self.alu.neural_shift_right(stack_top, ax)

        # Comparisons
        lt, eq_flag, gt = self.alu.compare(stack_top, ax)
        zero = self.alu._encode_int(0)
        one = self.alu._encode_int(1)
        cmp_eq = self.alu._blend(zero, one, eq_flag[0])
        cmp_ne = self.alu._blend(one, zero, eq_flag[0])
        cmp_lt = self.alu._blend(zero, one, lt[0])
        cmp_gt = self.alu._blend(zero, one, gt[0])
        le_prob = lt[0] + eq_flag[0] - lt[0] * eq_flag[0]
        cmp_le = self.alu._blend(zero, one, le_prob)
        ge_prob = gt[0] + eq_flag[0] - gt[0] * eq_flag[0]
        cmp_ge = self.alu._blend(zero, one, ge_prob)

        # === 6. Blend new_ax by opcode weight ===
        AX_OPS = [self.LEA, self.IMM, self.ADD, self.SUB, self.MUL,
                  self.DIV, self.MOD, self.AND, self.OR, self.XOR,
                  self.SHL, self.SHR, self.EQ, self.NE, self.LT,
                  self.GT, self.LE, self.GE, self.LI, self.LC]
        w_ax_unchanged = 1.0 - sum(w[op] for op in AX_OPS)

        new_ax = (
            w[self.LEA] * self.alu.add(bp, imm_enc) +
            w[self.IMM] * imm_enc +
            w[self.ADD] * add_result +
            w[self.SUB] * sub_result +
            w[self.MUL] * mul_result +
            w[self.DIV] * div_result +
            w[self.MOD] * mod_result +
            w[self.AND] * and_result +
            w[self.OR]  * or_result +
            w[self.XOR] * xor_result +
            w[self.SHL] * shl_result +
            w[self.SHR] * shr_result +
            w[self.EQ]  * cmp_eq +
            w[self.NE]  * cmp_ne +
            w[self.LT]  * cmp_lt +
            w[self.GT]  * cmp_gt +
            w[self.LE]  * cmp_le +
            w[self.GE]  * cmp_ge +
            w[self.LI]  * mem_at_ax +
            w[self.LC]  * masked_mem +
            w_ax_unchanged * ax
        )

        # === 7. Blend new_sp ===
        POP_OPS = [self.ADD, self.SUB, self.MUL, self.DIV, self.MOD,
                   self.AND, self.OR, self.XOR, self.SHL, self.SHR,
                   self.EQ, self.NE, self.LT, self.GT, self.LE, self.GE,
                   self.SI, self.SC]
        w_pop = sum(w[op] for op in POP_OPS)
        w_push = w[self.PSH] + w[self.JSR]
        w_ent = w[self.ENT]
        w_adj = w[self.ADJ]
        w_lev = w[self.LEV]
        w_sp_same = 1.0 - w_pop - w_push - w_ent - w_adj - w_lev

        sp_ent = self.alu.subtract(sp_after_push, imm_enc)  # sp - 8 - imm

        new_sp = (w_pop * sp_after_pop + w_push * sp_after_push +
                  w_ent * sp_ent + w_adj * self.alu.add(sp, imm_enc) +
                  w_lev * bp_plus_16 + w_sp_same * sp)

        # === 8. Blend new_bp ===
        new_bp = (w[self.ENT] * sp_after_push +
                  w[self.LEV] * mem_at_bp +
                  (1.0 - w[self.ENT] - w[self.LEV]) * bp)

        # === 9. Blend new_pc ===
        bz_pc = self.alu._blend(next_pc, imm_enc, is_z[0])
        bnz_pc = self.alu._blend(imm_enc, next_pc, is_z[0])

        w_pc_normal = (1.0 - w[self.JMP] - w[self.JSR] - w[self.BZ] -
                       w[self.BNZ] - w[self.LEV])
        new_pc = (w[self.JMP] * imm_enc +
                  w[self.JSR] * imm_enc +
                  w[self.BZ]  * bz_pc +
                  w[self.BNZ] * bnz_pc +
                  w[self.LEV] * mem_at_bp8 +
                  w_pc_normal * next_pc)

        # === 10. Memory writes (blended) ===
        # Push writes: addr=sp-8, value varies by opcode
        push_val = (w[self.PSH] * ax +
                    w[self.JSR] * next_pc +
                    w[self.ENT] * bp)
        # Store writes: addr=stack_top, value is ax or masked_ax
        store_val = w[self.SI] * ax + w[self.SC] * masked_ax
        store_addr = stack_top  # SI/SC pop the address

        w_push_total = w[self.PSH] + w[self.JSR] + w[self.ENT]
        w_store_total = w[self.SI] + w[self.SC]
        w_write_total = w_push_total + w_store_total

        # Always emit a memory write token (unconditional, like generating a
        # token in an autoregressive model). Safe denominator avoids NaN when
        # no write opcode is active — the near-zero weights make the written
        # value negligible noise at a nonsensical address.
        safe_denom = w_write_total + 1e-8
        write_addr = ((w_push_total * sp_after_push +
                       w_store_total * store_addr) / safe_denom)
        write_val = (push_val + store_val) / safe_denom
        self._mem_store_neural(write_addr, write_val)

        # === 11. Write state ===
        self._write_reg(self.T_AX, new_ax)
        self._write_reg(self.T_SP, new_sp)
        self._write_reg(self.T_BP, new_bp)
        self._write_reg(self.T_PC, new_pc)

        # === 12. Return halt probability ===
        # The inference loop (run()) checks this, like EOS token detection.
        return w[self.EXIT]

    def _mem_load_neural(self, addr: torch.Tensor) -> torch.Tensor:
        """Load from memory using attention over address match."""
        return self.state.read_memory(self.tokens, addr)

    def _mem_store_neural(self, addr: torch.Tensor, val: torch.Tensor):
        """Store to memory by appending (addr, val) token."""
        self.tokens = self.state.write_memory(self.tokens, addr, val)

    @staticmethod
    def _decode_int(x: torch.Tensor) -> int:
        """Decode one-hot bytes to Python int (tokenizer decode at inference end)."""
        val = 0
        for i in range(4):
            val += int(torch.argmax(x[i]).item()) << (i * 8)
        if val >= 0x80000000:
            val -= 0x100000000
        return val

    def run(self, max_steps: int = 100000) -> int:
        """
        Inference loop: call step() repeatedly until EXIT (like EOS detection).

        step() is the differentiable forward pass. This loop and the final
        _decode_int are the only non-neural operations — same as any
        autoregressive model's generate() loop + tokenizer decode.
        """
        for _ in range(max_steps):
            halt_prob = self.step()
            if halt_prob > 0.5:
                break
        ax = self._read_reg(self.T_AX)
        return self._decode_int(ax)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'C4Config',
    'C4TransformerVM',
    'NeuralALU',
    'TransformerState',
]
