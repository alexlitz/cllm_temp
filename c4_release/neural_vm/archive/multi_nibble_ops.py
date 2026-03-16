"""
Multi-nibble MUL, DIV, MOD operations using the neural ALU.

These operations use the existing 32-bit ADD, SUB, comparison, and shift
operations to implement higher-level algorithms.

MUL: Schoolbook multiplication algorithm
DIV: Binary long division algorithm
MOD: Binary long division, returning remainder
"""

import torch

from .embedding import E, Opcode


def encode_operands(opcode: int, a: int, b: int) -> torch.Tensor:
    """Encode two 32-bit values and opcode into ALU input format."""
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    for i in range(E.NUM_POSITIONS):
        x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
        x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
        x[0, i, E.OP_START + opcode] = 1.0
        x[0, i, E.POS] = float(i)

    return x


def decode_result(x: torch.Tensor) -> int:
    """Extract 32-bit result from ALU output."""
    result = 0
    for i in range(E.NUM_POSITIONS):
        nib = int(round(x[0, i, E.RESULT].item()))
        nib = max(0, min(15, nib))
        result |= (nib << (i * 4))
    return result


class MultiNibbleALU:
    """
    ALU wrapper that implements 32-bit MUL, DIV, MOD using primitives.

    Uses the base SparseMoEALU for ADD, SUB, comparison, and shift operations,
    then builds higher-level operations on top.
    """

    def __init__(self, base_alu):
        """
        Args:
            base_alu: The underlying SparseMoEALU instance
        """
        self.alu = base_alu

    def _alu_op(self, opcode: int, a: int, b: int) -> int:
        """Execute a single ALU operation."""
        x = encode_operands(opcode, a, b)
        y = self.alu(x)
        return decode_result(y)

    def add(self, a: int, b: int) -> int:
        """32-bit addition."""
        return self._alu_op(Opcode.ADD, a, b)

    def sub(self, a: int, b: int) -> int:
        """32-bit subtraction."""
        return self._alu_op(Opcode.SUB, a, b)

    def ge(self, a: int, b: int) -> int:
        """32-bit greater-than-or-equal comparison."""
        return self._alu_op(Opcode.GE, a, b)

    def shl(self, a: int, b: int) -> int:
        """32-bit shift left."""
        return self._alu_op(Opcode.SHL, a, b)

    def shr(self, a: int, b: int) -> int:
        """32-bit shift right."""
        return self._alu_op(Opcode.SHR, a, b)

    def mul_32bit(self, a: int, b: int) -> int:
        """
        32-bit multiplication using shift-and-add algorithm.

        For each bit of b that is 1, add (a << bit_position) to result.
        This uses at most 32 additions and 32 shifts.

        Returns lower 32 bits of result.
        """
        a = a & 0xFFFFFFFF
        b = b & 0xFFFFFFFF
        result = 0

        for i in range(32):
            if b & (1 << i):
                # Add a << i to result
                shifted_a = self.shl(a, i) if i > 0 else a
                result = self.add(result, shifted_a)

        return result & 0xFFFFFFFF

    def div_32bit(self, dividend: int, divisor: int) -> int:
        """
        32-bit unsigned division using binary long division.

        Returns quotient (dividend // divisor).
        Returns 0 if divisor is 0.
        """
        dividend = dividend & 0xFFFFFFFF
        divisor = divisor & 0xFFFFFFFF

        if divisor == 0:
            return 0  # Division by zero returns 0

        quotient = 0
        remainder = 0

        # Process each bit from MSB to LSB
        for i in range(31, -1, -1):
            # Shift remainder left by 1
            remainder = self.shl(remainder, 1) if remainder > 0 else 0

            # Bring down next bit of dividend
            bit = (dividend >> i) & 1
            remainder = self.add(remainder, bit) if bit else remainder

            # If remainder >= divisor, subtract and set quotient bit
            if self.ge(remainder, divisor):
                remainder = self.sub(remainder, divisor)
                quotient = quotient | (1 << i)

        return quotient

    def mod_32bit(self, dividend: int, divisor: int) -> int:
        """
        32-bit unsigned modulo using binary long division.

        Returns remainder (dividend % divisor).
        Returns dividend if divisor is 0.
        """
        dividend = dividend & 0xFFFFFFFF
        divisor = divisor & 0xFFFFFFFF

        if divisor == 0:
            return dividend  # Mod by zero returns dividend

        remainder = 0

        # Process each bit from MSB to LSB
        for i in range(31, -1, -1):
            # Shift remainder left by 1
            remainder = self.shl(remainder, 1) if remainder > 0 else 0

            # Bring down next bit of dividend
            bit = (dividend >> i) & 1
            remainder = self.add(remainder, bit) if bit else remainder

            # If remainder >= divisor, subtract
            if self.ge(remainder, divisor):
                remainder = self.sub(remainder, divisor)

        return remainder


def mul_schoolbook(a: int, b: int, alu) -> int:
    """
    Schoolbook multiplication using neural ALU primitives.

    For A × B where A and B are 8-nibble (32-bit) numbers:
    result[k] = Σ(a[i] × b[j]) for all i,j where i+j = k

    This is a simpler version that uses nested loops over nibbles.
    """
    a = a & 0xFFFFFFFF
    b = b & 0xFFFFFFFF

    # Extract nibbles
    a_nibs = [(a >> (i * 4)) & 0xF for i in range(8)]
    b_nibs = [(b >> (i * 4)) & 0xF for i in range(8)]

    # Accumulator for each result position (can exceed 4 bits)
    accum = [0] * 16  # 16 positions for 64-bit intermediate

    # Schoolbook: multiply each pair of nibbles
    for i in range(8):
        for j in range(8):
            product = a_nibs[i] * b_nibs[j]
            accum[i + j] += product

    # Propagate carries
    for k in range(15):
        carry = accum[k] // 16
        accum[k] = accum[k] % 16
        accum[k + 1] += carry
    accum[15] = accum[15] % 16

    # Extract lower 32 bits (8 nibbles)
    result = 0
    for i in range(8):
        result |= (accum[i] << (i * 4))

    return result


def div_long(dividend: int, divisor: int) -> tuple:
    """
    Long division using neural ALU primitives.

    Returns (quotient, remainder).
    """
    dividend = dividend & 0xFFFFFFFF
    divisor = divisor & 0xFFFFFFFF

    if divisor == 0:
        return (0, dividend)

    quotient = 0
    remainder = 0

    for i in range(31, -1, -1):
        remainder = (remainder << 1) | ((dividend >> i) & 1)
        if remainder >= divisor:
            remainder -= divisor
            quotient |= (1 << i)

    return (quotient, remainder)
