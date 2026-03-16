"""
Bitwise operations for Neural VM V7.

AND, OR, XOR - using bit extraction and combination.
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN


class E_BITS:
    """Additional slots for bit extraction (stored in TEMP region)."""
    BIT3_A = 40
    BIT2_A = 41
    BIT1_A = 42
    BIT0_A = 43
    BIT3_B = 44
    BIT2_B = 45
    BIT1_B = 46
    BIT0_B = 47


class ClearBitSlotsFFN(PureFFN):
    """Clear all bit extraction slots before extracting."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            slots = [E_BITS.BIT3_A, E_BITS.BIT2_A, E_BITS.BIT1_A, E_BITS.BIT0_A,
                     E_BITS.BIT3_B, E_BITS.BIT2_B, E_BITS.BIT1_B, E_BITS.BIT0_B]
            for i, slot in enumerate(slots):
                self.W_up[i, E.OP_START + self.opcode] = S
                self.W_gate[i, slot] = -1.0
                self.W_down[slot, i] = 1.0 / S


class ExtractBit3FFN(PureFFN):
    """
    Extract bit 3 (MSB) from nibbles A and B.
    bit3 = 1 if nibble >= 8, else 0.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Bit 3 of A: bounded_step(7, 8)
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 7.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT3_A, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 8.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT3_A, 1] = -1.0 / S

            # Bit 3 of B
            self.W_up[2, E.NIB_B] = S
            self.b_up[2] = -S * 7.0
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT3_B, 2] = 1.0 / S

            self.W_up[3, E.NIB_B] = S
            self.b_up[3] = -S * 8.0
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT3_B, 3] = -1.0 / S


class ExtractBit2FFN(PureFFN):
    """
    Extract bit 2 from nibbles A and B.
    bit2 = 1 if nibble in {4,5,6,7,12,13,14,15}.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=12)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For A: bounded steps at 4, 8, 12
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 3.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 4.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 1] = -1.0 / S

            self.W_up[2, E.NIB_A] = S
            self.b_up[2] = -S * 7.0
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 2] = -1.0 / S

            self.W_up[3, E.NIB_A] = S
            self.b_up[3] = -S * 8.0
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 3] = 1.0 / S

            self.W_up[4, E.NIB_A] = S
            self.b_up[4] = -S * 11.0
            self.W_gate[4, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 4] = 1.0 / S

            self.W_up[5, E.NIB_A] = S
            self.b_up[5] = -S * 12.0
            self.W_gate[5, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 5] = -1.0 / S

            # For B (same pattern)
            self.W_up[6, E.NIB_B] = S
            self.b_up[6] = -S * 3.0
            self.W_gate[6, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_B, 6] = 1.0 / S

            self.W_up[7, E.NIB_B] = S
            self.b_up[7] = -S * 4.0
            self.W_gate[7, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_B, 7] = -1.0 / S

            self.W_up[8, E.NIB_B] = S
            self.b_up[8] = -S * 7.0
            self.W_gate[8, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_B, 8] = -1.0 / S

            self.W_up[9, E.NIB_B] = S
            self.b_up[9] = -S * 8.0
            self.W_gate[9, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_B, 9] = 1.0 / S

            self.W_up[10, E.NIB_B] = S
            self.b_up[10] = -S * 11.0
            self.W_gate[10, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_B, 10] = 1.0 / S

            self.W_up[11, E.NIB_B] = S
            self.b_up[11] = -S * 12.0
            self.W_gate[11, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_B, 11] = -1.0 / S


class ExtractBit1FFN(PureFFN):
    """
    Extract bit 1 from nibbles A and B.
    bit1 = 1 if nibble in {2,3,6,7,10,11,14,15}.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=28)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For A: bounded steps at each toggle point
            row = 0
            for thresh, sign in [(1, 1), (2, -1), (3, -1), (4, 1),
                                  (5, 1), (6, -1), (7, -1), (8, 1),
                                  (9, 1), (10, -1), (11, -1), (12, 1),
                                  (13, 1), (14, -1)]:
                self.W_up[row, E.NIB_A] = S
                self.b_up[row] = -S * float(thresh)
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                self.W_down[E_BITS.BIT1_A, row] = float(sign) / S
                row += 1

            # For B (same pattern)
            for thresh, sign in [(1, 1), (2, -1), (3, -1), (4, 1),
                                  (5, 1), (6, -1), (7, -1), (8, 1),
                                  (9, 1), (10, -1), (11, -1), (12, 1),
                                  (13, 1), (14, -1)]:
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * float(thresh)
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                self.W_down[E_BITS.BIT1_B, row] = float(sign) / S
                row += 1


class ExtractBit0FFN(PureFFN):
    """
    Extract bit 0 (LSB) from nibbles A and B.
    bit0 = 1 if nibble is odd {1,3,5,7,9,11,13,15}.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=64)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            row = 0
            for i in range(16):
                threshold = float(i)
                self.W_up[row, E.NIB_A] = S
                self.b_up[row] = -S * threshold
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                if i % 2 == 0:
                    self.W_down[E_BITS.BIT0_A, row] = 1.0 / S
                else:
                    self.W_down[E_BITS.BIT0_A, row] = -1.0 / S
                row += 1

                self.W_up[row, E.NIB_A] = S
                self.b_up[row] = -S * (threshold + 1.0)
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                if i % 2 == 0:
                    self.W_down[E_BITS.BIT0_A, row] = -1.0 / S
                else:
                    self.W_down[E_BITS.BIT0_A, row] = 1.0 / S
                row += 1

            for i in range(16):
                threshold = float(i)
                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * threshold
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                if i % 2 == 0:
                    self.W_down[E_BITS.BIT0_B, row] = 1.0 / S
                else:
                    self.W_down[E_BITS.BIT0_B, row] = -1.0 / S
                row += 1

                self.W_up[row, E.NIB_B] = S
                self.b_up[row] = -S * (threshold + 1.0)
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                if i % 2 == 0:
                    self.W_down[E_BITS.BIT0_B, row] = -1.0 / S
                else:
                    self.W_down[E_BITS.BIT0_B, row] = 1.0 / S
                row += 1


class BitwiseAndCombineFFN(PureFFN):
    """
    Combine extracted bits using AND.
    result = 8*(bit3_a*bit3_b) + 4*(bit2_a*bit2_b) + 2*(bit1_a*bit1_b) + (bit0_a*bit0_b)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # bit3: contributes 8 when both set
            self.W_up[0, E_BITS.BIT3_A] = S
            self.W_gate[0, E_BITS.BIT3_B] = 1.0
            self.W_down[E.RESULT, 0] = 8.0 / S

            self.W_up[1, E_BITS.BIT3_A] = -S
            self.W_gate[1, E_BITS.BIT3_B] = -1.0
            self.W_down[E.RESULT, 1] = 8.0 / S

            # bit2: contributes 4
            self.W_up[2, E_BITS.BIT2_A] = S
            self.W_gate[2, E_BITS.BIT2_B] = 1.0
            self.W_down[E.RESULT, 2] = 4.0 / S

            self.W_up[3, E_BITS.BIT2_A] = -S
            self.W_gate[3, E_BITS.BIT2_B] = -1.0
            self.W_down[E.RESULT, 3] = 4.0 / S

            # bit1: contributes 2
            self.W_up[4, E_BITS.BIT1_A] = S
            self.W_gate[4, E_BITS.BIT1_B] = 1.0
            self.W_down[E.RESULT, 4] = 2.0 / S

            self.W_up[5, E_BITS.BIT1_A] = -S
            self.W_gate[5, E_BITS.BIT1_B] = -1.0
            self.W_down[E.RESULT, 5] = 2.0 / S

            # bit0: contributes 1
            self.W_up[6, E_BITS.BIT0_A] = S
            self.W_gate[6, E_BITS.BIT0_B] = 1.0
            self.W_down[E.RESULT, 6] = 1.0 / S

            self.W_up[7, E_BITS.BIT0_A] = -S
            self.W_gate[7, E_BITS.BIT0_B] = -1.0
            self.W_down[E.RESULT, 7] = 1.0 / S


class BitwiseOrCombineFFN(PureFFN):
    """
    Combine extracted bits using OR.
    OR(a,b) = a + b - a*b
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            for bit_idx, (bit_a, bit_b, weight) in enumerate([
                (E_BITS.BIT3_A, E_BITS.BIT3_B, 8.0),
                (E_BITS.BIT2_A, E_BITS.BIT2_B, 4.0),
                (E_BITS.BIT1_A, E_BITS.BIT1_B, 2.0),
                (E_BITS.BIT0_A, E_BITS.BIT0_B, 1.0),
            ]):
                row = bit_idx * 4
                # Add bit_a
                self.W_up[row, E.OP_START + Opcode.OR] = S
                self.W_gate[row, bit_a] = 1.0
                self.W_down[E.RESULT, row] = weight / S

                # Add bit_b
                self.W_up[row + 1, E.OP_START + Opcode.OR] = S
                self.W_gate[row + 1, bit_b] = 1.0
                self.W_down[E.RESULT, row + 1] = weight / S

                # Subtract bit_a * bit_b
                self.W_up[row + 2, bit_a] = S
                self.W_gate[row + 2, bit_b] = 1.0
                self.W_down[E.RESULT, row + 2] = -weight / S

                self.W_up[row + 3, bit_a] = -S
                self.W_gate[row + 3, bit_b] = -1.0
                self.W_down[E.RESULT, row + 3] = -weight / S


class BitwiseXorCombineFFN(PureFFN):
    """
    Combine extracted bits using XOR.
    XOR(a,b) = a + b - 2*a*b
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            for bit_idx, (bit_a, bit_b, weight) in enumerate([
                (E_BITS.BIT3_A, E_BITS.BIT3_B, 8.0),
                (E_BITS.BIT2_A, E_BITS.BIT2_B, 4.0),
                (E_BITS.BIT1_A, E_BITS.BIT1_B, 2.0),
                (E_BITS.BIT0_A, E_BITS.BIT0_B, 1.0),
            ]):
                row = bit_idx * 4
                # Add bit_a
                self.W_up[row, E.OP_START + Opcode.XOR] = S
                self.W_gate[row, bit_a] = 1.0
                self.W_down[E.RESULT, row] = weight / S

                # Add bit_b
                self.W_up[row + 1, E.OP_START + Opcode.XOR] = S
                self.W_gate[row + 1, bit_b] = 1.0
                self.W_down[E.RESULT, row + 1] = weight / S

                # Subtract 2*bit_a * bit_b
                self.W_up[row + 2, bit_a] = S
                self.W_gate[row + 2, bit_b] = 1.0
                self.W_down[E.RESULT, row + 2] = -2.0 * weight / S

                self.W_up[row + 3, bit_a] = -S
                self.W_gate[row + 3, bit_b] = -1.0
                self.W_down[E.RESULT, row + 3] = -2.0 * weight / S


class ClearBitsFFN(PureFFN):
    """Clear extracted bit slots after bitwise ops."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            for i, slot in enumerate([E_BITS.BIT3_A, E_BITS.BIT2_A, E_BITS.BIT1_A, E_BITS.BIT0_A,
                                      E_BITS.BIT3_B, E_BITS.BIT2_B, E_BITS.BIT1_B, E_BITS.BIT0_B]):
                self.W_up[i, E.OP_START + self.opcode] = S
                self.W_gate[i, slot] = -1.0
                self.W_down[slot, i] = 1.0 / S
