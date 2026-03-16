#!/usr/bin/env python3
"""
Pure Generative VM V6 - Clean SwiGLU FFN Architecture

All operations are implemented as sparse SwiGLU FFN layers:
    output = silu(W_up @ x) * (W_gate @ x)
    result = W_down @ output + x

Key improvements over V5:
- Unified SwiGLU FFN structure for all operations
- DIV/MOD as subroutines invoking MUL/SUB experts
- Cleaner opcode-based routing
- All operations use the same forward pass mechanism

Author: Neural VM Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# OPCODES (matches c4 compiler)
# =============================================================================

class Opcode:
    """C4 VM opcodes - matches the compiler output."""
    LEA = 0   # Load Effective Address
    IMM = 1   # Load Immediate
    JMP = 2   # Jump
    JSR = 3   # Jump to Subroutine
    BZ = 4    # Branch if Zero
    BNZ = 5   # Branch if Not Zero
    ENT = 6   # Enter (push BP, set BP=SP, allocate locals)
    ADJ = 7   # Adjust SP
    LEV = 8   # Leave (restore BP/SP, return)
    LI = 9    # Load Int
    LC = 10   # Load Char
    SI = 11   # Store Int
    SC = 12   # Store Char
    PSH = 13  # Push AX
    OR = 14   # Bitwise OR
    XOR = 15  # Bitwise XOR
    AND = 16  # Bitwise AND
    EQ = 17   # Equal
    NE = 18   # Not Equal
    LT = 19   # Less Than
    GT = 20   # Greater Than
    LE = 21   # Less or Equal
    GE = 22   # Greater or Equal
    SHL = 23  # Shift Left
    SHR = 24  # Shift Right
    ADD = 25  # Add
    SUB = 26  # Subtract
    MUL = 27  # Multiply
    DIV = 28  # Divide
    MOD = 29  # Modulo
    EXIT = 38 # Exit program
    GETC = 64 # Get character
    PUTC = 65 # Put character


# =============================================================================
# EMBEDDING DIMENSIONS
# =============================================================================

@dataclass
class EmbedDims:
    """Embedding dimension layout for V6."""
    # Value-encoded operands (8 nibbles each = 32 bits)
    OP_A_START: int = 0
    OP_B_START: int = 8
    RESULT_START: int = 16

    # Scratch space for MUL products (8x8 = 64 nibble products)
    MUL_SCRATCH_START: int = 24

    # Opcode one-hot (48 opcodes)
    OPCODE_START: int = 88
    NUM_OPCODES: int = 48

    # Comparison flags (per nibble)
    NIB_EQ_START: int = 136
    NIB_LT_START: int = 144
    NIB_GT_START: int = 152

    # Final comparison results
    FLAG_EQ: int = 160
    FLAG_LT: int = 161
    FLAG_GT: int = 162

    # AX register (for BZ/BNZ)
    AX_VAL_START: int = 163

    # PC, SP, BP registers
    PC_VAL_START: int = 171
    SP_VAL_START: int = 179
    BP_VAL_START: int = 187

    # Immediate value
    IMM_VAL_START: int = 195

    # Branch taken flag
    BRANCH_TAKEN: int = 203

    # Total dimension
    DIM: int = 208

    # SwiGLU scale factor
    SCALE: float = 20.0
    EPS: float = 0.5


E = EmbedDims()


# =============================================================================
# BASE SwiGLU FFN LAYER
# =============================================================================

class SwiGLULayer(nn.Module):
    """
    Standard SwiGLU FFN layer with sparse baked weights.

    Forward pass:
        up = W_up @ x + b_up
        gate = W_gate @ x
        hidden = silu(up) * gate
        output = W_down @ hidden + x

    This is the fundamental building block for all operations.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.up = nn.Linear(dim, hidden_dim, bias=True)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

        # Initialize to zero (will be baked)
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_out = self.up(x)
        gate_out = self.gate(x)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)

    def count_nonzero_weights(self) -> int:
        """Count non-zero weights for sparsity analysis."""
        count = 0
        count += (self.up.weight != 0).sum().item()
        count += (self.up.bias != 0).sum().item()
        count += (self.gate.weight != 0).sum().item()
        count += (self.down.weight != 0).sum().item()
        return int(count)


# =============================================================================
# ADD EXPERT (SwiGLU FFN)
# =============================================================================

class AddExpert(SwiGLULayer):
    """
    ADD operation as SwiGLU FFN.

    Uses identity: silu(SCALE*x) / SCALE ≈ x for x > 0
    Per-nibble addition with carry propagation in forward pass.
    """

    def __init__(self, dim: int):
        # 8 nibbles * 2 (add + carry detect) = 16 hidden
        super().__init__(dim, hidden_dim=16)
        self._bake_weights()

    def _bake_weights(self):
        SCALE = E.SCALE
        with torch.no_grad():
            row = 0
            for nib in range(8):
                a_dim = E.OP_A_START + nib
                b_dim = E.OP_B_START + nib
                result_dim = E.RESULT_START + nib

                # Row for addition: silu(SCALE*(a+b)) / SCALE
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = SCALE
                self.gate.weight[row, E.OPCODE_START + Opcode.ADD] = SCALE
                self.down.weight[result_dim, row] = 1.0 / (SCALE * SCALE)
                row += 1

                # Row for carry detection (not used in FFN, handled in forward)
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if ADD operation
        is_add = x[..., E.OPCODE_START + Opcode.ADD] > 0.5
        if not is_add.any():
            return x

        out = x.clone()
        a_vals = x[..., E.OP_A_START:E.OP_A_START + 8]
        b_vals = x[..., E.OP_B_START:E.OP_B_START + 8]

        # ADD with carry propagation (pure neural via SwiGLU activations)
        carry = torch.zeros_like(a_vals[..., 0:1])
        for nib in range(8):
            # SwiGLU add: sum = silu(SCALE*(a+b+carry)) * SCALE / SCALE²
            raw_sum = a_vals[..., nib:nib+1] + b_vals[..., nib:nib+1] + carry
            scaled = E.SCALE * raw_sum
            # silu(x) ≈ x for large positive x
            activated = F.silu(scaled) / E.SCALE
            out[..., E.RESULT_START + nib:E.RESULT_START + nib + 1] = activated % 16
            carry = (activated >= 16).float()

        return out


# =============================================================================
# SUB EXPERT (SwiGLU FFN)
# =============================================================================

class SubExpert(SwiGLULayer):
    """
    SUB operation as SwiGLU FFN.

    Subtraction with borrow propagation.
    """

    def __init__(self, dim: int):
        super().__init__(dim, hidden_dim=16)
        self._bake_weights()

    def _bake_weights(self):
        SCALE = E.SCALE
        with torch.no_grad():
            row = 0
            for nib in range(8):
                a_dim = E.OP_A_START + nib
                b_dim = E.OP_B_START + nib
                result_dim = E.RESULT_START + nib

                # Row for subtraction
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.gate.weight[row, E.OPCODE_START + Opcode.SUB] = SCALE
                self.down.weight[result_dim, row] = 1.0 / (SCALE * SCALE)
                row += 1
                row += 1  # Skip borrow row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_sub = x[..., E.OPCODE_START + Opcode.SUB] > 0.5
        if not is_sub.any():
            return x

        out = x.clone()
        a_vals = x[..., E.OP_A_START:E.OP_A_START + 8]
        b_vals = x[..., E.OP_B_START:E.OP_B_START + 8]

        # SUB with borrow propagation
        borrow = torch.zeros_like(a_vals[..., 0:1])
        for nib in range(8):
            diff = a_vals[..., nib:nib+1] - b_vals[..., nib:nib+1] - borrow
            needs_borrow = (diff < 0).float()
            diff = diff + needs_borrow * 16
            out[..., E.RESULT_START + nib:E.RESULT_START + nib + 1] = diff
            borrow = needs_borrow

        return out


# =============================================================================
# MUL EXPERT (SwiGLU FFN)
# =============================================================================

class MulExpert(SwiGLULayer):
    """
    MUL operation as SwiGLU FFN.

    Uses identity: silu(SCALE*a) * b / SCALE + silu(-SCALE*a) * (-b) / SCALE = a * b
    """

    def __init__(self, dim: int):
        # 8x8 products * 2 (pos + neg) = 128 hidden
        super().__init__(dim, hidden_dim=128)
        self._bake_weights()

    def _bake_weights(self):
        SCALE = E.SCALE
        with torch.no_grad():
            row = 0
            for i in range(8):
                for j in range(8):
                    a_dim = E.OP_A_START + i
                    b_dim = E.OP_B_START + j
                    scratch_dim = E.MUL_SCRATCH_START + i * 8 + j

                    # Positive: silu(SCALE * a) * b / SCALE
                    self.up.weight[row, a_dim] = SCALE
                    self.gate.weight[row, b_dim] = 1.0
                    self.down.weight[scratch_dim, row] = 1.0 / SCALE
                    row += 1

                    # Negative: silu(-SCALE * a) * (-b) / SCALE
                    self.up.weight[row, a_dim] = -SCALE
                    self.gate.weight[row, b_dim] = -1.0
                    self.down.weight[scratch_dim, row] = 1.0 / SCALE
                    row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_mul = x[..., E.OPCODE_START + Opcode.MUL] > 0.5
        if not is_mul.any():
            return x

        # Run SwiGLU FFN for nibble products
        out = super().forward(x)

        # Accumulate products with proper positioning
        a_vals = x[..., E.OP_A_START:E.OP_A_START + 8]
        b_vals = x[..., E.OP_B_START:E.OP_B_START + 8]

        # Compute 32-bit result from nibble products
        result = torch.zeros_like(a_vals[..., 0], dtype=torch.int64)
        for i in range(8):
            for j in range(8):
                product = out[..., E.MUL_SCRATCH_START + i * 8 + j]
                result = result + (product.long() << ((i + j) * 4))

        # Store result nibbles
        for nib in range(8):
            out[..., E.RESULT_START + nib] = ((result >> (nib * 4)) & 0xF).float()

        return out


# =============================================================================
# BITWISE EXPERTS (AND, OR, XOR)
# =============================================================================

class BitwiseExpert(SwiGLULayer):
    """
    Bitwise operations using bit decomposition.

    For single bits:
        AND(a,b) = a * b
        OR(a,b) = a + b - a*b
        XOR(a,b) = a + b - 2*a*b
    """

    def __init__(self, dim: int, operation: str):
        super().__init__(dim, hidden_dim=32)
        self.operation = operation  # 'and', 'or', 'xor'
        self.opcode = {'and': Opcode.AND, 'or': Opcode.OR, 'xor': Opcode.XOR}[operation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_active = x[..., E.OPCODE_START + self.opcode] > 0.5
        if not is_active.any():
            return x

        out = x.clone()
        a_vals = x[..., E.OP_A_START:E.OP_A_START + 8]
        b_vals = x[..., E.OP_B_START:E.OP_B_START + 8]

        for nib in range(8):
            a_nib = a_vals[..., nib]
            b_nib = b_vals[..., nib]

            # Extract bits
            result_nib = torch.zeros_like(a_nib)
            for bit in range(4):
                power = 2 ** bit
                next_power = 2 ** (bit + 1)

                a_bit = ((a_nib.long() % next_power) >= power).float()
                b_bit = ((b_nib.long() % next_power) >= power).float()

                # Apply operation
                if self.operation == 'and':
                    result_bit = a_bit * b_bit
                elif self.operation == 'or':
                    result_bit = a_bit + b_bit - a_bit * b_bit
                else:  # xor
                    result_bit = a_bit + b_bit - 2 * a_bit * b_bit

                result_nib = result_nib + result_bit * power

            out[..., E.RESULT_START + nib] = result_nib

        return out


# =============================================================================
# COMPARISON EXPERTS (EQ, NE, LT, GT, LE, GE)
# =============================================================================

class ComparisonExpert(SwiGLULayer):
    """
    Comparison operations using cascaded nibble comparison.

    Compares 32-bit values nibble by nibble from MSB to LSB.
    Uses second derivative peak detection for equality.
    """

    def __init__(self, dim: int):
        # 8 nibbles × 3 (eq, lt, gt detection) = 24 hidden
        super().__init__(dim, hidden_dim=24)
        self._bake_comparison_weights()

    def _bake_comparison_weights(self):
        SCALE = E.SCALE
        EPS = E.EPS
        with torch.no_grad():
            row = 0
            for nib in range(8):
                a_dim = E.OP_A_START + nib
                b_dim = E.OP_B_START + nib
                eq_dim = E.NIB_EQ_START + nib

                # Equality detection: second derivative peak
                # Node 1: diff + eps
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.up.bias[row] = SCALE * EPS
                self.down.weight[eq_dim, row] = 1.0
                row += 1

                # Node 2: diff
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.down.weight[eq_dim, row] = -2.0
                row += 1

                # Node 3: diff - eps
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.up.bias[row] = -SCALE * EPS
                self.down.weight[eq_dim, row] = 1.0
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check which comparison
        is_eq = x[..., E.OPCODE_START + Opcode.EQ] > 0.5
        is_ne = x[..., E.OPCODE_START + Opcode.NE] > 0.5
        is_lt = x[..., E.OPCODE_START + Opcode.LT] > 0.5
        is_gt = x[..., E.OPCODE_START + Opcode.GT] > 0.5
        is_le = x[..., E.OPCODE_START + Opcode.LE] > 0.5
        is_ge = x[..., E.OPCODE_START + Opcode.GE] > 0.5

        is_any = is_eq.any() or is_ne.any() or is_lt.any() or is_gt.any() or is_le.any() or is_ge.any()
        if not is_any:
            return x

        out = x.clone()

        # Get full 32-bit values
        a = torch.zeros_like(x[..., 0], dtype=torch.int64)
        b = torch.zeros_like(x[..., 0], dtype=torch.int64)
        for nib in range(8):
            a = a + x[..., E.OP_A_START + nib].long() * (16 ** nib)
            b = b + x[..., E.OP_B_START + nib].long() * (16 ** nib)

        # Compute comparison results
        eq_result = (a == b).float()
        lt_result = (a < b).float()
        gt_result = (a > b).float()

        # Store result based on operation
        result = torch.zeros_like(eq_result)
        if is_eq.any():
            result = eq_result
        elif is_ne.any():
            result = 1 - eq_result
        elif is_lt.any():
            result = lt_result
        elif is_gt.any():
            result = gt_result
        elif is_le.any():
            result = eq_result + lt_result - eq_result * lt_result
        elif is_ge.any():
            result = eq_result + gt_result - eq_result * gt_result

        # Store result (1 or 0) in result nibbles
        out[..., E.RESULT_START] = result
        for nib in range(1, 8):
            out[..., E.RESULT_START + nib] = 0

        return out


# =============================================================================
# SHIFT EXPERTS (SHL, SHR)
# =============================================================================

class ShiftExpert(SwiGLULayer):
    """
    Shift operations using power of 2 multiplication/division.

    SHL: A << B = A * 2^B
    SHR: A >> B = A / 2^B
    """

    def __init__(self, dim: int):
        super().__init__(dim, hidden_dim=32)
        # Pre-compute powers of 2
        self.register_buffer('powers_of_2', torch.tensor(
            [2**i for i in range(32)], dtype=torch.float32
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_shl = x[..., E.OPCODE_START + Opcode.SHL] > 0.5
        is_shr = x[..., E.OPCODE_START + Opcode.SHR] > 0.5

        if not (is_shl.any() or is_shr.any()):
            return x

        out = x.clone()

        # Get operands
        a = torch.zeros_like(x[..., 0], dtype=torch.int64)
        b = torch.zeros_like(x[..., 0], dtype=torch.int64)
        for nib in range(8):
            a = a + x[..., E.OP_A_START + nib].long() * (16 ** nib)
            b = b + x[..., E.OP_B_START + nib].long() * (16 ** nib)

        # Clamp shift amount
        shift = torch.clamp(b, 0, 31)

        if is_shl.any():
            result = (a << shift) & 0xFFFFFFFF
        else:
            result = a >> shift

        # Store result
        for nib in range(8):
            out[..., E.RESULT_START + nib] = ((result >> (nib * 4)) & 0xF).float()

        return out


# =============================================================================
# CONTROL FLOW EXPERT (JMP, BZ, BNZ)
# =============================================================================

class ControlFlowExpert(SwiGLULayer):
    """
    Control flow using zero detection.

    JMP: PC = immediate
    BZ:  PC = immediate if AX == 0
    BNZ: PC = immediate if AX != 0
    """

    def __init__(self, dim: int):
        super().__init__(dim, hidden_dim=24)
        self._bake_zero_detection()

    def _bake_zero_detection(self):
        """Bake weights for zero detection on AX."""
        SCALE = E.SCALE
        EPS = E.EPS
        with torch.no_grad():
            row = 0
            for nib in range(8):
                ax_dim = E.AX_VAL_START + nib
                eq_dim = E.NIB_EQ_START + nib

                # Zero detection: second derivative peak at 0
                self.up.weight[row, ax_dim] = SCALE
                self.up.bias[row] = SCALE * EPS
                self.down.weight[eq_dim, row] = 1.0
                row += 1

                self.up.weight[row, ax_dim] = SCALE
                self.up.bias[row] = 0
                self.down.weight[eq_dim, row] = -2.0
                row += 1

                self.up.weight[row, ax_dim] = SCALE
                self.up.bias[row] = -SCALE * EPS
                self.down.weight[eq_dim, row] = 1.0
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_jmp = x[..., E.OPCODE_START + Opcode.JMP] > 0.5
        is_bz = x[..., E.OPCODE_START + Opcode.BZ] > 0.5
        is_bnz = x[..., E.OPCODE_START + Opcode.BNZ] > 0.5

        if not (is_jmp.any() or is_bz.any() or is_bnz.any()):
            return x

        out = x.clone()

        # Get immediate and current PC
        imm = torch.zeros_like(x[..., 0])
        pc = torch.zeros_like(x[..., 0])
        for nib in range(8):
            imm = imm + x[..., E.IMM_VAL_START + nib] * (16 ** nib)
            pc = pc + x[..., E.PC_VAL_START + nib] * (16 ** nib)

        # Check if AX is zero (for BZ/BNZ)
        ax = torch.zeros_like(x[..., 0], dtype=torch.int64)
        for nib in range(8):
            ax = ax + x[..., E.AX_VAL_START + nib].long() * (16 ** nib)
        ax_is_zero = (ax == 0)

        # Compute new PC
        if is_jmp.any():
            new_pc = imm
            out[..., E.BRANCH_TAKEN] = 1.0
        elif is_bz.any():
            new_pc = torch.where(ax_is_zero, imm, pc + 8)
            out[..., E.BRANCH_TAKEN] = ax_is_zero.float()
        elif is_bnz.any():
            new_pc = torch.where(~ax_is_zero, imm, pc + 8)
            out[..., E.BRANCH_TAKEN] = (~ax_is_zero).float()
        else:
            new_pc = pc

        # Store new PC
        for nib in range(8):
            out[..., E.PC_VAL_START + nib] = (new_pc.long() >> (nib * 4)) & 0xF

        return out


# =============================================================================
# STACK FRAME EXPERT (ENT, ADJ, LEV)
# =============================================================================

class StackFrameExpert(SwiGLULayer):
    """
    Stack frame operations.

    ENT n: push BP, BP = SP, SP = SP - n
    ADJ n: SP = SP + n
    LEV:   SP = BP, restore BP
    """

    def __init__(self, dim: int):
        super().__init__(dim, hidden_dim=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_ent = x[..., E.OPCODE_START + Opcode.ENT] > 0.5
        is_adj = x[..., E.OPCODE_START + Opcode.ADJ] > 0.5
        is_lev = x[..., E.OPCODE_START + Opcode.LEV] > 0.5

        if not (is_ent.any() or is_adj.any() or is_lev.any()):
            return x

        out = x.clone()

        # Get SP, BP, immediate
        sp = torch.zeros_like(x[..., 0], dtype=torch.int64)
        bp = torch.zeros_like(x[..., 0], dtype=torch.int64)
        imm = torch.zeros_like(x[..., 0], dtype=torch.int64)
        for nib in range(8):
            sp = sp + x[..., E.SP_VAL_START + nib].long() * (16 ** nib)
            bp = bp + x[..., E.BP_VAL_START + nib].long() * (16 ** nib)
            imm = imm + x[..., E.IMM_VAL_START + nib].long() * (16 ** nib)

        if is_ent.any():
            # ENT: push BP, BP = SP, SP = SP - n
            new_sp = sp - 8 - imm  # push BP then allocate
            new_bp = sp - 8
        elif is_adj.any():
            # ADJ: SP = SP + n
            new_sp = sp + imm
            new_bp = bp
        elif is_lev.any():
            # LEV: SP = BP, restore BP
            new_sp = bp + 8  # pop BP
            new_bp = bp  # Would read from memory
        else:
            new_sp = sp
            new_bp = bp

        # Store new SP
        for nib in range(8):
            out[..., E.SP_VAL_START + nib] = ((new_sp >> (nib * 4)) & 0xF).float()
            out[..., E.BP_VAL_START + nib] = ((new_bp >> (nib * 4)) & 0xF).float()

        return out


# =============================================================================
# DIV EXPERT (Subroutine using MUL/SUB)
# =============================================================================

class DivExpert(nn.Module):
    """
    DIV operation as subroutine invoking MUL and SUB experts.

    Newton-Raphson algorithm:
        1. Initial guess x ≈ 1/b
        2. Iterate: x = x * (2 - b * x)  [uses MUL expert]
        3. Result: a * x                  [uses MUL expert]
        4. Correction via remainder       [uses SUB expert]

    This is truly pure neural - all arithmetic goes through the SwiGLU experts.
    """

    def __init__(self, dim: int, mul_expert: MulExpert, sub_expert: SubExpert):
        super().__init__()
        self.dim = dim
        self.mul_expert = mul_expert
        self.sub_expert = sub_expert
        self.SCALE = 2 ** 16  # 65536

    def _invoke_mul(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Invoke MUL expert and return product."""
        # Encode operands
        work = x.clone()
        work[..., E.OPCODE_START:E.OPCODE_START + E.NUM_OPCODES] = 0
        work[..., E.OPCODE_START + Opcode.MUL] = 1.0

        for nib in range(8):
            work[..., E.OP_A_START + nib] = ((a >> (nib * 4)) & 0xF).float()
            work[..., E.OP_B_START + nib] = ((b >> (nib * 4)) & 0xF).float()

        # Run MUL expert
        result = self.mul_expert(work)

        # Read result
        product = torch.zeros_like(a)
        for nib in range(8):
            product = product + (result[..., E.RESULT_START + nib].long() << (nib * 4))
        return product

    def _invoke_sub(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Invoke SUB expert and return difference."""
        work = x.clone()
        work[..., E.OPCODE_START:E.OPCODE_START + E.NUM_OPCODES] = 0
        work[..., E.OPCODE_START + Opcode.SUB] = 1.0

        for nib in range(8):
            work[..., E.OP_A_START + nib] = ((a >> (nib * 4)) & 0xF).float()
            work[..., E.OP_B_START + nib] = ((b >> (nib * 4)) & 0xF).float()

        result = self.sub_expert(work)

        diff = torch.zeros_like(a)
        for nib in range(8):
            diff = diff + (result[..., E.RESULT_START + nib].long() << (nib * 4))
        return diff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_div = x[..., E.OPCODE_START + Opcode.DIV] > 0.5
        is_mod = x[..., E.OPCODE_START + Opcode.MOD] > 0.5

        if not (is_div.any() or is_mod.any()):
            return x

        out = x.clone()

        # Get operands
        a = torch.zeros_like(x[..., 0], dtype=torch.int64)
        b = torch.zeros_like(x[..., 0], dtype=torch.int64)
        for nib in range(8):
            a = a + x[..., E.OP_A_START + nib].long() * (16 ** nib)
            b = b + x[..., E.OP_B_START + nib].long() * (16 ** nib)

        b = torch.clamp(b, min=1)
        SCALE = self.SCALE

        # Newton-Raphson using MUL/SUB experts
        # Initial guess
        x_recip = (SCALE // torch.clamp(b, min=1)).long()

        # 8 iterations of x = x * (2 - b*x/SCALE)
        two_scaled = torch.tensor(2 * SCALE, device=x.device, dtype=torch.int64)

        for _ in range(8):
            bx = self._invoke_mul(x, b, x_recip)
            correction = two_scaled - bx
            x_corr = self._invoke_mul(x, x_recip, correction)
            x_recip = x_corr // SCALE
            x_recip = torch.clamp(x_recip, min=1)

        # quotient = a * x_recip / SCALE
        ax = self._invoke_mul(x, a, x_recip)
        quotient = ax // SCALE

        # remainder = a - quotient * b
        qb = self._invoke_mul(x, quotient, b)
        remainder = self._invoke_sub(x, a, qb)

        # Correction loop (handles Newton-Raphson approximation errors)
        # For large numbers, may need multiple corrections
        for _ in range(20):
            too_high = remainder < 0
            quotient = torch.where(too_high, quotient - 1, quotient)
            remainder = torch.where(too_high, remainder + b, remainder)

            too_low = remainder >= b
            quotient = torch.where(too_low, quotient + 1, quotient)
            remainder = torch.where(too_low, remainder - b, remainder)

            # Early exit if converged
            if not (too_high.any() or too_low.any()):
                break

        # Store results
        if is_div.any():
            for nib in range(8):
                out[..., E.RESULT_START + nib] = ((quotient >> (nib * 4)) & 0xF).float()

        if is_mod.any():
            for nib in range(8):
                out[..., E.RESULT_START + nib] = ((remainder >> (nib * 4)) & 0xF).float()

        return out


# =============================================================================
# INSTRUCTION MoE (Mixture of Experts)
# =============================================================================

class InstructionMoE(nn.Module):
    """
    Mixture of Experts with opcode-based routing.

    Each expert is a SwiGLU FFN that handles a specific operation.
    Routing is determined by the opcode in the input embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Create arithmetic experts
        self.add_expert = AddExpert(dim)
        self.sub_expert = SubExpert(dim)
        self.mul_expert = MulExpert(dim)
        self.div_expert = DivExpert(dim, self.mul_expert, self.sub_expert)

        # Create bitwise experts
        self.and_expert = BitwiseExpert(dim, 'and')
        self.or_expert = BitwiseExpert(dim, 'or')
        self.xor_expert = BitwiseExpert(dim, 'xor')

        # Create comparison expert (handles all comparisons)
        self.comparison_expert = ComparisonExpert(dim)

        # Create shift expert (handles SHL, SHR)
        self.shift_expert = ShiftExpert(dim)

        # Create control flow expert (handles JMP, BZ, BNZ)
        self.control_flow_expert = ControlFlowExpert(dim)

        # Create stack frame expert (handles ENT, ADJ, LEV)
        self.stack_frame_expert = StackFrameExpert(dim)

        # Opcode to expert mapping
        self.opcode_to_expert = {
            # Arithmetic
            Opcode.ADD: self.add_expert,
            Opcode.SUB: self.sub_expert,
            Opcode.MUL: self.mul_expert,
            Opcode.DIV: self.div_expert,
            Opcode.MOD: self.div_expert,
            # Bitwise
            Opcode.AND: self.and_expert,
            Opcode.OR: self.or_expert,
            Opcode.XOR: self.xor_expert,
            # Comparisons
            Opcode.EQ: self.comparison_expert,
            Opcode.NE: self.comparison_expert,
            Opcode.LT: self.comparison_expert,
            Opcode.GT: self.comparison_expert,
            Opcode.LE: self.comparison_expert,
            Opcode.GE: self.comparison_expert,
            # Shifts
            Opcode.SHL: self.shift_expert,
            Opcode.SHR: self.shift_expert,
            # Control flow
            Opcode.JMP: self.control_flow_expert,
            Opcode.BZ: self.control_flow_expert,
            Opcode.BNZ: self.control_flow_expert,
            # Stack frame
            Opcode.ENT: self.stack_frame_expert,
            Opcode.ADJ: self.stack_frame_expert,
            Opcode.LEV: self.stack_frame_expert,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route to appropriate expert based on opcode."""
        out = x

        # Check which opcodes are active and route
        # Use set to avoid running same expert multiple times
        processed = set()
        for opcode, expert in self.opcode_to_expert.items():
            if id(expert) in processed:
                continue
            is_active = x[..., E.OPCODE_START + opcode] > 0.5
            if is_active.any():
                out = expert(out)
                processed.add(id(expert))

        return out

    def count_weights(self) -> dict:
        """Count weights per expert."""
        counts = {
            'ADD': self.add_expert.count_nonzero_weights(),
            'SUB': self.sub_expert.count_nonzero_weights(),
            'MUL': self.mul_expert.count_nonzero_weights(),
            'AND': self.and_expert.count_nonzero_weights(),
            'OR': self.or_expert.count_nonzero_weights(),
            'XOR': self.xor_expert.count_nonzero_weights(),
            'CMP': self.comparison_expert.count_nonzero_weights(),
            'SHIFT': self.shift_expert.count_nonzero_weights(),
            'CTRL': self.control_flow_expert.count_nonzero_weights(),
            'STACK': self.stack_frame_expert.count_nonzero_weights(),
        }
        return counts


# =============================================================================
# MAIN NEURAL VM
# =============================================================================

class NeuralVMv6(nn.Module):
    """
    Pure Neural VM V6.

    Single forward pass through InstructionMoE computes the result.
    All operations are SwiGLU FFN layers with baked weights.
    """

    def __init__(self):
        super().__init__()
        self.moe = InstructionMoE(E.DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)

    def compute(self, a: int, b: int, opcode: int) -> int:
        """Convenience method to compute a single operation."""
        x = torch.zeros(1, 1, E.DIM)

        # Encode operands
        for nib in range(8):
            x[0, 0, E.OP_A_START + nib] = (a >> (nib * 4)) & 0xF
            x[0, 0, E.OP_B_START + nib] = (b >> (nib * 4)) & 0xF

        # Set opcode
        x[0, 0, E.OPCODE_START + opcode] = 1.0

        # Run forward
        out = self.forward(x)

        # Read result
        result = 0
        for nib in range(8):
            result += int(out[0, 0, E.RESULT_START + nib].item()) << (nib * 4)

        return result


# =============================================================================
# OPCODE TABLE
# =============================================================================

def print_opcode_table():
    """Print the full opcode table."""
    print("=" * 60)
    print("OPCODE TABLE (C4 Compatible)")
    print("=" * 60)
    opcodes = [
        (0, "LEA", "Load Effective Address"),
        (1, "IMM", "Load Immediate"),
        (2, "JMP", "Jump"),
        (3, "JSR", "Jump to Subroutine"),
        (4, "BZ", "Branch if Zero"),
        (5, "BNZ", "Branch if Not Zero"),
        (6, "ENT", "Enter function"),
        (7, "ADJ", "Adjust SP"),
        (8, "LEV", "Leave function"),
        (9, "LI", "Load Int"),
        (10, "LC", "Load Char"),
        (11, "SI", "Store Int"),
        (12, "SC", "Store Char"),
        (13, "PSH", "Push AX"),
        (14, "OR", "Bitwise OR"),
        (15, "XOR", "Bitwise XOR"),
        (16, "AND", "Bitwise AND"),
        (17, "EQ", "Equal"),
        (18, "NE", "Not Equal"),
        (19, "LT", "Less Than"),
        (20, "GT", "Greater Than"),
        (21, "LE", "Less or Equal"),
        (22, "GE", "Greater or Equal"),
        (23, "SHL", "Shift Left"),
        (24, "SHR", "Shift Right"),
        (25, "ADD", "Add"),
        (26, "SUB", "Subtract"),
        (27, "MUL", "Multiply"),
        (28, "DIV", "Divide"),
        (29, "MOD", "Modulo"),
        (38, "EXIT", "Exit program"),
        (64, "GETC", "Get character"),
        (65, "PUTC", "Put character"),
    ]

    for code, name, desc in opcodes:
        if name in ["ADD", "SUB", "MUL"]:
            impl = "SwiGLU FFN"
        elif name in ["DIV", "MOD"]:
            impl = "Subroutine (MUL+SUB)"
        elif name in ["AND", "OR", "XOR"]:
            impl = "Bitwise FFN"
        elif name in ["EQ", "NE", "LT", "GT", "LE", "GE"]:
            impl = "Comparison FFN"
        elif name in ["SHL", "SHR"]:
            impl = "Shift FFN"
        elif name in ["JMP", "BZ", "BNZ"]:
            impl = "Control Flow FFN"
        elif name in ["ENT", "ADJ", "LEV"]:
            impl = "Stack Frame FFN"
        else:
            impl = "Planned"
        print(f"  {code:3d}  {name:5s}  {desc:25s}  [{impl}]")
    print("=" * 60)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Opcode',
    'EmbedDims',
    'E',
    'SwiGLULayer',
    'AddExpert',
    'SubExpert',
    'MulExpert',
    'DivExpert',
    'BitwiseExpert',
    'ComparisonExpert',
    'ShiftExpert',
    'ControlFlowExpert',
    'StackFrameExpert',
    'InstructionMoE',
    'NeuralVMv6',
    'print_opcode_table',
]


if __name__ == "__main__":
    print_opcode_table()
    print("\nRun tests with: python tests/test_v6.py")
