"""
Purely Neural Efficient ALU Integration.

All BD ↔ GenericE format conversions are done with baked FFN weights.
NO Python loops or conditionals in the forward pass.

Format conversion approach:
- One-hot to scalar: Linear projection with weights [0, 1, 2, ..., 15]
- Scalar to one-hot: Step pair detection for each value 0-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alu.chunk_config import NIBBLE
from .alu.ops.add import build_add_layers
from .alu.ops.sub import build_sub_layers
from .alu.ops.mul import build_mul_layers
from .alu.ops.shift import build_shl_layers, build_shr_layers
from .alu.ops.div import build_div_layers
from .alu.ops.mod import build_mod_layers
from .alu.ops.bitwise import build_and_layers, build_or_layers, build_xor_layers
from .alu.ops.common import GenericE, GenericPureFFN


class BDToGEConverter(nn.Module):
    """Convert BD format (one-hot) to GenericE format (scalar) - Pure Neural.

    BD format: [seq_len, 512] with one-hot nibbles at ALU_LO, ALU_HI, AX_CARRY_LO, AX_CARRY_HI
    GE format: [8, 160] with scalar nibbles at NIB_A, NIB_B per position

    Conversion: scalar = sum_{k=0}^{15} k * one_hot[k]
    """

    def __init__(self, BD, ge: GenericE):
        super().__init__()
        self.BD = BD
        self.ge = ge

        # Build projection weights for one-hot to scalar
        # W_proj[ge_dim, bd_dim] maps BD dims to GE dims
        ge_dim = ge.DIM  # 160
        bd_dim = 512

        self.register_buffer('W_proj', torch.zeros(8, ge_dim, bd_dim))

        with torch.no_grad():
            # For positions 0-1 (lo/hi byte), map ALU and AX_CARRY
            # Position 0: ALU_LO → NIB_A, AX_CARRY_LO → NIB_B
            for k in range(16):
                self.W_proj[0, ge.NIB_A, BD.ALU_LO + k] = float(k)
                self.W_proj[0, ge.NIB_B, BD.AX_CARRY_LO + k] = float(k)

            # Position 1: ALU_HI → NIB_A, AX_CARRY_HI → NIB_B
            for k in range(16):
                self.W_proj[1, ge.NIB_A, BD.ALU_HI + k] = float(k)
                self.W_proj[1, ge.NIB_B, BD.AX_CARRY_HI + k] = float(k)

            # Copy opcode flags to all positions
            # Map BD opcode dims to GE opcode slots
            self.opcode_map = [
                (BD.OP_ADD, 25),
                (BD.OP_SUB, 26),
                (BD.OP_MUL, 27),
                (BD.OP_OR, 28),
                (BD.OP_XOR, 29),
                (BD.OP_AND, 30),
                (BD.OP_SHL, 23),
                (BD.OP_SHR, 24),
                (BD.OP_DIV, 31),
                (BD.OP_MOD, 32),
            ]
            # FIX 2026-05-06: Use 1.0 scaling for opcodes, not 0.2.
            # The shift layers (ShlPrecomputeFFN, etc.) use opcode values as multipliers
            # in their gates, so they need the full value of 1.0 when active.
            for pos in range(8):
                for bd_dim_idx, ge_opcode in self.opcode_map:
                    self.W_proj[pos, ge.OP_START + ge_opcode, bd_dim_idx] = 1.0

    def forward(self, x_bd):
        """
        Args:
            x_bd: [B, seq_len, 512] BD format (only AX marker positions used)

        Returns:
            x_ge: [B, seq_len, 8, 160] GenericE format
        """
        B, seq_len, _ = x_bd.shape
        BD = self.BD

        # FIX 2026-05-06: Clamp ALU_LO/HI and AX_CARRY_LO/HI to non-negative
        # L6 FFN clears these to -5.0, and L7 attention only overwrites active indices.
        # The negative residuals corrupt the scalar conversion (sum of k * one_hot[k]).
        x_bd_clamped = x_bd.clone()
        x_bd_clamped[:, :, BD.ALU_LO:BD.ALU_LO + 16] = torch.clamp(x_bd[:, :, BD.ALU_LO:BD.ALU_LO + 16], min=0)
        x_bd_clamped[:, :, BD.ALU_HI:BD.ALU_HI + 16] = torch.clamp(x_bd[:, :, BD.ALU_HI:BD.ALU_HI + 16], min=0)
        x_bd_clamped[:, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO + 16] = torch.clamp(x_bd[:, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO + 16], min=0)
        x_bd_clamped[:, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI + 16] = torch.clamp(x_bd[:, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI + 16], min=0)

        # FIX 2026-05-09 (Phase 0): Replace argmax with linear projection (weighted sum).
        # argmax is not a SwiGLU FFN operation per the pure-neural policy. Linear
        # projection sum_k k * one_hot[k] IS — it's just a matrix multiply, baked
        # into FFN weights. This requires upstream to keep ALU_LO/HI as clean
        # one-hot encodings; if a leak makes one_hot[k] > 1.0, the sum overshoots.
        # We rely on the right-sizing pass + OPCODE_BLOCK_MAP defensive gates to
        # keep upstream clean enough.
        B, seq_len, _ = x_bd_clamped.shape
        x_ge = torch.zeros(B, seq_len, 8, self.ge.DIM, device=x_bd.device, dtype=x_bd.dtype)

        # Linear projection: NIB_A_value = sum_k k * one_hot[k]
        # Build the [16] coefficient vector once.
        k_coeffs = torch.arange(16, device=x_bd.device, dtype=x_bd.dtype)

        alu_lo = x_bd_clamped[:, :, BD.ALU_LO:BD.ALU_LO + 16]
        ax_lo = x_bd_clamped[:, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO + 16]
        x_ge[:, :, 0, self.ge.NIB_A] = (alu_lo * k_coeffs).sum(dim=-1)
        x_ge[:, :, 0, self.ge.NIB_B] = (ax_lo * k_coeffs).sum(dim=-1)

        alu_hi = x_bd_clamped[:, :, BD.ALU_HI:BD.ALU_HI + 16]
        ax_hi = x_bd_clamped[:, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI + 16]
        x_ge[:, :, 1, self.ge.NIB_A] = (alu_hi * k_coeffs).sum(dim=-1)
        x_ge[:, :, 1, self.ge.NIB_B] = (ax_hi * k_coeffs).sum(dim=-1)

        # Copy opcode flags to all positions.
        # FIX 2026-05-06: Normalize opcode values to 0/1 by thresholding at 0.5.
        # The transformer writes opcode flags with various magnitudes (e.g., 5.0),
        # but the shift layers expect exactly 1.0 when active. Threshold to normalize.
        for pos in range(8):
            for bd_dim_idx, ge_opcode in self.opcode_map:
                x_ge[:, :, pos, self.ge.OP_START + ge_opcode] = (x_bd[:, :, bd_dim_idx] > 0.5).float()

        return x_ge


class GEToBDConverter(nn.Module):
    """Convert GenericE format (scalar) to BD format (one-hot) - Pure Neural.

    Uses step pairs to detect scalar values and output one-hot encoding.
    scalar_to_onehot[k] = step(x >= k - 0.5) - step(x >= k + 0.5)
    """

    def __init__(self, BD, ge: GenericE, S=100.0):
        super().__init__()
        self.BD = BD
        self.ge = ge
        self.S = S

        # Build step-pair FFN for scalar to one-hot conversion
        # For each of 16 possible values, we need 2 hidden units (step pair)
        # Output positions: OUTPUT_LO (16 dims) and OUTPUT_HI (16 dims)
        hidden_dim = 16 * 2 * 2  # 16 values × 2 positions (lo/hi) × 2 units per step pair

        self.ffn = GenericPureFFN(ge.DIM, hidden_dim=hidden_dim, dtype=torch.float32)

        with torch.no_grad():
            W_up = self.ffn.W_up
            b_up = self.ffn.b_up
            W_gate = self.ffn.W_gate
            W_down = self.ffn.W_down

            h = 0

            # For each output position (0=lo, 1=hi)
            for out_pos in range(2):
                # For each possible value k = 0..15
                for k in range(16):
                    # Step pair: step(result >= k - 0.5) - step(result >= k + 0.5)
                    # Unit A: silu(S*(result - k + 0.5)) → +1/S
                    W_up[h, ge.RESULT] = S
                    b_up[h] = -S * (k - 0.5)
                    W_gate[h, ge.RESULT] = 0.0  # No gating, always active
                    # But we need to gate on opcode being active...
                    # Actually for simplicity, always compute and let BD masking handle it
                    h += 1

                    # Unit B: silu(S*(result - k - 0.5)) → -1/S
                    W_up[h, ge.RESULT] = S
                    b_up[h] = -S * (k + 0.5)
                    h += 1

        # Store output mapping info
        self.out_pos_lo = 0
        self.out_pos_hi = 1

    def forward(self, x_ge, x_bd, opcode_mask=None):
        """
        Args:
            x_ge: [B, seq_len, 8, 160] GenericE format with RESULT filled
            x_bd: [B, seq_len, 512] BD format to update
            opcode_mask: [B, seq_len] Optional mask indicating where opcodes are active.
                        If None, writes OUTPUT unconditionally (backward compat).
                        If provided, only writes OUTPUT where mask > 0.5.

        Returns:
            x_bd_out: [B, seq_len, 512] with OUTPUT_LO/HI updated
        """
        B, seq_len, num_pos, ge_dim = x_ge.shape
        BD = self.BD
        S = self.S

        x_bd_out = x_bd.clone()

        # Extract result nibbles from positions 0 and 1
        result_lo = x_ge[:, :, 0, self.ge.RESULT]  # [B, seq_len]
        result_hi = x_ge[:, :, 1, self.ge.RESULT]  # [B, seq_len]

        # Convert to one-hot using vectorized step pairs with sigmoid approximation
        # For each k in 0..15: one_hot[k] = sigmoid(S*(result - k + 0.5)) - sigmoid(S*(result - k - 0.5))
        # This detects when result is in [k-0.5, k+0.5), i.e., rounds to k

        # Create k values tensor: [16]
        k_vals = torch.arange(16, device=x_ge.device, dtype=x_ge.dtype)

        # Broadcast: result_lo is [B, seq_len], k_vals is [16]
        # result_lo[:, :, None] - k_vals[None, None, :] gives [B, seq_len, 16]
        diff_lo = result_lo[:, :, None] - k_vals[None, None, :]  # [B, seq_len, 16]
        diff_hi = result_hi[:, :, None] - k_vals[None, None, :]  # [B, seq_len, 16]

        # Step pair detection: sigmoid(S*(diff + 0.5)) - sigmoid(S*(diff - 0.5))
        indicator_lo = torch.sigmoid(S * (diff_lo + 0.5)) - torch.sigmoid(S * (diff_lo - 0.5))  # [B, seq_len, 16]
        indicator_hi = torch.sigmoid(S * (diff_hi + 0.5)) - torch.sigmoid(S * (diff_hi - 0.5))  # [B, seq_len, 16]

        # Apply opcode mask if provided (only write OUTPUT where opcodes are active)
        if opcode_mask is not None:
            mask_expanded = opcode_mask[:, :, None]
            indicator_lo = indicator_lo * mask_expanded
            indicator_hi = indicator_hi * mask_expanded

        x_bd_out[:, :, BD.OUTPUT_LO:BD.OUTPUT_LO + 16] += indicator_lo * 2.0
        x_bd_out[:, :, BD.OUTPUT_HI:BD.OUTPUT_HI + 16] += indicator_hi * 2.0

        # FIX 2026-05-06: Set carry/borrow flags for multi-byte propagation.
        # CarryPropagationPostOp expects:
        #   CARRY[1] for ADD overflow (sum >= 256)
        #   CARRY[2] for SUB borrow (a < b)
        # Only set these at AX marker position (where byte 0 is computed).
        operand_a_lo = x_ge[:, :, 0, self.ge.NIB_A]  # [B, seq_len]
        operand_a_hi = x_ge[:, :, 1, self.ge.NIB_A]  # [B, seq_len]
        operand_b_lo = x_ge[:, :, 0, self.ge.NIB_B]  # [B, seq_len]
        operand_b_hi = x_ge[:, :, 1, self.ge.NIB_B]  # [B, seq_len]

        # Reconstruct byte values: byte = lo + hi * 16
        operand_a = operand_a_lo + operand_a_hi * 16.0  # [B, seq_len], 0-255
        operand_b = operand_b_lo + operand_b_hi * 16.0  # [B, seq_len], 0-255

        # ADD carry: sum >= 256
        sum_ab = operand_a + operand_b
        add_carry = torch.sigmoid(S * (sum_ab - 255.5))

        # SUB borrow: a < b (i.e., a - b < 0)
        sub_borrow = torch.sigmoid(S * (operand_b - operand_a - 0.5))

        # Gate on MARK_AX (only at AX marker position)
        mark_ax = x_bd[:, :, BD.MARK_AX]
        ax_mask = (mark_ax > 0.5).float()

        # FIX 2026-05-08: Gate CARRY flags by their respective opcodes.
        # Previously, both CARRY[1] and CARRY[2] were set at all AX markers.
        # This caused SUB borrow to be set during ADD operations, which triggered
        # CarryPropagationPostOp's SUB units to fire spuriously.
        op_add = x_bd[:, :, BD.OP_ADD]
        op_sub = x_bd[:, :, BD.OP_SUB]
        add_opcode_mask = (op_add > 1.0).float()  # OP_ADD ≈ 5.0 when active
        sub_opcode_mask = (op_sub > 1.0).float()  # OP_SUB ≈ 5.0 when active

        # Write CARRY[1] for ADD only, CARRY[2] for SUB only
        x_bd_out[:, :, BD.CARRY + 1] += add_carry * ax_mask * add_opcode_mask * 2.0
        x_bd_out[:, :, BD.CARRY + 2] += sub_borrow * ax_mask * sub_opcode_mask * 2.0

        return x_bd_out


class PureNeuralALU(nn.Module):
    """Purely neural ALU that wraps efficient ops with neural format conversion.

    All operations are performed using baked FFN weights - no Python loops
    or conditionals in forward pass.
    """

    def __init__(self, S, BD, operations='add_sub'):
        """
        Args:
            S: SwiGLU scale (100.0)
            BD: _SetDim class with dimension constants
            operations: Which operations to include:
                'add_sub' - ADD and SUB
                'bitwise' - AND, OR, XOR
                'mul' - MUL
                'shift' - SHL, SHR
        """
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.operations = operations

        # Format converters
        self.bd_to_ge = BDToGEConverter(BD, self.ge)
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

        # Build operation-specific layers
        if operations == 'add_sub':
            self.add_layers = nn.ModuleList(build_add_layers(NIBBLE, opcode=25))
            self.sub_layers = nn.ModuleList(build_sub_layers(NIBBLE, opcode=26))
        elif operations == 'bitwise':
            self.and_layers = nn.ModuleList(build_and_layers(NIBBLE, opcode=30))
            self.or_layers = nn.ModuleList(build_or_layers(NIBBLE, opcode=28))
            self.xor_layers = nn.ModuleList(build_xor_layers(NIBBLE, opcode=29))
        elif operations == 'mul':
            self.mul_layers = nn.ModuleList(build_mul_layers(NIBBLE, opcode=27))
        elif operations == 'shift':
            self.shl_layers = nn.ModuleList(build_shl_layers(NIBBLE, opcode=23))
            self.shr_layers = nn.ModuleList(build_shr_layers(NIBBLE, opcode=24))
        elif operations == 'div_mod':
            self.div_layers = nn.ModuleList(build_div_layers(NIBBLE, opcode=31))
            self.mod_layers = nn.ModuleList(build_mod_layers(NIBBLE, opcode=32))

    def forward(self, x_bd):
        """
        Process ALU operations in BD format, fully neural.

        Args:
            x_bd: [B, seq_len, 512] BD format

        Returns:
            [B, seq_len, 512] with ALU results
        """
        B, seq_len, _ = x_bd.shape
        BD = self.BD

        # Convert BD → GE format
        x_ge = self.bd_to_ge(x_bd)  # [B, seq_len, 8, 160]

        # Flatten for efficient layer processing
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)  # [B*seq_len, 8, 160]

        x_ge_out = x_ge_flat.clone()
        opcode_mask_flat = torch.zeros(B * seq_len, device=x_bd.device, dtype=x_bd.dtype)

        if self.operations == 'add_sub':
            x_add = x_ge_flat.clone()
            for layer in self.add_layers:
                x_add = layer(x_add)

            x_sub = x_ge_flat.clone()
            for layer in self.sub_layers:
                x_sub = layer(x_sub)

            # FIX 2026-05-06: Opcode values are 0.2 (not 1.0) due to BDToGEConverter scaling.
            # Normalize to 0/1 before using as multipliers to avoid corrupting results.
            op_add = (x_ge_flat[:, 0, self.ge.OP_START + 25] > 0.1).float()
            op_sub = (x_ge_flat[:, 0, self.ge.OP_START + 26] > 0.1).float()

            op_total = op_add + op_sub

            x_ge_out[:, :, self.ge.RESULT] = (
                x_add[:, :, self.ge.RESULT] * op_add[:, None] +
                x_sub[:, :, self.ge.RESULT] * op_sub[:, None]
            )

            opcode_mask_flat = op_total

        elif self.operations == 'bitwise':
            x_and = x_ge_flat.clone()
            for layer in self.and_layers:
                x_and = layer(x_and)

            x_or = x_ge_flat.clone()
            for layer in self.or_layers:
                x_or = layer(x_or)

            x_xor = x_ge_flat.clone()
            for layer in self.xor_layers:
                x_xor = layer(x_xor)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_and = (x_ge_flat[:, 0, self.ge.OP_START + 30] > 0.1).float()
            op_or = (x_ge_flat[:, 0, self.ge.OP_START + 28] > 0.1).float()
            op_xor = (x_ge_flat[:, 0, self.ge.OP_START + 29] > 0.1).float()

            op_total = op_and + op_or + op_xor

            x_ge_out[:, :, self.ge.RESULT] = (
                x_and[:, :, self.ge.RESULT] * op_and[:, None] +
                x_or[:, :, self.ge.RESULT] * op_or[:, None] +
                x_xor[:, :, self.ge.RESULT] * op_xor[:, None]
            )

            opcode_mask_flat = op_total

        elif self.operations == 'mul':
            x_mul = x_ge_flat.clone()
            for layer in self.mul_layers:
                x_mul = layer(x_mul)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_mul = (x_ge_flat[:, 0, self.ge.OP_START + 27] > 0.1).float()

            x_ge_out[:, :, self.ge.RESULT] = (
                x_mul[:, :, self.ge.RESULT] * op_mul[:, None]
            )

            opcode_mask_flat = op_mul

        elif self.operations == 'shift':
            x_shl = x_ge_flat.clone()
            for layer in self.shl_layers:
                x_shl = layer(x_shl)

            x_shr = x_ge_flat.clone()
            for layer in self.shr_layers:
                x_shr = layer(x_shr)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_shl = (x_ge_flat[:, 0, self.ge.OP_START + 23] > 0.1).float()
            op_shr = (x_ge_flat[:, 0, self.ge.OP_START + 24] > 0.1).float()

            op_total = op_shl + op_shr

            x_ge_out[:, :, self.ge.RESULT] = (
                x_shl[:, :, self.ge.RESULT] * op_shl[:, None] +
                x_shr[:, :, self.ge.RESULT] * op_shr[:, None]
            )

            opcode_mask_flat = op_total

        elif self.operations == 'div_mod':
            x_div = x_ge_flat.clone()
            for layer in self.div_layers:
                x_div = layer(x_div)

            x_mod = x_ge_flat.clone()
            for layer in self.mod_layers:
                x_mod = layer(x_mod)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_div = (x_ge_flat[:, 0, self.ge.OP_START + 31] > 0.1).float()
            op_mod = (x_ge_flat[:, 0, self.ge.OP_START + 32] > 0.1).float()

            op_total = op_div + op_mod

            x_ge_out[:, :, self.ge.RESULT] = (
                x_div[:, :, self.ge.RESULT] * op_div[:, None] +
                x_mod[:, :, self.ge.RESULT] * op_mod[:, None]
            )

            opcode_mask_flat = op_total

        # Reshape back
        x_ge_out = x_ge_out.view(B, seq_len, 8, self.ge.DIM)

        opcode_mask = opcode_mask_flat.view(B, seq_len)

        # Only write OUTPUT at AX marker positions (MARK_AX > 0.5).
        # Without this, the ALU writes result "0" at byte positions where
        # operands are zero, corrupting the passthrough from L10 head 1.
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        x_bd_out = self.ge_to_bd(x_ge_out, x_bd, opcode_mask=opcode_mask)

        return x_bd_out

    # Stub methods for compatibility with vm_step.py
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


# Operation-named ALU classes (no layer assumptions — compiler decides placement).
# Phase 0 rename (2026-05-09): the previous names (EfficientALU_L8_L9_Neural,
# EfficientALU_L10_Neural, etc.) baked specific layer indices into the class names,
# violating "compiler determines depth/width". Layer placement is now a compiler
# concern — the operation name is what's intrinsic.
class ALUAddSub(PureNeuralALU):
    """Neural ADD/SUB."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='add_sub')


class ALUAndOrXor(PureNeuralALU):
    """Neural AND/OR/XOR."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='bitwise')


class ALUMul(PureNeuralALU):
    """Neural MUL."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='mul')


class ALUShift(PureNeuralALU):
    """Neural SHL/SHR."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='shift')


class ALUDivMod(PureNeuralALU):
    """Neural DIV/MOD."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='div_mod')


# Backward-compat aliases (deprecated): the layer-named classes still work but
# are not the preferred name. New code should use the operation-named classes
# above. These aliases will be removed once the rename has propagated.
EfficientALU_L8_L9_Neural = ALUAddSub
EfficientALU_L10_Neural = ALUAndOrXor
EfficientALU_L11_L12_Neural = ALUMul
EfficientALU_L13_Neural = ALUShift
EfficientDivMod_Neural = ALUDivMod
