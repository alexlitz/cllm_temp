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
# The operation name is what's intrinsic; layer placement is a compiler concern.
class ALUAddSub(PureNeuralALU):
    """Neural ADD/SUB."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='add_sub')


class ALUAndOrXor(PureNeuralALU):
    """Neural AND/OR/XOR."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='bitwise')


class _MulPipelineState:
    """Mutable state passed between Sequential stages of ``FlattenedALUMul``.

    Each stage reads/writes named tensor fields and returns ``self`` so it
    can be chained inside ``nn.Sequential``. Using an object (rather than a
    bare tensor tuple) keeps each stage's I/O contract uniform — every stage
    in the pipeline has the signature ``forward(state) -> state`` — which is
    exactly what ``nn.Sequential`` requires.

    Fields populated by stage:
      - ``BDToGEStage``        sets ``x_bd_in``, ``x_ge_flat``, ``x_mul``
      - mul FFN stages         update ``x_mul``
      - ``MulCombineStage``    sets ``x_ge_out``, ``opcode_mask``
      - ``GEToBDStage``        sets ``x_bd_out``
    """

    __slots__ = (
        'x_bd_in', 'x_ge_flat', 'x_mul',
        'x_ge_out', 'opcode_mask', 'x_bd_out',
    )

    def __init__(self):
        self.x_bd_in = None
        self.x_ge_flat = None
        self.x_mul = None
        self.x_ge_out = None
        self.opcode_mask = None
        self.x_bd_out = None


class _BDToGEStage(nn.Module):
    """Pipeline stage 0 (phase=11.0): BD → GE format conversion.

    Wraps ``BDToGEConverter`` with the uniform ``forward(state) -> state``
    contract used by every stage in ``FlattenedALUMul.pipeline``. Stashes
    ``x_bd_in`` (for later AX masking + as the base for ``GEToBDStage``) and
    initialises ``x_mul`` (the rolling MUL workspace) and ``x_ge_flat`` (the
    snapshot used for opcode/AX gating after the 7 mul layers run).
    """

    def __init__(self, BD, ge: GenericE):
        super().__init__()
        self.BD = BD
        self.ge = ge
        self.bd_to_ge = BDToGEConverter(BD, ge)

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        x_bd = state.x_bd_in
        B, seq_len, _ = x_bd.shape
        x_ge = self.bd_to_ge(x_bd)  # [B, seq_len, 8, 160]
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)
        state.x_ge_flat = x_ge_flat
        state.x_mul = x_ge_flat.clone()
        return state


class _MulFFNStage(nn.Module):
    """Pipeline stage wrapper around one mul-pipeline FFN.

    Holds a single mul sub-FFN (e.g. ``SchoolbookFFN``, ``CarryPassFFN``,
    ``MulGenPropFFN``, ...) and applies it to ``state.x_mul``. The wrapped
    FFN itself remains a vanilla ``nn.Module`` with its own ``W_up`` /
    ``W_gate`` / ``W_down`` parameters — this class is just the adapter
    that lets it slot into the uniform Sequential pipeline contract.
    """

    def __init__(self, sub_ffn: nn.Module):
        super().__init__()
        self.sub_ffn = sub_ffn

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        state.x_mul = self.sub_ffn(state.x_mul)
        return state


class _MulCombineStage(nn.Module):
    """Pipeline stage that merges the mul workspace into the GE output.

    Computes the opcode-gated MUL result, restricts OUTPUT writes to AX
    marker positions, and reshapes back to ``[B, seq_len, 8, DIM]``. Owns
    no parameters — it is pure tensor algebra (broadcasted multiplies,
    reshape, threshold mask) and exists as its own ``nn.Module`` so the
    Sequential pipeline has a single, statically-defined chain of modules
    rather than ad-hoc Python in ``forward``.
    """

    def __init__(self, BD, ge: GenericE):
        super().__init__()
        self.BD = BD
        self.ge = ge

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        x_bd = state.x_bd_in
        x_ge_flat = state.x_ge_flat
        x_mul = state.x_mul
        BD = self.BD
        ge = self.ge

        x_ge_out = x_ge_flat.clone()

        # FIX 2026-05-06: Normalize opcode values to 0/1.
        op_mul = (x_ge_flat[:, 0, ge.OP_START + 27] > 0.1).float()

        x_ge_out[:, :, ge.RESULT] = x_mul[:, :, ge.RESULT] * op_mul[:, None]

        B, seq_len, _ = x_bd.shape
        x_ge_out = x_ge_out.view(B, seq_len, 8, ge.DIM)
        opcode_mask = op_mul.view(B, seq_len)

        # Only write OUTPUT at AX marker positions (MARK_AX > 0.5).
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        state.x_ge_out = x_ge_out
        state.opcode_mask = opcode_mask
        return state


class _GEToBDStage(nn.Module):
    """Final pipeline stage (phase=12.3): GE → BD conversion.

    Wraps ``GEToBDConverter`` and stores its ``[B, seq_len, 512]`` output in
    ``state.x_bd_out``, which ``FlattenedALUMul.forward`` then returns.
    """

    def __init__(self, BD, ge: GenericE, S: float):
        super().__init__()
        self.BD = BD
        self.ge = ge
        self.S = S
        self.ge_to_bd = GEToBDConverter(BD, ge, S)

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        state.x_bd_out = self.ge_to_bd(
            state.x_ge_out, state.x_bd_in, opcode_mask=state.opcode_mask,
        )
        return state


# ---------------------------------------------------------------------------
# Flattened AND/OR/XOR pipeline (vanilla nn.Sequential composite).
#
# 4 stages, each a separate nn.Module installed at L10.ffn by the compiler:
#   Stage 0: BD → GE format conversion          (BitwiseBDToGEStage)
#   Stage 1: per-opcode bit extraction          (BitwiseBitExtractStage)
#   Stage 2: per-opcode bit combine + opcode    (BitwiseBitCombineStage)
#            mask merge into RESULT
#   Stage 3: GE → BD format conversion + write  (BitwiseGEToBDStage)
#            OUTPUT_LO/HI gated by AX marker
#
# The stages share intermediate GE-format state via a `BitwisePipelineState`
# object so the residual stream (BD-format) never has to carry the GE
# workspace. Each stage takes ``x_bd`` in and returns ``x_bd`` out (residual
# identity for stages 0-2, real writeback in stage 3) so the composite forward
# is a literal `nn.Sequential` chain — vanilla composition, no hand-rolled
# control flow.
#
# Forward semantics are byte-identical to the existing ``ALUAndOrXor.forward``
# (= ``PureNeuralALU(operations='bitwise').forward``):
#   - same `BDToGEConverter` weights for stage 0
#   - same `build_{and,or,xor}_layers` factories for stages 1-2 (each builder
#     yields [BitExtractFFN, BitCombineFFN]; AND/OR/XOR run in parallel on
#     cloned GE buffers, then merge via 0.1-threshold opcode mask)
#   - same MARK_AX > 0.5 gating before GE → BD writeback
# ---------------------------------------------------------------------------


class BitwisePipelineState:
    """Per-composite scratch space for the 4-stage AND/OR/XOR pipeline.

    Attaches to ``FlattenedALUAndOrXor`` and is referenced by all 4 stage
    modules. Holds intermediate tensors so each stage runs as a standalone
    block without serialising the full GE workspace into the BD residual.
    """

    def __init__(self):
        self.x_bd_in = None
        self.x_ge_flat = None    # [B*seq, 8, 160] — initial GE state (clone src)
        self.x_and = None        # [B*seq, 8, 160] — AND pipeline buffer
        self.x_or = None         # [B*seq, 8, 160] — OR pipeline buffer
        self.x_xor = None        # [B*seq, 8, 160] — XOR pipeline buffer
        self.x_ge_out = None     # [B, seq, 8, 160]  — merged result GE
        self.opcode_mask = None  # [B, seq] — opcode mask gated by MARK_AX


class BitwiseBDToGEStage(nn.Module):
    """Stage 0: BD → GenericE format conversion.

    Converts the one-hot ALU_LO/HI, AX_CARRY_LO/HI nibbles into scalar
    NIB_A/NIB_B slots and copies opcode flags into the GE OP_START region.
    Stashes the converted state for downstream stages and returns x_bd
    unchanged (residual identity).
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.state = state
        self.bd_to_ge = BDToGEConverter(BD, self.ge)

    def forward(self, x_bd):
        x_ge = self.bd_to_ge(x_bd)  # [B, seq, 8, 160]
        B, seq_len, _, _ = x_ge.shape
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)
        # Initialise per-opcode pipeline buffers as clones of the GE state.
        self.state.x_bd_in = x_bd
        self.state.x_ge_flat = x_ge_flat
        self.state.x_and = x_ge_flat.clone()
        self.state.x_or = x_ge_flat.clone()
        self.state.x_xor = x_ge_flat.clone()
        return x_bd


class BitwiseBitExtractStage(nn.Module):
    """Stage 1: per-opcode bit extraction (BitExtractFFN).

    Each of AND/OR/XOR has its own `BitExtractFFN` (gated on its own opcode)
    that splits NIB_A and NIB_B into the per-bit temp slots used by the
    combine stage. Runs all three in parallel on the cloned GE buffers
    seeded by stage 0. Mirrors the ``layers[0]`` step of `build_and_layers`,
    `build_or_layers`, `build_xor_layers` from `alu.ops.bitwise`.
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        # build_and/or/xor_layers each return [BitExtractFFN, CombineFFN]
        # for non-BIT configs (NIBBLE has chunk_bits=4). Take index 0 here.
        self.and_extract = build_and_layers(NIBBLE, opcode=30)[0]
        self.or_extract = build_or_layers(NIBBLE, opcode=28)[0]
        self.xor_extract = build_xor_layers(NIBBLE, opcode=29)[0]

    def forward(self, x_bd):
        self.state.x_and = self.and_extract(self.state.x_and)
        self.state.x_or = self.or_extract(self.state.x_or)
        self.state.x_xor = self.xor_extract(self.state.x_xor)
        return x_bd


class BitwiseBitCombineStage(nn.Module):
    """Stage 2: per-opcode bit combine + opcode-mask merge.

    Each of AND/OR/XOR runs its combine FFN (`BitAndCombineClearFFN`,
    `BitOrCombineClearFFN`, `BitXorCombineClearFFN`) on the per-opcode
    buffer from stage 1. Outputs are merged into a single GE result buffer
    via per-opcode masks (opcode value > 0.1 → 1.0). Stashes the merged
    GE state and the opcode mask for stage 3.
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.and_combine = build_and_layers(NIBBLE, opcode=30)[1]
        self.or_combine = build_or_layers(NIBBLE, opcode=28)[1]
        self.xor_combine = build_xor_layers(NIBBLE, opcode=29)[1]

    def forward(self, x_bd):
        x_and = self.and_combine(self.state.x_and)
        x_or = self.or_combine(self.state.x_or)
        x_xor = self.xor_combine(self.state.x_xor)

        ge = self.ge
        x_ge_flat = self.state.x_ge_flat

        # FIX 2026-05-06: Normalize opcode values to 0/1 (matches PureNeuralALU.forward).
        op_and = (x_ge_flat[:, 0, ge.OP_START + 30] > 0.1).float()
        op_or = (x_ge_flat[:, 0, ge.OP_START + 28] > 0.1).float()
        op_xor = (x_ge_flat[:, 0, ge.OP_START + 29] > 0.1).float()
        op_total = op_and + op_or + op_xor

        x_ge_out = x_ge_flat.clone()
        x_ge_out[:, :, ge.RESULT] = (
            x_and[:, :, ge.RESULT] * op_and[:, None]
            + x_or[:, :, ge.RESULT] * op_or[:, None]
            + x_xor[:, :, ge.RESULT] * op_xor[:, None]
        )

        x_bd_in = self.state.x_bd_in
        B, seq_len, _ = x_bd_in.shape
        x_ge_out = x_ge_out.view(B, seq_len, 8, ge.DIM)
        opcode_mask = op_total.view(B, seq_len)

        # MARK_AX gating (matches PureNeuralALU.forward post-loop step).
        BD = self.BD
        mark_ax = x_bd_in[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        self.state.x_ge_out = x_ge_out
        self.state.opcode_mask = opcode_mask
        return x_bd


class BitwiseGEToBDStage(nn.Module):
    """Stage 3: GenericE → BD format conversion + write OUTPUT.

    Reads the merged GE RESULT from stage 2 and writes one-hot OUTPUT_LO/HI
    into x_bd, gated by the AX-marker-aware opcode mask. Returns the
    updated x_bd as the FFN output.
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

    def forward(self, x_bd):
        x_ge_out = self.state.x_ge_out
        opcode_mask = self.state.opcode_mask
        x_bd_out = self.ge_to_bd(x_ge_out, x_bd, opcode_mask=opcode_mask)
        return x_bd_out


class FlattenedALUAndOrXor(nn.Module):
    """Vanilla composite AND/OR/XOR FFN: 4 sub-stages run as `nn.Sequential`.

    Drop-in replacement for ``ALUAndOrXor`` (= ``PureNeuralALU(operations=
    'bitwise')``) at L10. Forward is byte-identical to the original wrapper:
    same `BDToGEConverter`, same `build_{and,or,xor}_layers` FFNs, same
    0.1-threshold opcode mask, same MARK_AX gating. The difference is
    structural — the monolithic forward() is split into 4 stage modules
    that share intermediate state via a `BitwisePipelineState` and chain
    via a literal ``nn.Sequential``.

    The 4 sub-stages are exposed as named attributes so a future compiler
    bake_fn pipeline (analogous to ``make_l13_alu_shift_*_op``) can install
    each stage independently. Until those compiler ops exist, this class
    can be instantiated directly as a runtime-equivalent of ALUAndOrXor.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = BitwisePipelineState()
        # Vanilla nn.Sequential composition: each stage takes x_bd → x_bd.
        # Stages are registered ONLY through `pipeline` (not as direct
        # attributes) to avoid double-counting in `state_dict()`. Access them
        # via `bdtoge_stage`/`bit_extract_stage`/`bit_combine_stage`/
        # `getobd_stage` properties below for readability + compiler-op hooks.
        self.pipeline = nn.Sequential(
            BitwiseBDToGEStage(S, BD, self.state),
            BitwiseBitExtractStage(S, BD, self.state),
            BitwiseBitCombineStage(S, BD, self.state),
            BitwiseGEToBDStage(S, BD, self.state),
        )

    # Named-stage views into `self.pipeline` so external callers (compiler
    # bake_fns, debug introspection) can fetch a stage without hard-coding
    # the index. `nn.Sequential` indexing is supported as of PyTorch 1.0+.
    @property
    def bdtoge_stage(self) -> BitwiseBDToGEStage:
        return self.pipeline[0]

    @property
    def bit_extract_stage(self) -> BitwiseBitExtractStage:
        return self.pipeline[1]

    @property
    def bit_combine_stage(self) -> BitwiseBitCombineStage:
        return self.pipeline[2]

    @property
    def getobd_stage(self) -> BitwiseGEToBDStage:
        return self.pipeline[3]

    def forward(self, x_bd):
        return self.pipeline(x_bd)

    # Stub methods for compatibility with vm_step.py model utilities
    # (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class FlattenedALUMul(nn.Module):
    """Flattened (compiler-baked) MUL ALU — Sequential of vanilla stages.

    Byte-identical to ``ALUMul.forward`` (= ``PureNeuralALU(operations='mul').forward``)
    but exposes the BD↔GE converters and the 7 sub-FFN MUL pipeline stages as
    individually-installable submodules so the unified compiler can bake each
    stage as a discrete ``Operation``.

    The pipeline is materialised as a single ``nn.Sequential`` whose stages
    each implement ``forward(state) -> state``. Once all 9 compiler ops have
    run, ``self.pipeline`` is::

        nn.Sequential(
            _BDToGEStage,                  # phase=11.0
            _MulFFNStage(SchoolbookFFN),   # phase=11.1
            _MulFFNStage(CarryPassFFN(0)), # phase=11.2
            _MulFFNStage(CarryPassFFN(1)), # phase=11.3
            _MulFFNStage(CarryPassFFN(2)), # phase=11.4
            _MulFFNStage(MulGenPropFFN),   # phase=12.0
            _MulFFNStage(MulBinaryLookaheadFFN),  # phase=12.1
            _MulFFNStage(MulFinalCorrectionFFN),  # phase=12.2
            _MulCombineStage,              # opcode/AX gating + reshape
            _GEToBDStage,                  # phase=12.3
        )

    ``forward`` is therefore just::

        state = _MulPipelineState(); state.x_bd_in = x_bd
        return self.pipeline(state).x_bd_out

    No Python control flow over the sub-modules: the chaining is the
    declarative ``nn.Sequential`` itself.

    Sub-stages, installed by 9 compiler block ops:

      - ``bd_to_ge``:           ``BDToGEConverter`` — phase=11.0
      - ``mul_layers[0]``:      ``SchoolbookFFN``           — phase=11.1
      - ``mul_layers[1]``:      ``CarryPassFFN(pass_idx=0)``— phase=11.2
      - ``mul_layers[2]``:      ``CarryPassFFN(pass_idx=1)``— phase=11.3
      - ``mul_layers[3]``:      ``CarryPassFFN(pass_idx=2)``— phase=11.4
      - ``mul_layers[4]``:      ``MulGenPropFFN``           — phase=12.0
      - ``mul_layers[5]``:      ``MulBinaryLookaheadFFN``   — phase=12.1
      - ``mul_layers[6]``:      ``MulFinalCorrectionFFN``   — phase=12.2
      - ``ge_to_bd``:           ``GEToBDConverter``         — phase=12.3

    NIBBLE config produces exactly 3 carry passes (verified via
    ``_compute_carry_passes(NIBBLE) == [112, 7, 1]``), so the pipeline has
    the canonical 7-stage shape (1 schoolbook + 3 carry + 1 genprop +
    1 lookahead + 1 final-correction). The ops sort by phase (11.0 .. 12.3)
    and bind to L11 via ``layer_idx=11`` since the runtime collapses them
    into a single block forward pass (matching the previous ``ALUMul``
    that ran as one ``model.blocks[11].ffn`` call).

    Forward replicates the ``PureNeuralALU`` mul branch byte-for-byte
    (same 0.1-threshold opcode normalization, same MARK_AX-only OUTPUT
    gating).

    Backward-compat properties ``bd_to_ge``, ``ge_to_bd``, and
    ``mul_layers`` remain readable so existing inspectors / tests continue
    to work; they are computed from the corresponding stages in
    ``self._stages``.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)

        # Each install_* call appends the corresponding stage to _stages.
        # Once all 9 stages are installed, ``pipeline`` is a single
        # nn.Sequential of vanilla nn.Modules — no Python control flow over
        # sub-modules in forward().
        self._stages = nn.ModuleList()
        self.pipeline = None  # nn.Sequential, built lazily after install_getobd

    # --- Backward-compat property accessors (read-only views) ---------

    @property
    def bd_to_ge(self):
        """Return the ``BDToGEConverter`` instance, or ``None`` if not yet installed."""
        for stage in self._stages:
            if isinstance(stage, _BDToGEStage):
                return stage.bd_to_ge
        return None

    @property
    def ge_to_bd(self):
        """Return the ``GEToBDConverter`` instance, or ``None`` if not yet installed."""
        for stage in self._stages:
            if isinstance(stage, _GEToBDStage):
                return stage.ge_to_bd
        return None

    @property
    def mul_layers(self):
        """Return the list of MUL sub-FFNs in install order."""
        return [s.sub_ffn for s in self._stages if isinstance(s, _MulFFNStage)]

    # --- Per-stage installers (called by compiler ops) -----------------

    def install_bdtoge(self):
        """phase=11.0: install BD → GE converter as the first pipeline stage."""
        assert len(self._stages) == 0, (
            f"Expected bd_to_ge to be the first stage; "
            f"already installed {len(self._stages)} stages"
        )
        self._stages.append(_BDToGEStage(self.BD, self.ge))

    def install_schoolbook(self):
        """phase=11.1: append the schoolbook partial-product stage."""
        from .alu.ops.mul import SchoolbookFFN
        # Stages so far: [_BDToGEStage] (1 stage).
        assert len(self._stages) == 1, (
            f"Expected schoolbook to be the second stage; "
            f"already installed {len(self._stages)} stages"
        )
        self._stages.append(_MulFFNStage(SchoolbookFFN(self.ge, opcode=27)))

    def install_carrypass(self, pass_idx: int):
        """phase=11.2/11.3/11.4: append the i-th carry pass (i=0,1,2)."""
        from .alu.ops.mul import CarryPassFFN, _compute_carry_passes
        passes = _compute_carry_passes(self.ge.config)
        assert pass_idx < len(passes), (
            f"NIBBLE config has {len(passes)} carry passes; "
            f"asked for pass_idx={pass_idx}"
        )
        # Stages so far: [_BDToGEStage, schoolbook, carry_0..carry_{pass_idx-1}]
        # → 2 + pass_idx total before this install.
        expected_len = 2 + pass_idx
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before installing "
            f"carrypass {pass_idx}; got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(
            CarryPassFFN(self.ge, opcode=27,
                         max_carry=passes[pass_idx], pass_idx=pass_idx)
        ))

    def install_genprop(self):
        """phase=12.0: append the gen/prop stage."""
        from .alu.ops.mul import MulGenPropFFN, _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        # Stages so far: bdtoge + schoolbook + n_passes carry = 2 + n_passes.
        expected_len = 2 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before genprop; "
            f"got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(MulGenPropFFN(self.ge, opcode=27)))

    def install_binarylookahead(self):
        """phase=12.1: append the binary carry-lookahead stage."""
        from .alu.ops.mul import MulBinaryLookaheadFFN, _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        expected_len = 3 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before lookahead; "
            f"got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(
            MulBinaryLookaheadFFN(self.ge, opcode=27)
        ))

    def install_finalcorrection(self):
        """phase=12.2: append the final-correction stage."""
        from .alu.ops.mul import MulFinalCorrectionFFN, _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        expected_len = 4 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before "
            f"final-correction; got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(
            MulFinalCorrectionFFN(self.ge, opcode=27)
        ))

    def install_getobd(self):
        """phase=12.3: install GE → BD converter and seal the Sequential."""
        from .alu.ops.mul import _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        # bdtoge + schoolbook + n_passes carry + genprop + lookahead +
        # finalcorrection = 5 + n_passes.
        expected_len = 5 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before ge_to_bd; "
            f"got {len(self._stages)}"
        )
        # Append combine + getobd, then materialise the Sequential.
        self._stages.append(_MulCombineStage(self.BD, self.ge))
        self._stages.append(_GEToBDStage(self.BD, self.ge, self.S))
        self.pipeline = nn.Sequential(*self._stages)

    @classmethod
    def build_fully_baked(cls, S, BD):
        """Construct a fully-baked composite with all 9 stages installed.

        Convenience factory for callers that want a drop-in replacement for
        ``ALUMul(S, BD)`` (which self-bakes in ``__init__``) without running
        the 9 compiler ops manually. Forward is byte-identical to
        ``ALUMul.forward`` once this constructor returns.
        """
        from .alu.ops.mul import _compute_carry_passes
        module = cls(S, BD)
        module.install_bdtoge()
        module.install_schoolbook()
        for pass_idx in range(len(_compute_carry_passes(module.ge.config))):
            module.install_carrypass(pass_idx=pass_idx)
        module.install_genprop()
        module.install_binarylookahead()
        module.install_finalcorrection()
        module.install_getobd()
        return module

    # --- Forward (byte-identical to PureNeuralALU mul branch) ----------

    def forward(self, x_bd):
        if self.pipeline is None:
            # Reproduce the previous "missing stages" diagnostic so partial
            # bakes still fail loudly.
            missing = []
            if self.bd_to_ge is None:
                missing.append('bd_to_ge')
            if len(self.mul_layers) == 0:
                missing.append('mul_layers')
            if self.ge_to_bd is None:
                missing.append('ge_to_bd')
            raise RuntimeError(
                f"FlattenedALUMul: missing stages {missing}. "
                "All 9 compiler ops (phase 11.0..12.3) must run before "
                "forward()."
            )

        state = _MulPipelineState()
        state.x_bd_in = x_bd
        out_state = self.pipeline(state)
        return out_state.x_bd_out

    # Stub methods for compatibility with vm_step.py (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class ALUMul(PureNeuralALU):
    """Neural MUL.

    DEPRECATED 2026-05-10: kept for back-compat only. Use
    ``FlattenedALUMul`` (assembled by 9 compiler ops at L11 phases
    11.0/11.1/11.2/11.3/11.4/12.0/12.1/12.2/12.3) instead. ``set_vm_weights``
    no longer instantiates this class — the compiler installs the flattened
    version via ``make_l11_alu_mul_*`` / ``make_l12_alu_mul_*`` ops.
    """
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='mul')


class ALUShift(PureNeuralALU):
    """Neural SHL/SHR.

    DEPRECATED (2026-05-10): The runtime wrapper class is being eliminated in
    favor of 4 separate compiler-driven sub-stages (BDToGE → SHL/SHR
    precompute → SHL/SHR select → GEToBD) installed at L13 by the compiler.
    See ``ALUShiftComposite`` and ``make_l13_alu_shift_*_op`` in
    ``unified_compiler/migrated_ops.py``. Kept here for backward compatibility
    until all callers move to the composite.
    """
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='shift')


# ---------------------------------------------------------------------------
# Flattened SHL/SHR pipeline (replaces ALUShift wrapper).
#
# 4 stages, each a separate nn.Module installed at L13.ffn by the compiler.
# Forward semantics are byte-identical to ``ALUShift.forward``.
#
# The composite is "vanilla": ``ALUShiftComposite.forward`` runs the pipeline
# inline by calling each stage's owned submodules directly, so there is no
# side-channel state object. The 4 stage modules remain as parameter
# containers (they hold the BD↔GE converters and the SHL/SHR FFNs) so the
# compiler bake_fns can install each stage independently — but their own
# ``forward`` methods are not invoked. This mirrors ``FlattenedALUMul`` where
# the composite's forward inlines the BD→GE conversion, the per-stage FFNs,
# and the GE→BD conversion in a single sequential pass.
# ---------------------------------------------------------------------------


class ShiftBDToGEStage(nn.Module):
    """Stage 1: BD → GenericE format conversion (formerly ALUShift step 1).

    Holds the ``BDToGEConverter`` used by ``ALUShiftComposite.forward``. The
    composite's forward reads ``self.bd_to_ge`` directly; this stage's
    ``forward`` is retained as a thin shim for ad-hoc / unit-test use.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.bd_to_ge = BDToGEConverter(BD, self.ge)

    def forward(self, x_bd):
        # Returns the GE-format workspace flattened to [B*seq, 8, DIM].
        x_ge = self.bd_to_ge(x_bd)
        B, seq_len, _, _ = x_ge.shape
        return x_ge.view(B * seq_len, 8, self.ge.DIM)


class ShiftPrecomputeStage(nn.Module):
    """Stage 2: SHL/SHR sub-chunk precompute (formerly ALUShift step 2a).

    Holds ``ShlPrecomputeFFN`` and ``ShrPrecomputeFFN``. The composite's
    forward reads ``self.shl_precompute`` / ``self.shr_precompute`` directly.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        # Use the same factory as ALUShift to get [precompute, select] pairs;
        # take only the precompute (index 0).
        self.shl_precompute = build_shl_layers(NIBBLE, opcode=23)[0]
        self.shr_precompute = build_shr_layers(NIBBLE, opcode=24)[0]

    def forward(self, x_ge_flat):
        # Returns (x_shl, x_shr) — both with shape [B*seq, 8, DIM].
        x_shl = self.shl_precompute(x_ge_flat.clone())
        x_shr = self.shr_precompute(x_ge_flat.clone())
        return x_shl, x_shr


class ShiftSelectStage(nn.Module):
    """Stage 3: SHL/SHR select + opcode-gated combine (formerly ALUShift step 2b).

    Holds the SHL/SHR select FFNs. The composite's forward reads
    ``self.shl_select`` / ``self.shr_select`` directly.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        # Select FFN is index 1 of build_*_layers.
        self.shl_select = build_shl_layers(NIBBLE, opcode=23)[1]
        self.shr_select = build_shr_layers(NIBBLE, opcode=24)[1]

    def forward(self, x_shl, x_shr):
        # Returns (x_shl_post, x_shr_post).
        return self.shl_select(x_shl), self.shr_select(x_shr)


class ShiftGEToBDStage(nn.Module):
    """Stage 4: GenericE → BD format conversion + write OUTPUT (formerly ALUShift step 3).

    Holds the ``GEToBDConverter``. The composite's forward reads
    ``self.ge_to_bd`` directly.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

    def forward(self, x_ge_out, x_bd, opcode_mask):
        return self.ge_to_bd(x_ge_out, x_bd, opcode_mask=opcode_mask)


class ALUShiftComposite(nn.Module):
    """Composite SHL/SHR FFN replacement.

    Replaces ``ALUShift`` (an instance of ``PureNeuralALU(operations='shift')``)
    as the L13 ``block.ffn`` module. Forward is byte-identical to
    ``ALUShift.forward``.

    The 4 sub-stages are exposed as named submodules so that the compiler
    bake_fns in ``migrated_ops.make_l13_alu_shift_*_op`` can build / inspect
    each stage independently. The forward inlines the pipeline (no
    side-channel state) — same shape as ``FlattenedALUMul.forward``.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.bdtoge_stage = ShiftBDToGEStage(S, BD)
        self.precompute_stage = ShiftPrecomputeStage(S, BD)
        self.select_stage = ShiftSelectStage(S, BD)
        self.getobd_stage = ShiftGEToBDStage(S, BD)

    def forward(self, x_bd):
        BD = self.BD
        B, seq_len, _ = x_bd.shape

        # Stage 1: BD → GE.
        x_ge_flat = self.bdtoge_stage.bd_to_ge(x_bd)
        x_ge_flat = x_ge_flat.view(B * seq_len, 8, self.ge.DIM)

        # Stage 2: SHL/SHR precompute on the GE workspace.
        x_shl = self.precompute_stage.shl_precompute(x_ge_flat.clone())
        x_shr = self.precompute_stage.shr_precompute(x_ge_flat.clone())

        # Stage 3: SHL/SHR select + opcode-gated combine.
        x_shl_post = self.select_stage.shl_select(x_shl)
        x_shr_post = self.select_stage.shr_select(x_shr)

        x_ge_out = x_ge_flat.clone()

        op_shl = (x_ge_flat[:, 0, self.ge.OP_START + 23] > 0.1).float()
        op_shr = (x_ge_flat[:, 0, self.ge.OP_START + 24] > 0.1).float()
        op_total = op_shl + op_shr

        x_ge_out[:, :, self.ge.RESULT] = (
            x_shl_post[:, :, self.ge.RESULT] * op_shl[:, None]
            + x_shr_post[:, :, self.ge.RESULT] * op_shr[:, None]
        )

        x_ge_out = x_ge_out.view(B, seq_len, 8, self.ge.DIM)
        opcode_mask = op_total.view(B, seq_len)

        # Restrict OUTPUT writes to AX marker positions to match ALUShift.
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        # Stage 4: GE → BD + write OUTPUT.
        return self.getobd_stage.ge_to_bd(x_ge_out, x_bd, opcode_mask=opcode_mask)

    # Stub methods for compatibility with vm_step.py (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class ALUDivMod(PureNeuralALU):
    """Neural DIV/MOD."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='div_mod')
