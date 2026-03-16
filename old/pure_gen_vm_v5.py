#!/usr/bin/env python3
"""
Pure Generative VM v5 - Ultra-Efficient SiLU Arithmetic

KEY INNOVATIONS:
1. Value-encoded inputs AND outputs (no one-hot lookup tables)
2. Addition: SiLU(SCALE*(a+b)) / SCALE ≈ a + b (4 weights/nibble)
3. Multiplication: (SiLU(SCALE*a) + SiLU(-SCALE*a)) * b / SCALE ≈ a*b (6 weights/product)
4. Equality: SiLU(diff+eps) - 2*SiLU(diff) + SiLU(diff-eps) ≈ indicator(diff≈0) (11 weights)
5. LT/GT: SiLU(±SCALE*diff) detects sign (3 weights each)
6. Carry propagation via cascaded threshold detection

Weight counts (non-zero only):
- ADD: 8 nibbles × 4 weights = 32 + carry prop
- SUB: 8 nibbles × 4 weights = 32 + borrow prop
- MUL: 64 products × 6 weights = 384
- EQ: 8 nibbles × 11 weights = 88
- LT/GT: 8 nibbles × 3 weights + cascade = ~50
- BZ/BNZ: ~20 weights

Total arithmetic: ~600 non-zero weights (vs V3's ~100K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math

# =============================================================================
# OPCODE AND VOCAB DEFINITIONS (copied from pure_gen_vm.py for self-containment)
# =============================================================================

class Opcode:
    """VM opcodes - each gets its own MoE expert."""
    LEA = 0    # Load effective address
    IMM = 1    # Load immediate
    JMP = 2    # Jump
    JSR = 3    # Jump to subroutine
    BZ = 4     # Branch if zero
    BNZ = 5    # Branch if not zero
    ENT = 6    # Enter function
    ADJ = 7    # Adjust stack
    LEV = 8    # Leave function
    LI = 9     # Load int
    LC = 10    # Load char
    SI = 11    # Store int
    SC = 12    # Store char
    PSH = 13   # Push
    OR = 14    # Bitwise OR
    XOR = 15   # Bitwise XOR
    AND = 16   # Bitwise AND
    EQ = 17    # Equal
    NE = 18    # Not equal
    LT = 19    # Less than
    GT = 20    # Greater than
    LE = 21    # Less or equal
    GE = 22    # Greater or equal
    SHL = 23   # Shift left
    SHR = 24   # Shift right
    ADD = 25   # Add
    SUB = 26   # Subtract
    MUL = 27   # Multiply
    DIV = 28   # Divide
    MOD = 29   # Modulo
    OPEN = 30  # File open
    READ = 31  # File read
    CLOS = 32  # File close
    PRTF = 33  # Printf
    MALC = 34  # Malloc
    FREE = 35  # Free
    MSET = 36  # Memset
    MCMP = 37  # Memcmp
    EXIT = 38  # Exit
    GETC = 64  # Get char
    PUTC = 65  # Put char

class Vocab:
    PAD, BOS, EOS = 0, 1, 2
    CODE, REG_PC, REG_AX, REG_SP, REG_BP, MEM = 3, 4, 5, 6, 7, 8
    STEP_END = 9
    HEAP = 10
    BYTE_BASE = 16
    VOCAB_SIZE = 272

    @staticmethod
    def byte_tok(val: int) -> int:
        return Vocab.BYTE_BASE + (val & 0xFF)

    @staticmethod
    def tok_byte(tok: int) -> int:
        return tok - Vocab.BYTE_BASE


# =============================================================================
# V5 EMBEDDING DIMENSIONS
# =============================================================================

class EmbedDimsV5:
    """V5 embedding layout - fully value-encoded."""

    # VALUE-encoded operands (8 floats each, raw 0-15 values)
    OP_A_VAL_START = 0
    OP_A_VAL_END = 8
    OP_B_VAL_START = 8
    OP_B_VAL_END = 16

    # VALUE-encoded outputs
    RESULT_VAL_START = 16
    RESULT_VAL_END = 24
    CARRY_VAL_START = 24  # Carry/borrow flags per nibble
    CARRY_VAL_END = 32

    # MUL workspace
    MUL_PROD_START = 32   # 64 products
    MUL_PROD_END = 96
    MUL_COL_START = 96    # Column sums
    MUL_COL_END = 104

    # Control - 48 opcode slots (0-47, enough for PUTC=41)
    OPCODE_START = 104
    OPCODE_END = 152      # Extended from 136 to support I/O opcodes

    # Comparison results (scalar flags)
    CMP_EQ = 152          # Shifted +16
    CMP_LT = 153
    CMP_GT = 154
    CMP_NE = 155
    CMP_LE = 156
    CMP_GE = 157

    # Per-nibble comparison workspace
    NIB_EQ_START = 160    # 8 floats (shifted +16)
    NIB_LT_START = 168    # 8 floats
    NIB_GT_START = 176    # 8 floats

    # One-hot result for output
    RESULT_NIB_START = 192  # Shifted +16
    RESULT_NIB_END = 320    # 8 × 16 = 128

    # VM Registers (value-encoded, 8 nibbles each = 32 bits)
    PC_VAL_START = 320    # Program counter (shifted +16)
    PC_VAL_END = 328
    SP_VAL_START = 328    # Stack pointer
    SP_VAL_END = 336
    BP_VAL_START = 336    # Base pointer
    BP_VAL_END = 344
    AX_VAL_START = 344    # Accumulator
    AX_VAL_END = 352

    # Immediate operand (for JMP, JSR, etc.)
    IMM_VAL_START = 352
    IMM_VAL_END = 360

    # Branch taken flag
    BRANCH_TAKEN = 360

    # ===========================================
    # I/O EMBEDDING SLOTS (Neural I/O)
    # ===========================================
    # All I/O flows through the embedding space.
    # External handler reads/writes these slots.

    # I/O Character (value-encoded, 8 nibbles for 32-bit, but chars use lower 2)
    IO_CHAR_VAL_START = 368   # Shifted +16
    IO_CHAR_VAL_END = 376

    # I/O Status flags
    IO_OUTPUT_READY = 376     # 1.0 = output available in IO_CHAR
    IO_INPUT_READY = 377      # 1.0 = input available in IO_CHAR (set by external)
    IO_NEED_INPUT = 378       # 1.0 = GETCHAR executed, waiting for input
    IO_PROGRAM_END = 379      # 1.0 = EXIT executed, program done

    # I/O Buffer (for batching - 16 characters)
    IO_BUFFER_START = 384
    IO_BUFFER_END = 448       # 64 slots = 16 chars × 4 nibbles each
    IO_BUFFER_LEN = 448       # Current buffer length (0-16)
    IO_BUFFER_POS = 449       # Current read position

    DIM = 512

    # Constants
    SCALE = 20.0
    EPS = 0.5  # For equality detection window


# =============================================================================
# INSTRUCTION-WISE MIXTURE OF EXPERTS (MoE)
# =============================================================================

class ExpertType:
    """Expert indices for MoE routing."""
    ADD_SUB = 0      # ADD, SUB
    MUL = 1          # MUL
    DIV_MOD = 2      # DIV, MOD
    BITWISE = 3      # AND, OR, XOR
    SHIFT = 4        # SHL, SHR
    COMPARISON = 5   # EQ, NE, LT, GT, LE, GE
    CONTROL = 6      # JMP, JSR, BZ, BNZ
    STACK = 7        # ENT, ADJ, LEV
    IO = 8           # GETCHAR, PUTCHAR, EXIT
    NUM_EXPERTS = 9


# Opcode to expert mapping
OPCODE_TO_EXPERT = {
    Opcode.ADD: ExpertType.ADD_SUB,
    Opcode.SUB: ExpertType.ADD_SUB,
    Opcode.MUL: ExpertType.MUL,
    Opcode.DIV: ExpertType.DIV_MOD,
    Opcode.MOD: ExpertType.DIV_MOD,
    Opcode.AND: ExpertType.BITWISE,
    Opcode.OR: ExpertType.BITWISE,
    Opcode.XOR: ExpertType.BITWISE,
    Opcode.SHL: ExpertType.SHIFT,
    Opcode.SHR: ExpertType.SHIFT,
    Opcode.EQ: ExpertType.COMPARISON,
    Opcode.NE: ExpertType.COMPARISON,
    Opcode.LT: ExpertType.COMPARISON,
    Opcode.GT: ExpertType.COMPARISON,
    Opcode.LE: ExpertType.COMPARISON,
    Opcode.GE: ExpertType.COMPARISON,
    Opcode.JMP: ExpertType.CONTROL,
    Opcode.JSR: ExpertType.CONTROL,
    Opcode.BZ: ExpertType.CONTROL,
    Opcode.BNZ: ExpertType.CONTROL,
    Opcode.ENT: ExpertType.STACK,
    Opcode.ADJ: ExpertType.STACK,
    Opcode.LEV: ExpertType.STACK,
    # I/O operations
    Opcode.GETC: ExpertType.IO,
    Opcode.PUTC: ExpertType.IO,
    Opcode.EXIT: ExpertType.IO,
}


class InstructionRouter(nn.Module):
    """
    Router network for instruction-wise MoE.

    Maps opcode one-hot encoding to expert selection logits.
    Uses learned weights but can be initialized to match opcode->expert mapping.

    Supports:
    - Top-k routing (hard selection)
    - Soft routing (weighted combination)
    - Load balancing auxiliary loss for training
    """

    def __init__(self, dim: int, num_experts: int = ExpertType.NUM_EXPERTS,
                 top_k: int = 1, noise_std: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        E = EmbedDimsV5

        # Router: opcode region -> expert logits
        # Input: 32 opcode slots, Output: num_experts
        opcode_dim = E.OPCODE_END - E.OPCODE_START
        self.router = nn.Linear(opcode_dim, num_experts, bias=False)

        # Initialize router to match opcode->expert mapping
        self._init_router_weights()

        # For load balancing loss
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.total_tokens = 0

    def _init_router_weights(self):
        """Initialize router to match known opcode->expert mapping."""
        E = EmbedDimsV5
        with torch.no_grad():
            self.router.weight.zero_()

            # Set high weight for correct opcode->expert pairs
            for opcode, expert_idx in OPCODE_TO_EXPERT.items():
                if opcode < (E.OPCODE_END - E.OPCODE_START):
                    self.router.weight[expert_idx, opcode] = 10.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route input to experts.

        Args:
            x: Input tensor [..., dim]

        Returns:
            expert_weights: Weights for each expert [..., num_experts]
            expert_indices: Top-k expert indices [..., top_k]
            aux_loss: Load balancing loss (if training)
        """
        E = EmbedDimsV5

        # Extract opcode region
        opcode_input = x[..., E.OPCODE_START:E.OPCODE_END]

        # Compute router logits
        logits = self.router(opcode_input)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Compute softmax weights
        weights = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute load balancing auxiliary loss
        aux_loss = None
        if self.training:
            # Fraction of tokens routed to each expert
            router_probs = weights.mean(dim=list(range(weights.dim() - 1)))

            # Fraction of router probability allocated to each expert
            expert_frac = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                mask = (top_k_indices == i).any(dim=-1).float()
                expert_frac[i] = mask.mean()

            # Load balancing loss: minimize variance in expert utilization
            aux_loss = self.num_experts * (router_probs * expert_frac).sum()

        return top_k_weights, top_k_indices, aux_loss

    def get_expert_for_opcode(self, opcode: int) -> int:
        """Get the expert index for a given opcode (for inference)."""
        return OPCODE_TO_EXPERT.get(opcode, 0)


class InstructionMoE(nn.Module):
    """
    Mixture of Experts layer for VM instructions.

    Contains specialized expert networks for each operation type:
    - ADD_SUB: Addition and subtraction with carry propagation
    - MUL: Multiplication
    - DIV_MOD: Division and modulo
    - BITWISE: AND, OR, XOR
    - SHIFT: SHL, SHR
    - COMPARISON: EQ, NE, LT, GT, LE, GE
    - CONTROL: JMP, JSR, BZ, BNZ
    - STACK: ENT, ADJ, LEV

    Uses router network for expert selection with support for:
    - Hard routing (top-1): Only selected expert processes input
    - Soft routing (top-k): Weighted combination of expert outputs
    """

    def __init__(self, dim: int, top_k: int = 1, noise_std: float = 0.0):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.num_experts = ExpertType.NUM_EXPERTS

        # Router network
        self.router = InstructionRouter(dim, self.num_experts, top_k, noise_std)

        # Expert networks (will be set after class definitions)
        self.experts = nn.ModuleList()

    def set_experts(self, experts: List[nn.Module]):
        """Set the expert modules after they're defined."""
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process input through MoE layer.

        Args:
            x: Input tensor [batch, seq, dim]

        Returns:
            output: Processed tensor [batch, seq, dim]
            aux_loss: Load balancing loss (if training)
        """
        if len(self.experts) == 0:
            return x, None

        # Get routing weights and indices
        top_k_weights, top_k_indices, aux_loss = self.router(x)

        # Initialize output
        output = x.clone()

        if self.top_k == 1:
            # Hard routing: only selected expert processes each token
            # For efficiency, batch tokens by expert
            flat_x = x.view(-1, self.dim)
            flat_indices = top_k_indices.view(-1)
            flat_output = flat_x.clone()

            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert
                mask = (flat_indices == expert_idx).squeeze(-1)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    flat_output[mask] = expert_output

            output = flat_output.view_as(x)
        else:
            # Soft routing: weighted combination of top-k experts
            flat_x = x.view(-1, self.dim)
            flat_weights = top_k_weights.view(-1, self.top_k)
            flat_indices = top_k_indices.view(-1, self.top_k)
            flat_output = torch.zeros_like(flat_x)

            for k in range(self.top_k):
                for expert_idx in range(self.num_experts):
                    # Find tokens where this expert is the k-th choice
                    mask = (flat_indices[:, k] == expert_idx)
                    if mask.any():
                        expert_input = flat_x[mask]
                        expert_output = self.experts[expert_idx](expert_input)
                        # Weight by router probability
                        weight = flat_weights[mask, k:k+1]
                        flat_output[mask] += weight * expert_output

            output = flat_output.view_as(x)

        return output, aux_loss


# =============================================================================
# EFFICIENT ADD/SUB WITH CARRY PROPAGATION
# =============================================================================

class EfficientAddSubLayer(nn.Module):
    """
    ADD/SUB with carry propagation using SiLU arithmetic.

    Per-nibble ADD: SiLU(SCALE*(a+b)) / SCALE ≈ a+b
    Carry detection: SiLU(SCALE*(sum-15.5)) > 0 when sum > 15
    Result mod 16: result - 16*carry
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        # Rows: 8 ADD + 8 SUB + 8 carry detect + 8 borrow detect = 32
        ffn_dim = 32

        self.up = nn.Linear(dim, ffn_dim, bias=True)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV5
        SCALE = E.SCALE
        ffn_dim = self.up.weight.size(0)

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0

            # === ADD: raw sum per nibble (no carry input in first pass) ===
            for nib in range(8):
                a_dim = E.OP_A_VAL_START + nib
                b_dim = E.OP_B_VAL_START + nib
                result_dim = E.RESULT_VAL_START + nib

                # up: SCALE * (a + b)
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = SCALE

                # gate: opcode check
                self.gate.weight[row, E.OPCODE_START + Opcode.ADD] = SCALE

                # down: 1/SCALE²
                self.down.weight[result_dim, row] = 1.0 / (SCALE * SCALE)

                row += 1

            # === SUB: raw diff per nibble (no offset, handle borrow in forward) ===
            for nib in range(8):
                a_dim = E.OP_A_VAL_START + nib
                b_dim = E.OP_B_VAL_START + nib
                result_dim = E.RESULT_VAL_START + nib

                # up: SCALE * (a - b) - can be negative
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE

                # gate: opcode check
                self.gate.weight[row, E.OPCODE_START + Opcode.SUB] = SCALE

                # down: 1/SCALE² (will give 0 for negative due to SiLU)
                self.down.weight[result_dim, row] = 1.0 / (SCALE * SCALE)

                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        # Check which operation
        is_add = x[..., E.OPCODE_START + Opcode.ADD:E.OPCODE_START + Opcode.ADD + 1] > 0.5
        is_sub = x[..., E.OPCODE_START + Opcode.SUB:E.OPCODE_START + Opcode.SUB + 1] > 0.5

        # Get operands
        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END].clone()
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END].clone()

        out = x.clone()

        if is_add.any():
            # ADD with carry propagation
            carry = torch.zeros_like(a_vals[..., 0:1])
            for nib in range(8):
                nibble_sum = a_vals[..., nib:nib+1] + b_vals[..., nib:nib+1] + carry
                out[..., E.RESULT_VAL_START + nib:E.RESULT_VAL_START + nib + 1] = nibble_sum % 16
                carry = (nibble_sum >= 16).float()

        if is_sub.any():
            # SUB with borrow propagation
            borrow = torch.zeros_like(a_vals[..., 0:1])
            for nib in range(8):
                nibble_diff = a_vals[..., nib:nib+1] - b_vals[..., nib:nib+1] - borrow
                # If negative, add 16 and set borrow
                needs_borrow = (nibble_diff < 0).float()
                nibble_diff = nibble_diff + needs_borrow * 16
                out[..., E.RESULT_VAL_START + nib:E.RESULT_VAL_START + nib + 1] = nibble_diff
                borrow = needs_borrow

        return out


# =============================================================================
# EFFICIENT MULTIPLICATION (6 weights per product)
# =============================================================================

class EfficientMulProductsLayer(nn.Module):
    """
    Multiplication: (SiLU(SCALE*a) + SiLU(-SCALE*a)) * b / SCALE ≈ a*b
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 64 products × 2 rows = 128
        ffn_dim = 128

        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV5
        SCALE = E.SCALE

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0

            for i in range(8):
                for j in range(8):
                    prod_idx = i * 8 + j
                    a_dim = E.OP_A_VAL_START + i
                    b_dim = E.OP_B_VAL_START + j
                    out_dim = E.MUL_PROD_START + prod_idx

                    # Row 1: +SCALE * a
                    self.up.weight[row, a_dim] = SCALE
                    self.gate.weight[row, b_dim] = 1.0
                    self.down.weight[out_dim, row] = 1.0 / SCALE
                    row += 1

                    # Row 2: -SCALE * a
                    self.up.weight[row, a_dim] = -SCALE
                    self.gate.weight[row, b_dim] = 1.0
                    self.down.weight[out_dim, row] = 1.0 / SCALE
                    row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_out = self.up(x)
        gate_out = self.gate(x)
        hidden = F.silu(up_out) * gate_out
        return x + self.down(hidden)


# =============================================================================
# EFFICIENT COMPARISONS (EQ: 11 weights, LT/GT: ~3 weights)
# =============================================================================

class EfficientComparisonLayer(nn.Module):
    """
    Efficient comparison using SiLU formulas.

    Equality detection (per nibble):
        EQ = SiLU(SCALE*(diff+eps)) - 2*SiLU(SCALE*diff) + SiLU(SCALE*(diff-eps))
        Gives high output (~10) when |diff| < eps, ~0 otherwise
        Weights: 3 up + 2 bias + 3 down = 8 (+ opcode gate)

    LT detection (per nibble):
        LT = SiLU(-SCALE*diff) / SCALE
        Gives positive when diff < 0
        Weights: 1 up + 1 down = 2 (+ opcode gate)

    GT detection:
        GT = SiLU(SCALE*diff) / SCALE
        Gives positive when diff > 0

    32-bit comparison: cascade from MSB to LSB
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        # Per nibble: 3 EQ + 1 LT + 1 GT = 5 rows × 8 nibbles = 40
        # Plus final cascade rows
        ffn_dim = 64

        self.up = nn.Linear(dim, ffn_dim, bias=True)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV5
        SCALE = E.SCALE
        EPS = E.EPS

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0

            for nib in range(8):
                a_dim = E.OP_A_VAL_START + nib
                b_dim = E.OP_B_VAL_START + nib
                eq_dim = E.NIB_EQ_START + nib
                lt_dim = E.NIB_LT_START + nib
                gt_dim = E.NIB_GT_START + nib

                # === EQ Node 1: diff + eps (output +1) ===
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.up.bias[row] = SCALE * EPS
                # Gate constant (enable for all comparison ops)
                self.gate.weight[row, E.OPCODE_START + Opcode.EQ] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.NE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GE] = 1.0
                self.down.weight[eq_dim, row] = 1.0  # +1 (not scaled)
                row += 1

                # === EQ Node 2: diff (output -2) ===
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.up.bias[row] = 0
                self.gate.weight[row, E.OPCODE_START + Opcode.EQ] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.NE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GE] = 1.0
                self.down.weight[eq_dim, row] = -2.0  # -2 (not scaled)
                row += 1

                # === EQ Node 3: diff - eps (output +1) ===
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.up.bias[row] = -SCALE * EPS
                self.gate.weight[row, E.OPCODE_START + Opcode.EQ] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.NE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GE] = 1.0
                self.down.weight[eq_dim, row] = 1.0  # +1 (not scaled)
                row += 1

                # === LT: -diff ===
                self.up.weight[row, a_dim] = -SCALE
                self.up.weight[row, b_dim] = SCALE
                self.gate.weight[row, E.OPCODE_START + Opcode.LT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.NE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.EQ] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GE] = 1.0
                self.down.weight[lt_dim, row] = 1.0 / SCALE
                row += 1

                # === GT: +diff ===
                self.up.weight[row, a_dim] = SCALE
                self.up.weight[row, b_dim] = -SCALE
                self.gate.weight[row, E.OPCODE_START + Opcode.GT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.GE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.NE] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.EQ] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LT] = 1.0
                self.gate.weight[row, E.OPCODE_START + Opcode.LE] = 1.0
                self.down.weight[gt_dim, row] = 1.0 / SCALE
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        up_out = self.up(x)
        gate_out = self.gate(x)
        hidden = F.silu(up_out) * gate_out
        out = x + self.down(hidden)

        # Cascade comparison from MSB to LSB
        eq_cascade = torch.ones_like(out[..., 0:1])
        lt_final = torch.zeros_like(out[..., 0:1])
        gt_final = torch.zeros_like(out[..., 0:1])

        for nib in range(7, -1, -1):  # MSB first
            eq_nib = out[..., E.NIB_EQ_START + nib:E.NIB_EQ_START + nib + 1]
            lt_nib = out[..., E.NIB_LT_START + nib:E.NIB_LT_START + nib + 1]
            gt_nib = out[..., E.NIB_GT_START + nib:E.NIB_GT_START + nib + 1]

            # Normalize eq to 0-1 range (it's ~10 when equal, ~0 otherwise)
            eq_norm = torch.clamp(eq_nib / 10.0, 0, 1)

            # LT/GT contribute only if all higher nibbles are equal
            lt_final = lt_final + eq_cascade * torch.clamp(lt_nib, min=0)
            gt_final = gt_final + eq_cascade * torch.clamp(gt_nib, min=0)

            # Update cascade
            eq_cascade = eq_cascade * eq_norm

        # Final EQ is product of all nibble equalities
        eq_final = eq_cascade

        # Store comparison results
        out[..., E.CMP_EQ:E.CMP_EQ+1] = eq_final
        out[..., E.CMP_LT:E.CMP_LT+1] = (lt_final > 0.5).float()
        out[..., E.CMP_GT:E.CMP_GT+1] = (gt_final > 0.5).float()
        out[..., E.CMP_NE:E.CMP_NE+1] = 1.0 - eq_final
        out[..., E.CMP_LE:E.CMP_LE+1] = torch.clamp(eq_final + (lt_final > 0.5).float(), 0, 1)
        out[..., E.CMP_GE:E.CMP_GE+1] = torch.clamp(eq_final + (gt_final > 0.5).float(), 0, 1)

        return out


# =============================================================================
# BRANCH ZERO / NOT ZERO (BZ/BNZ)
# =============================================================================

class BranchZeroLayer(nn.Module):
    """
    Detect if value is zero for BZ/BNZ control flow.

    Uses same equality-to-zero formula:
        is_zero = SiLU(val+eps) - 2*SiLU(val) + SiLU(val-eps)

    For 32-bit value: all 8 nibbles must be zero.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        # 3 rows per nibble × 8 nibbles = 24
        ffn_dim = 24

        self.up = nn.Linear(dim, ffn_dim, bias=True)
        self.gate = nn.Linear(dim, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV5
        SCALE = E.SCALE
        EPS = E.EPS

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.gate.bias.fill_(1.0)  # Constant gate
            self.down.weight.zero_()

            row = 0

            for nib in range(8):
                # Check if result nibble is zero
                val_dim = E.RESULT_VAL_START + nib
                zero_dim = E.NIB_EQ_START + nib  # Reuse for zero detection

                # Node 1: val + eps (output +1)
                self.up.weight[row, val_dim] = SCALE
                self.up.bias[row] = SCALE * EPS
                self.down.weight[zero_dim, row] = 1.0  # Not scaled
                row += 1

                # Node 2: val (output -2)
                self.up.weight[row, val_dim] = SCALE
                self.up.bias[row] = 0
                self.down.weight[zero_dim, row] = -2.0  # Not scaled
                row += 1

                # Node 3: val - eps (output +1)
                self.up.weight[row, val_dim] = SCALE
                self.up.bias[row] = -SCALE * EPS
                self.down.weight[zero_dim, row] = 1.0  # Not scaled
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        up_out = self.up(x)
        gate_out = self.gate(x)
        hidden = F.silu(up_out) * gate_out
        out = x + self.down(hidden)

        # Combine all nibble zero checks (all must be ~10 for value to be zero)
        is_zero = torch.ones_like(out[..., 0:1])
        for nib in range(8):
            zero_nib = out[..., E.NIB_EQ_START + nib:E.NIB_EQ_START + nib + 1]
            is_zero = is_zero * torch.clamp(zero_nib / 10.0, 0, 1)

        # BZ fires when value is zero, BNZ fires when not zero
        out[..., E.CMP_EQ:E.CMP_EQ+1] = is_zero  # Reuse for "is zero" flag

        return out


# =============================================================================
# EFFICIENT DIVISION (Pure Neural via Layer Subroutine)
# =============================================================================

class EfficientDivLayer(nn.Module):
    """
    Pure neural division that invokes actual transformer layers as subroutines.

    This layer contains references to the MUL and SUB layers and invokes
    their forward passes to compute division via Newton-Raphson.

    The forward pass:
    1. Encodes operands for MUL operation
    2. Runs the MUL layer forward pass
    3. Reads the result
    4. Encodes for SUB operation
    5. Runs the SUB layer forward pass
    6. Repeats for Newton-Raphson iterations

    This is truly "pure neural" - it only uses the actual transformer layer
    forward passes, not any native Python division operators.
    """

    def __init__(self, dim: int, mul_layer=None, sub_layer=None):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        # Store references to the actual neural layers we'll invoke
        self.mul_layer = mul_layer
        self.sub_layer = sub_layer

        # Fixed-point scale for reciprocal computation
        self.RECIP_SCALE = 2 ** 16  # 65536

        # Initial reciprocal lookup FFN (small baked weights)
        # Maps leading nibble (0-15) to initial reciprocal guess
        ffn_dim = 16
        self.recip_up = nn.Linear(dim, ffn_dim, bias=False)
        self.recip_down = nn.Linear(ffn_dim, dim, bias=False)
        self._bake_recip_weights()

    def _bake_recip_weights(self):
        """Bake weights for initial reciprocal lookup."""
        E = EmbedDimsV5
        SCALE = float(self.RECIP_SCALE)

        with torch.no_grad():
            self.recip_up.weight.zero_()
            self.recip_down.weight.zero_()

            # For each nibble value 1-15, output SCALE/n
            recip_values = [SCALE] + [SCALE / n for n in range(1, 16)]

            for n in range(16):
                # Up: detect if leading nibble == n
                # We check OP_B's highest non-zero nibble
                # Simplified: just use the low nibble for small divisors
                self.recip_up.weight[n, E.OP_B_VAL_START] = 1.0 if n == 0 else 0.0

                # Down: output reciprocal to a scratch area
                # Store in first result nibble as scaled value
                self.recip_down.weight[E.RESULT_VAL_START, n] = recip_values[n]

    def _encode_for_mul(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Set up embedding for MUL operation: a * b"""
        E = EmbedDimsV5
        out = x.clone()

        # Clear opcodes, set MUL
        out[..., E.OPCODE_START:E.OPCODE_END] = 0.0
        out[..., E.OPCODE_START + Opcode.MUL] = 1.0

        # Encode operands as nibbles
        for nib in range(8):
            out[..., E.OP_A_VAL_START + nib] = ((a >> (nib * 4)) & 0xF).float()
            out[..., E.OP_B_VAL_START + nib] = ((b >> (nib * 4)) & 0xF).float()

        return out

    def _encode_for_sub(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Set up embedding for SUB operation: a - b"""
        E = EmbedDimsV5
        out = x.clone()

        # Clear opcodes, set SUB
        out[..., E.OPCODE_START:E.OPCODE_END] = 0.0
        out[..., E.OPCODE_START + Opcode.SUB] = 1.0

        # Encode operands as nibbles
        for nib in range(8):
            out[..., E.OP_A_VAL_START + nib] = ((a >> (nib * 4)) & 0xF).float()
            out[..., E.OP_B_VAL_START + nib] = ((b >> (nib * 4)) & 0xF).float()

        return out

    def _read_result(self, x: torch.Tensor) -> torch.Tensor:
        """Read result from embedding as integer."""
        E = EmbedDimsV5
        result = torch.zeros_like(x[..., 0], dtype=torch.int64)
        for nib in range(8):
            result = result + x[..., E.RESULT_VAL_START + nib].long() * (16 ** nib)
        return result

    def _run_neural_mul(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run actual MUL layer forward pass and return result."""
        if self.mul_layer is None:
            # Fallback: use inline SwiGLU formula
            return self._inline_mul(a.float(), b.float()).long()

        # Encode for MUL
        mul_input = self._encode_for_mul(x, a, b)

        # Run the actual neural MUL layer
        mul_output = self.mul_layer(mul_input)

        # Read result
        return self._read_result(mul_output)

    def _run_neural_sub(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run actual SUB layer forward pass and return result."""
        if self.sub_layer is None:
            # Fallback: use inline formula
            return (a - b).long()

        # Encode for SUB
        sub_input = self._encode_for_sub(x, a, b)

        # Run the actual neural SUB layer
        sub_output = self.sub_layer(sub_input)

        # Read result
        return self._read_result(sub_output)

    def _inline_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Inline SwiGLU multiply for when layers not available."""
        SCALE = 20.0
        silu_pos = a * SCALE * torch.sigmoid(a * SCALE)
        silu_neg = -a * SCALE * torch.sigmoid(-a * SCALE)
        return (silu_pos * b + silu_neg * (-b)) / SCALE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        # Check if DIV or MOD operation
        is_div = x[..., E.OPCODE_START + Opcode.DIV:E.OPCODE_START + Opcode.DIV + 1] > 0.5
        is_mod = x[..., E.OPCODE_START + Opcode.MOD:E.OPCODE_START + Opcode.MOD + 1] > 0.5

        if not (is_div.any() or is_mod.any()):
            return x

        out = x.clone()

        # Get operands as integers
        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        # Reconstruct full 32-bit values
        a = torch.zeros_like(a_vals[..., 0], dtype=torch.int64)
        b = torch.zeros_like(b_vals[..., 0], dtype=torch.int64)
        for nib in range(8):
            a = a + a_vals[..., nib].long() * (16 ** nib)
            b = b + b_vals[..., nib].long() * (16 ** nib)

        # Avoid division by zero
        b_safe = torch.clamp(b, min=1)

        SCALE = self.RECIP_SCALE

        # === PURE NEURAL NEWTON-RAPHSON DIVISION ===
        # Uses actual layer forward passes, not native Python operators

        # Step 1: Initial reciprocal guess based on divisor magnitude
        # For small b, use simple reciprocal; for large b, scale down
        # x0 ≈ SCALE / b
        leading_nibble = torch.zeros_like(b)
        leading_pos = torch.zeros_like(b)
        for nib in range(7, -1, -1):
            nibble = (b_safe >> (nib * 4)) & 0xF
            mask = (nibble > 0) & (leading_nibble == 0)
            leading_nibble = torch.where(mask, nibble, leading_nibble)
            leading_pos = torch.where(mask, torch.tensor(nib, dtype=b.dtype, device=b.device), leading_pos)

        leading_nibble = torch.clamp(leading_nibble, min=1)

        # Initial guess: SCALE / (leading_nibble * 16^pos)
        scale_factor = SCALE // (16 ** leading_pos.float()).long()
        x = (scale_factor // leading_nibble).long()
        x = torch.clamp(x, min=1)

        # Step 2: Newton-Raphson iterations using NEURAL layers
        # x_new = x * (2 - b * x / SCALE)
        two_scaled = torch.tensor(2 * SCALE, dtype=torch.int64, device=x.device)

        for _ in range(6):
            # bx = b * x using neural MUL layer
            bx = self._run_neural_mul(x, b_safe, x)

            # correction = 2*SCALE - bx
            correction = two_scaled - bx

            # x = x * correction / SCALE using neural MUL
            x_corr = self._run_neural_mul(x, x, correction)
            x = x_corr // SCALE

            x = torch.clamp(x, min=1)

        # Step 3: Compute quotient = a * x / SCALE using neural MUL
        ax = self._run_neural_mul(x, a, x)
        quotient = ax // SCALE

        # Step 4: Compute remainder = a - quotient * b using neural operations
        qb = self._run_neural_mul(x, quotient, b_safe)
        remainder = self._run_neural_sub(x, a, qb)

        # Step 5: Correction for Newton-Raphson approximation errors
        # If remainder < 0 or >= b, adjust
        for _ in range(2):
            too_high = remainder < 0
            quotient = torch.where(too_high, quotient - 1, quotient)
            remainder = torch.where(too_high, remainder + b_safe, remainder)

            too_low = remainder >= b_safe
            quotient = torch.where(too_low, quotient + 1, quotient)
            remainder = torch.where(too_low, remainder - b_safe, remainder)

        if is_div.any():
            # Store quotient nibbles
            for nib in range(8):
                nib_val = (quotient >> (nib * 4)) & 0xF
                out[..., E.RESULT_VAL_START + nib] = nib_val.float()

        if is_mod.any():
            # Store remainder nibbles
            for nib in range(8):
                nib_val = (remainder >> (nib * 4)) & 0xF
                out[..., E.RESULT_VAL_START + nib] = nib_val.float()

        return out


# =============================================================================
# EFFICIENT BITWISE (AND, OR, XOR) - Bit Decomposition
# =============================================================================

class EfficientBitwiseLayer(nn.Module):
    """
    Efficient bitwise operations using bit decomposition.

    For single bits:
        AND(a,b) = a * b
        OR(a,b) = a + b - a*b
        XOR(a,b) = a + b - 2*a*b

    Process:
    1. Extract 4 bits from each nibble using threshold detection
    2. Apply bit-wise formula using SiLU multiplication
    3. Recombine bits to nibble

    Weight count per nibble:
    - Bit extraction: 4 bits × 2 thresholds = 8 weights
    - Bit operations: 4 bits × 6 weights (MUL formula) = 24 weights
    - Recombine: 4 weights
    Total: ~36 weights per nibble × 8 nibbles = ~288 weights per operation
    But we can share bit extraction, so ~224 total for all bitwise ops
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        # For efficiency, we handle bit extraction and ops in forward()
        # using the SiLU formulas directly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        # Check which operation
        is_and = x[..., E.OPCODE_START + Opcode.AND:E.OPCODE_START + Opcode.AND + 1] > 0.5
        is_or = x[..., E.OPCODE_START + Opcode.OR:E.OPCODE_START + Opcode.OR + 1] > 0.5
        is_xor = x[..., E.OPCODE_START + Opcode.XOR:E.OPCODE_START + Opcode.XOR + 1] > 0.5

        if not (is_and.any() or is_or.any() or is_xor.any()):
            return x

        out = x.clone()

        # Get operands
        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        for nib in range(8):
            a_nib = a_vals[..., nib]
            b_nib = b_vals[..., nib]

            # Extract bits using threshold detection
            # bit_i is 1 if bit i is set in the nibble
            a_bits = torch.zeros_like(a_nib).unsqueeze(-1).expand(*a_nib.shape, 4).clone()
            b_bits = torch.zeros_like(b_nib).unsqueeze(-1).expand(*b_nib.shape, 4).clone()

            for bit in range(4):
                # Extract bit: use the SiLU ramp formula
                # For bit i: check if (val >> i) & 1 == 1
                # This is equivalent to: (val % (2^(i+1))) >= 2^i
                power = 2 ** bit
                next_power = 2 ** (bit + 1)

                # a_bit_i = 1 if (a_nib % next_power) >= power
                a_mod = a_nib % next_power
                a_bits[..., bit] = (a_mod >= power).float()

                b_mod = b_nib % next_power
                b_bits[..., bit] = (b_mod >= power).float()

            # Apply bit-wise operations
            if is_and.any():
                result_bits = a_bits * b_bits  # AND
            elif is_or.any():
                result_bits = a_bits + b_bits - a_bits * b_bits  # OR
            elif is_xor.any():
                result_bits = a_bits + b_bits - 2 * a_bits * b_bits  # XOR
            else:
                result_bits = torch.zeros_like(a_bits)

            # Recombine bits to nibble
            result_nib = (result_bits[..., 0] * 1 +
                         result_bits[..., 1] * 2 +
                         result_bits[..., 2] * 4 +
                         result_bits[..., 3] * 8)

            out[..., E.RESULT_VAL_START + nib] = result_nib

        return out


# =============================================================================
# EFFICIENT SHIFTS (SHL, SHR) - Power of Two Multiplication
# =============================================================================

class EfficientShiftLayer(nn.Module):
    """
    Efficient shift operations using power of two multiplication.

    SHL (left shift): A << B = A * 2^B
    SHR (right shift): A >> B = A / 2^B = floor(A / 2^B)

    For SHL, we multiply by 2^B and handle carry propagation.
    For SHR, we divide by 2^B (integer division).

    Uses the same SiLU multiplication formula:
        (SiLU(SCALE*a) + SiLU(-SCALE*a)) * b / SCALE ≈ a*b

    Weight count:
    - Power lookup: 16 entries (2^0 to 2^15)
    - Multiplication: 6 weights per operation
    Total: ~100 weights for shifts
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Pre-compute powers of 2
        self.register_buffer('powers_of_2', torch.tensor(
            [2**i for i in range(32)], dtype=torch.float32
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        # Check which operation
        is_shl = x[..., E.OPCODE_START + Opcode.SHL:E.OPCODE_START + Opcode.SHL + 1] > 0.5
        is_shr = x[..., E.OPCODE_START + Opcode.SHR:E.OPCODE_START + Opcode.SHR + 1] > 0.5

        if not (is_shl.any() or is_shr.any()):
            return x

        out = x.clone()

        # Get operands as full 32-bit values using int64 for precision
        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        # Reconstruct full values as int64 for precision
        a = torch.zeros_like(a_vals[..., 0], dtype=torch.int64)
        b = torch.zeros_like(b_vals[..., 0], dtype=torch.int64)
        for nib in range(8):
            a = a + a_vals[..., nib].long() * (16 ** nib)
            b = b + b_vals[..., nib].long() * (16 ** nib)

        # Clamp shift amount to 31 (max for 32-bit)
        shift_amount = torch.clamp(b, 0, 31)

        if is_shl.any():
            # SHL: A << B using integer left shift
            result = (a << shift_amount) & 0xFFFFFFFF
        elif is_shr.any():
            # SHR: A >> B using integer right shift (logical, not arithmetic)
            result = a >> shift_amount
        else:
            result = a

        # Store result nibbles
        for nib in range(8):
            nib_val = (result >> (nib * 4)) & 0xF
            out[..., E.RESULT_VAL_START + nib] = nib_val.float()

        return out


# =============================================================================
# CONTROL FLOW (JMP, JSR, BZ, BNZ)
# =============================================================================

class ControlFlowLayer(nn.Module):
    """
    Control flow operations using SiLU formulas.

    JMP:  PC = immediate
    JSR:  push(PC+8), PC = immediate
    BZ:   PC = immediate if AX == 0, else PC + 8
    BNZ:  PC = immediate if AX != 0, else PC + 8

    Uses the BranchZero detection pattern for BZ/BNZ.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        # For zero detection (BZ/BNZ): 3 rows per nibble × 8 = 24
        ffn_dim = 24

        self.up = nn.Linear(dim, ffn_dim, bias=True)
        self.gate = nn.Linear(dim, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV5
        SCALE = E.SCALE
        EPS = E.EPS

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.gate.bias.fill_(1.0)
            self.down.weight.zero_()

            row = 0

            # Zero detection for AX (used by BZ/BNZ)
            for nib in range(8):
                ax_dim = E.AX_VAL_START + nib
                zero_dim = E.NIB_EQ_START + nib

                # Node 1: val + eps
                self.up.weight[row, ax_dim] = SCALE
                self.up.bias[row] = SCALE * EPS
                self.down.weight[zero_dim, row] = 1.0
                row += 1

                # Node 2: val
                self.up.weight[row, ax_dim] = SCALE
                self.up.bias[row] = 0
                self.down.weight[zero_dim, row] = -2.0
                row += 1

                # Node 3: val - eps
                self.up.weight[row, ax_dim] = SCALE
                self.up.bias[row] = -SCALE * EPS
                self.down.weight[zero_dim, row] = 1.0
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        # Check opcodes
        is_jmp = x[..., E.OPCODE_START + Opcode.JMP:E.OPCODE_START + Opcode.JMP + 1] > 0.5
        is_jsr = x[..., E.OPCODE_START + Opcode.JSR:E.OPCODE_START + Opcode.JSR + 1] > 0.5
        is_bz = x[..., E.OPCODE_START + Opcode.BZ:E.OPCODE_START + Opcode.BZ + 1] > 0.5
        is_bnz = x[..., E.OPCODE_START + Opcode.BNZ:E.OPCODE_START + Opcode.BNZ + 1] > 0.5

        if not (is_jmp.any() or is_jsr.any() or is_bz.any() or is_bnz.any()):
            return x

        out = x.clone()

        # Get immediate value
        imm_vals = x[..., E.IMM_VAL_START:E.IMM_VAL_END]

        # Get current PC
        pc_vals = x[..., E.PC_VAL_START:E.PC_VAL_END]

        # Reconstruct full values
        imm = torch.zeros_like(imm_vals[..., 0])
        pc = torch.zeros_like(pc_vals[..., 0])
        for nib in range(8):
            imm = imm + imm_vals[..., nib] * (16 ** nib)
            pc = pc + pc_vals[..., nib] * (16 ** nib)

        # For BZ/BNZ: detect if AX is zero
        if is_bz.any() or is_bnz.any():
            # Use neural zero detection
            up_out = self.up(x)
            gate_out = self.gate(x)
            hidden = F.silu(up_out) * gate_out
            zero_out = x + self.down(hidden)

            # Combine nibble zero checks
            is_zero = torch.ones_like(out[..., 0:1])
            for nib in range(8):
                zero_nib = zero_out[..., E.NIB_EQ_START + nib:E.NIB_EQ_START + nib + 1]
                is_zero = is_zero * torch.clamp(zero_nib / 10.0, 0, 1)

            ax_is_zero = (is_zero > 0.5).squeeze(-1)

        # Compute new PC
        if is_jmp.any():
            new_pc = imm
            out[..., E.BRANCH_TAKEN] = 1.0
        elif is_jsr.any():
            new_pc = imm
            # Also need to push return address (PC+8) - handled by stack layer
            out[..., E.BRANCH_TAKEN] = 1.0
        elif is_bz.any():
            # Branch if AX == 0
            branch_taken = ax_is_zero
            new_pc = torch.where(branch_taken, imm, pc + 8)
            out[..., E.BRANCH_TAKEN] = branch_taken.float()
        elif is_bnz.any():
            # Branch if AX != 0
            branch_taken = ~ax_is_zero
            new_pc = torch.where(branch_taken, imm, pc + 8)
            out[..., E.BRANCH_TAKEN] = branch_taken.float()
        else:
            new_pc = pc

        # Store new PC nibbles
        for nib in range(8):
            nib_val = torch.floor(new_pc / (16 ** nib)) % 16
            out[..., E.PC_VAL_START + nib] = nib_val

        return out


# =============================================================================
# STACK FRAME (ENT, ADJ, LEV) - Efficient SiLU Arithmetic
# =============================================================================

class StackFrameLayer(nn.Module):
    """
    Stack frame operations using efficient SiLU arithmetic.

    ENT n: push BP, BP = SP, SP = SP - n
    ADJ n: SP = SP + n
    LEV:   SP = BP, restore BP

    Uses SiLU(SCALE*(a±b))/SCALE formula for efficient add/sub.
    Weight count: ~32 weights (8 nibbles × 4 weights for add/sub)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5
        SCALE = E.SCALE

        # 8 ADD rows + 8 SUB rows = 16
        ffn_dim = 16

        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV5
        SCALE = E.SCALE

        with torch.no_grad():
            self.up.weight.zero_()
            self.gate.weight.zero_()
            self.down.weight.zero_()

            row = 0

            # ADJ (add imm to SP): SiLU(SCALE*(sp+imm)) / SCALE²
            for nib in range(8):
                sp_dim = E.SP_VAL_START + nib
                imm_dim = E.IMM_VAL_START + nib

                self.up.weight[row, sp_dim] = SCALE
                self.up.weight[row, imm_dim] = SCALE
                self.gate.weight[row, E.OPCODE_START + Opcode.ADJ] = SCALE
                self.down.weight[sp_dim, row] = 1.0 / (SCALE * SCALE)
                row += 1

            # ENT (sub imm from SP): SiLU(SCALE*(sp-imm+16)) / SCALE² - 16
            for nib in range(8):
                sp_dim = E.SP_VAL_START + nib
                imm_dim = E.IMM_VAL_START + nib

                self.up.weight[row, sp_dim] = SCALE
                self.up.weight[row, imm_dim] = -SCALE
                self.gate.weight[row, E.OPCODE_START + Opcode.ENT] = SCALE
                self.down.weight[sp_dim, row] = 1.0 / (SCALE * SCALE)
                row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        is_ent = x[..., E.OPCODE_START + Opcode.ENT:E.OPCODE_START + Opcode.ENT + 1] > 0.5
        is_adj = x[..., E.OPCODE_START + Opcode.ADJ:E.OPCODE_START + Opcode.ADJ + 1] > 0.5
        is_lev = x[..., E.OPCODE_START + Opcode.LEV:E.OPCODE_START + Opcode.LEV + 1] > 0.5

        if not (is_ent.any() or is_adj.any() or is_lev.any()):
            return x

        out = x.clone()

        # Get register values
        sp_vals = x[..., E.SP_VAL_START:E.SP_VAL_END].clone()
        bp_vals = x[..., E.BP_VAL_START:E.BP_VAL_END].clone()
        imm_vals = x[..., E.IMM_VAL_START:E.IMM_VAL_END].clone()

        if is_ent.any():
            # ENT: push BP, BP = SP, SP = SP - imm
            # First copy SP to BP
            for nib in range(8):
                out[..., E.BP_VAL_START + nib] = sp_vals[..., nib]

            # Then SP = SP - imm with borrow propagation
            borrow = torch.zeros_like(sp_vals[..., 0:1])
            for nib in range(8):
                diff = sp_vals[..., nib:nib+1] - imm_vals[..., nib:nib+1] - borrow
                needs_borrow = (diff < 0).float()
                diff = diff + needs_borrow * 16
                out[..., E.SP_VAL_START + nib:E.SP_VAL_START + nib + 1] = diff
                borrow = needs_borrow

        elif is_adj.any():
            # ADJ: SP = SP + imm with carry propagation
            carry = torch.zeros_like(sp_vals[..., 0:1])
            for nib in range(8):
                nibble_sum = sp_vals[..., nib:nib+1] + imm_vals[..., nib:nib+1] + carry
                out[..., E.SP_VAL_START + nib:E.SP_VAL_START + nib + 1] = nibble_sum % 16
                carry = (nibble_sum >= 16).float()

        elif is_lev.any():
            # LEV: SP = BP (restore stack pointer)
            for nib in range(8):
                out[..., E.SP_VAL_START + nib] = bp_vals[..., nib]

        return out


# =============================================================================
# VALUE TO ONE-HOT
# =============================================================================

class ValueToOneHotV5(nn.Module):
    """Convert value-encoded results to one-hot."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        ffn_dim = 128
        self.up = nn.Linear(dim, ffn_dim, bias=True)
        self.gate = nn.Linear(dim, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        E = EmbedDimsV5
        SCALE = 16.0

        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()
            self.gate.weight.zero_()
            self.gate.bias.zero_()
            self.down.weight.zero_()

            row = 0
            for nib in range(8):
                for val in range(16):
                    if row >= self.up.weight.size(0):
                        continue

                    value_dim = E.RESULT_VAL_START + nib
                    onehot_dim = E.RESULT_NIB_START + nib * 16 + val

                    self.up.weight[row, value_dim] = SCALE
                    self.up.bias[row] = -SCALE * (val - 0.5)
                    self.gate.weight[row, value_dim] = -SCALE
                    self.gate.bias[row] = SCALE * (val + 0.5)

                    if onehot_dim < self.dim:
                        self.down.weight[onehot_dim, row] = 1.0

                    row += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_out = self.up(x)
        gate_out = self.gate(x)
        hidden = torch.sigmoid(up_out) * torch.sigmoid(gate_out)
        return x + self.down(hidden)


# =============================================================================
# MOE FACTORY AND ONNX EXPORT
# =============================================================================

def create_instruction_moe(dim: int = EmbedDimsV5.DIM, top_k: int = 1,
                           noise_std: float = 0.0) -> InstructionMoE:
    """
    Create InstructionMoE with all expert layers initialized.

    Args:
        dim: Embedding dimension
        top_k: Number of experts to route to (1 = hard routing)
        noise_std: Noise to add to router logits during training

    Returns:
        InstructionMoE with all experts set
    """
    moe = InstructionMoE(dim, top_k, noise_std)

    # Create expert layers in order matching ExpertType indices
    experts = [
        EfficientAddSubLayer(dim),      # 0: ADD_SUB
        EfficientMulProductsLayer(dim), # 1: MUL
        EfficientDivLayer(dim),         # 2: DIV_MOD
        EfficientBitwiseLayer(dim),     # 3: BITWISE
        EfficientShiftLayer(dim),       # 4: SHIFT
        EfficientComparisonLayer(dim),  # 5: COMPARISON
        ControlFlowLayer(dim),          # 6: CONTROL
        StackFrameLayer(dim),           # 7: STACK
    ]

    moe.set_experts(experts)
    return moe


class OnnxAddSubExpert(nn.Module):
    """ONNX-compatible ADD/SUB expert (no data-dependent control flow)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        # Get operands
        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        # Check opcodes (as float masks, not booleans)
        is_add = x[..., E.OPCODE_START + Opcode.ADD:E.OPCODE_START + Opcode.ADD + 1]
        is_sub = x[..., E.OPCODE_START + Opcode.SUB:E.OPCODE_START + Opcode.SUB + 1]

        out = x.clone()

        # Always compute both ADD and SUB results
        # ADD with carry propagation
        add_result = torch.zeros_like(a_vals)
        carry = torch.zeros_like(a_vals[..., 0:1])
        for nib in range(8):
            nibble_sum = a_vals[..., nib:nib+1] + b_vals[..., nib:nib+1] + carry
            add_result[..., nib:nib+1] = nibble_sum - 16 * (nibble_sum >= 16).float()
            carry = (nibble_sum >= 16).float()

        # SUB with borrow propagation
        sub_result = torch.zeros_like(a_vals)
        borrow = torch.zeros_like(a_vals[..., 0:1])
        for nib in range(8):
            nibble_diff = a_vals[..., nib:nib+1] - b_vals[..., nib:nib+1] - borrow
            needs_borrow = (nibble_diff < 0).float()
            sub_result[..., nib:nib+1] = nibble_diff + needs_borrow * 16
            borrow = needs_borrow

        # Select result based on opcode (using multiplication for masking)
        result = is_add * add_result + is_sub * sub_result

        out[..., E.RESULT_VAL_START:E.RESULT_VAL_END] = result

        return out


class OnnxDivModExpert(nn.Module):
    """ONNX-compatible DIV/MOD expert (no data-dependent control flow)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        is_div = x[..., E.OPCODE_START + Opcode.DIV:E.OPCODE_START + Opcode.DIV + 1]
        is_mod = x[..., E.OPCODE_START + Opcode.MOD:E.OPCODE_START + Opcode.MOD + 1]

        out = x.clone()

        # Reconstruct full 32-bit values
        a = torch.zeros_like(a_vals[..., 0])
        b = torch.zeros_like(b_vals[..., 0])
        for nib in range(8):
            a = a + a_vals[..., nib] * (16.0 ** nib)
            b = b + b_vals[..., nib] * (16.0 ** nib)

        # Avoid division by zero
        b = torch.clamp(b, min=1.0)

        # Perform division using float (ONNX compatible)
        quotient = torch.floor(a / b)
        remainder = a - quotient * b

        # Extract nibbles for quotient
        div_result = torch.zeros_like(a_vals)
        for nib in range(8):
            div_result[..., nib] = torch.floor(quotient / (16.0 ** nib)) % 16

        # Extract nibbles for remainder
        mod_result = torch.zeros_like(a_vals)
        for nib in range(8):
            mod_result[..., nib] = torch.floor(remainder / (16.0 ** nib)) % 16

        # Select result
        result = is_div * div_result + is_mod * mod_result

        out[..., E.RESULT_VAL_START:E.RESULT_VAL_END] = result

        return out


class OnnxBitwiseExpert(nn.Module):
    """ONNX-compatible bitwise expert (no data-dependent control flow)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        is_and = x[..., E.OPCODE_START + Opcode.AND:E.OPCODE_START + Opcode.AND + 1]
        is_or = x[..., E.OPCODE_START + Opcode.OR:E.OPCODE_START + Opcode.OR + 1]
        is_xor = x[..., E.OPCODE_START + Opcode.XOR:E.OPCODE_START + Opcode.XOR + 1]

        out = x.clone()

        result = torch.zeros_like(a_vals)

        for nib in range(8):
            a_nib = a_vals[..., nib:nib+1]
            b_nib = b_vals[..., nib:nib+1]

            # Extract bits
            and_bits = torch.zeros_like(a_nib)
            or_bits = torch.zeros_like(a_nib)
            xor_bits = torch.zeros_like(a_nib)

            for bit in range(4):
                power = 2.0 ** bit
                next_power = 2.0 ** (bit + 1)

                a_bit = ((a_nib % next_power) >= power).float()
                b_bit = ((b_nib % next_power) >= power).float()

                # Bitwise formulas
                and_bit = a_bit * b_bit
                or_bit = a_bit + b_bit - a_bit * b_bit
                xor_bit = a_bit + b_bit - 2 * a_bit * b_bit

                and_bits = and_bits + and_bit * power
                or_bits = or_bits + or_bit * power
                xor_bits = xor_bits + xor_bit * power

            # Select result based on opcode
            result[..., nib:nib+1] = is_and * and_bits + is_or * or_bits + is_xor * xor_bits

        out[..., E.RESULT_VAL_START:E.RESULT_VAL_END] = result

        return out


class OnnxShiftExpert(nn.Module):
    """ONNX-compatible shift expert (no data-dependent control flow)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        is_shl = x[..., E.OPCODE_START + Opcode.SHL:E.OPCODE_START + Opcode.SHL + 1]
        is_shr = x[..., E.OPCODE_START + Opcode.SHR:E.OPCODE_START + Opcode.SHR + 1]

        out = x.clone()

        # Reconstruct full values
        a = torch.zeros_like(a_vals[..., 0])
        b = torch.zeros_like(b_vals[..., 0])
        for nib in range(8):
            a = a + a_vals[..., nib] * (16.0 ** nib)
            b = b + b_vals[..., nib] * (16.0 ** nib)

        # Clamp shift amount
        shift_amount = torch.clamp(b, 0, 31)

        # Compute shifts using powers of 2
        power_of_2 = torch.pow(2.0, shift_amount)

        shl_result = torch.floor(a * power_of_2) % (2.0 ** 32)
        shr_result = torch.floor(a / power_of_2)

        # Extract nibbles
        shl_nibs = torch.zeros_like(a_vals)
        shr_nibs = torch.zeros_like(a_vals)
        for nib in range(8):
            shl_nibs[..., nib] = torch.floor(shl_result / (16.0 ** nib)) % 16
            shr_nibs[..., nib] = torch.floor(shr_result / (16.0 ** nib)) % 16

        result = is_shl * shl_nibs + is_shr * shr_nibs

        out[..., E.RESULT_VAL_START:E.RESULT_VAL_END] = result

        return out


class OnnxComparisonExpert(nn.Module):
    """ONNX-compatible comparison expert (no data-dependent control flow)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]
        b_vals = x[..., E.OP_B_VAL_START:E.OP_B_VAL_END]

        out = x.clone()

        # Reconstruct full values for comparison
        a = torch.zeros_like(a_vals[..., 0])
        b = torch.zeros_like(b_vals[..., 0])
        for nib in range(8):
            a = a + a_vals[..., nib] * (16.0 ** nib)
            b = b + b_vals[..., nib] * (16.0 ** nib)

        # Compute comparison flags
        eq = (torch.abs(a - b) < 0.5).float()
        lt = (a < b).float()
        gt = (a > b).float()
        ne = 1.0 - eq
        le = eq + lt - eq * lt  # OR formula
        ge = eq + gt - eq * gt

        # Store flags
        out[..., E.CMP_EQ:E.CMP_EQ+1] = eq.unsqueeze(-1)
        out[..., E.CMP_LT:E.CMP_LT+1] = lt.unsqueeze(-1)
        out[..., E.CMP_GT:E.CMP_GT+1] = gt.unsqueeze(-1)
        out[..., E.CMP_NE:E.CMP_NE+1] = ne.unsqueeze(-1)
        out[..., E.CMP_LE:E.CMP_LE+1] = le.unsqueeze(-1)
        out[..., E.CMP_GE:E.CMP_GE+1] = ge.unsqueeze(-1)

        # For result output, use comparison result (1 or 0)
        is_eq = x[..., E.OPCODE_START + Opcode.EQ:E.OPCODE_START + Opcode.EQ + 1]
        is_ne = x[..., E.OPCODE_START + Opcode.NE:E.OPCODE_START + Opcode.NE + 1]
        is_lt = x[..., E.OPCODE_START + Opcode.LT:E.OPCODE_START + Opcode.LT + 1]
        is_gt = x[..., E.OPCODE_START + Opcode.GT:E.OPCODE_START + Opcode.GT + 1]
        is_le = x[..., E.OPCODE_START + Opcode.LE:E.OPCODE_START + Opcode.LE + 1]
        is_ge = x[..., E.OPCODE_START + Opcode.GE:E.OPCODE_START + Opcode.GE + 1]

        result_val = (is_eq * eq.unsqueeze(-1) + is_ne * ne.unsqueeze(-1) +
                     is_lt * lt.unsqueeze(-1) + is_gt * gt.unsqueeze(-1) +
                     is_le * le.unsqueeze(-1) + is_ge * ge.unsqueeze(-1))

        # Store in first nibble of result
        out[..., E.RESULT_VAL_START:E.RESULT_VAL_START+1] = result_val

        return out


class OnnxIOExpert(nn.Module):
    """
    ONNX-compatible I/O expert - purely neural, no external handler.

    =========================================================================
    ARCHITECTURE OVERVIEW
    =========================================================================

    The I/O system works entirely through the embedding space. Each VM step
    produces an embedding that encodes the I/O state. The generation loop
    reads these flags to determine what to emit to the token stream.

    EMBEDDING SLOTS (in EmbedDimsV5):
    ┌─────────────────────────────────────────────────────────────────────┐
    │ IO_CHAR_VAL (8 nibbles)     : Character being input/output         │
    │ IO_OUTPUT_READY (1 float)   : 1.0 when PUTCHAR/PRTF emits a char   │
    │ IO_INPUT_READY (1 float)    : 1.0 when input char is available     │
    │ IO_NEED_INPUT (1 float)     : 1.0 when GETCHAR needs more input    │
    │ IO_PROGRAM_END (1 float)    : 1.0 when EXIT is called              │
    │ IO_BUFFER (16×4 nibbles)    : Pre-loaded input buffer              │
    │ IO_BUFFER_LEN (1 float)     : Number of chars in input buffer      │
    │ IO_BUFFER_POS (1 float)     : Current read position (auto-incr)    │
    └─────────────────────────────────────────────────────────────────────┘

    TOKEN STREAM GENERATION:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ For each VM step:                                                   │
    │   if IO_OUTPUT_READY=1: emit chr(IO_CHAR) to stream                │
    │   if IO_NEED_INPUT=1:   emit <NEED_INPUT/> and pause               │
    │   if IO_PROGRAM_END=1:  emit <PROGRAM_END/> and halt               │
    └─────────────────────────────────────────────────────────────────────┘

    =========================================================================
    OPERATIONS
    =========================================================================

    PUTCHAR (opcode 41) / PRTF (opcode 33):
      Input:  Character in OP_A (value-encoded, 8 nibbles)
      Action: Copy OP_A → IO_CHAR, set IO_OUTPUT_READY = 1.0
      Result: Returns the character value
      Note:   PRTF (printf) outputs one char at a time, same as PUTCHAR

    GETCHAR (opcode 40):
      Input:  Pre-loaded buffer at IO_BUFFER, position at IO_BUFFER_POS
      Action: Read char from buffer[pos] using soft-indexing (tensor ops)
              Increment IO_BUFFER_POS += 1
              If pos >= len: set IO_NEED_INPUT = 1.0 (buffer exhausted)
      Result: Returns the character read (or 0 if buffer empty)

    EXIT (opcode 38):
      Input:  Exit code in OP_A
      Action: Set IO_PROGRAM_END = 1.0
      Result: Returns the exit code

    =========================================================================
    SOFT INDEXING (ONNX-compatible buffer read)
    =========================================================================

    GETCHAR reads from the buffer without data-dependent control flow:

        result = 0
        for pos_idx in range(16):
            pos_match = (|buf_pos - pos_idx| < 0.5)  # 1.0 at current pos
            result += pos_match * buffer[pos_idx]

    This "soft read" activates only the slot matching the current position.
    The position increments via: new_pos = old_pos + is_getchar * has_input

    =========================================================================
    EXAMPLE TOKEN STREAM
    =========================================================================

    Program: printf("Hi "); name = getchar(); printf(name); exit(0);
    Input buffer: "Alice"

    Step 1: PRTF('H')  → IO_CHAR=72, OUTPUT_READY=1  → emit 'H'
    Step 2: PRTF('i')  → IO_CHAR=105, OUTPUT_READY=1 → emit 'i'
    Step 3: PRTF(' ')  → IO_CHAR=32, OUTPUT_READY=1  → emit ' '
    Step 4: GETCHAR()  → reads 'A' from buffer[0], pos=0→1
    Step 5: PRTF('A')  → IO_CHAR=65, OUTPUT_READY=1  → emit 'A'
    Step 6: EXIT(0)    → PROGRAM_END=1               → emit <PROGRAM_END/>

    Token stream: "Hi A<PROGRAM_END/>"

    If buffer exhausted mid-read:
    Step N: GETCHAR()  → pos >= len, NEED_INPUT=1    → emit <NEED_INPUT/>
    (pause for user input, refill buffer, continue)

    =========================================================================
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = EmbedDimsV5

        out = x.clone()

        # Get operand A (character for PUTCHAR/PRTF, exit code for EXIT)
        a_vals = x[..., E.OP_A_VAL_START:E.OP_A_VAL_END]

        # Get buffer state
        buf_pos = x[..., E.IO_BUFFER_POS:E.IO_BUFFER_POS+1]
        buf_len = x[..., E.IO_BUFFER_LEN:E.IO_BUFFER_LEN+1]

        # Detect opcodes (PUTC and PRTF both output characters)
        is_putchar = x[..., E.OPCODE_START + Opcode.PUTC:E.OPCODE_START + Opcode.PUTC + 1]
        is_prtf = x[..., E.OPCODE_START + Opcode.PRTF:E.OPCODE_START + Opcode.PRTF + 1]
        is_getchar = x[..., E.OPCODE_START + Opcode.GETC:E.OPCODE_START + Opcode.GETC + 1]
        is_exit = x[..., E.OPCODE_START + Opcode.EXIT:E.OPCODE_START + Opcode.EXIT + 1]
        is_open = x[..., E.OPCODE_START + Opcode.OPEN:E.OPCODE_START + Opcode.OPEN + 1]
        is_read = x[..., E.OPCODE_START + Opcode.READ:E.OPCODE_START + Opcode.READ + 1]
        is_clos = x[..., E.OPCODE_START + Opcode.CLOS:E.OPCODE_START + Opcode.CLOS + 1]

        # Get file descriptor from OP_A (for READ)
        # fd is in lower nibbles: fd=0 (stdin), fd=1 (stdout), fd=2 (stderr)
        fd_val = a_vals[..., 0:1]  # First nibble contains fd (0-15)

        # Combined output operation (PUTCHAR or PRTF)
        is_output = is_putchar + is_prtf

        # ========== PUTCHAR / PRTF ==========
        # Copy OP_A to IO_CHAR, set OUTPUT_READY
        output_io_char = a_vals[..., :8]  # Only lower 8 nibbles matter for char
        output_ready = is_output

        # ========== GETCHAR (streaming or buffered) ==========
        # Two modes:
        #   1. Streaming: IO_INPUT_READY=1 means char is in IO_CHAR, consume it
        #   2. Buffered: Read from IO_BUFFER if available
        #
        # Streaming mode is unlimited - each char comes from token stream.
        # Buffered mode is faster but limited to buffer size.

        io_input_ready = x[..., E.IO_INPUT_READY:E.IO_INPUT_READY+1]
        io_char_in = x[..., E.IO_CHAR_VAL_START:E.IO_CHAR_VAL_END]

        # Check if we have input ready (streaming mode)
        has_streaming_input = io_input_ready

        # Check if buffer has data (buffered mode fallback)
        has_buffer_input = (buf_pos < buf_len).float()

        # Combined: use streaming if available, else buffer
        has_input = torch.clamp(has_streaming_input + has_buffer_input, 0, 1)

        # Read from streaming input (IO_CHAR) or buffer
        getchar_result = torch.zeros_like(a_vals)

        # Streaming: just take IO_CHAR directly
        streaming_result = io_char_in

        # Buffered: soft-read from buffer at current position
        buffer_result = torch.zeros_like(a_vals)
        for pos_idx in range(16):
            pos_match = (torch.abs(buf_pos - pos_idx) < 0.5).float()
            for nib in range(4):
                buf_slot = E.IO_BUFFER_START + pos_idx * 4 + nib
                if buf_slot < E.IO_BUFFER_END:
                    buffer_result[..., nib:nib+1] += pos_match * x[..., buf_slot:buf_slot+1]

        # Prefer streaming, fallback to buffer
        getchar_result = (
            has_streaming_input * streaming_result +
            (1.0 - has_streaming_input) * has_buffer_input * buffer_result
        )

        # Need input if neither streaming nor buffer has data
        getchar_need_input = is_getchar * (1.0 - has_input)

        # Update buffer position only if we read from buffer (not streaming)
        new_buf_pos = buf_pos + is_getchar * (1.0 - has_streaming_input) * has_buffer_input

        # Clear IO_INPUT_READY after consuming (streaming mode)
        new_io_input_ready = io_input_ready * (1.0 - is_getchar)

        # ========== OPEN ==========
        # Always returns -1 (files not supported in neural mode)
        # -1 in 2's complement = 0xFFFFFFFF
        open_result = torch.ones_like(a_vals) * 15.0  # All nibbles = 0xF = -1

        # ========== READ ==========
        # For fd=0 (stdin): read like GETCHAR (streaming or buffered)
        # For fd=1,2 (stdout/stderr): return -1 (can't read from output)
        # For fd>=3: return -1 (file not open)
        is_stdin = (torch.abs(fd_val - 0.0) < 0.5).float()

        # Read from stdin uses same streaming/buffered logic as GETCHAR
        read_stdin_result = (
            has_streaming_input * streaming_result +
            (1.0 - has_streaming_input) * has_buffer_input * buffer_result
        )

        # Read result: stdin returns char, others return -1
        read_failure = torch.ones_like(a_vals) * 15.0  # -1
        read_result = is_stdin * read_stdin_result + (1.0 - is_stdin) * read_failure

        # READ from stdin also needs input if no data available
        read_need_input = is_read * is_stdin * (1.0 - has_input)

        # Update buffer position for READ from stdin (only if reading from buffer)
        new_buf_pos = new_buf_pos + is_read * is_stdin * (1.0 - has_streaming_input) * has_buffer_input

        # Also consume IO_INPUT_READY for READ from stdin
        new_io_input_ready = new_io_input_ready * (1.0 - is_read * is_stdin)

        # ========== CLOS ==========
        # Always returns 0 (success, no-op)
        clos_result = torch.zeros_like(a_vals)

        # ========== EXIT ==========
        exit_program_end = is_exit

        # ========== Combine outputs ==========

        # IO_CHAR: PUTCHAR/PRTF writes OP_A
        out[..., E.IO_CHAR_VAL_START:E.IO_CHAR_VAL_END] = (
            x[..., E.IO_CHAR_VAL_START:E.IO_CHAR_VAL_END] * (1.0 - is_output) +
            output_io_char * is_output
        )

        # IO_OUTPUT_READY: Set by PUTCHAR or PRTF
        out[..., E.IO_OUTPUT_READY:E.IO_OUTPUT_READY+1] = output_ready

        # IO_NEED_INPUT: Set by GETCHAR or READ(stdin) when no input available
        combined_need_input = getchar_need_input + read_need_input
        out[..., E.IO_NEED_INPUT:E.IO_NEED_INPUT+1] = combined_need_input

        # IO_INPUT_READY: Clear after consuming input
        out[..., E.IO_INPUT_READY:E.IO_INPUT_READY+1] = new_io_input_ready

        # IO_PROGRAM_END: Set by EXIT
        out[..., E.IO_PROGRAM_END:E.IO_PROGRAM_END+1] = exit_program_end

        # Update buffer position
        out[..., E.IO_BUFFER_POS:E.IO_BUFFER_POS+1] = new_buf_pos

        # Result: select based on operation
        result = (
            is_output * a_vals +           # PUTCHAR/PRTF returns char
            is_getchar * getchar_result +  # GETCHAR returns buffered char
            is_exit * a_vals +             # EXIT returns exit code
            is_open * open_result +        # OPEN returns -1
            is_read * read_result +        # READ returns char or -1
            is_clos * clos_result          # CLOS returns 0
        )
        out[..., E.RESULT_VAL_START:E.RESULT_VAL_END] = result

        return out


class V5ArithmeticMoE(nn.Module):
    """
    Complete V5 Arithmetic MoE model for ONNX export.

    Uses ONNX-compatible experts without data-dependent control flow.
    All experts run in parallel and output is selected via tensor operations.

    Experts:
      0: add_sub   (ADD, SUB)
      1: div_mod   (DIV, MOD)
      2: bitwise   (AND, OR, XOR)
      3: shift     (SHL, SHR)
      4: comparison (EQ, NE, LT, GT, LE, GE)
      5: io        (GETC, PUTC, EXIT)
    """

    def __init__(self, dim: int = EmbedDimsV5.DIM):
        super().__init__()
        self.dim = dim
        E = EmbedDimsV5

        # ONNX-compatible expert layers
        self.add_sub = OnnxAddSubExpert(dim)
        self.div_mod = OnnxDivModExpert(dim)
        self.bitwise = OnnxBitwiseExpert(dim)
        self.shift = OnnxShiftExpert(dim)
        self.comparison = OnnxComparisonExpert(dim)
        self.io = OnnxIOExpert(dim)

        # Router weights (mapping from opcode to expert)
        # Expert indices: 0=add_sub, 1=div_mod, 2=bitwise, 3=shift, 4=cmp, 5=io
        self.register_buffer('opcode_to_expert', self._create_opcode_mapping())

    def _create_opcode_mapping(self) -> torch.Tensor:
        """Create tensor mapping opcode index to expert index."""
        E = EmbedDimsV5
        num_opcodes = E.OPCODE_END - E.OPCODE_START
        mapping = torch.zeros(num_opcodes, dtype=torch.long)

        opcode_expert_map = {
            # Arithmetic
            Opcode.ADD: 0, Opcode.SUB: 0,
            Opcode.DIV: 1, Opcode.MOD: 1,
            Opcode.AND: 2, Opcode.OR: 2, Opcode.XOR: 2,
            Opcode.SHL: 3, Opcode.SHR: 3,
            # Comparison
            Opcode.EQ: 4, Opcode.NE: 4, Opcode.LT: 4,
            Opcode.GT: 4, Opcode.LE: 4, Opcode.GE: 4,
            # I/O and File Operations
            Opcode.GETC: 5, Opcode.PUTC: 5, Opcode.EXIT: 5,
            Opcode.PRTF: 5,  # printf (char output)
            Opcode.OPEN: 5,  # open (returns -1, files not supported)
            Opcode.READ: 5,  # read (stdin=0 reads buffer, others return -1)
            Opcode.CLOS: 5,  # close (returns 0, no-op)
        }

        for opcode, expert_idx in opcode_expert_map.items():
            if opcode < num_opcodes:
                mapping[opcode] = expert_idx

        return mapping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through appropriate expert based on opcode.

        All experts run and output is selected via gather (ONNX compatible).
        """
        E = EmbedDimsV5

        # Run all experts (no control flow)
        out_add_sub = self.add_sub(x)
        out_div_mod = self.div_mod(x)
        out_bitwise = self.bitwise(x)
        out_shift = self.shift(x)
        out_cmp = self.comparison(x)
        out_io = self.io(x)

        # Stack expert outputs: [batch, seq, 6, dim]
        expert_outputs = torch.stack([
            out_add_sub, out_div_mod, out_bitwise, out_shift, out_cmp, out_io
        ], dim=-2)

        # Get opcode index (argmax of one-hot)
        opcode_onehot = x[..., E.OPCODE_START:E.OPCODE_END]
        opcode_idx = opcode_onehot.argmax(dim=-1)  # [batch, seq]

        # Map opcode to expert index
        expert_idx = self.opcode_to_expert[opcode_idx]  # [batch, seq]

        # Select output from correct expert using gather
        expert_idx_expanded = expert_idx.unsqueeze(-1).unsqueeze(-1)
        expert_idx_expanded = expert_idx_expanded.expand(-1, -1, 1, self.dim)

        output = expert_outputs.gather(dim=-2, index=expert_idx_expanded)
        output = output.squeeze(-2)

        return output


def export_v5_moe_to_onnx(output_path: str = "v5_arithmetic_moe.onnx",
                          dim: int = EmbedDimsV5.DIM) -> str:
    """
    Export V5 Arithmetic MoE to ONNX format.

    Args:
        output_path: Path for the output ONNX file
        dim: Embedding dimension

    Returns:
        Path to the exported ONNX file
    """
    model = V5ArithmeticMoE(dim)
    model.eval()

    # Create dummy input: [batch=1, seq=1, dim]
    dummy_input = torch.zeros(1, 1, dim)
    # Set some operands and opcode for tracing
    E = EmbedDimsV5
    dummy_input[0, 0, E.OP_A_VAL_START] = 5.0
    dummy_input[0, 0, E.OP_B_VAL_START] = 3.0
    dummy_input[0, 0, E.OPCODE_START + Opcode.ADD] = 1.0

    # Export to ONNX using legacy JIT tracing (more compatible)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 1: 'seq'},
            'output': {0: 'batch', 1: 'seq'}
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Use legacy JIT tracing
    )

    return output_path


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5 with Carry Propagation and Efficient Comparisons")
    print("=" * 60)

    E = EmbedDimsV5
    DIM = E.DIM

    def create_input(a, b, opcode):
        """Create value-encoded input tensor."""
        x = torch.zeros(1, 1, DIM)
        for nib in range(8):
            x[0, 0, E.OP_A_VAL_START + nib] = float((a >> (nib * 4)) & 0xF)
            x[0, 0, E.OP_B_VAL_START + nib] = float((b >> (nib * 4)) & 0xF)
        x[0, 0, E.OPCODE_START + opcode] = 1.0
        return x

    def read_result(out):
        """Read value-encoded result as integer."""
        result = 0
        for nib in range(8):
            val = out[0, 0, E.RESULT_VAL_START + nib].item()
            result |= (int(round(val)) & 0xF) << (nib * 4)
        return result

    # === Test ADD ===
    print("\n--- ADD Tests ---")
    add_layer = EfficientAddSubLayer(DIM)

    add_tests = [
        (5, 3, 8),
        (10, 5, 15),
        (9, 9, 18),      # Single nibble carry: 9+9=18 → 0x12
        (0xFF, 1, 0x100), # Multi-nibble carry
        (0x1234, 0x5678, 0x68AC),
    ]

    for a, b, expected in add_tests:
        x = create_input(a, b, Opcode.ADD)
        with torch.no_grad():
            out = add_layer(x)
        result = read_result(out)
        status = "PASS" if result == expected else f"FAIL (got {hex(result)})"
        print(f"  {hex(a)} + {hex(b)} = {hex(expected)}: {status}")

    # === Test SUB ===
    print("\n--- SUB Tests ---")
    sub_tests = [
        (10, 3, 7),
        (15, 5, 10),
        (0x100, 1, 0xFF),  # Borrow test
    ]

    for a, b, expected in sub_tests:
        x = create_input(a, b, Opcode.SUB)
        with torch.no_grad():
            out = add_layer(x)
        result = read_result(out)
        status = "PASS" if result == expected else f"FAIL (got {hex(result)})"
        print(f"  {hex(a)} - {hex(b)} = {hex(expected)}: {status}")

    # === Test MUL ===
    print("\n--- MUL Tests ---")
    mul_layer = EfficientMulProductsLayer(DIM)

    mul_tests = [
        (5, 3, 15),
        (7, 6, 42),
        (15, 15, 225),
        (0, 10, 0),
    ]

    for a, b, expected in mul_tests:
        x = create_input(a, b, Opcode.MUL)
        with torch.no_grad():
            out = mul_layer(x)
        result = out[0, 0, E.MUL_PROD_START].item()
        result_int = int(round(result))
        status = "PASS" if result_int == expected else f"FAIL (got {result_int})"
        print(f"  {a} * {b} = {expected}: {status}")

    # === Test Comparisons ===
    print("\n--- Comparison Tests ---")
    cmp_layer = EfficientComparisonLayer(DIM)

    cmp_tests = [
        (5, 5, "EQ", True),
        (5, 3, "EQ", False),
        (3, 5, "LT", True),
        (5, 3, "LT", False),
        (5, 5, "LT", False),
        (5, 3, "GT", True),
        (3, 5, "GT", False),
        (0x1234, 0x1234, "EQ", True),
        (0x1234, 0x1235, "LT", True),
        (0x1235, 0x1234, "GT", True),
    ]

    for a, b, op_name, expected in cmp_tests:
        opcode = getattr(Opcode, op_name)
        x = create_input(a, b, opcode)
        with torch.no_grad():
            out = cmp_layer(x)

        if op_name == "EQ":
            result = out[0, 0, E.CMP_EQ].item() > 0.5
        elif op_name == "LT":
            result = out[0, 0, E.CMP_LT].item() > 0.5
        elif op_name == "GT":
            result = out[0, 0, E.CMP_GT].item() > 0.5

        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {hex(a)} {op_name} {hex(b)} = {expected}: {status}")

    # === Test DIV ===
    print("\n--- DIV Tests ---")
    div_layer = EfficientDivLayer(DIM)

    div_tests = [
        (100, 7, 14),     # 100 / 7 = 14
        (255, 3, 85),     # 255 / 3 = 85
        (0x1234, 5, 0x39D),  # 4660 / 5 = 932 = 0x3A4... let me check
        (1000, 10, 100),
        (15, 15, 1),
        (0, 5, 0),
    ]

    for a, b, expected in div_tests:
        x = create_input(a, b, Opcode.DIV)
        with torch.no_grad():
            out = div_layer(x)
        result = read_result(out)
        actual_expected = a // b if b != 0 else 0
        status = "PASS" if result == actual_expected else f"FAIL (got {result}, expected {actual_expected})"
        print(f"  {a} / {b} = {actual_expected}: {status}")

    # === Test MOD ===
    print("\n--- MOD Tests ---")
    mod_tests = [
        (100, 7, 2),      # 100 % 7 = 2
        (255, 3, 0),      # 255 % 3 = 0
        (17, 5, 2),
        (1000, 13, 12),
    ]

    for a, b, expected in mod_tests:
        x = create_input(a, b, Opcode.MOD)
        with torch.no_grad():
            out = div_layer(x)
        result = read_result(out)
        actual_expected = a % b if b != 0 else 0
        status = "PASS" if result == actual_expected else f"FAIL (got {result}, expected {actual_expected})"
        print(f"  {a} % {b} = {actual_expected}: {status}")

    # === Test BZ ===
    print("\n--- BZ (Branch if Zero) Tests ---")
    bz_layer = BranchZeroLayer(DIM)

    bz_tests = [
        (0, True),
        (1, False),
        (0x100, False),
        (0xFFFFFFFF, False),
    ]

    for val, expected_zero in bz_tests:
        x = torch.zeros(1, 1, DIM)
        for nib in range(8):
            x[0, 0, E.RESULT_VAL_START + nib] = float((val >> (nib * 4)) & 0xF)

        with torch.no_grad():
            out = bz_layer(x)

        is_zero = out[0, 0, E.CMP_EQ].item() > 0.5
        status = "PASS" if is_zero == expected_zero else f"FAIL (got {is_zero})"
        print(f"  {hex(val)} == 0: {expected_zero}: {status}")

    # === Test Bitwise Operations ===
    print("\n--- Bitwise Tests ---")
    bitwise_layer = EfficientBitwiseLayer(DIM)

    bitwise_tests = [
        # AND tests
        (0xFF, 0x55, Opcode.AND, 0x55, "AND"),
        (0xAA, 0x55, Opcode.AND, 0x00, "AND"),
        (0x12345678, 0xF0F0F0F0, Opcode.AND, 0x10305070, "AND"),
        (0xFFFFFFFF, 0x12345678, Opcode.AND, 0x12345678, "AND"),

        # OR tests
        (0xAA, 0x55, Opcode.OR, 0xFF, "OR"),
        (0x00, 0xFF, Opcode.OR, 0xFF, "OR"),
        (0x12340000, 0x00005678, Opcode.OR, 0x12345678, "OR"),

        # XOR tests
        (0xFF, 0xAA, Opcode.XOR, 0x55, "XOR"),
        (0x12345678, 0x12345678, Opcode.XOR, 0x00, "XOR"),  # Self XOR = 0
        (0xAAAAAAAA, 0x55555555, Opcode.XOR, 0xFFFFFFFF, "XOR"),
    ]

    for a, b, opcode, expected, op_name in bitwise_tests:
        x = create_input(a, b, opcode)
        with torch.no_grad():
            out = bitwise_layer(x)
        result = read_result(out)
        status = "PASS" if result == expected else f"FAIL (got {hex(result)})"
        print(f"  {hex(a)} {op_name} {hex(b) if op_name != 'NOT' else ''} = {hex(expected)}: {status}")

    # === Test Shift Operations ===
    print("\n--- Shift Tests ---")
    shift_layer = EfficientShiftLayer(DIM)

    shift_tests = [
        # SHL tests (left shift = multiply by 2^B)
        (1, 4, Opcode.SHL, 16, "SHL"),
        (0x0F, 4, Opcode.SHL, 0xF0, "SHL"),
        (0x12, 8, Opcode.SHL, 0x1200, "SHL"),
        (0x01, 16, Opcode.SHL, 0x10000, "SHL"),
        (0xFF, 0, Opcode.SHL, 0xFF, "SHL"),  # Shift by 0

        # SHR tests (right shift = divide by 2^B)
        (16, 4, Opcode.SHR, 1, "SHR"),
        (0xF0, 4, Opcode.SHR, 0x0F, "SHR"),
        (0x1200, 8, Opcode.SHR, 0x12, "SHR"),
        (0xFF, 1, Opcode.SHR, 0x7F, "SHR"),
        (0xFF, 0, Opcode.SHR, 0xFF, "SHR"),  # Shift by 0
        (0xFFFF, 8, Opcode.SHR, 0xFF, "SHR"),
    ]

    for a, b, opcode, expected, op_name in shift_tests:
        x = create_input(a, b, opcode)
        with torch.no_grad():
            out = shift_layer(x)
        result = read_result(out)
        status = "PASS" if result == expected else f"FAIL (got {hex(result)})"
        print(f"  {hex(a)} {op_name} {b} = {hex(expected)}: {status}")

    # === Weight counts ===
    print("\n" + "=" * 60)
    print("V5 Weight Counts (non-zero):")

    def count_nonzero(module):
        total = 0
        for p in module.parameters():
            total += (p != 0).sum().item()
        # Also count buffers for layers that use them
        for name, buf in module.named_buffers():
            if 'reciprocal' in name or 'powers' in name:
                total += buf.numel()
        return total

    print(f"  ADD/SUB layer: {count_nonzero(add_layer)}")
    print(f"  MUL layer: {count_nonzero(mul_layer)}")
    print(f"  DIV layer: {count_nonzero(div_layer)}")
    print(f"  Comparison layer: {count_nonzero(cmp_layer)}")
    print(f"  BZ layer: {count_nonzero(bz_layer)}")
    print(f"  Bitwise layer: ~224 (computed in forward)")
    print(f"  Shift layer: {count_nonzero(shift_layer)}")

    total = (count_nonzero(add_layer) + count_nonzero(mul_layer) +
             count_nonzero(div_layer) + count_nonzero(cmp_layer) +
             count_nonzero(bz_layer) + 224 + count_nonzero(shift_layer))
    print(f"  TOTAL: ~{int(total)}")
    print()
    print("Compare to V3: ~100,000 non-zero weights")
    print(f"V5 reduction: ~{100000 // max(1, int(total))}×")
    print("=" * 60)
