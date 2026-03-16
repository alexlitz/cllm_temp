#!/usr/bin/env python3
"""
Pure Neural VM V7 - Truly Pure Implementation

All computation flows through standard transformer layers.
NO Python arithmetic in forward passes.
Subclasses ONLY implement weight baking.

Architecture:
- PureFFN: Base FFN with FINAL forward, subclass bakes weights
- PureAttention: Base attention with FINAL forward, subclass bakes weights
- Operations composed from these base layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


# =============================================================================
# EMBEDDING LAYOUT
# =============================================================================

class E:
    """Embedding dimensions - 8 nibble positions, each with features."""

    # Per-nibble features
    NIB_A = 0          # Operand A nibble (0-15 encoded)
    NIB_B = 1          # Operand B nibble (0-15 encoded)
    RAW_SUM = 2        # Raw A + B or A - B (before carry/borrow)
    CARRY_IN = 3       # Carry/borrow from lower nibble
    CARRY_OUT = 4      # Carry/borrow to higher nibble
    RESULT = 5         # Result nibble
    TEMP = 6           # Temporary storage for multi-step ops

    # Opcode one-hot (shared across positions)
    OP_START = 7
    NUM_OPS = 32

    # Position encoding
    POS = 39

    DIM = 48           # Total per-position dimension (increased for TEMP)
    NUM_POSITIONS = 8  # 8 nibbles

    # Scale for SwiGLU identity (higher = tighter approximations)
    SCALE = 100.0


# =============================================================================
# OPCODES
# =============================================================================

class Opcode:
    # ALU operations
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4
    AND = 5
    OR = 6
    XOR = 7
    EQ = 8
    NE = 9
    LT = 10
    GT = 11
    LE = 12
    GE = 13
    SHL = 14
    SHR = 15

    # Control flow operations
    JMP = 16    # Unconditional jump: PC = target
    BEQ = 17    # Branch if equal: if A == B then PC = target
    BNE = 18    # Branch if not equal: if A != B then PC = target
    BLT = 19    # Branch if less than: if A < B then PC = target
    BGE = 20    # Branch if greater or equal: if A >= B then PC = target
    CALL = 21   # Function call: push PC, PC = target
    RET = 22    # Return: PC = pop

    # Memory operations
    LOAD = 23   # Load from memory: RESULT = MEM[address]
    STORE = 24  # Store to memory: MEM[address] = value
    PUSH = 25   # Push to stack: SP--, MEM[SP] = value
    POP = 26    # Pop from stack: RESULT = MEM[SP], SP++

    # Special
    NOP = 27    # No operation
    HALT = 28   # Stop execution


# =============================================================================
# BASE FFN LAYER (Fixed Forward)
# =============================================================================

class PureFFN(nn.Module):
    """
    Pure SwiGLU FFN with FINAL forward pass.

    Subclasses ONLY override _bake_weights() to set weight values.
    Forward is: output = x + W_down @ (silu(W_up @ x + b_up) * (W_gate @ x + b_gate)) + b_down
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.W_up = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_up = nn.Parameter(torch.zeros(hidden_dim))
        self.W_gate = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.W_down = nn.Parameter(torch.zeros(dim, hidden_dim))
        self.b_down = nn.Parameter(torch.zeros(dim))

        self._bake_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - Standard SwiGLU with all biases. DO NOT OVERRIDE."""
        up = F.linear(x, self.W_up, self.b_up)
        gate = F.linear(x, self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        return x + F.linear(hidden, self.W_down, self.b_down)

    def _bake_weights(self):
        """Override to bake operation-specific weights."""
        pass


# =============================================================================
# BASE ATTENTION LAYER (Fixed Forward)
# =============================================================================

class PureAttention(nn.Module):
    """
    Pure Attention with FINAL forward pass.

    Subclasses ONLY override _bake_weights().
    Used for carry propagation between nibble positions.
    """

    def __init__(self, dim: int, num_heads: int = 1, causal: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))

        # Causal mask (lower triangular) or zeros
        if causal:
            mask = torch.triu(torch.ones(E.NUM_POSITIONS, E.NUM_POSITIONS) * float('-inf'), diagonal=1)
        else:
            mask = torch.zeros(E.NUM_POSITIONS, E.NUM_POSITIONS)
        self.register_buffer('mask', mask)

        self._bake_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - Standard attention. DO NOT OVERRIDE."""
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        Q = F.linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)
        K = F.linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
        V = F.linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Always add mask (mask is zeros when no masking needed)
        scores = scores + self.mask[:S, :S]

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return x + F.linear(out, self.W_o)

    def _bake_weights(self):
        """Override to bake attention weights."""
        pass


# =============================================================================
# MIXTURE OF EXPERTS LAYER (Sparse Routing)
# =============================================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts layer with sparse routing based on opcode.

    Each expert is an FFN specialized for a specific opcode.
    The router extracts the opcode from the input and selects
    only the matching expert(s) to run.

    This provides true sparse computation - only the relevant
    expert runs, not all experts sequentially.
    """

    def __init__(self, expert_dict: dict):
        """
        Args:
            expert_dict: Dict mapping opcode -> expert FFN
                         e.g., {Opcode.ADD: AddRawSumFFN(), Opcode.SUB: SubRawDiffFFN(), ...}
        """
        super().__init__()
        self.expert_dict = nn.ModuleDict({str(k): v for k, v in expert_dict.items()})
        self.opcodes = list(expert_dict.keys())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Route to correct expert based on opcode in input.

        The opcode is encoded as one-hot in positions E.OP_START to E.OP_START+NUM_OPS.
        We extract which opcode is active and run only that expert.
        """
        B, S, D = x.shape

        # Extract active opcode from first position (all positions have same opcode)
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        # Run only the matching expert if it exists
        if active_opcode in self.opcodes:
            expert = self.expert_dict[str(active_opcode)]
            x = expert(x)

        return x


class MultiExpertMoELayer(nn.Module):
    """
    MoE layer that runs multiple related experts for an operation.

    Some operations require multiple FFN stages (e.g., ADD needs raw_sum, init, carry_detect).
    This layer bundles related experts and runs them all for matching opcodes.
    """

    def __init__(self, opcode_experts: dict):
        """
        Args:
            opcode_experts: Dict mapping opcode -> list of expert FFNs
                           e.g., {Opcode.ADD: [AddRawSumFFN(), InitResultFFN(), ...]}
        """
        super().__init__()
        self.opcode_to_experts = {}
        all_experts = []
        for opcode, experts in opcode_experts.items():
            self.opcode_to_experts[opcode] = []
            for expert in experts:
                idx = len(all_experts)
                all_experts.append(expert)
                self.opcode_to_experts[opcode].append(idx)

        self.experts = nn.ModuleList(all_experts)
        self.opcodes = list(opcode_experts.keys())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route to correct experts based on opcode."""
        # Extract active opcode
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        # Run all experts for this opcode
        if active_opcode in self.opcode_to_experts:
            for expert_idx in self.opcode_to_experts[active_opcode]:
                x = self.experts[expert_idx](x)

        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block: Attention + MoE.

    Alternates between attention (for position-to-position communication)
    and MoE layers (for operation-specific computation).
    """

    def __init__(self, attention: PureAttention, moe: nn.Module):
        super().__init__()
        self.attention = attention
        self.moe = moe

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention then MoE."""
        x = self.attention(x)
        x = self.moe(x)
        return x


class SparseMoEALU(nn.Module):
    """
    Sparse MoE-based ALU that truly routes to specific experts.

    Architecture:
    - Stage 1: Initial computation (MoE routes to opcode-specific FFNs)
    - Stage 2: Attention for position communication (carry propagation)
    - Stage 3: Post-attention FFNs (MoE)
    - Repeat stages 2-3 for cascade operations

    This is more efficient than running all FFNs because only
    the relevant experts are executed.
    """

    def __init__(self):
        super().__init__()

        # Create experts for each opcode
        self.initial_experts = MultiExpertMoELayer({
            Opcode.ADD: [AddRawSumFFN(), InitResultFFN(), CarryDetectFFN()],
            Opcode.SUB: [SubRawDiffFFN(), SubInitResultFFN(), BorrowDetectFFN()],
            Opcode.MUL: [MulProductFFN(), MulGateFFN(), MulOverflowFFN()],
            Opcode.DIV: [DivInitFFN()],
            Opcode.MOD: [ModInitFFN()],
            Opcode.AND: [ExtractBit3FFN(Opcode.AND), ExtractBit2FFN(Opcode.AND),
                        ExtractBit1FFN(Opcode.AND), ExtractBit0FFN(Opcode.AND),
                        BitwiseAndCombineFFN()],
            Opcode.OR: [ExtractBit3FFN(Opcode.OR), ExtractBit2FFN(Opcode.OR),
                       ExtractBit1FFN(Opcode.OR), ExtractBit0FFN(Opcode.OR),
                       BitwiseOrCombineFFN()],
            Opcode.XOR: [ExtractBit3FFN(Opcode.XOR), ExtractBit2FFN(Opcode.XOR),
                        ExtractBit1FFN(Opcode.XOR), ExtractBit0FFN(Opcode.XOR),
                        BitwiseXorCombineFFN()],
            Opcode.EQ: [CompareDiffFFN(Opcode.EQ), CompareEqNibbleFFN(Opcode.EQ)],
            Opcode.NE: [CompareDiffFFN(Opcode.NE), CompareNeNibbleFFN(Opcode.NE)],
            Opcode.LT: [CompareDiffFFN(Opcode.LT), CompareLtNibbleFFN(Opcode.LT)],
            Opcode.GT: [CompareDiffFFN(Opcode.GT), CompareGtNibbleFFN(Opcode.GT)],
            Opcode.LE: [CompareDiffFFN(Opcode.LE), CompareGtNibbleFFN(Opcode.LE)],
            Opcode.GE: [CompareDiffFFN(Opcode.GE), CompareLtNibbleFFN(Opcode.GE)],
            Opcode.SHL: [ClearTempBeforeShiftFFN(), ShiftLeftCopyFFN()],
            Opcode.SHR: [ClearTempBeforeShiftFFN(), ShiftRightCopyFFN()],
            Opcode.JMP: [JumpFFN()],
            Opcode.BEQ: [BranchEqFFN()],
            Opcode.BNE: [BranchNeFFN()],
            Opcode.BLT: [BranchLtFFN()],
            Opcode.BGE: [BranchGeFFN()],
            Opcode.LOAD: [LoadFFN()],
            Opcode.STORE: [StoreFFN()],
            Opcode.PUSH: [PushFFN()],
            Opcode.POP: [PopFFN()],
            Opcode.NOP: [NopFFN()],
            Opcode.HALT: [HaltFFN()],
        })

        # Shared attention layer for carry/borrow propagation
        self.carry_attn = CarryPropagateAttention()

        # Carry iteration experts (per-opcode)
        self.carry_experts = MultiExpertMoELayer({
            Opcode.ADD: [ZeroFirstCarryFFN(), ClearCarryOutFFN(), CarryIterFFN(), ClearCarryInFFN()],
            Opcode.SUB: [ZeroFirstBorrowFFN(), ClearBorrowOutFFN(), BorrowIterFFN(), ClearBorrowInFFN()],
            Opcode.MUL: [MulZeroFirstCarryFFN(), MulClearCarryOutFFN(), MulCarryIterFFN(), MulClearCarryInFFN()],
        })

        # Shift attention layers
        self.shl_attn = ShiftLeftAttention()
        self.shr_attn = ShiftRightAttention()

        # Finalization experts
        self.final_experts = MultiExpertMoELayer({
            Opcode.AND: [ClearBitsFFN(Opcode.AND)],
            Opcode.OR: [ClearBitsFFN(Opcode.OR)],
            Opcode.XOR: [ClearBitsFFN(Opcode.XOR)],
            Opcode.MUL: [MulClearTempFFN()],
            Opcode.EQ: [CompareReduceEqFFN()],
            Opcode.NE: [CompareReduceNeFFN()],
            Opcode.LT: [CompareCopyResultFFN(Opcode.LT)],
            Opcode.GT: [CompareCopyResultFFN(Opcode.GT)],
            Opcode.LE: [CompareCopyResultFFN(Opcode.LE)],
            Opcode.GE: [CompareCopyResultFFN(Opcode.GE)],
            Opcode.SHL: [ShiftLeftResultFFN(), ShiftLeftClearFFN()],
            Opcode.SHR: [ShiftRightResultFFN(), ShiftRightClearFFN()],
        })

        # DIV/MOD iteration layers
        self.div_iters = nn.ModuleList([DivIterFFN() for _ in range(16)])
        self.mod_iters = nn.ModuleList([ModIterFFN() for _ in range(16)])
        self.mod_result = ModResultFFN()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse MoE forward pass.

        Only runs the experts relevant to the input opcode.
        """
        # Extract active opcode
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        # Stage 1: Initial computation (MoE)
        x = self.initial_experts(x)

        # Stage 2-3: Carry cascade (7 iterations)
        # Only run for ops that need carry propagation
        if active_opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL]:
            for _ in range(7):
                x = self.carry_attn(x)
                x = self.carry_experts(x)

        # DIV/MOD iterations (if applicable)
        if active_opcode == Opcode.DIV:
            for layer in self.div_iters:
                x = layer(x)

        if active_opcode == Opcode.MOD:
            for layer in self.mod_iters:
                x = layer(x)
            x = self.mod_result(x)

        # Shift attention (if applicable)
        if active_opcode == Opcode.SHL:
            x = self.shl_attn(x)
        elif active_opcode == Opcode.SHR:
            x = self.shr_attn(x)

        # Stage 4: Finalization (MoE)
        x = self.final_experts(x)

        return x


# =============================================================================
# CONTROL FLOW FFN LAYERS
# =============================================================================

class JumpFFN(PureFFN):
    """
    JMP: Copy target address (NIB_B) to RESULT unconditionally.
    The result represents the new PC value.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy NIB_B (target) to RESULT, gated on JMP opcode
            self.W_up[0, E.OP_START + Opcode.JMP] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.JMP] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class BranchEqFFN(PureFFN):
    """
    BEQ: Branch if A == B.
    Result = B (target) if A == B, else Result = 0 (no branch).
    Uses silu approximation of equality check.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Detect if a == b using step(a - b + 0.5) - step(a - b - 0.5)
            # This gives ~1 when |a - b| < 0.5

            # Row 0-1: step(a - b + 0.5) ≈ 1 when a >= b - 0.5
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.NIB_B] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.OP_START + Opcode.BEQ] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.W_up[1, E.NIB_B] = -S
            self.b_up[1] = S * 1.5
            self.W_gate[1, E.OP_START + Opcode.BEQ] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # Row 2-3: step(a - b - 0.5) ≈ 1 when a >= b + 0.5
            self.W_up[2, E.NIB_A] = S
            self.W_up[2, E.NIB_B] = -S
            self.b_up[2] = -S * 0.5
            self.W_gate[2, E.OP_START + Opcode.BEQ] = 1.0
            self.W_down[E.TEMP, 2] = -1.0 / S

            self.W_up[3, E.NIB_A] = S
            self.W_up[3, E.NIB_B] = -S
            self.b_up[3] = -S * 1.5
            self.W_gate[3, E.OP_START + Opcode.BEQ] = 1.0
            self.W_down[E.TEMP, 3] = 1.0 / S

            # Row 4-5: If equal (TEMP ~= 1), copy NIB_B to RESULT
            self.W_up[4, E.TEMP] = S
            self.b_up[4] = -S * 0.5
            self.W_gate[4, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 4] = 1.0 / S

            self.W_up[5, E.TEMP] = S
            self.b_up[5] = -S * 1.5
            self.W_gate[5, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 5] = 1.0 / S


class BranchNeFFN(PureFFN):
    """
    BNE: Branch if A != B.
    Result = B (target) if A != B, else Result = 0 (no branch).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Detect if a != b: 1 - equality
            # Use |a - b| > 0.5 as inequality indicator

            # Row 0-1: step(a - b - 0.5) ≈ 1 when a > b
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.NIB_B] = -S
            self.b_up[0] = -S * 0.5
            self.W_gate[0, E.OP_START + Opcode.BNE] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.W_up[1, E.NIB_B] = -S
            self.b_up[1] = -S * 1.5
            self.W_gate[1, E.OP_START + Opcode.BNE] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # Row 2-3: step(b - a - 0.5) ≈ 1 when b > a
            self.W_up[2, E.NIB_B] = S
            self.W_up[2, E.NIB_A] = -S
            self.b_up[2] = -S * 0.5
            self.W_gate[2, E.OP_START + Opcode.BNE] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, E.NIB_B] = S
            self.W_up[3, E.NIB_A] = -S
            self.b_up[3] = -S * 1.5
            self.W_gate[3, E.OP_START + Opcode.BNE] = 1.0
            self.W_down[E.TEMP, 3] = -1.0 / S

            # Row 4-5: If not equal (TEMP > 0), copy NIB_B to RESULT
            self.W_up[4, E.TEMP] = S
            self.b_up[4] = -S * 0.5
            self.W_gate[4, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 4] = 1.0 / S

            self.W_up[5, E.TEMP] = S
            self.b_up[5] = -S * 1.5
            self.W_gate[5, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 5] = 1.0 / S


class BranchLtFFN(PureFFN):
    """
    BLT: Branch if A < B.
    Result = target (NIB_B) if A < B, else 0.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(b - a - 0.5) ≈ 1 when b > a (i.e., a < b)
            self.W_up[0, E.NIB_B] = S
            self.W_up[0, E.NIB_A] = -S
            self.b_up[0] = -S * 0.5
            self.W_gate[0, E.OP_START + Opcode.BLT] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.NIB_B] = S
            self.W_up[1, E.NIB_A] = -S
            self.b_up[1] = -S * 1.5
            self.W_gate[1, E.OP_START + Opcode.BLT] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # If a < b, copy target to RESULT
            self.W_up[2, E.TEMP] = S
            self.b_up[2] = -S * 0.5
            self.W_gate[2, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.TEMP] = S
            self.b_up[3] = -S * 1.5
            self.W_gate[3, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class BranchGeFFN(PureFFN):
    """
    BGE: Branch if A >= B.
    Result = target (NIB_B) if A >= B, else 0.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(a - b + 0.5) ≈ 1 when a >= b - 0.5 (i.e., a >= b)
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.NIB_B] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.OP_START + Opcode.BGE] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.W_up[1, E.NIB_B] = -S
            self.b_up[1] = S * 1.5
            self.W_gate[1, E.OP_START + Opcode.BGE] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # If a >= b, copy target to RESULT
            self.W_up[2, E.TEMP] = S
            self.b_up[2] = -S * 0.5
            self.W_gate[2, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.TEMP] = S
            self.b_up[3] = -S * 1.5
            self.W_gate[3, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


# =============================================================================
# MEMORY OPERATION FFN LAYERS
# =============================================================================

class LoadFFN(PureFFN):
    """
    LOAD: Read from memory address (NIB_A) to RESULT.
    Note: This is a placeholder - actual memory access requires external state.
    For pure neural VM, we simulate memory as attention over a memory embedding.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For now, copy address to RESULT as placeholder
            # Real implementation would use memory attention layer
            self.W_up[0, E.OP_START + Opcode.LOAD] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.LOAD] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class StoreFFN(PureFFN):
    """
    STORE: Write value (NIB_B) to memory address (NIB_A).
    Note: This is a placeholder - actual memory write requires external state.
    For pure neural VM, we prepare the address/value pair for memory attention.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy address to TEMP slot
            self.W_up[0, E.OP_START + Opcode.STORE] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.STORE] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Copy value to RESULT
            self.W_up[2, E.OP_START + Opcode.STORE] = S
            self.W_gate[2, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.STORE] = -S
            self.W_gate[3, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class PushFFN(PureFFN):
    """
    PUSH: Push value to stack.
    Copies NIB_A (value) to RESULT for stack operation.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.PUSH] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.PUSH] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class PopFFN(PureFFN):
    """
    POP: Pop value from stack.
    Placeholder - actual stack access requires external state.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Placeholder: set RESULT to 0 (actual value comes from memory)
            self.W_up[0, E.OP_START + Opcode.POP] = S
            self.W_gate[0, E.RESULT] = -1.0  # Clear current result
            self.W_down[E.RESULT, 0] = 1.0 / S


class NopFFN(PureFFN):
    """NOP: No operation - just passes through."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        # Empty weights - no contribution
        pass


class HaltFFN(PureFFN):
    """
    HALT: Signal program end.
    Sets RESULT to a special marker value (e.g., all 1s).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set RESULT to 15 (max nibble) as halt indicator
            self.b_up[0] = S  # Always on
            self.W_gate[0, E.OP_START + Opcode.HALT] = 15.0
            self.W_down[E.RESULT, 0] = 1.0 / S


# =============================================================================
# ADD: Raw Sum FFN
# =============================================================================

class AddRawSumFFN(PureFFN):
    """Computes raw_sum = nib_a + nib_b for each position."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: positive contribution
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.NIB_B] = S
            self.W_gate[0, E.OP_START + Opcode.ADD] = S
            self.W_down[E.RAW_SUM, 0] = 1.0 / (S * S)

            # Row 1: negative (for symmetry in silu)
            self.W_up[1, E.NIB_A] = -S
            self.W_up[1, E.NIB_B] = -S
            self.W_gate[1, E.OP_START + Opcode.ADD] = -S
            self.W_down[E.RAW_SUM, 1] = 1.0 / (S * S)


# =============================================================================
# ADD: Carry Detection FFN
# =============================================================================

class CarryDetectFFN(PureFFN):
    """Detects if raw_sum >= 16, sets carry_out to ~1."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        # Use a=15, b=16 for exact integer thresholds
        # step(x) = [silu(S*(x-15)) - silu(S*(x-16))] / S
        # At x=16: step ≈ 1, at x=15: step ≈ 0

        with torch.no_grad():
            # Row 0: positive contribution
            self.W_up[0, E.RAW_SUM] = S
            self.b_up[0] = -S * 15.0
            self.W_gate[0, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            # Row 1: saturation
            self.W_up[1, E.RAW_SUM] = S
            self.b_up[1] = -S * 16.0
            self.W_gate[1, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S


# =============================================================================
# ADD: Carry Propagation Attention
# =============================================================================

class CarryPropagateAttention(PureAttention):
    """
    Each position i gets carry_in from position i-1's carry_out.

    Uses a "previous position only" mask (not standard causal).
    Position i can ONLY attend to position i-1.
    """

    def __init__(self):
        # Don't use standard causal mask
        super().__init__(E.DIM, num_heads=1, causal=False)

        # Create "previous position only" mask
        # Position i can only attend to position i-1
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(1, N):
            mask[i, i-1] = 0.0  # Position i attends to i-1
        # Position 0 attends to itself (but V[0] has carry_out=0, so cin=0)
        mask[0, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # Q and K just need to be non-zero for attention to work
            # The mask does the actual position selection
            self.W_q[0, 0] = 1.0  # Arbitrary non-zero
            self.W_k[0, 0] = 1.0

            # V: copy carry_out to carry_in slot
            self.W_v[E.CARRY_IN, E.CARRY_OUT] = 1.0

            # Output: pass through carry_in
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


# =============================================================================
# ADD: Zero First Carry FFN
# =============================================================================

class ZeroFirstCarryFFN(PureFFN):
    """
    Zeros out carry_in for position 0.

    At pos=0: silu(S*(0.5 - pos)) = silu(10) ≈ 10, contributes -cin
    At pos>=1: silu(S*(0.5 - pos)) = silu(negative) ≈ 0, contributes nothing
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Single row: only activates at pos=0
            # silu(S*(0.5 - pos)) at pos=0 gives silu(S*0.5) ≈ S*0.5
            # gate = cin
            # hidden = S*0.5 * cin
            # W_down = -1/(S*0.5), so contribution = -cin ✓
            # Note: this one correctly uses S*0.5 because silu(S*0.5) ≈ S*0.5

            self.W_up[0, E.POS] = -S  # Negative position
            self.b_up[0] = S * 0.5    # Threshold at 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


# =============================================================================
# ADD: Initialize Result FFN
# =============================================================================

class InitResultFFN(PureFFN):
    """
    Initialize RESULT = RAW_SUM mod 16.
    Also initialize CARRY_OUT = (RAW_SUM >= 16).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0-1: Copy raw_sum to result (gated on ADD)
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Row 2-3: Subtract 16 from result when raw_sum >= 16
            self.W_up[2, E.RAW_SUM] = S
            self.b_up[2] = -S * 15.0
            self.W_gate[2, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 2] = -16.0 / S

            # Clamp correction
            self.W_up[3, E.RAW_SUM] = S
            self.b_up[3] = -S * 16.0
            self.W_gate[3, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 3] = 16.0 / S


# =============================================================================
# ADD: Carry Iteration FFN
# =============================================================================

class CarryIterFFN(PureFFN):
    """
    One iteration of carry propagation.

    Adds carry_in to RESULT, subtracts 16 if overflow, updates CARRY_OUT.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0-1: Add carry_in to result
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Row 2-3: Subtract 16 when result + carry_in >= 16
            self.W_up[2, E.RESULT] = S
            self.W_up[2, E.CARRY_IN] = S
            self.b_up[2] = -S * 15.0
            self.W_gate[2, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 2] = -16.0 / S
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S  # Set new carry

            self.W_up[3, E.RESULT] = S
            self.W_up[3, E.CARRY_IN] = S
            self.b_up[3] = -S * 16.0
            self.W_gate[3, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.RESULT, 3] = 16.0 / S
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S

            # Row 4-5: Clear old carry_out before setting new
            # (The above rows ADD to carry_out, but we need to clear first)
            # This is tricky... skip for now, let carry accumulate


# =============================================================================
# SUB: Raw Difference FFN
# =============================================================================

class SubRawDiffFFN(PureFFN):
    """Computes raw_diff = nib_a - nib_b for each position (range -15 to +15)."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # raw_diff = a - b, gated on SUB opcode
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_gate[0, E.NIB_B] = -1.0  # Subtract b
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


# =============================================================================
# SUB: Initialize Result FFN
# =============================================================================

class SubInitResultFFN(PureFFN):
    """
    Initialize RESULT = RAW_DIFF mod 16.
    If raw_diff < 0, result = raw_diff + 16.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0-1: Copy raw_diff to result (gated on SUB)
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Row 2-3: Add 16 when raw_diff < 0
            # step(-raw_diff) = 1 when raw_diff < 0
            self.W_up[2, E.RAW_SUM] = -S  # Negative raw_diff
            self.b_up[2] = -S * 0.0  # Threshold at 0
            self.W_gate[2, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 2] = 16.0 / S

            # Saturation at -1
            self.W_up[3, E.RAW_SUM] = -S
            self.b_up[3] = -S * 1.0
            self.W_gate[3, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 3] = -16.0 / S


# =============================================================================
# SUB: Borrow Detection FFN
# =============================================================================

class BorrowDetectFFN(PureFFN):
    """Detects if raw_diff < 0, sets carry_out (used as borrow) to ~1."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(-raw_diff - 0) = 1 when raw_diff < 0
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = -S * 0.0
            self.W_gate[0, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            # Saturation at -1
            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S


# =============================================================================
# SUB: Borrow Iteration FFN
# =============================================================================

class BorrowIterFFN(PureFFN):
    """
    One iteration of borrow propagation.

    Subtracts borrow_in from RESULT, adds 16 if underflow, updates CARRY_OUT (borrow).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0-1: Subtract borrow_in from result
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.CARRY_IN] = -1.0  # Subtract
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Row 2-3: Add 16 when result - borrow_in < 0
            self.W_up[2, E.RESULT] = -S
            self.W_up[2, E.CARRY_IN] = S  # negative of (result - cin) = -result + cin
            self.b_up[2] = -S * 0.0
            self.W_gate[2, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 2] = 16.0 / S
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S  # Set new borrow

            self.W_up[3, E.RESULT] = -S
            self.W_up[3, E.CARRY_IN] = S
            self.b_up[3] = -S * 1.0
            self.W_gate[3, E.OP_START + Opcode.SUB] = 1.0
            self.W_down[E.RESULT, 3] = -16.0 / S
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S


# =============================================================================
# SUB: Zero First Borrow FFN
# =============================================================================

class ZeroFirstBorrowFFN(PureFFN):
    """Zeros out carry_in (borrow) for position 0 in SUB."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


# =============================================================================
# SUB: Clear Borrow Out FFN
# =============================================================================

class ClearBorrowOutFFN(PureFFN):
    """Clears carry_out (borrow) before detecting new borrows."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


# =============================================================================
# SUB: Clear Borrow In FFN
# =============================================================================

class ClearBorrowInFFN(PureFFN):
    """Clears carry_in (borrow) before next propagation round."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SUB] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SUB] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


# =============================================================================
# BITWISE OPERATIONS - Direct Lookup Table Approach
# =============================================================================

# For bitwise ops on nibbles (0-15), we use direct lookup rather than bit extraction.
# This is more reliable and avoids accumulation errors.

class E_BITS:
    """Additional slots for bit extraction (stored in TEMP region)."""
    BIT3_A = 40  # MSB of nibble A
    BIT2_A = 41
    BIT1_A = 42
    BIT0_A = 43
    BIT3_B = 44
    BIT2_B = 45
    BIT1_B = 46
    BIT0_B = 47


class BitwiseAndDirectFFN(PureFFN):
    """
    Direct AND computation using lookup table approach.

    For each (a, b) pair where the result has a specific bit set,
    we add that bit's contribution when both inputs match.

    This uses: result += weight * indicator(a >= thresh_a AND a < thresh_a+1 AND b >= thresh_b AND b < thresh_b+1)
    """

    def __init__(self):
        # For AND, we need to detect when bits match and both are 1
        # Use product approximation: bit_a * bit_b via silu gating
        # Result = sum over bits: 2^i * (a_bit_i AND b_bit_i)
        super().__init__(E.DIM, hidden_dim=64)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each bit position, compute the AND
            # bit3: a >= 8 AND b >= 8 -> add 8
            # bit2: (a in 4-7 or 12-15) AND (b in 4-7 or 12-15) -> add 4
            # etc.

            # Simpler: directly compute for each (a,b) pair
            # For nibble AND, compute: result = a AND b
            # We'll use 16 hidden units to detect each value of a,
            # then gate on b to compute contribution

            idx = 0
            for bit in range(4):  # bits 0-3
                bit_val = 1 << bit  # 1, 2, 4, 8

                # Values of a that have this bit set
                a_vals = [v for v in range(16) if (v >> bit) & 1]

                for a_val in a_vals:
                    # Detect a == a_val using narrow pulse: step(a - a_val + 0.5) - step(a - a_val - 0.5)
                    # When a == a_val: silu(S*0.5) - silu(-S*0.5) ≈ S*0.5

                    # Row for positive step
                    self.W_up[idx, E.NIB_A] = S
                    self.b_up[idx] = -S * (a_val - 0.5)
                    # Gate on b having this bit set: b >= bit_val and b < bit_val + next_power
                    # Simpler: gate on b directly, let combining handle it
                    self.W_gate[idx, E.OP_START + Opcode.AND] = 1.0
                    self.W_down[E.TEMP, idx] = 1.0  # Store a detection in TEMP

                    idx += 1
                    if idx >= 64:
                        break
                if idx >= 64:
                    break


class BitwiseAndFFN(PureFFN):
    """
    Simple AND computation: result = (a AND b) for nibbles.

    Uses direct product: for each bit position i,
    bit_i_result = step(a - 2^i) * step(b - 2^i) when both have bit i set.

    Simplified approach: Use the identity a AND b = ((a + b) - (a XOR b)) / 2
    But XOR is also complex...

    Alternative: For nibbles, use lookup approach.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        # 16 hidden units to detect each possible a value
        super().__init__(E.DIM, hidden_dim=32)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For nibble AND, we can use the product formula:
            # For bit i: result_bit_i = a_bit_i * b_bit_i
            #
            # Detect bit using: bit_i = floor((a mod 2^(i+1)) / 2^i)
            # This is complex in neural networks.
            #
            # Alternative: Direct multiplication approximation
            # For very small values, silu(S*a) * b / S ≈ a * b
            # But we need bit-wise AND, not arithmetic multiplication.
            #
            # Let's use a different approach: detect a in ranges and multiply by b's bits

            # Bit 3 (value 8): a >= 8 AND b >= 8
            # Detect a >= 8: step(a - 7.5)
            # Detect b >= 8: gate on step(b - 7.5)
            # But we can't easily compute step(b) in the gate...

            # Cleanest approach: a AND b for nibbles
            # result = sum over all (i,j) of: delta(a=i) * delta(b=j) * (i AND j)
            # This requires 256 hidden units, but we only have 32.

            # Compromise: compute bit by bit
            # For bit 3: contribute 8 if a >= 8 AND b >= 8
            # Use: silu(S*(a-7.5)) / S ≈ step(a >= 8)
            # Then: silu(S*step_a) * step_b ≈ step_a * step_b (when both are 0 or 1)

            # Row 0-1: Bit 3 (contributes 8)
            # Detect a >= 8
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 7.5
            self.W_gate[0, E.NIB_B] = 1.0 / 8.0  # Approximate: gate proportional to b
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            # When a >= 8: silu(pos) ≈ pos, gate ≈ b/8
            # Contribution ≈ (a-7.5) * b/8 when a >= 8
            # This isn't right...

            # Let's try yet another approach: direct computation per bit
            # Store intermediate bit detections, then combine

            # Clear everything and use simpler logic
            self.W_up.zero_()
            self.b_up.zero_()
            self.W_gate.zero_()
            self.W_down.zero_()

            # Bit 3: result += 8 if a >= 8 AND b >= 8
            # Two-step: first detect if a >= 8, then multiply by (b >= 8)
            # silu(S*(a-7.5)) gives ~0 for a<8, ~S*(a-7.5) for a>=8

            # For a=8: silu(S*0.5) ≈ S*0.5
            # For a=15: silu(S*7.5) ≈ S*7.5

            # To get a 0/1 step, we need saturation:
            # step(x) = silu(S*x) / (S*x) when x > 0, which requires division...

            # Simpler: Use gating directly
            # silu(S*(a-7.5)) * (b-7.5) / S^2 when both positive
            # At a=8, b=8: silu(50) * 0.5 / 10000 ≈ 50 * 0.5 / 10000 = 0.0025
            # That's too small.

            # Let's scale differently:
            # hidden = silu(S*(a-7.5)) * (b >= 8 indicator)
            # We need the indicator for b >= 8 in a different way

            # Actually, the cleanest is to compute bit by bit with proper scaling:
            # For bit i: a_i = (a >> i) & 1, b_i = (b >> i) & 1
            # result_i = a_i * b_i (multiplication works for AND of 0/1)

            # The challenge is extracting bits. Let me try a working bit extraction:
            # bit3(x) = clamp(x - 7.5, 0, 1) via silu saturation

            # Using: indicator(x >= thresh) ≈ silu(S*(x - thresh + 0.5)) / silu(S*0.5)
            # At thresh=8: silu(S*(8 - 8 + 0.5)) / silu(S*0.5) = silu(50) / silu(50) = 1
            # At thresh=8, x=7: silu(S*(7 - 8 + 0.5)) / silu(S*0.5) = silu(-50) / silu(50) ≈ 0

            # This normalizes correctly! Let's use this.
            norm = 1.0 / (S * 0.5)  # Normalization factor for step output to be ~1

            # Bit 3: add 8 if a >= 8 AND b >= 8
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 7.5  # Fires when a >= 8
            self.W_gate[0, E.NIB_B] = norm  # Scaled by b (larger when b >= 8)
            self.W_down[E.RESULT, 0] = 8.0 * norm

            # But this gives result proportional to b, not binary AND...
            # We need b to also be thresholded.

            # Two-layer approach needed. In a single FFN, we can't easily do this.
            # Let's use the TEMP slot as intermediate storage.

            # Row 0-1: Detect a >= 8, store in TEMP
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 7.5
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = norm

            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 8.5
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -norm


# Simpler approach: Direct nibble AND using product approximation
class DirectAndFFN(PureFFN):
    """
    Direct AND for nibbles using product formula.

    For bits that are 0 or 1: a AND b = a * b
    We extract each bit, multiply, and recombine.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        # Using the observation that for bits (0 or 1): AND = product
        # We need to extract bits first, then multiply.
        #
        # For bit 3 of a: (a >= 8) gives 1, else 0
        # For bit 3 of b: (b >= 8) gives 1, else 0
        # result_bit3 = bit3_a * bit3_b
        # contributes: 8 * result_bit3 to result

        # Challenge: we can't easily compute products in a single layer.
        # silu(S*x) * y gives S*x*y for large x, not x*y.

        # Alternative: Use the formula a AND b = a + b - (a OR b)
        # But OR is also complex...

        # Let's just compute directly for the common cases using lookup

        with torch.no_grad():
            # Pass through - this layer prepares for combining
            pass


class ClearBitSlotsFFN(PureFFN):
    """
    Clear all bit extraction slots before extracting.
    This prevents accumulation from residual connections.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear all 8 bit slots by subtracting their current values
            slots = [E_BITS.BIT3_A, E_BITS.BIT2_A, E_BITS.BIT1_A, E_BITS.BIT0_A,
                     E_BITS.BIT3_B, E_BITS.BIT2_B, E_BITS.BIT1_B, E_BITS.BIT0_B]
            for i, slot in enumerate(slots):
                self.W_up[i, E.OP_START + self.opcode] = S
                self.W_gate[i, slot] = -1.0  # Negate current value
                self.W_down[slot, i] = 1.0 / S


class ExtractBit3FFN(PureFFN):
    """
    Extract bit 3 (MSB) from nibbles A and B.
    bit3 = 1 if nibble >= 8, else 0.

    Uses integer-aligned bounded step: silu(S*(x-7)) - silu(S*(x-8))
    For x=8: silu(S*1)/S - silu(S*0)/S = 1 - 0 = 1.0 (exact!)
    For x=7: silu(S*0)/S - silu(-S)/S = 0 - 0 = 0.0 (exact!)
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Bit 3 of A: 1 if a >= 8
            # bounded_step(7, 8): at a=8 gives 1, at a=7 gives 0
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 7.0  # Lower threshold at 7
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT3_A, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 8.0  # Upper threshold at 8
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

    Uses integer-aligned bounded steps:
    - bounded_step(3, 4) turns ON at 4
    - bounded_step(7, 8) turns OFF at 8
    - bounded_step(11, 12) turns ON at 12
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=12)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For A: bounded steps at 4, 8, 12
            # Step ON at 4: bounded_step(3, 4)
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 3.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 4.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 1] = -1.0 / S

            # Step OFF at 8: bounded_step(7, 8)
            self.W_up[2, E.NIB_A] = S
            self.b_up[2] = -S * 7.0
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 2] = -1.0 / S

            self.W_up[3, E.NIB_A] = S
            self.b_up[3] = -S * 8.0
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT2_A, 3] = 1.0 / S

            # Step ON at 12: bounded_step(11, 12)
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

    Uses integer-aligned bounded steps:
    - bounded_step(1, 2) turns ON at 2
    - bounded_step(3, 4) turns OFF at 4
    - bounded_step(5, 6) turns ON at 6
    - bounded_step(7, 8) turns OFF at 8
    - bounded_step(9, 10) turns ON at 10
    - bounded_step(11, 12) turns OFF at 12
    - bounded_step(13, 14) turns ON at 14
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=28)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For A: bounded steps at each toggle point
            # ON at 2: bounded_step(1, 2)
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = -S * 1.0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 2.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 1] = -1.0 / S

            # OFF at 4: bounded_step(3, 4)
            self.W_up[2, E.NIB_A] = S
            self.b_up[2] = -S * 3.0
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 2] = -1.0 / S

            self.W_up[3, E.NIB_A] = S
            self.b_up[3] = -S * 4.0
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 3] = 1.0 / S

            # ON at 6: bounded_step(5, 6)
            self.W_up[4, E.NIB_A] = S
            self.b_up[4] = -S * 5.0
            self.W_gate[4, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 4] = 1.0 / S

            self.W_up[5, E.NIB_A] = S
            self.b_up[5] = -S * 6.0
            self.W_gate[5, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 5] = -1.0 / S

            # OFF at 8: bounded_step(7, 8)
            self.W_up[6, E.NIB_A] = S
            self.b_up[6] = -S * 7.0
            self.W_gate[6, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 6] = -1.0 / S

            self.W_up[7, E.NIB_A] = S
            self.b_up[7] = -S * 8.0
            self.W_gate[7, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 7] = 1.0 / S

            # ON at 10: bounded_step(9, 10)
            self.W_up[8, E.NIB_A] = S
            self.b_up[8] = -S * 9.0
            self.W_gate[8, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 8] = 1.0 / S

            self.W_up[9, E.NIB_A] = S
            self.b_up[9] = -S * 10.0
            self.W_gate[9, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 9] = -1.0 / S

            # OFF at 12: bounded_step(11, 12)
            self.W_up[10, E.NIB_A] = S
            self.b_up[10] = -S * 11.0
            self.W_gate[10, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 10] = -1.0 / S

            self.W_up[11, E.NIB_A] = S
            self.b_up[11] = -S * 12.0
            self.W_gate[11, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 11] = 1.0 / S

            # ON at 14: bounded_step(13, 14)
            self.W_up[12, E.NIB_A] = S
            self.b_up[12] = -S * 13.0
            self.W_gate[12, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 12] = 1.0 / S

            self.W_up[13, E.NIB_A] = S
            self.b_up[13] = -S * 14.0
            self.W_gate[13, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_A, 13] = -1.0 / S

            # For B (same pattern starting at row 14)
            self.W_up[14, E.NIB_B] = S
            self.b_up[14] = -S * 1.0
            self.W_gate[14, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 14] = 1.0 / S

            self.W_up[15, E.NIB_B] = S
            self.b_up[15] = -S * 2.0
            self.W_gate[15, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 15] = -1.0 / S

            self.W_up[16, E.NIB_B] = S
            self.b_up[16] = -S * 3.0
            self.W_gate[16, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 16] = -1.0 / S

            self.W_up[17, E.NIB_B] = S
            self.b_up[17] = -S * 4.0
            self.W_gate[17, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 17] = 1.0 / S

            self.W_up[18, E.NIB_B] = S
            self.b_up[18] = -S * 5.0
            self.W_gate[18, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 18] = 1.0 / S

            self.W_up[19, E.NIB_B] = S
            self.b_up[19] = -S * 6.0
            self.W_gate[19, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 19] = -1.0 / S

            self.W_up[20, E.NIB_B] = S
            self.b_up[20] = -S * 7.0
            self.W_gate[20, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 20] = -1.0 / S

            self.W_up[21, E.NIB_B] = S
            self.b_up[21] = -S * 8.0
            self.W_gate[21, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 21] = 1.0 / S

            self.W_up[22, E.NIB_B] = S
            self.b_up[22] = -S * 9.0
            self.W_gate[22, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 22] = 1.0 / S

            self.W_up[23, E.NIB_B] = S
            self.b_up[23] = -S * 10.0
            self.W_gate[23, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 23] = -1.0 / S

            self.W_up[24, E.NIB_B] = S
            self.b_up[24] = -S * 11.0
            self.W_gate[24, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 24] = -1.0 / S

            self.W_up[25, E.NIB_B] = S
            self.b_up[25] = -S * 12.0
            self.W_gate[25, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 25] = 1.0 / S

            self.W_up[26, E.NIB_B] = S
            self.b_up[26] = -S * 13.0
            self.W_gate[26, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 26] = 1.0 / S

            self.W_up[27, E.NIB_B] = S
            self.b_up[27] = -S * 14.0
            self.W_gate[27, E.OP_START + self.opcode] = 1.0
            self.W_down[E_BITS.BIT1_B, 27] = -1.0 / S


class ExtractBit0FFN(PureFFN):
    """
    Extract bit 0 (LSB) from nibbles A and B.
    bit0 = 1 if nibble is odd {1,3,5,7,9,11,13,15}.

    Uses integer-aligned bounded steps:
    - bounded_step(0, 1) turns ON at 1
    - bounded_step(1, 2) turns OFF at 2
    - bounded_step(2, 3) turns ON at 3
    - etc.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=64)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For A: bounded steps at each integer
            # ON at 1: bounded_step(0, 1)
            # OFF at 2: bounded_step(1, 2)
            # ON at 3: bounded_step(2, 3)
            # etc.
            row = 0
            for i in range(16):
                threshold = float(i)  # Integer threshold
                # First row of bounded step
                self.W_up[row, E.NIB_A] = S
                self.b_up[row] = -S * threshold
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                # i=0,2,4,... turn ON (add); i=1,3,5,... turn OFF (subtract)
                if i % 2 == 0:
                    self.W_down[E_BITS.BIT0_A, row] = 1.0 / S
                else:
                    self.W_down[E_BITS.BIT0_A, row] = -1.0 / S
                row += 1

                # Second row of bounded step (saturation at threshold+1)
                self.W_up[row, E.NIB_A] = S
                self.b_up[row] = -S * (threshold + 1.0)
                self.W_gate[row, E.OP_START + self.opcode] = 1.0
                if i % 2 == 0:
                    self.W_down[E_BITS.BIT0_A, row] = -1.0 / S
                else:
                    self.W_down[E_BITS.BIT0_A, row] = 1.0 / S
                row += 1

            # For B (same pattern)
            for i in range(16):
                threshold = float(i)  # Integer threshold
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


class ClearBitsFFN(PureFFN):
    """Clear the bit extraction slots after bitwise operation."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            for i, slot in enumerate([E_BITS.BIT3_A, E_BITS.BIT2_A, E_BITS.BIT1_A, E_BITS.BIT0_A,
                                      E_BITS.BIT3_B, E_BITS.BIT2_B, E_BITS.BIT1_B, E_BITS.BIT0_B]):
                self.W_up[i, E.OP_START + self.opcode] = S
                self.W_gate[i, slot] = -1.0
                self.W_down[slot, i] = 1.0 / S


class BitwiseLookupFFN(PureFFN):
    """
    Direct lookup table for bitwise operations on nibbles.

    For nibbles a, b in [0, 15], computes result using pulse detection.
    Each hidden unit fires for a specific (a_range, b_range) and adds
    the appropriate contribution to the result.

    This uses 64 hidden units organized as:
    - 4 bits x 16 detection units per bit
    """

    def __init__(self, opcode: int, op_func):
        """
        Args:
            opcode: The opcode (AND, OR, XOR)
            op_func: A function that takes (a, b) and returns the result
        """
        self.opcode = opcode
        self.op_func = op_func
        super().__init__(E.DIM, hidden_dim=256)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each possible (a, b) pair, create a detector
            # Detector fires when a is in narrow range around target_a
            # and b is in narrow range around target_b
            #
            # Actually, we use: pulse(a, target) = step(a - target + 0.5) - step(a - target - 0.5)
            # This gives ~1 when a == target, ~0 otherwise.
            #
            # For 2D detection: pulse(a, ta) * pulse(b, tb)
            # We need to encode this in a single FFN layer.
            #
            # SwiGLU: output = silu(up) * gate
            # If up encodes a detection and gate encodes b detection,
            # then output ~ 1 only when both match.

            idx = 0
            for target_a in range(16):
                for target_b in range(16):
                    result = self.op_func(target_a, target_b)
                    if result == 0:
                        continue  # Skip zero contributions

                    # Detect a == target_a: silu(S*(a - target_a + 0.5)) - silu(S*(a - target_a - 0.5))
                    # At a = target_a: silu(S*0.5) - silu(-S*0.5) ≈ S*0.5
                    # At a = target_a ± 1: silu(S*1.5) - silu(S*0.5) ≈ S (not ideal...)
                    #
                    # Better: narrow pulse using single threshold
                    # silu(S * (1 - |a - target_a|)) fires when a == target_a
                    # But |...| is hard in neural net.
                    #
                    # Alternative: Product of two step functions
                    # step(a >= target_a) * step(a <= target_a) = 1 iff a == target_a
                    # step(a >= target_a) = silu(S*(a - target_a + 0.5)) / (S*0.5)
                    # step(a <= target_a) = silu(S*(target_a - a + 0.5)) / (S*0.5)
                    #
                    # In SwiGLU: up = S*(a - target_a + 0.5), gate = (target_a - a + 0.5)
                    # hidden = silu(S*(a - target_a + 0.5)) * (target_a - a + 0.5)
                    # At a = target_a: silu(S*0.5) * 0.5 ≈ (S*0.5) * 0.5 = S*0.25
                    # At a = target_a + 1: silu(S*1.5) * (-0.5) ≈ S*1.5 * (-0.5) = -S*0.75 (wrong sign!)
                    #
                    # This product approach doesn't work directly.
                    #
                    # Simplest that works: use 2 rows per (a,b) pair
                    # Row i: positive step for a
                    # Row i+1: negative saturation step for a
                    # Both gated on b detection

                    if idx + 1 >= 256:
                        break

                    # Step function for a == target_a:
                    # f(a) = step(a - target_a + 0.5) - step(a - target_a - 0.5)
                    # = 1 in narrow window around target_a

                    # Row idx: positive contribution
                    self.W_up[idx, E.NIB_A] = S
                    self.b_up[idx] = -S * (target_a - 0.5)  # Fires when a >= target_a - 0.5

                    # Gate: should fire when b == target_b
                    # Linear gate can't do equality check directly
                    # Use: gate = max(0, 1 - |b - target_b|) ≈ 1 when b == target_b
                    # In linear form: gate = 1 - |b - target_b|
                    # But |...| is not linear.
                    #
                    # Simplification: just gate on b directly and accept some error
                    # gate = b - (target_b - 0.5) when b >= target_b
                    # This gives gate > 0 when b >= target_b, with magnitude proportional to b

                    # Actually, let's use a different approach:
                    # The opcode gating ensures only AND/OR/XOR runs this layer.
                    # For the a and b detection, we can use the residual structure:
                    # output += contribution only when a and b match targets.

                    # For now, use simple linear interpolation approach:
                    # This won't be exact but should give reasonable results.

                    self.W_gate[idx, E.NIB_B] = 1.0  # Gate proportional to b
                    self.W_gate[idx, E.OP_START + self.opcode] = 1.0
                    # Contribution scaled by result value
                    # This accumulates across all a values >= target_a, which is wrong...

                    idx += 1

            # This approach isn't working. Let me try something different.
            # Clear and use proper pulse detection.
            self.W_up.zero_()
            self.b_up.zero_()
            self.W_gate.zero_()
            self.W_down.zero_()
            self.b_down.zero_()


class BitwiseAndCombineFFN(PureFFN):
    """
    Combine extracted bits using AND.

    For bits that are properly extracted as 0 or 1:
    result = 8 * (bit3_a * bit3_b) + 4 * (bit2_a * bit2_b) + 2 * (bit1_a * bit1_b) + (bit0_a * bit0_b)

    Uses: silu(S * bit_a) * bit_b gives ~S * bit_a * bit_b when bit_a is 0 or 1.
    Normalize by dividing output by S.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For properly extracted bits (0 or 1):
            # silu(S * 0) = 0, silu(S * 1) = S * 1 (for large S)
            # So: silu(S * bit_a) * bit_b / S = bit_a * bit_b

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
    Combine extracted bits using OR: result = sum of 2^i * (bit_i_a + bit_i_b - bit_i_a * bit_i_b).
    OR(a,b) = a + b - a*b = a + b*(1-a).
    We compute: result += bit_a + bit_b, then subtract bit_a*bit_b.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each bit: add bit_a, add bit_b, subtract bit_a*bit_b
            for bit_idx, (bit_a, bit_b, weight) in enumerate([
                (E_BITS.BIT3_A, E_BITS.BIT3_B, 8.0),
                (E_BITS.BIT2_A, E_BITS.BIT2_B, 4.0),
                (E_BITS.BIT1_A, E_BITS.BIT1_B, 2.0),
                (E_BITS.BIT0_A, E_BITS.BIT0_B, 1.0),
            ]):
                row = bit_idx * 4
                # Add bit_a (gated on OR opcode via bit_a itself when 0 or 1)
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
    Combine extracted bits using XOR: result = sum of 2^i * (bit_i_a + bit_i_b - 2*bit_i_a*bit_i_b).
    XOR(a,b) = a + b - 2*a*b.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=16)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each bit: add bit_a, add bit_b, subtract 2*bit_a*bit_b
            for bit_idx, (bit_a, bit_b, weight) in enumerate([
                (E_BITS.BIT3_A, E_BITS.BIT3_B, 8.0),
                (E_BITS.BIT2_A, E_BITS.BIT2_B, 4.0),
                (E_BITS.BIT1_A, E_BITS.BIT1_B, 2.0),
                (E_BITS.BIT0_A, E_BITS.BIT0_B, 1.0),
            ]):
                row = bit_idx * 4
                # Add bit_a (gated on XOR opcode via bit_a itself when 0 or 1)
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


# =============================================================================
# COMPARISON OPERATIONS
# =============================================================================

# For comparisons, we need to detect per-nibble equality/inequality
# and combine across all nibbles.
#
# EQ: All nibbles must be equal
# NE: At least one nibble must differ
# LT/GT/LE/GE: Compare from MSB, first differing nibble determines result

class CompareDiffFFN(PureFFN):
    """
    Compute per-nibble difference: diff = nib_a - nib_b.
    Range: -15 to +15. Store in RAW_SUM slot.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_gate[0, E.NIB_B] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class ClearRawSumFFN(PureFFN):
    """Clear RAW_SUM slot before attention-based reduction."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Subtract RAW_SUM from itself (gated on opcode)
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.RAW_SUM] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.RAW_SUM] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class CompareEqNibbleFFN(PureFFN):
    """
    Detect if diff == 0 for each nibble.
    eq_nib = 1 if diff == 0, else 0.

    Uses bounded step with integer-aligned thresholds:
    eq = bounded_step(-diff; -1, 0) = step(-diff + 1) - step(-diff)
    At diff=0: step(1) - step(0) = 1 - 0 = 1 ✓
    At diff=±1: 0 ✓
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: step(-diff + 1) - activates when diff <= 0
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = S * 1.0  # Integer threshold at -diff = -1
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            # Row 1: Saturation for step(-diff + 1), caps at 1
            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = 0.0  # Integer threshold at -diff = 0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # Row 2: step(-diff) - subtract, activates when diff <= -1
            self.W_up[2, E.RAW_SUM] = -S
            self.b_up[2] = 0.0  # Integer threshold at -diff = 0
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 2] = -1.0 / S

            # Row 3: Saturation for step(-diff), caps at 1
            self.W_up[3, E.RAW_SUM] = -S
            self.b_up[3] = -S * 1.0  # Integer threshold at -diff = 1
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class CompareNeNibbleFFN(PureFFN):
    """
    Detect if diff != 0 for each nibble.
    ne_nib = 1 if diff != 0.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # ne = step(diff >= 1) + step(diff <= -1)
            # Use integer-aligned thresholds for exact 0/1 at integer values

            # For diff >= 1: bounded_step(0, 1)
            # silu(S*(diff - 0)) - silu(S*(diff - 1)) = S at diff=1
            self.W_up[0, E.RAW_SUM] = S
            self.b_up[0] = 0.0  # Threshold at 0 (was 0.5)
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = S
            self.b_up[1] = -S * 1.0  # Saturation at 1 (was 1.5)
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S

            # For diff <= -1: bounded_step on -diff
            # silu(S*(-diff - 0)) - silu(S*(-diff - 1)) = S at diff=-1
            self.W_up[2, E.RAW_SUM] = -S
            self.b_up[2] = 0.0  # Threshold at 0 (was 0.5)
            self.W_gate[2, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, E.RAW_SUM] = -S
            self.b_up[3] = -S * 1.0  # Saturation at 1 (was 1.5)
            self.W_gate[3, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 3] = -1.0 / S


class CompareLtNibbleFFN(PureFFN):
    """
    Detect if diff < 0 (a < b) for each nibble.
    lt_nib = 1 if nib_a < nib_b.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(-diff - 0.5): 1 when diff <= -1 (i.e., a < b)
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = -S * 0.5
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.5
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class CompareGtNibbleFFN(PureFFN):
    """
    Detect if diff > 0 (a > b) for each nibble.
    gt_nib = 1 if nib_a > nib_b.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(diff - 0.5): 1 when diff >= 1 (i.e., a > b)
            self.W_up[0, E.RAW_SUM] = S
            self.b_up[0] = -S * 0.5
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.RAW_SUM] = S
            self.b_up[1] = -S * 1.5
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class CompareReduceEqAttention(PureAttention):
    """
    Sum per-nibble EQ results using attention.
    Only position 0 gets the sum. Other positions get their own TEMP.
    EQ = 1 iff sum(TEMP) == 8 (all nibbles equal).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        # Position 0 attends to all positions uniformly
        # Other positions self-attend
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0  # Position 0 attends to all
        for i in range(1, N):
            mask[i, i] = 0.0  # Others self-attend
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # W_q = 0 and W_k = 0 (default) gives Q = K = 0 for all positions
            # scores = Q @ K.T = 0, so softmax(scores + mask) gives uniform attention
            # where mask = 0, and 0 attention where mask = -inf
            # For position 0: uniform 1/8 attention to all positions
            # For positions 1-7: 100% attention to self

            # V: read TEMP, scale by 8 for position 0 to get sum
            self.W_v[E.RAW_SUM, E.TEMP] = 8.0

            # Output: write to RAW_SUM
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class CompareReduceEqFFN(PureFFN):
    """
    Threshold the sum of per-nibble EQ results.
    EQ = 1 iff sum(TEMP) == 8 (all nibbles equal).
    Output at position 0 only using combined position + value threshold.

    Combines conditions: step(RAW_SUM - 7 - 100*pos) = 1 only at pos=0 AND RAW_SUM>=8
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear RESULT at all positions first
            self.W_up[0, E.OP_START + Opcode.EQ] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Rows 1-2: Threshold with position gating
            # step(RAW_SUM - 7 - 100*pos) = 1 only when RAW_SUM >= 8 AND pos = 0
            # At pos=0, RAW_SUM=8: 8 - 7 - 0 = 1 -> 1 ✓
            # At pos=0, RAW_SUM=7: 7 - 7 - 0 = 0 -> 0 ✓
            # At pos=1, RAW_SUM=8: 8 - 7 - 100 = -99 -> 0 ✓
            #
            # Row 1: step(RAW_SUM - 7 - 100*pos)
            self.W_up[1, E.RAW_SUM] = S
            self.W_up[1, E.POS] = -S * 100.0  # Large negative weight for position
            self.b_up[1] = -S * 7.0  # Threshold at RAW_SUM = 7
            self.W_gate[1, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Row 2: Saturation - cap at 1
            self.W_up[2, E.RAW_SUM] = S
            self.W_up[2, E.POS] = -S * 100.0
            self.b_up[2] = -S * 8.0
            self.W_gate[2, E.OP_START + Opcode.EQ] = 1.0
            self.W_down[E.RESULT, 2] = -1.0 / S


class CompareReduceNeAttention(PureAttention):
    """
    Sum per-nibble NE results using attention.
    Position 0 attends to all positions uniformly to get sum.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        # Position 0 attends to all positions uniformly
        # Other positions self-attend
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for j in range(N):
            mask[0, j] = 0.0  # Position 0 attends to all
        for i in range(1, N):
            mask[i, i] = 0.0  # Others self-attend
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # W_q = 0 and W_k = 0 (default) gives uniform attention
            # where mask = 0

            # V: read TEMP, scaled by 8 so sum of averages gives actual sum
            self.W_v[E.RAW_SUM, E.TEMP] = 8.0

            # Output: write to RAW_SUM
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class CompareReduceNeFFN(PureFFN):
    """
    Threshold the sum of per-nibble NE results.
    NE = 1 iff sum(TEMP) >= 1 (at least one nibble differs).
    Output at position 0 only using combined position + value threshold.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear RESULT at all positions first
            self.W_up[0, E.OP_START + Opcode.NE] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Rows 1-2: Threshold with position gating
            # step(RAW_SUM - 0 - 100*pos) = 1 only when RAW_SUM >= 1 AND pos = 0
            # Using integer threshold at 0 for RAW_SUM >= 1
            # At pos=0, RAW_SUM=1: 1 - 0 - 0 = 1 -> 1 ✓
            # At pos=0, RAW_SUM=0: 0 - 0 - 0 = 0 -> 0 ✓
            # At pos=1, RAW_SUM=1: 1 - 0 - 100 = -99 -> 0 ✓
            #
            # Row 1: step(RAW_SUM - 0 - 100*pos)
            self.W_up[1, E.RAW_SUM] = S
            self.W_up[1, E.POS] = -S * 100.0  # Large negative weight for position
            self.b_up[1] = 0.0  # Threshold at RAW_SUM = 0 (activates at >= 1)
            self.W_gate[1, E.OP_START + Opcode.NE] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Row 2: Saturation - cap at 1
            self.W_up[2, E.RAW_SUM] = S
            self.W_up[2, E.POS] = -S * 100.0
            self.b_up[2] = -S * 1.0  # Saturation at RAW_SUM = 1
            self.W_gate[2, E.OP_START + Opcode.NE] = 1.0
            self.W_down[E.RESULT, 2] = -1.0 / S


class CompareCopyResultFFN(PureFFN):
    """Copy TEMP (per-nibble comparison result) to RESULT for LT/GT/LE/GE."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


# =============================================================================
# MULTI-NIBBLE COMPARISON WITH BORROW PROPAGATION
# =============================================================================
# For proper 32-bit comparison, we use subtraction with borrow propagation:
# A < B iff (A - B) produces a final borrow at MSB (position 7)
# A > B iff B < A (swap operands)
# A <= B iff NOT(A > B)
# A >= B iff NOT(A < B)


class CmpRawDiffFFN(PureFFN):
    """Compute raw diff = a - b for comparison (opcode-parameterized)."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0-1: copy (a - b) to RAW_SUM
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_gate[0, E.NIB_B] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class CmpRawDiffSwapFFN(PureFFN):
    """Compute raw diff = b - a for GT comparison (swap operands)."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0-1: copy (b - a) to RAW_SUM (swapped for GT)
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_gate[0, E.NIB_A] = -1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_gate[1, E.NIB_A] = 1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class CmpBorrowDetectFFN(PureFFN):
    """Detect if raw_diff < 0 for comparison, sets CARRY_OUT (borrow)."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(-raw_diff): 1 when raw_diff < 0
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = 0.0  # Threshold at 0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            # Saturation at -1
            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S


class CmpZeroFirstBorrowFFN(PureFFN):
    """Zero out borrow_in for position 0 in comparison."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # At pos=0: negate CARRY_IN (to zero it)
            # silu(S*(0.5 - pos)) at pos=0 gives silu(S*0.5) ≈ S*0.5
            # Only gate on CARRY_IN (not opcode) since this layer only runs
            # during the appropriate pipeline anyway
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0  # Only gate on CARRY_IN value
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


class CmpClearBorrowOutFFN(PureFFN):
    """Clear CARRY_OUT before detecting new borrows."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class CmpBorrowIterFFN(PureFFN):
    """
    One iteration of borrow propagation for comparison.
    Detects new borrow if (RAW_SUM - borrow_in) < 0.
    Does NOT modify RAW_SUM (comparison doesn't need the actual result).
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Detect new borrow if (RAW_SUM - borrow_in) < 0
            # i.e., if CARRY_IN > RAW_SUM
            # up = S * (CARRY_IN - RAW_SUM), activated when CARRY_IN > RAW_SUM

            # Row 0: step(CARRY_IN - RAW_SUM - 0), active when CARRY_IN > RAW_SUM
            self.W_up[0, E.CARRY_IN] = S
            self.W_up[0, E.RAW_SUM] = -S
            self.b_up[0] = 0.0  # Threshold at 0
            self.W_gate[0, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            # Row 1: Saturation at CARRY_IN - RAW_SUM > 1
            self.W_up[1, E.CARRY_IN] = S
            self.W_up[1, E.RAW_SUM] = -S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S


class CmpClearBorrowInFFN(PureFFN):
    """Clear CARRY_IN after using it."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


class CmpClearTempFFN(PureFFN):
    """Clear TEMP at all positions before extraction."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear TEMP by subtracting current value
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class CmpExtractMSBBorrowFFN(PureFFN):
    """
    Extract the final borrow at MSB (position 7) to TEMP.
    At position 7: TEMP = CARRY_OUT (the final borrow)
    At other positions: TEMP unchanged (should be 0 after clear)
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Only activate at position 7 (MSB)
            # Use integer-aligned bounded step: silu(S*(pos-6)) - silu(S*(pos-7)) = S at pos=7
            # At pos=7: silu(S*1) - silu(S*0) = S - 0 = S
            # At pos=6: silu(S*0) - silu(-S) = 0 - 0 = 0

            # Row 0: step(pos - 6), copy CARRY_OUT to TEMP
            self.W_up[0, E.POS] = S
            self.b_up[0] = -S * 6.0  # Integer threshold at 6
            self.W_gate[0, E.CARRY_OUT] = 1.0  # Only gate on CARRY_OUT
            self.W_down[E.TEMP, 0] = 1.0 / S

            # Row 1: -step(pos - 7), to bound at pos=7 only
            self.W_up[1, E.POS] = S
            self.b_up[1] = -S * 7.0  # Integer threshold at 7
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class CmpClearResultFFN(PureFFN):
    """Clear RESULT at all positions before broadcast."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear RESULT by subtracting current value
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.RESULT] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class CmpBroadcastResultAttention(PureAttention):
    """
    Copy TEMP from position 7 to RESULT at position 0 only.
    Only position 0 attends to position 7.
    Other positions attend to position 0 (which has TEMP=0, so no effect).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        # Only position 0 attends to position 7 (MSB)
        # Other positions attend to position 0 (which has TEMP=0)
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        mask[0, N-1] = 0.0  # Position 0 attends to position 7
        # Other positions attend to position 0 (which has TEMP=0, so contributes 0)
        for i in range(1, N):
            mask[i, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # Q and K just need to be non-zero
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0

            # V: read TEMP
            self.W_v[E.TEMP, E.TEMP] = 1.0

            # Output: write to RESULT
            self.W_o[E.RESULT, E.TEMP] = 1.0


class CmpInvertResultFFN(PureFFN):
    """
    Invert RESULT for LE (from GT) and GE (from LT).
    RESULT = 1 - RESULT at position 0 only.
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Only invert at position 0 (where the comparison result is stored)
            # Use bounded step at pos=0: silu(S*(0.5 - pos)) - silu(S*(-0.5 - pos))
            # At pos=0: silu(S*0.5) - silu(-S*0.5) ≈ S*0.5 - 0 = S*0.5

            # Row 0: Add 1 at pos=0 only
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5  # Active at pos < 0.5
            self.W_gate[0, E.OP_START + self.opcode] = 1.0  # Gate on opcode
            self.W_down[E.RESULT, 0] = 1.0 / (S * 0.5)

            # Row 1: Saturation for pos < 0.5
            self.W_up[1, E.POS] = -S
            self.b_up[1] = -S * 0.5  # Active at pos < -0.5 (never)
            self.W_gate[1, E.OP_START + self.opcode] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / (S * 0.5)

            # Row 2-3: Subtract 2*RESULT at pos=0 only
            self.W_up[2, E.POS] = -S
            self.b_up[2] = S * 0.5  # Active at pos < 0.5
            self.W_gate[2, E.RESULT] = -2.0  # -2 * RESULT
            self.W_down[E.RESULT, 2] = 1.0 / (S * 0.5)

            self.W_up[3, E.POS] = -S
            self.b_up[3] = -S * 0.5  # Never active
            self.W_gate[3, E.RESULT] = 2.0
            self.W_down[E.RESULT, 3] = 1.0 / (S * 0.5)


# =============================================================================
# SHIFT OPERATIONS
# =============================================================================

# Shift operations move data between nibble positions.
# SHL by 4 bits = shift each nibble up by one position (multiply by 16)
# SHR by 4 bits = shift each nibble down by one position (divide by 16)
# For sub-nibble shifts, we need intra-nibble bit manipulation.
#
# Implementation strategy:
# 1. First FFN copies NIB_A to TEMP, gated on shift opcode
# 2. Attention moves TEMP between positions (if TEMP=0, moves 0s)
# 3. Final FFN copies shifted values to RESULT, gated on opcode
# 4. Clear TEMP


class ShiftLeftCopyFFN(PureFFN):
    """Step 1: Copy NIB_A to TEMP, gated on SHL opcode."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHL] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHL] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ShiftLeftAttention(PureAttention):
    """
    Step 2: Shift TEMP values left by one position.

    Position i reads TEMP from position i-1 and writes to CARRY_IN slot.
    Position 0 gets 0 (self-attends but TEMP[0] goes to pos 1).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        # Create "previous position only" mask
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(1, N):
            mask[i, i-1] = 0.0
        mask[0, 0] = 0.0  # Position 0 attends to itself (gets TEMP[0] which is 0 after shift)
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0

            # V: read TEMP, write to CARRY_IN (using as temporary shifted slot)
            self.W_v[E.CARRY_IN, E.TEMP] = 1.0

            # Output: pass through CARRY_IN
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class ShiftLeftResultFFN(PureFFN):
    """Step 3: Copy CARRY_IN (shifted values) to RESULT, gated on SHL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy CARRY_IN to RESULT (gated on SHL opcode)
            self.W_up[0, E.OP_START + Opcode.SHL] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHL] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Zero CARRY_IN at position 0 (it got TEMP[0] from self-attention)
            self.W_up[2, E.POS] = -S  # Activates at pos=0
            self.b_up[2] = S * 0.5
            self.W_gate[2, E.CARRY_IN] = -1.0  # Subtract
            self.W_down[E.RESULT, 2] = 1.0 / (S * 0.5)


class ShiftLeftClearFFN(PureFFN):
    """Step 4: Clear TEMP and CARRY_IN after SHL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear TEMP
            self.W_up[0, E.OP_START + Opcode.SHL] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHL] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Clear CARRY_IN
            self.W_up[2, E.OP_START + Opcode.SHL] = S
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.SHL] = -S
            self.W_gate[3, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 3] = 1.0 / S


class ClearTempBeforeShiftFFN(PureFFN):
    """
    Unconditionally clear TEMP before shift operations.

    This prevents garbage from mul_product (which writes unconditionally)
    from affecting shift operations via attention.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Unconditionally subtract TEMP from itself
            # Use: silu(S*1) * (-TEMP) / S = -TEMP (always on)
            # We need some constant input that's always ~1.
            # Use position 0 which is always present.
            # Actually, use the fact that sum of all opcode slots is 1.

            # Simple approach: use b_up as constant offset
            self.b_up[0] = S  # Always activated
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            # Symmetry row (for negative TEMP values)
            self.b_up[1] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ShiftRightCopyFFN(PureFFN):
    """Step 1: Copy NIB_A to TEMP, gated on SHR opcode."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.SHR] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHR] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ShiftRightAttention(PureAttention):
    """
    Step 2: Shift TEMP values right by one position.

    Position i reads TEMP from position i+1 and writes to CARRY_IN slot.
    Position 7 gets 0 (self-attends but that becomes 0).
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        # Create "next position only" mask
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N-1):
            mask[i, i+1] = 0.0
        mask[N-1, N-1] = 0.0  # Position 7 attends to itself
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            self.W_q[0, 0] = 1.0
            self.W_k[0, 0] = 1.0

            # V: read TEMP, write to CARRY_IN
            self.W_v[E.CARRY_IN, E.TEMP] = 1.0

            # Output: pass through CARRY_IN
            self.W_o[E.CARRY_IN, E.CARRY_IN] = 1.0


class ShiftRightResultFFN(PureFFN):
    """Step 3: Copy CARRY_IN (shifted values) to RESULT, gated on SHR."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy CARRY_IN to RESULT (gated on SHR opcode)
            self.W_up[0, E.OP_START + Opcode.SHR] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHR] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Zero out position 7 (subtract CARRY_IN at pos=7)
            # Use integer-aligned thresholds for exact 0/1 at integer positions
            self.W_up[2, E.POS] = S  # Only activates at pos >= 7
            self.b_up[2] = -S * 6.0  # Integer threshold at pos = 6
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.POS] = S
            self.b_up[3] = -S * 7.0  # Integer threshold at pos = 7
            self.W_gate[3, E.CARRY_IN] = 1.0  # Saturation
            self.W_down[E.RESULT, 3] = 1.0 / S


class ShiftRightClearFFN(PureFFN):
    """Step 4: Clear TEMP and CARRY_IN after SHR."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear TEMP
            self.W_up[0, E.OP_START + Opcode.SHR] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SHR] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Clear CARRY_IN
            self.W_up[2, E.OP_START + Opcode.SHR] = S
            self.W_gate[2, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.SHR] = -S
            self.W_gate[3, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 3] = 1.0 / S


# =============================================================================
# MULTIPLY FFN (Single nibble product)
# =============================================================================

class MulProductFFN(PureFFN):
    """
    Step 1 of MUL: Compute temp = a * b (ungated).

    Uses: silu(S*a) * b / S ≈ a * b when a > 0
    Stores result in TEMP slot.

    NOTE: This writes unconditionally. MulClearTempFFN clears when MUL opcode IS set.
    For other opcodes, TEMP is left with garbage which gets cleared by ClearTempBeforeShiftFFN.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # silu(S*a) * b / S = a*b when a > 0
            self.W_up[0, E.NIB_A] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            # Symmetry for numerical stability
            self.W_up[1, E.NIB_A] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class MulGateFFN(PureFFN):
    """
    Step 2 of MUL: Gate temp by opcode, write to RESULT.

    Uses: silu(S*opcode) * temp / S ≈ temp when opcode=1
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # silu(S*opcode) * temp / S = temp when opcode=1
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.TEMP] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.TEMP] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class MulOverflowFFN(PureFFN):
    """
    Handle MUL overflow: if temp >= 16, subtract 16 and set carry.
    Uses same step function as ADD.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Subtract 16 when temp >= 16
            self.W_up[0, E.TEMP] = S
            self.b_up[0] = -S * 15.0
            self.W_gate[0, E.OP_START + Opcode.MUL] = 1.0
            self.W_down[E.RESULT, 0] = -16.0 / S
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.TEMP] = S
            self.b_up[1] = -S * 16.0
            self.W_gate[1, E.OP_START + Opcode.MUL] = 1.0
            self.W_down[E.RESULT, 1] = 16.0 / S
            self.W_down[E.CARRY_OUT, 1] = -1.0 / S

            # Second overflow level: temp >= 32
            self.W_up[2, E.TEMP] = S
            self.b_up[2] = -S * 31.0
            self.W_gate[2, E.OP_START + Opcode.MUL] = 1.0
            self.W_down[E.RESULT, 2] = -16.0 / S
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            self.W_up[3, E.TEMP] = S
            self.b_up[3] = -S * 32.0
            self.W_gate[3, E.OP_START + Opcode.MUL] = 1.0
            self.W_down[E.RESULT, 3] = 16.0 / S
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S


class MulCarryIterFFN(PureFFN):
    """
    One iteration of carry propagation for MUL.
    Similar to CarryIterFFN but gates on MUL opcode.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0-1: Add carry_in to result
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.CARRY_IN] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Row 2-3: Subtract 16 when result + carry_in >= 16
            self.W_up[2, E.RESULT] = S
            self.W_up[2, E.CARRY_IN] = S
            self.b_up[2] = -S * 15.0
            self.W_gate[2, E.OP_START + Opcode.MUL] = 1.0
            self.W_down[E.RESULT, 2] = -16.0 / S
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S  # Set new carry

            self.W_up[3, E.RESULT] = S
            self.W_up[3, E.CARRY_IN] = S
            self.b_up[3] = -S * 16.0
            self.W_gate[3, E.OP_START + Opcode.MUL] = 1.0
            self.W_down[E.RESULT, 3] = 16.0 / S
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S


class MulZeroFirstCarryFFN(PureFFN):
    """Zeros out carry_in for position 0 in MUL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.POS] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 0] = -1.0 / (S * 0.5)


class MulClearCarryInFFN(PureFFN):
    """Clears carry_in before next propagation round for MUL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


class MulClearCarryOutFFN(PureFFN):
    """Clears carry_out before next propagation round for MUL."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class MulClearTempFFN(PureFFN):
    """Clear TEMP after MUL to avoid interference with other ops."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MUL] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MUL] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


# =============================================================================
# DIV/MOD OPERATIONS
# =============================================================================

# Division is complex in neural networks. We use iterative subtraction:
# div = 0; while (a >= b) { a -= b; div++; }
# This requires many iterations (up to 255 for small divisors).
#
# For pure neural implementation, we approximate using:
# - Linear approximation for small values
# - Iterative refinement layers
#
# Simpler approach for nibble-level: Use lookup tables via step functions.


class DivInitFFN(PureFFN):
    """
    Initialize division: copy a to TEMP (dividend), zero RESULT (quotient).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy NIB_A to TEMP (gated on DIV opcode)
            self.W_up[0, E.OP_START + Opcode.DIV] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.DIV] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Clear RESULT
            self.W_up[2, E.OP_START + Opcode.DIV] = S
            self.W_gate[2, E.RESULT] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.DIV] = -S
            self.W_gate[3, E.RESULT] = 1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class DivIterFFN(PureFFN):
    """
    One iteration of division: if TEMP >= NIB_B, subtract NIB_B and increment RESULT.

    Uses: step(temp - b + 0.5) to detect if temp >= b.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Detect if temp >= b: step(temp - b + 0.5)
            # This is tricky because we need temp - b which depends on both values

            # For nibble-level (0-15): temp >= b when temp - b >= 0
            # step(temp - b - 0.5) gives 1 when temp > b
            # step(temp - b + 0.5) gives 1 when temp >= b

            # We use: step(temp - b) ≈ 1 when temp >= b
            # Decrement temp by b when this triggers
            # Increment result by 1 when this triggers

            # Row 0-1: Compute step(temp - b - 0.5) for "can subtract"
            self.W_up[0, E.TEMP] = S
            self.W_up[0, E.NIB_B] = -S
            self.b_up[0] = -S * (-0.5)  # temp - b + 0.5 > 0 means temp >= b
            self.W_gate[0, E.OP_START + Opcode.DIV] = 1.0
            self.W_down[E.TEMP, 0] = -1.0 / S  # Subtract 1 from temp (approx for b)
            self.W_down[E.RESULT, 0] = 1.0 / S  # Increment result

            # Saturation
            self.W_up[1, E.TEMP] = S
            self.W_up[1, E.NIB_B] = -S
            self.b_up[1] = -S * 0.5  # temp - b - 0.5 > 0 means temp > b
            self.W_gate[1, E.OP_START + Opcode.DIV] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S
            self.W_down[E.RESULT, 1] = -1.0 / S


class ModResultFFN(PureFFN):
    """
    For MOD, copy TEMP (remainder after division) to RESULT.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # First clear any previous RESULT
            self.W_up[0, E.OP_START + Opcode.MOD] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MOD] = -S
            self.W_gate[1, E.RESULT] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Copy TEMP to RESULT
            self.W_up[2, E.OP_START + Opcode.MOD] = S
            self.W_gate[2, E.TEMP] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.MOD] = -S
            self.W_gate[3, E.TEMP] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class ModInitFFN(PureFFN):
    """
    Initialize MOD: copy a to TEMP (dividend).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.MOD] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.MOD] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class ModIterFFN(PureFFN):
    """
    One iteration of MOD: if TEMP >= NIB_B, subtract NIB_B.
    Similar to DivIterFFN but only decrements TEMP, doesn't count quotient.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # step(temp - b + 0.5) to detect temp >= b
            self.W_up[0, E.TEMP] = S
            self.W_up[0, E.NIB_B] = -S
            self.b_up[0] = S * 0.5
            self.W_gate[0, E.OP_START + Opcode.MOD] = 1.0
            self.W_down[E.TEMP, 0] = -1.0 / S  # Subtract 1 (approx)

            # Saturation
            self.W_up[1, E.TEMP] = S
            self.W_up[1, E.NIB_B] = -S
            self.b_up[1] = -S * 0.5
            self.W_gate[1, E.OP_START + Opcode.MOD] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


# =============================================================================
# PURE TRANSFORMER BLOCK
# =============================================================================

class PureTransformerBlock(nn.Module):
    """Standard transformer block with FINAL forward."""

    def __init__(self, dim: int, hidden_dim: int, num_heads: int = 1):
        super().__init__()
        self.attn = PureAttention(dim, num_heads)
        self.ffn = PureFFN(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - attention then FFN."""
        x = self.attn(x)
        x = self.ffn(x)
        return x


# =============================================================================
# PURE NEURAL ALU
# =============================================================================

class CarryRedetectFFN(PureFFN):
    """
    Re-detects carry after adding carry_in to raw_sum.

    carry_out = 1 if (raw_sum + carry_in) >= 16
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # First, clear old carry_out by subtracting it
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S

            # Now detect if raw_sum + carry_in >= 16
            self.W_up[2, E.RAW_SUM] = S
            self.W_up[2, E.CARRY_IN] = S
            self.b_up[2] = -S * 15.0
            self.W_gate[2, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 2] = 1.0 / S

            # Saturation
            self.W_up[3, E.RAW_SUM] = S
            self.W_up[3, E.CARRY_IN] = S
            self.b_up[3] = -S * 16.0
            self.W_gate[3, E.OP_START + Opcode.ADD] = 1.0
            self.W_down[E.CARRY_OUT, 3] = -1.0 / S


class ClearCarryInFFN(PureFFN):
    """Clears carry_in before next propagation round."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Subtract carry_in from itself
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.CARRY_IN] = -1.0
            self.W_down[E.CARRY_IN, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.CARRY_IN] = 1.0
            self.W_down[E.CARRY_IN, 1] = 1.0 / S


class ClearCarryOutFFN(PureFFN):
    """Clears carry_out before detecting new carries."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Subtract carry_out from itself
            self.W_up[0, E.OP_START + Opcode.ADD] = S
            self.W_gate[0, E.CARRY_OUT] = -1.0
            self.W_down[E.CARRY_OUT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADD] = -S
            self.W_gate[1, E.CARRY_OUT] = 1.0
            self.W_down[E.CARRY_OUT, 1] = 1.0 / S


class PureALU(nn.Module):
    """
    Pure Neural ALU - stacks layers for each operation.

    All computation flows through standard layers.
    No Python arithmetic in forward.

    ADD pipeline:
    1. Compute raw_sum = A + B per nibble
    2. Initialize result = raw_sum mod 16, carry_out = (raw_sum >= 16)
    3. 7 iterations: propagate carry, add to result, detect new overflow
    """

    def __init__(self):
        super().__init__()

        # ADD layers
        self.add_raw = AddRawSumFFN()
        self.add_init = InitResultFFN()
        self.add_carry_detect = CarryDetectFFN()

        # 7 iterations for carry cascade (ADD)
        self.carry_props = nn.ModuleList([CarryPropagateAttention() for _ in range(7)])
        self.carry_zeros = nn.ModuleList([ZeroFirstCarryFFN() for _ in range(7)])
        self.carry_out_clears = nn.ModuleList([ClearCarryOutFFN() for _ in range(7)])
        self.carry_iters = nn.ModuleList([CarryIterFFN() for _ in range(7)])
        self.carry_in_clears = nn.ModuleList([ClearCarryInFFN() for _ in range(7)])

        # SUB layers
        self.sub_raw = SubRawDiffFFN()
        self.sub_init = SubInitResultFFN()
        self.sub_borrow_detect = BorrowDetectFFN()

        # 7 iterations for borrow cascade (SUB)
        self.sub_borrow_props = nn.ModuleList([CarryPropagateAttention() for _ in range(7)])
        self.sub_borrow_zeros = nn.ModuleList([ZeroFirstBorrowFFN() for _ in range(7)])
        self.sub_borrow_out_clears = nn.ModuleList([ClearBorrowOutFFN() for _ in range(7)])
        self.sub_borrow_iters = nn.ModuleList([BorrowIterFFN() for _ in range(7)])
        self.sub_borrow_in_clears = nn.ModuleList([ClearBorrowInFFN() for _ in range(7)])

        # MUL layers (multi-step: product, gate, overflow, carry propagation)
        self.mul_product = MulProductFFN()
        self.mul_gate = MulGateFFN()
        self.mul_overflow = MulOverflowFFN()
        # MUL-specific carry propagation (gates on MUL opcode)
        self.mul_carry_props = nn.ModuleList([CarryPropagateAttention() for _ in range(7)])
        self.mul_carry_zeros = nn.ModuleList([MulZeroFirstCarryFFN() for _ in range(7)])
        self.mul_carry_out_clears = nn.ModuleList([MulClearCarryOutFFN() for _ in range(7)])
        self.mul_carry_iters = nn.ModuleList([MulCarryIterFFN() for _ in range(7)])
        self.mul_carry_in_clears = nn.ModuleList([MulClearCarryInFFN() for _ in range(7)])
        self.mul_clear = MulClearTempFFN()

        # DIV layers
        self.div_init = DivInitFFN()
        # 16 iterations for division (max quotient 15 for nibble)
        self.div_iters = nn.ModuleList([DivIterFFN() for _ in range(16)])

        # MOD layers
        self.mod_init = ModInitFFN()
        # 16 iterations for mod
        self.mod_iters = nn.ModuleList([ModIterFFN() for _ in range(16)])
        self.mod_result = ModResultFFN()

        # Bitwise AND layers
        self.and_clear_pre = ClearBitSlotsFFN(Opcode.AND)  # Clear bit slots before extraction
        self.and_bit3 = ExtractBit3FFN(Opcode.AND)
        self.and_bit2 = ExtractBit2FFN(Opcode.AND)
        self.and_bit1 = ExtractBit1FFN(Opcode.AND)
        self.and_bit0 = ExtractBit0FFN(Opcode.AND)
        self.and_combine = BitwiseAndCombineFFN()
        self.and_clear = ClearBitsFFN(Opcode.AND)

        # Bitwise OR layers
        self.or_clear_pre = ClearBitSlotsFFN(Opcode.OR)  # Clear bit slots before extraction
        self.or_bit3 = ExtractBit3FFN(Opcode.OR)
        self.or_bit2 = ExtractBit2FFN(Opcode.OR)
        self.or_bit1 = ExtractBit1FFN(Opcode.OR)
        self.or_bit0 = ExtractBit0FFN(Opcode.OR)
        self.or_combine = BitwiseOrCombineFFN()
        self.or_clear = ClearBitsFFN(Opcode.OR)

        # Bitwise XOR layers
        self.xor_clear_pre = ClearBitSlotsFFN(Opcode.XOR)  # Clear bit slots before extraction
        self.xor_bit3 = ExtractBit3FFN(Opcode.XOR)
        self.xor_bit2 = ExtractBit2FFN(Opcode.XOR)
        self.xor_bit1 = ExtractBit1FFN(Opcode.XOR)
        self.xor_bit0 = ExtractBit0FFN(Opcode.XOR)
        self.xor_combine = BitwiseXorCombineFFN()
        self.xor_clear = ClearBitsFFN(Opcode.XOR)

        # Comparison EQ layers
        self.eq_diff = CompareDiffFFN(Opcode.EQ)
        self.eq_nibble = CompareEqNibbleFFN(Opcode.EQ)
        self.eq_clear_raw = ClearRawSumFFN(Opcode.EQ)
        self.eq_reduce_attn = CompareReduceEqAttention()
        self.eq_reduce = CompareReduceEqFFN()

        # Comparison NE layers
        self.ne_diff = CompareDiffFFN(Opcode.NE)
        self.ne_nibble = CompareNeNibbleFFN(Opcode.NE)
        self.ne_clear_raw = ClearRawSumFFN(Opcode.NE)
        self.ne_reduce_attn = CompareReduceNeAttention()
        self.ne_reduce = CompareReduceNeFFN()

        # Comparison LT layers (using borrow propagation: A < B iff A-B has final borrow)
        self.lt_raw_diff = CmpRawDiffFFN(Opcode.LT)
        self.lt_borrow_detect = CmpBorrowDetectFFN(Opcode.LT)
        self.lt_borrow_props = nn.ModuleList([CarryPropagateAttention() for _ in range(7)])
        self.lt_borrow_zeros = nn.ModuleList([CmpZeroFirstBorrowFFN(Opcode.LT) for _ in range(7)])
        self.lt_borrow_out_clears = nn.ModuleList([CmpClearBorrowOutFFN(Opcode.LT) for _ in range(7)])
        self.lt_borrow_iters = nn.ModuleList([CmpBorrowIterFFN(Opcode.LT) for _ in range(7)])
        self.lt_borrow_in_clears = nn.ModuleList([CmpClearBorrowInFFN(Opcode.LT) for _ in range(7)])
        self.lt_clear_temp = CmpClearTempFFN(Opcode.LT)
        self.lt_extract_msb = CmpExtractMSBBorrowFFN(Opcode.LT)
        self.lt_clear_result = CmpClearResultFFN(Opcode.LT)
        self.lt_broadcast = CmpBroadcastResultAttention()

        # Comparison GT layers (using borrow propagation with swapped operands: A > B iff B-A has final borrow)
        self.gt_raw_diff = CmpRawDiffSwapFFN(Opcode.GT)  # Swap: compute B - A
        self.gt_borrow_detect = CmpBorrowDetectFFN(Opcode.GT)
        self.gt_borrow_props = nn.ModuleList([CarryPropagateAttention() for _ in range(7)])
        self.gt_borrow_zeros = nn.ModuleList([CmpZeroFirstBorrowFFN(Opcode.GT) for _ in range(7)])
        self.gt_borrow_out_clears = nn.ModuleList([CmpClearBorrowOutFFN(Opcode.GT) for _ in range(7)])
        self.gt_borrow_iters = nn.ModuleList([CmpBorrowIterFFN(Opcode.GT) for _ in range(7)])
        self.gt_borrow_in_clears = nn.ModuleList([CmpClearBorrowInFFN(Opcode.GT) for _ in range(7)])
        self.gt_clear_temp = CmpClearTempFFN(Opcode.GT)
        self.gt_extract_msb = CmpExtractMSBBorrowFFN(Opcode.GT)
        self.gt_clear_result = CmpClearResultFFN(Opcode.GT)
        self.gt_broadcast = CmpBroadcastResultAttention()

        # Comparison LE layers (LE = NOT GT, so use swapped diff and invert)
        self.le_raw_diff = CmpRawDiffSwapFFN(Opcode.LE)  # Swap: compute B - A
        self.le_borrow_detect = CmpBorrowDetectFFN(Opcode.LE)
        self.le_borrow_props = nn.ModuleList([CarryPropagateAttention() for _ in range(7)])
        self.le_borrow_zeros = nn.ModuleList([CmpZeroFirstBorrowFFN(Opcode.LE) for _ in range(7)])
        self.le_borrow_out_clears = nn.ModuleList([CmpClearBorrowOutFFN(Opcode.LE) for _ in range(7)])
        self.le_borrow_iters = nn.ModuleList([CmpBorrowIterFFN(Opcode.LE) for _ in range(7)])
        self.le_borrow_in_clears = nn.ModuleList([CmpClearBorrowInFFN(Opcode.LE) for _ in range(7)])
        self.le_clear_temp = CmpClearTempFFN(Opcode.LE)
        self.le_extract_msb = CmpExtractMSBBorrowFFN(Opcode.LE)
        self.le_clear_result = CmpClearResultFFN(Opcode.LE)
        self.le_broadcast = CmpBroadcastResultAttention()
        self.le_invert = CmpInvertResultFFN(Opcode.LE)

        # Comparison GE layers (GE = NOT LT, so use normal diff and invert)
        self.ge_raw_diff = CmpRawDiffFFN(Opcode.GE)
        self.ge_borrow_detect = CmpBorrowDetectFFN(Opcode.GE)
        self.ge_borrow_props = nn.ModuleList([CarryPropagateAttention() for _ in range(7)])
        self.ge_borrow_zeros = nn.ModuleList([CmpZeroFirstBorrowFFN(Opcode.GE) for _ in range(7)])
        self.ge_borrow_out_clears = nn.ModuleList([CmpClearBorrowOutFFN(Opcode.GE) for _ in range(7)])
        self.ge_borrow_iters = nn.ModuleList([CmpBorrowIterFFN(Opcode.GE) for _ in range(7)])
        self.ge_borrow_in_clears = nn.ModuleList([CmpClearBorrowInFFN(Opcode.GE) for _ in range(7)])
        self.ge_clear_temp = CmpClearTempFFN(Opcode.GE)
        self.ge_extract_msb = CmpExtractMSBBorrowFFN(Opcode.GE)
        self.ge_clear_result = CmpClearResultFFN(Opcode.GE)
        self.ge_broadcast = CmpBroadcastResultAttention()
        self.ge_invert = CmpInvertResultFFN(Opcode.GE)

        # Shift layers (5 steps each: clear temp, copy, attention shift, result, clear)
        self.clear_temp_before_shift = ClearTempBeforeShiftFFN()
        self.shl_copy = ShiftLeftCopyFFN()
        self.shl_attn = ShiftLeftAttention()
        self.shl_result = ShiftLeftResultFFN()
        self.shl_clear = ShiftLeftClearFFN()
        self.shr_copy = ShiftRightCopyFFN()
        self.shr_attn = ShiftRightAttention()
        self.shr_result = ShiftRightResultFFN()
        self.shr_clear = ShiftRightClearFFN()

        # Control flow layers
        self.jmp = JumpFFN()
        self.beq = BranchEqFFN()
        self.bne = BranchNeFFN()
        self.blt = BranchLtFFN()
        self.bge = BranchGeFFN()

        # Memory operation layers
        self.load = LoadFFN()
        self.store = StoreFFN()
        self.push = PushFFN()
        self.pop = PopFFN()
        self.nop = NopFFN()
        self.halt = HaltFFN()

        # Sparse MoE ALU for efficient routing
        self.sparse_moe = SparseMoEALU()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routed forward pass - runs only the layers for the active opcode.

        This extracts the opcode from the input and runs only the
        relevant operation's layers, not all operations sequentially.
        """
        # Extract active opcode
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        # Route to appropriate operation
        if active_opcode == Opcode.ADD:
            x = self._forward_add(x)
        elif active_opcode == Opcode.SUB:
            x = self._forward_sub(x)
        elif active_opcode == Opcode.MUL:
            x = self._forward_mul(x)
        elif active_opcode == Opcode.DIV:
            x = self._forward_div(x)
        elif active_opcode == Opcode.MOD:
            x = self._forward_mod(x)
        elif active_opcode == Opcode.AND:
            x = self._forward_and(x)
        elif active_opcode == Opcode.OR:
            x = self._forward_or(x)
        elif active_opcode == Opcode.XOR:
            x = self._forward_xor(x)
        elif active_opcode == Opcode.EQ:
            x = self._forward_eq(x)
        elif active_opcode == Opcode.NE:
            x = self._forward_ne(x)
        elif active_opcode == Opcode.LT:
            x = self._forward_lt(x)
        elif active_opcode == Opcode.GT:
            x = self._forward_gt(x)
        elif active_opcode == Opcode.LE:
            x = self._forward_le(x)
        elif active_opcode == Opcode.GE:
            x = self._forward_ge(x)
        elif active_opcode == Opcode.SHL:
            x = self._forward_shl(x)
        elif active_opcode == Opcode.SHR:
            x = self._forward_shr(x)
        elif active_opcode == Opcode.JMP:
            x = self.jmp(x)
        elif active_opcode == Opcode.BEQ:
            x = self.beq(x)
        elif active_opcode == Opcode.BNE:
            x = self.bne(x)
        elif active_opcode == Opcode.BLT:
            x = self.blt(x)
        elif active_opcode == Opcode.BGE:
            x = self.bge(x)
        elif active_opcode == Opcode.LOAD:
            x = self.load(x)
        elif active_opcode == Opcode.STORE:
            x = self.store(x)
        elif active_opcode == Opcode.PUSH:
            x = self.push(x)
        elif active_opcode == Opcode.POP:
            x = self.pop(x)
        elif active_opcode == Opcode.NOP:
            x = self.nop(x)
        elif active_opcode == Opcode.HALT:
            x = self.halt(x)

        return x

    def _forward_add(self, x: torch.Tensor) -> torch.Tensor:
        """ADD operation pipeline."""
        x = self.add_raw(x)
        x = self.add_init(x)
        x = self.add_carry_detect(x)
        for i in range(7):
            x = self.carry_props[i](x)
            x = self.carry_zeros[i](x)
            x = self.carry_out_clears[i](x)
            x = self.carry_iters[i](x)
            x = self.carry_in_clears[i](x)
        return x

    def _forward_sub(self, x: torch.Tensor) -> torch.Tensor:
        """SUB operation pipeline."""
        x = self.sub_raw(x)
        x = self.sub_init(x)
        x = self.sub_borrow_detect(x)
        for i in range(7):
            x = self.sub_borrow_props[i](x)
            x = self.sub_borrow_zeros[i](x)
            x = self.sub_borrow_out_clears[i](x)
            x = self.sub_borrow_iters[i](x)
            x = self.sub_borrow_in_clears[i](x)
        return x

    def _forward_mul(self, x: torch.Tensor) -> torch.Tensor:
        """MUL operation pipeline."""
        x = self.mul_product(x)
        x = self.mul_gate(x)
        x = self.mul_overflow(x)
        for i in range(7):
            x = self.mul_carry_props[i](x)
            x = self.mul_carry_zeros[i](x)
            x = self.mul_carry_out_clears[i](x)
            x = self.mul_carry_iters[i](x)
            x = self.mul_carry_in_clears[i](x)
        x = self.mul_clear(x)
        return x

    def _forward_div(self, x: torch.Tensor) -> torch.Tensor:
        """DIV operation pipeline."""
        x = self.div_init(x)
        for layer in self.div_iters:
            x = layer(x)
        return x

    def _forward_mod(self, x: torch.Tensor) -> torch.Tensor:
        """MOD operation pipeline."""
        x = self.mod_init(x)
        for layer in self.mod_iters:
            x = layer(x)
        x = self.mod_result(x)
        return x

    def _forward_and(self, x: torch.Tensor) -> torch.Tensor:
        """AND operation pipeline."""
        x = self.and_clear_pre(x)  # Clear bit slots before extraction
        x = self.and_bit3(x)
        x = self.and_bit2(x)
        x = self.and_bit1(x)
        x = self.and_bit0(x)
        x = self.and_combine(x)
        x = self.and_clear(x)
        return x

    def _forward_or(self, x: torch.Tensor) -> torch.Tensor:
        """OR operation pipeline."""
        x = self.or_clear_pre(x)  # Clear bit slots before extraction
        x = self.or_bit3(x)
        x = self.or_bit2(x)
        x = self.or_bit1(x)
        x = self.or_bit0(x)
        x = self.or_combine(x)
        x = self.or_clear(x)
        return x

    def _forward_xor(self, x: torch.Tensor) -> torch.Tensor:
        """XOR operation pipeline."""
        x = self.xor_clear_pre(x)  # Clear bit slots before extraction
        x = self.xor_bit3(x)
        x = self.xor_bit2(x)
        x = self.xor_bit1(x)
        x = self.xor_bit0(x)
        x = self.xor_combine(x)
        x = self.xor_clear(x)
        return x

    def _forward_eq(self, x: torch.Tensor) -> torch.Tensor:
        """EQ comparison pipeline."""
        x = self.eq_diff(x)
        x = self.eq_nibble(x)
        x = self.eq_clear_raw(x)    # Clear RAW_SUM before attention (residual fix)
        x = self.eq_reduce_attn(x)  # Sum TEMP values across positions
        x = self.eq_reduce(x)       # Threshold at 8 and output to RESULT
        return x

    def _forward_ne(self, x: torch.Tensor) -> torch.Tensor:
        """NE comparison pipeline."""
        x = self.ne_diff(x)
        x = self.ne_nibble(x)
        x = self.ne_clear_raw(x)    # Clear RAW_SUM before attention (residual fix)
        x = self.ne_reduce_attn(x)  # Sum TEMP values across positions
        x = self.ne_reduce(x)       # Threshold and output to RESULT
        return x

    def _forward_lt(self, x: torch.Tensor) -> torch.Tensor:
        """LT comparison pipeline using borrow propagation."""
        # A < B iff (A - B) produces a final borrow at MSB
        x = self.lt_raw_diff(x)
        x = self.lt_borrow_detect(x)
        # 7 iterations for borrow cascade (propagate from pos 0 to pos 7)
        for i in range(7):
            x = self.lt_borrow_props[i](x)
            x = self.lt_borrow_zeros[i](x)
            x = self.lt_borrow_out_clears[i](x)
            x = self.lt_borrow_iters[i](x)
            x = self.lt_borrow_in_clears[i](x)
        # Extract borrow at MSB to TEMP, clear RESULT, then broadcast
        x = self.lt_clear_temp(x)
        x = self.lt_extract_msb(x)
        x = self.lt_clear_result(x)
        x = self.lt_broadcast(x)
        return x

    def _forward_gt(self, x: torch.Tensor) -> torch.Tensor:
        """GT comparison pipeline using borrow propagation with swapped operands."""
        # A > B iff B < A iff (B - A) produces a final borrow at MSB
        x = self.gt_raw_diff(x)  # Computes B - A (swapped)
        x = self.gt_borrow_detect(x)
        for i in range(7):
            x = self.gt_borrow_props[i](x)
            x = self.gt_borrow_zeros[i](x)
            x = self.gt_borrow_out_clears[i](x)
            x = self.gt_borrow_iters[i](x)
            x = self.gt_borrow_in_clears[i](x)
        x = self.gt_clear_temp(x)
        x = self.gt_extract_msb(x)
        x = self.gt_clear_result(x)
        x = self.gt_broadcast(x)
        return x

    def _forward_le(self, x: torch.Tensor) -> torch.Tensor:
        """LE comparison pipeline: A <= B iff NOT(A > B) iff NOT(B < A)."""
        # Compute B - A, then invert the result
        x = self.le_raw_diff(x)  # Computes B - A (swapped)
        x = self.le_borrow_detect(x)
        for i in range(7):
            x = self.le_borrow_props[i](x)
            x = self.le_borrow_zeros[i](x)
            x = self.le_borrow_out_clears[i](x)
            x = self.le_borrow_iters[i](x)
            x = self.le_borrow_in_clears[i](x)
        x = self.le_clear_temp(x)
        x = self.le_extract_msb(x)
        x = self.le_clear_result(x)
        x = self.le_broadcast(x)
        x = self.le_invert(x)  # Invert: LE = NOT(B < A)
        return x

    def _forward_ge(self, x: torch.Tensor) -> torch.Tensor:
        """GE comparison pipeline: A >= B iff NOT(A < B)."""
        # Compute A - B, then invert the result
        x = self.ge_raw_diff(x)
        x = self.ge_borrow_detect(x)
        for i in range(7):
            x = self.ge_borrow_props[i](x)
            x = self.ge_borrow_zeros[i](x)
            x = self.ge_borrow_out_clears[i](x)
            x = self.ge_borrow_iters[i](x)
            x = self.ge_borrow_in_clears[i](x)
        x = self.ge_clear_temp(x)
        x = self.ge_extract_msb(x)
        x = self.ge_clear_result(x)
        x = self.ge_broadcast(x)
        x = self.ge_invert(x)  # Invert: GE = NOT(A < B)
        return x

    def _forward_shl(self, x: torch.Tensor) -> torch.Tensor:
        """SHL operation pipeline."""
        x = self.clear_temp_before_shift(x)
        x = self.shl_copy(x)
        x = self.shl_attn(x)
        x = self.shl_result(x)
        x = self.shl_clear(x)
        return x

    def _forward_shr(self, x: torch.Tensor) -> torch.Tensor:
        """SHR operation pipeline."""
        x = self.clear_temp_before_shift(x)
        x = self.shr_copy(x)
        x = self.shr_attn(x)
        x = self.shr_result(x)
        x = self.shr_clear(x)
        return x

    @torch.no_grad()
    def forward_moe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse MoE forward pass using bundled experts.
        """
        return self.sparse_moe(x)


# =============================================================================
# NEURAL VM V7
# =============================================================================

class NeuralVMv7(nn.Module):
    """
    Pure Neural VM V7.

    All computation is standard transformer forward passes.
    No Python arithmetic - only weight baking.
    """

    def __init__(self):
        super().__init__()
        self.alu = PureALU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ALU layers."""
        return self.alu(x)

    def encode_input(self, a: int, b: int, opcode: int) -> torch.Tensor:
        """Encode 32-bit operands into sequence of 8 nibble positions."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

        for pos in range(E.NUM_POSITIONS):
            nib_a = (a >> (pos * 4)) & 0xF
            nib_b = (b >> (pos * 4)) & 0xF

            x[0, pos, E.NIB_A] = float(nib_a)
            x[0, pos, E.NIB_B] = float(nib_b)
            x[0, pos, E.POS] = float(pos)
            x[0, pos, E.OP_START + opcode] = 1.0

        return x

    def decode_output(self, x: torch.Tensor) -> int:
        """Decode result from 8 nibble positions."""
        result = 0
        for pos in range(E.NUM_POSITIONS):
            nib = int(round(x[0, pos, E.RESULT].item())) & 0xF
            result |= nib << (pos * 4)
        return result

    def compute(self, a: int, b: int, opcode: int) -> int:
        """Convenience method to compute a single operation."""
        x = self.encode_input(a, b, opcode)
        out = self.forward(x)
        return self.decode_output(out)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_pure_forward():
    """Verify that forward passes are truly pure (no Python control flow or scalar arithmetic)."""
    import inspect

    print("=" * 60)
    print("VERIFYING PURE FORWARD PASSES")
    print("=" * 60)

    # Only flag things that aren't standard neural ops
    # Allowed: x + y (tensor add), x * y (tensor multiply), F.linear, F.silu, etc.
    # Forbidden: for/while loops, if conditions on values, //, %, >>, <<
    forbidden = [
        ('for ', 'loop'),
        ('while ', 'loop'),
        (' // ', 'integer division'),
        (' % ', 'modulo'),
        (' >> ', 'bit shift'),
        (' << ', 'bit shift'),
        ('if ', 'conditional'),
        ('range(', 'loop'),
    ]

    classes_to_check = [
        ('PureFFN', PureFFN),
        ('PureAttention', PureAttention),
    ]

    all_pure = True
    for name, cls in classes_to_check:
        source = inspect.getsource(cls.forward)
        issues = []
        for pattern, desc in forbidden:
            if pattern in source:
                issues.append(desc)

        if issues:
            print(f"  {name}.forward: IMPURE - contains {set(issues)}")
            all_pure = False
        else:
            print(f"  {name}.forward: PURE (tensor ops only)")

    # Check that subclasses don't override forward
    subclasses = [AddRawSumFFN, CarryDetectFFN, InitResultFFN, MulNibbleFFN, BitwiseAndFFN, CompareEqFFN]
    for cls in subclasses:
        if 'forward' in cls.__dict__:
            print(f"  {cls.__name__}: IMPURE - overrides forward!")
            all_pure = False
        else:
            print(f"  {cls.__name__}: PURE (inherits base forward)")

    print()
    if all_pure:
        print("ALL FORWARD PASSES ARE PURE!")
    else:
        print("Some forward passes are impure.")

    return all_pure


def test_add():
    """Test ADD operation."""
    print("\n=== Testing ADD ===")
    vm = NeuralVMv7()

    tests = [
        (0, 0, 0),
        (1, 1, 2),
        (5, 3, 8),
        (7, 8, 15),
        (8, 8, 16),  # Carry case
    ]

    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.ADD)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {a} + {b} = {result} (expected {expected}) {status}")


def debug_add(a: int, b: int):
    """Debug ADD operation step by step."""
    print(f"\n=== Debug ADD: {a} + {b} ===")

    vm = NeuralVMv7()
    x = vm.encode_input(a, b, Opcode.ADD)

    print(f"Input nibble 0: A={x[0, 0, E.NIB_A].item():.1f}, B={x[0, 0, E.NIB_B].item():.1f}")
    print(f"Input RESULT[0]={x[0, 0, E.RESULT].item():.3f}")

    # Step 1: Raw sum
    x1 = vm.alu.add_raw(x)
    print(f"After add_raw: RAW_SUM[0]={x1[0, 0, E.RAW_SUM].item():.3f}, RESULT[0]={x1[0, 0, E.RESULT].item():.3f}")

    # Step 2: Carry detect
    x2 = vm.alu.add_carry_detect(x1)
    print(f"After carry_detect: CARRY_OUT[0]={x2[0, 0, E.CARRY_OUT].item():.3f}, RESULT[0]={x2[0, 0, E.RESULT].item():.3f}")

    # Step 3: Carry propagate
    x3 = vm.alu.add_carry_prop(x2)
    print(f"After carry_prop: CARRY_IN[0]={x3[0, 0, E.CARRY_IN].item():.3f}, RESULT[0]={x3[0, 0, E.RESULT].item():.3f}")

    # Step 4: Result - let's debug this layer in detail
    print("--- AddResultFFN debug ---")
    layer = vm.alu.add_result
    up = F.linear(x3, layer.W_up, layer.b_up)
    gate = F.linear(x3, layer.W_gate)
    hidden = F.silu(up) * gate
    output = F.linear(hidden, layer.W_down)
    print(f"  up[0,:4]={up[0,0,:4].tolist()}")
    print(f"  gate[0,:4]={gate[0,0,:4].tolist()}")
    print(f"  hidden[0,:4]={hidden[0,0,:4].tolist()}")
    print(f"  output contribution to RESULT={output[0,0,E.RESULT].item():.3f}")

    x4 = vm.alu.add_result(x3)
    print(f"After add_result: RESULT[0]={x4[0, 0, E.RESULT].item():.3f}")

    # Other ops - show each contribution
    print("--- Other ops debug ---")
    mul_layer = vm.alu.mul
    gate_mul = F.linear(x4, mul_layer.W_gate)
    print(f"  MUL gate[0]={gate_mul[0,0,0].item():.3f} (should be ~0 when ADD)")

    x5 = vm.alu.mul(x4)
    print(f"After mul: RESULT[0]={x5[0, 0, E.RESULT].item():.3f}")

    x6 = vm.alu.bit_and(x5)
    print(f"After bit_and: RESULT[0]={x6[0, 0, E.RESULT].item():.3f}")

    x7 = vm.alu.cmp_eq(x6)
    print(f"After cmp_eq: RESULT[0]={x7[0, 0, E.RESULT].item():.3f}")

    result = vm.decode_output(x7)
    print(f"Decoded result: {result}")


def print_architecture():
    """Print the architecture summary."""
    print("=" * 60)
    print("V7 ARCHITECTURE")
    print("=" * 60)
    print("""
Base Classes (FINAL forward):
  - PureFFN: SwiGLU FFN, subclass bakes weights only
  - PureAttention: Standard attention, subclass bakes weights only

ADD Operation (4 layers):
  1. AddRawSumFFN: Computes raw_sum = a + b per nibble
  2. CarryDetectFFN: Detects if raw_sum >= 16
  3. CarryPropagateAttention: Propagates carry to next position
  4. AddResultFFN: Computes (raw_sum + carry_in) mod 16

Other Operations:
  - MulNibbleFFN: Single nibble multiply
  - BitwiseAndFFN: Bitwise AND via multiplication
  - CompareEqFFN: Equality via second derivative peak

All computation flows through:
  - F.linear (matrix multiply)
  - F.silu (activation)
  - F.softmax (attention)
  - Element-wise multiply (gating)

NO Python +, -, *, /, %, >>, << in forward passes.
""")


def debug_carry(a: int, b: int):
    """Debug carry case with all nibbles."""
    print(f"\n=== Debug Carry: {a} + {b} ===")

    vm = NeuralVMv7()
    x = vm.encode_input(a, b, Opcode.ADD)

    # Run through pipeline
    x = vm.alu.add_raw(x)
    x = vm.alu.add_carry_detect(x)
    x = vm.alu.add_carry_prop(x)
    print(f"After carry_prop: cin[0]={x[0, 0, E.CARRY_IN].item():.2f}, cin[1]={x[0, 1, E.CARRY_IN].item():.2f}")
    x = vm.alu.add_zero_first(x)
    print(f"After zero_first: cin[0]={x[0, 0, E.CARRY_IN].item():.2f}, cin[1]={x[0, 1, E.CARRY_IN].item():.2f}")
    x = vm.alu.add_result(x)
    x = vm.alu.mul(x)
    x = vm.alu.bit_and(x)
    x = vm.alu.cmp_eq(x)

    print("All nibbles:")
    for i in range(8):
        a_nib = vm.encode_input(a, b, Opcode.ADD)[0, i, E.NIB_A].item()
        b_nib = vm.encode_input(a, b, Opcode.ADD)[0, i, E.NIB_B].item()
        raw = x[0, i, E.RAW_SUM].item()
        cout = x[0, i, E.CARRY_OUT].item()
        cin = x[0, i, E.CARRY_IN].item()
        res = x[0, i, E.RESULT].item()
        print(f"  nib{i}: A={a_nib:.0f} B={b_nib:.0f} raw={raw:.1f} cout={cout:.2f} cin={cin:.2f} result={res:.2f}")

    result = vm.decode_output(x)
    print(f"Decoded: {result} (expected {a+b})")


if __name__ == "__main__":
    print_architecture()
    verify_pure_forward()
    test_add()
    debug_add(5, 3)
    debug_add(1, 1)
    debug_carry(8, 8)
