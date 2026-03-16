"""
Control flow operations for Neural VM V7.

JMP, BEQ, BNE, BLT, BGE

Branch operations work as follows:
1. Condition is checked at position 0 only
2. If condition met, attention broadcasts "branch flag" to all positions
3. Each position copies NIB_B to RESULT if branch flag is set
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


class BranchConditionAttention(PureAttention):
    """
    Broadcasts the branch condition from position 0 to all positions.
    Used to synchronize the branch decision across all nibbles.

    After BranchCondEqFFN/BranchCondNeFFN writes TEMP at position 0,
    this attention copies TEMP[0] to RAW_SUM at all positions.
    Using RAW_SUM instead of TEMP to avoid residual doubling.
    """

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

        # All positions read from position 0
        N = E.NUM_POSITIONS
        mask = torch.full((N, N), float('-inf'))
        for i in range(N):
            mask[i, 0] = 0.0
        self.register_buffer('mask', mask)

    def _bake_weights(self):
        with torch.no_grad():
            # Copy TEMP from position 0 to RAW_SUM at all positions
            self.W_v[E.RAW_SUM, E.TEMP] = 1.0
            self.W_o[E.RAW_SUM, E.RAW_SUM] = 1.0


class BranchCondEqFFN(PureFFN):
    """
    Compute branch condition for BEQ at position 0 only.
    Writes 1 to TEMP[0] if NIB_A[0] == 0.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Only at position 0: TEMP = 1 if NIB_A == 0
            # silu(1 - A) - silu(-A), gated by position 0
            self.W_up[0, E.NIB_A] = -S
            self.W_up[0, E.POS] = -S * 100  # Position 0 gate
            self.b_up[0] = S * 1.0
            self.W_gate[0, E.OP_START + Opcode.BEQ] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = -S
            self.W_up[1, E.POS] = -S * 100
            self.b_up[1] = 0.0
            self.W_gate[1, E.OP_START + Opcode.BEQ] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class BranchCondNeFFN(PureFFN):
    """
    Compute branch condition for BNE at position 0 only.
    Writes 1 to TEMP[0] if NIB_A[0] != 0.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Only at position 0: TEMP = 1 if NIB_A >= 1
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.POS] = -S * 100
            self.b_up[0] = 0.0
            self.W_gate[0, E.OP_START + Opcode.BNE] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.W_up[1, E.POS] = -S * 100
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.OP_START + Opcode.BNE] = 1.0
            self.W_down[E.TEMP, 1] = -1.0 / S


class BranchCopyTargetFFN(PureFFN):
    """
    Copy NIB_B to RESULT if RAW_SUM == 1 (branch taken).
    RAW_SUM contains the broadcast condition from BranchConditionAttention.

    Uses integer-aligned thresholds: step(x) - step(x-1) = 1 when x ∈ [1, 2)
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: step(RAW_SUM) - activates when RAW_SUM > 0
            self.W_up[0, E.RAW_SUM] = S
            self.b_up[0] = 0.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Row 1: -step(RAW_SUM - 1) - saturates when RAW_SUM >= 1
            self.W_up[1, E.RAW_SUM] = S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class BranchClearTempFFN(PureFFN):
    """Clear TEMP after branch operation."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_gate[0, E.TEMP] = -1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + self.opcode] = -S
            self.W_gate[1, E.TEMP] = 1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


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
            self.W_up[0, E.OP_START + Opcode.JMP] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.JMP] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class BranchEqFFN(PureFFN):
    """
    BEQ (Branch if Zero): Branch if A == 0.
    A = condition (typically result of prior comparison, 0 or 1)
    B = target address
    Result = B if A == 0, else 0.

    Direct computation: silu(1-A)*B - silu(-A)*B gives B when A=0, 0 otherwise.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: silu(S*(1-A)) * B → B when A=0
            self.W_up[0, E.NIB_A] = -S
            self.b_up[0] = S * 1.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Row 1: -silu(-S*A) * B → subtract when A=0 for saturation
            self.W_up[1, E.NIB_A] = -S
            self.b_up[1] = 0.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / S


class BranchNeFFN(PureFFN):
    """
    BNE (Branch if Non-Zero): Branch if A != 0.
    A = condition (typically result of prior comparison, 0 or 1)
    B = target address
    Result = B if A != 0, else 0.

    Direct computation: silu(A)*B - silu(A-1)*B gives B when A>=1, 0 when A=0.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: silu(S*A) * B → B*A when A>0
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = 0.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Row 1: -silu(S*(A-1)) * B → saturate for A>=1
            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / S


class BranchLtFFN(PureFFN):
    """
    BLT: Branch if A < B (single nibble comparison).
    A = first operand
    B = second operand AND target address
    Result = B if A < B, else 0.

    Direct: silu(B-A)*B - silu(B-A-1)*B gives B when B>A, 0 otherwise.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: silu(S*(B-A)) * B → B*(B-A) when B>A
            self.W_up[0, E.NIB_B] = S
            self.W_up[0, E.NIB_A] = -S
            self.b_up[0] = 0.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Row 1: -silu(S*(B-A-1)) * B → saturate
            self.W_up[1, E.NIB_B] = S
            self.W_up[1, E.NIB_A] = -S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / S


class BranchGeFFN(PureFFN):
    """
    BGE: Branch if A >= B (single nibble comparison).
    A = first operand
    B = second operand AND target address
    Result = B if A >= B, else 0.

    Direct: silu(A-B+1)*B - silu(A-B)*B gives B when A>=B, 0 otherwise.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Row 0: silu(S*(A-B+1)) * B → B when A>=B
            self.W_up[0, E.NIB_A] = S
            self.W_up[0, E.NIB_B] = -S
            self.b_up[0] = S * 1.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Row 1: -silu(S*(A-B)) * B → saturate
            self.W_up[1, E.NIB_A] = S
            self.W_up[1, E.NIB_B] = -S
            self.b_up[1] = 0.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / S


class CallFFN(PureFFN):
    """
    CALL: Function call.
    A = current PC (to be pushed as return address)
    B = target address

    Output:
    - RESULT = target address (new PC)
    - TEMP = return address (PC + 1, to be pushed to stack)

    Stack management (push TEMP) happens externally via KV cache.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy target address (NIB_B) to RESULT
            self.W_up[0, E.OP_START + Opcode.CALL] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.CALL] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # Copy return address (NIB_A) to TEMP
            # Note: Caller should pass PC+1 as NIB_A
            self.W_up[2, E.OP_START + Opcode.CALL] = S
            self.W_gate[2, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.CALL] = -S
            self.W_gate[3, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class RetFFN(PureFFN):
    """
    RET: Return from function call.

    Input: NIB_A = return address (popped from stack externally)
    Output: RESULT = return address (new PC)

    Stack management (pop to NIB_A) happens externally via KV cache.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy return address (NIB_A) to RESULT
            self.W_up[0, E.OP_START + Opcode.RET] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.RET] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S
