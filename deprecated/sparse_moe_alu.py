"""
Unified Sparse MoE ALU for Neural VM V7.

Architecture:
- Each stage is an MoE layer that routes to opcode-specific experts
- Attention layers are shared where possible (carry propagation)
- Operations are organized as: Initial -> [Attention + Iteration]* -> Finalize

This provides true sparse computation - only the relevant experts run.
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention
from .moe_layers import MultiExpertMoELayer

# Import all operation layers
from .arithmetic_ops import (
    AddRawSumFFN, InitResultFFN, CarryDetectFFN, CarryPropagateAttention,
    ZeroFirstCarryFFN, ClearCarryOutFFN, CarryIterFFN, ClearCarryInFFN,
    SubRawDiffFFN, SubInitResultFFN, BorrowDetectFFN,
    ZeroFirstBorrowFFN, ClearBorrowOutFFN, BorrowIterFFN, ClearBorrowInFFN
)
from .shift_ops import (
    ClearTempBeforeShiftFFN, ShiftLeftCopyFFN, ShiftLeftAttention,
    ShiftLeftResultFFN, ShiftLeftClearFFN, ShiftRightCopyFFN,
    ShiftRightAttention, ShiftRightResultFFN, ShiftRightClearFFN,
    ClearCarryInFFNShift, ZeroShiftPos0FFN, ZeroShiftPos7FFN, CopyCarryToTempFFN
)
from .mul_div_ops import (
    MulProductFFN, MulGateFFN, MulOverflowFFN, MulZeroFirstCarryFFN,
    MulClearCarryOutFFN, MulCarryIterFFN, MulClearCarryInFFN, MulClearTempFFN,
    DivInitFFN, DivIterFFN, ModInitFFN, ModIterFFN, ModResultFFN
)
from .bitwise_ops import (
    ClearBitSlotsFFN, ExtractBit3FFN, ExtractBit2FFN,
    ExtractBit1FFN, ExtractBit0FFN, BitwiseAndCombineFFN,
    BitwiseOrCombineFFN, BitwiseXorCombineFFN, ClearBitsFFN
)
from .comparison_ops import (
    CompareDiffFFN, ClearRawSumFFN, CompareEqNibbleFFN, CompareNeNibbleFFN,
    CompareLtNibbleFFN, CompareGtNibbleFFN, CompareReduceEqAttention,
    CompareReduceEqFFN, CompareReduceNeAttention, CompareReduceNeFFN,
    CompareCopyResultFFN, CmpRawDiffFFN, CmpRawDiffSwapFFN,
    CmpBorrowDetectFFN, CmpZeroFirstBorrowFFN, CmpClearBorrowOutFFN,
    CmpBorrowIterFFN, CmpClearBorrowInFFN, CmpClearTempFFN,
    CmpExtractMSBBorrowFFN, CmpClearResultFFN, CmpBroadcastResultAttention,
    CmpInvertResultFFN
)
from .control_flow_ops import (
    JumpFFN, BranchEqFFN, BranchNeFFN, BranchLtFFN, BranchGeFFN,
    BranchConditionAttention, BranchCondEqFFN, BranchCondNeFFN,
    BranchCopyTargetFFN, BranchClearTempFFN, CallFFN, RetFFN
)
from .memory_ops import (
    LoadFFN, StoreFFN, PushFFN, PopFFN, NopFFN, HaltFFN
)
from .io_ops import (
    GetcharSetNeedInputFFN, GetcharReadInputFFN, PutcharWriteOutputFFN,
    ExitSetEndFFN, PrintfFFN, FileOpenFFN, FileReadFFN, FileCloseFFN
)


class SparseMoEALU(nn.Module):
    """
    Unified Sparse MoE-based ALU.

    Architecture:
    - Stage 1: Initial computation (MoE routes to opcode-specific FFNs)
    - Stage 2: Attention for position communication (carry propagation)
    - Stage 3: Post-attention FFNs (MoE)
    - Repeat stages 2-3 for cascade operations (7 iterations for 8 nibbles)
    - Stage 4: Finalization (MoE)

    Only the relevant experts for the active opcode are executed.
    """

    def __init__(self):
        super().__init__()

        # Stage 1: Initial computation experts per opcode
        self.initial_experts = MultiExpertMoELayer({
            Opcode.ADD: [AddRawSumFFN(), InitResultFFN(), CarryDetectFFN()],
            Opcode.SUB: [SubRawDiffFFN(), SubInitResultFFN(), BorrowDetectFFN()],
            Opcode.MUL: [MulProductFFN(), MulGateFFN(), MulOverflowFFN()],
            Opcode.DIV: [DivInitFFN()],
            Opcode.MOD: [ModInitFFN()],
            Opcode.AND: [ClearBitSlotsFFN(Opcode.AND),
                        ExtractBit3FFN(Opcode.AND), ExtractBit2FFN(Opcode.AND),
                        ExtractBit1FFN(Opcode.AND), ExtractBit0FFN(Opcode.AND),
                        BitwiseAndCombineFFN()],
            Opcode.OR: [ClearBitSlotsFFN(Opcode.OR),
                       ExtractBit3FFN(Opcode.OR), ExtractBit2FFN(Opcode.OR),
                       ExtractBit1FFN(Opcode.OR), ExtractBit0FFN(Opcode.OR),
                       BitwiseOrCombineFFN()],
            Opcode.XOR: [ClearBitSlotsFFN(Opcode.XOR),
                        ExtractBit3FFN(Opcode.XOR), ExtractBit2FFN(Opcode.XOR),
                        ExtractBit1FFN(Opcode.XOR), ExtractBit0FFN(Opcode.XOR),
                        BitwiseXorCombineFFN()],
            Opcode.EQ: [CompareDiffFFN(Opcode.EQ), CompareEqNibbleFFN(Opcode.EQ), ClearRawSumFFN(Opcode.EQ)],
            Opcode.NE: [CompareDiffFFN(Opcode.NE), CompareNeNibbleFFN(Opcode.NE), ClearRawSumFFN(Opcode.NE)],
            Opcode.LT: [CmpRawDiffFFN(Opcode.LT), CmpBorrowDetectFFN(Opcode.LT)],
            Opcode.GT: [CmpRawDiffSwapFFN(Opcode.GT), CmpBorrowDetectFFN(Opcode.GT)],
            Opcode.LE: [CmpRawDiffSwapFFN(Opcode.LE), CmpBorrowDetectFFN(Opcode.LE)],
            Opcode.GE: [CmpRawDiffFFN(Opcode.GE), CmpBorrowDetectFFN(Opcode.GE)],
            Opcode.SHL: [ClearTempBeforeShiftFFN(), ShiftLeftCopyFFN()],
            Opcode.SHR: [ClearTempBeforeShiftFFN(), ShiftRightCopyFFN()],
            Opcode.JMP: [JumpFFN()],
            Opcode.BEQ: [BranchCondEqFFN()],  # Compute condition at pos 0
            Opcode.BNE: [BranchCondNeFFN()],  # Compute condition at pos 0
            Opcode.BLT: [BranchLtFFN()],
            Opcode.BGE: [BranchGeFFN()],
            Opcode.CALL: [CallFFN()],
            Opcode.RET: [RetFFN()],
            Opcode.LOAD: [LoadFFN()],
            Opcode.STORE: [StoreFFN()],
            Opcode.PUSH: [PushFFN()],
            Opcode.POP: [PopFFN()],
            Opcode.NOP: [NopFFN()],
            Opcode.HALT: [HaltFFN()],
            # I/O Operations
            Opcode.GETCHAR: [GetcharSetNeedInputFFN()],
            Opcode.PUTCHAR: [PutcharWriteOutputFFN()],
            Opcode.EXIT: [ExitSetEndFFN()],
            Opcode.PRTF: [PrintfFFN()],
            Opcode.OPEN: [FileOpenFFN()],
            Opcode.READ: [FileReadFFN()],
            Opcode.CLOS: [FileCloseFFN()],
        })

        # Shared attention layer for carry/borrow propagation
        self.carry_attn = CarryPropagateAttention()

        # Stage 2-3: Carry/borrow iteration experts (per-opcode)
        self.carry_experts = MultiExpertMoELayer({
            Opcode.ADD: [ZeroFirstCarryFFN(), ClearCarryOutFFN(), CarryIterFFN(), ClearCarryInFFN()],
            Opcode.SUB: [ZeroFirstBorrowFFN(), ClearBorrowOutFFN(), BorrowIterFFN(), ClearBorrowInFFN()],
            Opcode.MUL: [MulZeroFirstCarryFFN(), MulClearCarryOutFFN(), MulCarryIterFFN(), MulClearCarryInFFN()],
            Opcode.LT: [CmpZeroFirstBorrowFFN(Opcode.LT), CmpClearBorrowOutFFN(Opcode.LT),
                       CmpBorrowIterFFN(Opcode.LT), CmpClearBorrowInFFN(Opcode.LT)],
            Opcode.GT: [CmpZeroFirstBorrowFFN(Opcode.GT), CmpClearBorrowOutFFN(Opcode.GT),
                       CmpBorrowIterFFN(Opcode.GT), CmpClearBorrowInFFN(Opcode.GT)],
            Opcode.LE: [CmpZeroFirstBorrowFFN(Opcode.LE), CmpClearBorrowOutFFN(Opcode.LE),
                       CmpBorrowIterFFN(Opcode.LE), CmpClearBorrowInFFN(Opcode.LE)],
            Opcode.GE: [CmpZeroFirstBorrowFFN(Opcode.GE), CmpClearBorrowOutFFN(Opcode.GE),
                       CmpBorrowIterFFN(Opcode.GE), CmpClearBorrowInFFN(Opcode.GE)],
        })

        # Shift attention layers (opcode-specific)
        self.shl_attn = ShiftLeftAttention()
        self.shr_attn = ShiftRightAttention()

        # Shift iteration helpers
        self.clear_carry_for_shift = ClearCarryInFFNShift()
        self.zero_shift_pos0 = ZeroShiftPos0FFN()
        self.zero_shift_pos7 = ZeroShiftPos7FFN()
        self.copy_carry_to_temp = CopyCarryToTempFFN()

        # EQ/NE reduction attention
        self.eq_reduce_attn = CompareReduceEqAttention()
        self.ne_reduce_attn = CompareReduceNeAttention()

        # LT/GT/LE/GE broadcast attention
        self.cmp_broadcast_attn = CmpBroadcastResultAttention()

        # Branch condition broadcast attention
        self.branch_cond_attn = BranchConditionAttention()

        # DIV/MOD iteration layers (fixed iterations)
        self.div_iters = nn.ModuleList([DivIterFFN() for _ in range(16)])
        self.mod_iters = nn.ModuleList([ModIterFFN() for _ in range(16)])

        # Stage 4: Finalization experts
        self.final_experts = MultiExpertMoELayer({
            Opcode.AND: [ClearBitsFFN(Opcode.AND)],
            Opcode.OR: [ClearBitsFFN(Opcode.OR)],
            Opcode.XOR: [ClearBitsFFN(Opcode.XOR)],
            Opcode.MUL: [MulClearTempFFN()],
            Opcode.MOD: [ModResultFFN()],
            Opcode.EQ: [CompareReduceEqFFN()],
            Opcode.NE: [CompareReduceNeFFN()],
            Opcode.BEQ: [BranchCopyTargetFFN(Opcode.BEQ), BranchClearTempFFN(Opcode.BEQ)],
            Opcode.BNE: [BranchCopyTargetFFN(Opcode.BNE), BranchClearTempFFN(Opcode.BNE)],
            Opcode.LT: [CmpClearTempFFN(Opcode.LT), CmpExtractMSBBorrowFFN(Opcode.LT),
                       CmpClearResultFFN(Opcode.LT)],
            Opcode.GT: [CmpClearTempFFN(Opcode.GT), CmpExtractMSBBorrowFFN(Opcode.GT),
                       CmpClearResultFFN(Opcode.GT)],
            Opcode.LE: [CmpClearTempFFN(Opcode.LE), CmpExtractMSBBorrowFFN(Opcode.LE),
                       CmpClearResultFFN(Opcode.LE)],
            Opcode.GE: [CmpClearTempFFN(Opcode.GE), CmpExtractMSBBorrowFFN(Opcode.GE),
                       CmpClearResultFFN(Opcode.GE)],
            Opcode.SHL: [ShiftLeftResultFFN(), ShiftLeftClearFFN()],
            Opcode.SHR: [ShiftRightResultFFN(), ShiftRightClearFFN()],
        })

        # Post-finalization (inversion for LE/GE)
        self.post_final_experts = MultiExpertMoELayer({
            Opcode.LE: [CmpInvertResultFFN(Opcode.LE)],
            Opcode.GE: [CmpInvertResultFFN(Opcode.GE)],
        })

        # Operations that need carry cascade (7 iterations)
        self.carry_ops = {Opcode.ADD, Opcode.SUB, Opcode.MUL,
                          Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE}

    def bit_shift_left(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Shift TEMP left by 1-3 bits within nibbles, propagating carries.

        For each nibble:
            new_val = (old_val * 2^bits) mod 16
            carry = (old_val * 2^bits) // 16
        The carry is added to the next nibble.
        """
        multiplier = 1 << bits
        with torch.no_grad():
            # Process nibbles from LSB to MSB
            carry = 0.0
            for i in range(E.NUM_POSITIONS):
                old_val = x[0, i, E.TEMP].item()
                new_val = old_val * multiplier + carry
                x[0, i, E.TEMP] = float(int(new_val) % 16)
                carry = float(int(new_val) // 16)
        return x

    def copy_temp_to_carry_in(self, x: torch.Tensor) -> torch.Tensor:
        """Copy TEMP to CARRY_IN for finalization step."""
        with torch.no_grad():
            for i in range(E.NUM_POSITIONS):
                x[0, i, E.CARRY_IN] = x[0, i, E.TEMP].item()
        return x

    def bit_shift_right(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Shift CARRY_IN right by 1-3 bits within nibbles, borrowing from higher nibbles.

        For each nibble (processing MSB to LSB):
            borrow_out = old_val mod 2^bits
            new_val = old_val // 2^bits + (borrow_in << (4 - bits))
        """
        divisor = 1 << bits
        shift_in = 4 - bits
        with torch.no_grad():
            # Process nibbles from MSB to LSB
            borrow = 0.0
            for i in range(E.NUM_POSITIONS - 1, -1, -1):
                old_val = int(x[0, i, E.CARRY_IN].item())
                borrow_out = old_val % divisor
                new_val = old_val // divisor + int(borrow * (1 << shift_in))
                x[0, i, E.CARRY_IN] = float(new_val)
                borrow = float(borrow_out)
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse MoE forward pass.

        Only runs the experts relevant to the input opcode.
        """
        # Extract active opcode
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        # Flag to track if we've handled the result directly (skip finalization)
        skip_finalization = False

        # Stage 1: Initial computation (MoE)
        x = self.initial_experts(x)

        # Stage 2-3: Carry/borrow cascade (7 iterations)
        if active_opcode in self.carry_ops:
            for _ in range(7):
                x = self.carry_attn(x)
                x = self.carry_experts(x)

        # DIV iterations (16 for nibble max quotient)
        if active_opcode == Opcode.DIV:
            for layer in self.div_iters:
                x = layer(x)

        # MOD iterations
        if active_opcode == Opcode.MOD:
            for layer in self.mod_iters:
                x = layer(x)

        # Shift attention - run multiple iterations based on shift amount
        # NIB_B contains shift amount (encoded in nibbles), decode it first
        if active_opcode == Opcode.SHL:
            # Decode full shift amount from all NIB_B nibbles
            shift_amount = 0
            for i in range(E.NUM_POSITIONS):
                nib = int(round(x[0, i, E.NIB_B].item()))
                shift_amount |= (nib << (i * 4))
            nibble_shifts = shift_amount // 4
            bit_shifts = shift_amount % 4

            # Handle bit-level shifts (within nibble) first
            if bit_shifts > 0:
                x = self.bit_shift_left(x, bit_shifts)

            # Handle nibble-level shifts (between positions)
            if nibble_shifts > 0:
                for i in range(nibble_shifts):
                    x = self.clear_carry_for_shift(x)  # Clear before attention
                    x = self.shl_attn(x)
                    x = self.zero_shift_pos0(x)  # Zero out position 0
                    if i < nibble_shifts - 1:
                        x = self.copy_carry_to_temp(x)  # Prepare for next iteration
            else:
                # No nibble shifts - copy TEMP directly to RESULT (bypass finalization)
                # This avoids the ZeroShiftPos0 logic in ShiftLeftResultFFN
                with torch.no_grad():
                    for i in range(E.NUM_POSITIONS):
                        x[0, i, E.RESULT] = x[0, i, E.TEMP].item()
                skip_finalization = True

        elif active_opcode == Opcode.SHR:
            # Decode full shift amount from all NIB_B nibbles
            shift_amount = 0
            for i in range(E.NUM_POSITIONS):
                nib = int(round(x[0, i, E.NIB_B].item()))
                shift_amount |= (nib << (i * 4))
            nibble_shifts = shift_amount // 4
            bit_shifts = shift_amount % 4

            # Handle nibble-level shifts (between positions) first
            if nibble_shifts > 0:
                for i in range(nibble_shifts):
                    x = self.clear_carry_for_shift(x)  # Clear before attention
                    x = self.shr_attn(x)
                    x = self.zero_shift_pos7(x)  # Zero out position 7
                    if i < nibble_shifts - 1:
                        x = self.copy_carry_to_temp(x)  # Prepare for next iteration
            else:
                # No nibble shifts - copy TEMP directly to CARRY_IN for finalization
                x = self.copy_temp_to_carry_in(x)

            # Handle bit-level shifts (within nibble) last
            if bit_shifts > 0:
                x = self.bit_shift_right(x, bit_shifts)

        # EQ/NE reduction attention
        if active_opcode == Opcode.EQ:
            x = self.eq_reduce_attn(x)
        elif active_opcode == Opcode.NE:
            x = self.ne_reduce_attn(x)

        # Branch condition broadcast (BEQ/BNE)
        if active_opcode in {Opcode.BEQ, Opcode.BNE}:
            x = self.branch_cond_attn(x)

        # Stage 4: Finalization (MoE) - skip if we've already handled the result
        if not skip_finalization:
            x = self.final_experts(x)

        # LT/GT/LE/GE broadcast attention (after extracting MSB borrow)
        if active_opcode in {Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE}:
            x = self.cmp_broadcast_attn(x)

        # Post-finalization (inversion for LE/GE)
        x = self.post_final_experts(x)

        return x


class UnifiedMoETransformer(nn.Module):
    """
    Unified MoE Transformer architecture.

    This organizes the ALU as a sequence of MoE transformer blocks:
    1. Input embedding (already encoded by caller)
    2. N x (Attention + MoE FFN) blocks
    3. Output projection (extract RESULT)

    Each MoE FFN layer routes to the appropriate expert based on opcode.
    """

    def __init__(self, num_carry_iters: int = 7, num_div_iters: int = 16):
        super().__init__()
        self.num_carry_iters = num_carry_iters
        self.num_div_iters = num_div_iters

        # Layer 0: Initial computation (no attention needed)
        self.layer0_moe = MultiExpertMoELayer({
            Opcode.ADD: [AddRawSumFFN(), InitResultFFN(), CarryDetectFFN()],
            Opcode.SUB: [SubRawDiffFFN(), SubInitResultFFN(), BorrowDetectFFN()],
            Opcode.MUL: [MulProductFFN(), MulGateFFN(), MulOverflowFFN()],
            Opcode.DIV: [DivInitFFN()],
            Opcode.MOD: [ModInitFFN()],
            Opcode.AND: [ClearBitSlotsFFN(Opcode.AND),
                        ExtractBit3FFN(Opcode.AND), ExtractBit2FFN(Opcode.AND),
                        ExtractBit1FFN(Opcode.AND), ExtractBit0FFN(Opcode.AND),
                        BitwiseAndCombineFFN()],
            Opcode.OR: [ClearBitSlotsFFN(Opcode.OR),
                       ExtractBit3FFN(Opcode.OR), ExtractBit2FFN(Opcode.OR),
                       ExtractBit1FFN(Opcode.OR), ExtractBit0FFN(Opcode.OR),
                       BitwiseOrCombineFFN()],
            Opcode.XOR: [ClearBitSlotsFFN(Opcode.XOR),
                        ExtractBit3FFN(Opcode.XOR), ExtractBit2FFN(Opcode.XOR),
                        ExtractBit1FFN(Opcode.XOR), ExtractBit0FFN(Opcode.XOR),
                        BitwiseXorCombineFFN()],
            Opcode.EQ: [CompareDiffFFN(Opcode.EQ), CompareEqNibbleFFN(Opcode.EQ), ClearRawSumFFN(Opcode.EQ)],
            Opcode.NE: [CompareDiffFFN(Opcode.NE), CompareNeNibbleFFN(Opcode.NE), ClearRawSumFFN(Opcode.NE)],
            Opcode.LT: [CmpRawDiffFFN(Opcode.LT), CmpBorrowDetectFFN(Opcode.LT)],
            Opcode.GT: [CmpRawDiffSwapFFN(Opcode.GT), CmpBorrowDetectFFN(Opcode.GT)],
            Opcode.LE: [CmpRawDiffSwapFFN(Opcode.LE), CmpBorrowDetectFFN(Opcode.LE)],
            Opcode.GE: [CmpRawDiffFFN(Opcode.GE), CmpBorrowDetectFFN(Opcode.GE)],
            Opcode.SHL: [ClearTempBeforeShiftFFN(), ShiftLeftCopyFFN()],
            Opcode.SHR: [ClearTempBeforeShiftFFN(), ShiftRightCopyFFN()],
            Opcode.JMP: [JumpFFN()],
            Opcode.BEQ: [BranchCondEqFFN()],  # Compute condition at pos 0
            Opcode.BNE: [BranchCondNeFFN()],  # Compute condition at pos 0
            Opcode.BLT: [BranchLtFFN()],
            Opcode.BGE: [BranchGeFFN()],
            Opcode.CALL: [CallFFN()],
            Opcode.RET: [RetFFN()],
            Opcode.LOAD: [LoadFFN()],
            Opcode.STORE: [StoreFFN()],
            Opcode.PUSH: [PushFFN()],
            Opcode.POP: [PopFFN()],
            Opcode.NOP: [NopFFN()],
            Opcode.HALT: [HaltFFN()],
        })

        # Layers 1-7: Carry propagation (Attention + MoE)
        self.carry_attn = CarryPropagateAttention()
        self.carry_moe = MultiExpertMoELayer({
            Opcode.ADD: [ZeroFirstCarryFFN(), ClearCarryOutFFN(), CarryIterFFN(), ClearCarryInFFN()],
            Opcode.SUB: [ZeroFirstBorrowFFN(), ClearBorrowOutFFN(), BorrowIterFFN(), ClearBorrowInFFN()],
            Opcode.MUL: [MulZeroFirstCarryFFN(), MulClearCarryOutFFN(), MulCarryIterFFN(), MulClearCarryInFFN()],
            Opcode.LT: [CmpZeroFirstBorrowFFN(Opcode.LT), CmpClearBorrowOutFFN(Opcode.LT),
                       CmpBorrowIterFFN(Opcode.LT), CmpClearBorrowInFFN(Opcode.LT)],
            Opcode.GT: [CmpZeroFirstBorrowFFN(Opcode.GT), CmpClearBorrowOutFFN(Opcode.GT),
                       CmpBorrowIterFFN(Opcode.GT), CmpClearBorrowInFFN(Opcode.GT)],
            Opcode.LE: [CmpZeroFirstBorrowFFN(Opcode.LE), CmpClearBorrowOutFFN(Opcode.LE),
                       CmpBorrowIterFFN(Opcode.LE), CmpClearBorrowInFFN(Opcode.LE)],
            Opcode.GE: [CmpZeroFirstBorrowFFN(Opcode.GE), CmpClearBorrowOutFFN(Opcode.GE),
                       CmpBorrowIterFFN(Opcode.GE), CmpClearBorrowInFFN(Opcode.GE)],
        })

        # DIV/MOD iteration layers
        self.div_iters = nn.ModuleList([DivIterFFN() for _ in range(num_div_iters)])
        self.mod_iters = nn.ModuleList([ModIterFFN() for _ in range(num_div_iters)])

        # Shift attention layers
        self.shl_attn = ShiftLeftAttention()
        self.shr_attn = ShiftRightAttention()

        # EQ/NE reduction attention
        self.eq_reduce_attn = CompareReduceEqAttention()
        self.ne_reduce_attn = CompareReduceNeAttention()

        # Comparison broadcast attention
        self.cmp_broadcast_attn = CmpBroadcastResultAttention()

        # Final layer MoE
        self.final_moe = MultiExpertMoELayer({
            Opcode.AND: [ClearBitsFFN(Opcode.AND)],
            Opcode.OR: [ClearBitsFFN(Opcode.OR)],
            Opcode.XOR: [ClearBitsFFN(Opcode.XOR)],
            Opcode.MUL: [MulClearTempFFN()],
            Opcode.MOD: [ModResultFFN()],
            Opcode.EQ: [CompareReduceEqFFN()],
            Opcode.NE: [CompareReduceNeFFN()],
            Opcode.LT: [CmpClearTempFFN(Opcode.LT), CmpExtractMSBBorrowFFN(Opcode.LT),
                       CmpClearResultFFN(Opcode.LT)],
            Opcode.GT: [CmpClearTempFFN(Opcode.GT), CmpExtractMSBBorrowFFN(Opcode.GT),
                       CmpClearResultFFN(Opcode.GT)],
            Opcode.LE: [CmpClearTempFFN(Opcode.LE), CmpExtractMSBBorrowFFN(Opcode.LE),
                       CmpClearResultFFN(Opcode.LE)],
            Opcode.GE: [CmpClearTempFFN(Opcode.GE), CmpExtractMSBBorrowFFN(Opcode.GE),
                       CmpClearResultFFN(Opcode.GE)],
            Opcode.SHL: [ShiftLeftResultFFN(), ShiftLeftClearFFN()],
            Opcode.SHR: [ShiftRightResultFFN(), ShiftRightClearFFN()],
        })

        # Post-final MoE (inversion)
        self.post_final_moe = MultiExpertMoELayer({
            Opcode.LE: [CmpInvertResultFFN(Opcode.LE)],
            Opcode.GE: [CmpInvertResultFFN(Opcode.GE)],
        })

        self.carry_ops = {Opcode.ADD, Opcode.SUB, Opcode.MUL,
                          Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE}

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through unified MoE transformer."""
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        # Layer 0: Initial
        x = self.layer0_moe(x)

        # Layers 1-7: Carry cascade
        if active_opcode in self.carry_ops:
            for _ in range(self.num_carry_iters):
                x = self.carry_attn(x)
                x = self.carry_moe(x)

        # DIV iterations
        if active_opcode == Opcode.DIV:
            for layer in self.div_iters:
                x = layer(x)

        # MOD iterations
        if active_opcode == Opcode.MOD:
            for layer in self.mod_iters:
                x = layer(x)

        # Shift attention
        if active_opcode == Opcode.SHL:
            x = self.shl_attn(x)
        elif active_opcode == Opcode.SHR:
            x = self.shr_attn(x)

        # EQ/NE reduction
        if active_opcode == Opcode.EQ:
            x = self.eq_reduce_attn(x)
        elif active_opcode == Opcode.NE:
            x = self.ne_reduce_attn(x)

        # Final layer
        x = self.final_moe(x)

        # Comparison broadcast
        if active_opcode in {Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE}:
            x = self.cmp_broadcast_attn(x)

        # Post-final (inversion)
        x = self.post_final_moe(x)

        return x
