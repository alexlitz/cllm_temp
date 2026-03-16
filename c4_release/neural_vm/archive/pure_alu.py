"""
Pure ALU - Parallel Pipeline with Zipper Merge.

All opcode pipelines run in parallel. At each depth, the zipper merge
combines all opcodes' d-th pipeline stage into a single MoE layer.
Total depth = max(pipeline lengths) instead of sum.

Key insight: MoE guarantees that when opcode X is active, only opcode X's
experts fire. Other opcodes' experts produce identity (via weight=0).
Therefore slot conflicts between different opcodes are impossible.

IMPORTANT: PureALU is implemented as nn.Sequential - it has NO forward() method.
Only PureFFN.forward() and PureAttention.forward() are used.

Optimized pipelines use carry-lookahead (FlattenedPureFFN) for ADD/SUB/CMP,
parallel schoolbook for MUL, and 2-layer shifts for SHL/SHR.
Critical path: MOD at 34 layers (26 long division + 8 remainder extraction).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention
from .pure_moe import MoE, IdentityFFN, IdentityAttention

# Fast arithmetic imports (carry-lookahead ADD/SUB/CMP)
from .fast_arithmetic import (
    AddRawAndGenFFN, AddCarryLookaheadFFN, AddFinalResultFFN,
    SubRawAndGenFFN, SubBorrowLookaheadFFN, SubFinalResultFFN,
    CmpRawDiffAndGenFFN, CmpBorrowLookaheadFFN, CmpClearRawSumFFN,
)

# Fast multiplication imports
from .fast_mul import (
    SchoolbookFlatFFN, MulCarryPass1FFN, MulCarryPass2FFN,
    MulCarryPass3FFN, MulGenPropFFN, MulBinaryLookaheadFFN,
    MulFinalCorrectionFFN,
)

# Fast shift imports
from .fast_shift import (
    ShiftPrecomputeFFN, ShiftSelectFFN,
    ShiftRPrecomputeFFN, ShiftRPrecompute2FFN, ShiftRSelectFFN,
)

# Keep old imports for unchanged pipelines
from .mul_div_ops import (
    ModInitFFN, ModIterFFN, ModResultFFN,
)
from .long_division_ops import build_long_division_layers
from .softmax1_division_ops import build_softmax1_division_layers
from .bitwise_ops import (
    ClearBitSlotsFFN, ExtractBit3FFN, ExtractBit2FFN,
    ExtractBit1FFN, ExtractBit0FFN, BitwiseAndCombineFFN,
    BitwiseOrCombineFFN, BitwiseXorCombineFFN, ClearBitsFFN
)
from .comparison_ops import (
    CompareDiffFFN, ClearRawSumFFN, CompareEqNibbleFFN, CompareNeNibbleFFN,
    CompareReduceEqFFN, CompareReduceNeFFN,
    CmpRawDiffFFN, CmpRawDiffSwapFFN,
    CmpBorrowDetectFFN, CmpZeroFirstBorrowFFN, CmpClearBorrowOutFFN,
    CmpBorrowIterFFN, CmpClearBorrowInFFN, CmpClearTempFFN,
    CmpExtractMSBBorrowFFN, CmpClearResultFFN,
    CmpInvertResultFFN
)
from .control_flow_ops import (
    JumpFFN, BranchEqFFN, BranchNeFFN, BranchLtFFN, BranchGeFFN,
    BranchCondEqFFN, BranchCondNeFFN,
    BranchCopyTargetFFN, BranchClearTempFFN, CallFFN, RetFFN
)
from .reduce_ffn import (
    CompareReduceEqFFNNew, CompareReduceNeFFNNew, CmpBroadcastResultFFN,
    BranchConditionFFN, BzReduceFFN, BnzReduceFFN, McmpReduceFFN
)
from .memory_ops import (
    LoadFFN, StoreFFN, PushFFN, PopFFN, NopFFN, HaltFFN
)
from .io_ops import (
    GetcharSetNeedInputFFN, PutcharWriteOutputFFN,
    ExitSetEndFFN
)
from .missing_ops import (
    LeaFFN, ImmFFN,
    BzFFN, BzBranchFFN,
    BnzFFN, BnzBranchFFN,
    EntFFN, AdjFFN,
    LcFFN, ScFFN,
    MalcFFN, FreeFFN,
    MsetFFN, McmpFFN,
    OpenFFN, ReadFFN, ClosFFN, PrtfFFN
)


# =================================================================
# Per-opcode pipeline builders (optimized with carry-lookahead)
# =================================================================

def _build_add_pipeline(num_carry_iters: int = 7) -> List[MoE]:
    """ADD pipeline: raw+gen, carry-lookahead, final result. (3 layers)"""
    return [
        MoE([AddRawAndGenFFN()], [Opcode.ADD]),
        MoE([AddCarryLookaheadFFN()], [Opcode.ADD]),
        MoE([AddFinalResultFFN()], [Opcode.ADD]),
    ]


def _build_sub_pipeline(num_carry_iters: int = 7) -> List[MoE]:
    """SUB pipeline: raw+gen, borrow-lookahead, final result. (3 layers)"""
    return [
        MoE([SubRawAndGenFFN()], [Opcode.SUB]),
        MoE([SubBorrowLookaheadFFN()], [Opcode.SUB]),
        MoE([SubFinalResultFFN()], [Opcode.SUB]),
    ]


def _build_mul_pipeline(num_carry_iters: int = 8) -> List[MoE]:
    """MUL pipeline: schoolbook(1), carry passes(3), lookahead(2), correct(1). (7 layers)

    Layer 1: All 36 partial products in one FlattenedPureFFN
    Layer 2: Mod 16 + carry extraction (handles up to 1800)
    Layer 3: Add carry, new mod 16 + carry (up to 127)
    Layer 4: Add carry, mod 16, binary carry (up to 22)
    Layer 5: Compute G/P for binary carry, add carry, mod16
    Layer 6: Carry-lookahead on G/P
    Layer 7: Apply lookahead carry, final mod16
    """
    return [
        MoE([SchoolbookFlatFFN()], [Opcode.MUL]),
        MoE([MulCarryPass1FFN()], [Opcode.MUL]),
        MoE([MulCarryPass2FFN()], [Opcode.MUL]),
        MoE([MulCarryPass3FFN()], [Opcode.MUL]),
        MoE([MulGenPropFFN()], [Opcode.MUL]),
        MoE([MulBinaryLookaheadFFN()], [Opcode.MUL]),
        MoE([MulFinalCorrectionFFN()], [Opcode.MUL]),
    ]


def _build_div_pipeline(div_method: str = "long") -> List[MoE]:
    """DIV pipeline.

    div_method="long": nibble-wise long division (26 layers)
    div_method="softmax1": softmax1 reciprocal + multiply (20 layers)
    """
    if div_method == "softmax1":
        return build_softmax1_division_layers(opcode=Opcode.DIV)
    return build_long_division_layers(opcode=Opcode.DIV)


def _build_mod_pipeline(div_method: str = "long") -> List[MoE]:
    """MOD pipeline: long division + cascade remainder extraction.

    26 layers of division produce the scalar remainder in SLOT_REMAINDER.
    8 layers extract remainder nibbles from MSB to LSB, each clearing
    the old quotient from RESULT and subtracting the contribution.
    """
    if div_method == "softmax1":
        return build_softmax1_division_layers(opcode=Opcode.MOD)
    return build_long_division_layers(opcode=Opcode.MOD)


def _build_shift_pipeline() -> List[MoE]:
    """SHL+SHR pipeline: precompute + select. (2 layers)

    Layer 1: Precompute all 4 sub-nibble shifts (SHL and SHR have separate experts)
    Layer 2: Select and combine based on shift amount (FlattenedPureFFN)
    """
    return [
        MoE(
            [ShiftPrecomputeFFN(), ShiftRPrecomputeFFN(), ShiftRPrecompute2FFN()],
            [Opcode.SHL, Opcode.SHR, Opcode.SHR]
        ),
        MoE(
            [ShiftSelectFFN(), ShiftRSelectFFN()],
            [Opcode.SHL, Opcode.SHR]
        ),
    ]


def _build_bitwise_pipeline() -> List[MoE]:
    """AND/OR/XOR pipeline: clear, bit3-0, combine, clear_bits. (7 layers)"""
    bitwise_ops = [Opcode.AND, Opcode.OR, Opcode.XOR]
    layers = []

    # Step 1: Clear bit slots
    layers.append(MoE(
        [ClearBitSlotsFFN(op) for op in bitwise_ops], bitwise_ops
    ))
    # Step 2: Extract bit 3
    layers.append(MoE(
        [ExtractBit3FFN(op) for op in bitwise_ops], bitwise_ops
    ))
    # Step 3: Extract bit 2
    layers.append(MoE(
        [ExtractBit2FFN(op) for op in bitwise_ops], bitwise_ops
    ))
    # Step 4: Extract bit 1
    layers.append(MoE(
        [ExtractBit1FFN(op) for op in bitwise_ops], bitwise_ops
    ))
    # Step 5: Extract bit 0
    layers.append(MoE(
        [ExtractBit0FFN(op) for op in bitwise_ops], bitwise_ops
    ))
    # Step 6: Combine
    layers.append(MoE(
        [BitwiseAndCombineFFN(), BitwiseOrCombineFFN(), BitwiseXorCombineFFN()],
        bitwise_ops
    ))
    # Step 7: Clear bit slots
    layers.append(MoE(
        [ClearBitsFFN(op) for op in bitwise_ops], bitwise_ops
    ))
    return layers


def _build_eq_ne_pipeline() -> List[MoE]:
    """EQ/NE pipeline: diff, nibble_cmp, clear_raw, reduce_new, reduce_old. (5 layers)"""
    layers = []
    layers.append(MoE(
        [CompareDiffFFN(Opcode.EQ), CompareDiffFFN(Opcode.NE)],
        [Opcode.EQ, Opcode.NE]
    ))
    layers.append(MoE(
        [CompareEqNibbleFFN(Opcode.EQ), CompareNeNibbleFFN(Opcode.NE)],
        [Opcode.EQ, Opcode.NE]
    ))
    layers.append(MoE(
        [ClearRawSumFFN(Opcode.EQ), ClearRawSumFFN(Opcode.NE)],
        [Opcode.EQ, Opcode.NE]
    ))
    layers.append(MoE(
        [CompareReduceEqFFNNew(), CompareReduceNeFFNNew()],
        [Opcode.EQ, Opcode.NE]
    ))
    layers.append(MoE(
        [CompareReduceEqFFN(), CompareReduceNeFFN()],
        [Opcode.EQ, Opcode.NE]
    ))
    return layers


def _build_cmp_pipeline(num_carry_iters: int = 7) -> List[MoE]:
    """LT/GT/LE/GE pipeline: diff+gen, borrow-lookahead+result, invert. (3 layers)"""
    cmp_ops = [Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE]

    # Step 1: Raw diff + G/P generation
    # LT, GE: A-B; GT, LE: B-A (swap=True)
    layers = [MoE(
        [CmpRawDiffAndGenFFN(Opcode.LT, swap=False),
         CmpRawDiffAndGenFFN(Opcode.GT, swap=True),
         CmpRawDiffAndGenFFN(Opcode.LE, swap=True),
         CmpRawDiffAndGenFFN(Opcode.GE, swap=False)],
        cmp_ops
    )]

    # Step 2: Borrow-lookahead + write final borrow to RESULT[0]
    layers.append(MoE(
        [CmpBorrowLookaheadFFN([Opcode.LT]),
         CmpBorrowLookaheadFFN([Opcode.GT]),
         CmpBorrowLookaheadFFN([Opcode.LE]),
         CmpBorrowLookaheadFFN([Opcode.GE])],
        cmp_ops
    ))

    # Step 3: Clean up + invert for LE/GE
    layers.append(MoE(
        [CmpClearRawSumFFN(Opcode.LT), CmpClearRawSumFFN(Opcode.GT),
         CmpClearRawSumFFN(Opcode.LE), CmpClearRawSumFFN(Opcode.GE),
         CmpInvertResultFFN(Opcode.LE), CmpInvertResultFFN(Opcode.GE)],
        cmp_ops + [Opcode.LE, Opcode.GE]
    ))

    return layers


def _build_bz_bnz_pipeline() -> List[MoE]:
    """BZ/BNZ pipeline: zero_detect, reduce, branch. (3 layers)"""
    layers = []
    layers.append(MoE(
        [BzFFN(), BnzFFN()],
        [Opcode.BZ, Opcode.BNZ]
    ))
    layers.append(MoE(
        [BzReduceFFN(), BnzReduceFFN()],
        [Opcode.BZ, Opcode.BNZ]
    ))
    layers.append(MoE(
        [BzBranchFFN(), BnzBranchFFN()],
        [Opcode.BZ, Opcode.BNZ]
    ))
    return layers


def _build_beq_bne_pipeline() -> List[MoE]:
    """BEQ/BNE pipeline: condition, broadcast, copy_target+clear. (3 layers)"""
    layers = []
    layers.append(MoE(
        [BranchCondEqFFN(), BranchCondNeFFN()],
        [Opcode.BEQ, Opcode.BNE]
    ))
    layers.append(MoE(
        [BranchConditionFFN(), BranchConditionFFN()],
        [Opcode.BEQ, Opcode.BNE]
    ))
    layers.append(MoE(
        [BranchCopyTargetFFN(Opcode.BEQ), BranchClearTempFFN(Opcode.BEQ),
         BranchCopyTargetFFN(Opcode.BNE), BranchClearTempFFN(Opcode.BNE)],
        [Opcode.BEQ, Opcode.BEQ, Opcode.BNE, Opcode.BNE]
    ))
    return layers


def _build_mcmp_pipeline() -> List[MoE]:
    """MCMP pipeline: diff, reduce. (2 layers)"""
    return [
        MoE([McmpFFN()], [Opcode.MCMP]),
        MoE([McmpReduceFFN()], [Opcode.MCMP]),
    ]


def _build_single_step() -> List[MoE]:
    """All single-step opcodes in one MoE. (1 layer)"""
    experts = []
    opcodes = []

    # Control flow
    experts.append(JumpFFN()); opcodes.append(Opcode.JMP)
    experts.append(BranchLtFFN()); opcodes.append(Opcode.BLT)
    experts.append(BranchGeFFN()); opcodes.append(Opcode.BGE)
    experts.append(CallFFN()); opcodes.append(Opcode.CALL)
    experts.append(RetFFN()); opcodes.append(Opcode.RET)

    # Memory
    experts.append(LoadFFN()); opcodes.append(Opcode.LOAD)
    experts.append(StoreFFN()); opcodes.append(Opcode.STORE)
    experts.append(PushFFN()); opcodes.append(Opcode.PUSH)
    experts.append(PopFFN()); opcodes.append(Opcode.POP)
    experts.append(NopFFN()); opcodes.append(Opcode.NOP)
    experts.append(HaltFFN()); opcodes.append(Opcode.HALT)

    # I/O
    experts.append(GetcharSetNeedInputFFN()); opcodes.append(Opcode.GETCHAR)
    experts.append(PutcharWriteOutputFFN()); opcodes.append(Opcode.PUTCHAR)
    experts.append(ExitSetEndFFN()); opcodes.append(Opcode.EXIT)

    # Address/Immediate
    experts.append(LeaFFN()); opcodes.append(Opcode.LEA)
    experts.append(ImmFFN()); opcodes.append(Opcode.IMM)

    # Stack frame
    experts.append(EntFFN()); opcodes.append(Opcode.ENT)
    experts.append(AdjFFN()); opcodes.append(Opcode.ADJ)

    # Char load/store
    experts.append(LcFFN()); opcodes.append(Opcode.LC)
    experts.append(ScFFN()); opcodes.append(Opcode.SC)

    # Memory allocation
    experts.append(MalcFFN()); opcodes.append(Opcode.MALC)
    experts.append(FreeFFN()); opcodes.append(Opcode.FREE)

    # Memory ops
    experts.append(MsetFFN()); opcodes.append(Opcode.MSET)

    # File I/O
    experts.append(OpenFFN()); opcodes.append(Opcode.OPEN)
    experts.append(ReadFFN()); opcodes.append(Opcode.READ)
    experts.append(ClosFFN()); opcodes.append(Opcode.CLOS)
    experts.append(PrtfFFN()); opcodes.append(Opcode.PRTF)

    return [MoE(experts, opcodes)]


# =================================================================
# Zipper merge — parallel ALU builder
# =================================================================

def build_parallel_alu(num_carry_iters: int = 7, div_method: str = "long") -> nn.Sequential:
    """
    Build PureALU as nn.Sequential with parallel opcode pipelines.

    Each opcode group has its own pipeline (list of MoE layers).
    The zipper merge combines all pipelines: at each depth d, all
    opcodes' d-th pipeline stages are packed into a single MoE layer.

    Args:
        div_method: "long" (26-layer nibble-wise) or "softmax1" (20-layer
                    reciprocal via softmax1 attention normalization constant).

    Optimized pipeline depths:
      ADD: 3 (carry-lookahead)
      SUB: 3 (borrow-lookahead)
      MUL: 7 (parallel schoolbook + multi-pass carry)
      DIV: 26 long / 20 softmax1
      MOD: 34 long / 28 softmax1 (critical path)
      SHL/SHR: 2 (precompute + select)
      CMP: 3 (borrow-lookahead)

    Total depth = max(pipeline lengths) = 34 long / 28 softmax1.
    """
    pipelines = [
        _build_add_pipeline(num_carry_iters),
        _build_sub_pipeline(num_carry_iters),
        _build_mul_pipeline(),  # MUL uses 8 carry iters (its own default)
        _build_div_pipeline(div_method),
        _build_mod_pipeline(div_method),
        _build_shift_pipeline(),
        _build_bitwise_pipeline(),
        _build_eq_ne_pipeline(),
        _build_cmp_pipeline(num_carry_iters),
        _build_bz_bnz_pipeline(),
        _build_beq_bne_pipeline(),
        _build_mcmp_pipeline(),
        _build_single_step(),
    ]

    max_depth = max(len(p) for p in pipelines)
    merged = []
    for d in range(max_depth):
        experts = []
        expert_opcodes = []
        for p in pipelines:
            if d < len(p):
                layer = p[d]
                assert isinstance(layer, MoE), \
                    f"Pipeline layer at depth {d} is {type(layer).__name__}, not MoE"
                experts.extend(layer.experts)
                expert_opcodes.extend(layer.expert_opcode_list)
        merged.append(MoE(list(experts), expert_opcodes))

    return nn.Sequential(*merged)


# Backward compatibility aliases
build_pure_alu = build_parallel_alu
PureALU = build_parallel_alu


def count_pure_alu_layers(num_carry_iters: int = 7, div_method: str = "long") -> Dict[str, int]:
    """Count layers in the parallel PureALU."""
    div_layers = 20 if div_method == "softmax1" else 26
    mod_layers = 28 if div_method == "softmax1" else 34
    counts = {
        'add': 3,                             # carry-lookahead
        'sub': 3,                             # borrow-lookahead
        'mul': 7,                             # parallel schoolbook + multi-pass carry
        'div': div_layers,                    # long: 26, softmax1: 20
        'mod': mod_layers,                    # long: 36, softmax1: 30
        'shift': 2,                           # precompute + select
        'bitwise': 7,
        'eq_ne': 5,
        'cmp': 3,                             # borrow-lookahead
        'bz_bnz': 3,
        'beq_bne': 3,
        'mcmp': 2,
        'single_step': 1,
    }
    counts['total_parallel'] = max(counts.values())
    return counts


def verify_purity():
    """Verify PureALU has no forward() method - it's nn.Sequential."""
    alu = PureALU()

    # Check it's an nn.Sequential
    assert isinstance(alu, nn.Sequential), "PureALU should be nn.Sequential"

    # Check no custom forward
    assert not hasattr(alu.__class__, 'forward') or alu.__class__.forward is nn.Sequential.forward, \
        "PureALU should use nn.Sequential.forward"

    # Check all layers are MoE
    for i, layer in enumerate(alu):
        assert isinstance(layer, MoE), f"Layer {i} is {type(layer).__name__}, not MoE"

    print("PureALU is PURE nn.Sequential - no custom forward()")
    print(f"Number of layers: {len(alu)}")
    return True


if __name__ == "__main__":
    verify_purity()
    print("\nPureALU Layer Counts (long division):")
    counts = count_pure_alu_layers(div_method="long")
    for stage, count in counts.items():
        print(f"  {stage}: {count}")

    print("\nPureALU Layer Counts (softmax1 division):")
    counts = count_pure_alu_layers(div_method="softmax1")
    for stage, count in counts.items():
        print(f"  {stage}: {count}")
