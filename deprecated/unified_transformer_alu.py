"""
Unified Transformer ALU - Fixed Layer Architecture.

Every operation goes through the same number of layers.
Each layer is: Attention -> MoE FFN

Operations that don't need a particular layer route to identity experts.
This is a true transformer architecture with sparse MoE routing.

Layer Structure (N layers total):
  Layer 0: Initial computation (all ops need this)
  Layer 1-7: Carry/borrow propagation (ADD/SUB use, others identity)
  Layer 8-15: Shift operations (SHL/SHR use, others identity)
  Layer 16-31: Extended operations (MUL iterations, DIV iterations)
  ...

Each layer:
  - Attention: Cross-position communication (or identity if not needed)
  - MoE FFN: Operation-specific expert (or identity if not needed)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


# =============================================================================
# IDENTITY LAYERS (No-op for operations that don't need this layer)
# =============================================================================

class IdentityFFN(PureFFN):
    """Identity FFN - passes input through unchanged."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        # All weights are zero by default, so output is zero
        # We need to pass through the input unchanged
        # But SwiGLU with zero weights gives zero output
        # So we need a different approach - just override forward
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class IdentityAttention(PureAttention):
    """Identity Attention - passes input through unchanged."""

    def __init__(self):
        super().__init__(E.DIM, num_heads=1, causal=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# =============================================================================
# MoE LAYER WITH IDENTITY DEFAULT
# =============================================================================

class UnifiedMoELayer(nn.Module):
    """
    MoE layer where each opcode routes to its expert.

    If an opcode doesn't have an expert for this layer, routes to identity.
    """

    def __init__(self, experts: Dict[int, nn.Module], default_identity: bool = True):
        """
        Args:
            experts: Dict mapping opcode to expert FFN for this layer
            default_identity: If True, unknown opcodes pass through unchanged
        """
        super().__init__()
        self.experts = nn.ModuleDict({str(k): v for k, v in experts.items()})
        self.default_identity = default_identity
        self.identity = IdentityFFN() if default_identity else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get active opcode
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        key = str(active_opcode)
        if key in self.experts:
            return self.experts[key](x)
        elif self.default_identity:
            return x  # Identity - pass through unchanged
        else:
            return x


class UnifiedAttentionLayer(nn.Module):
    """
    Attention layer where each opcode routes to its attention pattern.

    If an opcode doesn't need attention at this layer, uses identity.
    """

    def __init__(self, attention_patterns: Dict[int, nn.Module]):
        """
        Args:
            attention_patterns: Dict mapping opcode to attention module
        """
        super().__init__()
        self.patterns = nn.ModuleDict({str(k): v for k, v in attention_patterns.items()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        active_opcode = torch.argmax(opcode_vec).item()

        key = str(active_opcode)
        if key in self.patterns:
            return self.patterns[key](x)
        return x  # Identity for ops that don't need attention


# =============================================================================
# TRANSFORMER BLOCK (Attention + MoE FFN)
# =============================================================================

class TransformerALUBlock(nn.Module):
    """
    Single transformer block: Attention -> MoE FFN

    Both attention and FFN route based on opcode.
    """

    def __init__(self,
                 attention_patterns: Dict[int, nn.Module],
                 ffn_experts: Dict[int, nn.Module]):
        super().__init__()
        self.attention = UnifiedAttentionLayer(attention_patterns)
        self.ffn = UnifiedMoELayer(ffn_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)
        return x


# =============================================================================
# UNIFIED TRANSFORMER ALU
# =============================================================================

class UnifiedTransformerALU(nn.Module):
    """
    Unified Transformer ALU with fixed layer count.

    Architecture:
    - N transformer blocks (Attention + MoE FFN each)
    - Every operation goes through all N blocks
    - Operations route to their specific experts or identity

    Layer allocation:
    - Block 0: Initial computation (raw sum, bit extraction, etc.)
    - Blocks 1-7: Carry/borrow propagation
    - Blocks 8-15: Extended iterations (for MUL partial products)
    - Blocks 16-31: Division iterations
    - Block 32: Finalization

    Total: 33 blocks to handle all 32-bit operations
    """

    NUM_BLOCKS = 33  # Enough for 32-bit operations

    def __init__(self):
        super().__init__()

        # Import all operation FFNs
        from .arithmetic_ops import (
            AddRawSumFFN, InitResultFFN, CarryDetectFFN, CarryPropagateAttention,
            ZeroFirstCarryFFN, ClearCarryOutFFN, CarryIterFFN, ClearCarryInFFN,
            SubRawDiffFFN, SubInitResultFFN, BorrowDetectFFN,
            ZeroFirstBorrowFFN, ClearBorrowOutFFN, BorrowIterFFN, ClearBorrowInFFN
        )
        from .bitwise_ops import (
            ClearBitSlotsFFN, ExtractBit3FFN, ExtractBit2FFN,
            ExtractBit1FFN, ExtractBit0FFN, BitwiseAndCombineFFN,
            BitwiseOrCombineFFN, BitwiseXorCombineFFN, ClearBitsFFN
        )
        from .comparison_ops import (
            CompareDiffFFN, ClearRawSumFFN, CompareEqNibbleFFN, CompareNeNibbleFFN,
            CompareReduceEqAttention, CompareReduceEqFFN,
            CompareReduceNeAttention, CompareReduceNeFFN,
            CmpRawDiffFFN, CmpRawDiffSwapFFN, CmpBorrowDetectFFN,
            CmpZeroFirstBorrowFFN, CmpClearBorrowOutFFN, CmpBorrowIterFFN,
            CmpClearBorrowInFFN, CmpClearTempFFN, CmpExtractMSBBorrowFFN,
            CmpClearResultFFN, CmpBroadcastResultAttention, CmpInvertResultFFN
        )
        from .mul_div_ops import (
            MulProductFFN, MulGateFFN, MulOverflowFFN, MulZeroFirstCarryFFN,
            MulClearCarryOutFFN, MulCarryIterFFN, MulClearCarryInFFN, MulClearTempFFN,
            DivInitFFN, DivIterFFN, ModInitFFN, ModIterFFN, ModResultFFN
        )
        from .shift_ops import (
            ClearTempBeforeShiftFFN, ShiftLeftCopyFFN, ShiftLeftAttention,
            ShiftLeftResultFFN, ShiftLeftClearFFN, ShiftRightCopyFFN,
            ShiftRightAttention, ShiftRightResultFFN, ShiftRightClearFFN
        )
        from .io_ops import (
            GetcharSetNeedInputFFN, PutcharWriteOutputFFN, ExitSetEndFFN
        )

        # Build all transformer blocks
        self.blocks = nn.ModuleList()

        # =====================================================================
        # BLOCK 0: Initial computation
        # =====================================================================
        self.blocks.append(TransformerALUBlock(
            attention_patterns={},  # No attention needed for initial
            ffn_experts={
                # Arithmetic
                Opcode.ADD: SequentialExpert([AddRawSumFFN(), InitResultFFN(), CarryDetectFFN()]),
                Opcode.SUB: SequentialExpert([SubRawDiffFFN(), SubInitResultFFN(), BorrowDetectFFN()]),
                Opcode.MUL: SequentialExpert([MulProductFFN(), MulGateFFN(), MulOverflowFFN()]),
                Opcode.DIV: DivInitFFN(),
                Opcode.MOD: ModInitFFN(),
                # Bitwise
                Opcode.AND: SequentialExpert([
                    ClearBitSlotsFFN(Opcode.AND),
                    ExtractBit3FFN(Opcode.AND), ExtractBit2FFN(Opcode.AND),
                    ExtractBit1FFN(Opcode.AND), ExtractBit0FFN(Opcode.AND),
                    BitwiseAndCombineFFN()
                ]),
                Opcode.OR: SequentialExpert([
                    ClearBitSlotsFFN(Opcode.OR),
                    ExtractBit3FFN(Opcode.OR), ExtractBit2FFN(Opcode.OR),
                    ExtractBit1FFN(Opcode.OR), ExtractBit0FFN(Opcode.OR),
                    BitwiseOrCombineFFN()
                ]),
                Opcode.XOR: SequentialExpert([
                    ClearBitSlotsFFN(Opcode.XOR),
                    ExtractBit3FFN(Opcode.XOR), ExtractBit2FFN(Opcode.XOR),
                    ExtractBit1FFN(Opcode.XOR), ExtractBit0FFN(Opcode.XOR),
                    BitwiseXorCombineFFN()
                ]),
                # Comparison
                Opcode.EQ: SequentialExpert([CompareDiffFFN(Opcode.EQ), CompareEqNibbleFFN(Opcode.EQ), ClearRawSumFFN(Opcode.EQ)]),
                Opcode.NE: SequentialExpert([CompareDiffFFN(Opcode.NE), CompareNeNibbleFFN(Opcode.NE), ClearRawSumFFN(Opcode.NE)]),
                Opcode.LT: SequentialExpert([CmpRawDiffFFN(Opcode.LT), CmpBorrowDetectFFN(Opcode.LT)]),
                Opcode.GT: SequentialExpert([CmpRawDiffSwapFFN(Opcode.GT), CmpBorrowDetectFFN(Opcode.GT)]),
                Opcode.LE: SequentialExpert([CmpRawDiffSwapFFN(Opcode.LE), CmpBorrowDetectFFN(Opcode.LE)]),
                Opcode.GE: SequentialExpert([CmpRawDiffFFN(Opcode.GE), CmpBorrowDetectFFN(Opcode.GE)]),
                # Shift
                Opcode.SHL: SequentialExpert([ClearTempBeforeShiftFFN(), ShiftLeftCopyFFN()]),
                Opcode.SHR: SequentialExpert([ClearTempBeforeShiftFFN(), ShiftRightCopyFFN()]),
                # I/O
                Opcode.GETCHAR: GetcharSetNeedInputFFN(),
                Opcode.PUTCHAR: PutcharWriteOutputFFN(),
                Opcode.EXIT: ExitSetEndFFN(),
            }
        ))

        # =====================================================================
        # BLOCKS 1-7: Carry/Borrow Propagation
        # =====================================================================
        carry_attn = CarryPropagateAttention()

        for i in range(1, 8):
            self.blocks.append(TransformerALUBlock(
                attention_patterns={
                    # Operations that need carry propagation
                    Opcode.ADD: carry_attn,
                    Opcode.SUB: carry_attn,
                    Opcode.MUL: carry_attn,
                    Opcode.LT: carry_attn,
                    Opcode.GT: carry_attn,
                    Opcode.LE: carry_attn,
                    Opcode.GE: carry_attn,
                },
                ffn_experts={
                    Opcode.ADD: SequentialExpert([
                        ZeroFirstCarryFFN(), ClearCarryOutFFN(),
                        CarryIterFFN(), ClearCarryInFFN()
                    ]),
                    Opcode.SUB: SequentialExpert([
                        ZeroFirstBorrowFFN(), ClearBorrowOutFFN(),
                        BorrowIterFFN(), ClearBorrowInFFN()
                    ]),
                    Opcode.MUL: SequentialExpert([
                        MulZeroFirstCarryFFN(), MulClearCarryOutFFN(),
                        MulCarryIterFFN(), MulClearCarryInFFN()
                    ]),
                    Opcode.LT: SequentialExpert([
                        CmpZeroFirstBorrowFFN(Opcode.LT), CmpClearBorrowOutFFN(Opcode.LT),
                        CmpBorrowIterFFN(Opcode.LT), CmpClearBorrowInFFN(Opcode.LT)
                    ]),
                    Opcode.GT: SequentialExpert([
                        CmpZeroFirstBorrowFFN(Opcode.GT), CmpClearBorrowOutFFN(Opcode.GT),
                        CmpBorrowIterFFN(Opcode.GT), CmpClearBorrowInFFN(Opcode.GT)
                    ]),
                    Opcode.LE: SequentialExpert([
                        CmpZeroFirstBorrowFFN(Opcode.LE), CmpClearBorrowOutFFN(Opcode.LE),
                        CmpBorrowIterFFN(Opcode.LE), CmpClearBorrowInFFN(Opcode.LE)
                    ]),
                    Opcode.GE: SequentialExpert([
                        CmpZeroFirstBorrowFFN(Opcode.GE), CmpClearBorrowOutFFN(Opcode.GE),
                        CmpBorrowIterFFN(Opcode.GE), CmpClearBorrowInFFN(Opcode.GE)
                    ]),
                }
            ))

        # =====================================================================
        # BLOCKS 8-15: Shift iterations and comparison reduction
        # =====================================================================
        for i in range(8, 16):
            attn_patterns = {}
            ffn_experts = {}

            # Shift operations use their specific attention
            if i == 8:
                attn_patterns[Opcode.SHL] = ShiftLeftAttention()
                attn_patterns[Opcode.SHR] = ShiftRightAttention()
                attn_patterns[Opcode.EQ] = CompareReduceEqAttention()
                attn_patterns[Opcode.NE] = CompareReduceNeAttention()

                ffn_experts[Opcode.SHL] = SequentialExpert([ShiftLeftResultFFN(), ShiftLeftClearFFN()])
                ffn_experts[Opcode.SHR] = SequentialExpert([ShiftRightResultFFN(), ShiftRightClearFFN()])
                ffn_experts[Opcode.EQ] = CompareReduceEqFFN()
                ffn_experts[Opcode.NE] = CompareReduceNeFFN()

            self.blocks.append(TransformerALUBlock(
                attention_patterns=attn_patterns,
                ffn_experts=ffn_experts
            ))

        # =====================================================================
        # BLOCKS 16-31: DIV/MOD iterations
        # =====================================================================
        for i in range(16, 32):
            div_iter_idx = i - 16

            self.blocks.append(TransformerALUBlock(
                attention_patterns={},
                ffn_experts={
                    Opcode.DIV: DivIterFFN(),
                    Opcode.MOD: ModIterFFN(),
                }
            ))

        # =====================================================================
        # BLOCK 32: Finalization
        # =====================================================================
        self.blocks.append(TransformerALUBlock(
            attention_patterns={
                Opcode.LT: CmpBroadcastResultAttention(),
                Opcode.GT: CmpBroadcastResultAttention(),
                Opcode.LE: CmpBroadcastResultAttention(),
                Opcode.GE: CmpBroadcastResultAttention(),
            },
            ffn_experts={
                Opcode.AND: ClearBitsFFN(Opcode.AND),
                Opcode.OR: ClearBitsFFN(Opcode.OR),
                Opcode.XOR: ClearBitsFFN(Opcode.XOR),
                Opcode.MUL: MulClearTempFFN(),
                Opcode.MOD: ModResultFFN(),
                Opcode.LT: SequentialExpert([
                    CmpClearTempFFN(Opcode.LT), CmpExtractMSBBorrowFFN(Opcode.LT),
                    CmpClearResultFFN(Opcode.LT)
                ]),
                Opcode.GT: SequentialExpert([
                    CmpClearTempFFN(Opcode.GT), CmpExtractMSBBorrowFFN(Opcode.GT),
                    CmpClearResultFFN(Opcode.GT)
                ]),
                Opcode.LE: SequentialExpert([
                    CmpClearTempFFN(Opcode.LE), CmpExtractMSBBorrowFFN(Opcode.LE),
                    CmpClearResultFFN(Opcode.LE), CmpInvertResultFFN(Opcode.LE)
                ]),
                Opcode.GE: SequentialExpert([
                    CmpClearTempFFN(Opcode.GE), CmpExtractMSBBorrowFFN(Opcode.GE),
                    CmpClearResultFFN(Opcode.GE), CmpInvertResultFFN(Opcode.GE)
                ]),
            }
        ))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all transformer blocks."""
        for block in self.blocks:
            x = block(x)
        return x

    def num_layers(self) -> int:
        """Return number of transformer blocks."""
        return len(self.blocks)


class SequentialExpert(nn.Module):
    """Helper to combine multiple FFNs into one expert."""

    def __init__(self, ffns: List[nn.Module]):
        super().__init__()
        self.ffns = nn.ModuleList(ffns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ffn in self.ffns:
            x = ffn(x)
        return x


# =============================================================================
# LAYER SPECIFICATION DSL
# =============================================================================

class LayerSpec:
    """
    Declarative specification for transformer layers.

    Example:
        spec = LayerSpec()
        spec.layer(0)
            .attention({})
            .expert(ADD, [AddRawSumFFN(), InitResultFFN()])
            .expert(SUB, [SubRawDiffFFN(), SubInitResultFFN()])

        spec.layers(1, 8)
            .attention({ADD: carry_attn, SUB: carry_attn})
            .expert(ADD, [CarryIterFFN()])
            .expert(SUB, [BorrowIterFFN()])

        alu = spec.build()
    """

    def __init__(self):
        self.layer_specs: Dict[int, dict] = {}
        self._current_layer = None
        self._current_range = None

    def layer(self, idx: int) -> 'LayerSpec':
        """Start specifying a single layer."""
        self._current_range = None
        self._current_layer = idx
        if idx not in self.layer_specs:
            self.layer_specs[idx] = {'attention': {}, 'experts': {}}
        return self

    def layers(self, start: int, end: int) -> 'LayerSpec':
        """Start specifying a range of layers."""
        self._current_layer = None
        self._current_range = (start, end)
        for i in range(start, end):
            if i not in self.layer_specs:
                self.layer_specs[i] = {'attention': {}, 'experts': {}}
        return self

    def _get_layer_indices(self) -> List[int]:
        if self._current_layer is not None:
            return [self._current_layer]
        elif self._current_range is not None:
            return list(range(self._current_range[0], self._current_range[1]))
        return []

    def attention(self, patterns: Dict[int, nn.Module]) -> 'LayerSpec':
        """Add attention patterns for current layer(s)."""
        for idx in self._get_layer_indices():
            self.layer_specs[idx]['attention'].update(patterns)
        return self

    def expert(self, opcode: int, ffns: List[nn.Module]) -> 'LayerSpec':
        """Add expert FFN(s) for an opcode."""
        expert = SequentialExpert(ffns) if len(ffns) > 1 else ffns[0]
        for idx in self._get_layer_indices():
            self.layer_specs[idx]['experts'][opcode] = expert
        return self

    def build(self) -> nn.Module:
        """Build the transformer ALU from specifications."""
        max_layer = max(self.layer_specs.keys()) if self.layer_specs else 0
        blocks = []

        for i in range(max_layer + 1):
            if i in self.layer_specs:
                spec = self.layer_specs[i]
                blocks.append(TransformerALUBlock(
                    attention_patterns=spec['attention'],
                    ffn_experts=spec['experts']
                ))
            else:
                # Empty layer - all identity
                blocks.append(TransformerALUBlock({}, {}))

        return nn.Sequential(*blocks)


# =============================================================================
# DEMO
# =============================================================================

def demo_unified_transformer():
    """Demonstrate unified transformer ALU."""
    print("=" * 60)
    print("Unified Transformer ALU Demo")
    print("=" * 60)

    print("\nBuilding Unified Transformer ALU...")
    alu = UnifiedTransformerALU()
    print(f"Total layers: {alu.num_layers()}")

    # Test ADD
    print("\nTesting ADD: 100 + 200")
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    a, b = 100, 200
    for i in range(E.NUM_POSITIONS):
        x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
        x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
        x[0, i, E.OP_START + Opcode.ADD] = 1.0
        x[0, i, E.POS] = float(i)

    y = alu(x)
    result = sum(int(round(y[0, i, E.RESULT].item())) << (i*4) for i in range(8))
    print(f"  Result: {result} (expected 300)")

    # Test SUB
    print("\nTesting SUB: 500 - 123")
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    a, b = 500, 123
    for i in range(E.NUM_POSITIONS):
        x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
        x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
        x[0, i, E.OP_START + Opcode.SUB] = 1.0
        x[0, i, E.POS] = float(i)

    y = alu(x)
    result = sum(int(round(y[0, i, E.RESULT].item())) << (i*4) for i in range(8))
    print(f"  Result: {result} (expected 377)")

    print("\n" + "=" * 60)
    print("Architecture Summary:")
    print(f"  - {alu.num_layers()} transformer blocks")
    print(f"  - Each block: Attention (MoE) -> FFN (MoE)")
    print(f"  - Operations route to experts or identity")
    print(f"  - All ops go through all layers")
    print("=" * 60)


if __name__ == "__main__":
    demo_unified_transformer()
