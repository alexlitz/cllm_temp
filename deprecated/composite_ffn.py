"""
Composite FFN Factory for Neural VM.

Provides factory functions to compose PureFFNs into larger operations:
1. Sequential - Chain FFNs (output feeds to next)
2. Parallel - Run FFNs in parallel, combine outputs
3. MoE Routed - Route to different FFNs based on embedding slots
4. Loop Unroll - Unroll iterations into stacked layers
5. Conditional - Gate FFN execution based on embedding condition

This allows complex operations like 32-bit MUL to be expressed as
pure neural networks without Python control flow in forward pass.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Callable
from functools import reduce

from .embedding import E
from .base_layers import PureFFN, PureAttention


class SequentialPureFFN(nn.Module):
    """
    Chain multiple FFNs sequentially.

    Output of each FFN feeds into the next.
    Equivalent to: x = ffn_n(...ffn_2(ffn_1(x)))
    """

    def __init__(self, ffns: List[nn.Module]):
        super().__init__()
        self.ffns = nn.ModuleList(ffns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ffn in self.ffns:
            x = ffn(x)
        return x


class ParallelPureFFN(nn.Module):
    """
    Run multiple FFNs in parallel, combine outputs.

    Combine modes:
    - 'add': Sum all outputs (residual style)
    - 'mean': Average outputs
    - 'max': Element-wise max
    - 'concat': Concatenate (increases dimension)
    """

    def __init__(self, ffns: List[nn.Module], combine: str = 'add'):
        super().__init__()
        self.ffns = nn.ModuleList(ffns)
        self.combine = combine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [ffn(x) for ffn in self.ffns]

        if self.combine == 'add':
            return reduce(torch.add, outputs)
        elif self.combine == 'mean':
            return torch.stack(outputs).mean(dim=0)
        elif self.combine == 'max':
            return torch.stack(outputs).max(dim=0)[0]
        elif self.combine == 'concat':
            return torch.cat(outputs, dim=-1)
        else:
            raise ValueError(f"Unknown combine mode: {self.combine}")


class MoERoutedFFN(nn.Module):
    """
    Mixture-of-Experts routing between FFNs.

    Routes to different expert FFNs based on opcode or other
    embedding slots. Uses sparse computation - only active
    expert runs.
    """

    def __init__(self, experts: Dict[int, nn.Module], router_slot: int = E.OP_START):
        """
        Args:
            experts: Dict mapping opcode/key to expert FFN
            router_slot: Embedding slot to read routing key from
        """
        super().__init__()
        self.experts = nn.ModuleDict({str(k): v for k, v in experts.items()})
        self.router_slot = router_slot
        self.num_experts = len(experts)
        self.expert_keys = list(experts.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine active expert from embedding
        if self.router_slot == E.OP_START:
            # Opcode routing: find which opcode is active
            opcode_vec = x[0, 0, E.OP_START:E.OP_START + E.NUM_OPS]
            active_key = torch.argmax(opcode_vec).item()
        else:
            # Direct slot routing
            active_key = int(x[0, 0, self.router_slot].item())

        # Route to expert (sparse - only one runs)
        key_str = str(active_key)
        if key_str in self.experts:
            return self.experts[key_str](x)
        return x  # No expert for this key, pass through


class LoopUnrollFFN(nn.Module):
    """
    Unroll a loop into repeated FFN applications.

    Creates N copies of the body FFN (or shares weights).
    Useful for carry propagation, iterative algorithms.
    """

    def __init__(self, body_ffn: nn.Module, iterations: int, share_weights: bool = True):
        """
        Args:
            body_ffn: FFN to repeat
            iterations: Number of iterations
            share_weights: If True, reuse same FFN. If False, create copies.
        """
        super().__init__()
        self.iterations = iterations
        self.share_weights = share_weights

        if share_weights:
            self.body = body_ffn
        else:
            # Create independent copies
            self.bodies = nn.ModuleList([
                type(body_ffn)() if hasattr(body_ffn, '__class__') else body_ffn
                for _ in range(iterations)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.share_weights:
            for _ in range(self.iterations):
                x = self.body(x)
        else:
            for body in self.bodies:
                x = body(x)
        return x


class ConditionalFFN(nn.Module):
    """
    Conditionally execute FFN based on embedding slot value.

    Executes FFN only when condition slot exceeds threshold.
    Uses soft gating for differentiability.
    """

    def __init__(self, ffn: nn.Module, condition_slot: int,
                 threshold: float = 0.5, soft: bool = False):
        """
        Args:
            ffn: FFN to conditionally execute
            condition_slot: Embedding slot to check
            threshold: Activation threshold
            soft: If True, use sigmoid gating. If False, hard threshold.
        """
        super().__init__()
        self.ffn = ffn
        self.condition_slot = condition_slot
        self.threshold = threshold
        self.soft = soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        condition = x[0, 0, self.condition_slot].item()

        if self.soft:
            # Soft gating: blend original and FFN output
            gate = torch.sigmoid(torch.tensor(condition - self.threshold) * E.SCALE)
            ffn_out = self.ffn(x)
            return x * (1 - gate) + ffn_out * gate
        else:
            # Hard gating: execute or skip
            if condition > self.threshold:
                return self.ffn(x)
            return x


class AttentionThenFFN(nn.Module):
    """
    Attention layer followed by FFN (standard transformer block pattern).

    Useful for operations that need cross-position communication
    followed by per-position computation.
    """

    def __init__(self, attention: nn.Module, ffn: nn.Module):
        super().__init__()
        self.attention = attention
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)
        return x


class IteratedAttentionFFN(nn.Module):
    """
    Iterate attention + FFN block multiple times.

    Pattern: for i in range(n): x = ffn(attn(x))
    This is the core pattern for carry propagation.
    """

    def __init__(self, attention: nn.Module, ffn: nn.Module, iterations: int):
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.iterations):
            x = self.attention(x)
            x = self.ffn(x)
        return x


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def sequential(*ffns) -> SequentialPureFFN:
    """Create a sequential composition of FFNs."""
    return SequentialPureFFN(list(ffns))


def parallel(*ffns, combine: str = 'add') -> ParallelPureFFN:
    """Create a parallel composition of FFNs."""
    return ParallelPureFFN(list(ffns), combine)


def moe_routed(experts: Dict[int, nn.Module], router_slot: int = E.OP_START) -> MoERoutedFFN:
    """Create an MoE-routed FFN."""
    return MoERoutedFFN(experts, router_slot)


def loop_unroll(body: nn.Module, iterations: int, share_weights: bool = True) -> LoopUnrollFFN:
    """Create an unrolled loop FFN."""
    return LoopUnrollFFN(body, iterations, share_weights)


def conditional(ffn: nn.Module, slot: int, threshold: float = 0.5) -> ConditionalFFN:
    """Create a conditional FFN."""
    return ConditionalFFN(ffn, slot, threshold)


def attn_ffn(attention: nn.Module, ffn: nn.Module) -> AttentionThenFFN:
    """Create attention + FFN block."""
    return AttentionThenFFN(attention, ffn)


def iterated_attn_ffn(attention: nn.Module, ffn: nn.Module, iterations: int) -> IteratedAttentionFFN:
    """Create iterated attention + FFN block."""
    return IteratedAttentionFFN(attention, ffn, iterations)


# =============================================================================
# COMPOSED OPERATION BUILDERS
# =============================================================================

class ComposedOperationBuilder:
    """
    Builder for creating composed neural operations.

    Example:
        add_op = (ComposedOperationBuilder()
            .add(AddRawSumFFN())
            .add(InitResultFFN())
            .add(CarryDetectFFN())
            .iterate(CarryPropagateAttention(), CarryIterFFN(), 7)
            .build())
    """

    def __init__(self):
        self.stages: List[nn.Module] = []

    def add(self, ffn: nn.Module) -> 'ComposedOperationBuilder':
        """Add an FFN to the sequence."""
        self.stages.append(ffn)
        return self

    def add_all(self, *ffns) -> 'ComposedOperationBuilder':
        """Add multiple FFNs to the sequence."""
        self.stages.extend(ffns)
        return self

    def parallel(self, *ffns, combine: str = 'add') -> 'ComposedOperationBuilder':
        """Add a parallel block."""
        self.stages.append(ParallelPureFFN(list(ffns), combine))
        return self

    def iterate(self, attention: nn.Module, ffn: nn.Module,
                iterations: int) -> 'ComposedOperationBuilder':
        """Add an iterated attention + FFN block."""
        self.stages.append(IteratedAttentionFFN(attention, ffn, iterations))
        return self

    def loop(self, body: nn.Module, iterations: int,
             share_weights: bool = True) -> 'ComposedOperationBuilder':
        """Add an unrolled loop."""
        self.stages.append(LoopUnrollFFN(body, iterations, share_weights))
        return self

    def conditional(self, ffn: nn.Module, slot: int,
                   threshold: float = 0.5) -> 'ComposedOperationBuilder':
        """Add a conditional FFN."""
        self.stages.append(ConditionalFFN(ffn, slot, threshold))
        return self

    def moe(self, experts: Dict[int, nn.Module],
            router_slot: int = E.OP_START) -> 'ComposedOperationBuilder':
        """Add an MoE routing block."""
        self.stages.append(MoERoutedFFN(experts, router_slot))
        return self

    def build(self) -> SequentialPureFFN:
        """Build the composed operation."""
        return SequentialPureFFN(self.stages)


# =============================================================================
# BAKED COMPOSITE FFN (Combines weights at construction time)
# =============================================================================

class BakedCompositeFFN(PureFFN):
    """
    A composite FFN that bakes multiple FFN weights into one.

    For additive composition, the weights can be combined:
    out = ffn1(x) + ffn2(x)
        = silu(x@W1_up + b1_up) * (x@W1_gate + b1_gate) @ W1_down
        + silu(x@W2_up + b2_up) * (x@W2_gate + b2_gate) @ W2_down

    This can be approximated by a single larger FFN.
    """

    def __init__(self, ffns: List[PureFFN], combine: str = 'add'):
        """
        Bake multiple FFNs into one larger FFN.

        Args:
            ffns: List of PureFFN instances to combine
            combine: How to combine ('add' supported)
        """
        # Calculate total hidden dimension
        total_hidden = sum(ffn.hidden_dim for ffn in ffns)
        super().__init__(E.DIM, hidden_dim=total_hidden)

        self.source_ffns = ffns
        self.combine = combine
        self._bake_weights()

    def _bake_weights(self):
        """Combine weights from source FFNs."""
        with torch.no_grad():
            offset = 0
            for ffn in self.source_ffns:
                h = ffn.hidden_dim

                # Copy weight blocks
                self.W_up[offset:offset+h, :] = ffn.W_up
                self.W_gate[offset:offset+h, :] = ffn.W_gate
                self.b_up[offset:offset+h] = ffn.b_up
                self.b_gate[offset:offset+h] = ffn.b_gate
                self.W_down[:, offset:offset+h] = ffn.W_down

                offset += h


# =============================================================================
# EXAMPLE: BUILD 32-BIT ADD AS COMPOSED OPERATION
# =============================================================================

def build_add_operation():
    """
    Build 32-bit ADD as a composed neural operation.

    Architecture:
    1. Compute raw sum per nibble
    2. Initialize result
    3. Detect carries
    4. Propagate carries (7 iterations of attention + FFN)
    """
    from .arithmetic_ops import (
        AddRawSumFFN, InitResultFFN, CarryDetectFFN, CarryPropagateAttention,
        ZeroFirstCarryFFN, ClearCarryOutFFN, CarryIterFFN, ClearCarryInFFN
    )

    # Build composed carry iteration block
    carry_iter = sequential(
        ZeroFirstCarryFFN(),
        ClearCarryOutFFN(),
        CarryIterFFN(),
        ClearCarryInFFN()
    )

    # Build full ADD operation
    add_op = (ComposedOperationBuilder()
        .add(AddRawSumFFN())
        .add(InitResultFFN())
        .add(CarryDetectFFN())
        .iterate(CarryPropagateAttention(), carry_iter, 7)
        .build())

    return add_op


def build_sub_operation():
    """Build 32-bit SUB as a composed neural operation."""
    from .arithmetic_ops import (
        SubRawDiffFFN, SubInitResultFFN, BorrowDetectFFN, CarryPropagateAttention,
        ZeroFirstBorrowFFN, ClearBorrowOutFFN, BorrowIterFFN, ClearBorrowInFFN
    )

    borrow_iter = sequential(
        ZeroFirstBorrowFFN(),
        ClearBorrowOutFFN(),
        BorrowIterFFN(),
        ClearBorrowInFFN()
    )

    sub_op = (ComposedOperationBuilder()
        .add(SubRawDiffFFN())
        .add(SubInitResultFFN())
        .add(BorrowDetectFFN())
        .iterate(CarryPropagateAttention(), borrow_iter, 7)
        .build())

    return sub_op


# =============================================================================
# DEMO
# =============================================================================

def demo_composite_ffn():
    """Demonstrate composite FFN construction."""
    print("=" * 60)
    print("Composite FFN Demo")
    print("=" * 60)

    # Build ADD operation
    print("\n1. Building composed ADD operation...")
    try:
        add_op = build_add_operation()
        print(f"   ADD operation has {len(add_op.ffns)} stages")

        # Test it
        from .embedding import Opcode
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.NIB_A] = 5.0
            x[0, i, E.NIB_B] = 3.0
            x[0, i, E.OP_START + Opcode.ADD] = 1.0
            x[0, i, E.POS] = float(i)

        y = add_op(x)
        result = sum(int(y[0, i, E.RESULT].item()) << (i*4) for i in range(8))
        print(f"   Test: 0x55555555 + 0x33333333 = {hex(result)}")
        expected = 0x55555555 + 0x33333333
        print(f"   Expected: {hex(expected)}")
        print(f"   Match: {result == expected}")

    except Exception as e:
        print(f"   Error: {e}")

    # Demo builder pattern
    print("\n2. ComposedOperationBuilder pattern:")
    print("""
    add_op = (ComposedOperationBuilder()
        .add(AddRawSumFFN())
        .add(InitResultFFN())
        .add(CarryDetectFFN())
        .iterate(CarryPropagateAttention(), CarryIterFFN(), 7)
        .build())
    """)

    # Demo factory functions
    print("3. Factory functions:")
    print("   sequential(ffn1, ffn2, ffn3)  -> chain FFNs")
    print("   parallel(ffn1, ffn2)          -> run in parallel, add outputs")
    print("   moe_routed({1: ffn1, 2: ffn2}) -> route by opcode")
    print("   loop_unroll(body, 7)          -> unroll loop")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_composite_ffn()
