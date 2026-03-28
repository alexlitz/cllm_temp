"""
Layer Allocation Strategy

Uses graph coloring to minimize layers by sharing between non-conflicting operations.

Key insight: Since only one opcode executes at a time, operations with different opcodes
never conflict. We can share layers by having multiple operations at the same "pipeline depth".

Example:
- MUL layer 0, DIV layer 0, ADD layer 0 can all share Layer 0
- MUL layer 1, DIV layer 1, ADD layer 1 can all share Layer 1
- etc.

Total layers needed = max(depth of all operations)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from .embedding import Opcode


@dataclass
class LayerRange:
    """Range of layers allocated to an operation."""
    start: int
    end: int  # Exclusive
    num_layers: int

    def __post_init__(self):
        assert self.end - self.start == self.num_layers


class LayerAllocator:
    """Allocates operations to layer ranges using graph coloring."""

    def __init__(self, use_sharing: bool = True):
        """
        Args:
            use_sharing: If True, operations share layers (requires opcode gating).
                        If False, operations get separate layer ranges (simpler, more layers).
        """
        self.use_sharing = use_sharing
        self.layer_requirements = self._calculate_layer_requirements()

        if use_sharing:
            self.total_layers = self._calculate_total_layers_shared()
            self.allocations = self._allocate_layers_shared()
        else:
            self.total_layers = self._calculate_total_layers_sequential()
            self.allocations = self._allocate_layers_sequential()

    def _calculate_layer_requirements(self) -> Dict[str, int]:
        """Calculate how many layers each operation needs."""
        return {
            # Multi-layer ALU operations (from build_*_layers)
            'MUL': 7,    # Schoolbook + 3 carry passes + GenProp + Lookahead + Correction
            'DIV': 4,    # Clear+Gather+Reciprocal, Multiply, Floor, ChunkSubtract
            'MOD': 5,    # DivScalar, Floor, ChunkSubtract, MulSub, Correction
            'SHL': 2,    # Precompute, Select
            'SHR': 2,    # Precompute, Select

            # 3-layer carry/borrow operations
            'ADD': 3,    # RawAndGen, CarryLookahead, Finalize
            'SUB': 3,    # RawAndGen, BorrowLookahead, Finalize

            # Single-layer bitwise (bit-level primitives)
            'OR': 1,
            'XOR': 1,
            'AND': 1,

            # Single-layer comparisons (share layer with opcode gating)
            'COMPARISONS': 1,  # EQ, NE, LT, GT, LE, GE

            # Single-layer register operations
            'REGISTER': 1,  # LEA, IMM

            # Single-layer control flow
            'CONTROL': 1,  # JMP, BZ, BNZ, ADJ, MALC, FREE

            # Multi-layer memory operations (future: attention-based)
            'MEMORY': 2,  # LI, LC, SI, SC, PSH, JSR, ENT, LEV (placeholder)
        }

    def _calculate_total_layers_shared(self) -> int:
        """Calculate total layers with sharing (graph coloring).

        Since only one opcode runs at a time, all operations can share layers.
        Total layers = max pipeline depth across all operations.
        """
        return max(self.layer_requirements.values())

    def _calculate_total_layers_sequential(self) -> int:
        """Calculate total layers without sharing (sequential allocation)."""
        return sum(self.layer_requirements.values())

    def _allocate_layers_shared(self) -> Dict[str, LayerRange]:
        """Allocate operations to shared layers (graph coloring).

        All operations start at layer 0 and use as many layers as they need.
        They share layers because opcode gating ensures only one fires at a time.
        """
        allocations = {}

        for op_name, num_layers in self.layer_requirements.items():
            allocations[op_name] = LayerRange(
                start=0,
                end=num_layers,
                num_layers=num_layers
            )

        return allocations

    def _allocate_layers_sequential(self) -> Dict[str, LayerRange]:
        """Allocate operations to separate layers (no sharing)."""
        allocations = {}
        current_layer = 0

        # Allocate in order of dependency and efficiency
        allocation_order = [
            # Multi-layer arithmetic (most complex, allocate first)
            'MUL', 'DIV', 'MOD',

            # Multi-layer shifts
            'SHL', 'SHR',

            # 3-layer ADD/SUB
            'ADD', 'SUB',

            # Single-layer bitwise
            'OR', 'XOR', 'AND',

            # Single-layer logic/control/register
            'COMPARISONS', 'REGISTER', 'CONTROL',

            # Memory operations (last, may need special handling)
            'MEMORY',
        ]

        for op_name in allocation_order:
            num_layers = self.layer_requirements[op_name]
            allocations[op_name] = LayerRange(
                start=current_layer,
                end=current_layer + num_layers,
                num_layers=num_layers
            )
            current_layer += num_layers

        return allocations

    def get_layer_range(self, op_name: str) -> LayerRange:
        """Get layer range for an operation."""
        return self.allocations[op_name]

    def get_layer_for_opcode(self, opcode: int) -> LayerRange:
        """Get layer range for a specific opcode."""
        # Map opcodes to operation names
        opcode_map = {
            Opcode.MUL: 'MUL',
            Opcode.DIV: 'DIV',
            Opcode.MOD: 'MOD',
            Opcode.SHL: 'SHL',
            Opcode.SHR: 'SHR',
            Opcode.ADD: 'ADD',
            Opcode.SUB: 'SUB',
            Opcode.OR: 'OR',
            Opcode.XOR: 'XOR',
            Opcode.AND: 'AND',
            Opcode.EQ: 'COMPARISONS',
            Opcode.NE: 'COMPARISONS',
            Opcode.LT: 'COMPARISONS',
            Opcode.GT: 'COMPARISONS',
            Opcode.LE: 'COMPARISONS',
            Opcode.GE: 'COMPARISONS',
            Opcode.LEA: 'REGISTER',
            Opcode.IMM: 'REGISTER',
            Opcode.JMP: 'CONTROL',
            Opcode.BZ: 'CONTROL',
            Opcode.BNZ: 'CONTROL',
            Opcode.ADJ: 'CONTROL',
            Opcode.MALC: 'CONTROL',
            Opcode.FREE: 'CONTROL',
            Opcode.LI: 'MEMORY',
            Opcode.LC: 'MEMORY',
            Opcode.SI: 'MEMORY',
            Opcode.SC: 'MEMORY',
            Opcode.PSH: 'MEMORY',
            Opcode.JSR: 'MEMORY',
            Opcode.ENT: 'MEMORY',
            Opcode.LEV: 'MEMORY',
        }

        op_name = opcode_map.get(opcode)
        if op_name is None:
            raise ValueError(f"Unknown opcode {opcode}")

        return self.allocations[op_name]

    def print_allocation(self):
        """Print layer allocation table."""
        print("=" * 70)
        print("LAYER ALLOCATION" + (" (SHARED)" if self.use_sharing else " (SEQUENTIAL)"))
        print("=" * 70)
        print(f"Total layers required: {self.total_layers}")
        print()

        if self.use_sharing:
            print("Operations share layers via opcode gating:")
            print()
            # Group by layer range
            by_range = {}
            for op_name, layer_range in self.allocations.items():
                key = (layer_range.start, layer_range.end)
                if key not in by_range:
                    by_range[key] = []
                by_range[key].append(op_name)

            for (start, end), ops in sorted(by_range.items()):
                layers_str = f"{start}" if end - start == 1 else f"{start}-{end-1}"
                ops_str = ", ".join(ops)
                print(f"  Layers {layers_str:8s} → {ops_str}")
        else:
            for op_name, layer_range in self.allocations.items():
                layers_str = f"{layer_range.start}"
                if layer_range.num_layers > 1:
                    layers_str = f"{layer_range.start}-{layer_range.end-1}"
                print(f"  {op_name:15s} → Layers {layers_str:8s} ({layer_range.num_layers} layer{'s' if layer_range.num_layers > 1 else ''})")

        print("=" * 70)
