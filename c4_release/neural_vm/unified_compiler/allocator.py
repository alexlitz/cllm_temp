"""
Dimension Allocator for Neural VM Compiler.

Manages the allocation of embedding dimensions to semantic concepts.
Supports both fixed allocations (markers) and dynamic allocation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from .ir import DimensionAlloc, CompilerIR


@dataclass
class AllocationRegion:
    """A region of dimensions for a specific purpose."""
    name: str
    start: int
    end: int
    description: str = ""


# =============================================================================
# Standard Dimension Layout (from vm_step._SetDim)
# =============================================================================

# These are the fixed allocations that must be preserved for compatibility
STANDARD_REGIONS = [
    AllocationRegion("markers", 0, 12, "Register/memory identity flags"),
    AllocationRegion("address", 12, 60, "Address staging for memory access"),
    AllocationRegion("threshold", 60, 138, "Threshold detection outputs"),
    AllocationRegion("nibbles", 142, 206, "Input/output nibble encoding"),
    AllocationRegion("flags", 254, 296, "Opcode and transition flags"),
    AllocationRegion("clean", 306, 420, "Clean embed copies for V reads"),
    AllocationRegion("alu", 360, 396, "ALU computation staging"),
    AllocationRegion("staging", 420, 480, "MUL/DIV staging areas"),
    AllocationRegion("temp", 480, 512, "General temporaries"),
]


@dataclass
class DimensionAllocator:
    """Allocates embedding dimensions to semantic concepts.

    Manages dimension allocation with:
    - Fixed reservations for compatibility
    - Dynamic allocation for new concepts
    - Collision detection
    - Liveness tracking across layers

    Attributes:
        d_model: Total embedding dimension (default 512)
        allocations: Current allocations
        free_ranges: Available dimension ranges
    """
    d_model: int = 512
    allocations: Dict[str, DimensionAlloc] = field(default_factory=dict)
    _used: Set[int] = field(default_factory=set)

    def reserve(self, name: str, start: int, count: int,
                layer_written: int = -1) -> DimensionAlloc:
        """Reserve a fixed dimension range.

        Used for compatibility with existing dimension layout.

        Args:
            name: Dimension name
            start: Starting index
            count: Number of dimensions
            layer_written: Which layer writes (-1 for embedding)

        Returns:
            The allocation

        Raises:
            ValueError: If dimensions overlap with existing allocation
        """
        # Check for overlap
        for i in range(start, start + count):
            if i in self._used:
                # Find what's using it
                for existing in self.allocations.values():
                    if existing.start <= i < existing.end:
                        raise ValueError(
                            f"Cannot reserve {name} [{start}:{start+count}]: "
                            f"dimension {i} used by {existing.name}"
                        )

        # Create allocation
        alloc = DimensionAlloc(name, start, count, layer_written)
        self.allocations[name] = alloc

        # Mark as used
        for i in range(start, start + count):
            self._used.add(i)

        return alloc

    def allocate(self, name: str, count: int,
                 layer_written: int = -1,
                 prefer_region: Optional[str] = None) -> DimensionAlloc:
        """Dynamically allocate dimensions.

        Finds a free range of the requested size.

        Args:
            name: Dimension name
            count: Number of dimensions needed
            layer_written: Which layer writes
            prefer_region: Preferred region name (from STANDARD_REGIONS)

        Returns:
            The allocation

        Raises:
            ValueError: If no free space available
        """
        # Find preferred region if specified
        start_search = 0
        end_search = self.d_model

        if prefer_region:
            for region in STANDARD_REGIONS:
                if region.name == prefer_region:
                    start_search = region.start
                    end_search = region.end
                    break

        # Find first fit
        start = self._find_free_range(start_search, end_search, count)
        if start is None:
            # Try full range if preferred region failed
            if prefer_region:
                start = self._find_free_range(0, self.d_model, count)

        if start is None:
            raise ValueError(
                f"Cannot allocate {count} dimensions for {name}: "
                f"no free space available"
            )

        return self.reserve(name, start, count, layer_written)

    def _find_free_range(self, start: int, end: int, count: int) -> Optional[int]:
        """Find a free range of the requested size."""
        run_start = None
        run_length = 0

        for i in range(start, end):
            if i not in self._used:
                if run_start is None:
                    run_start = i
                run_length += 1

                if run_length >= count:
                    return run_start
            else:
                run_start = None
                run_length = 0

        return None

    def get(self, name: str) -> DimensionAlloc:
        """Get an allocation by name."""
        if name not in self.allocations:
            raise KeyError(f"Dimension '{name}' not allocated")
        return self.allocations[name]

    def get_start(self, name: str) -> int:
        """Get starting index of a dimension."""
        return self.get(name).start

    def get_range(self, name: str) -> range:
        """Get range of a dimension."""
        alloc = self.get(name)
        return range(alloc.start, alloc.end)

    def free_count(self) -> int:
        """Count free dimensions."""
        return self.d_model - len(self._used)

    def usage_summary(self) -> str:
        """Generate usage summary."""
        lines = [
            f"Dimension Allocator Summary",
            f"  Total: {self.d_model}",
            f"  Used: {len(self._used)}",
            f"  Free: {self.free_count()}",
            f"",
            f"Allocations ({len(self.allocations)}):",
        ]

        # Sort by start
        sorted_allocs = sorted(
            self.allocations.values(),
            key=lambda a: a.start
        )

        for alloc in sorted_allocs:
            layer = f"L{alloc.layer_written}" if alloc.layer_written >= 0 else "emb"
            lines.append(
                f"  [{alloc.start:3d}:{alloc.end:3d}] {alloc.name:20s} ({layer})"
            )

        return '\n'.join(lines)

    def to_ir(self) -> Dict[str, DimensionAlloc]:
        """Export allocations for CompilerIR."""
        return dict(self.allocations)


# =============================================================================
# Standard Allocator Factory
# =============================================================================

def create_standard_allocator() -> DimensionAllocator:
    """Create allocator with standard Neural VM dimension layout.

    This matches the layout in vm_step._SetDim for compatibility.
    """
    alloc = DimensionAllocator(d_model=512)

    # Markers (written by embedding, L-1) - must match vm_step._SetDim exactly
    alloc.reserve("MARK_PC", 0, 1, -1)
    alloc.reserve("MARK_AX", 1, 1, -1)
    alloc.reserve("MARK_SP", 2, 1, -1)
    alloc.reserve("MARK_BP", 3, 1, -1)
    alloc.reserve("MARK_MEM", 4, 1, -1)
    alloc.reserve("MARK_SE", 5, 1, -1)   # STEP_END or DATA_END
    alloc.reserve("IS_BYTE", 6, 1, -1)
    alloc.reserve("IS_MARK", 7, 1, -1)
    alloc.reserve("CONST", 8, 1, -1)     # Constant 1.0
    alloc.reserve("MARK_CS", 9, 1, -1)   # CODE_START only
    alloc.reserve("MARK_SE_ONLY", 10, 1, -1)  # STEP_END only (not DATA_END)
    alloc.reserve("MARK_STACK0", 11, 1, -1)
    alloc.reserve("HAS_SE", 137, 1, 1)   # Written by L1 head 3

    # Embed nibbles (written by embedding)
    alloc.reserve("EMBED_LO", 142, 16, -1)
    alloc.reserve("EMBED_HI", 158, 16, -1)

    # Output nibbles (written by ALU layers)
    alloc.reserve("OUTPUT_LO", 174, 16, 10)
    alloc.reserve("OUTPUT_HI", 190, 16, 10)

    # Threshold outputs (written by L0-L2)
    # L0 threshold heads H0-H7 (each has 7 marker outputs)
    alloc.reserve("H0", 60, 7, 0)
    alloc.reserve("H1", 67, 7, 0)
    alloc.reserve("H2", 74, 7, 0)
    alloc.reserve("H3", 81, 7, 0)
    alloc.reserve("H4", 88, 7, 0)
    alloc.reserve("H5", 95, 7, 0)
    alloc.reserve("H6", 102, 7, 0)
    alloc.reserve("H7", 109, 7, 0)
    # L1 threshold heads
    alloc.reserve("L1H0", 116, 7, 1)
    alloc.reserve("L1H1", 123, 7, 1)
    alloc.reserve("L1H2", 130, 7, 1)
    # Note: HAS_SE is at 11, allocated above with markers (written L1 head 3)
    # Byte index flags
    alloc.reserve("BYTE_INDEX_0", 138, 1, 1)
    alloc.reserve("BYTE_INDEX_1", 139, 1, 1)
    alloc.reserve("BYTE_INDEX_2", 140, 1, 1)
    alloc.reserve("BYTE_INDEX_3", 141, 1, 1)
    # L1H4 and L2H0 (later in dimension space)
    alloc.reserve("L1H4", 297, 7, 1)
    alloc.reserve("L2H0", 452, 7, 2)

    # STACK0 byte 0 flag (written by L1 FFN)
    alloc.reserve("STACK0_BYTE0", 304, 1, 1)

    # AX carry staging (written by L3 head 1)
    alloc.reserve("AX_CARRY_LO", 328, 16, 3)
    alloc.reserve("AX_CARRY_HI", 344, 16, 3)

    # MEM val byte position flags (written by L2 FFN)
    alloc.reserve("MEM_VAL_B0", 461, 1, 2)
    alloc.reserve("MEM_VAL_B1", 462, 1, 2)
    alloc.reserve("MEM_VAL_B2", 463, 1, 2)
    alloc.reserve("MEM_VAL_B3", 464, 1, 2)

    # AX full value (written by L3 head 5) - NOTE: aliases with TEMP region
    # Not reserving separately as they share dimensions (never used simultaneously)
    # alloc.reserve("AX_FULL_LO", 471, 16, 3)
    # alloc.reserve("AX_FULL_HI", 487, 16, 3)

    # NEXT_* transition flags (written by L0 FFN)
    alloc.reserve("NEXT_PC", 254, 1, 0)
    alloc.reserve("NEXT_AX", 255, 1, 0)
    alloc.reserve("NEXT_SP", 256, 1, 0)
    alloc.reserve("NEXT_BP", 257, 1, 0)
    alloc.reserve("NEXT_STACK0", 258, 1, 0)
    alloc.reserve("NEXT_MEM", 259, 1, 0)
    alloc.reserve("NEXT_SE", 260, 1, 0)
    alloc.reserve("NEXT_HALT", 261, 1, 5)

    # Opcode flags (written by L5)
    alloc.reserve("OP_LEA", 262, 1, 5)
    alloc.reserve("OP_IMM", 263, 1, 5)
    alloc.reserve("OP_JMP", 264, 1, 5)
    alloc.reserve("OP_JSR", 265, 1, 5)
    alloc.reserve("OP_BZ", 266, 1, 5)
    alloc.reserve("OP_BNZ", 267, 1, 5)
    alloc.reserve("OP_ENT", 268, 1, 5)
    alloc.reserve("OP_ADJ", 269, 1, 5)
    alloc.reserve("OP_LEV", 270, 1, 5)
    alloc.reserve("OP_LI", 271, 1, 5)
    alloc.reserve("OP_LC", 272, 1, 5)
    alloc.reserve("OP_SI", 273, 1, 5)
    alloc.reserve("OP_SC", 274, 1, 5)
    alloc.reserve("OP_PSH", 275, 1, 5)
    alloc.reserve("OP_OR", 276, 1, 5)
    alloc.reserve("OP_XOR", 277, 1, 5)
    alloc.reserve("OP_AND", 278, 1, 5)
    alloc.reserve("OP_EQ", 279, 1, 5)
    alloc.reserve("OP_NE", 280, 1, 5)
    alloc.reserve("OP_LT", 281, 1, 5)
    alloc.reserve("OP_GT", 282, 1, 5)
    alloc.reserve("OP_LE", 283, 1, 5)
    alloc.reserve("OP_GE", 284, 1, 5)
    alloc.reserve("OP_SHL", 285, 1, 5)
    alloc.reserve("OP_SHR", 286, 1, 5)
    alloc.reserve("OP_ADD", 287, 1, 5)
    alloc.reserve("OP_SUB", 288, 1, 5)
    alloc.reserve("OP_MUL", 289, 1, 5)
    alloc.reserve("OP_DIV", 290, 1, 5)
    alloc.reserve("OP_MOD", 291, 1, 5)
    alloc.reserve("OP_EXIT", 292, 1, 5)
    alloc.reserve("OP_NOP", 293, 1, 5)
    alloc.reserve("OP_PUTCHAR", 294, 1, 5)
    alloc.reserve("OP_GETCHAR", 295, 1, 5)

    # ALU staging (written by L7-L9)
    alloc.reserve("ALU_LO", 360, 16, 7)
    alloc.reserve("ALU_HI", 376, 16, 7)
    alloc.reserve("CARRY", 392, 4, 9)

    # CMP flags (written by L6 attention)
    # CMP[0] = IS_JMP, CMP[1] = IS_EXIT, CMP[2-7] = other flags
    alloc.reserve("CMP", 396, 8, 6)

    # Fetch results (written by L5)
    alloc.reserve("FETCH_LO", 420, 16, 5)
    alloc.reserve("FETCH_HI", 436, 16, 5)
    alloc.reserve("OPCODE_BYTE_LO", 238, 8, 5)
    alloc.reserve("OPCODE_BYTE_HI", 246, 8, 5)

    # TEMP space (multi-purpose)
    alloc.reserve("TEMP", 480, 32, -1)

    # Clean embed copies - CLEAN_EMBED_HI shifted to 404 to avoid collisions
    alloc.reserve("CLEAN_EMBED_LO", 306, 16, 3)
    alloc.reserve("CLEAN_EMBED_HI", 404, 16, 3)

    # Address key for memory (written by embedding, not layer)
    alloc.reserve("ADDR_KEY", 206, 16, -1)

    # PC_PLUS1 (aliases AX_FULL, used for L5 dynamic fetch at PC marker)
    # These are at 471-502 in the baseline, which overlaps with end of TEMP
    # For feature parity, we use the same positions but the compiler
    # handles the aliasing correctly since PC_PLUS1 is only used at PC marker
    # and TEMP is only used at AX marker
    # NOTE: Not reserving these as they're handled by position-specific logic

    return alloc


def allocator_from_ir(ir: CompilerIR) -> DimensionAllocator:
    """Create allocator from existing IR."""
    alloc = DimensionAllocator(d_model=ir.d_model)

    for name, dim_alloc in ir.dimensions.items():
        alloc.reserve(name, dim_alloc.start, dim_alloc.count, dim_alloc.layer_written)

    return alloc
