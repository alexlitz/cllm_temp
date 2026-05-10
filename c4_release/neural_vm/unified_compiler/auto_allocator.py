"""
Auto-Allocator for Neural VM Compiler.

Automatically allocates residual stream dimensions based on liveness analysis.
Supports pinned positions for baseline compatibility and automatic allocation
for optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass
class DimSpec:
    """Specification for a semantic dimension in the residual stream.

    Attributes:
        name: Semantic name (e.g., "MARK_PC", "ALU_LO")
        size: Number of dimensions needed
        written_by: Layer that writes this (-1 for embedding)
        read_by: Layers that read this dimension
        pinned: If set, force this exact start position (for baseline compat)
        aliasable: If True, can share space with non-overlapping dims
    """
    name: str
    size: int
    written_by: int = -1  # -1 means embedding
    read_by: List[int] = field(default_factory=list)
    pinned: Optional[int] = None
    aliasable: bool = True

    @property
    def live_start(self) -> int:
        """First layer where this dim is live (when written)."""
        return self.written_by if self.written_by >= 0 else -1

    @property
    def live_end(self) -> int:
        """Last layer where this dim is live (last read)."""
        return max(self.read_by) if self.read_by else self.live_start


class AutoAllocator:
    """Automatically allocates residual stream dimensions.

    Two modes:
    1. Pinned mode: All dims have pinned positions (baseline compatible)
    2. Auto mode: Dims allocated based on liveness (may differ from baseline)
    """

    def __init__(self, d_model: int = 512, n_layers: int = 17):
        self.d_model = d_model
        self.n_layers = n_layers
        self.specs: Dict[str, DimSpec] = {}
        self._allocated: Dict[str, int] = {}  # name -> start position
        self._is_allocated = False

    def declare(self, name: str, size: int, written_by: int = -1,
                read_by: Optional[List[int]] = None,
                pinned: Optional[int] = None,
                aliasable: bool = True) -> 'AutoAllocator':
        """Declare a semantic dimension.

        Args:
            name: Dimension name
            size: Number of dimensions
            written_by: Layer that writes (-1 for embedding)
            read_by: Layers that read this
            pinned: Force specific position (for baseline compat)
            aliasable: Can share space with non-overlapping dims

        Returns:
            self for chaining
        """
        self.specs[name] = DimSpec(
            name=name,
            size=size,
            written_by=written_by,
            read_by=read_by or [],
            pinned=pinned,
            aliasable=aliasable,
        )
        self._is_allocated = False
        return self

    def allocate(self) -> Dict[str, int]:
        """Assign positions to all dimensions.

        Uses pinned positions where specified, otherwise allocates
        based on liveness to minimize dimension usage.

        Returns:
            Dict of name -> start position
        """
        self._allocated = {}

        # First pass: assign pinned dimensions
        used_ranges: List[Tuple[int, int, int, int]] = []  # (start, end, live_start, live_end)

        for name, spec in self.specs.items():
            if spec.pinned is not None:
                self._allocated[name] = spec.pinned
                used_ranges.append((
                    spec.pinned,
                    spec.pinned + spec.size,
                    spec.live_start,
                    spec.live_end
                ))

        # Second pass: allocate unpinned dimensions
        # Sort by size descending for better packing
        unpinned = [(name, spec) for name, spec in self.specs.items()
                    if spec.pinned is None]
        unpinned.sort(key=lambda x: (-x[1].size, x[1].live_start))

        for name, spec in unpinned:
            pos = self._find_free_slot(used_ranges, spec)
            self._allocated[name] = pos
            used_ranges.append((pos, pos + spec.size, spec.live_start, spec.live_end))

        self._is_allocated = True
        return dict(self._allocated)

    def _find_free_slot(self, used: List[Tuple[int, int, int, int]],
                        spec: DimSpec) -> int:
        """Find first position that doesn't conflict."""
        for pos in range(0, self.d_model - spec.size + 1):
            if self._can_place(pos, spec, used):
                return pos
        raise ValueError(f"Cannot allocate {spec.size} dims for '{spec.name}' - no free space")

    def _can_place(self, pos: int, spec: DimSpec,
                   used: List[Tuple[int, int, int, int]]) -> bool:
        """Check if a dimension can be placed at position."""
        end = pos + spec.size

        for (u_start, u_end, u_live_start, u_live_end) in used:
            # Check position overlap
            if not (end <= u_start or pos >= u_end):
                # Positions overlap - check liveness overlap
                if spec.aliasable:
                    # Can alias if lifetimes don't overlap
                    if not (spec.live_end < u_live_start or spec.live_start > u_live_end):
                        return False
                else:
                    # Non-aliasable dims can't share space at all
                    return False
        return True

    def get(self, name: str) -> int:
        """Get allocated start position for a dimension."""
        if not self._is_allocated:
            self.allocate()
        if name not in self._allocated:
            raise KeyError(f"Dimension '{name}' not declared")
        return self._allocated[name]

    def get_range(self, name: str) -> range:
        """Get range of indices for a dimension."""
        start = self.get(name)
        return range(start, start + self.specs[name].size)

    def get_slice(self, name: str) -> slice:
        """Get slice for a dimension."""
        start = self.get(name)
        return slice(start, start + self.specs[name].size)

    def __getitem__(self, name: str) -> int:
        """Shorthand for get()."""
        return self.get(name)

    def usage_summary(self) -> str:
        """Generate usage summary."""
        if not self._is_allocated:
            self.allocate()

        lines = [
            f"AutoAllocator Summary",
            f"  d_model: {self.d_model}",
            f"  Declared: {len(self.specs)} dimensions",
            f"  Pinned: {sum(1 for s in self.specs.values() if s.pinned is not None)}",
            f"",
        ]

        # Sort by position
        sorted_dims = sorted(self._allocated.items(), key=lambda x: x[1])

        lines.append("Allocations:")
        for name, start in sorted_dims:
            spec = self.specs[name]
            pin = "P" if spec.pinned is not None else " "
            layer = f"L{spec.written_by:2d}" if spec.written_by >= 0 else "emb"
            lines.append(f"  {pin} [{start:3d}:{start+spec.size:3d}] {name:20s} written={layer}")

        return '\n'.join(lines)


# =============================================================================
# Standard Dimension Specifications
# =============================================================================

def create_standard_allocator(pinned: bool = True) -> AutoAllocator:
    """Create allocator with standard Neural VM dimensions.

    Args:
        pinned: If True, pin all positions to baseline values.
                If False, allow auto-allocation.

    Returns:
        Configured AutoAllocator
    """
    alloc = AutoAllocator(d_model=512, n_layers=17)

    def pin(pos: int) -> Optional[int]:
        return pos if pinned else None

    # =========================================================================
    # Markers (written by embedding, read by many layers)
    # =========================================================================
    alloc.declare("MARK_PC", 1, -1, [0, 1, 3, 4, 5, 6, 10, 14], pin(0))
    alloc.declare("MARK_AX", 1, -1, [0, 1, 3, 5, 6, 7, 9], pin(1))
    alloc.declare("MARK_SP", 1, -1, [0, 1, 3, 7, 8], pin(2))
    alloc.declare("MARK_BP", 1, -1, [0, 1, 2, 3, 7], pin(3))
    alloc.declare("MARK_MEM", 1, -1, [0, 1, 2, 14, 15], pin(4))
    alloc.declare("MARK_SE", 1, -1, [0, 1, 6], pin(5))
    alloc.declare("IS_BYTE", 1, -1, [1, 2], pin(6))
    alloc.declare("IS_MARK", 1, -1, [0, 1, 2], pin(7))
    alloc.declare("CONST", 1, -1, list(range(17)), pin(8), aliasable=False)  # Used everywhere
    alloc.declare("MARK_CS", 1, -1, [0, 5], pin(9))
    alloc.declare("MARK_SE_ONLY", 1, -1, [0, 1], pin(10))
    alloc.declare("MARK_STACK0", 1, -1, [0, 1, 3, 7], pin(11))

    # =========================================================================
    # Threshold outputs (written by L0-L2)
    # =========================================================================
    # L0 heads H0-H7: threshold detection
    alloc.declare("H0", 7, 0, [1, 2, 3], pin(60))
    alloc.declare("H1", 7, 0, [1, 2, 3], pin(67))
    alloc.declare("H2", 7, 0, [1, 2, 3], pin(74))
    alloc.declare("H3", 7, 0, [1, 2, 3], pin(81))
    alloc.declare("H4", 7, 0, [1, 2, 3], pin(88))
    alloc.declare("H5", 7, 0, [2, 3], pin(95))
    alloc.declare("H6", 7, 0, [2, 3], pin(102))
    alloc.declare("H7", 7, 0, [2, 3], pin(109))

    # L1 threshold heads
    alloc.declare("L1H0", 7, 1, [2, 3, 4], pin(116))
    alloc.declare("L1H1", 7, 1, [2, 3, 4], pin(123))
    alloc.declare("L1H2", 7, 1, [2, 3], pin(130))
    alloc.declare("HAS_SE", 1, 1, [3, 4, 5, 6], pin(137))

    # Byte index flags (L1)
    alloc.declare("BYTE_INDEX_0", 1, 1, [2, 3], pin(138))
    alloc.declare("BYTE_INDEX_1", 1, 1, [2, 3], pin(139))
    alloc.declare("BYTE_INDEX_2", 1, 1, [2, 3], pin(140))
    alloc.declare("BYTE_INDEX_3", 1, 1, [2, 3], pin(141))

    # L1H4 and L2H0
    alloc.declare("L1H4", 7, 1, [2, 3], pin(297))
    alloc.declare("L2H0", 7, 2, [3], pin(452))

    # =========================================================================
    # Embed nibbles (written by embedding)
    # =========================================================================
    alloc.declare("EMBED_LO", 16, -1, [3, 5, 7, 14], pin(142))
    alloc.declare("EMBED_HI", 16, -1, [3, 5, 7, 14], pin(158))

    # =========================================================================
    # Output nibbles (written by L10)
    # =========================================================================
    alloc.declare("OUTPUT_LO", 16, 10, [14, 15, 16], pin(174))
    alloc.declare("OUTPUT_HI", 16, 10, [14, 15, 16], pin(190))

    # =========================================================================
    # Address key (written by embedding) - 48 dims for 3 bytes x 16 nibbles
    # =========================================================================
    alloc.declare("ADDR_KEY", 48, -1, [5, 14, 15], pin(206))

    # =========================================================================
    # Opcode byte encoding (written by L5) - 16 dims each for nibble one-hot
    # Reuses ADDR_B0_LO/ADDR_B1_LO since those are unused in autoregressive mode
    # =========================================================================
    alloc.declare("OPCODE_BYTE_LO", 16, 5, [6], pin(12))   # aliases ADDR_B0_LO
    alloc.declare("OPCODE_BYTE_HI", 16, 5, [6], pin(28))   # aliases ADDR_B1_LO
    # Aliases for address-based code (same positions)
    alloc.declare("ADDR_B0_LO", 16, -1, [14], pin(12))
    alloc.declare("ADDR_B1_LO", 16, -1, [14], pin(28))

    # =========================================================================
    # Opcode flags aggregate (34 individual OP_* flags starting at 262)
    # This is a convenience alias - individual flags are declared below
    # =========================================================================
    alloc.declare("OPCODE_FLAGS", 34, 5, [6, 9], pin(262))

    # =========================================================================
    # NEXT_* transition flags (written by L0 FFN)
    # =========================================================================
    alloc.declare("NEXT_PC", 1, 0, [3, 4], pin(254))
    alloc.declare("NEXT_AX", 1, 0, [3, 4], pin(255))
    alloc.declare("NEXT_SP", 1, 0, [3, 4], pin(256))
    alloc.declare("NEXT_BP", 1, 0, [3, 4], pin(257))
    alloc.declare("NEXT_STACK0", 1, 0, [3, 4], pin(258))
    alloc.declare("NEXT_MEM", 1, 0, [3, 4], pin(259))
    alloc.declare("NEXT_SE", 1, 0, [3, 4, 6], pin(260))
    alloc.declare("NEXT_HALT", 1, 5, [6, 16], pin(261))

    # =========================================================================
    # Opcode flags (written by L5)
    # =========================================================================
    alloc.declare("OP_LEA", 1, 5, [6, 9], pin(262))
    alloc.declare("OP_IMM", 1, 5, [6, 9], pin(263))
    alloc.declare("OP_JMP", 1, 5, [6, 9], pin(264))
    alloc.declare("OP_JSR", 1, 5, [6, 9], pin(265))
    alloc.declare("OP_BZ", 1, 5, [6, 9], pin(266))
    alloc.declare("OP_BNZ", 1, 5, [6, 9], pin(267))
    alloc.declare("OP_ENT", 1, 5, [6, 9, 16], pin(268))
    alloc.declare("OP_ADJ", 1, 5, [6, 9], pin(269))
    alloc.declare("OP_LEV", 1, 5, [6, 9, 16], pin(270))
    alloc.declare("OP_LI", 1, 5, [6, 9, 14], pin(271))
    alloc.declare("OP_LC", 1, 5, [6, 9, 14], pin(272))
    alloc.declare("OP_SI", 1, 5, [6, 9, 15], pin(273))
    alloc.declare("OP_SC", 1, 5, [6, 9, 15], pin(274))
    alloc.declare("OP_PSH", 1, 5, [6, 9], pin(275))
    alloc.declare("OP_OR", 1, 5, [6, 9], pin(276))
    alloc.declare("OP_XOR", 1, 5, [6, 9], pin(277))
    alloc.declare("OP_AND", 1, 5, [6, 9], pin(278))
    alloc.declare("OP_EQ", 1, 5, [6, 9], pin(279))
    alloc.declare("OP_NE", 1, 5, [6, 9], pin(280))
    alloc.declare("OP_LT", 1, 5, [6, 9], pin(281))
    alloc.declare("OP_GT", 1, 5, [6, 9], pin(282))
    alloc.declare("OP_LE", 1, 5, [6, 9], pin(283))
    alloc.declare("OP_GE", 1, 5, [6, 9], pin(284))
    alloc.declare("OP_SHL", 1, 5, [6, 9], pin(285))
    alloc.declare("OP_SHR", 1, 5, [6, 9], pin(286))
    alloc.declare("OP_ADD", 1, 5, [6, 9], pin(287))
    alloc.declare("OP_SUB", 1, 5, [6, 9], pin(288))
    alloc.declare("OP_MUL", 1, 5, [6, 9, 11], pin(289))
    alloc.declare("OP_DIV", 1, 5, [6, 9, 12], pin(290))
    alloc.declare("OP_MOD", 1, 5, [6, 9, 12], pin(291))
    alloc.declare("OP_EXIT", 1, 5, [6], pin(292))
    alloc.declare("OP_NOP", 1, 5, [6], pin(293))
    alloc.declare("OP_PUTCHAR", 1, 5, [6], pin(294))
    alloc.declare("OP_GETCHAR", 1, 5, [6], pin(295))

    # =========================================================================
    # STACK0 byte 0 flag (written by L1 FFN)
    # =========================================================================
    alloc.declare("STACK0_BYTE0", 1, 1, [3], pin(304))

    # =========================================================================
    # Clean embed copies (written by L3)
    # =========================================================================
    alloc.declare("CLEAN_EMBED_LO", 16, 3, [5, 7], pin(306))

    # =========================================================================
    # AX carry staging (written by L3)
    # =========================================================================
    alloc.declare("AX_CARRY_LO", 16, 3, [6, 10], pin(328))
    alloc.declare("AX_CARRY_HI", 16, 3, [6, 10], pin(344))

    # =========================================================================
    # ALU staging (written by L7)
    # =========================================================================
    alloc.declare("ALU_LO", 16, 7, [9, 10], pin(360))
    alloc.declare("ALU_HI", 16, 7, [9, 10], pin(376))

    # =========================================================================
    # Carry flags (written by L9)
    # =========================================================================
    alloc.declare("CARRY", 4, 9, [10], pin(392))

    # =========================================================================
    # CMP flags (written by L6)
    # =========================================================================
    alloc.declare("CMP", 4, 6, [9, 10], pin(396))

    # =========================================================================
    # Clean embed HI (written by L3)
    # =========================================================================
    alloc.declare("CLEAN_EMBED_HI", 16, 3, [5, 7], pin(404))

    # =========================================================================
    # Fetch results (written by L5)
    # =========================================================================
    alloc.declare("FETCH_LO", 16, 5, [6, 10], pin(420))
    alloc.declare("FETCH_HI", 16, 5, [6, 10], pin(436))

    # =========================================================================
    # Address byte nibbles (for memory addressing)
    # =========================================================================
    alloc.declare("ADDR_B2_LO", 16, -1, [14], pin(44))

    # =========================================================================
    # IO flag (written by L6)
    # =========================================================================
    alloc.declare("IO_IS_PUTCHAR", 1, 6, [16], pin(296))

    # =========================================================================
    # ADJ implementation dimensions (SP + signed immediate)
    # =========================================================================
    alloc.declare("SP_OLD_LO", 8, 7, [8, 9], pin(297))
    alloc.declare("SP_OLD_HI", 8, 7, [8, 9], pin(305))
    alloc.declare("ADJ_CARRY", 2, 8, [9], pin(313))

    # =========================================================================
    # Reserved space
    # =========================================================================
    alloc.declare("RESERVED_315_327", 13, -1, [], pin(315))
    alloc.declare("RESERVED_400_415", 16, -1, [], pin(400))

    # =========================================================================
    # MUL/DIV staging (also FETCH staging)
    # =========================================================================
    alloc.declare("MUL_ACCUM", 16, 11, [12], pin(420))
    alloc.declare("DIV_STAGING", 16, 12, [13], pin(436))
    alloc.declare("FETCH_LO", 16, 5, [6, 10], pin(420))  # alias for MUL_ACCUM
    alloc.declare("FETCH_HI", 16, 5, [6, 10], pin(436))  # alias for DIV_STAGING

    # =========================================================================
    # MEM operation flags
    # =========================================================================
    alloc.declare("MEM_STORE", 1, 6, [7, 14], pin(459))
    alloc.declare("MEM_ADDR_SRC", 1, 6, [7, 14], pin(460))

    # =========================================================================
    # MEM val byte position flags (written by L2)
    # =========================================================================
    alloc.declare("MEM_VAL_B0", 1, 2, [14, 15], pin(461))
    alloc.declare("MEM_VAL_B1", 1, 2, [14, 15], pin(462))
    alloc.declare("MEM_VAL_B2", 1, 2, [14, 15], pin(463))
    alloc.declare("MEM_VAL_B3", 1, 2, [14, 15], pin(464))

    # =========================================================================
    # Conversational I/O flags (various aliases overlap with other dims)
    # =========================================================================
    alloc.declare("IO_IS_PRTF", 1, 5, [6], pin(464))  # aliases MEM_VAL_B3
    alloc.declare("IO_IS_READ", 1, 5, [6], pin(465))
    alloc.declare("IO_STATE", 1, 6, [16], pin(466))
    alloc.declare("IO_OUTPUT_COUNT", 1, 6, [16], pin(467))
    alloc.declare("IO_FORMAT_POS", 1, 6, [16], pin(468))
    alloc.declare("IO_IN_OUTPUT_MODE", 1, 6, [16], pin(469))
    alloc.declare("IO_OUTPUT_COMPLETE", 1, 6, [16], pin(470))

    # =========================================================================
    # PC+1 and AX_FULL staging
    # =========================================================================
    alloc.declare("PC_PLUS1_LO", 16, 4, [5], pin(471))
    alloc.declare("PC_PLUS1_HI", 16, 4, [5], pin(487))
    alloc.declare("AX_FULL_LO", 16, 3, [7], pin(471))  # alias for PC_PLUS1_LO
    alloc.declare("AX_FULL_HI", 16, 3, [7], pin(487))  # alias for PC_PLUS1_HI
    alloc.declare("FORMAT_PTR_LO", 16, 6, [16], pin(471))  # alias
    alloc.declare("FORMAT_PTR_HI", 16, 6, [16], pin(487))  # alias

    # =========================================================================
    # TEMP space (multi-purpose, written by various)
    # =========================================================================
    alloc.declare("TEMP", 32, -1, list(range(17)), pin(480), aliasable=False)
    alloc.declare("OUTPUT_BYTE_LO", 16, 6, [16], pin(480))  # overlaps TEMP
    alloc.declare("OUTPUT_BYTE_HI", 16, 6, [16], pin(496))  # overlaps TEMP+16

    # =========================================================================
    # Lookback detection flags
    # =========================================================================
    alloc.declare("LAST_WAS_THINKING_END", 1, 1, [2], pin(501))
    alloc.declare("LAST_WAS_THINKING_START", 1, 1, [2], pin(502))
    alloc.declare("LAST_WAS_BYTE", 1, 1, [2], pin(503))
    alloc.declare("ACTIVE_OPCODE_PRTF", 1, -1, [5], pin(504))
    alloc.declare("ACTIVE_OPCODE_READ", 1, -1, [5], pin(505))
    alloc.declare("MARK_THINKING_START", 1, -1, [1], pin(506))
    alloc.declare("MARK_THINKING_END", 1, -1, [1], pin(507))

    # =========================================================================
    # Address hi nibble dims
    # =========================================================================
    alloc.declare("ADDR_B0_HI", 16, -1, [14], pin(206))  # inside ADDR_KEY
    alloc.declare("ADDR_B1_HI", 16, -1, [14], pin(222))  # inside ADDR_KEY
    alloc.declare("ADDR_B2_HI", 16, -1, [14], pin(238))  # inside ADDR_KEY

    # =========================================================================
    # Tool call and thinking transitions
    # =========================================================================
    alloc.declare("IO_IS_TOOL_CALL", 1, 5, [6], pin(322))
    alloc.declare("NEXT_TOOL_CALL", 1, 5, [6], pin(323))
    alloc.declare("NEXT_THINKING_START", 1, 5, [6], pin(324))
    alloc.declare("NEXT_THINKING_END", 1, 5, [6], pin(325))
    alloc.declare("NEXT_IO_STATE_EMIT_BYTE", 1, 5, [6], pin(326))
    alloc.declare("NEXT_IO_STATE_EMIT_THINKING", 1, 5, [6], pin(327))

    # =========================================================================
    # I/O state detection (aliases)
    # =========================================================================
    alloc.declare("LAST_WAS_IO_STATE_EMIT_BYTE", 1, 1, [2], pin(462))  # aliases MEM_VAL_B1
    alloc.declare("LAST_WAS_IO_STATE_EMIT_THINKING", 1, 1, [2], pin(463))  # aliases MEM_VAL_B2

    # =========================================================================
    # Misc relay flags
    # =========================================================================
    alloc.declare("CMP_GROUP", 1, 5, [6], pin(305))
    alloc.declare("OP_LI_RELAY", 1, 7, [8], pin(465))  # alias
    alloc.declare("OP_LC_RELAY", 1, 7, [8], pin(466))  # alias
    alloc.declare("PSH_AT_SP", 1, 7, [8], pin(467))  # alias
    alloc.declare("MEM_EXEC", 1, 7, [14], pin(468))  # alias

    return alloc


# =============================================================================
# Compatibility with old allocator interface
# =============================================================================

def get_standard_positions() -> Dict[str, int]:
    """Get dictionary of dimension name -> start position."""
    alloc = create_standard_allocator(pinned=True)
    return alloc.allocate()
