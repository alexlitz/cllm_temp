"""
Dimension Registry & Layer Contract Tracking for the Autoregressive Neural VM.

Tracks all d_model dimension allocations, per-layer read/write contracts,
and validates data-flow invariants (no unintended overlaps, write-before-read,
double-write detection).

Also provides static weight inspection: auto-derive read/write contracts from
actual weight matrices after set_vm_weights() runs, and compare against
manually-declared contracts.

Usage:
    python3 -m neural_vm.dim_registry    # print dim map + validate + weight inspection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DimSlot:
    """A named allocation within the d_model embedding dimensions."""
    name: str
    start: int
    size: int
    desc: str

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def range(self) -> range:
        return range(self.start, self.end)

    def overlaps(self, other: 'DimSlot') -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class LayerIO:
    """Declares which dim slots a layer reads from and writes to."""
    layer: str          # e.g. "embed", "L0_attn", "L0_ffn", "L3_attn"
    reads: List[str]    # slot names read
    writes: List[str]   # slot names written
    notes: str = ""
    additive_writes: List[str] = field(default_factory=list)  # intentional multi-writer slots


class DimRegistry:
    """Registry of all dimension allocations within d_model."""

    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        self.slots: Dict[str, DimSlot] = {}

    def alloc(self, name: str, start: int, size: int, desc: str) -> DimSlot:
        """Register a dimension allocation. Returns the DimSlot."""
        if name in self.slots:
            raise ValueError(f"Duplicate slot name: {name}")
        if start < 0 or start + size > self.d_model:
            raise ValueError(f"Slot {name} [{start}, {start+size}) out of bounds [0, {self.d_model})")
        slot = DimSlot(name, start, size, desc)
        self.slots[name] = slot
        return slot

    def check_overlaps(self) -> List[str]:
        """Return error messages for any overlapping slots."""
        errors = []
        names = sorted(self.slots.keys(), key=lambda n: self.slots[n].start)
        for i, a_name in enumerate(names):
            for b_name in names[i+1:]:
                a, b = self.slots[a_name], self.slots[b_name]
                if a.overlaps(b):
                    errors.append(
                        f"OVERLAP: {a.name}[{a.start}:{a.end}) and "
                        f"{b.name}[{b.start}:{b.end})"
                    )
        return errors

    def free_ranges(self) -> List[Tuple[int, int]]:
        """Return list of (start, end) for unallocated dim ranges."""
        used = set()
        for slot in self.slots.values():
            used.update(slot.range)
        free = []
        start = None
        for d in range(self.d_model):
            if d not in used:
                if start is None:
                    start = d
            else:
                if start is not None:
                    free.append((start, d))
                    start = None
        if start is not None:
            free.append((start, self.d_model))
        return free

    def report(self) -> str:
        """Human-readable dim map."""
        lines = [f"Dimension Map (d_model={self.d_model})", "=" * 60]

        sorted_slots = sorted(self.slots.values(), key=lambda s: s.start)
        for slot in sorted_slots:
            if slot.size == 1:
                lines.append(f"  Dim {slot.start:3d}       : {slot.name:20s} — {slot.desc}")
            else:
                lines.append(
                    f"  Dims {slot.start:3d}-{slot.end-1:<3d}  : "
                    f"{slot.name:20s} ({slot.size:2d}) — {slot.desc}"
                )

        free = self.free_ranges()
        if free:
            lines.append("")
            lines.append("Free ranges:")
            for s, e in free:
                lines.append(f"  [{s}, {e})  ({e - s} dims)")

        used = sum(s.size for s in self.slots.values())
        lines.append("")
        lines.append(f"Used: {used}/{self.d_model} dims, Free: {self.d_model - used}")
        return "\n".join(lines)

    def resolve_names(self, patterns: List[str]) -> List[str]:
        """Resolve slot name patterns (supports trailing * wildcard)."""
        result = []
        for pat in patterns:
            if pat.endswith("*"):
                prefix = pat[:-1]
                matched = [n for n in self.slots if n.startswith(prefix)]
                if not matched:
                    result.append(pat)  # keep unresolved for error reporting
                else:
                    result.extend(matched)
            else:
                result.append(pat)
        return result


class ContractValidator:
    """Validates layer contracts against a DimRegistry."""

    def __init__(self, registry: DimRegistry, layers: List[LayerIO]):
        self.registry = registry
        self.layers = layers

    def validate(self) -> List[str]:
        """Run all validation checks. Returns list of error/warning strings."""
        errors = []
        errors.extend(self._check_overlaps())
        errors.extend(self._check_slot_refs())
        errors.extend(self._check_write_before_read())
        errors.extend(self._check_double_writes())
        return errors

    def _check_overlaps(self) -> List[str]:
        return self.registry.check_overlaps()

    def _check_slot_refs(self) -> List[str]:
        """Check that all referenced slot names exist in registry."""
        errors = []
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            for name in resolved_reads + resolved_writes + resolved_additive:
                if name not in self.registry.slots:
                    errors.append(f"UNKNOWN SLOT: {lio.layer} references '{name}'")
        return errors

    def _check_write_before_read(self) -> List[str]:
        """Check that every read has a prior write."""
        errors = []
        written: Set[str] = set()
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            for name in resolved_reads:
                if name not in self.registry.slots:
                    continue  # already caught by _check_slot_refs
                if name not in written:
                    errors.append(
                        f"READ-BEFORE-WRITE: {lio.layer} reads '{name}' "
                        f"but no prior layer writes it"
                    )
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            written.update(resolved_writes)
            written.update(resolved_additive)
        return errors

    def _check_double_writes(self) -> List[str]:
        """Warn when multiple layers write the same slot (unless marked additive)."""
        warnings = []
        writers: Dict[str, List[str]] = {}  # slot_name -> [layer_names]
        additive_slots: Set[str] = set()

        for lio in self.layers:
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            for name in resolved_writes:
                if name not in self.registry.slots:
                    continue
                writers.setdefault(name, []).append(lio.layer)
            for name in resolved_additive:
                if name not in self.registry.slots:
                    continue
                writers.setdefault(name, []).append(lio.layer)
                additive_slots.add(name)

        for name, layer_list in writers.items():
            if len(layer_list) > 1 and name not in additive_slots:
                warnings.append(
                    f"DOUBLE-WRITE: slot '{name}' written by "
                    f"{', '.join(layer_list)}"
                )
        return warnings

    def dep_graph(self) -> str:
        """Text dependency visualization: which layers produce/consume each slot."""
        lines = ["Layer Dependency Graph", "=" * 60]

        # Collect all slot names that are read or written
        all_slots: Set[str] = set()
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            all_slots.update(resolved_reads)
            all_slots.update(resolved_writes)
            all_slots.update(resolved_additive)

        # For each slot, show writers -> readers
        slot_writers: Dict[str, List[str]] = {}
        slot_readers: Dict[str, List[str]] = {}
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            for name in resolved_reads:
                slot_readers.setdefault(name, []).append(lio.layer)
            for name in resolved_writes + resolved_additive:
                slot_writers.setdefault(name, []).append(lio.layer)

        for name in sorted(all_slots):
            if name not in self.registry.slots:
                continue
            w = slot_writers.get(name, ["(none)"])
            r = slot_readers.get(name, ["(none)"])
            lines.append(f"  {name}:")
            lines.append(f"    writers: {', '.join(w)}")
            lines.append(f"    readers: {', '.join(r)}")

        # Layer execution order summary
        lines.append("")
        lines.append("Execution Order:")
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            r_str = ", ".join(resolved_reads) if resolved_reads else "(none)"
            w_str = ", ".join(resolved_writes + resolved_additive) if (resolved_writes or resolved_additive) else "(none)"
            lines.append(f"  {lio.layer}:")
            if lio.notes:
                lines.append(f"    {lio.notes}")
            lines.append(f"    reads:  {r_str}")
            lines.append(f"    writes: {w_str}")

        return "\n".join(lines)


# ============================================================================
# Default registry matching current _BakeDim allocations
# ============================================================================

def build_default_registry() -> DimRegistry:
    """Build a DimRegistry matching the current _BakeDim allocations (d_model=512)."""
    reg = DimRegistry(d_model=512)

    # Marker identity flags (set by embedding)
    reg.alloc("MARK_PC",      0, 1, "PC register marker flag")
    reg.alloc("MARK_AX",      1, 1, "AX register marker flag")
    reg.alloc("MARK_SP",      2, 1, "SP register marker flag")
    reg.alloc("MARK_BP",      3, 1, "BP register marker flag")
    reg.alloc("MARK_MEM",     4, 1, "MEM marker flag")
    reg.alloc("MARK_SE",      5, 1, "STEP_END/DATA_END marker flag")
    reg.alloc("IS_BYTE",      6, 1, "Token is a byte value (0-255)")
    reg.alloc("IS_MARK",      7, 1, "Token is a marker")
    reg.alloc("CONST",        8, 1, "Constant 1.0 on all tokens")
    reg.alloc("MARK_CS",      9, 1, "CODE_START only marker")
    reg.alloc("MARK_SE_ONLY", 10, 1, "STEP_END only (not DATA_END)")
    reg.alloc("MARK_STACK0",  11, 1, "STACK0 marker flag")

    # Address byte nibbles (gathered by memory address layers)
    reg.alloc("ADDR_B0_LO",  12, 16, "One-hot addr byte 0 low nibble")
    reg.alloc("ADDR_B1_LO",  28, 16, "One-hot addr byte 1 low nibble")
    reg.alloc("ADDR_B2_LO",  44, 16, "One-hot addr byte 2 low nibble")

    # Layer 0 attention output: 8 threshold heads
    reg.alloc("H0",  60, 7, "L0 head 0: marker within dist 3.5")
    reg.alloc("H1",  67, 7, "L0 head 1: marker within dist 4.5")
    reg.alloc("H2",  74, 7, "L0 head 2: marker within dist 5.5")
    reg.alloc("H3",  81, 7, "L0 head 3: marker within dist 9.5")
    reg.alloc("H4",  88, 7, "L0 head 4: marker within dist 10.5")
    reg.alloc("H5",  95, 7, "L0 head 5: marker within dist 14.5")
    reg.alloc("H6", 102, 7, "L0 head 6: marker within dist 15.5")
    reg.alloc("H7", 109, 7, "L0 head 7: marker within dist 19.5")

    # Layer 1 attention output: fine thresholds + SE detect
    reg.alloc("L1H0", 116, 7, "L1 head 0: marker within dist 0.5")
    reg.alloc("L1H1", 123, 7, "L1 head 1: marker within dist 1.5")
    reg.alloc("L1H2", 130, 7, "L1 head 2: marker within dist 2.5")
    reg.alloc("HAS_SE", 137, 1, "STEP_END existence flag")

    # Byte index within register
    reg.alloc("BYTE_INDEX_0", 138, 1, "Byte index 0 flag")
    reg.alloc("BYTE_INDEX_1", 139, 1, "Byte index 1 flag")
    reg.alloc("BYTE_INDEX_2", 140, 1, "Byte index 2 flag")
    reg.alloc("BYTE_INDEX_3", 141, 1, "Byte index 3 flag")

    # Nibble encoding
    reg.alloc("EMBED_LO",  142, 16, "Embedding input low nibble (one-hot)")
    reg.alloc("EMBED_HI",  158, 16, "Embedding input high nibble (one-hot)")
    reg.alloc("OUTPUT_LO", 174, 16, "Output decoding low nibble (one-hot)")
    reg.alloc("OUTPUT_HI", 190, 16, "Output decoding high nibble (one-hot)")

    # Memory address key
    reg.alloc("ADDR_KEY", 206, 48, "One-hot address key for memory matching (3 nibbles x 16)")

    # NEXT_* transition flags
    reg.alloc("NEXT_PC",     254, 1, "Next token is PC register")
    reg.alloc("NEXT_AX",     255, 1, "Next token is AX register")
    reg.alloc("NEXT_SP",     256, 1, "Next token is SP register")
    reg.alloc("NEXT_BP",     257, 1, "Next token is BP register")
    reg.alloc("NEXT_STACK0", 258, 1, "Next token is STACK0 marker")
    reg.alloc("NEXT_MEM",    259, 1, "Next token is MEM marker")
    reg.alloc("NEXT_SE",     260, 1, "Next token is STEP_END")
    reg.alloc("NEXT_HALT",   261, 1, "Emit HALT instead of STEP_END")

    # Opcode one-hot flags (34 opcodes)
    reg.alloc("OPCODE_FLAGS", 262, 34, "One-hot opcode flags (LEA..GETCHAR)")

    # IO PUTCHAR flag
    reg.alloc("IO_IS_PUTCHAR", 296, 1, "OP_PUTCHAR detected this step (L6 FFN)")

    # ADJ implementation dimensions (SP + signed immediate)
    reg.alloc("SP_OLD_LO", 297, 8, "ADJ: old SP value low nibbles (4 bytes)")
    reg.alloc("SP_OLD_HI", 305, 8, "ADJ: old SP value high nibbles (4 bytes)")
    reg.alloc("ADJ_CARRY", 313, 2, "ADJ: multi-byte carry propagation")

    # Reserved (remaining space for ENT/LEV)
    reg.alloc("RESERVED_315_327", 315, 13, "Reserved (ENT/LEV staging)")

    # AX carry-forward staging
    reg.alloc("AX_CARRY_LO", 328, 16, "Carried-forward AX lo nibble")
    reg.alloc("AX_CARRY_HI", 344, 16, "Carried-forward AX hi nibble")

    # ALU result staging
    reg.alloc("ALU_LO", 360, 16, "ALU result lo nibble")
    reg.alloc("ALU_HI", 376, 16, "ALU result hi nibble")

    # Carry / comparison
    reg.alloc("CARRY", 392, 4, "Inter-byte carry for ADD/SUB/MUL")
    reg.alloc("CMP",   396, 4, "Comparison cascade: LT, EQ, GT, ZERO")

    # Reserved (formerly PC_BIT, available for future use)
    reg.alloc("RESERVED_400_415", 400, 16, "Reserved (future PC binary encoding/IO)")

    # MUL/DIV staging
    reg.alloc("MUL_ACCUM",   416, 16, "Multiplication accumulator")
    reg.alloc("DIV_STAGING", 432, 16, "Division quotient/remainder")

    # Immediate staging
    reg.alloc("IMM_STAGING", 448, 16, "Fetched immediate bytes")

    # CS distance thermometer
    reg.alloc("CS_DIST_THERMO", 464, 16, "Thermometer-coded distance from CODE_START")

    # General temporaries
    reg.alloc("TEMP", 480, 32, "General temporaries / reserved")

    return reg


def build_default_contracts(registry: DimRegistry) -> List[LayerIO]:
    """Build layer contracts matching current set_vm_weights data flow (16 layers)."""
    return [
        LayerIO(
            layer="embed",
            reads=[],
            writes=[
                "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
                "MARK_SE", "IS_BYTE", "IS_MARK", "CONST", "MARK_CS",
                "MARK_SE_ONLY", "MARK_STACK0", "EMBED_LO", "EMBED_HI",
            ],
            notes="Embedding layer sets marker flags + nibble encodings",
        ),
        LayerIO(
            layer="L0_attn",
            reads=["CONST", "IS_MARK", "MARK_*"],
            writes=["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
            notes="Threshold attention: 8 heads for 39-token step structure",
        ),
        LayerIO(
            layer="L0_ffn",
            reads=["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
            writes=["NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
                    "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"],
            notes="Detect register transitions for 39-token step",
        ),
        LayerIO(
            layer="L1_attn",
            reads=["CONST", "IS_MARK", "MARK_*"],
            writes=["L1H0", "L1H1", "L1H2", "HAS_SE"],
            notes="Fine thresholds 0.5/1.5/2.5 + global STEP_END detection",
        ),
        LayerIO(
            layer="L1_ffn",
            reads=["HAS_SE", "NEXT_SE"],
            writes=["NEXT_HALT"],
            notes="HALT override (instruction decode moved to L4-L5)",
        ),
        # Layers 2-6: instruction fetch + opcode decode + routing
        LayerIO(
            layer="L6_ffn_io",
            reads=["OPCODE_FLAGS", "MARK_AX", "AX_CARRY_LO", "AX_CARRY_HI"],
            writes=["IO_IS_PUTCHAR"],
            additive_writes=["OUTPUT_LO", "OUTPUT_HI"],
            notes="PUTCHAR: detect opcode, route AX_CARRY → OUTPUT (autoregressive)",
        ),
        # Layers 7-14: reserved for STACK0, ALU, MEM ops
        # GETCHAR: runner-side IO (not in weights, see run_vm.py)
        LayerIO(
            layer="L15_attn",
            reads=["ADDR_KEY", "EMBED_LO", "EMBED_HI"],
            writes=[],
            additive_writes=["OUTPUT_LO", "OUTPUT_HI"],
            notes="Memory lookup via softmax1 + ALiBi (ZFOD semantics)",
        ),
        LayerIO(
            layer="output_head",
            reads=["OUTPUT_LO", "OUTPUT_HI", "NEXT_PC", "NEXT_AX",
                   "NEXT_SP", "NEXT_BP", "NEXT_STACK0", "NEXT_MEM",
                   "NEXT_SE", "NEXT_HALT"],
            writes=[],
            notes="Nibble decoding + transition token selection",
        ),
    ]


def validate_default() -> Tuple[DimRegistry, List[str]]:
    """Build and validate the default registry + contracts. Returns (registry, errors)."""
    reg = build_default_registry()
    contracts = build_default_contracts(reg)
    validator = ContractValidator(reg, contracts)
    errors = validator.validate()
    return reg, errors


# ============================================================================
# Static Weight Inspection — auto-derive contracts from actual weight matrices
# ============================================================================

def _nonzero_dims(weight, axis: int) -> Set[int]:
    """Find dims with any non-zero entry along the given axis.

    Args:
        weight: 2D tensor (dense or sparse COO)
        axis: 0 = find non-zero columns (input dims), 1 = find non-zero rows (output dims)

    For axis=0: returns set of column indices where any row has non-zero.
        i.e. dims d where weight[:, d] has any non-zero → "reads from d"
    For axis=1: returns set of row indices where any column has non-zero.
        i.e. dims d where weight[d, :] has any non-zero → "writes to d"
    """
    import torch
    if weight.is_sparse:
        indices = weight.coalesce().indices()
        if indices.numel() == 0:
            return set()
        # indices[0] = row indices, indices[1] = column indices
        # axis=0 → want columns → indices[1]; axis=1 → want rows → indices[0]
        if axis == 0:
            return set(indices[1].tolist())
        else:
            return set(indices[0].tolist())
    else:
        # W shape [M, N]. sum(dim=0) → [N] (column sums), sum(dim=1) → [M] (row sums)
        # axis=0 (columns/inputs): sum over rows → dim=0 → shape [N] → column indices
        # axis=1 (rows/outputs): sum over cols → dim=1 → shape [M] → row indices
        summed = weight.abs().sum(dim=axis)
        return set(torch.nonzero(summed).squeeze(-1).tolist())


def _nonzero_bias_dims(bias) -> Set[int]:
    """Find dims where bias vector has non-zero entries."""
    import torch
    if bias is None:
        return set()
    if bias.is_sparse:
        indices = bias.coalesce().indices()
        if indices.numel() == 0:
            return set()
        return set(indices[0].tolist())
    else:
        return set(torch.nonzero(bias.abs()).squeeze(-1).tolist())


def _build_dim_to_slot(registry: DimRegistry) -> Dict[int, str]:
    """Map each dim index to its registry slot name.

    Dims not covered by any slot are omitted (reported as unregistered).
    """
    dim_to_slot = {}
    for name, slot in registry.slots.items():
        for d in slot.range:
            dim_to_slot[d] = name
    return dim_to_slot


def _build_setdim_lookup() -> Dict[int, str]:
    """Build a fallback lookup from _SetDim constants for unregistered dims.

    Returns a mapping from dim index → _SetDim attribute name (e.g. 455 → "MEM_STORE").
    Multi-dim ranges (16-wide nibble arrays etc.) get names like "CLEAN_EMBED_LO+3".
    """
    from .vm_step import _SetDim

    # Collect all integer class attributes
    attrs = {}
    for name in dir(_SetDim):
        if name.startswith('_') or name.startswith('NUM_') or name == 'MARKS':
            continue
        val = getattr(_SetDim, name)
        if isinstance(val, int) and 0 <= val < 512:
            attrs[name] = val

    # Sort by value to detect ranges — attrs with same value are aliases
    by_val = sorted(attrs.items(), key=lambda x: x[1])

    # Known multi-dim ranges (name, start, size)
    RANGES = [
        ("ADDR_B0_LO", 12, 16), ("ADDR_B1_LO", 28, 16), ("ADDR_B2_LO", 44, 16),
        ("EMBED_LO", 142, 16), ("EMBED_HI", 158, 16),
        ("OUTPUT_LO", 174, 16), ("OUTPUT_HI", 190, 16),
        ("ADDR_KEY", 206, 48),
        ("OPCODE_FLAGS", 262, 34),
        ("CLEAN_EMBED_LO", 306, 16), ("CLEAN_EMBED_HI", 400, 16),
        ("AX_CARRY_LO", 328, 16), ("AX_CARRY_HI", 344, 16),
        ("ALU_LO", 360, 16), ("ALU_HI", 376, 16),
        ("CARRY", 392, 4), ("CMP", 396, 4),
        ("MUL_ACCUM", 416, 16), ("DIV_STAGING", 432, 16),
        ("TEMP", 480, 32),
        ("L1H4", 297, 7), ("L2H0", 448, 7),
    ]
    # Also the L0/L1 head ranges
    for h in range(8):
        base = 60 + h * 7
        RANGES.append((f"H{h}", base, 7))
    for h in range(3):
        base = 116 + h * 7
        RANGES.append((f"L1H{h}", base, 7))

    lookup = {}
    for rname, rstart, rsize in RANGES:
        for offset in range(rsize):
            d = rstart + offset
            if offset == 0:
                lookup[d] = rname
            else:
                lookup[d] = f"{rname}+{offset}"

    # Add single-dim constants
    for name, val in by_val:
        if val not in lookup:
            lookup[val] = name

    return lookup


def _dim_name(d: int, dim_to_slot: Dict[int, str], setdim_lookup: Dict[int, str]) -> str:
    """Get a human-readable name for dimension d.

    Prefers registry slot name, falls back to _SetDim name, then "unregistered[d]".
    """
    if d in dim_to_slot:
        return dim_to_slot[d]
    if d in setdim_lookup:
        return f"_{setdim_lookup[d]}"  # prefix with _ to indicate non-registry
    return f"unregistered[{d}]"


def _dims_to_slot_names(dims: Set[int], dim_to_slot: Dict[int, str]) -> Tuple[Set[str], Set[int]]:
    """Map dim indices to registry slot names.

    Returns (slot_names, unregistered_dims).
    """
    slot_names = set()
    unregistered = set()
    for d in dims:
        if d in dim_to_slot:
            slot_names.add(dim_to_slot[d])
        else:
            unregistered.add(d)
    return slot_names, unregistered


@dataclass
class LayerWeightInfo:
    """Auto-derived weight info for one layer component (FFN or attn head)."""
    layer: str              # e.g. "L0_ffn", "L0_attn", "L0_attn_h3"
    read_dims: Set[int]     # raw dim indices read
    write_dims: Set[int]    # raw dim indices written
    bias_dims: Set[int] = field(default_factory=set)  # dims activated by bias alone
    active_hidden: int = 0  # number of active hidden units (FFN only)
    total_hidden: int = 0   # total hidden units (FFN only)


def extract_weight_info(model) -> List[LayerWeightInfo]:
    """Extract per-layer read/write dim sets from actual weight matrices.

    Inspects W_up, W_gate, W_down for FFNs and W_q, W_k, W_v, W_o for
    attention layers. Works with both dense and sparse weights.

    Returns a list of LayerWeightInfo, two per block (attn + ffn), plus
    per-head detail for attention.
    """
    import torch
    infos = []

    for i, block in enumerate(model.blocks):
        # --- Attention ---
        attn = block.attn
        attn_read = set()
        attn_write = set()
        for w_name in ('W_q', 'W_k', 'W_v'):
            W = getattr(attn, w_name)
            attn_read |= _nonzero_dims(W, axis=0)
        attn_write = _nonzero_dims(attn.W_o, axis=1)

        infos.append(LayerWeightInfo(
            layer=f"L{i}_attn",
            read_dims=attn_read,
            write_dims=attn_write,
        ))

        # Per-head detail
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        for h in range(num_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim
            h_read = set()
            h_write = set()
            for w_name in ('W_q', 'W_k', 'W_v'):
                W = getattr(attn, w_name)
                if W.is_sparse:
                    idx = W.coalesce().indices()
                    if idx.numel() > 0:
                        mask = (idx[0] >= h_start) & (idx[0] < h_end)
                        h_read |= set(idx[1][mask].tolist())
                else:
                    h_read |= _nonzero_dims(W[h_start:h_end], axis=0)
            # W_o: columns h_start:h_end → which output dims
            W_o = attn.W_o
            if W_o.is_sparse:
                idx = W_o.coalesce().indices()
                if idx.numel() > 0:
                    mask = (idx[1] >= h_start) & (idx[1] < h_end)
                    h_write |= set(idx[0][mask].tolist())
            else:
                h_write = _nonzero_dims(W_o[:, h_start:h_end], axis=1)

            if h_read or h_write:
                infos.append(LayerWeightInfo(
                    layer=f"L{i}_attn_h{h}",
                    read_dims=h_read,
                    write_dims=h_write,
                ))

        # --- FFN ---
        ffn = block.ffn
        ffn_read = _nonzero_dims(ffn.W_up, axis=0) | _nonzero_dims(ffn.W_gate, axis=0)
        ffn_write = _nonzero_dims(ffn.W_down, axis=1)
        # Bias: find hidden units with non-zero b_up or b_gate (constant activations)
        bias_hidden = _nonzero_bias_dims(ffn.b_up) | _nonzero_bias_dims(ffn.b_gate)
        # Map bias-activated hidden units to which output dims they can affect
        # via W_down[d, h] — if W_down[d, h] != 0 and h has non-zero bias
        bias_output_dims = set()
        if bias_hidden:
            W_down = ffn.W_down
            if W_down.is_sparse:
                idx = W_down.coalesce().indices()
                vals = W_down.coalesce().values()
                if idx.numel() > 0:
                    for h in bias_hidden:
                        mask = (idx[1] == h) & (vals != 0)
                        bias_output_dims |= set(idx[0][mask].tolist())
            else:
                for h in bias_hidden:
                    if h < W_down.shape[1]:
                        col = W_down[:, h]
                        bias_output_dims |= set(torch.nonzero(col.abs()).squeeze(-1).tolist())

        # Active hidden units: rows of W_up or W_gate with any non-zero
        active_up = _nonzero_dims(ffn.W_up, axis=1)
        active_gate = _nonzero_dims(ffn.W_gate, axis=1)
        active_hidden = len(active_up | active_gate)
        total_hidden = ffn.hidden_dim

        infos.append(LayerWeightInfo(
            layer=f"L{i}_ffn",
            read_dims=ffn_read,
            write_dims=ffn_write,
            bias_dims=bias_output_dims,  # d_model dims affected by bias
            active_hidden=active_hidden,
            total_hidden=total_hidden,
        ))

    return infos


def extract_model_contracts(model, registry: DimRegistry) -> List[LayerIO]:
    """Auto-derive per-layer read/write contracts from actual weight matrices.

    Maps non-zero dimension indices back to DimRegistry slot names.
    Returns one LayerIO per block component (attn + ffn per layer).
    """
    dim_to_slot = _build_dim_to_slot(registry)
    infos = extract_weight_info(model)

    contracts = []
    for info in infos:
        # Skip per-head detail for contract comparison
        if '_h' in info.layer:
            continue
        read_slots, unregistered_reads = _dims_to_slot_names(info.read_dims, dim_to_slot)
        write_slots, unregistered_writes = _dims_to_slot_names(info.write_dims, dim_to_slot)

        notes_parts = []
        if unregistered_reads:
            notes_parts.append(f"unregistered reads: {sorted(unregistered_reads)}")
        if unregistered_writes:
            notes_parts.append(f"unregistered writes: {sorted(unregistered_writes)}")
        if info.active_hidden:
            notes_parts.append(f"{info.active_hidden}/{info.total_hidden} active hidden units")

        contracts.append(LayerIO(
            layer=info.layer,
            reads=sorted(read_slots),
            writes=sorted(write_slots),
            notes="; ".join(notes_parts),
        ))

    return contracts


def compare_contracts(
    declared: List[LayerIO],
    derived: List[LayerIO],
    registry: DimRegistry,
) -> List[str]:
    """Compare manually-declared contracts against auto-derived ones.

    Reports:
    - Undeclared reads: slots read by weights but not in declared contract
    - Undeclared writes: slots written by weights but not in declared contract
    - Phantom reads: declared but not actually read (stale contract)
    - Phantom writes: declared but not actually written
    """
    declared_map = {lio.layer: lio for lio in declared}
    derived_map = {lio.layer: lio for lio in derived}

    issues = []

    # Check derived layers not covered by any declaration
    for layer_name, derived_lio in sorted(derived_map.items()):
        if layer_name not in declared_map:
            d_reads = set(derived_lio.reads)
            d_writes = set(derived_lio.writes)
            if d_reads or d_writes:
                issues.append(
                    f"UNDECLARED LAYER: {layer_name} "
                    f"reads={sorted(d_reads)}, writes={sorted(d_writes)}"
                )
            continue

        decl = declared_map[layer_name]
        decl_reads = set(registry.resolve_names(decl.reads))
        decl_writes = set(registry.resolve_names(decl.writes))
        decl_additive = set(registry.resolve_names(decl.additive_writes))
        decl_all_writes = decl_writes | decl_additive

        derived_reads = set(derived_lio.reads)
        derived_writes = set(derived_lio.writes)

        undeclared_reads = derived_reads - decl_reads
        undeclared_writes = derived_writes - decl_all_writes
        phantom_reads = decl_reads - derived_reads
        phantom_writes = decl_all_writes - derived_writes

        for slot in sorted(undeclared_reads):
            issues.append(f"UNDECLARED READ: {layer_name} reads '{slot}' (not in contract)")
        for slot in sorted(undeclared_writes):
            issues.append(f"UNDECLARED WRITE: {layer_name} writes '{slot}' (not in contract)")
        for slot in sorted(phantom_reads):
            issues.append(f"PHANTOM READ: {layer_name} declares read '{slot}' but weights don't touch it")
        for slot in sorted(phantom_writes):
            issues.append(f"PHANTOM WRITE: {layer_name} declares write '{slot}' but weights don't touch it")

    return issues


def ffn_summary(model, registry: DimRegistry, per_head: bool = True) -> str:
    """Human-readable per-layer FFN/attn input/output report.

    For each layer, prints:
      L{i} attn: (reads/writes with slot names)
        Per head detail if per_head=True
      L{i} FFN: {active}/{total} active hidden units
        reads:  SLOT_A, SLOT_B, ...
        writes: SLOT_C, SLOT_D, ...
    """
    dim_to_slot = _build_dim_to_slot(registry)
    setdim_lookup = _build_setdim_lookup()
    infos = extract_weight_info(model)

    lines = ["Weight Inspection Summary", "=" * 70]

    def _format_dims(dims: Set[int]) -> str:
        """Format dim set as slot-grouped names.

        For fully-used registry slots, shows "SLOT_NAME".
        For partially-used slots, shows individual _SetDim names when available,
        e.g. "MEM_STORE, MEM_ADDR_SRC" instead of "IMM_STAGING[2/16]".
        """
        if not dims:
            return "(none)"
        # Group by registry slot
        by_slot: Dict[str, List[int]] = {}
        unregistered: List[int] = []
        for d in sorted(dims):
            if d in dim_to_slot:
                by_slot.setdefault(dim_to_slot[d], []).append(d)
            else:
                unregistered.append(d)

        parts = []
        for base in sorted(by_slot.keys(), key=lambda b: min(by_slot[b])):
            dim_list = by_slot[base]
            slot = registry.slots.get(base)
            if slot and len(dim_list) == slot.size:
                # Full slot used
                parts.append(base)
            elif slot and len(dim_list) < slot.size:
                # Partial slot — show _SetDim names if they exist and are
                # more specific than the slot name
                setdim_names = []
                for d in dim_list:
                    sd = setdim_lookup.get(d)
                    if sd and sd != base:
                        setdim_names.append(sd)
                    else:
                        setdim_names.append(f"{base}[{d - slot.start}]")
                # If all have _SetDim names, show them individually (more useful)
                if len(setdim_names) <= 6:
                    parts.extend(setdim_names)
                else:
                    parts.append(f"{base}[{len(dim_list)}/{slot.size}]")
            else:
                parts.append(base)

        # Unregistered dims — show _SetDim names
        for d in unregistered:
            sd = setdim_lookup.get(d)
            if sd:
                parts.append(f"_{sd}")
            else:
                parts.append(f"dim[{d}]")

        return ", ".join(parts)

    for info in infos:
        is_head = '_h' in info.layer
        indent = "    " if is_head else "  "

        if is_head:
            if not info.read_dims and not info.write_dims:
                continue
            lines.append(f"{indent}{info.layer}:")
            lines.append(f"{indent}  reads:  {_format_dims(info.read_dims)}")
            lines.append(f"{indent}  writes: {_format_dims(info.write_dims)}")
        elif info.layer.endswith("_attn"):
            lines.append("")
            lines.append(f"{indent}{info.layer}:")
            lines.append(f"{indent}  reads:  {_format_dims(info.read_dims)}")
            lines.append(f"{indent}  writes: {_format_dims(info.write_dims)}")
        else:
            # FFN
            hidden_str = ""
            if info.total_hidden:
                hidden_str = f" ({info.active_hidden}/{info.total_hidden} active hidden units)"
            lines.append(f"{indent}{info.layer}:{hidden_str}")
            lines.append(f"{indent}  reads:  {_format_dims(info.read_dims)}")
            lines.append(f"{indent}  writes: {_format_dims(info.write_dims)}")
            if info.bias_dims:
                lines.append(f"{indent}  bias→out:  {_format_dims(info.bias_dims)}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    reg = build_default_registry()
    contracts = build_default_contracts(reg)
    validator = ContractValidator(reg, contracts)

    print(reg.report())
    print()

    errors = validator.validate()
    if errors:
        print("Validation Issues:")
        for e in errors:
            print(f"  {e}")
    else:
        print("All contracts valid.")
    print()

    print(validator.dep_graph())
    print()

    # --- Weight inspection ---
    print("=" * 70)
    print("WEIGHT INSPECTION (auto-derived from actual weight matrices)")
    print("=" * 70)
    print()

    import torch
    from .vm_step import AutoregressiveVM, set_vm_weights

    print("Building model and setting weights...")
    model = AutoregressiveVM()
    set_vm_weights(model)
    print()

    # Per-layer summary
    print(ffn_summary(model, reg, per_head=True))
    print()

    # Compare against declared contracts
    derived = extract_model_contracts(model, reg)
    issues = compare_contracts(contracts, derived, reg)
    if issues:
        print("Contract Comparison (declared vs auto-derived):")
        print("-" * 50)
        for issue in issues:
            print(f"  {issue}")
    else:
        print("All declared contracts match auto-derived contracts.")
    print()
