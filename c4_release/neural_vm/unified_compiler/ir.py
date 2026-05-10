"""
Compiler Intermediate Representation (IR) for Neural VM.

Defines data structures for representing weight compilation:
- DimensionAlloc: Embedding dimension allocations
- AttentionOp: Attention head operations
- FFNOp: FFN unit operations
- LayerSpec: Complete layer specification
- CompilerIR: Full program IR
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Callable
from enum import Enum, auto
import json


# =============================================================================
# Enums for operation types
# =============================================================================

class AttentionOpType(Enum):
    """Types of attention head operations."""
    THRESHOLD = auto()      # Distance-based threshold detection
    RELAY = auto()          # Copy values between positions
    FETCH = auto()          # Memory address lookup
    GATHER = auto()         # Collect operands from multiple sources
    DETECT = auto()         # Pattern detection (HAS_SE, etc.)
    CARRY_FORWARD = auto()  # Register carry-forward (Q: marker, K: L1H1-L1H0 pattern)
    OP_FLAG_RELAY = auto()  # Relay OP_* flags from CODE to markers


class FFNOpType(Enum):
    """Types of FFN unit operations."""
    SWIGLU_GATE = auto()    # Threshold-gated output
    ALU_LOOKUP = auto()     # 256-entry lookup table
    NIBBLE_ROTATE = auto()  # Increment nibbles
    OPCODE_DECODE = auto()  # Opcode → flag conversion
    THRESHOLD_FLAG = auto() # Threshold → binary flag
    CLEAR_DIMS = auto()     # Zero out dimensions
    COPY = auto()           # Copy nibbles under gate
    CONDITIONAL_OUTPUT = auto()  # Output fixed nibble when conditions met
    GATED_NIBBLE_COPY = auto()   # Copy nibble k to output when gate fires
    PC_INCREMENT = auto()   # PC += INSTR_WIDTH with carry


# =============================================================================
# Dimension Allocation
# =============================================================================

@dataclass
class DimensionAlloc:
    """A range of dimensions allocated for a semantic purpose.

    Attributes:
        name: Semantic name (e.g., "MARK_PC", "ALU_LO")
        start: Starting dimension index
        count: Number of dimensions
        layer_written: Which layer writes this (-1 for embedding)
        layers_read: Which layers read this
    """
    name: str
    start: int
    count: int
    layer_written: int = -1
    layers_read: List[int] = field(default_factory=list)

    @property
    def end(self) -> int:
        """Exclusive end index."""
        return self.start + self.count

    @property
    def range(self) -> range:
        """Dimension range."""
        return range(self.start, self.end)

    def overlaps(self, other: 'DimensionAlloc') -> bool:
        """Check if this allocation overlaps with another."""
        return not (self.end <= other.start or other.end <= self.start)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'start': self.start,
            'count': self.count,
            'layer_written': self.layer_written,
            'layers_read': self.layers_read,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DimensionAlloc':
        """Deserialize from dictionary."""
        return cls(**d)


# =============================================================================
# Attention Operations
# =============================================================================

@dataclass
class AttentionOp:
    """An attention head operation.

    Represents what an attention head does: which dimensions it queries,
    which keys it matches, what values it reads, and where it writes.

    Attributes:
        op_type: Type of operation (threshold, relay, fetch, etc.)
        head_idx: Which head in the attention block
        q_dims: Query dimension indices
        k_dims: Key dimension indices
        v_dims: Value dimension indices (what to read)
        o_dims: Output dimension indices (where to write)
        params: Operation-specific parameters
    """
    op_type: AttentionOpType
    head_idx: int
    q_dims: List[int] = field(default_factory=list)
    k_dims: List[int] = field(default_factory=list)
    v_dims: List[int] = field(default_factory=list)
    o_dims: List[int] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'op_type': self.op_type.name,
            'head_idx': self.head_idx,
            'q_dims': self.q_dims,
            'k_dims': self.k_dims,
            'v_dims': self.v_dims,
            'o_dims': self.o_dims,
            'params': self.params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AttentionOp':
        """Deserialize from dictionary."""
        d = d.copy()
        d['op_type'] = AttentionOpType[d['op_type']]
        return cls(**d)


# =============================================================================
# FFN Operations
# =============================================================================

@dataclass
class FFNOp:
    """An FFN unit operation.

    Represents what FFN units do: gate conditions, inputs, and outputs.

    Attributes:
        op_type: Type of operation (swiglu_gate, alu_lookup, etc.)
        unit_start: Starting unit index
        unit_count: Number of units used
        gate_dims: Dimensions that gate this operation
        input_dims: Input dimension indices
        output_dims: Output dimension indices
        params: Operation-specific parameters
    """
    op_type: FFNOpType
    unit_start: int
    unit_count: int
    gate_dims: List[int] = field(default_factory=list)
    input_dims: List[int] = field(default_factory=list)
    output_dims: List[int] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'op_type': self.op_type.name,
            'unit_start': self.unit_start,
            'unit_count': self.unit_count,
            'gate_dims': self.gate_dims,
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'params': self.params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FFNOp':
        """Deserialize from dictionary."""
        d = d.copy()
        d['op_type'] = FFNOpType[d['op_type']]
        return cls(**d)


# =============================================================================
# Layer Specification
# =============================================================================

@dataclass
class LayerSpec:
    """Complete specification for one transformer layer.

    Attributes:
        layer_idx: Which layer (0-16)
        attention_ops: List of attention head operations
        ffn_ops: List of FFN unit operations
        reads: Set of dimension names read by this layer
        writes: Set of dimension names written by this layer
    """
    layer_idx: int
    attention_ops: List[AttentionOp] = field(default_factory=list)
    ffn_ops: List[FFNOp] = field(default_factory=list)
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)

    def add_attention_op(self, op: AttentionOp) -> None:
        """Add an attention operation."""
        self.attention_ops.append(op)

    def add_ffn_op(self, op: FFNOp) -> None:
        """Add an FFN operation."""
        self.ffn_ops.append(op)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'layer_idx': self.layer_idx,
            'attention_ops': [op.to_dict() for op in self.attention_ops],
            'ffn_ops': [op.to_dict() for op in self.ffn_ops],
            'reads': list(self.reads),
            'writes': list(self.writes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LayerSpec':
        """Deserialize from dictionary."""
        return cls(
            layer_idx=d['layer_idx'],
            attention_ops=[AttentionOp.from_dict(op) for op in d['attention_ops']],
            ffn_ops=[FFNOp.from_dict(op) for op in d['ffn_ops']],
            reads=set(d['reads']),
            writes=set(d['writes']),
        )


# =============================================================================
# Full Compiler IR
# =============================================================================

@dataclass
class CompilerIR:
    """Full intermediate representation for the compiler.

    Contains all information needed to generate weights:
    - Dimension allocations (the "register file")
    - Layer specifications (the "instructions")
    - Metadata about the compilation

    Attributes:
        dimensions: Map of dimension name → allocation
        layers: List of layer specifications
        d_model: Model dimension (default 512)
        n_layers: Number of layers (default 17)
        n_heads: Number of attention heads (default 8)
        metadata: Additional compilation metadata
    """
    dimensions: Dict[str, DimensionAlloc] = field(default_factory=dict)
    layers: List[LayerSpec] = field(default_factory=list)
    d_model: int = 512
    n_layers: int = 17
    n_heads: int = 8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize layers if empty."""
        if not self.layers:
            self.layers = [LayerSpec(layer_idx=i) for i in range(self.n_layers)]

    def get_dim(self, name: str) -> DimensionAlloc:
        """Get a dimension allocation by name."""
        if name not in self.dimensions:
            raise KeyError(f"Dimension '{name}' not allocated")
        return self.dimensions[name]

    def get_dim_start(self, name: str) -> int:
        """Get the starting index of a dimension."""
        return self.get_dim(name).start

    def get_dim_range(self, name: str) -> range:
        """Get the range of a dimension."""
        return self.get_dim(name).range

    def allocate(self, name: str, start: int, count: int,
                 layer_written: int = -1) -> DimensionAlloc:
        """Allocate a dimension range."""
        alloc = DimensionAlloc(name, start, count, layer_written)

        # Check for collisions
        for existing in self.dimensions.values():
            if alloc.overlaps(existing):
                raise ValueError(
                    f"Dimension collision: {name} [{start}:{start+count}] "
                    f"overlaps {existing.name} [{existing.start}:{existing.end}]"
                )

        self.dimensions[name] = alloc
        return alloc

    def get_layer(self, idx: int) -> LayerSpec:
        """Get a layer specification."""
        return self.layers[idx]

    def check_read_before_write(self) -> List[str]:
        """Check for dimensions read before they're written.

        Returns list of error messages.
        """
        errors = []
        written_by_layer = {}  # dim_name → first layer that writes

        for layer in self.layers:
            # Check reads
            for dim_name in layer.reads:
                if dim_name not in written_by_layer:
                    if dim_name in self.dimensions:
                        alloc = self.dimensions[dim_name]
                        if alloc.layer_written >= 0 and alloc.layer_written > layer.layer_idx:
                            errors.append(
                                f"L{layer.layer_idx} reads '{dim_name}' but it's "
                                f"written by L{alloc.layer_written}"
                            )

            # Record writes
            for dim_name in layer.writes:
                if dim_name not in written_by_layer:
                    written_by_layer[dim_name] = layer.layer_idx

        return errors

    def check_dimension_collisions(self) -> List[str]:
        """Check for overlapping dimension allocations.

        Returns list of error messages.
        """
        errors = []
        allocs = list(self.dimensions.values())

        for i, a in enumerate(allocs):
            for b in allocs[i+1:]:
                if a.overlaps(b):
                    errors.append(
                        f"Collision: {a.name} [{a.start}:{a.end}] "
                        f"overlaps {b.name} [{b.start}:{b.end}]"
                    )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'dimensions': {k: v.to_dict() for k, v in self.dimensions.items()},
            'layers': [layer.to_dict() for layer in self.layers],
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CompilerIR':
        """Deserialize from dictionary."""
        ir = cls(
            d_model=d['d_model'],
            n_layers=d['n_layers'],
            n_heads=d['n_heads'],
            metadata=d.get('metadata', {}),
        )
        ir.dimensions = {k: DimensionAlloc.from_dict(v)
                         for k, v in d['dimensions'].items()}
        ir.layers = [LayerSpec.from_dict(layer) for layer in d['layers']]
        return ir

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> 'CompilerIR':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        """Save IR to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'CompilerIR':
        """Load IR from file."""
        with open(path) as f:
            return cls.from_json(f.read())

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"CompilerIR Summary",
            f"  Model: d={self.d_model}, layers={self.n_layers}, heads={self.n_heads}",
            f"  Dimensions: {len(self.dimensions)} allocated",
            f"  Operations:",
        ]

        total_attn = sum(len(l.attention_ops) for l in self.layers)
        total_ffn = sum(len(l.ffn_ops) for l in self.layers)
        lines.append(f"    Attention: {total_attn} ops")
        lines.append(f"    FFN: {total_ffn} ops")

        # Check for issues
        errors = self.check_dimension_collisions() + self.check_read_before_write()
        if errors:
            lines.append(f"  Errors: {len(errors)}")
            for e in errors[:5]:
                lines.append(f"    - {e}")
            if len(errors) > 5:
                lines.append(f"    ... and {len(errors) - 5} more")
        else:
            lines.append("  Status: OK (no errors)")

        return '\n'.join(lines)


# =============================================================================
# Helper functions
# =============================================================================

def create_empty_ir(d_model: int = 512, n_layers: int = 17,
                    n_heads: int = 8) -> CompilerIR:
    """Create an empty IR with default configuration."""
    return CompilerIR(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
