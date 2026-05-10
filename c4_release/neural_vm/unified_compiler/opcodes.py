"""
Declarative Opcode Specifications for Neural VM.

Each opcode is specified declaratively with:
- What registers/memory it reads
- What registers/memory it writes
- What computation it performs
- Layer requirements for implementation
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum

from ..embedding import Opcode


# =============================================================================
# Opcode Categories
# =============================================================================

class OpcodeCategory(Enum):
    """Categories of opcodes for grouping similar operations."""
    IMMEDIATE = "immediate"      # IMM, LEA
    ARITHMETIC = "arithmetic"    # ADD, SUB, MUL, DIV, MOD
    BITWISE = "bitwise"          # OR, XOR, AND, SHL, SHR
    COMPARISON = "comparison"    # EQ, NE, LT, GT, LE, GE
    CONTROL = "control"          # JMP, BZ, BNZ
    FUNCTION = "function"        # JSR, ENT, ADJ, LEV
    MEMORY = "memory"            # LI, LC, SI, SC, PSH
    SYSTEM = "system"            # EXIT, NOP, PRTF, etc.


# =============================================================================
# Opcode Specification
# =============================================================================

@dataclass
class OpcodeSpec:
    """Declarative specification of a VM opcode.

    Attributes:
        opcode: Numeric opcode value
        name: Human-readable name
        category: Opcode category for grouping

        # Data flow
        reads_ax: Reads the AX register
        reads_stack: Pops from stack (reads STACK0)
        reads_memory: Reads from memory at address
        reads_immediate: Uses immediate value from instruction
        reads_bp: Reads base pointer

        writes_ax: Writes to AX register
        writes_pc: Modifies program counter
        writes_sp: Modifies stack pointer
        writes_bp: Modifies base pointer
        writes_memory: Writes to memory

        # Computation
        alu_op: Function (a, b) -> result for ALU operations
        is_binary: Takes two operands (pop + AX)
        is_unary: Takes one operand (AX only)

        # Special flags
        is_control_flow: Changes PC non-sequentially
        is_memory_op: Accesses memory
        needs_handler: Still requires Python handler (not fully neural)

        # Layer hints
        min_layers: Minimum layers needed
        layer_stages: List of (layer, purpose) tuples
    """
    opcode: int
    name: str
    category: OpcodeCategory = OpcodeCategory.SYSTEM

    # Data flow - reads
    reads_ax: bool = False
    reads_stack: bool = False
    reads_memory: bool = False
    reads_immediate: bool = False
    reads_bp: bool = False
    reads_sp: bool = False

    # Data flow - writes
    writes_ax: bool = False
    writes_pc: bool = False
    writes_sp: bool = False
    writes_bp: bool = False
    writes_memory: bool = False

    # Computation
    alu_op: Optional[Callable[[int, int], int]] = None
    is_binary: bool = False
    is_unary: bool = False

    # Special flags
    is_control_flow: bool = False
    is_memory_op: bool = False
    needs_handler: bool = False

    # Layer hints (computed from requirements)
    min_layers: int = 1
    layer_stages: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived properties."""
        # Binary ops need at least: fetch, gather, ALU, writeback
        if self.is_binary:
            self.min_layers = max(self.min_layers, 4)

        # Memory ops need address computation + fetch
        if self.is_memory_op:
            self.min_layers = max(self.min_layers, 5)

        # Control flow needs PC update logic
        if self.is_control_flow:
            self.min_layers = max(self.min_layers, 3)

    @property
    def nibble_lo(self) -> int:
        """Low nibble of opcode."""
        return self.opcode & 0xF

    @property
    def nibble_hi(self) -> int:
        """High nibble of opcode."""
        return (self.opcode >> 4) & 0xF

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excluding callables)."""
        return {
            'opcode': self.opcode,
            'name': self.name,
            'category': self.category.value,
            'reads_ax': self.reads_ax,
            'reads_stack': self.reads_stack,
            'reads_memory': self.reads_memory,
            'reads_immediate': self.reads_immediate,
            'reads_bp': self.reads_bp,
            'writes_ax': self.writes_ax,
            'writes_pc': self.writes_pc,
            'writes_sp': self.writes_sp,
            'writes_bp': self.writes_bp,
            'writes_memory': self.writes_memory,
            'is_binary': self.is_binary,
            'is_unary': self.is_unary,
            'is_control_flow': self.is_control_flow,
            'is_memory_op': self.is_memory_op,
            'needs_handler': self.needs_handler,
            'min_layers': self.min_layers,
        }


# =============================================================================
# ALU Operations (lambdas for lookup table generation)
# =============================================================================

def _add(a: int, b: int) -> int:
    return (a + b) & 0xFFFFFFFF

def _sub(a: int, b: int) -> int:
    return (a - b) & 0xFFFFFFFF

def _mul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF

def _div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a // b) & 0xFFFFFFFF

def _mod(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a % b) & 0xFFFFFFFF

def _or(a: int, b: int) -> int:
    return a | b

def _xor(a: int, b: int) -> int:
    return a ^ b

def _and(a: int, b: int) -> int:
    return a & b

def _shl(a: int, b: int) -> int:
    return (a << (b & 31)) & 0xFFFFFFFF

def _shr(a: int, b: int) -> int:
    return (a >> (b & 31)) & 0xFFFFFFFF

def _eq(a: int, b: int) -> int:
    return 1 if a == b else 0

def _ne(a: int, b: int) -> int:
    return 1 if a != b else 0

def _lt(a: int, b: int) -> int:
    return 1 if a < b else 0

def _gt(a: int, b: int) -> int:
    return 1 if a > b else 0

def _le(a: int, b: int) -> int:
    return 1 if a <= b else 0

def _ge(a: int, b: int) -> int:
    return 1 if a >= b else 0


# =============================================================================
# Opcode Specifications
# =============================================================================

OPCODES: Dict[int, OpcodeSpec] = {}


def _register(spec: OpcodeSpec) -> OpcodeSpec:
    """Register an opcode specification."""
    OPCODES[spec.opcode] = spec
    return spec


# -----------------------------------------------------------------------------
# Immediate operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.IMM,
    name="IMM",
    category=OpcodeCategory.IMMEDIATE,
    reads_immediate=True,
    writes_ax=True,
    layer_stages=[
        (5, "fetch immediate"),
        (6, "write to AX"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.LEA,
    name="LEA",
    category=OpcodeCategory.IMMEDIATE,
    reads_immediate=True,
    reads_bp=True,
    writes_ax=True,
    needs_handler=True,  # BP tracking still needed
    layer_stages=[
        (5, "fetch immediate"),
        (6, "add to BP"),
        (7, "write to AX"),
    ],
))

# -----------------------------------------------------------------------------
# Arithmetic operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.ADD,
    name="ADD",
    category=OpcodeCategory.ARITHMETIC,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,  # Implicit pop
    alu_op=_add,
    is_binary=True,
    layer_stages=[
        (5, "decode opcode"),
        (6, "relay AX"),
        (7, "gather operands"),
        (9, "ALU compute"),
        (10, "apply carry"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.SUB,
    name="SUB",
    category=OpcodeCategory.ARITHMETIC,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_sub,
    is_binary=True,
    layer_stages=[
        (5, "decode opcode"),
        (6, "relay AX"),
        (7, "gather operands"),
        (9, "ALU compute"),
        (10, "apply borrow"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.MUL,
    name="MUL",
    category=OpcodeCategory.ARITHMETIC,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_mul,
    is_binary=True,
    min_layers=6,  # Needs more stages for multi-byte
    layer_stages=[
        (5, "decode opcode"),
        (6, "relay AX"),
        (7, "gather operands"),
        (9, "partial products"),
        (10, "accumulate low"),
        (11, "accumulate high"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.DIV,
    name="DIV",
    category=OpcodeCategory.ARITHMETIC,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_div,
    is_binary=True,
    min_layers=6,
    needs_handler=True,  # Division is complex
))

_register(OpcodeSpec(
    opcode=Opcode.MOD,
    name="MOD",
    category=OpcodeCategory.ARITHMETIC,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_mod,
    is_binary=True,
    min_layers=6,
    needs_handler=True,  # Modulo is complex
))

# -----------------------------------------------------------------------------
# Bitwise operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.OR,
    name="OR",
    category=OpcodeCategory.BITWISE,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_or,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.XOR,
    name="XOR",
    category=OpcodeCategory.BITWISE,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_xor,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.AND,
    name="AND",
    category=OpcodeCategory.BITWISE,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_and,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.SHL,
    name="SHL",
    category=OpcodeCategory.BITWISE,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_shl,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.SHR,
    name="SHR",
    category=OpcodeCategory.BITWISE,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_shr,
    is_binary=True,
))

# -----------------------------------------------------------------------------
# Comparison operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.EQ,
    name="EQ",
    category=OpcodeCategory.COMPARISON,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_eq,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.NE,
    name="NE",
    category=OpcodeCategory.COMPARISON,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_ne,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.LT,
    name="LT",
    category=OpcodeCategory.COMPARISON,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_lt,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.GT,
    name="GT",
    category=OpcodeCategory.COMPARISON,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_gt,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.LE,
    name="LE",
    category=OpcodeCategory.COMPARISON,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_le,
    is_binary=True,
))

_register(OpcodeSpec(
    opcode=Opcode.GE,
    name="GE",
    category=OpcodeCategory.COMPARISON,
    reads_ax=True,
    reads_stack=True,
    writes_ax=True,
    writes_sp=True,
    alu_op=_ge,
    is_binary=True,
))

# -----------------------------------------------------------------------------
# Control flow operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.JMP,
    name="JMP",
    category=OpcodeCategory.CONTROL,
    reads_immediate=True,
    writes_pc=True,
    is_control_flow=True,
    needs_handler=True,  # PC propagation issue
    layer_stages=[
        (5, "fetch target"),
        (6, "compute new PC"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.BZ,
    name="BZ",
    category=OpcodeCategory.CONTROL,
    reads_ax=True,
    reads_immediate=True,
    writes_pc=True,
    is_control_flow=True,
    needs_handler=True,
    layer_stages=[
        (5, "fetch target"),
        (6, "check AX == 0"),
        (7, "conditional PC update"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.BNZ,
    name="BNZ",
    category=OpcodeCategory.CONTROL,
    reads_ax=True,
    reads_immediate=True,
    writes_pc=True,
    is_control_flow=True,
    needs_handler=True,
    layer_stages=[
        (5, "fetch target"),
        (6, "check AX != 0"),
        (7, "conditional PC update"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.JSR,
    name="JSR",
    category=OpcodeCategory.FUNCTION,
    reads_immediate=True,
    writes_pc=True,
    writes_sp=True,  # Push return address
    is_control_flow=True,
    needs_handler=True,
    layer_stages=[
        (5, "fetch target"),
        (6, "push return PC"),
        (7, "jump to target"),
    ],
))

# -----------------------------------------------------------------------------
# Function operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.ENT,
    name="ENT",
    category=OpcodeCategory.FUNCTION,
    reads_immediate=True,
    reads_sp=True,
    writes_sp=True,
    writes_bp=True,
    needs_handler=True,  # Stack frame setup
    layer_stages=[
        (5, "fetch frame size"),
        (14, "push BP"),
        (15, "set BP = SP"),
        (16, "SP -= frame_size"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.ADJ,
    name="ADJ",
    category=OpcodeCategory.FUNCTION,
    reads_immediate=True,
    reads_sp=True,
    writes_sp=True,
    layer_stages=[
        (5, "fetch adjustment"),
        (6, "SP += adjustment"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.LEV,
    name="LEV",
    category=OpcodeCategory.FUNCTION,
    reads_bp=True,
    reads_memory=True,  # Pop return address
    writes_sp=True,
    writes_bp=True,
    writes_pc=True,
    is_control_flow=True,
    needs_handler=True,  # Stack frame teardown
    layer_stages=[
        (14, "restore SP from BP"),
        (15, "pop BP"),
        (16, "pop return PC"),
    ],
))

# -----------------------------------------------------------------------------
# Memory operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.LI,
    name="LI",
    category=OpcodeCategory.MEMORY,
    reads_ax=True,  # Address in AX
    reads_memory=True,
    writes_ax=True,
    is_memory_op=True,
    needs_handler=True,
    layer_stages=[
        (13, "compute address"),
        (14, "fetch from memory"),
        (15, "write to AX"),
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.LC,
    name="LC",
    category=OpcodeCategory.MEMORY,
    reads_ax=True,
    reads_memory=True,
    writes_ax=True,
    is_memory_op=True,
    needs_handler=True,
))

_register(OpcodeSpec(
    opcode=Opcode.SI,
    name="SI",
    category=OpcodeCategory.MEMORY,
    reads_ax=True,  # Value to store
    reads_stack=True,  # Address from stack
    writes_memory=True,
    writes_sp=True,
    is_memory_op=True,
    needs_handler=True,
))

_register(OpcodeSpec(
    opcode=Opcode.SC,
    name="SC",
    category=OpcodeCategory.MEMORY,
    reads_ax=True,
    reads_stack=True,
    writes_memory=True,
    writes_sp=True,
    is_memory_op=True,
    needs_handler=True,
))

_register(OpcodeSpec(
    opcode=Opcode.PSH,
    name="PSH",
    category=OpcodeCategory.MEMORY,
    reads_ax=True,
    writes_sp=True,
    writes_memory=True,  # Push to stack
    needs_handler=True,  # SP tracking
))

# -----------------------------------------------------------------------------
# System operations
# -----------------------------------------------------------------------------

_register(OpcodeSpec(
    opcode=Opcode.EXIT,
    name="EXIT",
    category=OpcodeCategory.SYSTEM,
    reads_ax=True,  # Return value
    # EXIT is terminal - no writes needed
    layer_stages=[
        (5, "decode EXIT"),
        # Exit stops execution
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.NOP,
    name="NOP",
    category=OpcodeCategory.SYSTEM,
    # NOP does nothing
    layer_stages=[
        (5, "decode NOP"),
        # No operation
    ],
))

_register(OpcodeSpec(
    opcode=Opcode.PRTF,
    name="PRTF",
    category=OpcodeCategory.SYSTEM,
    reads_ax=True,
    reads_stack=True,
    reads_memory=True,
    needs_handler=True,  # I/O syscall
))

_register(OpcodeSpec(
    opcode=Opcode.GETCHAR,
    name="GETCHAR",
    category=OpcodeCategory.SYSTEM,
    writes_ax=True,
    needs_handler=True,  # I/O syscall
))

_register(OpcodeSpec(
    opcode=Opcode.PUTCHAR,
    name="PUTCHAR",
    category=OpcodeCategory.SYSTEM,
    reads_ax=True,
    needs_handler=True,  # I/O syscall
))


# =============================================================================
# Helper functions
# =============================================================================

def get_opcode_spec(opcode: int) -> Optional[OpcodeSpec]:
    """Get specification for an opcode."""
    return OPCODES.get(opcode)


def get_opcodes_by_category(category: OpcodeCategory) -> List[OpcodeSpec]:
    """Get all opcodes in a category."""
    return [spec for spec in OPCODES.values() if spec.category == category]


def get_neural_ready_opcodes() -> List[OpcodeSpec]:
    """Get opcodes that don't need handlers."""
    return [spec for spec in OPCODES.values() if not spec.needs_handler]


def get_handler_required_opcodes() -> List[OpcodeSpec]:
    """Get opcodes that still need handlers."""
    return [spec for spec in OPCODES.values() if spec.needs_handler]


def opcode_summary() -> str:
    """Generate summary of all opcodes."""
    lines = ["Opcode Summary", "=" * 60]

    by_category = {}
    for spec in OPCODES.values():
        cat = spec.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(spec)

    for cat, specs in sorted(by_category.items()):
        lines.append(f"\n{cat.upper()}:")
        for spec in specs:
            handler = "[H]" if spec.needs_handler else "   "
            lines.append(f"  {handler} {spec.name:8} (0x{spec.opcode:02x})")

    neural = len(get_neural_ready_opcodes())
    handler = len(get_handler_required_opcodes())
    lines.append(f"\nTotal: {len(OPCODES)} opcodes")
    lines.append(f"  Neural-ready: {neural}")
    lines.append(f"  Need handler: {handler}")

    return '\n'.join(lines)
