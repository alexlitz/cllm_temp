"""Typed dim system for the IR — gives ``DimRef`` a normal-IR shape.

Three additions, each backward-compatible with the existing string-keyed
``DimRef`` / ``ConditionTerm`` / ``WriteTerm`` API:

1. **``DimType``** — semantic categories for residual-stream slots:
   ``SCALAR``, ``BAND``, ``MARKER``, ``CARRY``, ``OPCODE_FLAG``,
   ``STRUCTURAL``, ``OUTPUT``, ``UNKNOWN``. Each carries an optional
   ``width`` for band types (16 for nibble bands, 8 for byte bands).

2. **``DimSchema``** — a registry mapping ``dim_name`` → ``DimType``.
   Populated declaratively at module import time from a curated list,
   with fallback inference from name conventions
   (e.g. ``MARK_*`` → ``MARKER``, ``OP_*`` → ``OPCODE_FLAG``).

3. **``Value``** — a typed reference to a residual-stream slot at a
   specific schedule point. Wraps a ``DimRef`` with its ``DimType`` and
   an optional SSA version (``None`` = current step, ``-1`` = previous
   step, ``layer_idx`` = after that layer).

These promote the IR from "strings into a float vector" to "typed Values
flowing through an explicit schedule." Existing rules still parse from
strings; the typed version coexists for new code + verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# DimType — semantic categories
# ---------------------------------------------------------------------------


class DimType(Enum):
    """Semantic category of a residual-stream slot.

    Categories aren't strictly disjoint — some dims serve double duty —
    but the dominant category is what the verifier checks.
    """

    SCALAR = "scalar"
    """One-cell scalar value (e.g. ``CONST``, ``IS_BYTE``, count slots)."""

    BAND = "band"
    """N-cell one-hot band where exactly one cell is hot per token row
    (e.g. ``ALU_LO`` with 16 cells, ``BYTE_INDEX`` with 4 cells)."""

    MARKER = "marker"
    """Single-cell row-identity flag (e.g. ``MARK_AX``, ``MARK_PC``).
    Hot iff this row is the named role."""

    CARRY = "carry"
    """Inter-byte / inter-nibble carry bit. Connects an arithmetic op's
    output to the next-position input (e.g. ``CARRY``, ``BORROW``)."""

    OPCODE_FLAG = "opcode_flag"
    """Single-cell opcode indicator (e.g. ``OP_ADD``, ``OP_LEV``). Hot
    on rows where the named opcode is the current instruction."""

    STRUCTURAL = "structural"
    """Step-structure marker emitted by the L0-L1 threshold heads
    (e.g. ``H0..H7``, ``L1H0..L1H4``, ``HAS_SE``)."""

    OUTPUT = "output"
    """Band that drives the model's output head logits (e.g.
    ``OUTPUT_LO``, ``OUTPUT_HI_THIS_STEP``)."""

    UNKNOWN = "unknown"
    """Type not declared; verifier won't check."""


@dataclass(frozen=True)
class DimSpec:
    """One typed dim declaration: name + category + optional width."""

    name: str
    type: DimType
    width: int = 1
    description: str = ""

    @property
    def is_band(self) -> bool:
        """Whether the dim is a one-hot / value-distribution band.

        Returns True for BAND and OUTPUT semantic types (and is used by
        the verifier to bias inference). For *cell addressability*
        (whether offset > 0 is valid), use ``is_multi_cell`` instead —
        STRUCTURAL dims like the L0 H0..H7 head bank are multi-cell
        (width 8) without being one-hot bands.
        """
        return self.type == DimType.BAND or self.type == DimType.OUTPUT

    @property
    def is_multi_cell(self) -> bool:
        """Whether the dim has more than one addressable cell.

        True iff ``width > 1``. Captures BAND / OUTPUT / STRUCTURAL /
        any-multi-cell-CARRY without overloading ``is_band``.
        """
        return self.width > 1

    def cell(self, offset: int) -> "Value":
        """Address one cell within this dim. Single-cell dims (width=1)
        only accept offset 0; multi-cell dims accept ``0 <= offset <
        width``.
        """
        if not self.is_multi_cell and offset != 0:
            raise ValueError(
                f"DimSpec({self.name!r}, {self.type.name}): single-cell "
                f"dim only valid at offset 0, got offset={offset}"
            )
        if self.is_multi_cell and not (0 <= offset < self.width):
            raise ValueError(
                f"DimSpec({self.name!r}, {self.type.name}): offset "
                f"{offset} out of width {self.width}"
            )
        return Value(name=self.name, offset=offset, type=self.type)


# ---------------------------------------------------------------------------
# DimSchema — name → DimSpec registry
# ---------------------------------------------------------------------------


class DimSchema:
    """Registry of declared dim types, with inference fallback.

    Use the module-level ``register_dim`` / ``schema_for`` to populate
    and query. The registry is global and singleton — the IR's residual
    stream is a shared resource, so its type schema is module-level.
    """

    def __init__(self):
        self._specs: Dict[str, DimSpec] = {}

    def register(self, spec: DimSpec) -> None:
        if spec.name in self._specs:
            existing = self._specs[spec.name]
            if existing != spec:
                raise ValueError(
                    f"DimSchema: conflicting spec for {spec.name!r}: "
                    f"existing={existing}, new={spec}"
                )
            return
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[DimSpec]:
        return self._specs.get(name)

    def get_or_infer(self, name: str) -> DimSpec:
        """Look up the spec; if absent, infer from name conventions."""
        if name in self._specs:
            return self._specs[name]
        return _infer_spec(name)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self):
        return len(self._specs)

    def to_dict(self) -> Dict[str, DimSpec]:
        return dict(self._specs)


# Module-level singleton registry.
_SCHEMA = DimSchema()


def register_dim(spec: DimSpec) -> None:
    """Register a typed dim spec at module-init time."""
    _SCHEMA.register(spec)


def schema_for(name: str) -> DimSpec:
    """Look up the registered spec for ``name``, or infer one."""
    return _SCHEMA.get_or_infer(name)


def global_schema() -> DimSchema:
    """Return the module-level singleton (for verifiers / introspection)."""
    return _SCHEMA


def _infer_spec(name: str) -> DimSpec:
    """Fallback type inference from naming conventions.

    Rules:
      * ``MARK_*`` → ``MARKER`` (width 1)
      * ``OP_*`` → ``OPCODE_FLAG`` (width 1)
      * ``H0..H7``, ``L1H*`` → ``STRUCTURAL`` (width 8 for the head bank)
      * Ends in ``_LO`` or ``_HI`` → ``BAND`` (width 16, nibble band)
      * Contains ``OUTPUT`` → ``OUTPUT`` (width 16)
      * Contains ``CARRY`` or ``BORROW`` → ``CARRY`` (width 1)
      * Otherwise → ``SCALAR`` (width 1)
    """
    if name.startswith("MARK_"):
        return DimSpec(name, DimType.MARKER, 1, "inferred from MARK_ prefix")
    if name.startswith("OP_"):
        return DimSpec(name, DimType.OPCODE_FLAG, 1, "inferred from OP_ prefix")
    if name.startswith("L1H") or name.startswith("H"):
        # H0..H7 are the 8 L0 threshold heads; treat as STRUCTURAL with
        # a default width of 8 (callers that need a different width
        # should declare explicitly).
        return DimSpec(name, DimType.STRUCTURAL, 8, "inferred from H* / L1H* prefix")
    if "OUTPUT" in name:
        return DimSpec(name, DimType.OUTPUT, 16, "inferred from OUTPUT substring")
    if "CARRY" in name or "BORROW" in name:
        return DimSpec(name, DimType.CARRY, 1, "inferred from CARRY/BORROW substring")
    if name.endswith("_LO") or name.endswith("_HI") or name.endswith("_THIS_STEP"):
        return DimSpec(name, DimType.BAND, 16, "inferred from _LO/_HI/_THIS_STEP suffix")
    return DimSpec(name, DimType.UNKNOWN, 1, "no inference rule matched")


# ---------------------------------------------------------------------------
# Value — typed reference to a residual slot at a schedule point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Value:
    """A typed reference to one residual-stream cell at a schedule point.

    SSA-ish: a ``Value`` represents the contents of cell ``name+offset``
    at version ``version``. ``version=None`` means "current step's
    value", ``version=-1`` means "previous step's value" (the existing
    ``DIM.*.-1`` cross-step alias), and ``version=layer_idx`` means
    "after layer ``layer_idx`` writes."

    The ``type`` field is the dim's declared semantic category; for
    inferred / unknown types it falls back to ``DimType.UNKNOWN``.
    """

    name: str
    offset: int = 0
    type: DimType = DimType.UNKNOWN
    version: Optional[int] = None

    @classmethod
    def of(cls, name: str, offset: int = 0) -> "Value":
        """Build a Value with type looked up from the schema."""
        spec = schema_for(name)
        return cls(name=name, offset=offset, type=spec.type)

    @classmethod
    def parse(cls, key: str) -> "Value":
        """Parse a string like ``"ALU_LO+5"`` or ``"OUTPUT_LO.*.-1"``.

        Supports the existing cross-step SSA alias notation
        ``"<dim>.*.-1"``.
        """
        if ".*." in key:
            # Cross-step alias form: DIM.*.-1
            base, _, ver_str = key.rpartition(".*.")
            ver = int(ver_str)
            base_dim = base.split("+", 1)[0]
            base_off = int(base.split("+", 1)[1]) if "+" in base else 0
            spec = schema_for(base_dim)
            return cls(name=base_dim, offset=base_off, type=spec.type, version=ver)
        if "+" in key:
            name, off = key.rsplit("+", 1)
            spec = schema_for(name)
            return cls(name=name, offset=int(off), type=spec.type)
        spec = schema_for(key)
        return cls(name=key, offset=0, type=spec.type)

    def to_key(self) -> str:
        """Render to the canonical string key used by the existing
        residual-state dicts.

        Roundtrips with ``Value.parse``: a zero-offset value renders
        without the ``+0`` suffix so the form matches the shorthand
        used by the existing ``state[name] = value`` dicts.
        """
        if self.version is not None:
            if self.offset == 0:
                return f"{self.name}.*.{self.version}"
            return f"{self.name}+{self.offset}.*.{self.version}"
        if self.offset == 0:
            return self.name
        return f"{self.name}+{self.offset}"

    @property
    def is_cross_step(self) -> bool:
        return self.version is not None and self.version < 0


# ---------------------------------------------------------------------------
# Standard dim registrations — the canonical set the codebase uses
# ---------------------------------------------------------------------------
#
# These mirror what's in the BD layout (see ``vm_step._SetDim``). Adding
# more declarations is cheap — the registry holds them in a dict.

# Markers
for _m in (
    "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
    "MARK_SE", "MARK_HALT",
):
    register_dim(DimSpec(_m, DimType.MARKER, 1, "row-identity flag"))

# Opcode flags
for _op in (
    "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
    "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI", "OP_SC",
    "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_SHL", "OP_SHR",
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_EXIT", "OP_NOP", "OP_PUTCHAR", "OP_GETCHAR",
):
    register_dim(DimSpec(_op, DimType.OPCODE_FLAG, 1, "opcode active flag"))

# Carry / borrow
for _c in ("CARRY", "BORROW"):
    register_dim(DimSpec(_c, DimType.CARRY, 1, "inter-byte carry/borrow"))

# Output bands
for _o in (
    "OUTPUT_LO", "OUTPUT_HI", "OUTPUT_HI_THIS_STEP", "OUTPUT_HI_PREV_STEP",
):
    register_dim(DimSpec(_o, DimType.OUTPUT, 16, "head logit driver"))

# Nibble / band dims
for _b in (
    "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
    "FETCH_LO", "FETCH_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
    "EMBED_LO", "EMBED_HI",
    "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO", "ADDR_B1_HI",
    "ADDR_B2_LO", "ADDR_B2_HI",
    "TEMP",
):
    register_dim(DimSpec(_b, DimType.BAND, 16, "nibble-value band"))

# CMP is a 4-cell band carrying the comparison-flag bits (EQ/NE/LT/GT
# encoded at offsets 0..3). L9 cmp rules write CMP+0..CMP+3.
register_dim(DimSpec("CMP", DimType.BAND, 4, "comparison flag bits"))

# Single-cell markers / flags
for _s in (
    "CONST", "IS_BYTE", "HAS_SE",
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    "STACK0_BYTE0",
    "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
    "NEXT_STACK0", "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
):
    register_dim(DimSpec(_s, DimType.SCALAR, 1, "single-cell scalar"))

# Structural threshold-head bands (L0/L1)
for _h in ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"):
    register_dim(DimSpec(_h, DimType.STRUCTURAL, 8, "L0 threshold head"))
for _h in ("L1H0", "L1H1", "L1H2", "L1H3", "L1H4"):
    register_dim(DimSpec(_h, DimType.STRUCTURAL, 8, "L1 threshold head"))


# ---------------------------------------------------------------------------
# Type verification — catches read/write type mismatches
# ---------------------------------------------------------------------------


@dataclass
class TypeIssue:
    """One detected type problem in a rule."""

    kind: str
    message: str
    rule_name: Optional[str] = None


def check_rule_types(rule) -> list:
    """Inspect an ``FFNRule`` for type-level issues.

    Returns a list of ``TypeIssue``s — empty if the rule is clean.

    Checks:
      * Conditions: each read's ``DimRef`` must address a valid cell of
        a registered dim. Bands need ``0 <= offset < width``;
        non-bands need ``offset == 0``.
      * Writes: same as conditions.
      * Gate: same.
    """
    issues = []
    name = getattr(rule, "name", None)

    def _check_dimref(dim_ref, role: str):
        spec = schema_for(dim_ref.name)
        if spec.type == DimType.UNKNOWN:
            return  # Don't error on unknown; just skip.
        if not spec.is_multi_cell and dim_ref.offset != 0:
            issues.append(TypeIssue(
                kind="single_cell_offset",
                message=(
                    f"{role} {dim_ref.name}+{dim_ref.offset}: dim is "
                    f"{spec.type.name} (single-cell, width=1), only "
                    f"offset 0 valid"
                ),
                rule_name=name,
            ))
            return
        if spec.is_multi_cell and not (0 <= dim_ref.offset < spec.width):
            issues.append(TypeIssue(
                kind="offset_out_of_range",
                message=(
                    f"{role} {dim_ref.name}+{dim_ref.offset}: dim is "
                    f"{spec.type.name} width {spec.width}, offset out "
                    f"of range"
                ),
                rule_name=name,
            ))

    for term in getattr(rule, "conditions", ()):
        _check_dimref(term.dim, "condition")
    for write in getattr(rule, "writes", ()):
        _check_dimref(write.dim, "write")
    if getattr(rule, "gate", None) is not None:
        _check_dimref(rule.gate, "gate")
    for term in getattr(rule, "gate_terms", ()):
        _check_dimref(term.dim, "gate_term")
    return issues


def check_rules(rules) -> list:
    """Run ``check_rule_types`` over a sequence of rules; return the
    flattened issue list."""
    all_issues = []
    for rule in rules:
        all_issues.extend(check_rule_types(rule))
    return all_issues


__all__ = [
    "DimType",
    "DimSpec",
    "DimSchema",
    "Value",
    "TypeIssue",
    "register_dim",
    "schema_for",
    "global_schema",
    "check_rule_types",
    "check_rules",
    "typed_reads",
    "typed_writes",
    "operand_use_def",
    "find_producers",
    "find_consumers",
]


# ===========================================================================
# Operand views (increment 2): typed read/write lists on Operation
# ===========================================================================
#
# Backward-compatible query layer over the existing string-keyed
# ``Operation.reads`` / ``Operation.writes`` sets. Promotes them to typed
# Value lists without touching the Operation dataclass.


def typed_reads(op) -> "list[Value]":
    """Return ``Operation.reads`` as a list of typed Values.

    Each read name is parsed (handling the ``DIM.*.-1`` cross-step alias
    form) and its DimType looked up via ``schema_for``. Order is stable
    (sorted by name then version) so two calls on the same Operation
    return identical lists.
    """
    reads = getattr(op, "reads", set()) or set()
    values = [Value.parse(name) for name in reads]
    return sorted(values, key=lambda v: (v.name, v.offset, v.version or 0))


def typed_writes(op) -> "list[Value]":
    """Same shape as ``typed_reads`` but over ``Operation.writes``."""
    writes = getattr(op, "writes", set()) or set()
    values = [Value.parse(name) for name in writes]
    return sorted(values, key=lambda v: (v.name, v.offset, v.version or 0))


def operand_use_def(op) -> "tuple[list[Value], list[Value]]":
    """Convenience: ``(typed_reads, typed_writes)`` in one call.

    Mirrors the def-use pair that traditional IRs surface on every
    instruction: the operands an op consumes plus the values it produces.
    """
    return typed_reads(op), typed_writes(op)


def find_producers(ops, dim_name: str) -> "list":
    """Return every op in ``ops`` whose write set contains ``dim_name``.

    Order is the input order (stable across calls). Useful for
    attribution: "which op produces this dim?"
    """
    matches = []
    for op in ops:
        writes = getattr(op, "writes", set()) or set()
        if dim_name in writes:
            matches.append(op)
            continue
        # Also match the cross-step alias form
        for write in writes:
            if write.startswith(f"{dim_name}+") or write.startswith(f"{dim_name}.*"):
                matches.append(op)
                break
    return matches


def find_consumers(ops, dim_name: str) -> "list":
    """Return every op in ``ops`` whose read set contains ``dim_name``."""
    matches = []
    for op in ops:
        reads = getattr(op, "reads", set()) or set()
        if dim_name in reads:
            matches.append(op)
            continue
        for read in reads:
            if read.startswith(f"{dim_name}+") or read.startswith(f"{dim_name}.*"):
                matches.append(op)
                break
    return matches
