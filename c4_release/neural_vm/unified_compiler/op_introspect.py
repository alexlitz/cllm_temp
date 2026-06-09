"""Op-introspection: derive ``Operation.reads`` / ``Operation.writes`` from rules.

The architectural fix for ``docs/UNDECLARED_DIM_AUDIT_2026_06_09.md``: instead
of hand-annotating ``reads={...}`` / ``writes={...}`` on every Operation and
hoping authors keep them in sync with the IR, *derive* the sets from the actual
``compiler_ir`` rule contents (FFN rules + attention head specs).

This module owns three layers of derivation infra:

  1. :func:`derive_op_reads_writes_from_rules` — pure derivation. Given an
     ``Operation`` and the layout's ``dim_positions`` / ``dim_sizes`` (so
     attention-head Q/K/V/O ints can be mapped back to dim names), returns
     ``(reads, writes)`` as sets of dim-name strings.

  2. :func:`assert_declared_matches_derived` — CI-style contract check that the
     manually-declared ``op.reads`` / ``op.writes`` are a *superset* of the
     derived sets (declared >= derived). In ``strict=True`` mode requires
     equality.

  3. :func:`derive_operation` — returns a *new* ``Operation`` with the manually
     declared sets replaced by the derived sets. Used by the migration
     opt-in ``Operation.derive_reads_writes`` flag.

Sources of truth for derivation:

  * **FFN rules** (``FFNRule.conditions`` / ``.gate`` / ``.gate_terms``) feed
    ``reads``. Rule ``writes`` feed ``writes`` (skipping ``weight == 0``
    explicit zero-writes — they don't shift the dep graph).
  * **Attention head specs** (``DeclarativeAttentionHeadSpec.q`` / ``.k`` /
    ``.v``) feed ``reads`` — Q/K/V projection inputs are residual-stream
    reads. ``DeclarativeAttentionHeadSpec.o`` feeds ``writes`` — O projection
    targets a residual column.
  * **Block ops** carry a multi-layer ``compiler_ir`` whose layers each have
    both FFN + attention sides. The derivation is the union over every
    contained sub-component, exactly mirroring how ``_walk_ops_with_layers``
    surfaces the IR for the dim-flow analyzer and audit tools.

What this module **does not** do:

  * Mutate any existing op's declared sets. Migration of all 162 ops to
    ``derive_reads_writes=True`` is a separate wave; this module provides the
    plumbing.
  * Modify the dep-graph contract. ``compiler.add_op`` continues to use
    ``op.writes`` exactly as declared.

See also:

  * ``tools/undeclared_dim_audit.py`` — the audit that drove this design
    (29 ops with undeclared reads, 13 ops with undeclared writes as of
    2026-06-09).
  * ``tools/lint_op_reads_writes.py`` — the per-op ratchet built on top of
    ``assert_declared_matches_derived``.
  * ``Operation.derive_reads_writes`` method on
    ``unified_compiler/layer_compiler.py`` — the op-level migration entry
    point that calls back into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Set, Tuple

from .dim_flow import (
    _attention_heads_from_ir,
    _dim_int_to_name,
    _ffn_rules_from_ir,
    _materialize_op_ir,
)


__all__ = [
    "DerivedReadsWrites",
    "DerivationMismatch",
    "derive_op_reads_writes_from_rules",
    "assert_declared_matches_derived",
    "normalize_declared_set",
    "derive_operation",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedReadsWrites:
    """Outcome of one ``derive_op_reads_writes_from_rules`` call.

    ``reads`` / ``writes`` are dim-name strings (the canonical
    ``Operation.reads`` / ``writes`` convention). ``ir_available`` is True
    iff the op has a non-None ``compiler_ir`` (or a factory that produced
    one). When False the derivation is necessarily empty — the op has no
    declarative IR to introspect, which is the case for legacy imperative
    bakes the migration has not yet reached.
    """

    reads: Set[str]
    writes: Set[str]
    ir_available: bool


@dataclass(frozen=True)
class DerivationMismatch:
    """One difference between declared and derived sets.

    ``op_name`` is the ``Operation.name``. ``undeclared_reads`` is the set of
    dim names the IR actually consumes but the declared ``reads`` omits;
    ``undeclared_writes`` is the same for writes. ``over_declared_reads`` /
    ``over_declared_writes`` are the inverse — declared names not present in
    the derivation. Under the default contract (``declared >= derived``)
    over-declarations are *not* a violation; under ``strict=True`` they are.
    """

    op_name: str
    undeclared_reads: Tuple[str, ...]
    undeclared_writes: Tuple[str, ...]
    over_declared_reads: Tuple[str, ...]
    over_declared_writes: Tuple[str, ...]

    @property
    def has_undeclared(self) -> bool:
        return bool(self.undeclared_reads or self.undeclared_writes)

    @property
    def has_over_declared(self) -> bool:
        return bool(
            self.over_declared_reads or self.over_declared_writes
        )

    @property
    def is_clean(self) -> bool:
        return not self.has_undeclared and not self.has_over_declared


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_declared_set(declared: Optional[Set[str]]) -> Set[str]:
    """Strip ``+N`` offset and ``.*.N`` step-alias suffixes off declared names.

    Mirrors ``undeclared_dim_audit._normalize_declared`` so this module's
    comparison semantics line up with the audit's. Declared sets carry both
    bare dim names (``"OUTPUT_LO"``) and offset/aliased variants
    (``"OUTPUT_LO+1"``, ``"OUTPUT_LO.*.-1"``); the IR walker only surfaces
    the base name, so the comparison strips the suffixes off the declared
    side first.
    """
    out: Set[str] = set()
    for n in (declared or ()):
        if not isinstance(n, str):
            continue
        base = n.split(".", 1)[0].split("+", 1)[0]
        out.add(base)
    return out


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def derive_op_reads_writes_from_rules(
    op: Any,
    *,
    dim_positions: Optional[Mapping[str, int]] = None,
    dim_sizes: Optional[Mapping[str, int]] = None,
    head_dim: int = 64,
) -> DerivedReadsWrites:
    """Derive ``(reads, writes)`` dim-name sets by walking the op's IR.

    The walk mirrors ``tools/undeclared_dim_audit._ir_dim_sets`` so the
    derivation result is byte-identical with the audit's "actual" sets
    (modulo the audit's alias-canonicalization, which is applied separately
    in :func:`assert_declared_matches_derived`).

    Args:
      op: The ``Operation`` whose IR to introspect. Must expose
        ``compiler_ir`` and/or ``compiler_ir_factory``. The ``kind`` field
        is inspected only for diagnostics — every op kind shares the same
        derivation path because the IR walker handles multi-layer block
        IRs natively.
      dim_positions: Layout-allocated dim positions. Required for ops with
        attention head specs (the spec stores Q/K/V as resolved ``int``
        residual-column indices, so the walker needs the layout's
        position table to reverse-map). When ``None`` and the op has any
        attention head, those Q/K/V/O contributions are silently skipped
        — callers that need them must supply the layout's positions.
      dim_sizes: Per-dim slot widths (``layout.dim_sizes``). Optional;
        defaults to an empty mapping (``size=1`` for every dim).
      head_dim: Passed to ``compiler_ir_factory`` when the op's IR is
        factory-built. Defaults to 64 (the transformer block default).

    Returns:
      A :class:`DerivedReadsWrites` with ``reads`` / ``writes`` as
      dim-name string sets and ``ir_available`` indicating whether any IR
      was found.
    """
    dim_positions = dict(dim_positions or {})
    dim_sizes = dict(dim_sizes or {})

    ir = _materialize_op_ir(op, dim_positions, head_dim)
    if ir is None:
        return DerivedReadsWrites(set(), set(), ir_available=False)

    reads: Set[str] = set()
    writes: Set[str] = set()

    # FFN side: rule.conditions / rule.gate / rule.gate_terms -> reads;
    # rule.writes -> writes (skip explicit zero weights, which don't shift
    # the dep graph and parallel the audit's filtering).
    for rule in _ffn_rules_from_ir(ir):
        for ct in rule.conditions:
            reads.add(ct.dim.name)
        if rule.gate is not None:
            reads.add(rule.gate.name)
        for ct in rule.gate_terms:
            reads.add(ct.dim.name)
        for wt in rule.writes:
            if wt.weight == 0.0:
                continue
            writes.add(wt.dim.name)

    # Attention side: q/k/v -> reads, o -> writes. Skip zero-weight writes
    # (dead heads / dead slots). Each spec.q/.k/.v[i].dim is an int residual
    # column; spec.o[i].out_dim is the target column. Both need the layout's
    # position tables for reverse-mapping.
    for head in _attention_heads_from_ir(ir):
        spec = getattr(head, "spec", None)
        if spec is None:
            continue
        for w in getattr(spec, "q", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
            if r is not None:
                reads.add(r[0])
        for w in getattr(spec, "k", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
            if r is not None:
                reads.add(r[0])
        for w in getattr(spec, "v", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
            if r is not None:
                reads.add(r[0])
        for w in getattr(spec, "o", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.out_dim), dim_positions, dim_sizes)
            if r is not None:
                writes.add(r[0])

    return DerivedReadsWrites(reads, writes, ir_available=True)


# ---------------------------------------------------------------------------
# Contract assertion
# ---------------------------------------------------------------------------


def assert_declared_matches_derived(
    op: Any,
    *,
    dim_positions: Optional[Mapping[str, int]] = None,
    dim_sizes: Optional[Mapping[str, int]] = None,
    head_dim: int = 64,
    strict: bool = False,
    alias_canon: Optional[Mapping[str, str]] = None,
) -> DerivationMismatch:
    """Compare ``op``'s declared reads/writes against the derivation.

    Default contract: ``normalize(op.reads) >= derived.reads`` AND
    ``normalize(op.writes) >= derived.writes``. Under-declaration
    (an IR consumes/produces something the declared set omits) is the
    primary violation surface and is reported as ``undeclared_reads``
    / ``undeclared_writes`` on the returned mismatch.

    With ``strict=True`` the contract becomes equality (the migration
    target — once the lint enforces strict, the declared set IS the
    derivation, and authors can drop the manual annotation entirely via
    :func:`derive_operation`).

    ``alias_canon`` is the per-layout position-alias canonicalization map
    (e.g. ``"OUTPUT_HI_THIS_STEP"`` → ``"OUTPUT_HI"`` when they share a
    residual slot). Built by ``undeclared_dim_audit._build_alias_groups``.
    When supplied, both sides are canonicalized before comparison so
    aliased names don't produce false positives.

    Returns a :class:`DerivationMismatch`. Inspect
    ``DerivationMismatch.is_clean`` / ``.has_undeclared`` to drive a CI
    gate; ``.has_over_declared`` only matters under ``strict=True``.
    """
    derived = derive_op_reads_writes_from_rules(
        op,
        dim_positions=dim_positions,
        dim_sizes=dim_sizes,
        head_dim=head_dim,
    )

    declared_reads = normalize_declared_set(getattr(op, "reads", set()))
    declared_writes = normalize_declared_set(getattr(op, "writes", set()))

    if alias_canon is not None:
        def canon(s: Set[str]) -> Set[str]:
            return {alias_canon.get(n, n) for n in s}
        declared_reads = canon(declared_reads)
        declared_writes = canon(declared_writes)
        derived_reads = canon(derived.reads)
        derived_writes = canon(derived.writes)
    else:
        derived_reads = set(derived.reads)
        derived_writes = set(derived.writes)

    undeclared_reads = tuple(sorted(derived_reads - declared_reads))
    undeclared_writes = tuple(sorted(derived_writes - declared_writes))
    over_declared_reads: Tuple[str, ...] = ()
    over_declared_writes: Tuple[str, ...] = ()
    if strict:
        over_declared_reads = tuple(sorted(declared_reads - derived_reads))
        over_declared_writes = tuple(
            sorted(declared_writes - derived_writes)
        )

    return DerivationMismatch(
        op_name=getattr(op, "name", "<anonymous>"),
        undeclared_reads=undeclared_reads,
        undeclared_writes=undeclared_writes,
        over_declared_reads=over_declared_reads,
        over_declared_writes=over_declared_writes,
    )


# ---------------------------------------------------------------------------
# Operation rewrite (migration entry point)
# ---------------------------------------------------------------------------


def derive_operation(
    op: Any,
    *,
    dim_positions: Optional[Mapping[str, int]] = None,
    dim_sizes: Optional[Mapping[str, int]] = None,
    head_dim: int = 64,
):
    """Return a copy of ``op`` with ``reads`` / ``writes`` set to the derived sets.

    This is the migration entry point for the
    ``Operation.derive_reads_writes=True`` opt-in: a per-op flag that says
    "drop the manual ``reads`` / ``writes`` annotation; let the IR speak
    for itself". The returned ``Operation`` is otherwise identical to the
    input.

    Note: when the op has no IR (``ir_available=False``), the returned
    operation gets *empty* reads/writes. This is intentional — an op
    with no IR has no derivation surface, and silently retaining the
    manual annotation would mask the missing IR.
    """
    from .layer_compiler import Operation  # local import: avoid cycle

    if not isinstance(op, Operation):
        raise TypeError(
            "derive_operation: argument must be an Operation, got "
            f"{type(op).__name__}"
        )

    derived = derive_op_reads_writes_from_rules(
        op,
        dim_positions=dim_positions,
        dim_sizes=dim_sizes,
        head_dim=head_dim,
    )

    # Build a new Operation by reusing the original op's fields. We only
    # override ``reads`` / ``writes`` — every other field (kind, bake_fn,
    # compiler_ir, claims, phase, etc.) is preserved.
    import dataclasses

    return dataclasses.replace(
        op,
        reads=set(derived.reads),
        writes=set(derived.writes),
    )
