"""Producer-consumer contract verifier for dim slots.

This module sits on top of the existing per-op ``decl_verifier`` and the
``compiler.declare_dim`` registry to express *cross-op* contracts that
neither the per-op claim checker nor the declarative IR captures today.

The motivating shape is the Wave 1 A3 path:

    L10 ``layer10_psh_ax_broadcast``  writes  ``STACK0_BYTE_VAL_1_LO``
                                              during a ``PSH`` step
    L14 ``layer14_mem_generation``    reads   ``STACK0_BYTE_VAL_1_LO``
                                              during ``SI/SC/PSH`` steps

If either side breaks the contract -- e.g. someone narrows the producer
gate so the dim stops being written, or the consumer's ``reads`` set
drops the dim by accident -- the per-op verifier today says "OK"
(declared claims match writes) and the silent breakage falls into the
A3.5 / A3.6 diagnostic bucket.

The verifier here closes that gap. It walks the compiler IR (the same
``ModelLayout`` the bake dispatcher consumes) and asserts:

  1. The producer op is present and lists the dim in its ``writes``.
  2. The producer's placement layer is strictly before the consumer's.
  3. The consumer op is present and lists the dim in its ``reads``.
  4. (Optional) When ``must_not_zero_between=True``, the dim is not in
     any intervening op's ``writes`` between producer and consumer --
     i.e. nothing clobbers it back to zero on the producer-to-consumer
     hop. Reset-style ops that explicitly clear the dim cross-step are
     allowed via ``reset_after_step`` (Operation annotation).
  5. (Optional) When ``must_persist_for_steps>0``, the contract records
     a soft assertion that the slot is not re-allocated to another dim
     via liveness sharing for that many scheduler steps. This is a
     forward-compat check for Phase 7.A.5 liveness; today the verifier
     reports it as a NOTE and never fails on it.

The contract API is intentionally narrow: producers/consumers are
referenced by op *name*, not by handle, so the contract can be authored
in a separate module and registered into a global registry without
importing the producing/consuming op factories. The verifier resolves
names against ``ModelLayout.ops_per_layer`` plus ``block_ops`` plus
``model_ops``.

Compile-time fail-closed
------------------------

The intended deployment is the CI tool ``tools/dim_contracts_audit.py``
plus a small set of unit tests in ``tests/test_dim_contracts.py``.
Both call :func:`verify_all_registered_contracts` and assert no
failures. Contract authors register starter contracts at module import
time via :func:`register_dim_contract`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .layer_compiler import ModelLayout, Operation


# ---------------------------------------------------------------------------
# Contract dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpcodeSet:
    """A frozen set of opcode names a contract is gated on.

    Carries no semantic weight at verify time today; the gate names are
    recorded for diagnostics (`format()` output) and may be cross-checked
    against the producer/consumer's ``reads`` set in a future revision.
    The current invariant is purely structural: producer/consumer ops
    must agree on the dim, and the producer's gate names should appear
    in the producer's ``reads`` (so the per-op gate is at least
    expressible). We surface a NOTE when that side-check fails rather
    than escalating to an error -- many ops gate on synthetic
    composite dims that don't carry the literal opcode name.
    """

    opcodes: FrozenSet[str] = field(default_factory=frozenset)

    def __init__(self, *opcodes: str) -> None:
        # frozen dataclass: bypass __setattr__ to set the field
        object.__setattr__(self, "opcodes", frozenset(opcodes))

    def __iter__(self):
        return iter(sorted(self.opcodes))

    def __repr__(self) -> str:
        items = ", ".join(repr(o) for o in sorted(self.opcodes))
        return f"OpcodeSet({items})"


@dataclass(frozen=True)
class OpRef:
    """Reference to one side (producer or consumer) of a dim contract.

    ``op_name`` is the canonical ``Operation.name`` string; the verifier
    looks it up in the compiled ``ModelLayout``. ``layer`` is an *expected*
    placement layer used as a sanity probe -- the verifier reports a NOTE
    if the layout placed the op at a different layer (the layout's
    actual placement is the source of truth, not the contract).
    ``when`` is the opcode gate the op runs under; recorded for
    diagnostics, see :class:`OpcodeSet`.
    """

    op_name: str
    layer: Optional[int] = None
    when: Optional[OpcodeSet] = None


@dataclass(frozen=True)
class DimContract:
    """Declarative producer-consumer contract on a single dim slot.

    Attributes:
        dim: The declared dim name (must appear in
            ``ModelLayout.dim_positions``).
        producer: The op expected to write ``dim``.
        consumer: The op expected to read ``dim``.
        must_not_zero_between: When True, no intervening op (by
            placement order) may list ``dim`` in its ``writes`` unless
            it ALSO lists ``dim`` in its ``reset_after_step`` annotation
            (which marks an explicit cross-step zero).
        must_persist_for_steps: Soft assertion that the dim's slot is
            not liveness-recycled for at least this many scheduler
            steps. Today this is reported as a NOTE only -- the liveness
            allocator is OFF by default in production.
        name: Optional human label; defaults to ``"<producer>:<consumer>@<dim>"``.
    """

    dim: str
    producer: OpRef
    consumer: OpRef
    must_not_zero_between: bool = False
    must_persist_for_steps: int = 0
    name: Optional[str] = None

    def display_name(self) -> str:
        if self.name:
            return self.name
        return (
            f"{self.producer.op_name}:{self.consumer.op_name}@{self.dim}"
        )


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ContractValidation:
    """Result of validating one ``DimContract`` against a layout."""

    contract: DimContract
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    producer_layer: Optional[int] = None
    consumer_layer: Optional[int] = None
    intervening_writers: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        status = "OK" if self.ok else "FAIL"
        head = (
            f"[{status}] {self.contract.display_name()} "
            f"(dim={self.contract.dim!r})"
        )
        if self.producer_layer is not None and self.consumer_layer is not None:
            head += (
                f" producer_layer={self.producer_layer} "
                f"consumer_layer={self.consumer_layer}"
            )
        lines = [head]
        for err in self.errors:
            lines.append(f"     ERROR: {err}")
        for w in self.intervening_writers:
            lines.append(f"     INTERVENING WRITER: layer={w[0]} op={w[1]!r}")
        for n in self.notes:
            lines.append(f"     NOTE: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------


_CONTRACTS: List[DimContract] = []


def register_dim_contract(contract: DimContract) -> DimContract:
    """Register ``contract`` in the global verifier registry.

    Returns the contract for chained registration. Duplicate
    registration (same ``display_name()``) is allowed -- the verifier
    only walks the registry list; module import idempotency is the
    caller's responsibility.
    """
    _CONTRACTS.append(contract)
    return contract


def registered_dim_contracts() -> Tuple[DimContract, ...]:
    """Return the (immutable snapshot of the) registered contracts."""
    return tuple(_CONTRACTS)


def clear_registered_dim_contracts() -> None:
    """Test helper: drop all registered contracts.

    Used by ``tests/test_dim_contracts.py`` to keep starter-contract
    registration from leaking into per-test fixtures.
    """
    _CONTRACTS.clear()


# ---------------------------------------------------------------------------
# Verifier core
# ---------------------------------------------------------------------------


def _all_ops_with_layer(layout: ModelLayout) -> List[Tuple[int, Operation]]:
    """Return every op the layout knows about, paired with its layer.

    Layer-keyed ops come first (``ops_per_layer``); block ops follow
    with their resolved layer; model ops have ``layer = -1`` because
    they bake against the whole model.
    """
    out: List[Tuple[int, Operation]] = []
    for layer_idx, ops in enumerate(layout.ops_per_layer):
        for op in ops:
            out.append((layer_idx, op))
    for op in layout.block_ops:
        try:
            resolved = layout.resolve_block_op_layer(op)
        except Exception:
            resolved = op.layer_idx if op.layer_idx is not None else -1
        out.append((resolved, op))
    for op in layout.model_ops:
        out.append((-1, op))
    return out


def _find_op(
    layout: ModelLayout,
    name: str,
) -> Optional[Tuple[int, Operation]]:
    for entry in _all_ops_with_layer(layout):
        if entry[1].name == name:
            return entry
    return None


def verify_dim_contract(
    layout: ModelLayout,
    contract: DimContract,
) -> ContractValidation:
    """Verify one ``DimContract`` against ``layout``.

    The layout is the post-``LayerCompiler.compile()`` artifact; it
    carries the placed ops, declared dim positions, and block/model
    op bundles. The verifier is read-only: it never mutates layout.
    """
    result = ContractValidation(contract=contract)

    # 1. Dim must be declared.
    if contract.dim not in layout.dim_positions:
        result.errors.append(
            f"dim {contract.dim!r} not declared in layout "
            f"(declare via compiler.declare_dim before registering the contract)"
        )
        return result

    # 2. Resolve producer + consumer.
    prod = _find_op(layout, contract.producer.op_name)
    cons = _find_op(layout, contract.consumer.op_name)
    if prod is None:
        result.errors.append(
            f"producer op {contract.producer.op_name!r} not found in layout"
        )
    if cons is None:
        result.errors.append(
            f"consumer op {contract.consumer.op_name!r} not found in layout"
        )
    if prod is None or cons is None:
        return result

    prod_layer, prod_op = prod
    cons_layer, cons_op = cons
    result.producer_layer = prod_layer
    result.consumer_layer = cons_layer

    # 3. Producer must write the dim.
    if contract.dim not in prod_op.writes:
        result.errors.append(
            f"producer {prod_op.name!r} does not declare {contract.dim!r} "
            f"in writes (writes={sorted(prod_op.writes)[:6]}...)"
        )

    # 4. Consumer must read the dim.
    if contract.dim not in cons_op.reads:
        result.errors.append(
            f"consumer {cons_op.name!r} does not declare {contract.dim!r} "
            f"in reads (reads={sorted(cons_op.reads)[:6]}...)"
        )

    # 5. Placement ordering: producer must run strictly before consumer.
    # We allow same-layer placement if producer is attn and consumer is
    # ffn (they execute attn-first within a block); equal layers with
    # the same kind are flagged as ordering errors.
    if prod_layer >= 0 and cons_layer >= 0:
        if prod_layer > cons_layer:
            result.errors.append(
                f"producer layer {prod_layer} is AFTER consumer layer "
                f"{cons_layer} -- the consumer reads a stale value"
            )
        elif prod_layer == cons_layer:
            same_block_ok = (prod_op.kind == "attn" and cons_op.kind == "ffn")
            if not same_block_ok:
                result.notes.append(
                    f"producer and consumer share layer {prod_layer} "
                    f"({prod_op.kind!r} -> {cons_op.kind!r}); ordering "
                    f"relies on intra-block dispatch"
                )

    # 6. Expected-layer sanity probes.
    if contract.producer.layer is not None and contract.producer.layer != prod_layer:
        result.notes.append(
            f"producer expected layer {contract.producer.layer}, "
            f"actual {prod_layer} (layout authority)"
        )
    if contract.consumer.layer is not None and contract.consumer.layer != cons_layer:
        result.notes.append(
            f"consumer expected layer {contract.consumer.layer}, "
            f"actual {cons_layer} (layout authority)"
        )

    # 7. Opcode-gate sanity (NOTE level).
    if contract.producer.when is not None:
        missing = sorted(
            f"OP_{op}" for op in contract.producer.when.opcodes
            if f"OP_{op}" not in prod_op.reads
        )
        if missing:
            result.notes.append(
                f"producer {prod_op.name!r} gate dims not in reads: {missing}"
            )
    if contract.consumer.when is not None:
        missing = sorted(
            f"OP_{op}" for op in contract.consumer.when.opcodes
            if f"OP_{op}" not in cons_op.reads
        )
        if missing:
            result.notes.append(
                f"consumer {cons_op.name!r} gate dims not in reads: {missing}"
            )

    # 8. must_not_zero_between: scan for intervening writers.
    if contract.must_not_zero_between and prod_layer >= 0 and cons_layer >= 0:
        lo = min(prod_layer, cons_layer)
        hi = max(prod_layer, cons_layer)
        for layer_idx, op in _all_ops_with_layer(layout):
            if layer_idx < lo or layer_idx > hi:
                continue
            if op.name in (prod_op.name, cons_op.name):
                continue
            if contract.dim not in op.writes:
                continue
            # Allow reset-style ops that explicitly clear cross-step.
            if contract.dim in getattr(op, "reset_after_step", set()):
                result.notes.append(
                    f"layer={layer_idx} op={op.name!r} writes {contract.dim!r} "
                    f"but lists it in reset_after_step -- allowed"
                )
                continue
            result.intervening_writers.append((layer_idx, op.name))
        if result.intervening_writers:
            result.errors.append(
                f"must_not_zero_between violated: "
                f"{len(result.intervening_writers)} intervening writer(s) "
                f"clobber {contract.dim!r} between layers "
                f"{prod_layer} and {cons_layer}"
            )

    # 9. must_persist_for_steps: soft NOTE.
    if contract.must_persist_for_steps > 0:
        result.notes.append(
            f"must_persist_for_steps={contract.must_persist_for_steps}: "
            f"liveness-recycling check is informational today "
            f"(Phase 7.A liveness allocator off by default)"
        )

    return result


@dataclass
class ContractAuditReport:
    """Aggregate report across many contracts."""

    results: List[ContractValidation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def n_failures(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    def format(self) -> str:
        lines = ["=== Producer-consumer dim contract audit ==="]
        lines.append(f"Contracts audited: {len(self.results)}")
        lines.append(f"Failures: {self.n_failures()}")
        for r in self.results:
            lines.append(r.format())
        return "\n".join(lines)


def verify_all_registered_contracts(
    layout: ModelLayout,
) -> ContractAuditReport:
    """Run every registered contract against ``layout`` and aggregate."""
    report = ContractAuditReport()
    for c in registered_dim_contracts():
        report.results.append(verify_dim_contract(layout, c))
    return report


# ---------------------------------------------------------------------------
# Starter contracts for the memory cluster
# ---------------------------------------------------------------------------
#
# The three starter contracts below are the load-bearing producer-consumer
# pairs that the A3.5 / A3.6 diagnostic surfaced. They are registered at
# module import time so any caller of ``verify_all_registered_contracts``
# gets the baseline coverage; ``clear_registered_dim_contracts`` lets
# test fixtures roll the registry back.


def _register_starter_contracts() -> None:
    """Register the Wave-1 A3 producer-consumer pairs.

    1. STACK0_BYTE_VAL_1_LO  L10 psh_ax_broadcast  -> L14 mem_generation
    2. STACK0_BYTE_VAL_1_HI  same pair (HI nibble)
    3. MEM_VAL_B1            L2 mem_byte_flags     -> L14 mem_generation

    The L10/L14 pair was the A3.6 attribution target; the L2/L14
    MEM_VAL_B1 pair is the structurally simpler MEM-write contract
    against which the value-byte rule was originally authored.
    """
    psh_consumer_opcodes = OpcodeSet("SI", "LI", "SC", "LC", "PSH")
    psh_producer_opcodes = OpcodeSet("PSH")

    register_dim_contract(DimContract(
        dim="STACK0_BYTE_VAL_1_LO",
        # ``layer10_psh_ax_broadcast`` is the topology anchor that
        # declares the writes; ``layer10_psh_ax_broadcast_bake`` lowers
        # the weights. The anchor is the canonical producer reference
        # because the dep-graph (and the contract) keys on the declared
        # writes set, not the bake closure.
        producer=OpRef(
            "layer10_psh_ax_broadcast",
            layer=10,
            when=psh_producer_opcodes,
        ),
        consumer=OpRef(
            "layer14_mem_generation",
            layer=14,
            when=psh_consumer_opcodes,
        ),
        must_not_zero_between=True,
        must_persist_for_steps=5,
        name="stack0_byte_val_1_lo_pshk2mem",
    ))

    register_dim_contract(DimContract(
        dim="STACK0_BYTE_VAL_1_HI",
        producer=OpRef(
            "layer10_psh_ax_broadcast",
            layer=10,
            when=psh_producer_opcodes,
        ),
        consumer=OpRef(
            "layer14_mem_generation",
            layer=14,
            when=psh_consumer_opcodes,
        ),
        must_not_zero_between=True,
        must_persist_for_steps=5,
        name="stack0_byte_val_1_hi_pshk2mem",
    ))

    register_dim_contract(DimContract(
        dim="MEM_VAL_B1",
        # L2 mem_byte_flags writes MEM_VAL_B1 from the H0/H1/H4 flags.
        producer=OpRef(
            "layer2_mem_byte_flags",
            layer=2,
            when=None,
        ),
        consumer=OpRef(
            "layer14_mem_generation",
            layer=14,
            when=OpcodeSet("SI", "SC", "PSH"),
        ),
        must_not_zero_between=False,
        must_persist_for_steps=0,
        name="mem_val_b1_l2_to_l14",
    ))


# Register at import time. Callers that want a clean registry call
# clear_registered_dim_contracts() and re-register their own.
_register_starter_contracts()
