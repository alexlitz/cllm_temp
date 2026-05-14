"""Layer-allocation compiler MVP.

Takes a list of declarative operations (each with reads/writes/kind/bake_fn) and
produces a ModelLayout that specifies:

    - d_model (auto-computed from peak live-dim usage)
    - n_layers (auto-computed from longest dependency chain)
    - which operation lives at which layer
    - which dim each name maps to

This is a thin layer on top of the existing AutoAllocator (which handles dim
positions). The new piece here is layer assignment by dependency analysis.

NOTHING about this MVP is wired to production yet. It's standalone and tested
in isolation. Production weight setting still goes through vm_step.set_vm_weights.

Example:
    compiler = LayerCompiler()
    compiler.declare_dim("MARK_PC", size=1)
    compiler.declare_dim("EMBED_LO", size=16)
    compiler.declare_dim("OUTPUT_LO", size=16)

    compiler.add_op(Operation(
        name="pc_carry_forward",
        reads={"MARK_PC"},
        writes={"EMBED_LO"},
        kind="attn",
        bake_fn=lambda module, dims, S: ...,
    ))
    compiler.add_op(Operation(
        name="pc_increment",
        reads={"EMBED_LO"},
        writes={"OUTPUT_LO"},
        kind="ffn",
        bake_fn=lambda module, dims, S: ...,
    ))

    layout = compiler.compile()
    # layout.d_model = 33  (computed from dim sizes + alignment)
    # layout.n_layers = 2  (carry_forward at layer 0, increment at layer 1)
    # layout.ops_at(0) = [pc_carry_forward]
    # layout.ops_at(1) = [pc_increment]
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple


# Allowed `scope` values for `Operation.claims`. Each scope tags a class of
# weight slots whose ownership can collide across bakes:
#
#   attn_W_v / attn_W_k / attn_W_q / attn_W_o : a single row of an attention
#       module's W_v / W_k / W_q / W_o projection. Row index encodes
#       (head_idx * HD) + slot, so the identifier is "<head_idx>_<slot>".
#       Today's L5 head 6 collision (commit c1a5398) lived here: both the
#       deprecated `_set_layer5_fetch` head 6 V relay and
#       `_set_function_call_weights`' ENT BP→TEMP relay claimed rows
#       6*HD+1 .. 6*HD+16 on `attn5.W_v`.
#   ffn_W_up / ffn_W_down / ffn_W_gate : one hidden-unit row of an FFN's
#       W_up / W_down / W_gate matrix. Identifier is the hidden-unit index
#       as a string, e.g. "1700".
#   embed_row : one row of the token-embedding table. Identifier is the
#       token-id as a string.
#
# These tags are conventions enforced by the registry; bake authors pick the
# tag matching the matrix slot they overwrite. The registry only checks for
# duplicate (layer_idx, scope, identifier) keys — it does not validate that
# the tag matches the underlying matrix.
ALLOWED_CLAIM_SCOPES = frozenset({
    "attn_W_v", "attn_W_k", "attn_W_q", "attn_W_o",
    "ffn_W_up", "ffn_W_down", "ffn_W_gate",
    "embed_row",
})


ALLOWED_DECLARATIVE_AUTHORITY = frozenset({
    "declarative",
    "spec_generated",
    "legacy_wrapper",
    "structural_model",
    "topology_anchor",
})


@dataclass
class Operation:
    """A single declarative operation that the compiler can place at any layer.

    `reads` and `writes` are dim names (declared via `compiler.declare_dim(...)`).
    `kind` is "attn", "ffn", or "block":
      - "attn": programs the block's attention module
      - "ffn": programs the block's feed-forward module
      - "block": targets the whole TransformerBlock (e.g., wrap ffn in a
        composite module). Block ops bypass dependency analysis and are
        pinned to `layer_idx`; they fire after attn/ffn ops at that layer.
    `bake_fn(module, dim_positions, S)` is invoked at compile time to actually
    write the weights. For block ops, `module` is the whole TransformerBlock.
    `declarative_bake_fn(module, dim_positions, S)` is the opt-in
    declarations-only implementation. When declarations-only baking is enabled,
    the dispatcher refuses to call `bake_fn` and only runs this generator (or
    skips `topology_anchor` ops, which intentionally write no weights).

    `phase` is an optional ordering hint. When the dep graph has ambiguity
    (e.g., two ops both read+write the same dim at different positions, which
    the dim-name-only dep model can't distinguish), the compiler uses phase to
    break ties: an edge u→v in the dep graph is dropped if u.phase > v.phase.
    Smaller phase = earlier. Use the original `_set_layerN_*` layer number as
    the phase for migrated ops to preserve hand-set order.

    `layer_idx` is required when kind="block" — it pins the op to that exact
    layer index. Ignored for attn/ffn kinds.

    `migrated` marks an op as having claimed its bake from the legacy
    set_vm_weights path. Both `build_model_from_layout` and the legacy bake's
    migrated-op dispatch hook respect this flag: when True the op runs in the
    new path, and the legacy path skips its corresponding inline bake.
    """

    name: str
    reads: Set[str]
    writes: Set[str]
    kind: str  # "attn", "ffn", "block", or "model"
    bake_fn: Callable
    declarative_bake_fn: Optional[Callable] = None
    phase: Optional[float] = None
    # For kind="block" or "model" ops, the layer_idx the op targets (if
    # block-scoped) or None (if it operates on the whole model).
    layer_idx: Optional[int] = None
    # When True, build_model_from_layout dispatches this op's bake_fn even when
    # legacy_bake is present. Used to incrementally migrate ops out of
    # legacy_bake — flip to True once the corresponding direct call is removed
    # from set_vm_weights so the op runs only via the compiler.
    migrated: bool = False
    # Op-reference binding for kind="block" ops: when set, the block op binds
    # to the layer that the named (attn/ffn) op was placed at by the compiler.
    # Resolved in build_model_from_layout from layout.ops_per_layer. Overrides
    # `layer_idx` when both are provided. This decouples block ops from
    # hardcoded layer numbers — they follow whatever layer the compiler picks
    # for the referenced op.
    target_op_name: Optional[str] = None
    # Dim-ownership claims. Each tuple is
    # ``(layer_idx, scope, identifier, column)``.
    # See ALLOWED_CLAIM_SCOPES for the legal scope strings. The compiler's
    # `_detect_claim_collisions` cross-checks claims at compile time and
    # warns when two ops claim the same
    # ``(layer_idx, scope, identifier, column)`` 4-tuple.
    #
    # `column` is the input-dim coordinate the claim writes into:
    #   - For attn scopes (``attn_W_v/k/q/o``): a string identifying the
    #     input dim + offset, e.g. ``"CLEAN_EMBED_LO+0"`` or
    #     ``"EMBED_HI+15"``. This distinguishes two ops that both write the
    #     same row but at distinct columns — the case the row-only registry
    #     produced false positives for.
    #   - For ffn scopes (``ffn_W_up/down/gate``): same convention, the
    #     input-dim name + offset.
    #   - For ``embed_row`` scope: ``None`` (one row is one unit; no
    #     column granularity).
    #
    # Backwards-compatible legacy 3-tuple ``(layer_idx, scope, identifier)``
    # is accepted at ``add_op`` time and auto-promoted to a 4-tuple with
    # ``column=None``. Existing 3-tuple call sites continue to work; the
    # collision detector treats ``column=None`` claims as wildcard for the
    # purposes of grouping (so two legacy 3-tuple writes to the same row
    # still collide as before).
    #
    # Defaults to an empty set so existing ops remain back-compat (opt-in).
    # See c4_release/docs/DIM_OWNERSHIP_REGISTRY.md for the bake-author API.
    claims: Set[Tuple[int, str, str, Optional[str]]] = field(default_factory=set)
    # Max FFN-hidden unit index (exclusive) this op writes at its layer. Used
    # by the compiler to pre-size each block's PureFFN hidden_dim per-layer
    # instead of allocating 4096 units everywhere and trimming after bake via
    # ``_right_size_ffns``. When None (default), the op is treated as "unknown
    # width" and the block falls back to the default ffn_hidden (4096) —
    # ``_right_size_ffns`` still trims those blocks. As authors annotate their
    # ops, blocks accumulate exact widths and skip trim allocation.
    #
    # Set this to the number of FFN hidden units the op's bake_fn writes into
    # at its target layer (e.g., L0's phase_a_ffn writes units 0..6 -> set to
    # 7). For ops that don't touch FFN (pure attn ops, model ops writing
    # alibi_slopes/head weights, etc.) leave as None.
    #
    # For ``kind="model"`` ops that write to a specific block's FFN (e.g.
    # ``function_call_weights`` writing L6 FFN units 1700-2158), set
    # ``layer_idx`` to the target block index. The compiler folds the
    # annotation into ``ModelLayout.ffn_widths`` for that layer so the
    # dynamic-FFN allocator pre-sizes it correctly. ``layer_idx`` on a
    # model op is otherwise informational — model ops are dispatched
    # against the whole model regardless of this field.
    ffn_units_used: Optional[int] = None
    # Residual-dim staleness invariants (Phase 3 / Agent G of
    # ARCH_LEAKAGE_FIX_PLAN.md). Both fields are opt-in (default empty):
    #
    #   ``produces``: maps a residual dim name to the *register* name whose
    #       fresh value this op writes. E.g. the L8 attn head 6 (commit
    #       3d1b700) declares
    #       ``produces={"AX_CARRY_LO": "AX_byte0",
    #                   "AX_CARRY_HI": "AX_byte0"}``
    #       because it writes the prev-step AX byte 0 value into AX_CARRY_LO
    #       and AX_CARRY_HI at the current AX marker.
    #
    #   ``consumes_fresh``: maps a residual dim name to the register the op
    #       expects to read a fresh in-step value from. The analyzer warns
    #       when a dim is declared ``consumes_fresh`` but no earlier-phase
    #       op in the same step ``produces`` the same (dim, register).
    #       Use this only for in-step freshness — ops that rely solely on
    #       cross-step values (e.g. L3 head 1's prev-step EMBED_LO/HI relay
    #       into AX_CARRY) should leave ``consumes_fresh`` empty.
    #
    # See c4_release/docs/STALENESS_INVARIANTS.md for the bake-author API
    # and the canonical AX_CARRY example.
    produces: Dict[str, str] = field(default_factory=dict)
    consumes_fresh: Dict[str, str] = field(default_factory=dict)
    # Tier B declarative-verifier annotations. These are opt-in and default
    # to no-op values so existing operation declarations remain valid:
    #
    #   alibi_slopes: head_idx -> expected attn.alibi_slopes value for ops
    #       that explicitly write a layer's ALiBi slope buffer.
    #   postcondition: residual cell -> invariant name, e.g.
    #       {"OUTPUT_LO": "monotonic_non_decreasing"}.
    #   step_idx: allowed VM steps for an op, using None/"every",
    #       "after_first", or a set of 0-indexed step numbers.
    alibi_slopes: Dict[int, float] = field(default_factory=dict)
    postcondition: Dict[str, str] = field(default_factory=dict)
    step_idx: Optional[object] = None
    # Tier C discoverability annotations. These are bookkeeping-only: smoke
    # coverage and spec coverage audits consume them, but they do not affect
    # compile placement or baking. ``compaction_safe`` is default-true so ops
    # opt out only when a known FFN footprint must remain in the shared expert.
    smoke_tests: Set[str] = field(default_factory=set)
    spec_section: Optional[str] = None
    compaction_safe: bool = True
    # Where this op's bake authority currently comes from. ``None`` means the
    # op has not been audited/classified yet. ``declarative`` and
    # ``spec_generated`` are authoritative declarative paths;
    # ``structural_model`` marks compiler-owned whole-model structural passes;
    # ``topology_anchor`` marks no-op graph-shape anchors whose paired block
    # op owns the actual bake; ``legacy_wrapper`` marks an opaque wrapper
    # around legacy bake code.
    declarative_authority: Optional[str] = None

    def __hash__(self):
        return hash(self.name)


class DeclarationsOnlyBakeError(RuntimeError):
    """Raised when declarations-only mode reaches an imperative-only op."""

    def __init__(self, unsupported_ops: List[str]):
        self.unsupported_ops = tuple(sorted(unsupported_ops))
        sample = ", ".join(self.unsupported_ops[:20])
        if len(self.unsupported_ops) > 20:
            sample += f", ... (+{len(self.unsupported_ops) - 20} more)"
        super().__init__(
            "Declarations-only bake cannot run because some dispatched ops do "
            "not expose declarative_bake_fn and are not topology anchors: "
            f"{sample}"
        )


def operation_supports_declarations_only(op: Operation) -> bool:
    """Return whether ``op`` can run without its imperative ``bake_fn``."""

    return (
        op.declarative_bake_fn is not None
        or op.declarative_authority == "topology_anchor"
    )


def validate_declarations_only_ops(ops: List[Operation]):
    """Raise when any op in dispatch order lacks a declarations-only bake."""

    unsupported = [
        op.name for op in ops if not operation_supports_declarations_only(op)
    ]
    if unsupported:
        raise DeclarationsOnlyBakeError(unsupported)


def dispatch_operation_bake(op: Operation, target, dim_positions, S, *,
                            declarations_only: bool = False):
    """Dispatch one operation through either the normal or declarations-only path."""

    if not declarations_only:
        op.bake_fn(target, dim_positions, S)
        return
    if op.declarative_bake_fn is not None:
        op.declarative_bake_fn(target, dim_positions, S)
        return
    if op.declarative_authority == "topology_anchor":
        return
    raise DeclarationsOnlyBakeError([op.name])


@dataclass
class ModelLayout:
    """Output of LayerCompiler.compile().

    Attributes:
        d_model: total residual-stream dimension, derived from peak live-dim usage
        n_layers: total number of transformer blocks needed
        ops_per_layer: ops_per_layer[i] is the list of ops the compiler placed at layer i
        dim_positions: map of dim_name -> start position in residual stream
        dim_sizes: map of dim_name -> size (so position range is [pos, pos+size))
        block_ops: block-level ops, each carrying its own layer_idx
        model_ops: model-level post-pass ops (head, embedding, right-size, etc.)
        ffn_widths: per-block FFN hidden_dim, computed as the max
            ``ffn_units_used`` across all ops targeting each block. Layers
            without any annotated FFN op are omitted (caller treats as the
            default ffn_hidden — 4096 by default). Used by ``compile_full_vm``
            to pre-size ``PureFFN.hidden_dim`` per-block, avoiding the
            allocate-4096-then-trim-via-``_right_size_ffns`` overhead.
    """

    d_model: int
    n_layers: int
    ops_per_layer: List[List[Operation]]
    dim_positions: Dict[str, int]
    dim_sizes: Dict[str, int]
    block_ops: List[Operation] = field(default_factory=list)
    model_ops: List[Operation] = field(default_factory=list)
    ffn_widths: Dict[int, int] = field(default_factory=dict)

    def ops_at(self, layer: int) -> List[Operation]:
        return self.ops_per_layer[layer]

    def dim_range(self, name: str) -> range:
        start = self.dim_positions[name]
        return range(start, start + self.dim_sizes[name])

    def resolve_block_op_layer(self, op: 'Operation') -> int:
        """Return the layer index a kind="block" op binds to.

        `target_op_name` (op-reference binding) takes precedence over the
        legacy `layer_idx` field. When `target_op_name` is set, the op binds
        to whichever layer the compiler placed the named attn/ffn op at.

        Raises:
            ValueError: if `target_op_name` doesn't match any op in the layout.
        """
        if op.target_op_name is not None:
            for layer_idx, ops_at_layer in enumerate(self.ops_per_layer):
                for placed in ops_at_layer:
                    if placed.name == op.target_op_name:
                        return layer_idx
            raise ValueError(
                f"Block op {op.name!r} target_op_name "
                f"{op.target_op_name!r} not found in layout "
                f"(must reference an attn/ffn op)"
            )
        return op.layer_idx


class LayerCompiler:
    """Layer-allocation compiler MVP.

    Algorithm:
      1. Build dependency graph: edge u -> v if v reads any dim that u writes.
      2. Topological sort.
      3. For each op in topo order, assign to earliest layer where all deps satisfied.
         Multiple ops can share a layer if they don't depend on each other AND
         they target different module kinds (attn vs ffn) — a single transformer
         block has one attention and one FFN per layer.
      4. Allocate dims: simple bump-pointer (no liveness reuse for MVP).
    """

    def __init__(self):
        self.ops: List[Operation] = []
        self.dims: Dict[str, int] = {}  # name -> size
        self._op_by_name: Dict[str, Operation] = {}
        # Block-level and model-level ops are bake-only; they don't participate
        # in dim-position allocation, so they're held separately.
        self.block_ops: List[Operation] = []
        self.model_ops: List[Operation] = []

    def declare_dim(self, name: str, size: int, pinned: Optional[int] = None):
        """Declare a dim with optional pinned start position.

        When pinned is given, the compiler MUST place the dim at that exact
        position (used for backward-compat with _SetDim aliasing where multiple
        names share the same physical position).

        When pinned is None, the dim is bump-pointer-allocated.
        """
        if name in self.dims and self.dims[name] != size:
            raise ValueError(
                f"Dim {name!r} already declared with size {self.dims[name]}; "
                f"got {size}"
            )
        self.dims[name] = size
        if pinned is not None:
            if not hasattr(self, "_pinned"):
                self._pinned: Dict[str, int] = {}
            self._pinned[name] = pinned

    def add_op(self, op: Operation):
        if op.kind not in ("attn", "ffn", "block", "model"):
            raise ValueError(
                f"op.kind must be 'attn', 'ffn', 'block', or 'model'; got {op.kind!r}"
            )
        if op.name in self._op_by_name:
            raise ValueError(f"Operation name {op.name!r} already added")
        # Validate that every read/write is declared.
        for d in op.reads | op.writes:
            if d not in self.dims:
                raise ValueError(
                    f"Op {op.name!r} references undeclared dim {d!r}"
                )
        # Validate staleness invariants (Phase 3 / Agent G of
        # ARCH_LEAKAGE_FIX_PLAN.md). Both ``produces`` and ``consumes_fresh``
        # map declared dim names to register identifiers.
        for fname, mapping in (("produces", op.produces),
                               ("consumes_fresh", op.consumes_fresh)):
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"Op {op.name!r} {fname} must be a dict; "
                    f"got {type(mapping).__name__}"
                )
            for dim_name, register in mapping.items():
                if not isinstance(dim_name, str):
                    raise ValueError(
                        f"Op {op.name!r} {fname} dim name must be str; "
                        f"got {dim_name!r}"
                    )
                if not isinstance(register, str):
                    raise ValueError(
                        f"Op {op.name!r} {fname}[{dim_name!r}] register "
                        f"must be str; got {register!r}"
                    )
                if dim_name not in self.dims:
                    raise ValueError(
                        f"Op {op.name!r} {fname} references undeclared dim "
                        f"{dim_name!r}"
                    )
        # Validate Tier B annotations. Postcondition cell names are resolved
        # by the detector because they may include an index suffix like
        # ``OUTPUT_LO[3]``.
        if not isinstance(op.alibi_slopes, dict):
            raise ValueError(
                f"Op {op.name!r} alibi_slopes must be a dict; "
                f"got {type(op.alibi_slopes).__name__}"
            )
        for head_idx, slope in op.alibi_slopes.items():
            if not isinstance(head_idx, int):
                raise ValueError(
                    f"Op {op.name!r} alibi_slopes key must be int "
                    f"(head index); got {head_idx!r}"
                )
            if not isinstance(slope, (int, float)):
                raise ValueError(
                    f"Op {op.name!r} alibi_slopes[{head_idx!r}] must be "
                    f"int|float; got {slope!r}"
                )
        if not isinstance(op.postcondition, dict):
            raise ValueError(
                f"Op {op.name!r} postcondition must be a dict; "
                f"got {type(op.postcondition).__name__}"
            )
        for cell_name, invariant in op.postcondition.items():
            if not isinstance(cell_name, str):
                raise ValueError(
                    f"Op {op.name!r} postcondition key must be str; "
                    f"got {cell_name!r}"
                )
            if not isinstance(invariant, str):
                raise ValueError(
                    f"Op {op.name!r} postcondition[{cell_name!r}] must "
                    f"be str; got {invariant!r}"
                )
        if op.step_idx is not None:
            if isinstance(op.step_idx, str):
                if op.step_idx not in ("every", "after_first"):
                    raise ValueError(
                        f"Op {op.name!r} step_idx str must be 'every' "
                        f"or 'after_first'; got {op.step_idx!r}"
                    )
            elif isinstance(op.step_idx, set):
                for idx in op.step_idx:
                    if not isinstance(idx, int):
                        raise ValueError(
                            f"Op {op.name!r} step_idx set entries must "
                            f"be int; got {idx!r}"
                        )
            else:
                raise ValueError(
                    f"Op {op.name!r} step_idx must be None, a set of ints, "
                    f"'every', or 'after_first'; got "
                    f"{type(op.step_idx).__name__}"
                )
        if not isinstance(op.smoke_tests, set):
            raise ValueError(
                f"Op {op.name!r} smoke_tests must be a set; "
                f"got {type(op.smoke_tests).__name__}"
            )
        for entry in op.smoke_tests:
            if not isinstance(entry, str):
                raise ValueError(
                    f"Op {op.name!r} smoke_tests entry must be str; "
                    f"got {entry!r}"
                )
        if op.spec_section is not None and not isinstance(op.spec_section, str):
            raise ValueError(
                f"Op {op.name!r} spec_section must be str or None; "
                f"got {type(op.spec_section).__name__}"
            )
        if not isinstance(op.compaction_safe, bool):
            raise ValueError(
                f"Op {op.name!r} compaction_safe must be bool; "
                f"got {type(op.compaction_safe).__name__}"
            )
        if (
            op.declarative_authority is not None
            and op.declarative_authority not in ALLOWED_DECLARATIVE_AUTHORITY
        ):
            allowed = ", ".join(sorted(ALLOWED_DECLARATIVE_AUTHORITY))
            raise ValueError(
                f"Op {op.name!r} declarative_authority must be one of "
                f"{allowed} or None; got {op.declarative_authority!r}"
            )
        # Validate dim-ownership claims (if any). Accept legacy 3-tuple
        # ``(layer_idx, scope, identifier)`` and auto-promote to 4-tuple with
        # ``column=None`` for back-compat with pre-column-granularity ops.
        promoted: Set[Tuple[int, str, str, Optional[str]]] = set()
        for claim in op.claims:
            if not isinstance(claim, tuple) or len(claim) not in (3, 4):
                raise ValueError(
                    f"Op {op.name!r} has malformed claim {claim!r}; "
                    "expected (layer_idx, scope, identifier) or "
                    "(layer_idx, scope, identifier, column)"
                )
            if len(claim) == 3:
                layer_idx, scope, identifier = claim
                column: Optional[str] = None
            else:
                layer_idx, scope, identifier, column = claim
            if not isinstance(layer_idx, int):
                raise ValueError(
                    f"Op {op.name!r} claim {claim!r}: layer_idx must be int"
                )
            if scope not in ALLOWED_CLAIM_SCOPES:
                raise ValueError(
                    f"Op {op.name!r} claim {claim!r}: scope {scope!r} not in "
                    f"{sorted(ALLOWED_CLAIM_SCOPES)}"
                )
            if not isinstance(identifier, str):
                raise ValueError(
                    f"Op {op.name!r} claim {claim!r}: identifier must be str"
                )
            if column is not None and not isinstance(column, str):
                raise ValueError(
                    f"Op {op.name!r} claim {claim!r}: column must be str "
                    f"or None"
                )
            # ``embed_row`` scope is row-granular only: a column value
            # there is meaningless. Enforce the convention.
            if scope == "embed_row" and column is not None:
                raise ValueError(
                    f"Op {op.name!r} claim {claim!r}: scope 'embed_row' "
                    f"must have column=None (row-granular only)"
                )
            promoted.add((layer_idx, scope, identifier, column))
        op.claims = promoted
        self._op_by_name[op.name] = op
        if op.kind == "block":
            if op.layer_idx is None and op.target_op_name is None:
                raise ValueError(
                    f"Block-scoped op {op.name!r} must specify layer_idx or "
                    f"target_op_name"
                )
            self.block_ops.append(op)
        elif op.kind == "model":
            self.model_ops.append(op)
        else:
            self.ops.append(op)

    # --------------------------------------------------------------------
    # Compilation
    # --------------------------------------------------------------------

    def compile(self) -> ModelLayout:
        """Produce a ModelLayout from the declared ops and dims.

        Also runs the dim-ownership claim collision scan: if two ops have
        opted into `Operation.claims` and claim the same
        (layer_idx, scope, identifier) tuple, a warning is printed via
        `warnings.warn`. The scan is opt-in (each op's claims default to
        empty), so legacy ops without claims don't participate and a clean
        compile prints nothing.
        """
        # Run claim-collision scan first so warnings fire even if subsequent
        # compile stages raise (e.g., dependency cycle).
        self._detect_claim_collisions()
        # Run staleness-invariant scan: warn when a consumer declares a dim
        # as ``consumes_fresh`` but no earlier-phase op in the same step
        # produces the same dim+register. The analyzer needs the per-op
        # phase ordering, which is fully available at this point.
        self._detect_staleness_violations()

        # Block ops are pinned to layer_idx and skip dep analysis.
        attn_ffn_ops = [op for op in self.ops if op.kind != "block"]
        block_ops = [op for op in self.ops if op.kind == "block"]

        topo = self._topological_sort(attn_ffn_ops)
        layer_assignment = self._assign_layers(topo)
        for op in block_ops:
            layer_assignment[op.name] = op.layer_idx

        dim_positions = self._allocate_dims()

        n_layers = (max(layer_assignment.values()) + 1) if layer_assignment else 0
        # d_model = highest position + size; supports both pinned and bump-pointer.
        d_model = 0
        for name, pos in dim_positions.items():
            d_model = max(d_model, pos + self.dims[name])

        ops_per_layer: List[List[Operation]] = [[] for _ in range(n_layers)]
        # attn/ffn first, then block ops, so blocks bake last on their layer.
        for op in attn_ffn_ops:
            ops_per_layer[layer_assignment[op.name]].append(op)
        for op in block_ops:
            ops_per_layer[layer_assignment[op.name]].append(op)

        # n_layers must be at least max(block_ops.layer_idx) + 1 if any
        # block-scoped ops with explicit layer_idx exist; otherwise their
        # target block won't exist. Block ops bound via target_op_name follow
        # the referenced op's layer (resolved at build time from
        # layout.ops_per_layer), which by construction is < n_layers.
        # target_op_name takes precedence: if a block op has both
        # target_op_name AND layer_idx, the target_op_name path is used and
        # layer_idx is ignored.
        layer_idx_block_ops = [
            o for o in self.block_ops
            if o.layer_idx is not None and o.target_op_name is None
        ]
        if layer_idx_block_ops:
            n_layers = max(
                n_layers,
                max(o.layer_idx for o in layer_idx_block_ops) + 1,
            )

        # Model-ops are applied after all layer-ops; sort by phase so the
        # original hand-set order is preserved (smaller phase = earlier).
        model_ops = [op for op in self.ops if op.kind == "model"]
        model_ops.sort(key=lambda o: (o.phase if o.phase is not None else 0))

        # Aggregate per-block FFN widths from ``Operation.ffn_units_used``
        # annotations. Walk every op that lands at a specific layer (attn/ffn
        # via ``ops_per_layer``, block ops via their resolved layer) and
        # take the per-block max. Ops without an annotation contribute
        # nothing; blocks with no annotated op are omitted from the dict
        # so the caller falls back to the default ffn_hidden (4096) and
        # ``_right_size_ffns`` still trims them. This makes annotation a
        # purely incremental migration — partial coverage = partial speedup.
        ffn_widths: Dict[int, int] = {}
        for layer_idx, ops_at_layer in enumerate(ops_per_layer):
            for op in ops_at_layer:
                if op.ffn_units_used is not None:
                    prev = ffn_widths.get(layer_idx, 0)
                    if op.ffn_units_used > prev:
                        ffn_widths[layer_idx] = op.ffn_units_used
        # Block ops in ``self.block_ops`` may target layers via either
        # explicit ``layer_idx`` or (resolved later) ``target_op_name``. For
        # the latter we use the same op-name -> layer map we built above.
        op_layer = {
            op.name: layer_idx
            for layer_idx, ops_at_layer in enumerate(ops_per_layer)
            for op in ops_at_layer
        }
        for op in self.block_ops:
            if op.ffn_units_used is None:
                continue
            if op.target_op_name is not None:
                target_layer = op_layer.get(op.target_op_name)
                if target_layer is None:
                    # Block op references an op not in ops_per_layer; defer
                    # to the trim fallback (don't record a partial width).
                    continue
            else:
                target_layer = op.layer_idx
            prev = ffn_widths.get(target_layer, 0)
            if op.ffn_units_used > prev:
                ffn_widths[target_layer] = op.ffn_units_used
        # Model-level ops (kind="model") may also write to a specific block's
        # FFN — e.g. ``function_call_weights`` writes L6 FFN units 1700-2277.
        # When such an op declares ``layer_idx`` + ``ffn_units_used``, fold
        # it into the per-block width aggregate so the dynamic-FFN allocator
        # pre-sizes the block large enough. Model ops without a ``layer_idx``
        # (e.g. head/embedding bakes that touch every block) stay out of the
        # aggregate and rely on the legacy ``_right_size_ffns`` trim.
        for op in self.model_ops:
            if op.ffn_units_used is None or op.layer_idx is None:
                continue
            prev = ffn_widths.get(op.layer_idx, 0)
            if op.ffn_units_used > prev:
                ffn_widths[op.layer_idx] = op.ffn_units_used

        return ModelLayout(
            d_model=d_model,
            n_layers=n_layers,
            ops_per_layer=ops_per_layer,
            dim_positions=dim_positions,
            dim_sizes=dict(self.dims),
            block_ops=list(self.block_ops),
            model_ops=list(self.model_ops),
            ffn_widths=ffn_widths,
        )

    # --------------------------------------------------------------------
    # Dim-ownership claim collision detection (Phase 1, Agent B of
    # ARCH_LEAKAGE_FIX_PLAN.md). Opt-in: each Operation.claims defaults to
    # an empty set, so unannotated ops don't participate. As bake authors
    # add claims to their factories, the scan grows coverage. Collisions
    # warn via `warnings.warn` so they surface in test output without
    # breaking the legacy bake path.
    # --------------------------------------------------------------------

    def build_claim_registry(
        self,
    ) -> Dict[Tuple[int, str, str, Optional[str]], List[str]]:
        """Return registry:
        ``(layer_idx, scope, identifier, column) -> [op_name, ...]``.

        Aggregates `Operation.claims` across `self.ops`, `self.block_ops`,
        and `self.model_ops`. Used by `_detect_claim_collisions`; exposed
        publicly so tests / debugging tools can inspect the full claims
        graph without re-running compile.

        Claims are always 4-tuples here: ``add_op`` promotes legacy
        3-tuples to ``(layer, scope, identifier, None)`` at registration.
        """
        registry: Dict[Tuple[int, str, str, Optional[str]], List[str]] = {}
        all_ops = list(self.ops) + list(self.block_ops) + list(self.model_ops)
        for op in all_ops:
            for claim in op.claims:
                # `add_op` guarantees 4-tuple shape; tolerate 3-tuple as a
                # belt-and-suspenders fallback (e.g., op constructed via a
                # path that bypassed add_op validation in tests).
                if len(claim) == 3:
                    claim = (claim[0], claim[1], claim[2], None)
                registry.setdefault(claim, []).append(op.name)
        return registry

    def _detect_claim_collisions(self) -> List[str]:
        """Scan the claim registry for collisions; warn on each.

        Two claims collide only when their full 4-tuple
        ``(layer, scope, identifier, column)`` matches. Same row + different
        column is NOT a collision — that's exactly the column-disjoint case
        the row-only registry produced false positives for (see the L5
        head 5 row 32 motivating example in the task / commit history).

        ``column=None`` is treated as a literal value, not a wildcard:
        a legacy 3-tuple claim and a 4-tuple claim with column=None match,
        but a 4-tuple with column=None does not collide with a 4-tuple
        with column='CLEAN_EMBED_LO+0'. This keeps annotated ops free of
        spurious warnings when they upgrade to column granularity ahead
        of their peers — partial migration is safe.

        Returns the list of warning messages produced (so tests can
        inspect them without recapturing warnings).
        """
        import warnings as _warnings

        registry = self.build_claim_registry()
        messages: List[str] = []
        # Deterministic ordering for stable test output. The 4th element
        # (column) may be ``None``, which sorts non-lexically against
        # strings; coerce to "" for the sort key only.
        def _sort_key(k):
            return (k[0], k[1], k[2], "" if k[3] is None else k[3])

        for key in sorted(registry.keys(), key=_sort_key):
            owners = registry[key]
            if len(owners) >= 2:
                layer_idx, scope, identifier, column = key
                col_str = (
                    f" column={column!r}" if column is not None else ""
                )
                msg = (
                    f"DIM-OWNERSHIP COLLISION: layer={layer_idx} "
                    f"scope={scope!r} identifier={identifier!r}"
                    f"{col_str} "
                    f"claimed by {len(owners)} ops: {sorted(owners)!r}"
                )
                _warnings.warn(msg, stacklevel=3)
                messages.append(msg)
        return messages

    # --------------------------------------------------------------------
    # Residual-dim staleness invariants (Phase 3 / Agent G of
    # ARCH_LEAKAGE_FIX_PLAN.md). Each Operation may declare:
    #
    #   ``produces``       : Dict[dim_name, register_name]
    #   ``consumes_fresh`` : Dict[dim_name, register_name]
    #
    # The analyzer scans every op that declares ``consumes_fresh`` and asks:
    # is there *some other op in the same step* whose ``phase`` precedes this
    # op's phase AND which ``produces`` the same (dim, register)?  If not, it
    # warns -- the consumer is reading a stale (cross-step / leftover) value.
    #
    # "Same step" maps to a single forward pass; the dep graph + phase order
    # is the temporal ordering inside a step (smaller phase = earlier). Two
    # ops at the SAME phase can fire in either order (same-phase tie); we
    # treat ``producer.phase <= consumer.phase`` as an "in-step producer" so
    # tightly-paired ops (e.g., bake_ops sharing a phase number) still count.
    #
    # Ops with no ``phase`` (None) are considered "unordered" -- they can
    # only count as in-step producers for consumers that also have no phase.
    # In practice every annotated op should set ``phase`` for the analyzer
    # to be useful.
    # --------------------------------------------------------------------

    def build_staleness_registry(self) -> Tuple[
        Dict[Tuple[str, str], List[Tuple[str, Optional[float]]]],
        Dict[Tuple[str, str], List[Tuple[str, Optional[float]]]],
    ]:
        """Return (producers, consumers) registries keyed by (dim, register).

        Each registry maps ``(dim_name, register_name)`` -> list of
        ``(op_name, phase)`` pairs collected across ``self.ops``,
        ``self.block_ops``, and ``self.model_ops``. Exposed publicly so
        tests can inspect the staleness graph without re-running compile.
        """
        producers: Dict[Tuple[str, str], List[Tuple[str, Optional[float]]]] = {}
        consumers: Dict[Tuple[str, str], List[Tuple[str, Optional[float]]]] = {}
        all_ops = list(self.ops) + list(self.block_ops) + list(self.model_ops)
        for op in all_ops:
            for dim_name, register in op.produces.items():
                producers.setdefault(
                    (dim_name, register), []
                ).append((op.name, op.phase))
            for dim_name, register in op.consumes_fresh.items():
                consumers.setdefault(
                    (dim_name, register), []
                ).append((op.name, op.phase))
        return producers, consumers

    def _detect_staleness_violations(self) -> List[str]:
        """Scan the staleness registry; warn on each consumer without an
        in-step producer.

        Returns the list of warning messages produced (so tests can
        inspect them without recapturing warnings).
        """
        import warnings as _warnings

        producers, consumers = self.build_staleness_registry()
        messages: List[str] = []
        # Deterministic ordering for stable test output.
        for key in sorted(consumers.keys()):
            dim_name, register = key
            for consumer_name, consumer_phase in sorted(consumers[key]):
                # An in-step producer is any op that produces the same
                # (dim, register) at a phase <= consumer.phase. Consumers
                # without a phase only accept producers without a phase.
                in_step_producer = None
                for prod_name, prod_phase in producers.get(key, ()):
                    if prod_name == consumer_name:
                        # An op that both produces and consumes_fresh the
                        # same (dim, register) self-satisfies the invariant.
                        in_step_producer = prod_name
                        break
                    if consumer_phase is None:
                        if prod_phase is None:
                            in_step_producer = prod_name
                            break
                    elif prod_phase is not None and prod_phase <= consumer_phase:
                        in_step_producer = prod_name
                        break
                if in_step_producer is None:
                    msg = (
                        f"STALENESS VIOLATION: op {consumer_name!r} "
                        f"consumes_fresh dim={dim_name!r} "
                        f"register={register!r} but no earlier-phase op in "
                        f"the same step produces it"
                    )
                    _warnings.warn(msg, stacklevel=3)
                    messages.append(msg)
        return messages

    # --------------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------------

    def _topological_sort(self, ops: Optional[List[Operation]] = None) -> List[Operation]:
        """Return ops sorted so each op comes after every op that writes a dim it reads.

        Cycles: writes-then-reads on the same dim across ops is fine (downstream op
        sees upstream's write). A *cycle* would be op A reads dim X written by B,
        and B reads dim Y written by A. We detect cycles and raise.
        """
        if ops is None:
            ops = self.ops
        # writers[d] = list of ops that write dim d
        writers: Dict[str, List[Operation]] = {d: [] for d in self.dims}
        for op in ops:
            for d in op.writes:
                writers[d].append(op)

        # Build edges: u -> v iff v.reads ∩ u.writes
        # Drop edges where the dim-name-only dep model would create spurious
        # cycles. Two pruning rules:
        # 1. Phase-based: u.phase > v.phase means u writes "later" in hand-set
        #    order than v reads, so v doesn't actually depend on u.
        # 2. Block-internal kind ordering: at the SAME phase, attn comes
        #    before ffn within a transformer block. So an ffn writer doesn't
        #    create a dep on an attn reader at the same phase.
        in_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
        out_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
        op_set = {op.name for op in ops}
        for v in ops:
            for d in v.reads:
                for u in writers.get(d, ()):
                    if u.name == v.name:
                        continue
                    if u.name not in op_set:
                        continue
                    # Phase-based pruning
                    if (u.phase is not None and v.phase is not None):
                        if u.phase > v.phase:
                            continue
                        # Same-phase attn-vs-ffn: attn precedes ffn, so an ffn
                        # writer doesn't create a dep on an attn reader.
                        if (u.phase == v.phase
                                and u.kind == "ffn" and v.kind == "attn"):
                            continue
                    in_edges[v.name].add(u.name)
                    out_edges[u.name].add(v.name)

        # Kahn's algorithm
        ready = [op for op in ops if not in_edges[op.name]]
        # Stable order: by insertion order, so determinism
        ready.sort(key=lambda o: ops.index(o))
        result: List[Operation] = []
        while ready:
            op = ready.pop(0)
            result.append(op)
            for v_name in sorted(out_edges[op.name]):
                in_edges[v_name].discard(op.name)
                if not in_edges[v_name]:
                    ready.append(self._op_by_name[v_name])
            ready.sort(key=lambda o: ops.index(o))

        if len(result) != len(ops):
            stuck = [name for name, ins in in_edges.items() if ins]
            raise ValueError(f"Dependency cycle detected; stuck ops: {stuck}")
        return result

    def _assign_layers(self, topo: List[Operation]) -> Dict[str, int]:
        """Earliest-layer-first assignment respecting deps.

        Multiple ops can share a layer + kind slot if they have the same phase
        (e.g., two FFN setups that bake into the same block's FFN sequentially).
        Without phase, each op needs its own layer kind slot.

        Pinning: when an attn/ffn op has an explicit ``layer_idx`` set, the
        op is placed at exactly that layer (not the dep-graph's
        earliest-feasible slot). The dep-graph invariant — every read dim's
        producer is at a strictly-earlier layer, OR at the same layer with
        lower phase — is enforced as a hard check, so a mispinned layer_idx
        raises rather than silently corrupting the bake. Pinning was added
        to fix the L1+ regression where ``migrated=True`` attn/ffn ops with
        no layer_idx were assigned to the wrong block by the dep-graph; see
        ``docs/MODEL_REGRESSION_BISECT.md``.
        """
        writes_layer: Dict[str, int] = {}
        # writers_at_layer[(layer, dim)] = (op_name, phase) for the op that
        # wrote `dim` to that layer; used to enforce same-layer phase ordering
        # for pinned ops (read.phase > write.phase).
        writers_at_layer: Dict[tuple, tuple] = {}
        # layer_phase_kinds[(layer, kind)] = phase value if a phase-set op is using it
        layer_phase_kinds: Dict[tuple, Optional[int]] = {}
        assignment: Dict[str, int] = {}

        for op in topo:
            if op.kind == "model":
                continue
            earliest = 0
            for d in op.reads:
                if d in writes_layer:
                    earliest = max(earliest, writes_layer[d] + 1)

            if op.layer_idx is not None and op.kind in ("attn", "ffn"):
                # Pinned attn/ffn op — must land at layer_idx exactly.
                # Validate that every read dim is produced at a strictly
                # earlier layer, OR at the same layer with a lower phase,
                # OR at the same layer/phase by an attn op when this op is
                # an ffn op (within a transformer block, attn runs before
                # ffn, mirroring the same-phase pruning in _topological_sort).
                for d in op.reads:
                    if d not in writes_layer:
                        continue
                    write_layer = writes_layer[d]
                    if write_layer < op.layer_idx:
                        continue
                    if write_layer == op.layer_idx:
                        # Same-layer write/read: allow if (a) writer.phase <
                        # reader.phase, or (b) attn-then-ffn at same phase.
                        writer_info = writers_at_layer.get(
                            (op.layer_idx, d)
                        )
                        if writer_info is not None:
                            writer_name, writer_phase = writer_info
                            if (op.phase is not None
                                    and writer_phase is not None
                                    and writer_phase < op.phase):
                                continue
                            writer_op = self._op_by_name.get(writer_name)
                            if (op.kind == "ffn"
                                    and writer_op is not None
                                    and writer_op.kind == "attn"
                                    and writer_phase == op.phase):
                                continue
                    raise ValueError(
                        f"Op {op.name!r} pinned to layer {op.layer_idx} reads "
                        f"dim {d!r} which is produced at layer {write_layer} "
                        f"(must be < {op.layer_idx}, or == with lower phase, "
                        f"or attn-then-ffn at same phase)"
                    )
                layer = op.layer_idx
            else:
                layer = earliest
                while True:
                    key = (layer, op.kind)
                    existing_phase = layer_phase_kinds.get(key, "unset")
                    if existing_phase == "unset":
                        layer_phase_kinds[key] = op.phase
                        break
                    # Slot taken; if same phase, share. Otherwise advance.
                    if op.phase is not None and existing_phase == op.phase:
                        break
                    layer += 1
            assignment[op.name] = layer
            for d in op.writes:
                writes_layer[d] = max(writes_layer.get(d, -1), layer)
                # Track writer (name, phase) for same-layer phase-ordering checks.
                prev = writers_at_layer.get((layer, d))
                if prev is None or (op.phase is not None
                                    and (prev[1] is None
                                         or op.phase > prev[1])):
                    writers_at_layer[(layer, d)] = (op.name, op.phase)
        return assignment

    def _allocate_dims(self) -> Dict[str, int]:
        """Allocate dim positions.

        Pinned dims (declared with `pinned=POS`) get their requested position.
        Unpinned dims are bump-pointer allocated AFTER the highest pinned
        endpoint, in declaration order.
        """
        positions: Dict[str, int] = {}
        pinned = getattr(self, "_pinned", {}) or {}
        # Place pinned dims first
        for name, pos in pinned.items():
            positions[name] = pos
        # Highest pinned endpoint becomes the start for bump-pointer
        max_pinned_end = 0
        for name, pos in pinned.items():
            max_pinned_end = max(max_pinned_end, pos + self.dims[name])
        # Bump-pointer the rest, starting after the highest pinned endpoint
        cursor = max_pinned_end
        for name, size in self.dims.items():
            if name in pinned:
                continue
            positions[name] = cursor
            cursor += size
        return positions


def build_model_from_layout(layout: ModelLayout, S: float = 100.0,
                            legacy_bake: Optional[Set[str]] = None,
                            declarations_only: bool = False):
    """Construct an AutoregressiveVM from a compiled layout and bake all ops.

    This is the bridge from "layout produced by compiler" to "working model".

    For each layer:
      - The model's block at that layer index has its attn programmed by all
        ops with kind="attn" assigned to that layer.
      - The block's ffn is programmed by all ops with kind="ffn".
      - Block ops (kind="block") are dispatched on the whole TransformerBlock
        after attn/ffn ops at that layer.

    Each op's bake_fn is called with (target_module, dim_positions, S) where
    target_module is the attn, ffn, or block depending on op.kind.

    Args:
        layout: ModelLayout from LayerCompiler.compile()
        S: SwiGLU activation scale (passed to each bake_fn)
        legacy_bake: optional set of op-name strings whose bakes are still
            handled by an external (legacy) path. Ops in this set are skipped
            UNLESS they have op.migrated=True, in which case they fire here.
        declarations_only: when True, refuse to call imperative ``bake_fn``
            bodies. Only ``declarative_bake_fn`` generators and topology
            anchors may dispatch.

    Returns:
        AutoregressiveVM instance with weights baked.
    """
    # Lazy import to avoid circular dep when unified_compiler is loaded standalone
    from ..vm_step import AutoregressiveVM

    if layout.n_layers == 0:
        n_layers = 1
    else:
        n_layers = layout.n_layers

    model = AutoregressiveVM(d_model=layout.d_model, n_layers=n_layers)

    # If a `legacy_bake` model op is present, it owns the bake pass for any
    # op that hasn't been individually migrated. Per-op migration: when an op's
    # `migrated` flag is True, its bake_fn runs even with legacy_bake present.
    has_legacy_bake = any(o.name == "legacy_bake" for o in layout.model_ops)

    def _should_dispatch(op):
        return op.migrated or not has_legacy_bake

    per_layer_dispatch = [
        (layer_idx, op)
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer)
        for op in ops_at_layer
        if _should_dispatch(op)
    ]
    block_dispatch = [
        op for op in sorted(
            layout.block_ops,
            key=lambda o: (layout.resolve_block_op_layer(o), o.phase or 0),
        )
        if _should_dispatch(op)
    ]
    model_dispatch = sorted(layout.model_ops, key=lambda o: (o.phase or 0))
    if declarations_only:
        validate_declarations_only_ops(
            [op for _, op in per_layer_dispatch]
            + block_dispatch
            + model_dispatch
        )

    # Wrap the entire bake pass in no_grad so bake_fn implementations can do
    # in-place writes to leaf Parameters (the pattern used throughout
    # vm_step.py's hand-set weights).
    import torch as _torch
    with _torch.no_grad():
        for layer_idx, op in per_layer_dispatch:
            block = model.blocks[layer_idx]
            if op.kind == "attn":
                target = block.attn
            elif op.kind == "ffn":
                target = block.ffn
            elif op.kind == "block":
                target = block
            else:
                raise ValueError(
                    f"Op {op.name!r} in ops_per_layer has kind={op.kind!r}; "
                    "expected 'attn' or 'ffn'"
                )
            dispatch_operation_bake(
                op, target, layout.dim_positions, S,
                declarations_only=declarations_only,
            )

        # Block-scoped ops run after all attn/ffn bakes for their layer.
        # `_n_layers_hint` lets block bake_fns gate on total layer count.
        # Block op binding via target_op_name (resolved from layout) takes
        # precedence over the legacy layer_idx field.
        for op in block_dispatch:
            block = model.blocks[layout.resolve_block_op_layer(op)]
            block._n_layers_hint = len(model.blocks)
            dispatch_operation_bake(
                op, block, layout.dim_positions, S,
                declarations_only=declarations_only,
            )

        # Model-level ops run last (head, embedding, defensive patches,
        # right-size, expand wrappers, legacy_bake). Sort by phase.
        # Always dispatch model ops (head/embedding/legacy_bake all need to run).
        for op in model_dispatch:
            dispatch_operation_bake(
                op, model, layout.dim_positions, S,
                declarations_only=declarations_only,
            )

    return model
