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

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from .ssa_dim import SSA_ANY_WRITER, is_ssa_form, parse_ssa_name


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


def _topology_anchor_noop_bake(target, dim_positions, S):
    """Default no-op bake used by ``declarative_authority='topology_anchor'``.

    Topology anchors intentionally write no weights — they only exist so the
    dep graph keeps a stable shape for adjacent ops. Authors no longer need
    to spell out a no-op ``bake_fn=`` argument: omitting ``bake_fn`` on a
    topology-anchor op causes ``Operation.__post_init__`` to substitute this
    function in.
    """
    return None


class _IRBakeCallable:
    """Module-level callable that mirrors the IR-dispatch closure formerly
    defined inside ``_resolve_default_bake_fn``.

    Refactored out of a local closure so the ``compile_full_vm`` disk
    cache can pickle baked operations. Local closures of the form
    ``_resolve_default_bake_fn.<locals>._ir_bake`` are unpicklable, which
    caused ``compile_full_vm: failed to save cache ... (Can't get local
    object '_resolve_default_bake_fn.<locals>._ir_bake')`` warnings on
    every cold compile. The class form has a stable qualified name and
    pickles via its ``op`` attribute.
    """

    __slots__ = ("op",)

    def __init__(self, op: "Operation") -> None:
        self.op = op

    def __call__(self, target, dim_positions, S):
        op = self.op
        ir = op.compiler_ir
        if ir is None:
            ir = _make_operation_ir(op, target, dim_positions)
        _dispatch_operation_ir(op, target, dim_positions, S, ir)


def _resolve_default_bake_fn(op: "Operation") -> Optional[Callable]:
    """Pick a default ``bake_fn`` for an op that didn't supply one.

    Resolution order (see ``Operation.bake_fn`` docstring for rationale):

      1. ``declarative_bake_fn`` — the dominant case (100/115 historical
         core ops passed the same callable to both fields).
      2. An IR-lowering callable when ``compiler_ir`` /
         ``compiler_ir_factory`` is set. Mirrors the declarations-only
         dispatch path in ``dispatch_operation_bake`` so behaviour is
         byte-identical regardless of which path triggers. Implemented
         as ``_IRBakeCallable`` (module-level class) so the compile-cache
         pickle path works.
      3. A no-op when ``declarative_authority == 'topology_anchor'``.

    Returns ``None`` when none of the above applies — the op will retain
    ``bake_fn = None`` and ``dispatch_operation_bake`` will raise at the
    first call. This mirrors the pre-6B behaviour for ops missing a bake.
    """
    if op.declarative_bake_fn is not None:
        return op.declarative_bake_fn
    if op.compiler_ir is not None or op.compiler_ir_factory is not None:
        return _IRBakeCallable(op)
    if op.declarative_authority == "topology_anchor":
        return _topology_anchor_noop_bake
    return None


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

    `phase` is an optional ordering hint retained as the cycle-breaker
    inside the dim-only dep graph's strongly-connected components (Phase
    8.G.4: out-of-SCC phase ordering deleted; the SCC tiebreaker in
    ``_topological_sort`` still uses ``u.phase > v.phase`` to drop
    back-edges so Kahn's algorithm can complete on today's op set, which
    has 18-22 SCC members in attn/ffn). Smaller phase = earlier. Use the
    original `_set_layerN_*` layer number as the phase for migrated ops
    to preserve hand-set order until B9 dim decomposition (the SCC's
    OUTPUT_HI / IF_VAR back-edges) lands.

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
    # Imperative bake. As of Phase 6 Wave 6B this is OPTIONAL: when omitted,
    # ``__post_init__`` resolves it from the declarative siblings:
    #
    #   1. ``declarative_bake_fn`` if set (the common case — 100/115 core
    #      ops historically passed ``bake_fn=bake, declarative_bake_fn=bake``
    #      with the same callable, so omitting ``bake_fn`` is byte-identical).
    #   2. An IR-lowering closure when ``compiler_ir`` or
    #      ``compiler_ir_factory`` is set (used by ops whose only bake is the
    #      declarative IR — see ``layer16_lev_routing``-style ops).
    #   3. A no-op when ``declarative_authority == "topology_anchor"``
    #      (anchors intentionally write no weights).
    #
    # An op that supplies none of {bake_fn, declarative_bake_fn, compiler_ir,
    # compiler_ir_factory, topology_anchor authority} ends up with
    # ``bake_fn = None`` and will raise at dispatch — that mirrors the
    # pre-6B behaviour for ops missing a bake entirely.
    bake_fn: Optional[Callable] = None
    declarative_bake_fn: Optional[Callable] = None
    # Declarative compiler IR owned by this operation. When populated, this is
    # the semantic source of truth: bake functions should lower this IR to
    # weights, and verifiers/debuggers can symbolically interpret the same IR.
    compiler_ir: Optional[Any] = None
    # Late-bound IR factory for specs that need compiler-allocated dim
    # positions or the active attention head dimension. Called as
    # ``compiler_ir_factory(dim_positions, head_dim)`` by symbolic/debug
    # tooling and, in declarations-only mode, by the bake dispatcher when no
    # direct ``compiler_ir`` or ``declarative_bake_fn`` is present.
    compiler_ir_factory: Optional[Callable] = None
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
    # Tier A verifier annotations. Empty means "not annotated"; these fields
    # are metadata only unless a verifier chooses to inspect them.
    reset_after_step: Set[str] = field(default_factory=set)
    # Ordering / placement constraints. Historically a Dict[str, str] of
    # residual-constraint strings (free-form), which the verifier treats as
    # documentation. Two reserved keys carry scheduler semantics (B10 of the
    # dynamic scheduler migration, see docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md
    # §B10):
    #
    #   requires["after"] = "<op_name>" | (op_name, ...) | [op_name, ...]
    #       This op must be scheduled strictly after every referenced op
    #       (treated as a dep-graph edge by the scheduler / analyzer).
    #
    #   requires["same_layer_as"] = "<op_name>" | (op_name, ...)
    #       This op must be assigned to the same ``layer_idx`` as the
    #       referenced op. Honoured by the dynamic compile path (B11). The
    #       static path treats it as an ``after`` edge plus a per-layer
    #       equality assertion at bake time.
    #
    # Any other key retains the legacy residual-constraint string semantics
    # (used by ad-hoc decl_verifier docstring scans only). Values for the
    # reserved keys may be a single string OR an iterable of strings to
    # reference multiple ops; ``requires_after_ops()`` and
    # ``requires_same_layer_as_ops()`` below are the canonical accessors.
    requires: Dict[str, Union[str, Tuple[str, ...], List[str]]] = field(
        default_factory=dict
    )
    opcodes: Set[str] = field(default_factory=set)
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
    # Human-facing semantic label for reports/work queues. ``name`` remains
    # the stable programmatic identity; reports should prefer this label so
    # dynamic placement does not leak old physical layer numbers into user
    # facing blocker lists.
    semantic_label: Optional[str] = None

    def __post_init__(self):
        # Phase 6 Wave 6B: resolve ``bake_fn`` from declarative siblings when
        # the caller omitted it. See the ``bake_fn`` field docstring above
        # for the resolution chain.
        if self.bake_fn is None:
            resolved = _resolve_default_bake_fn(self)
            if resolved is not None:
                self.bake_fn = resolved

    def __hash__(self):
        return hash(self.name)


# Reserved ``Operation.requires`` keys whose values are op-name references
# (single string or iterable of strings) rather than residual-constraint
# strings. See ``Operation.requires`` docstring for semantics.
REQUIRES_AFTER_KEY = "after"
REQUIRES_SAME_LAYER_AS_KEY = "same_layer_as"
# Phase 7.A / SCC-zero residual structural retirement: cross-step "after"
# edge. The reader consumes the referenced op's output from the PREVIOUS
# autoregressive step (via the KV cache / residual carry). Treated as a
# NON-CYCLE edge by every scheduler. See
# c4_release/docs/NEXT_STEP_AFTER_PRIMITIVE.md.
REQUIRES_NEXT_STEP_AFTER_KEY = "next_step_after"
REQUIRES_OP_NAME_KEYS = frozenset(
    {
        REQUIRES_AFTER_KEY,
        REQUIRES_SAME_LAYER_AS_KEY,
        REQUIRES_NEXT_STEP_AFTER_KEY,
    }
)


def _requires_op_names(value) -> Tuple[str, ...]:
    """Normalize a ``requires[key]`` value into a tuple of op-name strings.

    Accepts a single string, or an iterable of strings. Empty iterables
    become an empty tuple. Other types raise ``TypeError`` at lookup time
    so authors get an immediate error rather than a silently-dropped edge.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple, set, frozenset)):
        out: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"requires op-name reference must be str, got "
                    f"{type(item).__name__}: {item!r}"
                )
            if item:
                out.append(item)
        return tuple(out)
    raise TypeError(
        "requires op-name reference must be str or iterable of str; got "
        f"{type(value).__name__}: {value!r}"
    )


def requires_after_ops(op: "Operation") -> Tuple[str, ...]:
    """Return the op-name strings ``op`` requires to run before it.

    Returns an empty tuple when no ``requires["after"]`` is declared. The
    returned names are NOT validated against the global op set — callers
    (scheduler / analyzer) decide whether an unknown name is an error.
    """
    return _requires_op_names(op.requires.get(REQUIRES_AFTER_KEY))


def requires_same_layer_as_ops(op: "Operation") -> Tuple[str, ...]:
    """Return the op-name strings ``op`` must share a layer with.

    Returns an empty tuple when no ``requires["same_layer_as"]`` is
    declared. Same-layer-as implies an ``after`` edge for scheduling
    purposes (the referenced op must already have been placed when the
    layer for this op is decided) plus an equality assertion downstream.
    """
    return _requires_op_names(op.requires.get(REQUIRES_SAME_LAYER_AS_KEY))


def requires_next_step_after_ops(op: "Operation") -> Tuple[str, ...]:
    """Return the op-name strings ``op`` requires to run in the PREVIOUS step.

    Returns an empty tuple when no ``requires["next_step_after"]`` is
    declared. Treated as a NON-CYCLE edge by every scheduler — same-step
    data-flow edges on the referenced op's writes that this op reads are
    suppressed. See ``docs/NEXT_STEP_AFTER_PRIMITIVE.md``.
    """
    return _requires_op_names(op.requires.get(REQUIRES_NEXT_STEP_AFTER_KEY))


def validate_requires_op_refs(
    ops: Iterable["Operation"],
) -> List[str]:
    """Return error messages for any ``requires`` op-name reference whose
    target name is not present in ``ops``.

    Pure validation — never raises. Empty list means every op-name
    reference resolves to a known op.
    """
    names: Set[str] = {op.name for op in ops}
    errors: List[str] = []
    for op in ops:
        for key in REQUIRES_OP_NAME_KEYS:
            if key not in op.requires:
                continue
            try:
                refs = _requires_op_names(op.requires[key])
            except TypeError as exc:
                errors.append(
                    f"op {op.name!r} requires[{key!r}]: {exc}"
                )
                continue
            for ref in refs:
                if ref == op.name:
                    errors.append(
                        f"op {op.name!r} requires[{key!r}] references "
                        f"itself"
                    )
                    continue
                if ref not in names:
                    errors.append(
                        f"op {op.name!r} requires[{key!r}]={ref!r} "
                        "references an unknown op"
                    )
    return errors


_LAYER_PREFIX_RE = re.compile(r"^_?layer\d+_")
_L_PREFIX_RE = re.compile(r"^l\d+_")
_SEMANTIC_NAME_OVERRIDES = {
    "layer3_carry_forward_attn": "PC carry-forward attention",
    "layer3_ffn": "initial-PC cancellation FFN",
    "layer4_sp_to_addr_key": "SP-to-address-key relay",
    "layer6_routing_ffn": "opcode routing FFN",
    "layer8_alu": "add/sub ALU lookup",
    "layer8_mem_to_alu": "memory-to-ALU relay",
    "layer8_multibyte_routing": "multibyte routing",
    "layer9_alu": "comparison ALU lookup",
    "layer10_alu": "boolean/shift ALU lookup",
    "layer11_mul_partial": "multiply partial lookup",
    "layer12_mul_combine": "multiply combine lookup",
    "layer13_mem_addr_gather": "memory-address gather",
    "layer13_shifts": "shift lookup",
    "layer14_addr_key_neural_decode": "address-key neural decode",
    "layer14_alu_nocarry_ax_bytes_zero": "ALU no-carry AX-byte zeroing",
    "layer14_clear_addr_key_pollution": "address-key pollution clear",
    "layer14_clear_mem_marker_output": "memory-marker output clear",
    "layer14_clear_output_corruption": "output corruption clear",
    "layer14_jsr_ax_bytes_zero": "JSR AX-byte zeroing",
    "layer14_lc_ax_bytes_zero": "LC AX-byte zeroing",
    "layer14_mem_generation": "memory generation",
    "layer14_temp_clear": "TEMP clear",
    "layer15_memory_lookup": "memory lookup",
    "layer15_nibble_copy": "nibble copy",
    "layer16_lev_routing": "LEV routing",
}


def operation_display_label(op: Operation) -> str:
    """Return a human-facing semantic label for an operation.

    ``Operation.name`` is intentionally stable and may retain historical
    layer-number prefixes. Display labels are for reports and queues where
    dynamic placement makes those prefixes misleading.
    """

    if op.semantic_label:
        return op.semantic_label
    if op.name in _SEMANTIC_NAME_OVERRIDES:
        return _SEMANTIC_NAME_OVERRIDES[op.name]
    label = _LAYER_PREFIX_RE.sub("", op.name)
    label = _L_PREFIX_RE.sub("", label)
    return label or op.name


class DeclarationsOnlyBakeError(RuntimeError):
    """Raised when declarations-only mode reaches an imperative-only op."""

    def __init__(
        self,
        unsupported_ops: List[str],
        unsupported_display_labels: Optional[List[str]] = None,
    ):
        if unsupported_display_labels is None:
            unsupported_display_labels = list(unsupported_ops)
        pairs = sorted(zip(unsupported_ops, unsupported_display_labels))
        self.unsupported_ops = tuple(op for op, _ in pairs)
        self.unsupported_display_labels = tuple(label for _, label in pairs)
        sample = ", ".join(self.unsupported_display_labels[:20])
        if len(self.unsupported_display_labels) > 20:
            sample += (
                f", ... (+{len(self.unsupported_display_labels) - 20} more)"
            )
        super().__init__(
            "Declarations-only bake cannot run because some dispatched ops do "
            "not expose declarative_bake_fn and are not topology anchors: "
            f"{sample}"
        )


def operation_supports_declarations_only(op: Operation) -> bool:
    """Return whether ``op`` can run without its imperative ``bake_fn``."""

    return (
        op.compiler_ir is not None
        or
        op.compiler_ir_factory is not None
        or
        op.declarative_bake_fn is not None
        or op.declarative_authority == "topology_anchor"
    )


def validate_declarations_only_ops(ops: List[Operation]):
    """Raise when any op in dispatch order lacks a declarations-only bake."""

    unsupported_ops = [
        op for op in ops if not operation_supports_declarations_only(op)
    ]
    if unsupported_ops:
        raise DeclarationsOnlyBakeError(
            [op.name for op in unsupported_ops],
            [operation_display_label(op) for op in unsupported_ops],
        )


def dispatch_operation_bake(op: Operation, target, dim_positions, S, *,
                            declarations_only: bool = False):
    """Dispatch one operation through either the normal or declarations-only path."""

    if not declarations_only:
        op.bake_fn(target, dim_positions, S)
        return
    if op.declarative_bake_fn is not None:
        op.declarative_bake_fn(target, dim_positions, S)
        return
    if op.compiler_ir is not None:
        _dispatch_operation_ir(op, target, dim_positions, S, op.compiler_ir)
        return
    if op.compiler_ir_factory is not None:
        _dispatch_operation_ir(
            op,
            target,
            dim_positions,
            S,
            _make_operation_ir(op, target, dim_positions),
        )
        return
    if op.declarative_authority == "topology_anchor":
        return
    raise DeclarationsOnlyBakeError([op.name], [operation_display_label(op)])


def _make_operation_ir(op: Operation, target, dim_positions):
    factory = op.compiler_ir_factory
    if factory is None:
        raise DeclarationsOnlyBakeError([op.name], [operation_display_label(op)])

    attn = target
    if op.kind == "block":
        attn = getattr(target, "attn", None)
    if attn is None or not hasattr(attn, "W_q"):
        head_dim = 64
    else:
        head_dim = attn.W_q.shape[0] // attn.num_heads
    return factory(dim_positions, head_dim)


def _dispatch_operation_ir(op: Operation, target, dim_positions, S, ir):
    """Lower an op-owned CompilerIR into its target module."""

    if ir is None:
        raise DeclarationsOnlyBakeError([op.name], [operation_display_label(op)])
    if op.kind == "ffn":
        ir.lower_ffn(target, dim_positions, layer_idx=0, start_unit=0, S=S)
        return
    if op.kind == "attn":
        head_dim = target.W_q.shape[0] // target.num_heads
        ir.lower_attention(
            target,
            head_dim,
            layer_idx=0,
            dim_positions=dim_positions,
            S=S,
        )
        return
    if op.kind == "block":
        if getattr(target, "attn", None) is not None:
            head_dim = target.attn.W_q.shape[0] // target.attn.num_heads
            ir.lower_attention(
                target.attn,
                head_dim,
                layer_idx=0,
                dim_positions=dim_positions,
                S=S,
            )
        if getattr(target, "ffn", None) is not None:
            ir.lower_ffn(
                target.ffn,
                dim_positions,
                layer_idx=0,
                start_unit=0,
                S=S,
            )
        return
    raise DeclarationsOnlyBakeError([op.name], [operation_display_label(op)])


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
            default ffn_hidden — 4096 by default). Used by ``compile_full_vm_dynamic``
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

    def declare_dim(self, name: str, size: int, pinned: Optional[int] = None,
                    alias_of: Optional[str] = None):
        """Declare a dim with optional pinned start position.

        When pinned is given, the compiler MUST place the dim at that exact
        position (used for backward-compat with _SetDim aliasing where multiple
        names share the same physical position).

        When pinned is None, the dim is bump-pointer-allocated.

        When ``alias_of`` is given, the dim is registered as an alias of
        ``alias_of``: at allocation time it gets the SAME numeric position as
        the base regardless of whether the base is pinned or bump-allocated.
        Use this when the base is bump-pointer-allocated (not pinned) and the
        alias must follow it. Both ``pinned`` and ``alias_of`` can be supplied;
        ``alias_of`` takes precedence at allocation time. See Phase 7.A.3.c
        spec — PREV_STEP cross-step alias decomposition.
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
        if alias_of is not None:
            if not hasattr(self, "_aliases"):
                self._aliases: Dict[str, str] = {}
            self._aliases[name] = alias_of

    def add_op(self, op: Operation):
        if op.kind not in ("attn", "ffn", "block", "model"):
            raise ValueError(
                f"op.kind must be 'attn', 'ffn', 'block', or 'model'; got {op.kind!r}"
            )
        if op.name in self._op_by_name:
            raise ValueError(f"Operation name {op.name!r} already added")
        # Phase 9 SSA prototype: SSA-form reads/writes
        # (``BASE.WRITER.STEP_OFFSET``) are auto-declared as aliases of
        # the base dim if the base dim is already declared. This keeps the
        # bake-time ``dim_positions`` lookup byte-identical to the
        # unversioned form while letting the scheduler see the
        # cross-step semantics. See ssa_dim.py for the schema and
        # docs/PHASE_9_SSA_PROTOTYPE.md for the migration plan.
        for d in op.reads | op.writes:
            if d in self.dims:
                continue
            if is_ssa_form(d):
                parsed = parse_ssa_name(d)
                if parsed.base_dim not in self.dims:
                    raise ValueError(
                        f"Op {op.name!r} SSA dim {d!r} references "
                        f"undeclared base dim {parsed.base_dim!r}"
                    )
                # Auto-declare the SSA form as an alias of the base. Same
                # numeric slot, same size; matches the PREV_STEP pattern.
                self.declare_dim(
                    d,
                    self.dims[parsed.base_dim],
                    alias_of=parsed.base_dim,
                )
                continue
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
        if op.semantic_label is not None and not isinstance(op.semantic_label, str):
            raise ValueError(
                f"Op {op.name!r} semantic_label must be str or None; "
                f"got {type(op.semantic_label).__name__}"
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
            # Phase 8.A.4 retry: block ops may bind to a layer via
            # ``target_op_name`` (pointing at an attn/ffn op the topo loop
            # already placed) instead of a hardcoded ``layer_idx``. Resolve
            # the target's layer in that case so block ops without a
            # ``layer_idx`` pin still land at the correct block.
            if op.layer_idx is not None:
                layer_assignment[op.name] = op.layer_idx
            elif op.target_op_name is not None:
                target_layer = layer_assignment.get(op.target_op_name)
                if target_layer is None:
                    raise ValueError(
                        f"Block op {op.name!r} target_op_name "
                        f"{op.target_op_name!r} not found in placed "
                        f"attn/ffn ops"
                    )
                layer_assignment[op.name] = target_layer
            else:
                raise ValueError(
                    f"Block op {op.name!r} has neither layer_idx nor "
                    f"target_op_name"
                )

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
        # Phase 8.G.4 note: kept because model ops are not in the
        # ``_topological_sort`` dep graph and have no other ordering
        # signal; today's model ops cover a wide phase range (8.0 to
        # 1300) that does NOT match insertion order, so dropping this
        # sort would break byte-identity with the static path. Retire
        # only after model ops migrate to explicit ``requires["after"]``
        # chains (tracked in PHASE_8_PLAN.md 8.G.5).
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
    #
    # Phase 8.G.4 note: this analyzer's phase comparison is purely
    # DIAGNOSTIC -- it never affects compile output. The producer-after-
    # consumer test (``test_producer_after_consumer_warns``) pins the
    # phase-ordering semantic, so the rewrite to a dep-derived signal
    # ships with the SCC retirement in PHASE_8_PLAN.md 8.G.5.
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

        Phase as SCC tiebreaker (Phase 8.G.4)
        ------------------------------------
        Today's production op set has 18-22 attn/ffn ops trapped in an
        SCC of the unpruned dim-only dep graph (the OUTPUT_HI / IF_VAR
        carry chain; see ``docs/B9_OUTPUT_HI_SPLIT_SPEC.md``). To let
        Kahn's algorithm complete, this method drops back-edges where
        ``u.phase > v.phase``. The rule is purely a CYCLE-BREAKER
        inside the SCC -- every edge it drops is an edge that, kept,
        would prevent the topo sort from terminating. Phase 8.G.4
        deleted the out-of-SCC phase usages (the ``Operation.phase``
        deprecation scaffold, unused imports, etc.); this in-SCC
        tiebreaker stays until B9 dim decomposition breaks the cycle
        at the data-flow level, at which point ``phase`` retires
        entirely (see ``docs/PHASE_8_PLAN.md`` 8.G).
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
        # 1. Phase-based (SCC tiebreaker; see method docstring): u.phase
        #    > v.phase means u writes "later" in hand-set order than v
        #    reads, so v doesn't actually depend on u. Load-bearing for
        #    the production OUTPUT_HI carry SCC.
        # 2. Block-internal kind ordering: at the SAME phase, attn comes
        #    before ffn within a transformer block. So an ffn writer doesn't
        #    create a dep on an attn reader at the same phase.
        in_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
        out_edges: Dict[str, Set[str]] = {op.name: set() for op in ops}
        op_set = {op.name for op in ops}
        for v in ops:
            # Phase 7.A SCC retirement: ``requires["next_step_after"] = X``
            # declares ``v`` reads X's output from the PREVIOUS step.
            # Suppress same-step data-flow edges on every dim X writes
            # that v reads. NO replacement explicit edge is added. See
            # ``docs/NEXT_STEP_AFTER_PRIMITIVE.md``.
            next_step_suppressed_dims: Set[str] = set()
            for ref in requires_next_step_after_ops(v):
                if ref == v.name or ref not in op_set:
                    continue
                ref_op = self._op_by_name.get(ref)
                if ref_op is None:
                    continue
                for d in ref_op.writes & v.reads:
                    next_step_suppressed_dims.add(d)
            for d in v.reads:
                # Phase 9 SSA prototype: SSA cross-step reads
                # (``BASE.WRITER.STEP_OFFSET`` with step_offset != 0) are
                # cross-step by construction. Skip the back-edge entirely
                # at the read level so the dep graph stays acyclic. When
                # the writer is named explicitly (writer_op != "*"), we
                # could in principle add the edge to *that* writer only;
                # for the prototype we treat every SSA cross-step read as
                # a pure cross-step alias (matches the PREV_STEP
                # semantics today). See ssa_dim.py.
                if is_ssa_form(d):
                    parsed = parse_ssa_name(d)
                    if parsed.is_cross_step:
                        continue
                if d in next_step_suppressed_dims:
                    continue
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

            # B10: explicit ``requires["after"]`` / ``requires["same_layer_as"]``
            # op-name edges. Both reserved keys normally add an A->v dep edge
            # (v must run after the referenced op). Phase pruning does NOT
            # apply -- explicit op-name references are author intent and
            # override the dim-only dep model. Unknown names are silently
            # skipped here; ``validate_requires_op_refs`` is the place that
            # errors on them.
            #
            # B9 EXCEPTION: cross-step semantic. When requires["after"]
            # references an op at a STRICTLY LATER phase than v AND that op
            # writes some dim that v also reads, the constraint expresses a
            # PREV-STEP carry (v reads ref's output from the previous
            # autoregressive step). In the single-step static compile path
            # the edge would create a forward-cycle, so it is dropped here.
            # See docs/B9_OUTPUT_HI_SPLIT_SPEC.md §7.2 (R-OH-2).
            for ref in requires_after_ops(v):
                if ref == v.name or ref not in op_set:
                    continue
                ref_op = self._op_by_name[ref]
                if (v.phase is not None and ref_op.phase is not None
                        and ref_op.phase > v.phase
                        and (ref_op.writes & v.reads)):
                    # Cross-step semantic: skip the static edge.
                    continue
                in_edges[v.name].add(ref)
                out_edges[ref].add(v.name)
            for ref in requires_same_layer_as_ops(v):
                if ref == v.name or ref not in op_set:
                    continue
                in_edges[v.name].add(ref)
                out_edges[ref].add(v.name)

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

        Phase 8.G.4 note: every ``phase`` consult in this method is
        load-bearing for today's production op set (slot sharing via
        ``layer_phase_kinds`` packs the 17-layer model; the B9
        cross-step exception keeps the L3 / L16 carry compileable; the
        same-layer pinned check is the validator's escape valve).
        Retire as a wave once the B9 SCC breaks (dim decomposition lands
        in PHASE_8_PLAN.md 8.G.5) and op factories adopt explicit
        ``requires["after"]`` / ``requires["same_layer_as"]`` declarations
        in place of phase ordinals.
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

            # B10: ``requires["after"] = "<op>"`` forces strict later layer
            # than every referenced op. ``requires["same_layer_as"]`` forces
            # equality (asserted below after the layer is chosen).
            #
            # B9 EXCEPTION (mirrors _topological_sort): cross-step
            # requires["after"] skips the layer constraint when the
            # ref is at a later phase AND writes some dim ``op`` reads
            # (prev-step carry; see B9 spec §7.2 R-OH-2).
            for ref in requires_after_ops(op):
                ref_op = self._op_by_name.get(ref)
                if (ref_op is not None
                        and op.phase is not None and ref_op.phase is not None
                        and ref_op.phase > op.phase
                        and (ref_op.writes & op.reads)):
                    continue
                ref_layer = assignment.get(ref)
                if ref_layer is None and ref_op is not None and ref_op.kind == "block":
                    # Phase 8.A.4: block ops are placed by ``layer_idx`` after
                    # the attn/ffn topo loop, so ``assignment`` won't have
                    # them yet. Fall back to the pinned ``layer_idx`` (or to
                    # ``target_op_name``'s already-placed layer) so an
                    # attn/ffn op declaring ``requires["after"] = "<block_op>"``
                    # gets its layer bumped past the referenced block.
                    if ref_op.layer_idx is not None:
                        ref_layer = ref_op.layer_idx
                    elif ref_op.target_op_name is not None:
                        ref_layer = assignment.get(ref_op.target_op_name)
                if ref_layer is not None:
                    earliest = max(earliest, ref_layer + 1)
            same_layer_refs = requires_same_layer_as_ops(op)
            for ref in same_layer_refs:
                ref_op = self._op_by_name.get(ref)
                ref_layer = assignment.get(ref)
                if ref_layer is None and ref_op is not None and ref_op.kind == "block":
                    # Phase 8.A.4: same handling for ``requires["same_layer_as"]``
                    # block refs — pull the layer from the block op's pin so
                    # an attn/ffn op can co-place with a known block layer.
                    if ref_op.layer_idx is not None:
                        ref_layer = ref_op.layer_idx
                    elif ref_op.target_op_name is not None:
                        ref_layer = assignment.get(ref_op.target_op_name)
                if ref_layer is not None:
                    earliest = max(earliest, ref_layer)

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
            # B10: enforce ``requires["same_layer_as"]`` equality. If the
            # referenced op has already been placed and lives at a different
            # layer, the constraint is violated — raise rather than silently
            # corrupt the layout. References to ops not yet placed are
            # tolerated (topo order guarantees that for refs WITH outgoing
            # data flow to ``op`` they would already be placed; isolated
            # same_layer_as refs without dim deps may resolve in either
            # direction).
            for ref in same_layer_refs:
                ref_op = self._op_by_name.get(ref)
                ref_layer = assignment.get(ref)
                if ref_layer is None and ref_op is not None and ref_op.kind == "block":
                    # Phase 8.A.4: block-op ref — pull layer from the block's
                    # ``layer_idx`` pin (or its resolved ``target_op_name``)
                    # so the equality check fires even when block ops have
                    # not yet been assigned by the post-topo loop.
                    if ref_op.layer_idx is not None:
                        ref_layer = ref_op.layer_idx
                    elif ref_op.target_op_name is not None:
                        ref_layer = assignment.get(ref_op.target_op_name)
                if ref_layer is not None and ref_layer != layer:
                    raise ValueError(
                        f"Op {op.name!r} requires[\"same_layer_as\"] "
                        f"references {ref!r} at layer {ref_layer} but "
                        f"this op was placed at layer {layer}"
                    )
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
        endpoint, in declaration order. Aliases (declared via
        ``declare_dim(..., alias_of=BASE)``) get the same position as their
        base AND do not consume bump-pointer space.
        """
        positions: Dict[str, int] = {}
        pinned = getattr(self, "_pinned", {}) or {}
        aliases = getattr(self, "_aliases", {}) or {}
        # Place pinned dims first (skipping aliases — they resolve last)
        for name, pos in pinned.items():
            if name in aliases:
                continue
            positions[name] = pos
        # Highest pinned endpoint becomes the start for bump-pointer
        max_pinned_end = 0
        for name, pos in pinned.items():
            if name in aliases:
                continue
            max_pinned_end = max(max_pinned_end, pos + self.dims[name])
        # Bump-pointer the rest (excluding aliases), starting after the
        # highest pinned endpoint
        cursor = max_pinned_end
        for name, size in self.dims.items():
            if name in pinned or name in aliases:
                continue
            positions[name] = cursor
            cursor += size
        # Resolve aliases last: each alias inherits its base's position.
        # Aliases-of-aliases are resolved transitively.
        for name in list(aliases):
            base = aliases[name]
            # Walk alias chain (cap to dim count to avoid cycles).
            for _ in range(len(self.dims) + 1):
                if base not in aliases:
                    break
                base = aliases[base]
            if base not in positions:
                raise ValueError(
                    f"Alias {name!r} references unknown base dim {base!r}"
                )
            positions[name] = positions[base]
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
