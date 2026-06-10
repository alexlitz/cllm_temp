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

import os
import re
import warnings
from dataclasses import InitVar, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from .slot_registry import (
    ALLOWED_SLOT_KINDS as _SLOT_REGISTRY_ALLOWED_KINDS,
    SlotRegistry,
    derive_slot_ids_for_op,
)
from .ssa_dim import SSA_ANY_WRITER, is_ssa_form, parse_ssa_name


# Feature flag: compile-time dim slot sharing via liveness analysis.
# When ``True`` (constructor arg or env var ``C4_DIM_LIVENESS=1``), the
# compiler runs the register-allocation-style liveness pass in
# ``_compute_dim_layout_with_liveness`` and shares residual-stream slots
# between unpinned, non-aliased, single-cell scratch dims whose lifetimes
# do not overlap. Default is OFF: production keeps the bump-pointer layout
# until the liveness path is proven byte-identical on the full smoke set.
_DIM_LIVENESS_ENV = "C4_DIM_LIVENESS"


def _env_flag_dim_liveness() -> bool:
    # Default ON. Set ``C4_DIM_LIVENESS=0`` to opt out (e.g. for bisecting a
    # regression). The lifetime walker fix (471c1c08) makes the merge sound
    # — byte-identity ≡ 0 vs OFF — and the L0-L7 declaration audit gave the
    # allocator the real lifetime graph it needs to share slots safely.
    return os.environ.get(_DIM_LIVENESS_ENV, "1") != "0"


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
    # ARCH_LEAKAGE_FIX_PLAN.md).
    #
    # Step 5 of IR_INCREMENTAL_IMPROVEMENTS.md: ``produces`` and
    # ``consumes_fresh`` are now COMPUTED PROPERTIES (see the
    # ``@property`` definitions below) derived from
    # ``compiler_ir.layers[i].ffn.rules``. Op authors no longer declare
    # them explicitly — the derivation aggregates every dim name written by
    # any FFN rule's ``writes`` (→ ``produces``) and every condition-read
    # dim that is also in ``reads`` and not on the cross-step-durable
    # allowlist (→ ``consumes_fresh``). See ``tools.derive_produces_consumes``
    # for the canonical derivation logic.
    #
    # For backward compatibility (a small number of synthetic test ops that
    # have no ``compiler_ir``), the constructor still accepts ``produces=``
    # and ``consumes_fresh=`` kwargs; when provided they OVERRIDE the
    # derived value. Implemented as :class:`dataclasses.InitVar` slots with
    # *renamed* ``produces_override`` / ``consumes_fresh_override`` field
    # names so they don't shadow the ``@property`` definitions below; the
    # generated ``__init__`` exposes them as ``produces`` / ``consumes_fresh``
    # via the alias-init wrapper installed at the bottom of this class
    # (search for ``_install_legacy_init_aliases``).
    #
    # See c4_release/docs/STALENESS_INVARIANTS.md for the bake-author API
    # and the canonical AX_CARRY example.
    produces_override: InitVar[Optional[Dict[str, str]]] = None
    consumes_fresh_override: InitVar[Optional[Dict[str, str]]] = None
    # Audit marker (waves 4-7 of docs/PRODUCES_CONSUMES_MIGRATION.md). When
    # True, the empty ``produces`` / ``consumes_fresh`` dicts are a
    # deliberate, audited declaration that the op has no in-step semantic
    # residual-dim read/write surface (e.g. model_ops bake weights into
    # the FFN/attn matrices, not into residual dims; flag-gated ops are
    # no-ops at default flag config; ALU lookup tables are constants).
    # The audit script (``tools/audit_produces_consumes.py``) buckets such
    # ops as ``audited_empty`` instead of ``none``. Wave 8 will require
    # every op to declare either non-empty ``produces`` /
    # ``consumes_fresh`` OR ``audited_empty_produces=True``.
    audited_empty_produces: bool = False
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
    # Phase 1 (memory cluster fix plan, docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md):
    # slot-share opt-out for the compile-time slot-conflict registry. When this
    # op's bake legitimately shares a slot kind with another op at the same
    # layer (e.g. multiple L10 tail-correction families sharing block.ffn at
    # disjoint hidden-unit ranges), list those slot kinds here. Allowed values
    # are the entries of ``slot_registry.ALLOWED_SLOT_KINDS``: ``"ffn"``,
    # ``"attn"``, ``"attn_head"``, ``"attn_matrix"``, ``"post_ops"``,
    # ``"post_ops_append"``, ``"ffn_units"``. Empty (the default) means the
    # op claims its slots exclusively; the registry will refuse to compile
    # if another non-opted-in op claims the same slot at the same layer.
    # See docs/SLOT_REGISTRY_AUDIT_2026_06_05.md for the live audit.
    slot_share: Tuple[str, ...] = ()
    # Migration opt-in (docs/UNDECLARED_DIM_AUDIT_2026_06_09.md follow-up):
    # when True, the op's ``reads`` / ``writes`` annotations are treated as
    # advisory and the actual scheduling/contract sets are derived from
    # ``compiler_ir`` rule contents via
    # ``unified_compiler.op_introspect.derive_op_reads_writes_from_rules``.
    # See ``Operation.derive_reads_writes`` (the method) below for the
    # explicit derivation entry point and ``op_introspect.derive_operation``
    # for the rewrite. Default False keeps every existing op byte-identical
    # while authors migrate the 162-op corpus per the audit doc's
    # priority list.
    derive_reads_writes_flag: bool = False

    def __post_init__(
        self,
        produces_override: Optional[Dict[str, str]] = None,
        consumes_fresh_override: Optional[Dict[str, str]] = None,
    ):
        # Phase 6 Wave 6B: resolve ``bake_fn`` from declarative siblings when
        # the caller omitted it. See the ``bake_fn`` field docstring above
        # for the resolution chain.
        if self.bake_fn is None:
            resolved = _resolve_default_bake_fn(self)
            if resolved is not None:
                self.bake_fn = resolved
        # Step 5 of IR_INCREMENTAL_IMPROVEMENTS.md: explicit ``produces`` /
        # ``consumes_fresh`` overrides are stored on private slots. The
        # public ``produces`` / ``consumes_fresh`` attributes are
        # ``@property`` accessors below that prefer the override and
        # otherwise derive from ``compiler_ir`` rule contents.
        self._produces_explicit: Optional[Any] = produces_override
        self._consumes_fresh_explicit: Optional[Any] = consumes_fresh_override

    def __hash__(self):
        return hash(self.name)

    # ---- Step 5: derived produces / consumes_fresh ----------------------
    # Cross-step / embed-time durable dims — reads of these don't count as
    # ``consumes_fresh`` because their value is either embed-time-stable,
    # opcode-broadcast (one-hot per opcode, never refreshed in-step), or
    # explicitly cross-step relayed. The list is a superset of
    # ``tools.derive_produces_consumes._CROSS_STEP_DURABLE`` — it picks up
    # the opcode dims registered in
    # ``c4_release/neural_vm/dim_registry._OPCODES`` (all ``OP_*`` names),
    # PSH/IO cascade relays, MEM_VAL_BYTE/B nibble relays, and the SSA
    # prev-step form ``BASE.*.-1`` (handled separately via
    # ``_is_cross_step_dim_name``).
    _CROSS_STEP_DURABLE_DIMS = frozenset({
        "CONST",
        "IS_BYTE",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "MARK_PC", "MARK_BP", "MARK_AX", "MARK_SP",
        "MARK_STACK0", "MARK_STACK1", "MARK_STACK2",
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "EMBED_LO", "EMBED_HI",
        "H0", "H1",
        # Opcode-broadcast one-hots from dim_registry._OPCODES — every
        # ``OP_*`` dim is set at embed time from the instruction stream
        # and stays constant across the per-instruction cycle. Reads of
        # these are gate/scope flags, not fresh-consume residuals.
        "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR",
        "OP_BZ", "OP_BNZ", "OP_ENT", "OP_ADJ",
        "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
        "OP_SC", "OP_PSH", "OP_OR", "OP_XOR",
        "OP_AND", "OP_EQ", "OP_NE", "OP_LT",
        "OP_GT", "OP_LE", "OP_GE", "OP_SHL",
        "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
        "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
        "OP_PUTCHAR", "OP_GETCHAR",
        "OP_RET",  # legacy alias retained from the migration helper.
        # PSH / opcode cascade relays staged into SP/STACK0 at embed
        # time; constant across the inter-byte cascade window.
        "PSH_AT_SP",
        # CMP cascade per-opcode bytes — staged by L6 relay heads at
        # embed time, not refreshed by the consumer's same-step phase.
        "CMP+0", "CMP+1", "CMP+2", "CMP+3", "CMP+4",
        "CMP+5", "CMP+6", "CMP+7", "CMP+8", "CMP+9",
        # Token-position markers (set at embed time, durable through the
        # step's compute layers).
        "MARK_CS", "MARK_MEM", "MARK_SE_ONLY",
        "MARK_THINKING_START", "MARK_THINKING_END",
        "IS_MARK",
        # CLEAN_EMBED_* — clean embedding nibble pair, written by the
        # L0/L1 embed cleanup pass and durable through the rest of the
        # step's compute.
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        # NEXT_* — token-class next-register predictors, set at embed
        # time from the bytecode stream.
        "NEXT_AX", "NEXT_BP", "NEXT_MEM", "NEXT_PC",
        "NEXT_SP", "NEXT_SE", "NEXT_STACK0",
        "NEXT_THINKING_START", "NEXT_THINKING_END",
        # IO-state cross-step durables.
        "LAST_WAS_BYTE", "IO_IS_PUTCHAR", "IO_IS_PRTF",
        "IO_IN_OUTPUT_MODE", "IO_OUTPUT_COMPLETE",
        "IO_STATE", "IO_FORMAT_POS",
        # L1 head outputs are L1-baked and consumed cross-step by
        # downstream layers' attn/ffn reads — embed-time-equivalent.
        "L1H0", "L1H1", "L1H2", "L1H3", "L1H4",
        # Opcode-broadcast suffix carriers (one-hot per opcode, embed-time).
        "OPCODE_BASE", "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
        "OPCODE_BASE_BYTE0", "OPCODE_BASE_BYTE1",
        "OP_LC_RELAY", "OP_LI_RELAY", "OP_SI_RELAY",
        # MEM / STACK byte position flags (embed-time markers).
        "STACK0_BYTE0", "STACK0_BYTE1",
        "STACK0_BYTE2", "STACK0_BYTE3",
        "MEM_STORE", "MEM_ADDR_SRC",
        # HAS_SE — STEP_END existence flag, set at embed time.
        "HAS_SE",
        # CMP — base CMP marker flag.
        "CMP",
    })

    @staticmethod
    def _is_cross_step_dim_name(name: str) -> bool:
        """Return True for dim names that are cross-step by construction.

        Covers the static :attr:`_CROSS_STEP_DURABLE_DIMS` set plus SSA
        prev-step aliases of the form ``BASE.*.OFFSET`` where ``OFFSET``
        is a negative integer (e.g. ``OUTPUT_LO.*.-1``). The compiler
        treats those as alias-of-base reads from the previous VM step;
        they are by definition not refreshed by an in-step producer.
        """
        if name in Operation._CROSS_STEP_DURABLE_DIMS:
            return True
        # SSA prev-step form: e.g. ``OUTPUT_LO.*.-1``.
        if ".*." in name:
            tail = name.rsplit(".*.", 1)[-1]
            try:
                offset = int(tail)
            except ValueError:
                return False
            return offset < 0
        return False

    # Constant slot used for every derived ``produces`` / ``consumes_fresh``
    # entry. Step 5 of IR_INCREMENTAL_IMPROVEMENTS.md: the derivation can
    # no longer infer the semantic register-name strings the manual
    # annotations carried (``"AX_byte0"``, ``"SP_marker"``, ...) because
    # rule contents don't encode that hand-picked metadata. Collapsing
    # every derived entry to ``"<derived>"`` makes producer/consumer
    # pairing depend only on the dim name — the analyzer asks "does any
    # earlier-phase op write this dim within the step?", which is the
    # weaker but still useful in-step refresh contract the test corpus
    # exercises (see test_removing_l8_head6_surfaces_ax_carry_staleness:
    # removing the L8 head6 producer of ``AX_CARRY_LO`` still surfaces
    # the consumer's missing-producer warning under this scheme).
    _DERIVED_SLOT = "<derived>"

    def _derive_register_slot(self) -> str:
        """Return the slot string for this op's derived produces/consumes.

        See :attr:`_DERIVED_SLOT` for the rationale.
        """
        return self._DERIVED_SLOT

    def _derive_produces_consumes(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return ``(produces, consumes_fresh)`` derived from rule contents.

        ``produces`` aggregates every dim name written by any FFN rule
        across every layer of ``compiler_ir`` and unions in ``self.writes``
        so attention-head ops with no FFN rules still publish their
        residual writes (the staleness analyzer only needs the dim name
        set; the slot string is a constant — see :attr:`_DERIVED_SLOT`).

        ``consumes_fresh`` is the subset of FFN rule condition dims
        (conditions / gate / gate_terms) that ALSO appears in
        ``self.reads`` and is NOT cross-step durable. We deliberately do
        not derive a ``consumes_fresh`` from ``self.reads`` alone — the
        declared reads include many cross-step / embed-time durables and
        would surface false-positive staleness warnings.  Only FFN rule
        condition dims (which represent same-step gated reads) feed the
        consumer set. Ops with no FFN rules consume nothing fresh by
        derivation (still pass an explicit override via the constructor
        if a test needs to assert otherwise).
        """
        write_names: Set[str] = set()
        cond_names: Set[str] = set()
        ir = self.compiler_ir
        if ir is not None and getattr(ir, "layers", None):
            for layer in ir.layers:
                ffn = getattr(layer, "ffn", None)
                if ffn is None:
                    continue
                for rule in ffn.rules:
                    for w in rule.writes:
                        write_names.add(w.dim.name)
                    for cond in rule.conditions:
                        cond_names.add(cond.dim.name)
                    if rule.gate is not None:
                        cond_names.add(rule.gate.name)
                    for term in rule.gate_terms:
                        cond_names.add(term.dim.name)
        # Union ``self.writes`` in so attention-head ops (no FFN rules,
        # imperative bake) still publish their residual writes.
        write_names |= set(self.writes or ())
        slot = self._derive_register_slot()
        produces = {name: slot for name in sorted(write_names)}
        reads = set(self.reads or ())
        consumes_candidates = {
            name for name in (cond_names & reads)
            if not self._is_cross_step_dim_name(name)
        }
        consumes_fresh = {name: slot for name in sorted(consumes_candidates)}
        return produces, consumes_fresh

    @property
    def produces(self) -> Dict[str, str]:
        """Computed-property view of in-step residual dim writes.

        Returns the explicit override when set; otherwise derives from
        ``compiler_ir`` rule contents (every dim written by any FFN rule's
        ``writes``). Empty IR yields ``{}``.
        """
        if self._produces_explicit is not None:
            return self._produces_explicit
        return self._derive_produces_consumes()[0]

    @produces.setter
    def produces(self, value):
        # Preserves the legacy ``op.produces = ...`` mutate-after-construct
        # pattern (used by validation tests in test_staleness_invariants).
        self._produces_explicit = value

    @property
    def consumes_fresh(self) -> Dict[str, str]:
        """Computed-property view of in-step residual dim reads.

        Returns the explicit override when set; otherwise derives from
        ``compiler_ir`` rule conditions ∩ ``self.reads`` (minus the
        cross-step-durable allowlist).
        """
        if self._consumes_fresh_explicit is not None:
            return self._consumes_fresh_explicit
        return self._derive_produces_consumes()[1]

    @consumes_fresh.setter
    def consumes_fresh(self, value):
        # Mirror of the ``produces`` setter — see above.
        self._consumes_fresh_explicit = value

    # ---- Derive reads/writes from rules (migration entry point) ---------
    # See docs/UNDECLARED_DIM_AUDIT_2026_06_09.md follow-up and
    # ``unified_compiler/op_introspect.py`` for the underlying derivation.
    def derive_reads_writes(
        self,
        *,
        dim_positions: Optional[Mapping[str, int]] = None,
        dim_sizes: Optional[Mapping[str, int]] = None,
        head_dim: int = 64,
    ) -> "Operation":
        """Return a copy with ``reads`` / ``writes`` filled in from the IR.

        Walks ``self.compiler_ir`` (or invokes ``compiler_ir_factory``)
        and aggregates every dim name read / written across FFN rules and
        attention head specs. The returned ``Operation`` is otherwise
        byte-identical with ``self`` — only the ``reads`` / ``writes``
        slots change.

        Use this in op-factory call sites to drop the manual annotation:

            return make_some_op().derive_reads_writes(
                dim_positions=compiler.dim_positions,
                dim_sizes=compiler.dim_sizes,
            )

        Attention head specs encode Q/K/V/O as resolved residual-column
        ints; the derivation needs the layout's ``dim_positions`` /
        ``dim_sizes`` to reverse-map them. Pure-FFN ops can omit them
        (the derivation gracefully skips attention contributions when
        the maps are not supplied — pure-FFN ops have no attention
        spec to walk).
        """
        from .op_introspect import derive_operation

        return derive_operation(
            self,
            dim_positions=dim_positions,
            dim_sizes=dim_sizes,
            head_dim=head_dim,
        )


def _install_legacy_init_aliases() -> None:
    """Wrap ``Operation.__init__`` so callers can still pass ``produces=``
    and ``consumes_fresh=`` kwargs (the override slots).

    Step 5 of IR_INCREMENTAL_IMPROVEMENTS.md renamed the InitVar fields to
    ``produces_override`` / ``consumes_fresh_override`` so they don't
    collide with the ``@property`` accessors of the same public names.
    Test fixtures (``c4_release/tests/test_staleness_invariants.py`` and a
    handful of historical call sites) still pass ``produces={...}`` /
    ``consumes_fresh={...}`` to the constructor; this wrapper translates
    those kwargs into the override slots without disturbing the dataclass-
    generated ``__init__``.
    """

    original_init = Operation.__init__

    def _init(self, *args, **kwargs):
        if "produces" in kwargs:
            kwargs["produces_override"] = kwargs.pop("produces")
        if "consumes_fresh" in kwargs:
            kwargs["consumes_fresh_override"] = kwargs.pop("consumes_fresh")
        original_init(self, *args, **kwargs)

    _init.__doc__ = original_init.__doc__
    Operation.__init__ = _init


_install_legacy_init_aliases()


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
    head_dim = _derive_head_dim(op, target, attn, dim_positions)
    return factory(dim_positions, head_dim)


def _derive_head_dim(op: Operation, target, attn, dim_positions) -> int:
    """Derive ``head_dim`` for declarations-only IR factories.

    Step 3 (literal-fallback lint, audit 2026-06-03): the previous
    implementation silently substituted ``head_dim = 64`` whenever the
    target's attention module was unavailable. That was the exact
    failure shape of the L10 ``d_model=512`` fallback (cf. l10_ops.py
    derivation chain) — a wider model or a 12-head attention would
    have its IR factory invoked against a mismatched slot stride.

    Derivation order:
      1. ``attn.W_q.shape[0] // attn.num_heads`` — the canonical
         source whenever attention weights exist.
      2. ``attn.head_dim`` — set explicitly by structural resize
         passes (e.g. ``_resize_l15_attention``).
      3. ``getattr(target, 'attn', None).head_dim`` / ``W_q`` chain
         when ``target`` is a block but ``attn`` was passed directly.
      4. Explicit error — no silent literal substitution. Callers
         that genuinely need an attention-less factory should pass
         a target that exposes ``head_dim`` or migrate the IR to
         not depend on it.
    """
    if attn is not None:
        W_q = getattr(attn, "W_q", None)
        num_heads = getattr(attn, "num_heads", None)
        if W_q is not None and num_heads:
            try:
                return int(W_q.shape[0]) // int(num_heads)
            except (AttributeError, IndexError, TypeError, ZeroDivisionError):
                pass
        explicit = getattr(attn, "head_dim", None)
        if explicit is not None:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass
    raise DeclarationsOnlyBakeError(
        [op.name],
        [
            f"{operation_display_label(op)} — head_dim is undeterminable: "
            "target attention has no W_q/num_heads and no explicit "
            "head_dim attribute. Bare-literal fallback (head_dim=64) "
            "was removed by Step 3 (see "
            "docs/LITERAL_FALLBACK_AUDIT.md)."
        ],
    )


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
        layer_spec = ir.layer(0)
        # Phase 8.I closing audit: structural ops resize ``target.attn``
        # (number of heads, alibi slopes, W_{q,k,v,o} re-allocation) and
        # may run a follow-up imperative pass that depends on the new
        # shape. Dispatch them BEFORE attn/ffn lowering so any attention
        # specs declared on the same IR see the post-resize block.
        if layer_spec.structural_ops:
            ir.lower_structural_ops(
                target,
                dim_positions,
                layer_idx=0,
                S=S,
            )
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
         When ``enable_dim_liveness=True`` (constructor arg, default OFF;
         also gated by ``C4_DIM_LIVENESS=1`` env var), step 4 instead
         runs a register-allocation-style graph colouring over the
         scheduled lifetimes — see ``_compute_dim_layout_with_liveness``.
    """

    def __init__(self, *, enable_dim_liveness: Optional[bool] = None):
        self.ops: List[Operation] = []
        self.dims: Dict[str, int] = {}  # name -> size
        self._op_by_name: Dict[str, Operation] = {}
        # Block-level and model-level ops are bake-only; they don't participate
        # in dim-position allocation, so they're held separately.
        self.block_ops: List[Operation] = []
        self.model_ops: List[Operation] = []
        # Compile-time dim slot sharing via liveness analysis. ``None``
        # defers to the ``C4_DIM_LIVENESS`` env var; default OFF preserves
        # bump-pointer behaviour. See ``_compute_dim_layout_with_liveness``.
        if enable_dim_liveness is None:
            enable_dim_liveness = _env_flag_dim_liveness()
        self.enable_dim_liveness: bool = bool(enable_dim_liveness)
        # Populated by ``compile()`` after liveness allocation. Each entry
        # describes one liveness-merged slot:
        #   {"slot_idx": int, "members": [dim_name, ...], "size": int}
        # Empty when liveness is disabled. Used by tests/diagnostics and by
        # the metrics report in ``liveness_savings_report``.
        self.liveness_slots: List[Dict[str, Any]] = []
        self.liveness_stats: Dict[str, Any] = {}

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
                # Sentinel keys prefixed with ``__`` (e.g.
                # ``__module_replacement``) are structural-effect
                # annotations, not residual-dim references. Bake authors
                # use them on ops whose ``bake_fn`` swaps a whole
                # submodule (``model.blocks[N].ffn`` -> {ALU class}); the
                # sentinel documents the structural effect without
                # claiming any (layer, scope, identifier, column) cell.
                # Skip the ``dim_name in self.dims`` check for these
                # keys -- they are intentionally NOT declared dims.
                if dim_name.startswith("__"):
                    continue
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
        # An invalid slot_share entry would silently disable nothing
        # without this check (typo like "fnn_units" → no opt-out).
        if not isinstance(op.slot_share, (tuple, list)):
            raise ValueError(
                f"Op {op.name!r} slot_share must be a tuple of slot-kind "
                f"strings; got {type(op.slot_share).__name__}"
            )
        for kind in op.slot_share:
            if kind not in _SLOT_REGISTRY_ALLOWED_KINDS:
                raise ValueError(
                    f"Op {op.name!r} slot_share entry {kind!r} is not a "
                    f"recognized slot kind. Allowed: "
                    f"{sorted(_SLOT_REGISTRY_ALLOWED_KINDS)}"
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
        # Dead-consumer scan: warn on any dim read by an op but never
        # written by any op. Surfaced by the L8 sp_gather STACK0 audit
        # (``docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md``). Opt out
        # via ``C4_SKIP_DIM_INTEGRITY=1``.
        from .dim_integrity import run_dim_integrity_check
        self._last_dim_integrity = run_dim_integrity_check(self)

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

        if self.enable_dim_liveness:
            # Resolve block-op layer assignments for the lifetime walker so
            # dims consumed only by kind="block" ops (e.g. AX_FULL_LO read
            # by layer15_alu_high_byte_relay) get the correct last-use
            # layer. ``self.block_ops`` is the authoritative store; the
            # local ``block_ops`` filter above runs over ``self.ops``,
            # which never contains block ops since ``add_op`` routes them
            # to ``self.block_ops``. We thread these into a transient copy
            # of ``layer_assignment`` to avoid perturbing the downstream
            # ``ops_per_layer`` placement (which expects attn/ffn only).
            block_layer_assignment = dict(layer_assignment)
            for op in self.block_ops:
                if op.layer_idx is not None:
                    block_layer_assignment[op.name] = op.layer_idx
                elif op.target_op_name is not None:
                    tgt = layer_assignment.get(op.target_op_name)
                    if tgt is not None:
                        block_layer_assignment[op.name] = tgt
            dim_positions = self._compute_dim_layout_with_liveness(
                attn_ffn_ops + list(self.block_ops), block_layer_assignment
            )
        else:
            dim_positions = self._allocate_dims()

        # Attention-gate effectivity audit. Walks every op's
        # declarative attention spec and classifies each Q-side
        # condition gate (``MARK_*`` / ``OP_*`` / ``HAS_*`` / ``IS_*``)
        # against its K-side. Emits a single warning summarising the
        # no-op (K=CONST-only) and q-only (no K at slot) cases — the
        # 70 instances catalogued in
        # ``c4_release/docs/Q_SIDE_GATE_AUDIT_2026_06_07.md``.
        #
        # Opt out via ``C4_SKIP_GATE_CHECK=1``; promote to a hard error
        # via ``C4_STRICT_GATE_CHECK=1``.
        from .dsl_interpreter import run_attention_gate_audit
        self._last_attention_gate_audit = run_attention_gate_audit(
            self, dim_positions,
        )

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
        # Use a sentinel (``None``) to distinguish "layer has no annotated
        # op" (fall back to default 4096) from "layer is annotated with 0
        # units" (explicit zero-FFN sentinel for attention-only layers per
        # docs/DEAD_UNIT_AUDIT_2026_06_05.md). The plain ``>``-with-zero-
        # prev check below would have dropped the explicit 0 because
        # ``0 > 0`` is False.
        ffn_widths_acc: Dict[int, Optional[int]] = {}

        def _record_width(layer_idx: int, width: int) -> None:
            prev = ffn_widths_acc.get(layer_idx)
            if prev is None or width > prev:
                ffn_widths_acc[layer_idx] = width

        for layer_idx, ops_at_layer in enumerate(ops_per_layer):
            for op in ops_at_layer:
                if op.ffn_units_used is not None:
                    _record_width(layer_idx, op.ffn_units_used)
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
            _record_width(target_layer, op.ffn_units_used)
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
            _record_width(op.layer_idx, op.ffn_units_used)

        ffn_widths: Dict[int, int] = {
            li: w for li, w in ffn_widths_acc.items() if w is not None
        }

        # Phase 1 of memory cluster fix plan: slot-conflict registry scan.
        # Records every ``(layer, slot_id)`` claim derived from each op's
        # produces-sentinel / compiler_ir / ffn_units_used. Conflicts that
        # are not opted-into via ``Operation.slot_share`` raise
        # ``SlotConflictError``. The scan runs AFTER layer assignment so we
        # have the final per-op layer for every attn/ffn/block op.
        # NOTE: ``ops_per_layer`` carries only attn/ffn ops. Block ops live
        # in ``self.block_ops`` and resolve their layer via the same
        # ``layer_assignment`` map populated above.
        self._run_slot_conflict_scan(ops_per_layer, layer_assignment)

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
    # Slot-conflict registry (Phase 1 of memory cluster fix plan,
    # docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md). See
    # ``slot_registry.py`` for the schema. Catches the silent-overwrite
    # class of bug that V2/V3/V4 of the memory cluster fix all regressed
    # against. The scan is gated on a flag so the integration can be
    # disabled if a downstream regression surfaces — set
    # ``LayerCompiler.disable_slot_registry = True`` (class attribute) or
    # set the environment variable ``C4_DISABLE_SLOT_REGISTRY=1`` to skip.
    # --------------------------------------------------------------------

    def _run_slot_conflict_scan(
        self,
        ops_per_layer: List[List["Operation"]],
        layer_assignment: Dict[str, int],
    ) -> None:
        """Build a :class:`SlotRegistry` from placed ops and raise on conflict.

        Three op classes contribute claims:

          * attn/ffn ops, layer comes from ``ops_per_layer``;
          * block ops, layer resolved via ``layer_assignment`` then
            falling back to ``op.layer_idx`` / ``op.target_op_name``;
          * model-level ops with an explicit ``layer_idx``.

        Model ops without a ``layer_idx`` are skipped — head/embedding
        bakes touch every block uniformly and don't claim a structural
        slot at a specific layer.
        """
        if (
            getattr(type(self), "disable_slot_registry", False)
            or os.environ.get("C4_DISABLE_SLOT_REGISTRY", "") == "1"
        ):
            return
        registry = SlotRegistry()

        def _record(op: "Operation", layer: int) -> None:
            for slot_id in derive_slot_ids_for_op(op):
                registry.claim(
                    op_name=op.name,
                    layer_idx=layer,
                    slot_id=slot_id,
                    op_kind=op.kind,
                    slot_share=tuple(op.slot_share),
                )

        for layer_idx, ops_at_layer in enumerate(ops_per_layer):
            for op in ops_at_layer:
                _record(op, layer_idx)

        for op in self.block_ops:
            # ``compile()`` populates ``layer_assignment`` for block ops
            # that arrived via ``self.ops``, but every block op since
            # Phase 8.A.4 lands in ``self.block_ops`` via ``add_op`` and
            # so is absent from ``layer_assignment``. Resolve target
            # layer like :meth:`ModelLayout.resolve_block_op_layer`:
            # ``layer_idx`` wins, else ``target_op_name`` → that op's
            # layer.
            target_layer = layer_assignment.get(op.name)
            if target_layer is None and op.layer_idx is not None:
                target_layer = int(op.layer_idx)
            if target_layer is None and op.target_op_name is not None:
                target_layer = layer_assignment.get(op.target_op_name)
            if target_layer is None:
                # ``compile()`` would have raised on an unresolved block
                # op already; this branch is defensive.
                continue
            _record(op, int(target_layer))

        for op in self.model_ops:
            if op.layer_idx is not None:
                _record(op, int(op.layer_idx))

        # Audit tools (and the slow integration test in
        # tests/test_slot_registry.py) read this attribute to inspect
        # the registry without re-running the derivation.
        self._last_slot_registry = registry
        registry.raise_on_conflict()

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

    # ------------------------------------------------------------------
    # Compile-time dim slot sharing via liveness analysis (Phase 11.A).
    #
    # Replaces the bump-pointer allocator with a register-allocation-style
    # graph colouring over the residual-stream dims:
    #
    #   1. For each unpinned, non-aliased dim, compute a lifetime
    #      interval [def_layer, last_use_layer] from the scheduled layer
    #      assignment plus ``find_producers`` / ``find_consumers``.
    #   2. Cross-step dims (read as ``DIM.*.-1`` or in the cross-step
    #      durable allowlist) and every MARKER / OPCODE_FLAG / CONST dim
    #      are forced live-forever — they NEVER share.
    #   3. Build the interference graph: two dims interfere iff their
    #      lifetime intervals overlap by any layer.
    #   4. Greedy-colour the graph (first-fit, ordered by lifetime start).
    #      Bands (width > 1) share only with bands of the same width;
    #      single-cell scalars share with other compatible single-cell
    #      slots. Each colour class collapses onto one residual slot.
    #   5. Lay the colour classes out after the highest pinned endpoint
    #      so pinned dims and aliases continue to follow the bump-pointer
    #      semantics. Aliases bind to their base's chosen position last.
    #
    # Soundness invariant: dims that share a slot must NEVER be live at
    # runtime simultaneously. The lifetime computation is a conservative
    # over-approximation: cross-step durables and all role-bearing markers
    # are excluded from sharing, so a false-sharing bug would have to come
    # from a scratch dim with an under-reported lifetime. Tests enforce
    # the disjoint-lifetime contract end-to-end.
    # ------------------------------------------------------------------

    _LIVENESS_NEVER_SHARE_PREFIXES: Tuple[str, ...] = (
        "MARK_",
        "OP_",
        "NEXT_",
        "IS_",
        "HAS_",
        "L1H",
    )
    _LIVENESS_NEVER_SHARE_NAMES: frozenset = frozenset({
        "CONST",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "EMBED_LO", "EMBED_HI",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
        "OUTPUT_LO", "OUTPUT_HI", "OUTPUT_HI_THIS_STEP", "OUTPUT_HI_PREV_STEP",
        "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
        "MEM_STORE", "MEM_ADDR_SRC",
        "PSH_AT_SP",
        "LAST_WAS_BYTE", "IO_IS_PUTCHAR", "IO_IS_PRTF",
        "IO_IN_OUTPUT_MODE", "IO_OUTPUT_COMPLETE",
        "IO_STATE", "IO_FORMAT_POS",
        "OPCODE_BASE", "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
        "OPCODE_BASE_BYTE0", "OPCODE_BASE_BYTE1",
        "OP_LC_RELAY", "OP_LI_RELAY", "OP_SI_RELAY",
        "MARK_CS", "MARK_THINKING_START", "MARK_THINKING_END",
        "MARK_SE_ONLY", "MARK_SE", "MARK_MEM", "MARK_HALT", "MARK_BP",
        "MARK_STACK0", "MARK_STACK1", "MARK_STACK2",
        "IS_MARK",
        "CMP",
        # CMP cascade per-opcode bytes — embed-time durable.
        "CMP+0", "CMP+1", "CMP+2", "CMP+3", "CMP+4",
        "CMP+5", "CMP+6", "CMP+7", "CMP+8", "CMP+9",
        # 2026-06-10 Wave A v2: register-tagged STEP_END operand relay
        # mirror dims. Written at MARK_SE_ONLY by
        # ``layer9_step_end_operand_relay`` (two L9 attn heads) AND at
        # MARK_AX rows as a small softmax leak (the score-only slot-0
        # design reduces but does not eliminate the leak). Sharing a
        # slot with a live dim that fires at MARK_AX (e.g.
        # IN_STEP_FRESH, SP_BYTE0_IS_F8) would corrupt the partner.
        # See docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md and
        # memory note ``project_wave_b_cmp_needs_l9_internal_relay.md``.
        "SE_ALU_LO", "SE_ALU_HI",
        "SE_AX_CARRY_LO", "SE_AX_CARRY_HI",
        "SE_CMP", "SE_CMP_GROUP",
        "SE_OP_EQ", "SE_OP_NE", "SE_OP_LT",
        "SE_OP_GT", "SE_OP_LE", "SE_OP_GE",
    })

    def _liveness_never_share(self, name: str) -> bool:
        """Return True if a dim must keep a private slot (no sharing).

        Markers, opcode flags, NEXT_* predictors, structural / output bands,
        and the cross-step-durable allowlist all stay private even when
        their per-step lifetimes look short — they are live across VM
        steps or carry role identity that downstream queries assume is
        stable. This is a conservative over-approximation: false negatives
        (sharing-eligible dims wrongly flagged) cost slot budget but never
        correctness.
        """
        if name in self._LIVENESS_NEVER_SHARE_NAMES:
            return True
        for prefix in self._LIVENESS_NEVER_SHARE_PREFIXES:
            if name.startswith(prefix):
                return True
        if Operation._is_cross_step_dim_name(name):
            return True
        return False

    def _compute_dim_lifetimes(
        self,
        attn_ffn_ops: List[Operation],
        layer_assignment: Dict[str, int],
    ) -> Dict[str, Tuple[int, int]]:
        """Compute ``dim_name -> (def_layer, last_use_layer)`` intervals.

        Walks every attn/ffn op and tags the dim's def-layer with the
        smallest layer that writes it, and the last-use layer with the
        largest layer that reads it. Dims read across steps (``DIM.*.-1``)
        extend their last-use layer to the end of the schedule (effectively
        always live). Dims with no recorded layer (model-level or
        block-only writes) get a degenerate (0, n_layers-1) interval so
        the colouring treats them as long-lived rather than dead.

        Returns a dict keyed by *unversioned* base dim names — SSA forms
        (``BASE.WRITER.STEP``) are resolved to their base via the alias
        map so the lifetime tracks the physical residual slot, not the
        abstract version.
        """
        from .ssa_dim import base_of, is_ssa_form

        n_layers = (max(layer_assignment.values()) + 1) if layer_assignment else 1
        # Use n_layers - 1 as the "end" marker; we cap "live forever" to it.
        last_layer = max(0, n_layers - 1)

        def_layer: Dict[str, int] = {}
        last_use: Dict[str, int] = {}

        for op in attn_ffn_ops:
            layer = layer_assignment.get(op.name)
            if layer is None:
                continue
            for w in op.writes:
                base = base_of(w) if is_ssa_form(w) else w
                if base in def_layer:
                    if layer < def_layer[base]:
                        def_layer[base] = layer
                else:
                    def_layer[base] = layer
                # A write also counts as a use at its layer (covers
                # read-modify-write patterns).
                if base not in last_use or layer > last_use[base]:
                    last_use[base] = layer
            for r in op.reads:
                base = base_of(r) if is_ssa_form(r) else r
                # Cross-step reads extend live-to-end of the schedule.
                extended = Operation._is_cross_step_dim_name(r) or (
                    is_ssa_form(r) and parse_ssa_name(r).is_cross_step
                )
                if extended:
                    last_use[base] = last_layer
                else:
                    if base not in last_use or layer > last_use[base]:
                        last_use[base] = layer

        lifetimes: Dict[str, Tuple[int, int]] = {}
        for name in self.dims:
            has_def = name in def_layer
            has_use = name in last_use
            if not has_def and not has_use:
                # Dim is declared but no op in the IR writes OR reads it.
                # The conservative default of (0, last_layer) treats such
                # dims as live-forever, blocking the slot from being
                # donated. For truly-dead dims (e.g. ``MUL_ACCUM`` after
                # the L11 lookup-mul migration moved staging to ``TEMP``),
                # this hides a free-slot opportunity. Mark them dead at
                # the first layer so later same-width shareable dims can
                # claim the slot. Soundness: a dim with no producers AND
                # no consumers cannot interfere with any other dim's
                # value, by definition; making it shareable is purely a
                # slot-budget win. Never-share dims still skip sharing
                # via ``_liveness_never_share`` regardless of lifetime.
                lifetimes[name] = (0, 0)
                continue
            d = def_layer.get(name, 0)
            u = last_use.get(name, last_layer)
            # If a dim is read before it is written (model-level writer),
            # widen the interval to cover both endpoints conservatively.
            if u < d:
                u = d
            lifetimes[name] = (d, u)
        return lifetimes

    def _compute_dim_layout_with_liveness(
        self,
        attn_ffn_ops: List[Operation],
        layer_assignment: Dict[str, int],
    ) -> Dict[str, int]:
        """Allocate dim positions with slot sharing via liveness analysis.

        Returns a ``dim_name -> position`` mapping just like
        :meth:`_allocate_dims`. Pinned dims and aliases retain their
        bump-pointer semantics; sharing runs only over unpinned,
        non-aliased dims that are eligible.

        Layout strategy: preserve the bump-pointer dim ORDERING (so the
        starting position of every dim is at most the bump-pointer
        position) and only collapse a dim back onto an EARLIER dim's
        slot when the earlier dim's lifetime has fully ended before the
        new one starts. This guarantees:

          1. ``d_model`` never grows compared with bump-pointer.
          2. Downstream consumers that hardcode dim positions
             (e.g. ``efficient_alu_neural.py`` bakes a 512-wide W_proj
             that indexes ``BD.ALU_LO + k``) still see ALU_LO at the
             same or earlier slot index.
          3. The shared-slot members all have disjoint lifetimes by
             construction.

        The method is pure: it does NOT mutate ``self.dims`` or
        ``self._pinned`` / ``self._aliases``. It updates two pieces of
        bookkeeping for diagnostics:

          * ``self.liveness_slots`` — one entry per residual slot the
            sharing pass produced, listing the dims that share it.
          * ``self.liveness_stats`` — coarse counters (total dims,
            shared dims, savings) reported by
            :meth:`liveness_savings_report`.
        """
        pinned = getattr(self, "_pinned", {}) or {}
        aliases = getattr(self, "_aliases", {}) or {}

        positions: Dict[str, int] = {}
        # 1. Place pinned dims (excluding aliases) at their requested slot.
        for name, pos in pinned.items():
            if name in aliases:
                continue
            positions[name] = pos
        max_pinned_end = 0
        for name, pos in pinned.items():
            if name in aliases:
                continue
            max_pinned_end = max(max_pinned_end, pos + self.dims[name])

        # 2. Compute lifetimes for every dim.
        lifetimes = self._compute_dim_lifetimes(attn_ffn_ops, layer_assignment)

        # 3. Bump-pointer the unpinned, non-aliased dims in DECLARATION
        # order — but each shareable dim first tries to slot into an
        # EARLIER shareable dim's position whose lifetime ended before
        # ours starts and whose width matches. Earlier dims keep their
        # bump-pointer positions exactly, so any downstream consumer
        # that holds ``positions[X]`` as a structural index sees X at
        # the SAME slot it would have in the bump-pointer build.
        #
        # ``slot_index[size] = [(position, dim_name, last_use), ...]``
        # tracks every existing slot we might donate to a later dim,
        # bucketed by width.
        slot_index: Dict[int, List[Tuple[int, str, int]]] = {}
        # Reverse-lookup: position -> (slot_idx, members)
        slot_classes: List[Dict[str, Any]] = []
        position_to_class: Dict[int, int] = {}

        cursor = max_pinned_end
        for name, size in self.dims.items():
            if name in pinned or name in aliases:
                continue
            d, u = lifetimes[name]
            shared = False
            if not self._liveness_never_share(name):
                # Find an earlier shareable slot whose last_use is
                # strictly before ``d``. Same-size bucket only.
                bucket = slot_index.get(size, [])
                for entry_idx in range(len(bucket)):
                    pos, donor, donor_last = bucket[entry_idx]
                    if donor_last < d:
                        # Donor's lifetime ended before our def — share.
                        positions[name] = pos
                        cls_idx = position_to_class[pos]
                        slot_classes[cls_idx]["members"].append(name)
                        if u > slot_classes[cls_idx]["last_use"]:
                            slot_classes[cls_idx]["last_use"] = u
                        # Update bucket's last_use so a still-later
                        # shareable dim sees the union.
                        bucket[entry_idx] = (
                            pos, donor,
                            max(donor_last, u),
                        )
                        shared = True
                        break
            if shared:
                continue
            # Fresh slot at the bump cursor.
            positions[name] = cursor
            cls_idx = len(slot_classes)
            slot_classes.append({
                "size": size,
                "position": cursor,
                "members": [name],
                "last_use": u,
                "first_def": d,
                "shareable": not self._liveness_never_share(name),
            })
            position_to_class[cursor] = cls_idx
            # Only register as a donor if the dim is shareable; never-share
            # dims (markers, opcode flags, etc.) must NEVER be donated.
            if not self._liveness_never_share(name):
                slot_index.setdefault(size, []).append((cursor, name, u))
            cursor += size

        # 4. Aliases resolve last (same as bump-pointer path).
        for name in list(aliases):
            base = aliases[name]
            for _ in range(len(self.dims) + 1):
                if base not in aliases:
                    break
                base = aliases[base]
            if base not in positions:
                raise ValueError(
                    f"Alias {name!r} references unknown base dim {base!r}"
                )
            positions[name] = positions[base]

        # 5. Soundness check: every multi-member slot's members must
        # have pairwise-disjoint lifetimes.
        for cls in slot_classes:
            if len(cls["members"]) < 2:
                continue
            members = cls["members"]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    ad, au = lifetimes[a]
                    bd_, bu = lifetimes[b]
                    if not (au < bd_ or ad > bu):
                        raise RuntimeError(
                            "dim-liveness soundness check failed: dims "
                            f"{a!r} (live [{ad},{au}]) and {b!r} (live "
                            f"[{bd_},{bu}]) share slot but lifetimes overlap"
                        )

        # 6. Record diagnostics for ``liveness_savings_report``.
        self.liveness_slots = [
            {
                "slot_idx": idx,
                "members": list(cls["members"]),
                "size": cls["size"],
                "position": cls["position"],
            }
            for idx, cls in enumerate(slot_classes)
        ]
        merged_dims = sum(
            len(cls["members"]) for cls in slot_classes
            if len(cls["members"]) > 1
        )
        # Count dims that were *donated* into vs. fresh allocations.
        saved_dim_count = sum(
            len(cls["members"]) - 1 for cls in slot_classes
            if len(cls["members"]) > 1
        )
        shareable_dim_count = sum(
            len(cls["members"]) for cls in slot_classes
            if cls.get("shareable")
        )
        private_dim_count = sum(
            len(cls["members"]) for cls in slot_classes
            if not cls.get("shareable")
        )
        self.liveness_stats = {
            "total_dims": len(self.dims),
            "pinned": sum(1 for n in self.dims if n in pinned and n not in aliases),
            "aliases": len(aliases),
            "private": private_dim_count,
            "shareable": shareable_dim_count,
            "slot_classes": len(slot_classes),
            "merged_dims": merged_dims,
            "saved_dim_count": saved_dim_count,
            "d_model": cursor,
        }
        return positions

    def liveness_savings_report(self) -> Dict[str, Any]:
        """Return a structured savings report for the most recent compile.

        Empty when liveness was not enabled. Reported counters mirror
        ``self.liveness_stats`` and add a per-category breakdown computed
        from the dim-name schema. Useful for the headline numbers in
        the smoke / CI summary.
        """
        if not self.liveness_stats:
            return {}
        try:
            from .ir_types import schema_for
        except Exception:
            schema_for = None  # type: ignore[assignment]

        category_counts: Dict[str, int] = {}
        category_shared: Dict[str, int] = {}
        for slot in self.liveness_slots:
            for member in slot["members"]:
                if schema_for is not None:
                    cat = schema_for(member).type.name
                else:
                    cat = "UNKNOWN"
                category_counts[cat] = category_counts.get(cat, 0) + 1
                if len(slot["members"]) > 1:
                    category_shared[cat] = category_shared.get(cat, 0) + 1
        return {
            **self.liveness_stats,
            "category_counts": category_counts,
            "category_shared": category_shared,
            "slots": list(self.liveness_slots),
        }


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
