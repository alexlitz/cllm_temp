"""ISA-semantics DSL — declare register/value-flow semantics, generate the
low-level ``FFNRule`` / ``DeclarativeAttentionHeadSpec`` / residual-band
scaffolding.

This module sits ABOVE :mod:`building_blocks_dsl` (and below the per-layer
``ops/lN_ops.py`` factories that import it). It does NOT add any compiler
machinery: it emits the EXACT artifact shapes the hand-built carry/dump ops
already produce — a ``register_residual_band`` side effect, a
``DeclarativeAttentionHeadSpec`` builder, and a tuple of ``FFNRule`` dump
rules — so the compiler sees ordinary block ops.

First (and most-repeated) pattern: :func:`cross_step_carry`. The hand-built
cross-step carries (AX byte-1 ``l11_ops.py:1069``, BP_SAVE_PREV ``:3212``,
STACK0 byte-0 ``:2064``) share an IDENTICAL 4-part structure:

  1. a dedicated ``_PREV`` residual band (``register_residual_band``,
     ``never_share=True``, declared at MODULE-IMPORT scope — registration
     order is load-bearing for the tail ``dim_positions``);
  2. an UNCONDITIONAL carry head — ``kind="block"`` op bound to a layer anchor,
     ``alibi_slope=0.5``, Q/K match the prev row via a per-byte positional
     signature, V reads ``<src>.*.-1`` cross-step, O writes the ``_PREV`` band;
  3. a DUMP FFN — ``kind="block"`` post-op at the late (L25) tail, per-cell
     ``multi_way_and_rule`` GATED on the carried ``_PREV+j`` band, writing the
     emit band — THIS is where the carry-vs-fresh gate lives;
  4. the band-pass / kill discriminator, expressed IN the dump's gate
     conditions (so the carry head stays unconditional).

The generator takes the VARYING params via :class:`CrossStepCarrySpec` and
supplies the IDENTICAL structure. The head-unconditional + gate-in-FFN split
is ENFORCED BY THE API SHAPE: there is NO head-gate field. Gating can only
flow through :attr:`CrossStepCarrySpec.dump_gate_conditions`.

The first migration (BP_SAVE_PREV) re-expresses the hand-built
``_bp_save_prev_carry_head_spec`` + ``_bp_save_dump_repopulate_rules`` via
:func:`cross_step_carry`. It is byte-identity-gated against the HEAD golden
``state_dict`` hash (flag-on AND flag-off) — the proof the generator
reproduces hand-built weights bit-for-bit before it is trusted for new code.

See ``docs/...`` design plan + ``c4_release/CLAUDE.md`` (op-local residual
bands).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from .building_blocks_dsl import multi_way_and_rule
from .ir import FFNRule, StepWindowConstraint, TokenEmbeddingRule
from .primitives import AO, AP, DeclarativeAttentionHeadSpec
from .ops.residual_band_registry import register_residual_band


# ---------------------------------------------------------------------------
# Position-source split (the AX/STACK0 generalization over BP's layout-only)
# ---------------------------------------------------------------------------
#
# BP_SAVE_PREV resolves EVERY dim from the declarative LAYOUT (``dim_positions``):
# the legacy registry dims it taps (OP_ENT, OUTPUT_LO/HI, MARK_*) happen to sit
# at the SAME index in both maps OR the model residual carries them at the
# layout index. So ``position_source="layout"`` resolves the whole carry from
# ``dim_positions``.
#
# The AX byte-1 + STACK0 byte-0 carries are different: their H1/H2/H3 emission
# pipeline + the ADDR_B*/AX_CARRY/MARK_* gate taps are baked by the LEGACY
# imperative path at the dynamic *registry* positions (H1=67, ADDR_B0_LO=12,
# AX_CARRY_LO=328, ...), which DIFFER from the declarative layout
# (ADDR_B0_LO=506, AX_CARRY_LO=362). Reading those from the layout taps DEAD
# slots; they MUST resolve from ``build_default_registry_dynamic()`` — the same
# map every probe uses. Only the NEW carry bands (``H1_PREV_STEP`` etc.) live
# in the layout. ``position_source="mixed"`` + ``registry_dims`` expresses this
# split: a base dim name listed in ``registry_dims`` resolves from the registry,
# everything else from ``dim_positions``.


def _make_position_resolver(
    dim_positions: Dict[str, int],
    *,
    position_source: str,
    registry_dims: Sequence[str],
) -> Callable[[str], int]:
    """Return a ``base_dim_name -> int`` resolver honouring the split.

    ``position_source="layout"`` => every dim from ``dim_positions``.
    ``position_source="mixed"`` => dims in ``registry_dims`` from the dynamic
    registry (``build_default_registry_dynamic()``), the rest from
    ``dim_positions``. Building the registry is lazy (only the mixed path pays
    the import + build cost), matching the hand-built ops that build it inside
    their head-spec / bake bodies.
    """
    if position_source not in ("layout", "mixed"):
        raise ValueError(
            f"position_source must be 'layout' or 'mixed', got "
            f"{position_source!r}"
        )
    reg_set = set(registry_dims)
    _reg_holder: Dict[str, object] = {}

    def _registry():
        if "reg" not in _reg_holder:
            from ..dim_registry_dynamic import build_default_registry_dynamic
            _reg_holder["reg"] = build_default_registry_dynamic()
        return _reg_holder["reg"]

    def resolve(name: str) -> int:
        if position_source == "mixed" and name in reg_set:
            return int(_registry().slots[name].start)
        return int(dim_positions[name])

    return resolve


def _resolve_dim_token(token: str, resolve: Callable[[str], int]) -> int:
    """Resolve a ``BASE`` or ``BASE+offset`` token to an int position."""
    if "+" in token:
        base, off = token.rsplit("+", 1)
        return resolve(base) + int(off)
    return resolve(token)


# ---------------------------------------------------------------------------
# Explicit head directives (the head-shape generalization over BP per-byte)
# ---------------------------------------------------------------------------
#
# BP's carry head is a per-byte positional MATCH (Q@MEM_VAL_B{k} <->
# K@BYTE_INDEX_{k}) + k_prefer/k_reject + a single LO/HI value-copy block. The
# AX byte-1 + STACK0 byte-0 heads are structurally DIFFERENT (a sharp MARK/ADDR
# signature on slot 0, a CONST-driven one-hot-presence or AX_CARRY-preference
# K-loop on its own slot, a MARK_AX V=0 sink on slot 3, and 1-3 cross-step
# value-copy blocks). Rather than contort the per-byte-match builder, the spec
# may declare the head's Q/K/V/O writes EXPLICITLY via :class:`HeadWrite`
# directives. ``generate_attention_head`` lowers by DIRECT indexed assignment
# (``W_q[base+slot, dim] = w``), so the resulting weights depend ONLY on the
# final (slot, dim, weight) set — NOT the emit order — and an explicit directive
# list reproduces any hand-built head byte-identically.


@dataclass(frozen=True)
class HeadWrite:
    """One Q/K/V/O directive in an explicit carry-head spec.

    A directive expands to ``count`` writes at ``(slot, dim_base + j*1)`` for the
    V/O value-copy blocks (``count``/``src_count`` per-cell loops) or a single
    write when ``count == 1``. ``dim`` is a base dim NAME (registry- or
    layout-resolved per ``position_source``); ``+offset`` suffixes are honoured.

    For Q/K writes (``v_slot`` unused): emits ``AP(slot + i*slot_stride,
    resolve(dim) + i*dim_stride, weight)`` for ``i in range(count)``.

    For V writes: emits ``AP(slot + i, resolve(dim) + i, weight)`` for
    ``i in range(count)`` (the per-cell copy block — V slot and src dim advance
    together).

    For O writes: emits ``AO(resolve(out_dim) + i, slot + i, weight)`` for
    ``i in range(count)`` (out band cell j written from V slot ``slot + i``).
    ``dim`` is the O OUT band; ``slot`` is the V slot base it reads.
    """

    slot: int
    dim: str
    weight: float
    count: int = 1
    slot_stride: int = 1
    dim_stride: int = 1


# ---------------------------------------------------------------------------
# Explicit dump-rule directives (the direct-repoint + heterogeneous-gate
# generalization over BP's per-byte LO/HI split)
# ---------------------------------------------------------------------------
#
# BP's dump is a uniform per-byte LO/HI gate-copy: ``OUTPUT_{LO,HI}[j] =
# BP_SAVE_PREV[j]`` gated on ``OP_ENT + MEM_VAL_B{k}``. The AX byte-1 + STACK0
# byte-0 dumps differ on two axes the BP form cannot express:
#   * DIRECT REPOINT: the dump WRITE band == the cross-step SRC band (STACK0
#     re-supplies into the byte's OWN ``H1+j``/``H3+j`` LM-head emission cell,
#     gated by the precursor flags; AX writes the distinct ``H*_DUMP_OUT``).
#   * heterogeneous, NON-per-byte gate conditions (precursor-flag ANDs, an
#     unbounded AX_CARRY band sum, a two-stage ``*_OVERFLOW`` KILL).
# A :class:`DumpBlock` declares one ``(gate_band -> emit_band)`` copy block of
# ``width`` cells sharing a single ``conditions``/``threshold`` AND; the dump is
# a tuple of blocks. ``emit_band`` may equal ``gate_band``'s SRC band (the
# repoint) or differ (the separate DUMP band).


@dataclass(frozen=True)
class DumpBlock:
    """One gated per-cell copy block in an explicit dump-rule spec.

    For each cell ``j in range(width)`` emits a ``multi_way_and_rule`` named
    ``{name_prefix}_{j}`` whose AND is ``conditions`` over ``threshold``, gated
    on ``{gate_band}+{j}`` (multiplicative), writing ``({emit_band}+{j},
    write_scale)``. ``conditions`` dims are base names (registry/layout-resolved
    by NAME at lower time); ``gate_band`` / ``emit_band`` likewise.
    """

    name_prefix: str
    width: int
    gate_band: str
    emit_band: str
    write_scale: float
    conditions: Tuple[Tuple[str, float], ...]
    threshold: float


@dataclass(frozen=True)
class PrecursorFlagSpec:
    """A precursor FFN that writes a BOUNDED gate flag the dump ANDs.

    The two-stage KILL (AX ``AX_CARRY_OVERFLOW``) and the carried/sharp/etc.
    STACK0 gate flags share an IDENTICAL op shape: a standalone ``PureFFN``
    post_op, bound to an anchor, with a MIXED dim_map (input taps from the
    legacy registry, the NEW flag band from the layout), writing a small bounded
    value the dump reads (positively for an enable flag, or via a large negative
    weight for a KILL flag). The varying part is the RULE LIST and the band /
    anchor; the generator supplies the bake scaffolding via
    :func:`make_mixed_dim_map_ffn_op`.

    Attributes:
        op_name: the Operation name (also the rule-count-assert label).
        flag_band: the NEW residual band the precursor writes (layout dim).
        rules_builder: ``() -> tuple[FFNRule, ...]`` — the flag logic. Owns the
            step/AND rules (domain-specific weights stay in the op module).
        reads / writes: the Operation dep-graph dim sets.
        target_op_name: the bind anchor.
        requires: optional ``{"after": (...)}`` ordering.
        registry_dims: base dim names resolved from the dynamic registry (the
            input taps); the ``flag_band`` resolves from the layout.
        spec_section: doc tag.
    """

    op_name: str
    flag_band: str
    rules_builder: Callable[[], Tuple[FFNRule, ...]]
    reads: Set[str]
    writes: Set[str]
    target_op_name: str
    requires: Optional[Dict[str, object]] = None
    registry_dims: Tuple[str, ...] = ()
    spec_section: Optional[str] = None


# ---------------------------------------------------------------------------
# Spec — the varying parameters of a cross-step carry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossStepCarrySpec:
    """Declarative description of a cross-step value carry.

    A cross-step carry re-emits a register/value byte that the producing step
    decodes cleanly but a LATER step's same-step attention fails to fetch (or
    a tail corruptor nukes). Instead of fixing the failing same-step head, the
    carry copies the clean value across the step boundary into a dedicated
    ``_PREV`` band (the carry HEAD), then a gated dump FFN re-supplies it into
    the emit band at the consuming row (the DUMP FFN).

    Every field is a VARYING parameter; the IDENTICAL 4-part structure is
    supplied by :func:`cross_step_carry`. There is deliberately NO head-gate
    field — the carry head is ALWAYS unconditional, and gating flows only
    through :attr:`dump_gate_conditions` (enforced by the API shape).

    Position-source contract
    ------------------------
    The BP_SAVE_PREV migration resolves EVERY dim from the declarative LAYOUT
    (``dim_positions``) — both the head's taps and the dump's gate / write
    dims. This is the ``position_source="layout"`` case. (The AX/STACK0
    carries mix layout + registry positions; that generalization lands on a
    later migration and adds a ``position_source`` discriminator — kept out of
    this first increment so the API stays minimal and the BP proof is clean.)

    Attributes:
        name: carry family name (used for the band owner + rule-name prefix).
        band_name: dedicated ``_PREV`` residual band name. Registered at
            import time by :func:`cross_step_carry`.
        band_width: residual band width (cells). The dump emits one rule per
            cell per consuming byte.
        band_flag: optional zero-arg predicate gating the WHOLE feature (the
            band + head + dump). ``None`` => always present. When supplied,
            flag-off omits the band (smaller d_model) and the head/dump bake
            as no-ops — byte-identical to the pre-carry build.
        carry_head_alibi_slope: per-head ALiBi slope (positive => prefer the
            NEAREST prev step). BP uses 0.5.
        carry_byte_count: number of consuming bytes (BP carries 4: each val
            byte k attends the prev BP byte k).
        match_q_band / match_k_band: per-byte positional-match band PAIR —
            ``match_q_band{k}`` (Q, the consuming row) <-> ``match_k_band{k}``
            (K, the prev row). BP: ``MEM_VAL_B{k}`` <-> ``BYTE_INDEX_{k}``.
        match_weight: per-byte positional-match projection weight.
        k_prefer / k_reject: K-side preference / rejection signature dims.
            ``k_prefer`` biases TOWARD the prev row (BP: ``OP_JSR``);
            ``k_reject`` HARD-rejects same-step / wrong-source rows (BP:
            ``OP_ENT`` and the ``STACK0_BYTE{0..3}`` family). Each entry is a
            ``(slot, dim_name, weight, is_band)`` group — see
            :func:`cross_step_carry` for the slot layout it produces.
        value_src_lo / value_src_hi: the clean cross-step value bands the head
            V-copies (BP: ``CLEAN_EMBED_LO`` / ``CLEAN_EMBED_HI``). Each is
            16-wide (a nibble one-hot). The head reads ``<src>.*.-1`` so the
            edge points at the PREV step.
        value_o_write_scale: O-write weight (boosted so the carried one-hot
            lands at a large magnitude the dump reads as its multiplicative
            gate). BP: 200.0.
        value_v_slot_base: head-local V slot base for the value-copy block.
        dump_emit_lo / dump_emit_hi: emit bands the dump writes (BP:
            ``OUTPUT_LO`` / ``OUTPUT_HI``). The first ``band_width//2`` cells
            of ``_PREV`` go to ``dump_emit_lo``, the rest to ``dump_emit_hi``.
        dump_write_scale: dump OUTPUT write magnitude (must net positive over
            the tail corruptor's sentinel garbage). BP: 200000.0.
        dump_gate_conditions: the carry-vs-fresh GATE — a tuple of
            ``(dim_name, weight)`` AND conditions SHARED across all dump cells
            of a byte, PLUS the per-byte row marker. THE ONLY gating surface.
            BP: ``("OP_ENT", 1.0)`` (the global discriminator).
        dump_per_byte_marker / dump_per_byte_marker_weight: per-byte row
            selector folded into the gate AND. BP: ``MEM_VAL_B{k}`` @ 8.0.
        dump_marker_blockers: structural ``MARK_*`` blockers ANDed into every
            dump cell (so the re-supply never fires on a register-marker row).
        dump_threshold: explicit AND threshold (BP: OP_ENT>=6 floor + the
            marker weight). When ``None`` it is derived as
            ``dump_opent_floor + dump_per_byte_marker_weight``.
        dump_opent_floor: the opcode-discriminator floor folded into the
            derived threshold (BP: 6.0). Ignored if ``dump_threshold`` is set.
        carry_head_extra_reads: extra dim names the carry head's Operation
            should declare as reads (e.g. ``CONST``) beyond the auto-derived
            set.
    """

    name: str
    band_name: str
    band_width: int
    carry_head_alibi_slope: float
    carry_byte_count: int
    # --- BP per-byte-match head fields (the ``head_*`` explicit mode leaves
    #     these at their no-op defaults; only the BP layout-mode head uses them).
    match_q_band: str = ""
    match_k_band: str = ""
    match_weight: float = 0.0
    value_src_lo: str = ""
    value_src_hi: str = ""
    value_o_write_scale: float = 0.0
    # --- BP per-byte LO/HI dump fields (the ``dump_blocks`` explicit mode leaves
    #     these at their no-op defaults; only the BP layout-mode dump uses them).
    dump_emit_lo: str = ""
    dump_emit_hi: str = ""
    dump_write_scale: float = 0.0
    dump_per_byte_marker: str = ""
    dump_per_byte_marker_weight: float = 0.0
    dump_marker_blockers: Tuple[Tuple[str, float], ...] = ()
    dump_gate_conditions: Tuple[Tuple[str, float], ...] = ()
    dump_opent_floor: float = 0.0
    band_flag: Optional[Callable[[], bool]] = None
    value_v_slot_base: int = 10
    dump_threshold: Optional[float] = None
    # K-preference signature: ``(slot, dim_name, weight)`` written on the K
    # side, paired with a CONST-driven Q write at the same slot. Biases the
    # carry TOWARD the prev row.
    k_prefer: Tuple[Tuple[int, str, float], ...] = ()
    # K-reject signature: ``(slot, dim_name, weight)`` — NEGATIVE K writes that
    # HARD-reject same-step / wrong-source rows, paired with a CONST-driven Q
    # write at the same slot (the positive of the magnitude).
    k_reject: Tuple[Tuple[int, str, float], ...] = ()
    const_dim: str = "CONST"
    carry_head_extra_reads: Tuple[str, ...] = ()
    # ``register_residual_band(owner=...)`` diagnostic string. Does NOT affect
    # lowered weights, but a byte-identity MIGRATION keeps it identical to the
    # hand-built call (the registry is collision-checked on
    # (name, size, owner, never_share)). ``None`` => use ``name``.
    band_owner: Optional[str] = None
    # ``never_share`` for the band registration. BP omitted it (defaulted True
    # in its hand-built call); AX/STACK0 pass it explicitly. Carry/dump bands
    # hold cross-step state so this is ~always True.
    band_never_share: bool = True

    # === Generalization fields (the AX byte-1 + STACK0 byte-0 carries) ========
    #
    # position-source split: the carry head + dump + precursor resolve dims via
    # this. ``"layout"`` (BP) => all from ``dim_positions``; ``"mixed"`` =>
    # ``registry_dims`` from the dynamic registry, the rest from the layout.
    position_source: str = "layout"
    registry_dims: Tuple[str, ...] = ()
    # explicit carry-head mode: when ANY of these is non-None the head is built
    # from the explicit :class:`HeadWrite` directives (resolved by
    # ``position_source``) INSTEAD of the BP per-byte-match construction. The
    # head stays UNCONDITIONAL (no head-gate field — the API shape invariant).
    head_q: Optional[Tuple[HeadWrite, ...]] = None
    head_k: Optional[Tuple[HeadWrite, ...]] = None
    head_v: Optional[Tuple[HeadWrite, ...]] = None
    head_o: Optional[Tuple[HeadWrite, ...]] = None
    # explicit carry-head Operation dep-graph dims (the head READS its taps + the
    # cross-step ``X.*.-1`` value sources, WRITES the PREV bands). When the
    # explicit head mode is used these are declared verbatim (the auto-derived
    # BP sets do not apply).
    head_reads: Optional[Set[str]] = None
    head_writes: Optional[Set[str]] = None
    # explicit dump mode: a tuple of :class:`DumpBlock` (or a zero-arg callable
    # returning one — so a dump whose gate conditions depend on a SECONDARY
    # runtime flag, e.g. STACK0's ``C4_STACK0_NEXT_ARITH``, can rebuild the
    # blocks fresh at each compile). When non-None the dump is built from these
    # blocks (the direct-repoint + heterogeneous-gate path) INSTEAD of the BP
    # per-byte LO/HI split. Selected by ``emission_on``: when ``False`` the dump
    # may flip its blocks to an inert target (the AX/STACK0 repoint-vs-inert
    # flag) via ``dump_blocks_off``.
    dump_blocks: Optional[
        "Tuple[DumpBlock, ...] | Callable[[], Tuple[DumpBlock, ...]]"
    ] = None
    dump_blocks_off: Optional[
        "Tuple[DumpBlock, ...] | Callable[[], Tuple[DumpBlock, ...]]"
    ] = None
    # explicit dump Operation dep-graph dims.
    dump_reads_explicit: Optional[Set[str]] = None
    dump_writes_explicit: Optional[Set[str]] = None
    # precursor flag ops (the two-stage KILL precursor + the STACK0 bounded
    # gate-flag precursors). Each declares a standalone bounded-flag FFN the dump
    # ANDs; the generator supplies the bake scaffolding.
    precursors: Tuple[PrecursorFlagSpec, ...] = ()
    # band registration: BP (single-band) lets the generator register
    # ``band_name`` at import. The AX/STACK0 carries register MULTIPLE bands
    # (PREV + DUMP + flag bands) in a LOAD-BEARING order from explicit
    # module-scope ``register_residual_band`` calls; the generator must NOT
    # re-register (it would (a) need the per-band order and (b) collide). Set
    # ``register_band=False`` to keep the explicit calls authoritative.
    register_band: bool = True

    @property
    def explicit_head(self) -> bool:
        """True when the carry head is built from explicit ``head_*`` directives."""
        return any(
            d is not None
            for d in (self.head_q, self.head_k, self.head_v, self.head_o)
        )

    @property
    def explicit_dump(self) -> bool:
        """True when the dump is built from explicit :class:`DumpBlock` directives."""
        return self.dump_blocks is not None or self.dump_blocks_off is not None

    def __post_init__(self) -> None:
        # The BP per-byte LO/HI split needs an even band width; the explicit
        # multi-band carries register odd (7-wide) PREV bands separately and
        # build the dump from explicit blocks, so the even check applies only to
        # the BP layout-mode dump.
        if not self.explicit_dump and self.band_width % 2 != 0:
            raise ValueError(
                f"CrossStepCarrySpec({self.name!r}): band_width must be even "
                f"(LO+HI split), got {self.band_width}"
            )
        if self.carry_byte_count <= 0:
            raise ValueError(
                f"CrossStepCarrySpec({self.name!r}): carry_byte_count must be "
                f"positive, got {self.carry_byte_count}"
            )


# ---------------------------------------------------------------------------
# Bundle — the generated artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossStepCarryBundle:
    """The artifacts :func:`cross_step_carry` generates for one carry.

    The band has ALREADY been registered (import-time side effect) by the time
    this bundle exists. The two builders are pure (no global side effects) so
    the op factory can call them at bake time and IR-factory time identically.

    Attributes:
        spec: the originating :class:`CrossStepCarrySpec`.
        carry_head_spec_builder: ``(dim_positions, head_idx) ->
            DeclarativeAttentionHeadSpec``. Reproduces the hand-built carry
            head EXACTLY (same Q/K/V/O append order — load-bearing because
            ``generate_attention_head`` applies writes in order).
        dump_rules_builder: ``(emission_on: bool) -> tuple[FFNRule, ...]``.
            Returns ``()`` when ``emission_on`` is False (byte-identical
            flag-off — the dump writes nothing).
        carry_head_reads / carry_head_writes: dim-name sets for the carry
            head's Operation (flag-on).
        dump_reads / dump_writes: dim-name sets for the dump FFN's Operation
            (flag-on).
    """

    spec: CrossStepCarrySpec
    carry_head_spec_builder: Callable[
        [Dict[str, int], int], DeclarativeAttentionHeadSpec
    ]
    dump_rules_builder: Callable[[bool], Tuple[FFNRule, ...]]
    carry_head_reads: Set[str]
    carry_head_writes: Set[str]
    dump_reads: Set[str]
    dump_writes: Set[str]
    # Precursor flag ops generated from ``spec.precursors`` (the two-stage KILL +
    # the STACK0 bounded gate flags). Each is a fully-wired ``Operation`` ready
    # to add to the build. ``()`` for the BP carry (no precursors).
    precursor_ops_builder: Callable[[], Tuple[object, ...]] = (
        lambda: ()  # type: ignore[assignment]
    )


# ---------------------------------------------------------------------------
# Explicit-mode lowering helpers
# ---------------------------------------------------------------------------


def _expand_head_writes(
    directives: Sequence[HeadWrite],
    resolve: Callable[[str], int],
    *,
    kind: str,
) -> list:
    """Expand :class:`HeadWrite` directives to ``AP`` / ``AO`` writes.

    ``kind="proj"`` => Q/K/V writes (``AP(slot, dim, weight)``); ``kind="out"``
    => O writes (``AO(out_dim, slot, weight)``). A directive with ``count > 1``
    expands to the per-cell loop (V/O value-copy blocks + the K one-hot-presence
    loops): for ``i in range(count)`` the slot advances by ``slot_stride`` and
    the dim by ``dim_stride``. ``generate_attention_head`` lowers by direct
    indexed assignment, so the EMIT ORDER does not affect the weights — only the
    final (slot, dim, weight) set matters — but the expansion is deterministic.
    """
    out: list = []
    for d in directives:
        for i in range(d.count):
            slot = d.slot + i * d.slot_stride
            dim = _resolve_dim_token(d.dim, resolve) + i * d.dim_stride
            if kind == "proj":
                out.append(AP(slot, dim, d.weight))
            elif kind == "out":
                # O directive: ``dim`` is the OUT band cell, ``slot`` is the V
                # slot it reads. The per-cell loop advances BOTH.
                out.append(AO(dim, slot, d.weight))
            else:  # pragma: no cover - guarded by callers
                raise ValueError(f"_expand_head_writes: bad kind {kind!r}")
    return out


def _expand_dump_blocks(
    blocks: Sequence[DumpBlock],
) -> Tuple[FFNRule, ...]:
    """Expand :class:`DumpBlock` directives to per-cell ``multi_way_and_rule``s.

    Per block, per cell ``j in range(width)``: a balanced AND over ``conditions``
    at ``threshold``, gated multiplicatively on ``{gate_band}+{j}``, writing
    ``({emit_band}+{j}, write_scale)``. The dump dims resolve by NAME at lower
    time (``Primitives.lower_ffn_rules`` + the op's mixed dim_map), so the rules
    are position-source-agnostic here.
    """
    rules: list[FFNRule] = []
    for b in blocks:
        for j in range(b.width):
            rules.append(multi_way_and_rule(
                name=f"{b.name_prefix}_{j}",
                conditions=tuple(b.conditions),
                threshold=b.threshold,
                gate=f"{b.gate_band}+{j}",
                writes=((f"{b.emit_band}+{j}", b.write_scale),),
            ))
    return tuple(rules)


def make_mixed_dim_map_ffn_op(
    precursor: PrecursorFlagSpec,
    *,
    position_source: str,
):
    """Build the standalone bounded-flag precursor ``Operation``.

    Supplies the IDENTICAL ~40-line ``PureFFN`` post_op bake the hand-built
    precursors (AX ``ax_byte1_carry_overflow_flag``, STACK0
    ``stack0_byte0_{carried,sharp,...}_flag``) each duplicated: derive d_model,
    build a ``PureFFN`` sized to the rule count, lower with a MIXED dim_map (the
    NEW ``flag_band`` from the layout, the input taps from the dynamic registry
    when ``position_source="mixed"``), and append to ``block.post_ops``. The
    op's RULE LIST is the spec's ``rules_builder()`` (the domain-specific flag
    logic stays in the op module).
    """
    # Imported lazily so the ISA-DSL module stays import-light (the
    # ``layer_compiler.Operation`` import would otherwise pull the compiler).
    from .layer_compiler import Operation
    from .ir import CompilerIR
    from .primitives import Primitives

    rules = precursor.rules_builder()
    reg_set = set(precursor.registry_dims)

    def bake(block, dim_positions, S):
        from ..base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        ffn = PureFFN(d_model, len(rules))
        dim_map = _mixed_dim_map(
            rules, dim_positions,
            registry_dims=reg_set if position_source == "mixed" else set(),
            new_bands={precursor.flag_band},
        )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    op_kwargs = dict(
        name=precursor.op_name,
        reads=set(precursor.reads),
        writes=set(precursor.writes),
        kind="block",
        target_op_name=precursor.target_op_name,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
    )
    if precursor.requires is not None:
        op_kwargs["requires"] = precursor.requires
    if precursor.spec_section is not None:
        op_kwargs["spec_section"] = precursor.spec_section
    return Operation(**op_kwargs)


def _mixed_dim_map(
    rules,
    dim_positions: Dict[str, int],
    *,
    registry_dims: Set[str],
    new_bands: Set[str],
) -> Dict[str, int]:
    """Resolve every ``base+off`` dim a rule list touches under the split.

    ``new_bands`` resolve from the LAYOUT (``dim_positions``); everything in
    ``registry_dims`` from the dynamic registry; the remainder from the layout
    (the ``position_source="layout"`` fallback — the same map the BP dump uses).
    Mirrors the hand-built precursor / dump bake bodies EXACTLY.
    """
    from .primitives import Primitives

    _reg_holder: Dict[str, object] = {}

    def _registry():
        if "reg" not in _reg_holder:
            from ..dim_registry_dynamic import build_default_registry_dynamic
            _reg_holder["reg"] = build_default_registry_dynamic()
        return _reg_holder["reg"]

    dim_map: Dict[str, int] = {}
    for nm in Primitives.ffn_rule_dim_names(rules):
        base = nm.split("+", 1)[0]
        off = int(nm.split("+", 1)[1]) if "+" in nm else 0
        if base in new_bands:
            dim_map[nm] = int(dim_positions[base]) + off
        elif base in registry_dims:
            dim_map[nm] = int(_registry().slots[base].start) + off
        else:
            dim_map[nm] = int(dim_positions[base]) + off
    return dim_map


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def cross_step_carry(spec: CrossStepCarrySpec) -> CrossStepCarryBundle:
    """Generate the 4-part cross-step carry scaffolding for ``spec``.

    Side effect (import time): registers ``spec.band_name`` via
    :func:`register_residual_band` with ``never_share=True`` and
    ``flag=spec.band_flag``. Call this at MODULE-IMPORT scope of the owning
    ``lN_ops.py`` (the SAME position the hand-built ``register_residual_band``
    call occupied) — registration order is load-bearing for the tail
    ``dim_positions``.

    Returns a :class:`CrossStepCarryBundle` whose builders reproduce the
    hand-built carry head + dump FFN byte-identically.
    """
    # (1) Register the dedicated ``_PREV`` band (import-time side effect). The
    #     registry is idempotent on (name, size, owner, never_share), so a
    #     module reload under test re-registers cleanly. The multi-band AX/STACK0
    #     carries set ``register_band=False`` and keep their LOAD-BEARING-ordered
    #     module-scope ``register_residual_band`` calls authoritative.
    if spec.register_band:
        register_residual_band(
            spec.band_name,
            spec.band_width,
            owner=spec.band_owner if spec.band_owner is not None else spec.name,
            flag=spec.band_flag,
            never_share=spec.band_never_share,
        )

    half = spec.band_width // 2

    # (2) Carry head spec builder — UNCONDITIONAL (no head-gate field; the
    #     API-shape invariant). Two modes:
    #       * explicit-head (AX/STACK0): lower the explicit ``head_*`` directives
    #         resolved by ``position_source``;
    #       * BP per-byte-match (layout): the per-byte positional MATCH +
    #         k_prefer/k_reject + LO/HI value-copy construction.
    if spec.explicit_head:
        def carry_head_spec_builder(
            dim_positions: Dict[str, int], head_idx: int
        ) -> DeclarativeAttentionHeadSpec:
            resolve = _make_position_resolver(
                dim_positions,
                position_source=spec.position_source,
                registry_dims=spec.registry_dims,
            )
            q = _expand_head_writes(spec.head_q or (), resolve, kind="proj")
            k = _expand_head_writes(spec.head_k or (), resolve, kind="proj")
            v = _expand_head_writes(spec.head_v or (), resolve, kind="proj")
            o = _expand_head_writes(spec.head_o or (), resolve, kind="out")
            return DeclarativeAttentionHeadSpec(
                head_idx=head_idx,
                q=tuple(q),
                k=tuple(k),
                v=tuple(v),
                o=tuple(o),
                alibi_slope=spec.carry_head_alibi_slope,
            )
    else:
        def carry_head_spec_builder(
            dim_positions: Dict[str, int], head_idx: int
        ) -> DeclarativeAttentionHeadSpec:
            def _P(name: str) -> int:
                return int(dim_positions[name])

            src_lo = _P(spec.value_src_lo)
            src_hi = _P(spec.value_src_hi)
            band = int(dim_positions[spec.band_name])
            const = _P(spec.const_dim)

            q: list = []
            k: list = []

            # Slots 0..(carry_byte_count-1): per-byte positional match. Q reads
            # the consuming row's ``match_q_band{k}``; K reads the prev row's
            # ``match_k_band{k}``. Each on its OWN slot.
            for kk in range(spec.carry_byte_count):
                qb = _P(f"{spec.match_q_band}{kk}")
                kb = _P(f"{spec.match_k_band}{kk}")
                q.append(AP(kk, qb, spec.match_weight))
                k.append(AP(kk, kb, spec.match_weight))

            # K-preference slots: CONST-driven Q + positive K signature (bias the
            # carry TOWARD the prev row).
            for (slot, dim_name, weight) in spec.k_prefer:
                q.append(AP(slot, const, weight))
                k.append(AP(slot, _P(dim_name), weight))

            # K-reject slots: NEGATIVE K signature (hard-reject same-step / wrong
            # source rows) + CONST-driven Q at the positive magnitude. Multiple
            # entries may share a slot (e.g. the STACK0_BYTE{0..3} family); the
            # Q-side CONST write is emitted ONCE PER (slot) — in the hand-built
            # BP code the STACK0 group emits the four K writes first, then ONE Q
            # write. To reproduce that exact append order, emit all K writes for
            # a contiguous run of entries sharing a slot, then the single Q
            # write.
            ki = 0
            while ki < len(spec.k_reject):
                slot = spec.k_reject[ki][0]
                # Gather the contiguous run of entries on this slot.
                run = []
                while ki < len(spec.k_reject) and spec.k_reject[ki][0] == slot:
                    run.append(spec.k_reject[ki])
                    ki += 1
                # K writes (negative), in order.
                for (_slot, dim_name, weight) in run:
                    k.append(AP(slot, _P(dim_name), -weight))
                # Single Q write at the positive magnitude of the run's |weight|.
                q.append(AP(slot, const, run[0][2]))

            # V/O: copy ``value_src_lo[0..half-1]`` -> band[0..half-1] and
            # ``value_src_hi[0..half-1]`` -> band[half..band_width-1]. V slots
            # from ``value_v_slot_base`` (far from the Q/K gate slots).
            v: list = []
            o: list = []
            vb = spec.value_v_slot_base
            for j in range(half):
                v.append(AP(vb + j, src_lo + j, 1.0))
                o.append(AO(band + j, vb + j, spec.value_o_write_scale))
            for j in range(half):
                v.append(AP(vb + half + j, src_hi + j, 1.0))
                o.append(AO(band + half + j, vb + half + j,
                            spec.value_o_write_scale))

            return DeclarativeAttentionHeadSpec(
                head_idx=head_idx,
                q=tuple(q),
                k=tuple(k),
                v=tuple(v),
                o=tuple(o),
                alibi_slope=spec.carry_head_alibi_slope,
            )

    # (3) Dump rules builder — the GATE lives here. Two modes:
    #       * explicit-dump (AX/STACK0): per-block per-cell gate-copy from
    #         ``dump_blocks`` (emission_on) / ``dump_blocks_off`` (off — the
    #         repoint-vs-inert flip; the band stays present so the off build is
    #         NOT zero-rule but writes the inert DUMP band, byte-identical to the
    #         hand-built flag-off);
    #       * BP per-byte LO/HI (layout): per byte k, per band cell j the
    #         ``emit_band[j] = band[j]`` gate-copy; off => zero rules (the band is
    #         OMITTED from the layout, so the op is inert).
    if spec.explicit_dump:
        def dump_rules_builder(emission_on: bool) -> Tuple[FFNRule, ...]:
            blocks = spec.dump_blocks if emission_on else spec.dump_blocks_off
            if callable(blocks):
                blocks = blocks()
            if not blocks:
                return ()
            return _expand_dump_blocks(blocks)
    else:
        if spec.dump_threshold is not None:
            threshold = spec.dump_threshold
        else:
            threshold = spec.dump_opent_floor + spec.dump_per_byte_marker_weight

        def dump_rules_builder(emission_on: bool) -> Tuple[FFNRule, ...]:
            if not emission_on:
                return ()
            rules: list[FFNRule] = []
            for kk in range(spec.carry_byte_count):
                conditions = tuple(spec.dump_gate_conditions) + (
                    (f"{spec.dump_per_byte_marker}{kk}",
                     spec.dump_per_byte_marker_weight),
                ) + tuple(spec.dump_marker_blockers)
                for nib in range(half):
                    rules.append(multi_way_and_rule(
                        name=f"{spec.name}_val{kk}_lo_{nib}",
                        conditions=conditions,
                        threshold=threshold,
                        gate=f"{spec.band_name}+{nib}",
                        writes=((f"{spec.dump_emit_lo}+{nib}",
                                 spec.dump_write_scale),),
                    ))
                for nib in range(half):
                    rules.append(multi_way_and_rule(
                        name=f"{spec.name}_val{kk}_hi_{nib}",
                        conditions=conditions,
                        threshold=threshold,
                        gate=f"{spec.band_name}+{half + nib}",
                        writes=((f"{spec.dump_emit_hi}+{nib}",
                                 spec.dump_write_scale),),
                    ))
            return tuple(rules)

    # (4) Reads/writes dim sets (flag-on). Explicit mode declares them verbatim
    #     on the spec; BP mode derives them structurally.
    if spec.explicit_head:
        carry_head_reads = set(spec.head_reads or ())
        carry_head_writes = set(spec.head_writes or ())
    else:
        carry_head_reads = set()
        for kk in range(spec.carry_byte_count):
            carry_head_reads.add(f"{spec.match_q_band}{kk}")
            carry_head_reads.add(f"{spec.match_k_band}{kk}")
        for (_slot, dim_name, _w) in spec.k_prefer:
            carry_head_reads.add(dim_name)
        for (_slot, dim_name, _w) in spec.k_reject:
            carry_head_reads.add(dim_name)
        carry_head_reads.add(spec.const_dim)
        carry_head_reads.add(spec.value_src_lo)
        carry_head_reads.add(spec.value_src_hi)
        carry_head_reads.add(f"{spec.value_src_lo}.*.-1")
        carry_head_reads.add(f"{spec.value_src_hi}.*.-1")
        carry_head_reads.update(spec.carry_head_extra_reads)
        carry_head_writes = {spec.band_name}

    if spec.explicit_dump:
        dump_reads = set(spec.dump_reads_explicit or ())
        dump_writes = set(spec.dump_writes_explicit or ())
    else:
        dump_reads = set()
        for (dim_name, _w) in spec.dump_gate_conditions:
            dump_reads.add(dim_name)
        for kk in range(spec.carry_byte_count):
            dump_reads.add(f"{spec.dump_per_byte_marker}{kk}")
        for (dim_name, _w) in spec.dump_marker_blockers:
            dump_reads.add(dim_name)
        dump_reads.add(spec.band_name)
        dump_writes = {spec.dump_emit_lo, spec.dump_emit_hi, spec.band_name}

    # (5) Precursor flag ops builder — the two-stage KILL + STACK0 bounded gate
    #     flags. Each is a standalone bounded-flag FFN with a mixed dim_map; the
    #     generator owns the bake scaffolding (``make_mixed_dim_map_ffn_op``).
    def precursor_ops_builder() -> Tuple[object, ...]:
        return tuple(
            make_mixed_dim_map_ffn_op(p, position_source=spec.position_source)
            for p in spec.precursors
        )

    return CrossStepCarryBundle(
        spec=spec,
        carry_head_spec_builder=carry_head_spec_builder,
        dump_rules_builder=dump_rules_builder,
        carry_head_reads=carry_head_reads,
        carry_head_writes=carry_head_writes,
        dump_reads=dump_reads,
        dump_writes=dump_writes,
        precursor_ops_builder=precursor_ops_builder,
    )


# ---------------------------------------------------------------------------
# Full-width byte emission — lift the 16-cell one-hot cap
# ---------------------------------------------------------------------------
#
# The byte-1 (high-byte) emission path is a MARKER-DISTANCE positional one-hot
# SPREAD across the L0 ``H1``/``H2``/``H3`` bands (7 cells each, 16 distinct
# output cells), and the LM head reads those cells via
# ``head.weight[v, H<k>+off]`` columns that ALIAS mod 16
# (``head.weight[v] == head.weight[v+16] == head.weight[v+32]`` — verified by
# ``tools/probe_hband_byte1_map.py``, spec_k=0). So the emitter has only 16
# distinct output tokens and every byte ``>= 16`` collapses to ``byte mod 16``
# (the ``edge_literal`` cluster: ``got == exp & 0x0F``; see
# ``docs/EDGE_LITERAL_BYTE1_HIGH_NIBBLE_WALL_2026_06_15.md``).
#
# :func:`full_width_byte_emission` is the GENERATOR that lifts the cap. It
# supplies the two MECHANICAL halves the alias-break needs — independent of
# WHERE the value source comes from:
#
#   1. a dedicated WIDE value band (``2**bits`` cells, a FULL value-faithful
#      one-hot — NOT the 16-cell positional spread). One cell per byte value;
#      no aliasing.
#   2. the LM-head columns for the FULL value range (``head.weight[v, band+v]
#      = scale``) — one DISTINCT column per token, so values 16..255 emit their
#      own token instead of ``v mod 16``.
#
# Plus a gated DUMP FFN that fills ``band+v`` from a value-source one-hot at the
# consuming row (the same gate-in-FFN discriminator the cross-step carry uses).
# The band + the columns + the dump are gated by ``emission_flag``: flag-OFF
# omits the band (smaller d_model) AND emits zero columns / zero rules, so the
# model is BYTE-IDENTICAL to the pre-feature build.
#
# Scope contract (honest): the columns + band + dump are mechanical and
# byte-identity-gateable on CPU. Whether ``edge_literal`` (or any specific byte
# >= 16) actually CLEARS additionally requires a value-faithful one-hot SOURCE
# for the byte at the consuming row to feed ``dump_value_source``. On the IMM
# byte-1 predictor row that source does NOT exist (probed exhaustively: the
# high nibble is encoded only as a coarse ``H3+4`` hi==0-vs-hi>0 flag, and
# ``OUTPUT_HI`` / ``AX_FULL_HI`` / ``AX_CARRY_HI`` are empty at every block) —
# materializing it is the DEFERRED IMM-decode band + relay build (the plan's
# deferred list; the wall doc's two coordinated halves). This generator is the
# emission half that build plugs into.


@dataclass(frozen=True)
class FullWidthByteEmissionSpec:
    """Declarative description of a full-width (un-aliased) byte emission band.

    The hand-built byte-1 dump (``model_ops._ax_byte1_dump_band_for_value`` +
    ``_ax_byte1_dump_head_bake_rules``) MIRRORS the LM head's 16-cell H-band
    positional layout, so it inherits the mod-16 alias and caps byte values at
    15. This spec declares a flat ``2**bits`` value band that breaks the alias:
    one cell per value, one LM-head column per value.

    Every field is a VARYING parameter; the IDENTICAL 3-part structure (wide
    band + full-range LM-head columns + gated dump FFN) is supplied by
    :func:`full_width_byte_emission`. The band/columns/dump are ALL gated by
    :attr:`emission_flag` — flag-off omits the band and emits zero
    columns/rules (byte-identical to the pre-feature build).

    Attributes:
        name: feature name (band owner + rule-name prefix).
        band_name: dedicated WIDE value band name (``2**bits`` cells).
            Registered at import time by :func:`full_width_byte_emission`.
        bits: byte width in bits (8 => 256 cells, values 0..255).
        head_scale: LM-head column weight (mirrors the H-band ``+5.0``).
        lo_value: first value to emit columns for. The hand-built H-band
            columns already cover ``0..15`` (the alias range); the full-width
            generator covers ``lo_value..(2**bits - 1)``. Default 16 (the
            payoff: ADD the un-aliased high values WITHOUT re-emitting the
            already-correct 0..15 columns — keeps the off-build byte-identical
            and the on-build additive).
        emission_flag: zero-arg predicate gating the WHOLE feature (band +
            columns + dump). ``None`` => always on. When supplied, flag-off
            omits the band (smaller d_model) and bakes no columns / rules.
        dump_value_source: the value-source band the dump reads to fill the
            wide band — a ``2**bits``-wide one-hot of the byte value at the
            consuming row (e.g. a relay-populated band). The dump copies
            ``dump_value_source+v -> band+v`` gated on the row marker. ``None``
            => no dump rules are generated (the band is filled by a partner op,
            or this is a columns-only build).
        dump_gate_conditions: the carry-vs-fresh / row-selector GATE — a tuple
            of ``(dim_name, weight)`` AND conditions shared across all dump
            cells. THE ONLY dump gating surface. ``()`` => the dump fires
            whenever the source cell is active (gate = the source one-hot
            alone).
        dump_threshold: explicit AND threshold for the dump cells. ``None`` =>
            the sum of the ``dump_gate_conditions`` weights (so every declared
            condition must be present).
        dump_write_scale: dump band-write magnitude. Default mirrors the
            H-band carry's large write so it dominates the tail. Default 5.0.
        band_owner: ``register_residual_band(owner=...)`` diagnostic string.
            ``None`` => use ``name``.
    """

    name: str
    band_name: str
    bits: int = 8
    head_scale: float = 5.0
    lo_value: int = 16
    emission_flag: Optional[Callable[[], bool]] = None
    dump_value_source: Optional[str] = None
    dump_gate_conditions: Tuple[Tuple[str, float], ...] = ()
    dump_threshold: Optional[float] = None
    dump_write_scale: float = 5.0
    band_owner: Optional[str] = None
    # --- Nibble-pair source (the value carry re-point) ---------------------
    #
    # A flat ``dump_value_source`` one-hot does not always exist. The C4 byte
    # value frequently lives as a NIBBLE PAIR — two 16-wide one-hots (a LOW
    # nibble band + a HIGH nibble band), each optionally offset by a constant
    # (e.g. the ALU_LO/ALU_HI byte-dump bands encode the byte's nibbles at
    # cell ``nibble + 2``). When ``dump_nibble_lo`` AND ``dump_nibble_hi`` are
    # set the dump fills ``band+v`` by AND-ing the two nibble cells of value v
    # — ``{dump_nibble_lo}+{(v & (2**(bits//2) - 1)) + dump_nibble_lo_offset}``
    # and ``{dump_nibble_hi}+{(v >> (bits//2)) + dump_nibble_hi_offset}`` —
    # together with the row-selector ``dump_gate_conditions``. This is the
    # value-source RE-POINT half of the edge_literal fix: the wide band is fed
    # from the nibble pair that DOES carry the high byte at the consuming row,
    # instead of a (nonexistent) flat one-hot. ``bits`` must be even (a
    # symmetric lo/hi nibble split). Takes precedence over
    # ``dump_value_source`` when both are set.
    dump_nibble_lo: Optional[str] = None
    dump_nibble_hi: Optional[str] = None
    dump_nibble_lo_offset: int = 0
    dump_nibble_hi_offset: int = 0
    dump_nibble_lo_weight: float = 1.0
    dump_nibble_hi_weight: float = 1.0
    # Highest nibble value the source band faithfully distinguishes. An offset
    # source band has only ``nibble_width - offset`` usable cells, so nibble
    # values above ``nibble_width - offset - 1`` either overflow the band (read
    # a NEIGHBOURING band's cell -> spurious cross-talk fills) or alias a lower
    # nibble. The dump SKIPS any value whose lo OR hi nibble exceeds this cap
    # (no faithful source -> no fill rule -> the H-band mod-16 path keeps owning
    # it). ``None`` => derive ``nibble_width - max(lo_off, hi_off) - 1`` so the
    # offset never overflows the band. (For the ALU byte-dump the firmware
    # encoding ALSO collapses nibbles 14,15 onto cell 2, so the derived cap of
    # 13 is exactly the faithful range.)
    dump_nibble_max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.bits <= 0 or self.bits > 16:
            raise ValueError(
                f"FullWidthByteEmissionSpec({self.name!r}): bits must be in "
                f"1..16, got {self.bits}"
            )
        if (self.dump_nibble_lo is None) != (self.dump_nibble_hi is None):
            raise ValueError(
                f"FullWidthByteEmissionSpec({self.name!r}): dump_nibble_lo "
                "and dump_nibble_hi must BOTH be set or BOTH unset"
            )
        if self.dump_nibble_lo is not None and self.bits % 2 != 0:
            raise ValueError(
                f"FullWidthByteEmissionSpec({self.name!r}): nibble-pair source "
                f"requires an even bit width (got bits={self.bits})"
            )
        if self.lo_value < 0:
            raise ValueError(
                f"FullWidthByteEmissionSpec({self.name!r}): lo_value must be "
                f">= 0, got {self.lo_value}"
            )
        if self.lo_value >= (1 << self.bits):
            raise ValueError(
                f"FullWidthByteEmissionSpec({self.name!r}): lo_value "
                f"{self.lo_value} >= 2**bits {1 << self.bits} — nothing to emit"
            )

    @property
    def band_width(self) -> int:
        return 1 << self.bits

    @property
    def nibble_width(self) -> int:
        """Cell count of each nibble one-hot (``2**(bits/2)``)."""
        return 1 << (self.bits // 2)

    @property
    def has_nibble_source(self) -> bool:
        """True when the dump fills from a ``(lo, hi)`` nibble pair."""
        return self.dump_nibble_lo is not None

    @property
    def effective_nibble_max(self) -> int:
        """Highest nibble the offset source band reaches without overflow.

        Explicit ``dump_nibble_max`` wins; else derive ``nibble_width -
        max(offset) - 1`` so ``nibble + offset`` stays inside the band.
        """
        if self.dump_nibble_max is not None:
            return int(self.dump_nibble_max)
        max_off = max(self.dump_nibble_lo_offset, self.dump_nibble_hi_offset)
        return self.nibble_width - max_off - 1


@dataclass(frozen=True)
class FullWidthByteEmissionBundle:
    """The artifacts :func:`full_width_byte_emission` generates.

    The wide band has ALREADY been registered (import-time side effect) by the
    time this bundle exists. The builders are pure (no global side effects).

    Attributes:
        spec: the originating :class:`FullWidthByteEmissionSpec`.
        head_columns_builder: ``(emission_on: bool, vocab_size: int) ->
            tuple[TokenEmbeddingRule, ...]``. The LM-head columns for
            ``lo_value..min(2**bits - 1, vocab_size - 1)``. Returns ``()`` when
            ``emission_on`` is False (byte-identical flag-off).
        dump_rules_builder: ``(emission_on: bool, dim_positions) ->
            tuple[FFNRule, ...]``. The gated FFN that fills the wide band from
            ``dump_value_source``. Returns ``()`` when off OR when
            ``dump_value_source`` is None.
        head_reads / head_writes: dim-name sets for the LM-head bake (the wide
            band cells are the head's read columns).
        dump_reads / dump_writes: dim-name sets for the dump FFN's Operation.
    """

    spec: FullWidthByteEmissionSpec
    head_columns_builder: Callable[[bool, int], Tuple[TokenEmbeddingRule, ...]]
    dump_rules_builder: Callable[[bool, Dict[str, int]], Tuple[FFNRule, ...]]
    head_reads: Set[str]
    head_writes: Set[str]
    dump_reads: Set[str]
    dump_writes: Set[str]


def full_width_byte_emission(
    spec: FullWidthByteEmissionSpec,
) -> FullWidthByteEmissionBundle:
    """Generate the un-aliased full-width byte emission scaffolding for ``spec``.

    Side effect (import time): registers ``spec.band_name`` (``2**spec.bits``
    cells) via :func:`register_residual_band` with ``never_share=True`` and
    ``flag=spec.emission_flag``. Call at MODULE-IMPORT scope of the owning
    ``lN_ops``/``model_ops`` module.

    The bundle's two builders reproduce the un-aliased emission columns + the
    gated band-fill FFN. Flag-off => zero columns / zero rules / no band
    (byte-identical to the pre-feature build).
    """
    # (1) Register the dedicated WIDE value band (import-time side effect).
    register_residual_band(
        spec.band_name,
        spec.band_width,
        owner=spec.band_owner if spec.band_owner is not None else spec.name,
        flag=spec.emission_flag,
        never_share=True,
    )

    hi_value = spec.band_width - 1

    # (2) LM-head columns — one DISTINCT column per value (no mod-16 alias).
    def head_columns_builder(
        emission_on: bool, vocab_size: int
    ) -> Tuple[TokenEmbeddingRule, ...]:
        if not emission_on:
            return ()
        rules: list[TokenEmbeddingRule] = []
        top = min(hi_value, int(vocab_size) - 1)
        for v in range(spec.lo_value, top + 1):
            rules.append(TokenEmbeddingRule.head_weight_write(
                token_ids=[v],
                writes=((f"{spec.band_name}+{v}", spec.head_scale),),
                name=f"{spec.name}_head_token_{v}",
            ))
        return tuple(rules)

    # (3) Dump FFN — fill ``band+v`` from ``dump_value_source+v`` gated on the
    #     row conditions. Per value cell v in [lo_value, hi_value]. The gate is
    #     the value-source one-hot cell ``dump_value_source+v`` (so cell v fills
    #     iff the source's value == v); the conditions are the row selector.
    #
    #     Threshold derivation mirrors ``multi_way_and_rule``'s own midpoint
    #     ``(total + (total - max_w)) / 2`` (the "all conditions on" sum clears
    #     it; "any one missing" sinks below) so the AND fires only when EVERY
    #     declared condition is present. When there are no conditions the gate
    #     alone selects (threshold 0 => the source one-hot drives the write).
    if spec.dump_threshold is not None:
        threshold: Optional[float] = spec.dump_threshold
    elif spec.dump_gate_conditions:
        _weights = [w for (_d, w) in spec.dump_gate_conditions]
        _total = sum(_weights)
        threshold = (_total + (_total - max(_weights))) / 2.0
    else:
        threshold = 0.0

    # Nibble-pair dump threshold: the AND is over JUST the two nibble cells
    # (lo, hi); the row selector is a SEPARATE MULTIPLICATIVE gate (so the row
    # marker's large magnitude does NOT swamp the additive nibble AND and make
    # every value fire). The threshold sits between "one nibble present" and
    # "both present": with nibble weight ``w`` and an active cell value ``a``,
    # both-on ``= 2*w*a`` must clear and one-on ``= w*a (+ w*off)`` must sink.
    # ``dump_threshold`` overrides; else use ``1.5 * w`` (the balanced-AND
    # midpoint for two equal-weight conditions, which for a one-hot ``a~=1``
    # clears on both-on=2w and sinks on one-on=w). For continuous sources the
    # caller passes an explicit ``dump_threshold`` matched to the cell
    # magnitude.
    if spec.has_nibble_source:
        if spec.dump_threshold is not None:
            nib_threshold: Optional[float] = spec.dump_threshold
        else:
            _wl, _wh = spec.dump_nibble_lo_weight, spec.dump_nibble_hi_weight
            _t = _wl + _wh
            nib_threshold = (_t + (_t - max(_wl, _wh))) / 2.0
        nib_w = spec.nibble_width

    def dump_rules_builder(
        emission_on: bool, dim_positions: Dict[str, int]
    ) -> Tuple[FFNRule, ...]:
        del dim_positions  # all dims resolve by name at lower time
        if not emission_on:
            return ()
        rules: list[FFNRule] = []
        if spec.has_nibble_source:
            # Value v = hi*nib_w + lo. Fill band+v iff the LOW nibble cell
            # (value lo) AND the HIGH nibble cell (value hi) AND the row gate
            # are all active. Each nibble band cell carries its nibble at
            # ``base + nibble + offset``. SKIP any value whose lo OR hi nibble
            # exceeds the faithful range (``effective_nibble_max``): above it
            # the offset cell overflows the band (reading a neighbouring band's
            # cell => spurious cross-talk fills) — the H-band mod-16 path keeps
            # owning those values.
            nib_max = spec.effective_nibble_max
            # The AND is over the two nibble cells PLUS the row-selector
            # conditions, ALL ADDITIVE. ``dump_threshold`` is sized by the
            # caller so the AND requires BOTH nibble cells (each ~``cell_on``):
            # the threshold sits ABOVE "one nibble present + the full row gate"
            # so the row marker's magnitude alone (or one nibble) can NOT clear
            # it — only both nibbles + the row gate together. (A multiplicative
            # gate was tried but silu is not sharp enough to floor the off-row
            # contribution; an additive AND with a high threshold is sharper.)
            row_conditions = tuple(spec.dump_gate_conditions)
            for v in range(spec.lo_value, hi_value + 1):
                lo = v % nib_w
                hi = v // nib_w
                if lo > nib_max or hi > nib_max:
                    continue
                conditions = (
                    (f"{spec.dump_nibble_lo}+{lo + spec.dump_nibble_lo_offset}",
                     spec.dump_nibble_lo_weight),
                    (f"{spec.dump_nibble_hi}+{hi + spec.dump_nibble_hi_offset}",
                     spec.dump_nibble_hi_weight),
                ) + row_conditions
                rules.append(multi_way_and_rule(
                    name=f"{spec.name}_fill_{v}",
                    conditions=conditions,
                    threshold=nib_threshold,
                    writes=((f"{spec.band_name}+{v}", spec.dump_write_scale),),
                ))
            return tuple(rules)
        if spec.dump_value_source is None:
            return ()
        conditions = tuple(spec.dump_gate_conditions)
        for v in range(spec.lo_value, hi_value + 1):
            if conditions:
                rules.append(multi_way_and_rule(
                    name=f"{spec.name}_fill_{v}",
                    conditions=conditions,
                    threshold=threshold,
                    gate=f"{spec.dump_value_source}+{v}",
                    writes=((f"{spec.band_name}+{v}", spec.dump_write_scale),),
                ))
            else:
                # No row conditions: the source one-hot cell alone gates the
                # write. Use the source cell as the single "condition" so the
                # AND degenerates to "source cell active".
                rules.append(multi_way_and_rule(
                    name=f"{spec.name}_fill_{v}",
                    conditions=((f"{spec.dump_value_source}+{v}", 1.0),),
                    writes=((f"{spec.band_name}+{v}", spec.dump_write_scale),),
                ))
        return tuple(rules)

    head_reads: Set[str] = {spec.band_name}
    head_writes: Set[str] = set()  # writes the LM head, not a residual dim

    dump_reads: Set[str] = set()
    for (dim_name, _w) in spec.dump_gate_conditions:
        dump_reads.add(dim_name)
    if spec.has_nibble_source:
        dump_reads.add(spec.dump_nibble_lo)
        dump_reads.add(spec.dump_nibble_hi)
    elif spec.dump_value_source is not None:
        dump_reads.add(spec.dump_value_source)
    dump_writes: Set[str] = {spec.band_name}

    return FullWidthByteEmissionBundle(
        spec=spec,
        head_columns_builder=head_columns_builder,
        dump_rules_builder=dump_rules_builder,
        head_reads=head_reads,
        head_writes=head_writes,
        dump_reads=dump_reads,
        dump_writes=dump_writes,
    )


# ---------------------------------------------------------------------------
# Consumer-opcode lookahead gate — gate a band on the CONSUMER (next) opcode
# ---------------------------------------------------------------------------
#
# The #221 framing-drift fix (commits ``acdccc68..fb984bc1``) is a SIX-OP
# control-flow mechanism whose only purpose is to GATE the L25-tail STACK0
# dump on the CONSUMER opcode (the NEXT instruction). The dump over-fires on
# arithmetic-INTERMEDIATE operand frames (expr ``a*b/c``) but is load-bearing
# on comparison-result frames (if/bool). The two frames are batched-identical
# in every CURRENT residual band; the ONLY separator is the next instruction.
# That instruction is causally UNAVAILABLE at the operand frame within a pass
# (attention is causal, the consumer is a future position), but the single-slot
# C4 ISA puts it at a FIXED ``PC + INSTR_WIDTH`` in program memory — fetchable
# exactly like the L5 opcode fetch, offset by the instruction stride.
#
# The six ops share an IDENTICAL structure parameterized only by (the consumer
# opcode CLASS to detect, the PC offset to the consumer, whether to add the
# CAUSAL prior-class latch, and which band/dump the AND-gate drives):
#
#   1. SIX flag-gated ``_PREV``-style bands (the PC+offset address, the fetched
#      opcode nibbles, the consumer-class flag, the prior-class latch, and the
#      combined dump-block flag), registered at MODULE-IMPORT scope.
#   2. a PC+offset NIBBLE-ROTATION chain FFN (build the consumer's byte-address
#      from EMBED at the relayed-PC marker row) -> the address band.
#   3. an OPCODE-FETCH head (content-match the PC+offset address vs ADDR_KEY,
#      copy that CODE slot's CLEAN_EMBED opcode nibbles) -> the opcode band.
#   4. a CONSUMER-CLASS flag FFN (per-opcode two-nibble AND, OR'd) -> the
#      bounded consumer-class flag (1 dim).
#   5. a within-step RELAY head (broadcast the flag from the AX-marker row,
#      where the lookahead lives, to the marker row the dump fires on).
#   6. an OPTIONAL CAUSAL prior-class LATCH head (Q@target-marker, K matches the
#      prior-class opcode bands on PRIOR AX rows, V=those opcode dims) -> the
#      latch (1 dim) — the key NARROWING lever (single-op vs multi-op frame).
#   7. an AND-GATE flag FFN -> the combined dump-block flag (1 dim) the gated
#      band/dump reads as its hard blocker. When the latch is OMITTED the gate
#      degenerates to the consumer-class flag alone.
#
# :func:`consumer_lookahead_gate` takes the VARYING params via
# :class:`ConsumerLookaheadGateSpec` and supplies the IDENTICAL structure as a
# bundle of pure builders the op factories install (mirroring
# :func:`cross_step_carry`). NO compiler change — every builder lowers through
# the existing ``Operation(kind="block")`` path.
#
# The first migration re-expresses the LANDED #221 ops via
# ``consumer_lookahead_gate(ARITH_CONSUMER_SPEC)``. It is byte-identity-gated
# against the HEAD golden ``state_dict`` hash (flag-on AND flag-off) — the
# proof the generator reproduces the hand-built weights bit-for-bit.
#
# See ``docs/EXPR_MULDIV_ROOT_IS_STACK0_PERSISTENCE_2026_06_15.md``.


@dataclass(frozen=True)
class ConsumerLookaheadGateSpec:
    """Declarative description of a consumer-opcode lookahead gate.

    "Gate a band on the consumer (next) opcode": detect that the NEXT
    instruction's opcode is in a declared CLASS, optionally AND it with a
    causal prior-class latch, and drive a bounded dump-block flag the gated
    band/dump reads as a hard blocker.

    Every field is a VARYING parameter; the IDENTICAL 6-op structure is
    supplied by :func:`consumer_lookahead_gate`. The generator returns pure
    builders the op factories install — the band-flag, head indices, target
    anchors, and ``Operation`` wiring stay in the factory (so a migration keeps
    those byte-identical too).

    Attributes:
        name: feature name (band owner prefix + rule-name prefix).
        feature_flag: zero-arg predicate gating the WHOLE feature (every band +
            op). ``None`` => always present. Flag-off omits the bands (smaller
            d_model) and the ops bake as no-ops — byte-identical to the
            pre-feature build.

        pc_offset: byte offset from the current PC to the CONSUMER instruction
            (the C4 single-slot stride => ``INSTR_WIDTH`` == 8).
        pc_band_lo / pc_band_hi: the PC+offset address band names (16 cells
            each — a nibble one-hot pair). Built by the chain FFN.
        pc_chain_source_lo / pc_chain_source_hi: the EMBED bands the chain reads
            the current PC from (at the relayed-PC marker row).
        pc_chain_gate_marker: the marker dim the chain FFN gates on (the
            relayed-PC AX row). Also the chain ``scope``.
        pc_chain_magnitude: per-nibble write magnitude (chain ``magnitude=``).

        opcode_band_lo / opcode_band_hi: the fetched-opcode band names (16 each
            — the consumer's opcode-byte nibble one-hots). Written by the fetch
            head.
        fetch_addr_key / fetch_clean_embed_lo / fetch_clean_embed_hi: the
            content-match KEY band (per-CODE-position immutable address) and the
            CLEAN_EMBED value bands the fetch head copies.
        fetch_marker / fetch_const / fetch_has_se: the fetch head's gate dims
            (the relayed-PC marker, the CONST top-nibble match, the HAS_SE
            gate). Mirror the production L5 fetch head.
        fetch_addr_weight / fetch_marker_weight / fetch_gate_weight: the fetch
            head's per-nibble address weight, marker-slot weight, and the
            500-style hard gate weight.
        fetch_top_slot: the head-local slot index for the top-nibble CONST
            match (the production fetch's 35th slot).

        consumer_opcodes: the CONSUMER CLASS — a tuple of
            ``(name, lo_nibble, hi_nibble)`` opcode one-hot pairs. The flag
            fires iff the fetched next opcode matches ANY of these.
        consumer_flag_band: the bounded consumer-class flag band (1 cell).
        consumer_flag_threshold: the per-opcode two-nibble AND threshold.

        relay_target_marker: the marker dim the RELAY head's Q anchors (the row
            the dump fires on). K matches ``pc_chain_gate_marker`` (the AX row).
        relay_alibi_slope: the relay head's ALiBi slope (positive => step-local,
            the CURRENT step's AX row wins).

        prior_opcodes: the PRIOR-class opcode dim names the causal latch matches
            (the per-step arith opcodes on prior AX rows). EMPTY => NO latch
            head is generated and the dump-block gate degenerates to the
            consumer-class flag alone.
        prior_latch_band: the prior-class latch band (1 cell). Ignored when
            ``prior_opcodes`` is empty.
        prior_latch_k_weight: per-opcode K-match weight (the latch's ``OPW``).
        prior_latch_baseline_weight: the faint CONST/marker K baseline so a
            no-prior frame has a defined (non-matching) attention target.
        prior_latch_alibi_slope: the latch head's ALiBi slope (0 => flat, every
            prior row reachable).

        dump_block_band: the combined dump-block flag band (1 cell) the gated
            band/dump reads. = AND(consumer_flag, prior_latch) when the latch is
            present, else = consumer_flag.
        dump_block_consumer_weight / dump_block_prior_weight: the AND-gate input
            weights (bound the silu). The hand-built #221 uses 0.1 (consumer ~50
            -> +5) and 0.5 (latch ~5 -> +2.5).
        dump_block_threshold: the AND-gate threshold (both present clears it;
            either alone is dark).

        band_owner_pc / band_owner_opcode / band_owner_flag /
        band_owner_latch / band_owner_dump: ``register_residual_band(owner=...)``
            diagnostic strings (kept identical to the hand-built ``owner=`` for a
            byte-identity migration — the registry is collision-checked on
            ``(name, size, owner, never_share)``). ``None`` => use ``name``.
    """

    name: str

    # PC+offset chain
    pc_offset: int
    pc_band_lo: str
    pc_band_hi: str
    pc_chain_source_lo: str
    pc_chain_source_hi: str
    pc_chain_gate_marker: str

    # opcode fetch head
    opcode_band_lo: str
    opcode_band_hi: str
    fetch_addr_key: str
    fetch_clean_embed_lo: str
    fetch_clean_embed_hi: str
    fetch_marker: str
    fetch_const: str
    fetch_has_se: str

    # consumer-class flag
    consumer_opcodes: Tuple[Tuple[str, int, int], ...]
    consumer_flag_band: str

    # within-step relay
    relay_target_marker: str

    # combined dump-block flag
    dump_block_band: str

    feature_flag: Optional[Callable[[], bool]] = None

    pc_chain_magnitude: float = 2.0

    fetch_addr_weight: float = 20.0
    fetch_marker_weight: float = 20.0
    fetch_gate_weight: float = 500.0
    fetch_top_slot: int = 35

    consumer_flag_threshold: float = 1.5

    relay_alibi_slope: float = 1.0

    # prior-class causal latch (optional)
    prior_opcodes: Tuple[str, ...] = ()
    prior_latch_band: str = ""
    prior_latch_k_weight: float = 12.0
    prior_latch_baseline_weight: float = 0.5
    prior_latch_alibi_slope: float = 0.0

    dump_block_consumer_weight: float = 0.1
    dump_block_prior_weight: float = 0.5
    dump_block_threshold: float = 6.0

    band_owner_pc: Optional[str] = None
    band_owner_opcode: Optional[str] = None
    band_owner_flag: Optional[str] = None
    band_owner_latch: Optional[str] = None
    band_owner_dump: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pc_offset <= 0:
            raise ValueError(
                f"ConsumerLookaheadGateSpec({self.name!r}): pc_offset must be "
                f"positive, got {self.pc_offset}"
            )
        if not self.consumer_opcodes:
            raise ValueError(
                f"ConsumerLookaheadGateSpec({self.name!r}): consumer_opcodes "
                "must be non-empty"
            )
        if self.prior_opcodes and not self.prior_latch_band:
            raise ValueError(
                f"ConsumerLookaheadGateSpec({self.name!r}): prior_opcodes set "
                "but prior_latch_band empty"
            )

    @property
    def has_prior_latch(self) -> bool:
        return bool(self.prior_opcodes)


@dataclass(frozen=True)
class ConsumerLookaheadGateBundle:
    """The artifacts :func:`consumer_lookahead_gate` generates.

    The SIX bands have ALREADY been registered (import-time side effect) by the
    time this bundle exists. Every builder is pure (no global side effects) so
    the op factories call them at bake time identically.

    Attributes:
        spec: the originating :class:`ConsumerLookaheadGateSpec`.
        pc_chain_rules_builder: ``(S: float) -> tuple[FFNRule, ...]``. The
            PC+offset nibble-rotation chain rules.
        pc_chain_hidden_dim: the chain's exact hidden-unit count (the op
            factory asserts no rule-count drift).
        opcode_fetch_head_spec_builder: ``(dim_positions, head_idx) ->
            DeclarativeAttentionHeadSpec``. The content-match fetch head.
        consumer_flag_rules_builder: ``() -> tuple[FFNRule, ...]``. The
            consumer-class flag OR rules.
        relay_head_spec_builder: ``(dim_positions, head_idx, S) ->
            DeclarativeAttentionHeadSpec``. The within-step broadcast head.
        prior_latch_head_spec_builder: ``(dim_positions, head_idx, S) ->
            DeclarativeAttentionHeadSpec`` or ``None`` (when no latch).
        dump_block_rules_builder: ``() -> tuple[FFNRule, ...]``. The combined
            dump-block AND-gate rule(s).
        *_reads / *_writes: the per-op Operation read/write dim-name sets
            (flag-on).
    """

    spec: ConsumerLookaheadGateSpec
    pc_chain_rules_builder: Callable[[float], Tuple[FFNRule, ...]]
    pc_chain_hidden_dim: int
    opcode_fetch_head_spec_builder: Callable[
        [Dict[str, int], int], DeclarativeAttentionHeadSpec
    ]
    consumer_flag_rules_builder: Callable[[], Tuple[FFNRule, ...]]
    relay_head_spec_builder: Callable[
        [Dict[str, int], int, float], DeclarativeAttentionHeadSpec
    ]
    prior_latch_head_spec_builder: Optional[
        Callable[[Dict[str, int], int, float], DeclarativeAttentionHeadSpec]
    ]
    dump_block_rules_builder: Callable[[], Tuple[FFNRule, ...]]
    pc_chain_reads: Set[str]
    pc_chain_writes: Set[str]
    fetch_reads: Set[str]
    fetch_writes: Set[str]
    consumer_flag_reads: Set[str]
    consumer_flag_writes: Set[str]
    relay_reads: Set[str]
    relay_writes: Set[str]
    prior_latch_reads: Set[str]
    prior_latch_writes: Set[str]
    dump_block_reads: Set[str]
    dump_block_writes: Set[str]


def consumer_lookahead_gate(
    spec: ConsumerLookaheadGateSpec,
) -> ConsumerLookaheadGateBundle:
    """Generate the consumer-opcode lookahead-gate scaffolding for ``spec``.

    Side effect (import time): registers the SIX (five when no prior latch)
    feature bands via :func:`register_residual_band` with ``never_share=True``
    and ``flag=spec.feature_flag``. Call this at MODULE-IMPORT scope of the
    owning ``lN_ops.py`` (the SAME position the hand-built
    ``register_residual_band`` calls occupied) — registration order is
    load-bearing for the tail ``dim_positions``.

    Returns a :class:`ConsumerLookaheadGateBundle` whose builders reproduce the
    hand-built ops byte-identically.
    """
    has_latch = spec.has_prior_latch

    def _owner(explicit: Optional[str]) -> str:
        return explicit if explicit is not None else spec.name

    # (1) Register the SIX feature bands (import-time side effect). 16-wide
    #     address + opcode bands, 1-wide flag/latch/dump bands.
    register_residual_band(
        spec.pc_band_lo, 16, owner=_owner(spec.band_owner_pc),
        flag=spec.feature_flag, never_share=True,
    )
    register_residual_band(
        spec.pc_band_hi, 16, owner=_owner(spec.band_owner_pc),
        flag=spec.feature_flag, never_share=True,
    )
    register_residual_band(
        spec.opcode_band_lo, 16, owner=_owner(spec.band_owner_opcode),
        flag=spec.feature_flag, never_share=True,
    )
    register_residual_band(
        spec.opcode_band_hi, 16, owner=_owner(spec.band_owner_opcode),
        flag=spec.feature_flag, never_share=True,
    )
    register_residual_band(
        spec.consumer_flag_band, 1, owner=_owner(spec.band_owner_flag),
        flag=spec.feature_flag, never_share=True,
    )
    if has_latch:
        register_residual_band(
            spec.prior_latch_band, 1, owner=_owner(spec.band_owner_latch),
            flag=spec.feature_flag, never_share=True,
        )
    register_residual_band(
        spec.dump_block_band, 1, owner=_owner(spec.band_owner_dump),
        flag=spec.feature_flag, never_share=True,
    )

    # (2) PC+offset nibble-rotation chain rules. Reuses the declarative L4
    #     chain (the same one L4 uses for PC+1..+4) with ``offset=pc_offset``.
    pc_chain_hidden_dim = 32 + 32 * spec.pc_offset  # offset, with carry

    def pc_chain_rules_builder(S: float) -> Tuple[FFNRule, ...]:
        # Lazy import: the L4 chain helper lives in the ops package; importing
        # it at module scope would create an import cycle (l4_ops imports the
        # registry which this module's siblings populate).
        from .ops.l4_ops import _nibble_rotation_chain_rules
        return _nibble_rotation_chain_rules(
            name_prefix=f"{spec.name}_pc{spec.pc_offset}_ax",
            gate_marker_name=spec.pc_chain_gate_marker,
            source_lo_name=spec.pc_chain_source_lo, source_lo_offset=0,
            source_hi_name=spec.pc_chain_source_hi, source_hi_offset=0,
            target_lo_name=spec.pc_band_lo, target_lo_offset=0,
            target_hi_name=spec.pc_band_hi, target_hi_offset=0,
            offset=spec.pc_offset, with_carry=True, S=S,
            magnitude=spec.pc_chain_magnitude,
            scope=spec.pc_chain_gate_marker,
        )

    # (3) Opcode-fetch head: Q = PC+offset address (lo 0..15, hi 16..31) +
    #     marker gate + CONST top-nibble + hard MARK/HAS_SE gates; K = ADDR_KEY;
    #     V/O = CLEAN_EMBED -> opcode band. Mirrors the production L5 fetch head.
    def opcode_fetch_head_spec_builder(
        dim_positions: Dict[str, int], head_idx: int
    ) -> DeclarativeAttentionHeadSpec:
        def _P(name: str) -> int:
            return int(dim_positions[name])

        ADDR_KEY = _P(spec.fetch_addr_key)
        CLEAN_LO = _P(spec.fetch_clean_embed_lo)
        CLEAN_HI = _P(spec.fetch_clean_embed_hi)
        MARK = _P(spec.fetch_marker)
        CONST = _P(spec.fetch_const)
        HAS_SE = _P(spec.fetch_has_se)
        la_lo = _P(spec.pc_band_lo)
        la_hi = _P(spec.pc_band_hi)
        next_lo = _P(spec.opcode_band_lo)
        next_hi = _P(spec.opcode_band_hi)

        ADDR_L = spec.fetch_addr_weight
        L = spec.fetch_marker_weight
        TOP = spec.fetch_top_slot
        G = spec.fetch_gate_weight

        q = (
            tuple(AP(k, la_lo + k, ADDR_L) for k in range(16))
            + tuple(AP(16 + k, la_hi + k, ADDR_L) for k in range(16))
            + (AP(32, MARK, L),)
            + (AP(TOP, CONST, ADDR_L),)
            + (AP(33, MARK, G), AP(33, CONST, -G))
            + (AP(34, HAS_SE, G), AP(34, CONST, -G))
        )
        k = (
            tuple(AP(k_, ADDR_KEY + k_, ADDR_L) for k_ in range(16))
            + tuple(AP(16 + k_, ADDR_KEY + 16 + k_, ADDR_L) for k_ in range(16))
            + (AP(TOP, ADDR_KEY + 32, ADDR_L),)
            + (AP(33, MARK, G), AP(33, CONST, -G))
            + (AP(34, CONST, 5.0),)
        )
        v = (
            tuple(AP(32 + k_, CLEAN_LO + k_, 1.0) for k_ in range(16))
            + tuple(AP(48 + k_, CLEAN_HI + k_, 1.0) for k_ in range(16))
        )
        o = (
            tuple(AO(next_lo + k_, 32 + k_, 1.0) for k_ in range(16))
            + tuple(AO(next_hi + k_, 48 + k_, 1.0) for k_ in range(16))
        )
        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx, q=q, k=k, v=v, o=o, alibi_slope=0.0,
        )

    # (4) Consumer-class flag rules: per-opcode two-nibble AND, OR'd into the
    #     bounded flag (1.0 per match).
    def consumer_flag_rules_builder() -> Tuple[FFNRule, ...]:
        rules: list[FFNRule] = []
        for op_name, lo, hi in spec.consumer_opcodes:
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_consumer_{op_name.lower()}",
                conditions=(
                    (f"{spec.opcode_band_lo}+{lo}", 1.0),
                    (f"{spec.opcode_band_hi}+{hi}", 1.0),
                ),
                threshold=spec.consumer_flag_threshold,
                writes=((spec.consumer_flag_band, 1.0),),
            ))
        return tuple(rules)

    # (5) Within-step relay head: Q@relay_target_marker, K@pc_chain_gate_marker,
    #     V=flag, O->flag. Positive ALiBi keeps it step-local.
    def relay_head_spec_builder(
        dim_positions: Dict[str, int], head_idx: int, S: float
    ) -> DeclarativeAttentionHeadSpec:
        def _P(name: str) -> int:
            return int(dim_positions[name])
        TGT = _P(spec.relay_target_marker)
        SRC = _P(spec.pc_chain_gate_marker)
        FLAG = _P(spec.consumer_flag_band)
        L = float(S)
        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=(AP(0, TGT, L),),
            k=(AP(0, SRC, L),),
            v=(AP(0, FLAG, 1.0),),
            o=(AO(FLAG, 0, 1.0),),
            alibi_slope=spec.relay_alibi_slope,
        )

    # (6) Optional causal prior-class latch head: Q@relay_target_marker; K
    #     matches the prior-class opcode dims on PRIOR AX rows + a faint
    #     marker baseline; V = those opcode dims; O -> latch. Flat ALiBi.
    prior_latch_head_spec_builder: Optional[
        Callable[[Dict[str, int], int, float], DeclarativeAttentionHeadSpec]
    ] = None
    if has_latch:
        def prior_latch_head_spec_builder(  # type: ignore[misc]
            dim_positions: Dict[str, int], head_idx: int, S: float
        ) -> DeclarativeAttentionHeadSpec:
            def _P(name: str) -> int:
                return int(dim_positions[name])
            TGT = _P(spec.relay_target_marker)
            MARK_AX = _P(spec.pc_chain_gate_marker)
            LATCH = _P(spec.prior_latch_band)
            OPW = spec.prior_latch_k_weight
            q = (AP(0, TGT, float(S)),)
            k = (
                tuple(AP(0, _P(op), OPW) for op in spec.prior_opcodes)
                + (AP(0, MARK_AX, spec.prior_latch_baseline_weight),)
            )
            v = tuple(AP(0, _P(op), 1.0) for op in spec.prior_opcodes)
            o = (AO(LATCH, 0, 1.0),)
            return DeclarativeAttentionHeadSpec(
                head_idx=head_idx, q=q, k=k, v=v, o=o,
                alibi_slope=spec.prior_latch_alibi_slope,
            )

    # (7) Combined dump-block flag. AND(consumer_flag, prior_latch) when the
    #     latch is present, else the consumer flag alone.
    def dump_block_rules_builder() -> Tuple[FFNRule, ...]:
        if has_latch:
            return (
                multi_way_and_rule(
                    name=f"{spec.name}_dump_block_and",
                    conditions=(
                        (spec.consumer_flag_band, spec.dump_block_consumer_weight),
                        (spec.prior_latch_band, spec.dump_block_prior_weight),
                    ),
                    threshold=spec.dump_block_threshold,
                    writes=((spec.dump_block_band, 1.0),),
                ),
            )
        return (
            multi_way_and_rule(
                name=f"{spec.name}_dump_block_passthrough",
                conditions=(
                    (spec.consumer_flag_band, spec.dump_block_consumer_weight),
                ),
                threshold=spec.dump_block_threshold,
                writes=((spec.dump_block_band, 1.0),),
            ),
        )

    # Per-op reads/writes (flag-on), derived structurally so the op factory can
    # declare them on its Operation.
    pc_chain_reads = {spec.pc_chain_gate_marker,
                      spec.pc_chain_source_lo, spec.pc_chain_source_hi}
    pc_chain_writes = {spec.pc_band_lo, spec.pc_band_hi}

    fetch_reads = {spec.fetch_marker, spec.fetch_const, spec.fetch_has_se,
                   spec.fetch_addr_key, spec.fetch_clean_embed_lo,
                   spec.fetch_clean_embed_hi, spec.pc_band_lo, spec.pc_band_hi}
    fetch_writes = {spec.opcode_band_lo, spec.opcode_band_hi}

    consumer_flag_reads = {spec.opcode_band_lo, spec.opcode_band_hi}
    consumer_flag_writes = {spec.consumer_flag_band}

    relay_reads = {spec.relay_target_marker, spec.pc_chain_gate_marker,
                   spec.consumer_flag_band}
    relay_writes = {spec.consumer_flag_band}

    if has_latch:
        prior_latch_reads = ({spec.relay_target_marker,
                              spec.pc_chain_gate_marker}
                             | set(spec.prior_opcodes))
        prior_latch_writes = {spec.prior_latch_band}
        dump_block_reads = {spec.consumer_flag_band, spec.prior_latch_band}
    else:
        prior_latch_reads = set()
        prior_latch_writes = set()
        dump_block_reads = {spec.consumer_flag_band}
    dump_block_writes = {spec.dump_block_band}

    return ConsumerLookaheadGateBundle(
        spec=spec,
        pc_chain_rules_builder=pc_chain_rules_builder,
        pc_chain_hidden_dim=pc_chain_hidden_dim,
        opcode_fetch_head_spec_builder=opcode_fetch_head_spec_builder,
        consumer_flag_rules_builder=consumer_flag_rules_builder,
        relay_head_spec_builder=relay_head_spec_builder,
        prior_latch_head_spec_builder=prior_latch_head_spec_builder,
        dump_block_rules_builder=dump_block_rules_builder,
        pc_chain_reads=pc_chain_reads,
        pc_chain_writes=pc_chain_writes,
        fetch_reads=fetch_reads,
        fetch_writes=fetch_writes,
        consumer_flag_reads=consumer_flag_reads,
        consumer_flag_writes=consumer_flag_writes,
        relay_reads=relay_reads,
        relay_writes=relay_writes,
        prior_latch_reads=prior_latch_reads,
        prior_latch_writes=prior_latch_writes,
        dump_block_reads=dump_block_reads,
        dump_block_writes=dump_block_writes,
    )


# ---------------------------------------------------------------------------
# CAM / frame-lookup attention — the content-addressable relay head
# ---------------------------------------------------------------------------
#
# The recurring "content-addressable attention head" pattern: a single head
# that attends from a QUERY-MARKER row to the memory/stack/frame row whose
# SIGNATURE (an address / byte-index / opcode tag) matches a declared key, and
# RELAYS that matched row's VALUE band into a TARGET band. It is hand-built
# repeatedly across the VM:
#
#   * L7 ``layer7_operand_gather`` head 0 (``l7_ops.py:269-286``): Q@MARK_AX,
#     K@STACK0_BYTE0, V=CLEAN_EMBED_{LO,HI}, O->ALU_{LO,HI} — the operand-A
#     gather (the prev STACK0 byte-0 token's value -> the ALU at the AX marker).
#   * the L15 memory / LEV / LI lookups (Q@a memory-op marker, K@a byte-index
#     or address signature, V=the matched MEM row's value, O->OUTPUT / a LI
#     band).
#   * the C4_OPERAND_FROM_MEMSP operand->mem[SP] CAM (L4 SP->ADDR_KEY + L8
#     mem-to-ALU memory-attention).
#
# Each is the SAME shape; only the (query marker, key signature, value source
# band, target band, alibi slope, head layer/idx) vary. :func:`cam_lookup`
# lifts it to ONE generator. The CAM INVARIANT is made STRUCTURAL by the API
# shape: the key signature is a declared row-MATCH (``CamKeyMatch`` — a single
# ``(query_dim @ query, key_dim @ key)`` pair on the query slot), and the value
# relay is a declared ``(source_band -> target_band)`` block (``CamValueBand``).
# There are NO free Q/K/V/O writes — the only Q richness allowed is the
# declared OPCODE-BLOCKER overlay (``query_blockers``) and the optional
# CONST-anchored confirmation slot (``CamConfirmSlot``), both of which are still
# row-selection structure, not arbitrary projections.
#
# Like :func:`cross_step_carry`, the generator returns a pure builder bundle
# (``head_spec_builder(dim_positions, head_idx)``) the op factory installs; no
# compiler change — it lowers through the existing
# ``Primitives.generate_attention_heads`` path. The byte-identity proof
# re-expresses the SETTLED L7 operand-gather head 0 and gates the whole-model
# state_dict hash unchanged.


@dataclass(frozen=True)
class CamKeyMatch:
    """The content-address row MATCH — the CAM invariant made structural.

    A CAM head fires at the ``query_dim`` marker row and selects the K row
    carrying ``key_dim`` (the address / byte-index / opcode SIGNATURE). Both
    land on the head's ``query_slot`` (slot 0 in the L7 operand-gather head):
    ``AP(query_slot, query_dim, weight)`` on Q and ``AP(query_slot, key_dim,
    weight)`` on K. Expressing the match as a single declared pair — rather
    than free Q/K writes — is what makes the "attend to the row whose signature
    matches" semantics a structural contract: a CAM head has EXACTLY one key
    match slot.

    **Multi-nibble / multi-dim signature (G1(a) extension).** The L7
    operand-gather head is a SINGLE-DIM match (one query dim, one key dim). The
    MEMORY family heads that gate on a WIDER address signature — the L13
    ``mem_addr_gather`` addr-byte-J marker signature (``+L1H{n}[MEM]`` AND
    ``-L1H{n-1}[MEM]``, a two-dim positive/negative row signature) and the L15
    LI/LC load head's 24-bit binary address comparator (3 addr bytes × 2 nibbles
    × 4 bits, each a ``±scale`` per-nibble bit-encoding) — need the match to
    span MULTIPLE dims on the SAME query slot. :attr:`query_extra` /
    :attr:`key_extra` carry those extra ``(dim, weight)`` pairs so the row
    select stays ONE structural signature (still exactly one match SLOT), not a
    free Q/K field. A single-dim match leaves both empty (byte-identical to the
    L7 head). The 24-bit comparator is the degenerate case where the SAME dim
    list appears on both Q and K with matching ``±scale`` per-nibble weights
    (the bit-encoding makes the score peak on the row whose nibbles equal the
    queried address); express it by supplying the identical
    ``query_extra == key_extra`` tuple.

    Attributes:
        query_dim: the QUERY-MARKER residual dim (the row the head fires on;
            L7: ``MARK_AX``; L13 head 0: ``MEM_VAL_B0``).
        key_dim: the KEY-SIGNATURE residual dim the matched K row must carry
            (the content address; L7: ``STACK0_BYTE0``; L13 head 0:
            ``L1H1[MEM]`` at ``+weight``).
        weight: the shared Q/K projection weight for the primary pair (L7/L13:
            ``15.0``).
        query_slot: head-local slot the match lands on (L7/L13: ``0``).
        query_extra: additional ``(dim, weight)`` Q writes on the SAME
            ``query_slot`` (the multi-dim query signature — L13 head 0's
            ``MEM_VAL_B1/B2/B3`` fire dims; L15's 24-bit address Q block).
            Empty => single-dim query (L7).
        key_extra: additional ``(dim, weight)`` K writes on the SAME
            ``query_slot`` (the multi-dim key signature — L13 head 0's
            ``-L1H0[MEM]`` negative reject; L15's 24-bit address K block).
            Empty => single-dim key (L7).
    """

    query_dim: str
    key_dim: str
    weight: float
    query_slot: int = 0
    query_extra: Tuple[Tuple[str, float], ...] = ()
    key_extra: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CamConfirmSlot:
    """An optional CONST-anchored confirmation slot (sharpens the row select).

    The L7 operand-gather head adds a SECOND Q/K slot (slot 33) that anchors a
    CONST key and re-asserts the marker + amplified opcode blockers, so the
    softmax winner is pinned even when the primary signature is weakly present.
    Structurally it is: Q ``AP(slot, marker_dim, marker_weight)`` +
    ``AP(slot, const_dim, const_q_weight)`` + the amplified ``blockers``; K
    ``AP(slot, const_dim, const_k_weight)``. The marker / const dims default to
    the match's query dim / the spec ``const_dim``.

    Attributes:
        slot: head-local confirmation slot (L7: ``33``).
        marker_weight: the marker Q write on the confirm slot (L7: ``+15.0``).
        const_q_weight: the CONST Q write on the confirm slot (L7: ``-7.5``).
        const_k_weight: the CONST K write on the confirm slot (L7: ``+15.0``).
        blockers: amplified ``(opcode_dim, weight)`` Q rejects on the confirm
            slot (L7: ``OP_LEA/ADJ/ENT`` @ ``-150.0``). Negative => reject.
        marker_dim: the marker dim re-asserted on the confirm slot. ``None`` =>
            reuse the key-match ``query_dim``.
    """

    slot: int
    marker_weight: float
    const_q_weight: float
    const_k_weight: float
    blockers: Tuple[Tuple[str, float], ...] = ()
    marker_dim: Optional[str] = None


@dataclass(frozen=True)
class CamValidSlot:
    """The VALID-lifecycle bit slot (the addressed-load's "row found" flag).

    The L13 ``mem_addr_gather`` heads carry a THIRD gate slot (slot 34) that
    MIRRORS the key-match's fire + signature (so it selects the same row) and
    relays a SINGLE ``1.0`` VALID bit: V reads a ``valid_read_dim`` on the
    matched row (a dim guaranteed present on the addr-byte-J row and absent
    elsewhere, so the softmax-weighted V is ``1.0`` on a hit and ``0`` on a
    miss) and O writes it into ``valid_write_dim`` (``ADDR_BJ_VALID``). This is
    the CAM lifecycle datum G1 names: "did the address gather find its row".

    Structurally the slot re-declares the SAME query/key signature as the
    :class:`CamKeyMatch` (fire dims on Q at ``+weight``, the key signature on K)
    on its own ``slot`` so the match is independent of the value relay. The
    single V/O pair copies the VALID bit.

    Attributes:
        slot: head-local slot the VALID lifecycle lands on (L13: ``34``).
        valid_read_dim: the dim V reads on the matched row (delivers ``1.0`` on
            a hit; L13 head 0: ``L1H1[MEM]`` — the same signature dim).
        valid_write_dim: the VALID band O writes the bit into (L13 head 0:
            ``ADDR_B0_VALID``).
        v_slot: the head-local V/O slot for the bit copy (L13: reuses ``slot``).
        o_scale: O-write magnitude (L13: ``1.0``).
    """

    slot: int
    valid_read_dim: str
    valid_write_dim: str
    v_slot: Optional[int] = None
    o_scale: float = 1.0

    def resolved_v_slot(self) -> int:
        return self.slot if self.v_slot is None else self.v_slot


@dataclass(frozen=True)
class CamValueBand:
    """One ``source_band -> target_band`` value-relay block.

    V copies ``source_band[0..width-1]`` into head-local V slots
    ``v_slot_base..v_slot_base+width-1``; O writes those slots into
    ``target_band[0..width-1]`` at ``o_scale``. The L7 operand-gather head
    relays TWO bands: ``CLEAN_EMBED_LO -> ALU_LO`` (slots 1..16) and
    ``CLEAN_EMBED_HI -> ALU_HI`` (slots 17..32), both at ``o_scale=6.0``.

    Attributes:
        source_band: the matched row's VALUE band the V slots read.
        target_band: the band the O writes the relayed value into.
        width: cell count (L7: ``16`` — a byte's nibble one-hot).
        v_slot_base: head-local V slot base (L7: ``1`` and ``17``).
        o_scale: O-write magnitude (L7: ``6.0``).
    """

    source_band: str
    target_band: str
    width: int
    v_slot_base: int
    o_scale: float


@dataclass(frozen=True)
class CamLookupSpec:
    """Declarative description of a CAM / frame-lookup attention head.

    A CAM head attends from a query-marker row to the row whose declared
    SIGNATURE matches a key, and relays that row's value band to a target band.
    Every field is a VARYING parameter; the IDENTICAL head structure (the key
    MATCH + the opcode-blocker overlay + the optional confirm slot + the value
    relay blocks) is supplied by :func:`cam_lookup`.

    The CAM invariant is enforced by the API shape: the row selection is a
    single :class:`CamKeyMatch` (one query/key signature pair), the value flow
    is a tuple of :class:`CamValueBand` relay blocks, and the only extra Q
    structure is the declared :attr:`query_blockers` overlay + the optional
    :class:`CamConfirmSlot`. There is NO free Q/K/V/O field.

    Attributes:
        name: head family name (rule-name / diagnostic prefix).
        key_match: the content-address row MATCH (:class:`CamKeyMatch`).
        value_bands: the value-relay blocks (:class:`CamValueBand`), in the
            order the hand-built head appended them (the V/O write order is NOT
            load-bearing for the lowered weights — ``generate_attention_head``
            uses direct indexed assignment — but the order is kept identical for
            a clean diff).
        alibi_slope: per-head ALiBi slope. ``None`` => the op writes its own
            ``alibi_slopes`` (the L7 operand-gather head's slope is set by the
            op's ``attn.alibi_slopes.fill_(0.5)`` bake, NOT the spec, so the
            re-expression leaves it ``None`` to stay byte-identical).
        query_blockers: opcode-reject Q writes on the key-match query slot —
            ``(opcode_dim, weight)`` with a NEGATIVE weight rejecting wrong-op
            rows (L7: ``OP_LEA/ADJ/ENT`` @ ``-15.0``). These narrow WHICH marker
            rows the head fires on; they are row-selection structure, not free
            projections.
        confirm: the optional CONST-anchored confirmation slot
            (:class:`CamConfirmSlot`). ``None`` => no second gate slot.
        const_dim: the CONST residual dim name (the confirm slot's anchor).
        value_active: when ``False`` the value relay (V/O) is OMITTED while the
            Q/K row-select gates are KEPT — the C4_OPERAND_FROM_MEMSP
            "disable the old operand source but leave the head slot/layout
            unchanged" mode (``l7_ops.py:251-267``). Byte-identical to the
            hand-built suppressed head.
        extra_reads: extra dim names the head's Operation should declare as
            reads beyond the auto-derived set (e.g. a cross-step ``X.*.-1``
            alias the source band is read through).
        direction: the CAM data-flow DIRECTION (G1(b)). ``"load"`` (default) =
            read-by-address: attend to the row whose signature matches and RELAY
            its value into the target band (LI/LC, L7 operand gather, L13 addr
            gather — every hand-built CAM head today). ``"store"`` = the
            emit-with-address direction (SI/SC/PSH gather addr+value from the
            AX/SP frame INTO a MEM token). ``"store"`` is a SEMANTIC-INTENT tag
            on the same head machinery: the L14 mem-generation emit heads are
            still hand-built, so ``cam_lookup`` only LOWERS ``"load"`` today; a
            ``"store"`` spec raises unless :attr:`value_bands` is supplied with
            the emit routing (reserved for the L14 port). Recording the
            direction lets a generic engine know a store binds address→value at
            the marker (G2) vs a load reads it back.
        valid_slots: the VALID-lifecycle bit slots (:class:`CamValidSlot`) — the
            "row found" flags an addressed gather relays alongside its value
            (L13 ``ADDR_BJ_VALID``). Empty => no lifecycle bit (L7).
    """

    name: str
    key_match: CamKeyMatch
    value_bands: Tuple[CamValueBand, ...]
    alibi_slope: Optional[float] = None
    query_blockers: Tuple[Tuple[str, float], ...] = ()
    confirm: Optional[CamConfirmSlot] = None
    const_dim: str = "CONST"
    value_active: bool = True
    extra_reads: Tuple[str, ...] = ()
    direction: str = "load"
    valid_slots: Tuple[CamValidSlot, ...] = ()

    def __post_init__(self) -> None:
        if not self.value_bands:
            raise ValueError(
                f"CamLookupSpec({self.name!r}): value_bands must be non-empty "
                "(a CAM head relays at least one value band)"
            )
        for vb in self.value_bands:
            if vb.width <= 0:
                raise ValueError(
                    f"CamLookupSpec({self.name!r}): value band "
                    f"{vb.source_band!r}->{vb.target_band!r} width must be "
                    f"positive, got {vb.width}"
                )
        if self.direction not in ("load", "store"):
            raise ValueError(
                f"CamLookupSpec({self.name!r}): direction must be "
                f"'load' or 'store', got {self.direction!r}"
            )
        if self.direction == "store":
            # The store/emit lowering (L14 mem-generation) is not yet ported to
            # cam_lookup; a store spec is accepted as a SEMANTIC tag only when
            # the caller also supplies the emit value routing. Guard against a
            # silent no-op store head.
            if not self.value_active:
                raise ValueError(
                    f"CamLookupSpec({self.name!r}): a 'store' direction head "
                    "must relay its emit value bands (value_active=True)"
                )


@dataclass(frozen=True)
class CamLookupBundle:
    """The artifacts :func:`cam_lookup` generates for one CAM head.

    The builder is pure (no global side effects) so the op factory can call it
    at bake time and IR-factory time identically.

    Attributes:
        spec: the originating :class:`CamLookupSpec`.
        head_spec_builder: ``(dim_positions, head_idx) ->
            DeclarativeAttentionHeadSpec``. Reproduces the hand-built CAM head
            EXACTLY (same Q/K/V/O writes at the resolved dim positions).
        head_reads / head_writes: dim-name sets for the head's Operation. The
            head READS the key-match query/key dims + the blocker opcode dims +
            the value source bands; WRITES the value target bands. When
            ``value_active`` is False the value bands are still declared (the
            slots exist) but the source reads / target writes are omitted from
            the auto sets, matching the suppressed hand-built head's deps.
    """

    spec: CamLookupSpec
    head_spec_builder: Callable[
        [Dict[str, int], int], DeclarativeAttentionHeadSpec
    ]
    head_reads: Set[str]
    head_writes: Set[str]


def cam_lookup(spec: CamLookupSpec) -> CamLookupBundle:
    """Generate the content-addressable / frame-lookup attention head for ``spec``.

    Returns a :class:`CamLookupBundle` whose ``head_spec_builder`` reproduces
    the hand-built CAM head byte-identically. Unlike :func:`cross_step_carry`
    this generator registers NO residual band (a CAM head relays into EXISTING
    bands — ALU / OUTPUT / a LI band — so there is no import-time side effect).
    """
    km = spec.key_match
    confirm = spec.confirm

    def head_spec_builder(
        dim_positions: Dict[str, int], head_idx: int
    ) -> DeclarativeAttentionHeadSpec:
        # Resolve a ``BASE`` or ``BASE+offset`` dim token. The L7 operand-gather
        # head is single-dim / no-offset (``STACK0_BYTE0``, ``ALU_LO``), so
        # every token there resolves plainly. The MEMORY-family CAM signatures
        # address a MARKER-RELATIVE row byte via an OFFSET into a marker bank —
        # the L13 addr-byte-J key signature is ``L1H1+MEM_I`` / ``-L1H0+MEM_I``
        # (``MEM_I=4``), and the VALID read taps the same offset dim. Parsing the
        # ``+offset`` here (via the module-level :func:`_resolve_dim_token`) lets
        # a ``CamKeyMatch`` express the multi-nibble marker-relative comparator
        # without contorting the caller into pre-adding the offset — the offset
        # stays visible in the declared signature. Plain names are unchanged
        # (byte-identical to the L7 head).
        def _P(name: str) -> int:
            return _resolve_dim_token(name, lambda n: int(dim_positions[n]))

        # (1) The content-address row MATCH on the query slot: Q@query_dim,
        #     K@key_dim — the CAM invariant. ONE match SLOT. The primary
        #     (query_dim, key_dim) pair plus any multi-dim signature extras
        #     (query_extra/key_extra) — the L13 addr-byte-J +L1H/-L1H signature
        #     and the L15 24-bit binary address comparator. All land on the SAME
        #     query_slot so the row select stays one declared signature.
        q: list = [AP(km.query_slot, _P(km.query_dim), km.weight)]
        k: list = [AP(km.query_slot, _P(km.key_dim), km.weight)]
        for (q_dim, q_w) in km.query_extra:
            q.append(AP(km.query_slot, _P(q_dim), q_w))
        for (k_dim, k_w) in km.key_extra:
            k.append(AP(km.query_slot, _P(k_dim), k_w))

        # (2) The opcode-blocker overlay on the SAME query slot (narrow which
        #     marker rows fire — negative weights reject wrong-op rows).
        for (op_dim, w) in spec.query_blockers:
            q.append(AP(km.query_slot, _P(op_dim), w))

        # (3) The optional CONST-anchored confirmation slot (re-assert the
        #     marker + amplified blockers; anchor a CONST key so the softmax
        #     winner is pinned). Marker dim defaults to the match query dim.
        if confirm is not None:
            marker_dim = (
                confirm.marker_dim if confirm.marker_dim is not None
                else km.query_dim
            )
            const = _P(spec.const_dim)
            q.append(AP(confirm.slot, _P(marker_dim), confirm.marker_weight))
            q.append(AP(confirm.slot, const, confirm.const_q_weight))
            for (op_dim, w) in confirm.blockers:
                q.append(AP(confirm.slot, _P(op_dim), w))
            k.append(AP(confirm.slot, const, confirm.const_k_weight))

        # (4) The value relay blocks: V copies source_band -> V slots; O writes
        #     those slots into target_band. Omitted when value_active is False
        #     (the row-select gates stay, the relay is suppressed).
        v: list = []
        o: list = []
        if spec.value_active:
            for vb in spec.value_bands:
                src = _P(vb.source_band)
                tgt = _P(vb.target_band)
                for j in range(vb.width):
                    v.append(AP(vb.v_slot_base + j, src + j, 1.0))
                    o.append(AO(tgt + j, vb.v_slot_base + j, vb.o_scale))

        # (5) The VALID-lifecycle bit slots: each re-declares the key-match
        #     signature on its own slot (so it selects the same row independently
        #     of the value relay) and copies a single "row found" 1.0 bit from
        #     the matched row into the VALID band. Q mirrors the fire dims, K
        #     mirrors the key signature; V reads the valid_read_dim (present on
        #     the matched row), O writes valid_write_dim.
        for vs in spec.valid_slots:
            slot = vs.slot
            q.append(AP(slot, _P(km.query_dim), km.weight))
            for (q_dim, q_w) in km.query_extra:
                q.append(AP(slot, _P(q_dim), q_w))
            k.append(AP(slot, _P(km.key_dim), km.weight))
            for (k_dim, k_w) in km.key_extra:
                k.append(AP(slot, _P(k_dim), k_w))
            if spec.value_active:
                vslot = vs.resolved_v_slot()
                v.append(AP(vslot, _P(vs.valid_read_dim), 1.0))
                o.append(AO(_P(vs.valid_write_dim), vslot, vs.o_scale))

        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            alibi_slope=spec.alibi_slope,
        )

    # Operation dep-graph dim sets (structural derivation). A ``BASE+offset``
    # signature token (the L13 marker-relative addr-byte comparator) contributes
    # its BASE band name to the dep graph — the residual band is tracked by base
    # name, and the offset is a within-bank cell selector, not a separate dim.
    def _base(name: str) -> str:
        return name.split("+", 1)[0]

    head_reads: Set[str] = {_base(km.query_dim), _base(km.key_dim), spec.const_dim}
    for (q_dim, _w) in km.query_extra:
        head_reads.add(_base(q_dim))
    for (k_dim, _w) in km.key_extra:
        head_reads.add(_base(k_dim))
    for (op_dim, _w) in spec.query_blockers:
        head_reads.add(_base(op_dim))
    if confirm is not None:
        if confirm.marker_dim is not None:
            head_reads.add(_base(confirm.marker_dim))
        for (op_dim, _w) in confirm.blockers:
            head_reads.add(_base(op_dim))
    head_writes: Set[str] = set()
    if spec.value_active:
        for vb in spec.value_bands:
            head_reads.add(_base(vb.source_band))
            head_writes.add(_base(vb.target_band))
        # The VALID lifecycle bit reads its "row found" dim and writes the
        # VALID band; only active alongside the value relay.
        for vs in spec.valid_slots:
            head_reads.add(_base(vs.valid_read_dim))
            head_writes.add(_base(vs.valid_write_dim))
    head_reads.update(spec.extra_reads)

    return CamLookupBundle(
        spec=spec,
        head_spec_builder=head_spec_builder,
        head_reads=head_reads,
        head_writes=head_writes,
    )


# ===========================================================================
# FRAME-RELAY — the multi-slot FRAME-lookup value relay (L7 head-1 LEA/ADJ/ENT
# BP/SP OUTPUT -> ALU), derived GENERICALLY
# ===========================================================================
#
# The L7 operand-gather HEAD 1 (``layer7_operand_gather.head_1``) is the sibling
# of head 0: it gathers the LIVE frame BP/SP ``OUTPUT`` into the ALU at the AX
# marker (operand-A ADDRESS relay for LEA/ADJ/ENT). It is a value relay like
# :func:`cam_lookup`, but its ROW SELECT is NOT a single content-address key
# match: it fires on a FRAME opcode set (``MARK_AX`` ×10 amplified + ``OP_LEA/
# ADJ/ENT`` gates + an ``OP_IMM`` per-step suppressor + a ``CONST`` bias), its K
# selects the frame MARKER by a TWO-MARKER OR (``MARK_BP`` + ``MARK_SP``, both
# at the SAME weight, so the head attends to whichever frame marker is live),
# and it carries a SECOND CONST-anchored gate slot (slot 1: ``MARK_AX`` re-assert
# + ``CONST`` bias) plus an OPTIONAL campaign re-sharpen gate slot (slot 2:
# ``OP_LEA`` query / ``OP_ENT`` key). None of that fits ``cam_lookup``'s single
# shared-weight ``CamKeyMatch`` (the Q primary is ×10 the K primary; the key is a
# two-marker OR, not one address).
#
# :func:`frame_relay` is the GENERAL frame-lookup value relay of which
# :func:`cam_lookup` is the constrained content-address special case: a FREE
# multi-slot Q/K GATE signature (row-selection structure — markers, opcode gates,
# CONST anchors, per-step suppressors) drives the SAME :class:`CamValueBand`
# value flow (source band -> V slots -> target band) with the SAME
# ``value_active`` suppression semantics (keep the gates, drop the relay). The
# gates are raw ``(slot, dim, weight)`` tuples because a frame select is genuine
# row-selection structure, not a value projection — the CAM discipline (no free
# V/O) is preserved: V/O is EXCLUSIVELY the declared value bands.
#
# Returns a pure builder bundle (``head_spec_builder(dim_positions, head_idx)``)
# the op factory installs; no compiler change — lowers through
# ``Primitives.generate_attention_head``. Byte-identity re-expresses the SETTLED
# L7 head-1 frame relay and gates the whole-model state_dict hash unchanged.


@dataclass(frozen=True)
class FrameRelaySpec:
    """Declarative description of a multi-slot frame-lookup value-relay head.

    The head fires at a frame-op marker row (``q_gates``: a free multi-slot Q
    signature of marker + opcode-gate + CONST-bias writes), its K selects the
    frame marker (``k_gates``: a free multi-slot K signature — a two-marker OR,
    CONST anchors), and it relays value bands (``value_bands``) from the attended
    frame row into a target band. Every field is a VARYING parameter; the value
    flow is the SAME :class:`CamValueBand` machinery as :func:`cam_lookup`.

    Attributes:
        name: head family name (rule-name / diagnostic prefix).
        q_gates: the Q-projection ``(slot, dim, weight)`` writes (the row-select
            fire signature — markers, opcode gates, CONST bias, per-step
            suppressors). Each ``dim`` is a ``BASE`` or ``BASE+offset`` token.
        k_gates: the K-projection ``(slot, dim, weight)`` writes (the frame
            marker select — a two-marker OR, CONST anchors).
        value_bands: the value-relay blocks (:class:`CamValueBand`) — the frame
            row's OUTPUT band -> the ALU band, exactly as ``cam_lookup`` relays.
        alibi_slope: per-head ALiBi slope. ``None`` => the op writes its own
            ``alibi_slopes`` (L7 head-1's slope is set by the op bake, not the
            spec, so the re-expression leaves this ``None``).
        value_active: when ``False`` the value relay (V/O) is OMITTED while the
            Q/K row-select gates are KEPT — the ``C4_OPERAND_FROM_MEMSP``
            "disable the old operand source but keep the head slot/layout"
            mode. Mirrors :attr:`CamLookupSpec.value_active`.
        step_window: the step-scope contract the verifier enforces.
        extra_reads: extra dim names the head's Operation should declare beyond
            the auto-derived set (cross-step ``X.*.-1`` aliases the OUTPUT bands
            are read through — L7 head-1 reads OUTPUT_LO/HI cross-step).
    """

    name: str
    q_gates: Tuple[Tuple[int, str, float], ...]
    k_gates: Tuple[Tuple[int, str, float], ...]
    value_bands: Tuple[CamValueBand, ...]
    alibi_slope: Optional[float] = None
    value_active: bool = True
    step_window: StepWindowConstraint = (
        StepWindowConstraint.CURRENT_STEP_ONLY
    )
    extra_reads: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.q_gates:
            raise ValueError(
                f"FrameRelaySpec({self.name!r}): q_gates must be non-empty "
                "(a frame relay fires at a marker row)"
            )
        if not self.k_gates:
            raise ValueError(
                f"FrameRelaySpec({self.name!r}): k_gates must be non-empty "
                "(a frame relay selects a marker row)"
            )
        if not self.value_bands:
            raise ValueError(
                f"FrameRelaySpec({self.name!r}): value_bands must be non-empty "
                "(a frame relay relays at least one value band)"
            )
        for vb in self.value_bands:
            if vb.width <= 0:
                raise ValueError(
                    f"FrameRelaySpec({self.name!r}): value band "
                    f"{vb.source_band!r}->{vb.target_band!r} width must be "
                    f"positive, got {vb.width}"
                )


@dataclass(frozen=True)
class FrameRelayBundle:
    """The artifacts :func:`frame_relay` generates for one frame-relay head.

    Attributes:
        spec: the originating :class:`FrameRelaySpec`.
        head_spec_builder: ``(dim_positions, head_idx) ->
            DeclarativeAttentionHeadSpec``. Reproduces the hand-built frame-relay
            head EXACTLY (same Q/K/V/O writes at the resolved dim positions).
        head_reads / head_writes: dim-name sets for the head's Operation.
    """

    spec: FrameRelaySpec
    head_spec_builder: Callable[
        [Dict[str, int], int], DeclarativeAttentionHeadSpec
    ]
    head_reads: Set[str]
    head_writes: Set[str]


def frame_relay(spec: FrameRelaySpec) -> FrameRelayBundle:
    """Generate the multi-slot frame-lookup value-relay head for ``spec``.

    Returns a :class:`FrameRelayBundle` whose ``head_spec_builder`` reproduces
    the hand-built frame-relay head byte-identically. Registers NO residual band
    (the relay copies into EXISTING ALU / OUTPUT bands).
    """

    def head_spec_builder(
        dim_positions: Dict[str, int], head_idx: int
    ) -> DeclarativeAttentionHeadSpec:
        def _P(name: str) -> int:
            return _resolve_dim_token(name, lambda n: int(dim_positions[n]))

        q = [AP(slot, _P(d), w) for (slot, d, w) in spec.q_gates]
        k = [AP(slot, _P(d), w) for (slot, d, w) in spec.k_gates]

        v: list = []
        o: list = []
        if spec.value_active:
            for vb in spec.value_bands:
                src = _P(vb.source_band)
                tgt = _P(vb.target_band)
                for j in range(vb.width):
                    v.append(AP(vb.v_slot_base + j, src + j, 1.0))
                    o.append(AO(tgt + j, vb.v_slot_base + j, vb.o_scale))

        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            alibi_slope=spec.alibi_slope,
            step_window=spec.step_window,
        )

    def _base(name: str) -> str:
        return name.split("+", 1)[0]

    head_reads: Set[str] = set()
    for (_s, d, _w) in spec.q_gates:
        head_reads.add(_base(d))
    for (_s, d, _w) in spec.k_gates:
        head_reads.add(_base(d))
    head_writes: Set[str] = set()
    if spec.value_active:
        for vb in spec.value_bands:
            head_reads.add(_base(vb.source_band))
            head_writes.add(_base(vb.target_band))
    head_reads.update(spec.extra_reads)

    return FrameRelayBundle(
        spec=spec,
        head_spec_builder=head_spec_builder,
        head_reads=head_reads,
        head_writes=head_writes,
    )


# ===========================================================================
# CAM-BINARY-ADDRESS-MATCH — the multi-slot BINARY-address CAM (L15 LI/LC load,
# L14 mem-generation store), derived GENERICALLY
# ===========================================================================
#
# The G1(a) ``CamKeyMatch`` multi-dim signature lands EVERY extra Q/K write on
# ONE ``query_slot`` — the degenerate "24-bit comparator collapses to one slot"
# case the docstring names. That is exact for the L13 addr-byte gather (a
# two-dim ``+L1H/-L1H`` marker signature) but it CANNOT express the L15 LI/LC
# load head, whose address comparator is a BINARY per-bit encoding spanning 24
# SEPARATE match slots (3 addr bytes × 2 nibbles × 4 bits, slots 4-27): each
# slot ``s`` scores ``±scale`` per nibble cell by bit ``s`` of the cell index
# (``2*((k>>bit)&1)-1``), so the head-dim dot product peaks on the K row whose
# address nibbles EQUAL the queried address. A single-slot signature cannot
# carry a per-bit encoding — the bits must live on independent head-dim slots so
# their contributions ADD in the score.
#
# On TOP of that binary block the L15 head layers ~40 HETEROGENEOUS
# discriminator slots (opcode gates, marker blockers, per-store address
# one-hots, VALID/pop lifecycle rows, the campaign-gated suppressor cancels) —
# each a bespoke ``(slot, Q-writes, K-writes)`` row that is row-SELECTION
# structure, not a free projection. They are DATA: a list of
# :class:`CamDiscriminatorSlot`.
#
# :class:`CamBinaryAddressMatch` unifies the two into ONE structural CAM: a
# :class:`CamBinaryAddressBlock` (the per-bit comparator) + a tuple of
# :class:`CamDiscriminatorSlot` (the declared discriminator rows) + a tuple of
# :class:`CamValueBand` value relays. Its builder merges every write into
# per-``(slot, dim)`` maps (last-write-wins, exactly the lowerer's indexed
# assignment) so the produced head is byte-identical to the hand-authored L15
# LI/LC head. It is GENERAL: the SAME shape addresses the L14 mem-generation
# STORE (``direction="store"``) — a binary-addressed CAM in the emit direction.
#
# Like :func:`cam_lookup` the generator returns a pure builder bundle
# (``head_spec_builder(dim_positions, head_idx)``); no compiler change, it lowers
# through ``Primitives.generate_attention_head``.


@dataclass(frozen=True)
class CamBinaryAddressBlock:
    """The BINARY per-bit address comparator (the multi-slot CAM invariant).

    Unlike the single-slot ``CamKeyMatch`` signature, a binary-address block
    spans ``len(nibble_bands) * width_bits`` CONSECUTIVE head-dim slots starting
    at :attr:`slot_base`. For nibble band ``base`` (a 16-cell one-hot of a
    nibble value) and bit ``b`` in ``[0, width_bits)``, the slot
    ``slot_base + band_index*width_bits + b`` writes ``+scale`` on Q AND K at
    cell ``base + k`` when bit ``b`` of ``k`` is 1, and ``-scale`` when it is 0
    (``scale * (2*((k>>b)&1)-1)``). The head-dim dot product therefore peaks on
    the K row whose address nibbles EQUAL the queried address — the CAM select.

    This is the STRUCTURAL binary-CAM invariant: the address occupies exactly
    ``len(nibble_bands) * width_bits`` slots, each a declared per-bit ``±scale``
    comparator, with NO free Q/K writes.

    Attributes:
        nibble_bands: ordered nibble-band base dim names (each a
            ``2**width_bits``-cell one-hot). L15's 24-bit address:
            ``("ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO", "ADDR_B1_HI",
              "ADDR_B2_LO", "ADDR_B2_HI")``.
        scale: per-bit ``±scale`` Q/K weight (L15: ``10.0``).
        slot_base: head-local slot where the address block starts (L15: ``4``).
        width_bits: bits per nibble band (L15: ``4`` — a 16-cell nibble). Each
            band consumes ``width_bits`` slots.
    """

    nibble_bands: Tuple[str, ...]
    scale: float
    slot_base: int = 4
    width_bits: int = 4

    def slot_extent(self) -> int:
        """First slot index PAST the address block."""
        return self.slot_base + len(self.nibble_bands) * self.width_bits

    def cells_per_band(self) -> int:
        return 1 << self.width_bits


# Sentinel weight: an OVERLAY discriminator cell whose weight is ``CAM_DROP``
# REMOVES that ``(slot, dim)`` cell from the merged map instead of setting it —
# the DATA form of a hand-authored ``q_map.pop((slot, dim))`` / a wiped V/O row.
# Because the lowerer writes into a PRE-ZEROED weight matrix, an absent cell and
# an explicit ``0.0`` produce the SAME lowered weight; ``CAM_DROP`` exists so the
# emitted spec TUPLES also match a hand-built head that popped the cell (the
# byte-identity tests compare tuple SETS, not just lowered weights).
CAM_DROP = object()


@dataclass(frozen=True)
class CamDiscriminatorSlot:
    """One HETEROGENEOUS discriminator row of a binary-address CAM, as DATA.

    The L15 LI/LC head layers ~40 bespoke discriminator rows on top of the
    binary-address block: opcode gates (``OP_LI_RELAY``/``OP_LC_RELAY`` restore),
    marker blockers (``MARK_PC``/``MARK_SP`` suppress), the per-store address
    one-hots, the VALID/pop lifecycle rows, the sink bias. Each is a
    row-SELECTION structure — a declared ``(slot, Q-writes, K-writes)`` — not a
    free projection. Declaring them as DATA (a tuple of these) makes the whole
    head derivable without hand-authored per-cell ``W_q``/``W_k`` writes.

    A discriminator MAY also carry V/K value writes (the byte-select /
    per-head value-lane rows the L15 head interleaves into its slot range) via
    :attr:`v` / :attr:`o`; most carry only Q/K row-select.

    Each ``(dim, weight)`` name supports the ``BASE+offset`` token form (e.g.
    ``"CMP+3"``, ``"ADDR_B0_LO+8"``) resolved exactly like :func:`cam_lookup`.

    Attributes:
        slot: head-local slot the row lands on.
        q: ``(dim_name, weight)`` Q-side writes on ``slot``.
        k: ``(dim_name, weight)`` K-side writes on ``slot``.
        v: ``(dim_name, weight)`` V-side writes on ``slot`` (value-lane rows).
        o: ``(out_dim_name, weight)`` O-side writes FROM ``slot`` (value relay).
    """

    slot: int
    q: Tuple[Tuple[str, float], ...] = ()
    k: Tuple[Tuple[str, float], ...] = ()
    v: Tuple[Tuple[str, float], ...] = ()
    o: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CamBinaryAddressMatch:
    """A binary-addressed CAM head: per-bit address comparator + discriminators.

    The multi-slot generalization of :class:`CamLookupSpec` for heads whose row
    select is a BINARY per-bit address encoding (the L15 LI/LC value-load head,
    the L14 mem-generation store) rather than a single content-address key. The
    head is ONE structural CAM:

      * a :class:`CamBinaryAddressBlock` — the ``±scale`` per-bit comparator
        across ``len(nibble_bands) * width_bits`` slots (the binary address);
      * a tuple of :class:`CamDiscriminatorSlot` — the heterogeneous
        opcode/marker/lifecycle discriminator rows, declared as DATA;
      * a tuple of :class:`CamValueBand` — the matched row's value relay
        (``CLEAN_EMBED_{LO,HI} -> OUTPUT_{LO,HI}`` for L15).

    The builder merges every Q/K/V/O write into per-``(slot, dim)`` maps
    (last-write-wins) so the produced head is BYTE-IDENTICAL to a hand-authored
    head with the same final cell set — the lowerer is indexed assignment, so
    only the final set matters. The primitive is GENERAL: reuse it for any
    binary-addressed CAM.

    Attributes:
        name: head family name (rule-name / diagnostic prefix).
        address: the binary per-bit address comparator block.
        discriminators: the heterogeneous discriminator rows (DATA).
        value_bands: the matched row's value relay blocks.
        alibi_slope: per-head ALiBi slope. ``None`` => the op sets its own.
        value_active: when ``False`` the value relay is omitted (row-select
            gates + discriminators kept). Mirrors :attr:`CamLookupSpec.value_active`.
        direction: ``"load"`` (read-by-address, L15 LI/LC) or ``"store"`` (the
            emit direction, L14 mem-generation). A SEMANTIC tag on the same
            machinery (recorded so a generic engine knows the data flow); the
            builder LOWERS both — the store direction merely records that the
            head binds address→value at the marker.
        extra_reads: extra dim names to declare as reads beyond the auto set.
        step_window: the step-scope contract the verifier enforces.
        overlay: a SECOND tuple of :class:`CamDiscriminatorSlot`, merged
            last-write-wins AFTER the base discriminators AND after the value
            bands. Unlike :attr:`discriminators` the overlay MAY reuse a base
            slot (it is the DATA form of the campaign OVERRIDE layer that used to
            dict-merge cells over the fully-built base head), and a cell whose
            weight is :data:`CAM_DROP` REMOVES that ``(slot, dim)`` from the
            merged map (the DATA form of a ``q_map.pop`` / wiped V-O row). The
            overlay is the flag-conditioned-discriminator surface: an op appends
            only the cells whose gating flag is on, so flag-OFF (empty overlay)
            is byte-identical to the plain base CAM.
    """

    name: str
    address: CamBinaryAddressBlock
    discriminators: Tuple[CamDiscriminatorSlot, ...]
    value_bands: Tuple[CamValueBand, ...]
    alibi_slope: Optional[float] = None
    value_active: bool = True
    direction: str = "load"
    extra_reads: Tuple[str, ...] = ()
    step_window: StepWindowConstraint = StepWindowConstraint.ANY_STEP
    overlay: Tuple[CamDiscriminatorSlot, ...] = ()

    def __post_init__(self) -> None:
        if not self.address.nibble_bands and not self.discriminators:
            raise ValueError(
                f"CamBinaryAddressMatch({self.name!r}): must declare at least "
                "an address block or a discriminator slot"
            )
        if self.address.width_bits <= 0:
            raise ValueError(
                f"CamBinaryAddressMatch({self.name!r}): address width_bits must "
                f"be positive, got {self.address.width_bits}"
            )
        for vb in self.value_bands:
            if vb.width <= 0:
                raise ValueError(
                    f"CamBinaryAddressMatch({self.name!r}): value band "
                    f"{vb.source_band!r}->{vb.target_band!r} width must be "
                    f"positive, got {vb.width}"
                )
        if self.direction not in ("load", "store"):
            raise ValueError(
                f"CamBinaryAddressMatch({self.name!r}): direction must be "
                f"'load' or 'store', got {self.direction!r}"
            )
        # Guard against two discriminators claiming the same slot (a silent
        # last-write-wins collision the caller almost never intends). A
        # discriminator MAY intentionally re-write an ADDRESS-block cell (the
        # L15 local_slot_scale rescale of slots 4..11 is exactly that
        # last-write-wins override), so address-slot overlap is NOT an error —
        # only duplicate discriminator slots are.
        seen: Set[int] = set()
        for d in self.discriminators:
            if d.slot in seen:
                raise ValueError(
                    f"CamBinaryAddressMatch({self.name!r}): duplicate "
                    f"discriminator slot {d.slot}"
                )
            seen.add(d.slot)


@dataclass(frozen=True)
class CamBinaryAddressBundle:
    """The artifacts :func:`cam_binary_address_match` generates for one head.

    Attributes:
        spec: the originating :class:`CamBinaryAddressMatch`.
        head_spec_builder: ``(dim_positions, head_idx) ->
            DeclarativeAttentionHeadSpec``. Reproduces the hand-built binary-CAM
            head EXACTLY (same Q/K/V/O writes at the resolved dim positions).
        head_reads / head_writes: dim-name sets for the head's Operation.
    """

    spec: CamBinaryAddressMatch
    head_spec_builder: Callable[
        [Dict[str, int], int], DeclarativeAttentionHeadSpec
    ]
    head_reads: Set[str]
    head_writes: Set[str]


def cam_binary_address_match(
    spec: CamBinaryAddressMatch,
) -> CamBinaryAddressBundle:
    """Generate the binary-addressed CAM head for ``spec``.

    Returns a :class:`CamBinaryAddressBundle` whose ``head_spec_builder``
    reproduces the hand-authored binary-address CAM head byte-identically. Like
    :func:`cam_lookup` this registers NO residual band (a CAM head relays into
    EXISTING bands).
    """
    addr = spec.address

    def head_spec_builder(
        dim_positions: Dict[str, int], head_idx: int
    ) -> DeclarativeAttentionHeadSpec:
        def _P(name) -> int:
            # An overlay cell may carry an ALREADY-RESOLVED integer position
            # (the campaign OVERRIDE layer computes positions via a live
            # dim-proxy, e.g. ``BD.OP_JSR``); pass ints through unchanged. String
            # tokens ("BASE" / "BASE+offset") resolve through ``dim_positions``.
            if isinstance(name, int):
                return name
            return _resolve_dim_token(name, lambda n: int(dim_positions[n]))

        # Merge every write into per-(slot, dim) maps. The lowerer applies
        # indexed assignment (last-write-wins), so the produced weights depend
        # ONLY on the final cell set — an ordered emit list would be equally
        # byte-identical, but the map makes the last-write-wins semantics
        # STRUCTURAL (a discriminator can legitimately re-write an address-block
        # cell, e.g. the L15 local_slot_scale rescale of slots 4..11).
        q_map: Dict[Tuple[int, int], float] = {}
        k_map: Dict[Tuple[int, int], float] = {}
        v_map: Dict[Tuple[int, int], float] = {}
        o_map: Dict[Tuple[int, int], float] = {}  # keyed (out_dim, slot)

        # (1) The BINARY per-bit address comparator block. Each nibble band
        #     consumes ``width_bits`` consecutive slots; slot ``+bit`` scores
        #     ``±scale`` per cell by bit ``bit`` of the cell index.
        cells = addr.cells_per_band()
        slot = addr.slot_base
        for band in addr.nibble_bands:
            base = _P(band)
            for bit in range(addr.width_bits):
                for k in range(cells):
                    bit_val = 2 * ((k >> bit) & 1) - 1
                    w = addr.scale * bit_val
                    q_map[(slot, base + k)] = w
                    k_map[(slot, base + k)] = w
                slot += 1

        # (2) The heterogeneous discriminator rows (DATA). Merged AFTER the
        #     address block so a discriminator can legitimately re-write an
        #     address cell (the last-write-wins override the L15 head uses).
        for d in spec.discriminators:
            for (dim, w) in d.q:
                q_map[(d.slot, _P(dim))] = float(w)
            for (dim, w) in d.k:
                k_map[(d.slot, _P(dim))] = float(w)
            for (dim, w) in d.v:
                v_map[(d.slot, _P(dim))] = float(w)
            for (out_dim, w) in d.o:
                o_map[(_P(out_dim), d.slot)] = float(w)

        # (3) The value relay blocks: V copies source_band -> V slots; O writes
        #     those slots into target_band. Omitted when value_active is False.
        if spec.value_active:
            for vb in spec.value_bands:
                src = _P(vb.source_band)
                tgt = _P(vb.target_band)
                for j in range(vb.width):
                    v_map[(vb.v_slot_base + j, src + j)] = 1.0
                    o_map[(tgt + j, vb.v_slot_base + j)] = vb.o_scale

        # (4) The OVERLAY discriminators (the flag-conditioned OVERRIDE layer as
        #     DATA). Merged LAST — after the base discriminators and the value
        #     bands — so an overlay cell wins the last-write-wins over ANY base
        #     write (incl. a value-band O cell). A cell whose weight is CAM_DROP
        #     REMOVES that (slot, dim) from the merged map (the DATA form of the
        #     hand-authored q_map.pop / wiped V-O row). Overlay slots MAY reuse
        #     base slots by design (that IS the override), so no duplicate-slot
        #     guard applies here.
        def _apply(mp, key, w):
            if w is CAM_DROP:
                mp.pop(key, None)
            else:
                mp[key] = float(w)

        for d in spec.overlay:
            for (dim, w) in d.q:
                _apply(q_map, (d.slot, _P(dim)), w)
            for (dim, w) in d.k:
                _apply(k_map, (d.slot, _P(dim)), w)
            for (dim, w) in d.v:
                _apply(v_map, (d.slot, _P(dim)), w)
            for (out_dim, w) in d.o:
                _apply(o_map, (_P(out_dim), d.slot), w)

        q = tuple(AP(s, d, w) for (s, d), w in q_map.items())
        k = tuple(AP(s, d, w) for (s, d), w in k_map.items())
        v = tuple(AP(s, d, w) for (s, d), w in v_map.items())
        o = tuple(AO(od, s, w) for (od, s), w in o_map.items())
        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=q, k=k, v=v, o=o,
            alibi_slope=spec.alibi_slope,
            step_window=spec.step_window,
        )

    def _base(name):
        # Overlay cells may carry ALREADY-RESOLVED integer positions (no name to
        # recover); they contribute no NAMED read/write (the base discriminators,
        # address block, and value bands already declare the semantic bands).
        if isinstance(name, int):
            return None
        return name.split("+", 1)[0]

    head_reads: Set[str] = set()
    for band in addr.nibble_bands:
        head_reads.add(_base(band))
    for d in (*spec.discriminators, *spec.overlay):
        for (dim, _w) in d.q:
            head_reads.add(_base(dim))
        for (dim, _w) in d.k:
            head_reads.add(_base(dim))
        for (dim, _w) in d.v:
            head_reads.add(_base(dim))
    head_writes: Set[str] = set()
    for d in (*spec.discriminators, *spec.overlay):
        for (out_dim, _w) in d.o:
            head_writes.add(_base(out_dim))
    if spec.value_active:
        for vb in spec.value_bands:
            head_reads.add(_base(vb.source_band))
            head_writes.add(_base(vb.target_band))
    head_reads.discard(None)
    head_writes.discard(None)
    head_reads.update(spec.extra_reads)

    return CamBinaryAddressBundle(
        spec=spec,
        head_spec_builder=head_spec_builder,
        head_reads=head_reads,
        head_writes=head_writes,
    )


# ===========================================================================
# MARKER-BROADCAST — copy an opcode/flag band from the STEP's marker row to
# THIS op's own multi-byte byte-position rows, derived GENERICALLY
# ===========================================================================
#
# The LOWERING gap docs/DERIVE_DECODE_PILOT §G-IMM-RELAY names: every
# multi-byte opcode needs its per-step opcode marker (``OP_<NAME>``, set by
# decode at the step's MARK_<M> marker row) BROADCAST forward to the op's own
# byte-position rows so the later multi-byte routing / value-route FFN can gate
# on it there. IMM's L8 head-4 (``layer8_op_imm_relay``) is the canonical
# instance: it copies ``OP_IMM`` from the AX marker to the AX byte positions
# (``IS_BYTE`` + ``H1[AX_I]``). This is NOT a :func:`cam_lookup` — there is no
# content ADDRESS key; the head attends from a byte position to ITS OWN step's
# marker row (a fixed marker-bank slot), so the "row select" is a POSITIONAL
# marker match, not a value/address CAM. The relay is generic: the ONLY varying
# data is (which marker, which byte-position bank slot, which flag band, the
# ALiBi recency slope that pins the CURRENT step's marker).
#
# Like :func:`cam_lookup` this generator returns a pure builder bundle
# (``head_spec_builder(dim_positions, head_idx)``) the op factory installs; no
# compiler change — it lowers through ``Primitives.generate_attention_head``.
# The byte-identity proof re-expresses the SETTLED L8 head-4 OP_IMM relay and
# gates the whole-model state_dict hash unchanged.


@dataclass(frozen=True)
class MarkerBroadcastSpec:
    """Declarative description of a marker-broadcast (flag-relay) attention head.

    A marker-broadcast head fires at THIS op's multi-byte byte-position rows
    (``IS_BYTE`` AND the op's ``marker_bank`` slot in the H1 threshold bank),
    attends BACK to the step's own marker row (``source_marker``), and copies a
    flag band (``broadcast_bands`` — e.g. ``OP_IMM``) from that row onto its own
    byte positions. The row select is a POSITIONAL marker match (NOT a content
    address like :func:`cam_lookup`): the Q anchors on the fixed marker-bank
    slot, the K anchors on the source marker, and an ALiBi recency slope pins
    the CURRENT step's marker so multi-instance programs (``IMM;PSH;IMM``) do
    not dilute across prior marker rows.

    Every field is a VARYING parameter; the IDENTICAL head structure — the
    fire-site Q gate + the source-marker K select + the optional CONST-anchored
    confirm slot + the flag-band V/O broadcast — is supplied by
    :func:`marker_broadcast`.

    Attributes:
        name: head family name (rule-name / diagnostic prefix).
        marker_bank: the marker whose H1 threshold-bank slot the byte-position
            rows carry (``"AX"`` for the IMM relay). Resolved to a slot index
            by ``marker_bank_index`` at build time; the ``H1+<slot>`` cell is
            the fire-site anchor. NOTE: the resolved ``H1`` cell name is
            supplied by the caller via ``fire_slot_dim`` (the DSL stays free of
            the ``marker_bank_index`` import).
        fire_slot_dim: the resolved fire-site H1 cell dim name
            (``"H1+<AX_I>"``) — the byte-position marker-bank anchor. Supplied
            resolved so the DSL does not import the positional helper.
        source_marker: the marker dim the K side selects (the step's own marker
            row the flag was decoded onto; ``"MARK_AX"`` for IMM).
        broadcast_bands: the flag bands copied source-row -> byte positions.
            Each is a :class:`MarkerBroadcastBand`. For IMM: one 1-wide band
            ``OP_IMM -> OP_IMM``.
        weight: the shared Q/K fire-site + source-select projection weight
            (IMM: ``20.0``). The IS_BYTE reject / CONST sharpener weights are
            derived from it by the fixed multipliers below (matching the
            hand-built head exactly).
        is_byte_dim: the ``IS_BYTE`` predicate dim (fire-site + K reject).
        const_dim: the CONST residual dim (Q bias + K/confirm anchor).
        gate_slot: the CONST-anchored confirm/gate sub-head slot (IMM: ``1``).
            ``None`` => no confirm slot.
        gate_is_byte_weight / gate_const_q_weight / gate_const_k_weight: the
            confirm slot's Q ``IS_BYTE`` / Q ``CONST`` / K ``CONST`` writes
            (IMM: ``500.0`` / ``-500.0`` / ``5.0``).
        alibi_slope: per-head ALiBi recency slope pinning the CURRENT step's
            marker (IMM: ``0.5``). ``None`` => the op sets its own slope.
        query_slot: head-local slot the primary fire/select lands on (IMM: 0).
        step_window: the step-scope contract the verifier enforces (IMM:
            ``CURRENT_STEP_ONLY`` — the ALiBi slope keeps the relay mass on the
            current step's marker).
    """

    name: str
    fire_slot_dim: str
    source_marker: str
    broadcast_bands: Tuple["MarkerBroadcastBand", ...]
    weight: float
    marker_bank: str = "AX"
    is_byte_dim: str = "IS_BYTE"
    const_dim: str = "CONST"
    gate_slot: Optional[int] = 1
    gate_is_byte_weight: float = 500.0
    gate_const_q_weight: float = -500.0
    gate_const_k_weight: float = 5.0
    alibi_slope: Optional[float] = 0.5
    query_slot: int = 0
    step_window: StepWindowConstraint = StepWindowConstraint.CURRENT_STEP_ONLY

    def __post_init__(self) -> None:
        if not self.broadcast_bands:
            raise ValueError(
                f"MarkerBroadcastSpec({self.name!r}): broadcast_bands must be "
                "non-empty (a marker-broadcast relays at least one flag band)"
            )
        for bb in self.broadcast_bands:
            if bb.width <= 0:
                raise ValueError(
                    f"MarkerBroadcastSpec({self.name!r}): broadcast band "
                    f"{bb.source_band!r}->{bb.target_band!r} width must be "
                    f"positive, got {bb.width}"
                )


@dataclass(frozen=True)
class MarkerBroadcastBand:
    """One ``source_band -> target_band`` flag-broadcast block.

    V copies ``source_band[0..width-1]`` from the matched marker row into
    head-local V slots ``v_slot_base..v_slot_base+width-1``; O writes those
    slots into ``target_band[0..width-1]`` at ``o_scale``. For the IMM relay
    this is one 1-wide band: ``OP_IMM -> OP_IMM`` at ``v_slot_base=0``,
    ``o_scale=1.0`` (copy the flag onto its own byte positions).

    Attributes:
        source_band: the marker row's flag band the V slots read.
        target_band: the band the O writes the broadcast flag into (usually the
            SAME band — a flag broadcast to one's own byte positions).
        width: cell count (IMM: ``1`` — a single flag bit).
        v_slot_base: head-local V slot base (IMM: ``0``).
        o_scale: O-write magnitude (IMM: ``1.0``).
    """

    source_band: str
    target_band: str
    width: int = 1
    v_slot_base: int = 0
    o_scale: float = 1.0


@dataclass(frozen=True)
class MarkerBroadcastBundle:
    """The artifacts :func:`marker_broadcast` generates for one relay head.

    The builder is pure (no global side effects) so the op factory can call it
    at bake time and IR-factory time identically.

    Attributes:
        spec: the originating :class:`MarkerBroadcastSpec`.
        head_spec_builder: ``(dim_positions, head_idx) ->
            DeclarativeAttentionHeadSpec``. Reproduces the hand-built relay head
            EXACTLY (same Q/K/V/O writes at the resolved dim positions).
        head_reads / head_writes: dim-name sets for the head's Operation. The
            head READS the fire-site marker-bank slot + IS_BYTE + CONST + the
            source marker + the broadcast source bands; WRITES the broadcast
            target bands.
    """

    spec: MarkerBroadcastSpec
    head_spec_builder: Callable[
        [Dict[str, int], int], DeclarativeAttentionHeadSpec
    ]
    head_reads: Set[str]
    head_writes: Set[str]


def marker_broadcast(spec: MarkerBroadcastSpec) -> MarkerBroadcastBundle:
    """Generate the marker-broadcast (flag-relay) attention head for ``spec``.

    Returns a :class:`MarkerBroadcastBundle` whose ``head_spec_builder``
    reproduces the hand-built relay head byte-identically. Registers NO
    residual band (the relay copies into an EXISTING opcode/flag band, so there
    is no import-time side effect).

    The head structure is fixed; the varying data is the spec fields:

      * fire-site Q (slot ``query_slot``): ``IS_BYTE`` @ ``weight``,
        ``fire_slot_dim`` @ ``weight``, ``CONST`` @ ``-weight*1.5`` — fire at
        THIS op's byte positions.
      * source-select K (slot ``query_slot``): ``source_marker`` @ ``weight``,
        ``IS_BYTE`` @ ``-weight*10``, ``CONST`` @ ``weight*0.5`` — attend BACK
        to the step's own marker row (reject byte positions on the K side).
      * optional confirm slot (``gate_slot``): Q ``IS_BYTE`` @
        ``gate_is_byte_weight`` + ``CONST`` @ ``gate_const_q_weight``; K
        ``CONST`` @ ``gate_const_k_weight`` — pins the softmax winner.
      * broadcast V/O: each :class:`MarkerBroadcastBand` copies its flag band
        from the marker row onto the byte positions.
    """
    w = spec.weight

    def head_spec_builder(
        dim_positions: Dict[str, int], head_idx: int
    ) -> DeclarativeAttentionHeadSpec:
        def _P(name: str) -> int:
            return int(dim_positions[name])

        qs = spec.query_slot

        # (1) Fire-site Q: fire at THIS op's byte positions (IS_BYTE + the
        #     op's marker-bank slot), biased against the CONST anchor.
        q: list = [
            AP(qs, _P(spec.is_byte_dim), w),
            AP(qs, _P(spec.fire_slot_dim), w),
            AP(qs, _P(spec.const_dim), -w * 1.5),
        ]
        # (2) Source-select K: attend BACK to the step's own marker row; the
        #     IS_BYTE reject keeps the K mass off the byte positions.
        k: list = [
            AP(qs, _P(spec.source_marker), w),
            AP(qs, _P(spec.is_byte_dim), -w * 10),
            AP(qs, _P(spec.const_dim), w * 0.5),
        ]

        # (3) Optional CONST-anchored confirm/gate slot (sharpen the winner).
        if spec.gate_slot is not None:
            g = spec.gate_slot
            q.append(AP(g, _P(spec.is_byte_dim), spec.gate_is_byte_weight))
            q.append(AP(g, _P(spec.const_dim), spec.gate_const_q_weight))
            k.append(AP(g, _P(spec.const_dim), spec.gate_const_k_weight))

        # (4) Flag-band broadcast: V copies source_band -> V slots; O writes
        #     those slots into target_band (usually the SAME band).
        v: list = []
        o: list = []
        for bb in spec.broadcast_bands:
            src = _P(bb.source_band)
            tgt = _P(bb.target_band)
            for j in range(bb.width):
                v.append(AP(bb.v_slot_base + j, src + j, 1.0))
                o.append(AO(tgt + j, bb.v_slot_base + j, bb.o_scale))

        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            alibi_slope=spec.alibi_slope,
            step_window=spec.step_window,
        )

    # Operation dep-graph dim sets (structural derivation).
    head_reads: Set[str] = {
        spec.is_byte_dim, spec.fire_slot_dim, spec.const_dim,
        spec.source_marker,
    }
    head_writes: Set[str] = set()
    for bb in spec.broadcast_bands:
        head_reads.add(bb.source_band)
        head_writes.add(bb.target_band)

    return MarkerBroadcastBundle(
        spec=spec,
        head_spec_builder=head_spec_builder,
        head_reads=head_reads,
        head_writes=head_writes,
    )


# ===========================================================================
# VALUE-ROUTE — gated per-nibble VALUE copy (fetched imm -> OUTPUT), derived
# ===========================================================================
#
# The IMM value-route (docs/DERIVE_DECODE_PILOT §G-IMM-ROUTE): once decode set
# ``OP_IMM`` and :func:`marker_broadcast` relayed it to the AX byte positions,
# the fetched immediate (staged in ``AX_CARRY_LO/HI`` per nibble) is copied to
# ``OUTPUT`` at those byte positions. It is a clean gated per-cell copy block:
# a shared AND-context (fire at THIS op's byte positions when the opcode flag is
# relayed there) times a per-cell GATE on the source nibble writing the
# corresponding target nibble. The ONLY varying data is (the context, the
# source/target band pair, the nibble count, the write scale). :func:`value_route`
# lowers the whole block from a :class:`ValueRouteSpec` with ZERO hand-authored
# per-cell branch, byte-identical to ``_layer8_multibyte_routing_rules``.


@dataclass(frozen=True)
class ValueRouteChannel:
    """One ``source_band -> target_band`` per-nibble gated-copy channel.

    For each ``k in range(width)`` the channel emits ONE ``multi_way_and_rule``
    gated on ``source_band+k`` that writes ``target_band+k`` at ``write_scale``,
    under the route's shared AND-context. The IMM route has TWO channels:
    ``AX_CARRY_LO -> OUTPUT_LO`` (16 low nibbles) and ``AX_CARRY_HI ->
    OUTPUT_HI_THIS_STEP`` (16 high nibbles).

    Attributes:
        source_band: the per-nibble source band the GATE reads.
        target_band: the band each cell WRITES.
        width: nibble-cell count (IMM: ``16`` per channel).
        write_scale: the per-cell write magnitude (IMM: ``8.0 / S``).
        rule_name_fn: ``(k) -> str`` legacy rule name for cell ``k``.
    """

    source_band: str
    target_band: str
    write_scale: float
    width: int = 16
    rule_name_fn: Optional[Callable[[int], str]] = None


@dataclass(frozen=True)
class ValueRouteSpec:
    """Declarative description of a gated per-nibble value-route FFN block.

    A value route copies a fetched multi-nibble VALUE (staged per nibble in a
    source band) into a target OUTPUT band at THIS op's byte positions, gated on
    the relayed opcode flag. Every field is a per-route CONSTANT; the per-cell
    copy structure is supplied by :func:`value_route` with zero per-cell branch.

    Attributes:
        name: route family name (diagnostic).
        conditions: the shared AND-context every cell folds in beyond the
            per-cell gate — fire at THIS op's byte positions when the opcode
            flag is relayed there (IMM: ``(("IS_BYTE",1), ("H1+AX_I",1),
            ("OP_IMM",1), ("MARK_AX",-4))``). Supplied resolved (the
            ``H1+<slot>`` cell already resolved by the caller).
        threshold: the AND threshold (IMM: ``6.5``).
        channels: the per-nibble copy channels (:class:`ValueRouteChannel`).
    """

    name: str
    conditions: Tuple[Tuple[str, float], ...]
    threshold: float
    channels: Tuple[ValueRouteChannel, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError(
                f"ValueRouteSpec({self.name!r}): channels must be non-empty"
            )


@dataclass(frozen=True)
class ValueRouteBundle:
    """The artifacts :func:`value_route` generates for one value-route block.

    Attributes:
        spec: the originating :class:`ValueRouteSpec`.
        rules_builder: ``() -> tuple[FFNRule, ...]`` — the full ordered
            per-cell copy rule sequence, DERIVED from the channels with zero
            per-cell branch.
        reads / writes: the block's Operation dep-graph dim sets, derived
            structurally from the conditions + channels.
    """

    spec: ValueRouteSpec
    rules_builder: Callable[[], Tuple[FFNRule, ...]]
    reads: Set[str]
    writes: Set[str]


def value_route(spec: ValueRouteSpec) -> ValueRouteBundle:
    """Generate the gated per-nibble value-route FFN rule sequence for ``spec``.

    Reproduces the hand-authored per-cell copy block byte-for-byte (proof:
    ``tools/_isa_golden_hash.py`` unchanged under the flag-ON derived build).
    Registers no residual band (the route writes into an existing OUTPUT band).
    """

    def rules_builder() -> Tuple[FFNRule, ...]:
        rules: list[FFNRule] = []
        for ch in spec.channels:
            for k in range(ch.width):
                name = (
                    ch.rule_name_fn(k) if ch.rule_name_fn is not None
                    else f"{spec.name}_{ch.target_band.lower()}_{k}"
                )
                rules.append(multi_way_and_rule(
                    name=name,
                    conditions=spec.conditions,
                    threshold=spec.threshold,
                    gate=f"{ch.source_band}+{k}",
                    writes=((f"{ch.target_band}+{k}", ch.write_scale),),
                ))
        return tuple(rules)

    reads: Set[str] = set()
    for (dim, _w) in spec.conditions:
        reads.add(dim.split("+", 1)[0])
    writes: Set[str] = set()
    for ch in spec.channels:
        reads.add(ch.source_band)
        writes.add(ch.target_band)

    return ValueRouteBundle(
        spec=spec,
        rules_builder=rules_builder,
        reads=reads,
        writes=writes,
    )


# ===========================================================================
# SCALAR-RELAY — the marker-anchored opcode/flag relay bank, derived GENERICALLY
# ===========================================================================
#
# The L7 "memory heads" 5/6/7 are NOT :func:`cam_lookup` value relays (no
# content-address key + contiguous value band) nor :func:`marker_broadcast`
# byte-position relays (they fire AT the marker row, not at byte positions
# attending back). They are a THIRD, distinct ISA-semantic family: a
# **marker-anchored scalar-relay bank**. The head fires at a MARKER row (its Q
# signature is a fixed set of marker + threshold-bank discriminators), its K
# selects the SAME (or a sibling) marker row (a self / marker match, optionally
# doubled or with an opcode blocker), and it relays a BANK of INDIVIDUAL scalar
# flags from that row: each relay reads one-or-more source dims into ONE V slot
# and writes ONE target dim at a per-relay scale. This is exactly the L7 head-5
# opcode-flag relay (OP_LI/LC/LEA/AND/OR/XOR/JSR/SHR/SI/SC/ADD/SUB/ENT ->
# RELAY/CMP/TEMP/OP dims), head-6 PSH/CMP relay (CMP/PSH_AT_SP self-relay off
# the STACK0|SP marker) and head-7 MEM flag broadcast (MEM_STORE/MEM_ADDR_SRC/
# OP_JSR/OP_ENT self-relay off the MEM marker).
#
# The value flow a :class:`CamValueBand` cannot express and this family needs:
#   * MULTIPLE source dims summed into ONE V slot (head-5 slot 4 reads
#     OP_AND+OP_OR+OP_XOR — an OR-of-bitwise-opcode gate),
#   * a per-relay O WRITE SCALE (head-5 relays the JSR/SI/SC/ENT flags at 5.0,
#     the rest at 1.0),
#   * a raw multi-dim Q/K signature (marker + H1/H3/H4 threshold-bank
#     discriminators, a two-marker OR key, a K-side opcode blocker, a doubled
#     K weight) — pure row-selection structure, not a content address.
#
# Like the sibling generators this returns a pure builder bundle
# (``head_spec_builder(dim_positions, head_idx)``) the op factory installs; no
# compiler change — it lowers through ``Primitives.generate_attention_head``.
# The byte-identity proof re-expresses the SETTLED L7 memory heads 5/6/7 and
# gates the whole-model state_dict hash unchanged.


@dataclass(frozen=True)
class ScalarRelay:
    """One scalar (width-1) flag relay in a :class:`ScalarRelayBankSpec`.

    Reads ``sources`` (one-or-more ``(dim, weight)`` V-projection writes, all
    landing on the SAME head-local ``v_slot`` so their contributions SUM) from
    the attended marker row, and writes the summed value into ``target_dim`` via
    a single O write at ``o_scale``. A single-source relay is the common case
    (``sources == ((dim, 0.2),)``); the multi-source case is head-5 slot 4's
    ``OP_AND/OP_OR/OP_XOR`` OR-gate.

    Attributes:
        v_slot: head-local V/O slot the relay lands on.
        sources: the ``(source_dim, v_weight)`` reads folded into ``v_slot``.
        target_dim: the residual dim the O write targets (a ``BASE`` or
            ``BASE+offset`` token; the offset is honoured via
            :func:`_resolve_dim_token`).
        o_scale: the O-write magnitude.
    """

    v_slot: int
    sources: Tuple[Tuple[str, float], ...]
    target_dim: str
    o_scale: float = 1.0


@dataclass(frozen=True)
class ScalarRelayBankSpec:
    """Declarative description of a marker-anchored scalar-relay attention head.

    The head fires at a marker row (its ``query_sig`` is the marker + the
    threshold-bank discriminators that pin WHICH marker rows fire), its K
    selects the same/sibling marker row (``key_sig`` — a self/marker match,
    optionally doubled or carrying an opcode blocker), and it relays a BANK of
    individual scalar flags (``relays``) from that row. Every field is a
    VARYING parameter; the IDENTICAL head structure (the raw Q/K signature on
    the query slot + the per-relay V/O scalar copies) is supplied by
    :func:`scalar_relay`.

    Attributes:
        name: head family name (rule-name / diagnostic prefix).
        query_sig: the Q-projection ``(dim, weight)`` writes on ``query_slot``
            — the marker fire-site + threshold-bank discriminators (row select).
            Each ``dim`` is a ``BASE`` or ``BASE+offset`` token.
        key_sig: the K-projection ``(dim, weight)`` writes on ``query_slot`` —
            the marker self/sibling match + any doubled weight or opcode blocker.
        relays: the scalar flag relays (:class:`ScalarRelay`).
        query_slot: the head-local slot the Q/K signature lands on (L7: ``0``).
        alibi_slope: per-head ALiBi slope. ``None`` => the op writes its own
            ``alibi_slopes`` (the L7 memory heads' slopes are set by the op's
            bake, not the spec, so the re-expression leaves this ``None``).
        step_window: the step-scope contract the verifier enforces. The L7
            head-7 MEM flag broadcast reads the MEM marker across steps
            (memory persistence is by design) and declares ``ANY_STEP``; the
            others default to ``CURRENT_STEP_ONLY``.
    """

    name: str
    query_sig: Tuple[Tuple[str, float], ...]
    key_sig: Tuple[Tuple[str, float], ...]
    relays: Tuple[ScalarRelay, ...]
    query_slot: int = 0
    alibi_slope: Optional[float] = None
    step_window: StepWindowConstraint = StepWindowConstraint.CURRENT_STEP_ONLY

    def __post_init__(self) -> None:
        if not self.query_sig:
            raise ValueError(
                f"ScalarRelayBankSpec({self.name!r}): query_sig must be "
                "non-empty (a relay head fires at a marker row)"
            )
        if not self.key_sig:
            raise ValueError(
                f"ScalarRelayBankSpec({self.name!r}): key_sig must be "
                "non-empty (a relay head selects a marker row)"
            )
        if not self.relays:
            raise ValueError(
                f"ScalarRelayBankSpec({self.name!r}): relays must be non-empty"
            )
        for r in self.relays:
            if not r.sources:
                raise ValueError(
                    f"ScalarRelayBankSpec({self.name!r}): relay -> "
                    f"{r.target_dim!r} must read at least one source"
                )


@dataclass(frozen=True)
class ScalarRelayBundle:
    """The artifacts :func:`scalar_relay` generates for one relay-bank head.

    Attributes:
        spec: the originating :class:`ScalarRelayBankSpec`.
        head_spec_builder: ``(dim_positions, head_idx) ->
            DeclarativeAttentionHeadSpec``. Reproduces the hand-built relay-bank
            head EXACTLY (same Q/K/V/O writes at the resolved dim positions).
        head_reads / head_writes: dim-name sets for the head's Operation. The
            head READS the query/key signature dims + every relay source; WRITES
            every relay target.
    """

    spec: ScalarRelayBankSpec
    head_spec_builder: Callable[
        [Dict[str, int], int], DeclarativeAttentionHeadSpec
    ]
    head_reads: Set[str]
    head_writes: Set[str]


def scalar_relay(spec: ScalarRelayBankSpec) -> ScalarRelayBundle:
    """Generate the marker-anchored scalar-relay bank head for ``spec``.

    Returns a :class:`ScalarRelayBundle` whose ``head_spec_builder`` reproduces
    the hand-built relay head byte-identically. Registers NO residual band (the
    relays copy into EXISTING opcode/flag/CMP/TEMP bands, so there is no
    import-time side effect).
    """

    def head_spec_builder(
        dim_positions: Dict[str, int], head_idx: int
    ) -> DeclarativeAttentionHeadSpec:
        def _P(name: str) -> int:
            return _resolve_dim_token(name, lambda n: int(dim_positions[n]))

        qs = spec.query_slot
        q = [AP(qs, _P(d), w) for (d, w) in spec.query_sig]
        k = [AP(qs, _P(d), w) for (d, w) in spec.key_sig]

        v: list = []
        o: list = []
        for r in spec.relays:
            for (src, w) in r.sources:
                v.append(AP(r.v_slot, _P(src), w))
            o.append(AO(_P(r.target_dim), r.v_slot, r.o_scale))

        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            alibi_slope=spec.alibi_slope,
            step_window=spec.step_window,
        )

    def _base(name: str) -> str:
        return name.split("+", 1)[0]

    head_reads: Set[str] = set()
    for (d, _w) in spec.query_sig:
        head_reads.add(_base(d))
    for (d, _w) in spec.key_sig:
        head_reads.add(_base(d))
    head_writes: Set[str] = set()
    for r in spec.relays:
        for (src, _w) in r.sources:
            head_reads.add(_base(src))
        head_writes.add(_base(r.target_dim))

    return ScalarRelayBundle(
        spec=spec,
        head_spec_builder=head_spec_builder,
        head_reads=head_reads,
        head_writes=head_writes,
    )


# ===========================================================================
# DECODE — the ISA opcode table -> OP_<NAME> marker, derived GENERICALLY
# ===========================================================================
#
# The DECODE family (docs/semantic_spec_DECODE.md) is the decisive
# "opcode is a pure lookup" result: the L5 opcode-decode FFN sets each
# ``OP_<NAME>`` marker by a two-nibble one-hot AND on the fetched opcode byte,
# and EVERY per-opcode field except ``(opcode_byte, NAME)`` is a per-CONTEXT
# constant. :func:`decode_band` DERIVES the entire ordered decode-FFN rule
# tuple from:
#
#   * the ISA ``Opcode`` table (the sole source of ``(byte, NAME)``), and
#   * a small tuple of :class:`DecodeContext` band descriptors (the 3 marker
#     contexts + the scratch-clear band + the reserved blank), each carrying
#     the CONSTANT gate/threshold/HAS_SE/subset data the docs' gap-list G2/G7
#     say the engine must be TOLD (it cannot infer the first-step-vs-all-step
#     asymmetry from the ISA table alone).
#
# There is ZERO per-opcode branch in the lowering: ``lo = byte & 0xF`` and
# ``hi = (byte >> 4) & 0xF`` are computed from the scalar opcode value, and the
# single genuine per-(opcode, context) quirk — JSR writes the ``TEMP+0`` IS_JSR
# scratch flag instead of ``OP_JSR`` at the PC contexts (gap-list G3) — is
# expressed as ONE optional ``write_override`` DATA cell on the context, not a
# code path. This is the STEP-3 generic engine + STEP-4 100%-derivation proof
# for DECODE: :func:`decode_band` reproduces ``_opcode_decode_ffn_rules``
# byte-for-byte (flag-gated, golden-hash-held).


@dataclass(frozen=True)
class DecodeContext:
    """One marker-context band of the opcode-decode FFN (a docs G2 band).

    A context re-decodes a SUBSET of the ISA opcodes at ONE marker context
    (AX, first-step-PC, all-step-PC, ...). Every field is a per-context
    CONSTANT — the varying per-row data (``byte``, ``NAME``) comes from the
    ISA table alone. The lowering loops over ``opcodes`` (or the full table
    when ``opcodes is None``) and emits one ``multi_way_and_rule`` per opcode
    with ``lo = byte & 0xF`` / ``hi = (byte >> 4) & 0xF`` — no per-opcode
    branch.

    Attributes:
        name: context tag (diagnostic / rule-name prefix component).
        rule_name_fn: ``(op_name, out_dim) -> str`` producing the legacy
            rule name for a row (``op_name`` is the full ``"OP_<NAME>"`` slot
            string; kept identical for a clean diff — the lowered WEIGHTS do
            not depend on the name).
        extra_conditions: the CONSTANT AND conditions folded into EVERY row
            beyond the two opcode nibbles (e.g. ``(("MARK_PC", 1.0),
            ("HAS_SE", -1.0))`` for first-step-PC). ``()`` for the pure
            AX-marker main band.
        threshold: the AND threshold (docs G7: 1.5 for the 2-cond AX band,
            2.5 for the 3/4-cond PC bands).
        gate: optional multiplicative gate dim (the main band's
            ``dim_ref("marker","AX")``). ``None`` for the PC bands (they fold
            ``MARK_PC`` into ``extra_conditions`` instead).
        write_scale: the marker write magnitude (docs G7: ``10.0 / S``,
            supplied by the caller already divided).
        opcodes: the opcode-value SUBSET this context re-decodes, in emit
            order. ``None`` => the FULL ISA table (the main-at-AX band).
        write_override: optional ``{opcode_value: out_dim_name}`` — the ONE
            per-(opcode, context) quirk (JSR @ PC -> ``TEMP+0``, docs G3).
            An opcode absent from the map writes its uniform
            ``dim_ref("opcode_flag", NAME)`` marker.
        enabled: zero-arg predicate gating the WHOLE context (the Root-B
            all-step-JSR band is flag-gated ``C4_NESTED_JSR_PC_FIX``). ``None``
            => always present.
    """

    name: str
    rule_name_fn: Callable[[str, str], str]
    threshold: float
    write_scale: float
    extra_conditions: Tuple[Tuple[str, float], ...] = ()
    gate: Optional[str] = None
    opcodes: Optional[Tuple[int, ...]] = None
    write_override: Optional[Dict[int, str]] = None
    enabled: Optional[Callable[[], bool]] = None


@dataclass(frozen=True)
class ScratchClearBand:
    """The TEMP[1..31]-clear hygiene band (docs G4 — NOT decode).

    A separate generic primitive: per scratch slot ``k in range(lo..hi)`` emit
    a ``multi_way_and_rule`` gated on ``{slot_band}+{k}`` (weight -1) that
    clears the slot at ``marker_cond``. Opcode-independent — it shares the
    decode FFN block (unit-order load-bearing, docs G5) but carries no opcode
    data. Reserved slot 0 (JSR's IS_JSR flag) is skipped by starting at
    ``lo``.
    """

    name: str
    slot_band: str
    lo: int
    hi: int
    marker_cond: Tuple[str, float]
    threshold: float
    write_scale: float
    gate_weight: float = -1.0
    rule_name_fn: Optional[Callable[[int], str]] = None


@dataclass(frozen=True)
class BlankUnit:
    """A reserved-blank hidden unit (docs G5 unit-ordering constraint).

    Emits ONE no-op ``FFNRule`` (all-zero lowering) so the reserved unit-52
    JSR ``TEMP[0]`` slot keeps the legacy numbering. Not decode; a layout
    placeholder the engine threads through the ordered rule tuple.
    """

    name: str


@dataclass(frozen=True)
class DecodeSpec:
    """Declarative description of a full opcode-decode FFN block.

    The ordered ``bands`` tuple reproduces the exact unit layout
    (``_FETCH_FFN_UNIT_LAYOUT``): main-at-AX, first-step-PC, the reserved
    blank, the scratch-clear band, all-step-PC, and the flag-gated
    all-step-JSR. :func:`decode_band` lowers the whole tuple to a single
    ordered ``FFNRule`` sequence with ZERO per-opcode branch.

    Attributes:
        name: decode-block family name (diagnostic).
        opcode_table: the ISA table as ``((opcode_value, NAME), ...)`` in the
            main-band emit order. The SOLE source of ``(byte, NAME)``; every
            context's rows derive their nibbles from these values.
        bands: the ordered tuple of context / scratch / blank bands.
        nibble_lo_band / nibble_hi_band: the fetched-opcode nibble one-hot
            band names the AND conditions read (``OPCODE_BYTE_LO`` /
            ``OPCODE_BYTE_HI``).
        opcode_flag_ref: ``NAME -> out_dim_str`` — the ``(opcode_flag, NAME)``
            semantic-role resolver (``dim_ref("opcode_flag", NAME)``), injected
            so the DSL stays free of the ``dim_registry`` import.
    """

    name: str
    opcode_table: Tuple[Tuple[int, str], ...]
    bands: Tuple[object, ...]
    opcode_flag_ref: Callable[[str], str]
    nibble_lo_band: str = "OPCODE_BYTE_LO"
    nibble_hi_band: str = "OPCODE_BYTE_HI"


@dataclass(frozen=True)
class DecodeBundle:
    """The artifacts :func:`decode_band` generates for one decode FFN.

    Attributes:
        spec: the originating :class:`DecodeSpec`.
        rules_builder: ``() -> tuple[FFNRule, ...]`` — the full ordered
            decode-FFN rule sequence, DERIVED from the ISA table + band data
            with zero per-opcode branch. Re-evaluates ``DecodeContext.enabled``
            each call so a flag-gated band (all-step-JSR) tracks its flag.
        reads / writes: the decode FFN's Operation dep-graph dim sets, derived
            structurally from the bands.
    """

    spec: DecodeSpec
    rules_builder: Callable[[], Tuple[FFNRule, ...]]
    reads: Set[str]
    writes: Set[str]


def _decode_context_rules(
    ctx: "DecodeContext",
    spec: DecodeSpec,
) -> Tuple[FFNRule, ...]:
    """Lower ONE :class:`DecodeContext` — the generic per-opcode loop.

    For each opcode value in the context's subset (or the full table): compute
    ``lo = byte & 0xF`` / ``hi = (byte >> 4) & 0xF``, AND the two nibble
    one-hots with the context's constant ``extra_conditions``, and write the
    opcode's marker (or the per-(opcode, context) ``write_override``). This is
    the ONLY place opcode rows are produced, and it has NO per-opcode branch.
    """
    if ctx.enabled is not None and not ctx.enabled():
        return ()
    name_by_val = dict(spec.opcode_table)
    if ctx.opcodes is None:
        vals = tuple(v for (v, _n) in spec.opcode_table)
    else:
        vals = ctx.opcodes
    override = ctx.write_override or {}
    rules: list[FFNRule] = []
    for op_val in vals:
        lo = op_val & 0xF
        hi = (op_val >> 4) & 0xF
        name = name_by_val[op_val]
        out_dim = override.get(op_val, spec.opcode_flag_ref(name[3:]))
        conditions = (
            (f"{spec.nibble_lo_band}+{lo}", 1.0),
            (f"{spec.nibble_hi_band}+{hi}", 1.0),
        ) + tuple(ctx.extra_conditions)
        rules.append(multi_way_and_rule(
            name=ctx.rule_name_fn(name, out_dim),
            conditions=conditions,
            threshold=ctx.threshold,
            gate=ctx.gate,
            writes=((out_dim, ctx.write_scale),),
        ))
    return tuple(rules)


def _scratch_clear_rules(band: "ScratchClearBand") -> Tuple[FFNRule, ...]:
    """Lower the TEMP-clear hygiene band (docs G4) — opcode-independent."""
    rules: list[FFNRule] = []
    for k in range(band.lo, band.hi + 1):
        name = (
            band.rule_name_fn(k) if band.rule_name_fn is not None
            else f"{band.name}_{k}"
        )
        rules.append(multi_way_and_rule(
            name=name,
            conditions=(band.marker_cond,),
            threshold=band.threshold,
            gate=f"{band.slot_band}+{k}",
            gate_weight=band.gate_weight,
            writes=((f"{band.slot_band}+{k}", band.write_scale),),
        ))
    return tuple(rules)


def decode_band(spec: DecodeSpec) -> DecodeBundle:
    """Generate the full opcode-decode FFN rule sequence for ``spec``.

    The ``rules_builder`` derives every OP-marker rule from the ISA opcode
    table (``lo = byte & 0xF``, ``hi = (byte >> 4) & 0xF``) plus the constant
    per-context band data — ZERO per-opcode branches. Reproduces the
    hand-authored ``_opcode_decode_ffn_rules`` byte-for-byte when fed the
    matching ``DecodeSpec`` (proof: ``tools/_isa_golden_hash.py`` unchanged
    under the flag-ON derived build). Registers no residual band (decode
    writes into existing OP_* / TEMP dims).
    """

    def rules_builder() -> Tuple[FFNRule, ...]:
        out: list[FFNRule] = []
        for band in spec.bands:
            if isinstance(band, DecodeContext):
                out.extend(_decode_context_rules(band, spec))
            elif isinstance(band, ScratchClearBand):
                out.extend(_scratch_clear_rules(band))
            elif isinstance(band, BlankUnit):
                out.append(FFNRule(
                    conditions=(),
                    threshold=0.0,
                    writes=(),
                    gate=None,
                    gate_bias=0.0,
                    name=band.name,
                ))
            else:  # pragma: no cover - guarded by the spec authoring
                raise TypeError(
                    f"decode_band({spec.name!r}): unknown band type "
                    f"{type(band).__name__}"
                )
        return tuple(out)

    # Structural dep-graph derivation: reads = the nibble bands + every
    # context's extra-condition dims + gate + scratch marker/slot bands;
    # writes = every marker the contexts emit + the scratch slots.
    reads: Set[str] = set()
    writes: Set[str] = set()
    name_by_val = dict(spec.opcode_table)
    for band in spec.bands:
        if isinstance(band, DecodeContext):
            reads.add(spec.nibble_lo_band)
            reads.add(spec.nibble_hi_band)
            for (dim, _w) in band.extra_conditions:
                reads.add(dim.split("+", 1)[0])
            if band.gate is not None:
                reads.add(band.gate.split("+", 1)[0])
            vals = (
                tuple(v for (v, _n) in spec.opcode_table)
                if band.opcodes is None else band.opcodes
            )
            override = band.write_override or {}
            for op_val in vals:
                out_dim = override.get(
                    op_val, spec.opcode_flag_ref(name_by_val[op_val][3:])
                )
                writes.add(out_dim.split("+", 1)[0])
        elif isinstance(band, ScratchClearBand):
            reads.add(band.marker_cond[0].split("+", 1)[0])
            reads.add(band.slot_band)
            writes.add(band.slot_band)

    return DecodeBundle(
        spec=spec,
        rules_builder=rules_builder,
        reads=reads,
        writes=writes,
    )


# ===========================================================================
# PC-MUX — the CONTROL-family PC-source mux, derived GENERICALLY
# ===========================================================================
#
# ``docs/semantic_spec_CONTROL.md`` §2/§G1/§G2 shows the whole PC-next machine
# is a **source mux**: L3 writes the SEQUENTIAL default (``PC + INSTR_WIDTH``)
# unconditionally; each branch op OVERRIDES it with a uniform
# **cancel-then-write** idiom — 16 ``-OUTPUT_LO[k]`` cancel units + 16
# ``-OUTPUT_HI[k]`` cancel units (subtract the sequential default), then a
# 16-per-band target ENCODER that writes ``PC = <source>``. The six live
# hand-authored override builders (``_layer6_all_step_jmp_pc_override_rules``,
# ``_layer6_first_step_jmp``, ``_layer6_delayed_jmp``,
# ``_layer6_all_step_jsr``, ``_post_l9_bz``, ``_post_l9_bnz``) differ ONLY in:
#
#   * the GATE (opcode + condition + step guard) — a ``conditions`` tuple +
#     ``threshold``;
#   * the per-band cancel gate source (``OUTPUT_LO.*.-1`` cross-step vs the
#     same-step ``OUTPUT_HI_THIS_STEP``) + an optional extra cross-step gate
#     term (the BZ ``BZ_TARGET_FRESH`` re-fire suppressor, §G8);
#   * the target ENCODER:
#       - ``DIRECT_COPY``  — copy an already-encoded PC byte from a source band
#         (JMP first-step / delayed / all-step: ``AX_CARRY_*`` or ``FETCH_*``);
#       - ``IMM_TO_BYTE_ADDR`` — convert a raw instruction-INDEX immediate into
#         the encoded byte address ``imm*INSTR_WIDTH + PC_OFFSET`` (§G2's
#         ``idx_to_pc`` encoder; BZ/BNZ/JSR read the index from ``FETCH_LO``);
#   * an optional byte-0 HIGH-nibble ``+INSTR_WIDTH`` correction for an ODD
#     index-hi nibble (the ``imm*8`` shift's ``hi&1`` carry — two authoring
#     flavors, JSR ``gate_terms`` vs BZ/BNZ ``conditions``, both derived here);
#   * an optional cross-step "target fresh" writer bit (BZ, §G8).
#
# ``pc_mux(PcMuxSpec)`` takes that per-op DATA and emits the IDENTICAL
# ``FFNRule`` tuple the hand builders produce. It adds NO compiler machinery —
# it lowers through the same ``multi_way_and_rule`` / ``FFNRule`` shapes, so the
# compiler sees an ordinary FFN op. The byte-identity proof re-expresses all six
# live builders and gates the whole-model ``state_dict`` hash unchanged
# (``tools/_isa_golden_hash.py`` == ``91f55411``). It is the CONTROL analogue of
# :func:`decode_band` (the DECODE-family generic lowering).


@dataclass(frozen=True)
class PcMuxCancelBand:
    """One PC-override cancel band (``lo`` or ``hi``).

    Emits 16 ``-<output_base>[k]`` units gated by ``<output_gate>+k`` with
    ``gate_weight=-1`` — subtracting the SEQUENTIAL default (L3's ``PC+8``)
    from the OUTPUT bank before the encoder writes the branch target.

    Args:
        band: the band tag used in the rule name (``"lo"`` / ``"hi"``).
        output_base: the OUTPUT dim the cancel WRITES into (with the
            per-step ``write_scale``). ``OUTPUT_LO`` for the low band,
            ``OUTPUT_HI_THIS_STEP`` for the high band.
        output_gate: the OUTPUT dim the cancel READS as its ``-1`` gate.
            Cross-step ``OUTPUT_LO.*.-1`` for the LO band (subtract the
            PREVIOUS step's residual), same-step ``OUTPUT_HI_THIS_STEP`` for
            the HI band (already step-local).
        extra_gate_terms: additional multiplicative gate terms folded into the
            cancel (e.g. the BZ ``("BZ_TARGET_FRESH.*.-1", 1.0)`` re-fire
            suppressor, §G8) — cross-step bookkeeping only present on some ops.
    """

    band: str
    output_base: str
    output_gate: str
    extra_gate_terms: Tuple[Tuple[str, float], ...] = ()


# The two standard cancel bands every PC override uses (LO cross-step, HI
# same-step). Ops that add extra gate terms (BZ) build their own tuple.
_PC_MUX_STANDARD_CANCEL_BANDS: Tuple[PcMuxCancelBand, ...] = (
    PcMuxCancelBand("lo", "OUTPUT_LO", "OUTPUT_LO.*.-1"),
    PcMuxCancelBand("hi", "OUTPUT_HI_THIS_STEP", "OUTPUT_HI_THIS_STEP"),
)


@dataclass(frozen=True)
class PcMuxEncoder:
    """The target ENCODER for a PC override — WHERE the next PC comes from.

    Two modes cover every live branch op:

      * ``mode="direct_copy"`` — the PC byte is ALREADY encoded in a source
        band; copy nibble ``k`` straight through. ``lo_source``/``hi_source``
        name the two source bands (``AX_CARRY_LO``/``AX_CARRY_HI`` for the
        first-step/delayed JMP AX-route, ``FETCH_LO``/``FETCH_HI`` for the
        all-step JMP). ``gate_weight`` is the per-source one-hot scale (1.0 for
        the register bands).

      * ``mode="imm_to_byte_addr"`` — the source band carries a raw instruction
        INDEX ``i`` (BZ/BNZ/JSR read it from ``FETCH_LO``); write the encoded
        byte address ``target = i*INSTR_WIDTH + PC_OFFSET`` per nibble
        (§G2's ``idx_to_pc``). Only ``lo_source`` is used (the byte-0 hi nibble
        is determined entirely by ``i>>1`` via the SAME gate). ``gate_weight``
        is ``1/FETCH_PC_MARKER_AMP`` for the amplified JSR FETCH band, 1.0 for
        the BZ/BNZ FETCH band.

    ``odd_hi_correction`` optionally appends the byte-0 HIGH-nibble
    ``+INSTR_WIDTH`` fix for an ODD index-hi nibble (only meaningful for
    ``imm_to_byte_addr``): ``"jsr"`` reproduces the JSR authoring (odd-hi in
    ``gate_terms``, EVEN-hi blockers in ``conditions``, threshold ``+0.5``),
    ``"branch"`` reproduces the BZ/BNZ authoring (odd-hi + even-hi blockers in
    ``conditions`` at ``fetch_norm`` weight, threshold ``+2.0``). ``None``
    emits no correction (byte-0-only; the ``>=0x100`` byte-1 path is a SEPARATE
    known wall, §G3, left untouched).
    """

    mode: str  # "direct_copy" | "imm_to_byte_addr"
    lo_source: str
    hi_source: Optional[str] = None
    gate_weight: float = 1.0
    odd_hi_correction: Optional[str] = None  # None | "jsr" | "branch"
    # For the ``odd_hi_correction`` bands: the one-hot amplitude of the FETCH_HI
    # index-hi band (``jsr`` uses the raw 1.0 gate, ``branch`` normalizes by
    # 1/AMP). Set from the op's ``fetch_pc_marker_amp``.
    fetch_pc_marker_amp: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("direct_copy", "imm_to_byte_addr"):
            raise ValueError(
                f"PcMuxEncoder: mode must be 'direct_copy' | "
                f"'imm_to_byte_addr'; got {self.mode!r}"
            )
        if self.mode == "direct_copy" and self.hi_source is None:
            raise ValueError(
                "PcMuxEncoder(direct_copy): hi_source is required "
                "(the already-encoded PC byte-0 high nibble source)"
            )
        if self.odd_hi_correction not in (None, "jsr", "branch"):
            raise ValueError(
                f"PcMuxEncoder: odd_hi_correction must be None|'jsr'|'branch'; "
                f"got {self.odd_hi_correction!r}"
            )
        if self.odd_hi_correction is not None and self.mode != "imm_to_byte_addr":
            raise ValueError(
                "PcMuxEncoder: odd_hi_correction only applies to the "
                "imm_to_byte_addr encoder (the imm*8 shift's hi&1 carry)"
            )


@dataclass(frozen=True)
class PcMuxSpec:
    """Compact per-op description of one PC-source override.

    The whole hand-authored override collapses to: a name, a GATE
    (``conditions`` + ``threshold``), the cancel bands, the target encoder,
    and an optional cross-step "fresh" writer. ``pc_mux(spec)`` derives the
    ``FFNRule`` tuple byte-for-byte.

    Args:
        name: rule-name prefix (matches the hand builder's, e.g.
            ``"l6_jmp_all_step"``, ``"post_l9_bz"``).
        conditions: the AND gate (opcode + marker + step guard). Shared by the
            cancel + encoder bands.
        threshold: the AND threshold for the cancel + encoder bands.
        write_scale: the per-write output scale (``2.0 / S``).
        encoder: the :class:`PcMuxEncoder` (target source + optional odd-hi).
        cancel_bands: the :class:`PcMuxCancelBand` tuple (defaults to the
            standard LO cross-step / HI same-step pair).
        target_conditions: OPTIONAL distinct conditions for the encoder/target
            + odd-hi bands (BZ/BNZ add a ``MARK_STACK0`` blocker on the target
            band that the cancel band omits). Defaults to ``conditions``.
        odd_hi_conditions: OPTIONAL distinct base conditions for the odd-hi
            correction band (the JSR correction re-lists a fresh conditions
            tuple with a different opcode-nibble blocker weight). Defaults to
            ``target_conditions``.
        odd_hi_threshold: OPTIONAL explicit threshold for the odd-hi band
            (JSR uses ``threshold + 0.5``; branch uses ``threshold + 2.0``,
            derived when ``None``).
        fresh_writer: OPTIONAL ``(dim, extra_conditions)`` for a cross-step
            "this step took the branch" bit (BZ ``BZ_TARGET_FRESH``). Written
            at the cancel gate.
    """

    name: str
    conditions: Tuple[Tuple[str, float], ...]
    threshold: float
    write_scale: float
    encoder: PcMuxEncoder
    cancel_bands: Tuple[PcMuxCancelBand, ...] = _PC_MUX_STANDARD_CANCEL_BANDS
    target_conditions: Optional[Tuple[Tuple[str, float], ...]] = None
    odd_hi_conditions: Optional[Tuple[Tuple[str, float], ...]] = None
    odd_hi_threshold: Optional[float] = None
    fresh_writer: Optional[Tuple[str, Tuple[Tuple[str, float], ...]]] = None

    @property
    def effective_target_conditions(self) -> Tuple[Tuple[str, float], ...]:
        return (
            self.conditions
            if self.target_conditions is None
            else self.target_conditions
        )

    @property
    def effective_odd_hi_conditions(self) -> Tuple[Tuple[str, float], ...]:
        if self.odd_hi_conditions is not None:
            return self.odd_hi_conditions
        return self.effective_target_conditions


@dataclass(frozen=True)
class PcMuxBundle:
    """Result of :func:`pc_mux`.

    Attributes:
        spec: the originating :class:`PcMuxSpec`.
        rules_builder: ``() -> Tuple[FFNRule, ...]`` — the derived override
            rules, byte-identical to the hand-authored builder.
        reads / writes: dep-graph dims the op factory declares (base names).
    """

    spec: PcMuxSpec
    rules_builder: Callable[[], Tuple[FFNRule, ...]]
    reads: Set[str]
    writes: Set[str]


def _pc_encoder_target_lo(k: int, instr_width: int, pc_offset: int) -> int:
    return (k * instr_width + pc_offset) & 0xF


def _pc_encoder_target_hi(k: int, instr_width: int, pc_offset: int) -> int:
    return ((k * instr_width + pc_offset) >> 4) & 0xF


def _pc_mux_cancel_rules(spec: PcMuxSpec) -> list[FFNRule]:
    """The 16-per-band ``-OUTPUT[k]`` cancel units (subtract the seq default)."""

    rules: list[FFNRule] = []
    for cb in spec.cancel_bands:
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_cancel_{cb.band}_{k}",
                conditions=spec.conditions,
                threshold=spec.threshold,
                gate=f"{cb.output_gate}+{k}",
                gate_weight=-1.0,
                gate_terms=cb.extra_gate_terms,
                writes=((f"{cb.output_base}+{k}", spec.write_scale),),
            ))
    return rules


def _pc_mux_encoder_rules(
    spec: PcMuxSpec, instr_width: int, pc_offset: int,
) -> list[FFNRule]:
    """The 16-per-band target ENCODER units (WHERE the next PC comes from)."""

    enc = spec.encoder
    conds = spec.effective_target_conditions
    rules: list[FFNRule] = []
    if enc.mode == "direct_copy":
        # Copy the already-encoded PC byte nibble straight through.
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_target_lo_{k}",
                conditions=conds,
                threshold=spec.threshold,
                gate=f"{enc.lo_source}+{k}",
                gate_weight=enc.gate_weight,
                writes=((f"OUTPUT_LO+{k}", spec.write_scale),),
            ))
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_target_hi_{k}",
                conditions=conds,
                threshold=spec.threshold,
                gate=f"{enc.hi_source}+{k}",
                gate_weight=enc.gate_weight,
                writes=((f"OUTPUT_HI_THIS_STEP+{k}", spec.write_scale),),
            ))
    else:  # imm_to_byte_addr
        # Convert the raw instruction INDEX k -> encoded byte address nibbles.
        for k in range(16):
            target_lo = _pc_encoder_target_lo(k, instr_width, pc_offset)
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_byteaddr_lo_{k}",
                conditions=conds,
                threshold=spec.threshold,
                gate=f"{enc.lo_source}+{k}",
                gate_weight=enc.gate_weight,
                writes=((f"OUTPUT_LO+{target_lo}", spec.write_scale),),
            ))
        for k in range(16):
            target_hi = _pc_encoder_target_hi(k, instr_width, pc_offset)
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_byteaddr_hi_{k}",
                conditions=conds,
                threshold=spec.threshold,
                gate=f"{enc.lo_source}+{k}",
                gate_weight=enc.gate_weight,
                writes=((f"OUTPUT_HI_THIS_STEP+{target_hi}", spec.write_scale),),
            ))
    return rules


def _pc_mux_odd_hi_rules(
    spec: PcMuxSpec, instr_width: int, pc_offset: int,
) -> list[FFNRule]:
    """The optional byte-0 HIGH-nibble ``+INSTR_WIDTH`` odd-index-hi fix.

    ``imm*INSTR_WIDTH mod 256`` gains ``+INSTR_WIDTH*8/16`` in byte-0's high
    nibble exactly when the index-hi nibble is ODD (``hi&1``). Two authoring
    flavors, both derived here (JSR ``gate_terms`` vs BZ/BNZ ``conditions``).
    """

    enc = spec.encoder
    if enc.odd_hi_correction is None:
        return []
    rules: list[FFNRule] = []
    amp = enc.fetch_pc_marker_amp
    base_hi = spec.effective_odd_hi_conditions
    ws = spec.write_scale

    def base_target_hi(k: int) -> int:
        return _pc_encoder_target_hi(k, instr_width, pc_offset)

    def corrected_target_hi(k: int) -> int:
        # +INSTR_WIDTH on the byte-0 high nibble (== base + 8 for INSTR_WIDTH=8).
        return (base_target_hi(k) + instr_width) & 0xF

    if enc.odd_hi_correction == "jsr":
        odd_imm_hi_gate = tuple(
            (f"FETCH_HI+{k}", 1.0) for k in range(1, 16, 2)
        )
        even_imm_hi_blockers = tuple(
            (f"FETCH_HI+{k}", -10.0) for k in range(0, 16, 2)
        )
        thr = (
            spec.odd_hi_threshold
            if spec.odd_hi_threshold is not None
            else spec.threshold + 0.5
        )
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_target_hi_odd_imm_hi_correction_{k}",
                conditions=base_hi
                + ((f"FETCH_LO+{k}", 1.0),)
                + even_imm_hi_blockers,
                threshold=thr,
                gate_terms=odd_imm_hi_gate,
                writes=(
                    (f"OUTPUT_HI_THIS_STEP+{base_target_hi(k)}", -ws),
                    (f"OUTPUT_HI_THIS_STEP+{corrected_target_hi(k)}", ws),
                ),
            ))
    else:  # "branch"
        fetch_norm = 1.0 / amp
        odd_imm_hi_require = tuple(
            (f"FETCH_HI+{j}", fetch_norm) for j in range(1, 16, 2)
        )
        even_imm_hi_blockers = tuple(
            (f"FETCH_HI+{j}", -10.0 * fetch_norm) for j in range(0, 16, 2)
        )
        thr = (
            spec.odd_hi_threshold
            if spec.odd_hi_threshold is not None
            else spec.threshold + 2.0
        )
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_target_hi_odd_imm_hi_correction_{k}",
                conditions=base_hi
                + ((f"FETCH_LO+{k}", fetch_norm),)
                + odd_imm_hi_require
                + even_imm_hi_blockers,
                threshold=thr,
                gate=f"FETCH_LO+{k}",
                writes=(
                    (f"OUTPUT_HI_THIS_STEP+{base_target_hi(k)}", -ws),
                    (f"OUTPUT_HI_THIS_STEP+{corrected_target_hi(k)}", ws),
                ),
            ))
    return rules


def _pc_mux_fresh_writer_rules(spec: PcMuxSpec) -> list[FFNRule]:
    """The optional cross-step 'branch taken this step' bit (BZ, §G8)."""

    if spec.fresh_writer is None:
        return []
    dim, extra = spec.fresh_writer
    return [multi_way_and_rule(
        name=f"{spec.name}_target_fresh_write",
        conditions=spec.conditions + extra,
        threshold=spec.threshold,
        writes=((dim, spec.write_scale),),
    )]


def pc_mux(
    spec: PcMuxSpec,
    *,
    instr_width: int,
    pc_offset: int,
) -> PcMuxBundle:
    """Derive one PC-source override's FFN rules from ``spec``.

    Emits, in order: the cancel bands (subtract the SEQUENTIAL default), the
    target ENCODER (direct-copy OR imm->byte-addr), the optional odd-index-hi
    byte-0 correction, then the optional cross-step 'fresh' writer. Reproduces
    the six hand-authored override builders byte-for-byte (proof:
    ``tools/_isa_golden_hash.py`` unchanged with the derived form live).

    ``instr_width`` / ``pc_offset`` are the ISA constants (``constants.py``:
    ``INSTR_WIDTH=8``, ``PC_OFFSET=2``) — the encoder's only numeric inputs
    (``idx_to_pc(i) = i*INSTR_WIDTH + PC_OFFSET``). No compiler machinery: the
    rules lower through the ordinary ``multi_way_and_rule`` / ``FFNRule``
    shapes.
    """

    def rules_builder() -> Tuple[FFNRule, ...]:
        out: list[FFNRule] = []
        out.extend(_pc_mux_cancel_rules(spec))
        out.extend(_pc_mux_encoder_rules(spec, instr_width, pc_offset))
        out.extend(_pc_mux_odd_hi_rules(spec, instr_width, pc_offset))
        out.extend(_pc_mux_fresh_writer_rules(spec))
        return tuple(out)

    # Dep-graph derivation: reads = the gate + cancel gate + encoder source
    # dims; writes = the OUTPUT bands (+ the fresh bit).
    reads: Set[str] = set()
    writes: Set[str] = set()

    def _base(name: str) -> str:
        # Strip a trailing ``+k`` offset and any ``.*.-1`` cross-step alias.
        return name.split("+", 1)[0]

    for (dim, _w) in spec.conditions:
        reads.add(_base(dim))
    for cb in spec.cancel_bands:
        reads.add(_base(cb.output_gate))
        writes.add(_base(cb.output_base))
        for (dim, _w) in cb.extra_gate_terms:
            reads.add(_base(dim))
    for (dim, _w) in spec.effective_target_conditions:
        reads.add(_base(dim))
    reads.add(_base(spec.encoder.lo_source))
    if spec.encoder.hi_source is not None:
        reads.add(_base(spec.encoder.hi_source))
    writes.add("OUTPUT_LO")
    writes.add("OUTPUT_HI_THIS_STEP")
    if spec.encoder.odd_hi_correction is not None:
        reads.add("FETCH_HI")
        for (dim, _w) in spec.effective_odd_hi_conditions:
            reads.add(_base(dim))
    if spec.fresh_writer is not None:
        dim, extra = spec.fresh_writer
        writes.add(_base(dim))
        for (d, _w) in extra:
            reads.add(_base(d))

    return PcMuxBundle(
        spec=spec,
        rules_builder=rules_builder,
        reads=reads,
        writes=writes,
    )


# ===========================================================================
# REGISTER-DELTA / FRAME-STEP — the CONTROL-family per-step register updates,
# derived GENERICALLY as DATA
# ===========================================================================
#
# ``docs/semantic_spec_CONTROL.md`` §2a models every CONTROL op as a tuple of
# per-step register updates: a ``PcNextSpec`` (WHERE the next PC comes from) +
# an ordered ``FrameDeltaSpec[]`` (the SP/BP/STACK0 mutations). ``pc_mux`` above
# derives the branch-target PC OVERRIDES (the L6 cancel-then-write idiom). This
# section derives the OTHER half of the frame step — the register updates that
# the pc_mux agent left hand-authored in L3 + model_ops because they were
# "entangled with SP/BP/STACK0 frame defaults":
#
#   * the L3 SEQUENTIAL PC-next adder — ``PC_next = PC + INSTR_WIDTH`` — the
#     default every op starts from (``_register_default_ffn_rules`` units 0-3 +
#     86-133: a first-step CONSTANT default + a nibble-rotation ``reg + const``
#     adder with a lo->hi carry, gated ``MARK_PC ∧ HAS_SE ∧ ¬OP_LEV``);
#   * the model_ops JSR PC OVERRIDE — ``PC = imm*INSTR_WIDTH + PC_OFFSET`` — the
#     JSR branch-target, a cancel-then-write override that reads the raw
#     instruction INDEX from ``FETCH_LO`` (§1d).
#
# Both are *per-step register updates over the OUTPUT/EMBED bands*, expressed
# here as compact DATA:
#
#   RegisterDelta {
#     kind:       SEQUENTIAL_ADD | BRANCH_TARGET   # HOW the next value is formed
#     amount:     INSTR_WIDTH                       # the +const for SEQUENTIAL_ADD
#     ...gate + band + write-scale data...
#   }
#
# A ``frame_step`` groups the ordered deltas of ONE opcode's step (§2b's
# ``ControlOp.frame_delta``) and lowers them TOGETHER — the generic "apply
# these register deltas this step" the directive asks for. ``register_delta``
# lowers a single delta; ``frame_step`` concatenates a spec's deltas in
# declaration order (the same ordering §2a/§G5 requires for LEV's 4-way
# teardown, expressed as data rather than hand-sequenced FFN banks).
#
# NO compiler machinery — every delta lowers through the ordinary
# ``multi_way_and_rule`` / ``FFNRule`` shapes, so the compiler sees an ordinary
# FFN op. The byte-identity proof re-expresses the L3 sequential adder + the
# JSR PC override and gates the whole-model ``state_dict`` hash unchanged
# (``tools/_isa_golden_hash.py`` == ``91f55411``).


@dataclass(frozen=True)
class SequentialAddDelta:
    """``register + const`` per-step update as a nibble-rotation adder + carry.

    This is the L3 sequential PC-next: ``PC_next = PC + INSTR_WIDTH``. The old
    value's nibbles arrive in ``src_lo``/``src_hi`` (``EMBED_LO``/``EMBED_HI``);
    the update writes the new value's nibbles to ``dst_lo``/``dst_hi``
    (``OUTPUT_LO``/``OUTPUT_HI``). Three bands:

      * a FIRST-STEP CONSTANT default (``value = PC_OFFSET + INSTR_WIDTH``) —
        the register's step-0 value before any prior PC exists — as a
        marker-gated ``set`` + a ``fresh_key``-keyed ``undo`` pair (LO + HI).
        The set ALSO writes the same nibble into ``src_lo``/``src_hi`` so the
        downstream adder sees a consistent old-value band.
      * the INCREMENT band: 16 lo units ``new_lo = (k + amount) % 16`` gated on
        ``src_lo+k`` + 16 hi units copying ``src_hi+k -> dst_hi+k``.
      * the CARRY band: when the old lo nibble >= ``16 - amount`` a carry rolls
        into the hi nibble (``dst_hi+k -= v ; dst_hi+(k+1)%16 += v``).

    ``amount`` is the ISA ``INSTR_WIDTH`` (8); ``const_value`` is
    ``PC_OFFSET + INSTR_WIDTH`` (the first-step landing). ``suppress_op`` names
    the opcode that supplies its OWN next value so the increment must NOT fire
    (``OP_LEV`` — LEV pops the return-PC, §1a).
    """

    amount: int
    const_value: int
    src_lo: str = "EMBED_LO"
    src_hi: str = "EMBED_HI"
    dst_lo: str = "OUTPUT_LO"
    dst_hi: str = "OUTPUT_HI"
    # First-step default gate (a marker one-hot, e.g. MARK_PC) + the fresh key
    # that undoes the constant once a real prior value exists.
    default_marker: str = "MARK_PC"
    fresh_key: str = "HAS_SE"
    # The increment/carry gate: (marker, weight) + (fresh_key, weight) +
    # (suppress_op, weight). Kept as explicit weights so the derived rules
    # reproduce the hand tuning (OP_LEV at -1/5 for the increment, -1 for the
    # strict carry) byte-for-byte.
    suppress_op: str = "OP_LEV"
    incr_marker_weight: float = 1.0
    incr_fresh_weight: float = 1.0
    incr_suppress_weight: float = -1.0 / 5.0
    incr_threshold: float = 1.5
    carry_marker_weight: float = 4.0
    carry_fresh_weight: float = 1.0
    carry_suppress_weight: float = -1.0
    carry_threshold: float = 5.5


@dataclass(frozen=True)
class BranchTargetDelta:
    """A PC-source OVERRIDE that materializes ``PC = imm*INSTR_WIDTH+PC_OFFSET``.

    The model_ops JSR PC override (§1d): cancel the L3 sequential default in the
    OUTPUT bank (16 ``-OUTPUT_LO[k]`` + 16 ``-OUTPUT_HI[k]`` same-step self-gated
    units) then WRITE the branch target from the raw instruction INDEX carried in
    ``index_source`` (``FETCH_LO``): ``target_lo = (i*w+off)&0xF``,
    ``target_hi = (i*w+off)>>4`` per nibble. Between the two target bands a
    RESERVED index-hi band (16 units gated on ``index_hi_source`` with no
    down-write) keeps the unit cursor aligned with the legacy JSR layout (the
    >=0x100 byte-1 path is a SEPARATE flagged wall, §G3 — this delta emits the
    reserved no-op units, NOT the byte-1 stage).

    This is the SAME ``idx_to_pc`` encoder ``pc_mux``'s ``imm_to_byte_addr`` mode
    uses; it lives here (not in ``pc_mux``) because JSR's override writes the
    NON-cross-step ``OUTPUT_HI`` band (same-step self-gated cancel) and is part
    of the JSR opcode's full FRAME step (return-addr push + SP decrement), so it
    reads more naturally as one of JSR's ordered ``register_delta``s.
    """

    index_source: str = "FETCH_LO"
    index_hi_source: str = "FETCH_HI"
    dst_lo: str = "OUTPUT_LO"
    dst_hi: str = "OUTPUT_HI"
    reserved_hi_units: int = 16


@dataclass(frozen=True)
class PushDelta:
    """A ``PUSH(register)`` store: write a relayed register value onto a slot,
    cancelling the slot's arriving identity in the SAME hidden unit.

    This is the ENT ``push saved-BP`` frame delta (docs §G10 / §G5): at the
    STACK0 marker on an ENT step, the caller's BP (relayed by the L5 head into
    ``value_src``, a per-nibble TEMP one-hot band) is written onto ``OUTPUT``,
    while the ``identity_src`` (``EMBED``) copy that would otherwise pass through
    is subtracted in the same unit's gate. Both effects live in ONE
    ``multi_way_and_rule``: the AND ``conditions`` are the opcode+marker gate; the
    ``gate_terms`` carry the cancel-vs-write pair ``(identity_src+k, -1),
    (value_src+k, +1)`` so the unit fires on the arriving identity nibble and
    emits the relayed value nibble instead.

    LO band copies ``value_src[0..15] -> dst_lo`` with ``identity_src_lo`` cancel;
    HI band copies ``value_src[value_hi_offset + 0..15] -> dst_hi`` with
    ``identity_src_hi`` cancel. The value/identity offsets + the marker gate are
    the varying params; the 16-unit-per-band copy-with-cancel is the fixed shape.
    """

    value_src: str = "TEMP"
    value_hi_offset: int = 16
    identity_src_lo: str = "EMBED_LO"
    identity_src_hi: str = "EMBED_HI"
    dst_lo: str = "OUTPUT_LO"
    dst_hi: str = "OUTPUT_HI"
    threshold: float = 1.5


@dataclass(frozen=True)
class AssignDelta:
    """A ``register := other_register (± const)`` copy as a nibble-shift adder.

    This is the ENT ``BP := SP - 8`` frame delta (docs §G10): at the BP marker on
    an ENT step, the caller's SP (relayed into ``value_src`` per-nibble) is copied
    into the BP ``OUTPUT`` slot with the frame-link constant subtracted. Each LO
    unit gates on ``value_src+k`` and writes ``dst_lo[(k + lo_shift) % 16] += ws``
    while cancelling the arriving identity ``dst_lo[k] -= ws`` (so the assign
    REPLACES the slot rather than adding). Each HI unit gates on
    ``value_src[value_hi_offset + k]``, writes ``dst_hi[(k + hi_shift) % 16] += ws``
    and cancels ``dst_hi[k] -= ws``, guarded by ``borrow_blockers`` — a set of
    ``(value_src+lo_bit, -1)`` terms that veto the HI carry on the no-borrow half
    of the low nibble range (BP=SP-8: the top-8 low nibbles need no borrow, so the
    HI shift only applies when the low nibble is in ``[0, 8)``).

    ``lo_shift`` / ``hi_shift`` are ``(-amount) % 16`` and ``(-borrow) % 16`` for a
    subtract-by-``amount`` assign (``BP=SP-8`` => ``lo_shift=8, hi_shift=15``); a
    pure register copy uses ``lo_shift=hi_shift=0`` and empty ``borrow_blockers``.
    ``borrow_blocker_range`` is the ``[lo, hi)`` low-nibble range whose bits veto
    the HI carry (BP=SP-8: ``(8, 16)``); empty ``()`` disables the guard.
    """

    value_src: str = "TEMP"
    value_hi_offset: int = 16
    dst_lo: str = "OUTPUT_LO"
    dst_hi: str = "OUTPUT_HI"
    lo_shift: int = 8
    hi_shift: int = 15
    borrow_blocker_range: Tuple[int, int] = (8, 16)
    threshold: float = 1.5


@dataclass(frozen=True)
class PopCamDelta:
    """A ``register := pop(CAM)`` teardown: gate on a CAM/relay value band and
    write the popped value onto the register's ``OUTPUT`` slot.

    This is the LEV ``SP := BP`` / ``PC := return-addr`` frame deltas (docs §G5):
    the freed-stack-slot value has already been relayed (by an L9 CAM head) into a
    per-nibble one-hot band (``ADDR_B0_{LO,HI}`` for the SP=BP pop; ``TEMP`` for
    the PC pop). Each LO unit gates on ``value_src_lo+k`` and writes
    ``dst_lo[(k + lo_shift) % 16] += ws``; each HI unit gates on ``value_src_hi+k``
    (``value_src_hi`` may equal ``value_src_lo`` with a ``value_hi_offset``) and
    writes ``dst_hi[(k + hi_shift) % 16] += ws``. The pop is UNCONDITIONAL on the
    band cell (the CAM relay already selected the right slot); the opcode+marker
    gate is the shared ``conditions`` AND.

    ``gate_terms`` optionally carries a per-cell gate-side blocker (the LEV SP=BP
    pop's ``MARK_MEM`` hard blocker). ``value_gate`` selects whether the band cell
    is a positive AND condition folded into ``conditions`` (the SP=BP pop appends
    ``(value_src+k, +1)`` to conditions AND uses the same cell as the multiplicative
    gate) or ONLY the multiplicative gate (the PC pop uses ``gate="CONST"`` with the
    band cell as a plain condition). See :func:`_pop_cam_rules`.
    """

    value_src_lo: str = "ADDR_B0_LO"
    value_src_hi: str = "ADDR_B0_HI"
    value_hi_offset: int = 0
    dst_lo: str = "OUTPUT_LO"
    dst_hi: str = "OUTPUT_HI_THIS_STEP"
    lo_shift: int = 0
    hi_shift: int = 0
    # Gate mode: "band_cell" folds the band cell into BOTH the AND conditions and
    # the multiplicative gate (LEV SP=BP); "const" uses gate=const_gate with the
    # band cell as a plain condition (LEV PC pop).
    gate_mode: str = "band_cell"
    const_gate: str = "CONST"
    gate_terms: Tuple[Tuple[str, float], ...] = ()
    threshold: float = 40.0


@dataclass(frozen=True)
class RegisterDeltaSpec:
    """One per-step register update, tagged by ``kind``.

    Exactly one of ``sequential_add`` / ``branch_target`` / ``push`` / ``assign``
    / ``pop_cam`` is set (matching ``kind``). ``name`` is the rule-name prefix
    (matches the hand builder's). ``conditions`` is the shared AND gate (opcode +
    marker + step guard) for the override / push / assign / pop-CAM kinds;
    SEQUENTIAL_ADD builds its own gate from the delta's marker / fresh-key /
    suppress-op weights so its multi-band tuning is self-contained. ``threshold``
    / ``write_scale`` are the override gate threshold and the per-write ``2.0 / S``
    scale (for PUSH/ASSIGN/POP-CAM the per-delta dataclass carries its own
    threshold, so the spec ``threshold`` is unused for those).
    """

    name: str
    kind: str  # "sequential_add"|"branch_target"|"push"|"assign"|"pop_cam"
    write_scale: float
    sequential_add: Optional[SequentialAddDelta] = None
    branch_target: Optional[BranchTargetDelta] = None
    push: Optional[PushDelta] = None
    assign: Optional[AssignDelta] = None
    pop_cam: Optional[PopCamDelta] = None
    # The shared AND gate (opcode + marker + step guard). The override kind
    # (branch_target) uses it as the whole gate; PUSH/ASSIGN/POP-CAM use it as
    # the base conditions ANDed with the per-cell band term.
    conditions: Tuple[Tuple[str, float], ...] = ()
    threshold: float = 0.0

    _KIND_FIELD = {
        "sequential_add": "sequential_add",
        "branch_target": "branch_target",
        "push": "push",
        "assign": "assign",
        "pop_cam": "pop_cam",
    }

    def __post_init__(self) -> None:
        if self.kind not in self._KIND_FIELD:
            raise ValueError(
                f"RegisterDeltaSpec({self.name!r}): kind must be one of "
                f"{sorted(self._KIND_FIELD)}; got {self.kind!r}"
            )
        field = self._KIND_FIELD[self.kind]
        if getattr(self, field) is None:
            raise ValueError(
                f"RegisterDeltaSpec({self.name!r}): kind={self.kind!r} "
                f"requires {field}=... to be set"
            )


@dataclass(frozen=True)
class RegisterDeltaBundle:
    """Result of :func:`register_delta` / :func:`frame_step`.

    Attributes:
        specs: the originating :class:`RegisterDeltaSpec` tuple (one for
            :func:`register_delta`, N for :func:`frame_step`).
        rules_builder: ``() -> Tuple[FFNRule, ...]`` — the derived rules,
            byte-identical to the hand-authored builders, in delta order.
        sub_builders: named sub-band builders for a SEQUENTIAL_ADD delta whose
            bands are placed at NON-contiguous FFN unit positions (the L3 case:
            the first-step default at unit 0-3, the adder at 86-133). Keys
            ``"default"`` and ``"adder"``; empty for other kinds. Each maps to a
            ``() -> Tuple[FFNRule, ...]``.
        reads / writes: dep-graph dims the op factory declares (base names).
    """

    specs: Tuple[RegisterDeltaSpec, ...]
    rules_builder: Callable[[], Tuple[FFNRule, ...]]
    reads: Set[str]
    writes: Set[str]
    sub_builders: Dict[str, Callable[[], Tuple[FFNRule, ...]]] = None  # type: ignore[assignment]


def _seq_add_default_rules(spec: RegisterDeltaSpec) -> list[FFNRule]:
    """The SEQUENTIAL_ADD FIRST-STEP CONSTANT default band (4 units: LO/HI set +
    fresh_key-keyed undo). Split out so the L3 op can place it at its legacy
    unit position (0-3) independently of the adder band (86-133) — the two are
    ONE derived spec but separated in the L3 FFN unit layout."""

    d = spec.sequential_add
    assert d is not None
    ws = spec.write_scale
    rules: list[FFNRule] = []
    const_lo = d.const_value & 0xF
    const_hi = (d.const_value >> 4) & 0xF
    for band, dst, src, val in (
        ("lo", d.dst_lo, d.src_lo, const_lo),
        ("hi", d.dst_hi, d.src_hi, const_hi),
    ):
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_first_step_default_{band}_set",
            conditions=((d.default_marker, 1.0),),
            threshold=0.5,
            writes=((f"{dst}+{val}", ws), (f"{src}+{val}", ws)),
            scope=f"{d.default_marker} and not {d.fresh_key}",
        ))
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_first_step_default_{band}_undo",
            conditions=((d.fresh_key, 1.0),),
            threshold=0.5,
            gate=d.default_marker,
            writes=((f"{dst}+{val}", -ws), (f"{src}+{val}", -ws)),
            scope=f"{d.default_marker} and {d.fresh_key}",
        ))
    return rules


def _seq_add_adder_rules(spec: RegisterDeltaSpec) -> list[FFNRule]:
    """The SEQUENTIAL_ADD nibble-rotation adder band (48 units: increment lo/hi +
    carry). The ``reg + const`` core (§2c) — placed at the L3 unit position
    86-133."""

    d = spec.sequential_add
    assert d is not None
    ws = spec.write_scale
    rules: list[FFNRule] = []

    # --- INCREMENT lo nibble: new_k = (k + amount) % 16 ---
    incr_conds = (
        (d.fresh_key, d.incr_fresh_weight),
        (d.default_marker, d.incr_marker_weight),
        (d.suppress_op, d.incr_suppress_weight),
    )
    incr_scope = (
        f"{d.default_marker} and {d.fresh_key} and not {d.suppress_op}"
    )
    for k in range(16):
        new_k = (k + d.amount) % 16
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_increment_lo_{k}",
            conditions=incr_conds,
            threshold=d.incr_threshold,
            gate=f"{d.src_lo}+{k}",
            writes=((f"{d.dst_lo}+{new_k}", ws),),
            scope=incr_scope,
        ))

    # --- INCREMENT hi nibble: copy src_hi[k] -> dst_hi[k] ---
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_increment_hi_{k}",
            conditions=incr_conds,
            threshold=d.incr_threshold,
            gate=f"{d.src_hi}+{k}",
            writes=((f"{d.dst_hi}+{k}", ws),),
            scope=incr_scope,
        ))

    # --- CARRY correction: old lo nibble >= 16 - amount -> hi += 1 ---
    carry_threshold_lo = 16 - d.amount
    carry_conds = [
        (d.default_marker, d.carry_marker_weight),
        (d.fresh_key, d.carry_fresh_weight),
        (d.suppress_op, d.carry_suppress_weight),
    ]
    for lo_bit in range(carry_threshold_lo, 16):
        carry_conds.append((f"{d.src_lo}+{lo_bit}", 1.0))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_carry_correction_{k}",
            conditions=tuple(carry_conds),
            threshold=d.carry_threshold,
            gate=f"{d.src_hi}+{k}",
            writes=((f"{d.dst_hi}+{k}", -ws),
                    (f"{d.dst_hi}+{(k + 1) % 16}", ws)),
            scope=incr_scope,
        ))

    return rules


def _seq_add_rules(
    spec: RegisterDeltaSpec, instr_width: int, pc_offset: int,
) -> list[FFNRule]:
    """Derive the WHOLE SEQUENTIAL_ADD band (default + adder), contiguous.

    Reproduces the L3 sequential PC+8 bands byte-for-byte in one sequence.
    For the L3 op (which places the default at unit 0-3 and the adder at
    86-133) call :func:`_seq_add_default_rules` / :func:`_seq_add_adder_rules`
    via the bundle's ``sub_builders``.
    """

    return _seq_add_default_rules(spec) + _seq_add_adder_rules(spec)


def _branch_target_rules(
    spec: RegisterDeltaSpec, instr_width: int, pc_offset: int,
) -> list[FFNRule]:
    """Derive the BRANCH_TARGET PC override (cancel + idx_to_pc encoder) rules.

    Reproduces the model_ops JSR PC override byte-for-byte: 16 same-step
    self-gated OUTPUT_LO cancel + 16 OUTPUT_HI cancel + 16 idx->byte-addr LO
    target + 16 RESERVED index-hi no-op + 16 idx->byte-addr HI carry.
    """

    d = spec.branch_target
    assert d is not None  # guarded by RegisterDeltaSpec.__post_init__
    conds = spec.conditions
    thr = spec.threshold
    ws = spec.write_scale
    rules: list[FFNRule] = []

    # 16 OUTPUT_LO cancel: gate = -dst_lo[k], write dst_lo[k] (subtract the
    # L3 sequential default from the SAME-step OUTPUT bank).
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_cancel_output_lo_{k}",
            conditions=conds,
            threshold=thr,
            gate=f"{d.dst_lo}+{k}",
            gate_weight=-1.0,
            writes=((f"{d.dst_lo}+{k}", ws),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_cancel_output_hi_{k}",
            conditions=conds,
            threshold=thr,
            gate=f"{d.dst_hi}+{k}",
            gate_weight=-1.0,
            writes=((f"{d.dst_hi}+{k}", ws),),
        ))
    # 16 index->byte-addr LO target.
    for k in range(16):
        target_lo = ((k * instr_width) + pc_offset) & 0xF
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_target_lo_{k}",
            conditions=conds,
            threshold=thr,
            gate=f"{d.index_source}+{k}",
            writes=((f"{d.dst_lo}+{target_lo}", ws),),
        ))
    # 16 RESERVED index-hi no-op (legacy layout alignment; the >=0x100 byte-1
    # stage is a SEPARATE flagged path, §G3, NOT emitted here).
    for k in range(d.reserved_hi_units):
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_index_hi_reserved_{k}",
            conditions=conds,
            threshold=thr,
            gate=f"{d.index_hi_source}+{k}",
            writes=(),
        ))
    # 16 index->byte-addr HI carry.
    for k in range(16):
        target_hi = ((k * instr_width) + pc_offset) >> 4
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_target_hi_from_lo_{k}",
            conditions=conds,
            threshold=thr,
            gate=f"{d.index_source}+{k}",
            writes=((f"{d.dst_hi}+{target_hi}", ws),),
        ))
    return rules


def _push_rules(spec: RegisterDeltaSpec) -> list[FFNRule]:
    """Derive the PUSH (store relayed register onto slot) frame delta.

    Reproduces the ENT ``push saved-BP`` band (model_ops
    ``_function_call_ent_stack0_rules``) byte-for-byte: 16 LO + 16 HI
    copy-with-cancel units. Each unit's AND is ``spec.conditions`` (opcode +
    marker); the ``gate_terms`` carry ``(identity_src+k, -1), (value_src+k, +1)``
    so the arriving identity nibble is subtracted and the relayed value nibble
    written in one hidden unit.
    """
    d = spec.push
    assert d is not None  # guarded by RegisterDeltaSpec.__post_init__
    ws = spec.write_scale
    conds = spec.conditions
    rules: list[FFNRule] = []
    for band, dst, ident_src, val_off in (
        ("lo", d.dst_lo, d.identity_src_lo, 0),
        ("hi", d.dst_hi, d.identity_src_hi, d.value_hi_offset),
    ):
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"{spec.name}_{band}_{k}",
                conditions=conds,
                threshold=d.threshold,
                gate_terms=(
                    (f"{ident_src}+{k}", -1.0),
                    (f"{d.value_src}+{val_off + k}", 1.0),
                ),
                writes=((f"{dst}+{k}", ws),),
            ))
    return rules


def _assign_rules(spec: RegisterDeltaSpec) -> list[FFNRule]:
    """Derive the ASSIGN (``dst := src ± const`` register copy) frame delta.

    Reproduces the ENT ``BP := SP - 8`` band (model_ops
    ``_function_call_ent_bp_rules``) byte-for-byte: 16 LO + 16 HI shift-with-
    identity-cancel units. LO gates on ``value_src+k`` and writes
    ``dst_lo[(k+lo_shift)%16] += ws`` while cancelling ``dst_lo[k] -= ws``. HI
    gates on ``value_src[value_hi_offset+k]`` with the ``borrow_blocker_range``
    veto terms appended to the conditions, writing ``dst_hi[(k+hi_shift)%16]``
    and cancelling ``dst_hi[k]``.
    """
    d = spec.assign
    assert d is not None
    ws = spec.write_scale
    conds = spec.conditions
    lo_bl, hi_bl = d.borrow_blocker_range if d.borrow_blocker_range else (0, 0)
    borrow_blockers = tuple(
        (f"{d.value_src}+{lo_bit}", -1.0) for lo_bit in range(lo_bl, hi_bl)
    )
    rules: list[FFNRule] = []
    for k in range(16):
        new_k = (k + d.lo_shift) % 16
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_lo_{k}",
            conditions=conds,
            threshold=d.threshold,
            gate=f"{d.value_src}+{k}",
            writes=(
                (f"{d.dst_lo}+{new_k}", ws),
                (f"{d.dst_lo}+{k}", -ws),
            ),
        ))
    for k in range(16):
        new_k = (k + d.hi_shift) % 16
        rules.append(multi_way_and_rule(
            name=f"{spec.name}_hi_{k}",
            conditions=conds + borrow_blockers,
            threshold=d.threshold,
            gate=f"{d.value_src}+{d.value_hi_offset + k}",
            writes=(
                (f"{d.dst_hi}+{new_k}", ws),
                (f"{d.dst_hi}+{k}", -ws),
            ),
        ))
    return rules


def _pop_cam_rules(spec: RegisterDeltaSpec) -> list[FFNRule]:
    """Derive the POP-CAM (``dst := pop(CAM/relay band)``) frame delta.

    Reproduces the LEV ``SP := BP`` (l16 ``l16_lev_sp_bp_plus16_*``, gate_mode
    ``band_cell``) and ``PC := return-addr`` (l16 ``l16_lev_pc_temp_*``,
    gate_mode ``const``) pops byte-for-byte: 16 LO + 16 HI units gating on the
    relayed value band. In ``band_cell`` mode the band cell is folded into BOTH
    the AND conditions and the multiplicative gate (with an optional per-cell
    ``gate_terms`` blocker); in ``const`` mode the band cell is a plain
    condition and the gate is ``const_gate``.
    """
    d = spec.pop_cam
    assert d is not None
    ws = spec.write_scale
    conds = spec.conditions
    rules: list[FFNRule] = []
    for band, dst, src, shift in (
        ("lo", d.dst_lo, d.value_src_lo, d.lo_shift),
        ("hi", d.dst_hi, d.value_src_hi, d.hi_shift),
    ):
        for k in range(16):
            new_k = (k + shift) % 16
            cell = f"{src}+{d.value_hi_offset + k}" if band == "hi" else f"{src}+{k}"
            if d.gate_mode == "band_cell":
                rules.append(multi_way_and_rule(
                    name=f"{spec.name}_{band}_{k}",
                    conditions=conds + ((cell, 1.0),),
                    threshold=d.threshold,
                    gate=cell,
                    gate_terms=d.gate_terms,
                    writes=((f"{dst}+{new_k}", ws),),
                ))
            else:  # "const"
                rules.append(multi_way_and_rule(
                    name=f"{spec.name}_{band}_{k}",
                    conditions=conds + ((cell, 1.0),),
                    threshold=d.threshold,
                    gate=d.const_gate,
                    gate_terms=d.gate_terms,
                    writes=((f"{dst}+{new_k}", ws),),
                ))
    return rules


def _register_delta_rules(
    spec: RegisterDeltaSpec, instr_width: int, pc_offset: int,
) -> list[FFNRule]:
    if spec.kind == "sequential_add":
        return _seq_add_rules(spec, instr_width, pc_offset)
    if spec.kind == "push":
        return _push_rules(spec)
    if spec.kind == "assign":
        return _assign_rules(spec)
    if spec.kind == "pop_cam":
        return _pop_cam_rules(spec)
    return _branch_target_rules(spec, instr_width, pc_offset)


def _register_delta_reads_writes(
    spec: RegisterDeltaSpec,
) -> Tuple[Set[str], Set[str]]:
    reads: Set[str] = set()
    writes: Set[str] = set()

    def _base(name: str) -> str:
        return name.split("+", 1)[0]

    if spec.kind == "sequential_add":
        sd = spec.sequential_add
        assert sd is not None
        reads.update({sd.default_marker, sd.fresh_key, sd.suppress_op,
                      sd.src_lo, sd.src_hi})
        writes.update({sd.dst_lo, sd.dst_hi, sd.src_lo, sd.src_hi})
    elif spec.kind == "push":
        pd = spec.push
        assert pd is not None
        for (dim, _w) in spec.conditions:
            reads.add(_base(dim))
        reads.update({pd.value_src, pd.identity_src_lo, pd.identity_src_hi})
        writes.update({pd.dst_lo, pd.dst_hi})
    elif spec.kind == "assign":
        ad = spec.assign
        assert ad is not None
        for (dim, _w) in spec.conditions:
            reads.add(_base(dim))
        reads.add(ad.value_src)
        writes.update({ad.dst_lo, ad.dst_hi})
    elif spec.kind == "pop_cam":
        cd = spec.pop_cam
        assert cd is not None
        for (dim, _w) in spec.conditions:
            reads.add(_base(dim))
        for (dim, _w) in cd.gate_terms:
            reads.add(_base(dim))
        reads.update({cd.value_src_lo, cd.value_src_hi})
        if cd.gate_mode == "const":
            reads.add(_base(cd.const_gate))
        writes.update({cd.dst_lo, cd.dst_hi})
    else:
        bd = spec.branch_target
        assert bd is not None
        for (dim, _w) in spec.conditions:
            reads.add(_base(dim))
        reads.update({bd.index_source, bd.index_hi_source, bd.dst_lo, bd.dst_hi})
        writes.update({bd.dst_lo, bd.dst_hi})
    return reads, writes


def register_delta(
    spec: RegisterDeltaSpec,
    *,
    instr_width: int,
    pc_offset: int,
) -> RegisterDeltaBundle:
    """Derive ONE per-step register update's FFN rules from ``spec``.

    SEQUENTIAL_ADD emits the first-step constant default + the nibble-rotation
    ``reg + const`` adder (increment lo/hi + carry). BRANCH_TARGET emits the
    cancel-then-write PC override (same-step OUTPUT cancel + ``idx_to_pc``
    encoder + reserved index-hi band). Reproduces the hand-authored L3
    sequential adder / model_ops JSR override byte-for-byte (proof:
    ``tools/_isa_golden_hash.py`` unchanged with the derived form live).

    ``instr_width`` / ``pc_offset`` are the ISA constants (``constants.py``:
    ``INSTR_WIDTH=8``, ``PC_OFFSET=2``) — the adder amount + the encoder's
    numeric inputs. No compiler machinery: the rules lower through the ordinary
    ``multi_way_and_rule`` / ``FFNRule`` shapes.
    """

    return frame_step((spec,), instr_width=instr_width, pc_offset=pc_offset)


def frame_step(
    specs: Sequence[RegisterDeltaSpec],
    *,
    instr_width: int,
    pc_offset: int,
) -> RegisterDeltaBundle:
    """Derive an ORDERED group of per-step register updates ("apply these
    register deltas this step").

    ``specs`` is the opcode's ordered ``frame_delta`` list (§2a/§2b). Each
    delta lowers via :func:`register_delta`'s single-delta path and the rules
    concatenate in declaration order — the generic "frame step" the CONTROL
    directive asks for (the same ordering §G5's LEV 4-way teardown needs,
    expressed as data instead of hand-sequenced FFN banks). Reproduces the
    hand-authored builders byte-for-byte.
    """

    specs_t = tuple(specs)

    def rules_builder() -> Tuple[FFNRule, ...]:
        out: list[FFNRule] = []
        for spec in specs_t:
            out.extend(_register_delta_rules(spec, instr_width, pc_offset))
        return tuple(out)

    reads: Set[str] = set()
    writes: Set[str] = set()
    for spec in specs_t:
        r, w = _register_delta_reads_writes(spec)
        reads |= r
        writes |= w

    # Expose split sub-band builders for a lone SEQUENTIAL_ADD whose default +
    # adder bands land at NON-contiguous FFN unit positions (the L3 case).
    sub_builders: Dict[str, Callable[[], Tuple[FFNRule, ...]]] = {}
    if len(specs_t) == 1 and specs_t[0].kind == "sequential_add":
        only = specs_t[0]
        sub_builders = {
            "default": lambda: tuple(_seq_add_default_rules(only)),
            "adder": lambda: tuple(_seq_add_adder_rules(only)),
        }

    return RegisterDeltaBundle(
        specs=specs_t,
        rules_builder=rules_builder,
        reads=reads,
        writes=writes,
        sub_builders=sub_builders,
    )


# ===========================================================================
# REGISTER BYTE-DEFAULT WRITER — the marker-gated SP/BP/STACK0 byte defaults
# ===========================================================================
#
# The OTHER half of the CONTROL frame-step the frame agent flagged (docs
# §G5/§G10, alongside the ENT/LEV frame deltas): the SP/BP/STACK0 byte-default
# bands in ``l3_ops._register_default_ffn_rules``. These are NOT per-step deltas
# — they are the register's DEFAULT WRITER: at the register's marker/byte-index
# rows on the FIRST step (before any prior value exists), predict the register's
# constant default (mostly 0, with a small first-step landing for SP/BP byte-1).
#
# Every SP/BP/STACK0 default band is the SAME 4-sub-band shape expressed as DATA
# (:class:`RegisterByteDefaultSpec`):
#
#   1. MARKER default (SP/BP): at ``(MARK_X ∧ ¬HAS_SE)`` write 0 to OUTPUT byte0.
#   2. BYTE-INDEX defaults: at ``(select_conditions ∧ BYTE_INDEX_k)`` for each k
#      in ``byte_idx_default`` write 0 to OUTPUT byte0. ``select_conditions`` is
#      the register-selector AND (SP/BP: ``(H1+I, 1)``; STACK0: ``(H4+3, 1),
#      (H1+3, -1)`` — the H4-through-STACK0 minus the BP-area exclusion).
#   3. BYTE-1 FIRST-STEP (SP/BP): at ``(select_conditions ∧ BYTE_INDEX_1 ∧
#      ¬HAS_SE)`` write the register's first-step landing (SP/BP byte-1 lo = 1).
#   4. MARKER FIRST-STEP (STACK0): at ``(MARK_STACK0 ∧ ¬HAS_SE)`` write 0.
#
# The generator emits these bands in the SAME order + names the hand-authored
# code used, so the derived form is byte-identical (proof:
# ``tools/_isa_golden_hash.py`` == ``91f55411``).


@dataclass(frozen=True)
class RegisterByteDefaultSpec:
    """Declarative description of one register's marker-gated byte-default band.

    A register (SP / BP / STACK0) whose OUTPUT byte defaults to a constant on the
    first step, before any prior value has been carried across the step boundary.
    Every field is DATA; the fixed 4-sub-band shape is supplied by
    :func:`register_byte_defaults`.

    Attributes:
        name_prefix: rule-name prefix (matches the hand builder's, e.g.
            ``"layer3_ffn.sp"``). Sub-bands append ``_marker_default_{lo,hi}`` /
            ``_byte_idx_{k}_default_{lo,hi}`` / ``_byte_1_first_step_{lo,hi}`` /
            ``_first_step_default_{lo,hi}``.
        select_conditions: the register-selector AND that PRECEDES the
            ``BYTE_INDEX_k`` term in the byte-index / byte-1 sub-bands (SP/BP:
            ``(("H1+2", 1.0),)``; STACK0: ``(("H4+3", 1.0),)``).
        byte_idx_trailing_conditions: register-selector AND terms that FOLLOW the
            ``BYTE_INDEX_k`` term (STACK0's BP-area exclusion sits AFTER the
            byte-index in the hand layout: ``(("H1+3", -1.0),)``). Empty for
            SP/BP. Order is load-bearing for the rule-tuple match (weights are
            order-independent, but the derived rule reproduces the hand order).
        byte_idx_name: the name infix for the byte-index sub-band rules (SP/BP:
            ``"byte_idx"`` -> ``sp_byte_idx_0_default_lo``; STACK0: ``"byte"`` ->
            ``stack0_byte_0_default_lo``).
        dst_lo / dst_hi: OUTPUT bands the defaults write.
        write_scale: per-write magnitude (``2.0 / S``).
        marker: the register marker for the MARKER-default sub-band (SP/BP:
            ``"MARK_SP"`` / ``"MARK_BP"``). ``None`` disables sub-band 1.
        marker_first_step: the marker for the MARKER-FIRST-STEP sub-band (STACK0:
            ``"MARK_STACK0"``). ``None`` disables sub-band 4.
        fresh_key: the freshness one-hot subtracted on the ``¬HAS_SE`` gates
            (``"HAS_SE"``).
        byte_idx_default: the byte indices whose default is 0 (SP/BP: ``(0, 2)``;
            STACK0: ``(0, 1, 2)``).
        byte_idx_default_threshold: AND threshold for the byte-index defaults.
        marker_threshold: AND threshold for the marker / marker-first-step bands.
        byte1_first_step: emit the BYTE-1 FIRST-STEP sub-band (SP/BP: ``True``;
            STACK0: ``False``). The landing is ``dst_lo[1] = ws`` (lo nibble 1),
            ``dst_hi[0] = ws`` — the SP/BP frame's first-step byte-1 value.
        byte1_index_dim: the byte-1 index one-hot for the first-step band
            (``"BYTE_INDEX_1"``).
    """

    name_prefix: str
    select_conditions: Tuple[Tuple[str, float], ...]
    byte_idx_trailing_conditions: Tuple[Tuple[str, float], ...] = ()
    byte_idx_name: str = "byte_idx"
    dst_lo: str = "OUTPUT_LO"
    dst_hi: str = "OUTPUT_HI"
    write_scale: float = 0.02
    marker: Optional[str] = None
    marker_first_step: Optional[str] = None
    fresh_key: str = "HAS_SE"
    byte_idx_default: Tuple[int, ...] = (0, 2)
    byte_idx_default_threshold: float = 1.5
    marker_threshold: float = 0.5
    byte1_first_step: bool = True
    byte1_index_dim: str = "BYTE_INDEX_1"


def _byte_default_marker_rules(
    spec: RegisterByteDefaultSpec,
    marker: str,
    *,
    name_infix: str,
) -> list[FFNRule]:
    """The MARKER (or MARKER-FIRST-STEP) default: at ``(marker ∧ ¬fresh_key)``
    write 0 to OUTPUT byte 0 (LO + HI). Scope ``"{marker} and not {fresh_key}"``.
    """
    ws = spec.write_scale
    rules: list[FFNRule] = []
    for band, dst in (("lo", spec.dst_lo), ("hi", spec.dst_hi)):
        rules.append(multi_way_and_rule(
            name=f"{spec.name_prefix}_{name_infix}_{band}",
            conditions=((marker, 1.0), (spec.fresh_key, -1.0)),
            threshold=spec.marker_threshold,
            writes=((f"{dst}+0", ws),),
            scope=f"{marker} and not {spec.fresh_key}",
        ))
    return rules


def _byte_default_index_rules(spec: RegisterByteDefaultSpec) -> list[FFNRule]:
    """The BYTE-INDEX defaults: for each k in ``byte_idx_default`` at
    ``(select_conditions ∧ BYTE_INDEX_k ∧ byte_idx_trailing_conditions)`` write
    0 to OUTPUT byte 0 (LO + HI). The byte-index term is placed BETWEEN the
    leading + trailing selector conditions to match the STACK0 hand layout."""
    ws = spec.write_scale
    rules: list[FFNRule] = []
    for byte_idx in spec.byte_idx_default:
        conds = (
            spec.select_conditions
            + ((f"BYTE_INDEX_{byte_idx}", 1.0),)
            + spec.byte_idx_trailing_conditions
        )
        for band, dst in (("lo", spec.dst_lo), ("hi", spec.dst_hi)):
            rules.append(multi_way_and_rule(
                name=f"{spec.name_prefix}_{spec.byte_idx_name}_{byte_idx}"
                     f"_default_{band}",
                conditions=conds,
                threshold=spec.byte_idx_default_threshold,
                writes=((f"{dst}+0", ws),),
            ))
    return rules


def _byte_default_byte1_first_step_rules(
    spec: RegisterByteDefaultSpec,
) -> list[FFNRule]:
    """The BYTE-1 FIRST-STEP band (SP/BP): at ``(select_conditions ∧
    BYTE_INDEX_1 ∧ ¬fresh_key)`` write the first-step landing — ``dst_lo[1]``
    (lo nibble 1), ``dst_hi[0]`` (hi nibble 0)."""
    ws = spec.write_scale
    conds = spec.select_conditions + (
        (spec.byte1_index_dim, 1.0), (spec.fresh_key, -1.0),
    )
    return [
        multi_way_and_rule(
            name=f"{spec.name_prefix}_byte_1_first_step_lo",
            conditions=conds,
            threshold=spec.byte_idx_default_threshold,
            writes=((f"{spec.dst_lo}+1", ws),),
        ),
        multi_way_and_rule(
            name=f"{spec.name_prefix}_byte_1_first_step_hi",
            conditions=conds,
            threshold=spec.byte_idx_default_threshold,
            writes=((f"{spec.dst_hi}+0", ws),),
        ),
    ]


def register_byte_defaults(spec: RegisterByteDefaultSpec) -> Tuple[FFNRule, ...]:
    """Derive one register's marker-gated byte-default band from ``spec``.

    Emits the 4 sub-bands (marker default, byte-index defaults, byte-1
    first-step, marker first-step) in the SAME order + names the hand-authored
    ``l3_ops._register_default_ffn_rules`` code used, so the derived form is
    byte-identical (proof: ``tools/_isa_golden_hash.py`` == ``91f55411``).

    Sub-bands are emitted only when their spec field is set (SP/BP use marker +
    byte-index + byte-1-first-step; STACK0 uses byte-index + marker-first-step),
    matching the two hand-authored layouts. This is the DATA the frame agent
    flagged: the SP/BP/STACK0 default-writer expressed as a table, not
    hand-sequenced FFN banks.
    """
    rules: list[FFNRule] = []
    if spec.marker is not None:
        rules.extend(_byte_default_marker_rules(
            spec, spec.marker, name_infix="marker_default"))
    rules.extend(_byte_default_index_rules(spec))
    if spec.byte1_first_step:
        rules.extend(_byte_default_byte1_first_step_rules(spec))
    if spec.marker_first_step is not None:
        rules.extend(_byte_default_marker_rules(
            spec, spec.marker_first_step, name_infix="first_step_default"))
    return tuple(rules)
