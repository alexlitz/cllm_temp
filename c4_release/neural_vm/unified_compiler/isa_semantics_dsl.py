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
from .ir import FFNRule, TokenEmbeddingRule
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
    match.

    Attributes:
        query_dim: the QUERY-MARKER residual dim (the row the head fires on;
            L7: ``MARK_AX``).
        key_dim: the KEY-SIGNATURE residual dim the matched K row must carry
            (the content address; L7: ``STACK0_BYTE0``).
        weight: the shared Q/K projection weight (L7: ``15.0``).
        query_slot: head-local slot the match lands on (L7: ``0``).
    """

    query_dim: str
    key_dim: str
    weight: float
    query_slot: int = 0


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
        def _P(name: str) -> int:
            return int(dim_positions[name])

        # (1) The content-address row MATCH on the query slot: Q@query_dim,
        #     K@key_dim — the CAM invariant. ONE pair.
        q: list = [AP(km.query_slot, _P(km.query_dim), km.weight)]
        k: list = [AP(km.query_slot, _P(km.key_dim), km.weight)]

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

        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            alibi_slope=spec.alibi_slope,
        )

    # Operation dep-graph dim sets (structural derivation).
    head_reads: Set[str] = {km.query_dim, km.key_dim, spec.const_dim}
    for (op_dim, _w) in spec.query_blockers:
        head_reads.add(op_dim)
    if confirm is not None:
        if confirm.marker_dim is not None:
            head_reads.add(confirm.marker_dim)
        for (op_dim, _w) in confirm.blockers:
            head_reads.add(op_dim)
    head_writes: Set[str] = set()
    if spec.value_active:
        for vb in spec.value_bands:
            head_reads.add(vb.source_band)
            head_writes.add(vb.target_band)
    head_reads.update(spec.extra_reads)

    return CamLookupBundle(
        spec=spec,
        head_spec_builder=head_spec_builder,
        head_reads=head_reads,
        head_writes=head_writes,
    )
