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

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from .building_blocks_dsl import multi_way_and_rule
from .ir import FFNRule
from .primitives import AO, AP, DeclarativeAttentionHeadSpec
from .ops.residual_band_registry import register_residual_band


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
    match_q_band: str
    match_k_band: str
    match_weight: float
    value_src_lo: str
    value_src_hi: str
    value_o_write_scale: float
    dump_emit_lo: str
    dump_emit_hi: str
    dump_write_scale: float
    dump_per_byte_marker: str
    dump_per_byte_marker_weight: float
    dump_marker_blockers: Tuple[Tuple[str, float], ...]
    dump_gate_conditions: Tuple[Tuple[str, float], ...]
    dump_opent_floor: float
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

    def __post_init__(self) -> None:
        if self.band_width % 2 != 0:
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
    #     module reload under test re-registers cleanly.
    register_residual_band(
        spec.band_name,
        spec.band_width,
        owner=spec.band_owner if spec.band_owner is not None else spec.name,
        flag=spec.band_flag,
        never_share=True,
    )

    half = spec.band_width // 2

    # (2) Carry head spec builder — UNCONDITIONAL. Reproduces the hand-built
    #     Q/K/V/O writes in the SAME append order.
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

        # Slots 0..(carry_byte_count-1): per-byte positional match. Q reads the
        # consuming row's ``match_q_band{k}``; K reads the prev row's
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
        # Q-side CONST write is emitted ONCE PER (slot) — in the hand-built BP
        # code the STACK0 group emits the four K writes first, then ONE Q
        # write. To reproduce that exact append order, emit all K writes for a
        # contiguous run of entries sharing a slot, then the single Q write.
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
        # ``value_src_hi[0..half-1]`` -> band[half..band_width-1]. V slots from
        # ``value_v_slot_base`` (far from the Q/K gate slots).
        v: list = []
        o: list = []
        vb = spec.value_v_slot_base
        for j in range(half):
            v.append(AP(vb + j, src_lo + j, 1.0))
            o.append(AO(band + j, vb + j, spec.value_o_write_scale))
        for j in range(half):
            v.append(AP(vb + half + j, src_hi + j, 1.0))
            o.append(AO(band + half + j, vb + half + j, spec.value_o_write_scale))

        return DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            alibi_slope=spec.carry_head_alibi_slope,
        )

    # (3) Dump rules builder — the GATE lives here. Per consuming byte k, per
    #     band cell j: ``emit_band[j] = band[j]`` gated on
    #     (dump_gate_conditions AND per_byte_marker{k}) over threshold, with
    #     the MARK_* blockers ANDed in. Returns () when emission off (flag-off
    #     byte-identical: the band is omitted from the layout so there are zero
    #     rules and the op is inert).
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

    # Reads/writes dim sets (flag-on). Derived structurally from the spec so
    # the op factory can declare them.
    carry_head_reads: Set[str] = set()
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
    carry_head_writes: Set[str] = {spec.band_name}

    dump_reads: Set[str] = set()
    for (dim_name, _w) in spec.dump_gate_conditions:
        dump_reads.add(dim_name)
    for kk in range(spec.carry_byte_count):
        dump_reads.add(f"{spec.dump_per_byte_marker}{kk}")
    for (dim_name, _w) in spec.dump_marker_blockers:
        dump_reads.add(dim_name)
    dump_reads.add(spec.band_name)
    dump_writes: Set[str] = {spec.dump_emit_lo, spec.dump_emit_hi, spec.band_name}

    return CrossStepCarryBundle(
        spec=spec,
        carry_head_spec_builder=carry_head_spec_builder,
        dump_rules_builder=dump_rules_builder,
        carry_head_reads=carry_head_reads,
        carry_head_writes=carry_head_writes,
        dump_reads=dump_reads,
        dump_writes=dump_writes,
    )
