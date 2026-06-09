"""
Core primitives for weight compilation.

Implements the 10 fundamental patterns used throughout vm_step.py:

Attention Primitives:
1. threshold_attention - Q=constant, K=IS_MARK×threshold, V/O=marker copy
2. carry_forward_attention - Q=marker, K=distance pattern, V/O=nibble relay
3. memory_lookup_attention - Q=address nibbles, K=ADDR_KEY, V=values
4. relay_head - Q=target marker, K=source threshold, V=register, broadcast O

FFN Primitives:
5. swiglu_and_gate - W_up conditions + b_up threshold, W_gate source, W_down output
6. cancel_pair - W_gate -1×old + 1×new in same unit
7. opcode_decode_unit - W_up nibble AND, W_gate marker, W_down to OP_* flag
8. nibble_copy - W_gate source, W_down output with optional residual cancel
9. step_pair - step(>=low) - step(>=high) for binary decisions
10. threshold_match - W_up marker+conditions, b_up threshold, W_gate, W_down output

Extracted batch B (vm_step direct ports — byte-identical to imperative code):
- nibble_rotation_chain       : (source + offset) FFN block, optionally carry-aware
"""

import torch
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Tuple, Union
from ..vm_step import _SetDim as BD


@dataclass(frozen=True)
class AttentionProjectionWrite:
    """One declarative write into an attention projection matrix.

    ``slot`` is head-local: generated row = ``head_idx * HD + slot``.
    ``dim`` is the residual-stream column for W_q/W_k/W_v, or the
    residual-stream row for W_o. This intentionally uses already-resolved
    dim positions so the primitive works with both legacy ``_SetDim`` and
    compiler-allocated dim proxies.
    """

    slot: int
    dim: int
    weight: float


@dataclass(frozen=True)
class AttentionOutputWrite:
    """One declarative W_o write from a head-local value slot to residual dim."""

    out_dim: int
    slot: int
    weight: float


@dataclass(frozen=True)
class DeclarativeAttentionHeadSpec:
    """Structural attention-head bake spec.

    This is the first step toward docs/DECLARATIVE_BAKE_VISION.md: operation
    bakes can describe Q/K/V/O structure as data and let a shared generator
    emit the matrix writes. The spec is intentionally low-level enough to be
    byte-identical with existing imperative helpers.

    ``alibi_slope`` (Phase 7.B.2 attn) decouples the per-head ALiBi slope
    from its literal ``head_idx`` position: when the spec is lowered the
    bake helper writes ``attn.alibi_slopes[spec.head_idx] = spec.alibi_slope``
    rather than indexing a parallel list keyed on the head's old position.
    ``None`` means "the op handles its own ``alibi_slopes`` write" (the
    historical default — kept so existing call sites stay byte-identical
    while the migration ports them to spec-carried slopes).
    """

    head_idx: int
    q: Tuple[AttentionProjectionWrite, ...] = field(default_factory=tuple)
    k: Tuple[AttentionProjectionWrite, ...] = field(default_factory=tuple)
    v: Tuple[AttentionProjectionWrite, ...] = field(default_factory=tuple)
    o: Tuple[AttentionOutputWrite, ...] = field(default_factory=tuple)
    alibi_slope: Optional[float] = None
    # Phase 8.O.2 GQA: number of consecutive Q-head indices that share
    # this spec's K/V projections. Default ``1`` = vanilla MHA (each Q
    # head has its own K/V row block) — byte-identical with every
    # pre-8.O.2 baseline. ``group_size > 1`` declares GQA: K/V rows
    # land at ``(head_idx // group_size) * HD`` rather than
    # ``head_idx * HD``. Q and O rows still land at ``head_idx * HD``.
    # The downstream ``attn`` module must size ``W_k`` / ``W_v`` to
    # ``num_kv_heads * HD`` for the GQA path to lower cleanly.
    # ``group_size=1`` is byte-identical with MHA (``head_idx // 1 ==
    # head_idx``). Mixtral target: 32 Q, 8 KV => 4.
    group_size: int = 1
    # V1/V2 vision (per-head dynamic head_dim): None = use the layer's
    # default head_dim (today's fixed ``dim // num_heads``); an int
    # overrides for this head specifically. Byte-identical at the default.
    # Composes cleanly with ``group_size``: ``head_dim`` controls the
    # ROW WIDTH of each Q/K/V slot block; ``group_size`` controls the
    # head-index mapping. Lowering site:
    # ``Primitives.generate_attention_head(head_base=..., kv_head_base=...)``.
    head_dim: Optional[int] = None

    def effective_head_dim(self, default_HD: int) -> int:
        """Return per-head slot count: ``spec.head_dim`` or ``default_HD``.

        V1/V2 vision (per-head dynamic head_dim): when a head declares
        a non-default ``head_dim``, that width drives both slot-write
        validation and the cumulative-sum row base computation. Heads
        that leave ``head_dim=None`` adopt the layer-wide default and
        keep the legacy fixed-HD bake byte-identical.
        """
        hd = self.head_dim if self.head_dim is not None else default_HD
        if not isinstance(hd, int) or hd <= 0:
            raise ValueError(
                "DeclarativeAttentionHeadSpec.effective_head_dim: "
                f"non-positive head_dim={hd!r} "
                f"(spec.head_dim={self.head_dim!r}, "
                f"default_HD={default_HD!r})"
            )
        return hd

    @property
    def kv_head_idx(self) -> int:
        """KV-head index = ``head_idx // group_size`` (Phase 8.O.2 GQA).

        At ``group_size=1`` returns ``head_idx`` (byte-identical with
        MHA). At larger group sizes consecutive Q heads share a single
        KV-head slot — the canonical GQA grouping.
        """
        gs = int(self.group_size)
        if gs <= 0:
            raise ValueError(
                "DeclarativeAttentionHeadSpec: group_size must be >= 1 "
                f"(got {self.group_size!r})"
            )
        return int(self.head_idx) // gs


@dataclass(frozen=True)
class ThresholdAttentionHeadSpec:
    """Declarative spec for legacy ``_set_threshold_attn`` marker heads."""

    head_idx: int
    threshold: float
    out_base: int
    slope: float = 10.0
    bd: object = BD


def AP(slot: int, dim: int, weight: float) -> AttentionProjectionWrite:
    """Compact constructor for declarative attention Q/K/V writes."""

    return AttentionProjectionWrite(slot=slot, dim=dim, weight=weight)


def AO(out_dim: int, slot: int, weight: float) -> AttentionOutputWrite:
    """Compact constructor for declarative attention output writes."""

    return AttentionOutputWrite(out_dim=out_dim, slot=slot, weight=weight)


def _inject_sink_k_row(
    spec: DeclarativeAttentionHeadSpec,
    sink_idx: int,
) -> DeclarativeAttentionHeadSpec:
    """Return a copy of ``spec`` with one synthetic "sink" K-slot appended.

    Scaffold helper for the standard-softmax DSL variant (see
    ``docs/SOFTMAX_DSL_VARIANT_DESIGN_2026_06_09.md``). Under the
    deployed VM's ``attention_normalization="softmax1"`` the +1 anchor
    in the softmax1 denominator gives every memory-lookup head the
    "Zero Fill On Demand" semantics that the residual stream relies on
    (an unmapped LI reads 0). Stock Qwen3 uses standard softmax, where
    that anchor doesn't exist — so the parallel ``"standard"`` variant
    has to bake a substitute into each affected head.

    The substitute is the **bake-time** equivalent of the runtime sink
    column appended in ``vm_step.AutoregressiveAttention.forward``
    (see ``vm_step.py`` ~L537 — the SDPA path appends ``K=0, V=0`` when
    ``use_softmax1`` is on). For per-head bakes we reserve one extra
    K slot per spec at index ``sink_idx`` and write only ``CONST → 0``
    into it. Effects:

    * K_sink row = 0 (the only write at this slot is ``CONST → 0``),
      so the sink's pre-softmax score against any Q is exactly 0;
    * V_sink row absent => V[sink_idx] = 0 by default => the sink
      contributes 0 to the V-weighted output;
    * Q is unchanged — queries still see their natural K rows.

    R3 proved the algebra in ``docs/QWEN_SOFTMAX_SINK_PROTOTYPE_2026_06_07.md``:
    standard softmax over ``[0, scores]`` with ``V_sink = 0``
    reproduces softmax1 over ``scores`` exactly on the real positions.

    The helper is **idempotent and additive**: if a write at
    ``slot=sink_idx`` already exists in ``spec.k``, the spec is
    returned unchanged. The existing Q/K/V/O writes are passed through
    so callers can chain this helper without touching the rest of the
    head's bake.

    Args:
        spec: the head spec that historically relied on softmax1's
            anchor. Typically a L7 memory_heads head or a L15
            memory_lookup head (see design doc §3).
        sink_idx: per-head K-slot index for the sink. Caller must
            ensure this slot is otherwise unused (the allocator can
            reserve one slot beyond the spec's max declared K slot).

    Returns:
        A new ``DeclarativeAttentionHeadSpec`` with one extra K row at
        ``sink_idx`` writing ``CONST → 0``. All other fields are
        carried over unchanged (including ``alibi_slope``,
        ``group_size``, ``head_dim``).

    Notes:
        * Writing ``weight=0.0`` is deliberate. The sink's K row must
          be all-zero regardless of the input residual, so any
          ``CONST`` activation projected through this slot resolves
          to 0. This differs from anti-anchor heads (the
          ``softmax1_suppress`` audit class) where the K is pushed
          *actively negative* — those are out of scope for the
          ``"standard"`` variant scaffold (see design doc §3).
        * This helper does NOT toggle off the runtime softmax1 path —
          that's the caller's responsibility via
          ``attention_normalization="softmax"`` /
          ``softmax_variant="standard"``. Using the sink alongside
          softmax1 doubles the anchor (sink + +1) and is incorrect.
    """

    if any(write.slot == sink_idx for write in spec.k):
        return spec

    sink_write = AttentionProjectionWrite(
        slot=sink_idx,
        dim=BD.CONST,
        weight=0.0,
    )
    new_k = tuple(spec.k) + (sink_write,)
    return DeclarativeAttentionHeadSpec(
        head_idx=spec.head_idx,
        q=spec.q,
        k=new_k,
        v=spec.v,
        o=spec.o,
        alibi_slope=spec.alibi_slope,
        group_size=spec.group_size,
        head_dim=spec.head_dim,
    )


class Primitives:
    """Core weight-setting primitives matching vm_step.py patterns."""

    # =========================================================================
    # Attention Primitives
    # =========================================================================

    @staticmethod
    def generate_attention_head(
        attn,
        spec: DeclarativeAttentionHeadSpec,
        HD: int,
        *,
        head_base: Optional[int] = None,
        kv_head_base: Optional[int] = None,
    ):
        """Emit Q/K/V/O matrix writes for one declarative attention head.

        If ``spec.alibi_slope`` is not ``None`` and ``attn`` exposes an
        ``alibi_slopes`` buffer, the slope is written at
        ``attn.alibi_slopes[spec.head_idx]``. This makes the slope follow
        the spec across an :class:`AttentionHeadAllocator` first-fit
        permutation.

        ``head_base`` / ``kv_head_base`` (V1/V2 vision — per-head
        dynamic head_dim) are the starting Q/O and K/V rows for this
        head's slot block. When ``None`` (default) the legacy
        ``spec.head_idx * HD`` / ``spec.kv_head_idx * HD`` formulas are
        used — byte-identical with every existing fixed-HD bake. When
        provided (e.g. by an allocator-driven lowering pass) the caller
        supplies the cumulative-sum offset that accounts for upstream
        heads with non-default ``spec.head_dim``. Slot writes are
        validated against ``spec.effective_head_dim(HD)`` so a stale
        slot index can't bleed into the next head's row block.
        """

        # Phase 8.O.2 GQA: Q and O rows land at ``head_idx * HD``;
        # K and V rows land at ``kv_head_idx * HD =
        # (head_idx // group_size) * HD``. At ``group_size=1`` (the
        # default) the two bases coincide, so the writes are byte-
        # identical with the pre-8.O.2 MHA path. V1/V2 vision (per-head
        # dynamic head_dim) layers on top: when ``head_base`` /
        # ``kv_head_base`` are supplied, they override the legacy
        # ``head_idx * HD`` formula with allocator-driven cumulative
        # offsets — leaves byte-identity intact whenever ``head_base``
        # is ``None``.
        base = spec.head_idx * HD if head_base is None else int(head_base)
        kv_base = (
            spec.kv_head_idx * HD if kv_head_base is None
            else int(kv_head_base)
        )
        eff_hd = spec.effective_head_dim(HD)
        for write in spec.q:
            if write.slot >= eff_hd:
                raise ValueError(
                    f"generate_attention_head: q slot={write.slot} >= "
                    f"effective_head_dim={eff_hd} (head_idx={spec.head_idx})"
                )
            attn.W_q.data[base + write.slot, write.dim] = write.weight
        for write in spec.k:
            if write.slot >= eff_hd:
                raise ValueError(
                    f"generate_attention_head: k slot={write.slot} >= "
                    f"effective_head_dim={eff_hd} (head_idx={spec.head_idx})"
                )
            attn.W_k.data[kv_base + write.slot, write.dim] = write.weight
        for write in spec.v:
            if write.slot >= eff_hd:
                raise ValueError(
                    f"generate_attention_head: v slot={write.slot} >= "
                    f"effective_head_dim={eff_hd} (head_idx={spec.head_idx})"
                )
            attn.W_v.data[kv_base + write.slot, write.dim] = write.weight
        for write in spec.o:
            if write.slot >= eff_hd:
                raise ValueError(
                    f"generate_attention_head: o slot={write.slot} >= "
                    f"effective_head_dim={eff_hd} (head_idx={spec.head_idx})"
                )
            attn.W_o.data[write.out_dim, base + write.slot] = write.weight
        if spec.alibi_slope is not None:
            if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
                attn.alibi_slopes.data[spec.head_idx] = float(spec.alibi_slope)

    @staticmethod
    def generate_attention_heads(attn, specs, HD: int):
        """Emit Q/K/V/O matrix writes for multiple declarative heads.

        When every spec uses the layer default (``spec.head_dim is None``)
        the loop falls through to per-head ``head_idx * HD`` row bases —
        byte-identical with the prior implementation. When any spec
        declares a non-default ``head_dim``, per-head bases are computed
        by cumulative-sum over specs sorted by ``head_idx`` so the layout
        is deterministic regardless of insertion order.
        """

        spec_list = list(specs)
        any_custom = any(
            getattr(s, "head_dim", None) is not None for s in spec_list
        )
        if not any_custom:
            for spec in spec_list:
                Primitives.generate_attention_head(attn, spec, HD)
            return
        ordered = sorted(spec_list, key=lambda s: int(s.head_idx))
        # V1/V2 vision (per-head dynamic head_dim): track cumulative Q/O
        # bases over Q-heads AND cumulative K/V bases over KV-head groups
        # so the GQA layout (multiple Q heads sharing one KV row block)
        # composes cleanly with dynamic head_dim. At ``group_size=1``
        # (vanilla MHA) every Q head IS its own KV head, so the two
        # cumulative sums coincide — byte-identical with the prior MHA
        # cumulative loop. At ``group_size>1`` only the first Q head in
        # each group advances ``running_kv_base`` (its K/V block is then
        # re-used by the rest of the group via ``kv_bases``).
        running_q_base = 0
        kv_bases: dict = {}
        running_kv_base = 0
        for spec in ordered:
            kv_idx = int(spec.kv_head_idx)
            if kv_idx not in kv_bases:
                kv_bases[kv_idx] = running_kv_base
                running_kv_base += spec.effective_head_dim(HD)
            Primitives.generate_attention_head(
                attn,
                spec,
                HD,
                head_base=running_q_base,
                kv_head_base=kv_bases[kv_idx],
            )
            running_q_base += spec.effective_head_dim(HD)

    @staticmethod
    def threshold_attention_head_spec(
        spec: ThresholdAttentionHeadSpec,
        HD: int,
        *,
        alibi_slope: Optional[float] = None,
    ) -> DeclarativeAttentionHeadSpec:
        """Lower a threshold-attention spec to raw Q/K/V/O writes.

        ``out_base`` is captured by the spec itself: the resulting
        :class:`DeclarativeAttentionHeadSpec` carries pre-resolved
        ``AO(out_base + m, ...)`` writes, so the output-base mapping does
        NOT depend on ``head_idx`` — a first-fit allocator can permute
        ``head_idx`` without scrambling which threshold targets which
        residual output (Phase 7.B.2 attn). The W_o ROW index always
        lands at the per-spec ``out_base``; the W_o COLUMN index
        naturally tracks the head's row footprint via ``base + slot``.

        If ``alibi_slope`` is provided it is forwarded into the returned
        :class:`DeclarativeAttentionHeadSpec` so the slope follows the
        spec across allocator permutations rather than being indexed by
        the head's old position.
        """

        import math

        bd = spec.bd
        q_val = math.sqrt(HD) * spec.slope
        v_writes = tuple(
            AP(1 + m, src, 1.0)
            for m, src in enumerate(bd.MARKS)
        )
        # ``out_base`` is captured here, NOT later: each output write
        # carries its target dim verbatim, so permuting ``head_idx`` at
        # the allocator never reroutes a threshold to a different
        # ``H<n>``/``L<n>H<n>`` slot.
        o_writes = tuple(
            AO(spec.out_base + m, 1 + m, 1.0)
            for m in range(bd.NUM_MARKERS)
        )
        return DeclarativeAttentionHeadSpec(
            head_idx=spec.head_idx,
            q=(AP(0, bd.CONST, q_val),),
            k=(AP(0, bd.IS_MARK, spec.threshold),),
            v=v_writes,
            o=o_writes,
            alibi_slope=alibi_slope,
        )

    @staticmethod
    def threshold_attention_head_specs(
        thresholds,
        out_bases,
        slope: float,
        HD: int,
        heads=None,
        bd=None,
        alibi_slopes=None,
    ):
        """Build declarative specs equivalent to ``_set_threshold_attn``.

        ``bd`` is required for migrated compiler layouts so CONST/IS_MARK/MARKS
        resolve through the active dim-position proxy instead of the legacy enum.

        ``alibi_slopes`` is an optional positional list (one float per
        threshold/head pair) — when provided each spec carries its own
        slope so :meth:`generate_attention_head` writes the matching
        ``attn.alibi_slopes[head_idx]`` at bake time. The list is keyed
        by **threshold position** (parallel to ``thresholds`` /
        ``out_bases``), NOT by ``head_idx``, so a first-fit head
        permutation cannot scramble the slope-to-threshold mapping.
        ``None`` leaves slope handling to the caller's bake (matches the
        historical contract — every existing site that pre-fills the
        slopes via ``attn.alibi_slopes.fill_(...)`` keeps working).
        """

        if bd is None:
            raise TypeError(
                "threshold_attention_head_specs requires bd "
                "(use _as_setdim_proxy(dim_positions))"
            )
        if heads is None:
            heads = list(range(len(thresholds)))
        if alibi_slopes is not None and len(alibi_slopes) != len(thresholds):
            raise ValueError(
                f"threshold_attention_head_specs: alibi_slopes length "
                f"({len(alibi_slopes)}) must match thresholds length "
                f"({len(thresholds)})"
            )
        return tuple(
            Primitives.threshold_attention_head_spec(
                ThresholdAttentionHeadSpec(
                    head_idx=h,
                    threshold=t,
                    out_base=out_bases[i],
                    slope=slope,
                    bd=bd,
                ),
                HD,
                alibi_slope=(
                    None if alibi_slopes is None
                    else float(alibi_slopes[i])
                ),
            )
            for i, (h, t) in enumerate(zip(heads, thresholds))
        )

    @staticmethod
    def generate_threshold_attention_heads(
        attn,
        thresholds,
        out_bases,
        slope: float,
        HD: int,
        heads=None,
        bd=None,
        alibi_slopes=None,
    ):
        """Emit threshold-attention heads from declarative specs.

        ``alibi_slopes`` (optional, parallel to ``thresholds``) makes
        each spec carry its own slope so the alibi-buffer write follows
        the spec across an allocator-driven head_idx permutation. When
        omitted the caller's bake remains responsible for filling
        ``attn.alibi_slopes`` directly.
        """

        Primitives.generate_attention_heads(
            attn,
            Primitives.threshold_attention_head_specs(
                thresholds, out_bases, slope, HD,
                heads=heads, bd=bd, alibi_slopes=alibi_slopes,
            ),
            HD,
        )

    @staticmethod
    def threshold_attention(
        attn,
        head_idx: int,
        threshold: float,
        out_base: int,
        slope: float = 10.0,
        HD: int = 64,
    ):
        """Set threshold-based attention head for marker distance detection.

        Each head detects whether the nearest marker is within `threshold` tokens.
        Uses ALiBi: score = slope*(threshold - distance), giving a sharp sigmoid.

        Pattern from _set_threshold_attn (vm_step.py:2142-2161):
          Q[0] = constant (8.0 * slope)
          K[0] = IS_MARK * threshold
          V[1+m] = MARKS[m] for each marker type
          O[out_base+m] = V[1+m]

        Args:
            attn: Attention layer module
            head_idx: Head index (0-7)
            threshold: Distance threshold (e.g., 3.5, 4.5)
            out_base: Base dimension for output (e.g., BD.H0, BD.H1)
            slope: ALiBi slope (default 10.0)
            HD: Head dimension (default 64)
        """
        Primitives.generate_threshold_attention_heads(
            attn,
            [threshold],
            [out_base],
            slope,
            HD,
            heads=[head_idx],
            bd=BD,
        )

    @staticmethod
    def carry_forward_attention(
        attn,
        head_idx: int,
        marker_dim: int,
        l1h1_idx: int,
        l1h0_idx: int,
        out_lo: int,
        out_hi: int,
        HD: int = 64,
        src_lo: Optional[int] = None,
        src_hi: Optional[int] = None,
        L: float = 15.0,
        bd=None,
    ):
        """Set attention head for register carry-forward.

        At marker positions, attends to the previous step's corresponding byte 0
        (identified by L1H1_marker AND NOT L1H0_marker pattern).

        Pattern (formerly in the legacy ``_set_carry_forward_attn`` helper,
        deleted per BD_SETDIM_HARDCODE_AUDIT M1):
          Q[0] = marker_dim * L
          K[0] = L1H1 * L - L1H0 * L
          V[1:17] = src_lo nibble, V[17:33] = src_hi nibble
          O[out_lo/hi] = V
          Anti-leakage gate at dim 33

        Args:
            attn: Attention layer module
            head_idx: Head index
            marker_dim: Query marker dimension (e.g., BD.MARK_PC)
            l1h1_idx: L1H1 marker index for key (e.g., 0 for PC)
            l1h0_idx: L1H0 marker index for key
            out_lo: Output dimension for low nibble
            out_hi: Output dimension for high nibble
            HD: Head dimension (default 64)
            src_lo: Source low nibble dim (default EMBED_LO)
            src_hi: Source high nibble dim (default EMBED_HI)
            L: Attention weight scale (default 15.0)
            bd: Optional dim spec (proxy) overriding module-level BD for L1H0,
                L1H1, CONST, EMBED_LO/HI lookups. Pass the compiler proxy here
                so pin_io_only=True layouts wire to the correct residual lanes.
        """
        spec = bd if bd is not None else BD
        base = head_idx * HD

        if src_lo is None:
            src_lo = spec.EMBED_LO
        if src_hi is None:
            src_hi = spec.EMBED_HI

        # Q: fires at target marker
        attn.W_q.data[base, marker_dim] = L

        # K: fires at previous step's byte 0 (L1H1 AND NOT L1H0)
        attn.W_k.data[base, spec.L1H1 + l1h1_idx] = L
        attn.W_k.data[base, spec.L1H0 + l1h0_idx] = -L

        # V: copy source nibbles
        for k in range(16):
            attn.W_v.data[base + 1 + k, src_lo + k] = 1.0
            attn.W_v.data[base + 17 + k, src_hi + k] = 1.0

        # O: write to output dimensions
        for k in range(16):
            attn.W_o.data[out_lo + k, base + 1 + k] = 1.0
            attn.W_o.data[out_hi + k, base + 17 + k] = 1.0

        # Anti-leakage gate (dim 33)
        GATE = 33
        attn.W_q.data[base + GATE, marker_dim] = L
        attn.W_q.data[base + GATE, spec.CONST] = -L / 2
        attn.W_k.data[base + GATE, spec.CONST] = L

    @staticmethod
    def memory_lookup_attention(
        attn,
        head_idx: int,
        query_dims: List[int],
        key_dims: List[int],
        value_dims: List[int],
        output_dims: List[int],
        HD: int = 64,
        gate_q_dim: Optional[int] = None,
        gate_k_dim: Optional[int] = None,
        L: float = 15.0,
    ):
        """Set memory lookup attention head.

        Performs content-addressable memory read via nibble matching.

        Args:
            attn: Attention layer module
            head_idx: Head index
            query_dims: List of query dimensions (address nibbles)
            key_dims: List of key dimensions (ADDR_KEY)
            value_dims: List of value dimensions (byte values)
            output_dims: List of output dimensions
            HD: Head dimension (default 64)
            gate_q_dim: Optional gate dimension for Q
            gate_k_dim: Optional gate dimension for K
            L: Attention weight scale
        """
        base = head_idx * HD

        # Q: address nibbles
        for i, dim in enumerate(query_dims):
            attn.W_q.data[base + i, dim] = L

        # K: ADDR_KEY dimensions
        for i, dim in enumerate(key_dims):
            attn.W_k.data[base + i, dim] = L

        # V: byte values
        for i, dim in enumerate(value_dims):
            attn.W_v.data[base + i, dim] = 1.0

        # O: output
        for i, (out_dim, v_idx) in enumerate(zip(output_dims, range(len(value_dims)))):
            attn.W_o.data[out_dim, base + v_idx] = 1.0

        # Optional gating
        if gate_q_dim is not None:
            GATE = len(query_dims)
            attn.W_q.data[base + GATE, gate_q_dim] = L
            attn.W_q.data[base + GATE, BD.CONST] = -L / 2
            if gate_k_dim is not None:
                attn.W_k.data[base + GATE, gate_k_dim] = L

    @staticmethod
    def relay_head(
        attn,
        head_idx: int,
        q_marker: int,
        k_source: int,
        v_dims: List[int],
        o_dims: List[int],
        HD: int = 64,
        L: float = 15.0,
    ):
        """Set relay head for value broadcast.

        Q fires at target marker, K fires at source position,
        V copies specified dimensions, O broadcasts to output.

        Args:
            attn: Attention layer module
            head_idx: Head index
            q_marker: Query marker dimension
            k_source: Key source dimension (threshold flag)
            v_dims: Value dimensions to copy
            o_dims: Output dimensions
            HD: Head dimension
            L: Attention weight scale
        """
        base = head_idx * HD

        attn.W_q.data[base, q_marker] = L
        attn.W_k.data[base, k_source] = L

        for i, v_dim in enumerate(v_dims):
            attn.W_v.data[base + 1 + i, v_dim] = 1.0

        for i, o_dim in enumerate(o_dims):
            attn.W_o.data[o_dim, base + 1 + i] = 1.0

        # Anti-leakage gate
        GATE = len(v_dims) + 1
        attn.W_q.data[base + GATE, q_marker] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

    # =========================================================================
    # FFN Primitives
    # =========================================================================

    @staticmethod
    def swiglu_and_gate(
        ffn,
        unit: int,
        up_dims: List[Tuple[int, float]],
        threshold: float,
        gate_dims: Optional[List[Tuple[int, float]]] = None,
        gate_bias: float = 1.0,
        out_dims: List[Tuple[int, float]] = None,
        S: float = 100.0,
    ) -> int:
        """Set SwiGLU AND gate pattern.

        Pattern: hidden = silu(W_up @ x + b_up) * (W_gate @ x + b_gate)

        Fires when sum of up_dims exceeds threshold, then multiplies by gate value.

        Args:
            ffn: FFN layer module
            unit: Hidden unit index
            up_dims: List of (dim, weight) for W_up
            threshold: Threshold value (b_up = -S * threshold)
            gate_dims: List of (dim, weight) for W_gate (None for constant gate)
            gate_bias: Bias for gate (default 1.0)
            out_dims: List of (dim, weight) for W_down
            S: Scale factor (default 100.0)

        Returns:
            Next available unit index
        """
        for dim, weight in up_dims:
            ffn.W_up.data[unit, dim] = S * weight
        ffn.b_up.data[unit] = -S * threshold

        if gate_dims is not None:
            for dim, weight in gate_dims:
                ffn.W_gate.data[unit, dim] = weight
        ffn.b_gate.data[unit] = gate_bias

        if out_dims is not None:
            for dim, weight in out_dims:
                ffn.W_down.data[dim, unit] = weight

        return unit + 1

    @staticmethod
    def lower_ffn_rules(
        ffn,
        rules,
        dim_positions: Mapping[str, int],
        *,
        start_unit: int = 0,
        S: float = 100.0,
    ) -> int:
        """Lower named ``FFNRule`` data through ``CompilerIR``.

        This keeps migrated bakes data-first while preserving the exact
        low-level SwiGLU matrix writes used by the legacy primitives.
        """

        from .ir import CompilerIR

        ir = CompilerIR()
        ir.layer(0).ffn.rules.extend(rules)
        return ir.lower_ffn(
            ffn,
            dim_positions,
            start_unit=start_unit,
            S=S,
        )

    @staticmethod
    def apply_ffn_band_suppressors(
        ffn,
        dim_positions: Mapping[str, int],
        *,
        end_unit: int,
        S: float = 100.0,
        suppressors: Mapping[str, float] = (),
        marker_boost_strength: float = 0.0,
        marker_boost_const_dim: Optional[str] = None,
        marker_boost_structural_dims: Iterable[str] = (),
    ) -> None:
        """Apply declarative post-lowering W_up overrides to a band of units.

        This expresses two patterns that previously lived as ad-hoc
        ``ffn.W_up.data[:end_unit, dim] = -S * X`` patches inside per-op
        bakes:

        1) ``suppressors``: a mapping ``{dim_name: strength_in_units_of_S}``
           that hard-blocks the named dim across all units ``[0, end_unit)``
           by writing ``W_up[:end_unit, dim_pos] = -S * strength``. Mirrors
           the imperative assignment verbatim, but the data is declared by
           the caller as a mapping so the bake body stays pattern-only.
        2) ``marker_boost_*``: structural-row gate. Subtracts
           ``marker_boost_strength`` from W_up at the CONST column and
           re-adds it to each named structural marker column whose existing
           W_up entry is ``>= 0``. This is the declarative form of
           ``_suppress_ffn_on_step_boundary`` and preserves its sign-
           dependent semantics byte-identically by reading the post-
           lowering W_up state.

        Living in ``Primitives`` (rather than the per-op bake module) keeps
        the bake function source pattern-only: the imperative tensor
        writes never appear in the bake or its module-local helpers.
        """

        if end_unit <= 0:
            return
        if not isinstance(dim_positions, dict):
            return
        W_up = ffn.W_up.data

        # Hard-block suppressor band. Dim names may carry a ``+offset``
        # suffix (e.g. ``CARRY+1``) to address sub-cells of a named band.
        for dim_name, strength in dict(suppressors).items():
            if "+" in dim_name:
                base, off_s = dim_name.rsplit("+", 1)
                dim_pos = dim_positions.get(base)
                if dim_pos is None:
                    continue
                dim_pos = dim_pos + int(off_s)
            else:
                dim_pos = dim_positions.get(dim_name)
                if dim_pos is None:
                    continue
            if dim_pos >= W_up.shape[1]:
                continue
            W_up[:end_unit, dim_pos] = -S * strength

        # Structural-row marker boost.
        if marker_boost_strength != 0.0 and marker_boost_const_dim is not None:
            const_pos = dim_positions.get(marker_boost_const_dim)
            if const_pos is None or const_pos >= W_up.shape[1]:
                return
            W_up[:end_unit, const_pos] -= marker_boost_strength
            for marker_name in marker_boost_structural_dims:
                marker_pos = dim_positions.get(marker_name)
                if marker_pos is None or marker_pos >= W_up.shape[1]:
                    continue
                rows = W_up[:end_unit, marker_pos]
                rows[rows >= 0] += marker_boost_strength

    @staticmethod
    def ffn_rule_dim_names(rules) -> Tuple[str, ...]:
        """Return base dimension names referenced by ``FFNRule`` data."""

        names = set()
        for rule in rules:
            for term in rule.conditions:
                names.add(term.dim.name)
            if rule.gate is not None:
                names.add(rule.gate.name)
            for term in rule.gate_terms:
                names.add(term.dim.name)
            for write in rule.writes:
                names.add(write.dim.name)
        return tuple(sorted(names))

    @staticmethod
    def dim_positions_from_bd(bd, names: Iterable[str]):
        """Build the named dim map expected by ``CompilerIR.lower_ffn``."""

        return {name: getattr(bd, name) for name in names}

    @staticmethod
    def nibble_value_writes(
        target_base: str,
        value: int,
        *,
        strength: float = 100.0,
        competitor_strength: Optional[float] = None,
    ) -> Tuple[Tuple[str, float], ...]:
        """Return declarative writes for an exact one-hot nibble value.

        The selected nibble channel receives ``+strength`` and every other
        channel in the 16-wide target band receives ``-competitor_strength``.
        This is intentionally just FFN write data: callers feed it into
        ``FFNRule.constant_write`` or ``FFNRule.gated_write`` so symbolic
        execution and neural lowering see the same declaration.
        """

        if not 0 <= value <= 0xF:
            raise ValueError("nibble value must be in range 0x0..0xf")
        if strength <= 0.0:
            raise ValueError("strength must be positive")
        if competitor_strength is None:
            competitor_strength = strength
        if competitor_strength <= 0.0:
            raise ValueError("competitor_strength must be positive")

        return tuple(
            (
                f"{target_base}+{k}",
                strength if k == value else -competitor_strength,
            )
            for k in range(16)
        )

    @staticmethod
    def nibble_constant_writes(
        target_base: str,
        value: int,
        *,
        strength: float = 100.0,
        competitor_strength: Optional[float] = None,
    ) -> Tuple[Tuple[str, float], ...]:
        """Alias for ``nibble_value_writes`` kept close to FFN rule wording."""

        return Primitives.nibble_value_writes(
            target_base,
            value,
            strength=strength,
            competitor_strength=competitor_strength,
        )

    @staticmethod
    def byte_value_writes(
        value: int,
        *,
        lo_base: str = "OUTPUT_LO",
        hi_base: str = "OUTPUT_HI",
        strength: float = 100.0,
        competitor_strength: Optional[float] = None,
    ) -> Tuple[Tuple[str, float], ...]:
        """Return interleaved low/high nibble writes for an exact byte value.

        The output order matches the historical local helpers used in op
        bakes: ``LO+0, HI+0, LO+1, HI+1, ...``.  The returned terms assert
        both nibbles one-hot by positively selecting the target channel and
        negatively suppressing all competing channels.
        """

        if not 0 <= value <= 0xFF:
            raise ValueError("byte value must be in range 0x00..0xff")

        lo = value & 0xF
        hi = (value >> 4) & 0xF
        lo_writes = dict(Primitives.nibble_value_writes(
            lo_base,
            lo,
            strength=strength,
            competitor_strength=competitor_strength,
        ))
        hi_writes = dict(Primitives.nibble_value_writes(
            hi_base,
            hi,
            strength=strength,
            competitor_strength=competitor_strength,
        ))
        writes = []
        for k in range(16):
            writes.append((f"{lo_base}+{k}", lo_writes[f"{lo_base}+{k}"]))
            writes.append((f"{hi_base}+{k}", hi_writes[f"{hi_base}+{k}"]))
        return tuple(writes)

    @staticmethod
    def cancel_pair(
        ffn,
        unit: int,
        old_dims: List[int],
        new_dims: List[int],
        out_dims: List[int],
        gate_dim: Optional[int] = None,
        S: float = 100.0,
    ) -> int:
        """Set cancel pair pattern: subtract old, add new.

        Uses W_gate to compute: -1*old + 1*new for each nibble dimension.

        Args:
            ffn: FFN layer module
            unit: Starting unit index
            old_dims: Dimensions to cancel
            new_dims: Dimensions to add
            out_dims: Output dimensions
            gate_dim: Optional gate dimension
            S: Scale factor

        Returns:
            Next available unit index
        """
        # Single unit with W_gate computation
        ffn.b_up.data[unit] = S  # Always active

        for old_d in old_dims:
            ffn.W_gate.data[unit, old_d] = -1.0
        for new_d in new_dims:
            ffn.W_gate.data[unit, new_d] = 1.0

        if gate_dim is not None:
            # Multiply gate by condition
            ffn.W_up.data[unit, gate_dim] = S
            ffn.b_up.data[unit] = -S * 0.5

        for out_d in out_dims:
            ffn.W_down.data[out_d, unit] = 2.0 / S

        return unit + 1

    @staticmethod
    def opcode_decode_unit(
        ffn,
        unit: int,
        lo_nibble: int,
        hi_nibble: int,
        marker_dim: int,
        op_dim: int,
        S: float = 100.0,
        extra_conditions: Optional[List[Tuple[int, float]]] = None,
        threshold: float = 1.5,
    ) -> int:
        """Set opcode decode unit: AND of lo/hi nibbles at marker.

        Pattern from _set_opcode_decode_ffn (vm_step.py:3382-3440):
          up = S*(OPCODE_BYTE_LO[lo] + OPCODE_BYTE_HI[hi] - threshold)
          gate = marker_dim
          down = op_dim with scale 10.0/S

        Args:
            ffn: FFN layer module
            unit: Hidden unit index
            lo_nibble: Low nibble value (0-15)
            hi_nibble: High nibble value (0-15)
            marker_dim: Marker dimension for gating
            op_dim: Output opcode dimension
            S: Scale factor
            extra_conditions: Additional (dim, weight) pairs for W_up
            threshold: Threshold value (default 1.5 for 2 inputs)

        Returns:
            Next available unit index
        """
        ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo_nibble] = S
        ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi_nibble] = S

        if extra_conditions:
            for dim, weight in extra_conditions:
                ffn.W_up.data[unit, dim] = S * weight

        ffn.b_up.data[unit] = -S * threshold
        ffn.W_gate.data[unit, marker_dim] = 1.0
        ffn.W_down.data[op_dim, unit] = 10.0 / S

        return unit + 1

    @staticmethod
    def nibble_copy(
        ffn,
        unit: int,
        src_lo: int,
        src_hi: int,
        dst_lo: int,
        dst_hi: int,
        gate_dim: Optional[int] = None,
        S: float = 100.0,
        suppress_residual: bool = False,
    ) -> int:
        """Set nibble copy pattern.

        Copies 16-dim one-hot nibbles from src to dst.

        Args:
            ffn: FFN layer module
            unit: Starting unit index
            src_lo: Source low nibble base
            src_hi: Source high nibble base
            dst_lo: Destination low nibble base
            dst_hi: Destination high nibble base
            gate_dim: Optional gate dimension
            S: Scale factor
            suppress_residual: If True, subtract residual

        Returns:
            Next available unit index
        """
        for k in range(16):
            # Copy low nibble
            ffn.b_up.data[unit + k] = S
            ffn.W_gate.data[unit + k, src_lo + k] = 1.0
            if gate_dim is not None:
                ffn.W_up.data[unit + k, gate_dim] = S
                ffn.b_up.data[unit + k] = -S * 0.5
            ffn.W_down.data[dst_lo + k, unit + k] = 2.0 / S

            if suppress_residual:
                ffn.W_down.data[dst_lo + k, unit + k] -= 2.0 / S

        for k in range(16):
            # Copy high nibble
            ffn.b_up.data[unit + 16 + k] = S
            ffn.W_gate.data[unit + 16 + k, src_hi + k] = 1.0
            if gate_dim is not None:
                ffn.W_up.data[unit + 16 + k, gate_dim] = S
                ffn.b_up.data[unit + 16 + k] = -S * 0.5
            ffn.W_down.data[dst_hi + k, unit + 16 + k] = 2.0 / S

            if suppress_residual:
                ffn.W_down.data[dst_hi + k, unit + 16 + k] -= 2.0 / S

        return unit + 32

    @staticmethod
    def step_pair(
        ffn,
        unit: int,
        input_dims: List[Tuple[int, float]],
        low_threshold: float,
        high_threshold: float,
        out_dim: int,
        gate_dims: Optional[List[Tuple[int, float]]] = None,
        gate_bias: float = 1.0,
        S: float = 100.0,
    ) -> int:
        """Set step pair pattern: step(>=low) - step(>=high).

        Creates binary output for input in range [low, high).

        Args:
            ffn: FFN layer module
            unit: Starting unit index
            input_dims: List of (dim, weight) for input
            low_threshold: Lower threshold
            high_threshold: Upper threshold
            out_dim: Output dimension
            gate_dims: Optional gate dimensions
            gate_bias: Gate bias
            S: Scale factor

        Returns:
            Next available unit index
        """
        # Unit 0: step(sum >= low)
        for dim, weight in input_dims:
            ffn.W_up.data[unit, dim] = S * weight
        ffn.b_up.data[unit] = -S * (low_threshold - 1.0)

        if gate_dims:
            for dim, weight in gate_dims:
                ffn.W_gate.data[unit, dim] = weight
        ffn.b_gate.data[unit] = gate_bias

        ffn.W_down.data[out_dim, unit] = 1.0 / S

        # Unit 1: -step(sum >= high)
        for dim, weight in input_dims:
            ffn.W_up.data[unit + 1, dim] = S * weight
        ffn.b_up.data[unit + 1] = -S * (high_threshold - 1.0)

        if gate_dims:
            for dim, weight in gate_dims:
                ffn.W_gate.data[unit + 1, dim] = weight
        ffn.b_gate.data[unit + 1] = gate_bias

        ffn.W_down.data[out_dim, unit + 1] = -1.0 / S

        return unit + 2

    @staticmethod
    def threshold_match(
        ffn,
        unit: int,
        up_dim: int,
        out_dim: int,
        threshold: float = 0.3,
        gate_dim: Optional[int] = None,
        gate_negative: bool = False,
        S: float = 100.0,
    ) -> int:
        """Set threshold match pattern for step structure detection.

        Pattern from _set_phase_a_ffn (vm_step.py:2163-2211):
          up = S * threshold_head_dim
          b_up = -S * threshold
          gate = optional condition (negative for NOT)
          down = transition flag

        Args:
            ffn: FFN layer module
            unit: Hidden unit index
            up_dim: Input dimension (threshold head output)
            out_dim: Output dimension (NEXT_* flag)
            threshold: Activation threshold
            gate_dim: Optional gate dimension
            gate_negative: If True, gate fires when gate_dim is 0
            S: Scale factor

        Returns:
            Next available unit index
        """
        ffn.W_up.data[unit, up_dim] = S
        ffn.b_up.data[unit] = -S * threshold

        if gate_dim is not None:
            ffn.W_gate.data[unit, gate_dim] = -1.0 if gate_negative else 1.0
            ffn.b_gate.data[unit] = 1.0 if gate_negative else 0.0
        else:
            ffn.b_gate.data[unit] = 1.0

        ffn.W_down.data[out_dim, unit] = 1.0 / S

        return unit + 1

    # =========================================================================
    # Set 1 (re-extracted from agent a19962): register-decrement, marker-write,
    # opcode-gated PC override, byte passthrough chain.
    # =========================================================================

    @staticmethod
    def register_decrement_unit(
        ffn,
        *,
        unit: int,
        register_marker_dim: int,
        op_gate_dim: int,
        embed_lo_dim: int,
        embed_hi_dim: int,
        output_lo_dim: int,
        output_hi_dim: int,
        decrement: int,
        S: float,
        op_strength: float = 1.0,
    ) -> int:
        """Generate 32 FFN units (16 lo + 16 hi nibble) implementing
        register -= decrement at a marker token, with hi-nibble borrow.

        Mirrors the PSH/JSR/ENT SP-decrement pattern in
        `_set_layer6_routing_ffn`. Each unit fires only when both
        ``op_gate_dim`` and ``register_marker_dim`` are active (threshold
        T=1.5 against the sum of two unit-magnitude signals scaled by
        ``S`` and ``op_strength*S`` respectively).

        Lo nibble: rotate by ``-decrement`` (mod 16), cancel identity carry.
        Hi nibble: borrow (-=1) when the original lo nibble was >= 8, also
        cancels identity carry.
        """
        T = 1.5
        # Lo nibble: shifted copy + cancel identity
        for k in range(16):
            new_k = (k - decrement) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_lo_dim + k] = 1.0
            ffn.W_down.data[output_lo_dim + new_k, unit] = 2.0 / S
            ffn.W_down.data[output_lo_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        # Hi nibble: borrow when old lo >= 8
        for k in range(16):
            new_k_borrow = (k - 1) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_hi_dim + k] = 1.0
            for lo_bit in range(8, 16):
                ffn.W_gate.data[unit, embed_lo_dim + lo_bit] = -1.0
            ffn.W_down.data[output_hi_dim + new_k_borrow, unit] = 2.0 / S
            ffn.W_down.data[output_hi_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        return unit

    @staticmethod
    def marker_write_unit(
        ffn,
        *,
        unit: int,
        marker_dim: int,
        op_gate_dim: int,
        source_dims,
        target_dim: int,
        S: float,
        magnitude: float = 2.0 / 100.0,
    ) -> int:
        """Single FFN unit gated by marker AND op_gate, summing
        ``source_dims`` (gate-relayed, weight 1.0 each) into ``target_dim``."""
        ffn.W_up.data[unit, marker_dim] = S
        ffn.W_up.data[unit, op_gate_dim] = S
        ffn.b_up.data[unit] = -S * 1.5
        for src in source_dims:
            ffn.W_gate.data[unit, src] = 1.0
        ffn.W_down.data[target_dim, unit] = magnitude
        return unit + 1

    @staticmethod
    def opcode_gated_pc_override(
        ffn,
        *,
        unit: int,
        op_gate_dim: int,
        mark_pc_dim: int,
        target_pc_lo_dim: int,
        target_pc_hi_dim: int,
        source_lo_dim: int,
        source_hi_dim: int,
        S: float,
        extra_blockers=None,
    ) -> int:
        """Generate 64 FFN units (32 cancel + 32 write) implementing an
        opcode-gated PC override at the PC marker."""
        T = 4.5
        blockers = list(extra_blockers) if extra_blockers else []

        # === Cancel phase (32 units): clear OUTPUT_LO/HI ===
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, target_pc_lo_dim + k] = -1.0
            ffn.W_down.data[target_pc_lo_dim + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, target_pc_hi_dim + k] = -1.0
            ffn.W_down.data[target_pc_hi_dim + k, unit] = 2.0 / S
            unit += 1
        # === Write phase (32 units): copy source -> OUTPUT ===
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, source_lo_dim + k] = 1.0
            ffn.W_down.data[target_pc_lo_dim + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, source_hi_dim + k] = 1.0
            ffn.W_down.data[target_pc_hi_dim + k, unit] = 2.0 / S
            unit += 1
        return unit

    @staticmethod
    def byte_passthrough_chain(
        attn,
        *,
        head_idx: int,
        source_marker_dim: int,
        target_marker_dim: int,
        value_lo_dim: int,
        value_hi_dim: int,
        suppress_op_dims,
        S: float,
        HD: int,
        alibi_slope: float = 1.0,
        is_byte_strength: float = 3.0,
        has_se_strength: float = 1.0,
        suppress_strength: float = 3.0,
        q0_threshold: float = 3.5,
        gate_const: float = -20000.0,
        gate_target_marker: float = 10000.0,
        gate_has_se: float = 10000.0,
        gate_extras=None,
    ) -> None:
        """Configure a full attention head implementing byte passthrough
        across steps via shifted byte matching (Q byte K -> K byte K+1 of
        prev step).

        Mirrors `_set_layer10_byte_passthrough` (AX) and
        `_set_layer10_sp_byte_passthrough` (SP). Copies
        ``value_lo_dim`` / ``value_hi_dim`` (16 lo + 16 hi nibbles) into
        OUTPUT_LO/OUTPUT_HI with strength 2.0.

        Default Q[0] / Q[33] coefficients reproduce the AX function. The
        SP variant overrides ``is_byte_strength=1.0``, ``has_se_strength=2.0``,
        ``suppress_strength=2.0``, ``q0_threshold=1.5``, ``gate_const=-30000.0``,
        and adds ``gate_extras=[(IS_BYTE, 10000), (PSH_AT_SP, -10000),
        (CMP+3, -10000)]`` to encode the PSH/POP suppression.
        """
        L = S
        base = head_idx * HD

        # Q dim 0: IS_BYTE AND HAS_SE, suppressed by op flags
        attn.W_q.data[base + 0, BD.IS_BYTE] = L * is_byte_strength
        attn.W_q.data[base + 0, BD.HAS_SE] = L * has_se_strength
        for d in suppress_op_dims:
            attn.W_q.data[base + 0, d] = -L * suppress_strength
        attn.W_q.data[base + 0, BD.CONST] = -L * q0_threshold

        # Q dim 1: target marker discrimination
        attn.W_q.data[base + 1, target_marker_dim] = L
        attn.W_q.data[base + 1, BD.CONST] = -L / 2

        # Q dim 2: suppress byte 3 (predicts next register's marker)
        attn.W_q.data[base + 2, BD.BYTE_INDEX_3] = -L
        attn.W_q.data[base + 2, BD.CONST] = L / 2

        # K dim 0: IS_BYTE
        attn.W_k.data[base + 0, BD.IS_BYTE] = L
        # K dim 1: source marker (only target-register bytes are strong K)
        attn.W_k.data[base + 1, source_marker_dim] = L
        # K dim 2: suppress byte 0 in K (not a valid target for shifted matching)
        attn.W_k.data[base + 2, BD.BYTE_INDEX_0] = -L
        attn.W_k.data[base + 2, BD.CONST] = L / 2

        # Shifted byte matching: Q byte K -> K byte K+1 of prev step
        attn.W_q.data[base + 3, BD.BYTE_INDEX_0] = L
        attn.W_k.data[base + 3, BD.BYTE_INDEX_1] = L
        attn.W_q.data[base + 4, BD.BYTE_INDEX_1] = L
        attn.W_k.data[base + 4, BD.BYTE_INDEX_2] = L
        attn.W_q.data[base + 5, BD.BYTE_INDEX_2] = L
        attn.W_k.data[base + 5, BD.BYTE_INDEX_3] = L

        # Gate dim 33: hard AND of target_marker AND HAS_SE (kills leakage)
        attn.W_q.data[base + 33, BD.CONST] = gate_const
        attn.W_q.data[base + 33, target_marker_dim] = gate_target_marker
        attn.W_q.data[base + 33, BD.HAS_SE] = gate_has_se
        if gate_extras:
            for d, w in gate_extras:
                attn.W_q.data[base + 33, d] = w
        attn.W_k.data[base + 33, BD.CONST] = 5.0

        # V: copy 16 lo + 16 hi nibbles
        for k in range(16):
            attn.W_v.data[base + k, value_lo_dim + k] = 1.0
            attn.W_v.data[base + 16 + k, value_hi_dim + k] = 1.0

        # O: write to OUTPUT_LO/HI at strength 2.0
        for k in range(16):
            attn.W_o.data[BD.OUTPUT_LO + k, base + k] = 2.0
            attn.W_o.data[BD.OUTPUT_HI + k, base + 16 + k] = 2.0

        # Optional ALiBi slope override for this head
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[head_idx] = alibi_slope

    @staticmethod
    def register_increment_unit(
        ffn,
        *,
        unit: int,
        register_marker_dim: int,
        op_gate_dim: int,
        embed_lo_dim: int,
        embed_hi_dim: int,
        output_lo_dim: int,
        output_hi_dim: int,
        increment: int,
        S: float,
        op_strength: float = 1.0,
    ) -> int:
        """Generate 32 FFN units (16 lo + 16 hi nibble) implementing
        register += increment at a marker token, with hi-nibble carry.

        Companion to :meth:`register_decrement_unit` -- same shape, opposite
        direction.
        """
        T = 1.5
        # Lo nibble: shifted copy + cancel identity
        for k in range(16):
            new_k = (k + increment) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_lo_dim + k] = 1.0
            ffn.W_down.data[output_lo_dim + new_k, unit] = 2.0 / S
            ffn.W_down.data[output_lo_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        # Hi nibble: carry when old lo >= 8 (adding 8 overflows lo nibble)
        for k in range(16):
            new_k_carry = (k + 1) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_hi_dim + k] = 1.0
            for lo_bit in range(8):
                ffn.W_gate.data[unit, embed_lo_dim + lo_bit] = -1.0
            ffn.W_down.data[output_hi_dim + new_k_carry, unit] = 2.0 / S
            ffn.W_down.data[output_hi_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        return unit

    # =========================================================================
    # Extracted batch B (vm_step direct ports — byte-identical to imperative)
    # =========================================================================

    @staticmethod
    def nibble_rotation_chain(
        ffn,
        *,
        unit: int,
        gate_marker: int,
        source_lo_dim: int,
        source_hi_dim: int,
        target_lo_dim: int,
        target_hi_dim: int,
        offset: int = 1,
        with_carry: bool = True,
        S: float = 100.0,
        magnitude: float = 2.0,
        condition_dims: Optional[List[int]] = None,
        condition_threshold: Optional[float] = None,
    ) -> int:
        """Bake an FFN block that writes ``(source + offset) % 256`` to a
        target nibble pair, gated by an arbitrary AND of dims.

        Direct port of the L4 PC-rotation chains in ``_set_layer4_ffn``
        (vm_step.py): the offset=1 chain at AX marker, the offset∈{2,3,4}
        chains at AX byte positions (``with_carry=False`` because the
        TEMP source already has its carry applied), and the offset=1
        chain at PC marker → FETCH.

        For ``with_carry=False``: emits ``32`` units (16 lo rotations +
        16 hi copies). The hi copy assumes the source already has its
        carry applied (used for the multi-byte AX byte path that reads
        from TEMP, which itself was filled by an earlier carry-aware
        chain).

        For ``with_carry=True``: emits ``32 + 32*offset`` units (for
        offset=1: 64 units total; offset=2: 96; offset=3: 128;
        offset=4: 160) in the order::

            16 × lo_rotation:
                W_up[u, gate_marker]            = S       (+ extras)
                b_up[u]                         = -S*0.5  (or per cond)
                W_gate[u, source_lo+(k-off)%16] = 1.0
                W_down[target_lo+k, u]          = magnitude / S
            16 × hi_default_copy:
                W_up[u, gate_marker]            = S       (+ extras)
                b_up[u]                         = -S*0.5  (or per cond)
                W_gate[u, source_hi+k]          = 1.0
                W_down[target_hi+k, u]          = magnitude / S
            for carry_src in range(16-offset, 16):
              16 × {hi_cancel, hi_rotated} carry pairs:
                # Cancel default copy when source_lo[carry_src] == 1
                W_up[u,   gate_marker]                  = S       (+ extras)
                W_up[u,   source_lo+carry_src]          = S
                b_up[u]                                 = -S*1.5  (or per cond)
                W_gate[u, source_hi+k]                  = -1.0
                W_down[target_hi+k, u]                  = magnitude / S
                # Add rotated when source_lo[carry_src] == 1
                W_up[u+1, gate_marker]                  = S       (+ extras)
                W_up[u+1, source_lo+carry_src]          = S
                b_up[u+1]                               = -S*1.5  (or per cond)
                W_gate[u+1, source_hi+(k-1)%16]         = 1.0
                W_down[target_hi+k, u+1]                = magnitude / S

        For offset=1 (the canonical PC+1 case) carry_src ∈ {15} only,
        so the carry block emits 32 units (16 cancel + 16 rotated).
        For offset > 1 there are multiple carry sources (lo + N >= 16
        means lo ∈ [16-N, 15]), so the carry block emits 32*offset
        units. The hi-nibble carry block is always a +1 rotation
        because the carry from a +N lo rotation always contributes
        exactly +1 to the hi nibble (since lo nibble fits in 4 bits,
        max sum = 30 → carry ∈ {0, 1}). The lo-rotation source uses
        ``(k - offset) % 16`` but the hi-carry adjustment is
        ``(k - 1) % 16`` regardless.

        Math semantics: writes ``(source + offset) mod 256`` to the
        target nibble pair when the AND of (gate_marker,
        condition_dims...) is active.

        Args:
            ffn: FFN module.
            unit: First free hidden unit. Returns ``unit + 32`` (no
                carry) or ``unit + 32 + 32*offset`` (with carry; 64
                for offset=1, 96 for offset=2, etc.).
            gate_marker: Primary gate dim that scopes the rotation
                (e.g. ``BD.MARK_AX``, ``BD.MARK_PC``, or — for the
                multi-byte path that has no marker — ``BD.IS_BYTE``).
            source_lo_dim, source_hi_dim: Input nibble bases (e.g.
                ``BD.EMBED_LO``, ``BD.EMBED_HI``). Each is a 16-dim
                one-hot.
            target_lo_dim, target_hi_dim: Output nibble bases (e.g.
                ``BD.TEMP``, ``BD.TEMP+16``).
            offset: Integer rotation amount (typically 1, 2, 3, 4).
            with_carry: Emit the +16 carry-correction units (lo[15]==1
                triggers hi+=1). Set False for the multi-byte case
                where hi is just copied from a pre-rotated source.
            S: SwiGLU scale (default 100.0, matches vm_step.py).
            magnitude: ``W_down`` scale; the bake uses ``magnitude / S``.
                Default 2.0 matches the standard nibble-write scale.
            condition_dims: Extra W_up dims (with weight S) that AND
                with gate_marker (e.g. ``[BD.H1+AX_I,
                BD.BYTE_INDEX_0]`` for the multi-byte case). Must be a
                list of ints — all entries get weight S.
            condition_threshold: Override b_up base threshold. Defaults
                to ``0.5 + len(condition_dims)`` so b_up = ``-S*0.5``
                for 0 conditions, ``-S*2.5`` for 2 conditions, etc.
                Matches the vm_step.py threshold formula.

        Returns:
            New free unit index.
        """
        if condition_dims is None:
            condition_dims = []
        n_conds = len(condition_dims)
        # vm_step uses b_up = -S * (0.5 + n_conds) for the up gate — i.e.
        # threshold = (gate_marker + sum(condition_dims) - n_conds - 0.5).
        # For the carry block, threshold steps up by +1 (extra
        # source_lo+15 constraint) so b_up = -S * (1.5 + n_conds).
        if condition_threshold is None:
            base_thresh = 0.5 + n_conds
        else:
            base_thresh = condition_threshold
        carry_thresh = base_thresh + 1.0  # extra source_lo[15] AND
        u = unit
        scale = magnitude / S

        # 16 × lo rotation
        for k in range(16):
            src = (k - offset) % 16
            ffn.W_up[u, gate_marker] = S
            for cd in condition_dims:
                ffn.W_up[u, cd] = S
            ffn.b_up[u] = -S * base_thresh
            ffn.W_gate[u, source_lo_dim + src] = 1.0
            ffn.W_down[target_lo_dim + k, u] = scale
            u += 1

        # 16 × hi default copy
        for k in range(16):
            ffn.W_up[u, gate_marker] = S
            for cd in condition_dims:
                ffn.W_up[u, cd] = S
            ffn.b_up[u] = -S * base_thresh
            ffn.W_gate[u, source_hi_dim + k] = 1.0
            ffn.W_down[target_hi_dim + k, u] = scale
            u += 1

        if with_carry:
            # For offset=N, hi-nibble carries when (lo + N) >= 16, i.e.
            # lo ∈ [16-N, 15]. Emit a (cancel default, write rotated) pair
            # for each carry source bit, gated on source_lo[carry_src]=1.
            #   offset=1 → carry_src ∈ {15}        (32 units total)
            #   offset=2 → carry_src ∈ {14, 15}    (64 units total)
            #   offset=3 → carry_src ∈ {13, 14, 15} (96 units total)
            #   offset=4 → carry_src ∈ {12, …, 15} (128 units total)
            for carry_src in range(16 - offset, 16):
                for k in range(16):
                    # Cancel default copy when source_lo[carry_src] == 1
                    ffn.W_up[u, gate_marker] = S
                    ffn.W_up[u, source_lo_dim + carry_src] = S
                    for cd in condition_dims:
                        ffn.W_up[u, cd] = S
                    ffn.b_up[u] = -S * carry_thresh
                    ffn.W_gate[u, source_hi_dim + k] = -1.0
                    ffn.W_down[target_hi_dim + k, u] = scale
                    u += 1
                    # Add rotated +1 when source_lo[carry_src] == 1
                    hi_src = (k - 1) % 16
                    ffn.W_up[u, gate_marker] = S
                    ffn.W_up[u, source_lo_dim + carry_src] = S
                    for cd in condition_dims:
                        ffn.W_up[u, cd] = S
                    ffn.b_up[u] = -S * carry_thresh
                    ffn.W_gate[u, source_hi_dim + hi_src] = 1.0
                    ffn.W_down[target_hi_dim + k, u] = scale
                    u += 1

        return u


# Convenience aliases
P = Primitives
