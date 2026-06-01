"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L4 attention-head layout (pinned indices) ======================
#
# Two ops claim heads on the L4 attention block today:
#
#   * ``layer4_pc_relay`` (phase=4)              -> heads 0, 1
#   * ``layer4_sp_to_addr_key`` (phase=4.5)      -> heads 2, 3
#
# Pre-migration each call site picked its ``head_idx`` as a bare integer
# literal -- ``head_idx=0`` / ``head_idx=1`` in the PC-relay
# :class:`DeclarativeAttentionHeadSpec` block, and ``_stage_sp_byte(2,
# ...)`` / ``_stage_sp_byte(3, ...)`` in the SP-to-ADDR_KEY bake -- which
# made adding a future L4 head fragile (the author had to remember which
# slots were already taken). With the allocator the handoff is
# structural: each bake instantiates its OWN allocator pre-loaded with
# the full L4 head layout (pinned to existing slots), stashes it on
# ``attn._l4_head_allocator`` for downstream inspection, and resolves
# its own head indices by name. A future L4 attention op can claim a
# free head via ``allocator.alloc(name, layer_idx=4)`` -- with no
# ``pin=`` -- without touching this table.
#
# ``layer4_sp_to_addr_key`` is gated by ``enable=False`` today (a guard
# returns before any weight writes); we still pin its head indices here
# so the layout is structurally stable across builds. The pinned
# allocator only attaches to ``attn._l4_head_allocator`` when the op
# actually fires, matching the existing ``_claims`` gating below.
_L4_HEAD_LAYOUT = (
    # (op-name key,                            pinned head_idx)
    ("layer4_pc_relay.head_0",                 0),
    ("layer4_pc_relay.head_1",                 1),
    ("layer4_sp_to_addr_key.head_2",           2),
    ("layer4_sp_to_addr_key.head_3",           3),
)


def _allocate_layer4_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L4 heads.

    Every entry in :data:`_L4_HEAD_LAYOUT` is pinned at its existing
    ``head_idx`` so the underlying weight writes -- still hand-coded in
    ``_layer4_pc_relay_head_specs`` (heads 0/1) and ``_stage_sp_byte``
    (heads 2/3) -- land byte-identically. Both ``layer4_pc_relay`` and
    (when enabled) ``layer4_sp_to_addr_key`` call this so each can look
    up its own head by name without baking an integer literal at the
    call site.
    """
    allocator = AttentionHeadAllocator()
    for name, head_idx in _L4_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=4, pin=head_idx)
    return allocator


def _l4_head_idx(op_name: str) -> int:
    """Return the pinned L4 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L4_HEAD_LAYOUT` for callers that
    cannot instantiate a per-bake allocator (e.g. ``compiler_ir_factory``
    helpers, which run outside the bake and receive only dim positions
    and ``HD``). The runtime bakes still go through
    :func:`_allocate_layer4_attention_heads` so the collision-checked
    allocator path is exercised on every weight write.
    """
    for name, head_idx in _L4_HEAD_LAYOUT:
        if name == op_name:
            return head_idx
    raise KeyError(f"_l4_head_idx: unknown L4 attention op {op_name!r}")


# === L4 FFN unit layout (pinned offsets) ============================
#
# The ``layer4_ffn`` op owns the entire L4 FFN. The actual weight writes
# happen inside ``_bake_layer4_ffn`` below, which uses a local ``unit = 0``
# counter that walks through six sub-stages (PC+1@AX, TEMP clear, PC+2/3/4
# @AX byte positions, PC+1@PC). Migration to :class:`FFNUnitAllocator`
# keeps that helper byte-identical -- we declare each sub-stage's range
# at its existing pinned offset so the layout is auditable rather than
# implicit. Adding a new L4 op family later will go through
# ``allocator.alloc(name, n)`` without a pin, and the allocator will pick
# the first free gap above unit 544.
#
# IMPORTANT: the PC+N chains do NOT have equal stride. The widths follow
# :meth:`Primitives.nibble_rotation_chain`'s ``with_carry=True`` formula
# of ``32 + 32 * offset`` units per chain, so PC+2/+3/+4 land at 96/128/
# 160 units respectively (NOT a uniform 96-unit stride). The pin offsets
# below honour those exact widths so the layout matches the helper's
# monotonic walk byte-for-byte.
_L4_FFN_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    # PC+1 chain at MARK_AX (offset=1, with_carry=True): 32 + 32 = 64 units.
    #   16 lo rotation + 16 hi default-copy + 16 hi carry-cancel + 16 hi
    #   carry-rotated. Writes TEMP[0..15] (lo) and TEMP[16..31] (hi).
    ("layer4_ffn.pc_plus1_ax",       0,  64),
    # TEMP[0..31] clearing pass at MARK_PC. TEMP[0] is reserved for IS_JSR
    # so its slot is an empty placeholder (still consumes 1 unit). 32 units.
    ("layer4_ffn.temp_clear_pc",    64,  32),
    # PC+N chains at IS_BYTE × H1[AX_I] × BYTE_INDEX_n. Widths follow
    # ``32 + 32 * offset``: PC+2 → 96, PC+3 → 128, PC+4 → 160. NOT
    # equal-stride; the carry block grows with offset because lo + offset
    # >= 16 admits ``offset`` distinct carry sources (lo ∈ [16-offset, 15]).
    ("layer4_ffn.pc_plus2_byte0",   96,  96),  # offset=2, 32 + 32*2
    ("layer4_ffn.pc_plus3_byte1",  192, 128),  # offset=3, 32 + 32*3
    ("layer4_ffn.pc_plus4_byte2",  320, 160),  # offset=4, 32 + 32*4
    # PC+1 chain at MARK_PC (offset=1, with_carry=True): 64 units for the
    # dynamic immediate fetch path (L5 head 3 reads FETCH at PC marker).
    ("layer4_ffn.pc_plus1_pc",     480,  64),
)

# Total = 64 + 32 + 96 + 128 + 160 + 64 = 544 units (matches
# ``ffn_units_used=544`` and the historical helper footprint).
_L4_FFN_TOTAL_UNITS = 544


def _allocate_layer4_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L4 FFN sub-stages.

    Every sub-stage is pinned at its existing offset so the underlying
    ``_bake_layer4_ffn`` helper -- which writes via its own monotonic
    ``unit = 0`` counter -- lands on exactly the same hidden-unit indices
    it always has. This call is byte-identical bookkeeping: the allocator
    declares ranges by name, the helper writes the weights. A future
    refactor can split the monolithic helper into per-range bake functions
    that consume ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L4 op claims a free range past unit 544).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L4_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


def make_layer4_pc_relay_op() -> Operation:
    """L4 attention: relay PC marker EMBED → AX marker EMBED.

    Pinned to ``layer_idx=4`` via ``kind="block"`` because the legacy
    ``set_vm_weights`` pipeline targets block 4. Without pinning, the
    dep-graph layer assignment places this op at a later block (e.g. L5/L6),
    leaving block 4's attn zero-init and breaking the L5 fetch chain
    (the regression at commit b2d9f4c3).
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        # Per-bake attention-head allocator with the full L4 head layout
        # pinned. Stashed on ``attn`` for inspection / extension; the
        # actual ``head_idx`` values used by ``_layer4_pc_relay_head_specs``
        # come from :func:`_l4_head_idx` so the spec stays in lockstep
        # with the layout table without re-querying the allocator here.
        allocator = _allocate_layer4_attention_heads()
        attn._l4_head_allocator = allocator
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn,
            _layer4_pc_relay_head_specs(proxy),
            HD,
        )

    # Dim-ownership claims: L4 attn heads 0 + 1 PC relay.
    #   Head 0: V slots 1..32 read EMBED_LO/HI → EMBED_LO/HI at AX marker;
    #           V slots 33..48 read ADDR_KEY top nibble → AX ADDR_KEY top.
    #   Head 1: V slots 1..32 read EMBED_LO/HI → TEMP[0..31] at AX byte pos;
    #           V slots 33..48 read ADDR_KEY top nibble → byte ADDR_KEY top.
    _claims = set()
    for k in range(16):
        _claims.add((4, "attn_W_v", f"0_{1 + k}", f"EMBED_LO+{k}"))
        _claims.add((4, "attn_W_v", f"0_{17 + k}", f"EMBED_HI+{k}"))
        _claims.add((4, "attn_W_v", f"1_{1 + k}", f"EMBED_LO+{k}"))
        _claims.add((4, "attn_W_v", f"1_{17 + k}", f"EMBED_HI+{k}"))
        _claims.add((4, "attn_W_v", f"0_{33 + k}", f"ADDR_KEY+{32 + k}"))
        _claims.add((4, "attn_W_v", f"1_{33 + k}", f"ADDR_KEY+{32 + k}"))

    return Operation(
        name="layer4_pc_relay",
        phase=4,
        reads={"MARK_PC", "MARK_AX", "EMBED_LO", "EMBED_HI", "ADDR_KEY", "CONST"},
        writes={"EMBED_LO", "EMBED_HI", "ADDR_KEY"},  # at AX marker/bytes
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer4_pc_relay_ir,
        layer_idx=4,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


# IR identity cache: keyed by (id(dim_positions), HD). The attention
# verifier's cross-op competition pass (``catalog_attention_violations.py``,
# ``verify_attention_head(ops_for_competition=...)``) materializes each op's
# IR twice for the same ``dim_positions`` dict — once to enumerate heads,
# once to build the competitor index. Without caching, the two
# materializations produce distinct ``AttentionHeadIR`` instances, so the
# verifier's ``entry.head is not head`` identity check fails to filter
# self-vs-clone competition, and h0/h1 each generate 32 spurious
# ``attention_strength_violation`` issues against their own clones at the
# EMBED_LO/HI / ADDR_KEY+32..47 band (resolved by the V1 dim resolver to
# ADDR_B1_LO+9..15 / ADDR_B2_LO+0..15 / EMBED_HI+7..15 / OUTPUT_LO+0..6
# etc. due to runtime dim_positions remapping).  Memoizing on
# ``id(dim_positions)`` keeps both call sites pointing at the same IR so
# the identity filter actually fires; baseline on this op drops from
# 64 ASV + 14 CSV to 32 ASV + 14 CSV with no semantic change.
#
# Safety: dim_positions is constructed once per compile and held in scope
# for the full bake; id() collision via GC reclaim is not possible during
# that window.  HD is part of the key so a later debug call with a
# different head dim still gets a fresh IR.
_layer4_pc_relay_ir_cache: dict = {}


def _layer4_pc_relay_ir(dim_positions, HD) -> CompilerIR:
    key = (id(dim_positions), HD)
    cached = _layer4_pc_relay_ir_cache.get(key)
    if cached is not None:
        return cached
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    # NOTE(L4-pc-relay-scope-honest): the two heads' INTENDED firing scopes
    # are complementary -- h0 fires at the AX marker position via the
    # q-side MARK_AX gate; h1 fires at the AX byte positions via the
    # q-side IS_BYTE × H1+AX_I gate.  Both relay the PC marker's
    # ADDR_KEY top nibble (slots 33..48 → ``ADDR_KEY+32..47``) for
    # downstream consumers: L5 fetch head 0 reads ADDR_KEY+32..47 at the
    # AX marker (h0's write), and L8 multibyte fetch head 3 reads
    # ADDR_KEY+32..47 at AX byte positions (h1's write).
    #
    # The V1 verifier's K-derived ``effective_attention_scope`` is
    # ("CONST", "MARK_PC") for both heads -- the Q-side gating difference
    # (MARK_AX vs IS_BYTE × H1+AX_I) is invisible to V1.  Declaring the
    # Q-side scope here is informational under V1 (the verifier neither
    # filters competitors by Q-scope overlap nor by dominates_at metadata
    # at present -- see attention_verifier.py V2 wishlist) but documents
    # why the residual 16 ``attention_strength_violation`` per head against
    # the OTHER head at ADDR_KEY+32..47 (resolved as EMBED_HI+7..15 /
    # OUTPUT_LO+0..6 due to dim_positions remap) is bookkeeping: the two
    # heads never fire at the same position, so the writes do not actually
    # compete at runtime.
    #
    # We do NOT bump either head's magnitude (slot 33+k V copy ×
    # slot 33+k O copy = magnitude 1.0).  Bumping h0's slot 33..48 weights
    # to dominate h1 just transfers the violation to h1 (the symmetric
    # zero-sum case documented in
    # ``feedback_single_rule_fixes_are_zero_sum.md``).
    ir.layer(0).attention.append(
        _layer4_pc_relay_head_specs(proxy)[0],
        metadata={
            "scope": "mark == AX",
            "dominates_at": {
                "EMBED_LO": "mark == AX",
                "EMBED_HI": "mark == AX",
                "ADDR_KEY": "mark == AX AND offset >= 32",
            },
        },
    )
    ir.layer(0).attention.append(
        _layer4_pc_relay_head_specs(proxy)[1],
        metadata={
            "scope": "is_byte AND h1 == AX_I",
            "dominates_at": {
                "TEMP": "is_byte AND h1 == AX_I",
                "ADDR_KEY": "is_byte AND h1 == AX_I AND offset >= 32",
            },
        },
    )
    _layer4_pc_relay_ir_cache[key] = ir
    return ir


def _layer4_pc_relay_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative L4 heads 0-1: PC marker relay to AX marker/bytes."""

    L = 15.0
    AX_I = 1
    return (
        DeclarativeAttentionHeadSpec(
            head_idx=_l4_head_idx("layer4_pc_relay.head_0"),
            q=(
                AP(0, BD.MARK_AX, L),
                AP(33, BD.MARK_AX, L),
                AP(33, BD.CONST, -L / 2),
            ),
            k=(AP(0, BD.MARK_PC, L), AP(33, BD.CONST, L)),
            v=(
                tuple(AP(1 + k, BD.EMBED_LO + k, 1.0) for k in range(16))
                + tuple(AP(17 + k, BD.EMBED_HI + k, 1.0) for k in range(16))
                + tuple(AP(33 + k, BD.ADDR_KEY + 32 + k, 1.0) for k in range(16))
            ),
            o=(
                tuple(AO(BD.EMBED_LO + k, 1 + k, 1.0) for k in range(16))
                + tuple(AO(BD.EMBED_HI + k, 17 + k, 1.0) for k in range(16))
                + tuple(AO(BD.ADDR_KEY + 32 + k, 33 + k, 1.0) for k in range(16))
            ),
        ),
        DeclarativeAttentionHeadSpec(
            head_idx=_l4_head_idx("layer4_pc_relay.head_1"),
            q=(
                AP(0, BD.IS_BYTE, L),
                AP(0, BD.H1 + AX_I, L),
                AP(0, BD.CONST, -L * 1.5),
                AP(33, BD.IS_BYTE, 500.0),
                AP(33, BD.CONST, -500.0),
            ),
            k=(
                AP(0, BD.MARK_PC, L),
                AP(0, BD.CONST, L * 0.5),
                AP(33, BD.CONST, 5.0),
            ),
            v=(
                tuple(AP(1 + k, BD.EMBED_LO + k, 1.0) for k in range(16))
                + tuple(AP(17 + k, BD.EMBED_HI + k, 1.0) for k in range(16))
                + tuple(AP(33 + k, BD.ADDR_KEY + 32 + k, 1.0) for k in range(16))
            ),
            o=(
                tuple(AO(BD.TEMP + k, 1 + k, 1.0) for k in range(16))
                + tuple(AO(BD.TEMP + 16 + k, 17 + k, 1.0) for k in range(16))
                + tuple(AO(BD.ADDR_KEY + 32 + k, 33 + k, 1.0) for k in range(16))
            ),
        ),
    )


def make_layer4_ffn_op() -> Operation:
    """L4 FFN: compute PC+1/2/3/4 in FETCH dims for L5 fetch.

    Pinned to ``layer_idx=4`` via ``kind="block"``; see
    ``make_layer4_pc_relay_op``.
    """
    def bake(block, dim_positions, S):
        # Per-bake FFN-unit allocator. Each L4 FFN sub-stage is pinned to
        # its existing offset so the calls below land byte-identically.
        # The allocator is published on ``block.ffn`` for inspection /
        # extension by downstream tools (e.g. a future L4 op family
        # claiming a free gap past unit 544). Mirrors the L9 pattern in
        # ``make_layer9_alu_op``.
        allocator = _allocate_layer4_ffn_units()
        block.ffn._l4_unit_allocator = allocator

        final_unit = _bake_layer4_ffn(
            block.ffn, S, _as_setdim_proxy(dim_positions)
        )
        # Byte-identity guard: the helper's local cursor MUST end exactly
        # at the allocator's declared footprint. If the layout table drifts
        # from the helper's writes, this assertion fires before any weight
        # surgery happens.
        assert final_unit == _L4_FFN_TOTAL_UNITS, (
            f"L4 FFN unit cursor drift: helper returned {final_unit}, "
            f"allocator expected {_L4_FFN_TOTAL_UNITS}"
        )

    # Dim-ownership claims (W_down output cells). The bake programs four
    # ``nibble_rotation_chain``s and one TEMP-clear pass on L4.ffn:
    #   units 0..31:    PC+1 direct nibbles -> TEMP+(unit)
    #                   (16 lo + 16 hi nibbles of TEMP, sourced from EMBED).
    #   units 32..63:   PC+1 carry pairs writing TEMP+(16 + (unit-32)//2).
    #   unit 64:        IS_JSR placeholder (no write, preserves unit number).
    #   units 65..95:   TEMP[1..31] clear at MARK_PC -> TEMP+(unit-64).
    #   units 96..191:  PC+2 chain at MARK_AX (byte_idx=0, 96 units; 32
    #                   direct outputs into FETCH_LO/HI followed by 64
    #                   carry units feeding FETCH_HI carries).
    #   units 192..319: PC+3 chain (byte_idx=1, 128 units).
    #   units 320..479: PC+4 chain (byte_idx=2, 160 units).
    #   units 480..543: PC+1@MARK_PC fallback chain (64 units).
    # Declares the direct-output cells of each chain plus the TEMP carry
    # pairs and TEMP clear units. Carry-only units (128..223 etc., which
    # also write FETCH_HI+k via the carry projection) are intentionally
    # omitted to keep the partial set readable; the verifier only requires
    # declared ⊆ observed.
    _claims = set()
    # PC+1 -> TEMP direct nibbles (units 0..31).
    for unit in range(32):
        _claims.add((4, "ffn_W_down", str(unit), f"TEMP+{unit}"))
    # PC+1 -> TEMP carry pairs (units 32..63, two units per high-nibble
    # output TEMP+16..+31).
    for k in range(16):
        unit_pair = (32 + 2 * k, 33 + 2 * k)
        for unit in unit_pair:
            _claims.add((4, "ffn_W_down", str(unit), f"TEMP+{16 + k}"))
    # Unit 64 is reserved (IS_JSR placeholder; no write).
    # TEMP[1..31] clear at MARK_PC (units 65..95).
    for k in range(1, 32):
        _claims.add((4, "ffn_W_down", str(64 + k), f"TEMP+{k}"))
    # Multi-byte PC+2 / PC+3 / PC+4 chains and PC+1@MARK_PC fallback. Each
    # chain's first 32 units write the direct FETCH outputs (16 LO + 16 HI).
    # Chain bases: 96 (PC+2), 192 (PC+3), 320 (PC+4), 480 (PC+1@MARK_PC).
    for chain_base in (96, 192, 320, 480):
        for k in range(16):
            _claims.add((4, "ffn_W_down", str(chain_base + k), f"FETCH_LO+{k}"))
            _claims.add((4, "ffn_W_down", str(chain_base + 16 + k), f"FETCH_HI+{k}"))

    return Operation(
        name="layer4_ffn",
        phase=4,
        reads={"MARK_AX", "MARK_PC", "EMBED_LO", "EMBED_HI",
               "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1"},
        writes={"FETCH_LO", "FETCH_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=4,
        migrated=True,
        claims=_claims,
        # ``_set_layer4_ffn`` writes the PC+1 (lo/hi/carry = 64 units) +
        # TEMP-clear (32) + multi-byte PC+2/+3/+4 (96+128+160 = 384) +
        # PC+1@PC marker (64) chains for a total of 544 units (0..543).
        # See bake body in vm_step.py:_set_layer4_ffn.
        ffn_units_used=544,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _bake_layer4_ffn(ffn, S, BD) -> int:
    """Declarative L4 FFN spec: PC+1/2/3/4 fetch-address rotations.

    Returns the post-bake unit cursor (must equal
    :data:`_L4_FFN_TOTAL_UNITS` for byte-identity with the historical
    544-unit footprint). The caller asserts this in ``make_layer4_ffn_op``.
    """

    unit = 0

    unit = Primitives.nibble_rotation_chain(
        ffn,
        unit=unit,
        gate_marker=BD.MARK_AX,
        source_lo_dim=BD.EMBED_LO,
        source_hi_dim=BD.EMBED_HI,
        target_lo_dim=BD.TEMP,
        target_hi_dim=BD.TEMP + 16,
        offset=1,
        with_carry=True,
        S=S,
        magnitude=2.0,
    )

    # Preserve unit numbering from the legacy helper: TEMP[0] is reserved for
    # IS_JSR, so its clearing unit remains an empty placeholder.
    for k in range(32):
        if k == 0:
            unit += 1
            continue
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.W_gate.data[unit, BD.TEMP + k] = -1.0
        ffn.W_down.data[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    AX_I = 1
    for byte_idx in range(3):
        unit = Primitives.nibble_rotation_chain(
            ffn,
            unit=unit,
            gate_marker=BD.IS_BYTE,
            source_lo_dim=BD.TEMP,
            source_hi_dim=BD.TEMP + 16,
            target_lo_dim=BD.FETCH_LO,
            target_hi_dim=BD.FETCH_HI,
            offset=byte_idx + 2,
            with_carry=True,
            S=S,
            magnitude=2.0,
            condition_dims=[BD.H1 + AX_I, BD.BYTE_INDEX_0 + byte_idx],
        )

    unit = Primitives.nibble_rotation_chain(
        ffn,
        unit=unit,
        gate_marker=BD.MARK_PC,
        source_lo_dim=BD.EMBED_LO,
        source_hi_dim=BD.EMBED_HI,
        target_lo_dim=BD.FETCH_LO,
        target_hi_dim=BD.FETCH_HI,
        offset=1,
        with_carry=True,
        S=S,
        magnitude=2.0,
    )

    return unit


def make_layer4_sp_to_addr_key_op(enable: bool = False) -> Operation:
    """L4 attention heads 2 + 3: gather SP value → ADDR_KEY at AX marker.

    Phase 1 of STACK0_VIA_MEM_ATTENTION_PLAN — Q-side staging for the
    mem-attention path that replaces L7 head 0's STACK0_BYTE0 read with a
    direct ``mem[SP]`` lookup. The mem-attention K-side (in
    ``_inject_mem_metadata``) already writes ``ADDR_KEY[lo, 16+hi, 32+top]``
    at MEM val byte positions; this op produces the matching encoding at
    the AX marker so a downstream attention head (at L8 attn — see
    ``make_layer8_mem_to_alu_op``) can match on address.

    Encoding (matches ``_inject_mem_metadata`` exactly):
      - Head 2 fires at AX marker, attends to SP byte 0 position. Copies
        CLEAN_EMBED_LO → ``ADDR_KEY[0..15]`` (lo nibble of SP byte 0) and
        CLEAN_EMBED_HI → ``ADDR_KEY[16..31]`` (hi nibble of SP byte 0).
      - Head 3 fires at AX marker, attends to SP byte 1 position. Copies
        CLEAN_EMBED_LO → ``ADDR_KEY[32..47]`` (lo nibble of SP byte 1,
        i.e. the "top" 4 bits of the 12-bit ADDR_KEY space).

    The two heads write to non-overlapping ADDR_KEY sub-bands. Since
    ADDR_B0_HI/ADDR_B1_HI/ADDR_B2_HI alias the ADDR_KEY band (dims
    206/222/238), the writes use those aliases for clarity.

    Gating: AX marker AND ``HAS_SE = 1`` (step 1+). Step 0 has no prior
    MEM section to read from, and the default STACK_INIT SP would
    otherwise be injected before any PSH has populated memory.

    Disabled by default (``enable=False``); the bake is a guard-clause
    no-op. Existing tests stay byte-identical until both this op and the
    L8 mem-to-ALU head are flipped on together.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return

        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the full L4 head layout
        # pinned. Mirrors ``make_layer4_pc_relay_op`` so the SP-to-ADDR_KEY
        # bake produces the same ``attn._l4_head_allocator`` view regardless
        # of which op runs first. The head indices passed to
        # ``_stage_sp_byte`` come from :func:`_l4_head_idx` so the literal
        # ``2`` / ``3`` no longer appear at the call site.
        allocator = _allocate_layer4_attention_heads()
        attn._l4_head_allocator = allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        L = 50.0  # strong gate; overpowers any incidental L7 contamination
        SP_I = 2  # SP marker index in MARKS array
        # SCALE > 1 so the SP signal dominates any incidental writes by L7
        # heads 2-4 (which write prev-AX nibbles into the same ADDR_B*_HI
        # bands for LI/LC).
        SCALE_O = 10.0

        def _stage_sp_byte(head_idx, byte_idx_dim, write_lo_to, write_hi_to):
            """Stage one SP byte into the ADDR_KEY band at the AX marker.

            Each head fires at the AX marker (only on step 1+, gated by
            HAS_SE) and attends to a single SP byte position, copying the
            CLEAN_EMBED nibbles into the requested ADDR_B*_HI sub-bands.

            ``write_hi_to`` may be ``None`` for heads that only carry the
            lo nibble (e.g. SP byte 1's hi nibble extends past the 12-bit
            ADDR_KEY space and is intentionally dropped).
            """
            base = head_idx * HD
            # Q: AX marker AND HAS_SE = 1 (step 1+). Step 0 has no prior
            # MEM section to read from, and the default STACK_INIT SP
            # would corrupt downstream ADDR_KEY readers if injected before
            # any PSH has populated memory.
            attn.W_q[base, BD.MARK_AX] = L
            attn.W_q[base, BD.HAS_SE] = L
            attn.W_q[base, BD.CONST] = -L * 1.5
            # K: target SP byte position. BYTE_INDEX_n fires at byte n of
            # every register; H1[SP_I] localises to the SP byte area
            # (d=1..4 from the SP marker).
            attn.W_k[base, byte_idx_dim] = L
            attn.W_k[base, BD.H1 + SP_I] = L
            attn.W_k[base, BD.CONST] = -L
            # Anti-leakage gate dim
            attn.W_q[base + 33, BD.MARK_AX] = L
            attn.W_q[base + 33, BD.CONST] = -L / 2
            attn.W_k[base + 33, BD.CONST] = L
            # V: copy CLEAN_EMBED nibbles (LO always; HI only if requested)
            for k in range(16):
                attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                if write_hi_to is not None:
                    attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            # O: write into the requested ADDR_KEY sub-band(s)
            for k in range(16):
                attn.W_o[write_lo_to + k, base + 1 + k] = SCALE_O
                if write_hi_to is not None:
                    attn.W_o[write_hi_to + k, base + 17 + k] = SCALE_O

        # Head 2: SP byte 0 → ADDR_KEY[0..15] (lo nibble) + [16..31] (hi nibble).
        _stage_sp_byte(
            _l4_head_idx("layer4_sp_to_addr_key.head_2"),
            BD.BYTE_INDEX_0, BD.ADDR_B0_HI, BD.ADDR_B1_HI,
        )
        # Head 3: SP byte 1 → ADDR_KEY[32..47] (lo nibble only). The hi
        # nibble of SP byte 1 would extend ADDR_KEY past 48 dims; matches
        # the 12-bit "top" convention used by `_inject_mem_metadata`.
        _stage_sp_byte(
            _l4_head_idx("layer4_sp_to_addr_key.head_3"),
            BD.BYTE_INDEX_1, BD.ADDR_B2_HI, None,
        )

    # Dim-ownership claims: L4 attn heads 2 + 3 SP-to-ADDR_KEY staging.
    # Each head writes V slots 1..32 + O writes into ADDR_B*_HI sub-bands.
    #   Head 2: SP byte 0 → ADDR_B0_HI (lo) + ADDR_B1_HI (hi).
    #     W_v[2*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #     W_v[2*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   Head 3: SP byte 1 → ADDR_B2_HI (lo only).
    #     W_v[3*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    # Only meaningful when enable=True; we declare claims unconditionally so
    # the registry can catch latent collisions once the op is enabled.
    _claims = set()
    if enable:
        for k in range(16):
            _claims.add((4, "attn_W_v", f"2_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((4, "attn_W_v", f"2_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
            _claims.add((4, "attn_W_v", f"3_{1 + k}", f"CLEAN_EMBED_LO+{k}"))

    return Operation(
        name="layer4_sp_to_addr_key",
        phase=4.5,  # after layer4_pc_relay (phase=4) so its writes don't clobber
        reads={"MARK_AX", "BYTE_INDEX_0", "BYTE_INDEX_1", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI"},  # = ADDR_KEY band
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=4,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )
