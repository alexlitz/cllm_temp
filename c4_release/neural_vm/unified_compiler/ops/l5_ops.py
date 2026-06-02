"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..layer_compiler import Operation
from ..ir import CompilerIR, FFNRule
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy, _opcode_name_map


# === L5 FFN unit layout (auto-fit; legacy offsets retained as docs) ===
#
# The ``opcode_decode_ffn`` op owns the entire L5 FFN. The actual weight
# writes happen inside ``_bake_opcode_decode_ffn`` below, which uses a
# local ``unit = 0`` counter that walks through four CompilerIR rule
# batches plus one reserved blank slot.
#
# Phase 7.B.2: every entry below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because the layout is fully
# contiguous in declaration order, first-fit reproduces the legacy
# pinned offsets bit-for-bit; the ``_bake_opcode_decode_ffn`` helper's
# own unit-0 cursor is what positions the actual weight writes, so
# byte-identity is independent of allocator order. The
# ``legacy_start`` column is documentation only.
#
# Sibling L5 ops do NOT allocate FFN units:
#   * ``layer5_fetch`` writes attn5 W_q/W_k/W_v/W_o only (8 attention
#     heads); no FFN units.
#   * ``layer5_user_input_gather`` is a phase-1 ``enable=False`` no-op
#     (see ``user_input_ops.py``); phase 2 will allocate L5 attention
#     heads 8/9, still no FFN units.
# So this allocator covers the full L5 FFN footprint.
#
# The offsets below mirror the unit-counter walk in
# ``_bake_opcode_decode_ffn`` (main per-opcode AX decode -> first-step
# PC decode -> reserved JSR TEMP[0] blank -> TEMP[1..31] clear at PC ->
# all-step PC decode). Changing any helper's unit count requires
# updating this table in lock-step.
_L5_FFN_UNIT_LAYOUT = (
    # (sub-stage name, legacy_start (docs only), n_units)
    # 34 main per-opcode AX rules (one unit per opcode in the table at
    # ``_opcode_decode_main_rules``). Unit 3 writes TEMP+0 (JSR's IS_JSR
    # flag); the other 33 units each write the matching OP_* dim.
    ("opcode_decode_ffn.main_at_ax",          0, 34),
    # 18 first-step PC-marker decode rules (``HAS_SE == 0`` gates the
    # initial step). Unit 35 writes TEMP+0 (first-step JSR's IS_JSR
    # flag); the other 17 units write OP_* dims.
    ("opcode_decode_ffn.first_step_at_pc",   34, 18),
    # Reserved blank unit for the first-step JSR TEMP[0] slot -- the
    # bake skips ``unit += 1`` here to preserve legacy unit numbering
    # for the TEMP-clear band that follows. No weight writes land on
    # unit 52.
    ("opcode_decode_ffn.jsr_temp0_blank",    52,  1),
    # 31 TEMP[1..31] clearing rules at MARK_PC (TEMP[0] is owned by
    # the first-step JSR flag and is intentionally skipped).
    ("opcode_decode_ffn.temp_clear_at_pc",   53, 31),
    # 5 all-step PC-marker decode rules (BZ, BNZ, LEV, EXIT, JMP -- the
    # only opcodes whose first-step write also fires on the AX-marker
    # step's PC-marker copy).
    ("opcode_decode_ffn.all_step_at_pc",     84,  5),
)

# Total = 34 + 18 + 1 + 31 + 5 = 89 units (final cursor lands at 89;
# highest used unit index is 88, matching the 5/ffn_W_down/88 claim on
# OP_JMP+0 in ``make_opcode_decode_ffn_op``).
_L5_FFN_TOTAL_UNITS = 89


def _allocate_layer5_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L5 FFN sub-stages.

    Phase 7.B.2: ``pin=`` is dropped from every entry in
    :data:`_L5_FFN_UNIT_LAYOUT`. The allocator's default first-fit
    strategy walks the layout in declaration order and lands each
    sub-stage at the lowest free gap large enough to hold it. Because
    the layout is fully contiguous (every entry starts exactly where
    the previous one ended), first-fit reproduces the legacy pinned
    offsets bit-for-bit. The underlying ``_bake_opcode_decode_ffn``
    helper -- which writes via its own monotonic ``unit = 0`` counter
    -- lands on exactly the same hidden-unit indices regardless of
    allocator order, so byte-identity with the legacy bake is
    preserved. The allocator's role is bookkeeping: the layout
    declares ranges by name, the helper writes the weights. A future
    refactor can split the monolithic helper into per-range bake
    functions that consume ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L5 op claims a free range past unit 89).
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L5_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def make_layer5_fetch_op() -> Operation:
    """L5 attention: instruction-fetch heads (8 heads).

    Dispatched as a block op pinned to layer_idx=5 so the bake hits the same
    transformer block (block[5].attn) the legacy path used. Using kind="block"
    routes through compile_full_vm's block_ops dispatch even when legacy_bake
    is present, ensuring block[5] receives the L5 fetch logic for pure_neural
    execution. The companion `_layer5_fetch_dep_anchor` op declares the same
    reads/writes via kind="attn" so the LayerCompiler's dep graph still
    reserves a layer slot for it (preserving model n_layers).
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        BD = _as_setdim_proxy(dim_positions)
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.0)
            # L5 fetch heads read immutable code bytes by exact ADDR_KEY.
            # Recency bias makes later same-low-address aliases compete with
            # the exact code-prefix match once programs cross larger offsets.
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn,
            _layer5_fetch_head_specs(BD),
            HD,
        )

    # Dim-ownership claims (see c4_release/docs/DIM_OWNERSHIP_REGISTRY.md).
    # `_set_layer5_fetch` writes attn5.W_v rows {base+32+k, base+48+k} for
    # k in 0..15 across heads 0..5 (V slots 32..47 and 48..63 per head):
    #
    #   W_v[head*HD + 32 + k, CLEAN_EMBED_LO + k] = 1.0   (slot 32..47)
    #   W_v[head*HD + 48 + k, CLEAN_EMBED_HI + k] = 1.0   (slot 48..63)
    #
    # The 4-tuple claim's ``column`` carries the input-dim name + offset so
    # that a column-disjoint co-tenant on the same row (e.g.,
    # ``function_call_weights``' EMBED_HI+15 ENT relay on row 5_32) does
    # NOT register as a collision. Pre-column-granularity registry treated
    # this as ``(5, "attn_W_v", "5_32")`` and produced one false positive
    # that lived in ``KNOWN_BENIGN_COLLISIONS``; now retired.
    #
    # Heads 6 and 7 were deleted on 2026-05-11 (commit c1a5398) to break
    # the latent collision with `_set_function_call_weights`' head 6 ENT
    # relay (V slots 1..16) — the registry would have caught that as
    # (5, "attn_W_v", "6_<k>", "EMBED_LO+<k>") for k in 1..16.
    _claims = set()
    for head in range(6):  # heads 0..5
        for k in range(16):  # k = 0..15
            slot_lo = 32 + k
            slot_hi = 48 + k
            _claims.add(
                (5, "attn_W_v", f"{head}_{slot_lo}", f"CLEAN_EMBED_LO+{k}")
            )
            _claims.add(
                (5, "attn_W_v", f"{head}_{slot_hi}", f"CLEAN_EMBED_HI+{k}")
            )

    return Operation(
        name="layer5_fetch",
        # Reads: PC/AX markers + FETCH addr (PC+K) + ADDR_KEY (per CODE byte) +
        #        CLEAN_EMBED (the value at the matched CODE byte).
        # Note: heads 6/7 also read OP_* via V projection but that's the DEPRECATED
        # path (OP_* flags were removed from embeddings 2026-04-13). Excluding from
        # reads since they're not semantically active inputs.
        reads={"MARK_PC", "MARK_AX", "HAS_SE",
               "FETCH_LO", "FETCH_HI", "EMBED_LO", "EMBED_HI",
               "ADDR_KEY", "CONST", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                "FETCH_LO", "FETCH_HI",
                "OP_IMM", "OP_LEA", "OP_EXIT", "OP_JMP", "OP_JSR",
                "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
                "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_LT", "OP_SHL", "OP_SHR"},
        kind="block",
        # Phase 8.G.6: drop ``layer_idx=5`` literal; bind to the L5 attn
        # dep anchor so the block op resolves to whichever layer the
        # compiler places the anchor at.
        target_op_name="_layer5_fetch_dep_anchor",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer5_fetch_ir,
        migrated=True,
        claims=_claims,
        # Phase 8.A targeted (SCC audit step 7): pin the ADDR_KEY reader to
        # layer4_pc_relay so the R-OH-2 rule in the dim-flow analyser
        # suppresses the spurious back-edges from later ADDR_KEY writers
        # (``layer7_memory_heads``, ``layer14_clear_addr_key_pollution``,
        # ``layer14_addr_key_neural_decode``) into this op. L5 fetch's K-side
        # consumes the AX-marker top nibble that ``layer4_pc_relay`` writes
        # same-step; the L7/L14 ADDR_KEY writes are not the source.
        requires={"after": "layer4_pc_relay"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _layer5_fetch_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_layer5_fetch_head_specs(BD))
    return ir


def _band_projection_writes(slot_base: int, dim_base: int, weight: float = 1.0):
    return tuple(AP(slot_base + k, dim_base + k, weight) for k in range(16))


def _band_output_writes(dim_base: int, slot_base: int, weight: float = 1.0):
    return tuple(AO(dim_base + k, slot_base + k, weight) for k in range(16))


def _addr_key_match_writes(BD, weight: float, top_slot_base: int = 35):
    return (
        tuple(AP(k, BD.ADDR_KEY + k, weight) for k in range(16))
        + tuple(AP(16 + k, BD.ADDR_KEY + 16 + k, weight) for k in range(16))
        + tuple(AP(top_slot_base + k, BD.ADDR_KEY + 32 + k, weight) for k in range(16))
    )


def _code_fetch_v_writes(BD, weight: float = 1.0):
    return (
        _band_projection_writes(32, BD.CLEAN_EMBED_LO, weight)
        + _band_projection_writes(48, BD.CLEAN_EMBED_HI, weight)
    )


def _layer5_fetch_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative replacement for ``setup_helpers._set_layer5_fetch``."""

    from ...constants import PC_OFFSET

    L = 20.0
    ADDR_L = 20.0

    def ax_gate(slot: int = 33):
        return (
            (AP(slot, BD.MARK_AX, 500.0), AP(slot, BD.CONST, -500.0)),
            (AP(slot, BD.CONST, 5.0),),
        )

    def pc_gate(slot: int = 33):
        return (
            (AP(slot, BD.MARK_PC, 500.0), AP(slot, BD.CONST, -500.0)),
            (AP(slot, BD.CONST, 5.0),),
        )

    ax_q_gate, ax_k_gate = ax_gate()
    pc_q_gate, pc_k_gate = pc_gate()
    pc_lo = PC_OFFSET & 0xF
    pc_hi = (PC_OFFSET >> 4) & 0xF
    pc_top = (PC_OFFSET >> 8) & 0xF
    TOP = 35

    def dynamic_top_q():
        return tuple(AP(TOP + k, BD.ADDR_KEY + 32 + k, ADDR_L) for k in range(16))

    def first_step_top0_q():
        return (
            AP(TOP, BD.CONST, ADDR_L),
            AP(TOP, BD.HAS_SE, -ADDR_L),
        )

    specs = [
        # Head 0: non-first-step immediate fetch at AX from TEMP=PC+1.
        DeclarativeAttentionHeadSpec(
            head_idx=0,
            q=(
                tuple(AP(k, BD.TEMP + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.TEMP + 16 + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_AX, L),)
                + dynamic_top_q()
                + ax_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.FETCH_LO, 32)
                + _band_output_writes(BD.FETCH_HI, 48)
            ),
        ),
        # Head 1: non-first-step opcode fetch at AX from relayed PC.
        DeclarativeAttentionHeadSpec(
            head_idx=1,
            q=(
                tuple(AP(k, BD.EMBED_LO + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.EMBED_HI + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_AX, L),)
                + dynamic_top_q()
                + ax_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
        # Head 2: first-step opcode fetch at PC marker.
        DeclarativeAttentionHeadSpec(
            head_idx=2,
            q=(
                AP(pc_lo, BD.CONST, ADDR_L),
                AP(16 + pc_hi, BD.CONST, ADDR_L),
                AP(32, BD.MARK_PC, L),
                AP(TOP + pc_top, BD.CONST, ADDR_L),
                *pc_q_gate,
                AP(34, BD.HAS_SE, -500.0),
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + pc_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
        # Head 3: dynamic immediate fetch at PC marker.
        DeclarativeAttentionHeadSpec(
            head_idx=3,
            q=(
                tuple(AP(k, BD.FETCH_LO + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.FETCH_HI + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_PC, L),)
                + dynamic_top_q()
                + first_step_top0_q()
                + pc_q_gate
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + pc_k_gate,
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.FETCH_LO, 32, 40.0)
                + _band_output_writes(BD.FETCH_HI, 48, 40.0)
            ),
        ),
        # Head 4: first-step opcode fetch at AX marker.
        DeclarativeAttentionHeadSpec(
            head_idx=4,
            q=(
                AP(pc_lo, BD.CONST, ADDR_L),
                AP(16 + pc_hi, BD.CONST, ADDR_L),
                AP(32, BD.MARK_AX, L),
                AP(TOP + pc_top, BD.CONST, ADDR_L),
                *ax_q_gate,
                AP(34, BD.HAS_SE, -500.0),
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
        # Head 5: non-first-step opcode fetch at PC marker.
        DeclarativeAttentionHeadSpec(
            head_idx=5,
            q=(
                tuple(AP(k, BD.EMBED_LO + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.EMBED_HI + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_PC, L),)
                + dynamic_top_q()
                + pc_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + pc_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
    ]

    return tuple(specs)


def make_layer5_fetch_dep_anchor_op() -> Operation:
    """No-op companion for layer5_fetch: declares identical reads/writes so
    the LayerCompiler's dep graph reserves a layer slot for it. The actual
    bake happens in `layer5_fetch` (kind="block", layer_idx=5); this op's
    bake is a no-op (its layout-assigned attention block is unrelated to
    block[5] and is overwritten by legacy_bake of the corresponding L6 attn).
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer5_fetch` block op above.
        return

    return Operation(
        name="_layer5_fetch_dep_anchor",
        reads={"MARK_PC", "MARK_AX", "HAS_SE",
               "FETCH_LO", "FETCH_HI", "EMBED_LO", "EMBED_HI",
               "ADDR_KEY", "CONST", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                "FETCH_LO", "FETCH_HI",
                "OP_IMM", "OP_LEA", "OP_EXIT", "OP_JMP", "OP_JSR",
                "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
                "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_LT", "OP_SHL", "OP_SHR"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 8.A targeted (SCC audit step 7): mirror the
        # ``layer5_fetch`` ``requires["after"]`` so the dep anchor's
        # ADDR_KEY back-edges from L7/L14 are also suppressed via R-OH-2.
        # The anchor's reads/writes track the real ``layer5_fetch`` op.
        requires={"after": "layer4_pc_relay"},
        smoke_tests=set(),
        spec_section=None,
    )


def make_opcode_decode_ffn_op() -> Operation:
    """L5 FFN: decode opcode byte → 34 one-hot OP_* flags at OPCODE_BASE.

    Dispatched as a block op pinned to layer_idx=5 so the bake hits the same
    transformer block (block[5].ffn) the legacy path used. Using kind="block"
    routes through compile_full_vm's block_ops dispatch even when legacy_bake
    is present, ensuring block[5] receives the opcode decode logic for
    pure_neural execution. The companion `_opcode_decode_ffn_dep_anchor` op
    declares the same reads/writes via kind="ffn" so the LayerCompiler's dep
    graph still reserves a layer slot for it (preserving model n_layers).
    """
    def bake(block, dim_positions, S):
        # Per-bake FFN-unit allocator. Each L5 FFN sub-stage is pinned
        # to its existing offset so the call below lands byte-identically.
        # The allocator is published on ``block.ffn`` for inspection /
        # extension by downstream tools (e.g. a future L5 op family
        # claiming a free gap past unit 89). Mirrors the L9 pattern in
        # ``make_layer9_alu_op`` and the L4 pattern in
        # ``make_layer4_ffn_op``.
        allocator = _allocate_layer5_ffn_units()
        block.ffn._l5_unit_allocator = allocator

        # Phase 8.C inline cut: lower the ``_opcode_decode_ffn_rules`` IR
        # directly here so census v2 classifies this op as
        # ``declarative`` rather than ``declarative_via_helper`` (which
        # routed through the ``_bake_opcode_decode_ffn`` trampoline).
        # Byte-identical to the prior helper call.
        proxy = _as_setdim_proxy(dim_positions)
        rules = _opcode_decode_ffn_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy,
            Primitives.ffn_rule_dim_names(rules),
        )
        final_unit = Primitives.lower_ffn_rules(
            block.ffn, rules, rule_dim_positions, start_unit=0, S=S,
        )
        # Byte-identity guard: the helper's local cursor MUST end exactly
        # at the allocator's declared footprint. If the layout table
        # drifts from the helper's writes, this assertion fires before
        # any weight surgery happens.
        assert final_unit == _L5_FFN_TOTAL_UNITS, (
            f"L5 FFN unit cursor drift: helper returned {final_unit}, "
            f"allocator expected {_L5_FFN_TOTAL_UNITS}"
        )

    # Dim-ownership claims (W_down output cells). The bake programs four
    # blocks of L5 FFN units (see ``_bake_opcode_decode_ffn``):
    #   units 0..33:   main rules — one unit per opcode at MARK_AX,
    #                  writing the matching OP_* dim. Unit 35 is the JSR
    #                  detector writing TEMP+0.
    #   units 34..51:  first-step PC-marker decode (18 rules).
    #                  Unit 34: JMP, 35: JSR (TEMP+0), 36: IMM, 37: LEA,
    #                  38: EXIT, 39: NOP, 40..44: ADD..MOD,
    #                  45/46: OR/XOR, 47: AND, 48: EQ, 49: LT,
    #                  50: SHL, 51: SHR.
    #   unit 52:       reserved blank (preserves legacy unit numbering for
    #                  the first-step JSR-flag slot above).
    #   units 53..83:  TEMP[1..31] clear at MARK_PC — unit (52+k) writes
    #                  TEMP+k.
    #   units 84..88:  all-step PC-marker decode (BZ, BNZ, LEV, EXIT, JMP).
    _claims = set()
    # Main per-opcode units at MARK_AX, in the order from _opcode_decode_main_rules.
    _main_outputs = [
        "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
        "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
        "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
        "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
        "OP_SHL", "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
        "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
        "OP_PUTCHAR", "OP_GETCHAR",
    ]
    for unit, op_dim in enumerate(_main_outputs):
        _claims.add((5, "ffn_W_down", str(unit), f"{op_dim}+0"))
    # First-step PC-marker decode (18 rules at units 34..51).
    _first_step_outputs = [
        ("34", "OP_JMP+0"),
        ("35", "TEMP+0"),  # JSR's IS_JSR flag
        ("36", "OP_IMM+0"),
        ("37", "OP_LEA+0"),
        ("38", "OP_EXIT+0"),
        ("39", "OP_NOP+0"),
        ("40", "OP_ADD+0"),
        ("41", "OP_SUB+0"),
        ("42", "OP_MUL+0"),
        ("43", "OP_DIV+0"),
        ("44", "OP_MOD+0"),
        ("45", "OP_OR+0"),
        ("46", "OP_XOR+0"),
        ("47", "OP_AND+0"),
        ("48", "OP_EQ+0"),
        ("49", "OP_LT+0"),
        ("50", "OP_SHL+0"),
        ("51", "OP_SHR+0"),
    ]
    for unit, col in _first_step_outputs:
        _claims.add((5, "ffn_W_down", unit, col))
    # Unit 52 is the reserved blank for first-step JSR's TEMP[0] (see
    # ``_bake_opcode_decode_ffn`` -- ``unit += 1`` before TEMP-clear).
    # TEMP[1..31] clear units 53..83.
    for k in range(1, 32):
        _claims.add((5, "ffn_W_down", str(52 + k), f"TEMP+{k}"))
    # All-step PC-marker decode (BZ, BNZ, LEV, EXIT, JMP at units 84..88).
    for unit, col in (
        ("84", "OP_BZ+0"),
        ("85", "OP_BNZ+0"),
        ("86", "OP_LEV+0"),
        ("87", "OP_EXIT+0"),
        ("88", "OP_JMP+0"),
    ):
        _claims.add((5, "ffn_W_down", unit, col))

    return Operation(
        name="opcode_decode_ffn",
        # Phase 8.A SCC step 6: read OPCODE_BYTE_LO via the
        # ``OPCODE_BYTE_LO_PREV_STEP`` alias (same numeric base, see
        # shared.py ``_ALIAS_OF``) so the dynamic scheduler sees this
        # consumption as a prev-step residual read rather than a
        # same-layer dep on ``layer5_fetch``. Drops the L5
        # fetch->decode writes/reads edge from the SCC.
        reads={"OPCODE_BYTE_LO_PREV_STEP",
               "OPCODE_BYTE_HI", "MARK_AX", "MARK_PC", "HAS_SE"},
        writes={"OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
                "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
                "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                "OP_SHL", "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
                "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
                "OP_PUTCHAR", "OP_GETCHAR",
                "TEMP"},  # JSR writes IS_JSR to TEMP[0]
        kind="block",
        # Phase 8.G.6: drop ``layer_idx=5`` literal; bind to the L5
        # opcode-decode ffn dep anchor so the block op resolves to
        # whichever layer the compiler places the anchor at.
        target_op_name="_opcode_decode_ffn_dep_anchor",
        declarative_bake_fn=bake,
        compiler_ir=_opcode_decode_ffn_ir(),
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _opcode_decode_main_rules(S):
    """CompilerIR rules for opcode byte decode at the AX marker.

    Phase 8.D: the per-opcode write targets resolve through
    :func:`dim_ref` for the ``(opcode_flag, <op_name>)`` semantic
    family lookup -- each rule's output dim names the opcode-flag
    role it asserts. The ``MARK_AX`` gate also moves to
    ``dim_ref("marker", "AX")``. Structural ``OPCODE_BYTE_LO/HI+<lo|hi>``
    operand reads stay as ``+N`` (per-nibble one-hot lookup indices
    on the opcode byte's nibble decomposition).
    """

    from ...embedding import Opcode

    opcodes = [
        (Opcode.LEA, 0, 0),
        (Opcode.IMM, 1, 0),
        (Opcode.JMP, 2, 0),
        (Opcode.JSR, 3, 0),
        (Opcode.BZ, 4, 0),
        (Opcode.BNZ, 5, 0),
        (Opcode.ENT, 6, 0),
        (Opcode.ADJ, 7, 0),
        (Opcode.LEV, 8, 0),
        (Opcode.LI, 9, 0),
        (Opcode.LC, 10, 0),
        (Opcode.SI, 11, 0),
        (Opcode.SC, 12, 0),
        (Opcode.PSH, 13, 0),
        (Opcode.OR, 14, 0),
        (Opcode.XOR, 15, 0),
        (Opcode.AND, 0, 1),
        (Opcode.EQ, 1, 1),
        (Opcode.NE, 2, 1),
        (Opcode.LT, 3, 1),
        (Opcode.GT, 4, 1),
        (Opcode.LE, 5, 1),
        (Opcode.GE, 6, 1),
        (Opcode.SHL, 7, 1),
        (Opcode.SHR, 8, 1),
        (Opcode.ADD, 9, 1),
        (Opcode.SUB, 10, 1),
        (Opcode.MUL, 11, 1),
        (Opcode.DIV, 12, 1),
        (Opcode.MOD, 13, 1),
        (Opcode.EXIT, 6, 2),
        (Opcode.NOP, 7, 2),
        (Opcode.PUTCHAR, 1, 4),
        (Opcode.GETCHAR, 0, 4),
    ]
    op_names = _opcode_name_map()
    gate_mark_ax = dim_ref("marker", "AX")
    return tuple(
        FFNRule.gated_write(
            name=f"l5_decode_{op_names[op_val].lower()}_at_ax",
            conditions=(
                (f"OPCODE_BYTE_LO+{lo}", 1.0),
                (f"OPCODE_BYTE_HI+{hi}", 1.0),
            ),
            threshold=1.5,
            gate=gate_mark_ax,
            # ``op_names[op_val]`` is the ``"OP_<NAME>"`` slot string;
            # rewriting it via dim_ref names the (opcode_flag, NAME)
            # role lookup.
            writes=((dim_ref("opcode_flag", op_names[op_val][3:]), 10.0 / S),),
        )
        for op_val, lo, hi in opcodes
    )


def _opcode_decode_first_step_rules(S):
    """CompilerIR rules for first-step PC-marker opcode decode.

    Phase 8.D: ``OP_<NAME>`` write targets use :func:`dim_ref` for the
    ``(opcode_flag, <NAME>)`` semantic pair. The single ``TEMP+0``
    write (JSR's first-step IS_JSR flag) stays structural -- ``TEMP+0``
    is a scratch-slot offset, not a role-meaningful byte position
    or opcode-flag family member.
    """

    # Each entry is ``(opcode_lo_nibble, opcode_hi_nibble, out_dim)`` where
    # ``out_dim`` is either ``dim_ref("opcode_flag", NAME)`` (the
    # decoded opcode-flag family member) or the bare ``"TEMP+0"`` slot
    # (JSR's first-step IS_JSR flag, owned by the temp_scratch family
    # but without a dedicated role binding).
    first_step_opcodes = [
        (2, 0, dim_ref("opcode_flag", "JMP")),
        (3, 0, "TEMP+0"),
        (1, 0, dim_ref("opcode_flag", "IMM")),
        (0, 0, dim_ref("opcode_flag", "LEA")),
        (6, 2, dim_ref("opcode_flag", "EXIT")),
        (7, 2, dim_ref("opcode_flag", "NOP")),
        (9, 1, dim_ref("opcode_flag", "ADD")),
        (10, 1, dim_ref("opcode_flag", "SUB")),
        (11, 1, dim_ref("opcode_flag", "MUL")),
        (12, 1, dim_ref("opcode_flag", "DIV")),
        (13, 1, dim_ref("opcode_flag", "MOD")),
        (14, 0, dim_ref("opcode_flag", "OR")),
        (15, 0, dim_ref("opcode_flag", "XOR")),
        (0, 1, dim_ref("opcode_flag", "AND")),
        (1, 1, dim_ref("opcode_flag", "EQ")),
        (3, 1, dim_ref("opcode_flag", "LT")),
        (7, 1, dim_ref("opcode_flag", "SHL")),
        (8, 1, dim_ref("opcode_flag", "SHR")),
    ]
    return tuple(
        FFNRule.constant_write(
            # Preserve the legacy rule-name suffix shape by stripping
            # the ``+0`` produced by ``dim_ref`` (turning ``OP_JMP+0``
            # back into ``op_jmp``).
            name=(
                f"l5_first_step_decode_"
                f"{out_dim.lower().replace('+', '_')}"
                if out_dim.startswith("TEMP")
                else f"l5_first_step_decode_"
                     f"{out_dim.split('+', 1)[0].lower()}"
            ),
            conditions=(
                (f"OPCODE_BYTE_LO+{lo}", 1.0),
                (f"OPCODE_BYTE_HI+{hi}", 1.0),
                ("MARK_PC", 1.0),
                ("HAS_SE", -1.0),
            ),
            threshold=2.5,
            writes=((out_dim, 10.0 / S),),
        )
        for lo, hi, out_dim in first_step_opcodes
    )


def _opcode_decode_temp_clear_rules(S):
    """CompilerIR rules for TEMP[1..31] clearing at the PC marker.

    TEMP[0] is reserved for the first-step JSR flag, so the caller must leave
    one blank hidden unit before lowering these rules to preserve legacy unit
    numbering.
    """

    # No role-meaningful refs here: the ``gate=f"TEMP+{k}"`` / write target
    # use ``k`` as a structural scratch-slot index (not a role), and the
    # ``MARK_PC`` condition appears as an up-branch guard term (the L8 pilot
    # preserved that convention).
    return tuple(
        FFNRule.gated_write(
            name=f"l5_temp_clear_{k}_at_pc",
            conditions=(("MARK_PC", 1.0),),
            threshold=0.5,
            gate=f"TEMP+{k}",
            gate_weight=-1.0,
            writes=((f"TEMP+{k}", 2.0 / S),),
        )
        for k in range(1, 32)
    )


def _opcode_decode_all_step_pc_rules(S):
    """CompilerIR rules for all-step PC-marker opcode decode.

    Phase 8.D: the per-opcode write target resolves through
    :func:`dim_ref` for the ``(opcode_flag, <NAME>)`` semantic pair.
    Structural ``OPCODE_BYTE_LO/HI+<lo|hi>`` operand reads stay as
    ``+N`` and the ``MARK_PC`` up-branch guard condition stays bare
    (the L8 pilot kept marker-on-up-branch terms structural).
    """

    from ...embedding import Opcode

    all_step_opcodes = [
        (Opcode.BZ, 4, 0),
        (Opcode.BNZ, 5, 0),
        (Opcode.LEV, 8, 0),
        (Opcode.EXIT, 6, 2),
        (Opcode.JMP, 2, 0),
    ]
    op_names = _opcode_name_map()
    return tuple(
        FFNRule.constant_write(
            name=f"l5_all_step_decode_{op_names[op_val].lower()}_at_pc",
            conditions=(
                (f"OPCODE_BYTE_LO+{lo}", 1.0),
                (f"OPCODE_BYTE_HI+{hi}", 1.0),
                ("MARK_PC", 1.0),
            ),
            threshold=2.5,
            # ``op_names[op_val]`` is the ``"OP_<NAME>"`` slot string;
            # ``dim_ref("opcode_flag", NAME)`` names the role.
            writes=((dim_ref("opcode_flag", op_names[op_val][3:]), 10.0 / S),),
        )
        for op_val, lo, hi in all_step_opcodes
    )


def _opcode_decode_jsr_temp0_blank_rule() -> FFNRule:
    """Blank-unit placeholder for the reserved unit-52 JSR TEMP[0] slot.

    The legacy ``_set_opcode_decode_ffn`` increments its hidden-unit cursor
    past unit 52 without writing any weights there ("preserve legacy unit
    numbering for the TEMP-clear band that follows"). To carry the same
    layout through a single declarative ``CompilerIR`` lower, we emit one
    no-op ``FFNRule`` whose lowering matches a zero-initialised PureFFN row
    byte-for-byte:

      * ``conditions=()`` → ``W_up[52, :] = 0``
      * ``threshold=0.0`` → ``b_up[52] = -S * 0 = 0``
      * ``gate=None`` and ``gate_bias=0.0`` → ``b_gate[52] = 0`` (note:
        ``FFNRule.constant_write`` defaults ``gate_bias=1.0``, so we
        construct the rule directly to override the default to 0.0)
      * ``writes=()`` → ``W_down[:, 52] = 0``

    ``right_size_ffns`` correctly prunes this unit because every weight
    column / row remains all-zero. Symbolic execution and lowered forward
    both produce no state change for this rule (score=0 >= threshold=0
    fires but ``gate_value = 0`` plus empty ``writes`` is a no-op).
    """

    return FFNRule(
        conditions=(),
        threshold=0.0,
        writes=(),
        gate=None,
        gate_bias=0.0,
        name="l5_opcode_decode_jsr_temp0_blank",
    )


def _opcode_decode_ffn_rules(S: float) -> tuple[FFNRule, ...]:
    """Full ordered ``FFNRule`` sequence for ``opcode_decode_ffn``.

    Matches the 89-unit layout declared in ``_L5_FFN_UNIT_LAYOUT``:

      * units 0..33  — main per-opcode AX decode (34 rules)
      * units 34..51 — first-step PC-marker decode (18 rules)
      * unit 52      — reserved blank for JSR TEMP[0] (1 no-op rule)
      * units 53..83 — TEMP[1..31] clear at PC marker (31 rules)
      * units 84..88 — all-step PC-marker decode (5 rules)

    Concatenating them into a single ``FFNRule`` tuple lets the bake lower
    via one ``Primitives.lower_ffn_rules`` call (cursor walks 0..89) and
    lets the op expose its full ``compiler_ir`` for symbolic execution /
    ``compare_symbolic_to_lowered_ffn`` validation / declarative
    verifier tooling.
    """

    return (
        _opcode_decode_main_rules(S)
        + _opcode_decode_first_step_rules(S)
        + (_opcode_decode_jsr_temp0_blank_rule(),)
        + _opcode_decode_temp_clear_rules(S)
        + _opcode_decode_all_step_pc_rules(S)
    )


def _opcode_decode_ffn_ir(S: float = 100.0) -> CompilerIR:
    """Build the declarative ``CompilerIR`` exposed by ``opcode_decode_ffn``."""

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_opcode_decode_ffn_rules(S))
    return ir


def _lower_l5_opcode_rules(ffn, rules, BD, *, unit: int, S: float) -> int:
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=unit,
        S=S,
    )


def _bake_opcode_decode_ffn(ffn, S, BD) -> int:
    """Declarative L5 FFN spec: opcode-byte one-hot decode.

    Returns the post-bake unit cursor (must equal
    :data:`_L5_FFN_TOTAL_UNITS` for byte-identity with the historical
    89-unit footprint). The caller asserts this in
    ``make_opcode_decode_ffn_op``.

    Lowers the single ``_opcode_decode_ffn_rules`` tuple through
    ``Primitives.lower_ffn_rules`` -- the rule list embeds the unit-52
    blank placeholder (see :func:`_opcode_decode_jsr_temp0_blank_rule`)
    so the cursor walks 0..89 with no per-sub-stage cursor surgery. This
    matches the IR returned by :func:`_opcode_decode_ffn_ir`, which the
    op exposes via ``compiler_ir=`` for symbolic / verifier tooling.
    """

    return _lower_l5_opcode_rules(
        ffn,
        _opcode_decode_ffn_rules(S),
        BD,
        unit=0,
        S=S,
    )


def make_opcode_decode_ffn_dep_anchor_op() -> Operation:
    """No-op companion for opcode_decode_ffn: declares identical reads/writes
    so the LayerCompiler's dep graph reserves a layer slot for it. The actual
    bake happens in `opcode_decode_ffn` (kind="block", layer_idx=5); this op's
    bake is a no-op.
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is in `opcode_decode_ffn` block op above.
        return

    return Operation(
        name="_opcode_decode_ffn_dep_anchor",
        # Phase 8.A SCC step 6: matches opcode_decode_ffn's
        # OPCODE_BYTE_LO_PREV_STEP rename (same numeric base, prev-step
        # semantics).
        reads={"OPCODE_BYTE_LO_PREV_STEP",
               "OPCODE_BYTE_HI", "MARK_AX", "MARK_PC", "HAS_SE"},
        writes={"OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
                "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
                "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                "OP_SHL", "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
                "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
                "OP_PUTCHAR", "OP_GETCHAR",
                "TEMP"},
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests=set(),
        spec_section=None,
    )
