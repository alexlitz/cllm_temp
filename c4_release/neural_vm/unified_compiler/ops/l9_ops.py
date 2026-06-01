"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L9 attention-head layout (pinned indices) ======================
#
# Three migrated ops claim heads on the L9 attention block today:
#
#   * ``layer9_lev_addr_relay`` (phase=9.0)      -> head 0
#   * ``layer9_lev_bp_to_pc_relay`` (phase=9.1)  -> head 1
#   * ``layer9_alibi_mem_attn`` (phase=9.2)      -> head 2
#
# ``layer9_alibi_mem_attn`` is conditionally enabled at runtime via
# ``enable=False``; we still claim its head index in the allocator so
# the layout is structurally stable across builds (the bake itself
# early-returns when the gate is off, so no weights move either way).
# Pre-migration each call site picked its ``head_idx`` as a bare integer
# literal -- ``head_idx=0`` / ``head_idx=1`` in the relay
# :class:`DeclarativeAttentionHeadSpec` calls and ``head = 2`` in the
# ALiBi bake -- which made adding a future L9 head fragile (the author
# had to remember which slots were already taken). With the allocator
# the handoff is structural: each bake instantiates its OWN allocator
# pre-loaded with the full L9 head layout (pinned to existing slots),
# stashes it on ``attn._l9_head_allocator`` for downstream inspection,
# and resolves its own head index by name. A future L9 attention op
# can claim a free head via ``allocator.alloc(name, layer_idx=9)`` --
# with no ``pin=`` -- without touching this table.
#
# NOTE: ``format_string_fetch_head`` (gated by ``enable_conversational_io``)
# also writes head 0 in the conversational-I/O path, intentionally
# clobbering ``layer9_lev_addr_relay`` slopes/weights. That aliasing
# pre-dates the allocator and is not modeled here; the allocator
# forbids head aliasing, so the convo-I/O head stays outside this
# layout until a follow-up reconciles the two ops onto distinct slots.
_L9_HEAD_LAYOUT = (
    # (op-name key,                          pinned head_idx)
    ("layer9_lev_addr_relay",                0),
    ("layer9_lev_bp_to_pc_relay",            1),
    ("layer9_alibi_mem_attn",                2),
)


def _allocate_layer9_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L9 heads.

    Every entry in :data:`_L9_HEAD_LAYOUT` is pinned at its existing
    ``head_idx`` so the underlying primitive calls -- which still write
    the same weights to the same heads -- land byte-identically. Each
    of the three migrated L9 attention ops calls this so they can look
    up their own head by name without baking in an integer literal at
    the call site.
    """
    allocator = AttentionHeadAllocator()
    for name, head_idx in _L9_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=9, pin=head_idx)
    return allocator


def _l9_head_idx(op_name: str) -> int:
    """Return the pinned L9 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L9_HEAD_LAYOUT` for callers that
    cannot instantiate a per-bake allocator (e.g. the head-spec
    helpers consumed by both bake and ``compiler_ir_factory`` paths,
    where running the collision-checked allocator on every call
    would be wasteful). The runtime bakes still go through
    :func:`_allocate_layer9_attention_heads` so the collision-checked
    allocator path is exercised on every weight write.
    """
    for name, head_idx in _L9_HEAD_LAYOUT:
        if name == op_name:
            return head_idx
    raise KeyError(f"_l9_head_idx: unknown L9 attention op {op_name!r}")


# === L9 FFN unit layout (pinned offsets) ============================
#
# The ``layer9_alu`` op owns the entire L9 FFN. The actual weight writes
# happen inside ``vm_step._set_layer9_alu`` + ``_set_layer9_marker_suppress``,
# which use a local ``unit = 0`` counter that increments through 3405
# sub-stages. Migration to :class:`FFNUnitAllocator` keeps those helpers
# byte-identical -- we just declare each sub-stage's range at its existing
# pinned offset so the layout is auditable rather than implicit. Adding a new
# L9 op family later will go through ``allocator.alloc(name, n)`` without a
# pin, and the allocator will pick the first free gap below 3405 (or above).
#
# The offsets below mirror the unit-counter walk in
# ``vm_step._set_layer9_alu`` (carry/borrow doubled inner loops) followed by
# ``_set_layer9_marker_suppress`` (7 NEXT_* dims). Changing any helper's
# unit count requires updating this table in lock-step.
_L9_ALU_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    ("layer9_alu.add_hi_nibble",         0, 512),  # ADD hi nibble (carry x 256)
    ("layer9_alu.lea_hi_nibble",       512, 512),  # LEA hi nibble (carry x 256)
    ("layer9_alu.adj_hi_nibble",      1024, 512),  # ADJ hi nibble (carry x 256)
    ("layer9_alu.sub_hi_nibble",      1536, 512),  # SUB hi nibble (borrow x 256)
    ("layer9_alu.ent_hi_nibble",      2048, 512),  # ENT hi nibble (borrow x 256)
    ("layer9_alu.hi_eq",              2560,  16),  # CMP+1 hi-eq
    ("layer9_alu.lo_eq",              2576,  16),  # CMP+2 lo-eq
    ("layer9_alu.hi_lt",              2592, 120),  # CMP+0 hi-lt
    ("layer9_alu.lo_lt",              2712, 120),  # CMP+3 lo-lt
    ("layer9_alu.add_carry_out",      2832, 256),  # CARRY+1 add carry-out
    ("layer9_alu.sub_borrow_out",     3088, 256),  # CARRY+2 sub borrow-out
    ("layer9_alu.alu_lo_clear",       3344,  16),  # ALU_LO clear at non-ALU op
    ("layer9_alu.alu_hi_clear",       3360,  16),  # ALU_HI clear at non-ALU op
    ("layer9_alu.bp_plus8_shift",     3376,  16),  # ADDR_B0_LO BP+8 lo-nibble shift
    ("layer9_alu.addr_b1_lo_set",     3392,   1),  # ADDR_B1_LO+15
    ("layer9_alu.addr_b1_hi_set",     3393,   1),  # ADDR_B1_HI+15
    ("layer9_alu.cascade_b0_hi",      3394,   1),  # cascade ADDR_B0_HI[15]->[0]
    ("layer9_alu.cascade_b1_lo",      3395,   1),  # cascade ADDR_B1_LO[15]->[0]
    ("layer9_alu.cascade_b1_hi",      3396,   1),  # cascade ADDR_B1_HI[15]->[0]
    ("layer9_alu.cascade_b2_lo",      3397,   1),  # cascade ADDR_B2_LO+1
    ("layer9_alu.marker_suppress",    3398,   7),  # _set_layer9_marker_suppress NEXT_*
)


def _allocate_layer9_alu_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L9 ALU sub-stages.

    Every sub-stage is pinned at its existing offset so the underlying
    ``vm_step._set_layer9_alu`` helper -- which writes via its own
    monotonic ``unit = 0`` counter -- lands on exactly the same hidden-unit
    indices it always has. This call is byte-identical bookkeeping: the
    allocator declares ranges by name, the helper writes the weights. A
    future refactor can split the monolithic helper into per-range bake
    functions that consume ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L9 op claims a free range past unit 3405).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L9_ALU_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


def make_layer9_alu_op(alu_mode: str = "lookup") -> Operation:
    """L9 FFN: ADD/SUB hi nibble + bitwise ops byte 0, plus marker suppression.

    Originally two inline calls inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``n9 = _set_layer9_alu(ffn9, S, BD)``
        ``_set_layer9_marker_suppress(ffn9, S, BD, n9)``

    Combined into a single migrated bake_fn that captures ``n9`` and
    threads it to ``_set_layer9_marker_suppress`` as ``start_unit`` so the
    two routines share the FFN's hidden-unit allocator. Mirrors the
    combined-bake pattern proven safe by Unit 9's diagnosis (see
    ``c4_release/docs/LOOKUP_MODE_BUG_DIAGNOSIS.md``).

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``; the inline call pair has been removed from
    ``set_vm_weights`` to avoid double-bake. Phase stays at 9. Fires in
    both lookup and efficient ALU modes — the lookup-branch nesting was
    incidental and the helpers themselves are alu_mode-agnostic.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer9_alu, _set_layer9_marker_suppress
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L9 ALU sub-stage is pinned to
        # its existing offset so the calls below land byte-identically. The
        # ``marker_suppress`` range gives the start-unit for the
        # ``_set_layer9_marker_suppress`` chained call, replacing the
        # implicit ``n9`` cursor return.
        allocator = _allocate_layer9_alu_units()
        marker_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer9_alu.marker_suppress"
        )
        marker_start = marker_range.start
        # Make the allocator available for inspection / extension by
        # downstream tools (e.g. a future L9 op family claiming a free
        # gap). The block-level attribute mirrors the ``_l14_unit_counter``
        # convention used by sibling layers, but carries the allocator
        # object so the layout is structured, not just a monotonic int.
        block.ffn._l9_unit_allocator = allocator

        n9 = _set_layer9_alu(block.ffn, S, proxy)
        # Byte-identity guard: the helper's local cursor MUST end exactly
        # where the allocator's marker_suppress range starts. If the table
        # drifts from the helper's writes, this assertion fires before any
        # weight surgery happens.
        assert n9 == marker_start, (
            f"L9 ALU unit cursor drift: helper returned {n9}, allocator "
            f"expected {marker_start}"
        )
        if alu_mode == "efficient":
            _suppress_l9_legacy_addsub_writes(block.ffn, proxy)
        _set_layer9_marker_suppress(block.ffn, S, proxy, marker_start)

    # Dim-ownership claims (W_down output cells). Mirrors ``_set_layer9_alu``
    # in ``vm_step.py`` (3398 units) followed by ``_set_layer9_marker_suppress``
    # (7 units). The bake assigns hidden units in a fixed order from 0..3404
    # so the unit indices are deterministic per layer slot. We declare the
    # W_down output cells (partial-claims convention -- the per-output
    # ownership identifies which residual dims the op writes; input-side
    # W_up/W_gate selectors are left unclaimed to mirror the L14/L6 style).
    _claims: set = set()
    unit = 0
    # ADD hi nibble (carry_in in [0, 1]): 512 units -> W_down[OUTPUT_HI_THIS_STEP+result]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI_THIS_STEP+{result}"))
                unit += 1
    # LEA hi nibble: 512 units -> W_down[OUTPUT_HI_THIS_STEP+result]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI_THIS_STEP+{result}"))
                unit += 1
    # ADJ hi nibble: 512 units -> W_down[OUTPUT_HI_THIS_STEP+result]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI_THIS_STEP+{result}"))
                unit += 1
    # SUB hi nibble: 512 units -> W_down[OUTPUT_HI_THIS_STEP+result]
    for borrow_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a - b - borrow_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI_THIS_STEP+{result}"))
                unit += 1
    # ENT hi nibble: 512 units -> W_down[OUTPUT_HI_THIS_STEP+result]
    for borrow_in in (0, 1):
        for sp_hi in range(16):
            for imm_hi in range(16):
                result = (sp_hi - imm_hi - borrow_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI_THIS_STEP+{result}"))
                unit += 1
    # hi_eq: 16 units -> W_down[CMP+1]
    for _k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), "CMP+1"))
        unit += 1
    # lo_eq: 16 units -> W_down[CMP+2]
    for _k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), "CMP+2"))
        unit += 1
    # hi_lt: 120 units -> W_down[CMP+0]
    for a in range(16):
        for b in range(a + 1, 16):
            _claims.add((9, "ffn_W_down", str(unit), "CMP+0"))
            unit += 1
    # lo_lt: 120 units -> W_down[CMP+3]
    for a in range(16):
        for b in range(a + 1, 16):
            _claims.add((9, "ffn_W_down", str(unit), "CMP+3"))
            unit += 1
    # ADD hi-nibble carry-out: skips units where a+b+carry_in < 16.
    # 120 (carry_in=0) + 136 (carry_in=1) = 256 units -> W_down[CARRY+1]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                if a + b + carry_in < 16:
                    continue
                _claims.add((9, "ffn_W_down", str(unit), "CARRY+1"))
                unit += 1
    # SUB hi-nibble borrow-out: skips units with no borrow-out.
    # 120 (borrow_in=0: a<b) + 136 (borrow_in=1: a<=b) = 256 units -> W_down[CARRY+2]
    for borrow_in in (0, 1):
        for a in range(16):
            for b in range(16):
                if borrow_in == 0:
                    if a >= b:
                        continue
                else:
                    if a > b:
                        continue
                _claims.add((9, "ffn_W_down", str(unit), "CARRY+2"))
                unit += 1
    # ALU clearing LO: 16 units -> W_down[ALU_LO+k]
    for k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), f"ALU_LO+{k}"))
        unit += 1
    # ALU clearing HI: 16 units -> W_down[ALU_HI+k]
    for k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), f"ALU_HI+{k}"))
        unit += 1
    # BP+8 shift to ADDR_B0_LO: 16 units; each writes ADDR_B0_LO[k] (cancel)
    # and ADDR_B0_LO[(k+8)%16] (set).
    for k in range(16):
        new_k = (k + 8) % 16
        _claims.add((9, "ffn_W_down", str(unit), f"ADDR_B0_LO+{k}"))
        _claims.add((9, "ffn_W_down", str(unit), f"ADDR_B0_LO+{new_k}"))
        unit += 1
    # ADDR_B1=0xff at PC marker for LEV (2 units): ADDR_B1_LO[15], ADDR_B1_HI[15]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_LO+15"))
    unit += 1
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_HI+15"))
    unit += 1
    # Cascade carry units for BP=0xfff8 + 8 = 0x10000:
    # Unit +0: clear ADDR_B0_HI[15], set ADDR_B0_HI[0]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B0_HI+15"))
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B0_HI+0"))
    unit += 1
    # Unit +1: cancel ADDR_B1_LO[15], set ADDR_B1_LO[0]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_LO+15"))
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_LO+0"))
    unit += 1
    # Unit +2: cancel ADDR_B1_HI[15], set ADDR_B1_HI[0]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_HI+15"))
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_HI+0"))
    unit += 1
    # Unit +3: set ADDR_B2_LO[1]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B2_LO+1"))
    unit += 1
    # _set_layer9_marker_suppress: 7 units (one per NEXT_* dim), each writes
    # W_down[OUTPUT_LO+k] and W_down[OUTPUT_HI_THIS_STEP+k] for k in 0..15.
    for _next_dim in (
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
        "NEXT_STACK0", "NEXT_MEM", "NEXT_SE",
    ):
        for k in range(16):
            _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_LO+{k}"))
            _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI_THIS_STEP+{k}"))
        unit += 1
    # Total expected unit index after bake: 3405 (matches ffn_units_used).
    _claims = frozenset(_claims)

    return Operation(
        name="layer9_alu",
        phase=9,
        reads={"MARK_AX", "MARK_PC", "ALU_HI", "AX_CARRY_HI", "FETCH_HI", "CARRY",
               "OP_ADD", "OP_SUB", "OP_OR", "OP_XOR", "OP_AND",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "ALU_LO", "AX_CARRY_LO"},
        writes={"OUTPUT_HI_THIS_STEP", "CMP", "OUTPUT_LO", "CARRY"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=9,
        migrated=True,
        claims=_claims,
        # Staleness invariants: the L9 ALU consumes ALU_HI as operand A hi
        # nibble at the AX marker. Produced by ``layer7_operand_gather`` (L7
        # head 0 + head 1, phase=7) at AX byte 0.
        consumes_fresh={
            "ALU_HI": "AX_byte0",
        },
        # ``_set_layer9_alu`` writes the ADD/LEA/SUB/AND/OR/XOR/CMP/etc.
        # cross-product cluster (~3398 units), and the bake chains into
        # ``_set_layer9_marker_suppress`` for 7 more NEXT_* suppression
        # units. Cumulative L9 FFN max: 3405. No other op writes to L9
        # FFN so this op holds the per-layer width annotation.
        ffn_units_used=3405,
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
            "TestSmokeComparison::test_eq_false",
            "TestSmokeComparison::test_eq_true",
            "TestSmokeComparison::test_ge_true",
            "TestSmokeComparison::test_gt_true",
            "TestSmokeComparison::test_le_true",
            "TestSmokeComparison::test_lt_true",
            "TestSmokeComparison::test_ne_true",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def _suppress_l9_legacy_addsub_writes(ffn, BD) -> None:
    """Let the efficient add/sub block own byte carry/borrow state.

    L8/L9 legacy nibble logic leaves ``CARRY[0]`` as an intra-byte
    carry/borrow signal. The efficient AddSub5StageBlock that now runs before
    L9 computes the full byte result and the byte-level ``CARRY[1]/CARRY[2]``
    needed by the later carry-propagation post-ops. L9's legacy ADD/SUB
    hi-nibble units see amplified raw one-hot operands at the AX marker in the
    efficient path and can swamp the already-correct marker result. Zero just
    those ADD/SUB legacy output units and the legacy carry rows; comparisons,
    LEA/ADJ/ENT, and bitwise units remain intact.
    """

    # _set_layer9_alu unit layout:
    #   0..511: ADD hi nibble
    #   512..1023: LEA hi nibble
    #   1024..1535: ADJ hi nibble
    #   1536..2047: SUB hi nibble
    add_units = slice(0, 512)
    sub_units = slice(1536, 2048)
    ffn.W_down.data[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, add_units] = 0.0
    ffn.W_down.data[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, sub_units] = 0.0
    ffn.W_down.data[BD.CARRY + 1, :] = 0.0
    ffn.W_down.data[BD.CARRY + 2, :] = 0.0


def make_layer9_lev_addr_relay_op() -> Operation:
    """L9 attention head 0: BP byte 0 → ADDR_B0 at SP marker for LEV.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_addr_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.0 to preserve ordering with sibling
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1). Fires in both lookup
    and efficient ALU modes — the helper performs identical setup
    regardless of alu_mode, and the model is built once.

    Also sets ``alibi_slopes[0] = 0.2`` (shallow slope for d=29 relay
    from SP marker back to previous BP byte 0); previously set inline
    alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        # Per-bake attention-head allocator with the full L9 head layout
        # pinned. Looking up the relay head by name keeps its index
        # identical to the legacy ``head_idx=0`` literal without baking
        # the integer into the call site.
        allocator = _allocate_layer9_attention_heads()
        attn._l9_head_allocator = allocator
        relay_head_idx = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer9_lev_addr_relay"
        )
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            # head 0: shallow slope for d=29 relay. Index from the
            # allocator so the slot stays in sync with the spec below.
            attn.alibi_slopes[relay_head_idx] = 0.2
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer9_lev_addr_relay_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    # Dim-ownership claims: L9 attn head 0 LEV addr relay.
    #   W_v[0*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[0*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   W_o[ADDR_B0_LO + k, 0*HD + 1 + k]         for k=0..15
    #   W_o[ADDR_B0_HI + k, 0*HD + 17 + k]        for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((9, "attn_W_v", f"0_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((9, "attn_W_v", f"0_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_lev_addr_relay",
        phase=9.0,
        reads={"MARK_SP", "OP_LEV", "L1H1", "BYTE_INDEX_0",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer9_lev_addr_relay_ir,
        layer_idx=9,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def make_layer9_lev_bp_to_pc_relay_op() -> Operation:
    """L9 attention head 1: BP byte 0 → ADDR_B0 at PC marker for LEV return.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_bp_to_pc_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.1 so this op runs AFTER
    ``layer9_lev_addr_relay`` (phase=9.0), matching the legacy
    in-set_vm_weights ordering. Fires in both lookup and efficient
    ALU modes — the helper performs identical setup regardless of
    alu_mode.

    Also sets ``alibi_slopes[1] = 0.5`` (BP→PC relay slope for d=15
    tokens); previously set inline alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        # Per-bake attention-head allocator with the full L9 head layout
        # pinned. Resolving the BP→PC relay head by name reproduces the
        # legacy ``head_idx=1`` literal without depending on the
        # addr-relay op having already populated ``attn._l9_head_allocator``.
        allocator = _allocate_layer9_attention_heads()
        attn._l9_head_allocator = allocator
        relay_head_idx = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer9_lev_bp_to_pc_relay"
        )
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            # head 1: BP→PC relay for LEV (d=15 tokens). Index from the
            # allocator so the slot stays in sync with the spec below.
            attn.alibi_slopes[relay_head_idx] = 0.5
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer9_lev_bp_to_pc_relay_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    # Dim-ownership claims: L9 attn head 1 LEV BP→PC relay.
    #   W_v[1*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[1*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   W_o[ADDR_B0_LO + k, 1*HD + 1 + k]         for k=0..15
    #   W_o[ADDR_B0_HI + k, 1*HD + 17 + k]        for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((9, "attn_W_v", f"1_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((9, "attn_W_v", f"1_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_lev_bp_to_pc_relay",
        phase=9.1,
        reads={"MARK_PC", "OP_LEV", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "L1H1", "BYTE_INDEX_0"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer9_lev_bp_to_pc_relay_ir,
        layer_idx=9,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def _layer9_lev_addr_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer9_lev_addr_relay_head_spec(BD))
    return ir


def _layer9_lev_bp_to_pc_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer9_lev_bp_to_pc_relay_head_spec(BD))
    return ir


def _layer9_lev_addr_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L9 head 0: previous BP byte0 -> ADDR_B0 at SP marker."""

    L = 50.0
    BP_I = 3
    GATE = 33
    v = []
    o = []
    for k in range(16):
        v.append(AP(1 + k, BD.CLEAN_EMBED_LO + k, 3.0))
        v.append(AP(17 + k, BD.CLEAN_EMBED_HI + k, 3.0))
        o.append(AO(BD.ADDR_B0_LO + k, 1 + k, 1.0))
        o.append(AO(BD.ADDR_B0_HI + k, 17 + k, 1.0))
    return DeclarativeAttentionHeadSpec(
        # Pull the head index from the shared L9 layout rather than
        # baking in a ``head_idx=0`` literal here. Both the bake and IR
        # paths consult the same source of truth, so renumbering the
        # layout in one place stays consistent across every consumer.
        head_idx=_l9_head_idx("layer9_lev_addr_relay"),
        q=(
            AP(0, BD.MARK_SP, L),
            AP(0, BD.OP_LEV, L / 5),
            AP(0, BD.CONST, -2 * L),
            AP(GATE, BD.MARK_SP, L),
            AP(GATE, BD.CONST, -L / 2),
        ),
        k=(
            AP(0, BD.L1H1 + BP_I, L),
            AP(0, BD.BYTE_INDEX_0, L),
            AP(GATE, BD.CONST, L),
        ),
        v=tuple(v),
        o=tuple(o),
    )


def _layer9_lev_bp_to_pc_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L9 head 1: previous BP byte0 -> ADDR_B0 at PC marker."""

    L = 50.0
    BP_I = 3
    GATE = 33
    v = []
    o = []
    for k in range(16):
        v.append(AP(1 + k, BD.CLEAN_EMBED_LO + k, 3.0))
        v.append(AP(17 + k, BD.CLEAN_EMBED_HI + k, 3.0))
        o.append(AO(BD.ADDR_B0_LO + k, 1 + k, 1.0))
        o.append(AO(BD.ADDR_B0_HI + k, 17 + k, 1.0))
    return DeclarativeAttentionHeadSpec(
        # Pull the head index from the shared L9 layout rather than
        # baking in a ``head_idx=1`` literal here. Both the bake and IR
        # paths consult the same source of truth, so renumbering the
        # layout in one place stays consistent across every consumer.
        head_idx=_l9_head_idx("layer9_lev_bp_to_pc_relay"),
        q=(
            AP(0, BD.MARK_PC, L),
            AP(0, BD.OP_LEV, L / 5),
            AP(0, BD.CONST, -2 * L),
            AP(GATE, BD.MARK_PC, L),
            AP(GATE, BD.CONST, -L / 2),
        ),
        k=(
            AP(0, BD.L1H1 + BP_I, L),
            AP(0, BD.BYTE_INDEX_0, L),
            AP(GATE, BD.CONST, L),
        ),
        v=tuple(v),
        o=tuple(o),
    )


def make_format_string_fetch_head_op(enable_conversational_io: bool = False) -> Operation:
    """L9 attention head 0: fetch byte from format string at FORMAT_PTR+POS.

    Originally an inline call in ``set_vm_weights`` (nested under
    ``alu_mode == 'lookup'`` + ``enable_conversational_io``):
        ``_set_format_string_fetch_head(attn9, S, BD, HD)``
        plus ``attn9.alibi_slopes.fill_(0.5)``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False. The original lookup-mode
    nesting was incidental — the helper writes only attn head 0 weights
    and has no dependency on lookup-mode-specific weights, so this op
    fires regardless of alu_mode whenever the flag is set. Phase=9.5 so
    this runs AFTER ``layer9_lev_addr_relay`` (phase=9.0) and
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1); the ``fill_(0.5)`` call
    intentionally clobbers slopes[0] and [1] that those ops set, matching
    legacy ordering (the legacy convo-io block ran ``fill_(0.5)`` AFTER
    the L9 LEV setup as well).
    """
    def bake(block, dim_positions, S):
        del S
        if not enable_conversational_io:
            return
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _format_string_fetch_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    return Operation(
        name="format_string_fetch_head",
        phase=9.5,
        reads={"IO_IN_OUTPUT_MODE", "FORMAT_PTR_LO", "FORMAT_PTR_HI",
               "ADDR_KEY", "EMBED_LO", "EMBED_HI"},
        writes={"OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _format_string_fetch_head_ir
            if enable_conversational_io else None
        ),
        declarative_authority="spec_generated",
        layer_idx=9,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _format_string_fetch_head_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_format_string_fetch_head_spec(proxy))
    return ir


def _format_string_fetch_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    L = 15.0
    q = [AP(0, BD.IO_IN_OUTPUT_MODE, L)]
    k = []
    v = []
    o = []
    for idx in range(16):
        q.append(AP(1 + idx, BD.FORMAT_PTR_LO + idx, 1.0))
        q.append(AP(17 + idx, BD.FORMAT_PTR_HI + idx, 1.0))
        k.append(AP(1 + idx, BD.ADDR_KEY + idx, L))
        k.append(AP(17 + idx, BD.ADDR_KEY + 16 + idx, L))
        v.append(AP(1 + idx, BD.EMBED_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.EMBED_HI + idx, 1.0))
        o.append(AO(BD.OUTPUT_BYTE_LO + idx, 1 + idx, 1.0))
        o.append(AO(BD.OUTPUT_BYTE_HI + idx, 17 + idx, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def make_layer9_alibi_mem_attn_op(enable: bool = False) -> Operation:
    """L9 attention head 2: ALiBi-based memory-propagation attention.

    PROOF-OF-CONCEPT for the directive "all of the memory conversions should
    be alibi attention stuff" (2026-05-11). Replaces (eventually) the
    runner-side ``_inject_mem_store`` / ``_mem_history`` shadow-memory
    pipeline with attention that gathers PSH'd values from prior step
    OUTPUT positions via ALiBi recency bias.

    Design
    ------
    Q (at MEM val byte 0 positions during LI/LC/POP with ADDR_KEY = SP-1):
      - W_q[head, BD.OP_LI_RELAY] = L
      - W_q[head, BD.OP_LC_RELAY] = L
      - W_q[head, BD.CMP+3] = L              (POP group flag at STACK0)
      - W_q[head, BD.MEM_VAL_B0] = L         (gate to val byte 0)
      - W_q[head, BD.ADDR_KEY + k] = scale * (bit_val)   (address bits)
      - W_q[head, BD.CONST] = -threshold     (suppress non-fire positions)

    K (at PSH/SI/SC OUTPUT positions — STACK0 value byte 0):
      - W_k[head, BD.PSH_AT_SP] = L          (only match PSH output positions)
      - W_k[head, BD.ADDR_KEY + k] = scale * (bit_val)
      - W_k[head, BD.MEM_STORE] = L          (only match store entries)
      - W_k[head, BD.CONST] = -threshold

    V (PSH-output STACK0's CLEAN_EMBED value, copied to OUTPUT at the
      current load position):
      - W_v[head*HD + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0  for k in 0..15
      - W_v[head*HD + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
      - W_o[BD.OUTPUT_LO + k, head*HD + 1 + k] = 1.0
      - W_o[BD.OUTPUT_HI + k, head*HD + 17 + k] = 1.0

    ALiBi slope tuning
    ------------------
    Score budget at target Q (load) attending to PSH outputs:
      - Address match (24-bit, scale=10): +300 at exact match, -300 random
      - PSH gate match: +L^2/HD (target+PSH=+L*L/HD, target+non-PSH=-)
      - ALiBi penalty: -slope * |i - j|  (one VM step = 35 tokens)

    With slope=0.5, going back 1 step costs -0.5*35 = -17.5; with two
    PSHes at the same SP, the more recent one wins by +17.5 score (>>
    softmax noise threshold). Going back 10 steps costs -175 which is
    below the +300 address-match contribution, so legitimate matches still
    fire across the typical KV-cache window. The slope should be tuned
    higher if cross-PSH leak from old values becomes a problem; lower if
    long-range matches fail.

    Status
    ------
    ``enable=False`` by default: the op IS registered (so the dep graph and
    layer_idx gates see it), but the bake is a no-op. This keeps existing
    tests byte-identical. Set ``enable=True`` to activate the head and
    flip ``alibi_slopes[2]`` from 0 to the tuned value.

    Concrete next steps for full Phase 2 PSH/POP support
    -----------------------------------------------------
    1. Bake ADDR_KEY at PSH/SI/SC OUTPUT positions (STACK0 value byte 0
       carries SP-derived address) — new FFN at L8 or earlier, ~50 LoC.
    2. Verify K-side ADDR_KEY at PSH output matches what the load-side Q
       expects. Today ADDR_KEY only lives at code byte positions
       (``_add_code_addr_keys``) and at MEM section val-byte positions
       (``_inject_mem_metadata``); we need it at PSH-step STACK0 too.
    3. Set ``enable=True`` here and ``alibi_slopes[2] = 0.5``.
    4. Drop the runner-side ``_inject_mem_section`` / ``_track_mem_access``
       calls once attention-only mode is stable.

    Time budget proof-of-concept: registers the op (passes layer_idx gate)
    and demonstrates the design pattern without disturbing existing tests.
    """
    def bake(block, dim_positions, S):
        # When disabled: op is a no-op. The head's alibi_slopes[2] stays at
        # its module-init default (a small power-of-2 value from
        # AutoregressiveAttention.__init__), but the head's W_q/W_k/W_v/W_o
        # weights are all zero — so attention output for head 2 is 0 (V is 0)
        # and W_o for head 2 dims is 0 → no contribution to the residual.
        if not enable:
            return

        from ...vm_step import _SetDim as BD_DEFAULT
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator with the full L9 head layout
        # pinned. Looking up the ALiBi mem-attn head by name keeps its
        # index identical to the legacy ``head = 2`` literal without
        # baking the integer into the call site. Stash on the attn
        # block so downstream tooling can inspect the layout.
        allocator = _allocate_layer9_attention_heads()
        attn._l9_head_allocator = allocator
        head = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer9_alibi_mem_attn"
        )
        base = head * HD

        # Slope tuned to favor most-recent matching PSH within a typical
        # 4096-token (~117-step) KV-cache window. See docstring for analysis.
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head] = 0.5

        BD = _as_setdim_proxy(dim_positions)
        L = 50.0
        scale = 10.0

        # === Q side: fire at MEM val byte 0 position during LI/LC/POP ===
        # MEM_VAL_B0 is set at val-byte-0 positions in every MEM section.
        # During load ops, that position is the natural "I am about to read
        # a memory value" anchor.
        attn.W_q[base, BD.MEM_VAL_B0] = L
        attn.W_q[base, BD.OP_LI_RELAY] = L / 5  # OP relay gates the load
        attn.W_q[base, BD.OP_LC_RELAY] = L / 5
        attn.W_q[base, BD.CMP + 3] = L / 5      # POP group
        attn.W_q[base, BD.CONST] = -L * 1.5    # threshold

        # === K side: match PSH-output STACK0 positions ===
        # PSH_AT_SP is set at SP/STACK0 positions during PSH steps.
        # MEM_STORE is set at MEM val-bytes (and at PSH-step's STACK0
        # output, via L6 head 6 / L7 head 7 broadcast in step W).
        attn.W_k[base, BD.PSH_AT_SP] = L
        attn.W_k[base, BD.MEM_STORE] = L / 2
        attn.W_k[base, BD.CONST] = -L * 0.5

        # === Address matching: 24 binary bits across 3 address bytes ===
        # Both sides use the same ADDR_KEY encoding. Address match gives
        # +300 to score (per-dim contribution after /sqrt(HD)). Mismatch
        # gives ~0 (random) to -300 (anti-match).
        addr_dim = 4
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in [ab_lo, ab_hi]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

        # === V/O: copy CLEAN_EMBED bytes → OUTPUT at load position ===
        # The PSH step's STACK0 has CLEAN_EMBED_LO/HI = AX value at time of
        # PSH. Carrying it to OUTPUT at the load position writes the value
        # into the load result.
        scale_v = 1.0
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale_v
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale_v
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0

    # Dim-ownership claims: L9 attn head 2 ALiBi mem attn.
    # When enable=True, V slots 1..32 carry CLEAN_EMBED to OUTPUT.
    _claims = set()
    if enable:
        for k in range(16):
            _claims.add((9, "attn_W_v", f"2_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((9, "attn_W_v", f"2_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_alibi_mem_attn",
        phase=9.2,  # after lev_addr_relay (9.0) and lev_bp_to_pc_relay (9.1)
        reads={"MEM_VAL_B0", "OP_LI_RELAY", "OP_LC_RELAY", "CMP", "CONST",
               "PSH_AT_SP", "MEM_STORE", "ADDR_KEY",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake if not enable else None,
        layer_idx=9,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#the-attention-layer",
    )


def make_layer9_marker_suppress_op() -> Operation:
    """No-op dep anchor for marker suppression owned by ``layer9_alu``.

    ``make_layer9_alu_op`` owns the true migrated bake: it calls
    ``_set_layer9_alu`` and then threads the returned unit cursor into
    ``_set_layer9_marker_suppress``. This standalone op remains only to
    preserve dependency topology for downstream audits; baking the helper here
    would either overlap L9 ALU units or double-write marker-suppression
    weights.
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is chained from `layer9_alu`.
        return

    return Operation(
        name="layer9_marker_suppress",
        phase=9,
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_OR", "OP_XOR", "OP_AND"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
