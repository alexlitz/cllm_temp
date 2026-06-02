"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L13 attention-head layout (pinned indices) =====================
#
# ``layer13_mem_addr_gather`` owns heads 0, 1, 2 on the L13 attention
# block. Each head gathers one MEM addr byte's CLEAN_EMBED nibbles into
# ADDR_B{0,1,2}_LO/HI at the MEM val byte positions:
#
#   head 0 -> ADDR_B0_LO/HI, plus the ADDR_B0_VALID lifecycle bit
#             (slot 34, B7-4).
#   head 1 -> ADDR_B1_LO/HI, plus the ADDR_B1_VALID lifecycle bit
#             (slot 34, B8-A; routes into H5+4 / position 99).
#   head 2 -> ADDR_B2_LO/HI, plus the ADDR_B2_VALID lifecycle bit
#             (slot 34, B8-A; routes into H5+5 / position 100).
#
# Pre-migration the call sites used bare ``base = j * HD`` literals
# inside ``_set_layer13_mem_addr_gather`` and ``_l13_addr_bn_valid_extension``.
# Pinning the allocator preserves those exact slots so the lowering is
# byte-identical, while the layout table becomes the audited source of
# truth for future L13 attention extensions.
_L13_HEAD_LAYOUT = (
    # (op-name key,                            pinned head_idx)
    ("layer13_mem_addr_gather.head_0",         0),  # ADDR_B0 + ADDR_B0_VALID
    ("layer13_mem_addr_gather.head_1",         1),  # ADDR_B1 + ADDR_B1_VALID
    ("layer13_mem_addr_gather.head_2",         2),  # ADDR_B2 + ADDR_B2_VALID
)


def _allocate_layer13_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with the L13 heads pinned.

    Every entry in :data:`_L13_HEAD_LAYOUT` is pinned at its existing
    ``head_idx`` so the underlying weight writes -- now expressed as
    ``DeclarativeAttentionHeadSpec`` instances -- land byte-identically.
    A future L13 attention op can claim a free head past index 2 via
    ``allocator.alloc(name, layer_idx=13)`` (no ``pin=``) without
    touching this table.
    """
    allocator = AttentionHeadAllocator()
    for name, head_idx in _L13_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=13, pin=head_idx)
    return allocator


def _l13_head_idx(op_name: str) -> int:
    """Return the pinned L13 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L13_HEAD_LAYOUT` for callers (e.g.
    ``compiler_ir_factory`` helpers) that cannot instantiate a per-bake
    allocator. Mirrors the L4 ``_l4_head_idx`` pattern.
    """
    for name, head_idx in _L13_HEAD_LAYOUT:
        if name == op_name:
            return head_idx
    raise KeyError(f"_l13_head_idx: unknown L13 attention op {op_name!r}")


# === L13 FFN unit layout (pinned offsets) ============================
#
# The ``layer13_shifts`` op owns the entire L13 FFN. The actual weight
# writes happen inside ``vm_step._set_layer13_shifts``, which uses a
# local ``unit = 0`` counter that increments through two 2048-unit
# sub-stages (SHL then SHR, each spanning 8 shift amounts x 16 a_hi
# x 16 a_lo). Migration to :class:`FFNUnitAllocator` keeps that helper
# byte-identical -- we just declare each sub-stage's range at its
# existing pinned offset so the layout is auditable rather than
# implicit. Adding a new L13 op family later will go through
# ``allocator.alloc(name, n)`` without a pin, but note the SHL+SHR
# chain currently fills the 4096-unit pool exactly (see the trailing
# "fills L13 exactly" note in the helper docstring), so a future
# extension must first shrink one of the lookup tables or widen the
# pool.
#
# The offsets below mirror the unit-counter walk in
# ``vm_step._set_layer13_shifts``. Changing the helper's unit count
# requires updating this table in lock-step.
_L13_SHIFTS_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    ("layer13_shifts.shl",    0, 2048),  # OP_SHL: 8 shifts x 16 a_hi x 16 a_lo
    ("layer13_shifts.shr", 2048, 2048),  # OP_SHR: 8 shifts x 16 a_hi x 16 a_lo
)


def _allocate_layer13_shifts_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L13 shift sub-stages.

    Every sub-stage is pinned at its existing offset so the underlying
    ``vm_step._set_layer13_shifts`` helper -- which writes via its own
    monotonic ``unit = 0`` counter -- lands on exactly the same
    hidden-unit indices it always has. This call is byte-identical
    bookkeeping: the allocator declares ranges by name, the helper
    writes the weights. A future refactor can split the monolithic
    helper into per-range bake functions that consume
    ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L13 op claims a free range -- but see the layout-table note
    above: the SHL+SHR chain currently saturates the 4096-unit pool).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L13_SHIFTS_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


# === L13 SHL/SHR FFN rules (declarative) ============================
#
# Each shift sub-stage is a dense 8 x 16 x 16 = 2048-unit lookup table
# implementing the 8-bit nibble-level result for shift amounts 0..7.
# Each unit is a 5-way AND in the silu path:
#
#   conditions = MARK_AX + ALU_LO[a_lo] + ALU_HI[a_hi]
#              + AX_CARRY_LO[s] + AX_CARRY_HI[0]
#   threshold  = 4.5  (so b_up = -S * 4.5)
#   gate       = OP_SHL or OP_SHR (gate_weight = 1.0, gate_bias = 0.0)
#   writes     = OUTPUT_LO[result_lo] = 2.0/S, OUTPUT_HI[result_hi] = 2.0/S
#
# Walk order matches the imperative ``_set_layer13_shifts`` outer loop:
# ``for s in range(8): for a_hi in range(16): for a_lo in range(16):``
# so that with start_unit=0 the SHL substage lands on units 0..2047 and
# the SHR substage lands on units 2048..4095 -- the same indices the
# legacy helper writes. Byte-identity is gated by
# :func:`compare_symbolic_to_lowered_ffn` over the full 4096-rule IR.


def _layer13_shifts_substage_rules(
    op_dim_name: str,
    shift_fn,
    *,
    name_prefix: str,
    S: float,
) -> tuple[FFNRule, ...]:
    """Build the 2048-rule list for one L13 shift sub-stage.

    ``op_dim_name`` is the gating opcode dim name (``"OP_SHL"`` or
    ``"OP_SHR"``). ``shift_fn(value, s)`` computes the 8-bit shift result
    for ``(value, shift_amount)``; it must agree with the imperative
    helper's per-shift table.

    The 2048 rules iterate ``(s, a_hi, a_lo)`` in nested order to match
    the legacy unit counter in ``setup_helpers._set_layer13_shifts``;
    appending the SHL list followed by the SHR list lands on the
    pinned offsets declared in :data:`_L13_SHIFTS_UNIT_LAYOUT`.
    """
    write_scale = 2.0 / S
    rules: list[FFNRule] = []
    for s in range(8):
        for a_hi in range(16):
            for a_lo in range(16):
                value = (a_hi << 4) | a_lo
                result = shift_fn(value, s)
                result_lo = result & 0xF
                result_hi = (result >> 4) & 0xF
                rules.append(FFNRule.gated_write(
                    name=(
                        f"{name_prefix}_s{s}_ahi{a_hi}_alo{a_lo}"
                    ),
                    conditions=(
                        ("MARK_AX", 1.0),
                        (f"ALU_LO+{a_lo}", 1.0),
                        (f"ALU_HI+{a_hi}", 1.0),
                        (f"AX_CARRY_LO+{s}", 1.0),
                        ("AX_CARRY_HI+0", 1.0),
                    ),
                    threshold=4.5,
                    gate=op_dim_name,
                    gate_weight=1.0,
                    gate_bias=0.0,
                    writes=(
                        (f"OUTPUT_LO+{result_lo}", write_scale),
                        (f"OUTPUT_HI+{result_hi}", write_scale),
                    ),
                    scope=f"MARK_AX and {op_dim_name}",
                ))
    return tuple(rules)


def _layer13_shl_rules(S: float) -> tuple[FFNRule, ...]:
    """L13 SHL sub-stage: 2048 lookup-table units (units 0..2047)."""
    return _layer13_shifts_substage_rules(
        "OP_SHL",
        lambda v, s: (v << s) & 0xFF,
        name_prefix="l13_shl",
        S=S,
    )


def _layer13_shr_rules(S: float) -> tuple[FFNRule, ...]:
    """L13 SHR sub-stage: 2048 lookup-table units (units 2048..4095)."""
    return _layer13_shifts_substage_rules(
        "OP_SHR",
        lambda v, s: (v >> s) & 0xFF,
        name_prefix="l13_shr",
        S=S,
    )


def _layer13_shifts_rules(S: float) -> tuple[FFNRule, ...]:
    """Combined SHL + SHR rule list (4096 units total).

    Order matches the imperative ``_set_layer13_shifts`` outer
    ``[(OP_SHL, ...), (OP_SHR, ...)]`` loop so that lowering with
    ``start_unit=0`` writes the SHL block at units 0..2047 and the SHR
    block at units 2048..4095, byte-identical to the legacy helper.
    """
    return _layer13_shl_rules(S) + _layer13_shr_rules(S)


def _layer13_shifts_ir(S: float = 100.0) -> CompilerIR:
    """L13 FFN: declarative SHL+SHR lookup tables (4096 rules)."""
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer13_shifts_rules(S))
    return ir


def _bake_layer13_shifts(ffn, S, BD) -> int:
    """Lower the declarative L13 shifts IR into ``ffn``.

    Returns the next-free unit cursor (4096). Byte-identical to
    ``setup_helpers._set_layer13_shifts`` (verified via
    :func:`compare_symbolic_to_lowered_ffn` and per-substage parity
    tests against the legacy helper).
    """
    rules = _layer13_shifts_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


def _l13_addr_bn_valid_positions(BD):
    """Resolve the ADDR_B{1,2}_VALID destination dim positions.

    Position 99 (``ADDR_B1_VALID``) aliases ``H5+4``; position 100
    (``ADDR_B2_VALID``) aliases ``H5+5``. Use ``dim_positions`` when the
    compiler exposed the new names; otherwise derive from ``H5`` (the
    dormant L0 head-5 threshold output the B7/B8 dims alias onto). The
    numeric fallback (99 / 100) matches the dim_registry allocation so the
    producer fires even before the compiler learns the new names.
    """
    h5_base = getattr(BD, "H5", None)
    addr_b1_valid_pos = getattr(BD, "ADDR_B1_VALID", None)
    if addr_b1_valid_pos is None:
        addr_b1_valid_pos = (h5_base + 4) if h5_base is not None else 99
    addr_b2_valid_pos = getattr(BD, "ADDR_B2_VALID", None)
    if addr_b2_valid_pos is None:
        addr_b2_valid_pos = (h5_base + 5) if h5_base is not None else 100
    return addr_b1_valid_pos, addr_b2_valid_pos


def _layer13_mem_addr_gather_head_specs(BD) -> tuple:
    """Declarative L13 heads 0-2: MEM addr-byte gather + B7-4/B8-A VALID bits.

    Mirrors ``setup_helpers._set_layer13_mem_addr_gather`` (heads 0-2 slot 0
    + slot 33 anti-leak, slot 1..32 V/O CLEAN_EMBED -> ADDR_BJ_LO/HI, plus
    head 0 slot 34 ADDR_B0_VALID lifecycle) AND
    ``_l13_addr_bn_valid_extension`` (heads 1/2 slot 34 ADDR_B{1,2}_VALID
    extension). Per-head K rows pick the MEM addr-byte ``j`` row:

      - head 0 (ADDR_B0): K[+L1H1+MEM_I, -L1H0+MEM_I] -- addr byte 0 at d=1.
      - head 1 (ADDR_B1): K[+L1H2+MEM_I, -L1H1+MEM_I] -- addr byte 1 at d=2.
      - head 2 (ADDR_B2): K[+H0+MEM_I,   -L1H2+MEM_I] -- addr byte 2 at d=3.

    The VALID lifecycle V reads the K+ threshold dim so the attended
    addr-byte-J row delivers V=1.0 (and unrelated rows V=0) -- same logic
    as the head-0 slot-34 ADDR_B0_VALID producer (B7-4) extended to
    ADDR_B1_VALID / ADDR_B2_VALID (B8-A).
    """
    L = 15.0
    MEM_I = 4
    VALID_SLOT = 34
    addr_b1_valid_pos, addr_b2_valid_pos = _l13_addr_bn_valid_positions(BD)

    # Per-head wiring tables: layout for the J-th MEM addr byte gather head.
    # (head_idx, addr_lo_out, addr_hi_out, K_pos_dim, K_neg_dim,
    #  valid_v_read_dim, valid_o_dest)
    head_layout = (
        (
            _l13_head_idx("layer13_mem_addr_gather.head_0"),
            BD.ADDR_B0_LO, BD.ADDR_B0_HI,
            BD.L1H1 + MEM_I, BD.L1H0 + MEM_I,
            BD.L1H1 + MEM_I, BD.ADDR_B0_VALID,
        ),
        (
            _l13_head_idx("layer13_mem_addr_gather.head_1"),
            BD.ADDR_B1_LO, BD.ADDR_B1_HI,
            BD.L1H2 + MEM_I, BD.L1H1 + MEM_I,
            BD.L1H2 + MEM_I, addr_b1_valid_pos,
        ),
        (
            _l13_head_idx("layer13_mem_addr_gather.head_2"),
            BD.ADDR_B2_LO, BD.ADDR_B2_HI,
            BD.H0 + MEM_I, BD.L1H2 + MEM_I,
            BD.H0 + MEM_I, addr_b2_valid_pos,
        ),
    )

    specs = []
    for (head_idx, addr_lo_out, addr_hi_out,
         k_pos, k_neg, valid_v_read, valid_o_dest) in head_layout:
        # Q slot 0: fires at MEM val byte positions (d=5..8 from MEM)
        q = [
            AP(0, BD.MEM_VAL_B0, L),
            AP(0, BD.MEM_VAL_B1, L),
            AP(0, BD.MEM_VAL_B2, L),
            AP(0, BD.MEM_VAL_B3, L),
            # Slot 33 anti-leakage gate
            AP(33, BD.MEM_VAL_B0, L),
            AP(33, BD.CONST, -L / 2),
            # Slot 34 VALID lifecycle Q mirrors slot 0 (fires at MEM val bytes)
            AP(VALID_SLOT, BD.MEM_VAL_B0, L),
            AP(VALID_SLOT, BD.MEM_VAL_B1, L),
            AP(VALID_SLOT, BD.MEM_VAL_B2, L),
            AP(VALID_SLOT, BD.MEM_VAL_B3, L),
        ]
        # K slot 0: fires at MEM addr byte J position; slot 33 const anti-leak;
        # slot 34 mirrors slot 0 (picks the MEM addr byte J row).
        k = [
            AP(0, k_pos, L),
            AP(0, k_neg, -L),
            AP(33, BD.CONST, L),
            AP(VALID_SLOT, k_pos, L),
            AP(VALID_SLOT, k_neg, -L),
        ]
        # V slots 1..32 copy CLEAN_EMBED nibbles (addr byte value).
        # V slot 34 reads valid_v_read so the attended addr-byte-J row
        # delivers V=1.0 and unrelated rows deliver 0.
        v = [AP(1 + kk, BD.CLEAN_EMBED_LO + kk, 1.0) for kk in range(16)]
        v += [AP(17 + kk, BD.CLEAN_EMBED_HI + kk, 1.0) for kk in range(16)]
        v.append(AP(VALID_SLOT, valid_v_read, 1.0))
        # O slots 1..32 route gathered nibbles to ADDR_BJ_LO/HI; slot 34
        # routes the VALID lifecycle bit into ADDR_BJ_VALID.
        o = [AO(addr_lo_out + kk, 1 + kk, 1.0) for kk in range(16)]
        o += [AO(addr_hi_out + kk, 17 + kk, 1.0) for kk in range(16)]
        o.append(AO(valid_o_dest, VALID_SLOT, 1.0))

        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ))
    return tuple(specs)


def _layer13_mem_addr_gather_ir(dim_positions, HD) -> CompilerIR:
    """Build the declarative L13 mem-addr-gather IR for the compiler.

    Three heads (0/1/2), each gather one MEM addr byte's CLEAN_EMBED
    nibbles into ADDR_BJ_LO/HI at MEM val byte positions, plus a
    slot-34 VALID lifecycle bit per head (ADDR_B0_VALID/B1_VALID/B2_VALID).
    See :func:`_layer13_mem_addr_gather_head_specs` for the byte-level layout.
    """
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer13_mem_addr_gather_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    return ir


def make_layer13_mem_addr_gather_op() -> Operation:
    """L13 attention: gather MEM addr from STACK0 / AX_CARRY for SI/SC/LI/LC.

    Pinned to ``layer_idx=13`` via ``kind="block"``: dep-graph assignment
    otherwise lands at L15 (mismatch with legacy block 13).

    Wave 3I: fully migrated to ``DeclarativeAttentionHeadSpec`` form. The
    legacy ``_set_layer13_mem_addr_gather`` and ``_l13_addr_bn_valid_extension``
    imperative helpers are replaced by
    :func:`_layer13_mem_addr_gather_head_specs` (three head specs covering
    the B7-4 ADDR_B0_VALID producer and the B8-A ADDR_B{1,2}_VALID
    extension), exposed via ``compiler_ir_factory``. Byte-identity gated by
    ``compare_symbolic_to_lowered_attn``.
    """
    def bake(block, dim_positions, S):
        del S
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        # Per-bake attention-head allocator with the L13 head layout pinned.
        # Stashed on ``attn`` for downstream inspection / extension; the
        # actual ``head_idx`` values used by ``_layer13_mem_addr_gather_head_specs``
        # come from :func:`_l13_head_idx` so the spec stays in lockstep
        # with the layout table without re-querying the allocator here.
        allocator = _allocate_layer13_attention_heads()
        attn._l13_head_allocator = allocator
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn,
            _layer13_mem_addr_gather_head_specs(proxy),
            HD,
        )

    # Dim-ownership claims: L13 attn heads 0-2 mem addr gather. Each head
    # writes V slots 1..32 reading CLEAN_EMBED_LO/HI:
    #   W_v[h*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[h*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    # Head 0 additionally writes slot 34's V row from L1H1+MEM_I and routes
    # through W_o into ADDR_B0_VALID (B7-4 lifecycle bit; see setup_helpers
    # ``_set_layer13_mem_addr_gather`` docstring).
    #
    # B8-A note: heads 1/2 slot 34 also produce ADDR_B{1,2}_VALID at slots
    # 99/100 via ``_l13_addr_bn_valid_extension``. Those writes are NOT yet
    # in ``claims`` / ``writes`` -- the per-op contract test currently pins
    # the head-0-only invariant. A follow-up commit will (a) add the two
    # dims to ``declare_setdim_compat_dims``, (b) add the V claims and
    # writes entries, and (c) update the per-op test to expect three VALID
    # bits. Until then the verifier records the head-1/2 V cells under
    # ``written_but_not_declared`` (warning, not error in non-strict mode).
    _claims = set()
    for h in range(3):
        for k in range(16):
            _claims.add((13, "attn_W_v", f"{h}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((13, "attn_W_v", f"{h}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    # ADDR_B0_VALID lifecycle slot: head 0, slot 34, V row reads L1H1+MEM_I=4.
    # Verifier decodes the col position via ``_pos_to_column`` which produces
    # ``"<DIM>+<offset>"``; L1H1 has 7 lanes so MEM_I=4 lands at +4.
    _claims.add((13, "attn_W_v", "0_34", "L1H1+4"))

    return Operation(
        name="layer13_mem_addr_gather",
        phase=13,
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "AX_CARRY_LO", "AX_CARRY_HI", "OP_LI", "OP_LC", "OP_SI", "OP_SC",
               "MEM_ADDR_SRC", "L1H1"},
        writes={"ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI",
                "ADDR_B0_VALID"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer13_mem_addr_gather_ir,
        declarative_authority="spec_generated",
        layer_idx=13,
        migrated=True,
        claims=_claims,
        smoke_tests={
            "TestSmokeMemory::test_sc_lc_roundtrip",
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def make_layer13_shifts_op(alu_mode: str = "lookup") -> Operation:
    """L13 FFN: SHL/SHR shifts (lookup-mode entry point).

    Pinned to ``layer_idx=13`` via ``kind="block"``. See
    ``make_layer13_mem_addr_gather_op``.

    In ``alu_mode='lookup'`` we bake the standard SHL/SHR lookup table as a
    4096-rule declarative :class:`FFNRule` IR (Phase 6 Wave 4B migration);
    byte-identical to the legacy ``setup_helpers._set_layer13_shifts`` helper
    (gated by ``test_declarative_ffn_bakes_l13``).

    Declarations-only note: lookup mode is exposed through the migrated owner
    so strict builds do not fall back to legacy model bake. Efficient mode is
    represented by the structural 4-stage composite ops instead.

    In ``alu_mode='efficient'`` SHL/SHR are now handled by the 4-stage
    composite installed via the dedicated
    ``make_l13_alu_shift_{bdtoge,precompute,select,getobd}_op`` factories
    (each at phase=13 so they share L13 with ``make_layer13_mem_addr_gather_op``).
    The 4 ops together replace the runtime ``ALUShift`` wrapper that used to
    be attached by ``set_vm_weights``. This entry-point is a no-op in
    efficient mode so the lookup-table bake doesn't overwrite the composite's
    output.
    """
    if alu_mode not in ("lookup", "efficient"):
        raise ValueError(
            f"alu_mode must be 'lookup' or 'efficient'; got {alu_mode!r}"
        )

    if alu_mode == "efficient":
        def bake(block, dim_positions, S):
            return  # ALUShiftComposite (4-stage) owns SHL/SHR in efficient mode.

        compiler_ir = None
    else:
        def bake(block, dim_positions, S):
            proxy = _as_setdim_proxy(dim_positions)

            # Per-bake FFN-unit allocator. Each L13 shift sub-stage is
            # pinned to its existing offset so the call below lands
            # byte-identically. Stashed on the FFN module so downstream
            # tools (e.g. a future L13 op family) can inspect or extend
            # the layout. Mirrors the L1 / L9 convention from
            # 4639146 / ca775eb.
            allocator = _allocate_layer13_shifts_units()
            block.ffn._l13_unit_allocator = allocator

            n_units = _bake_layer13_shifts(block.ffn, S, proxy)
            # Byte-identity guard: the declarative lowering MUST land
            # exactly on the allocator's declared range total or a
            # downstream layer will read stale weights.
            expected_total = sum(n for _, _, n in _L13_SHIFTS_UNIT_LAYOUT)
            assert n_units == expected_total, (
                f"L13 layer13_shifts unit cursor drift: declarative "
                f"bake wrote {n_units} units, allocator declared "
                f"{expected_total}"
            )

        compiler_ir = _layer13_shifts_ir()

    return Operation(
        name="layer13_shifts",
        phase=13,
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_SHL", "OP_SHR"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=compiler_ir,
        declarative_authority="spec_generated",
        layer_idx=13,
        migrated=True,
        # Staleness invariants: L13 shift FFN consumes ALU_LO/HI (value to
        # shift) and AX_CARRY_LO (shift amount) at the AX marker for
        # OP_SHL / OP_SHR. Only meaningful when alu_mode='lookup' fires the
        # bake; in efficient mode the composite owns the consumes-fresh
        # chain via its own stages.
        consumes_fresh={
            "ALU_LO": "AX_byte0",
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
        } if alu_mode == "lookup" else {},
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
    )
