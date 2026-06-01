"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...ffn_unit_allocator import FFNUnitAllocator
from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


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


def _l13_addr_bn_valid_extension(attn, BD, HD, S):
    """B8-A: mirror the head-0 ADDR_B0_VALID slot-34 producer on heads 1 and 2.

    The B7-4 producer in ``_set_layer13_mem_addr_gather`` wires a single
    lifecycle bit on head 0 slot 34: Q fires at MEM val byte positions, K
    selects the MEM addr byte 0 row (L1H1+MEM_I=+L, L1H0+MEM_I=-L), V reads
    ``L1H1+MEM_I``, and W_o routes the gathered constant into
    ``ADDR_B0_VALID`` (slot 97).

    B8-A allocates two more lifecycle bits at slots 99 (``ADDR_B1_VALID``)
    and 100 (``ADDR_B2_VALID``). Heads 1 and 2 already gather the ADDR_B1 /
    ADDR_B2 nibbles via slots 1..32; this helper adds the matching slot-34
    VALID bit per head so downstream rules can gate on B1/B2 freshness with
    a dedicated witness instead of borrowing ADDR_B0_VALID.

    Per-head K rows mirror the primary-gather K:
      - head 1: K[+L1H2+MEM_I, -L1H1+MEM_I] -- picks MEM addr byte 1 (d=2).
      - head 2: K[+H0+MEM_I,   -L1H2+MEM_I] -- picks MEM addr byte 2 (d=3).

    V reads the same threshold dim that K's positive arm fires on (so the
    attended row delivers V=1 and unrelated rows V=0; same lifecycle logic
    as head 0 slot 34 with L1H1+MEM_I).

    Output W_o routes to position 99 (ADDR_B1_VALID) for head 1 and 100
    (ADDR_B2_VALID) for head 2. We resolve the destination through
    ``dim_positions`` when the compiler declared the slot, and fall back to
    the dim_registry-allocated aliases (H5+4 / H5+5 = 99 / 100) otherwise so
    the producer fires under both the legacy ``pin_to_setdim`` layout and
    pre-declaration layouts.

    The matrix writes are inlined here (not in ``setup_helpers``) so this
    B8-A producer ships in a single op-factory file. ``writes`` and
    ``claims`` are deliberately left untouched -- those declarations
    formalize ownership and are scheduled for a follow-up commit that also
    adds the dim names to ``declare_setdim_compat_dims`` and refreshes the
    per-op contract tests. The verifier reports the new cells under
    ``written_but_not_declared`` (non-strict OK) until that lands.
    """
    L = 15.0
    MEM_I = 4
    VALID_SLOT = 34
    # Position 99 (ADDR_B1_VALID) aliases H5+4; position 100 aliases H5+5.
    # Use dim_positions when the compiler exposed the new names; otherwise
    # derive from H5 (the dormant L0 head-5 threshold output the B7/B8 dims
    # all alias onto). The numeric fallback (99 / 100) matches the
    # dim_registry allocation so the producer fires even before the compiler
    # learns the new names.
    h5_base = getattr(BD, "H5", None)
    addr_b1_valid_pos = getattr(BD, "ADDR_B1_VALID", None)
    if addr_b1_valid_pos is None:
        addr_b1_valid_pos = (h5_base + 4) if h5_base is not None else 99
    addr_b2_valid_pos = getattr(BD, "ADDR_B2_VALID", None)
    if addr_b2_valid_pos is None:
        addr_b2_valid_pos = (h5_base + 5) if h5_base is not None else 100

    # Per-head wiring tables: (base, K_pos_dim, K_neg_dim, V_read_dim, O_dest)
    head_specs = (
        (1, BD.L1H2 + MEM_I, BD.L1H1 + MEM_I, BD.L1H2 + MEM_I, addr_b1_valid_pos),
        (2, BD.H0   + MEM_I, BD.L1H2 + MEM_I, BD.H0   + MEM_I, addr_b2_valid_pos),
    )

    for h, k_pos, k_neg, v_read, o_dest in head_specs:
        base = h * HD
        # Q mirrors the primary slot 0 -- fires at every MEM val byte position.
        attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B0] = L
        attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B1] = L
        attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B2] = L
        attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B3] = L
        # K mirrors the primary slot 0 -- picks the MEM addr byte j row.
        attn.W_k[base + VALID_SLOT, k_pos] = L
        attn.W_k[base + VALID_SLOT, k_neg] = -L
        # V reads the K+ dim so the attended addr-byte-j row delivers 1.0.
        attn.W_v[base + VALID_SLOT, v_read] = 1.0
        # O routes the gathered constant into ADDR_B{1,2}_VALID.
        attn.W_o[o_dest, base + VALID_SLOT] = 1.0


def make_layer13_mem_addr_gather_op() -> Operation:
    """L13 attention: gather MEM addr from STACK0 / AX_CARRY for SI/SC/LI/LC.

    Pinned to ``layer_idx=13`` via ``kind="block"``: dep-graph assignment
    otherwise lands at L15 (mismatch with legacy block 13).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer13_mem_addr_gather
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        BD = _as_setdim_proxy(dim_positions)
        _set_layer13_mem_addr_gather(attn, S, BD, HD)
        # B8-A: extend the producer with ADDR_B1_VALID / ADDR_B2_VALID slot-34
        # bakes on heads 1 and 2. Inlined here (not in setup_helpers) so the
        # follow-up cleanup -- which will also need to update the L13 per-op
        # test contract -- can move them in one go.
        _l13_addr_bn_valid_extension(attn, BD, HD, S)

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

    In ``alu_mode='lookup'`` we bake the standard SHL/SHR lookup table via
    ``_set_layer13_shifts`` into the L13 PureFFN block.

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
    else:
        def bake(block, dim_positions, S):
            from ...vm_step import _set_layer13_shifts
            proxy = _as_setdim_proxy(dim_positions)

            # Per-bake FFN-unit allocator. Each L13 shift sub-stage is
            # pinned to its existing offset so the call below lands
            # byte-identically. Stashed on the FFN module so downstream
            # tools (e.g. a future L13 op family) can inspect or extend
            # the layout. Mirrors the L1 / L9 convention from
            # 4639146 / ca775eb.
            allocator = _allocate_layer13_shifts_units()
            block.ffn._l13_unit_allocator = allocator

            _set_layer13_shifts(block.ffn, S, proxy)

    return Operation(
        name="layer13_shifts",
        phase=13,
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_SHL", "OP_SHR"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
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
