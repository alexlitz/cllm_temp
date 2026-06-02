"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L14 attention-head layout (pinned indices) =====================
#
# ``layer14_mem_generation`` owns heads 0-7 of the L14 attention block:
#
#   heads 0-3: MEM address-byte generation (PSH, SI/SC, JSR/ENT).
#     head 0 -> MEM addr byte 0 (predicted at MEM marker, d=0).
#     head 1 -> MEM addr byte 1 (predicted at addr_b0 token, d=1).
#     head 2 -> MEM addr byte 2 (predicted at addr_b1 token, d=2).
#     head 3 -> MEM addr byte 3 (predicted at addr_b2 token, d=3).
#   heads 4-7: MEM value-byte generation (PSH/SI/SC source AX,
#              JSR/ENT source STACK0).
#     head 4 -> MEM val byte 0 (predicted at addr_b3 token, d=4).
#     head 5 -> MEM val byte 1 (predicted at val_b0 token, d=5).
#     head 6 -> MEM val byte 2 (predicted at val_b1 token, d=6).
#     head 7 -> MEM val byte 3 (predicted at val_b2 token, d=7).
#
# Pre-migration the legacy ``_set_layer14_mem_generation`` helper used
# bare ``base = h * HD`` literals for h in 0..7. Pinning the allocator
# preserves those exact slots so the lowering is byte-identical, while
# the layout table becomes the audited source of truth for future L14
# attention extensions.
_L14_HEAD_LAYOUT = (
    # (op-name key,                            pinned head_idx)
    ("layer14_mem_generation.head_0",          0),  # MEM addr byte 0
    ("layer14_mem_generation.head_1",          1),  # MEM addr byte 1
    ("layer14_mem_generation.head_2",          2),  # MEM addr byte 2
    ("layer14_mem_generation.head_3",          3),  # MEM addr byte 3
    ("layer14_mem_generation.head_4",          4),  # MEM val byte 0
    ("layer14_mem_generation.head_5",          5),  # MEM val byte 1
    ("layer14_mem_generation.head_6",          6),  # MEM val byte 2
    ("layer14_mem_generation.head_7",          7),  # MEM val byte 3
)
_L14_HEAD_LAYOUT_BY_NAME = {n: h for n, h in _L14_HEAD_LAYOUT}


def _allocate_layer14_mem_generation_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with the L14 heads pinned.

    Every entry in :data:`_L14_HEAD_LAYOUT` is pinned at its existing
    ``head_idx`` so the underlying weight writes -- now expressed as
    ``DeclarativeAttentionHeadSpec`` instances -- land byte-identically.
    The legacy ``_set_layer14_mem_generation`` helper iterated h=0..7
    with ``base = h * HD``; those same indices are pinned here.
    """
    allocator = AttentionHeadAllocator()
    for name, head_idx in _L14_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=14, pin=head_idx)
    return allocator


# === L14 FFN cleanup-chain unit layout (pinned offsets) =================
#
# The 8 L14 cleanup ops historically chained their hidden-unit
# allocations through a shared monotonic counter on
# ``block.ffn._l14_unit_counter`` (first op starts at 0; each op reads the
# counter, calls its helper, and writes the returned ``next_unit`` back).
# Migration to :class:`FFNUnitAllocator` keeps the chain byte-identical by
# pinning every op to its existing chain offset -- the offsets below are
# the exact values the counter held when each op ran in production order.
#
# Each op's bake instantiates a fresh allocator with a single pinned
# claim; the allocator is bookkeeping rather than persistent state. A
# future L14 op family can claim a free gap via ``allocator.alloc(name,
# n)`` (no pin) instead of hand-picking another offset, after stitching
# the table here.
#
# Adding or resizing any helper requires updating this table in lock-step
# (mirrors the L9 ``_L9_ALU_UNIT_LAYOUT`` convention).
_L14_CLEANUP_CHAIN_LAYOUT = {
    "layer14_temp_clear":                    (   0,    4),
    "layer14_clear_addr_key_pollution":      (   4,   48),
    "layer14_clear_output_corruption":       (  52,   18),
    "layer14_clear_mem_marker_output":       (  70,   64),
    "layer14_addr_key_neural_decode":        ( 134, 1728),
    "layer14_jsr_ax_bytes_zero":             (1862,    4),
    "layer14_lc_ax_bytes_zero":              (1866,    4),
    "layer14_alu_nocarry_ax_bytes_zero":     (1870,    4),
}


def _l14_chain_alloc(op_name: str) -> int:
    """Return the pinned start unit for ``op_name`` in the L14 cleanup chain.

    Looks up ``op_name`` in :data:`_L14_CLEANUP_CHAIN_LAYOUT` and routes
    the pinned range through a fresh :class:`FFNUnitAllocator`. The
    allocator is per-bake (not shared across ops) so each call is
    independent; the byte-identity guarantee comes from the static pin
    matching the legacy ``_l14_unit_counter`` value the predecessor op
    left behind in production order. Adding the allocator call now keeps
    the structure auditable and ready for future ``pin=None`` extensions
    without changing any baked weight.
    """

    pin, n_units = _L14_CLEANUP_CHAIN_LAYOUT[op_name]
    allocator = FFNUnitAllocator()
    start, _end = allocator.alloc(op_name, n_units, pin=pin)
    return start


def _resolve_dim(dim_positions, name: str):
    if isinstance(dim_positions, dict) and name in dim_positions:
        return dim_positions[name]
    proxy = _as_setdim_proxy(dim_positions if isinstance(dim_positions, dict) else {})
    return getattr(proxy, name, None)


def _guard_l14_output_units_on_step_boundary(
    ffn,
    dim_positions,
    S: float,
    start_unit: int,
    end_unit: int,
) -> None:
    """Stop L14 OUTPUT cleanup units from firing between VM rows."""

    if end_unit <= start_unit:
        return
    const_dim = _resolve_dim(dim_positions, "CONST")
    output_lo = _resolve_dim(dim_positions, "OUTPUT_LO")
    output_hi = _resolve_dim(dim_positions, "OUTPUT_HI_THIS_STEP")
    if (
        const_dim is None
        or output_lo is None
        or output_hi is None
        or const_dim >= ffn.W_up.data.shape[1]
    ):
        return
    output_dims = [
        *(output_lo + i for i in range(16) if output_lo + i < ffn.W_down.data.shape[0]),
        *(output_hi + i for i in range(16) if output_hi + i < ffn.W_down.data.shape[0]),
    ]
    if not output_dims:
        return
    # This guard is only a boundary bias: it should suppress rows with no VM
    # structure, not override the cleanup rule's own blockers. Add the large
    # positive side only on structural dims the unit already uses positively;
    # otherwise a generic IS_BYTE/MEM_VAL_B* allow-list can invert explicit
    # blockers and make cleanup units fire on MEM value bytes.
    strength = S * 10_000_000
    marker_dims = [
        _resolve_dim(dim_positions, name)
        for name in (
            "MARK_AX",
            "MARK_PC",
            "MARK_SP",
            "MARK_BP",
            "MARK_STACK0",
            "MARK_MEM",
        )
    ]
    ranged_positive_dims = []
    for base_name in (
        "H0",
        "H1",
        "H2",
        "H3",
        "H4",
        "L1H0",
        "L1H1",
        "L1H2",
        "L1H4",
        "L2H0",
    ):
        base_dim = _resolve_dim(dim_positions, base_name)
        if base_dim is None:
            continue
        ranged_positive_dims.extend(base_dim + offset for offset in range(5))
    allow_dims = [
        dim
        for dim in (*marker_dims, *ranged_positive_dims)
        if dim is not None and dim < ffn.W_up.data.shape[1]
    ]
    next_marker_dims = [
        _resolve_dim(dim_positions, name)
        for name in (
            "NEXT_PC",
            "NEXT_AX",
            "NEXT_SP",
            "NEXT_BP",
            "NEXT_STACK0",
            "NEXT_MEM",
            "NEXT_SE",
            "NEXT_HALT",
        )
    ]
    next_marker_dims = [
        dim
        for dim in next_marker_dims
        if dim is not None and dim < ffn.W_up.data.shape[1]
    ]
    for unit in range(start_unit, end_unit):
        if float(ffn.W_down.data[output_dims, unit].abs().sum()) == 0.0:
            continue
        ffn.W_up.data[unit, const_dim] -= strength
        for structural_dim in allow_dims:
            if float(ffn.W_up.data[unit, structural_dim]) > 0.0:
                ffn.W_up.data[unit, structural_dim] += strength
        for next_dim in next_marker_dims:
            ffn.W_up.data[unit, next_dim] -= 2 * strength


def _block_l14_jsr_ax_zero_on_stack0_bytes(
    ffn,
    dim_positions,
    S: float,
    start_unit: int,
    end_unit: int,
) -> None:
    """Keep JSR AX-byte zeroing off JSR return-address STACK0 rows."""

    if end_unit <= start_unit:
        return
    stack0_byte_dims = [
        _resolve_dim(dim_positions, name)
        for name in (
            "STACK0_BYTE0",
            "STACK0_BYTE1",
            "STACK0_BYTE2",
            "STACK0_BYTE3",
        )
    ]
    stack0_byte_dims = [
        dim
        for dim in stack0_byte_dims
        if dim is not None and dim < ffn.W_up.data.shape[1]
    ]
    if not stack0_byte_dims:
        return
    strength = S * 1_000.0
    for unit in range(start_unit, end_unit):
        for dim in stack0_byte_dims:
            ffn.W_up.data[unit, dim] -= strength


def _disable_l14_stack0_jsr_hi0_default(
    ffn,
    dim_positions,
    start_unit: int,
    end_unit: int,
) -> None:
    """Remove the stale JSR STACK0 marker high-zero default.

    JSR's STACK0 marker emits the return-address byte. The byte value is
    routed earlier from the PC/AX-carry path and can have a nonzero high
    nibble, so a hard-coded OUTPUT_HI[0] repair is not a valid declaration.
    """

    if end_unit <= start_unit:
        return
    mark_stack0 = _resolve_dim(dim_positions, "MARK_STACK0")
    op_jsr = _resolve_dim(dim_positions, "OP_JSR")
    is_byte = _resolve_dim(dim_positions, "IS_BYTE")
    output_hi = _resolve_dim(dim_positions, "OUTPUT_HI_THIS_STEP")
    if (
        mark_stack0 is None
        or op_jsr is None
        or is_byte is None
        or output_hi is None
        or mark_stack0 >= ffn.W_up.data.shape[1]
        or op_jsr >= ffn.W_up.data.shape[1]
        or is_byte >= ffn.W_up.data.shape[1]
        or output_hi >= ffn.W_down.data.shape[0]
    ):
        return
    for unit in range(start_unit, end_unit):
        if (
            float(ffn.W_up.data[unit, mark_stack0]) <= 0.0
            or float(ffn.W_up.data[unit, op_jsr]) <= 0.0
            or float(ffn.W_up.data[unit, is_byte]) >= 0.0
            or float(ffn.W_down.data[output_hi + 0, unit]) <= 0.0
        ):
            continue
        ffn.W_up.data[unit, :] = 0.0
        ffn.b_up.data[unit] = 0.0
        ffn.W_gate.data[unit, :] = 0.0
        ffn.b_gate.data[unit] = 0.0
        ffn.W_down.data[:, unit] = 0.0


def _boost_l14_psh_mem_marker_high_nibbles(
    ffn,
    dim_positions,
    S: float,
    start_unit: int,
) -> int:
    """Make PSH MEM addr0 high nibble beat the zero default.

    L14 attention already places the post-PSH SP high nibble on the MEM marker,
    but the matched zero-nibble cancel is intentionally soft for real zero
    bytes. At local stack addresses such as 0xffe0, OUTPUT_HI[14] can land
    just below OUTPUT_HI[0]. These units only reinforce nonzero high nibbles
    that are already present on the PSH MEM marker.
    """

    mark_mem = _resolve_dim(dim_positions, "MARK_MEM")
    mem_store = _resolve_dim(dim_positions, "MEM_STORE")
    psh_at_sp = _resolve_dim(dim_positions, "PSH_AT_SP")
    output_hi = _resolve_dim(dim_positions, "OUTPUT_HI_THIS_STEP")
    if (
        mark_mem is None
        or mem_store is None
        or psh_at_sp is None
        or output_hi is None
    ):
        return start_unit

    unit = start_unit
    for nibble in range(1, 16):
        ffn.W_up.data[unit, mark_mem] = S
        ffn.W_up.data[unit, mem_store] = S
        ffn.W_up.data[unit, psh_at_sp] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, output_hi + nibble] = 1.0
        ffn.W_down.data[output_hi + nibble, unit] = 0.1 / S
        unit += 1
    return unit


def _layer14_mem_generation_head_specs(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative L14 heads 0-7 mirroring ``_set_layer14_mem_generation``.

    Heads 0-3 generate MEM address bytes (source: SP for PSH, STACK0 for
    SI/SC, BP/STACK0 for JSR/ENT via post-step ``_clear_*`` overrides).
    Heads 4-7 generate MEM value bytes (source: AX for PSH/SI/SC,
    STACK0 for JSR/ENT). Each head fires at exactly ONE MEM byte position
    via threshold-difference encoding (slot 0) gated by slot 33 (position)
    and slot 34 (MEM_STORE). Slot 38 is a shared non-MEM target blocker.

    Mirrors ``vm_step._set_layer14_mem_generation`` cell-for-cell; the
    only post-step modifications are the targeted overrides applied by
    :func:`_clear_l14_mem_generation_overbroad_sp_suppression`, which
    remain in the bake_fn as residual byte-identity bookkeeping.
    """

    L = 15.0
    PC_I = 0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4

    # Position flags: threshold-difference pairs selecting distance from
    # MEM. To predict addr_bJ (at d=J+1), L14 fires at d=J.
    addr_pos = [
        (BD.MARK_MEM, None),                           # d=0: predicts addr_b0
        (BD.L1H1 + MEM_I, BD.L1H0 + MEM_I),            # d=1: predicts addr_b1
        (BD.L1H2 + MEM_I, BD.L1H1 + MEM_I),            # d=2: predicts addr_b2
        (BD.H0 + MEM_I,   BD.L1H2 + MEM_I),            # d=3: predicts addr_b3
    ]
    val_pos = [
        (BD.H1 + MEM_I,   BD.H0 + MEM_I),              # d=4: predicts val_b0
        (BD.L2H0 + MEM_I, BD.H1 + MEM_I),              # d=5: predicts val_b1
        (BD.L1H4 + MEM_I, BD.L2H0 + MEM_I),            # d=6: predicts val_b2
        (BD.H2 + MEM_I,   BD.L1H4 + MEM_I),            # d=7: predicts val_b3
    ]

    target_block_s = 2000.0

    specs: list[DeclarativeAttentionHeadSpec] = []

    # === Heads 0-3: MEM addr byte generation ===
    for h in range(4):
        pos_up, pos_down = addr_pos[h]
        q: list[AP] = []
        k: list[AP] = []

        # Slot 0: Q position selection + suppression rows.
        q.append(AP(0, pos_up, L))
        if pos_down is not None:
            q.append(AP(0, pos_down, -L))
        q.append(AP(0, BD.MARK_STACK0, -L))
        q.append(AP(0, BD.H4 + BP_I,   -L))
        q.append(AP(0, BD.H1 + SP_I,   -L))

        # Slot 0: K source selection (head-0 dual K, others byte-index K).
        if h == 0:
            k.append(AP(0, BD.MARK_SP,      L))
            k.append(AP(0, BD.STACK0_BYTE0, L))
        else:
            byte_idx_dim = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
            k.append(AP(0, byte_idx_dim, L))

        # Slot 1: SP source bonus (PSH).
        q.append(AP(1, BD.CONST,         L))
        q.append(AP(1, BD.MEM_ADDR_SRC, -2 * L))
        k.append(AP(1, BD.H1 + SP_I,  L))
        k.append(AP(1, BD.H1 + AX_I, -L))

        # Slot 2: STACK0 source bonus (SI/SC).
        q.append(AP(2, BD.MEM_ADDR_SRC, L))
        if h == 0:
            k.append(AP(2, BD.STACK0_BYTE0, L))
        elif h == 1:
            k.append(AP(2, BD.L1H4 + BP_I,  L))
            k.append(AP(2, BD.H1   + BP_I, -L))
        elif h == 2:
            k.append(AP(2, BD.H2   + BP_I,  L))
            k.append(AP(2, BD.L1H4 + BP_I, -L))
        elif h == 3:
            k.append(AP(2, BD.H3   + BP_I,  L))
            k.append(AP(2, BD.H2   + BP_I, -L))
        k.append(AP(2, BD.H1 + AX_I,    -L))
        k.append(AP(2, BD.H1 + SP_I,    -L))
        k.append(AP(2, BD.MARK_STACK0,  -L))

        # Slot 33: Position gate.
        q.append(AP(33, BD.CONST,        -500.0))
        q.append(AP(33, pos_up,           500.0))
        if pos_down is not None:
            q.append(AP(33, pos_down,    -500.0))
        q.append(AP(33, BD.MARK_STACK0,  -500.0))
        q.append(AP(33, BD.L1H4 + BP_I,  -500.0))
        q.append(AP(33, BD.H1 + SP_I,    -500.0))
        k.append(AP(33, BD.CONST,         5.0))

        # Slot 34: MEM_STORE gate.
        q.append(AP(34, BD.CONST,        -500.0))
        q.append(AP(34, BD.MEM_STORE,     500.0))
        k.append(AP(34, BD.CONST,         5.0))

        # Slot 38: shared non-MEM target blocker.
        q.append(AP(38, BD.MARK_PC,      -target_block_s))
        q.append(AP(38, BD.MARK_AX,      -target_block_s))
        q.append(AP(38, BD.MARK_BP,      -target_block_s))
        q.append(AP(38, BD.MARK_STACK0,  -target_block_s))
        q.append(AP(38, BD.H1 + PC_I,    -target_block_s))
        q.append(AP(38, BD.H1 + AX_I,    -target_block_s))
        q.append(AP(38, BD.H1 + BP_I,    -target_block_s))
        q.append(AP(38, BD.H4 + BP_I,    -target_block_s))
        k.append(AP(38, BD.CONST,         5.0))

        # V: read CLEAN_EMBED + OUTPUT (sum).
        v: list[AP] = []
        v.append(AP(0, BD.CONST, 1.0))
        for kk in range(16):
            v.append(AP(1  + kk, BD.CLEAN_EMBED_LO + kk, 1.0))
            v.append(AP(1  + kk, BD.OUTPUT_LO      + kk, 1.0))
            v.append(AP(17 + kk, BD.CLEAN_EMBED_HI + kk, 1.0))
            v.append(AP(17 + kk, BD.OUTPUT_HI      + kk, 1.0))

        # O: write to OUTPUT_LO/HI + cancel L3 default at byte 0.
        o: list[AO] = []
        o.append(AO(BD.OUTPUT_LO + 0, 0, -1.0))
        o.append(AO(BD.OUTPUT_HI + 0, 0, -1.0))
        for kk in range(16):
            o.append(AO(BD.OUTPUT_LO + kk, 1  + kk, 1.0))
            o.append(AO(BD.OUTPUT_HI + kk, 17 + kk, 1.0))

        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=_L14_HEAD_LAYOUT_BY_NAME[f"layer14_mem_generation.head_{h}"],
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ))

    # === Heads 4-7: MEM val byte generation ===
    for h in range(4):
        head_idx = 4 + h
        pos_up, pos_down = val_pos[h]
        byte_idx_dim = [
            BD.BYTE_INDEX_0,
            BD.BYTE_INDEX_1,
            BD.BYTE_INDEX_2,
            BD.BYTE_INDEX_3,
        ][h]

        q = []
        k = []

        # Slot 0: Q position selection + suppression; K targets byte position.
        q.append(AP(0, pos_up,         L))
        q.append(AP(0, pos_down,      -L))
        q.append(AP(0, BD.MARK_STACK0, -L))
        q.append(AP(0, BD.H4 + BP_I,   -L))
        q.append(AP(0, BD.H1 + SP_I,   -L))
        k.append(AP(0, byte_idx_dim,    L))

        # Slot 1: AX source bonus (PSH/SI/SC).
        q.append(AP(1, BD.CONST,   L))
        q.append(AP(1, BD.OP_JSR, -2 * L))
        q.append(AP(1, BD.OP_ENT, -2 * L))
        k.append(AP(1, BD.H1 + AX_I, L))

        # Slot 2: STACK0 source bonus (JSR/ENT).
        q.append(AP(2, BD.OP_JSR, L))
        q.append(AP(2, BD.OP_ENT, L))
        if h == 0:
            k.append(AP(2, BD.STACK0_BYTE0, L))
        elif h == 1:
            k.append(AP(2, BD.H2   + BP_I,  L))
            k.append(AP(2, BD.L1H4 + BP_I, -L))
        elif h == 2:
            k.append(AP(2, BD.H3 + BP_I,  L))
            k.append(AP(2, BD.H2 + BP_I, -L))
        elif h == 3:
            k.append(AP(2, BD.H4 + BP_I,  L))
            k.append(AP(2, BD.H3 + BP_I, -L))
        k.append(AP(2, BD.H1 + AX_I,    -L))
        k.append(AP(2, BD.H1 + SP_I,    -L))
        k.append(AP(2, BD.MARK_STACK0,  -L))

        # Slot 33: Position gate.
        q.append(AP(33, BD.CONST,        -500.0))
        q.append(AP(33, pos_up,           500.0))
        q.append(AP(33, pos_down,        -500.0))
        q.append(AP(33, BD.MARK_STACK0,  -500.0))
        q.append(AP(33, BD.L1H4 + BP_I,  -500.0))
        q.append(AP(33, BD.H1 + SP_I,    -500.0))
        k.append(AP(33, BD.CONST,         5.0))

        # Slot 34: MEM_STORE gate.
        q.append(AP(34, BD.CONST,        -500.0))
        q.append(AP(34, BD.MEM_STORE,     500.0))
        k.append(AP(34, BD.CONST,         5.0))

        # Slot 38: shared non-MEM target blocker.
        q.append(AP(38, BD.MARK_PC,      -target_block_s))
        q.append(AP(38, BD.MARK_AX,      -target_block_s))
        q.append(AP(38, BD.MARK_BP,      -target_block_s))
        q.append(AP(38, BD.MARK_STACK0,  -target_block_s))
        q.append(AP(38, BD.H1 + PC_I,    -target_block_s))
        q.append(AP(38, BD.H1 + AX_I,    -target_block_s))
        q.append(AP(38, BD.H1 + BP_I,    -target_block_s))
        q.append(AP(38, BD.H4 + BP_I,    -target_block_s))
        k.append(AP(38, BD.CONST,         5.0))

        # V: copy CLEAN_EMBED only (no OUTPUT — see legacy 2026-04-16 fix).
        v = []
        v.append(AP(0, BD.CONST, 1.0))
        for kk in range(16):
            v.append(AP(1  + kk, BD.CLEAN_EMBED_LO + kk, 1.0))
            v.append(AP(17 + kk, BD.CLEAN_EMBED_HI + kk, 1.0))

        # O: write to OUTPUT_LO/HI + cancel L3 default at byte 0.
        o = []
        o.append(AO(BD.OUTPUT_LO + 0, 0, -1.0))
        o.append(AO(BD.OUTPUT_HI + 0, 0, -1.0))
        for kk in range(16):
            o.append(AO(BD.OUTPUT_LO + kk, 1  + kk, 1.0))
            o.append(AO(BD.OUTPUT_HI + kk, 17 + kk, 1.0))

        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=_L14_HEAD_LAYOUT_BY_NAME[f"layer14_mem_generation.head_{head_idx}"],
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ))

    return tuple(specs)


def _layer14_mem_generation_ir(dim_positions, HD) -> CompilerIR:
    """Build the declarative L14 mem-generation IR for the compiler.

    Eight heads (0-3 address, 4-7 value). See
    :func:`_layer14_mem_generation_head_specs` for the byte-level layout.
    """

    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer14_mem_generation_head_specs(proxy):
        ir.layer(0).attention.append(spec)
    return ir


def make_layer14_mem_generation_op() -> Operation:
    """L14 attention: generate MEM section tokens (addr + value) for SI/SC/PSH.

    Wave 2E (Phase 6): migrated to ``DeclarativeAttentionHeadSpec`` form.
    Heads 0-3 (address) and 4-7 (value) are expressed declaratively via
    :func:`_layer14_mem_generation_head_specs`; the imperative
    :func:`vm_step._set_layer14_mem_generation` helper is no longer
    called. Byte-identity is preserved end-to-end (legacy ``W_q``/``W_k``
    /``W_v``/``W_o`` writes are reproduced cell-for-cell via the spec,
    then ``_clear_l14_mem_generation_overbroad_sp_suppression`` applies
    the same residual overrides as before).
    """

    def bake(attn, dim_positions, S):
        del S
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator with the L14 head layout
        # pinned. Stashed on ``attn`` for downstream inspection/extension.
        head_allocator = _allocate_layer14_mem_generation_heads()
        attn._l14_head_allocator = head_allocator
        Primitives.generate_attention_heads(
            attn, _layer14_mem_generation_head_specs(proxy), HD,
        )
        _clear_l14_mem_generation_overbroad_sp_suppression(attn, proxy, HD)
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[:8] = 5.0

    # Dim-ownership claims: L14 attn 8 heads MEM generation.
    # Each head writes V slots 1..32 reading CLEAN_EMBED + OUTPUT.  For V
    # slots, the CLEAN_EMBED column is the load-bearing "source" — the
    # +OUTPUT addition just lets V pick up either side (marker or byte
    # position). We claim only the CLEAN_EMBED rows (OUTPUT collisions on
    # the same V slot are recurring in this op family and acceptable).
    _claims = set()
    for h in range(8):
        for k in range(16):
            _claims.add((14, "attn_W_v", f"{h}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((14, "attn_W_v", f"{h}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer14_mem_generation",
        phase=14,
        reads={"MARK_MEM", "MARK_SP", "MARK_STACK0", "OP_PSH", "OP_SI", "OP_SC",
               "OP_JSR", "OP_ENT", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_B0_LO", "ADDR_B0_HI",
               "MEM_STORE", "MEM_ADDR_SRC", "STACK0_BYTE0", "L1H0", "L1H1", "L1H2",
               "H0", "H1", "L1H4", "H2", "H3", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3", "IS_BYTE"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        layer_idx=14,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer14_mem_generation_ir,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        alibi_slopes={head: 5.0 for head in range(8)},
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeFunctionCall::test_simple_function",
            "TestSmokeMemory::test_sc_lc_roundtrip",
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer14_alu_high_byte_relay_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Relay staged wide-ALU byte 1 from the AX marker to AX byte 0."""

    PC_I = 0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4
    q = [
        AP(0, BD.IS_BYTE, 100.0),
        AP(0, BD.H1 + AX_I, 100.0),
        AP(0, BD.BYTE_INDEX_0, 100.0),
        AP(0, BD.BYTE_INDEX_1, -1000.0),
        AP(0, BD.BYTE_INDEX_2, -1000.0),
        AP(0, BD.BYTE_INDEX_3, -1000.0),
        AP(0, BD.CONST, -250.0),
        AP(1, BD.IS_BYTE, 100.0),
        AP(1, BD.H1 + AX_I, 100.0),
        AP(1, BD.BYTE_INDEX_0, 100.0),
        AP(1, BD.BYTE_INDEX_1, -1000.0),
        AP(1, BD.BYTE_INDEX_2, -1000.0),
        AP(1, BD.BYTE_INDEX_3, -1000.0),
        AP(1, BD.CONST, -250.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_I, 10000.0),
        AP(33, BD.BYTE_INDEX_0, 10000.0),
        AP(33, BD.BYTE_INDEX_1, -100000.0),
        AP(33, BD.BYTE_INDEX_2, -100000.0),
        AP(33, BD.BYTE_INDEX_3, -100000.0),
        AP(33, BD.CONST, -25000.0),
        AP(34, BD.OP_LI_RELAY, -10000.0),
        AP(34, BD.OP_LC_RELAY, -10000.0),
        AP(35, BD.TEMP + 8, 10000.0),
        AP(35, BD.TEMP + 9, 10000.0),
        AP(36, BD.IS_BYTE, 1000.0),
        AP(36, BD.H1 + AX_I, 1000.0),
        AP(36, BD.BYTE_INDEX_0, 1000.0),
        AP(36, BD.BYTE_INDEX_1, -3000.0),
        AP(36, BD.BYTE_INDEX_2, -3000.0),
        AP(36, BD.BYTE_INDEX_3, -3000.0),
        AP(36, BD.MARK_AX, -1000.0),
        AP(36, BD.MARK_PC, -6000.0),
        AP(36, BD.H1 + PC_I, -6000.0),
        AP(36, BD.MARK_SP, -6000.0),
        AP(36, BD.H1 + SP_I, -6000.0),
        AP(36, BD.H4 + SP_I, -6000.0),
        AP(36, BD.MARK_BP, -6000.0),
        AP(36, BD.H1 + BP_I, -6000.0),
        AP(36, BD.H4 + BP_I, -6000.0),
        AP(36, BD.H1 + MEM_I, -6000.0),
        AP(36, BD.H3 + MEM_I, -6000.0),
        AP(36, BD.H4 + MEM_I, -6000.0),
        AP(36, BD.MARK_STACK0, -6000.0),
        AP(36, BD.MARK_MEM, -6000.0),
        AP(36, BD.STACK0_BYTE0, -6000.0),
    ]
    k = [
        AP(0, BD.MARK_AX, 100.0),
        AP(1, BD.OP_MUL, 100.0),
        AP(1, BD.OP_SHL, 100.0),
        AP(33, BD.CONST, 5.0),
        AP(34, BD.CONST, 5.0),
        AP(35, BD.CONST, -20.0),
        AP(35, BD.MARK_AX, -10000.0),
        AP(35, BD.OP_MUL, 2000.0),
        AP(35, BD.OP_SHL, 2000.0),
        AP(36, BD.CONST, -100.0),
        AP(36, BD.OP_MUL, 200.0),
        AP(36, BD.OP_SHL, 200.0),
    ]
    v = []
    o = []
    for idx in range(16):
        v.append(AP(1 + idx, BD.AX_FULL_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.AX_FULL_HI + idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + idx, 1 + idx, 20.0))
        o.append(AO(BD.OUTPUT_HI + idx, 17 + idx, 20.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=8,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer14_alu_high_byte_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer14_alu_high_byte_relay_spec(BD))
    return ir


def make_layer14_alu_high_byte_relay_op() -> Operation:
    """L15 attention: relay staged MUL/SHL result byte 1 into AX bytes.

    The wide ALU composites stage result byte 1 in AX_FULL_LO/HI at the AX
    marker. The autoregressive token stream then needs the following AX byte
    token to emit that staged byte. This declarative head copies the staged
    byte from the current step's AX marker to the AX byte-0 query position.
    It lives in the resized L15 attention block because the base L14 attention
    has only heads 0-7 occupied by MEM generation.
    """

    def bake(target, dim_positions, S):
        del S
        attn = getattr(target, "attn", target)
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer14_alu_high_byte_relay_spec(proxy),
            HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[8] = 1.0

    _claims = set()
    for k in range(16):
        _claims.add((15, "attn_W_v", f"8_{1 + k}", f"AX_FULL_LO+{k}"))
        _claims.add((15, "attn_W_v", f"8_{17 + k}", f"AX_FULL_HI+{k}"))

    return Operation(
        name="layer15_alu_high_byte_relay",
        phase=15.05,
        reads={"IS_BYTE", "H1", "H3", "H4", "BYTE_INDEX_0", "MARK_AX",
               "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "STACK0_BYTE0", "OP_MUL",
               "OP_SHL", "OP_LI_RELAY", "OP_LC_RELAY",
               "AX_FULL_LO", "AX_FULL_HI", "TEMP", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        layer_idx=15,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir_factory=_layer14_alu_high_byte_relay_ir,
        migrated=True,
        claims=_claims,
        alibi_slopes={8: 1.0},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmoke32Bit::test_shl_8bit",
        },
        spec_section="BLOG_SPEC.md#wide-alu-byte-relay",
    )


def _clear_l14_mem_generation_overbroad_sp_suppression(attn, BD, HD) -> None:
    """Keep L14 MEM generation live when H1[SP] also fires at MEM positions.

    The legacy helper suppresses queries whenever ``H1[SP]`` is high to avoid
    early SP-byte pollution. In the strict autoregressive trace, ``H1[SP]`` is
    also high at the MEM marker, so heads 0-3 select the softmax1 sink instead
    of the SP marker and emit zero address bytes. The position gate already
    suppresses non-MEM targets strongly; this removes only the overbroad
    H1[SP] query penalties.
    """

    pc_i = 0
    ax_i = 1
    sp_i = 2
    bp_i = 3
    mem_i = 4
    for head in range(8):
        base = head * HD
        attn.W_q.data[base, BD.H1 + sp_i] = 0.0
        # MEM_STORE is represented with strength 2 at MEM positions, so the
        # legacy +/-500 position gate can exactly cancel the +500 MEM_STORE
        # gate at non-target MEM offsets. Doubling the position gate leaves
        # target positions at zero net gate but makes non-target heads stay
        # below the softmax1 sink.
        attn.W_q.data[base + 33, :] *= 2.0
        attn.W_q.data[base + 33, BD.H1 + sp_i] = 0.0
        # L14's sources are the prior SP/AX/STACK0 sections. Once MEM bytes
        # start emitting, their BYTE_INDEX flags are more recent than the true
        # sources and can win via ALiBi. Use an otherwise-unused score slot to
        # exclude the current MEM section as an attention source for every head.
        attn.W_q.data[base + 35, BD.CONST] = 40.0
        attn.W_k.data[base + 35, BD.MARK_MEM] = -40.0
        attn.W_k.data[base + 35, BD.H3 + 4] = -40.0

        # The legacy position gate was calibrated before the stricter
        # source-bonus rows below. In neural-authoritative smoke, PC/AX/BP
        # byte queries can still overcome that gate and make MEM generation
        # write OUTPUT outside the MEM section. Add a separate non-MEM target
        # blocker that stays inactive at the real MEM addr/value targets.
        target_block_s = 5000.0
        for dim in (
            BD.MARK_PC,
            BD.MARK_AX,
            BD.MARK_SP,
            BD.MARK_BP,
            BD.MARK_STACK0,
            BD.H1 + pc_i,
            BD.H1 + ax_i,
            BD.H1 + bp_i,
            BD.H4 + bp_i,
        ):
            attn.W_q.data[base + 38, dim] = -target_block_s
        attn.W_k.data[base + 38, BD.CONST] = 5.0

    # Address heads own only the MEM marker/address-byte lanes.  Keep them off
    # the MEM value lanes so their address source rows cannot overwrite stored
    # value bytes during ENT/JSR.
    for head in range(4):
        base = head * HD
        attn.W_q.data[base + 38, BD.H1 + sp_i] = -target_block_s
        for dim in (
            BD.MEM_VAL_B0,
            BD.MEM_VAL_B1,
            BD.MEM_VAL_B2,
            BD.MEM_VAL_B3,
        ):
            attn.W_q.data[base + 38, dim] = -target_block_s

    # Head 0 predicts address byte 0 from the SP marker for PSH. The legacy
    # source bonus keyed only on H1[SP], which is active on SP byte positions
    # but not on the SP marker where byte 0's fresh OUTPUT lives.
    attn.W_k.data[1, BD.MARK_SP] = 45.0
    # Heads 1-3 can read the already-emitted SP byte tokens directly. The
    # old shifted-OUTPUT path is fragile in strict neural mode; this bonus is
    # active only for PSH because Q[base+1] flips negative when MEM_ADDR_SRC=1.
    # Suppress the BP/STACK0 span on the same source dimension: BYTE_INDEX_*
    # is deliberately global, and without this guard the nearer STACK0 zero
    # bytes beat the SP address bytes under ALiBi in strict autoregressive
    # smoke.
    for head in range(4):
        attn.W_k.data[head * HD + 1, BD.H4 + bp_i] = -30.0
        attn.W_k.data[head * HD + 1, BD.MARK_STACK0] = -30.0
    for head, byte_dim in (
        (1, BD.BYTE_INDEX_1),
        (2, BD.BYTE_INDEX_2),
        (3, BD.BYTE_INDEX_3),
    ):
        # These heads now source the SP/STACK0 byte token directly, so reading
        # that token's OUTPUT would mix in the source position's prediction for
        # the following byte (usually the L3 zero default). Keep CLEAN_EMBED as
        # the only payload source while retaining V[0]'s default cancel.
        base = head * HD
        attn.W_v.data[base + 0, BD.CONST] = 1.0
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 0.0
            attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 0.0
        attn.W_k.data[head * HD + 1, byte_dim] = 60.0

    # SI/SC addresses come from the stack top before the pop.  By the time the
    # MEM section is emitted, the current step's STACK0 bytes have already been
    # regenerated to the post-pop value.  Those current rows carry MEM_STORE
    # leakage, so block them and let the most recent pre-store STACK0 row win.
    si_source_s = 200.0
    for head in range(4):
        base = head * HD
        for dim in (
            BD.MEM_STORE,
            BD.STACK0_BYTE0,
            BD.H1 + bp_i,
            BD.L1H4 + bp_i,
            BD.H2 + bp_i,
            BD.H3 + bp_i,
            BD.H4 + bp_i,
            BD.MARK_STACK0,
            BD.H1 + ax_i,
            BD.H1 + sp_i,
        ):
            attn.W_k.data[base + 2, dim] = 0.0
        attn.W_k.data[base + 2, BD.MEM_STORE] = -4.0 * si_source_s
        attn.W_k.data[base + 2, BD.H1 + ax_i] = -si_source_s
        attn.W_k.data[base + 2, BD.H1 + sp_i] = -si_source_s
        if head == 0:
            # SI/SC byte 0 must come from the pre-store stack-top byte. The
            # STACK0 marker can still carry the just-stored AX byte before the
            # late tail correction, which would turn the store value into the
            # store address.
            attn.W_k.data[base + 2, BD.STACK0_BYTE0] = si_source_s
        elif head == 1:
            attn.W_k.data[base + 2, BD.MARK_STACK0] = -si_source_s
            attn.W_k.data[base + 2, BD.H2 + bp_i] = si_source_s
            attn.W_k.data[base + 2, BD.L1H4 + bp_i] = -si_source_s
        elif head == 2:
            attn.W_k.data[base + 2, BD.MARK_STACK0] = -si_source_s
            attn.W_k.data[base + 2, BD.H3 + bp_i] = si_source_s
            attn.W_k.data[base + 2, BD.H2 + bp_i] = -si_source_s
        else:
            attn.W_k.data[base + 2, BD.MARK_STACK0] = -si_source_s
            attn.W_k.data[base + 2, BD.H4 + bp_i] = si_source_s
            attn.W_k.data[base + 2, BD.H3 + bp_i] = -si_source_s

    # ENT stores the old BP at the freshly established frame address
    # (BP = old SP - 8). Address heads must therefore source BP, not the
    # final SP value after local allocation.
    ent_addr_s = 50.0
    ent_wrong_target_s = 500.0
    for head in range(4):
        base = head * HD
        attn.W_q.data[base + 39, BD.OP_ENT] = ent_addr_s
        attn.W_q.data[base + 39, BD.HAS_SE] = ent_addr_s * 3.0
        attn.W_q.data[base + 39, BD.CONST] = -ent_addr_s * 6.0
        if head == 0:
            attn.W_q.data[base + 39, BD.IS_BYTE] = -12.0 * ent_addr_s
            attn.W_k.data[base + 39, BD.MARK_BP] = ent_addr_s
            continue
        attn.W_k.data[base + 39, BD.H1 + bp_i] = 0.0
        attn.W_q.data[base + 40, BD.OP_ENT] = ent_addr_s
        attn.W_q.data[base + 40, BD.HAS_SE] = 0.0
        attn.W_q.data[base + 40, BD.CONST] = 0.0
        for dim in (BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2):
            attn.W_q.data[base + 40, dim] = 0.0
        allowed_query_dim = (
            BD.BYTE_INDEX_0,
            BD.BYTE_INDEX_1,
            BD.BYTE_INDEX_2,
        )[head - 1]
        for dim in (
            BD.BYTE_INDEX_0,
            BD.BYTE_INDEX_1,
            BD.BYTE_INDEX_2,
            BD.BYTE_INDEX_3,
        ):
            if dim != allowed_query_dim:
                attn.W_q.data[base + 40, dim] = -ent_wrong_target_s
        attn.W_k.data[base + 40, BD.H1 + bp_i] = ent_addr_s
        attn.W_k.data[
            base + 40,
            (BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3)[head - 1],
        ] = ent_addr_s
    # At the MEM marker only head 0 should emit address byte 0. Heads 1-3 use
    # the already-emitted address bytes as their query positions, and the ENT
    # source rows can otherwise overpower their non-target position gate and
    # add zero-byte defaults into addr byte 0.
    mem_marker_block_s = 300.0
    for head in range(1, 4):
        base = head * HD
        attn.W_q.data[base + 41, BD.MARK_MEM] = -mem_marker_block_s
        attn.W_k.data[base + 41, BD.CONST] = mem_marker_block_s
    ent_target_gate_s = 200.0
    for head, query_dim in (
        (1, BD.BYTE_INDEX_0),
        (2, BD.BYTE_INDEX_1),
        (3, BD.BYTE_INDEX_2),
    ):
        base = head * HD
        attn.W_q.data[base + 42, BD.CONST] = 0.0
        attn.W_q.data[base + 42, BD.OP_ENT] = 0.0
        attn.W_q.data[base + 42, query_dim] = 0.0
        attn.W_k.data[base + 42, BD.CONST] = 0.0
        attn.W_q.data[base + 43, BD.CONST] = -ent_target_gate_s
        attn.W_q.data[base + 43, BD.H3 + mem_i] = ent_target_gate_s
        attn.W_k.data[base + 43, BD.CONST] = ent_target_gate_s

    # Value heads source AX for PSH/SI/SC and STACK0 for JSR/ENT. With the
    # steeper L14 ALiBi slope, the old H1[AX] bonus is not strong enough to
    # beat the softmax1 sink from MEM value positions. This score slot makes
    # the source choice explicit without changing the payload path.
    source_s = 50.0
    for head in range(4, 8):
        base = head * HD
        # The value heads double the CLEAN_EMBED nibble payload to overcome
        # downstream defaults. Keep V[0] as a matched cancel so nonzero bytes
        # beat the L3 zero default while real zero bytes still emit 0.
        attn.W_v.data[base + 0, BD.CONST] = 2.0
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 2.0
            attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 2.0
        attn.W_q.data[base + 36, BD.CONST] = source_s
        attn.W_q.data[base + 36, BD.OP_JSR] = -2.0 * source_s
        attn.W_q.data[base + 36, BD.OP_ENT] = -2.0 * source_s
        attn.W_k.data[base + 36, BD.H1 + ax_i] = source_s
        # Keep SP neutral for the value-source selector. With JSR/ENT the
        # query row is negative so a negative H1[SP] key makes the current SP
        # byte an attractive source; that corrupts the same step's SP byte
        # outputs by rereading SP_byte0 as SP_byte1.
        attn.W_k.data[base + 36, BD.H1 + sp_i] = 0.0
        attn.W_k.data[base + 36, BD.H4 + bp_i] = -source_s
        # STACK0 value bytes carry H4[BP], while the STACK0 marker carries
        # both H4[BP] and MARK_STACK0. For JSR/ENT the query is negative, so
        # H4[BP]'s negative K makes STACK0 bytes attractive. Give the marker a
        # matching positive term so it nets to zero instead of winning as a
        # payload-less source.
        attn.W_k.data[base + 36, BD.MARK_STACK0] = source_s
        # MEM-source exclusion is handled by the sign-stable row 35 above.
        # Do not put MEM dims on this sign-flipping selector row: for ENT/JSR
        # the query is negative, so negative MEM keys make the current MEM
        # section look like a source and beat the intended STACK0 bytes.
        attn.W_k.data[base + 36, BD.MARK_MEM] = 0.0
        attn.W_k.data[base + 36, BD.H3 + mem_i] = 0.0

        # Heads 4-7 generate MEM value bytes only. Keep them silent while the
        # MEM marker and address bytes are being emitted; heads 0-3 own those
        # address positions. Value byte 0 is predicted from the BYTE_INDEX_3
        # query, so only byte indexes 0..2 are blocked here.
        value_target_block_s = 15000.0
        for dim in (
            BD.MARK_MEM,
            BD.BYTE_INDEX_0,
            BD.BYTE_INDEX_1,
            BD.BYTE_INDEX_2,
        ):
            attn.W_q.data[base + 38, dim] = -value_target_block_s
        mem_val_dims = (
            BD.MEM_VAL_B0,
            BD.MEM_VAL_B1,
            BD.MEM_VAL_B2,
            BD.MEM_VAL_B3,
        )
        own_value_idx = head - 4
        for idx, dim in enumerate(mem_val_dims):
            if idx != own_value_idx:
                attn.W_q.data[base + 38, dim] = -value_target_block_s

        # ENT stores the caller's old BP. After a normal call prologue that
        # old BP lives in the previous JSR step's BP byte rows, not in the
        # current STACK0 rows used by JSR's return-address store.
        ent_old_bp_s = 80.0
        ent_value_target_s = 1000.0
        source_byte_dim = (
            BD.BYTE_INDEX_0,
            BD.BYTE_INDEX_1,
            BD.BYTE_INDEX_2,
            BD.BYTE_INDEX_3,
        )[own_value_idx]
        target_query_dim = (
            BD.MEM_VAL_B0,
            BD.MEM_VAL_B1,
            BD.MEM_VAL_B2,
            BD.MEM_VAL_B3,
        )[own_value_idx]
        attn.W_q.data[base + 44, BD.OP_ENT] = ent_old_bp_s
        attn.W_k.data[base + 44, BD.OP_JSR] = ent_old_bp_s
        attn.W_k.data[base + 44, BD.H1 + bp_i] = ent_old_bp_s
        attn.W_k.data[base + 44, source_byte_dim] = ent_old_bp_s
        attn.W_q.data[base + 45, BD.CONST] = -0.9 * ent_value_target_s
        attn.W_q.data[base + 45, target_query_dim] = ent_value_target_s
        attn.W_k.data[base + 45, BD.CONST] = ent_value_target_s

        # PSH store values are sourced from AX. STACK0 is generated later in
        # the same step and is not authoritative for the MEM value bytes here.

        # SI/SC AX preservation now happens late in L16 before the MEM value
        # bytes are generated. Keep the source selector byte-indexed on the
        # current AX bytes; the previous "prefer older AX" penalty makes byte 0
        # lose to the nearer byte 3 zero under strict neural ALiBi.
        attn.W_q.data[base + 37, :] = 0.0
        attn.W_k.data[base + 37, :] = 0.0

    # The legacy MEM-generation heads use V slot 0 as the matched zero-nibble
    # cancel. A full cancel leaves real zero nibbles at exactly the same logit
    # as every wrong nibble in that band, so batched GEMM/SDPA rounding can
    # choose any token with the same other nibble. Keep the cancel strong
    # enough to suppress zero when the copied nibble is nonzero, but leave a
    # positive margin when the copied nibble itself is zero.
    for head in range(8):
        base = head * HD
        attn.W_o.data[BD.OUTPUT_LO + 0, base + 0] = -0.5
        attn.W_o.data[BD.OUTPUT_HI + 0, base + 0] = -0.5


def _layer14_temp_clear_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for the ``layer14_temp_clear`` 4-unit substage chain.

    Spans the three legacy helpers consolidated under this op:
    :func:`_set_layer14_temp_clear` (1 unit), then
    :func:`_set_layer14_clear_addsub_temp_negative_residue` (2 units),
    then :func:`_set_layer14_add_byte1_high_zero_cleanup` (1 unit).
    Every unit uses ``gate="CONST"`` with ``gate_weight=1.0`` and
    ``gate_bias=0.0`` to reproduce the imperative
    ``ffn.W_gate[unit, BD.CONST] = 1.0`` / unset-``b_gate`` pair
    byte-identically (``constant_write`` would instead set
    ``b_gate=1.0`` with ``W_gate[CONST]=0.0`` — a different matrix).
    """
    AX_I = 1
    rules = (
        # Unit 0: clear TEMP[0] at PC marker when OP_LEV active.
        FFNRule.gated_write(
            name="l14_temp_clear_pc_lev",
            conditions=(
                ("OP_LEV", 0.1),    # S * 0.1 == legacy W_up[..., OP_LEV] = S/10
                ("MARK_PC", 1.0),
            ),
            threshold=1.5,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("TEMP+0", -5.0 / S),),
            scope="OP_LEV and MARK_PC",
            dominates_at={"TEMP+0": "OP_LEV and MARK_PC"},
        ),
        # Unit 1: TEMP[8] negative residue clamp.
        FFNRule.gated_write(
            name="l14_clear_addsub_temp_negative_residue_8",
            conditions=(("TEMP+8", -1.0),),
            threshold=0.0,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("TEMP+8", 2.0 / S),),
        ),
        # Unit 2: TEMP[9] negative residue clamp.
        FFNRule.gated_write(
            name="l14_clear_addsub_temp_negative_residue_9",
            conditions=(("TEMP+9", -1.0),),
            threshold=0.0,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("TEMP+9", 2.0 / S),),
        ),
        # Unit 3: ADD byte-1 high-nibble zero cleanup.
        FFNRule.gated_write(
            name="l14_add_byte1_high_zero_cleanup",
            conditions=(
                ("IS_BYTE", 1.0),
                (f"H1+{AX_I}", 1.0),
                ("BYTE_INDEX_0", 1.0),
                ("TEMP+8", 1.0),
                ("TEMP+9", -10.0),
                ("AX_CARRY_HI+15", -10_000_000.0),
                ("BYTE_INDEX_1", -10.0),
                ("BYTE_INDEX_2", -10.0),
                ("BYTE_INDEX_3", -10.0),
                ("MARK_AX", -100.0),
            ),
            threshold=3.5,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(
                ("OUTPUT_HI_THIS_STEP+0", 50.0 / S),
                *(
                    (f"OUTPUT_HI_THIS_STEP+{nonzero}", -5000.0 / S)
                    for nonzero in range(1, 16)
                ),
            ),
        ),
    )
    return rules


def _layer14_temp_clear_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_temp_clear_rules(S))
    return ir


def make_layer14_temp_clear_op() -> Operation:
    """L14 FFN: Clear TEMP[0] at PC marker when OP_LEV is active.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Chains with the other L14
    additive cleanup ops (``layer14_clear_addr_key_pollution``,
    ``layer14_clear_output_corruption``) via a shared FFN unit counter stored
    on ``block.ffn._l14_unit_counter``. First in the chain (phase=14.1).

    Migration (Phase 6 wave 3J): the 4 hidden units (across the legacy
    helpers ``_set_layer14_temp_clear``,
    ``_set_layer14_clear_addsub_temp_negative_residue``, and
    ``_set_layer14_add_byte1_high_zero_cleanup``) are now declared via
    :func:`_layer14_temp_clear_rules` and attached as ``compiler_ir``;
    bake lowers through :meth:`CompilerIR.lower_ffn`.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Pinned to the chain head (offset 0). Byte-identical with the
        # legacy ``_l14_unit_counter`` start (the counter is zero on a
        # fresh FFN before any L14 cleanup op has baked).
        start_unit = _l14_chain_alloc("layer14_temp_clear")
        ir = _layer14_temp_clear_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells; bias / W_up cells are not
    # part of the claim grid). First op in the L14 cleanup chain so units
    # start at 0.
    #   unit 0: _set_layer14_temp_clear writes TEMP[0]
    #   unit 1: _set_layer14_clear_addsub_temp_negative_residue writes TEMP[8]
    #   unit 2: _set_layer14_clear_addsub_temp_negative_residue writes TEMP[9]
    #   unit 3: _set_layer14_add_byte1_high_zero_cleanup writes OUTPUT_HI[0..15]
    _claims = {
        (14, "ffn_W_down", "0", "TEMP+0"),
        (14, "ffn_W_down", "1", "TEMP+8"),
        (14, "ffn_W_down", "2", "TEMP+9"),
    }
    for k in range(16):
        _claims.add((14, "ffn_W_down", "3", f"OUTPUT_HI_THIS_STEP+{k}"))

    return Operation(
        name="layer14_temp_clear",
        phase=14.1,
        reads={"OP_LEV", "MARK_PC", "TEMP", "IS_BYTE", "H1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "MARK_AX", "AX_CARRY_HI", "CONST"},
        writes={"TEMP", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer14_temp_clear_ir(),
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def _layer14_clear_addr_key_pollution_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_clear_addr_key_pollution``.

    48 ``gated_write`` rules, one per ADDR_KEY[k] cell. Each rule fires
    at positions that are NOT MEM value bytes AND NOT register markers
    (the latter list mirrors the imperative blockers) by combining a
    positive bias of 0.5 with -100 condition weights on every MEM_VAL_B*
    and MARK_* blocker. The W_down write is -4.0/S to gently cancel the
    ADDR_B*_HI residue that L9 attention leaves on ADDR_KEY-aliased
    cells. Per-rule scope and dominates_at are not declared because the
    rule is an additive defensive clear with no contested writes from
    other ops in the L14 chain.
    """
    suppress_weight = -100.0  # imperative: W_up[..., DIM] = -S * 100
    common_conditions = (
        ("MEM_VAL_B0", suppress_weight),
        ("MEM_VAL_B1", suppress_weight),
        ("MEM_VAL_B2", suppress_weight),
        ("MEM_VAL_B3", suppress_weight),
        ("MARK_PC", suppress_weight),
        ("MARK_BP", suppress_weight),
        ("MARK_AX", suppress_weight),
        ("MARK_STACK0", suppress_weight),
        ("MARK_SP", suppress_weight),
    )
    rules = tuple(
        FFNRule.gated_write(
            name=f"l14_clear_addr_key_pollution_{k}",
            conditions=common_conditions,
            threshold=-0.5,  # imperative: b_up = +S * 0.5 == -S * (-0.5)
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"ADDR_KEY+{k}", -4.0 / S),),
            scope=("not MEM_VAL_B0 and not MEM_VAL_B1 and not MEM_VAL_B2 "
                   "and not MEM_VAL_B3 and not MARK_PC and not MARK_BP "
                   "and not MARK_AX and not MARK_STACK0 and not MARK_SP"),
        )
        for k in range(48)
    )
    return rules


def _layer14_clear_addr_key_pollution_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_clear_addr_key_pollution_rules(S))
    return ir


def make_layer14_clear_addr_key_pollution_op() -> Operation:
    """L14 FFN: Clear ADDR_KEY pollution at non-MEM, non-marker positions.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Second in the chain (phase=14.2).

    Migration (Phase 6 wave 3J): per-unit writes declared via
    :func:`_layer14_clear_addr_key_pollution_rules` and attached as
    ``compiler_ir``; bake lowers through :meth:`CompilerIR.lower_ffn`.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Pinned to chain offset 4 (after layer14_temp_clear consumes 0..3).
        # Byte-identical with the legacy ``_l14_unit_counter`` start.
        start_unit = _l14_chain_alloc("layer14_clear_addr_key_pollution")
        ir = _layer14_clear_addr_key_pollution_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). Runs at phase 14.2 after
    # ``layer14_temp_clear`` which consumes units 0..3, so this op writes
    # units 4..51 (one unit per ADDR_KEY dim, k=0..47).
    _claims = set()
    for k in range(48):
        _claims.add((14, "ffn_W_down", str(4 + k), f"ADDR_KEY+{k}"))

    return Operation(
        name="layer14_clear_addr_key_pollution",
        phase=14.2,
        reads={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "MARK_PC", "MARK_BP", "MARK_AX", "MARK_STACK0", "MARK_SP",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer14_clear_addr_key_pollution_ir(),
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer14_clear_output_corruption_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_clear_output_corruption``.

    Three hidden units: the loop over ``k in [0, 16]`` produces unit 0
    (OUTPUT_LO[0] boost at STACK0 byte rows) and unit 1 (OUTPUT_HI[0]
    boost at STACK0 byte rows), each with the same blocker fan-out but
    with the "preserve already-nonzero nibble" guard pointing at its
    own band. Unit 2 is the JSR STACK0 marker HI[0] reassertion that
    :func:`_disable_l14_stack0_jsr_hi0_default` wipes immediately after
    bake (net W_down delta zero so the verifier observes no claim).

    Each rule uses ``gate="CONST"`` with ``gate_weight=1.0`` and
    ``gate_bias=0.0`` to reproduce the imperative
    ``W_gate[CONST] = 1.0`` / unset-``b_gate`` pair byte-identically.
    """
    PC_I = 0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4
    suppress_weight = -100.0  # imperative: W_up[..., DIM] = -S * 100

    def boost_unit_conditions(k: int, band_dim_name: str) -> tuple:
        cond = [
            (f"H4+{BP_I}", 1.0),
            (f"H1+{BP_I}", -20.0),
            ("MEM_VAL_B0", suppress_weight),
            ("MEM_VAL_B1", suppress_weight),
            ("MEM_VAL_B2", suppress_weight),
            ("MEM_VAL_B3", suppress_weight),
            (f"H1+{MEM_I}", suppress_weight),
            (f"H3+{MEM_I}", suppress_weight),
            ("MARK_PC", suppress_weight),
            ("MARK_AX", suppress_weight),
            ("MARK_SP", suppress_weight),
            ("MARK_BP", suppress_weight),
            ("MARK_MEM", suppress_weight),
            ("MARK_STACK0", suppress_weight),
            (f"H1+{PC_I}", suppress_weight),
            (f"H1+{AX_I}", suppress_weight),
            (f"H1+{SP_I}", suppress_weight),
            ("PSH_AT_SP", suppress_weight),
            ("CMP+3", suppress_weight),
        ]
        # k==0 (OUTPUT_LO band) also blocks ADD/SUB byte rows hard.
        if k == 0:
            cond.append(("TEMP+8", -1e20))
            cond.append(("TEMP+9", -1e20))
        cond.append(("BYTE_INDEX_3", suppress_weight))
        # Preserve already-computed nonzero nibbles.
        for nonzero in range(1, 16):
            cond.append((f"{band_dim_name}+{nonzero}", -2.0))
        return tuple(cond)

    rules = (
        FFNRule.gated_write(
            name="l14_clear_output_corruption_lo0_boost",
            conditions=boost_unit_conditions(0, "OUTPUT_LO"),
            threshold=0.5,  # imperative: b_up = -S * 0.5
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("OUTPUT_LO+0", 50.0 / S),),
        ),
        FFNRule.gated_write(
            name="l14_clear_output_corruption_hi0_boost",
            conditions=boost_unit_conditions(16, "OUTPUT_HI_THIS_STEP"),
            threshold=0.5,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("OUTPUT_HI_THIS_STEP+0", 50.0 / S),),
        ),
        # JSR STACK0 marker HI[0] reassertion. The bake post-pass
        # ``_disable_l14_stack0_jsr_hi0_default`` zeroes this entire unit
        # immediately after the rules lower, so the net residual cell
        # writes are zero. The unit is still allocated to keep the chain
        # offset / claim grid consistent with the legacy bake.
        FFNRule.gated_write(
            name="l14_clear_output_corruption_jsr_stack0_hi0_marker",
            conditions=(
                ("OP_JSR", 0.2),  # imperative: W_up[OP_JSR] = S/5
                ("MARK_STACK0", 1.0),
                ("IS_BYTE", -10.0),
                ("MARK_PC", suppress_weight),
                ("MARK_AX", suppress_weight),
                ("MARK_SP", suppress_weight),
                ("MARK_BP", suppress_weight),
                ("MARK_MEM", suppress_weight),
            ),
            threshold=1.5,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("OUTPUT_HI_THIS_STEP+0", 50.0 / S),),
        ),
    )
    return rules


def _layer14_clear_output_corruption_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_clear_output_corruption_rules(S))
    return ir


def make_layer14_clear_output_corruption_op() -> Operation:
    """L14 FFN: Boost OUTPUT[0] at STACK0 byte positions to fix attention bleed.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Third in the chain (phase=14.3).

    Migration (Phase 6 wave 3J): per-unit writes declared via
    :func:`_layer14_clear_output_corruption_rules` and attached as
    ``compiler_ir``; bake lowers through :meth:`CompilerIR.lower_ffn`
    then runs the boundary guard, the JSR STACK0 HI[0] disable pass
    (which wipes the third allocated unit by design), and the PSH MEM
    high-nibble boost extension.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Pinned to chain offset 52 (predecessors consume units 0..51).
        # Byte-identical with the legacy ``_l14_unit_counter`` start.
        start_unit = _l14_chain_alloc("layer14_clear_output_corruption")
        ir = _layer14_clear_output_corruption_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        _disable_l14_stack0_jsr_hi0_default(ffn, dim_positions, start_unit, next_unit)
        next_unit = _boost_l14_psh_mem_marker_high_nibbles(
            ffn, dim_positions, S, next_unit
        )
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). Runs at phase 14.3 after
    # ``layer14_temp_clear`` (units 0..3) and
    # ``layer14_clear_addr_key_pollution`` (units 4..51).
    #   unit 52: _set_layer14_clear_output_corruption k=0 -> OUTPUT_LO[0]
    #   unit 53: _set_layer14_clear_output_corruption k=16 -> OUTPUT_HI[0]
    #   unit 54: JSR STACK0 marker high-zero default unit (allocated by
    #            ``_set_layer14_clear_output_corruption`` then wiped by
    #            ``_disable_l14_stack0_jsr_hi0_default``; net W_down delta is
    #            zero so the verifier observes no change — intentionally
    #            unclaimed).
    #   units 55..69: _boost_l14_psh_mem_marker_high_nibbles writes one
    #            W_down[OUTPUT_HI_THIS_STEP+nibble] per unit for nibble in 1..15.
    _claims = {
        (14, "ffn_W_down", "52", "OUTPUT_LO+0"),
        (14, "ffn_W_down", "53", "OUTPUT_HI_THIS_STEP+0"),
    }
    for nibble in range(1, 16):
        _claims.add(
            (14, "ffn_W_down", str(54 + nibble), f"OUTPUT_HI_THIS_STEP+{nibble}")
        )

    return Operation(
        name="layer14_clear_output_corruption",
        phase=14.3,
        reads={"H4", "H1", "H3", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "OP_JSR", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "IS_BYTE", "BYTE_INDEX_3", "PSH_AT_SP", "CMP",
               "MEM_STORE", "OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "TEMP", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer14_clear_output_corruption_ir(),
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer14_clear_mem_marker_output_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_clear_mem_marker_output``.

    64 hidden units arranged as four 16-element blocks:
      block 0 (units +0..+15):  OP_JSR + OUTPUT_LO[k]   (k in 0..15)
      block 1 (units +16..+31): OP_JSR + OUTPUT_HI[k]   (k in 0..15)
      block 2 (units +32..+47): OP_ENT + OUTPUT_LO[k]   (k in 0..15)
      block 3 (units +48..+63): OP_ENT + OUTPUT_HI[k]   (k in 0..15)

    Each rule fires only at the MEM marker token when the matching op
    relay is active: positive contributions from OP_* (weight 0.2,
    matching the imperative ``W_up[OP_*] = S/5``) and MARK_MEM (weight
    1.0), strong -10 blockers on IS_BYTE / MARK_PC / MARK_AX / MARK_SP /
    MARK_BP / MARK_STACK0, threshold 1.5 so only the (relay+MEM marker)
    pair fires. The W_down weight is the per-S 76/S offset
    that cancels the L14 attention's -115 corruption per dim.
    """
    OFFSET = 76.0 / S  # imperative: same OFFSET = 76.0 / S
    common_blockers = (
        ("IS_BYTE", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
    )

    def rule_for(op_name: str, band: str, k: int) -> FFNRule:
        return FFNRule.gated_write(
            name=f"l14_clear_mem_marker_output_{op_name.lower()}_{band.lower()}_{k}",
            conditions=(
                (op_name, 0.2),  # imperative: W_up[OP_*] = S / 5
                ("MARK_MEM", 1.0),
                *common_blockers,
            ),
            threshold=1.5,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"{band}+{k}", OFFSET),),
            scope=f"{op_name} and MARK_MEM",
        )

    rules: list[FFNRule] = []
    for op_name in ("OP_JSR", "OP_ENT"):
        for k in range(16):
            rules.append(rule_for(op_name, "OUTPUT_LO", k))
        for k in range(16):
            rules.append(rule_for(op_name, "OUTPUT_HI_THIS_STEP", k))
    return tuple(rules)


def _layer14_clear_mem_marker_output_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_clear_mem_marker_output_rules(S))
    return ir


def make_layer14_clear_mem_marker_output_op() -> Operation:
    """L14 FFN: Clear OUTPUT at MEM marker for OP_JSR/OP_ENT.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops
    (``layer14_temp_clear``, ``layer14_clear_addr_key_pollution``,
    ``layer14_clear_output_corruption``). Last in the chain (phase=14.4).

    Migration (Phase 6 wave 3J): per-unit writes declared via
    :func:`_layer14_clear_mem_marker_output_rules` and attached as
    ``compiler_ir``; bake lowers through :meth:`CompilerIR.lower_ffn`.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Pinned to chain offset 70 (predecessors consume units 0..69).
        # Byte-identical with the legacy ``_l14_unit_counter`` start.
        start_unit = _l14_chain_alloc("layer14_clear_mem_marker_output")
        ir = _layer14_clear_mem_marker_output_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). Runs at phase 14.4 after
    # phases 14.1..14.3 which consume units 0..69. The helper writes 64 units
    # arranged as:
    #   units 70..85:    OP_JSR + OUTPUT_LO[k] for k in 0..15
    #   units 86..101:   OP_JSR + OUTPUT_HI[k] for k in 0..15
    #   units 102..117:  OP_ENT + OUTPUT_LO[k] for k in 0..15
    #   units 118..133:  OP_ENT + OUTPUT_HI[k] for k in 0..15
    _claims = set()
    base = 70
    for op_block in range(2):  # 0 = OP_JSR, 1 = OP_ENT
        for k in range(16):
            _claims.add(
                (14, "ffn_W_down", str(base + op_block * 32 + k),
                 f"OUTPUT_LO+{k}")
            )
            _claims.add(
                (14, "ffn_W_down", str(base + op_block * 32 + 16 + k),
                 f"OUTPUT_HI_THIS_STEP+{k}")
            )

    return Operation(
        name="layer14_clear_mem_marker_output",
        phase=14.4,
        reads={"OP_JSR", "OP_ENT", "MARK_MEM", "IS_BYTE",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer14_clear_mem_marker_output_ir(),
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer14_jsr_ax_bytes_zero_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_jsr_ax_bytes_zero``.

    Mirrors the 4 imperative hidden units: at AX byte positions
    (IS_BYTE + H1[AX]) gated by OP_JSR, each unit spreads -3/S across one
    nibble band (LO or HI) or boosts a single byte-value-0 slot (+5/S on
    OUTPUT_LO[0] / OUTPUT_HI[0]). The byte-value-0 token wins argmax →
    AX bytes 1-3 = 0x00 for the JSR-preserved AX. The W_down write weights
    encode the original ``-3.0 / S`` / ``5.0 / S`` constants directly so
    the lowerer's "no S scaling on writes" contract reproduces the
    imperative helper byte-for-byte.
    """
    AX_I = 1
    common_conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{AX_I}", 1.0),
    )
    common_kwargs = dict(
        conditions=common_conditions,
        threshold=1.5,
        gate="OP_JSR",
        gate_weight=1.0,
        gate_bias=0.0,
        scope="OP_JSR and IS_BYTE and H1+1",
    )
    rules = (
        FFNRule.gated_write(
            name="l14_jsr_ax_bytes_zero_lo_neg",
            writes=tuple((f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_jsr_ax_bytes_zero_hi_neg",
            writes=tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
            ),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_jsr_ax_bytes_zero_lo0_boost",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_jsr_ax_bytes_zero_hi0_boost",
            writes=(("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
            **common_kwargs,
        ),
    )
    return rules


def _layer14_jsr_ax_bytes_zero_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_jsr_ax_bytes_zero_rules(S))
    return ir


def make_layer14_jsr_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when OP_JSR is active.

    FIX 2026-05-12 (fix-jsr-ax-bytes-1-3): Per C4's 8-bit-AX convention, AX
    bytes 1-3 must be 0. For OP_JSR, the L9 ALU JSR-preserve routing writes
    the previous AX byte 0 at MARK_AX (correctly emitting AX byte 0), but
    bytes 1-3 of AX are predicted at AX byte 0/1/2 positions where L14
    attention / L7 head 1 SP gather contaminate OUTPUT with SP/PC bytes.

    This op mirrors ``BinaryOpByteZeroingPostOp``: at AX byte positions
    (IS_BYTE + H1[AX]) when OP_JSR is active, write -3/S to every OUTPUT_LO
    /OUTPUT_HI nibble dim and +5/S to OUTPUT_LO[0]/OUTPUT_HI[0], so the
    byte-value-0 token wins argmax — producing AX bytes 1-3 = 0x00. OP_JSR
    is broadcast to AX byte positions by L7 head 5 (V slot 8, also new in
    this commit).

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Phase 14.6: runs AFTER ``layer14_addr_key_neural_decode`` (14.5).

    Migration (Phase 6 wave 3J): the per-unit writes are now declared via
    :func:`_layer14_jsr_ax_bytes_zero_rules` and attached as ``compiler_ir``;
    the bake lowers the rule list via :meth:`CompilerIR.lower_ffn` at the
    pinned chain offset and then runs the boundary-guard / STACK0-block
    post-passes. Byte-identical with the legacy imperative helper.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Pinned to chain offset 1862 (predecessors through addr_key_neural_decode
        # consume units 0..1861). Byte-identical with the legacy
        # ``_l14_unit_counter`` start.
        start_unit = _l14_chain_alloc("layer14_jsr_ax_bytes_zero")
        ir = _layer14_jsr_ax_bytes_zero_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        _block_l14_jsr_ax_zero_on_stack0_bytes(ffn, dim_positions, S, start_unit, next_unit)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). Runs at phase 14.6 after
    # ``layer14_addr_key_neural_decode`` (14.5) which closes the chain at
    # unit 1862. The helper writes 4 units:
    #   unit 1862: -3/S on OUTPUT_LO[0..15]
    #   unit 1863: -3/S on OUTPUT_HI[0..15]
    #   unit 1864: +5/S on OUTPUT_LO[0]
    #   unit 1865: +5/S on OUTPUT_HI[0]
    _claims = set()
    for k in range(16):
        _claims.add((14, "ffn_W_down", "1862", f"OUTPUT_LO+{k}"))
        _claims.add((14, "ffn_W_down", "1863", f"OUTPUT_HI_THIS_STEP+{k}"))
    _claims.add((14, "ffn_W_down", "1864", "OUTPUT_LO+0"))
    _claims.add((14, "ffn_W_down", "1865", "OUTPUT_HI_THIS_STEP+0"))

    return Operation(
        name="layer14_jsr_ax_bytes_zero",
        phase=14.6,
        reads={"OP_JSR", "IS_BYTE", "H1", "CONST", "STACK0_BYTE0", "STACK0_BYTE1",
               "STACK0_BYTE2", "STACK0_BYTE3"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer14_jsr_ax_bytes_zero_ir(),
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def _layer14_alu_nocarry_ax_bytes_zero_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_alu_nocarry_ax_bytes_zero``.

    Mirrors the 4 imperative hidden units: at AX byte positions 1-3
    (IS_BYTE + H1[AX] + TEMP[7]*4 + NOT BYTE_INDEX_3*4) gated by TEMP[7]
    (the NOCARRY_ALU_OP relay populated by L7 head 5 for AND/OR/XOR/SHR),
    each unit spreads -3/S across one nibble band (LO or HI) or boosts a
    single byte-value-0 slot (+5/S on OUTPUT_LO[0] / OUTPUT_HI[0]). The
    elevated threshold of 5.0 plus the ``TEMP+7`` condition weight of
    4.0 reproduce the imperative ``b_up = -S * 5.0`` and
    ``W_up[TEMP+7] = S * 4`` lines exactly; without TEMP[7] the
    pre-silu drops to -300 (no firing) so byte 0 and non-target opcodes
    are both safe.
    """
    AX_I = 1
    common_conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{AX_I}", 1.0),
        ("TEMP+7", 4.0),
        ("BYTE_INDEX_3", -4.0),
    )
    common_kwargs = dict(
        conditions=common_conditions,
        threshold=5.0,
        gate="TEMP+7",
        gate_weight=1.0,
        gate_bias=0.0,
        scope="TEMP+7 and IS_BYTE and H1+1 and not BYTE_INDEX_3",
    )
    rules = (
        FFNRule.gated_write(
            name="l14_alu_nocarry_ax_bytes_zero_lo_neg",
            writes=tuple((f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_alu_nocarry_ax_bytes_zero_hi_neg",
            writes=tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
            ),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_alu_nocarry_ax_bytes_zero_lo0_boost",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_alu_nocarry_ax_bytes_zero_hi0_boost",
            writes=(("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
            **common_kwargs,
        ),
    )
    return rules


def _layer14_alu_nocarry_ax_bytes_zero_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_alu_nocarry_ax_bytes_zero_rules(S))
    return ir


def make_layer14_alu_nocarry_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when a non-carry ALU
    op is active (V7 Block 13, fix-v7-block13-ax-merge, 2026-05-12).

    Per ``c4_release/docs/V7_HEAP_OPS_NEURAL_PLAN.md`` §2 (AX merge): for
    non-carry ALU ops (AND/OR/XOR/SHR), AX bytes 1-3 must be 0 per C4's
    8-bit-AX-with-32-bit-register convention. The L10
    ``BinaryOpByteZeroingPostOp`` already attempts this clearing but can be
    overridden by downstream contamination at L11-L14 (MUL accumulators,
    L14 mem_addr_gather attention bleed, etc.). This L14 cleanup runs AFTER
    those sources and BEFORE L15 to restore AX bytes 1-3 = 0x00.

    Mirrors ``make_layer14_jsr_ax_bytes_zero_op`` and
    ``make_layer14_lc_ax_bytes_zero_op``: gates on ``TEMP[7]`` (NOCARRY_ALU_OP
    relay = OP_AND | OP_OR | OP_XOR | OP_SHR, supplied by L7 head 5 V slot
    9, also new in this commit) at AX byte positions 1-3, blocks at byte 0
    via ``BYTE_INDEX_0 = -S*4``, and zeros bytes 1-3 by writing -3/S to
    every OUTPUT_LO/HI nibble dim and +5/S to OUTPUT_LO[0] / OUTPUT_HI[0].

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Phase 14.8: runs AFTER ``layer14_lc_ax_bytes_zero`` (14.7) — JSR / LC /
    nocarry-ALU gate on disjoint relays (OP_JSR vs OP_LC_RELAY vs TEMP[7])
    so the relative order within 14.6-14.8 only matters for unit allocation.

    Migration (Phase 6 wave 3J): per-unit writes declared via
    :func:`_layer14_alu_nocarry_ax_bytes_zero_rules` and attached as
    ``compiler_ir``; bake lowers through :meth:`CompilerIR.lower_ffn`.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Pinned to chain offset 1870 (jsr_ax + lc_ax consume units 1862..1869).
        # Last op in the L14 cleanup chain. Byte-identical with the legacy
        # ``_l14_unit_counter`` start.
        start_unit = _l14_chain_alloc("layer14_alu_nocarry_ax_bytes_zero")
        ir = _layer14_alu_nocarry_ax_bytes_zero_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). Runs at phase 14.8 — last
    # op in the L14 cleanup chain. Predecessors leave the counter at 1870.
    # The helper writes 4 units mirroring jsr/lc_ax_bytes_zero:
    #   unit 1870: -3/S on OUTPUT_LO[0..15]
    #   unit 1871: -3/S on OUTPUT_HI[0..15]
    #   unit 1872: +5/S on OUTPUT_LO[0]
    #   unit 1873: +5/S on OUTPUT_HI[0]
    _claims = set()
    for k in range(16):
        _claims.add((14, "ffn_W_down", "1870", f"OUTPUT_LO+{k}"))
        _claims.add((14, "ffn_W_down", "1871", f"OUTPUT_HI_THIS_STEP+{k}"))
    _claims.add((14, "ffn_W_down", "1872", "OUTPUT_LO+0"))
    _claims.add((14, "ffn_W_down", "1873", "OUTPUT_HI_THIS_STEP+0"))

    return Operation(
        name="layer14_alu_nocarry_ax_bytes_zero",
        phase=14.8,
        reads={"TEMP", "IS_BYTE", "H1", "BYTE_INDEX_3", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer14_alu_nocarry_ax_bytes_zero_ir(),
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        claims=_claims,
        # Last op in the L14 FFN chain (``_l14_unit_counter`` reaches 1873
        # after this op runs). The chain is: temp_clear + temp residue clamp
        # + ADD byte-1 high cleanup (4 units) →
        # clear_addr_key_pollution (48) → clear_output_corruption (3) →
        # PSH MEM high-nibble boost (15) → clear_mem_marker_output (64) →
        # addr_key_neural_decode (1728) →
        # jsr_ax_bytes_zero (4) → lc_ax_bytes_zero (4) → this op (4).
        # Annotating only the chain tail with the cumulative max is
        # sufficient — the compiler aggregates per-layer max across all
        # ops, so this single annotation suffices for L14 dynamic sizing.
        ffn_units_used=1874,
        smoke_tests={
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
    )


def _layer14_lc_ax_bytes_zero_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_lc_ax_bytes_zero``.

    Mirrors the 4 imperative hidden units: at AX byte positions 1-3
    (IS_BYTE + H1[AX] + NOT BYTE_INDEX_0 via ``BYTE_INDEX_3 = -S*4``
    blocker) gated by OP_LC_RELAY, each unit spreads -3/S across one
    nibble band (LO or HI) or boosts a single byte-value-0 slot (+5/S on
    OUTPUT_LO[0] / OUTPUT_HI[0]). The ``BYTE_INDEX_3`` blocker is a
    "kill switch": at any byte index where it is 1 the condition sum
    drops by 4S and the SiLU side stops firing, which is why the
    imperative helper labels it as a "Block after AX byte 3" guard.
    """
    AX_I = 1
    common_conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{AX_I}", 1.0),
        ("BYTE_INDEX_3", -4.0),
    )
    common_kwargs = dict(
        conditions=common_conditions,
        threshold=1.5,
        gate="OP_LC_RELAY",
        gate_weight=1.0,
        gate_bias=0.0,
        scope="OP_LC_RELAY and IS_BYTE and H1+1 and not BYTE_INDEX_3",
    )
    rules = (
        FFNRule.gated_write(
            name="l14_lc_ax_bytes_zero_lo_neg",
            writes=tuple((f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_lc_ax_bytes_zero_hi_neg",
            writes=tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
            ),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_lc_ax_bytes_zero_lo0_boost",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
            **common_kwargs,
        ),
        FFNRule.gated_write(
            name="l14_lc_ax_bytes_zero_hi0_boost",
            writes=(("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
            **common_kwargs,
        ),
    )
    return rules


def _layer14_lc_ax_bytes_zero_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_lc_ax_bytes_zero_rules(S))
    return ir


def make_layer14_lc_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when OP_LC is active.

    FIX V7 Phase 7a (LC byte clearing, per V7_HEAP_OPS_NEURAL_PLAN.md):
    Per C4's LC (load-char) semantics, the loaded value is a single byte
    placed in AX byte 0; AX bytes 1-3 must be 0. L15 attention head 0
    writes the resolved memory byte into ``OUTPUT_LO/HI`` at the AX byte 0
    position. Heads 1-3 fire for LI (writing AX bytes 1-3 from the loaded
    word), but for LC they must be suppressed so bytes 1-3 emit 0.

    Mirrors ``make_layer14_jsr_ax_bytes_zero_op``: at AX byte positions
    1-3 (IS_BYTE + H1[AX] + NOT BYTE_INDEX_0) when OP_LC_RELAY is active,
    write -3/S to every OUTPUT_LO/HI nibble dim and +5/S to OUTPUT_LO[0]
    / OUTPUT_HI[0] so the byte-value-0 token wins argmax — producing AX
    bytes 1-3 = 0x00. Byte 0 is protected by a strong BYTE_INDEX_0
    blocker so this op does NOT override the L15 head 0 load result.

    OP_LC_RELAY at AX byte positions is supplied by L7 head 5 (V slot 2,
    already wired in ``_set_layer7_memory_heads``).

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Phase 14.7: runs AFTER ``layer14_jsr_ax_bytes_zero`` (14.6) — the JSR
    and LC ops gate on disjoint relays (OP_JSR vs OP_LC_RELAY) so the
    relative order within phase 14.6-14.7 only matters for unit allocation.

    Migration (Phase 6 wave 3J): per-unit writes declared via
    :func:`_layer14_lc_ax_bytes_zero_rules` and attached as
    ``compiler_ir``; bake lowers through :meth:`CompilerIR.lower_ffn`.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Pinned to chain offset 1866 (jsr_ax_bytes_zero consumes 1862..1865).
        # Byte-identical with the legacy ``_l14_unit_counter`` start.
        start_unit = _l14_chain_alloc("layer14_lc_ax_bytes_zero")
        ir = _layer14_lc_ax_bytes_zero_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). Runs at phase 14.7 after
    # ``layer14_jsr_ax_bytes_zero`` (14.6) consumes units 1862..1865. The
    # helper writes 4 units mirroring jsr_ax_bytes_zero:
    #   unit 1866: -3/S on OUTPUT_LO[0..15]
    #   unit 1867: -3/S on OUTPUT_HI[0..15]
    #   unit 1868: +5/S on OUTPUT_LO[0]
    #   unit 1869: +5/S on OUTPUT_HI[0]
    _claims = set()
    for k in range(16):
        _claims.add((14, "ffn_W_down", "1866", f"OUTPUT_LO+{k}"))
        _claims.add((14, "ffn_W_down", "1867", f"OUTPUT_HI_THIS_STEP+{k}"))
    _claims.add((14, "ffn_W_down", "1868", "OUTPUT_LO+0"))
    _claims.add((14, "ffn_W_down", "1869", "OUTPUT_HI_THIS_STEP+0"))

    return Operation(
        name="layer14_lc_ax_bytes_zero",
        phase=14.7,
        reads={"OP_LC_RELAY", "IS_BYTE", "H1", "BYTE_INDEX_3", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer14_lc_ax_bytes_zero_ir(),
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        claims=_claims,
        smoke_tests={"TestSmokeMemory::test_sc_lc_roundtrip"},
        spec_section="BLOG_SPEC.md#memory",
    )


# ---------------------------------------------------------------------------
# layer14_addr_key_neural_decode -- FFNRule program (5 substages, 1728 units).
#
# Migration (Phase 6 wave 4H): the 5 imperative substages in
# :func:`_bake_addr_key_neural_decode` are declared as ``FFNRule`` programs so
# the compiler IR sees the same per-unit writes the imperative bake produced.
# The substage layout, gate semantics, and per-write constants mirror the
# imperative helper one-for-one; ``compare_symbolic_to_lowered_ffn`` validates
# each substage byte-identically against the lowering contract.
#
# Substage layout (1728 units total):
#   (1) lo+hi nibble decode     : 4 value_gates x 16 hi x 16 lo  = 1024 units
#   (2) common top nibble       : 4 value_gates x 16 b1_lo       =   64 units
#   (3) carry-correction top    : 4 value_gates x carry_los x 16 =   96 units
#   (4) load-query decode       : 2 op_gates x (256 lo+hi + 16   =  544 units
#                                 top), interleaved per op to
#                                 preserve the legacy unit indices
#
# Value-byte gating mirrors the imperative ``value_gates`` table: bytes
# 0/1/2 are selected by ``MEM_VAL_B1/B2/B3`` (the L2 autoregressive flags;
# B1 marks the token predicting MEM value byte 0, etc.), and byte 3 is
# selected by ``H3+4`` with ``H2+4`` blocked. The first three substages
# use the ``W_gate[..., blocker_dim] = -1.0`` blocker pattern (a
# ``gated_write`` with ``gate_weight=-1.0`` / ``gate_bias=1.0``) for the
# byte-3 case and ``W_gate[..., CONST] = 0`` (a ``constant_write`` with
# ``gate_bias=1.0``) for bytes 0/1/2 where ``blocker_dim is None``.
# ---------------------------------------------------------------------------

# (gate_dim_name, blocker_dim_name | None, byte_off)
_ADDR_KEY_VALUE_GATES = (
    ("MEM_VAL_B1", None, 0),
    ("MEM_VAL_B2", None, 1),
    ("MEM_VAL_B3", None, 2),
    ("H3+4",       "H2+4", 3),
)

# (op_gate_name, op_label) for load-query AX substages.
_ADDR_KEY_LOAD_QUERY_OPS = (
    ("OP_LI_RELAY", "li"),
    ("OP_LC_RELAY", "lc"),
)

# Carry-into-byte-1 lo values per byte_off, mirroring ``carry_los`` in the
# imperative helper.
_ADDR_KEY_CARRY_LOS = {
    0: (),
    1: (15,),
    2: (14, 15),
    3: (13, 14, 15),
}


def _addr_key_gate_kwargs(blocker_dim_name):
    """Map (None | str) blocker to FFNRule gate / gate_bias / gate_weight.

    The imperative bake sets ``ffn.b_gate[unit] = 1.0`` unconditionally and,
    when ``blocker_dim is not None``, also ``ffn.W_gate[unit, blocker_dim]
    = -1.0``. In FFNRule land that is either a ``constant_write`` (gate=None,
    gate_bias=1.0) or a ``gated_write`` (gate=blocker, gate_weight=-1.0,
    gate_bias=1.0).
    """
    if blocker_dim_name is None:
        return None
    return dict(
        gate=blocker_dim_name,
        gate_weight=-1.0,
        gate_bias=1.0,
    )


def _addr_key_make_rule(
    *,
    name,
    conditions,
    threshold,
    writes,
    blocker_dim_name,
    scope,
):
    """Build either ``FFNRule.constant_write`` or ``FFNRule.gated_write``
    depending on whether ``blocker_dim_name`` is set.

    ``constant_write`` mirrors the imperative ``b_gate=1.0`` (no W_gate
    writes) form. ``gated_write`` with ``gate_weight=-1.0`` /
    ``gate_bias=1.0`` mirrors the ``W_gate[..., blocker]=-1.0`` blocker
    pattern used when a byte-3 case routes through ``H3+4`` with ``H2+4``
    blocked.
    """
    gate_kwargs = _addr_key_gate_kwargs(blocker_dim_name)
    if gate_kwargs is None:
        return FFNRule.constant_write(
            name=name,
            conditions=conditions,
            threshold=threshold,
            writes=writes,
            scope=scope,
        )
    return FFNRule.gated_write(
        name=name,
        conditions=conditions,
        threshold=threshold,
        writes=writes,
        scope=scope,
        **gate_kwargs,
    )


def _layer14_addr_key_neural_decode_lo_hi_rules(
    S: float,
) -> tuple[FFNRule, ...]:
    """Substage 1: lo+hi nibble decode (1024 rules / 4*16*16 units).

    Mirrors the first imperative loop in :func:`_bake_addr_key_neural_decode`:
    for each ``(value_gate, hi, lo)`` combination, emit one FFN unit with a
    3-way AND ``gate_dim + ADDR_B0_LO[lo] + ADDR_B0_HI[hi] >= 2.5`` and
    write ``2/S`` to ``ADDR_KEY[new_lo]`` and ``ADDR_KEY[16+new_hi]`` where
    ``new_lo = (((hi << 4) | lo) + byte_off) & 0xF`` and
    ``new_hi = (((hi << 4) | lo) + byte_off) >> 4 & 0xF``.

    Imperative writes per unit::

        ffn.W_up[unit, gate_dim]            = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B0_LO + lo]     = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B0_HI + hi]     = S          # condition weight 1.0
        ffn.b_up[unit]                      = -S * 2.5   # threshold = 2.5
        ffn.b_gate[unit]                    = 1.0        # constant gate
        if blocker_dim is not None:
            ffn.W_gate[unit, blocker_dim]   = -1.0       # H2+4 blocker for byte 3
        ffn.W_down[ADDR_KEY + new_lo, unit]      = 2.0 / S
        ffn.W_down[ADDR_KEY + 16 + new_hi, unit] = 2.0 / S
    """
    rules: list[FFNRule] = []
    for gate_dim_name, blocker_dim_name, byte_off in _ADDR_KEY_VALUE_GATES:
        for hi in range(16):
            for lo in range(16):
                byte_addr = ((hi << 4) | lo) + byte_off
                new_lo = byte_addr & 0xF
                new_hi = (byte_addr >> 4) & 0xF
                rules.append(_addr_key_make_rule(
                    name=(
                        f"l14_addr_key_lohi_off{byte_off}_hi{hi:x}_lo{lo:x}"
                    ),
                    conditions=(
                        (gate_dim_name, 1.0),
                        (f"ADDR_B0_LO+{lo}", 1.0),
                        (f"ADDR_B0_HI+{hi}", 1.0),
                    ),
                    threshold=2.5,
                    writes=(
                        (f"ADDR_KEY+{new_lo}", 2.0 / S),
                        (f"ADDR_KEY+{16 + new_hi}", 2.0 / S),
                    ),
                    blocker_dim_name=blocker_dim_name,
                    scope=None,
                ))
    return tuple(rules)


def _layer14_addr_key_neural_decode_top_common_rules(
    S: float,
) -> tuple[FFNRule, ...]:
    """Substage 2: common-case top nibble (64 rules / 4*16 units).

    Mirrors the second imperative loop: emit one FFN unit per
    ``(value_gate, b1_lo)`` that writes ``ADDR_B1_LO[b1_lo]`` straight into
    ``ADDR_KEY[32 + b1_lo]`` whenever the value-byte gate fires and
    ``ADDR_B1_LO[b1_lo]`` is hot (a 2-way AND with threshold 1.5).

    Imperative writes per unit::

        ffn.W_up[unit, gate_dim]            = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B1_LO + b1_lo]  = S          # condition weight 1.0
        ffn.b_up[unit]                      = -S * 1.5   # threshold = 1.5
        ffn.b_gate[unit]                    = 1.0        # constant gate
        if blocker_dim is not None:
            ffn.W_gate[unit, blocker_dim]   = -1.0       # H2+4 blocker for byte 3
        ffn.W_down[ADDR_KEY + 32 + b1_lo, unit] = 2.0 / S
    """
    rules: list[FFNRule] = []
    for gate_dim_name, blocker_dim_name, byte_off in _ADDR_KEY_VALUE_GATES:
        for b1_lo in range(16):
            rules.append(_addr_key_make_rule(
                name=(
                    f"l14_addr_key_top_common_off{byte_off}_b1lo{b1_lo:x}"
                ),
                conditions=(
                    (gate_dim_name, 1.0),
                    (f"ADDR_B1_LO+{b1_lo}", 1.0),
                ),
                threshold=1.5,
                writes=(
                    (f"ADDR_KEY+{32 + b1_lo}", 2.0 / S),
                ),
                blocker_dim_name=blocker_dim_name,
                scope=None,
            ))
    return tuple(rules)


def _layer14_addr_key_neural_decode_top_carry_rules(
    S: float,
) -> tuple[FFNRule, ...]:
    """Substage 3: carry-correction top nibble (96 rules total).

    Mirrors the third imperative loop: when ``hi == 15`` and
    ``lo + byte_off >= 16`` the high-byte adder carries, so the top nibble
    must be ``b1_lo + 1`` instead of ``b1_lo``. Each FFN unit cancels the
    substage-2 ``+2/S`` at ``ADDR_KEY[32+b1_lo]`` and adds ``+2/S`` at
    ``ADDR_KEY[32+((b1_lo+1)&0xF)]``. The trigger is a 4-way AND:
    ``gate_dim + ADDR_B0_HI[15] + ADDR_B0_LO[lo] + ADDR_B1_LO[b1_lo] >= 3.5``.

    Imperative writes per unit::

        ffn.W_up[unit, gate_dim]            = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B0_HI + 15]     = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B0_LO + lo]     = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B1_LO + b1_lo]  = S          # condition weight 1.0
        ffn.b_up[unit]                      = -S * 3.5   # threshold = 3.5
        ffn.b_gate[unit]                    = 1.0        # constant gate
        if blocker_dim is not None:
            ffn.W_gate[unit, blocker_dim]   = -1.0       # H2+4 blocker for byte 3
        ffn.W_down[ADDR_KEY + 32 + b1_lo, unit]              = -2.0 / S
        ffn.W_down[ADDR_KEY + 32 + ((b1_lo + 1) & 0xF), unit] =  2.0 / S
    """
    rules: list[FFNRule] = []
    for gate_dim_name, blocker_dim_name, byte_off in _ADDR_KEY_VALUE_GATES:
        for lo in _ADDR_KEY_CARRY_LOS[byte_off]:
            for b1_lo in range(16):
                top_next = (b1_lo + 1) & 0xF
                rules.append(_addr_key_make_rule(
                    name=(
                        f"l14_addr_key_top_carry_off{byte_off}"
                        f"_lo{lo:x}_b1lo{b1_lo:x}"
                    ),
                    conditions=(
                        (gate_dim_name, 1.0),
                        ("ADDR_B0_HI+15", 1.0),
                        (f"ADDR_B0_LO+{lo}", 1.0),
                        (f"ADDR_B1_LO+{b1_lo}", 1.0),
                    ),
                    threshold=3.5,
                    writes=(
                        (f"ADDR_KEY+{32 + b1_lo}", -2.0 / S),
                        (f"ADDR_KEY+{32 + top_next}",  2.0 / S),
                    ),
                    blocker_dim_name=blocker_dim_name,
                    scope=None,
                ))
    return tuple(rules)


def _layer14_addr_key_neural_decode_load_query_rules(
    S: float,
) -> tuple[FFNRule, ...]:
    """Substages 4+5: load-query AX-marker decode (544 rules).

    Mirrors the fourth+fifth imperative loops, which interleave per
    op_gate: for each of ``(OP_LI_RELAY, OP_LC_RELAY)`` the imperative
    first emits 256 lo+hi units, then 16 top-nibble units, before moving
    to the next op. The per-op block layout therefore is::

        unit  0..255 : OP_LI_RELAY * 16 hi * 16 lo  (lo+hi nibbles)
        unit 256..271: OP_LI_RELAY * 16 b1_lo       (top nibble)
        unit 272..527: OP_LC_RELAY * 16 hi * 16 lo  (lo+hi nibbles)
        unit 528..543: OP_LC_RELAY * 16 b1_lo       (top nibble)

    No blocker is used in either sub-block; the gate is the constant
    ``b_gate=1.0`` form (``constant_write``).

    Per-unit imperative writes (lo+hi sub-block)::

        ffn.W_up[unit, op_gate]              = S          # condition weight 1.0
        ffn.W_up[unit, MARK_AX]              = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B0_LO + lo]      = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B0_HI + hi]      = S          # condition weight 1.0
        ffn.b_up[unit]                       = -S * 3.5   # threshold = 3.5
        ffn.b_gate[unit]                     = 1.0        # constant gate
        ffn.W_down[ADDR_KEY + lo, unit]      = 2.0 / S
        ffn.W_down[ADDR_KEY + 16 + hi, unit] = 2.0 / S

    Per-unit imperative writes (top-nibble sub-block)::

        ffn.W_up[unit, op_gate]              = S          # condition weight 1.0
        ffn.W_up[unit, MARK_AX]              = S          # condition weight 1.0
        ffn.W_up[unit, ADDR_B1_LO + b1_lo]   = S          # condition weight 1.0
        ffn.b_up[unit]                       = -S * 2.5   # threshold = 2.5
        ffn.b_gate[unit]                     = 1.0        # constant gate
        ffn.W_down[ADDR_KEY + 32 + b1_lo, unit] = 2.0 / S
        if b1_lo != 0:
            ffn.W_down[ADDR_KEY + 32, unit] = -2.0 / S
    """
    rules: list[FFNRule] = []
    for op_gate_name, op_label in _ADDR_KEY_LOAD_QUERY_OPS:
        # --- (a) lo+hi nibble units (256 per op) ---
        for hi in range(16):
            for lo in range(16):
                rules.append(_addr_key_make_rule(
                    name=(
                        f"l14_addr_key_lq_lohi_{op_label}"
                        f"_hi{hi:x}_lo{lo:x}"
                    ),
                    conditions=(
                        (op_gate_name, 1.0),
                        ("MARK_AX", 1.0),
                        (f"ADDR_B0_LO+{lo}", 1.0),
                        (f"ADDR_B0_HI+{hi}", 1.0),
                    ),
                    threshold=3.5,
                    writes=(
                        (f"ADDR_KEY+{lo}", 2.0 / S),
                        (f"ADDR_KEY+{16 + hi}", 2.0 / S),
                    ),
                    blocker_dim_name=None,
                    scope=None,
                ))
        # --- (b) top nibble units (16 per op) ---
        for b1_lo in range(16):
            writes: list[tuple[str, float]] = [
                (f"ADDR_KEY+{32 + b1_lo}", 2.0 / S),
            ]
            if b1_lo != 0:
                writes.append(("ADDR_KEY+32", -2.0 / S))
            rules.append(_addr_key_make_rule(
                name=(
                    f"l14_addr_key_lq_top_{op_label}_b1lo{b1_lo:x}"
                ),
                conditions=(
                    (op_gate_name, 1.0),
                    ("MARK_AX", 1.0),
                    (f"ADDR_B1_LO+{b1_lo}", 1.0),
                ),
                threshold=2.5,
                writes=tuple(writes),
                blocker_dim_name=None,
                scope=None,
            ))
    return tuple(rules)


def _layer14_addr_key_neural_decode_rules(
    S: float,
) -> tuple[FFNRule, ...]:
    """All five substages chained into one 1728-rule FFN program.

    Order mirrors the imperative substage order in
    :func:`_bake_addr_key_neural_decode` exactly so the unit indices line
    up cell-for-cell with the legacy bake (validated by
    ``compare_symbolic_to_lowered_ffn``).
    """
    return (
        *_layer14_addr_key_neural_decode_lo_hi_rules(S),
        *_layer14_addr_key_neural_decode_top_common_rules(S),
        *_layer14_addr_key_neural_decode_top_carry_rules(S),
        *_layer14_addr_key_neural_decode_load_query_rules(S),
    )


def _layer14_addr_key_neural_decode_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_addr_key_neural_decode_rules(S))
    return ir


def _bake_addr_key_neural_decode(ffn, dim_positions, S, start_unit=0):
    """Bake the BLOG_SPEC.md:830 ADDR_KEY nibble decode into ``ffn``.

    Computes the same per-val-byte ADDR_KEY one-hot writes that
    ``NeuralVMEmbedding._inject_mem_metadata`` produces today, but via
    a baked FFN reading the addr byte values that L13's
    ``_set_layer13_mem_addr_gather`` has already gathered into
    ``ADDR_B0_LO``/``ADDR_B0_HI``/``ADDR_B1_LO`` at MEM val byte
    positions.

    Per val byte position (gated by the same source-position flags L15
    reads: MEM_VAL_B1/B2/B3 for value bytes 0/1/2, and H3[MEM] with
    H2[MEM] blocked for value byte 3):
      Let addr_b0 = (hi << 4 | lo) be the byte 0 of the MEM section's
      address.  Let addr_b1_lo be the low nibble of byte 1.  Then
      ``byte_addr = addr_b0 + byte_off`` for byte_off ∈ {0,1,2,3}.

      ADDR_KEY[byte_addr & 0xF]              = 1.0  (lo nibble)
      ADDR_KEY[16 + (byte_addr >> 4) & 0xF]  = 1.0  (hi nibble)
      ADDR_KEY[32 + (addr_b1 + carry) & 0xF] = 1.0  (top nibble)

      where carry = 1 iff (lo + byte_off) >= 16 and the hi nibble's
      addition (hi + carry_from_lo) overflowed past 15.  In the
      common case (byte_off ≤ 3, hi nibble < 15), the top nibble is
      just ``addr_b1_lo``; the high-byte carry case only occurs when
      hi==15 AND lo+byte_off >= 16, which is rare.

    Encoding strategy: enumerate the 16×16×4 combinations of
    (addr_b0_lo, addr_b0_hi, byte_off).  For each, compute byte_addr
    and emit two FFN units:
      - one writing the lo+hi nibbles of byte_addr into ADDR_KEY[0..31]
      - one writing the top nibble (addr_b1_lo+carry) into
        ADDR_KEY[32..47]

    Both units use a 3-way AND in the silu path:
      value-byte-gate + ADDR_B0_LO[lo] + ADDR_B0_HI[hi] >= 3
    with threshold ``-S*2.5`` so only all-3-match fires.

    The top-nibble unit additionally reads ADDR_B1_LO (a 4-way AND with
    threshold ``-S*3.5``) to source the high byte.  The carry case
    (rare) is handled by selecting addr_b1_lo+1 instead of addr_b1_lo
    when (lo + byte_off >= 16) AND (hi == 15).

    Returns ``next_unit`` so callers can chain.
    """
    BD = _as_setdim_proxy(dim_positions)
    unit = start_unit
    MEM_I = 4
    value_gates = [
        # The L2 MEM_VAL flags are autoregressive: B1 marks the token that
        # predicts MEM value byte 0, B2 marks byte 1, and B3 marks byte 2.
        # Value byte 3 is selected elsewhere with H3[MEM] and not H2[MEM].
        (BD.MEM_VAL_B1, None, 0),
        (BD.MEM_VAL_B2, None, 1),
        (BD.MEM_VAL_B3, None, 2),
        (BD.H3 + MEM_I, BD.H2 + MEM_I, 3),
    ]

    # Lo + Hi nibble units (write ADDR_KEY[0..31]): one unit per
    # (value gate, hi, lo) combination, 4*16*16 = 1024 units.
    for gate_dim, blocker_dim, byte_off in value_gates:
        for hi in range(16):
            for lo in range(16):
                byte_addr = ((hi << 4) | lo) + byte_off
                new_lo = byte_addr & 0xF
                new_hi = (byte_addr >> 4) & 0xF
                ffn.W_up[unit, gate_dim] = S
                ffn.W_up[unit, BD.ADDR_B0_LO + lo] = S
                ffn.W_up[unit, BD.ADDR_B0_HI + hi] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.b_gate[unit] = 1.0
                if blocker_dim is not None:
                    ffn.W_gate[unit, blocker_dim] = -1.0
                ffn.W_down[BD.ADDR_KEY + new_lo, unit] = 2.0 / S
                ffn.W_down[BD.ADDR_KEY + 16 + new_hi, unit] = 2.0 / S
                unit += 1

    # --- Top nibble units (write ADDR_KEY[32..47]) ---
    # For each (lo, hi, byte_off, b1_lo): top = (b1_lo + carry) & 0xF
    # where carry = 1 iff (byte_addr_full = ((hi<<4|lo) + byte_off) >> 8) > 0.
    # That carry-into-byte-1 happens only when hi==15 AND lo+byte_off >= 16
    # for byte_off ≤ 3. For correctness we still iterate the full lookup.
    # 16 * 16 * 4 * 16 = 16384 units — too many. We exploit the fact that
    # the high carry condition only depends on (hi, lo, byte_off), so we
    # split into:
    #   (a) common case (no carry): always emit ADDR_B1_LO[b1_lo] →
    #       ADDR_KEY+32+b1_lo. This is independent of addr_b0 / byte_off
    #       and just needs MEM_VAL_B{0..3} to gate the val-byte position.
    #   (b) carry case (hi==15, lo+byte_off>=16): emit ADDR_KEY+32+((b1_lo+1)&0xF)
    #       AND subtract the (a) unit's contribution. This is the
    #       (hi==15, lo, byte_off, b1_lo) combo — bounded by 16*4*16 = 1024.

    # (a) Common case: 4 byte_off × 16 b1_lo = 64 units.
    for gate_dim, blocker_dim, byte_off in value_gates:
        for b1_lo in range(16):
            ffn.W_up[unit, gate_dim] = S
            ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
            ffn.b_up[unit] = -S * 1.5
            ffn.b_gate[unit] = 1.0
            if blocker_dim is not None:
                ffn.W_gate[unit, blocker_dim] = -1.0
            ffn.W_down[BD.ADDR_KEY + 32 + b1_lo, unit] = 2.0 / S
            unit += 1

    # (b) Carry-correction: when hi==15 AND (lo + byte_off) >= 16:
    #     - subtract from ADDR_KEY+32+b1_lo (cancel (a))
    #     - add to ADDR_KEY+32+((b1_lo+1)&0xF)
    # The trigger is a 4-way AND: MEM_VAL_B{byte_off} + ADDR_B0_HI[15] +
    # ADDR_B0_LO[lo] + ADDR_B1_LO[b1_lo], for lo such that lo+byte_off>=16.
    # That set of lo values is: byte_off=0 → none; byte_off=1 → {15};
    # byte_off=2 → {14,15}; byte_off=3 → {13,14,15}.
    # 4 byte_off × variable × 16 b1_lo = 0+1+2+3 = 6 × 16 = 96 units.
    carry_los = {
        0: [],
        1: [15],
        2: [14, 15],
        3: [13, 14, 15],
    }
    for gate_dim, blocker_dim, byte_off in value_gates:
        for lo in carry_los[byte_off]:
            for b1_lo in range(16):
                ffn.W_up[unit, gate_dim] = S
                ffn.W_up[unit, BD.ADDR_B0_HI + 15] = S
                ffn.W_up[unit, BD.ADDR_B0_LO + lo] = S
                ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
                ffn.b_up[unit] = -S * 3.5
                ffn.b_gate[unit] = 1.0
                if blocker_dim is not None:
                    ffn.W_gate[unit, blocker_dim] = -1.0
                # Cancel the common-case write at b1_lo, add at (b1_lo+1)&0xF.
                ffn.W_down[BD.ADDR_KEY + 32 + b1_lo, unit] = -2.0 / S
                ffn.W_down[BD.ADDR_KEY + 32 + ((b1_lo + 1) & 0xF), unit] = 2.0 / S
                unit += 1

    # Load queries at the AX marker also carry a proven address in ADDR_B*
    # lanes, but no Python ADDR_KEY injection touches that marker. Materialize
    # byte_off=0 for LI/LC so L15's query-side ADDR_KEY match is exact.
    for op_gate in (BD.OP_LI_RELAY, BD.OP_LC_RELAY):
        for hi in range(16):
            for lo in range(16):
                ffn.W_up[unit, op_gate] = S
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ADDR_B0_LO + lo] = S
                ffn.W_up[unit, BD.ADDR_B0_HI + hi] = S
                ffn.b_up[unit] = -S * 3.5
                ffn.b_gate[unit] = 1.0
                ffn.W_down[BD.ADDR_KEY + lo, unit] = 2.0 / S
                ffn.W_down[BD.ADDR_KEY + 16 + hi, unit] = 2.0 / S
                unit += 1
        for b1_lo in range(16):
            ffn.W_up[unit, op_gate] = S
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.b_gate[unit] = 1.0
            ffn.W_down[BD.ADDR_KEY + 32 + b1_lo, unit] = 2.0 / S
            if b1_lo != 0:
                ffn.W_down[BD.ADDR_KEY + 32, unit] = -2.0 / S
            unit += 1

    return unit


def make_layer14_addr_key_neural_decode_op(enable: bool = False) -> Operation:
    """L14 FFN: BLOG_SPEC.md:830 neural ADDR_KEY decode at MEM val byte positions.

    Phase 0 of the V2 ADDR_KEY migration (see
    ``docs/V2_ADDR_KEY_NEURAL_DECODE_PLAN.md``).  Computes the same
    per-val-byte ADDR_KEY one-hot encoding that
    ``NeuralVMEmbedding._inject_mem_metadata`` produces today, baked
    into FFN weights instead of a Python loop.

    Per BLOG_SPEC.md:830:

        First the simple case key value retrieval, we want an exact
        match for the key so we take the binary key break it up into
        bytes or nibbles, perform an equality check for each byte or
        nibble, then in a subsequent layer we perform a logical AND
        over those results.

    The "binary key" here is the MEM section's 32-bit address.  L13's
    ``_set_layer13_mem_addr_gather`` already gathers the first 3 addr
    bytes to MEM val byte positions as 4-bit one-hot nibbles
    (``ADDR_B{0,1,2}_LO/HI``).  This op takes those nibbles plus the
    value-byte flags used by L15 and emits the
    ``ADDR_KEY[lo, 16+hi, 32+top]`` one-hot encoding of
    ``byte_addr = addr + byte_off`` (with carry handling).

    Gating: ``enable=False`` (default) → bake is a no-op.  Existing
    tests stay byte-identical.  When flipped to True (in a follow-up
    PR), the bake must produce identical ADDR_KEY values to
    ``_inject_mem_metadata`` on all (addr_b0_lo, addr_b0_hi,
    addr_b1_lo, byte_off) inputs.

    Pinned to ``layer_idx=14`` via ``kind="block"``.  Shares the FFN
    unit counter ``block.ffn._l14_unit_counter`` with the other L14
    cleanup ops.  Phase 14.5: runs AFTER
    ``layer14_clear_addr_key_pollution`` (phase 14.2) so the decode
    output is authoritative (not pre-cleared away).

    Total FFN units consumed when enabled: ~1728
    (1024 lo+hi + 64 common-top + 96 carry-top + 544 load-query decode).
    When disabled, 0.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return
        ffn = block.ffn
        # Pinned to chain offset 134 (cleanup chain through 14.1..14.4
        # consumes units 0..133). Byte-identical with the legacy
        # ``_l14_unit_counter`` start.
        start_unit = _l14_chain_alloc("layer14_addr_key_neural_decode")
        ir = _layer14_addr_key_neural_decode_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(ffn, dim_positions, S, start_unit, next_unit)
        ffn._l14_unit_counter = next_unit

    # ``compiler_ir`` is attached only when ``enable=True``; when
    # disabled the bake is an explicit no-op and a non-empty
    # ``compiler_ir`` would invite the declarations-only dispatcher to
    # lower the rules at unit 0 outside the chain.
    op_compiler_ir = _layer14_addr_key_neural_decode_ir() if enable else None

    return Operation(
        name="layer14_addr_key_neural_decode",
        phase=14.5,
        reads={"MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "OP_LI_RELAY", "OP_LC_RELAY", "MARK_AX",
               "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=op_compiler_ir,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )
