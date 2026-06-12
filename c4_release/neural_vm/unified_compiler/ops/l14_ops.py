"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule
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
    """Build a per-bake :class:`AttentionHeadAllocator` for the L14 heads.

    Phase 7.B: ``pin=`` has been dropped. :data:`_L14_HEAD_LAYOUT` lists
    the heads in declaration order ``(head_0..head_7)`` and the per-bake
    allocator starts empty, so first-fit deterministically lands each
    entry at the same ``head_idx`` the legacy pin claimed (0..7). The
    layout table remains the audited source of truth for downstream
    :data:`_L14_HEAD_LAYOUT_BY_NAME` lookups -- the allocator itself is
    bookkeeping for collision detection, not the source of ``head_idx``
    for the bake. Drop-of-pins is byte-identical because the legacy
    ``_set_layer14_mem_generation`` helper iterated ``h=0..7`` with
    ``base = h * HD``; first-fit on an empty per-layer pool reproduces
    that exact sweep.
    """
    allocator = AttentionHeadAllocator()
    for name, _expected_head_idx in _L14_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=14)
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
    # ``pin=None`` means auto-fit: the allocator picks the first free gap
    # past the previously-claimed chain ops. Phase 7.B.5 drops all
    # static pins on the cleanup chain (the addr_key_neural_decode
    # 1728-unit substage and every 4..64-unit cleanup before/after it).
    # ``_l14_chain_alloc`` pre-claims every preceding entry in declaration
    # order, so first-fit on a 4096-wide pool deterministically lands each
    # op at the same offset the legacy pin specified — byte-identical
    # because none of the rule families cross-reference the unit index.
    # Phase 6 Wave 6D landed the same auto-fit treatment on
    # ``layer14_jsr_ax_bytes_zero`` first as the demo (commit fd38b6e).
    "layer14_temp_clear":                    (None,    4),
    "layer14_clear_addr_key_pollution":      (None,   48),
    "layer14_clear_output_corruption":       (None,   18),
    "layer14_clear_mem_marker_output":       (None,   64),
    # var-cluster follow-up (2026-06-05): 8 units that cancel the L3
    # ``mem_byte_0_default`` +0.940 baseline at SI/SC store positions
    # (MEM_ADDR_SRC=1). Runs at phase 14.45, after clear_mem_marker_output
    # (14.4) and before addr_key_neural_decode (14.5). See
    # ``make_layer14_mem_addr_src_default_suppress_op``.
    "layer14_mem_addr_src_default_suppress": (None,    8),
    # var-cluster follow-up sibling (2026-06-06): cancel the L3
    # ``mem_byte_0_default`` +0.940 baseline at PSH/JSR/ENT store positions
    # (MEM_STORE=1 AND MEM_ADDR_SRC=0). The f4f9103d SI/SC cancel does not
    # fire on JSR step 0 (where addr=SP=0xFFFC), so the var-cluster
    # failures (var_simple_0, if_var_0, var_three_0) persisted.
    # NARROWED 2026-06-06 (8 → 4 units): the initial 8-unit helper covered
    # BYTE_INDEX_{0,1,2} (MEM_addr1/2/3) but over-cancelled at MEM_addr2/3
    # where the L3 baseline of 0x00 is correct for 16-bit addresses. The
    # narrowed scope keeps only the MEM marker (predicting addr_b0) and
    # BYTE_INDEX_0 (predicting MEM_addr1, the originally-failing slot).
    # Runs at phase 14.46, after mem_addr_src_default_suppress (14.45) and
    # before addr_key_neural_decode (14.5). See
    # ``make_layer14_jsr_mem_default_suppress_op``.
    "layer14_jsr_mem_default_suppress":      (None,    4),
    "layer14_addr_key_neural_decode":        (None, 1728),
    "layer14_jsr_ax_bytes_zero":             (None,    4),
    "layer14_lc_ax_bytes_zero":              (None,    4),
    "layer14_alu_nocarry_ax_bytes_zero":     (None,    4),
    # ENT AX bytes 1-3 zeroing (Wave 1 Cluster B1, 2026-06-07): per C4's
    # 8-bit-AX-with-32-bit-register convention, AX bytes 1-3 must be 0
    # at every step. At ENT step 0, the model leaks SP byte 0 (0xE8)
    # into AX bytes 1-3, producing AX=0xE8E8E800 and breaking
    # ``test_lea_basic`` (program ``ENT, IMM 0, LEA 2, EXIT`` halts with
    # exit_code=0 instead of the LEA-computed non-zero address). Mirrors
    # ``layer14_jsr_ax_bytes_zero`` but gates on ``OP_ENT`` (relayed by
    # L7 head 7 V slot 4) instead of ``OP_JSR``. Runs after
    # ``layer14_alu_nocarry_ax_bytes_zero`` in the cleanup chain.
    "layer14_ent_ax_bytes_zero":             (None,    4),
    # Phase 6 Wave 7 demo: pure-declaration corrective op. The op's single
    # rule is byte-identically a no-op on the live corpus -- it carries the
    # impossible condition ``CONST=-100`` so SiLU collapses to 0 and the
    # OUTPUT residual is unchanged -- so its sole purpose is to prove the
    # declare-only flow end to end. See ``make_layer14_demo_phase6_wave7_op``.
    "layer14_demo_phase6_wave7":             (None,    1),
}


def _l14_chain_alloc(op_name: str) -> int:
    """Return the start unit for ``op_name`` in the L14 cleanup chain.

    Looks up ``op_name`` in :data:`_L14_CLEANUP_CHAIN_LAYOUT` and routes
    every prior chain entry (in declaration order) through a fresh
    :class:`FFNUnitAllocator` so the target op's own claim sees the same
    occupied pool a runtime allocator would. Two placement modes are
    supported:

    * **Pinned** — the layout entry has an explicit integer ``pin``.
      ``alloc`` claims exactly that range; collisions raise
      :class:`FFNUnitAllocatorError`. This is the byte-identity
      guarantee path used by every legacy chain op.
    * **Auto-fit** (``pin=None``) — the layout entry leaves placement
      to the allocator. ``alloc`` walks the first-fit pool and returns
      the lowest free gap large enough for ``n_units``. Wave 6D uses
      this on :data:`layer14_jsr_ax_bytes_zero` to demonstrate the
      "drop the offset, let the allocator pick" workflow targeted by
      Phase 6 Wave 6D and the docs/PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md
      auto-fit story.

    Because the chain bakes one op at a time (each op's bake_fn is its
    own call), the allocator is rebuilt per call. Pre-claiming every
    preceding layout entry is what makes the auto-fit deterministic:
    the target op always sees the same occupied pool a stateful runtime
    allocator would after replaying the chain in declaration order.
    """

    allocator = FFNUnitAllocator()
    target_entry: tuple[int | None, int] | None = None
    for name, (pin, n_units) in _L14_CLEANUP_CHAIN_LAYOUT.items():
        if name == op_name:
            target_entry = (pin, n_units)
            break
        # Pre-claim prior chain entries so the target's pin/auto-fit
        # check sees the same occupied pool a stateful allocator would.
        # Prior entries are always pinned in the current layout; if a
        # future entry becomes auto-fit, replay it the same way (the
        # allocator's first-fit pick is deterministic).
        if pin is None:
            allocator.alloc(name, n_units)
        else:
            allocator.alloc(name, n_units, pin=pin)
    if target_entry is None:
        raise KeyError(op_name)
    pin, n_units = target_entry
    if pin is None:
        start, _end = allocator.alloc(op_name, n_units)
    else:
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

        # Wave 1 A3 migration: heads 1/2/3 attend to STACK0 byte h row during
        # SI/SC. The L10 ``layer10_psh_ax_broadcast`` (slots 8/9/10) populates
        # ``STACK0_BYTE_VAL_{h}_{LO,HI}`` at that row during the preceding
        # PSH step. Read those dims into V slots 32+kk (LO) / 48+kk (HI) so
        # the SI/SC addr byte h prediction picks up the pushed value. The
        # broadcast dims are zero on non-STACK0-byte-h rows, so the PSH
        # source path (slot 1 -> SP byte rows) is unaffected. HD=64, so
        # slot 32..47 (LO) and slot 48..63 (HI) fit within head_dim.
        if h in (1, 2, 3):
            val_lo_dim = getattr(BD, f"STACK0_BYTE_VAL_{h}_LO", None)
            val_hi_dim = getattr(BD, f"STACK0_BYTE_VAL_{h}_HI", None)
            if val_lo_dim is not None and val_hi_dim is not None:
                for kk in range(16):
                    v.append(AP(32 + kk, val_lo_dim + kk, 1.0))
                    v.append(AP(48 + kk, val_hi_dim + kk, 1.0))

        # O: write to OUTPUT_LO/HI + cancel L3 default at byte 0.
        o: list[AO] = []
        o.append(AO(BD.OUTPUT_LO + 0, 0, -1.0))
        o.append(AO(BD.OUTPUT_HI + 0, 0, -1.0))
        for kk in range(16):
            o.append(AO(BD.OUTPUT_LO + kk, 1  + kk, 1.0))
            o.append(AO(BD.OUTPUT_HI + kk, 17 + kk, 1.0))

        # Wave 1 A3 migration: matching O writes for the STACK0_BYTE_VAL_h
        # reads at V slots 32+kk / 48+kk -> OUTPUT_LO/HI nibbles.
        if h in (1, 2, 3):
            val_lo_dim = getattr(BD, f"STACK0_BYTE_VAL_{h}_LO", None)
            val_hi_dim = getattr(BD, f"STACK0_BYTE_VAL_{h}_HI", None)
            if val_lo_dim is not None and val_hi_dim is not None:
                for kk in range(16):
                    o.append(AO(BD.OUTPUT_LO + kk, 32 + kk, 1.0))
                    o.append(AO(BD.OUTPUT_HI + kk, 48 + kk, 1.0))

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
            # JSR step-0 guard: head 5 (h=1, MEM_value byte-1) slot-2 K
            # selectors are BP-frame byte-1 markers (``H2+BP_I``,
            # ``L1H4+BP_I``) which are absent on the function-prologue JSR
            # (no prior BP frame, HAS_SE=0). Without this guard the K side
            # degenerates and softmax routes CLEAN_EMBED 0xff bits into
            # MEM_value byte 1, breaking var_*/if_var_*/var_mul_* clusters.
            # See docs/VAR_MUL_ATTRIBUTION_2026_06_09.md and
            # docs/VAR_L3_SP_BYTE2_2026_06_07.md.
            q.append(AP(2, BD.HAS_SE,  L))
            q.append(AP(2, BD.CONST,  -L))
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


def _layer14_mem_generation_head_specs_with_overrides(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Merged L14 MEM-generation specs: base + overrides folded into one spec.

    Declarative replacement for the legacy post-bake patch
    :func:`_clear_l14_mem_generation_overbroad_sp_suppression`. The
    overrides are folded into the per-head ``q``/``k``/``v``/``o`` dicts
    so the lowered weights are byte-identical with running the override
    helper after :func:`_layer14_mem_generation_head_specs`.

    Each override cell is expressed as a (slot, dim) -> final-value
    assignment; the row-wide ``W_q[base + 33, :] *= 2.0`` step is handled
    by doubling the spec's existing slot-33 weights (every other column
    is zero pre-multiply and stays zero). The ``head_idx`` ordering of
    the returned specs matches :data:`_L14_HEAD_LAYOUT`.
    """

    pc_i = 0
    ax_i = 1
    sp_i = 2
    bp_i = 3
    mem_i = 4

    base_specs = _layer14_mem_generation_head_specs(BD)
    merged: list[DeclarativeAttentionHeadSpec] = []

    for spec in base_specs:
        head = spec.head_idx
        q_map: dict[tuple[int, int], float] = {
            (w.slot, w.dim): w.weight for w in spec.q
        }
        k_map: dict[tuple[int, int], float] = {
            (w.slot, w.dim): w.weight for w in spec.k
        }
        v_map: dict[tuple[int, int], float] = {
            (w.slot, w.dim): w.weight for w in spec.v
        }
        o_map: dict[tuple[int, int], float] = {
            (w.out_dim, w.slot): w.weight for w in spec.o
        }

        # === Common override block (all 8 heads) =====================
        # Slot 0: SP-suppression cancel.
        q_map[(0, BD.H1 + sp_i)] = 0.0
        # Slot 33: double every existing weight at this row (legacy
        # ``W_q[base + 33, :] *= 2.0`` only multiplies the cells the spec
        # touched; all other columns stay zero).
        slot33_keys = [key for key in q_map if key[0] == 33]
        for key in slot33_keys:
            q_map[key] = q_map[key] * 2.0
        q_map[(33, BD.H1 + sp_i)] = 0.0
        # Slot 35: MEM-source exclusion row (sign-stable).
        q_map[(35, BD.CONST)] = 40.0
        k_map[(35, BD.MARK_MEM)] = -40.0
        k_map[(35, BD.H3 + 4)] = -40.0
        # Slot 38: shared non-MEM target blocker (override strength 5000
        # supersedes the base spec's 2000).
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
            q_map[(38, dim)] = -target_block_s
        k_map[(38, BD.CONST)] = 5.0

        if head < 4:
            # === Heads 0-3: MEM addr byte generation overrides ========
            q_map[(38, BD.H1 + sp_i)] = -target_block_s
            for dim in (
                BD.MEM_VAL_B0,
                BD.MEM_VAL_B1,
                BD.MEM_VAL_B2,
                BD.MEM_VAL_B3,
            ):
                q_map[(38, dim)] = -target_block_s

            # Head 0: PSH addr byte 0 source bonus on MARK_SP (legacy
            # ``attn.W_k.data[1, BD.MARK_SP] = 45.0`` — row 1 lives in
            # head 0's row-block only).
            if head == 0:
                k_map[(1, BD.MARK_SP)] = 45.0

            # Slot-1 K guards (BP/STACK0 exclusion). Heads 1-3 also
            # source the SP byte row directly via byte-idx K bonus.
            k_map[(1, BD.H4 + bp_i)] = -30.0
            k_map[(1, BD.MARK_STACK0)] = -30.0

            if head in (1, 2, 3):
                # V slot 0 + zeroed OUTPUT reads (clean-payload only) so
                # the source byte row's own OUTPUT does not bleed into
                # the next-byte prediction.
                v_map[(0, BD.CONST)] = 1.0
                for kk in range(16):
                    v_map[(1 + kk, BD.OUTPUT_LO + kk)] = 0.0
                    v_map[(17 + kk, BD.OUTPUT_HI + kk)] = 0.0
                byte_dim_by_head = {
                    1: BD.BYTE_INDEX_1,
                    2: BD.BYTE_INDEX_2,
                    3: BD.BYTE_INDEX_3,
                }
                k_map[(1, byte_dim_by_head[head])] = 60.0

            # === Slot 2: SI/SC source selector rewrite ==============
            # Zero the base-spec K writes at slot 2 (every column the
            # override touches via ``= 0.0`` or per-head assignment).
            si_source_s = 200.0
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
                k_map[(2, dim)] = 0.0
            k_map[(2, BD.MEM_STORE)] = -4.0 * si_source_s
            k_map[(2, BD.H1 + ax_i)] = -si_source_s
            k_map[(2, BD.H1 + sp_i)] = -si_source_s
            if head == 0:
                k_map[(2, BD.STACK0_BYTE0)] = si_source_s
            elif head == 1:
                k_map[(2, BD.MARK_STACK0)] = -si_source_s
                k_map[(2, BD.H2 + bp_i)] = si_source_s
                k_map[(2, BD.L1H4 + bp_i)] = -si_source_s
            elif head == 2:
                k_map[(2, BD.MARK_STACK0)] = -si_source_s
                k_map[(2, BD.H3 + bp_i)] = si_source_s
                k_map[(2, BD.H2 + bp_i)] = -si_source_s
            else:  # head == 3
                k_map[(2, BD.MARK_STACK0)] = -si_source_s
                k_map[(2, BD.H4 + bp_i)] = si_source_s
                k_map[(2, BD.H3 + bp_i)] = -si_source_s

            # === Slots 39-43: ENT addr-source overrides =============
            ent_addr_s = 50.0
            ent_wrong_target_s = 500.0
            q_map[(39, BD.OP_ENT)] = ent_addr_s
            q_map[(39, BD.HAS_SE)] = ent_addr_s * 3.0
            q_map[(39, BD.CONST)] = -ent_addr_s * 6.0
            if head == 0:
                q_map[(39, BD.IS_BYTE)] = -12.0 * ent_addr_s
                k_map[(39, BD.MARK_BP)] = ent_addr_s
            else:
                k_map[(39, BD.H1 + bp_i)] = 0.0
                q_map[(40, BD.OP_ENT)] = ent_addr_s
                q_map[(40, BD.HAS_SE)] = 0.0
                q_map[(40, BD.CONST)] = 0.0
                for dim in (BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2):
                    q_map[(40, dim)] = 0.0
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
                        q_map[(40, dim)] = -ent_wrong_target_s
                k_map[(40, BD.H1 + bp_i)] = ent_addr_s
                k_map[(40, (BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3)[head - 1])] = ent_addr_s

                # MEM-marker block (heads 1..3 only).
                mem_marker_block_s = 300.0
                q_map[(41, BD.MARK_MEM)] = -mem_marker_block_s
                k_map[(41, BD.CONST)] = mem_marker_block_s

                # Slot-42 zero-out + slot-43 ENT target gate (heads 1..3).
                ent_target_gate_s = 200.0
                query_dim_by_head = {
                    1: BD.BYTE_INDEX_0,
                    2: BD.BYTE_INDEX_1,
                    3: BD.BYTE_INDEX_2,
                }[head]
                q_map[(42, BD.CONST)] = 0.0
                q_map[(42, BD.OP_ENT)] = 0.0
                q_map[(42, query_dim_by_head)] = 0.0
                k_map[(42, BD.CONST)] = 0.0
                q_map[(43, BD.CONST)] = -ent_target_gate_s
                q_map[(43, BD.H3 + mem_i)] = ent_target_gate_s
                k_map[(43, BD.CONST)] = ent_target_gate_s

        else:
            # === Heads 4-7: MEM val byte generation overrides ========
            # V scale-up: CLEAN_EMBED payload doubled, V[0] cancel x2.
            v_map[(0, BD.CONST)] = 2.0
            for kk in range(16):
                v_map[(1 + kk, BD.CLEAN_EMBED_LO + kk)] = 2.0
                v_map[(17 + kk, BD.CLEAN_EMBED_HI + kk)] = 2.0

            # Slot 36: value-source selector.
            source_s = 50.0
            q_map[(36, BD.CONST)] = source_s
            q_map[(36, BD.OP_JSR)] = -2.0 * source_s
            q_map[(36, BD.OP_ENT)] = -2.0 * source_s
            k_map[(36, BD.H1 + ax_i)] = source_s
            k_map[(36, BD.H1 + sp_i)] = 0.0
            k_map[(36, BD.H4 + bp_i)] = -source_s
            k_map[(36, BD.MARK_STACK0)] = source_s
            k_map[(36, BD.MARK_MEM)] = 0.0
            k_map[(36, BD.H3 + mem_i)] = 0.0

            # Slot 38: value-target blocks (current head's MEM_VAL stays
            # active; other 3 + MARK_MEM + BYTE_INDEX_0..2 are blocked).
            value_target_block_s = 15000.0
            for dim in (
                BD.MARK_MEM,
                BD.BYTE_INDEX_0,
                BD.BYTE_INDEX_1,
                BD.BYTE_INDEX_2,
            ):
                q_map[(38, dim)] = -value_target_block_s
            mem_val_dims = (
                BD.MEM_VAL_B0,
                BD.MEM_VAL_B1,
                BD.MEM_VAL_B2,
                BD.MEM_VAL_B3,
            )
            own_value_idx = head - 4
            for idx, dim in enumerate(mem_val_dims):
                if idx != own_value_idx:
                    q_map[(38, dim)] = -value_target_block_s

            # Slots 44-45: ENT old-BP source override.
            ent_old_bp_s = 80.0
            ent_value_target_s = 1000.0
            source_byte_dim = (
                BD.BYTE_INDEX_0,
                BD.BYTE_INDEX_1,
                BD.BYTE_INDEX_2,
                BD.BYTE_INDEX_3,
            )[own_value_idx]
            target_query_dim = mem_val_dims[own_value_idx]
            q_map[(44, BD.OP_ENT)] = ent_old_bp_s
            # OPCODE-BROADCAST HARDENING (2026-06-11, var_simple_12 / id 262):
            # slot 44 is the ENT-store value-SOURCE selector (it makes the head
            # attend the JSR-prologue old-BP position via K[OP_JSR]). Its only
            # Q gate was OP_ENT -- but OP_ENT does NOT stay one-hot at its own
            # marker; it BROADCASTS in-step onto every row at ~12-17 (audit
            # docs/OPCODE_BROADCAST_BLOCKER_AUDIT_2026_06_11.md). On the step-2
            # LEA PC-emit row (MARK_PC=1, OP_ENT=12.63, OP_JSR=0.86,
            # MEM_VAL_Bx=0) the slot-44 Q-score is 80 * 12.63 = 1010, so all
            # four value heads (4-7) attend the prologue position and copy its
            # CLEAN_EMBED_HI into OUTPUT_HI+0 (+1.0 each = +4.0), flipping the
            # PC high nibble 2->0 (0x2a -> 0x0a) and desyncing the program
            # (probe_var_full_chain.py 262: genesis at block 29 / logical L18).
            # FIX: hard subtractive NOT-blockers on the register-emit markers
            # (MARK_PC/AX/SP/BP). On the legitimate ENT MEM-value-emit firing
            # row those markers are ALL 0 (the value-target row carries
            # MARK_MEM + MEM_VAL_Bx, confirmed by the slot-38 base blocker which
            # already excludes MARK_PC/AX/BP), so the slot-44 score is
            # byte-identical there; on any register-emit row a single marker
            # contributes -1e6 << -(80 * OP_ENT_broadcast_max ~= 1400), burying
            # the OP_ENT broadcast and killing the misfire. Pure subtractive (no
            # net-zero compensation term) so it cannot perturb the legit row
            # even if a marker is fractionally nonzero.
            slot44_marker_block_s = 1000000.0  # 1e6 >> ent_old_bp_s * ENT_bcast
            for blk_dim in (BD.MARK_PC, BD.MARK_AX, BD.MARK_SP, BD.MARK_BP):
                q_map[(44, blk_dim)] = -slot44_marker_block_s
            k_map[(44, BD.OP_JSR)] = ent_old_bp_s
            k_map[(44, BD.H1 + bp_i)] = ent_old_bp_s
            k_map[(44, source_byte_dim)] = ent_old_bp_s
            q_map[(45, BD.CONST)] = -0.9 * ent_value_target_s
            q_map[(45, target_query_dim)] = ent_value_target_s
            k_map[(45, BD.CONST)] = ent_value_target_s

            # Slot 37: explicit zero-out in legacy override. The base
            # spec does not touch slot 37, so the assignments are
            # already 0; nothing to record.

        # === O override: zero-nibble cancel softened to -0.5 ========
        o_map[(BD.OUTPUT_LO + 0, 0)] = -0.5
        o_map[(BD.OUTPUT_HI + 0, 0)] = -0.5

        new_q = tuple(AP(slot, dim, w) for (slot, dim), w in q_map.items())
        new_k = tuple(AP(slot, dim, w) for (slot, dim), w in k_map.items())
        new_v = tuple(AP(slot, dim, w) for (slot, dim), w in v_map.items())
        new_o = tuple(AO(out_dim, slot, w) for (out_dim, slot), w in o_map.items())
        merged.append(DeclarativeAttentionHeadSpec(
            head_idx=spec.head_idx,
            q=new_q,
            k=new_k,
            v=new_v,
            o=new_o,
        ))

    return tuple(merged)


def _layer14_mem_generation_ir(dim_positions, HD) -> CompilerIR:
    """Build the declarative L14 mem-generation IR for the compiler.

    Eight heads (0-3 address, 4-7 value). See
    :func:`_layer14_mem_generation_head_specs` for the base layout and
    :func:`_layer14_mem_generation_head_specs_with_overrides` for the
    merged base+override spec used at bake time.
    """

    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for spec in _layer14_mem_generation_head_specs_with_overrides(proxy):
        ir.layer(0).attention.append(spec)
    return ir


def make_layer14_attn_dep_anchor_op() -> Operation:
    """No-op companion for ``layer14_mem_generation``: declares mirrored
    reads/writes so the LayerCompiler's dep graph reserves an L14 slot.

    Mirrors ``_layer13_attn_dep_anchor`` / ``_layer11_ffn_dep_anchor``
    / ``_layer3_ffn_dep_anchor``: the actual weight bake happens in
    ``layer14_mem_generation`` (kind="attn"); this op's bake is a no-op.

    Phase 8.G.6 follow-up (4-of-4 holdouts): lets
    ``layer14_mem_generation`` declare
    ``requires={"same_layer_as": "_layer14_attn_dep_anchor"}`` (kind="attn"
    cannot use ``target_op_name`` — that field is block-op-only) and drop
    its ``layer_idx=14`` literal. ``requires={"after":
    "_layer13_attn_dep_anchor"}`` pins this anchor strictly past L13 so
    the earliest landable layer is L14, mirroring the L13 anchor's pin
    past L12.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in ``layer14_mem_generation`` (kind="attn").
        return

    return Operation(
        name="_layer14_attn_dep_anchor",
        # Phase=14 matches ``layer14_mem_generation`` so the two attn ops
        # share the same (layer, kind) slot in ``_assign_layers``. The
        # ``requires["after"] = "_layer13_attn_dep_anchor"`` constraint
        # pushes the earliest landable layer to L14 (one past the L13
        # anchor); the shared phase keeps mem_generation co-placed at
        # L14 instead of advancing to L15 (mismatched phase would
        # consume a separate (layer, kind) slot via the round-robin
        # advance loop, defeating the ``same_layer_as`` equality
        # assertion downstream).
        phase=14,
        # Subset of layer14_mem_generation reads/writes. Excludes the
        # high-fan-in dims (CLEAN_EMBED, OUTPUT) the routing/attn fabric
        # writes everywhere, so the anchor's earliest landable layer is
        # not pushed past L14 by upstream same-step writers.
        reads={"MARK_MEM", "MARK_SP", "OP_PSH", "OP_SI", "OP_SC",
               "OP_JSR", "OP_ENT", "MEM_STORE", "MEM_ADDR_SRC"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Pin strictly after the L13 anchor so the earliest landable
        # layer is L14 (force a layer past L13).
        requires={"after": "_layer13_attn_dep_anchor"},
        smoke_tests=set(),
        spec_section=None,
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
    )


def make_layer14_mem_generation_op() -> Operation:
    """L14 attention: generate MEM section tokens (addr + value) for SI/SC/PSH.

    Wave 2E (Phase 6): migrated to ``DeclarativeAttentionHeadSpec`` form.
    Heads 0-3 (address) and 4-7 (value) are expressed declaratively via
    :func:`_layer14_mem_generation_head_specs_with_overrides`; the
    imperative :func:`vm_step._set_layer14_mem_generation` helper is no
    longer called, and the legacy post-bake override
    :func:`_clear_l14_mem_generation_overbroad_sp_suppression` has been
    folded into the spec (byte-identical with running the spec then the
    override imperatively).
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
            attn,
            _layer14_mem_generation_head_specs_with_overrides(proxy),
            HD,
        )
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
        # Phase 8.G.5 carve-out: SCC cycle-breaker. _layer14_attn_dep_anchor
        # has phase=14 and uses requires["same_layer_as"]; this co-placement
        # rule requires the phase ordinal here. Retire when the L14 anchor
        # gains its own placement primitive or migrates to requires["after"].
        phase=14,
        # Phase 8.A: ADDR_B0_HI_PREV_STEP marks the read as cross-step
        # relative to L15 store_stack0_sp_byte0_addr (phase 15.2), which
        # writes ADDR_B0_HI after L14 in the same step. All other
        # ADDR_B0_HI writers (L4 sp_to_addr_key, L8 sp_gather_bake, L9
        # lev_addr_relay/lev_bp_to_pc_relay, L13 mem_addr_gather) sit
        # earlier in the schedule — those forward edges still resolve to
        # the same numeric slot 206. The alias is position-identical so
        # weight bakes stay byte-identical; only the dep graph view changes.
        # Phase 8.A follow-up: ADDR_B0_LO_PREV_STEP matches the HI pattern
        # for the LO-nibble band. L15 store_stack0_sp_byte0_addr writes
        # ADDR_B0_LO after L14 in the same step; the PREV_STEP alias
        # retires the back-edge while keeping the numeric slot identical.
        reads={"MARK_MEM", "MARK_SP", "MARK_STACK0", "OP_PSH", "OP_SI", "OP_SC",
               "OP_JSR", "OP_ENT", "OP_LI", "OP_LC",
               "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_B0_LO.*.-1", "ADDR_B0_HI.*.-1",
               "MEM_STORE", "MEM_ADDR_SRC", "STACK0_BYTE0", "L1H0", "L1H1", "L1H2",
               "H0", "H1", "L1H4", "H2", "H3", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3", "IS_BYTE",
               # head specs (h=1,2,3) read STACK0_BYTE_VAL_h_{LO,HI} produced
               # by L10 ``layer10_psh_ax_broadcast`` for the SI/SC byte-h
               # source. Declared in ``reads`` so the producer-consumer dim
               # contract (``stack0_byte_val_*_pshk2mem``) verifies clean.
               "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
               "STACK0_BYTE_VAL_2_LO", "STACK0_BYTE_VAL_2_HI",
               "STACK0_BYTE_VAL_3_LO", "STACK0_BYTE_VAL_3_HI",
               # head-5 slot-2 JSR step-0 guard: requires HAS_SE=1 so the
               # BP-frame-byte-1 K selectors only fire on steps with a
               # prior STEP_END. See docs/VAR_MUL_ATTRIBUTION_2026_06_09.md.
               "HAS_SE", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        # Phase 8.G.6 follow-up: drop ``layer_idx=14`` literal and bind to
        # the L14 attn dep anchor via ``requires["same_layer_as"]``.
        # ``kind="attn"`` cannot use ``target_op_name`` (block-op-only),
        # so co-placement with ``_layer14_attn_dep_anchor`` is the
        # equivalent mechanism. The anchor in turn is pinned past
        # ``_layer13_attn_dep_anchor`` so the earliest landable layer is
        # L14 — byte-identical placement to the prior literal pin.
        requires={"same_layer_as": "_layer14_attn_dep_anchor"},
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
    # 16-bit OR/XOR byte-1 fix (2026-06-11): widen the K-gate from
    # OP_MUL/OP_SHL to also fire on OP_OR/OP_XOR. The L13
    # ``layer13_bitwise_byte1_gather`` head stages operand-A byte 1 into
    # AX_FULL_LO/HI at the OR/XOR MARK_AX row; this relay then copies that
    # staged byte 1 into OUTPUT at the byte-1 AX-emit token, exactly as it
    # does for MUL/SHL. OP_AND is intentionally NOT added: and_16bit needs
    # byte 1 = 0x00 (A_b1 AND 0), which the un-staged (empty AX_FULL)
    # default already emits -- so AND keeps its current passing behaviour.
    _BITWISE_RELAY_OPS = (BD.OP_OR, BD.OP_XOR)
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
    for _op in _BITWISE_RELAY_OPS:
        k.append(AP(1, _op, 100.0))
        k.append(AP(35, _op, 2000.0))
        k.append(AP(36, _op, 200.0))
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
        reads={"IS_BYTE", "H1", "H3", "H4", "BYTE_INDEX_0", "MARK_AX",
               "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "STACK0_BYTE0", "OP_MUL",
               "OP_SHL", "OP_LI_RELAY", "OP_LC_RELAY",
               # 16-bit OR/XOR byte-1 fix (2026-06-11): K-gate widened to
               # OP_OR/OP_XOR so the relay also fires for bitwise byte 1.
               "OP_OR", "OP_XOR",
               "AX_FULL_LO", "AX_FULL_HI", "TEMP", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=15`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer15_memory_lookup`` (the L15 attn op).
        target_op_name="layer15_memory_lookup",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir_factory=_layer14_alu_high_byte_relay_ir,
        migrated=True,
        claims=_claims,
        alibi_slopes={8: 1.0},
        requires={"after": "layer15_memory_lookup"},
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
        # OPCODE-BROADCAST HARDENING (2026-06-11, var_simple_12 / id 262):
        # mirror of the declarative spec
        # ``_layer14_mem_generation_head_specs_with_overrides`` slot 44. The
        # declarative spec is the LIVE path (make_layer14_mem_generation_op's
        # bake calls Primitives.generate_attention_heads on it); this imperative
        # ``_clear_l14_mem_generation_overbroad_sp_suppression`` helper is no
        # longer invoked (folded into the spec). Kept in sync to prevent a stale
        # re-enable from reintroducing the misfire. Hard subtractive NOT-blockers
        # on the register-emit markers bury OP_ENT's in-step broadcast (~12-17)
        # so slot 44 cannot fire on the step-2 LEA PC-emit row; byte-identical on
        # the legit ENT MEM-value row (MARK_PC/AX/SP/BP all 0 there).
        slot44_marker_block_s = 1000000.0  # 1e6 >> ent_old_bp_s * ENT_bcast
        for blk_dim in (BD.MARK_PC, BD.MARK_AX, BD.MARK_SP, BD.MARK_BP):
            attn.W_q.data[base + 44, blk_dim] = -slot44_marker_block_s
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
        multi_way_and_rule(
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
        multi_way_and_rule(
            name="l14_clear_addsub_temp_negative_residue_8",
            conditions=(("TEMP+8", -1.0),),
            threshold=0.0,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("TEMP+8", 2.0 / S),),
        ),
        # Unit 2: TEMP[9] negative residue clamp.
        multi_way_and_rule(
            name="l14_clear_addsub_temp_negative_residue_9",
            conditions=(("TEMP+9", -1.0),),
            threshold=0.0,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("TEMP+9", 2.0 / S),),
        ),
        # Unit 3: ADD byte-1 high-nibble zero cleanup.
        multi_way_and_rule(
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
        _claims.add((14, "ffn_W_down", "3", f"OUTPUT_HI+{k}"))

    return Operation(
        name="layer14_temp_clear",
        reads={"OP_LEV", "MARK_PC", "TEMP", "IS_BYTE", "H1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "MARK_AX", "AX_CARRY_HI", "CONST"},
        writes={"TEMP", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_temp_clear_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        claims=_claims,
        requires={"after": "layer14_mem_generation"},
        # Phase 7 produces/consumes_fresh POC (see
        # docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py`` from the IR rule writes /
        # condition reads, then slot-mapped manually:
        #   - TEMP+0 at PC marker (unit 0)        -> "PC_marker"
        #   - TEMP+8/9 position-uniform (1-2)     -> "PC_marker" (default)
        #   - OUTPUT_HI_THIS_STEP+0..15 at AX     -> "AX_byte0"
        # AX_CARRY_HI is a same-step read produced by L13 carry ALU at
        # AX_byte0; declaring it consumes_fresh lets the scheduler enforce
        # the L13 -> L14 ordering as a fresh-residual dependency.
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
        multi_way_and_rule(
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
        reads={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "MARK_PC", "MARK_BP", "MARK_AX", "MARK_STACK0", "MARK_SP",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_clear_addr_key_pollution_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Phase 7.A.2 backfill: this op cancels ADDR_KEY residue left on
        # non-MEM, non-marker rows by prior ADDR_KEY writers. The data dep
        # is on the *write side* (we read CONST/markers and write ADDR_KEY)
        # so the dim-only analyzer cannot infer the real upstream. The
        # actual ADDR_KEY writers we run after are:
        #   - layer4_pc_relay      (phase 4,   same step)
        #   - layer7_memory_heads  (phase 7,   same step)
        #   - layer14_addr_key_neural_decode (phase 14.5, prev-step carry --
        #     fires LATER in the current step but the residual we clear is
        #     last step's output; block ops are pinned to layer_idx so this
        #     listing is purely informational for the scheduler analyzer).
        requires={"after": [
            "layer4_pc_relay",
            "layer7_memory_heads",
            "layer14_addr_key_neural_decode",
            "layer14_mem_generation",
        ]},
        # Wave 2 (docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py``. 48 single-write rules,
        # each writes ADDR_KEY+k at non-MEM, non-marker positions
        # (scope: ``not MEM_VAL_B* and not MARK_*``). All structural
        # conditions are cross-step-durable markers (MEM_VAL_B*,
        # MARK_PC/BP/AX/SP/STACK0), so consumes_fresh is empty after the
        # _CROSS_STEP_DURABLE allowlist filter. The slot tag follows the
        # POC convention of naming the bind-target op for "everywhere
        # except marker rows" writes (no single anatomical register).
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
        multi_way_and_rule(
            name="l14_clear_output_corruption_lo0_boost",
            conditions=boost_unit_conditions(0, "OUTPUT_LO"),
            threshold=0.5,  # imperative: b_up = -S * 0.5
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("OUTPUT_LO+0", 50.0 / S),),
        ),
        multi_way_and_rule(
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
        multi_way_and_rule(
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
    #            W_down[OUTPUT_HI+nibble] per unit for nibble in 1..15.
    _claims = {
        (14, "ffn_W_down", "52", "OUTPUT_LO+0"),
        (14, "ffn_W_down", "53", "OUTPUT_HI+0"),
    }
    for nibble in range(1, 16):
        _claims.add(
            (14, "ffn_W_down", str(54 + nibble), f"OUTPUT_HI+{nibble}")
        )

    return Operation(
        name="layer14_clear_output_corruption",
        reads={"H4", "H1", "H3", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "OP_JSR", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "IS_BYTE", "BYTE_INDEX_3", "PSH_AT_SP", "CMP",
               "MEM_STORE", "OUTPUT_LO", "OUTPUT_HI", "TEMP", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_clear_output_corruption_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        claims=_claims,
        requires={"after": "layer14_mem_generation"},
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
        return multi_way_and_rule(
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
                 f"OUTPUT_HI+{k}")
            )

    return Operation(
        name="layer14_clear_mem_marker_output",
        reads={"OP_JSR", "OP_ENT", "MARK_MEM", "IS_BYTE",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_clear_mem_marker_output_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Wave 1 (docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py``. Rules fire at MARK_MEM
        # rows gated by OP_JSR or OP_ENT, writing OUTPUT_LO/HI at the
        # MEM marker -- new "MEM_marker" slot per the existing
        # AX_marker / PC_marker / SP_marker / BP_marker convention.
        # OP_JSR is the in-step opcode broadcast; OP_ENT is allowlisted
        # cross-step durable per derive's _CROSS_STEP_DURABLE and is
        # NOT in consumes_fresh (architecturally correct: ENT is a
        # multi-step routine -- see STALENESS_INVARIANTS.md).
        claims=_claims,
        requires={"after": "layer14_mem_generation"},
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#memory",
    )


def make_layer14_mem_addr_src_default_suppress_op() -> Operation:
    """L14 FFN: Cancel the L3 ``mem_byte_0_default`` baseline at SI/SC stores.

    Var-cluster follow-up (2026-06-05): the L3 FFN ``MEM DEFAULT`` rules
    write ~+0.940 to OUTPUT_LO[0]/HI[0] at the MEM marker row and the
    three MEM addr-byte query rows, biasing addr predictions toward
    ``0x00``. For SI/SC stores (MEM_ADDR_SRC=1) the addr comes from
    STACK0 and can be ANY value (e.g. 0xFFFC for var_simple_0), so the
    baseline is wrong-direction. This op subtracts the baseline only
    when MEM_ADDR_SRC=1, leaving the PSH/JSR/ENT/non-store paths intact.

    The brief's "gate the L3 rule on MEM_ADDR_SRC=0" cannot fire at L3
    because MEM_ADDR_SRC is decoded at the AX marker by L5 from
    OP_SI/OP_SC and relayed to the MEM marker by L6 head 6 / to MEM
    byte positions by L7 head 7 — so it is NOT available at L3 input.
    This op achieves the same algebra at L14 by a counter-write that
    subtracts -2.0/S only when MEM_ADDR_SRC=1, mirroring the L3 rule
    shape exactly.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Phase 14.45 places
    it AFTER ``clear_mem_marker_output`` (14.4) and BEFORE
    ``addr_key_neural_decode`` (14.5), within the cleanup chain that
    runs after the L14 mem_generation attention. Allocates 8 units via
    the chain layout (2 marker units + 6 byte-index units).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_mem_addr_src_default_suppress
        ffn = block.ffn
        start_unit = _l14_chain_alloc("layer14_mem_addr_src_default_suppress")
        next_unit = _set_layer14_mem_addr_src_default_suppress(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_mem_addr_src_default_suppress",
        phase=14.45,
        reads={"MARK_MEM", "H1", "BYTE_INDEX_0", "BYTE_INDEX_1",
               "BYTE_INDEX_2", "MEM_ADDR_SRC", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        # Phase 7 declarative-authority audit: this op's bake reaches the
        # ``_set_layer14_*`` legacy helper, but the helper's writes ARE
        # the declared rule program (counter-write against the L3
        # ``MEM DEFAULT`` baseline gated by ``MEM_ADDR_SRC=1``). Mark
        # explicitly as declarative so the audit classifier doesn't fall
        # back to the ``_set_*`` heuristic and flag it as a legacy
        # wrapper. Matches the sibling L14 cleanup-chain ops which use
        # ``spec_generated`` alongside an explicit ``compiler_ir=`` rule
        # bundle; this op pre-dates the IR factorisation but the bake
        # behaviour is identical in intent.
        declarative_authority="declarative",
        # Phase 8.A.4: use ``target_op_name`` to bind to whichever layer
        # the compiler placed ``layer14_mem_generation`` (the L14 attn
        # op). Matches the convention used by the other L14 cleanup ops.
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Tier A opcode gating: only fires when MEM_ADDR_SRC=1 (SI/SC
        # store address from STACK0). MEM_ADDR_SRC is decoded at the AX
        # marker by L5 from OP_SI/OP_SC and relayed to MEM marker by L6
        # head 6 / to MEM byte positions by L7 head 7. The matching
        # opcodes are OP_SI and OP_SC.
        opcodes={"OP_SI", "OP_SC"},
        requires={"after": "layer14_clear_mem_marker_output"},
        # Staleness invariants: subtracts the L3 +0.940 baseline at
        # OUTPUT_LO[0]/HI[0] for SI/SC stores. Canonical register is
        # the MEM_addr1 byte row (where the var-cluster diagnostic
        # flagged the 0xff-vs-0x00 inversion); the same shape applies
        # at the MEM marker (predicting addr_b0) and at MEM_addr2/
        # MEM_addr3 by symmetry.
        produces={
            "OUTPUT_LO": "MEM_addr1",
            "OUTPUT_HI": "MEM_addr1",
        },
    )


def make_layer14_jsr_mem_default_suppress_op() -> Operation:
    """L14 FFN: Cancel the L3 ``mem_byte_0_default`` baseline at JSR/PSH/ENT.

    Var-cluster JSR-path sibling of ``f4f9103d`` (2026-06-06): the prior
    SI/SC fix at phase 14.45 cancels the L3 ``MEM DEFAULT'' +0.940
    baseline when ``MEM_ADDR_SRC=1`` (SI/SC). The var-cluster failures
    are not at SI/SC, however — they are at the **JSR call** step where
    the return-address bytes are pushed to memory at SP (which sits at
    ``0xFFFC`` for the var-cluster fixtures, so addr byte 1 is ``0xff``).
    JSR (and PSH and ENT) have ``MEM_ADDR_SRC=0`` (the address source is
    SP, not STACK0), so the SI/SC cancel does NOT fire and the wrong-
    direction L3 baseline survives.

    JSR/PSH/ENT all set ``MEM_STORE=1`` (via the L6 opcode-relay head 6,
    broadcast to MEM byte positions by L7 head 7) but have
    ``MEM_ADDR_SRC=0``. So the gate ``MEM_STORE AND NOT MEM_ADDR_SRC``
    selects exactly the JSR/PSH/ENT store path and is disjoint from the
    SI/SC gate already handled by
    ``layer14_mem_addr_src_default_suppress``. Non-store opcodes have
    ``MEM_STORE=0`` so this cancel does not fire (the L3 baseline is
    correct for IMM / arithmetic / branches etc.).

    Bound with ``target_op_name="layer14_mem_generation"`` so it follows
    the compiler-selected layer for the L14 attention op. Phase 14.46
    places it AFTER ``mem_addr_src_default_suppress`` (14.45) and BEFORE
    ``addr_key_neural_decode`` (14.5), within the cleanup chain that runs
    after the L14 mem_generation attention. Allocates 4 units via the
    chain layout (2 marker units + 2 BYTE_INDEX_0 units).

    NARROWING 2026-06-06: dropped BYTE_INDEX_1 / BYTE_INDEX_2 byte rows
    (4 units) because the L3 baseline of 0x00 at MEM_addr2 / MEM_addr3
    is correct for 16-bit address pushes (SP=0xFFFC → return-address
    bytes ``0xFC, 0xFF, 0x00, 0x00`` little-endian). The earlier 8-unit
    version over-cancelled there and a 0x11 residue won the argmax.
    See VAR_CLUSTER_JSR_PATH_FINDINGS_2026_06_06.md.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_jsr_mem_default_suppress
        ffn = block.ffn
        start_unit = _l14_chain_alloc("layer14_jsr_mem_default_suppress")
        next_unit = _set_layer14_jsr_mem_default_suppress(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_jsr_mem_default_suppress",
        phase=14.46,
        reads={"MARK_MEM", "H1", "BYTE_INDEX_0",
               "MEM_STORE", "MEM_ADDR_SRC", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        # Phase 7 declarative-authority audit: same rationale as the
        # SI/SC sibling ``layer14_mem_addr_src_default_suppress`` above.
        # The bake calls a ``_set_layer14_*`` legacy helper, but the
        # helper IS the rule program (counter-write against the L3
        # ``MEM DEFAULT`` baseline gated by ``MEM_STORE=1 AND
        # MEM_ADDR_SRC=0`` for PSH/JSR/ENT). Mark explicitly as
        # declarative so the audit classifier doesn't flag it as a
        # legacy wrapper.
        declarative_authority="declarative",
        # Phase 8.A.4: use ``target_op_name`` to bind to whichever layer
        # the compiler placed ``layer14_mem_generation`` (the L14 attn
        # op). Matches the convention used by the other L14 cleanup ops.
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Tier A opcode gating: fires when MEM_STORE=1 AND MEM_ADDR_SRC=0
        # (PSH/JSR/ENT). MEM_STORE is set for SI/SC/PSH/JSR/ENT at the
        # AX marker by L5 FFN, relayed to MEM marker by L6 head 6, and
        # broadcast to MEM byte positions by L7 head 7. MEM_ADDR_SRC=1
        # only for SI/SC (whose cancel is handled by the prior op at
        # 14.45). The matching opcodes are OP_PSH, OP_JSR, OP_ENT.
        opcodes={"OP_PSH", "OP_JSR", "OP_ENT"},
        requires={"after": "layer14_mem_addr_src_default_suppress"},
        # Staleness invariants: subtracts the L3 +0.940 baseline at
        # OUTPUT_LO[0]/HI[0] for PSH/JSR/ENT stores. Canonical register is
        # the MEM_addr1 byte row (where the var-cluster diagnostic
        # flagged the 0xff-vs-0x00 inversion); the same shape applies
        # at the MEM marker (predicting addr_b0) and at MEM_addr2/
        # MEM_addr3 by symmetry.
        produces={
            "OUTPUT_LO": "MEM_addr1",
            "OUTPUT_HI": "MEM_addr1",
        },
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
    # Phase 8.D: the OP_JSR gate resolves through the
    # (opcode_flag, "JSR") semantic pair; byte-identical to the
    # legacy "OP_JSR" slot string via DimRef.parse.
    gate_jsr = dim_ref("opcode_flag", "JSR")
    common_kwargs = dict(
        conditions=common_conditions,
        threshold=1.5,
        gate=gate_jsr,
        gate_weight=1.0,
        gate_bias=0.0,
        scope="OP_JSR and IS_BYTE and H1+1",
    )
    rules = (
        multi_way_and_rule(
            name="l14_jsr_ax_bytes_zero_lo_neg",
            writes=tuple((f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_jsr_ax_bytes_zero_hi_neg",
            writes=tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
            ),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_jsr_ax_bytes_zero_lo0_boost",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
            **common_kwargs,
        ),
        multi_way_and_rule(
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
        # Phase 6 Wave 6D: this op's layout entry uses ``pin=None`` so the
        # allocator picks the first free gap past the prior chain claims.
        # First-fit on a 4096-wide pool with predecessors filling
        # [0, 1862) lands the 4-unit auto-fit range back at unit 1862, so
        # the FFN weights are byte-identical to the legacy pin even though
        # the author never wrote the offset. The rules don't cross-reference
        # the unit index, so moving the range elsewhere (e.g. if a future
        # predecessor expands) would still leave the FFN function invariant.
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
    # unit 1874. The bake's auto-fit allocator (Phase 6 Wave 6D) lands the
    # 4 units back at 1874 because the prior chain claims fill [0, 1874)
    # contiguously and first-fit on a 4096-wide pool picks the first free
    # gap. The helper writes 4 units (shifted +4 by var-cluster JSR-path
    # follow-up's jsr_mem_default_suppress insertion 2026-06-06; initially
    # +8 then narrowed to +4 the same day, see VAR_CLUSTER_JSR_PATH_
    # FINDINGS_2026_06_06.md):
    #   unit 1874: -3/S on OUTPUT_LO[0..15]
    #   unit 1875: -3/S on OUTPUT_HI[0..15]
    #   unit 1876: +5/S on OUTPUT_LO[0]
    #   unit 1877: +5/S on OUTPUT_HI[0]
    _claims = set()
    for k in range(16):
        _claims.add((14, "ffn_W_down", "1874", f"OUTPUT_LO+{k}"))
        _claims.add((14, "ffn_W_down", "1875", f"OUTPUT_HI+{k}"))
    _claims.add((14, "ffn_W_down", "1876", "OUTPUT_LO+0"))
    _claims.add((14, "ffn_W_down", "1877", "OUTPUT_HI+0"))

    return Operation(
        name="layer14_jsr_ax_bytes_zero",
        reads={"OP_JSR", "IS_BYTE", "H1", "CONST", "STACK0_BYTE0", "STACK0_BYTE1",
               "STACK0_BYTE2", "STACK0_BYTE3"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_jsr_ax_bytes_zero_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Wave 1 (docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py`` from the IR rule writes
        # then slot-mapped: the rule scope is ``IS_BYTE and H1+1`` (AX
        # byte positions), matching the existing AX_byte0 convention
        # used by the L14 temp_clear POC. OP_JSR is broadcast to AX byte
        # positions by L7 head 5 (V slot 8) in the same step, so it's
        # the fresh in-step value, not a cross-step durable.
        claims=_claims,
        requires={"after": "layer14_mem_generation"},
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
        multi_way_and_rule(
            name="l14_alu_nocarry_ax_bytes_zero_lo_neg",
            writes=tuple((f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_alu_nocarry_ax_bytes_zero_hi_neg",
            writes=tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
            ),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_alu_nocarry_ax_bytes_zero_lo0_boost",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
            **common_kwargs,
        ),
        multi_way_and_rule(
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
    # op in the L14 cleanup chain. Predecessors leave the counter at 1882.
    # The helper writes 4 units mirroring jsr/lc_ax_bytes_zero (shifted +4
    # by var-cluster JSR-path follow-up's jsr_mem_default_suppress 2026-06-06;
    # initial +8 was narrowed to +4 the same day, see findings doc):
    #   unit 1882: -3/S on OUTPUT_LO[0..15]
    #   unit 1883: -3/S on OUTPUT_HI[0..15]
    #   unit 1884: +5/S on OUTPUT_LO[0]
    #   unit 1885: +5/S on OUTPUT_HI[0]
    _claims = set()
    for k in range(16):
        _claims.add((14, "ffn_W_down", "1882", f"OUTPUT_LO+{k}"))
        _claims.add((14, "ffn_W_down", "1883", f"OUTPUT_HI+{k}"))
    _claims.add((14, "ffn_W_down", "1884", "OUTPUT_LO+0"))
    _claims.add((14, "ffn_W_down", "1885", "OUTPUT_HI+0"))

    return Operation(
        name="layer14_alu_nocarry_ax_bytes_zero",
        # Phase 1 (memory cluster fix plan): shares L14 FFN unit range with
        # ``layer14_demo_phase6_wave7`` at a disjoint sub-range (this op
        # owns units 0..1885; the demo op owns unit 1886). See
        # docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("ffn_units",),
        reads={"TEMP", "IS_BYTE", "H1", "BYTE_INDEX_3", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_alu_nocarry_ax_bytes_zero_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Wave 1 (docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py``; mirrors
        # ``layer14_jsr_ax_bytes_zero``. Rule scope ``IS_BYTE and H1+1``
        # = AX byte positions -> AX_byte0 slot. TEMP[7] (NOCARRY_ALU_OP
        # relay) is populated by L7 head 5 V slot 9 in the same step.
        claims=_claims,
        requires={"after": "layer14_mem_generation"},
        # The L14 FFN chain (``_l14_unit_counter`` reaches 1886 after this
        # op runs). The chain is: temp_clear + temp residue clamp +
        # ADD byte-1 high cleanup (4 units) →
        # clear_addr_key_pollution (48) → clear_output_corruption (3) →
        # PSH MEM high-nibble boost (15) → clear_mem_marker_output (64) →
        # mem_addr_src_default_suppress (8, var-cluster 2026-06-05) →
        # jsr_mem_default_suppress (4, JSR-path narrowed 2026-06-06) →
        # addr_key_neural_decode (1728) →
        # jsr_ax_bytes_zero (4) → lc_ax_bytes_zero (4) → this op (4) →
        # demo_phase6_wave7 (1). Annotating only the chain tail with the
        # cumulative max is sufficient — the compiler aggregates per-layer
        # max across all ops, so this single annotation suffices for L14
        # dynamic sizing.
        ffn_units_used=1886,
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


# === L14 ENT AX bytes 1-3 zero (Wave 1 Cluster B1, 2026-06-07) ===========


def _layer14_ent_ax_bytes_zero_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program: zero AX bytes 1-3 when OP_ENT is broadcast.

    Mirrors :func:`_layer14_jsr_ax_bytes_zero_rules` but gates on
    ``OP_ENT`` instead of ``OP_JSR``. The OP_ENT flag is broadcast by L7
    head 7 V slot 4 (see ``layer7_memory_heads`` claims) so OP_ENT is
    present at AX byte positions (IS_BYTE + H1[AX]) within the ENT step.
    At those positions, the four units spread -3/S across one nibble
    band (LO or HI) or boost a single byte-value-0 slot (+5/S on
    OUTPUT_LO[0] / OUTPUT_HI[0]). The byte-value-0 token then wins
    argmax -> AX bytes 1-3 = 0x00.

    Pre-fix observed token stream for ``ENT, IMM 0, LEA 2, EXIT``:
    step 0 emits ``REG_AX 0x00 0xE8 0xE8 0xE8`` (SP byte 0 leak); after
    fix the trailing 0xE8 bytes are pulled to 0x00, restoring the C4
    AX-byte-1..3 invariant and unblocking the LEA address path.
    """
    AX_I = 1
    common_conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{AX_I}", 1.0),
    )
    # Resolve the OP_ENT gate via the (opcode_flag, "ENT") semantic pair;
    # byte-identical to the raw "OP_ENT" slot string via DimRef.parse.
    gate_ent = dim_ref("opcode_flag", "ENT")
    common_kwargs = dict(
        conditions=common_conditions,
        threshold=1.5,
        gate=gate_ent,
        gate_weight=1.0,
        gate_bias=0.0,
        scope="OP_ENT and IS_BYTE and H1+1",
    )
    rules = (
        multi_way_and_rule(
            name="l14_ent_ax_bytes_zero_lo_neg",
            writes=tuple((f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_ent_ax_bytes_zero_hi_neg",
            writes=tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
            ),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_ent_ax_bytes_zero_lo0_boost",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_ent_ax_bytes_zero_hi0_boost",
            writes=(("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
            **common_kwargs,
        ),
    )
    return rules


def _layer14_ent_ax_bytes_zero_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_ent_ax_bytes_zero_rules(S))
    return ir


def make_layer14_ent_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when OP_ENT is active.

    Wave 1 Cluster B1 (2026-06-07): fixes ``test_lea_basic``. Per C4's
    8-bit-AX-with-32-bit-register convention, AX bytes 1-3 must be 0.
    At ENT step 0, the model leaks SP byte 0 (0xE8 from the ENT-pushed
    frame) into AX byte positions 1-3, producing AX=0xE8E8E800. The
    LEA step downstream depends on a clean AX (and a clean BP, which is
    cascaded from AX); the leak collapses LEA to AX=0x00 instead of
    BP + imm, and ``test_lea_basic`` (program
    ``[ENT, IMM 0, LEA 2, EXIT]``, expects AX != 0) fails with
    exit_code=0.

    Mirrors :func:`make_layer14_jsr_ax_bytes_zero_op` and the LC / ALU
    nocarry variants: gates on ``OP_ENT`` (broadcast by L7 head 7 V slot
    4) at AX byte positions (IS_BYTE + H1[AX]). Four units zero
    OUTPUT_LO/OUTPUT_HI nibble dims (-3/S each) and boost the byte-0
    slot (+5/S on OUTPUT_LO[0] / OUTPUT_HI[0]).

    Pinned to ``layer_idx=14`` via ``kind="block"``. Runs after
    ``layer14_alu_nocarry_ax_bytes_zero`` in the L14 cleanup chain. Uses
    auto-fit allocation via :func:`_l14_chain_alloc`.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        start_unit = _l14_chain_alloc("layer14_ent_ax_bytes_zero")
        ir = _layer14_ent_ax_bytes_zero_ir(S)
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

    return Operation(
        name="layer14_ent_ax_bytes_zero",
        # Same FFN unit range share as the sibling cleanup ops: this op
        # owns 4 units past the alu_nocarry tail; demo_phase6_wave7 sits
        # past it. See docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("ffn_units",),
        reads={"OP_ENT", "IS_BYTE", "H1", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_ent_ax_bytes_zero_ir(),
        declarative_authority="spec_generated",
        target_op_name="layer14_mem_generation",
        migrated=True,
        requires={"after": "layer14_mem_generation"},
        smoke_tests={"TestSmokeAddress::test_lea_basic"},
        spec_section="BLOG_SPEC.md#registers",
    )


# === L14 Phase 6 Wave 7 demo: pure-declaration corrective op =============
#
# A minimal demonstration of the declarative IR vision: adding a fix is one
# ``FFNRule`` edit; the compiler picks the layer/slot/unit (Phase 6 Wave 6D
# ``pin=None`` auto-fit on the L14 cleanup chain); and the weights are
# synthesised from the spec via :meth:`CompilerIR.lower_ffn`. The op is
# byte-identically a no-op on the live corpus -- its sole rule combines an
# impossible condition (``CONST = -100``) with a ``__never_fires__`` scope so
# SiLU collapses to 0 and the writes (``TEMP+0`` with weight 0) leave the
# residual unchanged. The point is the END-TO-END FLOW (declare -> byte
# identity gate -> compile -> corpus check), not a behavioural fix.
#
# Per ``docs/HOW_TO_ADD_A_CORRECTIVE_OP.md``: this demo is what step 5's
# ``compiler_ir=`` + slim ``bake_fn`` looks like in practice, and what step 7's
# byte-identity gate (``compare_symbolic_to_lowered_ffn``) clears for a single
# new rule.


def _layer14_demo_phase6_wave7_rules(S: float) -> tuple[FFNRule, ...]:
    """One :class:`FFNRule` proving the declare-only path end to end.

    The rule is byte-identically a no-op:

    * ``conditions=(("CONST", -100.0),)`` -- ``CONST`` is always 1.0, so the
      pre-SiLU activation is ``-100 * S`` for every position. ``SiLU(-100*S)``
      is numerically 0 (``-1e-200`` territory) so the unit's contribution to
      ``W_down`` is zero.
    * ``writes=(("TEMP+0", 0.0),)`` -- even if SiLU were nonzero, the write
      weight is 0, so no residual cell is touched.
    * ``scope="__never_fires__"`` -- the declared firing scope is empty, so
      the F-7 ``verify_rule_scopes`` checker has nothing to validate against
      the live corpus.

    Trivially passes :func:`compare_symbolic_to_lowered_ffn` (both the
    symbolic execution and the lowered ``PureFFN`` forward return the input
    residual unchanged at every dim).
    """
    return (
        multi_way_and_rule(
            name="l14_demo_phase6_wave7_decl_only_noop",
            conditions=(("CONST", -100.0),),
            threshold=0.0,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("TEMP+0", 0.0),),
            scope="__never_fires__",
        ),
    )


def _layer14_demo_phase6_wave7_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_demo_phase6_wave7_rules(S))
    return ir


def make_layer14_demo_phase6_wave7_op() -> Operation:
    """L14 FFN demo: pure-declaration corrective op (Phase 6 Wave 7).

    Demonstrates the end-to-end declarative authoring path described in
    ``docs/HOW_TO_ADD_A_CORRECTIVE_OP.md``:

    * **Declare** -- one :class:`FFNRule` in
      :func:`_layer14_demo_phase6_wave7_rules`.
    * **Allocate** -- ``pin=None`` in
      :data:`_L14_CLEANUP_CHAIN_LAYOUT`. The chain helper runs
      :class:`FFNUnitAllocator` in first-fit mode and picks the lowest free
      gap past the prior chain claims (unit ``1886`` after the
      var-cluster JSR-path narrowing -- right after
      ``layer14_alu_nocarry_ax_bytes_zero``). The
      author never wrote the offset.
    * **Compile** -- :class:`CompilerIR` lowers the rule into a single FFN
      hidden unit via :meth:`CompilerIR.lower_ffn`.
    * **Byte-identity gate** -- the rule is byte-identically a no-op on the
      live corpus (see the docstring of
      :func:`_layer14_demo_phase6_wave7_rules`), so the residual stream is
      unchanged at every position. ``compare_symbolic_to_lowered_ffn`` is
      clean.

    Pinned to ``layer_idx=14`` via ``kind="block"``. The bake function is a
    slim wrapper around the chain allocator + :meth:`CompilerIR.lower_ffn`
    -- the body is identical (modulo names) to
    :func:`make_layer14_alu_nocarry_ax_bytes_zero_op`'s ``bake``, which is
    exactly the point of the demo: one IR edit + one layout-table entry +
    one boilerplate wrapper = a new corrective op.
    """

    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Phase 6 Wave 6D auto-fit: this op's layout entry uses
        # ``pin=None`` so the allocator picks the first free gap past
        # the prior chain claims. The rule list is invariant to where
        # in the chain the unit lands.
        start_unit = _l14_chain_alloc("layer14_demo_phase6_wave7")
        ir = _layer14_demo_phase6_wave7_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        ffn._l14_unit_counter = next_unit

    # No ``claims`` set: ``writes`` weight is 0 so no W_down cell is
    # actually touched. The op exists to demonstrate the flow; if a
    # future iteration of this demo emits a real cell write, the claim
    # tuples go here.
    return Operation(
        name="layer14_demo_phase6_wave7",
        # Phase 1 (memory cluster fix plan): shares L14 FFN unit range with
        # ``layer14_alu_nocarry_ax_bytes_zero`` at a disjoint sub-range
        # (this op owns unit 1886). See
        # docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("ffn_units",),
        reads={"CONST"},
        # Phase 9.B (TEMP SCC fix): drop dead TEMP write. The demo bake
        # emits a zero-weight rule that does not touch any W_down cell
        # (see docstring + verifier output). Was producing 2 back-edges
        # into the L14 cleanup loop.
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_demo_phase6_wave7_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Phase 7.A.5 default-flip: declare the chain predecessor as an
        # explicit ``requires["after"]`` so cycle-aware strict mode can
        # place this op via the dep DAG instead of the static ``phase``
        # pin. ``layer14_alu_nocarry_ax_bytes_zero`` is the immediate
        # prior entry in ``_L14_CLEANUP_CHAIN_LAYOUT`` (line 97). Without
        # this declaration the demo op has no in-edges in the strict dep
        # graph, so its dep-depth is 0 while its ``phase=14.95`` places
        # it at L14 -- the analyzer flags this gap as
        # ``phase_required_but_undeclared``, breaking the
        # ``test_strict_mode_categoriser_is_clean_outside_the_scc``
        # invariant and the strict-default admission gate.
        requires={"after": [
            "layer14_alu_nocarry_ax_bytes_zero",
            "layer14_mem_generation",
        ]},
        # New chain tail: prior ops fill [0, 1890), the demo's auto-fit
        # picks unit 1890 (single-unit rule), so the cumulative max is 1891.
        # Var-cluster JSR-path follow-up (2026-06-06) added units between
        # ``mem_addr_src_default_suppress`` and ``addr_key_neural_decode``
        # (``jsr_mem_default_suppress``); the same-day narrowing reduced
        # that from 8 to 4. Wave 1 Cluster B1 (2026-06-07) added
        # ``layer14_ent_ax_bytes_zero`` (4 units) after
        # ``alu_nocarry_ax_bytes_zero``, shifting demo from 1886 to 1890
        # → tail = 1891.
        ffn_units_used=1891,
        smoke_tests=set(),
        spec_section="docs/HOW_TO_ADD_A_CORRECTIVE_OP.md",
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
        multi_way_and_rule(
            name="l14_lc_ax_bytes_zero_lo_neg",
            writes=tuple((f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_lc_ax_bytes_zero_hi_neg",
            writes=tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
            ),
            **common_kwargs,
        ),
        multi_way_and_rule(
            name="l14_lc_ax_bytes_zero_lo0_boost",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
            **common_kwargs,
        ),
        multi_way_and_rule(
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
    # ``layer14_jsr_ax_bytes_zero`` (14.6) consumes units 1874..1877. The
    # helper writes 4 units mirroring jsr_ax_bytes_zero (shifted +4 by
    # var-cluster JSR-path follow-up's jsr_mem_default_suppress 2026-06-06;
    # initial +8 was narrowed to +4 the same day, see findings doc):
    #   unit 1878: -3/S on OUTPUT_LO[0..15]
    #   unit 1879: -3/S on OUTPUT_HI[0..15]
    #   unit 1880: +5/S on OUTPUT_LO[0]
    #   unit 1881: +5/S on OUTPUT_HI[0]
    _claims = set()
    for k in range(16):
        _claims.add((14, "ffn_W_down", "1878", f"OUTPUT_LO+{k}"))
        _claims.add((14, "ffn_W_down", "1879", f"OUTPUT_HI+{k}"))
    _claims.add((14, "ffn_W_down", "1880", "OUTPUT_LO+0"))
    _claims.add((14, "ffn_W_down", "1881", "OUTPUT_HI+0"))

    return Operation(
        name="layer14_lc_ax_bytes_zero",
        reads={"OP_LC_RELAY", "IS_BYTE", "H1", "BYTE_INDEX_3", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_lc_ax_bytes_zero_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        # Wave 1 (docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py``; mirrors
        # ``layer14_jsr_ax_bytes_zero``. Rule scope ``IS_BYTE and H1+1``
        # = AX byte positions -> AX_byte0 slot. OP_LC_RELAY is broadcast
        # to AX byte positions by L7 head 5 (V slot 2) in the same step,
        # so it's the fresh in-step value.
        claims=_claims,
        requires={"after": "layer14_mem_generation"},
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
    """Build a rule via ``multi_way_and_rule``, dispatching between the
    ``constant_write`` and ``gated_write`` shapes via the building-blocks
    DSL's internal ``_build_rule`` selector.

    When ``blocker_dim_name is None``, ``multi_way_and_rule`` falls back
    to ``FFNRule.constant_write`` (gate is None, gate_terms is empty),
    mirroring the imperative ``b_gate=1.0`` (no W_gate writes) form.

    When a blocker is set, the gate kwargs from
    :func:`_addr_key_gate_kwargs` provide ``gate=blocker``,
    ``gate_weight=-1.0`` and ``gate_bias=1.0`` — the
    ``W_gate[..., blocker]=-1.0`` blocker pattern used when a byte-3
    case routes through ``H3+4`` with ``H2+4`` blocked.
    """
    gate_kwargs = _addr_key_gate_kwargs(blocker_dim_name)
    if gate_kwargs is None:
        return multi_way_and_rule(
            name=name,
            conditions=conditions,
            threshold=threshold,
            writes=writes,
            scope=scope,
        )
    # Mirror the conditions-side MARK_MEM hard blocker into gate_terms so
    # the dim_alias_verifier's gate-fallback predicate (used when the
    # AND-of-condition-semantics is unsat) is tightened to exclude MEM
    # rows. Without this, the gate-fallback at byte_off=3 reduces to
    # the tautological H2+4 semantics and re-admits MEM rows. Runtime
    # impact at intended firing positions (MARK_MEM == 0) is zero.
    gate_terms = (
        (("MARK_MEM", -1e6),) if "MARK_MEM" in {c[0] for c in conditions} else ()
    )
    return multi_way_and_rule(
        name=name,
        conditions=conditions,
        threshold=threshold,
        writes=writes,
        scope=scope,
        gate_terms=gate_terms,
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
                        # Hard MARK_MEM blocker: at MEM marker rows
                        # ADDR_B0_LO/HI carry the *address* byte (aliased
                        # with OPCODE_BYTE_LO/HI), so the dispatch atom
                        # would read garbage. Byte-identical at intended
                        # firing positions (MARK_MEM == 0).
                        # dim_alias_verifier HARD_BLOCKER_THRESHOLD = 1e6.
                        ("MARK_MEM", -1e6),
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
                    # Hard MARK_MEM blocker — see _lo_hi_rules.
                    ("MARK_MEM", -1e6),
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
                        # Hard MARK_MEM blocker — see _lo_hi_rules.
                        ("MARK_MEM", -1e6),
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
        # Phase 8.A: matches layer14_mem_generation's ADDR_B0_HI_PREV_STEP
        # rename — same back-edge against L15 store_stack0_sp_byte0_addr,
        # same numeric position (slot 206), no functional change.
        # Phase 8.A follow-up: ADDR_B0_LO_PREV_STEP matches the HI pattern
        # — same L15 back-edge, same numeric slot, no functional change.
        reads={"MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "OP_LI_RELAY", "OP_LC_RELAY", "MARK_AX",
               "ADDR_B0_LO.*.-1", "ADDR_B0_HI.*.-1", "ADDR_B1_LO",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=op_compiler_ir,
        declarative_authority="spec_generated",
        # Phase 8.A.4: dropped ``layer_idx=14`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer14_mem_generation`` (the L14 attn op).
        target_op_name="layer14_mem_generation",
        migrated=True,
        requires={"after": "layer14_mem_generation"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )
