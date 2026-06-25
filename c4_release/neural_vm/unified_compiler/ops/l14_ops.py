"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import os as _os

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import (
    _as_setdim_proxy,
    ffn_lint_clean_demo_enabled,
    ffn_lint_mull14_demo_enabled,
    func_lea_b0_restore_enabled,
    no_stack0_emit_enabled,
    operand_from_memsp_enabled,
    sili_b1_restore_enabled,
    sub_full_borrow_enabled,
)
from ..positional_invariant import marker_bank_index
from .residual_band_registry import register_residual_band


def _li_zeroaddr_indicator_on() -> bool:
    """PHASE-2 KEYSTONE indicator dim for var_simple zero-address LI (#318).

    Materializes ONE FFN dim (``LI_ZEROADDR_COMMITTED``) that fires ≈1 ONLY on a
    store VALUE row that is BOTH **committed** (``MEM_STORE_AT_VAL`` ≈ 1) AND at a
    **zero address** (both ``ADDR_B0_LO+0`` and ``ADDR_B0_HI+0`` ≈ 1) — the
    3-way AND ``silu(MSAV + ADDR_B0_LO+0 + ADDR_B0_HI+0 - 2.5)``. L15 head-0 then
    keys K on this SINGLE dim (the AND lives in the FFN, not the bilinear head),
    so the committed BP+0 local's value row out-scores a non-committed
    zero-address operand-frame row WITHOUT the linear-K over-sharpening that
    desynced the AR frame (see ``l15_ops._l15_li_zeroaddr_cam_on`` BLOCKER).

    Shares the ``C4_L15_LI_ZEROADDR_CAM`` kill-switch with the head-0 keying so
    the indicator + the consumer are one coordinated campaign feature. The
    indicator reads ``ADDR_B0_LO/HI`` (gathered by L13 ``mem_addr_gather``) and
    ``MEM_STORE_AT_VAL`` (produced by L7 ``mem_store_relay``) — both settled by
    L13 output (present at the L14 input) — and writes the fresh
    ``LI_ZEROADDR_COMMITTED`` band, available at L15's input. ``MEM_STORE_AT_VAL``
    only exists on the campaign (``C4_OPERAND_FROM_MEMSP``) path, so this is
    campaign-only; golden (35-token, flag-OFF) is byte-identical (the op is not
    registered and the band is not collected).
    """
    from .shared import no_stack0_emit_enabled
    raw = _os.environ.get("C4_L15_LI_ZEROADDR_CAM")
    if raw is not None:
        return raw != "0"
    return no_stack0_emit_enabled()


# Fresh over-width residual band carrying the (committed AND zero-address)
# store indicator. ``never_share=True`` keeps it in a private slot (it is a
# value-row one-hot that L15 head-0 reads one block downstream). Flag-gated on
# the campaign keystone switch so a flag-OFF build omits it entirely (smaller
# d_model, golden byte-identical). Sized 1 (a single indicator scalar).
register_residual_band(
    "LI_ZEROADDR_COMMITTED", 1,
    owner="make_layer14_li_zeroaddr_indicator_op",
    flag=_li_zeroaddr_indicator_on, never_share=True,
)


# Fresh over-width residual band carrying the SUB full-borrow (minuend byte1==0
# AND borrow-in) indicator. ``never_share=True`` keeps it in a private slot: it
# is a bounded 0/1 flag written at an EARLY L14 block (where STACK0_BYTE_VAL_1
# is still fresh) and read MUCH later by the L25-tail 0xFF writer, so it must
# survive every intervening block at the SUB byte-1 emit row. Flag-gated on the
# campaign config so a golden (flag-OFF) build omits it entirely (smaller
# d_model, byte-identical). Sized 1 (a single indicator scalar).
register_residual_band(
    "SUB_FULL_BORROW", 1,
    owner="make_layer14_sub_full_borrow_flag_op",
    flag=sub_full_borrow_enabled, never_share=True,
)


# Fresh over-width residual band carrying the si/li 16-bit LOAD byte-1 value
# (Inc-2 part-c). ``never_share=True`` keeps it in a private slot: a 32-wide
# band (16 OUTPUT_LO + 16 OUTPUT_HI nibble cells) snapshotting the loaded AX
# byte-1 nibble one-hots at an EARLY L14 block (before the block-32/L18 OUTPUT_HI
# slam) and read at the L25 tail to RESTORE OUTPUT byte-1 after the slam. It must
# survive every intervening block at the LI-reload byte-1 predictor row.
# Flag-gated on the campaign config so a golden (flag-OFF) build omits it
# entirely (smaller d_model, byte-identical). Sized 32 (LO 0..15 then HI 0..15).
register_residual_band(
    "LI_RELOAD_B1", 32,
    owner="make_layer14_sili_b1_capture_op",
    flag=sili_b1_restore_enabled, never_share=True,
)


# Fresh over-width residual band carrying the func re-read-LEA byte-0 LOW nibble
# one-hot (Bug #2; DEFAULT-OFF building block). ``never_share=True`` keeps it in
# a private slot: a 16-wide band snapshotting the 16 ``OUTPUT_LO`` byte-0 nibble
# cells at the EARLY mem-addr anchor (before the block-42 L21 byte-0 default) and
# read at the L25 tail to RESTORE ``OUTPUT_LO`` byte-0. Only the LO nibble is
# carried — the HI nibble (0xE) is already correct. Flag-gated (DEFAULT-OFF) so a
# golden / flag-OFF build omits it entirely (smaller d_model, byte-identical).
# See shared.func_lea_b0_restore_enabled (incl. the func_identity zero-sum note).
register_residual_band(
    "LEA_REREAD_B0", 16,
    owner="make_func_lea_b0_capture_op",
    flag=func_lea_b0_restore_enabled, never_share=True,
)


def _psh_arg_val_ax_enabled() -> bool:
    """PSH-of-argument store-value AX-source lock (DEFAULT-ON; opt out =0).

    The L14/L18 mem-generation value heads (4-7) choose their value SOURCE
    via slots 1/2/36, gated by ``OP_JSR`` / ``OP_ENT`` (STACK0 source for
    JSR/ENT, AX source otherwise). On a function-argument PSH that precedes a
    JSR;ENT prologue, ``OP_ENT`` (and ``OP_JSR``) BROADCAST a small residue
    (~0.3) back onto the PSH step's MEM value row via the KV cache. That weak
    residue is enough to tip the source selector toward the STACK0 path, so
    head 4 attends a far-away garbage row (value 0x0A) instead of the AX
    byte-0 row (the pushed argument), corrupting the stored byte-0. SI/SC and
    normal-PSH stores have ``OP_JSR==OP_ENT==0`` so they already lock onto AX.

    This flag adds a hard, MEM_STORE-gated AX-source boost (a fresh attention
    slot) that is suppressed by the GENUINE opcode (``OP_JSR`` / ``OP_ENT`` ~
    10.8) but survives the broadcast residue (~0.3), so a PSH-arg store sources
    its value from AX like every other PSH/SI/SC store. Output-affecting; flag
    OFF reverts to byte-identical pre-fix weights.
    """
    return _os.environ.get("C4_PSH_ARG_VAL_AX", "1") != "0"


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
    # SUB no-borrow multi-byte minuend-byte1 passthrough (2026-06-12).
    # 16 units (one per minuend byte-1 nibble value): emits OUTPUT byte 1
    # = relayed minuend byte 1 (STACK0_BYTE_VAL_1) at the SUB byte-1
    # predictor row when there is NO byte-0 borrow (CARRY+2 absent).
    # Completes the multi-byte SUB result that the borrow-gated L14 carry
    # cascade cannot reach on the no-borrow path (sub_0 ``827-26``-class).
    # See ``make_layer14_sub_noborrow_high_byte_passthrough_op``. The
    # campaign (30-token) config adds 16 BORROW-path rules in the SAME op
    # (the L10 cascade byte-1 SUB cells are gated OFF on TEMP+9 there, so
    # this op owns the borrow byte 1 -- see the op's comment), so it claims
    # 32 units; GOLDEN claims 16 and is byte-identical. The two trailing
    # entries below are the demo fixtures (both DEFAULT OFF), so growing
    # this op never shifts a real op's start_unit.
    "layer14_sub_noborrow_high_byte_passthrough":
        (None, 32 if operand_from_memsp_enabled() else 16),
    # Phase 6 Wave 7 demo: pure-declaration corrective op. The op's single
    # rule is byte-identically a no-op on the live corpus -- it carries the
    # impossible condition ``CONST=-100`` so SiLU collapses to 0 and the
    # OUTPUT residual is unchanged -- so its sole purpose is to prove the
    # declare-only flow end to end. See ``make_layer14_demo_phase6_wave7_op``.
    "layer14_demo_phase6_wave7":             (None,    1),
    # Cross-op FFN-lint demo fixtures (tools/lint_cross_op_ffn.py --demo).
    # TOOLING-ONLY, both DEFAULT OFF (C4_FFN_LINT_MULL14_DEMO /
    # C4_FFN_LINT_CLEAN_DEMO). These are the LAST entries in the chain: when a
    # demo flag is off the op is not registered and ``_l14_chain_alloc`` never
    # claims its slot, so a flag-off / production build is byte-identical to
    # golden 4958b35b (the chain alloc loop breaks at the requested op, so a
    # trailing entry can never shift a real op's start_unit). See
    # ``make_ffn_lint_mull14_demo_op`` / ``make_ffn_lint_clean_demo_op``.
    "ffn_lint_mull14_demo":                  (None,    1),
    "ffn_lint_clean_demo":                   (None,    1),
    # PHASE-2 KEYSTONE (#318): the (committed AND zero-address) store indicator.
    # ONE FFN unit, campaign-flag-gated (C4_L15_LI_ZEROADDR_CAM). The LAST
    # entry in the chain so it can never shift a real op's start_unit: when the
    # flag is OFF the op is not registered and ``_l14_chain_alloc`` never claims
    # its slot -> a flag-off / golden build is byte-identical. When ON it
    # pre-claims every (deterministic, fixed-size) preceding layout slot and
    # lands at the first free gap. See ``make_layer14_li_zeroaddr_indicator_op``.
    "layer14_li_zeroaddr_indicator":         (None,    1),
    # SUB full-borrow (minuend byte1==0 AND borrow) flag precursor
    # (CAMPAIGN-ONLY, C4_SUB_FULL_BORROW). ONE FFN unit writing the private
    # ``SUB_FULL_BORROW`` band on the SUB byte-1 emit row. A chain TAIL entry so
    # it can never shift a real op's start_unit: flag-OFF the op is not
    # registered and ``_l14_chain_alloc`` never claims its slot (golden
    # byte-identical). Runs at the EARLY L14 block where STACK0_BYTE_VAL_1 is
    # still fresh (the band is cleared by the block-32 L18 slam), so the
    # empty-band underflow detector reads a live operand. See
    # ``make_layer14_sub_full_borrow_flag_op``.
    "layer14_sub_full_borrow_flag":          (None,    1),
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
    # Class-1 marker-relative anchors. ``H<k>+<marker>_I`` / ``L1H*+<marker>_I``
    # / ``L2H0+MEM_I`` are offsets into the fixed-width 7-slot threshold-head
    # bank keyed on marker TYPE (PC/AX/SP/BP/MEM/SE), NOT token distances. The
    # L0/L1/L2 ALiBi distance attention TRACKS each marker as the frame shrinks,
    # so these bank-slot indices are frame-INVARIANT. Resolve them through the
    # positional-invariant mechanism (single source of truth, declares the
    # invariance to the audit) instead of the literals ``PC_I=0 .. MEM_I=4``.
    PC_I = marker_bank_index("PC")
    AX_I = marker_bank_index("AX")
    SP_I = marker_bank_index("SP")
    BP_I = marker_bank_index("BP")
    MEM_I = marker_bank_index("MEM")

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

    # Class-1 marker-relative anchors (see _layer14_mem_generation_head_specs):
    # frame-invariant threshold-bank slot indices resolved through the
    # positional-invariant mechanism rather than the literals ``pc_i=0 ..
    # mem_i=4``. Byte-identical (the helper returns the same integers).
    pc_i = marker_bank_index("PC")
    ax_i = marker_bank_index("AX")
    sp_i = marker_bank_index("SP")
    bp_i = marker_bank_index("BP")
    mem_i = marker_bank_index("MEM")

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
        # Slot 35: MEM-source exclusion row (sign-stable). ``BD.H3 + mem_i`` is
        # the H3 distance-bank read at the MEM slot (Class-1 marker-relative;
        # the bare literal ``+ 4`` re-expressed through ``marker_bank_index``).
        q_map[(35, BD.CONST)] = 40.0
        k_map[(35, BD.MARK_MEM)] = -40.0
        k_map[(35, BD.H3 + mem_i)] = -40.0
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

            # === Slot 2 OP-residue floor: PSH-of-argument value-source AX ==
            # lock (C4_PSH_ARG_VAL_AX, DEFAULT-ON; HEAD 4 / val byte 0 only).
            #
            # Slot 2 is the value-head STACK0/JSR-ENT source selector: its Q
            # is ``L*(OP_JSR + OP_ENT)`` and its K pushes TOWARD the STACK0
            # source row (``+si_source_s`` on STACK0_BYTE0) and AWAY from the
            # AX source row (``-si_source_s`` on H1+AX). On a PSH it should be
            # fully OFF (the value comes from AX). It IS off for a normal PSH
            # (OP_JSR==OP_ENT==0). BUT a function-argument PSH that precedes a
            # JSR;ENT prologue gets an OP_JSR/OP_ENT broadcast residue (~0.3)
            # on its MEM value row (KV-cache bleed from the later ENT step):
            # Q_slot2 = 15*0.34 = 5.1 > 0, a weak STACK0 pull that tips head 4
            # off the AX byte-0 row (the pushed argument) onto a far-away
            # garbage STACK0/JSR-ENT val row -> stored byte-0 = 0x0A.
            #
            # FIX: subtract a CONST FLOOR from Q_slot2 so the STACK0 source
            # only activates when the opcode is GENUINE (OP_JSR+OP_ENT above
            # the threshold), not on broadcast residue. With FLOOR = L*2 = 30
            # (threshold at OP sum = 2.0, between the 0.34 residue and the
            # 10.79 real opcode):
            #   PSH-arg (0.34): Q_slot2 = 5.1 - 30 = -24.9 -> slot 2 now
            #     pushes AWAY from STACK0 and TOWARD the AX source row (the K
            #     sign-flip: -24.9 * -si_source_s = + on the AX row), so head
            #     4 locks onto AX byte 0 (the argument). Other slots' per-step
            #     position gates (33/34/38) keep it on THIS step's AX row.
            #   real JSR (10.79): Q_slot2 = 162 - 30 = 132 (STACK0, correct).
            #   real ENT (12.07): Q_slot2 = 181 - 30 = 151 (STACK0, correct).
            #   normal PSH / SI / SC (OP sum = 0): Q_slot2 = -30 -> a uniform
            #     negative on the STACK0 row, +30*si_source_s toward AX. SI/SC
            #     source from AX anyway (byte-identical output); PSH likewise.
            # Scoped to head 4 (byte-0 = the documented gap, the bulk of small
            # arg values). Flag OFF: the floor is omitted (byte-identical).
            if _psh_arg_val_ax_enabled() and own_value_idx == 0:
                # === Slot 46: PSH-of-argument ENT-source cancel ========
                # The DOMINANT corruptor of the call-argument PSH store value
                # is slot 44 (the ENT old-BP value-SOURCE override): its Q is
                # gated only on OP_ENT, its K selects the JSR-prologue old-BP
                # position via K@OP_JSR. The existing broadcast hardening
                # blocks only REGISTER-emit rows (MARK_PC/AX/SP/BP), leaving
                # the MEM value-target row unguarded. On a function-argument
                # PSH that precedes a JSR;ENT prologue, the later ENT's OP_ENT
                # BROADCASTS a ~0.30 residue onto the PSH val row (KV-cache
                # bleed), so Q_slot44 = 80*0.30 = 24, and multiplied by the
                # prologue row's GENUINE OP_JSR=10.79 (K_slot44 ~ 1482) gives
                # a +35580 pull onto the garbage old-BP row (value 0x0A),
                # burying the AX byte-0 row (the pushed argument).
                #
                # We cancel slot 44 ONLY on the genuine JSR-prologue rows
                # (K46 @ -OP_JSR; OP_JSR is large ONLY on real JSR rows and
                # is exactly 0 in non-call programs, so SI/SC/normal-PSH
                # contexts are untouched -- K46 ~ 0 there). Q46 = MEM_STORE-
                # gated, sign-flipped by OP_ENT at threshold ~2.0 (CANCEL_W /
                # ENT_SUPPRESS = 2.0): on a PSH store (OP_ENT=0.30) Q46 ~
                # +42.5 -> Q46*K46 ~ -36.7k cancels the +35.6k slot-44 pull,
                # letting the clean slot-1/36 AX selection win; on a GENUINE
                # ENT store (OP_ENT=10.79) Q46 ~ -382 flips sign so
                # Q46*K46 = (neg)*(neg) REINFORCES slot 44 (the ENT old-BP
                # source stays authoritative). Scoped to head 4 (byte-0 = the
                # documented gap). Flag OFF omits the slot (byte-identical).
                CANCEL_W = 50.0          # MEM_STORE-gated cancel strength
                ENT_SUPPRESS = CANCEL_W / 2.0  # sign-flip at OP_ENT = 2.0
                q_map[(46, BD.MEM_STORE)] = CANCEL_W
                q_map[(46, BD.OP_ENT)] = -ENT_SUPPRESS
                # K negates the slot-44 ENT-prologue selector (OP_JSR only:
                # the cleanest JSR-prologue discriminator; 0 in non-call
                # programs). Magnitude matches slot 44's ent_old_bp_s so the
                # cancel tracks the slot-44 pull it counters.
                k_map[(46, BD.OP_JSR)] = -ent_old_bp_s

                # === Slot 47: AX byte-0 in-step boost ===================
                # With slot 44's cross-step garbage pull cancelled (slot 46),
                # the residual competition is in-step: the AX byte-0 source
                # row (the pushed argument) and the step's frame/SP byte rows
                # (e.g. 0xF0) score within ~300. Add a modest, MEM_STORE-gated
                # boost onto the AX byte-0 row (K @ H1+AX one-hot AND
                # BYTE_INDEX_0, both = 1 on that row -> K = 2) to tip it over,
                # suppressed by GENUINE OP_JSR/OP_ENT so JSR/ENT stores (which
                # source from STACK0) push AWAY from the AX row instead. PSH/
                # SI/SC (OP sum <= ~0.34): Q ~ +1864 -> +3.7k onto AX byte-0
                # (wins). Real JSR (10.79): Q ~ -2316; real ENT (12.07): Q ~
                # -2828 -> the boost vanishes / repels (correct). The boost is
                # value-content-blind (K targets the AX-byte-0 POSITION), so
                # ALiBi recency keeps the CURRENT step's AX row over earlier
                # ones; SI/SC already pick AX (byte-identical output).
                AX_BOOST_W = 2000.0
                AX_BOOST_OP_SUPPRESS = 400.0  # sign-flip at OP sum = 5.0
                q_map[(47, BD.MEM_STORE)] = AX_BOOST_W
                q_map[(47, BD.OP_JSR)] = -AX_BOOST_OP_SUPPRESS
                q_map[(47, BD.OP_ENT)] = -AX_BOOST_OP_SUPPRESS
                k_map[(47, BD.H1 + ax_i)] = 1.0
                k_map[(47, BD.BYTE_INDEX_0)] = 1.0

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
    if no_stack0_emit_enabled() and _os.environ.get("C4_MUL_BLK33_CLAWBACK") == "1":
        # === DEMO REGRESSION (re-applied ba06deaa, 2026-06-20) ===
        # This is the salvaged-but-UNVERIFIED mul byte-1 "claw-back": it
        # hard-excludes the MARK_AX marker row from being a Q-firing row of
        # the high-byte relay, intending to stop a spurious marker-row write
        # for MUL in the 30-token campaign frame. It is byte-identical golden
        # (this block only exists in the campaign config AND only when the
        # C4_MUL_BLK33_CLAWBACK kill-switch is set), but it REGRESSES
        # add/sub/div in the campaign config (the marker-row exclusion starves
        # their byte-1 relay) — the EXACT flag-OFF-clean-yet-campaign-regression
        # blind spot tools/flag_regression_gate.py exists to catch. Gated behind
        # its own kill-switch so the gate can toggle it ON vs OFF.
        q.append(AP(0, BD.MARK_AX, -100000.0))
        q.append(AP(1, BD.MARK_AX, -100000.0))
        q.append(AP(33, BD.MARK_AX, -100000.0))
        q.append(AP(35, BD.MARK_AX, -100000.0))
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


def _layer14_mem_addr_src_default_suppress_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_mem_addr_src_default_suppress``.

    8 ungated (``constant_write``) hidden units that cancel the L3
    ``MEM DEFAULT`` +0.940 baseline at SI/SC store positions
    (``MEM_ADDR_SRC=1``). Mirrors the imperative helper cell-for-cell:

      * 2 marker units (LO/HI nibble): fire when MARK_MEM=1 AND
        MEM_ADDR_SRC=1. Imperative ``W_up[MARK_MEM]=S`` +
        ``W_up[MEM_ADDR_SRC]=S`` + ``b_up=-S*1.5`` ->
        ``conditions=(("MARK_MEM",1.0),("MEM_ADDR_SRC",1.0))``,
        ``threshold=1.5``.
      * 6 byte units (LO/HI nibble for BYTE_INDEX_{0,1,2}): fire when
        H1[MEM]=1 AND BYTE_INDEX_K=1 AND MEM_ADDR_SRC=1. Imperative
        ``W_up[H1+MEM_I]=S`` + ``W_up[byte_idx]=S`` +
        ``W_up[MEM_ADDR_SRC]=S`` + ``b_up=-S*2.5`` -> three positive
        conditions, ``threshold=2.5``.

    Each unit writes ``-2.0/S`` into the LO/HI nibble's OUTPUT cell
    (``W_down[OUTPUT_*+0, unit] = -2.0/S``). The legacy helper used the
    ungated form (``b_gate=1.0``, no ``W_gate`` content dim) -> the
    ``gate=None`` (``constant_write``) path here, which lowers
    ``b_gate=1.0`` and leaves ``W_gate`` zero, byte-identically.
    """
    MEM_I = 4  # MEM marker index in MARKS array
    WRITE = -2.0 / S
    rules: list[FFNRule] = []

    # Marker rule (MARK_MEM AND MEM_ADDR_SRC), LO then HI nibble.
    for band in ("OUTPUT_LO", "OUTPUT_HI"):
        rules.append(
            multi_way_and_rule(
                name=f"l14_mem_addr_src_suppress_marker_{band.lower()}",
                conditions=(
                    ("MARK_MEM", 1.0),
                    ("MEM_ADDR_SRC", 1.0),
                ),
                threshold=1.5,
                writes=((f"{band}+0", WRITE),),
                scope="MARK_MEM and MEM_ADDR_SRC",
            )
        )

    # Byte rules (H1[MEM] AND BYTE_INDEX_K AND MEM_ADDR_SRC), LO then HI.
    for byte_k in (0, 1, 2):
        for band in ("OUTPUT_LO", "OUTPUT_HI"):
            rules.append(
                multi_way_and_rule(
                    name=f"l14_mem_addr_src_suppress_byte{byte_k}_{band.lower()}",
                    conditions=(
                        (f"H1+{MEM_I}", 1.0),
                        (f"BYTE_INDEX_{byte_k}", 1.0),
                        ("MEM_ADDR_SRC", 1.0),
                    ),
                    threshold=2.5,
                    writes=((f"{band}+0", WRITE),),
                    scope=f"H1 and BYTE_INDEX_{byte_k} and MEM_ADDR_SRC",
                )
            )
    return tuple(rules)


def _layer14_mem_addr_src_default_suppress_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_mem_addr_src_default_suppress_rules(S))
    return ir


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
        ffn = block.ffn
        start_unit = _l14_chain_alloc("layer14_mem_addr_src_default_suppress")
        ir = _layer14_mem_addr_src_default_suppress_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(ffn, dim_map, start_unit=start_unit, S=S)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). The 8 units start at the
    # chain offset and are arranged LO/HI per (marker, byte0, byte1, byte2);
    # every unit writes its band's OUTPUT_*+0 cell.
    _claims = set()
    _start = _l14_chain_alloc("layer14_mem_addr_src_default_suppress")
    for _i, _band in enumerate(
        ("OUTPUT_LO", "OUTPUT_HI",
         "OUTPUT_LO", "OUTPUT_HI",
         "OUTPUT_LO", "OUTPUT_HI",
         "OUTPUT_LO", "OUTPUT_HI")
    ):
        _claims.add((14, "ffn_W_down", str(_start + _i), f"{_band}+0"))

    return Operation(
        name="layer14_mem_addr_src_default_suppress",
        phase=14.45,
        reads={"MARK_MEM", "H1", "BYTE_INDEX_0", "BYTE_INDEX_1",
               "BYTE_INDEX_2", "MEM_ADDR_SRC", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        # Phase 7.C migration (this wave): the bake now lowers the
        # declarative rule program ``_layer14_mem_addr_src_default_suppress_ir``
        # through ``CompilerIR.lower_ffn`` instead of calling the imperative
        # ``vm_step._set_layer14_mem_addr_src_default_suppress`` helper. The
        # rules (counter-write against the L3 ``MEM DEFAULT`` baseline gated
        # by ``MEM_ADDR_SRC=1``) are byte-identical to the legacy writes --
        # proven by ``tools/verify_l14_mem_default_suppress_migration.py``.
        # ``spec_generated`` now matches the sibling cleanup-chain ops that
        # carry an explicit ``compiler_ir=`` rule bundle.
        declarative_authority="spec_generated",
        compiler_ir=_layer14_mem_addr_src_default_suppress_ir(),
        claims=_claims,
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


def _layer14_jsr_mem_default_suppress_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule program for ``_set_layer14_jsr_mem_default_suppress``.

    4 ungated (``constant_write``) hidden units that cancel the L3
    ``MEM DEFAULT`` +0.940 baseline at the PSH/JSR/ENT store path
    (``MEM_STORE=1 AND MEM_ADDR_SRC=0``). Mirrors the imperative helper
    cell-for-cell:

      * 2 marker units (LO/HI nibble): fire when MARK_MEM=1 AND
        MEM_STORE=1 AND MEM_ADDR_SRC=0. Imperative ``W_up[MARK_MEM]=S`` +
        ``W_up[MEM_STORE]=S`` + ``W_up[MEM_ADDR_SRC]=-S`` + ``b_up=-S*1.5``
        -> ``conditions=(("MARK_MEM",1.0),("MEM_STORE",1.0),
        ("MEM_ADDR_SRC",-1.0))``, ``threshold=1.5``.
      * 2 byte units (LO/HI nibble for BYTE_INDEX_0 only -- the narrowed
        scope): fire when H1[MEM]=1 AND BYTE_INDEX_0=1 AND MEM_STORE=1 AND
        MEM_ADDR_SRC=0. Imperative adds ``W_up[H1+MEM_I]=S`` +
        ``W_up[BYTE_INDEX_0]=S`` + ``b_up=-S*2.5`` -> four conditions,
        ``threshold=2.5``.

    The ``MEM_ADDR_SRC=-1.0`` condition (negative weight, ``W_up=-S``)
    is what makes the gate ``MEM_STORE AND NOT MEM_ADDR_SRC`` -- it
    drives the SiLU below threshold for SI/SC (MEM_ADDR_SRC=1). Each
    unit writes ``-2.0/S`` into the LO/HI nibble's OUTPUT cell.
    The legacy helper used the ungated form (``b_gate=1.0``, no
    ``W_gate`` content dim) -> the ``gate=None`` (``constant_write``)
    path here, byte-identically.
    """
    MEM_I = 4  # MEM marker index in MARKS array
    WRITE = -2.0 / S
    rules: list[FFNRule] = []

    # Marker rule (MARK_MEM AND MEM_STORE AND NOT MEM_ADDR_SRC), LO/HI.
    for band in ("OUTPUT_LO", "OUTPUT_HI"):
        rules.append(
            multi_way_and_rule(
                name=f"l14_jsr_mem_suppress_marker_{band.lower()}",
                conditions=(
                    ("MARK_MEM", 1.0),
                    ("MEM_STORE", 1.0),
                    ("MEM_ADDR_SRC", -1.0),
                ),
                threshold=1.5,
                writes=((f"{band}+0", WRITE),),
                scope="MARK_MEM and MEM_STORE and not MEM_ADDR_SRC",
            )
        )

    # Byte rule (H1[MEM] AND BYTE_INDEX_0 AND MEM_STORE AND NOT
    # MEM_ADDR_SRC), LO/HI. Narrowed to BYTE_INDEX_0 (MEM_addr1) only.
    for band in ("OUTPUT_LO", "OUTPUT_HI"):
        rules.append(
            multi_way_and_rule(
                name=f"l14_jsr_mem_suppress_byte0_{band.lower()}",
                conditions=(
                    (f"H1+{MEM_I}", 1.0),
                    ("BYTE_INDEX_0", 1.0),
                    ("MEM_STORE", 1.0),
                    ("MEM_ADDR_SRC", -1.0),
                ),
                threshold=2.5,
                writes=((f"{band}+0", WRITE),),
                scope="H1 and BYTE_INDEX_0 and MEM_STORE and not MEM_ADDR_SRC",
            )
        )
    return tuple(rules)


def _layer14_jsr_mem_default_suppress_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_jsr_mem_default_suppress_rules(S))
    return ir


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
        ffn = block.ffn
        start_unit = _l14_chain_alloc("layer14_jsr_mem_default_suppress")
        ir = _layer14_jsr_mem_default_suppress_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(ffn, dim_map, start_unit=start_unit, S=S)
        ffn._l14_unit_counter = next_unit

    # Dim-ownership claims (W_down output cells). 4 units: LO/HI for the
    # MEM marker rule then LO/HI for the BYTE_INDEX_0 rule.
    _claims = set()
    _start = _l14_chain_alloc("layer14_jsr_mem_default_suppress")
    for _i, _band in enumerate(
        ("OUTPUT_LO", "OUTPUT_HI", "OUTPUT_LO", "OUTPUT_HI")
    ):
        _claims.add((14, "ffn_W_down", str(_start + _i), f"{_band}+0"))

    return Operation(
        name="layer14_jsr_mem_default_suppress",
        phase=14.46,
        reads={"MARK_MEM", "H1", "BYTE_INDEX_0",
               "MEM_STORE", "MEM_ADDR_SRC", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        # Phase 7.C migration (this wave): the bake now lowers the
        # declarative rule program ``_layer14_jsr_mem_default_suppress_ir``
        # through ``CompilerIR.lower_ffn`` instead of calling the imperative
        # ``vm_step._set_layer14_jsr_mem_default_suppress`` helper. The rules
        # (counter-write against the L3 ``MEM DEFAULT`` baseline gated by
        # ``MEM_STORE=1 AND NOT MEM_ADDR_SRC`` for PSH/JSR/ENT) are
        # byte-identical to the legacy writes -- proven by
        # ``tools/verify_l14_mem_default_suppress_migration.py``.
        declarative_authority="spec_generated",
        compiler_ir=_layer14_jsr_mem_default_suppress_ir(),
        claims=_claims,
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
    # CAMPAIGN re-anchor (var_three step-0 OUTPUT_HI leak, 2026-06-21):
    # var_three step 0 IS a JSR (the call into main); AX must stay 0 (0x0000).
    # The L7 head-5 OP_JSR broadcast onto the AX byte rows is a token-distance
    # RAMP. In the golden 35-token frame OP_JSR lands ~6.6 on the AX byte rows,
    # so this AX-zero floor's silu(OP_JSR*1.0) gate fires hard and pins
    # OUTPUT_HI+0 (byte-1 high nibble -> 0). When C4_NO_STACK0_EMIT drops the
    # 5-token STACK0 block (35->30), the ramp shifts so the AX byte rows fall in
    # the broadcast DIP: OP_JSR collapses to ~0.2 there (GPU-probed, spec_k=0:
    # AX byte rows 0.239/0.211/0.201/0.191, vs >5 on every OTHER marker's byte
    # rows). silu(0.2)=0.13 -> the floor barely fires -> a competing block-32
    # OUTPUT_HI broadcast wins high-nibble 1 -> AX=0x1000. This is the campaign
    # POSITIONAL-SHIFT bug class. Re-anchor: amplify the OP_JSR gate WEIGHT in
    # the 30-token frame so silu(OP_JSR*W) is strong at the dipped AX byte rows
    # (0.19*40=7.6 -> silu~7.6, matching golden's ~6.6) while staying exactly 0
    # where OP_JSR==0 (silu(0)==0) so NO non-JSR row is touched. The unit's
    # IS_BYTE+H1+1 W_up scope already confines firing to AX byte rows, so the
    # SP/BP/MEM rows (where OP_JSR is strong) are W_up-blocked regardless.
    # Flag-OFF (golden 35-tok) keeps gate_weight=1.0 -> BYTE-IDENTICAL.
    jsr_gate_weight = 40.0 if no_stack0_emit_enabled() else 1.0
    common_kwargs = dict(
        conditions=common_conditions,
        threshold=1.5,
        gate=gate_jsr,
        gate_weight=jsr_gate_weight,
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


# === L14 SUB no-borrow multi-byte minuend-byte1 passthrough =============
# (2026-06-12) Mirrors the relay+cascade SUB fix (commit b7a3064a /
# fe82f8c0) for the NO-BORROW path the borrow-gated cascade cannot reach.
#
# Root (spec_k=0): the L14 inter-byte borrow cascade
# (``_l10_carry_propagation_rules.sub_rule_for``) computes byte 1 of a
# multi-byte SUB as ``minuend_byte1 - 1`` and is gated on the byte-0
# borrow-out (CARRY+2 at the byte-1 predictor row). When byte 0 does NOT
# underflow (e.g. ``827 - 26`` -> ``0x3B - 0x1A = 0x21``, no borrow), the
# cascade SUB cells never fire, so OUTPUT byte 1 stays at the 0x00 default
# instead of ``minuend_byte1`` (the subtrahend's byte 1 is 0x00 for every
# 1096 sub case, so ``result_byte1 = minuend_byte1 - 0 - 0``). That drops
# the whole high byte -- ``827-26`` decodes as ``0x21`` not ``0x321``.
#
# ``layer13_sub_minuend_relay`` (L13 head 4) already delivers the pushed
# minuend's byte 1 into STACK0_BYTE_VAL_1 at the SUB byte-1 predictor row
# (BYTE_INDEX_0 + TEMP+9). This op reads that relayed value and, ONLY when
# there is no byte-0 borrow (CARRY+2 absent), writes OUTPUT byte 1 =
# STACK0_BYTE_VAL_1. The minuend byte 1 is <= 0x07 for the whole corpus
# (operands < 1000), so it fits the low nibble and OUTPUT_HI byte 1 = 0.
#
# Discriminator verified spec_k=0 (block 15, SUB byte-1 predictor row):
#   * no-borrow (827-26, 50-8):     CARRY+2 = 0.0  -> this op fires
#   * borrow    (1537-87, 256-1, 0-1): CARRY+2 = 2.0 -> blocked (cascade)
# So the borrow path (sub_16bit / sub_borrow / borrow-class 1096) is
# untouched -- the cascade keeps owning it. For 8-bit SUB the relayed
# minuend byte 1 is 0x00, so the write is byte-identical to the 0x00
# default (sub_basic 50-8 -> 42 unchanged). The op gates on TEMP+9 (the
# SUB byte-row selector, dark on ADD/everything else) so it never fires
# off-SUB.


def _layer14_sub_noborrow_high_byte_passthrough_rules(
    S: float,
) -> tuple[FFNRule, ...]:
    """16 rules: OUTPUT byte 1 = relayed minuend byte 1 on no-borrow SUB.

    One rule per minuend-byte1 nibble value ``v`` (0..15). Each fires at
    the SUB byte-1 predictor row -- ``TEMP+9`` (SUB byte selector) +
    ``H1[AX]`` + ``IS_BYTE`` + ``BYTE_INDEX_0`` -- when ``STACK0_BYTE_VAL_1_LO
    == v`` and there is NO byte-0 borrow (``CARRY+2`` blocked). The single
    firing rule cancels OUTPUT_LO/HI byte 1 (the 0x00 default, cell 0 hot)
    and sets OUTPUT_LO cell ``v`` so the byte-``v`` token wins argmax.

    Byte-identity: ``v == 0`` (8-bit SUB) cancels every LO cell and
    re-boosts cell 0 -- the same 0x00 the default emits. ``v > 0`` is the
    corrective multi-byte path.
    """
    AX_I = 1
    BORROW_BLOCK = -10.0  # CARRY+2 borrow-out blocker (no-borrow gate)
    rules: list[FFNRule] = []
    for v in range(16):
        conditions = (
            ("IS_BYTE", 1.0),
            (f"H1+{AX_I}", 1.0),
            ("BYTE_INDEX_0", 1.0),
            (f"STACK0_BYTE_VAL_1_LO+{v}", 1.0),
            # No-borrow gate: byte-0 borrow-out rides CARRY+2 at this row.
            ("CARRY+2", BORROW_BLOCK),
        )
        # Cancel the 0x00 byte-1 default across both nibble bands, then
        # boost LO cell v (net +5/S like the sibling *_ax_bytes_zero ops)
        # and HI cell 0 so byte 1 = 0x0v.
        writes: list[tuple[str, float]] = []
        for k in range(16):
            writes.append((f"OUTPUT_LO+{k}", -3.0 / S))
            writes.append((f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S))
        writes.append((f"OUTPUT_LO+{v}", 8.0 / S))
        writes.append(("OUTPUT_HI_THIS_STEP+0", 8.0 / S))
        rules.append(
            multi_way_and_rule(
                name=f"l14_sub_noborrow_high_byte_v{v:x}",
                conditions=conditions,
                # All five real conditions on (one-hot) -> 5.0; missing any
                # drops below 4.5. CARRY+2 (-10) hard-blocks the borrow row.
                threshold=4.5,
                gate="TEMP+9",
                gate_weight=1.0,
                gate_bias=0.0,
                writes=tuple(writes),
                scope=(
                    "TEMP+9 and IS_BYTE and H1+1 and BYTE_INDEX_0 "
                    "and not CARRY+2"
                ),
            )
        )
    return tuple(rules)


def _layer14_sub_borrow_high_byte_passthrough_rules(
    S: float,
) -> tuple[FFNRule, ...]:
    """16 rules: OUTPUT byte 1 = (relayed minuend byte 1) - 1 on borrow SUB.

    The CAMPAIGN-only sibling of
    :func:`_layer14_sub_noborrow_high_byte_passthrough_rules`. In the
    30-token frame the L10 borrow cascade's byte_idx=0 SUB cells cannot
    discriminate the byte-1 high nibble (the L8 mem[SP] CAM delivers a
    ``+/-6`` LOW-nibble encoding that lands AFTER the cascade, and the L14
    step-boundary guard amplifies every cell so the HI match is washed
    out -> all matched-lo SUB cells spray OUTPUT_HI; 1537-87 -> 0x65AA).
    Those cascade cells are therefore gated OFF on the SUB byte-1 emit row
    (TEMP+9) in the campaign config (``_l10_carry_propagation_rules.
    sub_rule_for``), leaving the field clean for this op -- the same reason
    the ±0.08 no-borrow sibling works.

    One rule per minuend-byte1 nibble value ``v``: fires at the SUB byte-1
    predictor row (``TEMP+9`` + ``H1[AX]`` + ``IS_BYTE`` + ``BYTE_INDEX_0``)
    when ``STACK0_BYTE_VAL_1_LO == v`` AND there IS a byte-0 borrow
    (``CARRY+2`` present, +2.0). Emits OUTPUT byte 1 = ``(v - 1) & 0xFF``.
    The subtrahend's byte 1 is 0x00 for every 1096 sub case, so
    result_byte1 = minuend_byte1 - 1 on the borrow path. minuend byte1 is
    <= 0x07 (operands < 2000) so v-1 fits the low nibble and OUTPUT_HI
    byte 1 = 0 (v=0 -> 0xFF never occurs for a valid sub but is emitted
    correctly anyway).

    CAMPAIGN-only (gated by the IR builder on
    ``operand_from_memsp_enabled()``): GOLDEN keeps the cascade-owned
    borrow path byte-identical (these rules are not lowered there).
    """
    AX_I = 1
    # HARD borrow partition: the no-borrow sibling keys ``not CARRY+2`` with a
    # -10 blocker so borrow rows go deeply negative (no silu leak). We need the
    # MIRROR -- a HARD floor for no-borrow rows. CARRY+2 cannot be keyed
    # "absent" (it is 0 there), so floor with a -10 CONST baseline and lift the
    # borrow row back with CARRY+2*+10 (=+20 since CARRY+2=2.0).
    #   borrow    row: 4 (one-hots) + 20 (CARRY+2) - 10 (CONST) = +14  (fires)
    #   no-borrow row: 4 (one-hots) +  0          - 10 (CONST) =  -6  (dark,
    #     silu(-6-4.5)=silu(-10.5) ~= 0 -- no leak onto the no-borrow result)
    CONST_FLOOR = -10.0
    BORROW_REQ = 10.0  # CARRY+2 (=2.0) -> +20 lift on the borrow row
    rules: list[FFNRule] = []
    for v in range(16):
        new_v = (v - 1) & 0xFF
        new_lo = new_v & 0xF
        new_hi = (new_v >> 4) & 0xF
        conditions = (
            ("IS_BYTE", 1.0),
            (f"H1+{AX_I}", 1.0),
            ("BYTE_INDEX_0", 1.0),
            (f"STACK0_BYTE_VAL_1_LO+{v}", 1.0),
            # No-borrow HARD floor (CONST is always 1.0).
            ("CONST", CONST_FLOOR),
            # Borrow gate: byte-0 borrow-out rides CARRY+2 (=2.0) here.
            ("CARRY+2", BORROW_REQ),
        )
        writes: list[tuple[str, float]] = []
        for k in range(16):
            writes.append((f"OUTPUT_LO+{k}", -3.0 / S))
            writes.append((f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S))
        writes.append((f"OUTPUT_LO+{new_lo}", 8.0 / S))
        writes.append((f"OUTPUT_HI_THIS_STEP+{new_hi}", 8.0 / S))
        rules.append(
            multi_way_and_rule(
                name=f"l14_sub_borrow_high_byte_v{v:x}",
                conditions=conditions,
                # borrow row +14 >= 4.5 (fires); no-borrow row -6 (dark, no
                # silu leak -- the CONST floor is the load-bearing partition).
                threshold=4.5,
                gate="TEMP+9",
                gate_weight=1.0,
                gate_bias=0.0,
                writes=tuple(writes),
                scope=(
                    "TEMP+9 and IS_BYTE and H1+1 and BYTE_INDEX_0 "
                    "and CARRY+2"
                ),
            )
        )
    return tuple(rules)


def _layer14_sub_noborrow_high_byte_passthrough_ir(
    S: float = 100.0,
) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        _layer14_sub_noborrow_high_byte_passthrough_rules(S)
    )
    # Campaign-only borrow path (30-token frame): the L10 cascade byte-1 SUB
    # cells are gated OFF on TEMP+9 there, so this op owns the borrow byte 1.
    # GOLDEN keeps the cascade owner and is byte-identical (rules omitted).
    if operand_from_memsp_enabled():
        ir.layer(0).ffn.rules.extend(
            _layer14_sub_borrow_high_byte_passthrough_rules(S)
        )
    return ir


def make_layer14_sub_noborrow_high_byte_passthrough_op() -> Operation:
    """L14 FFN: emit OUTPUT byte 1 = relayed minuend byte 1 on no-borrow SUB.

    Completes the multi-byte SUB result on the no-borrow path that the
    borrow-gated L14 carry cascade cannot reach. Reads the minuend byte 1
    that ``layer13_sub_minuend_relay`` deposits into STACK0_BYTE_VAL_1 at
    the SUB byte-1 predictor row. See the module comment block above for
    the root cause, the CARRY+2 no-borrow discriminator, and the
    byte-identity property (8-bit SUB unchanged; borrow path owned by the
    cascade). 1096 ``sub`` cluster: covers the ~38 no-borrow multi-byte
    cases (e.g. ``sub_0: 827 - 26``).
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        start_unit = _l14_chain_alloc(
            "layer14_sub_noborrow_high_byte_passthrough"
        )
        ir = _layer14_sub_noborrow_high_byte_passthrough_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(
            ffn, dim_map, start_unit=start_unit, S=S,
        )
        _guard_l14_output_units_on_step_boundary(
            ffn, dim_positions, S, start_unit, next_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_sub_noborrow_high_byte_passthrough",
        slot_share=("ffn_units",),
        reads={"TEMP", "IS_BYTE", "H1", "BYTE_INDEX_0", "CARRY",
               "STACK0_BYTE_VAL_1_LO", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer14_sub_noborrow_high_byte_passthrough_ir(),
        declarative_authority="spec_generated",
        target_op_name="layer14_mem_generation",
        migrated=True,
        requires={"after": "layer14_mem_generation"},
        spec_section="BLOG_SPEC.md#multibyte-arithmetic",
    )


# === L14 SUB full-borrow (minuend byte1 == 0) multi-byte 0xFF completion ===
# (CAMPAIGN-ONLY, gated on ``sub_full_borrow_enabled``.) Closes the
# ``sub_borrow_cascade`` (``0 - 1 = 0xFFFFFFFF``) wall the per-byte cascade
# rules cannot reach in the 30-token campaign frame.
#
# See ``shared.sub_full_borrow_enabled`` for the full three-layer root. The
# discriminator (minuend byte1 == 0 WITH a byte-0 borrow) exists ONLY at the
# SUB byte-1 emit row (``TEMP+9`` + ``BYTE_INDEX_0`` + ``H1[AX]`` + ``CARRY+2``)
# and ONLY as the ABSENCE of a STACK0_BYTE_VAL_1 one-hot (the L8 mem[SP] CAM
# delivers only a NON-zero nibble; the v==0 band is empty). The
# STACK0_BYTE_VAL_1 band is CLEARED by block ~32, so the detector must run at an
# EARLY L14 block (where it is fresh) and write a persistent flag; the 0xFF
# writer (the L25-tail op below) reads that flag AFTER the block-32 (L18) slam.

_SUB_FULL_BORROW_FLAG_HIDDEN_DIM = 1  # single empty-band AND unit


def _layer14_sub_full_borrow_flag_rules(S: float = 100.0) -> tuple[FFNRule, ...]:
    """One FFNRule: ``SUB_FULL_BORROW = 1`` on the SUB byte-1 full-underflow row.

    Fires at the SUB byte-1 emit row (``TEMP+9`` SUB byte selector + ``IS_BYTE``
    + ``H1[AX]`` + ``BYTE_INDEX_0``) when there IS a byte-0 borrow (``CARRY+2``
    == 2.0) AND the minuend byte 1 == 0 (so the SUB byte-1 result is the 0xFF
    full-underflow, NOT ``minuend_byte1 - 1``).

    The L8 mem[SP] CAM encodes the minuend byte-1 nibble ``v`` as a SIGNED
    one-hot: ``+6`` at cell ``v`` and ``-6`` at cell 0 (spec_k=0, BUILT dims,
    campaign, block 18, SUB byte-1 row):
      * full underflow (``0-1``,    v=0): every cell ~0
      * non-zero byte1 (``0x100-1``, v=1): cell0 = -6, cell1 = +6
      * non-zero byte1 (``0x200-1``, v=2): cell0 = -6, cell2 = +6
    so a RAW SUM over all cells is ~0 for BOTH v=0 and v>=1 (the +6/-6 cancel).
    The clean discriminator is "no POSITIVE cell in 1..15": for v>=1 a +6 sits
    at cell ``v in 1..15``; for v=0 there is none (and the -6 lives at cell 0).
    So the empty-band veto weights cells 1..15 only (NOT cell 0), with enough
    magnitude that a single +6 sinks the silu below threshold.

    Discriminator outcome:
      * full underflow (``0-1``):     CARRY+2 = 2.0, no +6 in 1..15 -> FIRES
      * non-zero byte1 (``0x100/200-1``): +6 at cell v in 1..15      -> blocked
      * no borrow (``50-8``):         CARRY+2 = 0.0                  -> blocked
    """
    del S
    SV1_BLOCK = -2.0  # per-cell (1..15) blocker: a +6 nibble -> -12 (veto)
    conditions: list[tuple[str, float]] = [
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("BYTE_INDEX_0", 1.0),
        # byte-0 borrow-out rides CARRY+2 (=2.0) at the byte-1 row.
        ("CARRY+2", 0.5),
    ]
    # Empty-band detector: any lit STACK0_BYTE_VAL_1 nibble v>=1 (+6) vetoes the
    # flag. Cell 0 is EXCLUDED (it carries the -6 sign marker on EVERY v>=1, and
    # is 0 on the v=0 underflow -- including it would not discriminate).
    for k in range(1, 16):
        conditions.append((f"STACK0_BYTE_VAL_1_LO+{k}", SV1_BLOCK))
        conditions.append((f"STACK0_BYTE_VAL_1_HI+{k}", SV1_BLOCK))
    return (
        multi_way_and_rule(
            name="l14_sub_full_borrow_flag",
            conditions=tuple(conditions),
            # full underflow: 1+1+1+1 = 4.0 >= 3.5 (fires); a lit SV1 cell (+6)
            # subtracts 12 (dark); no-borrow loses the CARRY+2 term -> 3.0
            # < 3.5 (dark). TEMP+9 gates so it never fires off the SUB byte row.
            threshold=3.5,
            gate="TEMP+9",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("SUB_FULL_BORROW", 1.0),),
            scope=(
                "TEMP+9 and IS_BYTE and H1+1 and BYTE_INDEX_0 and CARRY+2 "
                "and not STACK0_BYTE_VAL_1"
            ),
        ),
    )


def _make_standalone_pure_ffn_post_op_bake(rules):
    """Return a ``bake(block, dim_positions, S)`` that appends a standalone
    ``PureFFN`` post_op lowering ``rules`` with a dim_map resolved entirely from
    the declarative ``dim_positions`` layout. Mirrors the L25-tail / flag-
    precursor bakes (``make_ax_byte23_dump_zero_op`` etc.)."""

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

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
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            _off = int(_nm.split("+", 1)[1]) if "+" in _nm else 0
            dim_map[_nm] = int(dim_positions[_base]) + _off
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    return bake


def _layer14_sub_full_borrow_flag_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_sub_full_borrow_flag_rules(S))
    return ir


def make_layer14_sub_full_borrow_flag_op() -> Operation:
    """Precursor FFN (L14 cleanup chain tail): writes the bounded
    ``SUB_FULL_BORROW`` flag (CAMPAIGN-ONLY, ``C4_SUB_FULL_BORROW``).

    Bakes ONE FFN unit INTO the L14 mem-generation block FFN (via
    ``_l14_chain_alloc``) — the EARLY block where ``STACK0_BYTE_VAL_1`` is still
    fresh (it is cleared by the block-32 L18 slam). The single rule fires on the
    SUB byte-1 full-underflow row and writes the private ``SUB_FULL_BORROW``
    band; the L25-tail 0xFF writer (``make_sub_full_borrow_byte1_ff_op``) reads
    the persisted flag AFTER the slam.

    Mirrors ``make_layer14_li_zeroaddr_indicator_op`` (the other campaign-gated
    chain-tail flag op): a TAIL entry in ``_L14_CLEANUP_CHAIN_LAYOUT`` so growing
    it can never shift a real op's start_unit; flag-OFF the op is not registered
    and ``_l14_chain_alloc`` never claims its slot -> golden byte-identical (the
    band is also flag-gated, so it is omitted from the layout entirely).
    """
    enabled = sub_full_borrow_enabled()

    def bake(block, dim_positions, S):
        if not enabled:
            return
        ffn = getattr(block, "ffn", None)
        if ffn is None or not hasattr(ffn, "W_up"):
            return
        start_unit = _l14_chain_alloc("layer14_sub_full_borrow_flag")
        ir = _layer14_sub_full_borrow_flag_ir(S)
        rules = ir.layer(0).ffn.rules
        assert len(rules) == _SUB_FULL_BORROW_FLAG_HIDDEN_DIM, (
            f"sub_full_borrow_flag rule-count drift: {len(rules)} "
            f"!= {_SUB_FULL_BORROW_FLAG_HIDDEN_DIM} (the chain layout reserves "
            f"{_SUB_FULL_BORROW_FLAG_HIDDEN_DIM} unit)"
        )
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(ffn, dim_map, start_unit=start_unit, S=S)
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_sub_full_borrow_flag",
        slot_share=("ffn_units",),
        reads={"TEMP", "IS_BYTE", "H1", "BYTE_INDEX_0", "CARRY",
               "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI"}
        if enabled else set(),
        writes={"SUB_FULL_BORROW"} if enabled else set(),
        kind="block",
        target_op_name="layer14_mem_generation",
        # Order after the li_zeroaddr indicator (the prior chain-tail flag op) so
        # the chain alloc pre-claims its deterministic slot, and after
        # mem_generation (the L14 attn op this block targets).
        requires={"after": [
            "layer14_li_zeroaddr_indicator",
            "layer14_mem_generation",
        ]} if enabled else {},
        declarative_bake_fn=bake,
        compiler_ir=_layer14_sub_full_borrow_flag_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        # Chain tail (after li_zeroaddr's unit 1925): this single-unit op lands
        # at unit 1926 in the campaign config, so the block must be sized to 1927
        # (a half-open [0,1927) pool). Only reserved when the flag is on; OFF =>
        # not registered, so li_zeroaddr's 1926 governs (golden byte-identical).
        ffn_units_used=1927 if enabled else 0,
        spec_section="BLOG_SPEC.md#multibyte-arithmetic",
    )


# --- The L25-tail 0xFF writer (reads the persisted SUB_FULL_BORROW flag) ----
# Runs AFTER ``tail_bit32_result_correction`` (the last OUTPUT writer before the
# LM head), so it DOMINATES the block-32 (L18) OUTPUT_HI slam additively. On the
# flag row it overwrites OUTPUT byte 1 = 0xFF: cancel the slam's OUTPUT_HI cell-0
# (the 0x00 hi default ~+664) and OUTPUT_LO cell-0, and boost OUTPUT_LO/HI cell
# 15 so both nibbles decode 0xF.

_SUB_FULL_BORROW_FF_HIDDEN_DIM = 1


def _sub_full_borrow_byte1_ff_rules() -> tuple[FFNRule, ...]:
    """One FFNRule: overwrite OUTPUT byte 1 = 0xFF on the persisted flag row.

    Fires where ``SUB_FULL_BORROW`` (written by the L14 precursor, persisted at
    the SUB byte-1 emit row) is lit. The write magnitudes must DOMINATE the
    block-32 (L18) slam, which left OUTPUT_HI cell-0 ~+664 and every other HI
    cell ~-434, OUTPUT_LO cells ~-348..-362. To flip the argmax of both nibbles
    to cell 15 we cancel the cell-0 hi peak and boost cell 15 in both bands.

    The balanced silu saturates to ~``S * (SUB_FULL_BORROW*FLAG_W - thr)`` on a
    fire (flag ~1.0). With ``FLAG_W = 6.0`` / ``thr = 4.5`` the silu output is
    ~``S*1.5 = 150``; the write weights below scale that to OUTPUT deltas of
    ~+1500 (cell 15) / ~-1500 (cell 0), decisively beating the ~+664/-434 slam.
    """
    FLAG_W = 6.0
    THRESHOLD = 4.5
    # OUTPUT delta ~= silu(~150) * WW. WW=10 -> ~+1500 / -1500.
    WW = 10.0
    writes = (
        # Low nibble -> 0xF: cancel cell 0, boost cell 15.
        ("OUTPUT_LO+0", -WW),
        ("OUTPUT_LO+15", WW),
        # High nibble -> 0xF: cancel the slam's cell-0 peak, boost cell 15.
        ("OUTPUT_HI+0", -WW),
        ("OUTPUT_HI+15", WW),
    )
    return (
        multi_way_and_rule(
            name="sub_full_borrow_byte1_ff",
            conditions=(("SUB_FULL_BORROW", FLAG_W),),
            threshold=THRESHOLD,
            writes=writes,
            scope="SUB_FULL_BORROW",
        ),
    )


def make_sub_full_borrow_byte1_ff_op() -> Operation:
    """L25-tail FFN: overwrite OUTPUT byte 1 = 0xFF on the SUB full-borrow row.

    Standalone ``PureFFN`` post_op on the L25 tail block, appended AFTER
    ``tail_bit32_result_correction`` so it is the LAST OUTPUT writer before the
    LM head and therefore beats the block-32 (L18) slam (which runs earlier).
    Reads the persisted ``SUB_FULL_BORROW`` flag. CAMPAIGN-ONLY: a no-op when
    ``sub_full_borrow_enabled`` is False (golden byte-identical).
    """
    if not sub_full_borrow_enabled():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="sub_full_borrow_byte1_ff",
            reads=set(),
            writes=set(),
            audited_empty_produces=True,
            kind="block",
            target_op_name="l10_post_ops_combined",
            requires={"after": "tail_bit32_result_correction"},
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            migrated=True,
            smoke_tests={"all"},
            spec_section="BLOG_SPEC.md#multibyte-arithmetic",
        )

    rules = _sub_full_borrow_byte1_ff_rules()
    assert len(rules) == _SUB_FULL_BORROW_FF_HIDDEN_DIM, (
        f"sub_full_borrow_byte1_ff rule-count drift: {len(rules)} "
        f"!= {_SUB_FULL_BORROW_FF_HIDDEN_DIM}"
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    return Operation(
        name="sub_full_borrow_byte1_ff",
        reads={"SUB_FULL_BORROW"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=_make_standalone_pure_ffn_post_op_bake(rules),
        compiler_ir=ir,
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#multibyte-arithmetic",
    )


# ===========================================================================
# si/li 16-bit LOAD byte-1 CAPTURE + RESTORE (Inc-2 part-c, C4_SILI_B1_RESTORE)
# ===========================================================================
#
# Mirrors the SUB_FULL_BORROW capture/restore precedent for a DATA-DEPENDENT
# byte (the loaded AX byte-1) rather than the constant 0xFF. Two standalone
# PureFFN ops, both campaign-gated on ``sili_b1_restore_enabled``:
#
#   * CAPTURE (``make_layer14_sili_b1_capture_op``): a post_op on the L13/L10
#     mem-addr anchor block (block 16, where the L10 head-1 slot-83 fix has just
#     delivered the loaded byte-1 into OUTPUT), BEFORE the block-32/L18 OUTPUT_HI
#     slam. 32 gate-copy rules snapshot the 16 OUTPUT_LO + 16 OUTPUT_HI byte-1
#     nibble cells into the private ``LI_RELOAD_B1`` band (cells 0..15 = LO,
#     16..31 = HI). The silu factor is UNIFORM across the 16 cells of each
#     nibble, so the byte-1 nibble argmax is preserved into the band.
#
#   * RESTORE (``make_sili_b1_restore_op``): a post_op on the L25 tail block
#     AFTER ``tail_bit32_result_correction`` (the LAST OUTPUT writer before the
#     LM head). 32 gate-write rules re-supply ``OUTPUT_{LO,HI}`` from the carried
#     ``LI_RELOAD_B1`` band at a DOMINANT magnitude (``_SILI_B1_RESTORE_WS``), so
#     the restored byte-1 out-votes the block-32 slam (~+18/-1700) additively.
#
# Both ops fire ONLY on the LI-reload byte-1 PREDICTOR row (the AX byte-0 row of
# an LI step): the discriminator AND is ``IS_BYTE + H1[AX]+1 + BYTE_INDEX_0 +
# Σ ADDR_B1`` MINUS hard ``OP_IMM`` / ``MARK_AX`` / ``MEM_STORE`` /
# ``BYTE_INDEX_1..3`` blockers. ``ADDR_B1`` (the gathered LOAD-address one-hot)
# is the LOAD-specific term: it is ~6.0 on an LI-reload byte-1 row and ~0 on
# PSH/SI/IMM register rows and on LEA-addressed var_simple loads, so the
# discriminator is mandatory-ADDR_B1 and the op never touches a non-LI-reload
# AX byte-1 row. On the 4 already-passing si/li cases (loaded byte-1 == 0x00)
# the band snapshots the 0x00 one-hot and the restore re-asserts 0x00 (no-op).

# Restore write scale: the slam left OUTPUT_HI cell-0 ~+18 and cells 1..15 ~-1700
# (and crushed OUTPUT_LO similarly at the L25 tail). The restore is ADDITIVE, so
# the re-supply must net POSITIVE at the loaded nibble cell. ``silu(S*(disc-thr))``
# is ~O(100s) on the firing row and ``LI_RELOAD_B1`` carries the (already
# silu-scaled) one-hot, so a modest WS makes the product dominate the slam.
_SILI_B1_RESTORE_WS = 20.0

# Discriminator weights (shared by capture + restore). ADDR_B1 is split into 32
# per-cell terms so any single active address-byte-1 nibble contributes; the sum
# is ~6 on an LI-reload row, 0 elsewhere.
_SILI_B1_DISC_BASE = (
    ("IS_BYTE", 1.0),
    ("H1+1", 1.0),
    ("BYTE_INDEX_0", 1.0),
    ("OP_IMM", -5.0),
    ("MARK_AX", -5.0),
    ("MEM_STORE", -5.0),
    ("BYTE_INDEX_1", -5.0),
    ("BYTE_INDEX_2", -5.0),
    ("BYTE_INDEX_3", -5.0),
)
# ADDR_B1 (32 cells, weight 1.0 each) makes the rule LOAD-context-specific: it
# is ~2..6 on an LI-reload/store byte-1 row and 0 on a fresh-IMM/PSH register row.
_SILI_B1_DISC_ADDR = tuple(
    (f"ADDR_B1_LO+{c}", 1.0) for c in range(16)
) + tuple(
    (f"ADDR_B1_HI+{c}", 1.0) for c in range(16)
)
# The DECISIVE 16-bit term: Σ OUTPUT_HI cells 1..15 (the nonzero-HIGH-nibble
# mass of the loaded byte-1). It is ~2 on a genuine 16-bit LI-reload (high nibble
# 0x1x) and ~0 on a byte-1 == 0x00 load (the high nibble sits in cell 0, which is
# EXCLUDED). Measured at the block-16 capture point BEFORE the block-32 slam:
# 16bit LI-reload Σ[1..15] = 2.0; every byte-1 == 0 load (roundtrip / zero /
# multiple / overwrite, single- AND multi-store) Σ[1..15] ~ 0. So this term makes
# the CAPTURE fire ONLY when the loaded byte-1 genuinely has a high nibble -- the
# exact rows the slam corrupts -- and leaves all already-passing si/li cases (and
# every store byte-1 row) untouched.
_SILI_B1_DISC_HINZ = tuple(
    (f"OUTPUT_HI+{c}", 1.0) for c in range(1, 16)
)
# CAPTURE conditions: base + ADDR_B1 (load context) + the OUTPUT_HI[1..15] 16-bit
# term. threshold tuned so a genuine 16-bit LI-reload (base ~3 + ADDR ~6 + HInz
# ~2 = ~11) clears it while a byte-1 == 0 load (base ~3 + ADDR ~6 + HInz ~0 = ~9)
# and any IMM-emit row (-5 OP_IMM) stay BELOW.
_SILI_B1_CAPTURE_CONDS = _SILI_B1_DISC_BASE + _SILI_B1_DISC_ADDR + _SILI_B1_DISC_HINZ
_SILI_B1_CAPTURE_THRESHOLD = 10.0
# RESTORE conditions: base + ADDR_B1 only (the OUTPUT_HI[1..15] term is unusable
# at the L25 tail -- the slam has already crushed it). The restore is self-gating
# instead: it gate-WRITES from the carried LI_RELOAD_B1 band, which is EMPTY (all
# 0) on every row except the genuine 16-bit LI-reload that the strict CAPTURE
# wrote. An empty band -> gate_terms == 0 -> the restore adds 0 to OUTPUT (a
# no-op), so it is harmless on byte-1 == 0 loads and store byte-1 rows even though
# the looser discriminator may fire there.
_SILI_B1_RESTORE_CONDS = _SILI_B1_DISC_BASE + _SILI_B1_DISC_ADDR
_SILI_B1_RESTORE_THRESHOLD = 5.0

_SILI_B1_CAPTURE_HIDDEN_DIM = 32   # 16 LO + 16 HI gate-copy units
_SILI_B1_RESTORE_HIDDEN_DIM = 32


def _sili_b1_capture_rules() -> tuple[FFNRule, ...]:
    """32 rules: ``LI_RELOAD_B1[j] = OUTPUT_{LO,HI}[nib]`` on the 16-bit LI row.

    For each nibble cell ``nib`` (0..15), one gate-copy unit writes the LOW
    nibble (``LI_RELOAD_B1+nib = silu(disc) * OUTPUT_LO+nib``) and one writes the
    HIGH nibble (``LI_RELOAD_B1+(16+nib) = silu(disc) * OUTPUT_HI+nib``). The
    ``gate_terms`` carry the per-cell ``OUTPUT_*`` value; the silu pre-activation
    is the shared CAPTURE discriminator (which requires a NONZERO high nibble via
    ``OUTPUT_HI[1..15]``), so the firing factor is identical for all 32 cells and
    the byte-1 nibble argmax is preserved into the band. On a byte-1 == 0 load the
    discriminator stays below threshold, so the band is left EMPTY (the restore is
    then a no-op there).
    """
    rules: list[FFNRule] = []
    for nib in range(16):
        rules.append(
            multi_way_and_rule(
                name=f"sili_b1_capture_lo_{nib}",
                conditions=_SILI_B1_CAPTURE_CONDS,
                threshold=_SILI_B1_CAPTURE_THRESHOLD,
                gate_terms=((f"OUTPUT_LO+{nib}", 1.0),),
                writes=((f"LI_RELOAD_B1+{nib}", 1.0),),
                scope="LI_RELOAD_B1",
            )
        )
    for nib in range(16):
        rules.append(
            multi_way_and_rule(
                name=f"sili_b1_capture_hi_{nib}",
                conditions=_SILI_B1_CAPTURE_CONDS,
                threshold=_SILI_B1_CAPTURE_THRESHOLD,
                gate_terms=((f"OUTPUT_HI+{nib}", 1.0),),
                writes=((f"LI_RELOAD_B1+{16 + nib}", 1.0),),
                scope="LI_RELOAD_B1",
            )
        )
    return tuple(rules)


def make_layer14_sili_b1_capture_op() -> Operation:
    """CAPTURE FFN: snapshot the loaded AX byte-1 into ``LI_RELOAD_B1``.

    Standalone ``PureFFN`` post_op on the L13/L10 mem-addr anchor block (block
    16), where the loaded byte-1 is fresh in OUTPUT and BEFORE the block-32 slam.
    CAMPAIGN-ONLY: a no-op when ``sili_b1_restore_enabled`` is False (the band is
    omitted and this op is not registered, golden ``7f6f2e5d`` byte-identical).
    """
    if not sili_b1_restore_enabled():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="layer14_sili_b1_capture",
            reads=set(),
            writes=set(),
            audited_empty_produces=True,
            kind="block",
            target_op_name="_layer13_mem_addr_anchor",
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            migrated=True,
            smoke_tests={"all"},
            spec_section="FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md",
        )

    rules = _sili_b1_capture_rules()
    assert len(rules) == _SILI_B1_CAPTURE_HIDDEN_DIM, (
        f"sili_b1_capture rule-count drift: {len(rules)} "
        f"!= {_SILI_B1_CAPTURE_HIDDEN_DIM}"
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    return Operation(
        name="layer14_sili_b1_capture",
        reads={"IS_BYTE", "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "OP_IMM", "MARK_AX", "MEM_STORE",
               "ADDR_B1_LO", "ADDR_B1_HI", "OUTPUT_LO", "OUTPUT_HI"},
        writes={"LI_RELOAD_B1"},
        kind="block",
        # Block 16 (the L10 head-1 byte-1 delivery / L13 mem-addr anchor host);
        # the loaded byte-1 is present in OUTPUT here, BEFORE the block-32 slam.
        target_op_name="_layer13_mem_addr_anchor",
        declarative_bake_fn=_make_standalone_pure_ffn_post_op_bake(rules),
        compiler_ir=ir,
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md",
    )


def _sili_b1_restore_rules() -> tuple[FFNRule, ...]:
    """32 rules: ``OUTPUT_{LO,HI}[nib] = LI_RELOAD_B1[j]`` on the LI-reload row.

    Re-supplies the captured loaded byte-1 nibbles into OUTPUT at a DOMINANT
    magnitude (``_SILI_B1_RESTORE_WS``) so the restored byte-1 out-votes the
    block-32 slam. The gate carries the carried-band cell. The conditions use the
    looser ADDR_B1 discriminator (the OUTPUT_HI 16-bit term is unusable post-slam);
    the restore is self-gating via the EMPTY ``LI_RELOAD_B1`` band -> a byte-1 == 0
    load (or a store byte-1 row) has an empty band -> the gate-write is 0 (no-op).
    """
    rules: list[FFNRule] = []
    for nib in range(16):
        rules.append(
            multi_way_and_rule(
                name=f"sili_b1_restore_lo_{nib}",
                conditions=_SILI_B1_RESTORE_CONDS,
                threshold=_SILI_B1_RESTORE_THRESHOLD,
                gate_terms=((f"LI_RELOAD_B1+{nib}", _SILI_B1_RESTORE_WS),),
                writes=((f"OUTPUT_LO+{nib}", 1.0),),
                scope="LI_RELOAD_B1",
            )
        )
    for nib in range(16):
        rules.append(
            multi_way_and_rule(
                name=f"sili_b1_restore_hi_{nib}",
                conditions=_SILI_B1_RESTORE_CONDS,
                threshold=_SILI_B1_RESTORE_THRESHOLD,
                gate_terms=((f"LI_RELOAD_B1+{16 + nib}", _SILI_B1_RESTORE_WS),),
                writes=((f"OUTPUT_HI+{nib}", 1.0),),
                scope="LI_RELOAD_B1",
            )
        )
    return tuple(rules)


def make_sili_b1_restore_op() -> Operation:
    """RESTORE FFN: re-supply the loaded AX byte-1 into OUTPUT after the slam.

    Standalone ``PureFFN`` post_op on the L25 tail block, appended AFTER
    ``tail_bit32_result_correction`` so it is the LAST OUTPUT writer before the
    LM head and therefore DOMINATES the block-32 (L18) slam additively. Reads the
    carried ``LI_RELOAD_B1`` band. CAMPAIGN-ONLY: a no-op when
    ``sili_b1_restore_enabled`` is False (golden ``7f6f2e5d`` byte-identical).
    """
    if not sili_b1_restore_enabled():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="sili_b1_restore",
            reads=set(),
            writes=set(),
            audited_empty_produces=True,
            kind="block",
            target_op_name="l10_post_ops_combined",
            requires={"after": "tail_bit32_result_correction"},
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            migrated=True,
            smoke_tests={"all"},
            spec_section="FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md",
        )

    rules = _sili_b1_restore_rules()
    assert len(rules) == _SILI_B1_RESTORE_HIDDEN_DIM, (
        f"sili_b1_restore rule-count drift: {len(rules)} "
        f"!= {_SILI_B1_RESTORE_HIDDEN_DIM}"
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    return Operation(
        name="sili_b1_restore",
        reads={"IS_BYTE", "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "OP_IMM", "MARK_AX", "MEM_STORE",
               "ADDR_B1_LO", "ADDR_B1_HI", "LI_RELOAD_B1"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=_make_standalone_pure_ffn_post_op_bake(rules),
        compiler_ir=ir,
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md",
    )


# ===========================================================================
# func re-read LEA byte-0 LO-nibble CAPTURE + RESTORE (Bug #2,
# C4_FUNC_LEA_B0_RESTORE)
# ===========================================================================
#
# See ``shared.func_lea_b0_restore_enabled`` for the GPU-confirmed root
# (``tools/_probe_lea_addr_trace.py``): with the L7 head-1 re-read re-sharpen
# (Bug #1) the re-read LEA's address byte-0 is computed correctly (OUTPUT_LO
# byte-0 argmax = LO nibble cell 0, ~810 through block 41), but a PureFFN in the
# L21 post-op chain (physical block 42) SLAMS OUTPUT_LO byte-0 to LO nibble cell
# 8 (+4.13e9, the prior LEA &a=0xE8's lo nibble stamped cross-step), emitting
# 0xE8 not 0xE0. The OUTPUT_HI byte-0 the slam leaves (HI nibble 0xE) is correct,
# so only the LO nibble is restored. The fix mirrors C4_SILI_B1_RESTORE (early
# CAPTURE into a private band + L25-tail RESTORE) but uses the ENT-AXCARRY
# winner-take-all RESTORE so it dominates the ~4.13e9 slam regardless of
# magnitude.

# Capture/restore ROW discriminator: the LEA AX-marker row. MARK_AX (100) +
# OP_LEA (60) -> +400 on a real LEA AX row (OP_LEA ~5.0); a clean non-LEA AX row
# (OP_LEA==0) -> 100. OP_IMM / MEM_STORE NOT-block the non-LEA cases; the non-AX
# marker rows are hard NOT-blocked so the op never touches a PC/SP/BP/STACK0/MEM
# row.
_FUNC_LEA_B0_ROW_DISC = (
    ("MARK_AX", 100.0),
    ("OP_LEA", 60.0),
    ("OP_IMM", -500.0),
    ("MEM_STORE", -500.0),
    ("MARK_PC", -1_000_000.0),
    ("MARK_SP", -1_000_000.0),
    ("MARK_BP", -1_000_000.0),
    ("MARK_STACK0", -1_000_000.0),
    ("MARK_MEM", -1_000_000.0),
)
# RESTORE per-cell winner-take-all magnitude. The firing-row restore write at a
# cell is ``DOM * silu(disc) * band_cell``; with ``silu(disc) ~ 1.45e5`` and the
# clean one-hot ``band_cell ~ 1.1e4`` the per-cell write is ``DOM * 1.6e9``. The
# block-42 slam crushes the captured cell to ``-4.13e9`` and lifts the wrong
# cell-8 to ``+4.13e9``, so dominance needs ``DOM * 1.6e9 - 4.13e9 >
# 4.13e9 - DOM * 1.6e9`` i.e. ``DOM > ~2.6``. ``DOM = 50`` clears it with ~20x
# headroom; it is safe to over-scale because the restore is gated by the CLEAN
# one-hot band (exactly one cell rule fires per captured LEA row, and the band is
# EMPTY on every non-captured row so nothing else is touched).
_FUNC_LEA_B0_DOM = 50.0
_FUNC_LEA_B0_HIDDEN_DIM = 16
# RESTORE row threshold: between a real LEA AX row (~400) and a clean non-LEA AX
# row (100). Mirrors the ENT-AXCARRY 145.
_FUNC_LEA_B0_RESTORE_THRESHOLD = 145.0

# CAPTURE cell-selection: at the capture block the freshly-computed LEA byte-0
# LO one-hot sits in OUTPUT_LO as ``809.6`` at the address nibble cell and
# ``609.6`` at every other cell (measured Bug-#1-only build, blocks 16..41,
# clean 200-margin separation). Each capture unit ANDs the LEA row discriminator
# with ``(OUTPUT_LO+k)`` weight 1.0; the threshold is set BETWEEN the argmax
# (809.6) and the runner-up (609.6) so ONLY the address nibble cell clears it.
# Row base (LEA) = MARK_AX*100 + OP_LEA*60(~5) = ~400; total at the argmax cell
# = 400 + 809.6 = ~1209.6, at a non-argmax cell = 400 + 609.6 = ~1009.6. The
# threshold 1100 cleanly separates them, so the band becomes a CLEAN one-hot
# (bounded indicator, no gate -> ``silu(disc)`` write at the argmax cell only).
_FUNC_LEA_B0_CAPTURE_THRESHOLD = 1100.0


def _func_lea_b0_capture_rules() -> tuple[FFNRule, ...]:
    """16 rules: clean one-hot ``LEA_REREAD_B0[k]`` at the LEA byte-0 nibble.

    Each unit ANDs the LEA row discriminator with ``OUTPUT_LO[k]`` (weight 1.0)
    and a threshold set between the argmax (809.6) and runner-up (609.6) LO band
    cells, so EXACTLY the address nibble cell clears it. The write is a bounded
    indicator (``silu(disc)`` into ``LEA_REREAD_B0+k``, no value-gate), giving a
    CLEAN one-hot band the winner-take-all restore can pick decisively. Fires on
    BOTH the first LEA (&a) and the re-read LEA (&b); each snapshots its OWN
    correct nibble (self-gating restore). On every non-LEA row the row
    discriminator keeps the score below threshold so the band is left EMPTY (the
    restore is then a no-op there).
    """
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(
            multi_way_and_rule(
                name=f"func_lea_b0_capture_{k}",
                conditions=_FUNC_LEA_B0_ROW_DISC + ((f"OUTPUT_LO+{k}", 1.0),),
                threshold=_FUNC_LEA_B0_CAPTURE_THRESHOLD,
                writes=((f"LEA_REREAD_B0+{k}", 1.0),),
                scope="LEA_REREAD_B0",
            )
        )
    return tuple(rules)


def make_func_lea_b0_capture_op() -> Operation:
    """CAPTURE FFN: snapshot the re-read LEA byte-0 LO nibble into LEA_REREAD_B0.

    Standalone ``PureFFN`` post_op on the block-16 ``_layer13_mem_addr_anchor``
    (the same early host SILI uses), where the LEA byte-0 LO one-hot is fresh in
    OUTPUT and BEFORE the block-42 slam. CAMPAIGN-ONLY: a no-op when
    ``func_lea_b0_restore_enabled`` is False (the band is omitted and this op is
    not registered, golden ``d0619711`` byte-identical).
    """
    if not func_lea_b0_restore_enabled():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="func_lea_b0_capture",
            reads=set(),
            writes=set(),
            audited_empty_produces=True,
            kind="block",
            target_op_name="_layer13_mem_addr_anchor",
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            migrated=True,
            smoke_tests={"all"},
            spec_section="FUNC_REREAD_LEA_BUG1_APPLIED_BUG2_BLK43_2026_06_25.md",
        )

    rules = _func_lea_b0_capture_rules()
    assert len(rules) == _FUNC_LEA_B0_HIDDEN_DIM, (
        f"func_lea_b0_capture rule-count drift: {len(rules)} "
        f"!= {_FUNC_LEA_B0_HIDDEN_DIM}"
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    return Operation(
        name="func_lea_b0_capture",
        reads={"MARK_AX", "OP_LEA", "OP_IMM", "MEM_STORE", "MARK_PC", "MARK_SP",
               "MARK_BP", "MARK_STACK0", "MARK_MEM", "OUTPUT_LO"},
        writes={"LEA_REREAD_B0"},
        kind="block",
        # Block 16 (the L13 mem-addr anchor); the LEA byte-0 LO one-hot is fresh
        # in OUTPUT here, BEFORE the block-42 L21 slam.
        target_op_name="_layer13_mem_addr_anchor",
        declarative_bake_fn=_make_standalone_pure_ffn_post_op_bake(rules),
        compiler_ir=ir,
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="FUNC_REREAD_LEA_BUG1_APPLIED_BUG2_BLK43_2026_06_25.md",
    )


def _func_lea_b0_restore_rules() -> tuple[FFNRule, ...]:
    """16 winner-take-all rules: re-establish the captured LO nibble in OUTPUT_LO.

    Each unit fires on the LEA AX row (the row discriminator) gated by the
    CLEAN captured one-hot band cell ``LEA_REREAD_B0+k`` and writes ``+DOM`` at
    ``OUTPUT_LO+k`` and ``-DOM`` at every other ``OUTPUT_LO`` cell. Because the
    band is a clean one-hot, EXACTLY ONE cell rule fires per LEA row (the
    captured nibble), so the result is a clean ``+DOM`` at the captured cell and
    ``-DOM`` everywhere else. The firing silu is so large that ``silu * DOM``
    dominates the +4.13e9 block-42 slam, re-establishing the captured LO nibble
    as the unique argmax. The band is EMPTY (all 0) on every non-captured row,
    so a unit gated by an empty band cell writes 0 (no-op) — ADD/SUB/IMM/CMP
    OUTPUT rows are untouched even though the discriminator only fires on LEA AX
    rows anyway.
    """
    rules: list[FFNRule] = []
    for k in range(16):
        writes = tuple(
            (f"OUTPUT_LO+{j}", (_FUNC_LEA_B0_DOM if j == k else -_FUNC_LEA_B0_DOM))
            for j in range(16)
        )
        rules.append(
            multi_way_and_rule(
                name=f"func_lea_b0_restore_{k}",
                conditions=_FUNC_LEA_B0_ROW_DISC,
                threshold=_FUNC_LEA_B0_RESTORE_THRESHOLD,
                gate=f"LEA_REREAD_B0+{k}",
                gate_weight=1.0,
                writes=writes,
                scope="LEA_REREAD_B0",
            )
        )
    return tuple(rules)


def make_func_lea_b0_restore_op() -> Operation:
    """RESTORE FFN: re-supply the re-read LEA byte-0 LO nibble after the slam.

    Standalone ``PureFFN`` post_op on the L25 tail block, appended AFTER
    ``tail_bit32_result_correction`` so it is the LAST OUTPUT writer before the
    LM head and therefore DOMINATES the block-42 (L21) slam. Reads the carried
    ``LEA_REREAD_B0`` band. CAMPAIGN-ONLY: a no-op when
    ``func_lea_b0_restore_enabled`` is False (golden ``d0619711`` byte-identical).
    """
    if not func_lea_b0_restore_enabled():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="func_lea_b0_restore",
            reads=set(),
            writes=set(),
            audited_empty_produces=True,
            kind="block",
            target_op_name="l10_post_ops_combined",
            requires={"after": "tail_bit32_result_correction"},
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            migrated=True,
            smoke_tests={"all"},
            spec_section="FUNC_REREAD_LEA_BUG1_APPLIED_BUG2_BLK43_2026_06_25.md",
        )

    rules = _func_lea_b0_restore_rules()
    assert len(rules) == _FUNC_LEA_B0_HIDDEN_DIM, (
        f"func_lea_b0_restore rule-count drift: {len(rules)} "
        f"!= {_FUNC_LEA_B0_HIDDEN_DIM}"
    )
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    return Operation(
        name="func_lea_b0_restore",
        reads={"MARK_AX", "OP_LEA", "OP_IMM", "MEM_STORE", "MARK_PC", "MARK_SP",
               "MARK_BP", "MARK_STACK0", "MARK_MEM", "LEA_REREAD_B0"},
        writes={"OUTPUT_LO"},
        kind="block",
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=_make_standalone_pure_ffn_post_op_bake(rules),
        compiler_ir=ir,
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="FUNC_REREAD_LEA_BUG1_APPLIED_BUG2_BLK43_2026_06_25.md",
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
        # New chain tail: prior ops fill [0, 1906), the demo's auto-fit
        # picks unit 1906 (single-unit rule), so the cumulative max is 1907.
        # Var-cluster JSR-path follow-up (2026-06-06) added units between
        # ``mem_addr_src_default_suppress`` and ``addr_key_neural_decode``
        # (``jsr_mem_default_suppress``); the same-day narrowing reduced
        # that from 8 to 4. Wave 1 Cluster B1 (2026-06-07) added
        # ``layer14_ent_ax_bytes_zero`` (4 units) after
        # ``alu_nocarry_ax_bytes_zero``, shifting demo from 1886 to 1890.
        # Multi-byte SUB follow-up (2026-06-12) added
        # ``layer14_sub_noborrow_high_byte_passthrough`` (16 units) after
        # ``layer14_ent_ax_bytes_zero``, shifting demo from 1890 to 1906
        # → tail = 1907. Campaign (30-token) SUB-borrow follow-up
        # (2026-06-21) grows that passthrough to 32 units (+16 BORROW-path
        # rules), shifting the demo +16 (1906 → 1922) → tail = 1923 in the
        # campaign config; GOLDEN keeps 1907 byte-identical. This sizes the
        # L14 mem_generation block FFN wide enough for the extra units.
        ffn_units_used=1923 if operand_from_memsp_enabled() else 1907,
        smoke_tests=set(),
        spec_section="docs/HOW_TO_ADD_A_CORRECTIVE_OP.md",
    )


def _layer14_li_zeroaddr_indicator_rules(S: float) -> tuple[FFNRule, ...]:
    """ONE FFNRule: the (committed AND zero-address) store indicator (#318).

    The 3-way AND of (committed) AND (zero-address LO nibble) AND (zero-address
    HI nibble), but factored as a MULTIPLICATIVE committedness GATE over a
    zero-address silu — NOT an additive 3-condition sum. The additive form
    ``silu(MSAV + ADDR_B0_LO+0 + ADDR_B0_HI+0 - 2.5)`` from the blueprint FALSE
    FIRES, because the value-row address-nibble one-hots are NOT clean unit
    one-hots: a NON-committed operand-frame row (probe #250 row 297) carries
    its zero-address nibbles at amplitude ~2.46 EACH, so the two nibble terms
    alone (2.46 + 2.46 = 4.92) clear 2.5 WITHOUT the MSAV term -> MSAV is not
    load-bearing in the additive threshold. (Verified on the built model:
    additive form fires on rows 297/327/357, all MSAV=0.)

    THE FACTORED FORM (the AND that actually holds):

        output = MEM_STORE_AT_VAL * silu(S*(ADDR_B0_LO+0 + ADDR_B0_HI+0) - S*thr)

    The SwiGLU gate (``gate=MEM_STORE_AT_VAL``) multiplies the silu, so a
    NON-committed row (MSAV=0) is forced to 0 REGARDLESS of its address
    amplitude (kills the row-297 false fire). The silu side then needs only the
    two-nibble zero-address AND: threshold ``thr=1.5`` fires when BOTH +0 cells
    are present (committed x-store row 267: 1.46 + 1.46 = 2.92 > 1.5) but not on
    a committed store with only ONE zero nibble (rows 147/207: a 0xX0 / 0x0X
    address gives 1.0 + 0 = 1.0 < 1.5) nor a non-zero-address store (0xF8 row
    117: 0 + 0 = 0). Only a genuine 0x00-address committed store clears it.

    Output goes to the private ``LI_ZEROADDR_COMMITTED`` band so L15 head-0 can
    key K on this SINGLE dim (the AND is done HERE in the FFN, not in the
    bilinear head — which cannot AND three conditions in one slot). Write
    magnitude ``1.0/S`` makes the output ≈1 at S=100 (silu(S*~0.9) ≈ 0.9*S,
    times 1/S, times MSAV=1 -> ≈0.9); the consumer head only needs a positive
    sign, so the exact magnitude is not load-bearing.
    """
    return (
        multi_way_and_rule(
            name="l14_li_zeroaddr_committed_indicator",
            conditions=(
                ("ADDR_B0_LO+0", 1.0),
                ("ADDR_B0_HI+0", 1.0),
            ),
            threshold=1.5,
            gate="MEM_STORE_AT_VAL",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("LI_ZEROADDR_COMMITTED+0", 1.0 / S),),
            scope=("MEM_STORE_AT_VAL and ADDR_B0_LO+0 and ADDR_B0_HI+0"),
        ),
    )


def _layer14_li_zeroaddr_indicator_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_li_zeroaddr_indicator_rules(S))
    return ir


def make_layer14_li_zeroaddr_indicator_op() -> Operation:
    """L14 FFN: materialize the (committed AND zero-address) store indicator.

    PHASE-2 KEYSTONE (#318) — the FFN half of the var_simple zero-address LI
    fix. One :class:`FFNRule` (:func:`_layer14_li_zeroaddr_indicator_rules`)
    fires ≈1 ONLY on a committed zero-address store VALUE row, writing the fresh
    private ``LI_ZEROADDR_COMMITTED`` band. L15 head-0
    (``_l15_li_zeroaddr_cam_on``) keys K on that single dim so the committed
    BP+0 local out-scores a non-committed zero-address operand-frame row WITHOUT
    the linear-K over-sharpening that desynced the AR frame.

    Campaign-flag-gated (``C4_L15_LI_ZEROADDR_CAM``, shared with the head): the
    op is registered ONLY when the flag is on (see ``all_core_ops``), so a
    flag-OFF / golden build never sees it. The reads (``MEM_STORE_AT_VAL`` from
    L7, ``ADDR_B0_LO/HI`` from L13) only exist on the campaign
    (``C4_OPERAND_FROM_MEMSP``) path; ``LI_ZEROADDR_COMMITTED`` is a flag-gated
    residual band (omitted otherwise), so the op is a no-op outside the campaign
    config too.

    LAST entry in the L14 cleanup chain (``_L14_CLEANUP_CHAIN_LAYOUT``): it
    pre-claims every deterministic, fixed-size preceding slot and lands at the
    first free gap. Because it is the chain tail, growing it can never shift a
    real op's start_unit; flag-OFF it is not registered, so the chain is
    byte-identical to golden.
    """
    enabled = _li_zeroaddr_indicator_on()

    def bake(block, dim_positions, S):
        if not enabled:
            return
        ffn = getattr(block, "ffn", None)
        if ffn is None or not hasattr(ffn, "W_up"):
            return
        start_unit = _l14_chain_alloc("layer14_li_zeroaddr_indicator")
        ir = _layer14_li_zeroaddr_indicator_ir(S)
        rules = ir.layer(0).ffn.rules
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = ir.lower_ffn(ffn, dim_map, start_unit=start_unit, S=S)
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_li_zeroaddr_indicator",
        slot_share=("ffn_units",),
        reads={"MEM_STORE_AT_VAL", "ADDR_B0_LO", "ADDR_B0_HI", "CONST"}
        if enabled else set(),
        writes={"LI_ZEROADDR_COMMITTED"} if enabled else set(),
        kind="block",
        target_op_name="layer14_mem_generation",
        # Order after the two demo fixtures (so the chain alloc pre-claims their
        # deterministic 1-unit slots) AND after mem_generation (the L14 attn op
        # this block targets). The demos are default-OFF, but the chain alloc
        # pre-claims their layout slots regardless, so the landing unit is
        # deterministic whether or not they are registered.
        requires={"after": [
            "layer14_demo_phase6_wave7",
            "layer14_mem_generation",
        ]} if enabled else {},
        declarative_bake_fn=bake,
        compiler_ir=_layer14_li_zeroaddr_indicator_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        # Chain tail: prior real ops + the two demo slots fill the pool; this
        # single-unit op lands at the next free gap (unit 1925 in the campaign
        # config), so the block must be sized to 1926 (a half-open [0,1926)
        # pool). Only reserved when the flag is on; OFF => not registered, so the
        # demo op's sizing (1923) governs (golden byte-identical).
        ffn_units_used=1926 if enabled else 0,
        smoke_tests=set(),
        spec_section="project_campaign_savedra_chain_flip (#318)",
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


# ===========================================================================
# CROSS-OP FFN-LINT DEMO OPS (tooling-only; both DEFAULT OFF).
# ---------------------------------------------------------------------------
# These two ops are the discrimination fixtures for tools/lint_cross_op_ffn.py
# --demo. Each lowers ONE leaky-silu FFNRule into the l14 ALU block's live
# PureFFN via the CLEANUP-CHAIN allocator (_l14_chain_alloc + CompilerIR.
# lower_ffn — the SAME robust path as make_layer14_demo_phase6_wave7_op, so the
# unit is registered and survives _right_size_ffns). Both are registered ONLY
# when their flag is on (the trailing chain-layout entries), so a flag-off /
# production build is byte-identical to golden 4958b35b. The single authored
# FFNRule pins the silu spillover exactly. (A hardcoded high unit slot does NOT
# work: it collides with an allocator-managed unit and is merged/renumbered by
# the build's FFN compaction, and a post-trim hand-write has no free slot — the
# right-sized l14 FFN is 100% packed.)
#
#   * make_ffn_lint_mull14_demo_op  (C4_FFN_LINT_MULL14_DEMO) — the ENTANGLEMENT.
#     A unit AUTHORED as "MUL-only" (W_up reads OP_MUL strongly) but with a
#     POSITIVE b_up so silu(up) is non-zero even when OP_MUL==0. It writes the
#     SHARED OUTPUT_LO band -> on ADD/SUB/DIV rows the leaked silu firing
#     perturbs the band other ops read (the exact -60). The lint must FLAG it.
#   * make_ffn_lint_clean_demo_op   (C4_FFN_LINT_CLEAN_DEMO) — the CLEAN control.
#     IDENTICAL silu spillover, but its W_down writes a PRIVATE TEMP scratch dim
#     (no OUTPUT/ALU band). Because no downstream op reads TEMP as an ALU result
#     the shared-band probe sees ZERO change -> the lint must PASS it.
# ===========================================================================
def _ffn_lint_demo_rule(write_dim_name: str) -> FFNRule:
    """One leaky-silu OP_MUL :class:`FFNRule` for the cross-op FFN-lint demo.

    Authored as a "MUL-only" unit whose pre-silu activation
    ``up = S*(0.04*OP_MUL) - S*(-0.006) = 0.6 + 4.0*OP_MUL`` is:

      * ``silu(0.6) ~= 0.39`` at the OP_MUL==0 BASELINE — the smooth-nonlinearity
        LEAK that perturbs OTHER opcodes' rows, AND
      * ``silu(4.6) ~= 4.55`` when OP_MUL fires — the intended write.

    ``gate=None, gate_bias=1.0`` makes the multiplicative gate a constant 1.0 so
    the product ``silu(up)*gate`` is just ``silu(up)``. ``writes`` routes that
    activation into ``write_dim_name+0`` — ``OUTPUT_LO+0`` (the SHARED l14 ALU
    band every binary-ALU op reads) for the ENTANGLEMENT fixture, or ``TEMP+0``
    (a PRIVATE scratch dim no downstream op reads as an ALU result) for the
    CLEAN control. Lowered via :meth:`CompilerIR.lower_ffn`, so the unit is a
    real, registered hidden unit that survives ``_right_size_ffns`` (its W_up /
    W_down are non-zero) — unlike a post-trim hand-write, which has no free slot
    to land in (the trimmed l14 FFN is 100% packed).
    """
    return FFNRule.gated_write(
        name=f"ffn_lint_demo_leaky_mul_{write_dim_name.lower()}",
        conditions=(("OP_MUL", 0.04),),
        threshold=-0.006,
        gate=None,
        gate_bias=1.0,
        writes=((f"{write_dim_name}+0", 1.0),),
        scope="__never_fires__",
    )


def _ffn_lint_demo_ir(write_dim_name: str) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(_ffn_lint_demo_rule(write_dim_name))
    return ir


def _ffn_lint_demo_bake(block, dim_positions, S, *, chain_op_name: str,
                        write_dim_name: str):
    """Lower the leaky-silu OP_MUL demo unit into the l14 block's live FFN.

    Mirrors :func:`make_layer14_demo_phase6_wave7_op`'s bake EXACTLY: take the
    chain-allocated start unit (:func:`_l14_chain_alloc`) and lower the single
    :class:`FFNRule` via :meth:`CompilerIR.lower_ffn`. This is the ONLY robust
    path: a hardcoded slot (the original 1000/1001 attempt) collides with an
    allocator-managed unit and is silently merged/renumbered by the build's FFN
    compaction (its ``b_up`` gets clobbered to a neighbour's, killing the leak),
    and a post-trim hand-write has NO free slot (the right-sized l14 FFN is 100%
    packed). The chain layout entry is the LAST in
    :data:`_L14_CLEANUP_CHAIN_LAYOUT`, so when the demo flag is OFF the op is
    never registered, ``_l14_chain_alloc`` never claims its slot, and no real
    op's start_unit shifts -> a flag-off build is byte-identical to golden.
    """
    ffn = getattr(block, "ffn", None)
    if ffn is None or not hasattr(ffn, "W_up"):
        return

    start_unit = _l14_chain_alloc(chain_op_name)
    ir = _ffn_lint_demo_ir(write_dim_name)
    rules = ir.layer(0).ffn.rules
    dim_map = Primitives.dim_positions_from_bd(
        _as_setdim_proxy(dim_positions),
        Primitives.ffn_rule_dim_names(rules),
    )
    next_unit = ir.lower_ffn(ffn, dim_map, start_unit=start_unit, S=S)
    ffn._l14_unit_counter = next_unit


def make_ffn_lint_mull14_demo_op() -> Operation:
    """Tooling-only ENTANGLEMENT fixture (DEFAULT OFF, C4_FFN_LINT_MULL14_DEMO).

    See the module banner above. Writes the SHARED OUTPUT_LO band with a leaky
    OP_MUL silu unit so the lint's ADD/SUB/DIV probe rows see a non-local change.
    """
    enabled = ffn_lint_mull14_demo_enabled()

    def bake(block, dim_positions, S):
        if not enabled:
            return
        _ffn_lint_demo_bake(
            block, dim_positions, S,
            chain_op_name="ffn_lint_mull14_demo",
            write_dim_name="OUTPUT_LO",
        )

    return Operation(
        name="ffn_lint_mull14_demo",
        slot_share=("ffn_units",),
        reads={"OP_MUL", "CONST"} if enabled else set(),
        writes={"OUTPUT_LO"} if enabled else set(),
        kind="block",
        target_op_name="layer14_mem_generation",
        requires={"after": [
            "layer14_demo_phase6_wave7",
            "layer14_mem_generation",
        ]} if enabled else {},
        declarative_bake_fn=bake,
        compiler_ir=_ffn_lint_demo_ir("OUTPUT_LO"),
        migrated=True,
        declarative_authority="spec_generated",
        # Chain tail: after ``layer14_demo_phase6_wave7`` (unit 1906), this op's
        # auto-fit lands at 1907 -> cumulative max 1908. Only reserved when the
        # flag is on; flag-off the op is not registered at all (byte-identical).
        ffn_units_used=1908 if enabled else 0,
        smoke_tests={"all"},
        spec_section="tools/lint_cross_op_ffn.py",
    )


def make_ffn_lint_clean_demo_op() -> Operation:
    """Tooling-only CLEAN-CONTROL fixture (DEFAULT OFF, C4_FFN_LINT_CLEAN_DEMO).

    See the module banner above. IDENTICAL leaky OP_MUL silu unit, but routes
    its W_down to a PRIVATE TEMP scratch dim (no OUTPUT/ALU band), so the lint's
    shared-band probe sees no change and PASSES.
    """
    enabled = ffn_lint_clean_demo_enabled()

    def bake(block, dim_positions, S):
        if not enabled:
            return
        _ffn_lint_demo_bake(
            block, dim_positions, S,
            chain_op_name="ffn_lint_clean_demo",
            write_dim_name="TEMP",
        )

    return Operation(
        name="ffn_lint_clean_demo",
        slot_share=("ffn_units",),
        reads={"OP_MUL", "CONST"} if enabled else set(),
        writes={"TEMP"} if enabled else set(),
        kind="block",
        target_op_name="layer14_mem_generation",
        # Only depend on always-registered ops (``ffn_lint_mull14_demo`` may be
        # absent when only the clean flag is on). The chain allocator still
        # pre-claims ``ffn_lint_mull14_demo``'s layout slot deterministically,
        # so clean lands at unit 1908 whether or not mull14 is registered.
        requires={"after": [
            "layer14_demo_phase6_wave7",
            "layer14_mem_generation",
        ]} if enabled else {},
        declarative_bake_fn=bake,
        compiler_ir=_ffn_lint_demo_ir("TEMP"),
        migrated=True,
        declarative_authority="spec_generated",
        # Chain tail after ``ffn_lint_mull14_demo`` (unit 1907) -> unit 1908,
        # cumulative max 1909. Only reserved when the flag is on; the
        # ``mull14_demo`` predecessor in ``requires`` is only present when ITS
        # flag is on, so when clean runs solo the chain alloc still pre-claims
        # ``ffn_lint_mull14_demo``'s layout slot (1907) deterministically and
        # clean lands at 1908 regardless. Flag-off -> not registered.
        ffn_units_used=1909 if enabled else 0,
        smoke_tests={"all"},
        spec_section="tools/lint_cross_op_ffn.py",
    )
