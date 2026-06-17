"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import os as _os_l15
import torch
import torch.nn as nn
from collections.abc import Mapping

from ...attention_head_allocator import AttentionHeadAllocator


def _l15_li_load_suppressor_inert_on() -> bool:
    """DEFAULT-ON flag (``C4_L15_LI_SUPPR_INERT``): make the L15 head-0
    PC/SP/STACK0-discriminator slots INERT on genuine LI/LC LOAD query rows.

    ROOT (spec_k=0, BUILT dims, func_identity_0 id550 step-7 LI, 2026-06-15):
    The L15 head-0 content-addressable load works cleanly in isolation
    (si_li_roundtrip: AX=42, attends the stored value). In a FUNCTION program
    the callee ENT step's ``OP_ENT`` BROADCASTS a ~0.3-2 residue onto the prior
    PSH-arg store tokens AND onto the LI query (AX-marker) row. That residue
    mis-fires a CHAIN of head-0 discriminator slots that are large-magnitude
    NOT-blockers for OTHER row types:

      * slot 58 (early-ENT STACK0 discriminator): Q ~= -1.2e10 on the LI row;
        K[58]=1e4*OP_ENT is nonzero ONLY on the OP_ENT-residue store tokens --
        i.e. it penalises EXACTLY the PSH-arg store the LI must read (-3.3e13),
        burying the whole softmax1 (sink wins -> AX=0).
      * slot 62 (sp_byte_blocker): q[62,OP_ENT]=5e5 * K[62]=-3e5*CONST adds a
        UNIFORM -1.1e11 to every key on the OP_ENT-residue LI row.
      * slot 35 (pc_byte_blocker): its 0xFFE8 e8-signature boost
        (ADDR_B0_LO+8 + ADDR_B0_HI+15) only half-cancels the existing -10000
        OP_LI_RELAY guard, leaving Q[35]~+1e4 * K[35]=-1e5 = a -1e9 baseline.

    PROOF: zeroing slots {34,35,58,59,60,61,62} for head 0 makes the func LI
    head-0 attend the value-70 store (pos 243, weight 1.0) -> step-7 AX=70, AND
    var_simple_0/12 still decode 990/28 (the address bits alone pick the right
    store once the suppressors are out of the way).

    FIX: on a genuine LI/LC LOAD query row (OP_LI_RELAY + OP_LC_RELAY > 0,
    MARK_AX=1) drive these slots' Q to a value whose product with their K is
    >= 0 (un-buried), so the address-bit slots (4-27) decide the lookup exactly
    as on the clean si_li path. Off-relay (OP_LI_RELAY==OP_LC_RELAY==0 -- every
    PC/SP/STACK0/pop row the suppressors actually guard) the writes are absent,
    so HEAD is byte-identical with the flag OFF.
    """
    return _os_l15.environ.get("C4_L15_LI_SUPPR_INERT", "1") != "0"
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import (
    attention_head_extension,
    binary_address_lookup_attention,
    multi_way_and_rule,
)
from ..ir import (
    CompilerIR,
    FFNRule,
    RuntimeAttentionFragment,
    StepWindowConstraint,
    StructuralOp,
)
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L15 attention head layout (auto-fit; legacy head_idx as docs) ===
#
# L15 attention is the load-side memory pipeline. The block is structurally
# resized by ``l15_attention_resize`` (phase 14.9) so the live head count
# varies per build:
#
#   * Default 16-layer build: ``num_heads = 9``. Heads 0-3 host the LI/LC
#     and pop-group STACK0 lookups; head 8 hosts the wide-ALU byte relay;
#     head 9 is the pop_d8_to_e0 lookup added by the suppress helper.
#   * 17-layer LEV build:    ``num_heads = 14``. Heads 4-7 add LEV
#     saved_bp reads, heads 10-11 add LEV return_addr byte 2/3 reads
#     (byte 0 is on head 8, byte 1 on head 9 which the suppress helper
#     wipes and rewrites for pop_d8_to_e0). Heads 12-13 host the
#     SI/SC-only STACK0 byte0 / addr0 overrides.
#
# This table is the single source of truth for the L15 head axis.
# ``layer_max_heads=14`` covers the widest configuration; runtime
# narrower-width builds simply leave the higher allocator slots unused.
# Every ``head_idx`` literal in the spec functions below (e.g.
# ``head_idx=12`` in :func:`_layer15_store_stack0_sp_byte0_addr_spec`) is
# resolved through :data:`_L15_HEAD_LAYOUT_BY_NAME`, so byte-identity
# with the legacy bake is preserved regardless of allocator order.
#
# Heads 0-11 are written by ``_set_layer15_memory_lookup`` in vm_step
# (still imperative) and the suppress helper in this module
# (also still imperative due to conditional num_heads logic). The
# already-declarative ops ``layer15_store_stack0_sp_byte0_addr``,
# ``layer15_si_mem_addr0_from_stack0``, and ``layer15_alu_high_byte_relay``
# (the last one lives in ``l14_ops.py`` but writes to the L15 attention
# block) resolve their head index by name from this table.
#
# Phase 7.B.6: the allocator now runs without ``pin=`` -- first-fit picks
# 0..13 in declaration order, which matches the legacy layout bit-for-bit
# because :data:`_L15_HEAD_LAYOUT` is contiguous and ordered. The
# ``legacy_head_idx`` column is kept purely as documentation; the
# load-bearing copy is :data:`_L15_HEAD_LAYOUT_BY_NAME`.
_L15_HEAD_LAYOUT = (
    # (op-name key,                                    legacy_head_idx (docs only))
    ("layer15_memory_lookup.li_lc_stack0_h0",          0),  # head 0: LI/LC byte 0 + STACK0 pop dual-role
    ("layer15_memory_lookup.li_lc_stack0_h1",          1),  # head 1: LI/LC byte 1 (BYTE_INDEX_0 gate)
    ("layer15_memory_lookup.li_lc_stack0_h2",          2),  # head 2: LI/LC byte 2 (BYTE_INDEX_1 gate)
    ("layer15_memory_lookup.li_lc_stack0_h3",          3),  # head 3: LI/LC byte 3 (BYTE_INDEX_2 gate)
    ("layer15_memory_lookup.lev_saved_bp_h4",          4),  # head 4: LEV saved_bp byte 0 (num_heads>=12)
    ("layer15_memory_lookup.lev_saved_bp_h5",          5),  # head 5: LEV saved_bp byte 1
    ("layer15_memory_lookup.lev_saved_bp_h6",          6),  # head 6: LEV saved_bp byte 2
    ("layer15_memory_lookup.lev_saved_bp_h7",          7),  # head 7: LEV saved_bp byte 3
    ("layer15_alu_high_byte_relay",                    8),  # head 8: wide-ALU staged byte 1 relay (l14_ops owns spec)
    ("layer15_memory_lookup.pop_d8_to_e0",             9),  # head 9: post-pop one-word pushed result lookup (num_heads>9)
    ("layer15_memory_lookup.lev_return_addr_h10",      10),  # head 10: LEV return_addr byte 2
    ("layer15_memory_lookup.lev_return_addr_h11",      11),  # head 11: LEV return_addr byte 3
    ("layer15_store_stack0_sp_byte0_addr",             12),  # head 12: SI/SC store-top SP byte0 -> ADDR_B0
    ("layer15_si_mem_addr0_from_stack0",               13),  # head 13: SI/SC MEM addr0 from pre-store STACK0 byte0
)
_L15_HEAD_LAYOUT_BY_NAME = {name: head_idx for name, head_idx in _L15_HEAD_LAYOUT}


def _allocate_layer15_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L15 heads.

    Phase 7.B.6: ``pin=`` is dropped from every entry. The allocator's
    first-fit picks the lowest free head index in declaration order;
    because :data:`_L15_HEAD_LAYOUT` is contiguous (0..13) and ordered,
    first-fit reproduces the legacy ``head_idx`` values bit-for-bit.
    The actual weight-write head indices are still looked up via
    :data:`_L15_HEAD_LAYOUT_BY_NAME` inside the head-spec factories
    below, so byte-identity with the legacy bake is preserved
    regardless of allocator order.

    ``layer_max_heads=14`` is the widest L15 configuration the resize op
    produces (17-layer LEV build). Narrower builds simply leave the high
    slots claimed-but-unused; the allocator never writes weights itself,
    it only records the layout for collision checks and downstream
    inspection. Stashed on ``attn._l15_head_allocator`` by both
    ``layer15_memory_lookup`` and ``l15_attention_resize`` bakes so
    downstream tooling can audit the layout.
    """
    allocator = AttentionHeadAllocator(layer_max_heads=14)
    for name, _legacy_head_idx in _L15_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=15)
    return allocator


def _l15_head_idx(op_name: str) -> int:
    """Return the pinned L15 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L15_HEAD_LAYOUT` for callers that
    cannot instantiate a per-bake allocator (e.g. the head-spec
    helpers consumed by both bake and ``compiler_ir_factory`` paths,
    where running the collision-checked allocator on every call
    would be wasteful). The runtime bakes still go through
    :func:`_allocate_layer15_attention_heads` so the collision-checked
    allocator path is exercised on every weight write.
    """
    try:
        return _L15_HEAD_LAYOUT_BY_NAME[op_name]
    except KeyError:
        raise KeyError(
            f"_l15_head_idx: unknown L15 attention op {op_name!r}"
        ) from None


# === L15 FFN unit layout (auto-fit; legacy offsets retained as docs) ==
#
# ``layer15_nibble_copy`` owns the entire L15 FFN. The actual weight
# writes happen inside ``lower_l15_nibble_copy_ir`` (and the legacy
# ``vm_step._set_nibble_copy_ffn`` path), which use a monotonic
# ``unit = 0`` counter that walks 42 sub-stages: 16 LO nibble-copy units,
# 16 HI nibble-copy units, 8 PSH stack-byte units, and 2 first-step LEA
# units (see ``make_l15_nibble_copy_ir`` and ``make_l15_psh_stack_ir``).
#
# Phase 7.B.6: every entry below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because the layout is fully
# contiguous in declaration order, first-fit reproduces the legacy
# pinned offsets bit-for-bit -- so byte-identity with the legacy
# ``vm_step._set_nibble_copy_ffn`` helper and ``lower_l15_nibble_copy_ir``
# unit-cursor survives the pin drop. The ``legacy_start`` column is
# kept purely as documentation.
#
# The other L15-named ops in this module (``layer15_memory_lookup``,
# ``layer15_alu_high_byte_relay`` -- which actually lives in
# ``l14_ops.py``, ``layer15_store_stack0_sp_byte0_addr``,
# ``layer15_si_mem_addr0_from_stack0``, ``l15_attention_resize``) are
# attention-side bakes; they do not consume FFN hidden units and are not
# represented in this table.
#
# The offsets below mirror the rule order in ``make_l15_nibble_copy_ir``
# (nibble_copy_lo_{0..15} then nibble_copy_hi_{0..15}) followed by the
# 8 rules from ``make_l15_psh_stack_ir`` and the final 2 LEA rules.
# Changing the rule list requires updating this table in lock-step.
_L15_FFN_UNIT_LAYOUT = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("layer15_nibble_copy.nibble_copy_lo",         0, 16),  # OUTPUT_LO copy
    ("layer15_nibble_copy.nibble_copy_hi",        16, 16),  # OUTPUT_HI_THIS_STEP copy
    ("layer15_nibble_copy.psh_sp_byte1_lo_ff",    32,  1),  # PSH SP byte1 lo=0xf
    ("layer15_nibble_copy.psh_sp_byte1_hi_ff",    33,  1),  # PSH SP byte1 hi=0xf
    ("layer15_nibble_copy.psh_sp_byte2_lo_00",    34,  1),  # PSH SP byte2 lo=0
    ("layer15_nibble_copy.psh_sp_byte2_hi_00",    35,  1),  # PSH SP byte2 hi=0
    ("layer15_nibble_copy.psh_sp_byte3_lo_00",    36,  1),  # PSH SP byte3 lo=0
    ("layer15_nibble_copy.psh_sp_byte3_hi_00",    37,  1),  # PSH SP byte3 hi=0
    ("layer15_nibble_copy.psh_bp_byte2_lo_01",    38,  1),  # PSH BP byte2 lo=1
    ("layer15_nibble_copy.psh_bp_byte2_hi_00",    39,  1),  # PSH BP byte2 hi=0
    ("layer15_nibble_copy.lea_first_step_lo_01",  40,  1),  # LEA AX byte2 lo=1
    ("layer15_nibble_copy.lea_first_step_hi_00",  41,  1),  # LEA AX byte2 hi=0
)


def _allocate_layer15_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L15 FFN sub-stages.

    Phase 7.B.6: ``pin=`` is dropped from every entry. The allocator's
    default first-fit walks :data:`_L15_FFN_UNIT_LAYOUT` in declaration
    order and lands each sub-stage at the lowest free gap large enough
    to hold it. Because the layout is fully contiguous (every entry
    starts exactly where the previous one ended), first-fit reproduces
    the legacy pinned offsets bit-for-bit -- so byte-identity with
    ``vm_step._set_nibble_copy_ffn`` and ``lower_l15_nibble_copy_ir``'s
    own ``unit = start_unit`` cursor is preserved without the author
    having to spell out the offsets.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L15 op claims a free range past unit 42).
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L15_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def make_l15_psh_stack_ir() -> CompilerIR:
    """Declarative L15 PSH SP/BP byte producer rules.

    These rules cover the PSH-specific SP byte outputs and BP byte-2
    preservation formerly hand-written in ``_set_nibble_copy_ffn``. Writes are
    expressed in semantic output-delta units; neural lowering applies ``1 / S``
    scaling to recover the legacy W_down coefficients.
    """
    ir = CompilerIR()
    rules = ir.layer(0).ffn
    sp_i = 2
    bp_i = 3
    threshold = 3.5

    all_byte_indices = ("BYTE_INDEX_0", "BYTE_INDEX_1",
                        "BYTE_INDEX_2", "BYTE_INDEX_3")

    def psh_byte_conditions(marker_index, byte_index_name):
        conds = [
            ("PSH_AT_SP", 1.0),
            (f"H1+{marker_index}", 1.0),
            ("IS_BYTE", 1.0),
            (byte_index_name, 1.0),
            # Strict traces can relay PSH_AT_SP at value ~2.0 onto STACK0 byte
            # rows.  Without an explicit stack-area blocker, that residue plus
            # IS_BYTE/BYTE_INDEX is enough to fire the SP-byte producer away
            # from the actual SP register row.
            (f"H1+10", -1.0),
            (f"H4+{bp_i}", -1.0),
        ]
        # PSH_AT_SP arrives at value ~2.0 on the var-store frame PSH step
        # (not the ~1.0 of the flat-PSH smoke path).  That +1.0 of unearned
        # AND-headroom otherwise lets the WRONG-byte-index and WRONG-marker
        # units clear the 3.5 threshold on every SP-byte row, leaking the
        # byte1 0xf nibble + spurious HI[0] writes onto the byte2/byte3
        # prediction rows (var_simple_12 step-3 SP byte2 -> 0x0f).  Make the
        # byte-index and marker discriminators load-bearing by subtracting
        # the *other* one-hot indices: on the legit row the negatives are 0
        # (byte-identical), on a wrong-index/wrong-marker row the active
        # one-hot drives the sum back below threshold regardless of the
        # PSH_AT_SP magnitude.
        for other_bi in all_byte_indices:
            if other_bi != byte_index_name:
                conds.append((other_bi, -1.0))
        # Cross-marker blocker (BP producer only), frame-scoped via OP_ENT.
        #
        # The BP byte2 preserver shares BYTE_INDEX_1 with the SP byte2
        # zero-writer, so on a PSH step (PSH_AT_SP=2.0) the +1.0 of unearned
        # AND-headroom lets it ride onto the SP byte2 row and leak
        # OUTPUT_HI[0]/OUTPUT_LO[1] (var step-3 PSH SP byte2 -> 0x0f / 0x01).
        #
        # We darken it there with an OP_ENT condition rather than a bare
        # ``-H1+sp`` marker blocker: a bare marker blocker also corrects the
        # *otherwise benign* SP byte2 residue on the flat PSH+ADJ smoke path
        # (``test_adj_sp``), which un-masks a latent L9 OP_ADJ AX writer and
        # turns that test red.  OP_ENT is the one clean discriminator between
        # the two: it is present (frame prologue residue ~0.5-16 on the var
        # store/recall path) on every var leak row and ~0 on the flat
        # PSH+ADJ row.  ``-2.0 * OP_ENT`` pulls the BP producer below the 3.5
        # threshold on the var SP byte2 row while leaving the flat-PSH path
        # (OP_ENT=0) byte-identical, and the BP producer never fires on a
        # genuine BP byte row in either trace (PSH_AT_SP=0 there), so this is
        # a pure leak suppression.
        if marker_index == bp_i:
            conds.append(("OP_ENT", -2.0))
        return tuple(conds)

    # SP byte 0 position predicts SP byte 1 = 0xff after SP -= 8.
    rules.append(multi_way_and_rule(
        name="psh_sp_byte1_lo_ff",
        conditions=psh_byte_conditions(sp_i, "BYTE_INDEX_0"),
        threshold=threshold,
        writes=(("OUTPUT_LO+15", 4.0), ("OUTPUT_LO+0", -4.0)),
    ))
    rules.append(multi_way_and_rule(
        name="psh_sp_byte1_hi_ff",
        conditions=psh_byte_conditions(sp_i, "BYTE_INDEX_0"),
        threshold=threshold,
        writes=(("OUTPUT_HI_THIS_STEP+15", 4.0), ("OUTPUT_HI_THIS_STEP+0", -4.0)),
    ))

    # SP byte 1 and byte 2 positions predict zero for the following bytes.
    for byte_index_name, predicted_byte in (
        ("BYTE_INDEX_1", "byte2"),
        ("BYTE_INDEX_2", "byte3"),
    ):
        rules.append(multi_way_and_rule(
            name=f"psh_sp_{predicted_byte}_lo_00",
            conditions=psh_byte_conditions(sp_i, byte_index_name),
            threshold=threshold,
            writes=(("OUTPUT_LO+0", 2.0),),
        ))
        rules.append(multi_way_and_rule(
            name=f"psh_sp_{predicted_byte}_hi_00",
            conditions=psh_byte_conditions(sp_i, byte_index_name),
            threshold=threshold,
            writes=(("OUTPUT_HI_THIS_STEP+0", 2.0),),
        ))

    # PSH leaves BP unchanged; preserve STACK_INIT byte 2 = 0x01.
    rules.append(multi_way_and_rule(
        name="psh_bp_byte2_lo_01",
        conditions=psh_byte_conditions(bp_i, "BYTE_INDEX_1"),
        threshold=threshold,
        writes=(("OUTPUT_LO+1", 4.0), ("OUTPUT_LO+0", -4.0)),
    ))
    rules.append(multi_way_and_rule(
        name="psh_bp_byte2_hi_00",
        conditions=psh_byte_conditions(bp_i, "BYTE_INDEX_1"),
        threshold=threshold,
        writes=(("OUTPUT_HI_THIS_STEP+0", 4.0),),
    ))
    return ir


def make_l15_nibble_copy_ir() -> CompilerIR:
    """Declarative L15 nibble-copy FFN program.

    Covers the full legacy ``_set_nibble_copy_ffn`` unit range:
    32 generic copy units, 8 PSH stack-byte units, and 2 first-step LEA units.
    Writes are semantic deltas; lowering applies ``1 / S``.
    """

    ir = CompilerIR()
    rules = ir.layer(0).ffn
    pc_i = 0
    ax_i = 1
    sp_i = 2
    bp_i = 3

    copy_conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{pc_i}", -1.0),
        (f"H1+{ax_i}", -1.0),
        (f"H1+{sp_i}", -1.0),
        (f"H1+{bp_i}", -1.0),
        (f"H4+{bp_i}", -1.0),
        ("MEM_STORE", -1.0),
        # Hard blockers on BP marker / BP byte stream so the wide L15
        # nibble-copy writer cannot dump ~+758 into OUTPUT_LO+0 across BP
        # byte rows during the local-frame post-store cadence; without
        # these the L16 ``l16_bp_frame_byte1_ff`` BP_byte1=0xff override
        # (50.0/S strength) is swamped and if_var_* (IDs 425-449) regress
        # to BP_byte1=0xf0.
        ("H1+3", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
    )
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"nibble_copy_lo_{k}",
            conditions=copy_conditions,
            threshold=0.5,
            gate=f"EMBED_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 2.0),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"nibble_copy_hi_{k}",
            conditions=copy_conditions,
            threshold=0.5,
            gate=f"EMBED_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 2.0),),
        ))

    rules.rules.extend(make_l15_psh_stack_ir().layer(0).ffn.rules)

    lea_conditions = (
        ("CMP+7", 1.0),
        (f"H1+{ax_i}", 1.0),
        ("IS_BYTE", 1.0),
        ("BYTE_INDEX_1", 1.0),
        ("HAS_SE", -1.0),
    )
    rules.append(multi_way_and_rule(
        name="lea_first_step_ax_byte2_lo_01",
        conditions=lea_conditions,
        threshold=4.5,
        writes=(("OUTPUT_LO+1", 4.0), ("OUTPUT_LO+0", -4.0)),
    ))
    rules.append(multi_way_and_rule(
        name="lea_first_step_ax_byte2_hi_00",
        conditions=lea_conditions,
        threshold=4.5,
        writes=(("OUTPUT_HI_THIS_STEP+0", 2.0),),
    ))
    return ir


def lower_l15_psh_stack_ir(ffn, dim_positions, *, start_unit, S):
    """Lower L15 PSH stack IR into ``ffn`` and return the next free unit."""
    return make_l15_psh_stack_ir().lower_ffn(
        ffn,
        dim_positions,
        start_unit=start_unit,
        S=S,
        write_scale=1.0 / S,
    )


def lower_l15_nibble_copy_ir(ffn, dim_positions, *, start_unit=0, S=100.0):
    """Lower the full L15 nibble-copy IR and return the next free unit."""
    ir = make_l15_nibble_copy_ir()
    rules = ir.layer(0).ffn.rules
    if not isinstance(dim_positions, Mapping):
        dim_positions = Primitives.dim_positions_from_bd(
            dim_positions,
            Primitives.ffn_rule_dim_names(rules),
        )
    return ir.lower_ffn(
        ffn,
        dim_positions,
        start_unit=start_unit,
        S=S,
        write_scale=1.0 / S,
    )


def make_nibble_copy_ffn_op() -> Operation:
    """Topology anchor for L15 nibble-copy FFN.

    The actual weight bake is owned by ``layer15_nibble_copy`` below, pinned
    to ``model.blocks[15].ffn``.
    """
    def bake(ffn, dim_positions, S):
        return None

    return Operation(
        name="nibble_copy_ffn",
        reads={"IS_BYTE", "H1", "H4", "MEM_STORE",
               "EMBED_LO", "EMBED_HI", "PSH_AT_SP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "MARK_BP", "MARK_STACK0", "HAS_SE", "CMP"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
        # Dead-unit budget (docs/DEAD_UNIT_AUDIT_2026_06_05.md): L7 is
        # attention-only by design -- ``l7_ops.py``'s docstring (lines
        # 74-83) notes there is no ``_set_layer7_ffn`` helper and no FFN
        # bake at this anchor's layer. The historical 4096-unit budget
        # was 100% dead (4096 / 4096 = 100.0%). Declaring 0 lets the
        # dynamic-FFN allocator pre-size block[L7].ffn to hidden_dim=0
        # instead of allocating 4096 dead rows. Savings: 4096 *
        # (2*d_model + 1) = ~6.55M params pre-rightsize.
        ffn_units_used=0,
    )


def _layer15_memory_lookup_heads_0_3_specs(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative L15 heads 0-3 mirroring ``_set_layer15_memory_lookup_heads_0_3``.

    Phase 7.C.3 migration: replaces the imperative ``W_q/W_k/W_v/W_o`` writes
    in :func:`vm_step._set_layer15_memory_lookup_heads_0_3` with a tuple of
    :class:`DeclarativeAttentionHeadSpec` so the always-on LI/LC + STACK0
    load heads are authored as data, lowered by
    :meth:`CompilerIR.lower_attention` through
    :func:`Primitives.generate_attention_heads`.

    The bodies mirror the legacy helper cell-for-cell. Where the imperative
    setter wrote the same ``(slot, dim)`` cell twice (e.g. slot 0 ``H1+BP_I``
    for ``h > 0``, first to ``-2000`` then to ``-50000``) only the final
    value is emitted -- the lowerer is assignment, not accumulation, so the
    last-write-wins semantics are preserved bit-for-bit.

    The 24-bit binary-address block at slots 4..27 is generated by the same
    nested-loop structure as the legacy helper (3 address bytes * 2 nibbles
    * 4 bits per nibble = 24 dims, each populated with the
    ``2*((k>>bit)&1)-1`` bit-encoding across all 16 nibble values).
    """

    PC_I = 0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4

    specs: list[DeclarativeAttentionHeadSpec] = []

    for h in range(4):
        byte_q_flag = [
            BD.MARK_AX, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2,
        ][h]
        MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]

        q: list[AP] = []
        k: list[AP] = []

        # === Slot 0: Bias -- suppress non-target Q positions ===
        q.append(AP(0, BD.CONST, -2000.0))
        q.append(AP(0, BD.OP_LI_RELAY, 2000.0))
        if h == 0:
            q.append(AP(0, BD.OP_LC_RELAY, 2000.0))
            q.append(AP(0, BD.CMP + 3, 2000.0))  # POP group -> stack memory read
        else:
            # L1H4[BP] gate active over STACK0 area. The legacy helper also
            # wrote H1[BP_I] = -2000 here, but the SP/BP byte blocker below
            # (-50000) overwrites that cell; emit only the final value.
            q.append(AP(0, BD.L1H4 + BP_I, 2000.0))
        q.append(AP(0, BD.CMP + 0, -2000.0))
        # LEV suppression and PC/SP marker blockers.
        q.append(AP(0, BD.OP_LEV, -1000.0))
        q.append(AP(0, BD.MARK_PC, -25000.0))
        q.append(AP(0, BD.MARK_SP, -100000.0))
        # SP/BP byte blocker -- last write wins on (slot=0, BD.H1+BP_I).
        q.append(AP(0, BD.H1 + SP_I, -50000.0))
        q.append(AP(0, BD.H1 + BP_I, -50000.0))
        k.append(AP(0, BD.CONST, 10.0))

        # === Slot 29: PC byte position blocker ===
        q.append(AP(29, BD.H1 + PC_I, -20000.0))
        k.append(AP(29, BD.CONST, 5.0))

        # === Slot 30: AX byte position default blocker ===
        q.append(AP(30, BD.H1 + AX_I, -20000.0))
        k.append(AP(30, BD.CONST, 5.0))

        # === Slot 31: restore AX-byte score for real LI loads against
        # stored MEM entries (head 0 also LC). ===
        q.append(AP(31, BD.OP_LI_RELAY, 20000.0))
        if h == 0:
            q.append(AP(31, BD.OP_LC_RELAY, 20000.0))
        k.append(AP(31, BD.MEM_STORE, 5.0))

        # === Slot 32: AX marker default blocker ===
        q.append(AP(32, BD.MARK_AX, -20000.0))
        k.append(AP(32, BD.CONST, 5.0))

        # === Slot 33: restore AX-marker score for head 0 LI/LC loads ===
        if h == 0:
            q.append(AP(33, BD.OP_LI_RELAY, 20000.0))
            q.append(AP(33, BD.OP_LC_RELAY, 20000.0))
            k.append(AP(33, BD.MEM_STORE, 5.0))

        # === Slot 1: Store anchor -- suppress non-store K at target Q ===
        q.append(AP(1, BD.OP_LI_RELAY, 50.0))
        if h == 0:
            q.append(AP(1, BD.OP_LC_RELAY, 50.0))
            q.append(AP(1, BD.CMP + 3, 50.0))  # POP group (matches slot 0)
        else:
            q.append(AP(1, BD.L1H4 + BP_I, 50.0))
            q.append(AP(1, BD.H1 + BP_I, -50.0))
        q.append(AP(1, BD.CMP + 0, -50.0))
        k.append(AP(1, BD.MEM_STORE, 100.0))
        k.append(AP(1, BD.CONST, -50.0))

        # === Slot 2: ZFOD negative offset for store entries ===
        q.append(AP(2, BD.CONST, -96.0))
        k.append(AP(2, BD.MEM_STORE, 50.0))

        # === Slot 3: Byte selection ===
        BS = 60.0  # Byte Selection weight (60*60/8 = 450 per matching byte)
        q.append(AP(3, byte_q_flag, BS))
        if h == 0:
            q.append(AP(3, BD.MARK_STACK0, BS))
            # Head 0 -> val byte 0 at d=5: L2H0[MEM]=1, H1[MEM]=0.
            k.append(AP(3, BD.L2H0 + MEM_I, BS))
            k.append(AP(3, BD.H1 + MEM_I, -BS))
        else:
            # Heads 1-3 -> val bytes 1,2,3 via MEM_VAL_B1/B2/B3.
            k.append(AP(3, MEM_VAL_DIMS[h], BS))

        # === Slots 4..27: 24-bit binary address encoding (scale=10) ===
        scale = 10.0
        addr_dim = 4
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in (ab_lo, ab_hi):
                for bit in range(4):
                    for nk in range(16):
                        bit_val = 2 * ((nk >> bit) & 1) - 1
                        q.append(AP(addr_dim, nibble_base + nk, scale * bit_val))
                        k.append(AP(addr_dim, nibble_base + nk, scale * bit_val))
                    addr_dim += 1

        # === Slot 28: Per-head position gate ===
        q.append(AP(28, BD.CONST, -500.0))
        q.append(AP(28, byte_q_flag, 500.0))
        if h == 0:
            q.append(AP(28, BD.MARK_STACK0, 500.0))
        k.append(AP(28, BD.CONST, 5.0))

        # === V/O: copy byte value to OUTPUT (slots 32..63) ===
        v: list[AP] = []
        o: list[AO] = []
        for kk in range(16):
            v.append(AP(32 + kk, BD.CLEAN_EMBED_LO + kk, 1.0))
            v.append(AP(48 + kk, BD.CLEAN_EMBED_HI + kk, 1.0))
            o.append(AO(BD.OUTPUT_LO + kk, 32 + kk, 1.0))
            o.append(AO(BD.OUTPUT_HI + kk, 48 + kk, 1.0))

        head_idx = _L15_HEAD_LAYOUT_BY_NAME[
            f"layer15_memory_lookup.li_lc_stack0_h{h}"
        ]
        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=head_idx,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            # STEP_WINDOW_AUDIT_2026_06_10: LI/LC + STACK0 load heads
            # read MARK_MEM tokens by address — memory persists across
            # step boundaries by design (runtime slope=0.05 for the L15
            # primary load heads keeps the most-recent write dominant).
            # Declares the cross-step intent the verifier
            # would otherwise misclassify as CURRENT_STEP_ONLY violation.
            step_window=StepWindowConstraint.ANY_STEP,
        ))

    return tuple(specs)


def _layer15_memory_lookup_heads_0_3_specs_with_overrides(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """L15 heads 0-3 base + ``_suppress_l15_lookup_heads_0_3`` merged.

    Declarative replacement for the legacy post-bake patch
    :func:`_suppress_l15_lookup_heads_0_3`. The override cells are
    dict-merged into the per-head ``q``/``k``/``v``/``o`` maps so the
    lowered weights are byte-identical with running the suppress
    helper after :func:`_layer15_memory_lookup_heads_0_3_specs`.

    Row wipes (``attn.W_q.data[base + row, :] = 0.0`` etc.) at
    rows 35-43, 58-63 in the legacy helper are no-ops here because
    the base spec does not write Q/K at those slots, except for V
    slot 63 and O column 63 (head 0 only, where the
    ``nonpop_stack0_marker_blocker`` wipe drops the base spec's
    ``CLEAN_EMBED_HI+15`` -> ``OUTPUT_HI+15`` cell).
    """

    PC_I = 0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4

    base_specs = _layer15_memory_lookup_heads_0_3_specs(BD)
    merged: list[DeclarativeAttentionHeadSpec] = []

    byte_q_flags = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]

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

        # === local_slot_scale: byte-0 nibble bit rows (slots 4..11) ===
        local_slot_scale = 100.0
        for nibble_offset, nibble_base in (
            (0, BD.ADDR_B0_LO), (4, BD.ADDR_B0_HI),
        ):
            for bit in range(4):
                row = 4 + nibble_offset + bit
                for k in range(16):
                    bit_val = 2 * ((k >> bit) & 1) - 1
                    q_map[(row, nibble_base + k)] = (
                        local_slot_scale * bit_val
                    )
                    k_map[(row, nibble_base + k)] = (
                        local_slot_scale * bit_val
                    )

        # === one-hot rows 43+k: byte-0 lo nibble match ===
        local_slot_onehot_scale = 100.0
        for k in range(16):
            row = 43 + k
            q_map[(row, BD.CONST)] = -local_slot_onehot_scale
            q_map[(row, BD.ADDR_B0_LO + k)] = local_slot_onehot_scale
            q_map[(row, BD.OP_LI_RELAY)] = local_slot_onehot_scale
            if head == 0:
                q_map[(row, BD.OP_LC_RELAY)] = local_slot_onehot_scale
            k_map[(row, BD.ADDR_B0_LO + k)] = local_slot_onehot_scale

        if head == 0:
            # === Strong default blocker + explicit LI/LC/pop restores ===
            lookup_bias = 200000.0
            q_map[(0, BD.CONST)] = -lookup_bias
            q_map[(0, BD.OP_LI_RELAY)] = lookup_bias
            q_map[(0, BD.OP_LC_RELAY)] = lookup_bias
            q_map[(0, BD.OP_LI)] = lookup_bias
            q_map[(0, BD.OP_LC)] = lookup_bias
            q_map[(0, BD.CMP + 3)] = lookup_bias / 4.0
            q_map[(1, BD.CMP + 3)] = 12.5

            non_load_suppression = -1000000.0
            q_map[(0, BD.OP_JSR)] = non_load_suppression
            q_map[(0, BD.OP_ENT)] = non_load_suppression
            q_map[(0, BD.OP_LEA)] = non_load_suppression
            q_map[(0, BD.OP_IMM)] = non_load_suppression

            q_map[(0, BD.MARK_STACK0)] = 75000.0
            q_map[(0, BD.HAS_SE)] = 75000.0
            q_map[(0, BD.ADDR_B0_LO + 8)] = 75000.0
            q_map[(0, BD.ADDR_B0_HI + 14)] = 75000.0
            q_map[(0, BD.ADDR_B0_HI + 15)] = -100000.0
            q_map[(0, BD.IS_BYTE)] = -2000.0
            q_map[(1, BD.IS_BYTE)] = -50.0
            q_map[(1, BD.MARK_STACK0)] = 50.0
            q_map[(1, BD.HAS_SE)] = 50.0
            q_map[(1, BD.ADDR_B0_LO + 8)] = 50.0
            q_map[(1, BD.ADDR_B0_HI + 14)] = 50.0
            q_map[(1, BD.ADDR_B0_HI + 15)] = -150.0
            q_map[(28, BD.IS_BYTE)] = -500.0
            q_map[(28, BD.CONST)] = -20000.0
            q_map[(28, BD.MARK_AX)] = 20000.0
            q_map[(28, BD.MARK_STACK0)] = 20000.0

            # Row 42: top-store e0 signature blocker.
            for row, low, high in ((42, 0, 14),):
                q_map[(row, BD.CONST)] = -60000.0
                q_map[(row, BD.MARK_STACK0)] = 10000.0
                q_map[(row, BD.HAS_SE)] = 10000.0
                q_map[(row, BD.MEM_STORE)] = 10000.0
                q_map[(row, BD.EMBED_LO + low)] = 10000.0
                q_map[(row, BD.EMBED_HI + high)] = 10000.0
                q_map[(row, BD.ADDR_B0_LO + low)] = 10000.0
                q_map[(row, BD.ADDR_B0_HI + high)] = 10000.0
                q_map[(row, BD.OP_LI_RELAY)] = 50000.0
                q_map[(row, BD.OP_LC_RELAY)] = 50000.0
                k_map[(row, BD.CONST)] = 20.0

            # Rows 59, 60, 61: wipe Q/K (base has no writes here either).
            # The base spec writes V at (slot, dim)=(59, CLEAN_EMBED_HI+11),
            # (60, CLEAN_EMBED_HI+12), (61, CLEAN_EMBED_HI+13). These rows
            # are valid V cells; only rows 62, 63 are touched by the V
            # wipe via slot 63 below. Q/K at 59-61 stays empty: no base
            # writes to drop.

            # Row 60: top-store e8 signature.
            row = 60
            q_map[(row, BD.CONST)] = -30000.0
            q_map[(row, BD.MARK_STACK0)] = 10000.0
            q_map[(row, BD.MARK_SP)] = -100000.0
            q_map[(row, BD.HAS_SE)] = 10000.0
            q_map[(row, BD.MEM_STORE)] = 150000.0
            q_map[(row, BD.ADDR_B0_LO + 8)] = 10000.0
            q_map[(row, BD.ADDR_B0_HI + 14)] = 10000.0
            q_map[(row, BD.ADDR_B0_HI + 15)] = -20000.0
            k_map[(row, BD.CONST)] = -20.0

            # Row 59: preserve e8 (STACK0 + HAS_SE + nibble match).
            preserve_e8_s = 5000.0
            row = 59
            q_map[(row, BD.CONST)] = -3.5 * preserve_e8_s
            q_map[(row, BD.MARK_STACK0)] = preserve_e8_s
            q_map[(row, BD.HAS_SE)] = preserve_e8_s
            q_map[(row, BD.ADDR_B0_LO + 8)] = preserve_e8_s
            q_map[(row, BD.ADDR_B0_HI + 14)] = preserve_e8_s
            q_map[(row, BD.IS_BYTE)] = -4.0 * preserve_e8_s
            q_map[(row, BD.MEM_STORE)] = -5.0 * preserve_e8_s
            k_map[(row, BD.ADDR_B0_LO + 8)] = preserve_e8_s
            k_map[(row, BD.ADDR_B0_HI + 14)] = preserve_e8_s
            k_map[(row, BD.STACK0_BYTE0)] = 5.0 * preserve_e8_s

            # Row 61: AX LI/LC e8 value discriminator.
            ax_li_e8_s = 10.0
            row = 61
            q_map[(row, BD.CONST)] = -3.5 * ax_li_e8_s
            q_map[(row, BD.MARK_AX)] = ax_li_e8_s
            q_map[(row, BD.OP_LI_RELAY)] = ax_li_e8_s
            q_map[(row, BD.OP_LC_RELAY)] = ax_li_e8_s
            q_map[(row, BD.MARK_STACK0)] = 2.5 * ax_li_e8_s
            q_map[(row, BD.ADDR_B0_LO + 8)] = ax_li_e8_s
            q_map[(row, BD.ADDR_B0_HI + 14)] = ax_li_e8_s
            q_map[(row, BD.IS_BYTE)] = -4.0 * ax_li_e8_s
            q_map[(row, BD.MEM_STORE)] = -4.0 * ax_li_e8_s
            k_map[(row, BD.MEM_VAL_B1)] = ax_li_e8_s
            k_map[(row, BD.ADDR_B0_LO + 8)] = ax_li_e8_s
            k_map[(row, BD.ADDR_B0_HI + 14)] = ax_li_e8_s

            # Row 58: early-ENT STACK0 discriminator.
            # The legacy helper wipes Q/K at slot 58 first; base spec
            # has no writes there, so the wipe is a no-op.
            #
            # OPCODE-BROADCAST HARDENING (2026-06-11, var_simple_12 / id 262):
            # the original gate cleared its -1e5 CONST bias with OP_ENT alone
            # (Q weight 2e8). OP_ENT does NOT stay one-hot at its own marker --
            # it BROADCASTS in-step onto every row at magnitude ~12-17 (audit
            # docs/OPCODE_BROADCAST_BLOCKER_AUDIT_2026_06_11.md). At 2e8 * 12.6
            # = 2.5e9 it overran even the -2e9 single-marker NOT-blockers, so on
            # the step-2 LEA PC byte0 prediction row (MARK_PC=1, OP_ENT=12.6,
            # MARK_STACK0=0) the discriminator fired, forced head-0 to attend to
            # an operand position, and copied its CLEAN_EMBED into OUTPUT_LO+0
            # at scale 40 -- flooding the PC byte to 0x00 and desyncing the
            # whole program (probe_var_full_chain.py 262 step-2 LEA PC byte0).
            # FIX: make MARK_STACK0 a HARD requirement. We offset CONST by
            # -stack0_gate and MARK_STACK0 by +stack0_gate so the net delta is
            # ZERO on the legitimate firing row (MARK_STACK0=1) -- byte-identical
            # there -- while on any non-STACK0 row (MARK_STACK0=0) the -1e10 bias
            # buries OP_ENT's largest broadcast (2e8 * ~17.5 = 3.5e9). OP_ENT
            # stays as the in-step confirming term; the K-side OP_ENT match is
            # unchanged.
            early_ent_stack0_q = 100000.0
            stack0_gate = 10000000000.0  # 1e10 >> 2e8 * OP_ENT_broadcast_max
            row = 58
            q_map[(row, BD.OP_ENT)] = 200000000.0
            q_map[(row, BD.MARK_STACK0)] = early_ent_stack0_q + stack0_gate
            q_map[(row, BD.CONST)] = -early_ent_stack0_q - stack0_gate
            q_map[(row, BD.IS_BYTE)] = -2000000000.0
            for marker_dim in (
                BD.MARK_AX, BD.MARK_PC, BD.MARK_SP,
                BD.MARK_BP, BD.MARK_MEM,
            ):
                q_map[(row, marker_dim)] = -2000000000.0
            k_map[(row, BD.OP_ENT)] = 10000.0

            # Row 34: pop_low8 nibble bias.
            row = 34
            q_map[(row, BD.CONST)] = -4000.0
            q_map[(row, BD.MARK_STACK0)] = 2000.0
            q_map[(row, BD.HAS_SE)] = 1000.0
            q_map[(row, BD.CMP + 3)] = 1000.0
            q_map[(row, BD.ADDR_B0_LO + 0)] = 1000.0
            q_map[(row, BD.ADDR_B0_LO + 8)] = 1000.0
            q_map[(row, BD.IS_BYTE)] = -10000.0
            q_map[(row, BD.MARK_SP)] = -10000.0
            q_map[(row, BD.MEM_STORE)] = -20000.0
            k_map[(row, BD.ADDR_B0_LO + 8)] = 1000.0
        else:
            # === Heads 1-3: byte_q_flags-gated overrides ===
            q_map[(0, BD.MARK_STACK0)] = -100000.0
            q_map[(0, BD.MARK_SP)] = -100000.0
            q_map[(28, BD.CONST)] = -20000.0
            q_map[(28, byte_q_flags[head])] = 20000.0

            # === MARK_AX hard darkening (slot 64): var_simple_12 / id 262 ===
            # Heads 1-3 are the LI/LC AX byte-1/2/3 value relays; they fire on
            # the byte-1/2/3 PREDICTION rows, which carry BYTE_INDEX_0/1/2 and
            # MARK_AX=0 (byte_q_flags[head], NOT MARK_AX). Head 0 alone owns the
            # MARK_AX (byte-0) row. But these heads had only a weak
            # MARK_AX=-20000 default blocker (slots 31/32) -- not enough to
            # drive their softmax1 (ZFOD) scores negative -- so on ANY MARK_AX
            # register-emit row WITHOUT an LI/LC load (e.g. the var_simple_12
            # step-3 PSH AX byte-0 emit row: MARK_AX=1, OP_LI_RELAY=0) all three
            # heads stayed positive and diffuse, averaging operand CLEAN_EMBED
            # bytes into OUTPUT at value_scale=40. The summed result (0x0f from
            # CLEAN_EMBED nibble debris) overwrote the residual-carried
            # psh_ax_broadcast value 0xe8 at block 30 / logical L19
            # (probe_var_full_chain.py 262: genesis block 30, AX byte0
            # 0xe8 -> 0x0f).
            #
            # FIX (mirrors head-0's slot-58 hard-marker darkening landed in
            # 8ad47bf4): a hard subtractive MARK_AX NOT-blocker on a free slot
            # (64). K[CONST]=1 makes the slot score MARK_AX-row positions at
            # -2e9 (CONST is present everywhere), driving EVERY score below the
            # softmax1 anchor (0) so the head outputs ZERO (wsum -> 0) and the
            # residual 0xe8 survives. On the legit LI/LC byte-1/2/3 firing rows
            # MARK_AX=0, so the slot contributes nothing and the head is
            # BYTE-IDENTICAL (verified spec_k=0: max|diff| = 0.0, head-0 LI
            # byte-0 value still 0x2a). Pure subtractive -- no net-zero
            # compensation -- so it cannot perturb any legit (MARK_AX=0) row.
            mark_ax_dark_slot = 64
            q_map[(mark_ax_dark_slot, BD.MARK_AX)] = -2000000000.0
            k_map[(mark_ax_dark_slot, BD.CONST)] = 1.0
            # Slot 3 K-side per-head re-tuning.
            for dim in (
                BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3,
                BD.H2 + MEM_I, BD.H3 + MEM_I,
            ):
                k_map[(3, dim)] = 0.0
            if head == 1:
                k_map[(3, BD.MEM_VAL_B2)] = 60.0
            elif head == 2:
                k_map[(3, BD.MEM_VAL_B3)] = 60.0
            elif head == 3:
                k_map[(3, BD.H3 + MEM_I)] = 60.0
                k_map[(3, BD.H2 + MEM_I)] = -60.0

        # === source_gate (slot 37): wipe K then re-author per head ===
        # Base has no Q/K at slot 37 -- wipe is a no-op.
        source_gate = 37
        q_map[(source_gate, BD.CONST)] = 0.0
        if head == 0:
            q_map[(source_gate, BD.MARK_STACK0)] = 3000.0
        else:
            q_map[(source_gate, byte_q_flags[head])] = 3000.0
        source_key_s = 10.0
        k_map[(source_gate, BD.CONST)] = -source_key_s
        k_map[(source_gate, BD.MEM_STORE)] = 0.5 * source_key_s
        if head == 0:
            k_map[(source_gate, BD.L2H0 + MEM_I)] = source_key_s
            k_map[(source_gate, BD.H1 + MEM_I)] = -70.0
        else:
            k_map[(source_gate, BD.H1 + MEM_I)] = -70.0
            if head == 1:
                k_map[(source_gate, BD.MEM_VAL_B2)] = source_key_s
            elif head == 2:
                k_map[(source_gate, BD.MEM_VAL_B3)] = source_key_s
            elif head == 3:
                k_map[(source_gate, BD.H3 + MEM_I)] = source_key_s
                k_map[(source_gate, BD.H2 + MEM_I)] = -source_key_s
        # SP/BP register byte blockers on source_gate K.
        for marker_i in (SP_I, BP_I):
            for dim in (
                BD.H1 + marker_i,
                BD.H2 + marker_i,
                BD.H3 + marker_i,
                BD.L2H0 + marker_i,
            ):
                k_map[(source_gate, dim)] = -80.0

        # === load_source_gate (slot 39): wipe Q/K then re-author ===
        load_source_gate = 39
        load_source_gate_s = 5000.0
        load_source_key_s = 30.0
        if head == 0:
            q_map[(load_source_gate, BD.OP_LI_RELAY)] = load_source_gate_s
            q_map[(load_source_gate, BD.OP_LC_RELAY)] = load_source_gate_s
            q_map[(load_source_gate, BD.MARK_AX)] = load_source_gate_s
            q_map[(load_source_gate, BD.CMP + 3)] = 2000.0
            q_map[(load_source_gate, BD.CONST)] = -1.5 * load_source_gate_s
            k_map[(load_source_gate, BD.MEM_VAL_B1)] = (
                2.0 * load_source_key_s
            )
            k_map[(load_source_gate, BD.MEM_ADDR_SRC)] = 40.0
        else:
            q_map[(load_source_gate, BD.OP_LI_RELAY)] = load_source_gate_s
            q_map[(load_source_gate, byte_q_flags[head])] = (
                load_source_gate_s
            )
            q_map[(load_source_gate, BD.CONST)] = -1.5 * load_source_gate_s
            if head == 1:
                k_map[(load_source_gate, BD.MEM_VAL_B2)] = (
                    2.0 * load_source_key_s
                )
            elif head == 2:
                k_map[(load_source_gate, BD.MEM_VAL_B3)] = (
                    2.0 * load_source_key_s
                )
            elif head == 3:
                k_map[(load_source_gate, BD.H3 + MEM_I)] = (
                    2.0 * load_source_key_s
                )
                k_map[(load_source_gate, BD.H2 + MEM_I)] = (
                    -2.0 * load_source_key_s
                )
            k_map[(load_source_gate, BD.MEM_ADDR_SRC)] = 40.0

        # === marker_value_gate (slot 40): head-0 only ===
        if head == 0:
            marker_value_gate = 40
            marker_value_gate_s = 1000.0
            q_map[(marker_value_gate, BD.OP_LI)] = marker_value_gate_s
            q_map[(marker_value_gate, BD.OP_LC)] = marker_value_gate_s
            k_map[(marker_value_gate, BD.MEM_VAL_B1)] = 80.0
            k_map[(marker_value_gate, BD.MEM_ADDR_SRC)] = 40.0
            k_map[(marker_value_gate, BD.CONST)] = -60.0

        # === O scaling: value_scale=40.0 for OUTPUT band ===
        # Replaces base spec's slot 32+k -> OUTPUT_LO+k weight=1.0
        # with weight=40.0; same for HI.
        value_scale = 40.0
        for k in range(16):
            o_map[(BD.OUTPUT_LO + k, 32 + k)] = value_scale
            o_map[(BD.OUTPUT_HI + k, 48 + k)] = value_scale

        # === addsub_blocker (slot 41) ===
        addsub_blocker = 41
        q_map[(addsub_blocker, BD.TEMP + 8)] = 10000.0
        q_map[(addsub_blocker, BD.TEMP + 9)] = 10000.0
        k_map[(addsub_blocker, BD.CONST)] = -20.0

        # === sp_byte_blocker (slot 62) ===
        sp_byte_blocker = 62
        q_map[(sp_byte_blocker, BD.H1 + 2)] = 100000.0
        q_map[(sp_byte_blocker, BD.MARK_BP)] = 100000.0
        q_map[(sp_byte_blocker, BD.TEMP + 10)] = 100000.0
        q_map[(sp_byte_blocker, BD.TEMP + 24)] = 100000.0
        q_map[(sp_byte_blocker, BD.IS_BYTE)] = 0.0
        k_map[(sp_byte_blocker, BD.CONST)] = -300000.0
        if head == 0:
            q_map[(sp_byte_blocker, BD.IS_BYTE)] = 500000.0
            q_map[(sp_byte_blocker, BD.OP_ENT)] = 500000.0

        # === pc_byte_blocker (slot 35) ===
        pc_byte_blocker = 35
        q_map[(pc_byte_blocker, BD.H1 + PC_I)] = 100000.0
        q_map[(pc_byte_blocker, BD.MARK_PC)] = 100000000.0
        q_map[(pc_byte_blocker, BD.IS_BYTE)] = 0.0
        k_map[(pc_byte_blocker, BD.CONST)] = -100000.0
        if head == 0:
            q_map[(pc_byte_blocker, BD.MARK_STACK0)] = 10000.0
            q_map[(pc_byte_blocker, BD.HAS_SE)] = 10000.0
            q_map[(pc_byte_blocker, BD.CMP + 3)] = 10000.0
            q_map[(pc_byte_blocker, BD.ADDR_B0_LO + 8)] = 10000.0
            q_map[(pc_byte_blocker, BD.ADDR_B0_HI + 15)] = 10000.0
            q_map[(pc_byte_blocker, BD.OP_LI_RELAY)] = -10000.0
            q_map[(pc_byte_blocker, BD.OP_LC_RELAY)] = -10000.0

            # === stack0_preserve (slot 36): head-0 only ===
            stack0_preserve_row = 36
            stack0_preserve_s = 1.0
            q_map[(stack0_preserve_row, BD.CONST)] = (
                -1.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.MARK_STACK0)] = (
                3.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.HAS_SE)] = (
                1.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.CMP + 3)] = (
                -5.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.MEM_STORE)] = (
                -10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.IS_BYTE)] = (
                -10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.MARK_AX)] = (
                -10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.H1 + AX_I)] = (
                10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.OP_LI_RELAY)] = (
                10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.OP_LC_RELAY)] = (
                10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.MARK_PC)] = (
                -10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.MARK_SP)] = (
                -10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.MARK_BP)] = (
                -10.0 * stack0_preserve_s
            )
            q_map[(stack0_preserve_row, BD.MARK_MEM)] = (
                -10.0 * stack0_preserve_s
            )
            k_map[(stack0_preserve_row, BD.CONST)] = (
                -2.0 * stack0_preserve_s
            )
            k_map[(stack0_preserve_row, BD.H1 + 10)] = (
                2.0 * stack0_preserve_s
            )
            k_map[(stack0_preserve_row, BD.BYTE_INDEX_0)] = (
                2.0 * stack0_preserve_s
            )
            k_map[(stack0_preserve_row, BD.OP_ENT)] = (
                -0.25 * stack0_preserve_s
            )
            for marker_i in range(5):
                k_map[(stack0_preserve_row, BD.H1 + marker_i)] = (
                    -4.0 * stack0_preserve_s
                )

        # === mem_addr_byte_blocker (slot 36) for heads 1-3 ===
        if head in (1, 2, 3):
            mem_addr_byte_blocker = 36
            q_map[(mem_addr_byte_blocker, BD.H1 + MEM_I)] = 100000.0
            k_map[(mem_addr_byte_blocker, BD.CONST)] = -20.0

        # === nonpop_stack0_marker_blocker (slot 63) ===
        # Legacy: ``attn.W_v[base+63, :] = 0`` then explicit Q/K writes,
        # then ``attn.W_o[:, base+63] = 0`` (wipe V row and O column 63).
        # In the dict-merged form: drop V at slot 63 and any O writes
        # targeting slot 63 (base has V (63, CLEAN_EMBED_HI+15)=1 and
        # O (OUTPUT_HI+15, 63)=1; rescaled to 40.0 above).
        nonpop_stack0_marker_blocker = 63
        v_map.pop((nonpop_stack0_marker_blocker, BD.CLEAN_EMBED_HI + 15), None)
        # Drop any O cell with slot=63 (we just set OUTPUT_HI+15 -> 63 = 40)
        for out_dim_key in [
            key for key in o_map
            if key[1] == nonpop_stack0_marker_blocker
        ]:
            o_map.pop(out_dim_key, None)
        q_map[(nonpop_stack0_marker_blocker, BD.MARK_STACK0)] = 60000.0
        q_map[(nonpop_stack0_marker_blocker, BD.CMP + 3)] = -15000.0
        q_map[(nonpop_stack0_marker_blocker, BD.IS_BYTE)] = 60000.0
        q_map[(nonpop_stack0_marker_blocker, BD.OP_LI_RELAY)] = -60000.0
        q_map[(nonpop_stack0_marker_blocker, BD.OP_LC_RELAY)] = -60000.0
        q_map[(nonpop_stack0_marker_blocker, BD.ADDR_B0_LO + 8)] = -40000.0
        q_map[(nonpop_stack0_marker_blocker, BD.ADDR_B0_HI + 14)] = -30000.0
        k_map[(nonpop_stack0_marker_blocker, BD.CONST)] = -20.0
        top_store_e8_from_e0_s = 10000.0
        q_map[(nonpop_stack0_marker_blocker, BD.MEM_STORE)] = (
            top_store_e8_from_e0_s
        )
        q_map[(nonpop_stack0_marker_blocker, BD.EMBED_LO + 8)] = (
            top_store_e8_from_e0_s
        )
        q_map[(nonpop_stack0_marker_blocker, BD.EMBED_HI + 14)] = (
            top_store_e8_from_e0_s
        )
        q_map[(nonpop_stack0_marker_blocker, BD.ADDR_B0_LO + 0)] = (
            top_store_e8_from_e0_s
        )
        q_map[(nonpop_stack0_marker_blocker, BD.ADDR_B0_HI + 14)] = (
            -2.0 * top_store_e8_from_e0_s
        )

        # === current_store_blocker (slot 38) ===
        current_store_blocker = 38
        current_store_block_s = 10000.0
        q_map[(current_store_blocker, BD.MARK_MEM)] = current_store_block_s
        q_map[(current_store_blocker, BD.H3 + MEM_I)] = current_store_block_s
        k_map[(current_store_blocker, BD.CONST)] = -20.0

        # === Broad current-MEM-section blockers on slots 0, 28-33 ===
        for slot in (0, 28, 29, 30, 31, 32, 33):
            q_map[(slot, BD.MARK_MEM)] = -100000.0
            q_map[(slot, BD.H3 + MEM_I)] = -100000.0
        # Slot 29: extras (PC H1).
        q_map[(29, BD.MARK_MEM)] = -100000.0
        q_map[(29, BD.H3 + MEM_I)] = -100000.0
        q_map[(29, BD.H1 + PC_I)] = -20000.0
        k_map[(29, BD.CONST)] = 5.0
        # Slot 30: AX H1 blocker.
        q_map[(30, BD.H1 + AX_I)] = -20000.0
        k_map[(30, BD.CONST)] = 5.0
        # Slot 31: LI/LC restore (head 0 also AX blocker reverse).
        q_map[(31, BD.OP_LI_RELAY)] = 20000.0
        if head == 0:
            q_map[(31, BD.OP_LC_RELAY)] = 20000.0
        else:
            q_map[(31, BD.MARK_AX)] = -20000.0
        q_map[(31, BD.OP_SI)] = -20000.0
        q_map[(31, BD.OP_SC)] = -20000.0
        k_map[(31, BD.MEM_STORE)] = 5.0

        # Slot 32: AX marker default blocker.
        q_map[(32, BD.MARK_AX)] = -20000.0
        k_map[(32, BD.CONST)] = 5.0
        if head == 0:
            q_map[(33, BD.OP_LI_RELAY)] = 20000.0
            q_map[(33, BD.OP_LC_RELAY)] = 20000.0
            k_map[(33, BD.MEM_STORE)] = 5.0

        # === C4_L15_LI_SUPPR_INERT: un-bury the head-0 load on LI/LC rows ===
        # See _l15_li_load_suppressor_inert_on for the full root + proof. The
        # OP_ENT broadcast in a function frame mis-fires the head-0 PC/SP/STACK0
        # discriminator slots on the LI/LC LOAD query row, burying the
        # content-addressable lookup. Neutralise them on genuine load rows
        # (OP_LI_RELAY/OP_LC_RELAY active) so the address bits (slots 4-27)
        # decide the store -- exactly as on the clean si_li path. The genuine
        # PC/SP/STACK0/pop rows these slots guard carry OP_LI_RELAY==
        # OP_LC_RELAY==0, so the overrides are no-ops there (byte-identical).
        if head == 0 and _l15_li_load_suppressor_inert_on():
            # These head-0 slots are PC/SP/STACK0-marker/pop discriminators that
            # MIS-FIRE on the OP_ENT-broadcast LI/LC load query row, burying /
            # mis-ordering the content-addressable lookup so a frame-local LI
            # returns 0 (the func/nested/rec/var first-LI wall). NEUTRALISING
            # them lets the address bits (slots 4-27) alone pick the store, the
            # clean si_li ordering (verified: zeroing these makes the func LI
            # attend the value store; var_simple_0/12 + rec still decode).
            #
            # Implementation: a per-suppressor K-side CANCEL slot (free
            # over-width slot, head 0) whose Q EQUALS the suppressor's full Q
            # (copied dim-for-dim) and whose K == -(suppressor's K). The cancel
            # slot's per-key product is therefore -(suppressor's per-key
            # product) on EVERY row, so suppressor + cancel == 0 everywhere ->
            # the slot is fully inert (equivalent to zeroing it), but expressed
            # additively so flag-OFF (cancel slots omitted) is byte-identical
            # with HEAD. slot 58's K is OP_ENT and its Q/K are 1e10/1e4 scale,
            # so RESCALE slot 58 by 1e4 first (the cancel of two ~1e6 numbers is
            # precision-safe; a 1e10 cancel is not).
            _row58_rescale = 10000.0
            for _d, _w in (
                (BD.OP_ENT, 200000000.0 / _row58_rescale),
                (BD.MARK_STACK0,
                 (early_ent_stack0_q + stack0_gate) / _row58_rescale),
                (BD.CONST,
                 -(early_ent_stack0_q + stack0_gate) / _row58_rescale),
                (BD.IS_BYTE, -2000000000.0 / _row58_rescale),
                (BD.MARK_AX, -2000000000.0 / _row58_rescale),
                (BD.MARK_PC, -2000000000.0 / _row58_rescale),
                (BD.MARK_SP, -2000000000.0 / _row58_rescale),
                (BD.MARK_BP, -2000000000.0 / _row58_rescale),
                (BD.MARK_MEM, -2000000000.0 / _row58_rescale),
            ):
                q_map[(58, _d)] = _w
            k_map[(58, BD.OP_ENT)] = 1.0
            _GATE = {34: 70, 35: 64, 58: 65, 59: 66, 60: 67, 61: 68, 62: 69}
            for _suppr, _gate in _GATE.items():
                for (_s, _d), _w in [
                    ((s, d), w) for (s, d), w in q_map.items() if s == _suppr
                ]:
                    q_map[(_gate, _d)] = _w
                for (_s, _d), _w in [
                    ((s, d), w) for (s, d), w in k_map.items() if s == _suppr
                ]:
                    k_map[(_gate, _d)] = -_w

        new_q = tuple(
            AP(slot, dim, weight) for (slot, dim), weight in q_map.items()
        )
        new_k = tuple(
            AP(slot, dim, weight) for (slot, dim), weight in k_map.items()
        )
        new_v = tuple(
            AP(slot, dim, weight) for (slot, dim), weight in v_map.items()
        )
        new_o = tuple(
            AO(out_dim, slot, weight)
            for (out_dim, slot), weight in o_map.items()
        )

        merged.append(DeclarativeAttentionHeadSpec(
            head_idx=head,
            q=new_q,
            k=new_k,
            v=new_v,
            o=new_o,
            # STEP_WINDOW_AUDIT_2026_06_10: propagate the base spec's
            # step-window declaration so the override pass doesn't drop
            # the ANY_STEP annotation on heads 0-3 (LI/LC + STACK0 load).
            step_window=spec.step_window,
            alibi_slope=spec.alibi_slope,
        ))

    return tuple(merged)


def _layer15_memory_lookup_lev_heads_4_11_specs(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative L15 heads 4-11 mirroring ``_set_layer15_memory_lookup_lev_heads_4_11``.

    Phase 7.C.3 follow-up: replaces the imperative ``W_q/W_k/W_v/W_o``
    writes in :func:`vm_step._set_layer15_memory_lookup_lev_heads_4_11`
    with a tuple of :class:`DeclarativeAttentionHeadSpec` so the
    LEV-only saved_bp (heads 4-7, read ``memory[BP]``) and return_addr
    (heads 8-11, read ``memory[BP+8]``) load heads are authored as
    data, lowered by :meth:`CompilerIR.lower_attention` through
    :func:`Primitives.generate_attention_heads`.

    Only emitted by :func:`_layer15_memory_lookup_ir` when
    ``num_heads >= 12`` (the 17-layer LEV build). Bodies mirror the
    legacy helper cell-for-cell; the lowerer is assignment, so every
    write is preserved bit-for-bit.

    Heads 8 and 9 are intentionally emitted even though the L15
    layout assigns those slots to other ops
    (``layer15_alu_high_byte_relay`` for head 8;
    ``layer15_memory_lookup.pop_d8_to_e0`` for head 9 via the suppress
    helper). The legacy imperative helper wrote those rows too --
    they are later overwritten by the alu_high_byte_relay spec and
    the pop_d8 rewrite -- so emitting the same writes here keeps the
    pre-overwrite snapshot byte-identical with the legacy bake.
    """

    AX_I = 1
    BP_I = 3

    specs: list[DeclarativeAttentionHeadSpec] = []

    # === Heads 4-7: saved_bp lookup from memory[BP] (LEV) ===
    for h in range(4, 8):
        byte_idx = h - 4  # 0..3

        q: list[AP] = []
        k: list[AP] = []

        # === Slot 0: bias ===
        if byte_idx == 0:
            q.append(AP(0, BD.CONST, -4000.0))
            q.append(AP(0, BD.OP_LEV, 2000.0))
            q.append(AP(0, BD.MARK_BP, 2000.0))
            q.append(AP(0, BD.MARK_PC, -25000.0))
            q.append(AP(0, BD.MARK_SP, -100000.0))
            k.append(AP(0, BD.CONST, 10.0))
        else:
            q.append(AP(0, BD.CONST, 10.0))
            k.append(AP(0, BD.CONST, 10.0))

        # === Slot 1: store anchor ===
        if byte_idx == 0:
            q.append(AP(1, BD.CONST, -50.0))
            q.append(AP(1, BD.OP_LEV, 50.0))
            q.append(AP(1, BD.MARK_BP, 50.0))
            q.append(AP(1, BD.MARK_PC, -200.0))
            q.append(AP(1, BD.MARK_SP, -200.0))
        else:
            q.append(AP(1, BD.CONST, 10.0))
        k.append(AP(1, BD.MEM_STORE, 100.0))
        k.append(AP(1, BD.CONST, -50.0))

        # === Slot 2: ZFOD negative offset ===
        q.append(AP(2, BD.CONST, -96.0))
        k.append(AP(2, BD.MEM_STORE, 50.0))

        # === Slot 3: byte selection ===
        if byte_idx == 0:
            BS = 150.0
            q.append(AP(3, BD.CONST, -BS))
            q.append(AP(3, BD.OP_LEV, BS))
            q.append(AP(3, BD.MARK_BP, BS))
            q.append(AP(3, BD.MARK_PC, -BS * 20))
            q.append(AP(3, BD.MARK_SP, -BS * 20))
            k.append(AP(3, BD.MEM_VAL_B0, BS))
            k.append(AP(3, BD.CONST, -BS))
        else:
            BS = 60.0
            q.append(AP(3, BD.CONST, BS))
            MEM_VAL_DIMS = [
                None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3,
            ]
            k.append(AP(3, MEM_VAL_DIMS[byte_idx], BS))

        # === Slots 4..35: 32-dim one-hot address-byte-0 matching ===
        L_addr = 50.0
        for kk in range(16):
            q.append(AP(4 + kk, BD.ADDR_B0_LO + kk, L_addr))
            q.append(AP(4 + 16 + kk, BD.ADDR_B0_HI + kk, L_addr))
            k.append(AP(4 + kk, BD.ADDR_KEY + kk, L_addr))
            k.append(AP(4 + 16 + kk, BD.ADDR_KEY + 16 + kk, L_addr))

        # === Slot 36: per-head position gate (with AX-byte suppression) ===
        GATE_DIM = 36
        SUPPRESS_AX_BYTE = -50000.0
        q.append(AP(GATE_DIM, BD.IS_BYTE, SUPPRESS_AX_BYTE))
        q.append(AP(GATE_DIM, BD.H1 + AX_I, SUPPRESS_AX_BYTE))
        q.append(AP(GATE_DIM, BD.MARK_AX, SUPPRESS_AX_BYTE))
        if byte_idx == 0:
            q.append(AP(GATE_DIM, BD.CONST, -500.0))
            q.append(AP(GATE_DIM, BD.MARK_BP, 500.0))
            q.append(AP(GATE_DIM, BD.MARK_PC, -50000.0))
        elif byte_idx == 1:
            q.append(AP(GATE_DIM, BD.CONST, -500.0))
            q.append(AP(GATE_DIM, BD.BYTE_INDEX_1, 500.0))
            q.append(AP(GATE_DIM, BD.L1H1 + BP_I, 500.0))
            q.append(AP(GATE_DIM, BD.MARK_BP, -1000.0))
            q.append(AP(GATE_DIM, BD.MARK_PC, -50000.0))
        elif byte_idx == 2:
            q.append(AP(GATE_DIM, BD.CONST, -500.0))
            q.append(AP(GATE_DIM, BD.BYTE_INDEX_2, 500.0))
            q.append(AP(GATE_DIM, BD.H0 + BP_I, 500.0))
            q.append(AP(GATE_DIM, BD.MARK_BP, -1000.0))
            q.append(AP(GATE_DIM, BD.MARK_PC, -50000.0))
        elif byte_idx == 3:
            q.append(AP(GATE_DIM, BD.CONST, -500.0))
            q.append(AP(GATE_DIM, BD.BYTE_INDEX_3, 500.0))
            q.append(AP(GATE_DIM, BD.H1 + BP_I, 500.0))
            q.append(AP(GATE_DIM, BD.MARK_BP, -1000.0))
            q.append(AP(GATE_DIM, BD.MARK_PC, -50000.0))
        k.append(AP(GATE_DIM, BD.CONST, 5.0))

        # === V/O: copy byte value from CLEAN_EMBED to staging slots ===
        v: list[AP] = []
        o: list[AO] = []
        for kk in range(16):
            v.append(AP(32 + kk, BD.CLEAN_EMBED_LO + kk, 1.0))
            v.append(AP(48 + kk, BD.CLEAN_EMBED_HI + kk, 1.0))
        # Head 4 (byte 0) writes saved_bp byte 0 to OUTPUT_LO/HI.
        if byte_idx == 0:
            for kk in range(16):
                o.append(AO(BD.OUTPUT_LO + kk, 32 + kk, 1.0))
                o.append(AO(BD.OUTPUT_HI + kk, 48 + kk, 1.0))

        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=h,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            # STEP_WINDOW_AUDIT_2026_06_10: LEV saved_bp lookup heads
            # (4-7) read MEM_VAL_B[0..3] across step boundaries by
            # design — memory persistence is the whole point. Runtime
            # slope is 0.01 so the verifier flags these as
            # CURRENT_STEP_ONLY violations. ANY_STEP encodes the
            # cross-step-OK intent the runtime already follows.
            step_window=StepWindowConstraint.ANY_STEP,
        ))

    # === Heads 8-11: return_addr lookup from memory[BP+8] (LEV) ===
    # ADDR_B0 already shifted by L9 FFN; no extra +8 here.
    for h in range(8, 12):
        byte_idx = h - 8

        q = []
        k = []

        # === Slot 0: bias -- fire at PC marker when OP_LEV active ===
        q.append(AP(0, BD.CONST, -4000.0))
        q.append(AP(0, BD.OP_LEV, 2000.0))
        q.append(AP(0, BD.MARK_PC, 2000.0))
        k.append(AP(0, BD.CONST, 10.0))

        # === Slot 1: store anchor ===
        q.append(AP(1, BD.CONST, -50.0))
        q.append(AP(1, BD.OP_LEV, 50.0))
        q.append(AP(1, BD.MARK_PC, 50.0))
        k.append(AP(1, BD.MEM_STORE, 100.0))
        k.append(AP(1, BD.CONST, -50.0))

        # === Slot 2: ZFOD offset ===
        q.append(AP(2, BD.CONST, -96.0))
        k.append(AP(2, BD.MEM_STORE, 50.0))

        # === Slot 3: byte selection ===
        BS = 60.0
        q.append(AP(3, BD.CONST, -BS))
        q.append(AP(3, BD.OP_LEV, BS))
        q.append(AP(3, BD.MARK_PC, BS))
        MEM_VAL_DIMS = [
            BD.MEM_VAL_B0, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3,
        ]
        k.append(AP(3, MEM_VAL_DIMS[byte_idx], BS))

        # === Slots 4..35: 32-dim one-hot address-byte-0 matching ===
        L_addr = 50.0
        # Dims 4-19: byte 0 lo nibble (L9 FFN already shifted by +8)
        for kk in range(16):
            q.append(AP(4 + kk, BD.ADDR_B0_LO + kk, L_addr))
            k.append(AP(4 + kk, BD.ADDR_KEY + kk, L_addr))
        # Dims 20-35: byte 0 hi nibble (L9 FFN handles carry)
        for kk in range(16):
            q.append(AP(20 + kk, BD.ADDR_B0_HI + kk, L_addr))
            k.append(AP(20 + kk, BD.ADDR_KEY + 16 + kk, L_addr))

        # === Slot 36: position gate ===
        GATE_DIM = 36
        q.append(AP(GATE_DIM, BD.CONST, -500.0))
        q.append(AP(GATE_DIM, BD.MARK_PC, 500.0))
        q.append(AP(GATE_DIM, BD.MARK_BP, -1000.0))
        q.append(AP(GATE_DIM, BD.IS_BYTE, -50000.0))
        q.append(AP(GATE_DIM, BD.H1 + AX_I, -50000.0))
        q.append(AP(GATE_DIM, BD.MARK_AX, -50000.0))
        k.append(AP(GATE_DIM, BD.CONST, 5.0))

        # === Slot 37: memory position suppression ===
        SUPPRESS_DIM = 37
        k.append(AP(SUPPRESS_DIM, BD.CONST, 40000.0))
        k.append(AP(SUPPRESS_DIM, BD.MEM_STORE, -10000.0))
        q.append(AP(SUPPRESS_DIM, BD.CONST, -1000.0))

        # === V/O: copy byte value to TEMP at PC marker ===
        v = []
        o = []
        for kk in range(16):
            v.append(AP(32 + kk, BD.CLEAN_EMBED_LO + kk, 1.0))
            v.append(AP(48 + kk, BD.CLEAN_EMBED_HI + kk, 1.0))
        # Head 8 (byte 0) writes to TEMP for return_addr byte 0.
        if byte_idx == 0:
            for kk in range(16):
                o.append(AO(BD.TEMP + kk, 32 + kk, 1.0))
                o.append(AO(BD.TEMP + 16 + kk, 48 + kk, 1.0))
        # Heads 9-11: V slots populated but no O projection (legacy parity).

        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=h,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
            # STEP_WINDOW_AUDIT_2026_06_10: LEV return_addr lookup heads
            # (8-11) read MEM_VAL_B[0..3] across step boundaries by
            # design — memory persistence is the whole point. Runtime
            # slope is 0.01 (heads 10/11 get none) so the verifier flags
            # these as CURRENT_STEP_ONLY violations. ANY_STEP encodes
            # the cross-step-OK intent the runtime already follows.
            step_window=StepWindowConstraint.ANY_STEP,
        ))

    return tuple(specs)


def _layer15_memory_lookup_lev_heads_4_11_specs_with_overrides(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """L15 LEV heads 4-11 base + ``_suppress_l15_lookup_lev_blockers_4_11`` merged.

    Phase 7.C.3 follow-up: drops head 9 from the lev_heads emission
    (head 9 is fully replaced by the pop_d8_to_e0 declarative spec
    via :func:`_pop_d8_head_9_spec` whenever ``num_heads > 9``; the
    legacy bake wipes head 9 before writing pop_d8). The remaining
    heads 4-8, 10-11 carry the LEV saved_bp/return_addr writes plus
    the always-on ``MARK_MEM`` / ``H3+MEM_I`` blocker cells on rows
    0, 36, 37.

    The blocker writes mirror the legacy
    ``_suppress_l15_lookup_lev_blockers_4_11`` loop, which iterates
    ``range(4, min(num_heads, 12))``. Heads 4-11 are inside that
    range when ``num_heads >= 12`` (the only configuration where
    this builder runs), so every head we emit also gets blocker
    overrides.

    The blocker writes overlap with the lev_heads slot-0
    bias/store-anchor cells but only at distinct ``dim``s
    (``MARK_MEM`` and ``H3+MEM_I`` are not touched by the lev_heads
    spec), so the merged map is a union of the two sets without
    conflicts.
    """

    MEM_I = 4

    base_specs = _layer15_memory_lookup_lev_heads_4_11_specs(BD)
    merged: list[DeclarativeAttentionHeadSpec] = []
    for spec in base_specs:
        head = spec.head_idx
        if head == 9:
            # Head 9 is fully wiped + rewritten by the pop_d8_to_e0
            # spec (see :func:`_pop_d8_head_9_spec`); skip it here so
            # the lowered head-9 weights are exactly the pop_d8 spec's
            # writes (matching the legacy ``W_q[base:base+HD, :] = 0``
            # wipe in :func:`_suppress_l15_lookup_pop_d8_head_9`).
            continue
        q_map: dict[tuple[int, int], float] = {
            (w.slot, w.dim): w.weight for w in spec.q
        }
        k_map: dict[tuple[int, int], float] = {
            (w.slot, w.dim): w.weight for w in spec.k
        }
        for row in (0, 36, 37):
            q_map[(row, BD.MARK_MEM)] = -100000.0
            q_map[(row, BD.H3 + MEM_I)] = -100000.0
        new_q = tuple(
            AP(slot, dim, weight) for (slot, dim), weight in q_map.items()
        )
        new_k = tuple(
            AP(slot, dim, weight) for (slot, dim), weight in k_map.items()
        )
        merged.append(DeclarativeAttentionHeadSpec(
            head_idx=head,
            q=new_q,
            k=new_k,
            v=spec.v,
            o=spec.o,
            # STEP_WINDOW_AUDIT_2026_06_10: propagate the base spec's
            # step-window declaration so the override pass doesn't drop
            # the ANY_STEP annotation on heads 8-11 (LEV memory lookup).
            step_window=spec.step_window,
            alibi_slope=spec.alibi_slope,
        ))
    return tuple(merged)


def _layer15_memory_lookup_lev_blockers_only_specs(
    BD, max_head: int,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Blocker-only specs for heads 4..max_head-1 (non-LEV builds).

    For ``num_heads`` configurations between 5 and 11 inclusive the
    legacy ``_suppress_l15_lookup_lev_blockers_4_11`` writes the
    blocker cells on heads 4..num_heads-1 without the LEV saved_bp /
    return_addr load body. The lev_heads_4_11 spec is NOT emitted in
    that range (the legacy umbrella gate is ``num_heads >= 12``), so
    here we emit minimal one-spec-per-head Q-only writes that match
    the legacy blocker exactly.

    ``max_head`` is ``min(num_heads, 12)`` so the body iterates
    ``range(4, max_head)``. For ``num_heads <= 4`` the caller passes
    ``max_head = 4`` and this returns an empty tuple.
    """

    MEM_I = 4
    specs: list[DeclarativeAttentionHeadSpec] = []
    for head in range(4, min(max_head, 12)):
        q: list[AP] = []
        for row in (0, 36, 37):
            q.append(AP(row, BD.MARK_MEM, -100000.0))
            q.append(AP(row, BD.H3 + MEM_I, -100000.0))
        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=head,
            q=tuple(q),
            k=(),
            v=(),
            o=(),
        ))
    return tuple(specs)


def _layer15_memory_lookup_ir(
    dim_positions,
    HD,
    num_heads=None,
) -> CompilerIR:
    """Build the L15 ``memory_lookup`` CompilerIR, shape-parameterized.

    DSL Wave W7 (this commit) replaces the Phase 7.C.2
    :class:`RuntimeAttentionFragment` ``runtime_predicate`` escape hatch
    with compile-time Python branching on the shape variable
    ``num_heads``. The five conditional fragments in the legacy bake are
    now selected by plain ``if`` statements at IR-build time rather than
    by ``should_emit(attn)`` at lowering time. The :class:`AttentionOp`
    only carries the fragments that *actually fire* on the target
    attention block, so the lowering pass becomes unconditional.

    Selection table (driven by ``num_heads``):

    * ``memory_lookup.heads_0_3`` — universal LI/LC + STACK0 load heads
      (always emitted).
    * ``memory_lookup.lev_heads_4_11`` — LEV-only saved_bp /
      return_addr reads; emitted when ``num_heads >= 12``.
    * ``suppress.heads_0_3`` — load-side suppression for heads 0-3
      (always emitted).
    * ``suppress.lev_blockers_4_11`` — blocker rows on heads 4-11
      keeping them silent during current-store generation; emitted when
      ``num_heads > 4``. The legacy ``range(4, min(num_heads, 12))``
      body is a no-op below that threshold, so byte-identity is
      preserved either way -- the explicit gate keeps the IR's intent
      visible.
    * ``suppress.pop_d8_head_9`` — head 9 wipe + pop_d8_to_e0 rewrite;
      emitted when ``num_heads > 9``.

    ``num_heads=None`` (the audit / declarations-only path that calls
    ``compiler_ir_factory(dim_positions, head_dim)`` without a live
    ``attn``) emits only the always-on fragments. This matches the
    pre-W7 audit-path semantics: the symbolic execution ignores
    fragments entirely, so the audit-side IR shape is a soft subset.

    ``dim_positions`` is wrapped into a SetDim proxy and captured into
    each fragment via closure so the writers see the same dim layout
    they did in the imperative helpers. The legacy
    ``_set_layer15_memory_lookup_*`` / ``_suppress_l15_lookup_*``
    bodies are unchanged -- W7 only moves the runtime-shape switch
    one level up, from the lowerer to the builder.
    """
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    attn_op = ir.layer(0).attention

    # Phase 7.C.3: heads 0-3 (always-on LI/LC + STACK0 loads) are a
    # tuple of :class:`DeclarativeAttentionHeadSpec` with the
    # always-on suppress overrides merged in by
    # :func:`_layer15_memory_lookup_heads_0_3_specs_with_overrides`
    # (L14 mem_generation override-merge pattern).
    for spec in _layer15_memory_lookup_heads_0_3_specs_with_overrides(proxy):
        attn_op.append(
            spec,
            name=f"layer15_memory_lookup.li_lc_stack0_h{spec.head_idx}",
            metadata={"role": "load_heads", "always_on": True},
        )

    # Phase 7.C.3 follow-up: heads 4-11 (LEV-only saved_bp +
    # return_addr loads) are now a tuple of declarative specs built
    # by :func:`_layer15_memory_lookup_lev_heads_4_11_specs` with the
    # ``num_heads > 4`` blocker rows merged in. Only emitted when
    # ``num_heads >= 12`` (the 17-layer LEV build); the audit /
    # declarations-only path (``num_heads=None``) also emits them so
    # the lowered weight footprint stays a soft superset across
    # configurations.
    if num_heads is None or int(num_heads) >= 12:
        for spec in _layer15_memory_lookup_lev_heads_4_11_specs_with_overrides(
            proxy
        ):
            attn_op.append(
                spec,
                name=f"layer15_memory_lookup.lev_heads_4_11.h{spec.head_idx}",
                metadata={"role": "lev_heads", "shape": "num_heads >= 12"},
            )
    elif num_heads is not None and int(num_heads) > 4:
        # Non-LEV builds (5 <= num_heads < 12): emit only the
        # blocker cells for heads 4..num_heads-1. The lev_heads body
        # is not emitted (it gates on ``num_heads >= 12``), matching
        # the legacy umbrella where
        # ``_suppress_l15_lookup_lev_blockers_4_11`` ran independently
        # of the LEV head writer.
        for spec in _layer15_memory_lookup_lev_blockers_only_specs(
            proxy, max_head=int(num_heads),
        ):
            attn_op.append(
                spec,
                name=(
                    f"layer15_memory_lookup."
                    f"suppress_lev_blockers_4_11.h{spec.head_idx}"
                ),
                metadata={
                    "role": "current_store_suppress_lev",
                    "shape": "num_heads > 4",
                },
            )
    # Head 9: pop_d8_to_e0 declarative spec.
    # Legacy gate is ``num_heads > 9`` (skipped in default 16-layer
    # build at num_heads=9). The audit / declarations-only path
    # (num_heads=None) emits the spec so the IR carries head 9's
    # writes.
    if num_heads is None or int(num_heads) > 9:
        # HD is captured from the IR builder param. The pop_d8 spec
        # places its discriminator row at ``min(HD-1, 63)``.
        attn_op.append(
            _pop_d8_head_9_spec(
                head_idx=9,
                head_dim=int(HD) if HD is not None else 64,
                BD=proxy,
            ),
            name="layer15_memory_lookup.pop_d8_head_9",
            metadata={"role": "pop_d8_to_e0", "shape": "num_heads > 9"},
        )
    return ir


def make_layer15_memory_lookup_op() -> Operation:
    """L15 attention: memory-lookup heads for LI/LC.

    Phase 7.C.2 (Option B): the bake no longer calls the legacy
    ``_set_layer15_memory_lookup`` /
    :func:`_suppress_l15_lookup_during_current_store_generation`
    helpers directly. The CompilerIR built by
    :func:`_layer15_memory_lookup_ir` carries the same writes as a
    sequence of :class:`RuntimeAttentionFragment` bake-fns, and the
    layer-compiler dispatches the bake through that IR. The legacy
    helpers stay around as the fragment bodies (and as the single
    legacy entry point for :mod:`tests.test_l15_per_op` and
    :func:`make_l15_attention_resize_op`).

    DSL Wave W7 (this commit): the per-fragment
    ``runtime_predicate`` escape hatch is gone. The IR builder takes
    the shape variable ``num_heads`` directly and selects the right
    fragments at IR-build time via plain Python ``if``. The
    declarations-only audit path keeps calling
    ``compiler_ir_factory(dim_positions, head_dim)`` without a live
    ``attn``; in that case ``num_heads`` defaults to ``None`` and the
    IR carries every fragment (the symbolic execution ignores fragments
    anyway, so the audit-side IR shape is a soft superset).

    Phase 6 wave 2F (head-axis migration, still in force): the bake
    instantiates a per-bake :class:`AttentionHeadAllocator` pre-loaded
    with the full L15 head layout (see :data:`_L15_HEAD_LAYOUT`) so the
    structurally-stable head axis is auditable without grepping for
    ``head_idx=`` literals.
    """
    def bake(attn, dim_positions, S):
        # Per-bake attention-head allocator with the full L15 head
        # layout pinned. Stashing on ``attn._l15_head_allocator`` lets
        # downstream tooling inspect the L15 head axis without grepping
        # for ``head_idx=`` literals. The actual Q/K/V/O writes go
        # through the IR fragments below (cut from the legacy
        # imperative helpers in Phase 7.C.2).
        head_allocator = _allocate_layer15_attention_heads()
        attn._l15_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads

        # DSL Wave W7: pass the LIVE head count into the IR builder so the
        # LEV / head-9 branches are selected at IR-build time via plain Python
        # ``if``. The lowering pass below emits every fragment
        # unconditionally -- shape gating has already happened.
        #
        # Fragment selection must key on the GOLDEN (pre-widen) head count,
        # NOT the global ``attn.num_heads``: the head-dim-preserving auto-widen
        # for an over-width band (e.g. ``C4_AX_BYTE1_FULL_WIDTH`` grows
        # n_heads 10 -> 13) would otherwise flip the ``num_heads >= 12`` LEV
        # branch on at this CONSTRUCTION-time bake (the only writer of the
        # LEV saved_bp / return_addr heads -- the resize op does not re-bake
        # them), baking heads 4-11 that the golden 10-head build never writes
        # and diverging the L15 attention. ``alibi_base_heads`` is the
        # over-width-band-invariant head count (== ``attn.num_heads`` on every
        # un-widened build, so this is byte-identical there).
        _sel_num_heads = int(
            getattr(attn, "alibi_base_heads", attn.num_heads)
        )
        ir = _layer15_memory_lookup_ir(
            dim_positions, HD, num_heads=_sel_num_heads
        )
        ir.lower_attention(attn, HD, dim_positions=dim_positions, S=S)
        # Mark so the legacy umbrella entry point
        # ``vm_step._set_layer15_memory_lookup`` does not double-bake the
        # same fragments if a downstream caller invokes it after the
        # declarative path. Both code paths now lower the same IR.
        attn._l15_memory_lookup_ir_baked = True

        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            # Memory reads are last-write-wins. Strict neural traces can leave
            # residual address features on older MEM value bytes; once source
            # gating has limited candidates to stored value bytes, use ALiBi
            # only as a same-address tie-breaker. A large slope makes a newer
            # write to a different local slot beat the correct older address.
            attn.alibi_slopes[:4] = 0.05

    # Dim-ownership claims: L15 attn heads 0-3 (memory lookup).
    # Each head writes V slots 32..47 + 48..62 reading CLEAN_EMBED_LO/HI:
    #   W_v[h*HD + 32 + k, CLEAN_EMBED_LO + k]   for k=0..15
    #   W_v[h*HD + 48 + k, CLEAN_EMBED_HI + k]   for k=0..14
    # Slot 63 is repurposed by the non-pop STACK0 marker blocker below.
    # When num_heads >= 12 (LEV-aware build), heads 4-11 are also active;
    # we restrict claims to heads 0-3 which are the universal load heads
    # to keep the claim set stable across head-count configurations.
    _claims = set()
    for h in range(4):
        for k in range(16):
            _claims.add((15, "attn_W_v", f"{h}_{32 + k}", f"CLEAN_EMBED_LO+{k}"))
            if k < 15:
                _claims.add((15, "attn_W_v", f"{h}_{48 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer15_memory_lookup",
        # Phase 1 (memory cluster fix plan): shares L15 ``block.attn`` with
        # ``l15_attention_resize`` (a structural-resize op that runs after
        # the head bake). The resize replaces the attn module's W_q/W_k/
        # W_v/W_o wholesale to change ``num_heads``, then the memory
        # lookup re-bakes against the resized module. Legitimate
        # post-bake structural mutation, not a silent overwrite.
        slot_share=("attn",),
        # Phase 11 SCC residual (cycle #1, 5-op LEV next-step C-instruction
        # loop): TEMP -> TEMP.*.-1 SSA cross-step rename. l15
        # memory_lookup has phase=None, so the analyser sees l11_mul_partial's
        # and l14_temp_clear's TEMP writes (phases 11 / 14.1) as back-edges
        # into l15. The TEMP residual the lookup heads consume is the
        # previous step's value carried through the KV cache (the current
        # step's L11 / L14 writes are produced AFTER l15 under dynamic
        # scheduling; the static phase=None just surfaces as layer 0 in
        # the analyser). The SSA `.*.-1` form aliases back to the base
        # dim's numeric slot, so baked weight cells stay byte-identical
        # while the analyser drops the back-edges.
        reads={"MARK_AX", "OP_LI", "OP_LC", "OP_LI_RELAY", "OP_LC_RELAY",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_KEY", "MARK_MEM", "MEM_STORE",
               "MEM_ADDR_SRC",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "MARK_STACK0", "IS_BYTE",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1", "H2", "H3", "L2H0", "TEMP.*.-1",
               "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3", "CMP", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        declarative_bake_fn=bake,
        # ``compiler_ir_factory`` surfaces the L15 head structure as
        # data so the declarations-only path can audit it without
        # invoking ``bake_fn``. Forwarded through the same fragment IR
        # the bake uses, so the symbolic / IR view stays in sync.
        compiler_ir_factory=_layer15_memory_lookup_ir,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        # Phase 8.A.4: dropped ``layer_idx=15``. The dep graph already
        # forces a layer >= L14 via reads on ``AX_CARRY_LO`` /
        # ``AX_CARRY_HI`` / ``ADDR_KEY`` (produced by L14 attn/ffn ops);
        # ``requires["after"] = "layer14_mem_generation"`` backs the
        # constraint with a block-op-aware edge so the dynamic scheduler
        # still lands the op at layer 15.
        requires={"after": "layer14_mem_generation"},
        smoke_tests={
            "TestSmokeMemory::test_sc_lc_roundtrip",
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer15_store_stack0_sp_byte0_addr_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Copy current post-pop SP byte0 into ADDR_B0 at store STACK0 markers."""

    q = (
        AP(0, BD.MARK_STACK0, 300.0),
        AP(0, BD.HAS_SE, 300.0),
        AP(33, BD.MEM_STORE, 10000.0),
        AP(33, BD.CONST, -50000.0),
    )
    k = (
        AP(0, BD.MARK_SP, 100.0),
        AP(33, BD.CONST, 1.0),
    )
    v = [AP(0, BD.CONST, 1.0)]
    o = []
    for idx in range(16):
        v.append(AP(1 + idx, BD.OUTPUT_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.OUTPUT_HI + idx, 1.0))
        o.append(AO(BD.ADDR_B0_LO + idx, 0, -2.0))
        o.append(AO(BD.ADDR_B0_HI + idx, 0, -2.0))
        o.append(AO(BD.ADDR_B0_LO + idx, 1 + idx, 3.0))
        o.append(AO(BD.ADDR_B0_HI + idx, 17 + idx, 3.0))
    return DeclarativeAttentionHeadSpec(
        # Pull the head index from the shared L15 layout rather than
        # baking in a ``head_idx=12`` literal here. Both the bake and IR
        # paths consult the same source of truth, so renumbering the
        # layout in one place stays consistent across every consumer.
        head_idx=_l15_head_idx("layer15_store_stack0_sp_byte0_addr"),
        q=q,
        k=k,
        v=tuple(v),
        o=tuple(o),
    )


def _layer15_store_stack0_sp_byte0_addr_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer15_store_stack0_sp_byte0_addr_spec(proxy)
    )
    return ir


def make_layer15_store_stack0_sp_byte0_addr_op() -> Operation:
    """L15 attention: expose post-pop SP byte0 for store-top discrimination."""

    def bake(target, dim_positions, S):
        del S
        attn = getattr(target, "attn", target)
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer15_store_stack0_sp_byte0_addr_spec(proxy),
            HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[12] = 1.0

    return Operation(
        name="layer15_store_stack0_sp_byte0_addr",
        # Phase 8.A G7: OUTPUT_HI_THIS_STEP read renamed to
        # OUTPUT_HI_PREV_STEP. Head 12 attends back to the post-pop
        # STACK0 / SP-marker token whose cached OUTPUT_HI residual
        # is the previous step's value -- not a same-step data flow
        # from layer16_lev_routing or tail_bit32_result_correction
        # (which both fire AFTER L15 in the same step). The alias
        # shares numeric position 190 with OUTPUT_HI so bakes stay
        # byte-identical. Breaks 2 cross-step back-edges.
        reads={
            "MARK_STACK0", "MARK_SP", "HAS_SE", "MEM_STORE",
            "OUTPUT_LO", "OUTPUT_HI.*.-1", "CONST",
        },
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=15`` pin. The block op binds
        # to whichever layer the compiler placed ``layer15_memory_lookup``
        # (the L15 attn op that owns the layer slot). ``requires["after"]``
        # encodes the same constraint at the scheduler level.
        target_op_name="layer15_memory_lookup",
        declarative_bake_fn=bake,
        declarative_authority="declarative",
        compiler_ir_factory=_layer15_store_stack0_sp_byte0_addr_ir,
        migrated=True,
        alibi_slopes={12: 1.0},
        requires={"after": "layer15_memory_lookup"},
        smoke_tests={
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#stack-store",
    )


def _layer15_si_mem_addr0_from_stack0_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Override SI/SC MEM addr byte 0 from pre-store STACK0 byte 0.

    L14's generic MEM-address head reads ``CLEAN_EMBED + OUTPUT``. That is
    correct for PSH, where the SP marker carries the freshly computed address
    in OUTPUT, but for SI/SC the source is a STACK0 byte token whose OUTPUT is
    the following stack byte. This head is SI/SC-only and reads only
    CLEAN_EMBED from the pre-store STACK0 byte 0.
    """

    q = (
        AP(0, BD.MARK_MEM, 100.0),
        AP(0, BD.MEM_STORE, 20.0),
        AP(0, BD.MEM_ADDR_SRC, 50.0),
        AP(0, BD.CONST, -140.0),
        AP(33, BD.MARK_MEM, 40000.0),
        AP(33, BD.MEM_STORE, 5000.0),
        AP(33, BD.MEM_ADDR_SRC, 10000.0),
        AP(33, BD.CONST, -55000.0),
        AP(37, BD.IS_BYTE, 50000.0),
    )
    k = (
        AP(0, BD.STACK0_BYTE0, 100.0),
        AP(0, BD.MEM_STORE, -400.0),
        AP(33, BD.CONST, 5.0),
        AP(37, BD.CONST, -20.0),
    )
    v = [AP(0, BD.CONST, 1.0)]
    o = []
    for idx in range(16):
        v.append(AP(1 + idx, BD.CLEAN_EMBED_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.CLEAN_EMBED_HI + idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + idx, 0, -10.0))
        o.append(AO(BD.OUTPUT_HI + idx, 0, -10.0))
        o.append(AO(BD.OUTPUT_LO + idx, 1 + idx, 20.0))
        o.append(AO(BD.OUTPUT_HI + idx, 17 + idx, 20.0))
    return DeclarativeAttentionHeadSpec(
        # Pull the head index from the shared L15 layout rather than
        # baking in a ``head_idx=13`` literal here. Both the bake and IR
        # paths consult the same source of truth, so renumbering the
        # layout in one place stays consistent across every consumer.
        head_idx=_l15_head_idx("layer15_si_mem_addr0_from_stack0"),
        q=q,
        k=k,
        v=tuple(v),
        o=tuple(o),
    )


def _layer15_si_mem_addr0_from_stack0_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    # NOTE(L15-si-mem-addr0-scope-honest): the head's INTENDED firing scope is
    # the SI/SC MEM-address token (mark == MEM AND mem_store AND NOT psh)
    # at the q position, with the key drawn from the pre-store STACK0 byte 0.
    # The K projection's positive dims are CONST and STACK0_BYTE0, so the V1
    # attention verifier's K-derived effective_attention_scope is
    # ("CONST", "STACK0_BYTE0") -- this is over-approximate (it ignores the Q
    # MEM_STORE/MEM_ADDR_SRC opcode gating) but is the best the V1 scope
    # heuristic can express without modelling Q-side conditioning.
    #
    # Declaring scope/dominates_at here is purely informational under V1:
    # ``verify_attention_head`` does NOT currently filter FFN-side
    # cross-modality competitors by scope overlap (see attention_verifier.py
    # V2 wishlist), so the declared scope cannot suppress the 32 CSV
    # violations head 13 surfaces against tail_mem_store_addr1_ff_*,
    # tail_bp_byte2_preserve_01, tail_sp_pop_marker_output_d8_to_e0, etc.
    # Those tail rules are gated to entirely different opcode/marker
    # combinations (mark == SP for the pop-marker family, MEM-store address
    # finalization at byte_index == 1 for the addr1_ff family), so a
    # scope-aware V2 verifier would rule them out as bookkeeping
    # competitors only.
    #
    # We do NOT bump the head's magnitude (slot 0 cleanup -10, slot 1+idx
    # copy +20 -> magnitude 30): bumping to dominate the 112-tier addr1_ff
    # competitors would require ~4x larger O/V weights, which scales the
    # actual residual delta proportionally and would over-write the SI/SC
    # MEM addr0 byte by the same factor (breaking the L13 mem-addr gather
    # downstream calibration).  The attention_strength_violation against
    # layer15_alu_high_byte_relay (head 8) is structurally zero-sum: head 8
    # also writes OUTPUT_LO/HI with magnitude 20 over a non-overlapping
    # opcode gate (MARK_AX/OP_MUL/OP_SHL vs MARK_MEM/MEM_STORE here), and
    # any magnitude swap between the two heads just relocates the
    # violation between them under the V1 sign-blind, scope-blind algebra.
    ir.layer(0).attention.append(
        _layer15_si_mem_addr0_from_stack0_spec(proxy),
        metadata={
            "scope": "mark == MEM AND mem_store",
            "dominates_at": {
                "OUTPUT_LO": "mark == MEM AND mem_store",
                "OUTPUT_HI_THIS_STEP": "mark == MEM AND mem_store",
            },
        },
    )
    return ir


def make_layer15_si_mem_addr0_from_stack0_op() -> Operation:
    """L15 attention: clean SI/SC MEM addr0 from STACK0 byte0."""

    def bake(target, dim_positions, S):
        del S
        attn = getattr(target, "attn", target)
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer15_si_mem_addr0_from_stack0_spec(proxy),
            HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[13] = 1.0

    return Operation(
        name="layer15_si_mem_addr0_from_stack0",
        reads={
            "MARK_MEM", "MEM_STORE", "MEM_ADDR_SRC", "STACK0_BYTE0",
            "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=15`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer15_memory_lookup`` (the L15 attn op).
        target_op_name="layer15_memory_lookup",
        declarative_bake_fn=bake,
        declarative_authority="declarative",
        compiler_ir_factory=_layer15_si_mem_addr0_from_stack0_ir,
        migrated=True,
        alibi_slopes={13: 1.0},
        # Phase 7.A.2 backfill: this op OVERRIDES the L14 generic MEM-address
        # head's OUTPUT_LO/OUTPUT_HI_THIS_STEP for SI/SC (see module docstring
        # at the top of the spec helper). The override only makes sense after
        # ``layer14_mem_generation`` has already written the generic result,
        # so it must run strictly later. The STACK0_BYTE0 source path
        # ultimately traces back to ``layer1_ffn`` (already declared) which
        # populates the STACK0 byte slot consumed by the K projection here.
        requires={"after": ["layer1_ffn", "layer14_mem_generation"]},
        smoke_tests={
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def _suppress_l15_lookup_heads_0_3(attn, BD, HD) -> None:
    """Always-on portion of :func:`_suppress_l15_lookup_during_current_store_generation`.

    Phase 7.C.2 factored split: heads 0-3 are universal LI/LC + STACK0
    load heads, so their suppress writes run on every L15 attention
    bake. Carried as a :class:`RuntimeAttentionFragment` in
    ``layer15_memory_lookup``'s CompilerIR.
    """
    pc_i = 0
    ax_i = 1
    mem_i = 4
    sp_i = 2
    bp_i = 3
    for head in range(4):
        base = head * HD
        # Local stack slots commonly differ only in byte 0 (BP-8, BP-16,
        # BP-24, ...). The generic 24-bit binary address rows are
        # intentionally modest, but after shallow ALiBi recency the byte-0
        # mismatch must dominate newer writes to adjacent locals. Strengthen
        # both byte-0 nibbles: three-local frames can share the same low
        # nibble (e.g. BP-8 and BP-24) and only differ in the high nibble.
        local_slot_scale = 100.0
        for nibble_offset, nibble_base in ((0, BD.ADDR_B0_LO), (4, BD.ADDR_B0_HI)):
            for bit in range(4):
                row = base + 4 + nibble_offset + bit
                for k in range(16):
                    bit_val = 2 * ((k >> bit) & 1) - 1
                    attn.W_q.data[row, nibble_base + k] = local_slot_scale * bit_val
                    attn.W_k.data[row, nibble_base + k] = local_slot_scale * bit_val
        # The bit rows above can be cancelled by negative adjacent-nibble
        # residue from address arithmetic. Add direct one-hot rows so an exact
        # low-byte slot match remains authoritative.
        local_slot_onehot_scale = 100.0
        for k in range(16):
            row = base + 43 + k
            attn.W_q.data[row, BD.CONST] = -local_slot_onehot_scale
            attn.W_q.data[row, BD.ADDR_B0_LO + k] = local_slot_onehot_scale
            attn.W_q.data[row, BD.OP_LI_RELAY] = local_slot_onehot_scale
            if head == 0:
                attn.W_q.data[row, BD.OP_LC_RELAY] = local_slot_onehot_scale
            attn.W_k.data[row, BD.ADDR_B0_LO + k] = local_slot_onehot_scale

        if head == 0:
            # L6/L8 relays can deliver the pop-group flag at ~4.0 by the time
            # L15 runs.  A small non-target bias is not enough once local
            # address residue reaches the STACK0 marker, so make row 0 a
            # strong default blocker and explicitly restore real LI/LC and
            # pop-stack queries.
            lookup_bias = 200000.0
            attn.W_q.data[base + 0, BD.CONST] = -lookup_bias
            attn.W_q.data[base + 0, BD.OP_LI_RELAY] = lookup_bias
            attn.W_q.data[base + 0, BD.OP_LC_RELAY] = lookup_bias
            attn.W_q.data[base + 0, BD.OP_LI] = lookup_bias
            attn.W_q.data[base + 0, BD.OP_LC] = lookup_bias
            attn.W_q.data[base + 0, BD.CMP + 3] = lookup_bias / 4.0
            attn.W_q.data[base + 1, BD.CMP + 3] = 12.5
            # JSR/ENT synthesize STACK0 values through the function-call path,
            # LEA computes AX from BP+imm through the arithmetic path, and IMM
            # preserves STACK0 while loading AX from the immediate. None of
            # these are memory loads; residual address/store metadata can
            # otherwise make the load-only L15 head read an old zero/SP value
            # over the value already produced upstream.
            non_load_suppression = -1000000.0
            attn.W_q.data[base + 0, BD.OP_JSR] = non_load_suppression
            attn.W_q.data[base + 0, BD.OP_ENT] = non_load_suppression
            attn.W_q.data[base + 0, BD.OP_LEA] = non_load_suppression
            attn.W_q.data[base + 0, BD.OP_IMM] = non_load_suppression
            # Preserve lookups at the exact 0xffe8 stack-top slot. This keeps
            # ordinary IMM steps in function-call setup from going through the
            # non-load zero sink while still excluding the adjacent 0xfff8
            # return-address slot.
            attn.W_q.data[base + 0, BD.MARK_STACK0] = 75000.0
            attn.W_q.data[base + 0, BD.HAS_SE] = 75000.0
            attn.W_q.data[base + 0, BD.ADDR_B0_LO + 8] = 75000.0
            attn.W_q.data[base + 0, BD.ADDR_B0_HI + 14] = 75000.0
            attn.W_q.data[base + 0, BD.ADDR_B0_HI + 15] = -100000.0
            attn.W_q.data[base + 0, BD.IS_BYTE] = -2000.0
            attn.W_q.data[base + 1, BD.IS_BYTE] = -50.0
            attn.W_q.data[base + 1, BD.MARK_STACK0] = 50.0
            attn.W_q.data[base + 1, BD.HAS_SE] = 50.0
            attn.W_q.data[base + 1, BD.ADDR_B0_LO + 8] = 50.0
            attn.W_q.data[base + 1, BD.ADDR_B0_HI + 14] = 50.0
            attn.W_q.data[base + 1, BD.ADDR_B0_HI + 15] = -150.0
            attn.W_q.data[base + 28, BD.IS_BYTE] = -500.0
            attn.W_q.data[base + 28, BD.CONST] = -20000.0
            attn.W_q.data[base + 28, BD.MARK_AX] = 20000.0
            attn.W_q.data[base + 28, BD.MARK_STACK0] = 20000.0
            # Store/pop STACK0 markers are usually load queries too: after
            # SI/SC pops the address, the visible stack top is memory[post-pop
            # SP].  The exception is a top-store, where the popped address is
            # exactly the post-pop SP.  In that case the current store value is
            # authoritative, but the current MEM row does not exist yet at the
            # STACK0 marker, so an unblocked L15 lookup would read the old
            # historical value.  Block only those equality signatures and keep
            # non-top stores available for historical memory lookup.
            for row, low, high in (
                (42, 0, 14),   # e0
            ):
                attn.W_q.data[base + row, BD.CONST] = -60000.0
                attn.W_q.data[base + row, BD.MARK_STACK0] = 10000.0
                attn.W_q.data[base + row, BD.HAS_SE] = 10000.0
                attn.W_q.data[base + row, BD.MEM_STORE] = 10000.0
                attn.W_q.data[base + row, BD.EMBED_LO + low] = 10000.0
                attn.W_q.data[base + row, BD.EMBED_HI + high] = 10000.0
                attn.W_q.data[base + row, BD.ADDR_B0_LO + low] = 10000.0
                attn.W_q.data[base + row, BD.ADDR_B0_HI + high] = 10000.0
                attn.W_q.data[base + row, BD.OP_LI_RELAY] = 50000.0
                attn.W_q.data[base + row, BD.OP_LC_RELAY] = 50000.0
                # A partial miss on this signature is negative; keep that
                # negative query from becoming positive stale-key evidence.
                attn.W_k.data[base + row, BD.CONST] = 20.0

            # Rows 59-61 are value lanes for OUTPUT_HI[11:13], not address
            # discriminators.  If they retain any Q/K miss terms, the
            # negative miss can multiply stale negative key residue into a
            # large positive score and undo the top-store blocker above.
            for row in (59, 60, 61):
                if row < HD:
                    attn.W_q.data[base + row, :] = 0.0
                    attn.W_k.data[base + row, :] = 0.0

            # Same top-store equality as the e0 row above, but for the common
            # one-local slot 0xffe8. At the SI/SC STACK0 marker the address is
            # present in ADDR_B0, while EMBED still carries marker/value
            # residue; block the historical lookup so L14's current store
            # value remains authoritative.
            top_store_e8_row = 60
            if top_store_e8_row < HD:
                row = base + top_store_e8_row
                attn.W_q.data[row, BD.CONST] = -30000.0
                attn.W_q.data[row, BD.MARK_STACK0] = 10000.0
                attn.W_q.data[row, BD.MARK_SP] = -100000.0
                attn.W_q.data[row, BD.HAS_SE] = 10000.0
                attn.W_q.data[row, BD.MEM_STORE] = 150000.0
                attn.W_q.data[row, BD.ADDR_B0_LO + 8] = 10000.0
                attn.W_q.data[row, BD.ADDR_B0_HI + 14] = 10000.0
                attn.W_q.data[row, BD.ADDR_B0_HI + 15] = -20000.0
                attn.W_k.data[row, BD.CONST] = -20.0

            # When preserving STACK0 at the one-argument call slot (0xffe8),
            # the adjacent return-address slot (0xfff8) shares byte-0 low
            # nibble 8 and can win by recency. Add an exact high-nibble source
            # discriminator for this marker lookup so the e8 store is selected.
            preserve_e8_row = 59
            preserve_e8_s = 5000.0
            attn.W_q.data[base + preserve_e8_row, BD.CONST] = -3.5 * preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.MARK_STACK0] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.HAS_SE] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.ADDR_B0_LO + 8] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.ADDR_B0_HI + 14] = preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.IS_BYTE] = -4.0 * preserve_e8_s
            attn.W_q.data[base + preserve_e8_row, BD.MEM_STORE] = -5.0 * preserve_e8_s
            attn.W_k.data[base + preserve_e8_row, BD.ADDR_B0_LO + 8] = preserve_e8_s
            attn.W_k.data[base + preserve_e8_row, BD.ADDR_B0_HI + 14] = preserve_e8_s
            attn.W_k.data[base + preserve_e8_row, BD.STACK0_BYTE0] = 5.0 * preserve_e8_s

            # AX LI/LC marker loads at 0xffe8 also need an exact e8 value-row
            # discriminator, but row 59's K side intentionally likes STACK0.
            # Use the otherwise-neutral row 61 to boost only MEM value byte 0
            # rows whose decoded source address is 0xffe8. Keep the key
            # non-negative; the STACK0 query term neutralizes this row during
            # STACK0 preserves, while broad H1 gates can turn unrelated byte
            # queries into false-positive memory loads.
            # Scaled-down (was 100000.0) by the same 10000x factor used on
            # row 36: at s=100000 the Q-side miss combination (CONST -3.5s,
            # IS_BYTE -4s, MEM_STORE -4s) produced Q[61]~=-1.5e5 at every
            # non-0xffe8 LI step, and W_k[MEM_VAL_B1]=1e5 then drove K@(MEM
            # val byte 1) to -1.5G, killing every normal LI/LC load. The
            # legitimate 0xffe8 path still wins via the relative ordering
            # of its Q + K terms; the absolute scale of this lane is not
            # needed once it no longer drowns the rest of the head.
            ax_li_e8_row = 61
            ax_li_e8_s = 10.0
            attn.W_q.data[base + ax_li_e8_row, BD.CONST] = -3.5 * ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.MARK_AX] = ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.OP_LI_RELAY] = ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.OP_LC_RELAY] = ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.MARK_STACK0] = 2.5 * ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.ADDR_B0_LO + 8] = ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.ADDR_B0_HI + 14] = ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.IS_BYTE] = -4.0 * ax_li_e8_s
            attn.W_q.data[base + ax_li_e8_row, BD.MEM_STORE] = -4.0 * ax_li_e8_s
            attn.W_k.data[base + ax_li_e8_row, BD.MEM_VAL_B1] = ax_li_e8_s
            attn.W_k.data[base + ax_li_e8_row, BD.ADDR_B0_LO + 8] = ax_li_e8_s
            attn.W_k.data[base + ax_li_e8_row, BD.ADDR_B0_HI + 14] = ax_li_e8_s

            # Immediately after ENT, the correct STACK0 preserve source is the
            # zero row produced by the ENT setup step.  The general OP_ENT
            # source blocker above is needed later to avoid stale frame setup
            # rows, so add a query-side early-ENT discriminator that only
            # fires while the STACK0 marker itself still carries OP_ENT
            # residue.  Keep K non-negative so non-target negative queries
            # cannot become positive evidence.
            early_ent_stack0_row = 58
            early_ent_stack0_q = 100000.0
            # OPCODE-BROADCAST HARDENING (2026-06-11, var_simple_12 / id 262):
            # OP_ENT does NOT stay one-hot at its own marker -- it BROADCASTS
            # in-step onto every row at magnitude ~12-17 (audit
            # docs/OPCODE_BROADCAST_BLOCKER_AUDIT_2026_06_11.md). At Q weight
            # 2e8 * ~12.6 = 2.5e9 it overran even the -2e9 single-marker
            # NOT-blockers, so on the step-2 LEA PC byte0 prediction row
            # (MARK_PC=1, OP_ENT=12.6, MARK_STACK0=0) this discriminator fired,
            # forced load-head 0 to attend an operand position and copy its
            # CLEAN_EMBED into OUTPUT_LO+0 at scale 40 -- flooding the PC byte
            # to 0x00 and desyncing the whole program (probe_var_full_chain.py
            # 262 step-2 LEA PC byte0). FIX: make MARK_STACK0 a HARD requirement
            # by offsetting CONST by -stack0_gate and MARK_STACK0 by
            # +stack0_gate. Net delta on the legitimate firing row
            # (MARK_STACK0=1) is ZERO -- byte-identical there -- while on any
            # non-STACK0 row the -1e10 bias buries OP_ENT's largest broadcast
            # (2e8 * ~17.5 = 3.5e9). OP_ENT stays the in-step confirming term;
            # the K-side OP_ENT match is unchanged.
            # NOTE: the declarative mirror
            # ``_layer15_memory_lookup_heads_0_3_specs_with_overrides`` slot 58
            # carries the identical change; this imperative writer is the one
            # that actually lands because make_l15_attention_resize_op re-runs
            # it after the declarative bake.
            early_ent_stack0_gate = 10000000000.0  # 1e10 >> 2e8 * ENT_bcast_max
            attn.W_q.data[base + early_ent_stack0_row, :] = 0.0
            attn.W_k.data[base + early_ent_stack0_row, :] = 0.0
            attn.W_q.data[base + early_ent_stack0_row, BD.OP_ENT] = 200000000.0
            attn.W_q.data[base + early_ent_stack0_row, BD.MARK_STACK0] = (
                early_ent_stack0_q + early_ent_stack0_gate
            )
            attn.W_q.data[base + early_ent_stack0_row, BD.CONST] = (
                -early_ent_stack0_q - early_ent_stack0_gate
            )
            attn.W_q.data[base + early_ent_stack0_row, BD.IS_BYTE] = (
                -2000000000.0
            )
            for marker_dim in (
                BD.MARK_AX,
                BD.MARK_PC,
                BD.MARK_SP,
                BD.MARK_BP,
                BD.MARK_MEM,
            ):
                attn.W_q.data[base + early_ent_stack0_row, marker_dim] = (
                    -2000000000.0
                )
            attn.W_k.data[base + early_ent_stack0_row, BD.OP_ENT] = 10000.0

            # Pop-group STACK0 marker lookups need the post-pop SP address.
            # The marker can still carry the pre-pop byte-0 address (for
            # example f0 while the value to reveal lives at f8), and the
            # same L15 attention block cannot consume the head-12 correction
            # it emits later in the block. Add a score-only row that shifts
            # low-nibble 0 queries toward historical low-nibble 8 MEM value
            # keys; the ordinary address rows still discriminate the high
            # address bytes.
            pop_low8_row = 34
            attn.W_q.data[base + pop_low8_row, BD.CONST] = -4000.0
            attn.W_q.data[base + pop_low8_row, BD.MARK_STACK0] = 2000.0
            attn.W_q.data[base + pop_low8_row, BD.HAS_SE] = 1000.0
            attn.W_q.data[base + pop_low8_row, BD.CMP + 3] = 1000.0
            attn.W_q.data[base + pop_low8_row, BD.ADDR_B0_LO + 0] = 1000.0
            attn.W_q.data[base + pop_low8_row, BD.ADDR_B0_LO + 8] = 1000.0
            attn.W_q.data[base + pop_low8_row, BD.IS_BYTE] = -10000.0
            attn.W_q.data[base + pop_low8_row, BD.MARK_SP] = -10000.0
            attn.W_q.data[base + pop_low8_row, BD.MEM_STORE] = -20000.0
            attn.W_k.data[base + pop_low8_row, BD.ADDR_B0_LO + 8] = 1000.0
        else:
            # The legacy byte-select rows for heads 1-3 target the source
            # byte behind the autoregressive query. At query byte N, logits
            # predict byte N+1, so the source must also be byte N+1.
            byte_q_flags = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]
            attn.W_q.data[base + 0, BD.MARK_STACK0] = -100000.0
            attn.W_q.data[base + 0, BD.MARK_SP] = -100000.0
            attn.W_q.data[base + 28, BD.CONST] = -20000.0
            attn.W_q.data[base + 28, byte_q_flags[head]] = 20000.0
            for dim in (BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3,
                        BD.H2 + mem_i, BD.H3 + mem_i):
                attn.W_k.data[base + 3, dim] = 0.0
            if head == 1:
                attn.W_k.data[base + 3, BD.MEM_VAL_B2] = 60.0
            elif head == 2:
                attn.W_k.data[base + 3, BD.MEM_VAL_B3] = 60.0
            elif head == 3:
                attn.W_k.data[base + 3, BD.H3 + mem_i] = 60.0
                attn.W_k.data[base + 3, BD.H2 + mem_i] = -60.0

        # Strict neural traces can leave large ADDR_KEY-like residue on code
        # and previous-step byte tokens. The lookup address rows are supposed
        # to choose among stored MEM value bytes only, so add a source-side
        # row that suppresses non-store/non-value keys while preserving the
        # same per-byte source positions selected by row 3.
        source_gate = 37
        source_gate_s = 3000.0
        attn.W_q.data[base + source_gate, BD.CONST] = 0.0
        if head == 0:
            attn.W_q.data[base + source_gate, BD.MARK_STACK0] = source_gate_s
        else:
            attn.W_q.data[base + source_gate, byte_q_flags[head]] = source_gate_s
        source_key_s = 20.0
        attn.W_k.data[base + source_gate, :] = 0.0
        source_key_s = 10.0
        attn.W_k.data[base + source_gate, BD.CONST] = -source_key_s
        attn.W_k.data[base + source_gate, BD.MEM_STORE] = 0.5 * source_key_s
        if head == 0:
            attn.W_k.data[base + source_gate, BD.L2H0 + mem_i] = source_key_s
            attn.W_k.data[base + source_gate, BD.H1 + mem_i] = -70.0
        else:
            attn.W_k.data[base + source_gate, BD.H1 + mem_i] = -70.0
            if head == 1:
                attn.W_k.data[base + source_gate, BD.MEM_VAL_B2] = source_key_s
            elif head == 2:
                attn.W_k.data[base + source_gate, BD.MEM_VAL_B3] = source_key_s
            elif head == 3:
                attn.W_k.data[base + source_gate, BD.H3 + mem_i] = source_key_s
                attn.W_k.data[base + source_gate, BD.H2 + mem_i] = -source_key_s

        # L15 lookup must source historical MEM value rows (or the prior
        # STACK0 row), not SP/BP register bytes.  Frame register bytes can
        # carry address-like residue and beat the intended MEM value source.
        for marker_i in (sp_i, bp_i):
            for dim in (
                BD.H1 + marker_i,
                BD.H2 + marker_i,
                BD.H3 + marker_i,
                BD.L2H0 + marker_i,
            ):
                attn.W_k.data[base + source_gate, dim] = -80.0

        # Load-only reinforcement: the row above also stabilizes pop/STACK0
        # lookup during store ops, so keep it conservative. LI/LC need a
        # stronger source preference to beat ADDR_KEY residue on non-MEM
        # register bytes. Keep the boost head-specific: head 0 owns the AX
        # marker byte, while heads 1-3 own AX byte positions 0-2 which predict
        # loaded bytes 1-3. If every head receives the same load-only boost at
        # every byte position, zero-valued higher-byte heads drown the correct
        # nonzero byte.
        load_source_gate = 39
        load_source_gate_s = 5000.0
        load_source_key_s = 30.0
        attn.W_q.data[base + load_source_gate, :] = 0.0
        attn.W_k.data[base + load_source_gate, :] = 0.0
        if head == 0:
            attn.W_q.data[base + load_source_gate, BD.OP_LI_RELAY] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, BD.OP_LC_RELAY] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, BD.MARK_AX] = load_source_gate_s
            # Pop-group STACK0 marker loads are real memory reads too.  Keep
            # this LI/LC reinforcement row neutral-to-positive there instead
            # of letting its negative bias force the softmax1 zero sink.
            attn.W_q.data[base + load_source_gate, BD.CMP + 3] = 2000.0
            attn.W_q.data[base + load_source_gate, BD.CONST] = -1.5 * load_source_gate_s
            attn.W_k.data[base + load_source_gate, BD.MEM_VAL_B1] = 2.0 * load_source_key_s
            attn.W_k.data[base + load_source_gate, BD.MEM_ADDR_SRC] = 40.0
        else:
            attn.W_q.data[base + load_source_gate, BD.OP_LI_RELAY] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, byte_q_flags[head]] = load_source_gate_s
            attn.W_q.data[base + load_source_gate, BD.CONST] = -1.5 * load_source_gate_s
            if head == 1:
                attn.W_k.data[base + load_source_gate, BD.MEM_VAL_B2] = 2.0 * load_source_key_s
            elif head == 2:
                attn.W_k.data[base + load_source_gate, BD.MEM_VAL_B3] = 2.0 * load_source_key_s
            elif head == 3:
                attn.W_k.data[base + load_source_gate, BD.H3 + mem_i] = 2.0 * load_source_key_s
                attn.W_k.data[base + load_source_gate, BD.H2 + mem_i] = -2.0 * load_source_key_s
            attn.W_k.data[base + load_source_gate, BD.MEM_ADDR_SRC] = 40.0

        # AX-marker byte-0 loads are especially sensitive to stale store-like
        # residue on non-value tokens. Add a marker-only source discriminator
        # that prefers SI/SC value byte 0, mildly allows PSH byte 0, and pushes
        # non-value store positions below the softmax1 anchor.
        if head == 0:
            marker_value_gate = 40
            marker_value_gate_s = 1000.0
            attn.W_q.data[base + marker_value_gate, :] = 0.0
            attn.W_k.data[base + marker_value_gate, :] = 0.0
            attn.W_q.data[base + marker_value_gate, BD.OP_LI] = marker_value_gate_s
            attn.W_q.data[base + marker_value_gate, BD.OP_LC] = marker_value_gate_s
            attn.W_k.data[base + marker_value_gate, BD.MEM_VAL_B1] = 80.0
            attn.W_k.data[base + marker_value_gate, BD.MEM_ADDR_SRC] = 40.0
            attn.W_k.data[base + marker_value_gate, BD.CONST] = -60.0

        # Loaded bytes must beat the default zero-emission path that has
        # accumulated in OUTPUT by this late layer. Keep the attention scores
        # unchanged, but make the selected memory byte authoritative in the
        # output projection.
        value_scale = 40.0
        for k in range(16):
            attn.W_o.data[BD.OUTPUT_LO + k, base + 32 + k] = value_scale
            attn.W_o.data[BD.OUTPUT_HI + k, base + 48 + k] = value_scale

        # L15 lookup is load-only. ADD/SUB rows can still carry stale ADDR_KEY
        # residue, and L10 marks arithmetic byte propagation through TEMP+8/9.
        # Force those queries to the softmax1 zero sink so L15 cannot overwrite
        # the arithmetic output that L10 just produced.
        addsub_blocker = 41
        attn.W_q.data[base + addsub_blocker, BD.TEMP + 8] = 10000.0
        attn.W_q.data[base + addsub_blocker, BD.TEMP + 9] = 10000.0
        attn.W_k.data[base + addsub_blocker, BD.CONST] = -20.0

        # SP value bytes and BP markers are not L15 lookup targets.  Head 0's
        # pop-group/preserve gates can otherwise make frame rows attend to
        # historical stack values when setup residue is large, overwriting
        # L3/L10's register bytes.
        sp_byte_blocker = 62
        attn.W_q.data[base + sp_byte_blocker, BD.H1 + 2] = 100000.0
        attn.W_q.data[base + sp_byte_blocker, BD.MARK_BP] = 100000.0
        attn.W_q.data[base + sp_byte_blocker, BD.TEMP + 10] = 100000.0
        attn.W_q.data[base + sp_byte_blocker, BD.TEMP + 24] = 100000.0
        attn.W_q.data[base + sp_byte_blocker, BD.IS_BYTE] = 0.0
        attn.W_k.data[base + sp_byte_blocker, BD.CONST] = -300000.0
        if head == 0:
            # Head 0 owns marker-byte lookups. At byte continuations, negative
            # exact-address rows can multiply negative keys and copy the
            # previous byte forward (for example JSR STACK0 byte0 -> byte1).
            attn.W_q.data[base + sp_byte_blocker, BD.IS_BYTE] = 500000.0
            # ENT is a frame-store opcode, not a load. Keep L15 head 0 from
            # reading the previous JSR return-address row into ENT's AX marker.
            attn.W_q.data[base + sp_byte_blocker, BD.OP_ENT] = 500000.0

        # PC value bytes are produced by the control-flow path, not by memory
        # lookup. Branch targets can carry address-like residue that otherwise
        # makes L15 copy PC byte0 into the upper PC bytes.
        pc_byte_blocker = 35
        attn.W_q.data[base + pc_byte_blocker, BD.H1 + pc_i] = 100000.0
        attn.W_q.data[base + pc_byte_blocker, BD.MARK_PC] = 100000000.0
        attn.W_q.data[base + pc_byte_blocker, BD.IS_BYTE] = 0.0
        attn.W_k.data[base + pc_byte_blocker, BD.CONST] = -100000.0
        if head == 0:
            # Binary-pop steps whose pre-pop SP is f8 reveal the empty stack
            # slot at 0x10000.  The STACK0 marker still carries the pre-pop
            # f8 address at L15, so do not let it read the just-popped value.
            attn.W_q.data[base + pc_byte_blocker, BD.MARK_STACK0] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.HAS_SE] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.CMP + 3] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.ADDR_B0_LO + 8] = 10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.ADDR_B0_HI + 15] = 10000.0
            # LI/LC steps also carry HAS_SE=1 at the AX marker, so the
            # +10000 HAS_SE contribution above turns row 35 into a -1G
            # constant penalty at every K position (W_k[CONST]=-100000),
            # collapsing softmax onto the zero sink and aliasing LI as a
            # return-the-address operation. Cancel HAS_SE at LI/LC by
            # subtracting the same magnitude from the relay dims so the
            # binary-pop guard only fires for pop steps (which have no
            # LI/LC relay).
            attn.W_q.data[base + pc_byte_blocker, BD.OP_LI_RELAY] = -10000.0
            attn.W_q.data[base + pc_byte_blocker, BD.OP_LC_RELAY] = -10000.0

            # Non-pop, non-store STACK0 markers preserve the visible stack top.
            # Give them a positive source row for the latest prior STACK0 byte0
            # token so preservation does not fall through to L16's ALU address
            # fallback.
            stack0_preserve_row = 36
            # Scaled-down (was 10000.0) to keep slot 36 within +-300, the L15
            # binary-address match scale.  At s=10000 this row scored 2.5e8
            # for any K with BYTE_INDEX_0=1 (e.g. STACK0 byte 0), aliasing
            # non-store rows as memory targets and destroying SI/LI roundtrip.
            # See tests/test_l15_memory_lookup_isolated.py for the gate.
            stack0_preserve_s = 1.0
            attn.W_q.data[base + stack0_preserve_row, BD.CONST] = (
                -1.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.MARK_STACK0] = (
                3.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.HAS_SE] = (
                1.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.CMP + 3] = (
                -5.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.MEM_STORE] = (
                -10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.IS_BYTE] = (
                -10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.MARK_AX] = (
                -10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.H1 + ax_i] = (
                10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.OP_LI_RELAY] = (
                10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.OP_LC_RELAY] = (
                10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.MARK_PC] = (
                -10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.MARK_SP] = (
                -10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.MARK_BP] = (
                -10.0 * stack0_preserve_s
            )
            attn.W_q.data[base + stack0_preserve_row, BD.MARK_MEM] = (
                -10.0 * stack0_preserve_s
            )
            attn.W_k.data[base + stack0_preserve_row, BD.CONST] = (
                -2.0 * stack0_preserve_s
            )
            attn.W_k.data[base + stack0_preserve_row, BD.H1 + 10] = (
                2.0 * stack0_preserve_s
            )
            attn.W_k.data[base + stack0_preserve_row, BD.BYTE_INDEX_0] = (
                2.0 * stack0_preserve_s
            )
            # Early frame-setup STACK0 rows can retain OP_ENT residue and
            # otherwise tie the latest visible stack top.  They are not valid
            # preservation sources for later non-pop markers.
            attn.W_k.data[base + stack0_preserve_row, BD.OP_ENT] = (
                -0.25 * stack0_preserve_s
            )
            for marker_i in range(5):
                attn.W_k.data[base + stack0_preserve_row, BD.H1 + marker_i] = (
                    -4.0 * stack0_preserve_s
                )

        # MEM address bytes are being generated by L14/L16, not loaded from
        # historical memory.  The autoregressive query for MEM_addr{N+1} sits
        # on MEM_addrN, which carries H1[MEM] + IS_BYTE + BYTE_INDEX_N but
        # not MARK_MEM/MEM_STORE.  Block the corresponding byte heads before
        # stale MEM values can project into OUTPUT_LO/HI.
        mem_addr_byte_blocker = 36
        if head in (1, 2, 3):
            mem_addr_block_s = 100000.0
            attn.W_q.data[base + mem_addr_byte_blocker, BD.H1 + mem_i] = (
                mem_addr_block_s
            )
            attn.W_k.data[base + mem_addr_byte_blocker, BD.CONST] = -20.0

        # STACK0 marker queries are memory lookups only for pop-group ops.
        # Non-pop steps should keep the upstream STACK0 passthrough value; L15
        # can otherwise read a stale historical zero through partial top-store
        # matches. Keep this broad blocker modest: CMP can leak onto SP bytes,
        # where an overlarge negative query term creates false-positive scores.
        nonpop_stack0_marker_blocker = 63
        if nonpop_stack0_marker_blocker < HD:
            if hasattr(attn, "W_v"):
                attn.W_v.data[base + nonpop_stack0_marker_blocker, :] = 0.0
            attn.W_o.data[:, base + nonpop_stack0_marker_blocker] = 0.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.MARK_STACK0
            ] = 60000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.CMP + 3
            ] = -15000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.IS_BYTE
            ] = 60000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.OP_LI_RELAY
            ] = -60000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.OP_LC_RELAY
            ] = -60000.0
            # IMM and other non-pop preservers can still need to reveal the
            # already-stored stack top. At the common one-argument call slot
            # 0xffe8, let the ordinary address-matched lookup rows compete
            # instead of forcing the softmax1 zero sink.
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.ADDR_B0_LO + 8
            ] = -40000.0
            attn.W_q.data[
                base + nonpop_stack0_marker_blocker, BD.ADDR_B0_HI + 14
            ] = -30000.0
            attn.W_k.data[
                base + nonpop_stack0_marker_blocker, BD.CONST
            ] = -20.0
            top_store_e8_from_e0_s = 10000.0
            row = base + nonpop_stack0_marker_blocker
            attn.W_q.data[row, BD.MEM_STORE] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.EMBED_LO + 8] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.EMBED_HI + 14] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.ADDR_B0_LO + 0] = top_store_e8_from_e0_s
            attn.W_q.data[row, BD.ADDR_B0_HI + 14] = -2.0 * top_store_e8_from_e0_s

        # A load-only source gate must not be the thing that keeps L15 quiet
        # during current store generation. Add an explicit current-MEM query
        # blocker with only a negative constant key, so softmax1 chooses the
        # zero sink instead of overwriting L14's in-flight store bytes.
        current_store_blocker = 38
        current_store_block_s = 10000.0
        attn.W_q.data[base + current_store_blocker, BD.MARK_MEM] = current_store_block_s
        attn.W_q.data[base + current_store_blocker, BD.H3 + mem_i] = current_store_block_s
        attn.W_k.data[base + current_store_blocker, BD.CONST] = -20.0

        # Current-store generation (PSH/JSR/ENT/SI/SC) can carry enough
        # address-like residue in the in-flight MEM section to look like a
        # valid lookup query. L15 lookup heads only read historical MEM
        # entries, so query-side gates with nonnegative K semantics get a
        # strong current-MEM-section blocker. Do not key this on MEM_STORE
        # alone: SI/SC legitimately load STACK0 while also broadcasting a
        # leaked MEM_STORE relay at the STACK0 marker.
        #
        # Do not apply this to rows 1 or 3: their K side can be negative at
        # non-target sources, and a huge negative query there would become a
        # false-positive score.
        for dim in (0, 28, 29, 30, 31, 32, 33):
            attn.W_q.data[base + dim, BD.MARK_MEM] = -100000.0
            attn.W_q.data[base + dim, BD.H3 + mem_i] = -100000.0
        attn.W_q.data[base + 29, BD.MARK_MEM] = -100000.0
        attn.W_q.data[base + 29, BD.H3 + mem_i] = -100000.0
        attn.W_q.data[base + 29, BD.H1 + pc_i] = -20000.0
        attn.W_k.data[base + 29, BD.CONST] = 5.0
        attn.W_q.data[base + 30, BD.H1 + ax_i] = -20000.0
        attn.W_k.data[base + 30, BD.CONST] = 5.0
        attn.W_q.data[base + 31, BD.OP_LI_RELAY] = 20000.0
        if head == 0:
            attn.W_q.data[base + 31, BD.OP_LC_RELAY] = 20000.0
        else:
            attn.W_q.data[base + 31, BD.MARK_AX] = -20000.0
        attn.W_q.data[base + 31, BD.OP_SI] = -20000.0
        attn.W_q.data[base + 31, BD.OP_SC] = -20000.0
        attn.W_k.data[base + 31, BD.MEM_STORE] = 5.0

        # The AX marker is the query position for byte 0. H1[AX] only covers
        # AX value-byte positions, so guard MARK_AX separately and restore the
        # marker path only for head 0 on real LI/LC loads.
        attn.W_q.data[base + 32, BD.MARK_AX] = -20000.0
        attn.W_k.data[base + 32, BD.CONST] = 5.0
        if head == 0:
            attn.W_q.data[base + 33, BD.OP_LI_RELAY] = 20000.0
            attn.W_q.data[base + 33, BD.OP_LC_RELAY] = 20000.0
            attn.W_k.data[base + 33, BD.MEM_STORE] = 5.0

        # C4_L15_LI_SUPPR_INERT: un-bury the head-0 content-addressable load on
        # LI/LC rows (mirror of the declarative override in
        # _layer15_memory_lookup_heads_0_3_specs_with_overrides). This is the
        # writer that actually lands (re-run by make_l15_attention_resize_op).
        # See _l15_li_load_suppressor_inert_on for the full root + proof.
        if head == 0 and _l15_li_load_suppressor_inert_on():
            # Mirror of the declarative override in
            # _layer15_memory_lookup_heads_0_3_specs_with_overrides. This is the
            # writer that actually lands (re-run by make_l15_attention_resize_op).
            # Rescale slot 58 (1e10->1e6) for a precision-safe cancel, drop the
            # slot-62 OP_ENT misfire, then for each suppressor add a gate slot
            # (K = -suppressor.K per-key, Q = +suppressor.load-row-Q only on load
            # rows) so suppressor+gate == 0 per key on a load row (address-
            # independent) and ZERO off-load. See _l15_li_load_suppressor_inert_on.
            _row58_rescale = 10000.0
            for _d, _w in (
                (BD.OP_ENT, 200000000.0 / _row58_rescale),
                (BD.MARK_STACK0,
                 (early_ent_stack0_q + early_ent_stack0_gate) / _row58_rescale),
                (BD.CONST,
                 -(early_ent_stack0_q + early_ent_stack0_gate) / _row58_rescale),
                (BD.IS_BYTE, -2000000000.0 / _row58_rescale),
                (BD.MARK_AX, -2000000000.0 / _row58_rescale),
                (BD.MARK_PC, -2000000000.0 / _row58_rescale),
                (BD.MARK_SP, -2000000000.0 / _row58_rescale),
                (BD.MARK_BP, -2000000000.0 / _row58_rescale),
                (BD.MARK_MEM, -2000000000.0 / _row58_rescale),
            ):
                attn.W_q.data[base + 58, _d] = _w
            attn.W_k.data[base + 58, BD.OP_ENT] = 1.0
            attn.W_q.data[base + 62, BD.OP_ENT] = 0.0
            _GATE = {34: 70, 35: 64, 58: 65, 59: 66, 60: 67, 61: 68, 62: 69}
            for _suppr, _gate in _GATE.items():
                attn.W_q.data[base + _gate, :] = 0.0
                attn.W_k.data[base + _gate, :] = 0.0
                # cancel.Q = suppressor.Q (copy), cancel.K = -suppressor.K, so
                # cancel.product == -suppressor.product on every key -> inert.
                _qrow = attn.W_q.data[base + _suppr]
                _krow = attn.W_k.data[base + _suppr]
                for _d in torch.nonzero(_qrow, as_tuple=False).flatten().tolist():
                    attn.W_q.data[base + _gate, _d] = float(_qrow[_d])
                for _d in torch.nonzero(_krow, as_tuple=False).flatten().tolist():
                    attn.W_k.data[base + _gate, _d] = -float(_krow[_d])


def _suppress_l15_lookup_lev_blockers_4_11(attn, BD, HD) -> None:
    """LEV blocker rows for heads 4-11 (fires when ``num_heads >= 12``).

    Phase 7.C.2 factored portion of the suppress helper: the LEV-aware
    12-head build adds heads 4-11 for saved-BP and return-PC memory
    reads. They are load-side heads too, so they must also stay silent
    while the current step is generating a store MEM section. Use only
    rows whose K side is positive for all sources; adding MEM_STORE
    blockers to address or negative-constant rows can create
    negative-query × negative-key false positives.

    Carried as a :class:`RuntimeAttentionFragment` in
    ``layer15_memory_lookup``'s CompilerIR. The legacy umbrella loops
    ``range(4, min(num_heads, 12))`` so this is a no-op when
    ``num_heads <= 4``. DSL Wave W7 selects this fragment at IR-build
    time on ``num_heads > 4`` -- byte-identical with always-emitting
    (the body's range is empty below the threshold) but the gate keeps
    the IR's runtime-shape intent ("these are heads 4+ blocker rows")
    visible at the builder site.
    """
    mem_i = 4
    for head in range(4, min(getattr(attn, "num_heads", 4), 12)):
        base = head * HD
        for row in (0, 36, 37):
            if base + row < attn.W_q.data.shape[0]:
                attn.W_q.data[base + row, BD.MARK_MEM] = -100000.0
                attn.W_q.data[base + row, BD.H3 + mem_i] = -100000.0


def _pop_d8_head_9_dim_positions(BD) -> dict:
    """Resolve the named dim layout used by :func:`_pop_d8_head_9_spec`.

    The V2.1 primitives (``binary_address_lookup_attention`` /
    ``attention_head_extension``) take a ``dim_positions`` map keyed by
    name; this helper materializes the names the pop_d8 head reads
    from ``BD`` so the spec builder stays free of direct ``BD``
    attribute access.
    """
    return {
        "CONST": BD.CONST,
        "MARK_STACK0": BD.MARK_STACK0,
        "HAS_SE": BD.HAS_SE,
        "CMP": BD.CMP,
        "ADDR_B0_LO": BD.ADDR_B0_LO,
        "ADDR_B0_HI": BD.ADDR_B0_HI,
        "IS_BYTE": BD.IS_BYTE,
        "MEM_STORE": BD.MEM_STORE,
        "MARK_AX": BD.MARK_AX,
        "MARK_PC": BD.MARK_PC,
        "MARK_SP": BD.MARK_SP,
        "MARK_BP": BD.MARK_BP,
        "MARK_MEM": BD.MARK_MEM,
        "MEM_VAL_B1": BD.MEM_VAL_B1,
        "CLEAN_EMBED_LO": BD.CLEAN_EMBED_LO,
        "CLEAN_EMBED_HI": BD.CLEAN_EMBED_HI,
        "OUTPUT_LO": BD.OUTPUT_LO,
        "OUTPUT_HI": BD.OUTPUT_HI,
    }


def _pop_d8_head_9_spec(
    *,
    head_idx: int,
    head_dim: int,
    BD,
) -> DeclarativeAttentionHeadSpec:
    """Declarative spec for the L15 head-9 pop_d8_to_e0 lookup.

    Built via the V2.1 attention primitives (proof-of-concept
    migration off the imperative writer body — see
    :func:`_suppress_l15_lookup_pop_d8_head_9` for the legacy form
    and ``docs/RUNTIME_ATTN_GAPS_2026_06_04.md`` for the design
    rationale).

    Structure:

    * Slot 0 — universal-sink bias row: ``W_q[CONST]=1``,
      ``W_k[CONST]=-1000``. Carried by
      :func:`binary_address_lookup_attention` (no binary address bands;
      the helper supports an empty ``addr_dim_bases`` for the
      bias-only case).
    * Slot ``min(head_dim-1, 63)`` — the pop_d8 discriminator row:
      strong negative bias (``CONST=-4S``) gated up by 5 positive Q
      writes (MARK_STACK0, HAS_SE, CMP+3, ADDR_B0_LO+8, ADDR_B0_HI+13),
      with IS_BYTE/MEM_STORE/marker suppressors on the Q side and
      MEM_VAL_B1 / ADDR_B0_LO+0 / ADDR_B0_HI+14 on the K side. Added
      via :func:`attention_head_extension`.
    * Slots 1..16 (LO) and 17..32 (HI) — V/O block copying
      ``CLEAN_EMBED`` to ``OUTPUT`` with V=1 and O=40. Added via
      :func:`attention_head_extension`.

    ALiBi slope is set to ``1.0`` for this head.
    """
    dp = _pop_d8_head_9_dim_positions(BD)

    # Slot 0 is the universal sink: Q[CONST]=+1, K[CONST]=-1000.
    # binary_address_lookup_attention places these via bias_dim /
    # key_bias_dim with no binary-address bands.
    base = binary_address_lookup_attention(
        head_idx=head_idx,
        addr_dim_bases=(),  # no binary address bands on this head
        addr_width_bits=4,
        addr_slot_base=4,
        bias_slot=0,
        bias_dim="CONST", bias_weight=1.0,
        key_bias_dim="CONST", key_bias_weight=-1000.0,
        head_dim=head_dim,
        dim_positions=dp,
    )

    # Slot for the pop_d8 discriminator row.
    row = min(head_dim - 1, 63)
    S = 50000.0  # ``pop_d8_to_e0_s`` in the legacy writer.

    extra_q: list = [
        AP(row, dp["CONST"], -4.0 * S),
        AP(row, dp["MARK_STACK0"], S),
        AP(row, dp["HAS_SE"], S),
        AP(row, dp["CMP"] + 3, S),
        AP(row, dp["ADDR_B0_LO"] + 8, S),
        AP(row, dp["ADDR_B0_HI"] + 13, S),
        AP(row, dp["IS_BYTE"], -5.0 * S),
        AP(row, dp["MEM_STORE"], -8.0 * S),
    ]
    # Marker negatives at -5*S. Order mirrors the legacy writer's
    # ``for marker_dim in (...)`` tuple so the produced AP list lines
    # up one-for-one when audited side-by-side.
    for marker_name in ("MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
                        "MARK_MEM"):
        extra_q.append(AP(row, dp[marker_name], -5.0 * S))

    extra_k = (
        AP(row, dp["MEM_VAL_B1"], 1.0),
        AP(row, dp["ADDR_B0_LO"] + 0, 1.0),
        AP(row, dp["ADDR_B0_HI"] + 14, 1.0),
    )

    # V/O block: V slot ``1+idx`` reads CLEAN_EMBED_LO+idx (V=1.0),
    # V slot ``17+idx`` reads CLEAN_EMBED_HI+idx; O writes the matched
    # value slot back to OUTPUT_LO/HI with scale 40.0. Mirrors the
    # legacy ``for idx in range(16):`` loop exactly.
    extra_v: list = []
    extra_o: list = []
    for idx in range(16):
        extra_v.append(AP(1 + idx, dp["CLEAN_EMBED_LO"] + idx, 1.0))
        extra_v.append(AP(17 + idx, dp["CLEAN_EMBED_HI"] + idx, 1.0))
        extra_o.append(AO(dp["OUTPUT_LO"] + idx, 1 + idx, 40.0))
        extra_o.append(AO(dp["OUTPUT_HI"] + idx, 17 + idx, 40.0))

    return attention_head_extension(
        base,
        extra_q_writes=tuple(extra_q),
        extra_k_writes=extra_k,
        extra_v_writes=tuple(extra_v),
        extra_o_writes=tuple(extra_o),
        alibi_slope=1.0,
    )


def _suppress_l15_lookup_pop_d8_head_9(attn, BD, HD) -> None:
    """Head-9 wipe + pop_d8_to_e0 rewrite (fires when ``num_heads > 9``).

    Phase 7.C.2 factored portion of the suppress helper: a pop after a
    one-word pushed result can leave the STACK0 marker keyed by the
    pre-pop 0xffd8 slot while the revealed value lives at the post-pop
    0xffe0 slot.  Use an otherwise-unused late head for this exact
    lookup so the fix does not perturb the dense legacy score rows in
    head 0.

    Wipes head 9's full Q/K/V/O bands first, then writes the pop_d8
    lookup row. Carried as a :class:`RuntimeAttentionFragment` in
    ``layer15_memory_lookup``'s CompilerIR; DSL Wave W7 selects it at
    IR-build time on ``num_heads > 9`` rather than via a runtime
    predicate.

    V2.1 migration (this commit): the dense row writes are now built
    declaratively via :func:`_pop_d8_head_9_spec` (which composes
    :func:`binary_address_lookup_attention` and
    :func:`attention_head_extension`) and lowered through
    :func:`Primitives.generate_attention_head`. The wipe-then-write
    semantics are preserved: the head's row block is zeroed in-place
    first so the LEV writer's earlier head-9 writes (when
    ``num_heads >= 12``) are dropped before the spec's writes land.
    Byte-identity gate lives at
    ``tests/test_l15_pop_d8_v21_migration.py``.
    """
    head = 9
    base = head * HD
    # Wipe head 9's Q/K/V/O bands before the spec writes land.
    # binary_address_lookup_attention + attention_head_extension emit
    # an exact superset of distinct (row, col) writes; the wipe ensures
    # any prior writes by _set_layer15_memory_lookup_lev_heads_4_11
    # (which touches head 9 at num_heads >= 12) are cleared first.
    attn.W_q.data[base:base + HD, :] = 0.0
    attn.W_k.data[base:base + HD, :] = 0.0
    attn.W_v.data[base:base + HD, :] = 0.0
    attn.W_o.data[:, base:base + HD] = 0.0

    spec = _pop_d8_head_9_spec(head_idx=head, head_dim=HD, BD=BD)
    Primitives.generate_attention_head(attn, spec, HD=HD)


def _suppress_l15_lookup_during_current_store_generation(attn, BD, HD) -> None:
    """Keep L15 memory lookup from overwriting L14's current store tokens.

    L15 is a load-side attention op: it reads historical MEM stores for LI/LC
    and pop-group STACK0 loads. During PSH/SI/SC/JSR/ENT, L14 is still
    generating the current MEM section. Those in-flight MEM byte positions have
    MEM_STORE set and can also carry byte-index/ADDR_KEY features, making L15
    look target-like and add a zero-valued load result over L14's freshly
    emitted store byte. Suppress only query positions with MEM_STORE set; the
    historical store tokens remain available as K-side memory entries.

    Phase 7.C.2 split this umbrella into three runtime-shape pieces so
    the ``layer15_memory_lookup`` op carries them as
    :class:`RuntimeAttentionFragment` entries in its CompilerIR. DSL
    Wave W7 (current) moves the shape gating into compile-time
    Python ``if`` branches inside :func:`_layer15_memory_lookup_ir`,
    parameterized on ``num_heads``:

    * :func:`_suppress_l15_lookup_heads_0_3` — always emits.
    * :func:`_suppress_l15_lookup_lev_blockers_4_11` — emits when
      ``num_heads > 4`` (the LEV-blocker rows live on heads 4+).
    * :func:`_suppress_l15_lookup_pop_d8_head_9` — emits when
      ``num_heads > 9`` (the head-9 pop_d8 rewrite).

    Kept as a single legacy entry point for the other callers
    (``vm_step._set_layer15_memory_lookup`` and
    ``make_l15_attention_resize_op``) and for ``tests/test_l15_per_op.py``;
    the production ``layer15_memory_lookup`` bake routes through the IR.
    """
    _suppress_l15_lookup_heads_0_3(attn, BD, HD)
    _suppress_l15_lookup_lev_blockers_4_11(attn, BD, HD)
    if getattr(attn, "num_heads", 0) > 9:
        _suppress_l15_lookup_pop_d8_head_9(attn, BD, HD)


def make_layer15_nibble_copy_op() -> Operation:
    """L15 FFN: Conditional nibble copy OUTPUT = EMBED for non-register byte values.

    Bound to the same block as ``layer15_memory_lookup`` via
    ``target_op_name`` (Phase 8.A.4 drop of the literal ``layer_idx=15``
    pin) so the bake hits whichever block the compiler picks for the L15
    attn op. ``migrated=True`` claims this bake from the legacy
    ``set_vm_weights`` pipeline (the inline call has been removed at the
    original site).
    """
    def bake(block, dim_positions, S):
        # Per-bake FFN-unit allocator. Each L15 nibble-copy sub-stage is
        # pinned to its existing offset so the IR lowerer below lands
        # byte-identically. The allocator is stashed on the FFN so
        # downstream tools (e.g. a future L15 op family claiming a free
        # gap past unit 42) can inspect or extend the layout. Mirrors
        # the ``_l9_unit_allocator`` convention introduced in commit
        # ca775eb.
        allocator = _allocate_layer15_units()
        block.ffn._l15_unit_allocator = allocator

        next_unit = lower_l15_nibble_copy_ir(
            block.ffn,
            _as_setdim_proxy(dim_positions),
            S=S,
        )
        # Byte-identity guard: the IR lowerer's local cursor MUST end
        # exactly at the layout table's total width (42). If the rule
        # list drifts from the table, this assertion fires before any
        # downstream consumer notices the offset mismatch.
        expected_end = _L15_FFN_UNIT_LAYOUT[-1][1] + _L15_FFN_UNIT_LAYOUT[-1][2]
        assert next_unit == expected_end, (
            f"L15 nibble-copy unit cursor drift: lowerer returned "
            f"{next_unit}, allocator expected {expected_end}"
        )

    return Operation(
        name="layer15_nibble_copy",
        reads={"IS_BYTE", "H1", "H4", "MEM_STORE",
               "EMBED_LO", "EMBED_HI", "PSH_AT_SP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "MARK_BP", "MARK_STACK0", "HAS_SE", "CMP"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=make_l15_nibble_copy_ir(),
        # Phase 8.A.4: dropped ``layer_idx=15`` pin in favour of
        # ``target_op_name``. The block binds to whichever layer the
        # compiler placed ``layer15_memory_lookup`` at, and the per-block
        # FFN sizing aggregate honours ``target_op_name`` as well so
        # ``ffn_units_used=42`` still routes to the same target layer.
        target_op_name="layer15_memory_lookup",
        migrated=True,
        requires={"after": "layer15_memory_lookup"},
        # ``_set_nibble_copy_ffn`` writes 40 units (see vm_step.py:2411):
        #   16 LO copy + 16 HI copy + 2 PSH SP byte0 + 2 PSH SP byte1 +
        #   2 PSH SP byte2 + 2 PSH BP byte2 + 2 LEA first-step AX byte2 = 42.
        ffn_units_used=42,
        # Wave 2 (docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py``. Rules write OUTPUT_LO /
        # OUTPUT_HI_THIS_STEP at multiple byte positions (16 LO + 16 HI
        # nibble copies, plus PSH SP/BP byte fixups and LEA AX byte2
        # first-step fixup). No single anatomical slot covers all rules;
        # tag with the bind-target op per the POC fallback convention.
        # consumes_fresh: gate / condition dims that survive the
        # _CROSS_STEP_DURABLE allowlist filter (IS_BYTE, H1, MARK_*,
        # BYTE_INDEX_* drop out). CMP / HAS_SE / MEM_STORE / PSH_AT_SP
        # are L6/L7 marker writes earlier in the same step; H4 is the
        # L1 threshold-attn nibble decode (same step at L1).
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def _l15_attention_resize_follow_up(block, dim_positions, S) -> None:
    """Post-resize bookkeeping + suppress-lookup helper for L15.

    Used as :attr:`StructuralOp.follow_up` by
    :func:`_l15_attention_resize_structural_ir`. Stashes the L15
    attention-head allocator on ``attn._l15_head_allocator`` so
    downstream tooling can audit the head axis after the resize, then
    invokes
    :func:`_suppress_l15_lookup_during_current_store_generation`
    against the resized block. The suppress helper is the legacy
    imperative writer of the L15 load-head Q/K/V cells; it remains
    helper-shaped because its writes branch on ``attn.num_heads`` after
    the resize.
    """

    del S  # the suppress helper does not consume S
    attn = block.attn
    head_allocator = _allocate_layer15_attention_heads()
    attn._l15_head_allocator = head_allocator
    head_dim = getattr(
        attn, "head_dim", attn.W_q.shape[1] // attn.num_heads
    )
    _suppress_l15_lookup_during_current_store_generation(
        attn,
        _as_setdim_proxy(dim_positions),
        head_dim,
    )


def _l15_attention_resize_structural_ir(dim_positions, head_dim) -> CompilerIR:
    """Build the declarative :class:`CompilerIR` for ``l15_attention_resize``.

    Phase 8.I closing audit: the bake's structural intent (resize L15
    attention to 14 / 9 heads depending on ``n_layers_hint``, pin load
    head ALiBi to 0.05) lives in a :class:`StructuralOp` with
    ``kind="attention_resize"``. The follow-up imperative pass (head
    allocator stash + suppress helper) is carried by
    :attr:`StructuralOp.follow_up`. The lowerer in
    :meth:`CompilerIR.lower_structural_ops` handles the
    ``num_heads``/``W_q``/``W_k``/``W_v``/``W_o`` reallocation and the
    ALiBi pin, then dispatches the follow-up.
    """

    del dim_positions, head_dim  # the structural op is shape-agnostic
    ir = CompilerIR()
    ir.layer(0).structural_ops.append(
        StructuralOp(
            kind="attention_resize",
            target_num_heads=14,
            small_num_heads=9,
            layers_threshold=16,
            alibi_pin_value=0.05,
            alibi_pin_count=4,
            follow_up=_l15_attention_resize_follow_up,
            metadata={
                "op_name": "l15_attention_resize",
                "spec_section": "BLOG_SPEC.md#registers",
            },
        )
    )
    return ir


def make_l15_attention_resize_op() -> Operation:
    """Resize L15 attention for late memory/ALU relay heads.

    Migrates the inline resize that previously lived in `set_vm_weights`. The
    extra heads (8-11) hold saved_bp and return_addr reads alongside the
    existing LI/LC/STACK0 reads (heads 0-3) and val heads (4-7). Required
    when the model has L16 (>=17 layers) -- `_set_layer15_memory_lookup`
    keys off `attn.num_heads >= 12` to populate the LEV-specific heads.

    LEV still needs 12 heads in 17-layer builds. The strict 16-layer neural
    smoke path needs one additional head for the staged wide-ALU byte relay,
    while keeping ``attn.num_heads < 12`` so the LEV-specific memory lookup
    bake remains disabled.

    Phase 8.I closing audit: the bake routes through
    :meth:`CompilerIR.lower_structural_ops`. The :class:`StructuralOp`
    (kind="attention_resize", target_num_heads=14, small_num_heads=9,
    layers_threshold=16, alibi_pin_value=0.05) carries the head-count
    branching and ALiBi pin as data; the follow-up callable
    (:func:`_l15_attention_resize_follow_up`) handles the L15
    head-allocator stash + suppress-lookup helper. The op upgrades to
    ``declarative_authority="spec_generated"`` to reflect that the
    structural intent now lives in the IR.
    """

    def bake(block, dim_positions, S):
        # Declarative dispatch: the structural intent (resize to 14/9
        # heads, ALiBi load-head pin, follow-up suppress helper) lives
        # in the CompilerIR built by ``_l15_attention_resize_structural_ir``.
        # ``ir.lower_structural_ops`` performs the resize then invokes
        # the follow-up against the resized block.
        ir = _l15_attention_resize_structural_ir(dim_positions, None)
        ir.lower_structural_ops(block, dim_positions, S=S)

    return Operation(
        name="l15_attention_resize",
        # Phase 1 (memory cluster fix plan): shares L15 ``block.attn`` with
        # ``layer15_memory_lookup``. See that op for the rationale +
        # docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("attn",),
        reads=set(),
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_l15_attention_resize_structural_ir,
        # Phase 8.A.4: dropped ``layer_idx=15`` pin in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer15_memory_lookup`` (the L15 attn op).
        target_op_name="layer15_memory_lookup",
        migrated=True,
        declarative_authority="spec_generated",
        # B12 backfill (wave 1c): structural cleanup that resizes L15
        # attention after the nibble-copy weights are baked. The op writes
        # no dims, so the dep DAG can't derive the post-bake placement on
        # its own. See docs/B12_BACKFILL_SPEC.md §1.
        requires={"after": "layer15_nibble_copy"},
        # Dim-ownership claims: empty. ``bake`` rebuilds
        # ``model.blocks[15].attn`` -- resizes head count (14 for 17-
        # layer LEV builds, 9 for default 16-layer builds) which
        # replaces the attention's ``W_q`` / ``W_k`` / ``W_v`` /
        # ``W_o`` parameters wholesale, then runs a follow-up that
        # writes ALiBi pin / suppress-lookup helper state. Module
        # replacement, not per-cell ``(layer, scope, identifier,
        # column)`` writes. Sentinel below documents the structural
        # effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L15.attn[resize num_heads]'},
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#registers",
    )
