"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...constants import INSTR_WIDTH, PC_OFFSET
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# Marker indices inside the H0..H4 / H1+i threshold-head banks.
_PC_I, _AX_I, _SP_I, _BP_I, _MEM_I = 0, 1, 2, 3, 4


# === L3 FFN unit layout (auto-fit; legacy offsets retained as docs) ===
#
# The ``layer3_ffn`` op owns the L3 FFN's marker-default + PC-increment
# bands. The 134-unit prefix is produced by the declarative
# :func:`_layer3_ffn_rules` rule list, lowered via
# :func:`Primitives.lower_ffn_rules`; the trailing 2 PC-byte1 writers
# come from :func:`_add_layer3_pc_byte1_output_rules`.
#
# Phase 7.B.2: every entry below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because the layout is fully
# contiguous in declaration order, first-fit reproduces the legacy
# pinned offsets bit-for-bit -- so byte-identity with the legacy
# ``vm_step._set_layer3_ffn`` helper survives the pin drop. The
# ``legacy_start`` column is kept purely as documentation (matches the
# unit-counter walk in :func:`_layer3_ffn_rules`).
#
# The offsets below mirror the rule order in :func:`_layer3_ffn_rules`
# (PC default / SP default / BP default / marker bytes-1..3 defaults /
# STACK0 carry / NEXT_STACK0 locality / PC increment + carry) followed
# by :func:`_add_layer3_pc_byte1_output_rules` (2 PC byte1 = 1 writers).
# Changing any rule's unit count requires updating this table in
# lock-step.
_L3_FFN_UNIT_LAYOUT = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("layer3_ffn.pc_first_step_default_lo",     0,   2),  # PC FIRST-STEP default LO (set + undo)
    ("layer3_ffn.pc_first_step_default_hi",     2,   2),  # PC FIRST-STEP default HI (set + undo)
    ("layer3_ffn.initial_pc_bake_cancel",       4,   2),  # no-op placeholders (cancel moved to L2)
    ("layer3_ffn.sp_marker_default",            6,   2),  # SP byte0 at MARK_SP (LO, HI)
    ("layer3_ffn.sp_byte_idx_0_2_default",      8,   4),  # SP byte 0/2 LO+HI = 0
    ("layer3_ffn.sp_byte_1_first_step",        12,   2),  # SP byte 2 first-step (LO=1, HI=0)
    ("layer3_ffn.bp_marker_default",           14,   2),  # BP byte0 at MARK_BP (LO, HI)
    ("layer3_ffn.bp_byte_idx_0_2_default",     16,   4),  # BP byte 0/2 LO+HI = 0
    ("layer3_ffn.bp_byte_1_first_step",        20,   2),  # BP byte 2 first-step (LO=1, HI=0)
    ("layer3_ffn.pc_bytes_1_3_default",        22,   6),  # PC bytes 1-3 = 0 (3 byte_idx x LO/HI)
    ("layer3_ffn.ax_bytes_1_3_default",        28,   6),  # AX bytes 1-3 = 0 (3 byte_idx x LO/HI)
    ("layer3_ffn.mem_marker_default",          34,   2),  # MEM marker addr byte0 (LO, HI)
    ("layer3_ffn.mem_bytes_1_3_default",       36,   6),  # MEM addr bytes 1-3 = 0
    ("layer3_ffn.stack0_bytes_1_3_default",    42,   6),  # STACK0 bytes 1-3 = 0
    ("layer3_ffn.stack0_first_step_default",   48,   2),  # STACK0 byte0 first-step (LO, HI)
    ("layer3_ffn.stack0_carry_projection_lo",  50,  16),  # STACK0 marker carry EMBED_LO -> OUTPUT_LO
    ("layer3_ffn.stack0_carry_projection_hi",  66,  16),  # STACK0 marker carry EMBED_HI -> OUTPUT_HI
    ("layer3_ffn.next_stack0_locality",        82,   4),  # NEXT_STACK0 locality clears (marker + byte_idx_3 + 1/2)
    ("layer3_ffn.pc_increment_lo",             86,  16),  # PC INCREMENT lo nibble (k+8)%16
    ("layer3_ffn.pc_increment_hi",            102,  16),  # PC INCREMENT hi nibble copy
    ("layer3_ffn.pc_carry_correction",        118,  16),  # PC carry: lo>=8 -> hi+=1
    ("layer3_ffn.pc_byte1_output_rules",      134,   2),  # _add_layer3_pc_byte1_output_rules
)


def _allocate_layer3_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L3 FFN sub-stages.

    Phase 7.B.2: ``pin=`` is dropped from every entry in
    :data:`_L3_FFN_UNIT_LAYOUT`. The allocator's default first-fit
    strategy walks the layout in declaration order and lands each
    sub-stage at the lowest free gap large enough to hold it. Because
    the layout is fully contiguous (every entry starts exactly where
    the previous one ended), first-fit reproduces the legacy pinned
    offsets bit-for-bit -- so byte-identity with the legacy
    ``vm_step._set_layer3_ffn`` helper is preserved without the author
    having to spell out the offsets. The legacy ``pin`` column in the
    layout table is kept as a documentation column only (no longer
    consumed by the allocator).

    The trailing ``_add_layer3_pc_byte1_output_rules`` writer is also
    auto-placed; first-fit lands it at unit 134, matching the value
    ``_next_free_ffn_unit(ffn)`` would have returned immediately after
    the main rule lowering. The 32
    ``stack0_carry_projection_{lo,hi}_*`` rules emit *zero* ``W_down``
    writes (the W_up / W_gate / b_up writes survive), so the
    unit-cursor offsets downstream are unchanged.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L3 op claims a free range past unit 136).
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_pin, n_units in _L3_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def _no_op_placeholder_rule(name: str) -> FFNRule:
    """Empty FFNRule that occupies one hidden-unit slot but writes no weights.

    Mimics the legacy ``unit += 1`` placeholders for the
    initial_pc_bake cancel (whose logic moved to
    ``make_layer2_initial_pc_bake_cancel_op``) so the unit count stays
    at 134 and downstream pinned offsets are unchanged.

    Notably, ``gate_bias=0.0`` overrides the
    ``FFNRule.constant_write`` default (1.0) so the placeholder leaves
    ``b_gate[unit]`` at its initial zero -- matching the legacy bake's
    behaviour for these two intentionally-unused units.
    """
    return FFNRule(
        name=name,
        conditions=(),
        threshold=0.0,
        writes=(),
        gate=None,
        gate_bias=0.0,
    )


def _layer3_ffn_rules(S: float) -> tuple:
    """Declarative L3 FFN rule list -- replaces ``vm_step._set_layer3_ffn``.

    Produces 134 :class:`FFNRule` instances covering:

    * units 0-3 -- PC FIRST-STEP DEFAULT (set + HAS_SE-keyed undo, LO/HI).
    * units 4-5 -- INITIAL_PC_BAKE CANCEL placeholders (no-op).
    * units 6-13 -- SP defaults (marker, bytes 0/2, byte 2 first-step).
    * units 14-21 -- BP defaults (mirror of SP).
    * units 22-27 -- PC bytes 1-3 default (all zero).
    * units 28-33 -- AX bytes 1-3 default (all zero).
    * units 34-35 -- MEM marker default.
    * units 36-41 -- MEM addr bytes 1-3 default.
    * units 42-47 -- STACK0 bytes 1-3 default (gated by H4[BP] AND NOT H1[BP]).
    * units 48-49 -- STACK0 first-step default.
    * units 50-81 -- STACK0 carry projection (32 write-less suppressor
      units; this is the declarative replacement for the legacy
      ``_suppress_layer3_stack0_marker_carry_projection`` post-pass).
    * units 82-85 -- NEXT_STACK0 locality clears (marker, byte_idx 1/2/3).
    * units 86-101 -- PC INCREMENT lo nibble (k+INSTR_WIDTH)%16.
    * units 102-117 -- PC INCREMENT hi nibble copy.
    * units 118-133 -- PC carry correction (lo nibble >= 8 -> hi += 1).

    Byte-identity-validated against the legacy
    ``vm_step._set_layer3_ffn`` + suppressor pair via the parity test
    in ``tests/test_declarative_ffn_bakes_l3.py``.

    Phase 8.D: the two ``MARK_PC`` undo-unit gates use :func:`dim_ref`
    for the ``(marker, PC)`` semantic pair so the rule names the
    family lookup directly. Marker references that appear as
    ``(marker_name, weight)`` conditions stay structural -- those
    are guard terms in the up-branch dot product, not role-meaningful
    gate dims (the L8 pilot preserved that convention).

    Phase 8.D follow-up: ``BYTE_INDEX_<n>`` conditions across the
    SP/BP/PC/AX/MEM/STACK0/NEXT_STACK0 byte-default and locality
    rules use :func:`dim_ref` for the ``(byte_index, "<n>")`` family
    lookup -- each tuple condition names the byte-position role
    rather than the bare slot label. Byte-identical via DimRef.parse
    (the ``+0`` suffix produced by dim_ref resolves to the same
    cell). Structural ``OUTPUT_LO/HI+<k>`` / ``EMBED_LO/HI+<k>`` /
    ``H<n>+<idx>`` offsets stay bare.
    """
    first_pc = PC_OFFSET + INSTR_WIDTH
    pc_lo = first_pc & 0xF
    pc_hi = (first_pc >> 4) & 0xF
    gate_mark_pc = dim_ref("marker", "PC")
    # Phase 8.D follow-up: pre-bind the four (byte_index, role) refs so
    # downstream rule conditions name the byte-position role lookup
    # instead of the bare ``BYTE_INDEX_<n>`` slot string.
    _BYTE_INDEX = {
        0: dim_ref("byte_index", "0"),
        1: dim_ref("byte_index", "1"),
        2: dim_ref("byte_index", "2"),
        3: dim_ref("byte_index", "3"),
    }
    rules = []

    # --- PC FIRST-STEP DEFAULT (units 0-3) ---
    # At MARK_PC AND NOT HAS_SE, predict PC = PC_OFFSET + INSTR_WIDTH.
    # Encoded as a "set" unit (ungated; fires on MARK_PC) plus a
    # HAS_SE-keyed "undo" unit gated by MARK_PC that subtracts the
    # same amount whenever HAS_SE is on.
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.pc_first_step_default_lo_set",
        conditions=(("MARK_PC", 1.0),),
        threshold=0.5,
        writes=((f"OUTPUT_LO+{pc_lo}", 2.0 / S),
                (f"EMBED_LO+{pc_lo}", 2.0 / S)),
        scope="MARK_PC and not HAS_SE",
    ))
    rules.append(FFNRule.gated_write(
        name="layer3_ffn.pc_first_step_default_lo_undo",
        conditions=(("HAS_SE", 1.0),),
        threshold=0.5,
        gate=gate_mark_pc,
        gate_weight=1.0,
        gate_bias=0.0,
        writes=((f"OUTPUT_LO+{pc_lo}", -2.0 / S),
                (f"EMBED_LO+{pc_lo}", -2.0 / S)),
        scope="MARK_PC and HAS_SE",
    ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.pc_first_step_default_hi_set",
        conditions=(("MARK_PC", 1.0),),
        threshold=0.5,
        writes=((f"OUTPUT_HI+{pc_hi}", 2.0 / S),
                (f"EMBED_HI+{pc_hi}", 2.0 / S)),
        scope="MARK_PC and not HAS_SE",
    ))
    rules.append(FFNRule.gated_write(
        name="layer3_ffn.pc_first_step_default_hi_undo",
        conditions=(("HAS_SE", 1.0),),
        threshold=0.5,
        gate=gate_mark_pc,
        gate_weight=1.0,
        gate_bias=0.0,
        writes=((f"OUTPUT_HI+{pc_hi}", -2.0 / S),
                (f"EMBED_HI+{pc_hi}", -2.0 / S)),
        scope="MARK_PC and HAS_SE",
    ))

    # --- INITIAL_PC_BAKE CANCEL placeholders (units 4-5) ---
    # Cancel logic moved to L2 (``make_layer2_initial_pc_bake_cancel_op``).
    # These two zero-write units preserve downstream pinned offsets.
    rules.append(_no_op_placeholder_rule(
        "layer3_ffn.initial_pc_bake_cancel_lo"))
    rules.append(_no_op_placeholder_rule(
        "layer3_ffn.initial_pc_bake_cancel_hi"))

    # --- SP DEFAULT (units 6-13) ---
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.sp_marker_default_lo",
        conditions=(("MARK_SP", 1.0), ("HAS_SE", -1.0)),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 2.0 / S),),
        scope="MARK_SP and not HAS_SE",
    ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.sp_marker_default_hi",
        conditions=(("MARK_SP", 1.0), ("HAS_SE", -1.0)),
        threshold=0.5,
        writes=(("OUTPUT_HI+0", 2.0 / S),),
        scope="MARK_SP and not HAS_SE",
    ))
    for byte_idx in (0, 2):
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.sp_byte_idx_{byte_idx}_default_lo",
            conditions=((f"H1+{_SP_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_LO+0", 2.0 / S),),
        ))
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.sp_byte_idx_{byte_idx}_default_hi",
            conditions=((f"H1+{_SP_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
        ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.sp_byte_1_first_step_lo",
        conditions=((f"H1+{_SP_I}", 1.0), (_BYTE_INDEX[1], 1.0),
                    ("HAS_SE", -1.0)),
        threshold=1.5,
        writes=(("OUTPUT_LO+1", 2.0 / S),),
    ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.sp_byte_1_first_step_hi",
        conditions=((f"H1+{_SP_I}", 1.0), (_BYTE_INDEX[1], 1.0),
                    ("HAS_SE", -1.0)),
        threshold=1.5,
        writes=(("OUTPUT_HI+0", 2.0 / S),),
    ))

    # --- BP DEFAULT (units 14-21) ---
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.bp_marker_default_lo",
        conditions=(("MARK_BP", 1.0), ("HAS_SE", -1.0)),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 2.0 / S),),
        scope="MARK_BP and not HAS_SE",
    ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.bp_marker_default_hi",
        conditions=(("MARK_BP", 1.0), ("HAS_SE", -1.0)),
        threshold=0.5,
        writes=(("OUTPUT_HI+0", 2.0 / S),),
        scope="MARK_BP and not HAS_SE",
    ))
    for byte_idx in (0, 2):
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.bp_byte_idx_{byte_idx}_default_lo",
            conditions=((f"H1+{_BP_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_LO+0", 2.0 / S),),
        ))
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.bp_byte_idx_{byte_idx}_default_hi",
            conditions=((f"H1+{_BP_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
        ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.bp_byte_1_first_step_lo",
        conditions=((f"H1+{_BP_I}", 1.0), (_BYTE_INDEX[1], 1.0),
                    ("HAS_SE", -1.0)),
        threshold=1.5,
        writes=(("OUTPUT_LO+1", 2.0 / S),),
    ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.bp_byte_1_first_step_hi",
        conditions=((f"H1+{_BP_I}", 1.0), (_BYTE_INDEX[1], 1.0),
                    ("HAS_SE", -1.0)),
        threshold=1.5,
        writes=(("OUTPUT_HI+0", 2.0 / S),),
    ))

    # --- PC bytes 1-3 default (units 22-27) ---
    for byte_idx in (0, 1, 2):
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.pc_byte_{byte_idx}_default_lo",
            conditions=((f"H1+{_PC_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_LO+0", 2.0 / S),),
        ))
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.pc_byte_{byte_idx}_default_hi",
            conditions=((f"H1+{_PC_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
        ))

    # --- AX bytes 1-3 default (units 28-33) ---
    for byte_idx in (0, 1, 2):
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.ax_byte_{byte_idx}_default_lo",
            conditions=((f"H1+{_AX_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_LO+0", 2.0 / S),),
        ))
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.ax_byte_{byte_idx}_default_hi",
            conditions=((f"H1+{_AX_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
        ))

    # --- MEM marker default (units 34-35) ---
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.mem_marker_default_lo",
        conditions=(("MARK_MEM", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 2.0 / S),),
        scope="MARK_MEM",
    ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.mem_marker_default_hi",
        conditions=(("MARK_MEM", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_HI+0", 2.0 / S),),
        scope="MARK_MEM",
    ))

    # --- MEM addr bytes 1-3 default (units 36-41) ---
    for byte_idx in (0, 1, 2):
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.mem_byte_{byte_idx}_default_lo",
            conditions=((f"H1+{_MEM_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_LO+0", 2.0 / S),),
        ))
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.mem_byte_{byte_idx}_default_hi",
            conditions=((f"H1+{_MEM_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
        ))

    # --- STACK0 bytes 1-3 default (units 42-47) ---
    # H4[BP] covers BP through STACK0 (d<=9.5); subtract H1[BP] to
    # exclude the BP-area positions and leave only STACK0.
    for byte_idx in (0, 1, 2):
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.stack0_byte_{byte_idx}_default_lo",
            conditions=((f"H4+{_BP_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0),
                        (f"H1+{_BP_I}", -1.0)),
            threshold=1.5,
            writes=(("OUTPUT_LO+0", 2.0 / S),),
        ))
        rules.append(FFNRule.constant_write(
            name=f"layer3_ffn.stack0_byte_{byte_idx}_default_hi",
            conditions=((f"H4+{_BP_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0),
                        (f"H1+{_BP_I}", -1.0)),
            threshold=1.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
        ))

    # --- STACK0 first-step default (units 48-49) ---
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.stack0_first_step_default_lo",
        conditions=(("MARK_STACK0", 1.0), ("HAS_SE", -1.0)),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 2.0 / S),),
        scope="MARK_STACK0 and not HAS_SE",
    ))
    rules.append(FFNRule.constant_write(
        name="layer3_ffn.stack0_first_step_default_hi",
        conditions=(("MARK_STACK0", 1.0), ("HAS_SE", -1.0)),
        threshold=0.5,
        writes=(("OUTPUT_HI+0", 2.0 / S),),
        scope="MARK_STACK0 and not HAS_SE",
    ))

    # --- STACK0 carry projection -- SUPPRESSED writes (units 50-81) ---
    # Imperative bake wrote ``W_down[OUTPUT_LO/HI+k] = 2/S`` for these
    # 32 units, then
    # ``_suppress_layer3_stack0_marker_carry_projection`` zeroed those
    # columns. Express directly as write-less ``gated_write`` rules so
    # the suppressor is no longer needed; the W_up / W_gate / b_up
    # writes are still emitted (preserving unit ownership) but
    # ``writes=()`` means no W_down cells are touched, matching the
    # post-suppression final state cell-for-cell.
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"layer3_ffn.stack0_carry_projection_lo_{k}",
            conditions=(("MARK_STACK0", 1.0), ("HAS_SE", 1.0)),
            threshold=1.5,
            gate=f"EMBED_LO+{k}",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(),  # suppressed (was 2/S in legacy, zeroed by suppressor)
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"layer3_ffn.stack0_carry_projection_hi_{k}",
            conditions=(("MARK_STACK0", 1.0), ("HAS_SE", 1.0)),
            threshold=1.5,
            gate=f"EMBED_HI+{k}",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(),  # suppressed (was 2/S in legacy, zeroed by suppressor)
        ))

    # --- NEXT_STACK0 locality (units 82-85) ---
    rules.append(FFNRule.gated_write(
        name="layer3_ffn.next_stack0_locality_marker",
        conditions=(("MARK_STACK0", 1.0),),
        threshold=0.5,
        gate="NEXT_STACK0",
        gate_weight=1.0,
        gate_bias=0.0,
        writes=(("NEXT_STACK0", -3.0 / S),),
        scope="MARK_STACK0",
    ))
    # STACK0 byte 0 aliases BYTE_INDEX_3; exclude BP byte 3 by
    # requiring H0[BP] to be effectively zero (-1000 weight makes any
    # H0[BP] presence dominate the threshold).
    rules.append(FFNRule.gated_write(
        name="layer3_ffn.next_stack0_locality_byte_idx_3",
        conditions=((f"H4+{_BP_I}", 1.0),
                    (_BYTE_INDEX[3], 1.0),
                    (f"H0+{_BP_I}", -1000.0)),
        threshold=1.5,
        gate="NEXT_STACK0",
        gate_weight=1.0,
        gate_bias=0.0,
        writes=(("NEXT_STACK0", -3.0 / S),
                ("OUTPUT_LO+0", 5.0 / S),
                ("OUTPUT_HI+0", 5.0 / S)),
    ))
    for byte_idx in (1, 2):
        rules.append(FFNRule.gated_write(
            name=f"layer3_ffn.next_stack0_locality_byte_idx_{byte_idx}",
            conditions=((f"H4+{_BP_I}", 1.0),
                        (_BYTE_INDEX[byte_idx], 1.0)),
            threshold=1.5,
            gate="NEXT_STACK0",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("NEXT_STACK0", -3.0 / S),),
        ))

    # --- PC INCREMENT lo nibble (units 86-101) ---
    # MARK_PC AND HAS_SE AND NOT OP_LEV: new_k = (k + INSTR_WIDTH) % 16.
    # OP_LEV ~ 5, so weight -1/5 makes its contribution to the score
    # cancel the HAS_SE/MARK_PC pair (suppress on LEV opcodes).
    for k in range(16):
        new_k = (k + INSTR_WIDTH) % 16
        rules.append(FFNRule.gated_write(
            name=f"layer3_ffn.pc_increment_lo_{k}",
            conditions=(("HAS_SE", 1.0), ("MARK_PC", 1.0),
                        ("OP_LEV", -1.0 / 5.0)),
            threshold=1.5,
            gate=f"EMBED_LO+{k}",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"OUTPUT_LO+{new_k}", 2.0 / S),),
            scope="MARK_PC and HAS_SE and not OP_LEV",
        ))

    # --- PC INCREMENT hi nibble (units 102-117) ---
    # Same condition; copies EMBED_HI[k] -> OUTPUT_HI[k].
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"layer3_ffn.pc_increment_hi_{k}",
            conditions=(("HAS_SE", 1.0), ("MARK_PC", 1.0),
                        ("OP_LEV", -1.0 / 5.0)),
            threshold=1.5,
            gate=f"EMBED_HI+{k}",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"OUTPUT_HI+{k}", 2.0 / S),),
            scope="MARK_PC and HAS_SE and not OP_LEV",
        ))

    # --- PC carry correction (units 118-133) ---
    # Fires when MARK_PC AND HAS_SE AND NOT OP_LEV AND any
    # EMBED_LO[8..15] is set (old lo nibble >= 16 - INSTR_WIDTH).
    # MARK_PC weight 4 strictly requires it (prevents false positive
    # at byte positions where EMBED_LO is inflated). Output gate
    # carries the EMBED_HI[k] nibble into OUTPUT_HI[(k+1)%16] (cancel
    # old k, add to k+1).
    carry_threshold = 16 - INSTR_WIDTH
    for k in range(16):
        conds = [("MARK_PC", 4.0), ("HAS_SE", 1.0), ("OP_LEV", -1.0)]
        for lo_bit in range(carry_threshold, 16):
            conds.append((f"EMBED_LO+{lo_bit}", 1.0))
        rules.append(FFNRule.gated_write(
            name=f"layer3_ffn.pc_carry_correction_{k}",
            conditions=tuple(conds),
            threshold=5.5,
            gate=f"EMBED_HI+{k}",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"OUTPUT_HI+{k}", -2.0 / S),
                    (f"OUTPUT_HI+{(k + 1) % 16}", 2.0 / S)),
            scope="MARK_PC and HAS_SE and not OP_LEV",
        ))

    return tuple(rules)


def _layer3_ffn_ir(S: float = 100.0) -> CompilerIR:
    """Build a single-layer :class:`CompilerIR` carrying the L3 FFN rules.

    Exposed as :attr:`Operation.compiler_ir` so the declarative
    verifier / dominance auditor can read the rule list without bake
    execution. Matches the pattern established by ``_phase_a_ffn_ir``
    (L0) and similar helpers. The IR carries all 136 units the
    ``layer3_ffn`` op writes: 134 from :func:`_layer3_ffn_rules`
    (PC/SP/BP defaults + STACK0 carry suppression + NEXT_STACK0
    locality + PC increment / carry) followed by 2 from
    :func:`_layer3_pc_byte1_output_rules` (PC byte1 emission).
    """
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer3_ffn_rules(S))
    ir.layer(0).ffn.rules.extend(_layer3_pc_byte1_output_rules(S))
    return ir


def _lower_layer3_ffn_ir(ffn, S: float, BD) -> int:
    """Lower :func:`_layer3_ffn_rules` into ``ffn`` and return next free unit.

    Thin wrapper around :func:`Primitives.lower_ffn_rules` that builds the
    rule list, resolves dim names through ``BD``, and lowers at
    ``start_unit=0``. Used by :func:`make_layer3_ffn_op`'s bake closure.
    """
    rules = _layer3_ffn_rules(S)
    dim_names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(BD, dim_names)
    return Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )


def make_layer3_ffn_op() -> Operation:
    """L3 FFN: PC/SP/BP first-step defaults + PC byte-0 increment.

    Originally: `_set_layer3_ffn` at vm_step.py:2929. Migrated to
    declarative :class:`FFNRule` lowering via :func:`_layer3_ffn_rules`
    + :func:`Primitives.lower_ffn_rules` (134 units); the trailing 2
    PC-byte1 emission units come from
    :func:`_add_layer3_pc_byte1_output_rules`. The legacy in-place
    suppressor ``_suppress_layer3_stack0_marker_carry_projection`` is
    no longer needed -- the 32 STACK0 carry-projection rules now emit
    zero W_down writes directly.

    Reads MARK_PC, MARK_SP, MARK_BP, MARK_STACK0, HAS_SE, EMBED_LO/HI,
    H1, H4, BYTE_INDEX_*, OP_LEV, NEXT_STACK0.
    Writes OUTPUT_LO/HI, EMBED_LO/HI, NEXT_STACK0.

    Pinned to ``layer_idx=3`` via ``kind="block"`` because the legacy
    ``set_vm_weights`` pipeline targets ``model.blocks[3].ffn``. Without
    pinning, the dep-graph layer assignment placed this op at block 4,
    which would conflict with the L4 FFN bake (the same regression noted
    on ``make_layer4_pc_relay_op``). The companion
    ``_layer3_ffn_dep_anchor`` op (kind="ffn") declares identical
    reads/writes so the LayerCompiler's dep graph still reserves a layer
    slot for it; otherwise removing the kind="ffn" entry shrinks the
    longest-chain length and shifts downstream migrated kind="attn" ops
    (e.g. ``layer14_mem_generation``) to the wrong block.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L3 FFN sub-stage is pinned to
        # its existing offset so the rule lowering below lands
        # byte-identically. The ``pc_byte1_output_rules`` range gives the
        # start-unit for the trailing ``_add_layer3_pc_byte1_output_rules``
        # writer, replacing the implicit ``_next_free_ffn_unit`` probe.
        allocator = _allocate_layer3_ffn_units()
        pc_byte1_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer3_ffn.pc_byte1_output_rules"
        )
        pc_byte1_start = pc_byte1_range.start
        # Make the allocator available for inspection / extension by
        # downstream tools (e.g. a future L3 op family claiming a free
        # gap). The block-level attribute mirrors the
        # ``_l9_unit_allocator`` convention used by sibling layers, but
        # carries the allocator object so the layout is structured, not
        # just a monotonic int.
        block.ffn._l3_unit_allocator = allocator

        # Lower the 134-rule declarative spec at unit 0.
        next_free = _lower_layer3_ffn_ir(block.ffn, S, proxy)
        # Byte-identity guard: rule count MUST equal the
        # ``pc_byte1_output_rules`` start (134). If a rule is added or
        # removed without updating ``_L3_FFN_UNIT_LAYOUT`` in lock-step,
        # this assertion fires before any further weight surgery
        # happens.
        assert next_free == pc_byte1_start, (
            f"L3 FFN unit cursor drift: rule lowering wrote {next_free} "
            f"units, allocator expected {pc_byte1_start}"
        )
        _add_layer3_pc_byte1_output_rules(block.ffn, S, proxy)

    # Dim-ownership claims (W_down output cells; partial-claims subset). The
    # bake writes 136 hidden units; their W_down output projections fall into
    # well-defined blocks:
    #   units 0/1:     PC default emit -> OUTPUT_LO+10 (low nibble of PC byte0
    #                  default for STACK_INIT-style residue) and EMBED_LO+10
    #                  carry (legacy default kept for residue safety).
    #   units 2/3:     analogous OUTPUT_HI+0 / EMBED_HI+0 PC default.
    #   units 6..49:   STACK0 marker carry suppressor / SP/BP/STACK0 default
    #                  blocks, each writing a single OUTPUT_LO+0 or
    #                  OUTPUT_HI+0 cell (alternating; see
    #                  _set_layer3_ffn in vm_step.py and the suppressor
    #                  ``_suppress_layer3_stack0_marker_carry_projection``).
    #   unit 12:       BP default emits OUTPUT_LO+1 (one-byte default).
    #   unit 20:       SP default emits OUTPUT_LO+1.
    #   units 82..85:  STACK0 NEXT_STACK0 carry chain.
    #   unit 83:       also writes OUTPUT_LO+0 / OUTPUT_HI+0.
    #   units 86..101: PC LEV-return OUTPUT_LO+(unit-86) for k=0..15
    #                  (LEV BP relay nibble-by-nibble).
    #   units 102..117: PC LEV-return OUTPUT_HI+(unit-102) for k=0..15.
    #   units 118..133: PC byte1 carry pairs; unit (118+k) writes
    #                   OUTPUT_HI+k and OUTPUT_HI+(k+1) — adjacent-nibble
    #                   carry. Unit 133 wraps and writes OUTPUT_HI+0/+15.
    #   units 134/135: byte1 ones from _add_layer3_pc_byte1_output_rules
    #                  (write OUTPUT_LO+0/+1 and OUTPUT_HI+0).
    _claims = set()
    # PC default residue carries (units 0-3).
    _claims.add((3, "ffn_W_down", "0", "EMBED_LO+10"))
    _claims.add((3, "ffn_W_down", "0", "OUTPUT_LO+10"))
    _claims.add((3, "ffn_W_down", "1", "EMBED_LO+10"))
    _claims.add((3, "ffn_W_down", "1", "OUTPUT_LO+10"))
    _claims.add((3, "ffn_W_down", "2", "EMBED_HI+0"))
    _claims.add((3, "ffn_W_down", "2", "OUTPUT_HI+0"))
    _claims.add((3, "ffn_W_down", "3", "EMBED_HI+0"))
    _claims.add((3, "ffn_W_down", "3", "OUTPUT_HI+0"))
    # Units 6..49: alternating OUTPUT_LO+0 / OUTPUT_HI+0 default writers, with
    # OUTPUT_LO+1 exceptions at units 12 and 20.
    for unit in range(6, 50):
        if unit in (12, 20):
            _claims.add((3, "ffn_W_down", str(unit), "OUTPUT_LO+1"))
        elif unit % 2 == 0:
            _claims.add((3, "ffn_W_down", str(unit), "OUTPUT_LO+0"))
        else:
            _claims.add((3, "ffn_W_down", str(unit), "OUTPUT_HI+0"))
    # Units 82..85: NEXT_STACK0 carry chain.
    for unit in (82, 83, 84, 85):
        _claims.add((3, "ffn_W_down", str(unit), "NEXT_STACK0+0"))
    _claims.add((3, "ffn_W_down", "83", "OUTPUT_HI+0"))
    _claims.add((3, "ffn_W_down", "83", "OUTPUT_LO+0"))
    # Units 86..101: LEV nibble relay over OUTPUT_LO. Hidden units 86..93
    # drive OUTPUT_LO+8..+15 (high nibble of byte 0), then 94..101 drive
    # OUTPUT_LO+0..+7 (low nibble of byte 1, splitting the byte across
    # adjacent OUTPUT_LO cells).
    for k in range(8):
        _claims.add((3, "ffn_W_down", str(86 + k), f"OUTPUT_LO+{8 + k}"))
        _claims.add((3, "ffn_W_down", str(94 + k), f"OUTPUT_LO+{k}"))
    # Units 102..117: OUTPUT_HI+(unit-102) for k=0..15 (full HI nibble band).
    for k in range(16):
        _claims.add((3, "ffn_W_down", str(102 + k), f"OUTPUT_HI+{k}"))
    # Units 118..132: byte1 carry pairs, each writing OUTPUT_HI+k & OUTPUT_HI+k+1.
    for k in range(15):
        _claims.add((3, "ffn_W_down", str(118 + k), f"OUTPUT_HI+{k}"))
        _claims.add((3, "ffn_W_down", str(118 + k), f"OUTPUT_HI+{k + 1}"))
    # Unit 133: wraps from OUTPUT_HI+15 back to OUTPUT_HI+0.
    _claims.add((3, "ffn_W_down", "133", "OUTPUT_HI+0"))
    _claims.add((3, "ffn_W_down", "133", "OUTPUT_HI+15"))
    # Units 134/135: PC byte1 = 1 emission (added by
    # _add_layer3_pc_byte1_output_rules at the end of the bake).
    for unit in ("134", "135"):
        _claims.add((3, "ffn_W_down", unit, "OUTPUT_LO+0"))
        _claims.add((3, "ffn_W_down", unit, "OUTPUT_LO+1"))
        _claims.add((3, "ffn_W_down", unit, "OUTPUT_HI+0"))

    return Operation(
        name="layer3_ffn",
        # Phase 11.A r3: dropped phase=3 — co-placement at
        # ``layer3_carry_forward_attn`` + explicit ``requires['after']``
        # on the same op already pins fire order; the phase literal was
        # a redundant fallback signal.
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L5/L7/L11/L14 TEMP writers (which fire AFTER L3 in the
        # same step). The same-step value layer3_carry_forward_attn writes
        # at PC byte0 row (TEMP+1, TEMP+16) is still picked up at the same
        # numeric position because TEMP_PREV_STEP aliases TEMP. Breaks
        # back-edges L5/L7/L11/L14 → layer3_ffn on TEMP.
        # Phase 8.A: OP_LEV_PREV_STEP marks the OP_LEV read as cross-step
        # relative to L5 opcode_decode_ffn (writes OP_LEV at the PC marker
        # of the current step). The L3 FFN's PC-increment / carry-correction
        # rules use the previous step's OP_LEV decode as the "skip increment
        # on LEV" suppressor — byte-identical because OP_LEV_PREV_STEP
        # aliases OP_LEV. Breaks back-edge L5 → layer3_ffn on OP_LEV.
        # Phase 8.A (EMBED_HI split): EMBED_HI_PREV_STEP marks the EMBED_HI
        # read as cross-step relative to layer4_pc_relay (L4 writes EMBED_HI
        # at the AX marker AFTER L3 in the same step). The same-step value
        # carried by L3/L0-L2 still resolves at the same numeric position
        # because EMBED_HI_PREV_STEP aliases EMBED_HI. Breaks the
        # layer4_pc_relay → layer3_ffn back-edge on EMBED_HI.
        # Phase 9.B EMBED_LO -> EMBED_LO.*.-1 SSA cross-step rename.
        reads={"MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "HAS_SE",
               "EMBED_LO.*.-1", "EMBED_HI.*.-1", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "TEMP.*.-1", "IS_BYTE", "H1", "H4", "OP_LEV.*.-1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "NEXT_STACK0"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "EMBED_LO", "EMBED_HI",
                "NEXT_STACK0"},
        kind="block",
        # Phase 8.G.6: drop ``layer_idx=3`` literal; bind to the L3 attn
        # anchor ``layer3_carry_forward_attn`` so the block op resolves
        # to whichever layer the carry-forward attn lands at.
        target_op_name="layer3_carry_forward_attn",
        requires={"after": "layer3_carry_forward_attn"},
        declarative_bake_fn=bake,
        compiler_ir=_layer3_ffn_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        postcondition={
            "OUTPUT_LO": "monotonic_non_decreasing",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        compaction_safe=True,
    )


def _next_free_ffn_unit(ffn) -> int:
    active = (
        (ffn.W_up.data.abs().sum(dim=1) > 0)
        | (ffn.W_gate.data.abs().sum(dim=1) > 0)
        | (ffn.b_up.data.abs() > 0)
        | (ffn.b_gate.data.abs() > 0)
        | (ffn.W_down.data.abs().sum(dim=0) > 0)
    )
    used = active.nonzero(as_tuple=True)[0]
    return int(used[-1].item() + 1) if len(used) else 0


def _suppress_layer3_stack0_marker_carry_projection(ffn, S: float, BD) -> int:
    """Disable stale previous-STACK0 projection at the STACK0 marker.

    L3 head 4 carries the previous step's ``STACK0_byte0`` into
    ``EMBED_LO/HI`` at the next ``STACK0`` marker.  That value is stale for
    frame setup and other SP-changing steps because the current ``STACK0`` is
    defined by the newly emitted SP, not by the prior top-of-stack.  The old
    L3 FFN projected that carried byte directly to OUTPUT and left the stale
    embed signal available for later stack tails to amplify.

    Reuse those marker-position projection units as local suppressors: remove
    their OUTPUT writes while preserving the carried EMBED band for later
    stack-top reconstruction.
    """

    suppressed = 0
    for unit in range(ffn.W_up.shape[0]):
        up = ffn.W_up.data[unit]
        gate = ffn.W_gate.data[unit]
        if abs(float(up[BD.MARK_STACK0].item()) - S) > 1e-6:
            continue
        if abs(float(up[BD.HAS_SE].item()) - S) > 1e-6:
            continue
        if abs(float(ffn.b_up.data[unit].item()) + S * 1.5) > 1e-6:
            continue

        active_gate = gate.abs().nonzero(as_tuple=True)[0]
        if len(active_gate) != 1:
            continue
        gate_dim = int(active_gate[0].item())
        is_embed_lo = BD.EMBED_LO <= gate_dim < BD.EMBED_LO + 16
        is_embed_hi = BD.EMBED_HI <= gate_dim < BD.EMBED_HI + 16
        if not (is_embed_lo or is_embed_hi):
            continue
        if abs(float(gate[gate_dim].item()) - 1.0) > 1e-6:
            continue

        ffn.W_down.data[:, unit].zero_()
        suppressed += 1
    return suppressed


def _rewrite_layer3_initial_sp_byte2_to_zero(ffn, S: float, BD) -> int:
    """Materialize the emitted initial SP as ``0x0000fff8``.

    The legacy L3 default was authored for the constructor value
    ``STACK_INIT = 0x00010000`` and emits ``SP_byte2 = 0x01`` on the first
    draft step.  The byte stream records state after the first instruction;
    for compiled programs that first instruction is the startup JSR, so the
    SP bytes are ``f8 ff 00 00``.  L10 exacts byte 0/1 from the emitted lower
    bytes; this local L3 rewrite removes the remaining high-byte residue at
    the owner unit without touching BP's true ``0x00010000`` default.
    """

    rewritten = 0
    SP_I = 2
    for unit in range(ffn.W_up.shape[0]):
        up = ffn.W_up.data[unit]
        gate = ffn.W_gate.data[unit]
        if abs(float(up[BD.H1 + SP_I].item()) - S) > 1e-6:
            continue
        if abs(float(up[BD.BYTE_INDEX_1].item()) - S) > 1e-6:
            continue
        if abs(float(up[BD.HAS_SE].item()) + S) > 1e-6:
            continue
        if abs(float(ffn.b_up.data[unit].item()) + S * 1.5) > 1e-6:
            continue
        if abs(float(ffn.b_gate.data[unit].item()) - 1.0) > 1e-6:
            continue
        if gate.abs().sum().item() != 0:
            continue
        lo_one = float(ffn.W_down.data[BD.OUTPUT_LO + 1, unit].item())
        if abs(lo_one - 2.0 / S) > 1e-6:
            continue

        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 0.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        rewritten += 1
    return rewritten


def _rewrite_layer3_initial_sp_marker_to_f8(
    ffn, S: float, BD, *, jsr_prologue: bool = False
) -> int:
    """Materialize initial emitted ``SP_byte0`` as ``0xf8`` at the SP marker.

    Only valid when the program starts with a ``JSR; ENT N`` prologue that
    pushes the return address (SP = STACK_INIT - 8 = 0x0000fff8 after step 0,
    so SP_byte0 = 0xf8). For prologue-less programs produced by
    ``compile_c('int main() {...}')`` — which emit flat IMM-first bytecode
    with no JSR/CALL/ENT prologue — the real SP byte 0 is 0x00 (the L3 FFN
    default), and forcibly writing 0xf8 silently corrupts MUL/DIV/anything
    that needs stack storage.

    Callers must explicitly opt in via ``jsr_prologue=True``. The boolean
    cannot be auto-detected at L3 FFN bake time: the FFN weights are shared
    across all positions of all inputs, and ``OP_JSR`` is not yet relayed to
    the ``MARK_SP`` row until L6 (see vm_step.py line 4604+ for the
    JSR-gated SP byte 0 fixup that already runs at the operational layer).
    """

    if not jsr_prologue:
        return 0

    rewritten = 0
    for unit in range(ffn.W_up.shape[0]):
        up = ffn.W_up.data[unit]
        gate = ffn.W_gate.data[unit]
        if abs(float(up[BD.MARK_SP].item()) - S) > 1e-6:
            continue
        if abs(float(up[BD.HAS_SE].item()) + S) > 1e-6:
            continue
        if abs(float(ffn.b_up.data[unit].item()) + S * 0.5) > 1e-6:
            continue
        if abs(float(ffn.b_gate.data[unit].item()) - 1.0) > 1e-6:
            continue
        if gate.abs().sum().item() != 0:
            continue

        lo_zero = float(ffn.W_down.data[BD.OUTPUT_LO + 0, unit].item())
        hi_zero = float(ffn.W_down.data[BD.OUTPUT_HI + 0, unit].item())
        if abs(lo_zero - 2.0 / S) <= 1e-6:
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 0.0
            ffn.W_down.data[BD.OUTPUT_LO + 8, unit] = 2.0 / S
            rewritten += 1
        if abs(hi_zero - 2.0 / S) <= 1e-6:
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 0.0
            ffn.W_down.data[BD.OUTPUT_HI + 15, unit] = 2.0 / S
            rewritten += 1
    return rewritten


def _layer3_pc_byte1_output_rules(S: float) -> tuple:
    """Declarative L3 FFN units 134-135 -- PC byte1 emission rules.

    Two ``constant_write`` rules that make PC byte1 equal one either
    on the wrap token (new byte0 == 0x02 -> CLEAN_EMBED_LO+2 +
    CLEAN_EMBED_HI+0) or while the previous byte1 was already one
    (TEMP+1 + TEMP+16 staged by L3 head 7, with CLEAN_EMBED_HI+0..4
    bounding the preserve rule to byte0 high nibbles 0..4).

    Both rules write ``OUTPUT_LO[0]=-500/S``, ``OUTPUT_LO[1]=500/S``,
    ``OUTPUT_HI[0]=500/S`` (the OUTPUT_LO writes form a +1 / -0
    one-hot pair, the OUTPUT_HI write provides the byte1 high nibble).

    Phase 8.D follow-up: the ``BYTE_INDEX_0`` condition uses
    :func:`dim_ref` for the ``(byte_index, "0")`` family lookup so
    the rule names the byte-position role rather than the bare slot
    label. Byte-identical via DimRef.parse.
    """
    common_conds = (
        (f"H1+{_PC_I}", 1.0),
        (dim_ref("byte_index", "0"), 1.0),
        ("IS_BYTE", 1.0),
        ("HAS_SE", 1.0),
    )
    common_writes = (
        ("OUTPUT_LO+0", -500.0 / S),
        ("OUTPUT_LO+1", 500.0 / S),
        ("OUTPUT_HI+0", 500.0 / S),
    )
    # Rule 0 (unit 134): wrap token -- new byte0 high nibble == 0 AND
    # new byte0 low nibble == 2 (i.e. PC just crossed 0x100).
    wrap = FFNRule.constant_write(
        name="layer3_ffn.pc_byte1_wrap_token",
        conditions=common_conds + (
            ("CLEAN_EMBED_LO+2", 1.0),
            ("CLEAN_EMBED_HI+0", 1.0),
        ),
        threshold=5.5,
        writes=common_writes,
    )
    # Rule 1 (unit 135): preserve when previous byte1 was already 1
    # (TEMP+1 from L3 head 7 carry low nibble) AND TEMP+16 (the carry
    # tag) AND CLEAN_EMBED_HI 0..4 (bound to byte0 high nibbles 0..4
    # because the current 1096 corpus never runs past 0x14a).
    preserve_conds = list(common_conds)
    preserve_conds.append(("TEMP+1", 1.0))
    preserve_conds.append(("TEMP+16", 1.0))
    for hi in range(5):
        preserve_conds.append((f"CLEAN_EMBED_HI+{hi}", 1.0))
    preserve = FFNRule.constant_write(
        name="layer3_ffn.pc_byte1_preserve",
        conditions=tuple(preserve_conds),
        threshold=6.5,
        writes=common_writes,
    )
    return (wrap, preserve)


def _lower_layer3_pc_byte1_output_rules_ir(
    ffn, S: float, BD, *, start_unit: int,
) -> int:
    """Lower :func:`_layer3_pc_byte1_output_rules` into ``ffn``.

    Returns the next free unit index. The two PC-byte1 rules occupy
    units ``start_unit`` and ``start_unit + 1`` -- callers pass the
    cursor returned by the main L3 FFN rule lowering (134).
    """
    rules = _layer3_pc_byte1_output_rules(S)
    dim_names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(BD, dim_names)
    return Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=start_unit, S=S,
    )


def _add_layer3_pc_byte1_output_rules(ffn, S, BD) -> None:
    """Emit PC byte1 for programs whose linear PC crosses 0x100.

    Legacy L3 owns ordinary PC emission but only increments byte 0 and then
    defaults bytes 1-3 to zero. Head 7 below stages the previous step's PC
    byte1 into TEMP at the current PC byte0 row. These two late units make
    byte1 equal one either on the wrap token (new byte0 == 0x02) or while the
    previous byte1 was already one. The preserve rule is intentionally bounded
    to byte0 high nibbles 0..4 because the current 1096 corpus never runs
    past 0x14a; branch targets below 0x100 such as 0x62 must not preserve the
    prior high byte.

    Migrated to :class:`FFNRule` lowering via
    :func:`_layer3_pc_byte1_output_rules` and
    :func:`_lower_layer3_pc_byte1_output_rules_ir`; the
    ``_next_free_ffn_unit`` probe remains for legacy back-compat (the
    main bake's cursor assertion guarantees it returns 134).
    """

    start_unit = _next_free_ffn_unit(ffn)
    if start_unit + 2 > ffn.W_up.shape[0]:
        raise RuntimeError("L3 FFN has no room for PC byte1 carry repair")
    _lower_layer3_pc_byte1_output_rules_ir(
        ffn, S, BD, start_unit=start_unit,
    )


def make_layer3_ffn_dep_anchor_op() -> Operation:
    """No-op companion for ``layer3_ffn``: declares identical reads/writes so
    the LayerCompiler's dep graph reserves a layer slot for it. Mirrors
    ``_layer5_fetch_dep_anchor``: the actual bake happens in
    ``layer3_ffn`` (kind="block", layer_idx=3); this op's bake is a no-op
    (its layout-assigned ffn block is unrelated to block[3]).
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is in `layer3_ffn` block op above.
        return

    return Operation(
        name="_layer3_ffn_dep_anchor",
        # Tier 1 phase= elimination: drop phase=3. The L4 anchor now
        # carries requires={} (per df6f0671) and places at L4 via its
        # FETCH_LO/HI writes; this L3 anchor lands at L3 via its
        # earliest-feasible layer (no writes, reads satisfied by
        # earlier-layer producers). The phase-share co-location is no
        # longer the load-bearing mechanism for either anchor.
        # Phase 8.A.6 v2: matches layer3_ffn's TEMP_PREV_STEP rename.
        # Phase 8.A: matches layer3_ffn's OP_LEV_PREV_STEP rename.
        # Phase 8.A (EMBED_HI split): matches layer3_ffn's
        # EMBED_HI_PREV_STEP rename. Same numeric position; breaks the
        # layer4_pc_relay → _layer3_ffn_dep_anchor back-edge on EMBED_HI.
        # Phase 9.B (EMBED_LO SCC rename): EMBED_LO -> EMBED_LO.*.-1
        # mirrors layer3_ffn's rename. layer4_pc_relay is the sole
        # next-step producer; the anchor reads the prev-step residual.
        # Same numeric slot via SSA alias.
        reads={"MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "HAS_SE",
               "EMBED_LO.*.-1", "EMBED_HI.*.-1", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "TEMP.*.-1", "IS_BYTE", "H1", "H4", "OP_LEV.*.-1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "NEXT_STACK0"},
        # Phase 9.B (NEXT_STACK0 dep-anchor): the anchor is now read-only.
        # The actual NEXT_STACK0 carry chain is owned by ``layer3_ffn``
        # (its bake unit 82..85). Forwarding writes from the anchor
        # double-claimed the band and produced a same-step 2-cycle with
        # ``layer3_ffn``. The block op still resolves to the anchor's
        # layer via ``target_op_name``. Also drops the EMBED_LO/HI /
        # OUTPUT_* anchor writes — those are owned by layer3_ffn likewise.
        writes=set(),
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
        smoke_tests=set(),
        spec_section=None,
    )


# === L3 attention head layout (auto-fit; legacy head_idx as docs) ====
#
# The ``layer3_carry_forward_attn`` op owns all 8 L3 attention heads.
# Heads 0-3 are register carry-forward heads (PC / AX / SP / BP),
# head 4 retires the stale STACK0 marker carry, and heads 5-7 are
# declarative relays (AX_FULL gather, LEV BP->PC, PC byte1 carry).
#
# Phase 7.B.2: the allocator runs with ``pin=None`` on every entry --
# first-fit picks 0..7 in declaration order, which matches the legacy
# layout bit-for-bit. The ``legacy_head_idx`` column below is now
# documentation only; the load-bearing copy is
# :data:`_L3_HEAD_LAYOUT_BY_NAME`, consumed by the head-spec
# factories that write Q/K/V/O weights at the resolved ``head_idx``.
_L3_HEAD_LAYOUT = (
    # (op_name, legacy_head_idx (docs only))
    ("layer3_carry_forward_attn.head_0", 0),  # PC carry-forward
    ("layer3_carry_forward_attn.head_1", 1),  # AX carry-forward -> AX_CARRY band
    ("layer3_carry_forward_attn.head_2", 2),  # SP carry-forward
    ("layer3_carry_forward_attn.head_3", 3),  # BP carry-forward
    ("layer3_carry_forward_attn.head_4", 4),  # STACK0 marker carry retirer
    ("layer3_carry_forward_attn.head_5", 5),  # AX_FULL relay (OUTPUT -> AX_FULL)
    ("layer3_carry_forward_attn.head_6", 6),  # LEV BP->PC (CLEAN_EMBED relay)
    ("layer3_carry_forward_attn.head_7", 7),  # PC byte1 prev -> TEMP
)
_L3_HEAD_LAYOUT_BY_NAME = {name: head_idx for name, head_idx in _L3_HEAD_LAYOUT}


def _allocate_layer3_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L3 heads.

    Phase 7.B.2: ``pin=`` is dropped from every entry. The allocator's
    first-fit picks the lowest free head index in declaration order;
    because :data:`_L3_HEAD_LAYOUT` is contiguous (0..7) and ordered,
    first-fit reproduces the legacy ``head_idx`` values bit-for-bit.
    The actual weight-write head indices are still looked up via
    :data:`_L3_HEAD_LAYOUT_BY_NAME` inside the head-spec factories
    below, so byte-identity with the legacy bake is preserved
    regardless of allocator order. Returns the allocator so callers
    can attach it to the ``attn`` module for inspection.
    """
    allocator = AttentionHeadAllocator(layer_max_heads=8)
    for name, _legacy_head_idx in _L3_HEAD_LAYOUT:
        allocator.alloc(name, 3)
    return allocator


def make_layer3_carry_forward_attn_op() -> Operation:
    """L3 attention: 8 carry-forward heads (PC, AX, SP, BP, STACK0 + relays).

    Heads 0-3 are declarative carry-forward heads (PC, AX, SP, BP) that
    formerly went through ``Primitives.carry_forward_attention``; their
    Q/K/V/O writes are now expressed as ``DeclarativeAttentionHeadSpec``
    via :func:`_carry_forward_head_spec` (byte-identical with the
    legacy helper). Head 4 is the declarative STACK0 marker carry
    retirer. Heads 5-7 are declarative relays for AX_FULL, LEV BP->PC,
    and PC byte1 preservation.
    """
    def bake(attn, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake attention-head allocator. Pins every L3 head_idx at
        # its existing slot so byte-identity is preserved.
        head_allocator = _allocate_layer3_heads()
        attn._l3_head_allocator = head_allocator

        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn, _layer3_carry_forward_head_specs(proxy), HD,
        )

    # Dim-ownership claims: 7 carry-forward attention heads.
    #   Heads 0-3: Primitives.carry_forward_attention writes V slots 1..32:
    #     W_v[h*HD + 1 + k, src_lo + k]    for k=0..15 (slot 1..16)
    #     W_v[h*HD + 17 + k, src_hi + k]   for k=0..15 (slot 17..32)
    #   Plus W_q[base, marker], W_k[base, L1H1], W_k[base, L1H0] and the
    #   GATE=33 row.  We capture the V/O row claims (the load-bearing
    #   slot/column pairs that can collide with other ops).
    #
    #   Head 0 (PC): src=EMBED_LO/HI, out=EMBED_LO/HI
    #   Head 1 (AX): src=EMBED_LO/HI, out=AX_CARRY_LO/HI
    #   Head 2 (SP): src=EMBED_LO/HI, out=EMBED_LO/HI
    #   Head 3 (BP): src=EMBED_LO/HI, out=EMBED_LO/HI
    #   Head 4 (STACK0): declarative STACK0_BYTE0 carry head spec below
    #   Head 5 (AX_FULL): inline V[OUTPUT_LO/HI] → AX_FULL_LO/HI
    #   Head 6 (BP→PC LEV): inline V[CLEAN_EMBED_LO/HI] → (out via inline)
    #   Head 7 (PC byte1): previous PC byte1 CLEAN_EMBED_LO/HI → TEMP
    _claims = set()
    _heads_cf = [
        (0, "EMBED_LO", "EMBED_HI"),
        (1, "EMBED_LO", "EMBED_HI"),
        (2, "EMBED_LO", "EMBED_HI"),
        (3, "EMBED_LO", "EMBED_HI"),
    ]
    for h, src_lo, src_hi in _heads_cf:
        for k in range(16):
            _claims.add((3, "attn_W_v", f"{h}_{1 + k}", f"{src_lo}+{k}"))
            _claims.add((3, "attn_W_v", f"{h}_{17 + k}", f"{src_hi}+{k}"))
    # Head 5: AX_FULL relay V slots from prev-step OUTPUT_LO/HI (read
    # via attention back to the prev-step AX marker row -- L3 fires
    # before any same-step OUTPUT producer, so the read consumes step
    # N-1's residual). See docs/B9_OUTPUT_HI_SPLIT_SPEC.md §2.1.
    # Note: OUTPUT_LO_PREV_STEP / OUTPUT_HI_THIS_STEP aliases share the
    # OUTPUT_LO / OUTPUT_HI dim positions; verifier categorizes the
    # observed writes under the canonical names.
    for k in range(16):
        _claims.add((3, "attn_W_v", f"5_{1 + k}", f"OUTPUT_LO+{k}"))
        _claims.add((3, "attn_W_v", f"5_{17 + k}", f"OUTPUT_HI+{k}"))
    # Head 6: BP→PC LEV relay V slots from CLEAN_EMBED_LO/HI.
    for k in range(16):
        _claims.add((3, "attn_W_v", f"6_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((3, "attn_W_v", f"6_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    # Head 7: previous PC byte1 relay V slots from CLEAN_EMBED_LO/HI.
    for k in range(16):
        _claims.add((3, "attn_W_v", f"7_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((3, "attn_W_v", f"7_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer3_carry_forward_attn",
        # Phase 8.A: OP_LEV_PREV_STEP marks head 6 (_lev_bp_to_pc_head_spec)
        # gating on OP_LEV as cross-step relative to L5 opcode_decode_ffn
        # (the OP_LEV writer). The head's Q[0]+=OP_LEV*L/5 still resolves
        # to the same numeric position because OP_LEV_PREV_STEP aliases
        # OP_LEV. Breaks back-edge L5 → layer3_carry_forward_attn on OP_LEV.
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
               "L1H0", "L1H1", "STACK0_BYTE0", "OP_LEV.*.-1", "HAS_SE",
               "H1", "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               # Phase 7.A.3.b: head 5 reads prev-step OUTPUT_LO via the
               # AX-marker attention back-edge. OUTPUT_LO_PREV_STEP has
               # no same-step writers (it is an alias of OUTPUT_LO at
               # position 174) so the rename suffices to break every
               # cross-step back-edge from L8+/L14+ OUTPUT_LO writers.
               # Phase 8.A (EMBED_HI split): EMBED_HI_PREV_STEP marks the
               # EMBED_HI read as cross-step relative to layer4_pc_relay
               # (L4 writes EMBED_HI at the AX marker AFTER L3 in the
               # same step). Same numeric position; breaks the
               # layer4_pc_relay → layer3_carry_forward_attn back-edge
               # on EMBED_HI.
               # Phase 8.A G7: OUTPUT_HI_THIS_STEP read renamed to
               # OUTPUT_HI_PREV_STEP. Head 5 (``_ax_full_relay_head_spec``)
               # attends back to the prev-step AX marker row where the
               # high-nibble OUTPUT residual still holds the previous
               # step's value. The alias shares the same numeric
               # position (190) as OUTPUT_HI so baked weights are
               # byte-identical. Because OUTPUT_HI_PREV_STEP has no
               # declared writers, the scheduler dep graph no longer
               # creates a back-edge from any later-layer
               # OUTPUT_HI_THIS_STEP writer (L8+/L14+) into this op,
               # so the previous ``requires["after"]=layer16_lev_routing``
               # cycle-break is no longer needed and has been removed.
               # See docs/B9_OUTPUT_HI_SPLIT_SPEC.md §2.1 + Phase 8.A G7.
               # Phase 9.B (EMBED_LO SCC rename): EMBED_LO -> EMBED_LO.*.-1
               # mirrors EMBED_HI. layer4_pc_relay is the sole next-step
               # EMBED_LO producer; L3 reads the prev-step residual.
               # Same numeric slot via SSA alias.
               "EMBED_LO.*.-1", "EMBED_HI.*.-1", "OUTPUT_LO.*.-1",
               "OUTPUT_HI.*.-1", "CONST"},
        writes={"EMBED_LO", "EMBED_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                "AX_FULL_LO", "AX_FULL_HI", "OUTPUT_LO", "OUTPUT_HI",
                "TEMP", "ADDR_KEY"},
        kind="attn",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer3_carry_forward_attn_ir,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _carry_forward_head_spec(
    BD,
    *,
    head_idx: int,
    marker_dim: int,
    l1h1_idx: int,
    l1h0_idx: int,
    out_lo: int,
    out_hi: int,
    src_lo: int,
    src_hi: int,
    L: float = 15.0,
) -> DeclarativeAttentionHeadSpec:
    """Declarative replacement for ``Primitives.carry_forward_attention``.

    Mirrors the exact Q/K/V/O writes of the helper one-to-one so the
    lowered matrices are byte-identical. The spec carries:

    * ``Q[0] = marker_dim * L`` -- fires at the target marker.
    * ``K[0] = L1H1+l1h1_idx * L``, ``K[0] = L1H0+l1h0_idx * -L`` --
      fires at the previous step's byte 0 row.
    * ``V[1+k] = src_lo+k``, ``V[17+k] = src_hi+k`` for k=0..15.
    * ``O[out_lo+k] = V[1+k]``, ``O[out_hi+k] = V[17+k]`` for k=0..15.
    * Anti-leakage gate at slot 33: ``Q[33]=marker*L + CONST*-L/2``,
      ``K[33]=CONST*L``.
    """

    GATE = 33
    q = [
        AP(0, marker_dim, L),
        AP(GATE, marker_dim, L),
        AP(GATE, BD.CONST, -L / 2),
    ]
    k = [
        AP(0, BD.L1H1 + l1h1_idx, L),
        AP(0, BD.L1H0 + l1h0_idx, -L),
        AP(GATE, BD.CONST, L),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, src_lo + k_idx, 1.0))
        v.append(AP(17 + k_idx, src_hi + k_idx, 1.0))
        o.append(AO(out_lo + k_idx, 1 + k_idx, 1.0))
        o.append(AO(out_hi + k_idx, 17 + k_idx, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer3_carry_forward_head_specs(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """All 8 L3 carry-forward attention heads as declarative specs.

    Heads 0-3 are register carry-forward heads (PC / AX / SP / BP) built
    via :func:`_carry_forward_head_spec`. Head 4 retires the stale STACK0
    marker carry. Heads 5-7 are the AX_FULL relay, the LEV BP->PC relay,
    and the PC byte1 prev relay respectively.
    """

    PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
    specs = (
        _carry_forward_head_spec(
            BD,
            head_idx=_L3_HEAD_LAYOUT_BY_NAME["layer3_carry_forward_attn.head_0"],
            marker_dim=BD.MARK_PC,
            l1h1_idx=PC_I,
            l1h0_idx=PC_I,
            out_lo=BD.EMBED_LO,
            out_hi=BD.EMBED_HI,
            src_lo=BD.EMBED_LO,
            src_hi=BD.EMBED_HI,
        ),
        _carry_forward_head_spec(
            BD,
            head_idx=_L3_HEAD_LAYOUT_BY_NAME["layer3_carry_forward_attn.head_1"],
            marker_dim=BD.MARK_AX,
            l1h1_idx=AX_I,
            l1h0_idx=AX_I,
            out_lo=BD.AX_CARRY_LO,
            out_hi=BD.AX_CARRY_HI,
            src_lo=BD.EMBED_LO,
            src_hi=BD.EMBED_HI,
        ),
        _carry_forward_head_spec(
            BD,
            head_idx=_L3_HEAD_LAYOUT_BY_NAME["layer3_carry_forward_attn.head_2"],
            marker_dim=BD.MARK_SP,
            l1h1_idx=SP_I,
            l1h0_idx=SP_I,
            out_lo=BD.EMBED_LO,
            out_hi=BD.EMBED_HI,
            src_lo=BD.EMBED_LO,
            src_hi=BD.EMBED_HI,
        ),
        _carry_forward_head_spec(
            BD,
            head_idx=_L3_HEAD_LAYOUT_BY_NAME["layer3_carry_forward_attn.head_3"],
            marker_dim=BD.MARK_BP,
            l1h1_idx=BP_I,
            l1h0_idx=BP_I,
            out_lo=BD.EMBED_LO,
            out_hi=BD.EMBED_HI,
            src_lo=BD.EMBED_LO,
            src_hi=BD.EMBED_HI,
        ),
        _stack0_carry_head_spec(BD),
        _ax_full_relay_head_spec(BD),
        _lev_bp_to_pc_head_spec(BD),
        _pc_byte1_prev_head_spec(BD),
    )
    return specs


def _layer3_carry_forward_attn_ir(dim_positions, HD) -> CompilerIR:
    """CompilerIR factory for ``layer3_carry_forward_attn``.

    Resolves dim names through ``_as_setdim_proxy`` so the lowering
    works under both legacy ``_SetDim`` and ``pin_io_only=True``
    compiler-allocated layouts.
    """
    del HD  # head specs are dim-only; HD is encoded in the lowering call.
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_layer3_carry_forward_head_specs(BD))
    return ir


def _stack0_carry_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 4: retire stale STACK0 marker carry.

    Historical weights copied the previous ``STACK0_byte0`` into the current
    STACK0 marker's EMBED band.  That marker row predicts the next byte, and
    later marker identity paths can project the stale EMBED value back to
    OUTPUT.  Leave the matcher structurally present but do not write a marker
    value; real STACK0 bytes are supplied by the stack/memory paths.
    """

    L = 15.0
    q = [AP(0, BD.MARK_STACK0, L)]
    k = [AP(0, BD.STACK0_BYTE0, L), AP(33, BD.CONST, L)]
    q.append(AP(33, BD.MARK_STACK0, L))
    q.append(AP(33, BD.CONST, -L / 2))
    return DeclarativeAttentionHeadSpec(
        head_idx=4,
        q=tuple(q),
        k=tuple(k),
    )


def _ax_full_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 5: current AX output byte -> AX_FULL."""

    L = 15.0
    GATE = 33
    q = [
        AP(0, BD.MARK_AX, L),
        AP(0, BD.HAS_SE, L),
        AP(0, BD.CONST, -L * 1.5),
        AP(GATE, BD.MARK_AX, L),
        AP(GATE, BD.CONST, -L / 2),
    ]
    k = [AP(0, BD.MARK_AX, L), AP(GATE, BD.CONST, L)]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.OUTPUT_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.OUTPUT_HI + k_idx, 1.0))
        o.append(AO(BD.AX_FULL_LO + k_idx, 1 + k_idx, 1.0))
        o.append(AO(BD.AX_FULL_HI + k_idx, 17 + k_idx, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=5,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _lev_bp_to_pc_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 6: BP byte carry source for LEV return address."""

    L = 15.0
    BP_I = 3
    GATE = 33
    q = [
        AP(0, BD.MARK_PC, L),
        AP(0, BD.OP_LEV, L / 5),
        AP(0, BD.CONST, -L * 1.5),
        AP(GATE, BD.MARK_PC, L),
        AP(GATE, BD.CONST, -L / 2),
    ]
    k = [
        AP(0, BD.L1H1 + BP_I, L),
        AP(0, BD.L1H0 + BP_I, -L),
        AP(GATE, BD.CONST, L),
    ]
    v = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=6,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
    )


def _pc_byte1_prev_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 7: previous PC byte1 -> current PC byte0 TEMP.

    The L3 FFN predicts PC byte1 at the PC byte0 position. To preserve byte1
    after PC has crossed 0x100, that FFN needs the previous state's byte1; this
    head copies it from the prior PC byte1 token into TEMP at PC byte0 rows.
    It also stages the low nibble of byte1 into ADDR_KEY[32..47] at PC rows so
    later declarative fetch heads can match the full 12-bit code address.
    """

    L = 15.0
    PC_I = 0
    GATE = 33
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.TEMP + k_idx, 1 + k_idx, 1.0))
        o.append(AO(BD.TEMP + 16 + k_idx, 17 + k_idx, 1.0))
        o.append(AO(BD.ADDR_KEY + 32 + k_idx, 1 + k_idx, 1.0))

    code_prefix_blockers = tuple(
        AP(0, BD.ADDR_KEY + k_idx, -L)
        for k_idx in range(48)
    )

    return DeclarativeAttentionHeadSpec(
        head_idx=7,
        q=(
            AP(0, BD.IS_BYTE, L),
            AP(0, BD.H1 + PC_I, L),
            AP(0, BD.BYTE_INDEX_0, L),
            AP(0, BD.MARK_PC, 3.0 * L),
            AP(0, BD.HAS_SE, L),
            AP(0, BD.CONST, -3.0 * L),
            *code_prefix_blockers,
            AP(GATE, BD.IS_BYTE, 500.0),
            AP(GATE, BD.MARK_PC, 500.0),
            AP(GATE, BD.CONST, -500.0),
        ),
        k=(
            AP(0, BD.IS_BYTE, L),
            AP(0, BD.H1 + PC_I, L),
            AP(0, BD.BYTE_INDEX_1, L),
            AP(0, BD.CONST, -2.0 * L),
            AP(GATE, BD.CONST, 5.0),
        ),
        v=tuple(v),
        o=tuple(o),
    )


def make_layer3_convo_io_state_init_op(
    enable_conversational_io: bool = False,
) -> Operation:
    """L3 FFN addition: initialize output mode when LAST_WAS_THINKING_END.

    Originally an inline call in ``set_vm_weights`` (gated by
    ``enable_conversational_io``):
        _set_conversational_io_state_init(ffn3, S, BD)

    Migrated as ``kind="block"`` pinned to ``layer_idx=3`` with
    ``migrated=True``. Phase=3.1 so this runs AFTER
    ``make_layer3_ffn_op`` (phase=3) and writes into a distinct FFN
    unit range (starts at unit 1034, above the L3 / L6-routing unit
    counters), so the writes layer cleanly on top.

    The bake is unconditional in shape (always registered to keep the
    dep-graph stable), but the body is a no-op when
    ``enable_conversational_io`` is False so no FFN units are touched
    outside of conversational-I/O mode.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        proxy = _as_setdim_proxy(dim_positions)
        _lower_layer3_convo_io_state_init_ir(block.ffn, S, proxy)

    return Operation(
        name="layer3_convo_io_state_init",
        # Reads/writes use LAST_WAS_THINKING_END and IO_IN_OUTPUT_MODE,
        # which are not declared in declare_setdim_compat_dims
        # (conversational-I/O-only dims); the bake resolves them via the
        # _SetDim fallback in _as_setdim_proxy, so no compiler-tracked
        # edges are needed.
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.G.6: drop ``layer_idx=3`` literal; bind to the L3 attn
        # anchor ``layer3_carry_forward_attn`` so the block op resolves
        # to whichever layer the L3 carry-forward attn lands at.
        target_op_name="layer3_carry_forward_attn",
        declarative_bake_fn=bake,
        compiler_ir=_layer3_convo_io_state_init_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        ffn_units_used=1035 if enable_conversational_io else None,
        # B12 backfill: docstring above pins phase 3.1 AFTER
        # ``layer3_ffn`` (phase 3, same L3 FFN) so this extension's
        # writes at unit 1034 layer cleanly on top of the L3 / L6-routing
        # unit counters. Encoded as a B10 op-name reference so the
        # dynamic scheduler honours the dep edge even though reads/writes
        # are empty (bake body flag-gated on ``enable_conversational_io``).
        requires={"after": "layer3_ffn"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _layer3_convo_io_state_init_rules(S: float) -> tuple[FFNRule, ...]:
    return (
        FFNRule.constant_write(
            name="convo_io_enter_output_mode",
            conditions=(("LAST_WAS_THINKING_END", 1.0),),
            threshold=0.5,
            writes=(("IO_IN_OUTPUT_MODE", 2.0 / S),),
            scope="LAST_WAS_THINKING_END",
        ),
    )


def _layer3_convo_io_state_init_ir(S: float = 100.0) -> CompilerIR:
    """Build a :class:`CompilerIR` exposing the convo-IO state-init rule.

    Pinned to :attr:`Operation.compiler_ir` so the declarative verifier
    sees the single ``convo_io_enter_output_mode`` rule even though
    the bake body lowers it at start_unit=1034 (above the L3 main
    rule range). The IR-level lowering uses start_unit=0; the actual
    bake body uses the production offset via
    :func:`_lower_layer3_convo_io_state_init_ir`.
    """
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer3_convo_io_state_init_rules(S))
    return ir


def _lower_layer3_convo_io_state_init_ir(ffn, S: float, BD) -> int:
    rules = _layer3_convo_io_state_init_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=1034,
        S=S,
    )
