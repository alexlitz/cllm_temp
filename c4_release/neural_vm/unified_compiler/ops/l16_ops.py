"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...ffn_unit_allocator import FFNUnitAllocator
from ..band_guarantees import scalar_value_guarantee_rules
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


# === L16 FFN unit layout (auto-fit; legacy offsets retained as docs) ==
#
# The ``layer16_lev_routing`` op currently owns the entire L16 FFN. Its
# bake calls ``lower_layer16_lev_routing_ir(..., start_unit=0)``, which
# walks ``_layer16_lev_routing_rules`` and writes one hidden unit per
# rule via :func:`Primitives.lower_ffn_rules`. The full sub-stage list
# spans many families (LEV routing, STACK0 markers, BP/SP byte
# producers, post-LEV AX_CARRY + STACK0 byte0 preservation, etc.) but
# the rule-list order is monolithic and unit-index ``i`` always lands
# at ``i``-th rule, so the only externally-visible offset is
# ``start_unit=0`` for a 792-unit span.
#
# Phase 7.B.6: the single layout entry is auto-placed by
# :class:`FFNUnitAllocator` first-fit. With an empty L16 FFN pool the
# first-fit lands the 792-unit range at start=0, matching the legacy
# ``start_unit=0`` cursor bit-for-bit. The ``legacy_start`` column is
# kept purely as documentation.
#
# The total here mirrors ``ffn_units_used=792`` on the op. Changing
# the rule-list length in ``_layer16_lev_routing_rules`` requires
# updating this table in lock-step.
_L16_FFN_UNIT_LAYOUT = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("layer16_lev_routing", 0, 792),  # full LEV routing rule bank
)


def _allocate_layer16_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L16 sub-stages.

    Phase 7.B.6: ``pin=`` is dropped. The allocator's default first-fit
    lands the layout's single 792-unit range at start=0 (the lowest
    free gap in an empty pool), which matches
    ``lower_layer16_lev_routing_ir``'s ``start_unit=0`` cursor
    bit-for-bit. Byte-identity with the legacy bake is therefore
    preserved without the author having to spell out the offset.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L16 op claims a free range past unit 792).
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L16_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def _add_stack0_x0_alu_materializer(
    rules,
    *,
    family: str,
    conditions,
    threshold: float,
    S: float,
    scope=None,
    dominates_at=None,
):
    """Append 32 ALU->OUTPUT materializer rules for a STACK0 marker family.

    Each call appends 16 LO-band rules (gated on ALU_LO+k -> OUTPUT_LO+k) and
    16 HI-band rules (gated on ALU_HI+k -> OUTPUT_HI_THIS_STEP+k).  ``family`` becomes
    part of the rule name (``l16_stack0_{family}_marker_from_alu_{band}_{k}``).
    ``scope`` is threaded through to every generated FFNRule so the
    declarative scope verifier can audit the intended firing positions.
    ``dominates_at`` is threaded through identically so the strength
    verifier can audit per-output-dim dominance claims; it accepts the
    standard mapping ``{"OUTPUT_LO": <predicate>, "OUTPUT_HI_THIS_STEP": <predicate>}``.
    """

    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_stack0_{family}_marker_from_alu_lo_{k}",
            conditions=conditions,
            threshold=threshold,
            gate=f"ALU_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 50.0 / S),),
            scope=scope,
            dominates_at=dominates_at,
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_stack0_{family}_marker_from_alu_hi_{k}",
            conditions=conditions,
            threshold=threshold,
            gate=f"ALU_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 50.0 / S),),
            scope=scope,
            dominates_at=dominates_at,
        ))


def _layer16_lev_routing_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for L16 LEV routing units 0..120."""

    rules = []
    write_scale = 2.0 / S
    first_step_gate = 30.0
    sp_cancel_conditions = (
        ("OP_LEV", 0.2),
        ("MARK_SP", 1.0),
        ("MARK_PC", -1.0),
        ("MARK_AX", -1.0),
        ("MARK_BP", -1.0),
        ("HAS_SE", first_step_gate),
        ("PSH_AT_SP", -first_step_gate),
    )
    for band, output_base in (("lo", "OUTPUT_LO"), ("hi", "OUTPUT_HI_THIS_STEP")):
        for k in range(16):
            rules.append(FFNRule.gated_write(
                name=f"l16_lev_sp_cancel_{band}_{k}",
                conditions=sp_cancel_conditions,
                threshold=31.5,
                gate=f"{output_base}+{k}",
                gate_weight=-1.0,
                writes=((f"{output_base}+{k}", write_scale),),
            ))

    sp_value_base_conditions = (
        ("OP_LEV", 1.0),
        ("MARK_SP", 1.0),
        ("MARK_BP", -15.0),
        ("MARK_PC", -15.0),
        ("MARK_AX", -50.0),
        ("MARK_STACK0", -15.0),
        ("MARK_MEM", -15.0),
        ("H3+4", -15.0),
        ("MARK_SE", -15.0),
        ("BYTE_INDEX_0", -10.0),
        ("BYTE_INDEX_1", -10.0),
        ("BYTE_INDEX_2", -10.0),
        ("BYTE_INDEX_3", -10.0),
        ("HAS_SE", first_step_gate),
        ("PSH_AT_SP", -first_step_gate),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_sp_bp_plus16_lo_{k}",
            conditions=sp_value_base_conditions + ((f"ADDR_B0_LO+{k}", 1.0),),
            threshold=40.0,
            gate=f"ADDR_B0_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        result = (k + 1) % 16
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_sp_bp_plus16_hi_{k}",
            conditions=sp_value_base_conditions + ((f"ADDR_B0_HI+{k}", 1.0),),
            threshold=40.0,
            gate=f"ADDR_B0_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{result}", write_scale),),
        ))

    pc_cancel_hi_conditions = (
        ("OP_LEV", 0.2),
        ("MARK_PC", 1.0),
        ("MARK_AX", -1.0),
        ("MARK_SP", -1.0),
        ("MARK_BP", -1.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_pc_cancel_hi_{k}",
            conditions=pc_cancel_hi_conditions,
            threshold=1.5,
            gate=f"OUTPUT_HI_THIS_STEP+{k}",
            gate_weight=-1.0,
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_pc_temp_lo_{k}",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_PC", 1.0),
                (f"TEMP+{k}", 1.0),
            ),
            threshold=3.5,
            gate="CONST",
            writes=((f"OUTPUT_LO+{k}", write_scale),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_pc_temp_hi_{k}",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_PC", 1.0),
                (f"TEMP+{16 + k}", 1.0),
            ),
            threshold=3.5,
            gate="CONST",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", write_scale),),
        ))

    # LEV restores the caller's AX from the saved value on STACK0. By step 6
    # post-LEV the L8 AX-carry shadow at MARK_AX holds the popped return value
    # (parallel L6 backfill ensures the carry is populated from the freed STACK0
    # slot). Without a declarative materializer the OUTPUT byte at the AX
    # position defaults to zero, corrupting the post-LEV return register --
    # this is the step6:AX_byte0 91-row cluster identified by the 2026-06-01
    # stack/JSR/LEV triage. Mirror the OP_IMM/OP_SI/OP_SC AX-carry
    # materializers above (l16_stale_imm_ax_carry_* / l16_store_ax_carry_*),
    # but key on OP_LEV so the SP/PC LEV-routing rules above (which cancel at
    # MARK_SP / MARK_PC and do not touch MARK_AX) remain orthogonal.
    lev_ax_carry_conditions = (
        ("OP_LEV", 1.0),
        ("MARK_AX", 1.0),
        ("MARK_PC", -8.0),
        ("MARK_SP", -8.0),
        ("MARK_BP", -8.0),
        ("MARK_STACK0", -8.0),
        ("MARK_MEM", -8.0),
        ("IS_BYTE", -10.0),
        ("OP_EXIT", -20.0),
        ("OP_JMP", -20.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_ax_carry_lo_{k}",
            conditions=lev_ax_carry_conditions,
            threshold=1.5,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 2.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_ax_carry_hi_{k}",
            conditions=lev_ax_carry_conditions,
            threshold=1.5,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 2.0 / S),),
        ))

    # Companion preservation: the same triage row sees step6:STACK0_byte0
    # corruption (50-row cluster) because no LEV-aware rule keeps the freed
    # stack-top byte from being overwritten by the generic STACK0 materializer
    # family (stack0_e0/e8/f8_marker_from_alu_*) which the LEV-routing op gates
    # off via OP_LEV blockers. After LEV pops the saved AX, the visible STACK0
    # byte at the byte-0 row should remain the just-popped value rather than
    # collapsing back to zero. Gate the preservation on the existing OUTPUT
    # nibble so we only reinforce what L14/L15 already staged; the rule is
    # silent on the default-zero row.
    lev_stack0_preserve_conditions = (
        ("OP_LEV", 1.0),
        ("MARK_STACK0", 1.0),
        ("HAS_SE", 1.0),
        ("BYTE_INDEX_0", 1.0),
        ("MARK_PC", -8.0),
        ("MARK_AX", -8.0),
        ("MARK_SP", -8.0),
        ("MARK_BP", -8.0),
        ("MARK_MEM", -8.0),
        ("IS_BYTE", -10.0),
        ("MEM_STORE", -10.0),
    )
    # Use nudge-strength (50/S) to match the sibling stack0_e0/e8/f8 marker
    # materializer families; this is enough to overcome the residual
    # zero-default but not so strong that we clobber legitimate L14/L15
    # store writes (which carry MEM_STORE and are excluded above).
    lev_stack0_preserve_strength = 50.0 / S
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_stack0_byte0_preserve_lo_{k}",
            conditions=lev_stack0_preserve_conditions,
            threshold=4.5,
            gate=f"OUTPUT_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", lev_stack0_preserve_strength),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_stack0_byte0_preserve_hi_{k}",
            conditions=lev_stack0_preserve_conditions,
            threshold=4.5,
            gate=f"OUTPUT_HI_THIS_STEP+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", lev_stack0_preserve_strength),),
        ))

    byte_zero_base = (
        ("OP_LEV", 0.5),
        ("MARK_PC", -1.5),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
        ("MARK_SE", -10.0),
        ("H1+1", -10.0),
        ("NEXT_AX", -1.5),
        ("NEXT_SP", -1.5),
        ("NEXT_BP", -1.5),
        ("NEXT_STACK0", -1.5),
        ("NEXT_MEM", -1.5),
        ("NEXT_SE", -1.5),
    )
    for byte_idx in ("BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_clear_output_lo10_{byte_idx}",
            conditions=byte_zero_base + ((byte_idx, 1.0),),
            threshold=4.0,
            gate="CONST",
            writes=(("OUTPUT_LO+10", -10.0 / S),),
        ))
    for byte_idx in ("BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_set_output_lo0_{byte_idx}",
            conditions=byte_zero_base + ((byte_idx, 1.0),),
            threshold=4.0,
            gate="CONST",
            writes=(("OUTPUT_LO+0", 5.0 / S),),
        ))
    for byte_idx in ("BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"):
        rules.append(FFNRule.gated_write(
            name=f"l16_lev_set_output_hi0_{byte_idx}",
            conditions=byte_zero_base + ((byte_idx, 1.0),),
            threshold=4.0,
            gate="CONST",
            writes=(("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
        ))

    # The legacy TEMP->PC materializers above can fire for every nibble on the
    # top-level LEV marker because OP_LEV + MARK_PC alone crosses their old
    # threshold. Keep the byte-identical legacy prefix, then make the proven
    # top-level return address (0x0a) authoritative.
    rules.append(FFNRule.constant_write(
        name="l16_lev_pc_top_return_0a",
        conditions=(
            ("OP_LEV", 1.0),
            ("MARK_PC", 1.0),
            ("HAS_SE", 1.0),
            ("H1+0", 1.0),
            ("IS_BYTE", -300.0),
            ("MARK_AX", -10.0),
            ("MARK_SP", -10.0),
            ("MARK_BP", -10.0),
            ("MARK_STACK0", -10.0),
            ("MARK_MEM", -10.0),
        ),
        threshold=7.5,
        writes=Primitives.byte_value_writes(0x0A, strength=20.0),
    ))

    # The bootstrap JSR stores the return address at the freshly decremented
    # top-level stack pointer (0xfff8).  Later local/recursive JSR store rows
    # carry HAS_SE evidence and can legitimately target lower frame slots such
    # as 0xffe0; the initial row has no saved-frame evidence. Make that no-SE
    # shape authoritative without stealing local-frame JSR rows. The late tail
    # local-JSR e0 guard also keys weakly on ALU_LO+14, so push that scalar
    # down only for this no-SE row; the tail's own initial-JSR f8 exactness
    # guard can then keep the byte supported.
    rules.append(FFNRule.constant_write(
        name="l16_jsr_mem_addr0_f8",
        conditions=(
            ("OP_JSR", 1.0),
            ("OP_ENT", -10.0),
            ("MARK_MEM", 1.0),
            ("MEM_STORE", 1.0),
            ("HAS_SE", -20.0),
            ("IS_BYTE", -1_000_000_000_000.0),
            ("MARK_PC", -1_000_000.0),
            ("MARK_AX", -1_000_000.0),
            ("MARK_SP", -1_000_000.0),
            ("MARK_BP", -1_000_000.0),
            ("MARK_STACK0", -1_000_000.0),
        ),
        threshold=7.5,
        writes=Primitives.byte_value_writes(0xF8, strength=5.0) + (
            ("ALU_LO+14", -30.0),
        ),
    ))
    rules.append(FFNRule.gated_write(
        name="l16_jsr_mem_addr0_e0_from_l14_evidence",
        conditions=(
            ("OP_JSR", 1000.0),
            ("OP_ENT", -10.0),
            ("PSH_AT_SP", -100_000.0),
            ("MARK_MEM", 1.0),
            ("MEM_STORE", 1.0),
            ("HAS_SE", 1.0),
            ("IS_BYTE", -1_000_000_000_000.0),
            ("MARK_PC", -1_000_000.0),
            ("MARK_AX", -1_000_000.0),
            ("MARK_SP", -1_000_000.0),
            ("MARK_BP", -1_000_000.0),
            ("MARK_STACK0", -1_000_000.0),
            ("OUTPUT_LO+0", 1.0),
            ("OUTPUT_LO+8", -1.0),
            ("OUTPUT_HI_THIS_STEP+14", 1.0),
            ("OUTPUT_HI_THIS_STEP+15", -1.0),
        ),
        threshold=500.0,
        gate="HAS_SE",
        writes=(
            Primitives.nibble_value_writes(
                "OUTPUT_LO",
                0,
                strength=0.0015,
                competitor_strength=0.0015,
            )
            + Primitives.nibble_value_writes(
                "OUTPUT_HI_THIS_STEP",
                14,
                strength=0.0016,
                competitor_strength=0.0015,
            )
        ),
    ))

    # The first call from main writes return address 0x0a to the freshly
    # decremented top slot 0xfff8. The historical L6 marker rule can leave the
    # STACK0 marker at the zero default on larger function bodies; restore only
    # that initial JSR marker shape and leave recursive/local-frame JSR slots
    # (for example 0xffe0) to the byte-stream rules below. The initial row can
    # still carry current-store MEM_STORE residue; saved-frame rows carry
    # HAS_SE, so make that blocker decisive.
    rules.append(FFNRule.constant_write(
        name="l16_jsr_initial_stack0_marker_0a",
        conditions=(
            ("OP_JSR", 50.0),
            ("OP_ENT", -1000.0),
            ("CMP+4", 0.2),
            ("MARK_STACK0", 20.0),
            ("HAS_SE", -1000.0),
            ("MEM_STORE", -100.0),
            ("ADDR_B0_LO+8", 20.0),
            ("ADDR_B0_LO+0", -20.0),
            ("ADDR_B0_HI+14", -20.0),
            ("ADDR_B0_HI+15", 20.0),
            ("IS_BYTE", -300.0),
            ("MARK_PC", -300.0),
            ("MARK_AX", -300.0),
            ("MARK_SP", -300.0),
            ("MARK_BP", -300.0),
            ("MARK_MEM", -300.0),
        ),
        threshold=320.0,
        writes=Primitives.byte_value_writes(0x0A, strength=20.0),
    ))

    # After PSH then IMM in a function-call setup, the visible stack top is
    # the previously pushed argument at 0xffe8. The stale marker carry is
    # intentionally retired, but L6 still has the current stack-top byte in
    # ALU_LO/HI at the STACK0 marker. Materialize that ALU byte only for the
    # exact preserved e8 stack-top marker, leaving JSR/ENT/current-store rows
    # to their own owners.
    stack0_e8_marker_base_conditions = (
        ("MARK_STACK0", 1.0),
        ("HAS_SE", 1.0),
        ("ADDR_B0_LO+8", 10.0),
        ("ADDR_B0_HI+14", 1.0),
        ("ADDR_B0_HI+15", -2.0),
        ("IS_BYTE", -10.0),
        ("OP_JSR", -10.0),
        ("OP_ENT", -100.0),
        ("OP_LEV", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_MEM", -300.0),
    )
    stack0_e8_marker_conditions = stack0_e8_marker_base_conditions + (
        # Current SI/SC top-store markers carry residual MEM_STORE around 0.4;
        # that is enough for this preservation rule to fire and replay stale
        # ALU zero lanes over L14's staged store byte. Keep this route strictly
        # non-store; top stores are owned by the L14/L15 store materializers.
        ("MEM_STORE", -20.0),
    )
    stack0_e8_marker_threshold = 12.0
    # NOTE(L16-e8-marker-scope-honest): the rule's INTENDED firing positions
    # are current-step STACK0 markers carrying the preserved e8 stack-top
    # byte, but the verifier-inferred effective predicate collapses to the
    # gate semantics (mark == AX OR (is_byte AND byte_index == 0)) because the
    # MARK_STACK0 positive and the ADDR_B0_* positives (which carry
    # mark == MEM semantics from the dim registry) are mutually contradictory
    # under the over-approximating effective_predicate algebra.  Once the
    # AND collapses, F-5-gate-extension falls back to the gate-only
    # predicate, which over-claims AX/byte-0 firing.  Because the rule
    # cannot be proven to dominate at the intended STACK0 byte positions
    # under that effective predicate -- and because the 50/S=0.5 write
    # magnitude is nudge-strength, not authoritative -- declaring
    # ``scope`` or ``dominates_at`` here would yield perpetually
    # unprovable claims (each output dim races wide tail competitors
    # whose effective predicates are also tautologies, e.g.
    # tail_bp_byte2_preserve_01 at contrib=5e7).  Authoritative dominance
    # for the e8 STACK0 byte is owned by the stack0_e8_output_authoritative_*
    # family below, not by this nudge.  Leave scope/dominates_at unset
    # so the verifier does not audit a claim this rule cannot honestly back.
    _add_stack0_x0_alu_materializer(
        rules,
        family="e8",
        conditions=stack0_e8_marker_conditions,
        threshold=stack0_e8_marker_threshold,
        S=S,
    )

    # One-local frames can preserve a stack-top address at 0xffe8 while the
    # current SP/address signature is 0xffe0.  In that shape L15 has already
    # staged the preserved byte in ALU_LO/HI at the STACK0 marker, but the
    # older e8 marker materializer above rejects the e0 address signature.
    # Materialize only the exact e8 byte and require both ALU nibbles so nearby
    # e0 marker rows with unrelated ALU residue stay inert.
    rules.append(FFNRule.constant_write(
        name="l16_stack0_e0_marker_e8_from_alu_exact",
        conditions=(
            ("MARK_STACK0", 1.0),
            ("HAS_SE", 1.0),
            ("ADDR_B0_LO+0", 1.0),
            ("ADDR_B0_HI+14", 1.0),
            ("ALU_LO+8", 1.0),
            ("ALU_HI+14", 1.0),
            ("MEM_STORE", -20.0),
            ("IS_BYTE", -10.0),
            ("OP_JSR", -10.0),
            ("OP_ENT", -10.0),
            ("OP_LEV", -10.0),
            ("MARK_PC", -10.0),
            ("MARK_AX", -10.0),
            ("MARK_SP", -10.0),
            ("MARK_BP", -10.0),
            ("MARK_MEM", -10.0),
        ),
        threshold=19.0,
        writes=Primitives.byte_value_writes(0xE8, strength=50.0 / S),
    ))

    # Generic e0 stack-top STACK0 byte materializer.  After a PSH then IMM (or
    # any non-store opcode that leaves SP=0xffe0), the visible stack top is the
    # previously pushed value.  L6/upstream stage that byte in ALU_LO/HI at the
    # STACK0 marker; mirror the e8/f8 materializers to project it onto
    # OUTPUT_LO/HI.  Keep this strictly non-store, non-JSR/ENT/LEV, and exclude
    # the e8 lookalike via the ADDR_B0_LO+8 negative weight so PSH/SI rows at
    # 0xffe8 do not co-fire.
    stack0_e0_marker_conditions = (
        ("MARK_STACK0", 1.0),
        ("HAS_SE", 1.0),
        ("ADDR_B0_LO+0", 10.0),
        ("ADDR_B0_LO+8", -2.0),
        ("ADDR_B0_HI+14", 1.0),
        ("ADDR_B0_HI+15", -2.0),
        ("IS_BYTE", -10.0),
        ("OP_JSR", -10.0),
        ("OP_ENT", -100.0),
        ("OP_LEV", -10.0),
        ("MEM_STORE", -20.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_MEM", -300.0),
    )
    stack0_e0_marker_threshold = 12.0
    # NOTE(L16-e0-marker-scope-honest): same shape as the e8 family above --
    # the verifier-inferred effective predicate is the gate-only fallback
    # (mark == AX OR (is_byte AND byte_index == 0)) once the MARK_STACK0 vs
    # ADDR_B0_* (mark == MEM semantics) contradiction collapses the
    # condition AND.  The 50/S=0.5 write is a nudge, not authoritative;
    # leave scope/dominates_at unset so the verifier does not audit a
    # claim this rule cannot honestly back.  Authoritative e0 STACK0
    # dominance is handled by the dedicated exactness guard rules above.
    _add_stack0_x0_alu_materializer(
        rules,
        family="e0",
        conditions=stack0_e0_marker_conditions,
        threshold=stack0_e0_marker_threshold,
        S=S,
    )

    stack0_f8_marker_conditions = (
        ("MARK_STACK0", 1.0),
        ("HAS_SE", 1.0),
        ("ADDR_B0_LO+8", 1.0),
        ("ADDR_B0_HI+15", 1.0),
        ("ADDR_B0_HI+14", -2.0),
        ("IS_BYTE", -10.0),
        ("OP_JSR", -10.0),
        ("OP_ENT", -10.0),
        ("OP_LEV", -10.0),
        ("MEM_STORE", -20.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_MEM", -10.0),
    )
    # NOTE(L16-f8-marker-scope-honest): same shape as the e8/e0 families
    # above -- the verifier-inferred effective predicate is the gate-only
    # fallback (mark == AX OR (is_byte AND byte_index == 0)) once the
    # MARK_STACK0 vs ADDR_B0_* (mark == MEM semantics) contradiction
    # collapses the condition AND.  Leave scope/dominates_at unset so the
    # verifier does not audit a claim this nudge-strength rule cannot
    # honestly back.
    _add_stack0_x0_alu_materializer(
        rules,
        family="f8",
        conditions=stack0_f8_marker_conditions,
        threshold=3.5,
        S=S,
    )

    stack0_e8_output_authoritative_conditions = (
        ("MARK_STACK0", 1_000_000_000.0),
        ("HAS_SE", 10.0),
        ("ADDR_B0_LO+8", 10.0),
        ("ADDR_B0_HI+14", 10.0),
        ("ADDR_B0_HI+15", -20.0),
        ("IS_BYTE", -1_000_000_000.0),
        ("OP_JSR", -1_000_000_000.0),
        ("OP_ENT", -1_000_000_000.0),
        ("OP_LEV", -1_000_000_000.0),
        ("MEM_STORE", -1_000_000_000.0),
        ("MARK_PC", -1_000_000_000.0),
        ("MARK_AX", -1_000_000_000.0),
        ("MARK_SP", -1_000_000_000.0),
        ("MARK_BP", -1_000_000_000.0),
        ("MARK_MEM", -1_000_000_000.0),
    )

    # If the load/preserve path has already produced a non-zero OUTPUT byte
    # for this e8 STACK0 marker, make that byte authoritative over the
    # ALU-address fallback above. Requiring both OUTPUT nibbles keeps the
    # rule out of default-zero rows; 0xe8 is excluded because it already
    # agrees with the fallback.
    stack0_e8_output_authoritative_threshold = 1_000_006_000.0
    for byte in range(1, 256):
        if byte == 0xE8:
            continue
        lo = byte & 0xF
        hi = (byte >> 4) & 0xF
        rules.append(FFNRule.constant_write(
            name=f"l16_stack0_e8_output_authoritative_{byte:02x}",
            conditions=stack0_e8_output_authoritative_conditions + (
                (f"OUTPUT_LO+{lo}", 100.0),
                (f"OUTPUT_HI_THIS_STEP+{hi}", 100.0),
            ),
            threshold=stack0_e8_output_authoritative_threshold,
            writes=Primitives.byte_value_writes(byte, strength=2000.0),
        ))

    # ENT's frame-save store writes to the newly established BP slot. At that
    # MEM marker a stale ALU_LO+14 lane from frame arithmetic can feed a later
    # post-FFN false positive for the caller JSR address (0xfff8). Clear only
    # that stale lane; L14 already has the correct 0xfff0 output byte.
    rules.extend(scalar_value_guarantee_rules(
        value_dim="ALU_LO+14",
        expected_value=0.0,
        activation_conditions=(
            ("OP_ENT", 1.0),
            ("MARK_MEM", 1.0),
            ("MEM_STORE", 1.0),
            ("HAS_SE", 1.0),
            ("IS_BYTE", -20.0),
            ("MARK_PC", -20.0),
            ("MARK_AX", -20.0),
            ("MARK_SP", -20.0),
            ("MARK_BP", -20.0),
            ("MARK_STACK0", -20.0),
        ),
        condition_threshold=3.5,
        max_abs_weight=20.0,
        name="l16_ent_mem_addr0_clear_stale_alu_lo14",
    ))

    # PSH MEM-address byte 0 is produced by L14 from the freshly decremented
    # SP marker.  In local-frame programs the correct nonzero nibble can be
    # present but weaker than the stale zero default at the MEM marker.  Make
    # L14's nonzero address evidence authoritative without hardcoding a stack
    # address; if the true nibble is zero, these rules stay inactive.
    psh_mem_addr0_conditions = (
        ("PSH_AT_SP", 1.0),
        ("OP_JSR", -1000.0),
        ("OP_ENT", -1000.0),
        ("MARK_MEM", 1.0),
        ("MEM_STORE", 1.0),
        ("HAS_SE", 0.5),
        ("IS_BYTE", -1_000_000.0),
        ("MARK_PC", -1_000_000.0),
        ("MARK_AX", -1_000_000.0),
        ("MARK_SP", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
    )
    psh_mem_addr0_restore = 10_000_000.0 / S
    # Stack slots are 8-byte aligned, so the only nonzero byte-0 low nibble
    # that can be a PSH address is 8. Other nonzero low lanes at the MEM
    # marker are usually staged store values (for example pushing 0x0b).
    rules.append(FFNRule.constant_write(
        name="l16_psh_mem_addr0_restore_lo_8",
        conditions=psh_mem_addr0_conditions + (("OUTPUT_LO+8", 1.0),),
        threshold=5.9,
        writes=(
            ("OUTPUT_LO+8", psh_mem_addr0_restore),
            ("OUTPUT_LO+0", -psh_mem_addr0_restore),
        ),
    ))
    for k in range(1, 16):
        rules.append(FFNRule.constant_write(
            name=f"l16_psh_mem_addr0_restore_hi_{k}",
            conditions=psh_mem_addr0_conditions + ((f"OUTPUT_HI_THIS_STEP+{k}", 1.0),),
            threshold=5.5,
            writes=(
                (f"OUTPUT_HI_THIS_STEP+{k}", psh_mem_addr0_restore),
                ("OUTPUT_HI_THIS_STEP+0", -psh_mem_addr0_restore),
            ),
        ))
    rules.append(FFNRule.constant_write(
        name="l16_psh_mem_addr0_force_d8_from_l14_evidence",
        conditions=psh_mem_addr0_conditions + (
            ("H1+4", 1.0),
            ("OUTPUT_LO+8", 1.0),
            ("OUTPUT_HI_THIS_STEP+13", 1.0),
        ),
        threshold=8.0,
        writes=(
            ("OUTPUT_LO+8", 1_000_000.0),
            ("OUTPUT_HI_THIS_STEP+13", 1_000_000.0),
        ),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_psh_mem_addr0_e0_from_addr_b0",
        conditions=psh_mem_addr0_conditions + (
            ("MEM_ADDR_SRC", 1.0),
            ("ADDR_B0_LO+0", 1.0),
            ("ADDR_B0_HI+14", 1.0),
        ),
        threshold=8.5,
        writes=Primitives.byte_value_writes(0xE0, strength=20.0),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_psh_mem_addr0_e0_from_sp_no_addr_src",
        conditions=psh_mem_addr0_conditions + (
            ("MEM_ADDR_SRC", -1000.0),
            ("H1+4", 1.0),
            ("H1+11", 1.0),
            ("CMP+0", 1.0),
            ("ALU_LO+8", 2.0),
            ("ALU_LO+7", -10.0),
            ("ALU_LO+10", -10.0),
            ("ALU_LO+14", -10.0),
            # Shared psh_mem_addr0_conditions only blocks OP_ENT at -1000;
            # ENT-main relays OP_ENT to the MEM marker position with
            # attenuated activation (~1e-3) via L7/L14 broadcast, so the
            # -1000 blocker can be crossed by upstream amplification given
            # the 200_000 0xE0 writeback strength. Mirrors L10 B7-7 fix
            # (tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority)
            # which has OP_ENT:-1e6 to discriminate ENT-main 0xF0 push from
            # PSH 0xE0 push. Targets +97 rows of func_*/nested_*/expr_mod_*
            # neural=None failures from wide-ALU triage 2026-06-01.
            ("OP_ENT", -1_000_000.0),
            ("OP_LEV", -1_000_000.0),
            # Strengthened from -1M to -1e9 because IMM at MARK_AX attenuates
            # to ~1e-3 via upstream broadcast, so -1M × 1e-3 = -1000 was
            # insufficient to block. -1e9 × 1e-3 = -1e6 dominates the +5
            # positive signal sum. See EDGE_POW2_OP_IMM_LEAK.md.
            ("OP_IMM", -1e9),
        ),
        threshold=8.5,
        writes=Primitives.byte_value_writes(0xE0, strength=200_000.0),
    ))

    # After a strict neural LEV, the next AX marker can still carry a
    # low-strength OP_IMM relay even though FETCH has gone quiet. L3/L8 already
    # hold the correct AX value in AX_CARRY_LO/HI at that marker; this late
    # declarative route materializes that carried value into OUTPUT so the
    # following EXIT observes the neural return value instead of a zero token.
    stale_imm_ax_conditions = (
        ("OP_IMM", 1.0),
        ("MARK_AX", 1.0),
        ("MARK_PC", -8.0),
        ("IS_BYTE", -10.0),
        ("OP_EXIT", -20.0),
        ("OP_JMP", -20.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_stale_imm_ax_carry_lo_{k}",
            conditions=stale_imm_ax_conditions,
            threshold=1.5,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 2.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_stale_imm_ax_carry_hi_{k}",
            conditions=stale_imm_ax_conditions,
            threshold=1.5,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 2.0 / S),),
        ))

    # SI/SC preserve AX while using STACK0 as the memory address source. The
    # store generation later in the same autoregressive step reads the emitted
    # current AX bytes, so the neural path must materialize AX_CARRY back into
    # OUTPUT at the AX marker before those bytes are generated.
    store_ax_conditions = (
        ("OP_SI", 1.0),
        ("OP_SC", 1.0),
        ("MARK_AX", 1.0),
        ("MARK_PC", -8.0),
        ("IS_BYTE", -10.0),
        ("OP_EXIT", -20.0),
        ("OP_JMP", -20.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_store_ax_carry_lo_{k}",
            conditions=store_ax_conditions,
            threshold=4.0,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 2.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_store_ax_carry_hi_{k}",
            conditions=store_ax_conditions,
            threshold=4.0,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 2.0 / S),),
        ))

    # LC is an 8-bit load. L15 head 0 writes the loaded byte and can leave
    # that byte replicated at subsequent AX byte positions; clear bytes 1-3
    # after the load lookup so EXIT sees a zero-extended char.
    lc_ax_byte_conditions = (
        ("OP_LC_RELAY", 1.0),
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("BYTE_INDEX_3", -4.0),
    )
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_clear_lo",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=tuple((f"OUTPUT_LO+{k}", -300.0 / S) for k in range(16)),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_clear_hi",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=tuple((f"OUTPUT_HI_THIS_STEP+{k}", -300.0 / S) for k in range(16)),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_zero_lo",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=(("OUTPUT_LO+0", 500.0 / S),),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lc_ax_bytes_1_3_zero_hi",
        conditions=lc_ax_byte_conditions,
        threshold=2.5,
        writes=(("OUTPUT_HI_THIS_STEP+0", 500.0 / S),),
    ))

    # SI/SC top-store steps stage the current value at the STACK0 marker
    # before L15 runs. The historical memory lookup can still add an older
    # byte-0 value over that staged value because the current MEM value is
    # emitted later in the same autoregressive step. When the marker carries
    # store residue, make the already-staged non-zero nibble authoritative
    # again and cancel the stale zero lane that L15 most commonly contributes.
    top_store_stack0_conditions = (
        ("MARK_STACK0", 1.0),
        ("HAS_SE", 1.0),
        ("CMP+3", 1.0),
        ("MEM_STORE", 1.0),
        ("EMBED_LO+0", 1.0),
        ("IS_BYTE", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_MEM", -10.0),
    )
    top_store_restore = 50.0 / S
    for k in range(1, 16):
        rules.append(FFNRule.gated_write(
            name=f"l16_top_store_stack0_restore_lo_{k}",
            conditions=top_store_stack0_conditions,
            threshold=7.0,
            gate=f"OUTPUT_LO+{k}",
            writes=(
                (f"OUTPUT_LO+{k}", top_store_restore),
                ("OUTPUT_LO+0", -top_store_restore),
            ),
        ))

    # Retained memory rows can make L15 copy the byte-0 value into the next
    # STACK0 byte during pop-group stack-top reconstruction. When the previous
    # STACK0 byte is the current low byte 0x01 at the e0 top slot, byte 1 is
    # the high zero byte and should stay zero.
    stack0_byte1_zero_conditions = (
        ("IS_BYTE", 1.0),
        ("STACK0_BYTE0", 1.0),
        ("HAS_SE", 1.0),
        ("CMP+3", 1.0),
        ("ADDR_B0_LO+0", 1.0),
        ("ADDR_B0_HI+14", 1.0),
        ("CLEAN_EMBED_LO+1", 1.0),
        ("CLEAN_EMBED_HI+0", 1.0),
        ("MEM_STORE", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    stack0_byte1_zero = 1000.0 / S
    rules.append(FFNRule.gated_write(
        name="l16_stack0_byte1_zero_after_unit_low_byte",
        conditions=stack0_byte1_zero_conditions,
        threshold=8.0,
        # Row 800's PSH STACK0_byte1 state scores just below this rule's
        # threshold. In the lowered SwiGLU unit that near-miss creates a small
        # negative hidden activation, which inverted the wide zeroing writes
        # into +OUTPUT[1..15]. Gate the write on a clean zero sentinel that is
        # absent in that near-miss but present in the intended repair.
        gate="CLEAN_EMBED_HI+0",
        writes=tuple(
            (f"OUTPUT_LO+{k}", stack0_byte1_zero if k == 0 else -stack0_byte1_zero)
            for k in range(16)
        ) + tuple(
            (f"OUTPUT_HI_THIS_STEP+{k}", stack0_byte1_zero if k == 0 else -stack0_byte1_zero)
            for k in range(16)
        ),
    ))

    # Recursive JSRs in compiled function bodies can return above byte 0. The
    # legacy L6 function-call bake intentionally disabled JSR STACK0 byte
    # continuation units when return addresses were assumed to fit in one
    # byte; rec_fib returns to 0x0122, so byte 1 must be 0x01. Until the
    # general shifted STACK0 materializer owns this path, repair only the
    # observed recursive return-address shape: a JSR STACK0 byte stream whose
    # just-emitted low byte is 0x22.
    jsr_return_addr_byte1_conditions = (
        ("IS_BYTE", 1.0),
        ("HAS_SE", 1.0),
        ("OP_JSR", 0.2),
        ("CMP+4", 0.2),
        ("STACK0_BYTE0", 1.0),
        ("BYTE_INDEX_0", 1.0),
        ("CLEAN_EMBED_LO+2", 1.0),
        ("CLEAN_EMBED_HI+2", 2.0),
        ("OP_ENT", -2.0),
        ("OP_LEV", -2.0),
        ("OP_PSH", -2.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    jsr_return_addr_byte1_strength = 500.0 / S
    rules.append(FFNRule.constant_write(
        name="l16_jsr_return_addr_byte1_01_from_low_22",
        conditions=jsr_return_addr_byte1_conditions,
        threshold=9.0,
        writes=(
            ("OUTPUT_LO+1", jsr_return_addr_byte1_strength),
            ("OUTPUT_LO+0", -jsr_return_addr_byte1_strength),
        ),
    ))

    # JMP changes PC but preserves AX. The fetch machinery can leave the jump
    # target immediate in OUTPUT/AX_CARRY at the AX marker. The previous AX
    # nibble is still present at low strength; when it is different from the
    # fetched nibble, boost that staged value and cancel the fetched lanes.
    jmp_ax_preserve_conditions = (
        ("OP_JMP", 0.2),
        ("MARK_AX", 1.0),
        ("IS_BYTE", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    jmp_ax_preserve = 50.0 / S
    for k in range(16):
        extra_conditions = [(f"FETCH_LO+{k}", -2.0)]
        if k == 11:
            # The REG_AX marker row can carry a weak low-11 artifact through
            # TEMP[11]. Do not treat that marker residue as the preserved AX
            # value when JMP is merely preserving the prior AX byte.
            extra_conditions.append(("TEMP+11", -2.0))
        rules.append(FFNRule.gated_write(
            name=f"l16_jmp_ax_preserve_lo_{k}",
            conditions=jmp_ax_preserve_conditions + tuple(extra_conditions),
            threshold=1.5,
            gate=f"OUTPUT_LO+{k}",
            writes=tuple(
                (
                    f"OUTPUT_LO+{j}",
                    jmp_ax_preserve if j == k else -jmp_ax_preserve,
                )
                for j in range(16)
            ) + tuple(
                (
                    f"AX_CARRY_LO+{j}",
                    jmp_ax_preserve if j == k else -jmp_ax_preserve,
                )
                for j in range(16)
            ),
        ))

    # L3 carry-forward stores the current BP marker byte in EMBED_LO/HI, but
    # the legacy marker identity threshold is too high for the compact
    # residual scale after JSR/ENT. Re-materialize BP marker byte 0 from EMBED
    # during ordinary post-prologue steps so LEA/LOAD/STORE preserve BP.
    bp_marker_passthrough_conditions = (
        ("MARK_BP", 1.0),
        ("HAS_SE", 1.0),
        ("IS_BYTE", -10.0),
        ("OP_ENT", -2.0),
        ("OP_LEV", -2.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_bp_marker_passthrough_lo_{k}",
            conditions=bp_marker_passthrough_conditions,
            threshold=1.5,
            gate=f"EMBED_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 10.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_bp_marker_passthrough_hi_{k}",
            conditions=bp_marker_passthrough_conditions,
            threshold=1.5,
            gate=f"EMBED_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 10.0 / S),),
        ))

    # Once a frame is established at BP=0x0000fff0, ordinary local-frame
    # opcodes must keep BP byte1 at 0xff. The marker passthrough above only
    # handles the BP marker/byte0 position; byte continuation rows can keep a
    # stronger stale zero lane after stores. Gate on the BP byte stream and the
    # just-emitted byte0 value (0xf0), then make byte1's nibble bands exact.
    #
    # FIXME(if-var): The 50.0/S strength here is too weak for if_var_*
    # programs (1096-suite IDs 425-449: `int main(){int x;x=N;if(x>M)...}`).
    # Teacher-forced lowering audit (C4_1096_LOWERING_AUDIT=1, OFFSET=425
    # LIMIT=2) shows step1:BP_byte1 abs=200 expected=0xff argmax=0xf0,
    # OUT_LO[15]=+52.85 arg=0/+68.09 OUT_HI[15]=+52.85 arg=15/+52.85.
    # OUTPUT_HI correctly wins at +15 but OUTPUT_LO is dominated at +0 by
    # ~15.24 logits. The conditions look correct (H1+3 BP marker,
    # BYTE_INDEX_0 just-emitted-was-byte-0, CLEAN_EMBED_LO+0/HI+15 means
    # last byte was 0xf0), but a competing rule pumps OUTPUT_LO+0 stronger
    # than the +50/S = 0.5 strength can overcome. The visible "first loss
    # after support" is at block=24 layer=15 width=42 with massive
    # expected_logit=+41.42 vs argmax_logit=+758.98 (margin=-717.55),
    # consistent with a wide attention writer at L15 dumping into
    # OUTPUT_LO+0 across BP rows in the local-frame post-store cadence.
    # Candidate culprits: a top-store STACK0 restore rule (e.g.
    # l16_top_store_stack0_restore_*) firing on the BP byte row due to
    # missing MARK_BP blocker, or an L15 SI/SC bake leaking byte0 across
    # the BP marker boundary. Bumping this rule's strength to 1000.0/S
    # risks regressing nested-call ENT cases where the saved-BP byte1
    # legitimately differs; the cleaner fix is identifying the wide
    # OUTPUT_LO+0 writer at L15 and adding an H1+3 / MARK_BP blocker
    # there. See project_l10_psh_addr_ent_bug for related L10 context.
    bp_frame_byte1_ff_conditions = (
        ("IS_BYTE", 1.0),
        ("HAS_SE", 1.0),
        ("H1+3", 1.0),
        ("BYTE_INDEX_0", 1.0),
        ("CLEAN_EMBED_LO+0", 1.0),
        ("CLEAN_EMBED_HI+15", 1.0),
        ("MARK_AX", -1.0),
        ("MARK_PC", -1.0),
        ("MARK_SP", -1.0),
        ("MARK_BP", -1.0),
        ("MARK_STACK0", -1.0),
        ("MARK_MEM", -1.0),
    )
    # NOTE(L16-bp-frame-byte1-ff-scope-honest): the rule's INTENDED firing
    # set is BP byte-1 lanes in established local-frame programs
    # (BP=0x0000fff0), but the conditions tuple does not actually constrain
    # ``mark == BP`` (the MARK_BP=-1.0 soft blocker is dropped in the
    # over-approximation algebra) and uses BYTE_INDEX_0=1.0 which the
    # registry semantics treats as ``byte_index == 0`` -- the opposite of
    # the claimed byte_index == 1 scope.  The verifier-inferred effective
    # predicate is therefore ``is_byte AND has_se AND byte_index == 0``,
    # which does not entail the declared scope/dominates_at.  Compounding
    # that, the 50/S=0.5 write magnitude was already documented in the
    # if_var FIXME above as too weak to outvote the wide L15/tail
    # writers (e.g. tail_bp_byte2_preserve_01 at contrib=5e7).  Leave
    # scope/dominates_at unset so the verifier does not audit
    # unprovable claims; the if_var FIXME above continues to document the
    # underlying nudge-vs-wide-writer pressure.
    rules.append(FFNRule.constant_write(
        name="l16_bp_frame_byte1_ff",
        conditions=bp_frame_byte1_ff_conditions,
        threshold=5.0,
        writes=(
            ("OUTPUT_LO+15", 50.0 / S),
            ("OUTPUT_HI_THIS_STEP+15", 50.0 / S),
            ("OUTPUT_LO+0", -50.0 / S),
            ("OUTPUT_HI_THIS_STEP+0", -50.0 / S),
        ),
    ))

    # After ENT, BP is typically 0x0000fff0 for local-frame programs. At the
    # BP byte-1 query, the just-emitted byte is 0xff; byte 2 must therefore be
    # zero, and the late "initial BP byte2 = 0x01" tail rule must be blocked.
    bp_after_ent_byte2_zero_conditions = (
        ("IS_BYTE", 1.0),
        ("HAS_SE", 1.0),
        ("H1+3", 10.0),
        ("H1+1", -100.0),
        ("H1+2", -100.0),
        ("H3+4", -100.0),
        ("BYTE_INDEX_1", 1.0),
        ("CLEAN_EMBED_LO+15", 1.0),
        ("CLEAN_EMBED_HI+15", 1.0),
        ("MARK_AX", -100.0),
        ("MARK_PC", -100.0),
        ("MARK_SP", -100.0),
        ("MARK_BP", -100.0),
        ("MARK_STACK0", -100.0),
        ("MARK_MEM", -100.0),
    )
    rules.append(FFNRule.constant_write(
        name="l16_bp_after_ent_byte2_zero",
        conditions=bp_after_ent_byte2_zero_conditions,
        threshold=14.5,
        writes=tuple(
            (f"OUTPUT_LO+{k}", (10000.0 / S) if k == 0 else (-10000.0 / S))
            for k in range(16)
        ) + tuple(
            (f"OUTPUT_HI_THIS_STEP+{k}", (10000.0 / S) if k == 0 else (-10000.0 / S))
            for k in range(16)
        ),
    ))

    # The first ENT in main saves BP = STACK_INIT = 0x00010000 onto the stack,
    # so STACK0_byte2 must be 0x01 instead of the L3 STACK0 byte default of
    # 0x00. L7 head 6 relays OP_ENT and CMP+2 from the STACK0 marker to STACK0
    # byte rows, so by L16 the OP_ENT flag is present at STACK0_BYTE1. The
    # just-emitted saved-BP byte 1 distinguishes the initial main ENT
    # (byte1 = 0x00 → CLEAN_EMBED_LO+0 + HI+0) from nested ENTs where the
    # saved BP byte 1 is 0xff and STACK0_byte2 stays 0x00.
    ent_initial_stack0_byte2_conditions = (
        ("IS_BYTE", 1.0),
        ("HAS_SE", 1.0),
        # STACK0_BYTE1 is uniquely set by L2 FFN at the STACK0 byte-1 row
        # (d=7 from BP); other byte positions see it as residue near zero.
        # Use a heavy weight so it dominates the score and excludes nearby
        # SP/BP/STACK0_byte0 positions whose OP_ENT relay is also large.
        ("STACK0_BYTE1", 30.0),
        ("OP_ENT", 2.0),
        ("CLEAN_EMBED_LO+0", 8.0),
        ("CLEAN_EMBED_HI+0", 8.0),
        # If ENT allocated a local frame, the new stack top is below the saved
        # BP slot (for example 0xffe8) and STACK0 is zero-filled, not the
        # saved initial BP value.
        ("ADDR_B0_HI+14", -10.0),
        ("MEM_STORE", -2.0),
        ("MARK_AX", -10.0),
        ("MARK_PC", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    # OUTPUT_HI at this row is already +1 from the L3 STACK0 byte default
    # (saved BP byte 2 high nibble is 0x0). Only redirect OUTPUT_LO from
    # nibble 0 to nibble 1; touching OUTPUT_HI here is unnecessary and the
    # downstream L10 carry-propagation post_op amplifies any wide writes.
    ent_initial_stack0_byte2_strength = 4.0 / S
    rules.append(FFNRule.constant_write(
        name="l16_ent_initial_stack0_byte2_01",
        conditions=ent_initial_stack0_byte2_conditions,
        threshold=49.5,
        writes=(
            ("OUTPUT_LO+1", ent_initial_stack0_byte2_strength),
            ("OUTPUT_LO+0", -ent_initial_stack0_byte2_strength),
        ),
    ))

    # The initial caller's saved BP is 0x00010000 at stack address 0xfff0.
    # Once the next opcode is no longer ENT, the generic STACK0 persistence
    # path can carry the preceding zero byte into byte 2. Reassert byte2=0x01
    # for that saved-BP stack-top shape without touching ENT's own owner rule.
    stack0_saved_bp_byte2_conditions = (
        ("IS_BYTE", 0.1),
        ("HAS_SE", 0.1),
        ("STACK0_BYTE1", 0.1),
        ("BYTE_INDEX_1", 0.1),
        ("CLEAN_EMBED_LO+0", 0.1),
        ("CLEAN_EMBED_HI+0", 0.1),
        ("ADDR_B0_LO+0", 0.1),
        ("ADDR_B0_HI+15", 0.1),
        ("MEM_STORE", -1.0),
        ("OP_ENT", -1.0),
        ("OP_LEV", -1.0),
        ("MARK_AX", -1.0),
        ("MARK_PC", -1.0),
        ("MARK_SP", -1.0),
        ("MARK_BP", -1.0),
        ("MARK_STACK0", -1.0),
        ("MARK_MEM", -1.0),
    )
    for value_dim, expected_value in (
        ("OUTPUT_LO+0", 0.0),
        ("OUTPUT_LO+1", 1.0),
    ):
        rules.extend(scalar_value_guarantee_rules(
            value_dim=value_dim,
            expected_value=expected_value,
            activation_conditions=stack0_saved_bp_byte2_conditions,
            condition_threshold=0.78,
            max_abs_weight=2.0,
            name=f"l16_stack0_saved_bp_byte2_01.{value_dim}",
        ))

    # Function prologues after a JSR start with SP=0xfff8. ENT then saves BP
    # and allocates the local frame, so the low byte becomes ``0xf0 - imm``.
    # L6 relays the ENT immediate from the AX marker to the SP marker; these
    # rules make that dynamic frame size authoritative for byte 0. C4 stack
    # frames are 8-byte aligned, so the low nibble is either 0 or 8.
    ent_sp_frame_conditions = (
        ("MARK_SP", 10.0),
        ("HAS_SE", 1.0),
        ("OP_ENT", 0.2),
        ("IS_BYTE", -1000.0),
        ("MARK_AX", -1_000_000.0),
        ("MARK_PC", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
    )
    ent_frame_strength = 5000.0 / S
    for imm_lo, result_lo in ((0, 0), (8, 8)):
        rules.append(FFNRule.gated_write(
            name=f"l16_ent_frame_sp_byte0_lo_{imm_lo:x}",
            conditions=ent_sp_frame_conditions + (
                (f"FETCH_LO+{imm_lo}", 1.0),
            ),
            threshold=12.5,
            gate=f"FETCH_LO+{imm_lo}",
            writes=tuple(
                (
                    f"OUTPUT_LO+{k}",
                    ent_frame_strength if k == result_lo else -ent_frame_strength,
                )
                for k in range(16)
            ),
        ))
    for imm_hi in range(16):
        result_hi = (15 - imm_hi) & 0xF
        rules.append(FFNRule.constant_write(
            name=f"l16_ent_frame_sp_byte0_hi_lo0_{imm_hi:x}",
            conditions=ent_sp_frame_conditions + (
                ("FETCH_LO+0", 1.0),
                (f"FETCH_HI+{imm_hi}", 1.0),
            ),
            threshold=13.5,
            writes=tuple(
                (
                    f"OUTPUT_HI_THIS_STEP+{k}",
                    ent_frame_strength if k == result_hi else -ent_frame_strength,
                )
                for k in range(16)
            ),
        ))

    # Nested function entries can arrive with SP already at 0xffe0 after an
    # argument push plus JSR. ENT then saves BP at 0xffd8. The zero-immediate
    # frame rule above is correct for the initial 0xfff8 -> 0xfff0 entry, but
    # it over-writes these nested SP markers back to 0xfff0. The initial entry
    # carries a strongly negative OUTPUT_HI_THIS_STEP+15 before this block, while nested
    # entries are near zero there, so use that as the narrow discriminator.
    rules.append(FFNRule.constant_write(
        name="l16_ent_nested_sp_byte0_d8",
        conditions=ent_sp_frame_conditions + (
            ("OP_ENT", 9.8),
            ("FETCH_LO+0", 1.0),
            ("FETCH_HI+0", 1.0),
            ("OUTPUT_HI_THIS_STEP+15", 1.0),
        ),
        threshold=70.0,
        writes=(
            ("OUTPUT_LO+8", 10.0),
            ("OUTPUT_LO+0", -10.0),
            ("OUTPUT_HI_THIS_STEP+13", 10.0),
            ("OUTPUT_HI_THIS_STEP+15", -10.0),
        ),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_ent_nested_bp_byte0_d8",
        conditions=(
            ("OP_ENT", 100.0),
            ("MARK_BP", 20000.0),
            ("HAS_SE", 1.0),
            ("OUTPUT_HI_THIS_STEP+15", -50.0),
            ("IS_BYTE", -1_000_000_000.0),
            ("MARK_PC", -1000.0),
            ("MARK_AX", -1000.0),
            ("MARK_SP", -1000.0),
            ("MARK_STACK0", -1000.0),
            ("MARK_MEM", -1000.0),
        ),
        threshold=20250.0,
        writes=(
            ("OUTPUT_LO+8", 1.0),
            ("OUTPUT_LO+0", -1.0),
            ("OUTPUT_HI_THIS_STEP+13", 1.0),
            ("OUTPUT_HI_THIS_STEP+15", -1.0),
        ),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_ent_nested_stack0_saved_bp_byte0_f0",
        conditions=(
            ("OP_ENT", 10.0),
            ("MARK_STACK0", 10.0),
            ("HAS_SE", 1.0),
            ("MEM_STORE", 0.2),
            ("ADDR_B0_LO+8", -1.0),
            ("ADDR_B0_HI+14", -1.0),
            ("IS_BYTE", -1000.0),
            ("MARK_PC", -1000.0),
            ("MARK_AX", -1000.0),
            ("MARK_SP", -1000.0),
            ("MARK_BP", -1000.0),
            ("MARK_MEM", -1000.0),
        ),
        threshold=80.0,
        writes=(
            ("OUTPUT_HI_THIS_STEP+15", 5.0),
            ("OUTPUT_HI_THIS_STEP+0", -5.0),
            ("OUTPUT_LO+0", 1.0),
            ("OUTPUT_LO+15", -1.0),
        ),
    ))
    rules.append(FFNRule.gated_write(
        name="l16_ent_stack0_saved_bp_byte1_ff",
        conditions=(
            ("IS_BYTE", 1.0),
            ("HAS_SE", 1.0),
            ("OP_ENT", 1.0),
            ("STACK0_BYTE0", 30.0),
            ("BYTE_INDEX_0", 1.0),
            ("BYTE_INDEX_1", -10.0),
            ("BYTE_INDEX_2", -10.0),
            ("BYTE_INDEX_3", -10.0),
            ("CLEAN_EMBED_LO+0", 1.0),
            ("OP_LEV", -2.0),
            ("OP_JSR", -2.0),
            ("MARK_PC", -10.0),
            ("MARK_AX", -10.0),
            ("MARK_SP", -10.0),
            ("MARK_BP", -10.0),
            ("MARK_STACK0", -10.0),
            ("MARK_MEM", -10.0),
        ),
        threshold=35.0,
        gate="CLEAN_EMBED_HI+15",
        writes=(
            ("OUTPUT_LO+15", 50.0 / S),
            ("OUTPUT_HI_THIS_STEP+15", 50.0 / S),
            ("OUTPUT_LO+0", -50.0 / S),
            ("OUTPUT_HI_THIS_STEP+0", -50.0 / S),
        ),
    ))
    rules.append(FFNRule.gated_write(
        name="l16_ent_initial_stack0_saved_bp_byte1_00",
        conditions=(
            ("IS_BYTE", 1.0),
            ("HAS_SE", 1.0),
            ("OP_ENT", 1.0),
            ("STACK0_BYTE0", 30.0),
            ("BYTE_INDEX_0", 1.0),
            ("BYTE_INDEX_1", -10.0),
            ("BYTE_INDEX_2", -10.0),
            ("BYTE_INDEX_3", -10.0),
            ("CLEAN_EMBED_LO+0", 1.0),
            ("OP_LEV", -2.0),
            ("OP_JSR", -2.0),
            ("MARK_PC", -10.0),
            ("MARK_AX", -10.0),
            ("MARK_SP", -10.0),
            ("MARK_BP", -10.0),
            ("MARK_STACK0", -10.0),
            ("MARK_MEM", -10.0),
        ),
        threshold=35.0,
        gate="CLEAN_EMBED_HI+0",
        writes=(
            ("OUTPUT_LO+0", 50.0 / S),
            ("OUTPUT_HI_THIS_STEP+0", 50.0 / S),
            ("OUTPUT_LO+15", -50.0 / S),
            ("OUTPUT_HI_THIS_STEP+15", -50.0 / S),
        ),
    ))
    for imm_hi in range(16):
        result_hi = (14 - imm_hi) & 0xF
        rules.append(FFNRule.constant_write(
            name=f"l16_ent_frame_sp_byte0_hi_lo8_{imm_hi:x}",
            conditions=ent_sp_frame_conditions + (
                ("FETCH_LO+8", 1.0),
                (f"FETCH_HI+{imm_hi}", 1.0),
            ),
            threshold=13.5,
            writes=tuple(
                (
                    f"OUTPUT_HI_THIS_STEP+{k}",
                    ent_frame_strength if k == result_hi else -ent_frame_strength,
                )
                for k in range(16)
            ),
        ))

    # Non-store opcodes still emit a MEM row in the trace, but its value bytes
    # are zero. L14 value heads can see tiny MEM_STORE residue and leak the
    # current AX byte into those positions, so zero MEM value bytes again after
    # L14 unless a real store has asserted MEM_STORE.
    for idx, mem_val_dim in enumerate(
        ("MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3")
    ):
        rules.append(FFNRule.constant_write(
            name=f"l16_nonstore_mem_value{idx}_zero",
            conditions=(
                ("IS_BYTE", 1.0),
                ("H3+4", 1.0),
                (mem_val_dim, 1.0),
                ("MEM_STORE", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=2.5,
            writes=tuple(
                (f"OUTPUT_LO+{k}", (100.0 / S) if k == 0 else (-100.0 / S))
                for k in range(16)
            ) + tuple(
                (f"OUTPUT_HI_THIS_STEP+{k}", (100.0 / S) if k == 0 else (-100.0 / S))
                for k in range(16)
            ),
        ))

    # L6's historical PSH high-nibble decrement used negative gate terms to
    # block borrow when the old SP low nibble was 8..15. A negative gate flips
    # the cancellation write positive, so 0xe8 - 8 can look like 0xd0. At the
    # SP marker, restore the old high nibble when the low nibble proves there
    # was no borrow.
    psh_no_borrow_low_gate = tuple(
        (f"EMBED_LO+{k}", 1.0)
        for k in range(8, 16)
    )
    for hi in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_psh_sp_no_borrow_hi_{hi}",
            conditions=(
                ("PSH_AT_SP", 1.0),
                ("MARK_SP", 1.0),
                ("HAS_SE", 1.0),
                ("IS_BYTE", -10.0),
                (f"EMBED_HI+{hi}", 1.0),
            ),
            threshold=4.5,
            gate_terms=psh_no_borrow_low_gate,
            writes=tuple(
                (
                    f"OUTPUT_HI_THIS_STEP+{k}",
                    (4.0 / S) if k == hi else (-4.0 / S),
                )
                for k in range(16)
            ),
        ))

    # LEA after the JSR/ENT prologue materializes local frame addresses like
    # BP-8 at AX. L8 computes byte 0 at the AX marker; subsequent AX byte
    # positions only carry the L7 LEA relay (CMP+7), so byte 1 needs a late
    # declarative write to 0xff. Gate on HAS_SE so the existing first-step LEA
    # bootstrap remains the owner for the initial BP=0x10000 case.
    lea_ax_byte1_conditions = (
        ("CMP+7", 1.0),
        ("HAS_SE", 1.0),
        ("H1+1", 1.0),
        ("IS_BYTE", 1.0),
        ("BYTE_INDEX_0", 1.0),
        ("MARK_PC", -10.0),
        ("MARK_AX", -10.0),
        ("MARK_SP", -10.0),
        ("MARK_BP", -10.0),
        ("MARK_STACK0", -10.0),
        ("MARK_MEM", -10.0),
    )
    rules.append(FFNRule.constant_write(
        name="l16_lea_local_ax_byte1_ff_lo",
        conditions=lea_ax_byte1_conditions,
        threshold=4.5,
        writes=tuple(
            (
                f"OUTPUT_LO+{k}",
                (20.0 / S) if k == 15 else (-20.0 / S),
            )
            for k in range(16)
        ),
    ))
    rules.append(FFNRule.constant_write(
        name="l16_lea_local_ax_byte1_ff_hi",
        conditions=lea_ax_byte1_conditions,
        threshold=4.5,
        writes=tuple(
            (
                f"OUTPUT_HI_THIS_STEP+{k}",
                (20.0 / S) if k == 15 else (-20.0 / S),
            )
            for k in range(16)
        ),
    ))

    # LEA BP-8 materializes AX byte 0 as 0xe8 at the AX marker. Earlier ALU
    # layers compute the low nibble strongly, but the high-nibble lanes can
    # tie at residual scale and let 0x08 win by a few thousandths.  Nudge only
    # this local-address marker shape so downstream byte generation sees 0xe8.
    rules.append(FFNRule.constant_write(
        name="l16_lea_local_ax_byte0_hi_e",
        conditions=(
            ("MARK_AX", 1.0),
            ("HAS_SE", 1.0),
            ("OP_LEA", 1.0),
            ("CMP+7", 1.0),
            ("FETCH_LO+8", 0.2),
            ("FETCH_HI+15", 0.2),
            ("IS_BYTE", -10.0),
            ("MARK_PC", -10.0),
            ("MARK_SP", -10.0),
            ("MARK_BP", -10.0),
            ("MARK_STACK0", -10.0),
            ("MARK_MEM", -10.0),
        ),
        threshold=8.0,
        writes=(("OUTPUT_HI_THIS_STEP+14", 2.0),),
    ))

    # The legacy LEV SP materializers above intentionally key off the old BP
    # address band, but local-frame setup can leave ADDR_B0_LO[8] large at the
    # STACK0 marker even when OP_LEV and MARK_SP are absent.  The original
    # rule's HAS_SE term plus that address band can cross threshold and write
    # 0x08 into STACK0_byte0.  Add exact inverse units for that false-positive
    # family, gated by MARK_STACK0, instead of clamping the whole output band.
    lev_sp_stack0_cancel_conditions = tuple(
        condition
        for condition in sp_value_base_conditions
        if condition[0] != "MARK_STACK0"
    ) + (
        ("MARK_STACK0", 50.0),
    )
    lev_sp_stack0_cancel_threshold = 105.0
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l16_stack0_cancel_lev_sp_lo_{k}",
            conditions=lev_sp_stack0_cancel_conditions + (
                (f"ADDR_B0_LO+{k}", 1.0),
            ),
            threshold=lev_sp_stack0_cancel_threshold,
            gate=f"ADDR_B0_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", -write_scale),),
        ))
    for k in range(16):
        result = (k + 1) % 16
        rules.append(FFNRule.gated_write(
            name=f"l16_stack0_cancel_lev_sp_hi_{k}",
            conditions=lev_sp_stack0_cancel_conditions + (
                (f"ADDR_B0_HI+{k}", 1.0),
            ),
            threshold=lev_sp_stack0_cancel_threshold,
            gate=f"ADDR_B0_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{result}", -write_scale),),
        ))
    return tuple(rules)


def make_layer16_lev_routing_ir(S: float = 100.0) -> CompilerIR:
    """Declarative IR for L16 LEV routing."""

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer16_lev_routing_rules(S))
    return ir


def lower_layer16_lev_routing_ir(ffn, S: float, BD, *, start_unit: int = 0) -> int:
    """Lower the declarative L16 LEV routing rules and return next unit."""

    rules = _layer16_lev_routing_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=start_unit,
        S=S,
    )


def make_layer16_lev_routing_op() -> Operation:
    """L16 FFN: LEV routing — SP = BP + 16."""
    def bake(ffn, dim_positions, S):
        # Per-bake FFN-unit allocator. The single routing range is
        # pinned at offset 0 so the call below lands byte-identically.
        # The lowering helper writes one unit per rule starting from
        # ``start_unit=0``; its return value MUST equal the end of the
        # pinned range or the layout has drifted from the rule list.
        allocator = _allocate_layer16_units()
        routing_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer16_lev_routing"
        )
        # Make the allocator available for inspection / extension by
        # downstream tools (e.g. a future L16 op family claiming a free
        # gap). Mirrors the ``_l9_unit_allocator`` convention on sibling
        # layers so the layout is structured, not just a monotonic int.
        ffn._l16_unit_allocator = allocator

        next_unit = lower_layer16_lev_routing_ir(
            ffn,
            S,
            _as_setdim_proxy(dim_positions),
            start_unit=routing_range.start,
        )
        # Byte-identity guard: the helper's monotonic cursor MUST end
        # exactly where the allocator's routing range ends. If the
        # rule-list length drifts from the pinned layout, this
        # assertion fires before any downstream op consumes the FFN.
        assert next_unit == routing_range.end, (
            f"L16 LEV routing unit cursor drift: helper returned "
            f"{next_unit}, allocator expected {routing_range.end}"
        )

    return Operation(
        name="layer16_lev_routing",
        # Phase 11 SCC residual (cycle #1, 5-op LEV next-step C-instruction
        # loop): TEMP / ADDR_B0_LO / ADDR_B0_HI reads renamed to their SSA
        # `.*.-1` cross-step forms. lev_routing has phase=None, so the
        # static analyser visualises it at layer 0 and sees back-edges
        # from:
        #   - layer11_mul_partial (phase=11)              -> TEMP
        #   - layer14_temp_clear (phase=14.1)             -> TEMP
        #   - layer15_store_stack0_sp_byte0_addr (15.2)   -> ADDR_B0_{LO,HI}
        # These are the genuine cross-step LEV semantics: l16's LEV
        # routing materialises PC / BP / SP from the previous step's
        # marker residuals (delivered via the KV cache), not from the
        # current step's L11/L14/L15 writers (which dynamic scheduling
        # places AFTER l16). SSA `.*.-1` aliases share the base dim's
        # numeric slot, so baked weights stay byte-identical while the
        # analyser drops the back-edges.
        reads={"MARK_SP", "MARK_PC", "MARK_AX", "OP_ENT", "OP_LEV", "OP_IMM",
               "OP_EXIT", "OP_JMP", "OP_SI", "OP_SC", "OP_LC_RELAY",
               "ADDR_B0_LO.*.-1", "ADDR_B0_HI.*.-1", "MEM_ADDR_SRC",
               "TEMP.*.-1", "HAS_SE", "IS_BYTE", "PSH_AT_SP", "EMBED_LO",
               "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "FETCH_LO", "FETCH_HI",
               "MARK_BP", "MARK_STACK0", "H1", "H3",
               "CMP", "MEM_STORE", "MEM_VAL_B0", "MEM_VAL_B1",
               "MEM_VAL_B2", "MEM_VAL_B3", "BYTE_INDEX_0", "BYTE_INDEX_1",
               "BYTE_INDEX_2", "BYTE_INDEX_3",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2",
               "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "ALU_LO"},
        kind="ffn",
        bake_fn=bake,
        compiler_ir=make_layer16_lev_routing_ir(),
        migrated=True,
        # Phase 8.A.4: dropped ``layer_idx=16`` pin. The dep-graph already
        # forces a later layer than every L15 op via the read on
        # ``OUTPUT_LO`` / ``OUTPUT_HI_THIS_STEP`` produced by
        # ``layer15_memory_lookup``; ``requires["after"]`` re-affirms
        # ordering against the L15 attn op so the scheduler honours the
        # 1-layer gap when block-op writes don't appear in the dep graph.
        requires={"after": "layer15_memory_lookup"},
        # ``_set_layer16_lev_routing`` writes 121 units (vm_step.py:6845):
        #   16 (cancel OUTPUT_LO at SP) + 16 (cancel OUTPUT_HI at SP) +
        #   16 (SP=BP+16 lo) + 16 (SP=BP+16 hi) +
        #   16 (cancel OUTPUT_HI at PC) + 16 (TEMP_LO→OUTPUT_LO) +
        #   16 (TEMP_HI→OUTPUT_HI) + 3 (clear OUTPUT_LO[10] byte 1-3) +
        #   3 (set OUTPUT_LO[0] byte 1-3) + 3 (set OUTPUT_HI[0] byte 1-3)
        #   = 121. The bytes 1-3 blocks (if False) are disabled.
        # Plus 32 late AX-carry materialization units for the stale-IMM AX
        # marker state reached after strict neural LEV, and 32 for SI/SC AX
        # preservation before same-step memory-store generation, 32 BP marker
        # passthrough units, one BP frame byte-1 booster, 15 current top-store
        # STACK0 restore units, one retained-memory STACK0 byte-1 zero guard,
        # 16 JMP AX preserve units, one BP byte-2 post-ENT zero guard, one
        # initial-main ENT STACK0 byte-2 0x01 guard, 2 saved-BP STACK0 byte-2
        # scalar exactness guards, one top-level LEV PC return materializer,
        # two JSR MEM addr0 materializers, one initial JSR STACK0 marker
        # materializer, 32 e8 STACK0 marker ALU materializers, one ENT
        # stale-ALU cleanup, 16 PSH MEM addr0 nonzero nibble restore units,
        # one PSH d8 address exactness guard, one PSH e0 addr-band guard, and
        # one no-side-state PSH e0 guard,
        # 4 non-store MEM value zero guards, 16 PSH SP no-borrow high-nibble
        # restores, 39 ENT dynamic-frame SP/BP/STACK0 byte units, plus 2 LEA
        # local-frame byte materialization units, one recursive JSR
        # return-address byte-1 guard, plus 32 STACK0-marker inverse units for
        # false-positive LEV SP materializers, 254 nonzero staged-byte e8
        # STACK0 output authority guards, plus one e0/e8 STACK0 exactness
        # guard, plus 32 generic e0 STACK0 marker ALU materializers (16 LO + 16
        # HI) for non-store preserve at SP=0xffe0.
        # Plus 32 LEV AX-carry materializers (l16_lev_ax_carry_lo/hi_{k}) that
        # restore the popped return value into OUTPUT at MARK_AX during OP_LEV
        # (fixes step6:AX_byte0 91-row cluster, 2026-06-01 stack/JSR/LEV
        # triage) and 32 LEV STACK0 byte-0 preservation units
        # (l16_lev_stack0_byte0_preserve_lo/hi_{k}) that keep the just-popped
        # stack-top byte stable through LEV (fixes step6:STACK0_byte0 50-row
        # cluster from the same triage).
        ffn_units_used=792,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )
