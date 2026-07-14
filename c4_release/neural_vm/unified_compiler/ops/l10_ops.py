"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import os
from dataclasses import replace
from typing import Mapping, Optional


# ---------------------------------------------------------------------------
# R-FRAME INCR-3 — the L25 tail register-EMISSION-FRAME guarantee collapse.
# RETIRED (P5, docs/P5_RFRAME_RETIRE_2026_07_13.md).
#
# The L25 tail bank (``_tail_bit32_result_correction_rules``) guaranteed each
# register frame byte via several ``range(256)`` byte-value-writeback banks that
# were PURE ENUMERATIONS of a single computed fact (``byte = f(source)``). INCR-3
# collapsed six of those pure-copy families to their computed per-nibble ROUTE
# form behind the unified ``C4_R_FRAME_TAIL`` super-switch (DEFAULT-ON), keeping
# the enumerated banks as a flag-OFF escape hatch.
#
# P5 RETIRE makes the collapse UNCONDITIONAL: the six routed families
# (``sp_pop_carry_byte2`` [SP], ``wide_mul_byte1`` [AX], and the four STACK0
# families ``stack0_store_loaded`` / ``stack0_pop_loaded`` / ``stack0_store_e8``
# / ``stack0_store_top_e0``) now emit ONLY their computed route form. The
# ``range(256)`` enumerated fallbacks, the ``C4_R_FRAME_TAIL`` /
# ``R_FRAME_TABLE`` / ``_r_frame_tail_enabled`` machinery, and the six per-family
# "M8" point-flags (``_sp_byte2_carry_computed_enabled`` / ``C4_SP_BYTE2_CARRY``,
# ``_stack0_*_computed_enabled`` / ``C4_STACK0_*_COMPUTED``,
# ``_wide_mul_byte1_computed_enabled`` / ``C4_WIDE_MUL_BYTE1_COMPUTED``) are
# DELETED. The DEFAULT model is UNCHANGED — the default was already the collapsed
# form, so the golden hash stays ``1c04c3fd`` — but the -478 tail units are now a
# real SOURCE-LOC deletion. This PERMANENTLY drops the ``C4_R_FRAME_TAIL=0``
# reversibility for these six families (the intended P5 trade).
# ---------------------------------------------------------------------------


def _nonfirst_psh_sp_fix_enabled() -> bool:
    """Flag for the non-first-PSH SP byte-0 over-correction fix.

    The L25 ``tail_sp_marker_byte0_f8_from_initial_stack_exact`` rule forces
    SP byte 0 = 0xF8 on any HAS_SE SP-marker row whose H1 distance pattern +
    MARK_SP gate clears its threshold. That is correct on the FIRST push of a
    program (input SP byte0 = 0x00, decrement result = 0xF8) and the
    JSR-bootstrap row, but it WRONGLY fires on every SUBSEQUENT push whose
    input SP byte0 is already 0xF8 (decrement result = 0xF0, not 0xF8),
    overwriting the correct 0xF0 the L6 decrement already produced with
    garbage byte 0 = 0x48. See the SP-decrement diagnosis appended to
    docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md.

    The fix adds an EMBED_LO+8 / EMBED_HI+15 NOT-blocker (the input SP byte0
    == 0xF8 evidence carried onto the MARK_SP row by the L3 carry-forward)
    so the rule is suppressed on a push whose incoming SP already ends in
    0xF8. Default ON; with ``C4_NONFIRST_PSH_SP_FIX=0`` the rule's conditions
    are byte-identical to the prior build.
    """
    return os.environ.get("C4_NONFIRST_PSH_SP_FIX", "1") != "0"


def _lea_local_e8_multilocal_guard_enabled() -> bool:
    """Flag for the multi-local LEA 0xE8 over-fire guard (var_mul / var_three).

    ``tail_lea_local_ax_marker_byte0_e8`` hardcodes the BP-8 effective-address
    low byte 0xE8 onto the LEA AX-marker row. Its FETCH discriminator
    (``FETCH_LO+8`` w=2.0, ``FETCH_HI+15`` w=0.2) is only ADDITIVE -- the 9-pt
    non-FETCH positive sum (MARK_AX+HAS_SE+OP_LEA+CMP+7+MEM_ADDR_SRC) already
    clears threshold=7 ALONE, so the 0xE8 writer over-fires on EVERY multi-local
    LEA. On a program's SECOND local (BP-16, imm=-16, correct low byte 0xE0) and
    THIRD local (BP-24, imm=-24, correct low byte 0xD8) it stamps 0xE8, so the
    two/three locals ALIAS to the same stack slot -- the var_mul step-6 / var_three
    step-9/10 AX 0xFFE8 instead of 0xFFE0 / 0xFFD8 (diag commit 18365452,
    tools/probe_var_fetch_band.py).

    The FETCH band carries the LEA immediate (NOT one-hot, magnitude ~40). The
    imm=-8 signature is FETCH_LO nib 8 + FETCH_HI nib F (15); imm=-16 has
    FETCH_LO nib 0 (the low nibble of 0xF0); imm=-24 has FETCH_HI nib E (14) and
    FETCH_LO nib 8 again. Adding strong NOT-blockers on the DISTINGUISHING
    nibbles -- ``FETCH_LO+0`` (the imm=-16 low-nibble) and ``FETCH_HI+14`` (the
    imm=-24 high-nibble) -- makes the imm=-8 fire a genuine requirement: on a
    2nd/3rd local LEA the competing nibble carries the ~40 broadcast, the -10
    blocker drives the sum well below 7, and the 0xE8 writer stays silent so the
    correct 0xE0 / 0xD8 byte (produced by the L8 effective-address compute and
    relayed through OUTPUT) survives.

    This is the load-bearing PER-STEP component the multi-local fix needs.
    GPU-confirmed (campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``,
    spec_k=0, #310): turning this ON moves var_mul's divergence step 6 -> 9 (the
    step-6 &b LEA 0xFFE8->0xFFE0 alias is RESOLVED), i.e. the per-step LEA
    disambiguator works; the residual step-9 LI value-load is the deeper
    L13/L15 CAM root tracked separately. The ROOT-A axmark + ROOT-B isbyte
    lev_routing hardens (already landed) removed the BP-byte1 framing desync, so
    PC stays correct and this LEA value-byte fix is now the active per-step lever.

    DEFAULT tracks ``no_stack0_emit_enabled()``: ON in the 30-token campaign
    config (the multi-local LEA frame), byte-identical golden (the two FETCH
    NOT-blocker terms omitted) otherwise. Force with
    ``C4_LEA_LOCAL_E8_MULTILOCAL_GUARD=0/1``. The legit imm=-8 LEA byte-0 0xE8
    emit (FETCH_LO+0 ~= 0, FETCH_HI+14 ~= 0) is unaffected; flag-OFF the golden
    build is bit-for-bit unchanged (``4958b35b``).
    """
    from .shared import no_stack0_emit_enabled

    forced = os.environ.get("C4_LEA_LOCAL_E8_MULTILOCAL_GUARD")
    if forced is not None:
        return forced != "0"
    return no_stack0_emit_enabled()


def _tail_lea_e8_divmod_guard_enabled() -> bool:
    """Flag for the 0xE8/744 sentinel-slam guard on DIV/MOD result rows
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_TAIL_LEA_E8_DIVMOD_GUARD=0``; only active when the STACK0 emission is
    dropped, i.e. ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; the bulk of the residual
    div/mod fails of the ``got_ax == 744`` pattern, e.g. ``462/13 -> 744``,
    ``390 % 19 -> 744`` -- the "0xE8 (744) slam" the
    ``divmod_axcarry_clear_enabled`` note explicitly flags as a SEPARATE
    downstream-tail corruptor):

    ``tail_lea_local_ax_marker_byte0_e8`` (the L25 tail bank, this module)
    hardcodes the BP-8 effective-address low byte 0xE8 onto a LEA AX-marker
    row, gated on ``MARK_AX + HAS_SE + OP_LEA + CMP+7 + FETCH_LO+8 +
    MEM_ADDR_SRC`` (threshold 7). In the 30-token campaign layout the DIV/MOD
    RESULT AX-marker row also carries ``MARK_AX=1 + HAS_SE=1 + MEM_ADDR_SRC=1``
    (BUILT-dim probe ``tools/probe_outband2_blk41_attrib.py``), which clears the
    rule's effective threshold even though ``OP_LEA ~= 0`` (the lowered AND uses
    a 1e9 ``MARK_AX`` / ``CONST`` pair so the small ``OP_LEA`` term cannot veto).
    The 1e6-strength 0xE8 writer then SIGN-INVERTS OUTPUT_LO[0] to ~-6.8e7 on
    the DIV/MOD result row -> the emitted AX byte 0 is 0xE8 (744) instead of the
    quotient/remainder.

    The fix adds ``OP_DIV`` / ``OP_MOD`` as HARD NOT-blockers (-1e9) so the
    sentinel writer can never fire on a DIV/MOD result row. On the legit LEA
    byte-0 0xE8 emit (OP_DIV == OP_MOD == 0) the rule is byte-identical.

    DEFAULT ON. Opt-out via ``C4_TAIL_LEA_E8_DIVMOD_GUARD=0`` (the byte-identical
    path: flag-OFF or ``C4_NO_STACK0_EMIT=0`` are both byte-identical to golden
    ``4958b35b``). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_TAIL_LEA_E8_DIVMOD_GUARD`` can A/B
    it inside the campaign config.
    """
    from .shared import no_stack0_emit_enabled

    return (
        os.environ.get("C4_TAIL_LEA_E8_DIVMOD_GUARD", "1") != "0"
        and no_stack0_emit_enabled()
    )


def _tail_lea_e8_arith_guard_enabled() -> bool:
    """Flag for the 0xE8/744 sentinel-slam guard on ADD/SUB/absdiff result
    rows (#309). The arith extension of ``_tail_lea_e8_divmod_guard_enabled``:
    DEFAULT ON in the campaign config — opt-out via
    ``C4_TAIL_LEA_E8_ARITH_GUARD=0``; only active when the STACK0 emission is
    dropped, i.e. ``C4_NO_STACK0_EMIT=1``.

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the residual ADD fails
    of the ``0xE8`` byte-0 pattern (e.g. 744=0x2E8, 488=0x1E8), the SUB
    residual, and absdiff (0/25, diverges step 11 on ax 0xFFE0 -> 0xFFE8 —
    the ``|a-b|`` body is a SUB result row).

    ``tail_lea_local_ax_marker_byte0_e8`` (the L25 tail bank, this module)
    hardcodes the BP-8 effective-address low byte 0xE8 onto a LEA AX-marker
    row, gated on ``MARK_AX + HAS_SE + OP_LEA + CMP+7 + FETCH_LO+8 +
    MEM_ADDR_SRC`` (threshold 7). In the 30-token campaign layout the ADD/SUB
    RESULT AX-marker row also carries ``MARK_AX=1 + HAS_SE=1 + MEM_ADDR_SRC=1``
    (the same signature the DIV/MOD guard documents), which clears the rule's
    effective threshold even though ``OP_LEA ~= 0`` (the lowered AND uses a
    1e9 ``MARK_AX`` / ``CONST`` pair so the small ``OP_LEA`` term cannot veto).
    The 1e6-strength 0xE8 writer then SIGN-INVERTS OUTPUT_LO[0] on the ADD/SUB
    result row -> the emitted AX byte 0 is 0xE8 (744) instead of the sum /
    difference.

    The fix adds ``OP_ADD`` / ``OP_SUB`` as HARD NOT-blockers (-1e9) so the
    sentinel writer can never fire on an ADD/SUB result row. absdiff is
    covered by ``OP_SUB`` (its ``a > b ? a-b : b-a`` body is a SUB). On the
    legit LEA byte-0 0xE8 emit (OP_ADD == OP_SUB == 0) the rule is
    byte-identical.

    DEFAULT ON. Opt-out via ``C4_TAIL_LEA_E8_ARITH_GUARD=0`` (the
    byte-identical path: flag-OFF or ``C4_NO_STACK0_EMIT=0`` are both
    byte-identical to golden ``4958b35b``). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_TAIL_LEA_E8_ARITH_GUARD`` can
    A/B it inside the campaign config.
    """
    from .shared import no_stack0_emit_enabled

    return (
        os.environ.get("C4_TAIL_LEA_E8_ARITH_GUARD", "1") != "0"
        and no_stack0_emit_enabled()
    )


def _tail_lea_e8_arith_guard_sharp_enabled() -> bool:
    """SHARP per-step replacement for the ADD/SUB arith-guard NOT-blockers
    (#325, the var_update LEA-after-store byte-0 staleness root).

    The wall this lifts (verified AUTOREGRESSIVE spec_k=0, BUILT dims, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; tools/
    probe_pcframe_tokdump.py + probe_sistore_lea_ar.py on var_update id325
    ``x=50; x=x+7; return x``): the ``return x`` body's ``LEA -8`` (step 14, the
    address-of-x) IMMEDIATELY FOLLOWS the ``x = x + 7`` ADD+SI store. Its AX
    register-dump byte-0 should be the local frame address 0xE8 (0xFFE8), but it
    comes out as the STALE stored value 0x39 (57) -> full_trace diverges at
    step 14. NOT a framing/pc desync (the 30-tok frame is intact: every
    inter-PC gap == 30, PC decodes at every step); it is an AX-VALUE root.

    Root (block-by-block residual scan, ``tools/probe_sistore_lea_axdiverge.py``
    -> ``probe_opadd.py``): the L25-tail 0xE8 byte-0 writer family
    (``tail_lea_local_ax_marker_byte0_e8`` + the campaign keystone
    ``tail_lea_local_ax_byte0_e8_alubp_memsp`` + the amplifier
    ``tail_lea_local_ax_byte0_amplify_*``, and the l16 sibling
    ``l16_lea_local_ax_byte0_hi_e``) carries an ``("OP_ADD", -1e9)`` /
    ``("OP_SUB", -1e9)`` NOT-blocker (the #309 arith-result sentinel guard). On
    the var_update step-14 LEA AX-marker row the prior ADD's ``OP_ADD`` opcode
    broadcast PERSISTS at ~+0.0116 (opcode markers are NOT one-hot in-step — the
    ``op_ent_in_step_broadcast`` corruptor family). ``-1e9 * 0.0116 = -1.16e7``
    single-handedly VETOES the legitimate 0xE8 writer even though ``OP_LEA`` is a
    strong +5.23 -> the byte-0 OUTPUT defaults to the stale 0x39 that a competing
    relay stamps. Confirmed causal: ``C4_TAIL_LEA_E8_ARITH_GUARD=0`` (the
    blockers removed) flips step-14 byte-0 0x39 -> 0xE8 (got_ax 57 -> 232) AND
    leaves the genuine ADD result row (step 12) byte-identical (the
    multiplicative ``OP_LEA`` gate alone keeps it off).

    THE CLEAN DISCRIMINATOR (measured BUILT dims, blocks 40/41): the per-step
    FETCHED opcode lives in ``OPCODE_BYTE_LO/HI`` as a SHARP one-hot that does
    NOT persist cross-step. A genuine ADD result row (opcode 25 = 0x19) has
    ``OPCODE_BYTE_LO+9 == 1.0`` EXACTLY; the LEA-after-ADD row (opcode 0 = 0x00)
    has ``OPCODE_BYTE_LO+9 == 0.0`` EXACTLY (whereas the legacy ``OP_ADD`` flag
    leaks +0.0116 there). SUB (opcode 26 = 0x1A) is ``OPCODE_BYTE_LO+10``. So
    swapping the leaky ``("OP_ADD"/"OP_SUB", -1e9)`` blockers for
    ``("OPCODE_BYTE_LO+9"/"+10", -1e9)`` keeps the genuine-arith-row protection
    (full -1e9 veto when the step IS an ADD/SUB) while NEVER vetoing a LEA row
    that merely FOLLOWS an arith step. Same fix pattern as the l15 head-14 /
    l16 LEV ``OPCODE_BYTE_LO+8 == LEV`` per-step opcode gate.

    DEFAULT tracks ``_tail_lea_e8_arith_guard_enabled()``: ON in the 30-token
    campaign config (where the guard itself is active), so the byte-identical
    paths are preserved -- flag-OFF (``=0``), ``C4_NO_STACK0_EMIT=0``, or the
    parent guard OFF are all byte-identical to the pre-fix default (and to golden
    ``4958b35b``: the SHARP variant only differs INSIDE the campaign
    arith-guard-ON branch, which the 35-token golden never enters). Dedicated
    kill-switch so ``tools/flag_regression_gate.py --flag
    C4_TAIL_LEA_E8_ARITH_GUARD_SHARP`` can A/B it inside the campaign config.
    """
    forced = os.environ.get("C4_TAIL_LEA_E8_ARITH_GUARD_SHARP")
    if forced is not None:
        return forced != "0" and _tail_lea_e8_arith_guard_enabled()
    # Default ON wherever the parent arith guard is active (campaign config).
    return _tail_lea_e8_arith_guard_enabled()


# Per-step ADD/SUB opcode low-nibble one-hots (OPCODE_BYTE_LO offsets). ADD is
# C4 opcode 25 = 0x19 -> low nibble 0x9; SUB is 26 = 0x1A -> low nibble 0xA(10).
# Sharp (non-cross-step-persistent) replacements for the leaky OP_ADD / OP_SUB
# residue flags in the L25-tail 0xE8 byte-0 writer arith guard. See
# ``_tail_lea_e8_arith_guard_sharp_enabled``.
_ARITH_GUARD_ADD_OPCODE_LO = "OPCODE_BYTE_LO+9"
_ARITH_GUARD_SUB_OPCODE_LO = "OPCODE_BYTE_LO+10"


def _arith_guard_addsub_blockers() -> tuple:
    """Return the ADD/SUB NOT-blocker condition pair for the L25-tail 0xE8
    byte-0 writer family.

    Both forms keep the genuine-arith-result-row protection (the #309 sentinel
    slam guard). The SHARP form (default, campaign config) keys on the per-step
    fetched opcode (``OPCODE_BYTE_LO+9`` ADD / ``+10`` SUB) instead of the leaky
    cross-step ``OP_ADD`` / ``OP_SUB`` broadcast, so a LEA row that merely
    FOLLOWS an arith step is no longer spuriously vetoed (#325, var_update). The
    LEGACY form (``C4_TAIL_LEA_E8_ARITH_GUARD_SHARP=0``) is byte-identical to the
    pre-fix default. Caller must already be inside an
    ``_tail_lea_e8_arith_guard_enabled()`` branch.
    """
    if _tail_lea_e8_arith_guard_sharp_enabled():
        return (
            (_ARITH_GUARD_ADD_OPCODE_LO, -1_000_000_000.0),
            (_ARITH_GUARD_SUB_OPCODE_LO, -1_000_000_000.0),
        )
    return (
        ("OP_ADD", -1_000_000_000.0),
        ("OP_SUB", -1_000_000_000.0),
    )


def _ax_byte1_signext_lea_enabled() -> bool:
    """Flag for the AX byte-1 sign-extension delivery on a negative LEA-local
    frame address (#343 — the #325 byte-0 follow-up). DEFAULT ON wherever the
    campaign STACK0 emission is dropped (``C4_NO_STACK0_EMIT=1``); opt-out via
    ``C4_AX_BYTE1_SIGNEXT_LEA=0``.

    The wall this lifts (verified AUTOREGRESSIVE spec_k=0, BUILT dims, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; tools/
    probe_axff_signext.py + probe_axff_nuke.py + probe_axff_adder.py on
    var_update id325 ``x=50; x=x+7; return x``): after the #325 byte-0 fix the
    ``return x`` body's ``LEA -8`` (step 14, immediately following the
    ``x = x + 7`` ADD+SI store) emits AX byte-0 = 0xE8 CORRECTLY, but byte-1 = 0x00
    instead of 0xFF -> got_ax 232 (0x000000E8) vs the oracle's 65512 (0xFFE8, the
    sign-extension of the negative stack address). full_trace diverges at step 14.

    ROOT (block-by-block residual scan): the byte-1 OUTPUT is sign-extended to
    0xFF CORRECTLY at physical block 35 (OUTPUT_LO+15/HI+15 ~+4585) and survives
    through block 44 -- EXACTLY as it does on the WORKING negative-LEA dumps
    (steps 2/6/8, all emit byte-1 = 0xFF). But at block 45 the
    ``l10_add_high_byte_adder`` (the multi-byte ADD high-byte completer, 128
    units) FALSE-FIRES: on the step-14 LEA-after-ADD byte-1 row the L13 add relay
    has spuriously stamped ``TEMP+12`` = 1.0 (the campaign ADD emit discriminator
    -- the LEA immediately follows the ``x+7`` ADD, the same cross-step opcode
    residue family as #325). With ``STACK0_BYTE_VAL_1_LO`` empty (a1=0),
    ``ADDR_B1_LO+0`` lit (b1=0) and no ``CARRY+1`` the adder's ``a0_b0_c0`` rule
    fires and OVER-writes byte-1 = 0x00 at ~2e16 strength, nuking the sign-ext
    0xFF. On the WORKING dumps (steps 2/6/8) ``TEMP+12`` is ABSENT so the adder is
    dark and the block-35 0xFF survives -- the crisp fire/no-fire discriminator.

    THE CLEAN, SAFE DISCRIMINATOR (measured BUILT dims at the byte-1 predictor
    row, block 44, tools/probe_axff_addsafe.py over 7 multi-byte ADD programs):
    the negative-LEA sign-ext byte-1 row carries ``AX_CARRY_LO+15`` ~3.0 AND
    ``AX_CARRY_HI+15`` ~3.0 (the sign-extension carry -- the SAME 0xFF flag the
    existing ``l14_add_byte1_high_zero_cleanup`` already keys its ``AX_CARRY_HI+15``
    NOT-blocker on) AND ``OP_LEA`` ~0.22 (the LEA opcode residue). On EVERY genuine
    ADD byte-1 emit row (``TEMP+12`` = 1.0, the rows the adder MUST fire on)
    ``AX_CARRY_LO+15 == AX_CARRY_HI+15 == OP_LEA == 0.0`` EXACTLY -- so the triple
    conjunction is present ONLY on the sign-ext LEA dump, never on a real ADD
    result (incl. operand-B-nibble-0xF cases like 4095+1: the genuine ADD emit row
    has AX_CARRY+15 == 0; the only AX_CARRY+15-lit rows there are NON-ADD with
    TEMP+12 == 0, where the adder is already dark).

    FIX (mirrors the #325 SHARP-guard philosophy + the existing
    ``l14_add_byte1_high_zero_cleanup`` AX_CARRY_HI+15 sign-ext guard): add a
    sign-ext NOT-blocker pair (``AX_CARRY_LO+15`` / ``AX_CARRY_HI+15``, strong
    negative) to EVERY ADD high-byte adder rule (campaign config only). On a
    genuine ADD row both are 0 -> ZERO contribution (byte-identical to the current
    adder firing margin). On the sign-ext LEA dump row each is ~3.0 -> a large
    negative term hard-sinks the adder below threshold, so it never fires and the
    upstream block-35 0xFF sign-extension survives to the LM head -> byte-1 = 0xFF.
    This PREVENTS the 2e16 nuke (rather than out-writing it), so no astronomical
    write magnitude is needed.

    DEFAULT tracks ``operand_from_memsp_enabled()`` (the adder's own campaign
    branch): the blockers are added ONLY in the 30-token campaign config and ONLY
    when this flag is on, so flag-OFF (``=0``), ``C4_NO_STACK0_EMIT=0``, or the
    35-token golden build are all byte-identical to the pre-fix default (golden
    ``4958b35b`` never enters the campaign adder branch). Dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_AX_BYTE1_SIGNEXT_LEA`` can A/B it.
    """
    from .shared import no_stack0_emit_enabled

    return (
        os.environ.get("C4_AX_BYTE1_SIGNEXT_LEA", "1") != "0"
        and no_stack0_emit_enabled()
    )


def _ax_byte1_signext_lea_blockers() -> tuple:
    """Return the sign-ext NOT-blocker pair for the ADD high-byte adder rules.

    Empty (no extra conditions) unless the campaign sign-ext fix is active, so
    the flag-OFF / golden adder is byte-identical. When active, returns the
    ``AX_CARRY_LO+15`` / ``AX_CARRY_HI+15`` sign-extension hard-blockers (each
    ~3.0 on the negative-LEA dump row, 0.0 on a genuine ADD result row). See
    ``_ax_byte1_signext_lea_enabled`` for the full root + safety analysis.
    """
    if not _ax_byte1_signext_lea_enabled():
        return ()
    # Each AX_CARRY_*+15 reads ~3.0 on the sign-ext row -> -5000*3.0 = -15000
    # per dim (-30000 combined), decisively sinking the adder's ~570-pt firing
    # margin; reads 0.0 on every genuine ADD emit row -> no effect there.
    return (
        ("AX_CARRY_LO+15", -5_000.0),
        ("AX_CARRY_HI+15", -5_000.0),
    )


def _tail_lea_e8_ent_guard_enabled() -> bool:
    """Flag for the ENT-step AX-dump 0xE8/0x02 (744) sentinel-slam guard (#311).

    The wall this lifts (verified spec_k=0 + bit-exact cpu_full_trace, BUILT
    dims, the DEFAULT 30-token config ``C4_NO_STACK0_EMIT=1
    C4_OPERAND_FROM_MEMSP=1``; tools/_probe_ifvar436_ent.py on var_simple id250
    ``x=990`` and if_var id436 ``x=66; if(x>24)``): on the **main ENT step**
    (step 1, the function prologue ``ENT 8`` that immediately follows the
    bootstrap ``JSR main``) the AX register MUST preserve the carried prior AX
    (``AX_CARRY`` == 0 here), but the AX dump emits **744 = 0x02E8** — byte-0 =
    0xE8 + byte-1 = 0x02 — so the per-step ``full_trace`` verdict diverges at
    step 1 with ``got_ax=744`` vs ``oracle_ax=0``. This gates the ENTIRE
    var_simple / if_var / (and every ``main``-ENT) cluster at the FIRST step.

    ROOT (two L25-tail ``_tail_bit32_result_correction_rules`` writers fire on
    the ENT-step AX rows where ``AX_CARRY`` is live):
      * ``tail_lea_local_ax_marker_byte0_e8`` (the BP-8 LEA byte-0 0xE8 writer,
        strength 1e6) fires on the ENT AX-marker row: it is ``scope="mark ==
        AX"`` so the lowered 1e9 ``MARK_AX`` term dominates and the ``OP_LEA``
        (== 0 on the ENT step) cannot veto — exactly the same threshold-clear
        the DIV/MOD and ADD/SUB (#309) sentinel guards already document.
      * ``tail_ax_add_byte1_missing_stack_high_02`` (a constant-0x02 byte-1
        materializer, strength 5000) fires on the ENT AX byte-1 row, driven by
        its ``FETCH_HI+1`` term (the fetched ENT instruction's high byte). It
        has no live ADD use (its own comment: "no live ADD smoke test exercises
        this" — byte-1 low nibble is 0 for every ADD target), so suppressing it
        on the carried-AX row is safe.

    THE FIX (mirrors the DIV/MOD + ADD/SUB sentinel guards — NOT-blockers
    appended to the EXISTING rules, so the width-sensitive 2059-rule tail bank
    keeps its rule count and is byte-identical when the guard is off):
      * 0xE8 byte-0 writer: add ``OPCODE_BYTE_LO+6`` (the per-step FETCHED ENT
        opcode low-nibble one-hot; ENT = C4 opcode 6 = 0x06) as a hard
        NOT-blocker. Measured CRISP at the AX-marker row (== 1.0 on the ENT
        step, == 0.0 on a genuine LEA step, which has ``OPCODE_BYTE_LO+0``), so
        the legit BP-8 LEA byte-0 0xE8 emit is byte-identical.
      * 0x02 byte-1 writer: add ``AX_CARRY_LO+0`` / ``AX_CARRY_HI+0`` (the
        carried-prior-AX-present signal; both ~3.0 on the ENT byte-1 row where
        AX_CARRY holds the preserved value 0, and 0.0 on every genuine
        freshly-computed value byte-1 row — measured on the LEA-0xff and IMM-0x02
        byte-1 rows, both cold) as hard NOT-blockers, so the writer cannot stamp
        0x02 over the carried AX.

    With both writers vetoed the upstream ``ent_ax_passthrough``
    (``AX_CARRY -> OUTPUT`` on the ENT AX row) survives to the LM head and the
    AX dump emits the carried 0 -> step-1 ``got_ax`` 744 -> 0.

    DEFAULT ON in the 30-token config. Opt-out via ``C4_TAIL_LEA_E8_ENT_GUARD=0``
    (the byte-identical path: flag-OFF or ``C4_NO_STACK0_EMIT=0`` are both
    byte-identical to the pre-fix build). Dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_TAIL_LEA_E8_ENT_GUARD`` can A/B it.
    """
    from .shared import no_stack0_emit_enabled

    return (
        os.environ.get("C4_TAIL_LEA_E8_ENT_GUARD", "1") != "0"
        and no_stack0_emit_enabled()
    )


# ENT is C4 opcode 6 = 0x06 -> low nibble 0x6. The per-step FETCHED opcode
# low-nibble one-hot ``OPCODE_BYTE_LO+6`` is +1.0 EXACTLY on the ENT AX-marker
# row and 0.0 on a genuine LEA AX-marker row (which is ``OPCODE_BYTE_LO+0``) --
# the SAME sharp (non-cross-step-persistent) discriminator the #325 var_update
# arith guard uses for ADD/SUB.
_ENT_GUARD_OPCODE_LO = "OPCODE_BYTE_LO+6"


def _tail_lea_e8_ent_byte0_blockers() -> tuple:
    """ENT NOT-blocker for the 0xE8 byte-0 writer (marker row).

    Empty unless the ENT guard is active (byte-identical off). When active,
    hard-blocks the writer on the per-step FETCHED ENT opcode one-hot so it
    cannot stamp the BP-8 0xE8 on the carried-AX ENT-marker row. See
    ``_tail_lea_e8_ent_guard_enabled``.
    """
    if not _tail_lea_e8_ent_guard_enabled():
        return ()
    return ((_ENT_GUARD_OPCODE_LO, -1_000_000_000.0),)


def _tail_lea_e8_ent_byte1_blockers() -> tuple:
    """ENT NOT-blockers for the constant-0x02 byte-1 materializer (byte row).

    Empty unless the ENT guard is active (byte-identical off). When active,
    hard-blocks the writer when the carried prior AX is present
    (``AX_CARRY_LO+0`` / ``AX_CARRY_HI+0`` both ~3.0 on the ENT byte-1 row, 0.0
    on every genuine freshly-computed value byte-1), so it cannot stamp 0x02
    over the carried AX. See ``_tail_lea_e8_ent_guard_enabled``.
    """
    if not _tail_lea_e8_ent_guard_enabled():
        return ()
    return (
        ("AX_CARRY_LO+0", -1_000_000_000.0),
        ("AX_CARRY_HI+0", -1_000_000_000.0),
    )


def _lea_e0d8_fetch_dominate_enabled() -> bool:
    """Flag for boosting the campaign ``e0_fetch_memsp`` (BP-16 0xE0) /
    ``d8_fetch_memsp`` (BP-24 0xD8) LEA byte-0 writers' strength so they DOMINATE
    the BP-8 ``e8_alubp_memsp`` (0xE8) writer on the rows where they fire (the
    var_three multi-local 2nd/3rd-local LEA byte-0 root). DEFAULT ON wherever the
    campaign LEA byte-0 relay is active; opt-out via
    ``C4_LEA_E0D8_FETCH_DOMINATE=0``.

    The wall this lifts (verified AUTOREGRESSIVE spec_k=0, BUILT dims, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; tools/
    _probe_vt_step6_tail.py + GPU full_trace on var_three id300 / var_mul id275):
    var_three full_trace diverges at **step 6** — the ``LEA &b`` (BP-16, imm=-16,
    correct AX byte-0 = 0xE0) emits 0xE8 (= &a, BP-8) so the 2nd local ALIASES the
    1st (got_ax 0xFFE8 vs oracle 0xFFE0). var_mul id275 has the SAME bug (baseline
    GPU FAIL step 6, 0xFFE8 vs 0xFFE0). The block-by-block residual scan shows
    OUTPUT byte-0 is the correct **0xE0 through block 41** then FLIPS to 0xE8 at
    **block 42** (the L25 tail bank): OUTPUT_LO[0] +758 -> -829M, OUTPUT_LO[8]
    +558 -> +829M.

    ROOT: the ``e8_alubp_memsp`` 0xE8 writer keys ONLY on ``0.2 * ALU_HI+15``,
    which the keystone (``_lea_byte0_memsp_relay_enabled``) assumed was ~+5.5 on
    BP-16/BP-24 frames. In var_three's DEEPER 3-local frame the ``&b`` LEA's
    ``ALU_HI+15`` is ~+72.92 (NOT +5.5), so ``0.2*72.92 + ~3 = ~17.6 >= threshold
    17``: the 0xE8 writer (strength 1e6) FIRES on the ``&b`` row and — being a TIE
    at 1e6 with the CORRECT ``e0_fetch_memsp`` 0xE0 writer — the 0xE8 wins the
    block-42 argmax.

    THE FIX — a strength DOMINANCE rather than a blocker. The e0/d8 writers
    ALREADY carry the precise multi-local FETCH discriminator that EXCLUDES
    func_identity: ``e0_fetch_memsp`` REQUIRES ``FETCH_LO+0`` AND ``FETCH_HI+15``
    (each weight 8, ABSENCE drops below threshold), and func_identity's ``&x``
    carries ``FETCH_HI`` nibble **1** (NOT 15) so e0_fetch is dark there (the
    keystone doc's own exclusion). So raising ONLY the e0/d8 write strength
    (1e6 -> 4e6) makes 0xE0/0xD8 out-vote the 0xE8 tie on var&b/&c WITHOUT
    touching any row where e0/d8 don't already fire:

      * var ``&a`` (imm=-8): e0/d8 dark (FETCH_LO+0 absent), only e8_alubp fires
        -> 0xE8 byte-identical.
      * var ``&b`` (imm=-16): e0_fetch (4e6) + e8_alubp (1e6) -> 0xE0 WINS.
      * var ``&c`` (imm=-24): d8_fetch (4e6) + e8_alubp (1e6) -> 0xD8 WINS.
      * func_identity ``&x`` (imm=-8, FETCH_HI nib 1): e0/d8 DARK (require
        FETCH_HI+15) -> only e8_alubp fires -> 0xE8 byte-identical (no regression
        — the earlier FETCH_LO+0 NOT-blocker approach FAILED here because func's
        ``&x`` ALSO carries FETCH_LO+0; keying off the e0/d8 FIRE condition,
        which needs FETCH_HI+15, is the func-safe discriminator).

    DEFAULT tracks ``_lea_byte0_memsp_relay_enabled()`` (the e0/d8 writers' own
    campaign branch): the strength bump is applied ONLY in the 30-token campaign
    config and ONLY when this flag is on, so flag-OFF (``=0``),
    ``C4_NO_STACK0_EMIT=0``, or the 35-token golden build are all byte-identical
    to the pre-fix default (golden ``4958b35b`` never emits these writers).
    Dedicated kill-switch so ``tools/flag_regression_gate.py --flag
    C4_LEA_E0D8_FETCH_DOMINATE`` can A/B it inside the campaign config.
    """
    forced = os.environ.get("C4_LEA_E0D8_FETCH_DOMINATE")
    if forced is not None:
        return forced != "0" and _lea_byte0_memsp_relay_enabled()
    return _lea_byte0_memsp_relay_enabled()


def _lea_e0d8_fetch_strength() -> float:
    """Return the OUTPUT-write strength for the campaign ``e0_fetch_memsp`` /
    ``d8_fetch_memsp`` LEA byte-0 writers. 4e6 when the dominance fix is active
    (so 0xE0/0xD8 out-vote the BP-8 0xE8 ``e8_alubp_memsp`` writer's 1e6 tie on
    the multi-local 2nd/3rd-local LEA), else the pre-fix 1e6 (byte-identical).
    See ``_lea_e0d8_fetch_dominate_enabled`` for the full root + safety analysis.
    """
    return 4_000_000.0 if _lea_e0d8_fetch_dominate_enabled() else 1_000_000.0


def _lea_e8_first_ent_gate_enabled() -> bool:
    """Flag for the func re-read-LEA ``&b`` byte-0 0xE8 over-fire FIX — gate the
    ``e8_alubp_memsp`` BP-8 0xE8 writer on the FIRST-LEA-after-ENT signal
    (``OP_ENT`` residue), so it stops slamming 0xE8 onto the SECOND-local re-read
    LEA (``&b``, BP-16, want 0xE0) in func_add / func_mul / func_max / func_min /
    absdiff (~125, the func/absdiff step-11 0xFFE0->0xFFE8 cluster).

    The wall this lifts (verified AUTOREGRESSIVE spec_k=0, BUILT dims, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; tools/
    _probe_lea_opent_cluster.py + run_1096_canonical full_trace on ids 575/600/
    650/675/1046): the func body reads a 2nd local ``&b`` (``LEA 16`` in a
    2-param frame, BP-16, effective-address byte-0 = 0xE0). The L8 ``lea_lo`` ALU
    ALREADY computes the CORRECT 0xE0 (pre-slam OUTPUT_LO argmax = cell 0 through
    block 43). But the ``e8_alubp_memsp`` writer keys only on ``0.2*ALU_HI+15``,
    which on func's ``&b`` is ~+86.9 (IDENTICAL to ``&a`` — NOT the assumed ~+5.5),
    so the 0xE8 writer (strength 1e6) FIRES on ``&b`` and stamps 0xE8 over the
    genuine 0xE0 -> step-11 ``got ax=0xFFE8`` vs oracle ``0xFFE0``. The
    ``e0_fetch_memsp`` 0xE0 writer does NOT rescue it because func encodes ``LEA
    16`` as ``FETCH_HI`` nibble **1** (NOT 15), so e0_fetch is DARK on func's
    ``&b`` (it was tuned for var's ``&b`` which carries FETCH_HI+15).

    THE DISCRIMINATOR (the func-safe key the FETCH band lacks): ``OP_ENT`` residue
    at the LEA AX-marker row separates the FIRST-LEA-after-ENT from the later
    re-read LEAs CLEANLY across the whole cluster (probe block-43 input):

      * FIRST LEA after an ENT (``&a`` / ``&x`` / nested first-LEA): OP_ENT ~+1.19
        (the in-step ENT-broadcast residue one VM step after the prologue ENT).
        These are exactly the rows where the 0xE8 slam is CORRECT (BP-8) or
        load-bearing (func_identity ``&x``: genuine 0xE0 lifted to 0xE8).
      * SECOND / re-read LEA (func ``&b``, var ``&a``/``&b`` re-reads): OP_ENT
        ~+0.013 (the residue has decayed). These are exactly the rows where the
        0xE8 slam is WRONG (func ``&b`` genuine 0xE0) or a no-op (var re-read
        already 0xE8 / handled by e0_fetch).

    THE FIX: add ``("OP_ENT", W)`` to the e8 writer's conditions and raise the
    threshold by ``0.6*W`` (the cut sits at the ~0.6 midpoint between the +1.19
    fire and the +0.013 veto). With ``W = 100`` the first-LEA score gains ~+59
    over threshold and the re-read score loses ~-59 -> vetoed. The genuine 0xE0
    survives on func ``&b``; func_identity ``&x`` / var first-LEA / nested
    first-LEA keep the slam (OP_ENT ~1.19). The var / func_square re-read rows the
    slam is silenced on are byte-identical (their pre-slam value is already 0xE8
    or, for var ``&b``, delivered by e0_fetch).

    DEFAULT tracks ``_lea_byte0_memsp_relay_enabled()`` (the e8 writer's own
    campaign branch): the OP_ENT gate is applied ONLY in the 30-token campaign
    config and ONLY when this flag is on, so flag-OFF (``=0``),
    ``C4_NO_STACK0_EMIT=0``, or the 35-token golden build are all byte-identical
    to the pre-fix default (golden never emits this writer). Dedicated
    kill-switch so ``tools/flag_regression_gate.py --flag
    C4_LEA_E8_FIRST_ENT_GATE`` can A/B it inside the campaign config.
    """
    forced = os.environ.get("C4_LEA_E8_FIRST_ENT_GATE")
    if forced is not None:
        return forced != "0" and _lea_byte0_memsp_relay_enabled()
    return _lea_byte0_memsp_relay_enabled()


def _lea_e8_nested_ent_axdump_enabled() -> bool:
    """Flag for the NESTED callee-ENT AX-dump over-fire FIX (#342) — hard-block
    the ``e8_alubp_memsp`` BP-8 0xE8 writer on the FETCHED-ENT-opcode one-hot so
    it stops stamping 0xE8 over the carried AX on a genuine callee ENT step in a
    NESTED call (``nested_sumsq`` / ``nested_quad``, ~25).

    DEFAULT OFF (opt in via ``C4_NESTED_ENT_AXDUMP=1``) — unlike the sibling
    campaign gates, this fix does NOT track the campaign floor, so the golden
    default build (flag unset) is byte-identical.

    The wall this lifts (verified bit-exact ``cpu_full_trace --spec-k 0``, BUILT
    dims, the DEFAULT 30-token campaign config; ``nested_sumsq`` id975
    ``int square(int x){return x*x;} int sum_squares(int a,int b){return
    square(a)+square(b);} main{sum_squares(2,2);}``): the run diverges at the
    inner ``square()`` callee ENT step, where the AX register MUST preserve the
    carried prior AX (the argument value that was just pushed / the running sum),
    but the AX dump emits **0xFFE8** — byte-0 = 0xE8, the stale BP-8 LEA local
    address from the first-level frame's ``&a`` LEA — instead of the carried
    value.

    ROOT (the SAME class as #345's if_var main-ENT 744-leak, one nesting level
    deeper): the campaign keystone ``tail_lea_local_ax_byte0_e8_alubp_memsp``
    (the BP-8 0xE8 LEA byte-0 writer, strength 1e6) FIRES on the callee ENT
    AX-marker row. Its ``_lea_e8_first_ent_gate_enabled`` branch adds
    ``("OP_ENT", +100)`` to distinguish the FIRST-LEA-after-ENT from the re-read
    LEA — but ``OP_ENT`` is a CROSS-STEP-PERSISTENT residue that is ALSO high
    (~+5.24, saturating the AND-sum well past the +60 threshold bump) on the
    genuine callee ENT step itself, so the 0xE8 slam over-fires and stamps 0xE8
    over the carried AX. The ``e0_fetch_memsp`` 0xE0 writer does not rescue it
    (this is a BP-8 frame, not a 2nd-local re-read).

    THE FIX (mirrors #345 ``_tail_lea_e8_ent_byte0_blockers``): append the SHARP
    per-step FETCHED ENT opcode low-nibble one-hot ``OPCODE_BYTE_LO+6`` as a hard
    NOT-blocker on the e8 writer. It is +1.0 EXACTLY on a genuine ENT step (ENT =
    C4 opcode 6 = 0x06) and 0.0 on a genuine LEA AX-marker row (which carries
    ``OPCODE_BYTE_LO+0``), so the legit BP-8 LEA byte-0 0xE8 emit — including
    func's first-LEA ``&a`` / ``&x`` and the nested first-level ``&a`` — is
    byte-identical; only the callee-ENT rows where the 0xE8 slam is WRONG are
    vetoed, leaving the upstream ``ent_ax_passthrough`` (``AX_CARRY -> OUTPUT``)
    to survive to the LM head. This is the CURRENT-fetched-opcode discriminator
    the persistent ``OP_ENT`` residue lacks.

    Note ``_tail_lea_e8_ent_byte0_blockers`` (#345) blocks the LEGACY
    ``tail_lea_local_ax_marker_byte0_e8`` corrector, NOT this campaign
    ``e8_alubp_memsp`` keystone (which the golden build never emits), so this is
    an INDEPENDENT blocker on a distinct writer.

    DEFAULT tracks its OWN env var (DEFAULT OFF, NOT floored ON by the campaign)
    AND requires ``_lea_byte0_memsp_relay_enabled()`` (the e8 writer's campaign
    branch): the NOT-blocker is appended ONLY when both hold, so flag-OFF,
    ``C4_NO_STACK0_EMIT=0``, or the 35-token golden build are all byte-identical
    to the pre-fix default. Dedicated kill-switch so ``tools/
    flag_regression_gate.py --flag C4_NESTED_ENT_AXDUMP`` can A/B it.
    """
    return (
        os.environ.get("C4_NESTED_ENT_AXDUMP", "0") != "0"
        and _lea_byte0_memsp_relay_enabled()
    )


def _lea_e8_nested_ent_axdump_blockers() -> tuple:
    """FETCHED-ENT NOT-blocker for the campaign ``e8_alubp_memsp`` 0xE8 byte-0
    writer (nested callee-ENT AX-dump fix, #342).

    Empty unless ``_lea_e8_nested_ent_axdump_enabled`` (byte-identical off). When
    active, hard-blocks the writer on the per-step FETCHED ENT opcode one-hot
    ``OPCODE_BYTE_LO+6`` so it cannot stamp the BP-8 0xE8 over the carried AX on a
    genuine callee-ENT AX-marker row. Byte-identical on genuine LEA rows
    (``OPCODE_BYTE_LO+6 == 0`` there).
    """
    if not _lea_e8_nested_ent_axdump_enabled():
        return ()
    return ((_ENT_GUARD_OPCODE_LO, -1_000_000_000.0),)


def _lea_byte0_memsp_relay_enabled() -> bool:
    """Flag for the PHASE-2 KEYSTONE — the campaign LEA byte-0 address relay
    (ROOT 1, gates func_identity step-6 + var_mul/three multi-local + nested).

    The wall this lifts (verified TEACHER-FORCED spec_k=0, BUILT dims, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): in the 30-token
    frame the LEA effective-address byte-0 (the BP-relative local address
    0xE8/0xE0/0xD8) is NOT delivered onto the AX-marker OUTPUT row. The legacy
    ``tail_lea_local_ax_marker_byte0_e8`` corrector keys on ``CMP+7`` and the
    ``FETCH``/``MEM_ADDR_SRC`` bands, but in the 30-tok frame ``CMP+7 == 0``,
    ``MEM_ADDR_SRC == 0``, and on ``func_identity``'s ``&x`` LEA the FETCH band
    is empty too — so the rule never fires and the AX-marker OUTPUT byte-0 keeps
    the stale ``AX_CARRY`` leak (``func_identity`` step-6 ``got ax=0x46`` instead
    of ``0xFFE8``). var_mul/three's 2nd/3rd locals (``&b`` BP-16, ``&c`` BP-24)
    are likewise never corrected to 0xE0/0xD8 (the multilocal-guard correctly
    SILENCES the 0xE8 stamp but nothing writes the right byte).

    THE BUILD — three campaign-only AX-marker OUTPUT byte-0 writers, each keyed
    on the clean per-frame discriminator measured at the LEA AX-marker row
    entering the L25 tail block (probe ``tools/_probe_learelay_tf.py``,
    teacher-forced so the rows are drift-free):

      * ``&x`` / ``&a`` (BP-8, want 0xE8): on ``func_identity`` the FETCH band is
        DEAD so the only surviving discriminator is the AUTOREGRESSIVE
        ``ALU_HI+15`` MAGNITUDE — it is ~+73..+90 on the BP-8 effective-address
        compute and ~+5.5 on the BP-16/BP-24 frames and ~-45 on every non-LEA AX
        row (the golden ROOT-1 discriminator, branch ``opcam-funcmax-attack``
        ``acf2be0d``). The 0xE8 writer keys on ``OP_LEA`` (multiplicative gate,
        ~5.23 on LEA / <=0.05 elsewhere) + ``0.2*ALU_HI+15``: BP-8 sum ~25.4
        crosses, BP-16/24 ~8.3 and non-LEA <0 stay silent. On the var first-LEA
        (ALU_HI+15 ~73, already 0xE8 upstream) it re-writes 0xE8 — a no-op; on
        the var ``&a`` re-read (small ALU_HI+15, already 0xE8 upstream) it stays
        silent (no correction needed).
      * ``&b`` (BP-16, want 0xE0): the multi-local re-read LEA carries small
        ``ALU_HI+15`` indistinguishably from ``&a`` re-read, so the RELIABLE
        discriminator is the FETCH IMMEDIATE (the LEA imm). imm=-16 = 0xF0 lights
        ``FETCH_LO+0`` + ``FETCH_HI+15`` (NOT ``FETCH_LO+8`` / ``FETCH_HI+14``);
        the 0xE0 writer REQUIRES ``FETCH_LO+0`` and ``FETCH_HI+15`` (each weight
        8 so their ABSENCE drops the score below threshold — this defeats the
        non-LEA ``-0.x*ALU_HI+15`` × negative-residual trap, since FETCH is 0 on
        every non-LEA AX row) and NOT-blocks ``FETCH_LO+8`` / ``FETCH_HI+14``.
        func_identity's ``&x`` (``FETCH_HI+15 == 0``) is excluded.
      * ``&c`` (BP-24, want 0xD8): imm=-24 = 0xE8 lights ``FETCH_LO+8`` +
        ``FETCH_HI+14`` (NOT ``FETCH_HI+15``); the 0xD8 writer mirrors the 0xE0
        rule on those nibbles.

    All three gate MULTIPLICATIVELY on ``OP_LEA`` (the only LEA-specific signal
    that is ~0 on non-LEA rows; ``CMP+7`` is dead in this frame) and carry the
    same hard NOT-blockers as the legacy e8 corrector (``OP_IMM`` -1e6,
    ``OP_ADD/SUB/DIV/MOD`` -1e9, ``IS_BYTE`` -10, the five competing markers
    -1e4) so a non-AX / non-LEA / arith-result row can never satisfy them.

    DEFAULT tracks ``no_stack0_emit_enabled()``: ON in the 30-token campaign
    config, ABSENT (the three rules omitted, tail bank count 2059) otherwise so
    the golden 35-token build is bit-for-bit unchanged. Force with
    ``C4_LEA_BYTE0_MEMSP_RELAY=0/1``.
    """
    from .shared import no_stack0_emit_enabled

    forced = os.environ.get("C4_LEA_BYTE0_MEMSP_RELAY")
    if forced is not None:
        return forced != "0"
    return no_stack0_emit_enabled()


def _lea_byte0_alu_amplify_enabled() -> bool:
    """Flag for the PHASE-2 multi-param / multi-local LEA byte-0 ALU-AMPLIFIER
    (ROOT 1 sibling — the func_square / func_max / func_min re-read LEAs).

    The wall this lifts (verified TEACHER-FORCED + AR spec_k=0, BUILT dims,
    campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): on the
    multi-param / multi-local RE-READ LEAs (a second reference to ``&a`` /
    ``&b`` inside an ``if`` body or a second use of a param) the keystone's
    ``ALU_HI+15`` magnitude is ~0 AND the FETCH band is genuinely AMBIGUOUS
    (func encodes the LEA imm as FETCH_HI nib 1 with FETCH_LO nib 0 for BOTH
    BP-8 ``&x`` and BP-16 ``&b`` — operand 16 maps to 0xE8 in a 1-param frame
    but 0xE0 in a 2-param frame), so NONE of the keystone's three per-frame
    discriminators survive. func_square step-9 (``&x`` re-read, want 0xE8),
    func_max / func_min step-11/15 (``&b`` re-read, want 0xE0) all keep a stale
    operand leak (got 0x08 / 0x24 / 0x00).

    THE ROOT (probe ``tools/_probe_multilea_tf.py`` + block-trace): the
    L8 ``lea_lo`` ALU + the downstream byte-1 0xFF/0xFE band DO compute the
    CORRECT effective-address byte-0 on the re-read LEAs — by physical block
    14 the OUTPUT band already carries 0xE0 / 0xE8 (HI nib 0xE one-hot, LO nib
    the offset). The L25 tail bank (block 41) PRESERVES it. But the L25
    POST-OP (physical block 42 — an opcode-gated cross-step OUTPUT relay that
    does NOT read OP_LEA) stamps a ~370-magnitude fixed pattern that OVERWRITES
    the correct byte whenever the upstream residual is WEAK (~50 on the re-read
    rows). On the var multi-local LEAs the keystone's 1e6-strength write leaves
    a ~1.8e9 residual that survives block 42 untouched; the func re-reads only
    have the L8 ALU's native ~50 magnitude, so block 42 wins.

    THE BUILD — a self-reinforcing AMPLIFIER in the L25 tail bank that
    re-asserts the ALU-computed byte at 1e6 strength so it survives block 42.
    For each lo nibble ``k`` in {0x0, 0x8} (frame-local addresses are 8-byte
    int-aligned so byte-0 is always 0xE0/0xE8/0xD0/0xD8) and each frame-address
    HI nibble ``h`` in {0xE (14), 0xD (13)} a rule fires iff (multiplicative
    ``OP_LEA`` gate) ``MARK_AX + HAS_SE + OUTPUT_HI_THIS_STEP+h + OUTPUT_LO+k``
    are all present, and re-writes the byte ``(h<<4)|k`` at 1e6 (4 rules).
    Restricting ``k`` to {0,8} keeps the amplifier from matching a DRIFTED LI
    AX row whose loaded value carries a 0xE-/0xD- high nibble with an arbitrary
    low nibble — the wider 16-nibble form was measured to shift the func_square
    step-7 PC; {0,8} restores it (PC 66 == oracle). The ``OUTPUT_HI+h`` requirement
    (h in {13,14}, i.e. the ALU produced a COMPLETE 0xFE-/0xFD- frame address)
    is the DISCRIMINATOR that survives where FETCH/ALU_HI+15 don't: it is the
    presence of the already-computed high nibble. The FIRST LEA in a frame
    (func_identity / func_square step-6, func_max step-8, nested) has HI nib 0
    (the ALU only did the low nibble; the magnitude path / rule (1) supplies
    0xE there) so the OUTPUT_HI+{13,14} requirement EXCLUDES it — no conflict
    with the keystone ALU_HI+15 rule. NO frame-offset discriminator is needed:
    the amplifier preserves WHATEVER the ALU correctly computed, per frame
    depth. Carries the same hard NOT-blockers as the keystone (OP_IMM/ADD/SUB/
    DIV/MOD, IS_BYTE, the five competing markers) and a hard ``OUTPUT_HI+0``
    NOT-block so it can never fire on the incomplete first-LEA row.

    DEFAULT **OFF** (opt-in via ``C4_LEA_BYTE0_ALU_AMPLIFY=1``; only active in
    the campaign config, ``C4_NO_STACK0_EMIT=1``). The byte-identical path
    (flag unset / =0 / ``C4_NO_STACK0_EMIT=0``) is bit-for-bit golden
    ``7f6f2e5d``.

    Why DEFAULT OFF (measured 2026-06-22, CPU ``cpu_full_trace`` spec_k=0,
    campaign config): the amplifier CORRECTLY delivers the multi-param re-read
    LEA byte-0 — teacher-forced, func_square step-9 ``&x`` 0x08->0xE8,
    func_max/min step-11/15 ``&b`` 0x24/0x00->0xE0 (gate 3/3). BUT the
    func_max / func_min / func_square autoregressive VERDICT is gated by an
    EARLIER root, the **LI value-load at step 7/9** (the param value loads as
    AX byte-0 = 0x00 instead of the argument; e.g. func_max OFF
    ``div_step=9 got=(pc=50,ax=0) oracle=(pc=50,ax=36)`` — PC correct, the LI
    AX is wrong), which sits BEFORE the re-read LEAs (steps 11/15) — so the
    amplifier's fix is never reached. This is the L13/L15 multi-local LI
    value-load CAM (tasks #313/#318), not a LEA byte. Worse, with the LI
    still drifting, the amplifier's OUTPUT_HI-nib{13,14} discriminator can
    match a DRIFTED LI AX row (its loaded value leaks 0xD-/0xE- high nibble
    while OP_LEA leaks ~0.05+) and shift the step-9 PC (func_max ON
    ``got pc=66`` vs OFF ``pc=50``) — a framing regression. So flipped ON now
    it nets 0 flips and risks a func PC regression. Kept in-tree DEFAULT OFF
    as VALIDATED, golden-safe, campaign-ready infra: the LEA byte-0 delivery
    is correct and rides along the moment the LI value-load root lands (it
    should then net positive on func_max/min/square + nested re-reads). A/B
    via ``tools/flag_regression_gate.py --flag C4_LEA_BYTE0_ALU_AMPLIFY``.

    UPDATE 2026-06-22 (PHASE-2 LI blocker RESOLVED): the gating LI value-load
    root is now FIXED by ``C4_L15_LI_VALROW_B1`` (the L15 head-0 MEM_VAL_B1
    value-row lift; see ``ops/l15_ops.py:_l15_li_valrow_b1_on``). With the LI
    fix ON the first-param LI resolves (teacher-forced gate 4/4) and the
    interp-oracle divergence ADVANCES: amplify-OFF the func targets stall at
    step 11 (the &b re-read LEA); amplify-ON they advance to step 13/19 — the
    actual ADD/MUL/GT compute or the return PC (func_add s11->s13 exp=0x44,
    func_square s9->s11 exp=0x40, func_max s11->s13, func_max_1 / func_min
    s11->s19 PC) with NO func PC regression (the prior step-9 PC=66 shift was
    the LI-drift artifact this fix removes). So flipped ON WITH the LI fix the
    amplifier now rides along as designed. It is kept DEFAULT OFF here pending
    a clean cross-cluster ``flag_regression_gate`` pass; flip it together with
    ``C4_L15_LI_VALROW_B1`` once that gate is green under an uncontended fleet.

    UMBRELLA (2026-07-04, ``C4_OPCAM_FRAME``): the operand-CAM frame-depth
    discriminator survey (R2) rolls the amplifier under ONE campaign-only
    default-OFF flag ``C4_OPCAM_FRAME`` — the single lever the ROOT brief asks
    for. ``C4_OPCAM_FRAME=1`` turns this amplifier ON in the campaign config
    (the gating LI value-load fix ``C4_L15_LI_VALROW_B1`` is ALREADY campaign
    default-ON, so the amplifier is the only remaining gate). The frame-depth
    discriminator is the AUTOREGRESSIVE ``OUTPUT_HI+{14,13}`` complete-frame-
    address presence (the byte the L8 ``lea_lo`` ALU computed for THIS frame
    depth), re-asserted at 1e6 so it survives the block-42 default overwrite —
    NOT a single-rule tail tweak: it preserves WHATEVER the ALU correctly
    computed per frame depth. The dedicated ``C4_LEA_BYTE0_ALU_AMPLIFY``
    kill-switch still wins when set explicitly (``=0`` forces OFF even under
    the umbrella; ``=1`` forces ON), so the flag-regression gate can A/B just
    this change. flag-OFF (neither flag set, or ``C4_NO_STACK0_EMIT=0``, or the
    35-token golden) is byte-identical to ``91f55411``: this branch is never
    entered there (+0 tail rules).
    """
    from .shared import no_stack0_emit_enabled

    forced = os.environ.get("C4_LEA_BYTE0_ALU_AMPLIFY")
    if forced is not None:
        # Dedicated kill-switch wins: explicit =1/=0 overrides the umbrella.
        return forced != "0" and no_stack0_emit_enabled()
    if os.environ.get("C4_OPCAM_FRAME", "0") != "0":
        # Operand-CAM frame-depth umbrella: campaign-only, default-OFF.
        return no_stack0_emit_enabled()
    return False


def _sp_pop_marker_cmp3_hardgate_enabled() -> bool:
    """Flag for the campaign-config CMP+3 HARD-gate on the binary-pop SP-marker
    ``e0 -> e8`` correction (#315 — the SP/BP cross-step tracking drift).

    The wall this lifts (verified AUTOREGRESSIVE spec_k=0, BUILT dims, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the
    ``tail_sp_pop_marker_e0_to_e8`` rule (this module's
    ``sp_pop_marker_increment_rules``) corrects the binary-pop ``SP += 8``
    marker from 0xE0 to 0xE8 when the pop relay ``CMP+3`` is active. Its base
    ``CMP+3`` term is a WEAK +1.5 positive, and its corruption-veto terms
    (``OUTPUT_HI_THIS_STEP+13`` with weight -10) were tuned for a 0/1 nibble
    indicator. In BOTH the 35-tok golden AND the 30-tok campaign frame the
    block-26/L14 OUTPUT-band#2 corruption drives ``OUTPUT_HI_THIS_STEP+13``
    to ~-1.0e4 (and ``+14`` to ~+1.0e4); the -10 * -1.0e4 = +1.0e5 term then
    SWAMPS the +9.0 threshold, so the rule fires on EVERY SP marker row, even
    on non-pop steps where ``CMP+3 == 0`` (e.g. the IMM / STORE-value step).

    On var_simple/var_update/if_var the model's per-step SP byte-0 then jumps
    +8 one step EARLY (at the IMM step instead of the SI/STORE pop step). The
    drifted SP byte-0 shifts the next LI query's ``ADDR_B0`` nibbles, the L15
    read-CAM ties and loses to a spurious BP-register row, and the LI returns
    the wrong value -> the `var`/`if_var` LI step diverges. (Autoregressive
    only — teacher-forcing the SP token HIDES this; #315 ablation proven.)

    The fix promotes ``CMP+3`` to a HARD gate: it adds ``("CMP+3", 1e6)`` and
    raises the threshold by ``3.5 * 1e6`` so the rule can ONLY clear threshold
    when the genuine binary-pop relay (``CMP+3 >= 4``) is present. On a non-pop
    SP marker row (``CMP+3 == 0``) the +1.0e5 corruption term is now ~1.5e6
    below threshold -> the rule is vetoed and SP byte-0 carries forward
    unchanged. On the real pop step (``CMP+3 == 4``) the +4e6 hard term clears
    the +3.5e6 offset AND overpowers the residual ``OP_SI`` veto (SI/STORE IS
    a valid pop of the just-stored address), so e0 -> e8 still applies on the
    correct step.

    DEFAULT **OFF** (opt-in via ``C4_SP_POP_MARKER_CMP3_HARDGATE=1``) — see the
    measured-impact note below. The byte-identical path (flag unset, flag=0, or
    ``C4_NO_STACK0_EMIT=0``) is bit-for-bit golden ``7aefc860``.

    Why DEFAULT OFF (measured 2026-06-21, GPU autoregressive, campaign config):
    this hard-gate CORRECTLY removes the SP byte-0 off-by-one drift — the
    probe-confirmed ``e0/e8`` swap at the IMM/STORE steps of var_simple /
    var_update / if_var is gone, and the autoregressive SP byte-0 then matches
    the DraftVM oracle at every step. BUT the SP/BP drift turned out NOT to be
    the gating root for the var/if_var full_trace VERDICT: a clean
    autoregressive ablation (patch SP+BP value tokens to oracle in the AR loop)
    shows the LI step's AX is STILL wrong (``0x00`` byte-0) — the
    ``var_simple`` blocker is the **LI value-load byte-0 = 0x00** (the L13/L15
    multi-local value-load CAM not delivering byte-0 in the 30-tok AR frame; see
    tasks #313/#289), independent of SP. So with the LI value-load still broken,
    this SP fix flips 0 programs and, by shifting the if_var comparison-step
    framing, REGRESSES if_var 13/25 -> 11/25 (GPU full_trace, ids 425-449). Kept
    in-tree DEFAULT OFF so it can ride along (it should net positive) the moment
    the LI value-load root lands. Flip ON for A/B via
    ``tools/flag_regression_gate.py --flag C4_SP_POP_MARKER_CMP3_HARDGATE``.
    """
    from .shared import no_stack0_emit_enabled

    return (
        os.environ.get("C4_SP_POP_MARKER_CMP3_HARDGATE", "0") == "1"
        and no_stack0_emit_enabled()
    )


def _sp_pop_carry_byte0_dominate_enabled() -> bool:
    """Flag for the campaign-config binary-pop SP byte-0 CARRY-case dominator
    (#319 — the expr_mod SP-tracking desync, ~17 programs IDs 875-899).

    The wall this lifts (verified AUTOREGRESSIVE spec_k=0, BUILT dims, campaign
    config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; probes
    ``tools/probe_exprmod_sp_carry.py`` + ``probe_exprmod_sp_logits.py`` +
    ``probe_exprmod_sp_dims.py``): on the binary-pop ``SP += 8`` step whose input
    SP byte-0 is exactly ``0xF8`` (stack at ``0x00fff8``), the oracle carries
    byte-0 ``0xf8 -> 0x00`` (``0x00fff8 + 8 = 0x010000``). The SP byte-0 VALUE is
    computed at the SP MARKER row (the row whose FFN OUTPUT band drives the
    byte-0 LM-head argmax). The marker-correction family
    (``sp_pop_marker_increment_rules``) has ``e0->e8`` / ``d0->d8`` / ``f0->f8``
    / ``d8->e0`` -- but is MISSING the ``f8->00`` carry case, the most common
    stack-pop boundary. On most programs the L6 binary-pop nibble rotation
    delivers ``0x00`` cleanly, but on certain MOD/ADD results (e.g. id876
    ``46%6+6``) a value-dependent OUTPUT-band leak stamps ``OUTPUT_LO+8`` /
    ``OUTPUT_HI_THIS_STEP+15`` (= ``0xF8``) at the SP marker row (``~+73`` at
    step3, ``~+2.9e4`` at step6) -- so the stale ``0xF8`` wins the byte-0 argmax,
    the drifted SP byte-0 feeds back AR, the next step misframes, and the run
    desyncs at the HALT step (``got pc=None``). On a PASSING expr_mod (id875
    ``55%8+1``) the marker-row OUTPUT already resolves to ``0x00`` cleanly
    (``OUTPUT_LO+0`` ``+0.996`` vs ``OUTPUT_LO+8`` ``-0.006``) -- i.e. the
    failure is purely value-dependent: same EMBED input ``0xF8``, same
    ``CMP+3``, only the OUTPUT leak differs.

    The fix adds a SINGLE campaign-gated marker-row rule
    (``tail_sp_pop_marker_f8_to_00``, the missing carry sibling) that fires ONLY
    on the genuine binary-pop CARRY signature -- ``MARK_SP`` + ``CMP+3``
    HARD-gated (the ``SP += 8`` relay, ``== +4`` on the pop step, ``0`` on the
    PUSH/decrement step) + the input SP byte-0 proven exactly ``0xF8`` via
    ``EMBED_LO+8`` + ``EMBED_HI+15`` (both ``~+1.0`` on the marker row). It
    writes ``0x00`` at strength ``5.0e5`` (``byte_writes`` also drives ``-5.0e5``
    onto the competing ``LO+8`` / ``HI+15`` nibbles), DOMINATING the ``~3e4``
    corruption -> the marker-row OUTPUT byte-0 resolves to ``0x00``.

    WHY IT IS LOAD-BEARING-SAFE (the SP band gates var/if_var/SI/SC/var_simple):
    the ``0xF8`` input requirement makes this VALUE-CORRECT, not a blanket
    ``CMP+3 -> 0x00``. ``0xF8 + 8`` is the ONLY binary-pop input that carries to
    ``0x00``; a deeper-frame pop whose input byte-0 is ``0xF0`` (-> ``0xF8``,
    EMBED_LO nibble 0 -> excluded by the ``EMBED_LO+0`` blocker) or ``0xE8``
    (-> ``0xF0``, EMBED_HI nibble E=14 -> excluded by the ``EMBED_HI+14``
    blocker) fails the AND and is handled by the existing ``f0->f8`` / e-family
    rules. The PUSH/decrement step (whose input byte-0 IS ``0xF8`` but must emit
    ``0xF8``) is excluded by the ``CMP+3`` hard gate (it carries ``CMP+0`` /
    ``PSH_AT_SP``, not ``CMP+3``). And on the already-correct passing rows
    (id875) the input IS ``0xF8`` and ``CMP+3`` IS set, so the rule ALSO fires
    there and writes the SAME ``0x00`` the model already produces -> no change.

    DEFAULT **OFF** (opt-in via ``C4_SP_POP_CARRY_BYTE0_DOMINATE=1`` AND the
    campaign ``C4_NO_STACK0_EMIT=1``). Flag unset / ``=0`` / golden 35-tok build
    -> ZERO rules appended -> bit-for-bit golden ``7f6f2e5d``. Cross-cluster A/B
    via ``tools/flag_regression_gate.py --flag C4_SP_POP_CARRY_BYTE0_DOMINATE``.
    """
    from .shared import no_stack0_emit_enabled

    return (
        os.environ.get("C4_SP_POP_CARRY_BYTE0_DOMINATE", "1") != "0"
        and no_stack0_emit_enabled()
    )


def _psh_stack0_highbyte_darken_enabled() -> bool:
    """Flag for the PSH-STACK0-passthrough high-byte (byte2/byte3) darkening.

    ``_layer10_psh_stack0_passthrough_head_spec`` (block 16 / logical L11,
    head 3) is the PSH STACK0-store passthrough. Its active gate (slot 33)
    keys on ``PSH_AT_SP + IS_BYTE + H4+BP`` and so fires not only at the
    STACK0 byte-0 producer row but ALSO on the STACK0 byte-1/2/3 query rows
    (``BYTE_INDEX_{1,2,3} ~= 0.97``), where it averages CLEAN_EMBED debris
    into OUTPUT at scale 3.0 and CRUSHES the OUTPUT band to ~-652. On the
    var PSH (push of the LEA-local address 0xFFE8) this kills the STACK0
    byte-3 emission: with OUTPUT dead nothing scores positively at the
    STACK0[3] LM row and a stray REGISTER MARKER token (261) wins, ending
    the step one token early -> the 35-token frame shifts by 1 from step 3
    onward -> the production per-step decode re-anchors on the wrong markers
    and reads step-4 PC = 66 instead of 58 (the var_simple full_trace
    blocker; all 25 var_simple diverge identically at step 4 PC).

    The STACK0 high bytes (byte 2/3) of any pushed value <= 0xFFFF are 0x00
    and the residual pre-block-16 OUTPUT already defaults to 0x00 (+0.94), so
    darkening the head on the byte-2/byte-3 query rows lets the clean 0x00
    default survive -> no stray marker -> frame stays aligned -> step-4 PC
    reads 58. Byte 0 (the pushed-value producer at the STACK0 marker row,
    BYTE_INDEX all ~0) and byte 1 (the 16-bit round-trip path exercised by
    test_si_li_16bit_value) are UNTOUCHED. Default ON; with
    ``C4_PSH_STACK0_HIGHBYTE_DARKEN=0`` the head spec is byte-identical to
    the prior build (the two slot-7 BYTE_INDEX terms are omitted).
    """
    return os.environ.get("C4_PSH_STACK0_HIGHBYTE_DARKEN", "1") != "0"


def _stack0_pop_loaded_shallow_crush_enabled() -> bool:
    """Flag for the L25 ``tail_stack0_pop_loaded`` shallow-competitor fix.

    The L25 ``tail_stack0_pop_loaded_byte_*`` family (255 rules,
    ``stack0_pop_loaded_output_rules`` below, baked into the
    ``tail_bit32_result_correction`` bank at physical block 41) reinforces a
    STACK0 byte that a strong upstream load relayed into the OUTPUT band. Each
    rule fires iff ``MARK_STACK0 + ... + 0.05*OUTPUT_LO[lo] + 0.05*OUTPUT_HI[hi]
    >= 12``. The ``0.05*OUTPUT`` term is meant as a near-binary one-hot selector
    (OUTPUT ~= 1 for a genuine one-hot byte), but the L20 ``layer16_lev_routing``
    SP/BP frame-address relay (``l16_lev_sp_bp_plus16_*``) writes the relayed
    nibble at HUGE magnitude (~3300 in the residual). The selector then drives
    the unit to ~32000x its intended firing strength, and the
    ``byte_writes(value, strength=500)`` COMPETITOR side-effects (``-500`` to the
    15 non-matching nibbles of each lane) scale by that runaway hidden to
    ~-16M per nibble. Summed across every sibling that matches EITHER the
    relayed lo OR hi nibble, the whole OUTPUT band is crushed to ~-8M and the
    genuine relayed value (e.g. 0xE8 frame address) lands at -8.4M -- below the
    LM-head reference, so a stray REGISTER-MARKER token (REG_PC=257) wins the
    argmax instead of the value byte. That stray 257 lands at the STACK0[0]
    offset and the production full_trace decoder (fixed-35-token slice, FIRST
    REG_PC scan) reads PC from the wrong offset -> the var/expr/if-bool
    full_trace step-5 PC desync (expected pc=66 got pc=74; all 100 var_*).

    Fix: drop the per-byte COMPETITOR strength from 500 -> 5 (keep the +500
    matching-nibble reinforcement) so the runaway crush stays shallow enough
    that the relayed value survives POSITIVE. The genuine small-magnitude
    SI/LI/SC/LC memory loads already win by their own reinforcement, so the
    weaker competitor is byte-identity-irrelevant to the memory smoke (verified
    si/li/sc/lc all green). Default ON; with
    ``C4_STACK0_POP_LOADED_SHALLOW_CRUSH=0`` the family is byte-identical to the
    prior build (competitor strength stays 500).
    """
    return os.environ.get("C4_STACK0_POP_LOADED_SHALLOW_CRUSH", "1") != "0"


def _l10_exit_axcarry_enabled() -> bool:
    """Flag for the L10 EXIT/no-clean-opcode AX-materialization source fix.

    DEFAULT-OFF (opt in with ``C4_L10_EXIT_AXCARRY=1``). Ships the
    ``test_simple_function`` win (``JSR 3; EXIT; NOP; ENT 0; IMM 42; LEV`` ->
    42) under the 6 LEV flags, the last blocker after the func_identity +25.

    ROOT (spec_k=0, softmax1-with-sink, 6 LEV flags ON; probes
    tools/_probe_l10_exit_root.py + _probe_output_across_blocks.py):
    On the POST-LEV EXIT step the opcode decode resolves NO clean opcode at the
    AX marker (OP_EXIT/OP_LEV/OP_IMM/OP_ENT all ~0) but ``OP_LEA`` LEAKS to
    ~1.71 (a dead/spurious value -- it is 0.00 on EVERY healthy AX row of
    lea_basic / func_identity / add / mul / mod). That leaked OP_LEA both (a)
    SUPPRESSES the L10 ``_layer10_alu_ax_passthrough`` (OP_LEA is in its
    suppressed-ops list) so AX_CARRY is NOT routed to OUTPUT, and (b) TRIGGERS
    the L10 LEA effective-address high-nibble materializer (block-14 ffn unit
    gated on OP_LEA, reads FETCH_HI -> writes OUTPUT_HI), which stamps the stale
    frame-pointer high nibble 0xF into OUTPUT_HI. Result: OUTPUT = 0xF0 instead
    of the correct 0x2A=42. ``AX_CARRY`` carries the correct 0x2A (hi nibble 2,
    lo nibble A) UNCORRUPTED from block 3 through the final block 52 -- this is a
    source-SELECT fix, not a value fix.

    FIX: a flag-gated standalone ``PureFFN`` post_op on the L25 tail block (after
    ``tail_bit32_result_correction``, where OP_LEA / AX_CARRY / MARK_AX all
    persist) that fires ONLY on this signature -- ``MARK_AX`` AND a LEAKED
    ``OP_LEA`` (>= ~1.0, never present on a healthy AX row) AND no
    OUTPUT-owning opcode (IMM/ADD/SUB/bitwise/cmp/MUL/DIV/MOD/SHL/SHR) AND no
    live ``MEM_ADDR_SRC`` (which a GENUINE LEA address-eval row carries) -- and
    routes ``AX_CARRY_LO[k] -> OUTPUT_LO[k]`` / ``AX_CARRY_HI[k] ->
    OUTPUT_HI[k]`` at a magnitude that DOMINATES the ~12 materializer, clearing
    the competing OUTPUT cells. Flag-off => zero rules appended, no post_op,
    byte-identical to HEAD. See ``_l10_exit_axcarry_rules`` /
    ``make_l10_exit_axcarry_op``.
    """
    return os.environ.get("C4_L10_EXIT_AXCARRY", "1") == "1"

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import (
    byte_copy_computed_rules,
    byte_route_rules,
    derived_comparison_rules,
    multi_way_and_rule,
)
from ..ir import CompilerIR, ConditionTerm, DimRef, FFNRule, StructuralOp
from ..layer_compiler import Operation
from ..band_guarantees import expected_byte_guarantee_rules
from ..positional_invariant import invariant_threshold, marker_bank_index
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from ..wide_alu_dsl import bitwise_rules
from .residual_band_registry import register_residual_band
from .shared import (
    _as_setdim_proxy,
    absdiff_fix_enabled,
    absdiff_ret_byte1_enabled,
    loop_lea_b0_e0_restore_enabled,
    loop_lea_b0_e8_oplea_req_enabled,
    loop_lea_b0_e8_restore_enabled,
    loop_lea_oplea_gate_enabled,
    loop_li_opcode_fetch_addrkey_clamp_enabled,
    loop_si_byterow_marker_clear_enabled,
    mul_stack0_byte39_guard_enabled,
    no_stack0_emit_enabled,
    operand_from_memsp_enabled,
    sili_cam_b1_enabled,
)


# Non-first-PSH SP byte-0 fix scratch band (flag-gated; default ON via
# ``_nonfirst_psh_sp_fix_enabled``). Holds the per-step AND signal "this
# step's SP-decrement RESULT byte0 == 0xF0" (OUTPUT_LO+0 & OUTPUT_HI+15 at
# the SP marker), written by ``make_l10_nonfirst_psh_sp_helper_op`` (before
# the L25 tail block) and read as a NOT-blocker by
# ``tail_sp_marker_byte0_f8_from_initial_stack_exact`` so it stops forcing
# 0xF8 over the genuine 0xF0 of a non-first push. Flag-off => band not
# collected (byte-identical d_model) and the blocker condition is skipped.
register_residual_band(
    "NONFIRST_PSH_SP_SUPPRESS", 1,
    owner="make_l10_nonfirst_psh_sp_helper_op",
    flag=_nonfirst_psh_sp_fix_enabled,
    never_share=True,
)

# absdiff LEV-return byte-1 leak flag (C4_ABSDIFF_RET_BYTE1). A BOUNDED
# step-function indicator = step(-(OUTPUT_LO+0) >= 10) written by the precursor
# ``make_l10_absdiff_ret_byte1_flag_op`` on the AX byte rows of the return step:
# it is 1 where the mis-firing L6 marker rule CRUSHED OUTPUT_LO+0 (the absdiff
# single-byte leak, crush -46..-199) and 0 where OUTPUT_LO+0 is clean positive
# (a genuine func_mul multi-byte return, +3.91). Bounded (unlike the raw crush)
# so the huge-magnitude SP/BP/LEA crushes cannot leak into the correction AND;
# the OP_LEV>=0.95 selector then excludes those non-AX / non-return rows.
# Flag-off => band not collected (byte-identical d_model).
register_residual_band(
    "ABSDIFF_RET_LEAK", 1,
    owner="make_l10_absdiff_ret_byte1_flag_op",
    flag=absdiff_ret_byte1_enabled,
    never_share=True,
)


# === L10 attention-head layout (auto-fit; legacy head_idx as docs) ===
#
# L10 attention hosts 8 primary heads, each owned by one ``kind="block"``
# bake op below. Pre-migration the spec helpers used bare ``head_idx=N``
# literals; resolving every literal through :func:`_l10_head_idx`
# preserves byte-identity while declaring the L10 head axis as audited
# data. Order mirrors the bake-op factory order below so the table
# reads top-to-bottom alongside the bakes that own each row.
#
# Head 1 (AX byte passthrough) and head 7 (BP byte passthrough) both reuse
# the byte-passthrough chain template :func:`_byte_passthrough_chain_spec`;
# heads 4/5/6 are co-owned by ``layer10_stack0_byte_relay_bake`` (one
# allocator row per spec for inspection clarity).
#
# Phase 7.B.6: the allocator runs without ``pin=`` -- first-fit picks
# 0..7 in declaration order, which matches the legacy layout bit-for-bit
# because :data:`_L10_HEAD_LAYOUT` is contiguous and ordered. The
# ``legacy_head_idx`` column is documentation only; the load-bearing
# copy is the :func:`_l10_head_idx` lookup, consumed by the head-spec
# factories that write Q/K/V/O weights at the resolved index.
_L10_HEAD_LAYOUT = (
    # (op-name key,                                       legacy_head_idx (docs only))
    ("layer10_carry_relay_bake.head_0",                  0),  # ADD/SUB byte carry
    ("layer10_byte_passthrough_bake.head_1",             1),  # AX byte passthrough
    ("layer10_sp_byte_passthrough_bake.head_2",          2),  # SP byte passthrough
    ("layer10_psh_stack0_passthrough_bake.head_3",       3),  # PSH STACK0 passthrough
    ("layer10_stack0_byte_relay_bake.head_4",            4),  # bitwise stack-byte relay
    ("layer10_stack0_byte_relay_bake.head_5",            5),  # non-bitwise stack-byte relay
    ("layer10_stack0_byte_relay_bake.head_6",            6),  # STACK0 byte persistence
    ("layer10_bp_byte_passthrough_bake.head_7",          7),  # BP byte passthrough
    # Wave 1 A3 broadcast heads: copy AX byte h CLEAN_EMBED to
    # STACK0_BYTE_VAL_h_LO/HI at the matching STACK0 byte row during
    # OP_PSH. See ``_layer10_psh_ax_broadcast_head_spec`` for the design.
    # Slot 11 reserved; A2 widened L10 attn budget to 12 (commit d39159a1).
    ("layer10_psh_ax_broadcast_bake.head_8",             8),  # PSH AX_b1 -> STACK0_BYTE_VAL_1
    ("layer10_psh_ax_broadcast_bake.head_9",             9),  # PSH AX_b2 -> STACK0_BYTE_VAL_2
    ("layer10_psh_ax_broadcast_bake.head_10",            10), # PSH AX_b3 -> STACK0_BYTE_VAL_3
    # JSR/LEV PC byte_passthrough head (slot 11): mirrors the AX/SP/BP
    # byte_passthrough pattern so PC bytes 1-3 are explicitly carried
    # from the prior step's PC byte 1-3 (MARK_PC + BYTE_INDEX_k) into
    # this step's MARK_PC byte rows. Suppressed on opcodes that
    # rewrite PC (OP_JSR, OP_JMP, OP_BZ, OP_BNZ, OP_LEV) so the L6
    # JSR/JMP/branch PC override (byte 0) and L9 LEV PC restore
    # (byte 0 from mem[BP+8]) are not stomped. Without this head, PC
    # bytes 1-3 at JSR/LEV steps fall back to L3 default (0x00),
    # leaking residual SP / return-addr bytes into the AX byte
    # passthrough chain at MARK_AX rows (see
    # ``docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md`` 2026-05-11; PC byte
    # contamination of 0xf8/0x03 at AX byte 1-3 was the documented
    # JSR clobber shape).
    ("layer10_pc_byte_passthrough_bake.head_11",         11), # PC byte passthrough
)


def _allocate_layer10_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with the L10 heads.

    Phase 7.B.6: ``pin=`` is dropped from every entry. The allocator's
    first-fit picks the lowest free head index in declaration order;
    because :data:`_L10_HEAD_LAYOUT` is contiguous (0..7) and ordered,
    first-fit reproduces the legacy ``head_idx`` values bit-for-bit.
    The actual weight-write head indices are still looked up via
    :func:`_l10_head_idx` inside the head-spec factories below, so
    byte-identity with the legacy bake is preserved regardless of
    allocator order.

    Wave 1 Cluster A2: ``layer_max_heads=12`` widens the L10 head budget
    from the legacy default (8) to leave four free slots (8..11) for the
    Wave 1 A3 broadcast heads. The freshly-widened slots stay empty at
    this stage — first-fit only consumes 0..7 because
    :data:`_L10_HEAD_LAYOUT` declares exactly 8 heads — so the bake is
    byte-identical with the pre-widen state.

    JSR/LEV follow-up: ``layer_max_heads`` bumped to 13 to host the
    PC byte_passthrough head at slot 11 (see
    :func:`_layer10_pc_byte_passthrough_head_spec`).
    """
    allocator = AttentionHeadAllocator(layer_max_heads=13)
    for name, _legacy_head_idx in _L10_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=10)
    return allocator


def _l10_head_idx(op_name: str) -> int:
    """Return the pinned L10 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L10_HEAD_LAYOUT` for callers (e.g.
    ``compiler_ir_factory`` helpers, spec functions) that cannot
    instantiate a per-bake allocator. Mirrors the L4/L13 pattern.
    """
    for name, head_idx in _L10_HEAD_LAYOUT:
        if name == op_name:
            return head_idx
    raise KeyError(f"_l10_head_idx: unknown L10 attention op {op_name!r}")


# === L10 FFN unit layouts (auto-fit; legacy offsets retained as docs) ==
#
# L10 hosts three distinct FFNs in the compiled model:
#
#   1. ``model.blocks[10].ffn`` -- baked by ``make_layer10_alu_op`` via
#      ``vm_step._set_layer10_alu``. 1846 units = comparison combine (18)
#      + bitwise OR/XOR/AND lo+hi (3 * 512 = 1536) + MUL lo (256) +
#      SHL/SHR zero shortcut (4) + AX passthrough (32).
#
#   2. A dependency-assigned FFN block carrying the *combined* L10 post-op
#      logic (``l10_post_ops_combined``, ``kind="ffn"``). 1562 units =
#      ``BinaryOpByteZeroingPostOp`` (8) + 3x ``CarryPropagationPostOp``
#      (512 each, slice later zeroed) + ``ComparisonCombine`` (18). The
#      carry slice keeps its unit range so OUTPUT-row offset accounting
#      stays byte-identical even though the weights are wiped.
#
#   3. A late ``tail_bit32_result_correction`` block on layer 17 -- its
#      own freshly-allocated ``PureFFN`` sized to ``len(rules)`` (2059).
#      Single-owner layout, declared here so a future second tenant in
#      that bank goes through ``allocator.alloc(...)``.
#
# Phase 7.B.6: every layout below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because each layout is fully
# contiguous in declaration order (every entry starts exactly where
# the previous one ended), first-fit reproduces the legacy pinned
# offsets bit-for-bit -- so byte-identity with the legacy bakes and
# their monotonic ``unit = 0`` / ``offset = 0`` cursors survives the
# pin drop. The ``legacy_start`` columns are kept purely as
# documentation; downstream MUL/LEA tail rules in the wide-MUL fix
# chain are unaffected because they consume layout-by-name, not by
# pin offset.
#
# The per-block ``make_l10_post_op_attach_op`` bake appends six (lookup)
# or seven (efficient) *independent* ``PureFFN`` modules onto
# ``block.post_ops`` -- each is a standalone bank, not a shared hidden
# axis, so it deliberately does NOT appear in these tables. Migrating
# those into an allocator would require flattening them into one FFN,
# which is the next refactor stage, not this commit.
#
# Migration is byte-identical bookkeeping: each underlying helper still
# writes via its own monotonic ``unit = 0`` / ``offset = 0`` cursor.
# The allocator declares ranges by name, the helpers write the weights,
# and an ``assert`` after each helper verifies the cursor lands exactly
# where the layout table says it should. Changing any helper's unit
# count requires updating the matching table in lock-step.

# Main L10 FFN (model.blocks[10].ffn). Walk mirrors the order of writes
# in ``vm_step._set_layer10_alu``.
_L10_FFN_UNIT_LAYOUT_MAIN = (
    # (sub-stage name, legacy_start (docs only), n_units)
    # bitwise_or/xor/and: 256 lo + 256 hi lookups + 62 stale-ALU
    # residue cancels per op (31 per nibble: 16 stale_alu0_b + 15
    # stale_a_carry0, skipping a=0 to avoid double-cancel at
    # op_fn(0, 0)). See ``_layer10_alu_bitwise_rules`` (2026-06-10).
    ("layer10_alu.cmp_combine",       0,   18),  # 6 default + 12 override
    ("layer10_alu.bitwise_or",       18,  574),  # 512 lookup + 62 cancel
    ("layer10_alu.bitwise_xor",     592,  574),  # 512 lookup + 62 cancel
    ("layer10_alu.bitwise_and",    1166,  574),  # 512 lookup + 62 cancel
    ("layer10_alu.mul_lo",         1740,  256),  # (a*b)%16 lookup
    ("layer10_alu.shl_shr_zero",   1996,    4),  # 2 per opcode (SHL, SHR)
    ("layer10_alu.ax_passthrough", 2000,   32),  # 16 lo + 16 hi
    # Wall-4 SESSION 2 per-nibble EQ engine (2026-06-12): 256-unit
    # 4-way AND on (ALU_HI+h, AX_CARRY_HI+h, ALU_LO+l, AX_CARRY_LO+l)
    # gated OP_EQ at MARK_AX. Writes OUTPUT_LO+1 / cancels OUTPUT_LO+0
    # for equal operands. See ``_layer10_alu_eq_engine_rules``.
    ("layer10_alu.eq_engine",      2032,  256),  # 16x16 nibble-pair EQ
    # Per-nibble ORDERING engine (2026-06-12): 272-unit CMP-cascade
    # recompute at MARK_AX gated CMP_GROUP — hi_lt(120)+lo_lt(120)+
    # hi_eq(16)+lo_eq(16) writing CMP+0/+3/+1/+2, feeding the live
    # ComparisonCombine for LT/GT/LE/GE (and EQ/NE; sole CMP-flag writer
    # for all six comparison opcodes). See
    # ``_layer10_alu_ordering_engine_rules``.
    ("layer10_alu.ordering_engine", 2288,  272),  # hi/lo lt + hi/lo eq
)
_L10_FFN_UNIT_LAYOUT_MAIN_TOTAL = 2560


def _l10_main_cmp_margin_extra() -> int:
    """Flag-conditional +6 for the C4_CMP_COMBINE_MARGIN OUTPUT-HI clamp bank.

    The clamp bank (``_layer10_alu_cmp_hi_clamp_rules``) is appended LAST in
    the L10-main FFN so it never shifts the shared cmp_combine / bitwise / ALU
    banks. Flag-OFF (incl golden 35-token) -> 0 -> byte-identical. Flag-ON
    (campaign) -> +6 (one clamp unit per comparison opcode).
    See ``shared.cmp_combine_margin_enabled``.
    """
    from .shared import cmp_combine_margin_enabled
    return 6 if cmp_combine_margin_enabled() else 0

# Combined post-op FFN baked by ``make_l10_post_ops_combined`` (kind="ffn",
# dependency-assigned). Each range maps 1:1 to a post-op class's
# ``hidden_dim`` and lands at the offset the inline ``offset`` counter
# walks to in the original bake.
_L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("l10_post_ops_combined.binary_op_byte_zeroing",     0,    8),  # PureFFN H=8
    ("l10_post_ops_combined.carry_propagation_byte0",    8,  512),  # PureFFN H=512
    ("l10_post_ops_combined.carry_propagation_byte1",  520,  512),  # PureFFN H=512
    ("l10_post_ops_combined.carry_propagation_byte2", 1032,  512),  # PureFFN H=512
    ("l10_post_ops_combined.comparison_combine",      1544,   18),  # PureFFN H=18
)
_L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED_TOTAL = 1562

# Tail bit32 result correction (lives on L17 block.post_ops as its own
# fresh PureFFN). Single tenant today; the layout makes the bank
# explicit so a future tenant claims through the allocator.
#
# P5 RETIRE (R-FRAME INCR-3 unconditional collapse): the six pure-copy tail
# frame-guarantee families now emit ONLY their computed per-nibble route form
# (the ``range(256)`` enumerated fallbacks + the ``C4_R_FRAME_TAIL`` escape
# hatch are DELETED). The tail bank is thereby permanently 690 units, not the
# retired 2059-unit fully-enumerated form: the six collapses drop 254 + 223 +
# 223 + 223 + 222 + 224 = 1369 units (2059 - 1369 = 690). This IS the default
# golden (``1c04c3fd``) build, which was already the collapsed form.
_L10_FFN_UNIT_LAYOUT_TAIL_BIT32 = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("tail_bit32_result_correction.rules", 0, 690),
)
_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL = 690


def _allocate_l10_main_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for ``model.blocks[10].ffn``.

    Phase 7.B.6: ``pin=`` is dropped. First-fit walks
    :data:`_L10_FFN_UNIT_LAYOUT_MAIN` in declaration order and lands
    each sub-stage at the lowest free gap; because the layout is fully
    contiguous (every entry starts where the previous one ended)
    first-fit reproduces the legacy offsets bit-for-bit, so
    ``_set_layer10_alu``'s monotonic ``unit = 0`` counter still lands
    on the same indices. The allocator is stashed on
    ``block.ffn._l10_unit_allocator`` for downstream auditing.
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L10_FFN_UNIT_LAYOUT_MAIN:
        allocator.alloc(name, n_units)
    # C4_CMP_COMBINE_MARGIN campaign clamp bank (flag-OFF -> 0 -> not declared).
    extra = _l10_main_cmp_margin_extra()
    if extra:
        allocator.alloc("layer10_alu.cmp_hi_clamp", extra)
    return allocator


def _allocate_l10_post_ops_combined_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for the combined post-op FFN.

    Mirrors the inline ``offset`` walk in ``make_l10_post_ops_combined``:
    one ``BinaryOpByteZeroingPostOp``, three ``CarryPropagationPostOp``,
    one ``ComparisonCombine`` -- each occupying the slot its
    predecessor's ``hidden_dim`` advances to. The carry slice is zeroed
    by the existing post-bake step but still occupies its declared
    range so the comparison-combine offset remains stable.

    Phase 7.B.6: ``pin=`` is dropped. First-fit reproduces the legacy
    offsets bit-for-bit because the layout is fully contiguous in
    declaration order.
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED:
        # C4_CMP_COMBINE_MARGIN campaign clamp: the comparison_combine bank is
        # the LAST tenant, so grow it in place by +6 (flag-OFF -> +0 -> golden).
        if name == "l10_post_ops_combined.comparison_combine":
            n_units += _l10_main_cmp_margin_extra()
        allocator.alloc(name, n_units)
    return allocator


_L10_BINARY_OP_BYTE_ZEROING_OP_DIMS = (
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_SHL", "OP_SHR", "OP_MUL", "OP_DIV", "OP_MOD",
)


def _l10_binary_op_byte_zeroing_rules(S: float) -> tuple[FFNRule, ...]:
    """Declarative rules for ``BinaryOpByteZeroingPostOp`` (8 units).

    Mirrors ``vm_step.BinaryOpByteZeroingPostOp._bake_weights``: four
    opcode-gated detectors (units 0..3, gate by the binary-op set) and
    four bitwise-gated detectors (units 4..7, gate by ``TEMP+3``).

    Per the legacy ``wire_zeroing_writes(unit_offset)`` helper, within
    each group of four:

      * unit_offset+0 wipes the OUTPUT_LO band (16 cells, each -3.0/S),
      * unit_offset+1 wipes the OUTPUT_HI band (16 cells, each -3.0/S),
      * unit_offset+2 adds OUTPUT_LO+0 += 5.0/S,
      * unit_offset+3 adds OUTPUT_HI+0 += 5.0/S.

    The opcode-gated detectors use ``gate=None`` with multi-term
    ``gate_terms`` summing all 11 opcode flags (each weight 1.0). The
    bitwise detectors use ``gate="TEMP+3"`` directly.
    """

    def conds_opcode_gated():
        return (
            ("IS_BYTE", 1.0),
            ("H1+1", 1.0),
            ("TEMP+8", -1000.0),
            ("TEMP+9", -1000.0),
        )

    def conds_bitwise_gated():
        return (
            ("IS_BYTE", 1.0),
            ("H1+1", 1.0),
            ("TEMP+3", 1.0),
            ("TEMP+8", -1000.0),
            ("TEMP+9", -1000.0),
        )

    opcode_gate_terms = tuple(
        (op_dim, 1.0) for op_dim in _L10_BINARY_OP_BYTE_ZEROING_OP_DIMS
    )

    output_lo_wipe = tuple(
        (f"OUTPUT_LO+{k}", -3.0 / S) for k in range(16)
    )
    output_hi_wipe = tuple(
        (f"OUTPUT_HI_THIS_STEP+{k}", -3.0 / S) for k in range(16)
    )

    rules: list[FFNRule] = []

    # Units 0..3: opcode-gated detectors.
    # DSL v4b: 4-condition AND with explicit threshold 1.5; the opcode
    # set is fed via gate_terms (multi-opcode disjunctive gate) so the
    # rule fires on any single binary-op being active.
    for unit_idx, writes in enumerate((
        output_lo_wipe,
        output_hi_wipe,
        (("OUTPUT_LO+0", 5.0 / S),),
        (("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
    )):
        rules.append(multi_way_and_rule(
            conditions=conds_opcode_gated(),
            threshold=1.5,
            gate=None,
            gate_terms=opcode_gate_terms,
            gate_bias=0.0,
            writes=writes,
            name=f"l10_binary_op_byte_zeroing_opcode_unit{unit_idx}",
            scope=(
                "IS_BYTE and ("
                + " or ".join(_L10_BINARY_OP_BYTE_ZEROING_OP_DIMS)
                + ")"
            ),
        ))

    # Units 4..7: bitwise-gated detectors (gate = TEMP+3).
    # DSL v4b: explicit-threshold AND; single TEMP+3 gate.
    for unit_idx, writes in enumerate((
        output_lo_wipe,
        output_hi_wipe,
        (("OUTPUT_LO+0", 5.0 / S),),
        (("OUTPUT_HI_THIS_STEP+0", 5.0 / S),),
    )):
        rules.append(multi_way_and_rule(
            conditions=conds_bitwise_gated(),
            threshold=2.5,
            gate="TEMP+3",
            gate_weight=1.0,
            writes=writes,
            name=f"l10_binary_op_byte_zeroing_bitwise_unit{unit_idx}",
            scope="IS_BYTE and TEMP+3",
        ))

    return tuple(rules)


_L10_CARRY_NON_ARITH_OPS = (
    "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
    "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
    "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND", "OP_EQ",
    "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_SHL",
    "OP_SHR", "OP_MUL", "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
    "OP_PUTCHAR", "OP_GETCHAR",
)

_L10_CARRY_BYTE_DIM_BY_IDX = (
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
)


def _l10_carry_propagation_rules(
    S: float, *, byte_idx: int, cascade: bool,
) -> tuple[FFNRule, ...]:
    """Declarative rules for one ``CarryPropagationPostOp`` instance (512 units).

    Mirrors ``vm_step.CarryPropagationPostOp._bake_weights``: 256 ADD
    carry units followed by 256 SUB borrow units, indexed by
    ``(lo, hi)`` over ``range(16) x range(16)`` for the active byte.

    Differences captured per (byte_idx, cascade):

      * ``byte_dim`` flag: BYTE_INDEX_0/1/2 (no byte_idx=3 here).
      * ``add_carry_in`` / ``sub_carry_in`` choose between CARRY+1/+2
        (non-cascade byte_idx=0) and CARRY+3/+3 (cascade=True for
        byte_idx=1/2).
      * ``cascade=True`` routes mutual exclusion through TEMP+8/9
        instead of the carry dim pair.
      * The threshold drops to ``cascade_carry_threshold=40`` when
        cascade=True; non-cascade uses ``carry_threshold=56``.

    Output cells: the imperative bake writes
    ``W_down[OUTPUT_LO+lo] = -2/S`` then later
    ``W_down[OUTPUT_LO+new_lo] = +2/S`` with ``=`` semantics. When
    ``new_lo == lo`` (or ``new_hi == hi``) the second assignment
    overwrites the first, giving a net +2/S on that cell. The
    ``CompilerIR.lower_ffn`` lowerer uses ``+=`` so the rule below
    emits only the surviving write per cell (``+2/S`` when they
    collide, both writes when they don't). The CARRY+3 high-overflow
    write only fires at ``(15,15)`` for ADD and ``(0,0)`` for SUB,
    and only when ``byte_idx < 2``.

    Bank-derivation status (M8 l10 survey, 2026-07): this bank is ALREADY a
    fully COMPUTED generator (``add_rule_for`` / ``sub_rule_for`` evaluate
    ``(lo + hi*16) +/- 1`` at build time over the 16x16 nibble cross-product)
    authored through the canonical ``multi_way_and_rule`` DSL — it is NOT a
    hand-written literal rule list. It does NOT route through
    ``wide_alu_dsl.nibble_alu_lane_rules`` (the L8/L9 lane generator) because
    it is a materially RICHER shape: (1) it is a whole-BYTE increment/decrement
    (both output nibbles + the ``CARRY+3`` inter-byte overflow), not a single
    ``f(a,b,cin) % 16`` result nibble; (2) the +1/-1 couples the two nibbles
    (a low-nibble carry crosses into the high nibble, e.g. ``0x0F+1=0x10``), so
    it does NOT factorise into the per-nibble route the M8 byte-writeback
    collapse uses; (3) it carries the SUB-minuend-relay source swap
    (``STACK0_BYTE_VAL_{k+1}`` for multi-byte) and the campaign-conditional
    ``TEMP+9`` byte-1 hand-off gate. Re-expressing it via ``nibble_alu_lane_
    rules`` would require extending that generator with byte-level (not
    nibble-level) semantics + a second output nibble + the CARRY+3 write +
    the minuend-source swap, risking the L8/L9 callers' byte-identity and
    ADDING complexity, not removing it. It is width-locked to
    ``_L10_CARRY_HIDDEN_DIM = 512`` (the ``PureFFN`` hidden_dim, hard-asserted
    in ``_build_l10_carry_post_op``); a count-reducing collapse would break
    that. Kept as the already-derived 2D nibble ALU cascade it is.
    """

    if byte_idx not in (0, 1, 2):
        raise ValueError(f"byte_idx must be 0, 1, or 2; got {byte_idx}")

    byte_dim_name = _L10_CARRY_BYTE_DIM_BY_IDX[byte_idx]
    wrong_byte_dim_names = tuple(
        name for i, name in enumerate(_L10_CARRY_BYTE_DIM_BY_IDX)
        if i != byte_idx
    )
    # Phase 8.D: carry-in / carry-out byte-cell refs name the
    # (carry, alu, byte_index) semantic family lookup; byte-identical
    # to the legacy "CARRY+<k>" strings via DimRef.parse.
    add_carry_in_name = dim_ref("carry", "alu", 3 if cascade else 1)
    sub_carry_in_name = dim_ref("carry", "alu", 3 if cascade else 2)
    carry_byte3 = dim_ref("carry", "alu", 3)
    threshold = 40.0 if cascade else 56.0

    # Phase 2 multi-byte SUB minuend source (2026-06-12). The L14 borrow
    # cascade is an INTER-byte stage: the rule scoped on BYTE_INDEX_k
    # produces output byte (k+1) and needs the MINUEND's byte (k+1). The
    # legacy bake matched the minuend from OUTPUT, but only operand byte
    # 0 is relayed into OUTPUT (L7 operand_gather) -- bytes 1/2/3 of
    # OUTPUT are always 0x00 (un-relayed). So every multi-byte SUB
    # computed (0x00 - borrow) = 0xFF for bytes 1/2/3, byte-identically
    # for 0x100-1 (wants 0x00) and 0-1 (wants 0xFF). The new SUB minuend
    # selector reads STACK0_BYTE_VAL_{byte_idx+1}, where
    # layer13_sub_minuend_relay (L13 head 4) deposits the pushed
    # operand's byte (byte_idx+1) at the BYTE_INDEX_{byte_idx} predictor
    # row. For 8-bit SUB those operand bytes are 0x00 (= the old OUTPUT
    # match) so the result is byte-identical; for multi-byte it is
    # corrective. ADD keeps its OUTPUT minuend match (ADD operands ARE
    # in OUTPUT via the add path); only the SUB selector moves, and only
    # the minuend SELECTOR -- the result still EMITS on OUTPUT so the
    # CARRY+3 borrow relay (which rides OUTPUT) is untouched.
    sub_minuend_lo = f"STACK0_BYTE_VAL_{byte_idx + 1}_LO"
    sub_minuend_hi = f"STACK0_BYTE_VAL_{byte_idx + 1}_HI"

    # carry_weight=1.0, output_weight=20.0, mismatch_weight=0.0 in the
    # legacy bake; mismatch writes drop out of the rule because they
    # multiply to zero (``-S * 0 = 0`` produces no W_up cell).
    carry_weight = 1.0
    output_weight = 20.0

    def base_conds(carry_in_name: str) -> list[tuple[str, float]]:
        conds: list[tuple[str, float]] = [
            (carry_in_name, carry_weight),
            ("IS_BYTE", 1.0),
            ("H1+1", 1.0),
            ("MARK_AX", -5000.0),
            ("MARK_PC", -5000.0),
            (byte_dim_name, 1.0),
        ]
        # Suppress non-arithmetic opcodes.
        for op_name in _L10_CARRY_NON_ARITH_OPS:
            conds.append((op_name, -20.0))
        # TEMP+3 suppression (BITWISE_OP indicator).
        conds.append(("TEMP+3", -10.0))
        # Wrong byte position suppression.
        for wrong_name in wrong_byte_dim_names:
            conds.append((wrong_name, -10.0))
        return conds

    def add_rule_for(lo: int, hi: int) -> FFNRule:
        new_val = lo + hi * 16 + 1
        new_lo = new_val & 0xF
        new_hi = (new_val >> 4) & 0xF
        conds = base_conds(add_carry_in_name)
        # ADD-specific mutual exclusion vs SUB.
        conds.append(("OP_SUB", -20.0))
        if cascade:
            conds.append(("TEMP+8", 1.0))
            conds.append(("TEMP+9", -10.0))
        else:
            # sub_carry_in (CARRY+2) suppresses ADD firing under SUB.
            conds.append((sub_carry_in_name, -10.0))
        # OUTPUT_LO/HI match boosts (only matching nibble; mismatches
        # are 0 because mismatch_weight=0 in the legacy bake).
        conds.append((f"OUTPUT_LO+{lo}", output_weight))
        conds.append((f"OUTPUT_HI_THIS_STEP+{hi}", output_weight))

        writes: list[tuple[str, float]] = []
        # Collapse the legacy ``= -2/S`` + ``= +2/S`` pair using the
        # rule that lower_ffn uses ``+=``: when new == old, emit only
        # the surviving +2/S; otherwise emit both writes as distinct
        # cells.
        if new_lo == lo:
            writes.append((f"OUTPUT_LO+{lo}", 2.0 / S))
        else:
            writes.append((f"OUTPUT_LO+{lo}", -2.0 / S))
            writes.append((f"OUTPUT_LO+{new_lo}", 2.0 / S))
        if new_hi == hi:
            writes.append((f"OUTPUT_HI_THIS_STEP+{hi}", 2.0 / S))
        else:
            writes.append((f"OUTPUT_HI_THIS_STEP+{hi}", -2.0 / S))
            writes.append((f"OUTPUT_HI_THIS_STEP+{new_hi}", 2.0 / S))
        if lo == 15 and hi == 15 and byte_idx < 2:
            writes.append((carry_byte3, 2.0 / S))

        # DSL v4b: explicit-threshold AND. Carry-in dim acts as both a
        # positive condition (weight 1.0 above) and the multiplicative
        # gate (gate_weight=0.5, gate_bias=0.0). The conditions tuple
        # carries the byte-position selector, opcode blockers, and the
        # OUTPUT match boosts that select the right (lo, hi) cell.
        return multi_way_and_rule(
            conditions=tuple(conds),
            threshold=threshold,
            gate=add_carry_in_name,
            gate_weight=0.5,
            writes=tuple(writes),
            name=f"l10_carry_byte{byte_idx}_add_lo{lo}_hi{hi}",
            scope=(
                f"IS_BYTE and {byte_dim_name} and not MARK_AX "
                f"and not MARK_PC"
            ),
        )

    def sub_rule_for(lo: int, hi: int) -> FFNRule:
        new_val = (lo + hi * 16 - 1) & 0xFF
        new_lo = new_val & 0xF
        new_hi = (new_val >> 4) & 0xF
        conds = base_conds(sub_carry_in_name)
        conds.append(("OP_ADD", -20.0))
        if cascade:
            conds.append(("TEMP+9", 1.0))
            conds.append(("TEMP+8", -10.0))
        else:
            conds.append((add_carry_in_name, -10.0))
        # CAMPAIGN (30-token) byte-1 hand-off (2026-06-21). In the campaign
        # config the byte-1 minuend is delivered by the L8 head-7 mem[SP] CAM
        # as a ``+/-6`` STACK0_BYTE_VAL_1_LO encoding (not the clean ``+3``
        # one-hot the golden psh_ax_broadcast delivers), AND it lands too late
        # (block 18, AFTER this L10 cascade) so the LOW nibble matches but the
        # HIGH nibble cannot. With the L14 output step-boundary guard
        # amplifying every IS_BYTE unit to ~+1e9, the (lo, hi) HI match is
        # washed out, so ALL (matched-lo, *) SUB cells fire and spray
        # OUTPUT_HI uniformly (1537-87 -> 0x65AA not 0x05AA). Rather than fight
        # the guard, gate the byte_idx=0 non-cascade SUB cells OFF on the
        # SUB byte-1 emit row (TEMP+9) in the campaign config and let
        # ``layer14_sub_borrow_high_byte_passthrough`` (a clean ±0.08 op that
        # works because the cascade is then quiet, mirroring the no-borrow
        # passthrough) own the byte 1. byte_idx=0 only (the byte-0->1 stage),
        # TEMP+9-gated (the SUB byte-1 selector), campaign-only -> GOLDEN
        # byte-identical and the byte-2/3 cascade stages untouched.
        if (
            not cascade
            and byte_idx == 0
            and operand_from_memsp_enabled()
        ):
            # HARD off: the L14 output step-boundary guard amplifies the
            # IS_BYTE term ~+1e7, so a small TEMP+9 blocker is swamped. Use a
            # blocker on the SAME order as the guard so the SUB byte-1 cell is
            # driven below the silu floor on the TEMP+9 row regardless.
            conds.append(("TEMP+9", -1.0e8))
        # SUB minuend match: the relayed STACK0_BYTE_VAL_{byte_idx+1}
        # band (see the ``sub_minuend_lo/hi`` note above), not OUTPUT.
        conds.append((f"{sub_minuend_lo}+{lo}", output_weight))
        conds.append((f"{sub_minuend_hi}+{hi}", output_weight))

        # The result EMITS on OUTPUT. At the cascade input OUTPUT byte
        # (byte_idx+1) is always 0x00 (cell 0 hot in LO and HI) because
        # only byte 0 was relayed there; the minuend now rides
        # STACK0_BYTE_VAL. So cancel OUTPUT's known current content
        # (cell 0) and set the computed new nibble. For 8-bit SUB the
        # minuend = 0x00 -> (lo,hi)=(0,0) -> cancel-0 == the legacy
        # cancel-lo, byte-identical; for multi-byte the cancel-0 + set
        # yields the corrected byte.
        writes: list[tuple[str, float]] = []
        if new_lo == 0:
            writes.append(("OUTPUT_LO+0", 2.0 / S))
        else:
            writes.append(("OUTPUT_LO+0", -2.0 / S))
            writes.append((f"OUTPUT_LO+{new_lo}", 2.0 / S))
        if new_hi == 0:
            writes.append(("OUTPUT_HI_THIS_STEP+0", 2.0 / S))
        else:
            writes.append(("OUTPUT_HI_THIS_STEP+0", -2.0 / S))
            writes.append((f"OUTPUT_HI_THIS_STEP+{new_hi}", 2.0 / S))
        if lo == 0 and hi == 0 and byte_idx < 2:
            writes.append((carry_byte3, 2.0 / S))

        # DSL v4b: same shape as add_rule_for, but with the SUB carry-in
        # dim driving the gate and the SUB-side mutual-exclusion conds.
        return multi_way_and_rule(
            conditions=tuple(conds),
            threshold=threshold,
            gate=sub_carry_in_name,
            gate_weight=0.5,
            writes=tuple(writes),
            name=f"l10_carry_byte{byte_idx}_sub_lo{lo}_hi{hi}",
            scope=(
                f"IS_BYTE and {byte_dim_name} and not MARK_AX "
                f"and not MARK_PC"
            ),
        )

    rules: list[FFNRule] = []
    for lo in range(16):
        for hi in range(16):
            rules.append(add_rule_for(lo, hi))
    for lo in range(16):
        for hi in range(16):
            rules.append(sub_rule_for(lo, hi))
    return tuple(rules)


# Number of hidden units in one carry instance (256 ADD + 256 SUB).
_L10_CARRY_HIDDEN_DIM = 512


def _build_l10_carry_post_op(
    *, d_model: int, S: float, byte_idx: int, cascade: bool, dim_positions,
):
    """Build one L10 carry/borrow post-op from the declarative DSL.

    Phase 7.C cut: replaces the imperative
    ``vm_step.CarryPropagationPostOp(...)._bake_weights`` bake with a
    declarative lowering of :func:`_l10_carry_propagation_rules` into a
    bare ``PureFFN``. ``compare_symbolic_to_lowered_ffn`` + an element-
    wise tensor diff (``tools/verify_carry_migration.py``) confirm the
    lowered weights are byte-identical to the legacy bake's, so the
    forward (``PureFFN.forward`` is final and shared) is unchanged.

    The post-construction ``_strengthen_*`` / ``_suppress_*`` helpers in
    ``make_l10_post_op_attach_op`` continue to run on the returned module
    exactly as before; they mutate the resulting weights identically
    regardless of how the base weights were authored.
    """
    from ...base_layers import PureFFN

    post_op = PureFFN(dim=d_model, hidden_dim=_L10_CARRY_HIDDEN_DIM)
    end = Primitives.lower_ffn_rules(
        post_op,
        _l10_carry_propagation_rules(S, byte_idx=byte_idx, cascade=cascade),
        dim_positions,
        start_unit=0,
        S=S,
    )
    assert end == _L10_CARRY_HIDDEN_DIM, (
        f"l10 carry post-op (byte_idx={byte_idx}, cascade={cascade}) "
        f"lowered {end} units, expected {_L10_CARRY_HIDDEN_DIM}"
    )
    # Carry forward-relevant attributes that the legacy
    # ``CarryPropagationPostOp`` exposed for downstream auditors.
    post_op.d_model = d_model
    post_op.S = S
    return post_op


# The six comparison opcodes, in the canonical L10 cmp-combine bank order.
_CMP_OPS_ORDER = ("EQ", "NE", "LT", "GT", "LE", "GE")
# The (HI_EQ ∧ LO_LT) lexicographic product term that gets the campaign guard
# threshold on GT / GE (see ``cmp_gt_lo_lt_hieq_guard_enabled``).
_CMP_HI_EQ_LO_LT_TERM = ("CMP+1", "CMP+3")


def _derived_cmp_combine_rules(
    S: float,
    *,
    opcode_gate_fmt: str,
    default_threshold: float,
    override3_threshold: float,
    override_include_blocker: bool,
    name_prefix_fmt: str,
    gt_ge_guard_threshold: float,
) -> tuple[FFNRule, ...]:
    """Derive all six comparison decoders from ONE zero-detector (task #446).

    Shared body for the two hand-authored cmp-combine banks
    (``_l10_comparison_combine_rules`` decode path and
    ``_layer10_alu_cmp_combine_rules`` L10-main ALU lane), selected by
    ``C4_DERIVE_CMP``. Both are the SAME §576-590 truth table over the CMP
    zero-detector/sign flags; the per-path DATA (gate spelling, thresholds,
    whether overrides carry the MARK_PC blocker) is entirely in the kwargs,
    matching the corresponding hand path byte-for-byte. Emits 18 units (one
    default + one-to-three overrides per op) in ``_CMP_OPS_ORDER``.
    """
    rules: list[FFNRule] = []
    for op in _CMP_OPS_ORDER:
        thr_override = (
            {_CMP_HI_EQ_LO_LT_TERM: gt_ge_guard_threshold}
            if op in ("GT", "GE") else {}
        )
        rules.extend(derived_comparison_rules(
            op=op,
            marker_dim="MARK_SE_ONLY",
            opcode_gate=opcode_gate_fmt.format(op=op),
            result_lo_band="OUTPUT_LO",
            default_write=(("OUTPUT_HI_THIS_STEP+0", 2.0),),
            blocker_dim="MARK_PC",
            blocker_weight=-50.0,
            default_threshold=default_threshold,
            override2_threshold=1.5,
            override3_threshold=override3_threshold,
            override3_hi_lt_blocker_weight=-0.1,
            default_write_value=2.0,
            override_write_value=4.0,
            scope_prefix="MARK_SE_ONLY",
            name_prefix=name_prefix_fmt.format(op=op.lower()),
            override3_threshold_override=thr_override,
            override_include_blocker=override_include_blocker,
            S=S,
        ))
    return tuple(rules)


def _l10_comparison_combine_rules(S: float) -> tuple[FFNRule, ...]:
    """Declarative rules for ``ComparisonCombine`` (18 units).

    Mirrors ``vm_step.ComparisonCombine._bake_weights``. For each of
    EQ/NE/LT/GT/LE/GE the post-op emits one *default* unit that writes
    the initial result and one or two *override* units that flip the
    result when CMP[0..3] flags indicate the opposite outcome.

      * Default unit: constant_write style (``W_gate[unit, CONST]``
        was the legacy gate, but ``ComparisonCombine`` actually sets
        ``b_gate = 1.0`` with no W_gate cell, matching
        ``constant_write``'s ``gate=None``/``gate_bias=1.0`` form).
        Writes ``OUTPUT_LO+default_result`` and ``OUTPUT_HI+0`` at
        +2/S.
      * Override 2-way: gated by the opcode dim, conditions sum
        MARK_AX + one CMP flag with threshold 1.5, writes a +4/S/-4/S
        pair on OUTPUT_LO.
      * Override 3-way: gated by the opcode dim, conditions sum
        MARK_AX + two CMP flags with threshold 2.5, writes a +4/S/
        -4/S pair on OUTPUT_LO.

    All units include a strong MARK_PC blocker (``-50``) to prevent
    leaked OP_NE/OP_GT/OP_GE or CMP residue from corrupting PC
    predictions; see the 2026-05-09 fix comment in vm_step.py.
    """

    MARK_PC_BLOCK = -50.0

    # if_var GT-TRUE lo_lt-leak guard (the symmetric companion to the GT-FALSE
    # ``cmp_hi_lt_alu15_leak_guard``). Campaign-only: RAISE the GT/GE
    # ``(hi_eq AND lo_lt) -> 0`` 3-way override threshold 2.5 -> 2.75 so a
    # spurious ``lo_lt`` (CMP+3 ~= 1.67) alone (``hi_eq`` absent, A.hi > B.hi)
    # can no longer trip the GT-result flip, while the genuine hi_eq+lo_lt
    # GT-FALSE override (CMP+1 ~= 1.24 AND CMP+3 ~= 1.46, sum 3.70) still fires.
    # Flag-OFF / non-campaign -> 2.5 -> golden byte-identical.
    # See ``shared.cmp_gt_lo_lt_hieq_guard_enabled``.
    from .shared import cmp_gt_lo_lt_hieq_guard_enabled
    _gt_gtge_3way_thresh = (
        2.75 if cmp_gt_lo_lt_hieq_guard_enabled() else 2.5
    )

    rules: list[FFNRule] = []

    # C4_DERIVE_CMP (task #446): the 18 core comparison units are the §576-590
    # truth table over the CMP zero-detector/sign flags, generated from the
    # single ``derived_comparison_rules`` DSL (via ``_derived_cmp_combine_rules``)
    # — two combinators ``A_EQ_B := HI_EQ ∧ LO_EQ`` / ``A_LT_B := HI_LT ∨
    # (HI_EQ ∧ LO_LT)`` and pure boolean algebra over them, ZERO per-op magic.
    # The former per-op default+override hand-enumeration (and its nested
    # ``cmp_default`` / ``cmp_override_2way`` / ``cmp_override_3way`` helpers)
    # was proven byte-identical to this derivation (golden e50521f3;
    # tools/verify_derive_cmp.py 18/18) and has been DELETED — the derivation is
    # the sole source. ``C4_DERIVE_CMP`` remains a registered no-op kill-switch
    # for cache-key isolation. The margin-clamp bank appended below is orthogonal.
    from .shared import derive_cmp_enabled
    _ = derive_cmp_enabled()  # keep the flag live for cache-key isolation
    rules.extend(_derived_cmp_combine_rules(
        S,
        opcode_gate_fmt="OP_{op}",
        default_threshold=1.5,
        override3_threshold=2.5,
        override_include_blocker=True,
        name_prefix_fmt="l10_cmp_{op}",
        gt_ge_guard_threshold=_gt_gtge_3way_thresh,
    ))

    # C4_CMP_COMBINE_MARGIN (campaign-only, default-OFF): +6 OUTPUT_HI clamp
    # units appended LAST so the 18-unit default+override footprint is
    # UNCHANGED flag-OFF (byte-identical golden). A comparison result byte is
    # provably in {0, 1} so its OUTPUT_HIGH nibble is invariantly 0; each clamp
    # (same gate as this copy's default: MARK_SE_ONLY + OP_<cmp> + MARK_PC
    # blocker) darkens every non-zero OUTPUT_HI_THIS_STEP+1..15 nibble and
    # reinforces +0, out-voting a leaked operand (hi<<4) leak at the decode row.
    # See ``shared.cmp_combine_margin_enabled``.
    from .shared import cmp_combine_margin_enabled
    if cmp_combine_margin_enabled():
        # SURGICAL GATE (2026-07): additionally require the FRESH-comparison
        # signature ``SE_CMP_GROUP+0`` so the OUTPUT-HI clamp fires ONLY on a
        # genuine comparison-result decode row (result provably in {0, 1}) and
        # never on a value-carrying row that merely inherited a stale/leaked
        # ``OP_<cmp>``. See the twin gate in ``_layer10_alu_cmp_hi_clamp_rules``
        # and the SE-row probe (tools/_probe_cmp_se_row_disc.py): SE_CMP_GROUP
        # is ~0.94 on the genuine cmp decode row, ~0.00 everywhere else. The
        # +0.94 group flag lifts the "all markers present" sum to 2.94 >= 2.5;
        # a leaked ``OP_<cmp>`` without the fresh group flag stays at 2.0 < 2.5.
        for _op in ("OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE"):
            _writes = [("OUTPUT_HI_THIS_STEP+0", 60.0 / S)]
            _writes += [(f"OUTPUT_HI_THIS_STEP+{h}", -60.0 / S)
                        for h in range(1, 16)]
            rules.append(multi_way_and_rule(
                conditions=(
                    ("MARK_SE_ONLY", 1.0),
                    (_op, 1.0),
                    ("SE_CMP_GROUP+0", 1.0),
                    ("MARK_PC", MARK_PC_BLOCK),
                ),
                threshold=2.5,
                writes=tuple(_writes),
                name=f"l10_cmp_hi_clamp_{_op.lower()}_step_end",
                scope=(
                    f"MARK_SE_ONLY and {_op} and SE_CMP_GROUP and not MARK_PC"
                ),
            ))

    return tuple(rules)


def _allocate_l10_tail_bit32_units(n_rules: int) -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for ``tail_bit32_result_correction``.

    The tail PureFFN's ``hidden_dim`` equals ``len(rules)`` so this
    layout is parameterised: ``n_rules`` must match the
    ``_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL`` constant. A mismatch
    means someone changed the tail rule set without updating the
    layout table; fail loudly rather than silently miss a declared range.

    Phase 7.B.6: ``pin=`` is dropped. With an empty L17 tail FFN pool
    first-fit lands the single 2059-unit range at start=0, matching
    the legacy bake bit-for-bit. The wide-MUL high-byte fix chain
    (MUL/LEA tail rules) is unaffected because those rules consume
    layout-by-name, not by pin offset.
    """
    # PHASE-2 KEYSTONE (ROOT 1, GOLDEN-SAFE): the campaign LEA byte-0 address
    # relay adds THREE extra tail rules (0xE8/0xE0/0xD8 writers) when enabled,
    # so the expected count is 2059 (flag-OFF, golden byte-identical) or 2062
    # (flag-ON campaign). The layout's single tenant range widens by the same
    # +3. See ``_lea_byte0_memsp_relay_enabled``.
    #
    # PHASE-2 multi-param ALU-AMPLIFIER (+4) + the #319 SP byte-0 f8->00 carry
    # dominator (+1) -- both DEFAULT-OFF campaign flags appended to this width-
    # sensitive tail bank; flag-OFF (incl golden 35-token) each adds 0 ->
    # bit-for-bit unchanged. See _lea_byte0_alu_amplify_enabled /
    # _sp_pop_carry_byte0_dominate_enabled.
    # P5 RETIRE: the six R-FRAME INCR-3 tail frame-guarantee collapses are now
    # UNCONDITIONAL (their computed per-nibble route form is the only form the
    # builders emit; the ``range(256)`` enumerated fallbacks + ``C4_R_FRAME_TAIL``
    # escape hatch are DELETED). Their -1369 units are baked into the base count
    # ``_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL`` (690, not 2059), so they no longer
    # appear as conditional ``extra`` adjustments. Only the three DEFAULT-OFF
    # campaign flags below still widen the single-tenant tail range.
    extra = 0
    if _lea_byte0_memsp_relay_enabled():
        extra += 3
    if _lea_byte0_alu_amplify_enabled():
        extra += 4
    if _sp_pop_carry_byte0_dominate_enabled():
        extra += 1
    expected = _L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL + extra
    if n_rules != expected:
        raise ValueError(
            f"tail_bit32_result_correction rule count drift: helper "
            f"produced {n_rules} rules, allocator expects {expected}"
        )
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L10_FFN_UNIT_LAYOUT_TAIL_BIT32:
        allocator.alloc(name, n_units + extra)
    return allocator


# === L10 ALU FFNRule generators (Phase 6 Wave 4I migration) ==========
#
# Per-sub-stage declarative rules that reproduce ``_set_layer10_alu``
# (vm_step.py) byte-for-byte. Each generator mirrors one contiguous
# range of the ``_L10_FFN_UNIT_LAYOUT_MAIN`` table so a downstream
# lower via ``Primitives.lower_ffn_rules`` lands on the same pinned
# offsets the imperative helper writes today.

def _layer10_alu_cmp_combine_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 cmp_combine rules: 18 units for EQ/NE/LT/GT/LE/GE.

    Derived from the single BLOG_SPEC §576-590 zero-detector/sign primitive via
    ``_derived_cmp_combine_rules`` (task #446, C4_DERIVE_CMP). Each comparison
    opcode still emits one default unit (writes a baseline 0 or 1 result +
    an OUTPUT_HI[0]=1 marker, ungated via ``b_gate=1.0``) followed by 1-3
    override units (each gated on the SE_OP_* dim, flipping the result via a
    +4.0/S / -4.0/S pair on OUTPUT_LO) that fire on CMP-flag combinations from
    L9 — but those 18 units now fall out of ONE truth table over the two
    combinators ``A_EQ_B``/``A_LT_B`` rather than a per-op hand-enumeration
    (which was byte-identical, golden e50521f3, and has been deleted).
    """

    # C4_DERIVE_CMP (task #446): this L10-main ALU cmp lane is generated from
    # the single ``derived_comparison_rules`` DSL (via
    # ``_derived_cmp_combine_rules``) — the same §576-590 zero-detector/sign
    # truth table as the decode-row bank, with this path's structural constants
    # (SE_OP gate, default threshold 2.5, override3 threshold 4.0, overrides omit
    # the MARK_PC blocker). The former per-op default+override hand-enumeration
    # (and its nested ``_cmp_default`` / ``_cmp_override_2way`` /
    # ``_cmp_override_3way`` helpers) was proven byte-identical to this
    # derivation (golden e50521f3; tools/verify_derive_cmp.py 18/18) and has been
    # DELETED — the derivation is the sole source. ``C4_DERIVE_CMP`` remains a
    # registered no-op kill-switch for cache-key isolation.
    from .shared import derive_cmp_enabled
    _ = derive_cmp_enabled()  # keep the flag live for cache-key isolation
    return _derived_cmp_combine_rules(
        S,
        opcode_gate_fmt="SE_OP_{op}+0",
        default_threshold=2.5,
        override3_threshold=4.0,
        override_include_blocker=False,
        name_prefix_fmt="l10_alu_cmp_{op}",
        gt_ge_guard_threshold=4.0,
    )


def _layer10_alu_cmp_hi_clamp_rules(S: float) -> tuple[FFNRule, ...]:
    """C4_CMP_COMBINE_MARGIN (campaign-only, default-OFF) OUTPUT-HIGH clamp.

    Appended LAST in the L10-main FFN (so it never shifts the shared
    cmp_combine / bitwise / ALU banks). Flag-OFF -> empty tuple ->
    byte-identical golden. Flag-ON -> +6 units (one per comparison opcode).

    A comparison RESULT byte is provably in {0, 1}, so its OUTPUT_HIGH nibble
    is invariantly 0. Each clamp unit -- same gate as the ComparisonCombine
    default (``MARK_SE_ONLY`` + relayed ``SE_OP_<cmp>`` + ``MARK_PC`` blocker)
    -- DARKENS every non-zero ``OUTPUT_HI_THIS_STEP+1..15`` nibble and
    reinforces ``OUTPUT_HI_THIS_STEP+0``, out-voting a leaked operand
    ``(hi<<4)`` high nibble at the compare decode row so the emitted result
    byte is the clean low-nibble result. Because a boolean high nibble is
    always 0, this cannot alter any already-correct comparison; it only pulls
    a leaked high nibble back to 0. See ``shared.cmp_combine_margin_enabled``.
    """
    from .shared import cmp_combine_margin_enabled
    if not cmp_combine_margin_enabled():
        return ()

    def _clamp(op_name: str) -> FFNRule:
        # Strong amplitudes (a comparison result HIGH nibble is invariantly 0,
        # so OUTPUT_HI+0 can be driven decisively) so the clamp out-votes even
        # a large leaked operand ``(hi<<4)`` (~13+ observed) at the decode row.
        writes = [("OUTPUT_HI_THIS_STEP+0", 60.0 / S)]
        writes += [(f"OUTPUT_HI_THIS_STEP+{h}", -60.0 / S) for h in range(1, 16)]
        # SURGICAL GATE (2026-07): require the FRESH-comparison signature
        # ``SE_CMP_GROUP+0`` in addition to ``SE_OP_<cmp>``. Probing the SE
        # step-end rows (tools/_probe_cmp_se_row_disc.py) shows ``SE_CMP_GROUP``
        # is ~0.94 on the genuine comparison-result decode row and ~0.00 on
        # EVERY other SE row (incl. rows carrying a weak stale CMP residue),
        # while ``SE_OP_<cmp>`` alone can survive as a stale/cross-frame leak on
        # a value-carrying row (a compare feeding a value, e.g. func_max's
        # RETURN 0x63). Gating on ``SE_CMP_GROUP+0`` too means the OUTPUT-HI
        # clamp fires ONLY when THIS step's opcode is a genuinely-relayed
        # comparison whose result is provably in {0, 1} -- it can no longer
        # clobber a legitimately-nonzero OUTPUT_HI high nibble on a
        # non-comparison / value row. The +0.94 group flag pushes the "all
        # three markers present" sum to 1 + 1 + 0.94 = 2.94 >= 2.5 (fires),
        # while a leaked ``SE_OP`` WITHOUT the fresh group flag stays at
        # 1 + 1 = 2.0 < 2.5 (blocked). This CANNOT weaken the intended fix:
        # every genuine comparison decode carries ``SE_CMP_GROUP`` (the same
        # L9 step_end_operand_relay that mirrors ``OP_<cmp> -> SE_OP_<cmp>``).
        return multi_way_and_rule(
            name=f"l10_cmp_{op_name.lower()}_hi_clamp_step_end",
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                (f"SE_OP_{op_name}", 1.0),
                ("SE_CMP_GROUP+0", 1.0),
                ("MARK_PC", -50.0),
            ),
            threshold=2.5,
            writes=tuple(writes),
        )

    return tuple(_clamp(op) for op in ("EQ", "NE", "LT", "GT", "LE", "GE"))


def _layer10_alu_bitwise_rules(
    S: float, *, op_name: str, op_fn,
) -> tuple[FFNRule, ...]:
    """L10 bitwise OR/XOR/AND rules: 256 lo + 256 hi units per opcode.

    Each unit is a 3-way AND across (MARK_AX, ALU_*[a], AX_CARRY_*[b])
    that fires only when the OP_* gate is hot. Weights (40, 30, 30) and
    threshold 80 implement the balanced 3-way AND from
    ``vm_step._set_layer10_alu`` (see the BUG FIX 2026-04-16 comment):

      * all three present: 40 + 30 + 30 = 100 > 80 -> fires
      * any two present:   max(40 + 30) = 70 < 80  -> blocked

    ``op_fn`` is the bitwise function (``operator.or_`` / ``xor`` /
    ``and_``) used to compute the result nibble.

    Stale-ALU residue cancel band (added 2026-06-10): on COLLAPSED
    IMM+OP steps the ``ALU_LO/HI+0`` and ``AX_CARRY_LO/HI+0`` channels
    carry ~1.04 stale residual from the prior sub-cycle. The plain
    3-way AND at threshold 80 happily fires on stale (1.04 vs the
    legit 1.0 one-hot) and writes spurious mass to ``OUTPUT_LO/HI``.
    For each (legit-operand, stale-+0) pair we emit one negative-write
    cancel unit gated on an asymmetric-weight stale detector: weight
    1000 on the ``+0`` dim with threshold ``40 + 1000*1.02 + 30 =
    1090`` fires only when ``+0 >= 1.04`` (stale) while the legit
    one-hot ``+0=1.0`` leaves the sum at ``40 + 1000 + 30 = 1070 <
    1090``. The cancel write targets the same OUTPUT cell the
    spurious lookup would have hit (``op_fn(0, b)`` for stale-A and
    ``op_fn(a, 0)`` for stale-B). Codified by the L9/L10 isolation
    test ``tests/test_l9_collapsed_imm_input_isolated.py``.

    DSL derivation (2026-07): the byte-0 nibble bitwise LOOKUP is the
    same ``for a: for b: op_fn(a, b)`` build-time computed table the ALU
    derivable regime describes, so this builder is now a thin call into
    the shared ``wide_alu_dsl.bitwise_rules`` generator (the SAME generator
    the lookup-mode L10 post-op uses; ``ops/alu_ops.py``). The 574 rules
    per op — 256 main + the 31-rule stale-cancel band, interleaved per
    nibble — are emitted by the generator with ``emit_stale_cancel_band``.
    Byte-identity vs the prior hand-authored loop is proven by
    ``compare_symbolic_to_lowered_ffn`` + ``tools/_isa_golden_hash.py``.
    The distinguishing L10-main-FFN DATA (``MARK_SE_ONLY`` marker under the
    Wave A step_end_operand_relay broadcast, the separate LO/HI operand
    bands, the ``_step_end`` rule-name suffix) is passed as kwargs — there
    is zero per-value logic left here.
    """

    return bitwise_rules(
        op=op_name.lower(),
        operand_a_lo="ALU_LO",
        operand_a_hi="ALU_HI",
        operand_b_lo="AX_CARRY_LO",
        operand_b_hi="AX_CARRY_HI",
        result_lo="OUTPUT_LO",
        result_hi="OUTPUT_HI_THIS_STEP",
        opcode_gate=dim_ref("opcode_flag", op_name),
        marker_gate="MARK_SE_ONLY",
        S=S,
        name_prefix="l10_bitwise",
        name_suffix="_step_end",
        emit_stale_cancel_band=True,
    )


def _layer10_alu_bitwise_or_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 bitwise OR: 574 units (256 lo + 256 hi + cancel) gated on OP_OR."""

    return _layer10_alu_bitwise_rules(S, op_name="OR", op_fn=None)


def _layer10_alu_bitwise_xor_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 bitwise XOR: 574 units (256 lo + 256 hi + cancel) gated on OP_XOR."""

    return _layer10_alu_bitwise_rules(S, op_name="XOR", op_fn=None)


def _layer10_alu_bitwise_and_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 bitwise AND: 574 units (256 lo + 256 hi + cancel) gated on OP_AND."""

    return _layer10_alu_bitwise_rules(S, op_name="AND", op_fn=None)


def _layer10_alu_shl_shr_zero_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 SHL/SHR shift-out-zero shortcut: 4 units (2 per opcode).

    For shifts >= 8 the result byte is 0x00. Two cases per opcode:

      * Case A (shift >= 16): high nibble of the shift count is non-zero
        so ``AX_CARRY_HI[0]`` is NOT hot. The unit fires on
        ``MARK_AX * 60`` after subtracting ``-S * AX_CARRY_HI[0]`` to
        suppress shifts 0-15; ``b_up = -S * 59`` requires MARK_AX to be
        present to clear the threshold.
      * Case B (shift 8-15): high nibble = 0 (so ``AX_CARRY_HI[0] = 1``)
        and the low nibble is in 8..15. The unit fires when MARK_AX +
        ``AX_CARRY_HI[0]`` plus any one of ``AX_CARRY_LO[8..15]`` are
        present (threshold 80 vs 60 + 1 + 1 = 62 forces all three terms).

    Both cases write ``OUTPUT_LO[0]`` and ``OUTPUT_HI[0]`` to 1 so the
    next position emits a 0x00 byte. The OP_* gate selects which opcode
    the shortcut fires for.
    """

    def _case_a(op_name: str) -> FFNRule:
        # Phase 8.D: OP_<NAME> gate -> (opcode_flag, NAME).
        # DSL v4b: explicit-threshold AND (MARK_AX dominant + AX_CARRY_HI[0]
        # suppressor). Threshold 59 fires only when MARK_AX is hot AND
        # AX_CARRY_HI[0] is absent (i.e. shift >= 16).
        #
        # Wave B Cluster 1 (2026-06-10): MARK_AX -> MARK_SE_ONLY under
        # Wave A step_end_operand_relay (10ca51a7), which broadcasts
        # ALU_LO/HI, OP_SHL, OP_SHR from MARK_AX into MARK_SE_ONLY in
        # the same step.
        return multi_way_and_rule(
            name=f"l10_{op_name.lower()}_shift_ge16_zero_step_end",
            conditions=(
                ("MARK_SE_ONLY", 60.0),
                ("AX_CARRY_HI+0", -1.0),
            ),
            threshold=59.0,
            gate=dim_ref("opcode_flag", op_name),
            gate_weight=1.0,
            writes=(
                ("OUTPUT_LO+0", 2.0 / S),
                ("OUTPUT_HI_THIS_STEP+0", 2.0 / S),
            ),
        )

    def _case_b(op_name: str) -> FFNRule:
        conditions = [
            ("MARK_AX", 60.0),
            ("AX_CARRY_HI+0", 1.0),
        ]
        for lo_bit in range(8, 16):
            conditions.append((f"AX_CARRY_LO+{lo_bit}", 1.0))
        # Phase 8.D: OP_<NAME> gate -> (opcode_flag, NAME).
        # DSL v4b: explicit-threshold AND across MARK_AX + AX_CARRY_HI[0] +
        # (AX_CARRY_LO[8..15]). Threshold 80 forces all three terms (MARK_AX
        # 60 + HI 1 + one of the LO 8..15 cells 1 = 62 < 80).
        return multi_way_and_rule(
            name=f"l10_{op_name.lower()}_shift_8_15_zero",
            conditions=tuple(conditions),
            threshold=80.0,
            gate=dim_ref("opcode_flag", op_name),
            gate_weight=1.0,
            writes=(
                ("OUTPUT_LO+0", 2.0 / S),
                ("OUTPUT_HI_THIS_STEP+0", 2.0 / S),
            ),
        )

    return (
        _case_a("SHL"),
        _case_b("SHL"),
        _case_a("SHR"),
        _case_b("SHR"),
    )


# Suppressed opcodes for L10 ALU AX passthrough -- mirrors the
# ``suppressed_ops`` list in ``vm_step._set_layer10_alu``. Each opcode
# already owns the AX-byte-0 emission for its specific lane (L6 routing,
# L8 ALU, L15 memory lookup, DivModModule, etc.), so the L10 passthrough
# must NOT fire for them.
_L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS = (
    "OP_IMM",
    "OP_ADD",
    "OP_SUB",
    "OP_OR",
    "OP_XOR",
    "OP_AND",
    "OP_EQ",
    "OP_NE",
    "OP_LT",
    "OP_GT",
    "OP_LE",
    "OP_GE",
    "OP_MUL",
    "OP_DIV",
    "OP_MOD",
    "OP_SHL",
    "OP_SHR",
    "OP_LEA",
    "OP_LI",
    "OP_LC",
    "OP_JMP",
    "OP_EXIT",
    "OP_NOP",
    "OP_PUTCHAR",
)


def _layer10_alu_ax_passthrough_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 AX passthrough: 32 units (16 lo + 16 hi).

    Each unit fires at the AX marker only when none of the
    ``_L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS`` opcodes are active --
    every suppressed opcode contributes a ``-S`` term that pushes the
    pre-activation below the ``-S * 0.5`` threshold whenever the
    opcode flag is hot. Units 0..15 gate on ``AX_CARRY_LO[k]`` and route
    that one-hot into ``OUTPUT_LO[k]``; units 16..31 do the same on the
    hi nibble via ``AX_CARRY_HI[k]`` and ``OUTPUT_HI[k]``.
    """

    # Reduction map ⑥: this is the shared cross-layer BYTE-ROUTE primitive
    # (per-cell gate on ``{carry}+k`` -> write ``{out}+k``); delegate to
    # ``byte_route_rules``. Byte-identical: band-major then cell-major rule
    # order, shared ``(MARK_AX, *suppressed-op NOT-terms)`` AND at threshold
    # 0.5, ``2.0 / S`` route write, legacy ``l10_ax_passthrough_{lo,hi}_{k}``
    # names via ``name_prefix`` + the "lo"/"hi" band labels.
    conditions = (("MARK_AX", 1.0),) + tuple(
        (op_dim, -1.0) for op_dim in _L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS
    )
    return byte_route_rules(
        band_specs=(
            ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
            ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
        ),
        conditions=conditions,
        threshold=0.5,
        write_value=2.0,
        S=S,
        name_prefix="l10_ax_passthrough",
    )


def _layer10_alu_mul_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """L10 MUL lo-nibble lookup: 256 units gated on OP_MUL.

    For each (a, b) in 0..15 x 0..15, one 3-way AND unit fires only on
    the matching ALU_LO[a] / AX_CARRY_LO[b] one-hot pair at the AX
    marker, writing the lo nibble of (a * b) (i.e. ``(a * b) % 16``) to
    ``OUTPUT_LO``. Weights and threshold reuse the (40, 30, 30) / 80
    balanced 3-way AND from the bitwise sub-stages above so a single
    spurious one-hot in either operand band cannot fire the unit.

    Bank-derivation status (M8 l10 survey, 2026-07): this is a genuine
    2-OPERAND multiplication lookup table (``result`` depends non-linearly
    on BOTH operand nibbles ``a`` and ``b``), NOT a 1-operand identity /
    increment / cross-lane copy — so the per-nibble byte-writeback ROUTE
    collapse (``_computed_byte_writeback_route_rules`` / the M8
    ``C4_STACK0_*_COMPUTED`` family) does NOT apply (that route factorises a
    same-value nibble copy into two independent per-nibble channels; a 16x16
    product table has no such factorisation). It is also NOT covered by
    ``C4_MUL_MULTIPASS`` (which replaces the L11 mul-partial / L12
    mul-combine LOOKUP at a DIFFERENT layer — see ``shared.mul_multipass_
    enabled`` — leaving this L10 mul-lo nibble table untouched). It is
    already authored through the canonical ``multi_way_and_rule`` DSL (not a
    hand-written literal rule list) and is width-locked into
    ``_L10_FFN_UNIT_LAYOUT_MAIN_TOTAL`` (the byte-identical 1846-unit main
    L10 ALU FFN). No count-preserving further collapse exists; kept as the
    2-operand ``lookup_table_rules``-class product table it already is.
    """

    # Phase 8.D: OP_MUL gate -> (opcode_flag, MUL).
    gate_mul = dim_ref("opcode_flag", "MUL")
    rules: list[FFNRule] = []
    for a in range(16):
        for b in range(16):
            result = (a * b) % 16
            # DSL v4b: 3-way balanced AND across (MARK_AX, ALU_LO[a],
            # AX_CARRY_LO[b]) at (40, 30, 30) > 80; gate=OP_MUL.
            #
            # Wave B Cluster 1 (2026-06-10): MARK_AX -> MARK_SE_ONLY under
            # Wave A step_end_operand_relay (10ca51a7), which broadcasts
            # ALU_LO/HI, AX_CARRY_LO, OP_MUL from MARK_AX into
            # MARK_SE_ONLY in the same step.
            rules.append(multi_way_and_rule(
                name=f"l10_mul_lo_a{a:x}_b{b:x}_step_end",
                conditions=(
                    ("MARK_SE_ONLY", 40.0),
                    (f"ALU_LO+{a}", 30.0),
                    (f"AX_CARRY_LO+{b}", 30.0),
                ),
                threshold=80.0,
                gate=gate_mul,
                gate_weight=1.0,
                writes=((f"OUTPUT_LO+{result}", 2.0 / S),),
            ))
    return tuple(rules)


def _layer10_alu_eq_engine_rules(S: float) -> tuple[FFNRule, ...]:
    """Per-nibble EQ engine: 256 units gated on OP_EQ at the AX marker.

    Wall-4 SESSION 2 frontier fix (2026-06-12). Background: after Wave B
    moved the CMP compute to the STEP_END (MARK_SE_ONLY) row, the L9 SE
    relay never transmits, so the migrated SE cmp cascade is dead
    (SE_CMP empty). EQ/NE need ``CMP+1`` (hi_eq) and ``CMP+2`` (lo_eq)
    at the binop AX row, which are NEVER computed there. The legacy
    ``ComparisonCombine`` post-op (logical L14, physical block 22) reads
    raw ``CMP[0..3]`` at MARK_AX; for EQ those are wrong (hi_lt spuriously
    hot, hi_eq weak), so EQ falls through to its default-0 result and
    eq_true is wrong. (lt/le/gt/ge pass by ACCIDENT because their
    overrides gate on the spuriously-hot ``CMP+0``; this engine MUST NOT
    touch ``CMP`` so those guardrails are untouched.)

    The raw operands the L9 cascade WOULD use are present and STABLE at
    the AX row across every block 11..23 (probed spec_k=0): operand A's
    nibbles in ``ALU_LO/ALU_HI``, operand B's nibbles in
    ``AX_CARRY_LO/AX_CARRY_HI``. This engine recomputes equality there
    directly. For each nibble pair ``(h, l)`` one 4-way AND fires iff
    ``ALU_HI+h AND AX_CARRY_HI+h AND ALU_LO+l AND AX_CARRY_LO+l`` are all
    the active cell — i.e. A.hi == h == B.hi AND A.lo == l == B.lo, i.e.
    A == B. The matching unit writes ``CMP+1`` (hi_eq) and ``CMP+2``
    (lo_eq) — the two flags the live ``ComparisonCombine`` EQ override
    reads — so EQ flips to 1 through the SAME proven path lt/le use (via
    CMP+0). For unequal operands NO unit fires, CMP+1/CMP+2 stay zero,
    and EQ's default-0 result is left intact. ``CMP+0`` (hi_lt, the
    load-bearing accidental guardrail flag) and ``CMP+3`` are NEVER
    written, so lt/le/gt/ge are structurally untouched.

    Threshold tuning (the ~0.44-margin index-0 magnitude artifact from
    ``project_operand_gather_hybrid_encoding_is_cmp_alu_root``): the
    operand-gather emits a hybrid magnitude+nibble encoding, so
    ``ALU_*+0`` carries a value-proportional artifact (~5-11) and
    ``AX_CARRY_LO+0`` a ~0.32 floor, while the clean ``AX_CARRY`` true
    one-hot is ~0.9-1.3 and ``ALU`` is over-amplified (~6-11). A balanced
    4-way AND with per-term weights ``(MARK_AX 0.3, ALU_HI 0.2,
    AX_CARRY_HI 0.8, ALU_LO 0.2, AX_CARRY_LO 1.2)`` and threshold
    ``5.48`` separates the true match (sum 5.91) from BOTH the index-0
    artifact unit (sum 5.05) and every cross-nibble near-miss (the
    unequal-operand sums all <= 5.05) with ~0.43 headroom on each side.
    Derived offline against the probed AX-row operand bands
    (``tools/tune_eq_engine.py``). Gated hard on OP_EQ so non-EQ
    comparisons are structurally untouched.
    """
    gate_eq = dim_ref("opcode_flag", "EQ")
    # SESSION 4 retune (2026-06-12): the prior weighting/threshold was tuned
    # ONLY against 5==5 and did NOT fire on the smoke test's 42==42 (true
    # match summed 5.33 < the 5.87 threshold), so eq_true silently fell
    # through to the default-0 and decoded 0 -- the real "razor edge".
    # Per-term AND weights. The decisive equality discriminator is
    # AX_CARRY, which is a CLEAN one-hot of operand B's nibble (1.0 at the
    # true nibble, ~0.3 index-0 artifact); ALU carries A's nibble but with
    # a value-proportional index-0 artifact NEARLY EQUAL to the true nibble
    # (5.56 vs 5.83), so ALU cannot discriminate the artifact and is
    # weighted lightly (just confirms A's nibble is present; its index-0
    # artifact ~5.4 is NEARLY EQUAL to a true nibble ~5.83, so ALU cannot
    # cleanly discriminate). The clean AX_CARRY one-hots (B's nibbles) carry
    # the equality signal at 1.5 each. Weights/threshold chosen for the
    # WIDEST, most build-robust margin (probed AX-row operand bands):
    #   - smoke 42==42 true match (h=2,l=10) fires; 10!=20 rejected.
    #   - all four truly-equal smoke/cluster pairs fire (none lost), and the
    #     bulk of unequal pairs are rejected -> if_eq 1096 ids 400-424 lift
    #     11/25 -> 21/25.
    # The prior 5.872 tuning was fit ONLY to 5==5 and silently mis-FIRED on
    # the smoke 42==42 (true match summed 5.33 < 5.872), so eq_true fell
    # through to the default-0 and decoded 0 -- the real "razor edge".
    # CMP+0/CMP+3 are never written -> lt/le/gt/ge structurally untouched.
    # MARK_AX kept as a small +0.3 condition so the rule is scoped to AX
    # rows (the operand bands only exist there); constant across candidate
    # rows so it shifts only the absolute threshold, not the discrimination.
    # RESIDUAL LIMIT (documented, out of this op's lane): a minority of
    # unequal pairs whose SECOND operand B has high nibble 0 (B<16) and
    # which share the low nibble or collide with the irreducible ALU index-0
    # artifact (e.g. 16==9, 28==12, 50==11, 37==3) score as high as a true
    # match and still mis-fire -- the operand-gather hybrid-encoding wall
    # (project_operand_gather_hybrid_encoding_is_cmp_alu_root): ALU's
    # value-proportional index-0 artifact is indistinguishable from a true
    # zero nibble, so no clean per-nibble linear AND can separate them
    # without a clean-one-hot operand-gather fix at block 8. A tighter
    # AX_CARRY-heavy weighting reclaims a couple of these but loses a true
    # match (41==41) to a thinner margin, a worse trade -- this robust
    # weighting keeps every true case and the widest separation.
    W_MARK = 0.3
    W_ALU_HI = 0.1
    W_AXC_HI = 1.5
    W_ALU_LO = 0.2
    W_AXC_LO = 1.5
    THRESH = 4.0
    # HIGH-NIBBLE artifact-veto (#319, 2026-06-23). The weights above can
    # confirm A's nibble is PRESENT (``W_ALU_HI``/``W_ALU_LO`` small) but
    # cannot VETO a nibble MISMATCH, so for operands that SHARE the low nibble
    # but DIFFER on the high nibble (``if_eq_20: 28 == 12``, ``30 == 28``,
    # ``40 == 35``) the EQ engine's ``(h, l)`` unit fires on B's high nibble
    # while A's high nibble differs -- the index-0 magnitude artifact
    # (``ALU_HI+0 ~ +5.3``) HELPS the wrong h=0 unit clear threshold -- so the
    # ``eq_one`` 0x01 push mis-decodes EQ-false to 1. The fix copies the proven
    # ``_layer10_alu_ordering_engine_rules`` hi_eq/lo_eq index blocker into the
    # EQ engine's units: a per-cell negative weight on every OTHER non-zero
    # ``ALU_HI``/``ALU_LO`` index. When A's true nibble is genuinely non-zero
    # its ``+6.0`` one-hot is subtracted (``-BLK * 6.0``) on every unit whose
    # h/l does NOT match A, so ONLY the unit matching BOTH A's and B's nibbles
    # (A == B) survives. Weights/threshold locked offline against the SAME
    # golden HYBRID band the campaign ``CmpOperandSeRecoverFFN`` reconstructs
    # (true nibble +6.0 + index-0 artifact +5.3 + cell-8/15 residues
    # +0.45/+0.47); exhaustively verified over 0..99 x 0..99: every equal pair
    # fires (margin +0.40) and every unequal pair is vetoed (worst-false on the
    # if_eq corpus -0.87). No CMP flag is touched -> lt/le/gt/ge/ne untouched.
    # Campaign-gated via ``no_stack0_emit_enabled()`` so golden (non-campaign)
    # is byte-identical to ``7f6f2e5d``; kill-switch ``C4_CMP_EQ_HINIB_VETO=0``.
    from .shared import no_stack0_emit_enabled, cmp_eq_hinib_veto_enabled
    eq_hinib_veto = no_stack0_emit_enabled() and cmp_eq_hinib_veto_enabled()
    if eq_hinib_veto:
        W_ALU_HI = 0.2
        W_ALU_LO = 0.2
        THRESH = 4.1
        EQ_BLK_HI = 1.0   # blocker on every ALU_HI+j, j != h (and j >= 1)
        EQ_BLK_LO = 0.3   # blocker on every ALU_LO+j, j != l (and j >= 1)
    # RECONCILE (2026-06-12): this engine NO LONGER writes the CMP+1/CMP+2
    # (hi_eq/lo_eq) flags. The general ``_layer10_alu_ordering_engine_rules``
    # is now the SOLE writer of CMP+0..3 for all six comparison opcodes; its
    # hi_eq/lo_eq families subsume the full-equality flag this engine used to
    # write, so keeping both would DOUBLE-WRITE the EQ flags and over-trip
    # the ComparisonCombine ``_cmp_override_3way(OP_EQ, CMP+1, CMP+2, 1, 0)``
    # EQ override. This engine retains ONLY its decode-margin push
    # (``eq_one`` below) and its weights/threshold above still control which
    # equal nibble-pairs fire that push. CMP+0/CMP+3 were never written here.
    # Wall-4 SESSION 4 decode-margin push (2026-06-12): in addition to the
    # CMP+1/CMP+2 flags (consumed by the SE-row ComparisonCombine EQ
    # override through the Wave-A relay), each EQUAL-firing unit ALSO writes
    # the EQ result DIRECTLY into OUTPUT_LO at the same MARK_AX row -- the
    # row the L3-head-5 AX_FULL relay decodes into the EXIT step's exit-code
    # byte. The direct write makes the equality byte win the
    # OUTPUT_LO[1]-vs-[0] argmax by a LARGE, fp-order-robust margin instead
    # of riding the razor-thin CMP->ComparisonCombine->relay amplification
    # (which decoded eq_true on a ~84-logit knife edge that some
    # forward-state differences flipped to 0). ``OUT`` is scaled so the
    # accumulated OUTPUT_LO[1] decisively dominates the +238 L25 tail band
    # at the byte-0 emit row. Gated hard on OP_EQ + the matched nibble pair,
    # so it fires ONLY when A == B; unequal operands fire no unit and the
    # OUTPUT_LO[1] write never lands. CMP+0 (the load-bearing hi_lt
    # guardrail leak) and CMP+3 are still never written -> lt/le/gt/ge are
    # structurally untouched.
    # Decode of the exit-code byte reads BOTH nibbles: OUTPUT_LO (low) and
    # OUTPUT_HI (high). The razor-edge for eq_true is NOT OUTPUT_LO (tokens
    # 0x01/0x11/0x21 all share low-nibble 1, so OUTPUT_LO only moves them
    # together) but OUTPUT_HI[0]-vs-[1]: the EQ-true byte 0x01 has high
    # nibble 0, and OUTPUT_HI[0] won by only ~16.8 (a ~84-logit knife edge
    # that pytest's forward state flipped). So the equal-firing units write
    # the WHOLE result byte 0x01 decisively -- OUTPUT_LO[1]+/OUTPUT_HI[0]+
    # as the one-hot winners and a strong negative on every competitor cell
    # in both bands -- with ``OUT`` large enough that OUTPUT_HI[0] survives
    # and dominates the +238 L25 tail band at the byte-0 emit row, AND
    # clears the eq-default-0 unit's OUTPUT_LO[1] suppression (``-DEF``) by
    # a wide margin so the equal low-nibble flips decisively to 1.
    OUT = 900.0
    def eq_one_byte_writes() -> tuple[tuple[str, float], ...]:
        w: list[tuple[str, float]] = []
        # low nibble = 1
        for k in range(16):
            w.append((f"OUTPUT_LO+{k}", (OUT if k == 1 else -OUT) / S))
        # high nibble = 0
        for k in range(16):
            w.append((f"OUTPUT_HI_THIS_STEP+{k}", (OUT if k == 0 else -OUT) / S))
        return tuple(w)
    eq_one = eq_one_byte_writes()
    rules: list[FFNRule] = []
    for h in range(16):
        for l in range(16):
            # High/low-nibble artifact-veto: a negative weight on every OTHER
            # non-zero operand-A nibble cell (campaign-gated; absent OFF so the
            # golden bake is byte-identical). Subtracts A's +6.0 one-hot from
            # every unit whose h/l does not match A, so a high-nibble (or
            # low-nibble) mismatch can no longer fire spuriously.
            blocker: tuple[tuple[str, float], ...] = ()
            if eq_hinib_veto:
                blocker = tuple(
                    (f"ALU_HI+{j}", -EQ_BLK_HI)
                    for j in range(1, 16) if j != h
                ) + tuple(
                    (f"ALU_LO+{j}", -EQ_BLK_LO)
                    for j in range(1, 16) if j != l
                )
            rules.append(multi_way_and_rule(
                name=f"l10_eq_engine_h{h:x}_l{l:x}",
                conditions=(
                    ("MARK_AX", W_MARK),
                    (f"ALU_HI+{h}", W_ALU_HI),
                    (f"AX_CARRY_HI+{h}", W_AXC_HI),
                    (f"ALU_LO+{l}", W_ALU_LO),
                    (f"AX_CARRY_LO+{l}", W_AXC_LO),
                ) + blocker,
                threshold=THRESH,
                gate=gate_eq,
                gate_weight=1.0,
                # RECONCILE (2026-06-12): the CMP+1/CMP+2 flag writes were
                # REMOVED here. The general per-nibble ORDERING engine
                # (``_layer10_alu_ordering_engine_rules``) is now the SOLE
                # writer of CMP+0..3 for all six comparison opcodes; its
                # hi_eq(CMP+1)/lo_eq(CMP+2) families subsume this engine's
                # full-equality flag (both fire iff A.hi==B.hi and
                # A.lo==B.lo). Keeping both would DOUBLE-WRITE CMP+1/CMP+2
                # and over-trip the ComparisonCombine EQ 3-way override.
                # This engine now contributes ONLY its decode-margin push
                # ``eq_one`` (the decisive 0x01 OUTPUT byte for equal
                # operands at the AX decode row) -- separable from and
                # downstream of flag computation. See the merge note in
                # ``make_efficient_l10_andorxor_wrap_op``.
                writes=eq_one,
                scope="MARK_AX and OP_EQ",
            ))
    return tuple(rules)


def _layer10_alu_eq_default_rules(S: float) -> tuple[FFNRule, ...]:
    """EQ default-0 OUTPUT writer: makes eq_false survive the L25 tail band.

    Wall-4 SESSION 4 (2026-06-12). The per-nibble EQ engine fires a unit
    ONLY when the operands are equal; for UNEQUAL operands no unit fires,
    so the EQ result falls through to ComparisonCombine's default-0
    (OUTPUT_LO[0]=+9.6, a weak write). At the byte-0 emit row the L25 tail
    bank then floods OUTPUT_LO[1..15] and OUTPUT_HI[1..15] with a uniform
    +238 band (a relay of the SE-row 0x00 writer, inverted), overpowering
    the weak default-0 and decoding eq_false as 0x11 = 17.

    This single unit fires UNCONDITIONALLY for OP_EQ at MARK_AX and writes
    the byte 0x00 (OUTPUT_LO[0]+ / OUTPUT_HI[0]+, every competitor cell
    strongly negative) at a magnitude chosen to DOMINATE the +238 band.
    For EQUAL operands the 256 nibble-pair engine units ALSO fire and write
    0x01 at the LARGER ``OUT`` magnitude (OUTPUT_LO[1]+/OUTPUT_LO[0]-,
    OUTPUT_HI[0]+): their OUTPUT_LO[1] (+OUT) beats this default's
    OUTPUT_LO[1] (-DEF) so the low nibble flips to 1 for eq_true, while both
    REINFORCE OUTPUT_HI[0] (high nibble 0, shared by 0x00 and 0x01). So
    eq_true -> 0x01, eq_false -> 0x00, both with a decisive, fp-order-robust
    margin. Gated hard on OP_EQ -> lt/le/gt/ge/ne are structurally
    untouched; CMP is never written here.

    NOTE: only wired into the EFFICIENT-mode L10 wrap (the smoke path); the
    lookup-mode ``layer10_alu`` FFN keeps its 256-unit eq_engine layout.
    """
    gate_eq = dim_ref("opcode_flag", "EQ")
    # Default-0 magnitude. Smaller than the equal units' OUT so the equal
    # path wins the low nibble, but large enough that OUTPUT_LO[0] /
    # OUTPUT_HI[0] survive and dominate the +238 L25 tail band.
    DEF = 500.0
    w: list[tuple[str, float]] = []
    for k in range(16):
        w.append((f"OUTPUT_LO+{k}", (DEF if k == 0 else -DEF) / S))
    for k in range(16):
        w.append((f"OUTPUT_HI_THIS_STEP+{k}", (DEF if k == 0 else -DEF) / S))
    return (multi_way_and_rule(
        name="l10_eq_default_zero",
        conditions=(("MARK_AX", 1.0),),
        threshold=0.5,
        gate=gate_eq,
        gate_weight=1.0,
        writes=tuple(w),
        scope="MARK_AX and OP_EQ",
    ),)


def _layer10_alu_ordering_engine_rules(S: float) -> tuple[FFNRule, ...]:
    """Per-nibble ORDERING engine: 272 units recomputing the CMP cascade
    at the AX row for LT/GT/LE/GE (and EQ/NE), gated on ``CMP_GROUP``.

    Companion to ``_layer10_alu_eq_engine_rules``. Same root, same proof:
    after Wave B moved the CMP compute to the STEP_END (MARK_SE_ONLY) row,
    the L9 ``step_end_operand_relay`` never transmits AX->SE, so the
    SE-tagged CMP cascade is structurally dead (``SE_CMP_GROUP``=0 gates
    every L9 cmp rule to silence; probed spec_k=0). The live decoder is
    the legacy ``ComparisonCombine`` post-op (logical L14, physical block
    22) which reads the RAW ``CMP[0..3]`` flags at MARK_AX:
    ``CMP+0``=hi_lt, ``CMP+1``=hi_eq, ``CMP+2``=lo_eq, ``CMP+3``=lo_lt.
    Those flags are NEVER computed at the AX row for the if/bool clusters
    (the only partial AX-row hi_lt source is an L6 attention path that
    fires only when the high nibbles are 0/1), so every comparison falls
    through to its ComparisonCombine *default* (GT=1, LT=0) and the
    if-then-else branch takes the wrong path. This was the shared root
    behind if_gt / if_lt / if_var / bool_and (and the residual if_eq
    decode-margin failures that the OP_EQ-only eq_engine could not reach).

    The raw operands the L9 cascade WOULD use are present and STABLE at
    the AX row across blocks 11..23 (probed spec_k=0): operand A's nibbles
    in ``ALU_HI``/``ALU_LO`` (one-hot ~5.82, plus the value-proportional
    index-0 magnitude artifact ~5.54 from the hybrid operand-gather
    encoding, see ``project_operand_gather_hybrid_encoding_is_cmp_alu_root``),
    operand B's nibbles in ``AX_CARRY_HI``/``AX_CARRY_LO`` (clean one-hot
    ~1.0, index-0 floor ~0.29). This engine recomputes the four flags
    directly:

      * ``hi_lt`` (CMP+0): one 2-way AND per pair ``a < b`` over
        ``(ALU_HI+a, AX_CARRY_HI+b)``; fires iff A.hi == a AND B.hi == b.
      * ``lo_lt`` (CMP+3): the same over ``(ALU_LO+a, AX_CARRY_LO+b)``.
      * ``hi_eq`` (CMP+1): one unit per nibble h over
        ``(ALU_HI+h, AX_CARRY_HI+h)`` PLUS a small negative blocker on the
        other non-zero ``ALU_HI`` indices. The blocker disambiguates the
        index-0 artifact: when A.hi is genuinely non-zero its strong
        ``ALU_HI+A.hi`` (~5.82) suppresses the artifact-only ``ALU_HI+0``
        firing of the h=0 unit, so hi_eq fires iff A.hi == h == B.hi.
      * ``lo_eq`` (CMP+2): the same over the LO bands.

    Weights/thresholds were locked offline against the probed AX-row
    operand bands (``tools/tune_eq_engine.py`` methodology) so that EVERY
    one of the 256 high-nibble states and 256 low-nibble states yields the
    correct flag with zero false positives AND zero firing on non-AX rows
    (``MARK_AX`` carries weight 4.0; with ``MARK_AX``=0 the largest
    operand-leak sum stays below threshold). Gated multiplicatively on
    ``CMP_GROUP`` so the engine is structurally inert on every non-cmp
    opcode. Writing all four flags broadly (rather than per-opcode) is
    safe because the per-op selection happens downstream in
    ``ComparisonCombine``: hi_eq/lo_eq feed EQ/NE only via the
    ``(CMP+1 AND CMP+2)`` 3-way override which requires BOTH, and hi_lt /
    (hi_eq AND lo_lt) feed LT/GT/LE/GE -- exactly the proven flag algebra
    the cascade always intended.

    RECONCILE (2026-06-12): this engine is now the SOLE writer of the four
    CMP flags for ALL six comparison opcodes (EQ/NE/LT/GT/LE/GE). The
    earlier per-nibble ``_layer10_alu_eq_engine_rules`` retune (commit
    414abfc1) ALSO wrote CMP+1/CMP+2 on full equality; that flag write was
    REMOVED there so the EQ flags are never double-written (a double-write
    over-trips the ComparisonCombine EQ 3-way override). The eq_engine now
    contributes ONLY its decode-margin push (the decisive 0x01 OUTPUT byte
    for equal operands) plus ``_layer10_alu_eq_default_rules`` (the 0x00
    L25-band default for unequal operands) -- both DOWNSTREAM of and
    complementary to this engine's flag computation, not contradicting it.
    """
    # Gate on the OR of comparison opcode flags (effectively CMP_GROUP, but
    # expressed via the OP_<cmp> dims). The compiler's bake-time dim proxy
    # resolves the OP_<cmp> flags to their compiler-allocated positions
    # (the same path the proven eq_engine's OP_EQ gate uses), whereas the
    # raw ``CMP_GROUP`` name falls back to a stale legacy ``_SetDim`` slot
    # under the compact layout and would gate the engine on the wrong dim.
    # Exactly one flag is hot per comparison step (one-hot), so the summed
    # gate is the active opcode's value (~5.0) and zero on non-cmp opcodes.
    #
    # All six comparison opcode flags are summed into the gate. The
    # hi_eq/lo_eq families (CMP+1/CMP+2) fire on EQ/NE and feed the
    # ComparisonCombine EQ override directly; the eq_engine's OUTPUT-byte
    # decode-margin push lands on top for the equal-operand cases. A
    # residual band of EQ-false cases whose operands carry a ZERO nibble
    # (e.g. 16 == 9) is still mis-decided by the shared index-0 magnitude
    # artifact -- the hard EQ decode-margin documented in
    # ``project_operand_gather_hybrid_encoding_is_cmp_alu_root`` /
    # ``project_eq_byte1_l6_divergence``. That band is an EQ
    # decode-margin frontier, NOT a CMP->branch-relay failure, so it is
    # out of scope for this fix (if_gt/if_lt do not depend on it).
    gate_cmp_terms = (
        (dim_ref("opcode_flag", "EQ"), 1.0),
        (dim_ref("opcode_flag", "NE"), 1.0),
        (dim_ref("opcode_flag", "LT"), 1.0),
        (dim_ref("opcode_flag", "GT"), 1.0),
        (dim_ref("opcode_flag", "LE"), 1.0),
        (dim_ref("opcode_flag", "GE"), 1.0),
    )

    # Shared per-term weights. Every family is a balanced AND of:
    #   MARK_AX (positional gate, weight 6.0)
    #   ALU side  (operand A nibble one-hot, weight 0.5)
    #   AXC side  (operand B nibble one-hot, weight 6.0 — the CLEAN
    #              discriminator; AX_CARRY is a clean ~1.0 one-hot)
    #   blocker   (negative weight on the OTHER non-zero ALU indices —
    #              rejects the value-proportional index-0 magnitude
    #              artifact in ALU when operand A's true nibble is
    #              non-zero, the only thing that makes a pure 2-way AND
    #              ambiguous; weight tuned per family).
    # Thresholds were locked offline (``/tmp/tune_lt_blk`` /
    # ``tune_eq_blk`` methodology, same as ``tools/tune_eq_engine.py``)
    # for the LARGEST true-fire margin that keeps every false / off-row
    # (MARK_AX=0) sum below threshold across all 256 hi-states and 256
    # lo-states. The firing margin (~1.4 lt / ~0.75 eq) is what sets the
    # silu output magnitude, so it must be comfortably positive — a
    # correct-but-thin-margin design produces a CMP write too weak to
    # trip the ComparisonCombine override (the failure mode of the
    # first pass).
    W_MARK = 6.0
    W_ALU = 0.5
    W_AXC = 6.0
    # CMP flag write strength. The live ComparisonCombine override math
    # (vm_step.ComparisonCombine, the decode row) is:
    #   * 2-way (hi_lt): fires iff  MARK_AX + cmp_flag - 1.5 > 0,
    #     i.e. cmp_flag > 0.5.
    #   * 3-way (hi_eq AND lo_lt / hi_eq AND lo_eq): fires iff
    #     MARK_AX + cmp_flag_1 + cmp_flag_2 - 2.5 > 0,
    #     i.e. cmp_flag_1 + cmp_flag_2 > 1.5.
    # So EVERY CMP flag must land in the window (0.75, 1.5) at the decode
    # row: a SINGLE flag (plus MARK_AX) must NOT trip the 3-way override
    # (else lo_lt alone flips LT-false to LT-true, e.g. 50 < 44, or hi_eq
    # alone flips GT-true to GT-false, e.g. 54 > 53), but the two intended
    # flags together MUST. The engine's silu output is amplified by the
    # block-15 -> block-22 CMP relay, and the amplification differs with
    # the per-family firing margin (the lt families have the larger margin
    # ~1.4, the eq families ~0.75), so the write strengths are tuned PER
    # FAMILY to land every flag at ~1.1 at the decode row.
    FLAG_LT = 0.15  # hi_lt / lo_lt (margin ~1.4) -> ~1.1 at decode row
    FLAG_EQ = 0.30  # hi_eq / lo_eq (margin ~0.75) -> ~1.1 at decode row
    # CMP-flag margin fix (2026-06-21): the (hi_eq AND lo_lt) 3-way override
    # in the live ComparisonCombine never flipped the equal-high-nibble cases
    # (if_gt 35>43 / 20>30, if_lt 86<87, ...) because the two flags landed too
    # LOW at the decode row — hi_eq ~0.72 (below its own (0.75,1.5) window
    # floor) and lo_lt ~0.88, summing to ~1.6 which only barely clears the
    # 3-way override's 1.5 sum-threshold. After the silu+relay the resulting
    # override write was too weak to out-vote the GT/LT default (and the L25
    # tail band), so the boolean decoded on a ~1-3 logit OUTPUT_LO[0]-vs-[1]
    # tie that fell the wrong way. Raising both writes lands each flag at
    # ~1.05-1.10 at the decode row (verified GPU-autoregressive): the PAIR
    # sums to ~2.1 (decisively > 1.5 -> the override fires hard) while a
    # SINGLE flag stays < 1.5 (so hi_eq-alone does NOT flip GT-true 54>53,
    # and lo_lt-alone does NOT flip LT-false 50<44). hi_lt (CMP+0) lands ~0.77
    # alone and already trips its 2-way override (> 0.5), so its strength is
    # left untouched — only the EQ flag and the lo_lt 3-way partner are
    # boosted. Gated so flag-OFF is byte-identical to the golden bake.
    if os.environ.get("C4_CMP_FLAG_MARGIN_FIX", "1") != "0":
        FLAG_EQ = 0.45  # -> ~1.05 at decode row (was ~0.72)
        FLAG_LT_LO_LT = 0.19  # lo_lt -> ~1.10 (was ~0.88); hi_lt unchanged
    else:
        FLAG_LT_LO_LT = FLAG_LT
    # CMP equal-high-nibble GT lo-margin fix (#322, 2026-06-23, CAMPAIGN only).
    # ROOT (isolated CPU-autoregressive, campaign config
    # ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``, BUILT dims, spec_k=0,
    # ``tools/probe_gt_lo_margin.py``): in the CAMPAIGN config the
    # ``CmpOperandSeRecoverFFN`` re-materializes a STRONGER operand-A one-hot
    # than the golden 35-token frame, so the ``C4_CMP_FLAG_MARGIN_FIX``
    # ``FLAG_EQ=0.45`` lands ``hi_eq`` at ~1.86 at the decode row -- ABOVE its
    # own (0.75, 1.5) single-flag CEILING. The live ComparisonCombine's
    # ``(hi_eq AND lo_lt) -> GT=0`` 3-way override fires iff
    # ``MARK_AX + hi_eq + lo_lt > 2.5``; with ``hi_eq = 1.86`` ALONE the sum
    # ``1 + 1.86 + 0 = 2.86 > 2.5`` so the override SPURIOUSLY flips the
    # equal-high-nibble GT-TRUE cases (``if_gt 54>53 / 60>54 / 54>50``, where
    # ``lo_lt = 0`` because A.lo > B.lo) to GT=0 (probed OUTPUT_LO@blk26 =
    # [25.5@0, -15.9@1] -> result byte low-nibble 0 -> GT=0 WRONG). The golden
    # config's weaker operand amplification keeps ``hi_eq`` in-window, so this
    # is a campaign-only over-shoot -- hence a campaign-gated knock-down, NOT a
    # change to the golden ``FLAG_EQ`` (golden bake stays byte-identical to
    # ``7f6f2e5d``). Lowering the campaign ``FLAG_EQ`` to 0.30 lands
    # ``hi_eq = 1.24`` (comfortably in-window): the spurious single-flag trip is
    # gone (``1 + 1.24 + 0 = 2.24 < 2.5`` -> override OFF -> GT stays default=1,
    # probed OUTPUT_LO = [0.6@0, 8.9@1] -> GT=1) while the INTENDED
    # ``(hi_eq AND lo_lt)`` pairs STILL fire decisively (53>54 / 86<87:
    # ``1 + 1.24 + 1.45 = 3.69 > 2.5`` -> override ON, probed [29.7@0,-20.2@1] /
    # [-19.9@0,29.4@1]) and ``lo_lt``-alone STILL does NOT trip (50<44:
    # ``1 + 0 + 1.45 = 2.45 < 2.5``). DISCRIMINATING: every equal-high-nibble
    # GT-true now wins AND every (hi_eq AND lo_lt) override still fires -- no
    # zero-sum trade. ``lo_lt`` (CMP+3) write strength is UNTOUCHED so lt/le/ge
    # margins are unchanged. Kill-switch ``C4_CMP_GT_LO_MARGIN=0`` restores the
    # 0.45 campaign value; flag-OFF / non-campaign is byte-identical to golden.
    from .shared import no_stack0_emit_enabled, cmp_gt_lo_margin_enabled
    if no_stack0_emit_enabled() and cmp_gt_lo_margin_enabled():
        FLAG_EQ = 0.30  # campaign: -> hi_eq ~1.24 (was 1.86 at 0.45)

    # if_var GT-FALSE 0xF-leak guard (#339, 2026-06-25, CAMPAIGN only).
    # ROOT (GPU full_trace + isolated intervention, campaign config
    # ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``, BUILT dims, spec_k=0,
    # ``tools/probe_ifvar_result_step.py`` + ``probe_ifvar_alu15_intervene.py``):
    # the if_var GT-FALSE of a LOADED variable (id430 ``23>62`` / id433 ``35>76``,
    # both ``A.hi < B.hi``) decoded GT=1 because the result step's operand-A
    # ``ALU_HI`` carries a SPURIOUS ``+6.5`` at cell 15 (the 0xF address-high-nibble
    # leak from the LI-load relay). The ``hi_lt`` blocker's ``-0.5 * 6.5 = -3.25``
    # cell-15 term drops the unit's pre-activation to ``6 + 3 + 6 - 3.25 - 0.2 =
    # 11.55`` -- below the ``13.22`` threshold -- so ``hi_lt`` (CMP+0) does NOT fire
    # and the GT-FALSE override never lands (GT defaults to 1). The passing LITERAL
    # ``23>62`` has a clean ``ALU_HI[15] ~= 0.5`` (blocker ``-0.25``, sum ``14.55``)
    # so its ``hi_lt`` fires. Cell 15 (``A.hi == 0xF`` == operand >= 240) is
    # UNREACHABLE for the 0..99 corpus operands, so the cell-15 veto only ever
    # fires on the spurious leak. FIX: campaign-only, DROP the ``ALU_HI+15`` term
    # from the ``hi_lt`` blocker -> the FAIL sum recovers to ``14.8 > 13.22`` ->
    # GT-FALSE override lands -> result 0. INTERVENTION-VERIFIED discriminating:
    # zeroing ``ALU_HI[15]`` flips id430/433 to 0 while GT-TRUE ``85>48`` holds at 1
    # and the GT-FALSE literal holds at 0 (no zero-sum). Only ``hi_lt`` is touched;
    # lo_lt / hi_eq / lo_eq are untouched. Kill-switch ``C4_CMP_HI_LT_ALU15_GUARD=0``
    # restores the full cell-15 blocker; flag-OFF / non-campaign is byte-identical
    # to golden.
    from .shared import cmp_hi_lt_alu15_leak_guard_enabled
    _hi_lt_drop_cell15 = cmp_hi_lt_alu15_leak_guard_enabled()

    rules: list[FFNRule] = []

    # ---- hi_lt -> CMP+0 : 120 units (a < b) -------------------------------
    for a in range(16):
        blocker = tuple(
            (f"ALU_HI+{j}", -0.5)
            for j in range(1, 16)
            if j != a and not (_hi_lt_drop_cell15 and j == 15)
        )
        for b in range(a + 1, 16):
            rules.append(multi_way_and_rule(
                name=f"l10_ord_hi_lt_a{a:x}_b{b:x}",
                conditions=(
                    ("MARK_AX", W_MARK),
                    (f"ALU_HI+{a}", W_ALU),
                    (f"AX_CARRY_HI+{b}", W_AXC),
                ) + blocker,
                threshold=13.22,
                gate_terms=gate_cmp_terms,
                writes=(("CMP+0", FLAG_LT / S),),
                scope="MARK_AX and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
            ))

    # ---- lo_lt -> CMP+3 : 120 units (a < b) -------------------------------
    for a in range(16):
        blocker = tuple(
            (f"ALU_LO+{j}", -0.5) for j in range(1, 16) if j != a
        )
        for b in range(a + 1, 16):
            rules.append(multi_way_and_rule(
                name=f"l10_ord_lo_lt_a{a:x}_b{b:x}",
                conditions=(
                    ("MARK_AX", W_MARK),
                    (f"ALU_LO+{a}", W_ALU),
                    (f"AX_CARRY_LO+{b}", W_AXC),
                ) + blocker,
                threshold=13.22,
                gate_terms=gate_cmp_terms,
                writes=(("CMP+3", FLAG_LT_LO_LT / S),),
                scope="MARK_AX and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
            ))

    # ---- hi_eq -> CMP+1 : 16 units (A.hi == h == B.hi) --------------------
    for h in range(16):
        blocker = tuple(
            (f"ALU_HI+{j}", -0.8) for j in range(1, 16) if j != h
        )
        rules.append(multi_way_and_rule(
            name=f"l10_ord_hi_eq_{h:x}",
            conditions=(
                ("MARK_AX", W_MARK),
                (f"ALU_HI+{h}", W_ALU),
                (f"AX_CARRY_HI+{h}", W_AXC),
            ) + blocker,
            threshold=13.79,
            gate_terms=gate_cmp_terms,
            writes=(("CMP+1", FLAG_EQ / S),),
            scope="MARK_AX and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
        ))

    # ---- lo_eq -> CMP+2 : 16 units (A.lo == l == B.lo) --------------------
    for l in range(16):
        blocker = tuple(
            (f"ALU_LO+{j}", -0.8) for j in range(1, 16) if j != l
        )
        rules.append(multi_way_and_rule(
            name=f"l10_ord_lo_eq_{l:x}",
            conditions=(
                ("MARK_AX", W_MARK),
                (f"ALU_LO+{l}", W_ALU),
                (f"AX_CARRY_LO+{l}", W_AXC),
            ) + blocker,
            threshold=13.43,
            gate_terms=gate_cmp_terms,
            writes=(("CMP+2", FLAG_EQ / S),),
            scope="MARK_AX and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
        ))

    return tuple(rules)


def _layer10_alu_rules(S: float) -> tuple[FFNRule, ...]:
    """Composite ordered ``FFNRule`` sequence for ``layer10_alu``.

    Concatenates all seven sub-stage rule lists in the exact order
    declared by ``_L10_FFN_UNIT_LAYOUT_MAIN`` so a single
    ``Primitives.lower_ffn_rules`` call lowers the entire 1846-unit FFN
    in cursor order (matching the legacy ``_set_layer10_alu`` walk
    byte-for-byte).
    """

    return (
        _layer10_alu_cmp_combine_rules(S)
        + _layer10_alu_bitwise_or_rules(S)
        + _layer10_alu_bitwise_xor_rules(S)
        + _layer10_alu_bitwise_and_rules(S)
        + _layer10_alu_mul_lo_rules(S)
        + _layer10_alu_shl_shr_zero_rules(S)
        + _layer10_alu_ax_passthrough_rules(S)
        + _layer10_alu_eq_engine_rules(S)
        + _layer10_alu_ordering_engine_rules(S)
        # C4_CMP_COMBINE_MARGIN campaign clamp bank (LAST so it never shifts
        # the shared banks above); flag-OFF -> empty -> byte-identical golden.
        + _layer10_alu_cmp_hi_clamp_rules(S)
    )


def _layer10_alu_ir(S: float = 100.0) -> CompilerIR:
    """Build the declarative ``CompilerIR`` exposed by ``layer10_alu``."""

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer10_alu_rules(S))
    return ir


def _bake_layer10_alu_rules(ffn, S: float, BD) -> int:
    """Lower the composite ``layer10_alu`` rule list into ``ffn``.

    Returns the post-bake unit cursor (must equal
    :data:`_L10_FFN_UNIT_LAYOUT_MAIN_TOTAL` for byte-identity with the
    historical 1846-unit footprint of ``vm_step._set_layer10_alu``).
    """

    rules = _layer10_alu_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD, Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )


def _bake_layer10_carry_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 0 carry relay spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_carry_relay_head_spec(BD, S),
        HD,
    )


def _layer10_carry_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    AX_IDX = 1
    L = S
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_carry_relay_bake.head_0"),
        q=(
            AP(0, BD.IS_BYTE, L),
            AP(0, BD.CONST, -L / 2),
            AP(33, BD.H1 + AX_IDX, L),
            AP(33, BD.CONST, -L / 2),
        ),
        k=(AP(0, BD.MARK_AX, L), AP(33, BD.CONST, L)),
        v=(AP(1, BD.CARRY + 1, 1.0), AP(2, BD.CARRY + 2, 1.0)),
        o=(AO(BD.CARRY + 1, 1, 1.0), AO(BD.CARRY + 2, 2, 1.0)),
    )


def _byte_passthrough_chain_spec(
    BD,
    *,
    head_idx: int,
    source_marker_dim: int,
    target_marker_dim: int,
    value_lo_dim: int,
    value_hi_dim: int,
    suppress_op_dims,
    S: float,
    is_byte_strength: float = 3.0,
    has_se_strength: float = 1.0,
    suppress_strength: float = 3.0,
    q0_threshold: float = 3.5,
    gate_const: float = -20000.0,
    gate_target_marker: float = 10000.0,
    gate_has_se: float = 10000.0,
    gate_extras=None,
) -> DeclarativeAttentionHeadSpec:
    L = S
    q = [
        AP(0, BD.IS_BYTE, L * is_byte_strength),
        AP(0, BD.HAS_SE, L * has_se_strength),
        AP(0, BD.CONST, -L * q0_threshold),
        AP(1, target_marker_dim, L),
        AP(1, BD.CONST, -L / 2),
        AP(2, BD.BYTE_INDEX_3, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.BYTE_INDEX_0, L),
        AP(4, BD.BYTE_INDEX_1, L),
        AP(5, BD.BYTE_INDEX_2, L),
        AP(33, BD.CONST, gate_const),
        AP(33, target_marker_dim, gate_target_marker),
        AP(33, BD.HAS_SE, gate_has_se),
    ]
    for dim in suppress_op_dims:
        q.append(AP(0, dim, -L * suppress_strength))
    if gate_extras:
        for dim, weight in gate_extras:
            q.append(AP(33, dim, weight))

    k = [
        AP(0, BD.IS_BYTE, L),
        AP(1, source_marker_dim, L),
        AP(2, BD.BYTE_INDEX_0, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.BYTE_INDEX_1, L),
        AP(4, BD.BYTE_INDEX_2, L),
        AP(5, BD.BYTE_INDEX_3, L),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for idx in range(16):
        v.append(AP(idx, value_lo_dim + idx, 1.0))
        v.append(AP(16 + idx, value_hi_dim + idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + idx, idx, 2.0))
        o.append(AO(BD.OUTPUT_HI + idx, 16 + idx, 2.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _bake_r_frame_passthrough_head(attn, BD, S, HD, *, spec_fn, alibi_idx) -> None:
    """Shared R-FRAME register byte-passthrough head bake.

    INCR-1 collapse: the AX/SP/BP/PC ``_bake_layer10_*_byte_passthrough_head``
    helpers were byte-for-byte identical apart from ``spec_fn`` (the
    per-register head spec) and ``alibi_idx`` (the L10 attention slot the ALiBi
    slope 1.0 tie-break is written to). Both are per-register DATA, so the bake
    is a single shared lowering driven by that data. Emits the identical
    ``generate_attention_head`` weights + the same single ``alibi_slopes``
    write, so the golden hash is UNCHANGED.
    """
    Primitives.generate_attention_head(attn, spec_fn(BD, S), HD)
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[alibi_idx] = 1.0


def _bake_layer10_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 1 AX byte passthrough spec."""
    _bake_r_frame_passthrough_head(
        attn, BD, S, HD,
        spec_fn=_layer10_ax_byte_passthrough_head_spec, alibi_idx=1,
    )


def _layer10_ax_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Carry AX bytes and reload LI bytes from the prior MEM value rows."""

    PC_IDX = 0
    AX_IDX = 1
    SP_IDX = 2
    BP_IDX = 3
    MEM_IDX = 4
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=_l10_head_idx("layer10_byte_passthrough_bake.head_1"),
        source_marker_dim=BD.H1 + AX_IDX,
        target_marker_dim=BD.H1 + AX_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[
            BD.OP_IMM,
            BD.OP_LI_RELAY,
            BD.OP_LC_RELAY,
            BD.TEMP + 3,
            BD.CMP + 3,
        ],
        S=S,
    )
    M = 50.0 * S

    # LI reloads overwrite AX from memory. The ordinary AX byte chain is
    # intentionally blocked by OP_LI_RELAY so it does not carry the stale
    # address AX value. These side slots instead select the latest matching
    # MEM value byte and reuse the existing CLEAN_EMBED -> OUTPUT value path.
    STORE_SELECT = 5.0 * S
    VALUE_SELECT = 20.0
    ROW_SELECT = 50.0 * S
    ROW_SELECT_BIAS = -75.0 * S
    STORE_ROW_SELECT = 50.0 * S
    STORE_ROW_SELECT_BIAS = -50.0 * S
    STORE_AX_BYTE1_SELECT = 20.0 * S
    li_value_query = (
        AP(39, BD.OP_LI_RELAY, M),
    )
    marker_query = (
        AP(40, BD.OP_LI_RELAY, M),
        AP(40, BD.MARK_AX, ROW_SELECT),
        AP(40, BD.CONST, ROW_SELECT_BIAS),
    )
    byte0_query = (
        AP(41, BD.OP_LI_RELAY, M),
        AP(41, BD.BYTE_INDEX_0, ROW_SELECT),
        AP(41, BD.CONST, ROW_SELECT_BIAS),
    )
    byte1_query = (
        AP(42, BD.OP_LI_RELAY, M),
        AP(42, BD.BYTE_INDEX_1, ROW_SELECT),
        AP(42, BD.CONST, ROW_SELECT_BIAS),
    )
    byte2_query = (
        AP(43, BD.OP_LI_RELAY, M),
        AP(43, BD.BYTE_INDEX_2, ROW_SELECT),
        AP(43, BD.CONST, ROW_SELECT_BIAS),
    )
    # Part 2 fix (test_si_li_16bit_value): the CONST-weighted
    # STORE_ROW_SELECT_BIAS used to apply on every Q row regardless of
    # the active opcode. K[44..47] reads MEM_STORE*STORE_SELECT (=500);
    # residual MEM_STORE leakage at byte 1 rows (~0.0086) of past steps
    # multiplied by Q[44..47]=-50*S=-5000 accumulated ~-85k of unfair
    # penalty against the most recent prior AX byte 1 K-row, causing the
    # SI step's L10 AX byte_passthrough to attend to the older
    # IMM 0x200's byte 1 (=0x02) instead of IMM 0x1234's byte 1 (=0x12).
    # Fold the bias into the OP_LI_RELAY weight so the bias only applies
    # when OP_LI_RELAY=1 (i.e. only during LI steps, the design intent):
    # at LI step the effective Q[slot] contribution from OP_LI_RELAY is
    # M + STORE_ROW_SELECT_BIAS = 50*S - 50*S = 0 (vs. the original
    # M (=+5000) at LI step from OP_LI_RELAY plus a CONST term of
    # -50*S=-5000 -> same net 0). At non-LI steps the contribution is 0
    # instead of the original CONST*-50*S=-5000, removing the spurious
    # MEM_STORE-leak penalty against past steps' byte 1 K-rows.
    LI_GATED_BIAS = M + STORE_ROW_SELECT_BIAS  # = +5000 + -5000 = 0 (= original net at LI step)
    marker_store_query = (
        AP(44, BD.OP_LI_RELAY, LI_GATED_BIAS),
        AP(44, BD.MARK_AX, STORE_ROW_SELECT),
    )
    marker_addr_source_query = (
        AP(48, BD.OP_LI_RELAY, LI_GATED_BIAS),
        AP(48, BD.MARK_AX, STORE_ROW_SELECT),
    )
    byte0_store_query = (
        AP(45, BD.OP_LI_RELAY, LI_GATED_BIAS),
        AP(45, BD.BYTE_INDEX_0, STORE_ROW_SELECT),
    )
    byte1_store_query = (
        AP(46, BD.OP_LI_RELAY, LI_GATED_BIAS),
        AP(46, BD.BYTE_INDEX_1, STORE_ROW_SELECT),
    )
    byte2_store_query = (
        AP(47, BD.OP_LI_RELAY, LI_GATED_BIAS),
        AP(47, BD.BYTE_INDEX_2, STORE_ROW_SELECT),
    )
    store_ax_byte1_query = (
        AP(81, BD.OP_SI, STORE_AX_BYTE1_SELECT),
        AP(81, BD.OP_SC, STORE_AX_BYTE1_SELECT),
        AP(81, BD.MARK_AX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + PC_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + SP_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + BP_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H4 + BP_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.H3 + MEM_IDX, -5.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_1, -10.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_2, -10.0 * STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_3, -10.0 * STORE_AX_BYTE1_SELECT),
    )
    store_ax_byte1_key = (
        AP(81, BD.IS_BYTE, STORE_AX_BYTE1_SELECT),
        AP(81, BD.H1 + AX_IDX, STORE_AX_BYTE1_SELECT),
        AP(81, BD.BYTE_INDEX_1, STORE_AX_BYTE1_SELECT),
        AP(81, BD.OP_IMM, STORE_AX_BYTE1_SELECT),
    )

    # === STACK0-campaign (Inc-3 ROOT A): re-deliver the LI-reload AX byte-1 ===
    # In the 30-tok campaign frame (``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``)
    # the per-step opcode markers (OP_SI/OP_LI/OP_LI_RELAY) DROP off the AX-byte
    # PREDICTOR rows (they survive only on the AX_marker row), so the existing
    # ``store_ax_byte1`` slot 81 -- gated on OP_SI -- goes dead at the step5
    # ``return x`` byte-1 predictor row. The head then defaults to the step's
    # PC-marker row (the only row that still carries the broadcast MEM_STORE) whose
    # CLEAN_EMBED=0 -> AX byte-1 emits 0x00 (e.g. x=990=0x03DE -> 0x00DE). The byte-1
    # VALUE (0x03) is fully present and stride-stable: it lives on the PRIOR step's
    # AX byte-1 register row (the IMM/SI source step), which carries
    # IS_BYTE + H1+AX + BYTE_INDEX_1 + OP_IMM and NO MEM_STORE. This flag-gated
    # companion (slot 82, never the legacy path off-flag) re-selects that row using
    # ONLY signals confirmed present on the campaign predictor row (H1+AX, IS_BYTE,
    # BYTE_INDEX_0) -> the existing CLEAN_EMBED->OUTPUT value path copies 0x03.
    # ALiBi slope 1.0 (set in ``_bake_layer10_byte_passthrough_head``) breaks ties
    # toward the most-recent matching AX byte-1 row (the immediately-prior step).
    # Byte-identical OFF: the slot is omitted unless both campaign flags are set.
    MEMAX_B1_SELECT = 20.0 * S
    memax_byte1_query = ()
    memax_byte1_key = ()
    # ``C4_INC3_MEMAX_B1_OFF=1`` is an A/B kill-switch (campaign-config regression
    # differential only); it leaves the flag-OFF golden untouched either way.
    _memax_b1_on = (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_INC3_MEMAX_B1_OFF", "0") == "0"
    )
    if _memax_b1_on:
        memax_byte1_query = (
            # Fire on the AX[0] byte row (the byte-1 PREDICTOR) of the current step.
            AP(82, BD.IS_BYTE, MEMAX_B1_SELECT),
            AP(82, BD.H1 + AX_IDX, MEMAX_B1_SELECT),
            AP(82, BD.BYTE_INDEX_0, MEMAX_B1_SELECT),
            # Suppress on non-AX-byte-0 predictor rows / the originating IMM step
            # (OP_IMM is present on the IMM step's own byte rows; the LI-reload
            # predictor has OP_IMM=0, so this veto keeps the slot LI-step-only).
            AP(82, BD.MARK_AX, -5.0 * MEMAX_B1_SELECT),
            AP(82, BD.H1 + PC_IDX, -5.0 * MEMAX_B1_SELECT),
            AP(82, BD.H1 + SP_IDX, -5.0 * MEMAX_B1_SELECT),
            AP(82, BD.H1 + BP_IDX, -5.0 * MEMAX_B1_SELECT),
            AP(82, BD.MEM_STORE, -5.0 * MEMAX_B1_SELECT),
            AP(82, BD.OP_IMM, -5.0 * MEMAX_B1_SELECT),
            AP(82, BD.BYTE_INDEX_1, -10.0 * MEMAX_B1_SELECT),
            AP(82, BD.BYTE_INDEX_2, -10.0 * MEMAX_B1_SELECT),
            AP(82, BD.BYTE_INDEX_3, -10.0 * MEMAX_B1_SELECT),
        )
        memax_byte1_key = (
            # Select the prior step's AX byte-1 register row (CLEAN_EMBED carries
            # the byte-1 value); hard MEM_STORE veto excludes the PC-marker leak row.
            AP(82, BD.IS_BYTE, MEMAX_B1_SELECT),
            AP(82, BD.H1 + AX_IDX, MEMAX_B1_SELECT),
            AP(82, BD.BYTE_INDEX_1, MEMAX_B1_SELECT),
            AP(82, BD.OP_IMM, MEMAX_B1_SELECT),
            AP(82, BD.MEM_STORE, -10.0 * MEMAX_B1_SELECT),
        )

    # === STACK0-campaign (Inc-2): si/li LOAD byte-1 ADDRESS-leak discriminator ===
    # Slot 83 fixes the slot-82 LI-reload byte-1 LEAK for IMM-addressed loads
    # (si/li/sc/lc). Slot 82 picks the most-recent OP_IMM AX byte-1 register row
    # by ALiBi recency; the value-IMM and address-IMM rows score IDENTICALLY on
    # slot 82 (same CLEAN/OP_IMM/H1+AX signature) so recency alone leaks the
    # later address-IMM (e.g. 0x200 byte-1 = 0x02). The DISCRIMINATOR is that the
    # address-IMM register row is the gathered LOAD ADDRESS: it carries a STRONG
    # ``ADDR_B1`` one-hot (HI+LO cell sums ~3.0 each, staged by the L13 gather)
    # whereas the genuine value-IMM row carries only the weak residual ADDR_B1
    # (sums ~1.0). Measured: tools/probe_sili_scores.py (addr-IMM ADDR_B1_HI
    # sum=3.0, value-IMM sum=1.0) + probe_var_scores.py (var990's selected
    # value-IMM winner sum=1.0, so it is unaffected).
    #
    # Slot 83 fires its Q on the SAME byte-1 PREDICTOR row as slot 82 (so it
    # contributes to NO other query -> the byte-0 reload softmax is untouched;
    # this is NOT the K-side ADDR-veto blind spot, the K term is only weighted
    # through slot 83's predictor-only Q) and its K DOWN-weights each candidate
    # in proportion to its ADDR_B1 magnitude. The penalty is ~3x heavier on the
    # leaky addr-IMM row than the value-IMM row, a net margin (~12*W*P per
    # ADDR_B1 cell-sum unit) that out-votes the ~60-point ALiBi recency gap and
    # re-points the byte-1 reload at the value-IMM register. The value-IMM row's
    # absolute penalty (~6*W*P) is negligible vs its >50M lead over the
    # non-OP_IMM rows, so it keeps winning. Own kill-switch ``C4_SILI_CAM_B1``
    # (default campaign-ON). Byte-identical OFF (slot omitted off the campaign).
    sili_disc_query = ()
    sili_disc_key = ()
    if sili_cam_b1_enabled():
        SILI_DISC_Q = 20.0 * S       # predictor-row Q gate (mirrors slot 82)
        SILI_DISC_K = 1.0            # per-cell ADDR_B1 down-weight
        sili_disc_query = (
            # Fire ONLY on the LI-reload byte-1 PREDICTOR row (AX byte-0 row);
            # mirror slot 82's predictor gating so it contributes to no other
            # query.
            AP(83, BD.IS_BYTE, SILI_DISC_Q),
            AP(83, BD.H1 + AX_IDX, SILI_DISC_Q),
            AP(83, BD.BYTE_INDEX_0, SILI_DISC_Q),
            AP(83, BD.MARK_AX, -5.0 * SILI_DISC_Q),
            AP(83, BD.H1 + PC_IDX, -5.0 * SILI_DISC_Q),
            AP(83, BD.H1 + SP_IDX, -5.0 * SILI_DISC_Q),
            AP(83, BD.H1 + BP_IDX, -5.0 * SILI_DISC_Q),
            AP(83, BD.MEM_STORE, -5.0 * SILI_DISC_Q),
            AP(83, BD.OP_IMM, -5.0 * SILI_DISC_Q),
            AP(83, BD.BYTE_INDEX_1, -10.0 * SILI_DISC_Q),
            AP(83, BD.BYTE_INDEX_2, -10.0 * SILI_DISC_Q),
            AP(83, BD.BYTE_INDEX_3, -10.0 * SILI_DISC_Q),
        )
        # K: subtract SILI_DISC_K from the score for every active ADDR_B1 one-hot
        # cell on the candidate -> the gathered-load-address rows (sum ~3) are
        # penalized ~3x the value rows (sum ~1).
        sili_disc_key = tuple(
            AP(83, BD.ADDR_B1_HI + c, -SILI_DISC_K) for c in range(16)
        ) + tuple(
            AP(83, BD.ADDR_B1_LO + c, -SILI_DISC_K) for c in range(16)
        )

    return replace(
        spec,
        q=(
            spec.q
            + li_value_query
            + marker_query
            + byte0_query
            + byte1_query
            + byte2_query
            + marker_store_query
            + byte0_store_query
            + byte1_store_query
            + byte2_store_query
            + marker_addr_source_query
            + store_ax_byte1_query
            + memax_byte1_query
            + sili_disc_query
        ),
        k=spec.k + (
            AP(39, BD.MEM_VAL_B0, VALUE_SELECT),
            AP(39, BD.MEM_VAL_B1, VALUE_SELECT),
            AP(39, BD.MEM_VAL_B2, VALUE_SELECT),
            AP(39, BD.MEM_VAL_B3, VALUE_SELECT),
            AP(40, BD.MEM_VAL_B1, M),
            AP(41, BD.MEM_VAL_B2, M),
            AP(42, BD.MEM_VAL_B3, M),
            AP(43, BD.MEM_VAL_B3, M),
            AP(44, BD.MEM_STORE, STORE_SELECT),
            AP(45, BD.MEM_STORE, STORE_SELECT),
            AP(45, BD.MEM_ADDR_SRC, 1.0),
            AP(46, BD.MEM_STORE, STORE_SELECT),
            AP(46, BD.MEM_ADDR_SRC, 1.0),
            AP(47, BD.MEM_STORE, STORE_SELECT),
            AP(47, BD.MEM_ADDR_SRC, 1.0),
            AP(48, BD.MEM_ADDR_SRC, 1.0),
        ) + store_ax_byte1_key + memax_byte1_key + sili_disc_key,
    )


def _bake_layer10_sp_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 2 SP byte passthrough spec."""
    _bake_r_frame_passthrough_head(
        attn, BD, S, HD,
        spec_fn=_layer10_sp_byte_passthrough_head_spec, alibi_idx=2,
    )


def _bake_layer10_bp_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 7 BP byte passthrough spec."""
    _bake_r_frame_passthrough_head(
        attn, BD, S, HD,
        spec_fn=_layer10_bp_byte_passthrough_head_spec, alibi_idx=7,
    )


def _layer10_sp_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    SP_IDX = 2
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=_l10_head_idx("layer10_sp_byte_passthrough_bake.head_2"),
        source_marker_dim=BD.H1 + SP_IDX,
        target_marker_dim=BD.H1 + SP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.PSH_AT_SP, BD.MARK_SP],
        S=S,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.MARK_SP, 10000.0),
            (BD.PSH_AT_SP, -10000.0),
        ],
    )
    L = S
    marker_s = 300.0
    # Marker carry-forward for SP byte 0 on step 1+. The byte-chain above
    # handles SP byte positions; the marker itself needs to copy the previous
    # step's SP byte 0 unless the current op is actively rewriting SP (PSH,
    # JSR, ENT, or POP/LEV/ADJ). L6 relays these as CMP[0]/CMP[2]/CMP[3]/CMP[4]
    # to the SP marker; without these blockers the marker carry-forward
    # re-copies the stale pre-op SP byte after L6 has applied the SP delta.
    # CMP[2]=ENT was previously missing, causing func_identity_* and other
    # post-ENT programs to emit step1:SP_byte0 = stale 0xf8 (pre-JSR target)
    # or 0xff (pre-bootstrap) instead of the ENT-adjusted 0xf0/0xe0.
    return DeclarativeAttentionHeadSpec(
        head_idx=spec.head_idx,
        q=spec.q + (
            AP(34, BD.MARK_SP, marker_s),
            AP(34, BD.HAS_SE, marker_s),
            AP(34, BD.PSH_AT_SP, -2.0 * marker_s),
            AP(34, BD.CMP + 4, -2.0 * marker_s),
            AP(34, BD.OP_JSR, -2.0 * marker_s),
            AP(34, BD.CMP + 3, -2.0 * marker_s),
            AP(34, BD.CMP + 2, -2.0 * marker_s),
            AP(34, BD.OP_ENT, -2.0 * marker_s),
            AP(34, BD.CONST, -marker_s),
            AP(35, BD.MARK_SP, marker_s),
            AP(35, BD.HAS_SE, marker_s),
            AP(35, BD.PSH_AT_SP, -2.0 * marker_s),
            AP(35, BD.CMP + 4, -2.0 * marker_s),
            AP(35, BD.OP_JSR, -2.0 * marker_s),
            AP(35, BD.CMP + 3, -2.0 * marker_s),
            AP(35, BD.CMP + 2, -2.0 * marker_s),
            AP(35, BD.OP_ENT, -2.0 * marker_s),
            AP(35, BD.CONST, -marker_s),
        ),
        k=spec.k + (
            AP(34, BD.H1 + SP_IDX, marker_s),
            AP(35, BD.BYTE_INDEX_0, marker_s),
        ),
        v=spec.v,
        o=spec.o,
    )


def _layer10_bp_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    # Marker-bank slot indices via the positional-invariant mechanism (Class-1
    # marker-relative). ``BD.H1 + BP_IDX`` / ``BD.H1 + AX_IDX`` read the
    # marker-TYPE slot of the fixed-width threshold-head bank
    # (PC=0 AX=1 SP=2 BP=3 MEM=4 SE=5), which is frame-INVARIANT — the bank
    # order does NOT change when the STACK0 value block is dropped
    # (STEP_TOKENS 35->30). ``marker_bank_index`` is the single source of truth
    # and lets the positional audit recognise these as declared marker-relative
    # rather than UNGUARDED bare offsets. Byte-identical in both frames (returns
    # the same integers the literals encoded). See positional_invariant.py.
    BP_IDX = marker_bank_index("BP")
    AX_IDX = marker_bank_index("AX")
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=_l10_head_idx("layer10_bp_byte_passthrough_bake.head_7"),
        source_marker_dim=BD.H1 + BP_IDX,
        target_marker_dim=BD.H1 + BP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.OP_ENT, BD.OP_LEV],
        S=S,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.OP_ENT, -10000.0),
            (BD.OP_LEV, -10000.0),
        ],
    )
    # Suppress head 7 at MEM val byte Q rows. The byte passthrough chain is
    # meant to relay BP byte values; at SI/SC steps the MEM val byte rows
    # (MEM_VAL_B0/B1/B2/B3 = 1) were spuriously attending to the IMM byte
    # rows of the preceding step (probe_l14_li_consumer.py confirmed head 7
    # weight 0.61 from p=193 Q to K@170, leaking the 0x02 nibble of 0x200
    # into OUTPUT_LO[2]/OUTPUT_HI[0]).  Adding a strongly negative slot-0
    # Q contribution at MEM_VAL_B* positions makes Q[0] fall well below the
    # q0_threshold so the head produces no measurable attention there.
    L = S
    mem_val_suppress = (
        AP(0, BD.MEM_VAL_B0, -L),
        AP(0, BD.MEM_VAL_B1, -L),
        AP(0, BD.MEM_VAL_B2, -L),
        AP(0, BD.MEM_VAL_B3, -L),
    )
    spec = replace(spec, q=spec.q + mem_val_suppress)
    # Stack-source top stores update STACK0 from the current AX byte, but the
    # ordinary stack persistence head shares an H1+AX key with a negative store
    # route.  Keep this top-store byte-0 route isolated in head 7 and select
    # the AX byte-0 row with both H1+AX and BYTE_INDEX_0, not H1+AX alone.
    M = 50.0 * S
    TOP_STORE_BIAS = -95.0 * S
    TOP_STORE_ADDR = 10.0 * S
    TOP_STORE_ADDR_BLOCK = -20.0 * TOP_STORE_ADDR
    TOP_STORE_CMP = 5.0 * S
    TOP_STORE_HAS_SE = 5.0 * S

    def top_store_query(
        slot: int, target_dim: int, target_strength: float = M
    ) -> tuple:
        return (
            AP(slot, BD.CONST, TOP_STORE_BIAS),
            AP(slot, BD.MEM_STORE, M),
            AP(slot, BD.MEM_ADDR_SRC, M),
            AP(slot, BD.CMP + 3, TOP_STORE_CMP),
            AP(slot, BD.HAS_SE, TOP_STORE_HAS_SE),
            AP(slot, target_dim, target_strength),
            AP(slot, BD.ADDR_B0_LO + 0, TOP_STORE_ADDR),
            AP(slot, BD.ADDR_B0_HI + 14, TOP_STORE_ADDR),
            AP(slot, BD.ADDR_B0_LO + 8, TOP_STORE_ADDR_BLOCK),
            AP(slot, BD.ADDR_B0_HI + 15, TOP_STORE_ADDR_BLOCK),
            # Marker-proximity blockers PC/AX/SP/BP (Class-1 marker-relative
            # bank slots, frame-INVARIANT). ``BD.H1 + marker_bank_index(...)``
            # resolves to the same 0/1/2/3 integers the literals encoded in both
            # frames; declared marker-relative so the audit no longer reads them
            # as bare offsets. See positional_invariant.py.
            AP(slot, BD.H1 + marker_bank_index("PC"), -3.0 * M),
            AP(slot, BD.H1 + marker_bank_index("AX"), -3.0 * M),
            AP(slot, BD.H1 + marker_bank_index("SP"), -3.0 * M),
            AP(slot, BD.H1 + marker_bank_index("BP"), -3.0 * M),
            # H1+4 (L0 head 1 "MEM marker within dist 4.5") blocker: the
            # top_store_query aux block was designed for STACK0-store contexts;
            # at the MEM-addr0 input position of step 0 the residual carries
            # H1+4=+1.0 (MEM marker nearby), MEM_STORE=+2.0, MEM_ADDR_SRC=+1.0,
            # CONST=+1.0, giving a slot-40 query sum of +5499 (POSITIVE -> fires)
            # which produces a spurious +2.0 add to OUTPUT_LO[0]/OUTPUT_HI[0]
            # via head 7's V/O passthrough. The four sibling H1+0..3 blockers
            # already exist for the PC/AX/SP/BP marker proximity dims; adding
            # H1+4 closes the MEM-marker leak. -3.0*M = -15000 mirrors the
            # existing sibling block magnitude.
            # See docs/VAR_REAL_ATTRIBUTION_2026_06_05.md "Concrete fix
            # hypothesis option 1" (the brief named the dim MARK_MEM; the
            # actual residual leak is via the H1+4 proximity output, which is
            # what the existing -3.0*M block applies to for H1+0..3).
            AP(slot, BD.H1 + marker_bank_index("MEM"), -3.0 * M),
        )

    # Class-2 absolute-slot anchors: the STACK0-byte target discriminators
    # (``BD.STACK0_BYTE0..2``) are the d=6..8-from-BP byte-POSITION flags that
    # VANISH (alias onto MEM addr bytes) when the STACK0 block is dropped
    # (STEP_TOKENS 35->30). This is the stack-source top-store route that
    # updates STACK0 from the current AX byte; under C4_NO_STACK0_EMIT there is
    # no STACK0 to update, so the route is structurally moot — and worse, the
    # STACK0_BYTE0 input flag misfires onto a MEM-addr row. ``invariant_threshold``
    # drives the target-discriminator Q strength to 0 in the dropped frame
    # (live=M at 35-tok -> byte-identical; suppressed=0 at 30-tok -> the
    # misfiring flag's contribution is killed and the route can't spuriously
    # boost head 7 at a MEM-addr row). The MARK_STACK0 slots (40/41) are a
    # marker-TYPE dim, not a byte-position flag — under the campaign the STACK0
    # marker token simply isn't emitted so the flag is naturally 0 with no
    # row-alias misfire, so they need no shift. See positional_invariant.py.
    def stack0_byte_target_strength(byte_k: int) -> float:
        # STACK0 byte N sits at d=(6+N)-from-BP in the full frame.
        return invariant_threshold(
            live=M, suppressed=0.0, marker="BP", k=6 + byte_k,
        )

    return replace(
        spec,
        q=spec.q + (
            *top_store_query(40, BD.MARK_STACK0),
            *top_store_query(41, BD.MARK_STACK0),
            *top_store_query(42, BD.STACK0_BYTE0, stack0_byte_target_strength(0)),
            *top_store_query(43, BD.STACK0_BYTE0, stack0_byte_target_strength(0)),
            *top_store_query(44, BD.STACK0_BYTE1, stack0_byte_target_strength(1)),
            *top_store_query(45, BD.STACK0_BYTE1, stack0_byte_target_strength(1)),
            *top_store_query(46, BD.STACK0_BYTE2, stack0_byte_target_strength(2)),
            *top_store_query(47, BD.STACK0_BYTE2, stack0_byte_target_strength(2)),
        ),
        k=spec.k + (
            AP(40, BD.H1 + AX_IDX, M),
            AP(41, BD.BYTE_INDEX_0, M),
            AP(42, BD.H1 + AX_IDX, M),
            AP(43, BD.BYTE_INDEX_1, M),
            AP(44, BD.H1 + AX_IDX, M),
            AP(45, BD.BYTE_INDEX_2, M),
            AP(46, BD.H1 + AX_IDX, M),
            AP(47, BD.BYTE_INDEX_3, M),
        ),
    )


def _layer10_pc_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Carry PC bytes 1-3 across non-branch steps.

    Mirrors the AX/SP/BP byte_passthrough heads (see
    :func:`_layer10_ax_byte_passthrough_head_spec`,
    :func:`_layer10_sp_byte_passthrough_head_spec`,
    :func:`_layer10_bp_byte_passthrough_head_spec`) but for the PC
    register marker (``H1 + PC_IDX``). At MARK_PC byte rows the head
    attends back to the previous step's MARK_PC byte row at the same
    BYTE_INDEX_k and copies CLEAN_EMBED -> OUTPUT_LO/HI.

    Suppressed on opcodes that rewrite PC so the L6/L7 JSR/JMP/branch
    PC override (byte 0 from the IMM bytes) and the L9 LEV PC restore
    (byte 0 from mem[BP+8]) are not overwritten by the carry:

    - ``OP_JSR`` / ``OP_JMP`` -- jumps to the IMM target; bytes 1-3
      should come from the IMM bytes (L6 override) or default to 0,
      not the prior step's PC.
    - ``OP_BZ`` / ``OP_BNZ`` -- conditional branches; the L9 PC
      override selects the target or fall-through and writes byte 0.
    - ``OP_LEV`` -- restores PC from ``mem[BP+8]`` via the L9 BP-to-
      PC ADDR_B0 relay; bytes 1-3 of the saved PC should not be
      clobbered with the current step's PC.

    Without this head, PC bytes 1-3 at JSR/LEV steps fall back to the
    L3 default (0x00). For the simple-function smoke test
    ``JSR 3; EXIT; NOP; ENT 0; IMM 42; LEV`` the JSR target PC = 3
    and the LEV return PC = 4 both fit in byte 0 — but downstream
    attention heads (notably the AX byte_passthrough and the SI/SC
    store routing) read PC byte rows during JSR for the return-
    address push, and the spurious L3-default zeros leak into AX
    bytes 1-3 (documented contamination shape: ``0xf8030063`` per
    ``docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md`` 2026-05-11). The
    byte_passthrough head supplies a clean PC value at MARK_PC byte
    1-3 rows that downstream heads can attend to without picking up
    SP / return-addr residue.
    """
    PC_IDX = 0
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=_l10_head_idx("layer10_pc_byte_passthrough_bake.head_11"),
        source_marker_dim=BD.H1 + PC_IDX,
        target_marker_dim=BD.H1 + PC_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        # PC-rewriting opcodes suppress the carry so downstream PC
        # override writers (L6/L7/L9) are not stomped at byte 0 and
        # the carry does not propagate stale upper bytes when the
        # target PC has bytes 1-3 != prior step's bytes 1-3.
        suppress_op_dims=[
            BD.OP_JSR,
            BD.OP_JMP,
            BD.OP_BZ,
            BD.OP_BNZ,
            BD.OP_LEV,
        ],
        S=S,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.OP_JSR, -10000.0),
            (BD.OP_JMP, -10000.0),
            (BD.OP_BZ, -10000.0),
            (BD.OP_BNZ, -10000.0),
            (BD.OP_LEV, -10000.0),
        ],
    )
    # Suppress the head at non-PC marker rows. The chain spec's Q[1] /
    # Q[33] gates select for ``H1 + PC_IDX`` (proximity to MARK_PC),
    # which is naturally low at MARK_AX/SP/BP/MEM byte rows. But because
    # the K side's ``H1 + PC_IDX`` band is also low at those non-PC K
    # positions, the row-level score reduces to other gates and the
    # softmax can still distribute attention across irrelevant K rows,
    # producing a uniform-attention contribution to OUTPUT_LO/HI that
    # leaks into the AX/SP/BP byte_passthrough writers at the same
    # dim band. Mirrors the ``mem_val_suppress`` block on
    # :func:`_layer10_bp_byte_passthrough_head_spec`: drive Q[0]
    # strongly negative at other-marker rows so the head's per-row
    # softmax output stays effectively zero at those positions.
    L = S
    non_pc_marker_suppress = (
        AP(0, BD.H1 + 1, -L),  # MARK_AX proximity
        AP(0, BD.H1 + 2, -L),  # MARK_SP proximity
        AP(0, BD.H1 + 3, -L),  # MARK_BP proximity
        AP(0, BD.H1 + 4, -L),  # MARK_MEM proximity
        AP(0, BD.MARK_STACK0, -L),
        AP(0, BD.MEM_VAL_B0, -L),
        AP(0, BD.MEM_VAL_B1, -L),
        AP(0, BD.MEM_VAL_B2, -L),
        AP(0, BD.MEM_VAL_B3, -L),
    )
    spec = replace(spec, q=spec.q + non_pc_marker_suppress)
    return spec


def _bake_layer10_psh_stack0_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 3 PSH STACK0 passthrough spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_psh_stack0_passthrough_head_spec(BD, S),
        HD,
    )


def _layer10_psh_stack0_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    PC_IDX = 0
    AX_IDX = 1
    SP_IDX = 2
    BP_IDX = 3
    MEM_IDX = 4
    L = S
    q = [
        AP(0, BD.IS_BYTE, L),
        AP(1, BD.H4 + BP_IDX, L),
        AP(1, BD.H1 + BP_IDX, -L),
        AP(1, BD.CONST, -L / 2),
        AP(2, BD.BYTE_INDEX_3, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.PSH_AT_SP, L),
        AP(3, BD.CONST, -L / 2),
        AP(4, BD.BYTE_INDEX_0, L),
        AP(5, BD.BYTE_INDEX_1, L),
        AP(6, BD.BYTE_INDEX_2, L),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H4 + BP_IDX, 10000.0),
        AP(33, BD.H1 + BP_IDX, -10000.0),
        AP(33, BD.PSH_AT_SP, 10000.0),
        AP(33, BD.MARK_STACK0, -10000.0),
        # === Register-byte-row hard darkening (slot 7): var_simple / id 262 ===
        # This head is the PSH STACK0-store passthrough: its ONLY legit firing
        # rows are STACK0 *frame* byte rows (STACK0_BYTE_h ~= 0.97, H1
        # register-marker band ALL-ZERO). But its active gate (slot 33) keys on
        # PSH_AT_SP + IS_BYTE without excluding register byte rows, so on a PSH
        # step it ALSO fires (softmax1 wsum -> 1.0) on the SP/BP/AX *register*
        # byte rows (PSH_AT_SP=2, all STACK0_BYTE=0, exactly one H1+{PC/AX/SP/
        # BP/MEM}=1). There it attends diffusely and averages CLEAN_EMBED /
        # (OUTPUT-CLEAN_EMBED) debris into OUTPUT at scale 3.0, CRUSHING the
        # OUTPUT band to ~-569 (probe_var_full_chain.py 262: OUTPUT_LO/HI
        # +0.94 through block 11, -568.79 at block 12 / logical L11 on the
        # step-3 PSH SP byte3 row pred_row=212). With OUTPUT dead the SP-byte3
        # LM logit dies and the BP-marker token (260) wins -> SP truncates ->
        # whole-program desync (the LINK5/6 root, docs/
        # VAR_SIMPLE_12_LINK5_6_DIAGNOSIS_2026_06_11.md).
        #
        # FIX (mirrors the l15 slot-64 / l18 slot-44 hard-marker darkening
        # landed in e457ba31 / 8ad47bf4): a hard subtractive NOT-blocker on a
        # fully-free Q/K slot (7), keyed on the H1 register-marker proximity
        # band (H1+0..H1+4 = PC/AX/SP/BP/MEM). On a register byte row exactly
        # one H1+idx=1 so the slot scores every key position at -2e9 (K[CONST]=1
        # everywhere), driving all real scores below the softmax1 anchor (0) ->
        # the head outputs ZERO (wsum -> 0) and the residual OUTPUT survives. On
        # the legit STACK0 frame byte rows H1+0..4 are ALL ZERO, so the slot
        # contributes nothing and the head is BYTE-IDENTICAL (no register-marker
        # proximity on a stack-frame byte). Pure subtractive -- no net-zero
        # compensation -- and it touches NEITHER CMP, PSH_AT_SP nor MEM_STORE
        # (the CMP path is owned by the if_* agent), only the register-marker
        # band that already (weakly) discriminates this head at slot 1/33.
        AP(7, BD.H1 + PC_IDX, -2000000000.0),
        AP(7, BD.H1 + AX_IDX, -2000000000.0),
        AP(7, BD.H1 + SP_IDX, -2000000000.0),
        AP(7, BD.H1 + BP_IDX, -2000000000.0),
        AP(7, BD.H1 + MEM_IDX, -2000000000.0),
    ]
    # === STACK0 high-byte (byte2/byte3) hard darkening (slot 7): var_simple ===
    # The active gate (slot 33) also fires on the STACK0 byte-1/2/3 QUERY rows
    # of a PSH step, crushing OUTPUT to ~-652. On the var PSH that kills the
    # STACK0[3] = 0x00 emission -> a stray register-marker token (261) wins ->
    # the step ends one token early -> 35-token frame shift -> step-4 PC misread
    # (the var_simple full_trace blocker). Darken the head on the byte-2 and
    # byte-3 query rows (BYTE_INDEX_{2,3} ~= 0.97) so the clean residual 0x00
    # default (OUTPUT +0.94 pre-block-16) survives. Reuses the slot-7 K[CONST]=1
    # complement already present below; on a byte-2/3 query row BYTE_INDEX_h=1
    # -> Q[7] scores -2e9 across all keys -> wsum -> 0 -> residual OUTPUT
    # survives. Byte 0 (STACK0 marker producer, BYTE_INDEX all ~0) and byte 1
    # (test_si_li_16bit_value 0x200 byte-1 round-trip) are NOT darkened.
    if _psh_stack0_highbyte_darken_enabled():
        q += [
            AP(7, BD.BYTE_INDEX_2, -2000000000.0),
            AP(7, BD.BYTE_INDEX_3, -2000000000.0),
        ]
    k = [
        AP(0, BD.IS_BYTE, L),
        AP(1, BD.H1 + AX_IDX, L),
        AP(2, BD.BYTE_INDEX_0, -L),
        AP(2, BD.CONST, L / 2),
        AP(4, BD.BYTE_INDEX_1, L),
        AP(5, BD.BYTE_INDEX_2, L),
        AP(6, BD.BYTE_INDEX_3, L),
        AP(33, BD.CONST, 5.0),
        # K-side complement for the slot-7 register-byte NOT-blocker: CONST is
        # present at EVERY position, so the slot-7 score is Q[7]*1 uniformly
        # across keys -> a register-byte query row scores -2e9 everywhere and
        # darkens; a STACK0-frame byte query row scores 0 (Q[7]=0) -> no effect.
        AP(7, BD.CONST, 1.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, k_idx, 3.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 16 + k_idx, 3.0))
    # ---- LEA-local AX byte 0 differential routing (bug #33) ----
    #
    # The existing slot 0..31 V/O routes CLEAN_EMBED_LO/HI from the attended
    # AX byte 0 source row into STACK0 byte 0's OUTPUT_LO/HI at the PSH step.
    # For most opcodes (IMM, ADD, SHR, ...) this is correct because the
    # source's CLEAN_EMBED nibbles match its byte-0 value -- L8 wrote
    # OUTPUT[k] = CLEAN_EMBED[k] for the byte value at the AX byte 0 row.
    #
    # For LEA-local (BP + signed offset, e.g. var_simple / var_update / if_var)
    # the AX-byte-0 source row's nibbles diverge: CLEAN_EMBED holds the raw
    # immediate operand (e.g. 0x18) while OUTPUT holds the L8-computed
    # effective-address low byte (e.g. 0xE8). The PSH-passthrough head needs
    # the OUTPUT-band value, not the immediate. The L10 post-op rule
    # ``tail_lea_local_ax_marker_byte0_e8`` writes 0xE8 into OUTPUT_LO[8]/
    # OUTPUT_HI[14] at the AX marker row but only after L10 post_ops execute,
    # whereas head 3 fires earlier at phase 10.3 -- so head 3 reads the
    # L8-produced OUTPUT directly. That L8 value is exactly the desired
    # LEA-local byte. 2026-06-01 triage attributes 78 ``step4:STACK0_byte0``
    # corruption rows (var_simple +25, var_update +25, if_var +25, ~3 loop)
    # to this missing producer.
    #
    # Differential sub-pattern (slots 32..63, free per HD=64; main routing
    # uses 0..31 in V/O and slot 33 in Q/K only):
    #   slot 32 + k (k=0..15): V reads (OUTPUT_LO+k - CLEAN_EMBED_LO+k),
    #                          O writes the diff into OUTPUT_LO+k at Q row.
    #   slot 48 + k (k=0..15): V reads (OUTPUT_HI+k - CLEAN_EMBED_HI+k),
    #                          O writes the diff into OUTPUT_HI+k at Q row.
    #
    # Semantic-neutrality on non-LEA paths: at AX byte 0 source rows for
    # IMM / ADD / SHR / etc. the L8 FFN writes OUTPUT_LO/HI[k] = the same
    # one-hot nibble pattern that CLEAN_EMBED_LO/HI[k] carries (the byte
    # value matches the immediate input). The diff is ~0, so the added
    # routing contributes nothing -- existing behavior is preserved
    # (including var_three_* whose STACK0_byte0 came out correct under
    # the CLEAN_EMBED-only routing).
    #
    # On LEA-local AX byte 0 rows the diff is (LEA_computed - immediate);
    # adding it on top of the existing CLEAN_EMBED routing yields the
    # LEA-computed byte at STACK0_byte0 -- the fix.
    #
    # Q/K attention scoring is unchanged: head 3 already attends from PSH
    # STACK0 byte 0 to the most recent AX byte 0 row. For programs with a
    # PSH following a LEA-local (var_*, if_var, loop_*), that AX byte 0 is
    # the LEA's, and the differential V routes the (0xE8-0x18) nibble
    # deltas into the STACK0_byte0 OUTPUT band.
    for k_idx in range(16):
        v.append(AP(32 + k_idx, BD.OUTPUT_LO + k_idx, 1.0))
        v.append(AP(32 + k_idx, BD.CLEAN_EMBED_LO + k_idx, -1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, 32 + k_idx, 3.0))
    for k_idx in range(16):
        v.append(AP(48 + k_idx, BD.OUTPUT_HI + k_idx, 1.0))
        v.append(AP(48 + k_idx, BD.CLEAN_EMBED_HI + k_idx, -1.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 48 + k_idx, 3.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_psh_stack0_passthrough_bake.head_3"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


# === Wave 1 A3: PSH AX byte 1/2/3 -> STACK0_BYTE_VAL_h_LO/HI broadcast ====
#
# The L8 sp_gather audit (docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md) showed
# no head broadcasts AX byte values 1/2/3 to the STACK0 byte rows during PSH.
# As a result, L14 mem_generation's BP-relative reads on byte 1/2/3 of the
# STACK0 frame pull zeros, breaking SI/LI/SC/LC round-trips on the memory
# smoke suite.
#
# This op adds 3 new heads (slots 8/9/10, freed by Wave 1 A2 d39159a1) that
# attend from the Q-side STACK0 byte h row to the K-side AX byte h source
# row during OP_PSH, copying CLEAN_EMBED to the new STACK0_BYTE_VAL_h_LO/HI
# dim family (scaffolded in Wave 1 A1 c31897aa).


def _layer10_psh_ax_broadcast_head_spec(BD, S, byte_h: int) -> DeclarativeAttentionHeadSpec:
    """Broadcast AX byte h CLEAN_EMBED to STACK0_BYTE_VAL_h_LO/HI on PSH.

    Wave 1 Cluster A3 — fills the missing producer identified in the L8
    sp_gather audit. Three sibling heads (h=1,2,3) each:

    * Q fires at: ``MARK_STACK0`` AND ``BYTE_INDEX_h`` AND ``OP_PSH``.
    * K fires at: ``MARK_AX`` AND ``BYTE_INDEX_h`` (the AX byte-h row).
    * V copies ``CLEAN_EMBED_{LO,HI}`` from the K row.
    * O writes the nibble pair to ``STACK0_BYTE_VAL_h_{LO,HI}`` at the Q row.

    K-side complement at slot 33 (softmax1-aware, per the L10
    ``stack0_persistence`` slot-33 fix in commit 9bb21bf4): the multi-
    condition Q gate has matching ``AP(33, MARK_AX, M)`` +
    ``AP(33, BYTE_INDEX_h, M)`` K complements so the gate routes
    positively only at the intended AX byte h row. Without the K-side
    positive, the uniform-negative Q gate softmax-cancels and the head
    has no per-row preference.

    Slot layout (HD=64, V/O slots 0..31 used, slot 0 for main Q/K
    selection, slot 33 for active-step gate):

      slot 0..15  : V[CLEAN_EMBED_LO + k] -> O[STACK0_BYTE_VAL_h_LO + k]
      slot 16..31 : V[CLEAN_EMBED_HI + k] -> O[STACK0_BYTE_VAL_h_HI + k]
      slot 0  Q/K: main row selection
      slot 33 Q/K: softmax1-aware active-step (OP_PSH) gate
    """
    AX_IDX = 1
    L = S
    # A3.9 (2026-06-09): drop slot-33 K-side magnitude. With M=50*S, slot-33
    # K-side carried ~2M=1e4 nats at MARK_AX d=0 rows (where MARK_AX +
    # OP_PSH both fire), and only ~M=5e3 nats at AX byte_h rows (only BI_h
    # fires). Combined with H1+AX_IDX firing at the MARK_AX d=0 row, the
    # broadcast head selected the marker row (K@p=100, post-PSH) where
    # CLEAN_EMBED is zero — V wrote nothing. Set M=0 to let slot 0
    # (MARK_AX + BI_h + IS_BYTE + H1+AX_IDX) plus ALiBi recency bias drive
    # K selection. The A3.7 ALiBi pin (slope=1.0) handles latest-write-wins
    # between successive AX byte_h rows; slot 33 is no longer needed for op
    # discrimination because softmax1 + ALiBi at scale=1 already isolates
    # the most-recent AX BI_h row.
    M = 0.0
    byte_index_dim = getattr(BD, f"BYTE_INDEX_{byte_h}")
    # The STACK0_BYTE_{h} position flag fires only at the byte-h row of a
    # STACK0 frame — MARK_STACK0 itself only fires at the marker (d=0) row,
    # NOT at the byte rows (verified 2026-06-09: row 116 has BYTE_INDEX_1=1,
    # STACK0_BYTE1=1, but MARK_STACK0=0). Per A3 diagnostic § "Recommended
    # fix": use a context-broadcast dim available at the STACK0 Q row.
    # STACK0_BYTE_h is that dim.
    stack0_byte_dim = getattr(BD, f"STACK0_BYTE{byte_h}")
    value_lo_dim = getattr(BD, f"STACK0_BYTE_VAL_{byte_h}_LO")
    value_hi_dim = getattr(BD, f"STACK0_BYTE_VAL_{byte_h}_HI")

    # Q gate: fire at STACK0 byte-h row. The OP_PSH gating lives on the K
    # side per docs/A3_BROADCAST_DIAGNOSTIC_2026_06_07.md — OP_PSH lives on
    # the MARK_AX K row (instruction-fetch row), not the STACK0 Q byte
    # rows. The Q-side anchor uses STACK0_BYTE_h (which DOES fire at the
    # byte-h row of every STACK0 frame) rather than MARK_STACK0 (which
    # only fires at the d=0 marker row).
    q = [
        AP(0, stack0_byte_dim, L),
        AP(0, byte_index_dim, L),
        AP(0, BD.IS_BYTE, L),
        AP(0, BD.CONST, -L * 2.0),
        # Suppress other byte indices: only the matching row should fire.
        *(
            [AP(0, getattr(BD, f"BYTE_INDEX_{j}"), -L)
             for j in (0, 1, 2, 3) if j != byte_h]
        ),
        # Suppress other marker rows so non-STACK0 byte rows don't fire.
        AP(0, BD.MARK_AX, -L),
        AP(0, BD.MARK_SP, -L),
        AP(0, BD.MARK_BP, -L),
        AP(0, BD.MARK_PC, -L),
        AP(0, BD.MARK_MEM, -L),

        # Slot 33: positive-only OP_PSH gate at the K row. Q-side stays
        # POSITIVE at the target STACK0 byte-h row so that K-side suppressors
        # (OP_SI/SC/JSR/ENT at slot 33) generate negative products at the
        # wrong op K rows, not positive ones. Use a positive baseline scaled
        # so STACK0_BYTE_h * BYTE_INDEX_h yields a target value of ~L.
        AP(33, stack0_byte_dim, 1.0),
        AP(33, byte_index_dim, 1.0),
        AP(33, BD.CONST, -1.0),
    ]

    # K gate: fire on AX byte-h source row during OP_PSH. The OP_PSH gating
    # and OP-exclusivity terms live here (not on Q) because the K row IS
    # the OP_PSH-active row (per A3 diagnostic 2026-06-07).
    #
    # Slot-33 K-side magnitude (M_K33) is intentionally small (50) so that
    # the ALiBi positional bias on broadcast heads 8/9/10 (slope=1.0 per
    # A3.7) can dominate row selection. Earlier M=5000 dwarfed ALiBi's
    # ~50-logit positional contribution by 100x.
    M_K33 = 50.0
    k = [
        AP(0, BD.MARK_AX, L),
        AP(0, byte_index_dim, L),
        AP(0, BD.IS_BYTE, L),
        AP(0, BD.H1 + AX_IDX, L),
        # A3.10 (2026-06-09): OP_IMM content discriminator. At AX BI_h K
        # rows, OP_IMM=1 only when the step IS an OP_IMM (relayed by L8
        # ``layer8_op_imm_relay``). At AX BI_h rows of PSH/SI/etc. steps,
        # OP_IMM=0 and CLEAN_EMBED carries opcode bytes (0x0d=PSH), not
        # the register value. Adding +L*OP_IMM at slot 0 boosts IMM-step
        # AX BI_h rows by L^2 = 10000 nats over non-IMM-step AX BI_h
        # rows, dominating ALiBi recency. Among IMM-step AX BI_h rows,
        # ALiBi slope=1.0 then picks the most recent (latest-write-wins).
        AP(0, BD.OP_IMM, L),

        # K-side gate: positive at OP_PSH AX byte-h K row, negative at
        # other ops' K rows. Q[33] is ~+1 at the target Q row; multiplied
        # by K[33] this either reinforces (+OP_PSH) or suppresses (-other
        # ops). At non-AX K rows everything is 0, so no contribution.
        AP(33, BD.MARK_AX, M_K33),
        AP(33, byte_index_dim, M_K33),
        AP(33, BD.OP_PSH, M_K33),
        # Block other ops at slot 33 so the head is OP_PSH-exclusive
        # (moved from Q side; only effective at the K row where the op
        # marker dims fire). Magnitude smaller than OP_PSH positive so a
        # benign K row (no op marker) stays near zero.
        AP(33, BD.OP_SI, -M_K33),
        AP(33, BD.OP_SC, -M_K33),
        AP(33, BD.OP_JSR, -M_K33),
        AP(33, BD.OP_ENT, -M_K33),
    ]

    # V: copy CLEAN_EMBED nibbles. O: write to new STACK0_BYTE_VAL_h dims.
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(value_lo_dim + k_idx, k_idx, 3.0))
        o.append(AO(value_hi_dim + k_idx, 16 + k_idx, 3.0))

    head_idx = _l10_head_idx(f"layer10_psh_ax_broadcast_bake.head_{7 + byte_h}")
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_psh_ax_broadcast_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    for byte_h in (1, 2, 3):
        ir.layer(0).attention.append(
            _layer10_psh_ax_broadcast_head_spec(proxy, 100.0, byte_h),
            name=f"layer10_psh_ax_broadcast_bake.head_{7 + byte_h}",
        )
    return ir


def _bake_layer10_stack0_byte_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 STACK0 byte relay specs."""
    Primitives.generate_attention_head(
        attn,
        _layer10_stack0_byte_relay_head_spec(BD, S),
        HD,
    )
    Primitives.generate_attention_head(
        attn,
        _layer10_nonbitwise_stack0_byte_relay_head_spec(BD, S),
        HD,
    )
    Primitives.generate_attention_head(
        attn,
        _layer10_stack0_persistence_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[6] = 1.0


def _layer10_stack0_byte_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    AX_IDX = 1
    L = S
    q = [
        AP(0, BD.CONST, -3000.0),
        AP(0, BD.IS_BYTE, 1000.0),
        AP(0, BD.H1 + AX_IDX, 1000.0),
        AP(0, BD.TEMP + 3, 1000.0),
        AP(0, BD.BYTE_INDEX_3, -3000.0),
        AP(1, BD.TEMP + 3, 50.0),
        AP(31, BD.BYTE_INDEX_0, 60.0),
        AP(32, BD.BYTE_INDEX_1, 60.0),
        AP(34, BD.BYTE_INDEX_2, 60.0),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.TEMP + 3, 10000.0),
        AP(33, BD.BYTE_INDEX_3, -10000.0),
    ]
    k = [
        AP(0, BD.CONST, 10.0),
        # K-side complement for the slot-0 IS_BYTE gate. The Q-side at
        # slot 0 carries IS_BYTE * 1000 alongside H1+AX/TEMP+3/BYTE_INDEX_3
        # discriminators; without a non-CONST K-side at the same slot the
        # IS_BYTE gate softmax-cancels. A tiny MARK_STACK0 weight breaks
        # the softmax symmetry (per
        # docs/DSL_ATTENTION_GATE_FINDING_2026_06_07.md) without
        # perturbing the substantive head selection driven by slots 1,
        # 31, 32, 34. See docs/Q_SIDE_GATE_AUDIT_2026_06_07.md L10
        # stack0_byte_relay.
        AP(0, BD.MARK_STACK0, 0.1),
        AP(1, BD.MEM_STORE, 100.0),
        AP(1, BD.MARK_MEM, -200.0),
        AP(1, BD.CONST, -50.0),
        AP(31, BD.MEM_VAL_B2, 60.0),
        AP(31, BD.STACK0_BYTE1, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(32, BD.STACK0_BYTE2, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(34, BD.STACK0_BYTE3, 60.0),
        # Tie-breaker: penalize MEM_STORE rows at the byte-select slots so
        # the STACK0_BYTE{1,2,3} relay path beats the MEM_VAL_B{2,3} legacy
        # path when both are present. Without this, ALiBi recency lets a
        # nearby MEM-region row (stale 0xFF stack init or a different
        # MEM-frame entry) win the slot-31/32/34 score by ~0.79 score
        # points and the head copies the wrong CLEAN nibble into ALU,
        # firing BitwiseBytePropagationPostOp with 0xFF at AX bytes 1..3.
        # STACK0_BYTE{1,2,3} marker rows never carry MEM_STORE, so the
        # penalty leaves the desired winner untouched. See
        # tools/probe_or_full_lo_hi.py for the OR-step trace.
        AP(31, BD.MEM_STORE, -5.0),
        AP(32, BD.MEM_STORE, -5.0),
        AP(34, BD.MEM_STORE, -5.0),
        AP(33, BD.CONST, 5.0),
        # K-side complement for the slot-33 IS_BYTE gate (mirrors slot 0).
        AP(33, BD.MARK_STACK0, 0.1),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.ALU_LO + k_idx, 1 + k_idx, 11.0))
        o.append(AO(BD.ALU_HI + k_idx, 17 + k_idx, 11.0))
    v.append(AP(0, BD.CONST, 1.0))
    for k_idx in range(16):
        o.append(AO(BD.ALU_LO + k_idx, 0, -8.0))
        o.append(AO(BD.ALU_HI + k_idx, 0, -8.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_stack0_byte_relay_bake.head_4"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_nonbitwise_stack0_byte_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Relay stored STACK0 bytes for non-bitwise pop ALU byte post-ops.

    Head 4 is shared with the existing bitwise byte propagation path. Its
    scoring is intentionally touchy, so ADD/SUB use this separate head to
    recover higher STACK0 bytes from the stored MEM row without perturbing
    AND/OR/XOR. TEMP[3] is the bitwise relay and suppresses this head.
    """
    AX_IDX = 1
    q = [
        AP(0, BD.CONST, -3000.0),
        AP(0, BD.IS_BYTE, 1000.0),
        AP(0, BD.H1 + AX_IDX, 1000.0),
        AP(0, BD.CMP + 3, 150000.0),
        AP(0, BD.TEMP + 3, -500.0),
        AP(0, BD.BYTE_INDEX_3, -3000.0),
        AP(1, BD.CMP + 3, 1000.0),
        # Do not put a negative TEMP[3] term in this match slot: the MEM
        # marker key uses negative MARK_MEM/CONST terms there, so weak
        # TEMP[3] residue turns into positive evidence for the wrong row.
        # The high-magnitude TEMP[3] blockers in slots 0 and 33 still suppress
        # true bitwise rows without polluting the stack-byte selection score.
        AP(31, BD.BYTE_INDEX_0, 60.0),
        AP(32, BD.BYTE_INDEX_1, 60.0),
        AP(34, BD.BYTE_INDEX_2, 60.0),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.CMP + 3, 1500000.0),
        AP(33, BD.TEMP + 3, -5000.0),
        AP(33, BD.BYTE_INDEX_3, -10000.0),
    ]
    k = [
        AP(0, BD.CONST, 10.0),
        # K-side complement for the slot-0 IS_BYTE gate (mirrors head 4).
        AP(0, BD.MARK_STACK0, 0.1),
        AP(1, BD.MEM_STORE, 100.0),
        AP(1, BD.MARK_MEM, -200.0),
        AP(1, BD.CONST, -50.0),
        AP(31, BD.MEM_VAL_B2, 60.0),
        AP(31, BD.STACK0_BYTE1, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(32, BD.STACK0_BYTE2, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(34, BD.STACK0_BYTE3, 60.0),
        AP(33, BD.CONST, 5.0),
        # K-side complement for the slot-33 IS_BYTE gate (mirrors head 4).
        AP(33, BD.MARK_STACK0, 0.1),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.ALU_LO + k_idx, 1 + k_idx, 6.0))
        o.append(AO(BD.ALU_HI + k_idx, 17 + k_idx, 6.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_stack0_byte_relay_bake.head_5"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_stack0_persistence_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Carry and update STACK0 bytes.

    L3 carries STACK0 byte 0 at the marker. This head handles bytes 1-3 by
    querying at the preceding byte position and reading the latest previous
    STACK0 byte through ALiBi preference.

    STORE rows are mutating: after SI/SC, the popped SP points at the just
    written local, so STACK0 must come from the current AX bytes. The store
    subroute below uses extra source-match slots so it dominates the ordinary
    persistence source and reads same-step AX byte 0..3 for STACK0
    marker/byte0/byte1/byte2 respectively.
    """
    AX_IDX = 1
    # The persistence route competes with inactive store-source slots. Those
    # store slots carry negative bias in Q but still see byte-index keys on
    # ordinary STACK0 byte rows; keep the direct STACK0-byte match dominant so
    # byte K reliably predicts byte K+1 across non-mutating steps.
    M = 50.0 * S
    STORE_TARGET = 50.0 * S
    STORE_GATE = 50.0 * S
    STORE_CMP = 5.0 * S
    STORE_HAS_SE = 5.0 * S
    STORE_BIAS = -85.0 * S
    q = [
        AP(0, BD.PSH_AT_SP, -300.0),
        AP(0, BD.OP_PSH, -300.0),
        AP(0, BD.CMP + 0, -300.0),
        AP(0, BD.CMP + 1, -300.0),
        AP(0, BD.CMP + 2, -300.0),
        AP(0, BD.CMP + 4, -300.0),
        AP(0, BD.OP_LEV, -300.0),
        AP(4, BD.STACK0_BYTE0, M),
        AP(4, BD.CMP + 3, -M),
        AP(5, BD.STACK0_BYTE1, M),
        AP(5, BD.CMP + 3, -M),
        AP(6, BD.STACK0_BYTE2, M),
        AP(6, BD.CMP + 3, -M),
        AP(7, BD.CONST, STORE_BIAS),
        AP(7, BD.MEM_STORE, STORE_GATE),
        AP(7, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(7, BD.CMP + 3, STORE_CMP),
        AP(7, BD.HAS_SE, STORE_HAS_SE),
        AP(7, BD.MARK_STACK0, STORE_TARGET),
        AP(8, BD.CONST, STORE_BIAS),
        AP(8, BD.MEM_STORE, STORE_GATE),
        AP(8, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(8, BD.CMP + 3, STORE_CMP),
        AP(8, BD.HAS_SE, STORE_HAS_SE),
        AP(8, BD.STACK0_BYTE0, STORE_TARGET),
        AP(9, BD.CONST, STORE_BIAS),
        AP(9, BD.MEM_STORE, STORE_GATE),
        AP(9, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(9, BD.CMP + 3, STORE_CMP),
        AP(9, BD.HAS_SE, STORE_HAS_SE),
        AP(9, BD.STACK0_BYTE1, STORE_TARGET),
        AP(10, BD.CONST, STORE_BIAS),
        AP(10, BD.MEM_STORE, STORE_GATE),
        AP(10, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(10, BD.CMP + 3, STORE_CMP),
        AP(10, BD.HAS_SE, STORE_HAS_SE),
        AP(10, BD.STACK0_BYTE2, STORE_TARGET),
        AP(11, BD.CONST, STORE_BIAS),
        AP(11, BD.MEM_STORE, STORE_GATE),
        AP(11, BD.MEM_ADDR_SRC, -STORE_GATE),
        AP(11, BD.CMP + 3, STORE_CMP),
        AP(11, BD.HAS_SE, STORE_HAS_SE),
        AP(11, BD.MARK_STACK0, STORE_TARGET),
        AP(11, BD.STACK0_BYTE0, STORE_TARGET),
        AP(11, BD.STACK0_BYTE1, STORE_TARGET),
        AP(11, BD.STACK0_BYTE2, STORE_TARGET),
        AP(33, BD.CONST, -15000.0),
        AP(33, BD.HAS_SE, 10000.0),
        AP(33, BD.PSH_AT_SP, -30000.0),
        AP(33, BD.OP_PSH, -30000.0),
        AP(33, BD.CMP + 0, -30000.0),
        AP(33, BD.CMP + 1, -30000.0),
        AP(33, BD.CMP + 2, -30000.0),
        AP(33, BD.CMP + 4, -30000.0),
        AP(33, BD.OP_LEV, -30000.0),
        AP(33, BD.STACK0_BYTE0, 10000.0),
        AP(33, BD.STACK0_BYTE1, 10000.0),
        AP(33, BD.STACK0_BYTE2, 10000.0),
        AP(33, BD.STACK0_BYTE3, -30000.0),
    ]
    k = [
        # K-side complement for the slot-0 OP_PSH / OP_LEV blockers. The
        # Q-side at slot 0 carries all-negative blockers
        # (PSH_AT_SP/OP_PSH/CMP+0/1/2/4/OP_LEV) without any positive
        # writes; with no K-side entry at slot 0 the blockers are dead
        # weight (q_only). MARK_STACK0 routes the blockers' negative
        # contribution to MARK_STACK0 K rows — i.e. blocks persistence
        # from landing on STACK0 K rows during those opcodes. See
        # docs/Q_SIDE_GATE_AUDIT_2026_06_07.md L10 stack0_persistence.
        AP(0, BD.MARK_STACK0, M),
        AP(4, BD.STACK0_BYTE1, M),
        AP(5, BD.STACK0_BYTE2, M),
        AP(6, BD.STACK0_BYTE3, M),
        AP(7, BD.BYTE_INDEX_0, M),
        AP(8, BD.BYTE_INDEX_1, M),
        AP(9, BD.BYTE_INDEX_2, M),
        AP(10, BD.BYTE_INDEX_3, M),
        AP(11, BD.H1 + AX_IDX, M),
        # K-side complement for the slot-33 "active step" gate. The Q-side
        # at slot 33 (above) carries a multi-condition active-step gate
        # (HAS_SE require + OP_PSH/CMP+0/1/2/4/OP_LEV blockers +
        # STACK0_BYTE0/1/2 positives). Without a non-CONST K-side at the
        # same slot, the Q-side gate is uniform across K rows and softmax
        # cancels, so the "blockers don't blocker" -- persistence can land
        # on non-STACK0 K positions and overwrite values (documented
        # `var_*` regression cluster). The complement here routes the gate
        # positively only at MARK_STACK0 K positions (where persistence
        # should land), and leaves a smaller CONST baseline so the existing
        # byte-relay slot competition stays bounded. See
        # docs/Q_SIDE_GATE_AUDIT_2026_06_07.md "risk #2".
        AP(33, BD.MARK_STACK0, M),
        AP(33, BD.CONST, 100.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, k_idx, 3.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 16 + k_idx, 3.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=_l10_head_idx("layer10_stack0_byte_relay_bake.head_6"),
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_carry_relay_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_carry_relay_head_spec(proxy, 100.0),
        name="layer10_carry_relay_bake.head_0",
    )
    return ir


def _r_frame_passthrough_ir(dim_positions, *, spec_fn, ir_name) -> CompilerIR:
    """Shared R-FRAME register byte-passthrough ``compiler_ir_factory``.

    INCR-1 collapse: the AX/SP/BP/PC ``_layer10_*_byte_passthrough_ir``
    factories were byte-for-byte identical apart from the head spec function
    and the IR head name (both per-register data). One shared factory appends
    the identical head IR, so census/IR output is UNCHANGED.
    """
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(spec_fn(proxy, 100.0), name=ir_name)
    return ir


def _layer10_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    return _r_frame_passthrough_ir(
        dim_positions,
        spec_fn=_layer10_ax_byte_passthrough_head_spec,
        ir_name="layer10_byte_passthrough_bake.head_1",
    )


def _layer10_sp_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    return _r_frame_passthrough_ir(
        dim_positions,
        spec_fn=_layer10_sp_byte_passthrough_head_spec,
        ir_name="layer10_sp_byte_passthrough_bake.head_2",
    )


def _layer10_bp_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    return _r_frame_passthrough_ir(
        dim_positions,
        spec_fn=_layer10_bp_byte_passthrough_head_spec,
        ir_name="layer10_bp_byte_passthrough_bake.head_7",
    )


def _layer10_pc_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    return _r_frame_passthrough_ir(
        dim_positions,
        spec_fn=_layer10_pc_byte_passthrough_head_spec,
        ir_name="layer10_pc_byte_passthrough_bake.head_11",
    )


def _layer10_psh_stack0_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_psh_stack0_passthrough_head_spec(proxy, 100.0),
        name="layer10_psh_stack0_passthrough_bake.head_3",
    )
    return ir


def _layer10_stack0_byte_relay_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_stack0_byte_relay_head_spec(proxy, 100.0),
        name="layer10_stack0_byte_relay_bake.head_4",
    )
    ir.layer(0).attention.append(
        _layer10_nonbitwise_stack0_byte_relay_head_spec(proxy, 100.0),
        name="layer10_stack0_byte_relay_bake.head_5",
    )
    ir.layer(0).attention.append(
        _layer10_stack0_persistence_head_spec(proxy, 100.0),
        name="layer10_stack0_byte_relay_bake.head_6",
    )
    return ir


def make_layer10_carry_relay_op() -> Operation:
    """Topology anchor for L10 head 0 carry relay.

    The actual weight bake is owned by ``layer10_carry_relay_bake`` below,
    pinned to ``model.blocks[10].attn``.

    Phase 3 (mem cluster fix) anchor split note: this anchor historically
    served BOTH the L10 attn-bake family (head_0/1/2/3/7 + stack0 relays)
    AND the L10 FFN family (``layer10_alu``, ``l10_post_op_attach``, divmod
    stages, the efficient andorxor wrap, etc.). The two families therefore
    could not migrate to different physical layers independently — pinning
    this anchor moved the 1846-unit ``layer10_alu`` FFN alongside, blowing
    the L10 FFN budget. Phase 3 split the attn family off to a sibling
    anchor ``_layer10_attn_anchor`` (added below). The 6 attn-bake ops
    (``layer10_carry_relay_bake``, ``layer10_byte_passthrough_bake``,
    ``layer10_sp_byte_passthrough_bake``,
    ``layer10_psh_stack0_passthrough_bake``,
    ``layer10_stack0_byte_relay_bake``,
    ``layer10_bp_byte_passthrough_bake``) now target the new attn anchor;
    this ``layer10_carry_relay`` anchor retains the FFN-side scheduling
    role (``layer10_alu`` + downstream post_op / wrap consumers still bind
    to it via ``target_op_name``). See
    ``docs/MEMORY_PHASE2_BLOCKER_2026_06_05.md`` for the why.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_carry_relay",
        # Phase 9.B (CARRY SCC rename): CARRY -> CARRY.*.-1 marks the read
        # as SSA cross-step relative to the same-step L10 CARRY writers
        # (``layer10_carry_relay_bake``, ``l10_post_ops_combined``).
        # Semantically the relay forwards the *previous* step's CARRY into
        # the current-step CARRY slot for the L10 AX-byte ADD/SUB sum.
        # ``CARRY.*.-1`` aliases the numeric ``CARRY`` slot via the SSA
        # rewriter so the lowered weights remain byte-identical; the
        # same-step writers' edges into this op are suppressed, breaking
        # the 3-op same-step structural sub-cycle on CARRY
        # (l10_post_ops_combined <-CARRY-> layer10_carry_relay{,_bake}),
        # per ``.agent-logs/scc_zero_audit.md §3.7`` option (b).
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY.*.-1"},
        writes={"CARRY"},  # broadcast
        kind="attn",
        # Phase 3 (mem cluster fix, 2026-06-05): explicit ``phase=10.0``
        # so the sibling ``_layer10_attn_anchor`` (added in Phase 3) can
        # share this same (layer, kind="attn") slot via the layer
        # assignment's "same phase, share" branch
        # (``_assign_layers``, layer_compiler.py:~2132). Without the
        # shared phase the slot tracker would push the second attn
        # anchor to the next layer and break byte-identity. The numeric
        # value is arbitrary — any non-None value the two anchors agree
        # on works; 10.0 matches the historical "L10 phase=10 baseline"
        # in the docstrings of ``layer10_alu`` &c.
        phase=10.0,
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        # Phase 8.A.4 retry: this op is the L10 layer anchor. Pointing at
        # ``layer9_marker_suppress`` (kind="ffn", pinned to L9 via its own
        # ``requires["after"]: layer8_alu``) creates a topo dep edge that
        # both orders the placement (anchor placed after suppress) and
        # forces ``earliest = L9 + 1 = L10``. L10 block ops then bind to
        # this anchor's resolved layer via ``target_op_name``.
        requires={"after": "layer9_marker_suppress"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_attn_anchor_op() -> Operation:
    """L10 attn-side layer anchor (Phase 3 mem cluster fix).

    Sibling of ``layer10_carry_relay`` (above). The historical
    ``layer10_carry_relay`` anchor co-served the L10 attn family AND the
    L10 FFN family because every L10 bake (attn or FFN) used the same
    ``target_op_name="layer10_carry_relay"``. The
    ``MEMORY_PHASE2_BLOCKER_2026_06_05.md`` analysis identified this
    coupling as the prerequisite blocker for Phase 2 (decoupling
    ``layer10_alu`` from the attn family so the attn cluster could move
    layers without dragging the 1846-unit FFN alongside).

    This anchor's only role is to give the 6 L10 attn-bake ops a stable
    ``target_op_name`` that resolves to L10 (via the same
    ``requires["after"]: layer9_marker_suppress`` chain as
    ``layer10_carry_relay``) but is independent of the FFN family. As a
    ``declarative_authority="topology_anchor"`` op it writes no weights;
    the slot registry derives no slot claims for it (per
    ``slot_registry.derive_slot_ids_for_op`` — topology anchors return
    early). Two attn anchors at the same layer are therefore admissible.

    Consumers (target_op_name="_layer10_attn_anchor"):
      - ``layer10_carry_relay_bake`` (attn head 0)
      - ``layer10_byte_passthrough_bake`` (attn head 1)
      - ``layer10_sp_byte_passthrough_bake`` (attn head 2)
      - ``layer10_psh_stack0_passthrough_bake`` (attn head 3)
      - ``layer10_bp_byte_passthrough_bake`` (attn head 7)
      - ``layer10_stack0_byte_relay_bake`` (attn heads 4/5/6)

    Consumers NOT moved (remain on ``layer10_carry_relay``):
      - ``layer10_alu`` (FFN bake, 1846 units)
      - ``l10_post_op_attach`` (block.post_ops attach)
      - ``l10_alu_postop_attach`` (block.post_ops attach)
      - ``l10_alu_divmod_{bdtoge,longdiv,getobd,install}`` (DivMod stages)
      - ``efficient_l10_andorxor_wrap`` (block.ffn replacement)
      - ``null_terminator_detection`` (L10 FFN convo-IO unit)
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="_layer10_attn_anchor",
        # Mirrors ``layer10_carry_relay``'s read/write semantics — same
        # MARK_AX gate, same CARRY-relay SSA cross-step rename — so the
        # dep graph treats the two anchors symmetrically.
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY.*.-1"},
        writes={"CARRY"},
        kind="attn",
        # Phase 3 (mem cluster fix, 2026-06-05): shared explicit phase
        # with ``layer10_carry_relay`` (the sibling FFN-side anchor) so
        # both anchors co-resolve to the same physical layer via the
        # layer assignment's "same phase, share" branch
        # (``_assign_layers``, layer_compiler.py:~2132). Without this the
        # slot tracker would bump the second attn anchor to the next
        # layer — the 6 attn-bake ops would then land at a different
        # block than ``layer10_alu``'s FFN bake, inverting the L10 attn-
        # before-ffn execution order. The Phase 4 retry of Phase 2 will
        # drop this constraint when relocating the attn family to a
        # non-L10 attn-free layer.
        phase=10.0,
        migrated=True,
        declarative_authority="topology_anchor",
        compiler_ir=CompilerIR(),
        # Phase 3c (mem cluster fix, 2026-06-06): pin the L10 attn family
        # to ``layer_idx=10`` so the 6 attn-bake ops (carry_relay,
        # byte_passthrough, sp_byte_passthrough, psh_stack0_passthrough,
        # stack0_byte_relay, bp_byte_passthrough) move out of
        # ``block[13].attn`` and into ``block[10].attn``. This frees
        # heads 0/1/2 at block[13].attn for Phase 4's
        # ``layer13_mem_addr_gather`` pin to L13 (the head_0 contest
        # documented in MEMORY_PHASE4_BLOCKER_2026_06_05.md and
        # MEMORY_PHASE3B_COMPLETE_2026_06_05.md "Why not pin to layer_idx
        # =13 in this commit"). ``layer_idx=10`` overrides the
        # ``phase=10.0`` / ``same_layer_as`` constraint that previously
        # co-located this anchor with ``layer10_carry_relay`` at L13.
        # The L10 FFN family (``layer10_alu`` + 1846 units, post-op
        # attach, divmod stages, andorxor wrap, null_terminator_detection)
        # is NOT moved — it still binds to ``layer10_carry_relay`` (the
        # FFN-side sibling), which remains at L13 via its
        # ``requires["after"]: layer9_marker_suppress`` chain. This
        # decoupling is exactly what Phase 3 split enabled.
        layer_idx=11,
        # ``after: layer9_marker_suppress`` kept defensively so the dep
        # graph still orders this anchor after L9 setup; ``same_layer_as``
        # dropped because Phase 3c intentionally separates the two L10
        # anchors onto different physical layers.
        requires={"after": "layer9_marker_suppress"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_byte_passthrough_op() -> Operation:
    """Topology anchor for L10 head 1 AX byte passthrough.

    The spec-generated weight bake is owned by
    ``layer10_byte_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_byte_passthrough",
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers (which fire after L10 in the
        # same step). The same-step values from L3/L5/L7 still resolve at
        # the same numeric position (TEMP_PREV_STEP aliases TEMP). Breaks
        # L11/L14 → layer10_byte_passthrough back-edges on TEMP.
        # Declaration audit (2026-06-05): added MARK_AX, H3, H4, MEM_VAL_B0,
        # MEM_ADDR_SRC, OP_SI, OP_SC, OP_LC_RELAY, CMP, CONST to mirror the
        # full Q/K read set of _layer10_ax_byte_passthrough_head_spec (the
        # head spec lowered by the paired ``layer10_byte_passthrough_bake``
        # block op). Anchor reads gate dim-lifetime analysis; bake op's
        # block kind filters it out of the same analysis.
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "OP_LI_RELAY", "OP_LC_RELAY",
               "OP_SI", "OP_SC", "TEMP.*.-1", "CMP", "CONST",
               "H1", "H3", "H4", "MARK_AX",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "MEM_STORE", "MEM_ADDR_SRC",
               "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_sp_byte_passthrough_op() -> Operation:
    """Topology anchor for L10 head 2 SP byte passthrough.

    The spec-generated weight bake is owned by
    ``layer10_sp_byte_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_sp_byte_passthrough",
        # Declaration audit (2026-06-05): added MARK_SP, CONST, PSH_AT_SP,
        # OP_ENT, OP_JSR, CMP, BYTE_INDEX_3 to mirror the Q/K read set of
        # _layer10_sp_byte_passthrough_head_spec lowered by the paired
        # ``layer10_sp_byte_passthrough_bake`` block op (the anchor's
        # reads gate dim-lifetime analysis; the bake's block kind filters
        # it out of the same analysis).
        reads={"IS_BYTE", "HAS_SE", "H1", "MARK_SP", "CONST", "PSH_AT_SP",
               "OP_ENT", "OP_JSR", "CMP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        # Phase 7.A.2 backfill: this is the topology anchor for L10 head 2's
        # SP byte-0 marker carry-forward. The actual spec
        # (``_layer10_sp_byte_passthrough_head_spec``) reads dims at Q
        # positions 34/35 that are NOT in this anchor's ``reads`` set:
        #   - PSH_AT_SP, OP_JSR  -- written by layer7_memory_heads
        #   - CMP+2/+3/+4         -- written by layer6_routing_ffn
        # These are the *gates* that decide whether to suppress the SP byte
        # carry-forward when the current op is rewriting SP (PSH/JSR/ENT/POP/
        # LEV/ADJ). The byte payload itself (CLEAN_EMBED) is residue from
        # the embedding, so the only same-step preds we have are the gating
        # writers; declare them explicitly so the scheduler analyzer knows
        # this op cannot float earlier than L7.
        requires={"after": [
            "layer6_routing_ffn",
            "layer7_memory_heads",
        ]},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_psh_stack0_passthrough_op() -> Operation:
    """Topology anchor for L10 head 3 PSH STACK0 passthrough.

    The actual weight bake is owned by
    ``layer10_psh_stack0_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_psh_stack0_passthrough",
        # Declaration audit (2026-06-05): added CONST, IS_BYTE, H1, H4,
        # PSH_AT_SP, BYTE_INDEX_*, CLEAN_EMBED_LO/HI to mirror the Q/K/V
        # read set of _layer10_psh_stack0_passthrough_head_spec lowered by
        # the paired ``layer10_psh_stack0_passthrough_bake`` block op.
        # The LEA-local differential routing reads OUTPUT_LO/HI; those are
        # tracked on the bake op via OUTPUT_LO.*.-1 / OUTPUT_HI.*.-1 (NOT
        # mirrored here -- adding them to a kind="attn" anchor trips the
        # cross-step baseline allowlist gate; the bake op carries the
        # cross-step semantics for those bands instead).
        reads={"MARK_STACK0", "OP_PSH", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_LI", "OP_LC", "OP_SI", "OP_SC",
               "CONST", "IS_BYTE", "H1", "H4", "PSH_AT_SP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_psh_ax_broadcast_op() -> Operation:
    """Topology anchor for the Wave 1 A3 broadcast heads (slots 8/9/10).

    The actual weight bake is owned by ``layer10_psh_ax_broadcast_bake``
    below; this op is the dep-graph anchor that downstream consumers
    (L14 mem_generation migration) target via ``STACK0_BYTE_VAL_h_LO/HI``
    reads.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_psh_ax_broadcast",
        reads={"MARK_STACK0", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_PC",
               "MARK_MEM", "IS_BYTE", "OP_PSH", "OP_SI", "OP_SC", "OP_JSR",
               "OP_ENT", "H1", "CONST",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
                "STACK0_BYTE_VAL_2_LO", "STACK0_BYTE_VAL_2_HI",
                "STACK0_BYTE_VAL_3_LO", "STACK0_BYTE_VAL_3_HI"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        compiler_ir=CompilerIR(),
        smoke_tests={"TestSmokeMemory::test_si_li_roundtrip"},
        spec_section="BLOG_SPEC.md#registers",
    )


# -- L10 attention bake ops (migrated 2026-05-10) -----------------------------
#
# These five ``kind="block", layer_idx=10, migrated=True`` ops bake the five
# inline ``_set_layer10_*`` attention calls that used to live in
# ``set_vm_weights`` (both the ``alu_mode == 'lookup'`` and
# ``alu_mode == 'efficient'`` branches). The inline calls have been removed
# from both branches; these ops now own the bake. Phases 10.0-10.4 preserve
# the original ordering. The five ``layer10_*`` kind="attn" placeholders
# above are retained as migrated no-op dep-graph anchors so the LayerCompiler
# topology does not shift downstream block assignments.
#
# All five target ``model.blocks[10].attn`` and run BEFORE legacy_bake (999),
# so the alibi_slopes mutations and the L10 FFN bake inside set_vm_weights
# still execute in their original order. The attn weight slots they write
# are NOT touched by legacy_bake after the inline removals.


def make_layer10_carry_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_carry_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both lookup and efficient
    branches): ``_set_layer10_carry_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.0 preserves the original
    ordering relative to the four sibling L10 attn bake ops below.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. The head_idx=0
    literal is replaced with a pinned-allocator lookup
    (``_l10_head_idx("layer10_carry_relay_bake.head_0")``); the bake_fn
    stashes a per-bake :class:`AttentionHeadAllocator` on ``attn`` so the
    L10 head axis is auditable. The Q/K/V/O write authority remains in
    :func:`_layer10_carry_relay_head_spec` (data, not imperative code);
    the bake_fn keeps the residual ``Primitives.generate_attention_head``
    call until Wave 6B shrinks it to ``_lower_via_compiler_ir``.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # Stashed on ``attn`` for downstream inspection / extension; the
        # actual ``head_idx`` value used by the spec comes from
        # :func:`_l10_head_idx` so the spec stays in lockstep with the
        # layout table without re-querying the allocator here.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the head spec directly into ``attn``
        # (was ``_bake_layer10_carry_relay_head``) so census v2 classifies
        # this op as ``declarative`` (no helper hop).
        Primitives.generate_attention_head(
            attn, _layer10_carry_relay_head_spec(proxy, S), HD,
        )

    # Dim-ownership claims: L10 attn head 0 CARRY relay (AX marker → AX bytes).
    #   W_v[0*HD + 1, CARRY + 1]  (CARRY[1] = ADD byte carry)
    #   W_v[0*HD + 2, CARRY + 2]  (CARRY[2] = SUB byte borrow)
    #   W_o[CARRY + 1, 0*HD + 1]
    #   W_o[CARRY + 2, 0*HD + 2]
    _claims = {
        (10, "attn_W_v", "0_1", "CARRY+1"),
        (10, "attn_W_v", "0_2", "CARRY+2"),
    }

    return Operation(
        name="layer10_carry_relay_bake",
        # Phase 9.B (SCC #2 dissolution): CARRY -> CARRY.*.-1 marks the
        # read as SSA cross-step relative to the same-step L10 CARRY
        # writer ``l10_post_ops_combined`` (phase=10.5). This bake op
        # runs at phase=10.0, so any CARRY value it reads must originate
        # from the PREVIOUS step (attention-broadcast residue) — the
        # later L10 post-op writer cannot influence the current-step
        # input. ``CARRY.*.-1`` aliases the numeric ``CARRY`` slot via
        # the SSA rewriter so the lowered weights remain byte-identical;
        # the same-step writer's edge into this op is suppressed,
        # breaking the CARRY back-edge ``l10_post_ops_combined ->
        # layer10_carry_relay_bake``. Direct parallel to the sub-cycle C
        # fix on ``layer10_carry_relay`` in commit 0fa605e8; paired with
        # the OUTPUT_LO.*.-1 rename in ``l10_post_ops_combined`` to
        # break the OUTPUT_LO back-edge from
        # ``tail_bit32_result_correction``.
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY.*.-1", "CONST"},
        writes={"CARRY"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_carry_relay_ir,
        # Phase 3 (mem cluster fix, 2026-06-05): retargeted from
        # ``layer10_carry_relay`` to the new sibling
        # ``_layer10_attn_anchor`` so the L10 attn family can migrate
        # layers independently of the L10 FFN family (``layer10_alu`` &c
        # still bind to ``layer10_carry_relay``). Both anchors resolve to
        # L10 today via the same ``requires["after"]:
        # layer9_marker_suppress`` chain — this change is metadata-only.
        # See ``docs/MEMORY_PHASE2_BLOCKER_2026_06_05.md``.
        target_op_name="_layer10_attn_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmokeAddress::test_lea_basic",
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def _make_r_frame_passthrough_bake_op(
    *,
    op_name: str,
    head_slot: int,
    spec_fn,
    ir_factory,
    reads: set,
    smoke_tests: set,
    requires: dict | None = None,
) -> Operation:
    """Shared R-FRAME register byte-passthrough ``Operation`` builder.

    INCR-1 collapse (register-emission-frame). The AX/SP/BP/PC
    ``make_layer10_*_byte_passthrough_bake_op`` functions were structurally
    identical: each pinned the L10 head allocator, lowered its per-register
    head spec via ``generate_attention_head``, wrote ``alibi_slopes[slot]=1.0``,
    declared the SAME OUTPUT_LO/HI ``CLEAN_EMBED`` V-claims (only the head slot
    varied), and returned an ``Operation`` with the identical
    ``target_op_name/migrated/declarative_authority/writes/spec_section``.
    The ONLY per-register data is the 5-tuple ``(op_name, head_slot, spec_fn,
    reads, smoke_tests)`` (+ PC's ``requires``). This builder is that data
    table's single lowering; each ``make_*`` below is a one-line call, so the
    emitted Operation — and therefore the golden hash — is UNCHANGED.

    The V/O nibble copy (``range(16)`` CLEAN_EMBED_LO/HI -> OUTPUT_LO/HI) is
    already shared inside ``_byte_passthrough_chain_spec``; this collapses the
    surrounding per-register bake/claim/Operation GLUE.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # See ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(attn, spec_fn(proxy, S), HD)
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[head_slot] = 1.0

    # Dim-ownership claims: the byte_passthrough chain writes V slots 0..31 +
    # O writes OUTPUT_LO/HI at this head slot:
    #   W_v[slot*HD + k, CLEAN_EMBED_LO + k]       for k=0..15
    #   W_v[slot*HD + 16 + k, CLEAN_EMBED_HI + k]  for k=0..15
    #   W_o[OUTPUT_LO + k, slot*HD + k]            for k=0..15
    #   W_o[OUTPUT_HI + k, slot*HD + 16 + k]       for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"{head_slot}_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add(
            (10, "attn_W_v", f"{head_slot}_{16 + k}", f"CLEAN_EMBED_HI+{k}")
        )

    kwargs = {}
    if requires is not None:
        kwargs["requires"] = requires
    return Operation(
        name=op_name,
        reads=reads,
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=ir_factory,
        # Phase 3 (mem cluster fix, 2026-06-05): retargeted from
        # ``layer10_carry_relay`` to ``_layer10_attn_anchor`` (the new
        # attn-only sibling anchor) so the L10 attn family can migrate layers
        # independently of ``layer10_alu`` (1846 FFN units). Metadata-only at
        # L10 today.
        target_op_name="_layer10_attn_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests=smoke_tests,
        spec_section="BLOG_SPEC.md#registers",
        **kwargs,
    )


def make_layer10_byte_passthrough_bake_op() -> Operation:
    """Bake the AX byte-passthrough head into ``model.blocks[10].attn`` (slot 1).

    INCR-1: thin call over ``_make_r_frame_passthrough_bake_op``. The AX row's
    per-register data is its head spec (``_layer10_ax_byte_passthrough_head_spec``
    — the LI-reload query blocks) + the full Q/K read set audited 2026-06-05.
    """
    return _make_r_frame_passthrough_bake_op(
        op_name="layer10_byte_passthrough_bake",
        head_slot=1,
        spec_fn=_layer10_ax_byte_passthrough_head_spec,
        ir_factory=_layer10_byte_passthrough_ir,
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "OP_LI_RELAY", "OP_LC_RELAY",
               "OP_SI", "OP_SC", "TEMP.*.-1", "CMP", "CONST",
               "H1", "H3", "H4", "MARK_AX",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "MEM_STORE", "MEM_ADDR_SRC",
               "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        smoke_tests={"all"},
    )


def make_layer10_sp_byte_passthrough_bake_op() -> Operation:
    """Bake the SP byte-passthrough head into ``model.blocks[10].attn`` (slot 2).

    INCR-1: thin call over ``_make_r_frame_passthrough_bake_op``.
    """
    return _make_r_frame_passthrough_bake_op(
        op_name="layer10_sp_byte_passthrough_bake",
        head_slot=2,
        spec_fn=_layer10_sp_byte_passthrough_head_spec,
        ir_factory=_layer10_sp_byte_passthrough_ir,
        reads={"IS_BYTE", "HAS_SE", "H1", "MARK_SP", "CONST", "PSH_AT_SP", "CMP",
               "OP_ENT", "OP_JSR",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        smoke_tests={"all"},
    )


def make_layer10_bp_byte_passthrough_bake_op() -> Operation:
    """Bake the BP byte-passthrough head into ``model.blocks[10].attn`` (slot 7).

    BP byte 0 is carried at the marker by L3. This head carries bytes 1-3
    across ordinary non-ENT/LEV steps so BP remains valid inside functions.
    INCR-1: thin call over ``_make_r_frame_passthrough_bake_op``.
    """
    return _make_r_frame_passthrough_bake_op(
        op_name="layer10_bp_byte_passthrough_bake",
        head_slot=7,
        spec_fn=_layer10_bp_byte_passthrough_head_spec,
        ir_factory=_layer10_bp_byte_passthrough_ir,
        reads={"IS_BYTE", "HAS_SE", "H1", "OP_ENT", "OP_LEV",
               "CONST", "MARK_STACK0", "MEM_STORE", "MEM_ADDR_SRC", "CMP",
               "ADDR_B0_LO", "ADDR_B0_HI",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeFunctionCall::test_simple_function",
        },
    )


def make_layer10_pc_byte_passthrough_bake_op() -> Operation:
    """Bake the PC byte-passthrough head into ``model.blocks[10].attn`` (slot 11).

    PC byte 0 is written by L3 and overridden by L6/L7 (JSR/JMP/branch target)
    or L9 (LEV mem[BP+8] byte 0). This head carries PC bytes 1-3 across ordinary
    non-branch / non-LEV steps. See ``docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md``.
    INCR-1: thin call over ``_make_r_frame_passthrough_bake_op``. The
    ``requires`` clause pins the attn resize (8 -> 13 heads) before this bake so
    slot 11 is in-bounds.
    """
    return _make_r_frame_passthrough_bake_op(
        op_name="layer10_pc_byte_passthrough_bake",
        head_slot=11,
        spec_fn=_layer10_pc_byte_passthrough_head_spec,
        ir_factory=_layer10_pc_byte_passthrough_ir,
        reads={"IS_BYTE", "HAS_SE", "H1", "CONST",
               "OP_JSR", "OP_JMP", "OP_BZ", "OP_BNZ", "OP_LEV",
               "MARK_STACK0",
               "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        smoke_tests={
            "TestSmokeFunctionCall::test_simple_function",
        },
        requires={"after": "l10_attention_resize"},
    )


def make_layer10_psh_stack0_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_psh_stack0_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_psh_stack0_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.3.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. The head_idx=3
    literal is replaced with a pinned-allocator lookup
    (``_l10_head_idx("layer10_psh_stack0_passthrough_bake.head_3")``); the
    bake_fn stashes a per-bake :class:`AttentionHeadAllocator` on ``attn``.
    See ``make_layer10_carry_relay_bake_op`` for the shared infrastructure.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # See ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the head spec directly into ``attn``
        # (was ``_bake_layer10_psh_stack0_passthrough_head``) so census v2
        # classifies this op as ``declarative``.
        Primitives.generate_attention_head(
            attn, _layer10_psh_stack0_passthrough_head_spec(proxy, S), HD,
        )

    # Dim-ownership claims: L10 attn head 3 PSH STACK0 passthrough.
    #   W_v[3*HD + k, CLEAN_EMBED_LO + k]       for k=0..15
    #   W_v[3*HD + 16 + k, CLEAN_EMBED_HI + k]  for k=0..15
    # LEA-local differential routing (bug #33):
    #   W_v[3*HD + 32 + k, OUTPUT_LO + k]       for k=0..15
    #   W_v[3*HD + 32 + k, CLEAN_EMBED_LO + k]  for k=0..15 (negative weight)
    #   W_v[3*HD + 48 + k, OUTPUT_HI + k]       for k=0..15
    #   W_v[3*HD + 48 + k, CLEAN_EMBED_HI + k]  for k=0..15 (negative weight)
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"3_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"3_{16 + k}", f"CLEAN_EMBED_HI+{k}"))
        _claims.add((10, "attn_W_v", f"3_{32 + k}", f"OUTPUT_LO+{k}"))
        _claims.add((10, "attn_W_v", f"3_{32 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"3_{48 + k}", f"OUTPUT_HI+{k}"))
        _claims.add((10, "attn_W_v", f"3_{48 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_psh_stack0_passthrough_bake",
        reads={"MARK_STACK0", "IS_BYTE", "PSH_AT_SP", "H1", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               # LEA-local differential routing (bug #33) reads the OUTPUT
               # bands at the attended AX byte 0 row. Phase 8.A G7 finisher:
               # OUTPUT_LO_PREV_STEP / OUTPUT_HI_PREV_STEP mark these as
               # cross-step reads relative to the L13/L14/L15/L16/L17
               # OUTPUT_LO/HI writers that all fire AFTER this L10 op in
               # the same step (the attention-V read is therefore step
               # N-1's value). PREV_STEP aliases share the same numeric
               # base as OUTPUT_LO/OUTPUT_HI (dim_registry _pin), so baked
               # weight cells stay byte-identical; only the dep-graph view
               # changes. Mirrors the L7 ``layer7_operand_gather`` rename
               # (commits 6967fe8f, 8e6ec805).
               "OUTPUT_LO.*.-1", "OUTPUT_HI.*.-1"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_psh_stack0_passthrough_ir,
        # Phase 3 (mem cluster fix, 2026-06-05): retargeted from
        # ``layer10_carry_relay`` to ``_layer10_attn_anchor`` (the new
        # attn-only sibling anchor). Metadata-only at L10 today.
        target_op_name="_layer10_attn_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"TestSmokeBasic::test_add_basic"},
        spec_section="BLOG_SPEC.md#registers",
    )


# === Wave 1 A3 attn resize + broadcast bake ==============================


def _layer10_attention_resize_follow_up(block, dim_positions, S) -> None:
    """Post-resize bookkeeping: restore the per-head ALiBi pins wiped by resize.

    The structural resize rebuilds ``attn.alibi_slopes`` from scratch with
    the standard ``2 ** (-8/N * (i+1))`` decay, wiping any per-head pins
    that earlier bakes had stamped. L10 head 1 (byte_passthrough) and
    head 2 (sp_byte_passthrough) had pin=1.0 set inside their bake_fns
    BEFORE this resize fires; head 6 (stack0_persistence) re-pins itself
    AFTER the resize from its own bake. Restore the head 1/2 pins here.
    """
    del dim_positions, S
    attn = block.attn
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[1] = 1.0
        attn.alibi_slopes.data[2] = 1.0


def _layer10_attention_resize_structural_ir(dim_positions, head_dim) -> CompilerIR:
    """Build the declarative CompilerIR for ``l10_attention_resize``.

    The L10 attention block default-builds with ``num_heads=8`` but Wave
    1 A3 broadcast heads land at slots 8/9/10 — the attention block
    must grow accordingly. Mirrors the L15 resize pattern with a single
    ``target_num_heads=13`` for all build configurations (slot 11
    hosts the JSR/LEV PC byte_passthrough head).
    """
    del dim_positions, head_dim
    ir = CompilerIR()
    ir.layer(0).structural_ops.append(
        StructuralOp(
            kind="attention_resize",
            target_num_heads=13,
            alibi_pin_value=None,
            follow_up=_layer10_attention_resize_follow_up,
            metadata={
                "op_name": "l10_attention_resize",
                "spec_section": "BLOG_SPEC.md#registers",
            },
        )
    )
    return ir


def make_l10_attention_resize_op() -> Operation:
    """Resize L10 attention from ``num_heads=8`` to ``num_heads=13``.

    Wave 1 A3 prerequisite: default L10 attn ships with 8 heads, so
    the A3 broadcast heads (slots 8/9/10) would write out-of-bounds
    into ``attn.W_q/W_k/W_v/W_o``. This op resizes the attention block
    to 13 heads before the broadcast / PC byte_passthrough bakes fire,
    mirroring ``l15_attention_resize``. Existing heads 0..7 are
    preserved bit-for-bit; the leading slice of each weight matrix is
    copied through unchanged. Runs AFTER the original L10 attn bakes
    (phase 10.0..10.3) and BEFORE the A3 broadcast bake (phase 10.35),
    PC byte_passthrough bake (phase 10.37), and stack0_byte_relay bake
    (phase 10.4). Slot 11 hosts the PC byte_passthrough head added by
    the JSR/LEV follow-up.
    """
    def bake(block, dim_positions, S):
        ir = _layer10_attention_resize_structural_ir(dim_positions, None)
        ir.lower_structural_ops(block, dim_positions, S=S)

    return Operation(
        name="l10_attention_resize",
        slot_share=("attn",),
        reads=set(),
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_attention_resize_structural_ir,
        target_op_name="_layer10_attn_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        # Run AFTER the original L10 attn bakes so we resize an already-
        # populated block; before the broadcast + stack0_byte_relay
        # bakes so they see the wider attn.
        requires={"after": "layer10_psh_stack0_passthrough_bake"},
        claims=set(),
        produces={'__module_replacement': 'L10.attn[resize num_heads]'},
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_psh_ax_broadcast_bake_op() -> Operation:
    """Wave 1 A3 — broadcast AX byte 1/2/3 -> STACK0_BYTE_VAL_h on PSH.

    Three heads (slots 8/9/10), one per byte h. Q fires at MARK_STACK0 +
    BYTE_INDEX_h + OP_PSH; K fires at MARK_AX + BYTE_INDEX_h; V copies
    CLEAN_EMBED_{LO,HI}; O writes STACK0_BYTE_VAL_{h}_{LO,HI} (Wave 1 A1
    dim family). Closes the missing producer identified in
    docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        for byte_h in (1, 2, 3):
            Primitives.generate_attention_head(
                attn,
                _layer10_psh_ax_broadcast_head_spec(proxy, S, byte_h),
                HD,
            )

    # Dim-ownership claims: heads 8/9/10 V slots 0..31 read CLEAN_EMBED.
    _claims = set()
    for byte_h in (1, 2, 3):
        head_idx = 7 + byte_h
        for k in range(16):
            _claims.add(
                (10, "attn_W_v", f"{head_idx}_{k}", f"CLEAN_EMBED_LO+{k}"),
            )
            _claims.add(
                (10, "attn_W_v", f"{head_idx}_{16 + k}",
                 f"CLEAN_EMBED_HI+{k}"),
            )

    return Operation(
        name="layer10_psh_ax_broadcast_bake",
        reads={"MARK_STACK0", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_PC",
               "MARK_MEM", "IS_BYTE", "OP_PSH", "OP_SI", "OP_SC", "OP_JSR",
               "OP_ENT", "H1", "CONST",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
                "STACK0_BYTE_VAL_2_LO", "STACK0_BYTE_VAL_2_HI",
                "STACK0_BYTE_VAL_3_LO", "STACK0_BYTE_VAL_3_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_psh_ax_broadcast_ir,
        target_op_name="_layer10_attn_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        # Ensure resize runs first so the block has 12 heads available.
        requires={"after": "l10_attention_resize"},
        claims=_claims,
        smoke_tests={"TestSmokeMemory::test_si_li_roundtrip"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_stack0_byte_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_stack0_byte_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (lookup branch only):
    ``_set_layer10_stack0_byte_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.4.

    Phase 6 Wave 2D: migrated to ``AttentionHeadIR`` form. This op owns
    heads 4, 5, and 6 (bitwise stack-byte relay, non-bitwise stack-byte
    relay, STACK0 persistence). The three head_idx literals (4/5/6) are
    replaced with pinned-allocator lookups via
    ``_l10_head_idx("layer10_stack0_byte_relay_bake.head_<n>")``; spec
    output is byte-identical to baseline. The bake_fn stashes a
    per-bake :class:`AttentionHeadAllocator` on ``attn``. See
    ``make_layer10_carry_relay_bake_op`` for the shared infrastructure.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        # Per-bake attention-head allocator with the L10 head layout pinned.
        # This bake op owns three heads (4, 5, 6). See
        # ``make_layer10_carry_relay_bake_op`` for the rationale.
        head_allocator = _allocate_layer10_attention_heads()
        attn._l10_head_allocator = head_allocator
        HD = attn.W_q.shape[0] // attn.num_heads
        # Phase 8.C inline: lower the three head specs directly into
        # ``attn`` (was ``_bake_layer10_stack0_byte_relay_head``) so census
        # v2 classifies this op as ``declarative``.
        Primitives.generate_attention_head(
            attn, _layer10_stack0_byte_relay_head_spec(proxy, S), HD,
        )
        Primitives.generate_attention_head(
            attn, _layer10_nonbitwise_stack0_byte_relay_head_spec(proxy, S), HD,
        )
        Primitives.generate_attention_head(
            attn, _layer10_stack0_persistence_head_spec(proxy, S), HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[6] = 1.0

    # Dim-ownership claims: L10 attn head 4/5 stack-memory byte relays
    # (→ ALU at AX byte) and head 6 STACK0 upper-byte carry.
    #   W_v[h*HD + 1 + k, CLEAN_EMBED_LO + k]  for k=0..15
    #   W_v[h*HD + 17 + k, CLEAN_EMBED_HI + k] for k=0..15
    _claims = set()
    for head_idx in (4, 5):
        for k in range(16):
            _claims.add((10, "attn_W_v", f"{head_idx}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((10, "attn_W_v", f"{head_idx}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    for k in range(16):
        _claims.add((10, "attn_W_v", f"6_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"6_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_stack0_byte_relay_bake",
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers (which fire after L10 in the
        # same step). Same numeric position as TEMP. See
        # layer10_byte_passthrough for the per-band rationale.
        # Declaration audit (2026-06-05): added MARK_STACK0 (head 6
        # persistence STORE-target Q rows) and MEM_ADDR_SRC (head 6 store
        # gate Q rows).
        reads={"IS_BYTE", "HAS_SE", "H1", "H4", "TEMP.*.-1", "CMP",
               "PSH_AT_SP", "MARK_STACK0", "MEM_ADDR_SRC",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
               "OP_PSH", "OP_SI", "OP_SC", "OP_LEV", "MEM_STORE", "MARK_MEM",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_stack0_byte_relay_ir,
        # Phase 3 (mem cluster fix, 2026-06-05): retargeted from
        # ``layer10_carry_relay`` to ``_layer10_attn_anchor`` (the new
        # attn-only sibling anchor). Metadata-only at L10 today.
        target_op_name="_layer10_attn_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_alu_op() -> Operation:
    """L10 FFN: AND/OR/XOR + DIV/MOD setup.

    Pinned to ``layer_idx=10`` via ``kind="block"``: the legacy
    ``set_vm_weights`` lookup branch targeted ``model.blocks[10].ffn``.
    Without pinning, dep-graph layer assignment could place this op on the
    wrong block. ``phase=10.2`` is before
    ``make_l10_post_op_attach_op`` (phase=10.7) and
    ``make_l10_alu_divmod_install_op`` (phase=10.8) so they don't conflict.

    Migrated 2026-05-10: the inline ``_set_layer10_alu(ffn10, S, BD)`` call
    in the lookup branch of ``set_vm_weights`` has been removed; this op
    now owns the bake. (Per Unit 9 diagnosis, this migration is SAFE so
    long as ``make_l10_post_op_attach_op`` is NOT modified.)

    Phase 6 Wave 4I (2026-06-01): the bake is now driven entirely by
    declarative ``FFNRule`` data via ``_layer10_alu_rules`` /
    ``_bake_layer10_alu_rules``. ``vm_step._set_layer10_alu`` is no
    longer called; the per-sub-stage byte-identity tests in
    ``test_declarative_ffn_bakes_l10_alu.py`` pin the rules against the
    legacy helper for all seven sub-stages (cmp_combine, bitwise OR /
    XOR / AND, mul_lo, shl_shr_zero, ax_passthrough).

    Declarations-only note: this migrated owner is now exposed through the
    declarations-only dispatcher so strict builds do not fall back to legacy
    model bake.
    """
    def bake(block, dim_positions, S):
        # Per-bake FFN-unit allocator. Every sub-stage of the legacy
        # ``_set_layer10_alu`` helper is pinned at its existing offset
        # so the rule lowering below lands byte-identically. The
        # allocator object is stashed on ``block.ffn`` so downstream
        # tools (a future L10 op family, the per-op audit, etc.) can
        # inspect or extend the layout without re-reading the helper
        # source.
        allocator = _allocate_l10_main_ffn_units()
        block.ffn._l10_unit_allocator = allocator

        proxy = _as_setdim_proxy(dim_positions)
        # Phase 8.C inline: lower the rule list directly (was
        # ``_bake_layer10_alu_rules``) so census v2 classifies this op
        # as ``declarative`` rather than ``declarative_via_helper``.
        rules = _layer10_alu_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy, Primitives.ffn_rule_dim_names(rules),
        )
        n10 = Primitives.lower_ffn_rules(
            block.ffn, rules, rule_dim_positions, start_unit=0, S=S,
        )
        # Byte-identity guard: the rule lowering's final cursor MUST
        # equal the total declared in ``_L10_FFN_UNIT_LAYOUT_MAIN``. If
        # any sub-stage rule generator drifts, this fires before the
        # mismatch propagates to downstream layers.
        _main_expected = _L10_FFN_UNIT_LAYOUT_MAIN_TOTAL + _l10_main_cmp_margin_extra()
        assert n10 == _main_expected, (
            f"L10 ALU unit cursor drift: rules lowered {n10} units, "
            f"allocator expected {_main_expected}"
        )

    return Operation(
        name="layer10_alu",
        # Phase 9.B (ALU_HI SCC rename): ALU_HI -> ALU_HI.*.-1 marks the
        # read as SSA cross-step. L10 stack0_byte_relay_bake (phase 10.4)
        # writes ALU_HI for the NEXT step's L9/L10 consumption; same-step
        # fresh ALU_HI from L7 operand_gather is still observed via
        # consumes_fresh (ALU_HI@AX_byte0) below. Same numeric slot via
        # SSA alias; byte-identical bake. Breaks 1 L10.4 -> L10.2
        # back-edge.
        reads={"MARK_AX", "ALU_LO", "AX_CARRY_LO", "ALU_HI.*.-1", "AX_CARRY_HI",
               "OP_OR", "OP_XOR", "OP_AND", "OP_DIV", "OP_MOD",
               # Wall-4 EQ engine (2026-06-12): reads OP_EQ + the raw
               # operand bands at MARK_AX to recompute equality directly.
               "OP_EQ",
               # Ordering engine (2026-06-12): gated on the OR of the six
               # comparison opcode flags, reads the same raw operand bands
               # at MARK_AX to recompute the full hi/lo lt/eq cascade for
               # LT/GT/LE/GE (and EQ/NE).
               "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               # Wave B Phase 2.2 (2026-06-12): the cmp_combine sub-stage
               # (_layer10_alu_cmp_combine_rules) now gates on the relayed
               # SE_OP_<cmp> dispatch flags at MARK_SE_ONLY (raw OP_<cmp>
               # is cold at the SE row). The CMP cascade it reads is
               # computed fresh at the SE row by the L9 CMP rules. The L9
               # step_end_operand_relay transmits SE_OP_<cmp> after the
               # slope fix (make_layer9_se_relay_slope_op). The MARK_AX
               # ordering engine still drives the live decode this phase.
               "SE_OP_EQ", "SE_OP_NE", "SE_OP_LT",
               "SE_OP_GT", "SE_OP_LE", "SE_OP_GE",
               # V2/G7 LEV detector: in-step topology edge replacing the
               # cross-step requires["after"]=layer16_lev_routing below.
               "PC_VIA_LEV_DETECTOR_LO"},
        # Ordering engine (2026-06-12): the SOLE CMP-flag writer for all
        # six comparison opcodes -- writes CMP+0 (hi_lt), CMP+1 (hi_eq),
        # CMP+2 (lo_eq), CMP+3 (lo_lt) at the AX row, feeding the live
        # ComparisonCombine for LT/GT/LE/GE and the EQ/NE override. The
        # eq_engine no longer writes CMP (only its OUTPUT decode-margin).
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "DIV_STAGING", "CMP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer10_alu_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        migrated=True,
        # Staleness invariants (Phase 3 / Agent G): L10 ALU consumes
        # ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) at the AX
        # marker for bitwise OR/XOR/AND + DIV/MOD setup. Both must be
        # current-step fresh values.
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        # ``_set_layer10_alu`` writes the comparison-combine (18 units) +
        # bitwise-cross-product (1722 = 1536 lookup + 186 stale-ALU
        # residue cancel) + MUL lookup (256) + SHL/SHR zero (4) + AX
        # passthrough (32), reaching unit 2031. No other op writes to
        # L10 FFN so this op holds the per-layer width annotation.
        # The +186 cancel units (2026-06-10) live inside
        # ``_layer10_alu_bitwise_rules`` -- see that helper's docstring.
        # Wall-4 EQ engine (2026-06-12): +256 units (eq_engine) reaching
        # 2288. See ``_layer10_alu_eq_engine_rules``.
        # Ordering engine (2026-06-12): +272 units reaching 2560. See
        # ``_layer10_alu_ordering_engine_rules``.
        # C4_CMP_COMBINE_MARGIN campaign clamp bank appended last: +6 flag-ON,
        # +0 flag-OFF (byte-identical golden).
        ffn_units_used=2560 + _l10_main_cmp_margin_extra(),
        smoke_tests={
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
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


def make_layer10_stack0_byte_relay_op() -> Operation:
    """Topology anchor for L10 stack byte relays.

    The actual weight bake is owned by ``layer10_stack0_byte_relay_bake``
    above, pinned to ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_stack0_byte_relay",
        # Phase 8.A.6 v2: matches layer10_stack0_byte_relay_bake's
        # TEMP_PREV_STEP rename. See that op for rationale.
        # Declaration audit (2026-06-05): added CONST, MARK_STACK0,
        # MEM_ADDR_SRC, HAS_SE, BYTE_INDEX_3 to mirror the persistence head
        # spec's STORE-target Q rows (head 6 stack persistence). Anchor
        # reads gate dim-lifetime analysis; the bake's block kind filters
        # it out of the same analysis.
        reads={"MARK_AX", "IS_BYTE", "HAS_SE", "H1", "H4", "TEMP.*.-1",
               "CMP", "CONST", "MARK_STACK0", "MEM_ADDR_SRC",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
               "PSH_AT_SP", "OP_PSH", "OP_SI", "OP_SC", "OP_LEV", "MEM_STORE", "MARK_MEM",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR makes the bake noop explicit and
        # unblocks Phase 10.E/F multiplexer's opcode-class derivation.
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _suppress_ffn_on_step_boundary(ffn, dim_positions, S: float, units: int | None = None) -> None:
    """Require a marker or byte-lane signal for byte/marker cleanup FFNs."""

    def resolve_dim(name: str):
        if isinstance(dim_positions, dict) and name in dim_positions:
            return dim_positions[name]
        proxy = _as_setdim_proxy(dim_positions if isinstance(dim_positions, dict) else {})
        return getattr(proxy, name, None)

    const_dim = resolve_dim("CONST")
    if const_dim is None:
        return
    unit_count = ffn.W_up.data.shape[0] if units is None else units
    if const_dim >= ffn.W_up.data.shape[1]:
        return
    strength = S * 10_000_000
    ffn.W_up.data[:unit_count, const_dim] -= strength
    for structural_name in (
        "IS_BYTE",
        "MARK_AX",
        "MARK_PC",
        "MARK_SP",
        "MARK_BP",
        "MARK_STACK0",
        "MARK_MEM",
    ):
        structural_dim = resolve_dim(structural_name)
        if (
            structural_dim is not None
            and structural_dim < ffn.W_up.data.shape[1]
        ):
            rows = ffn.W_up.data[:unit_count, structural_dim]
            rows[rows >= 0] += strength


def make_l10_post_ops_combined() -> Operation:
    """Combined L10 post_ops: BinaryOpByteZeroing + 3x CarryPropagation +
    ComparisonCombine, baked sequentially into one FFN.

    Originally these were 6 separate post_ops on L10 in vm_step.py. Per Phase 0
    policy they belong in their own blocks, but for the migration we combine
    the carry/zeroing/comparison subset additively into a single ffn at
    phase=10.5 so the compiler keeps the carry-dependent SUB/DIV smoke path.
    BitwiseBytePropagationPostOp is intentionally excluded here: the attached
    L10 post-op block already owns that propagation at the correct point in
    the pipeline, and re-running it in this dependency-assigned tail layer
    turns already-computed 16-bit XOR bytes back into zero.
    """
    def bake(ffn, dim_positions, S):
        # Per-bake FFN-unit allocator. Every sub-range matches the
        # hidden_dim of the corresponding FFNRule family lowered into
        # it and is pinned at the offset the historical inline walk
        # would naturally land on. Stashed on the FFN itself (this op
        # is ``kind="ffn"`` so ``ffn`` IS the block-equivalent target)
        # so a future second tenant in this dependency-assigned bank
        # can claim a free gap above unit 1562 through the allocator.
        allocator = _allocate_l10_post_ops_combined_units()
        ffn._l10_unit_allocator = allocator

        # Pull the pinned starts back out of the allocator so the
        # inline walk uses the table as its source of truth. Drift
        # between the walk and the rule-emitted cursors fails fast in
        # the ``assert offset == ...`` checks below.
        by_name = {r.op_name: r for r in allocator.ranges()}

        offset = by_name["l10_post_ops_combined.binary_op_byte_zeroing"].start
        # Migrated to FFNRule. Rule list lives in
        # ``_l10_binary_op_byte_zeroing_rules`` and is lowered through
        # ``Primitives.lower_ffn_rules`` so symbolic / declarative
        # verifiers see the same declarations the imperative
        # ``BinaryOpByteZeroingPostOp._bake_weights`` used to write.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_binary_op_byte_zeroing_rules(S),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.carry_propagation_byte0"].start, (
            f"L10 post_ops_combined zeroing cursor drift: {offset}"
        )
        carry_start = offset
        # Migrated to FFNRule. See ``_l10_carry_propagation_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_carry_propagation_rules(S, byte_idx=0, cascade=False),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.carry_propagation_byte1"].start, (
            f"L10 post_ops_combined carry0 cursor drift: {offset}"
        )
        # Migrated to FFNRule. See ``_l10_carry_propagation_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_carry_propagation_rules(S, byte_idx=1, cascade=True),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.carry_propagation_byte2"].start, (
            f"L10 post_ops_combined carry1 cursor drift: {offset}"
        )
        # Migrated to FFNRule. See ``_l10_carry_propagation_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_carry_propagation_rules(S, byte_idx=2, cascade=True),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        assert offset == by_name["l10_post_ops_combined.comparison_combine"].start, (
            f"L10 post_ops_combined carry2 cursor drift: {offset}"
        )
        carry_end = offset
        # Migrated to FFNRule. See ``_l10_comparison_combine_rules``.
        offset = Primitives.lower_ffn_rules(
            ffn,
            _l10_comparison_combine_rules(S),
            dim_positions,
            start_unit=offset,
            S=S,
        )
        _pooc_expected = (
            _L10_FFN_UNIT_LAYOUT_POST_OPS_COMBINED_TOTAL
            + _l10_main_cmp_margin_extra()
        )
        assert offset == _pooc_expected, (
            f"L10 post_ops_combined comparison cursor drift: helper "
            f"ended at {offset}, allocator expected {_pooc_expected}"
        )
        # The attached L10 post-op pipeline is now the authoritative carry
        # implementation. This late dependency-assigned copy sees very large
        # downstream OUTPUT residuals, and the legacy carry units can turn
        # those into runaway byte rewrites even when the earlier byte result
        # is already correct. Keep zeroing/comparison here, but remove the
        # stale carry slice entirely.
        if carry_end > carry_start:
            ffn.W_up.data[carry_start:carry_end, :].zero_()
            ffn.b_up.data[carry_start:carry_end].zero_()
            ffn.W_gate.data[carry_start:carry_end, :].zero_()
            ffn.b_gate.data[carry_start:carry_end].zero_()
            ffn.W_down.data[:, carry_start:carry_end].zero_()
        # This combined post-op block is dependency-assigned late in the
        # expanded model. The structural L10 post-op blocks already own ADD/SUB
        # carry propagation at the correct point in the pipeline; re-running
        # the same carry detectors here can increment or borrow bytes a second
        # time. The wide ALU composites likewise own MUL/SHL/SHR results by
        # this point; leaving the legacy carry detectors active increments the
        # high byte after the L15 relay. LI/LC also already have authoritative
        # bytes from L15.
        # Express the per-opcode / CARRY / TEMP / CMP / H1 W_up suppressor
        # band as a declarative mapping passed to
        # ``Primitives.apply_ffn_band_suppressors``. The previous code
        # form was a sequence of inline tensor-slice assignments plus a
        # call to ``_suppress_ffn_on_step_boundary``; both are subsumed by
        # the primitive's declarative parameters. Strength values are
        # kept byte-identical to the original assignments.
        _SUPPRESSORS = {
            "OP_IMM": 1000.0,
            "OP_JMP": 1000.0,
            "OP_LI_RELAY": 1000.0,
            "OP_LC_RELAY": 1000.0,
            "OP_ADD": 1000.0,
            "OP_SUB": 1000.0,
            "OP_MUL": 1000.0,
            "OP_SHL": 1000.0,
            "OP_SHR": 1000.0,
            "CARRY+1": 1000000.0,
            "CARRY+2": 1000000.0,
            "CARRY+3": 1000000.0,
            # L7 relays ADD/SUB to TEMP[8]/TEMP[9] at AX byte rows, and
            # the wide ALU path relays MUL byte-1 ownership to TEMP[10].
            # This late dependency-tail copy of legacy post-ops must not
            # rerun byte logic after the immediate structural blocks have
            # already materialized the authoritative result.
            "TEMP+8": 1000000.0,
            "TEMP+9": 1000000.0,
            "TEMP+10": 1000000.0,
            # L7 relays LEA to CMP[7] at AX byte rows. LEA bytes are
            # already materialized by L16; the dependency-tail copy of L10
            # post-ops must not rerun carry propagation over them.
            "CMP+7": 1000.0,
            # This dependency-assigned copy of the L10 byte post-ops runs
            # after many later corrections, where nonmatching OUTPUT
            # nibbles can be strongly negative. The legacy carry detectors
            # use negative OUTPUT blockers; on PC byte rows those blockers
            # become large positive evidence and can erase L3's PC-byte
            # output. The L10 byte post-ops are AX-oriented, so suppress
            # the whole combined block across the PC byte span.
            "H1+0": 10000.0,
        }
        # The dependency-assigned copy is byte/marker cleanup. At
        # step-boundary prediction rows there is no marker and no byte
        # lane yet, so stale OUTPUT residue can make the legacy units
        # overwhelm the next marker token. Require a structural row signal
        # while leaving real marker and byte rows unchanged. Sign-dependent
        # (rows[rows >= 0] += strength) byte-identical to the legacy
        # ``_suppress_ffn_on_step_boundary`` helper.
        Primitives.apply_ffn_band_suppressors(
            ffn,
            dim_positions,
            end_unit=offset,
            S=S,
            suppressors=_SUPPRESSORS,
            marker_boost_strength=S * 10_000_000,
            marker_boost_const_dim="CONST",
            marker_boost_structural_dims=(
                "IS_BYTE",
                "MARK_AX",
                "MARK_PC",
                "MARK_SP",
                "MARK_BP",
                "MARK_STACK0",
                "MARK_MEM",
            ),
        )

    # phase=10.5 so it lands AFTER layer10_alu (phase=10) but BEFORE later layers
    # which depend on its OUTPUT_LO/HI updates. Note: float phases work because
    # phase comparison uses < / >.
    return Operation(
        name="l10_post_ops_combined",
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers. Same numeric position as TEMP.
        # See layer10_byte_passthrough for the per-band rationale.
        # Phase 8.A G7: OUTPUT_HI_THIS_STEP read renamed to
        # OUTPUT_HI_PREV_STEP. The combined post-op block uses OUTPUT_HI
        # as a residual gate for carry-propagation / byte zeroing -- the
        # value it actually reads at phase 10.5 is the residual carried
        # from the PREVIOUS step's final OUTPUT writer, NOT a same-step
        # data flow from later-layer OUTPUT_HI_THIS_STEP writers (L12+/
        # L14+/L15+/L16). Mirrors the OUTPUT_LO_PREV_STEP rename for the
        # same op (commit e7ee64bd). The alias shares numeric position
        # 190 with OUTPUT_HI so bakes stay byte-identical. Breaks 9
        # cross-step back-edges into this op.
        reads={
            "CONST", "MARK_AX", "MARK_PC", "IS_BYTE", "H1",
            # Declaration audit (2026-06-05): added MARK_SP, MARK_BP,
            # MARK_STACK0, MARK_MEM (consumed by marker_boost_structural_dims
            # in apply_ffn_band_suppressors); added TEMP (TEMP+8/9/10 are
            # hard-blocked via suppressors, distinct from the cross-step
            # TEMP.*.-1 read).
            "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM", "TEMP",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_SHL", "OP_SHR",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_OR", "OP_XOR", "OP_AND",
            "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
            "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC",
            "OP_SI", "OP_SC", "OP_PSH", "OP_EXIT", "OP_NOP",
            "OP_PUTCHAR", "OP_GETCHAR",
            "OP_LI_RELAY", "OP_LC_RELAY",
            # Phase 9.B (SCC #2 dissolution): OUTPUT_LO -> OUTPUT_LO.*.-1
            # marks the read as SSA cross-step relative to the
            # downstream OUTPUT_LO writer ``tail_bit32_result_correction``
            # (phase=17.1). This op runs at phase=10.5 and uses
            # OUTPUT_LO as a residual gate for carry-propagation / byte
            # zeroing — the value read at phase 10.5 is the residual
            # carried from the PREVIOUS step's final OUTPUT_LO writer,
            # NOT a same-step data flow from the much-later L17 tail
            # correction. Mirrors the OUTPUT_HI.*.-1 alias on the line
            # below and the TEMP.*.-1 / CARRY.*.-1 aliases on the same
            # op; aliases share the numeric slot via the SSA rewriter so
            # bakes stay byte-identical. Breaks the OUTPUT_LO back-edge
            # ``tail_bit32_result_correction -> l10_post_ops_combined``.
            "OUTPUT_LO.*.-1", "OUTPUT_HI.*.-1", "ALU_LO", "ALU_HI",
            "CARRY", "CMP", "TEMP.*.-1",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            # V2/G7 LEV detector: in-step topology edge replacing the
            # cross-step requires["after"]=layer16_lev_routing below.
            "PC_VIA_LEV_DETECTOR_LO",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "CARRY"},
        kind="ffn",
        declarative_bake_fn=bake,
        migrated=True,
        declarative_authority="declarative",
        # Phase 11.A IR exposure: empty CompilerIR drops this op from the
        # ``no_ir`` census bucket (the last residual in 11.A). The bake
        # combines four FFNRule families (``_l10_binary_op_byte_zeroing_rules``,
        # 3x ``_l10_carry_propagation_rules``, ``_l10_comparison_combine_rules``)
        # with a declarative residual band suppressor + carry-slice zero-out.
        # Because ``declarative_bake_fn`` takes priority over ``compiler_ir``
        # in ``_resolve_bake_callable`` (layer_compiler.py L146-147), exposing
        # an empty IR here is byte-identical to the previous bake (the
        # imperative path is unchanged). A faithful IR factory would have to
        # carry both the rules and the residual band as a CompilerIR
        # extension (carry-slice zero + ``apply_ffn_band_suppressors`` --
        # the same Phase 7.C-grade pattern flagged in HANDOFF_2026_06_02.md
        # §"Phase 11"); deferred to Phase 11.A follow-up.
        compiler_ir=CompilerIR(),
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        # C4_CMP_COMBINE_MARGIN campaign clamp grows the LAST tenant
        # (comparison_combine) by +6; flag-OFF -> 1562 -> byte-identical golden.
        ffn_units_used=1562 + _l10_main_cmp_margin_extra(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _strengthen_l10_carry_wrong_byte_blockers(post_op, BD, byte_idx: int, S: float) -> None:
    """Harden L10-attached carry post-ops against compact-layout byte leakage."""

    byte_dims = [
        BD.BYTE_INDEX_0,
        BD.BYTE_INDEX_1,
        BD.BYTE_INDEX_2,
        BD.BYTE_INDEX_3,
    ]
    for i, wrong_dim in enumerate(byte_dims):
        if i == byte_idx or wrong_dim >= post_op.W_up.data.shape[1]:
            continue
        # The next byte lane can carry a small softmax residual on true rows
        # (for example BYTE_INDEX_1 ~= 0.013 while predicting from byte 0).
        # Far byte lanes should remain hard blockers because late OUTPUT
        # cleanup can leave large negative residue that otherwise inverts the
        # nibble blockers.
        post_op.W_up.data[:, wrong_dim] = (
            -S * 20 if i == byte_idx + 1 else -S * 100000
        )


def _strengthen_l10_addsub_wrong_byte_blockers(post_op, BD, S: float) -> None:
    """Harden L10-attached ADD/SUB byte-0 post-op against byte-span leakage."""

    # AddSubBytePropagationPostOp's first 1024 units are the byte-0 ADD/SUB
    # base rules. In the compact full model, BP byte3 rows can carry large
    # negative OUTPUT/ALU residue, turning weak wrong-byte blockers into false
    # positives. Leave the later borrow-continuation units untouched because
    # they intentionally target byte indexes 1 and 2.
    main_units = min(1024, int(post_op.W_up.data.shape[0]))
    if BD.BYTE_INDEX_1 < post_op.W_up.data.shape[1]:
        post_op.W_up.data[:main_units, BD.BYTE_INDEX_1] = -S * 20
    for wrong_dim in (BD.BYTE_INDEX_2, BD.BYTE_INDEX_3):
        if wrong_dim < post_op.W_up.data.shape[1]:
            post_op.W_up.data[:main_units, wrong_dim] = -S * 100000


def _strengthen_l10_first_carry_delta(post_op, BD) -> None:
    """Give L10 byte-0 carry enough margin on AX high-byte passthrough rows."""

    for base in (BD.OUTPUT_LO, BD.OUTPUT_HI):
        if base + 16 <= post_op.W_down.data.shape[0]:
            post_op.W_down.data[base:base + 16, :] *= 1.5


def _suppress_l10_addsub_on_wide_alu(post_op, BD, S: float) -> None:
    """Legacy hook retained for compatibility.

    ``TEMP+10`` is not a stable wide-ALU-only signature in the compact
    declarative layout: L4 PC staging also writes it at ordinary AX byte
    positions.  The attached ADD/SUB byte post-op is already gated by the
    explicit L7 ADD/SUB relays (``TEMP+8`` / ``TEMP+9``), so a blanket
    ``TEMP+10`` blocker suppresses valid high-byte arithmetic.
    """

    del post_op, BD, S


def _byte_value_writeback_rules(
    *,
    name_for,
    base_conditions,
    threshold: float,
    strength: float,
    lo_base: str = "OUTPUT_LO",
    hi_base: str = "OUTPUT_HI_THIS_STEP",
    lo_match_weight: float = 1.0,
    hi_match_weight: float = 1.0,
    tie_break_lo_base: Optional[str] = None,
    tie_break_hi_base: Optional[str] = None,
    tie_break_weight: float = 0.0,
    write_lo_base: str = "OUTPUT_LO",
    write_hi_base: str = "OUTPUT_HI",
    competitor_strength: Optional[float] = None,
    gate: Optional[object] = None,
    scope: Optional[str] = None,
    dominates_at: Optional[Mapping[str, str]] = None,
    skip=None,
):
    """Evidence-keyed byte-value writeback engine (Tier-3 tail-bank pattern).

    The ~40 hand-authored sub-generators inside
    :func:`_tail_bit32_result_correction_rules` are 40 variations of ONE
    shape: *"under a fixed structural evidence gate ``base_conditions``, for
    each byte value ``v = lo | (hi << 4)`` observed in a (nibble-low, nibble-
    high) source lane, guarantee OUTPUT byte = v"*.  Concretely each is a
    16x16 nibble loop that, per byte value, appends a per-value match
    condition ``(f"{lo_base}+{lo}", lo_match_weight)`` +
    ``(f"{hi_base}+{hi}", hi_match_weight)`` to the shared evidence and emits
    a :func:`Primitives.byte_value_writes` write of that byte.

    This helper parametrizes exactly those axes so a family collapses from a
    ~40-line nested loop to a single keyword-argument call, while emitting the
    byte-IDENTICAL ``FFNRule`` tuple (same order, same weights, same
    threshold, same gate) — a count-preserving AUTHORING refactor only.

    Args:
        name_for: ``value -> rule_name`` callable (the family's f-string).
        base_conditions: the shared per-family evidence gate tuple, emitted
            verbatim ahead of the per-value nibble match terms.
        threshold: per-rule AND threshold (uniform across the family).
        strength: ``byte_value_writes`` positive/negative write strength.
        lo_base / hi_base: the source-lane residual band names whose one-hot
            nibble encodes the observed byte value (default the L10 OUTPUT
            band; ``ALU_LO`` / ``ALU_HI`` families pass those instead).
        lo_match_weight / hi_match_weight: the per-value match condition
            weights (families vary between 1.0 and 0.05 / 0.001).
        tie_break_lo_base / tie_break_hi_base / tie_break_weight: an OPTIONAL
            secondary per-value nibble-match pair appended verbatim AFTER the
            primary ``lo_base``/``hi_base`` terms (e.g. the cross-lane
            ``stack0_store_top_e0`` family reads its source nibble from
            ``ALU_LO``/``ALU_HI`` at weight 1.0 and adds a
            ``OUTPUT_LO``/``OUTPUT_HI_THIS_STEP`` tie-breaker at weight 0.001).
            Left ``None`` (no extra terms) for the plain OUTPUT-lane banks.
        write_lo_base / write_hi_base: the OUTPUT nibble bands the resulting
            byte is written to (default ``OUTPUT_LO`` / ``OUTPUT_HI`` — the
            ``byte_value_writes`` defaults — so cross-lane READ families can
            still WRITE to OUTPUT).
        competitor_strength: forwarded to ``byte_value_writes`` for families
            that soften the losing-channel suppression (e.g. the shallow
            pop-loaded crush).
        gate / scope / dominates_at: passed through to ``multi_way_and_rule``.
        skip: optional ``(lo, hi) -> bool`` predicate; a value whose nibbles
            satisfy it is omitted (e.g. ``lo == 0 and hi == 0`` for the
            "already-zero is ambiguous" families, or ``value == 0xE0``).

    Returns:
        The family's ``FFNRule`` tuple, in ``value``-ascending order.
    """

    rules = []
    for lo in range(16):
        for hi in range(16):
            if skip is not None and skip(lo, hi):
                continue
            value = lo | (hi << 4)
            match_terms = (
                (f"{lo_base}+{lo}", lo_match_weight),
                (f"{hi_base}+{hi}", hi_match_weight),
            )
            if tie_break_lo_base is not None:
                match_terms += (
                    (f"{tie_break_lo_base}+{lo}", tie_break_weight),
                    (f"{tie_break_hi_base}+{hi}", tie_break_weight),
                )
            rules.append(
                multi_way_and_rule(
                    name=name_for(value),
                    scope=scope,
                    dominates_at=dominates_at,
                    conditions=tuple(base_conditions) + match_terms,
                    threshold=threshold,
                    gate=gate,
                    writes=Primitives.byte_value_writes(
                        value,
                        lo_base=write_lo_base,
                        hi_base=write_hi_base,
                        strength=strength,
                        competitor_strength=competitor_strength,
                    ),
                )
            )
    return tuple(rules)


def _computed_byte_writeback_route_rules(
    *,
    name_for,
    base_conditions,
    threshold: float,
    strength: float,
    lo_base: str = "OUTPUT_LO",
    hi_base: str = "OUTPUT_HI_THIS_STEP",
    match_weight: float = 1.0,
    competitor_strength: Optional[float] = None,
    gate: Optional[object] = None,
    scope: Optional[str] = None,
    dominates_at: Optional[Mapping[str, str]] = None,
):
    """COMPUTED counterpart to :func:`_byte_value_writeback_rules` (M8 pilot).

    Where the enumerated engine emits up to 255 per-VALUE AND units (one per
    byte, each reading BOTH nibbles and writing that whole byte), this emits 32
    per-NIBBLE-CHANNEL ROUTE units: 16 for ``lo_base`` + 16 for ``hi_base``.
    Each channel-``k`` unit:

      * FIRES iff the shared ``base_conditions`` structural evidence holds AND
        this channel's one-hot (``{band}+{k}``, weight ``match_weight``) is on
        AND the OTHER band carries a one-hot (the 16 ``{other}+{j}`` terms,
        weight ``match_weight`` each; exactly one is on for a valid byte).
        Summing the other band keeps the firing decision on the SAME 2-channel
        evidence magnitude and threshold the enumerated per-value AND used, so
        the route fires on exactly the same production contexts.  Because the
        OTHER band is one-hot in production (only the observed nibble carries a
        large magnitude), ``match_weight * sum_j other[j]`` reduces to
        ``match_weight * other[observed]`` — i.e. the route's channel-``k`` unit
        has the IDENTICAL firing predicate to the enumerated per-value unit for
        the byte whose LO (resp. HI) nibble is ``k``.  Verified numerically in
        ``tools/_probe_m8_computed_writeback.py`` (pilot, ``match_weight=1``)
        and ``tools/_probe_computed_writeback_banks.py`` (the 0.05 / 0.001
        weight banks): identical firing region + 0 argmax mismatch across all
        256 bytes.
      * WRITES ``nibble_value_writes(band, k)`` — ``+strength`` to channel ``k``
        and ``-competitor_strength`` to the 15 competitors of ITS band.

    When the LO and HI route units for the observed byte both fire, together
    they reconstruct the same OUTPUT byte the single enumerated unit wrote —
    a COMPUTED copy, not a 256-way lookup.  Same order (LO band then HI band,
    channel-ascending), same gate / scope / dominates_at.

    ``match_weight`` must equal the enumerated bank's per-value nibble match
    weight (1.0 for the pilot, 0.05 for ``stack0_pop_loaded``, 0.001 for
    ``stack0_store_top_e8_from_e0``) so the firing threshold algebra is
    preserved; ``competitor_strength`` is forwarded to
    ``nibble_value_writes`` (banks that soften the losing-channel suppression,
    e.g. the shallow pop-loaded crush, pass it explicitly).
    """

    rules = []
    for band, other in ((lo_base, hi_base), (hi_base, lo_base)):
        other_terms = tuple(
            (f"{other}+{j}", match_weight) for j in range(16)
        )
        for k in range(16):
            rules.append(
                multi_way_and_rule(
                    name=name_for(band, k),
                    scope=scope,
                    dominates_at=dominates_at,
                    conditions=tuple(base_conditions)
                    + ((f"{band}+{k}", match_weight),)
                    + other_terms,
                    threshold=threshold,
                    gate=gate,
                    writes=Primitives.nibble_value_writes(
                        band, k, strength=strength,
                        competitor_strength=competitor_strength,
                    ),
                )
            )
    return tuple(rules)


def _tail_bit32_result_correction_rules() -> tuple[FFNRule, ...]:
    """Late FFN correction rules after the dependency-assigned post-op tail.

    Phase 8.D: marker gates (MARK_SP / MARK_STACK0 / MARK_AX / MARK_SP)
    and a few opcode_flag / carry / byte_index gates are bound up
    front via :func:`dim_ref` so each inner rule generator names the
    semantic family/role rather than the bare slot string.
    """

    # Phase 8.D: pre-bound role-meaningful refs reused across the
    # inner generators below.
    gate_mark_sp = dim_ref("marker", "SP")
    gate_mark_stack0 = dim_ref("marker", "STACK0")
    gate_mark_ax = dim_ref("marker", "AX")

    def byte_writes(value: int, strength: float = 100.0):
        return Primitives.byte_value_writes(value, strength=strength)

    def exact_output_byte_rules(
        *,
        name: str,
        expected_byte: int,
        conditions,
        threshold: float,
        active_value: float = 4.0,
        max_abs_weight: float = 1_000_000.0,
        scope: Optional[str] = None,
        dominates_at: Optional[Mapping[str, str]] = None,
    ) -> tuple[FFNRule, ...]:
        """Lane-local exact output-byte guarantee from structural evidence."""

        return expected_byte_guarantee_rules(
            expected_byte=expected_byte,
            activation_conditions=conditions,
            condition_threshold=threshold,
            inactive_value=0.0,
            active_value=active_value,
            min_margin=1.0,
            max_abs_weight=max_abs_weight,
            name=name,
            scope=scope,
            dominates_at=dominates_at,
        )

    def clear_output_writes(strength: float = 100.0):
        writes = []
        for k in range(16):
            writes.append((f"OUTPUT_LO+{k}", -strength))
            writes.append((f"OUTPUT_HI_THIS_STEP+{k}", -strength))
        return tuple(writes)

    def addr_from_l13_rules(
        *,
        name: str,
        target_byte: int,
        lo_lane: int,
        hi_lane: int,
        extra_conditions: tuple = (),
        threshold: float = 140.0,
        strength: float = 10_000.0,
        scope: Optional[str] = None,
        dominates_at: Optional[Mapping[str, str]] = None,
    ) -> tuple[FFNRule, ...]:
        """Emit byte 0 for MEM-store rows directly from L13 ADDR_B0 lanes.

        This replaces the strength-escalation ad-hoc rules in the
        ``tail_mem_store_addr0_*`` family.  The discrimination evidence is the
        L13 one-hot address gather (``ADDR_B0_LO+{lo}`` and ``ADDR_B0_HI+{hi}``)
        instead of disjoint ``ALU_LO``/``CMP``/``PSH_AT_SP``/``OP_JSR``/
        ``MEM_ADDR_SRC`` witnesses that previous siblings had to overpower with
        ever-growing strengths.  Rule Q
        (``tail_mem_store_addr0_e8_from_local_frame_addr_exact``) is the design
        prototype this helper generalises.

        Strength defaults to 10k because the L13 ADDR_B0 lanes ARE the correct
        byte address; the evidence is decisive and the rule does not need to
        outvote other siblings via raw magnitude.

        ``extra_conditions`` are appended verbatim (e.g. ``OP_JSR`` /
        ``OP_ENT`` gates for the JSR or PSH-at-SP variants).
        """

        other_lo = tuple(
            (f"ADDR_B0_LO+{k}", -200.0) for k in range(16) if k != lo_lane
        )
        other_hi = tuple(
            (f"ADDR_B0_HI+{k}", -200.0) for k in range(16) if k != hi_lane
        )
        base = (
            ("MARK_MEM", 1.0),
            ("HAS_SE", 1.0),
            ("H1+4", 20.0),
            # E3 fix: hard PC-row blocker. The existing MARK_PC=-100 weight
            # was overwhelmed at PC byte0 rows by H1+0≈0.94 carrying enough
            # signal for the 0xe8 writer (and siblings) to misfire on
            # rec_power, emitting 0xe2 in the PC byte0 lane. Promoting H1+0
            # to a -1M blocker (matching H1+1/2/3/10) cleanly suppresses any
            # PC-row firing without affecting MEM rows (which sit on H1+4).
            ("H1+0", -1_000_000.0),
            ("H1+1", -1_000_000.0),
            ("H1+2", -1_000_000.0),
            ("H1+3", -1_000_000.0),
            ("H1+10", -1_000_000.0),
            ("MEM_STORE", 5.0),
            (f"ADDR_B0_LO+{lo_lane}", 50.0),
            (f"ADDR_B0_HI+{hi_lane}", 50.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1_000_000.0),
            ("MARK_PC", -100.0),
            ("MARK_SP", -100.0),
            ("MARK_BP", -100.0),
            ("MARK_STACK0", -100.0),
            ("NEXT_PC", -1_000_000.0),
            ("NEXT_AX", -1_000_000.0),
            ("NEXT_SP", -1_000_000.0),
            ("NEXT_BP", -1_000_000.0),
            ("NEXT_STACK0", -1_000_000.0),
            ("NEXT_MEM", -1_000_000.0),
            ("NEXT_SE", -1_000_000.0),
        ) + other_lo + other_hi + tuple(extra_conditions)
        return (
            multi_way_and_rule(
                name=name,
                scope=scope,
                dominates_at=dominates_at,
                conditions=base,
                threshold=threshold,
                writes=byte_writes(target_byte, strength=strength),
            ),
        )

    def sp_pop_carry_rules() -> tuple[FFNRule, ...]:
        """Autoregressive upper-byte carry for binary-pop SP += 8.

        L6 computes SP byte 0 at the marker. L10 SP passthrough carries old
        upper bytes into OUTPUT at SP byte positions; these rules increment the
        carried byte when an already-generated lower byte proves a carry.
        """

        rules = []
        non_pop_opcode_blockers = (
            ("OP_IMM", -1000000.0),
            ("OP_JMP", -1000000.0),
            ("OP_JSR", -1000000.0),
            ("OP_BZ", -1000000.0),
            ("OP_BNZ", -1000000.0),
            ("OP_ENT", -1000000.0),
            ("OP_ADJ", -1000000.0),
            ("OP_LEV", -1000000.0),
            ("OP_LI", -1000000.0),
            ("OP_LC", -1000000.0),
            ("OP_SI", -1000000.0),
            ("OP_SC", -1000000.0),
            ("OP_PSH", -1000000.0),
            ("OP_EXIT", -1000000.0),
            ("OP_NOP", -1000000.0),
            ("OP_PUTCHAR", -1000000.0),
            ("OP_GETCHAR", -1000000.0),
        )
        # Carry propagation is valid through byte 2 for the C4 stack range.
        # Byte 3 is the zero high byte; a previous byte token of 0x00 is
        # ambiguous there and is handled by tail_sp_pop_byte3_zero instead.
        for byte_idx in range(2):
            if byte_idx == 0:
                carry_terms = (
                    (("CLEAN_EMBED_HI+0", 1.0),)
                    + tuple((f"CLEAN_EMBED_LO+{k}", 1.0) for k in range(8))
                    + tuple(
                        (f"CLEAN_EMBED_HI+{k}", -1000.0)
                        for k in range(1, 16)
                    )
                )
            else:
                carry_terms = (
                    ("CLEAN_EMBED_LO+0", 30.0),
                    ("CLEAN_EMBED_HI+0", 30.0),
                ) + tuple(
                    (f"CLEAN_EMBED_LO+{k}", -1000.0) for k in range(1, 16)
                ) + tuple(
                    (f"CLEAN_EMBED_HI+{k}", -1000.0) for k in range(1, 16)
                )
            byte_index_terms = (
                (f"BYTE_INDEX_{byte_idx}", 5.0),
            ) + tuple(
                (f"BYTE_INDEX_{other}", -100.0)
                for other in range(4)
                if other != byte_idx
            )
            base_conditions = (
                ("IS_BYTE", 5.0),
                # These carry materializers are only valid after at least one
                # completed step.  Startup STACK0 byte rows can otherwise carry
                # enough CMP/clean-byte residue to activate every old-value
                # exactness unit with a nominally zero gate; in the SwiGLU
                # lowering that still leaks into OUTPUT.  Make HAS_SE a hard
                # structural part of the proof.
                ("HAS_SE", 2000.0),
                ("H1+2", 20.0),
                ("H1+1", -1000.0),
                ("H1+3", -1000.0),
                ("CMP+3", 1.0),
                ("MARK_SP", -10000.0),
                ("MARK_AX", -10000.0),
                ("MARK_PC", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
                ("MEM_STORE", -1000000.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
            ) + non_pop_opcode_blockers + byte_index_terms + carry_terms

            if byte_idx == 0:
                rules.append(
                    multi_way_and_rule(
                        name="tail_sp_pop_carry_byte1_zero",
                        # Same base_conditions contradiction as the
                        # byte_idx=1 family below (CMP+3 vs BYTE_INDEX_*
                        # via MARK_AX blocker); effective collapses to the
                        # gate H1+2 tautology fallback. No honest scope is
                        # tighter than tautology; leave unset.
                        conditions=base_conditions,
                        threshold=2025.0,
                        gate="H1+2",
                        writes=byte_writes(0x00, strength=150.0),
                    )
                )
                continue

            # Byte-2 carry has a much stronger "previous byte is exactly zero"
            # proof than byte-1 carry. Without a strong staged-byte match,
            # ordinary SP byte rows also cross threshold and every old-value
            # materializer fires. Require the pop relay plus both current
            # output nibbles to be present.
            #
            # The enumeration over ``range(256)`` writes ``(old + 1) & 0xFF`` for
            # every possible incoming byte-2. Across the WHOLE 1096 corpus SP
            # stays in ``[0x0FE10, 0x10000]`` so byte-2 is only ever 0x00 or 0x01
            # and the ONLY carry that fires is ``0x00 -> 0x01`` (the pop back to
            # the frame base ``0x010000``); ``0x02..0xFF`` are structurally
            # unreachable. With ``C4_SP_BYTE2_CARRY=1`` the bank collapses to
            # ``old in {0x00, 0x01}`` (2 rules, byte-for-byte identical to the
            # corresponding members of the 256-rule bank) -- a verdict-neutral
            # ~0.5k-LOC deletion. See ``_sp_byte2_carry_computed_enabled``.
            output_match_weight = 5.0
            # R-FRAME INCR-3 (SP pilot) — P5 RETIRE (unconditional collapse):
            # the byte-2 pop-carry is emitted ONLY in its computed 2-row form
            # (``old in {0x00, 0x01}``). Across the whole 1096 corpus SP stays in
            # ``[0x0FE10, 0x10000]`` so byte-2 is only ever 0x00 or 0x01 and the
            # ONLY carry that fires is ``0x00 -> 0x01`` (the pop back to the frame
            # base ``0x010000``); ``0x02..0xFF`` are structurally unreachable. The
            # two surviving rules are byte-for-byte identical to the corresponding
            # members of the retired 256-rule ``range(256)`` bank. The enumerated
            # fallback + its ``C4_R_FRAME_TAIL`` / ``C4_SP_BYTE2_CARRY`` flag branch
            # are DELETED (P5); the default was already this collapsed form.
            for old_value in range(2):
                rules.append(
                    multi_way_and_rule(
                        name=(
                            f"tail_sp_pop_carry_byte{byte_idx + 1}_"
                            f"{old_value:02x}"
                        ),
                        # The CMP+3 positive condition's semantics
                        # (``mark == AX OR (is_byte AND byte_index == 0)``)
                        # conflicts with BYTE_INDEX_1's positive semantics
                        # (``is_byte AND byte_index == 1``) and the MARK_AX
                        # hard blocker, making the conditions-only
                        # effective predicate unsatisfiable; F-5 falls back
                        # to the gate ``H1+2`` which has no semantics and
                        # collapses to tautology. A declared scope can't be
                        # honest here -- the gated-fallback effective set
                        # is not narrowable without code-level surgery to
                        # the shared base_conditions used by both byte_idx
                        # branches. Leave scope/dominates_at unset (no
                        # claim) until the structural conflict is resolved.
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{old_value & 0xF}", output_match_weight),
                            (f"OUTPUT_HI_THIS_STEP+{old_value >> 4}", output_match_weight),
                        ),
                        threshold=2105.0,
                        gate="H1+2",
                        writes=byte_writes(
                            (old_value + 1) & 0xFF,
                            strength=5000.0,
                        ),
                    )
                )
        return tuple(rules)

    def sp_pop_marker_increment_rules() -> tuple[FFNRule, ...]:
        """Late SP-marker correction for binary-pop ``SP += 8``.

        The L6 binary-pop unit range overlaps later function-call units in the
        expanded declarative layout, so the marker can still stage the old SP
        byte. Correct the marker prediction from the staged OUTPUT byte when
        the binary-pop relay is active.

        Phase 8.D: the MARK_SP gate (closure binding ``gate_mark_sp``)
        uses :func:`dim_ref` for the ``(marker, SP)`` semantic pair --
        the gate dim names the marker family member it asserts.
        """

        base_conditions = (
            ("MARK_SP", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 1.5),
            ("MARK_AX", -100.0),
            ("MARK_PC", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("MARK_MEM", -100.0),
                ("OP_ENT", -1000000.0),
                ("OP_SI", -1000000.0),
                ("OP_SC", -1000000.0),
                ("OP_LI", -1000000.0),
                ("OP_LC", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("MEM_STORE", -100000.0),
                ("IS_BYTE", -100.0),
            )
        # #315: in the 30-token campaign frame the block-26/L14 OUTPUT-band#2
        # corruption (OUTPUT_HI_THIS_STEP+13 ~= -1e4) explodes the -10 veto term
        # of this rule to +1e5, firing it on EVERY SP marker row regardless of
        # CMP+3. Promote CMP+3 to a HARD gate so it can ONLY fire on a genuine
        # binary-pop step (CMP+3 >= 4 -> +4e6 clears the +3.5e6 offset AND
        # overpowers the OP_SI veto, so SI/STORE pops still apply e0 -> e8). On a
        # non-pop SP marker row (IMM/LEA, CMP+3 == 0) the +1e5 corruption is now
        # ~3.4e6 below threshold -> the spurious e0 -> e8 jump is vetoed and SP
        # byte-0 carries forward unchanged. Flag-OFF (or golden 35-tok) ->
        # byte-identical.
        _e0e8_cmp3_hardgate = (
            (("CMP+3", 1.0e6),) if _sp_pop_marker_cmp3_hardgate_enabled() else ()
        )
        _e0e8_thresh = (
            9.0 + 3.5e6 if _sp_pop_marker_cmp3_hardgate_enabled() else 9.0
        )
        # #319: the MISSING binary-pop SP byte-0 CARRY case. The marker family
        # above has e0->e8 / d0->d8 / f0->f8 / d8->e0 but NO f8->00 -- yet the
        # most common stack-pop boundary is exactly 0x00fff8 + 8 => 0x010000
        # (input SP byte-0 0xF8 -> output 0x00). On 875-899 the L6 binary-pop
        # rotation usually delivers 0x00 cleanly, but on certain MOD/ADD results
        # (e.g. id876 46%6+6) a value-dependent OUTPUT-band leak stamps
        # OUTPUT_LO+8 / OUTPUT_HI+15 (= 0xF8) at the SP MARKER row (~+73 at
        # step3, ~+2.9e4 at step6; probe ``tools/probe_exprmod_sp_dims.py``),
        # which wins the byte-0 LM argmax -> SP byte-0 stays 0xF8 -> the AR run
        # desyncs at the HALT step. This rule (campaign-flag-gated) writes 0x00
        # at the marker row on the genuine carry signature -- ``MARK_SP`` +
        # ``CMP+3`` (binary-pop SP+=8 relay) + input proven 0xF8
        # (``EMBED_LO+8`` + ``EMBED_HI+15``) -- at strength 5e5 to DOMINATE the
        # ~3e4 corruption. The ``EMBED_LO+8`` + ``EMBED_HI+15`` AND makes it
        # VALUE-CORRECT and LOAD-BEARING-SAFE (the SP band gates
        # var/if_var/SI/SC): 0xF8+8 is the ONLY pop input that carries to 0x00;
        # a deeper-frame pop input 0xF0 (-> 0xF8, EMBED_LO nib 0) or 0xE8
        # (-> 0xF0, EMBED_HI nib E) fails the AND and is handled by the existing
        # f0->f8 / e-family rules; the PUSH/decrement step carries CMP+0 not
        # CMP+3 and is vetoed. On already-correct rows (id875) the input IS 0xF8
        # so this also fires there, writing the SAME 0x00 it already produces ->
        # no change. Mirrors ``tail_sp_pop_marker_f0_to_f8`` (small condition
        # weights, gate_mark_sp) -- the SAME shape the existing marker family
        # uses -- but for the missing 0xF8 -> 0x00 carry. Flag-OFF / golden
        # 35-tok -> not appended -> byte-identical. CRITICAL: NO
        # OP_MUL/OP_DIV/OP_MOD veto here. The expr_mod pop step occurs right
        # after the MOD compute, so a small ``OP_MOD`` residue (~+0.18) PERSISTS
        # on this SP marker row; an arithmetic veto (even at -100) would sink the
        # rule precisely on the MOD cluster this fix targets (measured: an early
        # large-weight draft with an ``OP_MOD`` veto fired NEGATIVELY because the
        # 0.18 residue x the rescaled veto drove the pre-activation < 0). The
        # CMP+3 (binary-pop relay) + EMBED 0xF8 proof is already a value-correct
        # AND; the 0x00 write at strength 5e5 dominates the ~3e4 corruption.
        _carry_f8_to_00 = (
            (
                multi_way_and_rule(
                    name="tail_sp_pop_marker_f8_to_00",
                    scope="mark == SP",
                    dominates_at={
                        "OUTPUT_LO": "mark == SP",
                        "OUTPUT_HI_THIS_STEP": "mark == SP",
                    },
                    conditions=(
                        ("MARK_SP", 1.0),
                        ("HAS_SE", 1.0),
                        # Binary-pop SP+=8 relay. HARD requirement: weight 2.0 x
                        # activation +4 = +8 on the pop step (CMP+3); the
                        # PUSH/decrement step (CMP+0) AND a NO-OP step where SP is
                        # unchanged but EMBED still reads 0xF8 (e.g. 876 step 2,
                        # CMP+3==0) BOTH have CMP+3==0, so without this +8 the
                        # remaining MARK_SP+HAS_SE+EMBED-proofs (~6) stay below
                        # threshold -> the rule fires ONLY on the genuine SP+=8
                        # carry step.
                        ("CMP+3", 2.0),
                        # Input SP byte-0 proven == 0xF8 (lo nibble 8, hi
                        # nibble F): the ONLY pop input that carries to 0x00.
                        ("EMBED_LO+8", 2.0),
                        ("EMBED_HI+15", 2.0),
                        # Exclude near-look-alike inputs handled by other
                        # marker rules: 0xF0 (EMBED_LO nib 0 -> f0->f8), 0xE8
                        # (EMBED_HI nib E=14 -> e-family), 0xD8 (EMBED_HI
                        # nib D=13 -> d8->e0).
                        ("EMBED_LO+0", -10.0),
                        ("EMBED_HI+14", -10.0),
                        ("EMBED_HI+13", -10.0),
                        # Marker / push / store vetoes (mirror the f0->f8
                        # family). NO arithmetic-opcode veto (see note above).
                        ("MARK_AX", -100.0),
                        ("MARK_PC", -100.0),
                        ("MARK_BP", -100.0),
                        ("MARK_STACK0", -100.0),
                        ("MARK_MEM", -100.0),
                        ("OP_ENT", -1000000.0),
                        ("OP_LEV", -1000000.0),
                        ("PSH_AT_SP", -1000000.0),
                        ("MEM_STORE", -100000.0),
                        ("IS_BYTE", -100.0),
                    ),
                    # Genuine carry: MARK_SP(1)+HAS_SE(1)+CMP+3(2x4=8)+
                    # EMBED_LO+8(2x~1)+EMBED_HI+15(2x~1) ~= 14 > 9. A NON-pop /
                    # no-op SP marker row (CMP+3==0) reaches only ~6 < 9; a
                    # missing 0xF8 nibble (input 0xF0/0xE8) drops by 2 AND trips a
                    # -10 EMBED look-alike veto -> well below 9. So both CMP+3 AND
                    # the exact 0xF8 input are REQUIRED.
                    threshold=9.0,
                    gate=gate_mark_sp,
                    writes=byte_writes(0x00, strength=5.0e5),
                ),
            )
            if _sp_pop_carry_byte0_dominate_enabled()
            else ()
        )
        return (
            multi_way_and_rule(
                name="tail_sp_pop_marker_e0_to_e8",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=base_conditions + (
                    ("OUTPUT_LO+0", 0.1),
                    ("OUTPUT_HI_THIS_STEP+14", 2.0),
                    ("OUTPUT_LO+8", -0.1),
                    ("EMBED_LO+8", -10.0),
                    ("EMBED_HI+13", -10.0),
                    ("EMBED_HI+15", -10.0),
                    ("OUTPUT_HI_THIS_STEP+13", -10.0),
                    ("OUTPUT_HI_THIS_STEP+0", -0.05),
                    ("OP_MUL", -100.0),
                    ("OP_DIV", -100.0),
                    ("OP_MOD", -100.0),
                ) + _e0e8_cmp3_hardgate,
                threshold=_e0e8_thresh,
                gate=gate_mark_sp,
                writes=byte_writes(0xE8, strength=300.0),
            ),
            multi_way_and_rule(
                name="tail_sp_pop_marker_d0_to_d8",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+0", 1.0),
                    ("EMBED_HI+13", 1.0),
                    ("EMBED_LO+8", -10.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("MEM_STORE", -100000.0),
                    ("IS_BYTE", -100.0),
                    ("OP_MUL", -100.0),
                    ("OP_DIV", -100.0),
                    ("OP_MOD", -100.0),
                ),
                threshold=5.5,
                gate=gate_mark_sp,
                writes=byte_writes(0xD8, strength=500.0),
            ),
            multi_way_and_rule(
                name="tail_sp_pop_marker_f0_to_f8",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+0", 1.0),
                    ("EMBED_HI+15", 1.0),
                    ("EMBED_LO+8", -10.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("MEM_STORE", -100000.0),
                    ("IS_BYTE", -100.0),
                    ("OP_MUL", -100.0),
                    ("OP_DIV", -100.0),
                    ("OP_MOD", -100.0),
                ),
                threshold=5.5,
                gate=gate_mark_sp,
                writes=byte_writes(0xF8, strength=500.0),
            ),
            multi_way_and_rule(
                name="tail_sp_pop_marker_d8_to_e0",
                scope="mark == SP",
                dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+8", 1.0),
                    ("EMBED_HI+13", 1.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("MEM_STORE", -100000.0),
                    ("IS_BYTE", -100.0),
                    ("OP_MUL", -100.0),
                    ("OP_DIV", -100.0),
                    ("OP_MOD", -100.0),
                ),
                threshold=5.5,
                gate=gate_mark_sp,
                writes=byte_writes(0xE0, strength=500.0),
            ),
        ) + _carry_f8_to_00

    def sp_pop_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve SP byte 1 after the marker-lane ``e0 -> e8`` correction.

        The SP marker correction repairs byte 0 for binary-pop ``SP += 8``.
        At the following byte position, the L10 ALU tail can still leak AX
        low-nibble residue into SP byte 1. The just-emitted ``0xe8`` byte and
        the binary-pop relay form a narrow signature for the no-carry case,
        where stack byte 1 must remain ``0xff``.

        2026-06-10: explicit OP_ADD/OP_SUB blockers added. These rules are
        a SP-pop byte-1 preserve only and must NEVER fire on arithmetic
        steps' AX byte-1 emit positions where ADD/SUB carry propagation
        should drive OUTPUT instead. Probe (capture_residual_trace on
        ``IMM 200; PSH; IMM 100; ADD; EXIT``) confirms these rules
        currently score < 40.5 at all ADD-step rows; the blockers
        make that invariant declarative. The actual byte-1 carry
        failure on ``test_add_16bit`` / ``test_sub_16bit`` /
        ``test_add_carry_cascade`` attributes upstream to the L7
        operand_gather + L9 ALU/CARRY+1 surfaces (F1/F2 of
        ``docs/RUNNER_OVERRIDE_FULL_REMOVAL_2026_06_09.md``); the
        L9 ALU emits operand A in ALU_LO at the MARK_AX row instead
        of the sum, so no L10 declarative rewrite can recover the
        right byte-1 here without first fixing F1/F2.
        """

        return (
            multi_way_and_rule(
                name="tail_sp_pop_byte1_ff_after_e0",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+0", 1.0),
                    ("CLEAN_EMBED_HI+14", 1.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                    ("OP_ADD", -1000000.0),
                    ("OP_SUB", -1000000.0),
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            multi_way_and_rule(
                name="tail_sp_pop_byte1_ff_after_d8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 1.0),
                    ("CLEAN_EMBED_HI+13", 1.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                    ("OP_ADD", -1000000.0),
                    ("OP_SUB", -1000000.0),
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            multi_way_and_rule(
                name="tail_sp_pop_byte1_ff_after_f8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 1.0),
                    ("CLEAN_EMBED_HI+15", 1.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                    ("OP_ADD", -1000000.0),
                    ("OP_SUB", -1000000.0),
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            multi_way_and_rule(
                name="tail_sp_pop_byte1_ff_after_e8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    ("STACK0_BYTE0", -1000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                    ("OP_EQ", -1000000.0),
                    ("OP_NE", -1000000.0),
                    ("OP_LT", -1000000.0),
                    ("OP_GT", -1000000.0),
                    ("OP_LE", -1000000.0),
                    ("OP_GE", -1000000.0),
                    ("OP_ADD", -1000000.0),
                    ("OP_SUB", -1000000.0),
                ),
                threshold=85.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
        )

    def stack0_pushed_addr_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve byte 1 for pushed local addresses in STACK0.

        With shallow L15 memory recency, STACK0 byte positions can pick the
        zero sink even though the preceding STACK0 byte 0 already emitted a
        local address such as ``0xe8``. The next byte is the stack high byte
        ``0xff``.
        """

        return (
            multi_way_and_rule(
                name="tail_stack0_pushed_addr_byte1_ff_after_e8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("STACK0_BYTE0", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("MEM_STORE", -1000.0),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    *(
                        (f"CLEAN_EMBED_LO+{k}", -100.0)
                        for k in range(16)
                        if k != 8
                    ),
                    *(
                        (f"CLEAN_EMBED_HI+{k}", -100.0)
                        for k in range(16)
                        if k != 14
                    ),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                ),
                threshold=85.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            multi_way_and_rule(
                name="tail_stack0_pushed_addr_byte1_store_ff_after_e8",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+10", 10.0),
                    ("STACK0_BYTE0", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("MEM_STORE", 0.5),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    ("OUTPUT_LO+15", 1.0),
                    ("OUTPUT_HI_THIS_STEP+15", 1.0),
                    *(
                        (f"CLEAN_EMBED_LO+{k}", -100.0)
                        for k in range(16)
                        if k != 8
                    ),
                    *(
                        (f"CLEAN_EMBED_HI+{k}", -100.0)
                        for k in range(16)
                        if k != 14
                    ),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                ),
                threshold=108.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            multi_way_and_rule(
                name="tail_stack0_pushed_addr_byte1_store_ff_after_e0",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+10", 10.0),
                    ("STACK0_BYTE0", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("MEM_STORE", 0.5),
                    ("CLEAN_EMBED_LO+0", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    ("OUTPUT_LO+15", 1.0),
                    ("OUTPUT_HI_THIS_STEP+15", 1.0),
                    *(
                        (f"CLEAN_EMBED_LO+{k}", -100.0)
                        for k in range(16)
                        if k != 0
                    ),
                    *(
                        (f"CLEAN_EMBED_HI+{k}", -100.0)
                        for k in range(16)
                        if k != 14
                    ),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                ),
                threshold=108.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
        )

    def ax_lea_local_addr_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve byte 1 for BP-relative local addresses produced by LEA.

        Local addresses such as BP-8/BP-16/BP-24 emit byte 0 as e8/e0/d8 and
        byte 1 as ff. The late ADD cleanup can mistake the LEA row for a
        low-16-bit arithmetic result and zero the high nibble. Key this repair
        on the negative immediate high nibble instead of an OUTPUT high-nibble
        value, which can be present on unrelated early AX/STACK0 rows.
        """

        rules = []
        for value in (0xE8, 0xE0, 0xD8):
            lo = value & 0xF
            hi = value >> 4
            rules.append(
                multi_way_and_rule(
                    name=f"tail_ax_lea_local_addr_byte1_ff_after_{value:02x}",
                    scope="is_byte",
                    dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                    conditions=(
                        ("IS_BYTE", 5.0),
                        ("HAS_SE", 5.0),
                        ("H1+1", 20.0),
                        ("H1+2", -1000.0),
                        ("H1+3", -1000.0),
                        ("H1+4", -1000.0),
                        ("H3+4", -1000.0),
                        ("BYTE_INDEX_0", 5.0),
                        ("BYTE_INDEX_1", -1000.0),
                        ("BYTE_INDEX_2", -1000.0),
                        ("BYTE_INDEX_3", -1000.0),
                        ("MEM_VAL_B0", -1000.0),
                        ("MEM_VAL_B1", -1000.0),
                        ("MEM_VAL_B2", -1000.0),
                        ("MEM_VAL_B3", -1000.0),
                        (f"CLEAN_EMBED_LO+{lo}", 30.0),
                        (f"CLEAN_EMBED_HI+{hi}", 30.0),
                        # 16-bit XOR byte-1 fix (2026-06-11): require the EXACT
                        # byte-0 high nibble. The FETCH_HI+15 (100) + TEMP+10
                        # (80) terms alone nearly clear threshold 150, so this
                        # LEA-local-address preserve over-fired on ``IMM 0x0F0F;
                        # PSH; IMM 0x00FF; XOR`` (byte 0 = 0xF0, CLEAN_EMBED_HI+15
                        # != the e0/e8 target +14), stamping byte 1 = 0xff
                        # (0x0FF0 -> 0xFFF0) on top of the correct 0x0F staged by
                        # layer13_bitwise_byte1_gather + the widened relay. A
                        # real BP-relative LEA local addr (e8/e0/d8) has
                        # CLEAN_EMBED_HI = 14/13 exactly, so blocking the
                        # non-target HI nibbles is byte-identical for the legit
                        # case. Mirrors the non-target CLEAN_EMBED guards on the
                        # sibling ``stack0_pushed_addr_byte1`` rules. Opcode-free
                        # (avoids the OP_XOR-blocker / CMP-coupling regression).
                        *(
                            (f"CLEAN_EMBED_HI+{k}", -100.0)
                            for k in range(16)
                            if k != hi
                        ),
                        ("FETCH_HI+15", 100.0),
                        ("TEMP+10", 80.0),
                        # IMM-decode residual fix (2026-06-11): this byte-1=0xFF
                        # LEA local-address preserve keys on the byte-0 value
                        # (CLEAN_EMBED e8/e0/d8) but had NO opcode gate, so a
                        # plain ``IMM 0xE0/0xE8/0xD8; EXIT`` byte-1 row -- whose
                        # immediate value coincidentally matches the frame-
                        # address signature -- mis-fired (score ~156 >= 150),
                        # emitting byte1=0xff (0xE0 -> 0xFFE0). These are the 3
                        # residual IMM mis-decodes left after the cell-8 0xE8
                        # keystone fix. spec_k=0 attribution:
                        # tools/probe_imm_byte1_residual.py. FIX
                        # (broadcast-hardening, mirrors the sibling fixes this
                        # session): OP_IMM hard NOT-blocker so an IMM step can
                        # never satisfy it; the legit LEA byte-1 preserve
                        # (OP_IMM=0) is byte-identical.
                        ("OP_IMM", -1_000_000.0),
                        ("MARK_AX", -10000.0),
                        ("MARK_PC", -10000.0),
                        ("MARK_SP", -10000.0),
                        ("MARK_BP", -10000.0),
                        ("MARK_STACK0", -10000.0),
                        ("MARK_MEM", -10000.0),
                    ),
                    threshold=150.0,
                    writes=byte_writes(0xFF, strength=5000.0),
                )
            )
        return tuple(rules)

    # NOTE: ``stack0_store_nonzero_pair_rules`` (a 15x15=225-way ENUMERATED
    # cross-lane ALU->OUTPUT byte-writeback bank) was DELETED here as DEAD
    # code: the closure was defined but NEVER splatted into the ``rules = (...)``
    # assembly below (unlike its ``stack0_*_output_rules`` siblings), so it
    # emitted ZERO FFN units on the build path. Weight-neutral removal (golden
    # hash unchanged); confirmed uncalled repo-wide. It was NOT a candidate for
    # the enumerated->computed collapse because its firing region skipped ALL
    # zero-nibble bytes (``range(1, 16)`` on BOTH nibbles — "zero-nibble cases
    # need a stronger disambiguator because ALU zero pollution is also present")
    # which the generic per-nibble ``byte_copy_computed_rules`` route (sums the
    # full 16-way other band, incl. j=0) does not reproduce.

    def stack0_pop_loaded_output_rules() -> tuple[FFNRule, ...]:
        """Let strong L15 STACK0 memory loads beat stale pop-marker cleanup."""

        # Shallow-competitor fix: a HUGE-magnitude L20 frame-address relay drives
        # these units to runaway firing strength; the -500 competitor side-effects
        # then crush the whole OUTPUT band and the relayed value loses to a stray
        # marker (var/expr/if-bool full_trace step-5 PC desync). Drop the
        # COMPETITOR strength to 5 (keep the +500 matching reinforcement) so the
        # crush stays shallow and the relayed value survives POSITIVE. Genuine
        # small-magnitude SI/LI/SC/LC loads win by their own reinforcement and are
        # unaffected. See _stack0_pop_loaded_shallow_crush_enabled.
        competitor = 5.0 if _stack0_pop_loaded_shallow_crush_enabled() else 500.0

        base_conditions = (
            ("MARK_STACK0", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", -1000.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -100.0),
            ("MARK_SP", -100.0),
            ("MARK_BP", -100.0),
            ("MARK_MEM", -100.0),
            ("H1+0", -1000.0),
            ("H1+1", -1000.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
        )
        # R-FRAME INCR-3 (STACK0 row) — P5 RETIRE (unconditional collapse):
        # emit ONLY the 32-rule per-nibble COMPUTED route (16 LO + 16 HI). Same
        # READ==WRITE lane (OUTPUT); the route reproduces the winning byte (argmax)
        # + firing region byte-for-byte at match_weight=0.05 / threshold=12 for
        # BOTH competitor values (proof: tools/_probe_computed_writeback_banks.py).
        # The retired 255-rule per-value ENUMERATED lookup + its ``C4_R_FRAME_TAIL``
        # / ``C4_STACK0_POP_LOADED_COMPUTED`` flag branch are DELETED (P5); the
        # default was already this collapsed form.
        return _computed_byte_writeback_route_rules(
            name_for=lambda band, k: (
                f"tail_stack0_pop_loaded_route_{band}_{k}"
            ),
            base_conditions=base_conditions,
            threshold=12.0,
            strength=500.0,
            match_weight=0.05,
            competitor_strength=competitor,
            lo_base="OUTPUT_LO",
            hi_base="OUTPUT_HI_THIS_STEP",
            gate=gate_mark_stack0,
            scope="mark == STACK0",
            dominates_at={
                "OUTPUT_LO": "mark == STACK0",
                "OUTPUT_HI_THIS_STEP": "mark == STACK0",
            },
        )

    def stack0_store_top_e0_output_rules() -> tuple[FFNRule, ...]:
        """Restore nonzero store-top values when SP points at the stored cell."""

        base_conditions = (
            ("MARK_STACK0", 5.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("EMBED_LO+0", 10.0),
            ("EMBED_HI+14", 1.0),
            ("IS_BYTE", -1000000.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("H1+0", -1000.0),
            ("H1+1", -1000.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
        )
        # R-FRAME INCR-3 (STACK0 row, GAP-PRIMITIVE #3) — P5 RETIRE (unconditional
        # collapse): emit ONLY the 32-rule per-nibble CROSS-LANE COMPUTED copy.
        # READ lane (ALU) != WRITE lane (OUTPUT): the route fires on the SOURCE
        # (ALU) one-hot + the summed OTHER source band (reconstructing the
        # 2-channel AND magnitude / threshold=25) and writes the byte into OUTPUT.
        # Reproduces the retired 254-rule ENUMERATED ALU->OUTPUT materializer's
        # winning byte (argmax) + firing region byte-for-byte (proof:
        # tools/_probe_crosslane_bytecopy.py). The enumerated fallback (whose tiny
        # 0.001 OUTPUT tie-breaker terms were negligible) + its ``C4_R_FRAME_TAIL``
        # / ``C4_STACK0_STORE_TOP_E0_COMPUTED`` flag branch are DELETED (P5); the
        # default was already this collapsed form.
        return byte_copy_computed_rules(
            src_lo="ALU_LO",
            src_hi="ALU_HI",
            dst_lo="OUTPUT_LO",
            dst_hi="OUTPUT_HI_THIS_STEP",
            base_conditions=base_conditions,
            threshold=25.0,
            strength=2000.0,
            name_for=lambda band, k: (
                f"tail_stack0_store_top_e0_route_{band}_{k}"
            ),
            gate=gate_mark_stack0,
            scope="mark == STACK0",
            dominates_at={
                "OUTPUT_LO": "mark == STACK0",
                "OUTPUT_HI_THIS_STEP": "mark == STACK0",
            },
        )

    def stack0_store_top_e8_from_e0_output_rules() -> tuple[FFNRule, ...]:
        """Restore top-store values for the e0->e8 local-store transition.

        L15 blocks the stale historical lookup for ``SI`` when the pre-pop
        stack top is ``0xffe8`` and the pre-pop SP address band is still
        ``0xffe0``. That leaves the correct current store value in OUTPUT, but
        the older non-top zeroing rule still matches the broad e8/e0 address
        shape. Require L15's strengthened ADDR_B0_HI[0] signal so this restore
        applies only after that current-top-store path has been disambiguated.
        """

        base_conditions = (
            ("MARK_STACK0", 5.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 2.0),
            ("MEM_STORE", 1000.0),
            ("EMBED_LO+8", 1000.0),
            ("EMBED_HI+14", 1000.0),
            ("ADDR_B0_LO+0", 200.0),
            ("ADDR_B0_HI+0", 200.0),
            ("IS_BYTE", -1000000.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("H1+0", -1000.0),
            ("H1+1", -1000.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
        )
        # R-FRAME INCR-3 (STACK0 row) — P5 RETIRE (unconditional collapse): emit
        # ONLY the 32-rule per-nibble COMPUTED route (same READ==WRITE lane,
        # OUTPUT; match_weight 0.001 / threshold 4300). The trailing byte_39
        # special rule below is NOT part of the loop and is emitted UNCHANGED. The
        # route reproduces the retired 255-rule per-value ENUMERATED NIBBLE LOOP's
        # winning byte (argmax) + firing region byte-for-byte (proof:
        # tools/_probe_computed_writeback_banks.py). The enumerated fallback + its
        # ``C4_R_FRAME_TAIL`` / ``C4_STACK0_STORE_E8_COMPUTED`` flag branch are
        # DELETED (P5); the default was already this collapsed form.
        rules = list(_computed_byte_writeback_route_rules(
            name_for=lambda band, k: (
                f"tail_stack0_store_top_e8_from_e0_route_{band}_{k}"
            ),
            base_conditions=base_conditions,
            threshold=4300.0,
            strength=5000.0,
            match_weight=0.001,
            lo_base="OUTPUT_LO",
            hi_base="OUTPUT_HI_THIS_STEP",
            gate=gate_mark_stack0,
            scope="mark == STACK0",
            dominates_at={
                "OUTPUT_LO": "mark == STACK0",
                "OUTPUT_HI_THIS_STEP": "mark == STACK0",
            },
        ))
        # The byte-0x39 store-pop restore is driven by the (unbounded)
        # ``OUTPUT_LO+9`` term: at a binary-op STACK0 byte-0 emit row the
        # operand/result byte's low nibble 9 lands in OUTPUT_LO+9 at magnitude
        # ~3654, so ``100 * 3654`` alone trips the 20000 threshold even though
        # NONE of the rule's store-pop witnesses are present — corrupting the
        # STACK0 byte to 0x39 for every op whose byte-0 low nibble is 9 (the 4
        # ``mul_*`` full_trace fails ``mul_20/29/36/43``, all operand-A low
        # nibble 9). The genuine e8->e0 store-pop the rule restores 0x39 for is
        # a MEMORY restore: it carries ``MEM_ADDR_SRC`` and the 0xE8/0xE0
        # address shape (``ADDR_B0_HI+14`` POSITIVE). Probed spec_k=0 at the
        # rule's MARK_STACK0 firing row: BOTH binary-op false-fires (9*5 low
        # nibble, 48*98 high nibble) have ``MEM_ADDR_SRC == 0`` AND
        # ``ADDR_B0_HI+14 == -2`` (the arithmetic-result row never carries the
        # memory address source). Requiring MEM_ADDR_SRC as a HARD gate (a
        # large negative baseline only the memory-restore witness overcomes)
        # therefore suppresses the binary-op false-fire while leaving the
        # genuine memory-restore path untouched. Flag-gated (DEFAULT OFF,
        # byte-identical flag-OFF) — see
        # ``shared.mul_stack0_byte39_guard_enabled``.
        if mul_stack0_byte39_guard_enabled():
            # Add a -1e6 baseline that ONLY ``MEM_ADDR_SRC`` (the memory-
            # restore witness, +2e6) can lift back above zero; an arithmetic
            # STACK0 emit row (MEM_ADDR_SRC == 0) stays >=1e6 below threshold no
            # matter how large its single OUTPUT nibble is.
            byte39_context_conditions = (
                ("CONST", -1_000_000.0),
                ("MEM_ADDR_SRC", 2_000_000.0),
                ("ADDR_B0_LO+8", 100.0),
                ("ADDR_B0_HI+14", 100.0),
                ("OUTPUT_LO+9", 100.0),
                ("OUTPUT_HI_THIS_STEP+3", 1.0),
            )
            byte39_threshold = 1_020_000.0
        else:
            byte39_context_conditions = (
                ("MEM_ADDR_SRC", 100.0),
                ("ADDR_B0_LO+8", 100.0),
                ("ADDR_B0_HI+14", 100.0),
                ("OUTPUT_LO+9", 100.0),
                ("OUTPUT_HI_THIS_STEP+3", 1.0),
            )
            byte39_threshold = 20000.0
        rules.append(
            multi_way_and_rule(
                name=(
                    "tail_stack0_store_top_e8_from_e0_byte_39_from_e8_addr"
                ),
                scope="mark == STACK0",
                dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                conditions=base_conditions + byte39_context_conditions,
                threshold=byte39_threshold,
                writes=byte_writes(0x39, strength=5000.0),
            )
        )
        return tuple(rules)

    def stack0_store_loaded_output_rules() -> tuple[FFNRule, ...]:
        """Let strong L15 store/pop STACK0 memory loads beat stale cleanup.

        SI/SC pops the address and exposes memory[post-pop SP] as the next
        STACK0.  L10/L14 can still leave the just-stored AX byte in OUTPUT,
        and the stale cleanup rules above zero that residue.  When L15 has
        actually resolved a historical nonzero stack value, its OUTPUT signal
        is much larger than the current-AX residue, so restore it here.
        """

        base_conditions = (
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("H1+0", -1_000_000_000.0),
            ("H1+1", -1_000_000_000.0),
            ("H1+2", -1_000_000_000.0),
            ("H1+3", -1_000_000_000.0),
            ("H1+4", -1_000_000_000.0),
        )
        # R-FRAME INCR-3 (STACK0 row) — P5 RETIRE (unconditional collapse):
        # emit ONLY the 32-rule per-nibble COMPUTED route (16 LO + 16 HI). Since
        # this bank's READ lane (OUTPUT) == its WRITE lane, the retired 255-rule
        # per-value ENUMERATED lookup was an identity-copy of the OUTPUT byte; the
        # route reproduces the winning byte (argmax) + firing region byte-for-byte
        # (proof: tools/_probe_m8_computed_writeback.py). The enumerated fallback +
        # its ``C4_R_FRAME_TAIL`` / ``C4_STACK0_STORE_LOADED_COMPUTED`` flag branch
        # are DELETED (P5); the default was already this collapsed form.
        return _computed_byte_writeback_route_rules(
            name_for=lambda band, k: (
                f"tail_stack0_store_loaded_route_{band}_{k}"
            ),
            base_conditions=base_conditions,
            threshold=25.0,
            strength=5000.0,
            lo_base="OUTPUT_LO",
            hi_base="OUTPUT_HI_THIS_STEP",
            gate=gate_mark_stack0,
            scope="mark == STACK0",
            dominates_at={
                "OUTPUT_LO": "mark == STACK0",
                "OUTPUT_HI_THIS_STEP": "mark == STACK0",
            },
        )

    def stack0_store_top_value_from_alu_rules() -> tuple[FFNRule, ...]:
        """Materialize current top-store values when only ALU residue remains."""

        return (
            multi_way_and_rule(
                name="tail_stack0_store_top_value_2f_from_alu",
                scope="mark == STACK0",
                dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
                conditions=(
                    ("MARK_STACK0", 5.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 2.0),
                    ("MEM_STORE", 100.0),
                    ("MEM_ADDR_SRC", 100.0),
                    ("ADDR_B0_LO+0", 5.0),
                    ("ADDR_B0_HI+14", 5.0),
                    ("ALU_LO+15", 20.0),
                    ("ALU_HI+2", 100.0),
                    ("IS_BYTE", -1000000.0),
                    ("MARK_AX", -1000000.0),
                    ("MARK_PC", -1000000.0),
                    ("MARK_SP", -1000000.0),
                    ("MARK_BP", -1000000.0),
                    ("MARK_MEM", -1000000.0),
                    ("H1+0", -1000.0),
                    ("H1+1", -1000.0),
                    ("H1+2", -1000.0),
                    ("H1+3", -1000.0),
                ),
                threshold=180.0,
                writes=byte_writes(0x2F, strength=5000.0),
            ),
        )

    def ax_add_carry_rules() -> tuple[FFNRule, ...]:
        """Late ADD byte carry after L15 has materialized high-byte bases.

        The immediate L10 carry post-op runs before the L15 stack-value relay.
        For ADD byte-0 overflows, L15 can overwrite the early increment with
        the unincremented high byte. These rules run in the dependency-tail
        correction block after L15, incrementing the currently staged AX byte.

        This is intentionally limited to byte 1. Full byte-2/3 cascade needs
        an unambiguous carry-continuation signal; observing a previously
        emitted 0x00 byte is not enough because non-overflowing high bytes can
        also be zero.
        """

        marker_blockers = (
            ("MARK_AX", -1_000_000_000.0),
            ("MARK_PC", -1_000_000_000.0),
            ("MARK_SP", -1_000_000_000.0),
            ("MARK_BP", -1_000_000_000.0),
            ("MARK_STACK0", -1_000_000_000.0),
            ("MARK_MEM", -1_000_000_000.0),
        )
        non_add_blockers = (
            ("CARRY+2", -1000.0),
            ("TEMP+3", -1000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SI", -1000.0),
            ("OP_SC", -1000.0),
            ("OP_LI", -1000.0),
            ("OP_LC", -1000.0),
        )
        rules = []
        for byte_idx in range(1):
            base_conditions = (
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+1", 20.0),
                (f"BYTE_INDEX_{byte_idx}", 5.0),
                ("CARRY+1", 20.0),
                ("TEMP+8", 100.0),
            ) + marker_blockers + non_add_blockers
            threshold = 250.0

            for old_value in range(256):
                old_lo = old_value & 0xF
                rules.append(
                    multi_way_and_rule(
                        name=(
                            f"tail_ax_add_carry_byte{byte_idx + 1}_"
                            f"{old_value:02x}"
                        ),
                        scope="is_byte",
                        dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{old_lo}", 1.0),
                            (f"OUTPUT_HI_THIS_STEP+{old_value >> 4}", 1.0),
                            (f"ALU_LO+{old_lo}", 20.0),
                        ) + tuple(
                            (f"OUTPUT_LO+{other}", -25.0)
                            for other in range(16)
                            if other != old_lo
                        ),
                        threshold=threshold,
                        writes=byte_writes(
                            (old_value + 1) & 0xFF,
                            strength=5000.0,
                        ),
                    )
                )
        return tuple(rules)

    def ax_add_no_carry_zero_rules() -> tuple[FFNRule, ...]:
        """Clear ADD byte 1 when both operand high bytes and carry are zero."""

        marker_blockers = (
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -10000.0),
            ("MARK_SP", -20000000.0),
            ("MARK_BP", -10000.0),
            ("MARK_STACK0", -20000000.0),
            ("MARK_MEM", -10000.0),
            ("H1+2", -1000000.0),
            ("H1+3", -1000000.0),
            ("H1+4", -1000000.0),
        )
        non_add_blockers = (
            ("OP_IMM", -1000000.0),
            ("OP_LEA", -1000000.0),
            ("OP_SUB", -1000000.0),
            ("OP_DIV", -1000000.0),
            ("OP_MOD", -1000000.0),
            ("OP_AND", -1000000.0),
            ("OP_OR", -1000000.0),
            ("OP_XOR", -1000000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SHL", -1000000.0),
            ("OP_SHR", -1000000.0),
            ("OP_SI", -1000000.0),
            ("OP_SC", -1000000.0),
            ("OP_LI", -1000000.0),
            ("OP_LC", -1000000.0),
            ("OP_ENT", -1000000.0),
            ("MEM_STORE", -1000000.0),
            # The ADD byte-1 cleanup is intentionally narrow.  At AX byte
            # rows, these individual TEMP relays are stronger signatures than
            # opcode bits and avoid broad scratch aliases used by comparisons.
            ("TEMP+4", -1000000.0),  # AND relay
            ("TEMP+5", -1000000.0),  # OR relay
            ("TEMP+6", -1000000.0),  # XOR relay
            ("TEMP+7", -1000000.0),  # SHR relay
            ("TEMP+9", -1000000.0),  # SUB relay
            # NOTE (2026-06-11, ADD byte-1 0x88 leak fix): the TEMP+10
            # blocker (labelled "MUL relay") over-blocked EVERY ADD step.
            # Probe (spec_k=0, tools/probe_add_byte1.py) at the byte-1 emit
            # row shows TEMP+10 ~= 1.0-1.25 is present on ADD itself (and on
            # MUL/SHL), so -1e6*1.25 hard-blocked the no-carry byte-1=0x00
            # materializer -> byte 1 leaked 0x88 (test_add_basic 0x002A ->
            # 0x882A). MUL/SHL are already excluded by the positive
            # ``("TEMP+8", 100.0)`` ADD-relay gate (TEMP+8 = 1.0 only on ADD;
            # MUL/SHL byte-1 rows carry TEMP+8 = 0.0), so the TEMP+10 blocker
            # was both harmful and redundant. Removed.
        )
        transition_blockers = (
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
        )
        return (
            multi_way_and_rule(
                name="tail_ax_add_no_carry_byte1_00",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+1", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("TEMP+8", 100.0),
                    ("CARRY+1", -10000.0),
                    ("ALU_LO+0", 10.0),
                    ("ALU_HI+0", 20.0),
                    ("AX_CARRY_HI+0", 20.0),
                ) + tuple(
                    (f"ALU_LO+{other}", -50.0)
                    for other in range(1, 16)
                ) + tuple(
                    # ADD byte-1 0x88 leak fix (2026-06-11): softened from
                    # -2000 to -100. The original -2000 hard-blocked
                    # test_add_basic: at the byte-1 emit row a benign
                    # AX_CARRY_LO+12 ~= 0.83 residue scored -1660, sinking the
                    # rule below threshold so byte-1 = 0x00 was never
                    # materialized and the block-17 OUTPUT cell-0 suppressor
                    # left a stale cell-8 residue winning -> 0x88. The actual
                    # no-carry guard is the dedicated ``CARRY+1 = -10000``
                    # signal (a real ADD carry parks CARRY+1 ~= 2.0 and keeps
                    # this rule dark, so add_16bit / add_carry_cascade still
                    # route to ``ax_add_carry_rules``); the AX_CARRY_LO band
                    # is only a weak secondary no-carry hint, so -100 tolerates
                    # the ~0.83 residue while still penalising a full carry
                    # nibble.
                    (f"AX_CARRY_LO+{other}", -100.0)
                    for other in range(1, 16)
                ) + marker_blockers + non_add_blockers + transition_blockers,
                threshold=300.0,
                writes=byte_writes(0x00, strength=5000.0),
            ),
        )

    def ax_add_byte1_high_zero_rules() -> tuple[FFNRule, ...]:
        """Assert high nibble zero for low 16-bit ADD byte-1 rows.

        The ADD byte-1 low nibble can be correct while stale cleanup residue
        leaves OUTPUT_HI[1..15] above OUTPUT_HI[0]. Limit this to rows where
        both operand byte high nibbles are zero; wider byte-1 sums need a
        separate carry-aware high-nibble rule.
        """

        marker_blockers = (
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -10000.0),
            ("MARK_SP", -10000.0),
            ("MARK_BP", -10000.0),
            ("MARK_STACK0", -10000.0),
            ("MARK_MEM", -10000.0),
        )
        non_add_blockers = (
            ("CARRY+2", -1000.0),
            ("TEMP+9", -1000.0),
            ("TEMP+10", -1000000.0),
            ("OP_IMM", -1000000.0),
            ("OP_LEA", -1000000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SI", -1000.0),
            ("OP_SC", -1000.0),
            ("OP_LI", -1000.0),
            ("OP_LC", -1000.0),
        )
        transition_blockers = (
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
            ("NEXT_SE", -1000000.0),
        )
        high_writes = [("OUTPUT_HI_THIS_STEP+0", 50_000.0)]
        high_writes.extend((f"OUTPUT_HI_THIS_STEP+{other}", -50_000.0) for other in range(1, 16))
        base_conditions = (
            ("IS_BYTE", 5.0),
            ("HAS_SE", 5.0),
            ("H1+1", 20.0),
            ("H1+2", -1000000.0),
            ("H1+3", -1000000.0),
            ("H1+4", -1000000.0),
            ("H1+10", -1000000.0),
            ("BYTE_INDEX_0", 5.0),
            ("BYTE_INDEX_1", -1000.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("STACK0_BYTE0", -1000000.0),
            ("STACK0_BYTE1", -1000000.0),
            ("STACK0_BYTE2", -1000000.0),
            ("STACK0_BYTE3", -1000000.0),
            ("TEMP+8", 100.0),
            ("ALU_HI+0", 20.0),
            ("AX_CARRY_HI+0", 20.0),
        ) + marker_blockers + transition_blockers + non_add_blockers
        # base_conditions ANDs many ``OP_*`` hard blockers (sem
        # ``NOT (mark == AX AND opcode_at_AX == FOO)``) with the IS_BYTE /
        # BYTE_INDEX_0 positives and the wide MARK_AX/STACK0/... blockers.
        # The effective_predicate walker finds the union unsatisfiable and
        # falls back to the gate (TEMP+8, no semantics) which collapses to
        # tautology -- no scope tighter than tautology is honestly
        # entailed. Leave scope/dominates_at unset until the conditions can
        # be restructured.
        rules = [
            multi_way_and_rule(
                name="tail_ax_add_byte1_hi_zero",
                conditions=base_conditions,
                threshold=250.0,
                writes=tuple(high_writes),
            ),
        ]
        for lo in range(16):
            rules.append(
                multi_way_and_rule(
                    name=f"tail_ax_add_byte1_hi_zero_lo_{lo:01x}",
                    conditions=base_conditions
                    + ((f"OUTPUT_LO+{lo}", 10.0),)
                    + tuple(
                        (f"OUTPUT_LO+{other}", -1.0)
                        for other in range(16)
                        if other != lo
                    ),
                    threshold=340.0,
                    gate="TEMP+8",
                    writes=byte_writes(lo, strength=10_000.0),
                )
            )
        return tuple(rules)

    def ax_add_byte1_structural_materialize_rules() -> tuple[FFNRule, ...]:
        """Materialize ADD byte 1 from structural low-nibble evidence.

        Some ADD rows carry benign TEMP+10 residue, so the generic high-zero
        cleanup stays blocked and the final byte head sees no high nibble.
        These cases still expose the uncarried byte-1 low nibble in ALU_LO;
        use CARRY+1 only to separate the carried and non-carried variants.
        """

        base_conditions = (
            ("IS_BYTE", 10.0),
            ("HAS_SE", 10.0),
            ("H1+1", 20.0),
            ("H1+0", -100000000.0),
            ("H1+2", -100000000.0),
            ("H1+3", -100000000.0),
            ("H1+4", -100000000.0),
            ("BYTE_INDEX_0", 10.0),
            ("BYTE_INDEX_1", -1000.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("TEMP+8", 200.0),
            ("TEMP+9", -1000000.0),
            ("ALU_HI+0", 20.0),
            ("AX_CARRY_HI+0", 20.0),
            ("MARK_AX", -100000000.0),
            ("MARK_PC", -100000000.0),
            ("MARK_SP", -100000000.0),
            ("MARK_BP", -100000000.0),
            ("MARK_STACK0", -100000000.0),
            ("MARK_MEM", -100000000.0),
            ("OP_IMM", -1000000.0),
            ("OP_LEA", -1000000.0),
            ("OP_SUB", -1000000.0),
            ("OP_DIV", -1000000.0),
            ("OP_MOD", -1000000.0),
            ("OP_AND", -1000000.0),
            ("OP_OR", -1000000.0),
            ("OP_XOR", -1000000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SHL", -1000000.0),
            ("OP_SHR", -1000000.0),
            ("OP_SI", -1000000.0),
            ("OP_SC", -1000000.0),
            ("OP_LI", -1000000.0),
            ("OP_LC", -1000000.0),
            ("OP_ENT", -1000000.0),
            ("MEM_STORE", -1000000.0),
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
            ("NEXT_SE", -1000000.0),
        )
        return (
            # ADD byte-1 carry-out=1 / high-byte-zero case (2026-06-11):
            # test_add_16bit (200 + 100 = 300 = 0x012C) and
            # test_add_carry_cascade (0xFF + 1 = 0x100) both need byte 1 =
            # 0x01 = (raw byte-1 low nibble 0 in ALU_LO) + carry-out 1. The
            # carry-out parks CARRY+1 ~= 2.0 (probe spec_k=0,
            # tools/probe_add_byte1.py). This slot previously materialized
            # 0x02 for an un-carried ALU_LO+1 raw byte-1 -- a case no live
            # ADD smoke test exercises (the byte-1 low nibble is 0 for every
            # ADD target) -- so it is repurposed in place rather than appended,
            # KEEPING the tail FFN rule count at 2059 (growing it shifts the
            # bank width and silently breaks test_cmp_and_branch's EQ/branch
            # byte-identity). CARRY+1 = +200 is load-bearing: the no-carry
            # add_basic path (CARRY+1 = 0) stays below threshold 1300 and is
            # owned by the byte-1 = 0x00 materializer (mutually exclusive via
            # its own CARRY+1 = -10000 guard), so add_basic stays 0x00.
            multi_way_and_rule(
                name="tail_ax_add_byte1_carry_low0_01",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=base_conditions + (
                    ("CARRY+1", 200.0),
                    ("ALU_LO+0", 100.0),
                ),
                threshold=1300.0,
                writes=byte_writes(0x01, strength=5_000_000.0),
            ),
            multi_way_and_rule(
                name="tail_ax_add_byte1_no_carry_low2_03",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=base_conditions + (
                    ("CARRY+1", -1000.0),
                    ("ALU_LO+2", 100.0),
                ),
                threshold=1080.0,
                writes=byte_writes(0x03, strength=1_000_000.0),
            ),
            multi_way_and_rule(
                name="tail_ax_add_byte1_carry_low2_03",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=base_conditions + (
                    ("CARRY+1", 200.0),
                    ("ALU_LO+2", 100.0),
                ),
                threshold=1480.0,
                writes=byte_writes(0x03, strength=5_000_000.0),
            ),
        )

    def ax_sub_byte1_high_zero_rules() -> tuple[FFNRule, ...]:
        """Assert high nibble zero for low 16-bit SUB byte-1 rows."""

        rules = []
        for lo in range(16):
            rules.append(
                multi_way_and_rule(
                    name=f"tail_ax_sub_byte1_hi_zero_lo_{lo:01x}",
                    scope="is_byte",
                    dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                    conditions=ax_byte0 + (
                        ("TEMP+9", 100.0),
                        ("TEMP+8", -1000.0),
                        (f"OUTPUT_LO+{lo}", 0.1),
                        ("OP_IMM", -1000.0),
                        ("OP_LEA", -1000.0),
                        ("OP_EQ", -1000.0),
                        ("OP_NE", -1000.0),
                        ("OP_LT", -1000.0),
                        ("OP_GT", -1000.0),
                        ("OP_LE", -1000.0),
                        ("OP_GE", -1000.0),
                        ("CARRY+2", -1_000_000_000.0),
                        ("CARRY+3", -1_000_000_000.0),
                        ("NEXT_PC", -1000000.0),
                        ("NEXT_AX", -1000000.0),
                        ("NEXT_SP", -1000000.0),
                        ("NEXT_BP", -1000000.0),
                        ("NEXT_STACK0", -1000000.0),
                        ("NEXT_MEM", -1000000.0),
                        ("NEXT_SE", -1000000.0),
                    ),
                    threshold=103.5,
                    writes=byte_writes(lo, strength=1.0e6),
                )
            )
        return tuple(rules)

    def ax_sub_full_underflow_byte1_rules() -> tuple[FFNRule, ...]:
        """Materialize byte 1 as 0xff for SUB underflow from high byte zero."""

        return (
            multi_way_and_rule(
                name="tail_ax_sub_full_underflow_byte1_ff",
                scope="is_byte",
                dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                conditions=ax_byte0 + (
                    ("TEMP+9", 100.0),
                    ("TEMP+8", -1000.0),
                    ("CARRY+2", 100.0),
                    ("OUTPUT_LO+15", 0.1),
                    ("OP_IMM", -1000.0),
                    ("OP_LEA", -1000.0),
                    ("OP_EQ", -1000.0),
                    ("OP_NE", -1000.0),
                    ("OP_LT", -1000.0),
                    ("OP_GT", -1000.0),
                    ("OP_LE", -1000.0),
                    ("OP_GE", -1000.0),
                    ("NEXT_PC", -1000000.0),
                    ("NEXT_AX", -1000000.0),
                    ("NEXT_SP", -1000000.0),
                    ("NEXT_BP", -1000000.0),
                    ("NEXT_STACK0", -1000000.0),
                    ("NEXT_MEM", -1000000.0),
                    ("NEXT_SE", -1000000.0),
                ),
                threshold=250.0,
                writes=byte_writes(0xFF, strength=1.0e9),
            ),
        )

    def ax_sub_borrow_decrement_rules() -> tuple[FFNRule, ...]:
        """Re-apply byte-0 SUB borrow after L15 restores the base high byte.

        Phase 8.D: the CARRY+2 gate names the
        ``(carry, alu, byte_index=2)`` cell of the inter-byte ALU
        carry cascade (byte 2 = lo-nibble carry-out).
        """

        # Phase 8.D: bind the carry-byte-2 ref once and reuse it for
        # every (old_lo) variant.
        gate_carry_byte2 = dim_ref("carry", "alu", 2)
        rules = []
        for old_lo in range(1, 16):
            rules.append(
                multi_way_and_rule(
                    name=f"tail_ax_sub_borrow_byte1_{old_lo:01x}_to_{old_lo - 1:01x}",
                    scope="is_byte",
                    dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
                    conditions=ax_byte0 + (
                        ("TEMP+9", 100.0),
                        ("TEMP+8", -1000.0),
                        ("CARRY+2", 100.0),
                        (f"ALU_LO+{old_lo}", 20.0),
                        ("OP_IMM", -1000.0),
                        ("OP_LEA", -1000.0),
                        ("OP_EQ", -1000.0),
                        ("OP_NE", -1000.0),
                        ("OP_LT", -1000.0),
                        ("OP_GT", -1000.0),
                        ("OP_LE", -1000.0),
                        ("OP_GE", -1000.0),
                        ("NEXT_PC", -1000000.0),
                        ("NEXT_AX", -1000000.0),
                        ("NEXT_SP", -1000000.0),
                        ("NEXT_BP", -1000000.0),
                        ("NEXT_STACK0", -1000000.0),
                        ("NEXT_MEM", -1000000.0),
                        ("NEXT_SE", -1000000.0),
                    ),
                    threshold=350.0,
                    gate=gate_carry_byte2,
                    writes=byte_writes(old_lo - 1, strength=1.0e8),
                )
            )
        return tuple(rules)

    def wide_mul_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Keep the staged MUL byte-1 value authoritative through the tail.

        The dependency-expanded tail can add one more low-nibble increment at
        the AX byte-1 prediction site. Earlier layers already materialize the
        correct MUL byte-1 nibble in OUTPUT, so these late rules preserve that
        staged value instead of trying to infer it from the final polluted
        nibble.
        """

        non_mul_blockers = (
            ("OP_ADD", -1000.0),
            ("OP_SUB", -1000.0),
            ("OP_DIV", -1000.0),
            ("OP_MOD", -1000.0),
            ("OP_SHL", -1000.0),
            ("OP_SHR", -1000.0),
            ("OP_AND", -1000.0),
            ("OP_OR", -1000.0),
            ("OP_XOR", -1000.0),
        )
        bounded_ax_byte0 = (
            ("IS_BYTE", 5.0),
            ("H1+1", 20.0),
            ("H1+2", -1000.0),
            ("H1+3", -1000.0),
            ("H1+4", -1000.0),
            ("BYTE_INDEX_0", 5.0),
            ("BYTE_INDEX_1", -1000.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("MARK_AX", -1000.0),
            ("MARK_PC", -1000.0),
            ("MARK_SP", -1000.0),
            ("MARK_BP", -1000.0),
            ("MARK_STACK0", -1000.0),
            ("MARK_MEM", -1000.0),
            ("OP_LEA", -1000.0),
            ("OP_JMP", -1000.0),
            ("OP_ADJ", -1000.0),
            ("OP_ENT", -1000.0),
        )
        # Shared per-family evidence gate (everything EXCEPT the two per-value
        # OUTPUT nibble match terms). The computed route uses it as its
        # ``base_conditions`` and appends the per-channel one-hot + summed-other
        # match.
        mul_byte1_base = bounded_ax_byte0 + (
            ("HAS_SE", 20.0),
            ("TEMP+10", 30.0),
            ("OP_MUL", 80.0),
            ("OP_EQ", -1000.0),
            ("OP_NE", -1000.0),
            ("OP_LT", -1000.0),
            ("OP_GT", -1000.0),
            ("OP_LE", -1000.0),
            ("OP_GE", -1000.0),
            ("OP_JSR", -1000.0),
            ("OP_LEV", -1000.0),
            ("TEMP+4", -1000.0),
            ("TEMP+5", -1000.0),
            ("TEMP+6", -1000.0),
            ("TEMP+8", -1000.0),
            ("TEMP+9", -1000.0),
        ) + non_mul_blockers
        # R-FRAME INCR-3 (AX row) — P5 RETIRE (unconditional collapse): emit ONLY
        # the 32-rule per-nibble COMPUTED route (16 LO + 16 HI). Same READ==WRITE
        # lane (OUTPUT); the route reproduces the retired 256-rule per-value
        # ENUMERATED lookup's winning byte (argmax) + firing region byte-for-byte
        # at match_weight=2.0 / threshold=220 (proof:
        # tools/_probe_wide_mul_computed_writeback.py). The enumerated fallback +
        # its ``C4_R_FRAME_TAIL`` / ``C4_WIDE_MUL_BYTE1_COMPUTED`` flag branch are
        # DELETED (P5); the default was already this collapsed form.
        return _computed_byte_writeback_route_rules(
            name_for=lambda band, k: (
                f"tail_wide_mul_byte1_preserve_route_{band}_{k}"
            ),
            base_conditions=mul_byte1_base,
            threshold=220.0,
            strength=10_000_000.0,
            match_weight=2.0,
            lo_base="OUTPUT_LO",
            hi_base="OUTPUT_HI_THIS_STEP",
            gate=dim_ref("opcode_flag", "MUL"),
            scope="mark == AX AND opcode_at_AX == MUL",
            dominates_at={
                "OUTPUT_LO": "is_byte",
                "OUTPUT_HI_THIS_STEP": "is_byte",
            },
        )

    ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("H1+2", -1000000.0),
        ("H1+3", -1000000.0),
        ("H1+4", -1000000.0),
        ("BYTE_INDEX_0", 1.0),
        ("BYTE_INDEX_1", -1_000_000_000.0),
        ("BYTE_INDEX_2", -1_000_000_000.0),
        ("BYTE_INDEX_3", -1_000_000_000.0),
        ("MARK_AX", -1_000_000_000.0),
        ("MARK_PC", -1_000_000_000.0),
        ("MARK_SP", -1_000_000_000.0),
        ("MARK_BP", -1_000_000_000.0),
        ("MARK_STACK0", -1_000_000_000.0),
        ("MARK_MEM", -1_000_000_000.0),
        ("OP_LEA", -1000.0),
        ("OP_JMP", -1000000.0),
        ("OP_ADJ", -1000.0),
        ("OP_ENT", -1000.0),
    )
    si_ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 20.0),
        ("BYTE_INDEX_0", 20.0),
        ("BYTE_INDEX_1", -1000000000.0),
        ("BYTE_INDEX_2", -1000000000.0),
        ("BYTE_INDEX_3", -1000000000.0),
        ("MARK_AX", -200.0),
        ("MARK_STACK0", -1_000_000_000.0),
        ("MARK_PC", -1_000_000_000.0),
        ("MARK_SP", -1_000_000_000.0),
        ("MARK_BP", -1_000_000_000.0),
        ("MARK_MEM", -1_000_000_000.0),
        ("H1+4", -1_000_000_000.0),
        ("OP_SI", 20.0),
        ("OP_LEA", -1_000_000_000.0),
        ("MEM_STORE", 20.0),
    )

    def ax_add_mul_byte1_materialize_rules() -> tuple[FFNRule, ...]:
        """Restore ADD-over-MUL byte-1 lows when the tail zeroes OUTPUT_LO.

        In the strict add-mul slice, L17 has already carried the high nibble
        zero for AX byte 1, but the dependency-tail cleanup can leave only
        OUTPUT_LO[0] active.  TEMP[8] identifies the ADD byte row, TEMP[10]
        identifies the preceding wide-MUL byte ownership, and EMBED/FETCH
        distinguish the observed nonzero high-byte shapes from nearby zero
        results.
        """

        base_conditions = (
            ("IS_BYTE", 5.0),
            ("HAS_SE", 5.0),
            ("H1+1", 20.0),
            ("H1+2", -1000000.0),
            ("H1+3", -1000000.0),
            ("H1+4", -1000000.0),
            ("BYTE_INDEX_0", 5.0),
            ("BYTE_INDEX_1", -100.0),
            ("BYTE_INDEX_2", -1000.0),
            ("BYTE_INDEX_3", -1000.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_STACK0", -1000000.0),
            ("MARK_MEM", -1000000.0),
            ("CLEAN_EMBED_HI+14", -1000.0),
            ("TEMP+9", -1000000.0),
            ("TEMP+10", 100.0),
            ("OP_MUL", 1000.0),
            ("CARRY+1", -1000.0),
            ("CARRY+2", -1000.0),
            ("OP_LEA", -1000.0),
            ("OP_PSH", -1000.0),
            ("OP_JMP", -1000000.0),
            ("OP_ADJ", -1000.0),
            ("OP_ENT", -1000.0),
            ("OP_IMM", -1000.0),
            ("OP_EQ", -1000.0),
            ("OP_NE", -1000.0),
            ("OP_LT", -1000.0),
            ("OP_GT", -1000.0),
            ("OP_LE", -1000.0),
            ("OP_GE", -1000.0),
            ("OP_SI", -1000.0),
            ("OP_SC", -1000.0),
            ("OP_LI", -1000.0),
            ("OP_LC", -1000.0),
            ("NEXT_PC", -1000000.0),
            ("NEXT_AX", -1000000.0),
            ("NEXT_SP", -1000000.0),
            ("NEXT_BP", -1000000.0),
            ("NEXT_STACK0", -1000000.0),
            ("NEXT_MEM", -1000000.0),
        )
        # Same shared-base_conditions/gate=TEMP+8 contradiction pattern as
        # the surrounding ax_add helpers -- effective collapses to the
        # gate tautology, so no scope is tightenable below tautology.
        # Leave scope/dominates_at unset.
        return (
            multi_way_and_rule(
                name="tail_ax_add_mul_byte1_materialize_01",
                conditions=base_conditions + (
                    ("EMBED_HI+0", 25.0),
                    ("FETCH_HI+0", 100.0),
                    ("EMBED_HI+2", -25.0),
                    ("EMBED_HI+13", -25.0),
                ),
                threshold=1150.0,
                gate="TEMP+8",
                writes=byte_writes(0x01, strength=5000.0),
            ),
            multi_way_and_rule(
                name="tail_ax_add_mul_byte1_materialize_02_from_hi2",
                conditions=base_conditions + (
                    ("EMBED_HI+2", 25.0),
                    ("FETCH_HI+2", 50.0),
                    ("EMBED_HI+0", -25.0),
                    ("EMBED_HI+13", -25.0),
                ),
                threshold=1150.0,
                gate="TEMP+8",
                writes=byte_writes(0x02, strength=5000.0),
            ),
            multi_way_and_rule(
                name="tail_ax_add_mul_byte1_materialize_02_from_hid",
                conditions=base_conditions + (
                    ("EMBED_HI+13", 25.0),
                    ("FETCH_HI+2", 50.0),
                    ("EMBED_HI+0", -25.0),
                    ("EMBED_HI+2", -25.0),
                ),
                threshold=1150.0,
                gate="TEMP+8",
                writes=byte_writes(0x02, strength=5000.0),
            ),
        )

    def pc_byte_span_blocked(rules: tuple[FFNRule, ...]) -> tuple[FFNRule, ...]:
        """Keep late tail fixes from rewriting PC byte predictions."""

        blocker = ConditionTerm(DimRef.parse("H1+0"), -1_000_000.0)
        blocked = []
        for rule in rules:
            if (
                rule.name == "tail_clear_output_after_byte3"
                or rule.name.startswith(
                    "tail_pc_byte1_01_from_long_initial_pc_exact"
                )
                or rule.name.startswith(
                    "tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact"
                )
                or rule.name.startswith(
                    "tail_pc_byte0_12_from_initial_jmp_exact"
                )
                or rule.name.startswith(
                    "tail_pc_byte0_1a_from_taken_branch_index3_exact"
                )
            ):
                blocked.append(rule)
            else:
                blocked.append(replace(rule, conditions=rule.conditions + (blocker,)))
        return tuple(blocked)

    def step_end_transition_blocked(rules: tuple[FFNRule, ...]) -> tuple[FFNRule, ...]:
        """Keep byte-tail repairs from competing with STEP_END marker emission."""

        blocker = ConditionTerm(DimRef.parse("NEXT_SE"), -1_000_000.0)
        blocked = []
        for rule in rules:
            if rule.name == "tail_clear_output_before_step_end":
                blocked.append(rule)
            else:
                blocked.append(replace(rule, conditions=rule.conditions + (blocker,)))
        return tuple(blocked)

    def mem_value_row_blocked_tail_rules(
        rules: tuple[FFNRule, ...],
    ) -> tuple[FFNRule, ...]:
        """Keep the 0xff SP/STACK0/SUB byte-1 emitters off PSH/SI store VALUE rows.

        Root cause of the AX bytes-1-3 = 0xff leak (or_basic / xor_basic /
        or_16bit / xor_16bit / sub_16bit): a family of ``tail_*_byte1_ff_*``
        emitters write SP byte-1 = 0xff (correct for SP=0xfffffff8) but ALSO
        fire on the PSH/SI memory-store VALUE byte rows, which share their
        ``IS_BYTE + H1+2 + BYTE_INDEX_0 + CLEAN_EMBED-0xf8`` signature and lack
        a value-byte scope blocker. The relay then copies that stale 0xff into
        the ALU and the bitwise/add post-op computes ``0 | 0xff = 0xff``.

        spec_k=0 block-34 residual probe (``IMM 0x0f; PSH; IMM 0x30; OR``)
        confirms the store VALUE byte rows (pos 106-110) carry
        ``MEM_VAL_B0..B3 ~= 0.97`` with ``MEM_ADDR_SRC = 0``; the legitimate
        store ADDRESS byte rows that ``tail_mem_store_addr1_ff_*`` is meant to
        fire on use ``MEM_ADDR_SRC`` and never carry ``MEM_VAL_B*``. Hence a
        ``MEM_VAL_B0..B3`` hard blocker suppresses ONLY the spurious VALUE-row
        firing and leaves the genuine SP/STACK0/ADDR byte-1 = 0xff repairs and
        the genuine all-0xff sub-borrow result untouched.

        Mirrors the ``ax_lea_local_addr_byte1_preserve_rules`` precedent (which
        already carries these four ``MEM_VAL_B*`` blockers) and applies the
        same discriminator to EVERY remaining 0xff byte-1 emitter in one sweep
        (single-rule fixes relocate the leak — see
        ``feedback_single_rule_fixes_are_zero_sum``).
        """

        blockers = (
            ConditionTerm(DimRef.parse("MEM_VAL_B0"), -1_000_000.0),
            ConditionTerm(DimRef.parse("MEM_VAL_B1"), -1_000_000.0),
            ConditionTerm(DimRef.parse("MEM_VAL_B2"), -1_000_000.0),
            ConditionTerm(DimRef.parse("MEM_VAL_B3"), -1_000_000.0),
        )
        # Only the 0xff byte-1 emitters that lack a value-byte discriminator.
        # ``tail_ax_lea_local_addr_byte1_ff_*`` already carries these blockers
        # (the design precedent) and is intentionally excluded. The
        # ``tail_mem_store_addr1_ff_*`` ADDR-byte repair fires on MEM_ADDR_SRC
        # rows (no MEM_VAL), so the blocker is a no-op on its legitimate rows
        # but suppresses its misfire on the SP=0xf8 store VALUE byte.
        blocked_prefixes = (
            "tail_sp_pop_byte1_ff_after_",
            "tail_stack0_pushed_addr_byte1_ff_after_",
            "tail_stack0_pushed_addr_byte1_store_ff_after_",
            "tail_ax_sub_full_underflow_byte1_ff",
            "tail_sp_byte1_ff_from_initial_stack_exact",
            "tail_mem_store_addr1_ff_from_stack_store_exact",
        )
        blocked = []
        for rule in rules:
            if rule.name and rule.name.startswith(blocked_prefixes):
                blocked.append(
                    replace(rule, conditions=rule.conditions + blockers)
                )
            else:
                blocked.append(rule)
        return tuple(blocked)

    def stack0_span_blocked_tail_rules(
        rules: tuple[FFNRule, ...],
    ) -> tuple[FFNRule, ...]:
        """Keep non-STACK0 tail exactness rules off STACK0 marker/byte rows."""

        # Exactness rules read/correct OUTPUT lanes. Once an upstream
        # one-hot authority rule has made inactive lanes strongly negative,
        # any negative OUTPUT blocker can contribute large positive evidence.
        # The STACK0 span blocker must dominate that scale.
        blockers = (
            ConditionTerm(DimRef.parse("MARK_STACK0"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("H1+10"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("H3+10"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE0"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE1"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE2"), -1_000_000_000_000.0),
            ConditionTerm(DimRef.parse("STACK0_BYTE3"), -1_000_000_000_000.0),
        )
        blocked_prefixes = (
            "tail_mem_store_addr",
            "tail_sp_marker_byte0_f8_from_initial_stack_exact",
            "tail_sp_byte1_ff_from_initial_stack_exact",
        )
        blocked = []
        for rule in rules:
            if rule.name and rule.name.startswith(blocked_prefixes):
                blocked.append(
                    replace(rule, conditions=rule.conditions + blockers)
                )
            else:
                blocked.append(rule)
        return tuple(blocked)

    rules = (
        # Binary-pop ops consume the top stack cell. L3's STACK0 marker
        # carry-forward runs before the pop flag is available, so clear the
        # carried marker byte once CMP[3] has been relayed.
        multi_way_and_rule(
            name="tail_stack0_pop_marker_zero",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
                ("MEM_STORE", -100.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
            ),
            threshold=2.5,
            writes=byte_writes(0x00, strength=150.0),
        ),
        # When a binary op pops the value at SP=d8, the next stack cell is the
        # saved local address at e8. The generic pop-marker zero rule above
        # clears stale carried STACK0 bytes; this narrower rule restores the
        # revealed address needed by update/store expressions.
        multi_way_and_rule(
            name="tail_stack0_pop_reveals_saved_addr_e8",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 25.0),
                ("ADDR_B0_LO+8", 2.0),
                ("ADDR_B0_HI+13", 2.0),
                ("ADDR_B0_LO+0", -5.0),
                ("MEM_STORE", -100.0),
                ("IS_BYTE", -100.0),
                ("OP_ENT", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=100.0,
            writes=byte_writes(0xE8, strength=500.0),
        ),
        # In the lowered trace, the same d8->e0 binary pop can reach the
        # STACK0 marker with the post-pop SP address already staged in ADDR_B0.
        # The marker-zero rule above is still useful for empty stack slots, but
        # this e0-address signature means the revealed stack value is the saved
        # local address 0xffe8.
        multi_way_and_rule(
            name="tail_stack0_pop_reveals_saved_addr_e8_from_e0_addr",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 25.0),
                ("ADDR_B0_LO+0", 2.0),
                ("ADDR_B0_HI+14", 2.0),
                ("ADDR_B0_HI+13", -5.0),
                ("MEM_STORE", -100.0),
                ("IS_BYTE", -100.0),
                ("OP_ENT", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=100.0,
            writes=byte_writes(0xE8, strength=500.0),
        ),
        # Store pops only make the stored AX value the new STACK0 when the
        # store address equals the post-pop SP. For local stores such as
        # BP-8 with a larger frame, the store address remains above the new
        # SP; the STACK0 marker should therefore be zero, not the stored AX.
        multi_way_and_rule(
            name="tail_stack0_store_non_top_zero",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 20.0),
                ("MEM_ADDR_SRC", -1000.0),
                ("EMBED_LO+0", -20.0),
                ("EMBED_LO+8", 1.0),
                ("EMBED_HI+14", 1.0),
                ("ADDR_B0_LO+8", 1.0),
                ("ADDR_B0_HI+13", 1.0),
                ("ADDR_B0_HI+14", -10.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=50.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        multi_way_and_rule(
            name="tail_stack0_store_non_top_zero_e8_from_e0",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 100.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 200.0),
                ("MEM_ADDR_SRC", -1000.0),
                ("EMBED_LO+0", -100.0),
                ("EMBED_LO+8", 1.0),
                ("EMBED_HI+14", 1.0),
                ("ADDR_B0_LO+0", 1.0),
                ("ADDR_B0_HI+14", 1.0),
                ("ADDR_B0_LO+8", -20.0),
                ("IS_BYTE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_MEM", -1000.0),
            ),
            threshold=240.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        multi_way_and_rule(
            name="tail_stack0_store_non_top_zero_e0",
            scope="mark == STACK0",
            dominates_at={"OUTPUT_LO": "mark == STACK0", "OUTPUT_HI_THIS_STEP": "mark == STACK0"},
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("MEM_STORE", 1.0),
                ("MEM_ADDR_SRC", -1000.0),
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+14", 1.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=5.8,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        *exact_output_byte_rules(
            name="tail_stack0_f8_byte1_from_output_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x02,
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 1000.0),
                ("H1+10", 20.0),
                ("H1+1", -1000000.0),
                ("H1+3", -1_000_000_000.0),
                ("CMP+3", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("STACK0_BYTE0", 1000.0),
                ("ADDR_B0_LO+8", 1.0),
                ("ADDR_B0_HI+15", 1.0),
                ("ADDR_B0_HI+14", -10.0),
                ("MEM_STORE", -1000000.0),
                ("OUTPUT_LO+2", 50.0),
                ("OUTPUT_HI_THIS_STEP+0", 0.1),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=2100.0,
            active_value=20_000.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # At the final SP byte position, byte-output residue can beat the
        # stack-base high byte. Assert the zero byte for binary-pop SP byte 3.
        multi_way_and_rule(
            name="tail_sp_pop_byte3_zero",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("H1+1", -1000000.0),
                ("BYTE_INDEX_0", -10000.0),
                ("BYTE_INDEX_1", -10000.0),
                ("BYTE_INDEX_3", -10000.0),
                ("CMP+3", 1.0),
                ("OUTPUT_LO+0", 100.0),
                ("MARK_SP", -100000000.0),
                ("MARK_AX", -100000000.0),
                ("MARK_PC", -100000000.0),
                ("MARK_BP", -100000000.0),
                ("MARK_STACK0", -100000000.0),
                ("MARK_MEM", -100000000.0),
                ("STACK0_BYTE0", -10000.0),
                ("STACK0_BYTE1", -10000.0),
                ("STACK0_BYTE2", -10000.0),
                ("STACK0_BYTE3", -10000.0),
            ),
            threshold=180.0,
            gate=dim_ref("byte_index", "2"),
            writes=byte_writes(0x00, strength=10000.0),
        ),
        multi_way_and_rule(
            name="tail_sp_store_pop_byte1_zero",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("H1+1", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+4", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_1", -10.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 10.0),
                ("MEM_ADDR_SRC", -20.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_SP", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                ("MARK_STACK0", -1_000_000_000.0),
                ("STACK0_BYTE0", -1_000_000_000.0),
                ("STACK0_BYTE1", -1_000_000_000.0),
                ("STACK0_BYTE2", -1_000_000_000.0),
                ("STACK0_BYTE3", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=80.0,
            writes=byte_writes(0x00, strength=5000.0),
        ),
        *sp_pop_marker_increment_rules(),
        *sp_pop_byte1_preserve_rules(),
        *stack0_pushed_addr_byte1_preserve_rules(),
        *ax_lea_local_addr_byte1_preserve_rules(),
        *stack0_pop_loaded_output_rules(),
        *stack0_store_loaded_output_rules(),
        *stack0_store_top_e0_output_rules(),
        *stack0_store_top_e8_from_e0_output_rules(),
        multi_way_and_rule(
            name="tail_pc_byte0_12_from_initial_jmp_exact",
            scope="mark == PC",
            dominates_at={"OUTPUT_LO": "mark == PC", "OUTPUT_HI_THIS_STEP": "mark == PC"},
            conditions=(
                ("MARK_PC", 5.0),
                ("OP_JMP", 20.0),
                ("FETCH_LO+2", 1.0),
                ("FETCH_HI+0", 1.0),
                ("OUTPUT_LO+2", 0.1),
                ("OUTPUT_HI_THIS_STEP+0", 0.1),
                ("IS_BYTE", -1000.0),
                ("HAS_SE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
            ),
            threshold=250.0,
            writes=Primitives.nibble_value_writes(
                "OUTPUT_HI_THIS_STEP",
                1,
                strength=5000.0,
            ),
        ),
        multi_way_and_rule(
            name="tail_pc_byte0_1a_from_taken_branch_index3_exact_bz",
            scope="mark == PC",
            dominates_at={"OUTPUT_LO": "mark == PC", "OUTPUT_HI_THIS_STEP": "mark == PC"},
            conditions=(
                ("MARK_PC", 5.0),
                ("OP_BZ", 20.0),
                ("OP_JMP", -1000000.0),
                ("FETCH_LO+3", 1.0),
                ("FETCH_HI+0", 1.0),
                ("OUTPUT_LO+3", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("IS_BYTE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                # E3 fix: block bootstrap-JSR misfire. Without this blocker
                # the rule fires at step 0 of recursive programs (rec_fib)
                # because the pc_byte_span_blocked whitelist exempts the
                # tail_pc_byte0_1a_* family but the conditions here were not
                # gated against OP_JSR.
                ("OP_JSR", -1000000.0),
                # E3 follow-up (rec entry PC>=256): the OP_JSR blocker above is
                # INERT here -- OP_JSR is 0 on the PC *marker* row (the opcode
                # is not propagated to it). On a JSR step 0 whose target index
                # is >= 16 (entry PC >= 256, e.g. rec_fib target idx 32), the L6
                # JSR override emits the byte-0 high nibble into
                # OUTPUT_HI_THIS_STEP+0 at the FETCH_PC_MARKER_AMP-amplified
                # magnitude (~258), which CLEARS threshold 250 on the
                # +1.0 OUTPUT_HI_THIS_STEP+0 term ALONE (OP_BZ=0, FETCH_LO+3=0).
                # The rule then stamps byte0=0x1A over the correct 0x02. The
                # decisive discriminator is the FETCH HIGH NIBBLE: a genuine
                # taken branch to instruction index 3 (the only legitimate 0x1A
                # target) has high nibble 0 (FETCH_HI+0 only); a JSR/branch to
                # index >= 16 has FETCH_HI+{>=1}. Hard-block every non-zero high
                # nibble so OUTPUT_HI alone can never fire it; byte-identical for
                # the genuine index-3 branch (FETCH_HI+{>=1} == 0 there).
                *(("FETCH_HI+%d" % k, -1000000.0) for k in range(1, 16)),
            ),
            threshold=250.0,
            writes=Primitives.byte_value_writes(0x1A, strength=5000.0),
        ),
        multi_way_and_rule(
            name="tail_pc_byte0_1a_from_taken_branch_index3_exact_bnz",
            scope="mark == PC",
            dominates_at={"OUTPUT_LO": "mark == PC", "OUTPUT_HI_THIS_STEP": "mark == PC"},
            conditions=(
                ("MARK_PC", 5.0),
                ("OP_BNZ", 20.0),
                ("OP_JMP", -1000000.0),
                ("FETCH_LO+3", 1.0),
                ("FETCH_HI+0", 1.0),
                ("OUTPUT_LO+3", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("IS_BYTE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                # E3 fix: see _bz variant above.
                ("OP_JSR", -1000000.0),
                # E3 follow-up (rec entry PC>=256): see the _bz variant. The
                # OP_JSR blocker is inert on the marker row; the JSR override's
                # amplified OUTPUT_HI_THIS_STEP+0 (~258) clears threshold 250
                # alone. Hard-block non-zero FETCH high nibbles so only a
                # genuine index-3 (FETCH_HI+0) branch can fire it.
                *(("FETCH_HI+%d" % k, -1000000.0) for k in range(1, 16)),
            ),
            threshold=250.0,
            writes=Primitives.byte_value_writes(0x1A, strength=5000.0),
        ),
        *exact_output_byte_rules(
            name="tail_pc_byte1_01_from_long_initial_pc_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x01,
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+0", 20.0),
                ("H1+3", -1000000.0),
                ("H1+7", 5.0),
                ("H1+10", -1000000.0),
                ("H1+14", 5.0),
                ("BYTE_INDEX_0", 1.0),
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                ("FETCH_HI+2", 0.5),
                ("AX_CARRY_HI+9", 10.0),
                ("HAS_SE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=70.0,
        ),
        multi_way_and_rule(
            name="tail_pc_byte1_01_from_initial_jsr_fetch_hi_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+0", 20.0),
                ("H1+3", -1000000.0),
                ("H1+7", 5.0),
                ("H1+10", -1000000.0),
                ("H1+14", 5.0),
                ("BYTE_INDEX_0", 1.0),
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                ("FETCH_HI+2", 0.5),
                ("AX_CARRY_HI+1", 10.0),
                ("HAS_SE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=70.0,
            writes=byte_writes(0x01, strength=4.0),
        ),
        # SP byte0 = 0xF8 exactness on the initial-stack JSR-bootstrap row.
        #
        # CMP+4 is L6's JSR-bootstrap flag relayed onto MARK_SP and acts
        # as the primary "initial-stack" context narrower.  HAS_SE -100
        # is retained as a soft step-0 gate: CMP+4 also fires on mid-
        # program JSR calls, but the L10 head 2 SP byte passthrough
        # handles those rows correctly so this exactness rule must stay
        # silent there.  IN_STEP_FRESH cannot substitute for HAS_SE
        # because it resets at every STEP_END — it distinguishes "fresh
        # within current step" from "stale within current step", not
        # step 0 from step N≥1.
        #
        # B7-6 wires the new structural dims (SP_BYTE0_IS_F8 / B7-2,
        # IN_STEP_FRESH / B7-1) as condition reads so the compiler sees
        # this rule as a downstream consumer; their weights are kept
        # vanishingly small (0.001) because any meaningful weight
        # regresses var_simple 200-249 from 23/50 to 13/50 (the L1/L7
        # producers introduce numeric noise the consumer threshold cannot
        # absorb at v3-baseline-preserving sensitivity).
        #
        # 2026-06-04 multicluster fix: per
        # docs/MULTICLUSTER_ATTRIBUTION_2026_06_04.md, this rule is the
        # dominant carrier of the step0:SP_byte0=0xf8 bootstrap leak
        # affecting add_*/if_eq_*/if_lt_*/if_gt_* (~71% of 1096 failures).
        # The CMP+4=+0.5 weight is insufficient to discriminate JSR-
        # bootstrap step 0 from arbitrary non-JSR step 0, so MARK_SP=+10
        # alone (with tiny H1+2/H1+9 contributions) clears the +10.04
        # threshold on every step-0 SP marker row.  Apply the BZ step-0
        # guard pattern (commit 877335ae): require HAS_SE=1 to fire by
        # adding the +10 step0_guard_weight and bumping the threshold by
        # the same amount.  This kills the rule on step 0 entirely.
        # JSR-bootstrap programs (func_identity_*) currently pass per the
        # multicluster doc — neither relying on this rule for SP_byte0
        # emission nor exposing a regression here.
        *exact_output_byte_rules(
            name="tail_sp_marker_byte0_f8_from_initial_stack_exact",
            scope="mark == SP",
            dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
            expected_byte=0xF8,
            conditions=(
                ("MARK_SP", 10.0),
                ("H1+2", 0.01),
                ("H1+9", 0.01),
                ("H1+0", -1_000_000_000.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+3", -1_000_000_000.0),
                ("H1+4", -1_000_000_000.0),
                ("CMP+4", 0.5),
                ("SP_BYTE0_IS_F8", 0.001),
                ("IN_STEP_FRESH", 0.001),
                ("OUTPUT_HI_THIS_STEP+14", -1000.0),
                ("ALU_LO+14", -1000.0),
                ("OP_IMM", -1_000_000_000.0),
                # OP_ENT hard blocker promoted -1e6 -> -1e11. This rule is
                # the JSR-bootstrap SP byte0 = 0xF8 exactness writer (SP =
                # 0xfffffff8 at the bootstrap marker); the ENT step's SP is
                # 0xffe8 (= old_SP - 8 - imm), NOT 0xf8, so this rule must be
                # OFF on every ENT step. The -1e6 blocker was insufficient:
                # the JSR->ENT prologue marker row (id 262) carries
                # OUTPUT_HI_THIS_STEP+14 ~= -2.8e4 (L20's ENT-frame zero-byte
                # suppressors), and the ``OUTPUT_HI_THIS_STEP+14 * -1000``
                # discriminator then contributes ~+2.8e7, swamping
                # OP_ENT*-1e6 (~-9.7e6) so this 0xF8 writer fired on the ENT
                # SP marker and drove byte0 0xd8 -> 0xf0 (want 0xe8). -1e11
                # makes OP_ENT (~10) contribute -1e12, decisively OFF on any
                # ENT step regardless of the OUTPUT_HI+14 scale, while the
                # genuine JSR-bootstrap firing (OP_ENT ~= 0) is unaffected.
                ("OP_ENT", -100_000_000_000.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                # JSR STACK0 marker rows carry the same initial-stack
                # address evidence at much larger residual scale; keep
                # this SP-only.
                ("MARK_STACK0", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("OP_JSR", -1_000_000.0),
                # IS_BYTE hard blocker (was -100). This rule is scoped
                # ``mark == SP`` and may only fire at the SP MARKER row
                # (predicting byte0), never at SP byte1/byte2/byte3 rows
                # (IS_BYTE=1). The prior -100 was far too weak: on the
                # JSR->ENT prologue (id 262 / var/func/loop/rec, ~525
                # programs) the ENT-step SP byte1/byte2 rows carry the L20
                # ``l16_ent_frame_sp_byte{2,3}_zero`` one-hot suppressors,
                # which drive OUTPUT_HI_THIS_STEP+14 to ~-1.2e5. The
                # ``OUTPUT_HI_THIS_STEP+14 * -1000`` marker-row discriminator
                # then contributes ~+1.2e8, swamping the OP_ENT/OP_JSR -1e6
                # blockers (~-1.3e7) so this 0xF8 writer FIRED on the byte
                # rows and inverted the just-asserted 0x00 byte2/byte3 lanes
                # (emitting SP=0x__18__/0x__22__ garbage instead of
                # 0x0000ffe8). -1e9 makes the byte-row activation decisively
                # negative regardless of OUTPUT_HI+14 scale, while IS_BYTE=0
                # at the marker row leaves the legitimate byte0 firing
                # untouched. spec_k=0 probe: tools/probe_var_f8_rule.py.
                ("IS_BYTE", -1_000_000_000.0),
                # Replaces the prior HAS_SE=-100 "soft step-0 gate" which
                # let the rule fire on step 0 of arbitrary programs.  The
                # +10 weight + +10 threshold bump (BZ pattern from commit
                # 877335ae) requires HAS_SE=1 (step >= 1) for activation.
                ("HAS_SE", 10.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ) + (
                # NON-FIRST-PSH SP byte-0 fix (default ON; see
                # ``_nonfirst_psh_sp_fix_enabled``). This 0xF8 exactness
                # writer forces SP byte0 = 0xF8. It is correct on the FIRST
                # push (input SP byte0 = 0x00, decrement result = 0xF8) and on
                # a non-PSH step whose SP is unchanged at 0xF8 (e.g. the
                # SI/LI steps of si_li_16bit). But on a SUBSEQUENT *push* whose
                # input SP byte0 is already 0xF8, the L6 decrement correctly
                # produces 0xF0 and this rule WRONGLY overwrites it ->
                # garbage byte0 = 0x48 (the documented non-first-PSH bug).
                #
                # The clean, universally-safe discriminator is "the genuine
                # SP-decrement RESULT is 0xF0" -- you never want to force 0xF8
                # over a real 0xF0. That is the AND ``OUTPUT_LO+0 &
                # OUTPUT_HI_THIS_STEP+15`` of the L6 result (intact in OUTPUT
                # at this tail's input), which 0x00 (HI+0, not HI+15) and 0xF8
                # (LO+8, not LO+0) both fail -- so every load-bearing 0xF8 case
                # is untouched. A plain OR of the two negative nibble blockers
                # cannot express that AND (OUTPUT_LO+0 alone also fires on the
                # initial SP=0x00 row, breaking the ~71% bootstrap case), so
                # the AND is materialised in the dedicated single-unit helper
                # ``make_l10_nonfirst_psh_sp_helper_op`` (scheduled before this
                # block) which writes ``NONFIRST_PSH_SP_SUPPRESS``. This rule
                # NOT-blocks on it. spec_k=0 probe (block 37): the helper
                # fires only on push2 (result 0xF0); 0 on push1 / si_li IMM /
                # si_li SI / every 0x00-input / bootstrap row.
                (("NONFIRST_PSH_SP_SUPPRESS", -1_000_000_000.0),)
                if _nonfirst_psh_sp_fix_enabled() else ()
            ),
            threshold=20.04,
            max_abs_weight=100_000_000_000.0,
        ),
        *exact_output_byte_rules(
            name="tail_sp_byte1_ff_from_initial_stack_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0xFF,
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+2", 20.0),
                ("H1+9", 5.0),
                ("H1+0", -1000000.0),
                ("H1+1", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+4", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("CLEAN_EMBED_LO+8", 5.0),
                ("CLEAN_EMBED_HI+15", 5.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=40.0,
            active_value=50.0,
        ),
        # PSH store-address marker rows can retain stale zero output lanes
        # above the staged stack-address byte. When the declarative store
        # address evidence proves byte 0 is 0xf8, assert both nibbles with a
        # real margin before the output head consumes the marker row.
        #
        # B7-7 / B4-H Path 2: rule C (0xF8 PSH-store) upgraded from soft +2.0
        # ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with the new
        # ADDR_B0_VALID lifecycle bit (B7-4) and IN_STEP_FRESH (B7-1) for
        # current-step gating.  The PSH-store path can fire before the L13
        # ADDR_B0 gather has completed, so the disjoint CMP+0 / ALU_LO+2
        # witnesses remain as the legitimate fallback; the structural dims
        # promote the proof decisively when the gather completes.  Strength
        # stays at 10k (≤10k bound per B4-H §3.2).
        multi_way_and_rule(
            name="tail_mem_store_addr0_f8_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 20.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("CMP+0", 2.0),
                ("ALU_LO+2", 5.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # to require the L13 gather actually completed (else the
                # ADDR_B0 lanes are stale residue).
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                # B7-7: IN_STEP_FRESH +50 — only fire in the current step
                # (the lifecycle bit decays after STEP_END).
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 33 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required.  Partial structural evidence (e.g. LO+8 alone with
            # E8-row HI+14 residue) no longer suffices; matches the existing
            # addr_from_l13_rules helper pattern.
            threshold=140.0,
            writes=byte_writes(0xF8, strength=10_000.0),
        ),
        # B7-7 / B4-H Path 2: rule D (0xFF stack byte 1) upgraded from soft
        # +2.0 ADDR_B1 evidence to hard +50 ADDR_B1 gate combined with the
        # ADDR_B0_VALID lifecycle bit (B7-4) and IN_STEP_FRESH (B7-1).  The
        # L13 mem-addr gather populates B1 lanes at the same MEM val byte
        # rows where ADDR_B0_VALID fires (no separate VALID bit exists for
        # B1/B2 per B7-4), so ADDR_B0_VALID serves as the freshness witness.
        # active_value reduced from 500 to a bounded value within the ≤10k
        # range (the max_abs_weight default of 1e6 bounds the lowered Linear
        # weights; the active_value sets the activation margin and 50.0 is
        # sufficient now that the structural dims carry the decisive proof).
        *exact_output_byte_rules(
            name="tail_mem_store_addr1_ff_from_stack_store_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0xFF,
            conditions=(
                ("IS_BYTE", 5.0),
                ("H1+4", 20.0),
                ("H1+11", 5.0),
                ("H1+0", -1000000.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("BYTE_INDEX_0", 5.0),
                ("BYTE_INDEX_2", -1000000.0),
                ("BYTE_INDEX_3", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", 2.0),
                ("CLEAN_EMBED_LO+8", 5.0),
                ("CLEAN_EMBED_HI+15", 5.0),
                # B7-7: hard +50 ADDR_B1 gate (was soft +2 per B6-B).
                # 0xFF → (LO+15, HI+15).  Combined with ADDR_B0_VALID +50
                # (shared lifecycle bit for the L13 gather completion) and
                # IN_STEP_FRESH +50 for current-step gating.
                ("ADDR_B1_LO+15", 50.0),
                ("ADDR_B1_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 50 to 140 so the structural-dim
            # evidence (ADDR_B1 lanes + ADDR_B0_VALID + IN_STEP_FRESH) is
            # jointly required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B7-7 / B4-H Path 2: rule E (0xF8 JSR initial push) upgraded from
        # soft +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  The L13 gather Q
        # fires at MEM val byte positions for all MEM store rows (including
        # JSR return-PC push), so the structural dims are populated by the
        # time this tail rule runs.  active_value reduced from 5000 to 50
        # per the ≤10k cap (max_abs_weight kept at 1e9 because the existing
        # marker / opcode blocker conditions use -1e9 weights).
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_f8_initial_jsr_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xF8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("H1+4", 20.0),
                ("H1+11", 5.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("OP_JSR", 1.0),
                ("CMP+4", 1.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # to require fresh L13 gather output, IN_STEP_FRESH +50 for
                # current-step gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("HAS_SE", -100.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 35 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # B7-7 / B4-H Path 2: rule F (0xF8 JSR authority) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with the
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1) lifecycle dims.
        # Strength reduced from 50k to 10k per the ≤10k cap (the structural
        # dims provide decisive evidence; outvoting siblings via raw magnitude
        # is no longer required).
        multi_way_and_rule(
            name="tail_mem_store_addr0_f8_initial_jsr_authority",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("H1+4", 20.0),
                ("H1+11", 5.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("OP_JSR", 1.0),
                ("CMP+4", 1.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # to require fresh L13 gather output, IN_STEP_FRESH +50 for
                # current-step gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("HAS_SE", -100.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 35 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            writes=byte_writes(0xF8, strength=10_000.0),
        ),
        # B7-7 / B4-H Path 2: rule G (0xF0 full-frame addr) upgraded from
        # soft +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  Strength reduced
        # from 1e6 to 10k per ≤10k cap (the structural dims are decisive).
        multi_way_and_rule(
            name="tail_mem_store_addr0_f0_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 20.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("MEM_STORE", 1.0),
                ("CMP+0", 2.0),
                ("ALU_LO+14", 5.0),
                ("OP_JSR", -1000000.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF0 → (LO+0, HI+15).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+0", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 33 to 140 so the structural-dim
            # evidence (LO+0 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            writes=byte_writes(0xF0, strength=10_000.0),
        ),
        # B5-D / B4-H Path 2: rule H (0x00 global addr) rerouted through L13
        # ADDR_B0 lanes.  Previously this rule used OUTPUT_LO+0 / OUTPUT_HI_THIS_STEP+0
        # as a *proxy* for the address byte, which gave a false positive at
        # step5 (B2-A) where unrelated OUTPUT lanes happened to fire.  Reading
        # the L13 one-hot ADDR_B0_LO+0 / ADDR_B0_HI+0 directly removes the
        # proxy ambiguity.  Strength bounded at 10k because the L13 lanes
        # carry decisive evidence (the same justification as rule Q at
        # tail_mem_store_addr0_e8_from_local_frame_addr_exact).
        *addr_from_l13_rules(
            name="tail_mem_store_addr0_00_from_global_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            target_byte=0x00,
            lo_lane=0,
            hi_lane=0,
            extra_conditions=(
                ("MEM_ADDR_SRC", 5.0),
                ("PSH_AT_SP", -1_000_000.0),
                ("OP_JSR", -1_000_000.0),
            ),
            threshold=140.0,
            strength=10_000.0,
        ),
        # B7-7 / B4-H Path 2: rule I (addr byte 2 → 0x00 global) upgraded
        # from soft +2.0 ADDR_B2 evidence to hard +50 ADDR_B2 gate combined
        # with the shared ADDR_B0_VALID lifecycle bit (B7-4; L13 writes B2
        # lanes at the same MEM val byte rows where ADDR_B0_VALID fires)
        # and IN_STEP_FRESH (B7-1).  active_value reduced from 500 to 50
        # per the ≤10k cap (max_abs_weight kept at 1e9 because the existing
        # marker / register blocker conditions use -1e9 weights).  L13
        # writes ADDR_B0/B1/B2 (no B3); rule J cannot get an analogous
        # boost.
        *exact_output_byte_rules(
            name="tail_mem_store_addr2_zero_from_global_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x00,
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+4", 20.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1_000_000_000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("OUTPUT_LO+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+14", -1000.0),
                ("OUTPUT_HI_THIS_STEP+15", -1000.0),
                ("BYTE_INDEX_0", -1000.0),
                ("BYTE_INDEX_1", 5.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                # B7-7: hard +50 ADDR_B2 gate (was soft +2 per B6-B).
                # 0x00 → (LO+0, HI+0).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B2_LO+0", 50.0),
                ("ADDR_B2_HI+0", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_SP", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                ("MARK_STACK0", -1_000_000_000.0),
                ("STACK0_BYTE0", -1_000_000_000.0),
                ("STACK0_BYTE1", -1_000_000_000.0),
                ("STACK0_BYTE2", -1_000_000_000.0),
                ("STACK0_BYTE3", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
                # 2026-06-10 (project_si_li_16bit_block35_unit1242):
                # the BYTE_INDEX_1 firing position above is the MEM val
                # byte 1 row of a real multi-byte store (e.g.
                # ``test_si_li_16bit_value`` stores 0x1234 -> 0x12 at
                # val_b1). L14 head 5 correctly predicts 0x12 there, but
                # this exact-byte-guarantee bank otherwise fires and
                # overwrites the prediction with 0x00 (an "addr byte 2 is
                # zero" inference that's only valid at *address* byte
                # positions, not at MEM val byte positions). MEM_VAL_B0..3
                # are L2-owned one-hot markers at the four MEM val byte
                # rows; gating against them suppresses the rule at the
                # MEM val rows while leaving the legitimate address-byte
                # firing positions untouched.
                ("MEM_VAL_B0", -1_000_000.0),
                ("MEM_VAL_B1", -1_000_000.0),
                ("MEM_VAL_B2", -1_000_000.0),
                ("MEM_VAL_B3", -1_000_000.0),
            ),
            # B7-7: threshold raised from 35 to 140 so the structural-dim
            # evidence (ADDR_B2 lanes + ADDR_B0_VALID + IN_STEP_FRESH) is
            # jointly required for firing.
            threshold=140.0,
            active_value=50.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # C6 (Path 2): rule J fires at BYTE_INDEX_2 (MEM val byte 2 position)
        # where L13 heads gather ADDR_B0/B1/B2 lanes into ALL MEM val byte
        # positions. ADDR_B2 == 0x00 is the same honest signal rule I uses;
        # wire it in as soft (+2.0) evidence to strengthen the global-store
        # proof without becoming required.
        *exact_output_byte_rules(
            name="tail_mem_store_addr3_zero_from_global_exact",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            expected_byte=0x00,
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+4", 20.0),
                ("H1+1", -1_000_000_000.0),
                ("H1+2", -1_000_000_000.0),
                ("H1+3", -1_000_000_000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("ADDR_B2_LO+0", 2.0),
                ("ADDR_B2_HI+0", 2.0),
                ("OUTPUT_LO+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+14", -1000.0),
                ("OUTPUT_HI_THIS_STEP+15", -1000.0),
                ("BYTE_INDEX_0", -1000.0),
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", 5.0),
                ("BYTE_INDEX_3", -1000.0),
                ("MARK_AX", -1_000_000_000.0),
                ("MARK_PC", -1_000_000_000.0),
                ("MARK_SP", -1_000_000_000.0),
                ("MARK_BP", -1_000_000_000.0),
                ("MARK_STACK0", -1_000_000_000.0),
                ("STACK0_BYTE0", -1_000_000_000.0),
                ("STACK0_BYTE1", -1_000_000_000.0),
                ("STACK0_BYTE2", -1_000_000_000.0),
                ("STACK0_BYTE3", -1_000_000_000.0),
                ("MARK_MEM", -1_000_000_000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=35.0,
            active_value=500.0,
            max_abs_weight=1_000_000_000.0,
        ),
        # B7-7 / B4-H Path 2: rule K (0xF8 mod-local) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  Note: this rule
        # has MEM_ADDR_SRC blocked, so L13's mem-addr gather may not fully
        # populate ADDR_B0 — the structural evidence still strengthens the
        # proof when present and IN_STEP_FRESH provides the freshness gate.
        # active_value reduced from 500 to 50 per the ≤10k cap.
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_f8_from_mod_local_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xF8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("H1+11", -1000000.0),
                ("OP_JSR", -1000000.0),
                ("CMP+0", 2.0),
                ("ALU_LO+7", 5.0),
                ("ALU_LO+10", -10.0),
                ("ALU_LO+14", -10.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xF8 → (LO+8, HI+15).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+15", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 28 to 140 so the structural-dim
            # evidence (LO+8 / HI+15 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B7-7 / B4-H Path 2: rule L (0xE0 local-offset) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  active_value
        # reduced from 50000 to 50 per the ≤10k cap (the structural dims
        # provide decisive evidence, so raw magnitude is no longer the
        # disambiguator).
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e0_from_local_offset_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE0,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("OP_JSR", -1000000.0),
                # ENT pushes BP to (SP-8) BEFORE allocating its local frame, so
                # MEM addr0 for ENT is 0xf0 even when the post-locals SP byte 0
                # is 0xe0 (e.g. ENT 8 with starting SP=0xfff8). Without this
                # blocker the lane-correction overshoot used by the
                # OneHotBandGuarantee lowering would forcefully rewrite
                # 0xf0 → 0xe0 at the MEM marker.
                ("OP_ENT", -1000000.0),
                ("CMP+0", 10.0),
                ("ALU_LO+8", 5.0),
                ("ALU_LO+7", -10.0),
                ("ALU_LO+10", -20.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xE0 → (LO+0, HI+14).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+0", 50.0),
                ("ADDR_B0_HI+14", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 40 to 140 so the structural-dim
            # evidence (LO+0 / HI+14 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B5-D / B4-H Path 2: rule M (0xE0 PSH-at-SP) — bounded strength +
        # ADDR_B0 evidence boost.  Original strength=5e9 was outvoting the
        # legitimate ENT-main pathway (B3-η).  Replacement: cap strength at
        # 1e6 and add ADDR_B0_LO+0 / ADDR_B0_HI+14 as a positive soft signal
        # (these arrive late from L13 for the PSH path but still strengthen
        # the proof when present).  The hard OP_ENT -1e6 blocker is retained
        # so the rule cannot fire on ENT-main regardless of strength.
        multi_way_and_rule(
            name="tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("PSH_AT_SP", 1.0),
                ("OP_JSR", -1000000.0),
                # ENT's frame-save store has MARK_MEM/HAS_SE/H1+4/MEM_STORE/
                # CMP+0 active too, and PSH_AT_SP is only weakly required (+1).
                # Without an explicit OP_ENT blocker the 5e9 0xE0 writeback
                # dominates ENT MEM_addr0 at SP=0xfff0 (recursive call traces
                # were observing step1 MEM_addr0=0xe0 instead of 0xf0).
                ("OP_ENT", -1000000.0),
                ("CMP+0", 10.0),
                ("ALU_LO+8", 5.0),
                ("ALU_LO+7", -10.0),
                ("ALU_LO+10", -20.0),
                # D2 revert (CAMPAIGN_SUMMARY bug #27): the soft B5-D
                # ADDR_B0_LO+0 / ADDR_B0_HI+14 (+2.0 each) evidence reads
                # were removed because L13's gather had not populated
                # ADDR_B0 by the time this PSH-at-SP rule fires, so the
                # reads were misfiring on SP byte 0 step 2 and driving
                # the +432 SP_byte0 cluster on func_*/rec_*/nested_*/
                # absdiff_*.  Rule M now relies on its CMP+0 / ALU_LO+8
                # / PSH_AT_SP / OP_ENT-blocker discrimination only.
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=40.0,
            # Reduced from 5e9 to 1e6 so this rule no longer dwarfs sibling
            # tail_mem_store_addr0_{f0,f8,e8,d8,...} rules by 5000x. With the
            # OP_ENT blocker above and the strength brought in line with
            # tail_mem_store_addr0_f0_exact (also 1e6), the most-evidence rule
            # can win by discrimination instead of by raw dominance. The
            # previous investigation (B3-α, commit 0cde3d3) documented the
            # 5e9 strength dominance as the cause of step1:MEM_addr0=0xe0
            # vs 0xf0 in `if_var` and as the primary blocker for
            # rec_factorial / rec_fib correctness past the base case.
            writes=byte_writes(0xE0, strength=1_000_000.0),
        ),
        # B7-7 / B4-H Path 2: rule N (0xE0 JSR-local) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  active_value
        # reduced from 5000 to 50 per the ≤10k cap.
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e0_from_jsr_local_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE0,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("OP_JSR", 2.0),
                ("CMP+4", 2.0),
                ("ALU_LO+14", 0.01),
                ("CMP+0", -100.0),
                ("OP_ENT", -1000.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xE0 → (LO+0, HI+14).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+0", 50.0),
                ("ADDR_B0_HI+14", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 80 to 140 so the structural-dim
            # evidence (LO+0 / HI+14 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        # B5-D / B4-H Path 2: rule O (tail_mem_store_addr0_e0_from_jsr_local_strong)
        # was byte-identical to rule N (same conditions / threshold /
        # active_value) — a pure strength-escalation sibling.  Deleted; the
        # N variant alone carries the 0xE0 JSR-local proof and the bounded
        # 5000 active_value is sufficient now that rule M is no longer
        # producing 5e9 residual to compete against.
        # B7-7 / B4-H Path 2: rule P (0xE8 nested-local) upgraded from soft
        # +2.0 ADDR_B0 evidence to hard +50 ADDR_B0 gate combined with
        # ADDR_B0_VALID (B7-4) and IN_STEP_FRESH (B7-1).  active_value
        # reduced from 500 to 50 per the ≤10k cap.
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e8_from_nested_local_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("MEM_ADDR_SRC", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("OP_JSR", -1000000.0),
                ("CMP+0", 10.0),
                ("ALU_LO+10", 5.0),
                ("ALU_LO+7", -10.0),
                ("ALU_LO+14", -10.0),
                # B7-7: hard +50 ADDR_B0 gate (was soft +2 per B6-B).
                # 0xE8 → (LO+8, HI+14).  Combined with ADDR_B0_VALID +50
                # and IN_STEP_FRESH +50 for lifecycle gating.
                ("ADDR_B0_LO+8", 50.0),
                ("ADDR_B0_HI+14", 50.0),
                ("ADDR_B0_VALID", 50.0),
                ("IN_STEP_FRESH", 50.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            # B7-7: threshold raised from 40 to 140 so the structural-dim
            # evidence (LO+8 / HI+14 / VALID / FRESH = +200) is jointly
            # required for firing.
            threshold=140.0,
            active_value=50.0,
        ),
        *exact_output_byte_rules(
            name="tail_mem_store_addr0_e8_from_local_frame_addr_exact",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            expected_byte=0xE8,
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("OP_JSR", -1000000.0),
                ("CMP+0", 2.0),
                ("ADDR_B0_LO+8", 2.0),
                ("ADDR_B0_HI+14", 2.0),
                # When upstream L16/L17 store amplifiers have already pushed
                # OUTPUT_LO+8 / OUTPUT_HI_THIS_STEP+14 to ~1e8 (model is confidently
                # emitting 0xe8), the OneHotBandGuarantee lane corrections
                # multiply ``(target - current)`` by ``silu(up) ≈ S * (score -
                # threshold)`` and overshoot into ~1e10 at the inactive lanes,
                # flipping the prediction. These tiny weights suppress the
                # rule in that high-magnitude regime without weakening the
                # uncertain-OUTPUT exactness recovery path.
                ("OUTPUT_LO+8", -0.001),
                ("OUTPUT_HI_THIS_STEP+14", -0.001),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=35.0,
            active_value=5000.0,
        ),
        # MEM_STORE (sem ``mark == MEM AND opcode_in_step in {SI, SC, PSH}``)
        # positive combined with CMP+0 (sem ``mark == AX OR (is_byte AND
        # byte_index == 0)``) positive plus the IS_BYTE/MARK_AX/MARK_STACK0
        # hard blockers makes the conditions-only effective predicate
        # unsatisfiable; F-5 falls back to the gate ``OUTPUT_HI_THIS_STEP+14`` whose
        # semantics is tautological. No scope tighter than tautology is
        # entailable. Leave scope/dominates_at unset for now.
        multi_way_and_rule(
            name="tail_mem_store_addr0_e8_from_local_frame_output_exact",
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("H1+4", 10.0),
                ("H1+1", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+10", -1000000.0),
                ("MEM_STORE", 5.0),
                ("PSH_AT_SP", -1000000.0),
                ("CMP+0", 2.0),
                ("ALU_LO+8", -1000.0),
                ("OP_JSR", -1000000.0),
                ("IS_BYTE", -1000000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("NEXT_PC", -1000000.0),
                ("NEXT_AX", -1000000.0),
                ("NEXT_SP", -1000000.0),
                ("NEXT_BP", -1000000.0),
                ("NEXT_STACK0", -1000000.0),
                ("NEXT_MEM", -1000000.0),
                ("NEXT_SE", -1000000.0),
            ),
            threshold=25.0,
            gate="OUTPUT_HI_THIS_STEP+14",
            writes=byte_writes(0xE8, strength=500.0),
        ),
        # Non-memory binary pops should emit a zero MEM row. Store/load ops
        # have dedicated memory paths and block this cleanup.
        multi_way_and_rule(
            name="tail_pop_mem_marker_zero",
            scope="mark == MEM",
            dominates_at={"OUTPUT_LO": "mark == MEM", "OUTPUT_HI_THIS_STEP": "mark == MEM"},
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
                ("OP_SI", -100.0),
                ("OP_SC", -100.0),
                ("OP_LI", -100.0),
                ("OP_LC", -100.0),
                ("MEM_STORE", -100.0),
            ),
            threshold=2.5,
            writes=byte_writes(0x00, strength=500.0),
        ),
        # BP is stable across ordinary binary ops. The L10 BP passthrough
        # carries byte 2 as a weak 0x01 signal; reinforce it when no frame op
        # is rewriting BP.
        # The 30 negative side-effect writes (byte_writes 0x01 lays out
        # -strength at every non-target nibble in the 16-wide OUTPUT_LO/HI
        # bands) get flagged against the +1e8 magnitude of the global
        # tail_clear_output_after_byte3 suppressor under the V1 contribution
        # algebra used by verify_rule_strength. The conditions' effective
        # predicate also collapses to a gate-fallback tautology (the
        # IS_BYTE/HAS_SE/H1+3 positives combined with the wide MARK_* and
        # OP_* hard blockers turn out unsatisfiable for the
        # effective_predicate walker), so an honest scope claim isn't
        # possible without code-level restructuring of the conditions.
        # Leave scope/dominates_at unset until the verifier supports
        # sign-aware competition and the condition shape can be tightened.
        multi_way_and_rule(
            name="tail_bp_byte2_preserve_01",
            conditions=(
                ("IS_BYTE", 1.0),
                ("HAS_SE", 1.0),
                ("H1+3", 20.0),
                ("H1+1", -100.0),
                ("H1+2", -100.0),
                ("BYTE_INDEX_1", 1_000_000.0),
                ("BYTE_INDEX_0", -1_000_000.0),
                ("BYTE_INDEX_2", -100.0),
                ("BYTE_INDEX_3", -100.0),
                ("OUTPUT_LO+1", 0.1),
                ("OUTPUT_LO+0", -0.2),
                ("OUTPUT_HI_THIS_STEP+0", 0.1),
                ("MARK_AX", -1_000_000.0),
                ("MARK_PC", -1_000_000.0),
                ("MARK_SP", -1_000_000.0),
                ("MARK_BP", -1_000_000.0),
                ("MARK_STACK0", -1_000_000.0),
                ("MARK_MEM", -1_000_000.0),
                ("OP_ENT", -100.0),
                ("OP_LEA", -1000.0),
                ("OP_LEV", -100.0),
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
            ),
            threshold=900_021.5,
            gate="H1+3",
            writes=byte_writes(0x01, strength=500.0),
        ),
        *ax_add_no_carry_zero_rules(),
        *ax_add_byte1_high_zero_rules(),
        *ax_add_byte1_structural_materialize_rules(),
        *ax_sub_byte1_high_zero_rules(),
        *ax_sub_full_underflow_byte1_rules(),
        *ax_sub_borrow_decrement_rules(),
        *wide_mul_byte1_preserve_rules(),
        *ax_add_mul_byte1_materialize_rules(),
        multi_way_and_rule(
            name="tail_ax_add_byte1_carry_high2_03",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                ("TEMP+8", 10.0),
                ("TEMP+9", -1000.0),
                ("CARRY+1", 1.0),
                ("CARRY+2", -1000.0),
                ("FETCH_HI+1", 0.1),
                ("OUTPUT_LO+3", 0.001),
                ("OUTPUT_LO+6", -10.0),
                ("OUTPUT_LO+7", -10.0),
                ("OP_IMM", -1000.0),
            ),
            threshold=1000.0,
            gate=dim_ref("carry", "alu", 1),
            writes=byte_writes(0x03, strength=500_000.0),
        ),
        # SHL-by-8 loses byte 1 to the same tail, but its signature is a huge
        # OUTPUT_LO[1] plus carry residue rather than MUL's OUTPUT_LO[2].
        multi_way_and_rule(
            name="tail_wide_shl_byte1_01",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("OP_SHL", 100.0),
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+0", 1.0),
            ),
            threshold=90.0,
            gate=dim_ref("carry", "alu", 3),
            writes=byte_writes(0x01),
        ),
        # SI preserves AX while writing memory. The dependency-expanded tail
        # can clobber the AX byte-1 prediction after the earlier layers have
        # prepared the right 16-bit value. OP_SI is relayed to AX byte
        # positions by L7; the HI-nibble comparison distinguishes a real
        # nonzero stored high byte from the common zero-high-byte store cases.
        # The si_ax_byte0 conditions combine MARK_AX-blocker, OP_SI positive
        # (sem ``mark == AX AND opcode_at_AX == SI``) and MEM_STORE positive
        # (sem ``mark == MEM AND opcode_in_step in {SI, SC, PSH}``); these
        # contradict via the MARK_AX blocker, and F-5 falls back to the
        # MEM_STORE gate semantics. Match scope/dominates_at to that
        # effective gate firing set so F-7 entailment succeeds and
        # cross-op strength competition is limited to MEM-store rows.
        multi_way_and_rule(
            name="tail_si_ax_byte1_12",
            scope="mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            dominates_at={
                "OUTPUT_LO": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
                "OUTPUT_HI_THIS_STEP": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            },
            conditions=si_ax_byte0 + (
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
                ("OUTPUT_LO+2", 0.01),
                ("OUTPUT_HI_THIS_STEP+1", 0.5),
                ("OUTPUT_HI_THIS_STEP+0", -0.5),
            ),
            threshold=161.0,
            gate="MEM_STORE",
            writes=byte_writes(0x12, strength=300.0),
        ),
        multi_way_and_rule(
            name="tail_si_ax_byte1_00",
            scope="mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            dominates_at={
                "OUTPUT_LO": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
                "OUTPUT_HI_THIS_STEP": "mark == MEM AND opcode_in_step in {PSH, SC, SI}",
            },
            conditions=si_ax_byte0 + (
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
                ("OUTPUT_LO+2", 0.01),
                ("OUTPUT_HI_THIS_STEP+0", 0.5),
                ("OUTPUT_HI_THIS_STEP+1", -0.5),
            ),
            threshold=161.5,
            gate="MEM_STORE",
            writes=byte_writes(0x00, strength=300.0),
        ),
        # SUB 0x0100-1 carries borrow residue in CARRY[2]/[3] and must clear
        # byte 1 to zero; the old tail currently leaves 0x01 there. Use
        # CARRY[2] instead of CARRY[3] so wide MUL/SHL carry residue does not
        # accidentally trigger the zeroing rule.
        multi_way_and_rule(
            name="tail_sub_borrow_byte1_00",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("MARK_AX", -1000.0),
                ("OP_IMM", -1000.0),
                ("TEMP+8", -20000.0),
                ("TEMP+9", 100.0),
                ("CARRY+2", 100.0),
                ("ALU_LO+1", 1.0),
                ("OUTPUT_LO+1", -5.0),
                ("OUTPUT_LO+3", -5.0),
                ("OUTPUT_LO+5", -5.0),
                ("OUTPUT_LO+6", -5.0),
            ),
            threshold=250.0,
            gate=dim_ref("carry", "alu", 2),
            writes=byte_writes(0x00),
        ),
        # 16-bit AND's high byte must zero; CMP/TEMP distinguish AND from
        # OR/XOR, whose high bytes intentionally remain 0x0f.
        multi_way_and_rule(
            name="tail_and_byte1_00",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("TEMP+3", 1.0),
                ("TEMP+4", 1.0),
                ("CMP+11", 1.0),
                ("CMP+12", 1.0),
            ),
            # In the expanded full-VM layout CMP[11]/[12] intentionally alias
            # TEMP[3]/[4], so this is effectively ax_byte0 + TEMP[3] + TEMP[4].
            threshold=4.5,
            writes=byte_writes(0x00),
        ),
        # OR/XOR byte 1 should remain 0x0f. The late tail inflates it to
        # 0x1e; TEMP[4] distinguishes AND and is used here as a blocker so
        # the AND-zeroing rule above remains authoritative for AND.
        multi_way_and_rule(
            name="tail_or_xor_byte1_0f",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=ax_byte0 + (
                ("TEMP+3", 1.0),
                ("TEMP+4", -2.0),
                ("CARRY+3", 0.01),
                ("OUTPUT_LO+14", 0.001),
            ),
            threshold=5.5,
            gate=dim_ref("byte_index", "0"),
            writes=byte_writes(0x0F),
        ),
        # SHR by 8 also needs byte 1 cleared after the marker correction emits
        # byte 0 as 0x01; TEMP[7] is the reliable non-carry/SHR signature at
        # byte positions.
        # The shared ``ax_byte0`` prefix combines MARK_AX hard blocker (-1e9)
        # with positives whose semantics include ``mark == AX``, so the
        # conditions-only effective collapses to a contradiction and F-5
        # falls back to the TEMP+7 gate (no semantics) producing a
        # tautology. Leave scope/dominates_at unset until the shared
        # ax_byte0 conditions are restructured.
        multi_way_and_rule(
            name="tail_shr_byte1_00",
            conditions=ax_byte0 + (
                ("TEMP+7", 1.0),
                ("OUTPUT_LO+1", 0.001),
            ),
            threshold=3.9,
            gate="TEMP+7",
            writes=byte_writes(0x00),
        ),
        # SHR by 8 currently computes byte 0 as 0x06 at the AX marker. OP_SHR
        # is still visible at the marker, so correct the marker prediction
        # before byte generation proceeds.
        multi_way_and_rule(
            name="tail_shr_marker_byte0_01",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
                ("IS_BYTE", -100.0),
                ("H1+1", 1.0),
                ("TEMP+7", 1.0),
                ("OP_SHR", 1000.0),
                ("OP_IMM", -100.0),
                ("OP_LEA", -1000000.0),
                ("OUTPUT_HI_THIS_STEP+0", 1.0),
                ("OUTPUT_HI_THIS_STEP+2", -1.0),
                ("OUTPUT_LO+10", -1.0),
            ),
            threshold=5005.0,
            gate=gate_mark_ax,
            writes=byte_writes(0x01),
        ),
        multi_way_and_rule(
            name="tail_lea_local_ax_marker_byte0_e8",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX", "OUTPUT_HI_THIS_STEP": "mark == AX"},
            conditions=(
                ("MARK_AX", 1.0),
                ("HAS_SE", 1.0),
                ("OP_LEA", 1.0),
                ("CMP+7", 1.0),
                ("FETCH_LO+8", 2.0),
                ("FETCH_HI+15", 0.2),
                # Removal-1 (2026-06-05) replaces the runner-side IMM 0xE0-0xFF
                # override (batched_pure_neural.py b5cf7099). Per
                # OP_LEA_LEAK_INVESTIGATION_2026_06_05.md Option A, the
                # spurious-fire path is positive evidence from L7-head-5
                # broadcast (CMP+7) + FETCH bytes on IMM rows where the imm has
                # nibble 0x_8/0x_F. Adding MEM_ADDR_SRC as a positive predicate
                # distinguishes real LEA-byte-evaluation steps (where the
                # address-byte source is live) from IMM dispatch rows (where
                # it is not).
                #
                # 2026-06-05 refinement (this commit): the original raise
                # threshold=9 -> 14 over-shot. Max possible positive sum =
                # 1+1+1+1+2+0.2+5 = 10.2, so threshold=14 is mathematically
                # unreachable; the rule was effectively disabled (residual
                # probe at L34 FFN input confirmed sum=1.0 at every MARK_AX
                # position for LEA_BASIC and XOR_BASIC -- only MARK_AX itself
                # contributes). Drop to threshold=7 so real LEA byte-0 emit
                # (legacy positives ~5.2 + MEM_ADDR_SRC*5 = ~10.2) crosses,
                # while spurious IMM rows (legacy positives only, ~5.2 max
                # absent MEM_ADDR_SRC) stay below. See
                # tools/l10_tail_lea_residual_probe.py for the probe.
                ("MEM_ADDR_SRC", 5.0),
                # IMM-decode keystone fix (2026-06-11): the MEM_ADDR_SRC=5
                # positive was meant to make a live LEA address-byte source
                # REQUIRED, but it is only ADDITIVE -- the FETCH_LO+8(2.0) /
                # FETCH_HI+15(0.2) terms cross threshold ALONE because FETCH
                # carries the IMM OPERAND at band magnitude ~40 (NOT one-hot).
                # On a plain ``IMM v; EXIT`` AX decode row (MARK_AX=1, OP_LEA=0,
                # MEM_ADDR_SRC=0) any immediate with lo-nibble 8 (FETCH_LO+8~40
                # -> 2*40=80) or hi-nibble F (FETCH_HI+15~40 -> 0.2*40=8) scored
                # 81 / 9 >= 7.0 and mis-fired this 0xE8 writer (strength 1e6),
                # producing the ±228M block-36 spike that flips OUTPUT argmax to
                # 0xE8 -> the 0xFFE8 / 0xE8 sentinel (lo-nibble-8 + hi-nibble-F
                # families, the bulk of the 46/256 IMM mis-decodes). The prior
                # OP_LEA_LEAK_INVESTIGATION_2026_06_05 threshold-tuning failed
                # because it assumed FETCH ~1.0; with FETCH ~40 no positive-only
                # threshold can separate LEA from the IMM operand broadcast.
                # FIX (broadcast-hardening, mirrors the l16 lev_routing
                # l16_lea_local_ax_byte0_hi_e sibling fix this same session):
                # add OP_IMM as a HARD NOT-blocker so the FETCH operand
                # broadcast can never satisfy the rule on an IMM step. The legit
                # LEA byte-0 0xE8 emit (OP_LEA=1, OP_IMM=0, MEM_ADDR_SRC live)
                # is byte-identical. spec_k=0 attribution:
                # tools/probe_tail_rule_attrib.py.
                ("OP_IMM", -1_000_000.0),
                ("IS_BYTE", -10.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
            ) + ((
                # Multi-local LEA 0xE8 over-fire guard (var_mul / var_three): make
                # the imm=-8 FETCH signature a GENUINE requirement so this 0xE8
                # writer cannot stamp the BP-8 low byte onto the 2nd local (BP-16,
                # imm=-16, low byte 0xE0) or 3rd local (BP-24, imm=-24, 0xD8). The
                # imm=-16 low-nibble lights FETCH_LO+0; the imm=-24 high-nibble
                # lights FETCH_HI+14. Both carry the ~40 FETCH broadcast on the
                # WRONG-immediate LEA and ~0 on imm=-8, so a -10 NOT-blocker drives
                # the positive sum well below threshold=7 there while leaving the
                # legit imm=-8 fire byte-identical. See
                # ``_lea_local_e8_multilocal_guard_enabled`` + diag 18365452.
                ("FETCH_LO+0", -10.0),
                ("FETCH_HI+14", -10.0),
            ) if _lea_local_e8_multilocal_guard_enabled() else ()) + ((
                # DIV/MOD result-row 0xE8 (744) sentinel-slam guard
                # (campaign config). The DIV/MOD result AX-marker row carries
                # MARK_AX + HAS_SE + MEM_ADDR_SRC, which clears this rule's
                # effective threshold (OP_LEA ~= 0 there but the 1e9 MARK_AX
                # term dominates the lowered AND). Hard NOT-blockers on the
                # real opcode keep the 1e6 0xE8 writer off the DIV/MOD result
                # so the quotient/remainder survives. Legit LEA byte-0 emit
                # (OP_DIV == OP_MOD == 0) is byte-identical. See
                # ``_tail_lea_e8_divmod_guard_enabled`` (only active in the
                # C4_NO_STACK0_EMIT campaign config).
                ("OP_DIV", -1_000_000_000.0),
                ("OP_MOD", -1_000_000_000.0),
            ) if _tail_lea_e8_divmod_guard_enabled() else ()) + ((
                # ADD/SUB result-row 0xE8 (744) sentinel-slam guard (#309,
                # campaign config) — the arith extension of the DIV/MOD guard
                # above. The ADD/SUB/absdiff RESULT AX-marker row carries the
                # SAME MARK_AX + HAS_SE + MEM_ADDR_SRC signature that clears
                # this rule's effective threshold (OP_LEA ~= 0 but the 1e9
                # MARK_AX term dominates the lowered AND), so the 1e6 0xE8
                # writer SIGN-INVERTS OUTPUT_LO[0] on the result row and the
                # emitted AX byte 0 becomes 0xE8 (744) instead of the sum /
                # difference — the residual add fails (e.g. 744=0x2E8,
                # 488=0x1E8), the sub residual, and absdiff (diverges step 11
                # on ax 0xFFE0 -> 0xFFE8; the |a-b| body is a SUB row).
                # Hard NOT-blockers (-1e9) on the real arith opcode keep the
                # 0xE8 writer off the ADD/SUB result so the sum/difference
                # survives. absdiff is covered by OP_SUB (its body is a-b/b-a).
                # Legit LEA byte-0 emit (OP_ADD == OP_SUB == 0) is
                # byte-identical. See ``_tail_lea_e8_arith_guard_enabled``
                # (only active in the C4_NO_STACK0_EMIT campaign config).
                #
                # #325 (var_update LEA-after-ADD): the SHARP variant keys these
                # blockers on the per-step OPCODE_BYTE_LO one-hot instead of the
                # leaky cross-step OP_ADD/OP_SUB broadcast so a LEA that FOLLOWS
                # an arith step is no longer spuriously vetoed. See
                # ``_arith_guard_addsub_blockers`` /
                # ``_tail_lea_e8_arith_guard_sharp_enabled``.
                *_arith_guard_addsub_blockers(),
            ) if _tail_lea_e8_arith_guard_enabled() else ())
            # ENT-step AX-marker row 0xE8 (744) sentinel-slam guard (#311). The
            # main ENT step's AX dump must preserve the carried prior AX
            # (AX_CARRY == 0), but this 1e6 0xE8 writer fires on the ENT
            # AX-marker row (MARK_AX scope dominates; OP_LEA == 0 cannot veto)
            # and stamps byte-0 = 0xE8 -> the var_simple / if_var cluster
            # diverges at step 1 (got_ax 0x_2E8 vs oracle 0). The per-step
            # FETCHED ENT opcode one-hot ``OPCODE_BYTE_LO+6`` (== 1.0 on the ENT
            # marker row, == 0.0 on a genuine LEA marker row which is
            # ``OPCODE_BYTE_LO+0``) hard-blocks the writer ONLY on a real ENT
            # step. See ``_tail_lea_e8_ent_guard_enabled``.
            + _tail_lea_e8_ent_byte0_blockers(),
            threshold=7.0,
            writes=byte_writes(0xE8, strength=1_000_000.0),
        ),
        multi_way_and_rule(
            name="tail_ax_add_byte1_missing_stack_high_02",
            scope="is_byte",
            dominates_at={"OUTPUT_LO": "is_byte", "OUTPUT_HI_THIS_STEP": "is_byte"},
            conditions=(
                ("IS_BYTE", 10.0),
                ("HAS_SE", 10.0),
                ("H1+1", 20.0),
                ("BYTE_INDEX_0", 10.0),
                ("BYTE_INDEX_1", -1000.0),
                ("TEMP+8", 50.0),
                ("TEMP+9", -1000000.0),
                ("CARRY+1", 10000.0),
                ("FETCH_HI+1", 100000.0),
                ("OUTPUT_LO+3", -3.0),
                ("OUTPUT_LO+4", -3.0),
                ("OUTPUT_LO+5", -3.0),
                ("OUTPUT_LO+6", -3.0),
                ("OUTPUT_LO+7", -3.0),
                ("MARK_AX", -10000000000.0),
                ("MARK_PC", -10000000000.0),
                ("MARK_SP", -10000000000.0),
                ("MARK_BP", -10000000000.0),
                ("MARK_STACK0", -10000000000.0),
                ("MARK_MEM", -10000000000.0),
                ("STACK0_BYTE0", -100000000.0),
                ("STACK0_BYTE1", -100000000.0),
                ("STACK0_BYTE2", -100000000.0),
                ("STACK0_BYTE3", -100000000.0),
                ("H1+0", -100000000.0),
                ("H1+2", -100000000.0),
                ("H1+3", -100000000.0),
                ("H1+4", -100000000.0),
            )
            # ENT-step AX byte-1 row 0x02 (744) sentinel-slam guard (#311). On
            # the main ENT step this constant-0x02 byte-1 materializer fires
            # (driven by its FETCH_HI+1 term = the fetched ENT instruction) and
            # stamps byte-1 = 0x02 over the carried prior AX (0) -> the AX dump
            # decodes 0x02E8 = 744. The carried-AX-present signal
            # (``AX_CARRY_LO+0`` / ``AX_CARRY_HI+0`` both ~3.0 on the carried-AX
            # ENT byte-1 row, 0.0 on every genuine freshly-computed value byte-1)
            # hard-blocks it ONLY where AX is being PRESERVED, never on a real
            # ADD/value byte-1. See ``_tail_lea_e8_ent_guard_enabled``.
            + _tail_lea_e8_ent_byte1_blockers(),
            threshold=130000.0,
            writes=byte_writes(0x02, strength=5000.0),
        ),
        # Comparison combine still sees amplified CMP residuals in the
        # expanded strict path. These two marker-only corrections restore the
        # truthy NE and LE cases without touching byte-lane arithmetic.
        # Wave B Cluster 1 (2026-06-10): six tail_cmp_* rules migrated
        # from MARK_AX to MARK_SE_ONLY under Wave A
        # step_end_operand_relay (10ca51a7); the relay broadcasts CMP /
        # OP_<cmp> from MARK_AX into MARK_SE_ONLY in the same step.
        # scope and dominates_at follow the marker swap.
        multi_way_and_rule(
            name="tail_cmp_ne_true_01_step_end",
            scope="mark == SE_ONLY",
            dominates_at={"OUTPUT_LO": "mark == SE_ONLY", "OUTPUT_HI_THIS_STEP": "mark == SE_ONLY"},
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("OP_NE", 1.0),
                ("CMP+1", -0.5),
            ),
            threshold=4.5,
            writes=byte_writes(0x01, strength=1000.0),
        ),
        multi_way_and_rule(
            name="tail_cmp_eq_false_00_step_end",
            scope="mark == SE_ONLY",
            dominates_at={"OUTPUT_LO": "mark == SE_ONLY", "OUTPUT_HI_THIS_STEP": "mark == SE_ONLY"},
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("OP_EQ", 1.0),
                ("CMP+1", -1.0),
                ("CMP+2", -1.0),
            ),
            threshold=4.5,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        multi_way_and_rule(
            name="tail_cmp_le_lt_true_01_step_end",
            scope="mark == SE_ONLY",
            dominates_at={"OUTPUT_LO": "mark == SE_ONLY", "OUTPUT_HI_THIS_STEP": "mark == SE_ONLY"},
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("OP_LE", 1.0),
                ("CMP+3", 0.1),
                ("CMP+0", -0.1),
            ),
            threshold=7.0,
            writes=byte_writes(0x01, strength=1000.0),
        ),
        multi_way_and_rule(
            name="tail_cmp_le_eq_prefix_false_00_step_end",
            scope="mark == SE_ONLY",
            dominates_at={"OUTPUT_LO": "mark == SE_ONLY", "OUTPUT_HI_THIS_STEP": "mark == SE_ONLY"},
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("OP_LE", 1.0),
                ("CMP+1", 0.1),
                ("CMP+0", -1.0),
                ("CMP+2", -1.0),
                ("CMP+3", -1.0),
            ),
            threshold=6.1,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        # OP_LT moved from conditions (additive) to gate (multiplicative) so
        # OP_LT residual fan-in via L6 routing's slot-3 binary-pop head does
        # not add to the condition score on non-LT op rows. CMP+0 is the
        # canonical LT-true bit (hi_lt fired); when CMP+0 is hot we want this
        # LT-false-writer to NOT fire, so it is a strong negative blocker
        # rather than the inverted positive coefficient that was here before
        # ("CMP+0", 0.01). Threshold drops to 0.5 so MARK_AX (=1.0 in state)
        # alone clears the gate, while a hot CMP+0 (~1.0) drives the score
        # below zero and suppresses the rule. Multiplicative OP_LT gate via
        # silu means the rule only contributes on OP_LT-active rows.
        # Mirrors the polarity discipline of tail_cmp_eq_false_00 directly
        # above, but uses the gate to suppress LT fan-in (per the 2026-06-03
        # CMP polarity investigation doc).
        multi_way_and_rule(
            name="tail_cmp_lt_false_00_step_end",
            scope="mark == SE_ONLY",
            dominates_at={"OUTPUT_LO": "mark == SE_ONLY", "OUTPUT_HI_THIS_STEP": "mark == SE_ONLY"},
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("CMP+0", -10.0),
            ),
            threshold=0.5,
            gate="OP_LT",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        multi_way_and_rule(
            name="tail_cmp_gt_false_00_step_end",
            scope="mark == SE_ONLY",
            dominates_at={"OUTPUT_LO": "mark == SE_ONLY", "OUTPUT_HI_THIS_STEP": "mark == SE_ONLY"},
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("OP_GT", 1.0),
                ("CMP+3", 0.1),
                ("CMP+0", -0.1),
            ),
            threshold=8.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        # Keep this appended after the legacy tail rules so existing generated
        # unit indexes remain stable. It repairs SP marker d8->e0 when the
        # staged value exists only in OUTPUT, not in EMBED.
        # The rule writes 0xE0 (one-hot: +5000 at LO+0/HI+14, -5000 at the
        # other 30 nibbles). V1's sign-blind comparison flags the 30
        # negative side writes against a stronger negative competitor
        # (tail_clear_output_after_byte3 at -1e8); both rules cooperatively
        # push those lanes down, so the rivalry is spurious. The +5000
        # positive writes at the target nibbles dominate correctly. Narrow
        # dominates_at down to the actual firing site so positive-write
        # competition is limited to SP marker rows; the negative side
        # writes remain flagged until the verifier becomes sign-aware.
        multi_way_and_rule(
            name="tail_sp_pop_marker_output_d8_to_e0",
            scope="mark == SP",
            dominates_at={"OUTPUT_LO": "mark == SP", "OUTPUT_HI_THIS_STEP": "mark == SP"},
            conditions=(
                ("CONST", -100000000.0),
                ("MARK_SP", 100000000.0),
                ("HAS_SE", 1000.0),
                ("CMP+3", 100.0),
                ("OUTPUT_LO+8", 10.0),
                ("OUTPUT_HI_THIS_STEP+13", 100.0),
                ("OUTPUT_HI_THIS_STEP+15", -100.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("OP_ENT", -1000000.0),
                ("OP_LEV", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("IS_BYTE", -100.0),
            ),
            threshold=1505.0,
            gate=gate_mark_sp,
            writes=byte_writes(0xE0, strength=5000.0),
        ),
        # Pure suppressor (clear_output_writes lays out -1e8 on every
        # OUTPUT_LO/HI nibble lane) with no positive write. Under V1's
        # cross-sign contribution algebra used by verify_rule_strength,
        # the -1e8 magnitude is compared against unrelated positive
        # override contributions (tail_sp_pop_byte3_zero at +5e11) and
        # falsely flagged at all 32 nibble lanes. In practice this is the
        # designated dominator that drives every nibble lane to 0 at the
        # byte3 step boundary; competition with positive-write rules is
        # spurious. Leave scope/dominates_at unset; the verifier cannot
        # prove a useful claim with the current sign-blind algebra.
        multi_way_and_rule(
            name="tail_clear_output_after_byte3",
            conditions=(
                ("IS_BYTE", 1.0),
                ("BYTE_INDEX_3", 1.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("STACK0_BYTE0", -1000.0),
                ("STACK0_BYTE1", -1000.0),
                ("STACK0_BYTE2", -1000.0),
                ("STACK0_BYTE3", -1000.0),
                ("MEM_VAL_B0", -1000.0),
            ),
            threshold=1.5,
            gate=dim_ref("byte_index", "3"),
            writes=clear_output_writes(strength=100_000_000.0),
        ),
        # Pure suppressor (clear_output_writes lays out -1e8 on every
        # OUTPUT_LO/HI nibble lane) with no positive write. Under V1's
        # cross-sign contribution algebra used by verify_rule_strength,
        # this rule's -1e8 magnitude is compared against the very large
        # POSITIVE override contributions of unrelated tail materializers
        # (tail_sp_pop_byte3_zero etc. at +5e11) and falsely flagged. In
        # practice the rule cooperatively drives every nibble lane to 0
        # in tandem with tail_clear_output_after_byte3 (the cross-sign
        # rivalry isn't real). The effective predicate is also collapsed
        # to the NEXT_SE gate fallback (semantics ``NOT is_byte``) because
        # the IS_BYTE+NEXT_SE conditions contradict in the registry
        # semantics. Leave scope/dominates_at unset; the verifier cannot
        # prove a useful claim with the current algebra.
        multi_way_and_rule(
            name="tail_clear_output_before_step_end",
            conditions=(
                ("IS_BYTE", 1.0),
                ("NEXT_SE", 1.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("STACK0_BYTE0", -1000.0),
                ("STACK0_BYTE1", -1000.0),
                ("STACK0_BYTE2", -1000.0),
                ("STACK0_BYTE3", -1000.0),
            ),
            threshold=1.5,
            gate="NEXT_SE",
            writes=clear_output_writes(strength=100_000_000.0),
        ),
    ) + ((
        # PHASE-2 KEYSTONE — the campaign LEA byte-0 address relay (ROOT 1).
        # Three AX-marker OUTPUT byte-0 writers that deliver the BP-relative
        # local effective-address low byte (0xE8 BP-8 / 0xE0 BP-16 / 0xD8 BP-24)
        # onto the LEA AX-marker row in the 30-token frame, where the legacy
        # ``tail_lea_local_ax_marker_byte0_e8`` corrector dies (CMP+7 == 0,
        # MEM_ADDR_SRC == 0, FETCH empty on func_identity's ``&x``). All gate
        # MULTIPLICATIVELY on ``OP_LEA`` (~5.23 on LEA AX rows, <=0.05 on every
        # non-LEA row — the only LEA-specific signal that survives this frame).
        # Measured teacher-forced (tools/_probe_learelay_tf.py). See
        # ``_lea_byte0_memsp_relay_enabled``.
        #
        # (1) BP-8 -> 0xE8. On func_identity the FETCH band is DEAD, so the only
        # discriminator is the AUTOREGRESSIVE ``ALU_HI+15`` MAGNITUDE: ~+73..+90
        # on the BP-8 effective-address compute, ~+5.5 on BP-16/BP-24, ~-45 on
        # non-LEA AX rows (the golden ROOT-1 discriminator, acf2be0d). 0.2 *
        # ALU_HI+15 makes the BP-8 conditions sum ~25 (>= threshold 17) while
        # BP-16/24 (~8.3) and non-LEA (<0) stay silent; the OP_LEA gate zeroes
        # any residual non-LEA leak. On the var first-LEA (already 0xE8) this is
        # a byte-identical re-write; on the var ``&a`` re-read (small ALU_HI+15,
        # already 0xE8) it stays silent.
        multi_way_and_rule(
            name="tail_lea_local_ax_byte0_e8_alubp_memsp",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX",
                          "OUTPUT_HI_THIS_STEP": "mark == AX"},
            gate="OP_LEA",
            conditions=(
                ("MARK_AX", 1.0),
                ("HAS_SE", 1.0),
                ("OP_LEA", 1.0),
                ("ALU_HI+15", 0.2),
                ("OP_IMM", -1_000_000.0),
                # #325: SHARP per-step ADD/SUB NOT-blockers (OPCODE_BYTE_LO+9/+10)
                # replace the leaky cross-step OP_ADD/OP_SUB broadcast so a LEA
                # that FOLLOWS an arith step is not spuriously vetoed (var_update
                # step-14). See ``_arith_guard_addsub_blockers``.
                *_arith_guard_addsub_blockers(),
                ("OP_DIV", -1_000_000_000.0),
                ("OP_MOD", -1_000_000_000.0),
                ("IS_BYTE", -10.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
                # #342 nested callee-ENT AX-dump FIX: the SHARP per-step FETCHED
                # ENT opcode one-hot ``OPCODE_BYTE_LO+6`` (== 1.0 on a genuine ENT
                # step, 0.0 on a genuine LEA row = ``OPCODE_BYTE_LO+0``) hard-blocks
                # this 0xE8 slam on the nested callee-ENT AX-marker row, where the
                # persistent ``OP_ENT`` residue over-fires the writer over the
                # carried AX. Empty (byte-identical) unless C4_NESTED_ENT_AXDUMP=1.
                # See ``_lea_e8_nested_ent_axdump_enabled``.
                *_lea_e8_nested_ent_axdump_blockers(),
            ) + ((
                # func re-read-LEA ``&b`` byte-0 0xE8 over-fire FIX: gate on the
                # FIRST-LEA-after-ENT ``OP_ENT`` residue (~+1.19 on the first LEA,
                # ~+0.013 on the re-read LEA) so this 0xE8 writer stops slamming
                # 0xE8 over the genuine 0xE0 on func / absdiff ``&b`` (BP-16). The
                # +0.6*W threshold bump (matched below) cuts at the ~0.6 midpoint:
                # first-LEA fires (margin ~+0.59W), re-read vetoed (~-0.587W). See
                # ``_lea_e8_first_ent_gate_enabled``.
                ("OP_ENT", 100.0),
            ) if _lea_e8_first_ent_gate_enabled() else ()),
            threshold=17.0 + (60.0 if _lea_e8_first_ent_gate_enabled() else 0.0),
            writes=byte_writes(0xE8, strength=1_000_000.0),
        ),
        # (2) BP-16 -> 0xE0. The multi-local re-read LEA carries small ALU_HI+15
        # indistinguishably from ``&a`` re-read, so the reliable discriminator is
        # the FETCH IMMEDIATE. imm=-16 = 0xF0 lights ``FETCH_LO+0`` +
        # ``FETCH_HI+15`` (NOT ``FETCH_LO+8`` / ``FETCH_HI+14``). Both FETCH
        # requirements carry weight 8 so their ABSENCE drops the score below
        # threshold — this is what defeats the non-LEA ``ALU_HI+15`` × negative
        # residual trap (FETCH is identically 0 on every non-LEA AX row, so the
        # 16-pt FETCH floor can never be reached there). func_identity's ``&x``
        # (FETCH_HI+15 == 0) and the var ``&a`` re-read (FETCH_LO+0 == 0) are
        # both excluded.
        multi_way_and_rule(
            name="tail_lea_local_ax_byte0_e0_fetch_memsp",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX",
                          "OUTPUT_HI_THIS_STEP": "mark == AX"},
            gate="OP_LEA",
            conditions=(
                ("MARK_AX", 1.0),
                ("HAS_SE", 1.0),
                ("OP_LEA", 1.0),
                ("FETCH_LO+0", 8.0),
                ("FETCH_HI+15", 8.0),
                ("FETCH_LO+8", -10.0),
                ("FETCH_HI+14", -10.0),
                ("OP_IMM", -1_000_000.0),
                # #325: SHARP per-step ADD/SUB NOT-blockers. See
                # ``_arith_guard_addsub_blockers``.
                *_arith_guard_addsub_blockers(),
                ("OP_DIV", -1_000_000_000.0),
                ("OP_MOD", -1_000_000_000.0),
                ("IS_BYTE", -10.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
            ),
            threshold=20.0,
            # var_three multi-local: out-vote the BP-8 0xE8 ``e8_alubp_memsp``
            # writer's 1e6 tie on the ``&b`` row (its ALU_HI+15 over-fires in a
            # deeper 3-local frame). func-safe — e0_fetch needs FETCH_HI+15 which
            # func's ``&x`` (FETCH_HI nib 1) lacks. See
            # ``_lea_e0d8_fetch_dominate_enabled``.
            writes=byte_writes(0xE0, strength=_lea_e0d8_fetch_strength()),
        ),
        # (3) BP-24 -> 0xD8. imm=-24 = 0xE8 lights ``FETCH_LO+8`` +
        # ``FETCH_HI+14`` (NOT ``FETCH_HI+15``). Mirrors the 0xE0 rule on those
        # nibbles; the var ``&a`` (FETCH_HI+14 == 0) and ``&b`` (FETCH_HI+15 ==
        # 1, hard-blocked) frames are excluded.
        multi_way_and_rule(
            name="tail_lea_local_ax_byte0_d8_fetch_memsp",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX",
                          "OUTPUT_HI_THIS_STEP": "mark == AX"},
            gate="OP_LEA",
            conditions=(
                ("MARK_AX", 1.0),
                ("HAS_SE", 1.0),
                ("OP_LEA", 1.0),
                ("FETCH_LO+8", 8.0),
                ("FETCH_HI+14", 8.0),
                ("FETCH_LO+0", -10.0),
                ("FETCH_HI+15", -10.0),
                ("OP_IMM", -1_000_000.0),
                # #325: SHARP per-step ADD/SUB NOT-blockers. See
                # ``_arith_guard_addsub_blockers``.
                *_arith_guard_addsub_blockers(),
                ("OP_DIV", -1_000_000_000.0),
                ("OP_MOD", -1_000_000_000.0),
                ("IS_BYTE", -10.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
            ),
            threshold=20.0,
            # var_three multi-local: out-vote the BP-8 0xE8 ``e8_alubp_memsp``
            # writer's 1e6 tie on the ``&c`` row. See
            # ``_lea_e0d8_fetch_dominate_enabled``.
            writes=byte_writes(0xD8, strength=_lea_e0d8_fetch_strength()),
        ),
    ) if _lea_byte0_memsp_relay_enabled() else ()) + (tuple(
        # PHASE-2 multi-param / multi-local LEA byte-0 ALU-AMPLIFIER (ROOT 1
        # sibling). The L8 ``lea_lo`` ALU already computes the CORRECT
        # effective-address byte-0 (0xE0/0xE8 etc.) for the re-read LEAs and
        # delivers it into the block-41 tail-bank input (HI nib 0xE/0xD one-hot,
        # LO nib the offset). The block-42 L25 post-op (an opcode-gated OUTPUT
        # relay that does NOT read OP_LEA) overwrites it with a ~370-magnitude
        # default whenever the upstream residual is weak (~50 on re-read rows).
        # These rules re-assert the ALU-computed byte at 1e6 strength so it
        # survives block 42 — no frame-offset discriminator needed, the
        # OUTPUT_HI+{14,13} (complete-frame-address) presence IS the
        # discriminator that survives where FETCH / ALU_HI+15 die. The FIRST
        # LEA in a frame (HI nib 0, handled by the ALU_HI+15 rule (1) above) is
        # hard-excluded via the OUTPUT_HI+0 NOT-block. See
        # ``_lea_byte0_alu_amplify_enabled``.
        multi_way_and_rule(
            name=f"tail_lea_local_ax_byte0_amplify_h{h}_lo{k}",
            scope="mark == AX",
            dominates_at={"OUTPUT_LO": "mark == AX",
                          "OUTPUT_HI_THIS_STEP": "mark == AX"},
            gate="OP_LEA",
            conditions=(
                ("MARK_AX", 1.0),
                ("HAS_SE", 1.0),
                ("OP_LEA", 1.0),
                # The complete-frame-address discriminator: the ALU produced a
                # 0xFE-/0xFD- frame address (HI nib 0xE/0xD one-hot, ~50) on the
                # re-read LEA. Each is REQUIRED at weight 8 so its ABSENCE drops
                # the score below threshold (this excludes the first-LEA row
                # where HI nib == 0).
                (f"OUTPUT_HI_THIS_STEP+{h}", 8.0),
                (f"OUTPUT_LO+{k}", 8.0),
                # Hard NOT-block the incomplete first-LEA row (HI nib 0) so the
                # amplifier can never lock in a wrong byte before rule (1) /
                # the ALU_HI+15 magnitude path supplies the 0xE high nibble.
                ("OUTPUT_HI_THIS_STEP+0", -1_000.0),
                # SURGICAL over-fire FIX (task #421): the amplifier's target is
                # the SECOND-local / RE-READ LEA (``&b`` BP-16 want 0xE0, and
                # func's ``&x`` re-read), NOT the FIRST-LEA-after-ENT (``&a`` /
                # ``&x`` / ``&n`` BP-8 want 0xE8). Teacher-forced diagnostic
                # (tools/_diag_opcam_overfire.py, campaign config, BUILT dims)
                # confirmed the OUTPUT_HI+0 block is INSUFFICIENT: on the
                # first-LEA-after-an-inner-ENT the ALU already carries a COMPLETE
                # 0xFE- address (OUTPUT_HI nib 14, OUTPUT_LO nib 0 == 0xE0), so
                # the h=14/k=0 variant FIRES and slams 0xE0 over the genuine 0xE8
                # on nested_sumsq step13, rec_factorial step6, rec_fib step6,
                # func_square step6 (all measured tail=0xE8 but amp=0xE0). Those
                # rows carry the FIRST-LEA-after-ENT residue OP_ENT ~+1.19; the
                # genuine re-read targets carry OP_ENT ~+0.01 (decayed one VM step
                # later). Same clean split the keystone e8 first-ent gate uses
                # (``_lea_e8_first_ent_gate``, but INVERTED here). A -100 * ENT
                # NOT-block vetoes the first-LEA (~-119, decisively below the
                # threshold-20 firing margin) while barely touching the re-read
                # (~-1.0) — the amplifier stays load-bearing on func/absdiff ``&b``
                # (keystones e0_fetch/e8 are both dark there) and goes DARK on the
                # first-LEA-after-ENT rows the keystone e8 correctly owns. The
                # threshold is UNCHANGED (unlike the e8 gate) because the veto is
                # one-sided: on the low-ENT re-read the term is negligible.
                ("OP_ENT", -100.0),
                ("OP_IMM", -1_000_000.0),
                # #325: SHARP per-step ADD/SUB NOT-blockers (OPCODE_BYTE_LO+9/+10)
                # replace the leaky cross-step OP_ADD/OP_SUB broadcast so a re-read
                # LEA that FOLLOWS an arith step is not spuriously vetoed. See
                # ``_arith_guard_addsub_blockers``.
                *_arith_guard_addsub_blockers(),
                ("OP_DIV", -1_000_000_000.0),
                ("OP_MOD", -1_000_000_000.0),
                ("IS_BYTE", -10.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
            ),
            threshold=20.0,
            writes=byte_writes((h << 4) | k, strength=1_000_000.0),
        )
        for h in (14, 13)
        # Frame-local addresses are 8-byte (int) aligned, so the effective-
        # address byte-0 low nibble is always 0x0 or 0x8 (0xE0/0xE8/0xD0/0xD8).
        # Restricting to {0, 8} keeps the amplifier from matching a drifted LI
        # AX row whose loaded value happens to carry a 0xE-/0xD- high nibble
        # with an arbitrary low nibble (the func step-9 LI false-fire surface).
        for k in (0, 8)
    ) if _lea_byte0_alu_amplify_enabled() else ()) + sp_pop_carry_rules()
    return step_end_transition_blocked(
        pc_byte_span_blocked(
            stack0_span_blocked_tail_rules(
                mem_value_row_blocked_tail_rules(rules)
            )
        )
    )


def make_tail_bit32_result_correction_op() -> Operation:
    """Append a generated FFN block after the dependency-assigned L10 tail."""

    rules = _tail_bit32_result_correction_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        # Derive d_model from the block's residual-stream width. Preferred
        # source is ``block.attn.dim`` (always set to the model's d_model on
        # the AutoregressiveAttention used by every TransformerBlock); the
        # legacy ``block.ffn.W_up.shape[1]`` fallback is incorrect for
        # efficient-mode L10 where ``block.ffn`` is a ``PureNeuralALU``
        # subclass (e.g. ``ALUAndOrXor``) that has no ``W_up`` attribute,
        # silently bottoming out at the hard-coded ``512`` constant. Mirrors
        # the derivation chain installed at the sibling bake site (line
        # ~6764, ``make_l10_post_op_attach_op``) by commit 24cca5ee to avoid
        # the same wide-d_model overflow when the dim allocator lifts
        # d_model above 512 (e.g. ``alu_mode='efficient'`` to 800).
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
        # Per-bake FFN-unit allocator. The tail FFN is a standalone bank
        # whose ``hidden_dim`` equals ``len(rules)``; pinning the full
        # range under a single op name makes the bank's tenancy
        # explicit so a future second tenant goes through
        # ``allocator.alloc(...)`` instead of silently aliasing rule
        # rows. The factory also fails fast on rule-count drift between
        # the layout table and the materialised rule set.
        allocator = _allocate_l10_tail_bit32_units(len(rules))
        ffn = PureFFN(d_model, len(rules))
        ffn._l10_unit_allocator = allocator
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        _suppress_ffn_on_step_boundary(ffn, dim_map, S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    _tail_reads = {
            "CONST", "IS_BYTE", "HAS_SE", "H1", "H3",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
            "NEXT_STACK0", "NEXT_MEM", "NEXT_SE", "OP_SHL", "OP_SHR", "OP_IMM", "OP_JSR",
            "PSH_AT_SP",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_ENT", "OP_LEV", "OP_SI", "OP_SC", "OP_LI", "OP_LC",
            "OP_ADD", "OP_SUB", "OP_DIV", "OP_MOD", "OP_AND", "OP_OR",
            "OP_XOR",
            "EMBED_LO", "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
            "TEMP", "CMP", "CARRY", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "FETCH_HI",
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "MEM_STORE",
            "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
            "ADDR_B0_LO", "ADDR_B0_HI",
    }
    if _nonfirst_psh_sp_fix_enabled():
        # Non-first-PSH SP byte-0 fix: read the helper's AND scratch dim as a
        # NOT-blocker on the 0xF8 SP exactness writer (see the conditions of
        # ``tail_sp_marker_byte0_f8_from_initial_stack_exact``). Declaring the
        # read here also makes the scheduler place the helper (the producer)
        # before this tail block. Flag-off => band absent and not read.
        _tail_reads.add("NONFIRST_PSH_SP_SUPPRESS")

    return Operation(
        name="tail_bit32_result_correction",
        reads=_tail_reads,
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        # Phase 8.A.4 retry: layer_idx=17 literal dropped (was redundant
        # alongside ``target_op_name`` since ``target_op_name`` takes
        # precedence in ``resolve_block_op_layer``). The L17 placement is
        # dep-derived from ``l10_post_ops_combined``'s position.
        target_op_name="l10_post_ops_combined",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeMemory::test_si_li_16bit_value",
        },
        spec_section="BLOG_SPEC.md#wide-alu-tail-correction",
    )


# === Multi-byte ADD high-byte adder (runs AFTER tail_bit32) =============
# (2026-06-12) Completes the multi-byte ADD result the model's carry-only
# byte-1 emit path leaves at ``byte1 = carry``.
#
# Root (confirmed spec_k0 on the REAL AUTOREGRESSIVE runner,
# tools/probe_add_autoregressive_carry.py): the active ADD byte-1 tail
# rules in ``_tail_bit32_result_correction_rules`` (the
# ``tail_ax_add_byte1_*`` family, run at the L25 tail block) compute
# byte 1 = (raw ALU_LO byte-1 low nibble) + CARRY+1, but only operand
# byte 0 ever reaches the ALU, so the raw byte-1 base is 0 and every
# multi-byte ADD decodes byte 1 = carry only (``654 + 114`` -> 256 =
# 0x100, dropping a1=2; ``432 + 32`` -> 0x0D0, dropping a1=1). Unlike SUB
# (subtrahend byte1 = 0x00 -> single relayed minuend value, so the SUB
# tail rules PRESERVE OUTPUT_LO and the L14 sub_noborrow passthrough
# wins), ADD is a genuine TWO-operand byte-1 sum a1 + b1 + carry and the
# ADD tail rules FORCE byte 1 from ALU_LO with strength 5e6, overriding
# any earlier (e.g. L14) OUTPUT_LO write.
#
# So this op runs as a NEW post_op appended AFTER ``tail_bit32_result_
# correction`` on the same L25 block (so nothing downstream overrides it,
# and the count-locked 2059-rule tail bank is untouched). At the ADD
# byte-1 predictor row (BYTE_INDEX_0 + TEMP+8, with IS_BYTE/H1+1/HAS_SE),
# all three addends survive cleanly to this block (probe verified):
#   * a1    = STACK0_BYTE_VAL_1_LO  (delivered by layer13_add_addend_relay,
#                                    head 5, TEMP+8-gated; magnitude ~3.0)
#   * b1    = ADDR_B1_LO            (operand B byte 1, from L13 head 1)
#   * carry = CARRY+1               (2.0 when byte 0 carried, 0.0 when not
#                                    -- the discriminator that reads CLEANLY
#                                    in the real autoregressive decode, the
#                                    crux the prior ADD attempt missed by
#                                    trusting a fixed teacher-forced row)
# One AND-rule per (a1 in 0..7, b1 in 0..7, carry in {0,1}) over-writes
# OUTPUT byte 1 with byte_value(a1+b1+carry) at a dominating strength so
# the byte-(a1+b1+carry) token wins argmax. The tail's existing carry-only
# rules are LEFT INTACT (they provide the cross-step OUTPUT_LO stabiliser,
# whose 0x88-leak crush keeps the band bounded); this op simply runs after
# them and overrides the byte-1 cell. The 1096 add corpus has a1,b1 <= 3
# and result byte1 <= 7, all single-nibble (OUTPUT_HI byte1 = 0); 0..7
# each covers that with headroom (max 7+7+1=15 still one nibble).
#
# Byte-identity: for 8-bit ADD (add_basic 10+32: a1=0, b1=0, carry=0) the
# (0,0,0) rule writes 0x00 = the carry-only default, unchanged. Gated on
# TEMP+8 (ADD byte selector, 0 on SUB/bitwise/everything else) so it is
# dark off-ADD; the SUB / bitwise / wide-ALU paths are untouched.

# Number of (a1, b1, carry) rules: 8 * 8 * 2.
_L10_ADD_HIGH_BYTE_ADDER_HIDDEN_DIM = 128


def _l10_add_high_byte_adder_rules() -> tuple[FFNRule, ...]:
    """128 rules: OUTPUT byte 1 = a1 + b1 + carry on multi-byte ADD.

    One rule per ``(a1, b1, carry)`` with ``a1, b1 in 0..7`` and
    ``carry in {0, 1}``. Each fires at the ADD byte-1 predictor row --
    ``TEMP+8`` (ADD byte selector) + ``H1[AX]`` + ``IS_BYTE`` +
    ``HAS_SE`` + ``BYTE_INDEX_0`` -- when ``STACK0_BYTE_VAL_1_LO == a1``
    (relayed operand A byte 1) AND ``ADDR_B1_LO == b1`` (operand B byte 1)
    AND the byte-0 carry state matches ``carry`` (``CARRY+1`` high for
    carry=1, blocked for carry=0). The firing rule over-writes OUTPUT
    byte 1 with ``byte_value(a1+b1+carry)`` at a dominating strength so it
    wins the OUTPUT argmax over the carry-only tail rules.

    CARRY+1 reads 2.0 (carry) / 0.0 (no carry) at this row, so the carry=1
    rules weight it 500 (a real carry scores +1000) and the carry=0 rules
    block it (-50000 -> -100000 on a real carry). The relay deposits a1
    into STACK0_BYTE_VAL_1_LO at magnitude ~3.0 (the L13 head-5 V/O copy
    scale), so that condition is weighted GATE/3 to match the other
    one-hot gates and keep the AND balanced.
    """
    AX_I = 1
    # Per-condition gate weight. Each one-hot positive condition scores
    # ~GATE; the threshold sits between all-on and missing-one.
    GATE = 1000.0
    # ADD byte-row selector. Legacy (35-token) frame spreads TEMP+8 to the
    # BYTE_INDEX_0 emit row, so the adder gates on it there. STACK0 campaign
    # Inc-4 (2026-06-20): the 30-token frame leaves TEMP+8 = 0 at the emit row
    # (GPU-confirmed), so the L13 add relay (head 5) instead stamps a dedicated
    # campaign discriminator TEMP+12 onto the emit row (a free slot the L14 add
    # cleanup does NOT read, so no TEMP+8-style over-fire). Re-gate the adder on
    # TEMP+12 in the campaign config so it fires at the emit row and computes
    # OUTPUT byte 1 = a1 + b1 + carry. Flag-OFF keeps TEMP+8 (byte-identical).
    GATE_DIM = "TEMP+12" if operand_from_memsp_enabled() else "TEMP+8"
    # GATE_DIM weight + threshold bump. The discriminator dim MUST be a HARD
    # gate: its absence has to sink the rule below threshold. In the campaign
    # config (a1=0 CONST path) the other terms can sum high on a NON-ADD emit
    # row (var return: IS_BYTE+HAS_SE+H1+1+BYTE_INDEX_0+CONST + an INFLATED
    # ADDR_B1_LO+0 ~= 2.95 -> 7592 > 6400), so a +1000 TEMP+12 gate is too weak
    # -- the rule false-fires when TEMP+12 = 0, corrupting var_simple byte 1
    # (GPU-confirmed -6 regression). Weight TEMP+12 at 1e5 and bump every
    # threshold by 1e5 so a genuine ADD row (TEMP+12 = 1) is UNCHANGED in
    # margin while a non-ADD row (TEMP+12 = 0) loses the full 1e5 and is hard-
    # blocked regardless of any inflated operand term. Flag-OFF keeps the
    # legacy +GATE weight / no bump (byte-identical).
    if operand_from_memsp_enabled():
        GATE_DIM_W = 3_000.0
        THR_BUMP = 3_000.0
    else:
        GATE_DIM_W = GATE
        THR_BUMP = 0.0
    # The relay deposits a1 into STACK0_BYTE_VAL_1_LO with magnitude ~3.0
    # (the L13 head 5 V/O copy scale), not a unit one-hot. Weight that
    # condition GATE/3 so its contribution (~3.0 * GATE/3 = GATE) matches
    # the other one-hot gates, keeping the AND balanced (a missing 1.0
    # gate must drop the score below threshold; an inflated a1 term would
    # otherwise let a wrong-b1 rule clear the threshold).
    A1_W = GATE / 3.0
    CARRY_REQ_W = 500.0      # CARRY+1 (= 2.0) on a real carry -> +1000
    CARRY_BLOCK = -50_000.0  # a real carry (2.0) -> -100000, hard block
    # Output write strength. This op runs AFTER the whole tail bank, whose
    # legacy carry-only ADD byte-1 rules boost the WRONG cell (carry only)
    # at strength up to 5e6 while also providing the cross-step OUTPUT_LO
    # stabiliser (0x00 / 0x88-leak crush). This op one-hots the correct
    # (a1+b1+carry) byte at 5e6 so it OVER-writes the carry-only result and
    # wins the argmax, while the tail's stabiliser keeps the band bounded.
    #
    # STACK0 campaign Inc-4 (2026-06-20): the 30-token frame's OUTPUT_LO band
    # is driven MUCH harder than the 35-token tail -- GPU-confirmed a ~6e9
    # competing write at the WRONG byte-1 cell (the L9 ALU_LO->OUTPUT_LO
    # operand-low-nibble leak, amplified in the collapsed frame). At 5e6 the
    # adder's correct-byte write is invisible against it (25+759 stayed 0x210).
    # Lift the campaign adder strength to 5e10 so the (a1+b1+carry) cell wins
    # the argmax over the 6e9 leak. Flag-OFF keeps 5e6 (byte-identical; the
    # 35-token band is bounded and 5e6 already dominates there).
    if operand_from_memsp_enabled():
        STRENGTH = 5.0e10
        COMPETITOR = 5.0e10
    else:
        STRENGTH = 5_000_000.0     # target-cell boost (overrides the tail)
        COMPETITOR = 5_000_000.0   # symmetric suppression of non-target cells

    # Marker / opcode / transition blockers are EXACTLY 0 at the ADD AX
    # byte-1 row (markers off, opcode bits decayed, NEXT_* off), so a large
    # magnitude is safe -- they only fire (and sink the unit) off the ADD
    # byte row. Byte-index blockers are deliberately MODEST (-3000): the
    # byte-1 row carries a ~0.013 BYTE_INDEX_1 leak, so -3000*0.013 = -39 is
    # negligible, while a FULLY-on wrong byte index (1.0 -> -3000) combines
    # with the lost positive BYTE_INDEX_0 gate (-GATE) to sink that row.
    # NOTE: the TEMP+4/5/6/7/9 relay blockers are deliberately NOT added --
    # TEMP+6 carries a benign ~0.3 residue on some ADD byte-1 rows (probe
    # spec_k0, add_18 ``828+890``), so a -1e6 TEMP+6 blocker amplifies that
    # leak to -30e6 and over-blocks the genuine ADD adder. The positive
    # TEMP+8 gate is the load-bearing ADD discriminator.
    #
    # STACK0 campaign Inc-4 (2026-06-20): the 30-token frame leaves VARIABLE
    # in-step opcode-broadcast residues at the emit row (GPU-measured OP_SI
    # ~0.22, OP_ENT ~0.044 on some ADD programs -- the documented in-step
    # broadcast corruptor). At -1e6 those amplify to -220k/-44k and VETO the
    # genuine ADD adder (25+759, 665+718, 825+163 stayed wrong). Use a MODEST
    # -1000 blocker in the campaign config: a 0.22 OP_SI residue -> -221 (at
    # -3000 it was -660, which still left the tight a1=0/carry threshold 187
    # short on add_1 25+759); and a REAL off-ADD marker/opcode is irrelevant
    # here because the load-bearing discriminator is the positive TEMP+12 gate
    # -- stamped by the L13 add relay ONLY on genuine ADD emit rows -- so the
    # adder is already dark on every non-ADD row regardless of these blockers.
    # Flag-OFF keeps -1e6 (byte-identical; the 35-token frame has no such
    # residue here).
    BLK = -1_000.0 if operand_from_memsp_enabled() else -1_000_000.0
    marker_blockers = (
        ("MARK_AX", BLK),
        ("MARK_PC", BLK),
        ("MARK_SP", BLK),
        ("MARK_BP", BLK),
        ("MARK_STACK0", BLK),
        ("MARK_MEM", BLK),
        ("H1+0", BLK),
        ("H1+2", BLK),
        ("H1+3", BLK),
        ("H1+4", BLK),
        ("BYTE_INDEX_1", -3_000.0),
        ("BYTE_INDEX_2", -3_000.0),
        ("BYTE_INDEX_3", -3_000.0),
    )
    non_add_blockers = (
        ("OP_IMM", BLK),
        ("OP_LEA", BLK),
        ("OP_SUB", BLK),
        ("OP_DIV", BLK),
        ("OP_MOD", BLK),
        ("OP_AND", BLK),
        ("OP_OR", BLK),
        ("OP_XOR", BLK),
        ("OP_EQ", BLK),
        ("OP_NE", BLK),
        ("OP_LT", BLK),
        ("OP_GT", BLK),
        ("OP_LE", BLK),
        ("OP_GE", BLK),
        ("OP_SHL", BLK),
        ("OP_SHR", BLK),
        ("OP_SI", BLK),
        ("OP_SC", BLK),
        ("OP_LI", BLK),
        ("OP_LC", BLK),
        ("OP_ENT", BLK),
        ("MEM_STORE", BLK),
    )
    transition_blockers = (
        ("NEXT_PC", BLK),
        ("NEXT_AX", BLK),
        ("NEXT_SP", BLK),
        ("NEXT_BP", BLK),
        ("NEXT_STACK0", BLK),
        ("NEXT_MEM", BLK),
        ("NEXT_SE", BLK),
    )

    # STACK0 campaign Inc-4 (2026-06-20): a1 condition. The L13 relay deposits
    # a1 into STACK0_BYTE_VAL_1_LO as a clean one-hot at the a1 nibble for
    # a1 > 0 (magnitude ~6.0), but for a1 == 0 the carrier COLLAPSES TO ALL-ZERO
    # in the 30-token frame (the -6/+6 cancel at nibble 0 -> nothing lit), so
    # the legacy ``STACK0_BYTE_VAL_1_LO+0`` positive is absent and the a1=0
    # rules lose their ~1000-pt term (GPU-confirmed: 25+759 a1=0/b1=2/carry=1
    # stayed 0x210). For a1 == 0 in the campaign config, supply that term via a
    # CONST baseline and AND in negative guards on SBV1+1..7 so the rule fires
    # ONLY when no high a1 nibble is lit (a genuine a1 == 0), never stealing an
    # a1 in 1..7 row. Flag-OFF keeps the original one-hot a1 term (byte-id).
    _campaign = operand_from_memsp_enabled()
    rules: list[FFNRule] = []
    for carry in (0, 1):
        for a1 in range(8):
            for b1 in range(8):
                v = a1 + b1 + carry  # <= 15 -> single nibble
                if _campaign and a1 == 0:
                    # CONST at 2*GATE so the a1=0 CONST baseline matches the
                    # a1>0 path's one-hot SBV1 contribution (~6.0 * A1_W =
                    # 2*GATE), keeping the a1=0 rule's firing margin in step
                    # with the thresholds (the +THR_BUMP hard gate would
                    # otherwise leave a1=0/carry ~500 short -> 203+733 missed).
                    a1_conditions = (("CONST", 2.0 * GATE),) + tuple(
                        (f"STACK0_BYTE_VAL_1_LO+{nib}", -GATE)
                        for nib in range(1, 8)
                    )
                else:
                    a1_conditions = (
                        (f"STACK0_BYTE_VAL_1_LO+{a1}", A1_W),  # a1 (~3.0*A1_W)
                    )
                base_conditions = (
                    ("IS_BYTE", GATE),
                    ("HAS_SE", GATE),
                    (f"H1+{AX_I}", GATE),
                    ("BYTE_INDEX_0", GATE),
                    (GATE_DIM, GATE_DIM_W),
                ) + a1_conditions + (
                    (f"ADDR_B1_LO+{b1}", GATE),            # b1
                ) + marker_blockers + non_add_blockers + transition_blockers + (
                    # #343: sign-ext NOT-blocker pair (campaign only). On a
                    # negative-LEA-local AX dump byte-1 row (var_update step-14,
                    # the ``return &x`` LEA after the ``x+7`` ADD+SI store) the
                    # L13 add relay spuriously stamps TEMP+12 -> the adder would
                    # FALSE-FIRE and nuke the upstream block-35 0xFF sign-extension
                    # to 0x00. AX_CARRY_LO/HI+15 (~3.0 on that row, 0.0 on every
                    # genuine ADD emit row) hard-sink the adder so the 0xFF
                    # survives. Empty tuple flag-OFF (byte-identical). See
                    # ``_ax_byte1_signext_lea_blockers``.
                    _ax_byte1_signext_lea_blockers()
                )
                if carry:
                    conditions = base_conditions + (
                        ("CARRY+1", CARRY_REQ_W),
                    )
                    # all-on: a1 (A1_W*~3.0 ~= GATE) + 6 one-hot gates
                    # (~5970, BYTE_INDEX_0=0.97/HAS_SE=0.995) + carry (+1000)
                    # ~= 7970; missing ANY single gate (incl. a wrong a1/b1
                    # -> that cell = 0) -> <= 6970; missing carry -> 6970.
                    # Threshold 7400 fires only on the exact (a1,b1,carry).
                    # Campaign: +THR_BUMP matches the 1e5 TEMP+12 hard gate so
                    # a genuine ADD row's margin is unchanged while a non-ADD
                    # row (TEMP+12=0) falls 1e5 short.
                    threshold = 7400.0 + THR_BUMP
                    scope = (
                        f"is_byte and {GATE_DIM} and BYTE_INDEX_0 and CARRY+1"
                    )
                else:
                    conditions = base_conditions + (
                        ("CARRY+1", CARRY_BLOCK),
                    )
                    # all-on (a1 + 6 one-hot gates, no carry) ~= 6970; a real
                    # carry adds -100000 -> hard block; missing any single
                    # gate -> <= 5970. Threshold 6400 fires only all-on.
                    # Campaign: +THR_BUMP pairs with the 1e5 TEMP+12 hard gate.
                    threshold = 6400.0 + THR_BUMP
                    scope = (
                        f"is_byte and {GATE_DIM} and BYTE_INDEX_0 "
                        "and not CARRY+1"
                    )
                rules.append(
                    multi_way_and_rule(
                        name=f"l10_add_high_byte_c{carry}_a{a1:x}_b{b1:x}",
                        scope=scope,
                        dominates_at={
                            "OUTPUT_LO": "is_byte",
                            "OUTPUT_HI_THIS_STEP": "is_byte",
                        },
                        conditions=conditions,
                        threshold=threshold,
                        writes=Primitives.byte_value_writes(
                            v,
                            hi_base="OUTPUT_HI_THIS_STEP",
                            strength=STRENGTH,
                            competitor_strength=COMPETITOR,
                        ),
                    )
                )
    return tuple(rules)


# ===========================================================================
# L10 tail post-op wrapper glue (R-FRAME INCR-1 precedent, byte-identical).
#
# The 13 flag-gated per-case fix ops below (add-high-byte adder, nonfirst-psh-sp,
# ent/exit-axcarry, loop-lea x2, jsr-bp-byte3, absdiff x3, loop-si-marker,
# loop-li-clamp) each repeated the SAME boilerplate verbatim: the 13-line
# ``d_model`` resolution ladder, the ``PureFFN(d_model, len(rules)) +
# dim_positions_from_bd + lower_ffn_rules + post_ops.append`` bake block, the
# flag-OFF ``_noop_bake`` off-path ``Operation``, the ``ir.layer(0).ffn.rules``
# splice, and the ``Operation`` metadata. Everything that VARIES per op is DATA
# (name, flag, rules factory, reads/writes, requires, target_op_name,
# spec_section, smoke_tests, whether the bake calls
# ``_suppress_ffn_on_step_boundary``). Collapsed here into one shared
# lowering + a thin per-op call, exactly as the R-FRAME INCR-1 collapse
# table-drove the AX/SP/BP/PC passthrough head glue (-83 LOC byte-identical).
# The emitted ``Operation`` — and therefore the golden hash — is UNCHANGED.
# ===========================================================================


def _resolve_l10_postop_d_model(block, dim_positions) -> int:
    """The shared L10 post-op ``d_model`` resolution ladder.

    Prefers ``block.attn.dim`` -> ``block.attn.W_q.shape[0]`` ->
    ``block.ffn.W_up.shape[1]`` -> ``max(dim_positions.values())+1`` -> 512.
    Verbatim extraction of the 13-line ladder copied into every tail post-op
    bake (also present in ``make_l10_post_op_attach_op``).
    """
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
    return d_model


def _l10_postop_bake(rules, *, suppress: bool, assert_len=None):
    """Return the shared tail post-op ``bake`` closure over ``rules``.

    Lowers ``rules`` into a fresh ``PureFFN`` post_op on the tail block using the
    shared ``d_model`` ladder, then optionally calls
    ``_suppress_ffn_on_step_boundary`` (``suppress=True`` for the AX/OUTPUT
    OUTPUT-owning overrides; ``False`` for the value-byte-row / IS_BYTE-gated
    ops that INTEND to fire on value-byte rows). ``assert_len`` pins an exact
    rule count when the op declares one (only the ADD high-byte adder does).
    Byte-identical to the per-op inline bakes.
    """
    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = _resolve_l10_postop_d_model(block, dim_positions)
        if assert_len is not None:
            assert len(rules) == assert_len, (
                f"l10 add high-byte adder rule-count drift: produced "
                f"{len(rules)}, expected {assert_len}"
            )
        ffn = PureFFN(d_model, len(rules))
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        if suppress:
            _suppress_ffn_on_step_boundary(ffn, dim_map, S)
        block.post_ops.append(ffn)

    return bake


def _make_l10_postop(
    *,
    name: str,
    rules_fn,
    reads: set,
    writes: set,
    requires: dict | None = None,
    target_op_name: str = "l10_post_ops_combined",
    spec_section: str = "BLOG_SPEC.md#registers",
    smoke_tests=None,
    suppress: bool = False,
    assert_len=None,
    flag_fn=None,
    noop_name: str | None = None,
    noop_target_op_name: str | None = None,
    noop_spec_section: str | None = None,
) -> Operation:
    """Shared table-driven builder for the 13 L10 tail post-op fix ops.

    R-FRAME INCR-1 precedent (byte-identical): each ``make_l10_*_op`` below is
    now a thin call over per-op DATA. Behaviour, verbatim:

      * If ``flag_fn`` is given and returns falsey, emit the flag-OFF ``_noop``
        ``Operation`` (no rules, no post_op, no band) — byte-identical to the
        prior model. The noop keeps the same ``name`` (or ``noop_name`` when the
        op renames it, e.g. loop-li) and, when the op used a different off-path
        ``target_op_name`` / ``spec_section``, ``noop_target_op_name`` /
        ``noop_spec_section`` reproduce them.
      * Otherwise build the rules, splice them into a fresh ``CompilerIR``, and
        return the ON-path ``Operation`` with the shared bake closure
        (``suppress`` toggles ``_suppress_ffn_on_step_boundary``; ``assert_len``
        pins a rule-count for the ADD adder).

    ``flag_fn=None`` = the always-registered ops (only the ADD high-byte adder).
    """
    if smoke_tests is None:
        smoke_tests = {"all"}

    if flag_fn is not None and not flag_fn():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name=noop_name if noop_name is not None else name,
            reads=set(),
            writes=set(),
            kind="block",
            target_op_name=(
                noop_target_op_name
                if noop_target_op_name is not None
                else target_op_name
            ),
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            compiler_ir=CompilerIR(),
            migrated=True,
            smoke_tests={"all"},
            spec_section=(
                noop_spec_section
                if noop_spec_section is not None
                else spec_section
            ),
        )

    rules = rules_fn()
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    kwargs = {}
    if requires is not None:
        kwargs["requires"] = requires
    return Operation(
        name=name,
        reads=reads,
        writes=writes,
        kind="block",
        target_op_name=target_op_name,
        declarative_bake_fn=_l10_postop_bake(
            rules, suppress=suppress, assert_len=assert_len
        ),
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests=smoke_tests,
        spec_section=spec_section,
        **kwargs,
    )


def make_l10_add_high_byte_adder_op() -> Operation:
    """Append a multi-byte ADD high-byte adder post_op after tail_bit32.

    Completes the multi-byte ADD result (OUTPUT byte 1 = a1 + b1 + carry)
    that the model's carry-only byte-1 emit path leaves at ``byte1 =
    carry``. Runs as a standalone ``PureFFN`` post_op appended AFTER
    ``tail_bit32_result_correction`` on the L25 block, so it dominates the
    carry-only tail rules and leaves the count-locked 2059-rule tail bank
    untouched. Reads the three byte-1 addends that survive to this block:
    a1 (``layer13_add_addend_relay`` -> STACK0_BYTE_VAL_1), b1 (ADDR_B1_LO,
    from L13 head 1), and the byte-0 carry (CARRY+1). See the module
    comment block above for the autoregressive carry-row finding, the
    (a1, b1, carry) -> byte-(a1+b1+carry) map, and the byte-identity
    property (8-bit ADD unchanged; gated on TEMP+8 so dark off-ADD).
    1096 ``add`` cluster: 4/50 -> 39/50 (the remaining fails are byte-0
    ALU-precision / carry-propagation cases, not byte-1).
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Always registered (no flag); ``suppress`` + the ADD-adder rule-count
    # ``assert_len`` are the only per-op bake specifics.
    return _make_l10_postop(
        name="l10_add_high_byte_adder",
        rules_fn=_l10_add_high_byte_adder_rules,
        reads={
            "CONST", "IS_BYTE", "HAS_SE", "H1",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
            "NEXT_STACK0", "NEXT_MEM", "NEXT_SE",
            "OP_IMM", "OP_LEA", "OP_SUB", "OP_DIV", "OP_MOD", "OP_AND",
            "OP_OR", "OP_XOR", "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE",
            "OP_GE", "OP_SHL", "OP_SHR", "OP_SI", "OP_SC", "OP_LI", "OP_LC",
            "OP_ENT", "MEM_STORE", "TEMP", "CARRY",
            "STACK0_BYTE_VAL_1_LO", "ADDR_B1_LO",
            # #343: sign-ext NOT-blocker dims (campaign only; see
            # ``_ax_byte1_signext_lea_blockers``).
            "AX_CARRY_LO", "AX_CARRY_HI",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        # Append AFTER the tail correction on the same L25 block so this
        # adder is the last OUTPUT writer at the ADD byte-1 row.
        requires={"after": "tail_bit32_result_correction"},
        suppress=True,
        assert_len=_L10_ADD_HIGH_BYTE_ADDER_HIDDEN_DIM,
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_add_carry_cascade",
        },
        spec_section="BLOG_SPEC.md#multibyte-arithmetic",
    )


def _l10_nonfirst_psh_sp_helper_rules() -> tuple[FFNRule, ...]:
    """One AND unit: ``NONFIRST_PSH_SP_SUPPRESS = (SP result byte0 == 0xF0)``.

    Fires at the SP MARKER row when the SP-decrement RESULT byte 0 (still
    intact in OUTPUT_LO/HI at this pre-tail block) is exactly 0xF0 -- i.e.
    ``OUTPUT_LO+0`` (low nibble 0) AND ``OUTPUT_HI_THIS_STEP+15`` (high
    nibble 0xF). That is true ONLY on a non-first push (e.g. the 2nd push of
    a program: 0xFFF8 - 8 = 0xFFF0); on the first push the result is 0xF8
    (OUTPUT_LO+8, not +0) and on the initial/bootstrap SP it is 0x00
    (OUTPUT_HI+0, not +15), so the AND is 0 in both load-bearing cases of the
    0xF8 exactness rule. The scratch flag NOT-blocks
    ``tail_sp_marker_byte0_f8_from_initial_stack_exact`` so it stops forcing
    0xF8 over the genuine 0xF0, fixing the non-first-PSH SP byte-0 bug while
    leaving every 0xF8 case (first push, JSR-bootstrap, unchanged-SP SI/LI)
    untouched (the 0xF8 result has OUTPUT_LO+0 ~0, so the AND -- and the
    blocker -- stay off). MARK_SP + HAS_SE gate it to the SP marker on
    non-first steps; IS_BYTE / other-marker blockers keep it off byte and
    non-SP rows. spec_k=0 probe (block 37): result 0xF0 -> fires on push2
    only.
    """
    # BOTH nibble one-hots must be present (a tight AND) -- 0xF8 and 0xF0
    # SHARE the high nibble 0xF (OUTPUT_HI+15), so the low nibble
    # (OUTPUT_LO+0, set for 0xF0 but NOT 0xF8) is the decisive term and must
    # not be out-voted. Weight each nibble 10 and require threshold 18: both
    # present (~0.9 each) -> ~18-20 fires; only HI+15 (0xF8 case) -> ~10+2 < 18
    # stays off; only LO+0 (e.g. 0x00, HI+0) -> ~10+2 < 18 stays off. MARK_SP
    # + HAS_SE add a small ~2 of headroom (they are 1.0/1.0 at the live SP
    # marker) without letting a single nibble cross alone. The -1e6 blockers
    # drive byte / non-SP / next-step rows decisively negative.
    return (
        multi_way_and_rule(
            name="l10_nonfirst_psh_sp_suppress_and",
            conditions=(
                ("OUTPUT_LO+0", 10.0),
                ("OUTPUT_HI_THIS_STEP+15", 10.0),
                ("MARK_SP", 1.0),
                ("HAS_SE", 1.0),
                ("IS_BYTE", -1_000_000.0),
                ("MARK_AX", -1_000_000.0),
                ("MARK_PC", -1_000_000.0),
                ("MARK_BP", -1_000_000.0),
                ("MARK_STACK0", -1_000_000.0),
                ("MARK_MEM", -1_000_000.0),
                ("NEXT_PC", -1_000_000.0),
                ("NEXT_AX", -1_000_000.0),
                ("NEXT_SP", -1_000_000.0),
                ("NEXT_BP", -1_000_000.0),
                ("NEXT_STACK0", -1_000_000.0),
                ("NEXT_MEM", -1_000_000.0),
                ("NEXT_SE", -1_000_000.0),
            ),
            threshold=18.0,
            writes=(("NONFIRST_PSH_SP_SUPPRESS", 1.0),),
        ),
    )


def make_l10_nonfirst_psh_sp_helper_op() -> Operation:
    """AND helper for the non-first-PSH SP byte-0 fix (flag-gated, default ON).

    Computes ``NONFIRST_PSH_SP_SUPPRESS`` (SP-decrement result byte0 == 0xF0
    at the SP marker) in a single standalone ``PureFFN`` post_op, BEFORE the
    L25 ``tail_bit32_result_correction`` block (the scheduler orders this
    producer first because that op reads the band). The flag-off build
    produces zero rules and the band is not collected, so it is byte-identical
    to the prior model. See ``_nonfirst_psh_sp_fix_enabled`` and
    ``_l10_nonfirst_psh_sp_helper_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical). Attaches on the L25
    # tail block; the produces/consumes dep (tail_bit32 reads
    # NONFIRST_PSH_SP_SUPPRESS) orders this helper BEFORE it.
    return _make_l10_postop(
        name="l10_nonfirst_psh_sp_helper",
        rules_fn=_l10_nonfirst_psh_sp_helper_rules,
        reads={
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "MARK_SP", "MARK_AX",
            "MARK_PC", "MARK_BP", "MARK_STACK0", "MARK_MEM", "HAS_SE",
            "IS_BYTE", "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
            "NEXT_STACK0", "NEXT_MEM", "NEXT_SE",
        },
        writes={"NONFIRST_PSH_SP_SUPPRESS"},
        suppress=True,
        flag_fn=_nonfirst_psh_sp_fix_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


# Opcodes that own the AX-marker OUTPUT byte through their own lane (the
# arithmetic/bitwise/compare ALU, IMM dispatch, and the shift shortcuts). On a
# row where any of these is the live opcode, OUTPUT is already correct and the
# EXIT/no-clean-opcode AX_CARRY override must stay silent. Mirrors the spirit
# of ``_L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS`` but for the LATE override.
# NOTE: OP_ENT is intentionally NOT in this list. On the post-LEV EXIT row
# OP_ENT leaks a small ~0.16 residue (from the prior ENT-0 frame), and a hard
# NOT-block on it would veto the legitimate override there. ENT does not own a
# computed AX-marker OUTPUT byte, so excluding it is safe.
_L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS = (
    "OP_IMM", "OP_ADD", "OP_SUB", "OP_OR", "OP_XOR", "OP_AND",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_MUL", "OP_DIV", "OP_MOD", "OP_SHL", "OP_SHR",
    "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
)


def _l10_ent_axcarry_enabled() -> bool:
    """Flag for the multilocal-ENT AX byte-0 (ENT-immediate leak) source fix.

    DEFAULT-ON in the campaign config (``C4_NO_STACK0_EMIT=1``); opt-out via
    ``C4_L10_ENT_AXCARRY=0``. Flag-off OR non-campaign build is bit-for-bit
    golden (``f2b040aa``) — the op then registers ZERO rules and appends NO
    post_op.

    ROOT (teacher-forced + AR spec_k=0, BUILT dims, campaign config; probes
    ``tools/_probe_loopent_tf.py`` / ``_probe_loopent_block.py`` /
    ``_probe_loopent_disc.py``): on the ``loop_sum`` / ``loop_mul`` /
    ``loop_pow2`` **multilocal** main-ENT step (frame size 16, i.e. >= 2
    locals) the opcode decode at the AX marker resolves NO clean ENT
    (``OP_ENT == 0``, where var_simple's 1-local ENT-8 cleanly decodes
    ``OP_ENT == 5``) but ``OP_LEA`` LEAKS to ~0.81. That leaked OP_LEA TRIGGERS
    the same L10 LEA effective-address high-nibble materializer that the
    post-LEV ``_l10_exit_axcarry`` fix targets (gated on OP_LEA, reads
    ``FETCH_HI`` -> writes ``OUTPUT_HI``). The ENT frame-size immediate's high
    nibble (16 -> ``FETCH_HI+1``) is stamped into ``OUTPUT_HI+1`` -> AX byte-0
    emits ``0x10`` (= the ENT immediate) instead of the carried prior AX
    (``AX_CARRY``, byte-0 = 0x00 here, the value ENT must preserve). var_simple
    (ENT-8) passes because its imm high nibble is 0, so the same materializer
    write lands on the (already-zero) ``OUTPUT_HI+0`` — invisible. The leak is
    therefore only visible at frame size >= 16 (>= 2 locals): the
    loop_sum/loop_mul/loop_pow2 cluster (~75 programs).

    WHY ``_l10_exit_axcarry`` does NOT already cover it: that op REQUIRES
    ``OP_LEA >= ~0.83`` (``MARK_AX*100 + OP_LEA*60 >= 150``) and HARD-blocks
    ``OP_JSR`` (-500). The multilocal-ENT leak row has ``OP_LEA == 0.81``
    (just under) AND ``OP_JSR == 1.19`` (a JSR residue, since the main ENT
    follows the bootstrap JSR), so the EXIT op's OP_JSR block vetoes it. This
    sibling lowers the OP_LEA bar (threshold 145) and drops the OP_JSR block
    (a clean JSR step has ``OP_LEA == 0`` so it never satisfies the required
    OP_LEA term regardless), while keeping the SAME proven exclusions: the
    hard ``MEM_ADDR_SRC`` NOT-block (a GENUINE LEA address-eval row carries it;
    the spurious leak does not) and the hard non-AX-marker blocks. On a clean
    ENT step (``OP_LEA == 0``) the score is 100 < 145 -> no fire (byte-
    identical), so this never disturbs the var_simple / nested-callee ENT path
    where ``ent_ax_passthrough`` already delivers the carried AX correctly.

    FIX: mirror ``_l10_exit_axcarry_rules`` — a flag-gated ``PureFFN`` post_op
    on the L25 tail block (after ``tail_bit32_result_correction``) that routes
    ``AX_CARRY_{LO,HI}[k] -> OUTPUT_{LO,HI}[k]`` at a magnitude (DOM=0.02) that
    dominates the ~9.6 materializer write, re-asserting the carried prior AX.
    A genuine LEA AX row (``OP_LEA == 5.2``) is excluded by the MEM_ADDR_SRC
    block; even were it reached, its keystone 0xE8 materializer writes at
    ~4.4e9 so DOM=0.02 cannot perturb it.
    """
    from .shared import no_stack0_emit_enabled

    if os.environ.get("C4_L10_ENT_AXCARRY", "1") == "0":
        return False
    return no_stack0_emit_enabled()


def _l10_ent_axcarry_rules() -> tuple[FFNRule, ...]:
    """Route AX_CARRY -> OUTPUT on the multilocal-ENT leaked-LEA AX row.

    Sibling of :func:`_l10_exit_axcarry_rules` (see
    :func:`_l10_ent_axcarry_enabled` for the full root). 32 units (16 LO + 16
    HI). Each fires iff:

      * ``MARK_AX`` (the AX marker row) is present, AND
      * ``OP_LEA`` is LEAKED (~0.81 on the multilocal-ENT row; EXACTLY 0.00 on
        every clean ENT / non-LEA AX row, so it is the required discriminator),
        AND
      * NO clean OUTPUT-owning opcode is live (IMM/ADD/SUB/bitwise/cmp/MUL/DIV/
        MOD/SHL/SHR/JMP/BZ/BNZ -- each a moderate -OPC_BLOCK NOT-block), AND
      * ``MEM_ADDR_SRC`` is cold (a GENUINE LEA address-eval row carries it; the
        spurious ENT leak does not) -- hard NOT-block, AND
      * we are not on any non-AX marker row (PC/SP/BP/STACK0/MEM -1e6).

    Threshold 145 (vs the EXIT sibling's 150) so the slightly weaker ENT-step
    OP_LEA leak (~0.81 -> ~149 score) fires while a clean ENT (OP_LEA 0 -> 100)
    does not. ``OP_JSR`` is intentionally NOT a NOT-block here: the post-JSR
    main ENT carries a ~1.19 OP_JSR residue, and a clean JSR step never
    satisfies the required OP_LEA term anyway.
    """
    DOM = 0.02
    OPC_BLOCK = 500.0
    # Same OUTPUT-owning opcode set as the EXIT sibling MINUS OP_JSR (the
    # multilocal-ENT leak row carries a JSR residue; OP_LEA already gates out
    # a clean JSR).
    owning_ops = tuple(
        op for op in _L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS if op != "OP_JSR"
    )
    rules: list[FFNRule] = []
    for nibble_label, carry_dim, out_dim in (
        ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for k in range(16):
            conditions: list[tuple[str, float]] = [
                ("MARK_AX", 100.0),
                # Leaked-OP_LEA discriminator: the ENT leak is ~0.81 and is
                # EXACTLY 0.00 on every clean ENT / non-LEA AX row, so weight
                # 60 (-> ~49) makes it the load-bearing required term.
                ("OP_LEA", 60.0),
                *((op, -OPC_BLOCK) for op in owning_ops),
                # Genuine LEA address-eval rows carry MEM_ADDR_SRC; the spurious
                # ENT leak does not. Clean one-hot -> hard NOT-block.
                ("MEM_ADDR_SRC", -1_000_000.0),
                ("MARK_PC", -1_000_000.0),
                ("MARK_SP", -1_000_000.0),
                ("MARK_BP", -1_000_000.0),
                ("MARK_STACK0", -1_000_000.0),
                ("MARK_MEM", -1_000_000.0),
            ]
            writes: list[tuple[str, float]] = []
            for j in range(16):
                writes.append((f"{out_dim}+{j}", (DOM if j == k else -DOM)))
            rules.append(multi_way_and_rule(
                name=f"l10_ent_axcarry_{nibble_label}_{k}",
                conditions=tuple(conditions),
                threshold=145.0,
                gate=f"{carry_dim}+{k}",
                gate_weight=1.0,
                writes=tuple(writes),
            ))
    return tuple(rules)


def make_l10_ent_axcarry_op() -> Operation:
    """Flag-gated L10 multilocal-ENT AX_CARRY -> OUTPUT override op.

    Standalone ``PureFFN`` post_op attached to the L25 tail block AFTER
    ``tail_bit32_result_correction`` (mirrors ``make_l10_exit_axcarry_op``).
    Flag-off (or non-campaign) produces ZERO rules and appends NO post_op ->
    byte-identical to golden ``f2b040aa``. Flag-ON (campaign default) keeps the
    ENT frame-size immediate out of the AX byte-0 dump on the multilocal ENT
    step (loop_sum/loop_mul/loop_pow2). See ``_l10_ent_axcarry_enabled`` /
    ``_l10_ent_axcarry_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``f2b040aa``).
    # Run AFTER tail_bit32_result_correction (the OUTPUT producer it overrides)
    # AND after the EXIT sibling so the two overrides compose deterministically
    # (they target disjoint rows — EXIT requires OP_LEA>=0.83+OP_JSR-clean, ENT
    # fires on the OP_LEA~0.81/OP_JSR-residue row — but a fixed order keeps the
    # bake reproducible).
    return _make_l10_postop(
        name="l10_ent_axcarry",
        rules_fn=_l10_ent_axcarry_rules,
        reads={
            "MARK_AX", "OP_LEA", "MEM_ADDR_SRC",
            "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
            "AX_CARRY_LO", "AX_CARRY_HI",
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
            *(op for op in _L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS if op != "OP_JSR"),
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        requires={"after": "l10_exit_axcarry"},
        suppress=True,
        flag_fn=_l10_ent_axcarry_enabled,
        spec_section="BLOG_SPEC.md#function-call",
    )


# ===========================================================================
# loop_sum in-loop ``LEA &i`` byte-0 0xE8 RESTORE (#330, campaign config).
#
# See ``shared.loop_lea_b0_e8_restore_enabled`` for the full two-part root. The
# in-loop ``LEA &i`` AX row wants byte-0 = 0xE8 but (1) the e8 keystone can't
# fire (needs CMP+7 + MEM_ADDR_SRC, both 0 here) so byte-0 is DEAD through the
# tail, and (2) ``_l10_ent_axcarry`` then slams it to 0x00. This op runs AFTER
# ent_axcarry and re-stamps 0xE8 on exactly that row.
#
# DISCRIMINATOR (measured spec_k=0, BUILT dims, campaign): MARK_AX=1 +
# OP_LEA=5.24 (a GENUINE LEA, not the 0.81 ENT leak) + imm=-8 FETCH signature
# (FETCH_LO+8 / FETCH_HI+15 lit, FETCH_LO+0 / FETCH_HI+14 cold) + MEM_ADDR_SRC
# cold + no owning opcode. The MEM_ADDR_SRC NOT-block keeps it off every genuine
# address-eval LEA (those carry MEM_ADDR_SRC=1, already 0xE8-correct via the
# keystone); the imm=-8 FETCH AND-gate (FETCH_LO+0 / FETCH_HI+14 NOT-blocks)
# keeps it off the 2nd/3rd-local LEAs (var_mul / var_three, want 0xE0 / 0xD8)
# and off every non-LEA AX row.
_LOOP_LEA_E8_DOM = 0.1           # per-cell winner-take-all magnitude. WITHOUT
#                                  the _suppress_ffn_on_step_boundary 1e9 gate
#                                  inflation the firing-row silu is ~3e4, so the
#                                  per-cell write is ~3e3 — enough to flip cell-8
#                                  above cell-0 over the ~1e3 ent_axcarry slam
#                                  (need > slam ~1033), but SMALL enough that the
#                                  fraction of this write copied to the PC
#                                  value-byte rows (via the late attention copy)
#                                  stays under the no_stack0_pc_highbyte_clear
#                                  -300 sink, so the PC high bytes still default
#                                  to 0x00 (a larger write leaks 0xE8 into PC
#                                  bytes 1-3).
_LOOP_LEA_E8_THRESHOLD = 500.0   # the imm=-8 row scores base(MARK_AX*100 +
#                                  OP_LEA*60 ~414) + FETCH(~160) = ~574 > 500;
#                                  the 2nd/3rd-local LEAs score base - 1000 < 0;
#                                  a clean non-LEA AX row scores 100. So only the
#                                  imm=-8 (1st-local &i) LEA crosses.
_LOOP_LEA_E8_LO_NIBBLE = 8       # 0xE8 low nibble
_LOOP_LEA_E8_HI_NIBBLE = 14      # 0xE8 high nibble (0xE)

# PROJECT_0XE8_SLAM Phase-2: the MULTIPLICATIVE OP_LEA gate weight
# (C4_LOOP_LEA_OPLEA_GATE, DEFAULT-OFF). The FFN gate is a plain LINEAR read of
# OP_LEA (gate = 0.0 + _LOOP_LEA_OPLEA_GATE_W * OP_LEA), UNSCALED by S. Chosen so
# the genuine in-loop LEA (OP_LEA ~5.23) yields gate ~= 1.0 -> the unit fires at
# the SAME magnitude as today (loop function byte-identical), while the leak
# IMM/comparison row (OP_LEA == 0 EXACTLY) yields gate == 0.0 -> hidden =
# silu(up) * 0 = 0, a true zero-out that NO FETCH amplitude can cross. A bias of
# 0.0 is deliberate: a NEGATIVE gate would INVERT the +/-DOM winner-take-all on a
# false-fire row (silu(up) stays large-positive) and stamp a DIFFERENT wrong
# byte, whereas gate==0 is a clean no-op. See shared.loop_lea_oplea_gate_enabled.
_LOOP_LEA_OPLEA_GENUINE = 5.23   # measured genuine in-loop LEA OP_LEA value
_LOOP_LEA_OPLEA_GATE_W = 1.0 / _LOOP_LEA_OPLEA_GENUINE  # -> genuine gate ~= 1.0


def _l10_loop_lea_b0_e8_rules() -> tuple[FFNRule, ...]:
    """32 winner-take-all rules: re-stamp byte-0 = 0xE8 on the in-loop LEA row.

    16 LO units drive ``OUTPUT_LO`` to nibble 8 (``+DOM`` at cell 8, ``-DOM``
    elsewhere) and 16 HI units drive ``OUTPUT_HI_THIS_STEP`` to nibble E
    (``+DOM`` at cell 14, ``-DOM`` elsewhere). Every unit shares the same in-loop
    ``LEA &i`` discriminator, so the whole op is a no-op on every other row.
    """
    OPC_BLOCK = 500.0
    # OP_JSR is intentionally EXCLUDED (mirrors _l10_ent_axcarry): the in-loop
    # LEA row carries an ~0.27 OP_JSR residue, and a clean JSR step has OP_LEA==0
    # so the required OP_LEA term gates it out regardless.
    owning_ops = tuple(
        op for op in _L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS if op != "OP_JSR"
    )
    # OP_LEA HARD-REQUIREMENT narrowing (POST-FLIP func_identity step-9 LI fix,
    # flag C4_LOOP_LEA_B0_E8_OPLEA_REQ, default ON). The pre-fix discriminator
    # made OP_LEA NON-load-bearing (weight 60 -> ~314, but FETCH_LO+8*1000 alone
    # cleared threshold 500), so on func_identity's ``LI`` row -- which carries
    # FETCH_LO+8==1.0 but OP_LEA==0 -- the op FALSE-FIRED and stamped 0xE8 over
    # the loaded value (got_ax 0xFFE8 vs oracle 70). Bump OP_LEA 60->200 and add
    # a CONST -650 bias so the genuine in-loop ``LEA &i`` (OP_LEA 5.24) still
    # FIRES (100 + 1048 + 160 - 650 = 658 > 500, 158-pt margin) while the LI row
    # (OP_LEA 0) is VETOED (100 + 0 + 1000 - 650 = 450 < 500). The 2nd/3rd-local
    # LEAs (FETCH net -1000) stay silent (-502). OFF reverts to the regressed
    # weights for A/B. See ``shared.loop_lea_b0_e8_oplea_req_enabled``.
    _oplea_req = loop_lea_b0_e8_oplea_req_enabled()
    _op_lea_w = 200.0 if _oplea_req else 60.0
    _const_bias: tuple[tuple[str, float], ...] = (
        (("CONST", -650.0),) if _oplea_req else ()
    )
    disc: tuple[tuple[str, float], ...] = _const_bias + (
        ("MARK_AX", 100.0),
        # GENUINE LEA (OP_LEA ~5.24): HARD requirement when the OPLEA_REQ
        # narrowing is on (weight 200 + CONST -650 -> OP_LEA load-bearing),
        # else the pre-fix weight 60 (~314, NON-load-bearing — the regression).
        ("OP_LEA", _op_lea_w),
        # imm=-8 DISCRIMINATOR (the LOAD-BEARING term). FETCH is a NON-one-hot
        # broadcast (the imm-low-nibble cell ~0.4-1.0, the rest ~0), and the only
        # thing that distinguishes the 1st local (imm=-8 -> 0xE8, FETCH_LO+8
        # dominant) from the 2nd local (imm=-16 -> 0xE0, FETCH_LO+0 dominant) and
        # 3rd local (imm=-24 -> 0xD8, FETCH_HI+14 dominant) is WHICH FETCH cell
        # dominates. So we score (FETCH_LO+8) - (FETCH_LO+0) - (FETCH_HI+14) at a
        # LARGE weight: measured step2 (&i) = 0.58 - 0.42 - 0 = +0.16 (-> +160);
        # step6 (&sum) = 0.00 - 1.00 - 0 = -1.00 (-> -1000); a 3rd local =
        # -1.00 (FETCH_HI+14). With threshold 500 the &i row (base 414 + 160 =
        # 574) FIRES while &sum / &c (base 414 - 1000 = -586) stay SILENT, and a
        # clean non-LEA AX row (base 100, no FETCH) is far below 500. The big
        # symmetric weights make the imm=-8 signature a GENUINE requirement
        # (additive FETCH_LO+8 alone could not separate it from a soft broadcast).
        ("FETCH_LO+8", 1000.0),
        ("FETCH_LO+0", -1000.0),
        ("FETCH_HI+14", -1000.0),
        # Genuine address-eval LEA rows carry MEM_ADDR_SRC (a clean one-hot, and
        # are already 0xE8-correct via the keystone); the in-loop LEA does not.
        # Hard NOT-block so this op touches ONLY the keystone-starved in-loop row.
        ("MEM_ADDR_SRC", -1_000_000.0),
        # HARD NOT-block IS_BYTE: the PC/AX VALUE-byte rows carry IS_BYTE=1 AND
        # the same OP_LEA / FETCH broadcast (OP_LEA*60 ~314 + FETCH alone clears
        # the threshold there), so WITHOUT this block the op fires on the PC value
        # bytes 1-3 and stamps 0xE8 into the PC high bytes (PC=0x..E8E8E8). The
        # AX-MARKER row has IS_BYTE=0, so this gate isolates the marker row.
        ("IS_BYTE", -1_000_000.0),
        # No owning opcode may be live (IMM/ADD/.../JMP/BZ/BNZ; OP_JSR excluded).
        *((op, -OPC_BLOCK) for op in owning_ops),
        # Never any non-AX marker row.
        ("MARK_PC", -1_000_000.0),
        ("MARK_SP", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
    ) + _tail_lea_e8_ent_byte0_blockers()
    # ^^ ENT-step guard (#311): the main ENT step's frame-size immediate ``ENT 8``
    # ALSO lights ``FETCH_LO+8`` (the imm=-8 LEA signature), so the
    # ``FETCH_LO+8 * 1000`` term clears threshold 500 here even though
    # ``OP_LEA == 0`` -> this op spuriously re-stamps 0xE8 onto the carried-AX
    # ENT AX dump (var_simple / if_var step-1 got_ax 744). The per-step FETCHED
    # ENT opcode one-hot ``OPCODE_BYTE_LO+6`` hard-blocks it on a real ENT step
    # while leaving every genuine in-loop LEA row (OPCODE_BYTE_LO+6 == 0)
    # byte-identical. See ``_tail_lea_e8_ent_guard_enabled``.
    # PROJECT_0XE8_SLAM Phase-2: the MULTIPLICATIVE OP_LEA gate (default OFF via
    # C4_LOOP_LEA_OPLEA_GATE). Added to EVERY unit so the whole op zeroes when
    # OP_LEA == 0 (the IMM/comparison leak rows) regardless of FETCH amplitude,
    # and is a ~no-op (gate ~= 1.0) on genuine OP_LEA ~= 5.23 in-loop LEA rows.
    _gate_kwargs: dict = (
        {"gate_bias": 0.0, "gate_terms": (("OP_LEA", _LOOP_LEA_OPLEA_GATE_W),)}
        if loop_lea_oplea_gate_enabled()
        else {}
    )
    rules: list[FFNRule] = []
    for out_dim, tgt in (
        ("OUTPUT_LO", _LOOP_LEA_E8_LO_NIBBLE),
        ("OUTPUT_HI_THIS_STEP", _LOOP_LEA_E8_HI_NIBBLE),
    ):
        for k in range(16):
            writes = tuple(
                (f"{out_dim}+{j}", (_LOOP_LEA_E8_DOM if j == tgt else -_LOOP_LEA_E8_DOM))
                for j in range(16)
            )
            rules.append(multi_way_and_rule(
                name=f"l10_loop_lea_b0_e8_{out_dim.lower()}_{k}",
                conditions=disc,
                threshold=_LOOP_LEA_E8_THRESHOLD,
                writes=writes,
                **_gate_kwargs,
            ))
    return tuple(rules)


def make_l10_loop_lea_b0_e8_op() -> Operation:
    """Flag-gated L10 in-loop ``LEA &i`` byte-0 0xE8 RESTORE op.

    Standalone ``PureFFN`` post_op attached to the L25 tail block AFTER
    ``l10_ent_axcarry`` (so it is the LAST OUTPUT writer at the in-loop LEA row
    and DOMINATES the ent_axcarry 0x00 slam). Flag-off (or non-campaign /
    golden) produces ZERO rules and appends NO post_op -> byte-identical to
    golden ``7f6f2e5d``. See ``loop_lea_b0_e8_restore_enabled`` /
    ``_l10_loop_lea_b0_e8_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``7f6f2e5d``).
    # Run AFTER l10_ent_axcarry (the slammer it overrides) so this is the LAST
    # OUTPUT writer at the in-loop LEA row.
    #
    # ``suppress=False`` is LOAD-BEARING: ``_suppress_ffn_on_step_boundary`` adds
    # IS_BYTE as an ALTERNATIVE structural gate (OR'd with the MARK_* dims), which
    # would let this op ALSO fire on the PC/AX VALUE-BYTE rows (IS_BYTE=1) —
    # leaking the 0xE8 stamp into the PC high bytes (PC=0x..E8E8E8). The
    # discriminator already REQUIRES MARK_AX and HARD-blocks every non-AX marker,
    # so the op fires ONLY on the AX-marker LEA row; the boundary gate is both
    # unnecessary and harmful here.
    return _make_l10_postop(
        name="l10_loop_lea_b0_e8",
        rules_fn=_l10_loop_lea_b0_e8_rules,
        reads={
            "CONST", "MARK_AX", "OP_LEA", "MEM_ADDR_SRC",
            "FETCH_LO", "FETCH_HI",
            "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
            *_L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS,
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        requires={"after": "l10_ent_axcarry"},
        suppress=False,
        flag_fn=loop_lea_b0_e8_restore_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


# ===========================================================================
# loop_sum in-loop 2nd-local ``LEA &sum`` byte-0 0xE0 RESTORE (#330, campaign).
#
# See ``shared.loop_lea_b0_e0_restore_enabled`` for the full root. After the
# merged ``C4_LOOP_LEA_B0_E8`` (step-2) fix advances loop_sum to step 6, the
# in-loop ``sum = sum + i`` body's ``LEA &sum`` AX row wants byte-0 = 0xE0 but
# the post-tail block-44 LEA effective-address materializer WTA-slams it to 0x01
# (cell 1). The e8-restore op DEFERS here (FETCH_LO+0 NOT-block), so nothing
# dominates the slam. This op runs AFTER l10_loop_lea_b0_e8 and re-stamps 0xE0
# on exactly the 2nd-local LEA row.
#
# DISCRIMINATOR (measured spec_k=0, BUILT dims, campaign; GPU AR-trace
# tools/_probe_loopsum_lea_b0.py 450 6 + _probe_state.py): MARK_AX=1 +
# OP_LEA=5.24 (a GENUINE LEA) + imm=-16 FETCH signature (FETCH_LO+0 lit =1.0,
# FETCH_LO+8 / FETCH_HI+14 cold) + MEM_ADDR_SRC cold + no owning opcode. The
# imm=-16 FETCH AND-gate is the EXACT inverse of the e8 op (FETCH_LO+8-dominant):
# the 2nd local (&sum, FETCH_LO+0 dominant -> 0xE0) is separated from the 1st
# (&i, FETCH_LO+8 -> 0xE8) and 3rd (&c, FETCH_HI+14 -> 0xD8) by which FETCH cell
# dominates. The MEM_ADDR_SRC + IS_BYTE + non-AX-marker NOT-blocks keep it off
# every genuine address-eval LEA, every value-byte row, and every non-LEA AX
# row.
_LOOP_LEA_E0_DOM = 0.1           # per-cell winner-take-all magnitude. Mirror of
#                                  _LOOP_LEA_E8_DOM: small enough that the
#                                  fraction copied to the PC value-byte rows
#                                  stays under the no_stack0_pc_highbyte_clear
#                                  -300 sink, but large enough (post silu
#                                  inflation) to flip cell-0 above cell-1 over
#                                  the ~679 block-44 slam.
_LOOP_LEA_E0_THRESHOLD = 1700.0  # CRITICAL: separates the GENUINE 2nd-local LEA
#                                  (OP_LEA ~5.24) from the multilocal main-ENT
#                                  step-1 row (OP_LEA LEAKS ~0.81 + ALSO carries
#                                  FETCH_LO+0 == 1.0, so it would over-fire on a
#                                  low OP_LEA gate). With OP_LEA weight 200:
#                                  step6 &sum  = 100 + 5.24*200 + 1000 = ~2148 (FIRES)
#                                  step1 ENT   = 100 + 0.81*200 + 1000 = ~1262 (SILENT)
#                                  step2 &i    = 100 + 5.24*200 - 160  = ~988  (SILENT, FETCH_LO+8)
#                                  3rd-local &c= 100 + 5.24*200 - 1000 = ~148  (SILENT, FETCH_HI+14)
#                                  clean non-LEA AX row = 100 (SILENT). Only the
#                                  genuine imm=-16 (2nd-local &sum) LEA crosses 1700.
_LOOP_LEA_E0_OP_LEA_W = 200.0    # the LOAD-BEARING OP_LEA weight: makes the
#                                  GENUINE-vs-LEAK OP_LEA gap (5.24 vs 0.81) the
#                                  deciding margin against the shared FETCH_LO+0.
_LOOP_LEA_E0_LO_NIBBLE = 0       # 0xE0 low nibble
_LOOP_LEA_E0_HI_NIBBLE = 14      # 0xE0 high nibble (0xE)


def _l10_loop_lea_b0_e0_rules() -> tuple[FFNRule, ...]:
    """32 winner-take-all rules: re-stamp byte-0 = 0xE0 on the 2nd-local LEA row.

    16 LO units drive ``OUTPUT_LO`` to nibble 0 (``+DOM`` at cell 0, ``-DOM``
    elsewhere) and 16 HI units drive ``OUTPUT_HI_THIS_STEP`` to nibble E
    (``+DOM`` at cell 14, ``-DOM`` elsewhere). Every unit shares the same
    2nd-local ``LEA &sum`` discriminator (the inverse-FETCH of the e8 op), so the
    whole op is a no-op on every other row.
    """
    OPC_BLOCK = 500.0
    # OP_JSR is intentionally EXCLUDED (mirrors _l10_loop_lea_b0_e8): the in-loop
    # LEA row carries an ~0.27 OP_JSR residue, and a clean JSR step has OP_LEA==0
    # so the required OP_LEA term gates it out regardless.
    owning_ops = tuple(
        op for op in _L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS if op != "OP_JSR"
    )
    disc: tuple[tuple[str, float], ...] = (
        ("MARK_AX", 100.0),
        # GENUINE LEA (OP_LEA ~5.24) vs the multilocal main-ENT step-1 LEAK
        # (OP_LEA ~0.81): weight 200 makes this gap the deciding margin against
        # the shared FETCH_LO+0 (the ENT-leak row ALSO carries FETCH_LO+0 == 1.0,
        # so a LOW OP_LEA gate would over-fire and stamp 0xE0 on the ENT row ->
        # AX = 0x2E0 = 736). 5.24*200 = 1048 (genuine), 0.81*200 = 162 (leak).
        ("OP_LEA", _LOOP_LEA_E0_OP_LEA_W),
        # imm=-16 DISCRIMINATOR (INVERSE of the e8 op). FETCH is a NON-one-hot
        # broadcast; the only thing distinguishing the 2nd local (imm=-16 ->
        # 0xE0, FETCH_LO+0 dominant) from the 1st (imm=-8 -> 0xE8, FETCH_LO+8
        # dominant) and 3rd (imm=-24 -> 0xD8, FETCH_HI+14 dominant) is WHICH
        # FETCH cell dominates. So we score (FETCH_LO+0) - (FETCH_LO+8) -
        # (FETCH_HI+14) at a LARGE weight: measured step6 (&sum) = 1.00 (->
        # +1000); step2 (&i) = 0.42 - 0.58 = -0.16 (-> -160); a 3rd local =
        # -1.00 (FETCH_HI+14). With threshold 1700 ONLY the genuine 2nd-local
        # &sum row (100 + 1048 + 1000 = ~2148) crosses; step1 ENT (100 + 162 +
        # 1000 = ~1262), step2 &i (100 + 1048 - 160 = ~988), the 3rd local
        # (100 + 1048 - 1000 = ~148) and a clean non-LEA AX row (100) all stay
        # SILENT. The big symmetric weights make the imm=-16 signature a GENUINE
        # requirement.
        ("FETCH_LO+0", 1000.0),
        ("FETCH_LO+8", -1000.0),
        ("FETCH_HI+14", -1000.0),
        # Genuine address-eval LEA rows carry MEM_ADDR_SRC (a clean one-hot, and
        # are already correct via the keystone); the in-loop LEA does not. Hard
        # NOT-block so this op touches ONLY the keystone-starved in-loop row.
        ("MEM_ADDR_SRC", -1_000_000.0),
        # HARD NOT-block IS_BYTE: the PC/AX VALUE-byte rows carry IS_BYTE=1 AND
        # the same OP_LEA / FETCH broadcast, so WITHOUT this block the op fires on
        # the PC value bytes 1-3 and stamps 0xE0 into the PC high bytes. The
        # AX-MARKER row has IS_BYTE=0, so this gate isolates the marker row.
        ("IS_BYTE", -1_000_000.0),
        # No owning opcode may be live (IMM/ADD/.../JMP/BZ/BNZ; OP_JSR excluded).
        *((op, -OPC_BLOCK) for op in owning_ops),
        # Never any non-AX marker row.
        ("MARK_PC", -1_000_000.0),
        ("MARK_SP", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
    )
    # PROJECT_0XE8_SLAM Phase-2: the MULTIPLICATIVE OP_LEA gate (default OFF via
    # C4_LOOP_LEA_OPLEA_GATE). Mirror of the e8 op: added to EVERY unit so the
    # whole op zeroes when OP_LEA == 0 (the IMM/comparison leak rows, e.g. if_eq
    # id402 wants byte-0 0x10 not 0xe0) regardless of FETCH amplitude, and is a
    # ~no-op (gate ~= 1.0) on genuine OP_LEA ~= 5.23 2nd-local LEA rows.
    _gate_kwargs: dict = (
        {"gate_bias": 0.0, "gate_terms": (("OP_LEA", _LOOP_LEA_OPLEA_GATE_W),)}
        if loop_lea_oplea_gate_enabled()
        else {}
    )
    rules: list[FFNRule] = []
    for out_dim, tgt in (
        ("OUTPUT_LO", _LOOP_LEA_E0_LO_NIBBLE),
        ("OUTPUT_HI_THIS_STEP", _LOOP_LEA_E0_HI_NIBBLE),
    ):
        for k in range(16):
            writes = tuple(
                (f"{out_dim}+{j}", (_LOOP_LEA_E0_DOM if j == tgt else -_LOOP_LEA_E0_DOM))
                for j in range(16)
            )
            rules.append(multi_way_and_rule(
                name=f"l10_loop_lea_b0_e0_{out_dim.lower()}_{k}",
                conditions=disc,
                threshold=_LOOP_LEA_E0_THRESHOLD,
                writes=writes,
                **_gate_kwargs,
            ))
    return tuple(rules)


def make_l10_loop_lea_b0_e0_op() -> Operation:
    """Flag-gated L10 in-loop 2nd-local ``LEA &sum`` byte-0 0xE0 RESTORE op.

    Standalone ``PureFFN`` post_op attached to the L25 tail block AFTER
    ``l10_loop_lea_b0_e8`` (so it is the LAST OUTPUT writer at the 2nd-local LEA
    row and DOMINATES the block-44 0x01 slam). Flag-off (or non-campaign /
    golden) produces ZERO rules and appends NO post_op -> byte-identical to
    golden ``fd60f5f4``. See ``loop_lea_b0_e0_restore_enabled`` /
    ``_l10_loop_lea_b0_e0_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``fd60f5f4``).
    # Run AFTER l10_loop_lea_b0_e8 (which itself runs after l10_ent_axcarry) so
    # this is the LAST OUTPUT writer at the 2nd-local LEA row, dominating the
    # block-44 0x01 slam.
    #
    # ``suppress=False`` is LOAD-BEARING (mirror of l10_loop_lea_b0_e8): the
    # boundary gate ORs IS_BYTE as an ALTERNATIVE structural gate, which would
    # let this op fire on the PC/AX VALUE-BYTE rows (IS_BYTE=1) and leak 0xE0 into
    # the PC high bytes. The discriminator already REQUIRES MARK_AX + HARD-blocks
    # every non-AX marker AND IS_BYTE, so the op fires ONLY on the AX-marker LEA
    # row.
    return _make_l10_postop(
        name="l10_loop_lea_b0_e0",
        rules_fn=_l10_loop_lea_b0_e0_rules,
        reads={
            "MARK_AX", "OP_LEA", "MEM_ADDR_SRC",
            "FETCH_LO", "FETCH_HI",
            "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
            *_L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS,
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        requires={"after": "l10_loop_lea_b0_e8"},
        suppress=False,
        flag_fn=loop_lea_b0_e0_restore_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


# ===========================================================================
# func-cluster step-0 (JSR) BP byte-3 high-byte CLEAR (campaign; DEFAULT OFF,
# opt in C4_JSR_BP_BYTE3_CLEAR=1). See ``shared.jsr_bp_byte3_clear_enabled`` for
# the full root. In one line: on the JSR that calls a callee, the caller's BP
# byte-3 must be 0x00 (BP = 0x00010000) but the OUTPUT band is a near-tie the
# WRONG way (OUTPUT_LO+1 +8.0 > OUTPUT_LO+0 +6.9) so byte-3 decodes 0x01, BP ->
# 0x01010000 and every downstream frame-relative read is poisoned. The ENT step
# has l6_ent_after_jsr_bp_byte3_00; the JSR step has NO clear.
# ===========================================================================
_JSR_BP_B3_DOM = 0.1               # per-cell winner-take-all magnitude (mirror
#                                   of _LOOP_LEA_E8_DOM: +DOM on nibble 0, -DOM
#                                   on every other cell, un-normalised residual).
_JSR_BP_B3_THRESHOLD = 6.0        # the JSR-step BP byte-3 predictor row scores
#                                   OP_JSR(0.1*6.85=0.685) + H1+3(1) + IS_BYTE(1)
#                                   + BYTE_INDEX_2(10*0.97) - BYTE_INDEX_3(10*0.01)
#                                   ~= 12.3 > 6.0; the byte-0/1 predictor rows
#                                   (BYTE_INDEX_0/1 ~0.97, BYTE_INDEX_2 ~0.01)
#                                   score ~-6.9 (BYTE_INDEX_2*10*0.01 -
#                                   BYTE_INDEX_0/1*10*0.97) << 6.0 -> SILENT, so
#                                   BP byte-0/1/2 are untouched. Every non-JSR /
#                                   non-BP / HAS_SE(ENT) / marker row is
#                                   hard-vetoed far below threshold.


def _l10_jsr_bp_byte3_clear_rules() -> tuple[FFNRule, ...]:
    """32 winner-take-all rules: force byte-3 = 0x00 on the JSR-step BP byte-3
    predictor row (the BP byte-2 value row, BYTE_INDEX_2 + H1+3).

    16 LO units drive ``OUTPUT_LO`` to nibble 0 and 16 HI units drive
    ``OUTPUT_HI_THIS_STEP`` to nibble 0 (``+DOM`` at cell 0, ``-DOM`` elsewhere),
    so the byte-3 low + high nibbles decode 0 -> byte-3 = 0x00. The
    discriminator REQUIRES a real JSR step (``OP_JSR``), the BP-register-byte
    ``H1+3`` staging one-hot, and ``BYTE_INDEX_2`` (the byte-3 predictor row),
    and HARD-blocks ``HAS_SE`` (the ENT-step byte rows carry HAS_SE~1 and are
    already handled by ``l6_ent_after_jsr_bp_byte3_00``), the non-BP register
    ``H1`` one-hots (PC=+0 / AX=+1 / SP=+2), the wrong ``BYTE_INDEX`` rows, and
    every non-BP marker. So the op is a strict no-op on every other row.
    """
    # CRITICAL (mirrors the l6_ent_after_jsr_bp_byte* comment): the wrong
    # BYTE_INDEX one-hots carry a ~0.01 ADJACENT-INDEX residue, so a -1e6 hard
    # NOT-block on BYTE_INDEX_1 / BYTE_INDEX_3 would spuriously veto THIS row
    # (BYTE_INDEX_3 residue ~0.01 * -1e6 = -1e4 crushes the AND). The
    # BYTE_INDEX_2 positive + the threshold ALREADY discriminate the byte-3
    # predictor row from the byte-0/1/3 predictor rows, so the adjacent BYTE
    # rows are separated ADDITIVELY, not by a hard block. Likewise the H1 one-
    # hots (PC=+0 / AX=+1 / SP=+2 / BP=+3) are clean 0/1 one-hots here, so the
    # non-BP register rows ARE hard-blockable via H1+0/+1/+2.
    disc: tuple[tuple[str, float], ...] = (
        # CONST (=1) bias so OP_JSR is a HARD, LOAD-BEARING requirement: a
        # non-JSR row (OP_JSR==0) scores >= -12 below every positive term and
        # falls under the threshold, so this op is a strict no-op on every
        # program without a JSR step (add/sub/var/loop/if) AND on the non-JSR
        # steps of func/nested/rec/gcd.
        ("CONST", -12.0),
        # GENUINE JSR step gate: OP_JSR ~6.85 on the byte rows of a JSR step.
        # Weight 2 -> +13.7 on a JSR row (with CONST -12 -> net +1.7 baseline);
        # 0 on any non-JSR row (net -12 baseline -> silent).
        ("OP_JSR", 2.0),
        # BP-register byte-row staging one-hot (PC=H1+0, AX=H1+1, SP=H1+2,
        # BP=H1+3). The clean, always-present BP discriminator on value rows.
        ("H1+3", 1.0),
        ("IS_BYTE", 1.0),
        # The byte-3 PREDICTOR row is the BP byte-2 value row (BYTE_INDEX_2).
        # LARGE symmetric weight (the LOAD-BEARING row selector): the byte-3
        # predictor scores +10*0.97; the byte-0/1 predictors (BYTE_INDEX_0/1
        # ~0.97, BYTE_INDEX_2 ~0.01) score -10*0.97 -> deep NEGATIVE, silent.
        ("BYTE_INDEX_2", 10.0),
        ("BYTE_INDEX_0", -10.0),
        ("BYTE_INDEX_1", -10.0),
        ("BYTE_INDEX_3", -10.0),
        # HARD NOT-block HAS_SE: the ENT-step BP byte rows carry HAS_SE~=0.98 and
        # are ALREADY cleared by l6_ent_after_jsr_bp_byte3_00; the JSR-step rows
        # carry HAS_SE==0. This keeps the op JSR-exclusive.
        ("HAS_SE", -1_000_000.0),
        # HARD NOT-block the non-BP register byte rows (their H1 one-hot != +3;
        # H1+0/+1/+2 are clean 0/1 one-hots here, no adjacent residue).
        ("H1+0", -1_000_000.0),
        ("H1+1", -1_000_000.0),
        ("H1+2", -1_000_000.0),
        # Never any register-MARKER row (value-byte target only).
        ("MARK_PC", -1_000_000.0),
        ("MARK_AX", -1_000_000.0),
        ("MARK_SP", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
    )
    rules: list[FFNRule] = []
    for out_dim in ("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"):
        for k in range(16):
            writes = tuple(
                (f"{out_dim}+{j}", (_JSR_BP_B3_DOM if j == 0 else -_JSR_BP_B3_DOM))
                for j in range(16)
            )
            rules.append(multi_way_and_rule(
                name=f"l10_jsr_bp_byte3_clear_{out_dim.lower()}_{k}",
                conditions=disc,
                threshold=_JSR_BP_B3_THRESHOLD,
                writes=writes,
            ))
    return tuple(rules)


def make_l10_jsr_bp_byte3_clear_op() -> Operation:
    """Flag-gated JSR-step BP byte-3 = 0x00 CLEAR op (C4_JSR_BP_BYTE3_CLEAR).

    Standalone ``PureFFN`` post_op attached to the L25 tail block AFTER the
    loop_lea ops (so it is the LAST OUTPUT writer at the JSR-step BP byte-3
    predictor row and DOMINATES the block-58 amplifier). Flag-off (or
    non-campaign / golden) produces ZERO rules and appends NO post_op ->
    byte-identical to golden ``f725c06e``. See ``jsr_bp_byte3_clear_enabled`` /
    ``_l10_jsr_bp_byte3_clear_rules``.
    """
    from .shared import jsr_bp_byte3_clear_enabled

    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``f725c06e``).
    # Run in the LATE L25 tail AFTER tail_bit32_result_correction (the
    # OUTPUT-band amplifier chain) -- mirror of make_l10_add_high_byte_adder_op --
    # so this is the LAST OUTPUT writer at the JSR-step BP byte-3 predictor row
    # and DOMINATES the block-58 amplifier (a post_ops_combined placement runs at
    # phase 10.5, BEFORE the amplifier, and gets overwritten).
    return _make_l10_postop(
        name="l10_jsr_bp_byte3_clear",
        rules_fn=_l10_jsr_bp_byte3_clear_rules,
        reads={
            "OP_JSR", "H1", "IS_BYTE", "HAS_SE",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        requires={"after": "tail_bit32_result_correction"},
        suppress=False,
        flag_fn=jsr_bp_byte3_clear_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


# ===========================================================================
# absdiff arg-b LI value byte-0 LO-nibble de-contamination (campaign; DEFAULT
# OFF, opt in C4_ABSDIFF_FIX=1). See ``shared.absdiff_fix_enabled`` for the full
# root. In one line: the 2-arg func arg-``b`` deref (LI at BP-relative 0xFFE0,
# addr byte-0 lo-nibble 0) has its L15 head-0-delivered value LO nibble
# out-competed at the LM head by the ALIASED address lo-nibble on the over-biased
# OUTPUT_LO+0 default cell, so byte-0 decodes ``value & 0xF0`` (b=0x55 -> 0x50).
# ===========================================================================
_ABSDIFF_LI_LO_DOM = 30.0          # per-cell winner-take-all magnitude at S=100:
#     restamped as +DOM/S on the delivered value nibble, -DOM/S on OUTPUT_LO+0.
#     Head-0 delivers the value nibble at OUTPUT_LO+k ~=40 and the address-0
#     contaminant at OUTPUT_LO+0 ~=67.5; +DOM on the value cell and -DOM on +0
#     (both un-normalised residual writes, not softmax) flip the ~+27 deficit.
_ABSDIFF_LI_LO_THRESHOLD = 4.0     # fires only when the arg-b-deref discriminator
#     sum (below) clears it; every non-(arg-b LI) row stays silent.


def _l10_absdiff_argb_li_lo_rules() -> tuple[FFNRule, ...]:
    """15 rules: at the arg-``b`` deref LI row (addr 0xE0) re-stamp the delivered
    NONZERO value LO nibble over the aliased address-0 default.

    Each rule k in 1..15 is gated on the head-0-delivered ``OUTPUT_LO+k`` (the
    value's own LO nibble, present at ~+40 for the arg-``b`` value) AND the
    arg-``b``-deref discriminator, and writes ``+DOM`` to ``OUTPUT_LO+k`` and
    ``-DOM`` to ``OUTPUT_LO+0`` so the genuine nonzero value nibble beats the
    address-0 contaminant. k==0 is intentionally omitted: a genuinely-zero value
    LO nibble MUST leave ``OUTPUT_LO+0`` untouched (no over-fire). The
    discriminator REQUIRES ``ADDR_B0_LO+0`` (lo-nibble 0, i.e. 0xE0 -- excludes
    func arg-``a`` @ 0xE8 which is ``ADDR_B0_LO+8``) AND ``ADDR_B0_HI+14``
    (hi-nibble 0xE -- excludes var_simple ``x`` @ BP+0 == 0x00 which is
    ``ADDR_B0_HI+0``), so it is a genuine arg-``b`` (deeper-frame 0xE0-addressed)
    LI requirement, NOT a broad LI touch.
    """
    OPC_BLOCK = 500.0
    owning_ops = tuple(
        op for op in _L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS if op != "OP_JSR"
    )
    disc: tuple[tuple[str, float], ...] = (
        ("MARK_AX", 100.0),
        # GENUINE LI at MARK_AX (~+5.24 opcode broadcast).
        ("OP_LI", 200.0),
        # 0xE0 address signature: lo-nibble 0 (excludes 0xE8 arg-a) AND
        # hi-nibble 14 (excludes 0x00 var_simple x). Large symmetric weights so
        # the (lo0 AND hi14) address is a genuine requirement.
        ("ADDR_B0_LO+0", 1000.0),
        ("ADDR_B0_HI+14", 1000.0),
        ("ADDR_B0_LO+8", -1000.0),   # NOT arg-a (0xE8)
        # Genuine address-eval LEA rows carry MEM_ADDR_SRC; the LI value row does
        # not. Hard NOT-block so this never touches an address-eval row.
        ("MEM_ADDR_SRC", -1_000_000.0),
        # HARD NOT-block IS_BYTE / every non-AX marker: fire ONLY on the AX
        # value-byte-0 MARKER row (IS_BYTE=0, MARK_AX=1).
        ("IS_BYTE", -1_000_000.0),
        ("MARK_PC", -1_000_000.0),
        ("MARK_SP", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
        # No owning opcode may be live (an ALU/branch step must not be restamped).
        *((op, -OPC_BLOCK) for op in owning_ops),
    )
    rules: list[FFNRule] = []
    for k in range(1, 16):
        rules.append(multi_way_and_rule(
            name=f"l10_absdiff_argb_li_lo_{k}",
            conditions=disc,
            threshold=_ABSDIFF_LI_LO_THRESHOLD,
            # gate on the head-0-delivered value LO nibble at cell k.
            gate=f"OUTPUT_LO+{k}",
            writes=(
                (f"OUTPUT_LO+{k}", _ABSDIFF_LI_LO_DOM),
                ("OUTPUT_LO+0", -_ABSDIFF_LI_LO_DOM),
            ),
        ))
    return tuple(rules)


def make_l10_absdiff_argb_li_lo_op() -> Operation:
    """Flag-gated arg-``b`` deref LI value byte-0 LO de-contamination op.

    Standalone ``PureFFN`` post_op attached to the L25 tail block AFTER
    ``l10_loop_lea_b0_e0`` (so it is the LAST OUTPUT-LO writer at the arg-``b`` LI
    row). Flag-OFF (or non-campaign / golden) produces ZERO rules and appends NO
    post_op -> byte-identical to golden ``b1dcae63``. See
    ``absdiff_fix_enabled`` / ``_l10_absdiff_argb_li_lo_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``b1dcae63``).
    # Run AFTER l10_loop_lea_b0_e0 so this is the LAST OUTPUT-LO writer at the
    # arg-b LI row (it must dominate the L15 head-0 address-0 contamination).
    return _make_l10_postop(
        name="l10_absdiff_argb_li_lo",
        rules_fn=_l10_absdiff_argb_li_lo_rules,
        reads={
            "MARK_AX", "OP_LI", "MEM_ADDR_SRC", "IS_BYTE",
            "ADDR_B0_LO", "ADDR_B0_HI",
            "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
            "OUTPUT_LO",
            *_L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS,
        },
        writes={"OUTPUT_LO"},
        requires={"after": "l10_loop_lea_b0_e0"},
        suppress=False,
        flag_fn=absdiff_fix_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


# ---------------------------------------------------------------------------
# absdiff / func-return AX byte-1 OUTPUT_LO stale-marker de-contamination
# (flag C4_ABSDIFF_RET_BYTE1, DEFAULT OFF). Root + safety in
# ``shared.absdiff_ret_byte1_enabled``. Clean discriminator: the AX value-byte
# rows of the LEV-return context.
# ---------------------------------------------------------------------------

_ABSDIFF_RET_B1_DOM = 500.0        # per-cell winner-take-all magnitude at S=100:
#     -DOM/S on the spurious value nibble OUTPUT_LO+k (k>=1), +DOM/S on
#     OUTPUT_LO+0, so the AX high-byte decode collapses to nibble 0 (byte value
#     0). 500 dominates the stale l6_psh_stack0_marker_final_lo_k leak (and its
#     paired OUTPUT_LO+0 crush, down to ~-200) with wide margin.
# ``OP_LEV`` is a PATH-INDEPENDENT return-step AX-byte-row SELECTOR: measured
# across ALL 25 absdiff at the ADJ/return step it is 1.004..1.167 on the AX byte
# rows, <=0.909 on the SP byte rows, <=0.708 on the BP byte rows, and 0 on every
# non-return step (both the ``a>b`` true and false paths). Weighting OP_LEV at
# _OP_LEV_W and setting the threshold at _OP_LEV_W * 0.95 makes the AND fire ONLY
# where OP_LEV>=~0.95 -> exactly the AX byte rows of the return step, with wide
# margin below the SP/BP rows (which ALSO read OUTPUT_LO, so must be excluded).
# The force is UNCONDITIONAL on the leak magnitude (no OUTPUT_LO+k gate): the leak
# magnitude varies 2.97..9.1 across the corpus AND is paired with a crushed
# OUTPUT_LO+0, so a leak-magnitude gate misses the weak cases; since every
# absdiff/func/nested return is <=255 (AX bytes 1..3 == 0), forcing nibble-0 on
# the AX high-byte rows is unconditionally correct.
_ABSDIFF_RET_B1_OP_LEV_W = 100.0
_ABSDIFF_RET_B1_THRESHOLD = _ABSDIFF_RET_B1_OP_LEV_W * 0.95   # OP_LEV >= 0.95
# The l6-crush flag BAND: fire the ABSDIFF_RET_LEAK indicator when OUTPUT_LO+0 is
# crushed into the MODERATE band -1000 <= OUTPUT_LO+0 <= -10 (i.e.
# 10 <= -(OUTPUT_LO+0) <= 1000). The absdiff single-byte leak crushes OUTPUT_LO+0
# to -46..-199 (fires); a genuine func_mul return with a REAL byte-1 either leaves
# it CLEAN POSITIVE (+3.91, below the low bound -> dark) OR SATURATES it hugely-
# negative (~-6e9, above the high bound -> dark). Both passing func_mul cases
# (id603 +3.91, id605 -6e9) are thus excluded. The band = step(>=10) MINUS
# step(>=1000) on -(OUTPUT_LO+0), so the saturated tail cancels to 0. The
# corrector reads the flag only as a gate>0 / gate==0 test.
_ABSDIFF_RET_LEAK_CRUSH_LO = 10.0     # min crush magnitude to fire.
_ABSDIFF_RET_LEAK_CRUSH_HI = 1000.0   # above this = saturated func_mul -> DRIVE < 0.
_ABSDIFF_RET_LEAK_INDICATOR = 2.0     # write_value -> ~1.0 flag magnitude.


def _l10_absdiff_ret_byte1_flag_rules() -> tuple[FFNRule, ...]:
    """2 precursor rules: write the BANDED ``ABSDIFF_RET_LEAK`` l6-crush indicator.

    Net flag = step(-(OUTPUT_LO+0) >= 10)*1 - step(-(OUTPUT_LO+0) >= 1000)*2. The
    corrector reads it as a MULTIPLICATIVE gate:
      * clean POSITIVE OUTPUT_LO+0 (func_mul +3.91, ``-x < 10``): both dark ->
        flag 0 -> gate 0 -> corrector DARK (byte-1 preserved).
      * MODERATE crush (absdiff -46..-199, ``10 <= -x < 1000``): only the +1 rule
        fires -> flag > 0 -> gate > 0 -> corrector FIRES (byte-1 -> 0).
      * SATURATED crush (genuine multi-byte func_mul ~-6e9, ``-x >= 1000``): BOTH
        fire and the -2 rule DOMINATES (2x slope) -> flag DRIVEN hugely NEGATIVE
        -> gate < 0. A negative multiplicative gate makes ``silu(up)*gate`` NEGATIVE
        so the corrector's writes FLIP SIGN (+DOM to OUTPUT_LO+k, -DOM to +0) — but
        on the saturated row those +-DOM (~500) are NEGLIGIBLE against the genuine
        ~6e9 delivered nibble, so the real byte-1 is PRESERVED. (The 2x cancel is
        load-bearing: an equal-weight cancel leaves a constant POSITIVE residual at
        saturation — silu is linear, not a true step — which would fire the gate and
        zero the genuine byte-1.)
    """
    # multi_way_and_rule fires on score >= threshold; we need -(OUTPUT_LO+0) >= T,
    # i.e. conditions=((OUTPUT_LO+0, -1.0),) with threshold=T -> silu score is
    # -OUTPUT_LO+0.
    return (
        multi_way_and_rule(
            name="l10_absdiff_ret_leak_flag_lo",
            conditions=(("OUTPUT_LO+0", -1.0),),
            threshold=_ABSDIFF_RET_LEAK_CRUSH_LO,
            writes=(("ABSDIFF_RET_LEAK", _ABSDIFF_RET_LEAK_INDICATOR / 100.0),),
        ),
        # 2x weight so at saturation the net flag is DRIVEN NEGATIVE (gate<0 =>
        # corrector dark on the genuine multi-byte row).
        multi_way_and_rule(
            name="l10_absdiff_ret_leak_flag_hi_cancel",
            conditions=(("OUTPUT_LO+0", -1.0),),
            threshold=_ABSDIFF_RET_LEAK_CRUSH_HI,
            writes=(("ABSDIFF_RET_LEAK", -2.0 * _ABSDIFF_RET_LEAK_INDICATOR / 100.0),),
        ),
    )


def _l10_absdiff_ret_byte1_rules() -> tuple[FFNRule, ...]:
    """15 rules: on the AX byte-1 leak row of the LEV-return step, force the
    OUTPUT_LO high-byte nibble to 0 (AX byte-1/2/3 -> value 0).

    Each rule k in 1..15 fires iff the return-step AX-byte-row discriminator is
    satisfied, and writes ``-DOM`` to ``OUTPUT_LO+k`` and ``+DOM`` to
    ``OUTPUT_LO+0`` so the AX high bytes decode nibble-0 (byte value 0). k==0 is
    omitted (the clean nibble-0 default is left untouched). The discriminator
    REQUIRES ``OP_LEV`` at weight _OP_LEV_W with a matched threshold so the AND
    clears ONLY where ``OP_LEV >= ~0.95`` — measured to be EXACTLY the AX byte
    rows of the ADJ/return step (AX 1.004..1.167, SP <=0.909, BP <=0.708, 0
    elsewhere), a path-INDEPENDENT selector — AND ``IS_BYTE`` (a value byte ->
    excludes the AX byte-0 / MARK_AX row so byte-0 keeps its real nibble) AND the
    BOUNDED ``ABSDIFF_RET_LEAK`` l6-crush flag (1 on the absdiff single-byte leak,
    0 on a genuine func_mul multi-byte return whose byte-1 is REAL — so func_mul
    is NOT touched), and HARD-NOT-blocks ``PSH_AT_SP`` / ``MARK_STACK0`` (the
    mis-firing L6 ``l6_psh_stack0_marker_final`` rule's own gates, provably OFF
    here) and every non-AX register marker, so it is a genuine leaking-AX-value-
    byte return-step requirement, NOT a broad OUTPUT-LO touch.
    """
    disc: tuple[tuple[str, float], ...] = (
        # Return-step AX-byte-row selector: OP_LEV >= ~0.95 (AX 1.004..1.167 at
        # the ADJ/return step, SP <=0.909, BP <=0.708, 0 elsewhere). The weight +
        # matched threshold make this a hard >=0.95 gate; path-INDEPENDENT.
        ("OP_LEV", _ABSDIFF_RET_B1_OP_LEV_W),
        # value byte (excludes the MARK_AX byte-0 predictor row, IS_BYTE=0). Small
        # positive weight so it never lifts an OP_LEV<0.95 row over the threshold.
        ("IS_BYTE", 1.0),
        # HARD NOT-block the L6 marker rule's own gates: if this were a GENUINE
        # PSH-STACK0 marker step the corrector must NOT fire.
        ("PSH_AT_SP", -1_000_000.0),
        ("MARK_STACK0", -1_000_000.0),
        # HARD NOT-block every non-AX register marker (only AX high-byte rows).
        ("MARK_PC", -1_000_000.0),
        ("MARK_AX", -1_000_000.0),
        ("MARK_SP", -1_000_000.0),
        ("MARK_BP", -1_000_000.0),
        ("MARK_MEM", -1_000_000.0),
        # Never touch an address-eval / memory row.
        ("MEM_ADDR_SRC", -1_000_000.0),
    )
    rules: list[FFNRule] = []
    for k in range(1, 16):
        rules.append(multi_way_and_rule(
            name=f"l10_absdiff_ret_byte1_{k}",
            conditions=disc,
            threshold=_ABSDIFF_RET_B1_THRESHOLD,
            # MULTIPLICATIVE gate on the BOUNDED l6-crush flag: flag=0 (a genuine
            # func_mul multi-byte return, OUTPUT_LO+0 clean positive) => gate=0 =>
            # NO write, so func_mul's real byte-1 is preserved; flag~1 (the absdiff
            # single-byte leak, OUTPUT_LO+0 crushed) => gate~1 => the write fires.
            # A multiplicative gate (not an additive condition) makes the flag a
            # HARD requirement independent of the large OP_LEV score.
            gate="ABSDIFF_RET_LEAK",
            writes=(
                (f"OUTPUT_LO+{k}", -_ABSDIFF_RET_B1_DOM),
                ("OUTPUT_LO+0", _ABSDIFF_RET_B1_DOM),
            ),
        ))
    return tuple(rules)


def make_l10_absdiff_ret_byte1_flag_op() -> Operation:
    """Flag-gated PRECURSOR: writes the bounded ``ABSDIFF_RET_LEAK`` l6-crush flag.

    Runs AFTER ``tail_bit32_result_correction`` (so it reads the final crushed
    OUTPUT_LO+0) and BEFORE ``make_l10_absdiff_ret_byte1_op`` (which gates on the
    flag). Flag-OFF -> ZERO rules, NO post_op, NO band -> byte-identical to golden
    ``f725c06e``. See ``_l10_absdiff_ret_byte1_flag_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``f725c06e``).
    return _make_l10_postop(
        name="l10_absdiff_ret_byte1_flag",
        rules_fn=_l10_absdiff_ret_byte1_flag_rules,
        reads={"OUTPUT_LO"},
        writes={"ABSDIFF_RET_LEAK"},
        requires={"after": "tail_bit32_result_correction"},
        suppress=False,
        flag_fn=absdiff_ret_byte1_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


def make_l10_absdiff_ret_byte1_op() -> Operation:
    """Flag-gated absdiff / func-return AX byte-1 OUTPUT_LO de-contamination op.

    Standalone ``PureFFN`` post_op attached to the L25 tail block AFTER
    ``l10_absdiff_ret_byte1_flag`` (so the ``ABSDIFF_RET_LEAK`` gate is fresh) and
    thus after ``tail_bit32_result_correction`` — it is the LAST OUTPUT-LO writer
    at the AX return-byte row, dominating the L6 stale marker leak that persists
    into the tail. Flag-OFF (or non-campaign / golden) produces ZERO rules and
    appends NO post_op -> byte-identical to golden ``f725c06e``. See
    ``absdiff_ret_byte1_enabled`` / ``_l10_absdiff_ret_byte1_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``f725c06e``).
    # Run AFTER the flag op (fresh ABSDIFF_RET_LEAK), hence after
    # tail_bit32_result_correction -> this is the LAST OUTPUT-LO writer at the AX
    # return-byte row (it must dominate the L6 stale-marker OUTPUT_LO leak that
    # persists into the tail block).
    return _make_l10_postop(
        name="l10_absdiff_ret_byte1",
        rules_fn=_l10_absdiff_ret_byte1_rules,
        reads={
            "OP_LEV", "IS_BYTE", "ABSDIFF_RET_LEAK", "PSH_AT_SP", "MARK_STACK0",
            "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
            "MEM_ADDR_SRC", "OUTPUT_LO",
        },
        writes={"OUTPUT_LO"},
        requires={"after": "l10_absdiff_ret_byte1_flag"},
        suppress=False,
        flag_fn=absdiff_ret_byte1_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


# ===========================================================================
# loop back-edge SI-step value-byte-row MARKER-residue clear (THE FIRST loop
# back-edge control-flow desync; campaign).
#
# See ``shared.loop_si_byterow_marker_clear_enabled`` for the full root + GPU
# AR-trace attribution. In one line: at the in-loop ``SI`` (store) step the
# ``MEM_STORE`` / ``OP_SI`` opcode dims carry a tiny NEGATIVE residue (~ -2.4e-3
# / -4.2e-4) on the SP/PC VALUE-BYTE rows where they SHOULD be exactly 0. The
# L25 tail bank (``tail_bit32_result_correction``) reads those dims at ``-1e8``
# (NOT-blockers assuming 0); the residue * -1e8 flips the bank's silu gate
# positive -> the whole OUTPUT-decode band fires asymmetrically and explodes to
# ~ -7.4e11 -> the LM byte head is crushed all-negative -> a register-MARKER
# token wins by default -> the SI step emits 41 tokens not 30 -> the fixed-30
# slicer mis-frames the next step (loop-condition ``LEA &i``: PC reads 114 not
# 106 == a +16 instead of +8 PC advance, the brief's symptom).
#
# FIX: ADD a small POSITIVE bias (+0.5) to ``MEM_STORE`` and ``OP_SI`` on EVERY
# value-byte row (``IS_BYTE``), scheduled IMMEDIATELY BEFORE the tail bank. The
# bias swamps the residue so ``residue + 0.5`` * -1e8 = ~ -5e7 keeps the bank's
# ``up`` deeply NEGATIVE (silu ~ 0 == OFF), exactly as on a clean (PSH/IMM) step.
# It fires ONLY on value-byte rows (``IS_BYTE`` AND every marker hard-blocked),
# so the tail's MEM-store ADDRESS materializers (which fire on MARK_MEM/MARK_SP
# MARKER rows, IS_BYTE=0) are untouched.
# EFFECTIVE residual bias landed on MEM_STORE / OP_SI when the IS_BYTE unit
# fires. CRITICAL MAGNITUDE: large vs the ~2.4e-3 SI residue (so ``residue +
# bias`` * -1e8 = ~ -2e6 swamps the +2.4e5 that flips the SP-row tail bank ON),
# yet SMALL enough that it does NOT flip the AX HIGH-BYTE sign-extension
# materializer NOR the multi-byte ADD high-byte adder (BOTH read MEM_STORE /
# OP_SI) -- at an effective bias 0.5 the AX byte 2/3 over-sign-extend to 0xFF
# and at 100 the ADD high byte is lost (768 -> 256). Measured safe window is
# ~0.005..0.1 (``_probe_loopsum_backedge`` bias sweep, hook form); 0.02 is the
# robust middle.
#
# The FFN unit's silu(up) SATURATES to ~5000 on a fired (IS_BYTE) value-byte row
# (the IS_BYTE*100 condition * S=100 step margin), and that saturation value is
# STABLE for every value-byte row (IS_BYTE is exactly 1 there). So the LANDED
# residual is ``silu(up) * W_down = 5000 * _LOOP_SI_CLEAR_WDOWN``; we choose
# W_down = 0.02 / 5000 = 4e-6 to land the effective +0.02 bias.
_LOOP_SI_CLEAR_SILU_SAT = 5000.0  # measured saturated silu(up) of the fired
#                                   IS_BYTE step unit (deterministic: IS_BYTE==1
#                                   on every value-byte row).
_LOOP_SI_CLEAR_EFFECTIVE_BIAS = 0.02
_LOOP_SI_CLEAR_WDOWN = _LOOP_SI_CLEAR_EFFECTIVE_BIAS / _LOOP_SI_CLEAR_SILU_SAT
_LOOP_SI_CLEAR_MARKER_BLOCK = 1_000_000.0


def _l10_loop_si_byterow_marker_clear_rules() -> tuple[FFNRule, ...]:
    """2 rules: bias MEM_STORE / OP_SI by +0.5 on value-byte rows.

    Each is a one-condition AND gated on ``IS_BYTE`` with every register MARKER
    hard NOT-blocked, so the op fires ONLY on a value-byte row (never a marker
    row, never a STEP_END boundary). The write is a constant POSITIVE bias into
    the shared opcode dim that the L25 tail bank reads at ``-1e8`` -- enough to
    keep the bank's silu gate OFF (the clean-step behaviour) regardless of the
    sign/size of the upstream SI residue.
    """
    disc: tuple[tuple[str, float], ...] = (
        ("IS_BYTE", 100.0),
        ("MARK_PC", -_LOOP_SI_CLEAR_MARKER_BLOCK),
        ("MARK_AX", -_LOOP_SI_CLEAR_MARKER_BLOCK),
        ("MARK_SP", -_LOOP_SI_CLEAR_MARKER_BLOCK),
        ("MARK_BP", -_LOOP_SI_CLEAR_MARKER_BLOCK),
        ("MARK_STACK0", -_LOOP_SI_CLEAR_MARKER_BLOCK),
        ("MARK_MEM", -_LOOP_SI_CLEAR_MARKER_BLOCK),
    )
    rules: list[FFNRule] = []
    for target in ("MEM_STORE", "OP_SI"):
        rules.append(multi_way_and_rule(
            name=f"l10_loop_si_byterow_clear_{target.lower()}",
            conditions=disc,
            threshold=50.0,  # IS_BYTE(100) alone (-> 100) crosses; any marker
            #                  row drops by 1e6 and is excluded.
            writes=((target, _LOOP_SI_CLEAR_WDOWN),),
        ))
    return tuple(rules)


def make_l10_loop_si_byterow_marker_clear_op() -> Operation:
    """Flag-gated L10 loop back-edge value-byte-row MEM_STORE/OP_SI bias op.

    Standalone ``PureFFN`` post_op attached to the L25 tail block; the scheduler
    orders it BEFORE ``tail_bit32_result_correction`` because that op READS
    ``MEM_STORE`` / ``OP_SI`` (producer-before-consumer). Flag-off (or
    non-campaign / golden) produces ZERO rules and appends NO post_op ->
    byte-identical to golden ``5acb3d23``. See
    ``loop_si_byterow_marker_clear_enabled`` /
    ``_l10_loop_si_byterow_marker_clear_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical golden ``5acb3d23``).
    # Writes the shared opcode dims the tail bank reads; the produces/consumes dep
    # (tail_bit32 reads MEM_STORE / OP_SI) orders this op BEFORE the tail bank so
    # the bias lands before the -1e8 NOT-blocker reads it.
    #
    # ``suppress=False`` is LOAD-BEARING: the op is INTENDED to fire on value-byte
    # rows (IS_BYTE=1); ``_suppress_ffn_on_step_boundary`` would gate exactly those
    # rows off. The IS_BYTE gate + the hard marker NOT-blocks already restrict it
    # to value-byte rows.
    return _make_l10_postop(
        name="l10_loop_si_byterow_marker_clear",
        rules_fn=_l10_loop_si_byterow_marker_clear_rules,
        reads={
            "IS_BYTE", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
            "MARK_STACK0", "MARK_MEM",
        },
        writes={"MEM_STORE", "OP_SI"},
        suppress=False,
        flag_fn=loop_si_byterow_marker_clear_enabled,
        spec_section="BLOG_SPEC.md#registers",
    )


# ===========================================================================
# In-loop opcode-fetch ADDR_KEY top-byte over-count clamp
# (flag C4_LOOP_LI_FETCH_ADDRKEY_CLAMP, default ON in the campaign config).
# Despite the l10_ file home (alongside the loop_* family) this op binds to the
# L4 FFN dep anchor (block 4) so it runs AFTER the L3 doubling (block 3) and
# BEFORE the L5 opcode fetch (block 6). See
# ``shared.loop_li_opcode_fetch_addrkey_clamp_enabled`` for the full root + GPU
# trace attribution. In one line: the L3 ``_pc_byte1_prev_head_spec`` stages a
# CLEAN_EMBED nibble into ``ADDR_KEY+32`` (the 3rd code-address nibble, for the
# 12-bit fetch match) ON TOP of the positional ADDR_KEY one-hot at certain zero
# prompt code bytes -> that cell reads 2.0 not 1.0 -> in the L5 opcode fetch
# CAM (head 1) the doubled cell's slot-35 contribution (~803 vs ~402) lets a
# WRONG zero code byte win a razor-thin tie (112.94 vs 112.71) over the true SI
# opcode byte for the in-loop store PCs (58/90, low nibble 0xa) -> OPCODE_BYTE
# decodes 0x00 not 0x0B -> OP_SI dead -> MEM_STORE never set on the SI store ->
# the in-loop LI value-load CAM (L15) finds no store and returns 0.
#
# FIX: clamp ``ADDR_KEY+32+k`` back to 1.0 on prompt code-byte rows by
# SUBTRACTING the over-count. Per cell k a unit fires iff the cell is DOUBLED
# (>= ~1.5) on an ``IS_BYTE`` row with every register MARKER hard NOT-blocked
# (so PROMPT code bytes only -- never an emitted PC/AX/SP value byte where the
# L3 head legitimately SINGLE-stages the nibble at 1.0, which stays < 1.5 and so
# never fires). A clean single cell is untouched -> short programs + the
# legitimate 12-bit code-address match are byte-identical; only the doubled
# stray-zero-byte cells are restored, after which the true SI opcode row wins
# the fetch p=1.000.
#
# CALIBRATION: like ``_l10_loop_si_byterow_marker_clear_*`` the fired unit's
# silu(up) SATURATES to a stable value on a fired row (the IS_BYTE*100 + doubled
# -cell*100 AND margin * S=100). The landed residual is ``silu(up) * W_down``;
# we choose W_down to subtract an effective ~ -1.5 from the doubled cell. The
# clamp is one-sided-safe: even a 2x miscalibration lands the cell in
# [-2.0, 0.5], all of which leave it BELOW the correct (non-doubled, 2-nibble-
# matching) SI row, so the fetch still resolves correctly. ``ADDR_KEY+32`` on a
# stray zero code byte is consumed ONLY by the L5 fetch K (making that row less
# attractive == the intended effect), so over-subtraction has no other reader.
_LOOP_LI_CLAMP_SILU_SAT = 5000.0  # measured saturated silu(up) of a fired
#                                   IS_BYTE-gated L4 step unit (deterministic).
_LOOP_LI_CLAMP_EFFECTIVE_SUB = 1.5  # effective amount subtracted from the
#                                     doubled ADDR_KEY+32 cell (restores ~1.0).
_LOOP_LI_CLAMP_WDOWN = -_LOOP_LI_CLAMP_EFFECTIVE_SUB / _LOOP_LI_CLAMP_SILU_SAT
_LOOP_LI_CLAMP_MARKER_BLOCK = 1_000_000.0


def _l10_loop_li_opcode_fetch_addrkey_clamp_rules() -> tuple[FFNRule, ...]:
    """16 rules: clamp each ``ADDR_KEY+32+k`` cell back to ~1.0 on doubled
    prompt code-byte rows.

    Per cell k an AND fires iff (the cell is DOUBLED to >= ~1.5) AND (IS_BYTE)
    AND (no register marker). The cell condition weight (100) means a clean
    single cell (1.0 -> +100) plus IS_BYTE (+100) totals 200 (< threshold 250),
    while a doubled cell (2.0 -> +200) plus IS_BYTE (+100) totals 300 (>= 250)
    -> fires. Every register MARKER carries a hard -1e6 NOT-block so an emitted
    PC/AX/SP/.. value byte (where the L3 head single-stages the nibble) can never
    be clamped. The write subtracts the over-count from the SAME cell.
    """
    rules: list[FFNRule] = []
    for k in range(16):
        cell = f"ADDR_KEY+{32 + k}"
        disc: tuple[tuple[str, float], ...] = (
            # The cell value gates firing: clean=1.0 -> +100, doubled=2.0 ->
            # +200. Threshold 250 (below) admits only the doubled case.
            (cell, 100.0),
            ("IS_BYTE", 100.0),
            ("MARK_PC", -_LOOP_LI_CLAMP_MARKER_BLOCK),
            ("MARK_AX", -_LOOP_LI_CLAMP_MARKER_BLOCK),
            ("MARK_SP", -_LOOP_LI_CLAMP_MARKER_BLOCK),
            ("MARK_BP", -_LOOP_LI_CLAMP_MARKER_BLOCK),
            ("MARK_STACK0", -_LOOP_LI_CLAMP_MARKER_BLOCK),
            ("MARK_MEM", -_LOOP_LI_CLAMP_MARKER_BLOCK),
        )
        rules.append(multi_way_and_rule(
            name=f"l10_loop_li_fetch_addrkey_clamp_{k}",
            conditions=disc,
            # cell=1 + IS_BYTE = 200 (NO fire); cell=2 + IS_BYTE = 300 (fire);
            # any marker drops by 1e6 (excluded).
            threshold=250.0,
            writes=((cell, _LOOP_LI_CLAMP_WDOWN),),
        ))
    return tuple(rules)


def make_l10_loop_li_opcode_fetch_addrkey_clamp_op() -> Operation:
    """Flag-gated L4 in-loop opcode-fetch ADDR_KEY top-byte over-count clamp.

    Standalone ``PureFFN`` post_op bound to the L4 FFN dep anchor; it runs AFTER
    the L3 doubling (block 3) and BEFORE the L5 opcode fetch (block 6). Flag-off
    (or non-campaign / golden) produces ZERO rules and appends NO post_op ->
    byte-identical to golden. See ``loop_li_opcode_fetch_addrkey_clamp_enabled``
    / ``_l10_loop_li_opcode_fetch_addrkey_clamp_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical).
    #
    # ``target_op_name="layer3_carry_forward_attn"`` binds to the L3
    # carry_forward attention op (block 3) -- the HOME of the
    # ``_pc_byte1_prev_head_spec`` head that creates the ADDR_KEY+32 doubling. A
    # block post_op here runs at block 3 AFTER that head, so the clamp lands in the
    # residual stream BEFORE the L5 opcode fetch (block 6) reads it. (The L4 FFN
    # dep anchor resolves to block 6 in the dynamic schedule, so an L4-anchored
    # post_op would run AFTER the fetch -- too late.)
    #
    # ``suppress=False`` is LOAD-BEARING: the op fires on prompt code-byte rows
    # (IS_BYTE=1, no marker); ``_suppress_ffn_on_step_boundary`` would gate exactly
    # those rows off.
    return _make_l10_postop(
        name="l10_loop_li_fetch_addrkey_clamp",
        rules_fn=_l10_loop_li_opcode_fetch_addrkey_clamp_rules,
        reads={
            "IS_BYTE", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
            "MARK_STACK0", "MARK_MEM", "ADDR_KEY",
        },
        writes={"ADDR_KEY"},
        target_op_name="layer3_carry_forward_attn",
        spec_section="BLOG_SPEC.md#memory",
        suppress=False,
        flag_fn=loop_li_opcode_fetch_addrkey_clamp_enabled,
    )


def _l10_exit_axcarry_rules() -> tuple[FFNRule, ...]:
    """Route AX_CARRY -> OUTPUT on the EXIT/no-clean-opcode AX row (32 units).

    See ``_l10_exit_axcarry_enabled`` for the full root. Per nibble k (0..15)
    one AND unit fires iff:

      * ``MARK_AX`` (the AX marker row) is present, AND
      * ``OP_LEA`` is LEAKED (>= ~1.0 -- it is 0.00 on every healthy AX row, so
        this term is the discriminator that isolates the post-LEV EXIT step),
        AND
      * the matching ``AX_CARRY_{LO,HI}[k]`` one-hot is present, AND
      * NO OUTPUT-owning opcode is live (each contributes a hard -1e6
        NOT-block), AND
      * ``MEM_ADDR_SRC`` is cold (a GENUINE LEA address-eval row carries it; the
        spurious leak does not) -- another hard NOT-block, AND
      * we are not on any non-AX marker row (PC/SP/BP/STACK0/MEM all -1e6).

    Each unit writes ``OUTPUT_{LO,HI}[k] = +DOM`` and every competing nibble
    ``OUTPUT_{LO,HI}[j!=k] = -DOM`` at a magnitude that dominates the ~12.3
    LEA-materializer write so the correct AX_CARRY byte wins the LM-head argmax.
    The gate term uses ``AX_CARRY_{LO,HI}[k]`` as the firing selector so the
    routing is data-dependent (carry nibble -> out nibble).
    """

    # Dominating magnitude: the leaked LEA materializer contributes ~+12.3 to
    # OUTPUT_HI+15. The firing hidden activation is large (~3850: the AND clears
    # threshold with a big margin and ``_suppress_ffn_on_step_boundary`` leaves
    # a ~+4000 MARK_AX residual on the pre-activation), so a small per-cell
    # W_down weight already DOMINATES: +DOM on the matching cell, -DOM on every
    # competitor. DOM=0.02 -> contribution ~+77 vs the ~+12.3 materializer (a
    # decisive ~6x margin) without an O(1e7) runaway in the OUTPUT band.
    DOM = 0.02
    # Margin budget (raw, pre-S). The CLEAN one-hot signals (MARK_*,
    # MEM_ADDR_SRC) are ~1.0 when present and ~0 when absent, so they can use a
    # hard -1e6 NOT-block. The OPCODE flags, however, LEAK a small residue
    # (OP_JSR ~0.02, OP_ENT ~0.16) onto unrelated rows; a hard -1e6 block there
    # would be tripped by the leak. So the opcode NOT-blocks use a MODERATE
    # weight (-OPC_BLOCK) sized so a CLEAN opcode (>= ~0.5) decisively vetoes
    # while the residue leak (<= ~0.3) is absorbed by a wide positive margin:
    #   present  : MARK_AX(100) + OP_LEA(60 * ~1.79 = ~107) + gate(~0.94) ~= 208
    #   threshold: 150  ->  margin ~+58
    #   leak veto: 0.3 * OPC_BLOCK(500) = -150 (survives a single leaked op),
    #              a clean op 0.5 * 500 = -250 (vetoes decisively).
    OPC_BLOCK = 500.0
    rules: list[FFNRule] = []
    for nibble_label, carry_dim, out_dim in (
        ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for k in range(16):
            conditions: list[tuple[str, float]] = [
                ("MARK_AX", 100.0),
                # Leaked-OP_LEA discriminator: the leak is ~1.7-1.8 on the bug
                # row and EXACTLY 0.00 on every healthy AX row, so weight 60
                # makes it the load-bearing required term (~107 here).
                ("OP_LEA", 60.0),
                # OUTPUT-owning opcode NOT-blocks (real ALU/IMM/branch rows).
                # Moderate weight: a clean opcode (>= ~0.5) vetoes; a residue
                # leak (<= ~0.3) is absorbed by the +58 positive margin.
                *((op, -OPC_BLOCK) for op in _L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS),
                # A genuine LEA address-eval row carries MEM_ADDR_SRC (~1.0); the
                # spurious post-LEV leak does not (~0.0). Clean one-hot -> hard
                # NOT-block.
                ("MEM_ADDR_SRC", -1_000_000.0),
                # Non-AX marker rows are off-limits (clean one-hot markers).
                ("MARK_PC", -1_000_000.0),
                ("MARK_SP", -1_000_000.0),
                ("MARK_BP", -1_000_000.0),
                ("MARK_STACK0", -1_000_000.0),
                ("MARK_MEM", -1_000_000.0),
            ]
            writes: list[tuple[str, float]] = []
            for j in range(16):
                writes.append((f"{out_dim}+{j}", (DOM if j == k else -DOM)))
            rules.append(multi_way_and_rule(
                name=f"l10_exit_axcarry_{nibble_label}_{k}",
                conditions=tuple(conditions),
                threshold=150.0,
                gate=f"{carry_dim}+{k}",
                gate_weight=1.0,
                writes=tuple(writes),
            ))
    return tuple(rules)


def make_l10_exit_axcarry_op() -> Operation:
    """Flag-gated L10 EXIT/no-clean-opcode AX_CARRY -> OUTPUT override op.

    Standalone ``PureFFN`` post_op attached to the L25 tail block AFTER
    ``tail_bit32_result_correction`` (it reads OUTPUT_LO/HI which that op
    writes, so the produces/consumes dep orders it last). Flag-off (default)
    produces ZERO rules and appends NO post_op -> byte-identical to HEAD. Flag
    ON ships ``test_simple_function`` (-> 42) under the 6 LEV flags. See
    ``_l10_exit_axcarry_enabled`` / ``_l10_exit_axcarry_rules``.
    """
    # R-FRAME INCR-1 style glue collapse: thin call over ``_make_l10_postop``.
    # Flag-off -> the shared noop Operation (byte-identical to HEAD). Must run
    # AFTER tail_bit32_result_correction (the OUTPUT producer it overrides),
    # mirroring make_l10_add_high_byte_adder_op's ordering (it reads OUTPUT so the
    # scheduler orders it last).
    return _make_l10_postop(
        name="l10_exit_axcarry",
        rules_fn=_l10_exit_axcarry_rules,
        reads={
            "MARK_AX", "OP_LEA", "MEM_ADDR_SRC",
            "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
            "AX_CARRY_LO", "AX_CARRY_HI",
            # Read OUTPUT so the scheduler orders this AFTER
            # tail_bit32_result_correction (the OUTPUT producer it overrides).
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
            *_L10_EXIT_AXCARRY_OUTPUT_OWNING_OPS,
        },
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        requires={"after": "tail_bit32_result_correction"},
        suppress=True,
        flag_fn=_l10_exit_axcarry_enabled,
        spec_section="BLOG_SPEC.md#function-call",
    )


def make_l10_post_op_attach_op(alu_mode: str = "lookup") -> Operation:
    """Block-level op: attach L10 post_op modules onto block.post_ops.

    Migrates the inline `model.blocks[10].post_ops.append(...)` calls in
    `set_vm_weights` for both lookup and efficient ALU modes into a compiler
    block op. The attached modules are the structural post-FFN passes that
    `_expand_wrapper_blocks` later splits into their own blocks.

    Modules attached (lookup mode):
      BinaryOpByteZeroingPostOp,
      AddSubBytePropagationPostOp,
      CarryPropagationPostOp x3 (byte 0 no-cascade, bytes 1-2 cascade),
      BitwiseBytePropagationPostOp.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    Modules attached (efficient mode):
      BinaryOpByteZeroingPostOp,
      AddSubBytePropagationPostOp,
      CarryPropagationPostOp x3,
      BitwiseBytePropagationPostOp,
      ComparisonCombine.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    The existing `make_l10_post_ops_combined` is unrelated: it bakes the
    LOGIC of the FFN-style post_ops into a single phase-10.5 FFN (a parallel
    representation), not the attached module list. Both can coexist.

    phase=10.7: runs after L10 FFN bake (phase=10) and the combined FFN
    (phase=10.5), but well before structural post-passes (1100+).
    """
    if alu_mode not in ("lookup", "efficient"):
        raise ValueError(
            f"alu_mode must be 'lookup' or 'efficient'; got {alu_mode!r}"
        )

    def bake(block, dim_positions, S):
        from ...vm_step import (
            BinaryOpByteZeroingPostOp,
            AddSubBytePropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
            _SetDim,
        )
        # Derive d_model from the block's residual-stream width. Preferred
        # source is ``block.attn.dim`` (always set to the model's d_model on
        # the AutoregressiveAttention used by every TransformerBlock); the
        # legacy ``block.ffn.W_up.shape[1]`` fallback is incorrect for
        # efficient-mode L10 where ``block.ffn`` is a ``PureNeuralALU``
        # subclass (e.g. ``ALUAndOrXor``) that has no ``W_up`` attribute,
        # silently bottoming out at the hard-coded ``512`` constant. With
        # the dynamic dim allocator (e.g. ``alu_mode='efficient'`` lifting
        # d_model to 800 to accommodate the V2/G7 LEV detector residual
        # band) this caused a stale ``BD.TEMP + 8 = 706`` to overflow the
        # 512-wide post-op ``W_up`` row and raise ``IndexError``.
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

        # Pass dim_positions so each post-op bakes against the compact layout
        # rather than legacy `_SetDim` positions. Without this, the post-ops
        # write to / read from `_SetDim.OUTPUT_LO/HI/CARRY/H1/OP_*` etc.,
        # which alias unrelated compact dims (e.g. `_SetDim.H1+1=68` aliases
        # compact `EMBED_HI[15]`; `_SetDim.CARRY=392` aliases a different
        # compact slot, etc.), corrupting OUTPUT/CARRY/CMP flags and silently
        # zeroing or scrambling the binary-op result. Threading dim_positions
        # to all 4 post-op classes is the L10 counterpart of the L1 fix in
        # commit 5fc519d (BinaryOpByteZeroingPostOp).
        zeroing = BinaryOpByteZeroingPostOp(
            d_model=d_model, S=S, dim_positions=dim_positions
        )
        _suppress_ffn_on_step_boundary(zeroing, dim_positions, S)
        block.post_ops.append(zeroing)
        BD = _as_setdim_proxy(dim_positions) if isinstance(dim_positions, dict) else _SetDim
        addsub = AddSubBytePropagationPostOp(
            d_model=d_model,
            S=S,
            dim_positions=dim_positions,
        )
        _strengthen_l10_addsub_wrong_byte_blockers(addsub, BD, S)
        _suppress_l10_addsub_on_wide_alu(addsub, BD, S)
        _suppress_ffn_on_step_boundary(addsub, dim_positions, S)
        block.post_ops.append(addsub)
        # Phase 7.C cut: the three carry/borrow post-ops are now authored
        # declaratively via ``_l10_carry_propagation_rules`` lowered into a
        # bare PureFFN by ``_build_l10_carry_post_op``, byte-identically to
        # the legacy ``CarryPropagationPostOp._bake_weights`` (gate:
        # ``tools/verify_carry_migration.py`` -- element-wise tensor diff
        # = 0 + lowering-contract OK). The post-construction strengthen /
        # suppress helpers run unchanged on the resulting weights.
        carry0 = _build_l10_carry_post_op(
            d_model=d_model, S=S, byte_idx=0, cascade=False,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry0, BD, byte_idx=0, S=S)
        _strengthen_l10_first_carry_delta(carry0, BD)
        _suppress_ffn_on_step_boundary(carry0, dim_positions, S)
        block.post_ops.append(carry0)
        carry1 = _build_l10_carry_post_op(
            d_model=d_model, S=S, byte_idx=1, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry1, BD, byte_idx=1, S=S)
        _suppress_ffn_on_step_boundary(carry1, dim_positions, S)
        block.post_ops.append(carry1)
        carry2 = _build_l10_carry_post_op(
            d_model=d_model, S=S, byte_idx=2, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry2, BD, byte_idx=2, S=S)
        _suppress_ffn_on_step_boundary(carry2, dim_positions, S)
        block.post_ops.append(carry2)
        bitwise = BitwiseBytePropagationPostOp(
            d_model=d_model, S=S, dim_positions=dim_positions
        )
        _suppress_ffn_on_step_boundary(bitwise, dim_positions, S)
        block.post_ops.append(bitwise)
        if alu_mode == "efficient":
            # Pass the model's actual d_model so the underlying PureFFN's
            # Linear input dim matches the residual stream width. Without
            # this, ComparisonCombine builds a Linear(512, 18) which fails
            # forward when d_model != 512 (e.g., pin_io_only=True paths).
            compare = ComparisonCombine(
                d_model=d_model, S=S, dim_positions=dim_positions
            )
            _suppress_ffn_on_step_boundary(compare, dim_positions, S)
            block.post_ops.append(compare)
        # DIV/MOD post_op (FlattenedDivMod) appended by
        # ``make_l10_alu_divmod_install_op`` (phase=10.8). Both modes use the
        # same flattened composite — its forward is byte-identical to the
        # previous EfficientDivMod_Neural.

    return Operation(
        name="l10_post_op_attach",
        # Phase 1 (memory cluster fix plan): co-bakes L10
        # block.post_ops alongside the divmod composite install stages
        # (l10_alu_divmod_*). Each appends a different structural FFN to
        # block[10].post_ops — they sequence, not contest. See
        # docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("post_ops_append",),
        reads=set(),
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        # Phase 11.A r3: dropped phase=10.7 — target_op_name and
        # requires['after'] already pin ordering at layer10_carry_relay.
        # Phase 8.A.4 retry: layer_idx=10 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer10_carry_relay``
        # (kind="attn", L10 anchor).
        target_op_name="layer10_carry_relay",
        requires={"after": "layer10_carry_relay"},
        migrated=True,
        declarative_authority="structural_model",
        # Dim-ownership claims: empty. ``bake`` appends 6-7 freshly
        # constructed post_op modules (BinaryOpByteZeroingPostOp,
        # AddSubBytePropagationPostOp, CarryPropagationPostOp x3,
        # BitwiseBytePropagationPostOp, optional ComparisonCombine) to
        # ``model.blocks[10].post_ops`` -- module attach, not per-cell
        # ``(layer, scope, identifier, column)`` writes. Sentinel below
        # documents the structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L10.post_ops[+6 structural FFNs]'},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
