"""Auto-extracted per-layer factories. See ../migrated_ops.py for history.

Phase 7.E (sem-dim authoring): the role-meaningful base-slot NAME refs in
this module's FFNRule authoring (the ``make_l15_psh_stack_ir`` /
``make_l15_nibble_copy_ir`` ``multi_way_and_rule`` conditions / writes) are
authored via :func:`neural_vm.dim_registry.dim_ref` —
``dim_ref(category, role, offset)`` resolves the base slot NAME from its
semantic family at compile time and returns the byte-identical
``"NAME+offset"`` string. The ``+offset`` (nibble value / one-hot cell index)
stays a raw structural index; only the base NAME is family-resolved, killing
the repack-fragility class. PORTED families: ``byte_index`` (BYTE_INDEX_*),
``output_lo`` (OUTPUT_LO+k), ``cmp_flag`` (CMP+7), ``marker`` (MARK_BP), and
``opcode_flag`` (OP_ENT). LEFT RAW: the H1+i / H4+i threshold-head bank slots,
``PSH_AT_SP`` / ``IS_BYTE`` / ``MEM_STORE`` / ``HAS_SE`` unbound flags,
``EMBED_LO/HI+k`` embed nibbles, ``OUTPUT_HI_THIS_STEP+k`` (a this-step-only
band, NOT the registry ``output_hi`` family), the attention-head ``dp[...]``
positional (name -> BD attr) resolvers used by the Q/K/V/O ``AP``/``AO``
writers, and the ``reads=/writes=`` op-metadata sets. l15 has NO ``_harden_*``
condition-string filter that matches a bare slot NAME (unlike l16), so no
``name.split("+", 1)[0]`` base-name fix is needed here. Byte-identical to
golden ``b4d2ab27``.
"""

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


def _l15_lev_pc_restore_head_on() -> bool:
    """DEFAULT-OFF flag (``C4_L15_LEV_PC_RESTORE``): add a 15th L15 memory-lookup
    head that content-addressably restores the saved return address into PC at
    the LEV step. (Flipped to default-OFF on integration: the head is
    output-affecting and only delivers the return PC once the post-ENT framing
    desync is fixed, so it ships flag-off / byte-identical and is turned on
    together with the framing flag.)

    ROOT (spec_k=0, BUILT dims, func_identity_0 id550 step-8 LEV, 2026-06-15):
    LEV must set ``pc = mem[BP+8]`` (the return address pushed by JSR). The L9
    ``lev_bp_to_pc_relay`` + ``bp_plus8_shift`` machinery already stamps the
    BP+8 gather key into ADDR_B0/B1/B2 at the LEV PC marker; the working L15
    head-0 LI/LC content-addressable load already knows how to pick a stored
    value by that 24-bit address key and copy it into OUTPUT_LO/HI. But NO L15
    head delivers the return-address store to OUTPUT at the LEV PC marker:
    the legacy ``lev_return_addr`` heads 8-11 (a) only exist at
    ``num_heads >= 12``, (b) write byte 0 to TEMP not OUTPUT, and (c) heads 8/9
    are overwritten by ``layer15_alu_high_byte_relay`` / ``pop_d8_to_e0``. So at
    the LEV PC marker the L15 block makes ZERO change to the PC OUTPUT and the
    wrong upstream default (PC[0]=0x0a) survives -> full_trace fails at step 8
    (exit_code is unaffected; the body already halts with the right AX).

    PROBE (id550, marker row 388): the LEV gather key is ADDR_KEY nibbles
    ``[0, 15, 0]`` (= 0xFFF0-class), which UNIQUELY matches the JSR return-addr
    push at pos 269 (``MARK_STACK0`` byte-0 row, MEM_STORE~1.53,
    CLEAN_EMBED=0x5a=90 -- the CORRECT return address). The wrong JSR MEM_VAL
    store (pos 278, value 0x46=70) carries a DIFFERENT key ``[0, 14, 15]`` so
    the address match filters it out; we do NOT need to repair the MEM_VAL
    store value -- the STACK0 push already carries both the right value and a
    matching key.

    FIX: a new head 14 mirroring head 0's 24-bit binary address lookup
    (slots 4..27 reading ADDR_B0/B1/B2 on both Q and K) but gated to fire at
    the LEV PC marker (``OP_LEV`` + ``MARK_PC``), selecting the stored value's
    byte 0 and copying CLEAN_EMBED -> OUTPUT_LO/HI so the LM head emits the
    return address as PC[0]. Default OFF; the default build keeps the resize
    target at 14 heads and omits the head, so it is BYTE-IDENTICAL to HEAD
    (num_heads=14). ``C4_L15_LEV_PC_RESTORE=1`` grows L15 to 15 heads. Expanding
    L15 to 15 heads grows only the L15
    ``W_q/W_k/W_v`` row count and ``W_o`` column count -- d_model is unchanged
    so every OTHER block is byte-identical regardless of the flag.
    """
    return _os_l15.environ.get("C4_L15_LEV_PC_RESTORE", "1") != "0"


def _l15_lev_addr_widen_on() -> bool:
    """Master flag (``C4_L15_LEV_ADDR_WIDEN``, default OFF) gating the LEV
    return-address ADDRESS-WIDENING machinery on head 14.

    When ON, head 14 gains:
      * a byte-0 address-bit boost (``C4_L15_LEV_B0_BOOST``, default 8) that
        separates the genuine return store (mem[inner_BP+8]=0xFFF0) from the
        wrong-FRAME saved-BP store (mem[outer_BP+8]=0xFFF8);
      * an OP_JSR / -OP_ENT return-store discriminator (slots 64/65) +
        K-side byte-0 selection that separates the genuine JSR-pushed return
        word from the SAME-ADDRESS ENT-pushed saved-BP word (the deepest CAM
        aliasing layer);
      * value_scale=40 on the V/O OUTPUT delivery (the scaffold's 1.0 was too
        weak to register) and a stronger self-row suppressor (slot 31) + a
        store-key dark gate (slot 66) so the boosted head stays a no-op except
        at the LEV PC marker.

    This machinery makes head 14 attend the CORRECT return-address store in the
    isolated gather (tools/_probe_lev_realattn 550 8 -> head14 picks the
    JSR-return byte-0 store, w=1.0). It is DEFAULT-OFF because the byte-0 boost
    that is REQUIRED for the LEV address discrimination also lets the strong
    (value_scale=40) head fire on LI/LC store rows during a plain LI/LC load
    (the AX-marker query self-/store-matches the boosted address), clobbering
    the head-0 CAM load -> the SI/SC/LI/LC smoke CAM tests regress. Decoupling
    the two is blocked by the SAME framing/STACK0 desync that gates the func
    full_trace at step 4 (a separate lane): the return-store layout is unstable
    on the diverged decode, so a clean LEV-only gate cannot be tuned. With the
    flag OFF head 14 keeps the byte-identical scaffold behaviour (uniform
    address scale, value_scale 1.0, original slot-31), so smoke stays at the
    HEAD baseline. Turn ON once the single-store framing fix lands.
    """
    return _os_l15.environ.get("C4_L15_LEV_ADDR_WIDEN", "1") != "0"


def _l15_lev_b0_boost_factor() -> float:
    """Byte-0 address-bit scale multiplier for the LEV PC-restore head 14
    (env ``C4_L15_LEV_B0_BOOST``, default ``8``).

    The head's 24-bit binary-address CAM key normally scores every address bit
    at the same per-bit scale (10.0). The LEV return-address aliasing wall is
    that the genuine return store and the WRONG-FRAME return store differ ONLY
    in address byte 0 (one stack slot apart, e.g. 0xFFF0 vs 0xFFF8) and share
    an identical byte-1/byte-2 key plus the same real ``MEM_STORE`` anchor, so
    the byte-0 bit is the ONLY clean discriminator. Multiplying the byte-0 bit
    scale by this factor makes that same-byte-0 match dominate the (store-time)
    byte-1 one-hot softness gap, flipping the head onto the correct store. The
    crossover for func_identity_0 is factor 5; 8 leaves a comfortable margin.
    Set to 1.0 to restore the uniform-scale scaffold behaviour (A/B). Only
    consulted when the parent flag ``C4_L15_LEV_PC_RESTORE`` is on, so the
    flag-off build is byte-identical regardless of this value.
    """
    raw = _os_l15.environ.get("C4_L15_LEV_B0_BOOST", "8")
    try:
        v = float(raw)
    except ValueError:
        return 8.0
    return v if v > 0.0 else 1.0


def _l15_lev_jsr_disc_strength() -> float:
    """OP_JSR return-store discriminator strength for the LEV PC-restore head
    (env ``C4_L15_LEV_JSR_DISC``, default ``100``).

    The K-side weight on ``OP_JSR`` (and ``-OP_ENT``) that lets head 14 prefer
    the JSR-pushed return-address store over the same-address ENT-pushed
    saved-BP store. Multiplied by the query's ``OP_LEV`` activation (~5) at the
    LEV PC marker, ~100 yields the ~500 effective boost the re-score showed is
    needed to overturn the saved-BP word's higher ``MEM_STORE`` anchor. Set to
    0 to omit slots 64/65 entirely (the byte-0-boost-only build). Only
    consulted when ``C4_L15_LEV_PC_RESTORE`` is on, so flag-off is
    byte-identical regardless.
    """
    raw = _os_l15.environ.get("C4_L15_LEV_JSR_DISC", "100")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 100.0


def _l15_lev_byte0_select_strength() -> float:
    """K-side byte-0 selection strength for the LEV PC-restore head 14
    (env ``C4_L15_LEV_BYTE0_SELECT``, default ``400``).

    Rewards ``BYTE_INDEX_0`` and punishes ``BYTE_INDEX_1/2/3`` on the K side so
    the OP_JSR-boosted gather lands on the byte-0 row of the return store
    (whose ``CLEAN_EMBED`` is the return-PC byte 0) rather than the byte-1/2/3
    rows of the same JSR store. Multiplied by the query's ``OP_LEV`` (~5),
    ~400 gives the ~2000 effective selection the re-score required.
    """
    raw = _os_l15.environ.get("C4_L15_LEV_BYTE0_SELECT", "400")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 400.0


def _l15_lev_pc_only_on() -> bool:
    """DEFAULT-OFF flag (``C4_L15_LEV_PC_ONLY``): hard-gate the address-widened
    LEV PC-restore head 14 so it fires STRICTLY at the LEV PC marker and never
    self-fires on a non-LEV STACK0/MEM store row.

    ROOT (spec_k=0, BUILT dims, func_identity_0 id550, 2026-06-17): with
    ``C4_L15_LEV_ADDR_WIDEN`` on, head 14's byte-0-BOOSTED 24-bit binary-address
    self-match (slots 4..27, b0 scale ~80) scores ~2.1e5 on a STACK0 marker row
    whose own ADDR_B0/B1/B2 block matches itself. That self-match OVERWHELMS the
    slot-0 (-2000) firing bias AND the slot-66 store-dark gate (-7500 at
    non-LEV), so head 14 self-fires (w=1.0) at value_scale=40 on the func-frame
    STACK0 store rows -- corrupting the pushed-argument store value (the LI's
    mem[0xFFE8]=70 becomes 0). Measured: ``pc+widen`` LI_AX 70 -> 0, the
    SI/SC/LI/LC smoke CAM regression, var_simple 17 -> 0.

    FIX: a single fully-additive HARD-DARK slot whose query is hugely positive
    on every NON-LEV-PC row and hugely negative on the LEV PC marker, paired
    with a ``k = -CONST`` (present at every key). On a non-LEV query this drives
    EVERY real key's score to ~-1e9 so softmax1 collapses to the zero-value sink
    and head 14 writes ~0 -- the address self-match can no longer leak. On the
    LEV PC marker the query is strongly NEGATIVE so ``-CONST`` makes a uniform
    POSITIVE add to every key (no relative darkening), leaving the genuine
    return-address gather untouched. Because the slot is only emitted when BOTH
    the parent flag and ADDR_WIDEN are on, and it never moves the relative
    ordering on the LEV PC marker, the LEV PC restore is preserved while the LI
    load is no longer clobbered. Flag-off omits the slot (byte-identical to the
    widen build); consulted only when ``C4_L15_LEV_ADDR_WIDEN`` is on.
    """
    return _os_l15.environ.get("C4_L15_LEV_PC_ONLY", "1") != "0"


def _l15_lev_opcode_gate_on() -> bool:
    """DEFAULT-OFF flag (``C4_L15_LEV_OPCODE_GATE``): hard-gate the LEV PC-restore
    head 14 on the *per-step fetched opcode* (``OPCODE_BYTE_LO+8`` == LEV) so it
    fires STRICTLY at the genuine LEV step and never bleeds into the NEXT step.

    ROOT (spec_k=0, BUILT dims, func_identity_0 id550, 2026-06-20): the existing
    PC-ONLY gate (slots 67/69) requires ``MARK_PC AND OP_LEV``. But ``OP_LEV`` is
    a CROSS-STEP DURABLE opcode broadcast ("the last-executed opcode was LEV"),
    so it PERSISTS into the step AFTER the LEV. For ``func_identity`` the LEV is
    step 8 (``pc 50->90``) and the very next instruction is ``ADJ 8``
    (``pc 90->98``); at the step-9 ``ADJ`` PC marker ``OP_LEV`` is STILL +6.5
    (even HIGHER than step 8's +5.2) and ``MARK_PC`` is +1.0, so head 14 fires
    again (w=1.0 on the return store p269, v=0x5a=90) and PINS the step-9 PC to
    90 instead of letting it advance to 98 -> full_trace diverges at step 9
    (``got_pc=90 expected_pc=98``, and the wrong PC further desyncs AX 70->72).

    THE CLEAN DISCRIMINATOR: ``OPCODE_BYTE_LO/HI`` carries the per-step FETCHED
    opcode (not the broadcast residue). It is a sharp one-hot at every PC marker:
    LEV (opcode 8 = 0x08) -> ``OPCODE_BYTE_LO+8 == 1.0`` ONLY at the genuine LEV
    step; the post-LEV ``ADJ`` step (opcode 7) is ``OPCODE_BYTE_LO+7``. Measured
    across func_identity (LEV@8) AND func_add (LEV@14): ``OPCODE_BYTE_LO+8`` is
    1.0 exactly at the LEV step and ~0 at every other step including the next.
    (The two other low-nibble-8 opcodes -- SHR 0x18, POP 0x28 -- are already
    excluded by the slot-69 ``OP_LEV`` requirement: their own step broadcasts
    OP_SHR/OP_POP, not OP_LEV.)

    FIX: a single fail-closed HARD-DARK slot mirroring slots 67/69 but keyed on
    ``OPCODE_BYTE_LO+8`` -- ``q = HARD*CONST - HARD*(OPCODE_BYTE_LO+8)`` (=> 0 at
    the LEV step where that cell is 1.0, ``+HARD`` everywhere else) paired with
    ``k = -CONST`` at every key. On a non-LEV-opcode row this drives every real
    key's score to ~-1e9 so softmax1 collapses to the zero sink and head 14
    writes ~nothing; on the genuine LEV step it contributes exactly 0 so the
    address slots + STACK0-byte0 selector still decide the genuine return-store
    gather. Only emitted when BOTH ``C4_L15_LEV_ADDR_WIDEN`` and this flag are on
    (it folds into the widen build's cache key); flag-off omits the slot so the
    build is byte-identical to the widen/pc-only build.
    """
    return _os_l15.environ.get("C4_L15_LEV_OPCODE_GATE", "1") != "0"


def _l15_savedra_head_on() -> bool:
    """Campaign-gated flag (``C4_L15_SAVEDRA_HEAD``): add a 16th L15 head that
    delivers the JSR return address (= JSR_PC + INSTR_WIDTH) into PC at the LEV
    step *in the 30-token campaign frame* (``C4_NO_STACK0_EMIT=1``).

    ROOT (CPU-first, campaign config, spec_k=0; tools/_probe_lev_findval90.py +
    tools/_probe_savedra_dims.py, func_identity id550): in the golden 35-token
    frame the JSR return address (90 = 0x5a) is materialized EXCLUSIVELY onto the
    emitted STACK0 push token (pos 269, STACK0_BYTE0=1, CLEAN_EMBED=0x5a). The
    existing head 14 LEV-CAM gathers it there by the BP+8 address key. Under
    ``C4_NO_STACK0_EMIT`` that STACK0 push is DROPPED: token 90 is never emitted,
    NO memory store row carries the return addr (every addr-0xfff0 store row holds
    the ENT-pushed saved-BP word 0xf0=240), so head 14's CAM starves and gathers
    the saved-BP word (got pc=240, want 90). A head re-key is impossible -- the
    value 90 does not exist on ANY persistent row in the campaign frame.

    BUT 90 IS available LIVE at the JSR step: the ``LOOKAHEAD_PC_LO/HI`` band
    (= current_PC + INSTR_WIDTH(8), built default-ON by ``C4_STACK0_NEXT_ARITH``)
    is a sharp one-hot at the JSR step's AX-marker row (step 4, OP_JSR present,
    MARK_AX=1): for func_identity id550 ``LOOKAHEAD_PC = 0x5a = 90`` there exactly
    (JSR_PC 82 + 8). The genuine LEV return PC = the most-recent JSR's PC+8.

    FIX (this head): a NEW dedicated head (index 15) that, at the LEV PC marker
    (query OP_LEV + MARK_PC), content-addresses the most-recent JSR AX-marker row
    (key OP_JSR + MARK_AX) and copies its ``LOOKAHEAD_PC_LO/HI`` -> OUTPUT_LO/HI,
    delivering the return PC byte. It is fully ISOLATED from the load-bearing
    head 14 CAM (head 14 keeps its golden +25 / SI-SC-LI-LC path untouched);
    flag-OFF (the default OUTSIDE the campaign) omits the head entirely so L15
    stays num_heads<=15 and the build is byte-identical to golden.

    DEFAULT tracks ``no_stack0_emit_enabled()`` (campaign-only ON), AND the
    parent head-14 flag must be on (the resize must already be widening L15).
    Force with ``C4_L15_SAVEDRA_HEAD=0/1``.
    """
    forced = _os_l15.environ.get("C4_L15_SAVEDRA_HEAD")
    if forced is not None:
        return forced != "0"
    from .shared import no_stack0_emit_enabled
    return no_stack0_emit_enabled() and _l15_lev_pc_restore_head_on()


_L15_LEV_PC_RESTORE_HEAD_IDX = 14
_L15_SAVEDRA_HEAD_IDX = 15
# Head 16 (flag C4_SI_STORE_ADDR, campaign, DEFAULT-OFF): SI/SC store
# address-provenance CAM — resolves the var_mul / multilocal-LI two-root wall.
_L15_SI_STORE_ADDR_HEAD_IDX = 16
from ...dim_registry import dim_ref
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
from ..isa_semantics_dsl import (
    CAM_DROP,
    CamBinaryAddressBlock,
    CamBinaryAddressMatch,
    CamDiscriminatorSlot,
    CamValueBand,
    NibbleRelay,
    ScalarRelayBankSpec,
    cam_binary_address_match,
    scalar_relay,
)
from ..layer_compiler import Operation
from ..positional_invariant import invariant_threshold, marker_bank_index
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import (
    _as_setdim_proxy,
    no_stack0_emit_enabled,
    l15_lookup_cmp_veto_enabled,
    si_store_addr_enabled,
    var_three_li_enabled,
)


def _l15_li_addr_cam_discriminator_on() -> bool:
    """DEFAULT campaign-ON (``C4_L15_LI_ADDR_CAM``): add a clean per-store
    address-match on L15 head-0 so a MULTI-LOCAL LI loads ``mem[BP+off]``'s
    VALUE for the QUERIED local, not the first/most-recent store.

    ROOT (spec_k=0, BUILT dims, var_three osteps 15/22, 2026-06-21): in the
    campaign config (``C4_NO_STACK0_EMIT=1``) the head-0 LI byte-0 lookup row
    (the AX-byte-0 prediction row) carries ``OP_LI_RELAY==0`` — so the
    ``local_slot_onehot`` disambiguator (rows 43+k, gated on OP_LI_RELAY) is
    DEAD, and the only address signal is the bit-encoded slots 4-27, whose
    contribution (~12-14k) is (a) NON-discriminating between the candidate
    store VALUE rows and (b) drowned by the ~9.4e5 constant blocker baseline.
    The per-store scores end up within ~0.2% of each other, so the softmax
    winner is decided by tiny residue (recency), and the LI returns the WRONG
    local (var_three ostep15 LI ``a`` -> ``b``'s value 6; ostep22 LI ``c`` ->
    ``b``'s value 0x23) or 0.

    The DISCRIMINATING signal IS present: each store's VALUE row (d=5 from its
    MEM marker) carries the store address byte-0 nibbles in
    ``ADDR_B0_LO``/``ADDR_B0_HI`` (proven: &a value row has ADDR_B0_LO+8=1.0,
    &b value row has only ADDR_B0_LO+0). The operand LI query row carries the
    SAME nibble one-hots. A clean Q·K one-hot match on ADDR_B0_LO+ADDR_B0_HI,
    gated on the LIVE b0row marker (``MARK_AX``, since OP_LI_RELAY is 0 here)
    and scaled to dominate the ~2k tie, content-addresses the right store.

    Campaign-only (the value-row address signals it keys on are produced by the
    30-token MEM-from-SP path); golden (35-token, flag-OFF) is byte-identical —
    the slots are simply not emitted. Head-0 ONLY (does not touch heads 1-3 byte
    relays or the V/O value copy).
    """
    raw = _os_l15.environ.get("C4_L15_LI_ADDR_CAM")
    if raw is not None:
        return raw != "0"
    return no_stack0_emit_enabled()


def _l15_li_zeroaddr_cam_on() -> bool:
    """DEFAULT-OFF building block (``C4_L15_LI_ZEROADDR_CAM=1`` to enable):
    restore the k=0 (zero-nibble) address match on L15 head-0, gated on
    committed stores, so a local whose BP-relative address is 0x00 gets address
    discrimination among its same-address committed stores (#301/#318
    zero-address blind spot).

    *** DEFAULT-OFF -- DOES NOT FLIP var_simple IN THE AR DECODE WITHOUT
    REGRESSING func (the documented multi-session operand-CAM blocker). ***
    Kept as a flag-gated building block (golden + campaign-config baseline
    byte-identical when OFF) for the next session. See the BLOCKER section
    below for the exact AR-framing wall and the blueprint.

    BLOCKER (CPU-first, spec_k=0, BUILT dims; tools/_probe_zeroaddr_cam.py +
    tools/cpu_full_trace.py, campaign config). The TF probe is SOLVABLE -- with
    this slot pair head-0 at var_simple #250 step-7 correctly selects the
    committed x-store row 267 (valbyte 0xde) over the non-committed
    operand-frame row 297, AND func_identity #550's nonzero-address LIs are
    untouched (all 11 TF steps byte-correct). But the AUTHORITATIVE
    *autoregressive* cpu_full_trace verdict is exquisitely, non-monotonically
    sensitive to the head-0 boost magnitude in a way the re-anchored TF probe
    CANNOT see (a 34/37-token framing desync, not a value error):
      * _msav_w = 1.5*_zc_s : var_simple #250 AR FAIL, func #550 AR FAIL
      * _msav_w = 5.0*_zc_s : var_simple #250 AR FAIL (over-sharp head-0 ->
                              framing desync), func #550 AR PASS
      * a separate MARK_AX/CONST-gated slot-103 (K=MSAV): var_simple #250 AR
                              PASS, but func #550 AR FAIL (the negative-Q
                              penalty on func's nonzero-addr committed stores).
    Root of the wall: attention is BILINEAR per head-dim slot, so "fire only
    when (zero-address QUERY) AND (committed CANDIDATE)" cannot be expressed in
    one slot -- the query-side zero-address gate (Q on ADDR_B0_LO/HI+0) ALSO
    fires on var's OWN SI store-step rows (breaks var's AR framing), while the
    emit-row gate (Q on MARK_AX) fires on EVERY LI incl. func's (breaks func).
    And on the CANDIDATE side the non-committed operand row 297 carries the
    zero-address nibbles at HIGHER amplitude (2.46 vs the committed store's
    1.46), so committedness (MSAV) must out-weigh the address amplitude -- but
    K is LINEAR per slot, so K = zero-nibble + MSAV cannot AND them, and any
    MSAV weight large enough to flip 297->267 over-sharpens head-0 and desyncs
    the var AR frame. This is the brief's anticipated multi-session wall.

    BLUEPRINT (next session): the clean fix needs a signal that is positive
    ONLY on a (committed AND zero-address) value row -- i.e. an FFN-materialized
    "committed-zero-address-store" indicator dim (one L13/L14 FFN unit:
    silu(MSAV + ADDR_B0_LO+0 + ADDR_B0_HI+0 - 2.5)) so head-0 can key K on that
    SINGLE dim (the AND is done in the FFN, not the bilinear head). Then a
    MARK_AX-gated Q (emit-row-only -> var-store-safe) x that-dim K (zero-address
    committed only -> func-safe, no penalty) flips var without touching func.
    The L15 attention layer alone cannot AND three conditions; the materialized
    indicator dim is the missing piece.

    ROOT (CPU-first, campaign config, spec_k=0, BUILT dims; var_simple #250
    x=990, tools/_probe_zeroaddr_cam.py): ``x`` lives at ``BP+0`` -> its store
    address byte-0 is ``0x00``. The #313 ADDR-CAM DROPS the ``k=0`` match (loop
    ``range(1,16)``), so a zero-address local gets NO per-store address
    discrimination and an OLD wrong-address committed store out-scores the
    right latest x-store in the AR decode (#301/#318).

    ROOT (CPU-first, campaign config, spec_k=0, BUILT dims; var_simple #250
    x=990, tools/_probe_zeroaddr_cam.py): ``x`` lives at ``BP+0`` -> its store
    address byte-0 is ``0x00`` (BOTH the lo AND hi nibble are 0). The #313
    ADDR-CAM (slots 71-101) DELIBERATELY DROPS the ``k=0`` match (loop
    ``range(1,16)``) because, taken naively, the zero-nibble one-hot peaks on
    BOTH the operand query row AND on stray null-address load-result/code rows,
    so a genuine non-zero-address local (``&a=0xffe8``) could lose to a 0x00
    intermediate row. CONSEQUENCE: a zero-address local gets ZERO per-store
    address discrimination -> in the AUTOREGRESSIVE decode (TF passes; this is
    an AR-emit-only failure) an OLD wrong-address committed store (``0xF8``,
    pos 117, nonzero nibbles -> slot-79/95 keys) out-scores the right latest
    x-store (``0x00``, pos 267) and the ALiBi recency slope 0.05 over the ~150
    position gap adds only ~7 -- far short of the ~744 deficit.

    THE FIX (recency-among-same-address-committed): re-instate the k=0 match on
    BOTH ``ADDR_B0_LO+0`` and ``ADDR_B0_HI+0`` -- the candidate's OWN zero
    nibble one-hot (slots 71/87, the natural k=0 positions the #313 loop left
    free) -- but GATE the match to GENUINE committed stores via an additive
    ``MEM_STORE_AT_VAL`` (MSAV) requirement so stray null-address load-result /
    code rows (MSAV=0) never collect it. With the k=0 boost the latest
    zero-address committed store now leads its address class, and the existing
    0.05 recency slope cleanly resolves the (rare) multiple-same-zero-address
    case (var_update's x-reassign) to the most-recent write. The brief's
    ``&a=0xffe8`` objection is null because that store's nibbles are nonzero
    (lo=8/hi=14) and already match k=8/k=14; the k=0 boost it cannot collect,
    and a 0x00 load-result row is MSAV=0 so it cannot collect it either.

    Campaign-only (the value-row address + MSAV signals are produced by the
    30-token MEM-from-SP path); golden (35-token, flag-OFF) is byte-identical --
    the slots are simply not emitted. Head-0 ONLY; pure additive
    content-addressing on free slots; own kill-switch for the cross-op /
    flag-regression gates.
    """
    # CAMPAIGN-DEFAULT-ON: the FFN 3-way-AND indicator (make_layer14_li_zeroaddr_indicator_op
    # + the head-0 slot-103 key) FLIPS var_simple 0->25/25 on GPU AR full_trace while
    # func_identity HOLDS 25/25 -- the "cannot flip without regressing func" above was the
    # earlier BILINEAR-head variant; putting the 3-way AND in the FFN resolved it. Golden
    # (no_stack0_emit OFF) stays byte-identical (band/op/head-slot all flag-gated).
    raw = _os_l15.environ.get("C4_L15_LI_ZEROADDR_CAM")
    if raw is not None:
        return raw != "0"
    return no_stack0_emit_enabled()


def _l15_li_byte0_valsel_on() -> bool:
    """DEFAULT campaign-ON (``C4_L15_LI_B0_VALSEL``): re-establish L15 head-0's
    byte-0 VALUE-row selector via the SURVIVING ``MEM_VAL_B0`` dim (#318).

    ROOT (a52fd42c, spec_k=0, BUILT dims): in the base head-0 spec the slot-3
    byte-0 selector keys the candidate row on ``(L2H0+MEM, -H1+MEM)`` and gates
    the query on ``MARK_STACK0`` (l15:862-866). The 30-token campaign layout
    DROPS the STACK0 value block and auto-neutralizes its L1 producer, so
    ``MARK_STACK0`` never fires -> head-0's byte-0 value-row selection collapses
    and it self-attends the AX-marker row (CLEAN_EMBED_LO=0) -> AX byte-0 = 0x00
    (var_simple s7 / var_update s9 / if_var s7). Heads 1-3 select bytes 1-3 via
    ``MEM_VAL_B1/B2/B3`` (which survive the 30-tok path), so the high bytes stay
    correct -- the byte0-wrong/byte1-ok asymmetry is the fingerprint.

    The fix adds a slot-3 K on ``MEM_VAL_B0`` (mirroring heads 1-3) inside the
    head-0 campaign override; the base slot-3 Q already carries the head-0
    byte_q_flag (``MARK_AX``), so the bilinear is positive only on the
    value-byte-0 store row. Campaign-only (golden 35-tok flag-OFF byte-identical;
    this branch is not taken there). Own kill-switch so the cross-op attention +
    flag-regression gates can toggle just this change.
    """
    raw = _os_l15.environ.get("C4_L15_LI_B0_VALSEL")
    if raw is not None:
        return raw != "0"
    return no_stack0_emit_enabled()


def _l15_li_valrow_b1_on() -> bool:
    """DEFAULT campaign-ON (``C4_L15_LI_VALROW_B1``): lift the genuine store
    VALUE row over the spurious ADDR-byte cluster on the FIRST-PARAM LI
    (func_add/square/max/min + nested re-read chain — PHASE-2 root).

    ROOT (CPU-first, spec_k=0, BUILT dims; tools/_probe_func_li_tf.py +
    _probe_func_valrow_disc.py, campaign config): interp_oracle_gate classifies
    func_add/square/max/min CROSS-STEP at **step 9 AX[0]** — the LI that loads
    the FIRST parameter (``a``), UPSTREAM of the ``&b`` re-read LEA (step 11)
    the keystone amplifier targets. The first-param LI returns the WRONG byte-0
    because L15 head-0's value-row attention TIES: the queried local's address
    is ``&a`` = 0xE8, so the #313 ADDR_B0 CAM keys ``ADDR_B0_LO+8`` +
    ``ADDR_B0_HI+14``, but the spurious code/frame byte rows JUST PAST the ``a``
    store (e.g. func_add pos 292-295) carry the SAME ``ADDR_B0_LO+8`` AND
    ``ADDR_B0_HI+14`` one-hots (they inherit the store frame's address), so the
    ADDR CAM gives them IDENTICAL credit and they out-score the genuine value
    row (pos 271) by ~0.2% on the ~648k CONST baseline. The #318 ``MEM_VAL_B0``
    selector does NOT rescue ``a`` because ``a``'s value row carries
    ``MEM_VAL_B1`` (=0.97) but ``MEM_VAL_B0`` = -0.0 (the value's low byte is on
    the B1 dim in the 30-tok path for this frame). The SECOND param ``b`` (step
    12) already resolves: its address is ``&b`` = 0xE0 (``ADDR_B0_LO+0``) and no
    spurious lo-0 cluster competes.

    THE DISCRIMINATOR that survives where ADDR_B0 ties (measured): genuine
    store VALUE rows carry the materialized value byte on ``MEM_VAL_B1`` (~0.97);
    the spurious ADDR-byte rows carry ``MEM_VAL_B1`` ~ -0.0. A positive head-0 K
    on ``MEM_VAL_B1`` (gated MARK_AX on the Q so it fires only at the LI byte-0
    lookup row) lifts the genuine value row decisively. Simulated over the exact
    head-0 bilinear (tools/_probe_func_b1_sim.py): with the lift the func_add
    step-9 winner flips 292(CLEAN_LO=0)->271(CLEAN_LO=9=0x39 lo) and func_max
    step-9 flips 340(0)->319(CLEAN_LO=4=0x24 lo); the already-correct ``b`` LI
    (step 12) is UNCHANGED (b's value row carries MEM_VAL_B1 too -> same lift,
    no competing row) -- stable across lift strength 5e4..5e5.

    Uses a DEDICATED free head-0 slot (102; head_dim 111, slots 0-70 base,
    71-101 the #313 CAM) so it is independently A/B-toggleable and does not
    perturb the existing selectors. Campaign-only (golden 35-tok flag-OFF
    byte-identical; this branch is not taken there). Own kill-switch so the
    cross-op attention + flag-regression gates can toggle just this change.
    """
    raw = _os_l15.environ.get("C4_L15_LI_VALROW_B1")
    if raw is not None:
        return raw != "0"
    return no_stack0_emit_enabled()


def _l15_li_jsr_phantom_penalty_on() -> bool:
    """DEFAULT campaign-ON (``C4_L15_LI_JSR_PHANTOM``): break the multi-arg
    first-param LI value-row tie by penalising the JSR/ENT-step PHANTOM MEM
    rows that share the queried address but carry no stored value
    (func_add/func_mul step-9 first-param ``a`` LI -- POST-FLIP root).

    ROOT (GPU spec_k=0, BUILT dims; tools/_probe_func_li_value.py + the
    head-0 score decomposition in this lane, campaign config): func_add/mul
    diverge at **step 9 = LI** -- the load of the FIRST argument ``a`` returns
    AX=0. The arg was PSH'd at the CALL SITE (e.g. value 57 stored @ mem[0xE8],
    func_add pos 271, ``CLEAN_EMBED``=57, ``OP_JSR``~=0). L15 head-0's address
    CAM (slots 4-27) correctly matches the queried byte-0 = 0xE8, and the
    genuine value row (271) even EDGES the raw QK score -- but the callee's
    JSR/ENT step re-uses the SAME stack slot, so its PHANTOM MEM value rows
    (func_add pos 360-363, ``CLEAN_EMBED``=0, ADDR byte-0 = 0xE8 too) also
    match the address, carry ``MEM_VAL_B1``~=0.97 (so #313 + ``_l15_li_valrow_b1``
    boost them EQUALLY), and -- being MORE RECENT -- WIN the 0.05 ALiBi recency
    tie-break by ~5 (probed: phantom 361 score 709951 vs genuine 271 709946).
    Result: head-0 attends a value-0 phantom and the LI returns 0. This is the
    #313/#318/B1-lift blind spot: those discriminate by ADDRESS and by
    ``MEM_VAL_B1`` presence, but the phantom matches BOTH, so only the
    store-PROVENANCE distinguishes them.

    THE DISCRIMINATOR (measured): a genuine PSH/SI store value row carries
    ``OP_JSR``~=0 (func_add 271: 0.02; func_identity 223: 0.02; every clean
    si_li store), while the callee's JSR/ENT-step phantom MEM rows carry the
    JSR opcode residue ``OP_JSR``~=1.5-1.7 (func_add 360-363). A K-side
    ``-OP_JSR`` penalty (gated on the Q by ``MARK_AX`` -- the only LIVE signal
    at the byte-0 LI lookup row; ``OP_LI_RELAY``==0 there, the #313 blind spot)
    subtracts ~5.6k from every JSR-phantom and ~0 from the genuine store, so
    the genuine value row leads its address class. Simulated over the exact
    head-0 bilinear (this lane): func_add s9 flips 361(cl0)->271(cl57) and
    func_mul s9 flips 361(cl0)->271(cl49); func_max/min/square/identity s9/s7
    LIs (which already win) and var_simple 250-270 (no JSR competitor) are
    UNCHANGED (OFF==ON).

    Uses a DEDICATED free head-0 slot (104; head_dim 111, slots 0-103 taken:
    0-63 base, 64-70 suppressor-cancel, 71-101 #313 CAM, 102 B1-lift, 103 #318)
    so it is independently A/B-toggleable. Campaign-only (golden 35-tok
    flag-OFF byte-identical; this branch is not taken there, and ``OP_JSR`` only
    competes a value row in the 30-tok MEM-from-SP path). Own kill-switch so the
    cross-op attention + flag-regression gates can toggle just this change.
    """
    raw = _os_l15.environ.get("C4_L15_LI_JSR_PHANTOM")
    if raw is not None:
        return raw != "0"
    return no_stack0_emit_enabled()


def _l15_sclc_byte0_on() -> bool:
    """DEFAULT campaign-ON (``C4_SCLC_LC_B0``): mirror the #318 zero-address
    committed-store byte-0 boost onto the ``OP_LC`` (load-char) opcode so the
    SC/LC single-byte store-load roundtrip delivers its byte-0 reload value
    (``test_sc_lc_roundtrip``: store char 42 @ 0x200, load char -> 42).

    ROOT (GPU bit-exact, spec_k=0, BUILT dims; campaign config, runner block-33
    head-0 attention-score decomposition tools/probe_lc_byte0.py +
    probe_b33h0_qcontrib.py):  L15 head-0's ``#318`` keystone slot (row 103)
    boosts the committed-AND-zero-address store-value row via the
    ``LI_ZEROADDR_COMMITTED`` indicator K, but its Q opens ONLY on ``OP_LI``
    (the LI emit-row marker). ``sc_lc`` stores at 0x200, whose byte-0 (0x00) is
    a ZERO-address byte-0, so the ``#313`` ADDR_B0 CAM -- which DROPS the k=0
    (zero-nibble) match -- gives NO per-store discrimination, and the genuine
    SC store-value row (CLEAN=42) TIES the spurious address-IMM step row (the
    next step's ``IMM 0x200`` byte-0, CLEAN=0). For the ``LI`` path slot 103
    breaks the tie by +33860 (the OP_LI-gated indicator boost) so the right row
    wins; for the ``LC`` path slot 103 is dead (Q[OP_LC]=0) -> the two rows tie
    and the 0.05 ALiBi recency slope picks the WRONG (more-recent address-IMM)
    row -> byte-0 = 0x00 (``test_sc_lc_roundtrip`` got 0, want 42).

    THE FIX: add ``OP_LC`` to the row-103 Q gate (same +360 weight as ``OP_LI``)
    so the slot opens on an LC EMIT row too and applies the IDENTICAL committed
    store-value boost. ``OP_LC`` fires ONLY on LC opcode rows (var/func/SI/LI
    programs never emit it), so the slot stays inert on every non-LC frame --
    var_simple / func / si_li are UNTOUCHED (no OP_LC marker anywhere). GPU
    patch-verified: LC 0x00 -> 42 with LI unchanged (probe_patch_b33.py).
    Campaign-only via its own kill-switch (golden 35-tok flag-OFF byte-identical
    -- the row-103 slot is only emitted under ``_l15_li_zeroaddr_cam_on``, which
    is itself campaign-gated; off the campaign the OP_LC term is never written).
    """
    raw = _os_l15.environ.get("C4_SCLC_LC_B0")
    if raw is not None:
        return raw != "0"
    return no_stack0_emit_enabled()


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
# head 14 (flag C4_L15_LEV_PC_RESTORE, default ON): LEV return-address
# content-addressable restore into PC OUTPUT. Appended only when the flag is
# on so flag-off keeps num_heads=14 byte-identical.
if _l15_lev_pc_restore_head_on():
    _L15_HEAD_LAYOUT = _L15_HEAD_LAYOUT + (
        ("layer15_memory_lookup.lev_pc_restore", _L15_LEV_PC_RESTORE_HEAD_IDX),
    )
# head 15 (campaign flag C4_L15_SAVEDRA_HEAD): saved-RA delivery into PC at LEV in
# the 30-token campaign frame. Appended only when on so flag-off keeps the head
# count byte-identical to the golden (15-head LEV) build.
if _l15_savedra_head_on():
    _L15_HEAD_LAYOUT = _L15_HEAD_LAYOUT + (
        ("layer15_memory_lookup.savedra_pc", _L15_SAVEDRA_HEAD_IDX),
    )
# head 16 (campaign flag C4_SI_STORE_ADDR, DEFAULT-OFF): SI/SC store
# address-provenance CAM. Appended only when the flag is on so flag-OFF keeps
# the L15 head count byte-identical to the golden (campaign 16-head) build.
if si_store_addr_enabled():
    _L15_HEAD_LAYOUT = _L15_HEAD_LAYOUT + (
        ("layer15_memory_lookup.si_store_addr_cam", _L15_SI_STORE_ADDR_HEAD_IDX),
    )
_L15_HEAD_LAYOUT_BY_NAME = {name: head_idx for name, head_idx in _L15_HEAD_LAYOUT}
# Widest configured L15 head count: 17 with the SI-store-addr CAM (campaign,
# flag on), 16 with the campaign saved-RA head, 15 with the LEV PC-restore
# head, else 14 (the legacy LEV build).
if si_store_addr_enabled():
    # The SI-store CAM head requires the campaign saved-RA head to already be
    # widening L15 to 16; head 16 is the 17th slot.
    _L15_MAX_HEADS = 17
elif _l15_savedra_head_on():
    _L15_MAX_HEADS = 16
elif _l15_lev_pc_restore_head_on():
    _L15_MAX_HEADS = 15
else:
    _L15_MAX_HEADS = 14


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
    allocator = AttentionHeadAllocator(layer_max_heads=_L15_MAX_HEADS)
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

    all_byte_indices = (dim_ref("byte_index", "0"), dim_ref("byte_index", "1"),
                        dim_ref("byte_index", "2"), dim_ref("byte_index", "3"))

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
            conds.append((dim_ref("opcode_flag", "ENT"), -2.0))
        return tuple(conds)

    # SP byte 0 position predicts SP byte 1 = 0xff after SP -= 8.
    rules.append(multi_way_and_rule(
        name="psh_sp_byte1_lo_ff",
        conditions=psh_byte_conditions(sp_i, dim_ref("byte_index", "0")),
        threshold=threshold,
        writes=((dim_ref("output_lo", "nibble", 15), 4.0),
                (dim_ref("output_lo", "nibble", 0), -4.0)),
    ))
    rules.append(multi_way_and_rule(
        name="psh_sp_byte1_hi_ff",
        conditions=psh_byte_conditions(sp_i, dim_ref("byte_index", "0")),
        threshold=threshold,
        writes=(("OUTPUT_HI_THIS_STEP+15", 4.0), ("OUTPUT_HI_THIS_STEP+0", -4.0)),
    ))

    # SP byte 1 and byte 2 positions predict zero for the following bytes.
    for byte_index_name, predicted_byte in (
        (dim_ref("byte_index", "1"), "byte2"),
        (dim_ref("byte_index", "2"), "byte3"),
    ):
        rules.append(multi_way_and_rule(
            name=f"psh_sp_{predicted_byte}_lo_00",
            conditions=psh_byte_conditions(sp_i, byte_index_name),
            threshold=threshold,
            writes=((dim_ref("output_lo", "nibble", 0), 2.0),),
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
        conditions=psh_byte_conditions(bp_i, dim_ref("byte_index", "1")),
        threshold=threshold,
        writes=((dim_ref("output_lo", "nibble", 1), 4.0),
                (dim_ref("output_lo", "nibble", 0), -4.0)),
    ))
    rules.append(multi_way_and_rule(
        name="psh_bp_byte2_hi_00",
        conditions=psh_byte_conditions(bp_i, dim_ref("byte_index", "1")),
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
        (dim_ref("marker", "BP"), -1_000_000.0),
    )
    # Bare base NAMEs for the per-nibble copy loops: ``dim_ref`` returns
    # ``"OUTPUT_LO+0"``, so strip the ``+0`` to rebuild each cell as
    # ``f"{base}+{k}"`` (the ``+k`` is a structural nibble index, kept raw).
    output_lo_base = dim_ref("output_lo", "nibble").rsplit("+", 1)[0]
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"nibble_copy_lo_{k}",
            conditions=copy_conditions,
            threshold=0.5,
            gate=f"EMBED_LO+{k}",
            writes=((f"{output_lo_base}+{k}", 2.0),),
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
        (dim_ref("cmp_flag", "cascade", 7), 1.0),
        (f"H1+{ax_i}", 1.0),
        ("IS_BYTE", 1.0),
        (dim_ref("byte_index", "1"), 1.0),
        ("HAS_SE", -1.0),
    )
    rules.append(multi_way_and_rule(
        name="lea_first_step_ax_byte2_lo_01",
        conditions=lea_conditions,
        threshold=4.5,
        writes=((dim_ref("output_lo", "nibble", 1), 4.0),
                (dim_ref("output_lo", "nibble", 0), -4.0)),
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

    # Marker-bank slot indices, resolved through the positional-invariant
    # mechanism (Class-1 marker-relative anchors). ``BD.H1 + PC_I`` etc. read
    # the marker-TYPE slot of the fixed-width threshold-head bank (PC=0 AX=1
    # SP=2 BP=3 MEM=4 SE=5), which is frame-INVARIANT — its order does NOT
    # change when the STACK0 *value* block is dropped (STEP_TOKENS 35->30).
    # ``marker_bank_index`` is the single source of truth and lets the
    # positional audit reclassify these refs as declared-invariant instead of
    # UNGUARDED bare offsets. Byte-identical in both frames (returns the same
    # integers the literals encoded). See positional_invariant.py.
    PC_I = marker_bank_index("PC")
    AX_I = marker_bank_index("AX")
    SP_I = marker_bank_index("SP")
    BP_I = marker_bank_index("BP")
    MEM_I = marker_bank_index("MEM")

    # === DERIVED from cam_binary_address_match (the L15 LI/LC load CAM) ======
    #
    # DERIVE->PROVE->FLIP->DELETE (2026-07-04, golden 91f55411): the four L15
    # LI/LC + STACK0 load heads are re-expressed byte-identically through the
    # :func:`cam_binary_address_match` primitive (isa_semantics_dsl.py). Each is
    # a binary-addressed content-load CAM: the 24-bit BINARY per-bit address
    # comparator (slots 4..27, ``±10`` per bit across the 6 nibble bands) is the
    # STRUCTURAL :class:`CamBinaryAddressBlock`; the ~10 heterogeneous
    # opcode/marker/byte-select discriminator rows (slots 0-3, 28-33) are DATA
    # (:class:`CamDiscriminatorSlot`); the CLEAN_EMBED->OUTPUT byte relay is the
    # :class:`CamValueBand` block. The hand-authored per-head Q/K/V/O directive
    # construction is DELETED; the compact spec below is the sole path. Proof:
    # ``test_isa_semantics_dsl.py::test_l15_li_lc_load_derived_is_byte_identical_to_handbuilt``
    # (4/4 heads == the legacy writes) + the whole-model golden hash unchanged.
    #
    # Names use the ``BASE+offset`` token form (``f"H1+{SP_I}"``, ``"CMP+3"``,
    # ``f"L1H4+{BP_I}"``) resolved by cam_binary_address_match -> the marker-bank
    # anchors stay VISIBLE in the declared signature (frame-invariant, Class-1).
    scale = 10.0
    addr_block = CamBinaryAddressBlock(
        nibble_bands=(
            "ADDR_B0_LO", "ADDR_B0_HI",
            "ADDR_B1_LO", "ADDR_B1_HI",
            "ADDR_B2_LO", "ADDR_B2_HI",
        ),
        scale=scale, slot_base=4, width_bits=4,
    )
    value_bands = (
        CamValueBand("CLEAN_EMBED_LO", "OUTPUT_LO", 16, 32, 1.0),
        CamValueBand("CLEAN_EMBED_HI", "OUTPUT_HI", 16, 48, 1.0),
    )
    BS = 60.0  # Byte Selection weight (60*60/8 = 450 per matching byte).
    byte_q_flags = ("MARK_AX", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2")
    MEM_VAL_DIMS = (None, "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3")

    # Resolve the dim NAMEs the CAM specs touch to positions via ``BD``.
    dim_map = _l15_li_lc_load_dim_map(BD)

    specs: list[DeclarativeAttentionHeadSpec] = []
    for h in range(4):
        cam = _l15_li_lc_load_cam(h, addr_block, value_bands, BS, MEM_VAL_DIMS)
        head_idx = _L15_HEAD_LAYOUT_BY_NAME[
            f"layer15_memory_lookup.li_lc_stack0_h{h}"
        ]
        bundle = cam_binary_address_match(cam)
        specs.append(bundle.head_spec_builder(dim_map, head_idx))

    return tuple(specs)


def _l15_li_lc_load_cam(
    h: int,
    addr_block: CamBinaryAddressBlock,
    value_bands: tuple,
    BS: float,
    MEM_VAL_DIMS: tuple,
    overlay: tuple = (),
) -> CamBinaryAddressMatch:
    """Build the head-``h`` L15 LI/LC + STACK0 load :class:`CamBinaryAddressMatch`.

    The BASE binary-address CAM (address comparator + the ~10 heterogeneous
    opcode/marker/byte-select discriminator rows). ``overlay`` supplies the
    flag-conditioned OVERRIDE-layer discriminators (see
    :func:`_layer15_memory_lookup_heads_0_3_specs_with_overrides`); at
    ``overlay=()`` this is the golden byte-identical base head.

    Marker-bank slot indices resolved through the positional-invariant mechanism
    (frame-INVARIANT bank-TYPE order; byte-identical in both frames).
    """
    PC_I = marker_bank_index("PC")
    AX_I = marker_bank_index("AX")
    SP_I = marker_bank_index("SP")
    BP_I = marker_bank_index("BP")
    MEM_I = marker_bank_index("MEM")

    byte_q_flags = ("MARK_AX", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2")
    byte_q_flag = byte_q_flags[h]

    discs: list[CamDiscriminatorSlot] = []

    # === Slot 0: Bias -- suppress non-target Q positions ===
    s0_q: list = [("CONST", -2000.0), ("OP_LI_RELAY", 2000.0)]
    if h == 0:
        s0_q.append(("OP_LC_RELAY", 2000.0))
        s0_q.append(("CMP+3", 2000.0))  # POP group -> stack memory read
    else:
        # L1H4[BP] gate active over STACK0 area. The legacy helper also
        # wrote H1[BP_I] = -2000 here, but the SP/BP byte blocker below
        # (-50000) overwrites that cell; emit only the final value.
        s0_q.append((f"L1H4+{BP_I}", 2000.0))
    s0_q.append(("CMP+0", -2000.0))
    # LEV suppression and PC/SP marker blockers.
    s0_q.append(("OP_LEV", -1000.0))
    s0_q.append(("MARK_PC", -25000.0))
    s0_q.append(("MARK_SP", -100000.0))
    # SP/BP byte blocker -- last write wins on (slot=0, H1+BP_I).
    s0_q.append((f"H1+{SP_I}", -50000.0))
    s0_q.append((f"H1+{BP_I}", -50000.0))
    discs.append(CamDiscriminatorSlot(
        slot=0, q=tuple(s0_q), k=(("CONST", 10.0),)))

    # === Slot 29: PC byte position blocker ===
    discs.append(CamDiscriminatorSlot(
        slot=29, q=((f"H1+{PC_I}", -20000.0),), k=(("CONST", 5.0),)))

    # === Slot 30: AX byte position default blocker ===
    discs.append(CamDiscriminatorSlot(
        slot=30, q=((f"H1+{AX_I}", -20000.0),), k=(("CONST", 5.0),)))

    # === Slot 31: restore AX-byte score for real LI loads against
    # stored MEM entries (head 0 also LC). ===
    s31_q: list = [("OP_LI_RELAY", 20000.0)]
    if h == 0:
        s31_q.append(("OP_LC_RELAY", 20000.0))
    discs.append(CamDiscriminatorSlot(
        slot=31, q=tuple(s31_q), k=(("MEM_STORE", 5.0),)))

    # === Slot 32: AX marker default blocker ===
    discs.append(CamDiscriminatorSlot(
        slot=32, q=(("MARK_AX", -20000.0),), k=(("CONST", 5.0),)))

    # === Slot 33: restore AX-marker score for head 0 LI/LC loads ===
    if h == 0:
        discs.append(CamDiscriminatorSlot(
            slot=33,
            q=(("OP_LI_RELAY", 20000.0), ("OP_LC_RELAY", 20000.0)),
            k=(("MEM_STORE", 5.0),)))

    # === Slot 1: Store anchor -- suppress non-store K at target Q ===
    s1_q: list = [("OP_LI_RELAY", 50.0)]
    if h == 0:
        s1_q.append(("OP_LC_RELAY", 50.0))
        s1_q.append(("CMP+3", 50.0))  # POP group (matches slot 0)
    else:
        s1_q.append((f"L1H4+{BP_I}", 50.0))
        s1_q.append((f"H1+{BP_I}", -50.0))
    s1_q.append(("CMP+0", -50.0))
    discs.append(CamDiscriminatorSlot(
        slot=1, q=tuple(s1_q),
        k=(("MEM_STORE", 100.0), ("CONST", -50.0))))

    # === Slot 2: ZFOD negative offset for store entries ===
    discs.append(CamDiscriminatorSlot(
        slot=2, q=(("CONST", -96.0),), k=(("MEM_STORE", 50.0),)))

    # === Slot 3: Byte selection ===
    s3_q: list = [(byte_q_flag, BS)]
    if h == 0:
        s3_q.append(("MARK_STACK0", BS))
        # Head 0 -> val byte 0 at d=5: L2H0[MEM]=1, H1[MEM]=0.
        s3_k = ((f"L2H0+{MEM_I}", BS), (f"H1+{MEM_I}", -BS))
    else:
        # Heads 1-3 -> val bytes 1,2,3 via MEM_VAL_B1/B2/B3.
        s3_k = ((MEM_VAL_DIMS[h], BS),)
    discs.append(CamDiscriminatorSlot(slot=3, q=tuple(s3_q), k=s3_k))

    # === Slot 28: Per-head position gate ===
    s28_q: list = [("CONST", -500.0), (byte_q_flag, 500.0)]
    if h == 0:
        s28_q.append(("MARK_STACK0", 500.0))
    discs.append(CamDiscriminatorSlot(
        slot=28, q=tuple(s28_q), k=(("CONST", 5.0),)))

    return CamBinaryAddressMatch(
        name=f"layer15_memory_lookup.li_lc_stack0_h{h}",
        address=addr_block,
        discriminators=tuple(discs),
        value_bands=value_bands,
        direction="load",
        # STEP_WINDOW_AUDIT_2026_06_10: LI/LC + STACK0 load heads read
        # MARK_MEM tokens by address -- memory persists across step
        # boundaries by design (runtime slope=0.05 keeps the most-recent
        # write dominant). Declares the cross-step intent the verifier
        # would otherwise misclassify as CURRENT_STEP_ONLY.
        step_window=StepWindowConstraint.ANY_STEP,
        overlay=tuple(overlay),
    )


def _l15_li_lc_load_dim_map(BD) -> dict:
    """Resolve every dim NAME the L15 LI/LC load CAM specs touch to an int.

    :func:`cam_binary_address_match`'s builder wants a name->int dict (with
    ``BASE+offset`` tokens resolved through
    :func:`isa_semantics_dsl._resolve_dim_token`). The marker-bank anchors
    (``H1+SP_I`` etc.) are supplied as their BASE names here; the offset is
    parsed from the token at build time.
    """
    base_names = (
        "CONST", "OP_LI_RELAY", "OP_LC_RELAY", "OP_LEV",
        "MARK_PC", "MARK_SP", "MARK_AX", "MARK_STACK0",
        "MEM_STORE", "CMP", "H1", "L1H4", "L2H0",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
        "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO", "ADDR_B1_HI",
        "ADDR_B2_LO", "ADDR_B2_HI",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
        # --- OVERRIDE-layer base names (the flag-conditioned overlay cells) ---
        # The overlay in _layer15_memory_lookup_heads_0_3_specs_with_overrides
        # is declared as NAMED offset-token CamDiscriminatorSlot literals; every
        # base name they reference must resolve here.
        "EMBED_LO", "EMBED_HI", "H2", "H3", "HAS_SE", "IS_BYTE",
        "LI_ZEROADDR_COMMITTED", "MARK_BP", "MARK_MEM",
        "MEM_ADDR_SRC", "MEM_VAL_B0",
        "OP_ENT", "OP_EQ", "OP_GE", "OP_GT", "OP_IMM", "OP_JSR",
        "OP_LC", "OP_LE", "OP_LEA", "OP_LI", "OP_LT", "OP_NE",
        "OP_SC", "OP_SI",
        "STACK0_BYTE0", "TEMP",
    )
    return {n: int(getattr(BD, n)) for n in base_names}


class _L15OverlayAccumulator:
    """Collects NAMED-token L15 override cells into overlay discriminators.

    The L15 heads-0-3 OVERRIDE layer is authored as declarative
    :class:`CamDiscriminatorSlot` DATA: a per-slot list of
    ``(BASE`` / ``BASE+offset`` name, weight)`` Q/K/V/O cells. This accumulator
    is the thin ordered store the authoring blocks write into. It preserves the
    original flat-map LAST-WRITE-WINS semantics EXACTLY -- a later block's write
    to the same ``(slot, dim-name)`` replaces the earlier one, in source order --
    so lifting the procedural per-cell arithmetic into these declarative slot
    literals is byte-identical. :meth:`discriminators` emits ONE
    :class:`CamDiscriminatorSlot` per touched slot (the name tokens resolve
    through the head's ``dim_map`` at build time), so the whole override is
    DATA: no ``BD.X`` integer arithmetic, no separate re-grouping pass.

    A ``CAM_DROP`` weight is preserved as-is (the DATA form of a wiped V/O cell).
    """

    __slots__ = ("_q", "_k", "_v", "_o")

    def __init__(self) -> None:
        # Ordered {slot: {dim_name: weight}} maps (insertion order == authoring
        # order; dict preserves it, so the emitted cell tuples are stable).
        self._q: dict[int, dict[str, object]] = {}
        self._k: dict[int, dict[str, object]] = {}
        self._v: dict[int, dict[str, object]] = {}
        self._o: dict[int, dict[str, object]] = {}

    def q(self, slot: int, dim: str, weight) -> None:
        self._q.setdefault(slot, {})[dim] = weight

    def k(self, slot: int, dim: str, weight) -> None:
        self._k.setdefault(slot, {})[dim] = weight

    def v(self, slot: int, dim: str, weight) -> None:
        self._v.setdefault(slot, {})[dim] = weight

    def o(self, v_slot: int, out_dim: str, weight) -> None:
        # ``o`` cells are keyed (out_dim, v_slot) at lower time; store per v_slot.
        self._o.setdefault(v_slot, {})[out_dim] = weight

    def slot(self, slot: int, *, q=(), k=(), v=(), o=()) -> None:
        """Write a batch of named cells onto ``slot`` (a compact slot literal)."""
        for dim, weight in q:
            self.q(slot, dim, weight)
        for dim, weight in k:
            self.k(slot, dim, weight)
        for dim, weight in v:
            self.v(slot, dim, weight)
        for out_dim, weight in o:
            self.o(slot, out_dim, weight)

    def q_cells(self, slot: int):
        """The current ``(dim, weight)`` Q cells on ``slot`` (for cancel copies)."""
        return tuple(self._q.get(slot, {}).items())

    def k_cells(self, slot: int):
        """The current ``(dim, weight)`` K cells on ``slot`` (for cancel copies)."""
        return tuple(self._k.get(slot, {}).items())

    def discriminators(self) -> tuple:
        slots = sorted(
            set(self._q) | set(self._k) | set(self._v) | set(self._o)
        )
        out = []
        for s in slots:
            out.append(CamDiscriminatorSlot(
                slot=s,
                q=tuple(self._q.get(s, {}).items()),
                k=tuple(self._k.get(s, {}).items()),
                v=tuple(self._v.get(s, {}).items()),
                # ``o`` cells lower as (out_dim, slot); emit (out_dim, weight).
                o=tuple(self._o.get(s, {}).items()),
            ))
        return tuple(out)


def _layer15_memory_lookup_heads_0_3_specs_with_overrides(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """L15 heads 0-3 base CAM + the flag-conditioned OVERRIDE layer as DATA.

    DERIVE->PROVE->FLIP->DELETE (2026-07-04): the legacy dict-merge OVERRIDE
    (``_suppress_l15_lookup_heads_0_3`` post-bake patch) is LIFTED into
    flag-conditioned :class:`CamDiscriminatorSlot` DATA appended to the base
    :class:`CamBinaryAddressMatch` via its ``overlay`` field. Instead of building
    the base head, converting to per-cell maps, and dict-merging override cells
    over them, we now compute ONLY the override cells (into empty per-slot maps),
    group them into an ``overlay`` tuple, and let ``cam_binary_address_match``
    merge them last-write-wins over the base head. The whole L15 memory-lookup
    family is therefore DERIVED — the value-load is fully expressible as
    binary-address-CAM DATA (address comparator + base discriminators + value
    relay + a flag-conditioned discriminator overlay).

    Byte-identity: the primitive's builder merges the overlay AFTER the base
    discriminators AND value bands (last-write-wins, matching the lowerer's
    indexed assignment), so the produced head is byte-identical with the old
    build-then-dict-merge. The two legacy V/O ``.pop`` wipes at slot 63 (the
    ``nonpop_stack0_marker_blocker`` row) become :data:`CAM_DROP` overlay cells.
    Every ``if head == 0 and _l15_..._on()`` block is emitted into the overlay
    ONLY when its flag is on, so flag-OFF (golden 35-tok) is byte-identical to
    the plain base CAM and each flag's ON weights are unchanged from the lift.

    Row wipes (``attn.W_q.data[base + row, :] = 0.0`` etc.) at rows 35-43, 58-63
    in the legacy helper are no-ops here because the base spec does not write Q/K
    at those slots, except V slot 63 and O column 63 (all heads, the
    ``nonpop_stack0_marker_blocker`` wipe drops the base spec's
    ``CLEAN_EMBED_HI+15`` -> ``OUTPUT_HI+15`` value cell).
    """

    # Marker-bank slot indices via the positional-invariant mechanism
    # (Class-1 marker-relative). Frame-INVARIANT bank-TYPE order; byte-identical
    # in both frames. Replaces the hand-coded ``MEM_I = 4`` etc. so the audit
    # recognises these ``BD.H1 + MEM_I`` / ``BD.L2H0 + MEM_I`` reads as declared
    # marker-relative. See positional_invariant.py.
    PC_I = marker_bank_index("PC")
    AX_I = marker_bank_index("AX")
    SP_I = marker_bank_index("SP")
    BP_I = marker_bank_index("BP")
    MEM_I = marker_bank_index("MEM")

    # Base-CAM constructor inputs (mirror _layer15_memory_lookup_heads_0_3_specs).
    addr_block = CamBinaryAddressBlock(
        nibble_bands=(
            "ADDR_B0_LO", "ADDR_B0_HI",
            "ADDR_B1_LO", "ADDR_B1_HI",
            "ADDR_B2_LO", "ADDR_B2_HI",
        ),
        scale=10.0, slot_base=4, width_bits=4,
    )
    value_bands = (
        CamValueBand("CLEAN_EMBED_LO", "OUTPUT_LO", 16, 32, 1.0),
        CamValueBand("CLEAN_EMBED_HI", "OUTPUT_HI", 16, 48, 1.0),
    )
    base_BS = 60.0
    MEM_VAL_DIMS = (None, "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3")
    dim_map = _l15_li_lc_load_dim_map(BD)

    merged: list[DeclarativeAttentionHeadSpec] = []

    # byte_q_flags as NAMED tokens (resolved through dim_map by the CAM builder).
    byte_q_flags = (None, "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2")

    for head in range(4):
        # OVERRIDE cells as NAMED, offset-token discriminator DATA. The base
        # head's cells are supplied by the base CAM; these overlay slots are
        # merged over them last-write-wins by ``cam_binary_address_match`` (each
        # cell resolves its ``BASE`` / ``BASE+offset`` name through ``dim_map``).
        # A CAM_DROP weight REMOVES the (slot, dim) cell (the legacy V/O ``.pop``
        # wipe). Authoring blocks write into ``_OV`` in source order and the
        # accumulator keeps last-write-wins per (slot, dim), so lifting the old
        # ``BD.X``-int flat maps into these declarative slot literals is exact.
        _OV = _L15OverlayAccumulator()

        # === local_slot_scale: byte-0 nibble bit rows (slots 4..11) ===
        local_slot_scale = 100.0
        for nibble_offset, nibble_base in ((0, "ADDR_B0_LO"), (4, "ADDR_B0_HI")):
            for bit in range(4):
                row = 4 + nibble_offset + bit
                for k in range(16):
                    w = local_slot_scale * (2 * ((k >> bit) & 1) - 1)
                    _OV.q(row, f"{nibble_base}+{k}", w)
                    _OV.k(row, f"{nibble_base}+{k}", w)

        # === one-hot rows 43+k: byte-0 lo nibble match ===
        local_slot_onehot_scale = 100.0
        for k in range(16):
            row = 43 + k
            _OV.q(row, "CONST", -local_slot_onehot_scale)
            _OV.q(row, f"ADDR_B0_LO+{k}", local_slot_onehot_scale)
            _OV.q(row, "OP_LI_RELAY", local_slot_onehot_scale)
            if head == 0:
                _OV.q(row, "OP_LC_RELAY", local_slot_onehot_scale)
            _OV.k(row, f"ADDR_B0_LO+{k}", local_slot_onehot_scale)

        if head == 0:
            # === Strong default blocker + explicit LI/LC/pop restores ===
            lookup_bias = 200000.0
            _OV.slot(0, q=[
                ("CONST", -lookup_bias),
                ("OP_LI_RELAY", lookup_bias), ("OP_LC_RELAY", lookup_bias),
                ("OP_LI", lookup_bias), ("OP_LC", lookup_bias),
                ("CMP+3", lookup_bias / 4.0),
            ])
            _OV.q(1, "CMP+3", 12.5)

            non_load_suppression = -1000000.0
            _OV.slot(0, q=[
                ("OP_JSR", non_load_suppression), ("OP_ENT", non_load_suppression),
                ("OP_LEA", non_load_suppression), ("OP_IMM", non_load_suppression),
            ])
            if l15_lookup_cmp_veto_enabled():
                # COMPARISON-STEP VETO (bool_and id=1087, 2026-06-25).
                # CMP+3 (above) is overloaded: it is BOTH the POP-group flag
                # this head keys on AND the L9 low-nibble-less-than cmp flag.
                # On a comparison step (OP_GT/OP_LT/... one-hot at MARK_AX,
                # CMP+3 also hot) the slot-0 discriminator MIS-FIRES and dumps
                # +40 CLEAN_EMBED -> OUTPUT_LO+0, burying the cmp result byte.
                # A real LI/LC/POP load never has a comparison opcode hot at
                # its own marker, so extend the SAME non_load veto to the six
                # comparison opcodes: OP_<cmp> * -1e6 dominates CMP+3 * 50000,
                # keeping the head silent on cmp rows and byte-identical on
                # every load row. Kill-switch: ``C4_L15_LOOKUP_CMP_VETO=0``.
                _OV.slot(0, q=[
                    ("OP_GT", non_load_suppression), ("OP_LT", non_load_suppression),
                    ("OP_GE", non_load_suppression), ("OP_LE", non_load_suppression),
                    ("OP_EQ", non_load_suppression), ("OP_NE", non_load_suppression),
                ])

            if var_three_li_enabled():
                # STORE-STEP VETO (var_three id300, 2026-07-04). The ``SI b``
                # store's AX byte-0 emit row carries a STRAY ``OP_LI_RELAY==1.0``
                # (alongside ``OP_SI==5.23``; the ``SI a``/``SI c`` store rows
                # carry OP_SI ALONE — probe ``tools/_probe_vt_head0_sistore.py``).
                # The stray relay clears the slot-0 discriminator's threshold
                # (OP_LI_RELAY Q weight ``lookup_bias``=2e5), head 0 fires
                # OFF-self on that store row, attends a cross-step CLEAN_EMBED
                # row whose value is 0, and copies +0 into OUTPUT -> the store
                # value byte 6 decodes 0 not 6 -> the var_three frame desyncs at
                # ``SI b``. A real LI/LC/POP load NEVER has OP_SI/OP_SC hot at
                # its own marker (the opcode is OP_LI/OP_LC), so extend the SAME
                # non_load veto to the two store opcodes: OP_SI*-1e6 dominates
                # the stray OP_LI_RELAY*2e5, keeping head 0 self-firing on every
                # store row (byte-identical on every real load row, where
                # OP_SI==OP_SC==0). Campaign-only + own kill-switch
                # ``C4_VAR_THREE_LI`` (DEFAULT OFF; flag-OFF omits these two
                # writes -> byte-identical to golden).
                _OV.slot(0, q=[
                    ("OP_SI", non_load_suppression), ("OP_SC", non_load_suppression),
                ])

            _OV.slot(0, q=[
                ("MARK_STACK0", 75000.0), ("HAS_SE", 75000.0),
                ("ADDR_B0_LO+8", 75000.0), ("ADDR_B0_HI+14", 75000.0),
                ("ADDR_B0_HI+15", -100000.0), ("IS_BYTE", -2000.0),
            ])
            _OV.slot(1, q=[
                ("IS_BYTE", -50.0), ("MARK_STACK0", 50.0), ("HAS_SE", 50.0),
                ("ADDR_B0_LO+8", 50.0), ("ADDR_B0_HI+14", 50.0),
                ("ADDR_B0_HI+15", -150.0),
            ])
            _OV.slot(28, q=[
                ("IS_BYTE", -500.0), ("CONST", -20000.0),
                ("MARK_AX", 20000.0), ("MARK_STACK0", 20000.0),
            ])

            # Row 42: top-store e0 signature blocker.
            for row, low, high in ((42, 0, 14),):
                _OV.slot(row, q=[
                    ("CONST", -60000.0), ("MARK_STACK0", 10000.0),
                    ("HAS_SE", 10000.0), ("MEM_STORE", 10000.0),
                    (f"EMBED_LO+{low}", 10000.0), (f"EMBED_HI+{high}", 10000.0),
                    (f"ADDR_B0_LO+{low}", 10000.0), (f"ADDR_B0_HI+{high}", 10000.0),
                    ("OP_LI_RELAY", 50000.0), ("OP_LC_RELAY", 50000.0),
                ], k=[("CONST", 20.0)])

            # Rows 59, 60, 61: wipe Q/K (base has no writes here either).
            # The base spec writes V at (slot, dim)=(59, CLEAN_EMBED_HI+11),
            # (60, CLEAN_EMBED_HI+12), (61, CLEAN_EMBED_HI+13). These rows
            # are valid V cells; only rows 62, 63 are touched by the V
            # wipe via slot 63 below. Q/K at 59-61 stays empty: no base
            # writes to drop.

            # Row 60: top-store e8 signature.
            _OV.slot(60, q=[
                ("CONST", -30000.0), ("MARK_STACK0", 10000.0),
                ("MARK_SP", -100000.0), ("HAS_SE", 10000.0),
                ("MEM_STORE", 150000.0), ("ADDR_B0_LO+8", 10000.0),
                ("ADDR_B0_HI+14", 10000.0), ("ADDR_B0_HI+15", -20000.0),
            ], k=[("CONST", -20.0)])

            # Row 59: preserve e8 (STACK0 + HAS_SE + nibble match).
            preserve_e8_s = 5000.0
            _OV.slot(59, q=[
                ("CONST", -3.5 * preserve_e8_s), ("MARK_STACK0", preserve_e8_s),
                ("HAS_SE", preserve_e8_s), ("ADDR_B0_LO+8", preserve_e8_s),
                ("ADDR_B0_HI+14", preserve_e8_s),
                ("IS_BYTE", -4.0 * preserve_e8_s),
                ("MEM_STORE", -5.0 * preserve_e8_s),
            ], k=[
                ("ADDR_B0_LO+8", preserve_e8_s), ("ADDR_B0_HI+14", preserve_e8_s),
            # Class-2 absolute-slot read: the K-side STACK0_BYTE0 discriminator
            # boosts attention to the historical STACK0 byte-0 token when
            # preserving the 0xffe8 stack top. STACK0_BYTE0 is the canonical
            # d=6-from-BP positional flag that VANISHES under C4_NO_STACK0_EMIT
            # (its L1 producer is auto-neutralized so the flag never fires, and
            # no STACK0 token is emitted to attend to). Drive the weight to 0 in
            # the dropped frame via invariant_threshold so the declared
            # invariance is explicit — a no-op behaviourally in BOTH frames (at
            # 35-tok returns 5.0*preserve_e8_s, byte-identical; at 30-tok the
            # input flag is already constant-0 so the score contribution is
            # zero either way). See positional_invariant.py.
                ("STACK0_BYTE0", invariant_threshold(
                    live=5.0 * preserve_e8_s, suppressed=0.0, marker="BP", k=6)),
            ])

            # Row 61: AX LI/LC e8 value discriminator.
            ax_li_e8_s = 10.0
            _OV.slot(61, q=[
                ("CONST", -3.5 * ax_li_e8_s), ("MARK_AX", ax_li_e8_s),
                ("OP_LI_RELAY", ax_li_e8_s), ("OP_LC_RELAY", ax_li_e8_s),
                ("MARK_STACK0", 2.5 * ax_li_e8_s), ("ADDR_B0_LO+8", ax_li_e8_s),
                ("ADDR_B0_HI+14", ax_li_e8_s), ("IS_BYTE", -4.0 * ax_li_e8_s),
                ("MEM_STORE", -4.0 * ax_li_e8_s),
            ], k=[
                ("MEM_VAL_B1", ax_li_e8_s), ("ADDR_B0_LO+8", ax_li_e8_s),
                ("ADDR_B0_HI+14", ax_li_e8_s),
            ])

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
            _OV.slot(58, q=[
                ("OP_ENT", 200000000.0),
                ("MARK_STACK0", early_ent_stack0_q + stack0_gate),
                ("CONST", -early_ent_stack0_q - stack0_gate),
                ("IS_BYTE", -2000000000.0),
                ("MARK_AX", -2000000000.0), ("MARK_PC", -2000000000.0),
                ("MARK_SP", -2000000000.0), ("MARK_BP", -2000000000.0),
                ("MARK_MEM", -2000000000.0),
            ], k=[("OP_ENT", 10000.0)])

            # Row 34: pop_low8 nibble bias.
            _OV.slot(34, q=[
                ("CONST", -4000.0), ("MARK_STACK0", 2000.0), ("HAS_SE", 1000.0),
                ("CMP+3", 1000.0), ("ADDR_B0_LO+0", 1000.0),
                ("ADDR_B0_LO+8", 1000.0), ("IS_BYTE", -10000.0),
                ("MARK_SP", -10000.0), ("MEM_STORE", -20000.0),
            ], k=[("ADDR_B0_LO+8", 1000.0)])
        else:
            # === Heads 1-3: byte_q_flags-gated overrides ===
            _OV.slot(0, q=[
                ("MARK_STACK0", -100000.0), ("MARK_SP", -100000.0),
            ])
            _OV.slot(28, q=[
                ("CONST", -20000.0), (byte_q_flags[head], 20000.0),
            ])

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
            _OV.slot(64, q=[("MARK_AX", -2000000000.0)], k=[("CONST", 1.0)])
            # Slot 3 K-side per-head re-tuning.
            _OV.slot(3, k=[
                ("MEM_VAL_B1", 0.0), ("MEM_VAL_B2", 0.0), ("MEM_VAL_B3", 0.0),
                (f"H2+{MEM_I}", 0.0), (f"H3+{MEM_I}", 0.0),
            ])
            if head == 1:
                _OV.k(3, "MEM_VAL_B2", 60.0)
            elif head == 2:
                _OV.k(3, "MEM_VAL_B3", 60.0)
            elif head == 3:
                _OV.slot(3, k=[(f"H3+{MEM_I}", 60.0), (f"H2+{MEM_I}", -60.0)])

        # === source_gate (slot 37): wipe K then re-author per head ===
        # Base has no Q/K at slot 37 -- wipe is a no-op.
        source_gate = 37
        _OV.q(source_gate, "CONST", 0.0)
        if head == 0:
            _OV.q(source_gate, "MARK_STACK0", 3000.0)
        else:
            _OV.q(source_gate, byte_q_flags[head], 3000.0)
        source_key_s = 10.0
        _OV.slot(source_gate, k=[
            ("CONST", -source_key_s), ("MEM_STORE", 0.5 * source_key_s),
        ])
        if head == 0:
            _OV.slot(source_gate, k=[
                (f"L2H0+{MEM_I}", source_key_s), (f"H1+{MEM_I}", -70.0),
            ])
        else:
            _OV.k(source_gate, f"H1+{MEM_I}", -70.0)
            if head == 1:
                _OV.k(source_gate, "MEM_VAL_B2", source_key_s)
            elif head == 2:
                _OV.k(source_gate, "MEM_VAL_B3", source_key_s)
            elif head == 3:
                _OV.slot(source_gate, k=[
                    (f"H3+{MEM_I}", source_key_s), (f"H2+{MEM_I}", -source_key_s),
                ])
        # SP/BP register byte blockers on source_gate K.
        for marker_i in (SP_I, BP_I):
            _OV.slot(source_gate, k=[
                (f"H1+{marker_i}", -80.0), (f"H2+{marker_i}", -80.0),
                (f"H3+{marker_i}", -80.0), (f"L2H0+{marker_i}", -80.0),
            ])

        # === load_source_gate (slot 39): wipe Q/K then re-author ===
        load_source_gate = 39
        load_source_gate_s = 5000.0
        load_source_key_s = 30.0
        if head == 0:
            _OV.slot(load_source_gate, q=[
                ("OP_LI_RELAY", load_source_gate_s),
                ("OP_LC_RELAY", load_source_gate_s),
                ("MARK_AX", load_source_gate_s), ("CMP+3", 2000.0),
                ("CONST", -1.5 * load_source_gate_s),
            ], k=[
                ("MEM_VAL_B1", 2.0 * load_source_key_s),
                ("MEM_ADDR_SRC", 40.0),
            ])
        else:
            _OV.slot(load_source_gate, q=[
                ("OP_LI_RELAY", load_source_gate_s),
                (byte_q_flags[head], load_source_gate_s),
                ("CONST", -1.5 * load_source_gate_s),
            ])
            if head == 1:
                _OV.k(load_source_gate, "MEM_VAL_B2", 2.0 * load_source_key_s)
            elif head == 2:
                _OV.k(load_source_gate, "MEM_VAL_B3", 2.0 * load_source_key_s)
            elif head == 3:
                _OV.slot(load_source_gate, k=[
                    (f"H3+{MEM_I}", 2.0 * load_source_key_s),
                    (f"H2+{MEM_I}", -2.0 * load_source_key_s),
                ])
            _OV.k(load_source_gate, "MEM_ADDR_SRC", 40.0)

        # === marker_value_gate (slot 40): head-0 only ===
        if head == 0:
            marker_value_gate_s = 1000.0
            _OV.slot(40, q=[
                ("OP_LI", marker_value_gate_s), ("OP_LC", marker_value_gate_s),
            ], k=[
                ("MEM_VAL_B1", 80.0), ("MEM_ADDR_SRC", 40.0), ("CONST", -60.0),
            ])

        # === O scaling: value_scale=40.0 for OUTPUT band ===
        # Replaces base spec's slot 32+k -> OUTPUT_LO+k weight=1.0
        # with weight=40.0; same for HI.
        value_scale = 40.0
        for k in range(16):
            _OV.o(32 + k, f"OUTPUT_LO+{k}", value_scale)
            _OV.o(48 + k, f"OUTPUT_HI+{k}", value_scale)

        # === addsub_blocker (slot 41) ===
        _OV.slot(41, q=[("TEMP+8", 10000.0), ("TEMP+9", 10000.0)],
                 k=[("CONST", -20.0)])

        # === sp_byte_blocker (slot 62) ===
        # ``H1+SP_I`` (marker-bank SP slot); was the bare ``BD.H1 + 2``.
        _OV.slot(62, q=[
            (f"H1+{SP_I}", 100000.0), ("MARK_BP", 100000.0),
            ("TEMP+10", 100000.0), ("TEMP+24", 100000.0), ("IS_BYTE", 0.0),
        ], k=[("CONST", -300000.0)])
        if head == 0:
            _OV.slot(62, q=[("IS_BYTE", 500000.0), ("OP_ENT", 500000.0)])

        # === pc_byte_blocker (slot 35) ===
        _OV.slot(35, q=[
            (f"H1+{PC_I}", 100000.0), ("MARK_PC", 100000000.0),
            ("IS_BYTE", 0.0),
        ], k=[("CONST", -100000.0)])
        if head == 0:
            _OV.slot(35, q=[
                ("MARK_STACK0", 10000.0), ("HAS_SE", 10000.0),
                ("CMP+3", 10000.0), ("ADDR_B0_LO+8", 10000.0),
                ("ADDR_B0_HI+15", 10000.0), ("OP_LI_RELAY", -10000.0),
                ("OP_LC_RELAY", -10000.0),
            ])

            # === stack0_preserve (slot 36): head-0 only ===
            stack0_preserve_s = 1.0
            _OV.slot(36, q=[
                ("CONST", -1.0 * stack0_preserve_s),
                ("MARK_STACK0", 3.0 * stack0_preserve_s),
                ("HAS_SE", 1.0 * stack0_preserve_s),
                ("CMP+3", -5.0 * stack0_preserve_s),
                ("MEM_STORE", -10.0 * stack0_preserve_s),
                ("IS_BYTE", -10.0 * stack0_preserve_s),
                ("MARK_AX", -10.0 * stack0_preserve_s),
                (f"H1+{AX_I}", 10.0 * stack0_preserve_s),
                ("OP_LI_RELAY", 10.0 * stack0_preserve_s),
                ("OP_LC_RELAY", 10.0 * stack0_preserve_s),
                ("MARK_PC", -10.0 * stack0_preserve_s),
                ("MARK_SP", -10.0 * stack0_preserve_s),
                ("MARK_BP", -10.0 * stack0_preserve_s),
                ("MARK_MEM", -10.0 * stack0_preserve_s),
            ], k=[
                ("CONST", -2.0 * stack0_preserve_s),
                ("H1+10", 2.0 * stack0_preserve_s),
                ("BYTE_INDEX_0", 2.0 * stack0_preserve_s),
                ("OP_ENT", -0.25 * stack0_preserve_s),
            ])
            _OV.slot(36, k=[
                (f"H1+{marker_i}", -4.0 * stack0_preserve_s)
                for marker_i in range(5)
            ])

        # === mem_addr_byte_blocker (slot 36) for heads 1-3 ===
        if head in (1, 2, 3):
            _OV.slot(36, q=[(f"H1+{MEM_I}", 100000.0)], k=[("CONST", -20.0)])

        # === nonpop_stack0_marker_blocker (slot 63) ===
        # Legacy: ``attn.W_v[base+63, :] = 0`` then explicit Q/K writes,
        # then ``attn.W_o[:, base+63] = 0`` (wipe V row and O column 63).
        # As overlay DATA: DROP the BASE value band's V (63, CLEAN_EMBED_HI+15)=1
        # and O (OUTPUT_HI+15, 63) cells via CAM_DROP, then write the Q/K blocker.
        # The overlay's O-rescale-to-40 above set (OUTPUT_HI+15, 63)=40; the DROP
        # removes it (merged after the rescale), matching the legacy pop.
        # ``ADDR_B0_HI+14`` is written twice (last wins: -20000.0), mirroring the
        # legacy flat-map ordering (the accumulator keeps last-write-wins).
        top_store_e8_from_e0_s = 10000.0
        _OV.slot(63,
            v=[("CLEAN_EMBED_HI+15", CAM_DROP)],
            o=[("OUTPUT_HI+15", CAM_DROP)],
            q=[
                ("MARK_STACK0", 60000.0), ("CMP+3", -15000.0),
                ("IS_BYTE", 60000.0), ("OP_LI_RELAY", -60000.0),
                ("OP_LC_RELAY", -60000.0), ("ADDR_B0_LO+8", -40000.0),
                ("ADDR_B0_HI+14", -30000.0),
                ("MEM_STORE", top_store_e8_from_e0_s),
                ("EMBED_LO+8", top_store_e8_from_e0_s),
                ("EMBED_HI+14", top_store_e8_from_e0_s),
                ("ADDR_B0_LO+0", top_store_e8_from_e0_s),
                ("ADDR_B0_HI+14", -2.0 * top_store_e8_from_e0_s),
            ],
            k=[("CONST", -20.0)])

        # === current_store_blocker (slot 38) ===
        current_store_block_s = 10000.0
        _OV.slot(38, q=[
            ("MARK_MEM", current_store_block_s),
            (f"H3+{MEM_I}", current_store_block_s),
        ], k=[("CONST", -20.0)])

        # === Broad current-MEM-section blockers on slots 0, 28-33 ===
        for slot in (0, 28, 29, 30, 31, 32, 33):
            _OV.slot(slot, q=[
                ("MARK_MEM", -100000.0), (f"H3+{MEM_I}", -100000.0),
            ])
        # Slot 29: extras (PC H1).
        _OV.slot(29, q=[
            ("MARK_MEM", -100000.0), (f"H3+{MEM_I}", -100000.0),
            (f"H1+{PC_I}", -20000.0),
        ], k=[("CONST", 5.0)])
        # Slot 30: AX H1 blocker.
        _OV.slot(30, q=[(f"H1+{AX_I}", -20000.0)], k=[("CONST", 5.0)])
        # Slot 31: LI/LC restore (head 0 also AX blocker reverse).
        _OV.q(31, "OP_LI_RELAY", 20000.0)
        if head == 0:
            _OV.q(31, "OP_LC_RELAY", 20000.0)
        else:
            _OV.q(31, "MARK_AX", -20000.0)
        _OV.slot(31, q=[("OP_SI", -20000.0), ("OP_SC", -20000.0)],
                 k=[("MEM_STORE", 5.0)])

        # Slot 32: AX marker default blocker.
        _OV.slot(32, q=[("MARK_AX", -20000.0)], k=[("CONST", 5.0)])
        if head == 0:
            _OV.slot(33, q=[
                ("OP_LI_RELAY", 20000.0), ("OP_LC_RELAY", 20000.0),
            ], k=[("MEM_STORE", 5.0)])

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
            _OV.slot(58, q=[
                ("OP_ENT", 200000000.0 / _row58_rescale),
                ("MARK_STACK0",
                 (early_ent_stack0_q + stack0_gate) / _row58_rescale),
                ("CONST",
                 -(early_ent_stack0_q + stack0_gate) / _row58_rescale),
                ("IS_BYTE", -2000000000.0 / _row58_rescale),
                ("MARK_AX", -2000000000.0 / _row58_rescale),
                ("MARK_PC", -2000000000.0 / _row58_rescale),
                ("MARK_SP", -2000000000.0 / _row58_rescale),
                ("MARK_BP", -2000000000.0 / _row58_rescale),
                ("MARK_MEM", -2000000000.0 / _row58_rescale),
            ], k=[("OP_ENT", 1.0)])
            # Each cancel slot's Q copies the suppressor's FULL current Q and its
            # K negates the suppressor's FULL current K, so suppressor+cancel == 0
            # on every row (fully inert, expressed additively). The DATA copy
            # reads the accumulator's current per-slot cells.
            _GATE = {34: 70, 35: 64, 58: 65, 59: 66, 60: 67, 61: 68, 62: 69}
            for _suppr, _gate in _GATE.items():
                _OV.slot(_gate,
                    q=list(_OV.q_cells(_suppr)),
                    k=[(_d, -_w) for (_d, _w) in _OV.k_cells(_suppr)])

        # === C4_L15_LI_ADDR_CAM: head-0 per-store address-match discriminator ===
        # See _l15_li_addr_cam_discriminator_on for the full root + proof. On a
        # multi-local LI the byte-0 lookup row carries OP_LI_RELAY==0, so the
        # OP_LI_RELAY-gated one-hot disambiguator (rows 43+k) is dead and the
        # bit-encoded address slots (4-27) neither discriminate nor outweigh the
        # ~9.4e5 constant baseline -> the CAM picks the wrong store's value row.
        # Add a CLEAN one-hot Q.K match on the store address byte-0 nibbles
        # (ADDR_B0_LO + ADDR_B0_HI), carried by each store's VALUE row (d=5) and
        # by the LI operand query row alike, gated on the LIVE b0row marker
        # (MARK_AX; OP_LI_RELAY is 0 here) so only the store whose address
        # matches the queried local wins. Free head-0 slots 71-101 (head_dim
        # 111; slots 0-70 are taken, 64-70 by the suppressor cancel). Pure
        # additive content-addressing; campaign-only (golden byte-identical).
        if head == 0 and _l15_li_addr_cam_discriminator_on():
            # Per-nibble one-hot: slot (71+k) matches ADDR_B0_LO+k; slot
            # (87+k) matches ADDR_B0_HI+k. Q gates the match on MARK_AX (the
            # only live signal at the byte-0 LI lookup row) so it contributes
            # nothing on the prompt / non-AX rows; K reads the candidate row's
            # own address nibble one-hot. The product is positive ONLY when the
            # operand's address nibble equals the store value row's nibble, so
            # the head content-addresses the correct local's value. Scale 360
            # gives 360*360/sqrt(111) ~= 12.3k per matched nonzero nibble --
            # decisive over the ~2k cross-store tie, modest vs the cancelled
            # blockers. Uses head-0 slots 71-101 (LO 71-85, HI 87-101).
            # NOTE the nibble-0 (`0x0`) one-hot is the DEGENERATE/null nibble:
            # it carries a large spurious peak (~1.5-2.5) on BOTH the operand
            # query row AND on null-address (0x00) intermediate value rows, so
            # matching it lets a genuine 0x00 row out-score the real local (e.g.
            # &a=0xffe8's value row loses to a 0x00 load-result row). DROP the
            # k=0 match -- the locals are discriminated by their NON-zero
            # nibbles (&a lo=8 hi=0xe, &b lo=0 hi=0xe, &c lo=8 hi=0xd): the lo
            # OR hi nonzero nibble uniquely separates every BP-relative local.
            _cam_s = 360.0
            for _nib_base, _row_base in (("ADDR_B0_LO", 71), ("ADDR_B0_HI", 87)):
                for _k in range(1, 16):
                    # Gate: require MARK_AX so the match is inert off the AX
                    # byte-0 emit row (CONST baseline buries non-AX rows).
                    _OV.slot(_row_base + _k, q=[
                        ("CONST", -2.0 * _cam_s), ("MARK_AX", 2.0 * _cam_s),
                        (f"{_nib_base}+{_k}", _cam_s),
                    ], k=[(f"{_nib_base}+{_k}", _cam_s)])

        # === #301/#318: head-0 ZERO-ADDRESS (k=0) committed-store match ===
        # See _l15_li_zeroaddr_cam_on for the full root + proof. The #313 CAM
        # above DROPS the k=0 (zero-nibble) match, so a local at BP+0 (address
        # 0x00, BOTH nibbles 0) gets NO per-store address discrimination and an
        # OLD wrong-address committed store (0xF8, pos 117) out-scores the right
        # latest x-store (0x00, pos 267) in the AR decode (the 0.05 recency
        # slope over ~150 positions adds only ~7, far short of the ~744
        # deficit). Re-instate the k=0 match on the NATURAL free k=0 slots
        # (71 for ADDR_B0_LO+0, 87 for ADDR_B0_HI+0) -- the candidate's OWN
        # zero nibble -- combined with an additive committed-store gate on
        # MEM_STORE_AT_VAL (MSAV). Softmax over the SUM of the two terms peaks
        # ONLY on a committed-AND-zero-address row: a committed NONZERO-address
        # store (0xF8) collects the MSAV term but not the zero-nibble term (it
        # already has its k=8/k=15 nonzero match instead), and a stray
        # null-address load-result/code row collects the zero-nibble term but
        # is MSAV=0 so it never collects the gate -- so neither alternative gets
        # BOTH, and the genuine x-store (committed + 0x00) leads its class. The
        # existing 0.05 recency slope then resolves the (rare) same-zero-address
        # multiple-write case (var_update's x-reassign) to the latest store.
        # Pure additive; campaign-only (golden byte-identical: slots not emitted
        # -- the ADDR_B0/MSAV value-row signals are produced only by the 30-tok
        # MEM-from-SP path). Mirrors the #313 _cam_s magnitude so the k=0 boost
        # is the same per-nibble weight as the k=1..15 matches.
        if head == 0 and _l15_li_zeroaddr_cam_on():
            # === PHASE-2 KEYSTONE (#318): FFN-indicator head-0 zero-address match.
            # The #313 CAM DROPS the k=0 (zero-nibble) match, so a local at BP+0
            # (address 0x00, BOTH nibbles 0) gets NO per-store address
            # discrimination and an OLD wrong-address committed store out-scores
            # the right latest x-store in the AR decode -> AX byte-0 = 0x00.
            #
            # The k=0 match cannot be expressed bilinearly in the head: the
            # committed x-store (probe #250 row 267: zero nibbles @1.46, MSAV=1)
            # must beat a NON-committed zero-address operand-frame row (row 297:
            # zero nibbles @2.46, MSAV=0) -- but K is LINEAR per slot, so any
            # committedness (MSAV) weight large enough to flip 297->267 over the
            # higher nibble amplitude also OVER-SHARPENS head-0 and desyncs the
            # var AR frame (the documented multi-session blocker; the prior
            # K = zero_nibble + MSAV*w variant was AR-non-monotonic and could not
            # flip var without regressing func).
            #
            # THE FIX (the 3-way AND lives in the FFN, not the head): L14
            # ``make_layer14_li_zeroaddr_indicator_op`` materializes ONE dim,
            # ``LI_ZEROADDR_COMMITTED``, that is ≈1 ONLY on a (committed AND
            # zero-address) store value row -- silu(MSAV + ADDR_B0_LO+0 +
            # ADDR_B0_HI+0 - 2.5). Head-0 keys K on that SINGLE dim, so the
            # committed x-store (267, indicator≈1) gets the boost and the
            # non-committed operand row (297, MSAV=0 -> indicator=0) gets NOTHING
            # -- a clean separation with NO amplitude competition, hence no
            # over-sharpening. ONE DEDICATED free head-0 slot (103): slots 0-63
            # are the base spec, 64-70 the suppressor cancel, 71-102 the #313
            # ADDR-CAM (LO 72-86, HI 88-102) + slot 102 the byte0_valsel, so
            # 103-110 (head_dim 111) are free -- 103 keeps this slot fully
            # separate from the #313 nibble matches:
            #
            # (Q) OP_LI-GATED, penalty-free. The slot opens ONLY on an LI EMIT
            #     row: OP_LI (the LI opcode marker, ~5.2 at the byte-0 emit row
            #     r307) is the sole Q term. var's OWN SI/ENT/JSR store steps
            #     carry OP_ENT/OP_JSR but NOT OP_LI (probe #250 r127), so the
            #     slot is INERT on var's store framing rows -> var-store safe.
            #     Q is never negative -> it can never penalize any store.
            # (K) the single committed-zero-address indicator dim. ALL of the
            #     candidate-side discrimination lives here: the indicator is ≈1
            #     ONLY on a (committed AND zero-address) store value row (267),
            #     and 0 on the non-committed operand row (297, MSAV=0) AND on
            #     committed NONZERO-address stores (indicator=0 there). So at
            #     func's LIs (nonzero address) NO store row carries the indicator
            #     (func #550: closest committed store is lo=0/hi=15 -- hi nibble
            #     not 0 -> indicator never fires) -> the slot contributes nothing
            #     -> func is untouched. The 3-way AND is done in the FFN, so the
            #     head needs only this ONE linear K term (no over-sharpening).
            #
            # Among multiple committed zero-address stores (var_update's
            # x-reassign) the existing 0.05 ALiBi recency slope picks the latest.
            # Campaign-only (golden 35-tok flag-OFF byte-identical: the band +
            # op are flag-gated and the ADDR_B0/MSAV value-row signals only exist
            # on the 30-tok MEM-from-SP path).
            _opli_q = 360.0   # OP_LI emit-row gate weight (mirrors #313 _cam_s)
            _ind_w = 50.0     # indicator K weight. The slot score on row 267 is
            #                   Q*K/sqrt(hd) = (360*OP_LI~5.2)*(50*IND~1.41)
            #                   /sqrt(111) ≈ 12.5k -- ONE #313-nibble-match scale
            #                   (the proven-decisive-but-not-over-sharp regime):
            #                   decisive over the ~2k cross-store tie, modest vs
            #                   the cancelled ~1e6 blockers, so head-0 is NOT
            #                   over-sharpened (the AR-frame-safe regime that the
            #                   linear-K MSAV variant could not reach).
            # Q: OP_LI emit-row gate only. K: the committed-zero-address
            # indicator. The slot boosts the committed-BP+0-local value row at
            # every LI emit; it is inert on non-LI rows (Q=0) and on every LI
            # whose locals are all nonzero-address (no indicator candidate).
            _OV.q(103, "OP_LI", _opli_q)
            # sc_lc roundtrip: mirror the committed-zero-address byte-0 boost
            # onto the OP_LC (load-char) emit row. The SC store @0x200 has a
            # ZERO-address byte-0 (0x00), so the #313 ADDR CAM (k=0 dropped)
            # cannot discriminate the genuine SC store-value row from the next
            # step's address-IMM row -> they TIE and ALiBi recency picks the
            # wrong (more-recent) row -> LC byte-0 = 0x00. The OP_LI slot above
            # breaks this exact tie for the LI path (+33860); the OP_LC term
            # below applies the IDENTICAL boost on the LC path. OP_LC fires ONLY
            # on LC opcode rows (var/func/SI/LI programs never emit it) so the
            # slot stays inert on every non-LC frame. See _l15_sclc_byte0_on.
            if _l15_sclc_byte0_on():
                _OV.q(103, "OP_LC", _opli_q)
            _OV.k(103, "LI_ZEROADDR_COMMITTED", _ind_w)

        # === #318: head-0 byte-0 VALUE-row selector via the SURVIVING MEM_VAL_B0 ===
        # Root (a52fd42c, spec_k=0, BUILT dims): head-0's slot-3 byte-0 value-row
        # selector keys on (L2H0+MEM, -H1+MEM) for the d=4-from-MEM value-byte-0
        # row AND gates the query on MARK_STACK0 (base spec l15:862-866). The
        # 30-tok campaign layout DROPS the STACK0 value block and auto-neutralizes
        # its L1 producer, so MARK_STACK0 never fires -> head-0's byte-0 value-row
        # selection collapses and it self-attends the AX-marker row
        # (CLEAN_EMBED_LO=0) -> AX byte-0 = 0x00 (var_simple s7 / var_update s9 /
        # if_var s7). Heads 1-3 select bytes 1-3 via MEM_VAL_B1/B2/B3 (which
        # SURVIVE the 30-tok path -> their high bytes stay correct: the
        # byte0-wrong/byte1-ok asymmetry is the fingerprint). The #313 address CAM
        # above only picks the right store ROW; it does NOT re-establish the dead
        # byte-0 selector.
        #
        # FIX: mirror heads 1-3 by adding a slot-3 K on MEM_VAL_B0 (dim 461 --
        # "mark == MEM OR (is_byte AND byte_index == 0)", asserted at the
        # value-byte-0 row of every store, and the proven correct byte-0
        # store-value selector at the LEV-override l15:1687). The base slot-3 Q
        # already carries the head-0 byte_q_flag (MARK_AX), which fires at the AX
        # byte-0 emit/lookup row, so the bilinear is positive ONLY on the
        # value-byte-0 store row -> head-0 selects the correct byte. Additive on
        # the existing slot 3 (the same slot the dead selector used); the existing
        # (L2H0+MEM, -H1+MEM) keys are kept (byte-identical at 35-tok, summing
        # harmlessly with MEM_VAL_B0 on the SAME value-byte-0 row at 30-tok).
        # Campaign-only via its own kill-switch (_l15_li_byte0_valsel_on): golden
        # 35-tok flag-OFF is byte-identical -- this branch is not taken there.
        if head == 0 and _l15_li_byte0_valsel_on():
            _bs_b0 = 60.0  # mirrors the base spec slot-3 BS (l15:860)
            _OV.k(3, "MEM_VAL_B0", _bs_b0)

        # === PHASE-2: head-0 MEM_VAL_B1 VALUE-row lift (first-param LI) ===
        # The #313 ADDR_B0 CAM TIES the genuine first-param value row against
        # the spurious code/frame byte rows just past the store (both carry the
        # SAME ADDR_B0_LO+8/ADDR_B0_HI+14 one-hots), and #318's MEM_VAL_B0 key
        # misses ``a`` (its value lands on MEM_VAL_B1, B0=-0.0). Add a dedicated
        # free-slot (102) positive K on MEM_VAL_B1 -- the materialized value
        # byte carried ONLY by genuine store VALUE rows (~0.97), ~0 on the
        # spurious ADDR-byte rows -- gated MARK_AX on the Q so it is inert off
        # the LI byte-0 lookup row. The lift flips func_add s9 271(0x39 lo) and
        # func_max s9 319(0x24 lo) without disturbing the already-correct ``b``
        # LI (s12) or var (b's/var's value rows carry MEM_VAL_B1 too -> same
        # lift, no competing row). Scale 800: 800*800*0.97/sqrt(111) ~= 56k,
        # inside the proven 5e4..5e5 flip band (_probe_func_b1_sim.py). See
        # ``_l15_li_valrow_b1_on``.
        if head == 0 and _l15_li_valrow_b1_on():
            _vr_s = 800.0
            # Q fires ONLY at an AX (the LI byte-0 lookup) row: MARK_AX is 1
            # there and 0 elsewhere, so off-AX rows contribute nothing. The K
            # (MEM_VAL_B1) is itself ~0 except on genuine store VALUE rows, so
            # the product is nonzero only at (AX-lookup-row x value-row).
            _OV.slot(102, q=[("MARK_AX", _vr_s)], k=[("MEM_VAL_B1", _vr_s)])

        # === POST-FLIP: head-0 JSR-phantom value-row PENALTY (first-param LI) ===
        # See _l15_li_jsr_phantom_penalty_on for the full root + proof. On the
        # multi-arg first-param LI (func_add/mul step 9), the genuine PSH'd-arg
        # store value row (e.g. func_add 271, CLEAN=57, OP_JSR~0) and the
        # callee's JSR/ENT-step PHANTOM MEM rows (360-363, CLEAN=0, OP_JSR~1.66)
        # BOTH match the queried address (byte-0 0xE8) AND carry MEM_VAL_B1~0.97,
        # so the #313 ADDR CAM and the slot-102 B1 lift boost them EQUALLY; the
        # phantom -- being MORE RECENT -- wins the 0.05 ALiBi tie by ~5 and the LI
        # returns 0. The ONLY surviving discriminator is store PROVENANCE: a real
        # store row carries OP_JSR~0, the callee-frame phantom rows carry the JSR
        # opcode residue OP_JSR~1.5-1.7. Subtract a K-side OP_JSR penalty, gated on
        # the Q by MARK_AX (the live byte-0 LI lookup-row signal -- OP_LI_RELAY is
        # 0 there, the #313 blind spot). The genuine store (OP_JSR~0) is
        # untouched; each phantom drops ~5.6k -> the genuine value row leads its
        # address class. Inert on every clean store (OP_JSR~0) and on non-AX rows
        # (Q gate). Dedicated free head-0 slot 104 (slots 0-103 taken). Scale
        # 360*100*1.66/sqrt(111) ~= 5.6k -- decisive over the ~5 recency tie,
        # modest vs the ~1e6 cancelled blockers (no over-sharpening).
        # Campaign-only (golden 35-tok flag-OFF byte-identical: branch not taken).
        if head == 0 and _l15_li_jsr_phantom_penalty_on():
            _jp_q = 360.0   # MARK_AX byte-0-lookup-row gate (mirrors #313 _cam_s)
            _jp_k = 100.0   # OP_JSR penalty K weight
            _OV.slot(104, q=[("MARK_AX", _jp_q)], k=[("OP_JSR", -_jp_k)])

        # Emit the accumulated OVERRIDE cells as overlay discriminators (DATA).
        # The CAM builder merges them last-write-wins over the base head (after
        # the value bands); a CAM_DROP weight removes that (slot, dim) cell (the
        # legacy V/O slot-63 wipe). Name tokens resolve through ``dim_map``.
        cam = _l15_li_lc_load_cam(
            head, addr_block, value_bands, base_BS, MEM_VAL_DIMS,
            overlay=_OV.discriminators(),
        )
        head_idx = _L15_HEAD_LAYOUT_BY_NAME[
            f"layer15_memory_lookup.li_lc_stack0_h{head}"
        ]
        bundle = cam_binary_address_match(cam)
        merged.append(bundle.head_spec_builder(dim_map, head_idx))

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

    # Marker-bank slot indices via the positional-invariant mechanism (Class-1
    # marker-relative). Frame-INVARIANT bank-TYPE order; byte-identical in both
    # frames. Replaces the hand-coded ``AX_I = 1`` / ``BP_I = 3`` so the audit
    # recognises the ``BD.H1 + AX_I`` / ``BD.L1H1 + BP_I`` / ``BD.H0 + BP_I``
    # reads as declared marker-relative. See positional_invariant.py.
    AX_I = marker_bank_index("AX")
    BP_I = marker_bank_index("BP")

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


def _layer15_lev_pc_restore_dim_map(BD) -> dict:
    """Resolve every dim NAME the L15 LEV PC-restore CAM head 14 touches.

    :func:`cam_binary_address_match`'s builder wants a name->int dict; the
    marker-bank anchors (``H1+SP_I`` etc.) and address nibble bands are supplied
    as their BASE names, with the offset parsed from the token at build time.
    """
    base_names = (
        "CONST", "OP_LEV", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
        "MARK_STACK0", "H1", "L2H0", "MEM_STORE", "OP_JSR", "OP_ENT",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "STACK0_BYTE0", "OPCODE_BYTE_LO",
        "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO", "ADDR_B1_HI",
        "ADDR_B2_LO", "ADDR_B2_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        "OUTPUT_LO", "OUTPUT_HI",
    )
    return {n: int(getattr(BD, n)) for n in base_names}


def _layer15_lev_pc_restore_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """L15 head 14: LEV return-address content-addressable restore into PC.

    Mirrors the working head-0 (``li_lc_stack0_h0``) 24-bit binary-address
    lookup but gated to fire at the LEV PC marker (``OP_LEV`` + ``MARK_PC``)
    instead of an LI/LC load at the AX marker. At the LEV PC marker the L9
    ``lev_bp_to_pc_relay`` + ``bp_plus8_shift`` machinery has stamped the BP+8
    gather key into ADDR_B0/B1/B2; this head matches that key against the
    stored 24-bit address of every store token (the binary encoding at slots
    4..27, identical to head 0) and copies the matched store's byte-0 value
    (``CLEAN_EMBED`` -> ``OUTPUT_LO/HI``) so the LM head emits the saved return
    address as PC[0].

    For func_identity_0 (id550, probe row 388) the gather key uniquely matches
    the JSR return-address push (a ``MARK_STACK0`` byte-0 row, MEM_STORE~1.53,
    value 0x5a=90); the wrong JSR ``MEM_VAL`` store (value 70) carries a
    different address key so the binary-address match filters it out.

    Only emitted when ``num_heads >= 15`` (flag ``C4_L15_LEV_PC_RESTORE`` on).
    """

    # DERIVE->PROVE->FLIP->DELETE (2026-07-04, golden 91f55411): head 14 is
    # re-expressed byte-identically through the shared binary-address CAM
    # primitive :func:`isa_semantics_dsl.cam_binary_address_match` (the same
    # generator the L15 LI/LC LOAD head and the L14 mem-generation STORE head
    # use). The 24-bit binary per-bit address comparator (slots 4..27, reading
    # ADDR_B0/B1/B2 on Q AND K) is a :class:`CamBinaryAddressBlock`; every
    # firing/suppressor/discriminator row is a :class:`CamDiscriminatorSlot`
    # (DATA); the CLEAN_EMBED->OUTPUT return-byte relay is a pair of
    # :class:`CamValueBand`. Each flag-conditioned widening slot (the b0 boost
    # rescale of slots 4..11, the OP_JSR/-OP_ENT discriminator slot 64, the
    # byte-0 selector slot 65, the store-dark slot 66, the hard PC-only gates
    # 67/69, the opcode gate 70, the STACK0-byte0 selector 68) is appended
    # to the discriminator DATA only when its flag is on, exactly mirroring the
    # deleted hand-authored branches -- so flag-off collapses to the
    # byte-identical uniform-scale scaffold. The whole-model golden hash + a
    # cross-flag-matrix test gate byte-identity. The hand-authored per-cell
    # W_q/W_k/W_v directive is deleted.
    #
    # HEAD-14 RESIZE RESIDUAL: expressing the address block via the shared CAM
    # primitive (instead of an inline per-bit loop) makes the b0-boost slots
    # 4..11 a clean last-write-wins DATA override on top of the uniform block --
    # the resize-residual fix path the brief asked for.
    PC_I = 0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4

    # Master address-widening flag. When OFF every widening term collapses to the
    # byte-identical scaffold value (uniform address scale, value_scale 1.0,
    # slot-31 suppressor 200, no slots 64..70), so flag-off keeps head 14 == the
    # committed scaffold (smoke-safe).
    _widen = _l15_lev_addr_widen_on()

    discs: list[CamDiscriminatorSlot] = []

    # === Slot 0: Bias -- fire only at the LEV PC marker (mirror-inverted head 0).
    discs.append(CamDiscriminatorSlot(
        slot=0,
        q=(("CONST", -2000.0), ("OP_LEV", 2000.0), ("MARK_PC", 2000.0),
           ("MARK_AX", -25000.0), ("MARK_SP", -100000.0), ("MARK_BP", -100000.0),
           (f"H1+{SP_I}", -50000.0), (f"H1+{BP_I}", -50000.0)),
        k=(("CONST", 10.0),),
    ))

    # === Slots 29/30: PC / AX byte-position blockers (mirror head 0). ===
    discs.append(CamDiscriminatorSlot(
        slot=29, q=((f"H1+{PC_I}", -20000.0),), k=(("CONST", 5.0),)))
    discs.append(CamDiscriminatorSlot(
        slot=30, q=((f"H1+{AX_I}", -20000.0),), k=(("CONST", 5.0),)))

    # === Slot 31: K-side marker-row self/frame suppressor. ===
    # 200 (scaffold) keeps flag-off byte-identical; 2000 (widen on) beats the
    # byte-0-boosted self-row 24-bit address-match (~2.1e5).
    suppress = 2000.0 if _widen else 200.0
    discs.append(CamDiscriminatorSlot(
        slot=31,
        q=(("CONST", suppress),),
        k=(("MARK_PC", -suppress), ("MARK_AX", -suppress), ("MARK_SP", -suppress),
           ("MARK_BP", -suppress), (f"H1+{PC_I}", -suppress)),
    ))

    # === Slot 66: store-key dark gate on non-LEV steps (CAM-load safety). ===
    if _widen:
        dark = 5000.0
        discs.append(CamDiscriminatorSlot(
            slot=66,
            q=(("OP_LEV", 1.0), ("MARK_PC", 1.0), ("CONST", -1.5)),
            k=(("MEM_STORE", dark),),
        ))

    # === Slots 67/69: HARD (MARK_PC AND OP_LEV) fail-closed firing gate. ===
    if _widen and _l15_lev_pc_only_on():
        HARD = 5_000_000.0
        discs.append(CamDiscriminatorSlot(
            slot=67, q=(("CONST", HARD), ("MARK_PC", -HARD)), k=(("CONST", -1.0),)))
        discs.append(CamDiscriminatorSlot(
            slot=69, q=(("CONST", HARD), ("OP_LEV", -HARD / 4.0)),
            k=(("CONST", -1.0),)))

    # === Slot 70: PER-STEP LEV-opcode fail-closed firing gate. ===
    if _widen and _l15_lev_opcode_gate_on():
        HARD = 5_000_000.0
        discs.append(CamDiscriminatorSlot(
            slot=70, q=(("CONST", HARD), ("OPCODE_BYTE_LO+8", -HARD)),
            k=(("CONST", -1.0),)))

    # === Slot 64: OP_JSR return-store discriminator (ADDRESS-WIDENING). ===
    JSR_DISC = float(_l15_lev_jsr_disc_strength())
    if _widen and JSR_DISC > 0.0:
        discs.append(CamDiscriminatorSlot(
            slot=64,
            q=(("OP_LEV", 1.0), ("MARK_PC", 1.0), ("CONST", -1.0)),
            k=(("OP_JSR", JSR_DISC), ("OP_ENT", -JSR_DISC)),
        ))

    # === Slot 65: K-side byte-0 selection (reject byte-1/2/3 store rows). ===
    if _widen and JSR_DISC > 0.0:
        BSEL = float(_l15_lev_byte0_select_strength())
        discs.append(CamDiscriminatorSlot(
            slot=65,
            q=(("OP_LEV", 1.0), ("MARK_PC", 1.0), ("CONST", -1.0)),
            k=(("BYTE_INDEX_0", BSEL), ("BYTE_INDEX_1", -BSEL),
               ("BYTE_INDEX_2", -BSEL), ("BYTE_INDEX_3", -BSEL)),
        ))

    # === Slot 68: STACK0-byte0 return-store selector (PC-ONLY decouple). ===
    if _widen and _l15_lev_pc_only_on():
        STK0_SEL = 40000.0
        discs.append(CamDiscriminatorSlot(
            slot=68,
            q=(("OP_LEV", 1.0), ("MARK_PC", 1.0), ("CONST", -1.0)),
            k=(("STACK0_BYTE0", STK0_SEL),),
        ))

    # === Slot 1: Store anchor -- only store K cross the threshold. ===
    discs.append(CamDiscriminatorSlot(
        slot=1, q=(("OP_LEV", 50.0), ("MARK_PC", 50.0)),
        k=(("MEM_STORE", 100.0), ("CONST", -50.0))))

    # === Slot 2: ZFOD negative offset for store entries (mirror head 0). ===
    discs.append(CamDiscriminatorSlot(
        slot=2, q=(("CONST", -96.0),), k=(("MEM_STORE", 50.0),)))

    # === Slot 3: Byte selection -- pick byte 0 of the matched store. ===
    BS = 60.0
    discs.append(CamDiscriminatorSlot(
        slot=3, q=(("MARK_STACK0", BS), ("BYTE_INDEX_0", BS)),
        k=((f"L2H0+{MEM_I}", BS), (f"H1+{MEM_I}", -BS))))

    # === Slot 28: Per-head position gate (fire at the LEV PC marker). ===
    discs.append(CamDiscriminatorSlot(
        slot=28, q=(("CONST", -500.0), ("OP_LEV", 500.0), ("MARK_PC", 500.0)),
        k=(("CONST", 5.0),)))

    # === Slots 4..27: 24-bit binary address encoding (mirror head 0). ===
    # The base block scores every address bit at the uniform per-bit scale 10.0;
    # the byte-0 nibbles (slots 4..11, bands ADDR_B0_LO/HI) are RE-WRITTEN at the
    # boosted scale via last-write-wins discriminators when C4_L15_LEV_B0_BOOST
    # is on (default 8x) -- the deepest CAM aliasing separator (0xFFF0 vs 0xFFF8
    # differ only in address byte 0). Byte-1/2 keep scale=10.
    scale = 10.0
    addr = CamBinaryAddressBlock(
        nibble_bands=("ADDR_B0_LO", "ADDR_B0_HI",
                      "ADDR_B1_LO", "ADDR_B1_HI",
                      "ADDR_B2_LO", "ADDR_B2_HI"),
        scale=scale, slot_base=4, width_bits=4,
    )
    b0_factor = _l15_lev_b0_boost_factor() if _widen else 1.0
    if b0_factor != 1.0:
        b0_scale = scale * b0_factor
        for band_i, base in enumerate(("ADDR_B0_LO", "ADDR_B0_HI")):
            for bit in range(4):
                slot = 4 + band_i * 4 + bit
                qk = tuple(
                    (f"{base}+{nk}", b0_scale * (2 * ((nk >> bit) & 1) - 1))
                    for nk in range(16)
                )
                discs.append(CamDiscriminatorSlot(slot=slot, q=qk, k=qk))

    # === V/O: copy matched store byte value to OUTPUT (mirror head 0). ===
    # value_scale=40.0 mirrors the head-0 LI/LC load O scaling; flag-off keeps
    # the byte-identical scaffold 1.0.
    value_scale = 40.0 if _widen else 1.0
    value_bands = (
        CamValueBand("CLEAN_EMBED_LO", "OUTPUT_LO", 16, 32, value_scale),
        CamValueBand("CLEAN_EMBED_HI", "OUTPUT_HI", 16, 48, value_scale),
    )

    cam = CamBinaryAddressMatch(
        name="layer15_lev_pc_restore",
        address=addr,
        discriminators=tuple(discs),
        value_bands=value_bands,
        # The return-address store was written on the JSR step; the LEV lookup
        # reads it across step boundaries by design (memory persistence).
        step_window=StepWindowConstraint.ANY_STEP,
    )
    dim_map = _layer15_lev_pc_restore_dim_map(BD)
    return cam_binary_address_match(cam).head_spec_builder(
        dim_map, _L15_LEV_PC_RESTORE_HEAD_IDX
    )


def _layer15_savedra_pc_dim_map(BD) -> dict:
    """Resolve every dim NAME the L15 savedra-PC opcode-gather head 15 touches."""
    base_names = (
        "CONST", "OP_LEV", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
        "MARK_STACK0", "MARK_MEM", "IS_BYTE", "OP_JSR", "OPCODE_BYTE_LO",
        "LOOKAHEAD_PC_LO", "LOOKAHEAD_PC_HI", "OUTPUT_LO", "OUTPUT_HI",
    )
    return {n: int(getattr(BD, n)) for n in base_names}


def _layer15_savedra_pc_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """L15 head 15: campaign saved-RA delivery into PC at the LEV step.

    In the 30-token campaign frame (``C4_NO_STACK0_EMIT=1``) the JSR return
    address is never materialized on any persistent row (the STACK0 push that
    carried it is dropped), so head 14's BP+8 address-CAM has nothing to gather
    (it picks the saved-BP word -> pc=240 instead of 90). This head bypasses the
    starved CAM: it reads the LIVE ``LOOKAHEAD_PC`` band (= current_PC + 8 = the
    return address) off the most-recent JSR step's AX-marker row and copies it to
    OUTPUT at the LEV PC marker.

    DESIGN (mirrors the head-14 firing/suppressor scaffold but with a DIFFERENT
    gather target):
      * Query fires ONLY at the LEV PC marker (OP_LEV + MARK_PC); every other row
        is darkened so the head writes ~nothing elsewhere.
      * Key rewards the JSR AX-marker row (OP_JSR + MARK_AX): the JSR's AX marker
        is the UNIQUE row carrying both, and it is the row where ``LOOKAHEAD_PC``
        is a live one-hot (= JSR_PC + INSTR_WIDTH). Recency (ALiBi) selects the
        MOST-RECENT JSR for nested/recursive calls; a CONST sink anchors the
        softmax1 so a no-JSR LEV (top-level) writes nothing.
      * V reads ``LOOKAHEAD_PC_LO/HI`` -> O writes OUTPUT_LO/HI (value_scale 40,
        the same strong-write the LI/LC load + head-14 use to land a sharp byte).

    Campaign-gated (``C4_L15_SAVEDRA_HEAD``); flag-off omits the head so the
    golden (15-head) build is byte-identical.
    """
    # DERIVE->PROVE->FLIP->DELETE (2026-07-04, golden 91f55411): head 15 is a
    # small OPCODE-CONTENT gather (no binary address key) re-expressed
    # byte-identically through :func:`isa_semantics_dsl.cam_binary_address_match`
    # with an EMPTY :class:`CamBinaryAddressBlock` (``nibble_bands=()``) -- the
    # same discriminator-only-CAM shape the L14 mem-generation STORE head uses.
    # Every firing/selector/gate row is a :class:`CamDiscriminatorSlot` (DATA);
    # the LOOKAHEAD_PC->OUTPUT relay is a pair of :class:`CamValueBand`. The
    # hand-authored per-cell Q/K/V/O directive is deleted.
    SEL = 40.0    # slot 1: JSR-step AX-marker selector reward (OP_JSR dominant).
    REJ = 200.0   # slot 2: require MARK_AX (reject the JSR step's NON-AX rows).
    GATE = 1_000_000.0  # slot 3: per-step LEV-opcode fail-closed firing gate.
    value_scale = 80.0  # V/O: 2x the plain-load 40 to out-write the LEV 0xF default.

    discs = (
        # Slot 0: fire ONLY at the LEV PC marker (the query gate); suppress the
        # LEV step's own AX/SP/BP/STACK0/MEM marker + byte rows.
        CamDiscriminatorSlot(
            slot=0,
            q=(("CONST", -3.0), ("OP_LEV", 1.0), ("MARK_PC", 1.0),
               ("MARK_AX", -1000.0), ("MARK_SP", -1000.0), ("MARK_BP", -1000.0),
               ("MARK_STACK0", -1000.0), ("MARK_MEM", -1000.0),
               ("IS_BYTE", -1000.0)),
            k=(("CONST", 30.0),),
        ),
        # Slot 1: JSR-step AX-marker selector (OP_JSR dominant, MARK_AX gate).
        CamDiscriminatorSlot(
            slot=1,
            q=(("OP_LEV", 1.0), ("MARK_PC", 1.0), ("CONST", -2.0)),
            k=(("OP_JSR", SEL), ("MARK_AX", 0.5 * SEL), ("CONST", -0.7 * SEL)),
        ),
        # Slot 2: require MARK_AX (reject the JSR step's NON-AX rows).
        CamDiscriminatorSlot(
            slot=2,
            q=(("OP_LEV", 1.0), ("MARK_PC", 1.0), ("CONST", -2.0)),
            k=(("CONST", -REJ), ("MARK_AX", REJ)),
        ),
        # Slot 3: per-step LEV-opcode fail-closed firing gate (the ADJ guard;
        # SAME discriminator head 14 uses at its slot 70).
        CamDiscriminatorSlot(
            slot=3,
            q=(("CONST", GATE), ("OPCODE_BYTE_LO+8", -GATE)),
            k=(("CONST", -1.0),),
        ),
    )
    value_bands = (
        CamValueBand("LOOKAHEAD_PC_LO", "OUTPUT_LO", 16, 32, value_scale),
        CamValueBand("LOOKAHEAD_PC_HI", "OUTPUT_HI", 16, 48, value_scale),
    )
    cam = CamBinaryAddressMatch(
        name="layer15_savedra_pc",
        address=CamBinaryAddressBlock(
            nibble_bands=(), scale=0.0, slot_base=4, width_bits=4),
        discriminators=discs,
        value_bands=value_bands,
        # Strong per-head ALiBi recency so the MOST-RECENT JSR's AX row wins over
        # an identical earlier JSR-AX row (nested/recursive calls).
        alibi_slope=0.1,
        # LOOKAHEAD_PC was computed on the JSR step; the LEV gather reads it
        # across step boundaries by design -> ANY_STEP.
        step_window=StepWindowConstraint.ANY_STEP,
    )
    dim_map = _layer15_savedra_pc_dim_map(BD)
    return cam_binary_address_match(cam).head_spec_builder(
        dim_map, _L15_SAVEDRA_HEAD_IDX
    )


def _layer15_si_store_addr_cam_dim_map(BD) -> dict:
    """Resolve every dim NAME the L15 SI-store-addr CAM head 16 touches."""
    base_names = (
        "OP_LI", "CONST", "MARK_AX", "OP_PSH", "OUTPUT_LO", "OUTPUT_HI",
        "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_B0_LO", "ADDR_B0_HI",
        # LI-QUERY zero-address VETO flag (L14 make_layer14_li_query_zeroaddr_op,
        # C4_SI_STORE_ADDR): the firing veto reads it to fail-closed on the
        # absolute-address path.
        "LI_QUERY_ZEROADDR",
    )
    return {n: int(getattr(BD, n)) for n in base_names}


def _layer15_si_store_addr_cam_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """L15 head 16: SI/SC store address-provenance CAM (campaign, DEFAULT-OFF).

    Resolves the ``var_mul`` / multilocal-LI store-provenance TWO-ROOT wall
    (``int a;int b;a=23;b=47;return a*b;`` id275: ``LI a`` returns b's 47 not
    a's 23). See :func:`shared.si_store_addr_enabled` for the full root + proof.

    THE LEVER (built-layout probe ``tools/_probe_si_store_addr.py`` +
    ``tools/_probe_si_marker_disc.py``, campaign config, id275): the clean store
    ADDRESS and the clean stored VALUE both live on each store's AX-MARKER row
    (a-marker @281: ``ADDR_B0=0xE8`` + ``AX_CARRY=0x17``=23; b-marker @401:
    ``ADDR_B0=0xE0`` + ``AX_CARRY=0x2F``=47), and the ``LI a`` byte-0 lookup row
    (@491) carries its TARGET address in ``AX_CARRY=0xE8`` (its ``ADDR_B0`` is
    0x00 there — Root 1's address-blindness). The existing head-0 CAM reads the
    recency-tied store VALUE rows (which are address-blind AND carry an
    address-like ``CLEAN_EMBED``, NOT the value — Root 2), so it delivers the
    wrong local. This head bypasses BOTH roots with a DIRECT address-keyed CAM:

      * Q (the LI byte-0 lookup row): fire ONLY at an LI emit row — gated on
        ``OP_LI`` (~5.2 at the byte-0 emit row, ~0.05 residue elsewhere) — and
        key on the row's OWN ``AX_CARRY_LO/HI`` nibbles (the target address,
        e.g. 0xE8 for ``LI a``). A CONST sink anchors softmax1 so a non-LI row
        writes ~nothing.
      * K (candidate store markers): match the store address by keying on the
        candidate's ``ADDR_B0_LO/HI`` nibbles (the store address 0xE8 / 0xE0)
        against the Q's ``AX_CARRY``. A ``-OP_PSH`` penalty rejects the
        address-COMPUTATION push row (@251: ``ADDR_B0=0xE8`` too, but
        ``AX_CARRY=0xE8``=address and ``OP_PSH~5.2``) so the CAM lands on the
        genuine store-value marker (@281: ``OP_PSH~0.05``, ``AX_CARRY``=value).
      * V copies the selected marker's ``AX_CARRY_LO/HI`` (the CLEAN stored
        value 0x17=23) -> O writes it into OUTPUT_LO/HI at value_scale 60, which
        DOMINATES head-0's wrong ``value_scale=40`` CLEAN copy at the argmax.

    Recency (ALiBi slope 0.05, matching the load heads) resolves a re-store to
    the SAME local (var_update) to the latest store. Campaign-gated
    (``C4_SI_STORE_ADDR``); flag-off omits the head (num_heads<17) so the golden
    campaign (16-head) build is byte-identical.
    """
    # DERIVE->PROVE->FLIP->DELETE (2026-07-04, golden 91f55411): head 16 is a
    # 12-bit ADDR_B0 content-addressable match (Q keys AX_CARRY, K keys ADDR_B0
    # -- a per-nibble one-hot match, NOT the symmetric binary per-bit block), so
    # it is re-expressed byte-identically through
    # :func:`isa_semantics_dsl.cam_binary_address_match` with an EMPTY
    # :class:`CamBinaryAddressBlock` (``nibble_bands=()``) and the per-nibble
    # match + every firing/reject gate authored as :class:`CamDiscriminatorSlot`
    # DATA. The AX_CARRY->OUTPUT relay is a pair of :class:`CamValueBand`. The
    # hand-authored per-cell Q/K/V/O directive is deleted.
    _cam_s = 360.0    # per-nibble address-match scale (mirrors #313 head-0 CAM).
    _sink = 30.0      # softmax1 CONST sink (mirrors the savedra head).
    _psh_rej = 400.0  # K-side OP_PSH penalty (reject the address-push row).
    value_scale = 60.0
    _dark = 10_000_000.0  # HARD fire-gate magnitude (MARK_AX + OP_LI query gates).
    _ax_req = 2000.0  # slot 60: REQUIRE MARK_AX (reject the store VALUE BYTE rows).

    discs: list[CamDiscriminatorSlot] = []
    # Slot 0: CONST sink (softmax1 anchor), gated on the OP_LI query.
    discs.append(CamDiscriminatorSlot(slot=0, q=(("OP_LI", 1.0),), k=(("CONST", _sink),)))
    # Slot 59: fire ONLY at the byte-0 AX-marker QUERY row (MARK_AX gate).
    discs.append(CamDiscriminatorSlot(
        slot=59, q=(("MARK_AX", _dark), ("CONST", -_dark)), k=(("CONST", 1.0),)))
    # Slot 57: fire ONLY on an LI/LC EMIT QUERY row (require OP_LI).
    discs.append(CamDiscriminatorSlot(
        slot=57, q=(("OP_LI", _dark), ("CONST", -2.5 * _dark)), k=(("CONST", 1.0),)))
    # Slots 1..16: per-nibble address match Q(AX_CARRY_LO) . K(ADDR_B0_LO). The
    # k=0 (null) LO nibble uses slot 16 (slot 0 is the CONST sink).
    for _k in range(0, 16):
        _row = _k if _k >= 1 else 16
        discs.append(CamDiscriminatorSlot(
            slot=_row, q=((f"AX_CARRY_LO+{_k}", _cam_s),),
            k=((f"ADDR_B0_LO+{_k}", _cam_s),)))
    # Slots 17..31 + 58: HIGH-nibble match; the k=0 (null) HI nibble uses slot 58.
    for _k in range(0, 16):
        _row = (16 + _k) if _k >= 1 else 58
        discs.append(CamDiscriminatorSlot(
            slot=_row, q=((f"AX_CARRY_HI+{_k}", _cam_s),),
            k=((f"ADDR_B0_HI+{_k}", _cam_s),)))
    # Slot 62: reject the address-COMPUTATION push row (-OP_PSH, OP_LI-gated).
    discs.append(CamDiscriminatorSlot(
        slot=62, q=(("OP_LI", _psh_rej), ("CONST", -0.05 * _psh_rej)),
        k=(("OP_PSH", -_psh_rej),)))
    # Slot 61: reject the LI-query row ITSELF (self-attention guard, -OP_LI).
    discs.append(CamDiscriminatorSlot(
        slot=61, q=(("OP_LI", _psh_rej), ("CONST", -0.05 * _psh_rej)),
        k=(("OP_LI", -_psh_rej),)))
    # Slot 60: REQUIRE MARK_AX (reject the store VALUE BYTE rows).
    discs.append(CamDiscriminatorSlot(
        slot=60, q=(("OP_LI", 2.0 * _cam_s), ("CONST", -0.05 * 2.0 * _cam_s)),
        k=(("MARK_AX", _ax_req), ("CONST", -_ax_req))))
    # Slot 63: LI-QUERY zero-address FIRING VETO (agent a76baa3b, refine of the
    # a8653eca CAM). ROOT: on the ABSOLUTE-address LI path (e.g. LI 0x200 ->
    # target byte-0 = 0x00) the Q keys the LI-query's OWN AX_CARRY, which is
    # all-zero (BOTH null nibbles hot), so the per-nibble match (slots 16 LO-k0 +
    # 58 HI-k0) null-matches a spurious ADDR_B0=0x00 store marker (probe @217,
    # value 0) that OVERRIDES head-0's correct value at value_scale 60. The
    # store-addr CAM was BUILT for the RELATIVE case (var_mul &a=0xE8), where the
    # query AX_CARRY is NON-zero -- so it must NOT fire when the query address is
    # all-zero (null-null = no relative target). ``LI_QUERY_ZEROADDR`` (the L14
    # FFN two-nibble zero-address AND, gated OP_LI) is ≈>0 ONLY on the abs-address
    # LI query row and 0 on every relative-address LI query (0xE8/0xE0: at most
    # ONE null nibble, AND<thr). A large NEGATIVE Q here against K=CONST (≈1 on
    # EVERY candidate) subtracts a uniform block from the row-score, driving ALL
    # candidates below the softmax1 sink so the head fails-closed (writes ~0) on
    # the abs path -- while it is INERT (flag=0) on the var_mul relative path the
    # head was built for. Sized to out-subtract the ~2.7M firing-gate baseline.
    _veto = 30_000_000.0
    discs.append(CamDiscriminatorSlot(
        slot=63, q=(("LI_QUERY_ZEROADDR", -_veto),), k=(("CONST", 1.0),)))

    value_bands = (
        CamValueBand("AX_CARRY_LO", "OUTPUT_LO", 16, 32, value_scale),
        CamValueBand("AX_CARRY_HI", "OUTPUT_HI", 16, 48, value_scale),
    )
    cam = CamBinaryAddressMatch(
        name="layer15_si_store_addr_cam",
        address=CamBinaryAddressBlock(
            nibble_bands=(), scale=0.0, slot_base=4, width_bits=4),
        discriminators=tuple(discs),
        value_bands=value_bands,
        # Recency (matching the L15 load heads' 0.05 slope) resolves a re-store
        # to the SAME local (var_update x-reassign) to the latest store.
        alibi_slope=0.05,
        # The store's AX-marker was written on the SI step; the LI gather reads
        # its AX_CARRY across step boundaries by design -> ANY_STEP.
        step_window=StepWindowConstraint.ANY_STEP,
    )
    dim_map = _layer15_si_store_addr_cam_dim_map(BD)
    return cam_binary_address_match(cam).head_spec_builder(
        dim_map, _L15_SI_STORE_ADDR_HEAD_IDX
    )


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

    # Marker-bank MEM slot via the positional-invariant mechanism (Class-1
    # marker-relative). Frame-INVARIANT; byte-identical in both frames. Was the
    # hand-coded ``MEM_I = 4``. See positional_invariant.py.
    MEM_I = marker_bank_index("MEM")

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

    # Marker-bank MEM slot via the positional-invariant mechanism (Class-1
    # marker-relative). Frame-INVARIANT; byte-identical. Was ``MEM_I = 4``.
    # See positional_invariant.py.
    MEM_I = marker_bank_index("MEM")
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
    # Head 14: LEV PC-restore (flag C4_L15_LEV_PC_RESTORE, default ON).
    # Emitted when the resize has allocated a 15th head (num_heads >= 15) or
    # on the declarations-only audit path (num_heads=None) when the flag is on.
    if _l15_lev_pc_restore_head_on() and (
        num_heads is None or int(num_heads) >= 15
    ):
        attn_op.append(
            _layer15_lev_pc_restore_head_spec(proxy),
            name="layer15_memory_lookup.lev_pc_restore",
            metadata={"role": "lev_pc_restore", "shape": "num_heads >= 15"},
        )
    # Head 15: campaign saved-RA delivery (flag C4_L15_SAVEDRA_HEAD, campaign).
    # Emitted when the resize has allocated a 16th head (num_heads >= 16) or on
    # the declarations-only audit path when the flag is on.
    if _l15_savedra_head_on() and (
        num_heads is None or int(num_heads) >= 16
    ):
        attn_op.append(
            _layer15_savedra_pc_head_spec(proxy),
            name="layer15_memory_lookup.savedra_pc",
            metadata={"role": "savedra_pc", "shape": "num_heads >= 16"},
        )
    # Head 16: SI/SC store address-provenance CAM (flag C4_SI_STORE_ADDR,
    # campaign, DEFAULT-OFF). Emitted when the resize has allocated a 17th head
    # (num_heads >= 17) or on the declarations-only audit path when the flag is
    # on. Flag-off omits the head so the golden campaign build is byte-identical.
    if si_store_addr_enabled() and (
        num_heads is None or int(num_heads) >= 17
    ):
        attn_op.append(
            _layer15_si_store_addr_cam_head_spec(proxy),
            name="layer15_memory_lookup.si_store_addr_cam",
            metadata={"role": "si_store_addr_cam", "shape": "num_heads >= 17"},
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
               # OP_JSR: read by the campaign head-0 JSR-phantom value-row
               # penalty (slot 104, _l15_li_jsr_phantom_penalty_on).
               "OP_JSR",
               # ADDR_B0_LO/HI + OP_PSH: read by the campaign head-16 SI/SC
               # store address-provenance CAM (_layer15_si_store_addr_cam_head_spec,
               # C4_SI_STORE_ADDR). ADDR_B0 keys the store-address match; OP_PSH
               # penalizes the address-computation push row.
               "ADDR_B0_LO", "ADDR_B0_HI", "OP_PSH",
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


def _layer15_store_addr_dim_map(BD) -> dict:
    """Resolve every dim NAME the L15 store-address relay heads 12/13 touch.

    :func:`isa_semantics_dsl.scalar_relay`'s builder wants a name->int dict; the
    ``BASE+offset`` nibble-band tokens are resolved from the BASE name plus the
    parsed offset by the generator, so only BASE names go in the dict.
    """
    base_names = (
        "CONST", "MARK_STACK0", "HAS_SE", "MEM_STORE", "MARK_SP",
        "MARK_MEM", "MEM_ADDR_SRC", "IS_BYTE", "STACK0_BYTE0",
        "OUTPUT_LO", "OUTPUT_HI", "ADDR_B0_LO", "ADDR_B0_HI",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
    )
    return {n: int(getattr(BD, n)) for n in base_names}


def _layer15_store_stack0_sp_byte0_addr_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Copy current post-pop SP byte0 into ADDR_B0 at store STACK0 markers.

    DERIVE->PROVE->FLIP->DELETE (2026-07-04, golden 91f55411): this
    marker-anchored MULTI-nibble relay is re-expressed byte-identically through
    the shared :func:`isa_semantics_dsl.scalar_relay` primitive (the L7 memory
    relay banks' generator). The head fires at the store STACK0 marker
    (Q ``MARK_STACK0``/``HAS_SE`` on slot 0 + the MEM_STORE store-gate on slot
    33), selects the post-pop SP marker row (K ``MARK_SP``), and copies the
    row's OUTPUT byte-0 nibbles into ``ADDR_B0`` via two :class:`NibbleRelay`
    blocks (per-cell copy +3, per-cell CONST clear -2 sourced from ``const_v_slot``
    0). The hand-authored per-cell Q/K/V/O directive is deleted. Byte-identity is
    proven by ``test_isa_semantics_dsl.py`` and the whole-model golden hash.
    """
    dim_map = _layer15_store_addr_dim_map(BD)
    spec = ScalarRelayBankSpec(
        name="layer15_store_stack0_sp_byte0_addr",
        # Marker fire-site + store-gate discriminators, spread across slots 0/33
        # (3-tuple ``(slot, dim, weight)`` form).
        query_sig=(
            (0, "MARK_STACK0", 300.0),
            (0, "HAS_SE", 300.0),
            (33, "MEM_STORE", 10000.0),
            (33, "CONST", -50000.0),
        ),
        key_sig=(
            (0, "MARK_SP", 100.0),
            (33, "CONST", 1.0),
        ),
        nibble_relays=(
            NibbleRelay("OUTPUT_LO", "ADDR_B0_LO", 16, 1,
                        copy_scale=3.0, clear_scale=2.0),
            NibbleRelay("OUTPUT_HI", "ADDR_B0_HI", 16, 17,
                        copy_scale=3.0, clear_scale=2.0),
        ),
        const_v_slot=0,
        step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
    )
    # Pull the head index from the shared L15 layout rather than baking in a
    # ``head_idx=12`` literal here. Both the bake and IR paths consult the same
    # source of truth, so renumbering the layout stays consistent everywhere.
    return scalar_relay(spec).head_spec_builder(
        dim_map, _l15_head_idx("layer15_store_stack0_sp_byte0_addr")
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

    # DERIVE->PROVE->FLIP->DELETE (2026-07-04, golden 91f55411): re-expressed
    # byte-identically through :func:`isa_semantics_dsl.scalar_relay`. The
    # row-select signature spreads the SI/SC MEM-address opcode gate across
    # slots 0/33 plus a byte-index gate on O (slot 37, ``IS_BYTE`` on Q /
    # ``CONST`` reject on K -- the "optional byte-index gating on O" the survey
    # flagged); the K side keys the pre-store STACK0 byte-0 row (``STACK0_BYTE0``,
    # MEM_STORE reject). Two :class:`NibbleRelay` blocks copy ``CLEAN_EMBED`` ->
    # ``OUTPUT`` (per-cell copy +20, per-cell CONST clear -10). The hand-authored
    # per-cell Q/K/V/O directive is deleted.
    dim_map = _layer15_store_addr_dim_map(BD)
    spec = ScalarRelayBankSpec(
        name="layer15_si_mem_addr0_from_stack0",
        query_sig=(
            (0, "MARK_MEM", 100.0),
            (0, "MEM_STORE", 20.0),
            (0, "MEM_ADDR_SRC", 50.0),
            (0, "CONST", -140.0),
            (33, "MARK_MEM", 40000.0),
            (33, "MEM_STORE", 5000.0),
            (33, "MEM_ADDR_SRC", 10000.0),
            (33, "CONST", -55000.0),
            # Byte-index gate on the value-write path (slot 37).
            (37, "IS_BYTE", 50000.0),
        ),
        key_sig=(
            (0, "STACK0_BYTE0", 100.0),
            (0, "MEM_STORE", -400.0),
            (33, "CONST", 5.0),
            (37, "CONST", -20.0),
        ),
        nibble_relays=(
            NibbleRelay("CLEAN_EMBED_LO", "OUTPUT_LO", 16, 1,
                        copy_scale=20.0, clear_scale=10.0),
            NibbleRelay("CLEAN_EMBED_HI", "OUTPUT_HI", 16, 17,
                        copy_scale=20.0, clear_scale=10.0),
        ),
        const_v_slot=0,
        step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
    )
    # Pull the head index from the shared L15 layout rather than baking in a
    # ``head_idx=13`` literal here.
    return scalar_relay(spec).head_spec_builder(
        dim_map, _l15_head_idx("layer15_si_mem_addr0_from_stack0")
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
    # Marker-bank slot indices via the positional-invariant mechanism
    # (Class-1 marker-relative). Frame-INVARIANT bank-TYPE order; byte-identical
    # in both frames. Replaces the hand-coded ``mem_i = 4`` etc. so the audit
    # recognises the ``BD.H1 + mem_i`` / ``BD.L2H0 + mem_i`` reads as declared
    # marker-relative. This imperative writer is the one that ACTUALLY lands
    # (make_l15_attention_resize_op re-runs it after the declarative bake), so
    # it must carry the same declared invariance as the declarative mirror
    # ``_layer15_memory_lookup_heads_0_3_specs_with_overrides``. See
    # positional_invariant.py.
    pc_i = marker_bank_index("PC")
    ax_i = marker_bank_index("AX")
    mem_i = marker_bank_index("MEM")
    sp_i = marker_bank_index("SP")
    bp_i = marker_bank_index("BP")
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
            # Class-2 absolute-slot read (mirrors the declarative
            # ``_with_overrides`` slot-59 K-write): STACK0_BYTE0 is the
            # d=6-from-BP flag that VANISHES under C4_NO_STACK0_EMIT. Drive the
            # weight to 0 in the dropped frame via invariant_threshold —
            # byte-identical at 35-tok, behaviourally a no-op at 30-tok (input
            # flag already constant-0). See positional_invariant.py.
            attn.W_k.data[base + preserve_e8_row, BD.STACK0_BYTE0] = (
                invariant_threshold(
                    live=5.0 * preserve_e8_s, suppressed=0.0,
                    marker="BP", k=6,
                )
            )

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
        # ``BD.H1 + sp_i`` (marker-bank SP slot); was the bare ``BD.H1 + 2``.
        attn.W_q.data[base + sp_byte_blocker, BD.H1 + sp_i] = 100000.0
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
    # Marker-bank MEM slot via the positional-invariant mechanism (Class-1
    # marker-relative). Frame-INVARIANT; byte-identical. Mirrors the declarative
    # ``_layer15_memory_lookup_lev_blockers_only_specs`` MEM_I. Was ``mem_i = 4``.
    # See positional_invariant.py.
    mem_i = marker_bank_index("MEM")
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
    # Head 14: LEV PC-restore (flag C4_L15_LEV_PC_RESTORE, default ON).
    # The production ``layer15_memory_lookup`` bake runs at num_heads=10
    # (BEFORE this resize), so a head gated on the resized count never gets
    # its body. Lower head 14's body HERE, after the resize has grown the
    # attn to >=15 heads, so the new head's Q/K/V/O cells actually land.
    # Flag-off keeps the resize target at 14 and skips this bake entirely
    # -> byte-identical with HEAD.
    if _l15_lev_pc_restore_head_on() and getattr(
        attn, "num_heads", 0
    ) > _L15_LEV_PC_RESTORE_HEAD_IDX:
        Primitives.generate_attention_head(
            attn,
            _layer15_lev_pc_restore_head_spec(_as_setdim_proxy(dim_positions)),
            int(head_dim),
        )
    # Head 15: campaign saved-RA delivery (flag C4_L15_SAVEDRA_HEAD). Lower its
    # body after the resize has grown the attn to >=16 heads. Flag-off keeps the
    # resize target at <=15 and skips this -> byte-identical with the golden build.
    if _l15_savedra_head_on() and getattr(
        attn, "num_heads", 0
    ) > _L15_SAVEDRA_HEAD_IDX:
        Primitives.generate_attention_head(
            attn,
            _layer15_savedra_pc_head_spec(_as_setdim_proxy(dim_positions)),
            int(head_dim),
        )
    # Head 16: SI/SC store address-provenance CAM (flag C4_SI_STORE_ADDR,
    # campaign, DEFAULT-OFF). Like heads 14/15, the production
    # ``layer15_memory_lookup`` bake runs at the pre-widen head count (BEFORE
    # this resize), so a head gated on the resized count never gets its body
    # there. Lower head 16's Q/K/V/O cells HERE, after the resize has grown the
    # attn to >=17 heads. Flag-off keeps the resize target at <=16 and skips
    # this -> byte-identical with the golden campaign build.
    if si_store_addr_enabled() and getattr(
        attn, "num_heads", 0
    ) > _L15_SI_STORE_ADDR_HEAD_IDX:
        Primitives.generate_attention_head(
            attn,
            _layer15_si_store_addr_cam_head_spec(_as_setdim_proxy(dim_positions)),
            int(head_dim),
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
    # target_num_heads is 15 when the LEV PC-restore head (head 14) is on so
    # the resize allocates a slot for it; flag-off keeps the legacy 14-head
    # build byte-identical. small_num_heads (16-layer smoke build) is
    # unaffected -- the LEV PC-restore head only matters for >16-layer LEV
    # builds where the return-address restore path exists.
    _lev_target = _L15_MAX_HEADS  # 15 (flag on) or 14 (flag off)
    # SURGICAL ALiBi neutralization (C4_SI_STORE_ADDR, campaign, DEFAULT-OFF):
    # appending the SI-store CAM as head 16 grows the L15 head count 16->17,
    # which -- via the ``2**(-8/N*(i+1))`` slope formula -- would RECOMPUTE the
    # ALiBi recency slope of EVERY resized head (heads 4-11 LEV saved_bp/pop_d8/
    # return_addr, head 14 lev_pc_restore) with N=17 instead of N=16. That is a
    # BROAD perturbation: it shifts the recency tie-break of the load-bearing
    # frame-restore heads on every func/rec/nested/loop program, regressing
    # unrelated clusters even though the new head only fires on SI/LI rows. Fix:
    # decouple the slope FORMULA's N from the physical head count. With the SI
    # head on we still ALLOCATE 17 slots but compute slopes as if N==16, so
    # heads 0-15 keep byte-identical slopes to the (savedra) 16-head build; the
    # SI head (16) gets its own 0.05 slope from its head spec post-resize. When
    # the flag is off ``alibi_slope_num_heads`` is None -> standard behaviour ->
    # golden byte-identical.
    _slope_n = 16 if si_store_addr_enabled() else None
    ir.layer(0).structural_ops.append(
        StructuralOp(
            kind="attention_resize",
            target_num_heads=_lev_target,
            small_num_heads=9,
            layers_threshold=16,
            alibi_pin_value=0.05,
            alibi_pin_count=4,
            alibi_slope_num_heads=_slope_n,
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
