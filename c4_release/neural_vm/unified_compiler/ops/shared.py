"""Shared helpers and constants used by the per-layer op modules.

Extracted from the legacy ``migrated_ops.py`` (2026-05-11) so the per-layer
factory modules can import these without circular dependencies. The public
API is unchanged: ``from c4_release.neural_vm.unified_compiler.migrated_ops
import _as_setdim_proxy, declare_setdim_compat_dims, ...`` continues to work
via the re-export shim in ``migrated_ops.py``.
"""

import os
from typing import Dict
import torch.nn as nn

from ..layer_compiler import Operation


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

def campaign_enabled() -> bool:
    """Return True iff the single CAMPAIGN-CONFIG entry point is set
    (``C4_CAMPAIGN=1``). DEFAULT OFF.

    This is the one flag that turns on the coherent 30-token *campaign* set
    without the caller having to juggle the growing pile of individual campaign
    flags (``C4_NO_STACK0_EMIT`` ``C4_OPERAND_FROM_MEMSP`` ``C4_SI_STORE_ADDR``
    ...). It is wired in by having each campaign-gating predicate OR-in
    ``campaign_enabled()``:

      * ``no_stack0_emit_enabled()``     -- STACK0 register block dropped (30-tok)
      * ``operand_from_memsp_enabled()`` -- operand-A read from ``mem[SP]``
      * ``si_store_addr_enabled()``      -- SI/SC store address-provenance CAM

    So ``C4_CAMPAIGN=1`` ALONE reproduces the full campaign config, and it is
    exactly equivalent to setting each of those explicit flags to its campaign
    value. The two predicates that ALREADY default ON in a bare env
    (``no_stack0_emit`` / ``operand_from_memsp``, both ``!= "0"``) are unchanged
    by the OR-in; the one that defaults OFF (``si_store_addr``)
    is the one ``C4_CAMPAIGN`` flips on.

    An explicit per-flag override still wins for the two default-ON predicates
    (setting ``C4_NO_STACK0_EMIT=0`` opts that one op out even under the
    campaign): the OR-in only supplies a default-ON floor for the normally-OFF
    campaign fixes; the default-ON predicates keep reading their own env first.
    DEFAULT OFF -> the golden (35-tok) build is byte-identical (``C4_CAMPAIGN``
    unset -> every predicate keeps its exact pre-existing value). Registered in
    BOTH cache-key snapshots in ``full_vm_compiler_dynamic.py`` so a campaign
    build never shares a memo / disk cache entry with a golden build.
    """
    return os.environ.get("C4_CAMPAIGN", "0") != "0"


def derive_imm_enabled() -> bool:
    """Flag for the FULLY-DERIVED IMM opcode (task #392). Default OFF.

    When ``C4_DERIVE_IMM=1`` the ENTIRE IMM opcode is derived from its
    declarative spec with ZERO hand-authored IMM rules, closing the last
    LOWERING gap the DERIVE_DECODE pilot named (docs/DERIVE_DECODE_PILOT
    §G-IMM-RELAY):

      * decode -> ``OP_IMM``: routed through the generic :func:`decode_band`
        engine (the same path ``C4_DERIVE_DECODE`` enables) — implied ON by
        ``C4_DERIVE_IMM`` (see ``l5_ops._derive_decode_enabled``).
      * relay ``OP_IMM`` -> AX byte positions: the L8 head-4 relay is
        re-expressed by the new generic :func:`marker_broadcast` head
        generator (``isa_semantics_dsl.py``) instead of the hand-built
        ``_layer8_op_imm_relay_head_spec``.
      * value route imm -> OUTPUT: the 32-rule per-nibble AX_CARRY->OUTPUT
        copy is re-expressed by the generic value-route lowering
        (``isa_semantics_dsl.value_route``) instead of the hand-built
        ``_layer8_multibyte_routing_rules``.

    All three derivations reproduce the hand-authored weights byte-for-byte
    (proof: ``tools/_isa_golden_hash.py`` unchanged under ``C4_DERIVE_IMM=1``),
    so IMM becomes the FIRST 100%-derived opcode. Default OFF => the
    hand-authored path stays the golden build.
    """
    return os.environ.get("C4_DERIVE_IMM", "0") != "0"


def derive_bitwise_enabled() -> bool:
    """Flag for the SPEC-DERIVED BITWISE family (OR/XOR/AND). Default ON.

    BLOG_SPEC §568: *"Bitwise: 10 weights — one formula (a+b-ab) for all"*.
    All three bitwise ops are ONE per-bit polynomial
    ``r_bit = c_a*a_bit + c_b*b_bit + c_ab*(a_bit*b_bit)`` applied across the
    4 nibble bits, with a single per-op coefficient triple read straight from
    the spec text:

      * OR  = ``a + b - a*b``   -> ``(c_a, c_b, c_ab) = ( 1,  1, -1)``
      * AND = ``        a*b``   -> ``( 0,  0,  1)``
      * XOR = ``a + b - 2*a*b`` -> ``( 1,  1, -2)``

    When ``C4_DERIVE_BITWISE=1`` the L10 bitwise result nibbles (lookup post-op
    AND both L10-main-FFN callers) are produced by this ONE shared formula
    (``wide_alu_dsl._bitwise_result_from_spec_formula``) instead of the three
    enumerated ``operator.and_/or_/xor`` bit functions
    (``wide_alu_dsl._BITWISE_OP_FN``). The result nibbles are provably
    identical per ``(a, b)`` pair (each per-bit result is in ``{0, 1}`` by
    construction), so ``tools/_isa_golden_hash.py`` is UNCHANGED under the flag
    — the derivation is a SOURCE collapse (3 distinct bit operators -> 1 spec
    formula + 3 coefficient triples), not a weight change.

    Default ON => the derived one-formula path is the golden build (the
    enumerated ``operator`` dispatch has been deleted; ``C4_DERIVE_BITWISE=0``
    is a legacy no-op kill-switch). Both ON/OFF are registered in the
    ``full_vm_compiler_dynamic.py`` cache-key snapshots so a build never shares
    a memo / disk entry across the flag. See ``docs/DERIVE_BITWISE_2026_07_09.md``.
    """
    return os.environ.get("C4_DERIVE_BITWISE", "1") != "0"


def derive_shift_enabled() -> bool:
    """Flag for the SPEC-DERIVED SHIFT family (SHL/SHR, task #448). Default ON.

    When ``C4_DERIVE_SHIFT=1`` the per-``(value, shift)`` result value that the
    L13 SHL/SHR lookup table stores is COMPUTED from the ``BLOG_SPEC`` "Shifts"
    building blocks (docs/BLOG_SPEC.md §597-599) instead of the hand-authored
    Python bit-shift operators ``(v << s) & 0xFF`` / ``v >> s``:

      * SHL by ``s`` = **multiply by the power of two** ``2**s`` then take the
        **modulus by floor** for 8-bit overflow: ``(v * 2**s) mod 256`` where
        ``x mod m = x - m*floor(x/m)`` (§555 MAGIC floor / §560 bit-range
        extraction — a floor then a mod by a power of two).
      * SHR by ``s`` = **floor-divide by the power of two**: ``floor(v / 2**s)``.

    The powers of two are taken from ONE derived table (``2**s`` for
    ``s in 0..7``) rather than a per-shift hand-typed constant, and the byte
    truncation is the generic mod-by-floor primitive rather than a per-op
    ``& 0xFF`` mask — so the derivation has ZERO per-op magic constants
    (see :mod:`shift_semantics_dsl`).

    The derived formula equals the hand-authored bit-op for EVERY ``(v, s)``
    with ``v in 0..255``, ``s in 0..7`` (proven exhaustively in
    ``shift_semantics_dsl._SPEC_MATCHES_BITOPS`` and by the whole-model golden
    hash held under ``C4_DERIVE_SHIFT=1``), so the L13 shifts FFN (SHL + SHR,
    4096 lookup units) is reproduced BYTE-FOR-BYTE. Default ON => the derived
    powers-of-two + mod-by-floor path is the golden build (the hand-authored
    ``(v << s) & 0xFF`` / ``v >> s`` lambdas have been deleted;
    ``C4_DERIVE_SHIFT=0`` is a legacy no-op kill-switch for cache-key isolation).
    """
    return os.environ.get("C4_DERIVE_SHIFT", "1") != "0"


def derive_cmp_enabled() -> bool:
    """Flag for the COMPARISON family derived from ONE zero-detector (task
    #446). DEFAULT ON.

    When ``C4_DERIVE_CMP=1`` the six comparison opcodes' L10 decode banks
    (both the ``ComparisonCombine`` decode-row path
    ``_l10_comparison_combine_rules`` and the L10-main ALU cmp lane
    ``_layer10_alu_cmp_combine_rules``) are re-expressed by the single DSL
    generator :func:`building_blocks_dsl.derived_comparison_rules`, which
    realizes BLOG_SPEC §576-590 literally: all of EQ/NE/LT/GT/LE/GE reduce to
    ONE zero-detector primitive ``Z(d)`` (the §510 +1/-2/+1 second-difference,
    computed per nibble by the upstream L9 comparator as the CMP equality/less
    flags) plus its sign. Two derived combinators ``A_EQ_B := HI_EQ ∧ LO_EQ``
    and ``A_LT_B := HI_LT ∨ (HI_EQ ∧ LO_LT)`` are built once, and every opcode
    is then pure boolean algebra over them (EQ=A_EQ_B, NE=¬A_EQ_B, LT=A_LT_B,
    GT=¬A_LT_B∧¬A_EQ_B, LE=A_LT_B∨A_EQ_B, GE=¬A_LT_B) — ZERO per-op magic
    constants, replacing the hand-authored per-op default+override
    enumeration.

    The derivation reproduces the hand-authored 18-unit banks byte-for-byte
    when passed the golden structural constants (proof:
    ``tools/verify_derive_cmp.py`` — the whole-model golden hash is UNCHANGED
    under this flag). Default ON => the single-zero-detector derivation is the
    golden build (the per-op default+override hand-enumeration in both banks has
    been deleted; ``C4_DERIVE_CMP=0`` is now a legacy no-op kill-switch, since
    either value is byte-identical). This is a pure architecture/derivation-
    provenance flip, not a behavior change.

    Registered in BOTH cache-key snapshots in
    ``full_vm_compiler_dynamic.py`` for cache-key isolation.
    """
    return os.environ.get("C4_DERIVE_CMP", "1") != "0"
# ---------------------------------------------------------------------------
# CONTROL-family PC-override gate derivation (task #391 + #395)
#
# Cherry-picked from the CONTROL branch (worktree-agent-a73192c6a3fb53696,
# docs/DERIVE_CONTROL_2026_07_09.md) so the ACTIVATION-SCALE pilot can extend it.
# ``C4_DERIVE_CONTROL`` re-derives the BLOCKER magnitudes (verdict-preserving);
# ``C4_DERIVE_GATE_SCALES`` additionally re-derives the POSITIVE weights +
# threshold from the per-dim runtime activation-scale datum
# (``verification.activation_scales``) — the piece the CONTROL branch flagged as
# not-yet-derivable from the ISA identity alone.
# ---------------------------------------------------------------------------

def derive_jmp_enabled() -> bool:
    """Flag for the SPEC-ALONE-derived JMP PC-override gate (task #391 pilot).
    Default OFF. See ``derive_pc_override_gate``: re-derives the JMP override
    gate's blocker magnitudes from the gate's own structure."""
    return os.environ.get("C4_DERIVE_JMP", "0") != "0"


def derive_control_enabled() -> bool:
    """Flag for the SPEC-ALONE-derived CONTROL-branch family (task #391 part 1).
    Default OFF.

    Generalizes the JMP pilot to JMP/BZ/BNZ/JSR, all routing their PC override
    through the shared ``pc_mux`` encoder. ``C4_DERIVE_CONTROL=1`` re-derives
    each branch op's PC-override gate BLOCKER magnitudes from the ISA
    branch-condition alone (``derive_pc_override_gate``), eliminating the per-op
    blocker magic constants while staying VERDICT-PRESERVING. The positive
    weights + threshold are KEPT (they carry the runtime-activation-scale +
    satisfiability data — the negative result in the CONTROL doc proves they are
    not over-margin). Default OFF => the hand path stays the golden build."""
    return os.environ.get("C4_DERIVE_CONTROL", "0") != "0"


def derive_gate_scales_enabled() -> bool:
    """Flag for the ACTIVATION-SCALE-derived CONTROL gate (task #395). Default OFF.

    The CONTROL branch (``derive_control_enabled``) derives only the gate BLOCKER
    magnitudes and KEEPS the hand positive weights + threshold, flagging the
    per-dim runtime activation SCALE as the missing spec datum needed to derive
    those too. ``C4_DERIVE_GATE_SCALES=1`` closes that datum: it re-derives the
    FULL gate — positive weights = ``1/activation_scale(dim)`` (from the
    calibration datum ``verification.activation_scales``), threshold from the
    NORMALIZED balanced-AND (``n_pos - 0.5`` over unit-scale contributions, plus
    the amplified step-guard's own weight), AND the blocker safety factor — with
    ZERO hand-tuned per-op numbers.

    This is a RE-DERIVATION (behaviour-correct, NOT byte-identical): it replaces
    the hand ``OP_BZ=0.2`` (= ``1/5``, the measured OP_BZ activation scale) +
    ``threshold=3.5+10`` with the derived form. Implies ``C4_DERIVE_CONTROL``
    (the blocker derivation) — turning gate-scales on turns control on. Default
    OFF => the hand-authored gate stays the golden build; the derived build must
    NEVER share a memo / disk cache entry with the golden (registered in both
    cache-key snapshots in ``full_vm_compiler_dynamic.py``)."""
    return os.environ.get("C4_DERIVE_GATE_SCALES", "0") != "0"


# Spec-structural safety factor for the derived PC-override gate blocker
# magnitude (task #391): ``blocker = -k * max_pos_weight * n_pos``. The ONLY
# numeric input to the blocker derivation and it is SPEC-STRUCTURAL (a fixed veto
# headroom that scales with the AND width), NOT a per-op magic constant.
# Overridable via ``C4_PC_OVERRIDE_K`` for the margin study.
def _pc_override_safety_factor() -> float:
    try:
        return float(os.environ.get("C4_PC_OVERRIDE_K", "1"))
    except ValueError:
        return 1.0


def derive_pc_override_gate(
    conditions, *, hand_threshold=None, safety_factor=None,
):
    """Re-derive a CONTROL-branch PC-override gate's BLOCKER magnitudes from the
    gate's own structure (task #391), eliminating the per-op blocker magic
    constants while staying VERDICT-PRESERVING by construction.

    The POSITIVE weights + threshold are PRESERVED (they carry the runtime-scale
    + satisfiability information). Only the BLOCKER magnitudes are re-derived:
    each ``w < 0`` dim -> ``-safety_factor * max_pos_weight * n_pos`` where
    ``max_pos_weight`` is the gate's own largest positive weight. Blockers only
    get STRONGER-or-equal vetoes on the same firing structure, so no should-fire
    row is newly blocked and no should-block row newly admitted.

    Returns ``(rebuilt_conditions, threshold)``.
    """
    if safety_factor is None:
        safety_factor = _pc_override_safety_factor()
    pos_weights = [w for _dim, w in conditions if w > 0]
    n_pos = len(pos_weights)
    max_pos = max(pos_weights) if pos_weights else 1.0
    block = safety_factor * float(max_pos) * float(max(n_pos, 1))
    out = tuple(
        (dim, w) if w > 0 else (dim, -block) for dim, w in conditions
    )
    thr = hand_threshold if hand_threshold is not None else n_pos - 0.5
    return out, thr


# The step-guard amplitude — an AMPLIFIED positive whose weight is a
# satisfiability lever (the AND is unsatisfiable unless the step-guard dim is
# active), NOT a scale-normalizer. It is preserved by the scale derivation (its
# weight is not ``1/scale``; it deliberately over-weights so the threshold offset
# it contributes hard-gates the step-0 kill). Detected structurally: a positive
# weight far above the reciprocal of any plausible activation scale (>= this).
_STEP_GUARD_AMP_FLOOR = 3.0


def derive_gate(conditions, *, position_class: str = "*",
                hand_threshold=None, safety_factor=None, scales=None,
                preserve_blockers: bool = False):
    """FULLY derive a balanced-AND gate — POSITIVE weights + threshold + BLOCKER
    magnitudes — from (a) the per-dim runtime ACTIVATION-SCALE datum and (b) the
    ISA-identity balanced-AND structure, with ZERO hand-tuned per-op numbers
    (task #395). Closes the CONTROL branch's flagged datum.

    ``scales`` — an optional :class:`ActivationScales` to use instead of the
    process-loaded datum (dependency injection for tests / a bespoke calibration).

    ``preserve_blockers`` (task #452) — when True, derive ONLY the positive
    weights (``1/scale``) + threshold and KEEP the hand blocker magnitudes. Use
    this for a gate whose negatives are BROADCAST-DEFEAT guards: they veto the
    in-step-broadcast AMPLIFIED opcode (~5.2) at NON-firing marker rows, a much
    larger perturbation than the firing-row positive sum, so the safety-factor
    veto ``-(pos_sum+1)`` is too weak there and would let the gate mis-fire on a
    broadcast row (regressing the autoregressive PC). The l16 jmp_ax_preserve /
    l6 jsr_sp_fixup families are this class — their hand ``-1e6`` / ``-10``
    blockers are the guard, NOT a magic constant the safety factor can replace.

    Derivation (see docs/DERIVE_ACTSCALE_2026_07_09.md):

      * Each SCALE-NORMALIZED positive dim (a discriminator one-hot / flag) gets
        weight ``1/activation_scale(dim, position_class)`` so its residual
        contribution ``scale * weight == 1.0`` — a unit AND term. (The hand
        ``OP_BZ=0.2`` == ``1/5.0`` == ``1/scale(OP_BZ)`` is REPRODUCED, not
        hand-authored.)
      * A STEP-GUARD positive (weight ``>= _STEP_GUARD_AMP_FLOOR``, e.g. the BZ
        ``HAS_SE=10`` step-0 kill) is PRESERVED verbatim — it is a satisfiability
        amplitude, not a scale-normalizer, so it is a spec datum (the guard MUST
        be active for the gate to fire), and it contributes its FULL weight to
        the threshold.
      * ``threshold = (n_norm - 0.5) + sum(step_guard_weights)`` — the balanced
        AND over the ``n_norm`` unit-scale positives (all-on ``n_norm`` vs
        missing-one ``n_norm - 1`` => midpoint ``n_norm - 0.5``) shifted up by
        each preserved step-guard amplitude. (The hand BZ ``threshold = 3.5 + 10``
        is REPRODUCED: 4 normalized positives -> 3.5, plus the HAS_SE guard 10.)
      * Each BLOCKER (``w < 0``) is re-derived to the spec-structural veto
        ``-safety_factor * (derived_pos_sum + 1)`` so a single active blocker
        vetoes past the threshold even when EVERY positive fires (scaled to the
        gate's FULL activation regime, including the amplified step-guard).

    Returns ``(rebuilt_conditions, threshold)``.

    **Satisfiability safety.** Deadness is decided by whether the HAND gate could
    EVER fire: ``sum(hand_weight * activation_scale)`` over positives (maximal
    firing) vs the hand threshold. A DEAD hand band (that maximal sum < hand
    threshold — unsatisfiable by design; the reserved delayed/first-step JMP bands
    whose ``CMP+0`` / ``HAS_SE==0`` discriminator never co-activates and whose
    ``CONST=-1000`` veto enforces deadness) is returned UNTOUCHED — positives,
    threshold, AND its deliberately-huge blockers all preserved — so the
    derivation can NEVER resurrect it. Only a LIVE band (maximal sum >= hand
    threshold — e.g. BZ ``1 + 0.2*5 + 1.28 + 1 + 10 = 14.28 >= 13.5``) gets the
    DERIVED balanced-AND (positives ``1/scale``, threshold ``n_norm - 0.5 +
    step_guard``, blockers from the safety factor).
    """
    from ...verification.activation_scales import load_activation_scales

    if safety_factor is None:
        safety_factor = _pc_override_safety_factor()
    if scales is None:
        scales = load_activation_scales()

    # --- Deadness check FIRST -------------------------------------------------
    # RAW hand runtime sum over positives: sum(hand_weight * activation_scale) —
    # what the HAND gate accumulates when every positive dim is active at its
    # measured scale. If that maximal firing sum is BELOW the hand threshold, the
    # band is unsatisfiable by design (a reserved / disabled override band — the
    # delayed / first-step JMP bands, whose CMP+0 / HAS_SE==0 discriminator never
    # co-activates and whose CONST=-1000 veto enforces deadness). A DEAD band is
    # returned UNTOUCHED: its positives, threshold, and (crucially) its
    # deliberately-huge blockers stay as-authored, so the derivation can NEVER
    # resurrect it. Only LIVE bands are re-derived.
    if hand_threshold is not None:
        raw_max_fire = sum(
            float(w) * scales.scale(dim, position_class)
            for dim, w in conditions if w > 0
        )
        if hand_threshold > raw_max_fire + 1e-6:
            return tuple(conditions), hand_threshold

    # --- LIVE band: full scale derivation -------------------------------------
    rebuilt = []
    n_norm = 0
    step_guard_total = 0.0
    norm_pos_weights = []
    for dim, w in conditions:
        if w <= 0:
            rebuilt.append((dim, w))  # blocker sign preserved; magnitude below
            continue
        if w >= _STEP_GUARD_AMP_FLOOR:
            # amplified satisfiability guard — preserve verbatim, add to threshold
            rebuilt.append((dim, w))
            step_guard_total += float(w)
            continue
        # scale-normalized discriminator: weight = 1/scale
        s = scales.scale(dim, position_class)
        nw = 1.0 / s if s > 0 else 1.0
        rebuilt.append((dim, nw))
        norm_pos_weights.append(nw)
        n_norm += 1

    thr = (n_norm - 0.5) + step_guard_total

    # Blocker magnitude: a single active blocker must veto the gate even when ALL
    # positives fire. The maximal positive contribution is ``derived_pos_sum``
    # (each normalized discriminator + each preserved step-guard); a blocker at
    # ``-safety_factor * (derived_pos_sum + 1)`` guarantees
    # ``derived_pos_sum - block < thr`` for any ``safety_factor >= 1``, so no
    # should-block row is admitted (verdict-PRESERVING). This scales the veto to
    # the gate's FULL activation regime (INCLUDING the amplified step-guard —
    # e.g. BZ's HAS_SE=10), which the normalized-positive-only max would miss.
    if preserve_blockers:
        # keep the hand blocker magnitudes (broadcast-defeat guards) verbatim;
        # only the positives + threshold are re-derived from the scale datum.
        out = tuple(rebuilt)
        return out, thr
    derived_pos_sum = sum(w for _d, w in rebuilt if w > 0)
    block = safety_factor * (float(derived_pos_sum) + 1.0)
    out = tuple(
        (dim, w) if w > 0 else (dim, -block) for dim, w in rebuilt
    )
    return out, thr


def derive_and_gate_maybe(conditions, threshold, *, position_class="*",
                          preserve_blockers: bool = False):
    """Family-wide ``derive_gate`` opt-in for a ``multi_way_and_rule`` caller
    (task #452 rollout). Returns ``(conditions, threshold)``:

      * when ``C4_DERIVE_GATE_SCALES`` is ON, the derived gate
        (:func:`derive_gate`, positives = ``1/activation_scale(dim,
        position_class)``, threshold from the normalized balanced-AND +
        step-guard) — reproducing the hand positives + threshold from the
        calibration datum (verdict-preserving);
      * OFF (the golden default), the hand ``conditions`` / ``threshold``
        UNCHANGED.

    ``preserve_blockers`` — pass True for a gate whose negatives are
    BROADCAST-DEFEAT guards (they veto the in-step-broadcast amplified opcode at
    NON-firing marker rows). Their magnitude is NOT a safety-factor veto sized to
    the firing-row positive sum, so re-deriving them is too weak at the broadcast
    rows and regresses the autoregressive PC (PROVEN on id550 div_step=6 with the
    l16/l6 JSR/JMP families). With ``preserve_blockers=True`` the positives +
    threshold derive from the scale datum but the hand blockers are KEPT.

    This is the generalization of the L6 ``pc_mux`` wiring
    (``l6_ops._maybe_derive_pc_mux_spec``) to any AND-shaped gate: an
    opcode-reciprocal / marker balanced-AND whose positive discriminators read
    at their calibrated ``activation_scale``. ``position_class`` selects the
    scale axis — e.g. the L16 branch/frame correctors fire at ``"mark==AX"``
    where an opcode one-hot reads ``5.2`` (so its hand ``0.2`` == ``1/scale``
    derives), whereas an L6 override band reads that same opcode at ``"*"``
    (``1.0``, keeping the reserved band dead). The deadness guard in
    :func:`derive_gate` (a hand-DEAD band stays dead) makes this always safe to
    wrap a mixed live/dead family.
    """
    if not derive_gate_scales_enabled():
        return tuple(conditions), threshold
    return derive_gate(
        tuple(conditions), position_class=position_class,
        hand_threshold=threshold, preserve_blockers=preserve_blockers,
    )


def derive_memory_enabled() -> bool:
    """Umbrella flag for the fully-DERIVED MEMORY family (LI/LC/SI/SC/PSH).
    Default OFF.

    BLOG_SPEC §408-412 ("Memory") models C4 memory as a KV-attention
    binary-address CAM: a STORE (SI/SC) attends to its registers for
    address+value, encodes the address in binary (key ``+scale`` for a 1-bit,
    ``-scale`` for a 0-bit), and ALiBi recency prioritises the exact-address /
    most-recent write; a LOAD (LI/LC) queries with a key identical to the store
    key and retrieves the value. That ONE mechanism is the DSL
    :func:`isa_semantics_dsl.cam_binary_address_match` primitive
    (:class:`CamBinaryAddressBlock` per-bit comparator +
    :class:`CamDiscriminatorSlot` gates + :class:`CamValueBand` value relay).

    The MEMORY heads are ALREADY DERIVED through that primitive on the DEFAULT
    (flag-OFF) path, byte-identically to the hand-authored form (proof:
    ``test_isa_semantics_dsl.py`` L15 LI/LC-load, L14 mem-generation store, L13
    relay, L7 operand-gather, L8 mem/fetch heads == handbuilt; whole-model
    golden hash unchanged). See ``docs/semantic_spec_MEMORY.md`` §2c and
    ``docs/DERIVE_MEMORY_2026_07_09.md``. So the byte-identical *lowering* work
    is complete and needs no flag.

    What this flag turns ON is the DERIVED-CAM MEMORY FIX PATH — the clean
    address-keyed store-provenance CAM that a correct derivation of BLOG_SPEC
    §408-412 IMPLIES but the hand-authored value-row load could not deliver
    (memory note ``project_si_store_provenance_two_root_wall``: the store VALUE
    rows are provenance-blind on BOTH axes; the clean ``(address, value)`` pair
    lives on the store AX-MARKER, so the derived CAM keys the LI-query on its
    ``AX_CARRY`` target address, matches the store marker's ``ADDR_B0``, and
    copies the marker's ``AX_CARRY`` clean value). It is the SINGLE entry point
    for the two derived-memory FIX heads, each already built as a clean
    :func:`cam_binary_address_match` head with ZERO per-op magic-number
    correctors:

      * ``si_store_addr_enabled()`` -- L15 head-16 SI/SC store-provenance CAM
        (the ``var_mul``/multilocal ``LI a`` returns b's value wall). Keyed on
        ``AX_CARRY`` (target addr) -> store-marker ``ADDR_B0`` -> clean marker
        ``AX_CARRY`` value.
      * ``var_three_li_enabled()`` -- the L15 head-0 OP_SI/OP_SC store-row veto
        (the ``var_three`` ``SI b`` stray-relay desync).

    Mirrors ``campaign_enabled()``: ``C4_DERIVE_MEMORY=1`` supplies an ON floor
    for those two predicates, and an explicit per-flag value still wins
    (``C4_SI_STORE_ADDR=0`` opts that one out even under this umbrella). DEFAULT
    OFF -> every predicate keeps its exact pre-existing value, so the golden
    (flag-OFF) build is byte-identical (``tools/_isa_golden_hash.py`` unchanged).
    Registered in BOTH cache-key snapshots in ``full_vm_compiler_dynamic.py`` so
    a derive-memory build never shares a memo / disk cache entry with a golden
    build. These derived-CAM heads use the 30-token campaign MEM-from-SP signals
    (``AX_CARRY``/``ADDR_B0`` marker provenance), so ``C4_DERIVE_MEMORY=1`` is
    only meaningful alongside ``C4_CAMPAIGN=1``; at the golden 35-token frame the
    flag-ON build is byte-identical to flag-OFF (the heads never install).
    """
    return os.environ.get("C4_DERIVE_MEMORY", "0") != "0"


def mul_width2_enabled() -> bool:
    """Return True iff the width=2 (16-bit) MUL path is active (DEFAULT ON).

    The width=2 (8-bit x 8-bit -> 16-bit) MUL is now the PRODUCTION default
    (2026-06-13). When enabled:
      * ``compile_full_vm_dynamic`` injects the dedicated
        ``MUL_RESULT_HI_LO/HI`` byte-1 result band (16+16 dims) into
        ``extra_residual_dims`` so the d_model auto-widen (872 -> 981,
        n_heads 8 -> 9) is HEAD-DIM-PRESERVING (adds heads, keeps every
        existing head's span) -- this is what makes the widen bnz-safe.
      * ``make_efficient_l11_alumul_wrap_op`` bakes the
        ``wide_mul_rules(width_bytes=2)`` 65,536-rule lookup with the
        operand-magnitude-matched 5-way AND + cell-0 artifact blocker,
        routing the product's BYTE 1 into ``MUL_RESULT_HI_*`` (NOT
        ``OUTPUT_LO+32`` = ADDR_KEY) and keeping BYTE 0 in OUTPUT_LO/HI.
      * ``make_layer13_mul_result_hi_relay_op`` stages byte 1 into AX_FULL
        for the existing ``layer15_alu_high_byte_relay`` byte-1 emit.

    Result: ``mul_overflow`` (100*5=500=0x01F4) emits both bytes correctly
    (smoke 49/2 -> 50/1) while ``mul_basic`` (6*7=42), bnz, and the 32-bit
    ALU / bitwise / shift / cmp suite stay green.

    Opt-OUT with ``C4_MUL_WIDTH2=0`` to restore the pre-width2 build
    (d_model 920, width=1 lo-byte MUL, ``mul_overflow`` decodes 20). Any
    other value (or unset) keeps the width=2 default ON.

    ``C4_MUL_MULTIPASS=1`` IMPLIES width=2: the multipass cascade routes the
    product's byte 1 into the ``MUL_RESULT_HI_LO/HI`` band, so that band (and
    the L13 relay that stages it into AX_FULL) MUST be present even if
    ``C4_MUL_WIDTH2=0`` was passed. OR-in the multipass flag so the band is
    always collected when the cascade is installed.
    """
    if os.environ.get("C4_MUL_MULTIPASS", "0") == "1":
        return True
    return os.environ.get("C4_MUL_WIDTH2", "1") != "0"


def mul_multipass_enabled() -> bool:
    """Return True iff the multi_pass (schoolbook cascade) MUL compute replaces
    the L11 mul-partial / L12 mul-combine lookup chain (DEFAULT OFF — opt in via
    ``C4_MUL_MULTIPASS=1``).

    GAP-PRIMITIVE #2 install. The live default (``alu_mode='lookup'``) MUL
    computes only the product's BYTE 0 across three lookup blocks:

      * L10 ``_layer10_alu_mul_lo`` (256 units) -> ``OUTPUT_LO`` nib0 = (a0*b0)%16
      * L11 ``mul_partial`` (4096 units)         -> ``TEMP[partial]``
      * L12 ``mul_combine`` (4096 units)         -> ``OUTPUT_HI`` nib1

    Byte 1 (bits 8..15) is never computed on the lookup path, so wide_mul
    programs cross-contaminate byte-0/byte-1 (#334). When this flag is on,
    ``make_mul_partial_op`` replaces ``block.ffn`` with a
    :class:`~neural_vm.efficient_alu_neural.MultiPassMulBlock` — the 7 lowered
    ``PureFFN`` passes of ``multi_pass_mul_rules`` packed into ONE physical
    block (like ``FlattenedALUMul``, so the absolute-position lea contract
    holds). The cascade computes the FULL 16-bit product from a compact
    schoolbook spec (2848 units vs the 4096+4096 lookup), routing byte 0 to
    ``OUTPUT_LO/OUTPUT_HI`` and byte 1 to the ``MUL_RESULT_HI_LO/HI`` band that
    the L13 relay stages into AX_FULL for the byte-1 emit.

    The redundant L10 mul_lo + L12 mul_combine byte-0 writers are gated OFF
    when this flag is on (the cascade owns every result nibble); the L11
    ``mul_partial`` TEMP band is no longer produced. Needs
    :func:`mul_width2_enabled` (the ``MUL_RESULT_HI_*`` byte-1 band + the L13
    relay) — implies it.

    DEFAULT OFF: flag-off is byte-identical to golden ``91f55411`` (the lookup
    chain is untouched). Opt in with ``C4_MUL_MULTIPASS=1``.
    """
    return os.environ.get("C4_MUL_MULTIPASS", "0") == "1"


def div_multipass_enabled() -> bool:
    """Return True iff the multi_pass (binary long-division cascade) DIV/MOD
    compute replaces the live L10 ``FlattenedDivMod`` composite (DEFAULT OFF —
    opt in via ``C4_DIV_MULTIPASS=1``).

    GAP-PRIMITIVE #2 DIV install (the DIV analogue of ``mul_multipass_enabled``).
    The live default (``alu_mode='lookup'``) computes DIV/MOD via the
    hand-authored ``FlattenedDivMod`` — an 8-outer x 3-inner GE-workspace long
    division (BD->GE, DIV pipeline, MOD pipeline, GE->BD). When this flag is on,
    the L10 divmod install (``make_alu_divmod_composite_ops`` -> install op)
    appends a :class:`~neural_vm.efficient_alu_neural.MultiPassDivBlock` instead
    — the lowered ``PureFFN`` passes of ``multi_pass_div_rules`` (bit-serial
    shift-subtract with a cross-pass running-remainder carry) packed into ONE
    physical block (like ``MultiPassMulBlock``, so the absolute-position lea
    contract holds). The cascade computes ``a // b`` (quotient) + ``a % b``
    (remainder) from a compact spec on dedicated result lanes, then routes
    q -> OUTPUT for OP_DIV and r -> OUTPUT for OP_MOD at MARK_AX. It reads the
    dividend from ``ALU_LO/HI`` (byte 0) and divisor from ``AX_CARRY_LO/HI``,
    the SAME operand bands the GE-format lookup / ``FlattenedDivMod`` consume.

    DEFAULT OFF: flag-off is byte-identical to golden ``91f55411`` (the
    ``FlattenedDivMod`` composite is untouched). Opt in with
    ``C4_DIV_MULTIPASS=1``.
    """
    return os.environ.get("C4_DIV_MULTIPASS", "0") == "1"


def mul_stack0_byte39_guard_enabled() -> bool:
    """Return True iff the L10 tail ``byte_39_from_e8_addr`` STACK0 restore rule
    is hardened to require its full store-pop context (DEFAULT OFF — opt-in via
    ``C4_MUL_STACK0_BYTE39_GUARD=1``).

    The bug this lifts (root-caused spec_k=0 GPU full_trace, 2026-06-18):
    the L10 tail-correction rule
    ``tail_stack0_store_top_e8_from_e0_byte_39_from_e8_addr``
    (``ops/l10_ops.py`` ``stack0_store_top_e8_from_e0_output_rules``) restores
    the stored byte value ``0x39`` (= 57) at the e8/e0 local-store address
    transition (load-bearing for genuine ``0x39`` store-pops:
    ``func_identity(57)``, ``func_add(.., 57)``, ...). Its activation is driven
    almost entirely by the UNBOUNDED ``("OUTPUT_LO+9", 100.0)`` term: at the
    binary-op STACK0 byte-0 emission row OUTPUT_LO+9 carries the operand's low
    nibble at magnitude ~3654, so ``100 * 3654 = 365428`` alone blows past the
    ``threshold=20000`` even though EVERY store-pop witness (MEM_STORE,
    EMBED_LO+8, EMBED_HI+14, MEM_ADDR_SRC) is COLD on that row. The rule then
    forces the byte to ``0x39``, corrupting the STACK0 byte-0 token to 57
    (low nibble 9 — coincidentally matching the operand — high nibble 3) for
    EVERY binary op whose operand-A low nibble is 9.

    For MUL specifically the downstream width=2 lookup reads that corrupted
    STACK0 byte-0 as operand A: ALU_LO=9 (right by accident), ALU_HI=3 (the
    phantom 0x39 high nibble), so the 16-bit product is computed against the
    wrong high nibble. That is the SOLE remaining root behind the 4 GPU
    full_trace ``mul_*`` fails — exactly the a_lo=9 cases ``mul_20 (9*98)``,
    ``mul_29 (89*26)``, ``mul_36 (9*44)``, ``mul_43 (9*5)`` (``9*9`` is a_lo=9
    too but its byte-0 is right by accident so it passes). ADD/SUB read the
    same corrupted ALU_HI but tolerate it (the byte-0 add only needs ALU_LO),
    which is why only MUL surfaces it. Verified: sweeping operand-A 1..255
    with B=5, step-3 STACK0 byte-0 is byte-correct for EVERY low nibble except
    9, where it is universally 0x39.

    The genuine ``0x39`` store-pop the rule was built for has the value's HIGH
    nibble ALSO present in OUTPUT (``OUTPUT_HI_THIS_STEP+3`` hot) — it is
    restoring 0x39, so both nibbles evidence 0x39. The binary-op false-fire has
    ONLY the low nibble (``OUTPUT_HI_THIS_STEP+3 == 0``). When enabled this
    guard rebalances the rule so the byte's HIGH nibble (``OUTPUT_HI_THIS_STEP+3``)
    is a load-bearing co-requirement at the same weight as the low nibble, and
    a hard negative blocker suppresses the rule when ``OUTPUT_LO+9`` is hot but
    ``OUTPUT_HI_THIS_STEP+3`` is cold (the binary-op-result signature). The
    genuine 0x39 store-pop (both nibbles hot) is unaffected.

    DEFAULT OFF so the flag-off build stays byte-identical to golden
    ``b9d8861f``. Opt-in via ``C4_MUL_STACK0_BYTE39_GUARD=1``.
    """
    return os.environ.get("C4_MUL_STACK0_BYTE39_GUARD", "0") == "1"


def div_multibyte_enabled() -> bool:
    """Return True iff the multi-byte-dividend DIV/MOD relay is active
    (DEFAULT ON — opt-out via ``C4_DIV_MULTIBYTE=0``; +51: div 21->48/50,
    mod 24->48/50, add/sub guard 84/100 unchanged, smoke 51/0).

    Background — the wall this lifts (verified spec_k=0, built dims):
    the lookup-mode ``FlattenedDivMod`` long-division pipeline
    (``alu/ops/divmod_longdiv.py::LongDivisionModule``) is ALREADY
    multi-byte-capable: it reads the dividend as a full 8-nibble vector
    from GE positions 0..7 and does a real MSB->LSB long division. The
    ``BDToGEConverter`` (``efficient_alu_neural.py``) maps BD operand
    bands into those GE positions: byte 0 from ALU_LO/HI (positions 0/1),
    byte 1 into positions 2/3. For DIV/MOD it RECOVERS byte 1 from the
    autoregressive prefix via a cummax over ``STACK0_BYTE1`` rows.

    The bug (NOT "STACK0_BYTE_VAL_1 written nowhere" — that was a
    static-dim / wrong-row artifact): the converter's DIV/MOD fallback
    gathered ``CLEAN_EMBED_LO/HI`` at the picked STACK0-frame row, where
    that band is 0x00 — the pushed value's high byte is deposited by
    ``layer10_psh_ax_broadcast`` into ``STACK0_BYTE_VAL_1_LO/HI`` at that
    SAME row (e.g. 1162/37: STACK0_BYTE_VAL_1 = 0x04 at the PSH frame).
    Compounded by TIMING: the broadcast runs at L11 (physical block 15)
    but the FlattenedDivMod compute ran at L10 (physical block 14), ONE
    block too early to see the high byte.

    When enabled this op:
      * routes the FlattenedDivMod install to ``layer_idx=11`` so the
        divmod compute runs AFTER the L11 ``layer10_psh_ax_broadcast``
        head populates ``STACK0_BYTE_VAL_1`` (operands ALU_LO/HI +
        AX_CARRY are byte-stable across blocks 14..20, verified
        ``tools/probe_div_operand_survival.py``);
      * has ``BDToGEConverter`` read ``STACK0_BYTE_VAL_1_LO/HI`` (the
        designated high-byte carrier) at the cummax-picked STACK0_BYTE1
        row instead of ``CLEAN_EMBED_LO/HI``.

    DEFAULT ON (verified +51 with no add/sub/smoke regression). Opt-out
    via ``C4_DIV_MULTIBYTE=0`` restores the byte-identical-to-HEAD path
    (divmod stays at block 14, converter reads CLEAN_EMBED). See
    ``docs/DIV_MOD_MULTIBYTE_DIVIDEND_BLOCKER_2026_06_12.md``.
    """
    # Unconditional as of the P5 flag-retire 2026-07-14 (the former
    # ``C4_DIV_MULTIBYTE`` escape hatch was retired as a proven default-ON fix).
    return True


def divmod_byte0_se_recover_enabled() -> bool:
    """Return True iff the divmod DIVIDEND byte-0 SE_ALU recovery is active
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_DIVMOD_BYTE0_SE_RECOVER=0``; only takes effect when the STACK0
    emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``):

    operand-A byte 0 (the dividend low byte) is delivered from ``mem[SP]``
    into ALU_LO/HI by the L8 ``make_layer8_mem_to_alu_op`` head 5 at block
    11. For a MULTI-BYTE dividend the L10 ALU-clear (block 14) then crushes
    ALU_LO/HI all-negative (probe ``tools/probe_div_ge_positions.py``,
    1162/37: ALU_LO == -39 at the divmod-input block 27), so the
    ``FlattenedDivMod`` (block 28) reconstructs dividend byte 0 as 0x00 and
    divides ``high_byte(dividend)*256 / divisor`` -> wrong quotient. The
    SINGLE-byte case is unaffected (ALU_LO survives at +6.0). The byte-1
    (positions 2/3) path is already correct here via AX_FULL /
    STACK0_BYTE_VAL_1 (L8 head 7, Part B) — this fix is the byte-0 half.

    The L9 ``step_end_operand_relay`` head mirrors ALU_LO/HI into
    SE_ALU_LO/HI at block 13 — BEFORE the L10 clear — and that mirror
    SURVIVES to the divmod block (probe: SE_ALU == 0xA/0x8 at block 27 for
    1162/37, == the same nibbles as ALU_LO for single-byte 100/7). When
    enabled, ``BDToGEConverter`` OR-recovers the dividend byte-0 one-hot
    from SE_ALU_LO/HI onto the (possibly crushed) ALU band, gated on the
    divmod opcode + the AX marker so it touches no other op/config.

    DEFAULT ON. Opt-out via ``C4_DIVMOD_BYTE0_SE_RECOVER=0`` restores the
    raw ALU_LO/HI read (the byte-identical-OFF path: flag-OFF or
    ``C4_NO_STACK0_EMIT=0`` are both byte-identical to golden
    ``4958b35b``). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_DIVMOD_BYTE0_SE_RECOVER`` can
    A/B it inside the campaign config.
    """
    # Unconditional as of the P5 flag-retire 2026-07-14 (the former
    # ``C4_DIVMOD_BYTE0_SE_RECOVER`` escape hatch was retired as a proven default-ON fix).
    return True


def mul_byte0_se_recover_enabled() -> bool:
    """Return True iff the MUL operand-A byte-0 SE_ALU recovery + the L11/L15
    MUL OUTPUT-flood cap are active (DEFAULT ON in the campaign config — opt-out
    via ``C4_MUL_BYTE0_SE_RECOVER=0``; only takes effect when the STACK0
    emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``, ~16 of 23 mul fails whose
    product decodes to 0, e.g. mul_2 93*34, mul_3 65*98, mul_28 97*94):

    This is the MUL analog of :func:`divmod_byte0_se_recover_enabled` —
    operand-A byte 0 is delivered from ``mem[SP]`` into ALU_LO/HI by the L8
    ``make_layer8_mem_to_alu_op`` head 5 at block 11 (CLEAN one-hot, +6.0 at
    the true nibble; spec_k=0 ALU-band block trace). The L10
    ALU-clear (block 14) then crushes ALU_LO/HI all-negative (~-39 at the true
    nibble, ~-45 elsewhere) — the SAME crush the divmod path hits. The
    ``BDToGEConverter._clean_onehot`` thresholds those negative cells to 0, so
    operand A reconstructs as 0x00 and the ``FlattenedALUMul`` (block 29 /
    logical L15) computes ``0 * b == 0``. (Operand B byte 0 is read CLEAN from
    AX_CARRY_LO/HI at +0.996 — only operand A is crushed.)

    Two coupled effects in the campaign config:

      (1) BYTE-0 RECOVER. The L9 ``step_end_operand_relay`` head mirrors
          ALU_LO/HI into SE_ALU_LO/HI at block 13 — BEFORE the L10 clear — and
          that mirror SURVIVES to the FlattenedALUMul block (spec_k=0:
          SE_ALU_LO==0xD/SE_ALU_HI==5 at +0.78/
          +0.69 across blocks 14..29 for 93*34, == the operand-A nibbles).
          ``BDToGEConverter`` OR-recovers the operand-A byte-0 one-hot from
          SE_ALU_LO/HI onto the (crushed) ALU band on OP_MUL+MARK_AX rows, so
          FlattenedALUMul multiplies the REAL operand A.

      (2) OUTPUT-FLOOD CAP. The L11 ``efficient_l11_alumul_wrap`` (block 16 /
          logical L11) wide_mul FFN reads the RAW crushed ALU_LO (its own
          threshold-AND rules, NOT through ``_clean_onehot``) and — driven by
          the ~-39 uniform-negative band — FLOODS OUTPUT_LO/HI + MUL_RESULT_HI
          to ~2.4e9 (spec_k=0 OUTPUT-band block trace: OUTPUT_HI sum
          0 -> 2.42e9 at block 16, self-amplified to 6.9e9 by L14, 1.2e20 by
          L20, +inf by L25). This flood (a) trips the ``_MulCombineStage``
          ``already_fired`` guard (OUTPUT_HI band > 1.5) so the CORRECT
          FlattenedALUMul product is NEVER written, and (b) out-votes the +2.0
          product one-hot at the LM-head argmax. When enabled, the
          ``_MulCombineStage`` CLEARS the OUTPUT_LO/HI band on the OP_MUL+
          MARK_AX row before the clean product write and IGNORES the
          ``already_fired`` veto on those rows, so the SE-recovered product
          survives. (The golden 35-token config keeps clean POSITIVE operand
          one-hots through L11, so the wide_mul never floods and this cap is a
          no-op there — but the flag is gated on ``C4_NO_STACK0_EMIT`` so it
          can never touch the golden path.)

    DEFAULT ON. Opt-out via ``C4_MUL_BYTE0_SE_RECOVER=0`` restores the raw
    ALU_LO/HI read + the raw OUTPUT-flood path (the byte-identical-OFF path:
    flag-OFF or ``C4_NO_STACK0_EMIT=0`` are both byte-identical to golden).
    Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_MUL_BYTE0_SE_RECOVER`` can A/B it
    inside the campaign config.
    """
    # Unconditional as of the P5 flag-retire 2026-07-14 (the former
    # ``C4_MUL_BYTE0_SE_RECOVER`` escape hatch was retired as a proven default-ON fix).
    return True


def sub_full_borrow_enabled() -> bool:
    """Return True iff the SUB full-borrow multi-byte 0xFF completion is active
    (DEFAULT ON in the campaign config — opt-out via ``C4_SUB_FULL_BORROW=0``;
    only takes effect when the operand is sourced from ``mem[SP]``, i.e. the
    campaign ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`` config).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config):
    ``sub_borrow_cascade`` (``0 - 1`` should = ``0xFFFFFFFF``) emits only
    ``0x000000FF`` — byte 0 (0xFF) is correct, but bytes 1-3 stay 0x00. It is a
    three-layer wall the per-byte cascade rules cannot reach in the 30-token
    campaign frame:

      1. The minuend byte 1 == 0x00 has NO ``STACK0_BYTE_VAL_1`` one-hot in the
         campaign (the L8 mem[SP] CAM delivers only the NON-zero nibble: ``+6``
         at cell ``v``, nothing for ``v == 0``). So the band is EMPTY and the
         ``_layer14_sub_borrow_high_byte_passthrough`` ``v==0`` rule — keyed on
         ``STACK0_BYTE_VAL_1_LO+0`` being lit — never fires.
      2. The byte-1 result is 0xFF = lo nibble 0xF AND hi nibble 0xF; the v>=1
         cascade rules only set OUTPUT_LO (assuming the hi nibble is 0).
      3. The block-32 (L18) OUTPUT_HI slam adds ``+664`` to OUTPUT_HI cell 0
         (the 0x00 hi default) and ``-434`` elsewhere — additively crushing any
         pre-slam OUTPUT_HI cell-15 write, so the high nibble can never win
         pre-slam.

    The fix is a two-part build, both campaign-gated:
      * a PRECURSOR FFN at an EARLY L14 block (where the STACK0_BYTE_VAL_1 band
        is still fresh) writes the bounded ``SUB_FULL_BORROW`` flag on the SUB
        byte-1 emit row (``TEMP+9`` + ``IS_BYTE`` + ``H1+1`` + ``BYTE_INDEX_0``
        + ``CARRY+2`` borrow) ONLY when the STACK0_BYTE_VAL_1 band is EMPTY (the
        minuend byte 1 == 0 full-underflow case);
      * a POST-SLAM writer on the L25 tail block (after
        ``tail_bit32_result_correction``, the last OUTPUT writer before the LM
        head) reads the persisted flag and overwrites OUTPUT byte 1 = 0xFF
        (cancel OUTPUT_HI cell-0, boost OUTPUT_LO/HI cell 15), which DOMINATES
        the slam because it runs after it.

    DEFAULT ON. Opt-out via ``C4_SUB_FULL_BORROW=0`` restores the byte-identical
    pre-fix path (flag-OFF or ``C4_OPERAND_FROM_MEMSP=0`` are both byte-identical
    to golden ``7f6f2e5d``: the band is flag-gated so a flag-off build omits it
    entirely → smaller d_model). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_SUB_FULL_BORROW`` can A/B it inside
    the campaign config.
    """
    if not operand_from_memsp_enabled():
        return False
    return os.environ.get("C4_SUB_FULL_BORROW", "1") != "0"


def l8_operand_sp_disc_enabled() -> bool:
    """Return True iff the L8 head-5 operand-A SP-frame discriminator is active
    (DEFAULT ON in the campaign config — opt-out via ``C4_L8_OPERAND_SP_DISC=0``;
    only takes effect when the operand-A read is routed to ``mem[SP]``, i.e. the
    campaign ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`` config).

    The wall this lifts (validated GPU spec_k=0, BUILT dims, campaign config —
    tools/probe_addmul_head5_cam.py id816 4+5*2): ``expr_add_mul`` (``a+b*c``) is
    the only depth-2 expr cluster with TWO live stack stores at once (PSH a,
    PSH b, IMM c, MUL pops b, ADD pops a). The ADD's operand-A = mem[SP] = a (the
    FIRST push; top-of-stack after the MUL popped b). The recency-only L8 head-5
    ``mem[SP]`` CAM (``make_layer8_mem_to_alu_op``) picks the more-recent matching
    STORE row (the popped b push) over the live a store — the two store value
    rows are byte-identical (MEM_STORE_AT_VAL=1, MEM_VAL_B1=1 on BOTH; ADDR_B0/1/2
    = 0; emitted mem address byte = 0xE0 for BOTH). So AX = b + b*c (id816:
    5+10=15 vs 14).

    The ONLY real discriminator is the SP at push time: store a was pushed at
    SP=0xF8, store b at SP=0xF0; after the MUL pop SP returns to 0xF8, so the
    ADD's operand address is 0xF8 == a's frame. The SP low-byte is a clean
    one-hot in OUTPUT_LO at the MARK_SP marker row (id816: SP rows 108->8 (0xF8),
    168->0 (0xF0), 228 (post-MUL)->8 (0xF8)).

    When this flag is on:
      * ``make_layer7_sp_addr_relay_op`` (L7, block 9 — BEFORE head-5 reads at
        block 11) relays the push-time SP low byte into the fresh per-row band
        ``SP_ADDR_LO``: Q fires at MEM store value rows + binary-op AX query
        rows; K matches MARK_SP; steep ALiBi recency (0.5) picks the NEAREST
        PRIOR MARK_SP; V copies its OUTPUT_LO one-hot. add-step query@253
        SP_ADDR_LO=8; store@123 (a=4) =8 (MATCH); store@183 (b=5) =0 (MISMATCH).
      * ``make_layer8_mem_to_alu_op`` head-5 adds an OP_*-gated SP-mismatch
        penalty (anti-complement form): K[base+30+i]=SP_ADDR_LO+i (store side),
        Q[base+30+i]=G*SP_ADDR_LO+i (query side) + one const dim K=1/Q=-G, so the
        head-5 score gains ``G*(match - 1)`` = 0 on a frame MATCH and ``-G`` on a
        MISMATCH. The popped (mismatched) store is demoted by G; recency then
        picks the live (matched) store. match==0 keeps single-store ops
        (mul_div/mod/paren/standalone add/sub) byte-identical (their live store
        SP-matches its query so the penalty is 0 there).

    DEFAULT ON. Opt-out via ``C4_L8_OPERAND_SP_DISC=0`` restores the byte-
    identical pre-fix path (flag-OFF or ``C4_OPERAND_FROM_MEMSP=0`` are both
    byte-identical to golden ``f2b040aa``: the SP_ADDR_LO band is flag-gated so a
    flag-off build omits it entirely → smaller d_model, and the head-5 penalty
    dims only bake under the flag). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_L8_OPERAND_SP_DISC`` can A/B it
    inside the campaign config.

    *** DEFAULT ON (2026-06-26) — BLOCKER-1 RESOLVED, +9 expr_add_mul. ***
    The DISCRIMINATOR CORE was GPU-validated 2026-06-25 (at the depth-2 expr_add_
    mul ADD step it demotes the popped-``b`` store and head-5 delivers operand-A =
    the live ``a``: probe_addmul_head5_cam.py id816 ALU_LO 5->4, AX 15->14) but
    shipped OFF pending BLOCKER-1. BLOCKER-1 is now FIXED by the block-10
    WINNER-TAKE-ALL sharpener (``make_layer7_sp_addr_sharpen_op``): it saturating-
    clamps each relayed SP_ADDR_LO cell to [0,1] (-> SP_ADDR_LO_SHARP /
    SP_ADDR_PRESENT_SHARP, which head-5's penalty now reads), so the bilinear
    G*(<q,k> - PRESENT_q*PRESENT_k) penalty stays bounded (~few hundred, was ~+80k)
    and cancels EXACTLY at the matched winner for ALL ops. var_simple no longer
    regresses. GPU full_trace 250-274,800-899 (campaign, spec_k=0): flag-ON
    85/125 vs flag-OFF 76/125 = +9 -- expr_add_mul 0/25 -> 9/25 with NO regression
    (var_simple 25/25, expr_mod 25/25, expr_mul_div 10/10, expr_paren 16/16 all
    HOLD). lint_cross_op_attention PASS (no shared-head softmax hazard). Flag-OFF
    and non-campaign builds stay byte-identical (the SP_ADDR* bands are flag-gated,
    omitted -> smaller d_model -> golden ``2d227d48`` unchanged), so this default
    flip only takes effect INSIDE the campaign config where it was validated.

    BLOCKER-2 (downstream expr_add_mul roots) is INDEPENDENT and remains: the 16
    still-failing expr_add_mul diverge later at step-5 (MUL large-value corruption)
    / step-7 (framing on small-value progs; id816 4+5*2 reaches neural=14=expected
    but mis-frames an intermediate step) -- the discriminator+sharpener is now net
    +9 on its own and BLOCKER-2 is the remaining headroom (0/25 -> 9/25 done).

    Opt-out: ``C4_L8_OPERAND_SP_DISC=0`` restores the byte-identical pre-fix path
    (kept as a kill-switch so ``tools/flag_regression_gate.py --flag
    C4_L8_OPERAND_SP_DISC`` can still A/B it inside the campaign).
    """
    if not operand_from_memsp_enabled():
        return False
    return os.environ.get("C4_L8_OPERAND_SP_DISC", "1") != "0"




def mul_multibyte_l19_boost_enabled() -> bool:
    """Return True iff the NARROWED MULTI-byte MUL byte-0 product is BOOSTED on
    the LITERAL-mul ``MUL+MARK_AX`` product row so it survives the block-34
    (logical L19) OUTPUT-byte-0 overwrite (DEFAULT ON in the campaign config —
    opt-out via ``C4_MUL_MULTIBYTE_L19_BOOST=0``; only takes effect under
    ``C4_NO_STACK0_EMIT=1`` + ``C4_MUL_BYTE0_SE_RECOVER=1``, so flag-OFF /
    non-campaign is byte-identical to golden ``7f6f2e5d``).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``): the 6 multi-byte
    LITERAL-mul fails ``{127,130,134,139,141,144}`` = ``28*70, 92*40, 58*16,
    76*29, 91*81, 43*10`` — the byte-0 of a product whose byte 1 != 0. The
    ``mul_l19_product_boost`` cap handles the SINGLE-byte products (byte 1 ==
    0); these multi-byte ones keep a CLEAN byte-0 one-hot (mag ~14.3 through
    block 32) but at block 34 (logical L19) a PureFFN ADDS the BYTE-1 value into
    ``OUTPUT_LO`` at a DOMINANT magnitude (~180.8 at the byte-1 cell) on the
    MARK_AX row, so the LM-head argmax flips the byte-0 LOW nibble to the byte-1
    value (``1960 -> 1799 = 0x0707``).

    THE NARROWING (vs the dropped 2026-06-24 version, which gated only on
    ``res_b1 >= 0.5`` and so ALSO fired on ``var_mul``'s multi-byte MUL row —
    where the L11 wide_mul has ALREADY mis-fired on the crushed multi-local
    operand band so the boosted OUTPUT is garbage, REGRESSING ``var_mul``): the
    boost is gated to fire ONLY on the LITERAL-mul product row. The
    DISCRIMINATOR (probed spec_k=0 READING the EXACT ``state.x_bd_in`` the
    FlattenedALUMul composite receives, BUILT dims, on the MUL+MARK_AX
    first-fire row): the ``STACK0_B0_H1_PREV`` + ``STACK0_B0_H3_PREV`` cross-step
    carry band SUM (the ``C4_STACK0_B0_DUMP`` re-supply that only runs in a
    MULTI-step / multi-local frame) is

      * LITERAL mul (``return N*M;`` -> IMM/PSH/IMM/MUL, no ENT frame): 55..667
      * ``var_mul`` (``int a;int b;a=N;b=M;return a*b;`` -> ENT frame +
        LI-loaded operands -> cross-step STACK0 carry active): 7379 (rock-solid
        UNIFORM across var_mul_0..23).

    The band is NON-ZERO on the literal row at the COMPOSITE INPUT (block-30
    attention adds it) even though it is ~0 at the block-29 OUTPUT — so the gate
    reads ``x_bd_in`` and splits on MAGNITUDE: ``var_frame_carry < 2000`` (a >3x
    margin on BOTH sides: 667 << 2000 << 7379) marks the literal frame. So the
    narrowed gate adds that test to the multi-byte test -> the boost fires on
    the 6 literal fails ONLY and leaves every ``var_mul`` row untouched.

    FIX. ``_MulCombineStage`` marks the LITERAL multi-byte MUL+MARK_AX rows
    (``res_b1 >= 0.5`` AND ``var_frame_carry < 2000`` AND NOT the single-byte
    cap row); ``_GEToBDStage`` SCALES (25x, NO clear — the band is a clean
    one-hot, not a flood) the byte-0 ``OUTPUT_LO/HI`` band there so the true
    product LOW nibble out-votes the L19 add. byte-0 (``OUTPUT_LO/HI``) ONLY ->
    the byte-1 ``AX_FULL`` relay (a different emit row) is untouched; a uniform
    scale of a single-cell one-hot is argmax-invariant so the PASSING multi-byte
    literal muls (21*59, 65*98, ...) are byte-identical.

    DEFAULT ON. Opt-out via ``C4_MUL_MULTIBYTE_L19_BOOST=0`` restores the
    unboosted byte-0 (flag-OFF, or ``C4_NO_STACK0_EMIT=0``, or
    ``C4_MUL_BYTE0_SE_RECOVER=0`` are all byte-identical to golden — the boost
    path is campaign-only). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py``.
    """
    # Unconditional as of the P5 flag-retire 2026-07-14 (the former
    # ``C4_MUL_MULTIBYTE_L19_BOOST`` escape hatch was retired as a proven default-ON fix).
    return True


def cmp_eq_hinib_veto_enabled() -> bool:
    """Return True iff the EQ engine's HIGH-nibble artifact-veto is active
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_CMP_EQ_HINIB_VETO=0``; only takes effect when the STACK0 emission is
    dropped, i.e. ``C4_NO_STACK0_EMIT=1``, so flag-OFF / non-campaign is
    byte-identical to golden ``7f6f2e5d``).

    The residual cmp wall this lifts (the NON-crushed operand-gather index-0
    artifact, ``project_operand_gather_hybrid_encoding_is_cmp_alu_root``): the
    per-nibble EQ engine (:func:`_layer10_alu_eq_engine_rules`) writes the
    decisive ``eq_one`` 0x01 OUTPUT decode-margin push at the AX row for EQUAL
    operands. Its 4-way AND weighted ``ALU_HI`` at only ``0.1`` so the
    high-nibble term cannot VETO a mismatch: for ``if_eq_20: 28 == 12`` (A=0x1C
    / B=0x0C, sharing low nibble 0xC) the unit ``(h=0, l=0xC)`` fires on B's
    high nibble 0 even though A's high nibble is 1 — the index-0 magnitude
    artifact (``ALU_HI+0 ≈ +5.3``) HELPS the wrong unit clear threshold — so
    the 0x01 push lands and ``28 == 12`` mis-decodes to 1 (and the same for
    ``30 == 28``, ``40 == 35``, the documented residual EQ-false band).

    FIX (CONSUMER-SIDE, the lower-risk lever): copy the proven
    ``_layer10_alu_ordering_engine_rules`` ``hi_eq``/``lo_eq`` index-blocker
    pattern into the EQ engine's ``(h, l)`` units — a per-cell negative weight
    on every OTHER non-zero ``ALU_HI``/``ALU_LO`` index. When operand A's true
    high (or low) nibble is genuinely non-zero its strong ``+6.0`` one-hot is
    subtracted by the blocker (``-BLK * 6.0``) on every unit whose ``h``/``l``
    does NOT match A's nibble, so only the unit matching BOTH A's AND B's
    nibbles (i.e. A == B) survives. The weights/threshold were locked offline
    against the SAME golden HYBRID operand band the
    :class:`CmpOperandSeRecoverFFN` reconstructs in campaign (true nibble
    ``+6.0`` + index-0 artifact ``+5.3`` + cell-8/15 residues ``+0.45/+0.47``),
    exhaustively verified over the full ``0..99 × 0..99`` operand space: EVERY
    equal pair fires (margin ``+0.40``) and EVERY unequal pair is vetoed
    (worst-false margin on the if_eq corpus ``-0.87``). No ``CMP`` flag is
    touched (the ordering engine remains the sole CMP-flag writer), so
    lt/le/gt/ge/ne are structurally untouched; only the EQ ``eq_one`` OUTPUT
    push gains the high-nibble discrimination it lacked.

    DEFAULT ON. Opt-out via ``C4_CMP_EQ_HINIB_VETO=0`` restores the
    low-nibble-only EQ engine (byte-identical-OFF). Kept as a dedicated
    kill-switch so ``tools/flag_regression_gate.py --flag C4_CMP_EQ_HINIB_VETO``
    can A/B it inside the campaign config.
    """
    # Unconditional as of the P5 flag-retire 2026-07-14 (the former
    # ``C4_CMP_EQ_HINIB_VETO`` escape hatch was retired as a proven default-ON fix).
    return True


def cmp_hi_lt_alu15_leak_guard_enabled() -> bool:
    """Return True iff the L10 ordering-engine ``hi_lt`` (CMP+0) blocker DROPS its
    ``ALU_HI+15`` (0xF address-high-nibble) veto term in the campaign config
    (DEFAULT ON — opt-out via ``C4_CMP_HI_LT_ALU15_GUARD=0``; only takes effect
    when the STACK0 emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``, so flag-OFF
    / non-campaign is byte-identical to golden).

    The residual cmp wall this lifts (#339, GPU full_trace + isolated
    intervention, campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``,
    BUILT dims, spec_k=0, ``tools/probe_ifvar_result_step.py`` +
    ``tools/probe_ifvar_alu15_intervene.py``): the if_var GT-FALSE comparisons of
    a LOADED variable -- ``if_var 23>62`` (id430), ``35>76`` (id433), both
    ``A.hi < B.hi`` so the genuine ``hi_lt`` GT-FALSE override must fire -- decoded
    GT=1 (returns 1 instead of 0).

    ROOT. On the LOADED-variable path the result step's operand-A ``ALU_HI``
    carries a SPURIOUS ``+6.5`` at cell 15 (the 0xF address-high-nibble leak from
    the ``LI``-load relay; IMPOSSIBLE for the corpus operands, which are 0..99 so
    ``A.hi <= 6``). The ``hi_lt`` unit is a balanced AND
    ``MARK_AX(6.0) + ALU_HI[a](0.5) + AX_CARRY_HI[b](6.0)`` with a ``-0.5`` blocker
    on every OTHER ``ALU_HI`` cell. The ``-0.5 * 6.5 = -3.25`` cell-15 blocker term
    drops the unit's pre-activation to ``6 + 3 + 6 - 3.25 - 0.2 = 11.55`` -- BELOW
    its ``13.22`` threshold -- so ``hi_lt`` (CMP+0) does NOT fire and the GT result
    defaults to 1. The PASSING LITERAL path (``ifGT 23>62``) has a clean
    ``ALU_HI[15] ~= 0.5`` (blocker ``-0.25``), so its ``hi_lt`` sum ``14.55`` clears
    threshold and GT decodes 0. (Probed: FAIL result-step AX-row
    ``OUTPUT_LO=[0.3@0, 9.4@1]`` -> byte 1 WRONG; literal ``[15.1@0, -5.3@1]`` ->
    byte 0 correct. CMP cascade at that row: FAIL ``[1.2@3]`` (lo_lt only, no
    hi_lt); literal ``[1.0@0, 1.2@3]`` (hi_lt AND lo_lt).)

    FIX. Campaign-only: DROP the ``ALU_HI+15`` term from the ``hi_lt`` blocker.
    Cell 15 (``A.hi == 0xF``, i.e. operand-A high byte >= 0xF0 == value >= 240) is
    UNREACHABLE for every single-byte comparison in the corpus, so the cell-15
    veto only ever fires on the spurious 0xF leak -- never on a legitimate
    operand. With cell 15 excluded the leak no longer penalizes ``hi_lt``: the
    FAIL sum recovers to ``6 + 3 + 6 - 0.2 = 14.8 > 13.22`` -> ``hi_lt`` fires ->
    GT-FALSE override lands -> result 0. INTERVENTION-VERIFIED to be DISCRIMINATING
    (``probe_ifvar_alu15_intervene.py``): zeroing ``ALU_HI[15]`` FLIPS id430/433 to
    0 (FIXED) while the GT-TRUE ``85>48`` (id427) HOLDS at 1 and the GT-FALSE
    literal HOLDS at 0 -- no zero-sum trade. Only the ``hi_lt`` (CMP+0) blocker is
    touched; ``lo_lt`` / ``hi_eq`` / ``lo_eq`` (CMP+3/+1/+2) write strengths and
    blockers are UNTOUCHED, so lt/le/ge/eq/ne margins are unchanged. Band-local to
    the CMP engine (no shared OUTPUT/ALU read perturbed).

    DEFAULT ON — now UNCONDITIONAL under the campaign frame. The former
    ``C4_CMP_HI_LT_ALU15_GUARD`` escape hatch was RETIRED 2026-07-14 (proven
    default-ON, intervention-verified discriminating fix). Non-campaign
    (``C4_NO_STACK0_EMIT=0``) still restores the full cell-15 blocker
    (byte-identical to golden — the guard never fires off the campaign frame).
    """
    return no_stack0_emit_enabled()


def cmp_gt_lo_lt_hieq_guard_enabled() -> bool:
    """Return True iff the ``GT``/``GE`` ``(hi_eq AND lo_lt) -> 0`` override
    RAISES its firing threshold from 2.5 to 2.75 in the campaign config so a
    spurious ``lo_lt`` (CMP+3) alone can no longer trip the GT-result flip
    (DEFAULT ON; opt-out via ``C4_CMP_GT_LO_LT_HIEQ_GUARD=0``; only takes
    effect when the STACK0 emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1``,
    so flag-OFF / non-campaign is byte-identical to golden).

    The residual cmp wall this lifts (the last 6 if_var fails 425/436/440/441/
    445/448 -- the SYMMETRIC GT-TRUE companion to the GT-FALSE
    ``cmp_hi_lt_alu15_leak_guard``; GPU full_trace + isolated CMP+3 intervention,
    campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``, BUILT dims,
    spec_k=0, ``tools/probe_ifvar_gt_true.py`` /
    ``probe_ifvar_gt_true_intervene.py`` / ``probe_gt_cmp_values.py``): the
    if_var GT-TRUE of a LOADED variable whose operands satisfy ``A.hi > B.hi``
    but ``A.lo < B.lo`` (e.g. ``66 > 24`` id436: ``A=0x42`` so ``A.lo=2 <
    B.lo=8`` while ``A.hi=4 > B.hi=1``) decoded GT=0 instead of 1.

    ROOT. GT is true (high nibble already orders ``A > B``) so the result must
    default to 1. But on the LOADED-variable path the campaign
    ``CmpOperandSeRecoverFFN`` re-materializes the operand-A one-hot strongly
    enough that the genuine ``lo_lt`` (``A.lo < B.lo``) flag lands at the
    GT-result-step AX row as ``CMP+3 ~= 1.67`` (clean GT-TRUE id427 ``85>48``
    has ``A.lo=5 > B.lo=0`` so NO lo_lt -> CMP+3 ~= 0). The live GT decoder is
    the ``ComparisonCombine`` ``(hi_eq AND lo_lt) -> 0`` 3-way override (block
    28 unit 9, OP_GT-gated, reading raw CMP at the result row): it fires iff
    ``marker(1) + CMP+1 + CMP+3 - 0.1*CMP+0 - 2.5 > 0``. With ``CMP+1`` (hi_eq)
    == 0 (A.hi != B.hi) and ``CMP+3`` == 1.67 alone the sum ``1 + 0 + 1.67 -
    2.5 = 0.17 > 0`` SPURIOUSLY trips the override -> GT flips to 0 (probed
    OUTPUT_LO@AX-row = ``[9.1@0, 0.6@1]`` -> result byte low-nibble 0 -> GT=0
    WRONG; clean GT-TRUE / GT-FALSE refs ``[0.1@0, 9.6@1]`` / ``[18.7@0,
    -9.0@1]``). The GENUINE ``(hi_eq AND lo_lt)`` GT-FALSE -- e.g. literal eq-hi
    ``35>43`` (id350), ``A.hi==B.hi`` so ``CMP+1 ~= 1.24`` AND ``CMP+3 ~=
    1.46`` -- sums ``1 + 1.24 + 1.46 = 3.70``, well above threshold, and MUST
    still fire (GT=0 correct).

    FIX. Campaign-only: RAISE the GT (and the symmetric GE) ``(CMP+1, CMP+3)``
    override threshold from 2.5 to 2.75 -- the SMALLEST raise that rejects the
    spurious lo_lt-alone trip (the loaded-var GT-TRUE result-step firing sum is
    just above 2.5; 2.75 drops it below) while the genuine hi_eq+lo_lt
    GT-FALSE override (firing sum ~3.7) STILL fires. A larger raise (3.0) ALSO
    works on the result-step decode but its bigger ``b_up`` shift (-0.5*S vs
    -0.25*S) perturbs the fp-accumulation at the saturated-tie step-0 operand
    leak of two ALREADY-MARGINAL GT-FALSE programs (``if_gt 8>27`` id360,
    ``bool_and`` id1074, both decided by hi_lt) and FLIPS their exit-code;
    2.75 leaves the hi_lt-decided cases' 3-way override firing UNCHANGED (no
    residual perturbation) so if_gt/if_lt/if_eq/bool_and all HOLD 25/25.
    INTERVENTION + REBUILD-VERIFIED discriminating
    (``probe_gt_combine_threshold_patch.py`` + ``run_1096_canonical
    --criterion exit_code``): 2.75 flips the 6 loaded-var GT-TRUE
    425/436/440/441/445/448 to GT=1 (FIXED, isolated 6/6) while the GT-TRUE
    pass id427, the GT-FALSE pass id430, the genuine eq-hi GT-FALSE
    id350/353/365/368, the eq-hi GT-TRUE id357/359/361, the hi_lt GT-FALSE
    id360, and bool_and all HOLD -- no zero-sum trade.
    Only the GT/GE ``(CMP+1, CMP+3)`` overrides are touched; the GT/GE 2-way
    (hi_lt) override and EQ/NE/LT/LE overrides are UNTOUCHED, so lt/le/eq/ne and
    the hi_lt-driven GT/GE-false margins are unchanged.

    DEFAULT ON — now UNCONDITIONAL under the campaign frame. The former
    ``C4_CMP_GT_LO_LT_HIEQ_GUARD`` escape hatch was RETIRED 2026-07-14 (proven
    default-ON, rebuild-verified discriminating fix). Non-campaign
    (``C4_NO_STACK0_EMIT=0``) still restores the 2.5 threshold (byte-identical to
    golden — the raise never applies off the campaign frame).
    """
    return no_stack0_emit_enabled()


def cmp_combine_margin_enabled() -> bool:
    """Return True iff the ComparisonCombine result byte gets an explicit
    OUTPUT-HIGH-nibble CLAMP so the boolean result cannot leak an operand
    ``(hi<<4)`` high nibble (DEFAULT OFF; opt-in via ``C4_CMP_COMBINE_MARGIN=1``;
    only takes effect when the STACK0 emission is dropped, i.e.
    ``C4_NO_STACK0_EMIT=1``, so flag-OFF / non-campaign is byte-identical to
    golden ``91f55411``).

    The residual cmp wall this lifts (survey R5, ~150 programs — if_gt/if_lt/
    if_eq step-3, bool_and, func_max/min tail step-13). ROOT: the six
    comparison opcodes (EQ/NE/LT/GT/LE/GE) write their boolean RESULT to
    ``OUTPUT_LO`` (0 or 1) via the ComparisonCombine default+override banks, but
    the accompanying ``OUTPUT_HI_THIS_STEP+0`` write is only ``+2.0/S`` (default
    unit) and the OVERRIDE units write ``OUTPUT_LO`` ONLY — they never re-assert
    ``OUTPUT_HI``. On the both-nibbles-nonzero / loaded-var / re-materialized-
    operand paths a leaked operand HIGH nibble can survive in the OUTPUT_HI band
    at the compare decode row with a comparable (~1-point) amplitude, so the
    emitted result byte decodes as ``(hi<<4) | result`` instead of the clean
    ``result`` in {0, 1}.

    FIX (CLEAN MARGIN AT THE SOURCE, not a tail darken). A comparison result is
    PROVABLY in {0, 1}, so its byte's HIGH nibble is ALWAYS 0 — there is no
    operand-dependent case where a comparison opcode legitimately writes a
    non-zero ``OUTPUT_HI``. So per comparison opcode we add ONE campaign-only
    clamp unit that DARKENS every non-zero ``OUTPUT_HI_THIS_STEP+1..15`` nibble
    (strong negative) AND reinforces ``OUTPUT_HI_THIS_STEP+0`` (positive),
    out-voting any leaked operand high nibble at the decode row.

    SURGICAL GATE (2026-07). The clamp fires on
    ``MARK_SE_ONLY`` + ``SE_OP_<cmp>`` + **``SE_CMP_GROUP+0``** + the
    ``MARK_PC`` blocker (threshold 2.5 so ALL THREE positive markers are
    required, not just two-of-three). The added ``SE_CMP_GROUP+0`` condition is
    the FRESH-comparison discriminator: probing the SE step-end rows
    (``tools/_probe_cmp_se_row_disc.py``) shows ``SE_CMP_GROUP`` is ~0.94 on the
    genuine comparison-result decode row and ~0.00 on EVERY other SE row, while
    ``SE_OP_<cmp>`` ALONE can survive as a stale/cross-frame leak on a
    value-carrying row (a compare feeding a value — e.g. ``func_max``'s RETURN
    value ``0x63`` in a deeper 30-token frame). Without the group flag the clamp
    could clobber a legitimately-nonzero OUTPUT_HI high nibble on such a
    non-comparison / value row; with it the clamp fires ONLY when THIS step's
    opcode is a genuinely-relayed comparison whose result is provably in
    {0, 1}. This CANNOT weaken the intended fix — every genuine comparison
    decode carries ``SE_CMP_GROUP`` (the SAME L9 ``step_end_operand_relay`` that
    mirrors ``OP_<cmp> -> SE_OP_<cmp>``). Because the high nibble of a boolean
    result is invariantly 0, the clamp still cannot change any already-correct
    comparison; it only pulls a leaked ``(hi<<4)`` back to a clean low-nibble
    result. ``OUTPUT_LO`` (the actual 0/1 result) and every non-cmp opcode's
    OUTPUT_HI (which DO carry value high-nibbles) stay UNTOUCHED.

    DEFAULT OFF. Opt-in via ``C4_CMP_COMBINE_MARGIN=1`` (campaign only). Kept as
    a dedicated kill-switch so ``tools/flag_regression_gate.py --flag
    C4_CMP_COMBINE_MARGIN`` can A/B it inside the campaign config. Flag-OFF (or
    ``C4_NO_STACK0_EMIT=0``) is byte-identical to golden.
    """
    if not no_stack0_emit_enabled():
        return False
    return os.environ.get("C4_CMP_COMBINE_MARGIN", "0") == "1"


def func_lea_reread_bp_resharpen_enabled() -> bool:
    """Return True iff the L7 head-1 re-read-LEA BP-frame re-sharpen is active
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_FUNC_LEA_REREAD_BP_RESHARPEN=0``; only takes effect when the operand
    read is re-routed to ``mem[SP]`` i.e. ``C4_OPERAND_FROM_MEMSP=1``, so
    flag-OFF / non-campaign builds are byte-identical to golden ``7f6f2e5d``).

    The wall this lifts (GPU-confirmed, agent adc27e branch ``worktree-agent-
    adc27e4fcbefaa6fd`` commit ``57363ac4``; re-verified spec_k=0, BUILT dims,
    campaign config ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``;
    ``tools/_probe_lea_head1_attn.py``): the func_add/mul/square/max/min
    clusters (575-699) fail at the **re-read LEA** (the 2nd ``LEA n`` for a
    callee's later parameter — e.g. ``func_add`` step 11 ``LEA 16`` for &b,
    want ax=0xFFE0; value-INDEPENDENT).

    L7 head-1 (``layer7_operand_gather.head_1``) gathers the live frame BP/SP
    OUTPUT into the ALU at the LEA AX marker (operand-A address relay). Its K
    attends to BP/SP MARKER rows recency-ordered by the head's ALiBi slope
    (0.5). The BP marker row that actually CARRIES the live frame value is the
    one emitted on the frame-establishing ``ENT`` step (probed: ``OP_ENT=5.0``
    and ``OUTPUT_LO[0]~=14.9`` on that row ONLY; every other BP marker row is
    EMPTY — ``OP_ENT~=0``, ``OUTPUT_LO[0]~=1.0``). The FIRST LEA after the ENT
    works because the ENT-frame BP row is still the most-recent BP marker (it
    wins ALiBi recency). By the RE-READ LEA, three+ EMPTY BP marker rows
    (emitted by the intervening LI/PSH/LEA steps) sit between the query and the
    ENT-frame row, so ALiBi recency picks an EMPTY recent BP marker over the
    value-carrying ENT-frame row (probed: re-read row 471 raw 205 EMPTY beats
    value row 381 raw 160). The L8 LEA ALU is then EMPTY on the re-read → the
    address relay collapses to background residue (~15x attenuated).

    FIX (campaign-only, this flag): add ONE Q/K scoring slot to head-1, gated
    on the QUERY side by ``OP_LEA`` and on the KEY side by ``OP_ENT``. The slot
    contributes ``qw*kw * resid[q,OP_LEA] * resid[k,OP_ENT] * scale`` to the
    attention score, so it fires ONLY when the query is a LEA step AND the key
    is an ENT-frame BP row — re-pinning the re-read LEA's gather onto the LIVE
    ENT-frame BP row (it out-scores the empty recent markers). ALiBi recency
    breaks the tie between the two ENT-frame rows (caller `main` vs callee)
    toward the more-recent INNER frame, which is exactly the frame the callee's
    re-read LEA wants. On ADJ/ENT/non-LEA queries ``resid[q,OP_LEA]~=0`` so the
    slot is inert (ADJ's SP gather is untouched); on EMPTY BP / SP / non-ENT
    rows ``resid[k,OP_ENT]~=0`` so the slot adds nothing. This MIRRORS the
    first-LEA-after-ENT behaviour for the re-read LEA without disturbing any
    other gather.

    SHARED-HEAD DISCIPLINE: ``layer7_operand_gather.head_1`` is a golden head
    shared by ALL LEA/ADJ/ENT operand-A relays. The new Q/K writes land in a
    FRESH scoring slot (head_dim is 111; slots 0/1 are the only ones used by
    the head's Q/K scoring) and are ONLY emitted when this flag is on, so a
    flag-OFF build never touches ``W_q``/``W_k`` → byte-identical golden.
    ``tools/lint_cross_op_attention.py`` (MANDATORY for shared-head edits)
    gates the post-softmax head OUTPUT at OTHER-op / OTHER-context probe rows.

    DEFAULT ON. Opt-out via ``C4_FUNC_LEA_REREAD_BP_RESHARPEN=0`` (the
    byte-identical-OFF path: flag-OFF, or ``C4_OPERAND_FROM_MEMSP=0``, are both
    byte-identical to golden ``7f6f2e5d``). Kept as a dedicated kill-switch so
    ``tools/flag_regression_gate.py --flag C4_FUNC_LEA_REREAD_BP_RESHARPEN`` can
    A/B it inside the campaign config.
    """
    if not operand_from_memsp_enabled():
        return False
    return os.environ.get("C4_FUNC_LEA_REREAD_BP_RESHARPEN", "1") != "0"


def mul_l19_flood_cap_enabled() -> bool:
    """Return True iff the MUL L19-EXPLODE flood cap fires on MODERATE-magnitude
    (not just >100) wide_mul OUTPUT floods (DEFAULT ON in the campaign config —
    opt-out via ``C4_MUL_L19_FLOOD_CAP=0``; only takes effect when the STACK0
    emission is dropped, i.e. ``C4_NO_STACK0_EMIT=1`` AND the MUL byte-0
    SE-recover is on, since it reuses that path's clear+rewrite machinery).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; the 4 "L19-EXPLODE" mul
    fails 3*15, 11*11, 1*10, 8*30 whose product decodes to a HUGE garbage AX
    (e.g. 11*11 -> 2752768)):

    For a SMALL product the L11 ``efficient_l11_alumul_wrap`` wide_mul writes the
    CORRECT byte-0 product into OUTPUT_LO/HI but at a MODERATE flood magnitude
    (spec_k=0 block trace, 11*11: OUTPUT band sum ~41 at block 16 — the correct
    0x79 one-hot, but inflated). That ~41 band is BELOW the existing
    ``mul_byte0_se_recover`` flood-cap threshold (``output_hi_band > 100.0``), so
    the cap does NOT fire and the inflated raw OUTPUT survives. The downstream
    block-33 (logical L19) attention then AMPLIFIES that ~41 band to ~555
    (spec_k=0: ATTN in_LO=40.7 -> out_LO=555.6 at the MUL-AX row), spreading the
    band so the LM-head argmax flips to OUTPUT cell 0 == byte 0x00 and the AX
    high bytes pick up the flood -> the huge garbage AX. (Passing muls whose
    product writes a low ~14 band, e.g. 21*59, stay below the L19 amplification
    onset and decode correctly.)

    FIX. When this flag is on, the ``_MulCombineStage`` flood-cap threshold is
    LOWERED from ``> 100.0`` to ``> 4.0`` (well above the legitimate single-fire
    ``+2.0`` product write, below the ~41 small-product wide_mul flood) so the
    cap ALSO fires for the moderate-magnitude floods: it clears the inflated raw
    OUTPUT band on the OP_MUL+MARK_AX row and lets the clean ``+2.0`` byte-0
    product (the FlattenedALUMul schoolbook result, which is byte-0-correct for
    every mul-cluster operand — verified spec_k=0) survive. The normalized
    ``+2.0`` byte-0 product is below the L19 amplification onset, so the L19
    attention no longer explodes it. OUTPUT is byte-0 ONLY (byte 1 rides
    AX_FULL), so clearing+rewriting it never disturbs the byte-1 relay.

    DEFAULT ON. Opt-out via ``C4_MUL_L19_FLOOD_CAP=0`` restores the ``> 100.0``
    threshold (the byte-identical-OFF path: flag-OFF, or ``C4_NO_STACK0_EMIT=0``,
    or ``C4_MUL_BYTE0_SE_RECOVER=0`` are all byte-identical to golden). Kept as a
    dedicated kill-switch for ``tools/flag_regression_gate.py``.
    """
    # Unconditional as of the P5 flag-retire 2026-07-14 (the former
    # ``C4_MUL_L19_FLOOD_CAP`` escape hatch was retired as a proven default-ON fix).
    return True


def mul_l19_product_boost_enabled() -> bool:
    """Return True iff the capped-MUL byte-0 product is re-written at a DOMINANT
    OUTPUT amplitude that out-votes the block-33 (logical L19) zero-default add
    (DEFAULT ON in the campaign config — opt-out via
    ``C4_MUL_L19_PRODUCT_BOOST=0``; only takes effect under
    ``C4_NO_STACK0_EMIT=1`` + ``C4_MUL_BYTE0_SE_RECOVER=1``, since it rides the
    same ``_MulCombineStage`` cap path).

    The wall this lifts (verified spec_k=0, BUILT dims, campaign config
    ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``; the expr_add_mul cluster
    0/25 + the single-byte standalone-mul fails ``11*11`` etc.):

    ``mul_l19_flood_cap`` clears the L11 wide_mul OUTPUT flood on the
    OP_MUL+MARK_AX row and lets ``GEToBDConverter`` re-write the byte-0 product
    one-hot at the default ``+2.0`` amplitude. That fixed the explosion (band ->
    555) but NOT the SECOND L19 mechanism: the block-33 (logical L19) attention
    UNCONDITIONALLY ADDS ``+40`` into ``OUTPUT_LO[0]`` / ``OUTPUT_HI[0]`` (a
    broad "OUTPUT zero-byte default" copy, present on the MUL emit row in the
    depth>=1 stack contexts — expr_add_mul / expr_paren / expr_mul_div AND some
    standalone single-byte muls like ``11*11``). With the product at only ``+2.0``
    that ``+40`` cell-0 add OUT-VOTES the true product cell at the LM-head argmax,
    so the byte decodes to ``0x00`` (expr_add_mul ``5*2`` -> 0) or a stale value.
    Multi-byte products are NOT capped (band ~41 survives), so they already beat
    the ``+40`` add — only the capped single-byte products are starved.

    FIX. When this flag is on, the ``_GEToBDStage`` re-writes the freshly-cleared
    capped-row OUTPUT band (which by construction now holds ONLY the clean
    GEToBD product one-hot, the flood having been cleared) scaled to a DOMINANT
    magnitude (``MUL_L19_PRODUCT_BOOST`` = 50.0, > the L19 ``+40`` zero-default)
    so the true product cell beats the cell-0 add. Scoped to the cap rows ONLY
    (``output_clear_mask``), byte-0 ONLY (byte 1 rides AX_FULL, untouched), so it
    cannot disturb the byte-1 relay or any non-capped (multi-byte / non-MUL) row.

    DEFAULT ON. Opt-out via ``C4_MUL_L19_PRODUCT_BOOST=0`` restores the ``+2.0``
    re-write (flag-OFF, or ``C4_NO_STACK0_EMIT=0``, or
    ``C4_MUL_BYTE0_SE_RECOVER=0`` are all byte-identical to golden — the cap path
    is campaign-only). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py``.
    """
    # Unconditional as of the P5 flag-retire 2026-07-14 (the former
    # ``C4_MUL_L19_PRODUCT_BOOST`` escape hatch was retired as a proven default-ON fix).
    return True


def addsub_declarative_enabled() -> bool:
    """Return True iff efficient-mode L8 ADD/SUB uses the DECLARATIVE wrap
    (DEFAULT OFF — opt-in via ``C4_ADDSUB_DECLARATIVE=1``).

    When enabled, ``make_efficient_l8_addsub_wrap_op`` installs the
    ``DeclarativeAddSubBlock`` composite (two ``PureFFN`` passes, both lowered
    purely from ``wide_alu_dsl.wide_add_rules`` + ``wide_sub_rules``) instead
    of the imperative ``AddSub5StageBlock`` (BDToGEConverter + GE add/sub
    layers + GEToBDConverter). The declarative band writes OUTPUT at a
    DOMINANT amplitude (the folded-in +5 fix) so the correct result cell
    out-votes the downstream L9 ALU_LO->OUTPUT_LO leak. The two passes run
    sequentially inside ONE block (carry CARRY+0 cascades lo->hi internally),
    keeping the physical block count identical to the imperative.

    DEFAULT OFF (2026-06-14): the declarative byte-0 add/sub + carry/borrow
    flags are byte-identical to the imperative on CLEAN one-hot operands
    (tests/test_addsub_decl_wrap.py: full 0..255 value grid) AND in isolation
    on the real model. BUT the live MARK_AX operand bands are DIRTY (operand A
    in ALU_LO/HI arrives ~6.0 at the true nibble plus a ~5.4 index-0
    magnitude artifact — the documented operand-gather hybrid encoding). On
    those dirty bands the lo pass's CARRY+0 over-accumulates (multiple partial
    rule firings) and the hi pass over-fires, so multi-byte add/sub (the
    add_16bit / sub_16bit / *_cascade smoke + 32-bit negative sub) regress.
    The imperative ``BDToGEConverter._clean_onehot`` thresholds operands to
    0/1 BEFORE computing, which is what makes it dirty-operand robust; the
    declarative wrap needs an analogous operand-cleanup pre-pass (the same
    technique ``make_efficient_l10_andorxor_wrap_op`` uses for bitwise) as a
    FOLLOW-UP wave before it can be the default. See
    docs/ADDSUB_DSL_MIGRATION_2026_06_14.md.
    """
    return os.environ.get("C4_ADDSUB_DECLARATIVE", "0") == "1"


def operand_from_memsp_enabled() -> bool:
    """Return True iff the binary-op operand-A read is routed to ``mem[SP]``
    via the L4 SP-to-ADDR_KEY + L8 mem-to-ALU memory-attention CAM
    (DEFAULT OFF — opt-in via ``C4_OPERAND_FROM_MEMSP=1``).

    This is the STACK0-emission-drop *prerequisite*. The binary-op operand-A
    (the top-of-stack value that ADD/SUB/AND/OR/XOR/EQ/NE/SI/SC read) is
    sourced today from the EMITTED STACK0 byte-0 token via L7 head 0's
    ``STACK0_BYTE0``-keyed gather. When ``C4_NO_STACK0_EMIT=1`` drops the
    STACK0 register block from the step, that emitted token is gone and the
    operand read is starved (-130 full_trace, see
    ``docs/NO_STACK0_EMIT_MEASURED_2026_06_16.md``).

    When this flag is on:
      * ``make_layer4_sp_to_addr_key_op`` (L4 attn heads 2-3) stages the live
        SP value into the ADDR_KEY band at the AX marker (Q-side address).
      * ``make_layer8_mem_to_alu_op`` (L8 attn head 5) reads ``mem[SP]`` byte 0
        via the ADDR_KEY CAM (most-recent matching ``MEM_STORE`` wins via
        ALiBi recency) and writes ``ALU_LO/HI`` at the AX marker — the exact
        source/dest L7 head 0 uses today, one block later, sourced from
        memory instead of the emitted token.
      * L7 head 0's ``STACK0_BYTE0`` -> ALU write is suppressed (the L8 head
        is the sole operand-A source; head 1's LEA/ADJ/ENT frame relay is
        untouched).

    This implements Phase 1 of ``docs/STACK0_VIA_MEM_ATTENTION_PLAN.md``
    (the L4 + L8 ops were built but left ``enable=False`` / unvalidated).
    DEFAULT OFF = byte-identical to HEAD. Flip on TOGETHER with
    ``C4_NO_STACK0_EMIT`` to eliminate the Root #2 framing-drift class without
    starving the operand read. With ``C4_NO_STACK0_EMIT=0`` (STACK0 still
    emitted) it is a no-regression equivalence check (operand now read from
    ``mem[SP]`` instead of the still-emitted token).

    Campaign entry point: ``C4_CAMPAIGN=1`` supplies the ON floor (see
    ``campaign_enabled``); an explicit ``C4_OPERAND_FROM_MEMSP=0`` still opts out.
    """
    if os.environ.get("C4_OPERAND_FROM_MEMSP") is None and campaign_enabled():
        return True
    return os.environ.get("C4_OPERAND_FROM_MEMSP", "1") != "0"


def no_stack0_emit_enabled() -> bool:
    """Return True iff the STACK0 register block is dropped from the emitted
    step (DEFAULT OFF — opt-in via ``C4_NO_STACK0_EMIT=1``).

    Mirror of ``l0_ops._no_stack0_emit`` (kept here so the L1 positional-gate
    fix and other ops can consult the same env switch without importing l0).
    When on, the 35-token step collapses to 30 tokens (STACK0 marker + 4 value
    bytes dropped), so the MEM register block shifts 5 positions earlier:
    the d=6-from-BP slot that used to be STACK0 byte 0 is now MEM addr byte 0.
    The ``STACK0_BYTE0`` positional flag (L1 FFN unit 0, fired at d=6 from BP)
    must therefore be neutralized so it does not misfire onto the MEM addr
    byte — see ``l1_ops._threshold_ffn_rules``.

    This flag does NOT itself drop the emission (that machinery — the L0
    marker-transition chain, ``Token.STEP_TOKENS``, the DraftVM oracle, the
    decode offsets — lives on the ``proto/drop-stack0-emit-measure`` branch).
    It is the per-op consultation point so flag-off is byte-identical.

    Campaign entry point: ``C4_CAMPAIGN=1`` supplies the ON floor (see
    ``campaign_enabled``); an explicit ``C4_NO_STACK0_EMIT=0`` still opts out.
    """
    if os.environ.get("C4_NO_STACK0_EMIT") is None and campaign_enabled():
        return True
    return os.environ.get("C4_NO_STACK0_EMIT", "1") != "0"


def emit_g5_rbyte_enabled() -> bool:
    """Return True iff the EMIT-G5 R-BYTE ``ax_bytes_zero`` fold routes through
    the ALTERNATE (per-op copy-loop) assembly path (DEFAULT OFF — opt in via
    ``C4_EMIT_G5_RBYTE=1``).

    BYTE-NEUTRAL VERIFICATION TOGGLE. The four structurally-identical L14
    ``layer14_{jsr,lc,alu_nocarry,ent}_ax_bytes_zero`` cleanup ops were folded
    into ONE spec-driven generator (``_ax_bytes_zero_rules`` /
    ``_make_ax_bytes_zero_op`` in ``l14_ops.py``; see
    ``docs/EMIT_G5_ROLLOUT_2026_07_13.md``). The four opcode gates
    (``OP_JSR`` / ``OP_LC_RELAY`` / ``TEMP+7`` / ``OP_ENT``) are proven MUTUALLY
    EXCLUSIVE — each is a distinct L5 one-hot ``OP_*`` decode (or its L7 relay),
    and no single instruction is two opcodes at once — so the four ops never
    co-fire on any row.

    When ON, each op's 4-unit rule program is rebuilt through a SECOND
    independent ``_ax_bytes_zero_rules(spec, S)`` call and asserted structurally
    identical to the first before lowering. Because the rules are PURE FUNCTIONS
    of the spec (no unit-index / no environment state), this exercises the spec
    table via a distinct code path without touching a single weight -> golden
    ``e50521f3`` is byte-identical flag-OFF and flag-ON. The toggle exists so the
    fold's spec-driven generator can be re-run as a regression guard in CI
    without a weight change.
    """
    return os.environ.get("C4_EMIT_G5_RBYTE", "0") != "0"


def si_store_addr_enabled() -> bool:
    """Return True iff the SI/SC store address-provenance CAM head is active
    (DEFAULT OFF — opt in via ``C4_SI_STORE_ADDR=1``; only meaningful in the
    30-token campaign config).

    ROOT (the ``var_mul``/multilocal-LI store-provenance TWO-ROOT wall, verified
    2026-06-18 by built-layout probe + hook-inject; ``var_mul`` id275
    ``int a;int b;a=23;b=47;return a*b;`` fails: ``LI a`` returns b's 47 not a's
    23). The existing L15 head-0 LI value-load CAM (``C4_L15_LI_ADDR_CAM``)
    discriminates two same-frame locals by matching ``ADDR_B0_LO/HI`` on each
    store's VALUE row and copies that row's ``CLEAN_EMBED`` into OUTPUT. But
    (Root 1) the SI/SC store VALUE rows carry ``ADDR_B0==0x00`` (address-blind)
    so the candidates tie and ALiBi recency picks the last store (b); AND (Root
    2) even with the correct ``ADDR_B0`` injected the value rows carry an
    address-like ``CLEAN_EMBED`` pattern, NOT the clean value 23. Provenance is
    missing on BOTH axes on the exact rows the CAM reads.

    THE LEVER (built-layout probe, ``tools/_probe_si_store_addr.py``, campaign
    config, id275): the clean address AND clean value BOTH live on the store's
    AX-MARKER row (a-marker: ``ADDR_B0=0xE8`` + ``AX_CARRY=0x17``=23; b-marker:
    ``ADDR_B0=0xE0`` + ``AX_CARRY=0x2F``=47), and the ``LI a`` query row carries
    its target address in ``AX_CARRY=0xE8`` (NOT in ``ADDR_B0``, which is 0x00
    there). So a DIRECT address-keyed CAM bypasses BOTH roots: LI-query row Q
    keyed on its ``AX_CARRY`` (target addr) -> SI/SC store AX-marker K keyed on
    the marker's ``ADDR_B0`` (store addr) -> V copies the marker's ``AX_CARRY``
    (the clean stored value) -> O writes the LI result into OUTPUT.

    Implemented as a dedicated flag-gated L15 attention head
    (``layer15_si_store_addr_cam``) appended to the L15 head layout only when the
    flag is on, so flag-OFF keeps the L15 head count + weights byte-identical to
    golden (the CAM never installs). Campaign-only: the ``AX_CARRY``/``ADDR_B0``
    marker signals it keys on are produced by the 30-token MEM-from-SP path;
    golden (35-token, flag-OFF) is byte-identical.

    Campaign entry point: ``C4_CAMPAIGN=1`` supplies the ON floor (see
    ``campaign_enabled``); an explicit ``C4_SI_STORE_ADDR`` value wins (so
    ``C4_SI_STORE_ADDR=0`` opts out even under the campaign).
    """
    explicit = os.environ.get("C4_SI_STORE_ADDR")
    if explicit is None and (campaign_enabled() or derive_memory_enabled()):
        return True
    return explicit == "1"


def var_three_li_enabled() -> bool:
    """Return True iff the L15 head-0 OP_SI/OP_SC store-row VETO is active
    (DEFAULT OFF — opt in via ``C4_VAR_THREE_LI=1``; only meaningful in the
    30-token campaign config where the stray relay flag appears).

    ROOT (verified spec_k=0, BUILT dims, campaign config, teacher-forced oracle
    tape, ``tools/_probe_vt_head0_sistore.py``; var_three id300
    ``int a;int b;int c;a=29;b=6;c=20;return a+b+c;`` diverges at the ``SI b``
    store step): the ``SI b`` store's AX byte-0 emit row carries a STRAY
    ``OP_LI_RELAY==1.0`` alongside ``OP_SI==5.23`` (unlike the ``SI a`` / ``SI c``
    store rows, which carry ``OP_SI`` alone). The L15 memory-lookup head 0
    (``li_lc_stack0_h0``, value_scale=40) slot-0 discriminator fires the lookup
    on ``OP_LI_RELAY`` (Q weight ``lookup_bias``=2e5) and already VETOES the
    non-load opcodes ``OP_JSR/OP_ENT/OP_LEA/OP_IMM`` at ``-1e6`` — but NOT the
    STORE opcodes ``OP_SI``/``OP_SC``. So on the ``SI b`` row the stray relay
    clears the veto, head 0 fires OFF-self (attends a cross-step ``CLEAN_EMBED``
    row whose value is 0), and copies ``+0`` into ``OUTPUT``: the store value
    byte 6 decodes ``0`` instead of ``6`` -> the frame desyncs at ``SI b`` and
    the whole var_three program drifts (probe: ``SI b`` row got=0x0000 want
    0x0006; the ``SI a``/``SI c`` rows head0 self_p=1.000, correct).

    FIX. A genuine LI/LC LOAD never has ``OP_SI``/``OP_SC`` hot at its own
    marker (the opcode is OP_LI/OP_LC), whereas an SI/SC store step has exactly
    one of ``OP_SI``/``OP_SC`` one-hot at MARK_AX. Extend the SAME slot-0
    ``non_load_suppression`` veto (already covering OP_JSR/ENT/LEA/IMM and — via
    ``C4_L15_LOOKUP_CMP_VETO`` — the six comparison opcodes) to ``OP_SI`` and
    ``OP_SC``: ``OP_SI*-1e6`` dominates the stray ``OP_LI_RELAY*2e5``, keeping
    head 0 silent (self-firing) on every store row so the store value survives.
    Scoped to head 0's slot-0 Q only; no V/O / scale change, so the LI/LC/POP
    delivery on real load rows is untouched (byte-identical there —
    OP_SI==OP_SC==0 on every load row). Additive slot-0 Q writes, gated by the
    flag, so flag-OFF omits them entirely -> byte-identical to golden.

    DEFAULT OFF. Reads its OWN env var (NOT floored ON by ``C4_CAMPAIGN``) so
    the campaign default set stays untouched until this fix is proven
    net-positive + HOLD-clean; opt in via ``C4_VAR_THREE_LI=1`` (only takes
    effect under the campaign / MEM-from-SP path, since the stray relay flag is
    produced only by the 30-token frame — flag-ON at golden 35-tok is
    byte-identical). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py`` and the flag-OFF golden byte-identity gate.

    ``C4_DERIVE_MEMORY=1`` (the derived-CAM MEMORY umbrella,
    :func:`derive_memory_enabled`) supplies an ON floor -- this store-row veto is
    part of the clean binary-address-CAM memory-fix path the umbrella activates
    -- but an explicit ``C4_VAR_THREE_LI`` value still wins
    (``C4_VAR_THREE_LI=0`` opts out even under the umbrella). DEFAULT OFF ->
    byte-identical golden.
    """
    explicit = os.environ.get("C4_VAR_THREE_LI")
    if explicit is None and derive_memory_enabled():
        return True
    return explicit == "1"


def li_value_load_enabled() -> bool:
    """Return True iff the L15 SI-store-addr CAM (head 16) NULL-HIGH-ADDRESS
    candidate VETO is active (DEFAULT OFF — opt in via ``C4_LI_VALUE_LOAD=1``;
    only meaningful in the 30-token campaign config where head 16 installs).

    ROOT (verified spec_k=0 teacher-forced, BUILT dims, campaign config,
    ``tools/_probe_l15_perhead_output.py``; func_max id650 / func_min id675
    ``int min(int a,int b){if(a<b)return a;return b;} main(){return min(13,57);}``
    LI that loads operand ``b`` at addr 0xE0): the head-16 SI-store-addr CAM
    (``layer15_si_store_addr_cam``, ``C4_SI_STORE_ADDR``, value_scale=60) is
    designed for the RELATIVE-store case (var_mul: local ``b`` IS ``SI``-stored,
    so a store AX-marker carries ``ADDR_B0=0xE0`` + ``AX_CARRY``=value). For a
    CALL-SITE-arg LI (func_max/min args come from a PSH at the call site, NO
    matching SI store exists in the callee frame), the CAM's per-nibble address
    match fires on the callee ENT-frame PHANTOM row (id675 pos 269:
    ``ADDR_B0=0x00``, ``AX_CARRY=0x0d``=13=``a``, OP_ENT) via a spurious
    LO-NULL-nibble match (query addr 0xE0 has LO nibble 0, which slot-16
    positive-matches the phantom's ``ADDR_B0_LO+0``; the query HI nibble 14 does
    NOT match the phantom's ``ADDR_B0_HI+0``, but the partial LO-only match +
    firing baseline still WINS). head 16 then delivers ``a``'s value into OUTPUT
    at value_scale 60, DOMINATING head-0's CORRECT value (value_scale 40) at the
    argmax. It fails visibly only when the phantom value's byte lands on the same
    nibble-0 default the LM head decodes as 0x00 (id675 ``a``=13=0x0d has HI
    nibble 0 -> HI nibble-0 = 66.5 buries head-0's correct HI nibble-3 = 40.0 ->
    the ``b`` LI decodes 0x00 not 0x39); func_max ``a``=36=0x24 has HI nibble 2,
    so it lands off nibble-0 and head-0 wins BY LUCK (the byte0-right/byte0-wrong
    asymmetry across arg values is the fingerprint).

    FIX. A genuine local-store target ALWAYS has a NON-null high address nibble
    (stack locals live at 0xE8/0xE0/0xD8/... -> ``ADDR_B0_HI`` = 14/13/...); only
    the callee ENT/phantom rows carry ``ADDR_B0=0x00`` (``ADDR_B0_HI+0`` hot).
    Add a K-side candidate VETO on ``ADDR_B0_HI+0`` (gated on the OP_LI query, so
    it applies only at the LI lookup): a candidate with a NULL high address
    nibble is driven below the softmax1 CONST sink, so head 16 fails-CLOSED
    (writes ~0) on the phantom and head-0's genuine value survives. GENUINE
    relative stores (0xE_/0xD_ HI nibble) are UNTOUCHED (``ADDR_B0_HI+0``==0
    there, so the veto contributes 0), so var_mul / var_three / var_update keep
    the store-addr CAM's value delivery byte-identically. The abs-address LI
    path (``LI 0x200`` -> store marker ``ADDR_B0_HI+0`` hot) is ALREADY
    fail-closed by the slot-63 ``LI_QUERY_ZEROADDR`` query-side veto, so this
    K-side veto is redundant-safe there.

    DEFAULT OFF. Reads its OWN env var (NOT floored ON by ``C4_CAMPAIGN``) so the
    campaign default set stays untouched until this fix is proven net-positive +
    HOLD-clean; opt in via ``C4_LI_VALUE_LOAD=1`` (only takes effect under the
    campaign / MEM-from-SP path, since head 16 installs only when
    ``C4_SI_STORE_ADDR`` is on — floored ON by ``C4_CAMPAIGN``; the veto slot is
    written into head 16 only when this flag is on, so flag-OFF omits it ->
    byte-identical to golden). Kept as a dedicated kill-switch for
    ``tools/flag_regression_gate.py`` and the flag-OFF golden byte-identity gate.
    """
    return os.environ.get("C4_LI_VALUE_LOAD", "0") == "1"


def absdiff_fix_enabled() -> bool:
    """Return True iff the absdiff arg-b LI value byte-0 LO-nibble
    de-contamination is active (DEFAULT OFF — opt in via ``C4_ABSDIFF_FIX=1``;
    only meaningful in the 30-token campaign config).

    ROOT (verified spec_k=0 AUTHORITATIVE AR verdict via ``tools/cpu_full_trace``
    AND teacher-forced ``tools/_probe_vt_head0_sistore.py`` + per-block
    OUTPUT_LO attribution, campaign config, BUILT dims; absdiff_0 id1046
    ``int abs_diff(int a,int b){if(a>b)return a-b;return b-a;} main(){return
    abs_diff(16,85);}`` diverges at step 12, the 2nd LI = the arg-``b`` deref
    inside the ``if(a>b)`` comparison): ``exp=(pc=74,ax=85) got=(pc=74,ax=80)``.
    PC is correct; only AX byte-0's LO nibble is wrong (0x55 -> 0x50).

    Arg ``a`` lives at BP-relative 0xFFE8 (addr byte-0 = 0xE8, lo-nibble 8) and
    arg ``b`` at 0xFFE0 (addr byte-0 = 0xE0, lo-nibble 0). The L15 head-0
    memory-lookup value delivery (value_scale=40) correctly content-addresses
    ``b``'s pushed value row (CLEAN_EMBED=0x55) and delivers OUTPUT_LO+5 = +40,
    but at this ZERO-lo-nibble arg address the same value row's CLEAN_EMBED_LO
    ALSO carries the ADDRESS lo-nibble 0, so head-0 delivers OUTPUT_LO+0 = +67.5
    as well. Attribution proves BOTH writes originate in block 35 (= logical
    L15). At the LM head the byte-0 token 0x50 (hi=5 correct, lo=0 contaminated)
    wins at logit 542 over the correct 0x55 at logit 403 (margin ~139), so byte-0
    decodes ``value & 0xF0``. This affects ALL 25 absdiff cases (the arg-``b``
    deref precedes the branch, so both taken and not-taken paths corrupt).

    This is the documented L15 head-0 LI value-CAM zero-lo-nibble aliasing WALL:
    the value nibble and the address nibble ALIAS in CLEAN_EMBED at a
    zero-lo-nibble arg address, and var_simple's ``x`` at BP+0 (addr 0x00) shares
    the exact separating signature, so every head-0 slot discriminator that
    flips absdiff also regresses var_simple (blueprint, 3 attempts). The clean
    fix needs an FFN-materialized (committed-zero-address) indicator dim so the
    AND is done outside the bilinear head — a multi-block build, NOT a single
    additive slot. DEFAULT OFF; flag-OFF registers NO rules -> byte-identical to
    golden ``b1dcae63``. Kept as a dedicated kill-switch for the flag-regression
    gate and the byte-identity gate.

    STATUS (measured, NOT net-positive — kept DEFAULT-OFF): the L10 tail-block
    ``l10_absdiff_argb_li_lo`` corrector (a per-nibble OUTPUT_LO winner-take-all
    gated on the head-0-delivered value LO nibble ``OUTPUT_LO+k`` + the 0xE0
    address discriminator) was BUILT and measured on the authoritative AR
    verdict (``cpu_full_trace --spec-k 0``). It REGRESSES its own target:
    absdiff_0 goes from ``div_step=12`` (byte-0 lo nibble) OFF to ``div_step=1``
    (``exp=(pc=210,ax=0) got=(pc=210,ax=2)``) ON — the ``OUTPUT_LO+k`` gate is a
    RESIDUAL value that is nonzero on non-value-delivery rows too, so the
    corrector over-fires at an earlier step. This CONFIRMS the wall: without an
    FFN-materialized indicator dim, the value LO nibble cannot be re-stamped
    cleanly (it aliases with residual OUTPUT_LO mass on other rows AND with the
    address-0 contaminant on the value row). The corrector stays flag-OFF; do
    NOT flip ON. Next session: build the (committed-zero-address) indicator dim
    at L13/L14 and gate a MARK_AX-only head-0 slot on it (blueprint).
    """
    return os.environ.get("C4_ABSDIFF_FIX", "0") == "1"


def absdiff_ret_byte1_enabled() -> bool:
    """Return True iff the absdiff / func-return AX byte-1 OUTPUT_LO stale-marker
    de-contamination is active (DEFAULT OFF — opt in via
    ``C4_ABSDIFF_RET_BYTE1=1``; only meaningful in the 30-token campaign config).

    NOTE (2026-07 round-1 land): held DEFAULT-OFF — flipping it ON regressed the
    ``test_lea_basic`` smoke (``ENT; IMM 0; LEA 2; EXIT`` -> address decodes 0):
    the ``ABSDIFF_RET_LEAK`` crush-band / OP_LEV-return selector over-fires on
    the ENT-frame LEA row and zeroes the real address OUTPUT_LO nibble. Needs a
    surgical re-gate (exclude non-LEV LEA rows) before it can go default-ON.

    ROOT (block-input attribution via ``tools/interp_oracle_gate`` +
    ``tools/_probe_absdiff_ret_byte1.py``, spec_k=0 campaign, BUILT dims,
    golden ``f725c06e``; absdiff_0 id1046, absdiff_7 id1053, absdiff_8 id1054):
    on the ADJ step that IMMEDIATELY FOLLOWS the ``abs_diff`` LEV (function
    return) — the failure-map "step-22" — the AX byte-1 predictor row emits
    token ``0x01`` instead of ``0`` (``got_ax_bytes=[69,1,0,0]`` for
    ``|16-85|=69``). The AX byte VALUE is decoded from the ``OUTPUT_LO`` nibble
    one-hots (``W[1]-W[0]`` peaks at ``OUTPUT_LO+1`` / ``OUTPUT_LO+0``), and at
    the leak row ``OUTPUT_LO+0`` is crushed to ``-199.5`` while ``OUTPUT_LO+1``
    rises to ``+9.1`` -> the argmax nibble is 1 -> byte 0x01.

    The runtime-dominant writer of ``OUTPUT_LO+1`` there is
    ``layer6_routing_ffn :: l6_psh_stack0_marker_final_lo_1`` (contrib +9.94 vs
    the clean_emitter default +0.96). That L6 rule is a 3-way AND
    ``PSH_AT_SP AND MARK_STACK0 AND ALU_LO+1`` (threshold 2.5, unit weights) that
    is only meant to fire at a genuine PSH-STACK0 marker step. On the post-LEV
    ADJ step BOTH ``PSH_AT_SP`` and ``MARK_STACK0`` are 0, but ``ALU_LO+1`` reads
    a NON-BINARY ``5.81`` (stale ALU low-nibble after the SUB) which alone clears
    the AND threshold — so the AND mis-fires and stamps the value-1 nibble into
    ``OUTPUT_LO``. This is NOT the ``H*_DUMP_OUT`` byte-1 dump path (already
    killed at LEV by ``C4_LEV_AX_BYTE1_KILL``); it is the parallel OUTPUT-band
    decode leak that still wins.

    FIX (this flag, correct-by-construction, CLEAN DISCRIMINATOR): on the AX
    value-byte rows of the LEV-return step — ``OP_LEV`` (the return-context
    opcode marker, > 0.8 ONLY on the ADJ/return step, PATH-INDEPENDENT across
    both the ``a>b`` true and false absdiff paths, and decaying AX~1.1 > SP~0.9 >
    BP~0.6 across the step) AND ``IS_BYTE`` (a value byte, excludes the
    byte-0/MARK_AX row so byte-0 keeps its real nibble) AND the ELEVATED
    ``OUTPUT_LO+k`` leak itself (additive threshold, so the ~1.0 clean nibble
    baseline stays dark) AND ``NOT PSH_AT_SP`` AND ``NOT MARK_STACK0`` (the L6
    marker rule's own gates are provably OFF here, so its OUTPUT_LO write is
    spurious) — for each nibble k in 1..15, write ``-DOM`` to ``OUTPUT_LO+k`` and
    ``+DOM`` to ``OUTPUT_LO+0`` so the AX high bytes decode nibble-0 -> byte
    value 0. (An earlier ``H2_PREV_STEP+0`` discriminator was PATH-DEPENDENT — 0
    on the ``a>b`` true path — and over-fired on the clean nibble baseline via a
    multiplicative gate; the current OP_LEV + additive-threshold form is the
    robust one.)

    NOT-a-multi-byte GUARD (the bounded ``ABSDIFF_RET_LEAK`` l6-crush flag):
    ``func_mul`` returns a PRODUCT that can EXCEED 255 (e.g. mul(49,36)=1764,
    byte-1=6; mul(17,21)=357, byte-1=1), so its byte-1 is REAL and must NOT be
    zeroed. func_mul shares the LEV-return-step OP_LEV signature, so OP_LEV alone
    wrongly fires there (regressing e.g. func_mul_3 PASS->FAIL). The separator is
    the l6-CRUSH: on the absdiff single-byte leak the mis-firing
    ``l6_psh_stack0_marker_cancel_output`` sub-loop CRUSHES ``OUTPUT_LO+0``
    NEGATIVE (measured -46..-199 on ALL 25 absdiff at the ADJ AX byte-1 row)
    while a genuine func_mul return leaves it CLEAN POSITIVE (+3.91) OR SATURATES
    it hugely-negative (~-6e9). A PRECURSOR FFN writes a BANDED indicator
    ``ABSDIFF_RET_LEAK = step(-(OUTPUT_LO+0) >= 10) - 2*step(-(OUTPUT_LO+0) >=
    1000)``, and the corrector MULTIPLICATIVELY gates on it: clean-POSITIVE
    OUTPUT_LO+0 (func_mul +3.91) -> flag 0 -> gate 0 -> NO write (legit byte-1
    preserved); MODERATE crush (absdiff -46..-199) -> flag > 0 -> the correction
    fires (byte-1 -> 0); SATURATED crush (genuine multi-byte func_mul ~-6e9, e.g.
    mul(26,33)=858) -> the 2x hi-cancel DRIVES the flag NEGATIVE -> the corrector's
    sign-flipped writes stay proportional to the ~6e9 delivered nibble so the
    genuine byte-1 is PRESERVED (verified: passing func_mul id603/605/608 all stay
    OK). The 2x cancel is load-bearing: silu is linear (not a true step) so an
    equal-weight cancel would leave a positive residual at saturation that zeroes
    the genuine byte-1. The raw (ungated) crush value was rejected: it let the
    huge SP/BP/LEA/func_mul crushes amplify the multiplicative gate.

    SAFETY: every absdiff / func_identity / nested single-byte return is <= 255,
    so AX bytes 1..3 are 0 at the return step — forcing OUTPUT_LO to nibble-0 on
    the l6-crushed rows is exactly correct (same invariant ``C4_LEV_AX_BYTE1_KILL``
    relies on), and the crush guard excludes the genuine multi-byte func_mul
    case. The OP_LEV>=~0.95 selector + crush term keep this off every non-AX byte
    row (SP OP_LEV<=0.909 -> score<threshold; BP<=0.708) and off the pre-LEV
    steps (OP_LEV=0); ``IS_BYTE`` keeps byte-0's real nibble; the marker NOT-gates
    keep it off a genuine PSH-STACK0 rewrite. DEFAULT OFF; flag-OFF registers NO
    rules -> byte-identical to golden ``f725c06e``. Kept as a dedicated
    kill-switch for the flag-regression gate and the byte-identity gate.
    """
    return os.environ.get("C4_ABSDIFF_RET_BYTE1", "0") == "1"


def clean_operand_enabled() -> bool:
    """Return True iff the DERIVED clean-one-hot operand delivery is active
    (``C4_CLEAN_OPERAND`` — DEFAULT-OFF feasibility flag).

    Installs :class:`CleanOperandOneHotFFN` on the L8 main FFN (physical block
    11, the operand-delivery block). On the binary-op / cmp MARK_AX rows it
    snaps the ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) bands to a
    clean per-nibble one-hot (keep the argmax at ~6.0, zero the cell-8/cell-0
    residues + address two-hot + index-0 artifact). This is the correct-by-
    construction generalisation of ALL the per-op address-leak / hybrid-rebuild
    correctors (``LoadedOperandAddHi15ClearFFN``, ``CmpOperandSeRecoverFFN``,
    the func-add hi-nibble clears).

    Feasibility question (CBC Phase 1): does a CLEAN operand let the downstream
    ALU carry side-signal + CMP decode compute correctly WITHOUT the correctors?
    Gated behind the ``no_stack0_emit`` campaign flag so the flag-OFF golden
    (``e50521f3``) build is byte-identical (the wrap is never installed off the
    campaign / off the flag).
    """
    if os.environ.get("C4_CLEAN_OPERAND", "0") == "0":
        return False
    return no_stack0_emit_enabled()


def clean_operand_add_enabled() -> bool:
    """Return True iff the ARITHMETIC-ONLY clean-one-hot operand delivery is
    active (``C4_CLEAN_OPERAND_ADD`` — DEFAULT-ON, CBC first correct-by-
    construction pass-gain; opt out with ``=0``).

    This is the narrowed sibling of ``clean_operand_enabled`` (``C4_CLEAN_OPERAND``).
    It installs the SAME :class:`CleanOperandOneHotFFN` wrap on the L8 main FFN,
    but the clean-snap fires ONLY on the ARITHMETIC opcode rows
    (``OP_ADD/OP_SUB/OP_MUL/OP_DIV/OP_MOD``) — the CMP/EQ/LT/GT/bool consumer
    rows (``OP_EQ..OP_GE``) are left byte-identical.

    RATIONALE (see docs/CLEAN_OPERAND_FEASIBILITY_2026_07_10.md): a clean
    operand FIXES arithmetic (the ADD inter-byte carry stays correct → +6
    flips) but BREAKS the CMP path (the CMP nibble comparators' ``-0.5/-0.8``
    per-nibble blockers are a magnitude CONTRACT calibrated to the dirty
    hybrid; a perfect one-hot overshoots the lt/eq decode window → −23 CMP
    regressions). Gating the clean-snap to the arithmetic opcodes ONLY captures
    the arithmetic gain with ZERO CMP regression (the CMP calibration contract
    is left intact), so this slice is byte-identical-OFF AND net-positive-ON.

    Gated behind the ``no_stack0_emit`` campaign flag so the non-campaign golden
    (``e50521f3``) build is byte-identical (the wrap is never installed off the
    campaign). The former ``C4_CLEAN_OPERAND_ADD`` escape hatch was RETIRED
    2026-07-14 (proven default-ON pass-gain); the fix is now unconditional under
    campaign.
    """
    return no_stack0_emit_enabled()


def clean_operand_bitwise_enabled() -> bool:
    """Return True iff the BITWISE (OP_AND/OP_OR/OP_XOR) clean-one-hot operand
    delivery is active (``C4_CLEAN_OPERAND_BITWISE`` — DEFAULT-ON in the campaign
    config; opt out with ``=0``).

    Extends the :class:`CleanOperandOneHotFFN` op-dim gate at the L8 main FFN
    (physical block 12, the operand-delivery block) to include the three bitwise
    opcodes, so the ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) bands are
    snapped to a clean per-nibble one-hot on the OP_AND/OP_OR/OP_XOR MARK_AX rows
    (the same clean-snap the arithmetic ``C4_CLEAN_OPERAND_ADD`` already applies
    to ADD/SUB/MUL/DIV/MOD).

    This SUBSUMES the former :class:`BitwiseOperandSeRecoverFFN` recover (deleted,
    2026-07-13). ROOT (spec_k=0, BUILT dims, campaign, hook-free probe
    ``tools/_probe_bitw_alu_trace.py``): the L9 non-ALU ALU-scrubber that fires at
    the physical block-21 row subtracts a FIXED pattern (~5.56 at cell 0, ~0.9 at
    cell 8, ~0.94 at cell 15) from the ALU band. The dirty operand-gather hybrid
    delivers the true A==0 nibble at only ~+5.48, so 5.48 − 5.56 = −0.08 → the
    cell-0 one-hot goes NEGATIVE and the rescaled bitwise lookup's A==0 rule
    cannot fire → ``or_16bit`` / ``xor_16bit`` lose byte-0. The clean snap
    delivers the SAME nibble at +6.0, so 6.0 − 5.56 = +0.44 → cell 0 stays
    POSITIVE and survives the crush to the lookup block, where the rescaled
    ``bitwise_rules`` (a+b−ab, operand_a_cw = 30/5.82) fires correctly. So the
    SeRecover's whole job (re-materialise the crushed operand at the lookup block)
    is unnecessary once the operand is delivered clean upstream — proven inert
    (all 6 bitwise smoke programs pass with C4_CLEAN_OPERAND_BITWISE=1 AND the
    SeRecover removed).

    Gated behind the ``no_stack0_emit`` campaign prerequisite so the non-campaign
    golden (35-tok) build is byte-identical (``e50521f3``): the wrap is never
    installed with ``C4_NO_STACK0_EMIT=0``. The former ``C4_CLEAN_OPERAND_BITWISE``
    escape hatch was RETIRED 2026-07-14 (proven default-ON, SUBSUMES the deleted
    BitwiseOperandSeRecoverFFN); the fix is now unconditional under campaign.
    """
    return no_stack0_emit_enabled()


def func_cmp_operand_clean_enabled() -> bool:
    """Return True iff the func_max/func_min CMP loaded-operand-A two-hot clean
    is active (DEFAULT-OFF ``C4_FUNC_CMP_OPERAND_CLEAN``; opt in with ``=1``).
    Gated behind the ``no_stack0_emit`` campaign prerequisite so the flag-OFF
    golden (35-tok) build is byte-identical (the L9 ``block.ffn`` wrap is never
    installed off the campaign, and the ``SE_ALU`` mirror dims it reads are
    campaign over-width dims absent from the narrow golden layout).

    ROOT (spec_k=0, BUILT dims, campaign + ``C4_JSR_BP_BYTE3_CLEAR=1``, GPU/CPU
    faithful; ``tools/_probe_funcmax_cmp_operand.py`` +
    ``tools/_probe_funcmax_cmp_sweep.py``): ``func_max`` / ``func_min`` (ids
    650-699) return via a ``GT`` / ``LT`` compare + ``BZ`` branch + ``LEV`` — the
    loaded local ``a`` (``LI`` -> ``PSH`` -> ``mem[SP]``) is operand A of that
    compare. The L8 head-5 mem-to-ALU operand-A read delivers ``ALU_HI`` as a
    TWO-HOT: the true ``a//16`` cell PLUS a spurious ``~+0.94`` one-hot at the cell
    equal to operand-B's high nibble (``b//16``) — operand B bleeding through the
    operand-A read (the SAME leak class as func_add's ``C4_FUNC_ADD_B0_HINIB``, on
    the CMP path). The ``layer9_step_end_operand_relay`` mirrors the two-hot into
    ``SE_ALU_HI`` and the L9 nibble comparator reads the wrong operand-A high
    nibble -> wrong ``GT``/``LT`` flag -> wrong branch -> returns the wrong
    operand. DECISIVE when ``a//16 == 0``: the LOADED path delivers NO positive
    cell-0 one-hot for a zero high nibble (unlike the immediate path, which lands
    ``SE_ALU_HI[0] ~= +5.12``), so the ``~0.94`` ``b_hi`` leak WINS the argmax and
    the comparator reads ``a_hi == b_hi`` instead of ``a_hi == 0``.

    THE FIX (``CmpLoadedOperandCleanFFN`` wrapping the L9 ``block.ffn``): on the
    ``MARK_SE_ONLY`` cmp rows only, (1) zero ``SE_ALU_*[c]`` where ``c == b``
    nibble (``SE_AX_CARRY_*`` one-hot) AND in the narrow leak window
    ``(0.5, 3.0)`` — the ``~0.94`` leak is zeroed while the immediate
    ``a_hi==b_hi`` true one-hot (``~+5.63``) and the loaded one (``~+6.59``) both
    survive; and (2) if no ``SE_ALU_*`` cell is strong after the clear, write a
    positive cell-0 one-hot (``+5.3``) so the ``a==0`` comparator unit fires. Both
    steps are provably inert on the shared ``if_gt``/``if_lt`` immediate path
    (immediates carry a strong cell-0 one-hot for ``a==0`` and a ``>=5.63``
    one-hot for ``a_hi==b_hi``, neither of which the window/recover touches). The
    window is chosen to sit strictly between the 0.94 leak and the 5.63 minimum
    true magnitude, so it is correct by construction (measured, not tuned).

    Campaign entry point: ``C4_CAMPAIGN=1`` does NOT auto-enable this (it is a
    net-new value fix pending its verdict-flip audit); opt in explicitly with
    ``C4_FUNC_CMP_OPERAND_CLEAN=1``.
    """
    return (
        no_stack0_emit_enabled()
        and os.environ.get("C4_FUNC_CMP_OPERAND_CLEAN", "0") == "1"
    )


def sili_cam_b1_enabled() -> bool:
    """Return True iff the si/li LOAD byte-1 address-leak discriminator (Inc-2)
    is active. DEFAULT campaign-ON (``C4_SILI_CAM_B1=1``), opt out with
    ``C4_SILI_CAM_B1=0``; gated behind the two campaign flags so the flag-OFF
    golden (35-tok) build is byte-identical (the slot is never written off the
    campaign).

    ROOT (measured, faithful real-runner probes tools/probe_sili_scores.py +
    probe_var_scores.py, GPU bit-exact, campaign config):
    the LI-reload AX byte-1 is delivered by the shared L10 head-1 slot 82
    (``memax_byte1``), which selects the most-recent OP_IMM AX byte-1 REGISTER
    row by ALiBi recency. The slot-82 K-side scores the value-IMM step and the
    address-IMM step IDENTICALLY (both carry IS_BYTE + H1+AX + BYTE_INDEX_1 +
    OP_IMM with the same CLEAN signature); the ONLY thing separating them is
    ALiBi recency (the addr-IMM step is later -> wins by ~60). For
    ``IMM addr; LI`` (si/li/sc/lc) that later OP_IMM row is the LOAD-ADDRESS
    IMM (0x200 byte-1 = 0x02) so the reload LEAKS the address byte-1
    (roundtrip 42->0x22A, 16bit 0x1234->0x0234). For ``return x`` (var_simple,
    LEA-addressed) there is NO competing IMM-address row, so slot 82 picks the
    value-IMM (load-bearing; var_simple 25/25).

    THE DISCRIMINATOR (slot 83, Q-side gated -> softmax-safe): the address-IMM
    register row is the gathered LOAD ADDRESS, so it carries a STRONG
    ``ADDR_B1`` one-hot (HI/LO cell sums ~3.0) staged by the L13 address-gather
    chain; the genuine value-IMM register row carries only the weak residual
    ``ADDR_B1`` (sums ~1.0). Slot 83 fires its Q on the SAME byte-1 predictor
    row as slot 82 (so it contributes to NO other query -> the byte-0 reload
    softmax is untouched, NOT the K-side ADDR-veto blind spot), and its K
    DOWN-weights candidates by their ADDR_B1 magnitude. The penalty is ~3x on
    the leaky addr-IMM row and ~1x on the value-IMM row, a net margin that
    out-votes the ~60-point ALiBi recency gap and re-points the byte-1 reload
    at the value-IMM register (var_simple's winner, sum ~1.0, keeps its huge
    OP_IMM margin so it is unaffected).

    STATUS: flips 4/6 TestSmokeMemory (roundtrip / zero / multiple / overwrite);
    those have a value byte-1 == 0x00 so once the slot-83 selection picks the
    value-IMM the reload is correct. The remaining 2 are part-c (the SI-store
    byte-1 / downstream OUTPUT_HI corruptor), still open:
      * test_si_li_16bit_value (0x1234): slot 83 DOES re-point the L10 head-1
        byte-1 to the value-IMM (0x12 lands in OUTPUT at block 16, GPU-traced);
        but a DOWNSTREAM L18 (runner block 32) op SLAMS OUTPUT_HI on the
        LI-reload AX byte-1 row -- cell 0 -> +18, cells 1..15 -> -1700 -> hi
        nibble forced to 0 -> 0x12 -> 0x02. The L14 *_ax_bytes_zero FFN rules
        (jsr/lc/alu_nocarry/ent) are NOT the cause (their OP_LC_RELAY / TEMP+7
        gates read 0 on this row); the slam is the layer14_mem_generation
        ADDRESS head (heads 1..3, MEM addr byte) whose addr_b1 position-distance
        ALIASES the LI-reload AX byte-1 row in the 30-tok frame and copies the
        store ADDRESS byte-1 into OUTPUT. FIX (part-c): sharpen its slot-34
        MEM_STORE gate in campaign (move CONST down + MEM_STORE up by the SAME
        delta so STORE rows score identically and non-store rows fall to the
        softmax1 sink) so the head stays silent on the non-store LI row -- but
        wire it through ``_layer14_mem_generation_head_specs_with_overrides``
        (the override q_map is the BAKED path; a base-spec-only edit is inert),
        then re-run the flag_regression_gate (the store-address head is
        load-bearing for ALL SI/SC/PSH stores).
      * test_sc_lc_roundtrip: byte-1 IS now fixed (0x02 -> 0x00); the residual
        fail is a SEPARATE pre-existing LC byte-0 reload bug (got 0, want 42)
        outside this byte-1 scope.
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_SILI_CAM_B1`` escape hatch was retired). Campaign gate preserved.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def sili_b1_restore_enabled() -> bool:
    """Return True iff the si/li 16-bit LOAD byte-1 value RESTORE (Inc-2 part-c)
    is active. DEFAULT campaign-ON (``C4_SILI_B1_RESTORE=1``), opt out with
    ``C4_SILI_B1_RESTORE=0``; gated behind the campaign config so the flag-OFF
    golden (35-tok) build is byte-identical (the band + both ops are omitted off
    the campaign).

    ROOT (measured spec_k=0, BUILT dim_positions, campaign config; GPU
    block-trace ``tools/probe_sili_slam32.py`` / ``probe_sili_real.py``):
    the slot-83 discriminator (``sili_cam_b1_enabled``) correctly re-points the
    L10 head-1 byte-1 to the value-IMM register, so the loaded byte-1 (0x12 for
    ``si 0x1234; li``) lands in OUTPUT at runner block 16. A DOWNSTREAM op then
    SLAMS OUTPUT_HI on the LI-reload AX byte-1 predictor row at runner BLOCK 32
    (logical L18): the FFN unit that forces the AX byte-1 HIGH nibble to 0 (the
    "AX byte-1 == 0 default" — ``OUTPUT_HI+1..15 = -50`` gated on
    ``H1+1``/``BYTE_INDEX_0``). That default is INTENTIONALLY suppressed on
    register rows whose AX legitimately has a high byte via the ``AX_CARRY``
    cross-step band, but on the LI-reload step that suppression band is empty, so
    the default fires and crushes the loaded 0x12 -> 0x02 (final reload 0x0234,
    smoke want 0x1234). The attn at block 32 is NOT the source (zeroing its
    ``W_o`` leaves the slam); the slam is FFN unit 3 of block 32 (ablating it
    restores 0x12) — so a head-spec edit cannot reach it.

    THE FIX (two PureFFN ops mirroring the ``C4_SUB_FULL_BORROW`` precedent: an
    early CAPTURE + an L25-tail RESTORE that DOMINATES the slam). On the
    LI-reload byte-1 predictor row (the AX byte-0 row of an LI step — discriminated
    by ``IS_BYTE + H1[AX]+1 + BYTE_INDEX_0 + ADDR_B1`` and NOT
    ``OP_IMM``/``MARK_AX``/``MEM_STORE``/``BYTE_INDEX_1..3``; ``ADDR_B1`` is the
    gathered LOAD-address one-hot, strong on LI-reload rows and ~0 on
    PSH/SI/IMM register rows, so the row is LOAD-specific):
      1. CAPTURE (``make_layer14_sili_b1_capture_op``, a standalone PureFFN at the
         EARLY L14 mem-gen block, BEFORE the block-32 slam, where OUTPUT byte-1 is
         still the loaded value): gate-copy the 16 ``OUTPUT_LO`` + 16 ``OUTPUT_HI``
         nibble cells into the private ``LI_RELOAD_B1_{LO,HI}`` band. The silu
         factor is uniform across the 16 cells so the byte-1 nibble argmax is
         preserved.
      2. RESTORE (``make_sili_b1_restore_op``, a standalone PureFFN on the L25 tail
         block AFTER ``tail_bit32_result_correction`` — the LAST OUTPUT writer
         before the LM head): gate-write ``OUTPUT_{LO,HI}`` back from
         ``LI_RELOAD_B1_{LO,HI}`` at a DOMINANT magnitude, so the restored 0x12
         out-votes the block-32 slam (~+18/-1700) additively.
    On the 4 already-passing si/li cases (value byte-1 == 0x00) the captured band
    is the 0x00 one-hot, so the restore re-asserts 0x00 (no-op). var_simple
    (LEA-addressed, no ADDR_B1 on its byte-1 row) and PSH/SI register rows are not
    matched by the discriminator, so they are untouched. Output-affecting only
    inside the campaign; flag OFF (or off-campaign) omits the band + both ops
    (golden ``7f6f2e5d`` byte-identical).
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_SILI_B1_RESTORE`` escape hatch was retired). Campaign gate preserved.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def store_ax_b0_override_enabled() -> bool:
    """Return True iff the SI/SC store-AX byte-0 OUTPUT materializer uses the
    zero-default OVERRIDE write form. DEFAULT campaign-ON
    (``C4_STORE_AX_B0_OVERRIDE=1``), opt out with ``C4_STORE_AX_B0_OVERRIDE=0``;
    gated behind the two campaign flags so the flag-OFF golden (35-tok) build is
    byte-identical (the ``l16_store_ax_carry_lo`` rules keep their bare additive
    ``2.0/S`` write off the campaign).

    ROOT (var_mul step-9 store, GPU/CPU full_trace; the ``var_mul`` 2nd-local
    SI store of ``b``): on the SI step's AX marker the store value is carried in
    ``AX_CARRY_LO/HI`` (correct), but the OUTPUT byte-0 LO band carries a STRONG
    zero-byte default (the L19 / block-33 OUTPUT-zero-default, ~+26/+40 on
    ``OUTPUT_LO+0``) that out-votes the weak additive ``+2.0/S`` write of
    ``l16_store_ax_carry_lo``. When the stored value's HIGH nibble of byte-0 is
    zero (``b <= 15``: var_mul ids 279/280/287/290/296) the carried LOW nibble
    is the only nonzero cell, and the bare additive write loses to the
    zero-default -> OUTPUT byte-0 = 0 -> ``SI`` stores 0 -> ``LI b`` reads 0 ->
    ``a * 0``.

    THE FIX (mirrors the campaign ``l16_psh_ax_carry_lo`` materializer added for
    the PSH func-arg path, which solved the IDENTICAL zero-default-out-votes
    problem): switch the ``l16_store_ax_carry_lo`` write to the OVERRIDE form —
    write ``+W`` to the carried LO nibble cell ``k`` AND ``-W`` to ``OUTPUT_LO+0``
    for ``k != 0`` so a NONZERO carried low nibble overrides the zero default.
    ``k == 0`` self-cancels (a genuinely-zero low byte stays at its zero
    default). The HI band keeps the plain additive form (no competing default on
    ``OUTPUT_HI``). The rule COUNT is unchanged (the 16 LO rules' write tuples
    are rewritten in place), so the flag-OFF golden bake is byte-identical and
    the L16 unit allocator's fixed-range assertion still holds.

    Output-affecting only inside the campaign; flag OFF (or off-campaign) keeps
    the bare additive ``2.0/S`` write (golden ``7f6f2e5d`` byte-identical).
    """
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
        and os.environ.get("C4_STORE_AX_B0_OVERRIDE", "0") != "0"
    )


def store_ax_b0_override_v2_enabled() -> bool:
    """Return True iff the SI/SC store-AX byte-0 OUTPUT materializer uses the
    zero-default OVERRIDE write form *with an ALU/cmp opcode ANTI-CONDITION gate*
    (the "clean discriminator" V2). DEFAULT **ON** (``C4_STORE_AX_B0_OVERRIDE_V2``,
    opt out ``=0``); gated behind the two campaign flags so the flag-OFF golden
    (35-tok) build is byte-identical.

    WHY V2 (root: var_three id300 / var_mul step-9 SI-store; task this session):
    the plain ``C4_STORE_AX_B0_OVERRIDE`` (default OFF since 7869e5c3) FIXES the
    store step (var_three id300 SI-of-b step-9: AX byte-0 token 0 -> 6, GPU/CPU
    teacher-forced verified) but was reverted OFF because "ON amplified the
    l16_store_ax_carry misfire onto ADD/SUB/cmp/mul rows (-24)". The plain
    ``store_ax_conditions`` gate (``OP_SI + OP_SC + MARK_AX - 8*MARK_PC -
    10*IS_BYTE - 20*OP_EXIT - 20*OP_JMP``, threshold 4.0) does NOT explicitly
    forbid the ALU/cmp opcodes, so any residual OP_SI/OP_SC energy at an
    ADD/SUB/MUL/DIV/MOD/cmp AX-marker row can lift the AND-gate ``up`` above zero
    and let the strong ``-W`` OUTPUT_LO+0 write bleed onto the arithmetic result
    byte. V2 adds a large-negative anti-condition on every ALU/cmp opcode flag
    (``-20`` each) so the AND-gate ``up`` is driven deeply negative (silu -> 0) on
    ANY ADD/SUB/MUL/DIV/MOD/EQ/NE/LT/GT/LE/GE row — the override CANNOT fire there
    by construction, while a genuine SI/SC store row (all ALU/cmp flags ~0) is
    unaffected. This is the clean store-only discriminator the plain override
    lacked. Rule COUNT is unchanged (only the SI/SC store rules' condition tuple +
    write tuples change, and only when V2 is ON), so the flag-OFF golden bake is
    byte-identical.

    Mutually exclusive with the plain override in practice: enable EITHER
    ``C4_STORE_AX_B0_OVERRIDE=1`` (broad, un-discriminated) OR
    ``C4_STORE_AX_B0_OVERRIDE_V2=1`` (discriminated). If both are set, V2 wins (the
    anti-condition gate is the strict superset guard). Output-affecting only inside
    the campaign; flag OFF (or off-campaign) keeps the bare additive ``2.0/S``
    write (golden ``f725c06e`` byte-identical).
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_STORE_AX_B0_OVERRIDE_V2`` escape hatch was retired). Campaign gate
    # preserved.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def loop_lea_b0_e8_restore_enabled() -> bool:
    """Return True iff the loop_sum in-loop ``LEA &i`` byte-0 0xE8 RESTORE
    (``C4_LOOP_LEA_B0_E8``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_LEA_B0_E8=0``. Flag-off OR
    a non-campaign / golden build registers NO rules and appends NO post_op, so
    the model is bit-for-bit identical to golden ``7f6f2e5d``.

    ROOT (measured spec_k=0, BUILT dim_positions, campaign config; GPU AR-trace
    ``tools/_probe_loopsum_lea_b0.py`` 450 2 + ``_probe_loopsum_blk43_attrib.py``
    + ``_probe_loopsum_entax_gate.py``): on the ``loop_sum`` (and loop_mul /
    loop_pow2 / loop_fact family) IN-LOOP comparison step the ``LEA &i`` (the
    1st-local address, want byte-0 = 0xE8) emits 0x00 because of a TWO-part
    failure:

      1. The ``tail_lea_local_ax_marker_byte0_e8`` keystone (block 42) does NOT
         fire on this row: it REQUIRES ``CMP+7`` AND ``MEM_ADDR_SRC`` (each 1.0 on
         a genuine address-eval LEA), but the in-loop LEA row carries
         ``CMP+7 == 0`` and ``MEM_ADDR_SRC == 0`` (its in-loop base is lower than
         the keystone was tuned for). So OUTPUT_LO byte-0 stays DEAD (~0.0)
         through block 42 — the 0xE8 stamp never lands.
      2. The ``_l10_ent_axcarry`` override op (block 43) THEN WRONGLY fires on
         this GENUINE LEA row and slams byte-0 to 0x00. That op's discriminator
         is ``MARK_AX + OP_LEA(>=4) + no-owning-opcode + MEM_ADDR_SRC cold``; it
         was designed for the multilocal MAIN-ENT step (where OP_LEA LEAKS ~0.81
         and ``MEM_ADDR_SRC == 0``) and EXCLUDES genuine address-eval LEAs via
         the ``MEM_ADDR_SRC`` NOT-block. But this in-loop ``LEA &i`` is a genuine
         LEA (``OP_LEA == 5.24``) that ALSO carries ``MEM_ADDR_SRC == 0``, so the
         exclusion fails — ent_axcarry routes ``AX_CARRY`` (byte-0 = 0x00) into
         OUTPUT, winner-take-all to cell 0. Final emitted AX byte-0 = 0x00.

    FETCH on this row carries the imm=-8 signature (``FETCH_LO+8`` argmax,
    ``FETCH_HI+15`` argmax, ``FETCH_LO+0`` / ``FETCH_HI+14`` cold) — the EXACT
    pattern the e8 keystone keys on for a 0xE8 byte-0, and DISTINCT from the
    multi-local 2nd local (``FETCH_LO+0``, want 0xE0) and 3rd local
    (``FETCH_HI+14``, want 0xD8).

    FIX (mirrors the ``C4_SILI_B1_RESTORE`` / ``_l10_ent_axcarry`` precedent):
    a flag-gated ``PureFFN`` post_op on the L25 tail block, appended AFTER
    ``l10_ent_axcarry`` (so it is the LAST OUTPUT writer at this row and DOMINATES
    the slam), that — gated on the in-loop ``LEA &i`` discriminator (genuine
    ``OP_LEA`` + imm=-8 FETCH signature + ``MEM_ADDR_SRC`` cold + no owning opcode
    + the multi-local FETCH NOT-blocks) — writes byte-0 = 0xE8 (``OUTPUT_LO+8``
    HIGH, ``OUTPUT_HI_THIS_STEP+14`` HIGH, all other nibble cells driven
    ``-DOM``) via a per-cell winner-take-all. The ``MEM_ADDR_SRC`` NOT-block keeps
    it OFF every genuine address-eval LEA (those carry ``MEM_ADDR_SRC == 1`` and
    are already correct via the keystone), and the imm=-8 FETCH AND-gate keeps it
    off the 2nd/3rd-local LEAs (``var_mul`` / ``var_three``) and every non-LEA AX
    row, so the whole op is a no-op everywhere except the loop in-loop ``LEA &i``
    row it targets.
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_LOOP_LEA_B0_E8`` escape hatch was retired). Campaign gate preserved.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def loop_lea_b0_e0_restore_enabled() -> bool:
    """Return True iff the loop_sum in-loop 2nd-local ``LEA &sum`` byte-0 0xE0
    RESTORE (``C4_LOOP_LEA_B0_E0``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_LEA_B0_E0=0``. Flag-off OR
    a non-campaign / golden build registers NO rules and appends NO post_op, so
    the model is bit-for-bit identical to golden ``fd60f5f4``.

    ROOT (measured spec_k=0, BUILT dim_positions, campaign config; GPU AR-trace
    ``tools/_probe_loopsum_lea_b0.py`` 450 6 + ``_probe_loopsum_blk43_attrib.py``
    450 6 44): after the merged step-2 fix (``C4_LOOP_LEA_B0_E8``) advances
    ``loop_sum`` to step 6, the in-loop ``sum = sum + i`` body issues a 2nd-local
    ``LEA &sum`` whose AX-marker wants byte-0 = 0xE0 (the full AX = 0xFFE0 =
    65504 sign-extended local address). It instead emits 0x01 because:

      1. The 0xE8 keystone / e8-restore op DEFERS here (its ``FETCH_LO+0``
         NOT-block) -- correctly, since this is the 2nd local, not ``&i``.
         Through the L25 tail (block 43, the post-block-43 residual) OUTPUT_LO
         byte-0 carries cell-0 ~49.5 (the correct LO nibble for 0xE0) but no
         keystone has stamped the 0xE high nibble.
      2. The post-tail block-44 LEA effective-address materializer (a 32-unit
         standalone PureFFN, gated MARK_AX + OP_LEA*60 + IS_BYTE) then WTA-slams
         OUTPUT_LO -> cell **1** and OUTPUT_HI_THIS_STEP -> cell 0 on this row,
         deriving byte-0 = 0x01 (its in-loop 2nd-local effective-address compute
         is wrong). The e8-restore op (which DOMINATES this slam for ``&i`` at a
         LATER post-op block) does not fire here, so the 0x01 ships.

    The 2nd-local is distinguished from the 1st (``&i`` -> 0xE8) and 3rd
    (``&c`` -> 0xD8) locals by WHICH FETCH cell dominates: the 2nd local carries
    ``FETCH_LO+0`` argmax (=1.0; ``FETCH_LO+8`` / ``FETCH_HI+14`` cold) -- the
    EXACT inverse of the e8 op's ``FETCH_LO+8``-dominant signature.

    FIX (mirrors ``C4_LOOP_LEA_B0_E8`` / ``_l10_loop_lea_b0_e8_rules``): a
    flag-gated ``PureFFN`` post_op appended AFTER ``l10_loop_lea_b0_e8`` (so it
    is the LAST OUTPUT writer at the 2nd-local LEA row and DOMINATES the block-44
    slam) that -- gated on the in-loop ``LEA &sum`` discriminator (genuine
    ``OP_LEA`` + imm=-16 ``FETCH_LO+0``-dominant signature + ``MEM_ADDR_SRC``
    cold + no owning opcode + the multi-local FETCH NOT-blocks) -- writes byte-0
    = 0xE0 (``OUTPUT_LO+0`` HIGH, ``OUTPUT_HI_THIS_STEP+14`` HIGH, all other
    nibble cells driven ``-DOM``) via a per-cell winner-take-all. The
    ``FETCH_LO+8`` / ``FETCH_HI+14`` NOT-blocks keep it OFF the 1st/3rd-local
    LEAs, the ``MEM_ADDR_SRC`` NOT-block keeps it OFF every genuine
    address-eval LEA, and the ``IS_BYTE`` + non-AX marker NOT-blocks keep it OFF
    every value-byte / non-AX row.
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_LOOP_LEA_B0_E0`` escape hatch was retired). Campaign gate preserved.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def jsr_bp_byte3_clear_enabled() -> bool:
    """Return True iff the func step-0 JSR-step BP byte-3 high-byte CLEAR is
    active (UNCONDITIONAL under campaign as of the P5 flag-retire 2026-07-14; the
    former ``C4_JSR_BP_BYTE3_CLEAR`` escape hatch was retired).

    Requires the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``). A non-campaign / golden build registers NO rules
    and appends NO post_op, so the model is bit-for-bit identical to golden
    ``f725c06e``.

    ROOT — THE func-cluster STEP-0 (JSR) POISONING BYTE (func_min id675 /
    func_max id650 / func_identity id550 / func_add id575; measured spec_k=0,
    BUILT dim_positions, campaign; ``tools/_probe_funcmin_vcorr.py`` +
    ``_probe_bp_b3_logit_attr.py`` + ``_probe_funcmin_inject.py``):

    Every func-cluster program's FIRST value-byte correction (the cross-step
    autoregressive poisoning point ``interp_oracle_gate._value_correction_step``)
    is at **step 0 (the JSR that calls the callee)**, on **BP byte 3**. The
    caller's BP is ``0x00010000`` (byte-2 = 0x01, byte-3 = 0x00), but the model
    emits **BP byte-3 = 0x01** -> BP decodes to ``0x01010000`` and every
    downstream frame-relative (``mem[BP-8]`` / ENT / LEV / LI) read is poisoned,
    flat-diverging the whole cluster. Hook-inject clearing this one byte advances
    the vcorr from step 0 to step 2 on ALL four programs (root confirmed).

    ATTRIBUTION (LM-head logit, BUILT dims): at the BP byte-3 PREDICTOR row (the
    BP byte-2 value row: ``IS_BYTE`` + ``BYTE_INDEX_2`` + ``H1+3`` + ``OP_JSR``,
    ``HAS_SE==0``) the OUTPUT band is a near-TIE the WRONG way -- ``OUTPUT_LO+0 =
    +6.94`` (nibble 0, correct 0x?0) vs ``OUTPUT_LO+1 = +8.00`` (nibble 1) -- so
    the byte-3 low nibble decodes to 1 (``logit[1]-logit[0] = +4.3``, driven
    100% by ``OUTPUT_LO+1`` via ``W_head[1,OUTPUT_LO+1]-W_head[0,·] = +5``). A
    weak ``+2.0`` enters ``OUTPUT_LO+1`` at physical block ~18 and is amplified
    ~3.4x at block ~58, and unlike the ENT step (which
    ``l6_ent_after_jsr_bp_byte3_00`` clears) the JSR step has NO byte-3 clear.
    The clean, always-present discriminator that isolates this exact row is
    ``OP_JSR`` (the ENT-step clear is gated on ``OP_ENT``/``HAS_SE`` instead) +
    the BP-register-byte ``H1+3`` staging signal + ``BYTE_INDEX_2``.

    FIX (mirrors ``l6_ent_after_jsr_bp_byte3_00`` but JSR-gated + placed in the
    L25 tail so it DOMINATES the block-58 amplifier): a flag-gated ``PureFFN``
    post_op appended after the loop_lea ops that -- on the JSR-step BP byte-3
    predictor row -- WTA-forces ``OUTPUT_LO`` and ``OUTPUT_HI_THIS_STEP`` to
    nibble 0 (byte-3 = 0x00). ``OP_JSR`` + ``H1+3`` + ``BYTE_INDEX_2`` hard-req,
    with ``HAS_SE`` (ENT rows) + every non-BP marker + wrong BYTE_INDEX
    NOT-blocked, so it is a no-op on every non-JSR-BP-byte3 row.
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_JSR_BP_BYTE3_CLEAR`` escape hatch was retired as a proven default-ON
    # fix). The campaign gate is preserved so the non-campaign golden build is
    # byte-identical.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def loop_si_byterow_marker_clear_enabled() -> bool:
    """Return True iff the loop back-edge SI-step value-byte-row MARKER-residue
    clear (``C4_LOOP_SI_BYTEROW_CLEAR``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_SI_BYTEROW_CLEAR=0``.
    Flag-off OR a non-campaign / golden build registers NO rules and appends NO
    post_op, so the model is bit-for-bit identical to golden ``5acb3d23``.

    ROOT — THE FIRST LOOP BACK-EDGE CONTROL-FLOW DESYNC (loop_sum id450,
    measured spec_k=0, BUILT dim_positions, campaign config; GPU AR-trace
    ``tools/_probe_loopsum_backedge.py`` + ``_probe_loopsum_spdrift.py`` +
    ``_probe_loopsum_explosion.py``):

    After the step-2 / step-6 in-loop LEA byte-0 fixes (``C4_LOOP_LEA_B0_E8`` /
    ``C4_LOOP_LEA_B0_E0``) advance ``loop_sum`` to step 9, the ``sum = 0`` store
    (step 9 = ``SI``) DESYNCS the 30-token frame: it emits **41 tokens not 30**.
    The over-run begins at the SP value-byte-0 row (the row whose input token is
    the just-emitted SP byte-0 = 0xE8). At that row the LM-head BYTE logits
    collapse to ~ -5.8e11 so a register-MARKER token (``REG_PC`` = 257) wins by
    DEFAULT, injecting a spurious PC/AX/SP block (+11 tokens). The fixed-30-token
    slicer then mis-reads the NEXT step: step 10 (the loop-condition ``LEA &i``,
    oracle ``pc_after`` = 106) is decoded as ``pc_after`` = 114 — the PC appears
    to advance +16 not +8, the brief's reported symptom.

    WHY the byte logits collapse (per-block + per-unit attribution): at the SI
    step's value-byte rows ``MEM_STORE`` (born ~block 11) and ``OP_SI`` (born
    ~block 31) carry a small NEGATIVE residue (~ -2.4e-3 / -4.2e-4) that SHOULD be
    exactly 0 on a value-byte row (markers/opcodes belong on the marker row). The
    L25 tail bank (``tail_bit32_result_correction``, block 43) reads these dims at
    ``-1e8`` as NOT-blockers ASSUMING they are 0. ``MEM_STORE * -1e8 = +2.4e5``
    (plus ``OP_SI * -1e8 = +4.2e4``) is enough to flip the bank's silu gate from
    ``up`` ~ -1.4e4 (OFF, the clean-row value) to ``up`` ~ +6.8e4 (ON): a whole
    band of OUTPUT-decode units fires asymmetrically -> the OUTPUT_LO/HI decode
    band explodes to ~ -7.4e11 -> the LM byte head (each band cell drives a byte
    token at +5.0) is crushed all-negative -> marker token. On a clean (PSH/IMM)
    step those residues are ~0 so the bank stays OFF and the band decodes a real
    byte (+14.3, the SP value).

    FIX: a flag-gated ``PureFFN`` block scheduled IMMEDIATELY BEFORE the L25 tail
    bank that, on EVERY value-byte row (``IS_BYTE`` gate), ADDS a small POSITIVE
    bias (effective +0.02) to ``MEM_STORE`` and ``OP_SI`` so their residue can no
    longer flip the tail bank's ``-1e8`` silu gate positive (``+0.02 * -1e8 =
    -2e6`` keeps ``up`` deeply negative -> bank OFF -> OUTPUT band decodes the
    real byte). The +0.02 magnitude is CRITICAL: it must be SMALL enough that it
    does not also flip the AX HIGH-BYTE sign-extension materializer (at +0.5 the
    AX byte 2/3 over-sign-extend to 0xFF) NOR the multi-byte ADD high-byte adder
    (at the unscaled FFN-saturated +100 the ADD high byte is lost, 768 -> 256) --
    both read the same MEM_STORE / OP_SI dims; the AX-safe window is ~0.005..0.1
    so 0.02 is the robust middle (calibrated via the deterministic ~5000 silu
    saturation, W_down = 0.02/5000 = 4e-6). It fires ONLY on value-byte rows
    (``IS_BYTE`` AND every marker NOT-blocked) so it never perturbs the marker
    rows the tail's MEM-store address materializers legitimately use. Verified
    (``_probe_loopsum_backedge.py`` + canonical id450 + a 72-id cross-cluster
    campaign sample, ZERO regressions): the SI step re-emits 30 tokens, step-10
    PC is 106 (correct), and steps 0-10 are PC+AX byte-correct. The remaining
    loop_sum residual (step-11 in-loop ``LI &i`` returning AX=0 not 1) is a
    DISTINCT downstream value-load / operand-CAM root (task #342 family), out of
    scope for the back-edge desync.
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_LOOP_SI_BYTEROW_CLEAR`` escape hatch was retired). Campaign gate
    # preserved.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def loop_li_opcode_fetch_addrkey_clamp_enabled() -> bool:
    """Return True iff the in-loop opcode-fetch ADDR_KEY top-byte over-count
    clamp (``C4_LOOP_LI_FETCH_ADDRKEY_CLAMP``) is active.

    DEFAULT **ON** in the campaign config (``C4_NO_STACK0_EMIT=1`` +
    ``C4_OPERAND_FROM_MEMSP=1``); opt-out via ``C4_LOOP_LI_FETCH_ADDRKEY_CLAMP=0``.
    Flag-off OR a non-campaign / golden build registers NO rules and appends NO
    post_op, so the model is bit-for-bit identical to golden.

    ROOT — THE IN-LOOP ``LI`` VALUE-LOAD RETURNS 0 IS AN UPSTREAM OPCODE-FETCH
    MISS ON THE STORE (loop_sum id450 step 11, measured spec_k=0, BUILT
    dim_positions, campaign config; GPU teacher-forced trace
    ``tools/_probe_loopsum_li_cam.py`` + ``_probe_loopsum_memstore_chain.py`` +
    ``_probe_loopsum_pcfetch.py`` + ``_probe_loopsum_fetchattn.py``):

    After the back-edge desync fix (``C4_LOOP_SI_BYTEROW_CLEAR``) advances
    ``loop_sum`` to step 11, the in-loop ``LI &i`` (loop-condition variable load,
    pc 106->114) returns AX=0 not 1. The L15 ``memory_lookup`` CAM (head 0) that
    content-addresses the load only attends to MEM rows carrying ``MEM_STORE=1``.
    But the two genuine in-frame ``SI`` stores -- step 5 (``i = 1`` -> 0xFFE8) and
    step 9 (``sum = 0`` -> 0xFFE0) -- carry ``MEM_STORE = -0.0`` on their MARK_MEM
    marker (only the prologue JSR + an in-loop PSH show ``MEM_STORE = 1``). So the
    LI CAM's candidate set is EMPTY of the real store and the load reads garbage.

    WHY ``MEM_STORE`` is missing on the SI stores: ``MEM_STORE`` is the relayed OR
    of ``OP_SI/OP_SC/OP_PSH/OP_JSR/OP_ENT`` (L6 head 6 broadcasts it from the AX
    marker to the MARK_MEM marker). At the SI steps the opcode dim ``OP_SI`` is
    DEAD (~0) because the L5 opcode-fetch (block 6) reads the WRONG opcode byte:
    OPCODE_BYTE decodes to 0x00 (LEA) instead of 0x0B (SI). The fetch is a content
    -addressed match of the relayed PC (``EMBED_LO/HI`` nibbles, head 1 Q) against
    each code byte's positional ``ADDR_KEY`` (head 1 K). For the SI PCs (58, 90 --
    low nibble 0xa) the correct code row (the SI opcode byte) and a STRAY zero
    high-byte code row sit in a RAZOR-THIN tie (score 112.71 vs 112.94); the wrong
    zero byte wins by 0.23 and its V copies 0x00 into OPCODE_BYTE.

    The 0.23 margin is created by a DOUBLED ``ADDR_KEY+32`` (3rd address nibble,
    top byte) on the wrong row: it carries 2.0 not 1.0, so its head-1 slot-35
    contribution is ~803 (vs the correct row's ~402), exactly enough to overcome
    its missing mid-nibble match. The doubling is an aliasing collision produced
    at block 3 (L3 ``_pc_byte1_prev_head_spec`` stages ``CLEAN_EMBED`` ->
    ``ADDR_KEY+32`` for the 12-bit code-address match) landing ON TOP of the
    positional ``ADDR_KEY+32`` one-hot (both index 0 for PC < 256) at certain
    zero code bytes (e.g. id450 pos 233). Short programs (var_simple id250, same
    SI PC=58) have NO such colliding zero byte, so their SI fetch wins cleanly and
    OP_SI/MEM_STORE are correct -- this is a LONG-PROGRAM / loop-body root.

    FIX: a flag-gated ``PureFFN`` block scheduled inside L4 (block 4, AFTER the L3
    doubling at block 3, BEFORE the L5 opcode fetch at block 6) that CLAMPS the
    over-counted ``ADDR_KEY+32`` band back to 1.0 on prompt code-byte rows. Per
    cell k a step-function unit fires iff ``ADDR_KEY+32+k >= 1.5`` (i.e. the cell
    was doubled to ~2.0) on an ``IS_BYTE`` row with every register MARKER hard
    NOT-blocked (so only PROMPT code bytes, never an emitted PC/AX/SP/.. value
    byte where L3 head 7 legitimately single-stages the nibble), and subtracts
    1.0 from that cell. A clean single-staged cell (1.0 < 1.5) never fires, so the
    legitimate 12-bit code-address match and every short program are byte-
    identical. With the doubling removed the correct SI code row wins the fetch
    p=1.000 -> OP_SI fires -> MEM_STORE=1 on the in-loop SI stores again
    (verified GPU AR: loop_sum id450 steps 0-10 PC+AX byte-correct; the L15 CAM
    candidate set now contains the genuine stores; teacher-forced LI returns the
    correct value).

    SCOPE / BLUEPRINT (NOT a full flip of id450 yet): this op is the NECESSARY
    upstream half. The in-loop LI still returns 0 in the AUTOREGRESSIVE path
    because the SI store's ADDRESS is never materialized into ``ADDR_KEY`` at the
    MEM store row -- in the AR run EVERY store row has ``ADDR_KEY_set=[]`` (the
    store's emitted MEM address bytes are 0x00000000, identical for the PASSING
    var_simple), so the L15 value-load lookup falls back to ALiBi recency. A
    single-store program (var_simple) is fine on recency; a LOOP has multiple
    intervening stores between the i-store (step 5) and the in-loop LI (step 11)
    so recency picks the wrong (later) store and the load returns 0. The
    downstream lever is to MATERIALIZE the store address into ``ADDR_KEY`` at the
    in-loop MEM store row (the #342 operand-CAM / value-load-CAM family) so the
    L15 CAM can content-address the correct store -- then this op's restored
    MEM_STORE flag (which gates the candidate set) lets the load land. Byte-
    identical flag-OFF (golden ``928e49ec`` / campaign ``53118713`` both UNCHANGED
    vs base bafc16b6). var_simple HOLDS (PASS); func_identity id550 fails
    IDENTICALLY flag-ON and flag-OFF (a pre-existing base failure, not a
    regression).
    """
    # Unconditional under campaign (P5 flag-retire 2026-07-14; the former
    # ``C4_LOOP_LI_FETCH_ADDRKEY_CLAMP`` escape hatch was retired). Campaign gate
    # preserved.
    return (
        no_stack0_emit_enabled()
        and operand_from_memsp_enabled()
    )


def ffn_lint_mull14_demo_enabled() -> bool:
    """Return True iff the cross-op FFN-lint MUL-L14-ENTANGLEMENT demo op is on.

    DEFAULT OFF — opt-in via ``C4_FFN_LINT_MULL14_DEMO=1``. TOOLING-ONLY: a
    faithful, reaches-the-build reproduction of the −60 mul-l14 entanglement
    class for ``tools/lint_cross_op_ffn.py --demo``. It registers ONE PureFFN
    post-op unit on the l14 ALU block, AUTHORED as "MUL-only" (its W_up reads
    ``OP_MUL`` strongly) but with a positive ``b_up`` bias so ``silu(up)`` is
    NON-ZERO even when ``OP_MUL==0`` — the smooth-nonlinearity leak that writes
    the SHARED ``OUTPUT_LO`` band on ADD/SUB/DIV rows too (the exact −60
    mechanism the lint exists to catch). The whole op is registered ONLY when
    the flag is on (lookahead-chain pattern), so a flag-off / production build
    is byte-identical to golden ``4958b35b``.
    """
    return os.environ.get("C4_FFN_LINT_MULL14_DEMO", "0") == "1"


def ffn_lint_clean_demo_enabled() -> bool:
    """Return True iff the cross-op FFN-lint CLEAN-CONTROL branch is on.

    DEFAULT OFF — opt-in via ``C4_FFN_LINT_CLEAN_DEMO=1``. This flag is
    TOOLING-ONLY: it exists so ``tools/lint_cross_op_ffn.py --demo`` has a
    same-layout, in-place modification of a SHARED l14 ALU attention head that
    nonetheless writes ONLY a PRIVATE scratch dim (``TEMP``), touching no
    OUTPUT/ALU residual band any downstream op reads. The lint must PASS it
    (the clean side of the discrimination contract) exactly as it FLAGS the
    band-perturbing mul-l14 change (``C4_NO_STACK0_EMIT``). DEFAULT OFF keeps
    every production build byte-identical to golden ``4958b35b``.
    """
    return os.environ.get("C4_FFN_LINT_CLEAN_DEMO", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Phase 11.A IR exposure: shared empty-IR factory for flag-gated ops whose
# bake bodies are no-ops when their gating flag is off.
def _empty_compiler_ir_factory(dim_positions, HD):
    """Return an empty ``CompilerIR``; used when a flag-gated bake is a no-op."""
    from ..ir import CompilerIR
    return CompilerIR()

# Reverse map (lazily populated): Opcode int value -> "OP_<NAME>" string
# for dim_positions lookup. Module-level so _SetDimProxy can be pickled.
_OP_NAME_CACHE: Dict[int, str] = {}


def _setdim_to_positions(BD) -> Dict[str, int]:
    """Build a ``dim_positions`` dict from a ``_SetDim``-like class.

    Phase 7.D.3 helper. Used by the legacy ``setup_token_embeddings`` /
    ``setup_head_weights`` fallback path when no ``dim_positions`` arg was
    supplied. Walks every public class attribute that resolves to an
    ``int`` so the returned mapping is closed.
    """
    positions: Dict[str, int] = {}
    for name in dir(BD):
        if name.startswith("_"):
            continue
        val = getattr(BD, name, None)
        if isinstance(val, int) and not isinstance(val, bool):
            positions[name] = val
    return positions


def _opcode_name_map() -> Dict[int, str]:
    """Return (and lazily build) the Opcode -> "OP_<NAME>" lookup map."""
    if _OP_NAME_CACHE:
        return _OP_NAME_CACHE
    from ...embedding import Opcode
    _OP_NAME_CACHE.update({
        Opcode.LEA: "OP_LEA", Opcode.IMM: "OP_IMM", Opcode.JMP: "OP_JMP",
        Opcode.JSR: "OP_JSR", Opcode.BZ: "OP_BZ", Opcode.BNZ: "OP_BNZ",
        Opcode.ENT: "OP_ENT", Opcode.ADJ: "OP_ADJ", Opcode.LEV: "OP_LEV",
        Opcode.LI: "OP_LI", Opcode.LC: "OP_LC", Opcode.SI: "OP_SI",
        Opcode.SC: "OP_SC", Opcode.PSH: "OP_PSH",
        Opcode.OR: "OP_OR", Opcode.XOR: "OP_XOR", Opcode.AND: "OP_AND",
        Opcode.EQ: "OP_EQ", Opcode.NE: "OP_NE", Opcode.LT: "OP_LT",
        Opcode.GT: "OP_GT", Opcode.LE: "OP_LE", Opcode.GE: "OP_GE",
        Opcode.SHL: "OP_SHL", Opcode.SHR: "OP_SHR",
        Opcode.ADD: "OP_ADD", Opcode.SUB: "OP_SUB", Opcode.MUL: "OP_MUL",
        Opcode.DIV: "OP_DIV", Opcode.MOD: "OP_MOD",
        Opcode.EXIT: "OP_EXIT", Opcode.NOP: "OP_NOP",
        Opcode.PUTCHAR: "OP_PUTCHAR", Opcode.GETCHAR: "OP_GETCHAR",
    })
    return _OP_NAME_CACHE


class _SetDimProxy:
    """BD-like object that resolves dim names via compiler-allocated positions.

    Module-level (not a closure) so instances of this class can be pickled —
    several runtime modules (``efficient_byte_alu``, ``efficient_wrappers``)
    hold a proxy as ``self.BD``, which means the model object must be
    picklable for the ``compile_full_vm_dynamic`` on-disk cache to work.

    Falls back to ``_SetDim`` for any attribute not in ``dim_positions``
    (e.g. constants like ``NUM_OPCODES``). ``opcode_dim`` resolves via
    ``dim_positions`` so OP_* writes land at compiler-allocated positions
    even under ``pin_io_only=True`` layouts.
    """

    def __init__(self, dim_positions: Dict[str, int]):
        # Store the dim_positions dict so __getattr__ / opcode_dim can use it
        # and so pickling round-trips correctly.
        object.__setattr__(self, "_dim_positions", dict(dim_positions))
        # Mirror declared dim positions as instance attributes for fast access
        # (matches the closure-based proxy's behavior).
        for name, pos in dim_positions.items():
            object.__setattr__(self, name, pos)

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup failed (so the mirrored
        # attrs above shadow this path for declared dims).
        from ...vm_step import _SetDim
        return getattr(_SetDim, name)

    def __getstate__(self):
        # Only persist the dim_positions; on load __setstate__ re-mirrors them.
        return {"_dim_positions": self._dim_positions}

    def __setstate__(self, state):
        object.__setattr__(self, "_dim_positions", dict(state["_dim_positions"]))
        for name, pos in self._dim_positions.items():
            object.__setattr__(self, name, pos)

    def opcode_dim(self, op_value):
        """Resolve op_value -> dim position via dim_positions (override).

        Falls back to _SetDim.opcode_dim if the OP_<NAME> entry isn't in
        dim_positions (e.g. opcodes that aren't declared by the compiler).
        """
        from ...vm_step import _SetDim
        name = _opcode_name_map().get(op_value)
        if name is not None and name in self._dim_positions:
            return self._dim_positions[name]
        return _SetDim.opcode_dim(op_value)


def _as_setdim_proxy(dim_positions: Dict[str, int]):
    """Build an object that mimics _SetDim using compiler dim positions.

    The original `_set_layerN_*` functions reference `BD.MARK_PC`, `BD.OUTPUT_LO`
    etc. We need to give them a BD-like object whose attribute values are the
    integer positions that the compiler chose.

    Returns an object where `proxy.MARK_PC == dim_positions['MARK_PC']`.
    Falls back to `_SetDim` for anything not declared (e.g., constants like
    ``NUM_OPCODES``).

    Note: ``proxy.opcode_dim(op_val)`` is overridden to resolve via
    ``dim_positions`` (looking up ``OP_<name>``) rather than ``_SetDim.OP_*``.
    Without this override, callers like ``_set_opcode_decode_ffn`` would
    write OP_* flags at the LEGACY ``_SetDim`` positions instead of the
    compiler-allocated ones, breaking pin_io_only=True layouts.

    If ``dim_positions`` is already a ``_SetDim``-like object (the legacy
    class itself or a previously-built proxy), return it unchanged. This
    lets legacy umbrella entry points (``vm_step._set_layerN_*``) that
    route through declarative IR factories pass ``BD = _SetDim`` directly
    instead of materializing a mirror dict.
    """
    if not isinstance(dim_positions, dict):
        # Already a class / proxy that supports ``.NAME`` attribute lookup
        # for the dim positions. Skip the dict→proxy wrap.
        return dim_positions
    return _SetDimProxy(dim_positions)


def _bake_post_op_into(ffn, post_op_instance, hidden_offset: int = 0) -> int:
    """Copy a post_op's weights into a target FFN starting at `hidden_offset`.

    The post_op classes (BinaryOpByteZeroingPostOp etc.) are PureFFN subclasses
    that bake their weights in __init__. We construct one and copy weights into
    the target block FFN's hidden-unit slots. Returns the next free hidden_offset.
    """
    H = post_op_instance.W_up.shape[0]
    end = hidden_offset + H
    target_H = ffn.W_up.shape[0]
    if end > target_H:
        raise ValueError(
            f"FFN hidden_dim={target_H} too small for post_op (needs +{H} at offset {hidden_offset})"
        )
    ffn.W_up.data[hidden_offset:end, :] = post_op_instance.W_up.data
    ffn.b_up.data[hidden_offset:end] = post_op_instance.b_up.data
    ffn.W_gate.data[hidden_offset:end, :] = post_op_instance.W_gate.data
    ffn.b_gate.data[hidden_offset:end] = post_op_instance.b_gate.data
    ffn.W_down.data[:, hidden_offset:end] = post_op_instance.W_down.data
    return end


# Per-ALU-class opcode gates. Each ALU module's forward applies an
# opcode_mask at the GE→BD writeback stage that zeros all residual writes
# except for the listed BD-format OP_* dims. Sources:
#   - AddSub5StageBlock (``efficient_alu_addsub_split.py``): op_add /
#     op_sub merge at stage 3 → OP_ADD, OP_SUB.
#   - FlattenedALUMul (``efficient_alu_neural.py:_MulCombineStage``):
#     op_mul gate at the combine stage → OP_MUL.
#   - ALUShiftComposite (``efficient_alu_neural.py:ALUShiftComposite``):
#     op_shl + op_shr merge → OP_SHL, OP_SHR.
#   - ALUAndOrXor (``efficient_alu_neural.py:PureNeuralALU(operations=
#     'bitwise')``): per-opcode OR/XOR/AND extract+combine + AX-marker
#     gate → OP_OR, OP_XOR, OP_AND.
# The attach op installs the module as a ``post_op`` whose forward gates
# every residual write on those opcodes. Declaring ``opcodes={...}`` on
# the attach op tells the per-opcode block-skip analyser the attached
# post-op layer is unreachable for any other opcode, unlocking the
# layer skip for the bulk of the opcode table.
_ALU_CLASS_OPCODES = {
    "ALUAddSub": {"OP_ADD", "OP_SUB"},
    "ALUMul": {"OP_MUL"},
    "ALUShift": {"OP_SHL", "OP_SHR"},
    "ALUAndOrXor": {"OP_AND", "OP_OR", "OP_XOR"},
}


def _make_alu_postop_attach_op(name: str, layer_idx: int, alu_cls_name: str,
                               alu_mode: str = 'lookup',
                               same_layer_as: str = None,
                               target_op_name: str = None) -> Operation:
    """Construct an ALU postop-attach Operation.

    Args:
        name: op name (e.g. ``l8_alu_postop_attach``).
        layer_idx: home layer for the postop attach.
        alu_cls_name: ALU class to instantiate (e.g. ``ALUAddSub``).
        alu_mode: ``'lookup'`` (only supported mode for now).
        same_layer_as: B12 backfill (plan §B12 / B10 schema). When provided,
            the returned ``Operation`` declares
            ``requires={"same_layer_as": same_layer_as}`` so the dynamic
            scheduler pins the postop-attach to the same layer as the
            wrapped ALU op (``layerN_alu`` / ``layer11_mul_partial`` etc.).
            Moves the op from ``phase_required_but_undeclared`` to
            ``phase_pinned_by_deps`` in ``analyze_scheduler``. None preserves
            pre-B12 behaviour (no ``requires`` declaration; analyzer flags
            it as ``phase_required_but_undeclared``).
    """
    if alu_mode != 'lookup':
        # TODO(efficient-mode): efficient alu_mode REPLACES ffn rather than
        # wrapping it (see vm_step.py:2385-2434), so the bake_fn semantics
        # differ. Migrate that branch in a follow-up.
        raise NotImplementedError(
            f"alu_mode={alu_mode!r} not yet supported for alu postop attach ops"
        )

    def bake(block, dim_positions, S):
        from ...vm_step import _SetDim
        from ... import efficient_alu_neural as eau
        # ALUAddSub has been replaced by the 5-stage flattened AddSub5StageBlock
        # (see efficient_alu_addsub_split.py). ALUMul / ALUShift are likewise
        # replaced by the already-flattened ``FlattenedALUMul`` /
        # ``ALUShiftComposite`` composites (nn.Sequential of PureFFNs, byte-
        # identical forward). Other ALU classes still come from
        # ``efficient_alu_neural``.
        proxy = _as_setdim_proxy(dim_positions)
        if alu_cls_name == "ALUAddSub":
            from ...efficient_alu_addsub_split import AddSub5StageBlock as alu_cls
            instance = alu_cls(S, proxy)
        elif alu_cls_name == "ALUMul":
            instance = eau.FlattenedALUMul.build_fully_baked(S, proxy)
        elif alu_cls_name == "ALUShift":
            instance = eau.ALUShiftComposite(S, proxy)
        else:
            alu_cls = getattr(eau, alu_cls_name)
            instance = alu_cls(S, proxy)
        # Attach as a post_op (rather than wrapping block.ffn with HybridALUBlock).
        # ``_expand_wrapper_blocks`` then splits each post_op into a passthrough
        # transformer block, preserving the original execution order.
        # Use compiler-allocated dim_positions (via proxy) so the structural
        # ALU wires inputs to layout-correct residual lanes; bare _SetDim
        # breaks pin_io_only=True (IO dims sit at different positions there).
        block.post_ops.insert(0, instance)

    # Phase=1180 + layer_idx*0.01: hybrid wraps must fire AFTER all FFN
    # bakes (including L14 cleanup and convo-IO ops at phases 8.5/10.6/15.1)
    # AND AFTER the dead-unit zero passes (l6_dead_unit_zero=1160,
    # l7_dead_unit_zero=1170 which require the original PureFFN), but BEFORE
    # right_size_ffns (1200) which prunes dead units after wrapping.
    #
    # B12 backfill: ``requires["same_layer_as"]`` pins the postop attach to
    # the same layer as the wrapped ALU op under the dynamic scheduler.
    # Without it, the analyzer flags this op as
    # ``phase_required_but_undeclared`` (the magic 1180+ phase carries the
    # ordering constraint but the dep DAG has no way to see it).
    requires: Dict[str, str] = {}
    if same_layer_as is not None:
        requires["same_layer_as"] = same_layer_as
    # Phase 8.G.6: drop the ``layer_idx=`` literal pin in favour of
    # ``target_op_name=`` pointing at an attn/ffn anchor that resolves to
    # the wrapped ALU op's layer. ``requires["same_layer_as"]`` is the
    # strict-mode dep-edge signal that this co-placement is intentional.
    # ``phase=1180+`` keeps the >= 100 post-pass short-circuit in the
    # strict admission gate (``_current_layer_for_strict`` returns
    # ``floor(phase) = 1180`` which the categoriser treats as ``ok``).
    #
    # The factories pass ``target_op_name=<wrapped op's anchor>`` because
    # the wrapped ``layerN_alu`` is itself a kind="block" op and
    # ``Operation.target_op_name`` can only reference attn/ffn ops (see
    # ``ModelLayout.resolve_block_op_layer``). When no ``target_op_name``
    # is provided we fall back to the legacy ``layer_idx`` pin (which
    # still triggers the strict-mode flag).
    return Operation(
        name=name,
        reads=set(),
        writes=set(),
        kind="block",
        target_op_name=target_op_name,
        layer_idx=layer_idx if target_op_name is None else None,
        bake_fn=bake,
        phase=1180 + layer_idx * 0.01,
        migrated=True,
        requires=requires,
        # Tier A opcode gating: the installed post_op module gates every
        # OUTPUT/CARRY write on the listed opcodes (see ``_ALU_CLASS_OPCODES``
        # above for the per-module derivations).
        opcodes=set(_ALU_CLASS_OPCODES.get(alu_cls_name, set())),
    )


def _ensure_l11_mul_module(block, S, dim_positions=None):
    """Get or install the FlattenedALUMul module on ``block.ffn``.

    The 9 phase-ordered installer ops each call this helper; the first one
    (lowest phase) installs the module, the rest re-use it. Idempotent.

    Args:
        block: target transformer block (FlattenedALUMul is installed on
            ``block.ffn``).
        S: number of sequence positions (passed through to FlattenedALUMul).
        dim_positions: optional dict mapping dim name -> start position. When
            provided, FlattenedALUMul receives a `_SetDim`-shaped proxy whose
            attribute values are the compiler-allocated positions; without
            this, the sub-stages (BDToGEConverter, _BDToGEStage,
            _MulCombineStage, _GEToBDStage) bake weights at the legacy
            ``_SetDim`` positions and silently mis-address ALU_LO/HI,
            AX_CARRY_LO/HI, OP_MUL, OUTPUT_LO/HI etc. under
            ``pin_io_only=True``. Falls back to raw ``_SetDim`` when None
            (legacy hand-set callers).
    """
    from ...efficient_alu_neural import FlattenedALUMul
    from ...vm_step import _SetDim
    existing = getattr(block, "ffn", None)
    if isinstance(existing, FlattenedALUMul):
        return existing
    BD = _as_setdim_proxy(dim_positions) if dim_positions is not None else _SetDim
    module = FlattenedALUMul(S, BD)
    block.ffn = module
    return module


# ---------------------------------------------------------------------------
# 4-stage SHL/SHR ops (efficient-mode replacement for ALUShift wrapper).
#
# Each op is kind="ffn" at phase=13 so the 4 ops + ``layer13_shifts`` all
# share L13's FFN slot (phase-equality => shared (layer, kind) slot per
# ``LayerCompiler._assign_layers``). The bake_fns cooperate:
#
#   1. bdtoge  : install the ``ALUShiftComposite`` on ``block.ffn`` and assign
#                the bdtoge stage. Subsequent bakes look up the existing
#                composite via ``block.ffn``.
#   2. precompute : assign the precompute stage onto the composite. (No-op if
#                   the composite was already fully built by another path.)
#   3. select  : same for select.
#   4. getobd  : same for getobd.
#
# Conceptually these are 4 distinct compiler ops carrying ownership of the
# 4 sub-FFN stages. Mechanically they share one layer because the rest of
# ``set_vm_weights`` (legacy_bake) still hardcodes ``model.blocks[14..16]``
# for downstream layers; spreading the stages across 4 layers would shift
# those indices and break that legacy bake until it migrates too. Once the
# downstream legacy bakes follow the layout, the phases can be split into
# 13.0/13.1/13.2/13.3 and the stages will land in their own layers.
# ---------------------------------------------------------------------------


class _ALUShiftCompositeBuilder:
    """Mutable holder shared across the 4 stage bake_fns + the install op.

    The compiler may assign the 4 ffn stage ops to whichever block its dep
    analyser picks (often a block far from the legacy ``model.blocks[13]``).
    The install op (kind="block", layer_idx=13) is what actually swaps the
    L13 ``block.ffn`` for the composite. The shared builder lets stage bakes
    populate the composite from any FFN module they happen to receive.
    """

    def __init__(self):
        self.composite = None

    def ensure(self, S, BD_proxy):
        from ...efficient_alu_neural import ALUShiftComposite
        if self.composite is None:
            self.composite = ALUShiftComposite(S, BD_proxy)
        return self.composite


# ---------------------------------------------------------------------------
# L10 DIV/MOD ALU flattening (2026-05-10)
#
# The previous lookup-mode override
#   model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)
# in ``set_vm_weights`` and the efficient-mode append
#   block.post_ops.append(EfficientDivMod_Neural(S, _SetDim))
# in ``make_l10_post_op_attach_op`` both wrapped 3 logical sub-stages
# (BD→GE convert, long-division pipeline, GE→BD convert) inside a single
# ``PureNeuralALU(operations='div_mod')`` runtime class (alias
# ``ALUDivMod`` / ``EfficientDivMod_Neural``). The 4 ops below split that
# wrapper into discrete compiler operations:
#
#   phase=10.0  install BD → GE converter         (FlattenedDivMod.bd_to_ge)
#   phase=10.1  install long-division pipeline    (FlattenedDivMod.div_layers + mod_layers)
#                                                  = ClearDivSlotsFFN +
#                                                    LongDivisionModule +
#                                                    EmitDivResultModule per opcode
#   phase=10.2  install GE → BD converter         (FlattenedDivMod.ge_to_bd)
#   phase=10.8  install composite onto post_ops   (model.blocks[10].post_ops.append)
#
# The first 3 stage ops are kind="block", layer_idx=10. They run after
# `make_l10_post_op_attach_op` (phase=10.7) since 10.0/10.1/10.2 are < 10.7
# only in numeric-phase comparison — but since BLOCK ops sort by
# (layer_idx, phase), the smaller phases run FIRST. That's fine: the
# first 3 ops only construct sub-stages on a builder; nothing depends on
# `block.post_ops` until the install op (phase=10.8) actually inserts
# the composite.
#
# The install op (phase=10.8, kind="block", layer_idx=10) appends the
# fully-constructed FlattenedDivMod composite to ``block.post_ops``.
# It runs AFTER `make_l10_post_op_attach_op` (phase=10.7) which appends
# the standard L10 post_ops (BinaryOpByteZeroingPostOp etc.) but no longer
# appends EfficientDivMod_Neural / DivModModule.
#
# The legacy lookup-mode override in set_vm_weights
# (`model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)`) is
# also removed so the composite isn't clobbered.
#
# Forward is byte-identical to the previous EfficientDivMod_Neural — see
# ``FlattenedDivMod.forward`` in efficient_alu_divmod_split.py.
# ---------------------------------------------------------------------------


class _FlattenedDivModBuilder:
    """Mutable holder shared across the 4 cooperating ops.

    Each of the 4 ops accesses the same ``FlattenedDivMod`` instance via
    this builder. Stage ops (phase=10.0/10.1/10.2) install one sub-stage
    each; the install op (phase=10.8) appends the fully-assembled composite
    to ``model.blocks[10].post_ops``.

    Idempotent: ``ensure`` returns the existing composite if any.
    """

    def __init__(self):
        self.composite = None

    def ensure(self, S, BD_proxy):
        from ...efficient_alu_divmod_split import FlattenedDivMod
        if self.composite is None:
            self.composite = FlattenedDivMod(S, BD_proxy)
        return self.composite


def setup_token_embeddings(embed_weight, dim_positions: Dict[str, int] = None) -> None:
    """Bake the per-token embedding values using compiler dim positions.

    Phase 0 M4 (2026-05-09): extracted from vm_step.set_vm_weights so the
    compiler path uses auto-allocated positions. Falls back to _SetDim when
    dim_positions is None.

    Phase 7.D.3 migration: replaced the per-token imperative writes with a
    call to ``CompilerIR.lower_token_embeddings`` using the same
    ``_embedding_bake_rules`` list that the active production op
    ``make_embedding_bake_op`` uses -- single source of truth.

    Args:
        embed_weight: nn.Embedding.weight tensor [vocab, d_model].
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ...vm_step import _SetDim
    from ..ir import CompilerIR
    # Local import to avoid module-load cycle (model_ops imports from shared).
    from .model_ops import _embedding_bake_rules

    if dim_positions is None:
        dim_positions = _setdim_to_positions(_SetDim)

    V = embed_weight.shape[0]

    with torch.no_grad():
        embed_weight.zero_()

    # Model-like shim so ``lower_token_embeddings`` can resolve
    # ``model.embed.embed.weight``. ``.head`` is stubbed so attribute
    # resolution succeeds even though the embedding bake never writes there.
    class _InnerEmbed:
        def __init__(self, w):
            self.weight = w

    class _OuterEmbed:
        def __init__(self, w):
            self.embed = _InnerEmbed(w)

    class _ModelShim:
        def __init__(self, w):
            self.embed = _OuterEmbed(w)
            self.head = None

    ir = CompilerIR()
    ir.embeddings.extend(_embedding_bake_rules(V))
    ir.lower_token_embeddings(_ModelShim(embed_weight), dim_positions)


def setup_head_weights(head, dim_positions: Dict[str, int] = None) -> None:
    """Bake the output-projection head weights using compiler dim positions.

    Phase 0 M4 (2026-05-09): extracted from vm_step.set_vm_weights so the
    compiler path can call it with auto-allocated dim positions instead of
    _SetDim constants. When `dim_positions` is None, falls back to _SetDim
    (backward-compat with hand-set path).

    Phase 7.D.3 migration: replaced the per-byte / per-marker imperative
    writes with a call to ``CompilerIR.lower_token_embeddings`` using the
    same ``_head_bake_rules`` list that the active production op
    ``make_head_bake_op`` uses -- single source of truth.

    Args:
        head: The model.head nn.Linear(d_model, vocab_size) module.
        dim_positions: Optional dict mapping dim name -> start position.
    """
    import torch
    from ...vm_step import _SetDim
    from ..ir import CompilerIR
    # Local import to avoid module-load cycle (model_ops imports from shared).
    from .model_ops import _head_bake_rules

    if dim_positions is None:
        dim_positions = _setdim_to_positions(_SetDim)

    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()

    # Model-like shim so ``lower_token_embeddings`` can resolve
    # ``model.head.weight`` / ``model.head.bias``. ``lower_token_embeddings``
    # also touches ``model.embed.embed.weight`` to set ``embed_weight`` when
    # ``chosen`` is truthy, even when no embed-target rules exist; stub
    # ``.embed`` to a no-op object so attribute resolution succeeds.
    class _NullEmbed:
        weight = None

    class _NullOuter:
        embed = _NullEmbed()

    class _ModelShim:
        def __init__(self, h):
            self.head = h
            self.embed = _NullOuter()

    vocab_size = head.weight.shape[0]
    ir = CompilerIR()
    ir.embeddings.extend(_head_bake_rules(vocab_size, dim_positions))
    ir.lower_token_embeddings(_ModelShim(head), dim_positions)


# ---------------------------------------------------------------------------
# Dim spec compatible with _SetDim
# ---------------------------------------------------------------------------

# Known limitation of the migration shims:
#
# Many ops both *read* and *write* dims like OUTPUT_LO/EMBED_LO. The reads happen
# at one position (e.g., MARK_PC) and writes at another (e.g., MARK_AX). My
# Operation declarations use dim *names* without position context, so the compiler
# can see both ops reading/writing the same name and infer a circular dependency
# where none truly exists. This is a real architectural limitation of the current
# LayerCompiler dep model — the next refinement needs per-position reads/writes
# (e.g., "EMBED_LO@MARK_PC" vs "EMBED_LO@MARK_AX") so the compiler can distinguish
# "reading the previous position's value" from "writing this position's value".
#
# Until that refinement, all_core_ops() compiled together produces a cycle. The
# work-around for now: the unit tests only exercise small subsets that don't
# create cycles, and full-spec compilation isn't wired to production.


# IO-required dim names that MUST stay pinned to their _SetDim positions even
# when the compiler is otherwise free to bump-pointer-allocate. These dims are
# read or written by external (non-bake) code paths — token embedding setup,
# the output head, and `NeuralVMEmbedding._inject_*` runtime injectors — that
# resolve dim positions either through the `_SetDim` enum directly or through
# `dim_positions` lookups that must agree with `_SetDim` for now.
#
# Membership rationale (cross-checked against
# `c4_release/neural_vm/neural_embedding.py:_inject_*`):
#
# - EMBED_LO/HI, OUTPUT_LO/HI: nibble-decode/projection. Token embedding sets
#   EMBED_*; head reads OUTPUT_*. _inject_initial_pc writes EMBED_*.
# - MARK_PC/AX/SP/BP/MEM/SE/STACK0/CS/SE_ONLY: per-token marker flags set by
#   token embedding; threshold heads scan for them.
# - NEXT_*: head reads these to project to token-type logits.
# - IS_BYTE/IS_MARK/CONST/HAS_SE/BYTE_INDEX_*: positional flags read by head
#   gating and by L0 thresholds.
# - OP_LEV/BZ/BNZ: decoded at MARK_PC by the L5 FFN all-step PC-marker
#   opcode decode (see vm_step.py); set at MARK_AX by the standard L5
#   opcode decoder. ACTIVE_OPCODE_PRTF/READ: legacy conversational-I/O
#   layout placeholders (no longer written from Python).
# - MARK_THINKING_START/END: baked into the embedding table on
#   THINKING_START/END tokens (see ``setup_token_embeddings``).
# - MEM_STORE / ADDR_KEY: written by `_inject_mem_store` /
#   `_inject_mem_metadata` for memory ops. (MEM_EXEC@468 is retained in the
#   IO set as a layout placeholder — Phase A 2026-05-11 removed the writes
#   and external-hints API but kept the dim slot so the compact-IO layout
#   stays stable. The slot is aliased by IO_FORMAT_POS.)
# - NEXT_TOOL_CALL / NEXT_THINKING_START / NEXT_THINKING_END: optional head
#   reads when conversational I/O is enabled (see setup_head_weights).
_IO_REQUIRED_DIMS = frozenset({
    # Markers (token embedding writes; threshold heads read)
    "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_SE",
    "MARK_CS", "MARK_SE_ONLY", "MARK_STACK0",
    "MARK_THINKING_START", "MARK_THINKING_END",
    # Positional flags (head + L0 thresholds)
    "IS_BYTE", "IS_MARK", "CONST", "HAS_SE",
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    # Nibble encoding (token embed in / head out / _inject_initial_pc)
    "EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
    "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
    # NEXT_* token-type transition flags (head reads)
    "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0",
    "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
    "NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END",
    # Active-opcode dims (decoded at PC by L5 FFN; layout retained for
    # ACTIVE_OPCODE_PRTF/READ as conversational-I/O placeholders).
    "OP_LEV", "OP_BZ", "OP_BNZ",
    "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
    # Memory injection slots (_inject_mem_store, _inject_mem_metadata).
    # MEM_EXEC is a retained placeholder — see header comment above.
    "MEM_STORE", "MEM_EXEC", "ADDR_KEY",
})


def declare_setdim_compat_dims(
    compiler,
    pin_to_setdim: bool = True,
    pin_io_only: bool = False,
) -> None:
    """Declare to a LayerCompiler all dims that match the existing _SetDim layout.

    Args:
        compiler: LayerCompiler to declare dims to
        pin_to_setdim: if True, each dim is pinned to its _SetDim position. This
            preserves _SetDim's aliasing scheme (e.g., FETCH_LO==MUL_ACCUM at
            position 420) so existing _set_layerN_* bake_fns work unchanged. If
            False, dims are bump-pointer allocated by declaration order — useful
            for testing the auto-allocation path but breaks _SetDim aliases.
        pin_io_only: if True, the dims in `_IO_REQUIRED_DIMS` (the
            externally-observable dims read/written by token embedding, the
            output head, and `NeuralVMEmbedding._inject_*` runtime injectors)
            are pinned to a *compact, contiguous block starting at position
            0*, in declaration order. Every non-IO dim is bump-pointer
            allocated by the compiler above the IO block. This unlocks
            compiler-driven internal dim allocation AND shrinks d_model:
            instead of pinning IO dims at their scattered `_SetDim` positions
            (which span up to ~507 with large gaps, forcing unpinned dims to
            stack on top for d_model ~1038), they are laid out densely so
            d_model collapses to roughly (IO total size) + (non-IO total
            size). Code that still reads `_SetDim.X` *directly* will get the
            wrong position — all baked weights must resolve dim positions
            through `dim_positions` (e.g., via `_as_setdim_proxy`). The
            `pin_to_setdim` flag is ignored when `pin_io_only=True`. Defaults
            to False for backward compatibility.
    """
    from ...vm_step import _SetDim
    from ...constants import INSTR_WIDTH  # noqa: F401 (touched for completeness)

    # Single-dim flags
    one_dim = [
        "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
        "MARK_SE", "IS_BYTE", "IS_MARK", "CONST", "MARK_CS",
        "MARK_SE_ONLY", "MARK_STACK0",
        "MARK_THINKING_START", "MARK_THINKING_END",
        "ACTIVE_OPCODE_PRTF", "ACTIVE_OPCODE_READ",
        "HAS_SE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "STACK0_BYTE0", "CMP_GROUP",
        # B7-1: in-step freshness lifecycle bit (L1 attn head 5; see
        # _SetDim.IN_STEP_FRESH docstring for semantics).
        "IN_STEP_FRESH",
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_STACK0",
        "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
        "NEXT_TOOL_CALL", "NEXT_THINKING_START", "NEXT_THINKING_END",
        "IO_IS_PUTCHAR", "IO_OUTPUT_READY",
        "IO_IN_OUTPUT_MODE", "IO_OUTPUT_COMPLETE",
        "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
        "OP_ENT", "OP_ADJ", "OP_LEV",
        # Phase 9.C: OP_LEV_PREV_STEP alias retired - corpus reads now
        # use SSA spellings (OP_LEV.<writer>.-1) instead of the
        # numeric alias.
        "OP_LI", "OP_LC",
        "OP_SI", "OP_SC", "OP_PSH",
        "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_NE", "OP_LT",
        "OP_GT", "OP_LE", "OP_GE", "OP_SHL", "OP_SHR",
        "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
        "OP_EXIT", "OP_NOP", "OP_PUTCHAR", "OP_GETCHAR",
        "MEM_STORE", "MEM_ADDR_SRC",
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        # MEM_EXEC is a layout placeholder — its writes were removed in
        # Phase A (2026-05-11) but the slot is retained so the compact-IO
        # layout is stable. IO_FORMAT_POS@468 aliases MEM_EXEC.
        "OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP", "MEM_EXEC",
        "OPCODE_BASE",
        # B7-4 / B6-K slot 97: ADDR_B0 lifecycle "VALID" bit. Written 1.0 by
        # L13 mem-addr-gather at MEM val byte positions whenever the ADDR_B0
        # one-hot lanes carry freshly-computed nibbles. L10 tail addr0 family
        # gates on this to distinguish fresh ADDR_B0 evidence from stale
        # residue (see B4-H tail-correction-family PLAN §3.2).
        "ADDR_B0_VALID",
        # Conversational I/O state (aliases noted in _SetDim):
        # IO_FORMAT_POS@468 aliases MEM_EXEC, IO_IN_OUTPUT_MODE@469 and
        # IO_OUTPUT_COMPLETE@470 are dedicated, LAST_WAS_BYTE@503 is
        # dedicated. Declared unconditionally so the compiler accepts the
        # convo-io migrated ops' reads/writes even when the flag is False.
        "IO_FORMAT_POS", "IO_IN_OUTPUT_MODE", "IO_OUTPUT_COMPLETE",
        "LAST_WAS_BYTE",
        # Conversational I/O state dims that were previously left out of the
        # declaration list and fell back to bare `_SetDim` positions via the
        # proxy. With `pin_io_only=True` those legacy positions collide with
        # compiler-allocated dims (ALU_LO, AX_FULL_*, OPCODE_BYTE_HI, ...);
        # declare them so the compiler hands out unique positions. Bake
        # functions that reference them (L2 lookback head, L3 state init,
        # null-terminator detection) only fire when conversational I/O is
        # enabled — they remain unused under the default smoke config but
        # must have collision-free positions in either layout.
        "LAST_WAS_THINKING_START", "LAST_WAS_THINKING_END",
        "LAST_WAS_IO_STATE_EMIT_BYTE", "LAST_WAS_IO_STATE_EMIT_THINKING",
        "IO_IS_PRTF", "IO_IS_READ", "IO_STATE", "IO_OUTPUT_COUNT",
        "IO_IS_TOOL_CALL",
        "NEXT_IO_STATE_EMIT_BYTE", "NEXT_IO_STATE_EMIT_THINKING",
        # B7-2 SP_BYTE0_IS_F8: L7 head-6 producer, 1.0 only when carry-forward
        # proves SP byte 0 is 0xF8 (slot 95 — dead L0 H5+0).
        "SP_BYTE0_IS_F8",
        # B7-1 IN_STEP_FRESH: L1 head-5 producer with ALiBi slope 0.5,
        # decays from 1.0 immediately after STEP_BOUNDARY toward 0.0;
        # resets at next STEP_BOUNDARY. Replaces HAS_SE -1e9 hammer
        # for L10 tail_* rules (slot 96 — dead L0 H5+1).
        "IN_STEP_FRESH",
        # B7-5 SP_GATHERED_THIS_STEP: 1.0 at MARK_SP positions after L8
        # SP gather has fired in the current step. Produced by
        # ``make_layer8_sp_gathered_sentinel_op`` (L8 FFN, phase 8.6);
        # consumed by L10 tail_sp_marker_* rules. See
        # ``investigation/bd-dim-usage-map`` REPORT Section 5.
        "SP_GATHERED_THIS_STEP",
        # C5 BZ branch-target re-fire fix (2026-06-09): 1.0 at MARK_PC
        # positions on BZ-taken steps. Written by
        # ``post_l9_bz_bnz_pc_override``; consumed by the SAME op on the
        # NEXT step via the ``BZ_TARGET_FRESH.*.-1`` cross-step alias as
        # a gate term on the OUTPUT_LO cancel band. See
        # ``docs/BZ_TARGET_FRESH_CROSS_STEP_2026_06_09.md``.
        "BZ_TARGET_FRESH",
    ]
    # 7-dim threshold head outputs (one per marker type)
    # H1_DUMP is a same-position alias of H1 (slots 67..73) declared AFTER
    # H1 so the alias machinery picks up H1's resolved position. The AX
    # byte-1 DUMP carry head writes the re-supplied byte-1 H1 one-hot here
    # on carried steps; distinct name keeps it off the 54 same-step H1
    # readers' dep-graph edges. See _ALIAS_OF below.
    seven_dim = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
                 "L1H0", "L1H1", "L1H2", "L1H4", "L2H0", "H1_DUMP"]
    # 16-dim nibble groups
    sixteen_dim = ["EMBED_LO",
                   "EMBED_HI",
                   "OUTPUT_LO", "OUTPUT_HI",
                   # B9 OUTPUT_HI split: OUTPUT_HI_THIS_STEP is the
                   # canonical name for the same-step write band. Same
                   # numeric base as OUTPUT_HI in _SetDim (190) so baked
                   # weight indices are byte-identical; the alias keeps
                   # ``BD.OUTPUT_HI`` lookups in legacy bake bodies
                   # working unchanged. Phase 9.C retired the sibling
                   # ``OUTPUT_HI_PREV_STEP`` alias - cross-step readers
                   # (layer3_carry_forward_attn head 5,
                   # layer8_head6_ax_carry_refresh) now use SSA reads
                   # plus ``requires["after"]`` for the cross-step
                   # boundary. See docs/B9_OUTPUT_HI_SPLIT_SPEC.md.
                   "OUTPUT_HI_THIS_STEP",
                   "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
                   "FETCH_LO", "FETCH_HI", "MUL_ACCUM", "DIV_STAGING",
                   "AX_FULL_LO", "AX_FULL_HI",
                   "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                   "ADDR_B0_LO",
                   "ADDR_B1_LO",
                   "ADDR_B2_LO",
                   "ADDR_B0_HI",
                   "ADDR_B1_HI",
                   "ADDR_B2_HI",
                   "FORMAT_PTR_LO", "FORMAT_PTR_HI",
                   "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"]
    # Phase 9.C: CARRY/CMP/ADDR_KEY/TEMP PREV_STEP aliases retired -
    # corpus reads now use SSA spellings (``<DIM>.<writer>.-1``).
    four_dim = ["CARRY"]
    eight_dim = ["CMP"]
    forty_eight_dim = ["ADDR_KEY"]
    thirty_two_dim = ["TEMP"]

    # Cursor for the compact IO block when pin_io_only=True. IO dims are
    # pinned at consecutive positions starting at 0, in declaration order
    # (the order of the `one_dim` / `seven_dim` / ... lists below). Non-IO
    # dims are left unpinned and bump-pointer-allocated above the IO block
    # by `_allocate_dims`.
    io_cursor = [0]

    # B9 OUTPUT_HI split: alias map. Dim names on the LHS share the same
    # numeric position as the dim on the RHS. The alias must be declared
    # AFTER the base in the relevant size-bucket list so the base's
    # pinned position is already in ``compiler._pinned`` before the alias
    # is declared. See docs/B9_OUTPUT_HI_SPLIT_SPEC.md §6.4.
    _ALIAS_OF = {
        "OUTPUT_HI_THIS_STEP": "OUTPUT_HI",
        # AX byte-1 DUMP carry: H1_DUMP shares H1's 7 slots (67..73). The
        # carry head writes the re-supplied byte-1 one-hot here on carried
        # steps; the LM head reads the physical slots so emission is
        # byte-identical, while the distinct name avoids the H1-write
        # 2-cycle. See vm_step.py:_SetDim.H1_DUMP.
        "H1_DUMP": "H1",
        # Phase 9.C: all ``*_PREV_STEP`` aliases (OUTPUT_HI/LO, TEMP,
        # ADDR_KEY, ALU_LO, AX_CARRY_{LO,HI}, EMBED_{LO,HI}, ADDR_B*_*,
        # CARRY, CMP, OP_LEV, OPCODE_BYTE_LO) were retired now that
        # Phase 9.B migrated every cross-step reader to its SSA spelling
        # (``<DIM>.<writer>.-1``). The numeric-position aliasing was
        # purely cosmetic since each alias shared its base's position.
    }

    def _declare(name, size):
        if not hasattr(_SetDim, name):
            return
        # Aliases inherit the base dim's position (in BOTH pinning modes)
        # so byte-identical residual cells are guaranteed regardless of
        # compaction layout. We declare via ``alias_of=`` so the compiler
        # resolves the position at _allocate_dims time, even when the base
        # is bump-pointer-allocated (which happens in pin_io_only=True
        # mode for non-IO-required dims like AX_CARRY_LO/HI, ALU_LO,
        # ADDR_KEY, TEMP, OUTPUT_LO).
        base = _ALIAS_OF.get(name)
        if base is not None:
            existing = getattr(compiler, "_pinned", {}) or {}
            if base in existing:
                pinned = existing[base]
            else:
                # Base is unpinned (will be bump-allocated). Fall back to
                # _SetDim if pin_to_setdim is set; otherwise leave
                # pinned=None — the compiler's alias machinery resolves the
                # position post-allocation via the ``alias_of=`` link.
                pinned = getattr(_SetDim, base, None) if pin_to_setdim else None
            compiler.declare_dim(name, size, pinned=pinned, alias_of=base)
            return
        if pin_io_only:
            if name in _IO_REQUIRED_DIMS:
                # Compact: assign consecutive positions starting at 0,
                # ignoring _SetDim's scattered legacy positions. Without
                # this compaction, IO dims pinned at their _SetDim positions
                # leave huge gaps (max IO position ~507) and force unpinned
                # dims to stack on top, producing d_model ~1038.
                pinned = io_cursor[0]
                io_cursor[0] += size
            else:
                pinned = None
        else:
            pinned = getattr(_SetDim, name) if pin_to_setdim else None
        compiler.declare_dim(name, size, pinned=pinned)

    for name in one_dim:
        _declare(name, 1)
    for name in seven_dim:
        _declare(name, 7)
    for name in sixteen_dim:
        _declare(name, 16)
    for name in four_dim:
        _declare(name, 4)
    for name in eight_dim:
        _declare(name, 8)
    for name in forty_eight_dim:
        _declare(name, 48)
    for name in thirty_two_dim:
        _declare(name, 32)
    # Internal-only STACK0 byte flags. Declare these last so adding them does
    # not renumber any pre-existing compiler-allocated non-IO dims.
    for name in ("STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3"):
        _declare(name, 1)

    # V2/G7 LEV detector head output dims (Phase 9 spike from
    # docs/CONTROL_FLOW_DETECTOR_HEADS.md §2.3). These are fresh residual
    # bands written by ``make_lev_detector_head_op`` (L8 attn head, default
    # ``enable=False``). The head's V/O projection materialises the saved
    # PC / BP / SP from the prev-step LEV row at the current-step PC/BP/SP
    # marker rows; downstream readers (L9 alu, L8 sp_gather_bake) can prefer
    # the detector dim when present, falling back to the existing
    # OUTPUT_LO/HI writes from L16 ``layer16_lev_routing`` on non-LEV-
    # following steps.
    #
    # The dims are declared unconditionally so the residual layout / dim
    # registry is stable regardless of the head's ``enable`` flag. The
    # explicit ``compiler.declare_dim`` call bypasses the
    # ``hasattr(_SetDim, name)`` guard inside ``_declare`` -- these slots
    # have no ``_SetDim`` legacy position (they are V2-native and emerge
    # from the bump-pointer allocator above the STACK0_BYTE3 high-water
    # mark; ``pinned=None`` lets the compiler pick the lowest free slot).
    #
    # PC has lo/hi nibble pair (mirrors OUTPUT_LO/HI structure); BP and SP
    # are single 16-wide bands. See docs/CONTROL_FLOW_DETECTOR_HEADS.md
    # §2.3 for the residual-stream rationale and §2.2 for the V/O write
    # table.
    for name in (
        "PC_VIA_LEV_DETECTOR_LO",
        "PC_VIA_LEV_DETECTOR_HI",
        "BP_VIA_LEV_DETECTOR",
        "SP_VIA_LEV_DETECTOR",
    ):
        compiler.declare_dim(name, 16, pinned=None)

    # Wave 1 A3: STACK0_BYTE_VAL_h_LO/HI family. 16-wide nibble bands
    # holding the AX byte h value broadcast to the matching STACK0 byte
    # row during PSH. Producer is ``layer10_psh_ax_broadcast`` (3 new
    # heads at L10 slots 8/9/10); consumer is the L14 ``mem_generation``
    # read migration (heads 5/6/7). The dim family was scaffolded in
    # Wave 1 A1 (commit c31897aa); _SetDim has no legacy positions for
    # these slots, so ``pinned=None`` lets the bump-pointer allocator
    # place them above the high-water mark in the compat path.
    # See docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md.
    for name in (
        "STACK0_BYTE_VAL_1_LO",
        "STACK0_BYTE_VAL_1_HI",
        "STACK0_BYTE_VAL_2_LO",
        "STACK0_BYTE_VAL_2_HI",
        "STACK0_BYTE_VAL_3_LO",
        "STACK0_BYTE_VAL_3_HI",
    ):
        compiler.declare_dim(name, 16, pinned=None)

    # 2026-06-10: STEP_END register-presence broadcast family. Each
    # ``SE_REG_<MARK>_PRESENT`` is a 1-wide flag written at MARK_SE_ONLY
    # rows by the L1 ``layer1_threshold_attn`` head 6 within-step relay
    # (only SE_REG_AX_PRESENT is wired in this commit; the other 5 slots
    # are declared scaffolding for follow-on broadcast heads). All slots
    # are unpinned so the bump-pointer allocator places them above the
    # wave-aligned high-water mark in the compat path. See
    # docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md.
    for name in (
        "SE_REG_AX_PRESENT",
        "SE_REG_PC_PRESENT",
        "SE_REG_SP_PRESENT",
        "SE_REG_BP_PRESENT",
        "SE_REG_STACK0_PRESENT",
        "SE_REG_MEM_PRESENT",
    ):
        compiler.declare_dim(name, 1, pinned=None)

    # 2026-06-10 Wave A v2: register-tagged STEP_END operand relay.
    # Written at MARK_SE_ONLY by ``layer9_step_end_operand_relay`` (two
    # attn heads in L9 attn), consumed by the migrated L9 CMP rules
    # (``_layer9_cmp_rules``, gated on MARK_SE_ONLY). The SE_ prefix
    # keeps the operand bands scoped to the SE row so they do not
    # collide with downstream readers of the raw ALU_LO/HI/CARRY/CMP
    # bands. Pinned to the ``_SetDim.SE_*`` positions (837..911) so
    # they sit above the existing dim layout and don't share slots
    # via the liveness allocator with unrelated live dims (the L9 CMP
    # rules' writes to SE_<NAME> at MARK_SE_ONLY would clobber any
    # collided dim's writes at non-SE rows otherwise). See
    # ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md`` and memory
    # note ``project_wave_b_cmp_needs_l9_internal_relay.md``.
    # The SE_* relay dims MUST NOT share slots with other live dims:
    # the relay's attention head produces a small leak at non-SE rows
    # (softmax over -ALiBi penalty distributes some mass to nearby K
    # rows, producing residual contribution at non-SE Q rows even with
    # the score-only slot-0 design). Sharing a slot with a dim that
    # fires at MARK_AX (e.g. IN_STEP_FRESH, SP_BYTE0_IS_F8) would
    # corrupt the partner's value. Leave the dims unpinned so the
    # liveness allocator places them above the existing layout BUT
    # the auto-share is disabled for SE_* by the bumped d_model: with
    # explicit size declarations the bump-pointer puts them above
    # the IO/scratch high-water mark unless the liveness pass coalesces
    # them. The L1 head 6 SE_REG_* family takes the same approach.
    for name, size in (
        ("SE_ALU_LO", 16),
        ("SE_ALU_HI", 16),
        ("SE_AX_CARRY_LO", 16),
        ("SE_AX_CARRY_HI", 16),
        ("SE_CMP", 4),
        ("SE_OP_EQ", 1),
        ("SE_OP_NE", 1),
        ("SE_OP_LT", 1),
        ("SE_OP_GT", 1),
        ("SE_OP_LE", 1),
        ("SE_OP_GE", 1),
        ("SE_CMP_GROUP", 1),
    ):
        compiler.declare_dim(name, size, pinned=None)

    # ------------------------------------------------------------------
    # Qwen R1 — opt-in NORM_COMPENSATOR slot
    # ------------------------------------------------------------------
    # When ``C4_QWEN_EXPORT_COMPAT=1`` is set, declare a width-1 dim
    # that the ``norm_compensator_seed`` model bake populates with the
    # known constant ``K`` (1000.0) for every token id. Declared as
    # unpinned so the bump-pointer allocator places it at the next
    # available position above the wave-aligned IO/scratch high-water
    # mark. With the flag OFF the dim is not declared, so d_model and
    # the residual layout stay byte-identical to pre-R1 main.
    # See ``make_norm_compensator_seed_op`` and
    # docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md §R1.
    import os as _os
    if _os.environ.get("C4_QWEN_EXPORT_COMPAT") == "1":
        compiler.declare_dim("NORM_COMPENSATOR", 1, pinned=None)

    # ------------------------------------------------------------------
    # width=2 MUL — MUL_RESULT_HI byte-1 result band (2026-06-13)
    # ------------------------------------------------------------------
    # The dedicated high-byte result band for the width=2 (8-bit x 8-bit
    # -> 16-bit) MUL (MUL_RESULT_HI_LO/HI; nib2 -> _LO, nib3 -> _HI) is NO
    # LONGER declared here. It is now declared OP-LOCALLY (next to the wide_mul
    # op in ``ops/alu_ops.py``) via ``register_residual_band(..., flag=
    # mul_width2_enabled)`` and AUTO-COLLECTED into ``extra_residual_dims`` at
    # the top of ``compile_full_vm_dynamic`` (see
    # ``ops/residual_band_registry.py``) so the d_model widen is
    # HEAD-DIM-PRESERVING (the auto-widen captures the base head_dim BEFORE the
    # extra bands are declared and rounds up to a multiple of it, ADDING heads
    # instead of repartitioning existing ones).
    # Declaring it here ran BEFORE the base_head_dim capture and re-derived
    # head_dim from the widened width, scrambling attention -> regressed
    # test_bnz_branch. See ``compile_full_vm_dynamic`` and
    # docs/MUL_WIDTH2_WIDEN_2026_06_13.md. With the flag OFF nothing is
    # declared, so d_model / the residual layout stay byte-identical to
    # pre-width2 main (920).
