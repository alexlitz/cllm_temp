"""Unit tests for the activation-scale gate derivation (task #395).

Fast, GPU-free: exercises ``ops.shared.derive_gate`` +
``verification.activation_scales`` against the hand CONTROL gate constants,
proving the FULL gate (positive weights + threshold + blockers) derives from the
activation-scale datum, verdict-preserving by construction.
"""

from neural_vm.verification.activation_scales import (
    ActivationScales, load_activation_scales,
)
from neural_vm.unified_compiler.ops.shared import (
    derive_gate, derive_pc_override_gate,
    derive_gate_scales_enabled, derive_control_enabled,
)


def _wmap(conds):
    return {d: round(w, 4) for d, w in conds}


def test_flags_default_off():
    assert derive_gate_scales_enabled() is False
    assert derive_control_enabled() is False


def test_activation_scales_canonical_and_fallback():
    sc = ActivationScales.canonical()
    # opcode one-hots amplified; markers/flags unit; unknown -> default 1.0
    assert sc.scale("OP_BZ") == 5.0
    assert sc.scale("MARK_PC") == 1.0
    assert sc.scale("HAS_SE") == 1.0
    assert sc.scale("SOMETHING_UNKNOWN") == 1.0
    # a non-positive measured scale falls back (never 1/0)
    assert sc.scale("OP_BNZ") > 0.0


def test_load_activation_scales_is_positive_everywhere():
    sc = load_activation_scales()
    for dim in ("MARK_PC", "OP_BZ", "OP_BNZ", "CMP+4", "CMP+5", "HAS_SE"):
        assert sc.scale(dim) > 0.0


def test_derive_bz_gate_reproduces_hand_positives_and_threshold():
    """The hand BZ gate: MARK_PC=1, OP_BZ=0.2 (=1/5), CMP+4=1, CMP+5=1,
    IS_BYTE=-10, HAS_SE=10 ; threshold=13.5. derive_gate must reproduce the
    positive weights (= 1/activation_scale) + threshold from the datum.

    Uses the CANONICAL baked table explicitly (scale(OP_BZ)==5.0) so the
    assertion is independent of any installed calibration JSON (which measures
    OP_BZ~5.2 -> weight~0.192, functionally equivalent)."""
    sc = ActivationScales.canonical()
    op_bz_scale = sc.scale("OP_BZ")
    # feed a scale-agnostic UNIT-weight ISA spec (OP_BZ as a unit discriminator);
    # derive_gate normalizes it by 1/scale.
    isa = (("MARK_PC", 1.0), ("OP_BZ", 1.0), ("CMP+4", 1.0), ("CMP+5", 1.0),
           ("IS_BYTE", -1.0), ("HAS_SE", 10.0))
    out, thr = derive_gate(isa, hand_threshold=13.5, scales=sc)
    w = _wmap(out)
    # OP_BZ derived to 1/scale(OP_BZ) — canonical 1/5 == 0.2 (the hand value)
    assert abs(w["OP_BZ"] - 1.0 / op_bz_scale) < 1e-6
    assert abs(w["OP_BZ"] - 0.2) < 1e-6
    assert abs(w["MARK_PC"] - 1.0) < 1e-6
    assert abs(w["CMP+4"] - 1.0) < 1e-6
    assert abs(w["HAS_SE"] - 10.0) < 1e-6  # step-guard preserved
    # threshold = (n_norm - 0.5) + step_guard = 3.5 + 10 = 13.5 (the hand value)
    assert abs(thr - 13.5) < 1e-6
    # the IS_BYTE blocker vetoes past threshold (verdict-preserving) — and here
    # is STRONGER-or-equal than the hand -10 (derived_pos_sum+1 dominates).
    assert w["IS_BYTE"] <= -10.0


def test_derive_bnz_groups_reproduce_hand_thresholds():
    isa_lo = (("MARK_PC", 1.0), ("OP_BNZ", 1.0), ("CMP+4", -1.0), ("HAS_SE", 10.0))
    out, thr = derive_gate(isa_lo, hand_threshold=11.5)
    assert abs(thr - 11.5) < 1e-6              # (2-0.5) + 10
    assert abs(_wmap(out)["OP_BNZ"] - 0.2) < 1e-6

    isa_hi = (("MARK_PC", 1.0), ("OP_BNZ", 1.0), ("CMP+4", 1.0), ("CMP+5", -1.0),
              ("HAS_SE", 10.0))
    out2, thr2 = derive_gate(isa_hi, hand_threshold=12.5)
    assert abs(thr2 - 12.5) < 1e-6             # (3-0.5) + 10


def test_dead_reserved_band_is_untouched():
    """A DEAD reserved band (hand threshold > max raw firing sum — the delayed
    JMP band with CONST=-1000) must be returned UNCHANGED so the derivation can't
    resurrect it."""
    delayed = (("MARK_PC", 1.0), ("CMP+0", 1.0), ("MARK_AX", -10.0),
               ("CONST", -1000.0))
    out, thr = derive_gate(delayed, hand_threshold=5.5)
    w = _wmap(out)
    assert thr == 5.5                          # hand threshold preserved
    assert w["CONST"] == -1000.0               # deliberately-huge veto preserved
    assert w["MARK_AX"] == -10.0
    assert w["MARK_PC"] == 1.0

    # all-step JMP (OP_JMP not amplified at its block -> raw sum ~2 < thr 4.5): dead
    allstep = (("MARK_PC", 1.0), ("OP_JMP", 1.0), ("MARK_AX", -10.0))
    out2, thr2 = derive_gate(allstep, hand_threshold=4.5)
    assert thr2 == 4.5
    assert _wmap(out2)["OP_JMP"] == 1.0        # untouched (dead)


def test_live_gate_blocker_vetoes_past_threshold():
    """A LIVE gate's derived blocker must veto even when every positive fires."""
    isa = (("MARK_PC", 1.0), ("OP_BZ", 1.0), ("CMP+4", 1.0), ("CMP+5", 1.0),
           ("IS_BYTE", -1.0), ("HAS_SE", 10.0))
    out, thr = derive_gate(isa, hand_threshold=13.5)
    w = _wmap(out)
    pos_sum = sum(x for x in w.values() if x > 0)
    blocker = min(x for x in w.values() if x < 0)  # most-negative
    # a single active blocker vetoes: pos_sum + blocker < threshold
    assert pos_sum + blocker < thr


def test_class_keyed_opcode_scale_ax_vs_star():
    """GATE-ROLLOUT (task #452): an opcode one-hot reads 5.2 at ``mark==AX`` (the
    AX-marker corrector rows) but stays 1.0 under the ``"*"`` wildcard (dead at
    the L6 all-step JMP override block). The same dim, different scale per class —
    the per-block-keying that lets the L6 deadness and the L16 liveness coexist."""
    sc = ActivationScales.canonical()
    # OP_JMP: dead-band default 1.0, amplified 5.2 at the AX marker
    assert sc.scale("OP_JMP", "*") == 1.0
    assert sc.scale("OP_JMP", "mark==AX") == 5.2
    assert sc.scale("OP_JMP", "mark==SP") == 5.2
    # OP_LEA / OP_ENT similarly amplified at AX (MEASURED 5.23)
    assert sc.scale("OP_LEA", "mark==AX") == 5.2
    assert sc.scale("OP_ENT", "mark==AX") == 5.2
    # markers stay unit-scale in every class
    assert sc.scale("MARK_AX", "mark==AX") == 1.0
    assert sc.scale("MARK_SP", "mark==SP") == 1.0


def test_l6_all_step_jmp_stays_dead_under_star_class():
    """The L6 pc_mux path queries the ``"*"`` class (default), where OP_JMP==1.0,
    so the all-step JMP override band stays DEAD (untouched) — the class-keyed
    opcode amplification does NOT resurrect it."""
    allstep = (("MARK_PC", 1.0), ("OP_JMP", 1.0), ("MARK_AX", -10.0))
    out, thr = derive_gate(allstep, position_class="*", hand_threshold=4.5)
    assert thr == 4.5
    assert _wmap(out)["OP_JMP"] == 1.0        # untouched (dead at "*")


def test_l16_jmp_ax_preserve_reproduces_hand_at_ax_class():
    """GATE-ROLLOUT: the l16 jmp_ax_preserve gate — OP_JMP=0.2 (=1/5.2 at the AX
    marker), MARK_AX=1.0, threshold=1.5 — is REPRODUCED by derive_gate at the
    ``mark==AX`` class. ``preserve_blockers=True``: the negatives are
    broadcast-defeat guards, KEPT verbatim (only positives + threshold derive)."""
    hand = (("OP_JMP", 0.2), ("MARK_AX", 1.0),
            ("IS_BYTE", -10.0), ("MARK_PC", -10.0), ("MARK_SP", -10.0),
            ("MARK_BP", -10.0), ("MARK_STACK0", -10.0), ("MARK_MEM", -10.0),
            ("FETCH_LO+3", -2.0))
    out, thr = derive_gate(hand, position_class="mark==AX", hand_threshold=1.5,
                           preserve_blockers=True)
    w = _wmap(out)
    assert abs(w["OP_JMP"] - 1.0 / 5.2) < 1e-3   # ~0.192 == the hand 0.2
    assert w["MARK_AX"] == 1.0
    assert abs(thr - 1.5) < 1e-6                 # (2 - 0.5) balanced-AND midpoint
    # blockers PRESERVED verbatim (broadcast-defeat guards, not re-derived)
    assert w["IS_BYTE"] == -10.0
    assert w["MARK_PC"] == -10.0
    assert w["FETCH_LO+3"] == -2.0


def test_l6_jsr_sp_fixup_positives_derive_blockers_preserved():
    """GATE-ROLLOUT: the l6 jsr_sp_fixup gate (OP_JSR=0.2, MARK_SP=1.0,
    HAS_SE=-1.0, threshold=1.5) with ``preserve_blockers=True``: OP_JSR derives to
    1/5.2, MARK_SP stays 1.0, threshold 1.5, and the -1e6 broadcast-defeat
    blockers + the HAS_SE=-1 soft term are KEPT verbatim (re-deriving them
    regressed id550 div_step=6)."""
    hand = (("OP_JSR", 0.2), ("MARK_SP", 1.0), ("HAS_SE", -1.0),
            ("IS_BYTE", -1e6), ("MARK_PC", -1e6), ("MARK_AX", -1e6),
            ("MARK_BP", -1e6), ("MARK_STACK0", -1e6), ("MARK_MEM", -1e6))
    out, thr = derive_gate(hand, position_class="mark==SP", hand_threshold=1.5,
                           preserve_blockers=True)
    w = _wmap(out)
    assert abs(w["OP_JSR"] - 1.0 / 5.2) < 1e-3   # ~0.192 == the hand 0.2
    assert w["MARK_SP"] == 1.0
    assert abs(thr - 1.5) < 1e-6
    # HAS_SE soft term + the -1e6 broadcast-defeat guards KEPT verbatim
    assert w["HAS_SE"] == -1.0
    assert w["MARK_PC"] == -1e6
    assert w["IS_BYTE"] == -1e6
    # the fire/veto regimes still hold with the derived positive:
    op = (1.0 / 5.2) * 5.2                        # OP fires at its SP-row scale
    assert op + 1.0 >= thr                        # bootstrap (HAS_SE=0) fires
    assert op + 1.0 - 1.0 < thr                   # later (HAS_SE=1) vetoed


def test_derive_pc_override_gate_blocker_only_preserves_positives():
    """The CONTROL blocker-only path preserves positives + threshold and
    re-derives blockers to the spec-structural veto -k*max_pos*n_pos."""
    # Use a gate whose max positive weight is large (JSR MARK_PC=20) so the
    # derived blocker -k*20*n_pos is meaningfully strong (the CONTROL regime).
    hand = (("MARK_PC", 20.0), ("OPCODE_LO", 1.0), ("OPCODE_HI", 1.0),
            ("MARK_AX", -100.0))
    out, thr = derive_pc_override_gate(hand, hand_threshold=21.5)
    w = _wmap(out)
    # positives + threshold untouched (blocker-only derivation)
    assert w["MARK_PC"] == 20.0
    assert w["OPCODE_LO"] == 1.0 and w["OPCODE_HI"] == 1.0
    assert thr == 21.5
    # blocker re-derived to -safety_factor * max_pos(=20) * n_pos(=3) = -60
    assert w["MARK_AX"] == -60.0
