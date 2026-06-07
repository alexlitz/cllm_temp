"""Per-op audit harness for L16 (LEV routing + STACK0 marker materializers).

This is the L16 instance of the per-op audit pattern. For every named
rule family inside ``_layer16_lev_routing_rules`` it asserts:

  (1) no symbolic-vs-lowered drift (via ``assert_no_drift``);
  (2) every rule actually fires during the bake — the lowered W_up,
      b_up, and W_down rows are non-empty (via ``assert_fires_during_bake``);
  (3) a hand-built residual state drives the expected OUTPUT_LO / OUTPUT_HI
      lanes for the four routing families that own the L16 bake's hot
      paths:

        * B6-D's new generic ``stack0_e0_marker_from_alu_*`` family
          (32 units, ALU_{LO,HI} → OUTPUT_{LO,HI} at SP=0xffe0);
        * the established ``stack0_e8_marker_from_alu_*`` family
          (32 units, parallel sanity for SP=0xffe8);
        * the established ``stack0_f8_marker_from_alu_*`` family
          (32 units, parallel sanity for SP=0xfff8);
        * the LEV result reroute (``lev_sp_bp_plus16_{lo,hi}`` and
          ``lev_pc_temp_{lo,hi}``), which materializes SP = BP + 16 and
          the saved-PC byte at the LEV result row.

The "fires-during-bake" pass also re-asserts the per-layer unit count is
exactly 728 — a regression there means a rule family was silently added
or dropped between commits.
"""

from __future__ import annotations

import pytest

from neural_vm.unified_compiler.ir import CompilerIR
from neural_vm.unified_compiler.ops.l16_ops import (
    _layer16_lev_routing_rules,
    lower_layer16_lev_routing_ir,
)
from neural_vm.unified_compiler.primitives import Primitives
from neural_vm.vm_step import _SetDim

from tests._per_op_audit import (
    StubFFN,
    assert_fires_during_bake,
    assert_no_drift,
)


# --------------------------------------------------------------------------- #
# Rule grouping
# --------------------------------------------------------------------------- #


# All families inside ``_layer16_lev_routing_rules``. Each prefix matches a
# unique block of named rules so the per-family drift / fires checks can be
# parameterized cleanly.
L16_RULE_FAMILIES: tuple[tuple[str, str, int | None], ...] = (
    # (family_id, name_prefix, expected_count_or_None)
    # ---- LEV routing core ----
    ("lev_sp_cancel_lo", "l16_lev_sp_cancel_lo_", 16),
    ("lev_sp_cancel_hi", "l16_lev_sp_cancel_hi_", 16),
    ("lev_sp_bp_plus16_lo", "l16_lev_sp_bp_plus16_lo_", 16),
    ("lev_sp_bp_plus16_hi", "l16_lev_sp_bp_plus16_hi_", 16),
    ("lev_pc_cancel_hi", "l16_lev_pc_cancel_hi_", 16),
    ("lev_pc_temp_lo", "l16_lev_pc_temp_lo_", 16),
    ("lev_pc_temp_hi", "l16_lev_pc_temp_hi_", 16),
    ("lev_clear_output_lo10", "l16_lev_clear_output_lo10_", 3),
    ("lev_set_output_lo0", "l16_lev_set_output_lo0_", 3),
    ("lev_set_output_hi0", "l16_lev_set_output_hi0_", 3),
    # ---- STACK0 marker materializers ----
    ("stack0_e8_marker_lo", "l16_stack0_e8_marker_from_alu_lo_", 16),
    ("stack0_e8_marker_hi", "l16_stack0_e8_marker_from_alu_hi_", 16),
    # B6-D's new e0 family (32 units; 16 LO + 16 HI):
    ("stack0_e0_marker_lo", "l16_stack0_e0_marker_from_alu_lo_", 16),
    ("stack0_e0_marker_hi", "l16_stack0_e0_marker_from_alu_hi_", 16),
    ("stack0_f8_marker_lo", "l16_stack0_f8_marker_from_alu_lo_", 16),
    ("stack0_f8_marker_hi", "l16_stack0_f8_marker_from_alu_hi_", 16),
    # ---- LEV SP STACK0 inverse cancels (false-positive guards) ----
    ("stack0_cancel_lev_sp_lo", "l16_stack0_cancel_lev_sp_lo_", 16),
    ("stack0_cancel_lev_sp_hi", "l16_stack0_cancel_lev_sp_hi_", 16),
    # ---- AX carry / preserve / clear materializers ----
    ("stale_imm_ax_carry_lo", "l16_stale_imm_ax_carry_lo_", 16),
    ("stale_imm_ax_carry_hi", "l16_stale_imm_ax_carry_hi_", 16),
    ("store_ax_carry_lo", "l16_store_ax_carry_lo_", 16),
    ("store_ax_carry_hi", "l16_store_ax_carry_hi_", 16),
    ("jmp_ax_preserve_lo", "l16_jmp_ax_preserve_lo_", 16),
    # ---- BP marker passthrough ----
    ("bp_marker_passthrough_lo", "l16_bp_marker_passthrough_lo_", 16),
    ("bp_marker_passthrough_hi", "l16_bp_marker_passthrough_hi_", 16),
    # ---- PSH SP no-borrow restores ----
    ("psh_sp_no_borrow_hi", "l16_psh_sp_no_borrow_hi_", 16),
    # ---- PSH MEM addr0 restore (1..15 plus one ":+8" lo) ----
    ("psh_mem_addr0_restore_hi", "l16_psh_mem_addr0_restore_hi_", 15),
    # ---- ENT frame SP byte0 (variable size) ----
    ("ent_frame_sp_byte0_hi_lo0", "l16_ent_frame_sp_byte0_hi_lo0_", 16),
    ("ent_frame_sp_byte0_hi_lo8", "l16_ent_frame_sp_byte0_hi_lo8_", 16),
    # ---- Top store STACK0 restore (1..15) ----
    ("top_store_stack0_restore_lo", "l16_top_store_stack0_restore_lo_", 15),
    # NOTE: ``l16_stack0_e8_output_authoritative_*`` (254 rules) is
    # exercised via ``assert_fires_during_bake`` on the entire layer
    # below; per-family drift coverage there would balloon the test.
)


# Singleton named rules (one rule per name; no enumeration).
L16_SINGLETON_RULES: tuple[str, ...] = (
    "l16_lev_pc_top_return_0a",
    "l16_jsr_mem_addr0_f8",
    "l16_jsr_mem_addr0_e0_from_l14_evidence",
    "l16_jsr_initial_stack0_marker_0a",
    "l16_stack0_e0_marker_e8_from_alu_exact",
    "l16_bp_frame_byte1_ff",
    "l16_bp_after_ent_byte2_zero",
    "l16_ent_initial_stack0_byte2_01",
    "l16_ent_nested_sp_byte0_d8",
    "l16_ent_nested_bp_byte0_d8",
    "l16_ent_nested_stack0_saved_bp_byte0_f0",
    "l16_ent_stack0_saved_bp_byte1_ff",
    "l16_ent_initial_stack0_saved_bp_byte1_00",
    "l16_lea_local_ax_byte1_ff_lo",
    "l16_lea_local_ax_byte1_ff_hi",
    "l16_lea_local_ax_byte0_hi_e",
    "l16_jsr_return_addr_byte1_01_from_low_22",
    "l16_stack0_byte1_zero_after_unit_low_byte",
    "l16_psh_mem_addr0_restore_lo_8",
    "l16_psh_mem_addr0_force_d8_from_l14_evidence",
    "l16_psh_mem_addr0_e0_from_addr_b0",
    "l16_psh_mem_addr0_e0_from_sp_no_addr_src",
)


# --------------------------------------------------------------------------- #
# Module-level fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def l16_rules():
    """All L16 LEV-routing FFN rules at the unit-test scale (S=1.0).

    S=1.0 makes the lowered SwiGLU width = symbolic write magnitude so
    ``assert_no_drift``'s lowered-vs-symbolic comparison normalizes
    cleanly via the `_SILU_ONE_INPUT` margin trick that
    ``_synthetic_ffn_state`` uses internally.
    """
    return tuple(_layer16_lev_routing_rules(1.0))


@pytest.fixture(scope="module")
def l16_rules_by_name(l16_rules):
    return {rule.name: rule for rule in l16_rules}


@pytest.fixture(scope="module")
def l16_dim_positions(l16_rules):
    """Dim positions for every dim name referenced by any L16 rule."""

    names = Primitives.ffn_rule_dim_names(l16_rules)
    return Primitives.dim_positions_from_bd(_SetDim, names)


# --------------------------------------------------------------------------- #
# (1) Drift checks per rule family
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "family_id, prefix, expected_count",
    L16_RULE_FAMILIES,
    ids=[fam[0] for fam in L16_RULE_FAMILIES],
)
def test_l16_family_no_drift(
    family_id, prefix, expected_count, l16_rules, l16_dim_positions,
):
    """Lowered SwiGLU output must agree with symbolic FFN forward."""

    family = [
        rule for rule in l16_rules
        if rule.name and rule.name.startswith(prefix)
    ]
    assert family, f"no rules found for family {family_id!r} (prefix {prefix!r})"
    if expected_count is not None:
        assert len(family) == expected_count, (
            f"family {family_id!r} expected {expected_count} rules, "
            f"got {len(family)}"
        )
    assert_no_drift(
        family,
        dim_positions=l16_dim_positions,
        msg=f"family={family_id} prefix={prefix}",
    )


@pytest.mark.parametrize("rule_name", L16_SINGLETON_RULES)
def test_l16_singleton_rule_no_drift(rule_name, l16_rules_by_name, l16_dim_positions):
    """Drift check for L16 singleton (non-enumerated) named rules."""

    rule = l16_rules_by_name.get(rule_name)
    assert rule is not None, f"singleton rule {rule_name!r} missing from L16"
    assert_no_drift(rule, dim_positions=l16_dim_positions, msg=rule_name)


# --------------------------------------------------------------------------- #
# (2) Fires-during-bake
# --------------------------------------------------------------------------- #


def test_l16_full_layer_fires_during_bake(l16_rules):
    """Bake every L16 rule into a stub FFN and verify every unit fires.

    Catches silent regressions where a rule loses its W_up / W_down rows
    (a rule that bakes to all zeros is invisible to the lowered model).
    Also pins the per-layer unit count at 728 — matches the
    ``ffn_units_used`` annotation on ``make_layer16_lev_routing_op``.
    """

    assert len(l16_rules) == 728
    stub = assert_fires_during_bake(l16_rules)
    # Sanity: total non-zero rows in W_up matches rule count.
    nonzero_units = (stub.W_up.abs().sum(dim=1) > 0).sum().item()
    assert nonzero_units >= 728, (
        f"some L16 units baked with empty W_up: only {nonzero_units} of 728"
    )


@pytest.mark.parametrize(
    "family_id, prefix, expected_count",
    L16_RULE_FAMILIES,
    ids=[fam[0] for fam in L16_RULE_FAMILIES],
)
def test_l16_family_fires_during_bake(family_id, prefix, expected_count, l16_rules):
    """Per-family fires check — narrows a baking regression to one family."""

    family = [
        rule for rule in l16_rules
        if rule.name and rule.name.startswith(prefix)
    ]
    assert family, f"no rules for family {family_id!r}"
    assert_fires_during_bake(family)


def test_l16_bake_matches_legacy_unit_count():
    """The full L16 bake must end at unit 728.

    Mirrors ``test_layer16_lev_routing_ir_matches_legacy_helper`` but
    only the unit count — it's a quick sanity check usable independently
    of the legacy ``_set_layer16_lev_routing`` helper.
    """

    stub = StubFFN(hidden_dim=1024)
    end = lower_layer16_lev_routing_ir(stub, 100.0, _SetDim)
    assert end == 728


# --------------------------------------------------------------------------- #
# Symbolic-forward helpers
# --------------------------------------------------------------------------- #


def _ir_from_named_rules(rules_by_name, names):
    """Build a tiny CompilerIR containing exactly the named rules."""

    ir = CompilerIR()
    for name in names:
        ir.layer(0).ffn.append(rules_by_name[name])
    return ir


def _ir_from_family(rules_by_name, prefix, count=16):
    """Convenience: build an IR holding ``prefix0..prefix{count-1}``."""

    return _ir_from_named_rules(
        rules_by_name, (f"{prefix}{k}" for k in range(count))
    )


# --------------------------------------------------------------------------- #
# (3) Symbolic forward — B6-D's new e0 marker family
# --------------------------------------------------------------------------- #
#
# Layout: ``l16_stack0_e0_marker_from_alu_{lo,hi}_{0..15}`` mirror the
# existing e8/f8 families but key on the SP=0xffe0 address signature
# (``ADDR_B0_LO+0`` + ``ADDR_B0_HI+14``) and explicitly suppress the
# 0xffe8 lookalike via ``ADDR_B0_LO+8`` = -2.0. Each lane ``k`` reads
# ``ALU_LO+k`` (gate) and writes ``OUTPUT_LO+k`` at strength 50.0 / S.


def _e0_marker_firing_state(*, alu_lo_lane=None, alu_hi_lane=None, **overrides):
    """Residual state for the e0 STACK0 marker materializers.

    Sets up the LEV-step SP=0xffe0 address signature and optionally
    primes one ALU lane so the corresponding OUTPUT lane lights up
    after the rule fires.

    Removal-2 take 2 (2026-06-07): the e0/e8/f8 marker families are now
    LEV-gated (OP_LEV flipped from -10 blocker to +1 positive predicate
    so they fire only on real LEV-step STACK0 marker positions; binop
    cascades and fn-call IMM positions are no longer false-fires).
    """

    state = {
        "OP_LEV": 10.0,  # L6-amplified OP_LEV value at the LEV-step STACK0 marker
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_LO+8": 0.0,  # explicitly NOT the e8 lookalike
        "ADDR_B0_HI+14": 1.0,
        "ADDR_B0_HI+15": 0.0,
    }
    if alu_lo_lane is not None:
        state[f"ALU_LO+{alu_lo_lane}"] = 1.0
    if alu_hi_lane is not None:
        state[f"ALU_HI+{alu_hi_lane}"] = 1.0
    state.update(overrides)
    return state


def test_b6d_e0_marker_lo_lanes_route_to_output_lo(l16_rules_by_name):
    """ALU_LO[k] should drive OUTPUT_LO[k] under the SP=0xffe0 marker."""

    ir = _ir_from_family(l16_rules_by_name, "l16_stack0_e0_marker_from_alu_lo_")
    for lane in (0, 7, 11, 15):
        out = ir.symbolic_ffn(_e0_marker_firing_state(alu_lo_lane=lane))
        assert out[f"OUTPUT_LO+{lane}"] > 0.0, (
            f"e0 marker ALU_LO+{lane} did not light OUTPUT_LO+{lane}: out={out}"
        )
        for other in range(16):
            if other == lane:
                continue
            assert out.get(f"OUTPUT_LO+{other}", 0.0) == 0.0, (
                f"unexpected OUTPUT_LO+{other} fire for ALU_LO+{lane}"
            )


def test_b6d_e0_marker_hi_lanes_route_to_output_hi(l16_rules_by_name):
    """ALU_HI[k] should drive OUTPUT_HI[k] under the SP=0xffe0 marker."""

    ir = _ir_from_family(l16_rules_by_name, "l16_stack0_e0_marker_from_alu_hi_")
    for lane in (0, 3, 9, 14):
        out = ir.symbolic_ffn(_e0_marker_firing_state(alu_hi_lane=lane))
        assert out[f"OUTPUT_HI_THIS_STEP+{lane}"] > 0.0, (
            f"e0 marker ALU_HI+{lane} did not light OUTPUT_HI_THIS_STEP+{lane}"
        )
        for other in range(16):
            if other == lane:
                continue
            assert out.get(f"OUTPUT_HI_THIS_STEP+{other}", 0.0) == 0.0


def test_b6d_e0_marker_suppresses_e8_lookalike(l16_rules_by_name):
    """``ADDR_B0_LO+8`` carries the SP=0xffe8 signature — that row must NOT fire.

    With ADDR_B0_LO+8 active, the e0 condition pulls the score below
    threshold (the -2.0 weight cancels the +10.0 ADDR_B0_LO+0 boost).
    Mirrors the original commit's "exclude the e8 lookalike" guarantee.
    """

    ir = _ir_from_named_rules(
        l16_rules_by_name, ("l16_stack0_e0_marker_from_alu_lo_5",),
    )
    e8_lookalike = _e0_marker_firing_state(
        alu_lo_lane=5,
        **{"ADDR_B0_LO+8": 2.0},
    )
    out = ir.symbolic_ffn(e8_lookalike)
    assert out.get("OUTPUT_LO+5", 0.0) == 0.0, (
        f"e0 rule fired on e8 lookalike address: out={out}"
    )


def test_b6d_e0_marker_suppresses_store_rows(l16_rules_by_name):
    """``MEM_STORE`` active marks the current SI/SC store row.

    The store materializers (L14/L15) own that path; the e0 marker
    must stay inert so it doesn't double-write OUTPUT.
    """

    ir = _ir_from_named_rules(
        l16_rules_by_name, ("l16_stack0_e0_marker_from_alu_lo_2",),
    )
    out = ir.symbolic_ffn(_e0_marker_firing_state(alu_lo_lane=2, MEM_STORE=1.0))
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0, (
        f"e0 rule fired on store row: out={out}"
    )


def test_b6d_e0_marker_suppresses_function_boundaries(l16_rules_by_name):
    """OP_JSR / OP_ENT rows must stay inert for the e0 marker.

    Removal-2 take 2 (2026-06-07): OP_LEV is no longer a blocker but a
    positive predicate (the rule's intended firing position IS the
    LEV-step STACK0 marker). JSR/ENT exclusions remain.
    """

    ir = _ir_from_named_rules(
        l16_rules_by_name, ("l16_stack0_e0_marker_from_alu_lo_9",),
    )
    for blocker in ("OP_JSR", "OP_ENT"):
        state = _e0_marker_firing_state(alu_lo_lane=9, **{blocker: 5.0})
        out = ir.symbolic_ffn(state)
        assert out.get("OUTPUT_LO+9", 0.0) == 0.0, (
            f"e0 rule fired with {blocker} active: out={out}"
        )


# --------------------------------------------------------------------------- #
# (4) Symbolic forward — existing e8 and f8 families (parallel sanity)
# --------------------------------------------------------------------------- #


def test_e8_marker_lo_lanes_route_to_output_lo(l16_rules_by_name):
    """Parallel sanity for the established e8 marker family.

    Same ALU → OUTPUT mapping as the e0 family but at SP=0xffe8. Now
    LEV-gated (Removal-2 take 2): OP_LEV at L6-amplified value 10 lifts
    the rule above its 15.5 threshold; IMM/PSH steps stay inert.
    """

    ir = _ir_from_family(l16_rules_by_name, "l16_stack0_e8_marker_from_alu_lo_")
    base = {
        "OP_LEV": 10.0,
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+14": 1.0,
    }
    for lane in (0, 4, 9, 13):
        out = ir.symbolic_ffn(dict(base, **{f"ALU_LO+{lane}": 1.0}))
        assert out[f"OUTPUT_LO+{lane}"] > 0.0, (
            f"e8 marker ALU_LO+{lane} -> OUTPUT_LO+{lane} failed: {out}"
        )


def test_e8_marker_suppresses_e0_lookalike(l16_rules_by_name):
    """e8 family's ADDR_B0_LO+8 = +10.0 means SP=0xffe0 (LO+0) stays inert."""

    ir = _ir_from_named_rules(
        l16_rules_by_name, ("l16_stack0_e8_marker_from_alu_lo_6",),
    )
    e0_lookalike = {
        "OP_LEV": 10.0,
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+0": 1.0,  # 0xffe0 byte 0 low nibble
        "ADDR_B0_HI+14": 1.0,
        "ALU_LO+6": 1.0,
    }
    out = ir.symbolic_ffn(e0_lookalike)
    assert out.get("OUTPUT_LO+6", 0.0) == 0.0


def test_e8_marker_non_lev_step_inert(l16_rules_by_name):
    """Removal-2 take 2: IMM-after-PSH at SP=0xffe8 (the old fn-call
    use-case the rule's pre-fix comment described) must NOT fire now
    that the family is LEV-gated. The L15 nibble_copy materializer
    handles those IMM positions instead.
    """

    ir = _ir_from_named_rules(
        l16_rules_by_name, ("l16_stack0_e8_marker_from_alu_lo_2",),
    )
    imm_at_e8 = {
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "ALU_LO+2": 1.0,
    }
    out = ir.symbolic_ffn(imm_at_e8)
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


def test_f8_marker_lo_lanes_route_to_output_lo(l16_rules_by_name):
    """Parallel sanity for the established f8 marker family (SP=0xfff8)."""

    ir = _ir_from_family(l16_rules_by_name, "l16_stack0_f8_marker_from_alu_lo_")
    base = {
        "OP_LEV": 10.0,
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
    }
    for lane in (0, 5, 11):
        out = ir.symbolic_ffn(dict(base, **{f"ALU_LO+{lane}": 1.0}))
        assert out[f"OUTPUT_LO+{lane}"] > 0.0


def test_f8_marker_hi_lanes_route_to_output_hi(l16_rules_by_name):
    ir = _ir_from_family(l16_rules_by_name, "l16_stack0_f8_marker_from_alu_hi_")
    base = {
        "OP_LEV": 10.0,
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
    }
    for lane in (0, 2, 14):
        out = ir.symbolic_ffn(dict(base, **{f"ALU_HI+{lane}": 1.0}))
        assert out[f"OUTPUT_HI_THIS_STEP+{lane}"] > 0.0


def test_f8_marker_non_lev_step_inert(l16_rules_by_name):
    """Removal-2 take 2: IMM-before-binop at SP=0xfff8 (the binop cascade
    failure shape this fix targets) must NOT fire. This is the false-fire
    the OP_LEV gate was added to suppress.
    """

    ir = _ir_from_named_rules(
        l16_rules_by_name, ("l16_stack0_f8_marker_from_alu_lo_2",),
    )
    imm_at_f8 = {
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "ALU_LO+2": 1.0,
    }
    out = ir.symbolic_ffn(imm_at_f8)
    assert out.get("OUTPUT_LO+2", 0.0) == 0.0


# --------------------------------------------------------------------------- #
# (5) Symbolic forward — LEV result reroute (SP = BP + 16, PC = TEMP)
# --------------------------------------------------------------------------- #
#
# Layout: the LEV reroute is split into three concurrent rule families:
#
#   * ``lev_sp_cancel_*``   — cancel old SP byte (gate=OUTPUT, weight=-1)
#   * ``lev_sp_bp_plus16_*``— write the new SP byte = BP byte + 16
#   * ``lev_pc_temp_*``     — copy TEMP[k] (or TEMP[16+k] for HI) to OUTPUT
#
# Symbolic verification of each family by itself, since combining them in
# one symbolic forward intertwines the cancel and write contributions in
# ways that make per-lane asserts noisier.


_LEV_SP_BASE = {"OP_LEV": 1.0, "MARK_SP": 1.0, "HAS_SE": 2.0}
_LEV_PC_BASE = {"OP_LEV": 1.0, "MARK_PC": 1.0, "CONST": 1.0}


def test_lev_result_reroute_sp_value_writes(l16_rules_by_name):
    """LEV SP reroute: ``ADDR_B0_LO[k]`` should drive ``OUTPUT_LO[k]``.

    Threshold (40.0) is dominated by HAS_SE × 30 (first-step gate);
    HAS_SE=2 lifts the score above 40 once ADDR_B0_LO+lane carries
    weight. The negative MARK_* blockers stay at 0 so they don't drag
    the score down.
    """

    ir = _ir_from_family(l16_rules_by_name, "l16_lev_sp_bp_plus16_lo_")
    for lane in (0, 8, 12):
        out = ir.symbolic_ffn(dict(_LEV_SP_BASE, **{f"ADDR_B0_LO+{lane}": 20.0}))
        assert out.get(f"OUTPUT_LO+{lane}", 0.0) > 0.0, (
            f"LEV SP+BP+16 ADDR_B0_LO+{lane} -> OUTPUT_LO+{lane} failed: out={out}"
        )


def test_lev_result_reroute_sp_value_hi_lane_shift(l16_rules_by_name):
    """LEV SP reroute HI lane writes to ``OUTPUT_HI[(k+1) % 16]``.

    SP = BP + 16 means the byte-0 high nibble carries +1; the rule's
    write target reflects that shift.
    """

    ir = _ir_from_family(l16_rules_by_name, "l16_lev_sp_bp_plus16_hi_")
    for lane in (0, 5, 14, 15):
        out = ir.symbolic_ffn(dict(_LEV_SP_BASE, **{f"ADDR_B0_HI+{lane}": 20.0}))
        expected_dst = (lane + 1) % 16
        assert out.get(f"OUTPUT_HI_THIS_STEP+{expected_dst}", 0.0) > 0.0, (
            f"LEV SP+BP+16 ADDR_B0_HI+{lane} did not light "
            f"OUTPUT_HI_THIS_STEP+{expected_dst}: out={out}"
        )


def test_lev_result_reroute_pc_temp_copy_lo(l16_rules_by_name):
    """LEV PC reroute: ``TEMP+k`` should copy to ``OUTPUT_LO+k`` at the PC row.

    The rule's gate is ``CONST``, so the firing state must drive CONST
    non-zero for the write to land in OUTPUT_LO.
    """

    ir = _ir_from_family(l16_rules_by_name, "l16_lev_pc_temp_lo_")
    for lane in (0, 2, 10):
        out = ir.symbolic_ffn(dict(_LEV_PC_BASE, **{f"TEMP+{lane}": 2.0}))
        assert out.get(f"OUTPUT_LO+{lane}", 0.0) > 0.0, (
            f"LEV PC TEMP+{lane} -> OUTPUT_LO+{lane} failed: {out}"
        )


def test_lev_result_reroute_pc_temp_copy_hi(l16_rules_by_name):
    """LEV PC reroute HI lane: ``TEMP+16+k`` -> ``OUTPUT_HI+k``."""

    ir = _ir_from_family(l16_rules_by_name, "l16_lev_pc_temp_hi_")
    for lane in (0, 5, 13):
        out = ir.symbolic_ffn(dict(_LEV_PC_BASE, **{f"TEMP+{16 + lane}": 2.0}))
        assert out.get(f"OUTPUT_HI_THIS_STEP+{lane}", 0.0) > 0.0, (
            f"LEV PC TEMP+{16 + lane} -> OUTPUT_HI_THIS_STEP+{lane} failed: {out}"
        )


def test_lev_sp_cancel_inverts_existing_output(l16_rules_by_name):
    """LEV SP cancel: ``OUTPUT_LO[k]`` already set should be reduced.

    The cancel rule's gate references OUTPUT_LO[k] with gate_weight=-1,
    so when that lane is non-zero the rule's write delivers a negative
    contribution back into the same lane.
    """

    ir = _ir_from_named_rules(l16_rules_by_name, ("l16_lev_sp_cancel_lo_4",))
    cancel_state = dict(_LEV_SP_BASE, **{"OUTPUT_LO+4": 5.0})
    out = ir.symbolic_ffn(cancel_state)
    assert out["OUTPUT_LO+4"] < cancel_state["OUTPUT_LO+4"], (
        f"LEV SP cancel did not subtract from OUTPUT_LO+4: {out}"
    )
