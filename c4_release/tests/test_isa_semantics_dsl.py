"""Byte-identity tests for ``isa_semantics_dsl`` (the ISA-semantics DSL).

Mirrors ``tests/test_building_blocks_dsl.py``. Per generator the contract is:

  (i)   ``compare_symbolic_to_lowered_{ffn,attn,embedding}`` asserts the
        GENERATED rules / head / columns lower byte-identically;
  (ii)  the WHOLE-MODEL ``state_dict`` SHA256 == the HEAD golden, parametrized
        over the feature flag via ``monkeypatch.setenv`` (the decisive gate —
        it proves the generator reproduces the hand-built ops bit-for-bit);
  (iii) the flag-OFF path emits ZERO rules / columns (byte-identical off);
  (iv)  smoke / 1096 >= the recorded baseline (read from env, not hardcoded —
        those live in the GPU gate, not this CPU unit suite).

The whole-model param-hash tests build ``compile_full_vm_dynamic(
disk_cache=False)`` on CPU and are SLOW (~30-90 s each); they are marked
``slow`` + guarded by ``C4_ISA_HASH_TEST=1`` so the fast unit suite (the
generator-shape + ``compare_symbolic_to_lowered_*`` checks) runs in seconds.
The golden hashes are read from env (``C4_ISA_GOLDEN_*``) with the
session-recorded values as the documented default — NEVER trusted as a
hardcoded regression target without a same-config re-baseline.
"""

from __future__ import annotations

import hashlib
import os

import pytest
import torch

from c4_release.neural_vm.base_layers import PureFFN
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    multi_way_and_rule,
)
from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    compare_symbolic_to_lowered_attn,
    compare_symbolic_to_lowered_embedding,
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.isa_semantics_dsl import (
    ConsumerLookaheadGateBundle,
    ConsumerLookaheadGateSpec,
    CrossStepCarryBundle,
    CrossStepCarrySpec,
    FullWidthByteEmissionBundle,
    FullWidthByteEmissionSpec,
    consumer_lookahead_gate,
    cross_step_carry,
    full_width_byte_emission,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives


# ---------------------------------------------------------------------------
# Golden hashes — read from env (session-recorded defaults). The HEAD param
# hash with the BP carry flag on / off (full-width emission OFF, the default).
# ---------------------------------------------------------------------------

_GOLDEN_BP_ON = os.environ.get(
    "C4_ISA_GOLDEN_BP_ON",
    "7474f26955e79619b386a9e301339a8dba0b7ba17022251e3572a42d64b3d989",
)
_GOLDEN_BP_OFF = os.environ.get(
    "C4_ISA_GOLDEN_BP_OFF",
    "d503c7ad43f7fca15d2df0b25c86e796541adfb01bf2fdf165b581600c473fd6",
)

# #221 consumer-lookahead-gate migration goldens (session-recorded, CPU,
# PYTHONHASHSEED=0, disk_cache=False). Flag-ON is the DEFAULT build, so it
# equals ``_GOLDEN_BP_ON``; flag-OFF omits the six feature bands (smaller
# d_model) and so has a DISTINCT golden. Both prove the generator re-expression
# reproduces the hand-built #221 ops bit-for-bit.
_GOLDEN_ARITH_GATE_ON = os.environ.get(
    "C4_ISA_GOLDEN_ARITH_GATE_ON",
    "7474f26955e79619b386a9e301339a8dba0b7ba17022251e3572a42d64b3d989",
)
_GOLDEN_ARITH_GATE_OFF = os.environ.get(
    "C4_ISA_GOLDEN_ARITH_GATE_OFF",
    "9773f06e00e1c20aecce523098fbeb63560377532ffff90f9e5e1f488e63d7e2",
)

_HASH_TESTS_ENABLED = os.environ.get("C4_ISA_HASH_TEST", "0") != "0"
_requires_hash = pytest.mark.skipif(
    not _HASH_TESTS_ENABLED,
    reason="whole-model param-hash tests are slow; set C4_ISA_HASH_TEST=1",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lowered_ffn_at_S(rules, dim_positions, *, S: float = 100.0) -> PureFFN:
    """Build + lower a fresh ``PureFFN`` for a rule list at scale ``S``."""
    n_rules = len(rules)
    d_model = max(dim_positions.values()) + 300
    ffn = PureFFN(dim=d_model, hidden_dim=max(n_rules, 1))
    end = Primitives.lower_ffn_rules(ffn, rules, dim_positions, start_unit=0, S=S)
    assert end == n_rules
    return ffn


def _forward_band(ffn, dim_positions, state, band, width):
    """Forward + return the per-cell delta of ``band`` (width cells)."""
    d_model = ffn.W_up.shape[1]
    x = torch.zeros(1, 1, d_model)
    for key, value in state.items():
        if "+" in key:
            base, off = key.rsplit("+", 1)
            x[0, 0, dim_positions[base] + int(off)] = float(value)
        else:
            x[0, 0, dim_positions[key]] = float(value)
    with torch.no_grad():
        y = ffn(x)
    base = dim_positions[band]
    return [float(y[0, 0, base + j].item()) for j in range(width)]


def _model_state_hash() -> str:
    """SHA256 of the full model ``state_dict`` (CPU, disk_cache=False)."""
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    model, _layout = compile_full_vm_dynamic(disk_cache=False)
    h = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd.keys()):
        t = sd[key]
        if not isinstance(t, torch.Tensor):
            continue
        h.update(key.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.detach().to(torch.float64).cpu().numpy().tobytes())
    return h.hexdigest()


def _built_dim_positions():
    """The BUILT layout's dim_positions (so band dims resolve like production).

    Slow (full compile); only the hash-gated tests need it. Built fresh so the
    flag state at call time is reflected.
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    _model, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = getattr(layout, "dim_positions", layout)
    return dict(dp)


# Test bands MUST never be collected into the production model build (their
# import-time register_residual_band side effect would widen d_model and break
# the whole-model param-hash). Gate every test band OFF: a flag that always
# returns False => collect_registered_residual_bands skips it.
def _never() -> bool:
    return False


# A minimal BP-like spec for the generator-shape tests (no full build needed).
def _bp_like_spec(**overrides) -> CrossStepCarrySpec:
    base = dict(
        name="bp_like",
        band_name="BP_LIKE_PREV",
        band_width=32,
        band_flag=_never,
        carry_head_alibi_slope=0.5,
        carry_byte_count=4,
        match_q_band="MEM_VAL_B",
        match_k_band="BYTE_INDEX_",
        match_weight=40.0,
        k_prefer=((4, "OP_JSR", 6.0),),
        k_reject=(
            (5, "OP_ENT", 8.0),
            (6, "STACK0_BYTE0", 8.0),
            (6, "STACK0_BYTE1", 8.0),
            (6, "STACK0_BYTE2", 8.0),
            (6, "STACK0_BYTE3", 8.0),
        ),
        value_src_lo="CLEAN_EMBED_LO",
        value_src_hi="CLEAN_EMBED_HI",
        value_o_write_scale=200.0,
        value_v_slot_base=10,
        dump_emit_lo="OUTPUT_LO",
        dump_emit_hi="OUTPUT_HI",
        dump_write_scale=200000.0,
        dump_gate_conditions=(("OP_ENT", 1.0),),
        dump_per_byte_marker="MEM_VAL_B",
        dump_per_byte_marker_weight=8.0,
        dump_marker_blockers=(
            ("MARK_PC", -1_000.0),
            ("MARK_AX", -1_000.0),
        ),
        dump_opent_floor=6.0,
    )
    base.update(overrides)
    return CrossStepCarrySpec(**base)


def _bp_like_dim_positions() -> dict:
    """Synthetic dim_positions covering every dim a BP-like carry touches."""
    dp = {
        "CONST": 0,
        "OP_JSR": 1,
        "OP_ENT": 2,
        "STACK0_BYTE0": 3,
        "STACK0_BYTE1": 4,
        "STACK0_BYTE2": 5,
        "STACK0_BYTE3": 6,
        "BP_LIKE_PREV": 200,
        "CLEAN_EMBED_LO": 250,
        "CLEAN_EMBED_HI": 270,
        "OUTPUT_LO": 300,
        "OUTPUT_HI": 320,
        "MARK_PC": 7,
        "MARK_AX": 8,
    }
    for k in range(4):
        dp[f"MEM_VAL_B{k}"] = 10 + k
        dp[f"BYTE_INDEX_{k}"] = 20 + k
    return dp


# ===========================================================================
# cross_step_carry — generator shape + byte-identity of the generated parts
# ===========================================================================


def test_cross_step_carry_returns_bundle_with_pure_builders():
    spec = _bp_like_spec()
    bundle = cross_step_carry(spec)
    assert isinstance(bundle, CrossStepCarryBundle)
    assert bundle.spec is spec
    # The carry head WRITES only the _PREV band; READS the cross-step value srcs.
    assert bundle.carry_head_writes == {"BP_LIKE_PREV"}
    assert "CLEAN_EMBED_LO.*.-1" in bundle.carry_head_reads
    assert "CLEAN_EMBED_HI.*.-1" in bundle.carry_head_reads
    # The dump WRITES the emit bands + the _PREV band; the gate is in the FFN.
    assert bundle.dump_writes == {"OUTPUT_LO", "OUTPUT_HI", "BP_LIKE_PREV"}


def test_cross_step_carry_has_no_head_gate_field():
    """The head-unconditional + gate-in-FFN split is enforced by the API shape:
    there is NO head-gate field on the spec (gating ONLY via dump conditions)."""
    fields = set(CrossStepCarrySpec.__dataclass_fields__)
    for forbidden in ("head_gate", "carry_head_gate", "head_conditions"):
        assert forbidden not in fields


def test_cross_step_carry_dump_off_is_empty():
    bundle = cross_step_carry(_bp_like_spec())
    assert bundle.dump_rules_builder(False) == ()
    assert len(bundle.dump_rules_builder(True)) > 0


def test_cross_step_carry_dump_rule_count():
    """4 bytes x 32 band cells = 128 dump rules (16 lo + 16 hi per byte)."""
    bundle = cross_step_carry(_bp_like_spec())
    rules = bundle.dump_rules_builder(True)
    assert len(rules) == 4 * 32


def test_cross_step_carry_head_spec_byte_identical():
    """The generated carry head lowers byte-identically (self-consistent)."""
    spec = _bp_like_spec()
    bundle = cross_step_carry(spec)
    dp = _bp_like_dim_positions()
    head = bundle.carry_head_spec_builder(dp, head_idx=7)
    report = compare_symbolic_to_lowered_attn(
        head, head_dim=64, num_heads=8, dim=max(dp.values()) + 32,
    )
    assert report.ok, report


def test_cross_step_carry_dump_rules_byte_identical():
    """Each generated dump cell lowers byte-identically (direct forward)."""
    spec = _bp_like_spec()
    bundle = cross_step_carry(spec)
    rules = list(bundle.dump_rules_builder(True))
    dp = _bp_like_dim_positions()
    ffn = _lowered_ffn_at_S(rules, dp, S=100.0)
    # Fire byte-0 (MEM_VAL_B0) on an ENT-store row with BP_LIKE_PREV cell 5 set:
    # only OUTPUT_LO+5 should carry the re-supply (the lo half).
    state = {
        "OP_ENT": 8.0, "MEM_VAL_B0": 1.0,
        "BP_LIKE_PREV+5": 1.0,
    }
    out_lo = _forward_band(ffn, dp, state, "OUTPUT_LO", 16)
    assert out_lo[5] > 1.0, out_lo
    for j in range(16):
        if j != 5:
            assert abs(out_lo[j]) < 1e-3, (j, out_lo[j])


def test_cross_step_carry_dump_gate_blocks_marker_rows():
    """The MARK_* blockers darken the dump on a register-marker row."""
    spec = _bp_like_spec()
    bundle = cross_step_carry(spec)
    rules = list(bundle.dump_rules_builder(True))
    dp = _bp_like_dim_positions()
    ffn = _lowered_ffn_at_S(rules, dp, S=100.0)
    # ENT row + MEM_VAL_B0 + carried cell, but ALSO MARK_PC (a marker row).
    state = {
        "OP_ENT": 8.0, "MEM_VAL_B0": 1.0,
        "BP_LIKE_PREV+5": 1.0, "MARK_PC": 1.0,
    }
    out_lo = _forward_band(ffn, dp, state, "OUTPUT_LO", 16)
    assert all(abs(v) < 1e-2 for v in out_lo), out_lo


def test_cross_step_carry_odd_band_width_rejected():
    with pytest.raises(ValueError, match="band_width must be even"):
        _bp_like_spec(band_width=31)


def test_cross_step_carry_zero_byte_count_rejected():
    with pytest.raises(ValueError, match="carry_byte_count"):
        _bp_like_spec(carry_byte_count=0)


# ===========================================================================
# full_width_byte_emission — generator shape + byte-identity
# ===========================================================================


def _fwbe_spec(**overrides) -> FullWidthByteEmissionSpec:
    base = dict(
        name="fwbe_t",
        band_name="FWBE_T_WIDE",
        bits=8,
        head_scale=5.0,
        lo_value=16,
        emission_flag=_never,  # never collected into the production build
    )
    base.update(overrides)
    return FullWidthByteEmissionSpec(**base)


def test_full_width_byte_emission_returns_bundle():
    spec = _fwbe_spec()
    bundle = full_width_byte_emission(spec)
    assert isinstance(bundle, FullWidthByteEmissionBundle)
    assert bundle.spec is spec
    assert spec.band_width == 256


def test_full_width_byte_emission_columns_off_is_empty():
    bundle = full_width_byte_emission(_fwbe_spec())
    assert bundle.head_columns_builder(False, 256) == ()


def test_full_width_byte_emission_column_count_and_range():
    """lo_value=16 .. 255 => 240 columns; each a distinct band cell."""
    bundle = full_width_byte_emission(_fwbe_spec())
    cols = bundle.head_columns_builder(True, 256)
    assert len(cols) == 240
    tokens = sorted(c.token_ids[0] for c in cols)
    assert tokens[0] == 16 and tokens[-1] == 255
    # No mod-16 alias: token v writes its OWN band cell v.
    for c in cols:
        v = c.token_ids[0]
        w = c.writes[0]
        assert w.dim.name == "FWBE_T_WIDE"
        assert w.dim.offset == v
        assert w.weight == 5.0


def test_full_width_byte_emission_respects_vocab_cap():
    """Columns are capped at vocab_size - 1."""
    bundle = full_width_byte_emission(_fwbe_spec())
    cols = bundle.head_columns_builder(True, 100)  # vocab 100
    assert max(c.token_ids[0] for c in cols) == 99


def test_full_width_byte_emission_columns_byte_identical():
    """The generated LM-head columns lower byte-identically."""
    bundle = full_width_byte_emission(_fwbe_spec())
    cols = bundle.head_columns_builder(True, 256)
    ir = CompilerIR()
    ir.embeddings.extend(cols)
    dp = {"FWBE_T_WIDE": 100}
    report = compare_symbolic_to_lowered_embedding(
        ir, dp, vocab_size=256, d_model=400,
    )
    assert report.ok, report.issues


def test_full_width_byte_emission_dump_off_or_no_source_is_empty():
    # No source declared => no dump rules even when on.
    bundle = full_width_byte_emission(_fwbe_spec())
    assert bundle.dump_rules_builder(True, {}) == ()
    # Source declared but emission off => empty.
    bundle2 = full_width_byte_emission(
        _fwbe_spec(name="fwbe_s", band_name="FWBE_S_WIDE",
                   dump_value_source="SRC"))
    assert bundle2.dump_rules_builder(False, {}) == ()
    assert len(bundle2.dump_rules_builder(True, {})) == 240


def test_full_width_byte_emission_dump_fills_only_matching_cell():
    """Source one-hot value v + row gate on => only band cell v fills."""
    bundle = full_width_byte_emission(
        _fwbe_spec(name="fwbe_d", band_name="FWBE_D_WIDE",
                   dump_value_source="SRC",
                   dump_gate_conditions=(("IS_BYTE", 1.0), ("MARK", 1.0)),
                   dump_write_scale=2.0))
    rules = list(bundle.dump_rules_builder(True, {}))
    assert len(rules) == 240
    dp = {"SRC": 300, "FWBE_D_WIDE": 600, "IS_BYTE": 6, "MARK": 7}
    ffn = _lowered_ffn_at_S(rules, dp, S=100.0)
    out = _forward_band(ffn, dp, {"SRC+200": 1.0, "IS_BYTE": 1.0, "MARK": 1.0},
                        "FWBE_D_WIDE", 256)
    assert out[200] > 1.0
    assert all(abs(out[j]) < 1e-2 for j in range(256) if j != 200)


def test_full_width_byte_emission_dump_gate_blocks_when_off():
    bundle = full_width_byte_emission(
        _fwbe_spec(name="fwbe_g", band_name="FWBE_G_WIDE",
                   dump_value_source="SRC",
                   dump_gate_conditions=(("IS_BYTE", 1.0), ("MARK", 1.0)),
                   dump_write_scale=2.0))
    rules = list(bundle.dump_rules_builder(True, {}))
    dp = {"SRC": 300, "FWBE_G_WIDE": 600, "IS_BYTE": 6, "MARK": 7}
    ffn = _lowered_ffn_at_S(rules, dp, S=100.0)
    # MARK off => the AND fails => no cell fills.
    out = _forward_band(ffn, dp, {"SRC+200": 1.0, "IS_BYTE": 1.0},
                        "FWBE_G_WIDE", 256)
    assert all(abs(v) < 1e-2 for v in out)


def test_full_width_byte_emission_bad_bits_rejected():
    with pytest.raises(ValueError, match="bits must be"):
        _fwbe_spec(bits=0)
    with pytest.raises(ValueError, match="bits must be"):
        _fwbe_spec(bits=17)


def test_full_width_byte_emission_bad_lo_value_rejected():
    with pytest.raises(ValueError, match="lo_value"):
        _fwbe_spec(lo_value=256)  # >= 2**8
    with pytest.raises(ValueError, match="lo_value"):
        _fwbe_spec(lo_value=-1)


# ===========================================================================
# consumer_lookahead_gate — generator shape + byte-identity of generated parts
# ===========================================================================
#
# These mirror the #221 hand-built params EXACTLY (same opcode classes, band
# names, scalars) so the generated rules / heads are the SAME the production op
# factories install. The bands use the SAME owners as the live l5_ops
# registration (idempotent re-registration); the feature flag is forced False
# via the test override so collect_registered_residual_bands never widens the
# production build from these unit tests.

# The arithmetic consumer class (name, lo, hi) and the prior-arith class — the
# exact #221 tables.
_ARITH_CONSUMER_OPCODES = (
    ("OR", 14, 0), ("XOR", 15, 0),
    ("AND", 0, 1), ("SHL", 7, 1), ("SHR", 8, 1),
    ("ADD", 9, 1), ("SUB", 10, 1), ("MUL", 11, 1),
    ("DIV", 12, 1), ("MOD", 13, 1),
)
_PRIOR_ARITH_DIMS = (
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_OR", "OP_XOR", "OP_AND", "OP_SHL", "OP_SHR",
)


def _arith_gate_spec(**overrides) -> ConsumerLookaheadGateSpec:
    """A spec matching the #221 ARITH_CONSUMER gate (flag forced OFF so the test
    bands are never collected into the production build). Owners match the live
    l5_ops registration so register_residual_band is idempotent."""
    base = dict(
        name="stack0_arith_consumer",
        feature_flag=_never,
        pc_offset=8,
        pc_band_lo="LOOKAHEAD_PC_LO", pc_band_hi="LOOKAHEAD_PC_HI",
        pc_chain_source_lo="EMBED_LO", pc_chain_source_hi="EMBED_HI",
        pc_chain_gate_marker="MARK_AX",
        opcode_band_lo="NEXT_OPCODE_LO", opcode_band_hi="NEXT_OPCODE_HI",
        fetch_addr_key="ADDR_KEY",
        fetch_clean_embed_lo="CLEAN_EMBED_LO",
        fetch_clean_embed_hi="CLEAN_EMBED_HI",
        fetch_marker="MARK_AX", fetch_const="CONST", fetch_has_se="HAS_SE",
        consumer_opcodes=_ARITH_CONSUMER_OPCODES,
        consumer_flag_band="STACK0_B0_NEXT_ARITH",
        relay_target_marker="MARK_STACK0",
        prior_opcodes=_PRIOR_ARITH_DIMS,
        prior_latch_band="STACK0_PRIOR_ARITH",
        dump_block_band="STACK0_B0_DUMP_BLOCK",
        band_owner_pc="make_lookahead_pc8_chain_op",
        band_owner_opcode="make_lookahead_opcode_fetch_op",
        band_owner_flag="make_next_arith_flag_op",
        band_owner_latch="make_prior_arith_latch_op",
        band_owner_dump="make_dump_block_flag_op",
    )
    base.update(overrides)
    return ConsumerLookaheadGateSpec(**base)


def _arith_gate_dim_positions() -> dict:
    """Synthetic dim_positions covering every dim the gate's heads touch."""
    dp = {
        "ADDR_KEY": 100, "CLEAN_EMBED_LO": 200, "CLEAN_EMBED_HI": 220,
        "MARK_AX": 5, "CONST": 0, "HAS_SE": 6, "MARK_STACK0": 7,
        "EMBED_LO": 40, "EMBED_HI": 60,
        "LOOKAHEAD_PC_LO": 300, "LOOKAHEAD_PC_HI": 320,
        "NEXT_OPCODE_LO": 340, "NEXT_OPCODE_HI": 360,
        "STACK0_B0_NEXT_ARITH": 400, "STACK0_PRIOR_ARITH": 401,
        "STACK0_B0_DUMP_BLOCK": 402,
    }
    for i, op in enumerate(_PRIOR_ARITH_DIMS):
        dp[op] = 500 + i
    return dp


def test_consumer_lookahead_gate_returns_bundle():
    spec = _arith_gate_spec()
    bundle = consumer_lookahead_gate(spec)
    assert isinstance(bundle, ConsumerLookaheadGateBundle)
    assert bundle.spec is spec
    assert spec.has_prior_latch
    # The pc8 chain is offset=8 with carry => 32 + 32*8 = 288 units.
    assert bundle.pc_chain_hidden_dim == 288


def test_consumer_lookahead_gate_has_no_head_gate_field():
    """Gating lives ONLY in the dump-block AND-gate FFN — there is no per-head
    gate field on the spec (the carry/relay/latch heads are unconditional)."""
    fields = set(ConsumerLookaheadGateSpec.__dataclass_fields__)
    for forbidden in ("head_gate", "relay_gate", "latch_gate"):
        assert forbidden not in fields


def test_consumer_lookahead_gate_builder_counts():
    bundle = consumer_lookahead_gate(_arith_gate_spec())
    assert len(bundle.pc_chain_rules_builder(100.0)) == 288
    assert len(bundle.consumer_flag_rules_builder()) == 10  # 10 arith opcodes
    assert len(bundle.dump_block_rules_builder()) == 1      # the AND-gate
    assert bundle.prior_latch_head_spec_builder is not None


def test_consumer_lookahead_gate_consumer_flag_rule_shape():
    """Each consumer-flag rule is the two-nibble AND of one opcode, writing the
    bounded consumer flag at 1.0."""
    bundle = consumer_lookahead_gate(_arith_gate_spec())
    rules = bundle.consumer_flag_rules_builder()
    for (op_name, lo, hi), r in zip(_ARITH_CONSUMER_OPCODES, rules):
        cond_dims = {(c.dim.name, c.dim.offset) for c in r.conditions}
        assert ("NEXT_OPCODE_LO", lo) in cond_dims
        assert ("NEXT_OPCODE_HI", hi) in cond_dims
        assert r.writes[0].dim.name == "STACK0_B0_NEXT_ARITH"


def test_consumer_lookahead_gate_pc_chain_byte_identical():
    """Every generated PC+8 chain rule lowers byte-identically."""
    bundle = consumer_lookahead_gate(_arith_gate_spec())
    dp = _arith_gate_dim_positions()
    rules = bundle.pc_chain_rules_builder(100.0)
    bad = 0
    from c4_release.neural_vm.unified_compiler.ir import (
        compare_symbolic_to_lowered_ffn,
    )
    for r in rules:
        rep = compare_symbolic_to_lowered_ffn(r, dp, S=100.0)
        if not rep.ok:
            bad += 1
    assert bad == 0


def test_consumer_lookahead_gate_consumer_flag_byte_identical():
    bundle = consumer_lookahead_gate(_arith_gate_spec())
    dp = _arith_gate_dim_positions()
    from c4_release.neural_vm.unified_compiler.ir import (
        compare_symbolic_to_lowered_ffn,
    )
    for r in bundle.consumer_flag_rules_builder():
        assert compare_symbolic_to_lowered_ffn(r, dp, S=100.0).ok


def test_consumer_lookahead_gate_dump_block_lowers_to_and():
    """The dump-block AND-gate fires only when BOTH inputs are present (the silu
    saturates to ~1.0 within 1e-3; the 1e-5 default is below the silu's
    intrinsic rounding, identical to the hand-built rule)."""
    bundle = consumer_lookahead_gate(_arith_gate_spec())
    dp = _arith_gate_dim_positions()
    rule = bundle.dump_block_rules_builder()[0]
    ffn = _lowered_ffn_at_S([rule], dp, S=100.0)
    # Both present => fires.
    on = _forward_band(
        ffn, dp,
        {"STACK0_B0_NEXT_ARITH": 50.0, "STACK0_PRIOR_ARITH": 5.0},
        "STACK0_B0_DUMP_BLOCK", 1,
    )
    assert on[0] > 0.99
    # Consumer alone (no prior arith) => dark (single-op operand frame).
    consumer_only = _forward_band(
        ffn, dp, {"STACK0_B0_NEXT_ARITH": 50.0},
        "STACK0_B0_DUMP_BLOCK", 1,
    )
    assert abs(consumer_only[0]) < 1e-2
    # Prior alone (comparison consumer) => dark.
    prior_only = _forward_band(
        ffn, dp, {"STACK0_PRIOR_ARITH": 5.0},
        "STACK0_B0_DUMP_BLOCK", 1,
    )
    assert abs(prior_only[0]) < 1e-2


def test_consumer_lookahead_gate_heads_byte_identical():
    """The fetch / relay / latch heads lower byte-identically (self-consistent).

    ``PureAttention`` requires ``dim == num_heads * head_dim`` exactly, so we let
    the comparator INFER ``dim`` (the synthetic dp positions all fit under
    ``8 * 64 == 512``)."""
    bundle = consumer_lookahead_gate(_arith_gate_spec())
    dp = _arith_gate_dim_positions()
    assert max(dp.values()) < 8 * 64  # dp fits the inferred 512-wide layout
    for spec_builder, args in (
        (bundle.opcode_fetch_head_spec_builder, (dp, 6)),
        (lambda d, h: bundle.relay_head_spec_builder(d, h, 100.0), (dp, 6)),
        (lambda d, h: bundle.prior_latch_head_spec_builder(d, h, 100.0), (dp, 7)),
    ):
        head = spec_builder(*args)
        report = compare_symbolic_to_lowered_attn(
            head, head_dim=64, num_heads=8,
        )
        assert report.ok, report


def test_consumer_lookahead_gate_no_latch_degenerates():
    """With NO prior_opcodes the latch head is omitted and the dump-block gate
    degenerates to the consumer-class flag alone (a 5-band feature)."""
    spec = _arith_gate_spec(
        name="arith_gate_nolatch",
        prior_opcodes=(), prior_latch_band="",
        # distinct band names so the 5 bands don't collide with the live ones
        pc_band_lo="NL_PC_LO", pc_band_hi="NL_PC_HI",
        opcode_band_lo="NL_OP_LO", opcode_band_hi="NL_OP_HI",
        consumer_flag_band="NL_NEXT_ARITH",
        dump_block_band="NL_DUMP_BLOCK",
        # the passthrough threshold must be clearable by the consumer flag alone
        dump_block_consumer_weight=0.1, dump_block_threshold=4.0,
        band_owner_pc=None, band_owner_opcode=None, band_owner_flag=None,
        band_owner_latch=None, band_owner_dump=None,
    )
    bundle = consumer_lookahead_gate(spec)
    assert not spec.has_prior_latch
    assert bundle.prior_latch_head_spec_builder is None
    assert bundle.prior_latch_reads == set()
    # The dump-block rule reads ONLY the consumer flag (no latch term).
    rule = bundle.dump_block_rules_builder()[0]
    cond_names = {c.dim.name for c in rule.conditions}
    assert cond_names == {"NL_NEXT_ARITH"}


def test_consumer_lookahead_gate_bad_offset_rejected():
    with pytest.raises(ValueError, match="pc_offset"):
        _arith_gate_spec(pc_offset=0)


def test_consumer_lookahead_gate_empty_consumer_rejected():
    with pytest.raises(ValueError, match="consumer_opcodes"):
        _arith_gate_spec(consumer_opcodes=())


def test_consumer_lookahead_gate_prior_without_band_rejected():
    with pytest.raises(ValueError, match="prior_latch_band"):
        _arith_gate_spec(prior_opcodes=("OP_ADD",), prior_latch_band="")


# ===========================================================================
# Whole-model param-hash — the DECISIVE byte-identity gate (slow, guarded)
# ===========================================================================


@_requires_hash
@pytest.mark.slow
@pytest.mark.parametrize("bp_on,golden", [
    ("1", _GOLDEN_BP_ON),
    ("0", _GOLDEN_BP_OFF),
])
def test_whole_model_hash_bp_migration_byte_identical(monkeypatch, bp_on, golden):
    """The cross_step_carry BP migration is byte-identical: the whole-model
    state_dict SHA256 == the HEAD golden, flag-ON and flag-OFF. Full-width
    emission stays OFF (its default) so this isolates the BP migration."""
    monkeypatch.setenv("C4_BP_SAVE_DUMP", bp_on)
    monkeypatch.delenv("C4_AX_BYTE1_FULL_WIDTH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    digest = _model_state_hash()
    assert digest == golden, (
        f"C4_BP_SAVE_DUMP={bp_on}: hash {digest} != golden {golden}. "
        "The BP migration drifted; diff the lowered head/dump."
    )


@_requires_hash
@pytest.mark.slow
def test_whole_model_hash_full_width_off_is_byte_identical(monkeypatch):
    """The full-width emission op, registered but flag-OFF (default), leaves
    the whole-model hash == the HEAD golden (the band + columns are omitted)."""
    monkeypatch.setenv("C4_BP_SAVE_DUMP", "1")
    monkeypatch.delenv("C4_AX_BYTE1_FULL_WIDTH", raising=False)  # default off
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    digest = _model_state_hash()
    assert digest == _GOLDEN_BP_ON, (
        f"full-width OFF hash {digest} != golden {_GOLDEN_BP_ON}: the op is "
        "NOT byte-identical off."
    )


@_requires_hash
@pytest.mark.slow
def test_whole_model_hash_full_width_on_differs_and_bakes_columns(monkeypatch):
    """With the flag ON the hash DIFFERS (band + 240 columns added) and the
    LM head carries the un-aliased columns 16..255 at 5.0 (built layout)."""
    monkeypatch.setenv("C4_BP_SAVE_DUMP", "1")
    monkeypatch.setenv("C4_AX_BYTE1_FULL_WIDTH", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    model, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = getattr(layout, "dim_positions", layout)
    band = dp["AX_BYTE1_FULL_WIDE"]
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    # Un-aliased: each v in 16..255 has its OWN column at 5.0.
    for v in (16, 100, 200, 255):
        assert abs(float(W[v, band + v]) - 5.0) < 1e-5
    # lo_value boundary: v=15 has NO full-width column (the H-band path owns it).
    assert abs(float(W[15, band + 15])) < 1e-9


@_requires_hash
@pytest.mark.slow
@pytest.mark.parametrize("gate_on,golden", [
    ("1", _GOLDEN_ARITH_GATE_ON),
    ("0", _GOLDEN_ARITH_GATE_OFF),
])
def test_whole_model_hash_arith_gate_migration_byte_identical(
    monkeypatch, gate_on, golden,
):
    """The #221 consumer-lookahead-gate migration is byte-identical: the whole-
    model state_dict SHA256 == the HEAD golden, flag-ON and flag-OFF. The
    re-expression via ``consumer_lookahead_gate(ARITH_CONSUMER_SPEC)`` replaces
    the six hand-built #221 ops (PC+8 chain, opcode-fetch head, consumer-arith
    flag FFN, AX->STACK0 relay, causal prior-arith latch, AND-gate FFN) with a
    single declaration — and reproduces the lowered weights bit-for-bit.

    Flag-ON is the DEFAULT build (== the BP-on golden). Flag-OFF omits the six
    feature bands (smaller d_model) and so has a DISTINCT golden — the proof the
    generator's flag-off path is byte-identical to the pre-feature build too.
    Run on CPU with a fixed PYTHONHASHSEED (the FFN-packing order is seed-
    dependent) via ``C4_ISA_HASH_TEST=1 PYTHONHASHSEED=0``."""
    monkeypatch.setenv("C4_STACK0_NEXT_ARITH", gate_on)
    monkeypatch.setenv("C4_BP_SAVE_DUMP", "1")  # default-on companion feature
    monkeypatch.delenv("C4_AX_BYTE1_FULL_WIDTH", raising=False)  # default off
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    digest = _model_state_hash()
    assert digest == golden, (
        f"C4_STACK0_NEXT_ARITH={gate_on}: hash {digest} != golden {golden}. "
        "The consumer_lookahead_gate re-expression drifted; diff the generated "
        "vs hand-built chain/fetch/flag/relay/latch/AND-gate."
    )
