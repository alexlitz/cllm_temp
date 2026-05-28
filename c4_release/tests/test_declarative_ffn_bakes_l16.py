"""Parity tests for L16 FFN bands migrated to declarative rules."""

import torch
import pytest

from neural_vm.unified_compiler.ops.l16_ops import (
    _layer16_lev_routing_rules,
    lower_layer16_lev_routing_ir,
)
from neural_vm.vm_step import _SetDim, _set_layer16_lev_routing


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 768):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _assert_same_ffn(actual: _StubFFN, expected: _StubFFN):
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def _assert_same_ffn_prefix(actual: _StubFFN, expected: _StubFFN, units: int):
    assert torch.equal(actual.W_up[:units], expected.W_up[:units]), "W_up"
    assert torch.equal(actual.b_up[:units], expected.b_up[:units]), "b_up"
    assert torch.equal(actual.W_gate[:units], expected.W_gate[:units]), "W_gate"
    assert torch.equal(actual.b_gate[:units], expected.b_gate[:units]), "b_gate"
    assert torch.equal(actual.W_down[:, :units], expected.W_down[:, :units]), "W_down"


def test_layer16_lev_routing_ir_matches_legacy_helper():
    actual = _StubFFN()
    expected = _StubFFN()

    end = lower_layer16_lev_routing_ir(actual, 100.0, _SetDim)
    legacy_end = _set_layer16_lev_routing(expected, 100.0, _SetDim)

    assert legacy_end == 121
    assert end == 441
    assert len(_layer16_lev_routing_rules(100.0)) == 441
    _assert_same_ffn_prefix(actual, expected, legacy_end)
    assert actual.W_down[:, legacy_end:end].abs().sum() > 0


def test_layer16_lea_local_frame_byte1_rules_materialize_ff():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    lo = rules["l16_lea_local_ax_byte1_ff_lo"]
    hi = rules["l16_lea_local_ax_byte1_ff_hi"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("CMP+7", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("H1+1", 1.0) in condition_dims
    assert ("IS_BYTE+0", 1.0) in condition_dims
    assert ("BYTE_INDEX_0+0", 1.0) in condition_dims
    assert ("MARK_PC+0", -10.0) in condition_dims
    assert lo.threshold == 4.5
    assert hi.threshold == 4.5

    lo_writes = {write.dim.key(): write.weight for write in lo.writes}
    hi_writes = {write.dim.key(): write.weight for write in hi.writes}
    assert lo_writes["OUTPUT_LO+15"] == 20.0 / 100.0
    assert hi_writes["OUTPUT_HI+15"] == 20.0 / 100.0
    assert lo_writes["OUTPUT_LO+0"] == -20.0 / 100.0
    assert hi_writes["OUTPUT_HI+0"] == -20.0 / 100.0


def test_layer16_bp_marker_passthrough_reads_embed_and_blocks_frame_ops():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    lo = rules["l16_bp_marker_passthrough_lo_15"]
    hi = rules["l16_bp_marker_passthrough_hi_15"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("MARK_BP+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("IS_BYTE+0", -10.0) in condition_dims
    assert ("OP_ENT+0", -2.0) in condition_dims
    assert ("OP_LEV+0", -2.0) in condition_dims
    assert lo.threshold == 1.5
    assert lo.gate.key() == "EMBED_LO+15"
    assert hi.gate.key() == "EMBED_HI+15"
    assert lo.writes[0].dim.key() == "OUTPUT_LO+15"
    assert hi.writes[0].dim.key() == "OUTPUT_HI+15"
    assert lo.writes[0].weight == 10.0 / 100.0
    assert hi.writes[0].weight == 10.0 / 100.0


def test_layer16_bp_frame_byte1_ff_restores_post_ent_bp_stream():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules_by_name = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules_by_name["l16_bp_frame_byte1_ff"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("IS_BYTE+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("H1+3", 1.0) in condition_dims
    assert ("BYTE_INDEX_0+0", 1.0) in condition_dims
    assert ("CLEAN_EMBED_LO+0", 1.0) in condition_dims
    assert ("CLEAN_EMBED_HI+15", 1.0) in condition_dims
    assert ("MARK_BP+0", -1.0) in condition_dims
    assert rule.threshold == 5.0

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+15"] == 50.0 / 100.0
    assert writes["OUTPUT_HI+15"] == 50.0 / 100.0
    assert writes["OUTPUT_LO+0"] == -50.0 / 100.0
    assert writes["OUTPUT_HI+0"] == -50.0 / 100.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    bp_byte1_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9983,
        "H1+3": 1.0,
        "BYTE_INDEX_0": 0.9701,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
        "OUTPUT_LO+0": 2.26,
        "OUTPUT_LO+15": 2.0,
        "OUTPUT_HI+0": 3.47,
        "OUTPUT_HI+15": 2.0,
    }
    out = ir.symbolic_ffn(bp_byte1_state)
    assert out["OUTPUT_LO+15"] > bp_byte1_state["OUTPUT_LO+15"]
    assert out["OUTPUT_HI+15"] > bp_byte1_state["OUTPUT_HI+15"]
    assert out["OUTPUT_LO+0"] < bp_byte1_state["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+0"] < bp_byte1_state["OUTPUT_HI+0"]

    initial_bp_state = dict(
        bp_byte1_state,
        **{
            "CLEAN_EMBED_LO+0": 1.0,
            "CLEAN_EMBED_HI+15": 0.0,
            "CLEAN_EMBED_HI+0": 1.0,
        },
    )
    out_initial = ir.symbolic_ffn(initial_bp_state)
    assert out_initial["OUTPUT_LO+15"] == bp_byte1_state["OUTPUT_LO+15"]

    marker_state = dict(bp_byte1_state, **{"MARK_BP": 1.0})
    out_marker = ir.symbolic_ffn(marker_state)
    assert out_marker["OUTPUT_LO+15"] == bp_byte1_state["OUTPUT_LO+15"]


def test_layer16_top_store_stack0_restore_boosts_staged_nonzero_nibbles():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    lo = rules["l16_top_store_stack0_restore_lo_1"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("MARK_STACK0+0", 1.0) in condition_dims
    assert ("CMP+3", 1.0) in condition_dims
    assert ("MEM_STORE+0", 1.0) in condition_dims
    assert ("EMBED_LO+0", 1.0) in condition_dims
    assert ("IS_BYTE+0", -10.0) in condition_dims
    assert ("MARK_SP+0", -10.0) in condition_dims
    assert lo.threshold == 7.0
    assert lo.gate.key() == "OUTPUT_LO+1"

    lo_writes = {write.dim.key(): write.weight for write in lo.writes}
    assert lo_writes["OUTPUT_LO+1"] == 50.0 / 100.0
    assert lo_writes["OUTPUT_LO+0"] == -50.0 / 100.0


def test_layer16_jmp_ax_preserve_blocks_active_fetch_nibble():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    rule = rules["l16_jmp_ax_preserve_lo_2"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("OP_JMP+0", 0.2) in condition_dims
    assert ("MARK_AX+0", 1.0) in condition_dims
    assert ("IS_BYTE+0", -10.0) in condition_dims
    assert ("FETCH_LO+2", -2.0) in condition_dims
    assert rule.threshold == 1.5
    assert rule.gate.key() == "OUTPUT_LO+2"

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+2"] == 50.0 / 100.0
    assert writes["OUTPUT_LO+12"] == -50.0 / 100.0
    assert writes["AX_CARRY_LO+2"] == 50.0 / 100.0
    assert writes["AX_CARRY_LO+12"] == -50.0 / 100.0

    low11 = rules["l16_jmp_ax_preserve_lo_11"]
    low11_conditions = {
        (term.dim.key(), term.weight) for term in low11.conditions
    }
    assert ("TEMP+11", -2.0) in low11_conditions


def test_layer16_stack0_byte1_zero_blocks_retained_byte0_replay():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    rule = rules["l16_stack0_byte1_zero_after_unit_low_byte"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("STACK0_BYTE0+0", 1.0) in condition_dims
    assert ("CMP+3", 1.0) in condition_dims
    assert ("ADDR_B0_LO+0", 1.0) in condition_dims
    assert ("ADDR_B0_HI+14", 1.0) in condition_dims
    assert ("CLEAN_EMBED_LO+1", 1.0) in condition_dims
    assert ("CLEAN_EMBED_HI+0", 1.0) in condition_dims
    assert rule.threshold == 8.0
    assert rule.gate.key() == "CLEAN_EMBED_HI+0"

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+0"] == 1000.0 / 100.0
    assert writes["OUTPUT_LO+1"] == -1000.0 / 100.0
    assert writes["OUTPUT_HI+0"] == 1000.0 / 100.0
    assert writes["OUTPUT_HI+1"] == -1000.0 / 100.0


def test_layer16_stack0_byte1_zero_has_lowered_near_miss_margin():
    rules = list(_layer16_lev_routing_rules(100.0))
    unit = next(
        idx
        for idx, rule in enumerate(rules)
        if rule.name == "l16_stack0_byte1_zero_after_unit_low_byte"
    )
    ffn = _StubFFN()
    lower_layer16_lev_routing_ir(ffn, 100.0, _SetDim)

    near_miss = torch.zeros(512)
    near_miss[_SetDim.IS_BYTE] = 1.0
    near_miss[_SetDim.STACK0_BYTE0] = 0.9734056
    near_miss[_SetDim.HAS_SE] = 0.9984541
    near_miss[_SetDim.CMP + 3] = 4.0
    near_miss[_SetDim.ADDR_B0_LO + 0] = 0.9999852

    intended = near_miss.clone()
    intended[_SetDim.STACK0_BYTE0] = 1.0
    intended[_SetDim.HAS_SE] = 1.0
    intended[_SetDim.ADDR_B0_LO + 0] = 1.0
    intended[_SetDim.ADDR_B0_HI + 14] = 1.0
    intended[_SetDim.CLEAN_EMBED_LO + 1] = 1.0
    intended[_SetDim.CLEAN_EMBED_HI + 0] = 1.0

    near_miss_up = torch.dot(ffn.W_up[unit], near_miss) + ffn.b_up[unit]
    near_miss_gate = torch.dot(ffn.W_gate[unit], near_miss) + ffn.b_gate[unit]
    near_miss_hidden = torch.nn.functional.silu(near_miss_up) * near_miss_gate
    near_miss_delta = ffn.W_down[:, unit] * near_miss_hidden
    intended_up = torch.dot(ffn.W_up[unit], intended) + ffn.b_up[unit]
    intended_gate = torch.dot(ffn.W_gate[unit], intended) + ffn.b_gate[unit]
    intended_hidden = torch.nn.functional.silu(intended_up) * intended_gate
    intended_delta = ffn.W_down[:, unit] * intended_hidden

    assert -5.0 < near_miss_up < 0.0
    assert torch.max(torch.abs(
        near_miss_delta[_SetDim.OUTPUT_LO : _SetDim.OUTPUT_LO + 16]
    )) < 1e-5
    assert torch.max(torch.abs(
        near_miss_delta[_SetDim.OUTPUT_HI : _SetDim.OUTPUT_HI + 16]
    )) < 1e-5
    assert intended_up > 250.0
    assert intended_delta[_SetDim.OUTPUT_LO + 0] > 100.0
    assert intended_delta[_SetDim.OUTPUT_HI + 0] > 100.0
    assert intended_delta[_SetDim.OUTPUT_LO + 1] < -100.0
    assert intended_delta[_SetDim.OUTPUT_HI + 1] < -100.0


def test_layer16_jsr_return_addr_byte1_from_low_22_materializes_01():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_jsr_return_addr_byte1_01_from_low_22"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("OP_JSR+0", 0.2) in condition_dims
    assert ("CMP+4", 0.2) in condition_dims
    assert ("STACK0_BYTE0+0", 1.0) in condition_dims
    assert ("BYTE_INDEX_0+0", 1.0) in condition_dims
    assert ("CLEAN_EMBED_LO+2", 1.0) in condition_dims
    assert ("CLEAN_EMBED_HI+2", 2.0) in condition_dims
    assert ("MARK_STACK0+0", -10.0) in condition_dims
    assert rule.threshold == 9.0

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+1"] == 500.0 / 100.0
    assert writes["OUTPUT_LO+0"] == -500.0 / 100.0
    assert "OUTPUT_HI+0" not in writes

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    recursive_jsr_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9965,
        "OP_JSR": 11.0,
        "CMP+4": 2.0,
        "STACK0_BYTE0": 0.9734,
        "BYTE_INDEX_0": 0.9734,
        "CLEAN_EMBED_LO+2": 1.0,
        "CLEAN_EMBED_HI+2": 1.0,
        "OUTPUT_LO+0": 3.95,
    }
    out = ir.symbolic_ffn(recursive_jsr_state)
    assert out["OUTPUT_LO+1"] > recursive_jsr_state.get("OUTPUT_LO+1", 0.0)
    assert out["OUTPUT_LO+0"] < recursive_jsr_state["OUTPUT_LO+0"]

    return_5a_state = dict(
        recursive_jsr_state,
        **{"CLEAN_EMBED_LO+2": 0.0, "CLEAN_EMBED_HI+2": 0.0},
    )
    out_5a = ir.symbolic_ffn(return_5a_state)
    assert out_5a.get("OUTPUT_LO+1", 0.0) == recursive_jsr_state.get("OUTPUT_LO+1", 0.0)

    return_c2_state = dict(recursive_jsr_state, **{"CLEAN_EMBED_HI+2": 0.0})
    out_c2 = ir.symbolic_ffn(return_c2_state)
    assert out_c2.get("OUTPUT_LO+1", 0.0) == recursive_jsr_state.get("OUTPUT_LO+1", 0.0)

    marker_state = dict(recursive_jsr_state, **{"MARK_STACK0": 1.0})
    out_marker = ir.symbolic_ffn(marker_state)
    assert out_marker.get("OUTPUT_LO+1", 0.0) == recursive_jsr_state.get("OUTPUT_LO+1", 0.0)

    psh_state = dict(recursive_jsr_state, **{"OP_JSR": 0.0, "OP_PSH": 5.0})
    out_psh = ir.symbolic_ffn(psh_state)
    assert out_psh.get("OUTPUT_LO+1", 0.0) == recursive_jsr_state.get("OUTPUT_LO+1", 0.0)


def test_layer16_jsr_initial_stack0_marker_materializes_0a():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_jsr_initial_stack0_marker_0a"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("OP_JSR+0", 50.0) in condition_dims
    assert ("OP_ENT+0", -1000.0) in condition_dims
    assert ("CMP+4", 0.2) in condition_dims
    assert ("MARK_STACK0+0", 20.0) in condition_dims
    assert ("HAS_SE+0", -150.0) in condition_dims
    assert ("MEM_STORE+0", -100.0) in condition_dims
    assert ("ADDR_B0_LO+8", 20.0) in condition_dims
    assert ("ADDR_B0_LO+0", -20.0) in condition_dims
    assert ("ADDR_B0_HI+14", -20.0) in condition_dims
    assert ("ADDR_B0_HI+15", 20.0) in condition_dims
    assert ("IS_BYTE+0", -300.0) in condition_dims
    assert ("MARK_AX+0", -300.0) in condition_dims
    assert ("MARK_MEM+0", -300.0) in condition_dims
    assert rule.threshold == 320.0

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+10"] == 20.0
    assert writes["OUTPUT_HI+0"] == 20.0
    assert writes["OUTPUT_LO+0"] == -20.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    out = ir.symbolic_ffn({
        "OP_JSR": 22.16,
        "CMP+4": 1.02,
        "MARK_STACK0": 2.0,
        "ADDR_B0_LO+8": 2.0,
        "ADDR_B0_HI+15": 2.0,
        "OUTPUT_LO+0": 0.98,
        "OUTPUT_HI+0": 1.0,
    })
    assert out["OUTPUT_LO+10"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+0"] > 1.0

    ax_byte_state = ir.symbolic_ffn({
        "OP_JSR": 11.1,
        "IS_BYTE": 1.0,
        "OUTPUT_LO+0": 0.98,
    })
    assert ax_byte_state["OUTPUT_LO+0"] == 0.98
    assert ax_byte_state.get("OUTPUT_LO+10", 0.0) == 0.0

    stale_after_ent = ir.symbolic_ffn({
        "MARK_STACK0": 2.0,
        "ADDR_B0_LO+8": -4.0,
        "ADDR_B0_HI+15": 15.9,
        "OUTPUT_LO+0": 0.98,
    })
    assert stale_after_ent["OUTPUT_LO+0"] == 0.98
    assert stale_after_ent.get("OUTPUT_LO+10", 0.0) == 0.0

    recursive_slot = ir.symbolic_ffn({
        "OP_JSR": 11.1,
        "CMP+4": 1.02,
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998,
        "ADDR_B0_LO+0": 1.98,
        "ADDR_B0_LO+8": -1.94,
        "ADDR_B0_HI+14": 1.97,
        "ADDR_B0_HI+15": -2.0,
        "OUTPUT_LO+0": 0.98,
    })
    assert recursive_slot["OUTPUT_LO+0"] == 0.98

    ent_stack0_state = {
        "OP_ENT": 8.5,
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998,
        "MEM_STORE": 0.4,
        "ADDR_B0_LO+0": -19.0,
        "ADDR_B0_LO+8": -0.6,
        "ADDR_B0_HI+14": -19.8,
        "ADDR_B0_HI+15": -2.0,
        "OUTPUT_LO+0": 15.8,
    }
    out_ent_stack0 = ir.symbolic_ffn(ent_stack0_state)
    assert out_ent_stack0["OUTPUT_LO+0"] == ent_stack0_state["OUTPUT_LO+0"]


def test_layer16_stack0_e8_marker_materializes_from_alu():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    lo = rules["l16_stack0_e8_marker_from_alu_lo_9"]
    hi = rules["l16_stack0_e8_marker_from_alu_hi_3"]
    auth = rules["l16_stack0_e8_output_authoritative_de"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("MARK_STACK0+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("ADDR_B0_LO+8", 10.0) in condition_dims
    assert ("ADDR_B0_HI+14", 1.0) in condition_dims
    assert ("ADDR_B0_HI+15", -2.0) in condition_dims
    assert ("MEM_STORE+0", -20.0) in condition_dims
    assert ("OP_JSR+0", -10.0) in condition_dims
    assert ("MARK_MEM+0", -300.0) in condition_dims
    assert lo.threshold == 12.0
    assert lo.gate.key() == "ALU_LO+9"
    assert hi.gate.key() == "ALU_HI+3"
    assert lo.writes[0].dim.key() == "OUTPUT_LO+9"
    assert hi.writes[0].dim.key() == "OUTPUT_HI+3"
    auth_conditions = {(term.dim.key(), term.weight) for term in auth.conditions}
    assert ("MARK_STACK0+0", 1_000_000.0) in auth_conditions
    assert ("OUTPUT_LO+14", 10.0) in auth_conditions
    assert ("OUTPUT_HI+13", 10.0) in auth_conditions
    assert ("MEM_STORE+0", -1_000_000.0) in auth_conditions
    assert auth.threshold == 1_000_200.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend((lo, hi, auth))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "ALU_LO+9": 0.994,
        "ALU_HI+3": 0.994,
        "OUTPUT_LO+0": 0.001,
        "OUTPUT_HI+0": 0.001,
    })
    assert out["OUTPUT_LO+9"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+3"] > out["OUTPUT_HI+0"]

    jsr_slot = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "OP_JSR": 5.0,
        "ALU_LO+9": 1.0,
    })
    assert jsr_slot.get("OUTPUT_LO+9", 0.0) == 0.0

    current_store = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998318076133728,
        "ADDR_B0_LO+8": 0.9888221025466919,
        "ADDR_B0_HI+14": 1.9716130495071411,
        "ADDR_B0_HI+15": -1.9999948740005493,
        "MEM_STORE": 0.40725404024124146,
        "ALU_LO+9": 3.5585670471191406,
        "ALU_HI+3": 3.1141154766082764,
        "OUTPUT_LO+14": 3.0,
        "OUTPUT_HI+13": 3.0,
    })
    assert current_store["OUTPUT_LO+14"] == 3.0
    assert current_store["OUTPUT_HI+13"] == 3.0
    assert current_store.get("OUTPUT_LO+9", 0.0) == 0.0
    assert current_store.get("OUTPUT_HI+3", 0.0) == 0.0

    ent_mem_addr0 = CompilerIR()
    ent_mem_addr0.layer(0).ffn.rules.extend(
        rule
        for rule in rules.values()
        if rule.name.startswith("l16_stack0_e8_marker_from_alu_")
    )
    ent_mem_addr0_out = ent_mem_addr0.symbolic_ffn({
        "OP_ENT": 10.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 4.0,
        "HAS_SE": 0.9955,
        "H1+4": 1.0,
        "H3+4": 1.0,
        "ADDR_B0_LO+0": -24.331579208374023,
        "ADDR_B0_LO+8": 25.4189453125,
        "ADDR_B0_HI+14": 20.25174903869629,
        "ADDR_B0_HI+15": -42.10127639770508,
        "ALU_LO+0": -83.75157928466797,
        "ALU_LO+14": -84.9998779296875,
        "ALU_HI+0": -84.26598358154297,
        "ALU_HI+15": -85.0,
        "OUTPUT_LO+0": 139.67910766601562,
        "OUTPUT_HI+15": 143.8682403564453,
    })
    assert ent_mem_addr0_out["OUTPUT_LO+0"] == 139.67910766601562
    assert ent_mem_addr0_out["OUTPUT_HI+15"] == 143.8682403564453
    assert ent_mem_addr0_out.get("OUTPUT_LO+14", 0.0) == 0.0
    assert ent_mem_addr0_out.get("OUTPUT_HI+0", 0.0) == 0.0

    e0_slot = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9981152415275574,
        "ADDR_B0_LO+0": 7.978802680969238,
        "ADDR_B0_LO+8": -1.9999994039535522,
        "ADDR_B0_HI+14": 7.971285343170166,
        "ADDR_B0_HI+15": -1.9984419345855713,
        "ALU_LO+7": 0.9937703609466553,
        "ALU_HI+1": 0.9937703609466553,
        "OUTPUT_LO+0": 3.9529022615170106e-05,
        "OUTPUT_HI+0": 3.4088981919921935e-05,
    })
    assert e0_slot["OUTPUT_LO+0"] == 3.9529022615170106e-05
    assert e0_slot["OUTPUT_HI+0"] == 3.4088981919921935e-05
    assert e0_slot.get("OUTPUT_LO+7", 0.0) == 0.0
    assert e0_slot.get("OUTPUT_HI+1", 0.0) == 0.0

    loaded_stack0 = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+14": 1.0,
        "ALU_LO+9": 1.0,
        "ALU_HI+3": 1.0,
        "OUTPUT_LO+14": 40.0,
        "OUTPUT_HI+13": 40.0,
    })
    assert loaded_stack0["OUTPUT_LO+14"] > 40.0
    assert loaded_stack0["OUTPUT_HI+13"] > 40.0
    assert loaded_stack0["OUTPUT_LO+14"] > loaded_stack0.get("OUTPUT_LO+9", 0.0)
    assert loaded_stack0["OUTPUT_HI+13"] > loaded_stack0.get("OUTPUT_HI+3", 0.0)


def test_layer16_stack0_e0_marker_exact_e8_materializes_from_alu():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_stack0_e0_marker_e8_from_alu_exact"]
    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}

    assert ("MARK_STACK0+0", 1.0) in condition_dims
    assert ("ADDR_B0_LO+0", 1.0) in condition_dims
    assert ("ADDR_B0_HI+14", 1.0) in condition_dims
    assert ("ALU_LO+8", 1.0) in condition_dims
    assert ("ALU_HI+14", 1.0) in condition_dims
    assert ("MEM_STORE+0", -20.0) in condition_dims
    assert rule.threshold == 19.0

    ir = CompilerIR()
    ir.layer(0).ffn.append(rule)

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9981496334075928,
        "ADDR_B0_LO+0": 7.978207588195801,
        "ADDR_B0_LO+8": -1.9999960660934448,
        "ADDR_B0_HI+14": 7.970700263977051,
        "ADDR_B0_HI+15": -1.9999877214431763,
        "ALU_LO+8": 0.9937704801559448,
        "ALU_HI+14": 0.9937731623649597,
        "OUTPUT_LO+0": 3.950224709114991e-05,
        "OUTPUT_HI+0": 3.413946251384914e-05,
    })
    assert out["OUTPUT_LO+8"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+14"] > out["OUTPUT_HI+0"]

    unrelated_e0_slot = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.9981152415275574,
        "ADDR_B0_LO+0": 7.978802680969238,
        "ADDR_B0_LO+8": -1.9999994039535522,
        "ADDR_B0_HI+14": 7.971285343170166,
        "ADDR_B0_HI+15": -1.9984419345855713,
        "ALU_LO+7": 0.9937703609466553,
        "ALU_HI+1": 0.9937703609466553,
        "OUTPUT_LO+0": 3.9529022615170106e-05,
        "OUTPUT_HI+0": 3.4088981919921935e-05,
    })
    assert unrelated_e0_slot["OUTPUT_LO+0"] == 3.9529022615170106e-05
    assert unrelated_e0_slot["OUTPUT_HI+0"] == 3.4088981919921935e-05
    assert unrelated_e0_slot.get("OUTPUT_LO+8", 0.0) == 0.0
    assert unrelated_e0_slot.get("OUTPUT_HI+14", 0.0) == 0.0

    current_store = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 1.0,
        "ADDR_B0_LO+0": 8.0,
        "ADDR_B0_HI+14": 8.0,
        "ALU_LO+8": 1.0,
        "ALU_HI+14": 1.0,
        "MEM_STORE": 0.4,
        "OUTPUT_LO+0": 0.1,
        "OUTPUT_HI+0": 0.1,
    })
    assert current_store["OUTPUT_LO+0"] == 0.1
    assert current_store["OUTPUT_HI+0"] == 0.1
    assert current_store.get("OUTPUT_LO+8", 0.0) == 0.0
    assert current_store.get("OUTPUT_HI+14", 0.0) == 0.0


def test_layer16_stack0_f8_marker_materializes_from_alu():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    lo = rules["l16_stack0_f8_marker_from_alu_lo_11"]
    hi = rules["l16_stack0_f8_marker_from_alu_hi_2"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("MARK_STACK0+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("ADDR_B0_LO+8", 1.0) in condition_dims
    assert ("ADDR_B0_HI+15", 1.0) in condition_dims
    assert ("ADDR_B0_HI+14", -2.0) in condition_dims
    assert ("MEM_STORE+0", -20.0) in condition_dims
    assert lo.threshold == 3.5
    assert lo.gate.key() == "ALU_LO+11"
    assert hi.gate.key() == "ALU_HI+2"

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend((lo, hi))

    out = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "ALU_LO+11": 0.994,
        "ALU_HI+2": 0.994,
    })
    assert out["OUTPUT_LO+11"] > 0.0
    assert out["OUTPUT_HI+2"] > 0.0

    current_store = ir.symbolic_ffn({
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998,
        "ADDR_B0_LO+8": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "MEM_STORE": 0.4,
        "ALU_LO+11": 1.0,
        "ALU_HI+2": 1.0,
    })
    assert current_store.get("OUTPUT_LO+11", 0.0) == 0.0
    assert current_store.get("OUTPUT_HI+2", 0.0) == 0.0


def test_layer16_stack0_cancels_false_positive_lev_sp_value_materializer():
    rules = list(_layer16_lev_routing_rules(100.0))
    original_unit = next(
        idx
        for idx, rule in enumerate(rules)
        if rule.name == "l16_lev_sp_bp_plus16_lo_8"
    )
    cancel_unit = next(
        idx
        for idx, rule in enumerate(rules)
        if rule.name == "l16_stack0_cancel_lev_sp_lo_8"
    )
    cancel_rule = rules[cancel_unit]
    condition_dims = {
        (term.dim.key(), term.weight) for term in cancel_rule.conditions
    }
    assert ("MARK_STACK0+0", 50.0) in condition_dims
    assert cancel_rule.threshold == 105.0
    assert cancel_rule.gate.key() == "ADDR_B0_LO+8"
    assert cancel_rule.writes[0].dim.key() == "OUTPUT_LO+8"
    assert cancel_rule.writes[0].weight == -2.0 / 100.0

    ffn = _StubFFN()
    lower_layer16_lev_routing_ir(ffn, 100.0, _SetDim)

    stack0_near_miss = torch.zeros(512)
    stack0_near_miss[_SetDim.MARK_STACK0] = 1.0
    stack0_near_miss[_SetDim.HAS_SE] = 0.9956
    stack0_near_miss[_SetDim.ADDR_B0_LO + 8] = 26.4189

    original_up = (
        torch.dot(ffn.W_up[original_unit], stack0_near_miss)
        + ffn.b_up[original_unit]
    )
    cancel_up = (
        torch.dot(ffn.W_up[cancel_unit], stack0_near_miss)
        + ffn.b_up[cancel_unit]
    )
    assert original_up > 0
    assert torch.isclose(original_up, cancel_up)

    original_hidden = (
        torch.nn.functional.silu(original_up)
        * (
            torch.dot(ffn.W_gate[original_unit], stack0_near_miss)
            + ffn.b_gate[original_unit]
        )
    )
    cancel_hidden = (
        torch.nn.functional.silu(cancel_up)
        * (
            torch.dot(ffn.W_gate[cancel_unit], stack0_near_miss)
            + ffn.b_gate[cancel_unit]
        )
    )
    original_delta = (
        ffn.W_down[_SetDim.OUTPUT_LO + 8, original_unit]
        * original_hidden
    )
    cancel_delta = (
        ffn.W_down[_SetDim.OUTPUT_LO + 8, cancel_unit]
        * cancel_hidden
    )
    assert original_delta > 0
    assert torch.isclose(
        original_delta + cancel_delta,
        torch.tensor(0.0),
        atol=1e-3,
    )

    intended_sp = torch.zeros(512)
    intended_sp[_SetDim.OP_LEV] = 5.0
    intended_sp[_SetDim.MARK_SP] = 1.0
    intended_sp[_SetDim.HAS_SE] = 1.0
    intended_sp[_SetDim.ADDR_B0_LO + 8] = 26.0
    intended_cancel_up = (
        torch.dot(ffn.W_up[cancel_unit], intended_sp)
        + ffn.b_up[cancel_unit]
    )
    assert intended_cancel_up < -100.0


def test_layer16_bp_after_ent_byte2_zero_blocks_initial_bp_tail():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    rule = rules["l16_bp_after_ent_byte2_zero"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("H1+3", 10.0) in condition_dims
    assert ("BYTE_INDEX_1+0", 1.0) in condition_dims
    assert ("CLEAN_EMBED_LO+15", 1.0) in condition_dims
    assert ("CLEAN_EMBED_HI+15", 1.0) in condition_dims
    assert ("H3+4", -100.0) in condition_dims
    assert ("MARK_BP+0", -100.0) in condition_dims
    assert rule.threshold == 14.5

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+0"] == 10000.0 / 100.0
    assert writes["OUTPUT_HI+0"] == 10000.0 / 100.0
    assert writes["OUTPUT_LO+1"] == -10000.0 / 100.0
    assert writes["OUTPUT_HI+1"] == -10000.0 / 100.0


def test_layer16_ent_initial_stack0_byte2_writes_01_only_on_initial_main_ent():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules_by_name = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules_by_name["l16_ent_initial_stack0_byte2_01"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("IS_BYTE+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("STACK0_BYTE1+0", 30.0) in condition_dims
    assert ("OP_ENT+0", 2.0) in condition_dims
    assert ("CLEAN_EMBED_LO+0", 8.0) in condition_dims
    assert ("CLEAN_EMBED_HI+0", 8.0) in condition_dims
    assert ("ADDR_B0_HI+14", -10.0) in condition_dims
    assert ("MEM_STORE+0", -2.0) in condition_dims
    assert ("MARK_BP+0", -10.0) in condition_dims
    assert ("MARK_STACK0+0", -10.0) in condition_dims
    assert rule.threshold == 49.5

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+1"] == 4.0 / 100.0
    assert writes["OUTPUT_LO+0"] == -4.0 / 100.0
    assert "OUTPUT_HI+0" not in writes  # OUTPUT_HI already correct from L3 default
    assert "OUTPUT_HI+1" not in writes

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    initial_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "STACK0_BYTE1": 1.0,
        "OP_ENT": 5.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
    }
    out = ir.symbolic_ffn(initial_state)
    assert out["OUTPUT_LO+1"] > 0.0
    assert out["OUTPUT_LO+0"] < 0.0

    nested_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "STACK0_BYTE1": 1.0,
        "OP_ENT": 5.0,
        # nested ENT: saved BP byte 1 = 0xff → CLEAN_EMBED_LO/HI+15 active,
        # the initial-state byte-0 sentinels stay zero.
        "CLEAN_EMBED_LO+15": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
    }
    out_nested = ir.symbolic_ffn(nested_state)
    assert out_nested.get("OUTPUT_LO+1", 0.0) == 0.0
    assert out_nested.get("OUTPUT_HI+0", 0.0) == 0.0

    local_frame_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9956,
        "STACK0_BYTE1": 0.9701,
        "OP_ENT": 2.1712,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "ADDR_B0_HI+14": 1.0,
    }
    out_local_frame = ir.symbolic_ffn(local_frame_state)
    assert out_local_frame.get("OUTPUT_LO+1", 0.0) == 0.0
    assert out_local_frame.get("OUTPUT_LO+0", 0.0) == 0.0

    non_ent_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "STACK0_BYTE1": 1.0,
        "OP_ENT": 0.0,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
    }
    out_non_ent = ir.symbolic_ffn(non_ent_state)
    assert out_non_ent.get("OUTPUT_LO+1", 0.0) == 0.0


def test_layer16_stack0_saved_bp_byte2_persists_after_initial_ent():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules_by_name = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    lo0_rule = rules_by_name["l16_stack0_saved_bp_byte2_01.OUTPUT_LO+0"]
    lo1_rule = rules_by_name["l16_stack0_saved_bp_byte2_01.OUTPUT_LO+1"]

    condition_dims = {
        (term.dim.key(), term.weight) for term in lo1_rule.conditions
    }
    assert ("IS_BYTE+0", 0.1) in condition_dims
    assert ("HAS_SE+0", 0.1) in condition_dims
    assert ("STACK0_BYTE1+0", 0.1) in condition_dims
    assert ("BYTE_INDEX_1+0", 0.1) in condition_dims
    assert ("CLEAN_EMBED_LO+0", 0.1) in condition_dims
    assert ("CLEAN_EMBED_HI+0", 0.1) in condition_dims
    assert ("ADDR_B0_LO+0", 0.1) in condition_dims
    assert ("ADDR_B0_HI+15", 0.1) in condition_dims
    assert ("OP_ENT+0", -1.0) in condition_dims
    assert ("MARK_STACK0+0", -1.0) in condition_dims
    assert lo0_rule.threshold == 0.78
    assert lo1_rule.threshold == 0.78
    assert lo0_rule.gate_terms[0].dim.key() == "OUTPUT_LO+0"
    assert lo0_rule.gate_terms[0].weight == -1.0
    assert lo0_rule.gate_bias == 0.0
    assert lo1_rule.gate_terms[0].dim.key() == "OUTPUT_LO+1"
    assert lo1_rule.gate_terms[0].weight == -1.0
    assert lo1_rule.gate_bias == 1.0

    writes0 = {write.dim.key(): write.weight for write in lo0_rule.writes}
    writes1 = {write.dim.key(): write.weight for write in lo1_rule.writes}
    assert writes0 == {"OUTPUT_LO+0": 1.0}
    assert writes1 == {"OUTPUT_LO+1": 1.0}

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend((lo0_rule, lo1_rule))

    saved_bp_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9963,
        "STACK0_BYTE1": 0.9701,
        "BYTE_INDEX_1": 0.9701,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "ADDR_B0_LO+0": 1.0,
        "ADDR_B0_HI+15": 1.0,
        "OUTPUT_LO+0": 0.9401,
        "OUTPUT_LO+1": 0.02,
    }
    out = ir.symbolic_ffn(saved_bp_state)
    assert out["OUTPUT_LO+1"] == 1.0
    assert out["OUTPUT_LO+0"] == 0.0

    ent_owned_state = dict(saved_bp_state, **{"OP_ENT": 5.0})
    out_ent = ir.symbolic_ffn(ent_owned_state)
    assert out_ent["OUTPUT_LO+1"] == saved_bp_state["OUTPUT_LO+1"]

    wrong_stack_addr_state = dict(saved_bp_state)
    wrong_stack_addr_state["ADDR_B0_HI+15"] = 0.0
    out_wrong_addr = ir.symbolic_ffn(wrong_stack_addr_state)
    assert out_wrong_addr["OUTPUT_LO+1"] == saved_bp_state["OUTPUT_LO+1"]

    byte0_state = dict(
        saved_bp_state,
        **{
            "STACK0_BYTE0": 0.9701,
            "BYTE_INDEX_0": 0.9701,
            "BYTE_INDEX_1": 0.02,
        },
    )
    out_byte0 = ir.symbolic_ffn(byte0_state)
    assert out_byte0["OUTPUT_LO+1"] == saved_bp_state["OUTPUT_LO+1"]

    return_addr_state = dict(
        saved_bp_state,
        **{"ADDR_B0_LO+0": 0.0, "ADDR_B0_LO+8": 0.9701},
    )
    out_return_addr = ir.symbolic_ffn(return_addr_state)
    assert out_return_addr["OUTPUT_LO+1"] == saved_bp_state["OUTPUT_LO+1"]


def test_layer16_lev_pc_top_return_overrides_legacy_temp_overfire():
    from neural_vm.unified_compiler.ir import CompilerIR

    all_rules = list(_layer16_lev_routing_rules(100.0))
    rules_by_name = {rule.name: rule for rule in all_rules}
    rule = rules_by_name["l16_lev_pc_top_return_0a"]

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(
        rule
        for rule in all_rules
        if (
            rule.name.startswith("l16_lev_pc_temp_")
            or rule.name == "l16_lev_pc_top_return_0a"
        )
    )

    # Preserve the legacy prefix behavior: these rules still fire broadly
    # without a TEMP lane because OP_LEV + MARK_PC crosses threshold.
    assert rules_by_name["l16_lev_pc_temp_lo_10"].threshold == 3.5
    assert all_rules.index(rule) >= 121

    out = ir.symbolic_ffn({
        "CONST": 1.0,
        "OP_LEV": 5.0,
        "MARK_PC": 1.0,
        "HAS_SE": 0.99846,
        "H1+0": 1.0,
        "OUTPUT_LO+0": 7.0,
        "OUTPUT_LO+10": 5.0,
        "OUTPUT_HI+0": 5.0,
    })
    assert out["OUTPUT_LO+10"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+0"] > out["OUTPUT_HI+1"]


def test_layer16_lev_pc_top_return_materializes_0a_marker():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_lev_pc_top_return_0a"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("OP_LEV+0", 1.0) in condition_dims
    assert ("MARK_PC+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("H1+0", 1.0) in condition_dims
    assert ("IS_BYTE+0", -300.0) in condition_dims
    assert rule.threshold == 7.5

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    out = ir.symbolic_ffn({
        "OP_LEV": 5.0,
        "MARK_PC": 1.0,
        "HAS_SE": 0.99846,
        "H1+0": 1.0,
        "OUTPUT_LO+0": 7.0,
        "OUTPUT_LO+10": 5.0,
        "OUTPUT_HI+0": 5.0,
    })
    assert out["OUTPUT_LO+10"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+0"] > out["OUTPUT_HI+1"]

    out_byte = ir.symbolic_ffn({
        "OP_LEV": 5.0,
        "MARK_PC": 1.0,
        "HAS_SE": 1.0,
        "H1+0": 1.0,
        "IS_BYTE": 1.0,
        "OUTPUT_LO+0": 7.0,
    })
    assert out_byte["OUTPUT_LO+0"] == 7.0


def test_layer16_jsr_mem_addr0_materializes_f8_marker():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_jsr_mem_addr0_f8"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("OP_JSR+0", 1.0) in condition_dims
    assert ("OP_ENT+0", -10.0) in condition_dims
    assert ("MARK_MEM+0", 1.0) in condition_dims
    assert ("MEM_STORE+0", 1.0) in condition_dims
    assert ("HAS_SE+0", -20.0) in condition_dims
    assert ("IS_BYTE+0", -1_000_000_000_000.0) in condition_dims
    assert rule.threshold == 7.5

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+8"] == 20.0
    assert writes["OUTPUT_HI+15"] == 20.0
    assert writes["OUTPUT_LO+0"] == -20.0
    assert writes["OUTPUT_HI+14"] == -20.0
    assert writes["ALU_LO+14"] == -30.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    out = ir.symbolic_ffn({
        "OP_JSR": 5.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "OUTPUT_LO+0": 176.0,
        "OUTPUT_LO+8": 174.0,
        "OUTPUT_HI+0": 177.0,
        "OUTPUT_HI+15": 174.0,
        "ALU_LO+14": -2.0,
    })
    assert out["OUTPUT_LO+8"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+15"] > out["OUTPUT_HI+0"]
    assert out["ALU_LO+14"] < -20.0

    out_local_frame = ir.symbolic_ffn({
        "OP_JSR": 5.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 1.0,
        "OUTPUT_LO+0": 176.0,
    })
    assert out_local_frame["OUTPUT_LO+0"] == 176.0

    out_byte = ir.symbolic_ffn({
        "OP_JSR": 5.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "IS_BYTE": 1.0,
        "OUTPUT_LO+0": 176.0,
    })
    assert out_byte["OUTPUT_LO+0"] == 176.0

    out_sp = ir.symbolic_ffn({
        "OP_JSR": 5.0,
        "MARK_SP": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 1.0,
        "OUTPUT_LO+0": 176.0,
    })
    assert out_sp["OUTPUT_LO+0"] == 176.0

    out_ent = ir.symbolic_ffn({
        "OP_JSR": 5.0,
        "OP_ENT": 2.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 1.0,
        "OUTPUT_LO+0": 176.0,
    })
    assert out_ent["OUTPUT_LO+0"] == 176.0


def test_layer16_jsr_mem_addr0_materializes_e0_when_l14_evidence_wins():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_jsr_mem_addr0_e0_from_l14_evidence"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("OP_JSR+0", 1000.0) in condition_dims
    assert ("PSH_AT_SP+0", -100_000.0) in condition_dims
    assert ("MARK_MEM+0", 1.0) in condition_dims
    assert ("MEM_STORE+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("OUTPUT_LO+0", 1.0) in condition_dims
    assert ("OUTPUT_LO+8", -1.0) in condition_dims
    assert ("OUTPUT_HI+14", 1.0) in condition_dims
    assert ("OUTPUT_HI+15", -1.0) in condition_dims
    assert rule.threshold == 500.0
    assert rule.gate.key() == "HAS_SE+0"

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+0"] == 800.0
    assert writes["OUTPUT_HI+14"] == 800.0
    assert writes["OUTPUT_LO+8"] == -800.0
    assert writes["OUTPUT_HI+15"] == -800.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    out = ir.symbolic_ffn({
        "OP_JSR": 12.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.998,
        "OUTPUT_LO+0": 177.1,
        "OUTPUT_LO+8": 173.2,
        "OUTPUT_HI+14": 174.2,
        "OUTPUT_HI+15": 173.2,
    })
    assert out["OUTPUT_LO+0"] > out["OUTPUT_LO+8"]
    assert out["OUTPUT_HI+14"] > out["OUTPUT_HI+15"]

    out_initial_f8 = ir.symbolic_ffn({
        "OP_JSR": 12.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "OUTPUT_LO+0": 175.1,
        "OUTPUT_LO+8": 175.7,
        "OUTPUT_HI+14": 173.2,
        "OUTPUT_HI+15": 175.8,
    })
    assert out_initial_f8["OUTPUT_LO+0"] == 175.1
    assert out_initial_f8["OUTPUT_HI+14"] == 173.2


@pytest.mark.lowering
def test_layer16_jsr_mem_addr0_teacher_forced_f8_and_e0_paths():
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.unified_compiler.decl_verifier import (
        build_teacher_forced_symbolic_trace,
        verify_teacher_forced_token_support,
    )
    from src.compiler import compile_c

    probes = (
        (
            "initial_jsr_mem_addr0_f8",
            "int main() { int x; x = 990; return x; }\n",
            0,
            0xF8,
        ),
        (
            "local_frame_jsr_mem_addr0_e0",
            (
                "int identity(int x) { return x; }\n"
                "int main() { return identity(70); }\n"
            ),
            4,
            0xE0,
        ),
    )
    runner = BatchedPureNeuralRunner(max_seq_len=512)

    for name, source, step, expected in probes:
        bytecode, data = compile_c(source)
        trace = build_teacher_forced_symbolic_trace(bytecode, data)
        token_index = trace.token_index(step, "MEM_addr0")
        assert trace.context[token_index] == expected

        report = verify_teacher_forced_token_support(
            runner.model,
            trace.context,
            token_index=token_index,
            prefix_len=trace.prefix_len,
            mem_store_positions=trace.mem_store_positions,
            output_band_min_margin=0.01,
            probe_name=name,
        )
        assert report.supported, report.format()


@pytest.mark.lowering
def test_layer16_ent_mem_addr0_teacher_forced_f0_path():
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.unified_compiler.decl_verifier import (
        build_teacher_forced_symbolic_trace,
        verify_teacher_forced_token_support,
    )
    from src.compiler import compile_c

    bytecode, data = compile_c("int main() { int x; x = 990; return x; }\n")
    trace = build_teacher_forced_symbolic_trace(bytecode, data)
    token_index = trace.token_index(1, "MEM_addr0")
    assert trace.context[token_index] == 0xF0

    runner = BatchedPureNeuralRunner(max_seq_len=512)
    report = verify_teacher_forced_token_support(
        runner.model,
        trace.context,
        token_index=token_index,
        prefix_len=trace.prefix_len,
        mem_store_positions=trace.mem_store_positions,
        output_band_min_margin=0.01,
        probe_name="ent_store_mem_addr0_f0",
    )
    assert report.supported, report.format()


def test_layer16_ent_mem_addr0_clears_stale_alu_lo14():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_ent_mem_addr0_clear_stale_alu_lo14"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("OP_ENT+0", 1.0) in condition_dims
    assert ("MARK_MEM+0", 1.0) in condition_dims
    assert ("MEM_STORE+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("IS_BYTE+0", -20.0) in condition_dims
    assert rule.threshold == 3.5
    assert rule.gate_terms[0].dim.key() == "ALU_LO+14"
    assert rule.gate_terms[0].weight == -1.0
    assert rule.gate_bias == 0.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    out = ir.symbolic_ffn({
        "OP_ENT": 10.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 1.0,
        "ALU_LO+14": -85.0,
    })
    assert out["ALU_LO+14"] == 0.0

    out_byte = ir.symbolic_ffn({
        "OP_ENT": 10.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 1.0,
        "IS_BYTE": 1.0,
        "ALU_LO+14": -85.0,
    })
    assert out_byte["ALU_LO+14"] == -85.0

    out_sp = ir.symbolic_ffn({
        "OP_ENT": 10.0,
        "MARK_SP": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 1.0,
        "ALU_LO+14": -85.0,
    })
    assert out_sp["ALU_LO+14"] == -85.0


def test_layer16_psh_mem_addr0_restores_nonzero_l14_address_nibbles():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    lo = rules["l16_psh_mem_addr0_restore_lo_8"]
    hi = rules["l16_psh_mem_addr0_restore_hi_14"]
    force_d8 = rules["l16_psh_mem_addr0_force_d8_from_l14_evidence"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("PSH_AT_SP+0", 1.0) in condition_dims
    assert ("OP_ENT+0", -1000.0) in condition_dims
    assert ("MARK_MEM+0", 1.0) in condition_dims
    assert ("MEM_STORE+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 0.5) in condition_dims
    assert ("IS_BYTE+0", -1_000_000.0) in condition_dims
    assert ("OUTPUT_LO+8", 1.0) in condition_dims
    assert lo.threshold == 5.9
    hi_condition_dims = {(term.dim.key(), term.weight) for term in hi.conditions}
    assert ("OUTPUT_HI+14", 1.0) in hi_condition_dims
    assert hi.threshold == 5.5
    d8_condition_dims = {(term.dim.key(), term.weight) for term in force_d8.conditions}
    assert ("H1+4", 1.0) in d8_condition_dims
    assert ("OP_JSR+0", -1000.0) in d8_condition_dims
    assert ("OUTPUT_LO+8", 1.0) in d8_condition_dims
    assert ("OUTPUT_HI+13", 1.0) in d8_condition_dims
    assert force_d8.threshold == 8.0

    lo_writes = {write.dim.key(): write.weight for write in lo.writes}
    hi_writes = {write.dim.key(): write.weight for write in hi.writes}
    assert lo_writes["OUTPUT_LO+8"] == 10_000_000.0 / 100.0
    assert lo_writes["OUTPUT_LO+0"] == -10_000_000.0 / 100.0
    assert hi_writes["OUTPUT_HI+14"] == 10_000_000.0 / 100.0
    assert hi_writes["OUTPUT_HI+0"] == -10_000_000.0 / 100.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend((lo, hi))

    out = ir.symbolic_ffn({
        "PSH_AT_SP": 1.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "OUTPUT_LO+0": 2.75,
        "OUTPUT_LO+8": 1.0,
        "OUTPUT_HI+0": 2.75,
        "OUTPUT_HI+14": 1.0,
    })
    assert out["OUTPUT_LO+8"] > out["OUTPUT_LO+0"]
    assert out["OUTPUT_HI+14"] > out["OUTPUT_HI+0"]

    out_zero_nibble = ir.symbolic_ffn({
        "PSH_AT_SP": 1.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "OUTPUT_LO+0": 2.75,
        "OUTPUT_HI+0": 2.75,
    })
    assert out_zero_nibble["OUTPUT_LO+0"] == 2.75
    assert out_zero_nibble["OUTPUT_HI+0"] == 2.75

    out_weak_low8_residue = ir.symbolic_ffn({
        "PSH_AT_SP": 1.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "OUTPUT_LO+0": 2.75,
        "OUTPUT_LO+8": 0.53,
    })
    assert out_weak_low8_residue["OUTPUT_LO+0"] == 2.75
    assert out_weak_low8_residue["OUTPUT_LO+8"] == 0.53

    out_ent_marker = ir.symbolic_ffn({
        "PSH_AT_SP": 26.0,
        "OP_ENT": 10.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "OUTPUT_LO+0": 2.75,
        "OUTPUT_LO+8": 26.0,
    })
    assert out_ent_marker["OUTPUT_LO+0"] == 2.75
    assert out_ent_marker["OUTPUT_LO+8"] == 26.0

    out_ent_l14_evidence = ir.symbolic_ffn({
        "PSH_AT_SP": 0.0,
        "OP_ENT": 10.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "OUTPUT_LO+0": 147.0,
        "OUTPUT_LO+8": 130.0,
        "OUTPUT_HI+0": 132.0,
        "OUTPUT_HI+14": 135.0,
    })
    assert out_ent_l14_evidence["OUTPUT_LO+0"] == 147.0
    assert out_ent_l14_evidence["OUTPUT_LO+8"] == 130.0
    assert out_ent_l14_evidence["OUTPUT_HI+0"] == 132.0
    assert out_ent_l14_evidence["OUTPUT_HI+14"] == 135.0

    out_ax_marker_residue = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "OP_LEA": 5.0,
        "HAS_SE": 0.99,
        "OUTPUT_LO+0": 288.0,
        "OUTPUT_LO+8": 484.0,
        "OUTPUT_HI+0": 18_950.0,
        "OUTPUT_HI+14": 18_950.0,
    })
    assert out_ax_marker_residue["OUTPUT_LO+0"] == 288.0
    assert out_ax_marker_residue["OUTPUT_LO+8"] == 484.0
    assert out_ax_marker_residue["OUTPUT_HI+0"] == 18_950.0
    assert out_ax_marker_residue["OUTPUT_HI+14"] == 18_950.0

    out_ax_byte_row = ir.symbolic_ffn({
        "PSH_AT_SP": 1.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "IS_BYTE": 1.0,
        "H1+1": 1.0,
        "OUTPUT_HI+0": 9728.78,
        "OUTPUT_HI+14": 365.14,
    })
    assert out_ax_byte_row["OUTPUT_HI+0"] == 9728.78
    assert out_ax_byte_row["OUTPUT_HI+14"] == 365.14

    d8_ir = CompilerIR()
    d8_ir.layer(0).ffn.rules.extend((lo, hi, force_d8))
    out_deeper_stack_slot = d8_ir.symbolic_ffn({
        "PSH_AT_SP": 1.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "H1+4": 1.0,
        "OUTPUT_LO+0": 1.58,
        "OUTPUT_LO+8": 1.54,
        "OUTPUT_HI+0": 1.59,
        "OUTPUT_HI+13": 1.19,
        "OUTPUT_HI+14": 0.65,
        "OUTPUT_HI+15": 1.63,
    })
    assert out_deeper_stack_slot["OUTPUT_LO+8"] > out_deeper_stack_slot["OUTPUT_LO+0"]
    assert out_deeper_stack_slot["OUTPUT_HI+13"] > out_deeper_stack_slot["OUTPUT_HI+15"]
    assert out_deeper_stack_slot["OUTPUT_HI+13"] > out_deeper_stack_slot["OUTPUT_HI+14"]

    out_weaker_l14_d8_evidence = d8_ir.symbolic_ffn({
        "PSH_AT_SP": 1.498309850692749,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99821937084198,
        "H1+4": 1.0,
        "OUTPUT_LO+8": 0.996588945388794,
        "OUTPUT_HI+13": 1.1914863586425781,
        "OUTPUT_HI+15": 1.9761877059936523,
    })
    assert out_weaker_l14_d8_evidence["OUTPUT_LO+8"] > 1000.0
    assert out_weaker_l14_d8_evidence["OUTPUT_HI+13"] > out_weaker_l14_d8_evidence["OUTPUT_HI+15"]

    out_startup_negative_bands = d8_ir.symbolic_ffn({
        "OUTPUT_LO+8": -240.0,
        "OUTPUT_HI+13": -240.0,
        "OUTPUT_HI+14": -240.0,
        "OUTPUT_HI+15": -240.0,
    })
    assert out_startup_negative_bands["OUTPUT_HI+13"] == -240.0
    assert out_startup_negative_bands["OUTPUT_HI+15"] == -240.0

    out_jsr_mem_row = d8_ir.symbolic_ffn({
        "OP_JSR": 12.0,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "H1+4": 1.0,
        "OUTPUT_LO+8": 174.0,
        "OUTPUT_HI+13": 174.0,
        "OUTPUT_HI+15": 175.0,
    })
    assert out_jsr_mem_row["OUTPUT_HI+13"] == 174.0
    assert out_jsr_mem_row["OUTPUT_HI+15"] == 175.0

    out_e8_stack_slot = d8_ir.symbolic_ffn({
        "PSH_AT_SP": 1.49,
        "MARK_MEM": 1.0,
        "MEM_STORE": 2.0,
        "HAS_SE": 0.99,
        "OUTPUT_LO+0": 2.56,
        "OUTPUT_LO+8": 0.53,
        "OUTPUT_HI+0": 1.57,
        "OUTPUT_HI+14": 1.83,
        "OUTPUT_HI+15": 1.66,
    })
    assert out_e8_stack_slot["OUTPUT_HI+14"] > out_e8_stack_slot["OUTPUT_HI+15"]
    assert out_e8_stack_slot["OUTPUT_HI+14"] > out_e8_stack_slot["OUTPUT_HI+0"]

    assert "l16_psh_mem_addr0_restore_lo_11" not in rules


def test_layer16_ent_frame_sp_byte0_rules_use_relayed_frame_size():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    lo = rules["l16_ent_frame_sp_byte0_lo_0"]
    hi_frame16 = rules["l16_ent_frame_sp_byte0_hi_lo0_1"]
    hi_frame24 = rules["l16_ent_frame_sp_byte0_hi_lo8_1"]
    nested = rules["l16_ent_nested_sp_byte0_d8"]
    nested_bp = rules["l16_ent_nested_bp_byte0_d8"]
    nested_stack0 = rules["l16_ent_nested_stack0_saved_bp_byte0_f0"]
    stack0_byte1 = rules["l16_ent_stack0_saved_bp_byte1_ff"]
    initial_stack0_byte1 = rules["l16_ent_initial_stack0_saved_bp_byte1_00"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("MARK_SP+0", 10.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("OP_ENT+0", 0.2) in condition_dims
    assert ("FETCH_LO+0", 1.0) in condition_dims
    assert ("MARK_PC+0", -1_000_000.0) in condition_dims
    assert lo.threshold == 12.5
    assert lo.gate.key() == "FETCH_LO+0"

    lo_writes = {write.dim.key(): write.weight for write in lo.writes}
    assert lo_writes["OUTPUT_LO+0"] == 5000.0 / 100.0
    assert lo_writes["OUTPUT_LO+8"] == -5000.0 / 100.0

    hi16_writes = {write.dim.key(): write.weight for write in hi_frame16.writes}
    hi24_writes = {write.dim.key(): write.weight for write in hi_frame24.writes}
    assert hi_frame16.threshold == 13.5
    assert hi_frame24.threshold == 13.5
    assert hi16_writes["OUTPUT_HI+14"] == 5000.0 / 100.0
    assert hi24_writes["OUTPUT_HI+13"] == 5000.0 / 100.0
    assert hi16_writes["OUTPUT_HI+1"] == -5000.0 / 100.0

    nested_conditions = {(term.dim.key(), term.weight) for term in nested.conditions}
    assert ("OP_ENT+0", 0.2) in nested_conditions
    assert ("OP_ENT+0", 9.8) in nested_conditions
    assert ("MARK_SP+0", 10.0) in nested_conditions
    assert ("FETCH_LO+0", 1.0) in nested_conditions
    assert ("FETCH_HI+0", 1.0) in nested_conditions
    assert ("OUTPUT_HI+15", 1.0) in nested_conditions
    assert nested.threshold == 70.0

    nested_writes = {write.dim.key(): write.weight for write in nested.writes}
    assert nested_writes["OUTPUT_LO+8"] == 10.0
    assert nested_writes["OUTPUT_HI+13"] == 10.0
    assert nested_writes["OUTPUT_LO+0"] == -10.0
    assert nested_writes["OUTPUT_HI+15"] == -10.0

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(nested)

    nested_state = {
        "OP_ENT": 6.5,
        "MARK_SP": 1.0,
        "HAS_SE": 0.998,
        "FETCH_LO+0": 1.0,
        "FETCH_HI+0": 1.0,
        "OUTPUT_LO+8": 0.12,
        "OUTPUT_HI+15": 0.0,
        "OUTPUT_LO+0": -5.7,
    }
    out_nested = ir.symbolic_ffn(nested_state)
    assert out_nested["OUTPUT_LO+8"] > nested_state["OUTPUT_LO+8"]
    assert out_nested["OUTPUT_HI+13"] > 0.0
    assert out_nested["OUTPUT_HI+15"] < nested_state["OUTPUT_HI+15"]

    initial_state = dict(nested_state, **{"OUTPUT_LO+8": 8.8, "OUTPUT_HI+15": -13.0})
    out_initial = ir.symbolic_ffn(initial_state)
    assert out_initial["OUTPUT_LO+8"] == initial_state["OUTPUT_LO+8"]

    ax_marker_state = {
        "OP_ENT": 0.116,
        "MARK_AX": 1.0,
        "HAS_SE": 0.998,
        "FETCH_LO+0": 1.0,
        "FETCH_HI+0": 1.0,
        "OUTPUT_LO+8": 484.0,
        "OUTPUT_HI+15": 18_950.0,
        "OUTPUT_HI+13": 18_950.0,
    }
    out_ax_marker = ir.symbolic_ffn(ax_marker_state)
    assert out_ax_marker["OUTPUT_LO+8"] == ax_marker_state["OUTPUT_LO+8"]
    assert out_ax_marker["OUTPUT_HI+13"] == ax_marker_state["OUTPUT_HI+13"]
    assert out_ax_marker["OUTPUT_HI+15"] == ax_marker_state["OUTPUT_HI+15"]

    bp_conditions = {(term.dim.key(), term.weight) for term in nested_bp.conditions}
    assert ("OP_ENT+0", 100.0) in bp_conditions
    assert ("MARK_BP+0", 20000.0) in bp_conditions
    assert ("OUTPUT_HI+15", -50.0) in bp_conditions
    assert nested_bp.threshold == 20250.0

    ir_bp = CompilerIR()
    ir_bp.layer(0).ffn.rules.append(nested_bp)

    nested_bp_state = {
        "OP_ENT": 8.4,
        "MARK_BP": 1.0,
        "HAS_SE": 0.998,
        "OUTPUT_LO+0": 8.9,
        "OUTPUT_LO+8": -5.0,
        "OUTPUT_HI+15": 9.0,
    }
    out_nested_bp = ir_bp.symbolic_ffn(nested_bp_state)
    assert out_nested_bp["OUTPUT_LO+8"] > nested_bp_state["OUTPUT_LO+8"]
    assert out_nested_bp["OUTPUT_HI+13"] > 0.0
    assert out_nested_bp["OUTPUT_HI+15"] < nested_bp_state["OUTPUT_HI+15"]

    initial_bp_state = dict(nested_bp_state, **{"OUTPUT_HI+15": 15.0})
    out_initial_bp = ir_bp.symbolic_ffn(initial_bp_state)
    assert out_initial_bp["OUTPUT_LO+8"] == nested_bp_state["OUTPUT_LO+8"]

    stack0_conditions = {
        (term.dim.key(), term.weight) for term in nested_stack0.conditions
    }
    assert ("OP_ENT+0", 10.0) in stack0_conditions
    assert ("MARK_STACK0+0", 10.0) in stack0_conditions
    assert ("ADDR_B0_LO+8", -1.0) in stack0_conditions
    assert ("ADDR_B0_HI+14", -1.0) in stack0_conditions
    assert nested_stack0.threshold == 80.0

    ir_stack0 = CompilerIR()
    ir_stack0.layer(0).ffn.rules.append(nested_stack0)
    nested_stack0_state = {
        "OP_ENT": 8.5,
        "MARK_STACK0": 1.0,
        "HAS_SE": 0.998,
        "MEM_STORE": 0.4,
        "ADDR_B0_LO+8": -0.6,
        "ADDR_B0_HI+14": -19.8,
        "OUTPUT_LO+0": 15.8,
        "OUTPUT_HI+0": 15.2,
        "OUTPUT_HI+15": 0.7,
    }
    out_nested_stack0 = ir_stack0.symbolic_ffn(nested_stack0_state)
    assert out_nested_stack0["OUTPUT_HI+15"] > nested_stack0_state["OUTPUT_HI+15"]
    assert out_nested_stack0["OUTPUT_HI+0"] < nested_stack0_state["OUTPUT_HI+0"]

    initial_stack0_state = dict(
        nested_stack0_state,
        **{"ADDR_B0_LO+8": 24.6, "ADDR_B0_HI+14": 20.2},
    )
    out_initial_stack0 = ir_stack0.symbolic_ffn(initial_stack0_state)
    assert out_initial_stack0["OUTPUT_HI+15"] == nested_stack0_state["OUTPUT_HI+15"]

    byte1_conditions = {(term.dim.key(), term.weight) for term in stack0_byte1.conditions}
    assert ("IS_BYTE+0", 1.0) in byte1_conditions
    assert ("OP_ENT+0", 1.0) in byte1_conditions
    assert ("STACK0_BYTE0+0", 30.0) in byte1_conditions
    assert ("CLEAN_EMBED_LO+0", 1.0) in byte1_conditions
    assert ("BYTE_INDEX_1+0", -10.0) in byte1_conditions
    assert ("BYTE_INDEX_2+0", -10.0) in byte1_conditions
    assert ("BYTE_INDEX_3+0", -10.0) in byte1_conditions
    assert stack0_byte1.gate.key() == "CLEAN_EMBED_HI+15"
    assert stack0_byte1.threshold == 35.0
    initial_byte1_conditions = {
        (term.dim.key(), term.weight) for term in initial_stack0_byte1.conditions
    }
    assert ("IS_BYTE+0", 1.0) in initial_byte1_conditions
    assert ("OP_ENT+0", 1.0) in initial_byte1_conditions
    assert ("STACK0_BYTE0+0", 30.0) in initial_byte1_conditions
    assert ("CLEAN_EMBED_LO+0", 1.0) in initial_byte1_conditions
    assert ("BYTE_INDEX_1+0", -10.0) in initial_byte1_conditions
    assert ("BYTE_INDEX_2+0", -10.0) in initial_byte1_conditions
    assert ("BYTE_INDEX_3+0", -10.0) in initial_byte1_conditions
    assert initial_stack0_byte1.gate.key() == "CLEAN_EMBED_HI+0"
    assert initial_stack0_byte1.threshold == 35.0

    ir_byte1 = CompilerIR()
    ir_byte1.layer(0).ffn.rules.append(stack0_byte1)
    byte1_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 0.998,
        "OP_ENT": 8.5,
        "STACK0_BYTE0": 0.97,
        "BYTE_INDEX_0": 0.97,
        "CLEAN_EMBED_LO+0": 1.0,
        "CLEAN_EMBED_HI+15": 1.0,
        "OUTPUT_LO+0": 3.9,
        "OUTPUT_HI+0": 42.9,
    }
    out_byte1 = ir_byte1.symbolic_ffn(byte1_state)
    assert out_byte1["OUTPUT_LO+15"] > 0.0
    assert out_byte1["OUTPUT_HI+15"] > 0.0
    assert out_byte1["OUTPUT_LO+0"] < byte1_state["OUTPUT_LO+0"]
    assert out_byte1["OUTPUT_HI+0"] < byte1_state["OUTPUT_HI+0"]

    out_byte1_initial = ir_byte1.symbolic_ffn(dict(
        byte1_state,
        **{"CLEAN_EMBED_HI+15": 0.0, "CLEAN_EMBED_HI+0": 1.0},
    ))
    assert out_byte1_initial["OUTPUT_LO+15"] == 0.0
    out_sp_byte2 = ir_byte1.symbolic_ffn(dict(
        byte1_state,
        **{
            "STACK0_BYTE0": 0.0,
            "BYTE_INDEX_0": 0.0,
            "BYTE_INDEX_1": 0.97,
            "CLEAN_EMBED_LO+0": 0.0,
            "CLEAN_EMBED_LO+15": 1.0,
            "CLEAN_EMBED_HI+15": 1.0,
        },
    ))
    assert out_sp_byte2.get("OUTPUT_LO+15", 0.0) == 0.0

    ir_initial_byte1 = CompilerIR()
    ir_initial_byte1.layer(0).ffn.rules.append(initial_stack0_byte1)
    initial_byte1_state = dict(
        byte1_state,
        **{
            "CLEAN_EMBED_HI+15": 0.0,
            "CLEAN_EMBED_HI+0": 1.0,
            "OUTPUT_LO+15": 3.3,
            "OUTPUT_HI+15": 4.4,
        },
    )
    out_initial_byte1 = ir_initial_byte1.symbolic_ffn(initial_byte1_state)
    assert out_initial_byte1["OUTPUT_LO+0"] > byte1_state["OUTPUT_LO+0"]
    assert out_initial_byte1["OUTPUT_HI+0"] > byte1_state["OUTPUT_HI+0"]
    assert out_initial_byte1["OUTPUT_LO+15"] < initial_byte1_state["OUTPUT_LO+15"]
    assert out_initial_byte1["OUTPUT_HI+15"] < initial_byte1_state["OUTPUT_HI+15"]

    out_recursive_initial_byte1 = ir_initial_byte1.symbolic_ffn(dict(
        initial_byte1_state,
        **{"CLEAN_EMBED_HI+0": 0.0, "CLEAN_EMBED_HI+15": 1.0},
    ))
    assert out_recursive_initial_byte1["OUTPUT_LO+0"] == byte1_state["OUTPUT_LO+0"]

    psh_stack0_byte1_state = {
        "IS_BYTE": 1.0,
        "HAS_SE": 0.9970132112503052,
        "STACK0_BYTE0": 0.9734055995941162,
        "BYTE_INDEX_0": 0.9734055995941162,
        "BYTE_INDEX_1": 0.013297064229846,
        "CLEAN_EMBED_LO+15": 1.0,
        "CLEAN_EMBED_HI+0": 1.0,
        "OUTPUT_LO+15": 4.893622875213623,
        "OUTPUT_LO+0": 2.053187847137451,
        "OUTPUT_HI+0": 5.053187847137451,
        "OUTPUT_HI+15": 1.8936229944229126,
    }
    out_psh_stack0_byte1 = ir_initial_byte1.symbolic_ffn(psh_stack0_byte1_state)
    assert out_psh_stack0_byte1["OUTPUT_LO+15"] == psh_stack0_byte1_state["OUTPUT_LO+15"]
    assert out_psh_stack0_byte1["OUTPUT_HI+15"] == psh_stack0_byte1_state["OUTPUT_HI+15"]
    assert out_psh_stack0_byte1["OUTPUT_LO+0"] == psh_stack0_byte1_state["OUTPUT_LO+0"]


def test_layer16_lea_local_ax_byte0_high_nibble_nudges_e8_marker():
    from neural_vm.unified_compiler.ir import CompilerIR

    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}
    rule = rules["l16_lea_local_ax_byte0_hi_e"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("MARK_AX+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("OP_LEA+0", 1.0) in condition_dims
    assert ("CMP+7", 1.0) in condition_dims
    assert ("FETCH_LO+8", 0.2) in condition_dims
    assert ("FETCH_HI+15", 0.2) in condition_dims
    assert ("IS_BYTE+0", -10.0) in condition_dims
    assert rule.threshold == 8.0
    assert rule.writes[0].dim.key() == "OUTPUT_HI+14"
    assert rule.writes[0].weight == 0.1

    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)

    out = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "HAS_SE": 1.0,
        "OP_LEA": 5.0,
        "CMP+7": 1.0,
        "FETCH_LO+8": 1.0,
        "FETCH_HI+15": 1.0,
    })
    assert out["OUTPUT_HI+14"] > 0.0

    out_byte_row = ir.symbolic_ffn({
        "MARK_AX": 1.0,
        "HAS_SE": 1.0,
        "OP_LEA": 5.0,
        "CMP+7": 1.0,
        "FETCH_LO+8": 1.0,
        "FETCH_HI+15": 1.0,
        "IS_BYTE": 1.0,
    })
    assert out_byte_row.get("OUTPUT_HI+14", 0.0) == 0.0

    out_pc_marker = ir.symbolic_ffn({
        "MARK_PC": 1.0,
        "HAS_SE": 1.0,
        "FETCH_LO+8": 40.0,
        "FETCH_HI+15": 0.0001,
    })
    assert out_pc_marker.get("OUTPUT_HI+14", 0.0) == 0.0


def test_layer16_nonstore_mem_value_zero_blocks_mem_store_residue():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    rule = rules["l16_nonstore_mem_value0_zero"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("IS_BYTE+0", 1.0) in condition_dims
    assert ("H3+4", 1.0) in condition_dims
    assert ("MEM_VAL_B0+0", 1.0) in condition_dims
    assert ("MEM_STORE+0", -100.0) in condition_dims
    assert ("MARK_MEM+0", -100.0) in condition_dims
    assert rule.threshold == 2.5

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+0"] == 100.0 / 100.0
    assert writes["OUTPUT_HI+0"] == 100.0 / 100.0
    assert writes["OUTPUT_LO+8"] == -100.0 / 100.0
    assert writes["OUTPUT_HI+14"] == -100.0 / 100.0


def test_layer16_psh_sp_no_borrow_high_restore_uses_positive_low_gate():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    rule = rules["l16_psh_sp_no_borrow_hi_14"]

    condition_dims = {(term.dim.key(), term.weight) for term in rule.conditions}
    assert ("PSH_AT_SP+0", 1.0) in condition_dims
    assert ("MARK_SP+0", 1.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("IS_BYTE+0", -10.0) in condition_dims
    assert ("EMBED_HI+14", 1.0) in condition_dims
    assert rule.threshold == 4.5
    assert rule.gate is None
    assert {term.weight for term in rule.gate_terms} == {1.0}
    assert {term.dim.key() for term in rule.gate_terms} == {
        f"EMBED_LO+{k}" for k in range(8, 16)
    }

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_HI+14"] == 4.0 / 100.0
    assert writes["OUTPUT_HI+13"] == -4.0 / 100.0
