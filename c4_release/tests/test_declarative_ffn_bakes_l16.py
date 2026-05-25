"""Parity tests for L16 FFN bands migrated to declarative rules."""

import torch

from neural_vm.unified_compiler.ops.l16_ops import (
    _layer16_lev_routing_rules,
    lower_layer16_lev_routing_ir,
)
from neural_vm.vm_step import _SetDim, _set_layer16_lev_routing


class _StubFFN:
    def __init__(self, *, d_model: int = 512, hidden_dim: int = 320):
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
    assert end == 311
    assert len(_layer16_lev_routing_rules(100.0)) == 311
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

    writes = {write.dim.key(): write.weight for write in rule.writes}
    assert writes["OUTPUT_LO+0"] == 1000.0 / 100.0
    assert writes["OUTPUT_LO+1"] == -1000.0 / 100.0
    assert writes["OUTPUT_HI+0"] == 1000.0 / 100.0
    assert writes["OUTPUT_HI+1"] == -1000.0 / 100.0


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


def test_layer16_ent_frame_sp_byte0_rules_use_relayed_frame_size():
    rules = {rule.name: rule for rule in _layer16_lev_routing_rules(100.0)}

    lo = rules["l16_ent_frame_sp_byte0_lo_0"]
    hi_frame16 = rules["l16_ent_frame_sp_byte0_hi_lo0_1"]
    hi_frame24 = rules["l16_ent_frame_sp_byte0_hi_lo8_1"]

    condition_dims = {(term.dim.key(), term.weight) for term in lo.conditions}
    assert ("MARK_SP+0", 10.0) in condition_dims
    assert ("HAS_SE+0", 1.0) in condition_dims
    assert ("OP_ENT+0", 0.2) in condition_dims
    assert ("FETCH_LO+0", 1.0) in condition_dims
    assert ("MARK_PC+0", -1000.0) in condition_dims
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
