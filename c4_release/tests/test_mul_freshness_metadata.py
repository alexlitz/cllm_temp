"""Focused regression tests for MUL L11/L12 freshness metadata."""

import torch

from c4_release.neural_vm.setup_helpers import (
    _set_layer11_mul_partial,
    _set_layer12_mul_combine,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import LayerCompiler
from c4_release.neural_vm.unified_compiler.migrated_ops import (
    all_core_ops,
    declare_setdim_compat_dims,
    make_layer11_mul_partial_op,
    make_layer12_mul_combine_op,
)
from c4_release.neural_vm.vm_step import _SetDim


class _FakeFFN:
    def __init__(self, dim=512, hidden_dim=4096):
        self.W_up = torch.zeros(hidden_dim, dim)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, dim)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(dim, hidden_dim)


def _nonzero_rows(matrix):
    return set(
        torch.nonzero(
            matrix.abs().sum(dim=1) > 0, as_tuple=False
        ).flatten().tolist()
    )


def _nonzero_cols(matrix):
    return set(
        torch.nonzero(
            matrix.abs().sum(dim=0) > 0, as_tuple=False
        ).flatten().tolist()
    )


def test_l11_l12_mul_freshness_metadata_uses_real_temp_staging_lane():
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)

    layout = compiler.compile()
    assert layout.dim_positions["TEMP"] != layout.dim_positions["MUL_ACCUM"]

    producers, consumers = compiler.build_staleness_registry()
    assert producers[("TEMP", "AX_byte0")] == [("layer11_mul_partial", 11)]
    assert ("layer12_mul_combine", 12) in consumers[("TEMP", "AX_byte0")]
    assert ("MUL_ACCUM", "AX_byte0") not in consumers


def test_efficient_noop_mul_lookup_ops_do_not_claim_temp_freshness():
    l11 = make_layer11_mul_partial_op(alu_mode="efficient")
    l12 = make_layer12_mul_combine_op(alu_mode="efficient")

    assert l11.produces == {}
    assert l11.consumes_fresh == {}
    assert l12.consumes_fresh == {}


def test_lookup_mul_helpers_stage_and_consume_temp_not_mul_accum():
    ffn11 = _FakeFFN()
    assert _set_layer11_mul_partial(ffn11, 100.0, _SetDim) == 4096

    temp_rows = set(range(_SetDim.TEMP, _SetDim.TEMP + 16))
    mul_accum_rows = set(range(_SetDim.MUL_ACCUM, _SetDim.MUL_ACCUM + 16))
    assert _nonzero_rows(ffn11.W_down) == temp_rows
    assert _nonzero_rows(ffn11.W_down).isdisjoint(mul_accum_rows)

    ffn12 = _FakeFFN()
    assert _set_layer12_mul_combine(ffn12, 100.0, _SetDim) == 4096

    up_cols = _nonzero_cols(ffn12.W_up)
    assert temp_rows.issubset(up_cols)
    assert up_cols.isdisjoint(mul_accum_rows)
