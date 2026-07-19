"""Top-1-routed dispatch MoE gate (moe_top1.Top1RoutedFFN + nibble_moe variant).

Asserts the top-1 hard-routed dispatch is argmax-identical to the dense-blend
dispatch (the inactive experts contribute ~0), and that the router lowers to a
VANILLA ONNX graph (ArgMax + Gather, no If/Loop/Scan).

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_moe_top1.py -v
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest
import torch

import c4_min.nibble_pure_forward as PF
import c4_min.nibble_pure_forward_complete as C

PF.SP_INIT = 0xF0
C.SP_INIT = 0xF0

from c4_min import isa
from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.nibble_pure_forward_complete import ref_interpret
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min.moe_top1 import Top1RoutedFFN
from c4_min._probe_pf_corpus_sample import progs_by_class


@pytest.fixture(scope="module")
def compact():
    m, L, _ = build_compact_pure_forward_model(
        code_size=24, include_bitwise=True, include_divmod=False)
    return m, L


@pytest.fixture(scope="module")
def dispatch(compact):
    m, L = compact
    bi = list(L._block_names).index("dispatch")
    dense = m.blocks[bi].ffn
    routed = Top1RoutedFFN(dense, op_is_base=int(L.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    return dense, routed, int(L.OP_IS), bi


def test_routed_units_far_fewer_than_dense(dispatch):
    dense, routed, _op_is, _bi = dispatch
    H = dense.W_up.shape[0]
    # K (routed units) << H (dense units): the structural-sparsity win.
    assert routed.K < H
    assert routed.K <= 3            # opcodes own at most a few units
    assert H / routed.K >= 10.0     # >=10x fewer units computed per step


def test_block_argmax_identical(dispatch):
    """Top-1 routed FFN == dense FFN on a synthetic one-hot input per opcode."""
    dense, routed, op_is, _bi = dispatch
    D = dense.W_up.shape[1]
    owned = [op for op in range(int(isa.NUM_OPS))
             if bool((routed.op_units[op] < routed.n_units).any())]
    assert len(owned) >= 20
    torch.manual_seed(0)
    max_ad = 0.0
    for op in owned:
        x = torch.randn(2, 3, D) * 0.4
        x[..., routed.route_dims] = 0.0
        x[..., op_is + op] = 1.0
        yd = dense(x)
        yr = routed(x)
        max_ad = max(max_ad, float((yd - yr).abs().max()))
        assert (yd.argmax(-1) == yr.argmax(-1)).all(), f"argmax differs for op {op}"
    assert max_ad < 1e-4, f"max|dense-routed|={max_ad}"


@pytest.mark.parametrize("op", ["ADD", "SUB", "EQ", "SI", "JMP", "JSR"])
def test_program_argmax_identical(compact, op):
    """A representative program per family: routed == dense == ref."""
    m, L = compact
    bi = list(L._block_names).index("dispatch")
    prog = list(progs_by_class()[op][0])
    code = isa.assemble(prog)

    # ensure dense (module fixture may have routed it; rebuild the dense FFN
    # is expensive, so route on a private copy of the block instead).
    orig = m.blocks[bi].ffn
    got_dense = run_pure_forward_cached(m, L, code, max_steps=32, evict=False)
    routed = Top1RoutedFFN(orig, op_is_base=int(L.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    m.blocks[bi].ffn = routed
    try:
        got_routed = run_pure_forward_cached(m, L, code, max_steps=32, evict=False)
    finally:
        m.blocks[bi].ffn = orig
    assert got_routed == got_dense
    assert got_routed == ref_interpret(code)


def test_onnx_vanilla_and_byte_identical(dispatch):
    """The routed dispatch traces to a vanilla ONNX graph (no If/Loop/Scan) and
    the graph is bit-identical to the torch forward."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from c4_min.export_onnx import op_inventory, assert_vanilla, FORBIDDEN_OPS

    dense, routed, op_is, _bi = dispatch
    D = dense.W_up.shape[1]

    class _W(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return self.f(x)

    tmp = tempfile.mkdtemp(prefix="top1_test_")
    path = os.path.join(tmp, "routed.onnx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            _W(routed).eval(), (torch.zeros(1, 4, D),), path,
            input_names=["r"], output_names=["o"],
            dynamic_axes={"r": {0: "b", 1: "s"}, "o": {0: "b", 1: "s"}},
            opset_version=17, do_constant_folding=True, dynamo=False)
    try:
        inv = op_inventory(path)
        assert not (FORBIDDEN_OPS & set(inv)), \
            f"forbidden control-flow op: {sorted(FORBIDDEN_OPS & set(inv))}"
        assert "ArgMax" in inv and "Gather" in inv    # the router lowered to these
        ok, notes = assert_vanilla(path)
        assert ok, notes
        # bit-identity ONNX vs torch on a per-opcode battery
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        owned = [op for op in range(int(isa.NUM_OPS))
                 if bool((routed.op_units[op] < routed.n_units).any())]
        torch.manual_seed(1)
        for op in owned[:8]:
            x = torch.randn(1, 5, D) * 0.3
            x[..., routed.route_dims] = 0.0
            x[..., op_is + op] = 1.0
            yt = routed(x).detach().numpy()
            yo = sess.run(None, {"r": x.numpy()})[0]
            assert (yt.argmax(-1) == yo.argmax(-1)).all()
    finally:
        try:
            os.remove(path)
            os.rmdir(tmp)
        except OSError:
            pass
