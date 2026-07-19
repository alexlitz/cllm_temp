"""Regression suite for the COMPACT whole-VM -> vanilla ONNX export (CHK-2).

Proves, on a small (LEAN, code_size=16) compact model so the tests are fast:

  1. the KV-cached windowed forward exports to ONNX and the graph is VANILLA
     (no Loop/Scan/If, no custom/external-memory ops, only standard-transformer
     ops),
  2. the sparse-initializer ONNX is far smaller than the dense one and stays
     vanilla + loadable,
  3. onnxruntime matches the torch ``forward_hidden_cached`` (per-step hidden
     argmax identical; fp residue only),
  4. driving ``run_pure_forward_cached`` with the ONNX model produces a
     byte-identical emitted trace to the torch model on a per-op battery
     (add / mul / cmp / memory / function / loop).
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa
from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min.export_onnx_compact import (
    CachedStepModule, export_cached_onnx, to_sparse_onnx, op_inventory,
    assert_vanilla, OnnxCachedModel, FORBIDDEN_OPS, VANILLA_OPS, _io_names,
    _example_inputs,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def compact_lean():
    # code_size=64 matches the production export config and is large enough for
    # the default value-liveness probe battery (whose compiled C sources exceed
    # 16 instructions).  bitwise=False keeps the LEAN block stack small/fast.
    model, L, stats = build_compact_pure_forward_model(
        code_size=64)
    model.eval()
    return model, L, stats


@pytest.fixture(scope="module")
def onnx_paths(compact_lean, tmp_path_factory):
    model, _L, _s = compact_lean
    d = tmp_path_factory.mktemp("onnx")
    dense = str(d / "vm.onnx")
    sparse = str(d / "vm_sparse.onnx")
    export_cached_onnx(model, dense)
    stats = to_sparse_onnx(dense, sparse)
    return dense, sparse, stats


def test_graph_is_vanilla(onnx_paths):
    dense, sparse, _ = onnx_paths
    for path in (dense, sparse):
        ok, notes = assert_vanilla(path)
        assert ok, f"{path} not vanilla: {notes}"
        inv = op_inventory(path)
        assert not (FORBIDDEN_OPS & set(inv)), f"forbidden ops: {inv}"
        assert set(inv) <= VANILLA_OPS, f"unknown ops: {set(inv) - VANILLA_OPS}"


def test_has_expected_transformer_ops(onnx_paths):
    _dense, sparse, _ = onnx_paths
    inv = op_inventory(sparse)
    # attention Q/K/V/O + 2 attn matmuls + FFN up/gate/down = 9 MatMul per block.
    assert inv["MatMul"] > 0
    # softmax1 primitives + SwiGLU gate present.
    for op in ("Exp", "ReduceMax", "ReduceSum", "Div", "Sigmoid", "Mul"):
        assert inv[op] > 0, f"missing softmax1/SwiGLU op {op}"
    # ALiBi bias plumbing.
    for op in ("Sub", "Abs", "Neg"):
        assert inv[op] > 0, f"missing ALiBi op {op}"


def test_sparse_is_smaller(onnx_paths):
    _dense, _sparse, stats = onnx_paths
    assert stats["sparse_file_bytes"] < stats["dense_file_bytes"]
    assert stats["tensors_sparsified"] >= 1


def test_ort_matches_torch_single_forward(compact_lean, onnx_paths):
    model, _L, _s = compact_lean
    _dense, sparse, _ = onnx_paths
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(sparse, so, providers=["CPUExecutionProvider"])
    args = _example_inputs(model, W=4, Sc=3)
    step = CachedStepModule(model).eval()
    with torch.no_grad():
        tref = step(*args)
    in_names, out_names = _io_names(len(model.blocks))
    feed = {}
    for name, a in zip(in_names, args):
        arr = a.detach().cpu().numpy()
        feed[name] = arr.astype(np.int64) if a.dtype == torch.long else arr.astype(np.float32)
    outs = sess.run(out_names, feed)
    # hidden argmax identical (the decode-relevant check); fp residue on values.
    assert np.array_equal(tref[0].numpy().argmax(-1), outs[0].argmax(-1))


def _battery():
    def I(op, imm=0):
        return isa.Instr(op, imm)
    return [
        ("add", [I(isa.IMM, 5), I(isa.PSH), I(isa.IMM, 3), I(isa.ADD), I(isa.HALT)], 20, 0xFF),
        ("mul", [I(isa.IMM, 6), I(isa.PSH), I(isa.IMM, 7), I(isa.MUL), I(isa.HALT)], 20, 0xFF),
        ("lt",  [I(isa.IMM, 3), I(isa.PSH), I(isa.IMM, 5), I(isa.LT), I(isa.HALT)], 20, 0xFF),
        ("si_li", [I(isa.IMM, 7), I(isa.PSH), I(isa.IMM, 200), I(isa.PSH),
                   I(isa.IMM, 7), I(isa.SI), I(isa.IMM, 200), I(isa.LI), I(isa.HALT)], 30, 0xFF),
        ("func", [I(isa.JSR, 3), I(isa.HALT), I(isa.NOP),
                  I(isa.ENT, 0), I(isa.IMM, 42), I(isa.LEV)], 30, 0xFF),
        ("loop", [I(isa.IMM, 8), I(isa.PSH), I(isa.IMM, 1), I(isa.SUB),
                  I(isa.BNZ, 1), I(isa.HALT)], 200, 0xFF),
    ]


@pytest.mark.parametrize("name,code,msteps,mask", _battery())
def test_onnx_trace_byte_identical_to_torch(compact_lean, onnx_paths, name,
                                            code, msteps, mask):
    model, L, _s = compact_lean
    _dense, sparse, _ = onnx_paths
    tr_t = run_pure_forward_cached(model, L, code, max_steps=msteps, mask=mask,
                                   evict=True, prune_interval=120)
    onnx_model = OnnxCachedModel(model, sparse)
    tr_o = run_pure_forward_cached(onnx_model, L, code, max_steps=msteps,
                                   mask=mask, evict=True, prune_interval=120)
    assert tr_t == tr_o, f"{name}: torch {tr_t} != onnx {tr_o}"
