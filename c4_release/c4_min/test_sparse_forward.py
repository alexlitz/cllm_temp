"""STEP 1 gate: the sparse-tensor forward is a storage/compute win AND is
byte-identical (dense_kernel, L-inf=0) / argmax-decode-identical (sparse_mm) to
the dense pure-forward model.

Uses the LEAN model (38 blocks) for CI speed; the same wrapper applies to the
bitwise (42-block, 30 GB dense) and divmod (304-block, 54 GB dense) configs,
where the storage win is the point (they finally fit in VRAM).

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_sparse_forward.py -v
"""
from __future__ import annotations

import pytest
import torch

from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete,
    ref_interpret, make_overlay_complete, _build_frame, SP_INIT)
from c4_min.sparse_forward import SparseTransformer
from c4_min import blogspec_vocab as V


@pytest.fixture(scope="module")
def dense_model():
    return build_pure_forward_complete_model(
        code_size=44)


def _stream(nframes):
    stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    for _ in range(nframes):
        stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
    return stream


def test_storage_is_far_smaller(dense_model):
    model, _L = dense_model
    sp = SparseTransformer(model, compute_mode="sparse_mm")
    st = sp.stats()
    # sparse storage must be a large fraction smaller than the dense-equivalent.
    assert st.sparse_bytes < st.dense_bytes / 10, (st.sparse_mb, st.dense_gb)
    assert st.n_sparsified > 0


def test_dense_kernel_is_bit_identical(dense_model):
    """dense_kernel mode: L-inf = 0 vs the dense block stack (same GEMM)."""
    model, L = dense_model
    sp = SparseTransformer(model, compute_mode="dense_kernel")
    toks = torch.tensor([_stream(4)])
    overlay = make_overlay_complete([], L)
    with torch.no_grad():
        xd = model.embed[toks].clone(); overlay(xd)
        for blk in model.blocks:
            xd = blk(xd)
        xs = sp.embed[toks].clone(); overlay(xs)
        for blk in sp.blocks:
            xs = blk(xs)
    assert (xd - xs).abs().max().item() == 0.0


def test_sparse_mm_residue_is_fp_order_only(dense_model):
    """sparse_mm mode: any residual is small relative to the values (fp accum
    order, not a bug) — a sanity bound, not L-inf=0."""
    model, L = dense_model
    sp = SparseTransformer(model, compute_mode="sparse_mm")
    toks = torch.tensor([_stream(4)])
    overlay = make_overlay_complete([], L)
    with torch.no_grad():
        xd = model.embed[toks].clone(); overlay(xd)
        for blk in model.blocks:
            xd = blk(xd)
        xs = sp.embed[toks].clone(); overlay(xs)
        for blk in sp.blocks:
            xs = blk(xs)
    diff = (xd - xs).abs().max().item()
    scale = xd.abs().max().item()
    # the residue is a tiny FRACTION of the (large) VM band magnitudes.
    assert diff <= 1e-3 * scale, (diff, scale)


_BATTERY = [
    ("add", "int main(){ return 500 + 700; }", 1200),
    ("mul", "int main(){ return 100 * 10; }", 1000),
    ("cmp", "int main(){ if (5 > 3) return 1; return 0; }", 1),
    ("var", "int main(){ int x; x = 1000; return x; }", 1000),
    ("func", "int identity(int x){ return x; } int main(){ return identity(1000); }", 1000),
]


@pytest.mark.parametrize("name,src,exp", _BATTERY, ids=[b[0] for b in _BATTERY])
def test_sparse_driver_argmax_matches(dense_model, name, src, exp):
    """The sparse (sparse_mm) model drives the SAME argmax-decoded VM answer."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    model, L = dense_model
    sp = SparseTransformer(model, compute_mode="sparse_mm")
    code = bytecode_to_isa(compile_c(src)[0])
    cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
    tr = run_pure_forward_complete(sp, L, code, max_steps=cap, mask=0xFFFFFFFF)
    got = tr[-1] & 0xFFFFFFFF if tr else None
    assert got == exp, f"{name}: sparse got {got}, want {exp}"
