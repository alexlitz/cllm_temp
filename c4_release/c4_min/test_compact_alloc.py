"""STEP 2 gate: dim-sharing (register allocation by liveness) is byte-identical
(L-inf=0 on the decode bands) AND argmax-correct on the SINGLE full-op-set model
— including the bitwise ops that used to corrupt AX to 0xFFFFFFFF because the
additive-residual value-carrying scratch bands (AX_VAL, the ALU / operand-one-hot
/ query-address families) were wrongly shared.

The fix: never-share CUR_NIB + AX_VAL, and REFINE the weight-only liveness with
the OBSERVED value-liveness (empirical_value_liveness) so no dim is coloured over
a slot that still holds a stale nonzero scratch value.

MEMORY: the full op set (incl. the ~300 DIV/MOD blocks) is built via the
STREAMING sparse builder (peak build RSS ~one dense block, ~9.5 GB) — NEVER the
~130 GB full-op dense build.  The heavy dense-compact cross-check (~48 GB RSS) is
gated behind C4_TEST_DENSE_COMPACT=1.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_compact_alloc.py -v
"""
from __future__ import annotations

import os

import pytest
import torch

from c4_min.compact_alloc import (build_compact_pure_forward_model,
                                  build_compact_sparse_streaming,
                                  save_sparse_transformer,
                                  load_sparse_transformer)
from c4_min.nibble_pure_forward_complete import (
    run_pure_forward_complete,
    ref_interpret, make_overlay_complete, _build_frame, SP_INIT)
from c4_min.sparse_forward import SparseTransformer
from c4_min import blogspec_vocab as V


_RUN_DENSE_COMPACT = os.environ.get("C4_TEST_DENSE_COMPACT") == "1"


def _battery(compact, L, cases):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    for name, src, exp in cases:
        code = bytecode_to_isa(compile_c(src)[0])
        cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
        tr = run_pure_forward_complete(compact, L, code, max_steps=cap,
                                       mask=0xFFFFFFFF)
        got = tr[-1] & 0xFFFFFFFF if tr else None
        assert got == exp, f"{name}: compact got {got}, want {exp}"


_FULL_BATTERY = [
    ("add", "int main(){ return 500 + 700; }", 1200),
    ("mul", "int main(){ return 100 * 10; }", 1000),
    ("cmp", "int main(){ if (5 > 3) return 1; return 0; }", 1),
    ("var", "int main(){ int x; x = 1000; return x; }", 1000),
    ("func", "int identity(int x){ return x; } "
             "int main(){ return identity(1000); }", 1000),
    ("bw_or", "int main(){ return 12 | 3; }", 15),
    ("bw_and", "int main(){ return 12 & 10; }", 8),
    ("div", "int main(){ return 720 / 6; }", 120),
    ("mod", "int main(){ return 84 % 5; }", 4),
    ("divzero", "int main(){ return 5 / 0; }", 0),
]


# ---------------------------------------------------------------------------
# STREAMING SPARSE BUILD (the OOM fix): peak live memory is ~one dense block,
# not all N.  This is the memory-safe path for the FULL op-set model, and the
# streamed sparse model is byte-identical (L-inf=0 in ``dense_kernel`` mode) to
# ``SparseTransformer(build_compact_pure_forward_model)``.
# ---------------------------------------------------------------------------
def _decode_band_row(x, L):
    row = x[0, -1]
    vals = []
    for nm in ("PC_VAL", "AX_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED"):
        vals.append(row[getattr(L, nm)].reshape(-1))
    for k in range(8):
        vals.append(row[L.AX + k].reshape(-1))
    return torch.cat(vals)


def _forward_decode(model, L, srcs):
    """Run each source through the block stack; return {src: decode-band row}."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    out = {}
    for src in srcs:
        code = bytecode_to_isa(compile_c(src)[0])
        stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for _ in range(4):
            stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
        toks = torch.tensor([stream])
        ov = make_overlay_complete(code, L)
        with torch.no_grad():
            x = model.embed[toks].clone(); ov(x)
            for blk in model.blocks:
                x = blk(x)
        out[src] = _decode_band_row(x, L)
    return out


_BID_SRCS = ["int main(){ return 500 + 700; }",
             "int main(){ int x; x = 1000; return x; }",
             "int main(){ return 12 | 3; }",
             "int main(){ return 720 / 6; }"]


@pytest.fixture(scope="module")
def streamed_full():
    sparse, L, stats = build_compact_sparse_streaming(
        code_size=44, compute_mode="dense_kernel")
    return sparse, L, stats


def test_streaming_dim_sharing_actually_shares(streamed_full):
    _, _, stats = streamed_full
    assert stats.dim_after < stats.dim_before   # something actually shared


def test_streaming_full_battery(streamed_full):
    """The streamed full-op-set model decodes the whole op battery (incl. bitwise
    AND div/mod) correctly."""
    sparse, L, _ = streamed_full
    _battery(sparse, L, _FULL_BATTERY)


def test_streaming_save_reload_byte_identical(streamed_full, tmp_path):
    sparse, L, stats = streamed_full
    path = str(tmp_path / "full_sparse.pt")
    save_sparse_transformer(sparse, L, stats, path)
    r, Lr = load_sparse_transformer(path)
    assert r.dim == sparse.dim and len(r.blocks) == len(sparse.blocks)
    a = _forward_decode(sparse, L, _BID_SRCS)
    b = _forward_decode(r, Lr, _BID_SRCS)
    worst = max((a[s] - b[s]).abs().max().item() for s in _BID_SRCS)
    assert worst < 1e-9, worst


@pytest.mark.skipif(not _RUN_DENSE_COMPACT,
                    reason="the full-op dense-compact build is ~48 GB RSS; "
                           "set C4_TEST_DENSE_COMPACT=1 to run the cross-check")
def test_streaming_matches_dense_compact(streamed_full):
    """The streamed sparse build == ``SparseTransformer(dense-compact)`` on the
    decode bands (L-inf=0 in dense_kernel).  Heavy: builds the ~48 GB dense-compact
    full-op model, so gated behind C4_TEST_DENSE_COMPACT=1."""
    sparse, Ln, _ = streamed_full
    compact, Lo, _ = build_compact_pure_forward_model(code_size=44)
    old = SparseTransformer(compact, compute_mode="dense_kernel")
    del compact
    a = _forward_decode(old, Lo, _BID_SRCS)
    b = _forward_decode(sparse, Ln, _BID_SRCS)
    worst = max((a[s] - b[s]).abs().max().item() for s in _BID_SRCS)
    assert worst < 1e-9, worst
