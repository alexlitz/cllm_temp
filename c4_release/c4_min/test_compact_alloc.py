"""STEP 2 gate: dim-sharing (register allocation by liveness) is byte-identical
(L-inf=0 on the decode bands) AND argmax-correct on ALL configs — including the
bitwise config that used to corrupt AX to 0xFFFFFFFF because the additive-residual
value-carrying scratch bands (AX_VAL, the ALU / operand-one-hot / query-address
families) were wrongly shared.

The fix: never-share CUR_NIB + AX_VAL, and REFINE the weight-only liveness with
the OBSERVED value-liveness (empirical_value_liveness) so no dim is coloured over
a slot that still holds a stale nonzero scratch value.

LEAN + bitwise are CI-fast.  divmod (304 blocks, 54 GB dense-equivalent, 106 GB
transient build) is gated behind C4_TEST_DIVMOD=1.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_compact_alloc.py -v
"""
from __future__ import annotations

import os

import pytest
import torch

from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete,
    ref_interpret, make_overlay_complete, _build_frame, SP_INIT)
from c4_min import blogspec_vocab as V


def _decode_band_linf(include_bitwise):
    """L-inf between the reference and compact block-stack outputs at the DECODE
    bands (the register images + AX nibbles the driver argmaxes)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    ref, Lref = build_pure_forward_complete_model(
        code_size=44, include_bitwise=include_bitwise, include_divmod=False)
    compact, Lc, stats = build_compact_pure_forward_model(
        code_size=44, include_bitwise=include_bitwise, include_divmod=False)
    srcs = ["int main(){ return 500 + 700; }",
            "int main(){ int x; x = 1000; return x; }"]
    if include_bitwise:
        srcs.append("int main(){ return 12 | 3; }")
    worst = 0.0
    for src in srcs:
        code = bytecode_to_isa(compile_c(src)[0])
        stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for _ in range(4):
            stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
        toks = torch.tensor([stream])
        ov_r = make_overlay_complete(code, Lref)
        ov_c = make_overlay_complete(code, Lc)
        with torch.no_grad():
            xr = ref.embed[toks].clone(); ov_r(xr)
            for blk in ref.blocks:
                xr = blk(xr)
            xc = compact.embed[toks].clone(); ov_c(xc)
            for blk in compact.blocks:
                xc = blk(xc)
        for nm in ("PC_VAL", "AX_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED"):
            worst = max(worst, (xr[0, -1, getattr(Lref, nm)]
                                - xc[0, -1, getattr(Lc, nm)]).abs().item())
        for k in range(8):
            worst = max(worst, (xr[0, -1, Lref.AX + k]
                                - xc[0, -1, Lc.AX + k]).abs().item())
    return worst, stats


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


def test_lean_dim_sharing_is_byte_identical():
    worst, stats = _decode_band_linf(include_bitwise=False)
    assert worst < 1e-9, worst
    assert stats.dim_after < stats.dim_before   # something actually shared


def test_bitwise_dim_sharing_is_byte_identical():
    """The AX_VAL / scratch-band never-share fix: bitwise now L-inf=0 (used to be
    ~1.6e5 with AX corrupted to 0xFFFFFFFF)."""
    worst, stats = _decode_band_linf(include_bitwise=True)
    assert worst < 1e-9, worst
    assert stats.dim_after < stats.dim_before


def test_bitwise_compact_battery():
    compact, L, _ = build_compact_pure_forward_model(
        code_size=44, include_bitwise=True, include_divmod=False)
    _battery(compact, L, [
        ("add", "int main(){ return 500 + 700; }", 1200),
        ("mul", "int main(){ return 100 * 10; }", 1000),
        ("cmp", "int main(){ if (5 > 3) return 1; return 0; }", 1),
        ("var", "int main(){ int x; x = 1000; return x; }", 1000),
        ("func", "int identity(int x){ return x; } "
                 "int main(){ return identity(1000); }", 1000),
        ("bw_or", "int main(){ return 12 | 3; }", 15),
        ("bw_and", "int main(){ return 12 & 10; }", 8),
    ])


@pytest.mark.skipif(os.environ.get("C4_TEST_DIVMOD") != "1",
                    reason="304-block divmod build is 106 GB transient / slow; "
                           "set C4_TEST_DIVMOD=1 to run")
def test_divmod_compact_battery():
    compact, L, _ = build_compact_pure_forward_model(
        code_size=44, include_bitwise=True, include_divmod=True)
    _battery(compact, L, [
        ("div", "int main(){ return 720 / 6; }", 120),
        ("mod", "int main(){ return 84 % 5; }", 4),
        ("divzero", "int main(){ return 5 / 0; }", 0),
    ])


# ---------------------------------------------------------------------------
# STREAMING SPARSE BUILD (the OOM fix): peak live memory is ~one dense block,
# not all N.  The streamed sparse model is byte-identical (L-inf=0 in
# ``dense_kernel`` mode) to ``SparseTransformer(build_compact_pure_forward_model)``.
# ---------------------------------------------------------------------------
def _decode_band_row(x, L):
    row = x[0, -1]
    vals = []
    for nm in ("PC_VAL", "AX_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED"):
        vals.append(row[getattr(L, nm)].reshape(-1))
    for k in range(8):
        vals.append(row[L.AX + k].reshape(-1))
    return torch.cat(vals)


def _stream_vs_old_linf(include_bitwise):
    """L-inf between ``SparseTransformer(compact-dense)`` and the streamed sparse
    build on the decode bands."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.compact_alloc import build_compact_sparse_streaming
    from c4_min.sparse_forward import SparseTransformer

    compact, Lo, _ = build_compact_pure_forward_model(
        code_size=44, include_bitwise=include_bitwise, include_divmod=False)
    old = SparseTransformer(compact, compute_mode="dense_kernel")
    new, Ln, stats = build_compact_sparse_streaming(
        code_size=44, include_bitwise=include_bitwise, include_divmod=False,
        compute_mode="dense_kernel")
    srcs = ["int main(){ return 500 + 700; }",
            "int main(){ int x; x = 1000; return x; }"]
    if include_bitwise:
        srcs.append("int main(){ return 12 | 3; }")
    worst = 0.0
    for src in srcs:
        code = bytecode_to_isa(compile_c(src)[0])
        stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for _ in range(4):
            stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
        toks = torch.tensor([stream])
        ov_o = make_overlay_complete(code, Lo)
        ov_n = make_overlay_complete(code, Ln)
        with torch.no_grad():
            xo = old.embed[toks].clone(); ov_o(xo)
            for blk in old.blocks:
                xo = blk(xo)
            xn = new.embed[toks].clone(); ov_n(xn)
            for blk in new.blocks:
                xn = blk(xn)
        worst = max(worst, (_decode_band_row(xo, Lo)
                            - _decode_band_row(xn, Ln)).abs().max().item())
    return worst, new, Ln, stats


def test_lean_streaming_is_byte_identical():
    worst, _, _, stats = _stream_vs_old_linf(include_bitwise=False)
    assert worst < 1e-9, worst
    assert stats.dim_after < stats.dim_before


def test_bitwise_streaming_is_byte_identical():
    worst, sparse, L, _ = _stream_vs_old_linf(include_bitwise=True)
    assert worst < 1e-9, worst
    # streamed model runs the battery correctly too.
    _battery(sparse, L, [
        ("add", "int main(){ return 500 + 700; }", 1200),
        ("var", "int main(){ int x; x = 1000; return x; }", 1000),
        ("bw_or", "int main(){ return 12 | 3; }", 15),
    ])


def test_streaming_save_reload_byte_identical(tmp_path):
    from c4_min.compact_alloc import (build_compact_sparse_streaming,
                                      save_sparse_transformer,
                                      load_sparse_transformer)
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    sparse, L, stats = build_compact_sparse_streaming(
        code_size=44, include_bitwise=True, include_divmod=False,
        compute_mode="dense_kernel")
    path = str(tmp_path / "lean_sparse.pt")
    save_sparse_transformer(sparse, L, stats, path)
    r, Lr = load_sparse_transformer(path)
    assert r.dim == sparse.dim and len(r.blocks) == len(sparse.blocks)
    worst = 0.0
    for src in ["int main(){ return 500 + 700; }",
                "int main(){ return 12 | 3; }"]:
        code = bytecode_to_isa(compile_c(src)[0])
        stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for _ in range(4):
            stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
        toks = torch.tensor([stream])
        with torch.no_grad():
            xa = sparse.embed[toks].clone(); make_overlay_complete(code, L)(xa)
            for b in sparse.blocks:
                xa = b(xa)
            xb = r.embed[toks].clone(); make_overlay_complete(code, Lr)(xb)
            for b in r.blocks:
                xb = b(xb)
        worst = max(worst, (_decode_band_row(xa, L)
                            - _decode_band_row(xb, Lr)).abs().max().item())
    assert worst < 1e-9, worst
