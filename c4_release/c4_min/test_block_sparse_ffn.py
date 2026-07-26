"""Byte-exactness gate for the static block-sparse FFN (block_sparse_ffn.py).

The COO gather-scale-scatter and the dense-active compact sub-block must reproduce
the dense SparseFFN.forward output up to fp-reduction ORDER only (the 1-nnz-per-row
majority is bit-exact; multi-nnz rows differ by an fp-accum residue far below the
integer decode margin). CPU, memory-light.

    OMP_NUM_THREADS=4 python -m pytest c4_min/test_block_sparse_ffn.py -v
"""
from __future__ import annotations

import pytest
import torch

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.block_sparse_ffn import BlockSparseFFN, CooLinear


@pytest.fixture(scope="module")
def ffns():
    sparse, L, _ = build_lib_model_streaming(
        code_size=24, recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
    sparse.materialize_dense(device="cpu")
    seen = set()
    out = []
    for b in sparse.blocks:
        if id(b) in seen:
            continue
        seen.add(id(b))
        if not getattr(b, "_routed", False):
            out.append(b.ffn)
    return out


@pytest.mark.parametrize("mode", ["coo", "dense_active"])
def test_block_sparse_ffn_decode_margin(ffns, mode):
    """Every non-routed FFN's block-sparse forward matches the dense forward to
    within the integer decode margin (residue is fp-reduction order)."""
    torch.manual_seed(0)
    D = ffns[0].W_up.in_dim
    S = 64
    for f in ffns:
        bsf = BlockSparseFFN(f, mode=mode)
        x = torch.randn(1, S, D) * 50.0
        with torch.no_grad():
            yd = f.forward(x)
            yb = bsf.forward(x)
        rel = (yd - yb).abs().max().item() / (yd.abs().max().item() + 1e-30)
        # far below the 0.5 integer decode margin relative to the band magnitude.
        assert rel < 1e-2, (mode, rel)


def test_coo_single_nnz_rows_bit_exact():
    """A weight whose every output row reads exactly ONE input dim (the majority
    case: median 1 nnz/row) is BIT-EXACT (no accumulation -> no order residue)."""
    torch.manual_seed(1)
    out_dim, in_dim = 40, 200
    w = torch.zeros(out_dim, in_dim)
    cols = torch.randint(0, in_dim, (out_dim,))
    for r in range(out_dim):
        w[r, cols[r]] = torch.randn(())
    coo = CooLinear.from_dense(w)
    x = torch.randn(1, 32, in_dim) * 100.0
    with torch.no_grad():
        import torch.nn.functional as F
        yd = F.linear(x, w)
        yb = coo.linear(x)
    assert torch.equal(yd, yb), (yd - yb).abs().max().item()
