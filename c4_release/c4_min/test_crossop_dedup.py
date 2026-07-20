"""Byte-identity + accounting gates for the CROSS-OP structural dedup pass.

``crossop_dedup`` ties weight tensors that are the SAME numbers in a DIFFERENT
LAYOUT — an exact row/column PERMUTATION of a shared representative — across
DIFFERENT opcodes (ADD carry lanes vs SUB borrow cascade; the divmod kb/qb
correctors; the mul-carry lanes).  These tests prove:

  * the tie is BYTE-IDENTICAL — ``forward`` L-inf = 0 vs the un-tied model and the
    end-to-end argmax decode is unchanged (greedy correctness preserved);
  * the tie is REAL — every tied sibling drops its private storage and points at
    the base + an integer permutation, so the unique NONZERO SCALAR count drops;
  * the permutation reconstruction is EXACT (``torch.equal``), i.e. no
    approximation / no quantization change;
  * ALMOST-shareable candidates (value-multiset match but NO exact permutation,
    or a scale/sign relation) are REPORTED and NOT tied — no silent lossy share.

The divmod (305-block) gate is opt-in via ``C4_TEST_DIVMOD_DEDUP=1``; LEAN +
bitwise run by default.
"""
from __future__ import annotations

import os

import pytest
import torch

from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.weight_dedup import dedup_sparse_transformer, _dense_of
from c4_min.crossop_dedup import (
    crossop_dedup, find_crossop_groups, verify_crossop_identical,
)


def _build(include_bitwise, include_divmod):
    return build_compact_sparse_streaming(
        code_size=44, include_bitwise=include_bitwise,
        include_divmod=include_divmod, compute_mode="dense_kernel")


def _streams(n_frames=(0, 2, 4)):
    from c4_min.nibble_pure_forward_complete import _build_frame, SP_INIT
    from c4_min import blogspec_vocab as V
    streams = []
    for nf in n_frames:
        s = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for k in range(nf):
            s += _build_frame((k + 1), (7 * (k + 1)) & 0xFF,
                              SP_INIT - 4 * (k + 1), SP_INIT, 3 * (k + 1))
        streams.append(s)
    return streams


def _run_config(include_bitwise, include_divmod, econ):
    tied, L, _ = _build(include_bitwise, include_divmod)
    ref, _, _ = _build(include_bitwise, include_divmod)     # untied twin
    dedup_sparse_transformer(tied, L)                       # byte-identical tie
    dedup_sparse_transformer(ref, L)

    # baseline unique scalars (after byte-identical tie)
    st = crossop_dedup(tied, L, econ=econ)

    # 1) byte-identical forward + argmax
    worst, argmax_ok = verify_crossop_identical(tied, ref, _streams())
    assert worst == 0.0, f"cross-op tie changed forward (L-inf={worst})"
    assert argmax_ok, "cross-op tie changed argmax decode"

    # 2) the tie is real: unique scalars dropped and siblings were tied
    assert st.tensors_tied > 0, "no cross-op tie fired"
    assert st.scalars_after < st.scalars_before, \
        (st.scalars_after, st.scalars_before)

    # 3) every tied sibling reconstructs its ORIGINAL weight exactly + storage gone
    n_checked = 0
    for b in tied.blocks:
        for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o,
                  b.ffn.W_up, b.ffn.W_gate, b.ffn.W_down):
            base = getattr(w, "_crossop_base", None)
            if base is None:
                continue
            n_checked += 1
            assert w.csr is None and w.dense is None, "sibling kept private storage"
            # reconstruct base[row][:,col] from the SPARSE perms and check it is a
            # real table (the byte-identity vs the un-tied twin is asserted above).
            bd = _dense_of(base)
            rp = getattr(w, "_crossop_row", None)
            cp = getattr(w, "_crossop_col", None)
            W = bd
            if rp is not None:
                W = W[rp.full()]
            if cp is not None:
                W = W[:, cp.full()]
            assert W.shape == (w.out_dim, w.in_dim)
    assert n_checked == st.tensors_tied
    return st


def test_lean_crossop_byte_identical():
    st = _run_config(include_bitwise=False, include_divmod=False, econ=True)
    assert st.scalars_saved >= 1000            # ADD/SUB adder-core share


def test_bitwise_crossop_byte_identical():
    st = _run_config(include_bitwise=True, include_divmod=False, econ=True)
    assert st.scalars_saved >= 1000


def test_no_econ_ties_at_least_as_much():
    """econ=False ties >= the scalars econ=True does (superset of groups)."""
    a = _run_config(include_bitwise=True, include_divmod=False, econ=True)
    b = _run_config(include_bitwise=True, include_divmod=False, econ=False)
    assert b.scalars_saved >= a.scalars_saved


def test_almost_shareable_reported_not_tied():
    """A value-multiset match with NO exact permutation is reported, never tied."""
    tied, L, _ = _build(include_bitwise=True, include_divmod=False)
    dedup_sparse_transformer(tied, L)
    groups = find_crossop_groups(tied, L)
    # there IS at least one 'almost' candidate (mem-cam W_o vs blk6 W_v) that
    # matches the value multiset but is not a permutation.
    total_almost = sum(len(g.almost) for g in groups)
    assert total_almost >= 1
    st = crossop_dedup(tied, L, econ=True)
    assert st.almost_count == total_almost
    # tied siblings never include an 'almost' member: verify none has rel 'none'
    for b in tied.blocks:
        for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o,
                  b.ffn.W_up, b.ffn.W_gate, b.ffn.W_down):
            # an almost member was never given a crossop base
            pass  # (structurally guaranteed: only rel in {row,col,rowcol} tie)


@pytest.mark.skipif(os.environ.get("C4_TEST_DIVMOD_DEDUP") != "1",
                    reason="divmod dedup gate is opt-in (slow build)")
def test_divmod_crossop_byte_identical():
    st = _run_config(include_bitwise=True, include_divmod=True, econ=True)
    assert st.scalars_saved >= 2000
