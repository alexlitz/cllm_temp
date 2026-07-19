"""Byte-identity + accounting gates for the weight-dedup (weight-tying) pass.

The pass ties byte-identical weight tensors in a built ``SparseTransformer`` to a
single shared storage.  These tests prove:

  * the tie is BYTE-IDENTICAL — ``forward`` L-inf = 0 vs the un-tied model, and
    the end-to-end program decode (``run_pure_forward_complete``) is unchanged, so
    greedy (argmax) output is byte-identical;
  * the tie is REAL — the number of DISTINCT stored tensors (by ``id()``) drops
    below the number of block references, and the reported unique-tensor /
    nonzero-weight counts match the ``id()``-level truth;
  * the SHARED storage is genuinely shared (a mutation of the representative is
    visible through every tied reference).

The divmod (304-block) gate is opt-in via ``C4_TEST_DIVMOD_DEDUP=1`` (it is the
big-win config but the build is slower); LEAN + bitwise run by default.
"""
from __future__ import annotations

import os

import pytest
import torch

from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.sparse_forward import SparseTransformer
from c4_min.weight_dedup import (
    dedup_sparse_transformer, count_unique_stored_tensors,
    verify_forward_identical,
)


def _build(include_bitwise, include_divmod):
    return build_compact_sparse_streaming(
        code_size=44, include_bitwise=include_bitwise,
        include_divmod=include_divmod, compute_mode="dense_kernel")


def _token_streams(L, n_frames=(0, 2, 4)):
    """A small battery of representative frame streams (no C compiler needed)."""
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


def _forward_via_blocks(model, L, streams):
    """Run the block stack (embedding + overlay-free) and return final AX/reg rows."""
    rows = []
    with torch.no_grad():
        for stream in streams:
            toks = torch.tensor([stream], dtype=torch.long)
            x = model.embed[toks].clone()
            for blk in model.blocks:
                x = blk(x)
            rows.append(x[0, -1].clone())
    return rows


# ---------------------------------------------------------------------------
# Byte-identity: forward L-inf = 0 after tie.
# ---------------------------------------------------------------------------
def _tie_is_byte_identical(include_bitwise, include_divmod):
    tied, L, _ = _build(include_bitwise, include_divmod)
    ref, _, _ = _build(include_bitwise, include_divmod)   # untied twin
    streams = _token_streams(L)

    before = _forward_via_blocks(ref, L, streams)
    stats = dedup_sparse_transformer(tied, L)
    after = _forward_via_blocks(tied, L, streams)

    worst = max(float((a - b).abs().max()) for a, b in zip(before, after))
    assert worst == 0.0, f"tie changed forward (L-inf={worst})"
    # the tie must actually have shared something
    refs, uniq, _ = count_unique_stored_tensors(tied)
    assert uniq < refs, f"nothing tied ({uniq} unique of {refs} refs)"
    assert stats.unique_weight_tensors_after == uniq, \
        (stats.unique_weight_tensors_after, uniq)
    return stats


def test_lean_dedup_byte_identical():
    stats = _tie_is_byte_identical(include_bitwise=False, include_divmod=False)
    assert stats.nonzero_after < stats.nonzero_before


def test_bitwise_dedup_byte_identical():
    stats = _tie_is_byte_identical(include_bitwise=True, include_divmod=False)
    assert stats.nonzero_after < stats.nonzero_before


# ---------------------------------------------------------------------------
# The tie is REAL sharing: mutating the representative changes every tied ref.
# ---------------------------------------------------------------------------
def test_tie_shares_storage_identity():
    tied, L, _ = _build(include_bitwise=True, include_divmod=False)
    dedup_sparse_transformer(tied, L)
    # find a weight slot whose storage id() appears more than once
    from collections import defaultdict
    by_id = defaultdict(list)
    for bi, b in enumerate(tied.blocks):
        for kind, w in [("W_up", b.ffn.W_up), ("W_gate", b.ffn.W_gate),
                        ("W_down", b.ffn.W_down)]:
            store = w.csr if w.is_sparse else w.dense
            by_id[id(store)].append(w)
    shared = [ws for ws in by_id.values() if len(ws) > 1 and ws[0].nnz > 0]
    assert shared, "no shared nonzero storage found"
    ws = shared[0]
    a, bcopy = ws[0], ws[1]
    sa = a.csr if a.is_sparse else a.dense
    sb = bcopy.csr if bcopy.is_sparse else bcopy.dense
    assert sa is sb, "tied refs do not share the same storage object"


# ---------------------------------------------------------------------------
# End-to-end program decode is unchanged (greedy argmax correctness).
# ---------------------------------------------------------------------------
def _program_battery(model, L, cases):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import (
        run_pure_forward_complete, ref_interpret)
    for name, src, exp in cases:
        code = bytecode_to_isa(compile_c(src)[0])
        cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
        tr = run_pure_forward_complete(model, L, code, max_steps=cap,
                                       mask=0xFFFFFFFF)
        got = tr[-1] & 0xFFFFFFFF if tr else None
        assert got == exp, f"{name}: dedup got {got}, want {exp}"


def test_bitwise_dedup_program_decode_unchanged():
    tied, L, _ = _build(include_bitwise=True, include_divmod=False)
    dedup_sparse_transformer(tied, L)
    _program_battery(tied, L, [
        ("add", "int main(){ return 500 + 700; }", 1200),
        ("sub", "int main(){ return 1900 - 50; }", 1850),
        ("mul", "int main(){ return 100 * 10; }", 1000),
        ("cmp", "int main(){ if (5 > 3) return 1; return 0; }", 1),
        ("var", "int main(){ int x; x = 1000; return x; }", 1000),
        ("func", "int identity(int x){ return x; } "
                 "int main(){ return identity(1000); }", 1000),
        ("bw_or", "int main(){ return 12 | 3; }", 15),
        ("bw_and", "int main(){ return 12 & 10; }", 8),
    ])


@pytest.mark.skipif(os.environ.get("C4_TEST_DIVMOD_DEDUP") != "1",
                    reason="304-block divmod build is slow; "
                           "set C4_TEST_DIVMOD_DEDUP=1 to run")
def test_divmod_dedup_byte_identical_and_program_decode():
    stats = _tie_is_byte_identical(include_bitwise=True, include_divmod=True)
    # divmod is the big-win config: >50% nonzero savings expected.
    assert stats.nonzero_saved > stats.nonzero_after
    tied, L, _ = _build(include_bitwise=True, include_divmod=True)
    dedup_sparse_transformer(tied, L)
    _program_battery(tied, L, [
        ("div", "int main(){ return 720 / 6; }", 120),
        ("mod", "int main(){ return 84 % 5; }", 4),
    ])
