"""CUDA-GRAPHED + VECTORIZED bounded-KV incremental driver (``qwen_lean_evict_graphed``).

Proves the landed graphed driver (:func:`qwen_lean_evict_graphed.run_program_lean_evict_graphed`)
decodes BYTE-IDENTICAL to the naive fresh-window lean driver and to the eager evict
driver on the arith / cmp / branch / loop / memory / latest-write-wins battery, and
that its vectorized helpers (the frame-template scatter + the batched value-argmax
decode) are byte-identical to the per-scalar ``_append_reg_frame`` / per-register
``_snap`` they replace.

Runs on CPU (the ``GraphedIncrementalForward`` transparently falls back to the eager
lean forward off-GPU, so the byte-identity assertions hold identically; the CUDA-graph
replay + the ms/step drop are exercised on-GPU by ``bench_bounded_kv_incremental``).
The DEFAULT ``Q.build`` (``code_from_memory=True``) is used — the same build the
production driver and the bench run on.
"""
from __future__ import annotations

import pytest
import torch

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min import qwen_lean_evict as EV
from c4_min import qwen_lean_evict_graphed as GEV


@pytest.fixture(scope="module")
def lean_memcmp():
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)      # DEFAULT code_from_memory
    return LF.LeanQwenVM.from_full_vm(vm, device="cpu")


def _A(prog):
    return isa.assemble(prog)


# The battery — one representative per hazard the graphed driver must survive.
BATTERY = [
    ("add", _A([("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)])),
    ("sub", _A([("IMM", 200), ("PSH", 0), ("IMM", 55), ("SUB", 0), ("HALT", 0)])),
    ("cmp_eq", _A([("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0), ("HALT", 0)])),
    ("cmp_gt", _A([("IMM", 9), ("PSH", 0), ("IMM", 7), ("GT", 0), ("HALT", 0)])),
    ("if_bz", _A([("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])),
    ("jmp", _A([("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)])),
    ("loop_cd20", _A([("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                      ("BNZ", 1), ("HALT", 0)])),
    ("si_li", _A([("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                  ("IMM", 5), ("LI", 0), ("HALT", 0)])),
    ("zfod", _A([("IMM", 50), ("LI", 0), ("HALT", 0)])),
    # latest-write-wins: two stores to the SAME address, then a load.
    ("lww", _A([("IMM", 30), ("PSH", 0), ("IMM", 1), ("SI", 0),
                ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
                ("IMM", 30), ("LI", 0), ("HALT", 0)])),
    # distinct-address live heap.
    ("heap3", _A([("IMM", 10), ("PSH", 0), ("IMM", 100), ("SI", 0),
                  ("IMM", 11), ("PSH", 0), ("IMM", 111), ("SI", 0),
                  ("IMM", 12), ("PSH", 0), ("IMM", 122), ("SI", 0),
                  ("IMM", 11), ("LI", 0), ("HALT", 0)])),
    # allocate then FREE (zero-store) a cell, then read a DIFFERENT live cell.
    ("free", _A([("IMM", 7), ("PSH", 0), ("IMM", 42), ("SI", 0),
                 ("IMM", 8), ("PSH", 0), ("IMM", 99), ("SI", 0),
                 ("IMM", 7), ("PSH", 0), ("IMM", 0), ("SI", 0),
                 ("IMM", 8), ("LI", 0), ("HALT", 0)])),
]


@pytest.mark.parametrize("name,code", BATTERY, ids=[n for n, _ in BATTERY])
def test_graphed_byte_identical_to_naive(lean_memcmp, name, code):
    """The graphed + vectorized driver's AX trace == the naive fresh-window driver's
    AND == the eager evict driver's."""
    naive = LF.run_program_lean(lean_memcmp, code, max_steps=256)
    eager = EV.run_program_lean_evict(lean_memcmp, code, max_steps=256, evict="async",
                                      prune_interval=4, watermark_rows=16)
    graphed = GEV.run_program_lean_evict_graphed(lean_memcmp, code, max_steps=256)
    assert graphed.ax_trace == naive["ax_trace"], (name, graphed.ax_trace,
                                                   naive["ax_trace"])
    assert graphed.ax_trace == eager.ax_trace, (name, graphed.ax_trace, eager.ax_trace)


def test_vectorized_frame_builder_byte_identical(lean_memcmp):
    """The ``VectorizedFrameBuilder`` residual == the per-scalar ``_append_reg_frame``
    residual, across a variety of register states + a load / non-load / load-clear
    transition (the buffer-reuse path must clear stale load lanes)."""
    lean = lean_memcmp
    code = _A([("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
               ("IMM", 5), ("LI", 0), ("HALT", 0)])
    fb = GEV.VectorizedFrameBuilder(lean, code)
    seq = [
        ({"PC": 0, "AX": 5, "SP": 0xFC, "BP": 0xFC, "STACK0": 0}, None),
        ({"PC": 5, "AX": 5, "SP": 0x100, "BP": 0xFC, "STACK0": 0x23}, 5),   # load
        ({"PC": 6, "AX": 0x23, "SP": 0x100, "BP": 0xFC, "STACK0": 0}, None),  # clear
        ({"PC": 3, "AX": 255, "SP": 0x10000, "BP": 0xF8, "STACK0": 0x23}, 0x2A),
        ({"PC": 17, "AX": 0x10000, "SP": 0, "BP": 0x1234, "STACK0": 200}, None),
    ]
    for rs, la in seq:
        x_ref, pos_ref, _ = EV._append_reg_frame(lean, code, rs, la, 42, 0)
        x_vec, pos_vec = fb.build(rs, la, 42)
        assert torch.equal(x_ref, x_vec), (rs, la, (x_ref != x_vec).nonzero().tolist())
        assert torch.equal(pos_ref, pos_vec), (rs, la)


def test_batched_snap_byte_identical(lean_memcmp):
    """The batched value-argmax requant == five separate ``_snap`` calls (same fp64
    arithmetic + tie-break), across small / byte-wrapped / 0x10000-range lane values."""
    from c4_min.qwen_full_vm import _snap
    bs = GEV.BatchedSnap(torch.device("cpu"))
    vals = torch.tensor([5.0, 255.3, 0x10000 + 0.4, -0.1, 200.9, 0.0, 0x100FF - 0.2])
    ref = [_snap(v) for v in vals]
    got = bs.snap_many(vals)
    assert ref == got, (ref, got)


def test_graphed_cache_is_bounded(lean_memcmp):
    """The graphed driver keeps the SAME bounded cache as the eager evict driver on a
    register-only loop (constant, step-count-independent)."""
    prog = [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("BNZ", 1), ("HALT", 0)]
    code = _A(prog)
    n_code = len(code) if lean_memcmp.code_from_memory else 0
    bound = 1 + n_code + 6                                # BOS + CODE frames + reg frame
    r = GEV.run_program_lean_evict_graphed(lean_memcmp, code, max_steps=256)
    assert r.steps == 82
    assert r.max_cache_rows <= bound, (r.max_cache_rows, bound)
    assert max(r.cache_size_trace) <= bound, (r.cache_size_trace, bound)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
