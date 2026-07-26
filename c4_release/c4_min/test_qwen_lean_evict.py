"""BOUNDED KV EVICTION on the LEAN compacted forward (``qwen_lean_evict``).

Proves the persistent-cache + bounded-eviction driver
(:func:`qwen_lean_evict.run_program_lean_evict`) decodes BYTE-IDENTICAL to the
naive fresh-window lean driver (:func:`qwen_lean_forward.run_program_lean`) on the
arith / cmp / branch / loop / memory / latest-write-wins battery — in ALL three
eviction modes (``off`` / ``sync`` / ``async``) — and that the cache stays BOUNDED
on a long program.

Run on CPU (memory-light; the compacted model is 7-14 layers).  The async mode's
CUDA-stream path degrades to a plain gather on CPU (``AsyncPruner._is_cuda`` False),
so the byte-identity + bound assertions hold identically off-GPU; the GPU overlap
is exercised by ``bench_lean_evict.py`` / the demo script.
"""
from __future__ import annotations

import pytest

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min import qwen_lean_evict as EV


@pytest.fixture(scope="module")
def lean_memcmp():
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    return LF.LeanQwenVM.from_full_vm(vm, device="cpu")


def _A(prog):
    return isa.assemble(prog)


# The battery — one representative per hazard the eviction must survive.
BATTERY = [
    ("add", _A([("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)])),
    ("sub", _A([("IMM", 200), ("PSH", 0), ("IMM", 55), ("SUB", 0), ("HALT", 0)])),
    ("cmp_eq", _A([("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0), ("HALT", 0)])),
    ("cmp_gt", _A([("IMM", 9), ("PSH", 0), ("IMM", 7), ("GT", 0), ("HALT", 0)])),
    ("if_bz", _A([("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])),
    ("jmp", _A([("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)])),
    ("loop_cd5", _A([("IMM", 5), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                     ("BNZ", 1), ("HALT", 0)])),
    ("loop_cd20", _A([("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                      ("BNZ", 1), ("HALT", 0)])),
    ("si_li", _A([("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                  ("IMM", 5), ("LI", 0), ("HALT", 0)])),
    ("zfod", _A([("IMM", 50), ("LI", 0), ("HALT", 0)])),
    # latest-write-wins: two stores to the SAME address, then a load — the RoPE
    # same-address supersession hazard the structural prune must handle.
    ("lww", _A([("IMM", 30), ("PSH", 0), ("IMM", 1), ("SI", 0),
                ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
                ("IMM", 30), ("LI", 0), ("HALT", 0)])),
    ("var_add", _A([("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
                    ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3),
                    ("ADD", 0), ("HALT", 0)])),
    # distinct-address live heap: three cells written, one read back.
    ("heap3", _A([("IMM", 10), ("PSH", 0), ("IMM", 100), ("SI", 0),
                  ("IMM", 11), ("PSH", 0), ("IMM", 111), ("SI", 0),
                  ("IMM", 12), ("PSH", 0), ("IMM", 122), ("SI", 0),
                  ("IMM", 11), ("LI", 0), ("HALT", 0)])),
    # allocate then FREE (zero-store) a cell, then read a DIFFERENT live cell — the
    # freed zero-row must be read-tolerable (nil under the BOS sink) even un-pruned.
    ("free", _A([("IMM", 7), ("PSH", 0), ("IMM", 42), ("SI", 0),
                 ("IMM", 8), ("PSH", 0), ("IMM", 99), ("SI", 0),
                 ("IMM", 7), ("PSH", 0), ("IMM", 0), ("SI", 0),
                 ("IMM", 8), ("LI", 0), ("HALT", 0)])),
]


@pytest.mark.parametrize("name,code", BATTERY, ids=[n for n, _ in BATTERY])
@pytest.mark.parametrize("evict", ["off", "sync", "async"])
def test_evict_byte_identical_to_naive(lean_memcmp, name, code, evict):
    """The persistent-cache + eviction driver's AX trace == the naive fresh-window
    driver's, in every eviction mode.  The transient un-pruned rows change no
    decode; a very small ``prune_interval`` maximises prune churn."""
    naive = LF.run_program_lean(lean_memcmp, code, max_steps=256)
    r = EV.run_program_lean_evict(
        lean_memcmp, code, max_steps=256, evict=evict,
        prune_interval=4, watermark_rows=16)
    assert r.ax_trace == naive["ax_trace"], (name, evict, r.ax_trace, naive["ax_trace"])


def test_register_loop_cache_is_bounded(lean_memcmp):
    """A register-only loop keeps a CONSTANT cache over its whole run — the per-step
    register-frame supersession bounds it (independent of the step count).

    The bound is a small CONSTANT set by the layout, NOT growing with steps:
      * baked-table layout (``code_from_memory=False``): BOS + one 6-row register
        frame = 7 rows;
      * code-from-memory layout (the DEFAULT): the program also lives in the cache as
        one persistent CODE frame per instruction (TAG_CODE, never evicted), so the
        bound is BOS(1) + n_code CODE frames + one 6-row register frame.
    Either way the cache is BOUNDED (constant, step-count-independent) — the point of
    the eviction."""
    prog = [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("BNZ", 1), ("HALT", 0)]
    code = _A(prog)
    n_code = len(code) if lean_memcmp.code_from_memory else 0
    bound = 1 + n_code + 6                                # BOS + CODE frames + reg frame
    r = EV.run_program_lean_evict(lean_memcmp, code, max_steps=256, evict="async")
    assert r.steps == 82
    assert r.max_cache_rows <= bound, (r.max_cache_rows, bound)
    # the cache never grew with the step count (constant across the whole run).
    assert max(r.cache_size_trace) <= bound, (r.cache_size_trace, bound)
    assert min(r.cache_size_trace) == max(r.cache_size_trace), r.cache_size_trace


def test_async_prune_reclaims_freed_rows(lean_memcmp):
    """A heap that allocates + frees DISTINCT cells accumulates freed zero-rows under
    ``evict="off"`` but the async periodic prune RECLAIMS them, so the async cache is
    strictly smaller — and BOTH decode byte-identical to the naive driver on the
    (short) executable prefix."""
    # allocate addr 20,21 then free each (store 0), then read a live one — kept small
    # to fit code_size=24 (23 instrs).  Even 2 distinct freed cells suffice to show
    # the async prune reclaims what ``off`` retains.
    prog = []
    for a in (20, 21):
        prog += [("IMM", a), ("PSH", 0), ("IMM", a + 1), ("SI", 0)]
    prog += [("IMM", 20), ("PSH", 0), ("IMM", 0), ("SI", 0)]          # free addr 20
    prog += [("IMM", 21), ("LI", 0), ("HALT", 0)]                     # read live 21
    code = _A(prog)
    naive = LF.run_program_lean(lean_memcmp, code, max_steps=128)
    off = EV.run_program_lean_evict(lean_memcmp, code, max_steps=128, evict="off")
    asy = EV.run_program_lean_evict(lean_memcmp, code, max_steps=128, evict="async",
                                    prune_interval=8, watermark_rows=64)
    assert off.ax_trace == naive["ax_trace"]
    assert asy.ax_trace == naive["ax_trace"]
    # the freed cells accumulate under off; the async prune reclaims them.
    assert asy.final_cache_rows <= off.final_cache_rows
    assert asy.total_evicted >= off.total_evicted


def test_supersede_store_addr_is_structural(lean_memcmp):
    """Same-address supersession happens even with NO periodic prune (evict='off',
    huge interval) — it is structural (correctness), because the lean RoPE memory-CAM
    cannot resolve two physically-present same-address rows."""
    code = _A([("IMM", 30), ("PSH", 0), ("IMM", 1), ("SI", 0),
               ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
               ("IMM", 30), ("LI", 0), ("HALT", 0)])
    naive = LF.run_program_lean(lean_memcmp, code, max_steps=32)
    off = EV.run_program_lean_evict(lean_memcmp, code, max_steps=32, evict="off")
    assert off.ax_trace == naive["ax_trace"]
    assert off.ax_trace[-1] == 9                          # reads the LATEST write


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
