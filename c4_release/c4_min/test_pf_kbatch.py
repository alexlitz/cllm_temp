"""#747 — WITHIN-PROGRAM BIG-K batching byte-exactness regression (bounded-KV path).

Proves the K-batched bounded-KV verify (``pf_kbatch.KBatchBoundedRunner``, K VM
steps per forward) decodes the IDENTICAL per-step AX trace as the K=1 sequential
bounded-KV driver (``pos_sparse_bounded.BoundedPosSparseRunner``), across the battery
(arith / DIV / MOD / mem-LI / SI / JSR-LEV) and a deep nested loop, for a range of K.

Memory-safe: routes to the streaming build (``build_compact_sparse_streaming``, peak
~5 GB); the model is module-cached so all K reuse one build.  CPU-safe (no GPU
needed).  This is the byte-exact-or-nothing gate for the K-batching lever.
"""
from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("C4_POS_SPARSE", "1")

from c4_min import isa
from c4_min.pf_kbatch import KBatchBoundedRunner
from c4_min.pf_speculative import draft_pf_program
import c4_min.bench_pf_kbatch as KB
from c4_min.bench_composed_fast_path import _battery, _nested_prog


_MODEL = None
_L = None
_RUNNER = None
_WINDOW = 64


def _model():
    global _MODEL, _L, _RUNNER
    if _MODEL is None:
        from c4_min.compact_alloc import build_compact_sparse_streaming
        KB.runner_window = _WINDOW
        _MODEL, _L, _ = build_compact_sparse_streaming(
            code_size=64, compute_mode="dense_kernel")
        _RUNNER = KBatchBoundedRunner(_MODEL, _L, window=_WINDOW)
    return _MODEL, _L, _RUNNER


@pytest.mark.parametrize("K", [1, 2, 4, 8, 16, 32, 64])
def test_kbatch_battery_byte_exact(K):
    """K-batch bounded == K=1 sequential bounded, per-step AX, on the battery."""
    model, L, runner = _model()
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        base = KB.drive_seq_bounded(model, L, code, seed_mem=seed)
        kb, _ = KB.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed)
        assert base == kb, f"K={K} {name}: seq={base} kbatch={kb}"


@pytest.mark.parametrize("K", [1, 4, 16, 64])
def test_kbatch_deep_nested_loop(K):
    """K-batch bounded == the FREE perfect draft over a deep nested loop (60 steps,
    multiple inner/outer loop crossings) — the multi-batch stress case."""
    model, L, runner = _model()
    deep_steps = 60
    deep = _nested_prog(3, 4)
    draft = draft_pf_program(deep, max_steps=deep_steps, mask=0xFF)
    draft_ax = [f["ax"] & 0xFF for f in draft.frames]
    kb, _ = KB.drive_kbatch(model, L, runner, deep, K=K, seed_mem={},
                            max_steps=deep_steps)
    n = min(len(kb), len(draft_ax))
    assert n > 0
    assert kb[:n] == draft_ax[:n], f"K={K}: kbatch={kb[:n]} draft={draft_ax[:n]}"


def test_kbatch_forward_count():
    """forwards == ceil(steps / K): the whole point (steps/K forwards, not steps)."""
    model, L, runner = _model()
    code = isa.assemble([("IMM", 12), ("PSH", 0), ("IMM", 30), ("ADD", 0),
                         ("HALT", 0)])
    n_steps = len(draft_pf_program(code, max_steps=200).frames)
    for K in (1, 2, 4):
        _, n_fwd = KB.drive_kbatch(model, L, runner, code, K=K, seed_mem={})
        assert n_fwd == (n_steps + K - 1) // K, (K, n_fwd, n_steps)


# ---------------------------------------------------------------------------
# #874 — PER-ROW block-skip byte-exactness.  Each block runs ONLY the query rows
# whose op uses it (not the union of all K).  The reference is the K=1 SINGLE-STEP
# decode (each step in its own forward -> no cross-row union), which is the TRUE
# per-step semantics.  (The older UNION K-batch is NOT byte-exact to it for a mixed
# batch — its SHL/EQ ax-recompose blocks corrupt the AX band of the other rows — so
# per-row is both correct and cheaper.  We therefore gate per-row against the K=1
# per-step reference, not against the union K-batch.)
# ---------------------------------------------------------------------------
def _perstep_ref(model, L, runner, code, *, seed, max_steps=200):
    """The K=1 per-step decode = each step its own forward over the frozen stream
    (no cross-row union)."""
    return KB.drive_kbatch(model, L, runner, code, K=1, seed_mem=seed,
                           max_steps=max_steps, perrow=False)[0]


@pytest.mark.parametrize("K", [2, 4, 8, 16, 32, 64])
def test_perrow_battery_byte_exact(K):
    """PER-ROW block-skip == the K=1 per-step decode, on the whole battery."""
    model, L, runner = _model()
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        ref = _perstep_ref(model, L, runner, code, seed=seed)
        pr, _ = KB.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed,
                                perrow=True)
        assert pr == ref, f"perrow K={K} {name}: ref={ref} perrow={pr}"


@pytest.mark.parametrize("K", [2, 4, 8, 16, 32, 64])
def test_perrow_graphed_battery_byte_exact(K):
    """GROUPED graphed PER-ROW (compacted same-subset FFN runs) == the K=1 per-step
    decode.  Same skip as ``perrow`` plus the launch-collapsing FFN-chain grouping."""
    model, L, runner = _model()
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        ref = _perstep_ref(model, L, runner, code, seed=seed)
        pr, _ = KB.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed,
                                perrow_graphed=True)
        assert pr == ref, f"perrow_graphed K={K} {name}: ref={ref} perrow={pr}"


@pytest.mark.parametrize("K", [4, 16, 64])
def test_perrow_deep_nested_loop(K):
    """PER-ROW and grouped-graphed PER-ROW == the K=1 per-step decode over a deep
    nested loop (the multi-batch, mixed-op stress case — a DIV-free loop where per-row
    skips the whole 179-block divmod span for every row)."""
    model, L, runner = _model()
    deep = _nested_prog(3, 4)
    ref = _perstep_ref(model, L, runner, deep, seed={}, max_steps=60)
    for perrow_kw in ({"perrow": True}, {"perrow_graphed": True}):
        pr, _ = KB.drive_kbatch(model, L, runner, deep, K=K, seed_mem={},
                                max_steps=60, **perrow_kw)
        assert pr == ref, f"{perrow_kw} K={K}: ref={ref[:20]} perrow={pr[:20]}"


def test_perrow_block_count_reduction():
    """The per-row row-mask shrinks blocks-per-row: a K-batch with one DIV pays the
    179-block divmod span ONLY on the DIV row, not on all K rows.  Union pays it on
    every row."""
    model, L, runner = _model()
    # 8 ADD steps + 1 DIV step: union = |live(ADD) U live(DIV)| ~ 188 for ALL 9 rows;
    # per-row = ADD's ~14 for the 8 ADD rows + DIV's 188 for the 1 DIV row.
    ops = [isa.ADD] * 8 + [isa.DIV]
    union = runner.live_union(ops)
    rows_by_block = runner.per_block_rows(ops)
    union_per_row = len(union)                       # every row pays the union
    perrow_avg = sum(len(rs) for rs in rows_by_block.values()) / len(ops)
    assert perrow_avg < union_per_row / 3, (perrow_avg, union_per_row)
    # the div blocks (present only for the 1 DIV row) must have subset size 1.
    div_only = [bi for bi, rs in rows_by_block.items() if rs == [8]]
    assert len(div_only) > 100, len(div_only)   # ~179 alu-div blocks, DIV-row only
