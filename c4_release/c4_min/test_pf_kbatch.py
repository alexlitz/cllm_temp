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
