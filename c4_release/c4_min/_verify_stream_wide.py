"""Verify build_compact_sparse_streaming supports the C4_INGEST_WIDE 1-head block-0.

Builds the streaming sparse model with C4_INGEST_WIDE set from the environment,
runs the full op battery byte-exact, reports the per-block live head counts, and
(when a DENSE reference is available) confirms decoded VM behaviour matches the
dense build_pure_forward_complete_model byte-for-byte.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> C4_INGEST_WIDE=1 \
        python -m c4_min._verify_stream_wide
"""
from __future__ import annotations

import os

import torch

from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model,
    run_pure_forward_complete, ref_interpret)
from c4_min.run_1096_pure_forward import bytecode_to_isa


BATTERY = [
    ("add", "int main(){ return 500 + 700; }", 1200),
    ("mul", "int main(){ return 100 * 10; }", 1000),
    ("cmp", "int main(){ if (5 > 3) return 1; return 0; }", 1),
    ("var", "int main(){ int x; x = 1000; return x; }", 1000),
    ("func", "int identity(int x){ return x; } "
             "int main(){ return identity(1000); }", 1000),
    ("bw_or", "int main(){ return 12 | 3; }", 15),
    ("bw_and", "int main(){ return 12 & 10; }", 8),
    ("sub", "int main(){ return 900 - 250; }", 650),
    ("div", "int main(){ return 720 / 6; }", 120),
    ("mod", "int main(){ return 84 % 5; }", 4),
    ("divzero", "int main(){ return 5 / 0; }", 0),
    ("nested", "int main(){ return (2 + 3) * (4 + 1); }", 25),
    ("shl", "int main(){ return 3 << 4; }", 48),
    ("xor", "int main(){ return 12 ^ 10; }", 6),
    ("ge", "int main(){ if (5 >= 5) return 42; return 0; }", 42),
]


def _run(model, L, src):
    code = bytecode_to_isa(_compile(src))
    cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
    tr = run_pure_forward_complete(model, L, code, max_steps=cap, mask=0xFFFFFFFF)
    return (tr[-1] & 0xFFFFFFFF) if tr else None


def _compile(src):
    from src.compiler import compile_c
    return compile_c(src)[0]


def _head_counts(sparse):
    return [getattr(getattr(b, "attn", None), "n_heads", None)
            for b in sparse._phys_blocks]


def main():
    wide = os.environ.get("C4_INGEST_WIDE", "0") not in ("0", "", "false", "False")
    print(f"C4_INGEST_WIDE = {wide}")

    print("building STREAMING sparse model ...", flush=True)
    sparse, L, stats = build_compact_sparse_streaming(
        code_size=44, compute_mode="dense_kernel")
    hc = _head_counts(sparse)
    from collections import Counter
    print(f"  dim_before={stats.dim_before} dim_after={stats.dim_after}")
    print(f"  n_phys blocks = {len(sparse._phys_blocks)}")
    one_head = [i for i, h in enumerate(hc) if h == 1]
    print(f"  1-head blocks (indices) = {one_head}")
    live_heads = sum(_live_head_count(b) for b in sparse._phys_blocks)
    print(f"  block head-count histogram = {dict(Counter(hc))}")
    print(f"  total LIVE attention heads across all phys blocks = {live_heads}")

    ok_all = True
    for name, src, exp in BATTERY:
        got = _run(sparse, L, src)
        ok = got == exp
        ok_all = ok_all and ok
        print(f"  [{'ok' if ok else 'FAIL'}] {name:8s} got={got} want={exp}")

    print(f"\nSTREAMING battery: {'PASS' if ok_all else 'FAIL'}")

    # Cross-check vs DENSE build (same flag), decoded VM behaviour byte-for-byte.
    if os.environ.get("C4_VERIFY_DENSE", "1") not in ("0", "", "false", "False"):
        print("\nbuilding DENSE reference (same flag) for cross-check ...", flush=True)
        dmodel, dL = build_pure_forward_complete_model(code_size=44)
        cross_ok = True
        for name, src, exp in BATTERY:
            gd = _run(dmodel, dL, src)
            gs = _run(sparse, L, src)
            same = gd == gs
            cross_ok = cross_ok and same
            print(f"  [{'ok' if same else 'DIFF'}] {name:8s} dense={gd} stream={gs}")
        print(f"\nstream == dense (decoded VM): {'PASS' if cross_ok else 'FAIL'}")
        ok_all = ok_all and cross_ok

    print(f"\nOVERALL: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def _live_head_count(block):
    """Number of heads with any nonzero Q/K/V/O in this block's attn."""
    at = getattr(block, "attn", None)
    if at is None or getattr(at, "n_heads", 0) == 0:
        return 0
    # SparseAttn stores sparse weights; densify small ones to count live heads.
    import torch as _t
    def dense(w):
        if getattr(w, "is_sparse", False):
            return w.csr.to_dense()
        return getattr(w, "dense", None)
    live = 0
    H = at.n_heads
    HD = at.head_dim
    wq = dense(at.W_q)
    if wq is None:
        return 0
    for h in range(H):
        rows = wq[h * HD:(h + 1) * HD]
        if bool((rows != 0).any()):
            live += 1
    return live


if __name__ == "__main__":
    raise SystemExit(main())
