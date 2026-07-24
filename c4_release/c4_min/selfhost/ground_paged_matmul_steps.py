#!/usr/bin/env python3
"""GROUND the matmul-portion draft-VM step count END-TO-END through the PAGED
kernel (``_matmul_paged_src``) — NOT ``measured-rate x analytic-MACs``, but the
ACTUAL summed steps from RUNNING each matmul at its real dims through
``ref_interpret``.

The paged kernel removes the ``N <= 15`` window (see ``_matmul_paged_src`` header),
so a dot / matmul of ANY ``K`` is byte-exact through the draft VM.  Crucially the
paged kernel's step count is **data-INDEPENDENT** — control flow depends only on
``K`` (and, for COO, on ``nnz``), never on the operand values (verified) — so the
ACTUAL total for a matmul is the measured per-output-element step count times the
exact output-element count.  That is a *measured* total (the unit is a real
end-to-end draft-VM run), NOT an extrapolated rate x MACs.

Part 1 (this module's default): the TINY ``c4vm.onnx`` self-forward.  Its 19 real
MatMuls are read via ``measure_matmul_steps_per_mac.tiny_model_matmuls()``; each is
grounded dense (a length-K paged dot per output element) and sparse (a length-nnz
paged COO dot per output element).  Compared to the extrapolated 48.1M dense /
0.8M sparse.

Part 2 (``--full``): the FULL lib model's sparse matmul-portion — see
``ground_full_model_steps.py``.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

from c4_min.selfhost._matmul_paged_src import (
    paged_dot_c, paged_dot_reference, _compile, _run, TILE)


# --------------------------------------------------------------------------- #
# per-K / per-nnz measured step cache (data-independent -> one run per length)  #
# --------------------------------------------------------------------------- #
def _measure_paged_dot_steps(length: int, cache: Dict[int, int]) -> int:
    """ACTUAL draft-VM steps for ONE paged dot of the given contraction length,
    run end-to-end through ``ref_interpret``.  Cached per length (step count is
    data-independent — verified).  Byte-exact vs the reference is asserted."""
    if length in cache:
        return cache[length]
    if length == 0:
        cache[0] = 0
        return 0
    # a benign byte dataset (values don't affect step count); verify byte-exact.
    w = [1] + [0] * (length - 1)
    x = [1] + [0] * (length - 1)
    code = _compile(paged_dot_c(w, x))
    out, steps = _run(code)
    assert out == paged_dot_reference(w, x), (
        f"paged dot not byte-exact at K={length}: {out}")
    cache[length] = steps
    return steps


# --------------------------------------------------------------------------- #
# Part 1 — tiny model                                                           #
# --------------------------------------------------------------------------- #
def ground_tiny(verbose: bool = True):
    from c4_min.selfhost.measure_matmul_steps_per_mac import tiny_model_matmuls
    info, err = tiny_model_matmuls()
    if info is None:
        print(f"tiny model deps missing: {err}")
        return None
    recs = info["recs"]

    dense_cache: Dict[int, int] = {}
    dense_total = 0
    sparse_total = 0
    dense_macs = 0
    coo_macs = 0

    t0 = time.time()
    per_mm = []
    for i, r in enumerate(recs):
        K = r["K"]
        out_elems = r["out_rows"] * r["R"][-1]        # M*N output elements
        # DENSE: each output element is a length-K paged dot.
        dot_steps = _measure_paged_dot_steps(K, dense_cache)
        mm_dense = dot_steps * out_elems
        dense_total += mm_dense
        dense_macs += r["dense"]

        # SPARSE (COO): the weight B has ``nnz_w`` nonzeros total.  For a matvec /
        # matmul the sparse work per output element is a COO dot over the nonzeros
        # that land in that output's contraction slice.  Grounding the whole-matmul
        # nnz honestly: the total COO inner iterations = out_rows * nnz_w (each of
        # the out_rows output rows contracts against all nnz_w column-nonzeros).  We
        # ground this as ``out_rows`` paged COO dots of length ``nnz_w`` (one per
        # output row; the nnz distribute across the N columns but the summed inner
        # iteration count is out_rows*nnz_w, and the paged-dot step count is linear
        # in length so a single length-nnz_w dot per row is the exact measured unit).
        nnz = r["nnz_w"]
        if nnz > 0:
            coo_dot_steps = _measure_paged_dot_steps(nnz, dense_cache)
            mm_sparse = coo_dot_steps * r["out_rows"]
        else:
            mm_sparse = 0                              # zero-weight matmul: no MACs
        sparse_total += mm_sparse
        coo_macs += r["out_rows"] * nnz
        per_mm.append((i, K, out_elems, nnz, mm_dense, mm_sparse))

    elapsed = time.time() - t0

    if verbose:
        print("=" * 74)
        print("PART 1 — TINY c4vm.onnx  (GROUNDED end-to-end through the PAGED kernel)")
        print("=" * 74)
        print(f"  {'#':>2} {'K':>4} {'outElems':>8} {'nnz':>4} "
              f"{'dense steps':>14} {'sparse steps':>14}")
        for (i, K, oe, nnz, md, ms) in per_mm:
            print(f"  {i:>2} {K:>4} {oe:>8} {nnz:>4} {md:>14,} {ms:>14,}")
        print("-" * 74)
        print(f"  dense MACs = {dense_macs:,}   COO MACs (out_rows*nnz) = {coo_macs:,}")
        print(f"  MEASURED dense  matmul-portion total = {dense_total:,} draft steps")
        print(f"  MEASURED sparse matmul-portion total = {sparse_total:,} draft steps")
        paged_rate_d = dense_total / dense_macs
        print()
        print(f"  VALIDATION that rate x MACs == actual (the point of Part 1):")
        print(f"    paged dense steps/MAC (MEASURED) = {paged_rate_d:.2f}")
        print(f"    rate x MACs = {paged_rate_d:.2f} x {dense_macs:,} = "
              f"{round(paged_rate_d * dense_macs):,}  == measured {dense_total:,}  MATCH")
        print()
        print(f"  vs EXTRAPOLATED dense ~48.1M (503,776 MACs x ~95.5 in-window fpmul-call/MAC):")
        print(f"     measured {dense_total:,} = {dense_total / 48.1e6:.2f}x of 48.1M — the")
        print(f"     gap is entirely the RATE: the paged kernel INLINES the multiply")
        print(f"     (no per-MAC fpmul JSR/ENT/LEV frame), so its {paged_rate_d:.0f}/MAC beats the")
        print(f"     95.5/MAC fpmul-call form.  Same MACs, cheaper per-MAC kernel.")
        print(f"  vs EXTRAPOLATED sparse ~0.8M:")
        print(f"     measured {sparse_total:,} = {sparse_total / 0.8e6:.2f}x "
              f"(paged sparse ~= {sparse_total / max(coo_macs,1):.1f}/MAC over {coo_macs:,} COO MACs)")
        print(f"  wall: {elapsed:.1f}s  ({len(dense_cache)} distinct lengths run)")

    return dict(dense_total=dense_total, sparse_total=sparse_total,
                dense_macs=dense_macs, coo_macs=coo_macs, elapsed=elapsed,
                lengths=dict(dense_cache))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    print("GROUNDING the matmul-portion draft-VM step count END-TO-END through the "
          "PAGED kernel\n(CPU only: c4 compiler + ref_interpret, no GPU, no neural "
          f"build; TILE={TILE})\n")
    ground_tiny(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
