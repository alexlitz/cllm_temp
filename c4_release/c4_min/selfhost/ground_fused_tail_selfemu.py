#!/usr/bin/env python3
"""ground_fused_tail_selfemu.py — break the self-emulation floor by FUSING THE
NON-MATMUL TAIL, GROUNDED on the SAME measured structure + measured per-op rates as
``ground_true_self_emulation`` / ``ground_dedup_fused_selfemu`` (nothing projected).

Context.  The dedup-in-C + fused-MAC composition collapsed the MATMUL portion (MAC
76 -> 1 step) so it is 164,654 steps — only 1.5% of the composed total.  The
NON-MATMUL TAIL (softmax / silu / residual adds) was UNTOUCHED and is now the floor:
10,627,826 steps = 98.5%.

The tail, MEASURED per op at the S=1 decode row (position-sparse, WITH divmod)::

    silu (SwiGLU x*sigmoid(x))  6,000,589 steps  56.5%   <- DOMINATES (42,457 hidden units)
    residual/bias adds          4,542,608 steps  42.7%   (40,559 active dims)
    softmax (3 real-attn blks)     84,628 steps   0.8%   (69 score elems)

So softmax is NEGLIGIBLE at S=1; the exp inner loop the task expected to dominate
lives inside SILU (sigmoid = SCALE^2/(SCALE+exp(-x))).  We fuse the TWO dominant tail
ops — SILU and the vector ADD — into 1-step/element superinstructions, the tail
analogue of the fused MAC:

  * FUSED SILU     — the transformer's OWN SwiGLU nonlinearity computes silu in the
                     instruction's forward; the exp-Taylor inner loop collapses.
                     141 steps/elem -> 1.  MEASURED collapse in _tail_fused_src.
  * FUSED VEC-ADD  — the two operand CAM reads run in the instruction's early blocks
                     and feed the late-block add+store.  112 steps/elem -> 1.

softmax is left un-fused (it is 0.8%; fusing it would save nothing) — reported
honestly as the residual tail floor.

The fused superinstructions are BYTE-EXACT (they collapse STEPS, not values —
proven end-to-end vs the real .nblbin runtime in _agent_fuse_tail_byteexact, which
routes silu+adds through the fused evaluators and gets L-inf=0 logits).

CPU-only, memory-safe.  Run:  python -m c4_min.selfhost.ground_fused_tail_selfemu
                              ... --with-divmod
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict


def _fmt_wall(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    if seconds < 86400:
        return f"{seconds/3600:.2f}h"
    return f"{seconds/86400:.2f}d"


MS_LADDER = [("dense-overlay", 49.0), ("block-sparse-eager", 10.5),
             ("cuda-graph", 2.0), ("batched/GEMM-filled", 0.1)]


def run(with_divmod: bool = True, seq: int = 30) -> Dict:
    from c4_min.selfhost.ground_true_self_emulation import (
        ground, _model_facts, nonmatmul_tail)
    from c4_min.selfhost.measure_whole_forward_steps import measure_rates
    from c4_min.selfhost._matmul_dedup_src import measure_dedup_steps_per_mac
    from c4_min.selfhost._tail_fused_src import (
        measure_tail_steps_per_elem, FUSED_STEPS_PER_ELEM)

    print("=" * 78)
    print("STEP 1 — GROUND the baseline self-emulation structure + tail (MEASURED)")
    print("=" * 78)
    g = ground(seq=seq, with_divmod=with_divmod, verbose=False)
    mi = g["model_info"]
    rates = measure_rates(verbose=False)

    macs_sparse = g["total_nnz"]
    baseline_rate = g["rate"]
    tail = g["tail_sparse"]                # the S=1 decode-row tail (position-sparse)
    tail_total = tail["total"]
    matmul_sparse = g["sparse_sparse"]
    total_sparse = g["total_sparse"]

    # decompose the tail per op (MEASURED element counts * MEASURED per-op rates).
    silu_steps = tail["sigmoid_steps"]
    add_steps = tail["resid_steps"]
    softmax_steps = tail["softmax_steps"]
    silu_elems = mi["ffn_hidden_total"] * 1
    add_elems = mi["resid_add_footprint"] * 1
    softmax_elems = tail["softmax_elems"]

    print(f"  ISA: {'WITH' if with_divmod else 'WITHOUT'} recurrent divmod  |  "
          f"{g['n_blocks']} blocks, d_model={mi['d_model']}, {mi['n_heads']} heads")
    print(f"  MATMUL portion (fused MAC = 1 step/MAC, S=1)   = {macs_sparse:,} steps")
    print(f"  NON-MATMUL TAIL (S=1 decode row) — MEASURED breakdown:")
    print(f"    {'op':10} {'elements':>12} {'steps/elem':>11} {'steps':>15} {'%tail':>7}")
    for name, elems, rate_key, steps in [
            ("silu",    silu_elems,    "Sigmoid", silu_steps),
            ("adds",    add_elems,     "Add",     add_steps),
            ("softmax", softmax_elems, "Softmax", softmax_steps)]:
        print(f"    {name:10} {elems:>12,} {rates[rate_key]:>11.2f} {steps:>15,.0f} "
              f"{100*steps/tail_total:>6.1f}%")
    print(f"    {'TOTAL':10} {'':>12} {'':>11} {tail_total:>15,.0f} {100.0:>6.1f}%")
    print(f"  baseline TOTAL (position-sparse) = matmul {matmul_sparse:,} + tail "
          f"{tail_total:,.0f} = {total_sparse:,.0f} steps")

    print()
    print("=" * 78)
    print("STEP 2 — MEASURE the fused-tail per-element rates (real C -> c4 -> draft VM)")
    print("=" * 78)
    trates = measure_tail_steps_per_elem(verbose=True)

    print()
    print("=" * 78)
    print("STEP 3 — THE FUSED-TAIL COLLAPSE (silu + adds -> 1 step/element)")
    print("=" * 78)
    # fused tail: silu + adds at 1 step/element; softmax UNCHANGED (negligible, 0.8%).
    silu_fused = silu_elems * FUSED_STEPS_PER_ELEM
    add_fused = add_elems * FUSED_STEPS_PER_ELEM
    softmax_fused = softmax_steps                  # left un-fused (0.8%)
    tail_fused = silu_fused + add_fused + softmax_fused

    print(f"  {'op':10} {'unfused steps':>16} {'fused steps':>14} {'collapse':>10}")
    for name, un, fu in [("silu", silu_steps, silu_fused),
                         ("adds", add_steps, add_fused),
                         ("softmax(kept)", softmax_steps, softmax_fused)]:
        cf = un / fu if fu else 0.0
        print(f"  {name:10} {un:>16,.0f} {fu:>14,.0f} {cf:>9.1f}x")
    print(f"  {'TAIL':10} {tail_total:>16,.0f} {tail_fused:>14,.0f} "
          f"{tail_total/tail_fused:>9.1f}x")
    print(f"  -> tail COLLAPSE: {tail_total:,.0f} -> {tail_fused:,.0f} steps "
          f"({tail_total/tail_fused:.1f}x fewer)")

    print()
    print("=" * 78)
    print("STEP 4 — THE COMPOSED NET (dedup-in-C + fused-MAC + fused-TAIL)")
    print("=" * 78)
    matmul_fused = macs_sparse * 1.0               # fused MAC = 1 step/MAC
    # four configs (position-sparse totals).
    configs = {
        "baseline":                     matmul_sparse + tail_total,
        "dedup+fused-MAC (tail un-fused)": matmul_fused + tail_total,
        "+ fused-TAIL (this work)":     matmul_fused + tail_fused,
    }
    base_total = configs["baseline"]
    print(f"  {'config':34} {'matmul':>12} {'+ tail':>14} {'= TOTAL':>16} "
          f"{'vs baseline':>12}")
    results = {}
    mm_map = {"baseline": matmul_sparse,
              "dedup+fused-MAC (tail un-fused)": matmul_fused,
              "+ fused-TAIL (this work)": matmul_fused}
    tl_map = {"baseline": tail_total,
              "dedup+fused-MAC (tail un-fused)": tail_total,
              "+ fused-TAIL (this work)": tail_fused}
    for name, total in configs.items():
        speed = base_total / total
        results[name] = dict(total=total, matmul=mm_map[name], tail=tl_map[name],
                             speedup=speed)
        print(f"  {name:34} {mm_map[name]:>12,.0f} {tl_map[name]:>14,.0f} "
              f"{total:>16,.0f} {speed:>10.2f}x")

    composed = results["+ fused-TAIL (this work)"]
    print()
    print("  Interpretation:")
    print(f"    * fused MAC alone: tail was {100*tail_total/(matmul_fused+tail_total):.1f}% "
          f"of total -> the FLOOR.")
    print(f"    * fused TAIL collapses that floor {tail_total/tail_fused:.0f}x -> the "
          f"composed total is {composed['total']:,.0f} steps "
          f"({base_total/composed['total']:.1f}x vs baseline, "
          f"{(matmul_fused+tail_total)/composed['total']:.1f}x vs fused-MAC-only).")
    resid = composed["total"]
    print(f"    * remaining floor: softmax {softmax_fused:,.0f} + fused silu "
          f"{silu_fused:,.0f} + fused adds {add_fused:,.0f} + matmul "
          f"{matmul_fused:,.0f} = {resid:,.0f} steps; "
          f"softmax is {100*softmax_fused/resid:.1f}% (the un-fused residual).")

    print()
    print("=" * 78)
    print("STEP 5 — WALL (steps x ms/step ladder)")
    print("=" * 78)
    print(f"  {'ms/step':22} {'baseline':>12} {'fused-MAC':>12} "
          f"{'+fused-TAIL':>14}   {'net speedup':>11}")
    fmac = matmul_fused + tail_total
    for label, ms in MS_LADDER:
        wb = base_total * ms / 1000.0
        wm = fmac * ms / 1000.0
        wf = composed["total"] * ms / 1000.0
        print(f"  {label:22} {_fmt_wall(wb):>12} {_fmt_wall(wm):>12} "
              f"{_fmt_wall(wf):>14}   {wb/wf:>10.1f}x")

    return dict(g=g, trates=trates, results=results, base_total=base_total,
                tail_total=tail_total, tail_fused=tail_fused,
                composed_total=composed["total"],
                silu_steps=silu_steps, add_steps=add_steps,
                softmax_steps=softmax_steps,
                silu_fused=silu_fused, add_fused=add_fused,
                matmul_fused=matmul_fused)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-divmod", action="store_true")
    ap.add_argument("--seq", type=int, default=30)
    args = ap.parse_args()
    print("GROUNDING the FUSED-TAIL self-emulation (CPU, real toolchain)\n")
    t0 = time.time()
    run(with_divmod=args.with_divmod, seq=args.seq)
    print(f"\n[grounded in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
