"""MEASURE the top-1 dispatch routing: per-step FFN FLOP/unit reduction (dense
blend -> 1 expert) + honest end-to-end per-step wall-time.

Reports:
  1. Dispatch-block FFN compute BEFORE (all opcodes' units) vs AFTER (active
     opcode's units): unit count + FLOP (2*units*dim for up + 2*units*dim for
     gate + 2*dim*units for down) reduction factor.  Both the COMPACT model's
     ragged dispatch and the *padded complete* model's global-max dispatch are
     reported (the complete model's dispatch is padded to the 21544 global-max,
     so its dense->routed unit reduction is the eye-popping one).
  2. End-to-end per-step wall-time on a representative program, dense vs
     top-1-routed.  The dispatch is ONE block of a ~42-block stack, so the honest
     whole-step speedup is far smaller than the per-block factor -- reported as
     measured.
"""
from __future__ import annotations
import sys, time

import c4_min.nibble_pure_forward as PF
import c4_min.nibble_pure_forward_complete as C
PF.SP_INIT = 0xF0
C.SP_INIT = 0xF0

import torch  # noqa: E402
from c4_min import isa  # noqa: E402
from c4_min.compact_alloc import build_compact_pure_forward_model  # noqa: E402
from c4_min.nibble_pure_forward_complete import (  # noqa: E402
    build_pure_forward_complete_model)
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached  # noqa: E402
from c4_min.moe_top1 import Top1RoutedFFN  # noqa: E402


def _ffn_flops(units: int, dim: int) -> int:
    """SwiGLU FFN FLOPs for ``units`` hidden units at width ``dim``:
    up (2*units*dim) + gate (2*units*dim) + down (2*dim*units)."""
    return 6 * units * dim


def dispatch_stats(model, L, label: str):
    names = list(L._block_names)
    bi = names.index("dispatch")
    dense = model.blocks[bi].ffn
    H = dense.W_up.shape[0]
    dim = dense.W_up.shape[1]
    routed = Top1RoutedFFN(dense, op_is_base=int(L.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    K = routed.K + routed.n_ungated                  # units actually computed/step
    A = routed.n_active_units                        # non-zero (real MoE) units
    dense_fl = _ffn_flops(H, dim)                    # what the dense matmul actually costs
    active_fl = _ffn_flops(A, dim)                   # the real experts (drops padding)
    routed_fl = _ffn_flops(K, dim)
    print(f"[{label}] dispatch FFN: dim={dim}")
    print(f"    dense units/step (all rows incl padding): {H:>6d}   "
          f"FLOPs/step {dense_fl:>12,d}")
    print(f"    dense units/step (real experts, nonzero): {A:>6d}   "
          f"FLOPs/step {active_fl:>12,d}")
    print(f"    routed units/step (active opcode only)  : {K:>6d}   "
          f"FLOPs/step {routed_fl:>12,d}  (+{routed.n_ungated} ungated)")
    print(f"    UNIT reduction vs real experts : {A / K:.1f}x   "
          f"vs padded-dense {H / K:.1f}x")
    print(f"    FLOP reduction vs real experts : {active_fl / routed_fl:.1f}x   "
          f"vs padded-dense {dense_fl / routed_fl:.1f}x")
    return bi, dense, routed


def bench_step(model, L, code, n_steps_prog: int = 5, repeats: int = 20):
    """Median per-step wall-time via the KV-cached driver over a short program."""
    # warm up
    run_pure_forward_cached(model, L, code, max_steps=n_steps_prog, evict=False)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_pure_forward_cached(model, L, code, max_steps=n_steps_prog, evict=False)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    return med / n_steps_prog          # per-step


def main():
    torch.set_num_threads(4)
    print("=" * 74)
    print("TOP-1 DISPATCH ROUTING — FLOP/unit reduction + honest wall-time")
    print("=" * 74)

    # -- COMPACT model (the corpus/ONNX target; dispatch already de-padded). ----
    t = time.time()
    cm, cL, _ = build_compact_pure_forward_model(
        code_size=24)
    print(f"\ncompact model built ({time.time()-t:.1f}s): "
          f"dim={cm.dim} blocks={len(cm.blocks)}")
    dispatch_stats(cm, cL, "COMPACT")

    # -- COMPLETE model (dispatch padded to global-max -> huge dense->routed). --
    # This measurement's POINT is the padded-dense cost, so it needs the DENSE
    # build (peak 54-108 GB RSS) — OPT-IN via C4_ALLOW_DENSE_BUILD=1.
    from c4_min._build_guard import dense_build_allowed
    if not dense_build_allowed():
        print("\n[skip] COMPLETE(padded) dense-cost measurement — the dense "
              "build peaks at 54-108 GB RSS.  Re-run with C4_ALLOW_DENSE_BUILD=1 "
              "to include it.")
        return 0
    t = time.time()
    xm, xL = build_pure_forward_complete_model(
        code_size=24)
    print(f"\ncomplete (padded) model built ({time.time()-t:.1f}s): "
          f"dim={xL.D} blocks={len(xm.blocks)}")
    dispatch_stats(xm, xL, "COMPLETE(padded)")

    # -- End-to-end per-step wall-time (COMPACT), dense vs routed. --------------
    print("\n" + "-" * 74)
    print("END-TO-END per-step wall-time (compact, KV-cached driver)")
    prog = [("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("HALT", 0)]
    code = isa.assemble(prog)
    names = list(cL._block_names)
    bi = names.index("dispatch")

    # ensure dense
    dense_per_step = bench_step(cm, cL, code)
    print(f"    dense  per-step: {dense_per_step*1e3:8.2f} ms")

    # route the dispatch block
    routed = Top1RoutedFFN(cm.blocks[bi].ffn, op_is_base=int(cL.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    cm.blocks[bi].ffn = routed
    routed_per_step = bench_step(cm, cL, code)
    print(f"    routed per-step: {routed_per_step*1e3:8.2f} ms")
    print(f"    whole-step speedup: {dense_per_step / routed_per_step:.2f}x  "
          f"(honest: dispatch is 1 of {len(cm.blocks)} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
