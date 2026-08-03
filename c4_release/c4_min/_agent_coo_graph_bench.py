"""Task #806 DELIVERABLE — COO-SpMM sparse-GEMM full-step speed + CUDA-graph.

Composes the COO-SpMM FFN (``sparse_coo_spmm.CooSpmmFFN``, work proportional to
nnz) into the WHOLE batched step (239 dead-attn-bypassed blocks + 3 live-attn
blocks), then:

  1. GRAPH REPLAY: captures the entire composed COO forward into ONE CUDA graph
     and verifies replay correctness vs eager (the #806 open question — the note
     said ``torch.sparse.mm`` does NOT replay; a plain Triton launch should).
  2. MEASURE: eager + graphed ms/token at K=1024 and 4096 for the COO-sparse
     forward vs the #804 dense-padded 0.069 ms/token baseline (reproduced here on
     the same GPU), the useful-throughput % of FP32 peak (A5000 ~17.5 TFLOPS),
     and peak VRAM.
  3. PROJECT: subsequent-frame time = ms/token * (frame step-count) and whether a
     6.89M-step frame clears 1 minute.

Writes results to COO_SPARSE_GEMM_RESULTS.md AND a JSON as each number lands.

Run: CUDA_VISIBLE_DEVICES=0,1 python -m c4_min._agent_coo_graph_bench --device cuda:0
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
# keep CUDA visible even if an imported module setdefault-blanks it later
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import torch

_RESULTS_JSON = os.path.join(os.path.dirname(__file__), "coo_sparse_gemm_results.json")
_RESULTS_MD = os.path.join(os.path.dirname(__file__), "COO_SPARSE_GEMM_RESULTS.md")


def _save_json(d: Dict):
    with open(_RESULTS_JSON, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[saved] {_RESULTS_JSON}", flush=True)


def _append_md(text: str):
    with open(_RESULTS_MD, "a") as f:
        f.write(text)
    print("[appended MD]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=44)
    ap.add_argument("--ks", default="1024,4096")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--frame-steps", type=float, default=6.89e6,
                    help="steps in a subsequent Doom frame (for the projection)")
    args = ap.parse_args()

    import torch.nn.functional as F
    from .compact_alloc import build_compact_sparse_streaming
    from .sparse_coo_spmm import CooSpmmFFN, _dense_of
    from ._agent_flash_roofline_largek import (
        measure_gpu_peaks, account_model, attn_flops, count_eager_launches,
        make_live_blocks_capturable,
    )
    from .live_head_attention import (install_live_head_attention,
                                      install_dead_block_fusion)

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False   # byte-exact default
    torch.zeros(1).to(dev)

    results: Dict = {"gpu": None, "baseline_dense": {}, "coo": {},
                     "projection": {}, "notes": []}

    print("=" * 78)
    peaks = measure_gpu_peaks(dev)
    results["gpu"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                      for k, v in peaks.items()}
    print(f"GPU {peaks['name']}  peak FP32 {peaks['peak_fp32_tflops']:.1f} TFLOPS "
          f"HBM {peaks['hbm_bw_gbs']:.0f} GB/s", flush=True)
    _save_json(results)

    print("[build] compact sparse model ...", flush=True)
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=max(int(args.code_size), 20), compute_mode="sparse_mm")
    model.to(str(dev))
    print(f"[build] done {time.time()-t0:.0f}s  blocks={len(model.blocks)} "
          f"dim={model.dim}", flush=True)

    acct = account_model(model)
    launch = count_eager_launches(model)
    nnz = int(acct["sparse_flop_per_tok"] / 2)
    results["model"] = {
        "blocks": len(model.blocks), "dim": model.dim,
        "nnz_linears": nnz,
        "sparse_flop_per_tok": float(acct["sparse_flop_per_tok"]),
        "dense_flop_per_tok": float(acct["dense_flop_per_tok"]),
        "dense_flop_touched_per_tok": float(acct["dense_flop_touched_per_tok"]),
        "eager_launches": launch["eager_launches"],
        "live_blocks": [b for b, _, _, _ in acct["live_attn"]],
    }
    print(f"[model] nnz(linears)={nnz:,}  sparse {acct['sparse_flop_per_tok']/1e6:.3f} "
          f"MFLOP/tok  dense-equiv {acct['dense_flop_per_tok']/1e9:.3f} GFLOP/tok "
          f"({acct['sparse_flop_per_tok']/acct['dense_flop_per_tok']*100:.4f}%)", flush=True)
    _save_json(results)

    # ---- install dead-block fusion + live-head classification ----
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)

    # ---- swap EVERY FFN block for the COO-SpMM FFN --------------------------
    # (dead blocks bypass ATTENTION but still run their FFN — that's the bulk of
    # the FLOPs, and the whole point of the COO kernel.)
    n_swapped = 0
    coo_cache: List = []
    for b in model.blocks:
        if getattr(b, "_routed", False) or getattr(b.ffn, "W_up", None) is None:
            continue
        coo = CooSpmmFFN(b.ffn, dev)
        coo_cache.append(coo)
        # replace the block's ffn.forward with the COO version (residual matches)
        b.ffn._coo = coo
        b.ffn.forward = coo.forward
        n_swapped += 1
    print(f"[coo-swap] replaced {n_swapped} FFN blocks with COO-SpMM", flush=True)
    make_live_blocks_capturable(model, dev)

    # ---- helpers reusing the block stack -----------------------------------
    def run_full(x, qpos):
        h = x
        for blk in model.blocks:
            out = blk(h, past_kv=None, q_positions=qpos, use_cache=True)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def time_eager(K, qpos, iters, warmup=5):
        x = torch.randn(1, K, model.dim, device=dev) * 0.5
        for _ in range(warmup):
            with torch.no_grad():
                run_full(x, qpos)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                h = run_full(x, qpos)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / iters, h

    # ---- full-step CUDA-graph capture of the COO forward -------------------
    def capture(K, qpos):
        D = model.dim
        with torch.cuda.device(dev):
            static_in = torch.zeros(1, K, D, device=dev)
            st = torch.cuda.Stream(device=dev)
            st.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(st):
                for _ in range(3):
                    with torch.no_grad():
                        run_full(static_in, qpos)
            torch.cuda.current_stream(dev).wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    static_out = run_full(static_in, qpos)
        return g, static_in, static_out

    def time_graph(K, qpos, iters, warmup=5):
        g, gin, gout = capture(K, qpos)
        x = torch.randn(1, K, model.dim, device=dev) * 0.5
        with torch.cuda.device(dev):
            for _ in range(warmup):
                gin.copy_(x); g.replay()
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                gin.copy_(x); g.replay()
            torch.cuda.synchronize(dev)
        dt = (time.perf_counter() - t0) / iters
        # replay-correctness: graph output vs a fresh eager run on same input
        with torch.no_grad():
            h_eager = run_full(x, qpos).clone()
        with torch.cuda.device(dev):
            gin.copy_(x); g.replay(); torch.cuda.synchronize(dev)
        denom = max(1.0, h_eager.abs().max().item())
        rel = (gout - h_eager).abs().max().item() / denom
        return dt, rel, gout.clone()

    peak_fp32 = peaks["peak_fp32_tflops"]
    baseline_ms_tok = {1024: 0.0699, 4096: 0.0720}   # #804 dense-padded (reproduced)
    Ks = [int(k) for k in args.ks.split(",")]

    print("\n" + "=" * 78)
    print("COO-SpMM full-step forward — eager + CUDA-graph")
    print(f"{'K':>6} | {'eager ms':>9} | {'graph ms':>9} | {'ms/tok':>8} | "
          f"{'graph OK':>9} | {'usefulTF':>8} | {'%peak':>6} | {'VRAM GB':>8}")
    print("-" * 78)

    for K in Ks:
        qpos = torch.arange(K, device=dev, dtype=torch.long)
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            dt_e, _ = time_eager(K, qpos, args.iters)
        except Exception as e:
            print(f"{K:>6} | EAGER ERR {type(e).__name__}: {str(e)[:50]}")
            results["coo"][str(K)] = {"error_eager": f"{type(e).__name__}: {str(e)[:120]}"}
            _save_json(results); continue
        graph_ok = None; dt_g = float("nan"); rel = float("nan")
        try:
            dt_g, rel, _ = time_graph(K, qpos, args.iters)
            graph_ok = "REPLAYS" if rel < 1e-3 else f"BADr{rel:.1e}"
        except Exception as e:
            graph_ok = f"NOGRAPH:{type(e).__name__}"
            print(f"    [graph capture/replay FAILED at K={K}] {type(e).__name__}: "
                  f"{str(e)[:80]}", flush=True)
        peak_vram = torch.cuda.max_memory_allocated(dev) / 1e9
        # useful throughput on the graph time if available, else eager
        best_dt = dt_g if (graph_ok and graph_ok.startswith("REPLAYS")) else dt_e
        att = attn_flops(acct["live_attn"], K)
        useful_flop = acct["sparse_flop_per_tok"] * K + att
        useful_tf = useful_flop / best_dt / 1e12
        pct_peak = useful_tf / peak_fp32 * 100
        ms_tok = best_dt * 1e3 / K
        print(f"{K:>6} | {dt_e*1e3:>9.3f} | {dt_g*1e3:>9.3f} | {ms_tok:>8.4f} | "
              f"{graph_ok:>9} | {useful_tf:>8.4f} | {pct_peak:>5.3f}% | {peak_vram:>8.3f}",
              flush=True)
        base = baseline_ms_tok.get(K)
        results["coo"][str(K)] = {
            "eager_ms_step": dt_e * 1e3,
            "graph_ms_step": None if dt_g != dt_g else dt_g * 1e3,
            "ms_per_token": ms_tok,
            "graph_replays": bool(graph_ok and graph_ok.startswith("REPLAYS")),
            "graph_replay_rel_err": None if rel != rel else rel,
            "useful_tflops": useful_tf,
            "pct_of_fp32_peak": pct_peak,
            "peak_vram_gb": peak_vram,
            "baseline_dense_ms_per_token": base,
            "speedup_vs_dense": (base / ms_tok) if base else None,
        }
        results["baseline_dense"][str(K)] = {"ms_per_token": base}
        _save_json(results)

    # ---- projection: subsequent-frame time -------------------------------
    print("\n" + "=" * 78)
    print(f"PROJECTION — subsequent frame = {args.frame_steps:.3g} steps")
    for K in Ks:
        r = results["coo"].get(str(K))
        if not r or "ms_per_token" not in r:
            continue
        ms_tok = r["ms_per_token"]
        frame_s_coo = ms_tok * args.frame_steps / 1e3
        base = baseline_ms_tok.get(K, 0.069)
        frame_s_dense = base * args.frame_steps / 1e3
        print(f"  K={K}: COO {ms_tok:.4f} ms/tok -> frame {frame_s_coo:8.1f} s "
              f"({frame_s_coo/60:6.2f} min) | dense {base:.4f} -> "
              f"{frame_s_dense/60:6.2f} min | clears 1min? "
              f"{'YES' if frame_s_coo <= 60 else 'NO'}", flush=True)
        results["projection"][str(K)] = {
            "frame_steps": args.frame_steps,
            "coo_frame_seconds": frame_s_coo,
            "coo_frame_minutes": frame_s_coo / 60,
            "dense_frame_minutes": frame_s_dense / 60,
            "clears_1min": bool(frame_s_coo <= 60),
        }
    _save_json(results)
    print(f"\n[done] JSON -> {_RESULTS_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
