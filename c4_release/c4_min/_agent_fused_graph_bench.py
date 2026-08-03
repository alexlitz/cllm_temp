"""#806 follow-up DELIVERABLE — FUSED segmented sparse-FFN full-step speed +
CUDA-graph, vs the COO 0.0328 and the dense 0.069 ms/token baselines.

Swaps every FFN block for the fused kernel (up+gate+silu fused + down w/ fused
residual, 2 launches/block; OR the single-launch FusedFullFFN), composes the
whole batched step (239 dead-attn-bypassed + 3 live-attn blocks), captures into
ONE CUDA graph, and measures eager + graphed ms/token at K=1024 and 4096.

Also prints per-form comparison and the frame projection (x 6.89M steps).

Run: CUDA_VISIBLE_DEVICES=0,1 python -m c4_min._agent_fused_graph_bench --device cuda:0
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import torch

_HERE = os.path.dirname(__file__)
_RESULTS_JSON = os.path.join(_HERE, "_fused_results", "fused_sparse_gemm_results.json")


def _save_json(d: Dict):
    os.makedirs(os.path.dirname(_RESULTS_JSON), exist_ok=True)
    with open(_RESULTS_JSON, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[saved] {_RESULTS_JSON}", flush=True)


def _swap_ffn(model, dev, form: str, block_k: int):
    from .fused_sparse_ffn import (FusedUpGateSiluFFN, FusedFullFFN, HybridFusedFFN,
                                   FusedUpGateSiluDeltaFFN)
    from .sparse_coo_spmm import CooSpmmFFN
    cache: List = []
    n = 0
    # each block keeps its ORIGINAL ffn to rebuild from (avoid re-wrapping a wrapper)
    for b in model.blocks:
        if getattr(b, "_routed", False):
            continue
        base = getattr(b, "_orig_ffn", None) or b.ffn
        if getattr(base, "W_up", None) is None:
            continue
        b._orig_ffn = base
        if form == "coo":
            k = CooSpmmFFN(base, dev)
        elif form == "upgate":
            k = FusedUpGateSiluFFN(base, dev, block_k=block_k)
        elif form == "delta":
            k = FusedUpGateSiluDeltaFFN(base, dev, block_k=block_k)
        elif form == "full":
            k = FusedFullFFN(base, dev, block_k=block_k)
        elif form == "hybrid":
            k = HybridFusedFFN(base, dev, block_k=block_k)
        else:
            raise ValueError(form)
        cache.append(k)
        b.ffn = base
        b.ffn._fused = k
        b.ffn.forward = k.forward
        n += 1
    return n, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=44)
    ap.add_argument("--ks", default="1024,4096")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--forms", default="coo,upgate,full")
    ap.add_argument("--block-k", type=int, default=128)
    ap.add_argument("--frame-steps", type=float, default=6.89e6)
    args = ap.parse_args()

    from .compact_alloc import build_compact_sparse_streaming
    from ._agent_flash_roofline_largek import (
        measure_gpu_peaks, account_model, attn_flops, count_eager_launches,
        make_live_blocks_capturable,
    )
    from .live_head_attention import (install_live_head_attention,
                                      install_dead_block_fusion)

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.zeros(1).to(dev)

    peaks = measure_gpu_peaks(dev)
    peak_fp32 = peaks["peak_fp32_tflops"]
    print(f"GPU {peaks['name']}  peak FP32 {peak_fp32:.1f} TFLOPS "
          f"HBM {peaks['hbm_bw_gbs']:.0f} GB/s", flush=True)

    print("[build] compact sparse model ...", flush=True)
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=max(int(args.code_size), 20), compute_mode="sparse_mm")
    model.to(str(dev))
    print(f"[build] done {time.time()-t0:.0f}s blocks={len(model.blocks)} dim={model.dim}",
          flush=True)

    acct = account_model(model)
    nnz = int(acct["sparse_flop_per_tok"] / 2)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)

    Ks = [int(k) for k in args.ks.split(",")]
    forms = args.forms.split(",")
    baseline_ms_tok = {1024: 0.0699, 4096: 0.0720}
    coo_ms_tok = {1024: 0.0328, 4096: 0.0417}

    results: Dict = {
        "gpu": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in peaks.items()},
        "model": {"blocks": len(model.blocks), "dim": model.dim, "nnz_linears": nnz,
                  "sparse_flop_per_tok": float(acct["sparse_flop_per_tok"]),
                  "live_blocks": [b for b, _, _, _ in acct["live_attn"]]},
        "baseline_dense": baseline_ms_tok,
        "baseline_coo": coo_ms_tok,
        "forms": {}, "projection": {}, "block_k": args.block_k,
    }
    _save_json(results)

    def run_full(x, qpos):
        h = x
        for blk in model.blocks:
            out = blk(h, past_kv=None, q_positions=qpos, use_cache=True)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def time_eager(K, qpos, iters, warmup=5):
        x = torch.randn(1, K, model.dim, device=dev) * 0.5
        for _ in range(warmup):
            with torch.no_grad(): run_full(x, qpos)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad(): h = run_full(x, qpos)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / iters, h

    def capture(K, qpos):
        D = model.dim
        with torch.cuda.device(dev):
            static_in = torch.zeros(1, K, D, device=dev)
            st = torch.cuda.Stream(device=dev)
            st.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(st):
                for _ in range(3):
                    with torch.no_grad(): run_full(static_in, qpos)
            torch.cuda.current_stream(dev).wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad(): static_out = run_full(static_in, qpos)
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
        with torch.no_grad():
            h_eager = run_full(x, qpos).clone()
        with torch.cuda.device(dev):
            gin.copy_(x); g.replay(); torch.cuda.synchronize(dev)
        denom = max(1.0, h_eager.abs().max().item())
        rel = (gout - h_eager).abs().max().item() / denom
        return dt, rel, gout.clone()

    print("\n" + "=" * 92)
    print(f"{'form':>8} | {'K':>6} | {'eager ms':>9} | {'graph ms':>9} | {'ms/tok':>8} | "
          f"{'vsCOO':>6} | {'vsDense':>7} | {'graph':>8} | {'%peak':>6} | {'VRAM GB':>7}")
    print("-" * 92)

    for form in forms:
        # fresh swap of the FFN forward each form
        n, cache = _swap_ffn(model, dev, form, args.block_k)
        make_live_blocks_capturable(model, dev)
        results["forms"][form] = {}
        for K in Ks:
            qpos = torch.arange(K, device=dev, dtype=torch.long)
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                dt_e, _ = time_eager(K, qpos, args.iters)
            except Exception as e:
                print(f"{form:>8} | {K:>6} | EAGER ERR {type(e).__name__}: {str(e)[:40]}")
                results["forms"][form][str(K)] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                _save_json(results); continue
            graph_ok = None; dt_g = float("nan"); rel = float("nan")
            try:
                dt_g, rel, _ = time_graph(K, qpos, args.iters)
                graph_ok = "REPLAYS" if rel < 1e-3 else f"BADr{rel:.1e}"
            except Exception as e:
                graph_ok = f"NOGRAPH:{type(e).__name__}"
            peak_vram = torch.cuda.max_memory_allocated(dev) / 1e9
            best_dt = dt_g if (graph_ok and graph_ok.startswith("REPLAYS")) else dt_e
            att = attn_flops(acct["live_attn"], K)
            useful_flop = acct["sparse_flop_per_tok"] * K + att
            useful_tf = useful_flop / best_dt / 1e12
            pct_peak = useful_tf / peak_fp32 * 100
            ms_tok = best_dt * 1e3 / K
            vs_coo = coo_ms_tok.get(K, 0.0328) / ms_tok
            vs_dense = baseline_ms_tok.get(K, 0.069) / ms_tok
            print(f"{form:>8} | {K:>6} | {dt_e*1e3:>9.3f} | {dt_g*1e3:>9.3f} | {ms_tok:>8.4f} | "
                  f"{vs_coo:>6.2f} | {vs_dense:>7.2f} | {graph_ok:>8} | {pct_peak:>5.3f}% | "
                  f"{peak_vram:>7.3f}", flush=True)
            results["forms"][form][str(K)] = {
                "eager_ms_step": dt_e * 1e3,
                "graph_ms_step": None if dt_g != dt_g else dt_g * 1e3,
                "ms_per_token": ms_tok,
                "graph_replays": bool(graph_ok and graph_ok.startswith("REPLAYS")),
                "graph_replay_rel_err": None if rel != rel else rel,
                "useful_tflops": useful_tf, "pct_of_fp32_peak": pct_peak,
                "peak_vram_gb": peak_vram,
                "speedup_vs_coo": vs_coo, "speedup_vs_dense": vs_dense,
                "n_ffn_swapped": n,
            }
            _save_json(results)

    # ---- projection ----
    print("\n" + "=" * 92)
    print(f"PROJECTION — subsequent frame = {args.frame_steps:.3g} steps")
    for form in forms:
        for K in Ks:
            r = results["forms"].get(form, {}).get(str(K))
            if not r or "ms_per_token" not in r:
                continue
            ms_tok = r["ms_per_token"]
            frame_s = ms_tok * args.frame_steps / 1e3
            clears = frame_s <= 60
            print(f"  {form:>8} K={K}: {ms_tok:.4f} ms/tok -> frame {frame_s:8.1f} s "
                  f"({frame_s/60:6.2f} min)  clears 1min? {'YES' if clears else 'NO'}",
                  flush=True)
            results["projection"].setdefault(form, {})[str(K)] = {
                "frame_steps": args.frame_steps,
                "frame_seconds": frame_s, "frame_minutes": frame_s / 60,
                "clears_1min": bool(clears),
                "ms_per_token_to_clear": 60.0 / args.frame_steps * 1e3,
            }
    _save_json(results)
    print(f"\n[done] JSON -> {_RESULTS_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
