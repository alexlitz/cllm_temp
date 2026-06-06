#!/usr/bin/env python3
"""
Throughput scaling benchmark: CSR vs dense across batch sizes.

Companion to ``c4_release/tools/sparse_inference_benchmark.py`` (commit
``cb0f396f``), which fixed B=8/T=128 and measured a 2.11x CSR/dense
speedup on an RTX A5000.

This script answers: does the CSR/dense speedup ratio grow with batch
size? Hypothesis: dense matmul on GPU is bandwidth-limited at large
batches (every token loads the same 184M params), while CSR amortises
the same load across more tokens with less arithmetic per token. So
CSR's relative advantage should grow with batch size.

For each batch size in ``--batch-sizes`` (default 1,8,32,64,128), we run
``dense`` then ``csr`` over ``--iters`` forward passes (after ``--warmup``
warmups). Tokens-per-second is computed from the mean latency; we report
the per-batch CSR/dense ratio and whether the scaling is super-linear.

Usage::

    python -m c4_release.tools.sparse_throughput_scaling
    python -m c4_release.tools.sparse_throughput_scaling --device cuda \\
        --batch-sizes 1,8,32,64,128 --seq 128 --iters 20 --warmup 3 \\
        --write-doc
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch

# Re-use the heavy lifting from the existing benchmark.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_inference_benchmark import (  # type: ignore  # noqa: E402
    _import_compile,
    _make_input,
    _benchmark_mode,
    _CSRLinearShim,
    build_dense,
    build_csr,
    _fmt_bytes,
)


@dataclass
class BatchPoint:
    batch: int
    seq: int
    dense_mean_ms: float
    dense_std_ms: float
    dense_tok_per_s: float
    csr_mean_ms: float
    csr_std_ms: float
    csr_tok_per_s: float
    latency_speedup: float  # dense_ms / csr_ms
    throughput_speedup: float  # csr_tok_per_s / dense_tok_per_s
    dense_error: Optional[str] = None
    csr_error: Optional[str] = None
    notes: str = ""


def _free_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _run_one_batch(
    base_model,
    device: torch.device,
    B: int,
    T: int,
    iters: int,
    warmup: int,
    skip_dense: bool = False,
    skip_csr: bool = False,
) -> BatchPoint:
    """Run dense and csr at (B, T). Returns a BatchPoint with both timings."""
    tokens = _make_input(base_model, B, T, device)
    tokens_per_pass = B * T

    dense_mean = float("nan")
    dense_std = float("nan")
    dense_err: Optional[str] = None
    if not skip_dense:
        try:
            dense_model = build_dense(base_model, device)
            r = _benchmark_mode(
                "dense", dense_model, tokens, device,
                iters=iters, warmup=warmup,
                reference_logits=None,
                notes=f"dense F.linear @ B={B}",
            )
            dense_mean = r.mean_ms
            dense_std = r.std_ms
            dense_err = r.error
            del dense_model
            _free_cuda(device)
        except Exception as exc:  # noqa: BLE001
            dense_err = f"{type(exc).__name__}: {exc}"
            _free_cuda(device)

    csr_mean = float("nan")
    csr_std = float("nan")
    csr_err: Optional[str] = None
    if not skip_csr:
        try:
            csr_model = build_csr(base_model, device)
            r = _benchmark_mode(
                "csr", csr_model, tokens, device,
                iters=iters, warmup=warmup,
                reference_logits=None,
                notes=f"csr torch.sparse.mm @ B={B}",
                forward_ctx=_CSRLinearShim,
            )
            csr_mean = r.mean_ms
            csr_std = r.std_ms
            csr_err = r.error
            del csr_model
            _free_cuda(device)
        except Exception as exc:  # noqa: BLE001
            csr_err = f"{type(exc).__name__}: {exc}"
            _free_cuda(device)

    dense_tok = (tokens_per_pass / (dense_mean / 1000.0)) if dense_mean > 0 else 0.0
    csr_tok = (tokens_per_pass / (csr_mean / 1000.0)) if csr_mean > 0 else 0.0
    lat_speedup = (dense_mean / csr_mean) if (csr_mean > 0 and dense_mean > 0) else float("nan")
    tput_speedup = (csr_tok / dense_tok) if dense_tok > 0 else float("nan")

    return BatchPoint(
        batch=B, seq=T,
        dense_mean_ms=dense_mean, dense_std_ms=dense_std,
        dense_tok_per_s=dense_tok,
        csr_mean_ms=csr_mean, csr_std_ms=csr_std,
        csr_tok_per_s=csr_tok,
        latency_speedup=lat_speedup,
        throughput_speedup=tput_speedup,
        dense_error=dense_err,
        csr_error=csr_err,
    )


def _print_summary(points: List[BatchPoint], device: str, T: int) -> None:
    print()
    print(f"=== CSR vs Dense throughput scaling ({device}, T={T}) ===")
    print()
    hdr = (
        f"{'B':>4} {'dense ms':>10} {'csr ms':>10} "
        f"{'dense tok/s':>12} {'csr tok/s':>12} "
        f"{'lat speedup':>12} {'tput ratio':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in points:
        if p.dense_error or p.csr_error:
            err = p.dense_error or p.csr_error
            print(f"{p.batch:>4}   ERROR: {err}")
            continue
        print(
            f"{p.batch:>4} {p.dense_mean_ms:>10.2f} {p.csr_mean_ms:>10.2f} "
            f"{p.dense_tok_per_s:>12,.1f} {p.csr_tok_per_s:>12,.1f} "
            f"{p.latency_speedup:>11.2f}x {p.throughput_speedup:>10.2f}x"
        )
    print()


def _scaling_verdict(points: List[BatchPoint]) -> str:
    valid = [p for p in points if not p.dense_error and not p.csr_error
             and p.throughput_speedup == p.throughput_speedup  # not NaN
             and p.throughput_speedup > 0]
    if len(valid) < 2:
        return "Insufficient valid points to judge scaling."
    first = valid[0]
    last = valid[-1]
    delta = last.throughput_speedup - first.throughput_speedup
    if delta > 0.05:
        trend = "GROWS"
    elif delta < -0.05:
        trend = "SHRINKS"
    else:
        trend = "FLAT"
    max_ratio = max(p.throughput_speedup for p in valid)
    max_at = next(p for p in valid if p.throughput_speedup == max_ratio)
    # "Super-linear" would require throughput speedup ratio growing
    # faster than linearly in batch size. Concretely: if dense throughput
    # plateaus while CSR throughput keeps growing linearly with B, the
    # ratio grows. We label it super-linear vs the smallest-B baseline.
    if trend == "GROWS":
        label = "SUPER-LINEAR in batch size (CSR's relative advantage grows)"
    elif trend == "FLAT":
        label = "LINEAR in batch size (CSR's relative advantage is constant)"
    else:
        label = "SUB-LINEAR in batch size (CSR's relative advantage shrinks)"
    return (f"{label}. CSR/dense throughput ratio at B={first.batch} is "
            f"{first.throughput_speedup:.2f}x; at B={last.batch} it is "
            f"{last.throughput_speedup:.2f}x. Peak ratio "
            f"{max_ratio:.2f}x at B={max_at.batch}.")


def _write_doc(out_path: str, payload: dict) -> None:
    lines: List[str] = []
    lines.append("# CSR vs Dense Throughput Scaling Across Batch Sizes (2026-06-06)")
    lines.append("")
    lines.append("Companion to "
                 "[`SPARSE_INFERENCE_BENCHMARK_2026_06_06.md`]"
                 "(SPARSE_INFERENCE_BENCHMARK_2026_06_06.md), which measured a "
                 "**2.11× CSR/dense speedup at (B=8, T=128)** on an RTX A5000.")
    lines.append("")
    lines.append("This benchmark asks: **does the CSR/dense throughput speedup "
                 "ratio grow with batch size?**")
    lines.append("")
    lines.append("**Hypothesis.** Dense matmul on GPU becomes memory-bandwidth "
                 "limited at large batches (every token loads the same 184M "
                 "params). CSR amortises the same weight load across more "
                 "tokens with less arithmetic per token. So CSR's *relative* "
                 "advantage over dense should grow with batch size.")
    lines.append("")
    lines.append("Source: [`c4_release/tools/sparse_throughput_scaling.py`]"
                 "(../tools/sparse_throughput_scaling.py). Re-run with "
                 "`python -m c4_release.tools.sparse_throughput_scaling --write-doc`.")
    lines.append("")
    lines.append("## Model under test")
    for k, v in payload["model_meta"].items():
        lines.append(f"- `{k}` = {v}")
    lines.append("")
    lines.append(f"## Throughput scaling on `{payload['device']}` (T={payload['T']})")
    lines.append("")
    lines.append("| B | dense mean ms | csr mean ms | dense tok/s | csr tok/s |"
                 " latency speedup | throughput ratio |")
    lines.append("|--:|--------------:|------------:|------------:|----------:|"
                 "----------------:|-----------------:|")
    for p in payload["points"]:
        if p["dense_error"] or p["csr_error"]:
            err = p["dense_error"] or p["csr_error"]
            lines.append(f"| {p['batch']} | ERROR | | | | | `{err}` |")
            continue
        lines.append(
            f"| {p['batch']} | {p['dense_mean_ms']:.2f} | "
            f"{p['csr_mean_ms']:.2f} | {p['dense_tok_per_s']:,.0f} | "
            f"{p['csr_tok_per_s']:,.0f} | "
            f"{p['latency_speedup']:.2f}x | "
            f"**{p['throughput_speedup']:.2f}x** |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(payload["verdict"])
    lines.append("")
    lines.append("### Per-batch CSR/dense throughput speedup table")
    lines.append("")
    lines.append("| B | tokens/pass | CSR/dense throughput ratio |")
    lines.append("|--:|------------:|---------------------------:|")
    for p in payload["points"]:
        if p["dense_error"] or p["csr_error"]:
            lines.append(f"| {p['batch']} | {p['batch']*payload['T']} | ERROR |")
            continue
        lines.append(
            f"| {p['batch']} | {p['batch']*payload['T']} | "
            f"**{p['throughput_speedup']:.2f}x** |"
        )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- iters per measurement: {payload['iters']}")
    lines.append(f"- warmup: {payload['warmup']}")
    lines.append("- Model compiled via `compile_full_vm_dynamic(disk_cache=True)` "
                 "(warm disk cache).")
    lines.append("- Each (mode, B) combo runs on a fresh `deepcopy` of the "
                 "baseline model to avoid mutation leakage.")
    lines.append("- Random int64 input tokens at shape `[B, T]` (token contents "
                 "do not affect matmul cost).")
    lines.append("- Latency is the mean of `iters` warm forward passes; "
                 "throughput = `B*T / mean_latency_seconds`.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```")
    lines.append("python -m c4_release.tools.sparse_throughput_scaling "
                 "--device cuda --batch-sizes "
                 + ",".join(str(p["batch"]) for p in payload["points"])
                 + f" --seq {payload['T']} --iters {payload['iters']} "
                 f"--warmup {payload['warmup']} --write-doc")
    lines.append("```")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda-index", type=int, default=None,
                        help="If --device cuda, use this device index "
                             "(e.g. 1 to pick the second GPU).")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128",
                        help="Comma-separated batch sizes to sweep.")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--write-doc", action="store_true",
                        help="Write results to "
                             "c4_release/docs/SPARSE_THROUGHPUT_SCALING_"
                             "2026_06_06.md.")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--stop-on-oom", action="store_true",
                        help="Stop the sweep once OOM is hit at a batch size "
                             "(otherwise mark and continue to next).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    if device.type == "cuda" and args.cuda_index is not None:
        device = torch.device(f"cuda:{args.cuda_index}")
        torch.cuda.set_device(device)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    batch_sizes.sort()

    compile_fn = _import_compile()
    print(f"[bench] device={device}")
    print(f"[bench] batch_sizes={batch_sizes}, seq={args.seq}, "
          f"iters={args.iters}, warmup={args.warmup}")
    print(f"[bench] loading model via {compile_fn.__name__}(disk_cache=True)")
    t0 = time.time()
    base_model, _layout = compile_fn(disk_cache=True)
    base_model.eval()
    print(f"[bench] compile took {time.time()-t0:.2f}s (warm cache)")

    n_params = sum(p.numel() for p in base_model.parameters())
    n_nonzero = sum((p.detach() != 0).sum().item()
                    for p in base_model.parameters())
    sparsity = (1.0 - n_nonzero / n_params) * 100
    model_meta = {
        "blocks": len(base_model.blocks),
        "d_model": base_model.d_model,
        "max_seq_len": base_model.max_seq_len,
        "params_total": n_params,
        "params_nonzero": n_nonzero,
        "sparsity_pct": round(sparsity, 4),
        "positional_encoding": getattr(base_model, "positional_encoding", "alibi"),
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else "cpu"),
    }
    print(f"[bench] model: {model_meta['blocks']} blocks, "
          f"d_model={model_meta['d_model']}, params={n_params:,}, "
          f"sparsity={sparsity:.3f}%")

    points: List[BatchPoint] = []
    oom_hit = False
    for B in batch_sizes:
        if oom_hit and args.stop_on_oom:
            print(f"[bench] skipping B={B} (--stop-on-oom after prior OOM)")
            continue
        print(f"[bench] --- B={B}, T={args.seq} ---")
        try:
            p = _run_one_batch(
                base_model, device, B, args.seq,
                iters=args.iters, warmup=args.warmup,
            )
        except torch.cuda.OutOfMemoryError as exc:  # noqa: F841
            print(f"[bench] B={B} OOM")
            p = BatchPoint(
                batch=B, seq=args.seq,
                dense_mean_ms=float("nan"), dense_std_ms=float("nan"),
                dense_tok_per_s=0.0,
                csr_mean_ms=float("nan"), csr_std_ms=float("nan"),
                csr_tok_per_s=0.0,
                latency_speedup=float("nan"),
                throughput_speedup=float("nan"),
                dense_error="CUDA OOM",
                csr_error="CUDA OOM",
            )
            oom_hit = True
            _free_cuda(device)
        if p.dense_error and "OutOfMemoryError" in str(p.dense_error):
            oom_hit = True
        if p.csr_error and "OutOfMemoryError" in str(p.csr_error):
            oom_hit = True
        points.append(p)
        if p.dense_error or p.csr_error:
            print(f"[bench] B={B}  ERROR  dense={p.dense_error} "
                  f"csr={p.csr_error}")
        else:
            print(f"[bench] B={B}  dense={p.dense_mean_ms:.2f}ms "
                  f"({p.dense_tok_per_s:,.0f} tok/s)  "
                  f"csr={p.csr_mean_ms:.2f}ms ({p.csr_tok_per_s:,.0f} tok/s)  "
                  f"throughput ratio={p.throughput_speedup:.2f}x")

    _print_summary(points, str(device), args.seq)
    verdict = _scaling_verdict(points)
    print(f"Verdict: {verdict}")

    payload = {
        "device": str(device),
        "T": args.seq,
        "iters": args.iters,
        "warmup": args.warmup,
        "model_meta": model_meta,
        "points": [asdict(p) for p in points],
        "verdict": verdict,
    }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[bench] wrote {args.out_json}")

    if args.write_doc:
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "docs",
            "SPARSE_THROUGHPUT_SCALING_2026_06_06.md",
        )
        out_path = os.path.abspath(out_path)
        _write_doc(out_path, payload)
        print(f"[bench] wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
