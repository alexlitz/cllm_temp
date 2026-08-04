"""_agent_fused_composed.py — wire the #808/#841 FUSED-DELTA sparse FFN into the
COMPOSED KV-cached verify_blocks path (#862/#869) and MEASURE.

Runs the REAL doom composed stack (dead-block-fusion #856 + overlay #851 +
direct-CAM #834 + direct-local #843 + bounded-KV/exact-evict + banded-local +
direct-CAM-vec #866, attention driven to ~ZERO) on the KV-cached verify_blocks
path, in TWO configs:

  (A) FFN OFF  — C4_FUSED_DELTA_FFN unset (the dense/COO SwiGLU GEMM, #869's 67%)
  (B) FFN ON   — C4_FUSED_DELTA_FFN=1  (the fused-delta sparse-COO kernel)

For each: (1) byte-exact gate (all_matched + decoded_final_ax equal to config A);
(2) ms/step + steps/sec + distance to 1s/frame (6.89M steps == the doom frame);
(3) a torch.profiler CUDA-time breakdown so we see the FFN GEMM (ampere_sgemm)
share collapse and what the new residual wall is.

LEAN STREAMING (C4_PF_CFM=1).  Stops < 25 GB host RAM.  Run:
    cd c4_release
    CUDA_VISIBLE_DEVICES=0,1 python -m c4_min._agent_fused_composed --device cuda:0 \
        --steps 2000 --K 512
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_PF_CFM"] = "1"
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("C4_EXACT_EVICT", "1")
os.environ.setdefault("C4_MEM_EFF", "2000000")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# the #869 composed lever stack (all EXCEPT the FFN, which we toggle below).
for f, d in [("C4_DIRECT_CAM_BATCHED", "1"), ("C4_DIRECT_LOCAL_CAM", "1"),
             ("C4_BANDED_LOCAL_ATTN", "1"), ("C4_DIRECT_CAM_LIVE_LOCAL", "1"),
             ("C4_FROZEN_ROW_SKIP", "1"), ("C4_BATCHED_BLOCK_SKIP", "1"),
             ("C4_DEAD_BLOCK_FUSION", "1"), ("C4_OVERLAY_BATCHED", "1"),
             ("C4_BATCHED_DECODE", "1"), ("C4_DIRECT_CAM_VEC", "1")]:
    os.environ.setdefault(f, d)

sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom")
import torch

DOOM = "/home/alexlitz/Documents/misc/c4_doom/doom.c"
FRAME_STEPS = 6_890_000     # #869: one doom frame == 6.89M VM steps


def _mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return 1e9


def _mem_guard(where=""):
    a = _mem_avail_gb()
    print(f"  [mem] {a:.1f}GB avail ({where})", flush=True)
    if a < 25.0:
        raise SystemExit(f"[MEM-GUARD] {a:.1f}GB < 25GB ({where}) STOP")


def _build(dev, code):
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.local_attention import install_local_attention
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    sparse = sparse.to(dev)
    install_local_attention(sparse, window=96, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    return sparse, L


def _time_verify(sparse, L, code, draft, K, dev, n=3):
    from c4_min.pf_speculative import verify_blocks
    # warmup
    verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                  mask=0xFFFFFFFF, fast=True)
    torch.cuda.synchronize()
    best = None
    vr = None
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.time()
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev,
                           evict=True, mask=0xFFFFFFFF, fast=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        best = dt if best is None else min(best, dt)
    return best, vr


def _profile_verify(sparse, L, code, draft, K, dev):
    from torch.profiler import profile, ProfilerActivity
    from c4_min.pf_speculative import verify_blocks
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev,
                           evict=True, mask=0xFFFFFFFF, fast=True)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    rows = sorted(ka, key=lambda e: getattr(e, "self_device_time_total", 0),
                  reverse=True)
    tot = sum(getattr(e, "self_device_time_total", 0) for e in ka) or 1
    out = []
    for e in rows[:14]:
        c = getattr(e, "self_device_time_total", 0)
        out.append((e.key[:44], e.count, c / 1e3, 100 * c / tot))
    return out, tot / 1e3, vr


def _is_ffn_gemm(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in ("sgemm", "gemm", "cutlass", "ampere", "volta",
                                "cublas", "wgrad", "gemv"))


def _is_elementwise(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in ("elementwise", "vectorized_elementwise",
                                "add", "mul", "silu", "sigmoid", "copy",
                                "cat", "index", "fill", "sub", "reduce"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    dev = args.device
    torch.cuda.init()
    _mem_guard("start")

    from pathlib import Path
    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from run_c4_min import (tag_compiler_syscalls,
                            install_compiler_abi_file_dispatcher, data_segment)

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}),
        stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=args.steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    ns = draft.step_count
    print(f"[fused-composed] doom steps={ns} K={args.K} instrs={len(code)} "
          f"dev={dev}", flush=True)
    _mem_guard("post-draft")

    results = {}
    for tag, ffn_on in (("A_dense_ffn", False), ("B_fused_delta_ffn", True)):
        if ffn_on:
            os.environ["C4_FUSED_DELTA_FFN"] = "1"
        else:
            os.environ.pop("C4_FUSED_DELTA_FFN", None)
        print(f"\n{'='*70}\n[{tag}] C4_FUSED_DELTA_FFN="
              f"{os.environ.get('C4_FUSED_DELTA_FFN','<unset>')}\n{'='*70}",
              flush=True)
        sparse, L = _build(dev, code)
        _mem_guard(f"{tag} post-build")

        dt, vr = _time_verify(sparse, L, code, draft, args.K, dev, n=args.n)
        ms_step = dt / max(ns, 1) * 1e3
        steps_sec = ns / dt
        frame_s = FRAME_STEPS / steps_sec
        print(f"  matched={vr.all_matched} final_ax={vr.decoded_final_ax} "
              f"forwards={vr.forwards}", flush=True)
        print(f"  wall={dt*1e3:.1f}ms over {ns} steps -> {ms_step:.4f} ms/step  "
              f"{steps_sec:,.0f} steps/sec  frame={frame_s:.1f}s "
              f"(goal 1s -> {frame_s:.1f}x)", flush=True)

        prof_rows, prof_tot, vr2 = _profile_verify(sparse, L, code, draft,
                                                   args.K, dev)
        ffn_pct = sum(p for (_, _, _, p) in prof_rows
                      if _is_ffn_gemm(prof_rows and _)) if False else None
        # recompute shares by class over the FULL key-average set is more honest;
        # but the top-14 rows dominate CUDA time, so classify them.
        ffn_pct = sum(p for (nm, _, _, p) in prof_rows if _is_ffn_gemm(nm))
        elem_pct = sum(p for (nm, _, _, p) in prof_rows
                       if (not _is_ffn_gemm(nm)) and _is_elementwise(nm))
        print(f"  [profile] total self CUDA={prof_tot:.1f}ms  "
              f"FFN-GEMM~={ffn_pct:.0f}%  elementwise~={elem_pct:.0f}%", flush=True)
        for nm, cnt, cms, pct in prof_rows:
            cls = "FFN " if _is_ffn_gemm(nm) else (
                "ELEM" if _is_elementwise(nm) else "    ")
            print(f"    [{cls}] {nm:44s} cnt={cnt:6d} "
                  f"cuda={cms:8.2f}ms ({pct:.0f}%)", flush=True)

        results[tag] = {
            "matched": bool(vr.all_matched),
            "final_ax": vr.decoded_final_ax,
            "ms_step": ms_step, "steps_sec": steps_sec, "frame_s": frame_s,
            "ffn_pct": ffn_pct, "elem_pct": elem_pct,
        }
        del sparse
        torch.cuda.empty_cache()
        _mem_guard(f"{tag} done")

    # byte-exact cross-check: B must match A (same final_ax; both all_matched).
    A, B = results["A_dense_ffn"], results["B_fused_delta_ffn"]
    byte_exact = (A["matched"] and B["matched"] and A["final_ax"] == B["final_ax"])
    speedup = A["ms_step"] / B["ms_step"] if B["ms_step"] else float("nan")
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    print(f"  BYTE-EXACT (A==B, both matched): {byte_exact}", flush=True)
    print(f"  ms/step  dense={A['ms_step']:.4f} -> fused={B['ms_step']:.4f}  "
          f"({speedup:.2f}x)", flush=True)
    print(f"  steps/s  dense={A['steps_sec']:,.0f} -> fused={B['steps_sec']:,.0f}",
          flush=True)
    print(f"  frame_s  dense={A['frame_s']:.1f}s -> fused={B['frame_s']:.1f}s "
          f"(goal 1s)", flush=True)
    print(f"  FFN%     dense={A['ffn_pct']:.0f}% -> fused={B['ffn_pct']:.0f}%",
          flush=True)
    print(f"  ELEM%    dense={A['elem_pct']:.0f}% -> fused={B['elem_pct']:.0f}%",
          flush=True)
    return 0 if byte_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
