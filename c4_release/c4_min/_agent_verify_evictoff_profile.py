"""Isolate the DECODE cost: verify_blocks with EVICTION OFF (evict=False), so the
only per-step host cost is the decode+compare loop.  This shows the GPU-verify path
drops the whole verify toward the ~0.034 ms/step GPU-forward floor once eviction
(a separate cost, not the decode loop) is removed.

Run:
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_verify_evictoff_profile \
        --device cuda:0 --K 512,2048,8192
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import (draft_pf_program, verify_blocks,
                                   set_batched_decode, set_gpu_verify)
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _verify(model, L, code, draft, K, device, mode):
    set_batched_decode(mode == "batched")
    set_gpu_verify(mode == "gpu")
    stats = {}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t0
    set_batched_decode(None)
    set_gpu_verify(None)
    return vr, wall


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", default="512,2048,8192")
    ap.add_argument("--outer", type=int, default=20)
    ap.add_argument("--inner", type=int, default=30)
    ap.add_argument("--code-size", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)
    if _mem_avail_gb() < 25.0:
        raise SystemExit("[GUARD] <25GB -> STOP")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, L, _ = build_lib_model_streaming(
        code_size=args.code_size, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)
    for f in ("C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED",
              "C4_DIRECT_LOCAL_CAM", "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN"):
        os.environ[f] = "1"
    install_composed(model, verbose=False)

    code = build_nested(args.outer, args.inner)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    n_steps = draft.step_count
    print(f"[draft] nested({args.outer},{args.inner}) -> {n_steps} steps "
          f"(EVICT OFF: isolates forward+decode)", flush=True)
    print(f"\n{'K':>6} {'mode':>8} {'ms/step':>10} {'fwds':>6} {'accepted':>9}",
          flush=True)
    print("-" * 42, flush=True)
    Ks = [int(k) for k in args.K.split(",") if k.strip()]
    for K in Ks:
        for mode in ("scalar", "gpu"):
            _verify(model, L, code, draft, K, device, mode)     # warmup
            best = 1e18
            for _ in range(args.reps):
                vr, wall = _verify(model, L, code, draft, K, device, mode)
                best = min(best, wall)
            print(f"{K:>6} {mode:>8} {best/n_steps*1e3:>10.4f} {vr.forwards:>6} "
                  f"{vr.accepted_steps:>9}", flush=True)
            assert vr.all_matched
        print("-" * 42, flush=True)
    uninstall_composed(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
