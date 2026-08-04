"""PROFILE the verify_blocks host-decode loop vs the GPU forward (COMPOSED path).

Confirms agent ada75424's finding: with the composed levers ON (dead-block fusion
+ live-head/banded/flash attention + frozen-skip + bounded-KV) the GPU forward is
~0.034 ms/step, but the whole verify is ~0.2-0.32 ms/step, dominated by the
O(steps) host-side Python decode+compare loop (per-step .item()/.cpu()/dict
compares).

This times ONE verify_blocks over a doom-scale deep loop, splitting:
  * total wall (whole verify),
  * the GPU-forward-only wall (a re-run with the per-step decode+compare STUBBED),
  * host-sync count per forward (scalar vs batched decode).

Run:
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_verify_decode_profile \
        --device cuda:0 --K 512,2048,8192 --scale doom
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


def _mem_avail_gb() -> float:
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP")


class _SyncCounter:
    def __init__(self):
        self.n_item = self.n_cpu = self.n_tolist = 0
        self._orig = {}

    def __enter__(self):
        import torch as _t
        c = self
        self._orig = {"item": _t.Tensor.item, "cpu": _t.Tensor.cpu,
                      "tolist": _t.Tensor.tolist}

        def _item(s, *a, **k):
            c.n_item += 1
            return c._orig["item"](s, *a, **k)

        def _cpu(s, *a, **k):
            if s.is_cuda:
                c.n_cpu += 1
            return c._orig["cpu"](s, *a, **k)

        def _tolist(s, *a, **k):
            c.n_tolist += 1
            return c._orig["tolist"](s, *a, **k)

        _t.Tensor.item, _t.Tensor.cpu, _t.Tensor.tolist = _item, _cpu, _tolist
        return self

    def __exit__(self, *e):
        import torch as _t
        _t.Tensor.item = self._orig["item"]
        _t.Tensor.cpu = self._orig["cpu"]
        _t.Tensor.tolist = self._orig["tolist"]

    @property
    def total(self):
        return self.n_item + self.n_cpu + self.n_tolist


def _verify(model, L, code, draft, K, device, mode, evict=True):
    # mode: "scalar" | "batched" | "gpu"
    set_batched_decode(mode == "batched")
    set_gpu_verify(mode == "gpu")
    stats = {}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=evict, mask=0xFFFFFFFF, stats=stats, fast=True,
                       evict_interval_steps=8, exact_evict=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t0
    set_batched_decode(None)
    set_gpu_verify(None)
    return vr, wall, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", default="512,2048,8192")
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--inner", type=int, default=24)
    ap.add_argument("--code-size", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args(argv)
    _guard()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    t0 = time.time()
    model, L, _ = build_lib_model_streaming(
        code_size=args.code_size, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"dev={device} build={time.time()-t0:.1f}s", flush=True)
    _guard()

    # doom-scale deep loop.
    code = build_nested(args.outer, args.inner)[0]
    os.environ["C4_FROZEN_ROW_SKIP"] = "1"
    os.environ["C4_DEAD_BLOCK_FUSION"] = "1"
    os.environ["C4_DIRECT_CAM_BATCHED"] = "1"
    os.environ["C4_DIRECT_LOCAL_CAM"] = "1"
    os.environ["C4_FLASH_ATTN"] = "1"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "1"
    install_composed(model, verbose=True)

    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    assert draft.halted
    n_steps = draft.step_count
    print(f"[draft] nested({args.outer},{args.inner}) -> {n_steps} steps (DIV-free)",
          flush=True)

    Ks = [int(k) for k in args.K.split(",") if k.strip()]
    print(f"\n{'='*80}\n[PROFILE] verify_blocks: total ms/step, scalar vs batched "
          f"decode, host-syncs/fwd\n{'='*80}", flush=True)
    print(f"  {'K':>6} {'mode':>8} {'ms/step':>10} {'fwds':>6} {'accept':>7} "
          f"{'syncs':>9} {'syncs/fwd':>10}", flush=True)
    print("  " + "-" * 62, flush=True)
    for K in Ks:
        for mode in ("scalar", "batched", "gpu"):
            for _ in range(args.warmup):
                _verify(model, L, code, draft, K, device, mode)
            with _SyncCounter() as sc:
                vr, _, _ = _verify(model, L, code, draft, K, device, mode)
            syncs = sc.total
            fwds = max(vr.forwards, 1)
            best = 1e18
            for _ in range(args.reps):
                vr, wall, _ = _verify(model, L, code, draft, K, device, mode)
                best = min(best, wall)
            print(f"  {K:>6} {mode:>8} {best/n_steps*1e3:>10.4f} {vr.forwards:>6} "
                  f"{vr.accepted_steps:>7} {syncs:>9} {syncs/fwds:>10.1f}", flush=True)
            assert vr.all_matched, f"K={K} {mode}: not all matched!"
        print("  " + "-" * 62, flush=True)
    uninstall_composed(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
