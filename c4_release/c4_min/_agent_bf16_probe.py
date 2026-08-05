#!/usr/bin/env python3
"""_agent_bf16_probe.py — isolated bf16-vs-fp32 residual chain probe (lever 5 / lever 3
low-precision). Builds ONE model, then times the dead-FFN mega chain with a fp32 vs a
bf16 resident residual at a VRAM-safe chunk. A ~2x speedup proves the chain is
HBM-BW-bound (residual/hidden streaming dominates)."""
from __future__ import annotations
import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import _frozen_skip_cut
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.fused_megablock import MegaBlockChain

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _cuda_time_ms(fn, reps=20, warmup=6, dev=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    s = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize(dev)
    ts = sorted(s[i].elapsed_time(e[i]) for i in range(reps))
    keep = ts[2:-2] if len(ts) > 6 else ts
    return sum(keep) / len(keep)


def main():
    for f in COMPOSED:
        os.environ[f] = "1"
    dev = torch.device("cuda:0")
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    D = model.dim
    install_composed(model, verbose=False)
    cut = _frozen_skip_cut(model)
    dead = [b for b in range(cut, len(model.blocks))
            if getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    model.materialize_dense(device=str(dev))
    print(f"[built] blocks={len(model.blocks)} dim={D} dead={len(dead)} {time.time()-t0:.1f}s",
          flush=True)

    C = 32768  # VRAM-safe (resid 185MB fp32 + hidden 552MB fp32) x per chain
    hq = torch.zeros(1, C, D, device=dev)
    out = {"chunk": C}
    for tag, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        chain = MegaBlockChain(model, dev, dead, block_k=64, resid_dtype=dt)
        _ = chain.run_graphed(hq); torch.cuda.synchronize(dev)
        ms = _cuda_time_ms(lambda: chain.run_graphed(hq), 25, 8, dev)
        out[tag] = {"ms": ms, "us_step": ms * 1e3 / C}
        print(f"  {tag} residual: {ms:.4f} ms  {ms*1e3/C:.5f} us/step", flush=True)
        del chain
        torch.cuda.empty_cache()
    sp = out["fp32"]["us_step"] / out["bf16"]["us_step"]
    out["bf16_speedup"] = sp
    print(f"  -> bf16 speedup {sp:.3f}x  "
          f"{'[BW-BOUND: residual streaming dominates]' if sp > 1.3 else '[compute/launch-bound]'}",
          flush=True)
    uninstall_composed(model)
    json.dump(out, open("/tmp/bf16_probe.json", "w"), indent=2)
    print("[written] /tmp/bf16_probe.json", flush=True)


if __name__ == "__main__":
    main()
