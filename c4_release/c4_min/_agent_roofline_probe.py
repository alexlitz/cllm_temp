#!/usr/bin/env python3
"""_agent_roofline_probe.py — pin down the mega dead-FFN chain roofline.

The dispatch breakdown showed the 238-block mega dead-FFN chain is ~70% of the
steady-state dispatch (1.79 us/step of 2.57).  This probe answers: is that chain
MEMORY-BW-bound (so 80 ns/step is unreachable) or launch/compute bound?

  (P1) mega-chain per-step time vs chunk (256..131072): flat => BW/compute saturated;
       falling => launch-bound. Also reports achieved GB/s from the true traffic model.
  (P2) the TRUE per-step traffic of the mega chain: per dead block, up/gate READ the
       residual at its input cols + WRITE the [Dff,C] hidden scratch; down READS the
       hidden + WRITES active residual rows. The residual [D,C] (371MB @ 65536) and the
       hidden [maxDff,C] are both >> L2 (6MB), so both stream from HBM each block.
       min residual traffic if fully resident = 2*D*4 B/step; measured is the real one.
  (P3) bf16 residual micro (lever 5 low-precision): does halving the residual dtype
       (kernels still compute fp32) roughly halve the chain time? -> the BW-bound proof.
"""
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
from c4_min.pf_speculative import draft_pf_program, _frozen_skip_cut
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested
from c4_min.fused_megablock import install_fused_megablock, MegaBlockChain

HBM_BW_GBs = 768.0
L2_BYTES = 6 * 1024 * 1024

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _on():
    for f in COMPOSED:
        os.environ[f] = "1"


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
    _on()
    dev = torch.device("cuda:0")
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    print(f"[built] blocks={len(model.blocks)} dim={model.dim} {time.time()-t0:.1f}s",
          flush=True)
    D = model.dim
    install_composed(model, verbose=False)
    cut = _frozen_skip_cut(model)
    # build the full dead-FFN mega chain (all [cut,N) dead blocks) as ONE MegaBlockChain,
    # matching the region's mega segments (the 238 dead blocks).
    dead = [b for b in range(cut, len(model.blocks))
            if getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    print(f"[dead blocks] {len(dead)} of [{cut},{len(model.blocks)})", flush=True)
    model.materialize_dense(device=str(dev))

    # ---- P2: the TRUE traffic model of the chain ----
    chain = MegaBlockChain(model, dev, dead, block_k=64)
    tot_dff = sum(mf.Dff for mf in chain._ffns)
    tot_active = sum(mf.n_active for mf in chain._ffns)
    tot_up_nnz = sum(int(mf.up_val.numel()) for mf in chain._ffns)
    tot_gt_nnz = sum(int(mf.gt_val.numel()) for mf in chain._ffns)
    tot_dn_nnz = sum(int(mf.dn_val.numel()) for mf in chain._ffns)
    n_blk = len(chain._ffns)
    print(f"[chain facts] blocks={n_blk} sum_Dff={tot_dff} sum_active={tot_active} "
          f"up_nnz={tot_up_nnz} gt_nnz={tot_gt_nnz} dn_nnz={tot_dn_nnz}", flush=True)

    # ---- P1: mega-chain time vs chunk (per-step us) ----
    print("\n=== P1: mega-chain per-step us vs chunk (graphed replay) ===", flush=True)
    print(f"{'chunk':>8} {'ms/replay':>10} {'us/step':>10} {'ach_GB/s':>10} "
          f"{'resid_MB':>9} {'hid_MB':>9}", flush=True)
    results = {}
    for C in (256, 1024, 4096, 16384, 65536, 131072):
        try:
            hq = torch.zeros(1, C, D, device=dev)
            # graphed replay of the whole chain
            _ = chain.run_graphed(hq)   # first call captures
            torch.cuda.synchronize(dev)
            ms = _cuda_time_ms(lambda: chain.run_graphed(hq), reps=25, warmup=8, dev=dev)
        except torch.cuda.OutOfMemoryError:
            print(f"{C:>8}  OOM", flush=True)
            torch.cuda.empty_cache()
            continue
        us_step = ms * 1e3 / C
        # true traffic per replay:
        # residual: initial load D*C + per block: up/gate re-read residual input cols
        #   (bounded by up_nnz+gt_nnz distinct cols; use nnz as upper) and down writes
        #   active rows. hidden scratch: each block writes Dff*C (upgate) + reads Dff*C (down).
        # Residual is [D,C]; hidden [maxDff,C].
        resid_MB = D * C * 4 / 1e6
        max_dff = max(mf.Dff for mf in chain._ffns)
        hid_MB = max_dff * C * 4 / 1e6
        # traffic: per block, upgate reads residual@cols + writes hidden(Dff*C);
        # down reads hidden(Dff*C) + writes residual active rows.
        # residual col reads (unique per block): approximate by up+gt nnz cols; but really
        # bounded by D. Assume worst = D read per block (residual band re-streamed):
        # We'll compute achieved BW two ways.
        # (a) hidden-scratch-dominated model (the [Dff,C] written+read each block):
        hid_traffic = sum(mf.Dff * C * 4 * 2 for mf in chain._ffns)  # write+read
        # (b) residual re-read: each block reads its input residual band. Bounded by D.
        # Conservative: residual streamed once per block (D*C read) + active writes.
        resid_traffic = sum((D + mf.n_active) * C * 4 for mf in chain._ffns) + D * C * 4
        total_traffic = hid_traffic + resid_traffic
        ach = total_traffic / (ms * 1e-3) / 1e9
        results[C] = {"ms": ms, "us_step": us_step, "ach_GBs": ach,
                      "hid_traffic": hid_traffic, "resid_traffic": resid_traffic}
        print(f"{C:>8} {ms:>10.4f} {us_step:>10.5f} {ach:>10.1f} {resid_MB:>9.1f} "
              f"{hid_MB:>9.1f}", flush=True)
        del hq
        torch.cuda.empty_cache()

    # ---- P3: bf16 residual (lever 5 low-precision) ----
    # free P1's per-chunk graph buffers first so the two P3 chains fit VRAM.
    del chain
    torch.cuda.empty_cache()
    chain = MegaBlockChain(model, dev, dead, block_k=64)  # keep a ref alive for facts below
    print("\n=== P3: bf16 residual chain vs fp32 (BW-bound proof; lever 5) ===", flush=True)
    C = 32768
    hq = torch.zeros(1, C, D, device=dev)
    chain_fp32 = MegaBlockChain(model, dev, dead, block_k=64, resid_dtype=torch.float32)
    chain_bf16 = MegaBlockChain(model, dev, dead, block_k=64, resid_dtype=torch.bfloat16)
    _ = chain_fp32.run_graphed(hq); torch.cuda.synchronize(dev)
    ms32 = _cuda_time_ms(lambda: chain_fp32.run_graphed(hq), 25, 8, dev)
    _ = chain_bf16.run_graphed(hq); torch.cuda.synchronize(dev)
    ms16 = _cuda_time_ms(lambda: chain_bf16.run_graphed(hq), 25, 8, dev)
    print(f"  fp32 residual: {ms32:.4f} ms  {ms32*1e3/C:.5f} us/step", flush=True)
    print(f"  bf16 residual: {ms16:.4f} ms  {ms16*1e3/C:.5f} us/step", flush=True)
    print(f"  -> bf16 speedup {ms32/ms16:.3f}x  (residual halved; kernels still fp32) "
          f"{'[BW-bound: halving resid dtype ~halves time]' if ms32/ms16 > 1.3 else '[NOT purely BW-bound]'}",
          flush=True)

    uninstall_composed(model)
    out = {"dead_blocks": n_blk, "sum_Dff": tot_dff, "sum_active": tot_active,
           "up_nnz": tot_up_nnz, "gt_nnz": tot_gt_nnz, "dn_nnz": tot_dn_nnz,
           "p1": results, "p3": {"fp32_ms": ms32, "bf16_ms": ms16,
                                 "bf16_speedup": ms32 / ms16, "chunk": C}}
    json.dump(out, open("/tmp/roofline_probe.json", "w"), indent=2, default=str)
    print("\n[written] /tmp/roofline_probe.json", flush=True)


if __name__ == "__main__":
    main()
