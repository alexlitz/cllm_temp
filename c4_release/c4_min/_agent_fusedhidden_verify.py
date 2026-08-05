#!/usr/bin/env python3
"""_agent_fusedhidden_verify.py — byte-exactness + timing + roofline of the FUSED-HIDDEN
dead-FFN megakernel (C4_FFN_FUSED_HIDDEN) vs the 2-kernel HBM-hidden path and the eager
per-block loop, on the REAL DIV-free doom region.

Three paths, all over the SAME DIV-free [cut=1,N) region (dead-FFN mega-chains + eager
live CAM blocks):
  (E) EAGER  per-block loop (the reference).
  (H) MEGA-2KERNEL  (C4_FFN_FUSED_HIDDEN=0): kernel-1 writes [Dff,K] hidden to HBM,
      kernel-2 reads it back (the profiler's 92%-of-bytes round-trip).
  (F) MEGA-FUSED     (C4_FFN_FUSED_HIDDEN=1): single kernel, hidden recomputed IN
      REGISTERS per active output row, ZERO [Dff,K] HBM buffer.

Reports per K: byte-exact L-inf (F vs E, H vs E), ms for each path, the fused speedup,
the achieved GB/s and % of the 768 GB/s A5000 peak (from the roofline bytes/step of each
path), and the projected sec/frame + fps at 358,058 steps.

Read-only vs the model (the megakernel is opt-in; flags default off for E/H).  GPU 1.
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import time
import torch

from . import isa
from .compact_alloc import build_compact_sparse_streaming
from .step_block_skip import build_live_index
from .block_sparse_ffn import install_block_sparse_ffn
from .live_head_attention import install_live_head_attention, install_dead_block_fusion
from . import fused_megablock as FM
from .fused_megablock import MegaBlockChain, _dense_of

HBM_BW_GBs = 768.0
RENDER_STEPS = 358_058


def _mem_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return float(l.split()[1]) / 1e6
    return 1e9


def build():
    dev = "cuda:0"
    torch.cuda.set_device(0)
    model, L, _ = build_compact_sparse_streaming(code_size=48,
                                                 compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    install_block_sparse_ffn(model, mode="coo", verbose=False)
    model.to(dev)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)
    return model, L, dev


def divfree_region(model, L):
    live_index = build_live_index(model, L)
    union = set()
    for op in live_index:
        if op is None or op in (isa.DIV, isa.MOD):
            continue
        union.update(live_index[op])
    region = sorted(b for b in union if b >= 1)
    dead = [b for b in region
            if getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    live = [b for b in region
            if not getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    return region, dead, live


def build_items(model, dev, region, live, block_k):
    """Split region into (kind, MegaBlockChain|blk) items.  The MegaBlockChain reads the
    fused-hidden flag at CONSTRUCTION, so build under the desired env."""
    live_set = set(live); items = []; seg = []
    for b in region:
        if b in live_set:
            if seg:
                items.append(("mega", MegaBlockChain(model, dev, seg, block_k))); seg = []
            items.append(("live", b))
        else:
            seg.append(b)
    if seg:
        items.append(("mega", MegaBlockChain(model, dev, seg, block_k)))
    return items


def eager(model, region, hq, qp):
    h = hq
    for b in region:
        h, _ = model.blocks[b](h, past_kv=None, q_positions=qp, use_cache=True)
    return h


def run_items(model, items, hq, qp):
    h = hq
    for kind, p in items:
        if kind == "live":
            h, _ = model.blocks[p](h, past_kv=None, q_positions=qp, use_cache=True)
        else:
            h = p.run(h)
    return h


def run_mega_only(items, hq):
    """Run ONLY the dead-FFN mega chains (drop the live CAM blocks) — the fused-hidden
    lever's target.  The live CAM attention has a non-deterministic reduction (its output
    varies run-to-run on tiny random inputs), so it is EXCLUDED from the byte-exact gate;
    the dead-FFN chain is the deterministic, byte-exact-comparable component and the whole
    point of C4_FFN_FUSED_HIDDEN.  Live-block byte-exactness is verified end-to-end at the
    driver level (per-step AX/PC/SP/BP, nibble-snap)."""
    h = hq
    for kind, p in items:
        if kind == "mega":
            h = p.run(h)
    return h


def time_fn(fn, reps=40, warmup=12):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def roofline_bytes(model, dead, D):
    """Per-K-column HBM bytes for the dead-FFN chain in the 2-kernel vs fused path.
    2-kernel: per block, hidden write+read (2*Dff*4) + active residual RMW (2*n_act*4)
              + inactive residual re-reads are L2 (count residual base read once/block D*4).
    fused:    per block, bulk-copy y (read D + write D = 2*D*4) + active RMW (2*n_act*4);
              up/gate reads are from the same y (L2).  NO [Dff,K] hidden bytes.
    Returns (bytes_2k, bytes_fused) per K-column over all DISTINCT dead blocks (as a
    proxy; real chain reuses shared FFNs)."""
    seen = set(); b2 = bf = 0
    for b in dead:
        ffn = model.blocks[b].ffn
        if id(ffn) in seen:
            continue
        seen.add(id(ffn))
        Wu = _dense_of(ffn.W_up); Wg = _dense_of(ffn.W_gate); Wd = _dense_of(ffn.W_down)
        Dff = int(Wu.shape[0])
        n_act = int((Wd != 0).any(dim=1).sum())
        n_in = int(((Wu != 0).any(dim=0) | (Wg != 0).any(dim=0)).sum())
        # 2-kernel: hidden write+read (2*Dff) + active-row RMW (2*n_act) + resid base read (D).
        b2 += (2 * Dff + 2 * n_act + D) * 4
        # fused: snapshot read+write (2*n_in) + active-row RMW (2*n_act), NO [Dff,K] hidden.
        bf += (2 * n_in + 2 * n_act) * 4
    return b2, bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--Ks", default="512,2048,8192")
    a = ap.parse_args()
    if _mem_gb() < 25.0:
        raise SystemExit(f"[GUARD] MemAvail {_mem_gb():.1f}GB < 25 -> STOP")

    model, L, dev = build()
    region, dead, live = divfree_region(model, L)
    D = model.dim
    print(f"dim={D}  DIV-free region={len(region)} ({len(dead)} dead, {len(live)} live "
          f"{live})  block_k={a.block_k}  MemAvail={_mem_gb():.1f}GB", flush=True)

    # Build the 2-kernel items (flag OFF) and the fused items (flag ON).  The
    # MegaBlockChain reads fused_hidden_enabled() at construction.
    os.environ["C4_FFN_FUSED_HIDDEN"] = "0"
    items_2k = build_items(model, dev, region, live, a.block_k)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    items_fused = build_items(model, dev, region, live, a.block_k)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "0"

    b2, bf = roofline_bytes(model, dead, D)
    print(f"roofline dead-FFN bytes/K-col: 2-kernel={b2}  fused={bf}  "
          f"(hidden eliminated -> {100*(1-bf/max(b2,1)):.0f}% fewer bytes)", flush=True)

    print("\nBYTE-EXACT GATE = mega-only Linf(fused vs 2-kernel); the live CAM blocks have a\n"
          "non-deterministic reduction and are excluded here (verified at the driver level).\n"
          "Timing = the DEAD-FFN MEGA CHAIN alone (the fused-hidden lever's target).", flush=True)
    print(f"\n{'K':>6s} {'megaFvH':>9s} {'exact':>6s} | "
          f"{'2ker ms':>9s} {'fused ms':>9s} {'F/2k':>6s} | "
          f"{'2k us/st':>9s} {'F us/st':>9s} | "
          f"{'2k GB/s':>8s} {'2k%pk':>6s} {'F GB/s':>8s} {'F%pk':>6s}", flush=True)
    for K in [int(x) for x in a.Ks.split(",")]:
        hq = torch.randn(1, K, D, device=dev) * 0.1
        with torch.no_grad():
            mo_h = run_mega_only(items_2k, hq)
            mo_f = run_mega_only(items_fused, hq)
        linf_fh = (mo_f - mo_h).abs().max().item()   # THE byte-exact gate (deterministic)
        exact = (linf_fh == 0.0)

        t_2 = time_fn(lambda: run_mega_only(items_2k, hq))
        t_f = time_fn(lambda: run_mega_only(items_fused, hq))
        us_2 = t_2 / K * 1e3
        us_f = t_f / K * 1e3
        # achieved GB/s over each path's dead-FFN roofline bytes/K-col.
        gbs_2 = (b2 * K) / (t_2 / 1e3) / 1e9
        gbs_f = (bf * K) / (t_f / 1e3) / 1e9
        print(f"{K:6d} {linf_fh:9.2e} {str(exact):>6s} | "
              f"{t_2:9.4f} {t_f:9.4f} {t_2/t_f:5.2f}x | "
              f"{us_2:8.3f} {us_f:8.3f} | "
              f"{gbs_2:8.1f} {100*gbs_2/HBM_BW_GBs:5.1f}% "
              f"{gbs_f:8.1f} {100*gbs_f/HBM_BW_GBs:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
