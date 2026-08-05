#!/usr/bin/env python3
"""_agent_ffn_cross_block_fusion.py — PROTOTYPE + MEASURE cross-block FFN fusion on the
DOOM-ACTIVE dead-FFN chain (the 49-block / 27-wave chain the block-skip already carves out).

Two levers, both DEFAULT-OFF, both BYTE-EXACT vs the doom-active 2-kernel skip baseline:
  (A) C4_FFN_WAVE_BATCH  — HORIZONTAL: the up-to-5 independent blocks per dependency wave
      concatenate into ONE up/gate + ONE down-delta launch (49*2 -> ~27*2 launches).
  (B) C4_FFN_LINFOLD     — VERTICAL: fold block N's W_down into block N+1's W_up/W_gate
      across the linear residual add (report fill-in factor; honest net-win verdict).

Reports: byte-exactness (L-inf vs 2-kernel baseline), kernel-eff (%HBM peak), us/step,
fps 1-GPU / 2-GPU, vs the 0.488us / 2.17fps block-skip baseline.  Also the byte-exact
gate on the REAL doom opcode stream (the 29-op corpus + the doom-lean region).
"""
from __future__ import annotations
import os
# MATCH the doom-active baseline build config EXACTLY (the 0.488us/49-block/dim-1416 chain
# the block-skip carves out): C4_PF_CFM=1 is what _agent_doom_occupancy sets, and it changes
# the model build (dim 1416 not 3404, 49 dead not 50).  Without it the baseline is a
# different, slower model and the numbers do NOT reconcile with the 0.488us reference.
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, time, torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import _frozen_skip_cut
from c4_min.tight_attn_compose import install_composed
from c4_min.step_block_skip import build_live_index
from c4_min.fused_megablock import (MegaBlockChain, MegaBlockRegion, _dense_of,
                                    divfree_carry_blocks)

HBM_BW_GBs = 768.0
RENDER_STEPS = 358_058
# whole-step non-chain term from the dispatch profile: live 0.69 + blk0 0.082 + decode 0.026
NONCHAIN = 0.69 + 0.082 + 0.026
COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _mem():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return float(l.split()[1]) / 1e6
    return 1e9


def doom_dead(model, L):
    li = build_live_index(model, L)
    union = set()
    for op in li:
        if op is None or op in (isa.DIV, isa.MOD):
            continue
        union.update(li[op])
    region = sorted(b for b in union if b >= 1)
    return [b for b in region if getattr(model.blocks[b].attn, "_dead_block_fused", False)]


def hbytes(model, dead):
    """hidden-scratch bytes/Kcol (the 92%-of-traffic term the %HBM is measured over)."""
    seen = set(); tot = 0
    for b in dead:
        ffn = model.blocks[b].ffn
        if id(ffn) in seen:
            continue
        seen.add(id(ffn)); tot += 2 * int(_dense_of(ffn.W_up).shape[0]) * 4
    return tot


def time_fn(fn, reps=30, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def _fps_line(name, us_step):
    sf = us_step * RENDER_STEPS / 1e6
    return (f"    {name:16s}: {us_step:.3f} us/step -> {sf:.4f} s/frame = "
            f"{1e6/(us_step*RENDER_STEPS):.2f} fps (1-GPU)  "
            f"{2e6/(us_step*RENDER_STEPS):.2f} fps (2-GPU)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--bk", type=int, default=64)
    a = ap.parse_args()
    if _mem() < 25:
        raise SystemExit(f"[GUARD] MemAvail {_mem():.1f}GB < 25 STOP")
    for f in COMPOSED:
        os.environ[f] = "1"
    dev = "cuda:0"; t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev); install_composed(model, verbose=False)
    cut = _frozen_skip_cut(model); Nb = len(model.blocks); D = model.dim
    dead = doom_dead(model, L)
    print(f"[built] n_blocks={Nb} dim={D} cut={cut} in {time.time()-t0:.1f}s  "
          f"DOOM-active dead={len(dead)}  MemAvail={_mem():.1f}GB", flush=True)

    K = a.chunk; bk = a.bk
    os.environ["C4_MEGABLOCK_BLOCK_K"] = str(bk)
    hb = hbytes(model, dead)
    hq = torch.randn(1, K, D, device=dev) * 0.1
    torch.cuda.reset_peak_memory_stats(dev)

    # ---- baseline: 2-kernel per-block (the block-skip doom-active chain) ----
    os.environ["C4_FFN_FUSED_HIDDEN"] = "0"; os.environ["C4_FFN_WAVE_BATCH"] = "0"
    ch_base = MegaBlockChain(model, dev, dead)
    with torch.no_grad():
        o_base = ch_base.run(hq).clone()
    t_base = time_fn(lambda: ch_base.run(hq))
    us_base = t_base / K * 1e3
    gbs_base = (hb * K) / (t_base / 1e3) / 1e9

    del ch_base
    # ---- (A) wave-batch ----
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    ch_wave = MegaBlockChain(model, dev, dead)
    with torch.no_grad():
        o_wave = ch_wave.run(hq)
    linf_wave = (o_base - o_wave).abs().max().item()
    t_wave = time_fn(lambda: ch_wave.run(hq))
    us_wave = t_wave / K * 1e3
    gbs_wave = (hb * K) / (t_wave / 1e3) / 1e9
    n_wave_launch = 2 * ch_wave.n_waves
    n_wave_parallel = ch_wave.max_wave_parallel
    n_waves = ch_wave.n_waves
    os.environ["C4_FFN_WAVE_BATCH"] = "0"

    del ch_wave
    print(f"\n=== chunk K={K} block_k={bk}  ({len(dead)} dead blocks; per-step = chain / K) ===",
          flush=True)
    print(f"{'variant':16s} {'launches':>9} {'us/step':>9} {'GB/s':>8} {'%HBM':>6} "
          f"{'Linf':>10} {'vs base':>8}", flush=True)
    n_base_launch = 2 * len(dead)
    print(f"{'2k baseline':16s} {n_base_launch:>9} {us_base:9.3f} {gbs_base:8.1f} "
          f"{100*gbs_base/HBM_BW_GBs:5.1f}% {'--':>10} {'1.00x':>8}", flush=True)
    print(f"{'(A) wave-batch':16s} {n_wave_launch:>9} {us_wave:9.3f} {gbs_wave:8.1f} "
          f"{100*gbs_wave/HBM_BW_GBs:5.1f}% {linf_wave:10.1e} {us_base/us_wave:7.2f}x",
          flush=True)
    print(f"   waves={n_waves}  max-parallel={n_wave_parallel}  "
          f"byte-exact(A)={'YES' if linf_wave == 0.0 else ('~ ' if linf_wave < 1e-3 else 'NO!!')}",
          flush=True)

    # ---- GRAPHED path (what production actually runs): run_graphed collapses the whole
    # chain into ONE CUDA-graph replay, so per-kernel LAUNCH overhead is already gone.
    # Wave-batch's launch reduction should add ~nothing here; measured honestly. ----
    os.environ["C4_FFN_WAVE_BATCH"] = "0"
    ch_bg = MegaBlockChain(model, dev, dead)
    with torch.no_grad():
        _ = ch_bg.run_graphed(hq)      # capture
    t_bg = time_fn(lambda: ch_bg.run_graphed(hq))
    us_bg = t_bg / K * 1e3
    del ch_bg
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    ch_wg = MegaBlockChain(model, dev, dead)
    with torch.no_grad():
        o_wg = ch_wg.run_graphed(hq)
    linf_wg = (o_base - o_wg).abs().max().item()
    t_wg = time_fn(lambda: ch_wg.run_graphed(hq))
    us_wg = t_wg / K * 1e3
    os.environ["C4_FFN_WAVE_BATCH"] = "0"
    del ch_wg
    print(f"{'2k GRAPHED':16s} {'1 replay':>9} {us_bg:9.3f} "
          f"{(hb*K)/(t_bg/1e3)/1e9:8.1f} {100*(hb*K)/(t_bg/1e3)/1e9/HBM_BW_GBs:5.1f}% "
          f"{'--':>10} {us_base/us_bg:7.2f}x", flush=True)
    print(f"{'(A)+GRAPHED':16s} {'1 replay':>9} {us_wg:9.3f} "
          f"{(hb*K)/(t_wg/1e3)/1e9:8.1f} {100*(hb*K)/(t_wg/1e3)/1e9/HBM_BW_GBs:5.1f}% "
          f"{linf_wg:10.1e} {us_base/us_wg:7.2f}x", flush=True)
    print(f"   GRAPHED note: run_graphed already collapses launches to 1 replay; "
          f"wave-batch-over-graph = {us_bg/us_wg:.2f}x", flush=True)
    peak_wave = torch.cuda.max_memory_allocated(dev) / 1e9   # VRAM for the 2k/wave path

    # ---- (B) lin-fold ----
    os.environ["C4_FFN_LINFOLD"] = "1"
    try:
        from c4_min.fused_megablock import LinFoldChain, linfold_fillin_report
        ch_lf = LinFoldChain(model, dev, dead, block_k=bk)
        with torch.no_grad():
            o_lf = ch_lf.run(hq)
        linf_lf = (o_base - o_lf).abs().max().item()
        t_lf = time_fn(lambda: ch_lf.run(hq))
        us_lf = t_lf / K * 1e3
        gbs_lf = (hb * K) / (t_lf / 1e3) / 1e9
        fillin = linfold_fillin_report(ch_lf)
        n_lf_launch = ch_lf.n_launches
        print(f"{'(B) lin-fold':16s} {n_lf_launch:>9} {us_lf:9.3f} {gbs_lf:8.1f} "
              f"{100*gbs_lf/HBM_BW_GBs:5.1f}% {linf_lf:10.1e} {us_base/us_lf:7.2f}x",
              flush=True)
        print(f"   folded pairs={fillin['n_folds']}  fill-in factor(fused nnz / factor nnz)="
              f"{fillin['fillin_factor']:.2f}x  ({fillin['fused_nnz']} vs {fillin['factor_nnz']})",
              flush=True)
        print(f"   byte-exact(B)={'YES' if linf_lf == 0.0 else ('~ ' if linf_lf < 1e-3 else 'NO!!')}  "
              f"net-win(B)={'YES' if us_lf < us_base*0.98 else 'NO (fill-in eats it)'}", flush=True)
        have_lf = True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[(B) lin-fold] FAILED to build/run: {e}", flush=True)
        us_lf = None; have_lf = False
    os.environ["C4_FFN_LINFOLD"] = "0"

    # ---- composed A+B ----
    us_ab = None
    if have_lf:
        os.environ["C4_FFN_WAVE_BATCH"] = "1"; os.environ["C4_FFN_LINFOLD"] = "1"
        try:
            ch_ab = LinFoldChain(model, dev, dead, block_k=bk, wave_batch=True)
            with torch.no_grad():
                o_ab = ch_ab.run(hq)
            linf_ab = (o_base - o_ab).abs().max().item()
            t_ab = time_fn(lambda: ch_ab.run(hq))
            us_ab = t_ab / K * 1e3
            gbs_ab = (hb * K) / (t_ab / 1e3) / 1e9
            print(f"{'(A+B) composed':16s} {ch_ab.n_launches:>9} {us_ab:9.3f} {gbs_ab:8.1f} "
                  f"{100*gbs_ab/HBM_BW_GBs:5.1f}% {linf_ab:10.1e} {us_base/us_ab:7.2f}x",
                  flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[(A+B)] FAILED: {e}", flush=True)
        os.environ["C4_FFN_WAVE_BATCH"] = "0"; os.environ["C4_FFN_LINFOLD"] = "0"

    peak = torch.cuda.max_memory_allocated(dev) / 1e9

    # ---- whole-step + fps projection ----
    print(f"\n=== WHOLE-STEP + fps (mega-chain + {NONCHAIN:.2f}us non-chain) ===", flush=True)
    print(_fps_line("2k baseline", us_base + NONCHAIN), flush=True)
    print(_fps_line("(A) wave-batch", us_wave + NONCHAIN), flush=True)
    if us_lf is not None:
        print(_fps_line("(B) lin-fold", us_lf + NONCHAIN), flush=True)
    if us_ab is not None:
        print(_fps_line("(A+B) composed", us_ab + NONCHAIN), flush=True)

    print(f"\n[VRAM peak] {peak:.2f} GB overall (dominated by lin-fold's dense A/B build); "
          f"{peak_wave:.2f} GB for the 2k/wave-batch path", flush=True)
    best = min([x for x in (us_base, us_wave, us_lf, us_ab) if x is not None])
    print(f"[best doom-active mega-chain] {best:.3f} us/step  "
          f"(vs 0.488us block-skip baseline: {0.488/best:.2f}x)", flush=True)
    print(_fps_line("BEST whole-step", best + NONCHAIN), flush=True)


if __name__ == "__main__":
    main()
