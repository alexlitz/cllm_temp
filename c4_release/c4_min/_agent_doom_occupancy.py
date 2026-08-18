#!/usr/bin/env python3
"""_agent_doom_occupancy.py — MEASURE us/step + %HBM peak for the DOOM-ACTIVE dead-FFN
chain (DIV cascade + non-doom blocks SKIPPED) vs the STATIC FULL 238-block chain, at the
doom render chunk.  (Recreated from 97e4ce9e; the pinned worktree base e28935ba lacks it.)"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM","1"); os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import argparse, time, torch
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import _frozen_skip_cut
from c4_min.tight_attn_compose import install_composed
from c4_min.step_block_skip import build_live_index
from c4_min.fused_megablock import MegaBlockChain, _dense_of
HBM_BW_GBs=768.0; RENDER_STEPS=358_058
COMPOSED=["C4_DEAD_BLOCK_FUSION","C4_DIRECT_CAM_BATCHED","C4_DIRECT_LOCAL_CAM","C4_FLASH_ATTN","C4_BANDED_LOCAL_ATTN","C4_FUSED_MEGABLOCK","C4_DIRECT_CAM_VEC"]

def _mem():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"): return float(l.split()[1])/1e6
    return 1e9

def doom_dead(model,L):
    li=build_live_index(model,L); union=set()
    for op in li:
        if op is None or op in (isa.DIV,isa.MOD): continue
        union.update(li[op])
    region=sorted(b for b in union if b>=1)
    return [b for b in region if getattr(model.blocks[b].attn,"_dead_block_fused",False)]

def hbytes(model,dead):
    seen=set(); tot=0
    for b in dead:
        ffn=model.blocks[b].ffn
        if id(ffn) in seen: continue
        seen.add(id(ffn)); tot+=2*int(_dense_of(ffn.W_up).shape[0])*4
    return tot

def time_fn(fn,reps=30,warmup=10):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize()
    return (time.perf_counter()-t0)/reps*1e3

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--chunk",type=int,default=65536)
    ap.add_argument("--bk",type=int,default=64); a=ap.parse_args()
    if _mem()<25: raise SystemExit(f"[GUARD] {_mem():.1f}GB<25 STOP")
    for f in COMPOSED: os.environ[f]="1"
    dev="cuda:0"; t0=time.time()
    model,L,_=build_lib_model_streaming(code_size=256,recurrent_divmod=True,addr32=True,compute_mode="dense_kernel")
    model=model.to(dev); install_composed(model,verbose=False)
    cut=_frozen_skip_cut(model); Nb=len(model.blocks); D=model.dim
    full_dead=[b for b in range(cut,Nb) if getattr(model.blocks[b].attn,"_dead_block_fused",False)]
    doom=doom_dead(model,L)
    print(f"built n_blocks={Nb} dim={D} in {time.time()-t0:.1f}s  FULL dead={len(full_dead)}  DOOM-active dead={len(doom)}  MemAvail={_mem():.1f}GB",flush=True)
    K=a.chunk; bk=a.bk
    hq=torch.randn(1,K,D,device=dev)*0.1
    os.environ["C4_MEGABLOCK_BLOCK_K"]=str(bk)
    torch.cuda.reset_peak_memory_stats(dev)
    print(f"\nchunk K={K} block_k={bk}  (per-step = whole-chain time / K)",flush=True)
    print(f"{'chain':>14} {'blocks':>7} | {'2k us/st':>9} {'F us/st':>9} {'F/2k':>6} | {'2k GB/s':>8} {'2k%pk':>6} {'F GB/s':>8} {'F%pk':>6}",flush=True)
    rows={}
    for name,dead in (("FULL-static",full_dead),("DOOM-active",doom)):
        hb=hbytes(model,dead)
        os.environ["C4_FFN_FUSED_HIDDEN"]="0"; ch2=MegaBlockChain(model,dev,dead)
        os.environ["C4_FFN_FUSED_HIDDEN"]="1"; chf=MegaBlockChain(model,dev,dead)
        os.environ["C4_FFN_FUSED_HIDDEN"]="0"
        with torch.no_grad(): o2=ch2.run(hq); of=chf.run(hq)
        linf=(o2-of).abs().max().item()
        t2=time_fn(lambda: ch2.run(hq)); tf=time_fn(lambda: chf.run(hq))
        us2=t2/K*1e3; usf=tf/K*1e3
        gbs2=(hb*K)/(t2/1e3)/1e9; gbsf=(hb*K)/(tf/1e3)/1e9
        print(f"{name:>14} {len(dead):>7} | {us2:8.3f} {usf:8.3f} {t2/tf:5.2f}x | {gbs2:8.1f} {100*gbs2/HBM_BW_GBs:5.1f}% {gbsf:8.1f} {100*gbsf/HBM_BW_GBs:5.1f}%   Linf(FvH)={linf:.1e}",flush=True)
        rows[name]=(us2,usf)
        del ch2,chf
    peak=torch.cuda.max_memory_allocated(dev)/1e9
    print(f"\n[VRAM peak] {peak:.2f} GB",flush=True)
    fu2,fuf=rows["FULL-static"]; du2,duf=rows["DOOM-active"]
    print(f"  DOOM-active mega chain  : {du2:.3f} us/step (2k) / {duf:.3f} (fused)",flush=True)
    NONCHAIN=0.69+0.082+0.026
    doom_step=du2+NONCHAIN
    print(f"    DOOM-active: {doom_step:.3f} us/step -> {doom_step*RENDER_STEPS/1e6:.4f} s/frame = {1e6/(doom_step*RENDER_STEPS):.2f} fps (1-GPU)  {2e6/(doom_step*RENDER_STEPS):.2f} fps (2-GPU)",flush=True)

if __name__=="__main__": main()
