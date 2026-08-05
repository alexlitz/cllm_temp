#!/usr/bin/env python3
"""FAITHFUL SINGLE-DISPATCH continuous fps on the REAL render-reduced doom frame.

Runs the byte-exact-verified id-doom bytecode through the FAITHFUL single-dispatch
(C4_FAITHFUL_SINGLE_DISPATCH): the FAST fused per-row map (byte-exact decode) PLUS the
independent routing/address/value verify from the model's genuine computation.  Measures
the continuous per-frame wall scaled to the 358,058-step render-reduced frame + a full
cost breakdown (fast dispatch vs the genuine verify vs its sub-parts), and compares to
the draft-trusted fast path (this same harness with the verify OFF).

Run: CUDA_VISIBLE_DEVICES=0 python -m c4_min._agent_faithful_sd_doom_fps --draft-steps 40000
"""
from __future__ import annotations
import argparse, os, sys, time, gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_CMP32", "1")
os.environ.setdefault("C4_CMP32_ORDER", "1")
os.environ.setdefault("C4_SHIFT32", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("C4_EXACT_EVICT", "1")
os.environ.setdefault("C4_MEM_EFF", "1")
os.environ.setdefault("C4_IMM_NIBS", "6")
os.environ.setdefault("C4_PC_WIDE", "1")
os.environ.setdefault("C4_CODE_ADDR_BITS", "20")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DOOM = "/home/alexlitz/Documents/misc/c4_doom"
sys.path.insert(0, DOOM)

import numpy as np
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0x10000

from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, PFDraft
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min import precomputed_schedule as PS
from c4_min.faithful_single_dispatch import (build_faithful_plan, verify_faithful,
                                             build_faithful_precompute, verify_faithful_fast,
                                             _read_frame_of_step)
from run_c4_min import (data_segment, tag_compiler_syscalls,
                        install_compiler_abi_file_dispatcher)

RENDER_REDUCED_FRAME = 358_058
RAW_FRAME = 6_889_264

COMPOSED = ["C4_DEAD_BLOCK_FUSION","C4_DIRECT_CAM_BATCHED","C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN","C4_BANDED_LOCAL_ATTN","C4_FUSED_MEGABLOCK","C4_DIRECT_CAM_VEC"]

def _levers_on(chunk):
    for f in COMPOSED: os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"]="1"; os.environ["C4_RESIDENT_BATCH"]="1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"]="1"; os.environ["C4_SCHED_FAST_BUILD"]="1"
    os.environ["C4_SCHED_GPU_BUILD"]="1"; os.environ["C4_SCHED_CHUNK"]=str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"]="1"; os.environ["C4_MEGABLOCK_BLOCK_K"]="512"

def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"): return int(ln.split()[1]) / 1e6
    return 1e9

def _guard():
    a = _mem_avail_gb()
    if a < 25.0: raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB<25 -> STOP")

def _apply_pow2(ops, imms):
    IMM_OP,DIV_OP,MOD_OP,SHR_OP,AND_OP = 1,28,29,24,16
    n=len(ops); npow=0
    for i in range(n-1):
        o0,m0,o1=int(ops[i]),int(imms[i]),int(ops[i+1])
        if o0==IMM_OP and m0>0 and (m0&(m0-1))==0:
            if o1==DIV_OP: imms[i]=m0.bit_length()-1; ops[i+1]=SHR_OP; npow+=1
            elif o1==MOD_OP: imms[i]=m0-1; ops[i+1]=AND_OP; npow+=1
    return npow

def _load_snapshot():
    snap = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..","_doom_bytecode_snapshot.npz"))
    return list(snap["ops"]), list(snap["imms"]), snap["data"]

def build_doom_draft(draft_steps):
    ops,imms,data=_load_snapshot(); n=len(ops)
    npow=_apply_pow2(ops,imms)
    print(f"[doom] pow2 strength-reduction: {npow} DIV/MOD sites -> SHR/AND", flush=True)
    install_compiler_abi_file_dispatcher()
    code_isa=tag_compiler_syscalls([isa.Instr(int(ops[i]),int(imms[i])&0xFFFFFFFF) for i in range(n)], isa)
    fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    print(f"[doom] drafting REAL doom bytecode ({n:,} instrs), max_steps={draft_steps:,} ...", flush=True)
    t0=time.time()
    d=draft_pf_program(code_isa, max_steps=draft_steps, mask=0xFFFFFFFF,
                       data_seg=data_segment([int(b) for b in data]), fio=fio)
    print(f"[doom] drafted {d.step_count:,} steps in {time.time()-t0:.0f}s", flush=True)
    dm=[i for i in range(d.step_count) if d.frames[i].get("op") in ("DIV","MOD")]
    assert not dm, f"draft not DIV-free: {dm[:5]}"
    print(f"[doom] DIV-free real-doom trace: {d.step_count:,} steps", flush=True)
    return d

def _dispatch(sched, sg, dev, n):
    got_pc=torch.empty(n,dtype=torch.long,device=dev); got_sp=torch.empty(n,dtype=torch.long,device=dev)
    got_bp=torch.empty(n,dtype=torch.long,device=dev); got_ax=torch.empty(n,dtype=torch.long,device=dev)
    onchip=sched.onchip; h0s=sched.h0_folded if onchip else sched.h0_table
    for lo in range(0,n,sg.chunk):
        hi=min(lo+sg.chunk,n); h0=h0s[lo:hi].unsqueeze(0)
        if onchip:
            delta={b:t[lo:hi] for b,t in sched.cam_delta_tables.items()}
            pc,sp,bp,ax=sg.replay(h0,None,None,delta=delta,resident=False)
        else:
            ing=sched.ing_table[lo:hi].permute(1,0,2).unsqueeze(0)
            cam={b:t[lo:hi].permute(1,0,2).unsqueeze(0) for b,t in sched.cam_tables.items()}
            pc,sp,bp,ax=sg.replay(h0,ing,cam,resident=False)
        got_pc[lo:hi].copy_(pc); got_sp[lo:hi].copy_(sp); got_bp[lo:hi].copy_(bp); got_ax[lo:hi].copy_(ax)
    return got_pc,got_sp,got_bp,got_ax

def _decode_queries(sched, sg, dev, n):
    """The genuine verify's GPU part: decode the model's query addresses (W_q(x) +
    sign-decode) at every live CAM block, for the whole batch (chunked, eager)."""
    onchip=sched.onchip; h0s=sched.h0_folded if onchip else sched.h0_table
    per_head={}
    for lo in range(0,n,sg.chunk):
        hi=min(lo+sg.chunk,n); h0=h0s[lo:hi].unsqueeze(0)
        if onchip:
            delta={b:t[lo:hi] for b,t in sched.cam_delta_tables.items()}
            d=sg.decode_model_query_addrs(h0,None,None,delta=delta)
        else:
            ing=sched.ing_table[lo:hi].permute(1,0,2).unsqueeze(0)
            cam={b:t[lo:hi].permute(1,0,2).unsqueeze(0) for b,t in sched.cam_tables.items()}
            d=sg.decode_model_query_addrs(h0,ing,cam)
        for key,t in d.items():
            per_head.setdefault(key,[]).append((lo,hi,t.detach().to("cpu").numpy().astype("int64")))
    model_addrs={}
    for (b,hh,knd),chunks in per_head.items():
        full=np.zeros(n,dtype=np.int64)
        for (lo,hi,arr) in chunks: full[lo:hi]=arr[:hi-lo]
        model_addrs[(b,knd)]=full
    return model_addrs

def _dispatch_with_qaddr(sched, sg, dev, n):
    """The FAITHFUL dispatch: one graph replay per chunk producing BOTH the register
    decode AND the in-graph model query addresses (last_qaddr)."""
    got_pc=torch.empty(n,dtype=torch.long,device=dev); got_sp=torch.empty(n,dtype=torch.long,device=dev)
    got_bp=torch.empty(n,dtype=torch.long,device=dev); got_ax=torch.empty(n,dtype=torch.long,device=dev)
    onchip=sched.onchip; h0s=sched.h0_folded if onchip else sched.h0_table
    per_head={}
    for lo in range(0,n,sg.chunk):
        hi=min(lo+sg.chunk,n); h0=h0s[lo:hi].unsqueeze(0)
        if onchip:
            delta={b:t[lo:hi] for b,t in sched.cam_delta_tables.items()}
            pc,sp,bp,ax=sg.replay(h0,None,None,delta=delta,resident=False)
        else:
            ing=sched.ing_table[lo:hi].permute(1,0,2).unsqueeze(0)
            cam={b:t[lo:hi].permute(1,0,2).unsqueeze(0) for b,t in sched.cam_tables.items()}
            pc,sp,bp,ax=sg.replay(h0,ing,cam,resident=False)
        got_pc[lo:hi].copy_(pc); got_sp[lo:hi].copy_(sp); got_bp[lo:hi].copy_(bp); got_ax[lo:hi].copy_(ax)
        for key,t in sg.last_qaddr(hi-lo).items():
            per_head.setdefault(key,[]).append((lo,hi,t.detach().to("cpu").numpy().astype("int64")))
    model_addrs={}
    for (b,hh,knd),chunks in per_head.items():
        full=np.zeros(n,dtype=np.int64)
        for (lo,hi,arr) in chunks: full[lo:hi]=arr[:hi-lo]
        model_addrs[(b,knd)]=full
    return got_pc,got_sp,got_bp,got_ax,model_addrs


def _serial_frame(sg, sched_builder, dev, n, faithful, plan, ws, draft, mask):
    """One serial frame: build tables (+ faithful value-verify PRECOMPUTE) + dispatch (+ the
    cheap vectorized value-verify COMPARE if faithful).  Returns (build_s, dispatch_s,
    verify_s).  ``build_s`` now INCLUDES the genuine value re-resolution precompute (pure
    numpy — a draft-only function of the store-log — which in TRUE-PIPE overlaps the prior
    frame's replay on the build thread); only the cheap ``verify_faithful_fast`` compare
    stays on the critical path (``verify_s``)."""
    t0=time.perf_counter()
    s2 = sched_builder()
    pre = build_faithful_precompute(draft, plan, ws, n, mask=mask) if faithful else None
    torch.cuda.synchronize(dev); b=time.perf_counter()-t0
    torch.cuda.synchronize(dev); t0=time.perf_counter()
    if faithful:
        _p,_s,_bp,_a,ma = _dispatch_with_qaddr(s2, sg, dev, n)
        torch.cuda.synchronize(dev); d=time.perf_counter()-t0
        t0=time.perf_counter(); verify_faithful_fast(pre, ma); v=time.perf_counter()-t0
    else:
        _dispatch(s2, sg, dev, n)
        torch.cuda.synchronize(dev); d=time.perf_counter()-t0; v=0.0
    del s2; gc.collect()
    return b,d,v

def _measure_one(model, L, code, draft, device, chunk, n_frames, faithful):
    """Measure ONE path (fast or faithful) with only ITS graph resident.  Returns serial +
    TRUE-PIPE per-frame medians + peak VRAM + byte-exact verdict."""
    dev=torch.device(device); n=draft.step_count; mask=0xFFFFFFFF
    _levers_on(chunk)
    if faithful: os.environ["C4_FAITHFUL_SINGLE_DISPATCH"]="1"
    else: os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    torch.cuda.reset_peak_memory_stats(dev)
    from c4_min.faithful_single_dispatch import build_faithful_plan, verify_faithful
    plan = build_faithful_plan(model, L, code, draft) if faithful else None
    ws = np.asarray(draft.win_starts[:n], dtype=np.int64)
    torch.cuda.synchronize(dev); t0=time.perf_counter()
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
    if faithful: _dispatch_with_qaddr(sched, sg, dev, n)
    else: _dispatch(sched, sg, dev, n)
    torch.cuda.synchronize(dev); t_build0=time.perf_counter()-t0
    peak=torch.cuda.max_memory_allocated(dev)/1e9
    del sched; gc.collect(); torch.cuda.empty_cache()
    tag = "FAITHFUL" if faithful else "FAST"
    print(f"  [{tag}] one-time build+capture {t_build0*1e3:.0f} ms  chunk={sg.chunk}  peak_vram {peak:.1f} GB", flush=True)
    # serial medians.
    B=[];D=[];VER=[]
    for _ in range(n_frames):
        gc.collect(); torch.cuda.empty_cache()
        b,d,v = _serial_frame(sg, lambda: PS.build_schedule_tables_only(model,L,code,draft,dev,sg,mask=mask),
                              dev, n, faithful, plan, ws, draft, mask)
        B.append(b);D.append(d);VER.append(v)
    b=np.median(B); d=np.median(D); ver=np.median(VER)
    total = b + d + ver
    # byte-exact + verdict (faithful) on a fresh frame.
    gc.collect(); torch.cuda.empty_cache()
    s_ser = PS.build_schedule_tables_only(model,L,code,draft,dev,sg,mask=mask)
    torch.cuda.synchronize(dev)
    if faithful:
        pc,sp,bp2,ax,ma=_dispatch_with_qaddr(s_ser,sg,dev,n); ax=ax&mask
        pre_ser=build_faithful_precompute(draft, plan, ws, n, mask=mask)
        vd=verify_faithful_fast(pre_ser, ma)
        # cross-check: the pipelined/vectorized verdict == the original verify_faithful.
        vd_ref=verify_faithful(draft, plan, ma, ws, mask=mask)
        assert (vd.ok==vd_ref.ok and vd.first_bad_step==vd_ref.first_bad_step
                and vd.kind==vd_ref.kind), f"pipelined verdict != verify_faithful: {vd} vs {vd_ref}"
    else:
        pc,sp,bp2,ax=_dispatch(s_ser,sg,dev,n); ax=ax&mask; vd=None
    bad=(((pc!=s_ser.want_pc)|(ax!=(s_ser.want_ax&mask))|(sp!=s_ser.want_sp)|(bp2!=s_ser.want_bp)))
    bad=torch.where(s_ser.is_halt, ax!=(s_ser.want_ax&mask), bad)&(~s_ser.is_file)
    nbad=int(bad.sum())
    del s_ser; gc.collect(); torch.cuda.empty_cache()
    # TRUE-PIPE.  The faithful value-verify PRECOMPUTE (the genuine latest-write-wins value
    # re-resolution) now rides on the SAME background build thread as the schedule build
    # (``faithful=True`` -> ``PipelinedScheduleBuilder`` runs ``build_faithful_precompute`` on
    # its thread concurrent with the prior frame's replay).  Only the cheap vectorized
    # ``verify_faithful_fast`` compare (model addrs vs precompute) stays on the critical path.
    os.environ["C4_SCHED_PIPELINE"]="1"
    WARMUP=2
    pb = PS.PipelinedScheduleBuilder(model, L, code, draft, dev, sg, mask=mask, faithful=faithful)
    cur = pb.build_blocking(); pre_cur=pb.last_precompute(); torch.cuda.synchronize(dev); pb.start()
    walls=[]; tot=n_frames+WARMUP
    for i2 in range(tot):
        torch.cuda.synchronize(dev); tf=time.perf_counter()
        if faithful:
            _p,_s,_b,_a,ma=_dispatch_with_qaddr(cur,sg,dev,n)
        else:
            _dispatch(cur,sg,dev,n)
        # the cheap vectorized compare uses THIS frame's precompute (built on the prior
        # thread iteration), overlapped-hidden; then join the build thread + swap.
        if faithful: verify_faithful_fast(pre_cur, ma)
        nxt=pb.wait(); del cur; cur=nxt; pre_cur=pb.last_precompute()
        if i2<tot-1: pb.start()
        torch.cuda.synchronize(dev); walls.append(time.perf_counter()-tf)
    del cur; gc.collect(); torch.cuda.empty_cache()
    os.environ.pop("C4_SCHED_PIPELINE", None)
    steady=sorted(walls[WARMUP:]); pipe=steady[len(steady)//2]
    del sg; gc.collect(); torch.cuda.empty_cache()
    os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    return dict(build=b, disp=d, ver=ver, serial=total, pipe=pipe, peak=peak,
                nbad=nbad, ok=(vd.ok if vd else None), vd=vd, n=n)

def measure(model, L, code, draft, device, chunk, n_frames):
    n=draft.step_count
    print("\n--- FAST path (draft-trusted) ---", flush=True)
    rf = _measure_one(model, L, code, draft, device, chunk, n_frames, faithful=False)
    print("\n--- FAITHFUL single-dispatch ---", flush=True)
    ra = _measure_one(model, L, code, draft, device, chunk, n_frames, faithful=True)
    print(f"\n  per-frame medians ({n} steps):", flush=True)
    print(f"    FAST     build {rf['build']*1e3:7.2f} + dispatch {rf['disp']*1e3:7.2f}"
          f"                     = serial {rf['serial']*1e3:7.2f} ms  pipe {rf['pipe']*1e3:7.2f} ms", flush=True)
    print(f"    FAITHFUL build {ra['build']*1e3:7.2f} + dispatch {ra['disp']*1e3:7.2f}"
          f" (in-graph addr) + value {ra['ver']*1e3:6.2f} = serial {ra['serial']*1e3:7.2f} ms  pipe {ra['pipe']*1e3:7.2f} ms", flush=True)
    print(f"    faithful dispatch adds {(ra['disp']-rf['disp'])*1e3:.2f} ms (in-graph W_q sign-decode); value-verify {ra['ver']*1e3:.2f} ms (CPU)", flush=True)
    print(f"    verify accepts all={ra['ok']} addr_chk={ra['vd'].n_addr_checked} val_chk={ra['vd'].n_value_checked} rt_chk={ra['vd'].n_routing_checked}", flush=True)
    print(f"  [BYTE-EXACT] faithful decode vs draft mismatches={ra['nbad']}  fast mismatches={rf['nbad']}", flush=True)
    return dict(fast_serial=rf['serial'], faith_serial=ra['serial'],
                fast_pipe=rf['pipe'], faith_pipe=ra['pipe'],
                build_f=ra['build'], disp_f=ra['disp'], disp_p=rf['disp'], ver=ra['ver'],
                n=n, peak=max(rf['peak'], ra['peak']), peak_faith=ra['peak'],
                byte_exact=(ra['nbad']==0 and ra['ok']))


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--draft-steps",type=int,default=40000)
    ap.add_argument("--chunk",type=int,default=131072)
    ap.add_argument("--n-frames",type=int,default=4)
    args=ap.parse_args(argv)
    _guard(); device=args.device
    d=build_doom_draft(args.draft_steps); _guard()
    t0=time.time()
    model,L,_=build_lib_model_streaming(code_size=max(d.code_off+2,256),
                                        recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
    model=model.to(device)
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} {time.time()-t0:.1f}s", flush=True)
    _guard()
    ops,imms,_data=_load_snapshot(); _apply_pow2(ops,imms)
    code=tag_compiler_syscalls([isa.Instr(int(o),int(i)&0xFFFFFFFF) for o,i in zip(ops,imms)], isa)
    install_composed(model, verbose=False)
    try:
        print(f"\n=== FAITHFUL SINGLE-DISPATCH REAL-DOOM ({d.step_count} DIV-free steps) ===", flush=True)
        r=measure(model,L,code,d,device,args.chunk,n_frames=args.n_frames)
    finally:
        uninstall_composed(model)
    n=r["n"]
    print(f"\n  === REAL-DOOM per-frame @ render-reduced ({RENDER_REDUCED_FRAME} steps) ===", flush=True)
    print(f"    {'path':>28}  {'SERIAL':>22}  {'TRUE-PIPE':>22}", flush=True)
    for label,ser,pipe in (("fast (draft-trusted)", r["fast_serial"], r["fast_pipe"]),
                           ("FAITHFUL single-dispatch", r["faith_serial"], r["faith_pipe"])):
        ss=ser*(RENDER_REDUCED_FRAME/n); sp=pipe*(RENDER_REDUCED_FRAME/n)
        print(f"    {label:>28}  {ss:8.3f}s {1.0/ss:6.3f}fps  {sp:8.3f}s {1.0/sp:6.3f}fps"
              f"  {'>=1!' if 1.0/sp>=1.0 else '<1'}", flush=True)
    print(f"\n  COST BREAKDOWN (us/step): faithful build {r['build_f']/n*1e6:.3f} | "
          f"faithful dispatch(incl in-graph addr) {r['disp_f']/n*1e6:.3f} | "
          f"fast dispatch {r['disp_p']/n*1e6:.3f} | value-verify(CPU) {r['ver']/n*1e6:.3f}", flush=True)
    print(f"  faithful/fast slowdown: serial {r['faith_serial']/r['fast_serial']:.2f}x | "
          f"pipe {r['faith_pipe']/r['fast_pipe']:.2f}x", flush=True)
    print(f"  BYTE-EXACT (faithful decode==draft AND verify accepts all): {r['byte_exact']}", flush=True)
    print(f"  VRAM peak (faithful graph only) {r['peak_faith']:.1f} GB", flush=True)
    print("\n=== COMPLETE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
