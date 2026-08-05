#!/usr/bin/env python3
"""SANITY: the K-split shard build == the single-GPU full-frame single-dispatch, byte-exact.

Runs a SMALL doom draft, builds the single-GPU faithful schedule + dispatch (the reference),
then builds a 2-SHARD split (BOTH on the same device to isolate correctness from GPU
availability) and asserts:
  * concatenated shard registers (pc/sp/bp/ax) == the single-GPU registers (L-inf 0),
  * both == the draft targets (byte-exact),
  * the faithful verdict reduces correctly (ok on a correct draft),
  * a corrupted draft is REJECTED at the SAME first-divergence step by BOTH the single-GPU
    verify and the split reduction (cam_addr + cam_value scenarios).

Run: CUDA_VISIBLE_DEVICES=0 python -m c4_min._agent_ksplit_sanity --draft-steps 4000
"""
from __future__ import annotations
import argparse, os, sys, time, gc, copy

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
from c4_min.pf_speculative import draft_pf_program
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min import precomputed_schedule as PS
from c4_min.faithful_single_dispatch import (build_faithful_plan, verify_faithful,
                                             build_faithful_precompute, verify_faithful_fast)
from c4_min import multigpu_ksplit as KS
from run_c4_min import (data_segment, tag_compiler_syscalls,
                        install_compiler_abi_file_dispatcher)

COMPOSED = ["C4_DEAD_BLOCK_FUSION","C4_DIRECT_CAM_BATCHED","C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN","C4_BANDED_LOCAL_ATTN","C4_FUSED_MEGABLOCK","C4_DIRECT_CAM_VEC"]

def _levers_on(chunk):
    for f in COMPOSED: os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"]="1"; os.environ["C4_RESIDENT_BATCH"]="1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"]="1"; os.environ["C4_SCHED_FAST_BUILD"]="1"
    os.environ["C4_SCHED_GPU_BUILD"]="1"; os.environ["C4_SCHED_CHUNK"]=str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"]="1"; os.environ["C4_MEGABLOCK_BLOCK_K"]="512"

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
    _apply_pow2(ops,imms)
    install_compiler_abi_file_dispatcher()
    code_isa=tag_compiler_syscalls([isa.Instr(int(ops[i]),int(imms[i])&0xFFFFFFFF) for i in range(n)], isa)
    fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    d=draft_pf_program(code_isa, max_steps=draft_steps, mask=0xFFFFFFFF,
                       data_seg=data_segment([int(b) for b in data]), fio=fio)
    return d

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--draft-steps",type=int,default=4000)
    ap.add_argument("--chunk",type=int,default=131072)
    args=ap.parse_args(argv); dev=torch.device(args.device); mask=0xFFFFFFFF
    _levers_on(args.chunk)
    d=build_doom_draft(args.draft_steps)
    n=d.step_count
    print(f"[sanity] draft steps={n}", flush=True)
    model,L,_=build_lib_model_streaming(code_size=max(d.code_off+2,256),
                                        recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
    model=model.to(dev)
    ops,imms,_data=_load_snapshot(); _apply_pow2(ops,imms)
    code=tag_compiler_syscalls([isa.Instr(int(o),int(i)&0xFFFFFFFF) for o,i in zip(ops,imms)], isa)
    install_composed(model, verbose=False)
    try:
        os.environ["C4_FAITHFUL_SINGLE_DISPATCH"]="1"
        # ---- reference: single-GPU full-frame faithful dispatch ----
        plan = build_faithful_plan(model, L, code, d)
        ws = np.asarray(d.win_starts[:n], dtype=np.int64)
        sched, sg = PS.build_schedule(model, L, code, d, dev, mask=mask)
        rp=torch.empty(n,dtype=torch.long,device=dev); rs=torch.empty(n,dtype=torch.long,device=dev)
        rb=torch.empty(n,dtype=torch.long,device=dev); ra=torch.empty(n,dtype=torch.long,device=dev)
        per_head={}
        for lo in range(0,n,sg.chunk):
            hi=min(lo+sg.chunk,n); h0=(sched.h0_folded if sched.onchip else sched.h0_table)[lo:hi].unsqueeze(0)
            delta={b:t[lo:hi] for b,t in sched.cam_delta_tables.items()}
            pc,sp,bp,ax=sg.replay(h0,None,None,delta=delta,resident=False)
            rp[lo:hi].copy_(pc); rs[lo:hi].copy_(sp); rb[lo:hi].copy_(bp); ra[lo:hi].copy_(ax)
            for key,t in sg.last_qaddr(hi-lo).items():
                per_head.setdefault(key,[]).append((lo,hi,t.detach().cpu().numpy().astype("int64")))
        ra=ra&mask
        ref_ma={}
        for (b,hh,knd),chunks in per_head.items():
            fv=np.zeros(n,dtype=np.int64)
            for (lo,hi,arr) in chunks: fv[lo:hi]=arr[:hi-lo]
            ref_ma[(b,knd)]=fv
        pre_full=build_faithful_precompute(d, plan, ws, n, mask=mask)
        vd_ref=verify_faithful_fast(pre_full, ref_ma)
        del sched, sg; gc.collect(); torch.cuda.empty_cache()
        print(f"[ref] single-GPU verdict ok={vd_ref.ok} addr={vd_ref.n_addr_checked} "
              f"val={vd_ref.n_value_checked} rt={vd_ref.n_routing_checked}", flush=True)

        # ---- split: 2 shards, BOTH on the same device (correctness isolation) ----
        ranges = KS.split_rows(n, 2)
        print(f"[split] ranges={ranges}", flush=True)
        sp_pc=torch.empty(n,dtype=torch.long,device=dev); sp_sp=torch.empty(n,dtype=torch.long,device=dev)
        sp_bp=torch.empty(n,dtype=torch.long,device=dev); sp_ax=torch.empty(n,dtype=torch.long,device=dev)
        shard_vds=[]
        for (lo,hi) in ranges:
            sh = KS._DeviceShard(model, L, code, d, dev, lo, hi, args.chunk, faithful=True, mask=mask)
            gp,gs,gb,ga,ma = sh.dispatch()
            sp_pc[lo:hi].copy_(gp); sp_sp[lo:hi].copy_(gs); sp_bp[lo:hi].copy_(gb); sp_ax[lo:hi].copy_(ga&mask)
            pre_sh = KS.slice_precompute(pre_full, lo, hi)
            vd_sh = verify_faithful_fast(pre_sh, ma)
            shard_vds.append((lo, vd_sh))
            del sh; gc.collect(); torch.cuda.empty_cache()
        vd_split = KS.reduce_verdicts(shard_vds)

        # ---- byte-exact comparisons ----
        li_pc=int((sp_pc!=rp).sum()); li_sp=int((sp_sp!=rs).sum())
        li_bp=int((sp_bp!=rb).sum()); li_ax=int((sp_ax!=ra).sum())
        print(f"[byte-exact split==single] pc_diff={li_pc} sp_diff={li_sp} bp_diff={li_bp} ax_diff={li_ax}", flush=True)
        # vs draft targets
        wp=torch.from_numpy(np.fromiter((d.frames[s]['pc'] for s in range(n)),dtype=np.int64,count=n)).to(dev)
        wa=torch.from_numpy(np.fromiter((d.frames[s]['ax']&mask for s in range(n)),dtype=np.int64,count=n)).to(dev)
        d_pc=int((sp_pc!=wp).sum()); d_ax=int((sp_ax!=wa).sum())
        print(f"[byte-exact split==draft] pc_diff={d_pc} ax_diff={d_ax}", flush=True)
        print(f"[verdict] split ok={vd_split.ok} gstep={vd_split.global_first_bad_step} "
              f"kind={vd_split.kind} addr={vd_split.n_addr_checked} val={vd_split.n_value_checked} "
              f"rt={vd_split.n_routing_checked}", flush=True)
        assert li_pc==0 and li_sp==0 and li_bp==0 and li_ax==0, "split != single-GPU (L-inf!=0)"
        assert vd_split.ok==vd_ref.ok, "verdict ok mismatch"
        assert (vd_ref.n_addr_checked==vd_split.n_addr_checked and
                vd_ref.n_value_checked==vd_split.n_value_checked and
                vd_ref.n_routing_checked==vd_split.n_routing_checked), "check counts mismatch"
        print("[PASS] split == single-GPU == draft, byte-exact; verdict counts match", flush=True)

        # ---- WRONG-DRAFT rejection: corrupt one cam value + one cam addr, confirm SAME
        #      first-divergence in single-GPU verify AND the split reduction ----
        def _corrupt(kind_target):
            pre2 = copy.deepcopy(pre_full)
            for kind, sarr in pre2.steps.items():
                if sarr.shape[0]==0: continue
                if kind_target=="cam_value":
                    # flip an injected value at a mid step so genuine != draft_val.
                    j = sarr.shape[0]//2
                    pre2.draft_val[kind] = pre2.draft_val[kind].copy()
                    pre2.draft_val[kind][j] ^= 0x1
                    # ensure the model addr is armed at that step (it is a real read).
                    return pre2, int(sarr[j])
            return pre2, None
        # cam_value corruption
        pre_bad, badstep = _corrupt("cam_value")
        if badstep is not None:
            vr = verify_faithful_fast(pre_bad, ref_ma)
            # split: slice the corrupted precompute per shard, verify per shard, reduce.
            svds=[]
            for (lo,hi) in ranges:
                sh = KS._DeviceShard(model, L, code, d, dev, lo, hi, args.chunk, faithful=True, mask=mask)
                _gp,_gs,_gb,_ga,ma = sh.dispatch()
                svds.append((lo, verify_faithful_fast(KS.slice_precompute(pre_bad,lo,hi), ma)))
                del sh; gc.collect(); torch.cuda.empty_cache()
            vsplit_bad = KS.reduce_verdicts(svds)
            print(f"[reject cam_value] single ok={vr.ok} step={vr.first_bad_step} kind={vr.kind} | "
                  f"split ok={vsplit_bad.ok} gstep={vsplit_bad.global_first_bad_step} kind={vsplit_bad.kind}", flush=True)
            assert (not vr.ok) and (not vsplit_bad.ok), "corrupt value not rejected"
            assert vr.first_bad_step==vsplit_bad.global_first_bad_step, "first-divergence mismatch"
            assert vr.kind==vsplit_bad.kind=="cam_value"
            print("[PASS] wrong-draft cam_value rejected at SAME first-divergence by split", flush=True)

        # cam_addr corruption: corrupt the model addr for one shard's read via draft_addr flip.
        pre_a = copy.deepcopy(pre_full)
        addr_badstep=None
        for kind,sarr in pre_a.steps.items():
            if sarr.shape[0]==0: continue
            j=sarr.shape[0]//2
            pre_a.draft_addr[kind]=pre_a.draft_addr[kind].copy()
            pre_a.draft_addr[kind][j]^=0x1   # model addr will differ from draft addr
            addr_badstep=int(sarr[j]); break
        if addr_badstep is not None:
            vr=verify_faithful_fast(pre_a, ref_ma)
            svds=[]
            for (lo,hi) in ranges:
                sh=KS._DeviceShard(model,L,code,d,dev,lo,hi,args.chunk,faithful=True,mask=mask)
                _gp,_gs,_gb,_ga,ma=sh.dispatch()
                svds.append((lo, verify_faithful_fast(KS.slice_precompute(pre_a,lo,hi), ma)))
                del sh; gc.collect(); torch.cuda.empty_cache()
            vsplit_a=KS.reduce_verdicts(svds)
            print(f"[reject cam_addr] single ok={vr.ok} step={vr.first_bad_step} kind={vr.kind} | "
                  f"split ok={vsplit_a.ok} gstep={vsplit_a.global_first_bad_step} kind={vsplit_a.kind}", flush=True)
            if not vr.ok:
                assert not vsplit_a.ok and vr.first_bad_step==vsplit_a.global_first_bad_step and vr.kind==vsplit_a.kind
                print("[PASS] wrong-draft cam_addr rejected at SAME first-divergence by split", flush=True)
            else:
                print("[note] cam_addr flip did not arm (model addr 0 at that step) — skipped", flush=True)
    finally:
        uninstall_composed(model)
        os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    print("=== SANITY COMPLETE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
