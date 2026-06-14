#!/usr/bin/env python3
"""Dump the FULL residual (all named built dims) at the spurious STEP_END emit
row vs a genuine SP-byte0 row, input to block 31 (after block 30), so we can find
a NOT-blocker dim present on the spurious row but absent on the genuine row."""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
_HERE=os.path.dirname(os.path.abspath(__file__)); _PKG=os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0,_PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
def _d(t):
    try:
        if t.layout!=torch.strided: return t.to_dense()
    except Exception: pass
    return t
BLK_IN=30
def main():
    pid=int(sys.argv[1]) if len(sys.argv)>1 else 550
    ms=int(sys.argv[2]) if len(sys.argv)>2 else 12
    src,exp,desc=generate_test_programs()[pid]; bc=compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe=build_groundtruth_probe()
        _,layout=compile_full_vm_dynamic(disk_cache=True, alu_mode='efficient')
    inv={}
    for k,v in layout.dim_positions.items(): inv.setdefault(int(v),[]).append(k)
    model=probe.model; device=probe._device; SE=int(Token.STEP_END); REG_SP=int(Token.REG_SP)
    ctx=probe._final_context(bc,max_steps=ms); pl=len(probe._build_context(bc))
    spur=[i-1 for i in range(pl,len(ctx)) if ctx[i]==255 and ctx[i-1]==SE]
    spb0=[i+1 for i in range(pl,len(ctx)-1) if ctx[i]==REG_SP]
    padded=torch.tensor([ctx],dtype=torch.long,device=device)
    resid=_d(model.forward(padded,stop_after_block=BLK_IN))[0]
    def named(row):
        r=resid[row]; out={}
        for d in range(r.shape[0]):
            v=float(r[d])
            if abs(v)>0.05:
                nm=inv.get(d,[f"dim{d}"]); out[(d,nm[0] if nm else f"dim{d}")]=round(v,2)
        return out
    sr=spur[0]; gr=spb0[0] if spb0 else None
    A=named(sr); B=named(gr) if gr is not None else {}
    print(f"spurious row {sr} (tok {ctx[sr]}) named-residual nonzero:")
    for k,v in sorted(A.items()): print("   ",k,v)
    print(f"\ngenuine SP-byte0 row {gr} (tok {ctx[gr]}) named-residual nonzero:")
    for k,v in sorted(B.items()): print("   ",k,v)
    print("\n=== dims present on SPURIOUS (|v|>0.5) but ~0 on GENUINE (|v|<0.05) ===")
    for (d,nm),v in sorted(A.items()):
        if abs(v)>0.5:
            gv=[vv for (dd,_),vv in B.items() if dd==d]
            if not gv or abs(gv[0])<0.05:
                print(f"   dim {d} {nm}: spur={v}  gen={gv[0] if gv else 0.0}")
if __name__=="__main__": main()
