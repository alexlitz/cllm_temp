"""BYTE-EXACT gate for C4_FAITHFUL_SINGLE_DISPATCH on CORRECT execution.

On a correct draft the faithful single-dispatch must (a) ACCEPT every step (verdict ok,
no false positive on routing/address/value), and (b) decode the SAME final AX +
accepted-prefix as the plain fast single-dispatch (draft-trusted) AND the reference
softmax K=1 verify -- L-inf=0.  Runs the DIV-free battery + a deep loop.
"""
import warnings; warnings.filterwarnings('ignore')
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.tight_attn_compose import install_composed, uninstall_composed

COMPOSED = ["C4_DEAD_BLOCK_FUSION","C4_DIRECT_CAM_BATCHED","C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN","C4_BANDED_LOCAL_ATTN","C4_FUSED_MEGABLOCK","C4_DIRECT_CAM_VEC"]

def _programs():
    P = {}
    P['add'] = ([isa.Instr(isa.IMM,300),isa.Instr(isa.PSH),isa.Instr(isa.IMM,44),
                 isa.Instr(isa.ADD),isa.Instr(isa.HALT)], 344)
    P['mul'] = ([isa.Instr(isa.IMM,12),isa.Instr(isa.PSH),isa.Instr(isa.IMM,11),
                 isa.Instr(isa.MUL),isa.Instr(isa.HALT)], 132)
    P['mem'] = ([isa.Instr(isa.ENT,1),isa.Instr(isa.LEA,0),isa.Instr(isa.PSH),
                 isa.Instr(isa.IMM,777),isa.Instr(isa.SI),
                 isa.Instr(isa.LEA,0),isa.Instr(isa.LI),isa.Instr(isa.HALT)], 777)
    func = [isa.Instr(isa.JSR,2),isa.Instr(isa.HALT),isa.Instr(isa.ENT,0),
            isa.Instr(isa.IMM,99),isa.Instr(isa.LEV)]
    P['func_lev'] = (func, 99)
    P['bigimm'] = ([isa.Instr(isa.IMM,70000),isa.Instr(isa.PSH),isa.Instr(isa.IMM,5000),
                    isa.Instr(isa.ADD),isa.Instr(isa.HALT)], 75000)
    P['jmp'] = ([isa.Instr(isa.JMP,3),isa.Instr(isa.IMM,99),isa.Instr(isa.HALT),
                 isa.Instr(isa.IMM,7),isa.Instr(isa.HALT)], 7)
    P['bnz'] = ([isa.Instr(isa.IMM,3),isa.Instr(isa.PSH),isa.Instr(isa.IMM,1),
                 isa.Instr(isa.SUB),isa.Instr(isa.BNZ,1),isa.Instr(isa.HALT)], 0)
    # DEEP loop (proven, from _agent_verify_addr_byteexact): sum 1..N countdown, heavy
    # pop/mem/lev traffic + backward branch; step_count ~ several thousand.
    N = 300
    loop = [
        isa.Instr(isa.ENT,2),
        isa.Instr(isa.LEA,0),isa.Instr(isa.PSH),isa.Instr(isa.IMM,N),isa.Instr(isa.SI),
        isa.Instr(isa.LEA,1),isa.Instr(isa.PSH),isa.Instr(isa.IMM,0),isa.Instr(isa.SI),
        isa.Instr(isa.LEA,0),isa.Instr(isa.LI),
        isa.Instr(isa.BZ,30),
        isa.Instr(isa.LEA,1),isa.Instr(isa.PSH),
        isa.Instr(isa.LEA,1),isa.Instr(isa.LI),isa.Instr(isa.PSH),
        isa.Instr(isa.LEA,0),isa.Instr(isa.LI),
        isa.Instr(isa.ADD),
        isa.Instr(isa.SI),
        isa.Instr(isa.LEA,0),isa.Instr(isa.PSH),
        isa.Instr(isa.LEA,0),isa.Instr(isa.LI),isa.Instr(isa.PSH),
        isa.Instr(isa.IMM,1),isa.Instr(isa.SUB),
        isa.Instr(isa.SI),
        isa.Instr(isa.JMP,9),
        isa.Instr(isa.LEA,1),isa.Instr(isa.LI),
        isa.Instr(isa.HALT),
    ]
    P['loop_sum300'] = (loop, N*(N+1)//2)
    return P

def _flags(fsd):
    for f in COMPOSED: os.environ[f] = "1"
    for f in ("C4_ONCHIP_RESIDUAL","C4_RESIDENT_BATCH","C4_PRECOMPUTED_SCHEDULE",
              "C4_SCHED_FAST_BUILD","C4_SCHED_GPU_BUILD","C4_FFN_FUSED_HIDDEN"):
        os.environ[f] = "1"
    os.environ.pop("C4_FAITHFUL_ATTN_EVICT", None)
    if fsd: os.environ["C4_FAITHFUL_SINGLE_DISPATCH"] = "1"
    else: os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)

def _run(model, L, code_isa, draft, device, fsd):
    _flags(fsd); install_composed(model, verbose=False); st={}
    try:
        vr = verify_blocks(model, L, code_isa, draft, block_steps=512, device=device,
                           evict=True, mask=0xFFFFFFFF, stats=st, fast=True,
                           evict_interval_steps=64, exact_evict=None)
    finally:
        uninstall_composed(model); os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    return vr, st

def main():
    device = "cuda:0"
    model, L, _ = build_lib_model_streaming(code_size=64); model = model.to(device)
    print("[build] done\n", flush=True)
    P = _programs()
    allok = True
    for name, (prog, want_ax) in P.items():
        code_isa = prog if isinstance(prog[0], isa.Instr) else isa.assemble(prog)
        draft = draft_pf_program(code_isa, max_steps=20000, mask=0xFFFFFFFF)
        vf, sf = _run(model, L, code_isa, draft, device, True)     # faithful
        vp, sp = _run(model, L, code_isa, draft, device, False)    # fast (trusted)
        fa = (vf.decoded_final_ax if vf.all_matched else None)
        pa = (vp.decoded_final_ax if vp.all_matched else None)
        same = (vf.accepted_steps == vp.accepted_steps and vf.all_matched == vp.all_matched
                and fa == pa)
        ax_ok = (vf.all_matched and (fa & 0xFFFF) == (want_ax & 0xFFFF))
        ok = same and ax_ok and sf.get("faithful_ok", None) is True
        allok = allok and ok
        print(f"  [{name:>12}] faithful acc={vf.accepted_steps}/{vf.total_steps} "
              f"ax={fa} | fast acc={vp.accepted_steps}/{vp.total_steps} ax={pa} "
              f"| want={want_ax} | verify_ok={sf.get('faithful_ok')} "
              f"chk[a={sf.get('faithful_addr_checked')} v={sf.get('faithful_value_checked')} "
              f"rt={sf.get('faithful_routing_checked')}] -> {'OK' if ok else 'FAIL'}",
              flush=True)
    print(f"\n=== BYTE-EXACT battery: {'ALL OK (L-inf=0, no false positive)' if allok else 'FAIL'} ===",
          flush=True)

if __name__ == "__main__":
    main()
