"""GENUINE-COMPUTATION PROOF for C4_FAITHFUL_SINGLE_DISPATCH.

Scenario E (audit): a self-consistent WRONG draft (LI read pointed at a planted
wrong store, frames declared consistent with the wrong value).  The plain fast
single-dispatch ACCEPTS it (rubber-stamp, injects 777).  The faithful single-dispatch
must REJECT it at the LI step (the model's genuine query decodes address 200 -> the
true value 66 != the draft's injected 777).

Compares three paths on the SAME wrong draft, all through the single-dispatch:
  * plain fast single-dispatch (draft-trusted)         -> ACCEPTS
  * faithful single-dispatch (this lever)              -> REJECTS at LI
And the CLEAN draft accepts on both (no false positive).
"""
from __future__ import annotations
import os, copy

os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.tight_attn_compose import install_composed, uninstall_composed

CODE = isa.assemble([("IMM", 200), ("PSH", 0), ("IMM", 66), ("SI", 0),
                     ("IMM", 200), ("LI", 0), ("HALT", 0)])
LI_STEP = 5

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]

def _guard():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                a = int(ln.split()[1]) / 1e6
                if a < 25.0:
                    raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB<25 -> STOP")
                return a

def clean_draft():
    return draft_pf_program(CODE, max_steps=2000, mask=0xFFFFFFFF)

def wrong_draft(true_val=66, wrong_val=777, wrong_addr=201):
    d = copy.deepcopy(clean_draft())
    d.store_log[5] = (wrong_addr, wrong_val & 0xFFFFFFFF)
    d.read_log[6] = [("mem", wrong_addr)]
    for s in range(LI_STEP, d.step_count):
        d.frames[s]["ax"] = wrong_val & 0xFFFFFFFF
    d.final_ax_masked = wrong_val & 0xFFFFFFFF
    return d

def _flags(faithful_sd):
    for f in COMPOSED:
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ.pop("C4_FAITHFUL_ATTN_EVICT", None)
    if faithful_sd:
        os.environ["C4_FAITHFUL_SINGLE_DISPATCH"] = "1"
    else:
        os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)

def run(model, L, draft, device, faithful_sd):
    _flags(faithful_sd)
    install_composed(model, verbose=False)
    stats = {}
    try:
        vr = verify_blocks(model, L, CODE, draft, block_steps=64, device=device,
                           evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                           evict_interval_steps=8, exact_evict=None)
    finally:
        uninstall_composed(model)
        os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
    return vr, stats

def li_accepted(vr):
    return vr.accepted_steps > LI_STEP

def main():
    device = os.environ.get("C4_DEVICE", "cuda:0")
    _guard()
    print(f"[build] lean streaming model on {device} ...", flush=True)
    model, L, _ = build_lib_model_streaming(code_size=32)
    model = model.to(device)
    _guard()
    print("[build] done.\n", flush=True)

    PATHS = [("fast-SD (trusted)", False), ("faithful-SD", True)]

    print("=== CLEAN draft (correct execution) ===", flush=True)
    cd = clean_draft()
    for name, fsd in PATHS:
        vr, st = run(model, L, cd, device, fsd)
        print(f"  [{name:>18}] all_matched={vr.all_matched} "
              f"accepted={vr.accepted_steps}/{vr.total_steps} "
              f"final_ax={vr.decoded_final_ax}  (LI accepted={li_accepted(vr)})"
              f"{'  chk[a=%d v=%d rt=%d]'%(st.get('faithful_addr_checked',-1),st.get('faithful_value_checked',-1),st.get('faithful_routing_checked',-1)) if fsd else ''}",
              flush=True)

    print("\n=== SCENARIO E: self-consistent WRONG draft (wrong memory read) ===", flush=True)
    wd = wrong_draft(66, 777, 201)
    print("  true mem[200]=66; draft LI resolves to WRONG 777 (@addr 201); frames ax=777",
          flush=True)
    results = {}
    for name, fsd in PATHS:
        vr, st = run(model, L, wd, device, fsd)
        acc = li_accepted(vr)
        verdict = ("ACCEPTED (RUBBER-STAMP)" if acc else "REJECTED (GENUINE)")
        results[name] = acc
        fm = vr.first_mismatch
        print(f"  [{name:>18}] LI accepted={acc} accepted={vr.accepted_steps}/{vr.total_steps}"
              f" -> {verdict}", flush=True)
        if fm:
            print(f"       first_mismatch: {fm}", flush=True)
        if fsd and st.get("faithful_divergence"):
            print(f"       faithful_divergence: {st['faithful_divergence']}", flush=True)

    print("\n=== HEADLINE ===", flush=True)
    ft = results["fast-SD (trusted)"]; fa = results["faithful-SD"]
    print(f"  fast-SD (trusted) accepted the wrong read: {ft}", flush=True)
    print(f"  faithful-SD       accepted the wrong read: {fa}", flush=True)
    if ft and not fa:
        print("  RESULT: faithful SINGLE-DISPATCH CATCHES the self-consistent wrong draft "
              "the fast single-dispatch ACCEPTS.  Genuine computation on the fast path proven.",
              flush=True)
    else:
        print(f"  RESULT: inconclusive (fast={ft}, faithful={fa}).", flush=True)
    _guard()

if __name__ == "__main__":
    main()
