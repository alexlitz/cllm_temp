"""HEADLINE: does the faithful-attn-evict path CATCH the self-consistent wrong
draft that direct-CAM ACCEPTS?  (genuine-computation proof)

The wedge (audit Q2): direct-CAM trusts ``resolve_load_rows(draft)`` for the
memory-read ADDRESS->row->value resolution; the softmax path uses the model's OWN
query.  So we corrupt the draft's READ RESOLUTION self-consistently:

  * keep the TRUE store in the KV (mem[200]=66 written at the real SI step -> the
    overlay writes ADDR_BIN=200, VAL_NIB=66 on that KV row, unchanged),
  * plant a WRONG store row (addr=201, val=777) BEFORE the read,
  * point the draft's read_log LI read at addr 201 (so resolve_load_rows returns
    777) AND set the LI-step (and downstream) frame ax = 777.

The LI TOKEN FRAME still carries address 200 (the model's genuine query), and the
true KV row (addr=200,val=66) is present and scored.  So:
  * direct-CAM injects 777 at the LI query row -> matches the wrong frame -> ACCEPTS
    the wrong read (accepts step 5, the memory read).
  * faithful-attn-evict issues the model's real query for 200 -> retrieves 66 ->
    66 != 777 -> CATCHES at the LI step.

We verify a draft TRUNCATED at the LI step so accepting the wrong read == accepting
a wrong final answer (removes the HALT-carry noise; the LI-loaded value IS the
answer).  We report accepted_steps + all_matched on both paths.

Memory-safe: lean streaming build, MemAvailable guard, GPU 0.
"""
from __future__ import annotations

import os
import copy

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


def _V_FRAME_LEN():
    from c4_min import blogspec_vocab as V
    return V.FRAME_LEN


def _guard(where=""):
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                a = int(ln.split()[1]) / 1e6
                if a < 25.0:
                    raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB<25 @ {where}")
                return a
    return 1e9


# store 66 at addr 200, then LOAD it back: AX = mem[200] = 66 is the answer.
CODE = isa.assemble([("IMM", 200), ("PSH", 0), ("IMM", 66), ("SI", 0),
                     ("IMM", 200), ("LI", 0), ("HALT", 0)])
LI_STEP = 5  # the memory read step (0-indexed): its AX = loaded value


def clean_draft():
    return draft_pf_program(CODE, max_steps=2000, mask=0xFFFFFFFF)


def wrong_draft(true_val=66, wrong_val=777, wrong_addr=201):
    """Self-consistent wrong READ RESOLUTION (see module docstring)."""
    d = copy.deepcopy(clean_draft())
    # plant a wrong store row at a free frame index < the read frame (6).
    d.store_log[5] = (wrong_addr, wrong_val & 0xFFFFFFFF)
    # point the LI read (read_frame 6) at the wrong address.
    d.read_log[6] = [("mem", wrong_addr)]
    # make the declared frames self-consistent with the WRONG resolution.
    for s in range(LI_STEP, d.step_count):
        d.frames[s]["ax"] = wrong_val & 0xFFFFFFFF
    d.final_ax_masked = wrong_val & 0xFFFFFFFF
    return d, true_val, wrong_val


def _set_flags(direct_cam, faithful):
    for f in ("C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM", "C4_FAITHFUL_ATTN_EVICT"):
        os.environ.pop(f, None)
    if direct_cam:
        os.environ["C4_DIRECT_CAM_BATCHED"] = "1"
        os.environ["C4_DIRECT_LOCAL_CAM"] = "1"
    if faithful:
        os.environ["C4_FAITHFUL_ATTN_EVICT"] = "1"


def run(model, L, draft, device, *, direct_cam, faithful):
    _set_flags(direct_cam, faithful)
    install_composed(model, verbose=False)
    stats = {}
    try:
        vr = verify_blocks(model, L, CODE, draft, block_steps=64, device=device,
                           evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                           evict_interval_steps=8, exact_evict=None)
    finally:
        uninstall_composed(model)
        for f in ("C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
                  "C4_FAITHFUL_ATTN_EVICT"):
            os.environ.pop(f, None)
    return vr, stats


def _li_accepted(vr):
    """Did the path ACCEPT the LI memory-read step (step 5)?  accepted_steps counts
    the causal-prefix of accepted steps, so LI is accepted iff accepted_steps > 5."""
    return vr.accepted_steps > LI_STEP


def main():
    device = os.environ.get("C4_DEVICE", "cuda:0")
    _guard("start")
    print(f"[build] lean streaming model on {device} ...", flush=True)
    model, L, _ = build_lib_model_streaming(code_size=32)
    model = model.to(device)
    _guard("post-build")
    print("[build] done.\n", flush=True)

    PATHS = [("direct-CAM", dict(direct_cam=True, faithful=False)),
             ("faithful-evict", dict(direct_cam=False, faithful=True))]

    # --- CLEAN: both paths accept the whole trace, answer 66 -------------------
    print("=== CLEAN draft (correct execution) ===", flush=True)
    cd = clean_draft()
    for name, fl in PATHS:
        vr, st = run(model, L, cd, device, **fl)
        print(f"  [{name:>15}] all_matched={vr.all_matched} "
              f"accepted={vr.accepted_steps}/{vr.total_steps} "
              f"final_ax={vr.decoded_final_ax}  (LI read accepted="
              f"{_li_accepted(vr)})", flush=True)

    # --- SCENARIO E: self-consistent wrong READ RESOLUTION --------------------
    print("\n=== SCENARIO E: self-consistent WRONG draft (wrong memory read) ===",
          flush=True)
    wd, tval, wval = wrong_draft(66, 777, 201)
    print(f"  true mem[200]={tval}; draft's LI read resolves to WRONG value {wval} "
          f"(planted @addr=201), declared frame ax={wval}", flush=True)
    results = {}
    for name, fl in PATHS:
        vr, st = run(model, L, wd, device, **fl)
        li_acc = _li_accepted(vr)
        # the headline verdict: did this path ACCEPT the wrong memory read?
        verdict = ("ACCEPTED the wrong read (RUBBER-STAMP)" if li_acc
                   else "CAUGHT/REJECTED the wrong read (GENUINE)")
        results[name] = (li_acc, vr)
        fm = vr.first_mismatch
        print(f"  [{name:>15}] LI-read accepted={li_acc} "
              f"accepted={vr.accepted_steps}/{vr.total_steps} -> {verdict}",
              flush=True)
        if fm:
            print(f"        first_mismatch: step={fm['step']} "
                  f"got_ax={fm['got']['ax']} want_ax={fm['want']['ax']} "
                  f"(model recomputed {fm['got']['ax']}, draft claimed "
                  f"{fm['want']['ax']})", flush=True)

    # --- headline summary -----------------------------------------------------
    dc_acc = results["direct-CAM"][0]
    fe_acc = results["faithful-evict"][0]
    print("\n=== HEADLINE ===", flush=True)
    print(f"  direct-CAM     accepted the wrong read: {dc_acc}", flush=True)
    print(f"  faithful-evict accepted the wrong read: {fe_acc}", flush=True)
    if dc_acc and not fe_acc:
        print("  RESULT: faithful-attn-evict CATCHES the self-consistent wrong draft "
              "that direct-CAM ACCEPTS.  Genuine computation proven.", flush=True)
    else:
        print(f"  RESULT: inconclusive (dc={dc_acc}, fe={fe_acc}); inspect above.",
              flush=True)
    _guard("end")


if __name__ == "__main__":
    main()
