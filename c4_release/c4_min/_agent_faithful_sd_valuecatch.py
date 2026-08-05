"""VALUE-CATCH: a self-consistent wrong-VALUE draft that PASSES the address check but is
caught ONLY by the genuine value re-resolution (the C4_FAITHFUL_ATTN_EVICT read).

Program stores 66 then 999 to addr 200, then LI@200 (true answer = latest-write = 999).
The wrong draft keeps read_log@200 (CORRECT address -> address check passes) but declares
frames ax = 66 (the STALE value) AND injects the stale 66 as the resolved value (by
pointing the read at a decoy that returns 66 while the true committed latest is 999).

We force the injected resolve to the stale 66 by pointing read_log at a decoy address
whose only store is 66, but we set the model's genuine query to 200 (via the token frame,
which the LI instruction's IMM 200 drives) -- so:
  * address check: model queries 200, draft resolved decoy-addr -> MISMATCH catches it.
That again reduces to address.  The pure value case (address agrees, value disagrees)
requires the draft's resolve to use a STALE store to the SAME address 200 -- i.e. the
draft dropped/evicted the newer 999 store from its resolution.  We simulate that by
deleting the 999 store from the draft's store_log (so latest-write-wins(draft)=66) while
the frames declare 66, but the GENUINE re-resolution uses the FULL committed store set
including 999.  To give the genuine re-resolution the 999, we keep 999 in a SEPARATE
'committed' view.  In practice the plan's store arrays come from draft.store_log, so we
instead show the mechanism directly: plan built from the FULL store set (with 999) vs a
draft whose read resolves to 66.
"""
from __future__ import annotations
import os, copy
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program
from c4_min.faithful_single_dispatch import (build_faithful_plan, verify_faithful,
                                             _read_frame_of_step, _genuine_value_at)

# store 66@200, store 999@200, then LI@200 -> answer 999 (latest-write-wins).
CODE = isa.assemble([("IMM", 200), ("PSH", 0), ("IMM", 66), ("SI", 0),
                     ("IMM", 200), ("PSH", 0), ("IMM", 999), ("SI", 0),
                     ("IMM", 200), ("LI", 0), ("HALT", 0)])

def main():
    device = "cuda:0"
    model, L, _ = build_lib_model_streaming(code_size=32); model = model.to(device)
    d = draft_pf_program(CODE, max_steps=2000, mask=0xFFFFFFFF)
    # find the LI step (op == 'LI') and its true answer.
    li = next(s for s in range(d.step_count) if d.frames[s].get("op") == "LI")
    print(f"[setup] LI step={li} true ax={d.frames[li]['ax']} (latest-write 999)", flush=True)
    assert d.frames[li]["ax"] == 999, "expected latest-write 999"

    # GENUINE plan is built from the FULL committed store_log (contains both 66 and 999).
    plan = build_faithful_plan(model, L, CODE, d)

    # Build a WRONG draft: same read (addr 200) but the draft dropped the 999 store from
    # its own resolution and declares the STALE 66.  We corrupt read_log to a stale value
    # by editing the draft's frames + resolved value to 66 while the PLAN (committed) keeps
    # 999.  We feed verify_faithful the model_addr = 200 (the genuine query) and the
    # draft's declared value dict = 66.
    wd = copy.deepcopy(d)
    for s in range(li, wd.step_count):
        wd.frames[s]["ax"] = 66
    # the draft's declared per-read value at the LI query row (what the fast decode trusts):
    ws = np.asarray(d.win_starts[:d.step_count], dtype=np.int64)
    qpos = int(ws[li])
    # override the plan's draft_val at that read to the STALE 66 (self-consistent wrong).
    plan.draft_val["mem"][qpos] = 66            # draft injected value (wrong/stale)
    plan.draft_addr["mem"][qpos] = 200          # draft address (CORRECT -> addr passes)

    # the model genuinely queries 200 at the LI step (address check will pass 200==200).
    n = d.step_count
    model_addrs = {(7, "mem"): np.zeros(n, dtype=np.int64)}
    model_addrs[(7, "mem")][li] = 200           # the model's decoded query addr = 200

    verdict = verify_faithful(d, plan, model_addrs, ws, mask=0xFFFFFFFF)
    print(f"[verify] ok={verdict.ok} first_bad_step={verdict.first_bad_step} "
          f"kind={verdict.kind} detail={verdict.detail}", flush=True)
    rf = _read_frame_of_step(d, n)
    gv = _genuine_value_at(np.array([200]), np.array([rf[li]]), plan)
    print(f"[genuine] latest-write-wins over committed stores at model-addr 200 = "
          f"{int(gv[0])} (== 999, the true answer)", flush=True)
    if (not verdict.ok) and verdict.kind == "cam_value" and verdict.first_bad_step == li:
        print("\nRESULT: the address check PASSED (model 200 == draft 200) but the GENUINE "
              "VALUE re-resolution (999) != the draft's injected stale 66 -> CAUGHT by the "
              "cam_value layer.  The C4_FAITHFUL_ATTN_EVICT value read on the single "
              "dispatch catches a self-consistent wrong VALUE the address check cannot.",
              flush=True)
    else:
        print("\nRESULT: value layer did NOT fire as expected -- inspect.", flush=True)

if __name__ == "__main__":
    main()
