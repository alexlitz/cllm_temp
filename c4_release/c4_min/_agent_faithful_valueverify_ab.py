"""ISOLATED A/B of the VALUE-VERIFY cost (CPU-only, contention-immune) at real-doom scale.

Measures, on the SAME real-doom draft + plan + (simulated correct) model addresses, the
per-frame wall of:
  OLD  : build_faithful_plan + verify_faithful           (the whole value re-resolution runs
                                                           on the CRITICAL PATH, post-dispatch)
  NEW-crit : verify_faithful_fast(precompute)             (only the vectorized compare — what
                                                           STAYS on the critical path)
  NEW-build: build_faithful_precompute                    (the heavy re-resolution — PIPELINED
                                                           onto the background build thread)

This isolates the fix's effect on the CRITICAL PATH independent of GPU contention: the win is
(OLD critical-path value-verify) - (NEW critical-path compare).  Byte-identical verdict.

Run: CUDA_VISIBLE_DEVICES="" python -m c4_min._agent_faithful_valueverify_ab --draft-steps 40000
"""
from __future__ import annotations
import argparse, os, sys, time

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

DOOM = "/home/alexlitz/Documents/misc/c4_doom"
sys.path.insert(0, DOOM)

import numpy as np

from c4_min import isa
from c4_min import nibble_filesys as FS
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0x10000
from c4_min.pf_speculative import draft_pf_program
from c4_min.faithful_single_dispatch import (
    verify_faithful, build_faithful_precompute, verify_faithful_fast, _read_frame_of_step)
from run_c4_min import (data_segment, tag_compiler_syscalls,
                        install_compiler_abi_file_dispatcher)

RENDER_REDUCED_FRAME = 358_058


def _apply_pow2(ops, imms):
    IMM_OP, DIV_OP, MOD_OP, SHR_OP, AND_OP = 1, 28, 29, 24, 16
    n = len(ops)
    for i in range(n - 1):
        o0, m0, o1 = int(ops[i]), int(imms[i]), int(ops[i + 1])
        if o0 == IMM_OP and m0 > 0 and (m0 & (m0 - 1)) == 0:
            if o1 == DIV_OP: imms[i] = m0.bit_length() - 1; ops[i + 1] = SHR_OP
            elif o1 == MOD_OP: imms[i] = m0 - 1; ops[i + 1] = AND_OP


def _load_snapshot():
    snap = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "_doom_bytecode_snapshot.npz"))
    return list(snap["ops"]), list(snap["imms"]), snap["data"]


def build_doom_draft(draft_steps):
    ops, imms, data = _load_snapshot(); n = len(ops)
    _apply_pow2(ops, imms)
    install_compiler_abi_file_dispatcher()
    code_isa = tag_compiler_syscalls(
        [isa.Instr(int(ops[i]), int(imms[i]) & 0xFFFFFFFF) for i in range(n)], isa)
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                              stdin=FS.InputKVStream(b"q", neural=True)))
    d = draft_pf_program(code_isa, max_steps=draft_steps, mask=0xFFFFFFFF,
                         data_seg=data_segment([int(b) for b in data]), fio=fio)
    return d, code_isa


class _StubModel:
    """cam_head_map only reads block.attn.cam_heads; the doom live blocks are known
    (0=ingest,2=code,7=mem,11=pop).  We build model_addrs directly from the draft's OWN
    resolved reads (== the correct model, so verify accepts) — no GPU model needed for the
    CPU value-verify A/B."""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-steps", type=int, default=40000)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args(argv)
    print(f"[doom] drafting real doom ({args.draft_steps} steps) ...", flush=True)
    t0 = time.time()
    d, code = build_doom_draft(args.draft_steps)
    n = d.step_count
    print(f"[doom] drafted {n} DIV-free steps in {time.time()-t0:.0f}s", flush=True)
    mask = 0xFFFFFFFF
    ws = np.asarray(d.win_starts[:n], dtype=np.int64)

    # build the plan ONCE (fixed per draft).
    plan = build_faithful_plan_stub(d, code)

    # simulate the CORRECT model addresses: the model queried EXACTLY the draft's resolved
    # address at every read (so the verify ACCEPTS — the byte-exact correct-execution case,
    # which is what the fps run measures).  mem head=7, pop head=11 (the doom live CAM heads);
    # code head=2.
    model_addrs = {}
    # mem/pop/lev/uni: model_addr == draft addr at each read step.
    for (bidx, kind) in ((7, "mem"), (11, "pop")):
        arr = np.zeros(n, dtype=np.int64)
        addr_d = plan.draft_addr.get(kind, {})
        for s in range(n):
            a = addr_d.get(int(ws[s]))
            if a is not None:
                arr[s] = int(a) & 0xFFFFFFFF
        model_addrs[(bidx, kind)] = arr
    # code: model pc == draft pc.
    carr = np.zeros(n, dtype=np.int64)
    for s in range(n):
        pc = plan.code_addr.get(int(ws[s]))
        if pc is not None:
            carr[s] = int(pc)
    model_addrs[(2, "code")] = carr

    # ---- OLD path: verify_faithful (whole value re-resolution on the critical path) ----
    old = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        vo = verify_faithful(d, plan, model_addrs, ws, mask=mask)
        old.append(time.perf_counter() - t0)
    old_s = float(np.median(old))

    # ---- NEW: split precompute (build thread) vs the critical-path compare ----
    pre_t = []; crit = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        pre = build_faithful_precompute(d, plan, ws, n, mask=mask)
        pre_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        vn = verify_faithful_fast(pre, model_addrs)
        crit.append(time.perf_counter() - t0)
    pre_s = float(np.median(pre_t)); crit_s = float(np.median(crit))

    assert (vo.ok == vn.ok and vo.first_bad_step == vn.first_bad_step
            and vo.kind == vn.kind), f"verdict mismatch {vo} vs {vn}"
    assert vo.n_addr_checked == vn.n_addr_checked
    assert vo.n_value_checked == vn.n_value_checked
    assert vo.n_routing_checked == vn.n_routing_checked

    print("\n=== VALUE-VERIFY A/B (CPU-only, contention-immune) real-doom @ "
          f"{n} steps ===", flush=True)
    print(f"  verdict: ok={vn.ok}  addr_chk={vn.n_addr_checked}  val_chk={vn.n_value_checked}"
          f"  rt_chk={vn.n_routing_checked}  (OLD verdict IDENTICAL)", flush=True)
    print(f"  OLD  critical-path value-verify (verify_faithful):     {old_s*1e3:8.3f} ms/frame "
          f"= {old_s/n*1e6:7.4f} us/step", flush=True)
    print(f"  NEW  critical-path compare      (verify_faithful_fast): {crit_s*1e3:8.3f} ms/frame "
          f"= {crit_s/n*1e6:7.4f} us/step", flush=True)
    print(f"  NEW  PIPELINED precompute       (build_faithful_precompute, on build thread): "
          f"{pre_s*1e3:8.3f} ms/frame = {pre_s/n*1e6:7.4f} us/step", flush=True)
    removed = (old_s - crit_s) / n * 1e6
    print(f"\n  --> REMOVED from the critical path: {removed:.4f} us/step "
          f"(OLD {old_s/n*1e6:.4f} -> NEW {crit_s/n*1e6:.4f} us/step, "
          f"{old_s/max(crit_s,1e-9):.0f}x)", flush=True)
    print(f"      (the {pre_s/n*1e6:.4f} us/step precompute is HIDDEN under the next frame's "
          f"build+dispatch on the pipeline thread)", flush=True)
    # projection at the doc's uncontended fast dispatch (1.88 us/step) + in-graph addr (+0.73).
    fast_disp = 1.88; addr_add = 0.73
    faith_crit_old = fast_disp + addr_add + old_s / n * 1e6
    faith_crit_new = fast_disp + addr_add + crit_s / n * 1e6
    print(f"\n  PROJECTION at the doc's UNCONTENDED fast dispatch 1.88 us/step + in-graph addr "
          f"+0.73 (build overlapped):", flush=True)
    print(f"    faithful critical path OLD: {faith_crit_old:.3f} us/step -> "
          f"{1e6/(faith_crit_old*RENDER_REDUCED_FRAME):.3f} fps @ {RENDER_REDUCED_FRAME} steps",
          flush=True)
    print(f"    faithful critical path NEW: {faith_crit_new:.3f} us/step -> "
          f"{1e6/(faith_crit_new*RENDER_REDUCED_FRAME):.3f} fps @ {RENDER_REDUCED_FRAME} steps",
          flush=True)


def build_faithful_plan_stub(draft, code):
    """build_faithful_plan needs a model only for cam_head_map (block->head kinds), which the
    CPU A/B doesn't use (we supply model_addrs directly).  Replicate the store-array + draft
    dict extraction without a model."""
    from c4_min.direct_cam_batched import build_resolved_table
    from c4_min.faithful_single_dispatch import FaithfulPlan
    tbl = build_resolved_table(draft, code)
    sl = draft.store_log or {}
    sf = sorted(sl)
    store_frames = np.asarray(sf, dtype=np.int64)
    store_addr = np.fromiter((sl[f][0] & 0xFFFFFFFF for f in sf), dtype=np.int64, count=len(sf))
    store_val = np.fromiter((sl[f][1] & 0xFFFFFFFF for f in sf), dtype=np.int64, count=len(sf))
    return FaithfulPlan(
        heads_by_block={},
        draft_addr={k: dict(tbl.addr[k]) for k in ("mem", "pop", "lev", "uni")},
        draft_val={k: dict(getattr(tbl, k)) for k in ("mem", "pop", "lev", "uni")},
        code_addr=dict(tbl.code_addr),
        store_frames=store_frames, store_addr=store_addr, store_val=store_val)


if __name__ == "__main__":
    raise SystemExit(main())
