#!/usr/bin/env python3
"""_agent_onchip_floor.py — C4_ONCHIP_RESIDUAL + C4_RESIDENT_BATCH byte-exact gate +
device/wall measurement vs the baseline single-dispatch (C4_PRECOMPUTED_SCHEDULE).

TASK 1 (C4_ONCHIP_RESIDUAL): kill the dense W_o GEMMs / cam_out transposes -> residual
on-chip.  TASK 2 (C4_RESIDENT_BATCH): no per-chunk static-input copy (single-chunk).
TASK 3 (measure): device+wall us/step, steps/s, sec/frame, x-from-realtime, x-above-floor.
TASK 4 (byte-exact): per-step AX/PC/SP/BP == K=1 reference AND == baseline single-dispatch.

Run: CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> \
     python -m c4_min._agent_onchip_floor --device cuda:0
"""
from __future__ import annotations
import argparse, os, time, traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested, build_loop_countdown
from c4_min.bench_composed_fast_path import _battery
from c4_min import precomputed_schedule as PS

DOOM_FRAME_INSTRS = 6_890_000
FLOP_FLOOR_US = 0.0071
RENDER_REDUCED_FRAME = 358_058

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]
LEVERS = ["C4_ONCHIP_RESIDUAL", "C4_RESIDENT_BATCH"]


def _composed_on():
    for f in COMPOSED:
        os.environ[f] = "1"


def _all_off():
    for f in COMPOSED + LEVERS + ["C4_PRECOMPUTED_SCHEDULE", "C4_SCHED_CHUNK"]:
        os.environ.pop(f, None)


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP")
    return a


def _run(model, L, code, draft, device, chunk, onchip, resident):
    """Run one config; returns PrecomputedResult."""
    _composed_on()
    if onchip:
        os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    else:
        os.environ.pop("C4_ONCHIP_RESIDUAL", None)
    if resident:
        os.environ["C4_RESIDENT_BATCH"] = "1"
    else:
        os.environ.pop("C4_RESIDENT_BATCH", None)
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    r = PS.run_verify(model, L, code, draft, device, mask=0xFFFFFFFF)
    os.environ.pop("C4_ONCHIP_RESIDUAL", None)
    os.environ.pop("C4_RESIDENT_BATCH", None)
    return r


def byte_exact(model, L, device):
    print("\n=== TASK 4: BYTE-EXACT (C4_ONCHIP_RESIDUAL [+RESIDENT_BATCH]) ===", flush=True)
    progs = []
    for name, prog, seed in _battery():
        if seed:
            continue
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        d = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
        if not d.halted:
            continue
        if any(d.frames[s].get("op") in ("DIV", "MOD") for s in range(d.step_count)):
            continue
        progs.append((name, code))
    progs.append(("loop_countdown60", build_loop_countdown(60)[0]))
    progs.append(("nested_6_16", build_nested(6, 16)[0]))
    progs.append(("nested_10_24", build_nested(10, 24)[0]))
    progs.append(("nested_12_28", build_nested(12, 28)[0]))   # 5455-step deep loop
    all_ok = True
    install_composed(model, verbose=False)
    print(f"{'prog':>18} {'steps':>7} {'base(acc,fin,m)':>20} {'onchip':>20} "
          f"{'oc+res':>20} {'Linf':>5} {'ok':>4}", flush=True)
    try:
        for name, code in progs:
            draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
            if not draft.halted:
                continue
            # BASELINE single-dispatch (dense W_o), chunk big enough to hold in one chunk.
            chunk = max(4096, draft.step_count)
            base = _run(model, L, code, draft, device, chunk, onchip=False, resident=False)
            oc = _run(model, L, code, draft, device, chunk, onchip=True, resident=False)
            ocr = _run(model, L, code, draft, device, chunk, onchip=True, resident=True)
            b = (base.accepted_steps, base.decoded_final_ax, base.all_matched)
            o = (oc.accepted_steps, oc.decoded_final_ax, oc.all_matched)
            r = (ocr.accepted_steps, ocr.decoded_final_ax, ocr.all_matched)
            # byte-exact to the K=1 reference draft (all steps accepted, final ax matches).
            def _draft_exact(res):
                return (res.all_matched and res.accepted_steps == draft.step_count
                        and res.decoded_final_ax == draft.final_ax_masked)
            de_o = _draft_exact(oc); de_r = _draft_exact(ocr)
            agree = (b == o == r)
            ok = de_o and de_r and agree and _draft_exact(base)
            all_ok = all_ok and ok
            linf = 0 if ok else 1
            print(f"{name:>18} {draft.step_count:>7} {str(b):>20} {str(o):>20} "
                  f"{str(r):>20} {linf:>5} {'OK' if ok else 'FAIL':>4}", flush=True)
            if not ok:
                print(f"    base={base.first_mismatch}", flush=True)
                print(f"    onchip={oc.first_mismatch}", flush=True)
                print(f"    oc+res={ocr.first_mismatch}", flush=True)
            _guard()
    finally:
        uninstall_composed(model)
    print(f"\n  -> {'ALL BYTE-EXACT (Linf=0)' if all_ok else 'DIVERGENCE FOUND'}", flush=True)
    return all_ok


def _wall(fn, iters=3):
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _device_us(model, L, code, draft, device, steps, chunk, onchip, resident):
    from torch.profiler import profile, ProfilerActivity
    _run(model, L, code, draft, device, chunk, onchip, resident)  # warmup
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _run(model, L, code, draft, device, chunk, onchip, resident)
        torch.cuda.synchronize(device)
    tot = sum(e.self_device_time_total for e in prof.key_averages()) or 1.0
    return tot / steps


def measure(model, L, device):
    print("\n=== TASK 3: WALL+DEVICE — baseline vs onchip vs onchip+resident ===", flush=True)
    name, code = "nested_12_28", build_nested(12, 28)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    steps = draft.step_count
    print(f"  program '{name}': {steps} DIV-free steps (deep nested loop)", flush=True)
    install_composed(model, verbose=False)
    # single chunk sized to hold the whole batch (resident path needs chunk >= steps).
    chunk = 8192
    configs = [("baseline(dense W_o)", False, False),
               ("onchip", True, False),
               ("onchip+resident", True, True)]
    rows = []
    try:
        for label, oc, res in configs:
            _guard()
            wall = _wall(lambda: _run(model, L, code, draft, device, chunk, oc, res))
            dev_us = _device_us(model, L, code, draft, device, steps, chunk, oc, res)
            us = wall * 1e6 / steps
            sps = steps / wall
            r = _run(model, L, code, draft, device, chunk, oc, res)
            print(f"  {label:>22}: wall {us:8.2f} us/step  device {dev_us:8.2f} us/step  "
                  f"{sps:10.0f} steps/s  matched={r.all_matched}", flush=True)
            rows.append((label, us, dev_us, sps, r.all_matched))
    finally:
        uninstall_composed(model)
    print("\n  --- SUMMARY (deep nested loop, chunk=%d) ---" % chunk, flush=True)
    base = rows[0]
    best = min(rows, key=lambda r: r[1])
    for label, us, dev_us, sps, m in rows:
        frame_red = RENDER_REDUCED_FRAME / sps
        print(f"  {label:>22}: {us:8.2f} us/step wall  {dev_us:8.2f} us/step device  "
              f"{sps:9.0f} sps  sec/frame(358K)={frame_red:8.2f}  "
              f"x-floor={us/FLOP_FLOOR_US:8.0f}  x-from-35fps={(1.0/35.0)/frame_red:.4f}",
              flush=True)
    print(f"\n  BEST: {best[0]} @ {best[1]:.2f} us/step wall, {best[2]:.2f} us/step device",
          flush=True)
    print(f"    -> wall {base[1]/best[1]:.2f}x, device {base[2]/best[2]:.2f}x vs baseline "
          f"single-dispatch", flush=True)
    frame_red = RENDER_REDUCED_FRAME / best[3]
    print(f"    sec/frame (358,058 reduced) = {frame_red:.3f} s  ({1.0/frame_red:.3f} fps)",
          flush=True)
    print(f"    sec/frame (6.89M full)      = {DOOM_FRAME_INSTRS/best[3]:.1f} s", flush=True)
    print(f"    x above 0.0071us FLOP floor = {best[1]/FLOP_FLOOR_US:.0f}x (wall) / "
          f"{best[2]/FLOP_FLOOR_US:.0f}x (device)", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-byte-exact", action="store_true")
    args = ap.parse_args(argv)
    _guard()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _all_off()
    _composed_on()
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()
    ok = True
    if not args.skip_byte_exact:
        ok = byte_exact(model, L, device)
    measure(model, L, device)
    print(f"\n{'=== ONCHIP FLOOR COMPLETE ===' if ok else '=== BYTE-EXACT FAILED ==='}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
