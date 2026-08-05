"""BYTE-EXACT: faithful-attn-evict (genuine softmax over the EVICTED / bounded
cache) == genuine softmax over the FULL store log == K=1 reference == draft.

For each program in the DIV-free battery + the 5455-step deep loop:
  * REF (K=1)                     : draft_pf_program is the bit-exact ISA one-step
                                    reference; its per-step ax IS the K=1 ground truth.
  * genuine full-log (evict=False): faithful softmax over the WHOLE growing store log.
  * faithful-evict (evict=True)   : faithful softmax over the EVICTED bounded cache.

We assert: both faithful runs accept the WHOLE trace (all_matched) and decode the
SAME final ax as each other AND as the draft's final ax (the K=1 ref).  That is
L-inf=0 across the three: eviction did NOT change the genuine-attention result.

We also cross-check the model's decoded per-step AX (via accepted_steps ==
step_count) is identical evict vs no-evict for every K.

Memory-safe: lean streaming build, MemAvailable guard, GPU 0, small Ks.
"""
from __future__ import annotations

import os

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
from c4_min.bench_composed_fast_path import _battery
from c4_min.bench_fast_path import build_loop_countdown, build_nested


def _guard(where=""):
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                a = int(ln.split()[1]) / 1e6
                if a < 25.0:
                    raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB<25 @ {where}")
                return a
    return 1e9


def faithful_run(model, L, code, draft, device, K, evict):
    os.environ["C4_FAITHFUL_ATTN_EVICT"] = "1"
    install_composed(model, verbose=False)
    stats = {}
    try:
        vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                           evict=evict, mask=0xFFFFFFFF, stats=stats, fast=True,
                           evict_interval_steps=8, exact_evict=(True if evict else False))
    finally:
        uninstall_composed(model)
        os.environ.pop("C4_FAITHFUL_ATTN_EVICT", None)
    return vr, stats


def main():
    device = os.environ.get("C4_DEVICE", "cuda:0")
    _guard("start")
    print(f"[build] lean streaming model on {device} ...", flush=True)
    model, L, _ = build_lib_model_streaming(code_size=32)
    model = model.to(device)
    _guard("post-build")
    print("[build] done.\n", flush=True)

    progs = []
    for name, prog, seed in _battery():
        if seed:
            continue                 # composed doom path is DIV-free / seed-mem-free
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        progs.append((name, code))
    progs.append(("loop_countdown60", build_loop_countdown(60)[0]))
    # DEEP loop ~5455 steps, BYTE-SAFE (all values <=255 so the model's 8-bit ALU is
    # byte-exact to the 32-bit draft): nested 14x77 => ~5455 steps.
    deep = build_nested(14, 77)[0]
    dd = draft_pf_program(deep, max_steps=400000, mask=0xFFFFFFFF)
    progs.append((f"nested_14x77_deep({dd.step_count}steps)", deep))
    # REAL-DOOM SLICE: the exact FixedMul renderer routine (((int64)a*b)>>16), the
    # kernel r_draw/r_segs/r_main call — a genuine doom fixed-point program.
    from c4_min.measure_doom_fixedpoint import _build_call_program
    from c4_min.doom_fixedpoint import FRACUNIT
    doom_img, _ = _build_call_program("mul", 3 * FRACUNIT, 2 * FRACUNIT)
    progs.append(("doom_FixedMul(3.0,2.0)", doom_img))
    doom_img2, _ = _build_call_program("mul", 137 * FRACUNIT + 4211, 91 * FRACUNIT + 55)
    progs.append(("doom_FixedMul(big)", doom_img2))

    Ks = [512, 2048]
    print(f"{'prog':>20} {'steps':>7} {'K':>6} {'ref_ax':>8} {'fulllog_ax':>10} "
          f"{'evict_ax':>9} {'ev_cache':>9} {'ev_dropped':>10} {'Linf0':>6}", flush=True)
    all_ok = True
    for name, code in progs:
        draft = draft_pf_program(code, max_steps=400000, mask=0xFFFFFFFF)
        if not draft.halted:
            print(f"{name:>20}  draft did not halt, skip", flush=True)
            continue
        ref_ax = draft.final_ax_masked      # K=1 bit-exact reference final AX
        for K in Ks:
            _guard(f"{name}-K{K}")
            vr_full, st_full = faithful_run(model, L, code, draft, device, K, evict=False)
            vr_ev, st_ev = faithful_run(model, L, code, draft, device, K, evict=True)
            full_ax = vr_full.decoded_final_ax
            ev_ax = vr_ev.decoded_final_ax
            ev_cache = st_ev.get("max_cache_size", -1)
            ev_dropped = st_ev.get("total_evicted", -1)
            linf0 = (vr_full.all_matched and vr_ev.all_matched
                     and full_ax == ref_ax and ev_ax == ref_ax
                     and vr_full.accepted_steps == vr_ev.accepted_steps == draft.step_count)
            all_ok = all_ok and linf0
            print(f"{name:>20} {draft.step_count:>7} {K:>6} {str(ref_ax):>8} "
                  f"{str(full_ax):>10} {str(ev_ax):>9} {ev_cache:>9} {ev_dropped:>10} "
                  f"{'YES' if linf0 else 'NO':>6}", flush=True)
    print(f"\n[byte-exact] ALL programs L-inf=0 (evict==full-log==ref==draft): "
          f"{'YES' if all_ok else 'NO'}", flush=True)
    _guard("end")


if __name__ == "__main__":
    main()
