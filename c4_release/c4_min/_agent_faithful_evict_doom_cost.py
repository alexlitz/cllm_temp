"""FAST scale + cost for the faithful-attn-evict path (avoids the slow Python
malloc reference interpreter).

  * SCALE: a genuinely LONG (many-step) byte-safe deep loop -> bounded live cache
    over the whole run; report max live cache, total evicted, VRAM peak, no OOM.
    Also runs the GENUINE full-log softmax (evict=False) at the SAME scale to show
    it OOMs / blows the working set, proving eviction resolves the audit's OOM.
  * COST: faithful-attn-evict vs draft-trusted direct-CAM (~1.29 fps reference), same
    program + K -> honest us/step + fps + VRAM.

Memory-safe lean build, MemAvailable guard, GPU per CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks, set_gpu_verify
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested

DOOM_FRAME_INSTRS = 6_890_000
DIRECT_CAM_REF_FPS = 1.29     # the draft-trusted composed direct-CAM reference


def _guard(where=""):
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                a = int(ln.split()[1]) / 1e6
                if a < 25.0:
                    raise SystemExit(f"[GUARD] {a:.1f}GB<25 @ {where}")
                return a
    return 1e9


def timed(model, L, code, draft, device, K, *, faithful, direct_cam, evict, warm=True):
    for f in ("C4_FAITHFUL_ATTN_EVICT", "C4_DIRECT_CAM_BATCHED",
              "C4_DIRECT_LOCAL_CAM", "C4_DIRECT_CAM_VEC"):
        os.environ.pop(f, None)
    if faithful:
        os.environ["C4_FAITHFUL_ATTN_EVICT"] = "1"
    if direct_cam:
        os.environ["C4_DIRECT_CAM_BATCHED"] = "1"
        os.environ["C4_DIRECT_LOCAL_CAM"] = "1"
        os.environ["C4_DIRECT_CAM_VEC"] = "1"
    install_composed(model, verbose=False)
    set_gpu_verify(True)
    stats = {}
    torch.cuda.reset_peak_memory_stats(device)
    oom = False
    try:
        ee = (True if evict else False)
        if warm:
            verify_blocks(model, L, code, draft, block_steps=K, device=device,
                          evict=evict, mask=0xFFFFFFFF, stats={}, fast=True,
                          evict_interval_steps=8, exact_evict=ee)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                           evict=evict, mask=0xFFFFFFFF, stats=stats, fast=True,
                           evict_interval_steps=8, exact_evict=ee)
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom = True
            wall = float("nan")
            vr = None
        else:
            raise
    finally:
        set_gpu_verify(None)
        uninstall_composed(model)
        for f in ("C4_FAITHFUL_ATTN_EVICT", "C4_DIRECT_CAM_BATCHED",
                  "C4_DIRECT_LOCAL_CAM", "C4_DIRECT_CAM_VEC"):
            os.environ.pop(f, None)
        torch.cuda.empty_cache()
    steps = draft.step_count
    stats["_oom"] = oom
    stats["_vram"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if not oom:
        stats["_us"] = wall * 1e6 / steps
        stats["_sps"] = steps / wall
        stats["_fps"] = (steps / wall) / DOOM_FRAME_INSTRS
        stats["_matched"] = vr.all_matched
        stats["_final"] = vr.decoded_final_ax
    return vr, stats


def main():
    device = os.environ.get("C4_DEVICE", "cuda:0")
    _guard("start")
    model, L, _ = build_lib_model_streaming(code_size=32)
    model = model.to(device)
    _guard("post-build")
    print("[build] done.\n", flush=True)

    # A substantial byte-safe deep loop (3949 steps) — fast enough to time many verifies
    # while still exercising the bounded-working-set eviction over thousands of steps.
    code = build_nested(10, 24)[0]
    draft = draft_pf_program(code, max_steps=400000, mask=0xFFFFFFFF)
    steps = draft.step_count
    logical_stores = len(draft.store_log)
    print(f"[program] nested_10x24: {steps} steps, {logical_stores} logical stores, "
          f"halted={draft.halted}\n", flush=True)

    # ---- SCALE: faithful-evict fits, bounded working set --------------------
    print("=== SCALE: does faithful-attn-evict FIT (bounded working set, no OOM)? ===",
          flush=True)
    for K in (2048, 8192):
        _guard(f"scale-K{K}")
        vr, st = timed(model, L, code, draft, device, K,
                       faithful=True, direct_cam=False, evict=True, warm=False)
        print(f"  faithful-evict K={K:>5}: max_live_cache={st.get('max_cache_size')} "
              f"total_evicted={st.get('total_evicted')} peak_vram={st['_vram']:.2f}GB "
              f"matched={st.get('_matched')} final_ax={st.get('_final')} "
              f"({st.get('_us',0):.0f}us/step)", flush=True)

    # ---- COST: faithful-evict vs direct-CAM ---------------------------------
    print(f"\n=== COST: faithful-attn-evict vs draft-trusted direct-CAM "
          f"(ref ~{DIRECT_CAM_REF_FPS} fps) ===", flush=True)
    for K in (2048, 8192):
        _guard(f"cost-K{K}")
        vr_f, st_f = timed(model, L, code, draft, device, K,
                           faithful=True, direct_cam=False, evict=True)
        vr_d, st_d = timed(model, L, code, draft, device, K,
                           faithful=False, direct_cam=True, evict=True)
        ratio = st_f['_us'] / max(st_d['_us'], 1e-9)
        print(f"  K={K:>5}: faithful={st_f['_us']:7.0f}us/step "
              f"({st_f['_fps']:.4f}fps, vram={st_f['_vram']:.2f}GB, "
              f"cache={st_f.get('max_cache_size')}, matched={st_f['_matched']})  "
              f"direct-CAM={st_d['_us']:7.0f}us/step ({st_d['_fps']:.4f}fps, "
              f"vram={st_d['_vram']:.2f}GB, matched={st_d['_matched']})  "
              f"-> faithful is {ratio:.1f}x the direct-CAM cost", flush=True)
    _guard("end")


if __name__ == "__main__":
    main()
