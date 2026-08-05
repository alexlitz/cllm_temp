#!/usr/bin/env python3
"""_agent_livecam_split.py — split the composed LIVE-CAM 0.69 us/step into its two
components on the FAST (composed single-dispatch) path, to attribute whether a HASH-CAM
could help the fast path.

The live-CAM per-step cost (measured 0.6945 us/step, 28% of the composed step) is, per
``precomputed_schedule._LiveCamBlock.forward_static_delta``:

    res = h.clone(); res[0, :, out_dims] += wo_delta   # (a) W_o-delta SCATTER-add
    return self.ffn.forward(res)                        # (b) block FFN GEMM

There is NO softmax scan, NO gather-over-stores at dispatch — the gather is precomputed
into ``wo_delta`` at BUILD time (``resolve_load_rows`` -> latest-write-wins hash dict).
So the live-CAM dispatch cost is entirely (a)+(b) = fixed head/FFN machinery.  A HASH-CAM
resolves the ADDRESS->SLOT step, which is already O(1) at build and ~0 at dispatch — it
cannot touch (a) or (b).  This probe measures (a) and (b) separately to quantify the
ceiling a search-eliminating lever (hash) could buy on the FAST path (= 0, modulo the
already-measured 0.05 us in-place scatter micro-win).

Run: CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_livecam_split
"""
from __future__ import annotations
import os, time, gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested
from c4_min import precomputed_schedule as PS
from c4_min.direct_cam_batched import cam_head_map


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


def _cuda_time_ms(fn, reps=30, warmup=8, dev=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    st = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize(dev)
    ts = sorted(st[i].elapsed_time(en[i]) for i in range(reps))
    keep = ts[2:-2] if len(ts) > 6 else ts
    return sum(keep) / max(1, len(keep))


def main():
    _guard()
    dev = torch.device("cuda:0")
    COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
                "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
                "C4_DIRECT_CAM_VEC", "C4_ONCHIP_RESIDUAL", "C4_RESIDENT_BATCH",
                "C4_SCHED_FAST_BUILD"]
    for f in COMPOSED:
        os.environ[f] = "1"

    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.dim} build={time.time()-t0:.1f}s",
          flush=True)
    _guard()

    code = build_nested(120, 255)[0]
    draft = draft_pf_program(code, max_steps=3_000_000, mask=0xFFFFFFFF)
    assert draft.halted
    print(f"[frame] nested_120_255 {draft.step_count} steps", flush=True)

    install_composed(model, verbose=False)
    try:
        chunk = 65536
        sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)
        sg.chunk = chunk
        h0 = sched.h0_folded[:chunk].unsqueeze(0)
        delta = {b: t[:chunk] for b, t in sched.cam_delta_tables.items()}
        sg.replay(h0, None, None, delta=delta, resident=True)
        torch.cuda.synchronize(dev)

        C, D = chunk, model.dim
        live = sg.live_blocks              # {block_idx: _LiveCamBlock}
        outdims = sg.live_out_dims         # {block_idx: LongTensor}

        # a representative residual [1,C,D] (the mega chain's output feeding live blocks).
        h_in = torch.randn(1, C, D, device=dev)

        # ---- (a) ONLY the W_o-delta scatter-add for all live blocks (no FFN) ----
        def _cap(bodyfn):
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        bodyfn()
            torch.cuda.current_stream(dev).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    bodyfn()
            return g

        deltas_s = {b: torch.randn(C, int(outdims[b].numel()), device=dev) for b in live}

        def _scatter_only():
            h = h_in
            for b in sg.live_order:
                lb = live[b]
                res = h.clone()
                res[0, :, outdims[b]] += deltas_s[b]
                h = res
            return h
        gS = _cap(_scatter_only)
        scatter_us = _cuda_time_ms(lambda: gS.replay(), 30, 8, dev) * 1e3 / C

        # ---- (a)+(b) the full live-CAM region: scatter + FFN, per _run_region ----
        def _scatter_ffn():
            h = h_in
            for b in sg.live_order:
                lb = live[b]
                h = lb.forward_static_delta(h, deltas_s[b], outdims[b])
            return h
        gSF = _cap(_scatter_ffn)
        scatter_ffn_us = _cuda_time_ms(lambda: gSF.replay(), 30, 8, dev) * 1e3 / C

        ffn_us = scatter_ffn_us - scatter_us

        print("\n=== LIVE-CAM 0.69 us/step SPLIT (fast composed dispatch) ===", flush=True)
        print(f"  live blocks: {[(b, int(outdims[b].numel())) for b in sg.live_order]}",
              flush=True)
        print(f"  (a) W_o-delta SCATTER-add only : {scatter_us:8.4f} us/step", flush=True)
        print(f"  (a)+(b) scatter + block FFN    : {scatter_ffn_us:8.4f} us/step", flush=True)
        print(f"  (b) block FFN GEMM (derived)   : {ffn_us:8.4f} us/step", flush=True)
        tot = max(scatter_ffn_us, 1e-9)
        print(f"\n  -> FFN GEMM is {100*ffn_us/tot:.0f}% of the live-CAM cost; scatter is "
              f"{100*scatter_us/tot:.0f}%.", flush=True)
        print(f"  -> NEITHER is a search/gather over stores (gather precomputed -> 0 at "
              f"dispatch).  A hash-CAM (address->slot O(1)) cannot reduce (a) or (b): the "
              f"resolution is already O(1) at BUILD (latest-write-wins dict) and ~0 at "
              f"dispatch.  The fast-path live-CAM wall is FIXED head/FFN machinery.",
              flush=True)
    finally:
        uninstall_composed(model)
        for f in COMPOSED:
            os.environ.pop(f, None)
        gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
