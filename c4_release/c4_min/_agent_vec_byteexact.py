#!/usr/bin/env python3
"""Byte-exact check: C4_DIRECT_CAM_VEC (vectorized CAM-output gather) vs the
per-row loop.  Runs verify_blocks with the full lever stack in both modes and
asserts identical all_matched + decoded_final_ax + per-step (implicit via match).
"""
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

LEVERS = ["C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM", "C4_BANDED_LOCAL_ATTN",
          "C4_DIRECT_CAM_LIVE_LOCAL", "C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION",
          "C4_BATCHED_BLOCK_SKIP", "C4_OVERLAY_BATCHED", "C4_BATCHED_DECODE",
          "C4_EXACT_EVICT", "C4_GRAPH_MEGAKERNEL", "C4_DIRECT_CAM_VEC"]


def _set(**kw):
    for f in LEVERS:
        os.environ.pop(f, None)
    for f, v in kw.items():
        os.environ[f] = v


def _fresh(dev):
    from .lib_neural import build_lib_model_streaming
    from .local_attention import install_local_attention
    model, L, _ = build_lib_model_streaming(code_size=192, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    return model, L


def main():
    dev = os.environ.get("PROBE_DEV", "cuda:0")
    K = int(os.environ.get("K", "128"))
    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import build_malloc
    code = build_malloc(int(os.environ.get("NMALLOC", "24")))[0]
    draft = draft_pf_program(code, max_steps=40000, mask=0xFFFFFFFF)
    n_steps = draft.step_count
    print(f"[vec-be] malloc steps={n_steps} K={K}", flush=True)

    BASE = dict(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
                C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
                C4_FROZEN_ROW_SKIP="1", C4_BATCHED_BLOCK_SKIP="1",
                C4_DEAD_BLOCK_FUSION="1", C4_OVERLAY_BATCHED="1",
                C4_BATCHED_DECODE="1", C4_EXACT_EVICT="1")

    res = {}
    for tag, vec in [("loop", "0"), ("vec", "1")]:
        _set(**dict(BASE, C4_DIRECT_CAM_VEC=vec))
        m, L = _fresh(dev)
        # collect the per-step accepted register stream via collect_out proxy: use
        # decoded_final_ax + accepted_steps + all_matched as the byte-exact signature.
        vr = verify_blocks(m, L, code, draft, block_steps=K, device=dev, evict=True,
                           mask=0xFFFFFFFF, fast=True)
        res[tag] = (vr.all_matched, vr.accepted_steps, vr.decoded_final_ax)
        del m
        torch.cuda.empty_cache()
        print(f"[vec-be] {tag}: matched={vr.all_matched} accepted={vr.accepted_steps} "
              f"final_ax={vr.decoded_final_ax}", flush=True)

    ok = res["loop"] == res["vec"]
    print(f"[vec-be] BYTE-EXACT loop==vec: {ok}  loop={res['loop']} vec={res['vec']}",
          flush=True)
    _set()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
