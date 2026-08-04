#!/usr/bin/env python3
"""Trace WHERE the .item()/_local_scalar_dense host-syncs come from: monkeypatch
torch.Tensor.item to tally callers (file:lineno of the caller frame)."""
import os, collections, traceback
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

_tally = collections.Counter()
_orig_item = torch.Tensor.item


def _traced_item(self):
    st = traceback.extract_stack(limit=4)
    # the caller is the frame two up (skip this wrapper).
    fr = st[-2]
    _tally[f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"] += 1
    return _orig_item(self)


torch.Tensor.item = _traced_item

LEVERS = ["C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM", "C4_BANDED_LOCAL_ATTN",
          "C4_DIRECT_CAM_LIVE_LOCAL", "C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION",
          "C4_BATCHED_BLOCK_SKIP", "C4_OVERLAY_BATCHED", "C4_BATCHED_DECODE",
          "C4_EXACT_EVICT", "C4_DIRECT_CAM_VEC"]


def _set(**kw):
    for f in LEVERS:
        os.environ.pop(f, None)
    for f, v in kw.items():
        os.environ[f] = v


def main():
    dev = os.environ.get("PROBE_DEV", "cuda:0")
    K = int(os.environ.get("K", "256"))
    from .lib_neural import build_lib_model_streaming
    from .local_attention import install_local_attention
    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import build_malloc
    _set(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1", C4_BANDED_LOCAL_ATTN="1",
         C4_DIRECT_CAM_LIVE_LOCAL="1", C4_FROZEN_ROW_SKIP="1", C4_BATCHED_BLOCK_SKIP="1",
         C4_DEAD_BLOCK_FUSION="1", C4_OVERLAY_BATCHED="1", C4_BATCHED_DECODE="1",
         C4_EXACT_EVICT="1", C4_DIRECT_CAM_VEC="1")
    code = build_malloc(int(os.environ.get("NMALLOC", "48")))[0]
    draft = draft_pf_program(code, max_steps=40000, mask=0xFFFFFFFF)
    n_steps = draft.step_count
    model, L, _ = build_lib_model_streaming(code_size=192, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    _tally.clear()
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, fast=True)
    print(f"[sync-trace] steps={n_steps} matched={vr.all_matched} "
          f"total .item()={sum(_tally.values())} ({sum(_tally.values())/n_steps:.1f}/step)",
          flush=True)
    print("[sync-trace] TOP .item() call sites:", flush=True)
    for site, cnt in _tally.most_common(20):
        print(f"    {cnt:7d}  {site}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
