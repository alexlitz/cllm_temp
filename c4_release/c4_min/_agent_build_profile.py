#!/usr/bin/env python3
"""_agent_build_profile.py — profile the schedule BUILD sub-components at scale.

The whole-frame giant-K harness (``_agent_wholeframe_giantk``) found the one-time
schedule BUILD (~5.05 us/step, 44% of a 358 K frame) is the new dominant cost.  This
tool drafts a medium DIV-free nested loop (fast CPU draft), then times each BUILD
sub-step of ``PS.build_schedule`` SEPARATELY so we know exactly which O(n_steps) piece
to move on-device / vectorize.  Run before/after the optimization.

Run: CUDA_VISIBLE_DEVICES=0 C4_PF_CFM=1 OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> \
     python -m c4_min._agent_build_profile --device cuda:0 --outer 120 --inner 200
"""
from __future__ import annotations
import argparse, os, time, gc

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


COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _composed_on():
    for f in COMPOSED:
        os.environ[f] = "1"


def _profile_build(model, L, code, draft, dev, mask=0xFFFFFFFF, fast_build=True):
    """Re-run the pieces of build_schedule with per-piece timers.  Mirrors the real
    build_schedule ordering; returns a dict of piece -> ms."""
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1" if fast_build else "0"
    from c4_min.fused_megablock import install_fused_megablock, MegaBlockRegion
    from c4_min.direct_cam_batched import cam_head_map, build_resolved_table
    from c4_min.pf_speculative import _frozen_skip_cut
    from c4_min.nibble_evict_schedule import resolve_load_rows

    n = draft.step_count
    tm = {}

    def _t(name, fn):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize(dev)
        tm[name] = (time.perf_counter() - t0) * 1e3
        return r

    cut = _frozen_skip_cut(model)
    mega = _t("mega_region", lambda: install_fused_megablock(model, dev, cut, L=None, verbose=False))
    if mega is None:
        mega = MegaBlockRegion(model, dev, cut)
    chm = _t("cam_head_map", lambda: cam_head_map(model, L))
    live_blocks = [PS._LiveCamBlock(model, bi, heads) for bi, heads in sorted(chm.items())]

    onchip = True
    h0_table = _t("query_embed", lambda: PS._build_query_embed(model, L, code, draft, dev))
    ing_table = _t("ingest_table", lambda: PS._build_ingest_table(model, L, draft, dev))
    if os.environ.get("_PROFILE_GPU_PATH") == "1":
        # profile the NEW GPU-build sub-pieces.
        reads_holder = {}
        _t("gpu_read_store_arrays",
           lambda: reads_holder.setdefault("rs", PS._draft_read_store_arrays(draft)))
        _t("gpu_resolve_reads_vec",
           lambda: PS._resolve_reads_vec(*reads_holder["rs"]))
        cam_sparse = _t("gpu_cam_sparse", lambda: PS._build_cam_sparse_gpu(draft, code, live_blocks, dev))
        _t("gpu_decode_targets", lambda: PS._decode_targets_gpu(draft, mask, dev))
        return tm
    # cam sparse build sub-timed: resolve_load_rows + build_resolved_table + scatter
    reads_holder = {}
    _t("resolve_load_rows", lambda: reads_holder.setdefault("r", resolve_load_rows(draft)))
    _t("build_resolved_table", lambda: reads_holder.setdefault("t", build_resolved_table(draft, code)))
    cam_sparse = _t("cam_sparse_scatter", lambda: PS._build_cam_sparse(draft, code, live_blocks))

    # decode targets loop
    def _decode_targets():
        import numpy as _np
        frames = draft.frames
        _pc = _np.empty(n, dtype=_np.int64); _ax = _np.empty(n, dtype=_np.int64)
        _sp = _np.empty(n, dtype=_np.int64); _bp = _np.empty(n, dtype=_np.int64)
        _hl = _np.empty(n, dtype=_np.bool_); _fl = _np.empty(n, dtype=_np.bool_)
        for s in range(n):
            f = frames[s]
            _pc[s] = f["pc"]; _ax[s] = f["ax"] & mask
            _sp[s] = f["sp"] & 0xFFFFFFFF; _bp[s] = f["bp"] & 0xFFFFFFFF
            _hl[s] = bool(f.get("is_halt")); _fl[s] = bool(f.get("is_file"))
        return (torch.from_numpy(_pc).to(dev), torch.from_numpy(_ax).to(dev),
                torch.from_numpy(_sp).to(dev), torch.from_numpy(_bp).to(dev),
                torch.from_numpy(_hl).to(dev), torch.from_numpy(_fl).to(dev))
    _t("decode_targets", _decode_targets)

    # wo delta build
    def _wo_delta():
        D = model.dim
        Wo0 = PS._wo_dense(model, 0, dev)
        ing_flat = ing_table.reshape(n, D)
        h0_folded = h0_table + ing_flat @ Wo0.transpose(0, 1)
        for lb in live_blocks:
            b = lb.block_idx
            Wo = PS._wo_dense(model, b, dev)
            vals_np, cols_np, _HDb = cam_sparse[b]
            vals = torch.from_numpy(vals_np).to(dev)
            cols = torch.from_numpy(cols_np).to(dev)
            Wo_sub = Wo.index_select(1, cols)
            full = vals @ Wo_sub.transpose(0, 1)
            out_dims = (full.abs().sum(0) > 0).nonzero(as_tuple=False).flatten()
            delta = full.index_select(1, out_dims).contiguous()
        return h0_folded
    _t("wo_delta", _wo_delta)

    for f in ("C4_ONCHIP_RESIDUAL", "C4_RESIDENT_BATCH", "C4_PRECOMPUTED_SCHEDULE",
              "C4_SCHED_FAST_BUILD"):
        os.environ.pop(f, None)
    return tm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outer", type=int, default=120)
    ap.add_argument("--inner", type=int, default=200)
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    _composed_on()
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} {time.time()-t0:.1f}s",
          flush=True)
    code = build_nested(args.outer, args.inner)[0]
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=3_000_000, mask=0xFFFFFFFF)
    print(f"[draft] nested({args.outer},{args.inner}) steps={draft.step_count} "
          f"halted={draft.halted} {time.time()-t0:.1f}s", flush=True)
    install_composed(model, verbose=False)
    try:
        # warm once (import/compile), then measure.
        PS.build_schedule(model, L, code, draft, dev)
        n = draft.step_count
        # ---- per-piece PROFILE (dict path) ----
        tm = _profile_build(model, L, code, draft, dev, fast_build=True)
        total = sum(tm.values())
        print(f"\n=== BUILD PROFILE (dict path)  n={n} steps ===", flush=True)
        for k, v in sorted(tm.items(), key=lambda kv: -kv[1]):
            print(f"  {k:>22}: {v:9.2f} ms  {v/n*1e6:8.3f} us/step  ({100*v/total:4.1f}%)",
                  flush=True)
        print(f"  {'TOTAL':>22}: {total:9.2f} ms  {total/n*1e6:8.3f} us/step", flush=True)

        # ---- END-TO-END build_schedule timing: dict path vs GPU-build path ----
        def _time_build(gpu_build):
            os.environ["C4_ONCHIP_RESIDUAL"] = "1"
            os.environ["C4_RESIDENT_BATCH"] = "1"
            os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
            os.environ["C4_SCHED_FAST_BUILD"] = "1"
            os.environ["C4_SCHED_GPU_BUILD"] = "1" if gpu_build else "0"
            ts = []
            for _ in range(3):
                torch.cuda.synchronize(dev); t0 = time.perf_counter()
                PS.build_schedule(model, L, code, draft, dev)
                torch.cuda.synchronize(dev); ts.append(time.perf_counter() - t0)
            for f in ("C4_ONCHIP_RESIDUAL", "C4_RESIDENT_BATCH", "C4_PRECOMPUTED_SCHEDULE",
                      "C4_SCHED_FAST_BUILD", "C4_SCHED_GPU_BUILD"):
                os.environ.pop(f, None)
            return min(ts)
        b_dict = _time_build(False)
        b_gpu = _time_build(True)
        print(f"\n=== END-TO-END build_schedule  n={n} steps ===", flush=True)
        print(f"  dict-path build : {b_dict*1e3:9.2f} ms  {b_dict/n*1e6:8.3f} us/step",
              flush=True)
        print(f"  GPU-build       : {b_gpu*1e3:9.2f} ms  {b_gpu/n*1e6:8.3f} us/step",
              flush=True)
        print(f"  -> BUILD speedup {b_dict/max(b_gpu,1e-9):.2f}x", flush=True)
    finally:
        uninstall_composed(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
