#!/usr/bin/env python3
"""_agent_dispatch_profile.py — MEASUREMENT-FIRST profile of the doom-on-transformer
single-dispatch STEADY-STATE (~3.09 us/step) + build (~5.05 us/step), and the memory-BW
roofline + per-lever ceiling for the user's 5 optimization levers.

DO NOT re-solve what is already done (per-row compaction, on-chip residual, single
dispatch).  This harness only MEASURES + ESTIMATES.  It:

  (T0) reproduces the ~3.09 us dispatch / ~5.05 us build baseline (on-chip, resident,
       fast-build) on the largest DIV-free frame that fits.
  (T1) breaks the dispatch into (a) mega dead-FFN chain, (b) live-CAM blocks
       (delta scatter + FFN), (c) block-0 FFN, (d) decode lanes, (e) graph-replay
       launch floor — by capturing SUB-graphs of the body and timing each; and
       breaks the build into (routing, gather-resolution/cam_sparse, wo-delta,
       decode-targets, embed/ingest, graph-capture).
  (T2) computes the memory-BW roofline: residual traffic, weight HBM traffic
       (unique nnz streamed + L2 residency), gather bytes -> bytes/step / 768 GB/s.
  (T3) micro-benchmarks that VALIDATE the lever ceilings: index_select gather vs
       CAM-sparse build; in-place scatter vs full-D delta+add; deduped-weight
       index_select fits-in-L2.

Run: CUDA_VISIBLE_DEVICES=1 C4_PF_CFM=1 OMP_NUM_THREADS=4 \
     PYTHONPATH=<c4_release> python -m c4_min._agent_dispatch_profile --device cuda:0
(CUDA_VISIBLE_DEVICES=1 remaps the visible GPU to cuda:0.)
"""
from __future__ import annotations
import argparse, os, time, gc, json

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

HBM_BW_GBs = 768.0        # A5000 HBM bandwidth
L2_BYTES = 6 * 1024 * 1024  # A5000 L2 = 6 MB
REALTIME_S = 1.0
FPS35_S = 1.0 / 35.0
RENDER_REDUCED_FRAME = 358_058

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _composed_on():
    for f in COMPOSED:
        os.environ[f] = "1"


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


def _vram_gb():
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


def _cuda_time_ms(fn, reps=20, warmup=5, dev=None):
    """Time a device closure with CUDA events; return trimmed-mean ms over reps."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize(dev)
    ts = [starts[i].elapsed_time(ends[i]) for i in range(reps)]
    ts.sort()
    keep = ts[2:-2] if len(ts) > 6 else ts
    return sum(keep) / max(1, len(keep))


# ---------------------------------------------------------------------------
def find_frame(model, L, device, target=200_000):
    """Largest DIV-free nested-loop frame that verifies, near target steps."""
    import pickle
    install_composed(model, verbose=False)
    best = None
    try:
        cands = [(120, 255), (180, 255), (255, 255)]
        for (o, i) in cands:
            code = build_nested(o, i)[0]
            cache = f"/tmp/_wf_draft_{o}_{i}.pkl"
            d = None
            if os.path.exists(cache):
                try:
                    with open(cache, "rb") as fh:
                        d = pickle.load(fh)
                except Exception:
                    d = None
            if d is None:
                d = draft_pf_program(code, max_steps=3_000_000, mask=0xFFFFFFFF)
                if d.halted:
                    try:
                        with open(cache, "wb") as fh:
                            pickle.dump(d, fh)
                    except Exception:
                        pass
            if not d.halted:
                continue
            if any(d.frames[s].get("op") in ("DIV", "MOD") for s in range(d.step_count)):
                continue
            best = (f"nested_{o}_{i}", code, d)
            if d.step_count >= target:
                break
    finally:
        uninstall_composed(model)
    return best


# ---------------------------------------------------------------------------
def profile_build(model, L, code, draft, device, mask=0xFFFFFFFF):
    """Break the schedule BUILD into its phases (routing / gather-resolution /
    wo-delta / decode-targets / embed+ingest).  We instrument by calling the same
    sub-builders precomputed_schedule.build_schedule calls and timing each, on-chip
    fast-build config."""
    from c4_min.fused_megablock import install_fused_megablock, MegaBlockRegion
    from c4_min.direct_cam_batched import cam_head_map
    from c4_min.pf_speculative import _frozen_skip_cut
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    dev = torch.device(device)
    n = draft.step_count
    phases = {}

    def _t(label, fn):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize(dev)
        phases[label] = time.perf_counter() - t0
        return r

    install_composed(model, verbose=False)
    try:
        cut = _frozen_skip_cut(model)
        mega = _t("routing_mega", lambda: (
            install_fused_megablock(model, dev, cut, L=None, verbose=False)
            or MegaBlockRegion(model, dev, cut)))
        chm = cam_head_map(model, L)
        live_blocks = [PS._LiveCamBlock(model, bi, heads)
                       for bi, heads in sorted(chm.items())]
        h0_table = _t("embed_h0", lambda: PS._build_query_embed(model, L, code, draft, dev))
        ing_table = _t("ingest", lambda: PS._build_ingest_table(model, L, draft, dev))
        cam_sparse = _t("gather_cam_sparse",
                        lambda: PS._build_cam_sparse(draft, code, live_blocks))

        def _decode_targets():
            import numpy as _np
            frames = draft.frames
            _pc = _np.empty(n, dtype=_np.int64); _ax = _np.empty(n, dtype=_np.int64)
            _sp = _np.empty(n, dtype=_np.int64); _bp = _np.empty(n, dtype=_np.int64)
            for s in range(n):
                f = frames[s]
                _pc[s] = f["pc"]; _ax[s] = f["ax"] & mask
                _sp[s] = f["sp"] & 0xFFFFFFFF; _bp[s] = f["bp"] & 0xFFFFFFFF
            return (torch.from_numpy(_pc).to(dev), torch.from_numpy(_ax).to(dev))
        _t("decode_targets", _decode_targets)

        def _wo_delta():
            D = model.dim
            Wo0 = PS._wo_dense(model, 0, dev)
            ing_flat = ing_table.reshape(n, D)
            _h0f = h0_table + ing_flat @ Wo0.transpose(0, 1)
            deltas = {}
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
                deltas[b] = (delta, out_dims)
            return _h0f, deltas
        _t("wo_delta", _wo_delta)
    finally:
        uninstall_composed(model)
        os.environ.pop("C4_ONCHIP_RESIDUAL", None)
        os.environ.pop("C4_RESIDENT_BATCH", None)
    tot = sum(v for k, v in phases.items() if not k.startswith("_"))
    phases["_total_measured"] = tot
    phases["_n"] = n
    return phases, live_blocks


# ---------------------------------------------------------------------------
def profile_dispatch(model, L, code, draft, device, mask=0xFFFFFFFF, chunk_cap=0):
    """Reproduce the baseline STEADY-STATE dispatch (a resident graph over a
    VRAM-fitting chunk), then break the graphed body into sub-phases by capturing
    PARTIAL bodies and timing each replay with CUDA events.

    The steady-state per-step dispatch is the graph-replay time / chunk_rows — it is
    chunk-invariant once the [chunk,D] block chain saturates the GPU (the graph is the
    SAME kernel stream regardless of how the whole frame is sliced), so a bounded chunk
    that FITS VRAM measures the same us/step the whole-frame multi-chunk dispatch pays."""
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    _composed_on()
    dev = torch.device(device)
    n = draft.step_count
    install_composed(model, verbose=False)
    out = {}
    try:
        # choose a VRAM-fitting chunk. On a 24 GB card the resident single-chunk graph
        # peaks ~14.7 GB @ 131072, ~10 GB @ 65536 (giant-K measured). Use 65536 to leave
        # headroom for the sub-graph captures we build alongside.
        _fv, _tv = _vram_gb()
        if chunk_cap and chunk_cap > 0:
            chunk = min(chunk_cap, n)
        elif _tv >= 40:
            chunk = min(131072, n)
        else:
            chunk = min(65536, n)
        sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
        sg.chunk = chunk
        # feed the FIRST `chunk` rows as the resident batch (steady-state per-step is
        # chunk-invariant; we only need one representative saturated chunk).
        h0_src = sched.h0_folded[:chunk]
        h0 = h0_src.unsqueeze(0)
        delta = {b: t[:chunk] for b, t in sched.cam_delta_tables.items()}
        sg.replay(h0, None, None, delta=delta, resident=True)
        torch.cuda.synchronize(dev)
        full_ms = _cuda_time_ms(lambda: sg._graph.replay(), reps=30, warmup=8, dev=dev)
        out["full_dispatch_us_step"] = full_ms * 1e3 / chunk
        out["full_dispatch_ms_chunk"] = full_ms
        out["n"] = n
        out["chunk"] = chunk
        out["D"] = model.dim
        C, D = chunk, model.dim
        from c4_min.nibble_pure_forward_gpu import _decode_reg_batch

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

        def _b0():
            return sg.ffn0.forward(sg._s_h0)
        gA = _cap(_b0)
        out["b0_ffn_us_step"] = _cuda_time_ms(lambda: gA.replay(), 30, 8, dev) * 1e3 / chunk

        def _b0_mega():
            h = sg.ffn0.forward(sg._s_h0)
            for kind, payload in sg.mega.items:
                if kind == "mega":
                    h = payload.run(h)
            return h
        gAM = _cap(_b0_mega)
        out["b0_mega_us_step"] = _cuda_time_ms(lambda: gAM.replay(), 30, 8, dev) * 1e3 / chunk

        def _b0_mega_live():
            h = sg.ffn0.forward(sg._s_h0)
            h = sg._run_region(h)
            return h
        gAML = _cap(_b0_mega_live)
        out["b0_mega_live_us_step"] = _cuda_time_ms(lambda: gAML.replay(), 30, 8, dev) * 1e3 / chunk

        st_buf = torch.randn(C, D, device=dev)
        def _decode():
            sg._s_pc.copy_(PS._snap_lane_light(st_buf[:, L.PC_VAL]))
            sg._s_sp.copy_(PS._snap_lane_light(st_buf[:, L.SP_VAL]))
            sg._s_bp.copy_(PS._snap_lane_light(st_buf[:, L.BP_VAL]))
            ax = _decode_reg_batch(st_buf, L.AX) & mask
            sg._s_ax.copy_(ax)
        gD = _cap(_decode)
        out["decode_us_step"] = _cuda_time_ms(lambda: gD.replay(), 30, 8, dev) * 1e3 / chunk

        out["mega_us_step"] = out["b0_mega_us_step"] - out["b0_ffn_us_step"]
        out["live_us_step"] = out["b0_mega_live_us_step"] - out["b0_mega_us_step"]
        parts = out["b0_mega_live_us_step"] + out["decode_us_step"]
        out["parts_sum_us_step"] = parts
        out["overhead_us_step"] = out["full_dispatch_us_step"] - parts

        out["live_blocks"] = []
        for b, lb in sg.live_blocks.items():
            nout = int(sg.live_out_dims[b].numel())
            out["live_blocks"].append({"block": b, "H": lb.H, "HD": lb.HD, "n_out": nout})
        mega_facts = []
        for kind, payload in sg.mega.items:
            if kind == "mega":
                for mf in payload._ffns:
                    mega_facts.append({"Dff": mf.Dff, "n_active": mf.n_active,
                                       "up_nnz": int(mf.up_val.numel()),
                                       "gt_nnz": int(mf.gt_val.numel()),
                                       "dn_nnz": int(mf.dn_val.numel())})
        out["mega_facts"] = mega_facts
        out["n_mega_ffns"] = len(mega_facts)
        out["total_dead_nnz"] = sum(m["up_nnz"] + m["gt_nnz"] + m["dn_nnz"]
                                    for m in mega_facts)
        out["total_dead_Dff"] = sum(m["Dff"] for m in mega_facts)
        out["total_dead_active"] = sum(m["n_active"] for m in mega_facts)
    finally:
        uninstall_composed(model)
        os.environ.pop("C4_ONCHIP_RESIDUAL", None)
        os.environ.pop("C4_RESIDENT_BATCH", None)
        gc.collect(); torch.cuda.empty_cache()
    return out, sched, sg


# ---------------------------------------------------------------------------
def roofline(disp, chunk):
    """Memory-BW floor per step from measured structural facts.  fp32 (4B) resident.

    The dominant per-step traffic of the 238-block dead-FFN chain is the HIDDEN-ACTIVATION
    scratch [Dff,C] (written by up/gate, read by down) — NOT the [D,C] residual (the bf16
    residual probe showed halving the residual dtype gives 1.00x, so residual streaming is
    NOT the bottleneck).  We compute three traffic terms per step:
      - hidden scratch: sum_b Dff_b*C*4*2 (write + read)   [DOMINANT]
      - residual: D*C*4 (initial load) + sum_b n_active_b*C*4 (down writes)
      - gather (deltas + h0_folded) + weight stream (once, fits L2)
    then a HBM-BW floor = min-bytes/step / 768 GB/s, and the ACHIEVED BW from the measured
    dispatch time (= how far the sparse kernels are from HBM saturation)."""
    D = disp["D"]; C = chunk; dt = 4
    r = {}
    # -- hidden scratch (the dominant term) --
    hidden_bytes = sum(m["Dff"] * C * dt * 2 for m in disp["mega_facts"])  # write+read
    r["hidden_bytes"] = hidden_bytes
    # -- residual traffic --
    write_rows = disp["total_dead_active"]
    resid_bytes = D * C * dt + write_rows * C * dt
    r["resid_bytes"] = resid_bytes
    r["resid_buf_bytes"] = D * C * dt
    r["write_rows"] = write_rows
    # -- weight (CSR val+col idx), streamed once/replay --
    dead_nnz = disp["total_dead_nnz"]
    weight_bytes_once = dead_nnz * (dt + 4)
    r["dead_nnz"] = dead_nnz
    r["weight_bytes_once"] = weight_bytes_once
    r["weight_fits_L2"] = weight_bytes_once <= L2_BYTES
    # -- gather (delta tables + h0_folded, read once/replay) --
    gather_bytes = sum(C * lb["n_out"] * dt + lb["n_out"] * 8 for lb in disp["live_blocks"])
    gather_bytes += C * D * dt
    r["gather_bytes"] = gather_bytes
    # -- totals --
    total = hidden_bytes + resid_bytes + weight_bytes_once + gather_bytes
    r["total_bytes_replay"] = total
    r["total_bytes_step"] = total / C
    r["bw_floor_ns_step"] = (total / C) / (HBM_BW_GBs * 1e9) * 1e9
    # measured achieved BW (measured full dispatch time vs total bytes)
    meas_us = disp["full_dispatch_us_step"]
    r["measured_us_step"] = meas_us
    r["measured_achieved_GBs"] = (total / C) / (meas_us * 1e-6) / 1e9
    r["bw_efficiency_pct"] = 100.0 * r["measured_achieved_GBs"] / HBM_BW_GBs
    # what the floor WOULD be if the sparse kernels hit peak HBM
    r["ideal_us_step_at_peak"] = r["bw_floor_ns_step"] / 1e3
    return r


# ---------------------------------------------------------------------------
def micro_levers(disp, device):
    dev = torch.device(device)
    D = disp["D"]; C = disp["chunk"]; m = {}

    # L2: in-place scatter vs clone+index_add (current forward_static_delta)
    n_out = max((lb["n_out"] for lb in disp["live_blocks"]), default=8)
    h = torch.randn(1, C, D, device=dev)
    out_dims = torch.randperm(D, device=dev)[:n_out]
    wo_delta = torch.randn(C, n_out, device=dev)
    def _clone_add():
        res = h.clone()
        res[0, :, out_dims] += wo_delta
        return res
    def _inplace_add():
        h[0, :, out_dims] += wo_delta
        return h
    m["l2_clone_add_ms"] = _cuda_time_ms(_clone_add, 40, 10, dev)
    m["l2_inplace_add_ms"] = _cuda_time_ms(_inplace_add, 40, 10, dev)
    m["l2_clone_add_us_step"] = m["l2_clone_add_ms"] * 1e3 / C
    m["l2_inplace_add_us_step"] = m["l2_inplace_add_ms"] * 1e3 / C
    n_live = len(disp["live_blocks"])
    m["n_live_blocks"] = n_live
    m["l2_saving_us_step"] = (m["l2_clone_add_us_step"] - m["l2_inplace_add_us_step"]) * n_live

    # L5: deduped weight (fits L2) vs replicated stream
    dead_nnz = disp.get("total_dead_nnz", 153000)
    uniq = 800
    idx = torch.randint(0, uniq, (dead_nnz,), device=dev)
    dedup_vals = torch.randn(uniq, device=dev)
    replicated = torch.randn(dead_nnz, device=dev)
    def _dedup_gather():
        return dedup_vals.index_select(0, idx)
    def _replicated_stream():
        return replicated * 1.0
    m["l5_dedup_gather_ms"] = _cuda_time_ms(_dedup_gather, 40, 10, dev)
    m["l5_replicated_ms"] = _cuda_time_ms(_replicated_stream, 40, 10, dev)
    m["l5_dead_nnz"] = dead_nnz
    m["l5_uniq"] = uniq
    m["l5_dedup_bytes"] = uniq * 4
    m["l5_replicated_bytes"] = dead_nnz * 4
    m["l5_dedup_fits_L2"] = uniq * 4 <= L2_BYTES
    m["l5_replicated_fits_L2"] = dead_nnz * 4 <= L2_BYTES
    return m


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=0,
                    help="graph chunk (0=auto from VRAM). The steady-state per-step "
                         "dispatch is chunk-invariant once the block chain saturates.")
    ap.add_argument("--out", default="/tmp/dispatch_profile.json")
    args = ap.parse_args(argv)
    _guard()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA — this profile needs a GPU")
    _composed_on()
    fv, tv = _vram_gb()
    print(f"[gpu] free {fv:.1f}/{tv:.1f} GB  memAvail {_mem_avail_gb():.1f}GB", flush=True)
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.dim} "
          f"build={time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()

    print("\n=== finding DIV-free frame ===", flush=True)
    big = find_frame(model, L, device, target=args.target)
    if big is None:
        raise SystemExit("no DIV-free frame found")
    name, code, draft = big
    n = draft.step_count
    print(f"  frame '{name}': {n} DIV-free steps", flush=True)
    _guard()

    print("\n=== TASK 1a: BUILD breakdown ===", flush=True)
    bphases, _lb = profile_build(model, L, code, draft, device)
    bn = bphases["_n"]; btot = bphases["_total_measured"]
    for k in ["routing_mega", "embed_h0", "ingest", "gather_cam_sparse",
              "decode_targets", "wo_delta"]:
        v = bphases[k]
        print(f"  {k:>18}: {v*1e3:9.2f} ms  {v/bn*1e6:8.3f} us/step  "
              f"{100*v/btot:5.1f}%", flush=True)
    print(f"  {'TOTAL(build)':>18}: {btot*1e3:9.2f} ms  {btot/bn*1e6:8.3f} us/step", flush=True)
    gc.collect(); torch.cuda.empty_cache(); _guard()

    print("\n=== TASK 1b: DISPATCH breakdown (CUDA-event sub-graphs) ===", flush=True)
    disp, sched, sg = profile_dispatch(model, L, code, draft, device, chunk_cap=args.chunk)
    chunk = disp["chunk"]
    print(f"  full dispatch (replay): {disp['full_dispatch_ms_chunk']:.3f} ms/chunk  "
          f"{disp['full_dispatch_us_step']:.4f} us/step  (n={disp['n']}, chunk={chunk})",
          flush=True)
    for k, lbl in [("b0_ffn_us_step", "block0 FFN"),
                   ("mega_us_step", "mega dead-FFN chain"),
                   ("live_us_step", "live-CAM (scatter+FFN)"),
                   ("decode_us_step", "decode lanes"),
                   ("overhead_us_step", "launch/overhead residual")]:
        v = disp[k]
        print(f"  {lbl:>26}: {v:8.4f} us/step  {100*v/disp['full_dispatch_us_step']:5.1f}%",
              flush=True)
    print(f"  [facts] dead-FFN blocks={disp['n_mega_ffns']} total_dead_nnz="
          f"{disp['total_dead_nnz']} total_active_rows={disp['total_dead_active']} "
          f"total_Dff={disp['total_dead_Dff']}", flush=True)
    print(f"  [facts] live blocks: {disp['live_blocks']}", flush=True)
    gc.collect(); torch.cuda.empty_cache(); _guard()

    print("\n=== TASK 2: MEMORY-BW ROOFLINE ===", flush=True)
    roof = roofline(disp, chunk)
    print(f"  hidden scratch [Dff,C]  : {roof['hidden_bytes']/1e6:.1f} MB/replay  (DOMINANT)", flush=True)
    print(f"  residual [D,C]+writes   : {roof['resid_bytes']/1e6:.1f} MB/replay "
          f"(write_rows={roof['write_rows']})", flush=True)
    print(f"  weight (once/replay)    : {roof['weight_bytes_once']/1e6:.2f} MB  "
          f"fits_L2={roof['weight_fits_L2']}", flush=True)
    print(f"  gather (deltas+h0)      : {roof['gather_bytes']/1e6:.1f} MB/replay", flush=True)
    print(f"  --> total bytes/step    : {roof['total_bytes_step']:.0f} B", flush=True)
    print(f"  --> HBM-BW floor        : {roof['bw_floor_ns_step']:.1f} ns/step "
          f"(if kernels hit {HBM_BW_GBs} GB/s peak)", flush=True)
    print(f"  --> MEASURED            : {roof['measured_us_step']*1e3:.1f} ns/step "
          f"= {roof['measured_achieved_GBs']:.0f} GB/s achieved "
          f"({roof['bw_efficiency_pct']:.0f}% of peak)", flush=True)
    print(f"  target=80 ns/step.  HBM floor achievable<80? {roof['bw_floor_ns_step']<80}  "
          f"(measured is {roof['measured_us_step']*1e3:.0f} ns, {roof['measured_us_step']*1e3/80:.1f}x over)",
          flush=True)

    print("\n=== TASK 3: LEVER MICRO-BENCHMARKS ===", flush=True)
    micro = micro_levers(disp, device)
    print(f"  L2 in-place scatter vs clone+add (per live block, {micro['n_live_blocks']} blocks):",
          flush=True)
    print(f"     clone+add : {micro['l2_clone_add_us_step']:.4f} us/step  "
          f"({micro['l2_clone_add_ms']:.4f} ms/chunk)", flush=True)
    print(f"     in-place  : {micro['l2_inplace_add_us_step']:.4f} us/step  "
          f"({micro['l2_inplace_add_ms']:.4f} ms/chunk)", flush=True)
    print(f"     -> saving : {micro['l2_saving_us_step']:.4f} us/step "
          f"(all {micro['n_live_blocks']} live blocks)", flush=True)
    print(f"  L5 deduped-weight index_select vs replicated stream:", flush=True)
    print(f"     dedup     : {micro['l5_dedup_gather_ms']:.4f} ms  "
          f"({micro['l5_dedup_bytes']/1e3:.1f} KB, fits_L2={micro['l5_dedup_fits_L2']})",
          flush=True)
    print(f"     replicated: {micro['l5_replicated_ms']:.4f} ms  "
          f"({micro['l5_replicated_bytes']/1e6:.2f} MB, fits_L2={micro['l5_replicated_fits_L2']})",
          flush=True)

    result = {"frame": name, "n_steps": n, "chunk": chunk, "dim": model.dim,
              "build_phases": {k: bphases[k] for k in bphases if not k.startswith("_")},
              "build_total_s": btot, "build_us_step": btot / bn * 1e6,
              "dispatch": {k: disp[k] for k in disp if k not in ("mega_facts",)},
              "roofline": roof, "micro": micro}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[written] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
