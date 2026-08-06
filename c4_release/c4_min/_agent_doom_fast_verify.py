#!/usr/bin/env python3
"""_agent_doom_fast_verify.py — prove C4_DOOM_FAST=1 == hand-set stack byte-exact.

Two modes on the SAME doom-scale program + draft:
  * ``--mode handset``   : sets the fast-doom member flags BY HAND (the canonical
                           _agent_block0dk_frame_fps env list), runs the composed
                           schedule replay, dumps decoded (PC,SP,BP,AX) lanes to --out.
  * ``--mode doomfast``  : sets ONLY ``C4_DOOM_FAST=1`` + calls
                           ``c4_min.doom_fast.expand_doom_fast()``, runs the SAME replay,
                           dumps lanes to --out.
  * ``--mode compare A B``: loads two dumps and asserts L-inf == 0 on every lane.

Run the two modes as separate processes (env is process-global), then compare.  L-inf=0
=> C4_DOOM_FAST reproduces the hand-listed stack BYTE-EXACT on a real doom slice.

LEAN (C4_PF_CFM=1).  Stops < 25 GB host RAM.
"""
from __future__ import annotations
import argparse, os, pickle, sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _guard():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                if int(ln.split()[1]) / 1e6 < 25.0:
                    raise SystemExit("[GUARD] <25GB -> STOP")


_HANDSET = [
    ("C4_DEAD_BLOCK_FUSION", "1"), ("C4_DIRECT_CAM_BATCHED", "1"),
    ("C4_DIRECT_LOCAL_CAM", "1"), ("C4_FLASH_ATTN", "1"),
    ("C4_BANDED_LOCAL_ATTN", "1"), ("C4_FUSED_MEGABLOCK", "1"),
    ("C4_DIRECT_CAM_VEC", "1"), ("C4_ONCHIP_RESIDUAL", "1"),
    ("C4_RESIDENT_BATCH", "1"), ("C4_PRECOMPUTED_SCHEDULE", "1"),
    ("C4_SCHED_FAST_BUILD", "1"), ("C4_SCHED_GPU_BUILD", "1"),
    ("C4_SCHED_CACHE_RESOLVED", "1"), ("C4_FFN_FUSED_HIDDEN", "1"),
    ("C4_FFN_WAVE_BATCH", "1"), ("C4_MEGABLOCK_BLOCK_K", "512"),
    ("C4_ATTN_MEGABLOCK", "1"), ("C4_BLOCK0_DK", "1"),
    # NOTE: C4_SCHED_PIPELINE is a BUILD-time double-buffer lever; the byte-exact
    # single-shot replay below does not exercise it and it does not affect the decoded
    # lanes.  It IS a C4_DOOM_FAST member (proven build-neutral separately); we set it in
    # both modes so the two env states are identical for this lane comparison too.
    ("C4_SCHED_PIPELINE", "1"),
]


def _apply_handset(chunk):
    for k, v in _HANDSET:
        os.environ[k] = v
    os.environ["C4_SCHED_CHUNK"] = str(chunk)


def _run_replay(chunk, block_k):
    import torch
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.bench_fast_path import build_nested
    from c4_min import precomputed_schedule as PS

    dev = torch.device("cuda:0")
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    _guard()
    with open("/tmp/_wf_draft_120_255.pkl", "rb") as fh:
        draft = pickle.load(fh)
    code = build_nested(120, 255)[0]
    install_composed(model, verbose=False)

    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)
    n = min(chunk, draft.step_count)
    onchip_ = sched.onchip
    block0_dk = os.environ.get("C4_BLOCK0_DK", "0") == "1"
    if block0_dk:
        h0 = sched.h0_folded[:n].transpose(0, 1).contiguous()   # [D, n]
    else:
        h0 = (sched.h0_folded if onchip_ else sched.h0_table)[:n].unsqueeze(0)
    delta = {b: t[:n] for b, t in sched.cam_delta_tables.items()} if onchip_ else None
    ing = None if onchip_ else sched.ing_table[:n].permute(1, 0, 2).unsqueeze(0)
    cam = None if onchip_ else {b: t[:n].permute(1, 0, 2).unsqueeze(0)
                                for b, t in sched.cam_tables.items()}
    pc, sp, bp, ax = sg.replay(h0, ing, cam, delta=delta, resident=False)
    torch.cuda.synchronize(dev)
    return {"PC": pc.cpu(), "SP": sp.cpu(), "BP": bp.cpu(), "AX": ax.cpu(),
            "n": int(n)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["handset", "doomfast", "compare"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--chunk", type=int, default=96000)
    ap.add_argument("--block-k", type=int, default=512)
    args = ap.parse_args(argv)

    if args.mode == "compare":
        import torch
        A = torch.load(args.a); B = torch.load(args.b)
        ok = True
        for lane in ("PC", "SP", "BP", "AX"):
            a, b = A[lane], B[lane]
            linf = int((a - b).abs().max()) if a.numel() else 0
            nbad = int((a != b).sum())
            print(f"  {lane}: n={a.numel()} mismatches={nbad} Linf={linf}", flush=True)
            ok = ok and (nbad == 0)
        print(f"\nBYTE-EXACT C4_DOOM_FAST==handset: {'YES (L-inf=0)' if ok else 'NO'}",
              flush=True)
        sys.exit(0 if ok else 1)

    _guard()
    if args.mode == "handset":
        _apply_handset(args.chunk)
        print("[handset] members set by hand", flush=True)
    else:  # doomfast
        os.environ["C4_DOOM_FAST"] = "1"
        os.environ.setdefault("C4_SCHED_CHUNK", str(args.chunk))
        from c4_min.doom_fast import expand_doom_fast
        applied = expand_doom_fast(verbose=True)
        print(f"[doomfast] C4_DOOM_FAST expanded {len(applied)} members", flush=True)

    res = _run_replay(args.chunk, args.block_k)
    import torch
    torch.save(res, args.out)
    print(f"[{args.mode}] n={res['n']} lanes dumped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
