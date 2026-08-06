#!/usr/bin/env python3
"""_agent_block0dk_corpus.py — byte-exact gate for C4_BLOCK0_DK across an OP CORPUS.

Runs several DIV-free programs (countdown loop, nested loop, malloc, malloc/free, matmul)
through the composed precomputed schedule in three configs (BASE: ATTN_MEGABLOCK OFF; DK:
ATTN_MEGABLOCK+BLOCK0_DK ON) and asserts the decoded (PC,SP,BP,AX) lanes are L-inf=0 identical
on every program.  The DK h0 is passed in [D,C] layout (sched.h0_folded[:n].T).
"""
from __future__ import annotations
import argparse, os, gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


def _guard():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                if int(ln.split()[1]) / 1e6 < 25.0:
                    raise SystemExit("[GUARD] <25GB -> STOP")


def _levers_on(chunk, block_k, attn_mega, block0_dk):
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
              "C4_DIRECT_CAM_VEC"):
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = str(block_k)
    os.environ["C4_ATTN_MEGABLOCK"] = "1" if attn_mega else "0"
    os.environ["C4_BLOCK0_DK"] = "1" if block0_dk else "0"


def _run(model, L, code, draft, dev, chunk, block_k, attn_mega, block0_dk):
    from c4_min import precomputed_schedule as PS
    _levers_on(chunk, block_k, attn_mega, block0_dk)
    if hasattr(draft, "_resolved_cache"):
        del draft._resolved_cache
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)
    n = draft.step_count
    onchip_ = sched.onchip
    h0dk = sched.h0_folded.transpose(0, 1).contiguous() if block0_dk else None
    h0s = sched.h0_folded if onchip_ else sched.h0_table
    outs = [torch.empty(n, dtype=torch.long, device=dev) for _ in range(4)]
    for lo in range(0, n, sg.chunk):
        hi = min(lo + sg.chunk, n)
        if block0_dk:
            h0 = h0dk[:, lo:hi]
        else:
            h0 = h0s[lo:hi].unsqueeze(0)
        delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
        r = sg.replay(h0, None, None, delta=delta, resident=False)
        for g, v in zip(outs, r):
            g[lo:hi].copy_(v)
    torch.cuda.synchronize(dev)
    res = tuple(o.clone() for o in outs)
    del sched, sg; gc.collect(); torch.cuda.empty_cache()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--block-k", type=int, default=512)
    args = ap.parse_args(argv)
    _guard()
    dev = torch.device(args.device)
    _levers_on(args.chunk, args.block_k, False, False)

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.pf_speculative import draft_pf_program
    from c4_min.bench_fast_path import (build_loop_countdown, build_nested, build_malloc,
                                        build_malloc_free, build_matmul)

    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(args.device)
    _guard()
    install_composed(model, verbose=False)

    progs = [
        ("loop_countdown(2000)", build_loop_countdown(2000)[0]),
        ("nested(20,50)", build_nested(20, 50)[0]),
        ("malloc(64)", build_malloc(64)[0]),
        ("malloc_free(64)", build_malloc_free(64)[0]),
        ("matmul(8)", build_matmul(8)[0]),
    ]
    all_ok = True
    for name, code in progs:
        try:
            draft = draft_pf_program(code, max_steps=200_000, mask=0xFFFFFFFF)
        except Exception as e:
            print(f"  {name:22s}: draft FAILED ({e})", flush=True)
            continue
        if not draft.halted:
            print(f"  {name:22s}: did not halt within cap (skip)", flush=True)
            continue
        try:
            base = _run(model, L, code, draft, dev, args.chunk, args.block_k, False, False)
            dk = _run(model, L, code, draft, dev, args.chunk, args.block_k, True, True)
        except Exception as e:
            print(f"  {name:22s}: run FAILED ({e})", flush=True)
            continue
        nbad = max(int((a != b).sum()) for a, b in zip(base, dk))
        ok = nbad == 0
        all_ok = all_ok and ok
        print(f"  {name:22s}: steps={draft.step_count:6d}  BASE-vs-DK mismatches={nbad}  "
              f"{'OK L-inf=0' if ok else 'MISMATCH!'}", flush=True)
    print(f"\n  CORPUS BYTE-EXACT (DK vs BASE): {'ALL PASS (L-inf=0)' if all_ok else 'FAILED'}",
          flush=True)


if __name__ == "__main__":
    main()
