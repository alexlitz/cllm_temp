#!/usr/bin/env python3
"""_agent_zeroattn_verify.py — BYTE-EXACT + ZERO-ATTENTION-COMPUTE verify of the
C4_DEAD_BLOCK_FUSION lever composed with direct-CAM / direct-local.

Runs a battery through verify_blocks in THREE configs and asserts:
  (1) softmax baseline (no levers)         -> all_matched, reference output
  (2) full lookup compose WITHOUT fusion   -> all_matched, output == (1)
  (3) full lookup compose WITH   fusion    -> all_matched, output == (1),
                                              AND zero residual attention arithmetic
and confirms (3)'s decoded output is byte-identical to the softmax baseline (1).

LEAN STREAMING (C4_PF_CFM=1).  Run:
    cd c4_release
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_zeroattn_verify --device cuda:0
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch


def _mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return 1e9


def _mem_guard(where=""):
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[MEM-GUARD] {a:.1f}GB < 25GB ({where}) STOP")


def _count_attn_arith():
    """Return (patch_ctx, counters): monkeypatch softmax1 + banded einsum counters."""
    import c4_min.banded_local_attn as BLA
    import c4_min.blogspec_model as BM
    import c4_min.direct_cam_batched as DCB
    import c4_min.local_attention as LAT
    counters = {"softmax1": 0, "banded_einsum": 0}
    orig = {"einsum": torch.einsum, "sm": BM.softmax1}

    def _einsum(eq, *ops):
        if eq in ("bhqd,bhqwd->bhqw", "bhqw,bhqwd->bhqd"):
            counters["banded_einsum"] += 1
        return orig["einsum"](eq, *ops)

    def _sm(x, dim=-1):
        counters["softmax1"] += 1
        return orig["sm"](x, dim=dim)

    class _Ctx:
        def __enter__(self):
            torch.einsum = _einsum
            BM.softmax1 = _sm
            BLA.softmax1 = _sm
            if hasattr(DCB, "softmax1"):
                DCB.softmax1 = _sm
            if hasattr(LAT, "softmax1"):
                LAT.softmax1 = _sm
            return counters

        def __exit__(self, *a):
            torch.einsum = orig["einsum"]
            BM.softmax1 = orig["sm"]
            BLA.softmax1 = orig["sm"]
            if hasattr(DCB, "softmax1"):
                DCB.softmax1 = orig["sm"]
            if hasattr(LAT, "softmax1"):
                LAT.softmax1 = orig["sm"]
    return _Ctx(), counters


def _fresh_model(dev):
    """A FRESH model+local-attn install (the forward wraps are stateful, so each
    config gets a clean model)."""
    from .lib_neural import build_lib_model_streaming
    from .local_attention import install_local_attention
    model, L, _ = build_lib_model_streaming(code_size=192, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    if dev != "cpu":
        model = model.to(dev)
    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    return model, L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=64)
    a = ap.parse_args(argv)
    _mem_guard("startup")
    dev = a.device if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    from .pf_speculative import draft_pf_program, verify_blocks
    from .bench_fast_path import (build_loop_countdown, build_malloc,
                                  build_malloc_free, build_nested)

    programs = [
        ("loop_countdown", build_loop_countdown(10)[0]),
        ("nested_call", build_nested(3, 12)[0]),
        ("malloc_heap", build_malloc(16)[0]),
        ("malloc_free", build_malloc_free(12)[0]),
    ]

    LEVER_FLAGS = ["C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
                   "C4_BANDED_LOCAL_ATTN", "C4_DIRECT_CAM_LIVE_LOCAL",
                   "C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION"]

    def _set_flags(**kw):
        for f in LEVER_FLAGS:
            os.environ.pop(f, None)
        for f, v in kw.items():
            os.environ[f] = v

    def _run(model, L, code, draft):
        out = []
        stats = {}
        vr = verify_blocks(model, L, code, draft, block_steps=a.K, device=dev,
                           evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                           collect_out=out)
        return vr, out

    all_ok = True
    print(f"{'prog':16s} {'base_match':>10s} {'lookup_match':>12s} "
          f"{'fuse_match':>10s} {'out==base':>10s} {'sm(fuse)':>9s} {'band(fuse)':>10s}",
          flush=True)
    for name, code in programs:
        _mem_guard(f"prog {name}")
        draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
        assert draft.halted, f"{name}: draft did not halt"

        # (1) softmax baseline (NO levers) — the golden reference.
        _set_flags()
        m1, L1 = _fresh_model(dev)
        vr1, out1 = _run(m1, L1, code, draft)
        del m1

        # (2) full lookup compose WITHOUT dead-block-fusion.
        _set_flags(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
                   C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
                   C4_FROZEN_ROW_SKIP="1")
        m2, L2 = _fresh_model(dev)
        vr2, out2 = _run(m2, L2, code, draft)
        del m2

        # (3) full lookup compose WITH dead-block-fusion — count attn arithmetic.
        _set_flags(C4_DIRECT_CAM_BATCHED="1", C4_DIRECT_LOCAL_CAM="1",
                   C4_BANDED_LOCAL_ATTN="1", C4_DIRECT_CAM_LIVE_LOCAL="1",
                   C4_FROZEN_ROW_SKIP="1", C4_DEAD_BLOCK_FUSION="1")
        m3, L3 = _fresh_model(dev)
        ctx, counters = _count_attn_arith()
        with ctx:
            vr3, out3 = _run(m3, L3, code, draft)
        del m3

        base_match = vr1.all_matched
        lookup_match = vr2.all_matched and (out2 == out1)
        fuse_match = vr3.all_matched and (out3 == out1)
        out_eq = (out3 == out1)
        zero_attn = (counters["softmax1"] == 0 and counters["banded_einsum"] == 0)
        ok = base_match and lookup_match and fuse_match and out_eq and zero_attn
        all_ok = all_ok and ok
        print(f"{name:16s} {str(base_match):>10s} {str(lookup_match):>12s} "
              f"{str(fuse_match):>10s} {str(out_eq):>10s} "
              f"{counters['softmax1']:>9d} {counters['banded_einsum']:>10d} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        _set_flags()

    print(f"\n{'ALL BYTE-EXACT + ZERO-ATTN-COMPUTE' if all_ok else 'FAILURE'}",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
