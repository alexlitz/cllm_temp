"""MISMATCH/ROLLBACK gate: corrupt a draft frame's AX so the model rejects it, and
prove the GPU-verify path reports the SAME first_mismatch (step + accepted prefix)
as the scalar path.  Also verifies the accepted-prefix length is exact.

Run:
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_gpu_verify_mismatch \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks, set_gpu_verify
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_loop_countdown


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _verify(model, L, code, draft, K, device, gpu):
    set_gpu_verify(gpu)
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=True, mask=0xFFFFFFFF, fast=True,
                       evict_interval_steps=8, exact_evict=True)
    set_gpu_verify(None)
    return vr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if _mem_avail_gb() < 25.0:
        raise SystemExit("[GUARD] <25GB")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, L, _ = build_lib_model_streaming(
        code_size=256, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)
    for f in ("C4_FROZEN_ROW_SKIP", "C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED",
              "C4_DIRECT_LOCAL_CAM", "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN"):
        os.environ[f] = "1"
    install_composed(model, verbose=False)

    code = build_loop_countdown(20)[0]
    ok_all = True
    # Corrupt the AX of an interior frame at several positions -> the model rejects it.
    for corrupt_at in (5, 50, 130, 200):
        for K in (512, 2048):
            good = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
            if corrupt_at >= good.step_count - 1:
                continue
            bad = copy.deepcopy(good)
            bad.frames[corrupt_at] = dict(bad.frames[corrupt_at])
            bad.frames[corrupt_at]["ax"] = (bad.frames[corrupt_at]["ax"] ^ 0xAB) & 0xFFFFFFFF
            vs = _verify(model, L, code, bad, K, device, gpu=False)
            vg = _verify(model, L, code, bad, K, device, gpu=True)
            ms = (vs.accepted_steps, vs.all_matched,
                  vs.first_mismatch["step"] if vs.first_mismatch else None)
            mg = (vg.accepted_steps, vg.all_matched,
                  vg.first_mismatch["step"] if vg.first_mismatch else None)
            ok = (ms == mg)
            ok_all = ok_all and ok
            print(f"  corrupt@{corrupt_at:>3} K={K:>4}  scalar={ms}  gpu={mg}  "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                print(f"     scalar first_mismatch={vs.first_mismatch}")
                print(f"     gpu    first_mismatch={vg.first_mismatch}")
    uninstall_composed(model)
    print(f"\n{'MISMATCH PATH BYTE-EXACT' if ok_all else 'MISMATCH DIVERGENCE'}",
          flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
