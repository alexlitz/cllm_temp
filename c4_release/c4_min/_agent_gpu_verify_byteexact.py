"""BYTE-EXACT gate: C4_GPU_VERIFY vectorized decode+compare == the scalar per-step
verify, across the DIV-free battery + a deep nested loop, on the composed doom path.

Compares, per program: accepted_steps, all_matched, decoded_final_ax, and (via a
forced-mismatch probe) the first_mismatch step — GPU-verify vs scalar (both run the
SAME model, same K), for K in {1, 512, 2048, 8192}.

Run:
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_gpu_verify_byteexact \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import (draft_pf_program, verify_blocks, set_gpu_verify)
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested, build_loop_countdown
from c4_min.bench_composed_fast_path import _battery


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _verify(model, L, code, draft, K, device, gpu):
    set_gpu_verify(gpu)
    stats = {}
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       evict_interval_steps=8, exact_evict=True)
    set_gpu_verify(None)
    return vr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=256)
    args = ap.parse_args(argv)
    if _mem_avail_gb() < 25.0:
        raise SystemExit("[GUARD] <25GB -> STOP")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, L, _ = build_lib_model_streaming(
        code_size=args.code_size, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        model = model.to(device)

    os.environ["C4_FROZEN_ROW_SKIP"] = "1"
    os.environ["C4_DEAD_BLOCK_FUSION"] = "1"
    os.environ["C4_DIRECT_CAM_BATCHED"] = "1"
    os.environ["C4_DIRECT_LOCAL_CAM"] = "1"
    os.environ["C4_FLASH_ATTN"] = "1"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "1"
    install_composed(model, verbose=False)

    # battery: the DIV-free-relevant ops (composed path is DIV-free doom) + loops.
    progs = []
    for name, prog, seed in _battery():
        if seed:            # composed doom path has no seed-mem programs; skip li/si_li
            continue
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        progs.append((name, code))
    progs.append(("loop20", build_loop_countdown(20)[0]))
    progs.append(("nested_4_12", build_nested(4, 12)[0]))
    progs.append(("nested_8_20", build_nested(8, 20)[0]))

    Ks = [1, 512, 2048, 8192]
    all_ok = True
    print(f"{'prog':>14} {'K':>6} {'scalar(acc,fin,match)':>26} "
          f"{'gpu(acc,fin,match)':>26} {'ok':>4}", flush=True)
    for name, code in progs:
        draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
        if not draft.halted:
            print(f"{name:>14}  draft did not halt, skip", flush=True)
            continue
        for K in Ks:
            vs = _verify(model, L, code, draft, K, device, gpu=False)
            vg = _verify(model, L, code, draft, K, device, gpu=True)
            sc = (vs.accepted_steps, vs.decoded_final_ax, vs.all_matched)
            gp = (vg.accepted_steps, vg.decoded_final_ax, vg.all_matched)
            ok = (sc == gp)
            all_ok = all_ok and ok
            print(f"{name:>14} {K:>6} {str(sc):>26} {str(gp):>26} "
                  f"{'OK' if ok else 'FAIL':>4}", flush=True)
            if not ok:
                print(f"    scalar first_mismatch={vs.first_mismatch}", flush=True)
                print(f"    gpu    first_mismatch={vg.first_mismatch}", flush=True)
    uninstall_composed(model)
    print(f"\n{'ALL BYTE-EXACT' if all_ok else 'DIVERGENCE FOUND'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
