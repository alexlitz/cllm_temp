#!/usr/bin/env python3
"""_agent_rung3_block0ffn.py — RUNG 3 (C4_BLOCK0_FUSED_FFN): verify byte-exact +
measure the block-0 dense-GEMM device drop when block-0's SwiGLU FFN is folded into
the fused-delta COO sparse kernel.
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks, set_gpu_verify
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.bench_fast_path import build_nested, build_loop_countdown
from c4_min.bench_composed_fast_path import _battery
from c4_min._agent_composed_floor import _levers_on, _levers_off, _mem_avail_gb


def _verify(model, L, code, draft, K, device, rung3):
    _levers_on(256)
    if rung3:
        os.environ["C4_BLOCK0_FUSED_FFN"] = "1"
    set_gpu_verify(True)
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device, evict=True,
                       mask=0xFFFFFFFF, fast=True, evict_interval_steps=8, exact_evict=True)
    set_gpu_verify(None)
    _levers_off(); os.environ.pop("C4_BLOCK0_FUSED_FFN", None)
    return vr


def byte_exact(model, L, device):
    print("=== RUNG 3 BYTE-EXACT (block0-fused-ffn ON vs certified draft) ===", flush=True)
    progs = []
    for name, prog, seed in _battery():
        if seed:
            continue
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        progs.append((name, code))
    progs.append(("loop_countdown60", build_loop_countdown(60)[0]))
    progs.append(("nested_6_16", build_nested(6, 16)[0]))
    progs.append(("nested_10_24", build_nested(10, 24)[0]))
    all_ok = True
    install_composed(model, verbose=False)
    try:
        for name, code in progs:
            draft = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF)
            if not draft.halted:
                continue
            for K in (512, 2048, 8192):
                vr = _verify(model, L, code, draft, K, device, rung3=True)
                ok = (vr.all_matched and vr.accepted_steps == draft.step_count
                      and vr.decoded_final_ax == draft.final_ax_masked)
                all_ok = all_ok and ok
                print(f"  {name:>16} K={K:>6} steps={draft.step_count:>6} "
                      f"acc={vr.accepted_steps} m={vr.all_matched} "
                      f"{'OK' if ok else 'FAIL'}", flush=True)
                if not ok:
                    print(f"     first_mismatch={vr.first_mismatch}", flush=True)
    finally:
        uninstall_composed(model)
    print(f"  -> {'ALL BYTE-EXACT' if all_ok else 'DIVERGENCE'}", flush=True)
    return all_ok


def measure(model, L, device):
    from torch.profiler import profile, ProfilerActivity
    print("\n=== RUNG 3 DEVICE GEMM: block-0 sgemm before vs after ===", flush=True)
    code = build_nested(12, 28)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    steps = draft.step_count
    install_composed(model, verbose=False)
    K = 8192
    def prof_gemm(rung3):
        _levers_on(256)
        if rung3:
            os.environ["C4_BLOCK0_FUSED_FFN"] = "1"
        set_gpu_verify(True)
        verify_blocks(model, L, code, draft, block_steps=K, device=device, evict=True,
                      mask=0xFFFFFFFF, fast=True, evict_interval_steps=8, exact_evict=True)
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as prof:
            verify_blocks(model, L, code, draft, block_steps=K, device=device, evict=True,
                          mask=0xFFFFFFFF, fast=True, evict_interval_steps=8, exact_evict=True)
            torch.cuda.synchronize(device)
        set_gpu_verify(None); _levers_off(); os.environ.pop("C4_BLOCK0_FUSED_FFN", None)
        evs = prof.key_averages()
        tot = sum(e.self_device_time_total for e in evs) or 1.0
        sgemm = sum(e.self_device_time_total for e in evs
                    if "sgemm" in e.key or "gemm" in e.key.lower())
        mega = sum(e.self_device_time_total for e in evs
                   if "delta_inplace" in e.key or "upgate_silu" in e.key)
        return tot, sgemm, mega
    try:
        t0, g0, m0 = prof_gemm(False)
        t1, g1, m1 = prof_gemm(True)
    finally:
        uninstall_composed(model)
    print(f"  RUNG3 OFF: device {t0/1e3:.1f}ms ({t0/steps:.1f}us/step)  "
          f"sgemm {g0/steps:.2f}us/step  mega {m0/steps:.2f}us/step", flush=True)
    print(f"  RUNG3 ON : device {t1/1e3:.1f}ms ({t1/steps:.1f}us/step)  "
          f"sgemm {g1/steps:.2f}us/step  mega {m1/steps:.2f}us/step", flush=True)
    print(f"  -> block-0 sgemm drop: {g0/steps:.2f} -> {g1/steps:.2f} us/step "
          f"(-{(g0-g1)/steps:.2f}us); total device {t0/steps:.1f} -> {t1/steps:.1f} us/step",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    device = args.device
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(device)
    print(f"[built] mem={_mem_avail_gb():.1f}GB", flush=True)
    ok = byte_exact(model, L, device)
    measure(model, L, device)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
