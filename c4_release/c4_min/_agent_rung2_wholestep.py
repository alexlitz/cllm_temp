#!/usr/bin/env python3
"""_agent_rung2_wholestep.py — RUNG 2 (C4_WHOLE_STEP_GRAPH): byte-exact + measure the
block-0 chunk-loop host-dispatch collapse (before vs after), K in {8192,65536,262144}.
Also composes RUNG 3 (C4_BLOCK0_FUSED_FFN).
"""
from __future__ import annotations
import argparse, os, time, collections, traceback
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

_orig_item = torch.Tensor.item
_tally = collections.Counter(); _tracing = [False]
def _traced_item(self):
    if _tracing[0]:
        fr = traceback.extract_stack(limit=4)[-2]
        _tally[f"{os.path.basename(fr.filename)}:{fr.lineno}"] += 1
    return _orig_item(self)
torch.Tensor.item = _traced_item


def _guard():
    if _mem_avail_gb() < 25.0:
        raise SystemExit("[GUARD] <25GB -> STOP")


def _run(model, L, code, draft, K, device, *, rung2, rung3):
    _levers_on(256)
    if rung2:
        os.environ["C4_WHOLE_STEP_GRAPH"] = "1"
    if rung3:
        os.environ["C4_BLOCK0_FUSED_FFN"] = "1"
    set_gpu_verify(True)
    stats = {}
    vr = verify_blocks(model, L, code, draft, block_steps=K, device=device, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, evict_interval_steps=8,
                       exact_evict=True)
    set_gpu_verify(None)
    _levers_off()
    os.environ.pop("C4_WHOLE_STEP_GRAPH", None)
    os.environ.pop("C4_BLOCK0_FUSED_FFN", None)
    return vr, stats


def byte_exact(model, L, device):
    print("=== RUNG 2+3 BYTE-EXACT (full stack + whole-step-graph + block0-ffn) ===",
          flush=True)
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
                # reference: full stack, rungs OFF
                vref, _ = _run(model, L, code, draft, K, device, rung2=False, rung3=False)
                # test: full stack + rungs 2+3 ON
                vt, _ = _run(model, L, code, draft, K, device, rung2=True, rung3=True)
                draft_exact = (vt.all_matched and vt.accepted_steps == draft.step_count
                               and vt.decoded_final_ax == draft.final_ax_masked)
                match_ref = (vt.accepted_steps == vref.accepted_steps
                             and vt.all_matched == vref.all_matched
                             and vt.decoded_final_ax == vref.decoded_final_ax)
                ok = draft_exact and match_ref
                all_ok = all_ok and ok
                print(f"  {name:>16} K={K:>6} steps={draft.step_count:>6} "
                      f"acc={vt.accepted_steps} m={vt.all_matched} "
                      f"{'OK' if ok else 'FAIL'}", flush=True)
                if not ok:
                    print(f"     ref=({vref.accepted_steps},{vref.all_matched},"
                          f"{vref.decoded_final_ax}) test=({vt.accepted_steps},"
                          f"{vt.all_matched},{vt.decoded_final_ax}) "
                          f"first_mismatch={vt.first_mismatch}", flush=True)
            _guard()
    finally:
        uninstall_composed(model)
    print(f"  -> {'ALL BYTE-EXACT (L-inf=0 vs full stack + vs draft)' if all_ok else 'DIVERGENCE'}",
          flush=True)
    return all_ok


def measure(model, L, device):
    print("\n=== RUNG 2 MEASURE: block-0 chunk-loop collapse, wall us/step ===", flush=True)
    code = build_nested(12, 28)[0]
    draft = draft_pf_program(code, max_steps=1_000_000, mask=0xFFFFFFFF)
    steps = draft.step_count
    DOOM = 6_890_000
    install_composed(model, verbose=False)
    try:
        for K in (8192, 65536, 262144):
            _guard()
            row = {}
            for label, r2, r3 in (("BASE(r2/r3 OFF)", False, False),
                                  ("RUNG2 ON", True, False),
                                  ("RUNG2+3 ON", True, True)):
                _levers_on(256)
                if r2: os.environ["C4_WHOLE_STEP_GRAPH"] = "1"
                if r3: os.environ["C4_BLOCK0_FUSED_FFN"] = "1"
                set_gpu_verify(True)
                _run_once = lambda: verify_blocks(model, L, code, draft, block_steps=K,
                    device=device, evict=True, mask=0xFFFFFFFF, fast=True,
                    evict_interval_steps=8, exact_evict=True)
                _run_once()   # warmup
                torch.cuda.synchronize(device)
                _tally.clear(); _tracing[0] = True
                t0 = time.perf_counter()
                vr = _run_once()
                torch.cuda.synchronize(device)
                wall = time.perf_counter() - t0
                _tracing[0] = False
                set_gpu_verify(None); _levers_off()
                os.environ.pop("C4_WHOLE_STEP_GRAPH", None)
                os.environ.pop("C4_BLOCK0_FUSED_FFN", None)
                syncs = sum(_tally.values())
                row[label] = (wall * 1e6 / steps, wall / steps, syncs, vr.all_matched)
            print(f"  K={K}:", flush=True)
            base = row["BASE(r2/r3 OFF)"][0]
            for label in ("BASE(r2/r3 OFF)", "RUNG2 ON", "RUNG2+3 ON"):
                us, sf, syncs, m = row[label]
                secf = DOOM * sf
                spd = base / us
                print(f"    {label:16s}: {us:8.1f} us/step  {spd:5.2f}x  "
                      f"frame={secf:8.1f}s  syncs={syncs}  matched={m}", flush=True)
    finally:
        uninstall_composed(model)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-byte-exact", action="store_true")
    args = ap.parse_args(argv)
    device = args.device
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(device)
    print(f"[built] mem={_mem_avail_gb():.1f}GB", flush=True)
    ok = True
    if not args.skip_byte_exact:
        ok = byte_exact(model, L, device)
    measure(model, L, device)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
