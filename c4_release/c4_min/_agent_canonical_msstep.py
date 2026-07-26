"""AGENT (aa11): measure the REAL ms/step on THE canonical full-ISA model —
qwen_full_vm full ISA with the DEFAULT lean radix-16 divide (107 applied per DIV
step), via full_native_fast's conditional-sparsity incremental driver.

This REPLACES the 11-layer mem+cmp lean toy (2 ms) and the 238-block composed
number as the honest self-emulation host ms/step.

Wait-loop for a stable free GPU (>=18 GB, no align_timing/video_analysis, 60 s stable).
"""
import os, sys, time, subprocess
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_aa11")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from c4_min import full_native_fast as F
from c4_min import qwen_full_vm as Q


def _forbidden_running() -> bool:
    try:
        out = subprocess.check_output(["ps", "aux"], timeout=5).decode()
    except Exception:
        return False
    return any(k in out for k in ("align_timing", "video_analysis"))


def _gpu_free(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits", f"--id={idx}"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip().splitlines()[0]
        f, u = out.split(",")
        return float(f) / 1024.0, float(u)
    except Exception:
        return 0.0, 100.0


def wait_gpu(idx, min_free_gb=18.0, stable_s=60.0, timeout_s=7200.0):
    t0 = time.time(); stable_since = None
    while time.time() - t0 < timeout_s:
        if _forbidden_running():
            print("  [gpu-wait] align_timing/video_analysis running -> wait", flush=True)
            stable_since = None; time.sleep(10); continue
        free, util = _gpu_free(idx)
        if free >= min_free_gb:
            if stable_since is None:
                stable_since = time.time()
            held = time.time() - stable_since
            if held >= stable_s:
                print(f"  [gpu-wait] cuda:{idx} stable {held:.0f}s free={free:.1f}GB "
                      f"util={util:.0f}% -> proceed", flush=True)
                return
            print(f"  [gpu-wait] cuda:{idx} free={free:.1f}GB util={util:.0f}% "
                  f"stable {held:.0f}/{stable_s:.0f}s", flush=True)
        else:
            stable_since = None
            print(f"  [gpu-wait] cuda:{idx} free={free:.1f}GB < {min_free_gb} wait", flush=True)
        time.sleep(5)
    raise SystemExit("no stable GPU")


def main():
    idx = int(os.environ.get("AGENT_GPU", "0"))
    dev = f"cuda:{idx}"
    print("=" * 78, flush=True)
    print("CANONICAL full-ISA ms/step — lean radix-16 divide (107 applied per DIV step)",
          flush=True)
    print("=" * 78, flush=True)
    wait_gpu(idx)

    # confirm the canonical build's per-step depth (lean divide default).
    print("\n[depth] confirming canonical applied depth (lean divide default) ...", flush=True)
    print(f"        C4_DIV_LEAN default on = {Q._div_lean()}  "
          f"C4_DIV_LONGDIV = {Q._div_longdiv()}", flush=True)

    t0 = time.time()
    bundle = F.build_full_native_fast(device=dev, verbose=True)
    print(f"\nARTIFACT: {bundle.n_layers_stored} stored / {bundle.n_layers_applied} "
          f"applied (DIV-step depth), H={bundle.hidden_size}, I={bundle.intermediate}",
          flush=True)
    print(f"  dense FFN={bundle.dense_gb:.1f}GB  active block={bundle.active_gb*1024:.1f}MB "
          f"({bundle.active_total} units)  build={time.time()-t0:.1f}s", flush=True)

    print("\n[verify] full-ISA byte-exact (conditional == isa.interpret == dense):",
          flush=True)
    vs = F.verify_full_isa(bundle, verbose=True)
    print(f"[verify] {vs['pass']}/{vs['total']} byte-exact"
          + (f"  FAILS {vs['fails']}" if vs['fails'] else ""), flush=True)

    print("\n[measure] REAL ms/step on the canonical lean full ISA:", flush=True)
    ms = F.measure(bundle, batches=(1, 16, 64), n=20, warmup=5, verbose=True)

    print("\n" + "=" * 78, flush=True)
    print("HEADLINE (canonical lean full ISA, 107-applied):", flush=True)
    for label, d in ms.items():
        per = d["forward_us_per_step"]
        b1 = per.get(1); b64 = per.get(64)
        print(f"  {label:28s} driver={d['driver_ms_per_step']:7.2f} ms/step | "
              f"fwd B1={b1 if b1 is None else round(b1,1)}us "
              f"B64={b64 if b64 is None else round(b64,1)}us | opmix={d['op_mix']}",
              flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
