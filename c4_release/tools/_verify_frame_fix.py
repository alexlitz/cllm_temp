"""One-shot neural verification of the frame-offset IMM_CLEAN fix.

Waits for a memory window (>= MIN_FREE_GB free), then runs, in ONE process so the
shared streaming lib model is built ONCE:
  1. malloc_printf THROUGH THE MODEL, byte-exact vs the gcc golden (the deliverable).
  2. the isolated runtime primitives (zfod/malloc/memset/memcmp) neural tests.
  3. the LEA frame-address sweep.
  4. an arith/func/loop pure-forward corpus sample (regression).

Each section prints PASS/FAIL; the process exits nonzero if any fail.  Streaming /
lean build only; OMP_NUM_THREADS=4.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import time
import gc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

MIN_FREE_GB = int(os.environ.get("C4_MIN_FREE_GB", "45"))
WAIT_MAX_S = int(os.environ.get("C4_WAIT_MAX_S", "5400"))


def _free_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // (1024 * 1024)
    return 0


def wait_for_mem():
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX_S:
        fg = _free_gb()
        if fg >= MIN_FREE_GB:
            print(f"[mem] {fg} GB free -> proceeding", flush=True)
            return True
        print(f"[mem] {fg} GB free (< {MIN_FREE_GB}); waiting ...", flush=True)
        time.sleep(30)
    print(f"[mem] timed out waiting for {MIN_FREE_GB} GB free", flush=True)
    return False


def main():
    if not wait_for_mem():
        return 2
    results = {}

    # -------- 1. malloc_printf byte-exact through the model (the deliverable) -----
    print("\n==== 1. malloc_printf THROUGH THE MODEL ====", flush=True)
    from c4_min import libprog_corpus as LC
    entry = [e for e in LC.CORPUS if e.name == "malloc_printf"][0]
    want = LC.golden_for(entry)
    t0 = time.time()
    got = LC.run_model(entry, shared=False).encode("latin-1")
    dt = time.time() - t0
    ok = got == want
    results["malloc_printf"] = ok
    print(f"golden: {want!r}\nmodel : {got!r}\nMATCH: {ok}  ({dt:.0f}s)", flush=True)
    gc.collect()

    # -------- 2. isolated runtime primitives (no regression) ---------------------
    print("\n==== 2. runtime primitives (zfod/malloc/memset/memcmp) ====", flush=True)
    import subprocess
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    prim = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x",
         "c4_min/test_nibble_runtime_neural.py"],
        cwd=os.path.dirname(_HERE), env=env, capture_output=True, text=True)
    print(prim.stdout[-2500:], flush=True)
    if prim.returncode != 0:
        print(prim.stderr[-1500:], flush=True)
    results["primitives"] = prim.returncode == 0
    gc.collect()

    # -------- 3. LEA frame-address sweep -----------------------------------------
    print("\n==== 3. LEA frame-address sweep ====", flush=True)
    sweep = subprocess.run(
        [sys.executable, "-m", "c4_release.tools._probe_lea_sweep"],
        cwd=os.path.dirname(os.path.dirname(_HERE)), env=env,
        capture_output=True, text=True)
    print(sweep.stdout[-1500:], flush=True)
    if sweep.returncode != 0:
        print(sweep.stderr[-1200:], flush=True)
    results["lea_sweep"] = sweep.returncode == 0
    gc.collect()

    # -------- 4. per-op-class regression (arith/func/loop/mem/frame) --------------
    print("\n==== 4. per-op-class regression (all 30 op classes) ====", flush=True)
    opreg = subprocess.run(
        [sys.executable, "-m", "c4_release.tools._probe_op_class_regress"],
        cwd=os.path.dirname(os.path.dirname(_HERE)), env=env,
        capture_output=True, text=True)
    print(opreg.stdout[-3000:], flush=True)
    if opreg.returncode != 0:
        print(opreg.stderr[-1200:], flush=True)
    results["op_class_regress"] = opreg.returncode == 0

    print("\n==== SUMMARY ====", flush=True)
    allok = True
    for k, v in results.items():
        print(f"  {k:16s}: {'PASS' if v else 'FAIL'}", flush=True)
        allok = allok and v
    print("ALL PASS:", allok, flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
