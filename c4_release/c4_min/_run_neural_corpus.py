"""One-shot neural-corpus runner (dev harness, not a test).

Runs a chosen set of libprog corpus programs THROUGH THE TRANSFORMER, one per
process invocation (or several sharing one grown streaming model), and prints a
per-program PASS/FAIL + wall-clock line.  Memory-safe: OMP_NUM_THREADS=4, ONE
streaming model (~4 GB), KV eviction on, gc between runs.

Usage:
    OMP_NUM_THREADS=4 python -m c4_min._run_neural_corpus printf_str printf_hex
    OMP_NUM_THREADS=4 python -m c4_min._run_neural_corpus --all
"""
from __future__ import annotations

import gc
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

from c4_min import libprog_corpus as C


def main(argv):
    if not argv or argv[0] == "--all":
        names = [e.name for e in C.CORPUS]
    else:
        names = argv
    goldens = C.load_goldens()
    npass = 0
    for name in names:
        e = C.CORPUS_BY_NAME[name]
        t0 = time.time()
        try:
            got = C.run_model(e, shared=True)
            dt = time.time() - t0
            want = goldens[name].decode("latin-1")
            ok = got == want
            npass += int(ok)
            print(f"{name:<20} {'PASS' if ok else 'FAIL'}  {dt:7.1f}s  "
                  f"got={got!r}", flush=True)
            if not ok:
                print(f"{'':<20}                 want={want!r}", flush=True)
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            import traceback
            traceback.print_exc()
            print(f"{name:<20} ERROR  {dt:7.1f}s  {exc!r}", flush=True)
        gc.collect()
    print(f"\n{npass}/{len(names)} neural PASS", flush=True)
    return 0 if npass == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
