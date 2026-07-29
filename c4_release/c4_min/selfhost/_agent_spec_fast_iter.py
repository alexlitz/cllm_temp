#!/usr/bin/env python3
"""_agent_spec_fast_iter.py — FAST iteration on the speculative driver: load the
cached embed + layout (no 120s model rebuild; the C binary IS the block stack), and
compare the speculative driver vs the single-step reference on tiny programs, with a
verbose per-round draft-vs-model trace to debug.
"""
from __future__ import annotations
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)

from c4_min import isa
from c4_min.selfhost.measure_incremental_speculative import (
    SpecServe, run_program_speculative, _emit_prog, build_binaries)
from c4_min.selfhost.measure_incremental_windowed import Serve, run_program

OUT = "/tmp/fullisa_sparse"
os.environ["INCR_THREADS"] = os.environ.get("INCR_THREADS", "20")


def main():
    embed = np.load(os.path.join(OUT, "embed.npy"))
    L = pickle.load(open(os.path.join(OUT, "layout.pkl"), "rb"))
    binp = os.path.join(OUT, "blockstack.nblbin")
    exes = build_binaries(OUT)
    exe = exes.get("prog_incr_mt", exes.get("prog_incr_simd", exes["prog_incr_o2"]))
    print(f"binary {os.path.basename(exe)}  D={embed.shape[1]}  L.D={L.D}", flush=True)

    progs = [("echo", b"hi\n"), ("yes3", b"y\n" * 3), ("hello", b"hello\n")]
    seeds = {name: None for name, _ in progs}
    # add the quine (loop + data-segment seed) — the real control-flow stress test
    try:
        from c4_min import quine_prtf as Q
        qcode, qseed, qout = Q.build_quine()
        progs.append(("quine", None))
        _QUINE = (qcode, list(qout), qseed)
    except Exception as e:
        _QUINE = None
        print(f"quine unavailable: {e}")
    all_ok = True
    for name, text in progs:
        if name == "quine":
            code, exp, seed_mem = _QUINE
        else:
            code, exp = _emit_prog(text); seed_mem = None
        mx = 6000 if name == "quine" else 400
        rt = Serve(exe, binp, True); ref = []
        run_program(rt, L, code, embed, max_steps=mx, out=ref, seed_mem=seed_mem)
        rt.close()
        rok = ref == list(exp)
        all_ok = all_ok and rok
        print(f"\n[{name}] single-step: {bytes(ref)[:40]!r}... ({rt.steps} forwards)  "
              f"{'OK' if rok else 'MISMATCH exp '+repr(bytes(exp)[:40])}", flush=True)
        for K in (1, 4, 8):
            srv = SpecServe(exe, binp, True); got = []
            run_program_speculative(srv, L, code, embed, K=K, max_steps=mx, out=got,
                                    verify=True, seed_mem=seed_mem)
            ok = got == ref and rok
            all_ok = all_ok and ok
            print(f"  spec K={K}: {bytes(got)!r} ({srv.forwards} batched-forwards vs "
                  f"{rt.steps} single-step)  {'BYTE-EXACT' if ok else 'MISMATCH'}",
                  flush=True)
            srv.close()
    print(f"\n=== {'ALL BYTE-EXACT' if all_ok else 'FAILURES'} ===")


if __name__ == "__main__":
    main()
