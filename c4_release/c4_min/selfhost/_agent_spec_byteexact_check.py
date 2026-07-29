#!/usr/bin/env python3
"""_agent_spec_byteexact_check.py — prove the SPECULATIVE (batched-K draft-verify)
driver is BYTE-EXACT to the single-step windowed-incremental driver, on the same
model, in ONE process (build the model once).

For each program + each K in {1,4,8,16}: run the speculative driver (draft VM K-ahead,
one batched neural forward per round, teacher-forced decode of the K frame-end rows)
and check its visible output == the single-step reference == the pure-python VM.  Also
report the batched-forward COUNT (fewer forwards = the memory-amortization win) vs the
single-step forward count.

Tooling only; additive; golden unchanged.
"""
from __future__ import annotations
import os
import sys
import time

# keep torch (embed/overlay) to a few threads so it does not oversubscribe the cores
# the 23-thread C serve process uses (severe contention otherwise).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
torch.set_num_threads(4)

from c4_min import isa
from c4_min import compact_alloc as CA
from c4_min.selfhost.measure_incremental_speculative import (
    SpecServe, run_program_speculative, _emit_prog, build_binaries, FRAME_LEN)
from c4_min.selfhost.measure_incremental_windowed import Serve, run_program


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse"
    binp = os.path.join(out_dir, "blockstack.nblbin")
    exes = build_binaries(out_dir)
    incr_exe = exes.get("prog_incr_mt", exes.get("prog_incr_simd", exes["prog_incr_o2"]))
    os.environ["INCR_THREADS"] = os.environ.get("INCR_THREADS", "23")
    print(f"binary: {os.path.basename(incr_exe)}  INCR_THREADS={os.environ['INCR_THREADS']}")

    print("building compact full-ISA model ...", flush=True)
    t0 = time.time()
    model, L, _ = CA.build_compact_pure_forward_model(code_size=48)
    model.eval()
    embed = model.embed.detach().numpy().astype(np.float32)
    print(f"  built in {time.time()-t0:.1f}s  D={L.D} blocks={len(model.blocks)}")

    progs = [
        ("echo", *_emit_prog(b"hi\n")),
        ("yes3", *_emit_prog(b"y\n" * 3)),
    ]

    all_ok = True
    for name, code, exp in progs:
        # single-step reference (the proven windowed-incremental driver)
        rt = Serve(incr_exe, binp, True); ref = []
        run_program(rt, L, code, embed, max_steps=2000, out=ref); rt.close()
        ref_forwards = rt.steps
        ref_ok = ref == list(exp)
        print(f"\n[{name}] expected {bytes(exp)!r}")
        print(f"  single-step: {bytes(ref)!r}  {'OK' if ref_ok else 'MISMATCH'}  "
              f"({ref_forwards} forwards)")
        for K in (1, 4, 8, 16):
            srv = SpecServe(incr_exe, binp, True); got = []
            run_program_speculative(srv, L, code, embed, K=K, max_steps=2000, out=got)
            nfwd = srv.forwards
            srv.close()
            ok = got == ref and got == list(exp)
            all_ok = all_ok and ok
            print(f"  spec K={K:>2}: {bytes(got)!r}  "
                  f"{'BYTE-EXACT' if ok else 'MISMATCH'}  "
                  f"({nfwd} batched-forwards vs {ref_forwards} single-step)")
            if not ok:
                print(f"      ref  {bytes(ref)!r}")
                print(f"      spec {bytes(got)!r}")

    print(f"\n=== {'ALL BYTE-EXACT' if all_ok else 'FAILURES PRESENT'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
