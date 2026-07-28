#!/usr/bin/env python3
"""_agent_run_programs.py — run WHOLE programs end-to-end through the SPARSE native
C runtime (block stack in C, embed/overlay/IO in Python) and check byte-exact
visible output vs the Python reference, + per-program wall.

Programs (self-contained, PRTF visible output, no stdin needed):
  * ECHO   — print a fixed baked string (printf-of-literal): the simplest visible
             end-to-end program on the full-ISA model.
  * QUINE  — the c4_min PRTF string quine (self-outputs its own source bytes).

Builds the compact full-ISA model once (for embed + layout), reuses the already
lowered .nblbin at <out_dir>/blockstack.nblbin.  Args: out_dir [which]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from c4_min import isa
from c4_min import compact_alloc as CA
from c4_min import nibble_pure_forward_complete as PFC
from c4_min.selfhost._agent_run_program_c import CServe, run_program_c


def _echo_prog(text: bytes):
    """A fixed-string printer: for each byte b, IMM b ; PRTF.  Uses only IMM/PRTF/
    HALT — the analogue of `printf("....")`.  Returns (code, expected_bytes)."""
    prog = []
    for b in text:
        prog.append(("IMM", int(b)))
        prog.append(("PRTF", 0))
    prog.append(("HALT", 0))
    return isa.assemble(prog), list(text)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse"
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    binp = os.path.join(out_dir, "blockstack.nblbin")
    # RT_EXE=prog uses the SELF-CONTAINED binary (embedded model; binp arg ignored).
    sparse_exe = os.path.join(out_dir, os.environ.get("RT_EXE", "rt_sparse"))
    assert os.path.exists(binp) and os.path.exists(sparse_exe)
    print(f"runtime: {sparse_exe}")

    print("building compact full-ISA model (embed + layout)...", flush=True)
    t0 = time.time()
    model, L, _ = CA.build_compact_pure_forward_model(code_size=48)
    model.eval()
    embed = model.embed.detach().numpy().astype(np.float32)
    print(f"  built in {time.time()-t0:.1f}s  D={L.D} blocks={len(model.blocks)}")

    results = []

    # ---- ECHO ----
    if which in ("all", "echo"):
        text = os.environ.get("ECHO_TEXT", "hi\n").encode()
        code, exp = _echo_prog(text)
        rt = CServe(sparse_exe, binp, L.D)
        out = []
        t0 = time.time()
        run_program_c(rt, L, code, embed, max_steps=len(code) + 5,
                      mask=0xFF, out=out)
        wall = time.time() - t0
        rt.close()
        got = bytes(out)
        ok = got == text
        print(f"\nECHO  ({len(code)} instrs, {rt.steps} steps): "
              f"{'BYTE-EXACT' if ok else 'MISMATCH'}")
        print(f"  expected: {text!r}")
        print(f"  got     : {got!r}")
        print(f"  wall {wall:.2f}s  ({rt.wall/max(rt.steps,1)*1000:.0f} ms/step C-runtime)")
        results.append(("echo", ok))

    # ---- QUINE ----
    if which in ("all", "quine"):
        from c4_min import quine_prtf as Q
        code, seed_mem, S = Q.build_quine()
        # widen memory horizon for the quine's data segment (as quine_bundle does)
        rt = CServe(sparse_exe, binp, L.D)
        out = []
        t0 = time.time()
        run_program_c(rt, L, code, embed, max_steps=4000, mask=0xFF,
                      out=out, seed_mem=seed_mem)
        wall = time.time() - t0
        rt.close()
        got = list(out)
        ok = got == S
        print(f"\nQUINE ({len(code)} instrs, {rt.steps} steps): "
              f"{'BYTE-EXACT (self-output == own source)' if ok else 'MISMATCH'}")
        print(f"  source bytes : {len(S)}")
        print(f"  printed bytes: {len(got)}")
        if not ok:
            # show first divergence
            for k in range(min(len(S), len(got))):
                if S[k] != got[k]:
                    print(f"  first diff at {k}: src={S[k]} got={got[k]}")
                    break
        print(f"  wall {wall:.2f}s  ({rt.wall/max(rt.steps,1)*1000:.0f} ms/step C-runtime)")
        results.append(("quine", ok))

    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"  {name:8s}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
