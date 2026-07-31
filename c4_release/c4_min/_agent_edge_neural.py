#!/usr/bin/env python3
"""Agent scratch: incremental NEURAL verifier for the edge corpus (prints each
case's byte-exact result immediately, flushed).  Additive/tooling — builds
nothing new, golden 069cc32f untouched.

Runs a curated subset (all critical char/IO cases + a per-cluster sample) through
``run_pure_forward_complete`` and prints PASS/FAIL vs the c4_min golden as it goes.

    OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/_agent_edge_neural.py [names...]
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

from c4_min.edge_corpus import generate_edge_cases
from c4_min.run_edge_ops import golden_value, neural_value
from c4_min.compact_alloc import build_compact_sparse_streaming

M32 = 0xFFFFFFFF

# Curated: every char/shift/io case + one representative per remaining cluster.
CURATED = [
    # char_rw: sign-extend battery + SC-trunc + char* walk
    "lc_signext_00", "lc_signext_7f", "lc_signext_80", "lc_signext_ff",
    "li_unsigned_80", "sc_trunc_1ff", "char_ptr_walk", "c_string_copy_scan",
    # char_shift: the doom-critical signed shifts
    "shr_char_neg1_by1", "shr_char_neg128_by1", "shr_char_pos_by3",
    "shr_uchar_ff_by1", "shl_0f_by4", "shl_01_by7", "shl_store_char_trunc",
    # char_arith: multi-byte carry/borrow/mul + digit format
    "char_add_7f_01", "char_sub_neg", "digit_format", "add_carry_out",
    "sub_borrow_cascade", "mul_ff_ff", "char_mul_int",
    # io
    "putchar_seq", "read_stdin_lc0", "getchar_stdin", "prtf_d_c",
    # bitwise / cmp / branch
    "and_ff_80", "or_80_01", "xor_aa_ff", "ge_5_5", "ge_3_5",
    "bnz_taken", "bnz_not_taken",
    # ranges + runtime + recursion (one deeper each)
    "div_by_zero", "div_neg1_1_unsigned", "mul_ff_ff",
    "char_fill_loop", "rec_fact_5",
]


def main():
    names = sys.argv[1:] or CURATED
    cases = {c.name: c for c in generate_edge_cases()}
    sel = [cases[n] for n in names if n in cases]

    t = time.monotonic()
    print(f"[edge-neural] building model (code_size=48)...", flush=True)
    model, L, _ = build_compact_sparse_streaming(code_size=48,
                                                 compute_mode="dense_kernel")
    print(f"[edge-neural] dim={L.D} blocks={len(model.blocks)} "
          f"({time.monotonic()-t:.1f}s); {len(sel)} cases\n", flush=True)

    npass = nfail = 0
    for i, c in enumerate(sel):
        gax, gout = golden_value(c)
        t0 = time.monotonic()
        try:
            nax, nout = neural_value(c, model, L)
            ok = (nax == (c.expected & M32))
            if c.expected_stdout is not None and nout is not None:
                ok = ok and (nout == c.expected_stdout)
            extra = ""
            if c.expected_stdout is not None:
                extra = f" gold_out={gout!r} neu_out={nout!r}"
            tag = "PASS" if ok else "FAIL"
            npass += ok
            nfail += (not ok)
            print(f"[{i+1:2d}/{len(sel)}] {tag} {c.name:26s} "
                  f"golden={gax} neural={nax}{extra} "
                  f"({time.monotonic()-t0:.1f}s) :: {c.note}", flush=True)
        except Exception as exc:  # noqa: BLE001
            nfail += 1
            print(f"[{i+1:2d}/{len(sel)}] ERROR {c.name:26s} {exc!r}", flush=True)
    print(f"\n[edge-neural] {npass} PASS / {nfail} FAIL of {len(sel)}", flush=True)


if __name__ == "__main__":
    main()
