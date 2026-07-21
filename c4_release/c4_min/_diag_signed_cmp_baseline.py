"""DIAGNOSTIC probe: signed comparison on the pure-forward model.

Runs the three #673 f3 signed programs (f3_signed_cmp / f3_if_neg /
f3_count_down_neg) and compares the model's final AX against the SIGNED 32-bit
golden (``ref_interpret(mask=0xFFFFFFFF)``, which this change fixed to compare
LT/GT/LE/GE as two's-complement SIGNED — matching C4 / gcc).

The negatives are genuine 32-bit two's-complement values, so this must run under
``C4_VM_WIDTH32=1`` (the 8-bit fold keeps every value < 2^31, i.e. unsigned).

DEFAULT (``--fast``) uses the small cmp-only ``build_pure_forward_model``, which
shares the exact ``compile_cmp_compute`` + ``compile_cmp_signed_finalize`` gadget
with the production build — BUT its single-slot 8-bit frame CANNOT carry a 32-bit
value on the STACK, so the f3 programs (which push/pop 32-bit values) are NOT
meaningful there; use ``--full`` (the production ``build_pure_forward_complete_model``,
whose KV stack carries full 32-bit values) for an end-to-end verdict, or run the
faster, deterministic gadget unit test
``test_pure_forward.test_cmp_signed_gadget_two_complement``.

Run:
    OMP_NUM_THREADS=4 C4_VM_WIDTH32=1 PYTHONPATH=<repo-root> \
        python -m c4_min._diag_signed_cmp_baseline --full
"""
from __future__ import annotations

import os
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

from c4_min import isa
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete, ref_interpret,
)
from c4_min.nibble_pure_forward import build_pure_forward_model, run_pure_forward

WORD = 4  # the pure-forward ISA addresses the stack in 4-byte slots


def signed_ideal_trace(code, max_steps=512):
    """The SIGNED 32-bit golden.  Identical to ``ref_interpret(mask=0xFFFFFFFF)``,
    which we fixed to compare LT/GT/LE/GE as two's-complement SIGNED 32-bit (bit
    31) — matching C4 / gcc.  Kept as a thin wrapper so the diag reads clearly."""
    return ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF)


# --- the three #673 f3 signed programs ---------------------------------------
# The C4 ISA has NO signed IMM literal; a negative is produced by 0 - k.
# We compute -k as (IMM 0 ; PSH ; IMM k ; SUB)  => AX = 0 - k = (256-k)&0xFF.

def prog_f3_signed_cmp():
    """(-10) < 5  -> should be TRUE (1) with signed LT; unsigned gives 246<5 = 0.
    Build AX=-10, push it, load 5, LT."""
    return isa.assemble([
        ("IMM", 0), ("PSH", 0), ("IMM", 10), ("SUB", 0),  # AX = 0-10 = 246 (=-10)
        ("PSH", 0),                                        # push (-10)
        ("IMM", 5),                                        # AX = 5
        ("LT", 0),                                         # (-10) < 5 ?
        ("HALT", 0),
    ]), "(-10) < 5 == 1 (signed)"


def prog_f3_if_neg():
    """if (a < 0) return 1 else 0, with a = -3.  a<0 signed -> TRUE.
    AX=a; push a; IMM 0; GT? -> we want a<0 i.e. LT with rhs 0.
    Build AX=-3, push, IMM 0, LT -> (-3) < 0 signed = 1; unsigned 253<0 = 0."""
    return isa.assemble([
        ("IMM", 0), ("PSH", 0), ("IMM", 3), ("SUB", 0),   # AX = -3 (=253)
        ("PSH", 0),                                        # push (-3)
        ("IMM", 0),                                        # AX = 0
        ("LT", 0),                                         # (-3) < 0 ?
        ("HALT", 0),
    ]), "(-3) < 0 == 1 (signed)"


def prog_f3_count_down_neg():
    """while (i > -3) i-- ; starting i=2.  Signed: runs while i in {2,1,0,-1,-2},
    stops when i == -3.  Final AX (the loop var) should be -3 (=253).
    Unsigned: i>-3 is i > 253, so 2>253 is FALSE immediately -> loop body never
    runs, final i stays 2.

    Program (i in AX):
      0 IMM 2          ; i = 2
      1 PSH            ; [head] push i
      2 IMM -3 via ... ; but we need -3 as a literal operand.
    We assemble -3 into a memory-free operand by computing it fresh each pass is
    costly; simpler: keep the loop bound -3 on the stack UNDER i is awkward.
    Instead test the CONDITION directly for i=2 and i=-3:
      i=2  : (2)  > (-3)  signed = 1  (unsigned 2 > 253 = 0)   [loop continues]
      i=-3 : (-3) > (-3)  signed = 0  (equal)                  [loop stops]
    We run the single-condition form for i=2 (the first, divergent, iteration)."""
    # i=2 is pushed as STK; bound=-3 goes in AX; GT computes STK > AX = i > bound.
    # Signed: 2 > -3 == 1 (loop continues).  Unsigned: 2 > 253 == 0 (loop never
    # runs -> the countdown would exit immediately, wrong).
    return isa.assemble([
        ("IMM", 2), ("PSH", 0),                            # push i = 2 (STK)
        ("IMM", 0), ("PSH", 0), ("IMM", 3), ("SUB", 0),   # AX = -3 (bound)
        ("GT", 0),                                         # i(2) > bound(-3) ?
        ("HALT", 0),
    ]), "2 > (-3) == 1 (signed)"


def main():
    import sys
    full = "--full" in sys.argv
    print(f"C4_VM_WIDTH32={os.environ.get('C4_VM_WIDTH32','0')}", flush=True)
    if full:
        print("building PRODUCTION model "
              "(build_pure_forward_complete_model, code_size=12) ...", flush=True)
        model, L = build_pure_forward_complete_model(code_size=12)
        runner = lambda c: run_pure_forward_complete(model, L, c, max_steps=64)
    else:
        # The FAST cmp-only pure-forward model uses the SAME compile_cmp_compute
        # gadget as the production build (it is imported into the complete build's
        # block_specs), so the CMP verdict is byte-shared with production.
        print("building FAST cmp-only pure-forward model (code_size=12) ...",
              flush=True)
        model, L = build_pure_forward_model(
            code_size=12, include_memory=False, include_cmp=True,
            include_bitwise=False, include_muldiv=False)
        runner = lambda c: run_pure_forward(model, L, c, max_steps=64)
    print(f"model: dim={L.D} blocks={len(model.blocks)}\n", flush=True)
    if os.environ.get("C4_VM_WIDTH32", "0") != "1":
        print("NOTE: the model's signed sign bit is fixed at 2^31; genuine 32-bit\n"
              "negatives require C4_VM_WIDTH32=1 (8-bit fold keeps every value < 2^31\n"
              "-> unsigned, byte-identical to the pre-fix gadget).\n", flush=True)

    progs = [
        ("f3_signed_cmp", *prog_f3_signed_cmp()),
        ("f3_if_neg", *prog_f3_if_neg()),
        ("f3_count_down_neg", *prog_f3_count_down_neg()),
    ]
    hdr = (f"{'name':<20} {'want(signed)':>12} {'MODEL':>7} {'model==signed?':>15}")
    print(hdr); print("-" * len(hdr))
    for name, code, desc in progs:
        model_trace = runner(code)
        model_ax = model_trace[-1] if model_trace else None
        signed_ax = signed_ideal_trace(code)[-1]
        ok = "PASS" if model_ax == signed_ax else "FAIL"
        print(f"{name:<20} {signed_ax:>12} "
              f"{str(model_ax):>7} {ok:>15}   ({desc})")


if __name__ == "__main__":
    main()
