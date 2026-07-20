"""Minimal, FAITHFUL repro of the frame-address IMM-leak (#648/#660).

Reproduces the malloc_printf failure pattern WITHOUT the ~23-min full program:
put a large heap-pointer literal (0x20000) in flight, store it to a 16-aligned
FRAME LOCAL (ENT + LEA + SI), then read it back (LEA + LI) and store the read-back
byte where PRTF would print it.  Compare the model's per-step AX trace against
``ref_interpret`` (the SAME low-window stack semantics the model runs), under the
``_low_stack_sp`` context the real corpus run uses.  Fast: ~2 min build + seconds.

Also (``--leak``) dumps the raw residual value of the leaky ``L.IMM`` scalar vs the
clean ``L.IMM_NIB``-reconstructed value at each LEA/ENT/ADJ step so the leak
magnitude is visible.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_C4REL = os.path.dirname(_HERE)                     # .../c4_release
sys.path.insert(0, _C4REL)

from c4_min import isa
from c4_min import nibble_pure_forward_complete as C

I = isa.Instr
ADJ = isa.ADJ if hasattr(isa, "ADJ") else 7


def build_prog():
    """malloc-pattern frame round-trip, with the corpus's LARGE-LITERAL DENSITY so
    the PC-one-hot leak actually accumulates past the round tolerance.

    malloc_printf carries ~43 big literals (data-seg + heap addresses) across 256
    positions; the leak on the fetched IMM scalar is the SUM of a sub-epsilon PC-one-
    hot residue times each nearby big literal, so a single big literal (as in the
    earlier passing repro) is NOT enough — we pad the tail with many big literals to
    reproduce the real >0.5 residue.  The frame ops (ENT/LEA at a 16-aligned local)
    run FIRST while all those big literals sit in CODE_IMM.

        0  ENT 4         ; frame with 4 locals -> BP; SP -= 16
        1  IMM 0x20000   ; big heap-pointer literal
        2  PSH           ; push ptr
        3  LEA -1        ; AX = &local0  (16-aligned frame byte)
        4  PSH           ; push addr
        5  IMM 72        ; value 'H'
        6  SC            ; *addr = 72
        7  LEA -1        ; AX = &local0 again (deep LEA, big imms in flight)
        8  LI            ; AX = *local0
        9  PRTF          ; print AX byte  (want 72)
        10 JMP end       ; skip the literal padding
        ...  IMM <big>    ; padding: many big literals in CODE_IMM (the leak source)
        end HALT
    """
    prog = [
        I(isa.ENT, 4),
        I(isa.IMM, 0x20000),
        I(isa.PSH, 0),
        I(isa.LEA, -1),
        I(isa.PSH, 0),
        I(isa.IMM, 72),
        I(isa.SC, 0),
        I(isa.LEA, -1),
        I(isa.LI, 0),
        I(isa.PRTF, 0),
    ]
    # JMP over the literal padding to a trailing HALT.
    n_pad = 40
    jmp_target = len(prog) + 1 + n_pad
    prog.append(I(isa.JMP, jmp_target))
    bigs = [0x20000, 0x10000, 0x10018, 0x10100, 0x20100, 0x20200, 0x10200, 0x20300]
    for k in range(n_pad):
        prog.append(I(isa.IMM, bigs[k % len(bigs)]))    # dead literals -> leak source
    prog.append(I(isa.HALT, 0))
    assert len(prog) - 1 == jmp_target, (len(prog), jmp_target)
    return prog


def main():
    code = build_prog()
    cs = 55
    os.system("free -g | head -2")
    print(f"building STREAMING model code_size={cs} ...", flush=True)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(
        code_size=cs, recurrent_divmod=True, addr32=True)
    print("model dim =", L.D, "n_blocks =", len(sparse.blocks), flush=True)
    os.system("free -g | head -2")

    from c4_min.libprog_corpus import _low_stack_sp
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    with _low_stack_sp():
        # faithful reference under the SAME low-window SP the model uses.
        out_ref = []
        ref_trace = C.ref_interpret(code, max_steps=64, mask=0xFF, out=out_ref)
        print("\nref AX trace:", ref_trace, flush=True)
        print("ref PRTF out:", out_ref, "->", bytes(out_ref), flush=True)

        model_out = []
        trace = run_pure_forward_cached(
            sparse, L, code, max_steps=64, verbose=True, mask=0xFF,
            evict=True, prune_interval=60, out=model_out)

    print("\nmodel AX trace:", trace, flush=True)
    print("model PRTF out:", model_out, "->", bytes(model_out), flush=True)
    ok_trace = trace == ref_trace
    ok_print = model_out == out_ref
    print("\nAX-TRACE MATCH:", ok_trace, flush=True)
    print("PRTF MATCH     :", ok_print, "(want byte 72 = 'H')", flush=True)
    return 0 if (ok_trace and ok_print) else 1


if __name__ == "__main__":
    sys.exit(main())
