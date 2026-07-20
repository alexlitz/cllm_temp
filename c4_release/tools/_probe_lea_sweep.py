"""LEA frame-address sweep (the base #648 fix's 9/9 gate) + a leak-stress variant.

For a range of BP low bytes, run ``LEA -1`` (and other small offsets) with a large
heap pointer in flight and assert the model's decoded frame byte equals
``(BP + 4*imm) & 0xFF``.  This is the regression gate for the LEA path — it must
still pass 9/9 with the IMM_CLEAN root fix in place.

Because the model runs a low-byte stack window, we drive LEA at a chosen BP by a
short prologue (IMM big ; PSH ; ENT n) that lands BP at a 16-aligned byte, then a
family of LEA offsets, comparing model AX to ``ref_interpret`` at each step.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from c4_min import isa
from c4_min import nibble_pure_forward_complete as C

I = isa.Instr


def sweep_prog(ent_n):
    # IMM big ; PSH ; ENT ent_n -> BP at SP_INIT-4-... ; then LEA offsets -4..+4.
    prog = [I(isa.IMM, 0x20000), I(isa.PSH, 0), I(isa.ENT, ent_n)]
    offs = [-4, -3, -2, -1, 1, 2, 3, 4]
    for o in offs:
        prog.append(I(isa.LEA, o))
    prog.append(I(isa.HALT, 0))
    return prog, offs


def main():
    os.system("free -g | head -2")
    from c4_min.lib_neural import build_lib_model_streaming
    print("building STREAMING model ...", flush=True)
    sparse, L, _ = build_lib_model_streaming(
        code_size=55, recurrent_divmod=True, addr32=True)
    print("built; dim", L.D, flush=True)
    os.system("free -g | head -2")

    from c4_min.libprog_corpus import _low_stack_sp
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    total = 0
    passed = 0
    with _low_stack_sp():
        # sweep the ENT size so BP lands on several distinct (incl. 16-aligned) bytes.
        for ent_n in range(0, 9):
            prog, offs = sweep_prog(ent_n)
            ref = C.ref_interpret(prog, max_steps=64, mask=0xFF)
            got = run_pure_forward_cached(sparse, L, prog, max_steps=64, mask=0xFF,
                                          evict=True, prune_interval=60)
            ok = got == ref
            total += 1
            passed += int(ok)
            # report the LEA slice
            lea_ref = ref[3:3 + len(offs)]
            lea_got = got[3:3 + len(offs)]
            print(f"ENT {ent_n}: {'PASS' if ok else 'FAIL'}  "
                  f"LEA ref={lea_ref} got={lea_got}", flush=True)
    print(f"\nLEA SWEEP: {passed}/{total} PASS", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
