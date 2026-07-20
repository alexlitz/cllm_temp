"""Per-op-class byte-identity regression for the IMM_CLEAN frame fix.

Runs one representative program per op class (arith/cmp/bitwise/mem/frame/func/
control) through the streaming model and asserts the per-step AX trace equals
``ref_interpret``.  This is the no-regression gate: the IMM_CLEAN block + the
frame-offset rule swap must leave every op class byte-identical (the non-frame ops
never read IMM_CLEAN; the frame ops (LEA/ENT/ADJ/JSR/LEV) reconstruct the exact
same offset).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from c4_min import isa
from c4_min import nibble_pure_forward_complete as C
from c4_min import _pf_op_class_progs as P

I = isa.Instr


def to_instrs(prog_tuples):
    prog = prog_tuples[0] if isinstance(prog_tuples, list) else prog_tuples
    out = []
    for name, imm in prog:
        out.append(I(getattr(isa, name), imm))
    return out


def main():
    os.system("free -g | head -2")
    from c4_min.lib_neural import build_lib_model_streaming
    print("building STREAMING model ...", flush=True)
    sparse, L, _ = build_lib_model_streaming(
        code_size=55, recurrent_divmod=True, addr32=True)
    print("built; dim", L.D, flush=True)
    os.system("free -g | head -2")

    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    progs = P.progs_by_class()

    passed = failed = 0
    fails = []
    for cls in P.ALL:
        code = to_instrs(progs[cls])
        ref = C.ref_interpret(code, max_steps=64, mask=0xFF)
        got = run_pure_forward_cached(sparse, L, code, max_steps=64, mask=0xFF,
                                      evict=True, prune_interval=60)
        ok = got == ref
        passed += int(ok)
        failed += int(not ok)
        if not ok:
            fails.append((cls, ref, got))
        print(f"  {cls:4s}: {'PASS' if ok else 'FAIL'}"
              + ("" if ok else f"  ref={ref} got={got}"), flush=True)
    print(f"\nOP-CLASS REGRESSION: {passed}/{passed+failed} PASS", flush=True)
    if fails:
        print("FAILS:", [f[0] for f in fails], flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
