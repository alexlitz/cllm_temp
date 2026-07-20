"""Run the REAL malloc_printf corpus program THROUGH THE MODEL and check the
byte-exact stdout against the gcc golden.  This is the deliverable's pass gate.

Uses the SAME path as ``test_libprog_corpus.test_model_byte_exact`` (retarget ->
streaming lib model -> KV-cached driver with fileop marshalling + low stack).
Fast-ish: malloc_printf is ~113 VM steps (~a few s/step + one ~4 GB build).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))       # .../c4_release on path -> import c4_min

from c4_min import libprog_corpus as LC

NAME = os.environ.get("C4_PROBE_PROG", "malloc_printf")


def main():
    os.system("free -g | head -2")
    entry = [e for e in LC.CORPUS if e.name == NAME][0]
    want = LC.golden_for(entry)
    print(f"golden ({NAME}): {want!r}", flush=True)
    print("running THROUGH THE MODEL (streaming lib build + KV-cached driver) ...",
          flush=True)
    got = LC.run_model(entry, shared=False).encode("latin-1")
    os.system("free -g | head -2")
    print(f"\nmodel stdout: {got!r}", flush=True)
    print(f"golden      : {want!r}", flush=True)
    ok = got == want
    print("\nBYTE-EXACT MATCH:", ok, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
