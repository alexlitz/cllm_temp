"""Bounded per-step malloc_printf trace: run the retargeted bytecode through the
model for a bounded number of steps, comparing the model's decoded PC/SP/BP/AX at
each step against ref_interpret, and report the FIRST divergence.  Fast diagnostic
(does NOT do the full ~30-min file-marshalled run) so we can confirm the frame ops
(ENT/LEA at the malloc-return-pointer store) are correct with the IMM_CLEAN fix.

The first printf (pc=30) reads *p (the malloc'd/memset byte); on the base it read
0 (byte=0), with the fix it must read 72 ('H').  We watch the AX at the LI feeding
that printf.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from c4_min import isa
from c4_min import libprog_corpus as LC
from c4_min import nibble_pure_forward_complete as C

STEPS = int(os.environ.get("C4_BOUND_STEPS", "60"))


def main():
    os.system("free -g | head -2")
    entry = [e for e in LC.CORPUS if e.name == "malloc_printf"][0]
    raw, data = LC.compile_entry(entry)
    instrs = LC.retarget_to_neural_abi(raw)
    print("n instrs:", len(instrs), flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    print("building STREAMING model ...", flush=True)
    sparse, L, _ = build_lib_model_streaming(
        code_size=len(instrs) + 2, recurrent_divmod=True, addr32=True)
    print("built; dim", L.D, flush=True)
    os.system("free -g | head -2")

    from c4_min import nibble_filesys as FS
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem(dict(entry.files)),
        stdin=FS.InputKVStream(entry.stdin)))

    # ref AX (word-width, low stack) for a per-step sanity trace (the file-marshalled
    # PRTF bytes are what run_model checks; here we just watch AX at the LI feeding
    # the first printf — it must be 72 not 0).
    evict = os.environ.get("C4_EVICT", "1") not in ("0", "")
    print(f"evict={evict}", flush=True)
    with LC._low_stack_sp():
        ref = C.ref_interpret(instrs, max_steps=STEPS, mask=0xFFFFFFFF)
        with LC._install_fileop_marshalling():
            got = run_pure_forward_cached(
                sparse, L, instrs, max_steps=STEPS, mask=0xFFFFFFFF, verbose=True,
                fio=fio, data_seg=LC._bytes_to_seg(data),
                evict=evict, prune_interval=60)

    print("\nstdout so far:", repr(bytes(fio.runner.stdout)), flush=True)
    n = min(len(ref), len(got))
    first_div = None
    for i in range(n):
        if ref[i] != got[i]:
            first_div = i
            break
    print(f"\nref AX[:{n}] = {ref[:n]}", flush=True)
    print(f"got AX[:{n}] = {got[:n]}", flush=True)
    if first_div is None:
        print(f"\nNO AX DIVERGENCE in the first {n} steps.", flush=True)
    else:
        print(f"\nFIRST AX DIVERGENCE at step {first_div}: "
              f"ref={ref[first_div]} got={got[first_div]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
