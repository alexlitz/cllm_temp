"""MINIMAL fast repro of the frame-local read-back alias bug (§Memory-CAM).

Hand-writes the smallest bytecode that reproduces the malloc_printf failure:

    * store pointer VALUE 0x20000 into frame-local ADDRESS 0xE4  (SI)
    * store a byte 72 into heap ADDRESS 0x20000                  (SC)
    * load the frame-local ADDRESS 0xE4                          (LI)  -> want 0x20000

The coincidence the bug hinges on: the pointer VALUE 0x20000 is simultaneously
the heap ADDRESS of the intervening SC.  A correct memory CAM keeps the two
apart (query 0xE4 must match the frame-local store's key, NOT the heap store's).

Runs ONLY a handful of VM steps through the REAL model.forward (the addr32
streaming lib model + KV-cached driver), so it iterates in seconds.  Prints the
per-step decoded AX and asserts the final LI reads 0x20000.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))       # .../c4_release on path

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min import libprog_corpus as LC

PTR_ADDR = 0xE4        # frame-local address of `p`
PTR_VAL = 0x20000      # the pointer value (= heap address)  <-- the alias
HEAP_BYTE = 72         # 'H'


def build_prog():
    # neural ABI: SI does addr=pop(); mem[addr]=ax.  SC likewise, byte.
    # 1) *0xE4 = 0x20000   : IMM 0xE4; PSH; IMM 0x20000; SI
    # 2) *0x20000 = 72     : IMM 0x20000; PSH; IMM 72; SC
    # 3) AX = *0xE4        : IMM 0xE4; LI            -> want 0x20000
    prog = [
        ("IMM", PTR_ADDR), ("PSH", 0), ("IMM", PTR_VAL), ("SI", 0),
        ("IMM", PTR_VAL), ("PSH", 0), ("IMM", HEAP_BYTE), ("SC", 0),
        ("IMM", PTR_ADDR), ("LI", 0),
        ("HALT", 0),
    ]
    return isa.assemble(prog)


def run(evict, prune_interval):
    code = build_prog()
    # code_size is padded well past the program so the default liveness-probe
    # battery (compiled from the C sources) fits its CODE_OP table.
    model, L, _ = build_lib_model_streaming(
        code_size=max(64, len(code) + 2), recurrent_divmod=True, addr32=True)
    with LC._low_stack_sp():
        trace = run_pure_forward_cached(
            model, L, code, max_steps=len(code) + 2, mask=0xFFFFFFFF,
            verbose=True, evict=evict, prune_interval=prune_interval)
    return trace


def main():
    os.system("free -g | head -2")
    evict = os.environ.get("C4_EVICT", "1") == "1"
    pi = int(os.environ.get("C4_PRUNE_INTERVAL", "60"))
    print(f"evict={evict} prune_interval={pi}", flush=True)
    trace = run(evict, pi)
    os.system("free -g | head -2")
    # The LI is the 10th instruction (index 9); its AX is trace[9] (per-step AX).
    li_ax = trace[9] if len(trace) > 9 else None
    print(f"\nfull per-step AX trace: {trace}", flush=True)
    if li_ax is not None:
        print(f"LI @ 0x{PTR_ADDR:X} read AX = {li_ax} (0x{li_ax:X})", flush=True)
    else:
        print("LI step missing", flush=True)
    ok = (li_ax == PTR_VAL)
    print(f"\nWANT 0x{PTR_VAL:X} ({PTR_VAL})  GOT {li_ax}  -> "
          f"{'PASS' if ok else 'FAIL (bug reproduced)'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
