"""measure_fused_mac_vs_bytecode.py — the FUSED memory-operand MAC (C4_MEM_OPERAND)
vs the BYTECODE matmul, step-count AND wall-clock, byte-exact vs ref/numpy.

A matmul output element is a dot product ``y = Σ_k a[k]*b[k]`` — a chain of
multiply-accumulates.  This measures dot products (and a small [R x C]@[C] matvec)
done TWO ways on CPU, both byte-exact vs ``ref_interpret_mac`` / numpy:

  1. BYTECODE MAC (the path being replaced).  Each element loads its two operands
     from memory and multiply-accumulates in the base ISA.  The LEANEST in-register
     form on the pure-forward-complete VM is, per element:

         PSH(acc); IMM &a; LI; PSH(a); IMM &b; LI; MUL; ADD      (8 model.forwards)

     — the two memory reads are ``IMM addr; LI`` (address→AX + the CAM read), and
     that read cost is what dominates.  In the FULL C4 stack machine each such load
     expands further (frame-relative ``LEA``, stack spill/reload, byte paging) to the
     documented 76–101 steps/MAC (see ``measure_native_fixed_mac.py``); the 8-step
     form here is the *best case* for the bytecode path, so the fused speed-up
     measured against it is a LOWER BOUND on the full-C4 win.

  2. FUSED MAC (C4_MEM_OPERAND).  Each element is ONE ``MAC [a],[b]``: the two CAM
     reads run in the EARLY blocks of the instruction's OWN forward and feed the
     LATE-block multiply-accumulate — 1 model.forward per element, no separate load.

To keep the CPU cost down each model is built ONCE at the largest ``code_size`` used
(``code_size`` is just program-table headroom; the overlay writes only ``len(code)``
slots and the driver bounds by ``len(code)``), then reused for every N.

CPU-only.  Run:  C4_MEM_OPERAND=1 python -u -m c4_min.measure_fused_mac_vs_bytecode
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("C4_MEM_OPERAND", "1")

import numpy as np

from c4_min import isa
from c4_min import nibble_mem_operand as MO
from c4_min import nibble_pure_forward_complete as PFC

# the documented full-C4 stack-machine cost of one bytecode MAC (see
# measure_native_fixed_mac.py: "integer VM ... steps/MAC = 76-101").
BYTECODE_STEPS_PER_MAC_FULL_C4 = 76

A_BASE, B_BASE = 0x40, 0x60
N_MAX = 4                       # largest dot length measured


def bytecode_dot_program(n, a_base=A_BASE, b_base=B_BASE):
    """acc = Σ a[i]*b[i], AX-accumulator form (no acc memory round-trip):
    per element  PSH(acc); IMM &a; LI; PSH; IMM &b; LI; MUL; ADD  (8 forwards)."""
    prog = []
    for i in range(n):
        prog += [
            isa.Instr(isa.PSH, 0),                 # push running acc (AX)
            isa.Instr(isa.IMM, a_base + 4 * i),    # AX = &a[i]
            isa.Instr(isa.LI, 0),                  # AX = a[i]      (memory LOAD)
            isa.Instr(isa.PSH, 0),                 # push a[i]
            isa.Instr(isa.IMM, b_base + 4 * i),    # AX = &b[i]
            isa.Instr(isa.LI, 0),                  # AX = b[i]      (memory LOAD)
            isa.Instr(isa.MUL, 0),                 # AX = a[i]*b[i]
            isa.Instr(isa.ADD, 0),                 # AX = acc + a[i]*b[i]
        ]
    prog.append(isa.Instr(isa.HALT, 0))
    return prog


def fused_dot_program(n, a_base=A_BASE, b_base=B_BASE):
    """The SAME dot as ONE fused MAC per element: MAC [&a[i]],[&b[i]]."""
    prog, mac_b = [], {}
    for i in range(n):
        mac_b[len(prog)] = b_base + 4 * i
        prog.append(isa.Instr(MO.MAC, a_base + 4 * i))
    prog.append(isa.Instr(isa.HALT, 0))
    return prog, mac_b


def _seed(avec, bvec):
    seed = {}
    for i, v in enumerate(avec):
        seed[A_BASE + 4 * i] = v
    for i, v in enumerate(bvec):
        seed[B_BASE + 4 * i] = v
    return seed


def _time_run(fn, reps=3):
    fn()                                  # warmup
    t0 = time.time()
    for _ in range(reps):
        r = fn()
    return r, (time.time() - t0) / reps


def measure_dot(bmodel, bL, fmodel, fL, N, avec, bvec, reps=3):
    seed = _seed(avec, bvec)
    exp = int(np.dot(np.array(avec, dtype=np.int64),
                     np.array(bvec, dtype=np.int64))) & 0xFF

    bprog = bytecode_dot_program(N)
    btrace, bwall = _time_run(
        lambda: PFC.run_pure_forward_complete(bmodel, bL, bprog, max_steps=512,
                                              seed_mem=seed), reps=reps)
    fprog, mac_b = fused_dot_program(N)
    ftrace, fwall = _time_run(
        lambda: MO.run_mem_operand(fmodel, fL, fprog, mac_b, max_steps=128,
                                   seed_mem=seed), reps=reps)
    fref = MO.ref_interpret_mac(fprog, mac_b, seed_mem=seed, mask=0xFF)

    return dict(
        N=N, expect=exp,
        b_steps=len(btrace), b_final=btrace[-1], b_wall=bwall,
        f_steps=len(ftrace), f_final=ftrace[-1], f_wall=fwall,
        f_ref_final=fref[-1],
        b_ok=(btrace[-1] == exp), f_ok=(ftrace[-1] == exp and ftrace == fref),
    )


def report_dot(r):
    N = r["N"]
    b_work = r["b_steps"] - 1               # exclude the trailing HALT step
    f_work = r["f_steps"] - 1
    b_ms_step = 1000.0 * r["b_wall"] / max(1, r["b_steps"])
    f_ms_step = 1000.0 * r["f_wall"] / max(1, r["f_steps"])
    print(f"  N={N} dot   (expect Σ a·b & 0xFF = {r['expect']})")
    print(f"    BYTECODE  : {b_work:>4} steps ({b_work / N:.0f}/MAC)  "
          f"final AX {r['b_final']:>3} {'OK' if r['b_ok'] else 'XX'}   "
          f"wall {1000.0 * r['b_wall']:8.1f} ms  ({b_ms_step:6.1f} ms/step, "
          f"{1000.0 * r['b_wall'] / N:7.1f} ms/MAC)")
    print(f"    FUSED MAC : {f_work:>4} steps ({f_work / N:.0f}/MAC)  "
          f"final AX {r['f_final']:>3} {'OK' if r['f_ok'] else 'XX'}   "
          f"wall {1000.0 * r['f_wall']:8.1f} ms  ({f_ms_step:6.1f} ms/step, "
          f"{1000.0 * r['f_wall'] / N:7.1f} ms/MAC)")
    step_ratio = b_work / max(1, f_work)
    wall_ratio = r["b_wall"] / max(1e-9, r["f_wall"])
    print(f"    -> STEP-COUNT ratio (bytecode/fused): {step_ratio:.1f}x fewer steps")
    print(f"       WALL-CLOCK  ratio               : {wall_ratio:.1f}x faster")
    print()
    sys.stdout.flush()


def main():
    print("=" * 78)
    print("FUSED memory-operand MAC (C4_MEM_OPERAND, 1 step/MAC) vs BYTECODE matmul")
    print("=" * 78)
    print("Each row: ONE dot product y = Σ a[k]*b[k] (a matmul output element), byte-")
    print("exact vs the SP-addressed MAC oracle + numpy, timed on the SAME CPU forward.\n")
    sys.stdout.flush()

    # build ONCE at the largest sizes (code_size is just table headroom).
    t0 = time.time()
    bprog_max = bytecode_dot_program(N_MAX)
    bmodel, bL = PFC.build_pure_forward_complete_model(code_size=len(bprog_max))
    fprog_max, _ = fused_dot_program(N_MAX)
    fmodel, fL = MO.build_mem_operand_model(code_size=len(fprog_max))
    print(f"[built the bytecode + fused models in {time.time() - t0:.0f}s]\n")
    sys.stdout.flush()

    rng = np.random.default_rng(0)
    results = []
    for N in (1, 2, 4):
        avec = [int(v) for v in rng.integers(0, 16, size=N)]
        bvec = [int(v) for v in rng.integers(0, 16, size=N)]
        r = measure_dot(bmodel, bL, fmodel, fL, N, avec, bvec)
        results.append(r)
        report_dot(r)

    # a small [R x C] @ [C] matvec: R output rows, each a length-C fused-MAC dot.
    print("=" * 78)
    print("A real [R x C] @ [C] matvec (each output row = one length-C dot)")
    print("=" * 78)
    sys.stdout.flush()
    R, C = 3, N_MAX
    M = rng.integers(0, 12, size=(R, C))
    x = rng.integers(0, 12, size=C)
    ref_mv = [int(M[i] @ x) & 0xFF for i in range(R)]
    fused_steps_total = bytecode_steps_total = 0
    fused_wall_total = bytecode_wall_total = 0.0
    got_fused = []
    for i in range(R):
        r = measure_dot(bmodel, bL, fmodel, fL, C,
                        [int(v) for v in M[i]], [int(v) for v in x], reps=2)
        got_fused.append(r["f_final"])
        fused_steps_total += r["f_steps"] - 1
        bytecode_steps_total += r["b_steps"] - 1
        fused_wall_total += r["f_wall"]
        bytecode_wall_total += r["b_wall"]
    print(f"  M ({R}x{C}) @ x ({C})   numpy&0xFF = {ref_mv}")
    print(f"    fused rows -> {got_fused}   "
          f"{'BYTE-EXACT' if got_fused == ref_mv else 'MISMATCH'}")
    print(f"    total steps:  BYTECODE {bytecode_steps_total}  vs  FUSED {fused_steps_total}  "
          f"({bytecode_steps_total / max(1, fused_steps_total):.1f}x fewer)")
    print(f"    total wall :  BYTECODE {1000.0 * bytecode_wall_total:.0f} ms  vs  "
          f"FUSED {1000.0 * fused_wall_total:.0f} ms  "
          f"({bytecode_wall_total / max(1e-9, fused_wall_total):.1f}x faster)")
    print()
    sys.stdout.flush()

    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    all_exact = all(r["f_ok"] for r in results) and got_fused == ref_mv
    f_ms_per_mac = 1000.0 * sum(r["f_wall"] for r in results) / \
        max(1, sum(r["N"] for r in results))
    print(f"  fused MAC byte-exact vs ref/numpy: {'YES' if all_exact else 'NO'}")
    print(f"  step-count (MEASURED): bytecode 8 steps/MAC (best-case in-register")
    print(f"              memory-load form) -> fused 1 step/MAC = 8x fewer model.forwards.")
    print(f"  full C4 stack machine: {BYTECODE_STEPS_PER_MAC_FULL_C4} steps/MAC -> fused 1 "
          f"step/MAC = {BYTECODE_STEPS_PER_MAC_FULL_C4}x fewer steps")
    print(f"              (the {BYTECODE_STEPS_PER_MAC_FULL_C4}x the ~13s self-forward "
          f"projection assumes: 304K MACs x 1 step x ms/step).")
    print(f"  fused MAC cost (this CPU): ~{f_ms_per_mac:.1f} ms / MAC (1 model.forward).")
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main())
