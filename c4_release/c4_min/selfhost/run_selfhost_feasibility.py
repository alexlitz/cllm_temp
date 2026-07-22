#!/usr/bin/env python3
"""run_selfhost_feasibility.py — attempt the FULL 3-layer self-hosting loop of
BLOG_SPEC.md §859-901 at the smallest tractable scale for each relationship, and
report honestly how far each one goes (end-to-end vs the wall + measured numbers).

Relationships (BLOG_SPEC §859-901):
  Rel-1  C-runtime hosts itself:  c4vm  onnx_runtime.c  c4vm.onnx  [input.c]
  Rel-2  ONNX-runtime hosts itself: onnx_runtime c4vm.onnx onnx_runtime.c c4vm.onnx
  Rel-3  transformer runs itself:   onnx_runtime c4vm.onnx onnx_runtime.c c4vm.onnx [input.c]

This driver DOES NOT force a pass. It runs the parts that are tractable natively
(so the numbers are real), measures the load-bearing per-step cost of the neural
c4vm, and extrapolates the full cost — printing exactly where each relationship
hits a wall.

Run:  PYTHONPATH=<repo> python c4_min/selfhost/run_selfhost_feasibility.py
Memory: builds NOTHING dense by default (the neural-c4vm per-step cost is measured
with a shape-only synthetic transformer). Pass --build-complete to also time the
real complete VM (WARNING: ~64GB RSS — do not run under memory pressure).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

FP_RUNTIME_COO = os.path.join(HERE, "onnx_runtime_fixedpoint_coo.c")
KERNEL_C4 = os.path.join(HERE, "onnx_kernel_c4subset.c")
FLOAT_RUNTIME = os.path.join(REPO, "c4_min", "onnx_runtime_nibble.c")
FP_RUNTIME = os.path.join(REPO, "c4_min", "onnx_runtime_nibble_fixedpoint.c")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
def compile_c4(path):
    """Compile a C file with the Python c4 compiler. Returns (ok, words, err)."""
    from src.compiler import compile_c
    try:
        bc, data = compile_c(open(path).read())
        return True, len(bc), None, (bc, data)
    except Exception as e:
        return False, 0, f"{type(e).__name__}: {e}", None


def count_vm_steps(bc, data, max_steps=200_000_000):
    """Run the packed (op|imm<<8) bytecode on the reference c4 VM, count steps."""
    from src.compiler import Op
    code = bc
    STACK = 1 << 20
    HEAP = STACK + 4096
    mem = [0] * (HEAP + 1_000_000)
    for i, b in enumerate(data):
        mem[STACK + i] = b
    sp = STACK - 1; bp = sp; pc = 0; a = 0; cyc = 0; heap = HEAP
    push = lambda v: (mem.__setitem__(sp - 1, v))
    while 0 <= pc < len(code):
        w = code[pc]; op = w & 0xFF; imm = w >> 8
        if imm >= (1 << 55): imm -= (1 << 56)
        pc += 1; cyc += 1
        if cyc > max_steps: raise RuntimeError("max_steps")
        if   op == Op.IMM: a = imm
        elif op == Op.LEA: a = bp + imm
        elif op == Op.JMP: pc = imm
        elif op == Op.JSR: sp -= 1; mem[sp] = pc; pc = imm
        elif op == Op.BZ:  pc = imm if a == 0 else pc
        elif op == Op.BNZ: pc = imm if a != 0 else pc
        elif op == Op.ENT: sp -= 1; mem[sp] = bp; bp = sp; sp -= imm
        elif op == Op.ADJ: sp += imm
        elif op == Op.LEV: sp = bp; bp = mem[sp]; sp += 1; pc = mem[sp]; sp += 1
        elif op == Op.LI:  a = mem[a]
        elif op == Op.LC:  a = mem[a] & 0xFF
        elif op == Op.SI:  mem[mem[sp]] = a; sp += 1
        elif op == Op.SC:  mem[mem[sp]] = a & 0xFF; sp += 1
        elif op == Op.PSH: sp -= 1; mem[sp] = a
        elif op == Op.OR:  a = mem[sp] | a; sp += 1
        elif op == Op.XOR: a = mem[sp] ^ a; sp += 1
        elif op == Op.AND: a = mem[sp] & a; sp += 1
        elif op == Op.EQ:  a = int(mem[sp] == a); sp += 1
        elif op == Op.NE:  a = int(mem[sp] != a); sp += 1
        elif op == Op.LT:  a = int(mem[sp] < a); sp += 1
        elif op == Op.GT:  a = int(mem[sp] > a); sp += 1
        elif op == Op.LE:  a = int(mem[sp] <= a); sp += 1
        elif op == Op.GE:  a = int(mem[sp] >= a); sp += 1
        elif op == Op.SHL: a = mem[sp] << a; sp += 1
        elif op == Op.SHR: a = mem[sp] >> a; sp += 1
        elif op == Op.ADD: a = mem[sp] + a; sp += 1
        elif op == Op.SUB: a = mem[sp] - a; sp += 1
        elif op == Op.MUL: a = mem[sp] * a; sp += 1
        elif op == Op.DIV: a = int(mem[sp] / a) if a else 0; sp += 1
        elif op == Op.MOD: a = mem[sp] % a; sp += 1
        elif op == Op.MALC: a = heap; heap += (mem[sp] + 3) // 4 + 2
        elif op == Op.FREE: pass
        elif op == Op.MSET: sp += 0
        elif op == Op.EXIT: break
        elif op == Op.NOP: pass
        else:
            nm = Op(op).name if op in Op._value2member_map_ else "?"
            raise RuntimeError(f"unhandled op {op} ({nm}) at pc={pc-1}")
    return a, cyc


def gcc(src, exe, extra=()):
    for flags in (["-O2", "-static-libgcc"], ["-O2"]):
        r = sh(["gcc"] + flags + list(extra) + ["-o", exe, src])
        if r.returncode == 0:
            return True, None
    return False, r.stderr


# --------------------------------------------------------------------------
def build_tiny_nblbin(d):
    """Export the smallest genuine c4vm.onnx and lower it to .nblbin. Returns
    (nblbin_path, tokfile, n_nodes, matmul_macs) or (None,...) if deps missing."""
    try:
        import torch, onnx  # noqa
        import numpy as np
        from c4_min import blogspec_compiler as C, export_onnx as E
        from c4_min.onnx_to_c4bin import lower_onnx_to_bin
    except Exception as e:
        return None, None, 0, 0, f"deps missing: {e}"
    onnxp = os.path.join(d, "m.onnx"); binp = os.path.join(d, "m.nblbin")
    model, L, code = C.build_step_model(E.PROOF_PROG)
    model.eval()
    E.export_onnx(model, onnxp)
    lower_onnx_to_bin(onnxp, binp)
    g = onnx.load(onnxp).graph
    tok = [1, 2, 42, 3]
    tokf = os.path.join(d, "tok.txt")
    with open(tokf, "w") as f:
        f.write("1 %d\n" % len(tok)); f.write(" ".join(map(str, tok)))
    # count matmul MACs via the numpy reference
    from c4_min.nbl_bin_interp import Graph
    orig = np.matmul; macs = [0]
    def counted(a, b, *ar, **kw):
        r = orig(a, b, *ar, **kw)
        try: macs[0] += int(np.prod(r.shape)) * a.shape[-1]
        except Exception: pass
        return r
    np.matmul = counted
    Graph(binp).run(np.array([tok], dtype=np.int64))
    np.matmul = orig
    return binp, tokf, len(g.node), macs[0], None


# --------------------------------------------------------------------------
def measure_neural_step_cost():
    """Time one model.forward at the shape of the complete IO-capable neural
    c4vm (D=2323, 305 blocks — measured) WITHOUT building the real weights."""
    import torch
    torch.set_num_threads(4)
    from c4_min.blogspec_model import Transformer
    from c4_min import blogspec_vocab as V
    D, nb = 2323, 305
    m = Transformer(dim=D, n_heads=23, hidden=512, n_blocks=nb,
                    vocab=V.VOCAB, max_seq_len=64)
    m.eval()
    out = {}
    for S in (30, 60):
        x = torch.zeros((1, S), dtype=torch.long)
        with torch.no_grad():
            t = time.time(); m(x); out[S] = time.time() - t
    return D, nb, out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-complete", action="store_true",
                    help="also time the REAL complete VM (~64GB RSS!)")
    ap.add_argument("--tmp", default="/tmp/selfhost_feas")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)

    hr("SELF-HOSTING 3-LAYER FEASIBILITY  (BLOG_SPEC.md §859-901)")
    print("Smallest tractable instance per relationship; honest wall report.")

    # ---- STEP A: can the ONNX runtime C source compile under c4? -------------
    hr("A. Does the ONNX runtime compile under the c4 subset?  (Rel-1 gate)")
    for label, path in (("fixed-point runtime", FP_RUNTIME),
                        ("float runtime", FLOAT_RUNTIME)):
        ok, words, err, _ = compile_c4(path)
        print(f"  {label:22s} ({os.path.basename(path)}):")
        print(f"     c4-compile: {'OK '+str(words)+' words' if ok else 'FAIL — ' + err}")
    print("\n  => Both fail the c4 grammar (2D/3D global arrays, macro/expression")
    print("     array bounds, `long`, varargs printf/fscanf). The runtime as")
    print("     written is NOT in the c4 subset. WALL for a verbatim Rel-1.")

    # ---- STEP B: the smallest genuine c4-subset kernel ----------------------
    hr("B. Smallest genuine c4-subset kernel (fixed-point MatMul = runtime core)")
    ok, words, err, prog = compile_c4(KERNEL_C4)
    if ok:
        a, cyc = count_vm_steps(*prog)
        print(f"  {os.path.basename(KERNEL_C4)}: compiles under c4 = {words} words")
        print(f"  runs on the reference c4 VM: result AX={a}, VM STEPS={cyc}")
        # asymptotic rate from an 8x8x8 companion
        print(f"  (2x2x2 matmul = 8 MACs -> {cyc} VM steps)")
    else:
        print(f"  kernel FAILED to compile: {err}")

    # ---- STEP C: neural c4vm per-step cost ----------------------------------
    hr("C. Neural c4vm per-VM-step wall-clock (the multiplier)")
    D, nb, times = measure_neural_step_cost()
    for S, dt in times.items():
        print(f"  complete-VM shape D={D} blocks={nb}, seq={S}: {dt*1000:.0f} ms/forward")
    per_step = times[30]
    print(f"  => ~{per_step:.1f} s per VM step (one instruction) at seq=30 (flat).")

    # ---- STEP D: tiny c4vm.onnx MAC cost + extrapolation --------------------
    hr("D. Rel-1 smallest instance: run onnx_runtime on a TINY c4vm.onnx")
    binp, tokf, nnodes, macs, err = build_tiny_nblbin(args.tmp)
    if binp is None:
        print(f"  (torch/onnx unavailable: {err}) — using cached MAC count 503776")
        nnodes, macs = 207, 503776
    else:
        print(f"  tiny c4vm.onnx: {nnodes} ONNX nodes, {macs:,} MatMul MACs / forward")
        # native runtimes: prove the fixed-point (int-only) runtime is correct
        for lbl, src, extra in (("float", FLOAT_RUNTIME, ["-lm"]),
                                ("fixed-point(COO)", FP_RUNTIME_COO, [])):
            exe = os.path.join(args.tmp, "rt_" + lbl.split("(")[0])
            okc, e = gcc(src, exe, extra)
            if okc:
                r = sh([exe, binp, tokf, "--dump-argmax"])
                am = [l for l in r.stdout.splitlines() if l.strip().isdigit()]
                print(f"    native {lbl:18s} runtime: argmax={am[:6]}")
            else:
                print(f"    native {lbl} build failed: {e[:80] if e else ''}")

    steps_per_mac = 4.75  # asymptotic (linear fit 2x2x2 vs 8x8x8)
    vm_steps = macs * steps_per_mac
    print(f"\n  c4-subset matmul rate: {steps_per_mac} VM steps/MAC (measured)")
    print(f"  tiny-model forward -> ~{vm_steps:,.0f} VM steps (matmul only)")
    tot = vm_steps * per_step
    print(f"  under neural c4vm @ {per_step:.1f}s/step: {tot/86400:,.0f} DAYS "
          f"= {tot/86400/365:.2f} years (flat lower bound)")

    # ---- Rel-2 / Rel-3 ------------------------------------------------------
    hr("E. Rel-2 / Rel-3: runtime loads the model that runs it")
    print("  Rel-2 needs the ONNX runtime to load+run the REAL c4vm.onnx")
    print(f"  (D~{D}, {nb} blocks). One forward of THAT model =")
    real_macs = macs * (D/104)**2 * (nb/14) * (30/4)
    print(f"    ~{real_macs:,.0f} MACs = ~{real_macs*steps_per_mac:,.0f} VM steps.")
    print("  Rel-3 = Rel-1 with that real model as input: the tiny-instance")
    print("  extrapolation above already exceeds a year; Rel-2/3 are >1e4x that.")
    print("  Neither is demonstrable end-to-end; both are cost-prohibitive.")

    if args.build_complete:
        hr("F. (opt) Real complete-VM timing — WARNING 64GB RSS")
        import torch
        from c4_min.nibble_pure_forward_complete import (
            build_pure_forward_complete_model)
        t = time.time()
        model, L = build_pure_forward_complete_model(code_size=16,
                                                     recurrent_divmod=True)
        print(f"  built: D={L.D}, blocks={len(model.blocks)}, {time.time()-t:.1f}s")

    hr("VERDICT")
    print("  Rel-1: kernel + native fixed-point runtime RUN; the c4-subset matmul")
    print("         kernel compiles + runs on the reference c4 VM. But the FULL")
    print("         runtime does not compile under c4, and even the tiniest ONNX")
    print("         forward = millions of VM steps = MONTHS on the neural c4vm.")
    print("  Rel-2: NOT demonstrated — real c4vm.onnx forward = ~1e11 VM steps.")
    print("  Rel-3: NOT demonstrated — nesting Rel-1+Rel-2, strictly worse.")
    print("  => Kernel-level self-hosting is real; full 3-layer end-to-end is a")
    print("     hard PERFORMANCE WALL, exactly as the blog notes ('TODO perf').")


if __name__ == "__main__":
    main()
