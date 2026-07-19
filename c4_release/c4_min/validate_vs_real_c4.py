"""VALIDATION: model-produced bytecode vs the REAL c4 compiler (bundler/c4_compile.c).

This proves the model-runs-C claim against ground truth, not just against the
in-module reference interpreter. For each C expression it:

  1. runs the compiler-in-WEIGHTS transformer (BakedCompilerMachine) on the raw C
     source string — the model reads the source, EMITs bytecode, runs it (argmax /
     greedy decode, one recurrent transformer forward pass per VM step);
  2. compiles the SAME expression with the real c4 compiler
     (``bundler/c4_compile.c``, built with gcc) wrapped as ``int main(){return E;}``;
  3. strips c4's JSR/ENT prologue + trailing LEV framing and asserts the model's
     produced arithmetic core is **byte-identical** to c4's ``op | imm<<8`` words;
  4. checks the model's decoded result equals ``eval(E) & 0xFF``.

Usage (needs a built c4 at ``/tmp/c4c`` or pass --c4 <path>):

    gcc -w -fpermissive -static-libgcc -o /tmp/c4c bundler/c4_compile.c
    PYTHONPATH=<repo> python c4_min/validate_vs_real_c4.py

Exit 0 iff every expression's model core matches real c4 AND the result is correct.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

from c4_min import isa
from c4_min import nibble_compiler as C

EXPRS = ["2+3*4", "2*3+4", "1+2+3", "2*3*4", "4+5*6", "3*4+5", "9+8+7", "5*6*7"]


def real_c4_words(c4_bin: str, expr: str):
    """Compile ``int main(){return E;}`` with the real c4 compiler; return its words."""
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write("int main(){ return %s; }\n" % expr)
        path = f.name
    try:
        out = subprocess.check_output([c4_bin, path]).decode()
    finally:
        os.unlink(path)
    m = re.search(r"program_code\[\] = \{(.*?)\};", out, re.S)
    if not m:
        raise RuntimeError("could not parse c4 dump_bytecode output")
    return [int(x, 16) for x in re.findall(r"0x[0-9a-f]+", m.group(1))]


def c4_arith_core(words):
    """Strip c4's ``JSR main; ENT ...`` prologue and trailing ``LEV`` framing."""
    body = words[2:]                                   # drop JSR main, ENT
    while body and (body[-1] & 0xFF) == isa.LEV:       # drop trailing LEV(8)s
        body = body[:-1]
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", default="/tmp/c4c", help="path to the built c4 compiler")
    args = ap.parse_args()
    if not os.path.exists(args.c4):
        print(f"error: built c4 not found at {args.c4}\n"
              f"  build it: gcc -w -fpermissive -static-libgcc -o {args.c4} "
              f"bundler/c4_compile.c", file=sys.stderr)
        return 2

    comp = C.expr_compiler_bytecode()
    bcm = C.BakedCompilerMachine(comp, code_size=176, src_size=8,
                                 mem_size=8, stack_depth=8)
    print(f"compiler-in-weights transformer: D={bcm.L.D}, {bcm.n_blocks} blocks, "
          f"GEN_SIZE={bcm.L.GEN_SIZE}\n")
    print(f"{'C expr':>8} {'model result':>12} {'core vs real c4':>16}  result")
    print("-" * 56)
    allok = True
    for e in EXPRS:
        trace, words = bcm.run(e, return_code=True, max_steps=4000)
        O = C.COMPILER_OUTBASE
        model_words = [words[O + k] for k in range(8)]
        model_core = [w for w in model_words if (w & 0xFF) != isa.HALT]  # drop HALT
        real_core = c4_arith_core(real_c4_words(args.c4, e))
        core_ok = model_core == real_core
        res_ok = trace[-1] == (eval(e) & 0xFF)
        allok &= core_ok and res_ok
        print(f"{e:>8} {trace[-1]:>12} {'MATCH' if core_ok else 'DIFFER':>16}  "
              f"{'ok' if res_ok else 'WRONG'}")
        if not core_ok:
            print("   model :", [hex(w) for w in model_core])
            print("   real  :", [hex(w) for w in real_core])
    print("-" * 56)
    print("model-produced arith cores byte-identical to real c4 (all):", allok)

    # ---- FETCH-DEDUP: the variable-length LOOP compiler (sub-linear D) ----
    from c4_min import loop_compiler as LC
    from c4_min import nibble_fetch_dedup as FD
    chain_exprs = ["2+3+4+5", "2*3*4*5", "7+8+9", "1+2+3+4+5+6",
                   "3+3+3+3+3+3+3", "1+1+1+1+1+1+1+1+1+1"]
    prog = LC.chain_compiler_bytecode()
    lm = FD.LoopCompilerMachine(prog, out_size=48, src_size=32,
                                mem_size=8, stack_depth=16)
    print(f"\nDEDUP loop compiler-in-weights: D={lm.L.D} (baseline for the same "
          f"code_size ~ {3*(lm.L.CODE_SIZE)+148}), {lm.n_blocks} blocks, "
          f"gen_size={lm.L.GEN_SIZE}")
    print(f"{'C chain':>22} {'len':>4} {'result':>7} {'core vs real c4':>16}")
    print("-" * 56)
    loopok = True
    for e in chain_exprs:
        src = [ord(c) for c in e] + [0]
        trace, words = lm.run(src, max_steps=30000, return_code=True)
        produced = []
        for w in words:
            produced.append(w)
            if (w & 0xFF) == isa.HALT:
                break
        model_core = [w for w in produced if (w & 0xFF) != isa.HALT]
        real_core = c4_arith_core(real_c4_words(args.c4, e))
        core_ok = model_core == real_core
        res_ok = trace[-1] == (eval(e) & 0xFF)
        loopok &= core_ok and res_ok
        print(f"{e:>22} {len(e):>4} {trace[-1]:>7} "
              f"{('MATCH' if core_ok else 'DIFFER') + ('' if res_ok else '!'):>16}")
        if not core_ok:
            print("   model :", [hex(w) for w in model_core])
            print("   real  :", [hex(w) for w in real_core])
    print("-" * 56)
    print("dedup loop-compiler cores byte-identical to real c4 (all):", loopok)

    # ---- the full-c4-scale D wall (dedup vs baseline projection) ----
    base = C._assemble(C.expr_compiler_bytecode(outbase=0))
    filler = isa.Instr(isa.IMM, 0)
    gen4k = list(base) + [filler] * (4000 - len(base))
    _, L4k = FD.build_dedup_baked_compiler_step(gen4k, out_size=48, src_size=32,
                                                mem_size=8, stack_depth=16)
    base_proj = 3 * (L4k.CODE_SIZE) + 148
    print(f"\nfull-c4-scale (gen_size=4000): dedup D={L4k.D} vs baseline "
          f"D~{base_proj} ({base_proj / L4k.D:.0f}x smaller residual)")

    return 0 if (allok and loopok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
