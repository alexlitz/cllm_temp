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
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
