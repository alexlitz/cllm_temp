"""Stage-1/2 validator for the extended C90 case set (NO transformer).

Checks, for every case in ``cases_ext.CASES``:
  * gcc reference : gcc-15 -std=c90 compiles it, run exit code
  * compiler      : src.compiler.compile_c parses it (in-subset)
  * native ./c4   : ``native_c4.run`` (faithful full-word c4 VM) final AX & 0xFF

The native c4 VM is the authoritative reference the transformer is compared to;
gcc is an INDEPENDENT cross-check.  Where native-c4 and gcc disagree ONLY because
of the c4 ``sizeof(int)==8`` / 8-byte-int-cell semantics (vs x86 4-byte int), we
tag it ``c4-vs-x86`` (expected), not a failure.  Any other native!=gcc is a real
bug to fix before the transformer battery runs.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # c4_release/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cases_ext import CASES  # noqa: E402
from c4_min import isa  # noqa: E402
from src.compiler import compile_c  # noqa: E402
import native_c4  # noqa: E402

GCC = os.environ.get("C90_GCC", "gcc-15")
DATA_BASE = 65536          # compiler's fixed data segment base


def _compile(src):
    # compile_c auto-links the stdlib .c4 (malloc/free/memset) when referenced.
    words, data = compile_c(src)
    code = []
    for w in words:
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append(isa.Instr(op, imm))
    return code, data, DATA_BASE


def _gcc_exit(src):
    with tempfile.TemporaryDirectory() as td:
        cpath = os.path.join(td, "t.c")
        epath = os.path.join(td, "t")
        with open(cpath, "w") as f:
            f.write(src + "\n")
        r = subprocess.run([GCC, "-std=c90", "-w", "-O0", cpath, "-o", epath],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, f"gcc-compile-error: {r.stderr.strip()[:120]}"
        run = subprocess.run([epath], capture_output=True)
        return run.returncode, None


# cases where native-c4 and gcc-on-x86 legitimately differ (c4 sizeof(int)=8)
C4_X86_GAP = {"sizeof_int"}


def main():
    print(f"gcc = {GCC}   cases = {len(CASES)}")
    n_gcc = n_compile = n_native = n_native_gcc = 0
    fails = []
    for name, cat, src, expect in CASES:
        gcc_rc, gerr = _gcc_exit(src)
        if gerr:
            fails.append((name, "gcc", gerr))
            print(f"  {name:22s} [{cat:9s}] GCC-ERR {gerr}")
            continue
        n_gcc += 1
        try:
            code, data, dbase = _compile(src)
            n_compile += 1
        except Exception as e:
            fails.append((name, "compile", f"{type(e).__name__}: {e}"))
            print(f"  {name:22s} [{cat:9s}] COMPILE-ERR {type(e).__name__}: {str(e)[:60]}")
            continue
        ax, steps = native_c4.run(code, data=data, data_base=dbase)
        nat = ax & 0xFF
        n_native += 1
        gap = name in C4_X86_GAP
        match = (nat == gcc_rc) or gap
        n_native_gcc += match
        if not match:
            fails.append((name, "value", f"gcc={gcc_rc} native={nat} steps={steps}"))
        tag = "OK" if match else "*** MISMATCH ***"
        if gap and nat != gcc_rc:
            tag = f"c4-vs-x86 (native={nat} gcc={gcc_rc})"
        print(f"  {name:22s} [{cat:9s}] gcc={gcc_rc:3} native={nat:3} steps={steps:6} {tag}")

    print(f"\ngcc-compiled  : {n_gcc}/{len(CASES)}")
    print(f"c4-compiled   : {n_compile}/{len(CASES)}")
    print(f"native ran    : {n_native}/{n_compile}")
    print(f"native==gcc   : {n_native_gcc}/{n_native}  (c4-vs-x86 gaps counted as OK)")
    if fails:
        print(f"\n{len(fails)} FAILS:")
        for nm, stage, why in fails:
            print(f"  {nm:22s} [{stage}] {why}")
        return 1
    print("\nALL GREEN (native-c4 == gcc for every case, modulo c4-vs-x86 sizeof)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
