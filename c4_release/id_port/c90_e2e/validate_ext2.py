"""Stage-1/2 CPU validator + static transformer-divergence classifier for WAVE 2.

CPU-ONLY (task #839 constraint): gcc + the faithful full-word ``native_c4`` VM. NO
transformer / model build.  For every case in ``cases_ext2.CASES2`` it checks:

  * gcc reference : ``gcc-15 -std=c90`` compiles + runs -> exit code (ground truth)
  * c4 compiler   : ``src.compiler.compile_c`` parses it (in-subset)
  * native ./c4   : ``native_c4.run`` final AX & 0xFF (the authoritative c4 reference)

The pass criterion is BYTE-EXACT ``native == gcc`` (mod 256), except the documented
``c4-vs-x86`` sizeof gap.  It ALSO statically tags each case with the transformer
divergence class it would MOST LIKELY hit on a future GPU pass, using the SAME
heuristic as ``run_battery_flash.classify`` (read off the compiled c4 bytecode; the
transformer is NEVER run here).  ``ok`` == predicted byte-exact on the transformer.

Emits ``CONFORMANCE_MATRIX_EXT2.json`` (same result schema as the existing matrix,
minus the ``transformer``/``tf_exact`` fields which a GPU pass fills in later).
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # c4_release/
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cases_ext2 import CASES2, C4_X86_GAP2, SUBSET_LIMIT_CASES  # noqa: E402
from c4_min import isa  # noqa: E402
from src.compiler import compile_c  # noqa: E402
import native_c4  # noqa: E402

GCC = os.environ.get("C90_GCC", "gcc-15")
DATA_BASE = 65536


def _compile(src):
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


def classify_static(code_tf, cat):
    """Predicted transformer divergence class, read off the compiled bytecode.

    Mirrors ``run_battery_flash.classify`` but works from the c4 bytecode ALONE
    (no transformer run): data-segment loads, arg-passing calls, and wide
    ALU/cmp/shift ops are the known transformer risk surfaces.  Returns "ok"
    when none of those risk surfaces are present.
    """
    ops = {isa.NAMES.get(i.op, i.op) for i in code_tf}
    uses_data = any(i.op == isa.IMM and i.imm >= 65536 for i in code_tf)
    passes_args = any(i.op == isa.ADJ for i in code_tf)
    n_funcs = sum(1 for t in {i.imm for i in code_tf if i.op == isa.JSR}
                  if 0 <= t < len(code_tf) and code_tf[t].op == isa.ENT)
    if uses_data:
        return "data-segment-recall (global/string literal load)"
    if passes_args and n_funcs >= 1:
        return "call-frame/arg-passing"
    if cat == "cmp" or ops & {"EQ", "NE", "LT", "GT", "LE", "GE"}:
        return "cmp/EQ-decode"
    if ops & {"MUL", "DIV", "MOD"}:
        return "width/ALU-32bit"
    if ops & {"SHL", "SHR"}:
        return "shift-width"
    if ops & {"LI", "SI"}:
        return "CAM-recall (memory load/store)"
    return "ok"


def main():
    print(f"gcc = {GCC}   wave-2 cases = {len(CASES2)}  (+ {len(SUBSET_LIMIT_CASES)} subset-limit)")
    n_gcc = n_compile = n_native = n_native_gcc = 0
    fails = []
    results = []
    tf_pred = {}
    for name, cat, src, expect, tf_note in CASES2:
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
        gap = name in C4_X86_GAP2
        match = (nat == gcc_rc) or gap
        n_native_gcc += match
        # tf_note is the AUTHORED prediction; recompute statically as a cross-check.
        tf_static = classify_static(code, cat)
        tf_pred[tf_static] = tf_pred.get(tf_static, 0) + 1
        if not match:
            fails.append((name, "value", f"gcc={gcc_rc} native={nat} steps={steps}"))
        tag = "OK" if match else "*** MISMATCH ***"
        if gap and nat != gcc_rc:
            tag = f"c4-vs-x86 (native={nat} gcc={gcc_rc})"
        note = "" if tf_static == "ok" else f"  ~tf:{tf_static.split(' ')[0]}"
        print(f"  {name:22s} [{cat:9s}] gcc={gcc_rc:3} native={nat:3} steps={steps:6} {tag}{note}")
        results.append({
            "name": name, "category": cat, "expect": expect,
            "gcc": gcc_rc, "native": nat, "native_steps": steps,
            "native_ok": bool(match), "c4_vs_x86": bool(gap),
            "tf_predicted_class": tf_static, "tf_note_authored": tf_note,
            "steps_code": len(code),
        })

    print(f"\ngcc-compiled  : {n_gcc}/{len(CASES2)}")
    print(f"c4-compiled   : {n_compile}/{len(CASES2)}")
    print(f"native ran    : {n_native}/{n_compile}")
    print(f"native==gcc   : {n_native_gcc}/{n_native}  (c4-vs-x86 gaps counted as OK)")

    print("\npredicted transformer-divergence classes (for a FUTURE GPU pass — NOT run here):")
    for cls, n in sorted(tf_pred.items(), key=lambda x: -x[1]):
        print(f"  {n:3d}  {cls}")

    print("\nsubset-limit (out-of-C90-subset) cases — gcc OK, c4 rejects BY DESIGN:")
    for name, feat, src, gexp, reason in SUBSET_LIMIT_CASES:
        gcc_rc, gerr = _gcc_exit(src)
        try:
            _compile(src)
            c4 = "c4-COMPILED (unexpected!)"
        except Exception as e:
            c4 = f"c4-REJECTED: {type(e).__name__}"
        print(f"  {name:20s} [{feat:20s}] gcc={gcc_rc}  {c4}  <- {reason}")

    matrix = os.path.join(HERE, "CONFORMANCE_MATRIX_EXT2.json")
    with open(matrix, "w") as f:
        json.dump({
            "cpu_only": True, "gcc": GCC, "n_cases": len(CASES2),
            "n_gcc": n_gcc, "n_compile": n_compile, "n_native": n_native,
            "n_native_gcc": n_native_gcc, "tf_predicted_classes": tf_pred,
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {matrix}")

    if fails:
        print(f"\n{len(fails)} FAILS:")
        for nm, stage, why in fails:
            print(f"  {nm:22s} [{stage}] {why}")
        return 1
    print("\nALL GREEN (native-c4 == gcc for every wave-2 case, modulo c4-vs-x86 sizeof)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
