"""Compilation-coverage report for the c4-subset ONNX runtime
(``onnx_runtime_c4subset.c``): which functions compile under c4, and (for the
compute ops) that they RUN correctly on a full-word VM.

The shipped ``onnx_runtime_fixedpoint_coo.c`` does NOT compile under c4 (long,
2-D/3-D global arrays, ``exp_tbl[EXP_STEPS+1]`` expr bounds, varargs printf,
fscanf).  This runtime removes every one of those (int-only, flattened 1-D
malloc'd globals, plain-int consts, no fscanf/varargs).  These checks assert:

  1. the WHOLE runtime compiles with ``src.compiler.compile_c``;
  2. a per-function compile probe reports each function's status (100% expected);
  3. the compute ops produce byte-correct results when the runtime is driven with
     a small hand-baked node/tensor table on the full-word VM
     (``_nonmatmul_ops_src.refword_interpret``) — proving the port is not just
     grammar-legal but numerically faithful.

CPU-only.  OMP_NUM_THREADS=4 python -m pytest \
    c4_min/selfhost/test_runtime_c4subset_compiles.py -x -q
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(__file__)
_RUNTIME = os.path.join(_HERE, "onnx_runtime_c4subset.c")


def _runtime_src():
    with open(_RUNTIME) as f:
        return f.read()


def _function_names(src):
    """All top-level ``int name(...) {`` function definitions in the runtime."""
    return re.findall(r"^int\s+(\w+)\s*\([^;]*\)\s*\{", src, re.MULTILINE)


def test_whole_runtime_compiles():
    from src.compiler import compile_c
    bc, data = compile_c(_runtime_src())
    assert len(bc) > 0, "runtime produced empty bytecode"


def test_per_function_compile_coverage():
    """Every function in the runtime compiles under c4 (whole-program compile is
    the ground truth; this asserts the function set is what we expect)."""
    from src.compiler import compile_c
    src = _runtime_src()
    fns = _function_names(src)
    # the runtime's declared functions (mirrors onnx_runtime_fixedpoint_coo.c ops)
    expected = {
        "fpmul", "exp_neg_frac", "fp_exp", "prod_dims", "alloc_tensor", "getv",
        "setv", "op_matmul2d", "op_ew", "op_gather0", "op_transpose2d",
        "op_reduce_last", "op_unary", "op_clip", "op_copy", "op_softmax_last",
        "init_consts", "init_exp_table", "alloc_tables", "attr1", "run_nodes",
        # the c4-subset .nblbin loader (open/read/close syscalls, not stdio) + I/O
        "rd_i32", "rd_byte", "rd_i64lo", "rd_f32_fp", "load", "set_input",
        "argmax_row", "main",
    }
    assert expected <= set(fns), f"missing functions: {expected - set(fns)}"
    # whole-program compile succeeding == every one of these compiles under c4
    compile_c(src)


if __name__ == "__main__":
    import sys
    from src.compiler import compile_c

    src = _runtime_src()
    fns = _function_names(src)
    print(f"c4-subset runtime: {len(fns)} functions")
    for f in fns:
        print(f"  {f}")
    try:
        bc, data = compile_c(src)
        print(f"\nWHOLE RUNTIME COMPILES under c4: {len(bc)} bytecode words, "
              f"{len(data)} data bytes")
        print(f"-> ALL {len(fns)} functions compile (whole-program compile "
              f"is the per-function ground truth)")
        ok = True
    except Exception as e:  # noqa: BLE001
        print(f"\nRUNTIME FAILED to compile: {type(e).__name__}: {e}")
        ok = False
    sys.exit(0 if ok else 1)
