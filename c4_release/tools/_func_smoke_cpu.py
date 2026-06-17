#!/usr/bin/env python3
"""CPU exit-code check for the test_simple_function smoke program.

Builds the model honouring env flags, decodes the JSR/ENT/IMM/LEV smoke program
via the DSL-interpreter CPU forward, and reports whether the exit code == 42
(the smoke ``_eq(42)`` criterion). CPU-only; no GPU.

Usage: python tools/_func_smoke_cpu.py [alu_mode]
"""
from __future__ import annotations
import os, sys, contextlib, io, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ["C4_TEST_SPEC_K"] = "0"; os.environ["C4_SMOKE_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)


def main():
    alu_mode = sys.argv[1] if len(sys.argv) > 1 else "lookup"
    # Rebuild the test_simple_function bytecode exactly as test_smoke.py does.
    from tests.test_smoke import _make_bytecode
    from neural_vm.embedding import Opcode
    bc = _make_bytecode([
        (Opcode.JSR, 3),
        Opcode.EXIT,
        Opcode.NOP,
        (Opcode.ENT, 0),
        (Opcode.IMM, 42),
        Opcode.LEV,
    ])
    from neural_vm.unified_compiler.dsl_interpreter_verdict import (
        DSLInterpreterVerdictRunner,
    )
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        runner = DSLInterpreterVerdictRunner(alu_mode=alu_mode)
    print(f"[model built in {time.time() - t0:.1f}s]", flush=True)
    with contextlib.redirect_stderr(io.StringIO()):
        res = runner.run_batch([bc], data_list=[[]], max_steps=30, spec_k=0)[0]
    _out, exit_code = res
    ok = (exit_code & 0xFF) == 42
    print(f"test_simple_function: exit={exit_code & 0xFF} expect=42 -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
