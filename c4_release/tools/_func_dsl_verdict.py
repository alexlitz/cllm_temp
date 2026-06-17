#!/usr/bin/env python3
"""CPU DSL-interpreter full_trace verdict for a program-id range.

Drives the production fail-fast decode with the per-token argmax coming from the
FaithfulInterpreter IR engine (``DSLInterpreterVerdictRunner``), which is the
non-re-anchored CPU verdict authority (byte-identical to the neural spec_k=0
full_trace per dsl_interpreter_verdict_validate.py). Honours the
C4_L15_LEV_* / C4_L16_* / C4_L8_* env flags via the cached compile.

Usage: python tools/_func_dsl_verdict.py <id_lo-id_hi[,...]> [spec_k] [alu_mode]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ["C4_TEST_SPEC_K"] = "0"; os.environ["C4_SMOKE_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
from tests.test_suite_1000 import generate_test_programs
from tests.declarative_oracle import declarative_oracle_for_program
from src.compiler import compile_c


def parse_ids(spec):
    ids = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def main():
    ids = parse_ids(sys.argv[1])
    spec_k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    alu_mode = sys.argv[3] if len(sys.argv) > 3 else "lookup"
    progs = generate_test_programs()
    from neural_vm.unified_compiler.dsl_interpreter_verdict import (
        DSLInterpreterVerdictRunner,
    )
    import time as _t
    _b0 = _t.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        runner = DSLInterpreterVerdictRunner(alu_mode=alu_mode)
    print(f"[model built in {_t.time() - _b0:.1f}s]", flush=True)
    npass = nfail = nerr = 0
    rows = []
    for pid in ids:
        if pid >= len(progs):
            continue
        src, exp, desc = progs[pid]
        try:
            bc = compile_c(src)[0]
            oracle = declarative_oracle_for_program(bc, suite_expected=exp)
            if oracle.error is not None or oracle.steps is None:
                nerr += 1; rows.append((pid, desc[:26], "ORACLE_ERR", "", "")); continue
            r = runner.run_batch_fail_fast(
                [bc], data_list=[[]], max_steps=None,
                expected_steps_list=[oracle.steps], spec_k=spec_k,
                criterion="full_trace",
            )[0]
            st = r.get("status")
            if st == "pass":
                npass += 1
            elif st == "error":
                nerr += 1
            else:
                nfail += 1
            ds = r.get("divergence_step", r.get("first_divergence_step", ""))
            gax = r.get("got_ax", r.get("decoded_ax", ""))
            rows.append((pid, desc[:26], st, f"div={ds}", f"ax={gax} exp={exp & 0xff}"))
            print(f"  [done] id{pid} {desc[:24]:26s} {st:8s} div={ds} ax={gax} exp={exp & 0xff}", flush=True)
        except Exception as e:
            nerr += 1
            rows.append((pid, desc[:26], "ERR", str(e)[:60], ""))
            print(f"  [done] id{pid} ERR {str(e)[:60]}", flush=True)
    print(f"PASS={npass} FAIL={nfail} ERR={nerr}  alu_mode={alu_mode} spec_k={spec_k} "
          f"(n={len([i for i in ids if i < len(progs)])})")
    for pid, d, st, a, b in rows:
        marker = "" if st == "pass" else "  <<"
        print(f"  id{pid} {d:28s} {st:10s} {a} {b}{marker}")


if __name__ == "__main__":
    main()
