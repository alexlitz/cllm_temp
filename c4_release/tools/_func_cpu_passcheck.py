#!/usr/bin/env python3
"""CPU passcheck (exit-code criterion) for a program-id range.

Mirrors the ORIGINAL func 25/25 measurement (tools/_probe_func_passcheck.py),
but on CPU via the DSL-interpreter IR forward (DSLInterpreterVerdictRunner)
instead of the GPU groundtruth probe. The criterion is the canonical full-trace
exit-code criterion: ``emitted exit_code & 0xFF == oracle exit_code & 0xFF``.
This is the SAME criterion the func-branch reported as 25/25; the stricter
per-token DSL full_trace verdict (tools/_func_dsl_verdict.py) is a DIFFERENT,
harsher criterion.

Honours the C4_L15_LEV_* / C4_L16_* / C4_L8_* env flags via the cached compile.

Usage: python tools/_func_cpu_passcheck.py <id_lo-id_hi[,...]> [alu_mode]
"""
from __future__ import annotations
import os, sys, contextlib, io, time
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
    alu_mode = sys.argv[2] if len(sys.argv) > 2 else "lookup"
    progs = generate_test_programs()
    from neural_vm.verification.dsl_interpreter_verdict import (
        DSLInterpreterVerdictRunner,
    )
    _b0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        runner = DSLInterpreterVerdictRunner(alu_mode=alu_mode)
    print(f"[model built in {time.time() - _b0:.1f}s]", flush=True)
    npass = nfail = nerr = 0
    rows = []
    for pid in ids:
        if pid >= len(progs):
            continue
        src, exp, desc = progs[pid]
        try:
            bc = compile_c(src)[0]
            oracle = declarative_oracle_for_program(bc, suite_expected=exp)
            nsteps = oracle.steps  # Optional[int]: declarative halt step count
            with contextlib.redirect_stderr(io.StringIO()):
                res = runner.run_batch(
                    [bc], data_list=[[]], max_steps=(nsteps + 4) if nsteps else 60,
                    spec_k=0,
                )[0]
            _out, exit_code = res
            ok = (exit_code & 0xFF) == (exp & 0xFF)
            if ok:
                npass += 1
                st = "PASS"
            else:
                nfail += 1
                st = "FAIL"
            rows.append((pid, desc[:26], st, exit_code & 0xFF, exp & 0xFF))
            print(f"  [done] id{pid} {desc[:24]:26s} {st} got={exit_code & 0xFF} exp={exp & 0xFF}", flush=True)
        except Exception as e:
            nerr += 1
            rows.append((pid, desc[:26], "ERR", "", str(e)[:50]))
            print(f"  [done] id{pid} ERR {str(e)[:60]}", flush=True)
    print(f"PASS={npass} FAIL={nfail} ERR={nerr}  alu_mode={alu_mode} "
          f"(n={len([i for i in ids if i < len(progs)])})")
    for pid, d, st, g, e in rows:
        if st != "PASS":
            print(f"  {st} id{pid} {d:28s} got={g} exp={e}")


if __name__ == "__main__":
    main()
