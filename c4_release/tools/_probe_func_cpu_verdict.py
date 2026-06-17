#!/usr/bin/env python3
"""CPU faithful-autoregressive full_trace verdict for a program-id range.

Uses FaithfulAutoregressiveRunner (CPU, byte-identical to the neural spec_k=0
canonical full_trace verdict) so we can A/B the func/nested chain without GPU
contention. Honours the C4_L15_LEV_* / C4_L16_* env flags via build_cpu_model.

Usage: python tools/_probe_func_cpu_verdict.py <id_lo-id_hi[,...]>
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
    spec_k = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    progs = generate_test_programs()
    from neural_vm.unified_compiler.faithful_autoregressive import FaithfulAutoregressiveRunner
    with contextlib.redirect_stdout(io.StringIO()):
        runner = FaithfulAutoregressiveRunner()
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
            rows.append((pid, desc[:26], st, f"div={ds}", f"ax={gax} exp={exp&0xff}"))
        except Exception as e:
            nerr += 1
            rows.append((pid, desc[:26], "ERR", str(e)[:50], ""))
    print(f"PASS={npass} FAIL={nfail} ERR={nerr}  (n={len([i for i in ids if i < len(progs)])})")
    for pid, d, st, a, b in rows:
        if st != "pass":
            print(f"  id{pid} {d:28s} {st:10s} {a} {b}")


if __name__ == "__main__":
    main()
