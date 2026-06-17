#!/usr/bin/env python3
"""Ground-truth neural full_trace on specific wide MUL/DIV/MOD corpus ids.

Runs the REAL BatchedPureNeuralRunner (alu_mode='efficient', spec_k=0) per the
production fail-fast path and prints pass/fail + divergence for each id. This
resolves the CPU interp-oracle-gate's CROSS-STEP deferrals (which it cannot
certify) into real PASS/FAIL on the production decode path.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/_mul16_gt.py 100,102,103,134,291,616,809,817,853,854
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import torch


def main():
    ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else []
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    tests = generate_test_programs()
    progs = []
    for i in ids:
        src, exp, desc = tests[i]
        bc, data = compile_c(src)
        try:
            orc = declarative_oracle_for_program(bc, data, label=desc)
            steps = orc.steps
        except Exception as e:
            steps = None
        progs.append((i, desc, bc, data, exp, steps))

    runner = BatchedPureNeuralRunner(max_seq_len=4096)
    bcs = [p[2] for p in progs]
    datas = [p[3] for p in progs]
    steps_list = [p[5] for p in progs]
    res = runner.run_batch_fail_fast(
        bcs, data_list=datas, expected_steps_list=steps_list,
        max_context_window=512, spec_k=0, criterion="full_trace",
    )
    npass = 0
    for (i, desc, bc, data, exp, steps), r in zip(progs, res):
        st = r.get("status")
        ds = r.get("divergence_step")
        epc, eax = r.get("expected_pc"), r.get("expected_ax")
        gpc, gax = r.get("got_pc"), r.get("got_ax")
        dexit = r.get("decoded_exit")
        ok = st == "pass"
        npass += ok
        extra = ""
        if not ok:
            extra = (f" div@{ds} exp_ax={eax}({hex(eax) if eax is not None else '?'}) "
                     f"got_ax={gax}({hex(gax) if gax is not None else '?'}) "
                     f"exit={dexit}")
        print(f"  {'PASS' if ok else 'FAIL'}  id{i}:{desc[:42]:42}  exp={exp}  {extra}")
    print(f"  ---- {npass}/{len(progs)} pass ----")


if __name__ == "__main__":
    main()
