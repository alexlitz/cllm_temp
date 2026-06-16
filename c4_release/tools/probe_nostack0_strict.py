#!/usr/bin/env python3
"""Authoritative no-STACK0 frame reproduction via run_batch_fail_fast strict_trace.

Runs a control slice AUTOREGRESSIVELY (spec_k=0) and compares the model's
EMITTED tokens against the DraftVM reference at SAFE offsets (skips MEM
addr/val bytes), reporting per-program first divergence (step, offset, name,
expected/got token). This is the exact criterion run_1096_canonical uses.

Run twice (cache cleared between): C4_NO_STACK0_EMIT unset vs =1.

Usage:
    CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 python tools/probe_nostack0_strict.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return list(bc), exp, desc


def main(ids, criterion="full_trace"):
    print(f"### Token.STEP_TOKENS = {Token.STEP_TOKENS}  criterion={criterion}")
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)

    from tests.declarative_oracle import declarative_oracle_for_program

    bcs, exps, descs, steps_list = [], [], [], []
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        bcs.append(bc)
        exps.append(exp)
        descs.append(desc)
        try:
            oracle = declarative_oracle_for_program(bc, b"", label="probe")
            steps_list.append(oracle.steps)
        except Exception:
            steps_list.append(None)

    results = runner.run_batch_fail_fast(
        bcs,
        data_list=[b"" for _ in bcs],
        max_steps=None,
        expected_steps_list=steps_list,
        max_context_window=4096,
        spec_k=0,
        criterion=criterion,
    )
    npass = 0
    for desc, exp, r in zip(descs, exps, results):
        st = r.get("status")
        if st == "pass":
            npass += 1
            print(f"  PASS  {desc[:42]:42} exp={exp}")
        elif st == "error":
            print(f"  ERROR {desc[:42]:42} {r.get('error')}")
        else:
            print(f"  FAIL  {desc[:42]:42} exp={exp} | "
                  f"step={r.get('divergence_step')} "
                  f"off={r.get('divergence_offset')} "
                  f"({r.get('divergence_offset_name')}) "
                  f"exp_tok={r.get('expected_tok')} got_tok={r.get('got_tok')} "
                  f"| got_pc={r.get('got_pc')} got_ax={r.get('got_ax')} "
                  f"exp_pc={r.get('expected_pc')} exp_ax={r.get('expected_ax')}")
    print(f"\n### {criterion}: {npass}/{len(ids)} pass")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.isdigit()]
    crit = "strict_trace" if "--strict" in sys.argv else "full_trace"
    # control slice: add/sub/div/mod first members
    ids = args or [0, 1, 2, 50, 51, 52]
    main(ids, criterion=crit)
