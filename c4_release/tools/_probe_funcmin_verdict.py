#!/usr/bin/env python3
"""AUTHORITATIVE per-id full_trace verdict on CPU (same criterion as the
2-hour ``run_1096_canonical --criterion full_trace --spec-k 0`` gate).

Runs ``BatchedPureNeuralRunner.run_batch_fail_fast`` (the production spec_k=0
autoregressive decode) over a handful of ids and reports pass/fail per id.
This is the criterion the fast_gate uses — NOT the teacher-forced
interp_oracle_gate (which over-flags cross-step programs). Use it to confirm
the func_min id675 regression + the 4 gains, on CPU, without the GPU gate.

  C4_ALU_OPERAND_SURVIVE=1 python tools/_probe_funcmin_verdict.py --ids 675,612,408,433,1088
  C4_ALU_OPERAND_SURVIVE=0 python tools/_probe_funcmin_verdict.py --ids 675,612,408,433,1088
"""
from __future__ import annotations
import os, sys, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tests.declarative_oracle import declarative_oracle_for_program  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402

PROGS = generate_test_programs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="675,612,408,433,1088")
    ap.add_argument("--max-steps-cap", type=int, default=40)
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    flag = os.environ.get("C4_ALU_OPERAND_SURVIVE", "1")
    print(f"=== C4_ALU_OPERAND_SURVIVE={flag} (full_trace, spec_k=0, CPU) ===")

    t0 = time.monotonic()
    runner = BatchedPureNeuralRunner(max_seq_len=2048)
    print(f"[bake {time.monotonic()-t0:.1f}s device={runner._device}]")

    bcs, datas, decl_exits, decl_steps_l, metas = [], [], [], [], []
    for pid in ids:
        src, exp, desc = PROGS[pid]
        bc, data = compile_c(src)
        oracle = declarative_oracle_for_program(bc, data)
        decl_exit = oracle.exit_code if oracle is not None else None
        decl_steps = oracle.steps if oracle is not None else None
        bcs.append(bc); datas.append(data if isinstance(data, (bytes, bytearray)) else bytes(data))
        decl_exits.append(decl_exit)
        decl_steps_l.append(min(decl_steps or args.max_steps_cap,
                                args.max_steps_cap))
        metas.append((pid, exp, desc, decl_exit))

    ff = runner.run_batch_fail_fast(
        bcs, data_list=datas, max_steps=None,
        expected_steps_list=decl_steps_l,
        max_context_window=2048, spec_k=32, criterion="full_trace",
    )
    for (pid, exp, desc, decl_exit), res in zip(metas, ff):
        status = res.get("status")
        div = res.get("divergence_step")
        nexit = res.get("decoded_exit")
        exp_pc, exp_ax = res.get("expected_pc"), res.get("expected_ax")
        got_pc, got_ax = res.get("got_pc"), res.get("got_ax")
        verdict = "PASS" if status == "pass" else "FAIL"
        print(f"  id{pid:<4d} {verdict}  exp={exp} decl_exit={decl_exit} "
              f"decoded_exit={nexit} div_step={div}  {desc!r}")
        if status != "pass":
            print(f"        expected(pc={exp_pc},ax={exp_ax}) "
                  f"got(pc={got_pc},ax={got_ax})")


if __name__ == "__main__":
    main()
