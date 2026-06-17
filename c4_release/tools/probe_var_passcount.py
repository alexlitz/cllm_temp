#!/usr/bin/env python3
"""Count full_trace PASSES over an id range using the production neural runner.

Mirrors the smoke gate's full_trace path: BatchedPureNeuralRunner.run_batch_fail_fast
with spec_k=0, expected_steps from the declarative oracle, criterion='full_trace'.
Prints per-cluster pass counts so we can measure var_mul / var_three / if_var and
controls (var_simple, func, etc.) before/after a fix.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/probe_var_passcount.py 275-324
  CUDA_VISIBLE_DEVICES=0 C4_LEA_LOCAL_E8_MULTILOCAL_GUARD=1 python tools/probe_var_passcount.py 275-299,300-324,425-449
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, contextlib, io, re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def parse_ids(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return ids


def cluster_of(desc):
    base = desc.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    return base


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "275-324"
    ids = parse_ids(spec)
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    tests = generate_test_programs()
    with contextlib.redirect_stdout(io.StringIO()):
        runner = BatchedPureNeuralRunner(csr_inference=False)

    by_cluster = {}
    for idx in ids:
        src, exp, desc = tests[idx]
        cl = cluster_of(desc)
        bc, data = compile_c(src)
        try:
            orc = declarative_oracle_for_program(bc, data, suite_expected=exp,
                                                  label=f"id={idx}", max_steps=60)
        except Exception as e:
            orc = None
        if orc is None or orc.error is not None or orc.steps is None:
            by_cluster.setdefault(cl, [0, 0, 0])[2] += 1  # skip
            print(f"  id={idx} {desc[:34]}: SKIP (oracle)", flush=True)
            continue
        r = runner.run_batch_fail_fast(
            [bc], data_list=[data], max_steps=None,
            expected_steps_list=[orc.steps], spec_k=0, criterion="full_trace",
        )[0]
        ok = r.get("status") == "pass"
        slot = by_cluster.setdefault(cl, [0, 0, 0])
        if ok:
            slot[0] += 1
        else:
            slot[1] += 1
        print(f"  id={idx} {desc[:34]}: {r.get('status')} "
              f"div={r.get('divergence_step')}", flush=True)

    print("\n=== per-cluster full_trace pass counts ===")
    tot_p = tot_f = tot_s = 0
    for cl in sorted(by_cluster):
        p, f, s = by_cluster[cl]
        tot_p += p; tot_f += f; tot_s += s
        print(f"  {cl:16s}: pass={p} fail={f} skip={s}")
    print(f"  {'TOTAL':16s}: pass={tot_p} fail={tot_f} skip={tot_s}")


if __name__ == "__main__":
    main()
