#!/usr/bin/env python3
"""Run the FULL smoke gate (spec_k=0 batched) in ONE model build.

Mirrors ``tests/test_smoke.py`` exactly: same ``BatchedPureNeuralRunner``
(via the ground-truth probe builder), same ``run_batch(..., spec_k=0,
bucket_by_predicted_length=False)`` per group, same declarative-oracle
step budget, same per-test ``check`` callable. Prints a PASS/FAIL line
per test plus a summary with the target / guardrail breakdown the
STEP_END CMP/ALU task tracks.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/run_full_smoke.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from tests.test_smoke import _SMOKE_GROUPS  # noqa: E402
from tests.declarative_oracle import declarative_oracle_for_program  # noqa: E402

# The 6 targets and the 8 must-not-regress guardrails for this task.
TARGETS = {
    "TestSmokeComparison::test_eq_true",
    "TestSmokeComparison::test_eq_false",
    "TestSmokeBasic::test_mul_basic",
    "TestSmokeBitwise::test_and_basic",
    "TestSmoke32Bit::test_and_16bit",
    "TestSmoke32Bit::test_add_carry_cascade",
}
GUARDRAILS = {
    "TestSmokeComparison::test_lt_true",
    "TestSmokeComparison::test_le_true",
    "TestSmokeComparison::test_ne_true",
    "TestSmokeComparison::test_gt_true",
    "TestSmokeComparison::test_ge_true",
    "TestSmokeShift::test_shl",
    "TestSmokeShift::test_shr",
    "TestSmokeIntegration::test_cmp_and_branch",
}


def run_group(runner, tests):
    oracles = [
        declarative_oracle_for_program(
            t["bytecode"], b"", suite_check=t["check"], label=t["name"])
        for t in tests
    ]
    runnable = [(t, o) for t, o in zip(tests, oracles) if o.error is None]
    out = {t["name"]: ("", None, o.error)
           for t, o in zip(tests, oracles) if o.error is not None}
    if not runnable:
        return out
    bcs = [t["bytecode"] for t, _o in runnable]
    expected = [o.steps for _t, o in runnable]
    res = runner.run_batch(
        bcs, max_steps=None, spec_k=0, expected_steps_list=expected,
        bucket_by_predicted_length=False,
    )
    for (t, _o), (output, code) in zip(runnable, res):
        out[t["name"]] = (output, code, None)
    return out


def main(only_groups=None):
    probe = build_groundtruth_probe()
    runner = probe.runner
    name_to_check = {}
    results = {}
    for gname, tests in _SMOKE_GROUPS.items():
        if only_groups and gname not in only_groups:
            continue
        for t in tests:
            name_to_check[t["name"]] = t["check"]
        results.update(run_group(runner, tests))

    n_pass = n_fail = 0
    tgt_pass, grd_pass, grd_fail = [], [], []
    lines = []
    for name, (output, code, err) in sorted(results.items()):
        ok = False
        detail = ""
        if err is not None:
            detail = f"ORACLE_ERR {err}"
        elif code is None:
            detail = "NO_RESULT"
        else:
            try:
                name_to_check[name](code)
                ok = True
            except AssertionError as e:
                detail = str(e)
        tag = ""
        if name in TARGETS:
            tag = " [TARGET]"
            if ok:
                tgt_pass.append(name)
        if name in GUARDRAILS:
            tag = " [GUARD]"
            (grd_pass if ok else grd_fail).append(name)
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        lines.append(f"  {status} {name}{tag}"
                     + (f"  ({detail})" if not ok else ""))

    print("\n".join(lines))
    print("=" * 60)
    print(f"SMOKE: {n_pass} passed, {n_fail} failed "
          f"(total {n_pass + n_fail})")
    print(f"TARGETS passing ({len(tgt_pass)}/6): "
          f"{sorted(n.split('::')[1] for n in tgt_pass)}")
    print(f"GUARDRAILS passing ({len(grd_pass)}/8): "
          f"{sorted(n.split('::')[1] for n in grd_pass)}")
    if grd_fail:
        print(f"!!! GUARDRAIL REGRESSIONS: "
              f"{sorted(n.split('::')[1] for n in grd_fail)}")
    return n_pass


if __name__ == "__main__":
    grps = sys.argv[1:] or None
    main(grps)
