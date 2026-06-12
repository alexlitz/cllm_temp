#!/usr/bin/env python3
"""Full smoke gate with a disk_cache=False model build (immune to the shared
compiled-vm cache that concurrent agents pollute). Production config
(trust_neural_alu=True -> efficient mode), spec_k=0 batched run_batch, the SAME
per-test checks as tools/run_full_smoke.py.
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_ROOT))
import torch
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
from tests.test_smoke import _SMOKE_GROUPS
from tests.declarative_oracle import declarative_oracle_for_program

TARGETS = {
    "TestSmokeComparison::test_eq_true", "TestSmokeComparison::test_eq_false",
    "TestSmokeBasic::test_mul_basic", "TestSmokeBitwise::test_and_basic",
    "TestSmoke32Bit::test_and_16bit", "TestSmoke32Bit::test_add_carry_cascade",
}
GUARDRAILS = {
    "TestSmokeComparison::test_lt_true", "TestSmokeComparison::test_le_true",
    "TestSmokeComparison::test_ne_true", "TestSmokeComparison::test_gt_true",
    "TestSmokeComparison::test_ge_true", "TestSmokeShift::test_shl",
    "TestSmokeShift::test_shr", "TestSmokeIntegration::test_cmp_and_branch",
}


def run_group(runner, tests):
    oracles = [declarative_oracle_for_program(t["bytecode"], b"", suite_check=t["check"], label=t["name"]) for t in tests]
    runnable = [(t, o) for t, o in zip(tests, oracles) if o.error is None]
    out = {t["name"]: ("", None, o.error) for t, o in zip(tests, oracles) if o.error is not None}
    if not runnable:
        return out
    bcs = [t["bytecode"] for t, _o in runnable]
    expected = [o.steps for _t, o in runnable]
    res = runner.run_batch(bcs, max_steps=None, spec_k=0, expected_steps_list=expected, bucket_by_predicted_length=False)
    for (t, _o), (output, code) in zip(runnable, res):
        out[t["name"]] = (output, code, None)
    return out


def main():
    print("compiling (disk_cache=False, EXACT production config)...", flush=True)
    # Match AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    # EXACTLY: alu_mode='efficient', DEFAULT_N_HEADS/DEFAULT_FFN_HIDDEN,
    # max_seq_len=4096. Passing defaults to compile_full_vm_dynamic produced
    # a STRUCTURALLY DIFFERENT model (lea_basic/cmp_and_branch baseline FAIL),
    # so the args must mirror run_vm.py's compile call.
    from neural_vm.vm_step import DEFAULT_N_HEADS, DEFAULT_FFN_HIDDEN
    model, _ = compile_full_vm_dynamic(
        strict=False, disk_cache=False, alu_mode="efficient",
        n_heads=DEFAULT_N_HEADS, ffn_hidden=DEFAULT_FFN_HIDDEN,
        max_seq_len=4096,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0, cache_model=False)
    mr.model = model
    mr._func_call_handlers = {}; mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)

    name_to_check = {}
    results = {}
    for gname, tests in _SMOKE_GROUPS.items():
        for t in tests:
            name_to_check[t["name"]] = t["check"]
        results.update(run_group(runner, tests))

    n_pass = n_fail = 0
    tgt_pass, grd_pass, grd_fail = [], [], []
    lines = []
    for name, (output, code, err) in sorted(results.items()):
        ok = False; detail = ""
        if err is not None:
            detail = f"ORACLE_ERR {err}"
        elif code is None:
            detail = "NO_RESULT"
        else:
            try:
                name_to_check[name](code); ok = True
            except AssertionError as e:
                detail = str(e)
        tag = ""
        if name in TARGETS:
            tag = " [TARGET]"
            if ok: tgt_pass.append(name)
        if name in GUARDRAILS:
            tag = " [GUARD]"
            (grd_pass if ok else grd_fail).append(name)
        status = "PASS" if ok else "FAIL"
        n_pass += ok; n_fail += (not ok)
        lines.append(f"  {status} {name}{tag}" + (f"  ({detail})" if not ok else ""))
    print("\n".join(lines))
    print("=" * 60)
    print(f"SMOKE: {n_pass} passed, {n_fail} failed (total {n_pass + n_fail})")
    print(f"TARGETS passing ({len(tgt_pass)}/6): {sorted(n.split('::')[1] for n in tgt_pass)}")
    print(f"GUARDRAILS passing ({len(grd_pass)}/8): {sorted(n.split('::')[1] for n in grd_pass)}")
    if grd_fail:
        print(f"!!! GUARDRAIL REGRESSIONS: {sorted(n.split('::')[1] for n in grd_fail)}")


if __name__ == "__main__":
    main()
