#!/usr/bin/env python3
"""Report the AR (autoregressive) full_trace divergence STEP for var_three ids.

Runs the byte-identical CPU faithful AR runner (same verdict as
run_1096_canonical --criterion full_trace, spec_k=0) on a small id list and
prints, per id: status + the divergence step (ff_div_step) + expected/got
(PC,AX) at that step. Honours whatever C4_* env is set, so run it twice
(C4_VAR_THREE_LI unset vs =1) to A/B the fix's divergence-step delta.

Run:
  C4_CAMPAIGN=1                python tools/_probe_vt_ar_divstep.py 300,304
  C4_CAMPAIGN=1 C4_VAR_THREE_LI=1 python tools/_probe_vt_ar_divstep.py 300,304
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_1096_fast import _compile_and_oracle  # type: ignore
from tests.test_suite_1000 import generate_test_programs
from neural_vm.verification.faithful_autoregressive import (
    FaithfulAutoregressiveRunner,
)


def main():
    ids = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "300").split(",")]
    max_steps = int(os.environ.get("VT_MAX_STEPS", "40"))
    all_tests = generate_test_programs()
    selected = [(i, all_tests[i][0], all_tests[i][1], all_tests[i][2]) for i in ids]
    prepared, _errs = _compile_and_oracle(selected)

    runner = FaithfulAutoregressiveRunner()
    print(f"[divstep] C4_VAR_THREE_LI={os.environ.get('C4_VAR_THREE_LI')} "
          f"C4_CAMPAIGN={os.environ.get('C4_CAMPAIGN')} max_steps={max_steps}")

    for entry in prepared:
        idx, _expected, desc, _decl_exit, decl_steps, bytecode, data = entry
        r = runner.run_batch_fail_fast(
            [bytecode], data_list=[data], max_steps=max_steps,
            expected_steps_list=[decl_steps], spec_k=0,
            criterion="full_trace",
        )[0]
        status = r.get("status")
        ds = r.get("divergence_step")
        epc = r.get("ff_expected_pc") if "ff_expected_pc" in r else r.get("expected_pc")
        eax = r.get("ff_expected_ax") if "ff_expected_ax" in r else r.get("expected_ax")
        gpc = r.get("got_pc")
        gax = r.get("got_ax")
        print(f"id{idx} [{desc}]: status={status} div_step={ds} "
              f"exp=(pc={epc},ax={eax}) got=(pc={gpc},ax={gax})")


if __name__ == "__main__":
    main()
