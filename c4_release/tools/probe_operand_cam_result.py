#!/usr/bin/env python3
"""Fast free-run exit-code check for the operand-CAM fix targets.

Free-runs (autoregressive, spec_k=0, campaign default) each target program to
HALT and decodes the final AX exit code, comparing to the expected result. This
is the pure-neural exit-code criterion (NOT the strict per-step full_trace
verdict, but a fast directional signal for whether clearing the ALU_HI address
leak corrects the emitted RESULT byte). Run once with C4_OPERAND_CAM_FIX unset
and once =1.

  CUDA_VISIBLE_DEVICES=1 [C4_OPERAND_CAM_FIX=1] python tools/probe_operand_cam_result.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

# Targets that the operand-CAM ALU_HI leak affects (loaded operand consumers)
# plus HOLD clusters that must NOT regress.
TARGET_IDS = [275, 276, 277, 425, 426, 427, 1046, 1047, 1048,
              650, 651, 675, 676]
HOLD_IDS = [0, 1, 50, 51, 250, 251, 550, 575, 325, 350, 400]


def main():
    p = build_groundtruth_probe()
    progs = generate_test_programs()
    flag = os.environ.get("C4_OPERAND_CAM_FIX", "0")
    print(f"C4_OPERAND_CAM_FIX={flag} "
          f"NOSTK={os.environ.get('C4_NO_STACK0_EMIT','1')} "
          f"MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP','1')}", flush=True)
    for label, ids in (("TARGET", TARGET_IDS), ("HOLD", HOLD_IDS)):
        print(f"--- {label} ---", flush=True)
        npass = 0
        for idx in ids:
            src, exp, desc = progs[idx]
            bc, data = compile_c(src)
            try:
                _, code = p.emitted_result(bc, max_steps=80)
            except Exception as e:
                print(f"  id={idx:5d} {desc[:30]:30s} ERROR {e}", flush=True)
                continue
            ok = (code == (exp & 0xFFFFFFFF)) or (code == exp)
            npass += int(ok)
            print(f"  id={idx:5d} {desc[:30]:30s} exp={exp:6d} got={code:6d} "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
        print(f"  {label} pass {npass}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
