#!/usr/bin/env python3
"""func_min/func_max RULE ATTRIBUTION via interp_oracle_gate.classify_program.

Reports, per program, the divergence step + reg/byte + expected/got + the
attributed owning declarative rule (block-input attribution). Run in the
campaign config to match the fix fleet's default 30-token model.

  C4_CAMPAIGN=1 C4_VM_CACHE_DIR=/tmp/funcattr_$$ \
    python tools/_probe_funcmin_attr.py --ids 675,650,676,651
"""
from __future__ import annotations
import os, sys, argparse

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
from tools.interp_oracle_gate import build_gate_context, classify_program  # noqa: E402

PROGS = generate_test_programs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="675,650,676,651")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    ctx = build_gate_context(verbose=True)
    for pid in ids:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        r = classify_program(ctx, name=f"id{pid}", bytecode=bc, data=b"",
                             cluster="func", max_steps=48, attribute=True)
        print(f"\n===== id{pid} {desc!r} exp={exp} =====")
        print(f"  class={r.classification} n_steps={r.n_steps} "
              f"conf={r.confidence} vcorr={r.value_correction_step}")
        if r.classification == "FAIL":
            print(f"  div_step={r.div_step} reg={r.div_reg} byte={r.div_byte} "
                  f"exp=0x{(r.expected or 0):02x} got=0x{(r.got or 0):02x} "
                  f"exp_reg={r.expected_reg} got_reg={r.got_reg}")
            print(f"  attributed_op={r.attributed_op} "
                  f"attributed_rule={r.attributed_rule} "
                  f"contrib={r.attributed_contrib}")
            print(f"  is_alu_step={r.is_alu_step} alu_op={r.alu_op}")
            print(f"  note: {r.note}")


if __name__ == "__main__":
    main()
