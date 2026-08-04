"""End-to-end flash byte-exactness: the FULL 242-block CFM forward must produce the
SAME per-step AX trace with C4_FLASH_ATTN=1 (flash) vs OFF (masked-full softmax1) on
real c90 cases, under the proven battery flag set.

This is the model-level gate (the isolated-Attn check verify_flash_byte_exact.py
proves the kernel; this proves the WHOLE model.forward is unchanged so
run_pure_forward_complete's decoded AX is identical).  If every step's AX agrees
flag-ON vs flag-OFF for a spread of cases, flash is safe for the full battery.

Run:  CUDA_VISIBLE_DEVICES=0,1 C4_PF_CFM=1 C4_CMP32=1 C4_CMP32_ORDER=1 \
        C4_MEM_ADDR_BITS=18 C4_EXACT_EVICT=1 C4_MEM_EFF=500000 \
        PYTHONPATH=<c4_release> python id_port/c90_e2e/verify_flash_e2e.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import c4_min.nibble_pure_forward as _PF  # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa  # noqa: E402
from c4_min.nibble_pure_forward_complete import (  # noqa: E402
    build_pure_forward_complete_model, run_pure_forward_complete)
from src.compiler import compile_c  # noqa: E402
import native_c4  # noqa: E402
from cases_ext import CASES  # noqa: E402


def _compile(src):
    words, data = compile_c(src)
    code = []
    for w in words:
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append(isa.Instr(op, imm))
    return code, data


# a representative spread (short + medium) so the e2e check is fast but exercises the
# arithmetic / control / storage / pointer / function paths.
PICK = {"op_add", "if_else", "global_rw", "ptr_deref", "fn_call", "while_sum",
        "ternary_basic", "str_first_char"}


def main():
    cases = [c for c in CASES if c[0] in PICK]
    compiled = [(n, cat, *_compile(src), exp) for (n, cat, src, exp) in cases]
    maxlen = max(len(code) for (_, _, code, _, _) in compiled)
    print(f"building CFM model (code_size={maxlen + 2}) ...", flush=True)
    model, L = build_pure_forward_complete_model(code_size=maxlen + 2)
    model.eval()

    all_ok = True
    for (name, cat, code, data, exp) in compiled:
        native_ax, nsteps = native_c4.run(code, data=data)
        cap = min(2500, nsteps * 3 + 80)
        seed_mem = {65536 + k: b for k, b in enumerate(data) if b} if data else None

        os.environ.pop("C4_FLASH_ATTN", None)
        tr_off = run_pure_forward_complete(model, L, code, max_steps=cap,
                                           mask=0xFFFFFFFF, seed_mem=seed_mem)
        os.environ["C4_FLASH_ATTN"] = "1"
        tr_on = run_pure_forward_complete(model, L, code, max_steps=cap,
                                          mask=0xFFFFFFFF, seed_mem=seed_mem)
        os.environ.pop("C4_FLASH_ATTN", None)

        same = (tr_off == tr_on)
        ax_off = (tr_off[-1] & 0xFF) if tr_off else None
        ax_on = (tr_on[-1] & 0xFF) if tr_on else None
        all_ok = all_ok and same
        print(f"  {name:18s} [{cat:9s}] steps_off={len(tr_off):4d} "
              f"steps_on={len(tr_on):4d} ax_off={ax_off} ax_on={ax_on} "
              f"nat={native_ax & 0xFF} trace-identical={'YES' if same else '*** NO ***'}",
              flush=True)
        if not same:
            # first divergent step
            for i, (a, b) in enumerate(zip(tr_off, tr_on)):
                if a != b:
                    print(f"      first diff at step {i}: off={a & 0xFFFFFFFF} "
                          f"on={b & 0xFFFFFFFF}")
                    break

    print(f"\nFLASH e2e byte-exact (full 242-block model.forward): "
          f"{'ALL IDENTICAL' if all_ok else '*** DIVERGENCE ***'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
