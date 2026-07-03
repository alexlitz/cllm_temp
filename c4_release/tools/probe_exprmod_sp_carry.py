#!/usr/bin/env python3
"""AR SP-byte-0 probe for the expr_mod binary-pop CARRY case (#319).

Runs the byte-identical CPU autoregressive decode (FaithfulAutoregressiveRunner,
spec_k=0) on a list of expr_mod programs and dumps, per completed VM step, the
decoded (PC, AX, SP) plus the RAW 35-token slice so we can read the actual SP
byte-0 token the model emits at the binary-pop step. This is the AR self-check
the brief mandates: it reproduces the production fixed-35-token framing exactly
(no re-anchoring), so the SP byte-0 we read is the one that feeds back into the
next step's context.

Usage:
    C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_spcarry \
    python tools/probe_exprmod_sp_carry.py --ids 875,876
"""
from __future__ import annotations
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")

from neural_vm.verification.faithful_autoregressive import (  # noqa: E402
    FaithfulAutoregressiveRunner,
)
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402

MARKERS = {
    int(Token.REG_PC): "PC",
    int(Token.REG_AX): "AX",
    int(Token.REG_SP): "SP",
    int(Token.REG_BP): "BP",
    int(Token.STEP_END): "STEP_END",
}


def _decode_reg(step_tokens, marker):
    for i, tk in enumerate(step_tokens):
        if tk == marker and i + 4 < len(step_tokens):
            val = 0
            for j in range(4):
                val |= (int(step_tokens[i + 1 + j]) & 0xFF) << (j * 8)
            return val, i
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--max-steps", type=int, default=20)
    args = ap.parse_args()
    ids = []
    for tok in args.ids.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif tok:
            ids.append(int(tok))

    progs = generate_test_programs()
    runner = FaithfulAutoregressiveRunner(disk_cache=True)
    inner = runner._inner
    STEP = int(Token.STEP_TOKENS)

    # Hook the per-step verdict checker to capture each completed step's slice.
    captured = {}  # id -> list of (step_idx, slice)
    orig_check = inner._ff_check_new_steps

    cur_id = {"v": None}

    def patched_check(s, *, criterion="full_trace"):
        completed = s.token_pos // STEP
        rec = captured.setdefault(cur_id["v"], [])
        already = len(rec)
        for step_idx in range(already, completed):
            start = s.prefix_len + step_idx * STEP
            rec.append((step_idx, list(s.context[start:start + STEP])))
        return orig_check(s, criterion=criterion)

    inner._ff_check_new_steps = patched_check

    for pid in ids:
        src, exp, desc = progs[pid]
        bc = compile_c(src)[0]
        cur_id["v"] = pid
        oracle = None
        verdicts = inner.run_batch_fail_fast(
            [bc],
            expected_steps_list=[None],
            max_steps=args.max_steps,
            spec_k=0,
            criterion="full_trace",
        )
        v = verdicts[0]
        print("=" * 78)
        print(f"id {pid}: {desc.strip()}  expected_exit={exp}")
        print(f"  verdict={v.get('status')} div_step={v.get('divergence_step')} "
              f"exp_pc={v.get('expected_pc')} got_pc={v.get('got_pc')} "
              f"exp_ax={v.get('expected_ax')} got_ax={v.get('got_ax')}")
        for step_idx, sl in captured.get(pid, []):
            pc, _ = _decode_reg(sl, int(Token.REG_PC))
            ax, _ = _decode_reg(sl, int(Token.REG_AX))
            sp, sp_i = _decode_reg(sl, int(Token.REG_SP))
            sp_b0 = (sp & 0xFF) if sp is not None else None
            sp_str = f"0x{sp:06X}" if sp is not None else "None"
            b0_str = f"0x{sp_b0:02X}" if sp_b0 is not None else "--"
            print(f"    step {step_idx:2d}: PC={pc} AX={ax} SP={sp_str} "
                  f"SP_b0={b0_str}")


if __name__ == "__main__":
    main()
