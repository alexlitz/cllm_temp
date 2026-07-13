#!/usr/bin/env python3
"""SHR OUTPUT byte-0 leak probe (campaign config, spec_k=0, no hooks).

Confirms the ShiftOutputClearFFN root: on the SHR (84>>1) MARK_AX compute row,
a stale OUTPUT_LO+0 / OUTPUT_HI+0 = 2.0 zero-default is written UPSTREAM (claimed
block 16 / logical L11) and, because GEToBDConverter ADDS the shift's own result
one-hot, SURVIVES to TIE the true 0x2A -> argmax breaks toward cell-0 -> 0x00.

Runs with the ShiftOutputClear DISABLED (C4_SHIFT_OUTPUT_B0_CLEAR=0) so we see the
raw leak, and dumps the OUTPUT_LO/OUTPUT_HI band on the SHR MARK_AX row at every
requested block, to localize the FIRST block that plants OUTPUT_{LO,HI}+0.

  python tools/probe_shr_output_b0_leak.py [blocks...]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
# Disable the corrector so the RAW leak is visible (unless caller overrides).
os.environ.setdefault("C4_SHIFT_OUTPUT_B0_CLEAR", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _make_bytecode(items):
    out = []
    for it in items:
        if isinstance(it, tuple):
            op, arg = it
            out.append(int(op))
            out.append(int(arg))
        else:
            out.append(int(it))
    return out


# 84 >> 1 == 42  (matches TestSmokeShift::test_shr)
SHR_PROG = _make_bytecode([
    (Opcode.IMM, 84), Opcode.PSH,
    (Opcode.IMM, 1), Opcode.SHR,
    Opcode.EXIT,
])
# 21 << 1 == 42  (the passing control; its OUTPUT band should be empty)
SHL_PROG = _make_bytecode([
    (Opcode.IMM, 21), Opcode.PSH,
    (Opcode.IMM, 1), Opcode.SHL,
    Opcode.EXIT,
])


def fmt(row, base, width=16, thr=0.3):
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.2f}@{i}" for i, v in enumerate(vals)
                            if abs(v) > thr) + "]"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    for pname, prog, op_dim in (("SHR 84>>1", SHR_PROG, "OP_SHR"),
                                ("SHL 21<<1", SHL_PROG, "OP_SHL")):
        _, oracle_tokens = runner._oracle_pc_ax_steps(
            prog, b"", "", expected_steps=None, with_tokens=True)
        prompt = runner._build_element(prog, b"", [], "", spec_k=1,
                                       adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in oracle_tokens:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        ax_base = dp["MARK_AX"]
        with torch.no_grad():
            emb = model.embed(padded)[0]
            resid_last = model.forward(padded, stop_after_block=11)[0]

        # Find the shift MARK_AX compute row (OP_SHR / OP_SHL fires at block 11).
        shift_row = None
        shift_step = None
        for stp_idx in range(len(oracle_tokens)):
            ss = len(prefix) + stp_idx * STEP
            for off in range(STEP):
                r = ss + off
                if r >= len(tape):
                    continue
                if emb[r, ax_base].abs().item() <= 0.5:
                    continue
                if op_dim in dp and resid_last[r, dp[op_dim]].item() > 0.5:
                    shift_row = r
                    shift_step = stp_idx
                    break
            if shift_row is not None:
                break

        print(f"\n=== {pname}: shift MARK_AX row = {shift_row} "
              f"(step {shift_step}) ===", flush=True)
        if shift_row is None:
            print("   NO shift MARK_AX row found", flush=True)
            continue

        # Emitted result (does it decode 0x00 or 0x2A?)
        code, val = p.emitted_result(prog, max_steps=20)
        print(f"   emitted_result: code={code!r} value={val}", flush=True)

        # OUTPUT band across blocks on the shift row.
        for blk in blocks:
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=blk)[0]
            row = resid[shift_row]
            print(f"   blk{blk:2d} OUTPUT_LO={fmt(row, dp['OUTPUT_LO'])} "
                  f"OUTPUT_HI={fmt(row, dp['OUTPUT_HI'])}", flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or list(range(0, 20))
    main(blks)
