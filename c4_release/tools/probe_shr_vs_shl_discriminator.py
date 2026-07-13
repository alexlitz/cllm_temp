#!/usr/bin/env python3
"""Find the SHR-vs-SHL discriminator at the L11 emission-head input (block 16).

The L11 emission heads copy CLEAN_EMBED -> OUTPUT. On the SHR MARK_AX compute row
one fires (planting OUTPUT+0=2.0); on the SHL row none do. Dump, for BOTH rows at
block 16, every dim the OUTPUT-writing heads' Q reads (BYTE_INDEX_*, CONST,
MARK_AX, H1, ADDR_B0_*, CMP, OP_*, HAS_SE, PSH_AT_SP, MARK_SP, MARK_STACK0), so we
see WHICH input differs -> the correct-by-construction gate.
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
os.environ.setdefault("C4_SHIFT_OUTPUT_B0_CLEAR", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(items):
    out = []
    for it in items:
        if isinstance(it, tuple):
            out.append(int(it[0])); out.append(int(it[1]))
        else:
            out.append(int(it))
    return out


SHR = _mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHR, Opcode.EXIT])
SHL = _mk([(Opcode.IMM, 21), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHL, Opcode.EXIT])

WATCH = ["MARK_AX", "MARK_SE_ONLY", "HAS_SE", "CONST", "IS_BYTE",
         "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
         "PSH_AT_SP", "MARK_SP", "MARK_STACK0", "MEM_STORE",
         "OP_SHL", "OP_SHR", "OP_PSH", "OP_LI", "OP_SI",
         "AX_CARRY_LO", "AX_CARRY_HI"]


def rowdump(model, dp, padded, row, blk):
    with torch.no_grad():
        r = model.forward(padded, stop_after_block=blk)[0][row]
    out = {}
    for nm in WATCH:
        if nm in dp:
            b = dp[nm]
            # band peak
            vals = [float(r[b + i].item()) for i in range(min(16, 16))]
            peak = max(vals, key=abs)
            if abs(peak) > 0.2:
                out[nm] = peak
    # H1 band
    if "H1" in dp:
        h1 = [f"{r[dp['H1']+i].item():.1f}@{i}" for i in range(8) if abs(r[dp['H1']+i].item()) > 0.3]
        out["H1"] = h1
    return out


def main():
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    for label, prog, opd in (("SHR", SHR, "OP_SHR"), ("SHL", SHL, "OP_SHL")):
        _, otoks = runner._oracle_pc_ax_steps(prog, b"", "", expected_steps=None, with_tokens=True)
        prompt = runner._build_element(prog, b"", [], "", spec_k=1, adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in otoks:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        ax = dp["MARK_AX"]
        with torch.no_grad():
            emb = model.embed(padded)[0]
            r11 = model.forward(padded, stop_after_block=11)[0]
        row = None
        for si in range(len(otoks)):
            ss = len(prefix) + si * STEP
            for off in range(STEP):
                rr = ss + off
                if rr < len(tape) and emb[rr, ax].abs().item() > 0.5 and r11[rr, dp[opd]].item() > 0.5:
                    row = rr; break
            if row is not None:
                break
        print(f"\n=== {label} compute row {row} @blk16 ===")
        d = rowdump(model, dp, padded, row, 16)
        for k, v in d.items():
            print(f"    {k:16} = {v}")


if __name__ == "__main__":
    main()
