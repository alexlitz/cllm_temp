#!/usr/bin/env python3
"""Run the AX_CARRY-clear hook (at the divmod AX row, blocks 28+) across ALL
residual div+mod fails to quantify the fix's win count BEFORE editing the model.
Reads the fail ids from the baseline jsonls.
"""
import os, sys, json
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe
from tests.test_suite_1000 import generate_test_programs


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            o, i = op; bc.append(o | (i << 8))
        else:
            bc.append(op)
    return bc


def main():
    progs = generate_test_programs()
    # collect fails
    fails = []
    for path, sep in [("/tmp/div_baseline.json.jsonl", "/"), ("/tmp/mod_baseline.json.jsonl", "%")]:
        for l in open(path):
            r = json.loads(l)
            if r["status"] not in ("ok", "skipped"):
                i = r["idx"]
                expr = progs[i][0].replace("int main() { return ", "").replace("; }", "")
                a, b = expr.split(f" {sep} "); a = int(a); b = int(b)
                fails.append((i, a, b, sep))

    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    AXC_LO = dp["AX_CARRY_LO"]; AXC_HI = dp["AX_CARRY_HI"]
    OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]; MARK_AX = dp["MARK_AX"]
    state = {"on": False}

    def hook(module, inp, out):
        if not state["on"]:
            return out
        x = out[0] if isinstance(out, tuple) else out
        divmod_ax = (
            ((x[:, :, OP_DIV] > 0.5) | (x[:, :, OP_MOD] > 0.5))
            & (x[:, :, MARK_AX] > 0.5)
        )
        mask = divmod_ax[:, :, None].to(x.dtype)
        x[:, :, AXC_LO:AXC_LO + 16] = x[:, :, AXC_LO:AXC_LO + 16] * (1 - mask)
        x[:, :, AXC_HI:AXC_HI + 16] = x[:, :, AXC_HI:AXC_HI + 16] * (1 - mask)
        return (x,) + out[1:] if isinstance(out, tuple) else x

    hs = [model.blocks[b].register_forward_hook(hook) for b in range(28, nblocks)]

    fixed = 0; broke = 0; still = 0
    div_fix = mod_fix = 0
    for (i, a, b, sep) in fails:
        op = Opcode.DIV if sep == "/" else Opcode.MOD
        exp = (a // b) if sep == "/" else (a % b)
        PROG = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), op, Opcode.EXIT])
        state["on"] = False
        _, base = probe.emitted_result(PROG, max_steps=6)
        state["on"] = True
        _, fx = probe.emitted_result(PROG, max_steps=6)
        state["on"] = False
        tag = ""
        if fx == exp and base != exp:
            fixed += 1; tag = "FIXED"
            if sep == "/": div_fix += 1
            else: mod_fix += 1
        elif base == exp and fx != exp:
            broke += 1; tag = "BROKE"
        else:
            still += 1
        if tag:
            print(f"  id{i} {a}{sep}{b}: base={base} fixed={fx} exp={exp} {tag}")
    for h in hs:
        h.remove()
    print(f"\n# AX_CARRY-clear on {len(fails)} residual fails: FIXED={fixed} (div {div_fix}, mod {mod_fix}) BROKE={broke} STILL_FAIL={still}")


if __name__ == "__main__":
    main()
