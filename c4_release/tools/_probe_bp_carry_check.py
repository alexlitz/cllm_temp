#!/usr/bin/env python3
"""Check: did the BP carry head fill BP_SAVE_PREV, and did the dump fire?
spec_k=0. Usage: python tools/_probe_bp_carry_check.py <id> <maxsteps> <ent_step>
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_rows(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], []).append(i); i += 1; continue
        i += 1
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    ent_step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    rows = step_rows(ctx, pl)
    nblk = len(probe.model.blocks)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    print(f"id{pid} {desc} ENT step={ent_step} BP_SAVE_PREV pos={dp.get('BP_SAVE_PREV')}")

    BP = dp["BP_SAVE_PREV"]; OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]
    mem_idx = rows[ent_step]["MEM"][0]
    # val byte k predictor row = mem_idx + 4 + k (off4..off7)
    # The carry head runs at block 16; the dump at the last tail block.
    print("\n=== BP_SAVE_PREV band @ block 16 (after carry head) at val predictor rows ===")
    for k in range(4):
        pos = mem_idx + 4 + k
        r = probe.model.forward(padded, stop_after_block=16)[0][pos]
        lo = int(torch.argmax(r[BP:BP+16])); lom = round(float(r[BP+lo]), 1)
        hi = int(torch.argmax(r[BP+16:BP+32])); him = round(float(r[BP+16+hi]), 1)
        bandsum = round(float(r[BP:BP+32].abs().sum()), 1)
        print(f"  val{k} pred pos={pos}: BP_SAVE byte=0x{(hi<<4)|lo:02x} "
              f"(lo={lo}@{lom} hi={hi}@{him}) |band|={bandsum}")

    print("\n=== OUTPUT_LO/HI @ final block at val predictor rows (after dump) ===")
    for k in range(4):
        pos = mem_idx + 4 + k
        r = probe.model.forward(padded, stop_after_block=nblk-1)[0][pos]
        lo = int(torch.argmax(r[OL:OL+16])); lom = round(float(r[OL+lo]), 1)
        hi = int(torch.argmax(r[OH:OH+16])); him = round(float(r[OH+hi]), 1)
        print(f"  val{k} pred pos={pos}: OUTPUT byte=0x{(hi<<4)|lo:02x} "
              f"(lo={lo}@{lom} hi={hi}@{him}) emitted_tok(next)={ctx[pos+1] if pos+1<len(ctx) else '-'}")

    # Also dump the OP_ENT gate value at the val predictor rows at the dump block.
    OE = dp["OP_ENT"]
    print("\n=== OP_ENT + MEM_VAL_B markers @ block 32 (dump gate read) ===")
    for k in range(4):
        pos = mem_idx + 4 + k
        r = probe.model.forward(padded, stop_after_block=32)[0][pos]
        mvb = {f"MEM_VAL_B{j}": round(float(r[dp[f'MEM_VAL_B{j}']]), 1) for j in range(4)
               if abs(float(r[dp[f'MEM_VAL_B{j}']])) > 0.4}
        print(f"  val{k} pred pos={pos}: OP_ENT={round(float(r[OE]),1)} {mvb}")


if __name__ == "__main__":
    main()
