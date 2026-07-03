#!/usr/bin/env python3
"""Find a dim that distinguishes the GENUINE LEV PC marker (step 8) from the
post-LEV ADJ PC marker (step 9), where OP_LEV residue leaks. head 14 fires at
BOTH (OP_LEV AND MARK_PC), so we need a clean discriminator.

Dumps, at the head-14 INPUT (pre-L15 block) for the step-8 and step-9 PC marker
rows, every dim whose value differs by >0.5 between the two rows, ranked by
abs difference. The biggest clean separators are candidate gate signals.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/_probe_func_lev_disc.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    lev = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    inv = {}
    for k, v in dp.items():
        inv.setdefault(v, k)
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    l15 = [ph for ph, b in enumerate(probe.model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    pre15 = probe.model.forward(padded, stop_after_block=l15 - 1)[0]

    r8 = pre15[sm[lev]["PC"]]
    r9 = pre15[sm[lev + 1]["PC"]]
    print(f"id{pid} {desc}: LEV-step{lev} PC-row vs step{lev+1} PC-row (head-14 input, pre-block-{l15})")
    diffs = []
    for d in range(r8.shape[0]):
        a, b = float(r8[d]), float(r9[d])
        if abs(a - b) > 0.5:
            diffs.append((abs(a - b), d, a, b))
    diffs.sort(reverse=True)
    print(f"  top discriminating dims (|step{lev} - step{lev+1}| > 0.5):")
    for ad, d, a, b in diffs[:40]:
        nm = inv.get(d, str(d))
        print(f"    {nm:28s}(d{d:4d}): step{lev}={a:+8.2f}  step{lev+1}={b:+8.2f}  |diff|={ad:.2f}")


if __name__ == "__main__":
    main()
