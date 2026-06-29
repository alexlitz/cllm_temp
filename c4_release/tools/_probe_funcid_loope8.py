#!/usr/bin/env python3
"""Probe the _l10_loop_lea_b0_e8 discriminator dims at func_identity id550's
step-8 (LEA &x) and step-9 (LI) AX-marker rows, plus the final OUTPUT byte-0.

Determines WHETHER the loop-LEA-e8 op fires on the LI row (step 9) and stamps
0xE8 over the loaded value.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/_probe_funcid_loope8.py [id] [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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


def byte(r, dp, lo, hi):
    if lo not in dp or hi not in dp:
        return None
    a = r[dp[lo]:dp[lo]+16]; b = r[dp[hi]:dp[hi]+16]
    return (int(b.argmax()) << 4) | int(a.argmax())


def cell(r, dp, name):
    """Return scalar value at a single-dim signal (or the +k cell if name has +)."""
    if "+" in name:
        base, k = name.split("+"); k = int(k)
        if base not in dp: return None
        return float(r[dp[base] + k])
    if name not in dp: return None
    return float(r[dp[name]])


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    nblk = len(probe.model.blocks)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    print(f"id{pid} {desc} exp={exp}  nblk={nblk}")

    disc_dims = ["MARK_AX", "OP_LEA", "OP_LI", "OP_LI_RELAY",
                 "FETCH_LO+8", "FETCH_LO+0", "FETCH_HI+14", "FETCH_HI+15",
                 "MEM_ADDR_SRC", "IS_BYTE", "OPCODE_BYTE_LO+6"]
    # The op runs late (after ent_axcarry, on the L25 tail block). Probe at the
    # block JUST BEFORE the tail op and at the FINAL block.
    for st in range(6, min(len(sm), 11)):
        row = sm[st].get("AX")
        if row is None:
            continue
        print(f"\n=== step {st} AX-marker row={row} ===")
        # discriminators at an early-ish block (decode is stable by ~blk30)
        for blk in (30, nblk - 2, nblk - 1):
            if blk >= nblk or blk < 0:
                continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][row]
            vals = []
            for d in disc_dims:
                v = cell(r, dp, d)
                if v is not None:
                    vals.append(f"{d}={v:+.2f}")
            axb = byte(r, dp, "OUTPUT_LO", "OUTPUT_HI")
            axb_this = byte(r, dp, "OUTPUT_LO", "OUTPUT_HI_THIS_STEP") if "OUTPUT_HI_THIS_STEP" in dp else None
            extra = f" OUTbyte0=0x{axb:02x}" if axb is not None else ""
            if axb_this is not None:
                extra += f" OUT_THIS=0x{axb_this:02x}"
            print(f"  blk{blk:2d}: " + "  ".join(vals) + extra)


if __name__ == "__main__":
    main()
