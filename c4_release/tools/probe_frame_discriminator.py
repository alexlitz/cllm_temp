#!/usr/bin/env python3
"""Find a CAUSALLY-AVAILABLE early-block discriminator between the EXPR
operand frame (dump must NOT fire) and the IF/BOOL comparison-drift frame
(dump SHOULD fire).

For each program, identify the SPECIFIC STACK0 frame the dump corrupts:
 - EXPR (arith consumer): the operand frame whose CONSUMER (next step) is an
   arithmetic op. We locate it as the STACK0 marker at step s where step s+1's
   AX row carries an ARITH opcode.
 - IF/BOOL (cmp consumer): the comparison-drift frame -- the STACK0 marker at
   step s where step s+1 is OP_BZ/OP_BNZ, OR the last comparison frame.

Then, at an EARLY block (before the L25 corruptor), scan EVERY residual dim and
report dims whose value SEPARATES the two frame classes (mean over expr frames
vs mean over if/bool frames, with a separation score). The top separators are
candidate carry-flag sources.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_frame_discriminator.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch
from src.compiler import compile_c
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

STACK0_MARK = 268
AX_MARK = 258
ARITH = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
         "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR"]
CMP = ["OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_EQ", "OP_NE"]
BRANCH = ["OP_BZ", "OP_BNZ"]


@torch.no_grad()
def full_residual_block(probe, ctx, block_idx):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    return x[0].float().cpu()


def step_of(ctx, position):
    step = 0
    for i in range(position):
        if ctx[i] == int(Token.STEP_END):
            step += 1
    return step


def main():
    BLK = 8   # early, before L25 corruptor; consumer opcode clean at AX rows
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    inv = {}
    for name, pos in dp.items():
        inv.setdefault(int(pos), name)
    Pa = {n: int(dp[n]) for n in ARITH if n in dp}
    Pc = {n: int(dp[n]) for n in (CMP + BRANCH) if n in dp}
    probe = GroundTruthProbe.build()

    expr_progs = {
        "mul_div(14*56/8)": "int main() { return 14 * 56 / 8; }",
        "add_mul(3+4*5)":   "int main() { return 3 + 4 * 5; }",
        "add_mul(26+13*1)": "int main() { return 26 + 13 * 1; }",
        "mod(46%6+6)":      "int main() { return 46 % 6 + 6; }",
        "mod(23%10+4)":     "int main() { return 23 % 10 + 4; }",
        "mul_div(10*9/3)":  "int main() { return 10 * 9 / 3; }",
    }
    cmp_progs = {
        "if_gt(28>9)":  "int main() { if (28 > 9) return 1; return 0; }",
        "if_lt(35<17)": "int main() { if (35 < 17) return 1; return 0; }",
        "if_eq(17==35)":"int main() { if (17 == 35) return 1; return 0; }",
        "if_gt(7>3)":   "int main() { if (7 > 3) return 1; return 0; }",
        "bool_and":     "int main() { return (17 > 3) && (5 < 9); }",
        "if_lt(2<8)":   "int main() { if (2 < 8) return 1; return 0; }",
    }

    def corrupted_frame_residual(src, want_arith_consumer):
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=20)
        full = full_residual_block(probe, ctx, BLK)
        ax_rows = {}
        for i in range(len(ctx)):
            if ctx[i] == AX_MARK:
                ax_rows[step_of(ctx, i)] = i
        stack0 = [(step_of(ctx, i), i) for i in range(len(ctx))
                  if ctx[i] == STACK0_MARK]
        picks = []
        for s, p in stack0:
            cons = ax_rows.get(s + 1)
            if cons is None:
                continue
            res_c = full[cons]
            arith_v = max((float(res_c[Pa[n]]) for n in Pa), default=0)
            cmp_v = max((float(res_c[Pc[n]]) for n in Pc), default=0)
            if want_arith_consumer and arith_v > 3.0:
                picks.append((s, p, full[p]))
            if (not want_arith_consumer) and cmp_v > 3.0:
                picks.append((s, p, full[p]))
        return picks

    expr_frames, cmp_frames = [], []
    for tag, src in expr_progs.items():
        expr_frames += [r for _, _, r in corrupted_frame_residual(src, True)]
    for tag, src in cmp_progs.items():
        cmp_frames += [r for _, _, r in corrupted_frame_residual(src, False)]

    print(f"expr (arith-consumer) frames: {len(expr_frames)}; "
          f"cmp (BZ/cmp-consumer) frames: {len(cmp_frames)}")
    if not expr_frames or not cmp_frames:
        print("not enough frames"); return

    E = torch.stack(expr_frames)  # [Ne, D]
    C = torch.stack(cmp_frames)   # [Nc, D]
    em, es = E.mean(0), E.std(0) + 1e-6
    cm, cs = C.mean(0), C.std(0) + 1e-6
    # separation score per dim: |mean diff| / (combined std), require either
    # class consistently nonzero
    diff = (em - cm).abs()
    sep = diff / (es + cs)
    # rank dims with large diff AND meaningful magnitude
    D = E.shape[1]
    scored = []
    for d in range(D):
        if diff[d].item() < 0.05:
            continue
        scored.append((sep[d].item(), d, em[d].item(), cm[d].item(),
                       es[d].item(), cs[d].item()))
    scored.sort(reverse=True)
    print("\nTop separators (sep_score, dim, expr_mean, cmp_mean, e_std, c_std, name):")
    for sc, d, eM, cM, eS, cS, in scored[:40]:
        print(f"  sep={sc:6.2f} dim={d:4d} expr={eM:8.3f} cmp={cM:8.3f} "
              f"e_std={eS:.3f} c_std={cS:.3f}  {inv.get(d,'?')}")


if __name__ == "__main__":
    main()
