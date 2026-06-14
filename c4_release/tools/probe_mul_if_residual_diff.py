#!/usr/bin/env python3
"""Diff the FULL residual at the (batched-indistinguishable) mul operand-setup PSH
row vs the if operand-setup PSH row to find ANY separating dim. Both carry
OP_PSH=0.019 and an identical weak-nibble mass; the only known difference is
program context (mul->OP_MUL vs if->OP_GT/OP_BZ). Find a local separator.

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_mul_if_residual_diff.py
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
LIVE_BLK = 44


@torch.no_grad()
def residual_at_block(probe, ctx, block_idx, position):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return (r.to_dense() if r.is_sparse else r).float().cpu()


def find_markers(ctx, prompt_len):
    rows, i, step = [], prompt_len, 0
    while i < len(ctx):
        if ctx[i] == STACK0_MARK:
            rows.append((step, i))
        if ctx[i] == int(Token.STEP_END):
            step += 1
        i += 1
    return rows


def setup_psh_row(probe, src, want_step):
    bc = compile_c(src)[0]
    ctx = probe._final_context(bc, max_steps=16)
    pl = len(probe._build_context(bc))
    rows = find_markers(ctx, pl)
    for step, p in rows:
        if step == want_step:
            return residual_at_block(probe, ctx, LIVE_BLK, p)
    return None


def main():
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    inv = {}
    for nm, pos in dp.items():
        inv.setdefault(int(pos), nm)
    probe = GroundTruthProbe.build()

    # Both step-2 PSH rows: mul (over-fire, must BLOCK) vs if (fix, must FIRE).
    # mul_21x59 step2 (H1 weak) and ifGT_28x9 step2 (H1 weak) — same nibble sig.
    mul = setup_psh_row(probe, "int main() { return 21 * 59; }", 2)
    iff = setup_psh_row(probe, "int main() { if (28 > 9) return 1; return 0; }", 2)
    # also the H3-weak pair: mul_97x94 vs ifGT_50x89
    mul2 = setup_psh_row(probe, "int main() { return 97 * 94; }", 2)
    iff2 = setup_psh_row(probe, "int main() { if (50 > 89) return 1; return 0; }", 2)

    d = (mul - iff).abs()
    order = torch.argsort(d, descending=True)
    print("=== TOP separating dims: mul_21x59.s2 (BLOCK) vs ifGT_28x9.s2 (FIRE) ===")
    print(f"{'dim':>5} {'name':<28} {'mul':>12} {'if':>12} {'|diff|':>12}")
    for k in order[:40].tolist():
        nm = inv.get(k, f"?{k}")
        mv, iv = float(mul[k]), float(iff[k])
        if abs(mv - iv) < 1e-3:
            break
        # also check the SECOND pair to see if the separator is consistent
        mv2, iv2 = float(mul2[k]), float(iff2[k])
        consistent = (mv - iv) * (mv2 - iv2) > 0 or (abs(mv2 - iv2) < 1e-3)
        flag = "" if consistent else "  (INCONSISTENT on pair2)"
        print(f"{k:>5} {nm:<28} {mv:>12.3f} {iv:>12.3f} {abs(mv-iv):>12.3f}"
              f"  | p2: mul={mv2:.2f} if={iv2:.2f}{flag}")


if __name__ == "__main__":
    main()
