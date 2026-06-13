#!/usr/bin/env python3
"""Validate an ARITHMETIC-opcode NOT-blocker as the Root-2 dump discriminator.

On the STACK0-marker row the widened layout carries a weak (~0.11) per-step
opcode. The over-fire rows are ARITHMETIC / JMP; the framing-drift rows are
COMPARISON. So gate the dump on the SUM of arith opcodes being ~0.

We dump, on the LAST STACK0 row at the dump block input (41), the exact values
of OP_ADD/SUB/MUL/DIV/MOD/JMP/BZ and the comparison opcodes, for a broad set.
A clean gate exists iff:  Σ(arith ops) is HIGH on every arith/jmp program and
~0 on every comparison program.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_stack0_arithgate.py
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
ARITH = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
         "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR"]
CTRL = ["OP_JMP", "OP_BZ", "OP_BNZ", "OP_JSR", "OP_LEV", "OP_ENT"]
CMP = ["OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_EQ", "OP_NE"]
MEMOP = ["OP_LI", "OP_SI", "OP_LC", "OP_SC", "OP_LEA", "OP_PSH", "OP_IMM"]
DUMP_BLK = 41


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


def main():
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    P = {n: int(dp[n]) for n in (ARITH + CTRL + CMP + MEMOP) if n in dp}
    probe = GroundTruthProbe.build()

    cmp_progs = {
        "if_gt": "int main() { if (17 > 35) return 1; return 0; }",
        "if_lt": "int main() { if (35 < 17) return 1; return 0; }",
        "if_eq": "int main() { if (17 == 35) return 1; return 0; }",
        "if_ne": "int main() { if (17 != 35) return 1; return 0; }",
        "if_ge": "int main() { if (35 >= 17) return 1; return 0; }",
        "if_le": "int main() { if (17 <= 35) return 1; return 0; }",
        "bool_and": "int main() { return (17 > 3) && (5 < 9); }",
        "bool_or":  "int main() { return (17 > 3) || (5 > 9); }",
    }
    ari_progs = {
        "add16": "int main() { return 100 + 200; }",
        "sub16": "int main() { return 300 - 100; }",
        "add_s": "int main() { return 10 + 32; }",
        "mul":   "int main() { return 6 * 7; }",
        "or":    "int main() { return 12 | 3; }",
        "and":   "int main() { return 12 & 10; }",
        "shl":   "int main() { return 3 << 2; }",
    }

    def row_at(src):
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=16)
        pl = len(probe._build_context(bc))
        rows = find_markers(ctx, pl)
        out = []
        for step, p in rows:
            res = residual_at_block(probe, ctx, DUMP_BLK, p)
            out.append((step, p, {n: round(float(res[P[n]]), 3) for n in P}))
        return out

    print("=== LAST STACK0 row, block", DUMP_BLK, "===")
    print(f"{'prog':<10} {'Σarith':>7} {'Σctrl':>7} {'Σcmp':>7} "
          f"{'Σmem':>7}   top-3 opcodes")
    summary = {"cmp": [], "ari": []}
    for cls, progs in (("CMP", cmp_progs), ("ARI", ari_progs)):
        print(f"--- {cls} ---")
        for tag, src in progs.items():
            rows = row_at(src)
            if not rows:
                print(f"{tag:<10} (no STACK0 rows)"); continue
            step, p, v = rows[-1]
            sa = sum(v[n] for n in ARITH if n in v)
            sc = sum(v[n] for n in CTRL if n in v)
            scmp = sum(v[n] for n in CMP if n in v)
            sm = sum(v[n] for n in MEMOP if n in v)
            top = sorted(v.items(), key=lambda kv: -abs(kv[1]))[:3]
            top = {k: x for k, x in top if abs(x) > 0.005}
            print(f"{tag:<10} {sa:>7.3f} {sc:>7.3f} {scmp:>7.3f} {sm:>7.3f}   {top}")
            (summary["cmp"] if cls == "CMP" else summary["ari"]).append(
                (tag, sa, sc, scmp))

    # The gate hypothesis: dump fires iff Σarith ~ 0 (NOT-arith blocker).
    cmp_arith = [sa for _, sa, _, _ in summary["cmp"]]
    ari_arith = [sa for _, sa, _, _ in summary["ari"]]
    print(f"\nΣarith  CMP range [{min(cmp_arith):.3f},{max(cmp_arith):.3f}]  "
          f"ARI range [{min(ari_arith):.3f},{max(ari_arith):.3f}]  "
          f"sep(ari_min - cmp_max)={min(ari_arith)-max(cmp_arith):.3f}")
    # Also Σ(arith+ctrl): JMP rows have ctrl not arith.
    cmp_ac = [sa+sc for _, sa, sc, _ in summary["cmp"]]
    ari_ac = [sa+sc for _, sa, sc, _ in summary["ari"]]
    print(f"Σ(arith+ctrl) CMP [{min(cmp_ac):.3f},{max(cmp_ac):.3f}]  "
          f"ARI [{min(ari_ac):.3f},{max(ari_ac):.3f}]")


if __name__ == "__main__":
    main()
