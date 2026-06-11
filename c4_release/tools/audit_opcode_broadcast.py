#!/usr/bin/env python3
"""Audit the "opcode-broadcast defeats intent-blocker" bug class.

spec_k=0, hook-free. Reads dims via ``probe.model.dim_positions`` (NOT the
stale ``dim_registry_dynamic``).

PART A — broadcast magnitudes
-----------------------------
For the frame/control opcodes (OP_JSR, OP_ENT, OP_LEV, OP_ADJ, OP_LEA) and a
control set of ALU opcodes (OP_ADD, OP_SUB, OP_OR, OP_MUL, OP_EQ, OP_LT), probe
the actual residual magnitude of each ``OP_<X>`` dim at EVERY row type
(register markers PC/AX/SP/BP, the per-register byte rows, STACK0 / STEP_END
markers) during a frame program. Reads the residual at the *input* to the L20
FFN block (physical block 28 output) — the block where the l16 frame/control
rules read the opcode flag — and also a late residual where it peaks.

This quantifies "how big is the broadcast" per opcode x row type, the number the
intent-blocker promotions were sized against (OP_JSR ~15.5, OP_ENT ~9.7-11.4).

Usage:
    python tools/audit_opcode_broadcast.py            # default: var_simple_12 (id 262)
    python tools/audit_opcode_broadcast.py 262 250 271
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402

from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402

MARKERS = {
    int(Token.REG_PC): "PC",
    int(Token.REG_AX): "AX",
    int(Token.REG_SP): "SP",
    int(Token.REG_BP): "BP",
    int(Token.STEP_END): "STEP_END",
    int(Token.HALT): "HALT",
}

# Frame/control opcodes (brief's primary set) + an ALU control set to verify the
# "ALU opcodes may NOT broadcast" hypothesis.
FRAME_OPS = ["OP_JSR", "OP_ENT", "OP_LEV", "OP_ADJ", "OP_LEA"]
ALU_OPS = ["OP_ADD", "OP_SUB", "OP_OR", "OP_AND", "OP_MUL", "OP_EQ", "OP_LT"]


def opcode_of_byte(b: int) -> str:
    from neural_vm.embedding import Opcode

    for nm in dir(Opcode):
        if nm.startswith("_"):
            continue
        try:
            if int(getattr(Opcode, nm)) == (b & 0xFF):
                return f"OP_{nm}"
        except Exception:
            pass
    return "?"


def build_program(idx: int):
    tests = generate_test_programs()
    src, exp, _ = tests[idx]
    bc, _ = compile_c(src)
    return bc, exp


def enumerate_rows(probe, bc):
    ctx = probe._final_context(bc)
    prompt_len = len(probe._build_context(bc))
    rows = {}
    step = 0
    i = prompt_len
    while i < len(ctx):
        t = ctx[i]
        nm = MARKERS.get(t)
        if nm == "STEP_END":
            rows[f"s{step}_STEP_END"] = i
            step += 1
            i += 1
            continue
        if nm in ("PC", "AX", "SP", "BP"):
            rows[f"s{step}_{nm}_mark"] = i
            for j in range(4):
                if i + 1 + j < len(ctx):
                    rows[f"s{step}_{nm}_b{j}"] = i + 1 + j
            i += 5
            continue
        i += 1
    return ctx, prompt_len, rows


def main():
    ids = [int(a) for a in sys.argv[1:]] or [262]

    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    dev = next(m.parameters()).device
    bl = probe.block_layer_map()
    n_blocks = len(m.blocks)

    op_dims = {}
    for nm in FRAME_OPS + ALU_OPS:
        if nm in dp:
            op_dims[nm] = int(dp[nm])
        else:
            print(f"# WARN: {nm} not in model.dim_positions")

    print("# dim_positions for measured opcodes:")
    for nm, d in op_dims.items():
        print(f"#   {nm} = dim {d}")
    print(f"# n_physical_blocks = {n_blocks}")

    L20_INPUT_BLOCK = 28
    LATE_BLOCK = min(n_blocks - 1, 33)

    for idx in ids:
        bc, exp = build_program(idx)
        ctx, prompt_len, rows = enumerate_rows(probe, bc)
        print("\n" + "=" * 78)
        print(f"# PROGRAM id={idx} exp={exp} | "
              f"opcodes={[opcode_of_byte(bc[k]) for k in range(0, len(bc), 8)]}")
        print(f"# {len(rows)} probed rows; reading OP_<X> after block "
              f"{L20_INPUT_BLOCK} (input to L20 FFN) and block {LATE_BLOCK}")

        for read_block, tag in ((L20_INPUT_BLOCK, "L20_input"),
                                (LATE_BLOCK, "late")):
            padded = torch.tensor([ctx], dtype=torch.long, device=dev)
            with torch.no_grad():
                resid = m.forward(padded, stop_after_block=read_block)[0].float()

            print(f"\n  --- {tag} (after phys block {read_block}, "
                  f"L{bl[read_block]['logical']}) ---")
            mark_rows = [p for k, p in rows.items() if k.endswith("_mark")]
            byte_rows = [p for k, p in rows.items() if "_b" in k]
            se_rows = [p for k, p in rows.items() if k.endswith("STEP_END")]

            print(f"  {'opcode':<8} {'mark_max':>9} {'mark_mean':>9} "
                  f"{'byte_max':>9} {'byte_mean':>9} {'SE_max':>8}")
            for nm, d in op_dims.items():
                def stats(positions):
                    vals = [float(resid[p, d]) for p in positions
                            if p < resid.shape[0]]
                    if not vals:
                        return 0.0, 0.0
                    return max(abs(v) for v in vals), sum(vals) / len(vals)
                mmax, mmean = stats(mark_rows)
                bmax, bmean = stats(byte_rows)
                semax, _ = stats(se_rows)
                print(f"  {nm:<8} {mmax:>9.3f} {mmean:>9.3f} "
                      f"{bmax:>9.3f} {bmean:>9.3f} {semax:>8.3f}")

        print("\n  --- per-step OP_<X> peak at register-marker rows (L20_input) ---")
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            resid = m.forward(padded, stop_after_block=L20_INPUT_BLOCK)[0].float()
        n_steps = max(int(k[1:].split("_")[0]) for k in rows
                      if k.startswith("s")) + 1
        for step in range(n_steps):
            cells = []
            for nm, d in op_dims.items():
                vmark = []
                for reg in ("PC", "AX", "SP", "BP"):
                    p = rows.get(f"s{step}_{reg}_mark")
                    if p is not None and p < resid.shape[0]:
                        vmark.append(float(resid[p, d]))
                if vmark:
                    peak = max(vmark, key=abs)
                    if abs(peak) > 0.5:
                        cells.append(f"{nm}={peak:+.1f}")
            if cells:
                print(f"    step{step}: {' '.join(cells)}")


if __name__ == "__main__":
    main()
