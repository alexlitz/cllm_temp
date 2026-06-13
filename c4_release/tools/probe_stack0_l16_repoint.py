#!/usr/bin/env python3
"""Verify the L16 e8 materializer re-point premises (Root 2).

At the carried CMP-step STACK0-marker row, confirm:
  (1) the OUTPUT band (OUTPUT_LO / OUTPUT_HI_THIS_STEP) the materializer reads is
      WRONG (a different byte than the true stack-top byte-0), and
  (2) the STACK0_B0_*_PREV band holds the CORRECT carried byte-0 one-hot
      (STACK0_B0_H1_PREV = low nibble at idx lo+2; STACK0_B0_H3_PREV = high nibble
      at idx hi+4), and STACK0_B0_CARRIED ~ 1 on the carried row / 0 on the fresh
      PSH row.

We read these bands at the residual entering the L16 materializer's physical
block (logical layer 16). Run:
  CUDA_VISIBLE_DEVICES=0 python tools/probe_stack0_l16_repoint.py
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
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

_REG = build_default_registry_dynamic()
STACK0_MARK = 268


@torch.no_grad()
def residual_at_block(probe, ctx, block_idx, position):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return (r.to_dense() if r.is_sparse else r).float().cpu()


def find_stack0_markers(ctx, prompt_len):
    rows, i, step = [], prompt_len, 0
    while i < len(ctx):
        if ctx[i] == STACK0_MARK:
            rows.append((step, i))
        if ctx[i] == int(Token.STEP_END):
            step += 1
        i += 1
    return rows


def reg_band(res, name, width):
    base = _REG.slots[name].start
    return [round(float(res[base + j]), 1) for j in range(width)]


def pos_band(res, base, width):
    return [round(float(res[base + j]), 1) for j in range(width)]


def argmax16(res, base):
    vals = [float(res[base + k]) for k in range(16)]
    return max(range(16), key=lambda k: vals[k]), round(max(vals), 1)


def main():
    # Get the layout's dim_positions (the STACK0_B0_* bands live only here).
    _model, layout = compile_full_vm_dynamic(
        alu_mode="efficient", strict=False,
    )
    dp = layout.dim_positions
    b_h1_prev = int(dp["STACK0_B0_H1_PREV"])
    b_h3_prev = int(dp["STACK0_B0_H3_PREV"])
    b_carried = int(dp["STACK0_B0_CARRIED"])
    print(f"dim_positions: H1_PREV={b_h1_prev} H3_PREV={b_h3_prev} "
          f"CARRIED={b_carried}  d_model={layout.d_model}")

    probe = GroundTruthProbe.build()
    rows_map = probe.block_layer_map()
    nblocks = len(probe.model.blocks)
    print(f"total physical blocks: {nblocks}")
    # In the WIDENED build the H1/H3 nuke (the stack0_e8 materializer) lands at
    # block 38 (per probe_stack0_byte0_blocktrace). Read the residual ENTERING
    # block 38 (after block 37).
    NUKE_BLOCK = 38
    for r in rows_map:
        if r["physical"] in (37, 38, 39):
            print(f"  block {r['physical']}: logical={r['logical']} "
                  f"exp={r['is_post_op_expansion']} ffn={r['ffn']}")
    l16_last = NUKE_BLOCK
    read_block = NUKE_BLOCK - 1

    src = "int main() { if (17 > 35) return 1; return 0; }"  # 0x11 DRIFT
    bc = compile_c(src)[0]
    ctx = probe._final_context(bc, max_steps=12)
    pl = len(probe._build_context(bc))
    rows = find_stack0_markers(ctx, pl)
    psh_marker = next(p for s, p in rows if s == 1)
    cmp_marker = next(p for s, p in rows if s == 2)
    print(f"PSH(step1) pos={psh_marker}  CMP(step2,carried) pos={cmp_marker}")
    print(f"reading residual AFTER block {read_block} (entering L16 block "
          f"{l16_last})\n")

    def show_output(res, tag):
        lo = [round(float(res[_REG.slots["OUTPUT_LO"].start + k]), 1) for k in range(16)]
        hi = [round(float(res[_REG.slots["OUTPUT_HI"].start + k]), 1) for k in range(16)]
        nz_lo = [(k, v) for k, v in enumerate(lo) if abs(v) > 0.5]
        nz_hi = [(k, v) for k, v in enumerate(hi) if abs(v) > 0.5]
        print(f"    {tag}: OUTPUT_LO nz={nz_lo}  OUTPUT_HI nz={nz_hi}")

    # Per-block OUTPUT trace around the L16 block for the carried row.
    print("### OUTPUT band trace (carried CMP row) around L16 block ###")
    for blk in range(read_block, min(l16_last + 3, len(rows_map))):
        res = residual_at_block(probe, ctx, blk, cmp_marker)
        show_output(res, f"after blk {blk}")
    print()

    for label, pos in (("PSH(fresh)", psh_marker), ("CMP(carried)", cmp_marker)):
        res = residual_at_block(probe, ctx, read_block, pos)
        lo_idx, lo_v = argmax16(res, _REG.slots["OUTPUT_LO"].start)
        hi_idx, hi_v = argmax16(res, _REG.slots["OUTPUT_HI"].start)
        out_byte = (hi_idx << 4) | lo_idx
        carried = round(float(res[b_carried]), 2)
        h1p = pos_band(res, b_h1_prev, 7)
        h3p = pos_band(res, b_h3_prev, 7)
        # PREV decode: H1_PREV idx = lo+2 -> lo = argmax-2 ; H3_PREV idx = hi+4
        h1p_arg = max(range(7), key=lambda k: h1p[k])
        h3p_arg = max(range(7), key=lambda k: h3p[k])
        prev_lo = h1p_arg - 2
        prev_hi = h3p_arg - 4
        prev_byte = (prev_hi << 4) | prev_lo if (prev_lo >= 0 and prev_hi >= 0) else None
        print(f"##### {label} pos={pos} #####")
        print(f"  OUTPUT band -> byte 0x{out_byte:02x}  "
              f"(LO_argmax={lo_idx}@{lo_v}, HI_argmax={hi_idx}@{hi_v})")
        print(f"  STACK0_B0_CARRIED = {carried}")
        print(f"  STACK0_B0_H1_PREV(lo+2) = {h1p}  argmax_idx={h1p_arg} -> lo={prev_lo}")
        print(f"  STACK0_B0_H3_PREV(hi+4) = {h3p}  argmax_idx={h3p_arg} -> hi={prev_hi}")
        if prev_byte is not None:
            print(f"  PREV-band decoded byte = 0x{prev_byte:02x}")
        print()


if __name__ == "__main__":
    main()
