#!/usr/bin/env python3
"""Inc-3: per-block OUTPUT scan at the AX byte-1 predictor row (step2 off6) for
var_simple_0. Find which physical block flips OUTPUT_LO to the strong negative
(the byte-0xFF one-hot selection) in GOLDEN, and whether it does in CAMPAIGN.
Pinpoints the owning op (via block_layer_map).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    dp = {}
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_layout.dim_positions)
    dim_names = {"OUTPUT_LO": dp["OUTPUT_LO"], "OUTPUT_LO+15": dp["OUTPUT_LO"] + 15,
                 "OUTPUT_LO+8": dp["OUTPUT_LO"] + 8, "OP_ENT": dp["OP_ENT"]}
    STEP = int(Token.STEP_TOKENS)
    nblk = len(p.runner.model.blocks)
    prompt_len = len(p._build_context(bytecode))
    pos = prompt_len + 2 * STEP + 6
    bl_map = p.block_layer_map()
    print(f"=== {cfg} STEP={STEP} blocks={nblk} pos={pos} (step2 off6) ===")
    # Build the final context ONCE, then one forward per block with the model's
    # stop_after_block probe kwarg (no re-decode per block).
    import torch
    ctx = p._final_context(bytecode, max_steps=6)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    prev = 0.0
    with torch.no_grad():
        for blk in range(nblk):
            resid = p.model.forward(padded, stop_after_block=blk)[0, pos]
            out = float(resid[dp["OUTPUT_LO"]].item())
            if abs(out - prev) > 50.0:
                lm = bl_map[blk]
                o15 = float(resid[dp["OUTPUT_LO"] + 15].item())
                o8 = float(resid[dp["OUTPUT_LO"] + 8].item())
                print(f" blk{blk:2d} (L{lm['logical']:2d} {lm['ffn'][:20]:20s}) "
                      f"OUTPUT_LO {prev:+12.1f} -> {out:+12.1f}  "
                      f"OUT15={o15:+10.1f} OUT8={o8:+10.1f}")
            prev = out


if __name__ == "__main__":
    main()
