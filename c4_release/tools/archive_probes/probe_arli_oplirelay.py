#!/usr/bin/env python3
"""AR-LI: where is OP_LI set, and does OP_LI_RELAY reach the AX byte-0 predictor
row, on the TRUE AR tape (campaign)? Dumps OP_LI / OP_LI_RELAY / MARK_AX / H1[AX]
per-row across the LI step, after the L7 relay block, and the L5 opcode block.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { int x; x = 28; return x; }")
LI_STEP = int(os.environ.get("PROBE_LI_STEP", "7"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30)" if nostk else "GOLDEN(35)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    prompt_len = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=LI_STEP + 1)
    padded = torch.tensor([ctx], device=p._device)
    blk_map = p.block_layer_map()
    nblk = len(p.model.blocks)

    s = prompt_len + LI_STEP * STEP
    AX_I = 1
    dims = {
        "OP_LI": dp["OP_LI"], "OP_LI_RELAY": dp["OP_LI_RELAY"],
        "MARK_AX": dp["MARK_AX"], "H1[AX]": dp["H1"] + AX_I,
        "OP_IMM": dp["OP_IMM"], "OP_LEA": dp["OP_LEA"],
    }
    # Find L5 and L7 physical blocks.
    l5 = [b for b in range(nblk) if blk_map[b].get("logical") == 5]
    l7 = [b for b in range(nblk) if blk_map[b].get("logical") == 7]
    print(f"=== {cfg} STEP={STEP} LI_STEP={LI_STEP} L5={l5} L7={l7} ===")
    print(f"  LI step tokens: {ctx[s:s+STEP]}")
    for label, blk in [("after L5", max(l5)), ("after L7", max(l7))]:
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        print(f"  --- {label} (blk{blk}) per-row dims at LI step ---")
        for o in range(STEP):
            row = x[s + o]
            vals = " ".join(f"{k}={float(row[d]):+.2f}" for k, d in dims.items())
            tok = ctx[s + o]
            print(f"    off{o:2d} tok={tok:3d}: {vals}")


if __name__ == "__main__":
    main()
