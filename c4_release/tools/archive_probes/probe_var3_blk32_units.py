#!/usr/bin/env python3
"""Attribute the block-32 OUTPUT_HI_THIS_STEP+1 over-write at the var_three
step-0 AX byte-0 EMIT row to its firing FFN units (campaign vs golden).

Reads the post-block-31 residual (input to block 32), computes block-32 PureFFN
SwiGLU per-unit hidden activations, ranks units by their contribution to
OUTPUT_HI_THIS_STEP+{0,1,2,3}, and prints the top contributors in BOTH frames.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }"
BLK = int(os.environ.get("PROBE_BLK", "32"))


def main():
    STEP = int(Token.STEP_TOKENS)
    print(f"STEP_TOKENS={STEP}  block={BLK}")
    _, layout = compile_full_vm_dynamic()
    dp = layout.dim_positions
    out_hi = dp["OUTPUT_HI_THIS_STEP"]

    bytecode, _ = compile_c(SRC)
    probe = build_groundtruth_probe()
    ctx = probe._final_context(bytecode, max_steps=2)
    prompt_len = len(probe._build_context(bytecode))
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)

    # locate AX marker -> emit row = marker+1 (AX byte0 row)
    with torch.no_grad():
        rm = probe.model.forward(padded, stop_after_block=12)[0]
    mark_ax = dp["MARK_AX"]
    ax_pos = None
    for j in range(STEP):
        pos = prompt_len + j
        if float(rm[pos][mark_ax]) > 0.5:
            ax_pos = pos
            break
    emit_pos = ax_pos + 1
    print(f"AX marker pos={ax_pos}, emit (AX byte0) row pos={emit_pos}, "
          f"emitted byte1 tok={ctx[ax_pos + 2]} (0x{ctx[ax_pos + 2]:02x})")

    # residual INPUT to block BLK = post-block (BLK-1)
    with torch.no_grad():
        resid_in = probe.model.forward(padded, stop_after_block=BLK - 1)[0]
    x = resid_in[emit_pos]  # [D]

    ffn = probe.model.blocks[BLK].ffn
    Wg, Wu, Wd = ffn.W_gate, ffn.W_up, ffn.W_down
    bg = ffn.b_gate if ffn.b_gate is not None else 0.0
    bu = ffn.b_up if ffn.b_up is not None else 0.0
    print("shapes Wg", tuple(Wg.shape), "Wu", tuple(Wu.shape), "Wd", tuple(Wd.shape))
    # PureFFN: hidden = silu(Wg@x + bg) * (Wu@x + bu); out = Wd@hidden
    # Wg/Wu shape [H, D]; Wd shape [D, H]
    with torch.no_grad():
        gate = F.silu(Wg @ x + bg)
        up = Wu @ x + bu
        hidden = gate * up           # [H]
        # contribution of each unit to OUTPUT_HI+k = hidden[h] * Wd[dim, h]
        for k in (0, 1, 2, 3):
            dim = out_hi + k
            contrib = hidden * Wd[dim, :]   # [H]
            top = torch.topk(contrib.abs(), 8)
            print(f"\n-- OUTPUT_HI_THIS_STEP+{k} (dim {dim}) top unit contributions --")
            for idx in top.indices.tolist():
                print(f"   unit {idx:5d}: contrib={float(contrib[idx]):+9.3f} "
                      f"hidden={float(hidden[idx]):+8.3f} Wd={float(Wd[dim, idx]):+7.3f}")


if __name__ == "__main__":
    main()
