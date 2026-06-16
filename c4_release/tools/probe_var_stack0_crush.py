#!/usr/bin/env python3
"""var step-5 STACK0[0] OUTPUT-crush attribution (spec_k=0, built dim layout).

Confirms the two-attractor OUTPUT-band crush at the IMM-assign STACK0[0] row:
- decode the free-run context, locate the failing STACK0[0] position;
- show the LM-head argmax there (the stray marker that wins);
- trace OUTPUT_LO/HI nibble bands pre-L20 / post-L20(block32) / post-L25(block41);
- attribute L20 (layer16_lev_routing) and L25 (tail_stack0_pop_loaded) FFN unit
  contributions to the OUTPUT band using silu(up)*gate.

ALL dims resolved from the BUILT layout (compile_full_vm_dynamic()[1].dim_positions).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe

SRC = "int main() { int x; x = 990; return x; }"


def dense(w):
    return w.to_dense() if w.layout != torch.strided else w


def main():
    p = build_groundtruth_probe()
    model = p.model
    dev = p._device

    # Built dim layout.
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    _m, layout = compile_full_vm_dynamic(disk_cache=True)
    dimpos = layout.dim_positions
    OUT_LO = [dimpos["OUTPUT_LO"] + k for k in range(16)]
    OUT_HI = [dimpos["OUTPUT_HI_THIS_STEP"] + k for k in range(16)]
    print("OUTPUT_LO base", OUT_LO[0], "OUTPUT_HI_THIS_STEP base", OUT_HI[0])

    # Physical blocks for L20 / L25.
    blm = p.block_layer_map()
    L20 = next(b["physical"] for b in blm if b["logical"] == 20)
    L25 = next(b["physical"] for b in blm if b["logical"] == 25)
    print(f"L20 phys block {L20}  L25 phys block {L25}")

    bc = compile_c(SRC)[0]
    plen = len(p._build_context(bc))
    ctx = p._final_context(bc, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    # The assign step is step 5; STACK0[0] is off 21 within the 35-frame.
    STEP = 35
    step = 5
    off = 21  # STACK0[0]
    # The PRED position whose logits emit the STACK0[0] token = previous index.
    emit_idx = plen + step * STEP + off
    pred = emit_idx - 1
    print(f"\nemit_idx={emit_idx} (step{step} off{off} STACK0[0]) pred={pred}")
    print("emitted token at STACK0[0]:", ctx[emit_idx])

    # LM-head argmax at pred (what actually gets emitted at STACK0[0]).
    logits = p._forward_logits(ctx)  # [S, V]
    row_logits = logits[pred]
    top = torch.argsort(row_logits, descending=True)[:8].tolist()
    print("\nTop-8 LM-head logits at the STACK0[0] emit position:")
    for t in top:
        m = " MARKER" if t in (257, 258, 259, 260, 261, 262, 268) else ""
        print(f"  tok={t:4d} logit={float(row_logits[t]):+9.2f}{m}")

    def band_vals(resid_row, base):
        return [round(float(resid_row[d]), 2) for d in base]

    # Trace the OUTPUT band across a wide block range to find the crush onset.
    for blk in range(24, len(model.blocks)):
        resid = model.forward(padded, stop_after_block=blk)[0]
        r = resid[pred].detach().float()
        lo = band_vals(r, OUT_LO)
        hi = band_vals(r, OUT_HI)
        lo_max = max(range(16), key=lambda k: r[OUT_LO[k]])
        hi_max = max(range(16), key=lambda k: r[OUT_HI[k]])
        lomx = float(r[OUT_LO[lo_max]]); himx = float(r[OUT_HI[hi_max]])
        lomn = min(float(r[d]) for d in OUT_LO)
        himn = min(float(r[d]) for d in OUT_HI)
        lg = next((b["logical"] for b in blm if b["physical"] == blk), "?")
        print(f"  blk {blk:2d} (L{lg}): LO argmax n{lo_max} {lomx:+8.1f} "
              f"(min {lomn:+8.1f})  HI argmax n{hi_max} {himx:+8.1f} (min {himn:+8.1f})")

    # Attribute L20 + L25 unit contributions to the OUTPUT band.
    for label, blk in (("L20", L20), ("L25", L25)):
        xin = model.forward(padded, stop_after_block=blk - 1)[0]
        x = xin[pred].detach().float()
        ffn = model.blocks[blk].ffn
        if not hasattr(ffn, "W_up"):
            print(f"\n{label} block {blk} has no PureFFN W_up; skip")
            continue
        Wup = dense(ffn.W_up).detach().float()
        Wgate = dense(ffn.W_gate).detach().float()
        Wdown = dense(ffn.W_down).detach().float()
        bup = ffn.b_up.detach().float()
        bgate = ffn.b_gate.detach().float()
        up = Wup @ x + bup
        gate = Wgate @ x + bgate
        hidden = F.silu(up) * gate  # [H]
        # Total contribution magnitude to the whole OUTPUT band.
        band_dims = OUT_LO + OUT_HI
        contrib = torch.zeros_like(hidden)
        for d in band_dims:
            contrib = contrib + Wdown[d] * hidden
        order = torch.argsort(contrib.abs(), descending=True)
        print(f"\n=== {label} block {blk}: top units by |Σ OUTPUT-band contribution| ===")
        for h in order[:12].tolist():
            c = float(contrib[h])
            if abs(c) < 0.5:
                break
            # which output dim does it hit hardest
            col = torch.argmax((Wdown[:, h].abs() * (
                torch.tensor([1.0 if i in band_dims else 0.0
                              for i in range(Wdown.shape[0])])
            )))
            print(f"  unit{h:5d}: Σband {c:+9.2f}  hidden={float(hidden[h]):+8.2f} "
                  f"up={float(up[h]):+8.2f} gate={float(gate[h]):+8.2f}")


if __name__ == "__main__":
    main()
