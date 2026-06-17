#!/usr/bin/env python3
"""Pin the step-9 LEV-epilogue over-emit (stray leading 0) for func_identity_0.

Free-runs (spec_k=0) to the LEV STEP_END row + the next-step PC marker row, then:
  * decodes OP_LEV / HAS_SE residual at those rows across blocks
  * captures block-36 (logical L22) FFN hidden activations (silu(up)*gate per unit)
  * reports which units fire and what they write into the OUTPUT band

Usage: python tools/_probe_lev_step9_overemit.py [id] [lev_step] [target_blk]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
import torch.nn.functional as F
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    lev_step = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    target_blk = int(sys.argv[3]) if len(sys.argv) > 3 else 36
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    OP_LEV = dp["OP_LEV"]; HAS_SE = dp["HAS_SE"]
    # named OUTPUT dims for attribution
    out_dims = {k: v for k, v in dp.items() if "OUTPUT" in k}
    ctx = probe._final_context(bc, max_steps=14)
    pl = len(probe._build_context(bc))
    steps = []; cur = []
    for p in range(pl, len(ctx)):
        cur.append(p)
        if ctx[p] == SE:
            steps.append(cur); cur = []
    print(f"id{pid} {desc} exp={exp} nsteps={len(steps)} target_blk={target_blk}")
    lev_se_row = steps[lev_step][-1]
    next_pc_row = steps[lev_step + 1][0] if lev_step + 1 < len(steps) else None
    # a "normal" STEP_END for comparison (step 7's STEP_END)
    norm_se_row = steps[lev_step - 1][-1]
    print(f"LEV STEP_END row={lev_se_row}  next-PC row={next_pc_row}  normal-SE row={norm_se_row}")

    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)

    # hook the target block's ffn to grab its input x
    captured = {}
    ffn = probe.model.blocks[target_blk].ffn
    def pre(m, i): captured["x"] = i[0].detach()
    h = ffn.register_forward_pre_hook(pre)
    _ = probe.model.forward(padded)
    h.remove()
    X = captured["x"][0]  # [seq, d_model]
    def dense(w):
        return w.to_dense() if (w.is_sparse or getattr(w, "is_sparse_csr", False)) else w
    Wu = dense(ffn.W_up); Wg = dense(ffn.W_gate); Wd = dense(ffn.W_down)
    bu = ffn.b_up; bg = ffn.b_gate

    rows = {"LEV_SE": lev_se_row, "NORMAL_SE": norm_se_row}
    if next_pc_row is not None:
        rows["NEXT_PC"] = next_pc_row

    for name, row in rows.items():
        x = X[row]
        up = Wu @ x + (bu if bu is not None else 0)
        gate = Wg @ x + (bg if bg is not None else 0)
        act = F.silu(up) * gate  # [hidden]
        # OUTPUT-band contribution per unit = Wd[out_dim, unit] * act[unit]
        topk = torch.topk(act.abs(), 10)
        print(f"\n[{name} row={row}] OP_LEV(in)={x[OP_LEV].item():+.3f} HAS_SE(in)={x[HAS_SE].item():+.3f}  nunits={act.shape[0]}")
        for ui in topk.indices.tolist():
            wd_col = Wd[:, ui]
            big = torch.nonzero(wd_col.abs() > 1.0).flatten().tolist()
            ow = []
            for d in big[:5]:
                nm = next((k for k, v in dp.items() if v == d), str(d))
                ow.append(f"{nm}:{wd_col[d].item()*act[ui].item():+.2e}")
            print(f"  unit{ui:4d}: act={act[ui].item():+.3e} up={up[ui].item():+.2f} gate={gate[ui].item():+.2f} "
                  f"W_up[OP_LEV]={Wu[ui,OP_LEV].item():+.0f} W_up[HAS_SE]={Wu[ui,HAS_SE].item():+.0f} -> {ow}")

    # OUTPUT band residual at the OUTPUT of target block, all three rows
    print("\n== OUTPUT band (post-block) at the three rows ==")
    out = probe.model.forward(padded, stop_after_block=target_blk)[0]
    for name, row in rows.items():
        r = out[row]
        vals = {k: round(r[v].item(), 1) for k, v in sorted(out_dims.items()) if abs(r[v].item()) > 50}
        print(f"  [{name}] {vals}")


if __name__ == "__main__":
    main()
