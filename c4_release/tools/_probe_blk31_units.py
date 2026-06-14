#!/usr/bin/env python3
"""Find which block-31 (logical L20) FFN units drive output dims 84 & 100
(low-nibble-15 / high-nibble-15 = the 0xFF byte-token decode) and what they read.

For the spurious-255 row of func_identity_0, computes the block's FFN forward,
finds the hidden units whose W_down contribution to dims 84/100 is largest at
that row, and reports each unit's W_up gate (which input dims it reads) so we can
identify the sentinel gate and a discriminator dim.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa


def _dense(t):
    try:
        if t.layout != torch.strided:
            return t.to_dense()
    except Exception:
        pass
    return t


BLK = int(os.environ.get("BLK", "31"))
OUT_DIMS = [84, 100]


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; device = probe._device
    SE = int(Token.STEP_END)

    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    spurious = [i for i in range(pl, len(ctx)) if ctx[i] == 255 and ctx[i-1] == SE]
    print(f"id{pid} {desc} spurious={spurious} blk={BLK}")
    emit_row = spurious[0] - 1

    padded = torch.tensor([ctx], dtype=torch.long, device=device)
    # input to block BLK = residual after block BLK-1
    x_in = _dense(model.forward(padded, stop_after_block=BLK - 1))[0, emit_row]  # [D]

    blk = model.blocks[BLK]
    ffn = blk.ffn
    Wup = _dense(ffn.W_up);   bup = _dense(ffn.b_up)     # [H,D],[H]
    Wgate = _dense(ffn.W_gate); bgate = _dense(ffn.b_gate)
    Wdown = _dense(ffn.W_down); bdown = _dense(ffn.b_down)  # [D,H]

    # FFN forward (GLU-ish: try act(W_up x + b)*(W_gate x + b) -> W_down)
    up = Wup @ x_in + bup        # [H]
    gate = Wgate @ x_in + bgate  # [H]
    # Guess activation: relu on up * gate (common). Reproduce numerically below.
    # We just want which hidden units make dims 84/100 large. Use hidden = up*sigmoid?
    # To be exact, ask the block to give us pre-down hidden via two forwards is hard;
    # instead approximate contribution = Wdown[d, h] * h_act. We'll try a few acts.
    import torch.nn.functional as F
    acts = {
        "silu(up)*gate": F.silu(up) * gate,
        "relu(up)*gate": F.relu(up) * gate,
        "up*gate": up * gate,
        "relu(up)": F.relu(up),
    }
    # Determine which activation reproduces the block's true dim-84/100 delta.
    r_after = _dense(model.forward(padded, stop_after_block=BLK))[0, emit_row]
    target = {d: float(r_after[d]) - float(x_in[d]) for d in OUT_DIMS}
    print(f"  true block delta dims {OUT_DIMS}: {target}")
    best = None
    for nm, h in acts.items():
        for d in OUT_DIMS:
            pred = float(Wdown[d] @ h) + float(bdown[d])
            # delta vs input contribution of residual stream (down only adds)
        # compute predicted post = x_in + Wdown@h + bdown
        post = x_in + (Wdown @ h) + bdown
        err = sum(abs(float(post[d]) - float(r_after[d])) for d in OUT_DIMS)
        print(f"  act={nm:16s} pred dims {OUT_DIMS} = "
              f"{[round(float(post[d]),1) for d in OUT_DIMS]}  err={err:.2f}")
        if best is None or err < best[1]:
            best = (nm, err, h)
    nm, err, h = best
    print(f"  --> activation '{nm}' (err {err:.2f}); using it for unit attribution")

    for d in OUT_DIMS:
        contrib = Wdown[d] * h  # [H] per-unit contribution to dim d
        top = torch.topk(contrib.abs(), k=8)
        print(f"\n  dim {d}: top units by |contribution|:")
        for v, u in zip(top.values.tolist(), top.indices.tolist()):
            u = int(u)
            print(f"    unit {u:5d}: down={float(Wdown[d,u]):+.3f} h={float(h[u]):+.4e} "
                  f"contrib={float(contrib[u]):+.4e}")
            # what does this unit READ (W_up gate dims)?
            wu = Wup[u]; nzu = (wu.abs() > 1e-6).nonzero().flatten().tolist()
            wg = Wgate[u]; nzg = (wg.abs() > 1e-6).nonzero().flatten().tolist()
            print(f"        W_up reads dims: {[(dd, round(float(wu[dd]),2)) for dd in nzu][:10]} b_up={float(bup[u]):+.2f}")
            print(f"        W_gate reads dims: {[(dd, round(float(wg[dd]),2)) for dd in nzg][:10]} b_gate={float(bgate[u]):+.2f}")
            print(f"        x_in at those dims: up={[round(float(x_in[dd]),2) for dd in nzu][:10]} "
                  f"gate={[round(float(x_in[dd]),2) for dd in nzg][:10]}")


if __name__ == "__main__":
    main()
