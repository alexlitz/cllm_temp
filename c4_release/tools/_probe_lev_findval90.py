#!/usr/bin/env python3
"""Find where the return PC value (90=0x5a) lives in the campaign frame, and
dump ALL store/MEM rows with their addr/value/opcode flags so we can see if the
JSR return-address is materialized anywhere at all.

Usage: python tools/_probe_lev_findval90.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    L15 = [i for i, b in enumerate(model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    xin = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    cl_lo = dimp["CLEAN_EMBED_LO"]; cl_hi = dimp["CLEAN_EMBED_HI"]
    def hot(p, b): return int(torch.argmax(xin[p, b:b + 16]).item())
    print(f"STEP_TOKENS={int(Token.STEP_TOKENS)} ctxlen={len(ctx)} desc={desc}")
    print("Looking for value 90 (0x5a) on any store/MEM row...")
    print(f"{'pos':>4} {'tok':>4} {'val':>4} {'addr':>6} {'MSTORE':>6} {'STK0':>5} "
          f"{'BIDX':>4} OP_JSR OP_ENT MARK_MEM CLEAN")
    for p in range(len(ctx)):
        mstore = float(xin[p, dimp["MEM_STORE"]].item()) if "MEM_STORE" in dimp else 0
        if mstore < 0.3:
            continue
        clv = hot(p, cl_lo) | (hot(p, cl_hi) << 4)
        a = (hot(p, dimp["ADDR_B0_LO"]) | (hot(p, dimp["ADDR_B0_HI"]) << 4)) | \
            ((hot(p, dimp["ADDR_B1_LO"]) | (hot(p, dimp["ADDR_B1_HI"]) << 4)) << 8)
        stk0 = float(xin[p, dimp["STACK0_BYTE0"]].item()) if "STACK0_BYTE0" in dimp else 0
        bidx = "".join(str(i) for i in range(4) if dimp.get(f"BYTE_INDEX_{i}") and xin[p, dimp[f"BYTE_INDEX_{i}"]].item() > 0.5)
        opjsr = float(xin[p, dimp["OP_JSR"]].item()) if "OP_JSR" in dimp else 0
        opent = float(xin[p, dimp["OP_ENT"]].item()) if "OP_ENT" in dimp else 0
        mm = float(xin[p, dimp["MARK_MEM"]].item()) if "MARK_MEM" in dimp else 0
        mark = "  <-- VAL90" if clv == 90 else ("  <-- VAL240" if clv == 240 else "")
        print(f"{p:>4} {ctx[p]:>4} {clv:>4} 0x{a:04x} {mstore:>6.2f} {stk0:>5.1f} "
              f"{bidx:>4} {opjsr:>6.1f} {opent:>6.1f} {mm:>8.1f}{mark}")


if __name__ == "__main__":
    main()
