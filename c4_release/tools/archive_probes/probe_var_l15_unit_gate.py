#!/usr/bin/env python3
"""Per-unit up/gate/hidden dump for block-30 L15 PSH-byte units at the var
leak row 211 and the comparison rows 210/212. Shows WHY unit 32/35/37/39 fire.
READ-ONLY, spec_k=0.

Usage: python tools/probe_var_l15_unit_gate.py [id] [block]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import torch
import torch.nn.functional as F
from tools.probe_groundtruth import build_groundtruth_probe
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs

UNIT_NAMES = {
    32: "psh_sp_byte1_lo_ff", 33: "psh_sp_byte1_hi_ff",
    34: "psh_sp_byte2_lo_00", 35: "psh_sp_byte2_hi_00",
    36: "psh_sp_byte3_lo_00", 37: "psh_sp_byte3_hi_00",
    38: "psh_bp_byte2_lo_01", 39: "psh_bp_byte2_hi_00",
}


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
    blk = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rows = [int(a) for a in sys.argv[3:]] or [210, 211, 212]
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)
    probe = build_groundtruth_probe()
    m = probe.model
    dev = next(m.parameters()).device
    ctx = probe._final_context(bc)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        x_in = m.forward(padded, stop_after_block=blk - 1)[0].float()

    ffn = m.blocks[blk].ffn
    def dn(p):
        p = p.data
        if p.layout != torch.strided:
            p = p.to_dense()
        return p.float()
    Wu = dn(ffn.W_up); Wg = dn(ffn.W_gate)
    bu = dn(ffn.b_up); bg = dn(ffn.b_gate)

    print(f"id={idx} {desc} block={blk}")
    for r in rows:
        x = x_in[r].to(Wu.device)
        up = (Wu @ x) + bu
        gate = (Wg @ x) + bg
        hidden = F.silu(up) * gate
        print(f"\n--- row {r} (tok={ctx[r]}) ---")
        print(f"{'unit':>5} {'name':<22} {'up(preact)':>12} {'silu(up)':>10} "
              f"{'gate':>10} {'hidden':>12}")
        for u in sorted(UNIT_NAMES):
            print(f"{u:>5} {UNIT_NAMES[u]:<22} {float(up[u]):>12.3f} "
                  f"{float(F.silu(up[u])):>10.4f} {float(gate[u]):>10.3f} "
                  f"{float(hidden[u]):>12.4f}")


if __name__ == "__main__":
    main()
