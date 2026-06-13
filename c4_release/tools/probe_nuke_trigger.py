#!/usr/bin/env python3
"""Identify the block-38 tail_bit32 units that NUKE H1/H3 on the carried CMP row,
and what gates them (so we can decide whether re-pointing OUTPUT prevents it).

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_nuke_trigger.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch, torch.nn.functional as F
from src.compiler import compile_c
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

_REG = build_default_registry_dynamic()
STACK0_MARK = 268


def dn(t):
    if t.is_sparse:
        t = t.to_dense()
    if not t.is_contiguous():
        t = t.contiguous()
    return t.detach().float().cpu()


def main():
    p = GroundTruthProbe.build()
    # Use a FRESH dense (non-CSR/compact) model for weight introspection; the
    # probe's converted model leaves some W_gate as storageless.
    m_dense, _layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    m = p.model
    H1 = _REG.slots["H1"].start
    H3 = _REG.slots["H3"].start
    src = "int main() { if (17 > 35) return 1; return 0; }"
    bc = compile_c(src)[0]
    ctx = p._final_context(bc, max_steps=12)
    pl = len(p._build_context(bc))
    i, step, cmp = pl, 0, None
    while i < len(ctx):
        if ctx[i] == STACK0_MARK and step == 2 and cmp is None:
            cmp = i
        if ctx[i] == int(Token.STEP_END):
            step += 1
        i += 1
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    x = m.forward(padded, stop_after_block=37)
    r = dn(x[0, cmp])
    ffn = m_dense.blocks[38].ffn  # dense weights for introspection
    Wu = dn(ffn.W_up); Wg = dn(ffn.W_gate); Wd = dn(ffn.W_down)
    bu = ffn.b_up.float().cpu(); bg = ffn.b_gate.float().cpu()
    up = Wu @ r + bu
    gate = Wg @ r + bg
    h = F.silu(up) * gate
    lines = []
    for tgt, nm in ((H1 + 3, "H1+3"), (H3 + 5, "H3+5")):
        contrib = (Wd[tgt] * h)
        vals = contrib.tolist()
        idx = sorted(range(len(vals)), key=lambda u: abs(vals[u]), reverse=True)[:4]
        lines.append(f"{nm}(dim{tgt}): total={sum(vals):.0f}")
        for u in idx:
            # which residual dims drive this unit's up (silu condition)?
            wu_row = Wu[u]
            drivers = sorted(range(len(wu_row)),
                             key=lambda d: abs(float(wu_row[d]) * float(r[d])),
                             reverse=True)[:6]
            drv = []
            for d in drivers:
                w = float(wu_row[d]); rv = float(r[d])
                if abs(w * rv) < 0.5:
                    continue
                nmd = None
                for s, sl in _REG.slots.items():
                    if sl.start <= d < sl.start + sl.size:
                        nmd = f"{s}+{d - sl.start}"; break
                drv.append(f"{nmd or d}(w{w:.0f}*r{rv:.1f})")
            lines.append(
                f"  unit {u}: up={float(up[u]):.1f} silu={float(F.silu(up[u])):.2f} "
                f"gate={float(gate[u]):.1f} h={float(h[u]):.2f} "
                f"Wd={float(Wd[tgt, u]):.0f} contrib={vals[u]:.0f}")
            lines.append(f"      up-drivers: {drv}")
    out = "\n".join(lines)
    open(os.path.join(_HERE, "_nuke.out"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
