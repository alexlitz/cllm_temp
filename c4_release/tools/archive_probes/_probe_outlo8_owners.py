#!/usr/bin/env python3
"""List every IR op (FFN rule OR attention W_o) that writes OUTPUT_LO[k].

Static ownership scan via attribute_residual_dim (covers attention.o, which the
runtime FFN attribution misses). Reveals the relay/head that over-amplifies
OUTPUT_LO[8] at the ADJ step.

Usage: python tools/_probe_outlo8_owners.py [k...]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)


def main():
    ks = [int(x) for x in sys.argv[1:]] or [6, 8]
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    for band in ("OUTPUT_LO", "OUTPUT_HI_THIS_STEP", "OUTPUT_HI"):
        base = dp.get(band)
        if base is None: continue
        for k in ks:
            col = base + k
            owners = ctx.interp.attribute_residual_dim(ctx.flat_ffn_ops, col)
            print(f"\n== {band}[{k}] (col {col}) writers ({len(owners)}) ==")
            seen = set()
            for opn, role in owners:
                key = (opn, role.split(' w=')[0])
                if key in seen: continue
                seen.add(key)
                if "attn.o" in role or "lev" in opn.lower() or "adj" in role.lower() or "passthrough" in role.lower() or "stack0" in role.lower() or "relay" in role.lower():
                    print(f"   {opn} :: {role}")


if __name__ == "__main__":
    main()
