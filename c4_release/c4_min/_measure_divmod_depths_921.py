"""#921 MEASURE — the ACTUAL assembled block counts + peak residual dim of every
divmod construction, and the base (non-divmod) whole-ISA pipeline depth.

Off-build-path measurement ONLY (block-specs, NO dense model materialised).
Golden 174ece66 untouched.  Reports MEASURED numbers, never projections.

Run:  PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._measure_divmod_depths_921
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple


def _specs(div_logsink: bool, lean: bool = True):
    import c4_min.qwen_full_vm as Q
    saved = {}
    env = {"C4_LOGSINK_DIV": "1" if div_logsink else "0"}
    if not div_logsink:
        env["C4_DIV_LEAN"] = "1" if lean else "0"
    for k, v in env.items():
        saved[k] = os.environ.get(k); os.environ[k] = v
    try:
        subset = Q.SUBSET_FULL
        QL = Q.QwenFullLayout(24, subset, efficient_alu=True, recurrent_divmod=False,
                              code_from_memory=True, shift_via_mul=True,
                              div_logsink=div_logsink)
        L = QL.L
        specs = Q._block_specs(L, 24, subset, efficient_alu=True, recurrent_divmod=False,
                               code_from_memory=True, shift_via_mul=QL.shift_via_mul,
                               div_logsink=QL.div_logsink)
        names = [nm for nm, _ in specs]
        max_units = max(int(s["W_up"].shape[0]) for _, s in specs)
        mux = names.index("alu-ax-mux")
        expand = names.index("alu-expand")
        aluspan = names[expand + 1:mux]
        if div_logsink:
            div_names = [n for n in aluspan if n.startswith("ls")]
            mul_names = [n for n in aluspan if n not in div_names]
        else:
            mul_names = [n for n in aluspan if n.startswith("mul") or n == "alu-carry"]
            div_names = [n for n in aluspan if n not in mul_names]
        base = len(names) - len(div_names)
        return dict(whole_isa=len(names), divmod=len(div_names), base=base,
                    mul=len(mul_names), D_used=QL.D_used, max_ffn_units=max_units,
                    div_first=div_names[0] if div_names else "-",
                    div_last=div_names[-1] if div_names else "-",
                    all_names=names, div_names=div_names)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    import json
    print("=== radix16-lean (fp32 DEFAULT DIV) ===")
    r = _specs(div_logsink=False, lean=True)
    print(json.dumps({k: v for k, v in r.items() if k not in ("all_names", "div_names")}, indent=2))
    print("=== logsink (fp64) ===")
    s = _specs(div_logsink=True)
    print(json.dumps({k: v for k, v in s.items() if k not in ("all_names", "div_names")}, indent=2))
    # base pipeline (non-divmod) breakdown for the reorder analysis
    print("\n=== base (non-divmod) block names, radix16-lean build ===")
    base_names = [n for n in r["all_names"] if n not in set(r["div_names"])]
    print(f"  count={len(base_names)}")
    for n in base_names:
        print("   ", n)
