#!/usr/bin/env python3
"""if_var GT-result residual probe over the REAL AR context (campaign, spec_k=0).

Unlike probe_ifvar_gt_operands.py (which teacher-forces the ORACLE tape and so
masks the AR divergence), this runs the production spec_k=0 AR decode
(_final_context) and then re-forwards the model on that REAL emitted context,
reading the ALU/SE_ALU/CMP/OUTPUT bands at the GT step rows. This exposes WHY
the AR decode lands AX=0 (GT result) for if_var while if_gt (literal) lands 1.

  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_pcframe python tools/probe_ifvar_ar_residual.py [blocks...]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PROGS = {
    "ifVAR_66gt24_T": ("int main() { int x; x = 66; if (x > 24) return 1; return 0; }", 10),
    "ifGT_66gt24_T":  ("int main() { if (66 > 24) return 1; return 0; }", 3),
}
# Override via PROBE_STEPS env: comma list of (prog:step) to focus the COMPARISON
# step instead of the result step. Default keeps (res_step-1, res_step).


def fmt(row, base, width=16, thr=0.4):
    vals = [(i, float(row[base + i].item())) for i in range(width)]
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in vals if abs(v) > thr) + "]"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    se_mark = dp.get("MARK_SE_ONLY")

    for pname, (src, res_step) in PROGS.items():
        bc, data = compile_c(src)
        ctx = p._final_context(bc, max_steps=25)
        prefix_len = len(p._build_context(bc))
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(padded)[0]
        # GT-result step is res_step; the row whose NEXT-step AX holds the result
        # is the AX row of res_step (PC advanced, AX = boolean). We probe the
        # rows of the GT *instruction* step (res_step-1) AND the result step.
        for stp in (res_step - 1, res_step):
            step_start = prefix_len + stp * STEP
            print(f"\n=== {pname} step {stp} (abs {step_start}) ===", flush=True)
            rows = {}
            for off in range(STEP):
                r = step_start + off
                if r >= len(ctx):
                    break
                if se_mark is not None and emb[r, se_mark].abs().item() > 0.5:
                    rows.setdefault("SE", r)
                if emb[r, dp["MARK_AX"]].abs().item() > 0.5:
                    rows.setdefault("AX", r)
            print(f"   rows: {rows}", flush=True)
            for rk, rr in rows.items():
                for b in blocks:
                    with torch.no_grad():
                        resid = model.forward(padded, stop_after_block=b)[0]
                    row = resid[rr]
                    parts = []
                    for dn in ("SE_ALU_LO", "SE_ALU_HI", "ALU_LO", "ALU_HI",
                               "AX_CARRY_LO", "AX_CARRY_HI", "CMP",
                               "OUTPUT_LO", "OUTPUT_HI"):
                        if dn in dp:
                            w = 8 if dn == "CMP" else 16
                            parts.append(f"{dn}={fmt(row, dp[dn], width=w)}")
                    print(f"   [{rk} r{rr}] blk{b:2d} " + " ".join(parts), flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [13, 14, 15]
    main(blks)
