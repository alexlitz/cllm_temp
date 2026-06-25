#!/usr/bin/env python3
"""Probe SE_OP_GT / SE_CMP_GROUP / OP_GT at the GT-result SE row (real AR ctx).

The cmp_combine GT default (writes OUTPUT_LO+1 = result 1) gates on SE_OP_GT.
The probe shows that for if_var the GT default does NOT fire (OUTPUT_LO cell1
untouched) while cell0 (result 0) gets +5. This dumps the gating flags to find
WHY the default is skipped / why cell0 is written for the variable path.

  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_pcframe python tools/probe_ifvar_se_op_gt.py
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
DIMS = ("SE_OP_GT", "SE_OP_LT", "SE_CMP_GROUP", "SE_CMP", "OP_GT",
        "MARK_SE_ONLY", "MARK_PC", "MARK_AX")
BLOCKS = (13, 14, 15, 16, 17, 19)


def main():
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
        step_start = prefix_len + res_step * STEP
        # find SE row
        se_row = None
        for off in range(STEP):
            r = step_start + off
            if r < len(ctx) and se_mark is not None and emb[r, se_mark].abs().item() > 0.5:
                se_row = r
                break
        print(f"\n=== {pname} result_step={res_step} SE_row={se_row} ===", flush=True)
        for b in BLOCKS:
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=b)[0]
            row = resid[se_row]
            vals = []
            for dn in DIMS:
                if dn in dp:
                    base = dp[dn]
                    # scalar dims: read [base]; CMP is a small band
                    if dn in ("SE_CMP_GROUP", "SE_OP_GT", "SE_OP_LT", "SE_CMP",
                              "MARK_SE_ONLY", "MARK_PC", "MARK_AX", "OP_GT"):
                        vals.append(f"{dn}={float(row[base].item()):.2f}")
            print(f"   blk{b:2d} " + " ".join(vals), flush=True)


if __name__ == "__main__":
    main()
