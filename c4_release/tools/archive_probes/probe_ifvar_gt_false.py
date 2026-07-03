#!/usr/bin/env python3
"""if_var GT-FALSE step CMP cascade probe (campaign, spec_k=0).

Targets the ACTUAL residual if_var fails on main 03e3ea24 (full_trace, campaign):
  id430  ifVAR x=23, x>62  -> GT FALSE, expect return 0, NEURAL returns 1 (FAIL)
  id433  ifVAR x=35, x>76  -> GT FALSE, expect return 0, NEURAL returns 1 (FAIL)
Both have A.hi < B.hi (2<6, 3<7) so the `hi_lt` (CMP+0) 2-way GT override should
fire -> GT=0. It is NOT, so GT decodes 1. Reference PASS cases:
  id427  ifVAR x=85, x>48  -> GT TRUE,  return 1  (a passing if_var, A.hi 5>B.hi 4)
  if_gt literal  23>62     -> GT FALSE, return 0  (the literal reference good path)

Dumps SE_ALU/ALU operand nibbles + the CMP+k cascade + OUTPUT at the GT-step's
SE and AX rows across a block sweep, to localize WHY the hi_lt GT override does
not land for the LOADED-variable GT-false path.

  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_ifvargt python tools/probe_ifvar_gt_false.py [blocks...]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
    # FAIL: GT FALSE (A<B, A.hi<B.hi), neural returns 1.
    "ifVAR_23gt62_F": ("int main() { int x; x = 23; if (x > 62) return 1; return 0; }", "FAIL exp0"),
    "ifVAR_35gt76_F": ("int main() { int x; x = 35; if (x > 76) return 1; return 0; }", "FAIL exp0"),
    # PASS reference: GT TRUE if_var (A.hi>B.hi).
    "ifVAR_85gt48_T": ("int main() { int x; x = 85; if (x > 48) return 1; return 0; }", "PASS exp1"),
    # PASS reference: literal GT FALSE (the good path, same compare as 430 but immediates).
    "ifGT_23gt62_F":  ("int main() { if (23 > 62) return 1; return 0; }", "PASS exp0"),
}


def fmt(row, base, width=16, thr=0.4):
    vals = [(i, float(row[base + i].item())) for i in range(width)]
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in vals if abs(v) > thr) + "]"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    se_mark = dp.get("MARK_SE_ONLY")

    for pname, (src, tag) in PROGS.items():
        bc, data = compile_c(src)
        opc_ax, otok = runner._oracle_pc_ax_steps(
            bc, data or b"", "", expected_steps=None, with_tokens=True)
        prompt = runner._build_element(bc, data or b"", [], "", spec_k=1,
                                       adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in otok:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        # GT step = the step BEFORE the AX becomes the boolean (0/1) result.
        gt_step = None
        for s in range(1, len(opc_ax) - 1):
            if opc_ax[s + 1][1] in (0, 1) and opc_ax[s][1] not in (0, 1):
                gt_step = s
                break
        if gt_step is None:
            for s in range(len(opc_ax)):
                if opc_ax[s][1] in (0, 1):
                    gt_step = max(0, s - 1)
                    break
        print(f"\n=== {pname} [{tag}] GT_step={gt_step} "
              f"oracle[gt]={opc_ax[gt_step]} "
              f"oracle[gt+1]={opc_ax[gt_step+1] if gt_step+1<len(opc_ax) else None} ===",
              flush=True)
        with torch.no_grad():
            emb = model.embed(padded)[0]
        step_start = len(prefix) + gt_step * STEP
        rows = {}
        for off in range(STEP):
            r = step_start + off
            if r >= len(tape):
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
                           "AX_CARRY_LO", "AX_CARRY_HI",
                           "CMP", "OUTPUT_LO", "OUTPUT_HI"):
                    if dn in dp:
                        w = 8 if dn == "CMP" else 16
                        parts.append(f"{dn}={fmt(row, dp[dn], width=w)}")
                print(f"   [{rk} r{rr}] blk{b:2d} " + " ".join(parts), flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [13, 15, 17, 19, 22, 26]
    main(blks)
