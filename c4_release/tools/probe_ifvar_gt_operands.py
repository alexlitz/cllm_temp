#!/usr/bin/env python3
"""if_var GT-step operand + CMP cascade probe (campaign, spec_k=0).

At the GT step (PC=98 for ifv x>24), dump the SE_ALU_LO/HI operand nibbles and
the CMP+k cascade at the MARK_SE_ONLY row across a block sweep, to see whether
the LOADED-and-pushed operand-a (the variable x) reaches SE_ALU so the L9 CMP
cascade can fire the GT override. Compares against the literal if_gt path which
PASSES (same compare, immediate operands).

  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_pcframe python tools/probe_ifvar_gt_operands.py [blocks...]
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
    # if_var: variable x loaded, pushed, compared > literal. FAILS (GT result 0).
    "ifVAR_66gt24_T": ("int main() { int x; x = 66; if (x > 24) return 1; return 0; }", 9),
    # if_gt: both literals. PASSES (the reference good path).
    "ifGT_35gt43_F":  ("int main() { if (35 > 43) return 1; return 0; }", None),
    "ifGT_66gt24_T":  ("int main() { if (66 > 24) return 1; return 0; }", None),
}


def fmt(row, base, width=16, thr=0.5):
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

    for pname, (src, gt_step) in PROGS.items():
        bc, data = compile_c(src)
        opc_ax, otok = runner._oracle_pc_ax_steps(
            bc, data or b"", "", expected_steps=None, with_tokens=True)
        # locate the GT step: oracle step whose NEXT step's ax is the bool result.
        # Heuristic: the step at PC==98 for ifVAR; else find GT by opcode scan.
        # Use provided gt_step, else find step where pc just before a 1/0 boolean.
        prompt = runner._build_element(bc, data or b"", [], "", spec_k=1,
                                       adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in otok:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        # find GT step = the step BEFORE the AX becomes the boolean (0/1) result.
        if gt_step is None:
            # find first step s where opc_ax[s+1].ax in (0,1) and opc_ax[s].ax not in(0,1)
            gt_step = 0
            for s in range(1, len(opc_ax) - 1):
                if opc_ax[s + 1][1] in (0, 1) and opc_ax[s][1] not in (0, 1):
                    gt_step = s
                    break
        print(f"\n=== {pname} GT_step={gt_step} "
              f"oracle[gt]={opc_ax[gt_step]} oracle[gt+1]={opc_ax[gt_step+1] if gt_step+1<len(opc_ax) else None} ===",
              flush=True)
        # find the SE row in the GT step (or AX row).
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
                           "CMP", "OUTPUT_LO", "OUTPUT_HI"):
                    if dn in dp:
                        w = 8 if dn == "CMP" else 16
                        parts.append(f"{dn}={fmt(row, dp[dn], width=w)}")
                print(f"   [{rk} r{rr}] blk{b:2d} " + " ".join(parts), flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [13, 17, 19, 22]
    main(blks)
