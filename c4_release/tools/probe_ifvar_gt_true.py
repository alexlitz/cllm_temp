#!/usr/bin/env python3
"""if_var GT-TRUE result-step CMP/OUTPUT probe (the 6 remaining if_var fails).

The 6 remaining if_var fails (425/436/440/441/445/448) are all GT-TRUE of a
LOADED variable (``x>k`` true -> ax should be 1, decodes 0). The symmetric
GT-FALSE (430/433) was fixed by C4_CMP_HI_LT_ALU15_GUARD. This probe dumps the
CMP cascade + OUTPUT/ALU at the GT-RESULT step's AX row (pc=106) across blocks
so we can see WHICH override spuriously fires (or which result cell is crushed)
for the GT-TRUE loaded-var case.

  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_ifvargtt python tools/probe_ifvar_gt_true.py [blocks...]
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


def _src(x, k):
    return f"int main() {{ int x; x = {x}; if (x > {k}) return 1; return 0; }}"


PROGS = {
    # the 6 GT-TRUE fails (A.hi > B.hi -> GT default 1 expected)
    "id425_96gt59_T": (_src(96, 59), "FAIL exp1", 106),
    "id436_66gt24_T": (_src(66, 24), "FAIL exp1", 106),
    "id440_97gt30_T": (_src(97, 30), "FAIL exp1", 106),
    "id441_88gt60_T": (_src(88, 60), "FAIL exp1", 106),
    "id445_37gt28_T": (_src(37, 28), "FAIL exp1", 106),
    "id448_84gt74_T": (_src(84, 74), "FAIL exp1", 106),
    # passing reference GT-TRUE (same A.hi>B.hi shape)
    "id427_85gt48_T": (_src(85, 48), "PASS exp1", 106),
    # passing GT-FALSE refs (fixed by alu15 guard)
    "id430_23gt62_F": (_src(23, 62), "PASS exp0", 106),
    # passing literal GT-TRUE
    "ifGT_85gt48_T":  ("int main() { if (85 > 48) return 1; return 0; }", "PASS exp1", 34),
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

    for pname, (src, tag, result_pc) in PROGS.items():
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
        rstep = None
        for s, (pc, ax) in enumerate(opc_ax):
            if pc == result_pc:
                rstep = s
                break
        print(f"\n=== {pname} [{tag}] result_step={rstep} "
              f"oracle[r-1]={opc_ax[rstep-1] if rstep else None} "
              f"oracle[r]={opc_ax[rstep] if rstep is not None else None} ===", flush=True)
        if rstep is None:
            print("   (no result step pc match)", flush=True)
            continue
        with torch.no_grad():
            emb = model.embed(padded)[0]
        step_start = len(prefix) + rstep * STEP
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
                for dn in ("ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                           "CMP", "OUTPUT_LO", "OUTPUT_HI"):
                    if dn in dp:
                        w = 8 if dn == "CMP" else 16
                        parts.append(f"{dn}={fmt(row, dp[dn], width=w)}")
                print(f"   [{rk} r{rr}] blk{b:2d} " + " ".join(parts), flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [9, 13, 17, 22, 26, 30, 33, 36]
    main(blks)
