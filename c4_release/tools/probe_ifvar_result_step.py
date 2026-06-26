#!/usr/bin/env python3
"""if_var GT-RESULT step (pc=106, the step AFTER the compare) CMP/OUTPUT probe.

The full_trace verdict (campaign, spec_k=0) on main 03e3ea24:
  id430 23>62 : step10 pc=106 got ax=1 want ax=0  (GT false -> wrongly returns 1)
  id433 35>76 : step10 pc=106 got ax=1 want ax=0
  id427 85>48 : PASS (GT true, ax=1 correct)
  ifGT literal 23>62 : PASS (GT false, ax=0 correct)

So the boolean RESULT is materialized at the step AFTER the compare (pc=106 for
ifVAR, pc=34 for the literal ifGT). This probe dumps CMP/OUTPUT/ALU at THAT
step's AX row across blocks to localize why the LOADED-var GT-false defaults to
1 while the literal GT-false (and the var GT-true) decode correctly.

  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_ifvargt python tools/probe_ifvar_result_step.py [blocks...]
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
    "id430_23gt62_F": ("int main() { int x; x = 23; if (x > 62) return 1; return 0; }", "FAIL exp0", 106),
    "id433_35gt76_F": ("int main() { int x; x = 35; if (x > 76) return 1; return 0; }", "FAIL exp0", 106),
    "id427_85gt48_T": ("int main() { int x; x = 85; if (x > 48) return 1; return 0; }", "PASS exp1", 106),
    "ifGT_23gt62_F":  ("int main() { if (23 > 62) return 1; return 0; }", "PASS exp0", 34),
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
        # locate the result step = the oracle step whose pc == result_pc.
        rstep = None
        for s, (pc, ax) in enumerate(opc_ax):
            if pc == result_pc:
                rstep = s
                break
        print(f"\n=== {pname} [{tag}] result_step={rstep} "
              f"oracle[r-1]={opc_ax[rstep-1] if rstep else None} "
              f"oracle[r]={opc_ax[rstep] if rstep is not None else None} ===", flush=True)
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
