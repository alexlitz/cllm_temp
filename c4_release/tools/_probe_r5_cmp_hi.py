#!/usr/bin/env python3
"""R5 cmp-combine (hi<<4) leak probe over func_max/min + bool_and, TEACHER-FORCED.

Contention-robust: ONE build, teacher-forced oracle tape (no AR replay),
a handful of truncated forwards. Dumps CMP + OUTPUT_LO/HI at EVERY AX/SE
row of EVERY step so the branch-decision (compare) step's result byte is
visible: emitted byte = OUTPUT_LO_arg | (OUTPUT_HI_arg << 4). The R5 wall
is a leaked operand HIGH nibble that turns the boolean compare result into
(hi<<4)|result at the branch-decision step.

Run:
  CUDA_VISIBLE_DEVICES="" C4_VM_CACHE_DIR=/tmp/c4cache_cmpmargin \
    python tools/_probe_r5_cmp_hi.py 26 28 30 33 36
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_cmpmargin")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "6")))
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PROGS = {
    "func_max_36_99": ("int max(int a, int b) { if (a > b) return a; return b; }"
                       " int main() { return max(36, 99); }", 99),
    "func_max_54_44": ("int max(int a, int b) { if (a > b) return a; return b; }"
                       " int main() { return max(54, 44); }", 54),
    "func_min_13_57": ("int min(int a, int b) { if (a < b) return a; return b; }"
                       " int main() { return min(13, 57); }", 13),
    "bool_and_96_95": ("int main() { if (96 > 95) { if (95 > 100) return 1; }"
                       " return 0; }", 0),
    "bool_and_5_52":  ("int main() { if (5 > 52) { if (52 > 46) return 1; }"
                       " return 0; }", 0),
}


def fmt(row, base, width=16, thr=0.4):
    vals = [(i, float(row[base + i].item())) for i in range(width)]
    hot = [(i, v) for i, v in vals if abs(v) > thr]
    hot.sort(key=lambda t: -t[1])
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in hot) + "]"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    se_mark = dp.get("MARK_SE_ONLY")
    ohi_key = "OUTPUT_HI_THIS_STEP" if "OUTPUT_HI_THIS_STEP" in dp else "OUTPUT_HI"

    for pname, (src, exp) in PROGS.items():
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
        with torch.no_grad():
            emb = model.embed(padded)[0]
        print(f"\n=== {pname} exp={exp} nsteps={len(otok)} ===", flush=True)
        # Precompute cached residuals per block (one forward each).
        resids = {}
        for b in blocks:
            with torch.no_grad():
                resids[b] = model.forward(padded, stop_after_block=b)[0]
        # Walk each oracle step; report the AX/SE rows where CMP is hot OR
        # where the emitted byte's high nibble is non-zero (the leak).
        for s, (pc, ax) in enumerate(opc_ax):
            step_start = len(prefix) + s * STEP
            for off in range(STEP):
                r = step_start + off
                if r >= len(tape):
                    break
                is_ax = emb[r, dp["MARK_AX"]].abs().item() > 0.5
                is_se = (se_mark is not None and
                         emb[r, se_mark].abs().item() > 0.5)
                if not (is_ax or is_se):
                    continue
                b = blocks[-1]
                row = resids[b][r]
                cmp = [float(row[dp["CMP"] + i].item()) for i in range(4)]
                lo_vec = torch.tensor([row[dp["OUTPUT_LO"] + i] for i in range(16)])
                hi_vec = torch.tensor([row[dp[ohi_key] + i] for i in range(16)])
                hi_arg = int(hi_vec.argmax())
                byte = int(lo_vec.argmax()) | (hi_arg << 4)
                cmphot = any(abs(c) > 0.4 for c in cmp)
                # Only surface rows of interest: CMP-hot OR hi-nibble leak.
                if not (cmphot or hi_arg != 0):
                    continue
                kind = "SE" if is_se else "AX"
                lo = fmt(row, dp["OUTPUT_LO"])
                hi = fmt(row, dp[ohi_key])
                print(f"  step{s:2d} pc={pc} {kind} r{r} byte=0x{byte:02x} "
                      f"CMP=[{cmp[0]:.1f},{cmp[1]:.1f},{cmp[2]:.1f},{cmp[3]:.1f}] "
                      f"LO={lo} HI={hi}", flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [26, 28, 30, 33, 36]
    main(blks)
