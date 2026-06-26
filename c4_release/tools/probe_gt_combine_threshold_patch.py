#!/usr/bin/env python3
"""VERIFY the GT (hi_eq AND lo_lt) 3-way override threshold fix by patching the
LIVE ComparisonCombine post-op's b_up in place and re-decoding.

The live decoder is the imperative ComparisonCombine (PureFFN, block 22 post-op)
which reads raw CMP[0..3] at MARK_AX. The GT (CMP+1 AND CMP+3 -> 0) override
fires on lo_lt(CMP+3=1.67) ALONE because MARK_AX(1)+CMP+3(1.67)-2.5 > 0 even
with hi_eq(CMP+1)=0. Genuine eq-hi GT-false has hi_eq=1.24 + lo_lt=1.46. Raising
the override threshold to ~3.18 rejects lo_lt-alone (1+1.67<3.18) while keeping
the genuine (1+1.24+1.46>3.18) firing. This patches the unit's b_up and checks
all discriminating cases flip/hold correctly.

  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_ifvargtt python tools/probe_gt_combine_threshold_patch.py [thresh]
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
from neural_vm.vm_step import ComparisonCombine  # noqa: E402
from tools.probe_gt_cmp_values import PROGS  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def patch_gt_combine(model, new_thresh, S=100.0):
    """Find the ComparisonCombine post-op and raise the GT (CMP+1,CMP+3)
    3-way override b_up to -S*new_thresh. The GT 3-way (hi_eq,lo_lt) unit is
    the 12th unit (idx 11): EQ(2)+NE(2)+LT(3)=7, GT default(1)+2way(1)=2 -> the
    (CMP+1,CMP+3) is unit 7+2 = 9 (0-based)."""
    patched = 0
    for blk in model.blocks:
        for po in getattr(blk, "post_ops", []):
            if isinstance(po, ComparisonCombine):
                # GT (CMP+1,CMP+3) override = unit index 9.
                po.b_up.data[9] = -S * new_thresh
                patched += 1
    return patched


def main(thresh):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    out_lo = dp["OUTPUT_LO"]

    n = patch_gt_combine(model, thresh)
    print(f"patched {n} ComparisonCombine GT-3way b_up -> -S*{thresh}\n", flush=True)

    nwrong = 0
    for pname, (src, want, result_pc) in PROGS.items():
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
        rstep = next(s for s, (pc, ax) in enumerate(opc_ax) if pc == result_pc)
        with torch.no_grad():
            emb = model.embed(padded)[0]
        step_start = len(prefix) + rstep * STEP
        ax_row = None
        for off in range(STEP):
            r = step_start + off
            if r < len(tape) and emb[r, dp["MARK_AX"]].abs().item() > 0.5:
                ax_row = r
                break
        with torch.no_grad():
            rf = model.forward(padded, stop_after_block=33)[0][ax_row]
        lo0, lo1 = float(rf[out_lo + 0].item()), float(rf[out_lo + 1].item())
        res = 1 if lo1 > lo0 else 0
        ok = "OK" if res == want else "**WRONG**"
        if res != want:
            nwrong += 1
        print(f"{pname:22s} want={want} got={res}  "
              f"[LO0={lo0:.1f} LO1={lo1:.1f}]  {ok}", flush=True)
    print(f"\n{nwrong} wrong of {len(PROGS)}", flush=True)


if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 3.18
    main(t)
