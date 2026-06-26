#!/usr/bin/env python3
"""Dump the AX-row CMP[0..3] values at the GT-result step (input to the
ComparisonCombine, physical block 22) for the discriminating cases:

  - loaded-var GT-TRUE  (fail): only lo_lt(CMP+3) present, hi_eq(CMP+1)~0
  - loaded-var GT-TRUE  (pass id427): no lo_lt
  - literal GT-FALSE eq-hi (id350/353/365/368): hi_eq AND lo_lt BOTH present
  - literal GT-TRUE  eq-hi (id357/359/361): hi_eq present, lo_lt absent
  - GT-FALSE hi_lt (id430): hi_lt(CMP+0)

This pins the exact CMP+1 / CMP+3 magnitudes the live imperative
ComparisonCombine GT (hi_eq AND lo_lt) override reads, so the campaign-gated
threshold/blocker fix can keep the genuine override firing while rejecting the
lo_lt-alone trip.

  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_ifvargtt python tools/probe_gt_cmp_values.py [block]
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


def _var(x, k):
    return f"int main() {{ int x; x = {x}; if (x > {k}) return 1; return 0; }}"


def _lit(a, b):
    return f"int main() {{ if ({a} > {b}) return 1; return 0; }}"


PROGS = {
    # loaded-var GT-TRUE fails (want GT=1)
    "id436_var66gt24_T": (_var(66, 24), 1, 106),
    "id441_var88gt60_T": (_var(88, 60), 1, 106),
    "id445_var37gt28_T": (_var(37, 28), 1, 106),
    "id427_var85gt48_T": (_var(85, 48), 1, 106),   # PASS ref (no lo_lt)
    "id430_var23gt62_F": (_var(23, 62), 0, 106),   # PASS ref (hi_lt)
    # literal eq-hi GT-FALSE (genuine hi_eq AND lo_lt override -> GT=0)
    "id350_lit35gt43_F": (_lit(35, 43), 0, 34),
    "id353_lit20gt30_F": (_lit(20, 30), 0, 34),
    "id365_lit49gt62_F": (_lit(49, 62), 0, 34),
    "id368_lit50gt54_F": (_lit(50, 54), 0, 34),
    # literal eq-hi GT-TRUE (must NOT fire -> GT=1)
    "id357_lit54gt53_T": (_lit(54, 53), 1, 34),
    "id359_lit60gt54_T": (_lit(60, 54), 1, 34),
    "id361_lit54gt50_T": (_lit(54, 50), 1, 34),
}


def main(block):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    cmp_base = dp["CMP"]
    out_lo = dp["OUTPUT_LO"]

    print(f"AX-row CMP[0..3] at block {block} (ComparisonCombine input), "
          f"+ final OUTPUT_LO[0]/[1] @ blk33\n", flush=True)
    print(f"{'prog':22s} {'want':>4s}  "
          f"{'CMP0(hilt)':>10s} {'CMP1(hieq)':>10s} {'CMP2(loeq)':>10s} "
          f"{'CMP3(lolt)':>10s}   {'res':>3s}", flush=True)

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
            resid_b = model.forward(padded, stop_after_block=block)[0]
            resid_f = model.forward(padded, stop_after_block=33)[0]
        row = resid_b[ax_row]
        c = [float(row[cmp_base + i].item()) for i in range(4)]
        rf = resid_f[ax_row]
        lo0, lo1 = float(rf[out_lo + 0].item()), float(rf[out_lo + 1].item())
        res = 1 if lo1 > lo0 else 0
        ok = "OK" if res == want else "**WRONG**"
        print(f"{pname:22s} {want:>4d}  "
              f"{c[0]:>10.2f} {c[1]:>10.2f} {c[2]:>10.2f} {c[3]:>10.2f}   "
              f"{res:>3d} {ok}", flush=True)


if __name__ == "__main__":
    blk = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    main(blk)
