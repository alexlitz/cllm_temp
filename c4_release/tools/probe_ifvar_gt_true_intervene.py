#!/usr/bin/env python3
"""INTERVENTION on the if_var GT-TRUE result step: zero CMP+3 (lo_lt) at the
result-step AX row across the cmp/combine blocks and check whether the loaded-var
GT-TRUE result decodes 1 (correct) instead of 0.

Theory (from probe_ifvar_gt_true): the 6 GT-TRUE fails (A.hi > B.hi but
A.lo < B.lo) decode GT=0 because the AX-row spuriously carries lo_lt (CMP+3),
which the downstream GT result writer treats as evidence to push OUTPUT_LO+0.
For a GT comparison lo_lt is IRRELEVANT once A.hi > B.hi -- the high-nibble
order already decides GT=1. id427 (85>48, A.lo>B.lo so no lo_lt) decodes 1.

We also test zeroing CMP+1 (hi_eq) and the (CMP+1 AND CMP+3) pair to find the
minimal causal flag. If zeroing CMP+3 flips 441/445/448 to 1 WITHOUT breaking
the GT-FALSE (id430) or any passing case, the CMP+3 leak is the lever.

  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_ifvargtt python tools/probe_ifvar_gt_true_intervene.py
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
    "id425_96gt59_T": (_src(96, 59), "FAIL exp1", 106),
    "id436_66gt24_T": (_src(66, 24), "FAIL exp1", 106),
    "id440_97gt30_T": (_src(97, 30), "FAIL exp1", 106),
    "id441_88gt60_T": (_src(88, 60), "FAIL exp1", 106),
    "id445_37gt28_T": (_src(37, 28), "FAIL exp1", 106),
    "id448_84gt74_T": (_src(84, 74), "FAIL exp1", 106),
    "id427_85gt48_T": (_src(85, 48), "PASS exp1", 106),  # GT-TRUE pass
    "id430_23gt62_F": (_src(23, 62), "PASS exp0", 106),  # GT-FALSE pass
}

# Hook the cmp/combine region so the clear survives into the combine + relay.
_hb = os.environ.get("HOOK_BLOCKS")
if _hb:
    a, b = _hb.split(":")
    HOOK_BLOCKS = list(range(int(a), int(b)))
else:
    HOOK_BLOCKS = list(range(15, 31))


def main(clear_cells):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    cmp_base = dp["CMP"]
    out_lo = dp["OUTPUT_LO"]
    se_mark = dp.get("MARK_SE_ONLY")

    print(f"clearing CMP cells {clear_cells} at the result-step AX row, "
          f"blocks {HOOK_BLOCKS[0]}..{HOOK_BLOCKS[-1]}\n", flush=True)

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
        rstep = next(s for s, (pc, ax) in enumerate(opc_ax) if pc == result_pc)
        with torch.no_grad():
            emb = model.embed(padded)[0]
        step_start = len(prefix) + rstep * STEP
        ax_row = None
        se_row = None
        for off in range(STEP):
            r = step_start + off
            if r >= len(tape):
                break
            if ax_row is None and emb[r, dp["MARK_AX"]].abs().item() > 0.5:
                ax_row = r
            if se_row is None and se_mark is not None \
                    and emb[r, se_mark].abs().item() > 0.5:
                se_row = r

        STOP_BLK = 33
        target_rows = [ax_row] if os.environ.get("CLEAR_ROW", "AX") == "AX" \
            else ([se_row] if os.environ.get("CLEAR_ROW") == "SE"
                   else [ax_row, se_row])

        def run(intervene):
            handles = []
            if intervene:
                def mk():
                    def hook(mod, inp):
                        x = inp[0]
                        for tr in target_rows:
                            if tr is None:
                                continue
                            for c in clear_cells:
                                x[0, tr, cmp_base + c] = 0.0
                        return (x,) + inp[1:]
                    return hook
                for b in HOOK_BLOCKS:
                    handles.append(model.blocks[b].register_forward_pre_hook(mk()))
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=STOP_BLK)[0]
            for h in handles:
                h.remove()
            return resid[ax_row]

        def lobit(row):
            c0 = float(row[out_lo + 0].item())
            c1 = float(row[out_lo + 1].item())
            return c0, c1, (1 if c1 > c0 else 0)

        base = run(False)
        interv = run(True)
        c0b, c1b, rb = lobit(base)
        c0i, c1i, ri = lobit(interv)
        want = 0 if "exp0" in tag else 1
        verdict = ("FIXED" if (ri == want and rb != want)
                   else "held" if ri == want
                   else "BROKE" if (rb == want and ri != want)
                   else "no-change")
        print(f"{pname:18s} [{tag}] want={want}  "
              f"base[LO0={c0b:.1f} LO1={c1b:.1f}->{rb}]  "
              f"cleared[LO0={c0i:.1f} LO1={c1i:.1f}->{ri}]  {verdict}", flush=True)


if __name__ == "__main__":
    cells = [int(x) for x in sys.argv[1:]] or [3]
    main(cells)
