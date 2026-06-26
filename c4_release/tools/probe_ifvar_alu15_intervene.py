#!/usr/bin/env python3
"""INTERVENTION: zero ALU_HI[15] (the 0xF address-high-nibble leak) at the cmp
block input on the GT-result-step AX row, and check whether the hi_lt GT-false
override re-fires so the loaded-var GT-false result decodes 0 (correct).

Theory (from probe_ifvar_result_step): on the LOADED-variable path the result
step's ALU_HI carries a spurious +6.5 at cell 15 (impossible for operands 0-99,
A.hi<=6). The hi_lt unit's blocker (-0.5 per OTHER ALU_HI cell) over-penalizes:
  6.0(MARK) + 0.5*6.0(ALU@1) + 6.0(AXC@3) - 0.5*6.5(@15) - 0.5*0.4(@8) = 11.55
  < threshold 13.22  -> hi_lt DOES NOT fire -> GT defaults to 1 (WRONG).
The literal path has ALU_HI[15]~0.5 so the blocker is ~-0.25 and hi_lt fires.

If zeroing ALU_HI[15] restores hi_lt and flips OUTPUT_LO argmax to cell 0 for
430/433 WITHOUT breaking 427 (GT-true) or the literal, the cell-15-exclusion fix
is the lever.

  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_ifvargt python tools/probe_ifvar_alu15_intervene.py
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

# Blocks to apply the ALU_HI[15]=0 intervention at (input residual), to bracket
# the ordering-engine read. We hook a RANGE so the clear survives into the engine.
HOOK_BLOCKS = list(range(15, 23))


def decode_byte(row, lo_base, hi_base):
    lo = int(torch.argmax(row[lo_base:lo_base + 16]).item())
    hi = int(torch.argmax(row[hi_base:hi_base + 16]).item())
    return (hi << 4) | lo


def main():
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    alu_hi = dp["ALU_HI"]
    out_lo = dp["OUTPUT_LO"]
    out_hi = dp["OUTPUT_HI"]

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
        for off in range(STEP):
            r = step_start + off
            if r < len(tape) and emb[r, dp["MARK_AX"]].abs().item() > 0.5:
                ax_row = r
                break

        STOP_BLK = 33  # OUTPUT band is clean here (pre-final-explosion)

        def run(intervene):
            handles = []
            if intervene:
                def mk(_b):
                    def hook(mod, inp):
                        x = inp[0]
                        x[0, ax_row, alu_hi + 15] = 0.0
                        return (x,) + inp[1:]
                    return hook
                for b in HOOK_BLOCKS:
                    handles.append(model.blocks[b].register_forward_pre_hook(mk(b)))
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=STOP_BLK)[0]
            for h in handles:
                h.remove()
            return resid[ax_row]

        def lobit(row):
            # OUTPUT_LO cell0 vs cell1 = result byte low-nibble 0 (=false) vs 1 (=true)
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
              f"alu15zeroed[LO0={c0i:.1f} LO1={c1i:.1f}->{ri}]  {verdict}", flush=True)


if __name__ == "__main__":
    main()
