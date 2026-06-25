#!/usr/bin/env python3
"""var_update LEA-after-SI AX-divergence block scan (campaign, spec_k=0).

CORRECTION to the brief: var_update id325 does NOT framing-drift at the SI store.
The 30-token frame is intact (all inter-PC gaps == 30, PC decodes at every step).
The real divergence is an AX VALUE error at step 14 = the `LEA -8` for `return x`
(the address-of-x), which FOLLOWS the SI store (step 13 = SI). The model leaves
AX = the stale SI value (57) instead of computing the local address (0xFFE8).

Crucially the IDENTICAL `LEA -8` at step 6 (which follows the FIRST SI store,
itself preceded by an IMM) computes AX correctly. So this probe teacher-forces
the oracle tape and reads the AX-relevant residual bands at the MARK_AX row of
BOTH the working LEA step (6) and the failing LEA step (14), across a block
sweep, to localise WHERE the step-14 LEA's address write is suppressed.

  CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_sistore \
    python tools/probe_sistore_lea_axdiverge.py [blocks...]
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

SRC = "int main() { int x; x = 50; x = x + 7; return x; }"

# steps of interest: 6 = LEA after IMM-SI (WORKS), 14 = LEA after ADD-SI (FAILS).
STEPS = [int(s) for s in os.environ.get("PROBE_STEPS", "6,14").split(",")]


def fmt(row, base, width=16, thr=0.5):
    if base is None:
        return "n/a"
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.0f}@{i}" for i, v in enumerate(vals) if abs(v) > thr) + "]"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    ax_base = dp["MARK_AX"]

    bc, data = compile_c(SRC)
    _, oracle_tokens = runner._oracle_pc_ax_steps(
        bc, data or b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(bc, data or b"", [], "", spec_k=1,
                                   adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.embed(padded)[0]

    # candidate AX-output bands to watch
    band_names = ["OUTPUT_LO", "OUTPUT_HI", "ALU_LO", "ALU_HI",
                  "AX_FULL_LO", "AX_FULL_HI", "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"]
    bands = [(nm, dp.get(nm)) for nm in band_names if dp.get(nm) is not None]

    # locate MARK_AX row for each step.
    ax_rows = {}
    for s in STEPS:
        step_start = len(prefix) + s * STEP
        for off in range(STEP):
            r = step_start + off
            if r < len(tape) and emb[r, ax_base].abs().item() > 0.5:
                ax_rows[s] = step_start + off
                break

    print(f"=== var_update id325 LEA-after-SI AX divergence  STEP={STEP} ===", flush=True)
    print(f"   step6=LEA-after-IMM-SI(WORKS, ax->addr)  "
          f"step14=LEA-after-ADD-SI(FAILS, ax stays 57)", flush=True)
    print(f"   ax_rows={ax_rows}", flush=True)
    for b in blocks:
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=b)[0]
        print(f" --- block {b} ---", flush=True)
        for s in STEPS:
            if s not in ax_rows:
                continue
            row = resid[ax_rows[s]]
            parts = " ".join(f"{nm}={fmt(row, base)}" for nm, base in bands)
            print(f"   step{s:2d}  {parts}", flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [6, 9, 10, 11, 14, 16, 20, 26, 30, 36]
    main(blks)
