#!/usr/bin/env python3
"""Block-by-block ALU_LO/HI + OUTPUT_LO/HI nibble-decode at the AX-marker row.

For a target VM step, decode the byte-0 value held in ALU_LO (operand A for the
binary ALU FFN) and in OUTPUT_LO (the emitted AX register byte) after EVERY
block, so we can pinpoint which block delivers / loses the operand under the
C4_OPERAND_FROM_MEMSP reroute.

The AX-marker row is found by re-deriving the offset of MARK_AX inside the step
from the produced token stream (the AX register marker token REG_AX).

Usage:
    CUDA_VISIBLE_DEVICES=0 [C4_OPERAND_FROM_MEMSP=1 C4_NO_STACK0_EMIT=0] \
        python tools/probe_operand_cam.py 9 3   # id, target_step
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import torch  # noqa: E402

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return list(bc), exp, desc


def _nib_argmax(vec, base):
    """Decode a 16-way nibble one-hot at residual[base:base+16] -> int (or -1)."""
    sl = vec[base:base + 16]
    m = float(sl.max())
    if m < 0.5:
        return -1
    return int(sl.argmax())


@torch.no_grad()
def main(idx, target_step, max_steps):
    ST = Token.STEP_TOKENS
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)
    model = runner.model
    dev = next(model.parameters()).device
    dp = model.dim_positions
    ALO, AHI = dp["ALU_LO"], dp["ALU_HI"]
    OLO, OHI = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
    MARK_AX = dp["MARK_AX"]

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)
    for _ in range(ST * max_steps):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        nxt = int(model.forward(padded)[0][-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    emitted = context[len(prefix):]
    # find AX-marker offset within the target step
    step_slice = emitted[target_step * ST:(target_step + 1) * ST]
    ax_off = None
    for i, t in enumerate(step_slice):
        if t == Token.REG_AX:
            ax_off = i
            break
    if ax_off is None:
        print(f"no REG_AX marker in step {target_step}; slice={step_slice}")
        return
    pos = len(prefix) + target_step * ST + ax_off
    padded = torch.tensor([context], dtype=torch.long, device=dev)
    nb = len(model.blocks)
    print(f"== id={idx} {desc[:36]} step{target_step} AXmarker@off{ax_off} pos{pos} "
          f"flags OPERAND={os.environ.get('C4_OPERAND_FROM_MEMSP','0')} "
          f"NOSTK={os.environ.get('C4_NO_STACK0_EMIT','0')} ==")
    print("  blk | ALU_LO ALU_HI | OUT_LO OUT_HI | MARK_AX | block")
    prev = None
    for b in range(nb):
        resid = model.forward(padded, stop_after_block=b)[0]
        v = resid[pos]
        alo = _nib_argmax(v, ALO)
        ahi = _nib_argmax(v, AHI)
        olo = _nib_argmax(v, OLO)
        ohi = _nib_argmax(v, OHI)
        max_ax = float(v[MARK_AX])
        cur = (alo, ahi, olo, ohi)
        bn = type(model.blocks[b]).__name__
        changed = (cur != prev)
        if changed or b == nb - 1:
            mark = " <==" if changed else ""
            print(f"  {b:3d} | {alo:6d} {ahi:6d} | {olo:6d} {ohi:6d} | {max_ax:7.1f} | {bn}{mark}")
        prev = cur


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.lstrip('-').isdigit()]
    idx = args[0] if args else 9
    step = args[1] if len(args) > 1 else 3
    ms = args[2] if len(args) > 2 else 6
    main(idx, step, ms)
