"""Probe: does L10 broadcast head fire at later STACK0 frames?

The broadcast Q gate uses STACK0_BYTE_h + BYTE_INDEX_h positives in slot 0
and slot 33. The Q row at any STACK0 byte h row SHOULD fire. K side needs
to find a MARK_AX BYTE_INDEX_h row with OP_PSH=1 (causally accessible).

If the broadcast IS firing at S0@114 BI_h, why does VAL_h_LO stay 0?
This probe samples the residual at every STACK0 frame's BI_h row
across multiple layers post-L10.
"""

import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(REPO_ROOT))

import torch

from c4_release.neural_vm.run_vm import AutoregressiveVMRunner
from c4_release.neural_vm.embedding import Opcode

PROG = [
    (Opcode.IMM, 0x200),
    Opcode.PSH,
    (Opcode.IMM, 42),
    Opcode.SI,
    (Opcode.IMM, 0x200),
    Opcode.LI,
    Opcode.EXIT,
]


def make_bc(prog):
    out = []
    for item in prog:
        if isinstance(item, tuple):
            opcode, imm = item
            out.append(opcode | (imm << 8))
        else:
            out.append(item)
    return out


def _band(after, pos, dim_start, width=16):
    band = after[0, pos, dim_start:dim_start + width]
    idx = int(torch.argmax(band).item())
    v = float(band[idx].item())
    return idx, v


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    _probe(runner)


def _probe(runner):
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []

    model = runner.model
    dp = model.embed._dim_positions

    captures = []

    def embed_hook(module, inputs, output):
        captures.append({"token_ids": inputs[0].detach().clone()})

    def block_hook(name):
        def fn(module, inputs, output):
            if captures:
                captures[-1][name] = output.detach().clone()
        return fn

    handles = [model.embed.register_forward_hook(embed_hook)]
    for li in range(min(32, len(model.blocks))):
        handles.append(
            model.blocks[li].register_forward_hook(block_hook(f"after_L{li}"))
        )

    bc = make_bc(PROG)
    try:
        try:
            result = runner.run(bc, b"", max_steps=30)
        except Exception as e:
            print(f"runner raised: {e}")
            result = None
    finally:
        for h in handles:
            h.remove()

    print(f"Result: {result}")

    best = None
    for c in captures:
        if "after_L12" not in c:
            continue
        if best is None or c["after_L12"].shape[1] > best["after_L12"].shape[1]:
            best = c

    if best is None:
        print("No usable capture.")
        return

    seq = best["after_L12"].shape[1]
    print(f"Full-context capture: seq_len={seq}")

    mark_stack0 = dp["MARK_STACK0"]
    mark_ax = dp["MARK_AX"]
    op_psh = dp["OP_PSH"]
    bi = [dp[f"BYTE_INDEX_{j}"] for j in range(4)]
    stack0_byte = [None] + [dp[f"STACK0_BYTE{i}"] for i in (1, 2, 3)]
    vlo = [None] + [dp[f"STACK0_BYTE_VAL_{i}_LO"] for i in (1, 2, 3)]
    vhi = [None] + [dp[f"STACK0_BYTE_VAL_{i}_HI"] for i in (1, 2, 3)]

    stack0_positions = torch.nonzero(
        best["after_L12"][0, :, mark_stack0] > 0.5, as_tuple=False
    ).flatten().tolist()
    ax_positions = torch.nonzero(
        best["after_L12"][0, :, mark_ax] > 0.5, as_tuple=False
    ).flatten().tolist()

    print(f"\nMARK_STACK0 positions: {stack0_positions}")
    print(f"MARK_AX positions:     {ax_positions}")

    # check OP_PSH at each AX row
    for ax_pos in ax_positions:
        for d in range(0, 5):
            p = ax_pos + d
            if p >= seq:
                continue
            op = float(best["after_L12"][0, p, op_psh].item())
            if op > 0.5 or d == 0:
                print(f"  AX@{ax_pos} d={d} p={p}  OP_PSH={op:.2f}  BI=[{float(best['after_L12'][0, p, bi[0]].item()):.1f},{float(best['after_L12'][0, p, bi[1]].item()):.1f},{float(best['after_L12'][0, p, bi[2]].item()):.1f},{float(best['after_L12'][0, p, bi[3]].item()):.1f}]")

    # Cross-layer check: after L10 (=phys 12) and other neighbors, dump
    # STACK0_BYTE_VAL_1 at every STACK0+BI1 row.
    print("\n== STACK0_BYTE_VAL_1_LO/HI at each STACK0 BI_1 row, across layers ==")
    for s_pos in stack0_positions:
        # find BI_1 row in this frame
        bi1_dim = bi[1]
        target = None
        for d in range(0, 10):
            p = s_pos + d
            if p >= seq:
                continue
            if float(best["after_L12"][0, p, bi1_dim].item()) > 0.5:
                target = p
                break
        if target is None:
            continue
        print(f"\n  S0@{s_pos:3d} BI1 row @ p={target}:")
        for li, lname in [(9, "L7 "), (10, "L8 "), (11, "L9 "), (12, "L10"), (13, "L11"), (15, "L13"), (25, "L13b"), (27, "L14")]:
            cap = best.get(f"after_L{li}")
            if cap is None:
                continue
            v_lo = _band(cap, target, vlo[1])
            v_hi = _band(cap, target, vhi[1])
            print(f"    {lname} (phys L{li:2d}): VAL1_LO={v_lo[0]:2d}/{v_lo[1]:5.2f}  VAL1_HI={v_hi[0]:2d}/{v_hi[1]:5.2f}")


if __name__ == "__main__":
    main()
