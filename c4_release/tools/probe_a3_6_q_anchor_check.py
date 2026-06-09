"""A3.6 follow-up: verify whether the Q anchor (STACK0_BYTE_1 +
BYTE_INDEX_1) fires at the post-PSH S0 frames (S0@114 BI_1 = p=116,
etc.) at the input of L12 (the broadcast head's block).

If STACK0_BYTE_1 is not set at p=116 pre-L12, the broadcast head has
no Q activation — explaining the lack of write.
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


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    model = runner.model
    dp = model.embed._dim_positions

    captures = []

    def embed_hook(module, inputs, output):
        captures.append({"embed": output.detach().clone()})

    handles = [model.embed.register_forward_hook(embed_hook)]
    for li in range(min(32, len(model.blocks))):
        block = model.blocks[li]

        def make_pre(idx):
            def fn(module, inputs):
                if captures:
                    captures[-1][f"pre_L{idx}"] = inputs[0].detach().clone()
            return fn
        handles.append(block.register_forward_pre_hook(make_pre(li)))

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
        last = c.get("pre_L31")
        if last is None:
            continue
        if best is None or last.shape[1] > best["pre_L31"].shape[1]:
            best = c
    seq = best["pre_L31"].shape[1]
    print(f"seq_len={seq}")

    dims_to_check = [
        "MARK_STACK0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "IS_BYTE", "OP_PSH", "MARK_AX",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
    ]
    # focus rows: each STACK0 BI_1 row, and each AX BI_1 row.
    mark_stack0 = dp["MARK_STACK0"]
    bi1 = dp["BYTE_INDEX_1"]
    mark_ax = dp["MARK_AX"]
    op_psh = dp["OP_PSH"]
    # Find positions
    pre_L12 = best["pre_L12"]
    s0_pos = torch.nonzero(pre_L12[0, :, mark_stack0] > 0.5, as_tuple=False).flatten().tolist()
    ax_pos = torch.nonzero(pre_L12[0, :, mark_ax] > 0.5, as_tuple=False).flatten().tolist()
    print(f"STACK0 positions: {s0_pos}")
    print(f"AX positions: {ax_pos}")

    s0_bi1 = []
    for s in s0_pos:
        for d in range(0, 10):
            p = s + d
            if p < seq and float(pre_L12[0, p, bi1].item()) > 0.5:
                s0_bi1.append((s, p))
                break
    ax_bi1 = []
    for a in ax_pos:
        for d in range(0, 10):
            p = a + d
            if p < seq and float(pre_L12[0, p, bi1].item()) > 0.5:
                ax_bi1.append((a, p))
                break

    print("\n== Pre-L12 residual at S0 BI_1 rows ==")
    for (s, p) in s0_bi1:
        vals = {d: float(pre_L12[0, p, dp[d]].item()) for d in dims_to_check}
        print(f"  S0@{s} BI_1 p={p}: " + ", ".join(f"{d}={vals[d]:.1f}" for d in dims_to_check))

    print("\n== Pre-L12 residual at AX BI_1 rows ==")
    for (a, p) in ax_bi1:
        vals = {d: float(pre_L12[0, p, dp[d]].item()) for d in dims_to_check}
        print(f"  AX@{a} BI_1 p={p}: " + ", ".join(f"{d}={vals[d]:.1f}" for d in dims_to_check))

    # === Inspect CLEAN_EMBED LO/HI bands at each AX BI_1 row.
    print("\n== Pre-L12 CLEAN_EMBED_{LO,HI} bands at AX BI_1 rows ==")
    cl_lo = dp["CLEAN_EMBED_LO"]
    cl_hi = dp["CLEAN_EMBED_HI"]
    for (a, p) in ax_bi1:
        lo = pre_L12[0, p, cl_lo:cl_lo+16].tolist()
        hi = pre_L12[0, p, cl_hi:cl_hi+16].tolist()
        lo_idx = max(range(16), key=lambda i: lo[i])
        hi_idx = max(range(16), key=lambda i: hi[i])
        print(f"  AX@{a} BI_1 p={p}: LO[max={lo_idx} val={lo[lo_idx]:.2f}], HI[max={hi_idx} val={hi[hi_idx]:.2f}]")

    # === Check OP_PSH at the MARK_AX d=0 row (p=100 expected).
    op_psh_dim = dp["OP_PSH"]
    print("\n== Pre-L12 OP_PSH at each AX d=0 row ==")
    for a in ax_pos:
        ps = float(pre_L12[0, a, op_psh_dim].item())
        print(f"  AX@{a} d=0 p={a}: OP_PSH={ps:.2f}")

    # === Check OP_PSH at each STACK0 row.
    print("\n== Pre-L12 OP_PSH at each STACK0 d=0 row ==")
    for s in s0_pos:
        ps = float(pre_L12[0, s, op_psh_dim].item())
        print(f"  S0@{s} d=0 p={s}: OP_PSH={ps:.2f}")


if __name__ == "__main__":
    main()
