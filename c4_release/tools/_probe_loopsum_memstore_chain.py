#!/usr/bin/env python3
"""Trace MEM_STORE + OP_SI/OP_PSH/OP_JSR per step for loop_sum id450, BOTH at the
opcode source (AX marker / STEP rows) and at the MARK_MEM marker, at the L15
input, to see WHY the SI stores (steps 5/9) don't carry MEM_STORE=1 (so the
in-loop LI CAM can't find them).

Run: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     C4_VM_CACHE_DIR=/tmp/c4cache_loops11 python tools/_probe_loopsum_memstore_chain.py [pid] [block]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PID = int(sys.argv[1]) if len(sys.argv) > 1 else 450
# block to capture INPUT of (default: L15 memory_lookup). Pass a block idx.
TGT = sys.argv[2] if len(sys.argv) > 2 else "L15"


def oracle_windows(bc, n=16):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(n):
        if vm.halted:
            break
        if not vm.step():
            break
        steps.append((int(vm.pc) & 0xFFFFFFFF, int(vm.ax) & 0xFFFF))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break
    return steps, toks


def main():
    p = build_groundtruth_probe()
    model = p.model
    dimpos = model.embed._dim_positions
    STEP = int(Token.STEP_TOKENS)

    op_li = dimpos["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi
    tgt = L15 if TGT == "L15" else int(TGT)

    src, exp, desc = generate_test_programs()[PID]
    bc = compile_c(src)[0]
    prompt = p._build_context(bc)
    pl = len(prompt)
    steps, toks = oracle_windows(bc, 14)
    ctx = list(prompt)
    for w in toks:
        ctx.extend(w)
    padded = torch.tensor([ctx], device=p._device)

    cap = {}

    def pre_hook(module, inputs):
        cap["pre"] = inputs[0].detach().clone()

    h = model.blocks[tgt].register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(padded)
    h.remove()
    pre = cap["pre"][0]
    seq = pre.shape[0]

    def g(pos, nm):
        return float(pre[pos, dimpos[nm]].item()) if nm in dimpos else None

    print(f"L15={L15} captured INPUT of block {tgt} seq={seq} pl={pl} STEP={STEP}")
    print("step pc | AX-marker: OP_SI OP_SC OP_PSH OP_JSR OP_ENT MEM_STORE | "
          "MEM-marker: MEM_STORE OP_SI OP_PSH OP_JSR | STEP_END row OP_SI")
    for si in range(len(steps)):
        base = pl + si * STEP
        axrow = memrow = serow = None
        for j in range(base, min(base + STEP, seq)):
            if axrow is None and pre[j, dimpos["MARK_AX"]].item() > 0.5:
                axrow = j
            if memrow is None and pre[j, dimpos["MARK_MEM"]].item() > 0.5:
                memrow = j
        # STEP_END row is the last token of the window
        serow = min(base + STEP - 1, seq - 1)
        pcb = steps[si][0]
        ax_si = g(axrow, "OP_SI") if axrow else None
        ax_sc = g(axrow, "OP_SC") if axrow else None
        ax_psh = g(axrow, "OP_PSH") if axrow else None
        ax_jsr = g(axrow, "OP_JSR") if axrow else None
        ax_ent = g(axrow, "OP_ENT") if axrow else None
        ax_ms = g(axrow, "MEM_STORE") if axrow else None
        m_ms = g(memrow, "MEM_STORE") if memrow else None
        m_si = g(memrow, "OP_SI") if memrow else None
        m_psh = g(memrow, "OP_PSH") if memrow else None
        m_jsr = g(memrow, "OP_JSR") if memrow else None
        se_si = g(serow, "OP_SI")

        def f(v):
            return f"{v:6.2f}" if v is not None else "  None"
        print(f" {si:2d} {pcb:4d} | AX(r{axrow}): {f(ax_si)} {f(ax_sc)} {f(ax_psh)} "
              f"{f(ax_jsr)} {f(ax_ent)} {f(ax_ms)} | "
              f"MEM(r{memrow}): {f(m_ms)} {f(m_si)} {f(m_psh)} {f(m_jsr)} | "
              f"SE_SI={f(se_si)}")


if __name__ == "__main__":
    main()
