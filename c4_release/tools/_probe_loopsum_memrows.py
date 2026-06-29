#!/usr/bin/env python3
"""Dump EVERY MEM row of loop_sum id450 at L15 input: per-step store/addr/value,
so we can see whether the prologue `i=1` store (step5, &i=0xFFE8) is present as a
retrievable MEM store frame for the in-loop LI CAM at step 11.

Run: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     C4_VM_CACHE_DIR=/tmp/c4cache_loops11 python tools/_probe_loopsum_memrows.py [pid]
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

    h = model.blocks[L15].register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(padded)
    h.remove()
    pre = cap["pre"][0]
    seq = pre.shape[0]

    def byteat(pp):
        clo = pre[pp, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
        chi = pre[pp, dimpos["CLEAN_EMBED_HI"]:dimpos["CLEAN_EMBED_HI"]+16]
        lo = int(clo.argmax().item()) if float(clo.max()) > 0.3 else 0
        hi = int(chi.argmax().item()) if float(chi.max()) > 0.3 else 0
        return hi * 16 + lo

    print(f"L15={L15} seq={seq} pl={pl} STEP={STEP}")
    print("step | pc(before) | instr | MEM_STORE | addr(b0..3) | val(b0..3) | MEM_VAL_B")
    from neural_vm.unified_compiler.symbolic_forward import decode_instr, _OPCODE_NAMES
    for si in range(len(steps)):
        base = pl + si * STEP
        # find MEM marker row in this step window
        memrow = None
        for j in range(base, min(base + STEP, seq)):
            if pre[j, dimpos["MARK_MEM"]].item() > 0.5:
                memrow = j
                break
        if memrow is None:
            print(f"  {si:2d} | (no MEM row)")
            continue
        mst = float(pre[memrow, dimpos["MEM_STORE"]].item())
        addr = [byteat(memrow + 1 + k) for k in range(4)]
        val = [byteat(memrow + 5 + k) for k in range(4)]
        mvb = [round(float(pre[memrow, dimpos[f"MEM_VAL_B{k}"]].item()), 1)
               for k in range(4) if f"MEM_VAL_B{k}" in dimpos]
        pcb = steps[si][0]
        # decode the executed instr at pc-8 (the instr that produced this step)
        addr_h = "".join(f"{b:02x}" for b in reversed(addr))
        val_h = "".join(f"{b:02x}" for b in reversed(val))
        print(f"  {si:2d} | pc={pcb:4d} | MEM_STORE={mst:.1f} | "
              f"addr=0x{addr_h} | val=0x{val_h} | MEM_VAL_B={mvb}")


if __name__ == "__main__":
    main()
