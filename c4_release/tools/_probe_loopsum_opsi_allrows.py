#!/usr/bin/env python3
"""For loop_sum id450 step5 (SI), dump OP_SI + OPCODE_BYTE at EVERY row of the
step window at several blocks, to find whether OP_SI / the SI opcode byte fires
anywhere (PC marker? a particular byte row?) — vs the PSH step12 where OP_PSH
is clean. Localizes whether the SI opcode is mis-fetched or cleared.

Run: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     C4_VM_CACHE_DIR=/tmp/c4cache_loops11 python tools/_probe_loopsum_opsi_allrows.py [pid] [step] [block]
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
STEP_I = int(sys.argv[2]) if len(sys.argv) > 2 else 5
BLK = int(sys.argv[3]) if len(sys.argv) > 3 else 7


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

    def hook(module, inputs):
        cap["pre"] = inputs[0].detach()

    h = model.blocks[BLK].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(padded)
    h.remove()
    pre = cap["pre"][0]
    seq = pre.shape[0]

    markers = {dimpos["MARK_PC"]: "PC", dimpos["MARK_AX"]: "AX",
               dimpos["MARK_SP"]: "SP", dimpos["MARK_BP"]: "BP",
               dimpos["MARK_MEM"]: "MEM", dimpos.get("MARK_STACK0", -1): "STK0"}

    base = pl + STEP_I * STEP
    pcb = steps[STEP_I][0]
    print(f"step{STEP_I} pc={pcb} block={BLK} window=[{base},{base+STEP}) "
          f"opcode SI=11=0x0b")
    print("row | marker | OP_SI OP_PSH OP_JSR OP_ENT | OPCODE(hi|lo) | MEM_STORE")
    for j in range(base, min(base + STEP, seq)):
        mk = ""
        for d, nm in markers.items():
            if d >= 0 and pre[j, d].item() > 0.5:
                mk += nm + ","
        si_v = float(pre[j, dimpos["OP_SI"]].item())
        psh_v = float(pre[j, dimpos["OP_PSH"]].item())
        jsr_v = float(pre[j, dimpos["OP_JSR"]].item())
        ent_v = float(pre[j, dimpos["OP_ENT"]].item())
        obl = pre[j, dimpos["OPCODE_BYTE_LO"]:dimpos["OPCODE_BYTE_LO"]+16]
        obh = pre[j, dimpos["OPCODE_BYTE_HI"]:dimpos["OPCODE_BYTE_HI"]+16]
        oli = int(obl.argmax().item()) if float(obl.max()) > 0.3 else -1
        ohi = int(obh.argmax().item()) if float(obh.max()) > 0.3 else -1
        ms = float(pre[j, dimpos["MEM_STORE"]].item())
        flag = ""
        if abs(si_v) > 0.3 or abs(psh_v) > 0.3 or abs(jsr_v) > 0.3:
            flag = " <<"
        print(f" {j:3d} | {mk:8s} | {si_v:6.2f} {psh_v:6.2f} {jsr_v:6.2f} "
              f"{ent_v:6.2f} | {ohi:x}{oli:x}      | {ms:6.2f}{flag}")


if __name__ == "__main__":
    main()
