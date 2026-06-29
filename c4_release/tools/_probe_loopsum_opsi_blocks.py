#!/usr/bin/env python3
"""For loop_sum id450 step5 (SI i=1) and step9 (SI sum=0), trace OP_SI at the
AX-marker row across ALL blocks, to find where the SI opcode broadcast dies
(vs OP_PSH at step12 which survives). Compares to the prologue JSR step0.

Run: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     C4_VM_CACHE_DIR=/tmp/c4cache_loops11 python tools/_probe_loopsum_opsi_blocks.py [pid]
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
STEPS_OF_INTEREST = [5, 9, 12, 0]


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
    nblk = len(model.blocks)

    src, exp, desc = generate_test_programs()[PID]
    bc = compile_c(src)[0]
    prompt = p._build_context(bc)
    pl = len(prompt)
    steps, toks = oracle_windows(bc, 14)
    ctx = list(prompt)
    for w in toks:
        ctx.extend(w)
    padded = torch.tensor([ctx], device=p._device)

    caps = {}

    def mk(bi):
        def hook(module, inputs):
            caps[bi] = inputs[0].detach()
        return hook

    handles = [model.blocks[bi].register_forward_pre_hook(mk(bi)) for bi in range(nblk)]
    with torch.no_grad():
        model(padded)
    for h in handles:
        h.remove()

    # find AX marker rows in each step at the EMBED input (block 0)
    pre0 = caps[0][0]
    seq = pre0.shape[0]

    def axrow(si):
        base = pl + si * STEP
        for j in range(base, min(base + STEP, seq)):
            if pre0[j, dimpos["MARK_AX"]].item() > 0.5:
                return j
        return None

    dims = {"OP_SI": dimpos["OP_SI"], "OP_PSH": dimpos["OP_PSH"],
            "OP_JSR": dimpos["OP_JSR"], "OPCODE_BYTE_LO": dimpos.get("OPCODE_BYTE_LO")}
    print(f"nblk={nblk} pl={pl} STEP={STEP}")
    for si in STEPS_OF_INTEREST:
        r = axrow(si)
        pcb = steps[si][0]
        print(f"\n=== step{si} pc={pcb} AXrow={r} ===")
        # opcode byte lo nibble (one-hot) at the AX row across blocks 0..6
        for bi in [0, 3, 5, 6, 7, 8, 10, 12, 14, 35]:
            if bi >= nblk or bi not in caps:
                continue
            pre = caps[bi][0]
            si_v = float(pre[r, dimpos["OP_SI"]].item())
            psh_v = float(pre[r, dimpos["OP_PSH"]].item())
            jsr_v = float(pre[r, dimpos["OP_JSR"]].item())
            # opcode byte lo one-hot
            obl = pre[r, dimpos["OPCODE_BYTE_LO"]:dimpos["OPCODE_BYTE_LO"]+16]
            obl_i = int(obl.argmax().item()) if float(obl.max()) > 0.3 else -1
            obh = pre[r, dimpos["OPCODE_BYTE_HI"]:dimpos["OPCODE_BYTE_HI"]+16]
            obh_i = int(obh.argmax().item()) if float(obh.max()) > 0.3 else -1
            print(f"  blk{bi:2d}: OP_SI={si_v:7.3f} OP_PSH={psh_v:7.3f} "
                  f"OP_JSR={jsr_v:7.3f} OPCODE_BYTE(hi|lo)={obh_i:x}{obl_i:x}")


if __name__ == "__main__":
    main()
