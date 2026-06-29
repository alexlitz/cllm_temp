#!/usr/bin/env python3
"""For loop_sum id450, dump the RELAYED PC (EMBED_LO/HI) and the fetched opcode
byte (OPCODE_BYTE_LO/HI) at the AX marker of each step, BEFORE the opcode-fetch
block (L5) and AFTER it, to see whether the SI steps (5/9) get a wrong relayed PC
or whether the CAM mis-fetches a correct PC.

Run: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     C4_VM_CACHE_DIR=/tmp/c4cache_loops11 python tools/_probe_loopsum_pcfetch.py [pid]
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

    # Find the L5 opcode-fetch block (head 1 writes OPCODE_BYTE from EMBED CAM).
    # Use OP_LI_RELAY-based L15 detection trick reversed: opcode fetch block is
    # where OPCODE_BYTE_LO first becomes nonzero. We hook blocks 5,6,7.
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

    BLKS = [4, 5, 6, 7]
    handles = [model.blocks[bi].register_forward_pre_hook(mk(bi)) for bi in BLKS]
    with torch.no_grad():
        model(padded)
    for h in handles:
        h.remove()

    pre0 = caps[BLKS[0]][0]
    seq = pre0.shape[0]

    def axrow(si):
        base = pl + si * STEP
        for j in range(base, min(base + STEP, seq)):
            if pre0[j, dimpos["MARK_AX"]].item() > 0.5:
                return j
        return None

    def hexband(pre, r, lo, hi):
        loband = pre[r, dimpos[lo]:dimpos[lo]+16]
        hiband = pre[r, dimpos[hi]:dimpos[hi]+16]
        li = int(loband.argmax().item()) if float(loband.max()) > 0.3 else -1
        hh = int(hiband.argmax().item()) if float(hiband.max()) > 0.3 else -1
        return hh, li

    print(f"pl={pl} STEP={STEP}  (fetched word low byte should equal opcode)")
    print("step pc | EMBED(relayed PC b0) at blk4/5/6/7 | OPCODE_BYTE(hi|lo) at blk5/6/7 | FETCH(hi|lo)@blk7")
    for si in range(13):
        r = axrow(si)
        if r is None:
            continue
        pcb = steps[si][0]
        embs = []
        for b in BLKS:
            hh, li = hexband(caps[b][0], r, "EMBED_LO", "EMBED_HI")
            embs.append(f"{hh:x}{li:x}")
        ops = []
        for b in [5, 6, 7]:
            hh, li = hexband(caps[b][0], r, "OPCODE_BYTE_LO", "OPCODE_BYTE_HI")
            ops.append(f"{hh:x}{li:x}")
        fh, fl = hexband(caps[7][0], r, "FETCH_LO", "FETCH_HI")
        # actual fetched opcode from bytecode at this PC
        want_op = (bc[pcb // 8] & 0xff) if 0 <= pcb // 8 < len(bc) else None
        print(f" {si:2d} {pcb:4d} | EMBED={embs} | OPCODE={ops} | "
              f"FETCH={fh:x}{fl:x} | want_op=0x{want_op:02x}")


if __name__ == "__main__":
    main()
