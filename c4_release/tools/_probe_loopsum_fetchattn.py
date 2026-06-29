#!/usr/bin/env python3
"""Trace the L5 opcode-fetch head-1 attention at the AX row of loop_sum id450
step5 (SI, relayed PC=58, should fetch opcode 0x0B) vs step4 (IMM, works). Which
code-prompt row does the CAM attend to, and what address-key does it carry? Reveals
WHY the SI PC mis-fetches opcode 0x00.

Run: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     C4_VM_CACHE_DIR=/tmp/c4cache_loops11 python tools/_probe_loopsum_fetchattn.py [pid] [step]
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
FETCH_HEAD = 1  # non-first-step opcode fetch at AX


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

    # The opcode-fetch is the FIRST block whose head-1 O writes OPCODE_BYTE_LO.
    # Detect: block where input has no OPCODE_BYTE but output does. Simpler: it's
    # the L5 block; find it by W_o writing into OPCODE_BYTE_LO with large magnitude.
    # OPCODE_BYTE band first materializes between block 6 and 7 (probed).
    # The opcode-fetch attention is block 6. Allow override via env.
    L5 = int(os.environ.get("C4_FETCH_BLOCK", "6"))
    print(f"L5 opcode-fetch block = {L5}")

    cap = {}

    def hook(module, inputs):
        cap["pre"] = inputs[0].detach()

    h = model.blocks[L5].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(padded)
    h.remove()
    pre = cap["pre"][0]
    seq = pre.shape[0]

    attn = model.blocks[L5].attn
    nH = attn.num_heads
    HD = attn.W_q.shape[0] // nH
    for w in ("W_q", "W_k"):
        t = getattr(attn, w)
        if t.is_sparse_csr or t.is_sparse:
            setattr(attn, "_d" + w, t.to_dense())
    Wq = getattr(attn, "_dW_q", attn.W_q)
    Wk = getattr(attn, "_dW_k", attn.W_k)

    base = pl + STEP_I * STEP
    axrow = None
    for j in range(base, min(base + STEP, seq)):
        if pre[j, dimpos["MARK_AX"]].item() > 0.5:
            axrow = j
            break
    pcb = steps[STEP_I][0]
    print(f"step{STEP_I} pc_before={pcb} AXrow={axrow}")
    # relayed PC at AX row (EMBED_LO/HI)
    el = pre[axrow, dimpos["EMBED_LO"]:dimpos["EMBED_LO"]+16]
    eh = pre[axrow, dimpos["EMBED_HI"]:dimpos["EMBED_HI"]+16]
    print(f"  relayed PC EMBED(hi|lo) = {int(eh.argmax()):x}{int(el.argmax()):x}")

    hb = FETCH_HEAD * HD
    qv = (pre[axrow] @ Wq.T)[hb:hb+HD]
    K = (pre @ Wk.T).view(seq, nH, HD)[:, FETCH_HEAD, :]
    scores = (K @ qv) / (HD ** 0.5)
    scores[axrow + 1:] = float("-inf")
    probs = torch.softmax(scores, dim=0)
    topk = torch.topk(probs, k=8)
    # identify code-prompt rows (the bytecode embedding rows, pos < pl)
    print("  top-8 attended K rows (head-1 opcode fetch):")
    for r in range(8):
        kp = int(topk.indices[r].item())
        pr = float(topk.values[r].item())
        sc = float(scores[kp].item())
        # what opcode/byte does this row carry? FETCH source rows have a byte.
        el2 = pre[kp, dimpos["EMBED_LO"]:dimpos["EMBED_LO"]+16]
        eh2 = pre[kp, dimpos["EMBED_HI"]:dimpos["EMBED_HI"]+16]
        addr = f"{int(eh2.argmax()):x}{int(el2.argmax()):x}" if float(el2.max()) > 0.3 else "??"
        tok = ctx[kp] if kp < len(ctx) else -1
        region = "PROMPT" if kp < pl else "EMIT"
        print(f"    K@{kp:3d} p={pr:.3f} sc={sc:8.1f} {region} tok={tok} addr={addr}")
    # Show what the winning row's V copies (the opcode byte)
    winner = int(topk.indices[0].item())
    print(f"  winner K@{winner} (tok={ctx[winner] if winner<len(ctx) else -1})")


if __name__ == "__main__":
    main()
