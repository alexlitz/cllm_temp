#!/usr/bin/env python3
"""Inc-3 if_gt: build the ORACLE's clean per-step token frame (the DraftVM
proposal that the production spec-decoder verifies), then re-forward it and read
the MODEL's argmax at every SAFE offset of the branch step (step4). Pinpoints
which offset the model CORRECTS away from the oracle (== the production
divergence), and what value it emits there. This mirrors the spec-decode verify
loop exactly (model argmax vs oracle draft, UNSAFE offsets skipped).

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_branchpc.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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
from neural_vm.batched_pure_neural import Token, _UNSAFE_OFFSETS  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")
TARGET_STEP = int(os.environ.get("PROBE_STEP", "4"))


def main():
    STEP = int(Token.STEP_TOKENS)
    bytecode, data = compile_c(SRC)
    p = build_groundtruth_probe()
    runner = p.runner
    # Build the oracle pc/ax steps + the full per-step token frames.
    oracle_steps, oracle_tokens = runner._oracle_pc_ax_steps(
        bytecode, data or b"", "", expected_steps=None, with_tokens=True)
    # Build the prompt + the oracle's clean tape (concatenate per-step frames).
    prompt = runner._build_element(bytecode, data or b"", [], "",
                                   spec_k=1, adaptive_start_k=0,
                                   expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=p._device)
    with torch.no_grad():
        logits = p.model.forward(padded)[0]   # [T, vocab]
    base = len(prefix) + TARGET_STEP * STEP
    dp = dict(p.model.dim_positions)
    inv = {}
    for nm, idx in dp.items():
        inv.setdefault(int(idx), nm)
    head_w = p.model.head.weight
    with torch.no_grad():
        resid = p.model.forward(padded, stop_after_block=len(p.model.blocks) - 1)[0]
    print(f"=== STEP={STEP} src={SRC!r} step{TARGET_STEP} base={base} "
          f"oracle_pc_ax(step{TARGET_STEP})="
          f"{oracle_steps[TARGET_STEP] if TARGET_STEP < len(oracle_steps) else '?'} ===")
    if TARGET_STEP >= len(oracle_tokens):
        print("  (step beyond oracle length)")
        return
    frame = oracle_tokens[TARGET_STEP]
    tn = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "MEM", 262: "SE", 268: "ST0"}
    for off in range(min(STEP, len(frame))):
        draft = int(frame[off])
        # model argmax predicting this position = logits[base+off-1]
        pred = int(logits[base + off - 1].argmax().item())
        unsafe = (off % STEP) in _UNSAFE_OFFSETS
        mark = ""
        if not unsafe and pred != draft:
            mark = "   <<< MODEL CORRECTS"
        dn = tn.get(draft, str(draft))
        pn = tn.get(pred, str(pred))
        print(f"  off {off:2d} {'UNSAFE' if unsafe else '      '} "
              f"oracle={dn:5s} model_argmax={pn:5s}{mark}")
        if mark and not unsafe and draft < 256 and pred < 256:
            # Decompose the byte logit at this correcting position.
            prow = base + off - 1
            r = resid[prow]
            for bv in (pred, draft):
                w = head_w[bv]
                contrib = torch.nan_to_num(w * r, nan=0.0, posinf=1e30, neginf=-1e30)
                tops = contrib.abs().topk(5).indices.tolist()
                ds = " ".join(
                    f"{inv.get(d,'d'+str(d))}={r[d].item():+.2e}*{w[d].item():.1f}"
                    f"={contrib[d].item():+.2e}" for d in tops)
                print(f"        byte{bv} logit={contrib.sum().item():+.3e} | {ds}")


if __name__ == "__main__":
    main()
