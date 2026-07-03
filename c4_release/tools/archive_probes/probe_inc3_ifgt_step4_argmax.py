#!/usr/bin/env python3
"""Inc-3 if_gt: per-token-position LM-head argmax for the BZ branch step (step4),
campaign config. Runs the production autoregressive decode, then RE-FORWARDS the
captured tape and prints, for each position in [step4_start, step4_start+37], the
argmax token + top-3 logits. Localizes WHICH position emits the spurious PC
marker (257) that over-emits the branch-taken step.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
      PROBE_SRC='int main(){ if (35 > 43) return 1; return 0; }' \
      python tools/probe_inc3_ifgt_step4_argmax.py
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
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")

_NAMES = {257: "REG_PC", 258: "REG_AX", 259: "REG_SP", 260: "REG_BP",
          261: "MEM", 262: "STEP_END", 263: "HALT", 268: "STACK0"}


def tname(t):
    return _NAMES.get(int(t), f"byte{int(t)}" if 0 <= t < 256 else f"tok{int(t)}")


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    STEP = int(Token.STEP_TOKENS)
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    ctx = p._final_context(bytecode, max_steps=8)
    prompt_len = len(p._build_context(bytecode))
    # Re-forward the full captured tape; the logit at position i predicts token i+1.
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    with torch.no_grad():
        logits = p.model.forward(padded)[0]  # [T, vocab]
    # step4 production slice begins at prompt_len + 4*STEP.
    s4 = prompt_len + 4 * STEP
    print(f"=== {cfg} STEP={STEP} src={SRC!r} step4_start={s4} ===")
    print(f"    (production fixed slice step4 = ctx[{s4}:{s4+STEP}])")
    # Show 38 emitted positions starting at step4 (covers the over-emission).
    for i in range(s4, min(s4 + 38, len(ctx) - 1)):
        emitted = ctx[i]              # the token actually at position i
        pred = logits[i - 1].argmax().item()  # logit at i-1 predicts token i
        top = torch.topk(logits[i - 1], 3)
        top_s = " ".join(f"{tname(t)}={v:.0f}" for v, t in
                         zip(top.values.tolist(), top.indices.tolist()))
        flag = "  <<<" if int(emitted) in (257, 258, 259, 260, 261, 268) else ""
        print(f" pos{i-s4:+3d} (abs {i}) emitted={tname(emitted):8s} | top3[pred]: {top_s}{flag}")


if __name__ == "__main__":
    main()
