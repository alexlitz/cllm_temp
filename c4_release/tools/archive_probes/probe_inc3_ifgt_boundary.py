#!/usr/bin/env python3
"""Inc-3 if_gt: dump the CONTIGUOUS emitted token stream and split it on the
STEP_END marker (Token.STEP_END = 262) so we can SEE how many tokens each step
actually emitted (the fixed-STEP slice hides over/under-emission). Localizes the
step-4 BZ-branch framing drift for if_gt.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
      PROBE_SRC='int main(){ if (35 > 43) return 1; return 0; }' \
      python tools/probe_inc3_ifgt_boundary.py
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
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")
STEP_END = int(Token.STEP_END)  # 262
HALT = int(Token.HALT)          # 263


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    STEP = int(Token.STEP_TOKENS)
    ctx = p._final_context(bytecode, max_steps=8)
    prompt_len = len(p._build_context(bytecode))
    emitted = ctx[prompt_len:]
    print(f"=== {cfg} STEP={STEP} prompt_len={prompt_len} emitted={len(emitted)} "
          f"src={SRC!r} ===")
    # Split on STEP_END/HALT markers (the production step boundary, regardless of
    # how many tokens were emitted).
    step = 0
    cur = []
    for t in emitted:
        cur.append(t)
        if t in (STEP_END, HALT):
            print(f" step{step}: n={len(cur):2d} {'<<< EXTRA' if len(cur)!=STEP else ''} {cur}")
            step += 1
            cur = []
            if t == HALT:
                break
    if cur:
        print(f" step{step}: n={len(cur):2d} (no terminator) {cur}")


if __name__ == "__main__":
    main()
