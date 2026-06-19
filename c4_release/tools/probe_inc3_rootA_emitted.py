#!/usr/bin/env python3
"""Inc-3 ROOT A: autoregressively decode x=990 and print the emitted AX token
bytes per step, GOLDEN vs CAMPAIGN. Ground truth of what value reaches AX.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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

SRC = "int main() { int x; x = 990; return x; }"


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    STEP = int(Token.STEP_TOKENS)
    ctx = p._final_context(bytecode, max_steps=9)
    pl = len(p._build_context(bytecode))
    print(f"=== {cfg} STEP={STEP} prompt_len={pl} total={len(ctx)} ===")
    REG_AX = int(Token.REG_AX)
    REG_PC = int(Token.REG_PC)
    # Walk emitted tokens; find REG_AX markers and print the 4 bytes after.
    for i in range(pl, len(ctx)):
        if ctx[i] == REG_AX and i + 4 < len(ctx):
            bs = [ctx[i + 1 + j] & 0xFF for j in range(4)]
            val = sum((bs[j] & 0xFF) << (8 * j) for j in range(4))
            step = (i - pl) // STEP
            print(f"  step{step} @pos{i} REG_AX bytes={[hex(b) for b in bs]} "
                  f"= {val}")
        if ctx[i] == REG_PC and i + 4 < len(ctx):
            bs = [ctx[i + 1 + j] & 0xFF for j in range(4)]
            val = sum((bs[j] & 0xFF) << (8 * j) for j in range(4))
            step = (i - pl) // STEP
            print(f"  step{step} @pos{i} REG_PC bytes={[hex(b) for b in bs]} "
                  f"= {val}")


if __name__ == "__main__":
    main()
