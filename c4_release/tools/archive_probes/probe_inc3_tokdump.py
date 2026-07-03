#!/usr/bin/env python3
"""Inc-3: dump the emitted AX bytes for var_simple_0 step 2 in either config."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token, _step_offset_field  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"


def tok_byte(t):
    # Byte-value tokens are 0..255 directly in this vocab (markers are >=256).
    return t if 0 <= t < 256 else None


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    STEP = int(Token.STEP_TOKENS)
    ctx = p._final_context(bytecode, max_steps=6)
    prompt_len = len(p._build_context(bytecode))
    emitted = ctx[prompt_len:]
    print(f"=== {cfg} STEP={STEP} prompt_len={prompt_len} emitted={len(emitted)} ===")
    for step in range(min(4, len(emitted) // STEP)):
        sl = emitted[step * STEP:(step + 1) * STEP]
        # AX bytes at offsets 6,7,8,9
        axb = sl[6:10]
        print(f" step{step}: AX bytes = {[hex(b) for b in axb]}  "
              f"AX={(axb[0] | axb[1] << 8 | axb[2] << 16 | axb[3] << 24) if len(axb)==4 else '?'}")


if __name__ == "__main__":
    main()
