#!/usr/bin/env python3
"""Inc-3: AUTOREGRESSIVE per-step register-byte dump for a var program, using the
faithful CPU autoregressive runner (the production decode), so the dump reflects
the SAME token stream the full_trace verdict reads. Prints PC/AX/SP/BP bytes per
step. Used to localize WHICH byte at the diverging step is wrong.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 PROBE_SRC='int main(){int x;x=28;return x;}' \
      python tools/probe_inc3_ar_stepdump.py
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

SRC = os.environ.get("PROBE_SRC", "int main() { int x; x = 28; return x; }")


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    STEP = int(Token.STEP_TOKENS)
    # Autoregressive decode: feed emitted tokens back. Use _final_context with a
    # large max_steps to get the full production tape.
    ctx = p._final_context(bytecode, max_steps=10)
    prompt_len = len(p._build_context(bytecode))
    emitted = ctx[prompt_len:]
    nsteps = len(emitted) // STEP
    print(f"=== {cfg} STEP={STEP} prompt_len={prompt_len} nsteps={nsteps} src={SRC!r} ===")
    # Register byte offsets within a step (30-tok: PC0..4, AX5..9, SP10..14,
    # BP15..19, MEM20..28, SE29; 35-tok adds STACK0 25..29 shifting MEM/SE).
    for step in range(nsteps):
        sl = emitted[step * STEP:(step + 1) * STEP]

        def reg(o):
            b = [t if 0 <= t < 256 else None for t in sl[o + 1:o + 5]]
            if None in b:
                return f"[{','.join(hex(x) if x is not None else '?' for x in b)}]"
            return hex(b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24)
        pc = reg(0)
        ax = reg(5)
        sp = reg(10)
        bp = reg(15)
        print(f" step{step}: PC={pc} AX={ax} SP={sp} BP={bp}  raw={sl}")


if __name__ == "__main__":
    main()
