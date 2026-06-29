#!/usr/bin/env python3
"""func_add step-13 TEACHER-FORCED probe: model argmax vs oracle at AX-byte rows.

Single forward over the oracle-teacher-forced tape. For step 13 prints the model's
ARGMAX token at each AX-byte-emitting row (which IS what production decodes) and the
OUTPUT_LO/HI residual there, so we localize WHICH AX byte the model gets wrong and
what the residual band says — without the slow AR loop.

  python tools/probe_funcadd_tf.py [step]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PROGS = {
    "add_57_11": ("int add(int a, int b) { return a + b; } "
                  "int main() { return add(57, 11); }", 57, 11, 68),
    "add_36_16": ("int add(int a, int b) { return a + b; } "
                  "int main() { return add(36, 16); }", 36, 16, 52),
    "add_8_30":  ("int add(int a, int b) { return a + b; } "
                  "int main() { return add(8, 30); }", 8, 30, 38),
}

STEP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 13


def fmt(row, base, width=16, thr=0.5):
    vals = [float(row[base + i].item()) for i in range(width)]
    idx = sorted(range(width), key=lambda i: -vals[i])[:5]
    return "[" + ", ".join(f"{vals[i]:.1f}@{i}" for i in idx if abs(vals[i]) > thr) + "]"


def main():
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    for pname, (src, aval, bval, exp) in PROGS.items():
        bc, data = compile_c(src)
        _, oracle_tokens = runner._oracle_pc_ax_steps(
            bc, data or b"", "", expected_steps=None, with_tokens=True)
        prompt = runner._build_element(bc, data or b"", [], "", spec_k=1,
                                       adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in oracle_tokens:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model.forward(padded)[0]         # [S, V]
            resid = model.forward(padded, stop_after_block=len(model.blocks) - 1)[0]
        amax = logits.argmax(dim=-1)                   # [S]

        start = len(prefix) + STEP_N * STEP
        toks = tape[start:start + STEP]
        ax_off = next((i for i, t in enumerate(toks) if t == int(Token.REG_AX)), None)
        print(f"=== {pname} a={aval}=0x{aval:02X} b={bval}=0x{bval:02X} "
              f"exp={exp}=0x{exp:02X} step={STEP_N} ax_marker@off{ax_off} ===",
              flush=True)
        if ax_off is None:
            print("  no AX marker", flush=True)
            continue
        # oracle AX bytes (teacher-forced tape) vs model argmax at the emitting row
        for j in range(4):
            tok_pos = start + ax_off + 1 + j   # position of AX byte j token
            emit_row = tok_pos - 1             # row that predicts it
            oracle_b = int(tape[tok_pos]) & 0xFF
            model_b = int(amax[emit_row].item()) & 0xFF
            r = resid[emit_row]
            flag = "  <-- DIVERGE" if model_b != oracle_b else ""
            print(f"  AXbyte{j}: oracle=0x{oracle_b:02X} model_argmax=0x{model_b:02X}{flag}",
                  flush=True)
            print(f"           row off{ax_off+1+j-1}: "
                  f"OUT_LO={fmt(r, dp['OUTPUT_LO'])} OUT_HI={fmt(r, dp['OUTPUT_HI'])}",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
