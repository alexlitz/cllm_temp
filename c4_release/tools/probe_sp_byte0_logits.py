#!/usr/bin/env python3
"""At the ENT-step SP-byte0 prediction row of id262 (the row right AFTER the
REG_SP marker), dump the FULL top-20 token logits + the byte-token logit for
0xe8 (232) vs the winning REG_PC marker (257). Shows the byte-vs-marker race
that the brief describes. spec_k=0, hook-free, true autoregressive context.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_CSR_INFERENCE"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe, _MARKER_NAMES  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END"}


def lbl(t):
    return _MARKER_NAMES.get(t, f"byte0x{t:02X}" if t < 256 else f"tok{t}")


def main():
    probe = build_groundtruth_probe()
    m = probe.model; dev = next(m.parameters()).device
    src, exp, _ = generate_test_programs()[262]
    bc = compile_c(src)[0]
    trace = probe.probe(bc, top_k=20, max_steps=9)
    # Walk to the step-1 SP marker, then the byte0 row right after it.
    step = 0; sp_pos = None
    for p in sorted(trace.keys()):
        tok = trace[p]["token"]; nm = MARKERS.get(tok)
        if nm == "STEP_END":
            step += 1; continue
        if nm == "SP" and step == 1:
            sp_pos = p; break
    print(f"step1 SP marker @pos {sp_pos}")
    byte0_pos = sp_pos + 1
    rec = trace[byte0_pos]
    print(f"\nSP byte0 row @pos {byte0_pos}: emitted {lbl(rec['token'])}")
    print("top-20 logits:")
    for t, v in rec["top_k_logits"]:
        star = " <-- want 0xE8" if t == 0xE8 else (" <-- emitted" if t == rec["token"] else "")
        print(f"   {t:4d} {lbl(t):<12} {v:+.4f}{star}")
    # explicit 0xe8 logit even if not in top-20
    e8 = next((v for t, v in rec["top_k_logits"] if t == 0xE8), None)
    print(f"\n0xE8 byte logit = {e8}  | REG_PC(257) logit = "
          f"{next((v for t,v in rec['top_k_logits'] if t==257), None)}")


if __name__ == "__main__":
    main()
