#!/usr/bin/env python3
"""AUTOREGRESSIVE SP/BP drift probe (campaign config).

Runs the spec_k=0 GPU autoregressive decode (GroundTruthProbe.probe — feeds the
emitted tokens back) for a target program, then decodes EACH step window's
PC/AX/SP/BP register byte values from the emitted tape and compares to the
DraftVM oracle. The point: SP/BP DRIFT in the autoregressive stream is HIDDEN by
teacher forcing, so we must read the model's OWN emitted register bytes.

Run:
  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    python tools/_probe_spbp_drift.py 325
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"; os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from neural_vm.speculative import DraftVM  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARK = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", int(Token.STEP_END): "STEP_END",
        int(Token.HALT): "HALT"}


def oracle_steps(bc, n=40):
    vm = DraftVM(list(bc))
    out = []
    for _ in range(n):
        if vm.halted:
            break
        if not vm.step():
            break
        out.append(dict(pc=int(vm.pc) & 0xFFFFFFFF, ax=int(vm.ax) & 0xFFFFFFFF,
                        sp=int(vm.sp) & 0xFFFFFFFF, bp=int(vm.bp) & 0xFFFFFFFF))
        if vm.halted:
            break
    return out


def decode_window(ctx, base, step_tokens):
    """Decode the 4 registers (PC/AX/SP/BP) from a step window starting at base.

    Layout: REG_PC + 4 bytes, REG_AX + 4 bytes, REG_SP + 4 bytes, REG_BP + 4
    bytes, [MEM section...], STEP_END. Scan for each marker token.
    """
    regs = {}
    i = base
    end = min(base + step_tokens, len(ctx))
    while i < end:
        t = ctx[i]
        nm = MARK.get(t)
        if nm in ("PC", "AX", "SP", "BP"):
            if i + 4 < len(ctx):
                val = 0
                for j in range(4):
                    val |= (ctx[i + 1 + j] & 0xFF) << (8 * j)
                regs[nm] = val
            i += 5
            continue
        i += 1
    return regs


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 325
    p = build_groundtruth_probe()
    STEP = int(Token.STEP_TOKENS)
    tests = generate_test_programs()
    src, exp, name = tests[idx]
    bc, _ = compile_c(src)
    print(f"=== id{idx} {name} ===\n  src: {src}")
    print(f"  STEP_TOKENS = {STEP}  (campaign={os.environ.get('C4_NO_STACK0_EMIT')})")

    osteps = oracle_steps(bc)
    nsteps = len(osteps)
    prompt = p._build_context(bc)
    pl = len(prompt)

    ctx = p._final_context(bc, max_steps=nsteps + 2)
    seq = len(ctx)
    print(f"  prompt_len={pl} seq={seq} nsteps_oracle={nsteps}")

    print("\n  step  reg  oracle      got         match")
    for si in range(nsteps):
        base = pl + si * STEP
        if base >= seq:
            print(f"  s{si}: window past seq end")
            break
        got = decode_window(ctx, base, STEP)
        orc = osteps[si]
        for r in ("PC", "AX", "SP", "BP"):
            ov = orc[r.lower()]
            gv = got.get(r)
            gvs = f"0x{gv:08x}" if gv is not None else "----"
            m = "OK" if (gv is not None and gv == ov) else "<-- DRIFT"
            print(f"  s{si:2d}   {r}   0x{ov:08x}  {gvs}  {m}")
        print()


if __name__ == "__main__":
    main()
