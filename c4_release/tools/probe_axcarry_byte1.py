#!/usr/bin/env python3
"""Confirm AX_CARRY_LO/HI carries byte-1 (lo,hi nibbles) at the predictor row.

Uses BUILT layout positions. Sweeps byte-1 lo + hi nibbles to nail the exact
(nibble -> cell) offset for AX_CARRY_LO and AX_CARRY_HI, and checks the row
gating + opcode scope (IMM vs ADD).

Run: CUDA_VISIBLE_DEVICES=0 [C4_AX_BYTE1_FULL_WIDTH=1] python tools/probe_axcarry_byte1.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.probe_groundtruth import GroundTruthProbe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic)


def nib(row, base, thr=0.2):
    v = row[base:base + 16]
    return (int(v.argmax()) if float(v.max()) > thr else None), float(v.max())


def main():
    flag = os.environ.get("C4_AX_BYTE1_FULL_WIDTH", "0")
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = getattr(layout, "dim_positions", layout)
    AC_LO, AC_HI, ISB = dp["AX_CARRY_LO"], dp["AX_CARRY_HI"], dp["IS_BYTE"]
    print(f"flag={flag} BUILT AX_CARRY_LO={AC_LO} AX_CARRY_HI={AC_HI}",
          flush=True)
    probe = GroundTruthProbe.build()
    model = probe.model
    last = len(model.blocks) - 1
    RAX = int(Token.REG_AX)

    def run(src):
        bc, _ = compile_c(src)
        ctx = probe._final_context(bc, max_steps=4)
        trace = probe.probe(bc, max_steps=4)
        m = [p for p in sorted(trace) if trace[p]["token"] == RAX][-1]
        padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
        with torch.no_grad():
            x = model.forward(padded, stop_after_block=last)
        return x[0], m

    # nibble sweep: byte1 = 0xH5 (hi sweep) and 0x2L (lo sweep)
    print("\n-- hi-nibble sweep (byte1 = 0xH5) --")
    for hi in range(8):
        b1 = (hi << 4) | 5
        x, m = run(f"int main() {{ return {(b1 << 8) | 0x55}; }}")
        row = x[m + 1]
        lo_am, _ = nib(row, AC_LO); hi_am, _ = nib(row, AC_HI)
        print(f"  byte1=0x{b1:02x} hi={hi}: AC_LO am={lo_am} AC_HI am={hi_am}")
    print("\n-- lo-nibble sweep (byte1 = 0x2L) --")
    for lo in range(8):
        b1 = (2 << 4) | lo
        x, m = run(f"int main() {{ return {(b1 << 8) | 0x55}; }}")
        row = x[m + 1]
        lo_am, _ = nib(row, AC_LO); hi_am, _ = nib(row, AC_HI)
        print(f"  byte1=0x{b1:02x} lo={lo}: AC_LO am={lo_am} AC_HI am={hi_am}")

    # Row scope: all byte rows for 0x25; opcode scope: ADD
    print("\n-- per-row (return 9557, byte1=0x25) --")
    x, m = run("int main() { return 9557; }")
    for b in range(4):
        row = x[m + b]
        lo_am, mlo = nib(row, AC_LO); hi_am, mhi = nib(row, AC_HI)
        print(f"  marker+{b}: AC_LO am={lo_am}({mlo:.1f}) AC_HI am={hi_am}"
              f"({mhi:.1f}) IS_BYTE={float(row[ISB]):.2f}")
    print("\n-- ADD 9000+557 (byte1=0x25, arith) byte-1 row --")
    x, m = run("int main() { return 9000 + 557; }")
    row = x[m + 1]
    lo_am, mlo = nib(row, AC_LO); hi_am, mhi = nib(row, AC_HI)
    print(f"  AC_LO am={lo_am}({mlo:.1f}) AC_HI am={hi_am}({mhi:.1f})")


if __name__ == "__main__":
    main()
