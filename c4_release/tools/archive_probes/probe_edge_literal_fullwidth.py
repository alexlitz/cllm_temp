#!/usr/bin/env python3
"""Validate the full-width byte-1 emission on the edge_literal cluster.

Runs the spec_k=0 ground-truth probe on several `return <literal>` programs
(byte1 >= 16) and checks the emitted AX byte-1. Honors C4_AX_BYTE1_FULL_WIDTH;
run twice (flag off vs on) to see the fix.

Run: CUDA_VISIBLE_DEVICES=0 C4_AX_BYTE1_FULL_WIDTH=1 python tools/probe_edge_literal_fullwidth.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.compiler import compile_c  # noqa: E402
from tools.probe_groundtruth import GroundTruthProbe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402

# edge_literal-style values (byte1 >= 16) + a control (byte1 < 16).
VALUES = [5561, 9647, 9257, 5182, 18293, 50007, 2563, 768]


def main():
    flag = os.environ.get("C4_AX_BYTE1_FULL_WIDTH", "0")
    print(f"C4_AX_BYTE1_FULL_WIDTH={flag}", flush=True)
    probe = GroundTruthProbe.build()
    RAX = int(Token.REG_AX)
    npass = 0
    ntotal = 0
    for val in VALUES:
        bc, _ = compile_c(f"int main() {{ return {val}; }}")
        trace = probe.probe(bc, max_steps=3)
        markers = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        m = markers[-1]
        bs = [trace.get(m + 1 + b, {}).get("token") for b in range(4)]
        got = (bs[0] or 0) | ((bs[1] or 0) << 8) \
            | ((bs[2] or 0) << 16) | ((bs[3] or 0) << 24)
        exp_b1 = (val >> 8) & 0xFF
        got_b1 = (got >> 8) & 0xFF
        ok = (got == val)
        ntotal += 1
        npass += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'} return {val:6d} "
              f"(0x{val:04x}): exp_b1=0x{exp_b1:02x} got_b1=0x{got_b1:02x} "
              f"full got={got}", flush=True)
    print(f"\n{npass}/{ntotal} fully correct", flush=True)


if __name__ == "__main__":
    main()
