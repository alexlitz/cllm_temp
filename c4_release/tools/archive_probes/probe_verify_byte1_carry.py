#!/usr/bin/env python3
"""Verify the value-general byte-1 carry: sub_4/sub_6 + a byte-1 5..7 sweep +
a fresh byte-1 <=4 byte-identity spot-check.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_verify_byte1_carry.py
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


def decode_ax(trace, marker):
    bs = [trace.get(marker + 1 + b, {}).get("token") for b in range(4)]
    if any(x is None for x in bs):
        return None, bs
    return bs[0] | (bs[1] << 8) | (bs[2] << 16) | (bs[3] << 24), bs


def run(probe, src, max_steps):
    bc, _ = compile_c(src)
    trace = probe.probe(bc, max_steps=max_steps)
    RAX = int(Token.REG_AX)
    ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
    out = []
    for s, m in enumerate(ms):
        dec, bs = decode_ax(trace, m)
        out.append((s, dec, bs))
    return out


def main():
    probe = GroundTruthProbe.build()

    print("### sub_4 / sub_6 (the brief's targets) ###")
    for label, src, want_step1 in (
        ("sub_4: 1347-81", "int main() { return 1347 - 81; }", 1347),
        ("sub_6: 1593-31", "int main() { return 1593 - 31; }", 1593),
    ):
        steps = run(probe, src, 4)
        s1 = next((d for (s, d, b) in steps if s == 1), None)
        verdict = "PASS" if s1 == want_step1 else "FAIL"
        print(f"  {label}: step1 AX={s1} (want {want_step1}) {verdict}  "
              f"all={[(s, d) for (s, d, b) in steps]}")

    print("\n### byte-1 5..7 carry sweep (a = b1*256 + 0x40, sub - 1) ###")
    for b1 in (5, 6, 7, 8, 11, 12, 15):
        a = b1 * 256 + 0x40
        steps = run(probe, f"int main() {{ return {a} - 1; }}", 4)
        s1 = next((d for (s, d, b) in steps if s == 1), None)
        verdict = "PASS" if s1 == a else "FAIL"
        print(f"  b1={b1:2d} a={a} (0x{a:04x}): step1 AX={s1} {verdict}")

    print("\n### fresh byte-1 <=4 byte-identity spot-check (must stay correct) ###")
    for b1 in (0, 1, 2, 3, 4):
        a = b1 * 256 + 0x40
        steps = run(probe, f"int main() {{ return {a} - 1; }}", 4)
        s0 = next((d for (s, d, b) in steps if s == 0), None)
        s1 = next((d for (s, d, b) in steps if s == 1), None)
        ok = (s0 == a and s1 == a)
        print(f"  b1={b1} a={a}: fresh(step0)={s0} carried(step1)={s1} "
              f"{'OK' if ok else 'BAD'}")


if __name__ == "__main__":
    main()
