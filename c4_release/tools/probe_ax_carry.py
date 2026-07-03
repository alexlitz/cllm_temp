#!/usr/bin/env python3
"""Probe the AX high-byte carry-forward truncation (id 0 canonical repro).

Reproduces the add_0 (654+114) per-step AX dump and inspects the byte-1
dump-row residual across blocks. spec_k=0, hook-free.

PORTED to ``tools/probe_lib`` — dims are resolved from the BUILT
``model.dim_positions`` via ``probe_lib.resolve`` (the old ``_pos`` read the
STALE ``build_default_registry_dynamic`` static slots, which point at the wrong
cell after the widen repack) and the REG_AX markers via
``probe_lib.register_marker_rows``.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_ax_carry.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.compiler import compile_c  # noqa: E402
from tools import probe_lib as P  # noqa: E402


def _pos(model, name):
    """BUILT-layout start dim for a band, or None if the band is absent."""
    dp = P.dim_positions(model)
    v = dp.get(name)
    return None if v is None else int(v)


SRC = {
    0: "int main() { return 654 + 114; }",   # byte1 = 0x02
    4: "int main() { return 754 + 104; }",   # byte1 = 0x02
    6: "int main() { return 913 + 558; }",   # byte1 = 0x03
}


def decode_step_bytes(probe, bytecode, max_steps):
    """Return {step: ([b0..b3], ax_marker_pos)} by locating REG_AX markers.

    The emitted-token stream is NOT perfectly 35-aligned (per-step drift), so
    we locate each REG_AX marker token and read the next 4 tokens as AX bytes.
    The byte-b predictor row (whose next-token logits emit byte b) is at
    ``ax_marker_pos + b``.
    """
    trace = probe.probe(bytecode, max_steps=max_steps)
    markers = P.register_marker_rows(trace, "REG_AX")
    out = {}
    for s, m in enumerate(markers):
        bs = P.decode_register(trace, m)
        out[s] = (bs, m)
    return out


def _nib_decode(res, lo_n, hi_n):
    lo = [res.get(f"{lo_n}+{k}", 0.0) for k in range(16)]
    hi = [res.get(f"{hi_n}+{k}", 0.0) for k in range(16)]
    li = max(range(16), key=lambda k: lo[k]) if max(lo) > 0.3 else None
    hizi = max(range(16), key=lambda k: hi[k]) if max(hi) > 0.3 else None
    val = (li | (hizi << 4)) if (li is not None and hizi is not None) else None
    return li, hizi, val, max(lo), max(hi)


def main():
    from neural_vm.batched_pure_neural import Token
    probe = P.build_probe()
    model = probe.model
    print("STEP_TOKENS =", int(Token.STEP_TOKENS))

    markers = {}
    for pid, src in SRC.items():
        bytecode, _data = compile_c(src)
        steps = decode_step_bytes(probe, bytecode, max_steps=4)
        markers[pid] = {s: m for s, (bs, m) in steps.items()}
        print(f"\n=== id {pid}: {src} ===")
        for s, (bs, m) in steps.items():
            dec = (bs[0] | (bs[1] << 8) | (bs[2] << 16) | (bs[3] << 24)
                   if all(x is not None for x in bs) else None)
            print(f"  step {s}: AX bytes {bs}  decoded={dec}  (marker pos {m})")

    # Residual inspection on byte-1 PREDICTOR row (= AX byte-0 token = marker+1).
    PAIRS = (
        ("AX_CARRY_LO", "AX_CARRY_HI"),
        ("AX_FULL_LO", "AX_FULL_HI"),
        ("OUTPUT_LO", "OUTPUT_HI"),
        ("STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI"),
    )
    SINGLE = ("STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
              "MARK_AX", "BYTE_INDEX_1", "IS_BYTE")
    dim_names = {}
    for lo_n, hi_n in PAIRS:
        for nm in (lo_n, hi_n):
            base = _pos(model, nm)
            if base is None:
                continue
            for k in range(16):
                dim_names[f"{nm}+{k}"] = base + k
    for nm in SINGLE:
        base = _pos(model, nm)
        if base is not None:
            dim_names[nm] = base

    print("\n=== id 0 byte-1 PREDICTOR row residual (block 39) ===")
    bytecode, _ = compile_c(SRC[0])
    for label, s in (("IMM-step(0)", 0), ("PSH-step(1)", 1)):
        m = markers[0][s]
        pos = m + 1  # AX byte-0 token = the byte-1 predictor row
        print(f"\n  -- {label} predictor row (pos {pos}, marker {m}) --")
        res = probe.residual_at(bytecode, block_idx=39, position=pos,
                                dim_names=dim_names, max_steps=4)
        for lo_n, hi_n in PAIRS:
            if f"{lo_n}+0" not in dim_names:
                continue
            li, hi, val, mlo, mhi = _nib_decode(res, lo_n, hi_n)
            vs = f"0x{val:02x}" if val is not None else "--"
            print(f"     {lo_n[:-3]:22s} lo={li} hi={hi} val={vs} "
                  f"(mlo={mlo:.2f} mhi={mhi:.2f})")
        for nm in SINGLE:
            if nm in dim_names:
                print(f"     {nm:22s} = {res.get(nm, 0.0):.3f}")


if __name__ == "__main__":
    main()
