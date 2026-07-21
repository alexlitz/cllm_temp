"""DIAGNOSTIC (Family-2 width): pin where compare/branch truncates.

Runs three probes:
  1. The compare EXPERT weights (compile_cmp_expert) directly at 8-bit vs >255
     operands -> does LT/GT/EQ produce the correct 0/1 for large values?
  2. The zero_detector / _step_ge1 primitives at large d (does the fp32 second
     difference still cancel when S*d is millions?).
  3. The recompose -> requant round-trip: can a value >255 survive one step?

Memory-lean: builds nothing large; probes weights + tiny forwards only.
Run: OMP_NUM_THREADS=4 PYTHONPATH=<repo> python -m c4_min._diag_family2_width
"""
from __future__ import annotations
import torch

from . import isa
from .nibble_vm_layout import NibbleVMLayout
from .nibble_unified import (
    compile_cmp_expert, _z_detector_units, _step_ge1_unit,
)
from .nibble_vm import _empty_spec, compile_nibble_to_scalar
from .nibble_moe import _ffn_from_spec


def _run_ffn_spec(spec, dim, state_overrides):
    """Apply one compiled FFN spec to a single residual position (attn=identity)."""
    ffn = _ffn_from_spec(spec, dim)
    x = torch.zeros(dim, dtype=torch.float32)
    for d, v in state_overrides.items():
        x[d] = v
    with torch.no_grad():
        y = ffn(x.view(1, 1, -1))[0, 0]
    return y


def probe_compare_expert():
    print("=" * 70)
    print("PROBE 1: compare EXPERT (compile_cmp_expert) at various magnitudes")
    print("=" * 70)
    L = NibbleVMLayout(8, n_heads=4)
    dim = L.D
    cases = [
        (3, 5), (5, 3), (5, 5),
        (200, 100), (100, 200), (255, 255),
        (300, 100), (100, 300), (300, 300),     # >255
        (1000, 5), (5, 1000), (1000, 1000),     # loop n=1000
        (65535, 1), (1, 65535),                 # near requant ceiling
    ]
    for op in (isa.LT, isa.GT, isa.EQ):
        name = isa.NAMES[op]
        spec = compile_cmp_expert(L, op, dim)
        print(f"\n  {name}:")
        for a, b in cases:
            y = _run_ffn_spec(spec, dim, {L.STK_VAL: float(a), L.AX_VAL: float(b),
                                          L.ONE: 1.0})
            final = float(y[L.AX_VAL])   # SET-write: y already IS the new AX value
            if op == isa.LT:
                exp = 1 if a < b else 0
            elif op == isa.GT:
                exp = 1 if a > b else 0
            else:
                exp = 1 if a == b else 0
            ok = abs(final - exp) < 0.4
            flag = "" if ok else "  <-- WRONG"
            print(f"    a={a:6d} b={b:6d}  final_AX={final:+9.4f}  exp={exp}{flag}")


def probe_primitives():
    print("\n" + "=" * 70)
    print("PROBE 2: zero_detector / step_ge1 primitives at large |d|")
    print("=" * 70)
    L = NibbleVMLayout(8, n_heads=4)
    dim = L.D
    stk, ax, one = L.STK_VAL, L.AX_VAL, L.ONE
    print("\n  Z(d) (EQ bump, want 1.0 at d==0, ~0 else):")
    for d in [0, 1, -1, 45, 100, 255, 256, 300, 1000, 65535]:
        spec = _empty_spec(dim, 8)
        _z_detector_units(spec, 0, {stk: 1.0, ax: -1.0}, 0.0, ax, 1.0, one)
        y = _run_ffn_spec(spec, dim, {stk: float(d), ax: 0.0, one: 1.0})
        print(f"    d={d:7d}  Z={float(y[ax]):+.6f}")
    print("\n  step(d>=1) (GT ramp, want 1.0 for d>=1, 0 for d<=0):")
    for d in [0, 1, 2, 45, 255, 256, 300, 1000, 65535]:
        spec = _empty_spec(dim, 8)
        _step_ge1_unit(spec, 0, {stk: 1.0, ax: -1.0}, 0.0, ax, 1.0, one)
        y = _run_ffn_spec(spec, dim, {stk: float(d), ax: 0.0, one: 1.0})
        print(f"    d={d:7d}  step={float(y[ax]):+.6f}")


def probe_recompose():
    print("\n" + "=" * 70)
    print("PROBE 3: recompose (nibble->scalar) hi_nibbles=5 range")
    print("=" * 70)
    from . import blogspec_vocab as Vv
    L = NibbleVMLayout(8, n_heads=4)
    dim = L.D
    spec = compile_nibble_to_scalar(L, dim)
    for v in [42, 255, 256, 300, 1000, 65535, 65536, 100000]:
        ov = {L.ONE: 1.0}
        for j, nv in enumerate(Vv.nibbles_of_value(v, 16)):
            ov[L.STACK0 + j] = float(nv)
        y = _run_ffn_spec(spec, dim, ov)
        got = float(y[L.STK_VAL])
        ok = abs(got - v) < 0.5
        print(f"    v={v:8d}  recomposed STK_VAL={got:14.4f}  {'ok' if ok else '<-- LOSS'}")


if __name__ == "__main__":
    probe_compare_expert()
    probe_primitives()
    probe_recompose()
