"""Numerical proof: L14 addr_key substage-1 byte_off==0 COMPUTED route reproduces
the ENUMERATED per-(lo,hi) AND bank's post-FFN ADDR_KEY residual byte-for-byte.

The M8 collapse (C4_L14_BYTE_COMPUTED) replaces the substage-1 MEM_VAL_B1
(byte_off==0) 256-way ENUMERATED per-(lo,hi) AND bank with a 32-way COMPUTED
per-nibble route (byte_copy_computed_rules, additive ADDR_KEY write,
dst_hi_offset=16, threshold 2.5). This probe builds each form as a standalone
isolated PureFFN (no full model, no disk cache) and sweeps all 256 (lo,hi)
address bytes plus the non-firing contexts (gate off; MARK_MEM block) to assert
the two forms produce an IDENTICAL post-FFN ADDR_KEY residual (max cell diff).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from c4_release.neural_vm.base_layers import PureFFN  # noqa: E402
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (  # noqa: E402
    byte_copy_computed_rules,
    multi_way_and_rule,
)
from c4_release.neural_vm.unified_compiler.ir import CompilerIR  # noqa: E402

S = 100.0
STRENGTH = 2.0 / S
THRESH = 2.5
GATE = "MEM_VAL_B1"

# ---------------------------------------------------------------------------
# Dim layout (dense, only what the rules touch). ADDR_KEY packs lo at +0..+15,
# hi at +16..+31 (the same 32-wide band the real op writes).
# ---------------------------------------------------------------------------
dim_positions = {}
_pos = 0


def _alloc(name, width=1):
    global _pos
    dim_positions[name] = _pos
    for k in range(width):
        dim_positions[f"{name}+{k}"] = _pos + k
    _pos += width


for nm in [GATE, "MARK_MEM"]:
    _alloc(nm)
_alloc("ADDR_B0_LO", 16)
_alloc("ADDR_B0_HI", 16)
_alloc("ADDR_KEY", 32)
D = _pos


def resolve(name):
    if name in dim_positions:
        return dim_positions[name]
    base, off = name.rsplit("+", 1)
    return dim_positions[base] + int(off)


def enumerated_rules():
    """The golden substage-1 byte_off==0 bank (256-way)."""
    rules = []
    for hi in range(16):
        for lo in range(16):
            rules.append(multi_way_and_rule(
                name=f"enum_lohi_hi{hi:x}_lo{lo:x}",
                conditions=(
                    (GATE, 1.0),
                    (f"ADDR_B0_LO+{lo}", 1.0),
                    (f"ADDR_B0_HI+{hi}", 1.0),
                    ("MARK_MEM", -1e6),
                ),
                threshold=THRESH,
                writes=(
                    (f"ADDR_KEY+{lo}", STRENGTH),
                    (f"ADDR_KEY+{16 + hi}", STRENGTH),
                ),
            ))
    return tuple(rules)


def computed_rules():
    """The M8 COMPUTED route (32-way)."""
    return byte_copy_computed_rules(
        src_lo="ADDR_B0_LO",
        src_hi="ADDR_B0_HI",
        dst_lo="ADDR_KEY",
        dst_hi="ADDR_KEY",
        dst_hi_offset=16,
        base_conditions=(
            (GATE, 1.0),
            ("MARK_MEM", -1e6),
        ),
        threshold=THRESH,
        strength=STRENGTH,
        additive=True,
        name_for=lambda band, ch: f"computed_{band}_{ch}",
    )


def build_ffn(rules):
    ffn = PureFFN(D, len(rules))
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    ir.lower_ffn(ffn, dim_positions, S=S)
    return ffn


def make_input(byte_val, *, gate_on=True, mem_block=False):
    x = torch.zeros(1, 1, D)
    if gate_on:
        x[0, 0, resolve(GATE)] = 1.0
    if mem_block:
        x[0, 0, resolve("MARK_MEM")] = 1.0
    lo = byte_val & 0xF
    hi = (byte_val >> 4) & 0xF
    x[0, 0, resolve(f"ADDR_B0_LO+{lo}")] = 1.0
    x[0, 0, resolve(f"ADDR_B0_HI+{hi}")] = 1.0
    return x


def addr_key_band(delta):
    return [delta[0, 0, resolve(f"ADDR_KEY+{k}")].item() for k in range(32)]


def main():
    enum = build_ffn(enumerated_rules())
    comp = build_ffn(computed_rules())
    print(f"D={D}  enumerated units={enum.hidden_dim}  "
          f"computed units={comp.hidden_dim}")

    max_abs_diff = 0.0
    worst = None
    mismatches = 0
    for byte_val in range(256):
        x = make_input(byte_val)
        de = enum(x) - x
        dc = comp(x) - x
        be, bc = addr_key_band(de), addr_key_band(dc)
        diff = max(abs(a - b) for a, b in zip(be, bc))
        if diff > 1e-6:
            mismatches += 1
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst = byte_val
    worst_s = f"{worst:02x}" if worst is not None else "--"
    print(f"[FIRING] 256 bytes: mismatches={mismatches} "
          f"max|Δe-Δc|={max_abs_diff:.6g} (worst={worst_s})")

    # Non-firing: gate off, and MARK_MEM block. Both forms must stay silent.
    nf_max = 0.0
    for label, kw in (("gate-off", dict(gate_on=False)),
                      ("mem-block", dict(mem_block=True))):
        x = make_input(0x2A, **kw)
        de = enum(x) - x
        dc = comp(x) - x
        be, bc = addr_key_band(de), addr_key_band(dc)
        me = max(abs(v) for v in be)
        mc = max(abs(v) for v in bc)
        nf_max = max(nf_max, abs(me - mc))
        print(f"[{label}] enum max|Δ|={me:.4g} computed max|Δ|={mc:.4g}")

    ok = mismatches == 0 and max_abs_diff <= 1e-6 and nf_max <= 1e-6
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} "
          f"(firing max diff {max_abs_diff:.3g}, non-firing diff {nf_max:.3g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
